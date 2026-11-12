"""Merge one or more feature zips (from Kaggle) into features/ and rebuild the indexes.

    python -m scripts.merge_features ../features_shanghaitech_all.zip
    python -m scripts.merge_features a.zip b.zip --overwrite

Each zip holds <dataset>/<feature folder>/... as written by src.preprocess.extract.
Files already present are kept unless --overwrite (identical settings give identical
features, so keeping them is safe). Afterwards every feature folder's
train_clips.csv / test_clips.csv is rebuilt from its per-video meta files, so scenes
extracted in different runs appear together.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from src.utils.common import load_config, resolve


def rebuild_indexes(features_root: Path) -> dict:
    """Rewrite <split>_clips.csv in every feature folder. Returns {folder: (n_train, n_test)}."""
    out = {}
    for meta in sorted(Path(features_root).glob("*/*/meta")):
        d = meta.parent
        counts = []
        for split in ("train", "test"):
            files = sorted((meta / split).glob("*.csv"))
            if files:
                df = pd.concat([pd.read_csv(f, dtype={"scene_id": str, "video_id": str})
                                for f in files])
                df.to_csv(d / f"{split}_clips.csv", index=False)
            counts.append(len(files))
        out[f"{d.parent.name}/{d.name}"] = tuple(counts)
    return out


def merge(zips, features_root: Path, overwrite: bool = False) -> dict:
    features_root = Path(features_root)
    features_root.mkdir(parents=True, exist_ok=True)
    added = kept = 0
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.endswith("/"):
                    continue
                if ".." in Path(name).parts or Path(name).is_absolute():
                    raise ValueError(f"unsafe path in zip: {name}")
                target = features_root / name
                if target.exists() and not overwrite and not name.endswith("info.json"):
                    kept += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
                added += 1
    print(f"written {added} files, kept {kept} existing")
    return rebuild_indexes(features_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+")
    ap.add_argument("--config", default="configs/shanghaitech.yaml")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    root = resolve(load_config(a.config)["features_root"])
    for folder, (n_tr, n_te) in merge(a.zips, root, a.overwrite).items():
        print(f"{folder:45s} train videos {n_tr:4d}   test videos {n_te:4d}")


if __name__ == "__main__":
    main()
