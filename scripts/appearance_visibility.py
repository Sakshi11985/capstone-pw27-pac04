"""Gate G2: how visible is each lighting preset to a memory of original clips?

    python -m scripts.appearance_visibility --config configs/shanghaitech.yaml
    python -m scripts.appearance_visibility --config configs/shanghaitech.yaml --channels diff

Applies the pre-registered rule of docs/protocol.md ("Appearance change for G2") using
visibility ONLY, before any re-lit stream is run: `dim` is the primary preset unless its
median visibility over scenes is below streams.min_novelty, then `night`.

Writes results/appearance_visibility_<dataset>[_<channels>].csv (scene x preset) and one
row per preset in results/runs.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.eval.scoring import load_features
from src.preprocess.appearance import PRESETS
from src.preprocess.extract import feature_dir
from src.streams.generator import appearance_novelty
from src.utils.common import REPO_ROOT, load_config, log_run, set_seed, with_channels


def run(cfg: dict, presets=None, log: bool = True, out_root=None) -> pd.DataFrame:
    set_seed(cfg.get("seed", 0))
    out_root = out_root or REPO_ROOT / "results"
    how = (cfg.get("score") or {}).get("aggregate", "mean")
    chans = (cfg.get("features") or {}).get("channels") or ["rgb"]
    min_nov = (cfg.get("streams") or {}).get("min_novelty", 0.1)
    presets = presets or [p for p in PRESETS
                          if all(feature_dir(cfg, None, None, c, p).exists() for c in chans)]
    if not presets:
        raise SystemExit("no re-lit features found: run notebooks/kaggle_relight.ipynb first")
    tr, trf = load_features(cfg, "train")
    scenes = cfg.get("scenes") or sorted(tr.scene_id.unique())
    rows = []
    for p in presets:
        ap, apf = load_features(cfg, "train", appearance=p)
        if not tr[["video_id", "clip_idx"]].equals(ap[["video_id", "clip_idx"]]):
            raise AssertionError(f"{p}: re-lit clips differ from the original training clips")
        for s in scenes:
            g = tr[tr.scene_id == s]
            X = np.stack([trf[v][c] for v, c in zip(g.video_id, g.clip_idx)])
            Xa = np.stack([apf[v][c] for v, c in zip(g.video_id, g.clip_idx)])
            rows.append({"scene": s, "preset": p,
                         "visibility": appearance_novelty(X, Xa, g.video_id.to_numpy(), how=how)})
    df = pd.DataFrame(rows)
    table = df.pivot(index="scene", columns="preset", values="visibility")
    med = table.median()
    primary = "dim" if med.get("dim", 0) >= min_nov else ("night" if "night" in med else None)
    tag = "" if chans == ["rgb"] else "_" + "+".join(chans)
    table.to_csv(out_root / f"appearance_visibility_{cfg['dataset']}{tag}.csv")
    print(f"share of re-lit clips flagged (1% false alarms on original lighting), channels {chans}")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print("median:", med.round(3).to_dict())
    print(f"visible (>= {min_nov}) in scenes:",
          {p: int((table[p] >= min_nov).sum()) for p in table.columns})
    print("pre-registered primary preset ->", primary)
    if log:
        for p in table.columns:
            log_run("scripts/appearance_visibility.py", cfg,
                    {"preset": p, "channels": chans, "aggregate": how},
                    {"median_visibility": round(float(med[p]), 4),
                     "scenes_visible": int((table[p] >= min_nov).sum()),
                     "primary": primary})
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--channels", nargs="+", default=None)
    ap.add_argument("--presets", nargs="+", default=None)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    run(with_channels(load_config(a.config), a.channels), a.presets, log=not a.no_log)


if __name__ == "__main__":
    main()
