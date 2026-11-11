"""Compare the new feature cache with the numbers from the earlier pre-processing
(work plan, Phase 1 "Already done", scenes 01-03):
    104 training videos, 47 test videos, 4,964 training clips,
    1,056 test clips, 468 test clips with anomalous frames.

    python -m scripts.check_counts --config configs/shanghaitech.yaml
"""
import argparse

import pandas as pd

from src.preprocess.extract import feature_dir
from src.utils.common import load_config

EXPECTED_01_03 = {"train_videos": 104, "test_videos": 47, "train_clips": 4964,
                  "test_clips": 1056, "anomalous_test_clips": 468}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    fdir = feature_dir(cfg)
    tr = pd.read_csv(fdir / "train_clips.csv", dtype={"scene_id": str})
    te = pd.read_csv(fdir / "test_clips.csv", dtype={"scene_id": str})
    got = {"train_videos": tr.video_id.nunique(), "test_videos": te.video_id.nunique(),
           "train_clips": len(tr), "test_clips": len(te),
           "anomalous_test_clips": int(te.label.sum())}
    print(f"{'quantity':<22}{'now':>8}{'earlier':>9}")
    for k, v in got.items():
        exp = EXPECTED_01_03[k] if cfg.get("scenes") == ["01", "02", "03"] else "-"
        flag = "" if exp == "-" or exp == v else "   <-- differs"
        print(f"{k:<22}{v:>8}{exp:>9}{flag}")
    print("\nper scene:")
    print(pd.concat([tr.groupby("scene_id").size().rename("train_clips"),
                     te.groupby("scene_id").size().rename("test_clips"),
                     te.groupby("scene_id").label.sum().rename("anomalous")], axis=1).to_string())


if __name__ == "__main__":
    main()
