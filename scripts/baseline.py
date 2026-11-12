"""W1.3 / gate G1: frozen-memory baseline on cached features.

    python -m scripts.baseline --config configs/shanghaitech.yaml

Memory per scene and region = all training clip features of that scene.
Prints and saves a table of micro / macro frame AUC for every (k, sigma), and
appends one row per setting to results/runs.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import frame_auc
from src.eval.scoring import clip_scores, labels_dir, load_features, load_frame_labels
from src.utils.common import REPO_ROOT, load_config, log_run, set_seed, with_channels


def run(cfg: dict, grid: int | None = None, log: bool = True,
        out_dir=None) -> pd.DataFrame:
    set_seed(cfg.get("seed", 0))
    scenes = cfg.get("scenes")
    length = cfg["clip"]["length"]
    how = (cfg.get("score") or {}).get("aggregate", "mean")
    channels = (cfg.get("features") or {}).get("channels") or ["rgb"]
    tr_df, tr_feats = load_features(cfg, "train", scenes, grid)
    te_df, te_feats = load_features(cfg, "test", scenes, grid)
    fdir = labels_dir(cfg, grid)

    # leakage guard: no test video may contribute to a memory
    overlap = set(tr_feats) & set(te_feats)
    assert not overlap, f"train/test video overlap: {sorted(overlap)[:5]}"

    labels = {vid: load_frame_labels(fdir, vid) for vid in te_feats}
    rows, scene_rows = [], []
    for k in cfg["baseline"]["k"]:
        scores = {}
        for scene in sorted(te_df.scene_id.unique()):
            tr_vids = tr_df[tr_df.scene_id == scene].video_id.unique()
            if len(tr_vids) == 0:
                raise ValueError(f"scene {scene} has test but no training videos")
            mem_all = np.concatenate([tr_feats[v] for v in tr_vids])   # [m, R, D]
            memories = [mem_all[:, r] for r in range(mem_all.shape[1])]
            for vid in te_df[te_df.scene_id == scene].video_id.unique():
                scores[vid] = clip_scores(te_feats[vid], memories, k, how)
        for sigma in cfg["baseline"]["sigma"]:
            res = frame_auc({v: (scores[v], labels[v]) for v in scores}, length, sigma)
            for scene in sorted(te_df.scene_id.unique()):          # per-scene breakdown
                vs = [v for v in scores if v in set(te_df[te_df.scene_id == scene].video_id)]
                rs = frame_auc({v: (scores[v], labels[v]) for v in vs}, length, sigma)
                scene_rows.append({"scene": scene, "k": k, "sigma": sigma, "videos": len(vs),
                                   "micro_auc": rs["micro_auc"], "macro_auc": rs["macro_auc"]})
            row = {"channels": "+".join(channels), "grid": grid or cfg["clip"]["grid"],
                   "aggregate": how, "k": k, "sigma": sigma, **res}
            rows.append(row)
            if log:
                log_run("scripts/baseline.py", cfg,
                        {"grid": row["grid"], "k": k, "sigma": sigma, "aggregate": how,
                         "channels": row["channels"], "encoder": cfg["encoder"]["name"]},
                        {"micro_auc": round(res["micro_auc"], 4),
                         "macro_auc": round(res["macro_auc"], 4)})
    table = pd.DataFrame(rows)
    tag = "" if channels == ["rgb"] else "_" + "+".join(channels)
    out = (Path(out_dir) if out_dir else REPO_ROOT / "results") / \
        f"baseline_{cfg['dataset']}{tag}_g{grid or cfg['clip']['grid']}_{how}.csv"
    table.to_csv(out, index=False)
    per_scene = pd.DataFrame(scene_rows)
    per_scene.to_csv(out.with_name(out.stem + "_per_scene.csv"), index=False)
    n_tr, n_te = len(tr_df), len(te_df)
    print(f"train clips {n_tr} ({tr_df.video_id.nunique()} videos) | test clips {n_te} "
          f"({te_df.video_id.nunique()} videos, {int(te_df.label.sum())} anomalous clips)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--channels", nargs="+", default=None,
                    help="override features.channels, e.g. rgb diff")
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    run(with_channels(load_config(a.config), a.channels), a.grid, log=not a.no_log)


if __name__ == "__main__":
    main()
