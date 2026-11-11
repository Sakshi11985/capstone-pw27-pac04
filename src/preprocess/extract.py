"""W1.2: turn every video into cached region features.

    python -m src.preprocess.extract --config configs/shanghaitech.yaml
    python -m src.preprocess.extract --config configs/shanghaitech.yaml --grid 1   # ablation caches
    python -m src.preprocess.extract --config configs/shanghaitech.yaml --limit 2  # quick smoke test

Output (under features_root/<dataset>/<encoder>_g<grid>/):
    train/<video_id>.npy     float16 [clips, grid*grid, D]   (L2-normalised)
    test/<video_id>.npy
    meta/<split>/<video_id>.csv   one row per clip
    labels/<video_id>.npy    int8 frame labels (test videos)
    train_clips.csv, test_clips.csv   all clip rows concatenated
    info.json                settings used, for reproducibility

Re-running skips videos that are already done, so an interrupted run
(e.g. a Colab session timing out) can simply be started again.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import load_config, resolve, set_seed
from .clips import (clip_bounds, clip_labels, sample_offsets, split_regions,
                    to_model_input)
from .datasets import iter_frames, list_videos, load_frame_labels
from .encoders import build_encoder

CLIP_COLUMNS = ["video_id", "scene_id", "split", "clip_idx", "start_frame", "end_frame",
                "n_frames_video", "label", "anom_frac"]


def feature_dir(cfg: dict, grid: int | None = None, encoder: str | None = None) -> Path:
    grid = grid or cfg["clip"]["grid"]
    encoder = encoder or cfg["encoder"]["name"]
    return resolve(cfg["features_root"]) / cfg["dataset"] / f"{encoder}_g{grid}"


def encode_video(item, encoder, length: int, frames_per_clip: int, grid: int,
                 batch_size: int):
    """Returns (features float16 [clips, R, D], n_frames)."""
    offsets = set(sample_offsets(length, frames_per_clip).tolist())
    R = grid * grid
    pending: list[np.ndarray] = []        # model inputs, F*R per finished clip
    current: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    n_frames = 0

    def flush(force=False):
        nonlocal pending
        per_clip = frames_per_clip * R
        n_ready = len(pending) // per_clip
        if n_ready == 0 or (not force and len(pending) < batch_size):
            return
        take = n_ready * per_clip
        feats = encoder.encode(np.stack(pending[:take]))
        chunks.append(feats.reshape(n_ready, frames_per_clip, R, -1))
        pending = pending[take:]

    for f, frame in enumerate(iter_frames(item)):
        n_frames = f + 1
        pos = f % length
        if pos in offsets:
            for reg in split_regions(frame, grid):
                current.append(to_model_input(reg, mean=encoder.mean, std=encoder.std))
        if pos == length - 1:              # clip complete
            pending.extend(current)
            current = []
            flush()
    flush(force=True)
    if not chunks:
        raise ValueError(f"{item.video_id}: shorter than one clip ({n_frames} frames)")
    x = np.concatenate(chunks).mean(axis=1)            # average the sampled frames
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    return x.astype(np.float16), n_frames


def run(cfg: dict, grid: int | None = None, encoder_name: str | None = None,
        splits=("train", "test"), limit: int | None = None, verbose: bool = True) -> Path:
    set_seed(cfg.get("seed", 0))
    length = cfg["clip"]["length"]
    fpc = cfg["clip"]["frames_per_clip"]
    grid = grid or cfg["clip"]["grid"]
    enc_cfg = cfg["encoder"]
    encoder_name = encoder_name or enc_cfg["name"]
    out = feature_dir(cfg, grid, encoder_name)
    root = resolve(cfg["data_root"])
    encoder = None

    for split in splits:
        items = list_videos(cfg["dataset"], root, split, cfg.get("scenes"))
        if limit:
            items = items[:limit]
        (out / split).mkdir(parents=True, exist_ok=True)
        (out / "meta" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, item in enumerate(items):
            fpath = out / split / f"{item.video_id}.npy"
            mpath = out / "meta" / split / f"{item.video_id}.csv"
            if fpath.exists() and mpath.exists():
                continue
            if encoder is None:
                encoder = build_encoder(encoder_name, enc_cfg.get("device", "cuda"),
                                        enc_cfg.get("fp16", True))
            feats, n_frames = encode_video(item, encoder, length, fpc, grid,
                                           enc_cfg.get("batch_size", 128))
            n_clips = feats.shape[0]
            flabels = load_frame_labels(item, n_frames)
            lab, frac = clip_labels(flabels, n_clips, length)
            rows = []
            for c in range(n_clips):
                s, e = clip_bounds(c, length)
                rows.append([item.video_id, item.scene_id, split, c, s, e, n_frames,
                             int(lab[c]), float(frac[c])])
            if split == "test":
                np.save(out / "labels" / f"{item.video_id}.npy", flabels)
            np.save(fpath, feats)                                   # features first,
            pd.DataFrame(rows, columns=CLIP_COLUMNS).to_csv(mpath, index=False)  # meta marks done
            if verbose:
                el = time.time() - t0
                print(f"[{split}] {i + 1}/{len(items)} {item.video_id}: {n_frames} frames, "
                      f"{n_clips} clips, {int(lab.sum())} anomalous  ({el:.0f}s)", flush=True)
        metas = sorted((out / "meta" / split).glob("*.csv"))
        if metas:
            df = pd.concat([pd.read_csv(m, dtype={"scene_id": str}) for m in metas])
            df.to_csv(out / f"{split}_clips.csv", index=False)

    info = {"dataset": cfg["dataset"], "scenes": cfg.get("scenes"), "encoder": encoder_name,
            "grid": grid, "clip_length": length, "frames_per_clip": fpc,
            "sample_offsets": sample_offsets(length, fpc).tolist(),
            "region_resize": "direct 224x224 bicubic, no centre crop",
            "dtype": "float16, L2-normalised after averaging sampled frames"}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="only the first N videos per split")
    a = ap.parse_args()
    cfg = load_config(a.config)
    splits = ("train", "test") if a.split == "both" else (a.split,)
    out = run(cfg, a.grid, a.encoder, splits, a.limit)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
