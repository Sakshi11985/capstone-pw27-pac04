"""W1.2: turn every video into cached region features, several kinds in ONE pass.

    python -m src.preprocess.extract --config configs/shanghaitech.yaml
    python -m src.preprocess.extract --config configs/shanghaitech.yaml --limit 2   # smoke test
    python -m src.preprocess.extract --config configs/shanghaitech.yaml --inputs rgb --grids 2

Each video is decoded once; every requested OUTPUT = (input, grid) is encoded from it:
  input "rgb"  : the sampled frames themselves (appearance)
  input "diff" : motion images |frame(o) - frame(o -/+ stride)| x gain, one per sampled
                 frame (the inter-frame RGB difference cue of Hu, Du & Wang, MM 2025)
  grid g       : g x g regions per frame (1 = whole frame)
  appearance a : "none" (the video as recorded) or an illumination preset of
                 src/preprocess/appearance.py ("dim", "night", "warm", "bright") applied
                 to the frames BEFORE rgb and motion images are formed (gate G2)
Settings come from the `extract:` section of the config (inputs, grids, appearances,
diff_stride, diff_gain); without it, only (rgb, clip.grid, none) is produced, as before.

    python -m src.preprocess.extract --config configs/shanghaitech.yaml \
        --split train --grids 2 --appearances dim night warm bright      # G2 pass

Output, one folder per output under features_root/<dataset>/:
  <encoder>_g<g>            for rgb   (e.g. clip_vitb16_g2, unchanged name)
  <encoder>_diff_g<g>       for diff  (e.g. clip_vitb16_diff_g2)
  ..._<appearance>          for a re-lit copy (e.g. clip_vitb16_g2_dim, clip_vitb16_diff_g2_dim)
each containing
    train/<video_id>.npy, test/<video_id>.npy   float16 [clips, g*g, D], L2-normalised
    meta/<split>/<video_id>.csv                 one row per clip
    labels/<video_id>.npy                       int8 frame labels (test videos)
    train_clips.csv, test_clips.csv, info.json
Re-running skips outputs that already exist for a video, so an interrupted run
(e.g. a Kaggle session timing out) can simply be started again.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import load_config, resolve, set_seed
from .appearance import NONE, apply_appearance, resolve_spec
from .clips import (clip_bounds, clip_labels, sample_offsets, split_regions,
                    to_model_input)
from .datasets import iter_frames, list_videos, load_frame_labels
from .encoders import build_encoder

CLIP_COLUMNS = ["video_id", "scene_id", "split", "clip_idx", "start_frame", "end_frame",
                "n_frames_video", "label", "anom_frac"]
INPUTS = ("rgb", "diff")


def feature_dir(cfg: dict, grid: int | None = None, encoder: str | None = None,
                input: str = "rgb", appearance: str | None = None) -> Path:
    grid = grid or cfg["clip"]["grid"]
    encoder = encoder or cfg["encoder"]["name"]
    tag = f"{encoder}_g{grid}" if input == "rgb" else f"{encoder}_{input}_g{grid}"
    if appearance and appearance != NONE:
        tag += f"_{appearance}"
    return resolve(cfg["features_root"]) / cfg["dataset"] / tag


def requested_outputs(cfg: dict, inputs=None, grids=None,
                      appearances=None) -> list[tuple[str, int, str]]:
    """Every (input, grid, appearance) to produce."""
    ex = cfg.get("extract") or {}
    inputs = inputs or ex.get("inputs") or ["rgb"]
    grids = grids or ex.get("grids") or [cfg["clip"]["grid"]]
    appearances = appearances or ex.get("appearances") or [NONE]
    bad = [i for i in inputs if i not in INPUTS]
    if bad:
        raise ValueError(f"unknown inputs {bad}; choose from {INPUTS}")
    for a in appearances:
        if a != NONE:
            resolve_spec(a, ex.get("appearance_presets"))       # fail early on a typo
    return [(i, int(g), a) for a in appearances for i in inputs for g in grids]


def _tag(o) -> str:
    i, g, a = o
    return f"{i}_g{g}" + ("" if a == NONE else f"_{a}")


def diff_partner(o: int, stride: int, length: int) -> int:
    """Frame paired with sampled offset o for the motion image (stays inside the clip)."""
    return o - stride if o - stride >= 0 else min(o + stride, length - 1)


def motion_image(a: np.ndarray, b: np.ndarray, gain: float) -> np.ndarray:
    """|a - b| per pixel and channel, amplified and clipped to uint8."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32) * gain
    return np.clip(d, 0, 255).astype(np.uint8)


class _Output:
    """Collects model inputs for one (input, grid) and encodes them in batches."""

    def __init__(self, inp, grid, fpc, encoder, batch_size):
        self.inp, self.grid, self.R, self.fpc = inp, grid, grid * grid, fpc
        self.encoder, self.batch = encoder, batch_size
        self.pending, self.chunks = [], []

    def add_clip(self, images):                 # images: fpc frames (RGB or motion)
        for im in images:
            for reg in split_regions(im, self.grid):
                self.pending.append(to_model_input(reg, mean=self.encoder.mean,
                                                   std=self.encoder.std))
        if len(self.pending) >= self.batch:
            self.flush()

    def flush(self):
        per_clip = self.fpc * self.R
        n = len(self.pending) // per_clip
        if n:
            f = self.encoder.encode(np.stack(self.pending[:n * per_clip]))
            self.chunks.append(f.reshape(n, self.fpc, self.R, -1))
            self.pending = self.pending[n * per_clip:]

    def result(self) -> np.ndarray:
        self.flush()
        x = np.concatenate(self.chunks).mean(axis=1)          # average the sampled frames
        x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
        return x.astype(np.float16)


def encode_video(item, encoder, length: int, frames_per_clip: int, outputs,
                 batch_size: int, diff_stride: int = 2, diff_gain: float = 4.0,
                 presets: dict | None = None):
    """outputs: list of (input, grid, appearance); a legacy (input, grid) pair means
    appearance "none". Returns ({output: float16 [clips, R, D]}, n_frames)."""
    outputs = [tuple(o) if len(o) == 3 else (o[0], o[1], NONE) for o in outputs]
    offsets = sample_offsets(length, frames_per_clip).tolist()
    need_diff = any(i == "diff" for i, _, _ in outputs)
    partners = {o: diff_partner(o, diff_stride, length) for o in offsets}
    needed = set(offsets) | (set(partners.values()) if need_diff else set())
    apps = sorted({a for _, _, a in outputs})
    specs = {a: resolve_spec(a, presets) for a in apps if a != NONE}
    bufs = {o: _Output(o[0], o[1], frames_per_clip, encoder, batch_size) for o in outputs}
    held, n_frames = {}, 0
    for f, frame in enumerate(iter_frames(item, keep=lambda k: k % length in needed)):
        n_frames = f + 1
        pos = f % length
        if pos in needed:
            held[pos] = (f, frame)
        if pos == length - 1:                                   # clip complete
            for a in apps:
                if a == NONE:
                    fr = {p: x for p, (_, x) in held.items()}
                else:
                    fr = {p: apply_appearance(x, specs[a], item.video_id, fi)
                          for p, (fi, x) in held.items()}
                rgb = [fr[o] for o in offsets]
                need_mot = any(i == "diff" and b == a for i, _, b in outputs)
                mot = [motion_image(fr[o], fr[partners[o]], diff_gain) for o in offsets] \
                    if need_mot else None
                for o, b in bufs.items():
                    if o[2] == a:
                        b.add_clip(rgb if o[0] == "rgb" else mot)
            held = {}
    if n_frames < length:
        raise ValueError(f"{item.video_id}: shorter than one clip ({n_frames} frames)")
    return {k: b.result() for k, b in bufs.items()}, n_frames


def _write_index(out: Path, split: str):
    metas = sorted((out / "meta" / split).glob("*.csv"))
    if metas:
        df = pd.concat([pd.read_csv(m, dtype={"scene_id": str, "video_id": str}) for m in metas])
        df.to_csv(out / f"{split}_clips.csv", index=False)


def run(cfg: dict, grid: int | None = None, encoder_name: str | None = None,
        splits=("train", "test"), limit: int | None = None, verbose: bool = True,
        inputs=None, grids=None, appearances=None) -> Path:
    """Returns the folder of the first output (the rgb one if requested)."""
    set_seed(cfg.get("seed", 0))
    length = cfg["clip"]["length"]
    fpc = cfg["clip"]["frames_per_clip"]
    enc_cfg = cfg["encoder"]
    ex = cfg.get("extract") or {}
    encoder_name = encoder_name or enc_cfg["name"]
    outputs = requested_outputs(cfg, inputs, [grid] if grid else grids, appearances)
    dirs = {o: feature_dir(cfg, o[1], encoder_name, o[0], o[2]) for o in outputs}
    root = resolve(cfg["data_root"])
    encoder = None

    for split in splits:
        items = list_videos(cfg["dataset"], root, split, cfg.get("scenes"))
        if limit:
            items = items[:limit]
        for d in dirs.values():
            for sub in (split, f"meta/{split}", "labels"):
                (d / sub).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, item in enumerate(items):
            todo = [o for o in outputs
                    if not ((dirs[o] / split / f"{item.video_id}.npy").exists()
                            and (dirs[o] / "meta" / split / f"{item.video_id}.csv").exists())]
            if not todo:
                continue
            if encoder is None:
                encoder = build_encoder(encoder_name, enc_cfg.get("device", "cuda"),
                                        enc_cfg.get("fp16", True))
            feats, n_frames = encode_video(item, encoder, length, fpc, todo,
                                           enc_cfg.get("batch_size", 128),
                                           ex.get("diff_stride", 2), ex.get("diff_gain", 4.0),
                                           ex.get("appearance_presets"))
            flabels = load_frame_labels(item, n_frames)
            n_clips = next(iter(feats.values())).shape[0]
            lab, frac = clip_labels(flabels, n_clips, length)
            rows = [[item.video_id, item.scene_id, split, c, *clip_bounds(c, length), n_frames,
                     int(lab[c]), float(frac[c])] for c in range(n_clips)]
            for o in todo:
                d = dirs[o]
                if split == "test":
                    np.save(d / "labels" / f"{item.video_id}.npy", flabels)
                np.save(d / split / f"{item.video_id}.npy", feats[o])          # features first,
                pd.DataFrame(rows, columns=CLIP_COLUMNS).to_csv(               # meta marks done
                    d / "meta" / split / f"{item.video_id}.csv", index=False)
            if verbose:
                print(f"[{split}] {i + 1}/{len(items)} {item.video_id}: {n_frames} frames, "
                      f"{n_clips} clips, {int(lab.sum())} anomalous, outputs "
                      f"{[_tag(o) for o in todo]}  ({time.time() - t0:.0f}s)", flush=True)
        for d in dirs.values():
            _write_index(d, split)

    for (inp, g, app), d in dirs.items():
        info = {"dataset": cfg["dataset"], "scenes": cfg.get("scenes"), "encoder": encoder_name,
                "input": inp, "grid": g, "appearance": app,
                "clip_length": length, "frames_per_clip": fpc,
                "sample_offsets": sample_offsets(length, fpc).tolist(),
                "region_resize": "direct 224x224 bicubic, no centre crop",
                "dtype": "float16, L2-normalised after averaging sampled frames"}
        if inp == "diff":
            info.update({"diff_stride": ex.get("diff_stride", 2),
                         "diff_gain": ex.get("diff_gain", 4.0),
                         "diff_definition": "|frame(o) - frame(o-stride)|, or o+stride at the "
                                            "clip start, times gain, clipped to 0-255"})
        if app != NONE:
            info["appearance_spec"] = resolve_spec(app, ex.get("appearance_presets"))
            info["appearance_definition"] = ("y = clip(gain * x**gamma * tint + N(0, noise)/255), "
                                             "x = frame/255; applied before rgb and motion images")
        (d / "info.json").write_text(json.dumps(info, indent=2, default=list))
    return dirs[outputs[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", nargs="+", default=None, help="override extract.inputs")
    ap.add_argument("--grids", nargs="+", type=int, default=None, help="override extract.grids")
    ap.add_argument("--appearances", nargs="+", default=None,
                    help="override extract.appearances, e.g. dim night warm bright (none = original)")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="only the first N videos per split")
    a = ap.parse_args()
    cfg = load_config(a.config)
    splits = ("train", "test") if a.split == "both" else (a.split,)
    run(cfg, None, a.encoder, splits, a.limit, inputs=a.inputs, grids=a.grids,
        appearances=a.appearances)
    print("done ->", ", ".join(str(feature_dir(cfg, g, a.encoder, i, ap_))
                               for i, g, ap_ in requested_outputs(cfg, a.inputs, a.grids,
                                                                  a.appearances)))


if __name__ == "__main__":
    main()
