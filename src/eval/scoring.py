"""Nearest-neighbour anomaly scoring against a region memory, plus feature loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def knn_distance(queries: np.ndarray, memory: np.ndarray, k: int = 1,
                 exclude_self: bool = False, chunk: int = 4096) -> np.ndarray:
    """Mean cosine distance (1 - cos) to the k nearest memory entries.

    queries [n, D] and memory [m, D] must be L2-normalised. With exclude_self,
    the single closest entry is dropped (used when the queries are themselves in
    the memory, e.g. calibrating a gate on training clips).
    """
    q = np.asarray(queries, dtype=np.float32)
    m = np.asarray(memory, dtype=np.float32)
    kk = k + int(exclude_self)
    if m.shape[0] < kk:
        raise ValueError(f"memory has {m.shape[0]} entries, need {kk}")
    out = np.empty(q.shape[0], dtype=np.float32)
    for i in range(0, q.shape[0], chunk):
        sims = q[i:i + chunk] @ m.T
        top = -np.partition(-sims, kk - 1, axis=1)[:, :kk]
        top = -np.sort(-top, axis=1)               # descending similarity
        if exclude_self:
            top = top[:, 1:]
        out[i:i + chunk] = 1.0 - top.mean(axis=1)
    return out


def region_scores(clip_feats: np.ndarray, memories: list[np.ndarray], k: int = 1) -> np.ndarray:
    """clip_feats [n, R, D], memories: R arrays [m_r, D] -> per-region distances [n, R]."""
    R = clip_feats.shape[1]
    if len(memories) != R:
        raise ValueError("one memory per region expected")
    return np.stack([knn_distance(clip_feats[:, r], memories[r], k) for r in range(R)], axis=1)


AGGREGATES = ("mean", "max")


def aggregate(reg_scores: np.ndarray, how: str = "mean") -> np.ndarray:
    """Combine per-region scores [..., R] into one clip score.

    mean (default): on ShanghaiTech 01-03 frame AUC 0.760 vs 0.752 for max; with max the
    noisiest region (top-left) decides 52% of normal clips. Memories and admission stay
    per region either way; only the alarm score changes.
    """
    if how == "mean":
        return reg_scores.mean(axis=-1)
    if how == "max":
        return reg_scores.max(axis=-1)
    raise ValueError(f"unknown aggregate {how!r}; choose from {AGGREGATES}")


def clip_scores(clip_feats, memories, k=1, how: str = "mean") -> np.ndarray:
    """Clip score = aggregate over regions of the k-NN distances."""
    return aggregate(region_scores(clip_feats, memories, k), how)


# ---------------------------------------------------------------- loading
def load_split(feat_dir: Path, split: str, scenes=None):
    """Returns (clips DataFrame, {video_id: float32 [clips, R, D]})."""
    feat_dir = Path(feat_dir)
    df = pd.read_csv(feat_dir / f"{split}_clips.csv", dtype={"scene_id": str, "video_id": str})
    if scenes:
        df = df[df.scene_id.isin(scenes)]
    feats = {}
    for vid in df.video_id.unique():
        x = np.load(feat_dir / split / f"{vid}.npy").astype(np.float32)
        n = int((df.video_id == vid).sum())
        if x.shape[0] != n:
            raise AssertionError(f"{vid}: {x.shape[0]} feature rows but {n} clip rows")
        feats[vid] = x
    return df.reset_index(drop=True), feats


def load_frame_labels(feat_dir: Path, video_id: str) -> np.ndarray:
    return np.load(Path(feat_dir) / "labels" / f"{video_id}.npy")


def fuse_channels(parts: list) -> np.ndarray:
    """Concatenate L2-normalised per-region features of several channels (e.g. rgb and
    diff) along the feature axis and rescale to unit norm. The cosine similarity of two
    fused vectors is then the MEAN of the per-channel cosine similarities."""
    if len(parts) == 1:
        return parts[0]
    x = np.concatenate(parts, axis=-1)
    return x / np.sqrt(len(parts))


def load_features(cfg: dict, split: str, scenes=None, grid: int | None = None,
                  appearance: str | None = None):
    """Load the channels listed in cfg['features']['channels'] (default ['rgb']) for one
    grid (and appearance: None = as recorded, or e.g. "dim") and fuse them.
    Returns (clips DataFrame, {video_id: float32 [clips, R, D_total]})."""
    from src.preprocess.extract import feature_dir
    channels = (cfg.get("features") or {}).get("channels") or ["rgb"]
    loaded = [load_split(feature_dir(cfg, grid, None, ch, appearance), split, scenes)
              for ch in channels]
    df0, f0 = loaded[0]
    for ch, (df, f) in zip(channels[1:], loaded[1:]):
        key = ["video_id", "clip_idx"]
        if not df0[key].reset_index(drop=True).equals(df[key].reset_index(drop=True)):
            raise AssertionError(f"channel {ch}: clips differ from {channels[0]}")
    feats = {v: fuse_channels([f[v] for _, f in loaded]) for v in f0}
    return df0, feats


def labels_dir(cfg: dict, grid: int | None = None) -> Path:
    """Frame labels live in the folder of the first channel."""
    from src.preprocess.extract import feature_dir
    channels = (cfg.get("features") or {}).get("channels") or ["rgb"]
    return feature_dir(cfg, grid, None, channels[0])