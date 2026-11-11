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


def clip_scores(clip_feats, memories, k=1) -> np.ndarray:
    """Clip score = maximum over regions (an anomaly in any region flags the clip)."""
    return region_scores(clip_feats, memories, k).max(axis=1)


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
