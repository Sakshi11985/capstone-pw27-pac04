"""Frame-level evaluation: clip -> frame scores, smoothing, normalisation, AUC.

Protocol (docs/protocol.md):
  clip scores --(Gaussian smoothing over clips, sigma)--> expand to frames
  (trailing frames take the last clip's score) --> min-max per video
  --> micro AUC (all test frames pooled; ShanghaiTech standard) and
      macro AUC (mean over videos containing both classes).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties count one half)."""
    y = np.asarray(labels).astype(bool).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.copy() if sigma <= 0 else gaussian_filter1d(x, sigma, mode="nearest")


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x, dtype=np.float64) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def clip_to_frame_scores(clip_scores: np.ndarray, n_frames: int, length: int) -> np.ndarray:
    n_clips = len(clip_scores)
    idx = np.minimum(np.arange(n_frames) // length, n_clips - 1)
    return np.asarray(clip_scores)[idx]


def video_frame_scores(clip_scores, n_frames, length, sigma, normalise=True):
    fs = clip_to_frame_scores(smooth(clip_scores, sigma), n_frames, length)
    return minmax(fs) if normalise else fs


def frame_auc(videos: dict, length: int, sigma: float, normalise: bool = True) -> dict:
    """videos: {video_id: (clip_scores [n_clips], frame_labels [n_frames])}."""
    all_s, all_y, per_video = [], [], {}
    for vid, (cs, fl) in videos.items():
        fs = video_frame_scores(cs, len(fl), length, sigma, normalise)
        if len(fs) != len(fl):
            raise AssertionError(f"{vid}: frame count mismatch")
        all_s.append(fs)
        all_y.append(fl)
        a = roc_auc(fl, fs)
        if not np.isnan(a):
            per_video[vid] = a
    micro = roc_auc(np.concatenate(all_y), np.concatenate(all_s))
    macro = float(np.mean(list(per_video.values()))) if per_video else float("nan")
    return {"micro_auc": micro, "macro_auc": macro, "n_videos": len(videos),
            "n_videos_macro": len(per_video)}
