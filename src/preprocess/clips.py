"""Clip indexing, frame sampling and region splitting (pure numpy, no GPU).

Conventions (fixed in docs/protocol.md):
* A clip is `length` consecutive frames, non-overlapping, starting at frame 0.
* Trailing frames that do not fill a whole clip are not encoded; at evaluation
  time they take the score of the last full clip.
* `frames_per_clip` frames are sampled evenly inside each clip (16 -> 0,5,10,15).
* Each sampled frame is split into a grid x grid set of regions, row-major
  (2x2: 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right).
* Each region is resized to 224x224 directly (no centre crop), so no part of
  the region is lost; the aspect-ratio squash is the same for all clips.
"""
from __future__ import annotations

import numpy as np
import cv2

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def sample_offsets(length: int, frames_per_clip: int) -> np.ndarray:
    """Evenly spaced frame offsets inside a clip, e.g. (16, 4) -> [0, 5, 10, 15]."""
    if not 1 <= frames_per_clip <= length:
        raise ValueError("need 1 <= frames_per_clip <= length")
    return np.round(np.linspace(0, length - 1, frames_per_clip)).astype(int)


def n_full_clips(n_frames: int, length: int) -> int:
    return n_frames // length


def clip_bounds(clip_idx: int, length: int) -> tuple[int, int]:
    """Inclusive start and end frame of a clip."""
    s = clip_idx * length
    return s, s + length - 1


def frame_to_clip(n_frames: int, n_clips: int, length: int) -> np.ndarray:
    """Clip index for every frame; trailing frames map to the last full clip."""
    if n_clips == 0:
        raise ValueError("video shorter than one clip")
    return np.minimum(np.arange(n_frames) // length, n_clips - 1)


def clip_labels(frame_labels: np.ndarray, n_clips: int, length: int):
    """Per clip: label (1 if any frame anomalous) and anomalous-frame fraction."""
    fl = np.asarray(frame_labels[: n_clips * length]).reshape(n_clips, length)
    return (fl.max(axis=1) > 0).astype(np.int8), fl.mean(axis=1).astype(np.float32)


def split_regions(frame: np.ndarray, grid: int) -> list[np.ndarray]:
    """Split an [H, W, C] frame into grid*grid regions, row-major. Edges absorb the remainder."""
    h, w = frame.shape[:2]
    ys = np.linspace(0, h, grid + 1).round().astype(int)
    xs = np.linspace(0, w, grid + 1).round().astype(int)
    return [frame[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] for i in range(grid) for j in range(grid)]


def to_model_input(region: np.ndarray, size: int = 224,
                   mean: np.ndarray = CLIP_MEAN, std: np.ndarray = CLIP_STD) -> np.ndarray:
    """RGB uint8 region -> normalised float32 [3, size, size]."""
    x = cv2.resize(region, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1))
