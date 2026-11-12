"""Dataset adapters: list videos, read frames (RGB) and frame-level labels.

ShanghaiTech Campus expected layout (the official release):

    data_root/
      training/videos/01_001.avi ...           (or training/frames/01_001/000.jpg ...)
      testing/frames/01_0014/000.jpg ...       (or testing/videos/01_0014.avi)
      testing/test_frame_mask/01_0014.npy      frame labels, 1 = anomalous

Each video becomes a VideoItem. Frames are yielded one at a time so a whole
video is never held in memory.
"""
from __future__ import annotations

import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
VID_EXT = {".avi", ".mp4", ".mov", ".mkv"}


@dataclass
class VideoItem:
    video_id: str
    scene_id: str
    split: str                  # "train" | "test"
    source: Path                # a video file or a folder of frames
    label_path: Optional[Path] = None


def _natural_key(p: Path):
    """Sort 2.jpg before 10.jpg (plain sorting would not)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def _read_rgb(p: Path) -> np.ndarray:
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Cannot read frame {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def iter_frames(item: VideoItem, keep: Optional[Callable[[int], bool]] = None,
                workers: int = 8) -> Iterator[Optional[np.ndarray]]:
    """Yield every frame of the video in temporal order, as RGB uint8 [H, W, 3].

    If `keep(frame_index)` is given, frames it rejects are yielded as None without
    being decoded (the frame still counts). For frame folders the kept frames are
    read by a thread pool, which is much faster on network storage such as Kaggle's.
    """
    keep = keep or (lambda f: True)
    src = item.source
    if src.is_dir():
        files = sorted((p for p in src.iterdir() if p.suffix.lower() in IMG_EXT), key=_natural_key)
        wanted = [f for f in range(len(files)) if keep(f)]
        ahead = 4 * workers                       # bounded read-ahead keeps memory small
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = deque(pool.submit(_read_rgb, files[f]) for f in wanted[:ahead])
            nxt_submit = ahead
            w_iter = iter(wanted)
            w = next(w_iter, None)
            for f in range(len(files)):
                if f == w:
                    frame = futures.popleft().result()
                    if nxt_submit < len(wanted):
                        futures.append(pool.submit(_read_rgb, files[wanted[nxt_submit]]))
                        nxt_submit += 1
                    yield frame
                    w = next(w_iter, None)
                else:
                    yield None
    else:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise IOError(f"Cannot open video {src}")
        try:
            f = 0
            while True:
                if keep(f):
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:
                    if not cap.grab():            # advance without decoding to an image
                        break
                    yield None
                f += 1
        finally:
            cap.release()


def load_frame_labels(item: VideoItem, n_frames: int) -> np.ndarray:
    """Frame-level labels (int8, 1 = anomalous). Training videos are all normal."""
    if item.split == "train" or item.label_path is None:
        return np.zeros(n_frames, dtype=np.int8)
    lab = np.load(item.label_path).astype(np.int8).ravel()
    if len(lab) != n_frames:
        raise ValueError(f"{item.video_id}: {len(lab)} labels but {n_frames} frames decoded")
    return lab


def _collect(folder: Path, split: str, scenes, label_dir: Optional[Path]) -> list[VideoItem]:
    items = []
    if not folder.exists():
        return items
    for p in sorted(folder.iterdir()):
        is_vid = p.is_file() and p.suffix.lower() in VID_EXT
        if not (p.is_dir() or is_vid):
            continue
        vid = p.stem if is_vid else p.name
        scene = vid.split("_")[0]
        if scenes and scene not in scenes:
            continue
        lab = None
        if label_dir is not None:
            lab = label_dir / f"{vid}.npy"
            if not lab.exists():
                raise FileNotFoundError(f"Missing label file {lab}")
        items.append(VideoItem(vid, scene, split, p, lab))
    return items


def list_shanghaitech(root: Path, split: str, scenes=None) -> list[VideoItem]:
    root = Path(root)
    scenes = set(scenes) if scenes else None
    if split == "train":
        base, label_dir = root / "training", None
    elif split == "test":
        base, label_dir = root / "testing", root / "testing" / "test_frame_mask"
    else:
        raise ValueError(split)
    # prefer extracted frames when both exist (exact frame count = label count)
    items = _collect(base / "frames", split, scenes, label_dir)
    if not items:
        items = _collect(base / "videos", split, scenes, label_dir)
    if not items:
        raise FileNotFoundError(
            f"No {split} videos found under {base}/frames or {base}/videos. "
            "Check data_root in the config.")
    return items


def list_videos(dataset: str, root: Path, split: str, scenes=None) -> list[VideoItem]:
    if dataset == "shanghaitech":
        return list_shanghaitech(root, split, scenes)
    # NWPU Campus and Street Scene adapters are added when the data arrives (W1.1).
    raise NotImplementedError(f"No adapter for dataset '{dataset}' yet")
