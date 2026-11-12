"""Controlled appearance (illumination) changes: the legitimate change for gate G2.

A deployed camera sees lighting change (dusk, lights switched off, artificial light,
strong sunlight). These presets re-render the SAME video frames under a different
illumination before encoding, so the memory meets a change that is truly normal (same
scene, same activities) but visually new. Motivated by the cross-illumination VAD
benchmark of Zhu et al. (ACM MM 2023).

Per pixel and channel, with x = frame / 255 in [0, 1]:

    y = clip( round(255 * gain * x ** gamma * tint) + noise * z , 0, 255 )

  gamma > 1 darkens mid-tones (low light), gain scales overall brightness, tint is a
  per-channel (R, G, B) colour cast; z is a standard-normal sensor-noise field and
  `noise` its standard deviation in grey levels.

Implementation (speed: ~5 ms per frame instead of ~20): the noise-free part is an 8-bit
lookup table (cv2.LUT); z comes from a fixed bank of NOISE_BANK fields drawn once from the
preset seed, and frame t of video v uses field (crc32(v) + t) mod NOISE_BANK. Consecutive
frames therefore always get different fields (the motion images keep their noise), and a
re-run (e.g. after a Kaggle timeout) reproduces the features bit for bit. Motion images
are computed from the re-lit frames, as a real camera would see them.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np

NONE = "none"
NOISE_BANK = 32           # > clip length (16), so all frames of a clip have distinct noise

PRESETS: dict[str, dict] = {
    # dusk / lights partly off: darker, slight sensor noise
    "dim":    {"gamma": 1.8, "gain": 0.60, "tint": (1.0, 1.0, 1.0), "noise": 3.0},
    # night with weak lighting: much darker, bluish, noisy
    "night":  {"gamma": 2.4, "gain": 0.35, "tint": (0.85, 0.95, 1.15), "noise": 8.0},
    # warm artificial (tungsten) lighting: colour cast, brightness almost unchanged
    "warm":   {"gamma": 1.0, "gain": 0.95, "tint": (1.20, 1.00, 0.75), "noise": 0.0},
    # strong sunlight / over-exposure: brighter, washed-out mid-tones
    "bright": {"gamma": 0.7, "gain": 1.25, "tint": (1.0, 1.0, 1.0), "noise": 0.0},
}


def resolve_spec(name: str, custom: dict | None = None) -> dict:
    """Preset by name; `custom` (e.g. from the config) overrides or defines presets."""
    table = {**PRESETS, **(custom or {})}
    if name not in table:
        raise ValueError(f"unknown appearance {name!r}; choose from {sorted(table)}")
    spec = {"gamma": 1.0, "gain": 1.0, "tint": (1.0, 1.0, 1.0), "noise": 0.0, "seed": 0}
    spec.update(table[name])
    return spec


_LUTS: dict = {}
_BANKS: dict = {}


def _lut(spec: dict) -> np.ndarray:
    """uint8 [256, 1, 3] lookup table of the noise-free re-lit value per channel (RGB)."""
    key = (spec["gamma"], spec["gain"], tuple(spec["tint"]))
    if key not in _LUTS:
        v = np.arange(256, dtype=np.float64) / 255.0
        y = spec["gain"] * np.power(v, spec["gamma"])[:, None] * np.asarray(spec["tint"])[None]
        _LUTS[key] = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8).reshape(256, 1, 3)
    return _LUTS[key]


def _noise_bank(seed: int, shape) -> np.ndarray:
    key = (int(seed), tuple(shape))
    if key not in _BANKS:
        rng = np.random.default_rng([int(seed), 7919])
        _BANKS[key] = rng.standard_normal((NOISE_BANK, *shape), dtype=np.float32)
    return _BANKS[key]


def noise_index(video_id: str, frame_idx: int) -> int:
    return (zlib.crc32(str(video_id).encode()) + int(frame_idx)) % NOISE_BANK


def apply_appearance(frame: np.ndarray, spec: dict, video_id: str = "", frame_idx: int = 0) -> np.ndarray:
    """frame: uint8 [H, W, 3] RGB. Returns the re-lit uint8 frame (input unchanged)."""
    y = cv2.LUT(np.ascontiguousarray(frame), _lut(spec))
    if spec["noise"] <= 0:
        return y
    z = _noise_bank(spec.get("seed", 0), frame.shape)[noise_index(video_id, frame_idx)]
    x = y.astype(np.float32)
    x += z * np.float32(spec["noise"])
    return (np.clip(x, 0.0, 255.0) + 0.5).astype(np.uint8)
