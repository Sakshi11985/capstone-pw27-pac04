"""W2.2: the four update rules (the self-updating detectors we later attack).

Every rule answers one question per clip: which regions of this clip may be
written into memory? It sees only the current clip's scores (computed against
the memory *before* the clip) and its own record of past clips, never the future.

  FrozenRule      never updates (reference)
  ScoreGateRule   MemStream-style: admit region r if its score < beta_r,
                  beta_r = a quantile of training-normal scores
  SceneMemoryRule after Hu, Du & Wang (2025), made causal: admit the clip if its
                  normalised score < 0.2 and the previous two clips were < 0.3
  TextGateRule    SAVEN: a visual rule AND caption agreement (hook; filled in W4.1)
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import numpy as np

from src.memory.memory import SceneMemory


def training_scores(memory: SceneMemory, feats: np.ndarray, uids: np.ndarray,
                    groups: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-region 1-NN distance of training clips to the memory. Returns [n, R].

    Each clip is scored as if it were new: the slot it wrote is ignored, and if
    `groups` (e.g. the video id of each clip, aligned with `uids`) is given, every
    slot written by a clip of the same group is ignored too. Without groups, the
    neighbouring clips of the same video remain in memory and are near-duplicates,
    so thresholds come out far too strict (calibration bug found on ShanghaiTech:
    0% of legitimately new clips could pass a 95% gate).
    """
    out = np.empty((len(feats), memory.n_regions), dtype=np.float32)
    uids = np.asarray(uids)
    if groups is not None:
        g_of = dict(zip(uids.tolist(), np.asarray(groups).tolist()))
        my_g = np.asarray(groups)
    for r, mem in enumerate(memory.regions):
        sims = feats[:, r] @ mem.feats.T                       # [n, M]
        sims[:, ~mem.valid] = -np.inf
        if groups is None:
            sims[uids[:, None] == mem.source[None, :]] = -np.inf   # leave-one-out
        else:
            src_g = np.array([g_of.get(int(u), None) for u in mem.source], dtype=object)
            sims[my_g[:, None] == src_g[None, :]] = -np.inf          # leave-one-group-out
        out[:, r] = 1.0 - sims.max(axis=1)
    return out


class UpdateRule:
    name = "base"

    def calibrate(self, memory: SceneMemory, train_feats: np.ndarray, train_uids: np.ndarray,
                  groups: Optional[np.ndarray] = None):
        """Set thresholds from training-normal data only. Called once per stream.
        `groups`: video id per training clip, for leave-one-video-out scores."""

    def reset(self):
        """Clear per-stream history."""

    def decide(self, region_scores: np.ndarray, uid: int, t: int) -> np.ndarray:
        raise NotImplementedError


class FrozenRule(UpdateRule):
    name = "frozen"

    def decide(self, region_scores, uid, t):
        return np.zeros(len(region_scores), dtype=bool)


class ScoreGateRule(UpdateRule):
    name = "score_gate"

    def __init__(self, quantile: float = 0.95):
        self.quantile = quantile
        self.beta: Optional[np.ndarray] = None

    def calibrate(self, memory, train_feats, train_uids, groups=None):
        s = training_scores(memory, train_feats, train_uids, groups)
        self.beta = np.quantile(s, self.quantile, axis=0)       # one threshold per region

    def decide(self, region_scores, uid, t):
        if self.beta is None:
            raise RuntimeError("calibrate() first")
        return region_scores < self.beta


class SceneMemoryRule(UpdateRule):
    """Clip-level admission on a normalised score with a 2-clip look-back.
    Normalisation: (s - min_train) / (q99_train - min_train) of training clip scores."""
    name = "scene_memory"

    def __init__(self, admit_thr: float = 0.2, history_thr: float = 0.3, history: int = 2):
        self.admit_thr, self.history_thr = admit_thr, history_thr
        self.hist: deque = deque(maxlen=history)
        self.lo = self.hi = None

    def calibrate(self, memory, train_feats, train_uids, groups=None):
        s = training_scores(memory, train_feats, train_uids, groups).max(axis=1)
        self.lo, self.hi = float(s.min()), float(np.quantile(s, 0.99))

    def reset(self):
        self.hist.clear()

    def normalise(self, clip_score: float) -> float:
        return (clip_score - self.lo) / max(self.hi - self.lo, 1e-12)

    def decide(self, region_scores, uid, t):
        if self.lo is None:
            raise RuntimeError("calibrate() first")
        z = self.normalise(float(np.max(region_scores)))
        ok = (z < self.admit_thr and len(self.hist) == self.hist.maxlen
              and all(h < self.history_thr for h in self.hist))
        self.hist.append(z)                                     # only past clips are used
        return np.full(len(region_scores), ok, dtype=bool)


class TextGateRule(UpdateRule):
    """SAVEN text gate: visual rule AND caption agreement with memory captions.
    `text_agree(uid, region) -> bool` is supplied in W4.1 (cached VLM captions)."""
    name = "saven_text"

    def __init__(self, visual: UpdateRule, text_agree: Optional[Callable[[int, int], bool]] = None):
        self.visual, self.text_agree = visual, text_agree

    def calibrate(self, memory, train_feats, train_uids, groups=None):
        self.visual.calibrate(memory, train_feats, train_uids, groups)

    def reset(self):
        self.visual.reset()

    def decide(self, region_scores, uid, t):
        if self.text_agree is None:
            raise NotImplementedError("text gate needs cached captions (W4.1)")
        v = self.visual.decide(region_scores, uid, t)
        return np.array([v[r] and self.text_agree(uid, r) for r in range(len(v))])


def build_rule(name: str, **kw) -> UpdateRule:
    return {"frozen": FrozenRule, "score_gate": ScoreGateRule,
            "scene_memory": SceneMemoryRule}[name](**kw)
