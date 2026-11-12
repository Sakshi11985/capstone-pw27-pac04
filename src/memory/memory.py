"""W2.1: bounded regional memory with fixed anchor slots, adaptive slots and provenance.

Per scene there is one RegionMemory per region. Each holds M slots:
  * round(alpha * M) anchor slots, filled with training-normal clips, never changed;
  * the rest are adaptive slots. They start filled with other training-normal
    clips and are replaced (oldest first) when an update rule admits a clip.
Every slot records the uid of the clip that created it (provenance) and when it
was inserted, so later defences can undo admissions (W4.2).

Strict causality is enforced by the stream runner: a clip is scored against the
memory as it was *before* that clip, and only then may it be admitted.
"""
from __future__ import annotations

import numpy as np

EMPTY = -1


class RegionMemory:
    def __init__(self, dim: int, size: int = 512, anchor_frac: float = 0.25):
        if not 0.0 <= anchor_frac <= 1.0:
            raise ValueError("anchor_frac must be in [0, 1]")
        self.size = size
        self.n_anchor = int(round(anchor_frac * size))
        self.feats = np.zeros((size, dim), dtype=np.float32)
        self.valid = np.zeros(size, dtype=bool)
        self.is_anchor = np.zeros(size, dtype=bool)
        self.is_anchor[: self.n_anchor] = True
        self.source = np.full(size, EMPTY, dtype=np.int64)    # clip uid that wrote the slot
        self.from_train = np.zeros(size, dtype=bool)
        self.t_insert = np.full(size, -1, dtype=np.int64)     # -1 = initial fill
        self.last_evicted_t = -1

    # ---------------------------------------------------------------- setup
    def init_from(self, feats: np.ndarray, uids: np.ndarray, rng: np.random.Generator):
        """Fill anchors, then adaptive slots, with a random sample of training clips."""
        n = len(feats)
        order = rng.permutation(n)[: self.size]
        self.feats[: len(order)] = feats[order]
        self.valid[: len(order)] = True
        self.source[: len(order)] = np.asarray(uids)[order]
        self.from_train[: len(order)] = True
        if len(order) < self.n_anchor:
            raise ValueError(f"only {n} training clips for {self.n_anchor} anchor slots")

    # ---------------------------------------------------------------- read
    def entries(self) -> np.ndarray:
        return self.feats[self.valid]

    def score(self, x: np.ndarray, k: int = 1) -> float:
        """Mean cosine distance from x [D] to its k nearest valid entries."""
        sims = self.feats[self.valid] @ x
        k = min(k, sims.shape[0])
        top = np.partition(-sims, k - 1)[:k]
        return float(1.0 + top.mean())          # 1 - mean(sim)

    # ---------------------------------------------------------------- write
    def insert(self, x: np.ndarray, uid: int, t: int) -> int | None:
        """Write x into an empty adaptive slot, else replace the oldest adaptive slot.
        Returns the uid that was evicted (EMPTY if the slot was empty), or None if
        there are no adaptive slots at all."""
        adaptive = ~self.is_anchor
        if not adaptive.any():
            return None
        empty = np.flatnonzero(adaptive & ~self.valid)
        if len(empty):
            slot = int(empty[0])
        else:
            cand = np.flatnonzero(adaptive)
            slot = int(cand[np.argmin(self.t_insert[cand])])   # oldest first (-1 = initial)
        evicted = int(self.source[slot]) if self.valid[slot] else EMPTY
        # when the evicted entry had been written during the stream (-1 = initial fill)
        self.last_evicted_t = int(self.t_insert[slot]) if self.valid[slot] else -1
        self.feats[slot] = x
        self.valid[slot] = True
        self.source[slot] = uid
        self.from_train[slot] = False
        self.t_insert[slot] = t
        return evicted

    def evict_sources(self, uids) -> int:
        """Undo: empty every adaptive slot written by one of `uids`. Returns count."""
        hit = np.isin(self.source, np.asarray(list(uids))) & ~self.is_anchor & self.valid
        self.valid[hit] = False
        self.source[hit] = EMPTY
        return int(hit.sum())

    def admitted_sources(self) -> np.ndarray:
        """uids of clips admitted during the stream (not the initial training fill)."""
        m = self.valid & ~self.from_train
        return self.source[m]


class SceneMemory:
    """One RegionMemory per region of a scene."""

    def __init__(self, n_regions: int, dim: int, size: int = 512, anchor_frac: float = 0.25):
        self.regions = [RegionMemory(dim, size, anchor_frac) for _ in range(n_regions)]

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    def init_from(self, feats: np.ndarray, uids: np.ndarray, seed: int = 0):
        """feats [n, R, D] training-normal clips of this scene."""
        rng = np.random.default_rng(seed)
        for r, mem in enumerate(self.regions):
            mem.init_from(feats[:, r], uids, rng)

    def score_regions(self, clip: np.ndarray, k: int = 1) -> np.ndarray:
        return np.array([m.score(clip[r], k) for r, m in enumerate(self.regions)], dtype=np.float32)

    def update(self, clip: np.ndarray, admit: np.ndarray, uid: int, t: int) -> list:
        """Insert region r of the clip wherever admit[r] is True."""
        return [m.insert(clip[r], uid, t) if admit[r] else None
                for r, m in enumerate(self.regions)]

    def evict_sources(self, uids) -> int:
        return sum(m.evict_sources(uids) for m in self.regions)
