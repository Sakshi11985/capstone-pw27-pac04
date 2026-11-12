"""Run one update rule over one stream, strictly causally.

For each clip t:  1) score against memory as it is now;  2) rule decides;
3) memory updated.  The score of clip t never depends on clip t or later.

Besides scores and admissions, the runner records every stream-written memory
entry as an event (region, uid, t_in, t_out); t_out = -1 if still in memory at the
end. `snapshots` maps a stream position to a function called with the memory
right after that clip (e.g. contamination at the end of phase D).
"""
from __future__ import annotations

import numpy as np

from src.eval.scoring import aggregate
from src.memory.memory import EMPTY, SceneMemory
from src.rules.rules import UpdateRule


def run_stream(memory: SceneMemory, rule: UpdateRule, feats: np.ndarray,
               uids: np.ndarray, k: int = 1, snapshots: dict | None = None,
               how: str = "mean", track=None) -> dict:
    """feats [T, R, D] in stream order, uids [T]. Mutates `memory`.
    `track(memory) -> array [R]` is evaluated just BEFORE each clip (e.g. the impurity
    of each region for the kernel measurement); results are returned as "tracked"."""
    T, R = feats.shape[:2]
    reg_scores = np.empty((T, R), dtype=np.float32)
    admitted = np.zeros((T, R), dtype=bool)
    events = []                                      # [region, uid, t_in, t_out]
    snaps = {}
    snapshots = snapshots or {}
    tracked = np.empty((T, R)) if track is not None else None
    rule.reset()
    for t in range(T):
        if track is not None:
            tracked[t] = track(memory)
        s = memory.score_regions(feats[t], k)        # memory before clip t
        a = rule.decide(s, int(uids[t]), t)
        evicted = memory.update(feats[t], a, int(uids[t]), t)   # only afterwards
        for r in np.flatnonzero(a):
            ev = evicted[r]
            mem = memory.regions[r]
            if ev is not None and ev != EMPTY and mem.last_evicted_t >= 0:
                events.append([r, ev, mem.last_evicted_t, t])
        reg_scores[t], admitted[t] = s, a
        if t in snapshots:
            snaps[t] = snapshots[t](memory)
    for r, mem in enumerate(memory.regions):         # entries still present at the end
        live = mem.valid & ~mem.from_train
        for uid, tin in zip(mem.source[live], mem.t_insert[live]):
            events.append([r, int(uid), int(tin), -1])
    return {"region_scores": reg_scores, "clip_scores": aggregate(reg_scores, how),
            "admitted": admitted,
            "events": np.array(events, dtype=np.int64).reshape(-1, 4),
            "snapshots": snaps, "tracked": tracked}
