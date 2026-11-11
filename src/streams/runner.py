"""Run one update rule over one stream, strictly causally.

For each clip t:  1) score against memory as it is now;  2) rule decides;
3) memory updated.  The score of clip t never depends on clip t or later.
The stream generator (W2.3) and the metrics T_adapt etc. (W2.4) build on this.
"""
from __future__ import annotations

import numpy as np

from src.memory.memory import SceneMemory
from src.rules.rules import UpdateRule


def run_stream(memory: SceneMemory, rule: UpdateRule, feats: np.ndarray,
               uids: np.ndarray, k: int = 1) -> dict:
    """feats [T, R, D] in stream order, uids [T]. Mutates `memory`."""
    T, R = feats.shape[:2]
    reg_scores = np.empty((T, R), dtype=np.float32)
    admitted = np.zeros((T, R), dtype=bool)
    rule.reset()
    for t in range(T):
        s = memory.score_regions(feats[t], k)          # memory before clip t
        a = rule.decide(s, int(uids[t]), t)
        memory.update(feats[t], a, int(uids[t]), t)    # only afterwards
        reg_scores[t], admitted[t] = s, a
    return {"region_scores": reg_scores, "clip_scores": reg_scores.max(axis=1),
            "admitted": admitted}
