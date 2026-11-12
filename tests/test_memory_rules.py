import numpy as np
import pytest

from src.memory.memory import RegionMemory, SceneMemory
from src.rules.rules import (FrozenRule, SceneMemoryRule, ScoreGateRule, TextGateRule,
                             training_scores)
from src.streams.runner import run_stream


def unit(rng, n, d):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_scene(n_train=200, R=4, D=16, size=32, alpha=0.25, seed=0):
    rng = np.random.default_rng(seed)
    centre = unit(rng, 1, D)[0]
    feats = centre + 0.15 * rng.normal(size=(n_train, R, D)).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=-1, keepdims=True)
    uids = np.arange(n_train) + 10_000
    mem = SceneMemory(R, D, size, alpha)
    mem.init_from(feats, uids, seed)
    return mem, feats, uids, centre, rng


def test_init_and_anchor_count():
    mem, *_ = make_scene()
    r = mem.regions[0]
    assert r.n_anchor == 8 and r.valid.all() and r.from_train.all()


def test_anchors_never_replaced_and_fifo():
    mem = RegionMemory(dim=4, size=4, anchor_frac=0.5)
    mem.init_from(np.eye(4, dtype=np.float32), np.array([1, 2, 3, 4]), np.random.default_rng(0))
    anchors_before = mem.source[mem.is_anchor].copy()
    x = np.ones(4, np.float32) / 2
    ev = [mem.insert(x, 100 + t, t) for t in range(5)]
    assert (mem.source[mem.is_anchor] == anchors_before).all()
    # the two initial adaptive slots go first, then the oldest admitted clip (100)
    assert ev[2] in (100,) and ev[3] == 101
    assert set(mem.admitted_sources()) == {103, 104}


def test_no_adaptive_slots():
    mem = RegionMemory(dim=4, size=4, anchor_frac=1.0)
    mem.init_from(np.eye(4, dtype=np.float32), np.arange(4), np.random.default_rng(0))
    assert mem.insert(np.ones(4, np.float32), 9, 0) is None


def test_undo_by_provenance():
    mem, feats, *_ = make_scene()
    x = feats[:1]
    mem.update(x[0], np.ones(4, bool), uid=777, t=0)
    assert mem.evict_sources([777]) == 4
    assert all(777 not in r.source for r in mem.regions)


def test_leave_one_out_training_scores():
    mem, feats, uids, *_ = make_scene()
    s = training_scores(mem, feats, uids)
    assert (s > 1e-4).all()                     # no clip matched to itself


def test_scoring_is_causal():
    """A clip's score must not change if it (or a later clip) is admitted."""
    mem, feats, uids, *_ = make_scene()
    stream = feats[:5] + 0.0
    a = run_stream(mem, FrozenRule(), stream, np.arange(5))
    mem2, *_ = make_scene()
    rule = ScoreGateRule(1.0)
    rule.calibrate(mem2, feats, uids)
    rule.beta[:] = 10.0                          # admit everything
    b = run_stream(mem2, rule, stream, np.arange(5))
    assert b["admitted"].all()
    assert b["clip_scores"][0] == pytest.approx(a["clip_scores"][0])


def test_score_gate_learns_new_normal_frozen_does_not():
    mem_f, feats, uids, centre, rng = make_scene(size=64)
    mem_g, *_ = make_scene(size=64)
    D = feats.shape[-1]
    # a legitimate change: clips drift slowly towards a new direction
    new = unit(rng, 1, D)[0]
    steps = []
    for i in range(300):
        w = min(i / 200, 1.0)
        c = (1 - w) * centre + w * new + 0.1 * rng.normal(size=(4, D))
        steps.append(c / np.linalg.norm(c, axis=-1, keepdims=True))
    stream = np.array(steps, dtype=np.float32)
    gate = ScoreGateRule(0.95)
    gate.calibrate(mem_g, feats, uids)
    fr = run_stream(mem_f, FrozenRule(), stream, np.arange(300))
    gr = run_stream(mem_g, gate, stream, np.arange(300))
    assert gr["admitted"].sum() > 0 and fr["admitted"].sum() == 0
    assert gr["clip_scores"][-50:].mean() < fr["clip_scores"][-50:].mean()


def test_scene_memory_rule_needs_history():
    mem, feats, uids, *_ = make_scene()
    rule = SceneMemoryRule()
    rule.calibrate(mem, feats, uids)
    low = np.full(4, rule.lo)                    # normalised score 0
    assert not rule.decide(low, 1, 0).any()      # no history yet
    assert not rule.decide(low, 2, 1).any()
    assert rule.decide(low, 3, 2).all()          # two good clips before it
    high = np.full(4, rule.hi * 2)
    rule.decide(high, 4, 3)
    assert not rule.decide(low, 5, 4).any()      # a bad clip in the look-back


def test_text_gate_hook():
    with pytest.raises(NotImplementedError):
        TextGateRule(FrozenRule()).decide(np.zeros(4), 0, 0)
    g = TextGateRule(FrozenRule(), text_agree=lambda uid, r: True)
    assert not g.decide(np.zeros(4), 0, 0).any()
