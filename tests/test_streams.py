"""W2.3/W2.4: stream construction rules and adaptation metrics."""
import math

import numpy as np
import pandas as pd
import pytest

from src.eval.adaptation import (first_good_window, fraction_settled_time, settled_time,
                                 alarm_threshold)
from src.memory.memory import SceneMemory
from src.rules.rules import ScoreGateRule, training_scores
from src.streams.generator import StreamConfig, build_scene_pools, make_stream


def synthetic_scene(seed=0, D=16, R=4):
    """One scene: 12 training videos in 3 visual groups (group 2 = the 'change'),
    6 test videos with anomalous runs. Returns tables + feature dicts like load_split."""
    rng = np.random.default_rng(seed)
    centres = [rng.normal(size=D) for _ in range(3)]
    tr_rows, te_rows, trf, tef = [], [], {}, {}
    for v in range(12):
        vid, g = f"01_{v + 1:03d}", v % 3
        n = 40
        x = centres[g] + 0.3 * rng.normal(size=(n, R, D))
        trf[vid] = (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)
        tr_rows += [dict(video_id=vid, scene_id="01", clip_idx=i, label=0) for i in range(n)]
    anom_c = rng.normal(size=D)
    for v in range(6):
        vid, n = f"01_{v + 1:04d}", 30
        lab = np.zeros(n, int)
        lab[10:22] = 1
        x = centres[0] + 0.3 * rng.normal(size=(n, R, D))
        x[lab == 1] += 1.5 * anom_c
        tef[vid] = (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)
        te_rows += [dict(video_id=vid, scene_id="01", clip_idx=i, label=int(lab[i]))
                    for i in range(n)]
    return pd.DataFrame(tr_rows), trf, pd.DataFrame(te_rows), tef


CFG = StreamConfig(n_clusters=[3], min_novelty=0.0, max_holdout_frac=0.5,
                   min_clips={"A": 20, "B": 30, "D": 20, "E": 20}, burst_len=8)


@pytest.fixture(scope="module")
def pools():
    tr, trf, te, tef = synthetic_scene()
    return build_scene_pools("01", tr, trf, te, tef, CFG)


def test_stream_leakage_rules(pools):
    st = make_stream(pools, 0, CFG)
    s, init = st["stream"], set(st["init_uids"].tolist())
    held = set(pools.train[pools.train.cluster == pools.heldout].uid)
    assert not (init & held), "held-out change entered the initial memory"
    stream_train = set(s.uid[s.uid < 1_000_000])
    assert not (init & stream_train), "a stream clip is also in the initial memory"
    vid = pools.train.set_index("uid").video_id
    assert not ({vid[u] for u in init} & {vid[u] for u in stream_train}), "video split leaked"
    for p in "ABE":                                    # no repeats outside phase D
        u = s.uid[s.phase == p]
        assert u.is_unique
    assert (s.label[s.phase.isin(list("ABE"))] == 0).all()
    assert (s.label[s.phase == "C"] == 1).all()
    assert set(st["probe_uids"]).isdisjoint(set(s.uid)), "probe clips must never be fed"
    assert list(s.phase.drop_duplicates()) == list("ABCDE")


def test_stream_is_deterministic_and_seed_dependent(pools):
    a = make_stream(pools, 1, CFG)["stream"].uid.tolist()
    b = make_stream(pools, 1, CFG)["stream"].uid.tolist()
    c = make_stream(pools, 2, CFG)["stream"].uid.tolist()
    assert a == b and a != c


def test_leave_one_video_out_calibration_is_less_strict():
    """Neighbouring clips of the same video are near-duplicates: leaving out only the
    clip itself makes the gate threshold far too strict."""
    rng = np.random.default_rng(0)
    vids = np.repeat(np.arange(10), 30)
    base = rng.normal(size=(10, 8))[vids]                  # one look per video
    x = base[:, None, :] + 0.02 * rng.normal(size=(300, 1, 8))
    x = (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)
    uids = np.arange(300)
    mem = SceneMemory(1, 8, 300, 0.25)
    mem.init_from(x, uids, 0)
    clip_loo = training_scores(mem, x, uids).max(1)
    video_loo = training_scores(mem, x, uids, groups=vids).max(1)
    assert np.median(video_loo) > 5 * np.median(clip_loo)
    g1, g2 = ScoreGateRule(0.95), ScoreGateRule(0.95)
    g1.calibrate(mem, x, uids)
    g2.calibrate(mem, x, uids, groups=vids)
    assert (g2.beta > g1.beta).all()


def test_settled_time():
    cp = [(0, 0.40), (10, 0.10), (20, 0.015), (30, 0.05), (40, 0.012), (50, 0.01)]
    assert settled_time(cp, 0.01, 0.01) == 40           # must STAY under from then on
    assert settled_time([(0, .5), (10, .4)], 0.01, 0.01) == math.inf


def test_fraction_settled_time():
    cp = [(0, 0.41), (10, 0.20), (20, 0.06), (30, 0.04), (40, 0.045)]
    # excess 0.40 -> 90% removed means <= 0.01 + 0.04 = 0.05
    assert fraction_settled_time(cp, 0.01, 0.9, 0.01) == 30
    assert math.isnan(fraction_settled_time([(0, 0.015), (10, 0.01)], 0.01, 0.9, 0.01))


def test_first_good_window_and_threshold():
    a = np.r_[np.ones(60), np.zeros(100)]
    assert first_good_window(a, 0.0, window=50, tol=0.0) == 60
    assert math.isnan(first_good_window(np.zeros(10), 0.0, window=50))
    assert alarm_threshold(np.arange(101.0), 0.01) == pytest.approx(99.0)
