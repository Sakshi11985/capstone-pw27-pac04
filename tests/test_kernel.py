"""Phase 3: kernel streams, impurity tracking, kernel fit and prediction."""
import math

import numpy as np
import pytest

from src.memory.memory import SceneMemory
from src.rules.rules import ScoreGateRule
from src.streams.kernel import (KernelConfig, fit_kernel, impurity_tracker, knee,
                                make_kernel_stream, plain_pools, predicted_equilibrium)
from src.streams.runner import run_stream
from tests.test_streams import synthetic_scene

KC = KernelConfig(length=200, min_normal=50, min_admissions=20, min_anomalous_admissions=3,
                  bootstrap=50)


@pytest.fixture(scope="module")
def pool():
    tr, trf, te, tef = synthetic_scene()
    return plain_pools("01", tr, trf, te, tef), te


@pytest.mark.parametrize("order", ["iid", "bursty"])
def test_kernel_stream_rate_and_split(pool, order):
    p, _ = pool
    ks = make_kernel_stream(p, 0, 0.2, order, KC)
    s = ks["stream"]
    assert abs(s.label.mean() - 0.2) < 0.02
    init = set(ks["init_uids"].tolist())
    vid = p.train.set_index("uid").video_id
    stream_train = [u for u in s.uid if u < 1_000_000]
    assert not (init & set(stream_train))
    assert not ({vid[u] for u in init} & {vid[u] for u in stream_train}), "video leak"
    if order == "bursty":                     # anomalies come in runs of burst_len
        lab = s.label.to_numpy()
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, lab, 0])))[::2]
        assert runs.max() >= KC.burst_len or lab.sum() < KC.burst_len


def test_kernel_stream_deterministic(pool):
    p, _ = pool
    a = make_kernel_stream(p, 3, 0.1, "iid", KC)["stream"].uid.tolist()
    b = make_kernel_stream(p, 3, 0.1, "iid", KC)["stream"].uid.tolist()
    assert a == b


def test_impurity_tracker_and_runner(pool):
    p, te = pool
    rho_fn, is_anom = impurity_tracker(te.label.to_numpy())
    ks = make_kernel_stream(p, 0, 0.3, "iid", KC)
    s = ks["stream"]
    x0 = np.stack([p.feats[int(u)] for u in ks["init_uids"]])
    mem = SceneMemory(x0.shape[1], x0.shape[2], 64, 0.25)
    mem.init_from(x0, ks["init_uids"], 0)
    assert rho_fn(mem).max() == 0.0                       # starts clean
    g = ScoreGateRule(1.0); g.calibrate(mem, x0, ks["init_uids"]); g.beta[:] = 10  # admit all
    X = np.stack([p.feats[int(u)] for u in s.uid])
    out = run_stream(mem, g, X, s.uid.to_numpy(), 1, track=rho_fn)
    assert out["tracked"].shape == (len(s), x0.shape[1])
    assert out["tracked"][0].max() == 0.0
    # admitting everything into a FIFO of 48 adaptive slots: impurity ~ the stream's rate
    assert abs(out["tracked"][-50:].mean() - 0.3) < 0.15
    assert is_anom(np.array([5, 1_000_000 + int(np.flatnonzero(te.label.values)[0])])).tolist() == [False, True]


def simulate_urn(a, b, n=4000, slots=200, seed=0):
    """FIFO memory whose next admission is anomalous with probability a + b*rho."""
    rng = np.random.default_rng(seed)
    mem = [0] * slots
    rho, w, t = [], [], []
    for i in range(n):
        r = sum(mem) / slots
        x = int(rng.random() < min(1.0, a + b * r))
        rho.append(r); w.append(x); t.append(i)
        mem.pop(0); mem.append(x)
    return np.array(rho), np.array(w), np.array(t)


def test_fit_recovers_known_kernel():
    """Over several simulated urns (true R0 = 0.6) the slope is recovered on average,
    the 95% block-bootstrap interval usually covers it, and the predicted equilibrium
    matches where the memory settles (checked over 20 urns: mean 0.53-0.56, coverage
    19-20/20)."""
    bs, cover, err = [], 0, []
    for s in range(6):
        rho, w, t = simulate_urn(0.05, 0.6, n=4000, slots=50, seed=s)
        f = fit_kernel(rho, w, t, KernelConfig(min_admissions=10, bootstrap=100), seed=s)
        bs.append(f.b)
        cover += f.b_lo <= 0.6 <= f.b_hi
        err.append(abs(predicted_equilibrium(f.a, f.b) - rho[-1000:].mean()))
    assert abs(np.mean(bs) - 0.6) < 0.12
    assert cover >= 4
    assert np.median(err) < 0.05


def test_fit_no_reinforcement():
    rho, w, t = simulate_urn(0.1, 0.0, seed=2)
    f = fit_kernel(rho, w, t, KernelConfig(min_admissions=10, bootstrap=100), seed=1)
    assert f.b_lo < 0.0 < f.b_hi or abs(f.b) < 0.3


def test_fit_refuses_thin_data():
    assert fit_kernel(np.zeros(500), np.zeros(500), np.arange(500), KernelConfig()) is None


def test_equilibrium_and_knee():
    assert predicted_equilibrium(0.1, 0.5) == pytest.approx(0.2)
    assert predicted_equilibrium(0.1, 1.2) == 1.0
    assert predicted_equilibrium(0.6, 0.5) == 1.0                 # a >= 1-b saturates
    assert math.isnan(predicted_equilibrium(math.nan, 0.1))
    assert knee([0.05, 0.1, 0.2], [0.1, 0.6, 0.9], 0.5) == 0.1
    assert math.isnan(knee([0.05, 0.1], [0.1, 0.2], 0.5))
