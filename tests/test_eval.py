import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts.baseline import run as baseline
from src.eval.metrics import clip_to_frame_scores, frame_auc, roc_auc, smooth
from src.eval.scoring import knn_distance
from src.preprocess.extract import run as extract


def test_auc_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = rng.normal(size=500) + y * 0.7
    s[::7] = s[0]                                    # include ties
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_frame_expansion_counts():
    fs = clip_to_frame_scores(np.array([0.1, 0.9]), 40, 16)
    assert len(fs) == 40 and fs[15] == 0.1 and fs[16] == 0.9 and fs[39] == 0.9


def test_smoothing_keeps_length():
    x = np.r_[np.zeros(10), 1, np.zeros(10)]
    assert len(smooth(x, 3)) == len(x) and smooth(x, 0)[10] == 1


def test_knn_distance():
    m = np.eye(4, dtype=np.float32)
    q = np.array([[1, 0, 0, 0]], np.float32)
    assert knn_distance(q, m, 1)[0] == pytest.approx(0)
    assert knn_distance(q, m, 2)[0] == pytest.approx(0.5)
    assert knn_distance(q, m, 1, exclude_self=True)[0] == pytest.approx(1)


def test_frame_auc_perfect():
    res = frame_auc({"a": (np.array([0, 0, 1, 0.]), np.r_[np.zeros(32), np.ones(16), np.zeros(16)])},
                    16, 0)
    assert res["micro_auc"] == 1.0 and res["macro_auc"] == 1.0


def test_baseline_end_to_end(fake_shanghaitech, cfg_for, tmp_path):
    cfg = cfg_for(fake_shanghaitech)
    extract(cfg, verbose=False)
    table = baseline(cfg, log=False)
    assert set(table.k) == {1, 5} and len(table) == 4
    # the synthetic anomaly is a big coloured box: any sane baseline must find it
    assert (table.micro_auc > 0.8).all()
