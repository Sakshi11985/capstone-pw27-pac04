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
    table = baseline(cfg, log=False, out_dir=tmp_path)
    assert (tmp_path / "baseline_shanghaitech_g2_mean.csv").exists()
    assert set(table.k) == {1, 5} and len(table) == 4
    # the synthetic anomaly is a big coloured box: any sane baseline must find it
    assert (table.micro_auc > 0.8).all()


def test_aggregate_and_fusion():
    from src.eval.scoring import aggregate, fuse_channels
    r = np.array([[0.1, 0.5], [0.2, 0.2]])
    assert aggregate(r, "max").tolist() == [0.5, 0.2]
    assert np.allclose(aggregate(r, "mean"), [0.3, 0.2])
    with pytest.raises(ValueError):
        aggregate(r, "median")
    rng = np.random.default_rng(0)
    u = rng.normal(size=(2, 3, 8)); u /= np.linalg.norm(u, axis=-1, keepdims=True)
    v = rng.normal(size=(2, 3, 8)); v /= np.linalg.norm(v, axis=-1, keepdims=True)
    f = fuse_channels([u, v])
    assert np.allclose(np.linalg.norm(f, axis=-1), 1)
    cos_fused = (f[0] * f[1]).sum(-1)
    assert np.allclose(cos_fused, ((u[0] * u[1]).sum(-1) + (v[0] * v[1]).sum(-1)) / 2)


def test_load_features_fuses_channels(fake_shanghaitech, cfg_for):
    from src.eval.scoring import load_features
    cfg = cfg_for(fake_shanghaitech)
    cfg["extract"] = {"inputs": ["rgb", "diff"], "grids": [2]}
    extract(cfg, verbose=False)
    cfg["features"] = {"channels": ["rgb", "diff"]}
    _, f = load_features(cfg, "test")
    assert f["01_0014"].shape == (4, 4, 128)
    cfg["features"] = {"channels": ["rgb", "diff"]}
    cfg["baseline"] = {"k": [1], "sigma": [0]}
    t = baseline(cfg, log=False, out_dir=fake_shanghaitech.parent)
    assert t.channels.iloc[0] == "rgb+diff"


def test_merge_features(fake_shanghaitech, cfg_for, tmp_path):
    """Two zips with different scenes merge into one folder with a joint index."""
    import zipfile
    import pandas as pd
    from scripts.merge_features import merge
    zips = []
    for scene in ("01", "02"):
        c = cfg_for(fake_shanghaitech)
        c["features_root"] = str(tmp_path / f"run{scene}")
        c["scenes"] = [scene]
        extract(c, verbose=False)
        zp = tmp_path / f"f{scene}.zip"
        root = tmp_path / f"run{scene}"
        with zipfile.ZipFile(zp, "w") as z:
            for p in root.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(root))
        zips.append(zp)
    out = tmp_path / "merged"
    counts = merge(zips, out)
    assert counts["shanghaitech/dummy_g2"] == (4, 2)
    te = pd.read_csv(out / "shanghaitech" / "dummy_g2" / "test_clips.csv", dtype={"scene_id": str})
    assert set(te.scene_id) == {"01", "02"}
