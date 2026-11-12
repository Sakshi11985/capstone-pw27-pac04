"""Gate G2: controlled illumination change as the legitimate change of phase B."""
import numpy as np
import pandas as pd
import pytest

from src.preprocess.appearance import PRESETS, apply_appearance, resolve_spec
from src.preprocess.extract import feature_dir, requested_outputs
from src.preprocess.extract import run as extract
from src.streams.generator import (APPEARANCE_UID_OFFSET, TEST_UID_OFFSET, SceneTooSmall,
                                   StreamConfig, build_appearance_pools, make_stream)
from tests.test_streams import synthetic_scene


def test_presets_change_brightness_as_named():
    rng = np.random.default_rng(0)
    f = rng.integers(40, 220, (48, 64, 3), dtype=np.uint8)
    m = {n: apply_appearance(f, resolve_spec(n), "01_001", 7).astype(float).mean(axis=(0, 1))
         for n in PRESETS}
    base = f.astype(float).mean(axis=(0, 1))
    assert m["dim"].mean() < 0.6 * base.mean()
    assert m["night"].mean() < m["dim"].mean()
    assert m["bright"].mean() > base.mean()
    assert m["warm"][0] > base[0] and m["warm"][2] < base[2]     # redder, less blue


def test_appearance_is_deterministic_and_frame_dependent():
    f = np.full((8, 8, 3), 128, np.uint8)
    s = resolve_spec("night")
    a = apply_appearance(f, s, "01_001", 3)
    assert np.array_equal(a, apply_appearance(f, s, "01_001", 3))
    assert not np.array_equal(a, apply_appearance(f, s, "01_001", 4))   # fresh noise per frame
    assert np.array_equal(apply_appearance(f, resolve_spec("warm"), "x", 0),
                          apply_appearance(f, resolve_spec("warm"), "y", 9))  # no noise
    assert f.max() == 128                                                # input untouched


def test_unknown_preset_fails_early(cfg_for, tmp_path):
    cfg = cfg_for(tmp_path)
    with pytest.raises(ValueError):
        requested_outputs(cfg, appearances=["dusky"])


def test_extract_appearance_outputs(fake_shanghaitech, cfg_for):
    cfg = cfg_for(fake_shanghaitech)
    cfg["extract"] = {"inputs": ["rgb", "diff"], "grids": [2]}
    extract(cfg, verbose=False, splits=("train",))
    extract(cfg, verbose=False, splits=("train",), appearances=["dim", "warm"])
    for inp in ("rgb", "diff"):
        orig = feature_dir(cfg, 2, None, inp)
        assert orig.name.endswith("_g2")
        for app in ("dim", "warm"):
            d = feature_dir(cfg, 2, None, inp, app)
            assert d.name.endswith(f"_g2_{app}")
            x, y = (np.load(p / "train" / "01_001.npy") for p in (orig, d))
            assert x.shape == y.shape and not np.allclose(x, y)
            a, b = (pd.read_csv(p / "train_clips.csv") for p in (orig, d))
            assert a[["video_id", "clip_idx"]].equals(b[["video_id", "clip_idx"]])
            assert not (d / "test").exists() or not any((d / "test").iterdir())


def relit_scene(shift=1.0, seed=0):
    tr, trf, te, tef = synthetic_scene(seed)
    rng = np.random.default_rng(99)
    light = rng.normal(size=next(iter(trf.values())).shape[-1])
    apf = {}
    for v, x in trf.items():
        y = x + shift * light / np.linalg.norm(light)
        apf[v] = (y / np.linalg.norm(y, axis=-1, keepdims=True)).astype(np.float32)
    return tr, trf, te, tef, tr.copy(), apf


CFG = StreamConfig(change="appearance", appearance="dim", change_video_frac=0.4,
                   min_novelty=0.2, min_clips={"A": 20, "B": 30, "D": 20, "E": 20}, burst_len=8)


def test_appearance_stream_leakage_rules():
    tr, trf, te, tef, ap, apf = relit_scene()
    p = build_appearance_pools("01", tr, trf, te, tef, ap, apf, CFG)
    assert p.mode == "appearance" and p.cluster_info["novelty"] >= 0.2
    st = make_stream(p, 0, CFG)
    s, init = st["stream"], set(st["init_uids"].tolist())
    B = s[s.phase == "B"]
    relit = lambda u: APPEARANCE_UID_OFFSET <= u < TEST_UID_OFFSET
    assert B.uid.map(relit).all() and (B.label == 0).all()
    assert all(relit(u) for u in st["probe_uids"]) and len(st["probe_uids"])
    assert not s[s.phase != "B"].uid.map(relit).any()
    # the originals of re-lit videos never enter the memory or the stream
    vid = p.train.set_index("uid").video_id
    change_v = set(B.video_id) | {vid[u - APPEARANCE_UID_OFFSET] for u in st["probe_uids"]}
    orig_used = {vid[u] for u in init} | set(s[s.uid < APPEARANCE_UID_OFFSET].video_id)
    assert not (change_v & orig_used)
    assert set(B.video_id).isdisjoint({vid[u - APPEARANCE_UID_OFFSET] for u in st["probe_uids"]})
    assert list(s.phase.drop_duplicates()) == list("ABCDE")
    a = make_stream(p, 1, CFG)["stream"].uid.tolist()
    assert a == make_stream(p, 1, CFG)["stream"].uid.tolist() and a != s.uid.tolist()


def test_invisible_appearance_change_is_skipped():
    tr, trf, te, tef, ap, apf = relit_scene(shift=0.0)
    with pytest.raises(SceneTooSmall):
        build_appearance_pools("01", tr, trf, te, tef, ap, apf, CFG)


def test_relit_clips_count_as_normal_for_contamination():
    from src.streams.kernel import impurity_tracker
    _, is_anom = impurity_tracker(np.ones(10, int))
    assert not is_anom(np.array([APPEARANCE_UID_OFFSET + 3])).any()
    assert is_anom(np.array([TEST_UID_OFFSET + 3])).all()
