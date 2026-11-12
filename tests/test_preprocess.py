import numpy as np
import pandas as pd
import pytest

from src.preprocess import clips as C
from src.preprocess.datasets import list_shanghaitech
from src.preprocess.extract import run as extract


def test_sample_offsets():
    assert C.sample_offsets(16, 4).tolist() == [0, 5, 10, 15]
    assert C.sample_offsets(16, 1).tolist() == [0]


def test_clip_indexing_and_tail():
    assert C.n_full_clips(40, 16) == 2
    f2c = C.frame_to_clip(40, 2, 16)
    assert f2c[15] == 0 and f2c[16] == 1 and f2c[39] == 1     # tail -> last clip


def test_clip_labels_any_frame():
    fl = np.zeros(48, np.int8)
    fl[20] = 1
    lab, frac = C.clip_labels(fl, 3, 16)
    assert lab.tolist() == [0, 1, 0]
    assert frac[1] == pytest.approx(1 / 16)


@pytest.mark.parametrize("grid", [1, 2, 3])
def test_regions_cover_frame(grid):
    f = np.arange(480 * 856 * 3, dtype=np.int64).reshape(480, 856, 3)
    regs = C.split_regions(f, grid)
    assert len(regs) == grid * grid
    assert sum(r.shape[0] * r.shape[1] for r in regs) == 480 * 856
    assert regs[0][0, 0, 0] == f[0, 0, 0]                    # row-major, top-left first


def test_model_input_shape():
    x = C.to_model_input(np.zeros((240, 428, 3), np.uint8))
    assert x.shape == (3, 224, 224) and x.dtype == np.float32


def test_dataset_listing(fake_shanghaitech):
    tr = list_shanghaitech(fake_shanghaitech, "train")
    te = list_shanghaitech(fake_shanghaitech, "test", scenes=["01"])
    assert len(tr) == 4 and len(te) == 1 and te[0].scene_id == "01"


def test_extract_end_to_end(fake_shanghaitech, cfg_for):
    out = extract(cfg_for(fake_shanghaitech), verbose=False)
    tr = pd.read_csv(out / "train_clips.csv", dtype={"scene_id": str})
    te = pd.read_csv(out / "test_clips.csv", dtype={"scene_id": str})
    assert len(tr) == 4 * (70 // 16)                          # 4 videos x 4 clips
    assert len(te) == 2 * (75 // 16)
    x = np.load(out / "train" / "01_001.npy")
    assert x.dtype == np.float16 and x.shape == (4, 4, 64)
    assert np.allclose(np.linalg.norm(x.astype(np.float32), axis=-1), 1, atol=1e-2)
    # labels: frames 30-49 anomalous -> clips 1 (16-31), 2 (32-47), 3 (48-63)
    v = te[te.video_id == "01_0014"].sort_values("clip_idx")
    assert v.label.tolist() == [0, 1, 1, 1]
    assert len(np.load(out / "labels" / "01_0014.npy")) == 75  # frame count matches labels
    # rerun is a no-op (resume)
    mtime = (out / "train" / "01_001.npy").stat().st_mtime
    extract(cfg_for(fake_shanghaitech), verbose=False)
    assert (out / "train" / "01_001.npy").stat().st_mtime == mtime


def test_selective_reading_matches_full_read(fake_shanghaitech):
    """Skipping unused frames must return exactly the same kept frames and frame count,
    for frame folders (thread pool) and video files (grab)."""
    from src.preprocess.datasets import iter_frames
    tr = list_shanghaitech(fake_shanghaitech, "train")[0]       # .avi
    te = list_shanghaitech(fake_shanghaitech, "test")[0]        # frame folder
    keep = lambda i: i % 16 in (0, 5, 10, 15)
    for item in (tr, te):
        full = list(iter_frames(item))
        part = list(iter_frames(item, keep=keep, workers=3))
        assert len(full) == len(part)
        for i, (a, b) in enumerate(zip(full, part)):
            if keep(i):
                assert np.array_equal(a, b)
            else:
                assert b is None


def test_natural_frame_order(tmp_path):
    import cv2
    from src.preprocess.datasets import VideoItem, iter_frames
    d = tmp_path / "01_0001"
    d.mkdir()
    for t in [1, 2, 10]:                                       # 10.jpg must come last
        cv2.imwrite(str(d / f"{t}.jpg"), np.full((32, 32, 3), t * 20, np.uint8))
    vals = [int(f[0, 0, 0]) for f in iter_frames(VideoItem("01_0001", "01", "test", d))]
    assert [round(v / 20) for v in vals] == [1, 2, 10]


# ---------------------------------------------------------------- multi-output extraction
def test_motion_image_and_partner():
    from src.preprocess.extract import diff_partner, motion_image
    assert [diff_partner(o, 2, 16) for o in (0, 5, 10, 15)] == [2, 3, 8, 13]
    a = np.full((4, 4, 3), 100, np.uint8)
    b = a.copy()
    b[0, 0] = 90
    m = motion_image(a, b, 4.0)
    assert m[0, 0, 0] == 40 and m[1, 1, 0] == 0 and m.dtype == np.uint8
    assert motion_image(np.zeros_like(a), np.full_like(a, 200), 4.0).max() == 255  # clipped


def test_extract_rgb_and_diff_all_grids_one_pass(fake_shanghaitech, cfg_for):
    from src.preprocess.extract import feature_dir
    cfg = cfg_for(fake_shanghaitech)
    cfg["extract"] = {"inputs": ["rgb", "diff"], "grids": [1, 2, 3]}
    extract(cfg, verbose=False)
    for inp in ("rgb", "diff"):
        for g in (1, 2, 3):
            d = feature_dir(cfg, g, None, inp)
            x = np.load(d / "test" / "01_0014.npy")
            assert x.shape == (4, g * g, 64), (inp, g, x.shape)
            te = pd.read_csv(d / "test_clips.csv", dtype={"scene_id": str})
            assert len(te) == 8 and te.label.sum() == 6
    rgb = np.load(feature_dir(cfg, 2, None, "rgb") / "test" / "01_0014.npy")
    dif = np.load(feature_dir(cfg, 2, None, "diff") / "test" / "01_0014.npy")
    assert not np.allclose(rgb, dif)                           # motion is a different signal


def test_rgb_features_unchanged_by_multi_output(fake_shanghaitech, cfg_for, tmp_path):
    """Adding diff/other grids to the pass must not change the rgb 2x2 features."""
    import copy
    from src.preprocess.extract import feature_dir
    a = cfg_for(fake_shanghaitech)
    b = copy.deepcopy(a)
    b["features_root"] = str(tmp_path / "multi")
    b["extract"] = {"inputs": ["rgb", "diff"], "grids": [2, 3]}
    extract(a, verbose=False)
    extract(b, verbose=False)
    x = np.load(feature_dir(a, 2) / "train" / "01_001.npy")
    y = np.load(feature_dir(b, 2) / "train" / "01_001.npy")
    assert np.array_equal(x, y)
