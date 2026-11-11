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
