import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _frame(rng, t, anomalous):
    """856x480 synthetic frame: textured background, a 'person' blob that walks;
    anomalous frames get a large bright object in the top-left region."""
    img = np.full((480, 856, 3), 90, np.uint8)
    img[::16] = 60
    x = 100 + (t * 7) % 600
    cv2.circle(img, (x, 360), 18, (200, 180, 160), -1)
    if anomalous:
        cv2.rectangle(img, (60, 40), (360, 220), (250, 30, 30), -1)
    noise = rng.integers(0, 6, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


@pytest.fixture(scope="session")
def fake_shanghaitech(tmp_path_factory):
    """Tiny dataset in the official layout: 2 scenes, avi training videos,
    frame-folder test videos with masks."""
    root = tmp_path_factory.mktemp("sht")
    rng = np.random.default_rng(0)
    (root / "training" / "videos").mkdir(parents=True)
    (root / "testing" / "frames").mkdir(parents=True)
    (root / "testing" / "test_frame_mask").mkdir(parents=True)
    for scene in ["01", "02"]:
        for v in range(2):                                   # training: 2 x 70 frames
            path = root / "training" / "videos" / f"{scene}_{v + 1:03d}.avi"
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (856, 480))
            for t in range(70):
                w.write(cv2.cvtColor(_frame(rng, t + 31 * v, False), cv2.COLOR_RGB2BGR))
            w.release()
        vid = f"{scene}_0014"                                # test: 75 frames, 30-49 anomalous
        d = root / "testing" / "frames" / vid
        d.mkdir()
        mask = np.zeros(75, np.int8)
        mask[30:50] = 1
        for t in range(75):
            cv2.imwrite(str(d / f"{t:03d}.jpg"),
                        cv2.cvtColor(_frame(rng, t, bool(mask[t])), cv2.COLOR_RGB2BGR))
        np.save(root / "testing" / "test_frame_mask" / f"{vid}.npy", mask)
    return root


@pytest.fixture
def cfg_for(tmp_path):
    def make(data_root, grid=2):
        return {"dataset": "shanghaitech", "data_root": str(data_root),
                "features_root": str(tmp_path / "features"), "scenes": None, "seed": 0,
                "clip": {"length": 16, "frames_per_clip": 4, "grid": grid},
                "encoder": {"name": "dummy", "device": "cpu", "batch_size": 32, "fp16": False},
                "baseline": {"k": [1, 5], "sigma": [0, 3]}}
    return make
