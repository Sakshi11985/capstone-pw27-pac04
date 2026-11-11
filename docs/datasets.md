# Datasets

Never commit or redistribute videos. Record where each copy came from and its terms.

| Dataset | Status | Source | Licence / terms | Notes |
|---|---|---|---|---|
| ShanghaiTech Campus | downloaded | https://svip-lab.github.io/dataset/campus_dataset.html | research use; check page | 13 scenes, 330 train / 107 test videos; frame masks in `testing/test_frame_mask/` |
| NWPU Campus | request access | https://campusvad.github.io | by request | 43 scenes, 305 train / 242 test |
| Street Scene | request from MERL | MERL website | by request | 46 train / 35 test, 1280×720 |
| LTD (+ Harborfront labels) | stretch (RQ5) | — | check before download | 8 months of drift |

Expected ShanghaiTech layout under `data_root`:

```
training/videos/01_001.avi ...        (or training/frames/01_001/000.jpg ...)
testing/frames/01_0014/000.jpg ...
testing/test_frame_mask/01_0014.npy
```
