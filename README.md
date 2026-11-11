# habituation-vad

Habituation attacks and adaptation safety in self-updating video anomaly detection
(UE24AM320A Capstone, PW_27_PAC_04, guide Prof. Pavan A C).

The full plan is in `../Capstone_Work_Plan.docx`. This repository follows its layout (Section 4).

## Status

| Work package | State |
|---|---|
| W0.1 repository, config, seeding, run log | done |
| W0.2 `docs/protocol.md` | draft, **needs team + guide sign-off** |
| W1.2 pre-processing + CLIP feature cache | code done and tested; **run it on the GPU** |
| W1.3 frozen-memory baseline + eval module | code done and tested; gives G1 once features exist |
| W2.1 memory (anchor + adaptive slots, provenance, undo) | code done and tested |
| W2.2 update rules: frozen, score gate, scene-memory, text-gate hook | code done and tested |
| W2.3 five-phase streams, W2.4 T_adapt and metrics | next |

## Setup (once)

Windows PowerShell or Linux, Python 3.10:

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pytest                   # 24 tests, CPU only, ~10 s
```

## Reproduce Phase 1 in three commands

1. Put ShanghaiTech Campus where `data_root` in `configs/shanghaitech.yaml` points
   (default `data/shanghaitech`, containing `training/` and `testing/`). An absolute
   path is fine. **Keep the raw videos outside OneDrive** if you can, it is several GB.

```bash
python -m src.preprocess.extract --config configs/shanghaitech.yaml   # features (GPU)
python -m scripts.check_counts   --config configs/shanghaitech.yaml   # compare with earlier counts
python -m scripts.baseline       --config configs/shanghaitech.yaml   # frozen baseline AUC (G1)
```

Try `--limit 2` on the first command for a 1-minute smoke test. The extractor
skips finished videos, so an interrupted run can just be restarted.
For the grid ablation: add `--grid 1` and `--grid 3` to the first and last command.
For all 13 scenes set `scenes: null`.

### On Colab or Kaggle instead of the laptop

Upload the repository and the dataset to Google Drive, then in a GPU notebook:

```python
%cd /content/drive/MyDrive/habituation-vad
!pip -q install open_clip_torch==2.24.0
!python -m src.preprocess.extract --config configs/shanghaitech.yaml
```

Copy `features/` back afterwards; everything after feature extraction runs on a CPU.

## Layout

```
configs/        one YAML per experiment
docs/           protocol.md (hypotheses, thresholds, metrics), datasets.md, decisions.md
src/preprocess  datasets.py (readers), clips.py (clips, sampling, regions), encoders.py, extract.py
src/eval        scoring.py (kNN distance, loading), metrics.py (smoothing, AUC)
src/memory      memory.py  (anchor + adaptive slots, provenance, undo)
src/rules       rules.py   (frozen, score gate, scene-memory, SAVEN text-gate hook)
src/streams     runner.py  (strictly causal score-then-update loop)
scripts/        baseline.py, check_counts.py
results/        runs.csv (append-only), baseline tables, final/
tests/          labels, regions, AUC vs scikit-learn, leakage, memory, causality, rules
```

Rules: every setting comes from a config; every number in the paper comes from a
script and a row in `results/runs.csv`; never edit old rows.
