# habituation-vad

Habituation attacks and adaptation safety in self-updating video anomaly detection
(UE24AM320A Capstone, PW_27_PAC_04, guide Prof. Pavan A C).

The full plan is in `../Capstone_Work_Plan.docx`. This repository follows its layout (Section 4).

## Status

| Work package | State |
|---|---|
| W0.1 repository, config, seeding, run log | done |
| W0.2 `docs/protocol.md` | draft, **needs team + guide sign-off** |
| W1.2 pre-processing + CLIP feature cache | rgb 2×2 done for scenes 01–03; **one Kaggle pass now gives rgb + motion, 1×1/2×2/3×3, all 13 scenes** |
| W1.3 frozen-memory baseline + eval module | done; **G1 passed** (micro AUC 0.752, k = 1, σ = 3) |
| W2.1 memory (anchor + adaptive slots, provenance, undo) | code done and tested |
| W2.2 update rules: frozen, score gate, scene-memory, text-gate hook | code done and tested |
| W2.3 five-phase streams | done (`src/streams/generator.py`) |
| W2.4 alarm threshold, T_adapt, contamination, recovery, retention | done (`src/eval/adaptation.py`); **G2 not yet passed** |
| Phase 3 kernel q(ρ), R₀, pₐ(ρ), composition law (H1, H1b) | done (`src/streams/kernel.py`, `scripts/run_kernel.py`) |

## Setup (once)

Windows PowerShell or Linux, Python 3.10:

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pytest                   # 46 tests, CPU only, ~1 min
```

## Reproduce Phase 1 in three commands

1. Put ShanghaiTech Campus where `data_root` in `configs/shanghaitech.yaml` points
   (default `data/shanghaitech`, containing `training/` and `testing/`). An absolute
   path is fine. **Keep the raw videos outside OneDrive** if you can, it is several GB.

```bash
python -m src.preprocess.extract --config configs/shanghaitech.yaml   # features (GPU)
python -m scripts.check_counts   --config configs/shanghaitech.yaml   # compare with earlier counts
python -m scripts.baseline       --config configs/shanghaitech.yaml   # frozen baseline AUC (G1)
python -m scripts.run_streams    --config configs/shanghaitech.yaml   # streams, T_adapt etc. (G2), ~1 min, CPU
python -m scripts.merge_features ../features_shanghaitech_all.zip     # add a Kaggle zip to features/
python -m scripts.run_kernel     --config configs/shanghaitech.yaml   # Phase 3: kernel, R0 (H1, H1b), CPU
```

**Feature types.** One extraction pass writes six folders under `features/shanghaitech/`:
`clip_vitb16_g{1,2,3}` (appearance) and `clip_vitb16_diff_g{1,2,3}` (motion: CLIP on
|frame(t) − frame(t−2)| × 4). Choose what experiments use in the config:
`features.channels: [rgb]`, `[diff]` or `[rgb, diff]` (fused), the grid with `clip.grid`,
and how region scores combine with `score.aggregate: mean` (default) or `max`.

Try `--limit 2` on the first command for a 1-minute smoke test. The extractor
skips finished videos, so an interrupted run can just be restarted.
For the grid ablation: add `--grid 1` and `--grid 3` to the first and last command.
For all 13 scenes set `scenes: null`.

### On Kaggle (recommended if the laptop is slow)

Use `notebooks/kaggle_extract.ipynb`: upload this folder (zipped) as a private Kaggle
dataset, add it and ShanghaiTech as inputs, choose **GPU T4** and **Internet on**, then
Run All. It finds both inputs itself, writes `configs/kaggle.yaml`, extracts every
feature type for all 13 scenes in one pass, checks counts, runs the baseline for each
feature type, and zips `features/` as `features_shanghaitech_all.zip`. Put that zip in
the capstone folder and run `python -m scripts.merge_features ../features_shanghaitech_all.zip`.

**Gate G2 (lighting change):** `notebooks/kaggle_relight.ipynb` adds re-lit copies of the
training videos (`dim`, `night`, `warm`, `bright`; rgb + motion, 2×2) as 8 new folders.
Unzip `features_shanghaitech_relit.zip` into `features/`, then

    python -m scripts.run_streams --config configs/shanghaitech.yaml --change appearance --appearance dim

### On Colab instead

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
