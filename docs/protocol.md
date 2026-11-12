# Protocol (pre-registration) — DRAFT

Status: **draft, not frozen.** Freeze date: ____  Agreed with Prof. Pavan A C on: ____
After freezing, change only with a dated note at the bottom of this file.

## Main question
For self-updating video anomaly detectors, how cheaply can a patient person teach
the detector to ignore an anomaly, compared with how fast it learns a legitimate
change, and which evidence (a semantic text gate, recurrence over time, sparse
human checks) breaks that link?

## Hypotheses and falsification thresholds (proposed)

| ID | Hypothesis | Falsified if |
|---|---|---|
| H1 | Visual-only gates keep ρ = T_attack / T_adapt in a narrow band (0.5–2). Framed as a video test of Balachandran (2026), Thm 15. | Any visual-only gate reaches ρ > 3 at matched false-alarm rate and matched T_adapt. |
| H2 | A grounded text gate raises ρ for kind targets (walk → climb) but not amount targets (walk → run). | ρ gain for amount ≥ gain for kind (paired test). |
| H3 | Time-based defences raise ρ only by slowing legitimate adaptation equally. | A defence raises ρ > 2× while T_adapt rises < 1.25×. |
| H4 | For most anomaly types a chain of real same-scene clips exists with every step under the alarm. | Chains exist for < 25% of target types on all datasets. |
| RQ5 | Constructed-stream results transfer to LTD. | Ranking of update rules on LTD disagrees. |

## Data and features (fixed in code, `src/preprocess/clips.py`)
* Clips: 16 consecutive frames, non-overlapping, from frame 0. Trailing frames
  (< 16) are not encoded and take the last clip's score.
* 4 frames sampled per clip at offsets 0, 5, 10, 15.
* 2×2 regions, row-major (0 TL, 1 TR, 2 BL, 3 BR); each region resized directly
  to 224×224 (no centre crop); frozen CLIP ViT-B/16 (OpenAI weights, open_clip 2.24.0).
* Region feature = mean of the 4 sampled frames' features, then L2-normalised; stored float16.
* Clip label = 1 if any frame anomalous.
* Splits: official ShanghaiTech train/test. No test clip ever enters a memory.

## Feature channels and score aggregation (added 2026-11-12)
* **Motion channel ("diff"):** for each sampled frame at offset o, the image
  |frame(o) − frame(o − 2)| (or o + 2 at the start of the clip) per pixel and colour,
  multiplied by 4 and clipped to 0–255, encoded by the same frozen CLIP and averaged
  like the rgb channel. Motivation: averaged per-frame CLIP features are nearly blind to
  motion (running, fighting); Hu, Du & Wang (MM 2025) use inter-frame RGB differences.
* **Fusion:** channels are concatenated per region and divided by √(number of channels),
  so the cosine similarity of fused features is the mean of the per-channel similarities.
* **Grids:** 1×1, 2×2 (main) and 3×3 are cached in the same pass.
* **The novelty test that picks the legitimate change uses the same aggregate as the alarms**
  (it used the max until 12 Nov; with the mean, scenes with a visible change went from 4 to 7).
* **Region scores are combined by their MEAN** (was max). Evidence (scenes 01–03, rgb 2×2,
  k = 1, σ = 3): mean 0.760 vs max 0.752; with max the noisiest region (top-left) gave the
  score for 52% of normal clips. Per-region normalisation added nothing (0.761). Memories,
  gates and admission stay per region.

## Frozen baseline (G1 sanity check)
* Memory per scene and region = all training clips of that scene.
* Clip score = mean over regions (max until 2026-11-12) of the mean cosine distance to the k nearest entries (k ∈ {1, 5}).
* Clip scores → Gaussian smoothing over clips (σ ∈ {0, 3, 7}) → frames → min–max per video.
* Report micro frame AUC (main) and macro (videos with both classes).

## Self-updating detectors (W2.1–W2.2)
* Memory per scene and region: M = 512 slots, anchor fraction α = 0.25, anchors
  and adaptive slots initially filled with a seeded random sample of training clips.
* Replacement: empty adaptive slot first, else the oldest adaptive slot.
* Strict causality: score clip t against memory before clip t, then update.
* Score gate: admit region r if its score < β_r, β_r = 90th / 95th percentile of
  leave-one-out training scores for region r.
* Scene-memory rule: normalised clip score z = (s − min_train) / (q99_train − min_train);
  admit all regions if z < 0.2 and the previous two clips had z < 0.3.
* SAVEN text gate: visual gate AND caption agreement (τ_t fixed in W4.1).
* Gate thresholds: leave-one-video-out (see Phase 2 decisions below).
* **Open decision for the team:** is the WARDEN "frozen reserve" the anchor slots,
  or a separate held-out set of training clips? (Balachandran's guarantee needs
  admission evidence that never reads the memory being fed.)

## Streams (W2.3, implemented in `src/streams/generator.py`; settings in the config)
* Legitimate change: k-means (k searched over 3–6) on mean features of whole training
  videos; the held-out group is the one a memory without it flags most ("novelty"),
  at most half the scene's clips. A scene is skipped if no group reaches novelty 0.1.
* Half of the held-out videos are fed in phase B (least novel first); the other half
  form a PROBE that is never fed and is used to measure adaptation.
* Old-normal training videos are split by video: half fill the memory, half feed
  phases A, D and E. No stream clip has a copy in memory.
* Burst (C): up to 10 consecutive anomalous clips of one test video. Persistent (D):
  20% anomalous clips from the 3 most similar anomalous test videos (a proxy for one
  anomaly type), spread evenly; clips repeat when the pool is small (counted).
* 5 stream seeds per scene (video split, order, burst video, anomaly type).

## Phase 3: admission kernel (implemented 2026-11-12, `src/streams/kernel.py`, `scripts/run_kernel.py`)
* **Kernel streams:** per scene, constant anomaly rate π ∈ {0.02, 0.05, 0.1, 0.2, 0.3, 0.5},
  order iid or bursty (blocks of 10), 3 seeds, ~800 clips. Old-normal clips from training
  videos not used to fill the memory (split by video); anomalies from the 3 most similar
  anomalous test videos (one "type"), repeated when the pool is small.
* **Impurity ρ:** anomalous share of the valid adaptive slots of a region, recorded just
  before every clip. Admission kernel q(ρ) = share of admissions that are anomalous.
* **Within-run slopes:** pooled slopes are estimated with each run's mean removed. Pooling
  across π gives a slope ≈ 1 by construction (high-π runs have both high ρ and high q).
* **Composition law:** q(ρ) = π·pₐ(ρ) / (π·pₐ(ρ) + (1−π)·pₙ(ρ)), with pₐ, pₙ the admission
  probabilities of anomalous and normal clips; ρ* is its stable fixed point from ρ = 0.
  Fitted on the first half of each run, compared with the mean ρ of the last quarter.
* **Proposed rewording of H1/H1b (for the guide):** H1 = self-reinforcement, pₐ increases
  with ρ (within-run slope, 95% CI > 0); report the within-run q slope as R₀. H1b = the
  composition law predicts the reached impurity with median absolute error ≤ 0.1, and the
  critical rate π_c within a factor 2; the affine urn of [7] is reported alongside.

## Phase 2 decisions to confirm with the guide (2026-11-12)
1. **Gate calibration is leave-one-VIDEO-out.** Thresholds of the score gate and the
   scene-memory rule come from training clips scored with every memory entry of their
   own video ignored. Leaving out only the clip itself made the 95% gate admit 0% of
   legitimately new clips (adjacent clips of a video are near-duplicates): a bug.
2. **T_adapt finish line = nominal alarm rate (1%) + 1 pp, same for every rule.** The
   rule's own phase-A rate (kept as `T_adapt_ownbase`) penalises rules that already
   adapted during A.
3. **Speed measure T_adapt90 (proposed for ρ):** clips until 90% of the probe's excess
   false alarms are removed and stay removed. On scene 01 the score gate removes most
   excess within ~40 clips and then plateaus at 2–7% (probe content that no fed video
   covers), so the strict T_adapt mostly measures that residual, not speed.

## Metrics
Copied from the work plan, Section 5: alarm threshold (1% of clean phase-A clips),
T_adapt (50-clip window, phase-A rate + 1 pp), T_attack, ρ, contamination share,
contamination lifetime, recovery, retention, chain existence.

## Statistics
5 stream seeds; medians with 95% bootstrap CIs (1,000 resamples); Wilcoxon
signed-rank with Holm correction for paired comparisons.

## Change log
* (none yet)
