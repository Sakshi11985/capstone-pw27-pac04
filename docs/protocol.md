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

## Frozen baseline (G1 sanity check)
* Memory per scene and region = all training clips of that scene.
* Clip score = max over regions of mean cosine distance to the k nearest entries (k ∈ {1, 5}).
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
* **Open decision for the team:** is the WARDEN "frozen reserve" the anchor slots,
  or a separate held-out set of training clips? (Balachandran's guarantee needs
  admission evidence that never reads the memory being fed.)

## Metrics
Copied from the work plan, Section 5: alarm threshold (1% of clean phase-A clips),
T_adapt (50-clip window, phase-A rate + 1 pp), T_attack, ρ, contamination share,
contamination lifetime, recovery, retention, chain existence.

## Statistics
5 stream seeds; medians with 95% bootstrap CIs (1,000 resamples); Wilcoxon
signed-rank with Holm correction for paired comparisons.

## Change log
* (none yet)
