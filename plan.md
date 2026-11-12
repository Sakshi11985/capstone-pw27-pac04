# Research plan: Measuring and Mitigating Habituation in Continual Video Anomaly Detection

**Project:** PW_27_PAC_04 · UE24AM320A Capstone (Phase I and II)
**Team:** Pradeep T, Prajwal A, Sakshi P Shetty, Shruti Barukule · **Guide:** Prof. Pavan A C
**Version:** 4.0, 12 Nov 2026 (rewritten from scratch). Supersedes v1–v3 and the research content
of `Capstone_Work_Plan.docx`; that document's working rules still apply (append-only results log,
weekly results check, pre-registration before testing).
**Status:** draft for the team and the guide. Hypothesis thresholds are proposals until agreed and
dated in `docs/protocol.md`.

---

## Summary

A camera that keeps learning from its own video can adapt to legitimate change (a new timetable,
moved furniture, different lighting). It can also slowly learn to accept abnormal behaviour that
keeps recurring: it becomes **habituated**. This project (1) measures how strongly a self-updating,
camera-specific video memory reinforces its own contamination, (2) tests whether behaviour-only
exposure sequences can habituate it, (3) measures the trade-off between adapting fast and resisting
habituation, and (4) compares three ways of deciding what the memory may learn: fixed rules,
independent validation (IEAC), and a **safety-constrained learned memory manager** (contextual
bandit, then DQN). Vision-language evidence (a semantic admission gate, and a test of whether
VLM context memories habituate) is part of the plan.

The question is never "which method has the highest AUC". It is: **does a method adapt to
legitimate change without becoming unsafe?** The headline number is ρ_safe = T_attack / T_adapt90.

**Changes in v4:** learned memory management (§7.6, H10–H13, experiment arms D–E, roadmap
stages); related work on bandit/RL memory management (§5); title options (§12.4); team roles
updated. Everything from v3 is kept.

---

## 1. Where we are (12 Nov 2026)

| Item | State | Evidence |
|---|---|---|
| Repository, config, seeding, append-only run log, 46 tests | done | `capstone-pw27-pac04/`, `results/runs.csv` |
| Features: all 13 ShanghaiTech scenes, CLIP ViT-B/16, RGB + motion, 1×1 / 2×2 / 3×3 | done (Kaggle) | 330 train / 107 test videos |
| **G1** frozen baseline sane | **passed** | §1.1 |
| Camera(scene)-specific memory: anchors + adaptive slots, provenance, undo | done | `src/memory/` |
| Update rules: frozen, score gate q90 / q95, causal scene-memory rule | done | `src/rules/` |
| Five-phase streams with held-out probe; T_adapt, T_adapt90, contamination, lifetime, recovery, retention | done | `src/streams/`, `src/eval/adaptation.py` |
| Self-reinforcement: kernel streams, pₐ(ρ), within-run R₀, composition law | done, all scenes, RGB and motion | `results/kernel_*` |
| **G2** adaptation measurable on ≥ 5 scenes | **not passed** | §1.3 |
| Attacker, WARDEN, quarantine, IEAC, RL memory manager, VLM gate, VLM context memory | not started | §7 |

### 1.1 Detection baselines (frame AUC, k = 1, σ = 3, mean of region scores, all 13 scenes)

| Features | Micro AUC | Mean per-scene AUC |
|---|---|---|
| RGB 2×2 | 0.657 | 0.679 |
| RGB 3×3 | 0.686 | 0.704 |
| Motion 2×2 | **0.697** | **0.708** |
| Motion 3×3 | 0.698 | 0.704 |
| RGB + motion 2×2 | 0.685 | – |

The 0.760 quoted in earlier slides is RGB on scenes 01–03 only and must not be compared with the
13-scene numbers. Scenes 04, 09, 10, 12 are hard (0.40–0.66); scene 13 has no test videos.

### 1.2 Self-reinforcement (kernel streams, RGB 2×2)

* An anomalous clip is admitted with probability ≈ 0.13–0.22 on a clean memory and 0.8–0.9 once
  ≈ 10% of the adaptive memory is anomalous; the within-run slope of pₐ(ρ) is > 0 (95% CI) in
  every measurable scene, for both score gates.
* Pooled within-run R₀ = 0.50–0.52 (CI 0.48–0.58): **self-reinforcing but subcritical**. The pooled
  across-run slope ≈ 1 is an artefact of pooling different anomaly rates.
* The **composition law** (§7.3) predicts where contamination settles with median error 0.04, vs
  0.24 for the affine urn of Balachandran [7].
* With **motion** features reinforcement is much weaker (R₀ 0.09–0.21; law error 0.018): the
  representation changes the poisoning dynamics.

### 1.3 Adaptation streams and G2

Using the mean aggregate for the novelty test raised the scenes with a visible legitimate change
from 4 to 7 (RGB); phase-length minima leave 4 usable scenes, and T_adapt90 is finite on ≥ 3/5
seeds in 3 of them. The q95 gate learns the change fast (probe false alarms 88% → 5% in scene 01)
while it stops flagging a repeating anomaly (detection 100% → 88% across phase D), with 11% of its
adaptive memory anomalous. This is the trade-off the project is about.
**Cause of the G2 failure:** within one ShanghaiTech scene the training videos are too similar, so
a held-out group of them is a weak "legitimate change". Planned fix: a controlled
illumination/appearance change (§14). The criterion is not relaxed to obtain a pass.

---

## 2. Problem

### 2.1 Setting
A college corridor camera learns normal activity: walking, gathering near classrooms, entering and
leaving rooms, occasional running, groups forming and breaking up. Then things change:

| Change | Desired behaviour |
|---|---|
| New classroom opens; furniture moved; new timetable; lighting changes | adapt |
| Festival or college event | recognise unusual but legitimate activity |
| Someone repeatedly enters a restricted laboratory | do **not** accept it as normal automatically |
| Someone gradually escalates suspicious behaviour | do **not** become habituated |

Without labels a benign drift and a contamination that resembles the tail of normal data can
produce identical unlabelled streams (Theorem 15 of [7]). The problem is therefore one of
**continual learning and memory reliability on an unlabelled stream**, not classification.

### 2.2 Research question
> Can a camera-specific, continually adapting video anomaly detector distinguish legitimate
> environmental adaptation from progressive abnormal-behaviour habituation, and can independent
> memory validation or a safety-constrained learned memory-management policy improve this
> trade-off without sacrificing adaptation speed?

Sub-questions:
* **Q1** Does a contaminated video memory become more likely to accept further anomalies?
* **Q2** Does the *order* of exposure to real same-scene clips matter (habituation attacks)?
* **Q3** What is the attack-cost / adaptation-speed trade-off of fixed rules, and which evidence
  (independent reserve, temporal recurrence, text, sparse labels) improves it?
* **Q4** Can a learned memory manager improve the trade-off over fixed rules, and does accounting
  for long-term consequences (DQN) matter beyond myopic decisions (bandit)?

### 2.3 Non-goals
No face recognition; no identification, attendance tracking or profiling of students; no weapon
recognition; no crime prediction; no unrestricted natural-language surveillance agent; no
multi-camera identity tracking; no autonomous security decisions. An anomaly is a signal for
human review, never proof of misconduct.

### 2.4 Hard parts
| Challenge | Response |
|---|---|
| Research vs 24/7 deployment | experiments are offline replays; deployment engineering is Priority 3 (§13) |
| CLIP sees objects better than motion | motion channel; DINOv2 / object-centric ablations (§7.2) |
| Legitimate change vs malicious behaviour may be indistinguishable | measure the trade-off instead of claiming to solve it; add independent evidence |
| A learned policy may only learn the simulator | held-out scenes, held-out attack types, adaptive attacker, NWPU (§7.6.7) |

---

## 3. System architecture

```
 Module 1  Camera input          recorded videos replayed with timestamps (live IP camera: Priority 3)
     │
 Module 2  Representation        16-frame clips · frozen CLIP ViT-B/16 · RGB + motion · 2×2 regions
     │
 Module 3  Camera memory         ANCHORS (frozen calibration entries) + ADAPTIVE slots
     │                           provenance · age · replacement · undo        [+ separate RESERVE]
 Module 4  Anomaly scoring       regional k-NN cosine distance → clip / frame scores → alarm
     │                           threshold calibrated per camera              [never moved by Module 5]
 Module 5  Admission controller  decides what the memory may learn, one of:
     │                             (a) fixed rules: score gate, scene rule
     │                             (b) IEAC: reserve validation → quarantine → commit / rollback
     │                             (c) learned manager: contextual bandit · DQN (· PPO)
     │                           each optionally with VLM semantic evidence and sparse human checks
 Module 6  Evaluation & operator event log · alerts · memory health · review queue · rollback
```

**Design principles**
1. One memory, one calibration, one reserve **per camera**.
2. **Scoring is independent of admission.** Module 5 can never change the alarm threshold or the
   scores; it only decides what enters memory. (A learned policy that could move the threshold
   would learn to hide its own contamination.)
3. Monitor continuously; **learn only when admitted**.
4. **When uncertain, do not learn:** keep memory unchanged, quarantine, ask a human, keep the
   evidence behind an alert, roll back.
5. The **independent reserve** is a separate set of held-out clean calibration videos, distinct
   from the anchors, never written online, never used to tune what it validates.
6. Report separately **whether the encoder can detect an anomaly at all** and **whether the memory
   learns safely**, so a weak encoder cannot make an unsafe rule look robust.
7. All Module 5 variants share the same memory capacity, scoring, features and protocol.

**Event log** (every decision): camera id, timestamp, clip id, scores, controller inputs, action,
memory version before/after, reason, later rollback. This reconstructs how a detector became
habituated and when a controller stopped it, and it also provides the RL training transitions.

---

## 4. Contributions and claims

**Primary (Priority 1)**
1. **Quantifying self-reinforcement in continual VAD:** ρ, pₐ(ρ), within-run R₀, composition law,
   with CIs per scene, RGB vs motion, memory size and threshold effects. Not claimed: that every
   adaptive system collapses (current evidence: subcritical).
2. **Behaviour-only gradual habituation attacks:** the attacker only orders real same-scene clips;
   the same clips in random vs stepping-stone order; budget and knowledge variants. A controlled
   offline threat model, not proof that a person can manipulate a physical camera this way.
3. **Adaptation-versus-safety trade-off:** T_adapt90, T_attack, ρ_safe at matched adaptation
   speed, and Pareto fronts (§9).

**Extensions (Priority 2)**
4. **Independent memory validation (IEAC):** frozen reserve (WARDEN), quarantine-and-commit,
   reversible memory, sparse human checks; is the combination better than its parts?
5. **Safety-constrained learned memory management:** a contextual bandit and a DQN that choose
   memory actions from compact, label-free state, trained with a contamination constraint, compared
   with fixed rules and IEAC on held-out scenes and held-out attacks.
6. **Semantic validation:** a LATERN-style grounded text gate as admission evidence.
7. **Habituation of VLM context memories.**

**Not claimed:** RL for adaptive memory as such (§5.3); a new CLIP architecture; a general
surveillance system; WARDEN, the urn theory or CDC (these are [7]); detection of all abnormal
behaviour; "first" anything until a systematic literature review supports it.

---

## 5. Related work and positioning

### 5.1 Continual and streaming VAD (verified Nov 2026)
| Work | What it does | Implication |
|---|---|---|
| COMMAND (Liu et al., Neurocomputing 674, 2026) | continual *weakly supervised* VAD, dual memory, Mamba, new anomaly types | continual VAD and dual memory are not novel; no contamination/attack analysis |
| TSMR (Sensors 26(11), 2026; builds on RareCLIP, ICCV 2025) | selective online memory updates for zero-shot *image* AD with a text-prior check | memory-update reliability and text gating exist; ours: behaviour-level contamination of causal video memories |
| StreamVAD (Han et al., Neurocomputing 658, 2025) | low-latency streaming VAD | streaming alone is not novel |
| StreamWatch (venue to confirm) | interpretable streaming surveillance AD | explainable streaming is studied |
| CVAF (Gupta & Prabhat, CAIS 2022) | campus-wide privacy-preserving video analytics | campus surveillance is not a contribution |
| MoniTor (2025), ESOM (2026), CADE (2025) | LLM/streaming memories, continual weakly supervised VAD | memories written online without measuring contamination |
| Mumcu et al. (TMLR 2026) | MLLM activity descriptions for scene-specific VAD | closest prior for our semantic channel |

### 5.2 Papers in `papers/`
| Paper | Use | Gap we address |
|---|---|---|
| Balachandran 2026 [7] | admission kernel, R₀, WARDEN, CDC, Thm 15 | no video; kernel must be re-measured; poison cluster structure left open |
| Su et al., ICLR 2025 [3] | realistic threat-model checklist | image TTA; no memory gates; no targeted habituation |
| MemStream [15] | score-gated FIFO memory | poisoning discussed, not measured |
| Hu, Du & Wang, MM 2025 [11] | per-scene memory + test-time update | non-causal; no adversary |
| MemoVAD [5], HuVAD/UCAL [2], causal SSM VAD [12] | self-updating video systems | only random contamination; slow drift may never trigger updates |
| LATERN [4], VERA [14], CSI-VAD [6] | VLM scoring, gated context memory | gate checks grounding, not normality |
| Parrino et al., KDD 2026 [8] | static often beats streaming | always report the frozen baseline and AUC-PR |
| DriftLens [13], cross-illumination VAD [10] | drift detection; illumination change | drift-aware ablation; appearance-type legitimate change |
| zxVAD [1], MIST [9], IRC-GS [16] | background; probe logic; commit-by-contribution | design ideas |

### 5.3 Learned memory management (limits our RL claim)
| Work | What it does | Implication |
|---|---|---|
| Adaptive Memory Replay (Smith et al., CVPR-W 2024) | phrases sampling of past data for replay as a multi-armed bandit (Boltzmann sampling over clusters) to reduce forgetting | bandit-based memory decisions exist |
| GRASP (Harun et al., CoLLAs 2024, PMLR 274) | a rehearsal *selection policy* (prototypical samples first, harder later); a fixed curriculum, not RL | learned/designed selection policies are a mature topic |
| Memory-R1 (2025), Mem-α (2025), Fine-Mem (2026), MemRL (2026), DeltaMem (2026) | RL-trained memory managers for LLM agents with ADD/UPDATE/DELETE/NOOP-style actions | "RL memory manager" with discrete memory operations is not novel |
| SSGM (2026) | governance of evolving agent memory: stability and safety risks | memory safety is recognised, not studied for continual VAD |

**What is new for us:** none of these studies an *adversarial or contaminating* stream where an
admission changes future admission probabilities (the self-reinforcement loop of §1.2), in video
anomaly detection, under a hard contamination constraint, evaluated by ρ_safe against fixed and
validation-based controllers. The RL work is positioned as a controller inside our measurement
framework, not as the contribution by itself.

### 5.4 Novelty statement (working)
> Existing continual VAD methods study adaptation to evolving environments, and memory-refinement
> and learned memory-management methods study which observations to keep. This research studies
> how sequential contamination and gradual behavioural transitions affect the admission dynamics of
> camera-specific video memories. It provides a controlled framework that measures
> self-reinforcement, habituation time and legitimate adaptation speed, and evaluates independent
> validation, semantic evidence and safety-constrained learned memory management for improving
> this trade-off.

---

## 6. Hypotheses (to pre-register; thresholds are proposals)

| ID | Hypothesis | Falsified if |
|---|---|---|
| **H1** | A score-gated video memory is self-reinforcing: pₐ rises with ρ (within-run slope CI > 0); R₀ reported | CI includes 0 on ≥ half of usable scenes |
| **H1b** | Fitted on the first half of a kernel stream, the composition law predicts last-quarter impurity with median error ≤ 0.1 and π_c within 2× | median error > 0.1 on most scenes, or π_c off by > 2× |
| **H1c** | Motion memories are less self-reinforcing than RGB memories (paired by scene) | R₀(motion) ≥ R₀(RGB) on most scenes |
| **H2** | Visual-only fixed gates have ρ_safe in a narrow band (0.5–2) | any reaches ρ_safe > 3 at matched FPR and T_adapt |
| **H3** | For most anomaly types an alarm-free chain of real same-scene clips exists | chains for < 25% of target types on all datasets |
| **H3b** | At equal anomalous admissions, stepping-stone order succeeds where random order of the same clips does not | random order succeeds as often (paired test) |
| **H4** | WARDEN keeps impurity ≤ δ under the chain attack (a) but does not adapt to a legitimate change (b) | (a) impurity > 2δ; (b) its T_adapt90 within 1.25× of the score gate's |
| **H5** | Quarantine-and-commit raises ρ_safe only by slowing adaptation equally | ρ_safe > 2× with T_adapt < 1.25× |
| **H6** | At matched T_adapt (±10%) IEAC beats its best single component on ρ_safe | not higher (paired test) |
| **H7** | The VLM text gate helps for *kind* targets (walk → climb), not *amount* targets (walk → run) | gain for amount ≥ gain for kind |
| **H8** | ≤ 5 human checks / 1,000 clips raise ρ_safe ≥ 2× with T_adapt ≤ 1.25×; no label-free controller does | no budget ≤ 5/1,000 does, or a label-free one does too |
| **H9** | Once a persistent anomaly fills a LATERN-style context, the VLM's anomaly rate on later occurrences falls ≥ 50% | fall < 20% |
| **H10** | A contextual bandit dominates the best fixed gate on the (T_adapt90, impurity) Pareto front on held-out scenes | its front is not better at any matched T_adapt (paired, held-out scenes) |
| **H11** | DQN beats the bandit, and the gap grows with self-reinforcement (larger in high-π / RGB streams than in low-π / motion streams) | no gap, or gap unrelated to R₀ |
| **H12** | A learned manager's advantage holds on held-out scenes and on NWPU, but shrinks on a held-out attack type (stepping-stone not seen in training) | advantage disappears on held-out scenes (overfitting) / does not shrink on held-out attacks (then report it as robust) |
| **H13** | A grey-box attacker that knows the learned policy reaches T_attack at least 2× faster than against the same policy with an oblivious attacker | speed-up < 1.25× |
| RQ14 (stretch) | Update-rule rankings on real long-term drift (LTD) match the constructed streams | rankings disagree |

A falsified hypothesis is a result and is reported.

---

## 7. Methods

### 7.1 Detector (built)
Per camera and region: k-NN cosine distance (k = 1) to memory; clip score = mean over regions;
frame scores by expansion and Gaussian smoothing (σ = 3); alarm threshold = 99th percentile of
calibration scores (1% nominal false alarms). Strictly causal: score, then update.

### 7.2 Features
Frozen CLIP ViT-B/16: compatible with [7]'s frozen-encoder k-NN setting, gives a shared image–text
space for the semantic gate, cached once. RGB and motion (CLIP on frame differences) channels,
separately and fused; 2×2 regions main, 1×1/3×3 ablations; max aggregation ablation. Planned:
DINOv2 ablation; optional object-centric variant (person crops) so conclusions do not rest on a
weak encoder.

### 7.3 Streams and measurements (built)
* **Five-phase streams:** A normal (calibration) → B legitimate change (behavioural; to add:
  illumination/appearance) → C anomaly burst → D persistent / stepping-stone anomalies → E return
  to old normal. A held-out probe is never admitted. Proposed variant: a "stable new normal"
  segment between B and C.
* **Kernel streams:** constant anomaly rate π ∈ {0.02, 0.05, 0.1, 0.2, 0.3, 0.5}, iid or bursty;
  memory and stream from disjoint videos.
* **Impurity** ρ: anomalous share of valid adaptive slots before each clip.
* **Composition law:** q(ρ) = π·pₐ(ρ) / (π·pₐ(ρ) + (1−π)·pₙ(ρ)); ρ* its stable fixed point;
  within-run slopes; block-bootstrap CIs.
* **T_adapt** (probe false alarms back to 1% + 1 pp), **T_adapt90** (90% of the drop),
  **T_attack** (clips until the target is below the alarm), **ρ_safe** = T_attack / T_adapt90,
  contamination share, lifetime, recovery, retention (definitions in `docs/protocol.md`).

### 7.4 Attacker (threat model after Su et al. [3])
Goal: make one anomalous target clip score below the alarm. Means: choose only which real clip of
the same scene comes next. Knowledge: oblivious; grey-box (knows encoder and controller, sees own
clips' scores; headline); white-box. Constraints: no alarm per step (relaxed: ≤ k); budget
L ∈ {50, 100, 200}; online. Variants: stepping-stone, text-aware, budget-limited, **policy-aware**
(against learned managers: greedy search using queries to the frozen policy); control: random
order of the same clips.

### 7.5 Fixed rules and IEAC
* **Baselines:** frozen; score gate q90 / q95; causal scene-memory rule [11].
* **WARDEN:** conformal p-values of candidates against the separate reserve, e-BH at δ = 0.1 [7];
  CDC recalibration of the threshold under drift.
* **Quarantine-and-commit:** candidates wait in a side pool; committed only after recurrence in
  separate time windows (commit-by-contribution after IRC-GS [16]).
* **Drift-aware gate** (DriftLens-style window distance; ablation).
* **Sparse human checks:** simulated operator, k labels per 1,000 clips on the most uncertain
  candidates.
* **IEAC** (Independent Evidence Admission Controller, working name): screen → reserve validation
  → temporal quarantine → optional VLM / human evidence → commit or reject, with provenance and
  rollback. Each component evaluated alone first.

Recurrence does not prove normality, and the reserve's exchangeability assumption is tested, not
assumed.

### 7.6 Safety-constrained learned memory manager

#### 7.6.1 Formulation
Memory management is a constrained MDP per camera. At each incoming clip t the manager observes a
label-free state s_t, chooses an action a_t, and the stream moves on. It maximises long-run
adaptation reward subject to keeping memory impurity below a budget:

  max_π E[Σ γᵗ r_t]   subject to   E[ρ_t] ≤ δ (and a hard cap enforced by a shield, §7.6.4).

Learned decisions have consequences: admitting one anomaly raises pₐ for the next ones (§1.2).
A myopic controller cannot see this; a sequential one (DQN) can. This is why the RL experiment is
connected to the Phase 3 results and is not a generic "RL for memory" add-on.

#### 7.6.2 State (compact, label-free, ≈ 20–30 numbers)
| Group | Features |
|---|---|
| Candidate | clip score / threshold; mean, max, std of region scores; RGB vs motion score disagreement |
| Independent evidence | conformal p-value vs reserve; distance to anchors; distance to nearest adaptive entry |
| Temporal | similar candidates seen in the last W clips and in separate windows; quarantine size and age |
| Memory health | adaptive fill; mean entry age; admissions in the last W clips; recent alarm rate; drift statistic |
| Optional semantic | LATERN grounding μ and entropy H; caption agreement with memory |

No label or impurity value enters the state: at deployment those are unknown. ρ is used only in
the training reward.

#### 7.6.3 Actions (discrete)
| a | Action |
|---|---|
| 0 | ignore (score only, do not learn) |
| 1 | admit to adaptive memory |
| 2 | quarantine (hold for later evidence) |
| 3 | commit the oldest quarantined candidate(s) that recurred |
| 4 | roll back the last k admissions |
| 5 | request a human label (budgeted; only in H8-style runs) |

The bandit uses actions {0, 1, 2}; DQN uses all.

#### 7.6.4 Reward, constraint and safety shield
Training happens only on simulated episodes built from labelled benchmark clips, so the reward may
use labels that the policy never sees:

  r_t = + w_a · (drop in held-out probe false-alarm rate) − λ · (anomalous entries added)
        − w_m · (missed alarm on an anomalous clip) − c_q · (quarantine cost) − c_h · (human query)

* λ is a Lagrange multiplier updated during training to keep E[ρ] ≤ δ (constrained DQN);
  training for several δ gives a Pareto front instead of one tuned point.
* **Safety shield:** a fixed rule vetoes "admit" when the reserve p-value is below a floor, and
  forces rollback above a hard impurity estimate; the same shield is applied to IEAC so the
  comparison is fair. Reported with and without the shield.
* Anti-gaming checks: a policy that never admits gets no adaptation reward and is caught by
  T_adapt90; the policy cannot touch scores or thresholds (principle 2).

#### 7.6.5 Algorithms
1. **Contextual bandit (first):** LinUCB and linear Thompson sampling over {ignore, admit,
   quarantine}, immediate reward. Tests whether *learned* decisions beat fixed rules at all.
2. **DQN (main):** double DQN with a small MLP (2 × 64), replay buffer, target network, Lagrangian
   constraint; trained on compact state vectors, never on raw video. Discrete actions and a frozen
   encoder make this cheap enough for Kaggle/CPU.
3. **PPO (optional comparison):** only if DQN results are unstable or if time allows.

#### 7.6.6 Training episodes
Episodes are generated from cached features: five-phase streams and kernel streams with randomised
π, burst length, change type (behavioural cluster, illumination), anomaly type and order (random,
bursty; stepping-stone held out, §7.6.7). Thousands of episodes per scene are possible because
features are cached and k-NN over ≤ a few hundred entries per region is cheap.

#### 7.6.7 Splits and generalisation (critical)
* **Scenes:** ShanghaiTech scenes split train / validation / test (e.g. 8 / 2 / 3, rotated as
  cross-validation folds); the test scenes are never seen in training.
* **Datasets:** NWPU Campus used only as a test set for policies trained on ShanghaiTech.
* **Attacks:** train on random and bursty contamination; test also on the stepping-stone attacker
  (held out) and the policy-aware attacker (H13).
* **Seeds:** ≥ 5 training seeds per policy; results reported over seeds.

#### 7.6.8 What RL cannot do
At deployment the policy sees no labels. By Theorem 15 of [7], if a contamination produces the same
unlabelled stream as a benign drift, no label-free policy can tell them apart; RL can only exploit
regularities present in its training episodes (e.g. legitimate changes being persistent and close
to the reserve, anomalies being bursty). An attacker who matches those regularities defeats it.
Hence: held-out attacks, the policy-aware attacker, and the human-query action as the only route
that escapes the impossibility.

### 7.7 VLM components
* **Captions:** a small open VLM that fits a Kaggle T4 (2–8 B parameters, 4-bit; e.g. Qwen-VL or
  InternVL family); one fixed prompt per region describing only visible people, objects and
  actions; captions and CLIP text embeddings cached once; 100 captions hand-checked for errors and
  for attribute mentions.
* **Semantic gate:** LATERN's grounding test [4] (μ > 0.30, H < 0.80) plus agreement with captions
  in memory; used inside IEAC and as optional state for the learned manager (H7).
* **VLM context-memory victim (H9):** a LATERN-style detector with the same frozen VLM (gated
  summary of the last 8 segments, refreshed every 5); persistent-anomaly streams; ShanghaiTech
  subset sized to the GPU budget.

---

## 8. Datasets and experiment matrix

### 8.1 Datasets
| Dataset | Role | State |
|---|---|---|
| ShanghaiTech Campus (13 scenes) | primary; RL training/validation/test scenes | features done |
| NWPU Campus (43 scenes, scene-dependent anomalies) | second dataset; held-out test for learned managers | **request access now** |
| CUHK Avenue | check that findings are not ShanghaiTech-specific | optional |
| Illumination transform / cross-illumination [10] | appearance-type legitimate change (G2) | next Kaggle pass |
| LTD | real long-term drift | stretch |
| Real college footage | external validation only | only with institutional approval, notice, restricted access (§11) |

### 8.2 Main comparison (same capacity, scoring, features, protocol)
| Arm | Memory strategy |
|---|---|
| A | Frozen memory |
| B | Score-based admission gate (q90 / q95) and scene rule |
| C | Rule-based validation: WARDEN, quarantine-and-commit, IEAC |
| D | Contextual-bandit memory manager |
| E | DQN memory manager (with / without shield) |
| (F) | PPO (optional) |
| + | each of C–E with VLM evidence; each with sparse human checks |

### 8.3 Further experiments
| Experiment | Question |
|---|---|
| RGB vs motion vs fused | which representation is more vulnerable? |
| q90 vs q95; memory capacity M; anchor fraction | permissiveness and size vs contamination |
| π sweep | is there a contamination threshold? |
| Random vs stepping-stone order | does order matter? |
| Bandit vs DQN across R₀ | does long-horizon reasoning matter where reinforcement is strong? |
| Held-out scenes / NWPU / held-out attacks | does the learned manager generalise? |
| Policy-aware attacker | is the learned manager brittle? |
| VLM context memory | do VLM detectors habituate? |
| Illumination / appearance change | not every shift is behavioural |
| Efficiency | latency and memory per clip and per camera, incl. policy inference |

---

## 9. Evaluation and statistics

| Dimension | Metrics |
|---|---|
| Detection | frame AUC-ROC (micro, per scene), AUC-PR, post-adaptation detection |
| False alarms | FPR at calibrated threshold |
| Adaptation | T_adapt, T_adapt90 (and the G2-agreed alternative) |
| Habituation resistance | T_attack, ρ_safe |
| Contamination | anomalous share of memory, lifetime |
| Self-reinforcement | pₐ(ρ), within-run R₀, composition-law error |
| Stability | retention, recovery |
| Learned managers | Pareto front (T_adapt90 vs impurity / T_attack) over δ; generalisation gap train → held-out scenes → NWPU; training cost |
| Efficiency | latency, memory use |

Comparisons at matched adaptation speed (T_adapt within ±10%) or as Pareto fronts, never single
cherry-picked points. ≥ 5 seeds; medians with 95% bootstrap CIs; paired Wilcoxon with Holm
correction; per-scene reporting; ablations of every controller component; failed hypotheses
reported; every number from a script and a row in `results/runs.csv`.

---

## 10. Roadmap (November 2026 – December 2027, closure to April 2028)

RL is an extension of the current work, not a prerequisite for G2.

| Phase | When | Work | Gate / deliverable |
|---|---|---|---|
| 1 Foundation | Nov–Dec 2026 | ✔ features; controlled lighting-change experiment with existing methods; request NWPU; freeze protocol; separate reserve videos | **G2**: adaptation measurable on ≥ 5 scenes |
| 2 Self-reinforcement | Jan–Feb 2027 | ✔ first kernel results; per-scene CIs; capacity and threshold sweeps; RGB vs motion (H1–H1c) | statistical analysis |
| 3 Attacks + RL environment | Feb–Mar 2027 | stepping-stone and random-order control; T_attack, ρ_safe; build the episode environment (gym-style) on cached features | trade-off results · **G3**: no realistic chains → measured-robustness study |
| 4 Rule-based defences + bandit | Apr–May 2027 | WARDEN, CDC, quarantine, IEAC, human checks; contextual bandit; VLM captions + semantic gate | arms B–D compared (H4–H8, H10) |
| 5 DQN + VLM context | Jun–Aug 2027 | constrained DQN with shield, held-out scenes and attacks, policy-aware attacker (H11–H13); VLM context-memory habituation (H9) in parallel | arm E · **G4 (Aug 2027)**: if DQN ≤ bandit, report and stop RL scaling |
| 6 Full validation | Aug–Oct 2027 | all arms on ShanghaiTech + NWPU, ≥ 5 seeds, ablations, efficiency | frozen results · **G5** |
| 7 Writing | Oct–Dec 2027 | paper, figures, code release where permitted | submission · **G6** |
| 8 Closure | Jan–Apr 2028 | final report; replay-as-live demo (alerts, memory health, review queue, rollback) | final review |

### 10.1 Next four weeks
| Week | Task | Output |
|---|---|---|
| 1 | Audit results, configs, seeds | reproducibility report |
| 1 | Agree G2 resolution with the guide | pre-registered criterion and stream design |
| 2 | Illumination-change Kaggle pass; rerun streams | multi-scene adaptation results |
| 2 | NWPU access request | access / download |
| 3 | Kernel CIs, capacity and threshold sweeps | kernel analysis |
| 3 | Stepping-stone chain audit; draft RL state/action/reward spec | feasibility report; RL spec in `docs/protocol.md` |
| 4 | Finalise protocol and experiment matrix (incl. RL splits) | signed-off protocol |

---

## 11. Team

| Member | Primary | Secondary |
|---|---|---|
| A | datasets, features, statistics | baselines, reproducibility, NWPU |
| B | memory architecture, fixed rules, IEAC, WARDEN | RL environment and shield |
| C | self-poisoning theory, attacks, kernel estimation | bandit and DQN training, policy-aware attacker |
| D | VLM captions, semantic gate, VLM context memory | paper coordination, figures, literature review |

Everyone understands the whole pipeline; each owns a measurable output; one shared experiment log.

---

## 12. Paper

### 12.1 Title options (final choice with the guide)
1. *Measuring and Mitigating Habituation in Continual Video Anomaly Detection through Independent
   Memory Validation* (if IEAC is the strongest result)
2. *Safe Adaptation or Habituation? Measuring Self-Reinforcement and Learning Safety-Constrained
   Memory Management for Continual Video Anomaly Detection* (if the learned manager wins)

The title is chosen after G5, according to what the results show.

### 12.2 Structure
Introduction · Related work · Problem and threat model · Self-reinforcement measurement ·
Habituation attacks · Controllers (rules, IEAC, learned) · Experiments · Limitations and ethics ·
Conclusion.

### 12.3 Venues
Journals in pattern recognition / video surveillance / trustworthy ML (e.g. Pattern Recognition,
IEEE TCSVT, Neurocomputing); a workshop paper on the measurement results earlier if timing allows.

---

## 13. Ethics, priorities and deployment

### 13.1 Privacy and ethics
1. Main research on public benchmark videos only.
2. No face recognition or student identification.
3. No inference of identity, caste, religion, gender or other personal attributes; VLM prompts
   describe only visible actions and objects and are checked for attribute mentions.
4. Anonymous camera identifiers.
5. Video retention separate from feature-memory retention; raw footage not redistributed.
6. Restricted access to stored footage.
7. Institutional approval and notice before any real college footage.
8. An anomaly score is a signal for human review, never grounds to accuse or punish.
9. A learned manager never acts on people; it only decides what the detector's memory stores.

### 13.2 Priorities
* **Priority 1 (must):** CLIP VAD, camera-specific memory, self-reinforcement measurement,
  habituation experiments, adaptation-vs-safety evaluation.
* **Priority 2 (main extension):** reserve validation, quarantine, reversible memory, sparse human
  checks, IEAC, contextual bandit and DQN managers, VLM gate, VLM context memory.
* **Priority 3 (demo):** live IP camera, continuous inference, per-camera memory, alerts,
  memory-health dashboard, review queue, rollback; plus stream-disconnection recovery, bounded
  latency, memory limits, fault handling. Engineering, not research claims; started only after
  Priorities 1–2 are validated.

---

## 14. Decisions needed from the guide

1. **G2:** (a) controlled illumination/appearance change (recommended); (b) NWPU as main stream
   dataset; (c) T_adapt50 next to T_adapt90, only if justified before testing. The criterion is
   not relaxed to obtain a pass.
2. Leave-one-video-out calibration of gate thresholds (bug fix): accept?
3. Same T_adapt finish line (1% + 1 pp) for all controllers: accept?
4. Reserve = separate held-out calibration videos, not the anchors: accept?
5. Five-phase protocol as implemented, or add the "stable new normal" segment?
6. VLM: GPU budget and model size.
7. RL scope: bandit + DQN as Priority 2, PPO optional, G4 stopping rule; scene split for training.
8. Agree and date the hypothesis thresholds of §6.

---

## 15. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Too few usable stream scenes (G2) | illumination-change streams; NWPU; report scene usability as a finding |
| No attack chains (H3 falsified) | measured-robustness paper plus kernel results |
| Composition law fails on NWPU | report scope; saturating pₐ(ρ) model |
| WARDEN adapts too slowly | that is H4b; report the cost |
| IEAC slows adaptation too much | report the trade-off (H6 falsified) |
| RL overfits training scenes / simulator | scene-held-out CV, NWPU test, shield; report generalisation gap |
| DQN unstable or no better than bandit | report (H11 falsified); bandit remains the learned arm; PPO only if time |
| Learned policy brittle to policy-aware attacker | that is H13; report as a limitation with the shield as mitigation |
| Reward hacking (never admit / always admit) | adaptation term + constraint + T_adapt90 check |
| VLM too slow / large | smaller model, caption only clips used in streams, fewer scenes |
| CLIP blind to motion | motion channel; object-centric variant |
| Dataset access delays | continue on ShanghaiTech; two datasets are the minimum for the paper |
| Similar paper appears | monitor arXiv; cite, position, submit |
| Exams and coursework | weekly results check; move work packages, never skip exit checks |

---

## References

**Papers in `papers/`**
1. A. Aich, K.-C. Peng, A. K. Roy-Chowdhury. Cross-domain video anomaly detection without target domain adaptation (zxVAD). arXiv:2212.07010, 2022.
2. A. Danesh Pazho et al. Towards adaptive human-centric video anomaly detection (HuVAD/UCAL). arXiv:2408.14329, 2025.
3. Y. Su et al. On the adversarial risk of test time adaptation: realistic test-time data poisoning. ICLR 2025.
4. M. Piehl, M. Ye. LATERN: test-time context-aware explainable video anomaly detection. arXiv:2605.15054, 2026.
5. G. Li et al. MemoVAD: resource-efficient VAD via dynamic semantic memory in edge computing. IJCAI 2026.
6. D. Kim, C. Oh, A. Cavallaro, J. Mo. Context-structured VAD with large vision-language models (CSI-VAD). arXiv:2607.19077, 2026.
7. V. B. Balachandran. Self-poisoning in adaptive out-of-distribution detection: a sharp-threshold theory and certified label-free calibration. arXiv:2607.21673, 2026.
8. M. Parrino et al. In a streaming world, should you stand still? A benchmark of anomaly detection in streams. KDD 2026.
9. W. Jin et al. Execution before memory: diagnosing memory failures in video world models (MIST). arXiv:2610.23542, 2026.
10. D. Zhu et al. Cross-illumination video anomaly detection benchmark. ACM MM 2023.
11. H. Hu, W. Du, B. Wang. Scene-dependent memory assisted inter-frame RGB difference reconstruction for VAD. ACM MM 2025.
12. Y. Kumar. Strictly causal streaming VAD with a state-space core. arXiv:2608.24810, 2026.
13. S. Greco et al. Unsupervised concept drift detection from deep learning representations in real-time (DriftLens). arXiv:2406.17813, 2025.
14. M. Ye, W. Liu, P. He. VERA: explainable VAD via verbalized learning of vision-language models. CVPR 2025.
15. S. Bhatia et al. MemStream: memory-based streaming anomaly detection. WWW 2022.
16. W. Xu et al. Self-evolving scene memory for continual 3D Gaussian splatting (IRC-GS). Pacific Graphics 2026.

**To add to `papers/`**
17. Y. Liu et al. COMMANDing anomalies: continual VAD via dual-memory and temporal Mamba modeling. Neurocomputing 674, 2026.
18. TSMR: TPB-guided selective memory refinement for online zero-shot anomaly detection. Sensors 26(11), 3537, 2026 (builds on RareCLIP, ICCV 2025).
19. L. Han et al. StreamVAD: progressive context integration for multi-temporal-scale VAD. Neurocomputing 658, 2025.
20. StreamWatch: decoupled streaming video understanding for interpretable surveillance anomaly detection (venue to confirm).
21. A. Gupta, P. Prabhat. Resource-efficient, privacy-preserving campus-wide video analytics (CVAF). Complex & Intelligent Systems, 2022.
22. F. Mumcu, M. J. Jones, A. Cherian, Y. Yilmaz. Multimodal LLM activity descriptions for explainable semi-supervised VAD. TMLR 2026.
23. MoniTor: LLMs with instruction for online VAD. arXiv:2510.21449, 2025.
24. Z. Liu et al. ESOM: streaming video anomalies with open-world dynamic definitions. arXiv:2604.07772, 2026.
25. S. Hashimoto et al. CADE: continual weakly-supervised VAD with ensembles. arXiv:2512.06840, 2025.
26. J. S. Smith et al. Adaptive memory replay for continual learning. CVPR Workshops 2024, pp. 3605–3615.
27. M. Y. Harun, J. Gallardo, J. Chen, C. Kanan. GRASP: a rehearsal policy for efficient online continual learning. CoLLAs 2024, PMLR 274:1032–1052.
28. S. Yan et al. Memory-R1: enhancing LLM agents to manage and utilize memories via reinforcement learning. arXiv:2508.19828, 2025.
29. Y. Wang et al. Mem-α: learning memory construction via reinforcement learning. arXiv:2509.25911, 2025.
30. Fine-Mem: fine-grained feedback alignment for long-horizon memory management. arXiv:2601.08435, 2026.
31. Governing evolving memory in LLM agents: the Stability and Safety Governed Memory (SSGM) framework. arXiv:2603.11768, 2026.

**Background (cited, not in folder):** Kloft & Laskov, Security analysis of online centroid anomaly detection, JMLR 2012; Kravchik & Shabtai, Can't boil this frog, arXiv:2002.02741; Doshi & Yilmaz, Continual learning VAD, WACV 2022; Cao et al., NWPU Campus, CVPR 2023; Radford et al., CLIP, ICML 2021; Mnih et al., DQN, Nature 2015; van Hasselt et al., Double DQN, AAAI 2016; Li et al., LinUCB, WWW 2010; Altman, Constrained Markov Decision Processes, 1999; Schulman et al., PPO, arXiv:1707.06347.
