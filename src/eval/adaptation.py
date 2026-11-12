"""W2.4: alarm threshold and adaptation metrics on one stream run.

Definitions (docs/protocol.md, plan Section 5):
* Alarm threshold (per scene and stream): the score at which 1% of clean phase-A clips
  raise an alarm, i.e. the 99th percentile of phase-A scores under the INITIAL
  (frozen) memory. The same threshold is used for every update rule on that stream.
* FA rate: fraction of normal clips that raise an alarm.
* T_adapt (main): the legitimate change is split by video into clips fed in phase B and
  a held-back PROBE set never fed. Every `probe_every` clips the probe is scored with
  the current memory. T_adapt = clips after the start of B from which the probe's
  alarm rate stays <= the NOMINAL alarm rate (1%, the rate the threshold was set for)
  + `tol` until the end of B; inf = not reached within B. The same finish line for
  every rule, so a rule is not penalised for having adapted during phase A.
  T_adapt_ownbase uses the rule's own phase-A rate instead (stricter for adaptive
  rules). Mirrors T_attack (steps until the target is accepted).
* T_adapt90 (speed): clips after the start of B from which the probe's EXCESS false
  alarms (above the nominal rate) stay <= 10% of their excess at the start of B.
  Measures how fast a rule adapts independently of the residual it may never remove
  (on ShanghaiTech the score gate removes most excess within ~40 clips, then plateaus
  at 2-7% because probe videos contain content no fed video covers). nan if the
  change was not visible (probe start <= nominal + tol).
* T_adapt_window (secondary): first `window`-clip window of fed B clips whose FA rate
  is <= phase-A rate + tol (sensitive to how novelty is spread over videos).
* Recovery: same rule on the old-normal clips of phase E.
* Contamination share: fraction of adaptive memory slots written by anomalous clips,
  at the end of phase D (mean over regions).
* Contamination lifetime: clips an anomalous entry stayed in memory (to eviction or
  end of stream); median over entries.
* Detection: fraction of anomalous clips that raise an alarm, in C and in D
  (first and last third of D separately shows habituation inside the stream).
* Retention: frame micro AUC of the final memory on test videos not used in the stream.
"""
from __future__ import annotations

import math

import numpy as np


def alarm_threshold(phase_a_scores: np.ndarray, false_alarm: float = 0.01) -> float:
    return float(np.quantile(phase_a_scores, 1.0 - false_alarm))


def first_good_window(alarms: np.ndarray, base: float, window: int = 50, tol: float = 0.01) -> float:
    """Start index of the first length-`window` window whose alarm rate <= base + tol."""
    a = np.asarray(alarms, dtype=np.float64)
    if len(a) < window:
        return math.nan                               # phase too short to measure
    rates = np.convolve(a, np.ones(window) / window, mode="valid")
    ok = np.flatnonzero(rates <= base + tol + 1e-12)
    return float(ok[0]) if len(ok) else math.inf


def probe_alarm_rate(memory, probe_feats: np.ndarray, threshold: float, k: int = 1,
                     how: str = "mean") -> float:
    """Alarm rate of held-back probe clips [n, R, D] under the memory as it is now."""
    from src.eval.scoring import clip_scores
    if len(probe_feats) == 0:
        return math.nan
    s = clip_scores(probe_feats, [m.entries() for m in memory.regions], k, how)
    return float((s > threshold).mean())


def settled_time(checkpoints, base: float, tol: float = 0.01) -> float:
    """checkpoints: [(clips since B start, probe alarm rate)] in order. Earliest
    checkpoint from which every later rate is <= base + tol; inf if the last fails."""
    if not checkpoints:
        return math.nan
    t_ok = math.inf
    for pos, rate in reversed(checkpoints):
        if rate <= base + tol + 1e-12:
            t_ok = pos
        else:
            break
    return float(t_ok)


def fraction_settled_time(checkpoints, base: float, frac: float = 0.9, tol: float = 0.01) -> float:
    """Clips until `frac` of the probe's excess false alarms (over `base`) are removed,
    and stay removed. checkpoints[0] must be the state at the start of the change."""
    if not checkpoints:
        return math.nan
    excess0 = checkpoints[0][1] - base
    if excess0 <= tol:
        return math.nan                           # change not visible to this memory
    return settled_time(checkpoints, base + (1 - frac) * excess0 - tol, tol)


def contamination_share(memory, labels: dict) -> float:
    """Fraction of valid adaptive slots whose source clip is anomalous (mean over regions)."""
    shares = []
    for mem in memory.regions:
        live = mem.valid & ~mem.is_anchor
        if live.sum() == 0:
            shares.append(0.0)
            continue
        src = mem.source[live]
        shares.append(float(np.mean([labels.get(int(u), 0) == 1 for u in src])))
    return float(np.mean(shares))


def contamination_lifetime(events: np.ndarray, labels: dict, t_end: int) -> dict:
    if len(events) == 0:
        return {"n_anom_entries": 0, "median_lifetime": 0.0, "still_in_memory": 0}
    anom = np.array([labels.get(int(u), 0) == 1 for u in events[:, 1]])
    ev = events[anom]
    if len(ev) == 0:
        return {"n_anom_entries": 0, "median_lifetime": 0.0, "still_in_memory": 0}
    t_out = np.where(ev[:, 3] < 0, t_end, ev[:, 3])
    return {"n_anom_entries": int(len(ev)),
            "median_lifetime": float(np.median(t_out - ev[:, 2])),
            "still_in_memory": int((ev[:, 3] < 0).sum())}


def stream_metrics(stream, scores: np.ndarray, admitted: np.ndarray, threshold: float,
                   window: int = 50, tol: float = 0.01, base: float | None = None) -> dict:
    """stream: DataFrame with columns phase, label (same order as scores).
    `base`: the false-alarm level a rule must return to (default: the nominal alarm
    rate used to set the threshold is passed in by the caller; None = the rule's own
    phase-A rate)."""
    ph = stream.phase.to_numpy()
    lab = stream.label.to_numpy()
    alarm = scores > threshold
    normal = lab == 0

    def rate(mask):
        return float(alarm[mask].mean()) if mask.any() else math.nan

    fa_a = rate((ph == "A") & normal)
    ref = fa_a if base is None else base
    b_alarms = alarm[(ph == "B") & normal]
    e_alarms = alarm[(ph == "E") & normal]
    d_idx = np.flatnonzero(ph == "D")
    d_anom = alarm[d_idx][lab[d_idx] == 1]
    third = max(1, len(d_anom) // 3)
    return {
        "threshold": threshold,
        "fa_A": fa_a,
        "fa_B": rate((ph == "B") & normal),
        "fa_D_normal": rate((ph == "D") & normal),
        "fa_E": rate((ph == "E") & normal),
        "T_adapt_window": first_good_window(b_alarms, ref, window, tol),
        "recovery": first_good_window(e_alarms, ref, window, tol),
        "det_C": rate((ph == "C") & ~normal),
        "det_D": float(d_anom.mean()) if len(d_anom) else math.nan,
        "det_D_first3": float(d_anom[:third].mean()) if len(d_anom) else math.nan,
        "det_D_last3": float(d_anom[-third:].mean()) if len(d_anom) else math.nan,
        "admit_rate_normal": float(admitted[normal].mean()),
        "admit_rate_anomalous": float(admitted[~normal].mean()) if (~normal).any() else math.nan,
    }
