"""Phase 3 (H1, H1b): measure the self-poisoning admission kernel on video memories.

Balachandran (arXiv:2607.21673) models an adaptive memory as a reinforced urn. Its
state is the impurity rho = anomalous share of the adaptive memory; its law is the
admission kernel

    q(rho) = P(next admitted entry is anomalous | current impurity rho),

and for an affine kernel q = a + b*rho the slope b acts as a reproduction number R0:
equilibrium rho* = a / (1 - b) if b < 1, complete poisoning if b >= 1 (Cor. 4).

Measuring q inside the five-phase streams would confound it with the phases
(anomalies only occur in C and D), so this module builds dedicated KERNEL STREAMS
with a constant anomaly rate pi, as in the paper:
  * old-normal clips from training videos NOT used to fill the memory (split by video),
  * anomalous clips of one anomaly "type" (the test videos most similar to a seeded
    pick, as in phase D) or of all types, repeated when the pool is small,
  * order "iid" (anomalies at random positions) or "bursty" (contiguous blocks).
Each run records rho of every region just before every clip. The kernel is fitted on
the FIRST half of the run, and its predicted equilibrium is compared with the impurity
actually reached in the LAST quarter: an out-of-sample test of the theory (H1b).

Composition law (found on ShanghaiTech scene 01, 12 Nov 2026). In a FIFO memory the
impurity is the anomalous share of recent admissions, so with stream anomaly rate pi
and admission probabilities p_a(rho) (anomalous clip) and p_n(rho) (normal clip)

    q(rho) = pi*p_a(rho) / (pi*p_a(rho) + (1 - pi)*p_n(rho)),   rho* solves q(rho*) = rho*.

Self-reinforcement means dp_a/drho > 0: anomalies already in memory make the next
anomaly look normal. We therefore also fit p_a and p_n (affine in rho) and predict rho*
from this law. The affine q of [7] is kept for comparison. Pooled slopes are computed
WITHIN runs (each run's mean removed): pooling streams with different pi otherwise
produces a slope near 1 by construction (high-pi streams have both high rho and high q).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.streams.generator import (TEST_UID_OFFSET, ScenePools, _video_blocks, scene_tables,
                                   similar_anomaly_videos, split_stream_videos)


@dataclass
class KernelConfig:
    pis: list = field(default_factory=lambda: [0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    orders: list = field(default_factory=lambda: ["iid", "bursty"])
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    length: int = 800                 # target stream length (capped by supply)
    burst_len: int = 10
    anomaly_pool: str = "type"        # type: 3 most similar anomalous videos | all
    pool_videos: int = 3
    stream_video_frac: float = 0.5    # share of training clips (by video) fed as stream
    min_normal: int = 150             # skip a scene with fewer old-normal stream clips
    min_admissions: int = 100         # fit only with enough admissions ...
    min_anomalous_admissions: int = 10  # ... and enough anomalous ones
    n_bins: int = 12
    knee_level: float = 0.5           # operational critical rate: rho* >= this
    bootstrap: int = 200
    block: int = 50                   # block length (clips) for the block bootstrap

    @classmethod
    def from_dict(cls, d):
        return cls(**dict(d or {}))


def plain_pools(scene, train_df, train_feats, test_df, test_feats) -> ScenePools:
    """All clips of a scene with the stream uid convention, without choosing a
    legitimate change (kernel streams have no phase B)."""
    tr, te, feats, labels = scene_tables(scene, train_df, train_feats, test_df, test_feats)
    tr["cluster"] = 0
    return ScenePools(scene, feats, labels, tr, te, -1, {})


def make_kernel_stream(p: ScenePools, seed: int, pi: float, order: str,
                       cfg: KernelConfig) -> dict:
    """Returns {"stream": DataFrame(uid, label), "init_uids", "info"} or raises ValueError."""
    if not 0 < pi < 1:
        raise ValueError("pi must be in (0, 1)")
    rng = np.random.default_rng(10_000 * seed + int(round(pi * 1000)) + (7 if order == "bursty" else 0))
    tr, te = p.train, p.test
    init, stream_vids = split_stream_videos(tr, rng, cfg.stream_video_frac)
    old = _video_blocks(tr[tr.video_id.isin(stream_vids)], rng)
    n_norm = min(int(round(cfg.length * (1 - pi))), len(old))
    if n_norm < cfg.min_normal:
        raise ValueError(f"scene {p.scene}: only {len(old)} old-normal stream clips")
    n_anom = max(1, int(round(pi / (1 - pi) * n_norm)))

    anom = te[te.label == 1]
    avids = list(anom.video_id.unique())
    if not avids:
        raise ValueError(f"scene {p.scene}: no anomalous test clips")
    if cfg.anomaly_pool == "all":
        pool = anom.sample(frac=1.0, random_state=seed)
        type_vids = avids
    else:
        pick = avids[rng.integers(len(avids))]
        type_vids = similar_anomaly_videos(p.feats, anom, avids, pick, cfg.pool_videos)
        pool = pd.concat([anom[anom.video_id == v].sort_values("clip_idx") for v in type_vids])
    reps = int(math.ceil(n_anom / len(pool)))
    anom_seq = pd.concat([pool] * reps).head(n_anom)

    total = n_norm + n_anom
    is_anom = np.zeros(total, bool)
    if order == "iid":
        is_anom[rng.choice(total, n_anom, replace=False)] = True
    elif order == "bursty":
        n_blocks = int(math.ceil(n_anom / cfg.burst_len))
        sizes = [cfg.burst_len] * (n_anom // cfg.burst_len) + ([n_anom % cfg.burst_len] if n_anom % cfg.burst_len else [])
        # place blocks in random order among the normal clips (gaps of normal clips between)
        cuts = np.sort(rng.choice(n_norm + 1, n_blocks, replace=True))
        ni, out = 0, []
        for b, c in zip(sizes, cuts):
            out += [False] * (c - ni) + [True] * b
            ni = c
        out += [False] * (n_norm - ni)
        is_anom = np.array(out, bool)
    else:
        raise ValueError(f"unknown order {order!r}")
    rows, ia, inn = [], 0, 0
    for a in is_anom:
        if a:
            r = anom_seq.iloc[ia]; ia += 1
        else:
            r = old.iloc[inn]; inn += 1
        rows.append((int(r.uid), int(r.label)))
    stream = pd.DataFrame(rows, columns=["uid", "label"])
    info = {"pi_target": pi, "pi_actual": float(stream.label.mean()), "order": order,
            "n": len(stream), "n_anomalous": int(stream.label.sum()),
            "anomaly_videos": list(map(str, type_vids)), "pool_unique": int(len(pool)),
            "n_init": int(len(init))}
    return {"stream": stream, "init_uids": init.uid.to_numpy(), "info": info}


def impurity_tracker(test_labels: np.ndarray):
    """Returns f(memory) -> rho per region: anomalous share of the valid adaptive slots.
    `test_labels[j]` is the clip label of global test row j (uid = j + TEST_UID_OFFSET)."""
    lab = np.asarray(test_labels).astype(bool)

    def is_anom(src):
        src = np.asarray(src)
        out = np.zeros(len(src), bool)
        t = src >= TEST_UID_OFFSET
        out[t] = lab[src[t] - TEST_UID_OFFSET]
        return out

    def rho(memory):
        r = np.zeros(memory.n_regions)
        for i, m in enumerate(memory.regions):
            live = m.valid & ~m.is_anchor
            r[i] = is_anom(m.source[live]).mean() if live.any() else 0.0
        return r
    return rho, is_anom


@dataclass
class KernelFit:
    n: int
    n_anom: int
    a: float
    b: float
    b_lo: float
    b_hi: float
    r2_binned: float
    bins: list            # [(rho_mid, mean_y, count), ...]


def _ols(x, y):
    vx = x.var()
    if vx < 1e-12:
        return math.nan, math.nan
    b = float(((x - x.mean()) * (y - y.mean())).mean() / vx)
    return float(y.mean() - b * x.mean()), b


def _demean(v, groups):
    out = v.astype(float).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean()
    return out


def fit_affine(x, y, t, cfg: KernelConfig, seed: int = 0, groups=None,
               min_n=None, min_pos=None) -> KernelFit | None:
    """y = a + b*x by least squares on individual events, 95% CI of b by a block
    bootstrap over time blocks of `cfg.block` clips. With `groups` (one id per run) the
    slope is estimated WITHIN groups (fixed effects); a is then the pooled intercept."""
    x, y, t = np.asarray(x, float), np.asarray(y, float), np.asarray(t)
    min_n = cfg.min_admissions if min_n is None else min_n
    min_pos = cfg.min_anomalous_admissions if min_pos is None else min_pos
    if len(x) < min_n or y.sum() < min_pos or x.std() < 1e-6:
        return None
    if groups is not None:
        groups = np.asarray(groups)
        xd, yd = _demean(x, groups), _demean(y, groups)
        b = _ols(xd, yd)[1]
        a = float(y.mean() - b * x.mean())
    else:
        a, b = _ols(x, y)
    if math.isnan(b):
        return None
    rng = np.random.default_rng(seed)
    blocks = t // cfg.block
    ub = np.unique(blocks)
    idx_of = {k: np.flatnonzero(blocks == k) for k in ub}
    bs = []
    for _ in range(cfg.bootstrap):
        ii = np.concatenate([idx_of[k] for k in rng.choice(ub, len(ub), replace=True)])
        if groups is not None:
            bb = _ols(_demean(x[ii], groups[ii]), _demean(y[ii], groups[ii]))[1]
        else:
            bb = _ols(x[ii], y[ii])[1]
        if not math.isnan(bb):
            bs.append(bb)
    lo, hi = (np.quantile(bs, [0.025, 0.975]) if bs else (math.nan, math.nan))
    edges = np.linspace(0, max(x.max(), 1e-6), cfg.n_bins + 1)
    which = np.clip(np.digitize(x, edges) - 1, 0, cfg.n_bins - 1)
    bins = [((edges[k] + edges[k + 1]) / 2, float(y[which == k].mean()), int((which == k).sum()))
            for k in range(cfg.n_bins) if (which == k).any()]
    mids, qs, cs = (np.array(z) for z in zip(*bins))
    pred = a + b * mids
    ss_res = float((cs * (qs - pred) ** 2).sum())
    ss_tot = float((cs * (qs - np.average(qs, weights=cs)) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return KernelFit(len(x), int(y.sum()), a, b, float(lo), float(hi), r2, bins)


def fit_kernel(rho, w, t, cfg: KernelConfig, seed: int = 0, groups=None) -> KernelFit | None:
    """The admission kernel q(rho) of [7]: y = 1 if an admitted entry is anomalous."""
    return fit_affine(rho, w, t, cfg, seed, groups)


def composition_equilibrium(pi: float, pa: tuple, pn: tuple, grid: int = 2001) -> float:
    """Stable fixed point of q(rho) = pi*p_a / (pi*p_a + (1-pi)*p_n), with affine
    p_a = pa[0] + pa[1]*rho and p_n = pn[0] + pn[1]*rho clipped to [0, 1].
    Starts from a clean memory (rho = 0) and returns the first downward crossing."""
    if any(math.isnan(v) for v in (*pa, *pn)):
        return math.nan
    r = np.linspace(0, 1, grid)
    p_a = np.clip(pa[0] + pa[1] * r, 0, 1)
    p_n = np.clip(pn[0] + pn[1] * r, 0, 1)
    den = pi * p_a + (1 - pi) * p_n
    q = np.where(den > 0, pi * p_a / np.maximum(den, 1e-12), 0.0)
    g = q - r
    down = np.flatnonzero((g[:-1] > 0) & (g[1:] <= 0))
    if len(down):
        return float(r[down[0] + 1])
    return 1.0 if g[-1] > 0 else 0.0


def predicted_equilibrium(a: float, b: float) -> float:
    """rho* of the affine urn (Cor. 4): a/(1-b) clipped to [0, 1]; 1 if b >= 1."""
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if b >= 1:
        return 1.0
    return float(min(1.0, max(0.0, a / (1 - b))))


def knee(pis, rhos, level: float) -> float:
    """Smallest pi whose equilibrium impurity reaches `level` (nan if none)."""
    for p_, r in sorted(zip(pis, rhos)):
        if not math.isnan(r) and r >= level:
            return float(p_)
    return math.nan
