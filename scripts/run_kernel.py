"""Phase 3: measure the admission kernel q(rho) and R0 of self-updating video memories
and test whether it predicts where contamination settles (H1, H1b). CPU only.

    python -m scripts.run_kernel --config configs/shanghaitech.yaml
    python -m scripts.run_kernel --config configs/shanghaitech.yaml --scenes 01 04 --quick

Outputs (results/):
  kernel_runs_<dataset><tag>.csv     one row per (scene, rule, pi, order, seed): admissions,
                                     first-half fit (a, b), predicted vs observed equilibrium
  kernel_scene_<dataset><tag>.csv    one row per (scene, rule): kernel pooled over all runs,
                                     R0 with 95% CI, binned R^2 -> H1 verdict per scene
  kernel_knees_<dataset><tag>.csv    one row per (scene, rule, order): observed and predicted
                                     critical rate pi_c (equilibrium >= knee_level) -> H1b
  figures/kernel_q_<dataset><tag>.png, figures/kernel_equilibrium_<dataset><tag>.png
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from scripts.run_streams import new_memory, rule_label
from src.eval.scoring import load_features
from src.rules.rules import build_rule
from src.streams.kernel import (KernelConfig, composition_equilibrium, fit_affine,
                                fit_kernel, impurity_tracker, knee, make_kernel_stream,
                                plain_pools, predicted_equilibrium)
from src.streams.runner import run_stream
from src.utils.common import REPO_ROOT, load_config, log_run, set_seed, with_channels

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4, fixed order


def run(cfg: dict, scenes=None, quick: bool = False, log: bool = True, out_root=None,
        make_plots: bool = True, verbose: bool = True):
    set_seed(cfg.get("seed", 0))
    out_root = out_root or REPO_ROOT / "results"
    out_root.mkdir(parents=True, exist_ok=True)
    kc = KernelConfig.from_dict(cfg.get("kernel"))
    if quick:
        kc.seeds, kc.pis, kc.bootstrap = kc.seeds[:1], [0.05, 0.2, 0.5], 50
    k = cfg.get("metrics", {}).get("k", 1)
    how = (cfg.get("score") or {}).get("aggregate", "mean")
    chans = (cfg.get("features") or {}).get("channels") or ["rgb"]
    tag = "" if chans == ["rgb"] else "_" + "+".join(chans)
    specs = [s for s in cfg["rules"] if s["name"] != "frozen"]   # frozen never admits

    tr_df, tr_feats = load_features(cfg, "train")
    te_df, te_feats = load_features(cfg, "test")
    rho_fn, is_anom = impurity_tracker(te_df.label.to_numpy())
    scenes = scenes or sorted(te_df.scene_id.unique())

    runs, pooled, skipped = [], {}, {}
    t0 = time.time()
    for scene in scenes:
        p = plain_pools(scene, tr_df, tr_feats, te_df, te_feats)
        groups_of = p.train.set_index("uid").video_id
        for seed in kc.seeds:
            for pi in kc.pis:
                for order in kc.orders:
                    try:
                        ks = make_kernel_stream(p, seed, pi, order, kc)
                    except ValueError as e:
                        skipped[scene] = str(e)
                        continue
                    S = ks["stream"]
                    X = np.stack([p.feats[int(u)] for u in S.uid])
                    uids = S.uid.to_numpy()
                    T = len(S)
                    for spec in specs:
                        name = rule_label(spec)
                        mem, x_init = new_memory(p, ks["init_uids"], cfg["memory"], seed)
                        rule = build_rule(spec["name"], **{a: b for a, b in spec.items() if a != "name"})
                        rule.calibrate(mem, x_init, ks["init_uids"],
                                       groups_of.loc[ks["init_uids"]].to_numpy())
                        out = run_stream(mem, rule, X, uids, k, how=how, track=rho_fn)
                        adm, rho = out["admitted"], out["tracked"]
                        lab_t = S.label.to_numpy().astype(bool)
                        tt, rr = np.nonzero(adm)                    # admissions
                        rho_b = rho[tt, rr]
                        w = is_anom(uids[tt]).astype(float)
                        half = tt < T // 2
                        fit_h = fit_kernel(rho_b[half], w[half], tt[half], kc, seed)
                        # arrivals: every (clip, region) with its impurity and admission
                        ta, ra = np.nonzero(np.ones_like(adm))
                        xa, ya, an = rho[ta, ra], adm[ta, ra].astype(float), lab_t[ta]
                        h2 = ta < T // 2
                        fa = fit_affine(xa[h2 & an], ya[h2 & an], ta[h2 & an], kc, seed, min_n=30, min_pos=5)
                        fn = fit_affine(xa[h2 & ~an], ya[h2 & ~an], ta[h2 & ~an], kc, seed, min_n=30, min_pos=5)
                        obs = float(rho[int(0.75 * T):].mean())
                        pi_act = ks["info"]["pi_actual"]
                        row = {"scene": scene, "rule": name, "pi": pi, "order": order,
                               "seed": seed, "T": T, "pi_actual": pi_act,
                               "admissions": int(len(tt)), "anomalous_admissions": int(w.sum()),
                               "admit_rate_anomalous": float(adm[lab_t].mean()) if lab_t.any() else math.nan,
                               "admit_rate_normal": float(adm[~lab_t].mean()),
                               "rho_observed": obs, "rho_final": float(rho[-1].mean())}
                        if fit_h:
                            row.update({"a_half": fit_h.a, "b_half": fit_h.b,
                                        "rho_pred_affine": predicted_equilibrium(fit_h.a, fit_h.b)})
                        if fa:
                            row.update({"pa0": fa.a, "pa_slope": fa.b})
                        if fn:
                            row.update({"pn0": fn.a, "pn_slope": fn.b})
                        if fa and fn:
                            row["rho_pred_composition"] = composition_equilibrium(
                                pi_act, (fa.a, fa.b), (fn.a, fn.b))
                        runs.append(row)
                        rid = len(runs)
                        off = rid * 10 ** 7                         # bootstrap blocks stay within a run
                        for key in ((scene, name), ("ALL", name)):
                            d = pooled.setdefault(key, {"q": [], "pa": []})
                            d["q"].append((rho_b, w, tt + off, np.full(len(w), rid)))
                            d["pa"].append((xa[an], ya[an], ta[an] + off, np.full(int(an.sum()), rid)))
        if verbose:
            print(f"scene {scene} done ({time.time() - t0:.0f}s)", flush=True)

    runs = pd.DataFrame(runs)
    scene_rows, bins_all = [], {}
    for (scene, name), d in sorted(pooled.items()):
        cat = lambda parts, i: np.concatenate([p_[i] for p_ in parts])
        r = {"scene": scene, "rule": name,
             "admissions": int(sum(len(p_[1]) for p_ in d["q"])),
             "anomalous_admissions": int(sum(p_[1].sum() for p_ in d["q"]))}
        fq_across = fit_kernel(cat(d["q"], 0), cat(d["q"], 1), cat(d["q"], 2), kc)
        fq_within = fit_kernel(cat(d["q"], 0), cat(d["q"], 1), cat(d["q"], 2), kc,
                               groups=cat(d["q"], 3))
        fpa = fit_affine(cat(d["pa"], 0), cat(d["pa"], 1), cat(d["pa"], 2), kc,
                         groups=cat(d["pa"], 3), min_n=30, min_pos=5)
        if fq_across:
            r.update({"q_slope_across_runs": fq_across.b, "q_r2_binned": fq_across.r2_binned})
        if fq_within:
            r.update({"R0_within": fq_within.b, "R0_lo": fq_within.b_lo, "R0_hi": fq_within.b_hi})
        if fpa:
            r.update({"pa_slope_within": fpa.b, "pa_lo": fpa.b_lo, "pa_hi": fpa.b_hi,
                      "self_reinforcing": bool(fpa.b_lo > 0)})
        if scene == "ALL":
            bins_all[name] = {"q": fq_across, "pa": fit_affine(cat(d["pa"], 0), cat(d["pa"], 1),
                                                             cat(d["pa"], 2), kc, min_n=30, min_pos=5)}
        scene_rows.append(r)
    scene_df = pd.DataFrame(scene_rows)

    knee_rows = []
    preds = [c for c in ("rho_pred_affine", "rho_pred_composition") if c in runs]
    if len(runs) and preds:
        med = runs.groupby(["scene", "rule", "order", "pi"])[["rho_observed"] + preds] \
            .median().reset_index()
        for (scene, name, order), g in med.groupby(["scene", "rule", "order"]):
            ko = knee(g.pi, g.rho_observed, kc.knee_level)
            row = {"scene": scene, "rule": name, "order": order, "pi_c_observed": ko,
                   "rho_observed_at_max_pi": float(g.sort_values("pi").rho_observed.iloc[-1])}
            for c in preds:
                tagc = c.replace("rho_pred_", "")
                kp = knee(g.pi, g[c], kc.knee_level)
                row[f"pi_c_{tagc}"] = kp
                row[f"error_{tagc}"] = float((g[c] - g.rho_observed).abs().median())
            knee_rows.append(row)
    knee_df = pd.DataFrame(knee_rows)

    ds = cfg["dataset"] + tag
    runs.to_csv(out_root / f"kernel_runs_{ds}.csv", index=False)
    scene_df.to_csv(out_root / f"kernel_scene_{ds}.csv", index=False)
    knee_df.to_csv(out_root / f"kernel_knees_{ds}.csv", index=False)
    (out_root / f"kernel_skipped_{ds}.json").write_text(json.dumps(skipped, indent=1))
    if log:
        for r in scene_rows:
            log_run("scripts/run_kernel.py", cfg, {"scene": r["scene"], "rule": r["rule"],
                    "kernel": {k_: v for k_, v in vars(kc).items()}, "aggregate": how,
                    "channels": chans},
                    {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()
                     if k_ not in ("scene", "rule")})
    if make_plots and len(runs):
        plot(bins_all, runs, out_root / "figures", ds, kc)
    if verbose:
        cols = [c for c in ["scene", "rule", "admissions", "anomalous_admissions",
                            "q_slope_across_runs", "R0_within", "R0_lo", "R0_hi",
                            "pa_slope_within", "pa_lo", "pa_hi", "self_reinforcing"] if c in scene_df]
        print(scene_df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if len(knee_df):
            print(knee_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if skipped:
            print("skipped:", json.dumps(skipped, indent=1))
    return runs, scene_df, knee_df


def _style(ax):
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot(bins_all, runs, fig_dir, ds, kc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    names = [n for n in sorted(runs.rule.unique())          # rules that admitted anomalies
             if n in bins_all and bins_all[n]["q"] is not None]
    col = {n: COLORS[i % len(COLORS)] for i, n in enumerate(names)}   # fixed order, plotted rules only

    # 1) left: kernel q(rho) pooled across all runs; right: p_a(rho), the mechanism
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    fits = [bins_all[n]["q"] for n in names if bins_all[n]["q"]]
    lim = max([max(m for m, _, _ in f.bins) for f in fits] + [0.05]) * 1.1
    ax.plot([0, lim], [0, lim], color="#a3a29c", lw=1, ls="--", zorder=1)
    ax.text(lim * 0.98, lim * 0.80, "q = ρ\n(critical)", ha="right", va="top", color="#52514e", fontsize=8)
    for n in names:
        f = bins_all[n]["q"]
        if not f:
            continue
        m, q, c = (np.array(z) for z in zip(*f.bins))
        ax.scatter(m, q, s=12 + 40 * np.log10(c) / np.log10(c.max() + 1), color=col[n],
                   edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        xs = np.linspace(0, m.max(), 50)
        ax.plot(xs, f.a + f.b * xs, color=col[n], lw=2, zorder=2, label=f"{n}: slope {f.b:.2f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, None)
    ax.set_xlabel("impurity ρ before the admission")
    ax.set_ylabel("anomalous share of admissions, q(ρ)")
    ax.set_title("Kernel pooled across anomaly rates (slope ≈ 1 by construction)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left"); _style(ax)
    ax = axes[1]
    for n in names:
        f = bins_all[n]["pa"]
        if not f:
            continue
        m, q, c = (np.array(z) for z in zip(*f.bins))
        ax.scatter(m, q, s=12 + 40 * np.log10(c) / np.log10(c.max() + 1), color=col[n],
                   edgecolor="#fcfcfb", linewidth=1.5, zorder=3, label=n)
    ax.set_ylim(0, 1)
    ax.set_xlabel("impurity ρ when an anomalous clip arrives")
    ax.set_ylabel("probability it is admitted, pₐ(ρ)")
    ax.set_title("Self-reinforcement: do anomalies in memory admit more?", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / f"kernel_q_{ds}.png", dpi=160)
    plt.close(fig)

    # 2) equilibrium impurity vs anomaly rate: observed vs the two predictions
    preds = [c for c in ("rho_pred_composition", "rho_pred_affine") if c in runs]
    if not preds:
        return
    orders = sorted(runs.order.unique())
    fig, axes = plt.subplots(1, len(orders), figsize=(4.4 * len(orders), 4.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, order in zip(axes, orders):
        g = runs[runs.order == order].groupby(["rule", "pi"])[["rho_observed"] + preds].median()
        for n in names:
            if n not in g.index.get_level_values(0):
                continue
            h = g.loc[n]
            ax.plot(h.index, h.rho_observed, color=col[n], lw=2, marker="o", ms=4, label=f"{n}: observed")
            if "rho_pred_composition" in h:
                ax.plot(h.index, h.rho_pred_composition, color=col[n], lw=1.5, ls="--",
                        label=f"{n}: composition law")
            if "rho_pred_affine" in h:
                ax.plot(h.index, h.rho_pred_affine, color=col[n], lw=1, ls=":",
                        label=f"{n}: affine urn [7]")
        ax.axhline(kc.knee_level, color="#a3a29c", lw=0.8)
        ax.set_xscale("log"); ax.set_ylim(0, 1.05)
        pis = sorted(runs.pi.unique())
        ax.set_xticks(pis); ax.set_xticklabels([f"{p_:g}" for p_ in pis]); ax.minorticks_off()
        ax.set_title(f"{order} anomalies", loc="left", fontsize=10)
        ax.set_xlabel("anomaly rate π of the stream"); _style(ax)
    axes[0].set_ylabel("impurity reached (median)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=7.5, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(fig_dir / f"kernel_equilibrium_{ds}.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--channels", nargs="+", default=None,
                    help="override features.channels, e.g. rgb diff")
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--quick", action="store_true", help="1 seed, 3 anomaly rates")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    run(with_channels(load_config(a.config), a.channels), a.scenes, a.quick, log=not a.no_log)


if __name__ == "__main__":
    main()
