"""W2.3 + W2.4 / gate G2: run every update rule on five-phase streams.

    python -m scripts.run_streams --config configs/shanghaitech.yaml

For each scene and stream seed: build the stream, initialise an identical memory for
every rule, set the alarm threshold from the frozen memory on phase A, run each rule
strictly causally, and compute the adaptation metrics.

Writes
  results/streams/<dataset>/<scene>_s<seed>.csv + .json   the streams (clip ids per phase)
  results/stream_metrics_<dataset>.csv                    one row per scene x seed x rule
  results/stream_summary_<dataset>.csv                    median over seeds
  results/figures/fa_over_time_<dataset>_<scene>.png      false alarms through phases A-E
and one row per scene x seed x rule in results/runs.csv.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from src.eval.adaptation import (alarm_threshold, contamination_lifetime,
                                 contamination_share, fraction_settled_time,
                                 probe_alarm_rate, settled_time, stream_metrics)
from src.eval.metrics import frame_auc
from src.eval.scoring import aggregate, clip_scores, labels_dir, load_features, load_frame_labels
from src.memory.memory import SceneMemory
from src.rules.rules import build_rule
from src.streams.generator import (SceneTooSmall, StreamConfig, build_scene_pools,
                                   make_stream)
from src.streams.runner import run_stream
from src.utils.common import REPO_ROOT, load_config, log_run, set_seed, with_channels

DEFAULT_RULES = [{"name": "frozen"}, {"name": "score_gate", "quantile": 0.90},
                 {"name": "score_gate", "quantile": 0.95}, {"name": "scene_memory"}]


def rule_label(spec: dict) -> str:
    extra = "_".join(f"q{v:.2f}" if k == "quantile" else f"{k}{v}"
                     for k, v in spec.items() if k != "name")
    return spec["name"] + (f"_{extra}" if extra else "")


def new_memory(pools, init_uids, mcfg, seed):
    x = np.stack([pools.feats[int(u)] for u in init_uids])
    size = min(mcfg.get("size", 512), len(init_uids))
    mem = SceneMemory(x.shape[1], x.shape[2], size, mcfg.get("anchor_frac", 0.25))
    mem.init_from(x, init_uids, seed)
    return mem, x


def retention_auc(memory, pools, videos, labels, length, k, sigma, how="mean"):
    if not videos:
        return math.nan
    mems = [m.entries() for m in memory.regions]
    te = pools.test
    res = {}
    for v in videos:
        g = te[te.video_id == v].sort_values("clip_idx")
        x = np.stack([pools.feats[int(u)] for u in g.uid])
        res[v] = (clip_scores(x, mems, k, how), labels[v])
    return frame_auc(res, length, sigma)["micro_auc"]


def run(cfg: dict, log: bool = True, out_root=None, make_plots: bool = True,
        suffix: str = "") -> pd.DataFrame:
    set_seed(cfg.get("seed", 0))
    out_root = out_root or REPO_ROOT / "results"
    fdir = labels_dir(cfg)
    how = (cfg.get("score") or {}).get("aggregate", "mean")
    chans = (cfg.get("features") or {}).get("channels") or ["rgb"]
    tag = ("" if chans == ["rgb"] else "_" + "+".join(chans)) + suffix   # keeps runs apart
    scfg = StreamConfig.from_dict(cfg.get("streams"))
    scfg.aggregate = how                                       # novelty uses the alarm aggregate
    seeds = (cfg.get("streams") or {}).get("seeds", [0, 1, 2, 3, 4])
    mcfg = cfg.get("memory", {})
    met = cfg.get("metrics", {})
    k, sigma = met.get("k", 1), met.get("sigma", 3)
    window, tol, fa_q = met.get("window", 50), met.get("tol", 0.01), met.get("false_alarm", 0.01)
    probe_every = met.get("probe_every", 10)
    rules = cfg.get("rules", DEFAULT_RULES)
    length = cfg["clip"]["length"]

    tr_df, tr_feats = load_features(cfg, "train")
    te_df, te_feats = load_features(cfg, "test")
    scenes = cfg.get("scenes") or sorted(te_df.scene_id.unique())
    labels_frame = {v: load_frame_labels(fdir, v) for v in te_feats}
    sdir = out_root / "streams" / (cfg["dataset"] + tag)
    sdir.mkdir(parents=True, exist_ok=True)

    rows, skipped, traces = [], {}, {}
    for scene in scenes:
        try:
            pools = build_scene_pools(scene, tr_df, tr_feats, te_df, te_feats, scfg)
        except SceneTooSmall as e:
            skipped[scene] = str(e); print("SKIP", e); continue
        for seed in seeds:
            try:
                st = make_stream(pools, seed, scfg)
            except SceneTooSmall as e:
                skipped[scene] = str(e); print("SKIP", e); break
            stream, info = st["stream"], st["info"]
            stream.to_csv(sdir / f"{scene}_s{seed}.csv", index=False)
            (sdir / f"{scene}_s{seed}.json").write_text(json.dumps(info, indent=1))
            X = np.stack([pools.feats[int(u)] for u in stream.uid])
            uids = stream.uid.to_numpy()
            end_D = int(stream.index[stream.phase == "D"].max())

            # threshold from the frozen memory on phase A (same for every rule)
            mem0, _ = new_memory(pools, st["init_uids"], mcfg, seed)
            a_idx = np.flatnonzero(stream.phase == "A")
            a_scores = aggregate(np.stack([mem0.score_regions(X[i], k) for i in a_idx]), how)
            thr = alarm_threshold(a_scores, fa_q)
            ret_videos = info["retention_videos"]
            ret0 = retention_auc(mem0, pools, ret_videos, labels_frame, length, k, sigma, how)

            # probe checkpoints: just before B, every `probe_every` clips of B, end of B
            b_idx = np.flatnonzero(stream.phase == "B")
            b0, b1 = int(b_idx[0]), int(b_idx[-1])
            ckpts = sorted(set([b0 - 1] + list(range(b0 - 1 + probe_every, b1, probe_every)) + [b1]))
            P = (np.stack([pools.feats[int(u)] for u in st["probe_uids"]])
                 if len(st["probe_uids"]) else np.zeros((0,) + X.shape[1:], np.float32))
            probe0 = probe_alarm_rate(mem0, P, thr, k, how)

            for spec in rules:
                name = rule_label(spec)
                mem, x_init = new_memory(pools, st["init_uids"], mcfg, seed)
                rule = build_rule(spec["name"], **{a: b for a, b in spec.items() if a != "name"})
                groups = pools.train.set_index("uid").video_id.loc[st["init_uids"]].to_numpy()
                rule.calibrate(mem, x_init, st["init_uids"], groups)   # leave-one-video-out
                snaps = {t: (lambda m: probe_alarm_rate(m, P, thr, k, how)) for t in ckpts}
                snaps[end_D] = lambda m: contamination_share(m, pools.labels)
                out = run_stream(mem, rule, X, uids, k, snapshots=snaps, how=how)
                m = stream_metrics(stream, out["clip_scores"], out["admitted"], thr, window, tol,
                                   base=fa_q)
                curve = [(t - (b0 - 1), out["snapshots"][t]) for t in ckpts]
                m["T_adapt"] = settled_time(curve, fa_q, tol)            # nominal finish line
                m["T_adapt_ownbase"] = settled_time(curve, m["fa_A"], tol)
                m["T_adapt90"] = fraction_settled_time(curve, fa_q, 0.9, tol)
                m["probe_fa_start"] = probe0
                m["probe_fa_endB"] = curve[-1][1]
                m["contamination_D"] = out["snapshots"][end_D]
                m.update(contamination_lifetime(out["events"], pools.labels, len(stream)))
                m["retention"] = retention_auc(mem, pools, ret_videos, labels_frame, length, k,
                                               sigma, how)
                m["retention_init"] = ret0
                row = {"scene": scene, "seed": seed, "rule": name, **info["phase_sizes"], **m}
                rows.append(row)
                if seed == seeds[0]:
                    traces.setdefault(scene, {"phase": stream.phase.to_numpy(),
                                              "label": stream.label.to_numpy(), "rules": {}})
                    traces[scene]["rules"][name] = out["clip_scores"] > thr
                if log:
                    log_run("scripts/run_streams.py", cfg,
                            {"scene": scene, "stream_seed": seed, "rule": name, "k": k,
                             "memory": mcfg, "aggregate": how,
                             "channels": (cfg.get("features") or {}).get("channels") or ["rgb"]},
                            {key: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else
                                   (str(v) if isinstance(v, float) else v))
                             for key, v in m.items()})
            print(f"scene {scene} seed {seed}: " + ", ".join(
                f"{r['rule']} T_adapt={r['T_adapt']} T90={r['T_adapt90']}" for r in rows[-len(rules):]), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / f"stream_metrics_{cfg['dataset']}{tag}.csv", index=False)
    if len(df):
        cols = ["fa_A", "probe_fa_start", "probe_fa_endB", "T_adapt", "T_adapt90", "T_adapt_ownbase", "T_adapt_window", "fa_B", "recovery", "det_C", "det_D_first3", "det_D_last3",
                "contamination_D", "median_lifetime", "retention", "retention_init"]
        summ = df.groupby(["scene", "rule"])[cols].median().reset_index()
        n_inf = df.groupby(["scene", "rule"]).T_adapt.apply(lambda s: int(np.isinf(s).sum()))
        summ["T_adapt_inf_seeds"] = n_inf.to_numpy()
        summ.to_csv(out_root / f"stream_summary_{cfg['dataset']}{tag}.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if skipped:
        print("skipped scenes:", json.dumps(skipped, indent=1))
    (out_root / f"stream_skipped_{cfg['dataset']}{tag}.json").write_text(json.dumps(skipped, indent=1))
    if make_plots and traces:
        plot_traces(traces, out_root / "figures", cfg["dataset"] + tag, window)
    return df


def plot_traces(traces, fig_dir, dataset, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    for scene, tr in traces.items():
        ph, lab = tr["phase"], tr["label"]
        fig, ax = plt.subplots(figsize=(10, 3.2))
        starts = {p: int(np.argmax(ph == p)) for p in "ABCDE"}
        for i, p in enumerate("ABCDE"):
            end = np.flatnonzero(ph == p).max() + 1
            ax.axvspan(starts[p], end, color="0.93" if i % 2 == 0 else "white", zorder=0)
            mid = (starts[p] + end) / 2
            if p == "C":                       # the burst is narrow: label it beside the band
                ax.text(starts[p] - 4, 0.96, "C", fontsize=9, va="top", ha="right", color="0.3")
            else:
                ax.text(mid, 0.96, p, fontsize=9, va="top", ha="center", color="0.3")
        for name, alarms in tr["rules"].items():
            a = np.where(lab == 0, alarms, np.nan).astype(float)    # normal clips only
            s = pd.Series(a).rolling(window, min_periods=10).mean()
            ax.plot(s.to_numpy(), label=name, lw=1.4)
        ax.set_ylim(0, 1)
        ax.set_xlabel("stream position (clips)")
        ax.set_ylabel(f"false-alarm rate\n(normal clips, {window}-clip window)")
        ax.set_title(f"{dataset} scene {scene}, stream seed 0", fontsize=10, loc="left")
        ax.legend(fontsize=8, ncol=4, loc="upper right", frameon=False, bbox_to_anchor=(1, 1.18))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fa_over_time_{dataset}_{scene}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--channels", nargs="+", default=None,
                    help="override features.channels, e.g. rgb diff")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--min-clips", nargs=4, type=int, default=None, metavar=("A", "B", "D", "E"),
                    help="override streams.min_clips (sensitivity runs)")
    ap.add_argument("--suffix", default="", help="appended to result file names")
    a = ap.parse_args()
    cfg = with_channels(load_config(a.config), a.channels)
    if a.min_clips:
        cfg.setdefault("streams", {})["min_clips"] = dict(zip("ABDE", a.min_clips))
    run(cfg, log=not a.no_log, suffix=a.suffix)


if __name__ == "__main__":
    main()
