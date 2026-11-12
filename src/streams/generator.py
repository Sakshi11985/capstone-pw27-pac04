"""W2.3: five-phase test streams built from real clips of ONE scene.

    A  normal               old-normal clips (training videos NOT used for the memory)
    B  legitimate change    change="cluster": one k-means cluster of training-normal clips,
                            held out of the initial memory: truly normal, but new to it.
                            change="appearance": held-out training videos re-rendered under
                            an illumination preset (src/preprocess/appearance.py), e.g. dusk;
                            same scene and activities, new lighting (gate G2)
    C  anomaly burst        a short contiguous run of anomalous test clips (one video)
    D  persistent anomaly   normal clips with anomalous clips of one "type" interleaved
                            at rate p (type = the most similar anomalous test videos; a
                            proxy, ShanghaiTech has no anomaly-type labels)
    E  return to old normal old-normal clips again

Design rules (docs/protocol.md):
* Clips are never repeated within a stream except, if unavoidable, anomalous clips in D
  (counted and reported as `d_repeats`).
* The memory is initialised only from training videos that never appear in the stream,
  minus the held-out cluster, so no stream clip has an exact copy in memory.
* Clip ids (uids): training clip = its row in train_clips.csv; test clip = 1_000_000 + row.
* Clusters are fixed per scene (seed 0); each stream seed re-draws the video split,
  the video order, the burst video and the persistent-anomaly type.
* A scene whose pools cannot fill the minimum phase sizes is skipped with a reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

TEST_UID_OFFSET = 1_000_000
APPEARANCE_UID_OFFSET = 500_000     # re-lit training clip = 500_000 + its train row


@dataclass
class StreamConfig:
    n_clusters: list = field(default_factory=lambda: [3, 4, 5, 6])   # searched; best cluster wins
    holdout_unit: str = "video"               # video: cluster whole videos | clip
    holdout: str = "novelty"                  # novelty | farthest | nearest | largest
    max_holdout_frac: float = 0.4             # held-out cluster at most this share of the scene
    min_novelty: float = 0.2                  # skip the scene if no cluster is this novel
    stream_video_frac: float = 0.5            # share of old-normal clips kept for the stream
    phase_split: dict = field(default_factory=lambda: {"A": 0.4, "D": 0.2, "E": 0.4})
    max_clips: dict = field(default_factory=lambda: {"A": 400, "B": 400, "D": 300, "E": 400})
    min_clips: dict = field(default_factory=lambda: {"A": 50, "B": 100, "D": 50, "E": 50})
    b_order: str = "gradual"                  # gradual: least-novel held-out videos first | random
    probe: bool = True                        # hold back part of the change as a probe set
    burst_len: int = 10
    persistent_frac: float = 0.2
    persistent_videos: int = 3
    aggregate: str = "mean"                   # region-score aggregate, same as the alarms
    change: str = "cluster"                   # legitimate change: cluster | appearance
    appearance: str = "dim"                   # preset for change="appearance"
    change_video_frac: float = 0.3            # appearance: share of training clips (by video)
                                              # re-lit for phase B + probe

    @classmethod
    def from_dict(cls, d: dict | None):
        d = dict(d or {})
        d.pop("seeds", None)
        return cls(**d)


class SceneTooSmall(Exception):
    pass


@dataclass
class ScenePools:
    scene: str
    feats: dict                  # uid -> [R, D] float32
    labels: dict                 # uid -> clip label (0/1)
    train: pd.DataFrame          # this scene's training clips, with uid and cluster
    test: pd.DataFrame           # this scene's test clips, with uid
    heldout: int                 # held-out cluster id (-1 for an appearance change)
    cluster_info: dict
    mode: str = "cluster"        # cluster | appearance


def _clip_score(Q, M, how: str = "mean"):
    """Region-aggregated 1-NN cosine distance (same aggregate as the alarms); Q [n,R,D], M [m,R,D]."""
    from src.eval.scoring import aggregate, knn_distance
    return aggregate(np.stack([knn_distance(Q[:, r], M[:, r], 1) for r in range(Q.shape[1])], 1), how)


def cluster_novelty(X, vid, held, n_splits=5, false_alarm=0.01, how: str = "mean"):
    """Share of held-out clips that a memory built WITHOUT them would flag, at a threshold
    giving `false_alarm` on other normal videos; mean over `n_splits` random splits of
    the remaining videos into memory / reference. Training-normal clips only."""
    rest = ~held
    out = []
    for sp in range(n_splits):
        vids = np.unique(vid[rest])
        np.random.default_rng(sp).shuffle(vids)
        ref = rest & np.isin(vid, vids[: len(vids) // 2])
        mem = rest & ~ref
        if ref.sum() == 0 or mem.sum() == 0:
            return 0.0
        thr = np.quantile(_clip_score(X[ref], X[mem], how), 1 - false_alarm)
        out.append(float((_clip_score(X[held], X[mem], how) > thr).mean()))
    return float(np.mean(out))


def scene_tables(scene, train_df, train_feats, test_df, test_feats):
    """One scene's clip tables with stream uids, plus uid -> features / label maps.
    train_df/test_df: full clip tables (row index = global position used for uids)."""
    tr = train_df[train_df.scene_id == scene].copy()
    te = test_df[test_df.scene_id == scene].copy()
    tr["uid"] = tr.index.astype(np.int64)
    te["uid"] = te.index.astype(np.int64) + TEST_UID_OFFSET
    feats, labels = {}, {}
    for df, fb in ((tr, train_feats), (te, test_feats)):
        for vid, g in df.groupby("video_id", sort=False):
            x = fb[vid]
            for uid, ci, lab in zip(g.uid, g.clip_idx, g.label):
                feats[int(uid)] = x[int(ci)]
                labels[int(uid)] = int(lab)
    return tr, te, feats, labels


def split_stream_videos(df, rng, frac):
    """Split clips BY VIDEO into memory-initialisation clips and stream videos (about
    `frac` of the clips), keeping at least one video for the memory. Consumes one
    rng.shuffle. Returns (init_df, stream_video_ids)."""
    vids = np.array(df.video_id.unique(), dtype=object)
    rng.shuffle(vids)
    per_vid = df.groupby("video_id").size()
    target = frac * len(df)
    stream_vids, n = [], 0
    for v in vids:
        if n >= target:
            break
        stream_vids.append(v)
        n += per_vid[v]
    if len(stream_vids) == len(vids):
        stream_vids = stream_vids[:-1]
    return df[~df.video_id.isin(stream_vids)], stream_vids


def similar_anomaly_videos(feats, anom, candidates, pick, k):
    """The k anomalous videos among `candidates` whose mean feature is most similar
    (cosine) to that of `pick`: a proxy for an anomaly "type" (no type labels)."""
    cent = {v: np.mean([feats[int(u)].reshape(-1) for u in anom[anom.video_id == v].uid], axis=0)
            for v in candidates}
    c0 = cent[pick] / np.linalg.norm(cent[pick])
    ranked = sorted(candidates, key=lambda v: -float(c0 @ (cent[v] / np.linalg.norm(cent[v]))))
    return ranked[:k]


def build_scene_pools(scene, train_df, train_feats, test_df, test_feats, cfg: StreamConfig) -> ScenePools:
    """train_df/test_df: full clip tables (row index = global position used for uids)."""
    tr, te, feats, labels = scene_tables(scene, train_df, train_feats, test_df, test_feats)

    Xr = np.stack([feats[int(u)] for u in tr.uid])               # [n, R, D]
    X = Xr.reshape(len(Xr), -1)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    vid = tr.video_id.to_numpy()
    uv = tr.video_id.unique()
    ks = cfg.n_clusters if isinstance(cfg.n_clusters, (list, tuple)) else [cfg.n_clusters]
    cands = []                                   # (k, cluster, labels, novelty, size, separation)
    for k in ks:
        if cfg.holdout_unit == "video":
            if len(uv) < k:
                continue
            vm = np.stack([X[vid == v].mean(0) for v in uv])
            vm /= np.linalg.norm(vm, axis=1, keepdims=True)
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(vm)
            vlab = dict(zip(uv, km.labels_))
            lab = np.array([vlab[v] for v in vid])
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
            lab = km.labels_
        sizes = np.bincount(lab, minlength=k)
        Cc = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        dist = 1 - Cc @ Cc.T
        np.fill_diagonal(dist, np.inf)
        for c in range(k):
            if cfg.min_clips["B"] <= sizes[c] <= cfg.max_holdout_frac * len(tr):
                cands.append((k, c, lab, cluster_novelty(Xr, vid, lab == c, how=cfg.aggregate), int(sizes[c]),
                              float(dist[c].min())))
    if not cands:
        raise SceneTooSmall(f"scene {scene}: no cluster with {cfg.min_clips['B']} to "
                            f"{int(cfg.max_holdout_frac * len(tr))} clips")
    key = {"novelty": lambda t: t[3], "farthest": lambda t: t[5],
           "nearest": lambda t: -t[5], "largest": lambda t: t[4]}[cfg.holdout]
    best = max(cands, key=key)
    tr["cluster"] = best[2]
    h = best[1]
    info = {"unit": cfg.holdout_unit, "k": best[0], "heldout": h, "heldout_clips": best[4],
            "novelty": round(best[3], 3), "separation": round(best[5], 4),
            "heldout_videos": int(tr[tr.cluster == h].video_id.nunique()),
            "candidates": [{"k": t[0], "cluster": t[1], "clips": t[4], "novelty": round(t[3], 3)}
                           for t in sorted(cands, key=lambda t: -t[3])[:8]]}
    if not best[3] >= cfg.min_novelty:
        raise SceneTooSmall(f"scene {scene}: no visible legitimate change (best novelty "
                            f"{best[3]:.2f} < {cfg.min_novelty})")
    return ScenePools(scene, feats, labels, tr, te, h, info)


def appearance_novelty(X, Xa, vid, n_splits=5, false_alarm=0.01, how: str = "mean"):
    """Share of RE-LIT clips that a memory of original clips flags, at a threshold giving
    `false_alarm` on original clips of the same reference videos; the reference videos
    are never in the memory, so the only difference measured is the lighting.
    X, Xa: [n, R, D] original / re-lit versions of the same clips; vid: video per clip."""
    out = []
    vids = np.unique(vid)
    for sp in range(n_splits):
        v = vids.copy()
        np.random.default_rng(sp).shuffle(v)
        ref = np.isin(vid, v[: len(v) // 2])
        mem = ~ref
        if ref.sum() == 0 or mem.sum() == 0:
            return 0.0
        thr = np.quantile(_clip_score(X[ref], X[mem], how), 1 - false_alarm)
        out.append(float((_clip_score(Xa[ref], X[mem], how) > thr).mean()))
    return float(np.mean(out))


def build_appearance_pools(scene, train_df, train_feats, test_df, test_feats,
                           app_df, app_feats, cfg: StreamConfig) -> ScenePools:
    """Pools for change="appearance": all clips of the scene plus a re-lit copy of every
    training clip (uid = APPEARANCE_UID_OFFSET + train row, label 0). app_df/app_feats:
    the re-lit training features (same clips as train_df, checked)."""
    if len(train_df) >= APPEARANCE_UID_OFFSET:
        raise ValueError("too many training clips for the appearance uid range")
    tr, te, feats, labels = scene_tables(scene, train_df, train_feats, test_df, test_feats)
    ta = app_df[app_df.scene_id == scene]
    key = ["video_id", "clip_idx"]
    if not tr[key].reset_index(drop=True).equals(ta[key].reset_index(drop=True)):
        raise ValueError(f"scene {scene}: re-lit clips differ from the original training clips")
    for row, (v, ci) in zip(tr.index, zip(tr.video_id, tr.clip_idx)):
        u = APPEARANCE_UID_OFFSET + int(row)
        feats[u] = app_feats[v][int(ci)]
        labels[u] = 0
    tr["cluster"] = 0
    X = np.stack([feats[int(u)] for u in tr.uid])
    Xa = np.stack([feats[APPEARANCE_UID_OFFSET + int(r)] for r in tr.index])
    nov = appearance_novelty(X, Xa, tr.video_id.to_numpy(), how=cfg.aggregate)
    info = {"mode": "appearance", "appearance": cfg.appearance, "novelty": round(nov, 3),
            "change_video_frac": cfg.change_video_frac}
    if not nov >= cfg.min_novelty:
        raise SceneTooSmall(f"scene {scene}: appearance '{cfg.appearance}' not visible to the "
                            f"memory (novelty {nov:.2f} < {cfg.min_novelty})")
    return ScenePools(scene, feats, labels, tr, te, -1, info, mode="appearance")


def _video_blocks(df, rng):
    """Clips grouped by video, videos in random order, temporal order inside a video."""
    vids = np.array(df.video_id.unique(), dtype=object)
    rng.shuffle(vids)
    return pd.concat([df[df.video_id == v].sort_values("clip_idx") for v in vids]) if len(vids) else df


def make_stream(p: ScenePools, seed: int, cfg: StreamConfig) -> dict:
    rng = np.random.default_rng(seed)
    tr, te = p.train, p.test
    if p.mode == "appearance":
        # change videos (re-lit) are drawn per seed; their originals never enter the
        # memory or the stream, so the memory has never seen these clips in any lighting
        rest, change_vids = split_stream_videos(tr, rng, cfg.change_video_frac)
        held = tr[tr.video_id.isin(change_vids)].copy()
        held["uid"] = held.index.astype(np.int64) + APPEARANCE_UID_OFFSET
    else:
        held = tr[tr.cluster == p.heldout]
        rest = tr[tr.cluster != p.heldout]

    # --- split old-normal training videos: memory vs stream (by video, no leakage)
    init, stream_vids = split_stream_videos(rest, rng, cfg.stream_video_frac)
    old = _video_blocks(rest[rest.video_id.isin(stream_vids)], rng)

    # --- phase sizes
    nA = min(int(cfg.phase_split["A"] * len(old)), cfg.max_clips["A"])
    nD = min(int(cfg.phase_split["D"] * len(old)), cfg.max_clips["D"])
    nE = min(len(old) - nA - nD, cfg.max_clips["E"])

    # --- legitimate change: held-out videos ranked by novelty w.r.t. the initial memory
    Xi = np.stack([p.feats[int(u)] for u in init.uid])
    hv = held.video_id.unique()
    nov_v = {}
    for v in hv:
        g = held[held.video_id == v]
        nov_v[v] = float(_clip_score(np.stack([p.feats[int(u)] for u in g.uid]), Xi, cfg.aggregate).mean())
    ranked = sorted(hv, key=lambda v: nov_v[v])          # least novel first
    if cfg.probe:                                         # alternate ranks: feed / probe
        feed_v, probe_v = ranked[0::2], ranked[1::2]
    else:
        feed_v, probe_v = ranked, []
    if cfg.b_order == "random":
        feed_v = list(feed_v)
        rng.shuffle(feed_v)
    B = pd.concat([held[held.video_id == v].sort_values("clip_idx") for v in feed_v]).head(
        cfg.max_clips["B"])
    probe = held[held.video_id.isin(probe_v)]
    sizes = {"A": nA, "B": len(B), "D": nD, "E": nE}
    short = {k: v for k, v in sizes.items() if v < cfg.min_clips[k]}
    if short:
        raise SceneTooSmall(f"scene {p.scene}: phases too short {short} "
                            f"(minimum {cfg.min_clips}); old-normal pool {len(old)} clips")
    A, Dn, E = old.iloc[:nA], old.iloc[nA:nA + nD], old.iloc[nA + nD:nA + nD + nE]

    # --- anomalies: burst video and persistent "type"
    anom = te[te.label == 1]
    avids = anom.video_id.unique()
    if len(avids) < 2:
        raise SceneTooSmall(f"scene {p.scene}: needs >= 2 anomalous test videos")
    burst_vid = avids[rng.integers(len(avids))]
    g = te[te.video_id == burst_vid].sort_values("clip_idx")
    lab = g.label.to_numpy()
    start = int(np.argmax(lab == 1))
    run_end = start
    while run_end < len(lab) and lab[run_end] == 1 and run_end - start < cfg.burst_len:
        run_end += 1
    C = g.iloc[start:run_end]

    others = [v for v in avids if v != burst_vid]
    type_vid = others[rng.integers(len(others))]
    type_vids = similar_anomaly_videos(p.feats, anom, others, type_vid, cfg.persistent_videos)
    pool = pd.concat([anom[anom.video_id == v].sort_values("clip_idx") for v in type_vids])
    n_anom = max(1, int(round(cfg.persistent_frac / (1 - cfg.persistent_frac) * len(Dn))))
    reps = max(0, n_anom - len(pool))
    D_anom = pd.concat([pool] * int(np.ceil(n_anom / len(pool)))).head(n_anom)
    # interleave anomalies evenly through the normal clips of D (small seeded jitter)
    nd = len(Dn) + n_anom
    slots = np.linspace(0, nd - 1, n_anom + 2)[1:-1].round().astype(int)
    slots = np.clip(slots + rng.integers(-1, 2, size=len(slots)), 0, nd - 1)
    slots = np.unique(slots)
    while len(slots) < n_anom:                   # jitter collisions: fill the gaps
        slots = np.unique(np.r_[slots, rng.choice(np.setdiff1d(np.arange(nd), slots))])
    is_anom = np.zeros(nd, bool)
    is_anom[slots] = True
    order, ia, inn = [], 0, 0
    for k in range(nd):
        if is_anom[k]:
            order.append(D_anom.iloc[ia]); ia += 1
        else:
            order.append(Dn.iloc[inn]); inn += 1
    D = pd.DataFrame(order)

    rows = []
    for phase, df in (("A", A), ("B", B), ("C", C), ("D", D), ("E", E)):
        for r in df.itertuples():
            rows.append((phase, int(r.uid), int(p.labels[int(r.uid)]), r.video_id, int(r.clip_idx)))
    stream = pd.DataFrame(rows, columns=["phase", "uid", "label", "video_id", "clip_idx"])
    stream.insert(0, "pos", np.arange(len(stream)))

    used_test = set(C.video_id) | set(type_vids)
    info = {"scene": p.scene, "seed": seed, "phase_sizes": stream.phase.value_counts().reindex(
                list("ABCDE")).fillna(0).astype(int).to_dict(),
            "init_clips": int(len(init)), "stream_videos": [str(v) for v in stream_vids],
            "b_videos": [str(v) for v in feed_v], "probe_videos": [str(v) for v in probe_v],
            "b_order": cfg.b_order,
            "burst_video": str(burst_vid), "type_videos": [str(v) for v in type_vids],
            "d_anomalies": int(n_anom), "d_repeats": int(reps),
            "retention_videos": sorted(str(v) for v in te.video_id.unique() if v not in used_test),
            "clusters": p.cluster_info}
    return {"stream": stream, "init_uids": init.uid.to_numpy(np.int64),
            "probe_uids": probe.uid.to_numpy(np.int64), "info": info}
