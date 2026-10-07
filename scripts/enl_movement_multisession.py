"""Pre-cue (ENL) position decoding after removing RUNNING + VIDEO movement, across sessions -- by residual AND by
movement-null subspace.

    python -m scripts.enl_movement_multisession                       # every registered regime-B session
    python -m scripts.enl_movement_multisession --labels PS93_0814 PS94_0821 --workers 4

Priya, 2026-10-07: "yes let's try the running + video ENL analysis. Should we try the 'movement null' subspace
analysis?" PS93 0814 (whole-session models) left 0.26-0.30 decodable after DLC + running + video removal, a LOWER
bound (the video may over-remove: a stationary rod occludes / changes contrast by position). DLC added ~nothing in
the lick-free ENL, so the movement model here is running + video only, which needs no whole-session DLC.

ENL-ONLY, so it scales to the cohort: per trial the window is [cue - 2 s, cue] (lick-free: no DAQ lick inside;
spout strobe >= 2 s before the cue so the spout is stationary; ENGAGED = a lick within decode.max_rt_s after the
cue -- the project's ENL decoder population), plus a 0.6 s lead-in for the movement lags. Motion energy is computed
only in those windows (4 cameras, ds 8, binned to imaging frames on the DAQ clock), PCA 30 per camera fitted on the
session's ENL frames; running = DAQ treadmill speed (`behavior_events.session_speed`). Movement design per imaging
frame = video PCs + running at lags 0 / 0.1 / 0.2 s (movement leads activity).

Decoding = the project's ENL decoder shape: all LocaNMF components, 4 x 0.5 s bin means, logistic C 0.5, trial
blocks (position runs) as GroupKFold groups, label-shuffle null with fixed predictions. EVERY movement model is fit
inside each CV fold on the TRAINING trials only:
  raw            no removal
  residual       ridge movement -> dF/F (RidgeCV) on training-trial frames; subtract its prediction on every frame
  null_k         movement-NULL SUBSPACE: the same ridge's predicted activity on training frames -> SVD -> top-k
                 "movement-potent" dimensions of the component space; project them out of ALL activity (train and
                 test) at every frame; k swept. Reported with the share of movement-predicted variance and of total
                 ENL variance those k dimensions carry (2pRAM `lick_subspace_check` convention: benefit, cost, what
                 survives).
  video_only     the movement features themselves as the decoder input
Output: <labcams>/enl_movement_multisession/enl_movement_multisession.csv (one row per session x arm) + a figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FPS_IMG = 31.23
LAGS_S = (0.0, 0.1, 0.2)
K_SWEEP = (1, 2, 3, 5, 8, 12)


def enl_table(s, rv, win_s, max_rt):
    """Clean ENL trials of one registered session: cue_s, pos_name, block (position runs)."""
    from wfield_local import dlc_frames
    from wfield_local import trial_windows as TW
    animal = s["label"].split("_")[0]
    date = Path(s["h5"]).parent.name
    b = TW.trial_bounds(animal, date, rv)
    sid = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob("*_trials.csv"))[-1].name[:-11]
    licks = np.sort(dlc_frames.lick_onsets(animal, date, sid, rv))
    ok = np.isfinite(b.cue_s) & np.isfinite(b.strobe_s) & (b.strobe_s <= b.cue_s - win_s)
    b = b[ok].copy()
    lo, hi = np.searchsorted(licks, b.cue_s - win_s), np.searchsorted(licks, b.cue_s)
    first = np.searchsorted(licks, b.cue_s)
    nxt = np.where(first < len(licks), licks[np.minimum(first, len(licks) - 1)], np.inf)
    b = b[((hi - lo) == 0) & ((nxt - b.cue_s) <= max_rt)].copy()
    b["block"] = (b.pos_name != b.pos_name.shift()).cumsum()
    return animal, date, b.reset_index(drop=True)


def _me_worker(video, tpl, windows, ft, ds):
    """Motion energy (n_imaging_frames_in_windows, n_px) for one camera, frames binned to the imaging clock."""
    import cv2

    from wfield_local import video_motion as VM
    fs, sl, ic = float(tpl["fs_daq"]), float(tpl["slope_daqSample_per_camFrame"]), float(tpl["intercept_daqSample"])
    cap = cv2.VideoCapture(str(video))
    rows = {}
    for t0, t1 in windows:
        f0, f1 = int(round((t0 * fs - ic) / sl)), int(round((t1 * fs - ic) / sl))
        cam_t = (ic + sl * np.arange(f0, f1 + 1)) / fs
        bins = VM.frame_bins(cam_t, ft)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0 - 1)
        prev = None
        for k in range(f0 - 1, f1 + 1):
            ok, im = cap.read()
            if not ok:
                break
            g = VM.downsample(im, ds)
            if prev is not None:
                b = int(bins[k - f0])
                if b >= 0:
                    acc = rows.setdefault(b, [np.zeros(g.size, np.float64), 0])
                    acc[0] += np.abs(g - prev).ravel()
                    acc[1] += 1
            prev = g
    cap.release()
    idx = np.array(sorted(rows))
    return idx, np.array([rows[i][0] / rows[i][1] for i in idx], np.float32)


def lagged(F, frame_idx, lags_frames):
    """Stack F at lags (movement LEADS activity: row t gets F at t - L) using the frame index; missing -> 0."""
    pos = {int(f): i for i, f in enumerate(frame_idx)}
    out = []
    for L in lags_frames:
        j = np.array([pos.get(int(f) - L, -1) for f in frame_idx])
        G = np.zeros_like(F)
        G[j >= 0] = F[j[j >= 0]]
        out.append(G)
    return np.hstack(out)


def features(Yw, frame_idx, trial_frames, n_bins=4):
    """Per trial: mean of each component in n_bins equal sub-windows of its ENL frames -> (n_trial, ncomp*n_bins)."""
    pos = {int(f): i for i, f in enumerate(frame_idx)}
    X = []
    for fr in trial_frames:
        rows = np.array([pos[int(f)] for f in fr if int(f) in pos])
        edges = np.linspace(0, len(rows), n_bins + 1).round().astype(int)
        X.append(np.concatenate([Yw[rows[a:b]].mean(0) for a, b in zip(edges[:-1], edges[1:])]))
    return np.array(X)


def run_session(s, cams, ds, workers, n_perm, log=print):
    from concurrent.futures import ProcessPoolExecutor

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    import scripts.movement_encoding_session as MS
    from wfield_local import config, epochs
    from wfield_local.behavior_events import session_speed
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    dp = config.defaults()["decode"]
    win_s, lead_s = float(dp["precue_post_s"]), 0.6
    animal, date, tr = enl_table(s, rv, win_s, float(dp["max_rt_s"]))
    y, groups = tr.pos_name.to_numpy(), tr.block.to_numpy()
    Y, reg, ft = MS.imaging(s["label"])
    enl = [(c - win_s - lead_s, c) for c in tr.cue_s]
    # imaging frames of each trial's ENL (feature) window and of the movement-model window (with lead-in)
    trial_frames = [np.flatnonzero((ft >= c - win_s) & (ft < c)) for c in tr.cue_s]
    wmask = np.zeros(len(ft), bool)
    for t0, t1 in enl:
        wmask |= (ft >= t0) & (ft < t1)
    frame_idx = np.flatnonzero(wmask)
    vids = {c: sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{c}_*.avi")) for c in cams}
    cams = [c for c in cams if vids[c]]
    tpls = {c: dict(np.load(Path(rv.root("alignment_templates")) / c / animal / f"{date}.npz", allow_pickle=True))
            for c in cams}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {c: ex.submit(_me_worker, str(vids[c][0]), tpls[c], enl, ft, ds) for c in cams}
        me = {c: f.result() for c, f in futs.items()}
    pcs = []
    for c in cams:
        idx, M = me[c]
        full = np.zeros((len(frame_idx), M.shape[1]), np.float32)
        pos = {int(f): i for i, f in enumerate(frame_idx)}
        keep = np.array([int(i) in pos for i in idx])
        full[[pos[int(i)] for i in idx[keep]]] = M[keep]
        pcs.append(PCA(30, random_state=0).fit_transform(full - full.mean(0)))
    h5 = Path(s["h5"])
    ts, sp = session_speed(h5)
    run = np.interp(ft[frame_idx], ts, sp)[:, None]
    Mv = np.hstack(pcs + [run])
    Mv = (Mv - Mv.mean(0)) / np.where(Mv.std(0) > 0, Mv.std(0), 1.0)
    D = lagged(Mv, frame_idx, [int(round(L * FPS_IMG)) for L in LAGS_S])
    Yw = Y[frame_idx]
    trial_of_frame = np.full(len(frame_idx), -1)
    for k, (t0, t1) in enumerate(enl):
        trial_of_frame[(ft[frame_idx] >= t0) & (ft[frame_idx] < t1)] = k
    n_bins = int(dp["bins"]["precue"])
    Xraw = features(Yw, frame_idx, trial_frames, n_bins)
    Xvid = features(Mv, frame_idx, trial_frames, n_bins)
    arms = ["raw", "residual", "video_only"] + [f"null_{k}" for k in K_SWEEP]
    pred = {a: np.empty(len(y), dtype=object) for a in arms}
    share = {k: [] for k in K_SWEEP}

    def fitpred(Xtr, ytr, Xte):
        sc = StandardScaler().fit(Xtr)
        return LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))

    for trn, tst in GroupKFold(min(5, np.unique(groups).size)).split(Xraw, y, groups):
        pred["raw"][tst] = fitpred(Xraw[trn], y[trn], Xraw[tst])
        pred["video_only"][tst] = fitpred(Xvid[trn], y[trn], Xvid[tst])
        fr_tr = np.isin(trial_of_frame, trn)
        rg = RidgeCV(alphas=np.logspace(0, 6, 7)).fit(D[fr_tr], Yw[fr_tr])
        R = Yw - rg.predict(D)
        Xres = features(R, frame_idx, trial_frames, n_bins)
        pred["residual"][tst] = fitpred(Xres[trn], y[trn], Xres[tst])
        Yhat = rg.predict(D[fr_tr]) - rg.predict(D[fr_tr]).mean(0)
        _, sv, vt = np.linalg.svd(Yhat, full_matrices=False)
        Yc = Yw[fr_tr] - Yw[fr_tr].mean(0)
        for k in K_SWEEP:
            V = vt[:k].T                                           # (ncomp, k) movement-potent directions
            Yn = Yw - (Yw @ V) @ V.T
            Xn = features(Yn, frame_idx, trial_frames, n_bins)
            pred[f"null_{k}"][tst] = fitpred(Xn[trn], y[trn], Xn[tst])
            share[k].append(((sv[:k] ** 2).sum() / (sv ** 2).sum(), float(((Yc @ V) ** 2).sum() / (Yc ** 2).sum())))
    rng = np.random.default_rng(0)
    rows = []
    ep = epochs.epoch_of(s["label"])
    day = epochs.days_since_stroke(s["label"])
    for a in arms:
        acc = float((pred[a] == y).mean())
        null = np.array([(pred[a] == rng.permutation(y)).mean() for _ in range(n_perm)])
        r = {"label": s["label"], "animal": animal, "epoch": ep, "day": day, "arm": a, "n_trials": len(y),
             "acc": acc, "null_mean": float(null.mean()), "p": float((np.sum(null >= acc) + 1) / (n_perm + 1))}
        if a.startswith("null_"):
            k = int(a.split("_")[1])
            r["movement_var_removed"] = float(np.mean([m for m, _ in share[k]]))
            r["total_var_removed"] = float(np.mean([t for _, t in share[k]]))
        rows.append(r)
    log(f"{s['label']} ({ep}, day {day}): n={len(y)} " + "  ".join(
        f"{r['arm']} {r['acc']:.2f}" for r in rows if r["arm"] in ("raw", "residual", "video_only", "null_3")))
    return rows


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="*", default=None, help="session labels (default: every regime-B session)")
    ap.add_argument("--cams", nargs="+", default=["cam1", "cam2", "cam3", "cam4"])
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4, help="one process per camera")
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args(argv)
    rv = PathResolver()
    sess = [s for s in config.load_sessions() if s["regime"] == "B" and epochs.epoch_of(s["label"]) is not None]
    if a.labels:
        sess = [s for s in sess if s["label"] in a.labels]
    out = Path(rv.root("labcams")) / "enl_movement_multisession"
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "enl_movement_multisession.csv"
    done = set(pd.read_csv(csv).label) if csv.exists() else set()
    print(f"{len(sess)} sessions ({len(done)} already done)", flush=True)
    for s in sess:
        if s["label"] in done:
            continue
        try:
            rows = run_session(s, a.cams, a.ds, a.workers, a.n_perm)
        except Exception as e:                                           # noqa: BLE001
            print(f"!! {s['label']}: {type(e).__name__}: {e}", flush=True)
            continue
        pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    d = pd.read_csv(csv)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for arm, c in [("raw", "k"), ("residual", "tab:blue"), ("null_3", "tab:orange"), ("video_only", "0.6")]:
        g = d[d.arm == arm]
        for an, ga in g.groupby("animal"):
            axs[0].plot(ga.day, ga.acc, "o-", color=c, alpha=0.6, lw=1, label=f"{arm}" if an == g.animal.iloc[0] else None)
    axs[0].axhline(1 / 6, color="0.5", ls=":")
    axs[0].axvline(0, color="r", lw=0.6)
    axs[0].set_xlabel("days from stroke")
    axs[0].set_ylabel("ENL position decoding accuracy")
    axs[0].legend(fontsize=7)
    nk = d[d.arm.str.startswith("null_")].assign(k=lambda x: x.arm.str[5:].astype(int))
    for ep_, g in nk.groupby("epoch"):
        m = g.groupby("k")[["acc", "movement_var_removed", "total_var_removed"]].mean()
        axs[1].plot(m.index, m.acc, "o-", label=f"{ep_}: decoding")
    axs[1].axhline(1 / 6, color="0.5", ls=":")
    axs[1].set_xlabel("movement-potent dimensions removed (k)")
    axs[1].set_ylabel("ENL decoding accuracy")
    axs[1].legend(fontsize=7)
    fig.suptitle("ENL (2 s pre-cue, lick-free) position decoding: raw, minus running+video (residual), "
                 "movement-null subspace", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "enl_movement_multisession.png", dpi=110)
    print("->", csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
