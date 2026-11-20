"""Step 2: TASK-INDEPENDENT MOVEMENT on Priya's 11-20 windows, DeepLabCut and FaceRhythm, per trial (six-position).

The model is `task_independent_movement` unchanged (functions reused: trial_design, _stats, _predict, RIDGE,
MIN_TRAIN, N_FOLDS, the working gate and the spout-block GroupKFold of `session_bins`): one ridge-regularized least
squares fit per (period, time bin) of each trace on the task regressors
  pre-cue / response: intercept + current spout (5 dummies) + previous spout (5) + previous contact [+ current contact
                      in the response window only];   ITI: intercept + spout of the trial that just ended + its contact,
cross-validated over spout blocks for working trials (other trials predicted by the fit on all working trials); TIM
= Euclidean distance between actual and predicted position, averaged over the trial's bins of the period.
Only the WINDOWS change (s1_extract_frames docstring): pre-cue = ENL (cue - 2 s .. cue, lick-free trials only), response
= cue .. +3.5 s, ITI = docked spout (dock + 0.2 .. next trial_start - 0.05 s, lick buffer [-0.5, +1] s removed);
time bins 100 ms (pre-cue, response) / 250 ms from the docked start (ITI).
Traces:
  DLC  tongue (x, y), jaw (x, y), nose (x, y): the module's own mouth-frame traces (`frame_table`), 40 ms bins -> 3
       parts, residual in px.
  FR   the session's FaceRhythm mesh displacement on the 20 master point PCs (proj ``points_pc``, per-session tracking)
       -> one 20-D 'part' ('face'), residual = Euclidean norm over the 20 PC scores (px).
Output: tim_new_trials.csv.gz (label, trial_id, period, part, tim, n_bins, working).
Usage: WF_REPO=... python s2_tim_new_windows.py <cache_dir> <out_dir> [--jobs 6]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser(os.environ.get("WF_REPO", "~/Github/widefield_pipeline")))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PERIOD = {1: "iti", 2: "precue", 3: "response"}
PNAME = {1: "iti", 2: "precue", 3: "response"}


def trial_meta(label):
    from sklearn.model_selection import GroupKFold
    from wfield_local import block_ids as bi
    from wfield_local import trial_classes as tc
    from wfield_local import task_independent_movement as tim
    S = tc.session_trials(label).sort_values("trial_id").reset_index(drop=True)
    S["hit"] = (S.cls == "contact").astype(float)
    work = tc.in_sel(S.cls.to_numpy()) & tc.before_session_end(S.cls.to_numpy(), S.trial_id.to_numpy(float),
                                                               position=S.position.to_numpy())
    codes = S.position.map({p: i for i, p in enumerate(tim.POSITIONS)}).fillna(-1).astype(int).to_numpy()
    blocks = bi.block_ids(codes)
    wi = np.flatnonzero(work & (blocks >= 0))
    fold = np.full(len(S), -1)
    if len(np.unique(blocks[wi])) >= tim.N_FOLDS:
        for f, (_tr, te) in enumerate(GroupKFold(tim.N_FOLDS).split(wi, groups=blocks[wi])):
            fold[wi[te]] = f
    M = pd.DataFrame({"trial_id": S.trial_id.to_numpy(), "working": work & (blocks >= 0), "fold": fold})
    return S, M, tim.trial_design(S)


def residuals(Y, per, own, tb, ok, M, des, parts):
    """Per frame residual per part (parts = list of column-index lists), CV as `within_predictions`."""
    from wfield_local import task_independent_movement as tim
    keep = ok & (per > 0) & (own >= 0)
    idx = np.flatnonzero(keep)
    wmap, fmap = dict(zip(M.trial_id, M.working)), dict(zip(M.trial_id, M.fold))
    o = own[idx]
    w = np.array([wmap.get(t, False) for t in o]); fo = np.array([fmap.get(t, -1) for t in o])
    X = np.vstack([des[t][PERIOD[p]] if t in des else np.full(tim.NX, np.nan) for t, p in zip(o, per[idx])])
    good = np.isfinite(X).all(1)
    idx, o, w, fo, X = idx[good], o[good], w[good], fo[good], X[good]
    Yk = Y[idx].astype(float)
    pred = np.full_like(Yk, np.nan)
    key = per[idx].astype(np.int64) * 10000 + tb[idx].astype(np.int64)
    for kv in np.unique(key):
        g = np.flatnonzero(key == kv)
        Xg, Yg, wg, fg = X[g], Yk[g], w[g], fo[g]
        if wg.sum() < tim.MIN_TRAIN:
            continue
        st_all = tim._stats(Xg[wg], Yg[wg])
        for f in np.unique(fg[wg & (fg >= 0)]):
            te = wg & (fg == f)
            st = tim._stats(Xg[wg & (fg != f)], Yg[wg & (fg != f)])
            pred[g[te]] = tim._predict(st, Xg[te])
        rest = ~(wg & (fg >= 0))
        if rest.any():
            pred[g[rest]] = tim._predict(st_all, Xg[rest])
    out = {}
    for name, cols in parts.items():
        out[name] = np.sqrt(((Yk[:, cols] - pred[:, cols]) ** 2).sum(1))
    return idx, o, out, w


def job(item):
    label, cache = item
    z = np.load(Path(cache) / f"{label}.npz")
    S, M, des = trial_meta(label)
    lick_pre = {int(t) for c, t, lk in zip(z["win_code"], z["win_own"], z["win_lick"]) if c == 2 and lk}
    rows = []
    for src, parts in (("tp", {"tongue": [0, 1], "jaw": [2, 3], "nose": [4, 5]}), ("pc", {"face": list(range(20))})):
        if f"{src}_per" not in z.files:
            continue
        Y = z["tp"] if src == "tp" else z["pc"].astype(np.float32)
        per, own, tb = z[f"{src}_per"].astype(int), z[f"{src}_own"].astype(int), z[f"{src}_tb"].astype(int)
        ok = np.isfinite(Y).all(1)
        ok &= ~((per == 1) & z[f"{src}_nearlick"])                                   # ITI lick buffer
        ok &= ~((per == 2) & np.isin(own, list(lick_pre)))                           # lick-free ENL only
        idx, o, res, w = residuals(Y, per, own, tb, ok, M, des, parts)
        for name, r in res.items():
            D = pd.DataFrame({"trial_id": o, "period": per[idx], "r": r}).dropna()
            g = D.groupby(["trial_id", "period"]).r.agg(["mean", "size"]).reset_index()
            g["part"] = name
            g["source"] = "DLC" if src == "tp" else "FR"
            rows.append(g)
    R = pd.concat(rows, ignore_index=True).rename(columns={"mean": "tim", "size": "n_bins"})
    R["period"] = R.period.map(PNAME)
    R = R.merge(M[["trial_id", "working"]], on="trial_id", how="left")
    R.insert(0, "label", label)
    return R


def main():
    cache, out = sys.argv[1], Path(sys.argv[2])
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
    f = out / "tim_new_trials.csv.gz"
    if f.exists():
        sys.exit(f"refusing to overwrite {f}")
    labels = sorted(p.stem for p in Path(cache).glob("PS*.npz") if "tmp" not in p.name)
    if "--labels" in sys.argv:
        labels = sys.argv[sys.argv.index("--labels") + 1].split(",")
    from multiprocessing import Pool
    with Pool(jobs) as pool:
        R = []
        for i, r in enumerate(pool.imap_unordered(job, [(lab, cache) for lab in labels])):
            R.append(r)
            print(i + 1, r.label.iat[0], flush=True)
    R = pd.concat(R, ignore_index=True)
    if "--labels" in sys.argv:
        print(R.groupby(["source", "part", "period"]).tim.describe())
        return
    R.to_csv(f, index=False, compression="gzip")


if __name__ == "__main__":
    main()
