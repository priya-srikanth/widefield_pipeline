"""Rest-period orofacial / limb cortex spikes AFTER regressing out face movement (FaceRhythm), pooled per epoch with
statistics (cohort1, 2026-10-31).

Priya 10-31, on quiet_area_spikes.py: "can you show pooled per epoch with statistics? is there a better way to quantify
this? we could try regressing fr". The worry it answers: rest (restdock05) excludes only DAQ CONTACT licks, so
non-contact tongue / jaw movements, chewing, grooming and whisking during rest could drive orofacial cortex and be
counted as "spontaneous" spikes, and post-stroke animals make more of those.

SAME TRACES AND REST FRAMES as quiet_area_spikes.py (`session_traces`: area dF/F % minus a 2-s running median,
restdock05 rest, swapped pairs dropped, engaged = before the quit).
FACE REGRESSORS (cam4, the frozen master set `master_pre_v1`, `face_rhythm_outputs`; August sessions only, so pre =
the AUGUST pre baseline, CLAUDE.md): the 12 FaceRhythm factors in ABSOLUTE expression (5 Hz between trials), the 20
dense-point displacement PCs (25 Hz, whole session) and each PC's absolute rate of change (face SPEED), all on the
imaging frames, each minus its own 2-s running median (same high-pass as the cortex), at lags -0.3 .. +0.3 s (0.1 s
steps: cortex may lead or follow the movement).
REGRESSION: ridge per session, fitted on ENGAGED + tail REST frames only, alpha by inner CV; the residual is OUT OF FOLD
(10 contiguous time blocks: fit on the other blocks' rest frames, predict every frame of this block), so the model
cannot remove a spike by memorising it. Frames with a missing regressor (FR gap) are excluded from rest for BOTH the raw
and the residual count, so the two are compared on identical frames.
SPIKES: as quiet_area_spikes (half-width 1.5 frames .. 0.5 s, peak in rest, prominence > 5 x the animal's August-pre
engaged-rest robust SD), detected on the raw trace and on the residual, each with its OWN threshold (residual SD from
the residual).
SCALE-FREE MEASURE: SKEWNESS of the rest fluctuation (raw and residual). Brief positive events skew the distribution;
a change in noise or in fluctuation size alone does not, so it is not moved by the SD drift that inflates PS92's
chronic counts under a fixed threshold.
MOVEMENT AT THE SPIKE: face speed (summed over the 20 PCs) max within +/-0.3 s of each raw spike, as a percentile of
that session's rest face speed; "movement-associated" = above the 90th percentile. CHANCE (`move_null`) = the same
fraction at 3000 random engaged-rest frames (a window max passes the 90th percentile far more often than 10%).
STATISTICS: per-animal epoch means printed first (rule 8); then `mixed_models.lmm_table` (session rows, animal random
effects ladder, August pre) -- `epoch` model (each epoch vs pre) and `time` model (day-1 step + log-day slope); rates on
log10(rate + 0.2 /min). FDR within model x term x measure family. Pooled figure: epoch mean +/- nested animal->session
bootstrap CI (`analysis_kit.boot_ci`), sessions as open markers (shape = animal, colour = epoch), stars in the top band.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_spikes_fr.py [--jobs 6] [--summarise]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))           # quiet_area_spikes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # scripts/ (iti_licks)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # repo root
from quiet_area_spikes import (CAND_SD, DETREND_S, EP, K_SD, MAX_HW_S, MIN_HW_FR, OUT,  # noqa: E402
                               columns, mad, session_traces)
from wfield_local import config  # noqa: E402

#: QS_FR_BASIS=frozen (default): the frozen master set's projections + its frozen point PCA, as above.
#: QS_FR_BASIS=own (Priya 11-01: "we have the FR factors per-session - can we regress on those too?"): each session's OWN
#: rank-12 FaceRhythm fit (`<set>/own/<key>.npz`, `fitsession`), projected here on that session's whole 5 Hz spectrogram
#: exactly as `project` does for the master (rows / animal scale, NNLS, x the FR normaliser D = absolute), plus a PCA of
#: that session's OWN point displacements (20 PCs, 25 Hz) -- so a movement that only exists post-stroke is in the model.
#: Env var, not an argument: spawned workers re-import this module.
BASIS = os.environ.get("QS_FR_BASIS", "frozen")
FR_OUT = OUT / ("fr_regressed" if BASIS == "frozen" else f"fr_regressed_{BASIS}")
LAGS_S = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
N_FOLD, ALPHAS = 10, (1e1, 1e2, 1e3, 1e4, 1e5)
MOVE_PCTL, MOVE_WIN_S = 90.0, 0.3
SHOW = ["orofacial_left", "orofacial_right", "limb_left", "limb_right"]
AREA_SHOW = ["SSp-m_left", "SSp-m_right", "tjM1_left", "tjM1_right", "ALM_left", "ALM_right", "SSp-n_left",
             "SSp-n_right", "SSp-ul_left", "SSp-ul_right", "SSp-ll_left", "SSp-ll_right"]


def _hp(x, w):
    """Fill NaNs by interpolation, subtract the running median (window w frames), re-mask."""
    from scipy.ndimage import median_filter
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return np.full_like(x, np.nan)
    i = np.arange(len(x))
    y = np.interp(i, i[ok], x[ok])
    y = y - median_filter(y, w, mode="nearest")
    y[~ok] = np.nan
    return y


def own_series(label):
    """Session's OWN FR fit projected on its 5 Hz spectrogram (absolute) + its own 20-PC point PCA at 25 Hz, cached in
    FR_OUT/own_series (the spectrogram read is 5-9 GB per session). -> (t5, W5 abs, t25, pcs25)."""
    cache = FR_OUT / "own_series" / f"{label}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["t5"], z["W5"], z["t25"], z["P25"]
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.face_rhythm.factors import cp, files as ff
    sd = fro.factor_dir()
    animal, mmdd = label.split("_")
    keys = [f.stem for f in fro.session_keys(sd, animal, f"2026{mmdd}")]
    if not keys:
        raise FileNotFoundError(f"no FaceRhythm task for {label}")
    _, _, scales, M = ff.load_master(sd)
    s = float(scales[animal])
    pr = ff.load_plan(sd)["params"]
    freqs = M["freqs"].astype(np.float64)
    fr_norm, fr_ex = float(pr.get("vqt_norm", 0.95)), float(pr.get("one_over_f", 0.5))
    od = ff.out_dir(sd)
    frt = ff._sidecar_module("face_rhythm_timing")
    T5, W5, T25, P25 = [], [], [], []
    for key in keys:
        zo = np.load(sd / "own" / f"{key}.npz")
        A, B = zo["A_own"].astype(np.float64), zo["B_own"].astype(np.float64)
        tm = frt.load(od.parent, key)
        t5 = ff.npz_small(od / f"{key}_session.npz", ("t_daq_s",))["t_daq_s"].astype(float)
        if tm is not None:
            t5 = tm["t5"][:len(t5)].astype(float)
        Ss = ff.npz_memmap(od / f"{key}_session.npz", "spec")
        Ws, _, _, Ds = cp.project_rows(cp._scaled(Ss, s), A, B,
                                       row_fn=lambda x: cp.normaliser(x * s, freqs, fr_norm, fr_ex))
        Ws = Ws * Ds[:, None]
        if tm is not None:
            Ws[frt.lossy(tm["drop5"][:len(t5)])] = np.nan
        T5.append(t5[:len(Ws)])
        W5.append(Ws)
        D = ff.npz_memmap(od / f"{key}_points.npz", "disp")             # (n, P, 2) 25 Hz displacements
        t25 = tm["t25"][:len(D)].astype(float) if tm is not None else             ff.npz_small(od / f"{key}_points.npz", ("t_daq_s",))["t_daq_s"].astype(float)
        X = np.asarray(D[::5], np.float32).reshape(-1, D.shape[1] * D.shape[2])
        X = X[np.isfinite(X).all(1)]
        mu = X.mean(0)
        comps = np.linalg.svd(X - mu, full_matrices=False)[2][:20]
        pcs = np.full((len(D), 20), np.nan, np.float32)
        for c0 in range(0, len(D), 20000):
            pcs[c0:c0 + 20000] = (np.asarray(D[c0:c0 + 20000], np.float32).reshape(-1, X.shape[1]) - mu) @ comps.T
        if tm is not None:
            pcs[frt.lossy(tm["drop25"][:len(pcs)])] = np.nan
        T25.append(t25)
        P25.append(pcs)
    out = dict(t5=np.concatenate(T5), W5=np.concatenate(W5), t25=np.concatenate(T25), P25=np.concatenate(P25))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out["t5"], out["W5"], out["t25"], out["P25"]


def face_regressors(label, ft, w):
    """(R frames x k design before lags, speed frames) from the cam4 FaceRhythm projection (BASIS: frozen / own)."""
    from wfield_local import face_rhythm_outputs as fro
    animal, mmdd = label.split("_")
    if BASIS == "own":
        t5, W5, tp, P = own_series(label)
        ok5 = np.isfinite(t5) & np.isfinite(W5).all(1)
        o5 = np.argsort(t5[ok5], kind="stable")
        F = fro.to_frames(t5[ok5][o5], W5[ok5][o5], ft, max_gap_s=0.5)
        tp, P = tp.astype(float), P.astype(float)
    else:
        files = fro.session_keys(fro.factor_dir(), animal, f"2026{mmdd}")
        if not files:
            raise FileNotFoundError(f"no FaceRhythm projection for {label}")
        t, Wf = fro.combined_series(files, "factors", absolute=True)
        F = fro.to_frames(t, Wf, ft, max_gap_s=0.5)
        ts, P = [], []
        for f in files:
            z = np.load(f)
            ts.append(z["points_t_daq_s"].astype(float))
            P.append(z["points_pc"].astype(float))
        tp, P = np.concatenate(ts), np.concatenate(P)
    ok = np.isfinite(tp) & np.isfinite(P).all(1)
    tp, P = tp[ok], P[ok]
    o = np.argsort(tp, kind="stable")
    tp, P = tp[o], P[o]
    dt = np.diff(tp)
    S = np.abs(np.diff(P, axis=0)) / np.where(dt > 0, dt, np.nan)[:, None]       # per-PC speed, 25 Hz
    S[dt > 0.1] = np.nan                                                        # across a gap: not a speed
    ts_ = 0.5 * (tp[1:] + tp[:-1])
    okS = np.isfinite(S).all(1)
    Pf = fro.to_frames(tp, P, ft, max_gap_s=0.2)
    Sf = fro.to_frames(ts_[okS], S[okS], ft, max_gap_s=0.2)
    speed = np.nansum(Sf, 1)
    speed[~np.isfinite(Sf).all(1)] = np.nan
    R = np.column_stack([F, Pf, Sf])
    R = np.column_stack([_hp(R[:, j], w) for j in range(R.shape[1])])
    return R, speed


def lagged(R, fs):
    """Stack R at LAGS_S (row t gets R[t + lag]); NaN where shifted off the end."""
    out = []
    for lag in LAGS_S:
        k = int(round(lag * fs))
        Y = np.full_like(R, np.nan)
        if k >= 0:
            Y[:len(R) - k] = R[k:]
        else:
            Y[-k:] = R[:k]
        out.append(Y)
    return np.hstack(out)


def ridge_oof(X, Y, fit_rows, fold):
    """Out-of-fold ridge prediction of Y (T x c) from X (T x p): per fold, standardise on the training rest rows, pick
    alpha by an inner 5-block CV on those rows, predict every frame of the held-out block that has a full X row."""
    def solve(Xa, Ya, a):
        G = Xa.T @ Xa
        G[np.diag_indices_from(G)] += a
        return np.linalg.solve(G, Xa.T @ Ya)
    okX = np.isfinite(X).all(1)
    pred = np.full(Y.shape, np.nan)
    alphas = []
    for k in range(N_FOLD):
        tr = fit_rows & (fold != k) & okX
        te = (fold == k) & okX
        if tr.sum() < 500 or not te.any():
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        ym = Y[tr].mean(0)
        Xt, Yt = (X[tr] - mu) / sd, Y[tr] - ym
        inner = np.arange(tr.sum()) * 5 // tr.sum()
        best, best_err = ALPHAS[0], np.inf
        for a in ALPHAS:
            err = 0.0
            for j in range(5):
                b = solve(Xt[inner != j], Yt[inner != j], a)
                err += float(((Yt[inner == j] - Xt[inner == j] @ b) ** 2).sum())
            if err < best_err:
                best, best_err = a, err
        alphas.append(best)
        b = solve(Xt, Yt, best)
        pred[te] = ((X[te] - mu) / sd) @ b + ym
    return pred, alphas


def _skew(x):
    x = x[np.isfinite(x)]
    if len(x) < 100:
        return np.nan
    m, s = x.mean(), x.std()
    return float(np.mean(((x - m) / s) ** 3)) if s > 0 else np.nan


def _cands(X, rest, sd, fs, cols):
    from scipy.signal import find_peaks
    C = []
    for i, c in enumerate(cols):
        if c == "cortex" or not np.isfinite(sd[i]) or sd[i] <= 0:
            continue
        x = np.nan_to_num(X[i])
        pk, pr = find_peaks(x, prominence=CAND_SD * sd[i], width=(MIN_HW_FR, MAX_HW_S * fs), rel_height=0.5)
        keep = rest[pk]
        C += [(i, int(p), float(q)) for p, q in zip(pk[keep], pr["prominences"][keep])]
    return np.array(C, float).reshape(-1, 3)


def extract(label):
    """One session -> raw and FR-residual rest candidates, R2, skewness, movement at each raw candidate."""
    f = FR_OUT / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    X, ft, fs, rest, engaged, quit_s = session_traces(label)
    w = int(round(DETREND_S * fs)) | 1
    R, speed = face_regressors(label, ft, w)
    XL = lagged(R, fs)
    okX = np.isfinite(XL).all(1)
    rest = rest & okX & np.isfinite(X).all(0)
    T = X.shape[1]
    fold = np.arange(T) * N_FOLD // T
    cols = columns()
    pred, alphas = ridge_oof(XL, X.T, rest, fold)
    Res = X - np.nan_to_num(pred.T)
    Res[:, ~okX] = 0.0                         # outside the face data: no peaks can be in rest there anyway
    re = rest & engaged
    r2 = 1 - ((X[:, re] - pred.T[:, re]) ** 2).sum(1) / ((X[:, re] - X[:, re].mean(1, keepdims=True)) ** 2).sum(1)
    sd_raw, sd_res = mad(X[:, re]), mad(Res[:, re])
    cr = _cands(X, rest, mad(X[:, rest]), fs, cols)
    cq = _cands(Res, rest, mad(Res[:, rest]), fs, cols)
    # face speed at each raw candidate, as a percentile of this session's rest face speed
    sp_rest = np.sort(speed[rest & np.isfinite(speed)])
    hw = int(round(MOVE_WIN_S * fs))

    def pctl_at(frames):
        out = np.full(len(frames), np.nan)
        for j, p in enumerate(frames):
            v = speed[max(0, p - hw):p + hw + 1]
            if np.isfinite(v).any() and len(sp_rest):
                out[j] = 100.0 * np.searchsorted(sp_rest, np.nanmax(v)) / len(sp_rest)
        return out
    mv = pctl_at(cr[:, 1].astype(int))
    # CHANCE: the same window max at random engaged-rest frames (a +/-0.3 s max exceeds the 90th pct far more than 10%)
    rr = np.flatnonzero(re)
    null = pctl_at(np.random.default_rng(0).choice(rr, min(3000, len(rr)), replace=False)) if len(rr) else np.array([])
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, cand_raw=cr, cand_res=cq, move_pctl=mv, move_null_pctl=null, r2=r2, sd_raw=sd_raw, sd_res=sd_res,
                        skew_raw=np.array([_skew(X[i, re]) for i in range(len(cols))]),
                        skew_res=np.array([_skew(Res[i, re]) for i in range(len(cols))]),
                        engaged_frame=engaged.astype(np.uint8), fs=fs, cols=np.array(cols),
                        n_rest=int(rest.sum()), n_rest_eng=int(re.sum()), alphas=np.array(alphas),
                        face_frac=float(okX.mean()))
    return label


def session_table():
    """One row per session (ENGAGED rest): raw / residual rates per column, R2, skewness, movement-associated
    fraction. Thresholds = 5 x the animal's AUGUST-pre median engaged-rest robust SD (raw and residual separately)."""
    import pandas as pd
    from wfield_local import epochs
    D = {f.stem: dict(np.load(f)) for f in sorted((FR_OUT / "sessions").glob("*.npz"))}
    cols = columns()
    pre = {k: d for k, d in D.items() if epochs.epoch_of(k) == "pre" and not config.early_pre(k)}
    thr = {}
    for an in sorted({k[:4] for k in D}):
        for kind in ("raw", "res"):
            thr[an, kind] = K_SD * np.nanmedian([d[f"sd_{kind}"] for k, d in pre.items() if k.startswith(an)], 0)
    rows = []
    for k, d in D.items():
        ep = epochs.epoch_of(k)
        if ep is None or (ep == "pre" and config.early_pre(k)):
            continue
        an, fs, eng = k[:4], float(d["fs"]), d["engaged_frame"].astype(bool)
        mins = int(d["n_rest_eng"]) / fs / 60
        if mins < 0.25:
            continue
        r = dict(label=k, animal=an, epoch=ep, day=epochs.days_since_stroke(k), rest_min=mins,
                 face_frac=float(d["face_frac"]))
        for kind in ("raw", "res"):
            c = d[f"cand_{kind}"]
            ci, fr, prom = c[:, 0].astype(int), c[:, 1].astype(int), c[:, 2]
            sel = (prom > thr[an, kind][ci]) & eng[np.minimum(fr, len(eng) - 1)]
            for i, cn in enumerate(cols[:-1]):
                m = sel & (ci == i)
                r[f"{cn}|rate_{kind}"] = m.sum() / mins
                if kind == "raw":
                    mv = d["move_pctl"][m]
                    r[f"{cn}|move_frac"] = float(np.mean(mv > MOVE_PCTL)) if np.isfinite(mv).any() else np.nan
        r["move_null"] = float(np.nanmean(d["move_null_pctl"] > MOVE_PCTL))
        for i, cn in enumerate(cols):
            r[f"{cn}|r2"] = float(d["r2"][i])
            r[f"{cn}|skew_raw"] = float(d["skew_raw"][i])
            r[f"{cn}|skew_res"] = float(d["skew_res"][i])
            r[f"{cn}|sd_raw"] = float(d["sd_raw"][i])
        rows.append(r)
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


MEASURES = {"rate_raw": "spikes / min (raw)", "rate_res": "spikes / min (face regressed)",
            "skew_raw": "skewness (raw)", "skew_res": "skewness (face regressed)",
            "r2": "face model CV R2", "move_frac": "fraction of raw spikes\nwith face movement"}


def stats(S):
    import pandas as pd
    from wfield_local import mixed_models as mm
    L = []
    for c in SHOW + ["orofacial", "limb"] + AREA_SHOW:
        for m in MEASURES:
            col = f"{c}|{m}"
            if col not in S:
                continue
            y = np.log10(S[col] + 0.2) if m.startswith("rate") else S[col]
            L.append(S[["label"]].assign(column=c, measure=m, y=y.to_numpy()))
    L = mm.add_time_covariates(pd.concat(L), august_pre=True).dropna(subset=["y"])
    T = mm.lmm_table(L, ["measure", "column"], models=("epoch", "time"), print_animals=True, fdr_by=("measure",))
    T.to_csv(FR_OUT / "lmm_fr_regressed.csv", index=False)
    return T


def figure(S, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.analysis_kit import boot_ci
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    rows = ["rate_raw", "rate_res", "skew_raw", "skew_res", "move_frac", "r2"]
    fig, ax = plt.subplots(len(rows), len(SHOW), figsize=(3.4 * len(SHOW), 2.5 * len(rows)), sharey="row")
    rng = np.random.default_rng(0)
    for i, m in enumerate(rows):
        for j, c in enumerate(SHOW):
            a = ax[i, j]
            col = f"{c}|{m}"
            for x, ep in enumerate(EP):
                B = S[S.epoch == ep]
                for an, A in B.groupby("animal"):
                    jit = rng.uniform(-0.18, 0.18, len(A))
                    a.scatter(x + jit, A[col], marker=ANIMAL_MARKER[an], facecolors="none", edgecolors=EPOCH_LINE[ep],
                              s=18, lw=0.8, alpha=0.8)
                ci = boot_ci({an: list(A[col].dropna()) for an, A in B.groupby("animal") if A[col].notna().any()},
                             rng)
                if ci is not None:
                    a.errorbar(x + 0.3, ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="_", color="k",
                               ms=12, capsize=3, lw=1.5, zorder=5)
                if m == "move_frac":                    # chance: random rest frames, same window
                    a.plot([x - 0.3, x + 0.3], [B.move_null.mean()] * 2, color="0.5", ls="--", lw=1)
                if T is not None and x > 0:
                    t = T[(T.model == "epoch") & (T.measure == m) & (T.column == c) & (T.term == f"C(epoch)[T.{ep}]")]
                    if len(t) and t.q.iloc[0] < 0.05:
                        p = t.q.iloc[0]
                        sig_marks.star_top(a, x, "***" if p < 0.001 else "**" if p < 0.01 else "*",
                                           color=EPOCH_LINE[ep])
            a.set_xticks(range(len(EP)), EP, fontsize=8)
            if j == 0:
                a.set_ylabel(MEASURES[m], fontsize=8)
            if i == 0:
                a.set_title(c.replace("_left", " ipsilesional").replace("_right", " contralesional"), fontsize=9)
            if m.startswith("skew"):
                a.axhline(0, color="0.6", lw=0.6)
    handles = [plt.Line2D([], [], marker=mk, ls="", mfc="none", mec="k", label=an) for an, mk in ANIMAL_MARKER.items()]
    ax[0, 0].legend(handles=handles, fontsize=7, loc="upper left")
    sig_marks.reserve_headroom(fig)
    fig.suptitle("cohort1 ENGAGED-rest spikes (restdock05), raw vs after regressing out face movement (FaceRhythm cam4, "
                 "factors + point PCs + speed, lags +-0.3 s, out of fold).\nOpen markers = sessions (shape = animal, "
                 "colour = epoch); black = pooled mean, nested animal->session bootstrap 95% CI; stars = LMM epoch vs "
                 "August pre, FDR", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FR_OUT / "rest_spikes_pooled_epochs.png", dpi=110)
    print("wrote", FR_OUT / "rest_spikes_pooled_epochs.png")


def summarise():
    import pandas as pd
    pd.set_option("display.width", 240)
    S = session_table()
    S.to_csv(FR_OUT / "rest_spikes_fr_sessions.csv", index=False)
    print(f"{len(S)} sessions; face data cover median {S.face_frac.median():.2f} of frames")
    T = stats(S)
    show = T[(T.model == "epoch") & (T.term != "pre_level") & T.column.isin(SHOW + ["orofacial", "limb"])]
    print("\nLMM epoch model (vs August pre), engaged rest; rates on log10(rate + 0.2):")
    print(show[["measure", "column", "term", "coef", "se", "p", "q", "re", "n"]].round(4).to_string(index=False))
    tm = T[(T.model == "time") & (T.term == "post") & T.column.isin(SHOW + ["orofacial", "limb"] + AREA_SHOW)]
    print("\nLMM time model, day-1 step:")
    print(tm[["measure", "column", "coef", "se", "p", "q", "re"]].round(4).to_string(index=False))
    figure(S, T)
    (FR_OUT / "README.txt").write_text(__doc__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not a.summarise:
        from wfield_local.analysis_kit import curated_labels
        from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
        labels = a.only or [k for k in sorted(curated_labels()) if not config.early_pre(k)]
        res, fail = fan_sessions(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
        missing = [k for k, *_ in fail]
        if len(missing) > 3:
            raise SystemExit("too many failures")
    summarise()


if __name__ == "__main__":
    main()
