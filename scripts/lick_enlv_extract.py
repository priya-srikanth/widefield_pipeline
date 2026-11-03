"""ENLv LICKS (ENL-violation: cam4 licks_all context ``pre_cue``, position strobe <= onset < cue; spout stationary at the
trial's position, no cue, no reward) -- per-lick imaging features + the frozen pre-stroke READERS (Priya 10-29: hemispheric
balance and decoder offset on ENLv licks, compared with cued / ITI). NEVER call these licks ITI.

Lick set: <OUT>/enl_licks/enl_licks.csv.gz (scripts/enl_lick_audit.py), every ENLv lick of an imaging session; 3-D side_ml /
reach_mm from pose3d kinematics_all (joined on session x lick_uid, context pre_cue); exact DAQ peak from licks_all.csv.
Signals: exactly iti_licks.extract (Allen areas + ALM / tjM1 ROIs v8 of the adopted hemo-corrected SVD, slow drift removed,
dF/F x1e3), sampled at the lick PEAK + iti_licks.LAGS_D and averaged:
  Xdec   0..0.2 s     (the KEY-row decoder window of iti_lick_order / iti_where_decoder)
  Xpost  0.1..0.45 s  (lick_balance_target_model._wm(0.1, 0.5): the primary balance window)
  Xbase  -1.0..-0.5 s (iti_licks.BASE_RESP)
  cT/iT  pooled ALM + tjM1 + SSp-m + SSp-n + SSs per hemisphere at every LAGS_D (PETH; c = right = contralesional)
ARRIVAL check (Priya 10-29): strobe-aligned traces of every area on LICK-FREE trials (no lick onset of any context from
1 s before the strobe to the cue; cue >= 2 s after the strobe), lags -1..+4 s, samples after the cue masked; per session
mean + count per spout (arr_mean (7, lags, cols), arr_n; index 6 = all spouts) -- also the
template of the arrival correction (lick_enlv_balance_offset.py).

READERS (rule 10, frozen_models.load_or_fit, LOCAL store only, kind ``lick_reader``): the iti_lick_order KEY-row lateral
decoders (split basis, r8; StandardScaler + RidgeCV(logspace(-1, 5, 13)) on each animal's August pre licks of the set,
leave-one-pre-session-out siblings), align ``iti`` and ``trial`` (cued). iti_lick_order fits them in memory every run;
here the identical recipe is frozen and VERIFIED against the published prediction cache (lick_order_preds_areas_splitr8).
ENLv-trained decoder: same recipe on the pre ENLv licks of the primary event set (align ``enlv_on1``), see
lick_enlv_balance_offset.py.
Outputs <OUT>/enlv_licks/sessions/<label>.npz, enlv_licks/reader_check.csv.
Usage: python scripts/lick_enlv_extract.py [--jobs 4]     (extract)
       python scripts/lick_enlv_extract.py --readers       (freeze + verify the ITI / cued readers)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

OUT = il.OUT_W / "enlv_licks"
SESS = OUT / "sessions"
POOL = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]          # lick_recovery_primary.POOL
LAGS_A = np.round(np.arange(-1.0, 4.0001, 0.05), 3)       # arrival-aligned lags
POS6 = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
VERSION = 1


def _wm(lo_, hi_):
    L = il.LAGS_D
    return (L >= lo_ - 1e-9) & (L <= hi_ + 1e-9) if hi_ <= 0 else (L >= lo_ - 1e-9) & (L < hi_ - 1e-9)


W_DEC = (il.LAGS_D >= 0.0) & (il.LAGS_D <= 0.2)
W_POST = _wm(0.1, 0.5)
W_BASE = _wm(*il.BASE_RESP)


def one(item) -> dict:
    """(label, lick DataFrame, trial DataFrame) -> sessions/<label>.npz. Module-level for spawn."""
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    label, L, T = item
    f = SESS / f"{label}.npz"
    if f.exists():
        return {"label": label, "cached": True}
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv",
                    usecols=["context", "daq_on_s", "daq_peak_s"])
    uid = L.lick_uid.to_numpy(int)
    if not (A.context.to_numpy()[uid] == "pre_cue").all():
        raise RuntimeError(f"{label}: lick_uid join to licks_all.csv is not pre_cue")
    if not np.allclose(A.daq_on_s.to_numpy(float)[uid], L.daq_on_s.to_numpy(float), atol=1e-4):
        raise RuntimeError(f"{label}: onset mismatch with the audit table")
    peak = A.daq_peak_s.to_numpy(float)[uid]
    ok = np.isfinite(peak) & (peak + il.LAGS_D[0] > ft[0]) & (peak + il.LAGS_D[-1] < ft[-1])
    X = il._sample(R, ft, np.where(ok, peak, ft[0] + 2), il.LAGS_D)          # (n, lags, cols) float32
    pool = {s: np.sum([X[:, :, cols.index(f"{a}_{h}")] for a in POOL], 0) for s, h in (("c", "right"), ("i", "left"))}
    # ARRIVAL: lick-free trials (no onset of any context in [strobe - 1, cue]), cue >= strobe + 2
    on = np.sort(A.daq_on_s.to_numpy(float)[np.isfinite(A.daq_on_s.to_numpy(float))])
    s, c = T.strobe_s.to_numpy(float), T.cue_s.to_numpy(float)
    n_in = np.searchsorted(on, c, side="right") - np.searchsorted(on, s - 1.0)
    free = np.isfinite(s) & np.isfinite(c) & (c - s >= 2.0) & (n_in == 0) & (s + LAGS_A[0] > ft[0]) & \
        (s + LAGS_A[-1] < ft[-1])
    # per spout (+ index 6 = all spouts): mean trace and trial count per lag; samples at / after the cue masked
    arr_mean = np.full((7, len(LAGS_A), len(cols)), np.nan)
    arr_n = np.zeros((7, len(LAGS_A)), int)
    if free.any():
        Y = il._sample(R, ft, s[free], LAGS_A).astype(np.float64)
        Y[(LAGS_A[None, :] >= (c[free] - s[free])[:, None])] = np.nan
        pf = T.pos.astype(str).to_numpy()[free]
        for k, sel in [*((k, pf == p) for k, p in enumerate(POS6)), (6, np.ones(len(pf), bool))]:
            if sel.any():
                with np.errstate(invalid="ignore"):
                    arr_mean[k] = np.nanmean(Y[sel], 0)
                arr_n[k] = np.isfinite(Y[sel][:, :, 0]).sum(0)
    nan = np.float32(np.nan)
    np.savez_compressed(
        f, version=VERSION, cols=np.array(cols), lick_uid=uid, peak=peak, ok=ok,
        Xdec=np.where(ok[:, None], X[:, W_DEC].mean(1), nan).astype(np.float32),
        Xpost=np.where(ok[:, None], X[:, W_POST].mean(1), nan).astype(np.float32),
        Xbase=np.where(ok[:, None], X[:, W_BASE].mean(1), nan).astype(np.float32),
        cT=np.where(ok[:, None], pool["c"], nan).astype(np.float16),
        iT=np.where(ok[:, None], pool["i"], nan).astype(np.float16),
        arr_mean=arr_mean.astype(np.float32), arr_n=arr_n, arr_trials=int(free.sum()))
    return {"label": label, "n": int(ok.sum()), "n_free_trials": int(free.sum())}


def _signals(label: str):
    """(R frames x cols, ft, cols): the iti_licks.extract area / ROI signals (same code as `one`)."""
    from wfield_local import combined_encoding as ce
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    R = V.astype(np.float64) @ np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    return R * 1e3, ft, cols


# ARRIVAL + LICK REGRESSION (Priya 10-29: separate the overlapping spout-arrival and lick responses for the >= 0 s and
# first-lick sets; NOT a lick-free-trial template, which is thin pre-stroke; NO cue kernel: fit on PRE-CUE data only).
# Per session, every area / ROI column: R(t) = b0 + arrival kernel (strobe, lags ARR_LAGS, raised-cosine basis
# ARR_SPACING) + lick-onset kernel (onsets of EVERY cam4 lick of any context, LICK_LAGS, LICK_SPACING), ridge, fitted on
# the frames from 1.5 s before each strobe up to its cue (frames at / after the cue never enter). Alpha by 5-fold CV
# with folds = whole trials. Split-half stability: the same alpha refit on odd / even trials, r of the two arrival
# kernels per column. The ARRIVAL-REMOVED signal R - arrival part is then sampled at the ENLv lick peaks exactly as in
# `one` (Xdec_c, Xpost_c, Xbase_c, cT_c, iT_c).
ARR_LAGS, ARR_SPACING = (-1.0, 4.0), 0.2
LICK_LAGS, LICK_SPACING = (-0.5, 1.5), 0.1
FIT_PRE = 1.5
ALPHAS = np.logspace(-1, 4, 11)
SESS_REG = OUT / "sessions_reg"
REG_VERSION = 2              # 2: arrival kernel referenced to its pre-strobe level (1 = two test files, overwritten)
LAGS_K = np.round(np.arange(-1.0, 4.0001, 0.05), 3)


def regress(item) -> dict:
    """(label, trial DataFrame) -> sessions_reg/<label>.npz. Module-level for spawn."""
    from sklearn.linear_model import Ridge

    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    label, T = item
    f = SESS_REG / f"{label}.npz"
    if f.exists() and int(np.load(f)["version"]) == REG_VERSION:
        return {"label": label, "cached": True}
    R, ft, cols = _signals(label)
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_on_s"])
    on = np.sort(A.daq_on_s.to_numpy(float)[np.isfinite(A.daq_on_s.to_numpy(float))])
    T = T[np.isfinite(T.strobe_s) & np.isfinite(T.cue_s) & (T.cue_s > T.strobe_s)].sort_values("strobe_s")
    s, c = T.strobe_s.to_numpy(float), T.cue_s.to_numpy(float)
    Xa = ce._event_cols(s, ft, ARR_LAGS, spacing=ARR_SPACING).astype(np.float64)
    Xl = ce._event_cols(on, ft, LICK_LAGS, spacing=LICK_SPACING).astype(np.float64)
    na = Xa.shape[1]
    X = np.hstack([Xa, Xl])
    j = np.searchsorted(s - FIT_PRE, ft, side="right") - 1                # trial whose fit window may hold the frame
    fit = (j >= 0) & (ft < c[np.clip(j, 0, None)])
    grp = j[fit]
    Xf, Yf = X[fit], R[fit]
    tri = np.unique(grp)
    fold = {t: k % 5 for k, t in enumerate(np.random.default_rng(0).permutation(tri))}
    fk = np.array([fold[g] for g in grp])
    sse = np.zeros(len(ALPHAS))
    for k in range(5):
        tr, te = fk != k, fk == k
        for ia, al in enumerate(ALPHAS):
            m = Ridge(alpha=al).fit(Xf[tr], Yf[tr])
            sse[ia] += float(((Yf[te] - m.predict(Xf[te])) ** 2).sum())
    al = float(ALPHAS[int(np.argmin(sse))])
    m = Ridge(alpha=al).fit(Xf, Yf)
    cv_r2 = 1 - sse.min() / float(((Yf - Yf.mean(0)) ** 2).sum())
    # impulse responses on a grid with the session's frame interval (the basis depends on it), then onto LAGS_K
    dt = float(np.median(np.diff(ft)))
    fg = np.arange(LAGS_K[0] - 0.5, LAGS_K[-1] + 0.5, dt)

    def _imp(lags, spacing):
        B = ce._event_cols(np.array([0.0]), fg, lags, spacing=spacing).astype(np.float64)
        return np.stack([np.interp(LAGS_K, fg, B[:, q]) for q in range(B.shape[1])], 1)
    Kg, Kl = _imp(ARR_LAGS, ARR_SPACING), _imp(LICK_LAGS, LICK_SPACING)
    if Kg.shape[1] != na or Kl.shape[1] != Xl.shape[1]:
        raise RuntimeError(f"{label}: impulse basis {Kg.shape[1]}/{Kl.shape[1]} != design {na}/{Xl.shape[1]}")
    k_arr, k_lick = Kg @ m.coef_[:, :na].T, Kl @ m.coef_[:, na:].T        # (lags, cols)
    half = []
    for par in (0, 1):
        sel = (grp % 2) == par
        mh = Ridge(alpha=al).fit(Xf[sel], Yf[sel])
        hk = Kg @ mh.coef_[:, :na].T
        half.append(hk - hk[(LAGS_K >= -1.0) & (LAGS_K <= -0.5)].mean(0))
    sh_r = np.array([np.corrcoef(half[0][:, q], half[1][:, q])[0, 1] for q in range(len(cols))])
    # the kernel's level is only identified up to the intercept: REFERENCE it to its pre-strobe value (lags -1..-0.5 s,
    # spout still moving out) -- the removed arrival component = kernel - that level, inside the kernel's support
    pre_k = (LAGS_K >= -1.0) & (LAGS_K <= -0.5)
    k_pre = k_arr[pre_k].mean(0)
    sup = (np.abs(Xa).sum(1) > 0)[:, None]
    Rc = R - (Xa @ m.coef_[:, :na].T - sup * k_pre[None, :])
    z = np.load(SESS / f"{label}.npz")
    peak, ok = z["peak"], z["ok"]
    Xs = il._sample(Rc, ft, np.where(ok, peak, ft[0] + 2), il.LAGS_D)
    pool = {q: np.sum([Xs[:, :, cols.index(f"{a}_{h}")] for a in POOL], 0) for q, h in (("c", "right"), ("i", "left"))}
    nan = np.float32(np.nan)
    np.savez_compressed(
        f, version=REG_VERSION, cols=np.array(cols), alpha=al, cv_r2=cv_r2, n_trials=len(tri), n_frames=int(fit.sum()),
        k_arr=(k_arr - k_pre).astype(np.float32), k_lick=k_lick.astype(np.float32), k_arr_half=np.stack(half).astype(np.float32),
        splithalf_r=sh_r, lags_k=LAGS_K,
        Xdec_c=np.where(ok[:, None], Xs[:, W_DEC].mean(1), nan).astype(np.float32),
        Xpost_c=np.where(ok[:, None], Xs[:, W_POST].mean(1), nan).astype(np.float32),
        Xbase_c=np.where(ok[:, None], Xs[:, W_BASE].mean(1), nan).astype(np.float32),
        cT_c=np.where(ok[:, None], pool["c"], nan).astype(np.float16),
        iT_c=np.where(ok[:, None], pool["i"], nan).astype(np.float16))
    return {"label": label, "alpha": al, "cv_r2": cv_r2, "n_trials": len(tri)}


def regress_all(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    SESS_REG.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(il.OUT / "enl_licks" / "enl_trials.csv.gz", usecols=["label", "trial_id", "pos", "strobe_s", "cue_s"])
    labs = sorted(p.stem for p in SESS.glob("*.npz"))
    items = [(lab, T[T.label == lab].reset_index(drop=True)) for lab in labs]
    got, fail = ak.fan_sessions(items, regress, jobs=jobs, key=lambda it: it[0], label="enlv regress")
    for it, err in fail:
        print(f"  !! {it[0]}: {err}", flush=True)
    print(f"regressed {len(got)} sessions, {len(fail)} failed", flush=True)


def extract(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    SESS.mkdir(parents=True, exist_ok=True)
    L = pd.read_csv(il.OUT / "enl_licks" / "enl_licks.csv.gz", usecols=["label", "lick_uid", "daq_on_s"])
    T = pd.read_csv(il.OUT / "enl_licks" / "enl_trials.csv.gz", usecols=["label", "trial_id", "pos", "strobe_s", "cue_s"])
    run = set(il.run_labels())
    items = [(lab, g.reset_index(drop=True), T[T.label == lab].reset_index(drop=True))
             for lab, g in L.groupby("label", sort=True) if lab in run]
    got, fail = ak.fan_sessions(items, one, jobs=jobs, key=lambda it: it[0], label="enlv extract")
    for it, err in fail:
        print(f"  !! {it[0]}: {err}", flush=True)
    print(f"extracted {len(got)} sessions, {len(fail)} failed", flush=True)


# ------------------------------------------------------------------------------------------------ frozen readers
def _fit(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    return sc, RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(X), y)


def freeze(a: str, align: str, S: dict, script: str) -> dict:
    """RULE 10: {held-out pre label | None: (scaler, ridge)} for one animal x training set via load_or_fit (local store).
    S = {label: (X scaled, y)} of the animal's pre sessions."""
    import hashlib

    from wfield_local import frozen_models as fm
    labs = sorted(S)
    dig = hashlib.sha1(b"".join(np.ascontiguousarray(S[x][0]).tobytes() + np.ascontiguousarray(S[x][1]).tobytes()
                                for x in labs)).hexdigest()
    spec = fm.make_spec(a, "lick_reader", align=align, source=f"areas_split_r{il.ROI_VERSION}", train_labels=labs,
                        post_s=0.2, n_features=S[labs[0]][0].shape[1])

    def fit():
        out = {"xy_sha1": dig, "fits": {}}
        for held in [*labs, None]:
            trn = [q for q in labs if q != held]
            out["fits"][held] = _fit(np.vstack([S[q][0] for q in trn]), np.concatenate([S[q][1] for q in trn]))
        return out
    P, status = fm.load_or_fit(spec, fit, meta={"script": script, "align": align,
                                                "n_licks": int(sum(len(S[x][1]) for x in labs))})
    print(f"[frozen] {a} lick_reader {align}: {status}", flush=True)
    if P.get("xy_sha1") != dig:
        print(f"[frozen] !! {a} {align}: stored model trained on DIFFERENT data than today's selection -- stored used",
              flush=True)
    return P["fits"]


def reader_training(a: str, st: str, D: dict, names, SD) -> dict:
    """The iti_lick_order.predictions training rows (no ITI filter) of animal a, set st, pre sessions only."""
    import iti_first_lick as fl
    import iti_lick_order as lo

    from wfield_local import orofacial_cohort as oc
    S = {}
    for x, d in sorted(D.items()):
        if d["animal"] != a or d["epoch"] != "pre":
            continue
        X, C = d[st]
        t = il._load(x)["iti_t" if st == "iti" else "tr_t"].astype(float)
        idx = lo._bout_idx(x, t) if st == "iti" else fl._lick_idx(x, t)
        pos = np.array([q if q in lo.POS or "_" not in q else oc.pos_key(q, "L") for q in d[f"{st}_pos"].astype(str)])
        ok = np.isfinite(C[:, 0]) & np.isfinite(idx) & np.isin(pos, lo.POS)
        if ok.sum() >= 20:
            S[x] = (X[ok] / SD.loc[x, names].to_numpy(float)[None, :], C[ok, 0])
    return S


def readers(jobs: int) -> dict:
    """Freeze + verify the ITI and cued readers; returns {(animal, set): fits}."""
    import iti_lick_order as lo
    import iti_where_decoder as wd

    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    names, SD = wd.area_meta(basis="split")
    pre = [x for x in il.labels() if str(epochs.epoch_of(x)) == "pre"]
    res, fail = ak.fan_sessions([(x, "areas_split", "") for x in pre], wd._load_one, jobs=jobs, key=lambda it: it[0],
                                label="reader load", log=lambda m: None)
    if fail:
        raise RuntimeError(f"reader load failed: {fail[:2]}")
    D = {x: rec for _it, (x, rec) in res if rec is not None}
    P = pd.read_csv(lo.preds_path())
    out, chk = {}, []
    for st in ("iti", "trial"):
        for a in ["PS92", "PS93", "PS94", "PS95"]:
            S = reader_training(a, st, D, names, SD)
            fits = freeze(a, st, S, "lick_enlv_extract.py")
            out[(a, st)] = fits
            for x, (X, y) in S.items():
                sc, m = fits[x]
                dec = m.predict(sc.transform(X))
                ref = P[(P.set == st) & (P.label == x)]
                same_n = len(ref) == len(dec)
                chk.append({"set": st, "animal": a, "label": x, "n": len(dec), "n_cache": len(ref),
                            "max_abs_diff": float(np.max(np.abs(ref.dec.to_numpy() - dec))) if same_n else np.nan,
                            "act_match": bool(same_n and np.allclose(ref.act.to_numpy(), y))})
    C = pd.DataFrame(chk)
    C.to_csv(OUT / "reader_check.csv", index=False)
    print("== reader verification vs lick_order_preds (pre sessions, held-out siblings)\n"
          + C.groupby(["set", "animal"]).agg(n_sess=("label", "size"), max_abs_diff=("max_abs_diff", "max"),
                                              act_match=("act_match", "all")).to_string())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--readers", action="store_true")
    ap.add_argument("--regress", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.regress:
        regress_all(min(a.jobs, 4))
    elif a.readers:
        readers(min(a.jobs, 4))
    else:
        extract(min(a.jobs, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
