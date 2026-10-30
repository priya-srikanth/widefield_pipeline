"""CUE-TO-LICK LEVEL, ACCUMULATION-TO-THRESHOLD TEST and FIXED PRE-LICK WINDOWS (Priya 10-29, tasks A-C). PROVISIONAL.

DATA  per-trial raw area traces of lick_cue_to_lick_extract.py (= the Fig 14.21 restw-trace construction; 8 areas x 2
      hemispheres, VISp contralesional only; contralesional = right), ENGAGED trials, first lick IN the trial window
      (cue .. stop; revup_neural's rule), latency L = first lick - cue, 0.05 <= L <= 3.6 s; sessions with restw_ok only.
      y(t) = trace - session RESTW (Fig 14.21 baseline); p0 = the trial's own pre-cue level, mean y over -0.5..0 s.
A  CUE-TO-LICK MEAN LEVEL per trial = mean of y over [0, L] (trapezoid on the 0.025 s grid, interpolated at L):
      'mean_rw' (above restw) and 'mean_own' (above the trial's own pre-cue level). Per session x spout (>= 3 trials;
      'all' = all spouts pooled, >= 10) -> animal x epoch -> change vs pre with analysis_kit.boot_delta.
      LATENCY-MATCHED: per animal x spout and per comparison (pre vs epoch E), latency bins EDGES; common weights
      w_b = min(share of pre trials in b, share of E trials in b); each session's matched value = sum_b w_b m_sb / sum_b w_b
      over its populated bins (pre sessions re-weighted for every E). 'overlap' = sum_b w_b (1 = identical latency
      distributions); 'all' matched = mean over the session's spouts.
B  ACCUMULATION-TO-THRESHOLD. Per trial: AUC_own = int_0^L (y - p0) dt; AUC_rw = int_0^L y dt; lev_own / lev_rw = the
      level at the lick (lick-aligned mean -0.1..0 s); slope_end = lev_own / L (cue -> lick ramp); mean_own (A).
      CUE-LOCKED COMPONENT (the sensory response the lick rides on): template c(t) per animal x epoch x spout class = mean
      of (y - p0)(t) over trials that have NOT licked by t + 0.3 s (in-window licks with L > t + 0.3, plus no-lick / late
      trials up to stop - 0.3 s), >= 10 trials per t. AUC_res = AUC_own - int_0^L c; lev_res = lev_own - c(L).
      M4 'pure cue-locked' prediction for a trial = the template values at its L (AUC_tpl, lev_tpl).
      MODELS (exponent of the quantity on L, linear ramp from the cue):
        M1 accumulate-to-bound (AUC reaches theta): AUC 0, level-at-lick -1, slope -2, mean level -1
        M2 fixed level crossing (rise to a fixed level): AUC +1, level 0, slope -1, mean level 0
        M3 delayed start, fixed trajectory (rev-up result): AUC ~0 (own pre), level 0, slope(cue->lick) -1, mean level -1
        M4 cue-locked response, lick timing independent: the template's own relation (empirical)
      TESTS per session x spout class (all / far_contra / other; >= 8 trials, sd(log L) > 0.1): OLS slope b of the
      quantity on log(L) ('per e-fold of latency'); flat = b ~ 0. Animal: mean b over sessions (boot_ci levels, boot_delta
      changes vs pre). EXPONENTS: per animal x epoch x class, latency-binned means (sessions pooled, >= 5 trials per bin,
      positive means) -> log-log slope beta, compared with the model table. Sensitivity: everything also on a
      deconvolved trace s = y + tau dy/dt (tau 0.6 s, 0.15 s boxcar first; GCaMP6s).
      ON THE PRE RELATION: per animal, pre-stroke fit Q = a + b log L (pooled pre trials of the class); acute (and other
      epochs') residuals of far-contra trials, fast (L <= 0.5 s) vs slow (L > 0.5 s), with the pre latency range for the
      extrapolation flag.
C  FIXED WINDOW BEFORE THE LICK: lick-aligned mean y over -0.3..0 and -0.5..0 s; the window must start after the cue,
      so trials with L < window length are DROPPED (reported: kept fraction); '_all' keeps them (flagged, contains the
      pre-cue period).
GCaMP6s CAVEAT: fluorescence lags spikes ~0.1-0.2 s and decays over ~0.5-1 s (widefield smoother still): the AUC of a slow
      indicator is a low-passed, lagged integral, NOT a spike count, and the 'level at the lick' already contains
      activity from ~0.5 s before it. Short windows (median L ~0.2 s) sit on the RISING edge of the cue response.
Outputs: <OUT_W>/cue_to_lick/ c2l_trials.csv.gz, c2l_A_*.csv, c2l_B_*.csv, c2l_C_*.csv; figures/ lick_cue_to_lick_*.png;
figdata/<stem>.pkl (redraw with --plot-only). Usage: python scripts/lick_cue_to_lick.py [--jobs 4] [--plot-only]
[--rest-variant restnl]
RESTnl VARIANT (`--rest-variant restnl`, default restw unchanged): per-trial cache sessions_restnl/ (CURRENT engaged gate,
y = trace - the LICK-FREE rest level, scripts/restnl.py); every output (csv, figdata, png) gets the suffix _restnl.
Own-pre-cue quantities (mean_own, AUC_own, lev_own, slope_end, B) do not depend on the rest level at all; under restnl
they move only through the trial set (current gate) and the sessions without a RESTnl level (left out).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import figdata as _fd  # noqa: E402
import lick_cue_to_lick_extract as cx  # noqa: E402
import restnl  # noqa: E402
from figdata import plot_only  # noqa: E402

SFX = ""                        # "_restnl" under --rest-variant restnl (set in main; every output name carries it)


def save_fd(stem, obj, root=None):
    return _fd.save_fd(stem + SFX, obj, root)


def load_fd(stem, root=None):
    return _fd.load_fd(stem + SFX, root)


def _o(name: str) -> str:
    """Output file name with the variant suffix before the extension(s)."""
    if not SFX:
        return name
    stem, dot, ext = name.partition(".")
    return f"{stem}{SFX}{dot}{ext}"


def _rl(txt: str) -> str:
    return restnl.relabel(txt) + " Trials = CURRENT engaged gate." if SFX else txt

OUT = cx.OUT
FIGS = OUT / "figures"
FDR = OUT / "figdata"
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]   # orofacial_cohort.POS_ORDER
SPOUTS = POS + ["all"]
CLS = ("all", "far_contra", "other")
EP = il.EP4
POST = EP[1:]
KEY = ["ALM", "tjM1", "SSp-m", "SSp-n"]
COLS = cx.COLS
DT = cx.DT
LC = cx.LAGS_C
LL = cx.LAGS_L
EDGES = np.array([0.05, 0.125, 0.175, 0.225, 0.3, 0.4, 0.55, 0.8, 1.2, 3.6])
TAUS = (0.0, 0.6)
LAG_EXCL = 0.3                  # template: trials not licked by t + 0.3 s
MIN_TPL = 10
QS = ["mean_rw", "mean_own", "AUC_rw", "AUC_own", "AUC_res", "lev_rw", "lev_own", "lev_res", "slope_end", "w03", "w05"]
BQ = ["AUC_own", "AUC_res", "AUC_rw", "lev_own", "lev_res", "lev_rw", "slope_end", "mean_own"]
PRED = {   # model exponents (quantity ~ L^beta)
    "AUC_own": {"M1 accumulate-to-bound": 0, "M2 fixed level": 1, "M3 delayed start": 0},
    "lev_own": {"M1 accumulate-to-bound": -1, "M2 fixed level": 0, "M3 delayed start": 0},
    "slope_end": {"M1 accumulate-to-bound": -2, "M2 fixed level": -1, "M3 delayed start": -1},
    "mean_own": {"M1 accumulate-to-bound": -1, "M2 fixed level": 0, "M3 delayed start": -1},
}
N_BOOT = 2000


def _w(lags, a, b):
    return (lags >= a - 1e-9) & (lags <= b + 1e-9)


def deconv(X: np.ndarray, tau: float, axis: int = 1) -> np.ndarray:
    """s = x + tau dx/dt after a 0.15 s boxcar (inverse of a first-order exponential kernel; as revup_neural.deconv)."""
    if not tau:
        return X
    from scipy.ndimage import uniform_filter1d
    Xs = uniform_filter1d(X, int(round(0.15 / DT)), axis=axis, mode="nearest")
    return Xs + tau * np.gradient(Xs, DT, axis=axis)


# ------------------------------------------------------------------------------------------------ per-session trial table
def session_trials(label: str) -> dict | None:
    """Per-trial scalars (long over columns) + template partial sums, for both taus. Module-level (fan_sessions)."""
    return _session_trials(label, cx.SESS)


def session_trials_restnl(label: str) -> dict | None:
    return _session_trials(label, cx.SESS_NL)


def _session_trials(label: str, sess) -> dict | None:
    z = np.load(sess / f"{label}.npz")
    if not bool(z["restw_ok"]):
        return None
    an, ep = str(z["animal"]), str(z["epoch"])
    restw = z["restw"].astype(np.float64)
    pos = z["pos"].astype(str)
    phase = z["phase"].astype(str)
    lat = z["fl"] - z["cue"]
    stop_l = z["stop"] - z["cue"]
    Xc0 = z["Xc"].astype(np.float64) - restw[None, None, :]
    Xl0 = z["Xl"].astype(np.float64) - restw[None, None, :]
    inw = (phase == "in_window") & (lat >= EDGES[0]) & (lat <= EDGES[-1])
    tp = LC >= -1e-9
    tg = LC[tp]
    rows, tpl = [], {}
    for tau in TAUS:
        Xc, Xl = deconv(Xc0, tau), deconv(Xl0, tau)
        p0 = np.nanmean(Xc[:, _w(LC, -0.5, 0.0)], 1)                       # (n, c)
        Y = Xc[:, tp]                                                       # (n, t, c)
        Z = Y - p0[:, None, :]
        F = np.concatenate([np.zeros((Y.shape[0], 1, Y.shape[2])), np.cumsum((Y[:, 1:] + Y[:, :-1]) / 2 * DT, 1)], 1)
        n = len(lat)
        auc_rw = np.full((n, len(COLS)), np.nan)
        for i in np.flatnonzero(inw):
            k = lat[i] / DT
            k0 = int(np.floor(k))
            fr = k - k0
            auc_rw[i] = F[i, k0] + fr * (F[i, min(k0 + 1, len(tg) - 1)] - F[i, k0])
        lev_rw = np.nanmean(Xl[:, _w(LL, -0.1, 0.0)], 1)
        w03 = np.nanmean(Xl[:, _w(LL, -0.3, 0.0)], 1)
        w05 = np.nanmean(Xl[:, _w(LL, -0.5, 0.0)], 1)
        L = lat[:, None]
        R = {"mean_rw": auc_rw / L, "AUC_rw": auc_rw, "AUC_own": auc_rw - p0 * L, "mean_own": auc_rw / L - p0,
             "lev_rw": lev_rw, "lev_own": lev_rw - p0, "slope_end": (lev_rw - p0) / L, "w03": w03, "w05": w05, "p0": p0}
        # template partial sums: (y - p0)(t) on trials not licked by t + LAG_EXCL
        lim = np.where(inw, lat - LAG_EXCL, np.where(phase == "in_window", -np.inf, stop_l - LAG_EXCL))
        lim = np.where(np.isfinite(lim), lim, -np.inf)
        keep = (tg[None, :] <= lim[:, None])                                # (n, t)
        for cl in CLS:
            mc = np.ones(n, bool) if cl == "all" else ((pos == "far_contra") == (cl == "far_contra"))
            K = keep & mc[:, None]
            Zk = np.where(K[:, :, None], Z, 0.0)
            Zk = np.where(np.isfinite(Zk), Zk, 0.0)
            tpl[(tau, cl)] = (Zk.sum(0), (K[:, :, None] & np.isfinite(Z)).sum(0))
        idx = np.flatnonzero(inw)
        for j, c in enumerate(COLS):
            d = {"label": label, "animal": an, "epoch": ep, "trial": idx, "pos": pos[idx], "lat": lat[idx], "tau": tau,
                 "col": c}
            for k, v in R.items():
                d[k] = v[idx, j]
            rows.append(pd.DataFrame(d))
    return {"label": label, "animal": an, "epoch": ep, "T": pd.concat(rows, ignore_index=True), "tpl": tpl,
            "n_eng": len(lat), "n_inw": int(inw.sum()), "n_late": int((phase == "late").sum()),
            "n_never": int((phase == "never").sum())}


def build(jobs: int, var: str = "restw") -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    from wfield_local import analysis_kit as ak
    labs = cx.extract(jobs, var)
    res, fails = ak.fan_sessions(labs, session_trials_restnl if var == "restnl" else session_trials, jobs=min(jobs, 4))
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    res = [v for _x, v in res if v is not None]
    T = pd.concat([v["T"] for v in res], ignore_index=True)
    T["cls"] = np.where(T.pos == "far_contra", "far_contra", "other")
    T["bin"] = np.clip(np.digitize(T.lat, EDGES) - 1, 0, len(EDGES) - 2)
    # templates per animal x epoch x class x tau
    TPL = {}
    for (an, ep), grp in pd.DataFrame([{"an": v["animal"], "ep": v["epoch"], "i": i} for i, v in enumerate(res)]
                                      ).groupby(["an", "ep"]):
        for tau in TAUS:
            for cl in CLS:
                S = sum(res[i]["tpl"][(tau, cl)][0] for i in grp.i)
                N = sum(res[i]["tpl"][(tau, cl)][1] for i in grp.i)
                c = np.where(N >= MIN_TPL, S / np.maximum(N, 1), np.nan)
                C = np.concatenate([np.zeros((1, c.shape[1])), np.cumsum((c[1:] + c[:-1]) / 2 * DT, 0)], 0)
                TPL[(an, ep, tau, cl)] = (c, C, N)
    tg = LC[LC >= -1e-9]
    T["AUC_tpl"], T["lev_tpl"] = np.nan, np.nan
    for (an, ep, tau, col, cl), g in T.groupby(["animal", "epoch", "tau", "col", "cls"]):
        c, C, _N = TPL[(an, ep, tau, cl)]
        j = COLS.index(col)
        T.loc[g.index, "AUC_tpl"] = np.interp(g.lat, tg, C[:, j], right=np.nan)
        T.loc[g.index, "lev_tpl"] = np.interp(g.lat, tg, c[:, j], right=np.nan)
    T["AUC_res"] = T.AUC_own - T.AUC_tpl
    T["lev_res"] = T.lev_own - T.lev_tpl
    meta = pd.DataFrame([{k: v[k] for k in ("label", "animal", "epoch", "n_eng", "n_inw", "n_late", "n_never")}
                         for v in res])
    return T, TPL, meta


# ------------------------------------------------------------------------------------------------ helpers: tables + boot
def per_session(T: pd.DataFrame, q: str, valid: pd.Series | None = None) -> pd.DataFrame:
    """session x spout(+all) x col mean of q (>= 3 trials per spout, >= 10 all)."""
    t = T if valid is None else T[valid]
    g = t.groupby(["label", "animal", "epoch", "col", "pos"])[q].agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= 3]
    a = t.groupby(["label", "animal", "epoch", "col"])[q].agg(["mean", "count"]).reset_index()
    a = a[a["count"] >= 10]
    a["pos"] = "all"
    return pd.concat([g, a], ignore_index=True).rename(columns={"mean": "v", "count": "n"})


def matched(T: pd.DataFrame, q: str) -> pd.DataFrame:
    """Latency-matched session values per comparison epoch (see A in the docstring)."""
    t = T[["label", "animal", "epoch", "col", "pos", "bin", q]]
    cnt = t[t.col == COLS[0]].groupby(["animal", "epoch", "pos", "bin"]).size()
    sb = t.groupby(["label", "animal", "epoch", "col", "pos", "bin"])[q].mean().rename("m").reset_index()
    out = []
    for E in POST:
        for (an, p), _g in cnt.groupby(level=["animal", "pos"]):
            try:
                cp = cnt.loc[(an, "pre", p)]
                ce = cnt.loc[(an, E, p)]
            except KeyError:
                continue
            fp, fe = cp / cp.sum(), ce / ce.sum()
            w = pd.concat([fp, fe], axis=1).fillna(0).min(1)
            w = w[w > 0]
            if w.empty:
                continue
            ov = float(w.sum())
            s = sb[(sb.animal == an) & (sb.pos == p) & sb.epoch.isin(["pre", E]) & sb.bin.isin(w.index)].copy()
            s["w"] = s.bin.map(w)
            s["wm"] = s.w * s.m
            r = s.groupby(["label", "animal", "epoch", "col", "pos"])[["wm", "w"]].sum().reset_index()
            r["v"] = r.wm / r.w
            r["cmp"], r["overlap"] = E, ov
            out.append(r.drop(columns=["wm", "w"]))
    M = pd.concat(out, ignore_index=True)
    A = M.groupby(["label", "animal", "epoch", "col", "cmp"]).agg(v=("v", "mean"), overlap=("overlap", "mean")
                                                                    ).reset_index()
    A["pos"] = "all"
    return pd.concat([M, A], ignore_index=True)


def deltas(S: pd.DataFrame, tag: dict, rng, cmp_col: bool = False) -> list[dict]:
    """boot_delta (epoch - pre) per col x pos x epoch from session values S (v)."""
    from wfield_local import analysis_kit as ak
    rows = []
    for (col, p), g in S.groupby(["col", "pos"], sort=True):
        for E in POST:
            gg = g[g.cmp == E] if cmp_col else g
            pre = {a: list(x.v) for a, x in gg[gg.epoch == "pre"].groupby("animal", sort=True)}
            post = {a: list(x.v) for a, x in gg[gg.epoch == E].groupby("animal", sort=True)}
            iv = ak.boot_delta(post, pre, rng, n_boot=N_BOOT)
            per = {a: float(np.mean(post[a]) - np.mean(pre[a])) for a in sorted(set(pre) & set(post))}
            if iv is None:
                continue
            same = sum(np.sign(v) == np.sign(iv.point) for v in per.values())
            rows.append({**tag, "col": col, "pos": p, "epoch": E, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                         "n_animals": iv.n_animals, "n_same_sign": same, "sig": iv.excludes_zero,
                         "flag_lt3": (iv.n_animals < 3) or (same < 3),
                         **{f"d_{a}": v for a, v in per.items()}})
    return rows


def animal_table(S: pd.DataFrame) -> pd.DataFrame:
    return S.groupby(["col", "pos", "animal", "epoch"]).v.mean().unstack("epoch").reindex(columns=EP)


# ------------------------------------------------------------------------------------------------ A and C
def task_A(T: pd.DataFrame, rng) -> dict:
    T0 = T[T.tau == 0]
    out, rows = {}, []
    for q in ("mean_rw", "mean_own"):
        S = per_session(T0, q)
        S["cmp"] = ""
        PA = animal_table(S)
        PA.to_csv(OUT / _o(f"c2l_A_{q}_per_animal.csv"))
        rows += deltas(S, {"measure": q, "match": "raw"}, rng)
        Mt = matched(T0, q)
        Mt.to_csv(OUT / _o(f"c2l_A_{q}_matched_sessions.csv.gz"), index=False)
        rows += deltas(Mt, {"measure": q, "match": "latency-matched"}, rng, cmp_col=True)
        PM = Mt.groupby(["col", "pos", "animal", "cmp", "epoch"]).v.mean().unstack(["cmp", "epoch"])
        PM.to_csv(OUT / _o(f"c2l_A_{q}_matched_per_animal.csv"))
        ov = Mt[Mt.pos != "all"].groupby(["pos", "animal", "cmp"]).overlap.first().unstack("cmp").reindex(index=POS, level=0)
        ov.to_csv(OUT / _o("c2l_A_latency_overlap.csv"))
        out[q] = {"S": S, "M": Mt}
    D = pd.DataFrame(rows)
    D.to_csv(OUT / _o("c2l_A_changes.csv"), index=False)
    return {"D": D, "sess": {q: v["S"] for q, v in out.items()}, "match": {q: v["M"] for q, v in out.items()}}


def task_C(T: pd.DataFrame, rng) -> dict:
    T0 = T[T.tau == 0]
    rows, sess, keep = [], {}, []
    for q, wl in (("w03", 0.3), ("w05", 0.5)):
        for drop in (True, False):
            S = per_session(T0, q, (T0.lat >= wl) if drop else None)
            S["cmp"] = ""
            nm = q + ("" if drop else "_all")
            sess[nm] = S
            animal_table(S).to_csv(OUT / _o(f"c2l_C_{nm}_per_animal.csv"))
            rows += deltas(S, {"measure": nm, "window": f"-{wl}..0 s", "drop_precue": drop}, rng)
        t1 = T0[T0.col == COLS[0]]
        k = t1.assign(kept=t1.lat >= wl).groupby(["animal", "epoch", "pos"]).kept.mean()
        keep.append(k.rename(q))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / _o("c2l_C_changes.csv"), index=False)
    K = pd.concat(keep, axis=1).reset_index()
    K.to_csv(OUT / _o("c2l_C_kept_fraction.csv"), index=False)
    return {"D": D, "sess": sess, "kept": K}


# ------------------------------------------------------------------------------------------------ B
def _ols_b(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8 or np.std(x[ok]) < 0.1:
        return np.nan
    x, y = x[ok], y[ok]
    return float(np.polyfit(x, y, 1)[0])


def task_B(T: pd.DataFrame, rng) -> dict:
    from wfield_local import analysis_kit as ak
    rec = []
    for (lab, an, ep, tau, col), g in T.groupby(["label", "animal", "epoch", "tau", "col"], sort=True):
        for cl in CLS:
            h = g if cl == "all" else g[g.cls == cl]
            if len(h) < 8:
                continue
            x = np.log(h.lat.to_numpy())
            r = {"label": lab, "animal": an, "epoch": ep, "tau": tau, "col": col, "cls": cl, "n": len(h),
                 "lat_med": float(h.lat.median())}
            for q in BQ + ["AUC_tpl", "lev_tpl"]:
                r[f"b_{q}"] = _ols_b(x, h[q].to_numpy())
                r[f"m_{q}"] = float(h[q].mean())
            rec.append(r)
    B = pd.DataFrame(rec)
    B.to_csv(OUT / _o("c2l_B_session_slopes.csv.gz"), index=False)
    # per animal (rule 8) + pooled levels / changes
    PA = B.groupby(["tau", "cls", "col", "animal", "epoch"])[[f"b_{q}" for q in BQ + ["AUC_tpl", "lev_tpl"]]].mean()
    PA.to_csv(OUT / _o("c2l_B_slopes_per_animal.csv"))
    lev, chg = [], []
    for (tau, cl, col), g in B.groupby(["tau", "cls", "col"], sort=True):
        for q in BQ + ["AUC_tpl", "lev_tpl"]:
            k = f"b_{q}"
            for E in EP:
                d = {a: [v for v in x[k] if np.isfinite(v)] for a, x in g[g.epoch == E].groupby("animal", sort=True)}
                d = {a: v for a, v in d.items() if v}
                iv = ak.boot_ci(d, rng, n_boot=N_BOOT) if d else None
                if iv is not None:
                    per = {a: float(np.mean(v)) for a, v in d.items()}
                    lev.append({"tau": tau, "cls": cl, "col": col, "q": q, "epoch": E, "b": iv.point, "lo": iv.lo,
                                "hi": iv.hi, "n_animals": iv.n_animals, "sig": iv.excludes_zero,
                                "n_pos": sum(v > 0 for v in per.values()), "n_neg": sum(v < 0 for v in per.values()),
                                **{f"b_{a}": v for a, v in per.items()}})
                if E == "pre":
                    continue
                pre = {a: [v for v in x[k] if np.isfinite(v)] for a, x in g[g.epoch == "pre"].groupby("animal", sort=True)}
                pre = {a: v for a, v in pre.items() if v}
                iv = ak.boot_delta(d, pre, rng, n_boot=N_BOOT) if d and pre else None
                if iv is not None:
                    chg.append({"tau": tau, "cls": cl, "col": col, "q": q, "epoch": E, "delta": iv.point, "lo": iv.lo,
                                "hi": iv.hi, "n_animals": iv.n_animals, "sig": iv.excludes_zero})
    LV, CH = pd.DataFrame(lev), pd.DataFrame(chg)
    LV.to_csv(OUT / _o("c2l_B_slopes_levels.csv"), index=False)
    CH.to_csv(OUT / _o("c2l_B_slopes_changes.csv"), index=False)
    # exponents from latency-binned means (sessions pooled per animal x epoch x class)
    ex, bins = [], []
    for (tau, col, an, ep), g in T.groupby(["tau", "col", "animal", "epoch"], sort=True):
        for cl in CLS:
            h = g if cl == "all" else g[g.cls == cl]
            bm = h.groupby("bin").agg(n=("lat", "size"), lat=("lat", "median"),
                                      **{q: (q, "mean") for q in BQ + ["AUC_tpl", "lev_tpl"]})
            bm = bm[bm.n >= 5]
            for b, r in bm.iterrows():
                bins.append({"tau": tau, "col": col, "animal": an, "epoch": ep, "cls": cl, "bin": b, **r.to_dict()})
            for q in list(PRED) + ["AUC_tpl", "lev_tpl"]:
                v = bm[q]
                ok = (v > 0) & np.isfinite(v)
                beta = float(np.polyfit(np.log(bm.lat[ok]), np.log(v[ok]), 1)[0]) if ok.sum() >= 3 else np.nan
                ex.append({"tau": tau, "col": col, "animal": an, "epoch": ep, "cls": cl, "q": q, "beta": beta,
                           "n_bins": int(ok.sum()), "n_bins_all": len(bm)})
    EX, BM = pd.DataFrame(ex), pd.DataFrame(bins)
    EX.to_csv(OUT / _o("c2l_B_exponents_per_animal.csv"), index=False)
    BM.to_csv(OUT / _o("c2l_B_binned_means.csv.gz"), index=False)
    # acute (and later) far-contra trials on the PRE relation
    onpre = []
    for (tau, col, an), g in T.groupby(["tau", "col", "animal"], sort=True):
        for ref in ("all", "far_contra"):
            pr = g[(g.epoch == "pre") & ((g.cls == "far_contra") if ref == "far_contra" else True)]
            if len(pr) < 20:
                continue
            for q in ("lev_own", "AUC_own", "lev_rw", "AUC_res", "mean_own"):
                ok = np.isfinite(pr[q])
                a1, a0 = np.polyfit(np.log(pr.lat[ok]), pr[q][ok], 1)
                fc = g[(g.cls == "far_contra")]
                res = fc[q] - (a0 + a1 * np.log(fc.lat))
                for (ep, lab), h in fc.assign(res=res).groupby(["epoch", "label"], sort=True):
                    for spd, m in (("fast<=0.5", h.lat <= 0.5), ("slow>0.5", h.lat > 0.5)):
                        if m.sum() >= 2:
                            onpre.append({"tau": tau, "col": col, "animal": an, "ref": ref, "q": q, "epoch": ep,
                                          "label": lab, "speed": spd, "n": int(m.sum()), "res": float(h.res[m].mean()),
                                          "q_mean": float(h[q][m].mean()), "pre_lat_p95": float(pr.lat.quantile(0.95)),
                                          "pre_lat_max": float(pr.lat.max()), "pre_b": a1, "pre_a": a0})
    OP = pd.DataFrame(onpre)
    OP.to_csv(OUT / _o("c2l_B_onpre_sessions.csv"), index=False)
    opl = []
    for (tau, col, ref, q, ep, spd), g in OP.groupby(["tau", "col", "ref", "q", "epoch", "speed"], sort=True):
        d = {a: list(x.res) for a, x in g.groupby("animal", sort=True)}
        iv = ak.boot_ci(d, rng, n_boot=N_BOOT)
        if iv is None:
            continue
        per = {a: float(np.mean(v)) for a, v in d.items()}
        opl.append({"tau": tau, "col": col, "ref": ref, "q": q, "epoch": ep, "speed": spd, "res": iv.point, "lo": iv.lo,
                    "hi": iv.hi, "n_animals": iv.n_animals, "n_trials": int(g.n.sum()), **{f"r_{a}": v for a, v in per.items()}})
    OPL = pd.DataFrame(opl)
    OPL.to_csv(OUT / _o("c2l_B_onpre_summary.csv"), index=False)
    return {"LV": LV, "CH": CH, "EX": EX, "BM": BM, "OP": OP, "OPL": OPL}


# ------------------------------------------------------------------------------------------------ printing (rule 8)
def report(A: dict, B: dict, C: dict) -> None:
    pd.set_option("display.width", 250)
    kc = [f"{a}_{h}" for a in KEY for h in ("right", "left")]
    print("\n==== A. cue->lick mean level above restw (mean_rw), per animal, contralesional key areas, far_contra / all")
    for q in ("mean_rw",):
        PA = animal_table(A["sess"][q])
        sub = PA.loc[PA.index.get_level_values("col").isin(kc[::2]) & PA.index.get_level_values("pos").isin(
            ["far_contra", "all"])]
        print(sub.round(1).to_string())
    D = A["D"]
    for match in ("raw", "latency-matched"):
        s = D[(D.measure == "mean_rw") & (D.match == match) & D.col.isin(kc) & D.pos.isin(["far_contra", "all", "close_contra"])]
        print(f"\n== A boot_delta mean_rw ({match})\n" + s[["col", "pos", "epoch", "delta", "lo", "hi", "n_same_sign",
                                                           "sig", "flag_lt3"]].round(2).to_string(index=False))
    print("\n==== B. within-session slope b of Q on log(L) (per e-fold of latency), tau 0, PRE, class all")
    LV = B["LV"]
    s = LV[(LV.tau == 0) & (LV.cls == "all") & (LV.epoch == "pre") & LV.col.isin(kc)]
    print(s.pivot_table(index="col", columns="q", values="b").round(2).to_string())
    print("animals with b>0 / b<0 (pre, all):")
    print(s.assign(pn=s.n_pos.astype(str) + "/" + s.n_neg.astype(str)).pivot_table(index="col", columns="q", values="pn",
                                                                                   aggfunc="first").to_string())
    EX = B["EX"]
    e = EX[(EX.tau == 0) & (EX.cls == "all") & EX.col.isin(kc)]
    print("\n==== B exponents beta (log-log of binned means), per animal, tau 0, class all")
    print(e.pivot_table(index=["col", "q"], columns=["epoch", "animal"], values="beta").reindex(columns=EP, level=0
                                                                                                ).round(2).to_string())
    OPL = B["OPL"]
    s = OPL[(OPL.tau == 0) & (OPL.ref == "all") & OPL.col.isin(kc[::2]) & (OPL.epoch == "acute")]
    print("\n==== B acute far-contra trials vs the PRE all-spout relation (residual; per animal)\n"
          + s.round(2).to_string(index=False))
    Dc = C["D"]
    s = Dc[Dc.col.isin(kc[::2]) & Dc.pos.isin(["far_contra", "all"])]
    print("\n==== C fixed pre-lick windows, boot_delta\n" + s[["measure", "col", "pos", "epoch", "delta", "lo", "hi",
                                                             "n_same_sign", "sig", "flag_lt3"]].round(2).to_string(index=False))


# ------------------------------------------------------------------------------------------------ figures
def _acol():
    from wfield_local import config
    ac = config.animal_color()
    return {a: ac.get(a, "0.5") for a in ("PS92", "PS93", "PS94", "PS95")}


def _wrap(txt: str, fig_w: float, fs: float) -> str:
    """Pre-wrap a suptitle to the figure width (chars ~ width / (0.55 * fontsize)). Constrained layout then reserves
    exactly the title's height: no overlap with the panels and no blank band under it (tight_layout's fixed rect
    left either one or the other depending on the figure height)."""
    import textwrap
    return textwrap.fill(txt, max(40, int(fig_w * 72 / (0.55 * fs))))


def draw_dotgrid(stem: str) -> None:
    """Rows = areas (one hemisphere), columns = spouts (canonical order + all); x = epoch; per-animal dots + mean."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
    from wfield_local.orofacial_cohort import EPOCH_LINE
    F = load_fd(stem, FDR)
    PA, D, acol = F["PA"], F["D"], F["acol"]
    areas = [a for a in cx.AREAS if f"{a}_{F['hemi']}" in COLS]
    fw = 2.0 * len(SPOUTS)
    fig, axs = plt.subplots(len(areas), len(SPOUTS), figsize=(fw, 1.7 * len(areas)), squeeze=False, layout="constrained")
    for i, a in enumerate(areas):
        col = f"{a}_{F['hemi']}"
        sub = PA.xs(col, level="col") if col in PA.index.get_level_values("col") else None
        vals = sub.to_numpy(float) if sub is not None else np.array([[np.nan]])
        lim = np.nanpercentile(np.abs(vals), 98) * 1.15 if np.isfinite(vals).any() else 1
        for j, p in enumerate(SPOUTS):
            ax = axs[i, j]
            ax.axhline(0, color="0.6", lw=0.6)
            if sub is not None and p in sub.index.get_level_values("pos"):
                t = sub.xs(p, level="pos")
                for an, r in t.iterrows():
                    ax.plot(range(len(EP)), r.reindex(EP).to_numpy(float), "-o", color=acol.get(an, "0.5"), ms=2.5,
                            lw=0.6, alpha=0.8)
                mu = t.reindex(columns=EP).mean(0)
                ax.scatter(range(len(EP)), mu, marker="_", s=180, c=[EPOCH_LINE[e] for e in EP], lw=2.2, zorder=4)
                dd = D[(D.col == col) & (D.pos == p)]
                tr = blended_transform_factory(ax.transData, ax.transAxes)
                for _k, r in dd.iterrows():
                    x = EP.index(r.epoch)
                    if r.sig:   # above the axes, so a star never sits on a data line
                        ax.text(x, 1.0, "*" if not r.flag_lt3 else "(*)", ha="center", va="bottom", fontsize=9,
                                color=EPOCH_LINE[r.epoch], transform=tr, clip_on=False)
            ax.set_ylim(-lim if F.get("sym", True) else min(0, -0.1 * lim), lim)
            ax.set_xticks(range(len(EP)), [e[:3] for e in EP], fontsize=6)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(p.replace("_", " "), fontsize=8, pad=12)
            if j == 0:
                ax.set_ylabel(f"{a}\n{F['ylab']}", fontsize=7)
    hs = [plt.Line2D([], [], color=c, marker="o", ms=3, lw=0.6, label=an) for an, c in sorted(acol.items())]
    fig.legend(handles=hs, loc="outside lower center", ncol=len(hs), fontsize=7, frameon=False)
    fig.suptitle(_wrap(_rl(F["title"]), fw, 9), fontsize=9)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}{SFX}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def dotgrid_data(stem: str, S: pd.DataFrame, D: pd.DataFrame, hemi: str, ylab: str, title: str, cmp: str | None = None
                 ) -> None:
    if cmp is None:
        PA = animal_table(S)
    else:   # latency-matched: pre column = pre matched to that epoch's distribution; one figure per comparison
        s = S[S.cmp == cmp]
        PA = s.groupby(["col", "pos", "animal", "epoch"]).v.mean().unstack("epoch").reindex(columns=EP)
    save_fd(stem, {"PA": PA, "D": D, "hemi": hemi, "ylab": ylab, "title": title, "acol": _acol()}, FDR)
    draw_dotgrid(stem)


HL = {"right": "CONTRALESIONAL (right)", "left": "IPSILESIONAL (left)"}
N_NOTE = ("Per-animal dots (animal x epoch mean over sessions; lines join an animal's epochs), bar = mean of animals; "
          "* = change vs pre whose paired nested bootstrap CI (analysis_kit.boot_delta, animals > sessions) excludes 0, "
          "(*) = significant but fewer than 3 animals share the sign. Spout columns in canonical order, 'all' = all "
          "spouts pooled. PROVISIONAL, n = 4 mice.")


def figs_AC(A: dict, C: dict) -> None:
    for hemi in ("right", "left"):
        h = "contra" if hemi == "right" else "ipsi"
        D = A["D"]
        dotgrid_data(f"lick_cue_to_lick_A_level_{h}", A["sess"]["mean_rw"], D[(D.measure == "mean_rw") & (D.match == "raw")],
                     hemi, "mean cue->lick\n(dF/F x1e3)",
                     f"NEURAL, baseline = session RESTW: mean dF/F from the cue to the first lick, per trial, ENGAGED trials "
                     f"with an in-window first lick; {HL[hemi]} hemisphere, areas = rows. " + N_NOTE)
        dotgrid_data(f"lick_cue_to_lick_A_level_own_{h}", A["sess"]["mean_own"],
                     D[(D.measure == "mean_own") & (D.match == "raw")], hemi, "cue->lick minus\nown pre-cue",
                     f"NEURAL, baseline = the trial's own pre-cue level (-0.5..0 s): mean dF/F from the cue to the first lick "
                     f"minus the pre-cue level; {HL[hemi]} hemisphere. " + N_NOTE)
        for E in POST:
            dotgrid_data(f"lick_cue_to_lick_A_level_matched_{E}_{h}", A["match"]["mean_rw"],
                         D[(D.measure == "mean_rw") & (D.match == "latency-matched") & (D.epoch == E)], hemi,
                         "matched cue->lick\n(dF/F x1e3)",
                         f"NEURAL, baseline = session RESTW: mean dF/F cue -> first lick, LATENCY-MATCHED pre vs {E} (per "
                         f"animal x spout, latency bins re-weighted to the common part of the two latency distributions; "
                         f"only the pre and {E} columns are matched to each other); {HL[hemi]} hemisphere. " + N_NOTE,
                         cmp=E)
        Dc = C["D"]
        for nm, wl in (("w03", "-0.3..0"), ("w05", "-0.5..0")):
            dotgrid_data(f"lick_cue_to_lick_C_{nm}_{h}", C["sess"][nm], Dc[Dc.measure == nm], hemi,
                         f"lick-aligned {wl} s\n(dF/F x1e3)",
                         f"NEURAL, baseline = session RESTW: lick-aligned mean dF/F {wl} s before the first lick; trials whose "
                         f"window would start before the cue (latency < {wl[1:4]} s) DROPPED (kept fraction in "
                         f"c2l_C_kept_fraction.csv); {HL[hemi]} hemisphere. " + N_NOTE)


def figs_B(T: pd.DataFrame, TPL: dict, B: dict) -> None:
    # (1) pre-stroke Q vs latency with model curves; (2) exponent dot plot per epoch; (3) acute far-contra on pre relation
    kc = [f"{a}_{h}" for h in ("right", "left") for a in KEY]
    BM = B["BM"]
    for tau in TAUS:
        tg = "" if not tau else f"_tau{tau:g}"
        save_fd(f"lick_cue_to_lick_B_pre_models{tg}", {"BM": BM[(BM.tau == tau) & (BM.cls == "all")], "cols": kc,
                                                      "acol": _acol(), "tau": tau}, FDR)
        draw_B_pre(f"lick_cue_to_lick_B_pre_models{tg}")
        for cl in ("all", "far_contra"):
            save_fd(f"lick_cue_to_lick_B_exponents_{cl}{tg}", {"EX": B["EX"][(B["EX"].tau == tau) & (B["EX"].cls == cl)],
                                                               "cols": kc, "acol": _acol(), "cls": cl, "tau": tau}, FDR)
            draw_B_exp(f"lick_cue_to_lick_B_exponents_{cl}{tg}")
    save_fd("lick_cue_to_lick_B_farcontra_on_pre", {"BM": BM[BM.tau == 0], "OPL": B["OPL"][B["OPL"].tau == 0],
                                                    "cols": kc[:4], "acol": _acol()}, FDR)
    draw_B_onpre("lick_cue_to_lick_B_farcontra_on_pre")


QLAB = {"AUC_own": "AUC cue->lick above own pre\n(dF/F x1e3 * s)", "AUC_res": "AUC minus cue-locked\ntemplate",
        "lev_own": "level at lick above\nown pre (-0.1..0 s)", "slope_end": "ramp slope cue->lick\n(level / latency, per s)",
        "mean_own": "mean cue->lick\nabove own pre", "AUC_tpl": "template AUC", "lev_tpl": "template level"}


def draw_B_pre(stem: str) -> None:
    import matplotlib.pyplot as plt
    F = load_fd(stem, FDR)
    BM, cols, acol = F["BM"], F["cols"], F["acol"]
    qs = ["AUC_own", "lev_own", "slope_end", "mean_own"]
    fw = 2.3 * len(cols)
    fig, axs = plt.subplots(len(qs), len(cols), figsize=(fw, 2.2 * len(qs)), squeeze=False, layout="constrained")
    mc = {"M1 accumulate-to-bound": "tab:red", "M2 fixed level": "tab:blue", "M3 delayed start": "tab:purple"}
    for j, col in enumerate(cols):
        b = BM[(BM.col == col) & (BM.epoch == "pre")]
        for i, q in enumerate(qs):
            ax = axs[i, j]
            for an, g in b.groupby("animal"):
                ax.plot(g.lat, g[q], "-o", color=acol.get(an, "0.5"), ms=2.5, lw=0.7)
            # pooled: mean of animals per bin
            pm = b.groupby("bin").agg(lat=("lat", "median"), v=(q, "mean"), n=("animal", "nunique"))
            pm = pm[pm.n >= 3]
            ax.plot(pm.lat, pm.v, "k-", lw=2)
            tq = {"AUC_own": "AUC_tpl", "lev_own": "lev_tpl"}.get(q)
            if tq:
                tm = b.groupby("bin").agg(lat=("lat", "median"), v=(tq, "mean"), n=("animal", "nunique"))
                tm = tm[tm.n >= 3]
                ax.plot(tm.lat, tm.v, color="tab:green", lw=1.5, ls="--", label="M4 cue-locked template")
            if len(pm) and np.isfinite(pm.v).any():
                k = int(np.argmin(np.abs(np.log(pm.lat) - np.log(pm.lat.median()))))
                L0, V0 = pm.lat.iloc[k], pm.v.iloc[k]
                xx = np.geomspace(pm.lat.min(), pm.lat.max(), 30)
                for m, be in PRED[q].items():
                    ax.plot(xx, V0 * (xx / L0) ** be, color=mc[m], lw=1, ls=":", label=m)
            ax.set_xscale("log")
            ax.axhline(0, color="0.7", lw=0.5)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(col.replace("_right", " contra").replace("_left", " ipsi"), fontsize=8)
            if j == 0:
                ax.set_ylabel(QLAB[q], fontsize=7)
            if i == len(qs) - 1:
                ax.set_xlabel("first-lick latency (s, log)", fontsize=7)
    h, lab = axs[0, 0].get_legend_handles_labels()
    h2, l2 = axs[1, 0].get_legend_handles_labels()
    keep = dict(zip(lab + l2, h + h2))
    fig.legend(keep.values(), keep.keys(), loc="outside lower center", ncol=5, fontsize=7, frameon=False)
    fig.suptitle(_wrap("NEURAL, PRE-STROKE, baseline = the trial's own pre-cue level (-0.5..0 s; restw cancels): quantity vs "
                 "first-lick latency, latency-binned means (all spouts; sessions pooled per animal, bins >= 5 trials), "
                 "coloured = animals, black = mean of animals (bins with >= 3 animals). Dotted = model predictions scaled "
                 "through the black curve at the median latency: M1 accumulate-to-bound (AUC fixed), M2 fixed level "
                 "crossing (level fixed), M3 delayed start with a fixed trajectory; dashed green = M4, the cue-locked "
                 "template (trials not yet licked by t + 0.3 s) at that latency." + (
                     f" DECONVOLVED trace s = y + {F['tau']} dy/dt." if F["tau"] else " Raw GCaMP6s dF/F (slow, lagged)."),
                 fw, 8), fontsize=8)
    fig.savefig(FIGS / f"{stem}{SFX}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def draw_B_exp(stem: str) -> None:
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE
    F = load_fd(stem, FDR)
    EX, cols, acol = F["EX"], F["cols"], F["acol"]
    qs = list(PRED)
    mc = {"M1 accumulate-to-bound": "tab:red", "M2 fixed level": "tab:blue", "M3 delayed start": "tab:purple"}
    fw = 2.1 * len(cols)
    fig, axs = plt.subplots(len(qs), len(cols), figsize=(fw, 2.0 * len(qs)), squeeze=False, sharey="row",
                            layout="constrained")
    for j, col in enumerate(cols):
        for i, q in enumerate(qs):
            ax = axs[i, j]
            e = EX[(EX.col == col) & (EX.q == q)]
            for an, g in e.groupby("animal"):
                g = g.set_index("epoch").reindex(EP)
                ax.plot(np.arange(len(EP)) + 0.04 * (sorted(acol).index(an) - 1.5), g.beta, "o", color=acol.get(an, "0.5"),
                        ms=3.5)
            mu = e.groupby("epoch").beta.mean().reindex(EP)
            ax.scatter(range(len(EP)), mu, marker="_", s=200, c=[EPOCH_LINE[x] for x in EP], lw=2.2, zorder=4)
            tq = {"AUC_own": "AUC_tpl", "lev_own": "lev_tpl"}.get(q)
            if tq:
                t = EX[(EX.col == col) & (EX.q == tq)].groupby("epoch").beta.mean().reindex(EP)
                ax.plot(range(len(EP)), t, color="tab:green", ls="--", lw=1.2, label="M4 template")
            for m, be in PRED[q].items():
                ax.axhline(be + 0.03 * list(mc).index(m), color=mc[m], ls=":", lw=1.2, label=m)
            ax.set_xticks(range(len(EP)), [x[:3] for x in EP], fontsize=6)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(col.replace("_right", " contra").replace("_left", " ipsi"), fontsize=8)
            if j == 0:
                ax.set_ylabel(f"exponent beta\n{QLAB[q].splitlines()[0]}", fontsize=7)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=5, fontsize=7, frameon=False)
    fig.suptitle(_wrap(f"NEURAL, baseline = own pre-cue level: latency EXPONENT beta (quantity ~ latency^beta; log-log slope of "
                 f"latency-binned means, per animal x epoch, {'all spouts' if F['cls'] == 'all' else 'far-contra trials'}"
                 f"; bins with positive means only) against the model predictions (dotted). Dots = animals, bar = mean. "
                 f"Pre-stroke latencies span only ~0.1-0.4 s, so exponents are noisy." + (
                     f" DECONVOLVED (tau {F['tau']} s)." if F["tau"] else " Raw GCaMP6s dF/F."), fw, 8), fontsize=8)
    fig.savefig(FIGS / f"{stem}{SFX}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def draw_B_onpre(stem: str) -> None:
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE
    F = load_fd(stem, FDR)
    BM, cols, acol = F["BM"], F["cols"], F["acol"]
    qs = ["lev_own", "AUC_own", "lev_rw"]
    ylab = {"lev_own": "level at lick above own pre", "AUC_own": "AUC cue->lick above own pre",
            "lev_rw": "level at lick above RESTW"}
    ans = sorted(acol)
    fw = 2.4 * len(ans)
    fig, axs = plt.subplots(len(qs) * len(cols), len(ans), figsize=(fw, 1.9 * len(qs) * len(cols)),
                            squeeze=False, layout="constrained")
    for ci, col in enumerate(cols):
        for qi, q in enumerate(qs):
            i = ci * len(qs) + qi
            for j, an in enumerate(ans):
                ax = axs[i, j]
                b = BM[(BM.col == col) & (BM.animal == an)]
                pre = b[(b.epoch == "pre") & (b.cls == "all")]
                ax.plot(pre.lat, pre[q], "-o", color="k", ms=2.5, lw=1.2, label="pre, all spouts")
                for ep in POST:
                    fc = b[(b.epoch == ep) & (b.cls == "far_contra")]
                    ax.plot(fc.lat, fc[q], "-o", color=EPOCH_LINE[ep], ms=2.5, lw=0.9, label=f"{ep}, far contra")
                ax.set_xscale("log")
                ax.axhline(0, color="0.7", lw=0.5)
                ax.tick_params(labelsize=6)
                if i == 0:
                    ax.set_title(an, fontsize=8, color=acol[an])
                if j == 0:
                    ax.set_ylabel(f"{col.replace('_right', ' contra')}\n{ylab[q]}", fontsize=6)
                if i == len(axs) - 1:
                    ax.set_xlabel("first-lick latency (s, log)", fontsize=7)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle(_wrap("NEURAL, contralesional key areas: do slow post-stroke FAR-CONTRA first licks sit on the PRE-stroke "
                 "latency relation? Latency-binned means (bins >= 5 trials) per animal: black = pre, all spouts; colours = "
                 "post-stroke epochs, far-contra trials. Baselines: own pre-cue level (-0.5..0 s) or session RESTW (row "
                 "label). Pre latencies rarely exceed ~0.5 s, so slow-trial comparisons extrapolate the pre relation "
                 "(c2l_B_onpre_summary.csv).", fw, 8), fontsize=8)
    fig.savefig(FIGS / f"{stem}{SFX}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ SIMPLE page versions (Priya 10-29: "S12 is WAY too
# many panels"). Plotting only, from the saved figure data; no new statistics (CIs / stars / residuals are the saved ones).
MCOL = {"M1 accumulate-to-bound": "tab:red", "M2 fixed level": "tab:blue", "M3 delayed start": "tab:purple"}
PRE_RANGE = (0.1, 0.6)          # where pre-stroke first-lick latencies sit (beyond it, the pre relation is extrapolated)


def draw_threshold_simple(src: str = "lick_cue_to_lick_B_farcontra_on_pre",
                          out: str = "lick_cue_to_lick_threshold_simple") -> None:
    """Contralesional ALM / tjM1 / SSp-m / SSp-n POOLED (per animal x epoch x class x latency bin: mean of the four
    areas' saved binned means; then mean of animals, bins with >= 3 animals). Level at lick, AUC, ramp slope vs latency."""
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE
    F = load_fd(src, FDR)
    BM, cols = F["BM"], F["cols"]
    qs = ["lev_own", "AUC_own", "slope_end"]
    b = BM[(BM.tau == 0) & BM.col.isin(cols)]
    P = b.groupby(["animal", "epoch", "cls", "bin"]).agg(lat=("lat", "median"), nc=("col", "nunique"),
                                                         **{q: (q, "mean") for q in qs}).reset_index()
    P = P[P.nc == len(cols)]    # a pooled bin needs all four areas

    def cohort(ep, cl, q):
        g = P[(P.epoch == ep) & (P.cls == cl)].groupby("bin").agg(lat=("lat", "median"), v=(q, "mean"),
                                                                  n=("animal", "nunique"))
        return g[g.n >= 3]
    ylab = {"lev_own": "(a) LEVEL AT THE LICK\nabove own pre-cue (dF/F x1e3)",
            "AUC_own": "(b) AUC cue -> lick above own pre\n(dF/F x1e3 * s)",
            "slope_end": "(c) ramp slope cue -> lick\n(level / latency, per s; log)"}
    xlo, xhi = 0.07, 3.0
    fig, axs = plt.subplots(1, len(qs), figsize=(13.5, 4.6), layout="constrained")
    for ax, q in zip(axs, qs):
        ax.axvspan(PRE_RANGE[1], xhi, color="0.93", zorder=0, lw=0)
        ax.axhline(0, color="0.7", lw=0.5)
        for _an, g in P[(P.epoch == "pre") & (P.cls == "all")].groupby("animal"):
            v = g[q].where(g[q] > 0) if q == "slope_end" else g[q]
            ax.plot(g.lat, v, "-", color="k", lw=0.7, alpha=0.3)
        for ep in POST:
            c = cohort(ep, "all", q)
            ax.plot(c.lat, c.v, "-o", color=EPOCH_LINE[ep], lw=1.4, ms=3, label=f"{ep}, all spouts")
        fc = P[(P.epoch == "acute") & (P.cls == "far_contra")]
        ax.plot(fc.lat, fc[q].where(fc[q] > 0) if q == "slope_end" else fc[q], "o", mfc="none",
                mec=EPOCH_LINE["acute"], ms=3.5, mew=0.6, alpha=0.6, label="acute far contra, per animal")
        c = cohort("acute", "far_contra", q)
        ax.plot(c.lat, c.v, "--o", color=EPOCH_LINE["acute"], mfc="white", lw=1.6, ms=4.5,
                label="acute far contra, mean")
        pm = cohort("pre", "all", q)
        ax.plot(pm.lat, pm.v, "-o", color="k", lw=2.4, ms=4, label="pre-stroke, all spouts", zorder=5)
        if len(pm) and np.isfinite(pm.v).any():   # model predictions scaled through the pre curve at its median latency
            k = int(np.argmin(np.abs(np.log(pm.lat) - np.log(pm.lat.median()))))
            L0, V0 = pm.lat.iloc[k], pm.v.iloc[k]
            xx = np.geomspace(xlo, xhi, 50)
            for m, be in PRED[q].items():
                ax.plot(xx, V0 * (xx / L0) ** be, color=MCOL[m], lw=1.3, ls=":", label=m)
        ax.set_xscale("log")
        ax.set_xlim(xlo, xhi)
        if q == "slope_end":
            ax.set_yscale("log")
        ax.set_xticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"])
        ax.set_xlabel("first-lick latency (s, log)", fontsize=8)
        ax.set_ylabel(ylab[q], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("predicted exponent (quantity ~ latency^b): " + ", ".join(
            f"{m.split()[0]} {be:+d}" for m, be in PRED[q].items()), fontsize=7.5)
        ax.text(0.98, 0.98, "beyond pre\nlatency range", transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="0.4")
    for ax, q in zip(axs, qs):   # the extended M1 curve can run off scale: keep the data's range
        d = np.concatenate([ln.get_ydata() for ln in ax.lines if ln.get_linestyle() != ":"
                            and len(np.atleast_1d(ln.get_ydata())) > 2]).astype(float)
        d = d[np.isfinite(d)]
        if q == "slope_end":
            d = d[d > 0]
            ax.set_ylim(d.min() / 1.6, d.max() * 1.6)
        elif d.size:
            pad = 0.12 * (d.max() - d.min())
            ax.set_ylim(min(0, d.min()) - pad, d.max() + 2.2 * pad)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    O = F["OPL"]
    o = O[(O.tau == 0) & (O.ref == "all") & (O.epoch == "acute") & (O.speed == "slow>0.5") & O.col.isin(cols)]
    def _rs(q):
        r = o[o.q == q].set_index("col").reindex(cols)
        return ", ".join(f"{c.split('_')[0]} {v.res:+.1f}{'*' if v.lo > 0 or v.hi < 0 else ''}" for c, v in r.iterrows())
    fig.suptitle(_wrap(
        "Cue -> first lick is NOT accumulation to a bound (M1): the LEVEL AT THE LICK does not fall with latency (a; M1 "
        "predicts 1/latency) and the ramp slope falls ~1/latency, not 1/latency^2 (c). The AUC stays small and flat "
        "instead of growing with latency (b), which also fails M2 (a ramp from the cue to a fixed level); a DELAYED START "
        "of a fixed trajectory (M3) fits all three. Slow acute far-contra licks reach the pre level at the lick but "
        "with LESS area (residual from each area's own pre relation, acute far contra > 0.5 s, level: " + _rs("lev_own")
        + "; AUC: " + _rs("AUC_own") + "; * = CI excludes 0, c2l_B_onpre_summary.csv). "
        "NEURAL, contralesional ALM, tjM1, SSp-m, SSp-n POOLED (mean of the four areas per animal x latency bin), "
        "baseline = the trial's own pre-cue level (-0.5..0 s), raw GCaMP6s dF/F (slow, lagged). Thin black = pre per "
        "animal, thick = mean of animals (bins >= 3 animals); dotted = models scaled through the pre curve at its median "
        "latency. CAVEAT: pre-stroke latencies sit at ~0.1-0.6 s, so the grey region (and every slow-trial comparison) extrapolates the pre relation. PROVISIONAL, n = 4 mice."
        + (" Trials = CURRENT engaged gate, sessions with a RESTnl level (rest-independent quantity)." if SFX else ""),
        13.5, 9), fontsize=9)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{out}{SFX}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def draw_A_simple(src: str = "lick_cue_to_lick_A_level_contra", out: str = "lick_cue_to_lick_A_level_simple") -> None:
    """Change vs pre of the mean cue->lick level (above RESTW): key areas x {all spouts, far contra}; contra filled,
    ipsi open; per-animal dots, saved boot_delta point + CI, stars."""
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE
    F = load_fd(src, FDR)
    D, acol = F["D"], F["acol"]
    ans = sorted(acol)
    rows = [("all", "all spouts"), ("far_contra", "far contra only")]
    fig, axs = plt.subplots(len(rows), len(KEY), figsize=(12, 6.4), sharey="row", squeeze=False, layout="constrained")
    for i, (p, pl) in enumerate(rows):
        for j, a in enumerate(KEY):
            ax = axs[i, j]
            ax.axhline(0, color="0.6", lw=0.6)
            for h, off, filled in (("right", -0.17, True), ("left", 0.17, False)):
                dd = D[(D.col == f"{a}_{h}") & (D.pos == p)].set_index("epoch").reindex(POST)
                for x, (E, r) in enumerate(dd.iterrows()):
                    if not np.isfinite(r.delta):
                        continue
                    xc = x + off
                    for k, an in enumerate(ans):
                        v = r.get(f"d_{an}")
                        if v is not None and np.isfinite(v):
                            ax.plot(xc + 0.035 * (k - 1.5), v, "o", ms=3.2, color=acol[an],
                                    mfc=acol[an] if filled else "none", mew=0.8, alpha=0.85)
                    ax.errorbar(xc, r.delta, yerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="s", ms=6, capsize=3,
                                color=EPOCH_LINE[E], mfc=EPOCH_LINE[E] if filled else "white", mew=1.4, lw=1.4, zorder=4)
                    if r.sig:
                        ax.text(xc, 1.0, "*" if not r.flag_lt3 else "(*)", ha="center", va="bottom", fontsize=11,
                                color=EPOCH_LINE[E], transform=ax.get_xaxis_transform(), clip_on=False)
            ax.set_xticks(range(len(POST)), POST, fontsize=7)
            ax.set_xlim(-0.55, len(POST) - 0.45)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(a, fontsize=10, pad=14)
            if j == 0:
                ax.set_ylabel(restnl.relabel(f"{pl.upper()}\nchange vs pre, mean cue->lick\nabove RESTW (dF/F x1e3)")
                              if SFX else f"{pl.upper()}\nchange vs pre, mean cue->lick\nabove RESTW (dF/F x1e3)", fontsize=8)
    hs = [plt.Line2D([], [], color=c, marker="o", ls="", ms=4, label=an) for an, c in sorted(acol.items())]
    hs += [plt.Line2D([], [], color="0.3", marker="s", ls="", ms=6, label="contralesional (filled)"),
           plt.Line2D([], [], color="0.3", marker="s", mfc="white", ls="", ms=6, label="ipsilesional (open)")]
    fig.legend(handles=hs, loc="outside lower center", ncol=len(hs), fontsize=8, frameon=False)
    fig.suptitle(_wrap(_rl(
        "NEURAL, baseline = session RESTW: change from pre of the mean dF/F from the cue to the first lick (ENGAGED "
        "trials with an in-window first lick), orofacial areas; contralesional (left of each pair, filled) and "
        "ipsilesional (right, open). Dots = per-animal change (colour = animal); square + bar = paired nested bootstrap "
        "point and 95% CI (analysis_kit.boot_delta, animals > sessions); * = CI excludes 0, (*) = significant but fewer "
        "than 3 animals share the sign. Not latency-matched (slower post-stroke licks lengthen the window). "
        "PROVISIONAL, n = 4 mice."), 12, 9), fontsize=9)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{out}{SFX}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def draw_simple() -> None:
    draw_threshold_simple()
    draw_A_simple()


DOT_STEMS = ([f"lick_cue_to_lick_A_level_{h}" for h in ("contra", "ipsi")]
             + [f"lick_cue_to_lick_A_level_own_{h}" for h in ("contra", "ipsi")]
             + [f"lick_cue_to_lick_A_level_matched_{E}_{h}" for E in POST for h in ("contra", "ipsi")]
             + [f"lick_cue_to_lick_C_{w}_{h}" for w in ("w03", "w05") for h in ("contra", "ipsi")])


def main() -> int:
    global SFX
    from wfield_local import parallel
    parallel.pin_blas()
    var = restnl.variant()
    SFX = restnl.SFX if var == "restnl" else ""
    if plot_only():
        for s in DOT_STEMS:
            draw_dotgrid(s)
        for tau in TAUS:
            tg = "" if not tau else f"_tau{tau:g}"
            draw_B_pre(f"lick_cue_to_lick_B_pre_models{tg}")
            for cl in ("all", "far_contra"):
                draw_B_exp(f"lick_cue_to_lick_B_exponents_{cl}{tg}")
        draw_B_onpre("lick_cue_to_lick_B_farcontra_on_pre")
        draw_simple()
        return 0
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    T, TPL, meta = build(jobs, var)
    OUT.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUT / _o("c2l_sessions.csv"), index=False)
    T[T.tau == 0].to_csv(OUT / _o("c2l_trials.csv.gz"), index=False)
    print(f"== {meta.label.nunique()} sessions (restw_ok); in-window trials {meta.n_inw.sum()}, late {meta.n_late.sum()}, "
          f"never {meta.n_never.sum()}")
    nt = T[(T.tau == 0) & (T.col == COLS[0])]
    print("== median latency (s) per animal x epoch, far_contra / other\n"
          + nt.groupby(["cls", "animal", "epoch"]).lat.median().unstack("epoch").reindex(columns=EP).round(3).to_string())
    rng = np.random.default_rng(0)
    A = task_A(T, rng)
    C = task_C(T, rng)
    B = task_B(T, rng)
    report(A, B, C)
    FIGS.mkdir(parents=True, exist_ok=True)
    figs_AC(A, C)
    figs_B(T, TPL, B)
    draw_simple()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
