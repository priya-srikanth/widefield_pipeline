"""REV-UP BEFORE THE FIRST LICK -- NEURAL SIDE (Priya 10-29: after the stroke the animals seem to 'rev up' before
licking the FAR-CONTRA spout: much longer cue -> first-lick latencies, sometimes the first lick only after the spout has
left).

QUESTIONS
 (4) In SLOW trials, does cortical activity ramp MORE SLOWLY, START LATER, or reach a DIFFERENT LEVEL before the first
     lick? Is there a common level at lick onset ('ramp to threshold', as for ALM preparatory activity)? Far-contra vs
     the other spouts; pre vs the post-stroke epochs; contralesional (right, image right) vs ipsilesional (left) cortex.
     LATE first licks (first lick after the trial stop = after the spout has left; Priya 10-29 addition) as their own
     group: do their lick-aligned responses look like normal post-cue first licks or like spontaneous ITI licks (bout
     starts after >= 2 s without a lick)?
 (5b) Does a trial's PRE-CUE level (-1..0 s before the cue, above the session's RESTW baseline; Fig 14.21,
     lick_precue_restw.py) predict that trial's latency, within session x spout? And the session's level?

METHOD
  traces : the area construction of lick_precue_restw_traces / lick_cued_ramp.wide_session / iti_licks.extract (ROI v8
           masks, area means of U_atlas on the adopted SVT, slow-drift basis removed, dF/F x1e3, NO per-trial baseline)
           MINUS the session's RESTW baseline read from <OUT_W>/precue_restw/<label>.npz (same numbers as Fig 14.21;
           restw_ok False sessions are left out of the statistics). Whole-session series, so the cue-aligned window can
           run -3..+10 s (into the ITI; the stored cued_wide files stop at +3 s) and late first licks can be aligned.
  areas  : ALM, tjM1, SSp-m, SSp-n, SSs per hemisphere; controls RSPd (both) and VISp (contralesional ROI only).
  trials : ENGAGED trials (behaviour table engaged & finite cue) = the precue_restw trial set (checked equal).
  first lick per trial (computed HERE -- revup_behaviour's table was not on disk when this ran; if
           <OUT_W>/revup/revup_trials.csv.gz exists, `--join` reports the agreement):
           in_window = first cam4 behaviour lick onset in cue .. trial stop (the iti_licks._first_lick_times rule);
           late      = no lick in cue .. stop, first video lick onset (licks_all.csv, context iti, any segment) in
                       stop .. next trial's position strobe (spout retracted / moving; Priya 10-26/10-29);
           never     = neither.
  latency groups: QUARTILES of the in-window latencies within animal x spout x epoch (sessions pooled for the cut
           points), plus 'late'. Fixed bins comparable across epochs: <0.25, 0.25-0.5, 0.5-1.0, 1.0-stop s, late.
  per-trial-group means per session x spout class (far_contra / other five) x group; then per-session metrics:
       L_level   lick-aligned level -0.1..+0.05 s (restw-referenced)  -> common level at lick = ramp to threshold
       L_slope   lick-aligned linear slope -0.5..0 s (per s)           -> slower ramp
       C_slope   cue-aligned linear slope 0..0.5 s                     -> slower ramp from the cue
       C_pre     cue-aligned level -0.5..0 s
       t_half    lick-aligned time of the last upward 50 % crossing between C_pre and L_level (-2..0 s; relative)
                 -> delayed start predicts the same L_slope / t_half in slow and fast trials with a flat cue-aligned
                    stretch; a slower ramp predicts lower L_slope / C_slope and earlier t_half in slow trials.
     The verdict reads Q4 (slowest) minus Q1 (fastest) per animal (rule 8) before any bootstrap. C_slope Q4 - Q1 is
     CONFOUNDED (Q1's own lick response falls inside 0..0.5 s); the cue-locked slope is read on the slow-only contrast
     'gt1-0.5to1' (fixed bins >1.0 s minus 0.5-1.0 s: no first lick before +0.5 s in either group).
  late vs normal vs spontaneous: lick-aligned (-3..+2 s) means of in-window first licks, late first licks and ITI bout
           starts (iti_licks session file bout_t with gap >= GAP_MAIN = 2 s, excluding the late first licks); metrics
           ramp = mean(-0.2..0) - mean(-1.5..-1.0), resp = mean(0.1..0.5) - mean(-1.5..-1.0), and the shape correlation
           (-1.5..+1 s) of the late mean with the session's normal and spontaneous means.
  pre-cue level vs latency: per session x spout (>= 8 trials with a first lick), Spearman rho(trial pre-cue level,
           latency rank; late licks rank after every in-window lick, latency = their onset - cue); Fisher-z mean per
           animal x epoch x spout class. Control: trials with NO lick in -2..0 s before the cue (pre-cue licking raises
           the level AND delays the cue/lick). Session level: per animal x epoch, r(session far-contra pre-cue level,
           session median far-contra latency) over sessions.
  stats  : per-animal tables first (ground rule 8); changes vs pre with analysis_kit.boot_delta (paired nested
           animals > sessions); levels with boot_ci. AUGUST-only pre (il.labels()).
  GCaMP6s CAVEAT: fluorescence lags spikes by ~0.1-0.2 s and decays over ~0.5-1 s (widefield smoother still), so the
           comparison is of SLOPES / SHAPES between latency groups (same lag in both), not absolute onset times; the
           previous trial's / pre-cue licks' responses decay into the window. DECONVOLUTION SENSITIVITY: every metric is
           recomputed on s(t) = x(t) + tau dx/dt (inverse of a first-order exponential kernel, tau 0.6 and 1.0 s; applied to
           the group-mean traces -- it is linear, so it commutes with the averaging; 0.15 s boxcar before the derivative).

OUTPUTS (under <OUT_W>/revup/): sessions/<label>.npz (per-trial traces cache), revup_neural_trials.csv.gz (per trial:
  latency, phase, group, pre-cue level per area), revup_neural_metrics.csv.gz (session x spout class x group x area x
  deconv metrics), revup_neural_per_animal.csv (rule 8 tables), revup_neural_tests.csv (boot), revup_late_compare.csv,
  revup_precue_latency.csv (+ _per_animal, _session), figdata/*.pkl; figures in <OUT_W>/figures/revup_neural_*.png.
Usage: python scripts/revup_neural.py [--jobs 8] [--plot-only] [--join] [--rest-variant restnl]
RESTnl VARIANT (`--rest-variant restnl`, default restw unchanged; plan docs/status/RESTNL_ROLLOUT_PLAN_20261101.md G1): no
re-extraction -- every stored trace (trace - RESTW, sessions/<label>.npz) is shifted at LOAD by + RESTW - RESTnl per session
x area (exact; RESTnl = the LICK-FREE rest level of scripts/restnl.py, rule wr by default); sessions without a RESTnl level
(ok False) are left out. A flat per-session shift leaves every WITHIN-session-and-group shape metric unchanged (slopes,
Q4 - Q1 / slow-only contrasts of slopes and t_half, shape correlations, within-session rho); only LEVELS (traces, L_level /
C_pre, pre-cue level between sessions -> the session-level r) move. Every output name (csv, figdata, png) gets the suffix
_restnl (restnl.SFX); figure text is relabelled (restnl.relabel).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import restnl  # noqa: E402

OUT = il.OUT_W / "revup"
SESS = OUT / "sessions"
FIGS = il.OUT_W / "figures"
VERSION = 2
LAGS_C = np.round(np.arange(-3.0, 10.0001, 0.05), 3)     # cue-aligned, into the ITI
LAGS_L = np.round(np.arange(-3.0, 2.0001, 0.05), 3)      # lick-aligned
AREAS = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs", "RSPd", "VISp"]
CTRL = {"RSPd", "VISp"}
HEMI = (("right", "contralesional (right)"), ("left", "ipsilesional (left)"))
COLS = [f"{a}_{h}" for a in AREAS for h, _ in HEMI if not (a == "VISp" and h == "left")]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
GAP_SPONT = il.GAP_MAIN
SFX = ""                        # "_restnl" under --rest-variant restnl (set in main; every output name carries it)


def _rl(txt: str) -> str:
    return restnl.relabel(txt) if SFX else txt


def _o(name: str) -> Path:
    """Output path under OUT with the variant suffix before the extension(s)."""
    stem, dot, ext = name.partition(".")
    return OUT / f"{stem}{SFX}{dot}{ext}"


_SHIFT: dict = {}


def _shift(x: str):
    """Per-column shift RESTW - RESTnl for session x (None = restw variant, no shift; False = no RESTnl level)."""
    if not SFX:
        return None
    if x not in _SHIFT:
        nl, ok = restnl.level(x, COLS)
        if not ok:
            _SHIFT[x] = False
        else:
            src = np.load(il.OUT_W / "precue_restw" / f"{x}.npz")
            sc = [str(c) for c in src["cols"]]
            _SHIFT[x] = (np.asarray(src["restw"], float)[[sc.index(c) for c in COLS]] - nl).astype(np.float32)
    return _SHIFT[x]


def _sess(x: str) -> dict | None:
    """The session cache with the traces on the active baseline (restw stored; restnl = + RESTW - RESTnl, exact).
    None = left out (restw_ok False, or no RESTnl level under the restnl variant)."""
    z = np.load(SESS / f"{x}.npz")
    if not bool(z["restw_ok"]):
        return None
    sh = _shift(x)
    if sh is False:
        return None
    d = {k: z[k] for k in z.files}
    for k in ("Xc", "Xl", "Xs"):
        v = d[k].astype(np.float32)
        if sh is not None and v.size:
            v = v + sh[None, None, :]
        d[k] = v
    return d


# ------------------------------------------------------------------------------------------------ per-session extraction
def session_worker(label: str) -> dict:
    """One session (module-level for spawn) -> <OUT_W>/revup/sessions/<label>.npz."""
    f = SESS / f"{label}.npz"
    if f.exists():
        z = np.load(f)
        if int(z["version"]) == VERSION:
            return {"label": label, "cached": True}
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    src = np.load(il.OUT_W / "precue_restw" / f"{label}.npz")
    d = il._load(label)
    # == lick_precue_restw(_traces).session_worker's trace construction
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    if cols != list(d["cols"]):
        raise ValueError(f"{label}: ROI columns differ from the session file")
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    scols = [str(c) for c in src["cols"]]
    idx = [cols.index(c) for c in COLS]
    R = R[:, idx] - np.asarray(src["restw"], float)[[scols.index(c) for c in COLS]][None, :]
    B = il._behaviour(label)
    T = B.trials
    Eg = T[T.engaged.astype(bool) & np.isfinite(T.cue_s.to_numpy(float))].reset_index(drop=True)
    pos = np.asarray(Eg.pos.astype(str).to_numpy(), str)
    if not np.array_equal(pos, src["pos"].astype(str)):
        raise ValueError(f"{label}: engaged trials differ from the precue_restw cache")
    cue = Eg.cue_s.to_numpy(float)
    stop = Eg.stop_s.to_numpy(float)
    stop = np.where(np.isfinite(stop), stop, cue + 3.5)
    # next trial's strobe (all trials, not only engaged): the spout of the NEXT trial is in place from there
    st_all = np.sort(T.strobe_s.to_numpy(float)[np.isfinite(T.strobe_s.to_numpy(float))])
    j = np.searchsorted(st_all, cue, side="right")
    nxt = np.where(j < len(st_all), st_all[np.minimum(j, len(st_all) - 1)], ft[-1])
    # licks: behaviour (cam4 in-trial) onsets + whole-session video licks (licks_all.csv, ITI included)
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["context", "daq_on_s"])
    bo = np.sort(np.asarray(B.lick_on_s, float)[np.isfinite(B.lick_on_s)])
    ao = np.sort(A.daq_on_s.to_numpy(float)[np.isfinite(A.daq_on_s.to_numpy(float))])
    iti_on = np.sort(A.daq_on_s.to_numpy(float)[(A.context == "iti").to_numpy() & np.isfinite(A.daq_on_s.to_numpy(float))])
    t_all = np.unique(np.concatenate([bo, ao]))
    n = len(cue)
    fl, phase = np.full(n, np.nan), np.full(n, "never", dtype=object)
    quiet, npre = np.full(n, np.nan), np.zeros(n, int)
    for i in range(n):
        k = np.searchsorted(bo, cue[i])
        if k < len(bo) and bo[k] <= stop[i]:
            fl[i], phase[i] = bo[k], "in_window"
        else:
            k = np.searchsorted(iti_on, stop[i], side="right")
            if k < len(iti_on) and iti_on[k] < nxt[i]:
                fl[i], phase[i] = iti_on[k], "late"
        k = np.searchsorted(t_all, cue[i]) - 1
        quiet[i] = cue[i] - t_all[k] if k >= 0 else np.inf
        npre[i] = int(((t_all >= cue[i] - 2.0) & (t_all < cue[i])).sum())
    # gap before a late first lick (a late lick right after other ITI licks is mid-bout, not a bout start)
    late_gap = np.full(n, np.nan)
    for i in np.flatnonzero(phase == "late"):
        k = np.searchsorted(t_all, fl[i] - 0.05) - 1
        late_gap[i] = fl[i] - t_all[k] if k >= 0 else np.inf

    def samp(t, lags):
        X = il._sample(R, ft, t, lags).astype(np.float32)
        out = (t[:, None] + lags[None, :] < ft[0]) | (t[:, None] + lags[None, :] > ft[-1])
        X[out] = np.nan
        return X
    Xc = samp(cue, LAGS_C)
    ok = np.isfinite(fl)
    Xl = np.full((n, len(LAGS_L), len(COLS)), np.nan, np.float32)
    if ok.any():
        Xl[ok] = samp(fl[ok], LAGS_L)
    # spontaneous ITI bout starts (gap >= 2 s), not one of this session's late first licks
    bt, bg = np.asarray(d["bout_t"], float), np.asarray(d["bout_gap"], float)
    lt = fl[phase == "late"]
    is_late = np.array([np.any(np.abs(lt - t) < 0.05) for t in bt], bool) if len(lt) else np.zeros(len(bt), bool)
    sp = bt[(bg >= GAP_SPONT) & ~is_late]
    Xs = samp(sp, LAGS_L) if len(sp) else np.zeros((0, len(LAGS_L), len(COLS)), np.float32)
    # sanity vs precue_restw: trial -1..0 s mean (p10 = the trial's raw -1..0 s mean; restw subtracted downstream)
    w = (LAGS_C >= -1.0 - 1e-9) & (LAGS_C <= 0.0 + 1e-9)
    ci = [scols.index(c) for c in COLS]
    p10 = np.asarray(src["p10"], float)[:, ci] - np.asarray(src["restw"], float)[ci][None, :]
    maxdiff = float(np.nanmax(np.abs(np.nanmean(Xc[:, w], 1) - p10)))
    f.parent.mkdir(parents=True, exist_ok=True)
    f16 = np.float16
    np.savez_compressed(f, version=VERSION, cols=np.array(COLS), pos=pos, cue=cue, stop=stop, nxt=nxt, fl=fl,
                        phase=np.asarray(phase, str), quiet=quiet, npre=npre, late_gap=late_gap,
                        Xc=Xc.astype(f16), Xl=Xl.astype(f16), Xs=Xs.astype(f16), sp_t=sp,
                        restw_ok=bool(src["restw_ok"]), animal=str(d["animal"]), epoch=str(d["epoch"]),
                        maxdiff=maxdiff)
    return {"label": label, "cached": False, "n": n, "n_in": int((phase == "in_window").sum()),
            "n_late": int((phase == "late").sum()), "n_spont": len(sp), "maxdiff": maxdiff}


def extract(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    labs = il.labels()
    res, fails = ak.fan_sessions(labs, session_worker, jobs=min(jobs, 8))
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    V = pd.DataFrame([v for _x, v in res])
    if "maxdiff" in V:
        print(f"== sanity: max |trace -1..0 s mean - precue_restw p10| = {V.maxdiff.max():.2e}")


# ------------------------------------------------------------------------------------------------ analysis
KEY = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]
CLS = ("far_contra", "other")
QG = ["Q1", "Q2", "Q3", "Q4", "late"]
FIX_EDGES = [0.0, 0.25, 0.5, 1.0, np.inf]
FG = ["<0.25", "0.25-0.5", "0.5-1.0", ">1.0", "late"]
TAUS = (0.0, 0.6, 1.0)                          # 0 = raw
METRICS = ("C_pre", "C_slope", "L_level", "L_rise", "L_slope", "t_half")
TEST_METRICS = ("L_level", "L_slope", "C_slope", "t_half")
MIN_N = 3                                       # trials per session x class x group (late: 2)
N_BOOT = 2000
EP = il.EP4


def _w(lags, a, b):
    return (lags >= a - 1e-9) & (lags <= b + 1e-9)


def _slope(y, lags, a, b):
    w = _w(lags, a, b)
    x = lags[w] - lags[w].mean()
    Y = y[w]
    ok = np.isfinite(Y).all(0)
    out = np.full(y.shape[1], np.nan)
    out[ok] = (x[:, None] * (Y[:, ok] - Y[:, ok].mean(0))).sum(0) / (x ** 2).sum()
    return out


def deconv(X: np.ndarray, tau: float) -> np.ndarray:
    """Inverse of a first-order exponential kernel: s = x + tau dx/dt, after a 0.15 s boxcar (axis 0 = lags, 0.05 s)."""
    if not tau:
        return X
    from scipy.ndimage import uniform_filter1d
    Xs = uniform_filter1d(X, 3, axis=0, mode="nearest")
    return Xs + tau * np.gradient(Xs, 0.05, axis=0)


def metrics(c: np.ndarray, lk: np.ndarray) -> dict:
    """Per column: the ramp metrics of one group-mean cue trace c (LAGS_C x cols) and lick trace lk (LAGS_L x cols)."""
    C_pre = np.nanmean(c[_w(LAGS_C, -0.5, 0.0)], 0)
    L_level = np.nanmean(lk[_w(LAGS_L, -0.1, 0.05)], 0)
    rise = L_level - C_pre
    th = np.full(len(C_pre), np.nan)
    w = _w(LAGS_L, -2.0, 0.0)
    L = LAGS_L[w]
    for j in range(len(C_pre)):
        if not np.isfinite(rise[j]) or rise[j] <= 0:
            continue
        v = lk[w, j] - (C_pre[j] + 0.5 * rise[j])
        below = np.flatnonzero(~(v >= 0))
        th[j] = L[0] if not len(below) else (L[below[-1] + 1] if below[-1] + 1 < len(L) else 0.0)
    return {"C_pre": C_pre, "C_slope": _slope(c, LAGS_C, 0.0, 0.5), "L_level": L_level, "L_rise": rise,
            "L_slope": _slope(lk, LAGS_L, -0.5, 0.0), "t_half": th}


def _load_meta(x: str) -> dict | None:
    return _sess(x)


def trial_table(labels: list) -> pd.DataFrame:
    rows = []
    for x in labels:
        z = _load_meta(x)
        if z is None:
            continue
        Xc = z["Xc"]
        lvl = np.nanmean(Xc[:, _w(LAGS_C, -1.0, 0.0)], 1)
        n = len(z["pos"])
        df = pd.DataFrame({"label": x, "animal": str(z["animal"]), "epoch": str(z["epoch"]), "trial": np.arange(n),
                           "pos": z["pos"].astype(str), "phase": z["phase"].astype(str), "lat": z["fl"] - z["cue"],
                           "stop_lat": z["stop"] - z["cue"], "next_strobe_lat": z["nxt"] - z["cue"],
                           "quiet_before_cue": z["quiet"], "n_licks_m2_0": z["npre"], "late_gap": z["late_gap"]})
        for j, c in enumerate(z["cols"].astype(str)):
            df[f"pc_{c}"] = lvl[:, j]
        rows.append(df)
    T = pd.concat(rows, ignore_index=True)
    T["q"], T["fb"] = "", ""
    m = T.phase == "in_window"

    def qcut(v):
        if v.notna().sum() < 8:
            return pd.Series("", index=v.index)
        return pd.qcut(v.rank(method="first"), 4, labels=QG[:4]).astype(str)
    T.loc[m, "q"] = T[m].groupby(["animal", "pos", "epoch"], group_keys=False).lat.apply(qcut)
    T.loc[m, "fb"] = pd.cut(T.loc[m, "lat"], FIX_EDGES, labels=FG[:4], right=False).astype(str)
    T.loc[T.phase == "late", ["q", "fb"]] = "late"
    return T


def group_traces(T: pd.DataFrame) -> dict:
    """{(label, cls, scheme, group): (cue mean, lick mean, n)} + spontaneous ITI bout starts (cls 'all', 'spont')."""
    G = {}
    for x, t in T.groupby("label", sort=True):
        z = _sess(x)
        Xc, Xl = z["Xc"], z["Xl"]
        for cls in CLS + ("all",):
            mc = np.ones(len(t), bool) if cls == "all" else ((t.pos == "far_contra") == (cls == "far_contra")).to_numpy()
            for scheme, col, gs in (("q", "q", QG), ("fixed", "fb", FG)):
                for g in gs:
                    idx = t.trial.to_numpy()[mc & (t[col] == g).to_numpy()]
                    if len(idx) >= (2 if g == "late" else MIN_N):
                        G[(x, cls, scheme, g)] = (np.nanmean(Xc[idx], 0), np.nanmean(Xl[idx], 0), len(idx))
            idx = t.trial.to_numpy()[mc & (t.phase == "in_window").to_numpy()]
            if len(idx) >= MIN_N:
                G[(x, cls, "all", "in_window")] = (np.nanmean(Xc[idx], 0), np.nanmean(Xl[idx], 0), len(idx))
        Xs = z["Xs"]
        if len(Xs) >= MIN_N:
            G[(x, "all", "spont", "spont")] = (None, np.nanmean(Xs, 0), len(Xs))
    return G


def metric_table(G: dict, T: pd.DataFrame) -> pd.DataFrame:
    meta = T.groupby("label")[["animal", "epoch"]].first()
    rows = []
    for (x, cls, scheme, g), (c, lk, n) in G.items():
        if c is None:
            continue
        for tau in TAUS:
            M = metrics(deconv(c, tau), deconv(lk, tau))
            for j, col in enumerate(COLS):
                rows.append({"label": x, "animal": meta.loc[x, "animal"], "epoch": meta.loc[x, "epoch"], "cls": cls,
                             "scheme": scheme, "group": g, "n": n, "tau": tau, "area": col,
                             **{k: float(v[j]) for k, v in M.items()}})
    return pd.DataFrame(rows)


def contrasts(MT: pd.DataFrame) -> pd.DataFrame:
    """Per session x cls x area x tau: Q4 - Q1 and late - Q1 for each metric (quartile scheme)."""
    Q = MT[MT.scheme == "q"].pivot_table(index=["label", "animal", "epoch", "cls", "area", "tau"], columns="group",
                                         values=list(METRICS))
    out = []
    for m in METRICS:
        for g in ("Q4", "late"):
            if (m, g) in Q.columns and (m, "Q1") in Q.columns:
                d = (Q[(m, g)] - Q[(m, "Q1")]).rename("d").reset_index()
                d["metric"], d["contrast"] = m, f"{g}-Q1"
                out.append(d)
    # SLOW-ONLY contrast (fixed bins, >1.0 s minus 0.5-1.0 s): neither group has licked by +0.5 s, so the cue-aligned
    # 0..0.5 s slope carries no first-lick response in either (in Q4 - Q1 the fast quartile's lick response at ~0.1-0.3 s
    # sits inside that window and makes C_slope Q4 - Q1 negative by construction)
    F = MT[MT.scheme == "fixed"].pivot_table(index=["label", "animal", "epoch", "cls", "area", "tau"], columns="group",
                                             values=list(METRICS))
    for m in METRICS:
        if (m, ">1.0") in F.columns and (m, "0.5-1.0") in F.columns:
            d = (F[(m, ">1.0")] - F[(m, "0.5-1.0")]).rename("d").reset_index()
            d["metric"], d["contrast"] = m, "gt1-0.5to1"
            out.append(d)
    return pd.concat(out, ignore_index=True).dropna(subset=["d"])


def tests(D: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(0)
    rows = []
    sub = D[D.metric.isin(TEST_METRICS) & D.area.isin([f"{a}_{h}" for a in AREAS for h in ("right", "left")])]
    for (cls, area, tau, m, ct), g in sub.groupby(["cls", "area", "tau", "metric", "contrast"], sort=True):
        by = {ep: {a: list(v.d) for a, v in g[g.epoch == ep].groupby("animal", sort=True)} for ep in EP}
        for ep in EP:
            if ct == "late-Q1" and ep == "pre":
                continue
            ci = ak.boot_ci(by[ep], rng, n_boot=N_BOOT) if by[ep] else None
            r = {"cls": cls, "area": area, "tau": tau, "metric": m, "contrast": ct, "epoch": ep,
                 "n_animals": len(by[ep]), "n_sessions": sum(len(v) for v in by[ep].values())}
            if ci is not None:
                r.update(level=ci[0], level_lo=ci[1], level_hi=ci[2])
            if ep != "pre" and ct != "late-Q1":
                dl = ak.boot_delta(by[ep], by["pre"], rng, n_boot=N_BOOT)
                if dl is not None:
                    r.update(change=dl[0], change_lo=dl[1], change_hi=dl[2], change_n_animals=dl[3])
            rows.append(r)
    return pd.DataFrame(rows)


def late_compare(G: dict, T: pd.DataFrame) -> pd.DataFrame:
    """Per session with >= 2 late first licks: late vs normal (in-window first licks, all spouts) vs spontaneous ITI bout
    starts -- ramp / response amplitudes per area and the shape correlation (-1.5..+1 s, key areas both hemispheres)."""
    meta = T.groupby("label")[["animal", "epoch"]].first()
    wb, wr, wp = _w(LAGS_L, -1.5, -1.0), _w(LAGS_L, -0.2, 0.0), _w(LAGS_L, 0.1, 0.5)
    ws = _w(LAGS_L, -1.5, 1.0)
    kc = [COLS.index(f"{a}_{h}") for a in KEY for h in ("right", "left")]
    rows = []
    for x in meta.index:
        if (x, "all", "q", "late") not in G or (x, "all", "spont", "spont") not in G or (x, "all", "all", "in_window") not in G:
            continue
        tr = {"late": G[(x, "all", "q", "late")], "normal": G[(x, "all", "all", "in_window")],
              "spont": G[(x, "all", "spont", "spont")]}
        for tau in TAUS:
            L = {k: deconv(v[1], tau) for k, v in tr.items()}
            Z = {k: (v[ws][:, kc] - v[ws][:, kc].mean(0)).ravel() for k, v in L.items()}

            def r(a, b, Z=Z):
                ok = np.isfinite(Z[a]) & np.isfinite(Z[b])
                return float(np.corrcoef(Z[a][ok], Z[b][ok])[0, 1]) if ok.sum() > 10 else np.nan
            base = {"label": x, "animal": meta.loc[x, "animal"], "epoch": meta.loc[x, "epoch"], "tau": tau,
                    "n_late": tr["late"][2], "n_normal": tr["normal"][2], "n_spont": tr["spont"][2],
                    "r_late_normal": r("late", "normal"), "r_late_spont": r("late", "spont"),
                    "r_normal_spont": r("normal", "spont")}
            for j, col in enumerate(COLS):
                rec = dict(base, area=col)
                for k, v in L.items():
                    b = np.nanmean(v[wb, j])
                    rec[f"ramp_{k}"] = float(np.nanmean(v[wr, j]) - b)
                    rec[f"resp_{k}"] = float(np.nanmean(v[wp, j]) - b)
                    rec[f"lvl_{k}"] = float(np.nanmean(v[_w(LAGS_L, -0.1, 0.05), j]))
                rows.append(rec)
    return pd.DataFrame(rows)


def precue_latency(T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within session x spout Spearman rho(trial pre-cue level, latency) on trials with a first lick (in-window or late);
    'quiet' = trials with no lick in -2..0 s before the cue. Session level: per animal x epoch r over sessions."""
    from scipy.stats import spearmanr
    rows = []
    U = T[T.phase.isin(["in_window", "late"])]
    for (x, p), t in U.groupby(["label", "pos"], sort=True):
        for sel, tt in (("all", t), ("quiet", t[t.n_licks_m2_0 == 0])):
            if len(tt) < 8 or tt.lat.nunique() < 4:
                continue
            for col in COLS:
                v = tt[f"pc_{col}"].to_numpy(float)
                ok = np.isfinite(v)
                if ok.sum() < 8:
                    continue
                rho = spearmanr(v[ok], tt.lat.to_numpy(float)[ok])[0]
                rows.append({"label": x, "animal": t.animal.iloc[0], "epoch": t.epoch.iloc[0], "pos": p, "sel": sel,
                             "area": col, "n": int(ok.sum()), "n_late": int((tt.phase == "late").sum()), "rho": rho,
                             "z": np.arctanh(np.clip(rho, -0.999, 0.999))})
    R = pd.DataFrame(rows)
    # session level (far-contra): session mean pre-cue level vs session median latency, per animal x epoch over sessions
    S = U[U.pos == "far_contra"].groupby(["label", "animal", "epoch"]).agg(
        lat_med=("lat", "median"), late_frac=("phase", lambda s: (s == "late").mean()),
        **{f"pc_{c}": (f"pc_{c}", "mean") for c in COLS}).reset_index()
    srows = []
    for (a, ep), s in S.groupby(["animal", "epoch"], sort=True):
        for col in COLS:
            if len(s) >= 3:
                srows.append({"animal": a, "epoch": ep, "area": col, "n_sessions": len(s),
                              "r_session": float(np.corrcoef(s[f"pc_{col}"], s.lat_med)[0, 1])})
    for a, s in S.groupby("animal", sort=True):          # across ALL sessions of the animal (epoch-confounded)
        for col in COLS:
            srows.append({"animal": a, "epoch": "all", "area": col, "n_sessions": len(s),
                          "r_session": float(np.corrcoef(s[f"pc_{col}"], s.lat_med)[0, 1])})
    return R, pd.concat([S.assign(kind="session"), pd.DataFrame(srows).assign(kind="r")], ignore_index=True)


def curves(G: dict, T: pd.DataFrame, scheme: str, groups: list, tau: float) -> dict:
    """{(cls, epoch, group): (cue mean, lick mean, n_animals)}: session means -> animal means -> mean of animals."""
    meta = T.groupby("label")[["animal", "epoch"]].first()
    acc: dict = {}
    for (x, cls, sc, g), (c, lk, _n) in G.items():
        if sc != scheme or g not in groups:
            continue
        acc.setdefault((cls, meta.loc[x, "epoch"], g, meta.loc[x, "animal"]), []).append(
            (deconv(c, tau) if c is not None else None, deconv(lk, tau)))
    out: dict = {}
    for (cls, ep, g, _a), v in sorted(acc.items()):
        cm = np.nanmean([u[0] for u in v], 0) if v[0][0] is not None else None
        out.setdefault((cls, ep, g), []).append((cm, np.nanmean([u[1] for u in v], 0)))
    return {k: (np.nanmean([u[0] for u in v], 0) if v[0][0] is not None else None,
                np.nanmean([u[1] for u in v], 0), len(v)) for k, v in out.items()}


def analyse() -> None:
    from figdata import save_fd
    labs = il.labels()
    T = trial_table(labs)
    if SFX:
        print(f"== {restnl.NAME} (rule {restnl.RULE}): sessions kept {T.label.nunique()} / {len(labs)}; per animal x epoch\n"
              + T.groupby(["animal", "epoch"]).label.nunique().unstack("epoch").to_string())
    T.to_csv(_o("revup_neural_trials.csv.gz"), index=False)
    j = OUT / "revup_trials.csv.gz"
    if j.exists():
        print(f"== behaviour table present ({j}); join check below")
        join_check(T, pd.read_csv(j))
    else:
        print("== revup_trials.csv.gz (behaviour helper) not on disk: latency / phase defined here (see docstring)")
    # rule 8 counts first
    C = T.groupby(["animal", "epoch", T.pos.eq("far_contra").map({True: "far_contra", False: "other"}), "phase"]).size()
    print("== trials per animal x epoch x class x phase\n" + C.unstack("phase").fillna(0).astype(int).to_string())
    L = T[T.phase == "in_window"].groupby(["animal", "epoch", "pos"]).lat.median().unstack("pos")[POS].round(3)
    print("== median in-window latency (s), per animal x epoch x spout (canonical order)\n" + L.to_string())
    G = group_traces(T)
    MT = metric_table(G, T)
    MT.to_csv(_o("revup_neural_metrics.csv.gz"), index=False)
    D = contrasts(MT)
    D.to_csv(_o("revup_neural_contrasts.csv.gz"), index=False)
    PA = D.groupby(["tau", "cls", "area", "metric", "contrast", "animal", "epoch"]).d.mean().unstack("epoch")[EP]
    PA.to_csv(_o("revup_neural_per_animal.csv"))
    LV = MT[(MT.scheme == "q")].groupby(["tau", "cls", "area", "animal", "epoch", "group"])[list(METRICS)].mean()
    LV.to_csv(_o("revup_neural_levels_per_animal.csv"))
    keyc = [f"{a}_right" for a in KEY[:4]]
    for m in TEST_METRICS:
        s = PA.xs((0.0, "far_contra", m, "Q4-Q1"), level=("tau", "cls", "metric", "contrast"))
        s = s[s.index.get_level_values("area").isin(keyc)]
        print(f"== per animal (rule 8): {m} Q4 - Q1, far-contra, contralesional, raw\n{s.round(3).to_string()}")
    TS = tests(D)
    TS.to_csv(_o("revup_neural_tests.csv"), index=False)
    LC = late_compare(G, T)
    LC.to_csv(_o("revup_late_compare.csv"), index=False)
    if len(LC):
        k = LC[(LC.tau == 0.0)].groupby(["animal", "epoch", "label"])[["n_late", "r_late_normal", "r_late_spont",
                                                                         "r_normal_spont"]].first()
        print("== late vs normal vs spontaneous: shape r per session (raw)\n" + k.round(2).to_string())
    R, S = precue_latency(T)
    R.to_csv(_o("revup_precue_latency.csv"), index=False)
    S.to_csv(_o("revup_precue_latency_session.csv"), index=False)
    RP = R.assign(cls=np.where(R.pos == "far_contra", "far_contra", "other")).groupby(
        ["sel", "cls", "area", "animal", "epoch"]).z.mean().pipe(np.tanh).unstack("epoch")[EP]
    RP.to_csv(_o("revup_precue_latency_per_animal.csv"))
    print("== per animal: within session x spout rho(pre-cue level, latency), contralesional key areas, all trials\n"
          + RP.loc["all"][RP.loc["all"].index.get_level_values("area").isin(keyc)].round(2).to_string())
    # figure data
    fd = {"q": {tau: curves(G, T, "q", QG, tau) for tau in TAUS},
          "fixed": {0.0: curves(G, T, "fixed", FG, 0.0)},
          "late": {tau: {**curves(G, T, "q", ["late"], tau), **curves(G, T, "all", ["in_window"], tau),
                         **curves(G, T, "spont", ["spont"], tau)} for tau in TAUS},
          "D": D[D.area.isin(COLS)], "TS": TS, "R": R, "S": S, "LC": LC}
    save_fd("revup_neural" + SFX, fd, root=OUT / "figdata")


def join_check(T: pd.DataFrame, J: pd.DataFrame) -> None:
    """Agreement with the behaviour helper's table (columns permitting): first-lick phase / latency per trial."""
    print(f"   behaviour table columns: {list(J.columns)[:30]}")
    keys = [c for c in ("label", "cue_s") if c in J.columns]
    if len(keys) < 2:
        print("   (no label + cue_s columns to join on; agreement not checked)")
        return
    Z = pd.read_csv(_o("revup_neural_trials.csv.gz"))
    cue = pd.concat([pd.DataFrame({"label": x, "trial": np.arange(len(np.load(SESS / f"{x}.npz")["cue"])),
                                   "cue_s": np.load(SESS / f"{x}.npz")["cue"]}) for x in Z.label.unique()])
    Z = Z.merge(cue, on=["label", "trial"])
    Z["cue_r"], J["cue_r"] = Z.cue_s.round(2), J.cue_s.round(2)
    M = Z.merge(J, on=["label", "cue_r"], suffixes=("", "_b"))
    print(f"   joined {len(M)} / {len(Z)} trials")
    for c in J.columns:
        if "phase" in c:
            print(pd.crosstab(M.phase, M[c]).to_string())


# ------------------------------------------------------------------------------------------------ figures
def draw_all() -> None:
    from figdata import load_fd
    F = load_fd("revup_neural" + SFX, root=OUT / "figdata")
    FIGS.mkdir(parents=True, exist_ok=True)
    for tau in (0.0, 0.6):
        for cls in CLS:
            for h, hl in HEMI:
                draw_traces(F["q"][tau], cls, h, hl, tau, QG, "quartile")
    draw_traces(F["fixed"][0.0], "far_contra", "right", HEMI[0][1], 0.0, FG, "fixed")
    draw_metrics(F["D"], F["TS"])
    draw_late(F["late"][0.0], F["LC"])
    draw_precue(F["R"])


def _suptitle(fig, text: str) -> float:
    """Suptitle wrapped to the figure width (<= 150 chars per line); returns the top of the axes area for tight_layout."""
    import textwrap
    w = max(60, min(150, int(fig.get_figwidth() * 14)))
    txt = textwrap.fill(text, w)
    fig.suptitle(txt, fontsize=9, y=0.995, va="top")
    # tight_layout (matplotlib >= 3.3) reserves the suptitle's own box; reserving it again left an empty band
    return 1.0


def _fig_legend(fig, handles, ncol: int | None = None, fontsize: float = 9) -> float:
    """ONE figure-level legend below the panels (never on data); returns the bottom for tight_layout's rect."""
    ncol = ncol or len(handles)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=fontsize, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    return ((-(-len(handles) // ncol)) * fontsize * 1.6 + 8) / 72 / fig.get_figheight()


def _ep_an_handles(animals):
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    return ([plt.Line2D([], [], ls="", marker="s", color=EPOCH_LINE[e], label=e) for e in EP] +
            [plt.Line2D([], [], ls="", marker=ANIMAL_MARKER.get(a, "o"), mfc="white", mec="k", label=a)
             for a in animals])


def _gcol(groups):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("viridis")
    return {g: ("#d62728" if g == "late" else cm(i / max(len(groups) - 2, 1))) for i, g in enumerate(groups)}


def draw_traces(CV: dict, cls: str, h: str, hl: str, tau: float, groups: list, scheme: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [a for a in AREAS if f"{a}_{h}" in COLS]
    col = _gcol(groups)
    fig, axs = plt.subplots(len(rows), 8, figsize=(22, 2.1 * len(rows)), squeeze=False, sharex="col")
    for i, a in enumerate(rows):
        j = COLS.index(f"{a}_{h}")
        for k, ep in enumerate(EP):
            for side, (ax, lags, xl) in enumerate(((axs[i, k], LAGS_C, (-1.5, 8.0)), (axs[i, 4 + k], LAGS_L, (-3.0, 1.5)))):
                ax.axvline(0, color="0.4", lw=0.7)
                ax.axhline(0, color="0.7", lw=0.5)
                for g in groups:
                    v = CV.get((cls, ep, g))
                    if v is None or v[side] is None:
                        continue
                    ax.plot(lags, v[side][:, j], color=col[g], lw=1.2, ls="-" if v[2] >= 4 else ":",
                            label=f"{g} ({v[2]})")
                ax.set_xlim(*xl)
                ax.tick_params(labelsize=6)
                if i == 0:
                    # n animals per group in the column title (the per-panel legends sat on the traces)
                    ns = [f"{g} {CV[(cls, ep, g)][2]}" for g in groups
                          if CV.get((cls, ep, g)) is not None and CV[(cls, ep, g)][2] < 4]
                    ax.set_title(f"{ep} -- {'cue' if side == 0 else 'first lick'}-aligned"
                                 + (f"\n(n animals < 4: {', '.join(ns)})" if ns else "\n(all groups 4 animals)"),
                                 fontsize=8)
                if i == len(rows) - 1:
                    ax.set_xlabel(f"time from {'cue' if side == 0 else 'first lick'} (s)", fontsize=7)
                if k == 0:
                    ax.set_ylabel(f"{a}{'  [control]' if a in CTRL else ''}\n" + ("deconv " if tau else "")
                                  + _rl("dF/F x1e3 - restw"), fontsize=7)
    bot = _fig_legend(fig, [plt.Line2D([], [], color=col[g], lw=1.6,
                                       label=g + (" (first lick after the trial stop)" if g == "late" else ""))
                            for g in groups] +
                      [plt.Line2D([], [], color="0.3", lw=1.2, ls=":", label="dotted = < 4 animals")], fontsize=10)
    # share y within a row
    for i in range(len(rows)):
        lo = min(ax.get_ylim()[0] for ax in axs[i])
        hi = max(ax.get_ylim()[1] for ax in axs[i])
        for ax in axs[i]:
            ax.set_ylim(lo, hi)
    what = "far-contra target trials" if cls == "far_contra" else "the other five spouts (pooled)"
    grp = ("latency QUARTILES within animal x spout x epoch (Q1 fastest .. Q4 slowest) + LATE (first lick after the trial "
           "stop, spout gone; red)" if scheme == "quartile" else "FIXED latency bins (s) + late")
    top = _suptitle(fig, _rl(f"NEURAL: {hl} cortex, {what}: area traces above RESTW by first-lick latency group -- {grp}. Left four "
                 f"columns cue-aligned (into the ITI), right four first-lick-aligned. Session mean per group -> animal mean "
                 f"-> mean of animals (dotted = < 4 animals; their n in the column title)."
                 + (f" DECONVOLVED: s = x + {tau} dx/dt (first-order inverse GCaMP kernel)." if tau else
                    " GCaMP6s: lags spikes ~0.1-0.2 s and decays ~0.5-1 s -- compare slopes / shapes, not onsets.")))
    fig.tight_layout(rect=(0, bot, 1, top))
    stem = f"revup_neural_traces_{scheme}_{cls}_{h}" + (f"_deconv{tau}" if tau else "")
    fig.savefig(FIGS / f"{stem}{SFX}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def draw_metrics(D: pd.DataFrame, TS: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    areas = [f"{a}_{h}" for h, _ in HEMI for a in KEY]
    CT = {"Q4-Q1": ("slowest - fastest quartile (Q4 - Q1)", "quartile_contrasts"),
          "gt1-0.5to1": ("slow-only: >1.0 s minus 0.5-1.0 s bins", "slowonly_contrasts")}
    for (tau, ct) in [(t, c) for c in CT for t in TAUS]:
        fig, axs = plt.subplots(len(TEST_METRICS), 2, figsize=(16, 3.0 * len(TEST_METRICS)), squeeze=False)
        for i, m in enumerate(TEST_METRICS):
            for k, cls in enumerate(CLS):
                ax = axs[i, k]
                ax.axhline(0, color="0.5", lw=0.7)
                s = D[(D.tau == tau) & (D.cls == cls) & (D.metric == m) & (D.contrast == ct)]
                P = s.groupby(["area", "epoch", "animal"]).d.mean()
                for ai, a in enumerate(areas):
                    for e_i, ep in enumerate(EP):
                        x0 = ai * 5 + e_i
                        if (a, ep) in P.index.droplevel("animal"):
                            pa = P.xs((a, ep), level=("area", "epoch"))
                            for an, v in pa.items():
                                ax.scatter(x0, v, color=EPOCH_LINE[ep], marker=ANIMAL_MARKER.get(an, "o"), s=14, zorder=3)
                        t = TS[(TS.tau == tau) & (TS.cls == cls) & (TS.metric == m) & (TS.contrast == ct)
                               & (TS.area == a) & (TS.epoch == ep)]
                        if len(t) and np.isfinite(t.level.iloc[0]):
                            ax.plot([x0 + 0.3] * 2, [t.level_lo.iloc[0], t.level_hi.iloc[0]], color=EPOCH_LINE[ep], lw=1.5)
                            ax.plot(x0 + 0.3, t.level.iloc[0], "_", color=EPOCH_LINE[ep], ms=8)
                ax.set_xticks([ai * 5 + 1.5 for ai in range(len(areas))])
                ax.set_xticklabels([a.replace("_right", "\ncontra").replace("_left", "\nipsi") for a in areas], fontsize=7)
                ax.axvline(len(KEY) * 5 - 1, color="0.6", lw=0.8)
                ax.set_title(f"{m}: {CT[ct][0]}, {cls}", fontsize=9)
                ax.tick_params(labelsize=7)
        top = _suptitle(fig, f"NEURAL: within-animal latency CONTRAST of the ramp metrics (per session {CT[ct][0]}; markers = "
                     "animals (o PS92, s PS93, ^ PS94, D PS95), colour = epoch (pre / acute / subacute / chronic); bar = "
                     "boot_ci over animals > sessions). L_level = level at lick onset (-0.1..+0.05 s): ~0 = common "
                     "threshold. L_slope = lick-aligned slope -0.5..0 s, C_slope = cue-aligned slope 0..0.5 s: < 0 = slower "
                     "ramp in slow trials (C_slope in Q4 - Q1 is confounded: Q1's own lick response falls in 0..0.5 s; read it on the slow-only contrast). t_half = lick-aligned time of the 50 % crossing: < 0 = earlier relative to the "
                     "lick (slow ramp), ~0 = same shape shifted (delayed start)."
                     + (f" DECONVOLVED tau = {tau} s." if tau else " Raw dF/F."))
        hd = _ep_an_handles(sorted(D.animal.dropna().unique())) + \
            [plt.Line2D([], [], color="0.3", lw=1.5, marker="_", ms=8, label="boot_ci (animals > sessions)")]
        bot = _fig_legend(fig, hd, fontsize=10)
        fig.tight_layout(rect=(0, bot, 1, top), h_pad=1.5)
        fig.savefig(FIGS / f"revup_neural_{CT[ct][1]}{'_deconv' + str(tau) if tau else ''}{SFX}.png", dpi=100,
                    bbox_inches="tight")
        plt.close(fig)
    _draw_metrics_simple(D, TS)


def _draw_metrics_simple(D: pd.DataFrame, TS: pd.DataFrame) -> None:
    """SIMPLE view of the slow-only contrast (raw dF/F): far-contra trials, contralesional key areas only, one panel per
    metric, larger markers -- the same D / TS numbers as revup_neural_slowonly_contrasts.png (which keeps ipsi, the
    other five spouts and the deconvolved variants)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    tau, ct, cls = 0.0, "gt1-0.5to1", "far_contra"
    areas = [f"{a}_right" for a in KEY]
    fig, axs = plt.subplots(1, len(TEST_METRICS), figsize=(4.4 * len(TEST_METRICS), 4.4), squeeze=False)
    for i, m in enumerate(TEST_METRICS):
        ax = axs[0, i]
        ax.axhline(0, color="0.5", lw=0.7)
        s = D[(D.tau == tau) & (D.cls == cls) & (D.metric == m) & (D.contrast == ct)]
        P = s.groupby(["area", "epoch", "animal"]).d.mean()
        for ai, a in enumerate(areas):
            for e_i, ep in enumerate(EP):
                x0 = ai * 5 + e_i
                if (a, ep) in P.index.droplevel("animal"):
                    for an, v in P.xs((a, ep), level=("area", "epoch")).items():
                        ax.scatter(x0, v, color=EPOCH_LINE[ep], marker=ANIMAL_MARKER.get(an, "o"), s=24, zorder=3)
                t = TS[(TS.tau == tau) & (TS.cls == cls) & (TS.metric == m) & (TS.contrast == ct)
                       & (TS.area == a) & (TS.epoch == ep)]
                if len(t) and np.isfinite(t.level.iloc[0]):
                    ax.plot([x0 + 0.35] * 2, [t.level_lo.iloc[0], t.level_hi.iloc[0]], color=EPOCH_LINE[ep], lw=1.8)
                    ax.plot(x0 + 0.35, t.level.iloc[0], "_", color=EPOCH_LINE[ep], ms=10)
        ax.set_xticks([ai * 5 + 1.5 for ai in range(len(areas))])
        ax.set_xticklabels(KEY, fontsize=9)
        ax.set_title(m, fontsize=11)
    hd = _ep_an_handles(sorted(D.animal.dropna().unique())) + \
        [plt.Line2D([], [], color="0.3", lw=1.8, marker="_", ms=10, label="boot_ci (animals > sessions)")]
    bot = _fig_legend(fig, hd, fontsize=10)
    top = _suptitle(fig, "NEURAL (simple): slow-only latency contrast (>1.0 s minus 0.5-1.0 s first-lick bins, per session) "
                    "of the ramp metrics, FAR-CONTRA trials, CONTRALESIONAL key areas, raw dF/F. L_level ~0 = common "
                    "threshold; L_slope / C_slope < 0 = slower ramp in slow trials; t_half < 0 = earlier 50 % crossing "
                    "(slow ramp), ~0 = same shape shifted. Full set: revup_neural_slowonly_contrasts.png.")
    fig.tight_layout(rect=(0, bot, 1, top))
    fig.savefig(FIGS / f"revup_neural_slowonly_contrasts_simple{SFX}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def draw_late(CV: dict, LC: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    areas = [f"{a}_{h}" for a in AREAS for h, _ in HEMI if f"{a}_{h}" in COLS]
    eps = [e for e in EP if ("all", e, "late") in CV]
    fig, axs = plt.subplots(len(eps) + 1, len(areas), figsize=(2.0 * len(areas), 2.4 * (len(eps) + 1)), squeeze=False)
    cc = {"in_window": "k", "late": "#d62728", "spont": "#1f77b4"}
    for r, ep in enumerate(eps):
        for j, a in enumerate(areas):
            ax = axs[r, j]
            ci = COLS.index(a)
            for g in ("in_window", "late", "spont"):
                v = CV.get(("all", ep, g))
                if v is not None:
                    ax.plot(LAGS_L, v[1][:, ci], color=cc[g], lw=1.1, label=f"{g} ({v[2]})")
            ax.axvline(0, color="0.5", lw=0.6)
            ax.set_xlim(-2.5, 1.5)
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(a.replace("_right", " contra").replace("_left", " ipsi"), fontsize=8)
            if j == 0:
                # n animals per group moved from a legend on the traces into the row label
                nn = "/".join(str(CV[("all", ep, g)][2]) if CV.get(("all", ep, g)) is not None else "-"
                              for g in ("in_window", "late", "spont"))
                ax.set_ylabel(_rl(f"{ep}  (n win/late/spont {nn})\ndF/F x1e3 - restw"), fontsize=7)
    # bottom row: shape correlations per session -- one wider panel (was 1 of 13 tiny axes) + its legend beside it
    gs = axs[-1, 0].get_gridspec()
    for a_ in axs[-1]:
        a_.remove()
    ax = fig.add_subplot(gs[-1, 0:3])
    lax = fig.add_subplot(gs[-1, 3:6])
    lax.axis("off")
    s = LC[LC.tau == 0.0].groupby(["label", "animal", "epoch"])[["r_late_normal", "r_late_spont", "r_normal_spont"]].first()
    for k, c in enumerate(["r_late_normal", "r_late_spont", "r_normal_spont"]):
        for (lab, an, ep), v in s[c].items():
            ax.scatter(k + np.random.default_rng(hash(lab) % 2**32).uniform(-0.15, 0.15), v, color=EPOCH_LINE[ep],
                       marker=ANIMAL_MARKER.get(an, "o"), s=12)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["late ~ normal", "late ~ spont", "normal ~ spont"], fontsize=8)
    ax.set_ylabel("per-session shape r\n(-1.5..+1 s)", fontsize=8)
    ax.tick_params(labelsize=7)
    lax.legend(handles=[plt.Line2D([], [], color=cc[g], lw=1.6, label=f"trace: {g}") for g in cc] +
               _ep_an_handles(sorted(LC.animal.dropna().unique())), loc="center left", ncol=2, fontsize=9,
               frameon=False, title="traces (rows 1-4) / shape r markers (bottom)", title_fontsize=9)
    top = _suptitle(fig, _rl("NEURAL: LATE first licks (first lick after the trial stop, spout gone) vs NORMAL first licks (in-window, all "
                 "spouts) vs SPONTANEOUS ITI bout starts (>= 2 s quiet), first-lick-aligned, above RESTW. Sessions with >= 2 "
                 "late licks enter the late curve; mean of animal means (row label: n animals in_window / late / spont). Bottom: per-session shape "
                 "correlation over key areas x hemispheres (-1.5..+1 s, colour = epoch, marker = animal)."))
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(FIGS / f"revup_neural_late_vs_normal_vs_spont{SFX}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def draw_precue(R: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    areas = [f"{a}_{h}" for h, _ in HEMI for a in KEY]
    fig, axs = plt.subplots(2, len(areas) // 2, figsize=(4.0 * len(areas) // 2, 7), squeeze=False)
    for i, a in enumerate(areas):
        ax = axs[i // (len(areas) // 2), i % (len(areas) // 2)]
        s = R[(R.sel == "all") & (R.area == a)].groupby(["pos", "epoch", "animal"]).z.mean().pipe(np.tanh)
        for pi, p in enumerate(POS):
            for e_i, ep in enumerate(EP):
                x0 = pi * 5 + e_i
                if (p, ep) in s.index.droplevel("animal"):
                    for an, v in s.xs((p, ep), level=("pos", "epoch")).items():
                        ax.scatter(x0, v, color=EPOCH_LINE[ep], marker=ANIMAL_MARKER.get(an, "o"), s=12)
                    ax.plot([x0 - 0.3, x0 + 0.3], [s.xs((p, ep), level=("pos", "epoch")).mean()] * 2, color=EPOCH_LINE[ep])
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xticks([pi * 5 + 1.5 for pi in range(len(POS))])
        ax.set_xticklabels([p.replace("_", "\n") for p in POS], fontsize=7)
        ax.set_title(a.replace("_right", " contralesional").replace("_left", " ipsilesional"), fontsize=9)
        ax.tick_params(labelsize=7)
    top = _suptitle(fig, _rl("NEURAL: does the trial's PRE-CUE level (-1..0 s above RESTW) predict its first-lick latency? Spearman rho "
                 "within session x spout (trials with a first lick, in-window or late; >= 8 trials), Fisher-z mean per "
                 "animal (markers) x epoch (colour); bar = mean of animals. Spouts in canonical order. > 0 = higher "
                 "pre-cue level -> slower first lick."))
    bot = _fig_legend(fig, _ep_an_handles(sorted(R.animal.dropna().unique())) +
                      [plt.Line2D([], [], color="0.3", lw=1.5, label="bar = mean of animals")], fontsize=10)
    fig.tight_layout(rect=(0, bot, 1, top))
    fig.savefig(FIGS / f"revup_neural_precue_vs_latency{SFX}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    global SFX
    SFX = restnl.SFX if restnl.variant() == "restnl" else ""
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
    if "--plot-only" not in sys.argv:
        if not SFX:   # restnl: no re-extraction, the restw-referenced cache is shifted at load
            extract(jobs)
        if "--extract-only" in sys.argv:
            return 0
        analyse()
    draw_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
