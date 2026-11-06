"""Does PRE-CUE (ENL) cortical activity predict the ANGLE of the upcoming first post-cue lick? (Priya 10-29: "we should be
able to decode lick angle from ENL, no?")

Per engaged trial with a first post-cue lick (the trial rows of the ITI session files, `iti_licks._first_lick_times`):
  features  ENL-window mean of every Allen area x hemisphere in the split basis (ALM / tjM1 + MOs / MOp rest;
            `iti_licks.area_features`) from the CUE-aligned traces <OUT_W>/cued_wide/<label>.npz (drift-removed dF/F
            x1e3, no per-trial baseline), each area / its frame-wise SD in the session (`iti_licks.area_sd`, the
            KEY-row decoders' z-scoring; no trial-dependent centring, so the frozen decoder transfers unchanged).
            Window = lags [-W, -0.10] s from the cue (W = 1.0 primary, 0.5 variant); it stops 0.10 s before the cue so
            interpolation never reaches a post-cue frame.
  lick-free SLIDING (Priya 11-04 ~20:00, "the between-trial lick ENL family and fr_enl_decode also SLIDE, for
            consistency"; PRIMARY, lickfree "slide"): the imaging ENL decoders' own search (`enl_window.sliding_lags` ->
            `precue_window_start`): the window [-W, -0.10] when no licks_all onset falls in [cue - W, cue), else the
            latest lick-free W-s gap after the strobe, else the trial is dropped; a slid window must start >= strobe +
            0.3 s (`enl_window.SETTLE_S`, the spout-arrival transient) and inside the traces (>= -2 s), else dropped.
            Features "s10:" / "s05:", window start per trial "slo10" / "slo05". ROBUSTNESS "2s": the old fixed window
            [-W, -0.10] on trials with no lick onset in [cue - 2 s, cue) ("w10:" / "w05:"). The fixed-drop variant "W"
            (no lick in [cue - W, cue)) is retired by the slide decision; `select` still reads it for old tables.
            Until 11-05 "2s" was primary and the window never slid (the 11-04 ~19:00 "keep them fixed" note).
  target    angle = degrees(arcsin(side_ml / reach_mm)) of the FIRST post-cue lick, from <OUT_W>/firstlick_full/
            <label>.npz (3-D kinematics of every bout-first post-cue lick), matched by DAQ time: the kinematic lick whose
            peak lies in [onset - 0.02, onset + MATCH_S] of the behaviour first-lick onset. + = toward + side_ml.
Per animal (rule 8): ridge (StandardScaler + RidgeCV, as lick_refit_decoders) trained on PRE sessions, leave-one-pre-
session-out; held-out r pooled over trials:
  raw       target = angle (partly = which spout)
  spout     baseline: angle predicted by the training sessions' mean angle at the trial's spout (no neural data)
  partial   r between the raw model's prediction and the angle, BOTH residualised within session x spout
  beyond    model trained on features AND target residualised within session x spout (the trial-to-trial aim part)
  nulls     200x: angle permuted within session x spout (for partial / beyond) and within session (for raw)
Post-stroke: the decoders refit on ALL pre trials of the animal (fixed training set = the August pre sessions; the
coefficients are written to enl_angle_decode_coefs.csv) applied to every post session: r per session (raw, partial,
beyond), and the LIKE-FOR-LIKE offset (decoded - the Gaussian-weighted mean of what the animal's pre trials with the same
actual angle decoded; `likeforlike.offset`, method nw, pre trials leave-one-pre-session-out) by epoch; change vs pre =
per-animal table + paired nested bootstrap (analysis_kit.boot_delta).
Outputs <OUT_W>/enl_angle_decode_trials.csv.gz, enl_angle_decode_pre.csv, enl_angle_decode_sessions.csv,
enl_angle_decode_epochs.csv, enl_angle_decode_tests.csv, enl_angle_decode_coefs.csv, figures/lick_enl_angle_decode.png.
Usage: python scripts/lick_enl_angle_decode.py [--extract-only] [--jobs N] [--nshuf 200] [--plot-only] [--controls --nshuf 50]
  --controls: time-trend / previous-spout / previous-angle checks of the pre beyond-spout r -> enl_angle_decode_controls.csv
RULE 10 (10-29): the all-pre raw / beyond decoders AND their leave-one-pre-session-out siblings (the pre rows) are frozen
references (`frozen_fits` -> frozen_models.load_or_fit, kind `enl_angle`); the permutation nulls refit in memory (they
score nothing post-stroke). Follow-ups (nested history test, offset exploration): scripts/lick_enl_offset_explore.py.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import likeforlike as lfl  # noqa: E402

il._behaviour = functools.lru_cache(maxsize=2)(il._behaviour)   # _first_lick_times + our lick read share one load

OUT = il.OUT_W
TRIALS = OUT / "enl_angle_decode_trials.csv.gz"
WINDOWS = (1.0, 0.5)
W_END = -0.10
LF_LONG = 2.0
#: the lick-free rules run (Priya 11-04 ~20:00 / 11-05): "slide" PRIMARY (the decoders' sliding lick-free window),
#: "2s" robustness (fixed window, no lick in the last 2 s)
LF_RULES = ("slide", "2s")
LF_PRIMARY = "slide"


def fcols(W: float, lf: str, names) -> list:
    """The feature columns of window W under lick-free rule lf: "s10:" / "s05:" for the slid window, else the fixed
    window's "w10:" / "w05:"."""
    return [f"{'s' if lf == 'slide' else 'w'}{'10' if W == 1.0 else '05'}:{n}" for n in names]

MATCH_S = 0.25
ALPHAS = np.logspace(-1, 5, 13)
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
MIN_SESS = 10            # lick-free matched trials for a post session's r / offset
MIN_LF_PRE = 150         # the 2 s lick-free set is primary for an animal only if it keeps >= this many pre trials
SEED = 0


# ------------------------------------------------------------------------------------------------------------ EXTRACT
def session_trials(label: str) -> dict:
    """One session (module-level for spawn): per first-lick trial row of cued_wide -> cue, onset, pre-cue lick counts,
    matched kinematics, ENL features."""
    from wfield_local import config, epochs
    z = np.load(OUT / "cued_wide" / f"{label}.npz")
    d = il._load(label)
    cue, fl = il._first_lick_times(label, d["first_rt"])
    ok = np.isfinite(fl) & np.isfinite(cue)
    cue, fl = cue[ok], fl[ok]
    if len(cue) != len(z["pos"]):
        raise RuntimeError(f"{label}: {len(cue)} matched first licks vs {len(z['pos'])} cued_wide rows")
    if not np.array_equal(np.asarray(d["first_pos"], str)[ok], z["pos"]):
        raise RuntimeError(f"{label}: spout labels differ from cued_wide")
    # pre-cue licks: EVERY lick of the session (licks_all.csv, all contexts) -- the behaviour table's lick_on_s holds
    # only the trial-context licks and has no pre-cue licks at all (checked 10-29: lick-free 100 % with it, ~82 % here)
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_on_s"])
    bo = np.sort(A.daq_on_s.to_numpy(float))
    bo = bo[np.isfinite(bo)]
    nl = {L: np.searchsorted(bo, cue) - np.searchsorted(bo, cue - L) for L in (0.5, 1.0, LF_LONG)}
    # a licks_all lick between the cue and the behaviour first-lick onset -> that lick, not fl, was the first: flag
    n_between = np.searchsorted(bo, fl - 0.005) - np.searchsorted(bo, cue)
    # kinematics: bout-first post-cue licks, exact DAQ peak t
    k = np.load(OUT / "firstlick_full" / f"{label}.npz")
    kt = np.asarray(k["t"], float)
    o = np.argsort(kt)
    kt = kt[o]
    j = np.searchsorted(kt, fl - 0.02)
    hit = (j < len(kt)) & (kt[np.minimum(j, len(kt) - 1)] <= fl + MATCH_S) if len(kt) else np.zeros(len(fl), bool)
    ji = o[np.minimum(j, max(len(kt) - 1, 0))]
    side = np.where(hit, k["side_ml"][ji], np.nan) if len(kt) else np.full(len(fl), np.nan)
    reach = np.where(hit, k["reach_mm"][ji], np.nan) if len(kt) else np.full(len(fl), np.nan)
    kpos = np.where(hit, k["pos"][ji].astype(str), "") if len(kt) else np.full(len(fl), "")
    lag = np.where(hit, kt[np.minimum(j, len(kt) - 1)] - fl, np.nan) if len(kt) else np.full(len(fl), np.nan)
    # features
    lags = z["lags"].astype(float)
    cols = [str(c) for c in z["cols"]]
    names = il.area_names(cols)
    sd = il.area_sd().loc[label, names].to_numpy(float)
    C = z["cue"].astype(float)
    F, FS, LO = {}, {}, {}
    from wfield_local import enl_window
    strobe = enl_window.strobe_for_cues(label, cue)
    for W in WINDOWS:
        m = (lags >= -W - 1e-9) & (lags <= W_END + 1e-9)
        F[W] = il.area_features(np.nanmean(C[:, m, :], 1), cols, label) / sd
        # THE SLIDING LICK-FREE WINDOW (11-04 ~20:00 decision), on the same licks_all onsets as the fixed-drop rule
        LO[W], hi = enl_window.sliding_lags(cue, strobe, bo, W, end_lag=W_END, span_lo=float(lags.min()))
        FS[W] = il.area_features(enl_window.window_mean(C, lags, LO[W], hi), cols, label) / sd
    return {"label": label, "animal": config.animal_of(label), "epoch": epochs.epoch_of(label), "cue": cue, "fl": fl,
            "pos": z["pos"].astype(str), "kpos": kpos, "n_between": n_between, "nl05": nl[0.5], "nl1": nl[1.0], "nl2": nl[LF_LONG],
            "side_ml": side, "reach_mm": reach, "kin_lag": lag, "F10": F[1.0], "F05": F[0.5], "names": names,
            "S10": FS[1.0], "S05": FS[0.5], "slo10": LO[1.0], "slo05": LO[0.5], "strobe": strobe}


def extract(jobs: int) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(TRIALS)
    res, fails = ak.fan_sessions(il.labels(), session_trials, jobs=jobs)
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    rows = []
    names = res[0][1]["names"]
    for _x, v in res:
        if v["names"] != names:
            raise RuntimeError(f"{v['label']}: feature names differ")
        T = pd.DataFrame({c: v[c] for c in ("cue", "fl", "pos", "kpos", "n_between", "nl05", "nl1", "nl2", "side_ml", "reach_mm",
                                            "kin_lag", "strobe", "slo10", "slo05")})
        T.insert(0, "label", v["label"]); T.insert(1, "animal", v["animal"]); T.insert(2, "epoch", v["epoch"])
        T = pd.concat([T, pd.DataFrame(v["F10"], columns=[f"w10:{n}" for n in names]),
                       pd.DataFrame(v["F05"], columns=[f"w05:{n}" for n in names]),
                       pd.DataFrame(v["S10"], columns=[f"s10:{n}" for n in names]),
                       pd.DataFrame(v["S05"], columns=[f"s05:{n}" for n in names])], axis=1)
        rows.append(T)
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(TRIALS, index=False, float_format="%.6g")
    m = np.isfinite(T.side_ml)
    print(f"== {len(T)} first-lick trials, {m.sum()} with matched kinematics; onset->peak lag "
          f"{np.nanpercentile(T.kin_lag, [5, 50, 95]).round(3)} s; spout agree "
          f"{(T.kpos[m].str.replace('center', 'mid') == T.pos[m]).mean():.4f}; licks between cue and onset "
          f"{(T.n_between > 0).mean():.4f}")
    return T


# ------------------------------------------------------------------------------------------------------------ DECODE
def _resid(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    """v (n,) or (n, p) minus its mean within each group label g."""
    v = np.asarray(v, float).copy()
    for q in np.unique(g):
        m = g == q
        v[m] -= v[m].mean(0)
    return v


def _fit(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    return sc, RidgeCV(alphas=ALPHAS).fit(sc.transform(X), y)


def _pred(m, X):
    return m[1].predict(m[0].transform(X))


def _r(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 3 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else np.nan


def _digest(*arrs) -> str:
    import hashlib
    h = hashlib.sha1()
    for a in arrs:
        a = np.asarray(a)
        h.update("|".join(a.astype(str)).encode() if a.dtype.kind in "OUS" else
                 np.ascontiguousarray(a.astype(np.float64)).tobytes())
    return h.hexdigest()


def frozen_fits(a: str, W: float, lf: str, X, y, sess, cell, variant: str = "") -> dict:
    """RULE 10 (CLAUDE.md, docs/FROZEN_MODELS.md section 7): the animal's all-pre raw + beyond-spout ridges and their
    leave-one-pre-session-out siblings, through `frozen_models.load_or_fit` (fitted once, then read back verbatim).
    Identity = animal + kind `enl_angle` + window / lick-free rule / feature variant + the sorted PRE labels with their
    data signatures. The spec cannot see the trial-level inputs (kinematic matching, lick-free selection), so the payload
    carries a SHA-1 of (X, y) and a mismatch on a hit is reported loudly (the stored reference is still used)."""
    from wfield_local import frozen_models as fm
    labs = sorted(set(np.asarray(sess, str)))
    Xr, yr = _resid(X, cell), _resid(y, cell)
    dig = _digest(X, y, np.asarray(cell, str))
    spec = fm.make_spec(a, "enl_angle", align=f"enl_w{W:g}_lf{lf}{variant}", source=f"areas_split_r{il.ROI_VERSION}",
                        train_labels=labs, post_s=W, n_features=X.shape[1],
                        train_data={"xy_sha1": dig, "sess": np.asarray(sess, str)})  # training-data fingerprint (FREEZE_VERSION 2, rebuild 11-03)

    def fit():
        loso_ = {s: (_fit(X[sess != s], y[sess != s]), _fit(Xr[sess != s], yr[sess != s])) for s in labs}
        return {"raw": _fit(X, y), "beyond": _fit(Xr, yr), "loso": loso_, "xy_sha1": dig}
    P, status = fm.load_or_fit(spec, fit, meta={"script": "lick_enl_angle_decode.py", "n_trials": int(len(y)),
                                                "variant": variant or "base"})
    print(f"[frozen] {a} enl_angle W={W:g} lf={lf}{variant}: {status}", flush=True)
    if P.get("xy_sha1") != dig:
        print(f"[frozen] !! {a} W={W:g} lf={lf}{variant}: the stored model was trained on DIFFERENT trial data than "
              "today's selection (same pre sessions) -- the stored reference is used; refreeze if today's is right",
              flush=True)
    return P


def loso(X, y, sess, cell, pos, fits: dict | None = None) -> dict:
    """Leave-one-session-out on pre trials: raw / spout-only / beyond predictions. fits = the frozen LOSO models
    {held-out label: (raw, beyond)} (frozen_fits) for the real data; None = refit (the permutation nulls)."""
    Xr, yr = _resid(X, cell), _resid(y, cell)
    p_raw, p_sp, p_bey = (np.full(len(y), np.nan) for _ in range(3))
    for s in np.unique(sess):
        te, tr = sess == s, sess != s
        m_raw, m_bey = fits[s] if fits is not None else (_fit(X[tr], y[tr]), _fit(Xr[tr], yr[tr]))
        p_raw[te] = _pred(m_raw, X[te])
        p_bey[te] = _pred(m_bey, Xr[te])
        mu = pd.Series(y[tr]).groupby(pos[tr]).mean()
        p_sp[te] = mu.reindex(pos[te]).to_numpy(float)
    return {"raw": p_raw, "spout": p_sp, "beyond": p_bey, "yr": yr}


def scores(P: dict, y, cell) -> dict:
    return {"r_raw": _r(P["raw"], y), "r_spout": _r(P["spout"], y),
            "r_partial": _r(_resid(P["raw"], cell), P["yr"]), "r_beyond": _r(P["beyond"], P["yr"]),
            "r_raw_beyond_spout": _r(_resid(P["raw"] - np.nan_to_num(P["spout"]), cell), P["yr"])}


def _perm_within(y, g, rng):
    y = y.copy()
    for q in np.unique(g):
        i = np.flatnonzero(g == q)
        y[i] = y[rng.permutation(i)]
    return y


def select(T: pd.DataFrame, W: float, lf: str) -> pd.DataFrame:
    """The trials of window W under lick-free rule lf: "slide" = the trial has a sliding lick-free window (`slo10` /
    `slo05` finite); "2s" / "W" = no licks_all onset in the last 2 s / W s (the fixed window)."""
    if lf == "slide":
        c = f"slo{'10' if W == 1.0 else '05'}"
        if c not in T:
            raise SystemExit(f"{TRIALS} predates the sliding ENL window (no {c}): re-run with --extract")
        free = np.isfinite(T[c].to_numpy(float))
    else:
        free = T[{"W": {1.0: "nl1", 0.5: "nl05"}[W], "2s": "nl2"}[lf]] == 0
    m = np.isfinite(T.angle) & free & (T.n_between == 0) & T.epoch.isin(EP)
    return T[m].reset_index(drop=True)


def run_variant(T: pd.DataFrame, W: float, lf: str, nshuf: int, names: list) -> tuple:
    S = select(T, W, lf)
    fc = fcols(W, lf, names)
    pre_rows, sess_rows, trial_rows, coef_rows = [], [], [], []
    rng = np.random.default_rng(SEED)
    for a in AN:
        A = S[S.animal == a].reset_index(drop=True)
        Pr = A[A.epoch == "pre"].reset_index(drop=True)
        if Pr.label.nunique() < 2 or len(Pr) < 30:
            continue
        X, y = Pr[fc].to_numpy(float), Pr.angle.to_numpy(float)
        sess, pos = Pr.label.to_numpy(str), Pr.pos.to_numpy(str)
        cell = np.char.add(np.char.add(sess, "|"), pos)
        FZ = frozen_fits(a, W, lf, X, y, sess, cell)
        P = loso(X, y, sess, cell, pos, fits=FZ["loso"])
        sc = scores(P, y, cell)
        null = {k: [] for k in ("r_raw", "r_partial", "r_beyond")}
        for _ in range(nshuf):
            yc = _perm_within(y, cell, rng)
            Q = loso(X, yc, sess, cell, pos)
            s2 = scores(Q, yc, cell)
            null["r_partial"].append(s2["r_partial"]); null["r_beyond"].append(s2["r_beyond"])
            ys = _perm_within(y, sess, rng)
            null["r_raw"].append(_r(loso(X, ys, sess, cell, pos)["raw"], ys))
        row = {"animal": a, "window_s": W, "lickfree": lf, "n_pre_trials": len(Pr), "n_pre_sessions": Pr.label.nunique(),
               "sd_angle": float(y.std()), "sd_angle_within": float(P["yr"].std()), **sc}
        for k, v in null.items():
            v = np.asarray(v)
            row[f"{k}_null95"] = float(np.nanpercentile(v, 95)) if len(v) else np.nan
            row[f"p_{k}"] = float((1 + np.sum(v >= sc[k])) / (1 + len(v))) if len(v) else np.nan
        pre_rows.append(row)
        # frozen all-pre decoders -> every session (pre rows keep their LOSO predictions)
        M_raw, M_bey = FZ["raw"], FZ["beyond"]
        for kind, M in (("raw", M_raw), ("beyond", M_bey)):
            coef_rows += [{"animal": a, "window_s": W, "lickfree": lf, "model": kind, "feature": n, "coef_std": c,
                           "alpha": float(M[1].alpha_)} for n, c in zip(names, M[1].coef_)]
        Pr = Pr.assign(pred=P["raw"], pred_bey=P["beyond"], angle_w=P["yr"])
        Po = A[A.epoch != "pre"].reset_index(drop=True)
        if len(Po):
            cpo = np.char.add(np.char.add(Po.label.to_numpy(str), "|"), Po.pos.to_numpy(str))
            Xo = Po[fc].to_numpy(float)
            Po = Po.assign(pred=_pred(M_raw, Xo), pred_bey=_pred(M_bey, _resid(Xo, cpo)),
                           angle_w=_resid(Po.angle.to_numpy(float), cpo))
        B = pd.concat([Pr, Po], ignore_index=True)
        B["cell"] = B.label + "|" + B.pos
        trial_rows.append(B[["label", "animal", "epoch", "pos", "cell", "angle", "angle_w", "pred", "pred_bey"]])
    if not trial_rows:
        return pd.DataFrame(pre_rows), pd.DataFrame(), pd.DataFrame(coef_rows)
    R = pd.concat(trial_rows, ignore_index=True)
    R["set"] = "enl"
    off, _out = lfl.offset(R, "angle", "pred", lfl.SIGMA_DEG, what=f"ENL angle W={W} lf={lf}", method="nw")
    R["offset"] = off
    for (lab, a, ep), g in R.groupby(["label", "animal", "epoch"], sort=True):
        if len(g) < MIN_SESS:
            continue
        cell = g.cell.to_numpy(str)
        sess_rows.append({"window_s": W, "lickfree": lf, "label": lab, "animal": a, "epoch": ep, "n": len(g),
                          "r_raw": _r(g.pred.to_numpy(), g.angle.to_numpy()),
                          "r_partial": _r(_resid(g.pred.to_numpy(), cell), g.angle_w.to_numpy()),
                          "r_beyond": _r(g.pred_bey.to_numpy(), g.angle_w.to_numpy()),
                          "offset": float(np.nanmean(g.offset)), "offset_n": int(np.isfinite(g.offset).sum()),
                          "mean_angle": float(g.angle.mean()), "mean_pred": float(g.pred.mean())})
    return pd.DataFrame(pre_rows), pd.DataFrame(sess_rows), pd.DataFrame(coef_rows)


def epoch_tables(SS: pd.DataFrame):
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(SEED)
    E = SS.groupby(["window_s", "lickfree", "animal", "epoch"]).agg(
        n_sessions=("label", "size"), n_trials=("n", "sum"), r_raw=("r_raw", "mean"), r_partial=("r_partial", "mean"),
        r_beyond=("r_beyond", "mean"), offset=("offset", "mean")).reset_index()
    tests = []
    for (W, lf), g in SS.groupby(["window_s", "lickfree"]):
        for val in ("r_raw", "r_partial", "r_beyond", "offset"):
            pre = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            for ep in EP[1:]:
                post = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal")}
                post = {a: v for a, v in post.items() if len(v)}
                pre_ = {a: v for a, v in pre.items() if len(v)}
                ci, dr = ak.boot_delta(post, pre_, rng, draws=True)
                per = {a: float(np.mean(post[a]) - np.mean(pre_[a])) for a in sorted(set(post) & set(pre_))}
                tests.append({"window_s": W, "lickfree": lf, "value": val, "epoch": ep,
                              "delta": None if ci is None else ci.point, "lo": None if ci is None else ci.lo,
                              "hi": None if ci is None else ci.hi, "p": ak.boot_p(dr) if ci is not None else np.nan,
                              "n_animals": len(per), "n_pos": int(sum(v > 0 for v in per.values())),
                              **{f"d_{a}": per.get(a, np.nan) for a in AN}})
    return E, pd.DataFrame(tests)


# ------------------------------------------------------------------------------------------------------------ FIGURE
def figure(PRE: pd.DataFrame, SS: pd.DataFrame, TS: pd.DataFrame, W: float, lf: str) -> None:
    import matplotlib.pyplot as plt
    from wfield_local import config
    col = {a: config.animal_color().get(a) for a in AN}
    P = PRE[(PRE.window_s == W) & (PRE.lickfree == lf)].set_index("animal")
    S = SS[(SS.window_s == W) & (SS.lickfree == lf)]
    fig, ax = plt.subplots(1, 4, figsize=(19, 5.0))
    ks = [("r_spout", "spout\nmeans only"), ("r_raw", "ENL ->\nangle (raw)"), ("r_partial", "raw model,\nbeyond spout"),
          ("r_beyond", "beyond-spout\nmodel")]
    for i, a in enumerate([a for a in AN if a in P.index]):
        for j, (k, _l) in enumerate(ks):
            x = j + (i - 1.5) * 0.17
            ax[0].bar(x, P.loc[a, k], 0.16, color=col[a], label=a if j == 0 else None)
            if f"{k}_null95" in P.columns:
                ax[0].plot([x - 0.08, x + 0.08], [P.loc[a, f"{k}_null95"]] * 2, color="k", lw=1.2)
    ax[0].set_xticks(range(len(ks)), [l_ for _k, l_ in ks], fontsize=8)
    ax[0].axhline(0, color="0.5", lw=0.6)
    ax[0].set_ylabel("held-out r (pre, leave-one-session-out)")
    ax[0].set_title("1. pre-stroke: ENL predicts first-lick angle?\n(black tick = 95th pct of shuffle null)", fontsize=9)
    for k_i, (val, ttl) in enumerate((("r_raw", "2. frozen decoder r (raw angle)"),
                                      ("r_beyond", "3. frozen beyond-spout r"),
                                      ("offset", "4. like-for-like offset (deg)\n+ = read toward + side_ml"))):
        a_ = ax[k_i + 1]
        for i, a in enumerate(AN):
            g = S[S.animal == a]
            for e_i, ep in enumerate(EP):
                v = g[g.epoch == ep][val].dropna()
                xx = e_i + (i - 1.5) * 0.15
                a_.scatter(np.full(len(v), xx), v, s=9, color=col[a], alpha=0.5)
                if len(v):
                    a_.plot([xx - 0.07, xx + 0.07], [v.mean()] * 2, color=col[a], lw=2, label=a if e_i == 0 else None)
        t = TS[(TS.window_s == W) & (TS.lickfree == lf) & (TS.value == val)].set_index("epoch")
        for e_i, ep in enumerate(EP[1:], 1):
            if ep in t.index and pd.notna(t.loc[ep, "delta"]):
                a_.text(e_i, 0.99, f"d={t.loc[ep, 'delta']:+.2f}\np={t.loc[ep, 'p']:.2f}", ha="center", va="top",
                        transform=a_.get_xaxis_transform(), fontsize=7)
        a_.axhline(0, color="0.5", lw=0.6)
        a_.set_xticks(range(len(EP)), EP)
        a_.set_title(ttl + "\n(d, p = change vs pre, paired nested bootstrap)", fontsize=9, pad=4)
        lo_, hi_ = a_.get_ylim()
        a_.set_ylim(lo_, hi_ + 0.18 * (hi_ - lo_))
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=len(l_), fontsize=9, frameon=False)
    rule = {"slide": f"sliding lick-free {W:g} s window (ends <= {W_END:g} s)", "W": f"no lick in the last {W:g} s",
            "2s": "no lick in the last 2 s"}[lf]
    fig.suptitle(f"ENL ({-W:g}..{W_END:g} s before cue; {rule}) "
                 "-> angle of the first post-cue lick. Ridge on Allen areas (ALM/tjM1 split), per animal; post = frozen "
                 "all-pre decoder. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = OUT / "figures" / "lick_enl_angle_decode.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, out, dpi=110)
    print(f"== figure {out}")


def _resid_cov(v: np.ndarray, Z: np.ndarray, g: np.ndarray) -> np.ndarray:
    """v (n,) or (n, p) minus its least-squares fit on [1, Z] within each group g."""
    v = np.asarray(v, float).copy()
    for q in np.unique(g):
        m = np.flatnonzero(g == q)
        D = np.column_stack([np.ones(len(m)), Z[m]])
        b = np.linalg.lstsq(D, v[m], rcond=None)[0]
        v[m] -= D @ b
    return v


def controls(T: pd.DataFrame, names: list, nshuf: int) -> pd.DataFrame:
    """Confound checks of the PRE beyond-spout result (W = 1.0 s): the beyond-spout model refit after removing, within
    session x spout, (time) a quadratic trend in trial time, (prev) the previous first-lick trial's spout (cell = session x
    spout x previous spout; previous = the preceding row of the session's first-lick table if its cue was <= 30 s
    earlier, else 'none'), (prevang) the previous trial's within-spout angle deviation (rows with a previous trial only).
    Each with an nshuf within-cell permutation null."""
    rows = []
    rng = np.random.default_rng(SEED + 1)
    T = T.sort_values(["label", "cue"]).reset_index(drop=True)
    T["cell0"] = T.label + "|" + T.pos
    prev_ok = (T.label == T.label.shift()) & (T.cue - T.cue.shift() <= 30)
    T["prev_pos"] = np.where(prev_ok, T.pos.shift().astype(str), "none")
    for lf in LF_RULES:
        fc = fcols(1.0, lf, names)
        # previous-trial angle deviation from ALL matched trials (not only lick-free ones)
        ok = np.isfinite(T.angle)
        aw = np.full(len(T), np.nan)
        aw[ok] = _resid(T.angle[ok].to_numpy(float), T.cell0[ok].to_numpy(str))
        T["prev_aw"] = np.where(prev_ok, pd.Series(aw).shift(), np.nan)
        S = select(T, 1.0, lf)
        for a in AN:
            Pr = S[(S.animal == a) & (S.epoch == "pre")].reset_index(drop=True)
            sess, pos = Pr.label.to_numpy(str), Pr.pos.to_numpy(str)
            cell = np.char.add(np.char.add(sess, "|"), pos)
            tz = Pr.groupby("label").cue.transform(lambda c: (c - c.mean()) / c.std()).to_numpy(float)
            variants = {"base": (Pr, cell, None),
                        "time": (Pr, cell, np.column_stack([tz, tz ** 2])),
                        "prev": (Pr, np.char.add(np.char.add(cell, "|"), Pr.prev_pos.to_numpy(str)), None)}
            m = np.isfinite(Pr.prev_aw.to_numpy(float))
            variants["prevang"] = (Pr[m].reset_index(drop=True), cell[m], Pr.prev_aw.to_numpy(float)[m][:, None])
            for name, (D, c, Z) in variants.items():
                X, y = D[fc].to_numpy(float), D.angle.to_numpy(float)
                s_ = D.label.to_numpy(str)
                Xr, yr = (_resid(X, c), _resid(y, c)) if Z is None else (_resid_cov(X, Z, c), _resid_cov(y, Z, c))

                def score(yy, s_=s_, Xr=Xr):
                    p = np.full(len(yy), np.nan)
                    for s in np.unique(s_):
                        te = s_ == s
                        p[te] = _pred(_fit(Xr[~te], yy[~te]), Xr[te])
                    return _r(p, yy)
                obs = score(yr)
                null = np.array([score(_perm_within(yr, c, rng)) for _ in range(nshuf)])
                rows.append({"animal": a, "lickfree": lf, "control": name, "n": len(y), "r_beyond": obs,
                             "null95": float(np.percentile(null, 95)), "p": float((1 + np.sum(null >= obs)) / (1 + nshuf)),
                             "r_prev_angle_vs_angle_w": _r(D.prev_aw.to_numpy(float), _resid(y, c))})
                print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "enl_angle_decode_controls.csv", index=False)
    return R


def prepare(T: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Trial table -> (with pos as str and the first-lick angle, feature names). + angle = toward + side_ml (contra)."""
    T["pos"] = T.pos.astype(str)
    T["angle"] = np.degrees(np.arcsin(np.clip(T.side_ml / T.reach_mm, -1, 1)))
    return T, [c[4:] for c in T.columns if c.startswith("w10:")]


def main() -> int:
    from wfield_local import writeguard
    writeguard.assert_writable(OUT / "enl_angle_decode_pre.csv")
    il.split_only("lick_enl_angle_decode.py")
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
    nshuf = int(sys.argv[sys.argv.index("--nshuf") + 1]) if "--nshuf" in sys.argv else 200
    T = extract(jobs) if (not TRIALS.exists() or "--extract" in sys.argv) else pd.read_csv(TRIALS)
    if "--extract-only" in sys.argv:
        return 0
    T, names = prepare(T)
    if "--plot-only" in sys.argv:
        PRE, SS, TS = (pd.read_csv(OUT / f"enl_angle_decode_{k}.csv") for k in ("pre", "sessions", "tests"))
        figure(PRE, SS, TS, 1.0, LF_PRIMARY)
        return 0
    if "--controls" in sys.argv:
        print(controls(T, names, nshuf).round(3).to_string(index=False))
        return 0
    print("== sign check (pre, mean side_ml by spout):")
    print(T[T.epoch == "pre"].groupby(["animal", "pos"]).side_ml.mean().unstack().round(2).to_string())
    cnt = []
    for (a, ep), g in T[T.epoch.isin(EP)].groupby(["animal", "epoch"]):
        k = np.isfinite(g.angle)
        k = k & (g.n_between == 0)
        cnt.append({"animal": a, "epoch": ep, "n_trials": len(g), "matched_kin": int(k.sum()),
                    "lf_0.5": int((k & (g.nl05 == 0)).sum()), "lf_1.0": int((k & (g.nl1 == 0)).sum()),
                    "lf_2.0": int((k & (g.nl2 == 0)).sum()),
                    **{f"slide_{W:g}": int((k & np.isfinite(g[f"slo{'10' if W == 1.0 else '05'}"])).sum())
                       for W in WINDOWS if f"slo{'10' if W == 1.0 else '05'}" in g}})
    C = pd.DataFrame(cnt)
    print("== trial counts (matched = first post-cue lick with 3-D kinematics; lf_L = + no lick onset in the last L s)")
    print(C.to_string(index=False))
    C.to_csv(OUT / "enl_angle_decode_counts.csv", index=False)
    PRE, SS, CF = [], [], []
    for W in WINDOWS:
        for lf in LF_RULES:
            p, s, c = run_variant(T, W, lf, nshuf, names)
            PRE.append(p); SS.append(s); CF.append(c)
            print(f"\n== PRE W={W} lickfree={lf}\n" + p.round(3).to_string(index=False), flush=True)
    PRE, SS, CF = (pd.concat(x, ignore_index=True) for x in (PRE, SS, CF))
    E, TS = epoch_tables(SS)
    PRE.to_csv(OUT / "enl_angle_decode_pre.csv", index=False)
    SS.to_csv(OUT / "enl_angle_decode_sessions.csv", index=False)
    E.to_csv(OUT / "enl_angle_decode_epochs.csv", index=False)
    TS.to_csv(OUT / "enl_angle_decode_tests.csv", index=False)
    CF.to_csv(OUT / "enl_angle_decode_coefs.csv", index=False)
    print("\n== per animal x epoch (frozen decoder; pre = LOSO)\n" + E.round(3).to_string(index=False))
    print("\n== change vs pre (paired nested bootstrap)\n" + TS.round(3).to_string(index=False))
    figure(PRE, SS, TS, 1.0, LF_PRIMARY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
