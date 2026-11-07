"""RECOVERY TIMING, HITS ONLY (Priya 10-25: "how strong is the evidence that cortex recovers at / after behaviour" -> run
the hits-only test; and "can we predict animals' recovery from cortical activity?").

Per post-stroke SESSION: behaviour = far-contra hit rate, THE Fig 0.2 / 1b rate (Priya 11-06, A21): column
``frac_hit_far_contra`` of the run's trial_attempts_sessions.csv (`trial_attempts.session_summary`: camera class contact
/ far-contra trials inside the CAMERA WORKING PERIOD -- denominator contact + incomplete + jaw_only + no_response; stopped,
stopped-licking, early-run, after-end and unknown trials excluded); a table built without the camera gate is refused.
Before 11-06 the denominator was every far-contra trial_attempts row (no period). Neural =
frozen-decoder far-contra retention on HITS only (`attempt_coding/decoder_retention_sessions.csv`, sessions with >= 5
far-contra hits; removes the composition dependence -- failed attempts carry no code). Both normalised within animal to
its pre-stroke level (behaviour: pre far-contra hit rate; neural: 1 = pre hits by construction).

  1. per animal x series (aim = PRIMARY behaviour, Priya 11-07; hit rate secondary; code on hits), on the recovered-
     fraction scale z = (x - acute) / (pre - acute) (1 = pre, 0 = the animal's acute epoch mean):
     COMPLETENESS = z by chronic; HALF-RECOVERY RELATIVE TO PRE = the day the exponential fit
     y(d) = a - (a - b) exp(-d / tau) reaches z = 0.5 (fit-free check: first crossing by interpolation between sessions,
     day 0 anchored at z = 0). The old tau ln 2 is the time to half of the animal's OWN plateau (free a) -- PS94's aim
     plateaus at ~0.55, so it never half-recovers to pre -- and survives only as ``*_t_to_own_plateau_half``.
     Fit-quality flag (A11, 11-07): ok / few_points / not_reached / bound (any parameter at FIT_BOUNDS) / unstable (fit
     fails, a leave-one-session-out fit fails or never reaches 0.5, or LOO max / min > 10) / few_hits (acute code on hits
     < analysis_kit.MIN_HITS = 20 hits) / no_acute / no_deficit. The headline ``*_half_to_pre`` is NaN unless "ok"; the
     raw fitted day stays in ``*_half_to_pre_fit``. 95 % CI from resampling the post sessions (1000 draws); cortex -
     behaviour (aim and hit rate) on the half-recovery-to-pre day with its CI.
  2. session-to-session lagged models (post sessions in day order, consecutive pairs within animal; animal fixed effects):
     behaviour[t] ~ behaviour[t-1] + neural[t-1]   (does cortex predict the NEXT session's behaviour beyond behaviour?)
     neural[t]    ~ neural[t-1] + behaviour[t-1]   (does behaviour predict the next session's cortex beyond cortex?)
     OLS with animal dummies, cluster-robust SE by animal is meaningless at n = 4, so a session-level bootstrap within
     animal is used for the CIs. With 4 animals, between-animal prediction (who recovers fastest) is descriptive only.
Caveat: acute sessions often have < 5 far-contra hits, so the neural series starts later than the behavioural one.

Usage: python scripts/recovery_timing.py [--window lick]
       python scripts/recovery_timing.py --aim [--plot-only]   (Fig 0.12 + far-contra AIM panel, 10-30 -> figures/
       recovery_timing_pooled_day_<window>_aim.png, recovery_timing_pooled_day_<window>_aim.csv,
       recovery_timing_aim_per_animal_<window>.csv, figdata/recovery_timing_pooled_day_<window>_aim.pkl)
Output: <E>/attempt_coding/recovery_timing_*.csv, figures/recovery_timing.png
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse

import numpy as np
import pandas as pd
from wfield_local.orofacial_outputs import encoding_root as _encoding_root  # rebuild root (11-03)

E = _encoding_root()
from wfield_local import attempt_coding as _ac  # noqa: E402
from wfield_local import lp_licks as _lp_licks  # noqa: E402  (coordinator 11-04: the run's attempts table)
from wfield_local import analysis_kit as ak  # noqa: E402  (MIN_HITS / hits_ok: the hit-gated minimum, Priya 11-07)

A = _ac.out_dir()   # the writer's folder under the run mode (review N1 / N10)
AN = ["PS92", "PS93", "PS94", "PS95"]
ACOL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}


#: THE camera working-period hit rate of `trial_attempts.session_summary` (as fig_behaviour_hits.CAMERA_COL)
HIT_COL = "frac_hit_far_contra"
DATA_HIT = ("far-contra hit rate = contact / camera-working-period trials (first cue -> last camera lick, stopped runs "
            "A2 + excise out; as Fig 0.2 / 1b)")


def far_contra_hit(W: pd.DataFrame) -> pd.DataFrame:
    """[label, hit] from the run's trial_attempts_sessions.csv ``W`` (`trial_attempts.session_summary`): ``HIT_COL``, the
    camera working-period hit rate. Never re-derived here (A21). REFUSES a table built without the camera trial gate
    (no ``frac_stopped_*`` columns), like fig_behaviour_hits.camera_metrics; sessions without far-contra trials drop."""
    if not any(c.startswith("frac_stopped_") for c in W.columns):
        raise SystemExit("trial_attempts_sessions.csv is not camera-gated (no frac_stopped_* columns) -- build it with "
                         "scripts/trial_attempts.py under WIDEFIELD_TRIAL_GATE=camera (the render profile)")
    return W[["label", HIT_COL]].rename(columns={HIT_COL: "hit"}).dropna(subset=["hit"])


def series(window: str) -> pd.DataFrame:
    from wfield_local import config
    f = _lp_licks.attempts_path()
    T = pd.read_csv(f, usecols=["label", "animal", "epoch", "day"]).drop_duplicates("label")
    T = T[~T.label.map(config.early_pre)]
    H = far_contra_hit(pd.read_csv(f.with_name("trial_attempts_sessions.csv")))
    beh = T.merge(H, on="label", how="inner")[["animal", "epoch", "label", "day", "hit"]]
    pre = beh[beh.epoch == "pre"].groupby("animal").hit.mean()
    beh["beh"] = beh.hit / beh.animal.map(pre)
    N = pd.read_csv(A / "decoder_retention_sessions.csv")
    N = N[(N.window == window) & (N.cls2 == "hit")][["label", "ret", "n"]].rename(columns={"ret": "neu", "n": "n_hits"})
    return beh.merge(N, on="label", how="left").sort_values(["animal", "day"]).reset_index(drop=True)


#: exponential recovery fit y(d) = a - (a - b) exp(-d / tau): parameter bounds (a, b, tau) -- unchanged since 10-25
FIT_BOUNDS = ([-2.0, -2.0, 0.3], [3.0, 3.0, 200.0])
#: leave-one-session-out instability: the LOO half-recovery days spread by more than this factor (max / min)
LOO_MAX_RATIO = 10.0
#: half-recovery RELATIVE TO PRE (Priya 11-07): the day the series reaches 0.5 on the 1 = pre / 0 = acute scale
HALF = 0.5
#: fit-quality flags (A11, Priya 11-07 "fitting code flag ok"). Only "ok" fills the headline column.
FLAGS = ("ok", "few_points", "not_reached", "bound", "unstable", "few_hits", "no_deficit")


def _expf(x, a, b, tau):
    return a - (a - b) * np.exp(-x / tau)


def fit_params(d, y):
    """(a, b, tau) of the exponential fit to the finite points, or None (< 4 points or a failed fit)."""
    from scipy.optimize import curve_fit
    d, y = np.asarray(d, float), np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(d)
    d, y = d[ok], y[ok]
    if len(d) < 4:
        return None
    try:
        p, _ = curve_fit(_expf, d, y, p0=(np.nanmax(y), np.nanmin(y), 5.0), bounds=FIT_BOUNDS, maxfev=20000)
        return tuple(float(v) for v in p)
    except (RuntimeError, ValueError):
        return None


def exp_fit(d, y):
    """tau ln 2 of the exponential fit = the time to HALF OF THE ANIMAL'S OWN FITTED PLATEAU (free a), NOT half-recovery
    to pre (Priya 11-07: PS94's aim plateaus at ~0.55, so its 3.1 d was "time to stop improving"). Reported only as
    the secondary column ``t_to_own_plateau_half``; the headline is `recovery_metrics`' ``half_to_pre``."""
    p = fit_params(d, y)
    return p[2] * np.log(2) if p is not None else np.nan


def at_bound(p) -> bool:
    """Any fitted parameter on (within 0.1 % of the range of) its FIT_BOUNDS bound."""
    lo, hi = np.asarray(FIT_BOUNDS[0]), np.asarray(FIT_BOUNDS[1])
    tol = 1e-3 * (hi - lo)
    p = np.asarray(p, float)
    return bool(np.any(np.abs(p - lo) <= tol) or np.any(np.abs(p - hi) <= tol))


def cross_day(p, level: float = HALF, d_max: float = np.inf) -> float:
    """First day the fitted curve reaches ``level`` (recovery = rising toward 1). 0 when the curve starts at/above it
    and its plateau stays there; NaN = not reached (plateau a < level, or the crossing lies beyond ``d_max``, the last
    observed post day)."""
    if p is None:
        return np.nan
    a, b, tau = p
    if a < level:
        return np.nan
    if b >= level:
        return 0.0
    x = -tau * np.log((a - level) / (a - b))
    return float(x) if x <= d_max else np.nan


def half_day_interp(d, z, level: float = HALF) -> float:
    """Fit-free check: first crossing of ``level`` by linear interpolation between consecutive post sessions in day
    order, with day 0 anchored at the acute level (z = 0). NaN = never reached."""
    d, z = np.asarray(d, float), np.asarray(z, float)
    ok = np.isfinite(d) & np.isfinite(z)
    o = np.argsort(d[ok], kind="stable")
    dd, zz = np.r_[0.0, d[ok][o]], np.r_[0.0, z[ok][o]]
    hit = np.flatnonzero(zz >= level)
    if not len(hit):
        return np.nan
    i = hit[0]
    if zz[i] == zz[i - 1]:
        return float(dd[i])
    return float(dd[i - 1] + (level - zz[i - 1]) / (zz[i] - zz[i - 1]) * (dd[i] - dd[i - 1]))


def recovered_fraction(x, acute, pre=1.0):
    """(x - acute) / (pre - acute): 1 = the animal's pre level, 0 = its acute level. NaN without a defined deficit."""
    x = np.asarray(x, float)
    if not np.isfinite(acute) or acute == pre:
        return x * np.nan
    return (x - acute) / (pre - acute)


def half_to_pre_fit(d, z):
    """Fitted half-recovery-to-pre day of a recovered-fraction series (NaN if not reached / no fit)."""
    d = np.asarray(d, float)
    z = np.asarray(z, float)
    ok = np.isfinite(z) & np.isfinite(d)
    return cross_day(fit_params(d, z), HALF, d[ok].max() if ok.any() else np.nan)


def recovery_metrics(d, z, chronic, gate: str | None = None) -> dict:
    """Per animal x series, on the recovered-fraction scale ``z`` (1 = pre, 0 = acute) of its post sessions ``d``:
    completeness (mean z over the ``chronic`` sessions), the fitted and the interpolated half-recovery-to-pre day,
    the old tau ln 2 (``t_to_own_plateau_half``), the leave-one-session-out range of the fitted day, and the
    fit-quality flag. ``half_to_pre`` (THE headline) is the fitted day when the flag is "ok", else NaN.
    Flags, first match wins: ``gate`` (e.g. "few_hits", "no_deficit"); "few_points" (< 4 sessions); "unstable" (fit
    failed); "not_reached" (fitted curve never reaches 0.5 within the observed days); "bound" (any parameter at its
    FIT_BOUNDS bound); "unstable" (any LOO fit fails / does not reach 0.5, or LOO max / min > LOO_MAX_RATIO)."""
    d, z, chronic = np.asarray(d, float), np.asarray(z, float), np.asarray(chronic, bool)
    ok = np.isfinite(z) & np.isfinite(d)
    zc = z[chronic & ok]
    r = {"complete": float(zc.mean()) if len(zc) and gate is None else np.nan, "half_to_pre": np.nan,
         "half_to_pre_fit": np.nan, "half_to_pre_interp": np.nan, "t_to_own_plateau_half": np.nan,
         "loo_lo": np.nan, "loo_hi": np.nan, "n_sessions": int(ok.sum()), "flag": gate or "ok"}
    if gate is not None:
        return r
    r["half_to_pre_interp"] = half_day_interp(d, z)
    if ok.sum() < 4:
        r["flag"] = "few_points"
        return r
    p = fit_params(d, z)
    if p is None:
        r["flag"] = "unstable"
        return r
    dmax = float(d[ok].max())
    r["half_to_pre_fit"] = cross_day(p, HALF, dmax)
    r["t_to_own_plateau_half"] = p[2] * np.log(2)
    idx = np.flatnonzero(ok)
    loo = np.array([half_to_pre_fit(d[np.setdiff1d(idx, [i])], z[np.setdiff1d(idx, [i])]) for i in idx])
    if np.isfinite(loo).any():
        r["loo_lo"], r["loo_hi"] = float(np.nanmin(loo)), float(np.nanmax(loo))
    lo_, hi_ = r["loo_lo"], r["loo_hi"]
    ratio = np.inf if not np.isfinite(loo).all() else (hi_ / lo_ if lo_ > 0 else (1.0 if hi_ == 0 else np.inf))
    if not np.isfinite(r["half_to_pre_fit"]):
        r["flag"] = "not_reached"
    elif at_bound(p):
        r["flag"] = "bound"
    elif ratio > LOO_MAX_RATIO:
        r["flag"] = "unstable"
    if r["flag"] == "ok":
        r["half_to_pre"] = r["half_to_pre_fit"]
    return r


def fmt_half(r: dict, unit: str = "d") -> str:
    """Figure annotation of one recovery_metrics row: the headline day, or 'unresolved (<flag>)'."""
    if r.get("flag") == "few_hits":
        from wfield_local.analysis_kit import HIT_GATE_NA
        return HIT_GATE_NA
    h = r.get("half_to_pre", np.nan)
    c = r.get("complete", np.nan)
    cs = f", {c:.0%} complete" if np.isfinite(c) else ""
    return (f"{h:.1f} {unit}" if np.isfinite(h) else f"unresolved ({r.get('flag')})") + cs


#: series, PRIMARY FIRST (Priya 11-07: "we should use aim rather than hit recovery"): key, label, pre level
SERIES = [("aim", "far-contra first-lick aim", 0.0), ("beh", "far-contra hit rate", 1.0),
          ("neu", "cortex: far-contra code on hits", 1.0)]
SER_SHORT = {"aim": "aim (primary)", "beh": "hit rate (secondary)", "neu": "cortex code on hits"}
#: flags that mean "no acute reference": the recovered-fraction scale is undefined
NO_SCALE = ("few_hits", "no_acute", "no_deficit")


def epoch_value(g: pd.DataFrame, m: str, ep: str) -> float:
    """Animal x epoch mean of series ``m`` over its sessions. Code on hits (``neu``) is NaN below
    analysis_kit.MIN_HITS hit trials summed over the contributing sessions (Priya 11-07)."""
    from wfield_local import analysis_kit as ak
    v = g[(g.epoch == ep) & np.isfinite(g[m].to_numpy(float))]
    if m == "neu" and not ak.hits_ok(v.n_hits.to_numpy(float)):
        return np.nan
    return float(v[m].mean()) if len(v) else np.nan


def per_animal_recovery(M: pd.DataFrame) -> pd.DataFrame:
    """Per animal (rule 8) x series in SERIES present in ``M`` (columns animal, epoch, day, <series>, n_hits):
    epoch means (code on hits gated by MIN_HITS), recovered fraction of the acute deficit (x - acute) / (pre - acute)
    by subacute / chronic (``*_complete`` = by chronic = COMPLETENESS), half-recovery RELATIVE TO PRE (fit + interp),
    the old tau ln 2 as ``*_t_to_own_plateau_half``, LOO range, fit-quality flag, and the summed hits behind the
    acute / chronic code values."""
    from wfield_local import analysis_kit as ak
    rows = []
    for an in AN:
        g = M[M.animal == an]
        post = g[g.epoch != "pre"].sort_values("day")
        r = {"animal": an}
        for m, _lab, pre_v in SERIES:
            if m not in g:
                continue
            for ep in EP4:
                r[f"{m}_{ep}"] = epoch_value(g, m, ep)
            ac = r[f"{m}_acute"]
            gate = None
            if m == "neu":
                for ep in ("acute", "chronic"):
                    v = g[(g.epoch == ep) & np.isfinite(g.neu.to_numpy(float))]
                    r[f"neu_hits_{ep}"] = int(np.nansum(v.n_hits.to_numpy(float)))
                if not ak.hits_ok([r["neu_hits_acute"]]):
                    gate = "few_hits"
            if gate is None and not np.isfinite(ac):
                gate = "no_acute"
            elif gate is None and (ac == pre_v or (pre_v == 1.0 and ac > pre_v)):
                gate = "no_deficit"
            for ep in ("subacute", "chronic"):
                r[f"{m}_recov_{ep}"] = ((r[f"{m}_{ep}"] - ac) / (pre_v - ac)) if gate is None else np.nan
            z = recovered_fraction(post[m].to_numpy(float), ac if gate is None else np.nan, pre_v)
            q = recovery_metrics(post.day.to_numpy(float), z, (post.epoch == "chronic").to_numpy(), gate)
            q["complete"] = r[f"{m}_recov_chronic"]          # ONE definition: the gated epoch-mean recovered fraction
            r.update({f"{m}_{k}": v for k, v in q.items()})
        rows.append(r)
    return pd.DataFrame(rows)


def print_per_animal(PA: pd.DataFrame, title: str) -> None:
    """Rule 8 table, one block per series (primary first)."""
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(title + "\n  complete = recovered fraction of the acute deficit by chronic; half_to_pre = fitted day the "
          "series gets half-way from acute back to PRE (headline; NaN unless flag ok); _interp = fit-free crossing; "
          "t_to_own_plateau_half = old tau ln2 (half of the animal's OWN plateau, NOT recovery to pre)")
    for m, _l, _p in SERIES:
        c = ["animal"] + [f"{m}_{k}" for k in ("complete", "half_to_pre", "half_to_pre_fit", "half_to_pre_interp", "flag",
                                               "loo_lo", "loo_hi", "t_to_own_plateau_half", "n_sessions")
                          if f"{m}_{k}" in PA]
        if m == "neu":
            c += [x for x in ("neu_hits_acute", "neu_hits_chronic") if x in PA]
        if len(c) > 1:
            print(f"  -- {SER_SHORT[m]}")
            print(PA[c].round(2).to_string(index=False))


def half_lines(r, sep: str = " | ") -> str:
    """'aim 4.2 d, 55% complete | hit ... | cortex ...' for one per-animal row (columns of per_animal_recovery)."""
    out = []
    for m, _l, _p in SERIES:
        if f"{m}_flag" in r:
            out.append(("cortex" if m == "neu" else "hit" if m == "beh" else "aim") + " "
                       + fmt_half({k: r[f"{m}_{k}"] for k in ("half_to_pre", "complete", "flag")}))
    return sep.join(out)


def diff_line(r, b: str) -> str:
    """cortex - behaviour on the half-recovery-to-pre day, with its CI, or 'unresolved'."""
    nm = "aim" if b == "aim" else "hit"
    v = r.get(f"neural_minus_{b}", np.nan)
    if not np.isfinite(v):
        return f"cortex - {nm}: unresolved"
    return f"cortex - {nm} {v:+.1f} d [{r[f'neural_minus_{b}_lo']:+.1f}, {r[f'neural_minus_{b}_hi']:+.1f}]"


def _aim_by_label() -> pd.DataFrame | None:
    try:
        return aim_series()[["label", "aim"]]
    except (FileNotFoundError, OSError) as e:
        print(f"!! aim table unavailable ({e}); aim rows skipped")
        return None


def halftime_bootstrap(S: pd.DataFrame, Sm: pd.DataFrame, PA: pd.DataFrame, rng, n: int = 1000) -> pd.DataFrame:
    """95 % CI of each series' half-recovery-to-pre day and of cortex - behaviour (aim primary, hit rate secondary)
    from resampling each animal's post sessions (the pre-11-07 session bootstrap of this script, same draw sequence;
    acute references held fixed). Draws whose fit never reaches 0.5 are dropped; *_boot_share_reached = the share
    that did. Not an analysis_kit primitive (none resamples a per-animal fit); the LOO range in PA is the
    stability readout the fit-quality flag uses."""
    post, postm = S[S.epoch != "pre"], Sm[Sm.epoch != "pre"]
    keys = [m for m, _l, _p in SERIES]
    rows = []
    for an in AN:
        g, gm = post[post.animal == an], postm[postm.animal == an]
        d = g.day.to_numpy(float)
        pr = PA[PA.animal == an].iloc[0]
        Z = {m: recovered_fraction(gm[m].to_numpy(float), np.nan if pr[f"{m}_flag"] in NO_SCALE else pr[f"{m}_acute"],
                                   pre_v) for m, _l, pre_v in SERIES}
        bs = []
        for _ in range(n):
            i = rng.integers(0, len(g), len(g))
            bs.append([half_to_pre_fit(d[i], Z[m][i]) if np.isfinite(Z[m]).any() else np.nan for m in keys])
        bs = np.array(bs)
        q = lambda x: np.nanpercentile(x, [2.5, 97.5]) if np.isfinite(x).sum() > 50 else (np.nan, np.nan)  # noqa: E731
        r = {"animal": an}
        for k, m in enumerate(keys):
            r[f"{m}_boot_lo"], r[f"{m}_boot_hi"] = q(bs[:, k])
            r[f"{m}_boot_share_reached"] = float(np.isfinite(bs[:, k]).mean())
        for b in ("aim", "beh"):                 # cortex vs behaviour on the half-recovery-to-pre day
            diff = bs[:, keys.index("neu")] - bs[:, keys.index(b)]
            ok = np.isfinite(diff)
            r[f"neural_minus_{b}"] = pr["neu_half_to_pre"] - pr[f"{b}_half_to_pre"]
            r[f"neural_minus_{b}_lo"], r[f"neural_minus_{b}_hi"] = q(diff)
            r[f"share_neural_later_{b}"] = float((diff[ok] > 0).mean()) if ok.any() else np.nan
        r.update({"first_neural_day": float(g[np.isfinite(g.neu)].day.min()) if np.isfinite(g.neu).any() else np.nan,
                  "n_post_sessions": len(g), "n_neural_sessions": int(np.isfinite(g.neu).sum())})
        rows.append(r)
    H = PA.merge(pd.DataFrame(rows), on="animal")
    print("== per animal: half-recovery-to-pre day, 95% CI from resampling post sessions; cortex - behaviour on that day")
    print(H[["animal"] + [c for c in H if "boot" in c or c.startswith(("neural_minus", "share_"))]].round(2)
          .to_string(index=False))
    return H


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default="lick", choices=("precue", "cue", "lick"))
    a = ap.parse_args(argv)
    S = series(a.window)
    G = _aim_by_label()
    Sm = S.merge(G, on="label", how="left") if G is not None else S.assign(aim=np.nan)
    PA = per_animal_recovery(Sm)
    print_per_animal(PA, "== per animal (rule 8), BEFORE the bootstrap: completeness + half-recovery RELATIVE TO PRE")
    rng = np.random.default_rng(0)
    H = halftime_bootstrap(S, Sm, PA, rng)      # consumes rng exactly as the pre-11-07 loop did (lagged CIs unchanged)
    post = S[S.epoch != "pre"].copy()

    # lagged models, consecutive post sessions within animal (both measures present)
    L = []
    for an, g in post.groupby("animal"):
        g = g.sort_values("day").reset_index(drop=True)
        for i in range(1, len(g)):
            L.append({"animal": an, "day": g.day[i], "beh": g.beh[i], "neu": g.neu[i], "beh_prev": g.beh[i - 1],
                      "neu_prev": g.neu[i - 1], "gap": g.day[i] - g.day[i - 1]})
    L = pd.DataFrame(L).dropna()

    def fit(df, y, xs):
        X = np.column_stack([df[x].to_numpy(float) for x in xs] + [(df.animal == an).to_numpy(float) for an in AN])
        X = X[:, np.abs(X).sum(0) > 0]
        beta = np.linalg.lstsq(X, df[y].to_numpy(float), rcond=None)[0]
        return beta[:len(xs)]

    out = []
    for y, xs in (("beh", ["beh_prev", "neu_prev"]), ("neu", ["neu_prev", "beh_prev"])):
        b0 = fit(L, y, xs)
        bb = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(np.flatnonzero(L.animal == an), (L.animal == an).sum())
                                  for an in L.animal.unique()])
            bb.append(fit(L.iloc[idx], y, xs))
        bb = np.array(bb)
        for k, x in enumerate(xs):
            lo, hi = np.percentile(bb[:, k], [2.5, 97.5])
            out.append({"outcome": y + "[t]", "predictor": x.replace("_prev", "[t-1]"), "beta": b0[k], "lo": lo, "hi": hi,
                        "n_pairs": len(L)})
    M = pd.DataFrame(out)
    print("== session-to-session lagged models (animal fixed effects; bootstrap within animal)")
    print(M.round(3).to_string(index=False))
    H.to_csv(A / f"recovery_timing_halftimes_{a.window}.csv", index=False)
    M.to_csv(A / f"recovery_timing_lagged_{a.window}.csv", index=False)
    S.to_csv(A / f"recovery_timing_series_{a.window}.csv", index=False)

    per_animal_figure(Sm, H, a.window, plt)
    pooled_figures(S, a.window, plt)
    return 0


# ---------------------------------------------------------------------------------------- figure conventions (11-07)
# Priya 11-07: per-animal SESSION points = OPEN markers, shape = animal, COLOUR = EPOCH (the repo's maps:
# cat_points.animal_marker, epoch_figures.epoch_dot_color -- as section_g_retained); pooled day-bin means = filled
# markers coloured by the bin's epoch, CI band per epoch; lines only WITHIN an epoch, never across an epoch boundary;
# measures in separate panels; top spine hidden (sig_marks.reserve_headroom).
MEAS_LABEL = {"aim": "far-contra first-lick aim,\nchange vs own pre (deg)\n+ = contralesional",
              "beh": "far-contra hit rate / pre", "neu": "cortex: far-contra frozen\ncode on hits / pre",
              "gap": "gap: cortex - behaviour"}
MEAS_REF = {"aim": 0.0, "beh": 1.0, "neu": 1.0, "gap": 0.0}


def _ecol(ep):
    from wfield_local import epoch_figures as ef
    return ef.epoch_dot_color(ep)


def _bin_epoch(epochs) -> str:
    """The epoch of a day bin = the most common epoch of its sessions (bins never straddle pre / post)."""
    e = pd.Series(list(epochs)).dropna()
    return str(e.mode().iloc[0]) if len(e) else ""


def _epoch_runs(T: pd.DataFrame):
    """Contiguous same-epoch runs of rows (already in x / day order): lines are drawn within a run only."""
    if not len(T):
        return
    k = (T.epoch != T.epoch.shift()).cumsum()
    for _, run in T.groupby(k, sort=False):
        yield run.epoch.iloc[0], run


def _session_points(ax, P: pd.DataFrame, col: str, lines: bool = False) -> None:
    """Open markers per session (shape = animal, colour = epoch); optional thin per-animal lines within an epoch."""
    from wfield_local import cat_points as cp
    P = P[np.isfinite(P[col].to_numpy(float))].sort_values(["animal", "day"])
    for an, g in P.groupby("animal"):
        for ep, run in _epoch_runs(g):
            if lines and len(run) > 1:
                ax.plot(run.day, run[col], color=_ecol(ep), lw=0.7, alpha=0.5, zorder=2)
            ax.scatter(run.day, run[col], marker=cp.animal_marker(an), facecolors="none", edgecolors=[_ecol(ep)],
                       s=18 if not lines else 22, lw=0.9, alpha=0.6 if not lines else 0.9, zorder=3)


def _bin_means(ax, T: pd.DataFrame) -> None:
    """Pooled day-bin means: filled markers + CI band coloured by the bin's epoch, joined only within an epoch."""
    T = T.sort_values("x")
    for ep, run in _epoch_runs(T):
        c = _ecol(ep)
        if len(run) > 1:
            ax.fill_between(run.x, run.lo, run.hi, color=c, alpha=0.16, lw=0, zorder=1)
            ax.plot(run.x, run["mean"], color=c, lw=2, zorder=4)
        else:
            ax.vlines(run.x, run.lo, run.hi, color=c, lw=2, alpha=0.5, zorder=4)
        ax.scatter(run.x, run["mean"], color=c, s=34, edgecolor="k", lw=0.4, zorder=5)


def _conv_legend(fig, animals, epochs, extra=(), **kw) -> None:
    from matplotlib.lines import Line2D

    from wfield_local import epoch_figures as ef
    eh = [Line2D([], [], ls="none", marker="o", mfc="none", mec=_ecol(e), mew=1.1, ms=6, label=e) for e in epochs]
    fig.legend(handles=ef._shape_handles(animals) + eh + list(extra), frameon=False, **kw)


def _tidy_axes(fig) -> None:
    from wfield_local import sig_marks
    for ax in fig.get_axes():
        ax.spines["right"].set_visible(False)
    sig_marks.reserve_headroom(fig)          # hides the top spine on every panel (no stars here)


def per_animal_figure(Sm: pd.DataFrame, H: pd.DataFrame, window: str, plt) -> None:
    """Per animal (columns) x measure (rows: aim PRIMARY, hit rate, code on hits): session points (open, shape =
    animal, colour = epoch, lines within an epoch only) + the per-series half-recovery-to-pre / completeness / flag."""
    rows = [m for m, _l, _p in SERIES if m in Sm]
    fig, axs = plt.subplots(len(rows), 4, figsize=(17, 2.9 * len(rows) + 1.6), sharex=True, sharey="row", squeeze=False)
    for k, an in enumerate(AN):
        g = Sm[Sm.animal == an]
        r = H[H.animal == an].iloc[0]
        for i, m in enumerate(rows):
            ax = axs[i, k]
            ax.axvspan(g.day.min() - 1, 0, color="0.95", lw=0, zorder=0)
            ax.axhline(MEAS_REF[m], color="0.5", ls="--", lw=0.8)
            _session_points(ax, g.assign(**{m: g[m]}), m, lines=True)
            t = f"{an} {('aim', 'hit rate', 'cortex')[['aim', 'beh', 'neu'].index(m)]}: " + fmt_half(
                {q: r[f"{m}_{q}"] for q in ("half_to_pre", "complete", "flag")})
            if m == "neu":
                t += "\n" + diff_line(r, "aim") + "; " + diff_line(r, "beh")
            ax.set_title(t, fontsize=7.5)
            if k == 0:
                ax.set_ylabel(MEAS_LABEL[m], fontsize=8.5)
            if i == len(rows) - 1:
                ax.set_xlabel("days from stroke")
    _tidy_axes(fig)
    _conv_legend(fig, AN, [e for e in EP4 if (Sm.epoch == e).any()], loc="lower center", ncol=8, fontsize=8.5)
    fig.suptitle(f"Recovery timing per animal ({window} window): aim (PRIMARY behaviour), far-contra hit rate (secondary), "
                 "frozen far-contra code on HITS\n" + DATA_HIT + "\nTitles: half-recovery RELATIVE TO PRE = day the "
                 "exponential fit gets half-way from the acute level back to pre;\n% = completeness (recovered fraction "
                 "of the acute deficit by chronic); 'unresolved' = fit flag not ok; code on hits needs >= "
                 f"{ak.MIN_HITS} hits per animal x epoch (acute sets the scale); 95% CI by resampling sessions",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    out = A / "figures" / f"recovery_timing_{window}.png"
    save_png_svg(fig, out, dpi=120)
    plt.close(fig)
    print("->", out)


#: day bins for pooling across animals (sessions fall on nearly the same days; narrow early, wider late)
DAY_BINS = [(-13, -1), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 9), (10, 12), (13, 16), (17, 20), (21, 24),
            (25, 28), (29, 33), (34, 38), (39, 44), (45, 51)]
EP4 = ["pre", "acute", "subacute", "chronic"]


def _boot_mean_of_animals(by: dict, rng, n=2000):
    """Mean of animal means and 95 % CI, resampling animals then sessions (levels; animal-weighted)."""
    an = [k for k, v in by.items() if len(v)]
    if not an:
        return np.nan, np.nan, np.nan
    point = float(np.mean([np.mean(by[k]) for k in an]))
    if len(an) < 2:
        return point, np.nan, np.nan
    d = [np.mean([np.mean(rng.choice(by[k], len(by[k]))) for k in rng.choice(an, len(an))]) for _ in range(n)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return point, float(lo), float(hi)


def _hit_gated(g: pd.DataFrame, m: str) -> np.ndarray:
    """Session values of series ``m`` for ONE animal x epoch; a code-on-hits series (``neu`` and the ``gap`` built
    from it) is empty when its sessions sum to < analysis_kit.MIN_HITS hits (Priya 11-07)."""
    v = g[np.isfinite(g[m].to_numpy(float))]
    if m in ("neu", "gap") and not ak.hits_ok(v.n_hits.to_numpy(float)):
        return np.array([])
    return v[m].to_numpy(float)


def pooled_figures(S: pd.DataFrame, window: str, plt) -> None:
    """POOLED ACROSS ANIMALS (Priya 10-25): per-animal-normalised behaviour and cortex (hits) (a) by day bin and (b) by
    epoch with one dot per session, plus the per-session GAP cortex - behaviour (does behaviour recover more than cortex?)."""
    from matplotlib.lines import Line2D

    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(0)
    S = S.copy()
    S["gap"] = S.neu - S.beh
    MEAS = [("beh", "behaviour: far-contra hit rate", "0.25"), ("neu", "cortex: far-contra code on hits", "#c0392b")]
    # (a) by day bin
    rows = []
    for lo, hi in DAY_BINS:
        g = S[(S.day >= lo) & (S.day <= hi)]
        x = float(np.mean([lo, hi])) if lo > 0 else -6.0
        for m, _l, _c in MEAS:
            by = {an: g[g.animal == an][m].dropna().to_numpy() for an in AN}
            pt, l_, h_ = _boot_mean_of_animals(by, rng)
            rows.append({"bin": f"{lo}..{hi}", "x": x, "measure": m, "mean": pt, "lo": l_, "hi": h_,
                         "n_animals": sum(len(v) > 0 for v in by.values()),
                         **{an: (float(np.mean(v)) if len(v) else np.nan) for an, v in by.items()},
                         "epoch": _bin_epoch(g[g[m].notna()].epoch)})
    Dd = pd.DataFrame(rows)
    Dd.to_csv(A / f"recovery_timing_pooled_day_{window}.csv", index=False)
    fig, axs = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True)
    for ax, (m, lab, _col) in zip(axs, MEAS):
        ax.axvspan(-13, 0, color="0.95", lw=0, zorder=0)
        ax.axhline(1, color="0.5", ls="--", lw=0.8)
        _session_points(ax, S, m)
        _bin_means(ax, Dd[Dd.measure == m])
        ax.set_ylabel(lab + "\n/ animal's pre level", fontsize=9)
    axs[-1].set_xlabel("days from stroke (binned; pre = all August pre sessions at x = -6)")
    _tidy_axes(fig)
    _conv_legend(fig, AN, [e for e in EP4 if (S.epoch == e).any()], loc="lower center", ncol=8, fontsize=8.5)
    fig.suptitle(f"Recovery, pooled across animals ({window} window, hits only for cortex): open = sessions (shape = "
                 "animal, colour = epoch);\nfilled = mean of animals per day bin, band = 95% CI (animals > sessions)\n"
                 + DATA_HIT, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save_png_svg(fig, A / "figures" / f"recovery_timing_pooled_day_{window}.png", dpi=120)
    plt.close(fig)

    # (b) by epoch, one dot per session; change vs pre (boot_delta); the gap panel
    PAN = MEAS + [("gap", "gap: cortex - behaviour", "#2471a3")]
    MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    tst = []
    for k, (m, lab, _col) in enumerate(PAN):
        ax = axs[k]
        ax.axhline(1 if m != "gap" else 0, color="0.5", ls="--", lw=0.8)
        ref = {an: _hit_gated(S[(S.animal == an) & (S.epoch == "pre")], m) for an in AN}
        ref = {an: v for an, v in ref.items() if len(v)}
        for i, ep in enumerate(EP4):
            g = S[S.epoch == ep]
            by = {an: _hit_gated(g[g.animal == an], m) for an in AN}
            na = [an for an in AN if m != "beh" and len(g[g.animal == an].neu.dropna()) and not len(by[an])]
            if na:                                   # animal x epoch below MIN_HITS: named, not plotted / tested
                ax.text(i, -0.25 if m != "gap" else -1.05, ak.HIT_GATE_NA + ": " + "/".join(na), ha="center",
                        va="bottom", fontsize=6.5, color="0.35")
            by = {an: v for an, v in by.items() if len(v)}
            for j, an in enumerate(AN):
                if an in by:
                    xs = i + (j - 1.5) * 0.07 + np.random.default_rng(10 * i + j).uniform(-0.02, 0.02, len(by[an]))
                    ax.scatter(xs, by[an], marker=MK[an], s=22, facecolors="none", edgecolors=[_ecol(ep)], lw=0.9,
                               alpha=0.8, zorder=4)
            if by:
                ax.plot([i - 0.3, i + 0.3], [np.mean([v.mean() for v in by.values()])] * 2, color="k", lw=2.2, zorder=5)
            com = [an for an in by if an in ref]
            if ep != "pre" and len(com) >= 2:
                iv = ak.boot_delta({an: by[an] for an in com}, {an: ref[an] for an in com}, np.random.default_rng(0))
                if iv is not None:
                    sig = iv.lo > 0 or iv.hi < 0
                    tst.append({"measure": m, "epoch": ep, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                                "animals": "/".join(com), "n_sessions": sum(len(by[an]) for an in com)})
                    ax.text(i, 1.42 if m != "gap" else 0.62, f"{iv.point:+.2f}\n[{iv.lo:+.2f}, {iv.hi:+.2f}]", ha="center",
                            va="top", fontsize=7.5, fontweight="bold" if sig else "normal")
        ax.set_xticks(range(4)); ax.set_xticklabels(EP4)
        ax.set_ylim(*((-0.3, 1.45) if m != "gap" else (-1.1, 0.65)))
        ax.set_title(lab, fontsize=10)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if k == 0:
            ax.set_ylabel("relative to the animal's pre-stroke level")
    Tt = pd.DataFrame(tst)
    Tt.to_csv(A / f"recovery_timing_pooled_epoch_{window}.csv", index=False)
    print("== pooled by epoch: change vs pre (nested bootstrap animals > sessions)\n" + Tt.round(3).to_string(index=False))
    _tidy_axes(fig)
    _conv_legend(fig, AN, EP4, [Line2D([], [], color="k", lw=2.2, label="mean of animals")], loc="lower center",
                 ncol=9, fontsize=8.5)
    fig.suptitle(f"Recovery by epoch, one dot per session, normalised within animal ({window} window; cortex = hits only). "
                 "Numbers = change vs pre, 95% nested CI (bold = excludes 0)\ncortex: sessions with >= 5 hits, and an "
                 f"animal x epoch needs >= {ak.MIN_HITS} hits summed (else {ak.HIT_GATE_NA})\n"
                 + DATA_HIT,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    save_png_svg(fig, A / "figures" / f"recovery_timing_pooled_epoch_{window}.png", dpi=120)
    plt.close(fig)
    print("->", A / "figures" / f"recovery_timing_pooled_epoch_{window}.png")


import iti_licks as _il  # noqa: E402
AIM_CSV = _il.OUT / "aim_by_spout_sessions.csv"                  # il.OUT: the ITI family's mode-routed root (N10)      # scripts/lick_cued_aim_by_spout.py (THE aim definition)
AIM_COL = "#2471a3"


def aim_series() -> pd.DataFrame:
    """Per session: far-contra FIRST post-cue lick 3-D aim offset, change vs the animal's own (August) pre mean, deg,
    + = contralesional (iti_recovery_measure.aim_licks via lick_cued_aim_by_spout.session_spout; >= 5 first licks)."""
    from wfield_local import config
    G = pd.read_csv(AIM_CSV)
    G = G[(G.pos == "far_contra") & (~G.label.map(config.early_pre))]
    return G[["animal", "label", "epoch", "day", "chg", "n"]].rename(columns={"chg": "aim"}).sort_values(["animal", "day"])


def aim_compute(window: str) -> dict:
    """Fig 0.12 + AIM (Priya 10-30: "can 0.12 include the far contra lick angle index?"). Panel 1 = the existing pooled
    day-bin figure (hit rate / pre, code on hits / pre; same rng order, so it reproduces recovery_timing_pooled_day);
    panel 2 = far-contra aim change vs own pre (deg) by the same day bins, per-animal + animal-weighted mean + 95% CI.
    Also per animal (`per_animal_recovery`; aim = PRIMARY behaviour, Priya 11-07): acute / subacute / chronic means,
    recovered fraction of the acute deficit (x - acute) / (pre - acute) per measure (code on hits NaN below
    analysis_kit.MIN_HITS hits per animal x epoch), and half-recovery RELATIVE TO PRE (fit + interpolation) with its
    fit-quality flag and LOO range; the old tau ln 2 survives only as ``*_t_to_own_plateau_half``."""
    S = pd.read_csv(A / f"recovery_timing_series_{window}.csv")
    G = aim_series()
    rng = np.random.default_rng(0)
    rows = []
    for lo, hi in DAY_BINS:                                    # == pooled_figures (a), same draw order
        g = S[(S.day >= lo) & (S.day <= hi)]
        x = float(np.mean([lo, hi])) if lo > 0 else -6.0
        for m in ("beh", "neu"):
            by = {an: g[g.animal == an][m].dropna().to_numpy() for an in AN}
            pt, l_, h_ = _boot_mean_of_animals(by, rng)
            rows.append({"bin": f"{lo}..{hi}", "x": x, "measure": m, "mean": pt, "lo": l_, "hi": h_,
                         "n_animals": sum(len(v) > 0 for v in by.values()),
                         **{an: (float(np.mean(v)) if len(v) else np.nan) for an, v in by.items()},
                         "epoch": _bin_epoch(g[g[m].notna()].epoch)})
    rng2 = np.random.default_rng(1)
    for lo, hi in DAY_BINS:
        g = G[(G.day >= lo) & (G.day <= hi)]
        x = float(np.mean([lo, hi])) if lo > 0 else -6.0
        by = {an: g[g.animal == an].aim.dropna().to_numpy() for an in AN}
        pt, l_, h_ = _boot_mean_of_animals(by, rng2)
        rows.append({"bin": f"{lo}..{hi}", "x": x, "measure": "aim", "mean": pt, "lo": l_, "hi": h_,
                     "n_animals": sum(len(v) > 0 for v in by.values()),
                     **{an: (float(np.mean(v)) if len(v) else np.nan) for an, v in by.items()},
                     "epoch": _bin_epoch(g[g.aim.notna()].epoch)})
    D = pd.DataFrame(rows)
    # per animal (rule 8): epoch means, recovered fraction of the acute deficit, half-recovery day
    M = S.merge(G[["label", "aim"]], on="label", how="outer")
    M["animal"] = M.label.str[:4]
    M["epoch"] = M.epoch.fillna(M.label.map(G.set_index("label").epoch))
    M["day"] = M.day.fillna(M.label.map(G.set_index("label").day))
    PA = per_animal_recovery(M)           # aim (primary), hit rate, code on hits: completeness, half-to-pre, flags
    P = M[["animal", "epoch", "day", "beh", "neu", "aim"]].copy()       # session points for the figure (plot-only)
    return {"D": D, "PA": PA, "P": P, "window": window}


def aim_draw(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, window = F["D"], F["window"]
    P = F.get("P")
    # aim = PRIMARY behavioural recovery measure (Priya 11-07: "we should use aim rather than hit recovery") -> panel A;
    # hit rate (secondary) -> B; code on hits -> C. Conventions as per_animal_figure (11-07).
    PAN = [("aim", "A  PRIMARY: far-contra aim, first post-cue lick, 3-D angle minus the animal's pre mean "
                   "(- = shifted ipsilesional)"),
           ("beh", "B  far-contra hit rate (secondary behaviour measure)"),
           ("neu", "C  frozen decoder far-contra code on hits")]
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 10.5), sharex=True)
    for ax, (m, ttl) in zip(axs, PAN):
        ax.axvspan(-13, 0, color="0.95", lw=0, zorder=0)
        ax.axhline(MEAS_REF[m], color="0.5", ls="--", lw=0.8)
        if P is not None and "epoch" in D:
            _session_points(ax, P, m)
            _bin_means(ax, D[D.measure == m])
        else:                                        # figdata from before 11-07: no session points / bin epochs
            t = D[D.measure == m].sort_values("x")
            ax.fill_between(t.x, t.lo, t.hi, color="0.6", alpha=0.15, lw=0)
            ax.plot(t.x, t["mean"], "-o", color="0.3", lw=2, ms=4)
        ax.set_ylabel(MEAS_LABEL[m], fontsize=9)
        ax.set_title(ttl, fontsize=10, loc="left")
    axs[-1].set_xlabel("days from stroke (binned; pre = all August pre sessions at x = -6)")
    _tidy_axes(fig)
    if P is not None:
        _conv_legend(fig, AN, [e for e in EP4 if (P.epoch == e).any()], loc="lower center", ncol=8, fontsize=8,
                     bbox_to_anchor=(0.5, 0.115))
    PA = F.get("PA")
    if PA is not None and "aim_flag" in PA:     # per-animal completeness + half-recovery relative to pre (fit, flagged)
        txt = ["Per animal: half-recovery RELATIVE TO PRE (fitted day half-way from acute back to pre);",
               "% = completeness (recovered fraction of the acute deficit by chronic); 'unresolved' = fit flag not ok"]
        txt += [f"{r.animal}: {half_lines(r)}" for _, r in PA.iterrows()]
        fig.text(0.01, 0.005, "\n".join(txt), fontsize=7.2, va="bottom", ha="left", family="monospace")
    fig.suptitle(f"Recovery pooled across animals by day bin ({window} window; cortex = hits only): open = sessions "
                 "(shape = animal, colour = epoch),\nfilled = mean of animals, band = 95% CI (animals > sessions). "
                 "n = 4, PROVISIONAL\n" + DATA_HIT, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.145, 1, 0.94))
    out = A / "figures" / f"recovery_timing_pooled_day_{window}_aim.png"
    save_png_svg(fig, out, dpi=120)
    plt.close(fig)
    print("->", out)


def aim_main(window: str, plot_only: bool) -> int:
    from figdata import load_fd, save_fd
    stem, root = f"recovery_timing_pooled_day_{window}_aim", A / "figdata"
    if not plot_only:
        F = aim_compute(window)
        pd.set_option("display.width", 250)
        print_per_animal(F["PA"], "== per animal (rule 8): aim (PRIMARY), hit rate, code on hits")
        print("  epoch means (beh, neu = / pre; aim = deg vs pre) + recovered fraction by epoch:\n"
              + F["PA"][["animal"] + [c for c in F["PA"] if c.split("_", 1)[-1] in EP4 or "_recov_" in c]]
              .round(2).to_string(index=False))
        F["D"].to_csv(A / f"recovery_timing_pooled_day_{window}_aim.csv", index=False)
        F["PA"].to_csv(A / f"recovery_timing_aim_per_animal_{window}.csv", index=False)
        ref = pd.read_csv(A / f"recovery_timing_pooled_day_{window}.csv")
        mine = F["D"][F["D"].measure != "aim"].reset_index(drop=True)
        print("== check vs recovery_timing_pooled_day (beh/neu): max |diff| =",
              float(np.nanmax(np.abs(mine[["mean", "lo", "hi"]].to_numpy() - ref[["mean", "lo", "hi"]].to_numpy()))))
        save_fd(stem, F, root)
    aim_draw(load_fd(stem, root))
    return 0


if __name__ == "__main__":
    import sys
    if "--aim" in sys.argv:          # Fig 0.12 + aim panel: python scripts/recovery_timing.py --aim [--plot-only] [--window lick]
        w = sys.argv[sys.argv.index("--window") + 1] if "--window" in sys.argv else "lick"
        raise SystemExit(aim_main(w, "--plot-only" in sys.argv))
    raise SystemExit(main())
