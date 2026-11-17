"""REV-UP FOLLOW-UPS (Priya 10-29): (A) reconcile the LATE first licks between the behavior and neural rev-up analyses;
(B) jaw "ramping" energy before the first lick. Reads revup_behaviour's / revup_neural's outputs; adds files, overwrites
none of theirs.

(A) RECONCILE LATE LICKS. revup_behaviour (canonical = reference-gate trials) found first licks after the spout left
    (after_left) at / below the spontaneous ITI chance rate; revup_neural (its "ENGAGED" trials, 48 late licks) found
    cued-size, cued-shaped lick responses for them. TRIAL SETS, measured here: revup_neural's engaged set is
    `combined_encoding.load_behaviour(...).trials.engaged` as built when lick_precue_restw ran, and it equals
    trial_attempts.csv `attempted` (OROFACIAL_GATE=attempted, the frozen-encoder env) on 105 / 107 sessions -- NOT the
    reference (sated-tail) gate = `canonical`. The attempted gate keeps only trials with a jaw / lick attempt INSIDE the
    window, so it conditions on a cue response. Gates reported side by side (tag in every output):
      canonical            reference gate (sated tail out) -- revup_behaviour's set, numbers recomputed here as a check
      neural               the exact revup_neural trial set (matched on label, spout, stop and next-strobe time)
      attempted            trial_attempts `attempted`
      canonical_attempted  both
    Per gate, per animal x spout x epoch: revup_behaviour._cell (sh_after_left, after_left_given_nolick, ...) with the
    chance rate RECOMPUTED on the gate's own ITIs (bout starts = a lick with none in the preceding 1 s, in stop + 1 s ..
    next strobe, summed over the gate's trials of the session; p = 1 - exp(-rate x (bound - stop)) per trial).
    Late-lick TIMING: share of after_left first licks within 1 s of the stop vs the constant-hazard (spontaneous)
    expectation mean(min(1, L) / L), L = bound - stop. PLATEAU chance (`_plateau`): the first-lick hazard of the same
    no-lick trials 1.5 s .. bound after the stop (spontaneous, spout gone) -> expected P(first lick within 1 s of the
    stop) = 1 - exp(-lambda), vs the observed share (early_obs / early_exp_plateau).
(B) JAW RAMPING ENERGY before the first lick, trials = canonical, first lick in the response window (phase 'window'),
    close_mid excluded (cam4 chin hidden). Latency groups as revup_neural: quartiles within animal x spout x epoch, and
    fixed bins <0.25 / 0.25-0.5 / 0.5-1.0 / >1.0 s.
      FR prep   FaceRhythm fr_prep_jaw z (revup_fr.npz 'prep_cue', session pre-cue z), cue-aligned; lick-aligned by
                resampling the cue-aligned row at latency + lag (valid from cue - 0.5 s).
      cam4 jaw  v36 cleaned jaw y (clean.npz jaw_y, px, + = opening; windows.npz frame DAQ times; gaps > 12 ms = NaN):
                displacement D = y - the trial's pre-cue (-0.5..0 s) median; motion energy ME = |dy/dt| (20-ms boxcar)
                / the session's median pre-cue ME (1 = the usual resting jitter).
    Per trial: wait_me / wait_disp / wait_active (ME > 3) = mean over cue + 0.1 .. lick - 0.15 s (needs latency >=
    0.35 s); pre_me (cue - 0.5 .. cue); L_slope_disp (D, lick - 0.5 .. lick, px/s); L_slope_me (lick - 0.5 .. - 0.05);
    L_level (D, lick - 0.1 .. lick); prep_area (z.s, cue .. lick), prep_mean, prep_wait, prep_L_slope.
    Session values = MEDIANS over trials (ME ratios are heavy-tailed). Contrasts per session x spout class: Q4 - Q1 (lick-aligned slopes / levels), gt1 - 0.5to1 (wait measures: both
    groups have >= 0.5 s of wait), Q4 wait_me - pre_me ("revving" > 0 vs quiet wait <= 0).
STATS per-animal tables first (rule 8); levels boot_ci; epoch vs pre and far_contra - others vs pre boot_delta
(analysis_kit, paired nested animals > sessions).

OUTPUTS (<OUT_W>/revup/): jaw_sessions/<label>.npz (per-session cache), revup_cells_engaged.csv (gate column),
  revup_late_engaged.csv (+ _session), revup_late_engaged_boot.csv, revup_jaw_trials.csv.gz, revup_jaw_session.csv,
  revup_jaw_per_animal.csv, revup_jaw_boot.csv, figdata/revup_jaw.pkl, figdata/revup_late_engaged.pkl;
  figures <OUT_W>/figures/revup_jaw_traces.png, revup_jaw_metrics.png, revup_late_engaged.png.
Usage: python scripts/revup_followup.py [--jobs 8]      |     python scripts/revup_followup.py --plot-only
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revup_behaviour as rb  # noqa: E402

REV, FIG = rb.REV, rb.FIG
JS = REV / "jaw_sessions"
FD = REV / "figdata"
VERSION = 2                        # 2: ITI licks filtered as revup_behaviour (finite onset AND peak)
EP4 = rb.EP4
GATES = ["canonical", "neural", "attempted", "canonical_attempted"]
DTJ = 0.01
LAGS_JC = np.round(np.arange(-0.5, 4.0 + 1e-9, DTJ), 3)
LAGS_JL = np.round(np.arange(-1.5, 0.3 + 1e-9, DTJ), 3)
LAGS_PL = np.round(np.arange(-1.5, 0.3 + 1e-9, 0.04), 3)
MAXGAP = 0.012
QS = ["Q1", "Q2", "Q3", "Q4"]
FB = ["<0.25", "0.25-0.5", "0.5-1.0", ">1.0"]
SEED = 0


# ------------------------------------------------------------------------------------------------ per session
def _samp(t, v, q):
    q = np.asarray(q, float)
    out = np.full(q.shape, np.nan)
    ok = np.isfinite(q)
    j = np.searchsorted(t, q[ok])
    good = (j > 0) & (j < len(t))
    j = np.clip(j, 1, len(t) - 1)
    t0, t1 = t[j - 1], t[j]
    w = np.where(t1 > t0, (q[ok] - t0) / np.where(t1 > t0, t1 - t0, 1), 0)
    r = v[j - 1] + (v[j] - v[j - 1]) * w
    r[~good | ((t1 - t0) > MAXGAP)] = np.nan
    out[ok] = r
    return out


def session_worker(item) -> dict:
    """item = (label, cue, first_lick, stop, bound) -> JS/<label>.npz (module-level for spawn)."""
    label, cue, fl, stop, bound = item
    f = JS / f"{label}.npz"
    if f.exists() and int(np.load(f)["version"]) == VERSION:
        return {"label": label, "cached": True}
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    sd = oo.session_dir(rv, oc.RUN, an, date)
    # ITI bout starts per trial (revup_behaviour's rule), so a gate's rate can be summed over its own trials
    A = pd.read_csv(sd / "licks_all.csv", usecols=["daq_on_s", "daq_peak_s"])
    A = A[np.isfinite(A.daq_on_s) & np.isfinite(A.daq_peak_s)]
    lk = np.sort(A.daq_on_s.to_numpy(float))
    starts = lk[(lk - np.r_[-np.inf, lk[:-1]]) >= rb.BOUT_GAP]
    lo, hi = stop + 1.0, np.maximum(bound, stop + 1.0)
    n_st = rb._count(starts, lo, hi).astype(float)
    dur = np.clip(bound - lo, 0, None)
    # cam4 cleaned jaw
    w = np.load(sd / "windows.npz")
    c = np.load(sd / "clean.npz")
    t = w["ix_daq_s"].astype(float)
    y = c["jaw_y"].astype(float)
    o = np.argsort(t, kind="stable")
    t, y = t[o], y[o]
    keep = np.r_[True, np.diff(t) > 1e-6]
    t, y = t[keep], y[keep]
    dt = np.diff(t)
    sp = np.abs(np.diff(y)) / np.where(dt > 0, dt, np.nan)
    sp[dt > MAXGAP] = np.nan
    sp = np.r_[sp, np.nan]
    ker = np.ones(5)
    num = np.convolve(np.nan_to_num(sp), ker, "same")
    den = np.convolve(np.isfinite(sp).astype(float), ker, "same")
    me = np.where(den >= 3, num / np.maximum(den, 1), np.nan)
    gc = cue[:, None] + LAGS_JC[None, :]
    gl = fl[:, None] + LAGS_JL[None, :]
    yc, mc = _samp(t, y, gc), _samp(t, me, gc)
    yl, ml = _samp(t, y, gl), _samp(t, me, gl)
    pre = LAGS_JC < 0
    with np.errstate(all="ignore"):
        rest = np.nanmedian(yc[:, pre], 1)
        pre_me = np.nanmean(mc[:, pre], 1)
    ref = float(np.nanmedian(pre_me)) if np.isfinite(pre_me).any() else np.nan
    JS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, version=VERSION, n_st=n_st, dur=dur, rest=rest, pre_me_raw=pre_me, me_ref=ref,
                        dc=(yc - rest[:, None]).astype(np.float32), mc=mc.astype(np.float32),
                        dl=(yl - rest[:, None]).astype(np.float32), ml=ml.astype(np.float32))
    return {"label": label, "cached": False}


# ------------------------------------------------------------------------------------------------ trial table + gates
def _trials() -> pd.DataFrame:
    T = pd.read_csv(REV / "revup_trials.csv.gz", low_memory=False)
    for c in ("canonical", "attempted"):
        T[c] = T[c].astype(str).str.lower().eq("true")
    N = pd.read_csv(REV / "revup_neural_trials.csv.gz", usecols=["label", "pos", "stop_lat", "next_strobe_lat",
                                                                 "phase"])
    N["k1"], N["k2"] = (N.stop_lat * 1000).round().astype(int), (N.next_strobe_lat * 1000).round().astype(int)
    T["k1"], T["k2"] = T.stop_ms.round().fillna(-1).astype(int), T.bound_ms.round().fillna(-1).astype(int)
    key = N[["label", "pos", "k1", "k2"]].drop_duplicates().assign(neural=True)
    T = T.merge(key, on=["label", "pos", "k1", "k2"], how="left")
    T["neural"] = T.neural.fillna(False).astype(bool)
    T["canonical_attempted"] = T.canonical & T.attempted
    T["cls2"] = np.where(T.pos == "far_contra", "far_contra", "other")
    return T


def _fill(T: pd.DataFrame) -> pd.DataFrame:
    """attach the per-session cache columns (row order = revup_trials per label)."""
    parts = []
    for x, g in T.groupby("label", sort=True):
        z = np.load(JS / f"{x}.npz")
        g = g.sort_values("trial_k").copy()
        g["n_st"], g["iti_dur"] = z["n_st"], z["dur"]
        g["me_ref"] = float(z["me_ref"])
        parts.append(g)
    return pd.concat(parts).sort_index()


# ------------------------------------------------------------------------------------------------ (A)
def _plateau(nl: pd.DataFrame) -> dict:
    """Chance from the SAME no-lick trials: the first-lick hazard after the spout left, 1.5 s .. bound (the plateau =
    spontaneous licking with the spout gone) -> expected P(first lick in stop .. stop + 1 s) = 1 - exp(-lambda);
    observed = the share of those trials whose first lick falls in stop .. stop + 1 s."""
    if not len(nl):
        return {"early_obs": np.nan, "lam_plateau_hz": np.nan, "early_exp_plateau": np.nan}
    L = ((nl.bound_ms - nl.stop_ms) / 1000).to_numpy(float)
    dt = np.where(nl.first_lick_phase == "after_left", (nl.first_lick_s - nl.spout_leave_s).to_numpy(float), np.inf)
    risk = (dt > 1.5) & (L > 1.5)
    ev = risk & (dt <= L)
    expo = np.clip(np.minimum(dt, L) - 1.5, 0, None)[risk].sum()
    lam = ev.sum() / expo if expo > 0 else np.nan
    return {"early_obs": float((dt <= 1.0).mean()), "lam_plateau_hz": lam,
            "early_exp_plateau": 1 - np.exp(-lam) if np.isfinite(lam) else np.nan}


def part_a(T: pd.DataFrame) -> None:
    from wfield_local import analysis_kit as ak
    POS = rb._pos_order()
    cells, late, sess = [], [], []
    for gate in GATES:
        G = T[T[gate]].copy()
        # chance rate from the gate's own ITIs (per session)
        r = G.groupby("label").apply(lambda g: g.n_st.sum() / g.iti_dur.sum() if g.iti_dur.sum() > 0 else np.nan)
        G["rate_gate"] = G.label.map(r)
        G["p_spont_alltrials"] = G.p_spont_after_left
        G["p_spont_after_left"] = 1 - np.exp(-G.rate_gate * np.clip(G.bound_ms - G.stop_ms, 0, None) / 1000)
        for (a, p, e), g in G.groupby(["animal", "pos", "epoch"]):
            c = rb._cell(g)
            nl = g.first_lick_phase.isin(["after_left", "none"])
            c.update({"gate": gate, "animal": a, "pos": p, "epoch": e, "n_nolick": int(nl.sum()),
                      "n_after_left": int((g.first_lick_phase == "after_left").sum()),
                      "exp_given_nolick": float(g.p_spont_after_left[nl].mean()) if nl.any() else np.nan,
                      "exp_given_nolick_alltrialrate": float(g.p_spont_alltrials[nl].mean()) if nl.any() else np.nan})
            cells.append(c)
        G["after_left"] = (G.first_lick_phase == "after_left").astype(float)
        G["nolick"] = G.first_lick_phase.isin(["after_left", "none"])
        L = (G.bound_ms - G.stop_ms) / 1000
        G["dt_stop"] = (G.first_lick_s - G.spout_leave_s)
        G["exp_w1"] = np.minimum(1, L) / L
        for cls in ("all", "far_contra", "other"):
            H = G if cls == "all" else G[G.cls2 == cls]
            for keys, g in H.groupby(["animal", "epoch"]):
                nl = g[g.nolick]
                al = g[g.first_lick_phase == "after_left"]
                late.append({"gate": gate, "cls": cls, "animal": keys[0], "epoch": keys[1], "n_trials": len(g),
                             "n_nolick": len(nl), "n_after_left": len(al), "sh_after_left": len(al) / len(g),
                             "obs_given_nolick": nl.after_left.mean() if len(nl) else np.nan,
                             "exp_given_nolick": nl.p_spont_after_left.mean() if len(nl) else np.nan,
                             "exp_given_nolick_alltrialrate": nl.p_spont_alltrials.mean() if len(nl) else np.nan,
                             "late_within1s": (al.dt_stop <= 1.0).mean() if len(al) else np.nan,
                             "late_within1s_exp": al.exp_w1.mean() if len(al) else np.nan,
                             "late_dt_med_s": al.dt_stop.median() if len(al) else np.nan,
                             "rate_gate_hz": g.drop_duplicates("label").rate_gate.mean(), **_plateau(nl)})
            for keys, g in H.groupby(["label", "animal", "epoch"]):
                nl = g[g.nolick]
                if len(nl):
                    sess.append({"gate": gate, "cls": cls, "label": keys[0], "animal": keys[1], "epoch": keys[2],
                                 "n_nolick": len(nl), "obs": nl.after_left.mean(), "exp": nl.p_spont_after_left.mean(),
                                 "excess": nl.after_left.mean() - nl.p_spont_after_left.mean(),
                                 "early_obs": float(((nl.first_lick_phase == "after_left")
                                                     & (nl.dt_stop <= 1.0)).mean())})
    C = pd.DataFrame(cells)
    C["pos"] = pd.Categorical(C.pos, POS, ordered=True)
    C["epoch"] = pd.Categorical(C.epoch, EP4, ordered=True)
    lead = ["gate", "animal", "pos", "epoch"]
    C = C[lead + [c for c in C if c not in lead]].sort_values(lead)
    C.to_csv(REV / "revup_cells_engaged.csv", index=False, float_format="%.4g")
    LT = pd.DataFrame(late)
    LT["epoch"] = pd.Categorical(LT.epoch, EP4, ordered=True)
    LT = LT.sort_values(["gate", "cls", "animal", "epoch"])
    LT.to_csv(REV / "revup_late_engaged.csv", index=False, float_format="%.4g")
    S = pd.DataFrame(sess)
    # early excess per session vs the animal x epoch PLATEAU chance (the session alone is too sparse for a hazard)
    pl = LT.set_index(["gate", "cls", "animal", "epoch"]).early_exp_plateau
    S["early_excess"] = S.early_obs - [pl.get((r.gate, r.cls, r.animal, r.epoch), np.nan) for r in S.itertuples()]
    S.to_csv(REV / "revup_late_engaged_session.csv", index=False, float_format="%.4g")
    # boot: pooled (session-weighted) level of obs - exp given no lick; trial-weighted is in the per-animal table
    out = []
    for (gate, cls, e), g in S.groupby(["gate", "cls", "epoch"]):
        for meas in ("excess", "early_excess"):
            by = {a: h[meas].dropna().tolist() for a, h in g.groupby("animal")}
            by = {a: v for a, v in by.items() if v}
            iv = ak.boot_ci(by, np.random.default_rng(SEED)) if by else None
            out.append({"gate": gate, "cls": cls, "epoch": e, "measure": meas, "est": iv.point if iv else np.nan,
                        "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan, "n_sessions": len(g),
                        "n_animals": len(by), **{f"m_{a}": float(np.mean(v)) for a, v in sorted(by.items())}})
    Bt = pd.DataFrame(out)
    Bt.to_csv(REV / "revup_late_engaged_boot.csv", index=False, float_format="%.4g")
    FD.mkdir(parents=True, exist_ok=True)
    (FD / "revup_late_engaged.pkl").write_bytes(pickle.dumps({"late": LT, "boot": Bt}))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    cols = ["animal", "epoch", "n_trials", "n_nolick", "n_after_left", "obs_given_nolick", "exp_given_nolick",
            "late_within1s", "late_within1s_exp", "early_obs", "early_exp_plateau"]
    for gate in GATES:
        print(f"\n== (A) gate={gate}, all spouts, per animal x epoch")
        print(LT[(LT.gate == gate) & (LT.cls == "all")][cols].round(3).to_string(index=False))
    print("\n== (A) boot of session excess (obs - chance | no lick in window)")
    print(Bt.round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------------ (B)
def _slope(X, lags, lo, hi):
    m = (lags >= lo) & (lags <= hi)
    x = lags[m] - lags[m].mean()
    Y = X[:, m]
    ok = np.isfinite(Y)
    n = ok.sum(1)
    Yc = np.where(ok, Y, 0)
    xm = (ok * x).sum(1) / np.maximum(n, 1)
    ym = Yc.sum(1) / np.maximum(n, 1)
    xx = (ok * (x - xm[:, None]) ** 2).sum(1)
    xy = (ok * (x - xm[:, None]) * (Yc - ym[:, None])).sum(1)
    return np.where((n >= 0.6 * m.sum()) & (xx > 0), xy / np.where(xx > 0, xx, 1), np.nan)


def _wmean(X, lags, lo, hi, min_n=3):
    """mean of X over per-row windows [lo_i, hi_i] (arrays) on lags."""
    m = (lags[None, :] >= lo[:, None]) & (lags[None, :] <= hi[:, None]) & np.isfinite(X)
    n = m.sum(1)
    return np.where(n >= min_n, np.where(m, X, 0).sum(1) / np.maximum(n, 1), np.nan)


def part_b(T: pd.DataFrame) -> None:
    from wfield_local import analysis_kit as ak
    Z = np.load(REV / "revup_fr.npz")
    if not (np.array_equal(Z["label"].astype(str), T.sort_index().label.to_numpy(str))
            and np.array_equal(Z["trial_id"], T.sort_index().trial_id.to_numpy(int))):
        raise ValueError("revup_fr.npz rows differ from revup_trials")
    lc = Z["lags_cue"].astype(float)
    PC = Z["prep_cue"].astype(np.float32)
    keep = (T.canonical & (T.first_lick_phase == "window") & np.isfinite(T.lat_ms) & (T.pos != "close_mid")).to_numpy()
    lat = T.lat_ms.to_numpy(float) / 1000
    # jaw arrays from the cache, row-aligned to T
    nC, nL = len(LAGS_JC), len(LAGS_JL)
    DC, MC = np.full((len(T), nC), np.nan, np.float32), np.full((len(T), nC), np.nan, np.float32)
    DL, ML = np.full((len(T), nL), np.nan, np.float32), np.full((len(T), nL), np.nan, np.float32)
    pre_me = np.full(len(T), np.nan)
    for x, g in T.groupby("label", sort=True):
        z = np.load(JS / f"{x}.npz")
        idx = g.sort_values("trial_k").index.to_numpy()
        ref = float(z["me_ref"]) or np.nan
        DC[idx], DL[idx] = z["dc"], z["dl"]
        MC[idx], ML[idx] = z["mc"] / ref, z["ml"] / ref
        pre_me[idx] = z["pre_me_raw"] / ref
    E = T.loc[keep].copy()
    i = np.flatnonzero(keep)
    la = lat[i]
    # FR prep lick-aligned (resample the cue-aligned row)
    q = la[:, None] + LAGS_PL[None, :]
    PL = np.full(q.shape, np.nan, np.float32)
    for r in range(len(i)):
        PL[r] = np.interp(q[r], lc, PC[i[r]].astype(float), left=np.nan, right=np.nan)
    PCi = PC[i].astype(float)
    E["prep_area"] = [np.trapz(np.nan_to_num(PCi[r][(lc >= 0) & (lc <= la[r])]), dx=0.04) for r in range(len(i))]
    E["prep_mean"] = _wmean(PCi, lc, np.zeros(len(i)), la, 1)
    E["prep_wait"] = _wmean(PCi, lc, np.full(len(i), 0.1), la - 0.15, 2)
    E["prep_L_slope"] = _slope(PL, LAGS_PL, -0.5, 0)
    E["prep_L_level"] = _wmean(PL, LAGS_PL, np.full(len(i), -0.1), np.zeros(len(i)), 1)
    w_lo, w_hi = np.full(len(i), 0.1), la - 0.15
    MCi, DCi = MC[i].astype(float), DC[i].astype(float)
    E["jaw_wait_me"] = _wmean(MCi, LAGS_JC, w_lo, w_hi, 10)
    E["jaw_wait_disp"] = _wmean(np.abs(DCi), LAGS_JC, w_lo, w_hi, 10)
    E["jaw_wait_active"] = _wmean((MCi > 3).astype(float) + 0 * MCi, LAGS_JC, w_lo, w_hi, 10)
    E["jaw_pre_me"] = pre_me[i]
    E["jaw_rev"] = E.jaw_wait_me - E.jaw_pre_me
    E["jaw_L_slope_disp"] = _slope(DL[i].astype(float), LAGS_JL, -0.5, 0)
    E["jaw_L_slope_me"] = _slope(ML[i].astype(float), LAGS_JL, -0.5, -0.05)
    E["jaw_L_level"] = _wmean(DL[i].astype(float), LAGS_JL, np.full(len(i), -0.1), np.zeros(len(i)), 3)
    E["q"] = E.groupby(["animal", "pos", "epoch"]).lat_ms.transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=QS) if len(s) >= 8 else pd.Series(np.nan, s.index))
    E["fb"] = pd.cut(E.lat_ms / 1000, [0, 0.25, 0.5, 1.0, np.inf], labels=FB, right=False).astype(str)
    MET = ["lat_ms", "prep_area", "prep_mean", "prep_wait", "prep_L_slope", "prep_L_level", "jaw_wait_me",
           "jaw_wait_disp", "jaw_wait_active", "jaw_pre_me", "jaw_rev", "jaw_L_slope_disp", "jaw_L_slope_me",
           "jaw_L_level"]
    E[["label", "animal", "epoch", "trial_id", "pos", "cls2", "q", "fb", *MET]].to_csv(
        REV / "revup_jaw_trials.csv.gz", index=False, float_format="%.4g")
    # session x class x group means (>= 3 trials)
    rows = []
    for grp_col in ("q", "fb"):
        for (x, a, e, c, gq), g in E.groupby(["label", "animal", "epoch", "cls2", grp_col], observed=True):
            if len(g) >= 3:
                rows.append({"label": x, "animal": a, "epoch": e, "cls": c, "gtype": grp_col, "grp": str(gq),
                             "n": len(g), **{m: g[m].median() for m in MET}})
    S = pd.DataFrame(rows)
    S.to_csv(REV / "revup_jaw_session.csv", index=False, float_format="%.4g")
    # per-animal table (rule 8): pooled trials per animal x epoch x class x quartile / fixed bin
    PA = []
    for grp_col in ("q", "fb"):
        for (a, e, c, gq), g in E.groupby(["animal", "epoch", "cls2", grp_col], observed=True):
            PA.append({"animal": a, "epoch": e, "cls": c, "gtype": grp_col, "grp": str(gq), "n": len(g),
                       **{m: g[m].median() for m in MET}})
    PA = pd.DataFrame(PA)
    PA["epoch"] = pd.Categorical(PA.epoch, EP4, ordered=True)
    PA = PA.sort_values(["gtype", "cls", "animal", "epoch", "grp"])
    PA.to_csv(REV / "revup_jaw_per_animal.csv", index=False, float_format="%.4g")
    # contrasts per session x class
    CON = {"Q4-Q1": ("q", "Q4", "Q1", ["prep_L_slope", "prep_L_level", "jaw_L_slope_disp", "jaw_L_slope_me",
                                         "jaw_L_level", "prep_area", "prep_mean"]),
           "gt1-0.5to1": ("fb", ">1.0", "0.5-1.0", ["jaw_wait_me", "jaw_wait_disp", "jaw_wait_active", "prep_wait",
                                                    "jaw_rev", "jaw_L_slope_disp", "prep_L_slope"]),
           "Q4": ("q", "Q4", None, ["jaw_rev", "jaw_wait_me", "jaw_wait_active", "prep_wait"])}
    cr = []
    for name, (gt, hi, lo, ms) in CON.items():
        Sg = S[S.gtype == gt].set_index(["label", "animal", "epoch", "cls", "grp"])
        for m in ms:
            v = Sg[m].unstack("grp")
            if hi not in v:
                continue
            d = v[hi] - (v[lo] if lo else 0)
            for k, val in d.dropna().items():
                cr.append({"contrast": name, "measure": m, "label": k[0], "animal": k[1], "epoch": k[2], "cls": k[3],
                           "value": val})
    CR = pd.DataFrame(cr)
    out = []
    for (name, m), g in CR.groupby(["contrast", "measure"]):
        W = g.pivot_table(index=["label", "animal", "epoch"], columns="cls", values="value").reset_index()
        if "far_contra" in W and "other" in W:
            W["fc-oth"] = W.far_contra - W.other
        for kind in [k for k in ("far_contra", "other", "fc-oth") if k in W]:
            for e in EP4:
                by = {a: h[kind].dropna().tolist() for a, h in W[W.epoch == e].groupby("animal")}
                by = {a: v for a, v in by.items() if v}
                iv = ak.boot_ci(by, np.random.default_rng(SEED)) if by else None
                row = {"contrast": name, "measure": m, "kind": kind, "epoch": e, "test": "level",
                       "est": iv.point if iv else np.nan, "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan,
                       "n_animals": len(by), **{f"a_{a}": float(np.mean(v)) for a, v in sorted(by.items())}}
                out.append(row)
                if e != "pre":
                    pre = {a: h[kind].dropna().tolist() for a, h in W[W.epoch == "pre"].groupby("animal")}
                    pre = {a: v for a, v in pre.items() if v}
                    iv, dr = ak.boot_delta(by, pre, np.random.default_rng(SEED), draws=True)
                    per = {f"a_{a}": float(np.mean(by[a]) - np.mean(pre[a])) for a in sorted(set(by) & set(pre))}
                    out.append({"contrast": name, "measure": m, "kind": kind, "epoch": e, "test": "vs_pre",
                                "est": iv.point if iv else np.nan, "lo": iv.lo if iv else np.nan,
                                "hi": iv.hi if iv else np.nan, "p": ak.boot_p(dr) if iv else np.nan,
                                "n_animals": iv.n_animals if iv else 0, **per})
    BT = pd.DataFrame(out)
    BT.to_csv(REV / "revup_jaw_boot.csv", index=False, float_format="%.4g")
    # figure data: session-mean traces per class x epoch x quartile, averaged within animal then across animals
    tr = {}
    E["_r"] = np.arange(len(E))
    for nm, X, lags in (("jaw_me_L", ML[i], LAGS_JL), ("jaw_disp_L", DL[i], LAGS_JL), ("prep_L", PL, LAGS_PL),
                        ("jaw_me_C", MC[i], LAGS_JC), ("jaw_disp_C", DC[i], LAGS_JC), ("prep_C", PCi, lc)):
        X = np.asarray(X, float)
        for (c, e, gq), g in E.groupby(["cls2", "epoch", "q"], observed=True):
            am = []
            for _a, h in g.groupby("animal"):
                with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                    __import__("warnings").simplefilter("ignore")
                    sm = [np.nanmean(X[hh._r.to_numpy()], 0) for _, hh in h.groupby("label")]
                    am.append(np.nanmean(sm, 0))
            tr[(nm, c, e, str(gq))] = (lags, np.asarray(am), len(g), float(g.lat_ms.median()))
    FD.mkdir(parents=True, exist_ok=True)
    (FD / "revup_jaw.pkl").write_bytes(pickle.dumps({"traces": tr, "boot": BT, "per_animal": PA}))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    sh = ["animal", "epoch", "grp", "n", "lat_ms", "jaw_pre_me", "jaw_wait_me", "jaw_rev", "jaw_wait_active",
          "jaw_L_slope_disp", "jaw_L_level", "prep_wait", "prep_L_slope", "prep_area"]
    for c in ("far_contra", "other"):
        print(f"\n== (B) per animal, {c}, fixed latency bins (medians)")
        print(PA[(PA.gtype == "fb") & (PA.cls == c)][sh].round(3).to_string(index=False))
        print(f"\n== (B) per animal, {c}, quartiles (medians)")
        print(PA[(PA.gtype == "q") & (PA.cls == c)][sh].round(3).to_string(index=False))
    print("\n== (B) boot")
    print(BT[["contrast", "measure", "kind", "epoch", "test", "est", "lo", "hi", "n_animals"] +
             [c for c in BT if c.startswith("a_")]].round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------ figures
def plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    d = pickle.loads((FD / "revup_jaw.pkl").read_bytes())
    tr, BT = d["traces"], d["boot"]
    qcol = {"Q1": "#c6dbef", "Q2": "#6baed6", "Q3": "#2171b5", "Q4": "#08306b"}
    for align, tag, xl in (("L", "lick", "time from first lick (s)"), ("C", "cue", "time from cue (s)")):
        meas = [(f"jaw_me_{align}", "jaw ME / rest ME"), (f"jaw_disp_{align}", "jaw displacement (px, + open)"),
                (f"prep_{align}", "FR prep-jaw (z)")]
        fig, axs = plt.subplots(len(meas) * 2, 4, figsize=(13, 2.1 * len(meas) * 2), sharex=True, sharey="row")
        for ci, c in enumerate(("far_contra", "other")):
            for mi, (nm, yl) in enumerate(meas):
                r = ci * len(meas) + mi
                for j, e in enumerate(EP4):
                    ax = axs[r, j]
                    for qq in QS:
                        v = tr.get((nm, c, e, qq))
                        if v is None or not len(v[1]):
                            continue
                        lags, am, n, lm = v
                        mu = np.nanmean(am, 0)
                        ax.plot(lags, mu, color=qcol[qq], lw=1.5, label=f"{qq} ({lm / 1000:.2f} s)")
                        if align == "C":
                            ax.axvline(lm / 1000, color=qcol[qq], lw=0.6, ls=":")
                    ax.axvline(0, color="k", lw=0.6)
                    if r == 0:
                        ax.set_title(e)
                    if j == 0:
                        ax.set_ylabel(f"{c}\n{yl}", fontsize=8)
                    # ONE legend per (spout class, epoch) -- on the top row of each block, in the corner the traces
                    # leave empty (lick-aligned: flat before the lick -> upper left; cue-aligned: decayed by 2-3 s ->
                    # upper right). Was also drawn on every chronic panel, over the prep-jaw traces.
                    if r % len(meas) == 0:
                        ax.legend(fontsize=7, frameon=False, loc="upper left" if align == "L" else "upper right")
                    if align == "L":
                        ax.set_xlim(-1.5, 0.3)
                    else:
                        ax.set_xlim(-0.5, 3.0)
        for ax in axs[-1]:
            ax.set_xlabel(xl)
        top = rb._sup(fig, f"Fig J{1 if align == 'L' else 2}. Jaw activity before the first lick, {tag}-aligned, by "
                           "latency quartile (within animal x spout x epoch; legend = median latency; dotted = it). "
                           "Canonical trials, first lick in the window, close_mid out. Mean over animals of session "
                           "means. cam4 cleaned jaw: ME = |dy/dt| / session rest ME; FaceRhythm fr_prep_jaw z. "
                           "BEHAVIOUR", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, top))
        save_png_svg(fig, FIG / f"revup_jaw_traces_{tag}.png", dpi=110)
        plt.close(fig)
    # metrics: per-animal dots + pooled CI per epoch, far_contra vs others
    show = [("gt1-0.5to1", "jaw_wait_me"), ("gt1-0.5to1", "jaw_wait_active"), ("Q4", "jaw_rev"),
            ("Q4-Q1", "jaw_L_slope_disp"), ("Q4-Q1", "jaw_L_level"), ("gt1-0.5to1", "prep_wait"),
            ("Q4-Q1", "prep_L_slope"), ("Q4-Q1", "prep_area")]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    EC = rb._ep_col()
    for k, (cn, m) in enumerate(show):
        ax = axs.flat[k]
        for ci, c in enumerate(("other", "far_contra")):
            for j, e in enumerate(EP4):
                g = BT[(BT.contrast == cn) & (BT.measure == m) & (BT.kind == c) & (BT.epoch == e) &
                       (BT.test == "level")]
                if not len(g):
                    continue
                g = g.iloc[0]
                x = ci * 5 + j
                ax.errorbar(x, g.est, yerr=[[g.est - g.lo], [g.hi - g.est]], fmt="o", color=EC[e], ms=6, capsize=2)
                av = [g[a] for a in g.index if a.startswith("a_") and np.isfinite(g[a])]
                ax.scatter(np.full(len(av), x + 0.25), av, s=10, color="0.5", zorder=3)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks([1.5, 6.5])
        ax.set_xticklabels(["other five", "far_contra"])
        ax.set_title(f"{m}\n[{cn}]", fontsize=9)
    bot = rb._leg_below(fig, [plt.Line2D([], [], marker="o", ls="", color=EC[e], label=f"{e} (pooled, boot CI)")
                              for e in EP4] + [plt.Line2D([], [], marker="o", ls="", ms=4, color="0.5",
                                                          label="per-animal mean")], fontsize=9)
    top = rb._sup(fig, "Fig J3. Jaw 'revving' metrics, session contrasts: pooled level (nested boot CI, colors = epochs) "
                       "+ per-animal means (gray). gt1-0.5to1 = first lick >1 s minus 0.5-1 s (both wait >= 0.5 s); "
                       "Q4 jaw_rev = slowest-quartile wait ME minus that trial's pre-cue ME (>0 = revving, <=0 = quiet "
                       "wait); Q4-Q1 = slowest minus fastest quartile, lick-aligned. BEHAVIOR", fontsize=9)
    fig.tight_layout(rect=(0, bot, 1, top))
    save_png_svg(fig, FIG / "revup_jaw_metrics.png", dpi=110)
    plt.close(fig)
    # (A) late licks vs chance, per gate
    L = pickle.loads((FD / "revup_late_engaged.pkl").read_bytes())["late"]
    L = L[L.cls == "all"]
    AN = sorted(L.animal.unique())
    rows = (("obs_given_nolick", "exp_given_nolick", "P(first lick after spout left\n| no lick in window)",
             "black line = chance", 1.0),
            ("early_obs", "early_exp_plateau", "P(first lick within 1 s of spout\nleaving | no lick in window)",
             "plateau chance", 0.6))
    # full figure = all four trial gates (sensitivity); _simple = the canonical gate only (the reporting set)
    for gates, stem in ((GATES, "revup_late_engaged"), (["canonical"], "revup_late_engaged_simple")):
        fig, axs = plt.subplots(2, len(gates), figsize=(max(4.6 * len(gates), 7.5), 8.0), sharey="row",
                                squeeze=False)
        for r, (ob, ex, yl, _chl, ymax) in enumerate(rows):
            for k, gate in enumerate(gates):
                ax = axs[r, k]
                for j, a in enumerate(AN):
                    for jj, e in enumerate(EP4):
                        g = L[(L.gate == gate) & (L.animal == a) & (L.epoch == e)]
                        if not len(g):
                            continue
                        g = g.iloc[0]
                        x = j * 5 + jj
                        ax.bar(x, g[ob], color=EC[e], width=0.8)
                        ax.plot([x - 0.4, x + 0.4], [g[ex]] * 2, color="k", lw=1.4)
                        ax.text(x, -0.02 * ymax, f"{int(g.n_nolick)}", ha="center", va="top",
                                fontsize=6 if len(gates) > 1 else 8)
                ax.set_xticks([j * 5 + 1.5 for j in range(len(AN))])
                ax.set_xticklabels(AN)
                ax.tick_params(axis="x", pad=12)
                ax.set_title(f"gate: {gate}", fontsize=10)
                ax.set_ylim(-0.1 * ymax, ymax)
            axs[r, 0].set_ylabel(yl, fontsize=9)
        bot = rb._leg_below(fig, [plt.Rectangle((0, 0), 1, 1, color=EC[e], label=e) for e in EP4] +
                            [plt.Line2D([], [], color="k", lw=1.4, label="top: ITI-rate chance"),
                             plt.Line2D([], [], color="k", lw=1.4, label="bottom: plateau chance (same trials, 1.5 s+)")],
                            ncol=6 if len(gates) > 1 else 3, fontsize=9)
        top = rb._sup(fig, "Fig A1" + (" (simple, canonical gate)" if len(gates) == 1 else "") + ". Late first licks "
                      "(after the spout left) on trials with no lick in the window, per animal x epoch, all spouts; numbers "
                      "under bars = such trials. Top: any time before the next strobe vs the ITI bout-start chance (ITIs "
                      "mostly follow licked trials -> high rate). Bottom: within 1 s of the spout leaving vs the plateau "
                      "hazard of the SAME trials 1.5 s on (spontaneous licking with the spout gone)."
                      + (" 'neural' = revup_neural's trial set (= the attempted gate)." if len(gates) > 1 else "")
                      + " BEHAVIOR", fontsize=9)
        fig.tight_layout(rect=(0, bot, 1, top), h_pad=1.5)
        save_png_svg(fig, FIG / f"{stem}.png", dpi=110)
        plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.plot_only:
        from wfield_local import analysis_kit as ak
        T = _trials()
        items = []
        for x, g in T.groupby("label", sort=True):
            g = g.sort_values("trial_k")
            items.append((x, g.cue_s.to_numpy(float), g.first_lick_s.to_numpy(float), g.spout_leave_s.to_numpy(float),
                          g.bound_s.to_numpy(float)))
        res, fails = ak.fan_sessions(items, session_worker, jobs=min(a.jobs, 8), key=lambda it: it[0])
        if fails:
            for s, e in fails:
                print(f"{s[0] if isinstance(s, tuple) else s}: FAILED {e}")
            raise SystemExit("session failures -- not writing tables")
        T = _fill(T)
        print("gate sizes:", {g: int(T[g].sum()) for g in GATES},
              "neural matched:", int(T.neural.sum()), "of", len(pd.read_csv(REV / "revup_neural_trials.csv.gz",
                                                                         usecols=["label"])))
        part_a(T)
        part_b(T)
    plots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
