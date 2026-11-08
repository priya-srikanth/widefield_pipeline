"""WITHIN-SESSION DECLINE OF AIM AND LATENCY vs QUITTING (Priya 11-07, cohort1 PS92-95 only).

Observation (state-space branch, `state_space_within_session_behavior.py`): on engaged first-lick trials the within-
session slope of first-lick LATENCY is larger after the stroke than before in 4 / 4 animals, and FAR-CONTRA first-lick
lateral landing drifts IPSILESIONALLY within subacute sessions in 4 / 4. Priya: *"this is important; i wonder how this
relates to the animals 'quitting' after stroke. can we look into this"*.

Questions (docs/WITHIN_SESSION_DECLINE_QUITTING.md):
  Q1 does the decline survive the repo's per-trial definitions (camera working period, 3-D first lick, per HOUR as well
     as per normalized session), and which measures decline (aim, latency, contact, vigor = speed / reach / camera
     licks per trial)?
  Q2 PRODROME: does aim / latency degrade in the minutes BEFORE the camera quit beyond the session's clock trend, and
     more after the stroke (quit window minus time-matched windows of the same animal's other sessions)?
  Q3 TIME vs EFFORT vs SATIETY: within animal x epoch, with session fixed effects, the measure on elapsed time,
     cumulative camera licks (effort), cumulative collected water (satiety) and cumulative far-contra trials.
  Q4 does a session's drift predict how early it quits (within animal, within-animal shuffle)?
  Q5 spout specificity (canonical spout order) and the link to the far-contra aim recovery axis.

DEFINITIONS (all existing, rule 9): sessions = `analysis_kit.curated_sessions` with the AUGUST-ONLY pre
(WIDEFIELD_PRE_FROM=0801); trials + DAQ cue times + camera classes = `trial_classes.session_trials` /
`daq_cue_seconds` / `camera_period_trials` (camera working period, A20 of 11-06); quit = `trial_classes.camera_quit`
(first cue of the tail after the last camera lick, >= 6 trials, >= 300 s; else censored); 3-D licks = pose3d
kinematics `licks.csv.gz` (lick_idx 0 = first post-cue lick; side_ml mm, - = ipsilesional/left; t_peak_cam4 ms from
the cue); collected water = the disengagement-tests session cache (`disengagement/sessions/beh_<label>.pkl`, DAQ
reward_ttl deliveries on DAQ-responded trials x reward_ul in force; cache version 2 = pre-11-05 DAQ lick detector,
which moves 0.6 % of licks); aim offset = side_ml minus the animal's AUGUST pre mean side_ml at that spout (first
licks, working period). Lesions are LEFT, so L = ipsi.

STAGES:  python scripts/within_session_decline.py extract [--jobs 6]   -> OUT/trials.csv.gz (per trial)
         python scripts/within_session_decline.py analyze              -> OUT/tables/*.csv (+ printed per-animal tables)
         python scripts/within_session_decline.py plot                 -> OUT/figures/*.png (+ svg)
OUT = M:/MICROSCOPE/Priya/Widefield/within_session_decline/. Exploratory, n = 4 animals.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("WIDEFIELD_PRE_FROM", "0801")          # AUGUST-ONLY pre baseline (primary); before config import
os.environ.setdefault("WIDEFIELD_COHORT", "cohort1")         # never cohort2

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("M:/MICROSCOPE/Priya/Widefield/within_session_decline")
TAB, FIG = OUT / "tables", OUT / "figures"
LICKS = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d/kinematics/licks.csv.gz")
BEH = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
           "disengagement/sessions")
AIM = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/"
           "imaging_encoding/iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-"
           "endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/aim_deficit_sessions.csv")
SS_SLOPES = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery/pertrial_lick/within_session_behavior_slopes.csv")
#: licks.csv.gz / trial tables use image-side names; all four lesions are LEFT, so L = ipsi, R = contra
SIDE = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
        "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}
EPS = ("pre", "acute", "subacute", "chronic")
POST = ("acute", "subacute", "chronic")
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
SEED = 20261107
CUE_MATCH_S = 0.05


# =============================================================================================== extract
def session_worker(item) -> pd.DataFrame:
    """One session -> its per-trial table (module-level for spawn). ``item`` = (label, its 3-D lick rows)."""
    lab, K = item
    from wfield_local import config, epochs
    from wfield_local import trial_classes as tc

    cs = np.asarray(tc.daq_cue_seconds(lab), float)
    T = tc.session_trials(lab)
    if T is None or not len(T) or not cs.size:
        raise RuntimeError(f"{lab}: no trial table / cue train")
    T = T.dropna(subset=["cue_daq"]).copy()
    work, _contact, has = tc.camera_period_trials(lab, T.cue_daq.to_numpy(float))
    q = tc.camera_quit(lab, cs)
    k = np.searchsorted(cs, T.cue_daq.to_numpy(float) - CUE_MATCH_S)
    k = np.clip(k, 0, cs.size - 1)
    ok = np.abs(cs[k] - T.cue_daq.to_numpy(float)) <= CUE_MATCH_S
    T = T.assign(k=np.where(ok, k, -1), working=work.astype(bool))
    T = T[T.k >= 0].copy()
    T["pos"] = T.position.map(SIDE).fillna(T.position)
    T["t_min"] = (T.cue_daq - cs[0]) / 60.0

    # 3-D licks: first lick (lick_idx 0) + per-trial aggregates over all post-cue licks of the trial
    first = K[K.lick_idx == 0].set_index("trial_id")
    agg = K.groupby("trial_id").agg(n3d=("lick_idx", "size"), speed_all=("speed_peak", "mean"),
                                    reach_all=("reach_mm", "mean"), contact_all=("contact", "mean"))
    tid = T.trial_id.astype(int)
    for c, src in (("side_ml", "side_ml"), ("contact1", "contact"), ("lat_cam", "t_peak_cam4"),
                   ("reach1", "reach_mm"), ("speed1", "speed_peak")):
        T[c] = tid.map(first[src]).astype(float).to_numpy() if len(first) else np.nan
    for c in agg.columns:
        T[c] = tid.map(agg[c]).astype(float).to_numpy() if len(agg) else np.nan

    # DAQ response / water from the disengagement-tests cache (one definition; cue index = its `order`)
    beh = BEH / f"beh_{lab}.pkl"
    T["coll_ul"] = np.nan
    if beh.exists():
        with open(beh, "rb") as fh:
            B = pickle.load(fh)
        bt = B["trials"].set_index("order")
        for c, src in (("hit_daq", "hit"), ("lat_daq", "lat"), ("n_daq", "n_post"), ("ili", "ili"),
                       ("coll_ul", "coll"), ("deliv_ul", "deliv")):
            T[c] = T.k.map(bt[src]).astype(float).to_numpy()
        T["beh_t0_check"] = float(B["t0"]) - float(cs[0])
    T = T.sort_values("k").reset_index(drop=True)
    # cumulative quantities accumulated BEFORE each trial, over every trial of the session (not only working ones)
    T["cum_water_ul"] = T.coll_ul.fillna(0).cumsum() - T.coll_ul.fillna(0)
    T["cum_licks_cam"] = T.n_licks.fillna(0).cumsum() - T.n_licks.fillna(0)
    fc = (T.pos == "far_contra").astype(float)
    T["cum_fc_trials"] = fc.cumsum() - fc
    T["label"], T["animal"], T["epoch"] = lab, config.animal_of(lab), epochs.epoch_of(lab)
    T["day"] = float(epochs.days_since_stroke(lab))
    T["has_classes"], T["censored"], T["n_tail"] = bool(has), bool(q["censored"]), int(q["n_tail"])
    T["quit_k"] = int(q["quit_cue"])
    T["quit_min"] = float((cs[q["quit_cue"]] - cs[0]) / 60.0) if not q["censored"] else np.nan
    T["end_min"] = float((cs[-1] - cs[0]) / 60.0)
    keep = ["label", "animal", "epoch", "day", "k", "trial_id", "pos", "t_min", "cls", "working", "n_licks",
            "n_contacts", "side_ml", "contact1", "lat_cam", "reach1", "speed1", "n3d", "speed_all", "reach_all",
            "contact_all", "hit_daq", "lat_daq", "n_daq", "ili", "coll_ul", "deliv_ul", "cum_water_ul",
            "cum_licks_cam", "cum_fc_trials", "has_classes", "censored", "n_tail", "quit_k", "quit_min", "end_min",
            "beh_t0_check"]
    return T[[c for c in keep if c in T.columns]]


def extract(jobs: int | None):
    from wfield_local import analysis_kit as ak
    labs = [lab for lab in ak.curated_labels(order="sorted") if lab[:4] in ANIMALS]
    K = pd.read_csv(LICKS, usecols=["session", "trial_id", "lick_idx", "contact", "t_peak_cam4", "reach_mm",
                                    "side_ml", "speed_peak"])
    K["lab"] = K.session.str[:4] + "_" + K.session.str[-4:]
    G = {lab: g.drop(columns=["session", "lab"]) for lab, g in K.groupby("lab")}
    items = [(lab, G[lab]) for lab in labs if lab in G]
    miss = [lab for lab in labs if lab not in G]
    print(f"{len(labs)} curated sessions; {len(items)} with 3-D licks; without: {miss}")
    res, fail = ak.fan_sessions(items, session_worker, jobs=jobs, key=lambda it: it[0])
    for it, err in fail:
        print("FAILED", it[0] if isinstance(it, tuple) else it, err)
    if fail:
        raise SystemExit("extraction failures: refusing to write a partial table")
    D = pd.concat([v for _it, v in res], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "trials.csv.gz", index=False)
    print("wrote", OUT / "trials.csv.gz", D.shape)


# =============================================================================================== main
def times_worker(lab: str) -> pd.DataFrame:
    """One session's per-trial strobe / stop (ms from the cue; `trial_classes._trial_times` via `session_trials`)."""
    from wfield_local import trial_classes as tc
    T = tc.session_trials(lab)
    return T[["trial_id", "strobe_ms", "stop_ms"]].assign(label=lab)


def extract_times(jobs: int | None):
    """Added 11-08 for the slope-unit test (Priya: "trial lengths can vary post-stroke though (enl licking changes)"):
    the per-trial strobe -> cue wait and cue -> trial end, joined later on (label, trial_id)."""
    from wfield_local import analysis_kit as ak
    labs = sorted(pd.read_csv(OUT / "trials.csv.gz", usecols=["label"]).label.unique())
    res, fail = ak.fan_sessions(labs, times_worker, jobs=jobs)
    if fail:
        raise SystemExit(f"failures: {fail}")
    pd.concat([v for _k, v in res], ignore_index=True).to_csv(OUT / "trial_times.csv.gz", index=False)
    print("wrote", OUT / "trial_times.csv.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "times", "analyze", "plot"])
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    if a.stage == "times":
        extract_times(a.jobs)
    elif a.stage == "extract":
        extract(a.jobs)
    elif a.stage == "analyze":
        analyze()
    else:
        plot()


# =============================================================================================== analyze helpers
MIN_ALL, MIN_FC = 15, 8          # trials a session slope needs (all spouts / one spout), as the state-space script
PRO_LATE, PRO_EARLY = 10.0, 30.0  # prodrome windows (min before the quit): late [-10, 0), early [-30, -10)
N_BOOT = 2000


def load_trials() -> pd.DataFrame:
    """The per-trial table + per-trial deviations from the animal's AUGUST pre mean at the same spout (working first
    licks): dside (mm, - = ipsilesional), dlat (ms, camera first-lick peak), dcontact, dspeed, dreach, dnlick."""
    D = pd.read_csv(OUT / "trials.csv.gz")
    D = D[D.animal.isin(ANIMALS) & D.epoch.isin(EPS)].copy()
    D["first"] = D.side_ml.notna()
    W = D[D.working & D["first"]]
    for c, src in (("dside", "side_ml"), ("dlat", "lat_cam"), ("dcontact", "contact1"), ("dspeed", "speed1"),
                   ("dreach", "reach1")):
        ref = W[W.epoch == "pre"].groupby(["animal", "pos"])[src].mean()
        D[c] = D[src] - pd.MultiIndex.from_frame(D[["animal", "pos"]]).map(ref).to_numpy(float)
    ref = D[D.working & (D.epoch == "pre")].groupby(["animal", "pos"]).n_licks.mean()
    D["dnlick"] = D.n_licks - pd.MultiIndex.from_frame(D[["animal", "pos"]]).map(ref).to_numpy(float)
    D["t_hr"] = D.t_min / 60.0
    return D


def _slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


#: per-session drift measures: (name, column, spout subset (None = all, demeaned within spout), minimum trials)
MEASURES = [("fc_dside", "dside", "far_contra", MIN_FC), ("oth_dside", "dside", "not_fc", MIN_ALL),
            ("all_dside", "dside", None, MIN_ALL), ("lat", "dlat", None, MIN_ALL),
            ("fc_lat", "dlat", "far_contra", MIN_FC), ("contact", "dcontact", None, MIN_ALL),
            ("speed", "dspeed", None, MIN_ALL), ("reach", "dreach", None, MIN_ALL),
            ("nlick", "dnlick", None, MIN_ALL), ("far_dside", "dside", "far", MIN_ALL)]
UNITS = {"fc_dside": "mm", "oth_dside": "mm", "all_dside": "mm", "lat": "ms", "fc_lat": "ms", "contact": "prop.",
         "speed": "mm/s", "reach": "mm", "nlick": "licks/trial", "far_dside": "mm"}


def _subset(g, sub):
    if sub is None:
        return g
    if sub == "not_fc":
        return g[g.pos != "far_contra"]
    if sub == "far":
        return g[g.pos.str.startswith("far")]
    return g[g.pos == sub]


def session_slopes(D: pd.DataFrame, cut_before_quit_min: float = 0.0) -> pd.DataFrame:
    """Per session x measure, on working trials with a 3-D first lick (n_licks rows need only working); multi-spout
    measures demeaned within spout first; ``cut_before_quit_min`` drops trials that close to the camera quit:
      ``_hr``    slope per HOUR of elapsed time -- PRIMARY (Priya 11-08: post-stroke sessions are shorter, so a
                 per-normalized-session slope biases against the stroke effect; the pacing tests say the drift is
                 clock / state-like, so time is the natural unit);
      ``_trial`` slope per 100 TRIALS of the session (DAQ cue index, every trial counted) -- ROBUSTNESS row (it mixes
                 the drift with the trial rate, which itself changes after the stroke);
      ``_ord``   per normalized order (first -> last trial) -- kept ONLY to check the reproduction of the state-space
                 numbers; no longer tested or reported (Priya 11-08)."""
    rows = []
    for lab, g in D[D.working].groupby("label", sort=True):
        if cut_before_quit_min and np.isfinite(g.quit_min.iloc[0]):
            g = g[g.t_min < g.quit_min.iloc[0] - cut_before_quit_min]
        r = dict(label=lab, animal=g.animal.iloc[0], epoch=g.epoch.iloc[0], day=g.day.iloc[0],
                 censored=bool(g.censored.iloc[0]), quit_min=g.quit_min.iloc[0], end_min=g.end_min.iloc[0],
                 n_working=len(g), work_span_min=float(g.t_min.max() - g.t_min.min()) if len(g) else np.nan)
        for name, col, sub, nmin in MEASURES:
            h = _subset(g, sub).dropna(subset=[col])
            if len(h) < nmin:
                r[f"{name}_hr"] = r[f"{name}_trial"] = r[f"{name}_ord"] = r[f"{name}_mean"] = np.nan
                continue
            y = h[col].to_numpy(float)
            if sub in (None, "not_fc", "far"):
                y = y - h.groupby("pos")[col].transform("mean").to_numpy(float)
            order = np.argsort(np.argsort(h.k.to_numpy()))
            r[f"{name}_hr"] = _slope(h.t_hr.to_numpy(float), y)
            r[f"{name}_trial"] = _slope(h.k.to_numpy(float) / 100.0, y)
            r[f"{name}_ord"] = _slope(order / max(len(h) - 1, 1), y)
            r[f"{name}_mean"] = float(h[col].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def per_animal(S: pd.DataFrame, col: str) -> pd.DataFrame:
    return S.groupby(["animal", "epoch"])[col].mean().unstack("epoch").reindex(columns=list(EPS))


def by_animal(S: pd.DataFrame, col: str, epoch: str) -> dict:
    g = S[(S.epoch == epoch)].dropna(subset=[col])
    return {a: g[g.animal == a][col].tolist() for a in sorted(g.animal.unique())}


def delta_rows(S: pd.DataFrame, cols, family: str) -> pd.DataFrame:
    """boot_delta (change vs own pre, animal-weighted) per column x post epoch; per-animal table printed FIRST
    (rule 8). Returns the long contrast table with bootstrap p and the n animals positive."""
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(SEED)
    out = []
    for c in cols:
        P = per_animal(S, c)
        print(f"\n[{family}] per-animal session means: {c}")
        print(P.round(3).to_string())
        for ep in POST:
            post, pre = by_animal(S, c, ep), by_animal(S, c, "pre")
            iv, o = ak.boot_delta(post, pre, rng, N_BOOT, draws=True)
            if iv is None:
                continue
            d = (P[ep] - P["pre"]).dropna()
            out.append(dict(family=family, measure=c, epoch=ep, delta=iv.point, lo=iv.lo, hi=iv.hi,
                            p_boot=ak.boot_p(o), n_animals=iv.n_animals, n_pos=int((d > 0).sum()),
                            n_neg=int((d < 0).sum()),
                            per_animal=" ".join(f"{a}:{v:+.3g}" for a, v in d.items())))
    R = pd.DataFrame(out)
    if len(R):
        print(R.drop(columns=["family"]).round(3).to_string(index=False))
    return R


def level_rows(S: pd.DataFrame, cols, family: str) -> pd.DataFrame:
    """Is the within-session slope itself non-zero in an epoch? `boot_ci` on the session slopes (a LEVEL)."""
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(SEED + 1)
    out = []
    for c in cols:
        for ep in EPS:
            ba = by_animal(S, c, ep)
            iv = ak.boot_ci(ba, rng, N_BOOT)
            if iv is None:
                continue
            am = {a: float(np.mean(v)) for a, v in ba.items() if v}
            out.append(dict(family=family, measure=c, epoch=ep, level=iv.point, lo=iv.lo, hi=iv.hi,
                            n_animals=iv.n_animals, n_pos=sum(v > 0 for v in am.values()),
                            n_neg=sum(v < 0 for v in am.values()),
                            per_animal=" ".join(f"{a}:{v:+.3g}" for a, v in am.items())))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- Q2 prodrome
PRO_COLS = ["dside_fc", "dside_oth", "dlat", "dcontact", "dspeed", "dnlick"]


def _window_mean(g: pd.DataFrame, lo: float, hi: float) -> dict:
    w = g[(g.t_min >= lo) & (g.t_min < hi)]
    fc = w[w.pos == "far_contra"]
    ot = w[w.pos != "far_contra"]
    return dict(dside_fc=fc.dside.mean() if fc.dside.notna().sum() >= 3 else np.nan,
                dside_oth=ot.dside.mean() if ot.dside.notna().sum() >= 5 else np.nan,
                dlat=w.dlat.mean() if w.dlat.notna().sum() >= 5 else np.nan,
                dcontact=w.dcontact.mean() if w.dcontact.notna().sum() >= 5 else np.nan,
                dspeed=w.dspeed.mean() if w.dspeed.notna().sum() >= 5 else np.nan,
                dnlick=w.dnlick.mean() if w.dnlick.notna().sum() >= 5 else np.nan)


def prodrome(D: pd.DataFrame) -> pd.DataFrame:
    """Per QUITTING session (camera quit): late-minus-early window change (late = the 10 min before the quit, early =
    the 20 min before that) on working trials, minus the SAME clock windows in the same animal's other sessions (same
    epoch) that were still working through the late window (time-matched controls, as `quit_prodrome`). Positive dlat
    / negative dside = degradation."""
    W = D[D.working].copy()
    sess = W.groupby("label").agg(animal=("animal", "first"), epoch=("epoch", "first"), quit=("quit_min", "first"),
                                  cens=("censored", "first"), last=("t_min", "max"))
    rows = []
    for lab, s in sess.iterrows():
        if s.cens or not np.isfinite(s.quit) or s.quit < PRO_EARLY + 5:
            continue
        Q = s.quit
        g = W[W.label == lab]
        a, b = _window_mean(g, Q - PRO_EARLY, Q - PRO_LATE), _window_mean(g, Q - PRO_LATE, Q)
        ctrl = sess[(sess.animal == s.animal) & (sess.epoch == s.epoch) & (sess.index != lab)
                    & (sess["last"] >= Q) & ((sess.cens) | (sess.quit >= Q + 5))]
        cd = {c: [] for c in PRO_COLS}
        for cl in ctrl.index:
            gc = W[W.label == cl]
            ca, cb = _window_mean(gc, Q - PRO_EARLY, Q - PRO_LATE), _window_mean(gc, Q - PRO_LATE, Q)
            for c in PRO_COLS:
                cd[c].append(cb[c] - ca[c])
        r = dict(label=lab, animal=s.animal, epoch=s.epoch, quit_min=Q, n_ctrl=len(ctrl))
        for c in PRO_COLS:
            r[f"{c}_quit"] = b[c] - a[c]
            r[f"{c}_ctrl"] = float(np.nanmean(cd[c])) if np.isfinite(cd[c]).any() else np.nan
            r[f"{c}_excess"] = r[f"{c}_quit"] - r[f"{c}_ctrl"]
        rows.append(r)
    return pd.DataFrame(rows)


QA_BINS = np.arange(-40, 1, 5.0)


def quit_aligned_curves(D: pd.DataFrame, bins=QA_BINS) -> pd.DataFrame:
    """Per quitting session, 5-min bins of working trials before the quit (dside far contra / others, dlat, dcontact),
    with the time-matched control sessions' values in the same clock bins."""
    W = D[D.working]
    sess = W.groupby("label").agg(animal=("animal", "first"), epoch=("epoch", "first"), quit=("quit_min", "first"),
                                  cens=("censored", "first"), last=("t_min", "max"))
    rows = []
    for lab, s in sess.iterrows():
        if s.cens or not np.isfinite(s.quit):
            continue
        ctrl = sess[(sess.animal == s.animal) & (sess.epoch == s.epoch) & (sess.index != lab)
                    & (sess["last"] >= s.quit) & ((sess.cens) | (sess.quit >= s.quit + 5))].index
        for arm, labs in (("quit", [lab]), ("ctrl", list(ctrl))):
            for b0 in bins[:-1]:
                vals = {c: [] for c in PRO_COLS}
                for lb in labs:
                    m = _window_mean(W[W.label == lb], s.quit + b0, s.quit + b0 + 5.0)
                    for c in PRO_COLS:
                        vals[c].append(m[c])
                rows.append(dict(label=lab, animal=s.animal, epoch=s.epoch, arm=arm, bin=b0 + 2.5,
                                 **{c: (float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
                                    for c, v in vals.items()}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- Q3 time / effort / water
PRED = {"t_hr": ("elapsed time", "per h"), "licks_k": ("cumulative camera licks", "per 1000 licks"),
        "water_100": ("cumulative collected water", "per 100 ul"), "fc_10": ("cumulative far-contra trials",
                                                                            "per 10 trials")}


def _r2(X, Y, cols):
    """R2 (of the already-demeaned Y) and coefficients of Y on the columns ``cols`` of X (no intercept: demeaned)."""
    Xi = X[:, cols]
    b, *_ = np.linalg.lstsq(Xi, Y, rcond=None)
    return 1 - float(((Y - Xi @ b) ** 2).sum()) / float((Y ** 2).sum()), b


def pacing(D: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within animal x epoch, working trials pooled over sessions, session x spout fixed effects (all within-session):
    y on each predictor ALONE (R2 of the within-session part) and on all four jointly (coefficients + unique R2 =
    drop in R2 when that predictor is left out). Also the within-session correlations between the predictors (the
    collinearity that limits the separation)."""
    W = D[D.working].copy()
    W["licks_k"], W["water_100"], W["fc_10"] = W.cum_licks_cam / 1000, W.cum_water_ul / 100, W.cum_fc_trials / 10
    W = W.dropna(subset=["water_100"])
    ys = {"fc_dside": ("dside", "far_contra"), "oth_dside": ("dside", "not_fc"), "lat": ("dlat", None),
          "contact": ("dcontact", None)}
    rows, cors = [], []
    preds = list(PRED)
    for (an, ep), g0 in W.groupby(["animal", "epoch"]):
        for yname, (ycol, sub) in ys.items():
            g = _subset(g0, sub).dropna(subset=[ycol])
            if len(g) < 60 or g.label.nunique() < 1:
                continue
            grp = g.label + "|" + g.pos
            Y = g[ycol].to_numpy(float)
            Y = Y - g.groupby(grp)[ycol].transform("mean").to_numpy(float)
            X = np.column_stack([g[p].to_numpy(float) - g.groupby(grp)[p].transform("mean").to_numpy(float)
                                 for p in preds])
            full, bf = _r2(X, Y, list(range(len(preds))))
            r = dict(animal=an, epoch=ep, y=yname, n=len(g), n_sess=g.label.nunique(), r2_full=full)
            for i, p in enumerate(preds):
                r[f"r2_{p}"] = _r2(X, Y, [i])[0]
                r[f"uniq_{p}"] = full - _r2(X, Y, [j for j in range(len(preds)) if j != i])[0]
                r[f"b_{p}"] = float(bf[i])
            rows.append(r)
            if yname == "lat":
                C = np.corrcoef(X.T)
                cors.append(dict(animal=an, epoch=ep, **{f"r_{preds[i]}_{preds[j]}": C[i, j]
                                                          for i in range(len(preds)) for j in range(i + 1, len(preds))}))
    return pd.DataFrame(rows), pd.DataFrame(cors)


def accrual_instrument(S: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    """Session-level pacing (the 09-20 design): if the drift is paced by WATER, its per-hour slope should grow with
    the session's water accrual rate and its per-ul slope should not; if paced by the CLOCK, the reverse. The rate is
    the EARLY rate (first 30 min, collected ul / min) and the drop size (median ul per collected drop, set by the
    experimenter = exogenous). Within animal x epoch-group the session values are centered; output = per-session rows."""
    W = D.copy()
    early = W[W.t_min < 30].groupby("label").agg(w30=("coll_ul", "sum"), l30=("n_licks", "sum"))
    drop = W[W.coll_ul > 0].groupby("label").coll_ul.median().rename("drop_ul")
    tot = W[W.working].groupby("label").agg(w_work=("coll_ul", "sum"), t_work=("t_min", "max"),
                                            l_work=("n_licks", "sum"))
    X = S.set_index("label").join(early).join(drop).join(tot)
    X["water_rate"] = X.w30 / 30.0
    X["lick_rate"] = X.l30 / 30.0
    X["grp"] = np.where(X.epoch == "pre", "pre", np.where(X.epoch == "chronic", "chronic", "early_post"))
    return X.reset_index()


def within_animal_assoc(X: pd.DataFrame, xcol: str, ycol: str, groups=("pre", "early_post", "chronic"),
                        n_perm: int = 5000) -> pd.DataFrame:
    """Within-animal Spearman r of x and y across sessions, per animal x group (>= 4 sessions), the mean over animals,
    and a within-animal shuffle null (y permuted among each animal's sessions)."""
    from scipy import stats
    rng = np.random.default_rng(SEED + 2)
    out = []
    for gname in groups:
        G = X[(X.grp == gname)].dropna(subset=[xcol, ycol])
        per = {a: g for a, g in G.groupby("animal") if len(g) >= 4}
        if not per:
            continue
        rs = {a: float(stats.spearmanr(g[xcol], g[ycol])[0]) for a, g in per.items()}
        obs = float(np.nanmean(list(rs.values())))
        null = np.empty(n_perm)
        for i in range(n_perm):
            null[i] = np.nanmean([stats.spearmanr(g[xcol], rng.permutation(g[ycol].to_numpy()))[0]
                                  for g in per.values()])
        p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
        out.append(dict(x=xcol, y=ycol, group=gname, mean_r=obs, p_shuffle=p, n_animals=len(per),
                        n_sessions=int(sum(len(g) for g in per.values())),
                        per_animal=" ".join(f"{a}:{v:+.2f}(n={len(per[a])})" for a, v in rs.items())))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- slope unit (11-08)
UNIT_MEASURES = ("lat", "fc_dside", "far_dside")
DUR_COLS = ("rate_tph", "cycle_s", "prewait_s", "post_s", "iti_s")


def session_durations(D: pd.DataFrame) -> pd.DataFrame:
    """Per session, working trials: trial RATE (trials per working hour), median cue -> next cue CYCLE (s), and its
    parts: PRE-CUE WAIT strobe -> cue (s; the ENL period, lengthened by ENL licks that reset it), POST = cue -> trial
    end (stop, s), ITI = the rest (trial end -> next strobe: spout move / dock). Means over working trials."""
    TT = pd.read_csv(OUT / "trial_times.csv.gz")
    X = D.merge(TT, on=["label", "trial_id"], how="left")
    rows = []
    for lab, g in X.sort_values(["label", "k"]).groupby("label", sort=True):
        cyc = np.diff(g.t_min.to_numpy() * 60.0)
        nxt_work = (g.working.to_numpy()[:-1]) & (np.diff(g.k.to_numpy()) == 1)
        w = g[g.working]
        span_h = (w.t_min.max() - w.t_min.min()) / 60.0 if len(w) > 1 else np.nan
        cycle = float(np.median(cyc[nxt_work])) if nxt_work.any() else np.nan
        pre = float(np.nanmedian(-w.strobe_ms)) / 1000.0
        post = float(np.nanmedian(w.stop_ms)) / 1000.0
        rows.append(dict(label=lab, rate_tph=len(w) / span_h if span_h and span_h > 0 else np.nan, cycle_s=cycle,
                         prewait_s=pre, post_s=post, iti_s=cycle - pre - post))
    return pd.DataFrame(rows)


def _elasticity(X: pd.DataFrame, ycol: str, group: str) -> float:
    """Pooled within-animal elasticity of a session slope to the trial rate: y = a_animal + c * log(rate), divided by
    the group's mean slope. Clock-paced drift: per-hour e = 0, per-trial e = -1; trial-paced: per-hour +1, per-trial 0.
    Descriptive only (unstable when the mean slope is near 0)."""
    G = X[X.grp == group].dropna(subset=[ycol, "rate_tph"])
    G = G[G.groupby("animal").label.transform("size") >= 4]
    if len(G) < 8:
        return np.nan
    xr = np.log(G.rate_tph) - G.groupby("animal").rate_tph.transform(lambda v: np.log(v).mean())
    yr = G[ycol] - G.groupby("animal")[ycol].transform("mean")
    c = float((xr * yr).sum() / (xr ** 2).sum())
    return c / float(G.groupby("animal")[ycol].mean().mean())


def unit_test(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Priya 11-08: which slope unit is independent of the session's trial rate? Per measure x unit, within-animal
    Spearman (shuffle null) of the session slope with the trial rate and with each duration part, by epoch group,
    + the pooled elasticity; and the duration parts' change vs own pre."""
    from wfield_local import analysis_kit as ak
    A, E = [], []
    for m in UNIT_MEASURES:
        for unit in ("hr", "trial"):
            y = f"{m}_{unit}"
            for d in DUR_COLS:
                A.append(within_animal_assoc(X, d, y).assign(measure=m, unit=unit))
            for gname in ("pre", "early_post", "chronic"):
                E.append(dict(measure=m, unit=unit, group=gname, elasticity=_elasticity(X, y, gname)))
    A = pd.concat(A, ignore_index=True)
    E = pd.DataFrame(E)
    rng = np.random.default_rng(SEED + 5)
    out = []
    for d in DUR_COLS:
        print(f"\n[unit] per-animal session means: {d}")
        print(per_animal(X, d).round(3).to_string())
        for ep in POST:
            iv = ak.boot_delta(by_animal(X, d, ep), by_animal(X, d, "pre"), rng, N_BOOT)
            if iv is None:
                continue
            Pm = per_animal(X, d)
            dd = (Pm[ep] - Pm["pre"]).dropna()
            out.append(dict(measure=d, epoch=ep, delta=iv.point, lo=iv.lo, hi=iv.hi, n_pos=int((dd > 0).sum()),
                            n_neg=int((dd < 0).sum()), per_animal=" ".join(f"{a}:{v:+.3g}" for a, v in dd.items())))
    return A, E, pd.DataFrame(out)


def spout_slopes(D: pd.DataFrame) -> pd.DataFrame:
    """Per session x spout, per-hour slope of dside and dlat (working, 3-D first lick, >= MIN_FC trials)."""
    rows = []
    for (lab, pos), g in D[D.working].groupby(["label", "pos"], sort=True):
        r = dict(label=lab, pos=pos, animal=g.animal.iloc[0], epoch=g.epoch.iloc[0])
        for c in ("dside", "dlat"):
            h = g.dropna(subset=[c])
            r[f"{c}_hr"] = _slope(h.t_hr.to_numpy(float), h[c].to_numpy(float)) if len(h) >= MIN_FC else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def analyze():
    from wfield_local import orofacial_cohort as oc
    TAB.mkdir(parents=True, exist_ok=True)
    D = load_trials()
    print("sessions per animal x epoch:\n", D.groupby(["animal", "epoch"]).label.nunique().unstack()
          .reindex(columns=list(EPS)))
    print("t0 check (cache t0 - DAQ first cue, s): max |.| =", float(np.nanmax(np.abs(D.beh_t0_check))))
    print("sessions without the water cache:", sorted(D[D.coll_ul.isna()].label.unique()))
    q = D.groupby("label").agg(animal=("animal", "first"), epoch=("epoch", "first"), cens=("censored", "first"),
                               has=("has_classes", "first"))
    print("camera quits (quit / sessions):\n", q[q.has].assign(qq=lambda x: ~x.cens).groupby(["animal", "epoch"])
          .qq.agg(lambda v: f"{int(v.sum())}/{len(v)}").unstack().reindex(columns=list(EPS)))

    # ---- Q1: reproduce + extend (per normalized order AND per hour), full working period and away from the quit
    S = session_slopes(D)
    S.to_csv(TAB / "session_slopes.csv", index=False)
    S15 = session_slopes(D, cut_before_quit_min=15.0)
    S15.to_csv(TAB / "session_slopes_cut15.csv", index=False)
    cols_hr = [f"{m[0]}_hr" for m in MEASURES]
    cols_tr = [f"{m[0]}_trial" for m in MEASURES]
    R = pd.concat([delta_rows(S, cols_hr, "Q1_hour"), delta_rows(S, cols_tr, "Q1_trial"),
                   delta_rows(S15, ["fc_dside_hr", "lat_hr", "oth_dside_hr", "contact_hr"], "Q1_hour_cut15")])
    L = pd.concat([level_rows(S, cols_hr, "Q1_hour_level"), level_rows(S, cols_tr, "Q1_trial_level")])
    L.to_csv(TAB / "q1_slope_levels.csv", index=False)
    print("\n[Q1 levels] within-session slope itself, boot_ci (flat pool):")
    print(L.round(3).to_string(index=False))

    # cross-check against the state-space observation (same sessions)
    if SS_SLOPES.exists():
        O = pd.read_csv(SS_SLOPES)[["label", "s_rt", "s_fc_dside", "s_dside", "s_contact"]]
        J = S.merge(O, on="label")
        print("\n[check] r(ours, state-space) over sessions: lat", round(J[["lat_ord", "s_rt"]].corr().iloc[0, 1], 2),
              "| fc dside", round(J[["fc_dside_ord", "s_fc_dside"]].corr().iloc[0, 1], 2),
              "| all dside", round(J[["all_dside_ord", "s_dside"]].corr().iloc[0, 1], 2),
              "| contact", round(J[["contact_ord", "s_contact"]].corr().iloc[0, 1], 2), "| n", len(J))

    # ---- Q2: prodrome
    P = prodrome(D)
    P.to_csv(TAB / "q2_prodrome_sessions.csv", index=False)
    print("\n[Q2] quitting sessions with >= 35 min before the quit, per animal x epoch:\n",
          P.groupby(["animal", "epoch"]).size().unstack().reindex(columns=list(EPS)))
    C = quit_aligned_curves(D)
    C.to_csv(TAB / "q2_quit_aligned_bins.csv", index=False)
    pro = []
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(SEED + 3)
    for c in PRO_COLS:
        for kind in ("quit", "excess"):
            col = f"{c}_{kind}"
            print(f"\n[Q2] per-animal mean {col}:")
            print(P.groupby(["animal", "epoch"])[col].mean().unstack().reindex(columns=list(EPS)).round(3).to_string())
            for ep in EPS:
                ba = by_animal(P, col, ep)
                iv = ak.boot_ci(ba, rng, N_BOOT)
                if iv is None:
                    continue
                am = {a: float(np.nanmean(v)) for a, v in ba.items() if v}
                pro.append(dict(measure=col, epoch=ep, level=iv.point, lo=iv.lo, hi=iv.hi, n_animals=iv.n_animals,
                                n_sessions=sum(len(v) for v in ba.values()),
                                per_animal=" ".join(f"{a}:{v:+.3g}" for a, v in am.items())))
    PR = pd.DataFrame(pro)
    PR.to_csv(TAB / "q2_prodrome_levels.csv", index=False)
    print(PR.round(3).to_string(index=False))

    # ---- Q3: time vs effort vs water
    PA, CO = pacing(D)
    PA.to_csv(TAB / "q3_pacing_fe.csv", index=False)
    CO.to_csv(TAB / "q3_predictor_correlations.csv", index=False)
    with pd.option_context("display.width", 250):
        print("\n[Q3] within-session predictor correlations (after session x spout FE):\n", CO.round(2).to_string())
        print("\n[Q3] single-predictor R2 / unique R2 (x100):")
        show = PA[["animal", "epoch", "y", "n"] + [f"r2_{p}" for p in PRED] + [f"uniq_{p}" for p in PRED]].copy()
        for c in show.columns[4:]:
            show[c] = (show[c] * 100).round(2)
        print(show.to_string(index=False))
    X = accrual_instrument(S, D)
    X = X.merge(session_durations(D), on="label", how="left")
    X.to_csv(TAB / "q3_session_accrual.csv", index=False)

    # ---- slope unit (Priya 11-08): per hour vs per trial, which is independent of the trial rate / duration?
    UA, UE, UD = unit_test(X)
    UA.to_csv(TAB / "unit_rate_assoc.csv", index=False)
    UE.to_csv(TAB / "unit_elasticity.csv", index=False)
    UD.to_csv(TAB / "unit_durations_vs_pre.csv", index=False)
    with pd.option_context("display.width", 250):
        print("\n[unit] duration parts, change vs own pre:\n", UD.round(3).to_string(index=False))
        print("\n[unit] within-animal r of session slope with trial rate / duration parts:\n",
              UA[["measure", "unit", "x", "group", "mean_r", "p_shuffle", "per_animal"]].round(3).to_string(index=False))
        print("\n[unit] pooled elasticity to trial rate (clock: hr 0 / trial -1; trial-paced: hr +1 / trial 0):\n",
              UE.round(2).to_string(index=False))
    A = pd.concat([within_animal_assoc(X, xc, yc) for yc in ("fc_dside_hr", "lat_hr", "oth_dside_hr")
                   for xc in ("water_rate", "drop_ul", "lick_rate")])
    print("\n[Q3 instrument] within-animal r of the per-hour drift with the early accrual rates:\n",
          A.round(3).to_string(index=False))

    # ---- Q4: drift vs quitting
    X["quit_or_end"] = np.where(X.censored, X.end_min, X.quit_min)
    X["quit_early"] = ~X.censored
    A4 = pd.concat([within_animal_assoc(X, xc, yc) for yc in ("quit_or_end", "n_working")
                    for xc in ("fc_dside_hr", "lat_hr", "oth_dside_hr", "fc_dside_trial", "lat_trial")])
    Xq = X[~X.censored]
    A4q = pd.concat([within_animal_assoc(Xq, xc, "quit_min") for xc in ("fc_dside_hr", "lat_hr")]).assign(
        note="quitting sessions only")
    A4 = pd.concat([A4, A4q])
    print("\n[Q4] within-animal r: drift vs time to quit-or-end / working trials:\n", A4.round(3).to_string(index=False))

    # ---- Q5: per spout + recovery axis
    SP = spout_slopes(D)
    SP.to_csv(TAB / "q5_spout_slopes.csv", index=False)
    q5 = []
    rng = np.random.default_rng(SEED + 4)
    for pos in oc.POS_ORDER:
        G = SP[SP.pos == pos]
        for c in ("dside_hr", "dlat_hr"):
            for ep in POST:
                iv, o = ak.boot_delta(by_animal(G, c, ep), by_animal(G, c, "pre"), rng, N_BOOT, draws=True)
                if iv is None:
                    continue
                Pm = per_animal(G, c)
                d = (Pm[ep] - Pm["pre"]).dropna()
                q5.append(dict(pos=pos, measure=c, epoch=ep, delta=iv.point, lo=iv.lo, hi=iv.hi, p_boot=ak.boot_p(o),
                               n_animals=iv.n_animals, n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()),
                               per_animal=" ".join(f"{a}:{v:+.3g}" for a, v in d.items())))
            print(f"\n[Q5] per-animal {pos} {c}:\n", per_animal(G, c).round(3).to_string())
    Q5 = pd.DataFrame(q5)
    print(Q5.round(3).to_string(index=False))
    if AIM.exists():
        AM = pd.read_csv(AIM)[["label", "farcontra_first", "fcf_h1", "fcf_h2"]]
        X = X.merge(AM, on="label", how="left")
        A5 = pd.concat([within_animal_assoc(X, "farcontra_first", yc, groups=("early_post", "chronic"))
                        for yc in ("fc_dside_hr", "lat_hr", "fc_dside_mean")])
        print("\n[Q5] within-animal r: session far-contra aim (deg, aim_deficit_sessions) vs drift:\n",
              A5.round(3).to_string(index=False))
    else:
        A5 = pd.DataFrame()
    X.to_csv(TAB / "session_table.csv", index=False)
    pd.concat([R, Q5.assign(family="Q5_spout")]).to_csv(TAB / "contrasts_vs_pre.csv", index=False)
    pd.concat([A.assign(family="Q3_instrument"), A4.assign(family="Q4_quit"), A5.assign(family="Q5_aim")]).to_csv(
        TAB / "within_animal_assoc.csv", index=False)
    print("\nwrote", TAB)


# =============================================================================================== plot
EP_X = {e: i for i, e in enumerate(EPS)}
PROV = ("Data: cohort1 PS92-95, curated sessions, August-only pre; trials = camera working period (to the last camera "
        "lick); first lick = 3-D lick_idx 0 (pose3d licks.csv.gz); aim = lateral landing minus the animal's pre mean at "
        "that spout (mm, - = ipsilesional); latency = 3-D first-lick peak after the cue (ms); stars = boot_delta vs own "
        "pre (animals -> sessions, 95 % CI excludes 0).")


def _sessions_panel(ax, S, col, stars, title):
    from wfield_local import orofacial_cohort as oc
    from wfield_local import sig_marks as sm
    rng = np.random.default_rng(0)
    for (an, ep), g in S.dropna(subset=[col]).groupby(["animal", "epoch"]):
        x = EP_X[ep] + rng.uniform(-0.22, 0.22, len(g))
        ax.scatter(x, g[col], s=16, marker=oc.ANIMAL_MARKER[an], facecolors="none",
                   edgecolors=oc.EPOCH_LINE[ep], linewidths=0.8, zorder=3)
    am = S.groupby(["animal", "epoch"])[col].mean().unstack().reindex(columns=list(EPS))
    for ep in EPS:
        v = am[ep].dropna()
        if len(v):
            ax.hlines(v.mean(), EP_X[ep] - 0.3, EP_X[ep] + 0.3, color=oc.EPOCH_LINE[ep], lw=2.2, zorder=4)
    ax.axhline(0, color="0.6", lw=0.6, zorder=1)
    if np.isfinite(am["pre"]).any():
        ax.axhline(am["pre"].mean(), color="k", lw=0.8, ls="--", zorder=1)
    for ep in POST:
        r = stars.get((col, ep))
        if r is not None and (r["lo"] > 0 or r["hi"] < 0):
            sm.star_top(ax, EP_X[ep], "*", color=oc.EPOCH_LINE[ep])
    ax.set_xticks(range(len(EPS)), EPS, fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import orofacial_cohort as oc
    from wfield_local import sig_marks as sm
    from wfield_local.figure_layout import save_png_svg
    FIG.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(TAB / "session_slopes.csv")
    R = pd.read_csv(TAB / "contrasts_vs_pre.csv")
    stars = {(r["measure"], r["epoch"]): r for r in R[R.family == "Q1_hour"].to_dict("records")}
    leg = ([Line2D([], [], ls="", marker=m, mfc="none", mec="k", label=a) for a, m in oc.ANIMAL_MARKER.items()]
           + [Line2D([], [], color=c, lw=2, label=e) for e, c in oc.EPOCH_LINE.items()])

    # ---- Fig 1: within-session slope per hour, by epoch
    panels = [("fc_dside_hr", "far-contra aim drift (mm / h)"), ("oth_dside_hr", "other spouts aim drift (mm / h)"),
              ("lat_hr", "first-lick latency drift (ms / h)"), ("contact_hr", "first-lick contact drift (prop. / h)"),
              ("speed_hr", "first-lick peak speed drift (mm/s / h)"), ("nlick_hr", "camera licks / trial drift (/ h)")]
    fig, axs = plt.subplots(2, 3, figsize=(9.5, 5.6))
    for ax, (c, t) in zip(axs.flat, panels):
        _sessions_panel(ax, S, c, stars, t)
    fig.legend(handles=leg, fontsize=6, frameon=False, ncol=8, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Fig 1. Within-session drift per hour of the working period (session = open marker; bar = mean of "
                 "animal means; dashed = pre)", fontsize=9)
    fig.text(0.01, 0.005, PROV + " Slope = OLS on elapsed hours, multi-spout measures demeaned within spout.",
             fontsize=5.5, wrap=True)
    sm.reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_png_svg(fig, FIG / "wsd_fig1_slopes_per_hour.png", dpi=200)
    plt.close(fig)

    # ---- Fig 1b: the same per 100 trials (robustness row, Priya 11-08)
    stars_t = {(r["measure"], r["epoch"]): r for r in R[R.family == "Q1_trial"].to_dict("records")}
    fig, axs = plt.subplots(2, 3, figsize=(9.5, 5.6))
    for ax, (c, t) in zip(axs.flat, panels):
        _sessions_panel(ax, S, c.replace("_hr", "_trial"), stars_t, t.replace("/ h)", "/ 100 trials)"))
    fig.legend(handles=leg, fontsize=6, frameon=False, ncol=8, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Fig 1b. Within-session drift per 100 trials (robustness row; session = open marker; bar = mean of "
                 "animal means; dashed = pre)", fontsize=9)
    fig.text(0.01, 0.005, PROV + " Slope = OLS on the session trial index (every trial counted) / 100.",
             fontsize=5.5, wrap=True)
    sm.reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_png_svg(fig, FIG / "wsd_fig1b_slopes_per_trial.png", dpi=200)
    plt.close(fig)

    # ---- Fig 2: quit-aligned (camera quit) vs time-matched control sessions
    C = pd.read_csv(TAB / "q2_quit_aligned_bins.csv")
    meas = [("dside_fc", "far-contra aim (mm vs pre)"), ("dside_oth", "other spouts aim (mm vs pre)"),
            ("dlat", "first-lick latency (ms vs pre)"), ("dcontact", "first-lick contact (vs pre)"),
            ("dnlick", "camera licks / trial (vs pre)")]
    fig, axs = plt.subplots(len(meas), len(EPS), figsize=(10, 10), sharex=True, sharey="row")
    for i, (m, t) in enumerate(meas):
        for j, ep in enumerate(EPS):
            ax = axs[i, j]
            for arm, ls in (("quit", "-"), ("ctrl", ":")):
                g = C[(C.epoch == ep) & (C.arm == arm)]
                if not len(g):
                    continue
                # animal-weighted: mean over animals of the animal's session mean per bin
                A = g.groupby(["animal", "bin"])[m].mean().unstack("animal")
                mu, se = A.mean(1), A.std(1) / np.sqrt(A.notna().sum(1).clip(lower=1))
                ax.plot(mu.index, mu, ls=ls, color=oc.EPOCH_LINE[ep], lw=1.6, label=arm)
                ax.fill_between(mu.index, mu - se, mu + se, color=oc.EPOCH_LINE[ep], alpha=0.12 if arm == "quit"
                                else 0.05)
                if arm == "quit":
                    for an, gg in g.groupby("animal"):
                        v = gg.groupby("bin")[m].mean()
                        ax.plot(v.index, v, ls="", marker=oc.ANIMAL_MARKER[an], mfc="none",
                                mec=oc.EPOCH_LINE[ep], ms=3, mew=0.6)
            ax.axhline(0, color="0.6", lw=0.6)
            ax.axvline(0, color="k", lw=0.6)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=6)
            if i == 0:
                n = C[(C.epoch == ep) & (C.arm == "quit")]
                ax.set_title(f"{ep} (quit sessions {n.label.nunique()}, animals {n.animal.nunique()})", fontsize=7)
            if j == 0:
                ax.set_ylabel(t, fontsize=7)
            if i == len(meas) - 1:
                ax.set_xlabel("minutes before the camera quit", fontsize=7)
    axs[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("Fig 2. Aim, latency, contact, licks before the camera quit (solid) vs the same clock time in the same\n"
                 "animal's still-working sessions of the epoch (dotted); mean of animal means +- SEM; markers = animals",
                 fontsize=8)
    fig.text(0.01, 0.005, PROV + " Quit = trial_classes.camera_quit (tail after the last camera lick, >= 6 trials).",
             fontsize=5.5, wrap=True)
    sm.reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_png_svg(fig, FIG / "wsd_fig2_quit_aligned.png", dpi=200)
    plt.close(fig)

    # ---- Fig 3: pacing (unique R2 of time / licks / water / far-contra trials)
    PA = pd.read_csv(TAB / "q3_pacing_fe.csv")
    ys = [("fc_dside", "far-contra aim"), ("lat", "first-lick latency"), ("oth_dside", "other spouts aim")]
    fig, axs = plt.subplots(2, len(ys), figsize=(10, 5.6), sharey="row")
    for j, (y, t) in enumerate(ys):
        for i, kind in enumerate(("r2", "uniq")):
            ax = axs[i, j]
            g = PA[PA.y == y]
            for k, p in enumerate(PRED):
                for _, r in g.iterrows():
                    ax.scatter(k + (EP_X[r.epoch] - 1.5) * 0.17, 100 * r[f"{kind}_{p}"], s=16,
                               marker=oc.ANIMAL_MARKER[r.animal], facecolors="none",
                               edgecolors=oc.EPOCH_LINE[r.epoch], linewidths=0.8)
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_xticks(range(len(PRED)), ["time", "cum. licks", "cum. water", "cum. far-contra\ntrials"],
                          fontsize=6)
            ax.tick_params(labelsize=6)
            ax.set_title(f"{t}: {'R2 alone' if kind == 'r2' else 'unique R2'} (% within-session variance)",
                         fontsize=7)
    fig.legend(handles=leg, fontsize=6, frameon=False, ncol=8, loc="upper center", bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Fig 3. What paces the within-session drift: elapsed time vs effort (camera licks) vs water vs "
                 "far-contra attempts (animal x epoch, session x spout fixed effects)", fontsize=8)
    fig.text(0.01, 0.005, PROV + " Unique R2 = R2 lost when that predictor is dropped from the joint model.",
             fontsize=5.5, wrap=True)
    sm.reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_png_svg(fig, FIG / "wsd_fig3_pacing.png", dpi=200)
    plt.close(fig)

    # ---- Fig 4: drift vs time to quit (within animal, centered)
    X = pd.read_csv(TAB / "session_table.csv")
    X["quit_or_end"] = np.where(X.censored, X.end_min, X.quit_min)
    A = pd.read_csv(TAB / "within_animal_assoc.csv")
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (xc, t) in zip(axs, (("fc_dside_hr", "far-contra aim drift (mm / h)"),
                                 ("lat_hr", "latency drift (ms / h)"), ("oth_dside_hr", "other spouts aim drift"))):
        for (an, _grp), g in X.dropna(subset=[xc]).groupby(["animal", "grp"]):
            xx = g[xc] - g[xc].mean()
            yy = g.quit_or_end - g.quit_or_end.mean()
            for e, gg in g.groupby("epoch"):
                ax.scatter(xx[gg.index], yy[gg.index], s=18, marker=oc.ANIMAL_MARKER[an], facecolors="none",
                           edgecolors=oc.EPOCH_LINE[e], linewidths=[1.4 if c else 0.8 for c in ~gg.censored])
        txt = []
        for gname in ("pre", "early_post", "chronic"):
            r = A[(A.family == "Q4_quit") & (A.x == xc) & (A.y == "quit_or_end") & (A.group == gname)]
            if len(r):
                txt.append(f"{gname}: r {r.mean_r.iloc[0]:+.2f}, p {r.p_shuffle.iloc[0]:.2f}")
        ax.text(0.02, 0.02, "\n".join(txt), transform=ax.transAxes, fontsize=6)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.axvline(0, color="0.7", lw=0.5)
        ax.set_xlabel(t + ", centered within animal x epoch group", fontsize=7)
        ax.set_title(t, fontsize=8)
        ax.tick_params(labelsize=6)
    axs[0].set_ylabel("minutes to quit-or-end, centered", fontsize=7)
    fig.suptitle("Fig 4. Does a session's drift predict how early it quits? (within-animal Spearman r, within-animal "
                 "shuffle p; thick edge = quit, thin = censored)", fontsize=8)
    fig.text(0.01, 0.005, PROV, fontsize=5.5, wrap=True)
    sm.reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_png_svg(fig, FIG / "wsd_fig4_drift_vs_quit.png", dpi=200)
    plt.close(fig)

    # ---- Fig 5: per spout, change vs own pre of the per-hour drift
    SP = pd.read_csv(TAB / "q5_spout_slopes.csv")
    Q5 = R[R.family == "Q5_spout"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (c, t) in zip(axs, (("dside_hr", "aim drift change vs own pre (mm / h)"),
                                ("dlat_hr", "latency drift change vs own pre (ms / h)"))):
        for j, ep in enumerate(POST):
            for i, pos in enumerate(oc.POS_ORDER):
                Pm = per_animal(SP[SP.pos == pos], c)
                d = (Pm[ep] - Pm["pre"]).dropna()
                x = i + (j - 1) * 0.25
                for an, v in d.items():
                    ax.scatter(x, v, s=14, marker=oc.ANIMAL_MARKER[an], facecolors="none",
                               edgecolors=oc.EPOCH_LINE[ep], linewidths=0.8)
                r = Q5[(Q5.pos == pos) & (Q5.measure == c) & (Q5.epoch == ep)]
                if len(r):
                    r = r.iloc[0]
                    ax.errorbar(x, r.delta, yerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="_", color=oc.EPOCH_LINE[ep],
                                ms=9, lw=1.2)
                    if r.lo > 0 or r.hi < 0:
                        sm.star_top(ax, x, "*", color=oc.EPOCH_LINE[ep], row=j)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xticks(range(6), [p.replace("_", "\n") for p in oc.POS_ORDER], fontsize=7)
        ax.set_title(t, fontsize=8)
        ax.tick_params(labelsize=6)
    fig.legend(handles=leg, fontsize=6, frameon=False, ncol=8, loc="upper center", bbox_to_anchor=(0.5, 0.92))
    fig.suptitle("Fig 5. Is the within-session drift specific to the impaired direction? Per spout, per-hour slope, "
                 "change vs own pre (markers = animals; bar = boot_delta 95 % CI)", fontsize=8)
    fig.text(0.01, 0.005, PROV, fontsize=5.5, wrap=True)
    sm.reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_png_svg(fig, FIG / "wsd_fig5_per_spout.png", dpi=200)
    plt.close(fig)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
