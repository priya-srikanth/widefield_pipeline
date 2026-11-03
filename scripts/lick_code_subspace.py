"""SAME CODE, OFFSET / RESCALED -- OR RE-MAPPED? The post-stroke lick-direction code vs the pre-stroke one (Priya's
item 2, 10-29: "does the post-stroke lick-direction code use the same activity patterns as pre-stroke -- same code
offset / rescaled, or re-mapped?").

FEATURES. Exactly the frozen lick decoders' basis (iti_lick_order / iti_where_decoder.area_basis): Allen areas WITH the
ALM / tjM1 split (+ MOs_rest / MOp_rest), lags 0..0.2 s after the 3-D lick peak averaged, each feature divided by its
frame-level SD in that session (iti_licks.area_sd). Target y = side_ml (3-D lateral tongue-tip position, mm; + =
CONTRALESIONAL). Spout = the cued target (cued) / the spout in place (ITI), canonical order (iti_first_lick.POS).
LICK SETS: cued_first = the first post-cue lick of each trial (lick_idx 0); iti_all = every ITI lick (the primary all-ITI
set of iti_lick_order); iti_far1 = ITI licks >= 1 s after the response-window end and >= 1 s before the next spout move
(iti_timing_audit.csv, the iti_lick_order 'far1' rule: spout docked). Every quantity is computed per set.
MATCHED LICKS (primary): post-stroke licks kept only if their side_ml lies inside the animal's PRE 2.5-97.5 percentile for
that spout (lick_balance_matched_error's rule, on side_ml), so a range difference cannot fake a re-map. 'all' rows = no
matching (secondary).

(1) CODING AXIS / SUBSPACE, per animal x set x epoch.
  ENCODING AXIS b = per-feature OLS slope of activity on y with session fixed effects (X and y centred within session):
    activity change per mm of lateral lick position (= the PLS-1 weight direction). cos(b_post, b_pre) = same pattern?
    GAIN = b_post . b_pre / |b_pre|^2 (post slope ALONG the pre axis, 1 = unchanged) -> rescaling.
  RIDGE AXIS = ridge weights on the same centred features (alpha chosen by RidgeCV on the animal's pre licks, then held
    fixed), cosine pre vs post (a decoder-side check; the encoding axis is the primary 'which activity pattern').
  CLASS-MEAN SUBSPACE = top-2 PCs of the 6 spout means (session-centred features); overlap = mean cos^2 of the two
    principal angles (1 = same plane, chance ~ 2 / n_features).
  CEILING (pre-vs-pre, sample-matched): for each post epoch, R_CEIL draws of k = min(n_post_sessions, n_pre - 2) random
    pre sessions as a pseudo-post (licks subsampled to the post epoch's matched lick count) vs the remaining pre
    sessions; the REAL comparison uses the SAME draws' reference sessions, so reference size is matched exactly.
  FLOOR: post y permuted within session (R_CEIL draws) vs the full pre reference.
  NORMALISED = (real - floor) / (ceiling - floor) (1 = as similar as two pre sets; 0 = chance). Per animal: one-sided
    p = share of ceiling draws <= real.
  SESSION LEVEL (pooled CIs): per session, its axis from N_SUB licks (R_SUB subsamples, mean) vs the pre axis (pre
    sessions: leave-that-session-out = the session-level ceiling); change vs pre by analysis_kit.boot_delta (paired
    animals > sessions). Same for the GAIN along the pre axis and the shuffle floor.
  DRIFT (scripts/chronic_drift_vs_reorg.py logic): pairwise cos between per-session axes (same N_SUB subsampling) vs
    |day gap|; the PRE pairs give the animal's drift line c(gap) = c0 + m * gap; per session the residual vs that line
    (post sessions vs every pre session at their gap; pre sessions vs the other pre sessions). Change in residual vs pre
    (boot_delta) = the post change BEYOND what the same elapsed time produces pre-stroke. Post gaps exceed the pre gaps
    -> the line is extrapolated (pre gap range reported; the August pre span is only ~8 days, so this line is weak).
  JUNE REFERENCE (the usable drift floor; iti_june_ref.py, Priya 10-27): the June 6-8 pre-stroke sessions (~60 days
    before the August pre, about the chronic gap, no lesion between; never trained on) are run through EVERY comparison
    as epoch 'june' exactly like a post epoch (matched to the August pre range, same ceiling draws, same transfer
    directions). Post vs June = change beyond ~2 months of no-stroke drift (June also includes learning -> conservative).
(2) TRANSFER ASYMMETRY (session level; decoded vs actual per test session: r, OLS slope, intercept at y = 0 mm,
  + = read contralesional).
  pre->pre   frozen pre decoder's leave-one-pre-session-out siblings, on pre sessions
  pre->post  the FROZEN pre decoder (RULE 10: frozen_models.load_or_fit, kind 'lick_code', align = the set; the
             iti_lick_order recipe -- StandardScaler + RidgeCV(logspace(-1, 5, 13)) on uncentred features, all pre
             licks of the set) on post sessions
  post->pre  a decoder trained on the post epoch's matched licks (all its sessions), tested on every pre session --
             POST-STROKE-TRAINED, NOT a reference: refit in memory, NOT frozen (rule 10 covers pre-stroke references)
  post->post within-epoch leave-one-session-out (one-session epochs: 5 contiguous time folds), in memory, not frozen
  Re-map = low transfer BOTH ways with intact within-epoch decoding; offset / gain change = high r with a shifted
  intercept / slope. Transfer ratio = r(transfer) / r(within) per animal.
(3) WHERE: per-feature decomposition of the axis change. ||u_post - u_pre||^2 = 2 (1 - cos) splits exactly into one
  term per feature (u = unit axes); EXCESS = real term - mean ceiling term (sums to 2 (cos_ceiling - cos_real)). Summed
  per area (both hemispheres) and per hemisphere; plus hemisphere- and group-restricted axes (orofacial = ALM, tjM1,
  SSp-m, SSp-n, SSs vs the rest; left = ipsilesional, right = contralesional) with their own ceilings.
Per-animal tables FIRST (rule 8), pooled after. Per-session loads: analysis_kit.fan_sessions (<= 4 workers).
Outputs <OUT_W>/lick_code_subspace/*.csv (+ licks_cache.pkl, figdata/*.pkl); figures <OUT_W>/figures/
lick_code_subspace_{axes,transfer,drift,areas}.png.
ENLv LICKS (Priya 10-29): LICK_CODE_FAMILY=enlv -> sets enlv_on1 (primary) / enlv_bout (bout starts), epochs acute /
subacute / chronic (no June ENLv), every output file name suffixed _enlv (see FAMILY below).
Usage: python scripts/lick_code_subspace.py [--jobs 4] [--reload]      (compute + figures)
       LICK_CODE_FAMILY=enlv python scripts/lick_code_subspace.py [--jobs 4]   (ENLv licks)
       python scripts/lick_code_subspace.py --plot-only                (redraw from figdata)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["june", "acute", "subacute", "chronic"]     # june = no-stroke reference ~60 d before the August pre (drift)
EPS = ["pre", *EP]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
SETS = {"cued_first": "cued first post-cue lick", "iti_all": "ITI licks (all)",
        "iti_far1": "ITI licks, far1 (spout docked)"}
SET_COL = {"cued_first": "#c0392b", "iti_all": "#1f6fb4", "iti_far1": "#7fb2dc"}
ORO = ("ALM", "tjM1", "SSp-m", "SSp-n", "SSs")
Q = (2.5, 97.5)
R_CEIL = 100
R_SUB = 20
MIN_SESS_LICKS = 30
K_SUB = 2
OUT = il.OUT_W / "lick_code_subspace"
FIGD = OUT / "figdata"
FIG = il.OUT_W / "figures"
ALPHAS = np.logspace(-1, 5, 13)
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
# LICK FAMILY (env LICK_CODE_FAMILY, read at import so spawned workers inherit it). "" = cued + ITI (the published run,
# file names unchanged). "enlv" = ENLv licks (Priya 10-29; ENL-violation: cam4 pre_cue licks, spout stationary at the
# trial's position, no cue / reward; NEVER called ITI): enlv_on1 = PRIMARY (onset >= 1 s after spout arrival, lick window
# ends before the cue, lick_enlv_balance_offset.select 'on1'), enlv_bout = its bout starts ('on1_clean': clean_start,
# >= 1 s since any lick). Same features (lick_enlv_extract Xdec = the 0..0.2 s decoder window, area_features / area SD),
# y = 3-D side_ml, spout = the trial's position. No June ENLv extraction -> epochs acute / subacute / chronic only.
# Every output file gets the suffix _enlv (nothing of the cued / ITI run is overwritten); frozen decoders kind
# 'lick_code', align = the set name.
FAMILY = os.environ.get("LICK_CODE_FAMILY", "")
if FAMILY not in ("", "enlv"):
    raise SystemExit(f"LICK_CODE_FAMILY={FAMILY!r}: use '' or 'enlv'")
SFX = f"_{FAMILY}" if FAMILY else ""
if FAMILY == "enlv":
    EP = ["acute", "subacute", "chronic"]
    EPS = ["pre", *EP]
    SETS = {"enlv_on1": "ENLv licks (onset >= 1 s after arrival)",
            "enlv_bout": "ENLv bout starts (>= 1 s since any lick)"}
    SET_COL = {"enlv_on1": "#7570b3", "enlv_bout": "#8c2d04"}
    ENLV_VARIANT = {"enlv_on1": "on1", "enlv_bout": "on1_clean"}
CACHE = OUT / f"licks_cache{SFX}.pkl"


def W(s: str, n: int = 95) -> str:
    return "\n".join(textwrap.fill(p, n) for p in s.split("\n"))


# ------------------------------------------------------------------------------------------------ data
_AUD = {}


JUNE_DIR = il.OUT / "june_ref" / il.SESS_NAME


def _far1_june(label: str, t: np.ndarray) -> np.ndarray:
    """far1 for a June reference session (not in iti_timing_audit.csv): the audit's since_stop / to_strobe rule
    (iti_lick_timing_audit.one), recomputed here read-only."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_on_s", "daq_peak_s"])
    T = il._behaviour(label).trials
    m = pd.Series(A.daq_on_s.to_numpy(float), index=A.daq_peak_s.to_numpy(float))
    to = m[~m.index.duplicated()].reindex(np.asarray(t, float)).to_numpy(float)
    stop = T.stop_s.to_numpy(float) if "stop_s" in T else T.cue_s.to_numpy(float) + 3.5
    stop = np.sort(np.where(np.isfinite(stop), stop, T.cue_s.to_numpy(float) + 3.5))
    j = np.searchsorted(stop, to, side="right") - 1
    since_stop = np.where(j >= 0, to - stop[np.maximum(j, 0)], np.inf)
    ev = np.sort(T.strobe_s.to_numpy(float)[np.isfinite(T.strobe_s.to_numpy(float))])
    k = np.searchsorted(ev, to)
    to_strobe = np.where(k < len(ev), ev[np.minimum(k, len(ev) - 1)] - to, np.inf)
    return (since_stop >= 1.0) & (to_strobe >= 1.0)


def _far1(label: str, t: np.ndarray) -> np.ndarray:
    if "A" not in _AUD:
        A = pd.read_csv(il.OUT / "iti_timing_audit.csv").replace(np.inf, 1e9)
        _AUD["A"] = A.set_index(["label", "t"])
    A = _AUD["A"].loc[label].reindex(np.asarray(t, float))
    return ((A.since_stop >= 1.0) & (A.to_strobe >= 1.0)).to_numpy()


def _session(item):
    """(label, sess_dir) -> per-lick rows of every set + z-scored features (module-level for fan_sessions). June
    reference sessions (sess_dir = JUNE_DIR) get epoch 'june' (a no-stroke ~60-day-earlier reference, never trained on)."""
    import iti_first_lick as fl
    import iti_lick_order as lo
    import iti_where_decoder as wd
    from wfield_local import orofacial_cohort as oc
    from wfield_local.epochs import days_since_stroke
    x, sdir = item
    sdir = Path(sdir) if sdir else None
    names, SD = wd.area_meta(sdir, "split")
    _x, rec = wd._load_one((x, "areas_split", str(sdir) if sdir else ""))
    z = il._load(x, sdir)
    if sdir is not None:
        rec["epoch"] = "june"
    sd = SD.loc[x, names].to_numpy(float)
    day = days_since_stroke(x)
    out_rows, out_X = [], []
    for st, kt in (("trial", "tr_t"), ("iti", "iti_t")):
        X, C = rec[st]
        t = z[kt].astype(float)
        idx = lo._bout_idx(x, t) if st == "iti" else fl._lick_idx(x, t)
        pos = np.array([q if q in POS or "_" not in q else oc.pos_key(q, "L") for q in rec[f"{st}_pos"].astype(str)])
        ok = np.isfinite(C[:, 0]) & np.isfinite(idx) & np.isin(pos, POS) & np.isfinite(X).all(1)
        if st == "trial":
            sets = {"cued_first": ok & (idx == 0)}
        else:
            sets = {"iti_all": ok, "iti_far1": ok & (_far1_june(x, t) if sdir is not None else _far1(x, t))}
        for name, m in sets.items():
            if m.sum() < 1:
                continue
            out_rows.append(pd.DataFrame({"set": name, "animal": rec["animal"], "label": x, "epoch": rec["epoch"],
                                          "day": day if day is not None else np.nan, "pos": pos[m], "order": idx[m],
                                          "t": t[m], "y": C[m, 0]}))
            out_X.append(X[m] / sd[None, :])
    return pd.concat(out_rows, ignore_index=True), np.vstack(out_X), names


def _enlv_session(label: str):
    """label -> (lick_uid, z-scored decoder features (n, F), names): lick_enlv_extract sessions/<label>.npz Xdec (raw,
    0..0.2 s after the lick peak) in the split area basis / the session's frame SD -- exactly lick_enlv_balance_offset.table
    (which builds the frozen ENLv readers). Module-level for fan_sessions."""
    import iti_where_decoder as wd
    import lick_enlv_extract as ex
    names, SD = wd.area_meta(basis="split")
    z = np.load(ex.SESS / f"{label}.npz")
    cols = [str(c) for c in z["cols"]]
    Xd = il.area_features(z["Xdec"].astype(float), cols, label) / SD.loc[label, names].to_numpy(float)[None, :]
    return z["lick_uid"].astype(int), Xd, list(names)


def _load_enlv(jobs: int):
    """ENLv lick sets in the load() format (L rows: set, animal, label, epoch, day, pos, order, t, y; X features)."""
    import lick_enlv_balance_offset as bo

    from wfield_local.epochs import days_since_stroke
    T = pd.read_csv(bo.OUT / "enlv_table.csv.gz")
    for c in ("first", "clean_start", "contact"):
        T[c] = T[c].astype(bool)
    labs = sorted(T.label.unique())
    res, fails = ak.fan_sessions(labs, _enlv_session, jobs=jobs, key=lambda s: s, label="lick_code_subspace enlv load")
    if fails:
        raise RuntimeError(f"failed sessions: {[f[0] for f in fails][:5]} {fails[0][1]}")
    feats = dict(res)
    Ls, Xs, names = [], [], None
    for st, v in ENLV_VARIANT.items():
        m = bo.select(T, v) & np.isfinite(T.act) & T.pos.isin(POS)
        for lab, g in T[m].groupby("label", sort=True):
            uid, Xd, names = feats[lab]
            row = pd.Series(np.arange(len(uid)), index=uid).reindex(g.lick_uid.to_numpy(int)).to_numpy()
            if np.isnan(row.astype(float)).any():
                raise RuntimeError(f"{lab}: ENLv lick_uid missing from sessions/{lab}.npz")
            Xg = Xd[row.astype(int)]
            ok = np.isfinite(Xg).all(1)
            g = g[ok]
            day = days_since_stroke(lab)
            Ls.append(pd.DataFrame({"set": st, "animal": g.animal.to_numpy(), "label": lab, "epoch": g.epoch.to_numpy(),
                                    "day": day if day is not None else np.nan, "pos": g.pos.to_numpy(),
                                    "order": g["rank"].to_numpy(float), "t": g.peak.to_numpy(float),
                                    "y": g.act.to_numpy(float)}))
            Xs.append(Xg[ok])
    L = pd.concat(Ls, ignore_index=True)
    X = np.vstack(Xs)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"L": L, "X": X, "names": list(names)}, f)
    return L, X, list(names)


def load(jobs: int, reload: bool = False):
    if CACHE.exists() and not reload:
        with open(CACHE, "rb") as f:
            c = pickle.load(f)
        return c["L"], c["X"], c["names"]
    if FAMILY == "enlv":
        return _load_enlv(jobs)
    import iti_where_decoder as wd
    labs = [(x, "") for x in wd.sess_labels(None)] + [(x, str(JUNE_DIR)) for x in wd.sess_labels(JUNE_DIR)]
    res, fails = ak.fan_sessions(labs, _session, jobs=jobs, key=lambda s: s[0], label="lick_code_subspace load")
    if fails:
        raise RuntimeError(f"failed sessions: {[f[0] for f in fails][:5]} {fails[0][1]}")
    Ls, Xs, names = [], [], None
    for _lab, (L, X, nm) in res:
        Ls.append(L)
        Xs.append(X)
        names = nm
    L = pd.concat(Ls, ignore_index=True)
    X = np.vstack(Xs)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"L": L, "X": X, "names": list(names)}, f)
    return L, X, list(names)


def add_match(L: pd.DataFrame) -> pd.DataFrame:
    """matched = pre licks always; post licks inside the animal's pre 2.5-97.5 percentile of side_ml for their spout."""
    L = L.copy()
    L["matched"] = True
    for (_st, _a), g in L.groupby(["set", "animal"]):
        pre = g[g.epoch == "pre"]
        lo_ = g.pos.map(pre.groupby("pos").y.quantile(Q[0] / 100))
        hi_ = g.pos.map(pre.groupby("pos").y.quantile(Q[1] / 100))
        inside = (g.y >= lo_) & (g.y <= hi_)
        L.loc[g.index, "matched"] = (g.epoch == "pre") | inside
    return L


# ------------------------------------------------------------------------------------------------ axes
def _centre(X, y, sess):
    Xc, yc = X.copy(), y.astype(float).copy()
    for s in np.unique(sess):
        m = sess == s
        Xc[m] -= Xc[m].mean(0)
        yc[m] -= yc[m].mean()
    return Xc, yc


def enc_axis(Xc, yc):
    d = (yc ** 2).sum()
    return (Xc * yc[:, None]).sum(0) / d if d > 0 else np.full(Xc.shape[1], np.nan)


def ridge_axis(Xc, yc, alpha):
    F = Xc.shape[1]
    return np.linalg.solve(Xc.T @ Xc + alpha * np.eye(F), Xc.T @ yc)


def class_sub(Xc, pos, k=K_SUB):
    M = np.vstack([Xc[pos == p].mean(0) for p in POS if (pos == p).sum() >= 3])
    if len(M) < k + 1:
        return None
    M = M - M.mean(0)
    _u, _s, vt = np.linalg.svd(M, full_matrices=False)
    return vt[:k].T


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def overlap(U, V):
    if U is None or V is None:
        return np.nan
    return float((np.linalg.svd(U.T @ V, compute_uv=False) ** 2).mean())


def unit(a):
    n = np.linalg.norm(a)
    return a / n if n > 0 else a * np.nan


class Cell:
    """One animal x set: pre and post licks, centred per session."""

    def __init__(self, L: pd.DataFrame, X: np.ndarray, matched: bool):
        L = L.reset_index(drop=True)
        keep = L.matched.to_numpy() if matched else np.ones(len(L), bool)      # pre rows are always matched
        self.L = L[keep].reset_index(drop=True)
        self.X = X[keep]
        self.keep = np.ones(len(self.L), bool)
        self.sess = self.L.label.to_numpy()
        self.Xc, self.yc = _centre(self.X, self.L.y.to_numpy(float), self.sess)
        self.pos = self.L.pos.to_numpy()
        self.pre = sorted(self.L[self.L.epoch == "pre"].label.unique())

    def rows(self, labels):
        return np.isin(self.sess, list(labels)) & self.keep


def epoch_level(c: Cell, ep: str, alpha: float, rng, names) -> tuple[dict, dict]:
    post = sorted(c.L[c.L.epoch == ep].label.unique())
    mpost = c.rows(post)
    n_post = int(mpost.sum())
    if not post or n_post < MIN_SESS_LICKS or len(c.pre) < 3:
        return {}, {}
    Xp, yp, pp, sp = c.Xc[mpost], c.yc[mpost], c.pos[mpost], c.sess[mpost]
    b_post, r_post, U_post = enc_axis(Xp, yp), ridge_axis(Xp, yp, alpha), class_sub(Xp, pp)
    k = min(len(post), len(c.pre) - 2)
    hemi = {h: np.array([n.endswith(f"_{h}") for n in names]) for h in il.HEMIS}
    oro = np.array([n.rsplit("_", 1)[0] in ORO for n in names])
    groups = {"left": hemi["left"], "right": hemi["right"], "oro": oro, "nonoro": ~oro,
              "oro_left": oro & hemi["left"], "oro_right": oro & hemi["right"],
              "nonoro_left": ~oro & hemi["left"], "nonoro_right": ~oro & hemi["right"]}
    real = {"enc": [], "ridge": [], "sub": [], "gain": [], **{f"enc_{g}": [] for g in groups},
            **{f"gain_{g}": [] for g in groups}}
    ceil = {kk: [] for kk in real}
    term_real, term_ceil = [], []
    for _ in range(R_CEIL):
        ps = list(rng.choice(c.pre, k, replace=False))
        ref = [q for q in c.pre if q not in ps]
        mr, mp = c.rows(ref), c.rows(ps)
        Xr, yr, pr = c.Xc[mr], c.yc[mr], c.pos[mr]
        b_ref, r_ref, U_ref = enc_axis(Xr, yr), ridge_axis(Xr, yr, alpha), class_sub(Xr, pr)
        ip = np.flatnonzero(mp)
        if len(ip) > n_post:
            ip = np.sort(rng.choice(ip, n_post, replace=False))
        Xq, yq, pq = c.Xc[ip], c.yc[ip], c.pos[ip]
        # pseudo-post re-centred on its own (subsampled) session means
        Xq, yq = _centre(Xq, yq, c.sess[ip])
        b_q, r_q, U_q = enc_axis(Xq, yq), ridge_axis(Xq, yq, alpha), class_sub(Xq, pq)
        for dst, (bb, rr, UU) in ((real, (b_post, r_post, U_post)), (ceil, (b_q, r_q, U_q))):
            dst["enc"].append(cos(bb, b_ref))
            dst["ridge"].append(cos(rr, r_ref))
            dst["sub"].append(overlap(UU, U_ref))
            dst["gain"].append(float(bb @ b_ref / (b_ref @ b_ref)))
            for g, m in groups.items():
                dst[f"enc_{g}"].append(cos(bb[m], b_ref[m]))
                dst[f"gain_{g}"].append(float(bb[m] @ b_ref[m] / (b_ref[m] @ b_ref[m])))
        term_real.append((unit(b_post) - unit(b_ref)) ** 2)
        term_ceil.append((unit(b_q) - unit(b_ref)) ** 2)
    # floor: post y permuted within session vs the full pre reference
    mr = c.rows(c.pre)
    b_ref_all, r_ref_all, U_ref_all = enc_axis(c.Xc[mr], c.yc[mr]), ridge_axis(c.Xc[mr], c.yc[mr], alpha), class_sub(
        c.Xc[mr], c.pos[mr])
    flo = {"enc": [], "ridge": [], "sub": [], "gain": []}
    for _ in range(R_CEIL):
        ys, ps_ = yp.copy(), pp.copy()
        for s in np.unique(sp):
            m = np.flatnonzero(sp == s)
            perm = rng.permutation(m)
            ys[m], ps_[m] = yp[perm], pp[perm]
        bs = enc_axis(Xp, ys)
        flo["enc"].append(cos(bs, b_ref_all))
        flo["ridge"].append(cos(ridge_axis(Xp, ys, alpha), r_ref_all))
        flo["sub"].append(overlap(class_sub(Xp, ps_), U_ref_all))
        flo["gain"].append(float(bs @ b_ref_all / (b_ref_all @ b_ref_all)))
    row = {"epoch": ep, "n_post_sessions": len(post), "n_pre_sessions": len(c.pre), "k_pseudo": k, "n_post_licks": n_post,
           "n_post_all": int(c.L.epoch.eq(ep).sum())}
    for kk in real:
        rv, cv = np.array(real[kk], float), np.array(ceil[kk], float)
        row[f"{kk}_real"] = np.nanmean(rv)
        row[f"{kk}_ceil"] = np.nanmean(cv)
        row[f"{kk}_ceil_lo"] = np.nanpercentile(cv, 5) if np.isfinite(cv).any() else np.nan
        row[f"{kk}_p_below"] = float(np.mean(cv <= np.nanmean(rv))) if np.isfinite(cv).any() else np.nan
        if kk in flo:
            row[f"{kk}_floor"] = float(np.nanmean(flo[kk]))
            den = row[f"{kk}_ceil"] - row[f"{kk}_floor"]
            row[f"{kk}_norm"] = (row[f"{kk}_real"] - row[f"{kk}_floor"]) / den if abs(den) > 1e-9 else np.nan
    terms = {"real": np.mean(term_real, 0), "ceil": np.mean(term_ceil, 0)}
    return row, terms


def session_axes(c: Cell, n_sub: int, rng) -> dict:
    """Per session: R_SUB subsampled encoding axes (and a within-session shuffled copy). {label: (axes, shuf_axes)}."""
    out = {}
    for s in sorted(np.unique(c.sess)):
        idx = np.flatnonzero(c.rows([s]))
        if len(idx) < n_sub:
            continue
        A, S = [], []
        for _ in range(R_SUB):
            ii = np.sort(rng.choice(idx, n_sub, replace=False))
            Xs, ys = _centre(c.Xc[ii], c.yc[ii], c.sess[ii])
            A.append(enc_axis(Xs, ys))
            S.append(enc_axis(Xs, rng.permutation(ys)))
        out[s] = (np.array(A), np.array(S))
    return out


def session_level(c: Cell, SA: dict, days: dict) -> pd.DataFrame:
    rows = []
    for s, (A, S) in SA.items():
        ref = [q for q in c.pre if q != s]
        mr = c.rows(ref)
        b_ref = enc_axis(c.Xc[mr], c.yc[mr])
        ep = c.L.loc[c.sess == s, "epoch"].iloc[0]
        rows.append({"label": s, "epoch": ep, "day": days[s], "n_licks": int(c.rows([s]).sum()),
                     "cos": float(np.nanmean([cos(a, b_ref) for a in A])),
                     "cos_floor": float(np.nanmean([cos(a, b_ref) for a in S])),
                     "gain": float(np.nanmean([a @ b_ref / (b_ref @ b_ref) for a in A]))})
    T = pd.DataFrame(rows)
    # drift: pairwise per-session cosines vs day gap
    pre = [s for s in c.pre if s in SA]
    pc = {}
    for s in SA:
        for q in pre:
            if q != s:
                pc[(s, q)] = float(np.nanmean([cos(a, b) for a, b in zip(SA[s][0], SA[q][0])]))
    gp = [(abs(days[s] - days[q]), pc[(s, q)]) for s in pre for q in pre if s < q]
    T["drift_resid"] = np.nan
    if len(gp) >= 3:
        g, v = np.array(gp).T
        m, c0 = np.polyfit(g, v, 1)
        for i, s in enumerate(T.label):
            res = [pc[(s, q)] - (c0 + m * abs(days[s] - days[q])) for q in pre if q != s]
            T.loc[i, "drift_resid"] = np.mean(res) if res else np.nan
        T.attrs["drift"] = {"m": m, "c0": c0, "gap_max_pre": float(g.max()), "pairs_pre": gp,
                            "pairs_post": [(abs(days[s] - days[q]), pc[(s, q)], T.set_index("label").epoch[s])
                                           for s in SA if s not in pre for q in pre]}
    return T


# ------------------------------------------------------------------------------------------------ transfer (2)
def _fit(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    return sc, RidgeCV(alphas=ALPHAS).fit(sc.transform(X), y)


def _pred(m, X):
    return m[1].predict(m[0].transform(X))


def _score(dec, act) -> dict:
    if len(act) < 10 or np.std(act) == 0 or np.std(dec) == 0:
        return {"r": np.nan, "slope": np.nan, "intercept": np.nan, "offset": np.nan}
    sl, ic = np.polyfit(act, dec, 1)
    return {"r": float(np.corrcoef(act, dec)[0, 1]), "slope": float(sl), "intercept": float(ic),
            "offset": float(np.mean(dec - act))}


def frozen_pre(a: str, st: str, X, y, sess) -> dict:
    """RULE 10: the animal's pre decoder for this lick set (+ LOSO siblings) via frozen_models.load_or_fit."""
    from wfield_local import frozen_models as fm
    labs = sorted(set(sess))
    dig = hashlib.sha1(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()
                       + "|".join(sess).encode()).hexdigest()
    spec = fm.make_spec(a, "lick_code", align=st, source=f"areas_split_r{il.ROI_VERSION}", train_labels=labs,
                        post_s=0.2, n_features=X.shape[1], train_data={"xy_sha1": dig})  # training-data fingerprint (FREEZE_VERSION 2, rebuild 11-03)

    def fit():
        return {"full": _fit(X, y), "loso": {s: _fit(X[sess != s], y[sess != s]) for s in labs}, "xy_sha1": dig}
    P, status = fm.load_or_fit(spec, fit, meta={"script": "lick_code_subspace.py", "n_licks": int(len(y)), "set": st})
    print(f"[frozen] {a} lick_code {st}: {status}", flush=True)
    if P.get("xy_sha1") != dig:
        print(f"[frozen] !! {a} {st}: stored model trained on DIFFERENT lick data than today's selection -- stored "
              "reference used", flush=True)
    return P


def transfer(L: pd.DataFrame, X: np.ndarray, a: str, st: str) -> pd.DataFrame:
    """L / X = the animal x set's licks, UNFILTERED (matched flag carried); features uncentred (decoder recipe)."""
    L = L.reset_index(drop=True)

    class _C:
        pass
    c = _C()
    c.L, c.X, c.sess = L, X, L.label.to_numpy()
    c.pre = sorted(L[L.epoch == "pre"].label.unique())
    y = L.y.to_numpy(float)
    mpre = (L.epoch == "pre").to_numpy()
    P = frozen_pre(a, st, c.X[mpre], y[mpre], c.sess[mpre])
    rows = []

    def add(direction, ep_train, s, dec, act, matched):
        rows.append({"direction": direction, "train_epoch": ep_train, "label": s,
                     "test_epoch": L.loc[c.sess == s, "epoch"].iloc[0], "licks": "matched" if matched else "all",
                     "n": len(act), **_score(dec, act)})
    for s in c.pre:
        m = c.sess == s
        dec = _pred(P["loso"][s], c.X[m])
        for mt in (True, False):
            add("pre->pre", "pre", s, dec, y[m], mt)
    for ep in EP:
        post = sorted(L[L.epoch == ep].label.unique())
        if not post:
            continue
        for s in post:
            m = c.sess == s
            for mt in (True, False):
                mm = m & (L.matched.to_numpy() if mt else True)
                if mm.sum() >= 10:
                    add("pre->post", "pre", s, _pred(P["full"], c.X[mm]), y[mm], mt)
        for mt in (True, False):
            mtr = np.isin(c.sess, post) & (L.matched.to_numpy() if mt else True)
            if mtr.sum() < MIN_SESS_LICKS:
                continue
            M = _fit(c.X[mtr], y[mtr])                       # post-trained: NOT a reference, not frozen
            for s in c.pre:
                m = c.sess == s
                add("post->pre", ep, s, _pred(M, c.X[m]), y[m], mt)
            if len(post) >= 2:
                for s in post:
                    tr, te = mtr & (c.sess != s), mtr & (c.sess == s)
                    if te.sum() >= 10 and tr.sum() >= MIN_SESS_LICKS:
                        add("post->post", ep, s, _pred(_fit(c.X[tr], y[tr]), c.X[te]), y[te], mt)
            else:
                idx = np.flatnonzero(mtr)
                idx = idx[np.argsort(L.t.to_numpy()[idx])]
                dec = np.full(len(idx), np.nan)
                for te in np.array_split(np.arange(len(idx)), 5):
                    tr = np.setdiff1d(np.arange(len(idx)), te)
                    dec[te] = _pred(_fit(c.X[idx[tr]], y[idx[tr]]), c.X[idx[te]])
                add("post->post", ep, post[0], dec, y[idx], mt)
    T = pd.DataFrame(rows)
    T.insert(0, "animal", a)
    T.insert(0, "set", st)
    return T


# ------------------------------------------------------------------------------------------------ main compute
def compute(L, X, names):
    from sklearn.linear_model import RidgeCV
    rng = np.random.default_rng(0)
    L = add_match(L)
    EPL, SESL, TERMS, TR, DRIFT, RET = [], [], [], [], {}, []
    for st in SETS:
        Ls = L[L.set == st]
        cnt = Ls[Ls.matched].groupby("label").size()
        n_sub = int(max(MIN_SESS_LICKS, np.percentile(cnt, 10)))
        print(f"== {st}: N_SUB {n_sub} licks / session (10th pct of matched counts)", flush=True)
        for a in AN:
            g = Ls[Ls.animal == a]
            if g.empty:
                continue
            Xg = X[g.index.to_numpy()]
            for ep in EPS:
                h = g[g.epoch == ep]
                if len(h):
                    RET.append({"set": st, "animal": a, "epoch": ep, "n_all": len(h), "frac_matched": h.matched.mean()})
            for mt in (True, False):
                c = Cell(g, Xg, mt)
                mr = c.rows(c.pre)
                alpha = float(RidgeCV(alphas=ALPHAS).fit(c.Xc[mr], c.yc[mr]).alpha_)
                for ep in EP:
                    row, terms = epoch_level(c, ep, alpha, rng, names)
                    if row:
                        EPL.append({"set": st, "animal": a, "licks": "matched" if mt else "all", "alpha": alpha, **row})
                        if mt:
                            TERMS.append({"set": st, "animal": a, "epoch": ep, **{f"real_{n}": v for n, v in zip(
                                names, terms["real"])}, **{f"ceil_{n}": v for n, v in zip(names, terms["ceil"])}})
                if mt:
                    days = g.groupby("label").day.first().to_dict()
                    SA = session_axes(c, n_sub, rng)
                    T = session_level(c, SA, days)
                    T.insert(0, "animal", a)
                    T.insert(0, "set", st)
                    SESL.append(T)
                    if "drift" in T.attrs:
                        DRIFT[(st, a)] = T.attrs["drift"]
            TR.append(transfer(g, Xg, a, st))
            print(f"   {st} {a} done", flush=True)
    return (pd.DataFrame(EPL), pd.concat(SESL, ignore_index=True), pd.DataFrame(TERMS), pd.concat(TR, ignore_index=True),
            DRIFT, pd.DataFrame(RET))


def pooled_session(S: pd.DataFrame, cols, by=("set",)) -> pd.DataFrame:
    """change vs pre (pre = leave-one-session-out ceiling at session level) per set x epoch: boot_delta + n same sign."""
    out = []
    for keys, g in S.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for col in cols:
            pre = {a: h[col].dropna().to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            pre = {a: v for a, v in pre.items() if len(v)}
            pm = {a: v.mean() for a, v in pre.items()}
            for ep in EP:
                post = {a: h[col].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in pre}
                post = {a: v for a, v in post.items() if len(v)}
                if not post:
                    continue
                ci = ak.boot_delta(post, pre, np.random.default_rng(0))
                d = {a: post[a].mean() - pm[a] for a in post}
                pt = ci.point if ci is not None else np.mean(list(d.values()))
                out.append({**dict(zip(by, keys)), "measure": col, "epoch": ep,
                            "pre_level": np.mean(list(pm.values())), "post_level": np.mean([post[a].mean() for a in post]),
                            "delta": pt, "lo": ci.lo if ci is not None else np.nan, "hi": ci.hi if ci is not None else np.nan,
                            "n_same_sign": int(sum(np.sign(v) == np.sign(pt) for v in d.values())), "n": len(d),
                            **{f"d_{a}": d.get(a, np.nan) for a in AN}})
    return pd.DataFrame(out)


def transfer_tests(T: pd.DataFrame) -> pd.DataFrame:
    """per set x licks x epoch x metric: pre->post vs pre->pre and post->pre vs post->post (boot_delta)."""
    out = []
    for (st, lk), g in T.groupby(["set", "licks"]):
        for met in ("r", "slope", "intercept", "offset"):
            def dd(direction, ep=None, test=None, g=g, met=met):
                h = g[(g.direction == direction)]
                if ep is not None:
                    h = h[h.train_epoch == ep]
                if test is not None:
                    h = h[h.test_epoch == test]
                return {a: v[met].dropna().to_numpy() for a, v in h.groupby("animal") if v[met].notna().any()}
            pp = dd("pre->pre")
            for ep in EP:
                for name, post, ref in (("pre->post minus pre->pre", dd("pre->post", test=ep), pp),
                                        ("post->pre minus post->post", dd("post->pre", ep=ep), dd("post->post", ep=ep)),
                                        ("post->pre minus pre->pre", dd("post->pre", ep=ep), pp)):
                    ani = sorted(set(post) & set(ref))
                    if not ani:
                        continue
                    ci = ak.boot_delta({a: post[a] for a in ani}, {a: ref[a] for a in ani}, np.random.default_rng(0))
                    d = {a: post[a].mean() - ref[a].mean() for a in ani}
                    pt = ci.point if ci is not None else np.mean(list(d.values()))
                    out.append({"set": st, "licks": lk, "metric": met, "epoch": ep, "contrast": name,
                                "level_a": np.mean([post[a].mean() for a in ani]), "level_b": np.mean([ref[a].mean() for a in ani]),
                                "delta": pt, "lo": ci.lo if ci is not None else np.nan,
                                "hi": ci.hi if ci is not None else np.nan,
                                "n_same_sign": int(sum(np.sign(v) == np.sign(pt) for v in d.values())), "n": len(d),
                                **{f"d_{a}": d.get(a, np.nan) for a in AN}})
    return pd.DataFrame(out)


def transfer_animals(T: pd.DataFrame) -> pd.DataFrame:
    """per set x licks x animal x epoch: mean r / slope / intercept per direction + the two transfer ratios."""
    rows = []
    for (st, lk, a), g in T.groupby(["set", "licks", "animal"]):
        pp = g[g.direction == "pre->pre"]
        for ep in EP:
            r = {"set": st, "licks": lk, "animal": a, "epoch": ep}
            for dname, h in (("prepre", pp), ("prepost", g[(g.direction == "pre->post") & (g.test_epoch == ep)]),
                             ("postpre", g[(g.direction == "post->pre") & (g.train_epoch == ep)]),
                             ("postpost", g[(g.direction == "post->post") & (g.train_epoch == ep)])):
                for met in ("r", "slope", "intercept"):
                    r[f"{met}_{dname}"] = h[met].mean() if len(h) else np.nan
            if np.isfinite(r["r_prepost"]):
                r["ratio_pre_to_post"] = r["r_prepost"] / r["r_prepre"]
                r["ratio_post_to_pre"] = r["r_postpre"] / r["r_postpost"]
                rows.append(r)
    return pd.DataFrame(rows)


def area_tables(TM: pd.DataFrame, names) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, r in TM.iterrows():
        for n in names:
            area, h = n.rsplit("_", 1)
            rows.append({"set": r.set, "animal": r.animal, "epoch": r.epoch, "area": area, "hemi": h, "feature": n,
                         "real": r[f"real_{n}"], "ceil": r[f"ceil_{n}"], "excess": r[f"real_{n}"] - r[f"ceil_{n}"]})
    A = pd.DataFrame(rows)
    P = (A.groupby(["set", "epoch", "feature", "area", "hemi"])
         .agg(excess=("excess", "mean"), n_pos=("excess", lambda v: int((v > 0).sum())), n=("excess", "size"))
         .reset_index())
    return A, P


def june_contrasts() -> pd.DataFrame:
    """POST epoch vs the JUNE no-stroke reference (both scored against the same August pre): is the post change beyond
    ~2 months of no-stroke drift? Session-level axis measures (session_axes.csv) and pre->X transfer metrics
    (transfer_sessions.csv, matched licks), boot_delta(post sessions, June sessions), animal-paired. From the CSVs."""
    S = pd.read_csv(OUT / f"session_axes{SFX}.csv")
    T = pd.read_csv(OUT / f"transfer_sessions{SFX}.csv")
    T = T[(T.licks == "matched") & (T.direction == "pre->post")].rename(columns={"test_epoch": "epoch"})
    out = []
    for src, tab, cols in (("session_axis", S, ["cos", "gain"]), ("pre->X transfer", T, ["r", "slope", "intercept"])):
        for st, g in tab.groupby("set"):
            for col in cols:
                ref = {a: h[col].dropna().to_numpy() for a, h in g[g.epoch == "june"].groupby("animal")}
                ref = {a: v for a, v in ref.items() if len(v)}
                for ep in [e for e in EP if e != "june"]:
                    post = {a: h[col].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in ref}
                    post = {a: v for a, v in post.items() if len(v)}
                    if not post:
                        continue
                    ci = ak.boot_delta(post, ref, np.random.default_rng(0))
                    d = {a: post[a].mean() - ref[a].mean() for a in post}
                    pt = ci.point if ci is not None else np.mean(list(d.values()))
                    out.append({"source": src, "set": st, "measure": col, "epoch": ep,
                                "june_level": np.mean([ref[a].mean() for a in post]),
                                "post_level": np.mean([post[a].mean() for a in post]), "delta_vs_june": pt,
                                "lo": ci.lo if ci is not None else np.nan, "hi": ci.hi if ci is not None else np.nan,
                                "n_same_sign": int(sum(np.sign(v) == np.sign(pt) for v in d.values())), "n": len(d),
                                **{f"d_{a}": d.get(a, np.nan) for a in AN}})
    J = pd.DataFrame(out)
    J.to_csv(OUT / f"vs_june_tests{SFX}.csv", index=False)
    print("\n== POST vs JUNE no-stroke reference (boot_delta, animal-paired)")
    print(J.round(3).to_string(index=False))
    return J


def run(jobs: int, reload: bool):
    t0 = time.time()
    L, X, names = load(jobs, reload)
    print(f"loaded {len(L)} lick rows x {X.shape[1]} features in {time.time() - t0:.0f} s", flush=True)
    E_, S_, TM, TR, DRIFT, RET = compute(L, X, names)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGD.mkdir(parents=True, exist_ok=True)
    RET.to_csv(OUT / f"retained{SFX}.csv", index=False)
    E_.to_csv(OUT / f"epoch_axes_animals{SFX}.csv", index=False)
    S_.to_csv(OUT / f"session_axes{SFX}.csv", index=False)
    TR.to_csv(OUT / f"transfer_sessions{SFX}.csv", index=False)
    TA = transfer_animals(TR)
    TA.to_csv(OUT / f"transfer_animals{SFX}.csv", index=False)
    TT = transfer_tests(TR)
    TT.to_csv(OUT / f"transfer_tests{SFX}.csv", index=False)
    SP = pooled_session(S_, ["cos", "gain", "cos_floor", "drift_resid"])
    SP.to_csv(OUT / f"session_axes_tests{SFX}.csv", index=False)
    A, AP = area_tables(TM, names)
    A.to_csv(OUT / f"area_terms_animals{SFX}.csv", index=False)
    AP.to_csv(OUT / f"area_terms_pooled{SFX}.csv", index=False)
    D = pd.DataFrame([{"set": st, "animal": a, "m_pre": d["m"], "c0_pre": d["c0"], "gap_max_pre": d["gap_max_pre"],
                       "n_pairs_pre": len(d["pairs_pre"])} for (st, a), d in DRIFT.items()])
    D.to_csv(OUT / f"drift_lines{SFX}.csv", index=False)
    with open(FIGD / f"lick_code_subspace{SFX}.pkl", "wb") as f:
        pickle.dump({"E": E_, "S": S_, "SP": SP, "TR": TR, "TA": TA, "TT": TT, "A": A, "AP": AP, "DRIFT": DRIFT,
                     "names": names}, f)
    report(E_, SP, TA, TT, AP, D, RET)
    if "june" in EP:
        june_contrasts()
    figures()
    print(f"total runtime {time.time() - t0:.0f} s", flush=True)


def report(E_, SP, TA, TT, AP, D, RET):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n== retained (fraction of post licks inside the pre range per spout)")
    print(RET.pivot_table(index=["set", "epoch"], columns="animal", values="frac_matched").round(2).to_string())
    m = E_[E_.licks == "matched"]
    for kk in ("enc", "gain", "ridge", "sub", "enc_left", "enc_right", "enc_oro", "enc_nonoro"):
        cols = [f"{kk}_real", f"{kk}_ceil", f"{kk}_ceil_lo", f"{kk}_p_below"] + (
            [f"{kk}_floor", f"{kk}_norm"] if f"{kk}_floor" in m else [])
        print(f"\n== PER ANIMAL epoch-level {kk} (matched licks)")
        print(m[["set", "animal", "epoch", *cols]].round(3).to_string(index=False))
    print("\n== PER ANIMAL transfer (matched licks)")
    print(TA[TA.licks == "matched"].round(3).to_string(index=False))
    print("\n== drift lines (pre pairs)")
    print(D.round(4).to_string(index=False))
    print("\n== POOLED session-level change vs pre (boot_delta)")
    print(SP.round(3).to_string(index=False))
    print("\n== POOLED transfer contrasts (matched)")
    print(TT[TT.licks == "matched"].round(3).to_string(index=False))
    print("\n== area excess (top 8 per set x epoch, mean over animals)")
    for (st, ep), g in AP.groupby(["set", "epoch"]):
        print(st, ep, g.sort_values("excess", ascending=False).head(8)[["feature", "excess", "n_pos"]].round(4)
              .to_string(index=False).replace("\n", " | "))


# ------------------------------------------------------------------------------------------------ figures
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.epoch_figures import EPOCH_GREY as _EG
    EPOCH_GREY = {**_EG, "june": "#6baed6"}
    with open(FIGD / f"lick_code_subspace{SFX}.pkl", "rb") as f:
        F = pickle.load(f)
    E_, TA, DRIFT, AP, S_ = F["E"], F["TA"], F["DRIFT"], F["AP"], F["S"]
    FIG.mkdir(parents=True, exist_ok=True)
    x0 = {ep: i for i, ep in enumerate(EP)}

    # ---- axes figure
    meas = [("enc", "encoding axis cosine (pre vs post)"), ("gain", "gain along the pre axis (1 = unchanged)"),
            ("sub", "6-spout class-mean subspace overlap (k = 2)"), ("ridge", "ridge decoder-weight cosine")]
    fig, axs = plt.subplots(len(meas), len(SETS), figsize=(5 * len(SETS), 3.3 * len(meas)), squeeze=False)
    m = E_[E_.licks == "matched"]
    for j, st in enumerate(SETS):
        for i, (kk, lab) in enumerate(meas):
            ax = axs[i, j]
            g = m[m.set == st]
            for ep in EP:
                h = g[g.epoch == ep]
                if h.empty:
                    continue
                ax.fill_between([x0[ep] - 0.35, x0[ep] + 0.35], h[f"{kk}_ceil_lo"].mean(), h[f"{kk}_ceil"].mean(),
                                color="#bcd8bc", alpha=0.6, lw=0)
                if f"{kk}_floor" in h:
                    ax.hlines(h[f"{kk}_floor"].mean(), x0[ep] - 0.35, x0[ep] + 0.35, color="#b0413e", ls=":", lw=1.5)
                for _, r in h.iterrows():
                    xx = x0[ep] - 0.24 + 0.16 * AN.index(r.animal)
                    ax.plot([xx, xx], [r[f"{kk}_ceil_lo"], r[f"{kk}_ceil"]], color="#6a9f6a", lw=1)
                    ax.plot(xx, r[f"{kk}_real"], MARK[r.animal], color=EPOCH_GREY[ep], mec="k", ms=7)
            ax.set_xticks(range(len(EP)), EP)
            ax.set_title(W(f"{SETS[st]}: {lab}", 55), fontsize=9)
            if kk == "gain":
                ax.axhline(1, color="0.6", lw=0.8)
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], marker=MARK[a], ls="", color="0.5", mec="k", label=a) for a in AN] + [
        Line2D([], [], color="#6a9f6a", label="pre-vs-pre ceiling (mean; line to 5th pct)"),
        Line2D([], [], color="#b0413e", ls=":", label="shuffle floor")]
    fig.legend(handles=hand, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(W("Is the post-stroke lick-direction code the pre-stroke code? Per animal, post epoch vs pre, "
                   "matched licks (post licks inside the pre side_ml range per spout). Green = sample-matched pre-vs-pre "
                   "ceiling (same reference sessions), red = within-session shuffle floor. Encoding axis = activity "
                   "slope per mm of lateral lick position (session fixed effects); gain = post slope along the pre axis.",
                   150), fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(FIG / f"lick_code_subspace_axes{SFX}.png", dpi=130)
    plt.close(fig)

    # ---- transfer figure
    dirs = [("prepre", "pre->pre (LOSO)", "#c9c9c9"), ("prepost", "pre->post (frozen)", "#2c7fb8"),
            ("postpre", "post->pre", "#e08214"), ("postpost", "post->post (within)", "#4d4d4d")]
    fig, axs = plt.subplots(3, len(SETS), figsize=(5 * len(SETS), 10), squeeze=False)
    T = TA[TA.licks == "matched"]
    for j, st in enumerate(SETS):
        for i, met in enumerate(("r", "slope", "intercept")):
            ax = axs[i, j]
            g = T[T.set == st]
            for k, (dk, dl, col) in enumerate(dirs):
                for ep in EP:
                    h = g[g.epoch == ep]
                    xx = x0[ep] - 0.3 + 0.2 * k
                    v = h[f"{met}_{dk}"].to_numpy(float)
                    ax.bar(xx, np.nanmean(v), 0.18, color=col, edgecolor="k", lw=0.5, label=dl if ep == EP[0] else None)
                    for a, vv in zip(h.animal, v):
                        ax.plot(xx, vv, MARK[a], color="w", mec="k", ms=4)
            ax.set_xticks(range(len(EP)), EP)
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_title(W(f"{SETS[st]}: decoded vs actual {met}" + (" (mm at y = 0; + = read contralesional)"
                                                                         if met == "intercept" else ""), 55), fontsize=9)
    hd, lb = axs[0, 0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=len(lb), fontsize=8, frameon=False)
    fig.suptitle(W("Transfer asymmetry: pre-trained (frozen) -> post vs post-trained -> pre vs within-epoch. Bars = mean "
                   "over animals, markers = animals; matched licks. Re-map = low transfer both ways with intact within; "
                   "offset / gain = high r with shifted intercept / slope.", 150), fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(FIG / f"lick_code_subspace_transfer{SFX}.png", dpi=130)
    plt.close(fig)

    # ---- drift figure
    fig, axs = plt.subplots(len(SETS), 4, figsize=(16, 3.4 * len(SETS)), squeeze=False)
    for j, st in enumerate(SETS):
        for k, a in enumerate(AN):
            ax = axs[j, k]
            d = DRIFT.get((st, a))
            if not d:
                ax.axis("off")
                continue
            gp = np.array(d["pairs_pre"])
            ax.plot(gp[:, 0], gp[:, 1], "o", color=EPOCH_GREY["pre"], mec="k", ms=4, label="pre-pre pairs")
            for ep in EP:
                q = [(g_, v) for g_, v, e in d["pairs_post"] if e == ep]
                if q:
                    q = np.array(q)
                    ax.plot(q[:, 0], q[:, 1], ".", color=EPOCH_GREY[ep], ms=4, label=f"{ep}-pre pairs")
            xs = np.linspace(0, max(gp[:, 0].max(), max([p[0] for p in d["pairs_post"]] or [1])), 50)
            ax.plot(xs, d["c0"] + d["m"] * xs, color="#b0413e", lw=1.2, label="pre drift line (extrapolated)")
            ax.axvline(d["gap_max_pre"], color="#b0413e", ls=":", lw=0.8)
            ax.set_title(W(f"{a} {SETS[st]}: m_pre {d['m'] * 30:+.3f} / 30 d", 45), fontsize=8.5)
            ax.set_xlabel("|day gap|", fontsize=8)
            ax.set_ylabel("per-session axis cosine", fontsize=8)
    hd, lb = axs[0, 0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=len(lb), fontsize=8, frameon=False)
    fig.suptitle(W("Drift vs re-organisation: pairwise cosine of per-session encoding axes (common lick subsample) vs "
                   "elapsed days. Red = the animal's pre-stroke drift line; dotted = longest pre-pre gap (beyond it the "
                   "line is extrapolated).", 150), fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(FIG / f"lick_code_subspace_drift{SFX}.png", dpi=130)
    plt.close(fig)

    # ---- areas figure: excess term per area x hemisphere
    fig, axs = plt.subplots(len(SETS), 1, figsize=(15, 3.7 * len(SETS)), squeeze=False)
    feats = sorted(AP.feature.unique(), key=lambda n: (n.rsplit("_", 1)[0], n.rsplit("_", 1)[1]))
    areas = sorted({f.rsplit("_", 1)[0] for f in feats})
    for j, st in enumerate(SETS):
        ax = axs[j, 0]
        g = AP[AP.set == st]
        M = np.full((2 * len(EP), len(areas)), np.nan)
        ylabs = []
        for i, (ep, h) in enumerate([(ep, h) for ep in EP for h in ("left", "right")]):
            ylabs.append(f"{ep} {'ipsi (L)' if h == 'left' else 'contra (R)'}")
            for k, ar in enumerate(areas):
                v = g[(g.epoch == ep) & (g.area == ar) & (g.hemi == h)].excess
                M[i, k] = v.mean() if len(v) else np.nan
        vm = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax.set_yticks(range(2 * len(EP)), ylabs, fontsize=8)
        ax.set_xticks(range(len(areas)), areas, rotation=60, fontsize=8)
        ax.set_title(W(f"{SETS[st]}: per-area excess contribution to the axis change (real minus ceiling term of "
                       "||u_post - u_pre||^2; + = this area changed more than pre-vs-pre)", 120), fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.02)
    fig.suptitle(W("Where does the encoding-axis change live? Mean over animals, matched licks.", 150), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / f"lick_code_subspace_areas{SFX}.png", dpi=130)
    plt.close(fig)
    _ = S_
    print("figures written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if il.ITI_SUBSET or il.AREA_BASIS != "split":
        raise SystemExit("run without WFIELD_ITI_FILTER / with the split basis")
    if a.plot_only:
        figures()
        return
    run(min(a.jobs, 4), a.reload)


if __name__ == "__main__":
    main()
