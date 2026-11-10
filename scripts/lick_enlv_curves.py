"""ENLv lick CURVES: decoded lateral position vs actual landing (lateral place AND 3-D angle) per epoch, in the form of
page Fig 12.2 (lick_matched_gauss_curves.png, `lick_matched_readout.py --ref gauss`), with ITI and cued alongside
(Priya 10-29: "landing or angle" curves for ENLv licks). PLUS (Priya 10-29, addition): GAIN vs OFFSET of the decoded-vs-
landing relation per animal x epoch, and an edge-compression check.

ENLv = ENL-violation lick (cam4 licks_all context ``pre_cue``: strobe <= onset < cue; spout stationary, no cue, no reward).
NEVER call them ITI. Two event sets, both with onset >= 1.0 s after spout ARRIVAL (strobe) and the lick-aligned window
ending before the measured cue (peak + 0.5 s < cue, lick_enlv_balance_offset.VARIANTS cue_end):
  enlv_a  the FIRST ENLv lick of each trial among the licks >= 1 s after arrival (that first lick must have a 3-D
          landing and clear the cue rule; a later lick is never substituted)
  enlv_b  the first lick of each ENLv BOUT (clean_start: >= 1 s since any lick) >= 1 s after arrival
          (= lick_enlv_balance_offset BOUTS, `select(L, "on1_clean")`)
READERS (frozen, rule 10; values read from enlv_table.csv.gz, nothing refitted): the frozen pre ITI decoder (dec_iti; the
reader of the Fig 12.2 ITI curves, verified bit-identical to lick_order_preds) = PRIMARY row; the frozen pre ENLv decoder
(dec_enlv, trained on the primary pre ENLv licks) = second row. ITI / cued panels = the Fig 12.2 inputs
(lick_order_preds_areas_splitr8: ITI reader on ITI licks, cued reader on cued licks). Pre sessions are always scored by
their held-out sibling.
CURVES: lick_matched_readout.curves (Gaussian-smoothed mean decoded position on a landing grid, sigma 0.1 mm for lateral
place; 2.0 deg for angle = the angle sigma chosen in lick_matched_angle_reach), per animal x epoch where the Gaussian weight
>= 10 licks, then mean +/- SEM over animals, solid = 4 animals, dashed = 3. Bars on top: lick_matched_readout.bin_tests
(resid = decoded - the animal's PRE same-set Gaussian reference (pre LOSO), session means per 0.25-mm / 5-deg landing bin
with >= 5 licks, change vs pre by analysis_kit.boot_delta, BH within set x epoch; >= 3 animals per bin).
ANGLE = degrees(arcsin(lateral / 3-D reach)) for every set (enl_lick_audit's definition); for ITI / cued the 3-D reach is
taken from where_angle_reach_preds (same lick rows, matched on the lateral landing in order within the session).
GAIN (Priya 10-29 addition): per panel x animal x post epoch, the landing range shared by pre and that epoch
(intersection of their 2.5-97.5 % landing quantiles; no extrapolation), center = its midpoint; per SESSION (>= 8 licks in
range) an OLS line decoded ~ landing -> SLOPE (gain) and value at the center (offset); change vs pre = per-animal table
first (rule 8), then analysis_kit.boot_delta. RESIDUAL: (epoch curve - pre curve) minus the gain + offset prediction from
the animal's pooled-lick lines, on the curve grid inside the shared range. EDGE check: local slope (gradient) of the
smoothed curves per epoch -- the decoder's own compression shows in the PRE (held-out) curve; a same-place comparison
cancels it unless the activity differs.
Outputs <OUT_W>/enlv_licks/: curves_counts.csv, curves_coverage.csv, curves_bin_tests.csv, curves_gain_sessions.csv,
curves_gain_tests.csv, curves_edge.csv; figures/lick_enlv_curves.png (lateral place), lick_enlv_curves_angle.png,
lick_enlv_curves_gain.png; figdata/lick_enlv_curves.pkl.
10-29 (Priya, ALL lick types in one figure): two more sets, each with its own pre same-set reference -- iti_q = ITI FIRST
lick after >= 2 s quiet (lick_recovery_markers.QSET / _gaps; the primary ITI set) and trial_1 = cued FIRST post-cue lick
(order 0, lick_matched_readout cls 'first'); rows ADDED to every CSV above (existing sets' values unchanged: all tests are
per set). figures/lick_like_for_like_all.png (ITI all, ITI 1st after quiet, cued all, cued 1st, ENLv a, ENLv b with their
Fig 12.2 readers + gain / offset change), figdata/lick_like_for_like_all.pkl. Far-contra end of the cued curves:
curves_farcontra_sessions.csv / curves_farcontra_tests.csv (resid by post-cue lick order and spout contact).
Usage: python scripts/lick_enlv_curves.py [--plot-only [--all-only]]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_enlv_balance_offset as bo  # noqa: E402
import lick_matched_readout as lmr  # noqa: E402

lmr.REF = "gauss"            # the Fig 12.2 form (Gaussian curves, same-place bins); this script never writes lmr outputs

OUT = bo.OUT
FIG = OUT / "figures"
FD = OUT / "figdata"
STEM = "lick_enlv_curves"
AN = bo.AN
EPS = bo.EPS
POST = bo.POST
POS = bo.POS
# panel key -> (lick set, reader, label)
PANELS = {"iti|iti": ("iti", "iti", "ITI licks"), "trial|trial": ("trial", "trial", "cued licks"),
          "enlv_a|iti": ("enlv_a", "iti", "ENLv (a) FIRST lick / trial >= 1 s"),
          "enlv_b|iti": ("enlv_b", "iti", "ENLv (b) BOUT starts >= 1 s"),
          "enlv_a|enlv": ("enlv_a", "enlv", "ENLv (a) FIRST lick / trial >= 1 s"),
          "enlv_b|enlv": ("enlv_b", "enlv", "ENLv (b) BOUT starts >= 1 s"),
          "iti_q|iti": ("iti_q", "iti", "ITI FIRST lick after >= 2 s quiet"),
          "trial_1|trial": ("trial_1", "trial", "cued FIRST lick (post-cue order 1)")}
# the combined like-for-like figure (Priya 10-29): every lick set with its Fig 12.2 reader
ALL_PANELS = ["iti|iti", "iti_q|iti", "trial|trial", "trial_1|trial", "enlv_a|iti", "enlv_b|iti"]
SHORT = {"iti|iti": "ITI all", "iti_q|iti": "ITI 1st\nafter quiet", "trial|trial": "cued all", "trial_1|trial": "cued 1st",
         "enlv_a|iti": "ENLv (a)", "enlv_b|iti": "ENLv (b)"}
FC_BAND = (1.1, 1.9)     # far-contralesional landing band (mm) of the far-contra-end check
ORD_CLS = {"1st": (0, 0), "2nd-3rd": (1, 2), "4th+": (3, 999)}
RLAB = {"iti": "frozen pre ITI reader", "trial": "frozen pre cued reader", "enlv": "frozen pre ENLv reader"}
# axis -> (column, sigma, grid, bins, unit, label)
AXES = {"lat": ("act", 0.1, np.round(np.arange(-3.0, 3.001, 0.05), 3), lmr.FINE, "mm", "landing, lateral place"),
        "ang": ("ang", 2.0, np.round(np.arange(-60.0, 60.001, 1.0), 3), np.arange(-60.0, 60.01, 5.0), "deg",
                "landing, 3-D angle")}
#: enlv_a = the first ENLv lick of each trial among the PRIMARY candidates (onset / cue rule of VARIANTS["on1"])
ENLV_A_ONSET = float(bo.VARIANTS[bo.PRIMARY][1][0])
ENLV_A_CUE_END = float(bo.VARIANTS[bo.PRIMARY][3])
MIN_G = 8           # licks per session inside the shared range for a session line
QR = (2.5, 97.5)    # landing quantiles defining each epoch's range


# ================================================================================================ trial-set captions
def set_sentence(st: str) -> str:
    """The lick set ``st`` of `load` in words (Priya 11-09: every figure states exactly which licks)."""
    from wfield_local import trial_set_text as tst
    filt = lo.ITI_FILTER or il.ITI_SUBSET
    cap = (f"iti_licks.extract post-cue set: every trial, no engagement / camera-class gate, at most "
           f"{int(il.N_TRIAL_MAX)} per session (random draw, seed 0)")
    land = "3-D landing (lateral) and decoded value present"
    if st == "iti":
        return tst.iti_licks(filt=filt, extra=["the decoder cache's ITI licks", land])
    if st == "iti_q":
        return tst.iti_licks(filt=filt, extra=[f"FIRST lick after >= {il.GAP_MAIN:g} s quiet (no licks_all onset of "
                                               f"any context in the preceding {il.GAP_MAIN:g} s)", land])
    if st == "trial":
        return tst.cued_licks(extra=[cap, land])
    if st == "trial_1":
        return tst.cued_licks(first_only=True, extra=[cap, land])
    if st == "enlv_a":
        return tst.events(f"{bo.ENLV_WHAT} [enlv_a]", keep=[
            f"the FIRST ENLv lick of each trial among those with onset >= {ENLV_A_ONSET:g} s after arrival (a later lick is "
            "never substituted)", f"lick peak + {ENLV_A_CUE_END:g} s before the cue", f"spouts {', '.join(POS)}", land],
            excluded=["ITI and post-cue licks"])
    if st == "enlv_b":
        return bo.variant_sentence(bo.BOUTS, extra=[f"spouts {', '.join(POS)}", land])
    raise ValueError(st)


def trial_set_sentence(panels=None) -> str:
    """The lick sets of the panels shown (default: the curve / gain figures' four sets)."""
    from wfield_local import trial_set_text as tst
    panels = list(panels) if panels is not None else GAIN_ROWS
    sets = list(dict.fromkeys(PANELS[p][0] for p in panels))
    return tst.union(*(set_sentence(st) for st in sets))


def _trial_set_for(panels):
    import contextlib

    from wfield_local import trial_set_text as tst
    if tst.current() is None:
        return contextlib.nullcontext()
    return tst.active(None, text=trial_set_sentence(panels))


# ================================================================================================ inputs
def _angle_iti_cued(P: pd.DataFrame) -> np.ndarray:
    """3-D angle (arcsin(lateral / reach), deg) per lick_order_preds row, from where_angle_reach_preds (same session lick
    stream, a superset of rows): matched in order within set x session on the lateral landing (lateral = sag x tan(ang))."""
    W = pd.read_csv(il.pred_file("where_angle_reach_preds.csv.gz"))
    W["lat"] = np.round(W.act_sag * np.tan(np.deg2rad(W.act_ang)), 3)
    W["ang"] = np.degrees(np.arcsin(np.clip(W.lat / W.act_reach, -1, 1)))
    out = np.full(len(P), np.nan)
    wg = {k: (g.lat.to_numpy(), g.ang.to_numpy()) for k, g in W.groupby(["set", "label"], sort=False)}
    for k, g in P.groupby(["set", "label"], sort=False):
        if k not in wg:
            continue
        wl, wa = wg[k]
        pv = np.round(g.act.to_numpy(float), 3)
        res, j = np.full(len(pv), np.nan), 0
        for i, v in enumerate(pv):
            while j < len(wl) and wl[j] != v:
                j += 1
            if j < len(wl):
                res[i] = wa[j]
                j += 1
        out[P.index.get_indexer(g.index)] = res
    return out


def _iti_gaps(P: pd.DataFrame) -> pd.DataFrame:
    """Time since the previous lick ONSET per ITI row (lick_recovery_markers._gaps, the QSET definition: rows keyed by
    k = cumcount within set x label of the full KEY cache, as in lick_iti_prelick.lick_set). Session licks_all files only."""
    import lick_recovery_markers as lrm
    from wfield_local import analysis_kit as ak
    res, fail = ak.fan_sessions(sorted(P[P.set == "iti"].label.unique()), lrm._gaps, jobs=4)
    if fail:
        raise RuntimeError(f"gaps failed: {fail[:5]}")
    return pd.concat([v for _i, v in res], ignore_index=True)


def _contact(label: str) -> pd.DataFrame:
    """Cued licks of one session -> licks_all `contact` (DAQ spout-contact inside the cam4 lick; lick_balance_contact's
    definition), matched on the exact DAQ peak time t of covref_covariates. Module-level for spawn."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "t"])
    C = C[(C.label == label) & (C.set == "trial")]
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_peak_s", "contact"])
    A = A[np.isfinite(A.daq_peak_s)].drop_duplicates("daq_peak_s").set_index("daq_peak_s").contact
    return C.assign(contact=A.reindex(C.t.to_numpy(float)).to_numpy()).drop(columns="t")


def load() -> pd.DataFrame:
    """One long per-lick frame: panel, set, reader, animal, label, epoch, pos, act (lateral mm), ang (deg), dec, order, k.
    10-29 (Priya, combined figure): + iti_q = ITI FIRST lick after >= il.GAP_MAIN (2 s) quiet (lick_recovery_markers.QSET,
    the primary ITI set) and trial_1 = cued FIRST post-cue lick (order 0, lick_matched_readout cls 'first'), each read by
    its own Fig 12.2 reader (ITI / cued) and each with its OWN pre same-set reference (new sets; existing ones unchanged)."""
    P = pd.read_csv(lo.preds_path())
    P["k"] = P.groupby(["set", "label"]).cumcount()       # covref_covariates key (before any subset)
    P = P.merge(_iti_gaps(P), on=["set", "label", "k"], how="left", validate="one_to_one")
    if il.ITI_SUBSET:                         # same subset hook as lick_matched_readout.per_lick (off by default)
        P = P[il.iti_subset_keep(P)].reset_index(drop=True)
    P["ang"] = _angle_iti_cued(P)
    miss = P.ang.isna().groupby(P.set).mean()
    print("ITI / cued rows without a matched 3-D angle (share):", miss.round(4).to_dict(), flush=True)
    keep = ["panel", "set", "reader", "animal", "label", "epoch", "pos", "act", "ang", "dec", "order", "k"]
    parts = [P.assign(reader=P.set, panel=P.set + "|" + P.set)[keep]]
    q = P[(P.set == "iti") & (P.gap >= il.GAP_MAIN)]
    print(f"ITI first-after-quiet (gap >= {il.GAP_MAIN} s): {len(q)} of {int((P.set == 'iti').sum())} ITI rows; "
          f"no gap matched: {int(P[P.set == 'iti'].gap.isna().sum())}", flush=True)
    parts.append(q.assign(set="iti_q", reader="iti", panel="iti_q|iti")[keep])
    parts.append(P[(P.set == "trial") & (P.order == 0)].assign(set="trial_1", reader="trial", panel="trial_1|trial")[keep])
    L = pd.read_csv(OUT / "enlv_table.csv.gz")
    L = L.sort_values(["label", "trial_id", "daq_on_s"]).reset_index(drop=True)
    L["ang"] = L.angle
    cand = L.t_from_strobe >= ENLV_A_ONSET
    first = cand & (L[cand].groupby(["label", "trial_id"]).cumcount() == 0).reindex(L.index, fill_value=False)
    sel = {"enlv_a": first & (L.peak_to_cue > ENLV_A_CUE_END), "enlv_b": bo.select(L, "on1_clean")}
    for st, m in sel.items():
        g = L[m & L.pos.isin(POS) & np.isfinite(L.act) & np.isfinite(L.ang)]
        for r in ("iti", "enlv"):
            parts.append(g.assign(set=st, reader=r, panel=f"{st}|{r}", dec=g[f"dec_{r}"], order=np.nan, k=-1)[keep])
    Q = pd.concat(parts, ignore_index=True)
    return Q[np.isfinite(Q.act) & np.isfinite(Q.dec)].reset_index(drop=True)


# ================================================================================================ curves + bin tests
def curves(Q: pd.DataFrame, ax: str) -> pd.DataFrame:
    col, sg, grid, *_ = AXES[ax]
    P = Q.dropna(subset=[col]).assign(set=Q.panel, act=Q[col])[["set", "epoch", "animal", "label", "act", "dec"]]
    P.attrs["sigma"] = sg
    keep = lmr.CURVE_GRID
    try:
        lmr.CURVE_GRID = grid
        return lmr.curves(P)
    finally:
        lmr.CURVE_GRID = keep


def bins(Q: pd.DataFrame, ax: str) -> pd.DataFrame:
    col, sg, _grid, edges, *_ = AXES[ax]
    P = Q.dropna(subset=[col]).assign(set=Q.panel, act=Q[col])[["set", "epoch", "animal", "label", "act", "dec"]]
    P = P.reset_index(drop=True)
    P["resid"] = P.dec - lmr.gauss_ref(P, sg)
    P["bin_a"] = pd.cut(P.act, edges)
    T = lmr.bin_tests(P)
    T["axis"] = ax
    T["width"] = float(np.diff(edges)[0])
    return T


def counts(Q: pd.DataFrame) -> pd.DataFrame:
    g = Q[Q.reader.isin(["iti", "trial"])].groupby(["set", "animal", "epoch"])
    C = g.agg(n_licks=("dec", "size"), n_sessions=("label", "nunique")).reset_index()
    C["licks_per_session"] = (C.n_licks / C.n_sessions).round(1)
    return C


def coverage(A: pd.DataFrame, ax: str) -> pd.DataFrame:
    """Where each animal x epoch is drawn (Gaussian weight >= 10) and where 4 / 3 animals contribute."""
    rows = []
    for (pn, ep), g in A.groupby(["set", "epoch"]):
        for a, h in g.groupby("animal"):
            rows.append(dict(axis=ax, panel=pn, epoch=ep, who=a, lo=h.mid.min(), hi=h.mid.max(), n_pts=len(h)))
        nn = g.groupby("mid").animal.nunique()
        for k in (4, 3):
            m = nn[nn >= k]
            rows.append(dict(axis=ax, panel=pn, epoch=ep, who=f">={k} animals", lo=m.index.min() if len(m) else np.nan,
                             hi=m.index.max() if len(m) else np.nan, n_pts=len(m)))
    return pd.DataFrame(rows)


# ================================================================================================ gain vs offset
def _line(x, y):
    if len(x) < MIN_G or np.std(x) == 0:
        return np.nan, np.nan
    b1, b0 = np.polyfit(x, y, 1)
    return b1, b0


def gain(Q: pd.DataFrame, ax: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Session lines (slope, value at the shared-range center) per panel x animal x post epoch, and the animal's
    pooled-lick lines (for the residual)."""
    col = AXES[ax][0]
    srows, arows = [], []
    for (pn, a), g in Q.dropna(subset=[col]).groupby(["panel", "animal"]):
        pre = g[g.epoch == "pre"]
        if pre.label.nunique() < 2:
            continue
        for ep in POST:
            po = g[g.epoch == ep]
            if len(po) < MIN_G:
                continue
            lo_ = max(np.percentile(pre[col], QR[0]), np.percentile(po[col], QR[0]))
            hi_ = min(np.percentile(pre[col], QR[1]), np.percentile(po[col], QR[1]))
            if hi_ <= lo_:
                continue
            c = 0.5 * (lo_ + hi_)
            for kind, h in (("pre", pre), (ep, po)):
                h = h[(h[col] >= lo_) & (h[col] <= hi_)]
                b1, b0 = _line(h[col].to_numpy(float), h.dec.to_numpy(float))
                arows.append(dict(axis=ax, panel=pn, animal=a, comp=ep, epoch=kind, lo=lo_, hi=hi_, centre=c, n=len(h),
                                  slope=b1, offset=b0 + b1 * c if np.isfinite(b1) else np.nan))
                for lab, s in h.groupby("label"):
                    b1, b0 = _line(s[col].to_numpy(float), s.dec.to_numpy(float))
                    if np.isfinite(b1):
                        srows.append(dict(axis=ax, panel=pn, animal=a, comp=ep, epoch=kind, label=lab, n=len(s),
                                          lo=lo_, hi=hi_, centre=c, slope=b1, offset=b0 + b1 * c))
    return pd.DataFrame(srows), pd.DataFrame(arows)


def gain_tests(S: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    out = []
    for (axn, pn, ep), g in S.groupby(["axis", "panel", "comp"]):
        for val in ("slope", "offset"):
            prd = {a: h[val].to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            post = {a: h[val].to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in prd}
            am = (g[g.epoch == ep].groupby("animal")[val].mean() - g[g.epoch == "pre"].groupby("animal")[val].mean())
            am = am.dropna()
            pm = g[g.epoch == "pre"].groupby("animal")[val].mean()
            r = dict(axis=axn, panel=pn, epoch=ep, measure=val, pre_mean=pm.mean(), n=len(am),
                     **{f"animal_{a}": v for a, v in am.items()},
                     **{f"nsess_{a}": int((g[(g.epoch == ep) & (g.animal == a)]).shape[0]) for a in am.index})
            if len(post) >= 2:
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                if ci is not None:
                    r.update(point=ci.point, lo=ci.lo, hi=ci.hi, n_dir=int((am * np.sign(ci.point) > 0).sum()))
            out.append(r)
    return pd.DataFrame(out)


def residuals(A: pd.DataFrame, AL: pd.DataFrame, ax: str) -> pd.DataFrame:
    """(epoch curve - pre curve) and that minus the gain + offset prediction (and minus the offset alone), per animal on
    the curve grid inside the shared range."""
    rows = []
    for (pn, a, ep), g in AL.groupby(["panel", "animal", "comp"]):
        pr, po = g[g.epoch == "pre"].iloc[0], g[g.epoch == ep].iloc[0]
        cp = A[(A.set == pn) & (A.animal == a) & (A.epoch == "pre")].set_index("mid").y
        ce = A[(A.set == pn) & (A.animal == a) & (A.epoch == ep)].set_index("mid").y
        d = (ce - cp).dropna()
        d = d[(d.index >= pr.lo) & (d.index <= pr.hi)]
        if not len(d) or not (np.isfinite(po.slope) and np.isfinite(pr.slope)):
            continue
        x = d.index.to_numpy(float)
        pred = (po.offset - pr.offset) + (po.slope - pr.slope) * (x - pr.centre)
        rows += [dict(axis=ax, panel=pn, animal=a, epoch=ep, mid=m, diff=v, resid_gain=v - p, resid_offset=v - (po.offset - pr.offset))
                 for m, v, p in zip(x, d.to_numpy(float), pred)]
    return pd.DataFrame(rows)


def edge(A: pd.DataFrame, AL: pd.DataFrame, ax: str) -> pd.DataFrame:
    """Per panel x animal x post epoch: mean (epoch - pre) curve difference in the LOWER and UPPER 20 % of the shared
    range and in the middle 60 %, plus the PRE curve's local slope in those thirds (the decoder's own edge compression)."""
    rows = []
    for (pn, a, ep), g in AL.groupby(["panel", "animal", "comp"]):
        pr = g[g.epoch == "pre"].iloc[0]
        cp = A[(A.set == pn) & (A.animal == a) & (A.epoch == "pre")].set_index("mid").y
        ce = A[(A.set == pn) & (A.animal == a) & (A.epoch == ep)].set_index("mid").y
        w = pr.hi - pr.lo
        for zone, (z0, z1) in {"low": (pr.lo, pr.lo + 0.2 * w), "mid": (pr.lo + 0.2 * w, pr.hi - 0.2 * w),
                               "high": (pr.hi - 0.2 * w, pr.hi)}.items():
            p_, e_ = cp[(cp.index >= z0) & (cp.index <= z1)], ce[(ce.index >= z0) & (ce.index <= z1)]
            d = (e_ - p_).dropna()
            sl = lambda s: np.polyfit(s.index.to_numpy(float), s.to_numpy(float), 1)[0] if len(s) >= 3 else np.nan  # noqa: E731
            rows.append(dict(axis=ax, panel=pn, animal=a, epoch=ep, zone=zone, diff=d.mean() if len(d) else np.nan,
                             pre_local_slope=sl(p_), ep_local_slope=sl(e_)))
    return pd.DataFrame(rows)


# ================================================================================================ far-contra end
def farcontra(Q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Far-contralesional end of the cued curves (Priya 10-29): resid = decoded - the animal's PRE Gaussian reference
    (lick_matched_readout.gauss_ref, sigma 0.1 mm, pre leave-one-session-out). Reference 'cued all' = all pre cued licks
    (= matched_gauss_tests, where cls 'first' is a subset of the cued-all resid); 'own 1st' = pre FIRST licks only (= the
    trial_1 curve panel). Scopes: spout far_contra; landing in FC_BAND. Split by post-cue lick order (ORD_CLS) and by
    spout contact (licks_all `contact`). Session means (>= lmr.MIN licks), change vs pre per animal, then boot_delta."""
    from wfield_local import analysis_kit as ak
    sg = AXES["lat"][1]
    res, fail = ak.fan_sessions(sorted(Q[Q.panel == "trial|trial"].label.unique()), _contact, jobs=4)
    if fail:
        raise RuntimeError(f"contact failed: {fail[:5]}")
    K = pd.concat([v for _i, v in res], ignore_index=True)
    K["contact"] = K.contact.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    D = {}
    for ref, pn in (("cued all", "trial|trial"), ("own 1st", "trial_1|trial")):
        T = Q[Q.panel == pn].reset_index(drop=True)
        T["resid"] = T.dec - lmr.gauss_ref(T, sg)
        D[ref] = T.merge(K[["label", "k", "contact"]], on=["label", "k"], how="left", validate="many_to_one")
    print("cued rows without a contact value (share):", round(float(D["cued all"].contact.isna().mean()), 4), flush=True)
    srows = []
    for ref, T in D.items():
        for scope, ms in (("spout far_contra", T.pos == "far_contra"),
                          (f"landing {FC_BAND[0]}..{FC_BAND[1]} mm", T.act.between(*FC_BAND))):
            for cls, (o0, o1) in ({"all": (0, 999), **ORD_CLS} if ref == "cued all" else {"1st": (0, 0)}).items():
                for ct, mc in (("all", True), ("contact", T.contact == 1), ("miss", T.contact == 0)):
                    g = T[ms & T.order.between(o0, o1) & mc & np.isfinite(T.resid)]
                    for (a, lab, ep), h in g.groupby(["animal", "label", "epoch"]):
                        srows.append(dict(ref=ref, scope=scope, cls=cls, contact=ct, animal=a, label=lab, epoch=ep,
                                          n=len(h), resid=h.resid.mean(), act=h.act.mean()))
    S = pd.DataFrame(srows)
    S = S[S.n >= lmr.MIN]
    out = []
    for (ref, scope, cls, ct), g in S.groupby(["ref", "scope", "cls", "contact"], sort=False):
        pre = g[g.epoch == "pre"]
        prd = {a: h.resid.to_numpy() for a, h in pre.groupby("animal")}
        for ep in POST:
            po = g[g.epoch == ep]
            am = (po.groupby("animal").resid.mean() - pre.groupby("animal").resid.mean()).dropna()
            r = dict(ref=ref, scope=scope, cls=cls, contact=ct, epoch=ep, n=len(am),
                     **{f"animal_{a}": am.get(a, np.nan) for a in AN},
                     **{f"nlick_{a}": int(po[po.animal == a].n.sum()) for a in AN},
                     **{f"nlickpre_{a}": int(pre[pre.animal == a].n.sum()) for a in AN})
            post = {a: h.resid.to_numpy() for a, h in po.groupby("animal") if a in prd}
            if len(post) >= 2:
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                if ci is not None:
                    r.update(point=ci.point, lo=ci.lo, hi=ci.hi, n_dir=int((am * np.sign(ci.point) > 0).sum()))
            out.append(r)
    return S, pd.DataFrame(out)


# ================================================================================================ figures
def _sty(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def fig_curves(D: dict, ax_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    A, BT = D["A"][ax_name], D["BT"][ax_name]
    _c, sg, _g, _e, unit, xlab = AXES[ax_name]
    layout = [["iti|iti", "trial|trial", "enlv_a|iti", "enlv_b|iti"], [None, None, "enlv_a|enlv", "enlv_b|enlv"]]
    fig, axs = plt.subplots(2, 4, figsize=(21, 10.5), squeeze=False)
    epl = [Line2D([], [], color=col[e], lw=2, label=e) for e in EPS]
    ylim = [np.inf, -np.inf]
    for i, row in enumerate(layout):
        for j, pn in enumerate(row):
            ax = axs[i, j]
            if pn is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "the frozen ENLv reader is applied\nto ENLv licks only\n(ITI / cued panels: top row)",
                        ha="center", va="center", fontsize=9, color="0.4", transform=ax.transAxes)
                continue
            st, rd, plab = PANELS[pn]
            for ep in EPS:
                q = A[(A.set == pn) & (A.epoch == ep)].groupby("mid").y
                mu, se, nn = q.mean(), q.sem(), q.count()
                m4 = mu.where(nn >= 4)
                x = mu.index.to_numpy(float)
                ax.plot(x, mu.where(nn >= 3), color=col[ep], lw=0.9, ls="--", zorder=3)
                ax.plot(x, m4, color=col[ep], lw=2.0, zorder=4)
                ax.fill_between(x, m4 - se, m4 + se, color=col[ep], alpha=0.08, lw=0, zorder=2)
                ok = mu.where(nn >= 3).dropna()
                if len(ok):
                    ylim = [min(ylim[0], ok.min()), max(ylim[1], ok.max())]
            hw = BT["width"].iloc[0] / 2 if len(BT) else 0
            for r_i, ep in enumerate(POST):
                bt = BT[(BT.set == pn) & (BT.epoch == ep) & BT.sig] if len(BT) else BT
                for m_ in bt.mid:
                    ax.axvspan(m_ - hw, m_ + hw, ymin=0.975 - 0.018 * r_i, ymax=0.987 - 0.018 * r_i, color=col[ep],
                               alpha=0.35, lw=0, zorder=0)
            ax.axhline(0, color="0.7", lw=0.6, ls=":")
            ax.axvline(0, color="0.7", lw=0.6, ls=":")
            nsig = {ep: int(((BT.set == pn) & (BT.epoch == ep) & BT.sig).sum()) if len(BT) else 0 for ep in POST}
            ntest = {ep: int(((BT.set == pn) & (BT.epoch == ep)).sum()) if len(BT) else 0 for ep in POST}
            ax.set_title(f"NEURAL: {plab}\nreader: {RLAB[rd]}\nBH bins (sig/tested) " +
                         ", ".join(f"{e[:3]} {nsig[e]}/{ntest[e]}" for e in POST), fontsize=8.5)
            ax.set_xlabel(f"{xlab} ({unit}, + = contralesional)", fontsize=8)
            if j == 0 or (i == 1 and j == 2):
                ax.set_ylabel("decoded lateral (mm, + = contralesional)", fontsize=8)
            ax.legend(handles=epl, fontsize=7, frameon=False, loc="lower right")
            _sty(ax)
    for ax in axs.ravel():
        if ax.axison and np.isfinite(ylim[0]):
            ax.set_ylim(ylim[0] - 0.1, ylim[1] + 0.25)
    xr = {"lat": (-3, 3), "ang": (-60, 60)}[ax_name]
    for ax in axs.ravel():
        if ax.axison:
            ax.set_xlim(*xr)
    fig.suptitle(f"NEURAL, LIKE-FOR-LIKE: decoded lateral position vs actual {xlab} per epoch -- ITI, cued and ENLv licks "
                 f"(frozen KEY-row lateral decoders; pre = held-out sessions).\nGaussian-smoothed, sigma {sg:g} {unit}; "
                 "per animal x epoch where the Gaussian weight >= 10 licks; mean +/- SEM of 4 animals, dashed = only 3 "
                 "animals. A curve shifted UP = read more contralesional than pre licks that landed at the same place.\n"
                 "ENLv (a) = first ENLv lick of each trial among licks >= 1 s after spout arrival; (b) = ENLv bout starts "
                 "(>= 1 s since any lick) >= 1 s after arrival; both windows end before the cue. Bars at the top = landing "
                 "bins whose change vs pre survives BH\n(paired nested bootstrap on session means, >= 5 licks per bin, >= 3 "
                 "animals; rows acute / subacute / chronic). Top row = ITI reader for ITI and ENLv licks (cued panel: cued "
                 "reader, as Fig 12.2); bottom = ENLv reader. n = 4 mice, PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_png_svg(fig, FIG / f"{STEM}{'' if ax_name == 'lat' else '_angle'}.png", dpi=100)
    plt.close(fig)


GAIN_ROWS = ["iti|iti", "trial|trial", "enlv_a|iti", "enlv_b|iti", "enlv_a|enlv", "enlv_b|enlv"]


def fig_gain(D: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    GT, R, A = D["GT"], D["R"], D["A"]
    cols = [("lat", "slope"), ("lat", "offset"), ("lat", "resid"), ("lat", "local"), ("ang", "slope"), ("ang", "offset"),
            ("ang", "resid")]
    fig, axs = plt.subplots(len(GAIN_ROWS), len(cols), figsize=(27, 3.3 * len(GAIN_ROWS)), squeeze=False)
    for i, pn in enumerate(GAIN_ROWS):
        st, rd, plab = PANELS[pn]
        for j, (axn, kind) in enumerate(cols):
            ax = axs[i, j]
            unit = AXES[axn][4]
            if kind in ("slope", "offset"):
                g = GT[(GT.axis == axn) & (GT.panel == pn) & (GT.measure == kind)].set_index("epoch")
                for k, ep in enumerate(POST):
                    if ep not in g.index:
                        continue
                    r = g.loc[ep]
                    c = EPOCH_LINE[ep]
                    for q, a in enumerate(AN):
                        v = r.get(f"animal_{a}", np.nan)
                        if np.isfinite(v):
                            ax.plot(k - 0.15 + 0.1 * q, v, ANIMAL_MARKER[a], mfc="none", mec=c, ms=6, mew=1.2)
                    if np.isfinite(r.get("point", np.nan)):
                        sig = r.lo > 0 or r.hi < 0
                        ax.errorbar(k + 0.3, r.point, yerr=[[r.point - r.lo], [r.hi - r.point]], fmt="o", color="k",
                                    mfc=c if sig else "white", ms=6, capsize=3)
                        ax.text(k + 0.3, 1.01, f"{r.point:+.2f}{'*' if sig else ''}\n{int(r.n_dir)}/{int(r.n)}",
                                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5,
                                fontweight="bold" if sig else "normal")
                ax.axhline(0, color="0.6", lw=0.7, ls="--")
                ax.set_xticks(range(3), POST, fontsize=7.5)
                ax.set_xlim(-0.5, 2.7)
                ttl = (f"GAIN change vs pre\n(slope, mm/{unit})" if kind == "slope" else
                       "OFFSET change vs pre\n(value at shared-range center, mm)")
                ax.set_title(f"{'lateral' if axn == 'lat' else 'angle'}: {ttl}", fontsize=8, pad=20)
            elif kind == "resid":
                g = R[(R.axis == axn) & (R.panel == pn)]
                for ep in POST:
                    h = g[g.epoch == ep]
                    if h.empty:
                        continue
                    c = EPOCH_LINE[ep]
                    for v, ls, lw in (("diff", ":", 1.0), ("resid_offset", "--", 1.0), ("resid_gain", "-", 1.8)):
                        q = h.groupby("mid")[v]
                        mu, nn = q.mean(), q.count()
                        ax.plot(mu.index.to_numpy(float), mu.where(nn >= 3), color=c, ls=ls, lw=lw)
                ax.axhline(0, color="0.6", lw=0.7)
                ax.set_title(f"{'lateral' if axn == 'lat' else 'angle'}: epoch - pre curve (dotted), minus offset "
                             "(dashed),\nminus gain + offset (solid); mean of >= 3 animals", fontsize=7.5)
                ax.set_xlabel(f"landing ({unit})", fontsize=7.5)
            else:                                        # local slope of the smoothed curves (edge compression)
                g = A["lat"][A["lat"].set == pn]
                for ep in EPS:
                    sl = []
                    for _a, h in g[g.epoch == ep].groupby("animal"):
                        h = h.sort_values("mid")
                        s = pd.Series(np.gradient(h.y.to_numpy(float), h.mid.to_numpy(float)), index=h.mid.to_numpy())
                        gap = np.r_[False, np.diff(h.mid.to_numpy()) > 0.051]
                        s[gap] = np.nan
                        sl.append(s.rolling(11, center=True, min_periods=7).mean())   # ~0.5 mm
                    if not sl:
                        continue
                    M = pd.concat(sl, axis=1)
                    mu = M.mean(1).where(M.count(1) >= 3)
                    ax.plot(mu.index.to_numpy(float), mu, color={"pre": "k", **EPOCH_LINE}[ep], lw=1.5)
                ax.axhline(0, color="0.6", lw=0.7)
                ax.set_xlim(-3, 3)
                ax.set_title("lateral: LOCAL slope of the smoothed curve\n(d decoded / d landing, ~0.5-mm running mean; pre = held-out = the "
                             "decoder's own compression)", fontsize=7.5)
                ax.set_xlabel("landing (mm)", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{plab}\n{RLAB[rd]}", fontsize=8)
            _sty(ax)
    fig.suptitle("NEURAL: GAIN vs OFFSET of decoded lateral position on landing (lateral place, angle), per animal x "
                 "epoch, over the landing range SHARED by pre and that epoch (2.5-97.5 % of each; no extrapolation).\n"
                 "Session OLS lines (>= 8 licks in range): slope = gain, value at the range center = offset; open markers "
                 "= animals (circle PS92, square PS93, triangle PS94, diamond PS95), black = change vs pre, 95% paired "
                 "nested bootstrap, * = excludes 0, k/n animals.\nResidual columns: if gain + offset describe the change, "
                 "the solid line is flat at 0 across the range (both ends). n = 4 mice, PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save_png_svg(fig, FIG / f"{STEM}_gain.png", dpi=100)
    plt.close(fig)


def fig_all(D: dict) -> None:
    """lick_like_for_like_all.png (Priya 10-29): every lick set in one figure -- like-for-like curves (lateral place) with
    BH bars, and a gain / offset change summary (per-animal dots + paired nested bootstrap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    A, BT, GT = D["A"]["lat"], D["BT"]["lat"], D["GT"]
    _c, sg, _g, _e, unit, xlab = AXES["lat"]
    fig = plt.figure(figsize=(27, 13.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.8], left=0.045, right=0.99, top=0.875, bottom=0.075, hspace=0.55,
                          wspace=0.22)
    axs = [fig.add_subplot(gs[0, j]) for j in range(6)]
    ylim = [np.inf, -np.inf]
    for j, pn in enumerate(ALL_PANELS):
        ax = axs[j]
        _st, rd, plab = PANELS[pn]
        for ep in EPS:
            q = A[(A.set == pn) & (A.epoch == ep)].groupby("mid").y
            mu, se, nn = q.mean(), q.sem(), q.count()
            m4 = mu.where(nn >= 4)
            x = mu.index.to_numpy(float)
            ax.plot(x, mu.where(nn >= 3), color=col[ep], lw=0.9, ls="--", zorder=3)
            ax.plot(x, m4, color=col[ep], lw=2.0, zorder=4)
            ax.fill_between(x, m4 - se, m4 + se, color=col[ep], alpha=0.08, lw=0, zorder=2)
            ok = mu.where(nn >= 3).dropna()
            if len(ok):
                ylim = [min(ylim[0], ok.min()), max(ylim[1], ok.max())]
        hw = BT["width"].iloc[0] / 2 if len(BT) else 0
        for r_i, ep in enumerate(POST):
            for m_ in BT[(BT.set == pn) & (BT.epoch == ep) & BT.sig].mid:
                ax.axvspan(m_ - hw, m_ + hw, ymin=0.975 - 0.018 * r_i, ymax=0.987 - 0.018 * r_i, color=col[ep],
                           alpha=0.35, lw=0, zorder=0)
        ax.axhline(0, color="0.7", lw=0.6, ls=":")
        ax.axvline(0, color="0.7", lw=0.6, ls=":")
        nsig = {ep: int(((BT.set == pn) & (BT.epoch == ep) & BT.sig).sum()) for ep in POST}
        ntest = {ep: int(((BT.set == pn) & (BT.epoch == ep)).sum()) for ep in POST}
        ax.set_title(f"NEURAL: {plab}\nreader: {RLAB[rd]}\nBH bins (sig/tested) " +
                     ", ".join(f"{e[:3]} {nsig[e]}/{ntest[e]}" for e in POST), fontsize=8.5, pad=6)
        ax.set_xlabel(f"{xlab} ({unit}, + = contralesional)", fontsize=8)
        if j == 0:
            ax.set_ylabel("decoded lateral (mm, + = contralesional)", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        _sty(ax)
    for ax in axs:
        if np.isfinite(ylim[0]):
            ax.set_ylim(ylim[0] - 0.1, ylim[1] + 0.25)
        ax.set_xlim(-3, 3)
    # ---- gain / offset change vs pre (lateral place)
    for k_, (meas, ttl) in enumerate((("slope", "GAIN change vs pre (slope of decoded on landing, mm/mm)"),
                                      ("offset", "OFFSET change vs pre (decoded at the shared-range center, mm)"))):
        ax = fig.add_subplot(gs[1, 3 * k_:3 * k_ + 3])
        g = GT[(GT.axis == "lat") & (GT.measure == meas)]
        for i, pn in enumerate(ALL_PANELS):
            h = g[g.panel == pn].set_index("epoch")
            for e_i, ep in enumerate(POST):
                if ep not in h.index:
                    continue
                r, x0, c = h.loc[ep], i * 4 + e_i, EPOCH_LINE[ep]
                for q, a in enumerate(AN):
                    v = r.get(f"animal_{a}", np.nan)
                    if np.isfinite(v):
                        ax.plot(x0 - 0.3 + 0.12 * q, v, ANIMAL_MARKER[a], mfc="none", mec=c, ms=5.5, mew=1.1, zorder=3)
                if np.isfinite(r.get("point", np.nan)):
                    sig = r.lo > 0 or r.hi < 0
                    ax.errorbar(x0 + 0.3, r.point, yerr=[[r.point - r.lo], [r.hi - r.point]], fmt="o", color="k",
                                mfc=c if sig else "white", ms=5.5, capsize=2.5, zorder=4)
                    ax.text(x0 + 0.05, 1.005, f"{r.point:+.2f}{'*' if sig else ''}\n{int(r.n_dir)}/{int(r.n)}",
                            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.3,
                            fontweight="bold" if sig else "normal", color=c)
            if i:
                ax.axvline(i * 4 - 1.5, color="0.85", lw=0.8)
        ax.axhline(0, color="0.6", lw=0.7, ls="--")
        ax.set_xticks([i * 4 + 1 for i in range(len(ALL_PANELS))], [SHORT[p] for p in ALL_PANELS], fontsize=8)
        ax.set_xticks([i * 4 + e for i in range(len(ALL_PANELS)) for e in range(3)], minor=True)
        ax.tick_params(axis="x", which="major", length=0, pad=4)
        ax.set_xlim(-0.8, 4 * len(ALL_PANELS) - 1.2)
        ax.set_title(f"NEURAL: {ttl}\neach set: acute / subacute / chronic (color); open = animals, black = 95% paired "
                     "nested bootstrap (* = excludes 0), number above = point, k/n animals", fontsize=8.5, pad=30)
        ax.set_ylabel("change vs pre (mm/mm)" if meas == "slope" else "change vs pre (mm)", fontsize=8.5)
        _sty(ax)
    h1 = [Line2D([], [], color=col[e], lw=2, label=e) for e in EPS]
    h2 = [Line2D([], [], color="0.3", lw=2, label="mean of 4 animals (+/- SEM)"),
          Line2D([], [], color="0.3", lw=0.9, ls="--", label="only 3 animals")]
    h3 = [Line2D([], [], ls="", marker=ANIMAL_MARKER[a], mfc="none", mec="0.3", label=a) for a in AN]
    fig.legend(handles=h1 + h2 + h3, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=len(h1 + h2 + h3),
               fontsize=8.5, frameon=False)
    fig.suptitle("NEURAL, LIKE-FOR-LIKE, ALL LICK TYPES: decoded lateral position vs actual lateral landing place per "
                 "epoch, each lick set read by its Fig 12.2 frozen pre reader (ITI reader for ITI and ENLv licks, cued "
                 "reader for cued licks; pre = held-out sessions).\n"
                 f"Gaussian-smoothed (sigma {sg:g} mm) per animal x epoch where the weight >= 10 licks; mean +/- SEM over "
                 "animals. A curve shifted UP = read more contralesional than pre licks of the SAME set that landed at the "
                 "same place. Bars at the top of each panel = 0.25-mm landing bins whose change vs pre survives BH\n"
                 "(resid vs the animal's own pre same-set Gaussian reference; session means, >= 5 licks per bin, >= 3 "
                 "animals; rows acute / subacute / chronic). ITI 1st after quiet = first ITI lick >= 2 s after the "
                 "previous lick (the primary ITI set); cued 1st = first post-cue lick of each trial; ENLv (a) = first ENLv "
                 "lick / trial >= 1 s after spout\narrival, (b) = ENLv bout starts (>= 1 s since any lick) >= 1 s after "
                 "arrival, windows ending before the cue. Bottom: per session OLS of decoded on landing over the range "
                 "SHARED by pre and that epoch (2.5-97.5 % of each), slope = gain, value at the range center = offset. "
                 "n = 4 mice, PROVISIONAL", fontsize=9.5, y=0.995)
    save_png_svg(fig, FIG / "lick_like_for_like_all.png", dpi=100)
    plt.close(fig)


def draw(D: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_curves(D, "lat")
    fig_curves(D, "ang")
    fig_gain(D)
    with _trial_set_for(ALL_PANELS):
        fig_all(D)


# ================================================================================================ main
def run() -> dict:
    t0 = time.time()
    Q = load()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    C = counts(Q)
    C.to_csv(OUT / "curves_counts.csv", index=False)
    print("== licks per set x animal x epoch (rule 8):")
    print(C.pivot_table(index=["set", "animal"], columns="epoch", values=["n_licks", "n_sessions"])
          .reindex(columns=EPS, level=1).to_string())
    A, BT, CV, GS, GT, R, E = {}, {}, [], [], [], [], []
    for ax in AXES:
        A[ax] = curves(Q, ax)
        BT[ax] = bins(Q, ax)
        CV.append(coverage(A[ax], ax))
        S, AL = gain(Q, ax)
        GS.append(S)
        R.append(residuals(A[ax], AL, ax))
        E.append(edge(A[ax], AL, ax))
        print(f"  axis {ax} done ({time.time() - t0:.0f} s)", flush=True)
    GS = pd.concat(GS, ignore_index=True)
    GT = gain_tests(GS)
    CV, R, E = (pd.concat(x, ignore_index=True) for x in (CV, R, E))
    BTa = pd.concat(BT.values(), ignore_index=True)
    for df, f in ((CV, "curves_coverage"), (BTa, "curves_bin_tests"), (GS, "curves_gain_sessions"),
                  (GT, "curves_gain_tests"), (E, "curves_edge")):
        df.to_csv(OUT / f"{f}.csv", index=False)
    print("== curve coverage (drawn x range per animal x epoch; >=4 / >=3 animals):")
    print(CV.round(2).to_string(index=False))
    print("== BH landing bins (sig / tested):")
    print(BTa.groupby(["axis", "set", "epoch"]).agg(sig=("sig", "sum"), tested=("sig", "size"))
          .reset_index().to_string(index=False))
    print(BTa[BTa.sig].round(3).to_string(index=False))
    acols = [c for c in GT.columns if c.startswith("animal_") or c.startswith("nsess_")]
    print("== GAIN / OFFSET: per-animal change vs pre (rule 8), then the paired nested bootstrap:")
    print(GT[["axis", "panel", "epoch", "measure", "pre_mean", *acols]].round(3).to_string(index=False))
    print(GT[["axis", "panel", "epoch", "measure", "point", "lo", "hi", "n_dir", "n"]].round(3).to_string(index=False))
    print("== residual RMS over the shared range (mean over animals): diff / minus offset / minus gain + offset")
    rr = R.groupby(["axis", "panel", "epoch", "animal"])[["diff", "resid_offset", "resid_gain"]].apply(
        lambda d: np.sqrt((d ** 2).mean())).groupby(["axis", "panel", "epoch"]).mean()
    print(rr.round(3).to_string())
    print("== edge check: (epoch - pre) curve difference by zone, mean over animals (n = animals with a value)")
    ez = E.groupby(["axis", "panel", "epoch", "zone"]).agg(diff=("diff", "mean"), pre_slope=("pre_local_slope", "mean"),
                                                           ep_slope=("ep_local_slope", "mean"),
                                                           n=("diff", "count")).unstack("zone")
    print(ez.round(3).to_string())
    print("== licks per NEW set x animal x epoch (iti_q, trial_1) and ENLv sets (rule 8):")
    print(Q[Q.reader.isin(["iti", "trial"])].groupby(["panel", "animal", "epoch"]).size().unstack("epoch")
          .reindex(columns=EPS).to_string())
    FS, FT = farcontra(Q)
    FS.to_csv(OUT / "curves_farcontra_sessions.csv", index=False)
    FT.to_csv(OUT / "curves_farcontra_tests.csv", index=False)
    ac = [f"animal_{a}" for a in AN]
    nc = [f"nlick_{a}" for a in AN]
    print("== FAR-CONTRA END, cued: resid change vs pre per animal (rule 8) + licks per animal, then bootstrap")
    print(FT[["ref", "scope", "cls", "contact", "epoch", *ac, *nc, "point", "lo", "hi", "n_dir", "n"]].round(3)
          .to_string(index=False))
    D = dict(A=A, BT=BT, GT=GT, R=R, E=E, C=C, CV=CV, FT=FT)
    figdata.save_fd(STEM, D, root=FD)
    figdata.save_fd("lick_like_for_like_all", dict(A={"lat": A["lat"]}, BT={"lat": BT["lat"]}, GT=GT), root=FD)
    print(f"analysis {time.time() - t0:.0f} s", flush=True)
    return D


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())
    if figdata.plot_only() and "--all-only" in sys.argv:       # just the combined figure, from its own figdata
        FIG.mkdir(parents=True, exist_ok=True)
        with _trial_set_for(ALL_PANELS):
            fig_all(figdata.load_fd("lick_like_for_like_all", root=FD))
        return 0
    D = figdata.load_fd(STEM, root=FD) if figdata.plot_only() else run()
    draw(D)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
