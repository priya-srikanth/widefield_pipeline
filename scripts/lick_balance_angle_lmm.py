"""Hemispheric BALANCE vs LICK ANGLE, pre vs post -- mixed models per session x spout (Priya 10-28: "can we do LMM on
per-session balance (including multiple regions) and landing (for multiple spouts)?", "or maybe just balance and lick
angle", "pre vs post").

Unit = one session x spout (the spout = the target for cued licks, the PREVIOUS trial's spout for ITI licks), so each
session contributes one point per spout and the lick angle varies across spouts and sessions.
  LICK ANGLE  lateral angle of the 3-D tongue vector at the lick peak from the midline (sagittal) plane, degrees,
              asin(side_ml / reach) from the pose3d table (12.16's per-lick covariates file); + = contralesional.
  BALANCE     per pair (ALM, tjM1, SSp-m, SSp-n), two measures:
              dec  = X_right - X_left in the decoder's own window (mean 0..0.2 s from the lick peak; the decoder's inputs,
                     x1e3 dF/F, no per-lick baseline) -- the measure of 12.17
              resp = log2(contra / ipsi response AUC, baseline -1.0..-0.5 s) -- the measure of 14.9, on its primary
                     window per set (Priya 11-17): cued 0..1.0 s, ITI 0..0.5 s (iti_licks.CUED_RESP_WIN / ITI_RESP_WIN)
              (cued: first cued licks; ITI: bout starts)
              WHICH DECODER (11-16): the per-lick lateral WHERE-decoder (iti_lick_order / iti_where_decoder; ridge on
              Allen areas with the ALM / tjM1 split, lags 0..0.2 s, z-scored within session, trained on August pre
              licks; predicts lateral landing in mm) -- NOT the frozen spout-position decoder. X = its input features
              before z-scoring. Its lateral weights are largely ipsi - contra (11.13), so dec is roughly a one-number
              summary of what it reads (partly circular). dec has NO baseline: it carries the pre-lick level as well
              as the evoked response, is defined for every lick, and as a difference scales with response size;
              resp is baseline-corrected and a ratio (gain-invariant) but needs lick-free windows (first licks only).
              The two disagree on ITI licks (handoff 00-COMPACT-41).
              Two BOUNDED, baseline-corrected indices (Priya 11-17; requested with --idx, otherwise every output is as
              before), bounded_index(c, i) = (c - i) / (|c| + |i|), defined for every
              row, = (r - 1) / (r + 1) for r = c / i when both > 0 (c = X_right, contralesional; i = X_left):
              dec_idx  = on dec's own per-lick traces (the session file's iti_X / tr_X, LAGS_D -1..+1 s from the lick
                         PEAK, the decoder's inputs): each hemisphere baseline-subtracted (-1.0..-0.5 s before the PEAK,
                         lick_cued_ramp.BASE_SPOUT = resp's baseline, there before the ONSET), mean over 0..0.2 s, averaged over the session x spout
                         cell's licks, THEN the index of the two cell means (as dec = mean c - mean i; orofacial = mean of
                         the four areas per hemisphere, as resp). For an index the mean and the AUC over one window are
                         identical (the window length cancels), so dec_idx = the bounded index of the 0..0.2 s AUC.
              resp_idx = the same index of resp's two AUCs (c_<win>, i_<win> of lateral_ratio_sessions.csv; baseline
                         -1.0..-0.5 s, cued 0..1.0 s, ITI 0..0.5 s; onset-aligned first licks / bout starts >= 2 s)
              dec -> dec_idx removes the pre-lick level and the overall gain; dec_idx vs resp_idx = the window / lick set
              (0..0.2 s from the peak, KEY licks) vs the response window (onset-aligned first licks); resp vs resp_idx =
              log ratio vs bounded index. Later licks: dec_idx's baseline may hold the previous lick (descriptive only).
              --gap2 (11-17): ITI bout starts with gap >= il.GAP_MAIN (2 s, resp's rule) only, instead of >= BOUT_GAP
              (1 s), so the baseline starts >= 1 s after the previous lick (gap_keep).
MODELS (statsmodels MixedLM, random intercept per animal, ML; centered angle):
  y ~ angle * post          post = any post-stroke epoch: (1) slope of balance on angle pre-stroke; (2) angle x post =
                            does the slope change; (3) post = the shift at the SAME angle (the landing-matched balance
                            shift, the question 12.17 vs 14.9 left open)
  y ~ angle * C(epoch)      the same per epoch
Sessions x spouts are the observations (anti-conservative at 4 animals); the per-animal slopes and shifts are printed
first (rule 8; OLS per animal x period) and drawn.
Outputs <OUT_W>/balance_angle_points.csv, balance_angle_lmm.csv, balance_angle_animals.csv; figure
lick_balance_angle.png (figdata + --plot-only).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_reference as lbr  # noqa: E402
import lick_cued_ramp as cr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

PAIRS = [*lbr.PAIRS, "orofacial"]       # + the four pooled (Priya 10-28)
SETMAP = {"trial": "cued", "iti": "iti"}
# resp = lateral_ratio_sessions.csv column of each set's primary window (Priya 11-17: ITI 0..0.5 s, cued 0..1.0 s)
RESP_COL = {"cued": f"logr_{il.win_tag(il.CUED_RESP_WIN)}", "iti": f"logr_{il.win_tag(il.ITI_RESP_WIN)}"}
RESP_LAB = (f"hemispheric BALANCE (ratio) log2(contra / ipsi AUC; cued {il.win_label(il.CUED_RESP_WIN)}, "
            f"ITI {il.win_label(il.ITI_RESP_WIN)})")
MIN_LICKS = 5
EP = ["acute", "subacute", "chronic"]
# the bounded indices are REQUESTED with --idx (Priya 11-17); without it every output is byte-identical to before
MEAS_BASE = ("dec", "resp")
MEAS_IDX = ("dec_idx", "resp_idx")
# dec_idx: baseline = resp's (lick_cued_ramp.BASE_SPOUT, -1.0..-0.5 s), here before the lick PEAK on dec's own LAGS_D
# traces; window = dec's (lick_balance_reference.W, 0..0.2 s)
_BD = (il.LAGS_D >= cr.BASE_SPOUT[0]) & (il.LAGS_D <= cr.BASE_SPOUT[1])
ZC = [f"zc_{a}" for a in lbr.PAIRS] + [f"zi_{a}" for a in lbr.PAIRS]


def idx_on() -> bool:
    """--idx on the command line (this script or lick_iti_matched_markers balance-angle) adds dec_idx / resp_idx."""
    return "--idx" in sys.argv


def gap_keep(P: pd.DataFrame) -> np.ndarray:
    """--gap2 (Priya 11-17): keep only ITI bout starts (set 'iti', order 0) with gap >= il.GAP_MAIN (2 s, resp's rule)
    instead of >= il.BOUT_GAP (1 s), so the -1.0..-0.5 s baseline starts >= 1 s after the previous lick. Gap = time since
    the previous lick onset, from <OUT>/iti_dock_audit.csv (key label x k, as il.iti_subset_keep). Later licks and cued
    rows are untouched. All True without the flag. Needs P['k'] = the prediction-cache index."""
    if "--gap2" not in sys.argv:
        return np.ones(len(P), bool)
    A = pd.read_csv(il.OUT / "iti_dock_audit.csv", usecols=["label", "k", "gap"]).set_index(["label", "k"]).gap
    g = A.reindex(pd.MultiIndex.from_arrays([P.label, P.k])).to_numpy(float)
    first = ((P.set == "iti") & (P.order == 0)).to_numpy()
    keep = ~first | (g >= il.GAP_MAIN)
    print(f"--gap2: ITI bout starts kept {int((first & keep).sum())} of {int(first.sum())} (gap >= {il.GAP_MAIN:g} s; "
          f"gap missing for {int((first & ~np.isfinite(g)).sum())})", flush=True)
    return keep


def meas_list() -> tuple:
    return MEAS_BASE + (MEAS_IDX if idx_on() else ())


def bounded_index(c, i):
    """THE bounded contra / ipsi index (c - i) / (|c| + |i|) in [-1, 1] (Priya 11-17): defined for every row (NaN only
    where c = i = 0 or either is NaN); = (r - 1) / (r + 1) for r = c / i when both > 0. For one window the mean and the
    AUC give the same index (the window length cancels)."""
    c, i = np.asarray(c, float), np.asarray(i, float)
    den = np.abs(c) + np.abs(i)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (c - i) / den, np.nan)


def _session_bal_idx(label: str) -> pd.DataFrame:
    """lbr._session_bal (dec, unchanged) + per lick and area the baseline-subtracted decoder-window mean of each
    hemisphere (zc_<a> = X_right, zi_<a> = X_left), on the same traces and the same lick match as lbr."""
    import lick_covariate_reference as cv
    B = lbr._session_bal(label)
    acts = cv._acts(label)
    d = il._load(label)
    cols = list(d["cols"])
    out = []
    for st, (xk, sk) in {"iti": ("iti_X", "iti_side"), "trial": ("tr_X", "tr_side")}.items():
        act = np.asarray(acts.get(st, []), float)
        if not len(act):
            continue
        m = il.subseq_match(d[sk].astype(float), act)
        X = d[xk].astype(float)
        Z = X[:, lbr.W, :].mean(1) - X[:, _BD, :].mean(1)
        R = pd.DataFrame({"set": st, "label": label, "k": np.arange(len(act))})
        for a in lbr.PAIRS:
            for h, side in (("zc", "right"), ("zi", "left")):
                v = Z[:, cols.index(f"{a}_{side}")]
                R[f"{h}_{a}"] = np.where(m >= 0, v[np.maximum(m, 0)], np.nan)
        out.append(R)
    return B.merge(pd.concat(out, ignore_index=True), on=["set", "label", "k"], how="left", validate="one_to_one")


def session_worker():
    """Per-session fan-out worker: lbr._session_bal, or with --idx _session_bal_idx (its superset)."""
    return _session_bal_idx if idx_on() else lbr._session_bal


def bal_cols() -> list:
    """Per-lick columns averaged per session x spout."""
    return [*lbr.BAL, *(ZC if idx_on() else [])]


def add_dec_idx(S: pd.DataFrame) -> pd.DataFrame:
    """dec_idx_<pair> = bounded_index of the cell means zc_ / zi_ (orofacial = the mean of the four areas per hemisphere,
    as lick_lateral_ratio's orofacial pool). No-op without --idx."""
    if not idx_on():
        return S
    for a in lbr.PAIRS:
        S[f"dec_idx_{a}"] = bounded_index(S[f"zc_{a}"], S[f"zi_{a}"])
    S["dec_idx_orofacial"] = bounded_index(S[[f"zc_{a}" for a in lbr.PAIRS]].mean(1),
                                           S[[f"zi_{a}" for a in lbr.PAIRS]].mean(1))
    return S


def resp_tables(R: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    """resp_<pair> (the log2 ratio column of each set's primary window, colmap) per set x label x pos from a
    lick_lateral_ratio sessions table; with --idx also resp_idx_<pair> = bounded_index of the same window's AUCs
    (c_<win>, i_<win>, lick_lateral_ratio.compute)."""
    R = R[R.spout.isin(lo.POS) & R.set.isin(list(colmap))].copy()
    vals = {"y": "resp_"}
    R["y"] = [r_[colmap[st]] for st, (_i, r_) in zip(R.set, R.iterrows(), strict=True)]
    if idx_on():
        vals["yi"] = "resp_idx_"
        R["yi"] = [float(bounded_index(r_[colmap[st].replace("logr_", "c_")], r_[colmap[st].replace("logr_", "i_")]))
                   for st, (_i, r_) in zip(R.set, R.iterrows(), strict=True)]
    out = []
    for v, pre in vals.items():
        W = R.pivot_table(index=["set", "label", "spout"], columns="pair", values=v)
        W.columns = [f"{pre}{c}" for c in W.columns]
        out.append(W)
    W = out[0] if len(out) == 1 else out[0].join(out[1], how="outer")
    return W.reset_index().rename(columns={"spout": "pos"})


def points() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    P = P[gap_keep(P)].reset_index(drop=True)          # --gap2 only (k = cache index kept)
    C = pd.read_csv(il.in_path(f"covref_covariates{lo.SRC_TAG}.csv.gz"),
                    usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), session_worker())
    if fail:
        raise RuntimeError(f"balance failed for {len(fail)} sessions: {fail[:5]}")
    B = pd.concat([v for _it, v in res], ignore_index=True)
    P = P.merge(B, on=["set", "label", "k"], how="left", validate="one_to_one")
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    print("lick angle finite on", f"{P.angle.notna().mean():.3f}", "of licks")
    # first cued lick / ITI bout start (the licks of 14.9) AND the LATER licks of the same trials / bouts (order >= 1;
    # Priya 10-28: "add a 'later licks' that excludes lick 1"; decoder-window balance only -- the response-ratio AUC of
    # a later lick would overlap the next lick)
    P["cls"] = np.where(P.order == 0, "", " later")
    g = P.groupby(["set", "cls", "label", "animal", "epoch", "pos"])
    S = g[["angle", *bal_cols()]].mean().join(g.size().rename("n")).reset_index()
    S = S[S.n >= MIN_LICKS]
    S["set"] = S.set.map(SETMAP) + S.cls
    S = S.drop(columns="cls")
    S = S.merge(resp_tables(pd.read_csv(il.in_path("lateral_ratio_sessions.csv")), RESP_COL),
                on=["set", "label", "pos"], how="left")
    S = S.rename(columns={f"bal_{a}": f"dec_{a}" for a in lbr.PAIRS})
    S["dec_orofacial"] = S[[f"dec_{a}" for a in lbr.PAIRS]].mean(1)   # = the mean of the four pair balances
    S = add_dec_idx(S)
    S["post"] = (S.epoch != "pre").astype(int)
    return S


def _fit(df: pd.DataFrame, formula: str):
    import statsmodels.formula.api as smf
    return smf.mixedlm(formula, df, groups=df["animal"]).fit(reml=False)


def inside_shift(G: pd.DataFrame, min_post: int = 8) -> dict:
    """Post shift INSIDE the pre angle range: pre cells + post cells whose angle lies within the animal's own pre cells'
    range, y ~ ac + post + (1 | animal) (as lick_iti_matched_markers.farcontra). {} when < min_post post cells."""
    rng_ = G[G.post == 0].groupby("animal").angle.agg(["min", "max"])
    lo_, hi_ = G.animal.map(rng_["min"]), G.animal.map(rng_["max"])
    Gi = G[(G.post == 0) | ((G.angle >= lo_) & (G.angle <= hi_))]
    if Gi.post.sum() < min_post:
        return {}
    try:
        fi = _fit(Gi, "y ~ ac + post")
    except Exception as ex:                      # noqa: BLE001  reported in the row, not hidden
        return {"n_post_inside": int(Gi.post.sum()), "note_inside": f"FAILED {ex}"}
    return {"n_post_inside": int(Gi.post.sum()), "shift_inside_pre_range": fi.fe_params["post"],
            "p_shift_inside": fi.pvalues["post"]}


def compute() -> dict:
    S = points()
    S.to_csv(il.OUT_W / "balance_angle_points.csv", index=False)
    rows, an = [], []
    for st, G0 in S.groupby("set"):
        mu = G0[G0.epoch == "pre"].angle.mean()
        for meas in meas_list():
            for a in PAIRS:
                y = f"{meas}_{a}"
                G = G0.dropna(subset=["angle", y]).rename(columns={y: "y"})
                if G.empty:                              # later licks have no response-ratio measure
                    continue
                G["ac"] = G.angle - mu
                for (animal, ep), h in G.groupby(["animal", "epoch"]):      # rule 8: per animal x EPOCH OLS
                    if len(h) >= 6 and h.ac.std() > 0:
                        b = np.polyfit(h.ac, h.y, 1)
                        an.append({"set": st, "measure": meas, "pair": a, "animal": animal, "epoch": ep,
                                   "post": int(ep != "pre"), "slope": b[0], "at_mean_angle": b[1], "n": len(h)})
                try:
                    f = _fit(G, "y ~ ac * post")
                    fe, p_ = f.fe_params, f.pvalues
                    rows.append({"set": st, "measure": meas, "pair": a, "model": "post", "n": len(G),
                                 "slope_pre": fe["ac"], "p_slope_pre": p_["ac"],
                                 "slope_change_post": fe["ac:post"], "p_slope_change": p_["ac:post"],
                                 "shift_post_same_angle": fe["post"], "p_shift": p_["post"], "angle_centre": mu})
                    if idx_on():                           # 11-17 (--idx): added columns only
                        rows[-1].update(inside_shift(G))
                    f = _fit(G, "y ~ ac * C(epoch, Treatment('pre'))")
                    fe, p_ = f.fe_params, f.pvalues
                    for ep in EP:
                        t = f"C(epoch, Treatment('pre'))[T.{ep}]"
                        if t in fe:
                            rows.append({"set": st, "measure": meas, "pair": a, "model": ep, "n": len(G),
                                         "slope_pre": fe["ac"], "p_slope_pre": p_["ac"],
                                         "slope_change_post": fe.get(f"ac:{t}", np.nan),
                                         "p_slope_change": p_.get(f"ac:{t}", np.nan),
                                         "shift_post_same_angle": fe[t], "p_shift": p_[t], "angle_centre": mu})
                except Exception as ex:                  # reported, not hidden
                    rows.append({"set": st, "measure": meas, "pair": a, "model": f"FAILED {ex}"})
    M, A = pd.DataFrame(rows), pd.DataFrame(an)
    M.to_csv(il.OUT_W / "balance_angle_lmm.csv", index=False)
    A.to_csv(il.OUT_W / "balance_angle_animals.csv", index=False)
    pd.set_option("display.width", 230)
    print("== per animal (rule 8): OLS slope (per deg) and value at the pre-mean angle, per epoch")
    print(A.pivot_table(index=["set", "measure", "pair", "animal"], columns="epoch",
                        values=["slope", "at_mean_angle"]).reindex(columns=il.EP4, level=1).round(4).to_string())
    print("== mixed models\n" + M.round(4).to_string(index=False))
    return {"S": S, "M": M, "A": A}


TTL = {"dec": "DIFFERENCE (X_right - X_left)", "resp": "BALANCE (ratio)",
       "dec_idx": "INDEX (decoder window, baseline-corrected)", "resp_idx": "INDEX (response AUCs)"}
ROWLAB = {"cued": "CUED first lick", "cued later": "CUED LATER licks\n(order >= 1)", "iti": "ITI bout start",
          "iti later": "ITI LATER licks\n(order >= 1 in the bout)"}


def _b(v: float, p: float, fmt: str) -> str:
    """Value as text, BOLD (mathtext) when p < 0.05 (Priya 10-28: "bold significance in the titles")."""
    if not np.isfinite(v):
        return "n/a"
    s = format(v, fmt)
    return rf"$\mathbf{{{s}}}$" if np.isfinite(p) and p < 0.05 else s


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    S, M = F["S"], F["M"]
    for meas, lab in (("dec", "hemispheric DIFFERENCE X_right - X_left, 0..0.2 s, no baseline (x1e3 dF/F)"),
                      ("resp", RESP_LAB),
                      ("dec_idx", "bounded INDEX (c - i) / (|c| + |i|), 0..0.2 s from the peak,\n"
                                  f"baseline {il.win_label(cr.BASE_SPOUT)}"),
                      ("resp_idx", "bounded INDEX (c - i) / (|c| + |i|) of the response AUCs\n"
                                   f"(cued {il.win_label(il.CUED_RESP_WIN)}, ITI {il.win_label(il.ITI_RESP_WIN)}; "
                                   f"baseline {il.win_label(cr.BASE_SPOUT)})")):
        if f"{meas}_orofacial" not in S:                  # the indices only with --idx
            continue
        sets = ("cued", "cued later", "iti", "iti later") if meas.startswith("dec") else ("cued", "iti")
        fig, axs = plt.subplots(len(sets), len(PAIRS), figsize=(24, 4.75 * len(sets)), squeeze=False)
        for i, st in enumerate(sets):
            G = S[S.set == st]
            mu = G[G.epoch == "pre"].angle.mean()
            for j, a in enumerate(PAIRS):
                ax = axs[i, j]
                y = f"{meas}_{a}"
                for ep in il.EP4:
                    h = G[G.epoch == ep]
                    ax.plot(h.angle, h[y], "o", color=col[ep], ms=2.5, alpha=0.18, mec="none")
                # one fit line per epoch (Priya 10-28: "a slope for each epoch"), from y ~ angle * C(epoch)
                me = M[(M.set == st) & (M.measure == meas) & (M.pair == a) & M.model.isin(EP)].set_index("model")
                if len(me) and np.isfinite(me.slope_pre.iloc[0]):
                    sp0 = me.slope_pre.iloc[0]
                    x = np.linspace(np.nanpercentile(G.angle, 2), np.nanpercentile(G.angle, 98), 50)
                    # intercepts are not stored; the pre line goes through the pre group mean, epochs = pre + shift
                    pre = G[G.post == 0]
                    b0 = np.nanmean(pre[y]) - sp0 * (np.nanmean(pre.angle) - mu)
                    ax.plot(x, b0 + sp0 * (x - mu), color="k", lw=1.7, zorder=4)
                    for ep in EP:
                        if ep in me.index:
                            r = me.loc[ep]
                            ax.plot(x, b0 + r.shift_post_same_angle + (sp0 + r.slope_change_post) * (x - mu),
                                    color=col[ep], lw=1.4, ls="--", zorder=4)
                    sc = " / ".join(_b(me.slope_change_post.get(e, np.nan), me.p_slope_change.get(e, np.nan), "+.3f")
                                    for e in EP)
                    sh = " / ".join(_b(me.shift_post_same_angle.get(e, np.nan), me.p_shift.get(e, np.nan), "+.2f")
                                    for e in EP)
                    ax.set_title(f"{a}, {st}: pre slope {_b(sp0, me.p_slope_pre.iloc[0], '+.3f')} per deg\n"
                                 f"slope change a/s/c {sc}\nshift at same angle a/s/c {sh}", fontsize=8)
                ax.axhline(0, color="0.8", lw=0.6)
                ax.axvline(0, color="0.8", lw=0.6)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if i == len(sets) - 1:
                    ax.set_xlabel("lick angle from midline (deg, + = contralesional)", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(f"{ROWLAB[st]}\n{lab}", fontsize=8)
        # ONE y scale per column (Priya 11-16: "make the ITI scale more similar to the cued first lick scale so the
        # relationship is clearer"): every row takes the cued row's 1-99th percentile range (+10 %); points beyond it are
        # counted in the panel, not dropped from any fit.
        for j, a in enumerate(PAIRS):
            v = S.loc[S.set == "cued", f"{meas}_{a}"].dropna()
            if not len(v):
                continue
            lo, hi = np.nanpercentile(v, [1, 99])
            pad = 0.1 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            for i, st in enumerate(sets):
                ax = axs[i, j]
                ax.set_ylim(lo, hi)
                w = S.loc[S.set == st, f"{meas}_{a}"].dropna()
                off = int(((w < lo) | (w > hi)).sum())
                if off:
                    ax.text(0.99, 0.02, f"{off} of {len(w)} points off scale", transform=ax.transAxes, ha="right",
                            va="bottom", fontsize=7, color="0.4")
        fig.legend(handles=[Line2D([], [], color=col[e], marker="o", lw=0, label=e) for e in il.EP4]
                   + [Line2D([], [], color="k", lw=1.7, label="pre fit")]
                   + [Line2D([], [], color=col[e], lw=1.4, ls="--", label=f"{e} fit") for e in EP],
                   loc="lower center", ncol=8, fontsize=9, frameon=False)
        fig.suptitle(f"Hemispheric {TTL[meas]} vs lick angle by epoch ({meas}): one point per session x spout (cued: target; "
                     "ITI: previous trial's spout); mixed model y ~ angle * epoch + (1 | animal), sessions x spouts as "
                     "observations\n(anti-conservative at 4 animals; per-animal x epoch slopes in balance_angle_animals.csv). "
                     "Titles: a/s/c = acute / subacute / chronic vs pre; BOLD = p < 0.05; 'shift at same angle' = epoch "
                     "minus pre at the pre-mean angle. PROVISIONAL", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.04, 1, 0.92))
        save_png_svg(fig, il.OUT_W / "figures" / f"lick_balance_angle_{meas}.png", dpi=100)
        plt.close(fig)


def trial_set_sentence() -> str:
    """The lick sets of lick_balance_angle_lmm (Priya 11-09): the KEY decoder's lick set split into first / later licks
    (cued: lick_idx 0 vs >= 1; ITI: bout start vs later bout licks), session x spout cells with >= MIN_LICKS licks; the
    response-ratio (resp) measure = lick_lateral_ratio's first cued licks and ITI bout starts."""
    import lick_covariate_reference as cv
    cell = f"session x spout cells with >= {MIN_LICKS} licks"
    return cv.key_lick_set(
        iti_extra=[f"split: bout start (index 0, bouts split at onset gaps >= {il.BOUT_GAP:g} s) vs later bout licks",
                   cell, f"resp measure: bout starts with >= {il.GAP_MAIN:g} s since the last lick only"],
        cued_extra=["split: first post-cue lick (lick_idx 0) vs later licks", cell,
                    "resp measure: first post-cue licks only"])


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    if plot_only():
        plot(load_fd("lick_balance_angle"))
        return 0
    F = compute()
    save_fd("lick_balance_angle", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
