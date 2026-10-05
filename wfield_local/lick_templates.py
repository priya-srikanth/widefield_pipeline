"""Lick-triggered cortical TEMPLATE MATCHING across sessions: does a post-stroke lick's activity pattern resemble
pre-stroke licks to the same TARGET (cued spout position) or pre-stroke licks that went the same WAY (executed tongue
angle)? Pure numpy / pandas; the session I/O lives in `scripts/lick_template_match.py`.

Priya, 2026-10-05: acute post-stroke licks "triggered with far R spout but executed leftward or centrally". First
result (DECISIONS 2026-10-05): PS93 acute 0821, the 12 far_R-cued licks executed toward the mouse's LEFT matched
pre-stroke licks of the same executed angle better than pre-stroke far_R licks, movement-removed activity:
executed-minus-target r = +0.13 (bootstrap 95 % CI +0.015 .. +0.25, Wilcoxon p = 0.027); raw +0.05 (n.s.). To be
expanded on whole-session predictions.

Pieces
  area_signals         LocaNMF component dF/F -> mean per Allen region (common across sessions), robust-clipped
  lick_vectors         region x time pattern per lick (tongue onset + window, minus a baseline: just before the
                       onset, or before the trial's cue -- the pre-onset window falls inside the previous lick in fast
                       bouts)
  position_templates   mean pattern per target position (pre-stroke licks)
  compare              per post lick: r with the pre template of its cued position (TARGET), r with the mean of pre
                       licks within +-tol deg of its executed angle (EXECUTED), r with every position template, best
                       match, executed-angle class relative to the pre contact-lick direction of that position
  own_template_accuracy  control: how often a pre lick best-matches its own position (leave-one-out)
  paired_stats         mean of (executed - target), bootstrap CI, Wilcoxon
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POS_ORDER = ["far_L", "close_L", "far_center", "close_center", "close_R", "far_R"]
DEFAULTS = {"win_s": [0.0, 0.6], "base_s": [-0.2, 0.0], "baseline": "pre_onset", "precue_base_s": [-0.5, 0.0],
            "angle_tol_deg": 6.0, "class_deg": 10.0, "min_matched": 5, "clip_mad": 8.0, "n_boot": 5000}


def params(overrides: dict | None = None) -> dict:
    from wfield_local import config
    p = config._deep_merge(DEFAULTS, config.defaults().get("lick_template_match", {}) or {})
    return config._deep_merge(p, overrides) if overrides else p


def area_signals(Ydff: np.ndarray, reg: np.ndarray, clip_mad: float = 8.0):
    """(T, n_regions) mean dF/F per region label, clipped to median +- clip_mad * MAD per region (single-frame
    artefacts), and the region labels."""
    labs = np.unique(reg)
    S = np.column_stack([Ydff[:, reg == lab].mean(axis=1) for lab in labs])
    med = np.nanmedian(S, axis=0)
    mad = np.nanmedian(np.abs(S - med), axis=0) * 1.4826
    return np.clip(S, med - clip_mad * mad, med + clip_mad * mad), labs


def lick_vectors(A: np.ndarray, frame_times_s, onset_s, *, win_s=(0.0, 0.6), base_s=(-0.2, 0.0),
                 baseline: str = "pre_onset", cue_s=None, precue_base_s=(-0.5, 0.0)):
    """Per lick a flattened (region-major) pattern: A over onset + win_s minus a per-region baseline -- the mean over
    onset + base_s ('pre_onset') or over cue + precue_base_s of the lick's trial ('pre_cue', needs ``cue_s`` per
    lick). Returns (vectors (n_licks, n_regions * n_t) with NaN rows for unusable licks, ok mask)."""
    from wfield_local.movement_encoding import frame_fs, nearest_frame
    ft = np.asarray(frame_times_s, float)
    fs = frame_fs(ft)
    w = np.arange(int(round(win_s[0] * fs)), int(round(win_s[1] * fs)) + 1)
    f = nearest_frame(onset_s, ft)
    if baseline == "pre_onset":
        bw = np.arange(int(round(base_s[0] * fs)), int(round(base_s[1] * fs)))
        fb = f
    elif baseline == "pre_cue":
        if cue_s is None:
            raise ValueError("baseline 'pre_cue' needs cue_s per lick")
        bw = np.arange(int(round(precue_base_s[0] * fs)), int(round(precue_base_s[1] * fs)))
        fb = nearest_frame(cue_s, ft)
    else:
        raise ValueError(f"unknown baseline {baseline!r}")
    n_out = A.shape[1] * len(w)
    V = np.full((len(f), n_out), np.nan)
    ok = np.zeros(len(f), bool)
    for i, (fi, fbi) in enumerate(zip(f, fb)):
        if fi < 0 or fbi < 0 or fi + w.max() >= len(ft) or fbi + bw.min() < 0 or fbi + bw.max() >= len(ft):
            continue
        seg, base = A[fi + w], A[fbi + bw]
        if np.isfinite(seg).all() and np.isfinite(base).all():
            V[i] = (seg - base.mean(axis=0)).T.ravel()
            ok[i] = True
    return V, ok


def corr(a, b) -> float:
    a, b = np.asarray(a, float) - np.mean(a), np.asarray(b, float) - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def position_templates(V: np.ndarray, positions) -> dict:
    pos = np.asarray(positions)
    return {p: V[pos == p].mean(axis=0) for p in POS_ORDER if (pos == p).any()}


def own_template_accuracy(V: np.ndarray, positions) -> float:
    """Fraction of licks whose best-correlated position template (leave-one-out for their own) is their own."""
    pos = np.asarray(positions)
    hits = []
    for i in range(len(V)):
        rs = {}
        for p in POS_ORDER:
            m = pos == p
            if p == pos[i]:
                m = m.copy()
                m[i] = False
            if m.any():
                rs[p] = corr(V[i], V[m].mean(axis=0))
        hits.append(max(rs, key=lambda k: -np.inf if np.isnan(rs[k]) else rs[k]) == pos[i])
    return float(np.mean(hits)) if hits else np.nan


def compare(pre: pd.DataFrame, Vpre: np.ndarray, post: pd.DataFrame, Vpost: np.ndarray, *, angle_tol_deg=6.0,
            class_deg=10.0, min_matched=5) -> pd.DataFrame:
    """Per post lick (``post`` rows: position, angle, contact; ``Vpost`` its patterns) against the pre licks.
    ref angle per position = median executed angle of the PRE contact licks there; class = 'toward mouse-LEFT'
    (angle - ref > +class_deg: image-right on cam4), 'toward mouse-right' (< -class_deg) or 'on-target'."""
    tmpl = position_templates(Vpre, pre.position)
    ref = {p: float(pre.loc[(pre.position == p) & (pre.contact == True), "angle"].median())    # noqa: E712
           for p in tmpl}
    rows = []
    for (_, r), v in zip(post.iterrows(), Vpost):
        near = (np.abs(pre.angle.to_numpy() - r.angle) <= angle_tol_deg)
        rp = {p: corr(v, t) for p, t in tmpl.items()}
        dev = r.angle - ref.get(r.position, np.nan)
        cls = ("toward mouse-LEFT" if dev > class_deg else "toward mouse-right" if dev < -class_deg
               else "on-target" if np.isfinite(dev) else "")
        rows.append({"position": r.position, "angle": r.angle, "dev_vs_pre": dev, "cls": cls,
                     "contact": r.contact, "r_target": rp.get(r.position, np.nan),
                     "r_executed": corr(v, Vpre[near].mean(axis=0)) if near.sum() >= min_matched else np.nan,
                     "n_matched": int(near.sum()),
                     "matched_mostly": pre.position[near].value_counts().index[0] if near.any() else "",
                     "best_position": max(rp, key=rp.get) if rp else "", **{f"r_{k}": x for k, x in rp.items()}})
    return pd.DataFrame(rows)


def paired_stats(d, n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean of paired differences, bootstrap 95 % CI, Wilcoxon signed-rank p (n >= 6; else NaN)."""
    from scipy import stats
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return {"n": len(d), "mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_wilcoxon": np.nan}
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    p = float(stats.wilcoxon(d).pvalue) if len(d) >= 6 else np.nan
    return {"n": len(d), "mean": float(d.mean()), "ci_lo": float(np.percentile(bs, 2.5)),
            "ci_hi": float(np.percentile(bs, 97.5)), "p_wilcoxon": p}
