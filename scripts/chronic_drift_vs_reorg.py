"""Drift vs reorganization, per animal.

Is the still-rising chronic crossnobis-from-pre representational DRIFT (the ordinary week-scale
turnover a behaviourally stable animal shows) or ongoing stroke REORGANIZATION?

A fixed pre-stroke template cannot separate the two: both grow distance-from-pre over time. So this
instead measures the per-day RATE of session-to-session change in two windows, in the SAME
pre-stroke between-position units the 8-series uses:

  m_pre  = slope of pairwise crossnobis(pre_i, pre_j)   vs |day_i - day_j|   (the DRIFT FLOOR)
  m_chr  = slope of pairwise crossnobis(chr_i, chr_j)   vs |day_i - day_j|   (within-chronic rate)

  m_chr ~= m_pre            -> the chronic representation is drifting at the baseline rate: the
                              rising distance-from-pre is DRIFT, not reorganization.
  m_chr  >  m_pre (CI>0)    -> the chronic representation is still changing FASTER than baseline:
                              residual REORGANIZATION on top of drift.

Pairwise distance = mean own-position (diagonal) of grant_geometry._crossnobis_cross, normalized by
each animal's pre-stroke between-position scale (mean triu of _crossnobis_within on the pooled pre
reference) -- identical units to epoch_8diag. Pre coverage is sparse but spans ~2 months, which is
exactly the range of day-gaps a drift slope needs; we read the SLOPE, not the intercept.

cue/working to match the deck's crossnobis readout (epoch_8diag_matrices_crossnobis_cue_working).

    python chronic_drift_vs_reorg.py [--out DIR] [--n-split N] [--n-boot N]
"""
from __future__ import annotations

import argparse
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wfield_local.grant_kit import _collect_7, _pre_reference, CONF_LABELS
from wfield_local.grant_geometry import _crossnobis_cross, _crossnobis_within, _triu_vals
from wfield_local.epoch_figures import epoch_of_day
from wfield_local.epochs import days_since_stroke

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
ALIGN, VARIANT, MIN_TRIALS = "cue", "working", 10


def _pair_dist(A, B, scale, n_split, seed):
    """Mean own-position (diagonal) crossnobis between two sessions' pattern dicts, normalized.

    Averaged over n_split random half-splits to tame the single-split noise of the estimator.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_split):
        D = _crossnobis_cross(A, B, rng, CONF_LABELS)
        d = np.diag(D)
        if np.isfinite(d).any():
            vals.append(np.nanmean(d))
    if not vals:
        return np.nan
    return float(np.nanmean(vals)) / scale


def _scale_for(full_ref, n_split, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_split):
        s = np.nanmean(_triu_vals(_crossnobis_within(full_ref, rng, CONF_LABELS)))
        if np.isfinite(s):
            vals.append(s)
    s = float(np.nanmean(vals)) if vals else np.nan
    return s if (np.isfinite(s) and s > 0) else 1.0


def _slope(gaps, dists):
    g = np.asarray(gaps, float)
    y = np.asarray(dists, float)
    ok = np.isfinite(g) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan
    m, c = np.polyfit(g[ok], y[ok], 1)
    return float(m), float(c)


def _boot_slopes(idx, pair, n_boot, rng):
    """Bootstrap the slope by resampling SESSIONS (session = the unit), reusing precomputed pairs.

    pair: dict {(i,j): (gap, dist)} for i<j over session indices in idx.
    """
    out = []
    n = len(idx)
    if n < 3:
        return np.array([])
    for _ in range(n_boot):
        samp = rng.choice(n, size=n, replace=True)
        gaps, dists = [], []
        for a, b in combinations(sorted(set(samp.tolist())), 2):
            key = (idx[a], idx[b]) if idx[a] < idx[b] else (idx[b], idx[a])
            g, d = pair.get(key, (np.nan, np.nan))
            if np.isfinite(g) and np.isfinite(d):
                gaps.append(g); dists.append(d)
        m, _c = _slope(gaps, dists)
        if np.isfinite(m):
            out.append(m)
    return np.asarray(out)


def _pairs(sessions, scale, n_split):
    """sessions: list of (key, day, patterns). Returns dict {(ka,kb):(gap,dist)} and the per-session days."""
    pair = {}
    for (ka, da, pa), (kb, db, pb) in combinations(sessions, 2):
        key = (ka, kb) if ka < kb else (kb, ka)
        gap = abs(da - db)
        seed = abs(hash((key, "pair"))) % (2**32)
        pair[key] = (gap, _pair_dist(pa, pb, scale, n_split, seed))
    return pair


def analyse(n_split, n_boot, log=print):
    store, _days = _collect_7(ALIGN, VARIANT, MIN_TRIALS)
    rows, curves = [], {}
    for an in ANIMALS:
        if an not in store:
            log(f"{an}: absent from collector"); continue
        pre_by_sess, by_day = store[an]
        full_ref = _pre_reference(pre_by_sess)
        scale = _scale_for(full_ref, n_split, seed=abs(hash((an, "scale"))) % (2**32))

        # PRE sessions: key by MMDD, day = days_since_stroke (negative)
        pre = []
        for mmdd, pat in pre_by_sess.items():
            d = days_since_stroke(f"{an}_{mmdd}")
            if d is not None:
                pre.append((f"pre{mmdd}", int(d), pat))
        # CHRONIC sessions: by_day already keyed by days-since-stroke
        chr_ = []
        for day, pat in by_day.items():
            if epoch_of_day(an, int(day)) == "chronic":
                chr_.append((f"chr{int(day)}", int(day), pat))
        pre.sort(key=lambda x: x[1]); chr_.sort(key=lambda x: x[1])
        log(f"{an}: scale={scale:.4f}  pre_sessions={len(pre)}  chronic_sessions={len(chr_)}")

        pre_pair = _pairs(pre, scale, n_split)
        chr_pair = _pairs(chr_, scale, n_split)

        def flat(pairdict):
            gs = [g for g, d in pairdict.values() if np.isfinite(g) and np.isfinite(d)]
            ds = [d for g, d in pairdict.values() if np.isfinite(g) and np.isfinite(d)]
            return gs, ds

        gpre, dpre = flat(pre_pair)
        gchr, dchr = flat(chr_pair)
        m_pre, c_pre = _slope(gpre, dpre)
        m_chr, c_chr = _slope(gchr, dchr)

        rng = np.random.default_rng(abs(hash((an, "boot"))) % (2**32))
        bp = _boot_slopes([s[0] for s in pre], pre_pair, n_boot, rng)
        bc = _boot_slopes([s[0] for s in chr_], chr_pair, n_boot, rng)
        # delta on paired bootstrap draws (independent windows -> pair by draw index)
        k = min(len(bp), len(bc))
        delta = (bc[:k] - bp[:k]) if k else np.array([])

        def ci(a):
            return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)
        mpre_lo, mpre_hi = ci(bp)
        mchr_lo, mchr_hi = ci(bc)
        dl, dh = ci(delta)

        verdict = "n/a"
        if np.isfinite(dl) and np.isfinite(dh):
            if dl > 0:
                verdict = "REORG (chronic faster than baseline drift)"
            elif dh < 0:
                verdict = "chronic SLOWER than pre-drift"
            else:
                verdict = "DRIFT-consistent (chronic ~ baseline drift)"

        # vs-pre per-session (for the overlay): distance to pooled pre, normalized
        vspre = []
        for key, day, pat in chr_:
            seed = abs(hash((an, key, "vspre"))) % (2**32)
            vspre.append((day, _pair_dist(pat, full_ref, scale, n_split, seed)))

        rows.append(dict(animal=an, scale=round(scale, 4),
                         n_pre_sessions=len(pre), n_chronic_sessions=len(chr_),
                         m_pre=round(m_pre, 5), m_pre_lo=round(mpre_lo, 5), m_pre_hi=round(mpre_hi, 5),
                         m_chronic=round(m_chr, 5), m_chr_lo=round(mchr_lo, 5), m_chr_hi=round(mchr_hi, 5),
                         delta_chr_minus_pre=round(m_chr - m_pre, 5), delta_lo=round(dl, 5), delta_hi=round(dh, 5),
                         intercept_pre=round(c_pre, 4), intercept_chr=round(c_chr, 4),
                         verdict=verdict))
        curves[an] = dict(gpre=gpre, dpre=dpre, gchr=gchr, dchr=dchr,
                          m_pre=m_pre, c_pre=c_pre, m_chr=m_chr, c_chr=c_chr, vspre=vspre)
    return pd.DataFrame(rows), curves


def figure(df, curves, out_png):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    for ax, an in zip(axes, ANIMALS):
        if an not in curves:
            ax.set_visible(False); continue
        c = curves[an]
        ax.scatter(c["gpre"], c["dpre"], s=14, c="0.6", label="pre-pre (drift floor)", zorder=2)
        ax.scatter(c["gchr"], c["dchr"], s=16, c="tab:blue", label="chronic-chronic", zorder=3)
        xs = np.array([0, max([*c["gpre"], *c["gchr"], 1])])
        if np.isfinite(c["m_pre"]):
            ax.plot(xs, c["c_pre"] + c["m_pre"] * xs, "-", color="0.4", lw=1.5)
        if np.isfinite(c["m_chr"]):
            ax.plot(xs, c["c_chr"] + c["m_chr"] * xs, "-", color="tab:blue", lw=1.5)
        r = df[df.animal == an].iloc[0]
        ax.set_title(f"{an}\nm_pre={r.m_pre:.4f} [{r.m_pre_lo:.4f},{r.m_pre_hi:.4f}]\n"
                     f"m_chr={r.m_chronic:.4f} [{r.m_chr_lo:.4f},{r.m_chr_hi:.4f}]\n"
                     f"Δ=[{r.delta_lo:.4f},{r.delta_hi:.4f}]", fontsize=8)
        ax.set_xlabel("|day gap| between sessions", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("crossnobis distance\n(pre between-position units)", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Within-window representational change rate: pre-stroke drift floor vs within-chronic "
                 "(crossnobis per day-gap, cue/working)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-split", type=int, default=15)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.out:
        out_dir = Path(a.out)
    else:
        from wfield_local.paths import PathResolver
        out_dir = Path(PathResolver().root("labcams")) / "chronic_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df, curves = analyse(a.n_split, a.n_boot)
    df.to_csv(out_dir / "chronic_drift_vs_reorg.csv", index=False)
    figure(df, curves, out_dir / "chronic_drift_vs_reorg.png")
    print("\n==== DRIFT vs REORGANIZATION (cue/working) ====")
    cols = ["animal", "n_pre_sessions", "n_chronic_sessions", "m_pre", "m_chronic",
            "delta_chr_minus_pre", "delta_lo", "delta_hi", "verdict"]
    print(df[cols].to_string(index=False))
    print(f"\nwrote {out_dir / 'chronic_drift_vs_reorg.csv'} and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
