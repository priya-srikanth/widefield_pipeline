"""GRANT FIGURES — GEOMETRY

Representational geometry: crossnobis, asymmetry, position structure, recovery trajectory.

Split out of `grant_figures` on 2026-09-21, one module per figure family. THE GROUPING IS FROM THE
CALL GRAPH, not from the names: every function here is reached from this family's entry points and
from no other family's. Anything shared with a sibling lives in `grant_kit` -- which is why this
imports from there and never from `grant_figures`, a direction that would be a cycle.

Entry points, registered in `grant_figures.JOBS`:
  - `fig_crossnobis_cross`
  - `fig_crossnobis_geometry`
  - `fig_crossnobis_delta`
  - `fig_asymmetry`
  - `fig_geometry_by_position`
  - `fig_delta_trajectory`
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wfield_local import config
from wfield_local.grant_kit import (  # noqa: F401
    _BUNDLE_CACHE,
    ANIMALS,
    BOOT_CACHE_VERSION,
    CONF_LABELS,
    N_BOOT_DELTA,
    N_BOOT_RDM,
    N_LOO_DRAW,
    POS,
    POS_SHORT,
    WINDOWS,
    _anchor,
    _best_match,
    _block_boot,
    _block_index,
    _boot_cached,
    _cd_labels,
    _class_note,
    _class_select,
    _collect_7,
    _colw,
    _corr_matrix,
    _day,
    _delta_cis,
    _delta_diag_ci,
    _delta_diag_one,
    _delta_grid,
    _diag,
    _digest,
    _excludes_zero,
    _feed,
    _fig_root,
    _fit_bottom,
    _fit_header,
    _footer,
    _impaired,
    _matrices_pattern,
    _mats_pattern,
    _means,
    _nanmean_stack,
    _out,
    _overlaps,
    _pct3,
    _pooled_bundle,
    _position_metrics,
    _pre_reference,
    _runs_to_blocks,
    _save,
    _seed,
    _session_trials,
    _sessions,
    _sg_labels,
    _short,
    _suptitle,
    _twinned,
    _txt,
    _variants,
    _windows,
    coverage_note,
    only,
    pos_style,
    set_only,
)


def _whitener(res):
    """Inverse residual covariance, Ledoit-Wolf shrunk, with a diagonal fallback.

    Estimated from trials MINUS their own cell mean, so it is independent of the differences it
    later whitens -- which is what keeps the cross-validated product unbiased.
    """
    if not res:
        return None
    R = np.vstack(res)
    if len(R) < 3:
        return None
    try:
        from sklearn.covariance import LedoitWolf
        return np.linalg.pinv(LedoitWolf().fit(R).covariance_)
    except Exception:                                                # noqa: BLE001
        return np.diag(1.0 / np.maximum(R.var(axis=0), 1e-12))
def _halves(src, rng, labels, min_n=4):
    """Per-position (half A mean, half B mean) plus the residuals both halves leave behind."""
    m0, m1, res = {}, {}, []
    for q in labels:
        Z = src.get(q)
        if Z is None or len(Z) < min_n:
            continue
        idx = rng.permutation(len(Z))
        h = len(Z) // 2
        A, B = Z[idx[:h]], Z[idx[h:2 * h]]
        m0[q], m1[q] = A.mean(0), B.mean(0)
        res.append(A - m0[q])
        res.append(B - m1[q])
    return m0, m1, res
def _crossnobis_within(src, rng, labels):
    """Noise-unbiased 6x6 RDM within ONE set of trials. Diagonal is 0 by construction, left NaN."""
    m0, m1, res = _halves(src, rng, labels)
    P = _whitener(res)
    if P is None or len(m0) < 2:
        return np.full((len(labels), len(labels)), np.nan)
    D = np.full((len(labels), len(labels)), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i >= j or a not in m0 or b not in m0:
                continue
            D[i, j] = D[j, i] = float((m0[a] - m0[b]) @ P @ (m1[a] - m1[b]))
    return D
def _crossnobis_cross(post, ref, rng, labels):
    """Unbiased d(post at P, reference at Q) for every ordered pair -- figure 6's matrix as distance.

    The two factors of the product use DISJOINT halves on both sides (post half A against reference
    half 1, post half B against reference half 2), so trial noise contributes zero in expectation
    and the diagonal is an unbiased estimate of how far the post-stroke pattern has actually moved.
    A raw squared distance would instead grow with noise alone, which is the bias this removes and
    the reason a plain distance version of figure 6 would have been unreadable across sessions whose
    amplitude differs 2-3x.
    """
    pm0, pm1, pres = _halves(post, rng, labels)
    rm0, rm1, rres = _halves(ref, rng, labels)
    P = _whitener(pres + rres)
    if P is None:
        return np.full((len(labels), len(labels)), np.nan)
    D = np.full((len(labels), len(labels)), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if a not in pm0 or b not in rm0:
                continue
            D[i, j] = float((pm0[a] - rm0[b]) @ P @ (pm1[a] - rm1[b]))
    return D
def _triu_vals(D):
    iu = np.triu_indices(D.shape[0], 1)
    return D[iu]
def _lw_cov(R):
    """Ledoit-Wolf shrunk covariance, computed directly instead of through sklearn.

    Bit-identical to ``LedoitWolf().fit(R).covariance_`` -- asserted, not assumed, in
    ``tests/test_fast_rdm.py`` -- and about four times faster. sklearn stays the route the point
    estimates take; this exists only so a per-draw whitener is affordable inside a bootstrap.
    """
    n, p = R.shape
    X = R - R.mean(0)
    S = X.T @ X / n
    mu = np.trace(S) / p
    d2 = float(((S - mu * np.eye(p)) ** 2).sum() / p)
    if d2 <= 0:
        return S
    b2 = min(float((((X ** 2).T @ (X ** 2)) / n - S ** 2).sum() / p / n), d2)
    return (b2 / d2) * mu * np.eye(p) + (1 - b2 / d2) * S
def _fast_rdm(src, rng, labels):
    """`_crossnobis_within` without the 380x380 pseudo-inverse. THE SAME ESTIMATOR, not a new one.

    Every quadratic form (m0_a - m0_b) P (m1_a - m1_b) shares one right-hand side, so with
    A = M0 C^-1 M1' the entry is A[a,a] - A[a,b] - A[b,a] + A[b,b] and ONE Cholesky solve replaces a
    pinv per call: ~25 ms against ~125 ms. That is the difference between a bootstrap that finishes
    and one that runs for four hours.

    IT HAD TO BE THE SAME ESTIMATOR. The obvious speed-up -- fix the whitener once per animal and
    reuse it across draws, as `_mats_crossnobis` does for the distance figures -- is not available
    here: measured over all four animals it moves 8b's whole-RDM correlation by up to 0.60 and its
    post-minus-PRE contrast from -0.07 to -0.51 (DECISIONS.md, 2026-08-26). An interval computed
    under a different whitener rule would not describe the number printed beside it, which is
    exactly how figure 8d's "+0.28 [+6.63, +69.41]" announced itself.
    """
    m0, m1, res = _halves(src, rng, labels)
    n = len(labels)
    D = np.full((n, n), np.nan)
    if len(m0) < 2 or not res:
        return D
    R = np.vstack(res)
    if len(R) < 3:
        return D
    keys = [q for q in labels if q in m0]
    M0 = np.stack([m0[q] for q in keys])
    M1 = np.stack([m1[q] for q in keys])
    try:
        from scipy.linalg import cho_factor, cho_solve
        A = M0 @ cho_solve(cho_factor(_lw_cov(R)), M1.T)
    except Exception:                                                # noqa: BLE001
        # SAME FALLBACK AS `_whitener`, so a box without scipy or sklearn degrades identically
        # rather than silently producing a second kind of number.
        P = _whitener(res)
        if P is None:
            return D
        A = M0 @ P @ M1.T
    at = {q: i for i, q in enumerate(keys)}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i >= j or a not in at or b not in at:
                continue
            x, y = at[a], at[b]
            D[i, j] = D[j, i] = float(A[x, x] - A[x, y] - A[y, x] + A[y, y])
    return D
def _rdm_scores(D, Dref):
    """(whole-RDM r, per-position row r) for one RDM against a reference RDM.

    ONE DEFINITION for figure 8b, figure 8g and the bootstrap that puts intervals on both. It was
    written out three times; a per-position row that means one thing in the heatmap and another in
    the trajectory is precisely the divergence this repo has already been bitten by.
    """
    a, b = _triu_vals(D), _triu_vals(Dref)
    ok = np.isfinite(a) & np.isfinite(b)
    whole = (float(np.corrcoef(a[ok], b[ok])[0, 1])
             if ok.sum() >= 4 and np.std(a[ok]) and np.std(b[ok]) else np.nan)
    rows = np.full(D.shape[0], np.nan)
    for i in range(D.shape[0]):
        ra, rb = np.delete(D[i], i), np.delete(Dref[i], i)
        m = np.isfinite(ra) & np.isfinite(rb)
        # A ROW IS FIVE NUMBERS. Below four usable ones a correlation is not an estimate of
        # anything, so the cell stays blank rather than printing an r built from three points.
        if m.sum() >= 4 and np.std(ra[m]) and np.std(rb[m]):
            rows[i] = float(np.corrcoef(ra[m], rb[m])[0, 1])
    return whole, rows
def fig_crossnobis_cross(out_dir, min_trials=10):
    """8: figure 6 rebuilt on cross-validated (crossnobis) distances instead of correlations.

    ROWS = the post-stroke position, COLUMNS = the pre-stroke reference position, matching figure 6.
    The DIAGONAL is what changed: near zero means the pattern did not move, and unlike a correlation
    it is not bounded by how repeatable either mean was -- the cross-validated product is unbiased by
    trial noise, so a session with more variable responses does not automatically read as a bigger
    distance. That is the one thing figure 6 cannot do and the reason this exists.

    UNITS. Distances are divided by the mean pre-stroke within-set pairwise distance for that animal
    and window, so 1.0 = "as far apart as two different pre-stroke positions were". Raw crossnobis
    units depend on the whitener and the dimensionality and are not comparable across animals.

    WHAT IT STILL CANNOT DO: it is a distance between two patterns, so a uniform post-stroke gain
    change moves every cell -- the same exposure figure 6 has. 8b is the gain-invariant companion.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            store, days = _collect_7(align, v, min_trials)
            if not days:
                continue
            ncol = 1 + len(days)
            # Height and hspace raised with `_colw` -- see the note in `_draw_5c`.
            fig, axes = plt.subplots(len(ANIMALS), ncol, figsize=(_colw() * ncol + 1.2, 9.5),
                                     gridspec_kw={"hspace": 0.60},
                                     squeeze=False)
            im = None
            for ri, an in enumerate(ANIMALS):
                got = store.get(an)
                if not got:
                    for ci in range(ncol):
                        axes[ri][ci].axis("off")
                    continue
                pre_by_sess, by_day = got
                rng = np.random.default_rng(_seed(an, align, v))
                full_ref = _pre_reference(pre_by_sess)
                scale = np.nanmean(_triu_vals(_crossnobis_within(full_ref, rng, CONF_LABELS)))
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
                for ci in range(ncol):
                    ax = axes[ri][ci]
                    # COLUMN 0 IS THE NO-LESION EXPECTATION, LEAVE-ONE-SESSION-OUT: each
                    # pre-stroke session scored against the pool of the others, averaged. One
                    # session against a pool, exactly like every post-stroke column, so the columns
                    # are comparable; disjoint, so nothing is scored against itself. Its diagonal is
                    # what "did not move" looks like measured this way, and it is not 0.
                    if ci == 0:
                        Ds = [_crossnobis_cross(pat, _pre_reference(pre_by_sess, exclude=s),
                                                rng, CONF_LABELS)
                              for s, pat in pre_by_sess.items()]
                        if not Ds:
                            ax.axis("off")
                            continue
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            D = np.nanmean(np.stack(Ds), axis=0) / scale
                    else:
                        src = by_day.get(days[ci - 1])
                        if not src:
                            ax.axis("off")
                            continue
                        D = _crossnobis_cross(src, full_ref, rng, CONF_LABELS) / scale
                    # SALIENCE FOLLOWS THE RESULT. `magma_r` at vmax=2.0 made LARGE distances
                    # darkest -- so the eye landed on off-diagonal noise while the finding (a LOW
                    # diagonal = the pattern did not move) faded to pale yellow. It also clipped:
                    # PS93 day 7 holds 3.3-3.6 and every such cell rendered identically black.
                    # `magma` at vmax=2.5 puts the bright end on SMALL distances and leaves the top
                    # of the range distinguishable.
                    im = ax.imshow(np.ma.masked_invalid(D), vmin=0, vmax=2.5, cmap="magma")
                    # NO IN-CELL NUMBERS. Priya, 2026-09-13: "H8 (drop cell numbers)". This is a
                    # 4 x 13 grid of 6 x 6 matrices -- 1,872 cells across the figure -- and at
                    # fontsize 6 on a slide they are illegible clutter that fights the colour map
                    # doing the actual work. The numbers are not lost: every cell is written to
                    # this figure's value sidecar, which is where a reader who wants a specific
                    # distance should get it (and where a deck caption can quote it through
                    # `deck_values` rather than a reader squinting at a panel).
                    #
                    # THE DIAGONAL MEAN STAYS ON THE TITLE, because that one number is the panel's
                    # headline rather than a lookup: "did this position's pattern move" is read
                    # off it directly, and a reader should not have to average six cells by eye.
                    ax.set_xticks(range(len(CONF_LABELS)))
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                       rotation=90, fontsize=9)
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
                    head = "PRE" if ci == 0 else f"day {days[ci - 1]}"
                    ax.set_title(f"{head}\ndiag {np.nanmean(np.diag(D)):.2f}", fontsize=10,
                                 fontweight="bold" if ci == 0 else "normal")
                    if ci == 0:
                        ax.set_ylabel(f"{an}\nthis position", fontsize=11, fontweight="bold")
            if im is None:
                plt.close(fig)
                continue
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02,
                         label="crossnobis distance -- BRIGHT = unchanged "
                               "(1.0 = mean pre-stroke between-position distance)")
            cls = _class_note(v)
            _suptitle(fig, 
                f"Cross-validated (crossnobis) distance to the pre-stroke pattern — {wname} window\n"
                f"Post-stroke class: {cls}.  Rows = the post-stroke position, columns = the "
                f"PRE-STROKE reference position. Figure 6's layout, as DISTANCE.\n"
                f"DIAGONAL = did this position's pattern move. LOW is unchanged. Unlike figure 6 "
                f"this is NOISE-UNBIASED: a noisier session does not read as a larger distance.\n"
                f"First column is the no-lesion expectation. Units: mean pre-stroke between-position "
                "distance for that animal.\n"
                "UNBIASED IS NOT GAIN-INVARIANT, and the difference bites: a uniform amplitude "
                "change moves every cell here while leaving 8b untouched. Where this figure and 8b "
                "disagree, 8b is the one to believe about GEOMETRY.\n"
                f"Concretely (post-cue, working): PS92 and PS93 read their WORST diagonal on day 7 "
                f"(1.01, 1.31) while 8b puts day 7 among their BEST (0.66, 0.80) -- that spike is "
                f"amplitude, not a code that moved further away.", fontsize=8.8)
            _footer(fig)
            p = _out(out_dir, f"grant_8_crossnobis_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made
def fig_crossnobis_geometry(out_dir, min_trials=10):
    """8b: RSA proper -- each session's own 6x6 crossnobis RDM against the pre-stroke RDM.

    THIS IS THE ONE THAT ANSWERS THE GAIN QUESTION. Correlating two RDMs is invariant to scaling
    every distance by a constant, so a uniform post-stroke amplitude change cannot move it. Figure
    6's headline -- every position drops, far_R most -- is precisely the signature a global change
    would leave, and this is the measure that cannot be fooled by one.

    PER-POSITION INFORMATION SURVIVES, in a weaker form. A whole-RDM correlation is one number per
    session. Each position's ROW of the RDM -- its five distances to the other positions -- gives a
    per-position number, so the question "is far_R still arranged relative to everything else the
    way it was" is answerable. The question "did far_R's pattern move" is NOT, because second-order
    RSA has thrown away the patterns themselves. That is the trade against figure 8, which keeps the
    positions and gives up gain-invariance.

    Rows = position, columns = day; the strip above each animal is the whole-RDM correlation.

    WHAT THE INTERVALS CHANGED (2026-08-26). This figure carried no uncertainty for two weeks and
    was being read as a positive result: post-stroke correlations of 0.7-0.9 against a ceiling of
    0.88 look like "the geometry is preserved". With a block bootstrap the median 95% interval on a
    post-stroke whole-RDM correlation is 0.28 wide, and of 27 post-stroke sessions only TWO have a
    change from the ceiling that excludes zero. The rest are UNDETERMINED, not preserved -- PS93
    day 1 reads 0.64 against a ceiling of 0.89, a drop of 0.25, with an interval of [-0.69, +0.08].
    At this trial count the whole-RDM correlation cannot distinguish "unchanged" from "substantially
    rearranged", and the figure must not be quoted as evidence for either. The per-position rows are
    the sharper instrument, and close_center is where they most often exclude zero.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            store, days = _collect_7(align, v, min_trials)
            if not days:
                continue
            ci_store, _cd = _rdm_ci(align, v, min_trials)
            fig, axes = plt.subplots(len(ANIMALS), 2, figsize=(4.6 + 0.62 * len(days), 10.4),
                                     squeeze=False,
                                     gridspec_kw={"width_ratios": [len(days) + 1, 3.4]})
            drew = False
            for ri, an in enumerate(ANIMALS):
                got = store.get(an)
                if not got:
                    for ci in range(2):
                        axes[ri][ci].axis("off")
                    continue
                pre_by_sess, by_day = got
                rng = np.random.default_rng(_seed(an, align, v, "8b"))
                full_ref = _pre_reference(pre_by_sess)
                Dpre = _crossnobis_within(full_ref, rng, CONF_LABELS)
                rows = np.full((len(CONF_LABELS), 1 + len(days)), np.nan)
                whole = np.full(1 + len(days), np.nan)
                # `_rdm_scores` is shared with figure 8g and with the bootstrap that puts intervals
                # on both -- it used to be written out here a third time.
                _score_rdm = _rdm_scores
                for ci in range(1 + len(days)):
                    # COLUMN 0 IS GENUINELY LEAVE-ONE-SESSION-OUT (corrected 2026-08-25). It used to
                    # correlate the MEAN of the per-session RDMs against the RDM of the POOLED set --
                    # which CONTAINS every one of those sessions. That is circular, and it showed:
                    # the ceiling read 0.90-1.00 while the post columns it was meant to calibrate ran
                    # 0.52-0.86. Each pre-stroke session is now scored against an RDM built from the
                    # OTHER sessions only, and the resulting correlations are averaged -- one session
                    # against other days, exactly like every post column.
                    if ci == 0:
                        got = [_score_rdm(_crossnobis_within(pat, rng, CONF_LABELS),
                                          _crossnobis_within(
                                              _pre_reference(pre_by_sess, exclude=s),
                                              rng, CONF_LABELS))
                               for s, pat in pre_by_sess.items()]
                        if not got:
                            continue
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            whole[ci] = float(np.nanmean([g[0] for g in got]))
                            rows[:, ci] = np.nanmean([g[1] for g in got], axis=0)
                        continue
                    src = by_day.get(days[ci - 1])
                    if not src:
                        continue
                    whole[ci], rows[:, ci] = _score_rdm(
                        _crossnobis_within(src, rng, CONF_LABELS), Dpre)
                crec = ci_store.get(an) or {}
                ax = axes[ri][0]
                im = ax.imshow(np.ma.masked_invalid(rows), vmin=-1, vmax=1, cmap="RdBu_r",
                               aspect="auto")
                for i in range(len(CONF_LABELS)):
                    for j in range(1 + len(days)):
                        if np.isfinite(rows[i, j]):
                            _txt(ax, j, i, f"{rows[i, j]:.2f}", ha="center", va="center",
                                    fontsize=7.5)
                        # A BOXED CELL is one whose block-bootstrap interval on the CHANGE from the
                        # pre-stroke ceiling excludes zero. Drawn as an outline rather than printed
                        # as a number so it survives the compact variant, which drops in-cell text.
                        if j == 0:
                            continue
                        iv = ((crec.get(days[j - 1]) or {}).get("drows") or {}).get(CONF_LABELS[i])
                        if _excludes_zero(iv):
                            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                       edgecolor="k", lw=1.6, zorder=5))
                ax.set_xticks(range(1 + len(days)))
                ax.set_xticklabels((["PRE"] + [f"d{d}" for d in days])
                                   if ri == len(ANIMALS) - 1 else [], fontsize=9.5)
                ax.set_yticks(range(len(CONF_LABELS)))
                ax.set_yticklabels(_short(CONF_LABELS), fontsize=9.5)
                ax.set_ylabel(an, fontsize=11.5, fontweight="bold")
                if ri == 0:
                    ax.set_title("per-position: is this position's ROW of the RDM preserved?",
                                 fontsize=11)
                ax2 = axes[ri][1]
                xs = np.arange(1 + len(days))
                # THE SHADED BAND is the 95% block-bootstrap interval on each point. Where a post
                # column's band clears the PRE band the geometry demonstrably changed; where they
                # overlap the figure is not entitled to say so, and until now it had no way to
                # express the difference.
                lo = np.full(1 + len(days), np.nan)
                hi = np.full(1 + len(days), np.nan)
                for j in range(1 + len(days)):
                    iv = (crec.get("PRE" if j == 0 else days[j - 1]) or {}).get("whole")
                    if iv:
                        lo[j], hi[j] = iv[0], iv[1]
                m = np.isfinite(lo)
                if m.any():
                    ax2.fill_between(xs[m], lo[m], hi[m], color="#2166ac", alpha=0.18, lw=0)
                    if m[0]:
                        ax2.axhspan(lo[0], hi[0], color="0.55", alpha=0.20, lw=0, zorder=0)
                ax2.plot(xs, whole, "o-", color="#2166ac", ms=4.5, lw=1.4)
                ax2.axhline(0, color="k", lw=0.8)
                ax2.set_ylim(-0.3, 1.05)
                ax2.set_xticks(xs)
                ax2.set_xticklabels((["PRE"] + [f"d{d}" for d in days])
                                    if ri == len(ANIMALS) - 1 else [], fontsize=9.5)
                ax2.grid(alpha=0.25, lw=0.5)
                if ri == 0:
                    ax2.set_title("whole-RDM correlation\n(gain-invariant)", fontsize=11)
                drew = True
            if not drew:
                plt.close(fig)
                continue
            cls = _class_note(v)
            _suptitle(fig,
                f"Second-order RSA on crossnobis RDMs — {wname} window\n"
                f"Post-stroke class: {cls}.  Each session's OWN 6x6 crossnobis RDM correlated "
                f"against the pre-stroke RDM.\n"
                f"INVARIANT TO A GLOBAL AMPLITUDE CHANGE, which figures 6 and 8 are not: scaling "
                f"every distance leaves a correlation between RDMs unchanged.\n"
                f"PRE column = each pre-stroke session against an RDM built from the OTHERS only "
                f"(leave-one-session-out), which is the ceiling. Per-position numbers are that "
                "position's five distances to the others --\n'is it still arranged the same way', "
                f"NOT 'did its pattern move'. Read the post columns against PRE, never against 1.\n"
                f"SHADED BAND / BOXED CELL = 95% block bootstrap over the scheduler's position "
                f"blocks; a box means the change from the PRE ceiling excludes zero. Sessions are "
                f"held FIXED, so this is TRIAL noise only.",
                fontsize=9.5)
            fig.tight_layout(rect=(0, 0, 1, 1.0))   # top reserved by _suptitle
            # A SCALE THE COMPACT VARIANT STILL HAS. `--compact` drops every in-cell number, and
            # without a colour bar this heatmap would carry no scale at all -- a reader could not
            # tell 0.9 from -0.9.
            #
            # CREATED AFTER `tight_layout`, and that ordering is the whole of it. `tight_layout`
            # moves only axes belonging to the gridspec, so a colour bar made before it stayed put
            # while the panels expanded rightwards underneath -- the fault at the top of this file,
            # reproduced here on the first attempt and caught by `_overlaps` before it shipped.
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02,
                         label="row of the RDM preserved (r)")
            _footer(fig)
            p = Path(out_dir) / f"grant_8b_crossnobis_geometry_{align}_{v}.png"
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made
def _mats_crossnobis(_an, rng, sign=-1):
    """Crossnobis with the whitener fixed at the first call and reused for every later resample.

    ``sign=-1`` (the default) negates the distances so "larger diagonal = more preserved" holds as
    it does for the correlation figures -- see the note at the return. ``sign=+1`` gives the raw
    distances, which is what the ASYMMETRY needs: D[P,Q] - D[Q,P] must be in distance units to be
    read as "post P is further from pre Q than post Q is from pre P".
    """
    held = {}

    def build(pat, ref):
        pm0, pm1, pres = _halves(pat, rng, CONF_LABELS)
        rm0, rm1, rres = _halves(ref, rng, CONF_LABELS)
        if "P" not in held:
            held["P"] = _whitener(pres + rres)
            # SAME UNITS AS THE FIGURE. `_matrices_crossnobis` divides every distance by the mean
            # pre-stroke between-position distance, so 1.0 reads as "as far apart as two different
            # pre-stroke positions". The interval was computed in RAW crossnobis units and printed
            # beside a normalised point estimate -- figure 8d showed "day 1 +0.28 [+6.63, +69.41]",
            # an interval not containing its own estimate, which is how the mismatch announced
            # itself. Fixed from the first reference seen, so both numbers share one scale.
            s = np.nanmean(_triu_vals(_crossnobis_within(ref, rng, CONF_LABELS)))
            held["scale"] = float(s) if np.isfinite(s) and s > 0 else 1.0
        P = held["P"]
        D = np.full((len(CONF_LABELS), len(CONF_LABELS)), np.nan)
        if P is None:
            return D
        for i, a in enumerate(CONF_LABELS):
            for j, b in enumerate(CONF_LABELS):
                if a in pm0 and b in rm0:
                    D[i, j] = float((pm0[a] - rm0[b]) @ P @ (pm1[a] - rm1[b])) / held["scale"]
        # NEGATED BY DEFAULT so "larger diagonal = more preserved" holds here as it does for the
        # correlation figures. `_delta_diag_ci` differences mean diagonals, and for a DISTANCE a
        # SMALLER diagonal means less change -- without the flip the interval would carry the
        # opposite sign to the number printed beside it, which is worse than no interval at all.
        return sign * D

    return build
@lru_cache(maxsize=12)
def _matrices_crossnobis(align, variant, min_trials=10, row_centre=False):
    """{animal: {"PRE": D, day: D}} of cross-set crossnobis distances, in pre-stroke units.

    ``row_centre`` subtracts each ROW's own mean, which is the difference between asking "did this
    position move" and "which position did it move TOWARD".

    WHY THAT MATTERS (Priya, 2026-08-26, on whole rows shifting together). Writing the
    cross-validated distance out, in the whitened metric:

        d(post P, pre Q) = |mu_postP|^2 - 2 mu_postP . mu_preQ + |mu_preQ|^2

    the first term depends ONLY ON P. So a change in the overall magnitude of position P's
    post-stroke response moves its distance to EVERY pre-stroke position by the same amount, and the
    panel shows a uniform orange or purple row. That is amplitude, not "this position came to
    resemble all six". Row-centring removes the term that carries it and leaves the CONTRAST within
    the row, which is where a substitution lives. Same gain sensitivity that makes 8b the arbiter
    for anything about geometry.
    """
    store, days = _collect_7(align, variant, min_trials)
    out = {}
    for an, (pre_by_sess, by_day) in store.items():
        rng = np.random.default_rng(_seed(an, align, variant, "8"))
        full_ref = _pre_reference(pre_by_sess)
        scale = np.nanmean(_triu_vals(_crossnobis_within(full_ref, rng, CONF_LABELS)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        d = {}
        base = _nanmean_stack([_crossnobis_cross(pat, _pre_reference(pre_by_sess, exclude=s),
                                                 rng, CONF_LABELS)
                               for s, pat in pre_by_sess.items()])
        if base is not None:
            d["PRE"] = base / scale
        for day, pat in by_day.items():
            d[day] = _crossnobis_cross(pat, full_ref, rng, CONF_LABELS) / scale
        if row_centre:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)      # a row can be entirely NaN
                d = {k: M - np.nanmean(M, axis=1, keepdims=True) for k, M in d.items()}
        if d:
            out[an] = d
    return out, days
def fig_crossnobis_delta(out_dir, min_trials=10):
    """8d: figure 8 as DIFFERENCES from the pre-stroke reference.

    The reference distances are NOT uniform -- close positions sit nearer each other than far ones
    do, and each animal's baseline has its own texture -- so an absolute cell of 1.2 means different
    things in different places. Subtracting leaves only what the lesion did.

    POSITIVE = further from the pre-stroke pattern than a held-out pre-stroke session is; NEGATIVE =
    closer. The diagonal is the headline and the off-diagonal carries the substitution: a post-stroke
    far_R row going NEGATIVE under the close_L column means far_R trials moved TOWARD pre-stroke
    close_L.

    Read beside 8b before concluding anything about geometry: these are distances, so a uniform
    amplitude change shifts the whole panel while leaving 8b untouched.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            mats, days = _matrices_crossnobis(align, v, min_trials)
            if not days or not mats:
                continue
            # SIGN: _mats_crossnobis returns NEGATED distances so bigger = more preserved, matching
            # the correlation figures; flip the interval back into distance units for display.
            cis = {a2: {d: (-hi, -lo) for d, (lo, hi) in v2.items()}
                   for a2, v2 in _delta_cis(align, v, min_trials, _mats_crossnobis, "8d").items()}
            cls = _class_note(v)
            p = _delta_grid(
                mats, days, out_dir, f"grant_8d_crossnobis_delta_{align}_{v}.png",
                title=(f"Crossnobis distance to the pre-stroke pattern, CHANGE FROM PRE-STROKE — "
                       f"{wname} window\nPost-stroke class: {cls}.  Column 1 is the pre-stroke "
                       f"reference (leave-one-session-out); every later column is THAT DAY MINUS "
                       f"IT.\nPOSITIVE = further from the pre-stroke pattern than a held-out "
                       f"pre-stroke session is. NEGATIVE = closer. ZERO = an ordinary pre-stroke "
                       f"day.\nDiagonal = did the pattern move. A NEGATIVE off-diagonal cell is a "
                       f"substitution: that row's trials moved TOWARD the column's pre-stroke "
                       f"pattern. Distances are NOT gain-invariant -- read with 8b."),
                abs_label="pre-stroke distance",
                delta_label="change in distance (1.0 = mean pre-stroke between-position)",
                vmin=0, vmax=2.5, cmap="magma", dmax=1.5, summary=_diag,
                # DISTANCES: a row's best match is its SMALLEST entry, not its largest.
                ylab="this position", cis=cis, higher_is_better=False)
            if p:
                made.append(p)
    return made
def fig_delta_trajectory(out_dir, min_trials=10):
    """9: THE BOOTSTRAP RESULTS AS A FIGURE -- change from pre-stroke over days, with intervals.

    Priya, 2026-08-26: "is there a figure that shows the bootstrap results?" There was not. The
    intervals existed only as text above the 6x6 matrices of 6d/7d/8d, which is the wrong shape for
    the question they answer: whether a position is recovering, holding or getting worse is a
    TRAJECTORY, and a trajectory read out of twenty-eight small matrices by eye is not read at all.

    LEFT PANEL, per animal: the mean own-position change with its 95% block-bootstrap interval, one
    point per day. This is the number printed in 6d's panel titles, plotted.
    RIGHT PANEL: the same split BY POSITION, which is where the deficit lives -- far_R is the
    position the lesion takes and the others are the control it has to be read against.

    ZERO IS THE NULL AND IT IS A REAL ONE: it means "this day differs from the pre-stroke reference
    no more than one pre-stroke day differs from the others", because the baseline is
    leave-one-session-out rather than a correlation of 1. A point whose interval excludes zero has
    changed by more than ordinary day-to-day drift.

    SESSIONS ARE NOT RESAMPLED, so an interval says how well determined a day is GIVEN THESE DAYS
    and licenses no claim about days not recorded. The trajectory is what speaks to that, which is
    the whole reason this figure is per-day rather than pooled.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            _, days = _collect_7(align, v, min_trials)
            if not days:
                continue
            cis = _delta_cis(align, v, min_trials, _mats_pattern, "9", full=True)
            if not any(cis.values()):
                continue
            # More row height and an explicit hspace. This is figure 9, not
            # fig_asymmetry -- an earlier edit matched here by mistake because both
            # call plt.subplots(len(ANIMALS), 2, ...). The extra room is kept because
            # it is an improvement on its own terms, but the claim it carried was not
            # about this figure.
            fig, axes = plt.subplots(len(ANIMALS), 2, figsize=(10.0, 2.4 * len(ANIMALS) + 1.6),
                                     gridspec_kw={"hspace": 0.60},
                                     squeeze=False, sharex=True)
            drew = False
            for ri, an in enumerate(ANIMALS):
                rec = cis.get(an) or {}
                ax = axes[ri][0]
                xs = [d for d in days if d in rec]
                if xs:
                    med = [rec[d]["mean"][2] for d in xs]
                    lo = [rec[d]["mean"][2] - rec[d]["mean"][0] for d in xs]
                    hi = [rec[d]["mean"][1] - rec[d]["mean"][2] for d in xs]
                    ax.errorbar(xs, med, yerr=[lo, hi], fmt="o-", color="#b2182b", ms=5,
                                capsize=3, lw=1.5)
                    drew = True
                ax.axhline(0, color="k", lw=1.2)
                ax.set_ylabel(f"{an}\nchange in r", fontsize=11.5, fontweight="bold")
                ax.grid(alpha=0.25, lw=0.5)
                if ri == 0:
                    ax.set_title("mean own-position change (95% block bootstrap)", fontsize=9.5)

                ax2 = axes[ri][1]
                for q in CONF_LABELS:
                    col, mk, _ls = pos_style()[q]
                    qx = [d for d in days if d in rec and q in rec[d]["pos"]]
                    if not qx:
                        continue
                    qm = [rec[d]["pos"][q][2] for d in qx]
                    ql = [rec[d]["pos"][q][2] - rec[d]["pos"][q][0] for d in qx]
                    qh = [rec[d]["pos"][q][1] - rec[d]["pos"][q][2] for d in qx]
                    ax2.errorbar(qx, qm, yerr=[ql, qh], fmt=mk + "-", color=col, ms=4,
                                 capsize=2, lw=1.1, alpha=0.9,
                                 label=q if ri == 0 else None)
                ax2.axhline(0, color="k", lw=1.2)
                ax2.grid(alpha=0.25, lw=0.5)
                if ri == 0:
                    ax2.set_title("the same, per position", fontsize=9.5)
                if ri == len(ANIMALS) - 1:
                    ax.set_xlabel("days from lesion")
                    ax2.set_xlabel("days from lesion")
            if not drew:
                plt.close(fig)
                continue
            h, lab = axes[0][1].get_legend_handles_labels()
            if h:
                # ABOVE THE FOOTER, not on it. `_footer` writes at y=0.004, and a legend at
                # "lower center" lands in the same place -- the two overprinted each other in the
                # first render of this figure. This is the only figure in the module carrying both.
                fig.legend(h, lab, loc="lower center", ncol=len(POS), fontsize=11, frameon=False,
                           bbox_to_anchor=(0.5, 0.035))
            cls = _class_note(v)
            _suptitle(fig, 
                f"Change from pre-stroke over days, with block-bootstrap intervals — {wname} "
                f"window\n"
                f"Post-stroke class: {cls}.  ZERO = this day differs from the pre-stroke reference "
                f"no more than one pre-stroke day differs from the others\n"
                f"(the baseline is leave-one-session-out, NOT a correlation of 1). An interval "
                f"excluding zero is a change beyond ordinary day-to-day drift.\n"
                f"Blocks resampled within session; sessions NOT resampled, so an interval is "
                f"conditional on these days and the trajectory is what speaks to the rest.",
                fontsize=9.5)
            # Bottom band holds BOTH the legend (y=0.035) and the footer (y=0.004), so it needs
            # more than the usual 0.05.
            fig.tight_layout(rect=(0, 0.10, 1, 1.0))   # top reserved by _suptitle
            _footer(fig)
            p = Path(out_dir) / f"grant_9_delta_trajectory_{align}_{v}.png"
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made
def _asymmetry_ci(align, variant, min_trials, n_boot=N_BOOT_DELTA):
    """{animal: {"PRE"|day: (observed 6x6 asymmetry, bool 6x6 "interval excludes zero")}}.

    THE QUANTITY. A[P,Q] = d(post at P, pre at Q) - d(post at Q, pre at P). Rows and columns index
    genuinely different sets, so there is no reason for D to be symmetric and the gap between the
    two orderings is the substitution signal: a large positive A[far_R, close_L] says post-stroke
    far_R sits further from pre-stroke close_L than post-stroke close_L sits from pre-stroke far_R.

    WHY THIS REPLACES THE EARLIER COMPARISON. The first pass compared each post column's mean
    asymmetry against a SINGLE held-out pre-stroke session (n = 1, no spread), after establishing
    that the leave-one-out AVERAGE was not a fair baseline -- averaging six matrices shrinks the
    noise whose absolute value is being measured, so it sits systematically low. One draw with no
    spread is not a baseline either: "PS93 post 0.31-0.47 vs pre 0.523" could be an unlucky 0606.
    A bootstrap interval per PAIR needs no pre-stroke baseline at all -- if the interval on
    A[P,Q] excludes zero, that pair is asymmetric, full stop -- and trial counts are matched by
    construction, which disposes of the other objection to the earlier comparison.

    NOT A PERMUTATION. Shuffling position labels equalises the condition means, so the true
    distances collapse toward zero -- but the sampling variance of a crossnobis distance scales with
    the true difference vector, so a real and perfectly SYMMETRIC separation still yields a larger
    |A| than permuted data does. The permuted null sits too low and would call noise asymmetry.

    The PRE column is leave-one-session-out and is computed ONCE, not per day: it does not depend on
    which post-stroke day is being scored.
    """
    x_store, days = _collect_7(align, variant, min_trials)
    b_store, _ = _collect_7(align, variant, min_trials, "blk")
    out = {}
    for an in ANIMALS:
        if an not in x_store or an not in b_store:
            continue
        (pre_x, day_x), (pre_b, day_b) = x_store[an], b_store[an]
        rec = _boot_cached(
            "8e_asym", (pre_x, pre_b, day_x, day_b, days,
                        (an, align, variant, min_trials, n_boot)),
            lambda an=an, pre_x=pre_x, pre_b=pre_b, day_x=day_x,
            day_b=day_b: _asymmetry_one(
                an, align, variant, pre_x, pre_b, day_x, day_b, days, n_boot))
        if rec:
            out[an] = rec
    return out
def fig_asymmetry(out_dir, min_trials=10):
    """8e: is the post-stroke distance matrix ASYMMETRIC, and where -- with intervals.

    Priya, 2026-08-25: "the first column is prestroke LOSO right? so we don't necessarily expect it
    to be symmetric across the diagonal?" Right, and the two column kinds differ. Column 1 scores one
    pre-stroke session against OTHER pre-stroke sessions, so both sides estimate the same underlying
    patterns and A[P,Q] has expectation zero -- whatever is there is noise. Every later column scores
    a genuinely different distribution against the pre-stroke reference, and there the asymmetry is
    the SUBSTITUTION: which way a position moved, not merely that it moved.

    A GREEN RING marks a pair whose 95% block-bootstrap interval excludes zero. Read the PRE column
    first: rings there are the false-positive rate this construction actually achieves, and they
    should be few.

    THIS FIGURE MUST NOT BE SYMMETRISED, unlike figure 7. There the two cells estimated ONE quantity
    and differed only by which random half went where, so averaging them was strictly better. Here
    they estimate different quantities and the difference is the result.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            _, days = _collect_7(align, v, min_trials)
            if not days:
                continue
            cis = _asymmetry_ci(align, v, min_trials)
            if not cis:
                continue
            cols = ["PRE"] + list(days)
            # SHORTEST FIGURE IN THE MODULE and it set no hspace, so its per-panel titles had the
            # least room of any of them -- 30 of the 41 faults in one render. 0.60 at a taller
            # figure is the combination verified clean on the sibling grids.
            fig, axes = plt.subplots(len(ANIMALS), len(cols),
                                     figsize=(_colw() * len(cols) + 1.4, 9.5), squeeze=False,
                                     gridspec_kw={"hspace": 0.60})
            im = None
            for ri, an in enumerate(ANIMALS):
                rec = cis.get(an) or {}
                for ci, key in enumerate(cols):
                    ax = axes[ri][ci]
                    got = rec.get(key)
                    if got is None:
                        ax.axis("off")
                        continue
                    A, sig = got
                    lim = float(np.nanpercentile(np.abs(A), 95)) or 1.0
                    im = ax.imshow(np.ma.masked_invalid(A), vmin=-lim, vmax=lim, cmap="PuOr_r")
                    for i in range(len(CONF_LABELS)):
                        for j in range(len(CONF_LABELS)):
                            if sig[i, j]:
                                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                                           edgecolor="lime", lw=1.4))
                    ax.set_xticks(range(len(CONF_LABELS)))
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                       rotation=90, fontsize=9)
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
                    n_sig = int(sig.sum() // 2)          # antisymmetric: each pair rings twice
                    head = "PRE" if key == "PRE" else f"day {key}"
                    ax.set_title(f"{head}\n{n_sig}/15", fontsize=10,
                                 fontweight="bold" if ci == 0 else "normal")
                    if ci == 0:
                        ax.set_ylabel(f"{an}\nthis position", fontsize=11, fontweight="bold")
            if im is None:
                plt.close(fig)
                continue
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02,
                         label="d(post P, pre Q) - d(post Q, pre P)")
            cls = _class_note(v)
            _suptitle(fig, 
                f"Is the distance matrix ASYMMETRIC, and where? -- {wname} window\n"
                f"Post-stroke class: {cls}.  A[P,Q] = d(post at P, pre at Q) - d(post at Q, pre at "
                f"P). Rows and columns index different sets, so symmetry is not expected.\n"
                f"GREEN RING = 95% block-bootstrap interval excludes zero. The count above each "
                f"panel is rung pairs out of 15.\n"
                f"READ THE PRE COLUMN FIRST: both sides there estimate the same patterns, so its "
                f"asymmetry has expectation zero and its rings are this construction's own "
                f"false-positive rate.", fontsize=9.5)
            _footer(fig)
            p = _out(out_dir, f"grant_8e_asymmetry_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made
def _asymmetry_one(an, align, variant, pre_x, pre_b, day_x, day_b, days, n_boot):
    """One ANIMAL's 8e asymmetry intervals: ``{"PRE"|day: (obs, sig)}``, or None.

    Extracted so one animal is the cache unit, as `_disatt_one` is for 7b and `_rdm_one` for
    8b/8g. The same reasoning applies: the draws loop is outer and the days inner, so an
    animal's days share each draw's leave-one-out pre-stroke pool -- and that shared pool is
    what the PRE column is a baseline FOR. Caching per day would dissolve it. An animal's
    draws depend on nothing outside that animal (the RNG is seeded from its own name and
    `build` is bound to it), so the boundary is exact rather than approximate.

    `days` is a parameter rather than a closure because it comes from `_collect_7` at
    function scope, not from this animal: it decides which columns are scored, so it changes
    the answer and therefore belongs in the cache key.
    """
    rng = np.random.default_rng(_seed(an, align, variant, "asym"))
    build = _mats_crossnobis(an, rng, sign=+1)

    def _pool(pre_r, exclude=None):
        acc = {}
        for s, Z in pre_r.items():
            if s == exclude:
                continue
            for q, z in Z.items():
                acc.setdefault(q, []).append(z)
        return {q: np.vstack(v) for q, v in acc.items()}

    def _summarise(draws):
        if len(draws) < n_boot // 4:
            return None
        S = np.stack(draws)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obs = np.nanmedian(S, axis=0)
            lo = np.nanpercentile(S, 2.5, axis=0)
            hi = np.nanpercentile(S, 97.5, axis=0)
        sig = np.isfinite(lo) & np.isfinite(hi) & ((lo > 0) | (hi < 0))
        np.fill_diagonal(sig, False)                 # A[P,P] is 0 by construction
        return obs, sig

    rec, pre_draws, day_draws = {}, [], {d: [] for d in days}
    for _ in range(n_boot):
        pre_r = {s: _block_boot(pre_x[s], pre_b[s], rng) for s in pre_x}
        pre_r = {s: v for s, v in pre_r.items() if v}
        if not pre_r:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            As = []
            for s, held in pre_r.items():
                rest = _pool(pre_r, exclude=s)
                if held and rest:
                    D = build(held, rest)
                    As.append(D - D.T)
            if As:
                pre_draws.append(np.nanmean(np.stack(As), axis=0))
            full = _pool(pre_r)
            for d in days:
                if d not in day_x:
                    continue
                day_r = _block_boot(day_x[d], day_b[d], rng)
                if not day_r or not full:
                    continue
                D = build(day_r, full)
                day_draws[d].append(D - D.T)
    got = _summarise(pre_draws)
    if got:
        rec["PRE"] = got
    for d in days:
        got = _summarise(day_draws[d])
        if got:
            rec[d] = got
    return rec or None
@lru_cache(maxsize=6)
def _rdm_rows(align, variant, min_trials=10):
    """{animal: {"PRE"|day: (per-position row r, whole-RDM r, n_positions)}} for figures 8b and 8g.

    Extracted so the per-position TRAJECTORY figure and the heatmap cannot disagree: they are the
    same numbers drawn two ways.
    """
    x_store, days = _collect_7(align, variant, min_trials)
    out = {}
    for an in ANIMALS:
        if an not in x_store:
            continue
        pre_by_sess, by_day = x_store[an]
        rng = np.random.default_rng(_seed(an, align, variant, "8g"))
        full_ref = _pre_reference(pre_by_sess)
        Dpre = _crossnobis_within(full_ref, rng, CONF_LABELS)

        def score(D, Dref):
            whole, rows = _rdm_scores(D, Dref)
            n_pos = int(np.isfinite(np.diag(Dref)).sum() or 0)
            return rows, whole, n_pos

        rec = {}
        loo = [score(_crossnobis_within(pat, rng, CONF_LABELS),
                     _crossnobis_within(_pre_reference(pre_by_sess, exclude=s), rng, CONF_LABELS))
               for s, pat in pre_by_sess.items()]
        if loo:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rec["PRE"] = (np.nanmean([r for r, _w, _n in loo], axis=0),
                              float(np.nanmean([w for _r, w, _n in loo])), 6)
        for d, pat in by_day.items():
            D = _crossnobis_within(pat, rng, CONF_LABELS)
            r, w, _n = score(D, Dpre)
            # POSITIONS PRESENT IN THIS SESSION, which is what governs whether a row is computable
            rec[d] = (r, w, len(pat))
        if rec:
            out[an] = rec
    return out, days
def _rdm_pct(ws, rs, n_boot):
    """{"whole": (lo, hi, med), "rows": {position: (lo, hi, med)}} from a draw list, or None."""
    w = np.array([x for x in ws if np.isfinite(x)])
    if len(w) < n_boot // 4:
        return None
    rec = {"whole": _pct3(w), "rows": {}}
    if rs:
        R = np.stack(rs)
        for k, q in enumerate(CONF_LABELS):
            col = R[:, k]
            col = col[np.isfinite(col)]
            if len(col) >= n_boot // 4:
                rec["rows"][q] = _pct3(col)
    return rec
@lru_cache(maxsize=6)
def _rdm_ci(align, variant, min_trials=10, n_boot=N_BOOT_RDM, n_loo=N_LOO_DRAW):
    """Block-bootstrap intervals for figures 8b and 8g.

    Returns ``({animal: {"PRE"|day: rec}}, days)`` where a post-stroke ``rec`` carries the day's own
    correlation (``whole``, ``rows``) AND its change from the leave-one-session-out pre-stroke
    ceiling (``dwhole``, ``drows``). Both come from the same draws, so the delta is taken draw by
    draw and the two share their noise -- differencing two independently published intervals would
    overstate the spread.

    8b IS THE ARBITER for anything about geometry -- it is the one measure here that a global
    amplitude change cannot move -- and it shipped for two weeks with no uncertainty at all. A
    post-stroke session reading 0.82 against a pre-stroke ceiling of 0.90 is either a real loss or
    nothing, and the figure gave the reader no way to tell.

    WHAT IS RESAMPLED: the scheduler's ~6-trial position blocks, within session. SESSIONS ARE HELD
    FIXED, following every other interval in this module, so this is trial-level noise only and says
    nothing about how much a NEW post-stroke day would differ. The spread of the PRE ceiling across
    sessions is the figure's own estimate of that, and it is the larger of the two.
    """
    x_store, days = _collect_7(align, variant, min_trials)
    b_store, _ = _collect_7(align, variant, min_trials, "blk")
    # HOISTED OUT OF THE LOOP so it can form part of the cache key, and because it is a
    # function of (align, variant) alone -- it was recomputed once per animal for no reason
    # beyond where the call happened to sit.
    obs_all, _od = _rdm_rows(align, variant, min_trials)
    out = {}
    for an in ANIMALS:
        if an not in x_store or an not in b_store:
            continue
        (pre_x, day_x), (pre_b, day_b) = x_store[an], b_store[an]
        if len(pre_x) < 2 or not day_x:
            continue
        rec = _boot_cached(
            "8b_rdm", (pre_x, pre_b, day_x, day_b, obs_all.get(an) or {},
                       (an, align, variant, min_trials, n_boot, n_loo)),
            lambda an=an, pre_x=pre_x, pre_b=pre_b, day_x=day_x, day_b=day_b,
            orec=(obs_all.get(an) or {}): _rdm_one(
                an, align, variant, pre_x, pre_b, day_x, day_b, orec, n_boot, n_loo))
        if rec:
            out[an] = rec
    return out, days
def _rdm_one(an, align, variant, pre_x, pre_b, day_x, day_b, orec, n_boot, n_loo):
    """One ANIMAL's 8b/8g intervals: ``{"PRE"|day: rec}``, or None.

    Extracted so a single animal is the unit that gets cached, exactly as `_disatt_one` is
    for 7b. THE ANIMAL IS THE RIGHT GRAIN AND THE DAY IS NOT: the draws loop runs
    draws-outer and days-inner precisely so every day of an animal shares each draw's
    leave-one-out reference resample, which is what makes those days' intervals comparable
    to one another. Caching per day would break that silently. One animal's draws depend on
    nothing outside that animal -- the RNG is seeded from its own name -- so this boundary
    is exact rather than approximate.

    `orec` (the plotted `_rdm_rows` estimate the bands are anchored to) is passed IN so it
    forms part of the cache key: a cached band anchored to a stale point estimate would sit
    off the value drawn on top of it, which is the one failure here a reader could not see.
    """
    rng = np.random.default_rng(_seed(an, align, variant, "8bci"))
    pre_w, pre_r = [], []
    acc = {d: {"w": [], "r": [], "dw": [], "dr": []} for d in day_x}
    try:
        for _ in range(n_boot):
            drawn = {s: _block_boot(pre_x[s], pre_b[s], rng) for s in sorted(pre_x)}
            drawn = {s: v for s, v in drawn.items() if v}
            if len(drawn) < 2:
                continue

            def _pool(exclude=None, drawn=drawn):
                a = {}
                for s, Z in drawn.items():
                    if s == exclude:
                        continue
                    for q, z in Z.items():
                        a.setdefault(q, []).append(z)
                return {q: np.vstack(v) for q, v in a.items()}

            # THE CEILING IS LEAVE-ONE-SESSION-OUT, never the reference against itself: a set
            # correlated with an RDM built from a pool CONTAINING it reads ~1 by construction,
            # and every delta would come out at about -1 regardless of the data.
            keys = list(drawn)
            if len(keys) > n_loo:
                keys = [keys[k] for k in rng.choice(len(keys), n_loo, replace=False)]
            ws, rs = [], []
            for s in keys:
                rest = _pool(exclude=s)
                if not rest:
                    continue
                w, rr = _rdm_scores(_fast_rdm(drawn[s], rng, CONF_LABELS),
                                    _fast_rdm(rest, rng, CONF_LABELS))
                ws.append(w)
                rs.append(rr)
            if not ws:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ceil_w = float(np.nanmean(ws))
                ceil_r = np.nanmean(np.stack(rs), axis=0)
            pre_w.append(ceil_w)
            pre_r.append(ceil_r)
            Dref = _fast_rdm(_pool(), rng, CONF_LABELS)
            for d in day_x:
                dr = _block_boot(day_x[d], day_b[d], rng)
                if not dr:
                    continue
                w, rr = _rdm_scores(_fast_rdm(dr, rng, CONF_LABELS), Dref)
                acc[d]["w"].append(w)
                acc[d]["r"].append(rr)
                acc[d]["dw"].append(w - ceil_w)
                acc[d]["dr"].append(rr - ceil_r)
    except Exception as ex:                                          # noqa: BLE001
        print(f"  !! 8b CI {an} {align}/{variant}: {type(ex).__name__} {str(ex)[:80]}",
              flush=True)
        return None

    def _fix(rc, col, delta=False, orec=orec):
        t = orec.get(col)
        if not rc or t is None:
            return rc
        b0 = orec.get("PRE") if delta else None
        if delta and b0 is None:
            return rc
        # A CORRELATION LIVES IN [-1, 1]; a DIFFERENCE of two of them does not.
        bd = {} if delta else {"lo": -1.0, "hi": 1.0}
        rc["whole"] = _anchor(rc["whole"], t[1] - b0[1] if delta else t[1], **bd)
        for k, q in enumerate(CONF_LABELS):
            if q in rc["rows"]:
                th = t[0][k] - b0[0][k] if delta else t[0][k]
                rc["rows"][q] = _anchor(rc["rows"][q], th, **bd)
        return rc

    rec = {}
    base = _fix(_rdm_pct(pre_w, pre_r, n_boot), "PRE")
    if base:
        rec["PRE"] = base
    for d, a in acc.items():
        cur = _fix(_rdm_pct(a["w"], a["r"], n_boot), d)
        dlt = _fix(_rdm_pct(a["dw"], a["dr"], n_boot), d, delta=True)
        if cur:
            if dlt:
                cur["dwhole"], cur["drows"] = dlt["whole"], dlt["rows"]
            rec[d] = cur
    return rec or None
def fig_geometry_by_position(out_dir, min_trials=10):
    """8g: figure 8b split BY SPOUT POSITION -- one panel per position, animals as lines.

    Priya, 2026-08-26. 8b's left panel is already per position, but as a heatmap of animal-major
    rows: comparing one position ACROSS animals and days means reading four separate blocks. Here
    each position gets a panel and each animal a line, which is the comparison the deficit is about
    -- far_R against the positions that were spared, in every animal at once.

    THE DASHED LINE IS THAT ANIMAL'S PRE CEILING for that position, leave-one-session-out. Read a
    trace against its own dashed line, not against 1: a held-out pre-stroke session does not
    reproduce the others perfectly either.

    A GAP IS NOT A ZERO. A row correlation needs at least four of the five partner positions, so a
    session missing two positions has every row uncomputable -- including the positions the animal
    licked normally. That is why whole days vanish for PS94 in the LICK class, and it is a property
    of the estimator, not of the animal. The `working` class keeps miss-while-working trials and
    fills most of them; the panel titles carry how many sessions actually contributed.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            rows, days = _rdm_rows(align, v, min_trials)
            if not days or not rows:
                continue
            ci_store, _cd = _rdm_ci(align, v, min_trials)
            fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), squeeze=False, sharex=True,
                                     sharey=True, gridspec_kw={"hspace": 0.32})
            drew = False
            for k, q in enumerate(CONF_LABELS):
                ax = axes[k // 3][k % 3]
                n_have = 0
                for an in ANIMALS:
                    rec = rows.get(an) or {}
                    xs = [d for d in days if d in rec and np.isfinite(rec[d][0][k])]
                    ys = [rec[d][0][k] for d in xs]
                    col = (config.animals().get(an) or {}).get("color", "0.4")
                    if xs:
                        # 95% block-bootstrap band, drawn per animal. Four overlaid bands would be
                        # unreadable at full opacity; at 0.13 the traces stay legible and a band
                        # that clears its own dashed ceiling is still obvious.
                        crec = ci_store.get(an) or {}
                        bl = [((crec.get(d) or {}).get("rows") or {}).get(q) for d in xs]
                        if any(bl):
                            bx = [x for x, iv in zip(xs, bl) if iv]
                            ax.fill_between(bx, [iv[0] for iv in bl if iv],
                                            [iv[1] for iv in bl if iv],
                                            color=col, alpha=0.13, lw=0)
                        ax.plot(xs, ys, "o-", color=col, ms=4, lw=1.4,
                                label=an if k == 0 else None)
                        n_have += 1
                        drew = True
                    if "PRE" in rec and np.isfinite(rec["PRE"][0][k]):
                        ax.axhline(rec["PRE"][0][k], color=col, ls=(0, (2, 3)), lw=1.0, alpha=0.8)
                ax.axhline(0, color="k", lw=1.0)
                ax.set_ylim(-1.05, 1.05)
                ax.grid(alpha=0.25, lw=0.5)
                ax.set_title(f"{q}   ({n_have}/4 animals)", fontsize=11, fontweight="bold")
                if k % 3 == 0:
                    ax.set_ylabel("row of the RDM preserved (r)", fontsize=10)
                if k // 3 == 1:
                    ax.set_xlabel("days from lesion", fontsize=11)
            if not drew:
                plt.close(fig)
                continue
            h, lab = axes[0][0].get_legend_handles_labels()
            if h:
                fig.legend(h, lab, loc="lower center", ncol=len(ANIMALS), fontsize=10,
                           frameon=False, bbox_to_anchor=(0.5, 0.035))
            cls = _class_note(v)
            # LAY THE PANELS OUT INTO WHAT THE HEADER LEFT, not into the full height. `tight_layout`
            # first and `_suptitle` after -- which compresses every axes into [0, top] -- spread the
            # panels over the whole figure and then shrank them away from the header, wasting about
            # a tenth of the height. `_suptitle` returns that `top`; target it. Same fault, same fix
            # as figure 11.
            top = _suptitle(fig,
                      f"Is each position's RDM row preserved? -- by POSITION -- {wname} window\n"
                      f"Post-stroke class: {cls}.  Figure 8b split so one position can be compared "
                      f"across animals and days in a single panel.\n"
                      f"DASHED = that animal's own pre-stroke ceiling for that position "
                      f"(leave-one-session-out). Read a trace against its own dashed line, never "
                      f"against 1.\n"
                      f"A GAP IS NOT A ZERO: a row needs 4 of its 5 partner positions, so a session "
                      f"missing two positions has EVERY row uncomputable -- including positions the "
                      f"animal licked normally.\n"
                      f"SHADED = 95% block bootstrap over the scheduler's position blocks, sessions "
                      f"held FIXED (trial noise only). A band overlapping its own dashed ceiling is "
                      f"a session this figure cannot call changed.")
            fig.tight_layout(rect=(0, 0.10, 1, top))
            _footer(fig)
            p = _out(out_dir, f"grant_8g_geometry_by_position_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made
