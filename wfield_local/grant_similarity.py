"""GRANT FIGURES — SIMILARITY

Pattern similarity and split-half reliability -- is a change real, or is the estimate noisy?

Split out of `grant_figures` on 2026-09-21, one module per figure family. THE GROUPING IS FROM THE
CALL GRAPH, not from the names: every function here is reached from this family's entry points and
from no other family's. Anything shared with a sibling lives in `grant_kit` -- which is why this
imports from there and never from `grant_figures`, a direction that would be a cycle.

Entry points, registered in `grant_figures.JOBS`:
  - `fig_pattern_similarity`
  - `fig_pattern_similarity_per_session`
  - `fig_pattern_delta`
  - `fig_splithalf_matrix`
  - `fig_reliability_verdict`
  - `fig_splithalf_delta`
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

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


def fig_pattern_similarity_per_session(out_dir, min_trials=10):
    """6b: figure 6 unpooled -- one column per post-stroke DAY, animals aligned by day.

    Same construction as figure 6: every panel is scored against ONE half of the pre-stroke trials,
    so the first column (the other pre-stroke half) is the no-lesion expectation and its diagonal is
    the split-half ceiling. Read every later column against that first one.

    A SINGLE SESSION'S MEAN PATTERN IS NOISIER than the pooled one, and at an impaired position it
    can rest on a few dozen trials, so `min_trials` gates each cell and a position below it is blank
    rather than drawn. The pooled figure 6 is the one to quote; this is the one that shows whether a
    pooled cell is a steady state or an average of a collapse and a recovery.
    """
    # NO rng: the pre-stroke split is no longer a random draw over trials but a leave-one-SESSION-out
    # over the days themselves, so nothing here is stochastic.
    made = []
    for _disp, align, wname in _windows():
        variants = _variants(align)
        store = {v: {} for v in variants}
        all_days = set()
        for an in ANIMALS:
            try:
                # THE SHARED POOLING -- see `_pooled_bundle`. Figure 6b and figure 6 now
                # read the same trials by construction, not by two recipes agreeing.
                bd = _pooled_bundle(an, align)
                XE, GE, XU, GU = bd["XE"], bd["GE"], bd["XU"], bd["GU"]
                kept, pre_i, e_pre = bd["kept"], bd["pre_i"], bd["e_pre"]
                not_eng, en, un = bd["not_eng"], bd["en"], bd["un"]
                # THE REFERENCE IS ALL PRE-STROKE TRIALS; the no-lesion column is LEAVE-ONE-SESSION-
                # OUT (corrected 2026-08-25). This used to take a random half of the pooled trials
                # as the reference and the other half as the ceiling, so both came from the SAME
                # DAYS and the ceiling carried no between-session drift at all -- while every post
                # column compares a DIFFERENT day against those days. It was an unreachable ceiling.
                # Now each pre-stroke session in turn is scored against the pool of the OTHERS and
                # the results averaged: one session against a pool, exactly like every post column.
                ref, loo = {}, []
                pre_ids = sorted(pre_i)
                for q in CONF_LABELS:
                    idx = np.flatnonzero(e_pre & (en == q))
                    if len(idx) < 2 * min_trials:
                        continue
                    ref[q] = _mean_pattern(XE[idx])
                for i in pre_ids:
                    held, rest = {}, {}
                    for q in CONF_LABELS:
                        h = XE[(GE == i) & (en == q)]
                        r = XE[e_pre & (GE != i) & (en == q)]
                        if len(h) >= min_trials and len(r) >= min_trials:
                            held[q], rest[q] = _mean_pattern(h), _mean_pattern(r)
                    if held:
                        loo.append((held, rest))
                for v in variants:
                    by_day = {}
                    for i, lab in enumerate(kept):
                        if i in pre_i:
                            continue
                        day = _day(an, lab.split("_")[-1])
                        pat = {}
                        for q in CONF_LABELS:
                            _me, _mu = _class_select(v, (GE == i) & (en == q),
                                                     (GU == i) & (un == q), not_eng)
                            parts = [XE[_me]] + ([XU[_mu]] if _mu.any() else [])
                            Xp = [z for z in parts if len(z)]
                            if Xp:
                                Z = np.vstack(Xp)
                                if len(Z) >= min_trials:
                                    pat[q] = _mean_pattern(Z)
                        if pat:
                            by_day[day] = pat
                            all_days.add(day)
                    store[v][an] = (ref, loo, by_day)
            except Exception as ex:                                       # noqa: BLE001
                print(f"  !! 6b {an} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
        if not all_days:
            continue
        days = sorted(all_days)
        for v in variants:
            ncol = 1 + len(days)
            # Height and hspace raised with `_colw` -- see the note in `_draw_5c`.
            fig, axes = plt.subplots(len(ANIMALS), ncol, figsize=(_colw() * ncol + 1.2, 9.5),
                                     gridspec_kw={"hspace": 0.60},
                                     squeeze=False)
            im = None
            for ri, an in enumerate(ANIMALS):
                got = store[v].get(an)
                for ci in range(ncol):
                    ax = axes[ri][ci]
                    if not got:
                        ax.axis("off")
                        continue
                    ref, loo, by_day = got
                    # COLUMN 0 averages the per-held-out-session matrices; every later column is one
                    # post-stroke session against the pooled pre-stroke reference. Both are "one
                    # session against a pool of other days", which is the comparison being made.
                    pairs = loo if ci == 0 else ([(by_day[days[ci - 1]], ref)]
                                                 if days[ci - 1] in by_day else [])
                    if not pairs:
                        ax.axis("off")
                        continue
                    Ms = []
                    for src, rf in pairs:
                        m1 = np.full((len(CONF_LABELS), len(CONF_LABELS)), np.nan)
                        for i, pp in enumerate(CONF_LABELS):
                            for j, q in enumerate(CONF_LABELS):
                                if src.get(pp) is None or rf.get(q) is None:
                                    continue
                                m1[i, j] = float(np.corrcoef(src[pp], rf[q])[0, 1])
                        Ms.append(m1)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN cells
                        M = np.nanmean(np.stack(Ms), axis=0)
                    im = ax.imshow(np.ma.masked_invalid(M), vmin=-1, vmax=1, cmap="RdBu_r")
                    ax.set_xticks(range(len(CONF_LABELS)))
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                       rotation=90, fontsize=9)
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
                    diag = np.nanmean(np.diag(M))
                    # TWO LINES, and the head shortened to "PRE". The column-0 title was several times wider
                    # than a day column's ("PRE, leave-1-out  diag 0.77" against "day 1  0.59"), so it
                    # overran into column 1 and reached left into its own row's ylabel -- the ax0/8/16/24
                    # stride in the layout report, one fault per row across three whole families. What
                    # "PRE" means is in the header of every one of these figures.
                    head = "PRE" if ci == 0 else f"day {days[ci - 1]}"
                    ax.set_title(f"{head}\ndiag {diag:.2f}", fontsize=10,
                                 fontweight="bold" if ci == 0 else "normal")
                    if ci == 0:
                        ax.set_ylabel(f"{an}\nthis position", fontsize=11, fontweight="bold")
            if im is None:
                plt.close(fig)
                continue
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label="pattern correlation r")
            cls = _class_note(v)
            _suptitle(fig, f"Mean-pattern similarity session by session — {wname} window\n"
                         f"Post-stroke class: {cls}.  Rows within a panel = the pattern being "
                         f"described; columns within a panel = the PRE-STROKE reference.\n"
                         "FIRST COLUMN is the no-lesion expectation and its diagonal is the "
                         "CEILING: each pre-stroke session in turn scored against the pool of the "
                         "OTHERS, averaged -- one session against other days, exactly like every "
                         "post column.\nIt is NOT 1.0 and must not be read against 1.0: two "
                         "pre-stroke days differ by ordinary drift. 'diag' above each panel is the "
                         "mean of that panel's diagonal.\nColumns are DAYS FROM LESION; a blank is "
                         "a session that animal does not have.", fontsize=9.5)
            _footer(fig)
            p = _out(out_dir, f"grant_6b_pattern_per_session_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made[0] if len(made) == 1 else (made or None)
#: Resamples for the pattern-similarity intervals and null. 400 is enough for a 95% percentile
#: interval and keeps a full render (4 animals x 3 windows x 2 variants) inside a few minutes.
N_BOOT = 400
def _strat_mean(by_session, rng):
    """One stratified bootstrap mean: resample TRIALS WITHIN each session, sessions kept fixed.

    SESSIONS ARE NOT RESAMPLED, and that is the whole point (Priya, 2026-08-25: "session-level is
    problematic with things dynamically changing over sessions"). A session bootstrap assumes days
    are exchangeable draws from one distribution; they are not -- PS94's frozen accuracy runs
    0.39 -> 0.76 across six days and PS95 sits at 0.81-0.84 then falls to 0.60. Resampling days
    would fold that trajectory into "sampling noise" and produce an interval for a post-stroke
    state that does not exist.

    Conditioning on the sessions instead makes the interval mean: "how well determined is this
    estimate GIVEN THESE DAYS". It does NOT license generalisation to other days -- that question
    needs the trajectory, which is what the per-session figure shows rather than summarises.
    """
    tot = None
    n = 0
    for X in by_session:
        if not len(X):
            continue
        idx = rng.integers(0, len(X), len(X))       # trials within this session, count preserved
        s = X[idx].sum(0)
        tot = s if tot is None else tot + s
        n += len(X)
    return None if not n else tot / n
def _pattern_stats(post_by_session, ref_by_session, labels, rng, n_boot=N_BOOT):
    """(r, lo, hi, null_hi) per (row, col), all from the same stratified resampling scheme.

    `null_hi` is the 97.5th percentile of |r| under a POSITION-LABEL PERMUTATION: the post-stroke
    trials keep their session, their count and the global post-stroke pattern, and only which
    position they belong to is shuffled. That is the right null for an off-diagonal claim, because
    it asks "is there position-specific structure here" rather than "is r different from zero" --
    and positions are intrinsically similar, so a zero-null would call almost every cell
    significant.
    """
    K = len(labels)
    obs = np.full((K, K), np.nan)
    ref_mean = {q: _strat_mean(ref_by_session.get(q, []), rng) for q in labels}
    post_mean = {p: _strat_mean(post_by_session.get(p, []), rng) for p in labels}
    for i, p in enumerate(labels):
        for j, q in enumerate(labels):
            if post_mean.get(p) is None or ref_mean.get(q) is None:
                continue
            obs[i, j] = float(np.corrcoef(post_mean[p], ref_mean[q])[0, 1])
    boots = np.full((n_boot, K, K), np.nan)
    nulls = np.full((n_boot, K, K), np.nan)
    # PER-SESSION STACKS, built ONCE. The first version rebuilt Python lists of single trial rows on
    # every resample -- O(sessions x trials) list work per iteration, minutes per animal. Stacking
    # each session's trials with a label vector makes a permutation one `rng.permutation` and six
    # boolean means in numpy.
    stacks = []
    for si in range(max((len(v) for v in post_by_session.values()), default=0)):
        Xs, ys = [], []
        for p in labels:
            v = post_by_session.get(p, [])
            if si < len(v) and len(v[si]):
                Xs.append(v[si])
                ys += [p] * len(v[si])
        if Xs:
            stacks.append((np.vstack(Xs), np.array(ys)))
    for b in range(n_boot):
        rm = {q: _strat_mean(ref_by_session.get(q, []), rng) for q in labels}
        pm = {p: _strat_mean(post_by_session.get(p, []), rng) for p in labels}
        for i, p in enumerate(labels):
            for j, q in enumerate(labels):
                if pm.get(p) is None or rm.get(q) is None:
                    continue
                boots[b, i, j] = float(np.corrcoef(pm[p], rm[q])[0, 1])
        tot, cnt = {}, {}
        for Xs, ys in stacks:
            yp = ys[rng.permutation(len(ys))]        # labels shuffled WITHIN this session
            for p in labels:
                m = yp == p
                if m.any():
                    tot[p] = Xs[m].sum(0) if p not in tot else tot[p] + Xs[m].sum(0)
                    cnt[p] = cnt.get(p, 0) + int(m.sum())
        pmn = {p: tot[p] / cnt[p] for p in tot if cnt.get(p)}
        for i, p in enumerate(labels):
            for j, q in enumerate(labels):
                if pmn.get(p) is None or rm.get(q) is None:
                    continue
                nulls[b, i, j] = float(np.corrcoef(pmn[p], rm[q])[0, 1])
    with np.errstate(invalid="ignore"):
        lo = np.nanpercentile(boots, 2.5, axis=0)
        hi = np.nanpercentile(boots, 97.5, axis=0)
        null_hi = np.nanpercentile(np.abs(nulls), 97.5, axis=0)
    # `boots` is returned so a CALLER can difference two matrices draw-by-draw. Differencing the
    # published CIs instead would be wrong: the post and baseline panels share the same resampled
    # reference, so their errors are correlated and an unpaired difference overstates the interval.
    return obs, lo, hi, null_hi, boots
def _mean_pattern(X):
    return X.mean(0) if len(X) else None
def fig_pattern_similarity(out_dir, min_trials=10):
    """6: WITHIN- and ACROSS-position pattern similarity, model-free.

    Priya, 2026-08-25. The complement to the coding directions: correlate the post-stroke MEAN
    ACTIVITY PATTERN at each position against the pre-stroke mean pattern at EVERY position. No
    discriminant, no contrast -- which is exactly why it survives where the coding directions do
    not. A pairwise axis needs trials on BOTH sides, so it fails at the positions the lesion broke;
    a mean pattern for far_R is perfectly well defined from 400 miss trials with no partner at all.

    DIAGONAL = within-position ("is this still the same code"). OFF-DIAGONAL = across-position
    ("what does it look like instead"). `pattern_similarity` in `poststroke_compare` already computes
    the diagonal; the off-diagonal is what is new here.

    THE BASELINE PANEL IS NOT OPTIONAL. Positions are intrinsically similar before any lesion, so a
    raw r of 0.8 between post far_R and pre far_L means nothing on its own. Both panels are scored
    against the SAME reference -- one half of the pre-stroke trials -- so:
        LEFT  corr(other pre-stroke half at P, reference at Q): the no-lesion expectation, and its
              DIAGONAL is the split-half reliability, i.e. the ceiling this measure can reach.
        RIGHT corr(post-stroke at P, reference at Q).
    Comparing the right panel to the left is the only way to read it; comparing it to 1.0 is not.

    CLASS VARIANTS. `lick` uses post-stroke trials with a lick. `working` adds miss-while-working
    (everything but the terminal quit period) and exists for ENL and cue ONLY -- in the lick window
    a no-lick trial is placed at the CUE, so pooling the two classes there would average patterns
    from two different times and call the result a position effect.

    WHAT THIS SHARES WITH NOTHING ELSE, and its weakness: mean-pattern correlation is sensitive to
    global gain and offset, so a uniform post-stroke amplitude change moves every cell together.
    The coding directions are immune to that by construction (unit vectors). The two measures agree
    or they do not, and agreement is the claim worth making.
    """
    made = []
    for _disp, align, wname in _windows():
        variants = _variants(align)
        store = {v: {} for v in variants}
        for an in ANIMALS:
            try:
                # THE SHARED POOLING -- see `_pooled_bundle`, whose docstring was written
                # when this site and 6b were character-identical. They share the object now.
                #
                # No e_pre mask is unpacked: the pre-stroke split here is over SESSION IDS,
                # not over a pooled trial mask, so trials are selected per session below.
                bd = _pooled_bundle(an, align)
                XE, GE, XU, GU = bd["XE"], bd["GE"], bd["XU"], bd["GU"]
                kept, pre_i = bd["kept"], bd["pre_i"]
                not_eng, en, un = bd["not_eng"], bd["en"], bd["un"]
                # SPLIT THE PRE-STROKE **SESSIONS**, NOT THE TRIALS (corrected 2026-08-25).
                #
                # Until now this drew a random half of the pooled pre-stroke TRIALS as the
                # reference and the other half as the no-lesion expectation. Both halves then came
                # from the SAME DAYS, so the baseline panel contained within-session trial noise
                # and NO between-session drift whatever -- while the post panel compares different
                # days against those days. The baseline was therefore an upper bound no
                # across-session comparison can reach, and "post minus baseline is negative at
                # every position" was measured against it.
                #
                # Splitting by session makes both sides "one set of DAYS against another set of
                # DAYS", which is what the post panel is. Trials stay GROUPED BY SESSION because
                # the stratified bootstrap resamples within sessions and a mean has already thrown
                # that structure away.
                ref, other = {}, {}
                pre_ids = sorted(pre_i)
                # SEEDED PER (animal, alignment), not drawn from one stream shared across the
                # whole figure. This picks WHICH pre-stroke sessions form the reference half, so
                # an order-dependent stream meant the cue panel's reference set depended on how
                # many draws the pre-cue panel happened to take before it -- and rendering one
                # alignment alone, as a parallel worker does, would silently choose a different
                # split than the same alignment got in a full serial run. Not CI noise: a
                # different set of sessions in the reference.
                sh = np.random.default_rng(_seed(an, align, "pre-split")).permutation(len(pre_ids))
                g_ref = {pre_ids[k] for k in sh[:max(1, len(pre_ids) // 2)]}
                g_oth = {pre_ids[k] for k in sh[max(1, len(pre_ids) // 2):]}
                for p in CONF_LABELS:
                    a = [XE[(GE == i) & (en == p)] for i in pre_ids if i in g_ref]
                    b = [XE[(GE == i) & (en == p)] for i in pre_ids if i in g_oth]
                    a = [z for z in a if len(z)]
                    b = [z for z in b if len(z)]
                    if sum(len(z) for z in a) < min_trials or sum(len(z) for z in b) < min_trials:
                        continue
                    ref[p], other[p] = a, b
                post_ids = [i for i in range(len(kept)) if i not in pre_i]
                for v in variants:
                    postm = {}
                    for p in CONF_LABELS:
                        by_sess, tot = [], 0
                        for i in post_ids:
                            _me, _mu = _class_select(v, (GE == i) & (en == p),
                                                     (GU == i) & (un == p), not_eng)
                            parts = [XE[_me]] + ([XU[_mu]] if _mu.any() else [])
                            keep = [q for q in parts if len(q)]
                            if keep:
                                Z = np.vstack(keep)
                                by_sess.append(Z)
                                tot += len(Z)
                        if tot >= min_trials:
                            postm[p] = by_sess
                    store[v][an] = (ref, other, postm)
            except Exception as ex:                                       # noqa: BLE001
                print(f"  !! 6 {an} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
        for v in variants:
            # TICK COUNT BOUNDED BY PANEL WIDTH, not by the data range (other window,
            # 2026-08-28). matplotlib's default locator asks for a tick every 0.25 over
            # whatever span the data happens to have: a delta-r range of -1.25..+0.5 wants
            # 8 labels at ~0.45in = 3.6in inside a 2.4in panel. It bites only the _working
            # variants and only for some animals, because it depends on the range -- which
            # is exactly why it survived a spot check.
            fig, axes = plt.subplots(len(ANIMALS), 3, figsize=(10.4, 12.6), squeeze=False,
                                     gridspec_kw={"hspace": 0.60})
            drew = False
            for ri, an in enumerate(ANIMALS):
                got = store[v].get(an)
                if not got:
                    for ci in range(3):
                        axes[ri][ci].axis("off")
                    continue
                ref, other, postm = got
                # SAME SEED FOR BOTH PANELS so the reference is resampled identically and the
                # post-minus-baseline difference can be taken draw by draw.
                seed = _seed(an, align, v)
                base_obs, _bl, _bh, base_null, base_bt = _pattern_stats(
                    other, ref, CONF_LABELS, np.random.default_rng(seed))
                post_obs, _pl, _ph, post_null, post_bt = _pattern_stats(
                    postm, ref, CONF_LABELS, np.random.default_rng(seed))
                for ci, (M, NH, ptitle) in enumerate(
                        ((base_obs, base_null, "PRE-stroke, other half\n(no-lesion expectation)"),
                         (post_obs, post_null, "POST-stroke"))):
                    ax = axes[ri][ci]
                    im = ax.imshow(np.ma.masked_invalid(M), vmin=-1, vmax=1, cmap="RdBu_r")
                    for i in range(len(CONF_LABELS)):
                        for j in range(len(CONF_LABELS)):
                            if not np.isfinite(M[i, j]):
                                continue
                            _txt(ax, j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                                    fontsize=7.5, color="k")
                            # RING = beats the position-shuffled null. Not "r != 0": the null keeps
                            # the global post-stroke pattern and shuffles only WHICH position a
                            # trial belongs to, so a ring means position-specific structure.
                            if np.isfinite(NH[i, j]) and abs(M[i, j]) > NH[i, j]:
                                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                                           edgecolor="lime", lw=1.6))
                    ax.set_xticks(range(len(CONF_LABELS)))
                    ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                       rotation=90, ha="center", fontsize=9.5)
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9.5)
                    ax.set_title(f"{an if ci == 0 else ''}  {ptitle}", fontsize=11,
                                 fontweight="bold" if ci == 0 else "normal")
                    if ci == 0:
                        ax.set_ylabel("this position's pattern", fontsize=9.5)
                    if ri == len(ANIMALS) - 1:
                        ax.set_xlabel("vs PRE-STROKE reference at", fontsize=9.5)
                    drew = True
                # THIRD PANEL: the claim, with an interval. Own-position r post MINUS baseline,
                # differenced draw by draw so the shared reference cancels.
                ax = axes[ri][2]
                d = np.array([post_bt[:, k, k] - base_bt[:, k, k]
                              for k in range(len(CONF_LABELS))])
                y = np.arange(len(CONF_LABELS))
                with np.errstate(invalid="ignore"):
                    med = np.nanmedian(d, axis=1)
                    lo = np.nanpercentile(d, 2.5, axis=1)
                    hi = np.nanpercentile(d, 97.5, axis=1)
                ok = np.isfinite(med)
                ax.errorbar(med[ok], y[ok], xerr=[med[ok] - lo[ok], hi[ok] - med[ok]],
                            fmt="o", ms=5, color="#b2182b", capsize=3, lw=1.2)
                ax.axvline(0, color="k", lw=1.0)
                ax.set_yticks(y)
                ax.set_yticklabels([])
                ax.set_ylim(len(CONF_LABELS) - 0.5, -0.5)
                ax.set_title("post − baseline (own position)", fontsize=11)
                ax.grid(alpha=0.25, lw=0.5)
                # AT MOST FOUR X TICKS, and smaller labels. The default locator picks a tick every
                # 0.25 over whatever span the data has, so a delta-r range of -1.25..+0.5 asks for
                # eight labels at ~0.45in inside a 2.4in panel. Four at 8.5pt is ~1.9in and fits.
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.tick_params(axis="x", labelsize=8.5)
                if ri == len(ANIMALS) - 1:
                    ax.set_xlabel("Δr, 95% stratified bootstrap", fontsize=9.5)
            if not drew:
                plt.close(fig)
                continue
            fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03, label="pattern correlation r")
            cls = _class_note(v)
            _suptitle(fig, f"Mean-pattern similarity, within and across positions — {wname} window\n"
                         f"Post-stroke class: {cls}.  Rows = the pattern being described, columns = "
                         f"the PRE-STROKE reference it is correlated with.\n"
                         "DIAGONAL = is it still the same code. OFF-DIAGONAL = what it looks like "
                         "instead. READ THE RIGHT PANEL AGAINST THE LEFT, not against 1.0:\n"
                         "positions are intrinsically similar before any lesion, and the left "
                         "diagonal is the split-half ceiling this measure can reach.", fontsize=9)
            _footer(fig)
            p = Path(out_dir) / f"grant_6_pattern_{align}_{v}.png"
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made[0] if len(made) == 1 else (made or None)
#: Reliability below this makes the disattenuation ratio unstable -- dividing by sqrt(0.1) inflates
#: both the estimate and its error without bound. Same threshold and same reasoning as the coding
#: directions' MIN_REL, so the two analyses declare a cell uninterpretable on the same criterion.
MIN_REL = 0.5
#: Split-half draws per cell. The split is random, so one draw is itself a noisy estimate of the
#: reliability; averaging over draws costs nothing (a correlation of two means) and removes it.
SPLIT_REPS = 40
def _split_half(Z, rng, reps=SPLIT_REPS):
    """Mean correlation between the means of two disjoint halves of ``Z``. NaN under 4 trials.

    This is the reliability of the MEAN PATTERN this many trials can support -- the ceiling any
    correlation involving that mean can reach, and the quantity figure 6 does not show.
    """
    n = len(Z)
    if n < 4:
        return np.nan
    h = n // 2
    rs = []
    for _ in range(reps):
        idx = rng.permutation(n)
        a, b = Z[idx[:h]].mean(0), Z[idx[h:2 * h]].mean(0)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(rs)) if rs else np.nan
def _reliability(Z, rng, reps=SPLIT_REPS):
    """Reliability of the mean of ALL of ``Z``, not of half of it.

    `_split_half` correlates two means built from n/2 trials each, so it estimates the reliability
    of an n/2-trial mean -- but the quantity figure 6 correlates is the mean of all n. Spearman-Brown
    projects the split-half value up to the full length: rel(n) = 2r / (1 + r).

    THIS IS NOT COSMETIC. Skipping it makes every reliability too LOW, and since 7b divides by
    sqrt(rel_post * rel_pre), too low a denominator makes every disattenuated correlation too HIGH --
    i.e. it would systematically manufacture the "the code moved" verdict the panel exists to test.
    """
    r = _split_half(Z, rng, reps)
    if not np.isfinite(r):
        return np.nan
    if r <= -1:
        return np.nan
    return float(2.0 * r / (1.0 + r))
def _split_half_matrix(src, rng, labels=None):
    """6x6 WITHIN-set split-half matrix, SYMMETRIC by construction.

    The diagonal is each position's own reliability -- corr(half A at P, half B at P), which must
    use the two halves because a mean correlated with itself is 1 by definition. The off-diagonal is
    how similar two positions look to each other MEASURED IN THE SAME SESSION, the within-session
    counterpart of figure 6's off-diagonal.

    WHY IT IS AVERAGED OVER BOTH PAIRINGS (Priya, 2026-08-25: "why aren't the fig 7 matrices
    symmetrical about the diagonal?"). The first version took M[P,Q] = corr(A_P, B_Q) and left it,
    so cell (far_R, close_L) and cell (close_L, far_R) were corr(A_far_R, B_close_L) and
    corr(A_close_L, B_far_R) -- two estimates of ONE quantity, differing only in which random half
    of each position's trials landed on which side. The asymmetry was therefore pure estimation
    noise being drawn as if it were structure, in a figure whose entire job is to say how much
    estimation noise there is. Averaging the two pairings is symmetric, has the same expectation, is
    still cross-validated (no half is ever correlated with itself) and halves the variance.

    The remaining asymmetry is zero by construction; if the two pairings disagreed a lot that fact
    is worth knowing, so `_split_half_asymmetry` reports it separately rather than smuggling it into
    the picture.
    """
    labels = labels or CONF_LABELS
    n = len(labels)
    M = np.full((n, n), np.nan)
    halves = {}
    for q in labels:
        Z = src.get(q)
        if Z is None or len(Z) < 4:
            continue
        idx = rng.permutation(len(Z))
        h = len(Z) // 2
        halves[q] = (Z[idx[:h]].mean(0), Z[idx[h:2 * h]].mean(0))

    def _r(u, v):
        return float(np.corrcoef(u, v)[0, 1]) if (np.std(u) and np.std(v)) else np.nan

    for i, p in enumerate(labels):
        if p not in halves:
            continue
        for j, q in enumerate(labels):
            if q not in halves or j < i:
                continue
            if i == j:
                M[i, j] = _r(halves[p][0], halves[p][1])
                continue
            both = [_r(halves[p][0], halves[q][1]), _r(halves[q][0], halves[p][1])]
            both = [x for x in both if np.isfinite(x)]
            M[i, j] = M[j, i] = float(np.mean(both)) if both else np.nan
    return M
def fig_splithalf_matrix(out_dir, min_trials=10):
    """7: the WITHIN-session split-half matrix for every post-stroke session.

    Figure 6 asks whether the post-stroke pattern still matches the pre-stroke one. It cannot
    distinguish a code that MOVED from a code that merely became NOISIER, because a correlation
    between two means is bounded above by the reliability of each mean, and a session with more
    variable responses has a lower ceiling at every position at once. That is a live alternative
    here: the headline result is that own-position similarity drops at EVERY position with far_R
    largest, which is precisely the signature of a global change in how repeatable the responses are.

    This figure supplies the missing ceiling. Both halves come from the SAME session, so the
    diagonal is that session's own reliability and nothing about the lesion, the pre-stroke
    reference or the alignment enters it. Off-diagonal is how similar the positions look to each
    other WITHIN one session.

    Read it beside figure 6 panel by panel: where 6's diagonal falls and this diagonal does not, the
    code moved; where both fall together, the code is noisier and 6 cannot see the difference. 7b
    does that division explicitly.
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
                for ci in range(ncol):
                    ax = axes[ri][ci]
                    src = None
                    if got:
                        pre_by_sess, by_day = got
                        src = pre_by_sess if ci == 0 else by_day.get(days[ci - 1])
                    if not src:
                        ax.axis("off")
                        continue
                    rng = np.random.default_rng(_seed(an, align, v, ci))
                    if ci == 0:
                        # COLUMN 0 IS ONE PRE-STROKE SESSION AT A TIME, AVERAGED -- not the pooled
                        # set. Pooling six sessions gives the reliability of a six-session mean and
                        # compares it against one-session post-stroke means, so most of the gap
                        # would be trial count rather than the lesion. Averaging per-session
                        # matrices matches the units of every other column in the row.
                        Ms = [_split_half_matrix(p, rng) for p in pre_by_sess.values()]
                        M = np.nanmean(np.stack(Ms), axis=0) if Ms else np.full((6, 6), np.nan)
                        n_med = int(np.median([len(z) for p in pre_by_sess.values()
                                               for z in p.values()] or [0]))
                    else:
                        M = _split_half_matrix(src, rng)
                        n_med = int(np.median([len(z) for z in src.values()] or [0]))
                    im = ax.imshow(np.ma.masked_invalid(M), vmin=-1, vmax=1, cmap="RdBu_r")
                    ax.set_xticks(range(len(CONF_LABELS)))
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                       rotation=90, fontsize=9)
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
                    head = "PRE" if ci == 0 else f"day {days[ci - 1]}"
                    # 'sh', NOT 'rel': this is the split-half correlation itself, the reliability of
                    # a HALF-length mean. 7b applies the Spearman-Brown step that turns it into the
                    # reliability of the full mean it actually divides by.
                    # n IS PRINTED because sh depends on it, and a reader comparing two panels needs
                    # to know whether they rest on comparable amounts of data.
                    ax.set_title(f"{head}\nsh {np.nanmean(np.diag(M)):.2f}  n{n_med}", fontsize=9.5,
                                 fontweight="bold" if ci == 0 else "normal")
                    if ci == 0:
                        ax.set_ylabel(f"{an}\nhalf A at", fontsize=11, fontweight="bold")
            if im is None:
                plt.close(fig)
                continue
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label="split-half correlation r")
            cls = _class_note(v)
            _suptitle(fig, 
                f"WITHIN-session split-half pattern similarity — {wname} window\n"
                f"Post-stroke class: {cls}.  BOTH HALVES COME FROM THE SAME SESSION: no lesion "
                f"comparison, no pre-stroke reference, no alignment inference enters this.\n"
                f"SYMMETRIC BY CONSTRUCTION: an off-diagonal cell averages both half-pairings, "
                "since (P,Q) and (Q,P) estimate one quantity and differ only by split noise.\n"
                f"DIAGONAL ('sh' above each panel) = that session's own reliability, i.e. the "
                f"CEILING figure 6's correlations can reach. Off-diagonal = how similar the "
                f"positions look to each other within one session.\n"
                f"If a post-stroke diagonal is low HERE, figure 6's matching low value is a noisier "
                "code, not a moved one. Columns are days from lesion.\n"
                f"FIRST COLUMN IS ONE PRE-STROKE SESSION AT A TIME, AVERAGED -- not the pooled set: "
                f"split-half reliability rises with trial count, and pooling six sessions would "
                f"compare a six-session mean with one-session post-stroke means. 'n' is the median "
                f"trials per position in that panel.", fontsize=9.0)
            _footer(fig)
            p = _out(out_dir, f"grant_7_splithalf_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made
def _disattenuated_ci(align, variant, min_trials=10, n_boot=200):
    """{animal: {day: {position: (lo, hi)}}} on the DISATTENUATED own-position similarity.

    This is figure 7b's right-hand panel -- the number the whole "the code MOVED rather than got
    noisier" reading rests on -- and it had no uncertainty at all. It is a RATIO of three estimated
    quantities, raw / sqrt(rel_post * rel_pre), so its sampling distribution is not the raw
    correlation's and cannot be guessed from it: at low reliability the denominator is itself noisy
    and the ratio is skewed, which is exactly the regime the impaired positions sit in.

    Blocks resampled within session, sessions held fixed, and the reference resampled in the SAME
    draw as the day so the two share their noise -- the convention used by every other interval in
    this module.

    A draw whose reliability falls below MIN_REL on either side is DISCARDED rather than clipped:
    the point estimate suppresses those cells, so an interval that quietly included them would not
    describe the number printed beside it.
    """
    x_store, days = _collect_7(align, variant, min_trials)
    b_store, _ = _collect_7(align, variant, min_trials, "blk")
    out = {}
    for an in ANIMALS:
        if an not in x_store or an not in b_store:
            continue
        (pre_x, day_x), (pre_b, day_b) = x_store[an], b_store[an]
        # THE PRE-STROKE REFERENCE IS SHARED BY EVERY DAY, so its digest is taken once and folded
        # into each day's key: re-preprocessing a pre-stroke session must invalidate every day of
        # that animal, and re-preprocessing one post-stroke session must invalidate only that day.
        pre_key = _digest(pre_x, pre_b)
        params = (align, variant, min_trials, n_boot, MIN_REL, SPLIT_REPS)
        per_day = {}
        for d in days:
            if d not in day_x:
                continue
            # SEEDED PER DAY, not drawn from one stream shared across the animal's days. A shared
            # stream made a day's interval depend on how many days preceded it in that run -- so
            # the same session gave different numbers depending on what else was rendered, and no
            # per-day result could be cached and reused. One fix, both problems.
            rec = _boot_cached(
                "7b_disatt", (pre_key, day_x[d], day_b[d], params),
                lambda an=an, d=d: _disatt_one(
                    pre_x, pre_b, day_x[d], day_b[d],
                    np.random.default_rng(_seed(an, align, variant, d, "7bci")), n_boot))
            if rec:
                per_day[d] = rec
        if per_day:
            out[an] = per_day
    return out
def _disatt_one(pre_x, pre_b, dx, db, rng, n_boot):
    """One day's disattenuated own-position interval: ``{position: (lo, hi)}``.

    Extracted from `_disattenuated_ci` so a single day is the unit that gets cached. The arithmetic
    is unchanged -- blocks resampled within session, sessions held fixed, and the reference drawn
    in the SAME iteration as the day so the two share their noise.
    """
    draws = {q: [] for q in CONF_LABELS}
    for _ in range(n_boot):
        ref_r = {}
        for s in pre_x:
            got = _block_boot(pre_x[s], pre_b[s], rng)
            for q, Z in got.items():
                ref_r.setdefault(q, []).append(Z)
        ref_r = {q: np.vstack(v) for q, v in ref_r.items()}
        day_r = _block_boot(dx, db, rng)
        if not ref_r or not day_r:
            continue
        for q in CONF_LABELS:
            Z, R = day_r.get(q), ref_r.get(q)
            if Z is None or R is None:
                continue
            rp, rr = _reliability(Z, rng), _reliability(R, rng)
            if not (np.isfinite(rp) and np.isfinite(rr)):
                continue
            if rp < MIN_REL or rr < MIN_REL:
                continue
            m, rm = Z.mean(0), R.mean(0)
            if not (np.std(m) and np.std(rm)):
                continue
            raw = float(np.corrcoef(m, rm)[0, 1])
            draws[q].append(raw / np.sqrt(rp * rr))
    rec = {}
    for q, v in draws.items():
        v = np.array([x for x in v if np.isfinite(x)])
        if len(v) >= n_boot // 4:
            rec[q] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return rec
def fig_reliability_verdict(out_dir, min_trials=10):
    """7b: figure 6's own-position similarity, RAW and DISATTENUATED by both sides' reliability.

    THE ARITHMETIC. A correlation between two independently estimated means is attenuated by each
    side's reliability: E[r_obs] ~ r_true * sqrt(rel_post * rel_ref). Dividing it out estimates the
    correlation the two patterns would have if both were measured without noise -- so a drop that
    survives disattenuation is a code that MOVED, and a drop that disappears was a code measured
    less repeatably. This is the same correction the coding-direction analysis has used since
    2026-08-20, applied to the pattern measure, which has never had it.

    WHERE IT CANNOT BE TRUSTED. Dividing by sqrt(rel) with rel near zero inflates without bound, so
    a cell whose reliability is below MIN_REL on either side is drawn hollow and its number
    suppressed. That is not a formality: an impaired position post-stroke is exactly where trials
    are fewest and reliability lowest, which is exactly where the ratio is least stable -- the
    correction is weakest precisely where the question is sharpest, and the figure has to say so
    rather than print a confident number.

    A disattenuated value may exceed 1. That is expected of a ratio estimator at low reliability and
    is left visible rather than clipped: clipping would hide the instability the hollow marker is
    there to declare.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            store, days = _collect_7(align, v, min_trials)
            if not days:
                continue
            cis = _disattenuated_ci(align, v, min_trials)
            fig, axes = plt.subplots(len(ANIMALS), 3, figsize=(10.8, 9.4), squeeze=False)
            drew = False
            for ri, an in enumerate(ANIMALS):
                got = store.get(an)
                if not got:
                    for ci in range(3):
                        axes[ri][ci].axis("off")
                    continue
                pre_by_sess, by_day = got
                rng = np.random.default_rng(_seed(an, align, v))
                # COLUMN 0 IS THE NO-LESION EXPECTATION, built LEAVE-ONE-SESSION-OUT: each
                # pre-stroke session in turn is scored against the pool of the OTHERS, then the six
                # results are averaged. Disjoint, so it is not a session correlated against itself;
                # and one session against a pool, exactly like every post-stroke column, so the
                # reliability shown for it is a ONE-SESSION reliability and can be compared with
                # them.
                #
                # IT DOES NOT COME OUT AT 1, AND THAT IS THE POINT. PS94 post-cue disattenuates to
                # 0.75-0.86 here, not 1.0, because disattenuating by a WITHIN-session reliability
                # removes within-session trial noise and nothing else: a pre-stroke session's mean
                # pattern also differs from the other sessions' by day-to-day drift, which no
                # within-session split half can see. So this column, not 1.0, is the ceiling the
                # post-stroke columns are to be read against -- the same reason figure 6 has a
                # baseline panel and the same reason the axis work uses a matched null instead of
                # comparing cosines with unity.
                cols = ["PRE"] + [f"d{d}" for d in days]
                # column index -> day, so a cell can find its own interval
                cols_day = [None] + list(days)
                shape = (len(CONF_LABELS), len(cols))
                rel = np.full(shape, np.nan)
                raw = np.full(shape, np.nan)
                dis = np.full(shape, np.nan)

                def _score(scored, ref_trials, rng=rng):
                    """(reliability, raw r, disattenuated r) per position for one scored session."""
                    o_rel = np.full(len(CONF_LABELS), np.nan)
                    o_raw = np.full(len(CONF_LABELS), np.nan)
                    o_dis = np.full(len(CONF_LABELS), np.nan)
                    for i, q in enumerate(CONF_LABELS):
                        Z, R = scored.get(q), ref_trials.get(q)
                        if Z is None or R is None:
                            continue
                        o_rel[i] = _reliability(Z, rng)
                        rr = _reliability(R, rng)
                        m, rm = Z.mean(0), R.mean(0)
                        if np.std(m) and np.std(rm):
                            o_raw[i] = float(np.corrcoef(m, rm)[0, 1])
                        if (np.isfinite(rr) and np.isfinite(o_rel[i])
                                and rr >= MIN_REL and o_rel[i] >= MIN_REL):
                            o_dis[i] = o_raw[i] / np.sqrt(rr * o_rel[i])
                    return o_rel, o_raw, o_dis

                held = [_score(pat, _pre_reference(pre_by_sess, exclude=s))
                        for s, pat in pre_by_sess.items()]
                if held:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN position rows
                        rel[:, 0] = np.nanmean([h[0] for h in held], axis=0)
                        raw[:, 0] = np.nanmean([h[1] for h in held], axis=0)
                        dis[:, 0] = np.nanmean([h[2] for h in held], axis=0)
                full_ref = _pre_reference(pre_by_sess)
                for ci in range(1, len(cols)):
                    a, b, c = _score(by_day.get(days[ci - 1]) or {}, full_ref)
                    rel[:, ci], raw[:, ci], dis[:, ci] = a, b, c
                for ci, (M, ttl, vmin, vmax, cmap) in enumerate((
                        (rel, ("RELIABILITY of that session's own mean\n"
                               "(split half, Spearman-Brown to full length)"),
                         0.0, 1.0, "viridis"),
                        (raw, "RAW similarity to pre-stroke\n(this is figure 6's diagonal)",
                         -1.0, 1.0, "RdBu_r"),
                        (dis, "DISATTENUATED\n(raw / sqrt(rel_post x rel_pre))",
                         -1.0, 1.0, "RdBu_r"))):
                    ax = axes[ri][ci]
                    ax.imshow(np.ma.masked_invalid(M), vmin=vmin, vmax=vmax, cmap=cmap,
                              aspect="auto")
                    for i in range(len(CONF_LABELS)):
                        for j in range(len(cols)):
                            if not np.isfinite(M[i, j]):
                                # A CELL SUPPRESSED BY MIN_REL IS NOT A CELL WITH NO DATA. Mark the
                                # first case so it cannot be read as the second.
                                if ci == 2 and np.isfinite(raw[i, j]):
                                    _txt(ax, j, i, "·", ha="center", va="center", fontsize=11,
                                            color="0.35")
                                continue
                            # THE INTERVAL IS DRAWN, NOT PRINTED (Priya, 2026-09-08). It used to go
                            # under the value as "[lo,hi]" at 6.2pt. Twelve day columns in a ~3.5in
                            # panel is a ~0.29in cell, and that string needs about twice that, so
                            # every interval overlapped its neighbours into unreadable runs like
                            # "[0.21,0.0948,0.0873,0.0818,...]" -- the numbers were on the figure
                            # and could not be read off it, which is worse than omitting them.
                            #
                            # A BOX INSTEAD, AND IT ANSWERS THE FIGURE'S OWN QUESTION. This panel
                            # exists to ask whether a drop SURVIVES disattenuation, i.e. whether the
                            # code moved. So the box marks cells whose 95% interval lies entirely
                            # BELOW that animal-and-position's own PRE ceiling: a drop that the
                            # interval separates from the pre-stroke reference. Cells without a box
                            # are not "no change" -- they are drops the interval cannot resolve,
                            # which at these reliabilities is a real and frequent state.
                            lab = f"{M[i, j]:.2f}"
                            if ci == 2 and j > 0:
                                band = ((cis.get(an) or {}).get(cols_day[j]) or {}).get(
                                    CONF_LABELS[i])
                                # PRE IS THE COMPARATOR, NOT ZERO. Against zero almost every cell
                                # would be "significant" and the mark would carry no information;
                                # the question is always post versus that animal's own pre.
                                if (band and np.isfinite(dis[i, 0])
                                        and np.isfinite(band[1]) and band[1] < dis[i, 0]):
                                    ax.add_patch(plt.Rectangle(
                                        (j - .5, i - .5), 1, 1, fill=False, edgecolor="k",
                                        lw=1.5, zorder=5))
                            _txt(ax, j, i, lab, ha="center", va="center", fontsize=7.5,
                                 color="w" if (ci == 0 and M[i, j] < 0.5) else "k")
                    ax.set_xticks(range(len(cols)))
                    ax.set_xticklabels(cols if ri == len(ANIMALS) - 1 else [], fontsize=9.5)
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9.5)
                    if ci == 0:
                        ax.set_ylabel(an, fontsize=11.5, fontweight="bold")
                    if ri == 0:
                        ax.set_title(ttl, fontsize=11)
                    drew = True
                # THE DENOMINATOR, spelled out. The ratio divides by the reference's reliability as
                # well as the session's, and a reader cannot otherwise tell whether a low
                # disattenuated value came from a weak numerator or a strong denominator.
                rel_ref = {q: _reliability(Z, rng) for q, Z in full_ref.items()}
                axes[ri][0].set_xlabel(
                    "pooled pre-stroke reference reliability: "
                    + "  ".join(f"{q.replace('close_', 'c').replace('far_', 'f')}"
                                f" {rel_ref.get(q, float('nan')):.2f}" for q in CONF_LABELS),
                    fontsize=5.6)
            if not drew:
                plt.close(fig)
                continue
            cls = _class_note(v)
            _suptitle(fig, 
                f"Is the lost code a MOVED code or a NOISIER one? — {wname} window\n"
                f"Post-stroke class: {cls}.  Rows = spout position, columns = days from lesion.\n"
                f"LEFT: how repeatable that session's own pattern is (split half, within session). "
                f"MIDDLE: figure 6's own-position correlation to the pre-stroke reference.\n"
                f"RIGHT: the middle divided by sqrt(rel_post x rel_pre) -- what the correlation "
                f"would be if both means were measured without noise.\n"
                f"A drop that SURVIVES the right panel is a code that moved; a drop that "
                f"DISAPPEARS was a code measured less repeatably. A grey dot = reliability below "
                f"{MIN_REL} on one side, where the ratio is not stable enough to print.\n"
                f"BOXED cell (right panel) = its 95% cluster-bootstrap interval lies entirely "
                f"BELOW that animal-and-position's own PRE value: a drop the interval separates "
                f"from the pre-stroke reference. An unboxed cell is NOT 'no change' -- it is a "
                f"drop the interval cannot resolve, which at these reliabilities is common.\n"
                f"PRE COLUMN IS LEAVE-ONE-SESSION-OUT: each pre-stroke session scored against the "
                f"pool of the others, averaged -- one session against a pool, exactly like every "
                f"post-stroke column, so the columns are comparable.\n"
                f"IT IS THE CEILING, AND IT IS NOT 1.0: disattenuating by a within-session "
                f"reliability removes trial noise and NOT day-to-day drift, which a pre-stroke "
                f"session also carries. Read the post columns against PRE, never against 1.",
                fontsize=9.5)
            fig.tight_layout(rect=(0, 0, 1, 1.0))   # top reserved by _suptitle
            _footer(fig)
            p = Path(out_dir) / f"grant_7b_reliability_{align}_{v}.png"
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made
@lru_cache(maxsize=6)
def _matrices_splithalf(align, variant, min_trials=10):
    """{animal: {"PRE": M, day: M, ...}} of WITHIN-session split-half matrices (figures 7 / 7d)."""
    store, days = _collect_7(align, variant, min_trials)
    out = {}
    for an, (pre_by_sess, by_day) in store.items():
        rng = np.random.default_rng(_seed(an, align, variant))
        d = {}
        base = _nanmean_stack([_split_half_matrix(pat, rng) for pat in pre_by_sess.values()])
        if base is not None:
            d["PRE"] = base
        for day, pat in by_day.items():
            d[day] = _split_half_matrix(pat, rng)
        if d:
            out[an] = d
    return out, days
def _mats_splithalf(_an, rng):
    # WITHIN one set, so the reference argument is unused by design -- figure 7's whole point is
    # that no pre-stroke reference enters a single panel.
    return lambda pat, _ref: _split_half_matrix(pat, rng)
def fig_pattern_delta(out_dir, min_trials=10):
    """6d: figure 6b as DIFFERENCES from the pre-stroke reference.

    Every post-stroke panel minus the leave-one-session-out pre-stroke matrix. Zero means "this day
    looks exactly like one pre-stroke day looks against the others" -- which is the honest null, not
    a correlation of 1.

    READ THE OFF-DIAGONAL AS WELL AS THE DIAGONAL. A negative diagonal cell says the position lost
    its own code; a POSITIVE off-diagonal cell at (far_R, far_L) says far_R trials came to look more
    like pre-stroke far_L than they used to -- a substitution, which the absolute panel shows only if
    the reader remembers what that cell looked like before. The delta is the whole reason the
    substitution result is legible at a glance.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            mats, days = _matrices_pattern(align, v, min_trials)
            if not days or not mats:
                continue
            cis = _delta_cis(align, v, min_trials, _mats_pattern, "6d")
            cls = _class_note(v)
            p = _delta_grid(
                mats, days, out_dir, f"grant_6d_pattern_delta_{align}_{v}.png",
                title=(f"Mean-pattern similarity, CHANGE FROM PRE-STROKE — {wname} window\n"
                       f"Post-stroke class: {cls}.  Column 1 is the pre-stroke reference "
                       f"(leave-one-session-out); every later column is THAT DAY MINUS IT.\n"
                       f"Rows = the pattern being described, columns within a panel = the "
                       f"pre-stroke position it is correlated with. ZERO = indistinguishable from "
                       f"an ordinary pre-stroke day.\nNegative on the DIAGONAL = the position lost "
                       f"its own code. Positive OFF-DIAGONAL = it came to look like a different "
                       "position.\n"
                       "Above each panel: the change in mean diagonal and its 95% BLOCK-BOOTSTRAP "
                       "interval -- the scheduler's ~6-trial position blocks resampled within each "
                       "session,\nsessions NOT resampled (days are not exchangeable while an animal "
                       "recovers), and the baseline resampled in the SAME draw so the difference is "
                       "taken draw by draw."),
                abs_label="pre-stroke r", delta_label="change in r vs pre-stroke",
                vmin=-1, vmax=1, cmap="RdBu_r", dmax=1.0, summary=_diag,
                ylab="this position", cis=cis)
            if p:
                made.append(p)
    return made
def fig_splithalf_delta(out_dir, min_trials=10):
    """7d: figure 7 as DIFFERENCES from the pre-stroke reference.

    The within-session split-half matrix minus the per-pre-session average. Zero means this session
    reproduces its own patterns exactly as repeatably as a pre-stroke session did.

    THIS IS THE CONTROL FIGURE IN ITS MOST DIRECT FORM. If the pattern deltas in 6d were really a
    reliability story, the diagonal here would fall by a comparable amount at the same positions on
    the same days. Where 6d falls and this does not, the code moved.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            mats, days = _matrices_splithalf(align, v, min_trials)
            if not days or not mats:
                continue
            cis = _delta_cis(align, v, min_trials, _mats_splithalf, "7d")
            cls = _class_note(v)
            p = _delta_grid(
                mats, days, out_dir, f"grant_7d_splithalf_delta_{align}_{v}.png",
                title=(f"WITHIN-session split-half similarity, CHANGE FROM PRE-STROKE — {wname} "
                       f"window\nPost-stroke class: {cls}.  Column 1 is the average over pre-stroke "
                       f"sessions; every later column is THAT DAY MINUS IT.\n"
                       f"BOTH HALVES COME FROM THE SAME SESSION, so nothing about the lesion or the "
                       f"pre-stroke reference enters a single panel -- only how repeatable that "
                       f"day's own patterns are.\nA diagonal that falls HERE as much as it falls in "
                       f"6d means a noisier code; a 6d fall without one here means a MOVED code. "
                       f"The number above each panel is the change in mean diagonal."),
                abs_label="pre-stroke split-half r", delta_label="change in split-half r",
                vmin=-1, vmax=1, cmap="RdBu_r", dmax=1.0, summary=_diag,
                ylab="half A at", cis=cis)
            if p:
                made.append(p)
    return made
