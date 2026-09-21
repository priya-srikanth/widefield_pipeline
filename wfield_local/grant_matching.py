"""GRANT FIGURES — MATCHING

Best-match destination -- where a lost position's code went, if anywhere.

Split out of `grant_figures` on 2026-09-21, one module per figure family. THE GROUPING IS FROM THE
CALL GRAPH, not from the names: every function here is reached from this family's entry points and
from no other family's. Anything shared with a sibling lives in `grant_kit` -- which is why this
imports from there and never from `grant_figures`, a direction that would be a cycle.

Entry points, registered in `grant_figures.JOBS`:
  - `fig_best_match`
  - `fig_best_match_by_session`
"""
from __future__ import annotations

from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

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


@lru_cache(maxsize=6)
def _pre_loo_matrices(align, variant, min_trials=10):
    """{animal: [one matrix per held-out pre-stroke session]} -- the ceiling as SESSIONS, not a mean.

    `_matrices_pattern` averages these eleven matrices into a single "PRE", which is right for a
    heatmap of typical values and WRONG for anything that COUNTS. Figure 10 counted argmax over the
    average and got 6/6 in every animal: a ceiling pinned at 100% by construction, printed directly
    beneath a caption instructing the reader never to compare against 100%. Averaging removes the
    per-session noise the post-stroke columns still carry, so the two panels were not like for like
    and the post-stroke deficit was measured against an unreachable standard.
    """
    store, _days = _collect_7(align, variant, min_trials)
    out = {}
    for an, (pre_by_sess, _by_day) in store.items():
        loo = [_corr_matrix(_means(pat), _means(_pre_reference(pre_by_sess, exclude=s)))
               for s, pat in pre_by_sess.items()]
        if loo:
            out[an] = loo
    return out
@lru_cache(maxsize=6)
def _match_tables(align, variant, min_trials=10):
    """{animal: (pre counts 6x6, post counts 6x6, {day: (acc, mean rank)})}.

    Counts how often each post-stroke position's BEST MATCH is each pre-stroke position, pooled over
    sessions. The pre-stroke table is the same thing computed leave-one-session-out and is the
    ceiling: even with no lesion a held-out day does not always match itself best.
    """
    mats, days = _matrices_pattern(align, variant, min_trials)
    loo_all = _pre_loo_matrices(align, variant, min_trials)
    out = {}
    for an, d in mats.items():
        n = len(CONF_LABELS)
        pre_C, post_C = np.zeros((n, n)), np.zeros((n, n))
        per_day = {}
        # ONE COUNT PER HELD-OUT PRE-STROKE SESSION, exactly as the post panel counts one per day.
        # Counting argmax over the AVERAGE of these matrices instead gave 6/6 in every animal --
        # averaging eleven sessions removes the noise a single session has, so the "ceiling" was
        # 100% by construction and the post-stroke panel was being read against perfection.
        for base in loo_all.get(an, []):
            b, _r = _best_match(base)
            for i, j in enumerate(b):
                if j >= 0:
                    pre_C[i, j] += 1
        for day in days:
            M = d.get(day)
            if M is None:
                continue
            b, r = _best_match(M)
            for i, j in enumerate(b):
                if j >= 0:
                    post_C[i, j] += 1
            hit = [(i == j) for i, j in enumerate(b) if j >= 0]
            per_day[day] = (float(np.mean(hit)) if hit else np.nan,
                            float(np.nanmean(r)) if np.isfinite(r).any() else np.nan)
        out[an] = (pre_C, post_C, per_day)
    return out, days
def fig_best_match(out_dir, min_trials=10):
    """10: which PRE-STROKE position does each post-stroke position match BEST?

    Priya, 2026-08-26: "I'm not yet sure that the diagonal is all that matters -- is there a way to
    take into account the other cross-correlations?" This is that figure. Every panel above reduces a
    row to its diagonal; this reduces it to its ARGMAX, which uses all six entries and answers "moved
    where" rather than only "moved".

    LEFT: the pre-stroke ceiling, leave-one-session-out -- how often a held-out pre-stroke day
    matches itself best. It is not 6/6, and reading the right panel against 6/6 rather than against
    this would overstate everything.
    MIDDLE: the same over post-stroke sessions. A row's mass moving OFF the diagonal names the
    substitute directly: far_R landing on far_L is the substitution the coding directions and the
    off-diagonal of figure 6 both report, here as a count.
    RIGHT: per day, the fraction of positions whose best match is themselves, and the mean RANK of
    the true position among the six. Rank degrades gracefully where the fraction is all-or-nothing --
    a position that slips from first to second is not the same as one that slips to sixth.

    IMMUNE TO THE AMPLITUDE TERM. Argmax and rank do not change under a monotone transform of a row,
    and the uniform row shifts in figures 8/8d are exactly that. So this summary cannot be moved by
    the gain change that 8b exists to rule out.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            tables, days = _match_tables(align, v, min_trials)
            if not tables or not days:
                continue
            fig, axes = plt.subplots(len(ANIMALS), 3, figsize=(11.0, 2.5 * len(ANIMALS) + 1.6),
                                     squeeze=False, gridspec_kw={"width_ratios": [1, 1, 1.5],
                                                                 "hspace": 0.45})
            im = None
            for ri, an in enumerate(ANIMALS):
                got = tables.get(an)
                if not got:
                    for ci in range(3):
                        axes[ri][ci].axis("off")
                    continue
                pre_C, post_C, per_day = got
                for ci, (C, ttl) in enumerate(((pre_C, "PRE (leave-1-out)"),
                                               (post_C, "POST-stroke sessions"))):
                    ax = axes[ri][ci]
                    im = ax.imshow(C, vmin=0, vmax=max(1, C.max()), cmap="magma")
                    for i in range(len(CONF_LABELS)):
                        for j in range(len(CONF_LABELS)):
                            if C[i, j]:
                                _txt(ax, j, i, f"{int(C[i, j])}", ha="center", va="center",
                                     fontsize=8, color="w" if C[i, j] < C.max() * 0.6 else "k")
                    ax.set_xticks(range(len(CONF_LABELS)))
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                       rotation=90, fontsize=9)
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
                    hit = np.trace(C) / max(1, C.sum())
                    # THE TOTAL DIFFERS BETWEEN THE PANELS -- one count per held-out pre-stroke
                    # session on the left, one per post-stroke day on the right -- so the cell
                    # numbers are not on one scale and only the percentage is comparable. Say so.
                    # A ROW's sum is the number of sessions in which that position was scorable, so
                    # the largest row sum is how many sessions the panel actually rests on.
                    n_unit = int(C.sum(axis=1).max()) if C.size else 0
                    ax.set_title(f"{ttl}\n{hit:.0%} match self  (n={n_unit})", fontsize=9.5,
                                 fontweight="bold" if ci == 0 else "normal")
                    if ci == 0:
                        ax.set_ylabel(f"{an}\nthis position", fontsize=11, fontweight="bold")
                ax = axes[ri][2]
                xs = [d for d in days if d in per_day]
                if xs:
                    ax.plot(xs, [per_day[d][0] for d in xs], "o-", color="#b2182b", ms=5, lw=1.5,
                            label="matches self" if ri == 0 else None)
                    ax2 = ax.twinx()
                    ax2.plot(xs, [per_day[d][1] for d in xs], "s--", color="#2166ac", ms=4, lw=1.2,
                             label="mean rank" if ri == 0 else None)
                    ax2.set_ylim(6.4, 0.6)
                    ax2.set_ylabel("mean rank of true position", fontsize=8.5, color="#2166ac")
                    ax2.tick_params(labelsize=8, colors="#2166ac")
                ax.set_ylim(-0.05, 1.05)
                ax.grid(alpha=0.25, lw=0.5)
                ax.set_ylabel("fraction matching self", fontsize=8.5, color="#b2182b")
                ax.tick_params(labelsize=8)
                if ri == len(ANIMALS) - 1:
                    ax.set_xlabel("days from lesion", fontsize=10)
            if im is None:
                plt.close(fig)
                continue
            cls = _class_note(v)
            _suptitle(fig,
                      f"Which PRE-STROKE position does each post-stroke position match BEST? -- "
                      f"{wname} window\n"
                      f"Post-stroke class: {cls}.  Uses the WHOLE ROW of the similarity matrix, not "
                      f"its diagonal: 0.2 against everything and 0.2 against itself with 0.7 "
                      f"against far_L are the same diagonal and different results.\n"
                      f"ARGMAX AND RANK ARE INVARIANT to a monotone change across a row, so the "
                      f"uniform row shifts that dominate figures 8 and 8d -- amplitude, not "
                      f"resemblance -- cannot move this.\n"
                      f"LEFT is the ceiling: even with no lesion a held-out pre-stroke day does not "
                      f"always match itself best. Read the middle panel against it, never against "
                      f"100%.")
            _footer(fig)
            p = _out(out_dir, f"grant_10_best_match_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made
def fig_best_match_by_session(out_dir, min_trials=10):
    """10b: figure 10 unpooled -- WHICH position each one matched, session by session.

    Priya, 2026-08-26: *"make a version that shows the matching matrix for each session over the
    post-stroke course (like our other first-column pre-stroke, subsequent columns post-stroke
    sessions)"*. Figure 10 pools every post-stroke day into one 6x6 count, which answers "where did
    this position go" but not "when", and a substitution present on one day and absent on the next
    is indistinguishable there from one that held all week.

    ROWS = position, COLUMNS = PRE then each post-stroke day.
    THE TEXT IN A CELL is the position that day's trials matched BEST -- read it as "this row's
    trials looked most like THAT position's pre-stroke pattern".
    THE COLOUR is the RANK of the true position among the six, which the text alone cannot give:
    a cell reading `fL` is a different result when the correct answer ranked second than when it
    ranked sixth. Green = the position still matched itself, red = it ranked last.
    A BOXED CELL is one that still matched itself, so the intact diagonal is visible at a glance
    and survives `--compact`, which drops the text.

    THE PRE COLUMN IS ELEVEN SESSIONS COLLAPSED, not one: colour is the MEAN rank over held-out
    pre-stroke sessions and the text is the modal best match, with the fraction that agreed. It is
    the same quantity as a post column, computed the same way, and it is NOT a perfect score.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            mats, days = _matrices_pattern(align, v, min_trials)
            loo_all = _pre_loo_matrices(align, v, min_trials)
            if not mats or not days:
                continue
            cols = ["PRE"] + list(days)
            fig, axes = plt.subplots(len(ANIMALS), 1, squeeze=False,
                                     figsize=(2.6 + 0.92 * len(cols), 2.15 * len(ANIMALS) + 1.9),
                                     gridspec_kw={"hspace": 0.30})
            drew, im = False, None
            for ri, an in enumerate(ANIMALS):
                ax = axes[ri][0]
                d = mats.get(an)
                if not d:
                    ax.axis("off")
                    continue
                nL = len(CONF_LABELS)
                rank = np.full((nL, len(cols)), np.nan)
                lab = [["" for _ in cols] for _ in range(nL)]
                selfm = np.zeros((nL, len(cols)), bool)

                loo = loo_all.get(an) or []
                if loo:
                    bs = [_best_match(M) for M in loo]
                    for i in range(nL):
                        rs = [r[i] for _b, r in bs if np.isfinite(r[i])]
                        picks = [int(b[i]) for b, _r in bs if b[i] >= 0]
                        if not rs or not picks:
                            continue
                        rank[i, 0] = float(np.mean(rs))
                        modal = max(set(picks), key=picks.count)
                        frac = picks.count(modal) / len(picks)
                        lab[i][0] = f"{_short([CONF_LABELS[modal]])[0]}\n{frac:.0%}"
                        selfm[i, 0] = modal == i
                for cj, day in enumerate(days, start=1):
                    M = d.get(day)
                    if M is None:
                        continue
                    b, r = _best_match(M)
                    for i in range(nL):
                        if b[i] < 0:
                            continue
                        rank[i, cj] = r[i]
                        lab[i][cj] = _short([CONF_LABELS[int(b[i])]])[0]
                        selfm[i, cj] = int(b[i]) == i
                if not np.isfinite(rank).any():
                    ax.axis("off")
                    continue
                drew = True
                im = ax.imshow(np.ma.masked_invalid(rank), vmin=1, vmax=len(CONF_LABELS),
                               cmap="RdYlGn_r", aspect="auto")
                for i in range(nL):
                    for j in range(len(cols)):
                        if lab[i][j]:
                            _txt(ax, j, i, lab[i][j], ha="center", va="center", fontsize=7.5)
                        if selfm[i, j]:
                            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                       edgecolor="k", lw=1.7, zorder=5))
                ax.set_yticks(range(nL))
                ax.set_yticklabels(_short(CONF_LABELS), fontsize=9)
                ax.set_ylabel(f"{an}\nthis position", fontsize=10.5, fontweight="bold")
                ax.set_xticks(range(len(cols)))
                last = ri == len(ANIMALS) - 1
                ax.set_xticklabels((["PRE"] + [f"d{x}" for x in days]) if last else [],
                                   fontsize=9.5)
                if last:
                    ax.set_xlabel("days from lesion", fontsize=10.5)
            if not drew or im is None:
                plt.close(fig)
                continue
            cls = _class_note(v)
            top = _suptitle(fig,
                            f"Which pre-stroke position did each one match BEST, SESSION BY "
                            f"SESSION? -- {wname} window\n"
                            f"Post-stroke class: {cls}.  Figure 10 unpooled: the text is the "
                            f"best-matching pre-stroke position, the COLOUR is the rank of the "
                            f"TRUE one among six.\n"
                            f"A cell reading fL means different things when the correct answer "
                            f"ranked second and when it ranked sixth -- the text alone cannot say "
                            f"which, so the colour carries it. BOXED = still matched itself.\n"
                            f"PRE = eleven held-out pre-stroke sessions collapsed (colour = mean "
                            f"rank, text = modal match and the fraction agreeing). It is NOT a "
                            f"perfect score, and it is the standard the post columns are read "
                            f"against.")
            fig.tight_layout(rect=(0, 0.055, 1, top))
            cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.016, pad=0.02)
            cb.set_label("rank of the TRUE position (1 = still itself)", fontsize=9)
            _footer(fig)
            p = _out(out_dir, f"grant_10b_best_match_by_session_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made
