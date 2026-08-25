"""Figures for deck Section G — POST-STROKE, read from the saved JSON rather than recomputed.

ORDER IS THE ARGUMENT. Behaviour comes first (G1), because on 8/17 both animals stopped attempting
the far positions -- PS94 has ZERO engaged trials at far_center and far_R -- and every decoding
number after it is uninterpretable without that. Putting decoding first is exactly the error that
produced a "PS94 neural deficit" headline which was mostly trial composition.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]


def _num(v, default=0.0):
    """``v`` unless it is None or NaN, in which case ``default``.

    ``v != v`` is the NaN test. Ruff flags the idiom as a self-comparison (PLR0124) because it
    cannot tell it from a mistake, so it lives here once with a noqa rather than four times inline.
    A missing bar must be drawn at zero rather than skipped: skipping it silently shifts every
    later bar in the group left, which reads as a different position.
    """
    return default if v is None or v != v else v      # noqa: PLR0124


#: Post-stroke sessions per counts figure. One row of N panels was fine at N=2 and illegible at
#: N=14: the figure is 5.2 in per session and the slide is 12.5 in, so every added session shrinks
#: every existing one. Chunking keeps the panel width fixed and spends slides instead (Priya,
#: 2026-08-22, on a 12.50 x 1.10 in slide).
COUNTS_PER_FIG = 4


def fig_behaviour(counts, out, name="poststroke_G1_behaviour.png", suptitle=None,
                  per_fig=COUNTS_PER_FIG):
    """G1: per-position engaged / undetected counts, pre vs post. The primary finding.

    Returns a LIST of paths -- one figure per ``per_fig`` sessions. The first keeps ``name`` so an
    existing reference still resolves; the rest get ``_2``, ``_3`` ... The deck globs them.
    """
    ans = sorted(counts)
    chunks = [ans[i:i + per_fig] for i in range(0, len(ans), max(per_fig, 1))] or [[]]
    made = []
    stem = Path(name).stem
    for ci, group in enumerate(chunks):
        made.append(_fig_behaviour_one(
            counts, group, out,
            name if ci == 0 else f"{stem}_{ci + 1}.png",
            suptitle, part=(ci + 1, len(chunks))))
    return made


def _fig_behaviour_one(counts, ans, out, name, suptitle, part):
    fig, axes = plt.subplots(2, len(ans), figsize=(5.2 * len(ans), 6.4), squeeze=False)
    x = np.arange(len(POS))
    w = 0.38
    for k, an in enumerate(ans):
        c = counts[an]
        for r, arm in enumerate(("engaged", "undetected")):
            ax = axes[r][k]
            ax.bar(x - w / 2, [c["pre"][arm].get(p, 0) for p in POS], w,
                   label="pre-stroke (per-session mean)", color="tab:blue",
                   edgecolor="k", linewidth=0.4)
            ax.bar(x + w / 2, [c["post"][arm].get(p, 0) for p in POS], w,
                   label="this session (post-stroke)", color="tab:red", edgecolor="k",
                   linewidth=0.4)
            for xi, p in zip(x + w / 2, POS):
                if c["post"][arm].get(p, 0) == 0:
                    ax.text(xi, 1, "0", ha="center", va="bottom", fontsize=8,
                            color="firebrick", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(POS, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{an} — {arm} trials", fontsize=10)
            ax.set_ylabel("trials")
            if r == 0 and k == 0:
                ax.legend(fontsize=7)
    head = suptitle or ("POST-STROKE BEHAVIOUR FIRST: which positions the animal still attempts. "
                        "Zero engaged trials at a position means no decoding number for it can "
                        "exist.")
    if part[1] > 1:
        head = f"[{part[0]} of {part[1]}] {head}"
    fig.suptitle(head, fontsize=10, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = Path(out) / name
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_matched(matched, out, chance=0.25, name="poststroke_G2_matched.png", suptitle=None):
    """G2: position-matched decoding in each condition against the pre-stroke LOSO band.

    `chance` is a scalar OR a dict keyed like `matched`, and must match the number of positions the
    arm was scored over -- 0.25 when matched to four
    preserved positions (PS94/PS95), 1/6 when the animal still attempts all six (PS92/PS93, which are
    NOT position-restricted because nothing was lost). Hardcoding 0.25 would have drawn a chance line
    50% too high on the small-lesion comparison figure.
    """
    ans = sorted(matched)
    conds = ["post-cue", "post-lick", "pre-cue"]
    _rows, _cols = _grid_shape(len(ans))
    fig, axes = plt.subplots(_rows, _cols, figsize=(4.9 * _cols, 4.4 * _rows), squeeze=False)
    for k, an in enumerate(ans):
        ax = axes[k // _cols][k % _cols]
        m = matched[an]
        for i, cname in enumerate(conds):
            r = m.get(cname)
            if not r:
                continue
            b = r["pre_band"]
            ax.add_patch(plt.Rectangle((i - 0.34, b["min"]), 0.68, b["max"] - b["min"],
                                       color="tab:blue", alpha=0.20, zorder=1))
            ax.plot([i - 0.34, i + 0.34], [b["mean"]] * 2, color="tab:blue", lw=2, zorder=2)
            ax.plot([i], [r["accuracy"]], "o", ms=11, color="tab:red", zorder=3,
                    markeredgecolor="k")
            if r.get("below_every_pre_session"):
                ax.text(i, max(r["accuracy"] - 0.055, 0.02), "below all", ha="center",
                        fontsize=7.5, color="firebrick", fontweight="bold")
        # PER-PANEL chance. On the ALL-trials arm every session is scored over six positions and one
        # line is right for the whole figure; on the LICK-ONLY arm the position set is that session's
        # own preserved set, so PS95 is 4-way on 8/17 and 6-way on 8/18 and a single line would be
        # wrong for at least one panel. `chance` therefore accepts a dict keyed like `matched`.
        ch = chance.get(an, 1 / 6) if isinstance(chance, dict) else chance
        ax.axhline(ch, color="k", ls=":", lw=1)
        ax.text(len(conds) - 0.5, ch + 0.01,
                f"chance ({round(1 / ch)}-way)", fontsize=7, ha="right")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{an} — position-matched", fontsize=10)
        ax.set_ylabel("accuracy")
    fig.suptitle(suptitle or
                 "Frozen PRE-stroke decoder on POST-stroke trials, matched to the positions the "
                 "animal still attempts. Band = pre-stroke leave-one-session-out range under the "
                 "SAME restriction. 4-way: NOT comparable to 6-way numbers elsewhere in this deck.",
                 fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = Path(out) / name
    _blank_unused(axes, len(ans), _rows, _cols)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _blank_unused(axes, n, rows, cols):
    """Hide grid cells the cohort does not fill, so a partial row is empty space not empty boxes."""
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")


def _grid_shape(n, max_cols=4):
    """Rows x cols for n panels. One long row makes every panel tiny once the cohort grows past a
    few sessions (Priya, 2026-08-21); wrapping keeps each one readable."""
    cols = max(1, min(n, max_cols))
    return int(np.ceil(n / cols)), cols


def fig_identity(ident, out):
    """G4: do post-stroke no-lick trials look like pre-stroke licking? WITH the control."""
    ans = sorted(ident)
    # CHUNKED INTO ROWS. One long row put 10 post-stroke sessions in a single axes, and each
    # session's four bars plus its two lines of annotation became unreadable (Priya, 2026-08-21).
    per_row = 4      # 5 made the long 'CONTROL UNDETERMINED' annotations collide
    chunks = [ans[i:i + per_row] for i in range(0, len(ans), per_row)] or [[]]
    fig, axes = plt.subplots(len(chunks), 1, squeeze=False,
                             figsize=(3.9 * per_row + 3.0, 6.1 * len(chunks)))
    w = 0.2
    keys = [("scale_pre_engaged", "pre ENGAGED", "tab:blue"),
            ("scale_pre_undetected", "pre UNDETECTED", "tab:grey"),
            ("CONTROL_post_engaged_frac_engaged_like", "post ENGAGED (control)", "tab:green"),
            ("post_undetected_frac_classified_ENGAGED_like", "post UNDETECTED", "tab:red")]
    for _r, _chunk in enumerate(chunks):
        ax = axes[_r][0]
        ans_c = _chunk
        x = np.arange(len(ans_c))
        for j, (k, lab, col) in enumerate(keys):
            ax.bar(x + (j - 1.5) * w, [ident[a].get(k, np.nan) for a in ans_c], w, label=lab,
                   color=col, edgecolor="k", linewidth=0.4)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        for i, a in enumerate(ans_c):
            d = ident[a]
            gap = d.get("engaged_minus_undetected_post", float("nan"))
            lo, hi = d.get("control_gap_ci95", [float("nan"), float("nan")])
            ok = d.get("boundary_still_discriminates_post")
            # three states: an interval spanning zero is UNRESOLVED, not a failure -- the distinction
            # decides whether the analysis is broken or merely underpowered
            if ok:
                msg, col = "control PASSES — read the red bar", "darkgreen"
            elif np.isfinite(hi) and hi < 0:
                msg, col = "CONTROL INVERTED — the boundary tracks something else", "firebrick"
            else:
                msg, col = "CONTROL UNDETERMINED — too few trials to tell; do NOT read the red bar", "darkorange"
            ax.text(i, 1.02, msg, ha="center", fontsize=8, fontweight="bold", color=col)
            ax.errorbar(i + 0.30, 0.5 + gap / 2, yerr=[[abs(gap - lo) / 2], [abs(hi - gap) / 2]],
                        fmt="none", ecolor="k", elinewidth=1.4, capsize=4, zorder=5)
            # two short lines under the tick label; one long line ran into the neighbouring animal
            ax.text(i, -0.19, f"control gap {gap:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]",
                    ha="center", fontsize=7.5, transform=ax.get_xaxis_transform())
            ax.text(i, -0.25,
                    f"pre-sep {d.get('pre_separability_cv', float('nan')):.2f}"
                    f"{' (grouped)' if d.get('pre_separability_cv_grouped_by_session') else ' (UNGROUPED)'}"
                    f"   n post-no-lick = {d.get('n_post_undetected', 0)}",
                    ha="center", fontsize=7, color="dimgrey",
                    transform=ax.get_xaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels(ans_c)
        ax.set_ylim(0, 1.30)
        ax.set_ylabel("fraction classified ENGAGED-like (licking)")
        # OUTSIDE the axes: at loc="lower center" it covered the pre-UNDETECTED bars, which are the
        # baseline the whole comparison is against.
    axes[0][0].legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              frameon=False)
    fig.suptitle("Do post-stroke NO-LICK trials look like pre-stroke LICKING trials?  "
                 "Boundary trained on PRE-stroke engaged-vs-undetected, position-balanced so it "
                 "cannot simply answer 'far'.  READ THE CONTROL FIRST: post ENGAGED (green) must sit "
                 "ABOVE post UNDETECTED (red), or the boundary is tracking 'post-stroke' rather than "
                 "licking and the answer means nothing. The control gap and its bootstrap 95% CI are printed under each animal: the whole interval must clear zero, and an interval spanning zero means UNDETERMINED, not failed.", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    p = Path(out) / "poststroke_G4_identity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_similarity(sim, out, name="poststroke_G5_similarity.png", suptitle=None):
    """G5: per-position pattern correlation between phases - same code or different code?

    ONE PANEL PER ANIMAL, sessions as the series within it, so each animal's TRAJECTORY across
    post-stroke days is readable. It was a single axes with one bar per session, which at 4 animals
    x 3 days is 10 series x 6 positions = 60 bars competing for one legend: the per-animal
    time course -- the thing the figure is for -- could not be read out of it (Priya, 2026-08-20).

    Sessions are ordered in time within each animal and coloured by their position in that order,
    so the same colour means the same day across panels.
    """
    by_animal = {}
    for lab in sorted(sim):
        by_animal.setdefault(lab.split("_")[0], []).append(lab)
    animals = sorted(by_animal)
    if not animals:
        return None

    # colour by timepoint INDEX, not by date string: animals do not all start on the same day
    # (PS94/PS95 were lesioned 8/16, PS92/PS93 effectively 8/17), so "day 1" is what lines up.
    ncols = max(len(v) for v in by_animal.values())
    colours = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, max(ncols, 2)))

    fig, axes = plt.subplots(1, len(animals), figsize=(3.6 * len(animals) + 1.6, 4.4),
                             squeeze=False, sharey=True)
    x = np.arange(len(POS))
    for k, an in enumerate(animals):
        ax = axes[0][k]
        labs = by_animal[an]
        w = 0.8 / max(len(labs), 1)
        for si, lab in enumerate(labs):
            vals = [sim[lab].get(p, {}).get("r", np.nan) for p in POS]
            ax.bar(x + (si - (len(labs) - 1) / 2) * w, vals, w,
                   label=f"day {si + 1} ({lab.split('_')[1]})", color=colours[si],
                   edgecolor="k", linewidth=0.4)
        ax.axhline(0, color="k", lw=1)
        # An empty column reads as r = 0, i.e. "no similarity", when it means "the animal stopped
        # attempting this position so there is no post-stroke pattern to correlate against".
        for xi, q in zip(x, POS):
            if all(sim[lab].get(q, {}).get("r") is None or q not in sim[lab] for lab in labs):
                ax.text(xi, 0.04, "not attempted", ha="center", va="bottom", fontsize=6.5,
                        rotation=90, color="firebrick", style="italic")
        ax.set_xticks(x)
        ax.set_xticklabels(POS, rotation=30, ha="right", fontsize=7.5)
        ax.set_title(an, fontsize=10, fontweight="bold")
        # THE CEILING, PER POSITION, DRAWN AS A GREY BAND (added 2026-08-25). Without it a reader
        # compares these bars with 1.0, and 1.0 is unreachable: two mean patterns measured on
        # DIFFERENT DAYS differ by ordinary day-to-day drift and by however noisily each was
        # estimated, in a healthy animal with no lesion at all. `r_pre_loo` is exactly that quantity
        # -- each pre-stroke session against the pool of the others -- so it is what a bar would
        # reach if the lesion changed nothing. On the grant figures the equivalent runs 0.75-0.86,
        # so reading a post value of 0.7 against 1.0 rather than against 0.8 turns "unchanged" into
        # "30% lost".
        #
        # ABSENT ON OLD JSON, and that is expected rather than an error: `r_pre_loo` is written by
        # `poststroke_compare.pattern_similarity` from 2026-08-25 onward, so a section_g.json built
        # before then simply has no band and the figure falls back to what it always drew.
        ceil = [np.nanmean([sim[lab].get(p, {}).get("r_pre_loo") for lab in labs
                            if sim[lab].get(p, {}).get("r_pre_loo") is not None] or [np.nan])
                for p in POS]
        if np.isfinite(ceil).any():
            for xi, c in zip(x, ceil):
                if np.isfinite(c):
                    ax.plot([xi - 0.42, xi + 0.42], [c, c], color="0.35", lw=1.8,
                            solid_capstyle="butt",
                            label="pre-stroke ceiling (leave-1-session-out)"
                                  if (k == 0 and xi == x[0]) else None)
        ax.legend(fontsize=6.5, loc="lower left", framealpha=0.85)
        ax.set_ylim(-0.6, 1.0)
    axes[0][0].set_ylabel("r (pre-stroke vs post-stroke mean pattern)")
    fig.suptitle((suptitle or "Per-position correlation between the pre- and post-stroke mean "
                  "activity patterns, one panel per animal, one series per post-stroke day.")
                 + "  GREY LINE = the pre-stroke ceiling (each pre-stroke session against the pool "
                   "of the others). Read the bars against IT, never against 1.0: two pre-stroke "
                   "days already differ by ordinary drift.",
                 fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = Path(out) / name
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_confusion(conf, out):
    """G3: crossed confusion — frozen pre-stroke decoder, pre vs post, full 6x6."""
    ans = sorted(conf)
    fig, axes = plt.subplots(len(ans), 2, figsize=(10.4, 4.8 * len(ans)), squeeze=False)
    for r, an in enumerate(ans):
        for c, phase in enumerate(("pre", "post")):
            ax = axes[r][c]
            d = conf[an][phase]
            M = np.array(d["matrix"], dtype=float)
            im = ax.imshow(np.ma.masked_invalid(M), vmin=0, vmax=1, cmap="magma")
            ax.set_xticks(range(len(POS)))
            ax.set_xticklabels(POS, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(POS)))
            ax.set_yticklabels([f"{p} (n={n})" for p, n in
                                zip(POS, d["n_per_true_position"])], fontsize=7)
            for i in range(len(POS)):
                if d["n_per_true_position"][i] == 0:
                    ax.text(len(POS) / 2 - 0.5, i, "no trials attempted", ha="center",
                            va="center", fontsize=8, color="firebrick", fontweight="bold")
            ttl = "PRE (leave-one-session-out)" if phase == "pre" else "POST (frozen pre-stroke model)"
            ax.set_title(f"{an} — {ttl}", fontsize=9)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Crossed confusion: the frozen PRE-stroke decoder applied to POST-stroke trials. "
                 "The diagonal is per-position recall; the OFF-diagonal is where the errors go, "
                 "which a scalar accuracy discards.", fontsize=10, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = Path(out) / "poststroke_G3_confusion.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p



CONDS = ["post-cue", "post-lick", "pre-cue WITH lick", "pre-cue NO lick"]
MIN_N = 10        # below this a per-position recall is marked, not trusted


def fig_per_position(pp, out, name="poststroke_G2b_per_position.png"):
    """G2b: per-position recall pre vs post in all four decoder conditions (Priya, 2026-08-17).

    One row per condition, one column per animal; the pre-stroke bar then ONE BAR PER POST-STROKE
    DAY at every position. A position the animal stopped attempting gets an explicit "n/a" mark
    rather than a zero bar -- the distinction the scalar summaries kept losing, and the one that
    turned a "PS94 neural deficit" headline into a statement about trial composition.

    It was pre vs a SINGLE post session, because it was built from a day-1-only scratchpad JSON
    covering the two animals that had an effective lesion on 8/17. It is now derived from
    section_g.json, so it covers every animal and every post-stroke day (Priya, 2026-08-20).
    """
    ans = sorted(pp)
    fig, axes = plt.subplots(len(CONDS), len(ans), figsize=(6.4 * len(ans), 3.3 * len(CONDS)),
                             squeeze=False, sharey=True)
    x = np.arange(len(POS))
    ndays = max((len(d.get("posts", [])) for a in pp for c in CONDS for d in [pp[a].get(c) or {}]),
                default=1)
    daycols = plt.get_cmap("autumn")(np.linspace(0.0, 0.62, max(ndays, 2)))
    for r, cond in enumerate(CONDS):
        for k, an in enumerate(ans):
            ax = axes[r][k]
            d = pp[an].get(cond)
            if not d or not d.get("posts"):
                ax.text(0.5, 0.5, cond + ": not computed for this animal",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9,
                        color="firebrick")
                ax.set_xticks([])
                continue
            series = [("pre", d["pre"], "tab:blue", "pre-stroke (LOSO)")]
            for di, (lab, rec) in enumerate(d["posts"]):
                # NO date in the legend: it is drawn once for the whole figure, but day 1 is 8/17
                # for PS94/PS95 and 8/18 for PS92/PS93 (their lesion took a day later). A dated
                # legend would therefore be wrong for half the columns. Each panel title carries
                # that animal's own dates.
                series.append((lab, rec, daycols[di], f"day {di + 1}"))
            w = 0.82 / len(series)
            for j, (_key, rec, col, lbl) in enumerate(series):
                vals = [rec.get(q, {}).get("recall", np.nan) for q in POS]
                ns = [rec.get(q, {}).get("n", 0) for q in POS]
                xs = x + (j - (len(series) - 1) / 2) * w
                ax.bar(xs, [_num(v) for v in vals], w, color=col,
                       edgecolor="k", linewidth=0.4, label=lbl)
                # A recall computed on a handful of trials is not a number a reader should weigh the
                # same as one computed on a hundred, and the extreme values are exactly where n is
                # smallest: PS95 far_R post-stroke reads 1.00 off ONE trial.
                for xi, n, v in zip(xs, ns, vals):
                    if n == 0:
                        ax.text(xi, 0.02, "n/a", ha="center", va="bottom", fontsize=5.5,
                                rotation=90, color="firebrick", fontweight="bold")
                    elif n < MIN_N:
                        ax.bar(xi, _num(v), w, color="none", edgecolor="firebrick",
                               linewidth=1.0, hatch="////", zorder=3)
            ax.axhline(1 / 6, color="k", ls=":", lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels(POS if r == len(CONDS) - 1 else [], rotation=45, ha="right",
                               fontsize=7)
            ax.set_ylim(0, 1.05)
            if k == 0:
                ax.set_ylabel(cond + "\nrecall", fontsize=8)
            bal = "  ".join(f"{lab.split('_')[1]} {d['balanced'].get(lab, float('nan')):.2f}"
                            for lab, _ in d["posts"])
            ax.set_title(f"{an} - {cond}   (balanced: pre {d['pre_balanced']:.2f} -> {bal})",
                         fontsize=8)
            if r == 0 and k == 0:
                ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("Per-position recall in FOUR conditions, every post-stroke day. Training is ALWAYS "
                 "on pre-stroke ENGAGED trials;\n'with/without lick' is the RESPONSE lick, i.e. "
                 "engaged vs undetected trials. Dotted line = 1/6.\n'n/a' = position not attempted "
                 f"(NOT zero recall);   RED HATCHED = fewer than {MIN_N} trials, do not weigh these",
                 fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    q = Path(out) / name
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def fig_nolick_readout(rd, out):
    """G6: was a PLAN formed on post-stroke no-lick trials? IMPAIRED vs PRESERVED positions.

    Replaces the working/disengaged split, retired 2026-08-18: "disengaged" has no valid post-stroke
    construction, so that figure compared against a class that was never established. This contrast
    splits on the true spout position, which is measured rather than inferred, and needs no
    engagement label anywhere in it.
    """
    ans = sorted(rd)
    _rows, _cols = _grid_shape(len(ans))
    fig, axes = plt.subplots(_rows, _cols, figsize=(5.6 * _cols + 1.0, 5.2 * _rows),
                             squeeze=False)
    for k, an in enumerate(ans):
        ax = axes[k // _cols][k % _cols]
        xt, lab, i = [], [], 0.0
        for al in ("precue", "cue"):
            r = rd[an].get(al, {})
            for arm, col in (("preserved", "tab:green"), ("impaired", "tab:red")):
                a = r.get(arm, {})
                if "balanced_accuracy" in a:
                    ax.bar(i, a["balanced_accuracy"], 0.66, color=col, edgecolor="k", linewidth=0.5)
                    ax.plot([i - 0.33, i + 0.33], [a["bal_null_mean"]] * 2, color="k", lw=2.2, zorder=3)
                    # Star on the CORRECTED p, matching the verdict under the panel. Starring
                    # the raw p put a "*" on PS95 cue/impaired while the caption on the same
                    # panel said the effect was not supported -- the reader believes the star.
                    praw = a["bal_p"]
                    pc_ = a.get("bal_p_bonferroni_x8", min(1.0, praw * 8))
                    star = ("***" if pc_ < 0.001 else "**" if pc_ < 0.01 else
                            "*" if pc_ < 0.05 else "n.s.")
                    ax.text(i, a["balanced_accuracy"] + 0.02,
                            f"{star}\nn={a['n']}\np={praw:.3f} raw",
                            ha="center", fontsize=6.5)
                else:
                    ax.text(i, 0.04, "n=" + str(a.get("n", 0)) + "\ntoo few", ha="center",
                            fontsize=7, color="dimgrey")
                xt.append(i)
                lab.append(al + "\n" + arm)
                i += 1
            i += 0.4
        ax.set_xticks(xt)
        ax.set_xticklabels(lab, fontsize=7.5)
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("balanced accuracy (6-way)")
        ax.set_title(an, fontsize=11, fontweight="bold")
        # The verdict wrapped UNDER the panel. Untreated, two of these sentences overlapped into an
        # unreadable overlay -- and the verdict is the figure's conclusion, so losing it loses the
        # slide. Only the pre-cue verdict is shown: post-cue on a no-lick trial is ambiguous because
        # the spout is physically present (see impaired_nolick_readout).
        v = rd[an].get("precue", {}).get("verdict", "")
        v = v.split(". CAVEAT")[0]
        ax.set_xlabel("PRE-CUE verdict: " + textwrap.fill(v, 62), fontsize=7, labelpad=8)
    fig.suptitle("Post-stroke NO-LICK trials: was the position represented anyway?\n"
                 "Frozen pre-stroke decoder; BLACK RULE = that arm's own permutation null. Above the "
                 "null at IMPAIRED positions = plan formed, movement failed. CAVEAT: 'no lick "
                 "detected' is not 'no tongue protrusion' -- DLC replaces this inference with a "
                 "measurement. Stars are Bonferroni-corrected for the 8 tests in a pass; the raw p "
                 "is printed under each bar.", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    q = Path(out) / "poststroke_G6_nolick_readout.png"
    _blank_unused(axes, len(ans), _rows, _cols)
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def fig_confusion_alltrials(conf, out, align="cue", name=None, arm_name="ALL trials"):
    """G3b: crossed confusion for one POST arm -- by default ALL trials, engaged and no-lick alike.

    ``arm_name`` LABELS THE POST PANELS and must match the arm whose matrices were passed in. It
    exists because it was missing: this function is called once per arm (`section_g_figures`
    passes the arm into the FILENAME only), while the post-panel titles and the suptitle said
    "ALL trials" unconditionally -- so every LICK-ONLY figure was captioned as the all-trials one
    (Priya, 2026-08-24: "the matrices are labeled the same"). The DATA was always right, indexed
    by `arms[arm]["confusion"]`; only the caption lied, which is the harder failure to notice
    because nothing about the numbers looks wrong.

    Priya, 2026-08-18: "even though far R lick was never successful, pre-cue looked like far R, or
    far center looks like far R (because tongue is deviated leftward, animal has to try harder to get
    tongue to the right)."

    The engaged-only matrix cannot answer that: PS94 has ZERO engaged trials at far_center and far_R,
    so those rows were blank -- the two positions the lesion abolished, and the two worth reading. The
    animal was still CUED to them (104 and 105 no-lick trials), and those trials are the only evidence
    that exists there.

    THE COLUMN BASELINE IS ON THE FIGURE FOR A REASON. Reading a diagonal against 1/6 is wrong here:
    the frozen decoder predicts far_R on ~35% of ALL PS94 post-stroke trials, so far_R "recall" is
    inflated by that bias before any position information is involved. Under a label permutation the
    expected recall for a position is exactly its PREDICTION rate, which is what the grey bar under
    each column shows. The finding is the OFF-diagonal pull, and the diagonal only counts where it
    exceeds that column's own baseline.
    """
    ans = sorted(conf)
    # three panels per animal: PRE (row), POST (row = recall), POST (column = precision). The PRE
    # panel needs only one normalisation -- pre-stroke trials are balanced across positions by design,
    # so row and column are nearly the same picture there. They diverge post-stroke, which is why that
    # is where both are shown.
    # FOUR panels since 2026-08-19. The PRE-NO-LICK one is the matched control: comparing
    # post-stroke no-lick trials against a pre-stroke ENGAGED panel confounds the lesion with the
    # absence of a movement, because the post rows have no lick and the pre rows do.
    PANELS = (("pre", "row"), ("pre_nolick", "row"), ("post", "row"), ("post", "col"))
    # ONE FIGURE PER SESSION. Stacking every session as a row made this 25 inches tall for a
    # 7.5 inch slide: placed at full width it ran off the bottom and the axis labels were never
    # visible, and scaled to fit it was illegible. Four panels across is close to slide aspect.
    made = []
    for an in ans:
        fig, axes = plt.subplots(1, 4, figsize=(19.0, 5.4), squeeze=False)
        r = 0
        if align not in conf[an]:
            continue
        for c, (phase, norm) in enumerate(PANELS):
            ax = axes[r][c]
            if phase not in conf[an][align]:      # older JSONs predate the matched control
                ax.axis("off")
                ax.text(0.5, 0.5, f"no {phase} panel in this record", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
                continue
            d = conf[an][align][phase]
            Mrow = np.array(d["matrix"], float)          # stored row-normalised
            n = np.array(d["n_per_true_position"], float)
            nl = d.get("n_nolick_per_true_position", [0] * len(n))

            # counts matrix, from which either normalisation follows exactly
            C = Mrow * n[:, None]
            colsum = np.nansum(C, axis=0)
            tot = n.sum() or 1
            pred_rate = colsum / tot                     # = expected recall under a label permutation
            with np.errstate(invalid="ignore", divide="ignore"):
                Mcol = C / colsum[None, :]               # P(true i | predicted j)
                precision = np.array([Mcol[j, j] if colsum[j] else np.nan
                                      for j in range(len(n))])

            M = Mrow if norm == "row" else Mcol
            im = ax.imshow(np.ma.masked_invalid(M), vmin=0, vmax=1, cmap="magma")
            ax.set_xticks(range(len(POS)))
            ax.set_xticklabels([f"{p}\n(pred {pr:.2f})\n(prec {pc:.2f})"
                                for p, pr, pc in zip(POS, pred_rate, precision)],
                               rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(len(POS)))
            ax.set_yticklabels([f"{p} (n={int(a)}, {100*b/a:.0f}% no-lick)" if a else f"{p} (n=0)"
                                for p, a, b in zip(POS, n, nl)], fontsize=6.5)
            for i in range(len(POS)):
                if not n[i]:
                    ax.text(len(POS) / 2 - 0.5, i, "no trials", ha="center", va="center",
                            fontsize=8, color="firebrick", fontweight="bold")
                    continue
                # ring the diagonal only where RECALL beats its own column's prediction rate -- the
                # same criterion on both panels, so the two normalisations stay comparable
                if Mrow[i, i] > pred_rate[i]:
                    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="lime", lw=2.0))
            ttl = ("PRE (engaged, LOSO) — row / recall" if phase == "pre" else
                   "PRE (NO-LICK, frozen) — row / recall  [matched control]"
                   if phase == "pre_nolick" else
                   f"POST (frozen, {arm_name}) — row / RECALL" if norm == "row" else
                   f"POST (frozen, {arm_name}) — column / PRECISION")
            ax.set_title(f"{an} — {ttl}", fontsize=8.5)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            fig.colorbar(im, ax=ax, fraction=0.046)
        _armblurb = ("ALL trials (engaged + no detected lick). Rows the lesion abolished are "
                     "filled by no-lick trials, the only trials that exist there."
                     if arm_name == "ALL trials" else
                     f"{arm_name} — ONLY trials with a detected lick. A position the animal "
                     "abandoned has NO row here at all, which is exactly the gap the ALL-trials "
                     "arm exists to fill; do not read an absent row as a failure to decode.")
        fig.suptitle(
            f"{an} — crossed confusion, {align}-aligned, POST arm = {_armblurb} "
            "PANEL 2 IS THE MATCHED CONTROL: "
            "pre-stroke NO-LICK trials scored by a decoder trained on the OTHER pre-stroke "
            "sessions' engaged trials, so it differs from the post panel in PHASE alone rather "
            "than in phase and the absence of a movement together. BOTH NORMALISATIONS for the "
            "post panel: ROW = P(pred|true) = RECALL, the question the hypothesis asks; COLUMN = "
            "P(true|pred) = PRECISION, which exposes what row normalisation structurally hides "
            "— a decoder predicting one position everywhere earns high recall there without "
            "carrying information. '(pred)' under each column is how often that position is "
            "predicted at all, which IS the recall expected under a label permutation; '(prec)' "
            "is precision. GREEN BOX = recall beats its own column's prediction rate. Read the "
            "OFF-diagonal: a systematic pull toward one position is the result, not the "
            "diagonal.", fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        stem = (name or f"poststroke_G3b_confusion_alltrials_{align}.png")[:-4]
        q = Path(out) / f"{stem}_{an}.png"
        fig.savefig(q, dpi=150)
        plt.close(fig)
        made.append(q)
    return made


def fig_fits_engaged(fits, out, align="precue", name=None):
    """Does the post-stroke NO-LICK session sit inside the PRE-stroke ENGAGED distribution?

    Priya, 2026-08-18: "my argument/hypothesis is that on the 'failed' trials, the animal was indeed
    trying to lick but couldn't, so the pre-cue activity post-stroke looks like successful pre-cue
    activity from pre-stroke trials... maybe a confidence interval that can tell us if the session's
    value does or does not confidently fit in the 'engaged' pre-cue distribution?"

    This replaces the G4 control for that question. G4 asked whether post-stroke ENGAGED and NO-LICK
    trials still separate, and treated failure to separate as a broken boundary -- but the
    execution-failure hypothesis predicts they should NOT separate, so that control could disqualify
    the very result it was meant to license. Here the references come from PRE-stroke sessions, where
    the answer is known, and the post-stroke session is simply placed against them.

    Each dot is a SESSION, not a trial: the spread of those dots is the interval that matters, because
    sessions differ from one another far more than trials within a session do.
    """
    ans = sorted(fits)
    ans = [a for a in ans if align in fits[a] and "post_value" in fits[a].get(align, {})]
    if not ans:
        return None
    _rows, _cols = _grid_shape(len(ans))
    fig, axes = plt.subplots(_rows, _cols, figsize=(4.6 * _cols + 1.6, 4.9 * _rows), squeeze=False,
                            sharey=True)
    for k, a in enumerate(ans):
        ax = axes[k // _cols][k % _cols]
        d = fits[a][align]
        for j, (arm, col, lab) in enumerate(((("engaged"), "tab:blue", "pre-stroke ENGAGED"),
                                             (("nolick"), "tab:grey", "pre-stroke NO-LICK"))):
            ref = d[f"reference_{arm}_per_session"]
            v = np.array(list(ref.values()), float)
            # band confined to its own group: at xmin/xmax 0.04-0.46 it ran under the neighbouring
            # column and read as though the reference extended across the whole panel
            ax.axhspan(v.min(), v.max(), xmin=0.03 + j * 0.31, xmax=0.30 + j * 0.31,
                       color=col, alpha=0.18)
            ax.plot(np.full(len(v), j) + np.linspace(-0.12, 0.12, len(v)), v, "o", ms=5,
                    color=col, markeredgecolor="k", lw=0.3, label=lab)
            ax.plot([j - 0.22, j + 0.22], [v.mean()] * 2, color=col, lw=2.2)
        P, ci = d["post_value"], d["post_value_ci95"]
        ax.errorbar(2, P, yerr=[[P - ci[0]], [ci[1] - P]], fmt="o", ms=11, color="tab:red",
                    markeredgecolor="k", ecolor="k", elinewidth=1.6, capsize=5, zorder=5,
                    label="POST-stroke NO-LICK")
        if np.isfinite(d.get("post_engaged_value", np.nan)):
            ax.plot(2.35, d["post_engaged_value"], "s", ms=8, color="tab:green",
                    markeredgecolor="k", label="POST-stroke engaged")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["pre\nENGAGED", "pre\nNO-LICK", "POST\nNO-LICK"], fontsize=8)
        ax.set_xlim(-0.5, 2.7)
        ie = d["engaged"]["post_inside_range"]
        inl = d["nolick"]["post_inside_range"]
        # take the tag from the VERDICT the analysis computed, not a second copy of the logic here --
        # they had already diverged once, with "outside BOTH" on a panel the analysis called
        # INTERMEDIATE (PS94 pre-cue, 40% of the way toward engaged)
        frac = d.get("fraction_of_gap_toward_engaged")
        tag = ("INSIDE engaged, OUTSIDE no-lick" if (ie and not inl) else
               "INSIDE no-lick, OUTSIDE engaged" if (inl and not ie) else
               "inside BOTH — references overlap" if (ie and inl) else
               f"INTERMEDIATE — {frac:.0%} of the way toward engaged"
               if _num(frac, None) is not None else "outside BOTH references")
        col = ("darkgreen" if (ie and not inl) else "firebrick" if (inl and not ie) else
               "darkorange")
        ax.set_title(f"{a}\n{tag}", fontsize=9, color=col, fontweight="bold")
        ax.set_xlabel(f"z vs engaged {d['engaged']['z_of_post']:+.2f}   "
                      f"vs no-lick {d['nolick']['z_of_post']:+.2f}", fontsize=7.5)
        if k == 0:
            ax.set_ylabel("fraction classified ENGAGED-like (licking)")
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"Do post-stroke NO-LICK trials fall inside the PRE-stroke ENGAGED distribution? "
        f"({align}-aligned)\nEach dot is a SESSION, held out from the discriminator that scored it, "
        "so the spread IS the confidence interval — and it is the right one, because sessions differ "
        "from each other far more than trials within a session. Unlike the G4 control this makes NO "
        "assumption that the two post-stroke classes should differ, which is what the "
        "execution-failure hypothesis denies.", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    q = Path(out) / (name or f"poststroke_G4b_fits_engaged_{align}.png")
    _blank_unused(axes, len(ans), _rows, _cols)
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def fig_grid(rec, out, arm="all", name=None):
    """G2c: WITHIN-SESSION decoding at each alignment, against that animal's own pre-stroke range.

    The deck's headline figure, and it lived only in a scratchpad script until 2026-08-19 -- so the
    one slide carrying the four-animal result could not be regenerated from the repo.

    LAYOUT IS THE ARGUMENT. One column per animal; within it, sessions run left to right in time, so
    PS92/PS93 show their INEFFECTIVE-lesion session (8/17, neither phase) beside their effective
    day 1 (8/18). That pairing is a within-animal before/after control -- same rig, one day apart --
    and it exists only because the excluded sessions were kept analysable rather than discarded.

    DIRECTION IS COLOURED SEPARATELY, because it has to be. Colouring on `inside_pre_range` alone
    paints a value ABOVE the pre-stroke band the same as a collapse, and PS95's post-lick sits at
    z=+1.7/+2.4 -- better than pre-stroke. That mistake was made twice before this figure existed.
    """
    ALIGN = [("pre-cue", "PRE-cue\n(plan)"), ("post-cue", "POST-cue\n(execution)"),
             ("post-lick", "POST-lick")]
    sessions = sorted(rec)
    animals = sorted({rec[s]["animal"] for s in sessions})
    if not animals:
        return None
    fig, axes = plt.subplots(1, len(animals), figsize=(4.3 * len(animals) + 1.0, 5.8),
                             squeeze=False, sharey=True)
    chance = None
    for k, an in enumerate(animals):
        ax = axes[0][k]
        labs = [s for s in sessions if rec[s]["animal"] == an]
        width = 0.8 / max(len(labs), 1)
        bands = {}
        for si, lab in enumerate(labs):
            excluded = rec[lab].get("phase_tag") == "excluded"
            arms = rec[lab].get("arms", {}).get(arm, {})
            for i, (cond, _nice) in enumerate(ALIGN):
                r = arms.get(f"{cond} within-session")
                if not r or not r.get("post"):
                    continue
                b = r["within_pre_band"]
                v = next(iter(r["post"].values()))
                chance = r.get("chance", chance)
                x = i + (si - (len(labs) - 1) / 2) * width
                # ONE band per alignment. Each session carries its own pre-stroke band (computed
                # over ITS preserved positions), so drawing per session stacked two translucent
                # rectangles that read as two distributions. Where the sessions' bands differ the
                # union is drawn and the panel says so, rather than implying a single reference.
                bands.setdefault(i, []).append(b)
                if excluded:
                    col = "grey"
                elif v["inside_pre_range"]:
                    col = "tab:green"
                elif v["z"] > 0:
                    col = "tab:purple"                    # outside, but BETTER than pre-stroke
                else:
                    col = "tab:red"
                ax.plot(x, v["within_accuracy"], "s" if excluded else "o", ms=9, color=col,
                        markeredgecolor="k", zorder=4)
                ax.text(x, v["within_accuracy"] - 0.055, f"{v['z']:+.1f}", ha="center", fontsize=6.5,
                        color=("dimgrey" if (excluded or v["inside_pre_range"])
                               else "purple" if v["z"] > 0 else "firebrick"))
                ax.text(x, 0.03, lab.split("_")[-1], ha="center", fontsize=5.5, rotation=90,
                        color=("grey" if excluded else "k"))
        for i, bs in bands.items():
            lo = min(b["min"] for b in bs)
            hi = max(b["max"] for b in bs)
            ax.add_patch(plt.Rectangle((i - 0.42, lo), 0.84, max(hi - lo, 1e-9),
                                       color="tab:blue", alpha=0.13, zorder=1))
            for b in bs:                      # one mean line per session's own band
                ax.plot([i - 0.42, i + 0.42], [b["mean"]] * 2, color="tab:blue", lw=1.4, zorder=2)
        if chance:
            ax.axhline(chance, color="k", ls=":", lw=1)
        ax.set_xticks(range(len(ALIGN)))
        ax.set_xticklabels([n for _c, n in ALIGN], fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_title(an, fontsize=11, fontweight="bold")
        if k == 0:
            ax.set_ylabel("within-session decoding accuracy")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markeredgecolor="k", label=t)
               for c, t in (("tab:green", "inside pre-stroke range"),
                            ("tab:red", "outside pre-stroke range"),
                            ("tab:purple", "outside, but ABOVE pre-stroke"))]
    handles.append(plt.Line2D([], [], marker="s", ls="", color="grey", markeredgecolor="k",
                              label="lesion did not take (neither phase)"))
    axes[0][0].legend(handles=handles, fontsize=6.5, loc="lower left")
    arm_name = "ALL trials, six positions" if arm == "all" else "LICK-ONLY, per-session positions"
    fig.suptitle(
        "DAY 1 AFTER AN EFFECTIVE LESION: the PLAN survives, EXECUTION does not.\n"
        f"{arm_name}; dotted line = chance. BAND = that animal's pre-stroke range for the same "
        "measure. Pre-cue and post-cue are two windows on the SAME TRIALS, so LED power, baseline F, "
        "amplitude, arousal, engagement and trial count act on both equally and cannot produce a "
        "difference between them.\n"
        "GREY SQUARES are sessions after a laser that did NOT take -- nothing outside the band. The "
        "same animals one day later, after the effective lesion, show the dissociation: a "
        "within-animal before/after control.", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    p = Path(out) / (name or f"poststroke_grid_{arm}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


