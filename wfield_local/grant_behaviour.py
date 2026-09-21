"""GRANT FIGURES — BEHAVIOUR

Licking accuracy per spout position, and the pre-stroke decoding capability claim.

Split out of `grant_figures` on 2026-09-21, one module per figure family. THE GROUPING IS FROM THE
CALL GRAPH, not from the names: every function here is reached from this family's entry points and
from no other family's. Anything shared with a sibling lives in `grant_kit` -- which is why this
imports from there and never from `grant_figures`, a direction that would be a cycle.

Entry points, registered in `grant_figures.JOBS`:
  - `fig_behaviour`
  - `fig_behaviour_collapsed`
  - `fig_prestroke_decoding`
  - `fig_prestroke_decoding_cohort`
"""
from __future__ import annotations

import json
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

#: Sessions earlier than this many days before the lesion are collapsed into ONE baseline point.
#: The June block sits at day -70 and the whole post-stroke story inside +/-8, so plotting the true
#: axis spends 85% of the width on empty space and squeezes the result into a sliver.
BASELINE_BEFORE = -20
BASELINE_X = -16
def _wilson(hits, n, z=1.96):
    """Wilson 95% interval -- the same construction the per-session CSV uses, for pooled counts."""
    if not n:
        return (float("nan"),) * 3
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)
def fig_behaviour(out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.6), sharey=True, squeeze=False)
    for k, an in enumerate(ANIMALS):
        ax = axes[0][k]
        for pos in POS:
            col, mk, ls = pos_style()[pos]
            pre_x, pre_y, pre_e = [], [], [[], []]
            post_x, post_y, post_e = [], [], [[], []]
            base_h = base_n = 0
            for mmdd, day in _sessions(an):
                m = _position_metrics(an, mmdd).get(pos)
                if not m or m[3] < 5:
                    continue
                if day < BASELINE_BEFORE:
                    base_h += round(m[0] * m[3]); base_n += m[3]
                    continue
                # CLAMP AT ZERO. A position at ceiling has hit_rate == ci_hi == 1.0 and the
                # subtraction lands on -1e-16, which matplotlib rejects outright rather than
                # rounding -- 51 cells across the cohort are at ceiling.
                tgt = (pre_x, pre_y, pre_e) if day < 0 else (post_x, post_y, post_e)
                tgt[0].append(day); tgt[1].append(m[0])
                tgt[2][0].append(max(0.0, m[0] - m[1])); tgt[2][1].append(max(0.0, m[2] - m[0]))
            if base_n:
                p, lo, hi = _wilson(base_h, base_n)
                ax.errorbar([BASELINE_X], [p], yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]],
                            color=col, marker=mk, ms=5.5, capsize=2, elinewidth=0.7, lw=0)
            # PRE AND POST ARE DRAWN AS SEPARATE SEGMENTS. Joining day -2 to day +1 draws a line
            # through the lesion and reads as a continuous decline that was measured; nothing was
            # recorded in between.
            for xs, ys, es in ((pre_x, pre_y, pre_e), (post_x, post_y, post_e)):
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ls=ls, ms=4.5, lw=1.4,
                                capsize=2, elinewidth=0.7,
                                label=pos if (xs is pre_x and k == 0) else None)
        ax.axvline(0, color="k", lw=1.6, ls=":")
        ax.text(0, 1.035, "lesion", ha="center", fontsize=8, color="k",
                transform=ax.get_xaxis_transform())
        ax.axvline((BASELINE_X + BASELINE_BEFORE) / 2 + 2, color="0.6", lw=1.0, ls=(0, (2, 3)))
        ax.text(BASELINE_X, -0.075, "June\nbaseline", ha="center", va="top", fontsize=8,
                color="0.35", transform=ax.get_xaxis_transform())
        ax.set_title(an, fontsize=12, fontweight="bold")
        ax.set_xlabel("days from lesion")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(BASELINE_X - 2.5, None)          # else the collapsed baseline sits on the spine
        if k == 0:
            ax.set_ylabel("hit rate (engaged trials)")
        ax.grid(alpha=0.25, lw=0.5)
    # Below the panels, for the same reason as 1b: lower-left is where the impaired traces live.
    _h, _lab = axes[0][0].get_legend_handles_labels()
    if _h:
        fig.legend(_h, _lab, loc="lower center", ncol=len(POS), fontsize=11.5, frameon=False)
    _suptitle(fig, "Licking accuracy at each spout position, relative to the lesion. "
                 "Engaged trials only (the terminal quit period is excluded); bars are Wilson 95% "
                 "CIs. The two sessions after the laser that did not take are omitted.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 1.0))          # reserve the band the legend sits in
    p = Path(out_dir) / "grant_1_behaviour_by_position.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p
def fig_behaviour_collapsed(out_dir, jitter=0.11):
    """1b: EVERY pre-stroke day as one mean +/- SEM per position, then the post-stroke days.

    Priya, 2026-08-24. The dated version (panel 1) shows that baseline is flat and stays flat, which
    is worth showing once; after that the pre-stroke days are 11 points of the same number and the
    eye has to do the averaging. Here the whole baseline is ONE point per position and the deficit
    is read directly against it.

    SEM IS ACROSS SESSIONS, not across trials. The claim being made is "this position's hit rate on
    a typical day", so the day is the unit -- a trial-level interval would be several times tighter
    and would understate how much a position varies from session to session.

    Positions are offset horizontally because at baseline five of the six sit on top of each other
    near 1.0 and are otherwise unreadable.
    """
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.7), sharey=True, squeeze=False)
    for k, an in enumerate(ANIMALS):
        ax = axes[0][k]
        imp = _impaired(an)
        post_days = sorted({d for _m, d in _sessions(an, phases=("post",))})
        for pi, pos in enumerate(POS):
            col, mk, _ls = pos_style()[pos]
            off = (pi - (len(POS) - 1) / 2) * jitter
            pre_vals = [m[0] for mmdd, d in _sessions(an, phases=("pre",))
                        if (m := _position_metrics(an, mmdd).get(pos)) and m[3] >= 5]
            if pre_vals:
                ax.errorbar([-1 + off], [np.mean(pre_vals)],
                            yerr=[np.std(pre_vals, ddof=1) / np.sqrt(len(pre_vals))]
                            if len(pre_vals) > 1 else None,
                            color=col, marker=mk, ms=7, capsize=3, lw=0, elinewidth=1.2,
                            markeredgecolor="k", markeredgewidth=0.5,
                            # NO PER-POSITION ASTERISK IN THE LEGEND: the legend is drawn from
                            # the first animal only, so a mark meaning "impaired" there would be
                            # read as applying to all four. The per-panel title carries it.
                            label=pos if k == 0 else None)
            xs, ys = [], []
            for mmdd, d in _sessions(an, phases=("post",)):
                m = _position_metrics(an, mmdd).get(pos)
                if not m or m[3] < 5:
                    continue
                xs.append(d + off); ys.append(m[0])
            if xs:
                ax.errorbar(xs, ys, color=col, marker=mk, ms=4.5, lw=1.3, alpha=0.95)
        # ONE dotted line, at the TRUE ZERO of the axis (Priya, 2026-08-25). There were two -- a
        # grey one at -0.5 marking the break between the collapsed baseline point and the post days,
        # and a black one at an arbitrary +0.35 for the lesion -- which read as a pair of unexplained
        # rules with a gap between them. The x axis is already "days from lesion", so x = 0 IS the
        # lesion and needs no second marker; the "ALL pre" tick says where the collapsed point sits.
        # NO "lesion" TEXT: it collided with the per-panel title, which carries the impaired
        # positions and is the more useful text.
        ax.axvline(0, color="k", lw=1.4, ls=":")
        ax.set_xticks([-1] + post_days)
        ax.set_xticklabels(["ALL\npre"] + [str(d) for d in post_days], fontsize=11)
        ax.set_title(f"{an}\nimpaired: {', '.join(sorted(imp)) or 'none'}", fontsize=10,
                     fontweight="bold")
        ax.set_xlabel("days from lesion")
        ax.set_ylim(-0.02, 1.05)
        if k == 0:
            ax.set_ylabel("hit rate (engaged trials)")
        ax.grid(alpha=0.25, lw=0.5)
    # LEGEND BELOW ALL PANELS, not inside one (Priya, 2026-08-25). At lower-left of PS92 it sat
    # exactly on that animal's far_R trace, which runs 0.0-0.5 over the early post-stroke days --
    # i.e. it covered the deficit the figure exists to show. A figure-level legend cannot collide
    # with data whatever the values turn out to be, which an in-panel "best" location cannot promise.
    _h, _lab = axes[0][0].get_legend_handles_labels()
    if _h:
        fig.legend(_h, _lab, loc="lower center", ncol=len(POS), fontsize=11.5, frameon=False)
    _suptitle(fig, "Licking accuracy per spout position: the WHOLE pre-stroke baseline as one point "
                 "(mean +/- SEM across sessions), then each day after the lesion.\n"
                 "Engaged trials only; positions offset horizontally so all six are visible. "
                 "Each panel title lists the positions that dropped below 50% on any post-stroke day.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 1.0))          # reserve the band the legend sits in
    p = Path(out_dir) / "grant_1b_behaviour_pre_collapsed.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p
# ------------------------------------------------------------------ 2. pre-stroke decoding
def fig_prestroke_decoding(out_dir):
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    width = 0.26
    x = np.arange(len(ANIMALS))
    any_drawn = False
    for wi, (_disp, align, wname) in enumerate(WINDOWS):
        f = _fig_root() / f"joint_xsession_decoder_{align}.json"
        if not f.exists():
            print(f"  [grant] MISSING {f.name} -- run `joint_xsession --align {align}`", flush=True)
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        xs = x + (wi - (len(WINDOWS) - 1) / 2) * width
        for k, an in enumerate(ANIMALS):
            r = d.get(an)
            if not r:
                continue
            pre = {lab for lab in config.phase_labels("pre") if lab.startswith(an)}
            vals = [v for lab, v in (r.get("per_session") or {}).items() if lab in pre]
            if not vals:
                continue
            m = float(np.mean(vals))
            # CI ACROSS HELD-OUT SESSIONS. The claim is that the code generalises to a session the
            # decoder never saw, so the session is the unit -- a trial-level CI would be far tighter
            # and would be answering a different question.
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ax.bar(xs[k], m, width * 0.92, color=f"C{wi}", edgecolor="k", linewidth=0.5,
                   label=wname if k == 0 else None, zorder=2)
            ax.errorbar(xs[k], m, yerr=1.96 * sem, color="k", capsize=3, lw=1.1, zorder=3)
            ax.plot(np.full(len(vals), xs[k]) + np.linspace(-0.05, 0.05, len(vals)), vals,
                    "o", ms=2.6, color="k", alpha=0.45, zorder=4)
            any_drawn = True
    if not any_drawn:
        plt.close(fig)
        return None
    ax.axhline(1 / 6, color="k", ls=":", lw=1.2)
    ax.text(len(ANIMALS) - 0.4, 1 / 6 + 0.012, "chance (1/6)", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(ANIMALS, fontsize=11)
    ax.set_ylabel("cross-session decoding accuracy")
    ax.set_ylim(0, 1.0)
    # LEGEND BELOW THE AXES. Bars run from 0 to ~0.95 in every animal, so there is no interior
    # corner it can sit in without covering data -- in-axes it landed on PS92's post-cue bar.
    ax.legend(fontsize=11.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_title("Spout position decodes across sessions from a shared LocaNMF basis (pre-stroke)\n"
                 "Leave-one-session-out: the decoder never saw the session it is scored on.\n"
                 "Bar = mean over held-out sessions, error bar = 95% CI across sessions, "
                 "dots = individual sessions.", fontsize=9.5)
    fig.tight_layout()
    p = Path(out_dir) / "grant_2_prestroke_crossday_decoding.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p
def fig_prestroke_decoding_cohort(out_dir):
    """2b: the cohort version of panel 2 -- one bar per window, all four animals.

    THE ANIMAL IS THE UNIT. Bar = mean of the four per-animal leave-one-session-out accuracies,
    error bar = SEM across ANIMALS (n=4), and each animal is drawn as a labelled point. Pooling all
    ~44 held-out sessions into one bar instead would give a far tighter interval that describes how
    much a SESSION varies, not how much an ANIMAL does -- and a cohort claim is a claim about
    animals. With n=4 the SEM is wide, which is honest: it is what four mice support.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    marks = ("o", "s", "^", "D")
    for wi, (_disp, align, wname) in enumerate(WINDOWS):
        f = _fig_root() / f"joint_xsession_decoder_{align}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        per_animal = []
        for an in ANIMALS:
            r = d.get(an)
            if not r:
                continue
            pre = {lab for lab in config.phase_labels("pre") if lab.startswith(an)}
            vals = [v for lab, v in (r.get("per_session") or {}).items() if lab in pre]
            if vals:
                per_animal.append((an, float(np.mean(vals))))
        if not per_animal:
            continue
        ys = [v for _a, v in per_animal]
        m = float(np.mean(ys))
        sem = float(np.std(ys, ddof=1) / np.sqrt(len(ys))) if len(ys) > 1 else 0.0
        ax.bar(wi, m, 0.62, color=f"C{wi}", edgecolor="k", linewidth=0.6, zorder=2, alpha=0.9)
        ax.errorbar(wi, m, yerr=sem, color="k", capsize=4, lw=1.3, zorder=3)
        for j, (an, v) in enumerate(per_animal):
            ax.plot(wi + (j - (len(per_animal) - 1) / 2) * 0.12, v, marks[j % len(marks)],
                    ms=6, color="k", mfc="white", zorder=4,
                    label=an if wi == 0 else None)
        ax.text(wi, m + sem + 0.03, f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1 / 6, color="k", ls=":", lw=1.2)
    ax.text(len(WINDOWS) - 0.55, 1 / 6 + 0.015, "chance (1/6)", fontsize=8, ha="right")
    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels([w[2] for w in WINDOWS], fontsize=11)
    ax.set_ylabel("cross-session decoding accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_title("Spout position decodes across sessions, all four animals (pre-stroke)\n"
                 "Leave-one-session-out in a shared LocaNMF basis. Bar = mean across animals, "
                 "error bar = SEM across animals (n=4),\npoints = individual animals.",
                 fontsize=9.5)
    fig.tight_layout()
    p = Path(out_dir) / "grant_2b_prestroke_crossday_cohort.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p
