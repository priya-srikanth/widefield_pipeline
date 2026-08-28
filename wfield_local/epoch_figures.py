"""Pooled CROSS-ANIMAL grant figures, stratified by recovery EPOCH instead of a time axis.

Priya, 2026-08-28. Three panels -- pre-stroke, acute, subacute -- pooled over all four animals, sized
to be read at a QUARTER PAGE or smaller.

NOTHING HERE RECOMPUTES ANYTHING. Every population these figures need already exists in
`grant_figures`' collectors, and the collectors already carry the trial-inclusion variant Priya asked
for:

  * `_collect_5c(align, variant)` -> ``{animal: (pre-stroke LOSO record, {day: record})}``, where a
    record is ``(y_true, y_pred, blocks)`` at TRIAL level and ``variant='working'`` is exactly
    "lick plus miss-while-working". Keyed by DAY, which is what an epoch is defined on.
  * `_collect_7(align, variant, min_trials)` -> per-day post trials kept as trials, pre-stroke
    sessions kept separate.
  * `_pooled_bundle(animal, align)` -> the shared joint-basis load behind figures 6, 6b, 7 and 8,
    memoised per (animal, alignment).
  * `_position_metrics(animal, mmdd)` -> per-position ``(hit_rate, ci_lo, ci_hi, n)`` behaviour.

So an epoch figure is an AGGREGATION: group the existing per-day records by `epochs.epoch_of`, then
concatenate. Trial-level records concatenate across animals directly; confusion counts are raw, so
pooling is a sum. That is the same summability that made the early/late split free, and it is why
this module adds no new analysis and cannot disagree with the per-animal figures -- it holds the same
objects.

QUARTER PAGE IS A SIZE CONSTRAINT ON TYPE, NOT ON INCHES. What a reader sees is
``fontsize x (placed width / figure width)``. A figure placed at a quarter page is ~6.2in wide, so a
6.2in figure renders 1:1 and an 11in one halves every label. These figures are therefore SMALL in
inches with LARGE fonts -- the opposite of the instinct, and the same measurement that drove
`coding_cross_` from 22.7in to 10.8in.

DAYS COME FROM `epochs.days_since_stroke`, not from `grant_figures._day`. The two agree for this
cohort and are not the same function: `_day` is month*31 ordering, exact only while every date sits
in a 31-day month, and its own docstring says so. `tests/test_epoch_figures.py` asserts they still
agree, so a September session divides them loudly rather than silently moving an epoch boundary.
"""
from __future__ import annotations

import pathlib

import numpy as np

from wfield_local import config, epochs

#: Panel order, left to right. `epochs.EPOCHS` is the single definition; restated for readability.
PANELS = epochs.EPOCHS

#: Placed at a quarter page (~6.2in on a 13.33in slide). Figures are built at this width so type
#: arrives at ~1:1; see the module docstring.
QUARTER_IN = 6.2

#: Font sizes for a figure that will NOT be scaled down. Roughly 1.6x the per-animal figures', which
#: are built at 10-16in and shrink to fit.
FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT = 11.0, 10.0, 9.0, 8.5


def epoch_of_day(animal: str, day: int) -> str | None:
    """Epoch for a post-stroke DAY NUMBER, the key `grant_figures`' collectors use.

    Pre-stroke days are negative and return ``'pre'``. Delegates the post-stroke boundaries to
    `epochs.EPOCH_SPEC` so there is one definition; a day between the acute range and the first
    subacute day returns None rather than being rounded to a neighbour.
    """
    if day < 0:
        return "pre"
    spec = epochs.EPOCH_SPEC.get(animal)
    if spec is None:
        return None
    lo, hi = spec["acute"]
    if lo <= day <= hi:
        return "acute"
    if day >= spec["subacute_from"]:
        return "subacute"
    return None


def group_days_by_epoch(per_animal: dict) -> dict[str, list[tuple[str, int]]]:
    """``{epoch: [(animal, day), ...]}`` from a collector's ``{animal: (pre, {day: rec})}``.

    Days that belong to no epoch are DROPPED and reported by `epoch_coverage`, never folded into a
    neighbouring panel.
    """
    out: dict[str, list[tuple[str, int]]] = {e: [] for e in PANELS if e != "pre"}
    for an, (_pre, by_day) in per_animal.items():
        for day in sorted(by_day):
            e = epoch_of_day(an, int(day))
            if e in out:
                out[e].append((an, int(day)))
    return out


def epoch_coverage(per_animal: dict) -> dict:
    """Per-epoch session counts per animal, plus anything unassigned.

    EVERY POOLED PANEL MUST BE ABLE TO STATE THIS. The epochs are not balanced -- PS95 contributes
    ONE acute session and PS92 TWO subacute ones -- so a panel that does not say so implies four
    equal contributors when one animal can dominate.
    """
    grouped = group_days_by_epoch(per_animal)
    cov = {e: {an: sum(1 for a, _ in v if a == an) for an in sorted(per_animal)}
           for e, v in grouped.items()}
    unassigned = [(an, int(d)) for an, (_p, bd) in per_animal.items() for d in bd
                  if epoch_of_day(an, int(d)) is None]
    return {"per_epoch": cov, "n": {e: len(v) for e, v in grouped.items()},
            "unassigned": sorted(unassigned)}


def pool_records(per_animal: dict, epoch: str) -> tuple | None:
    """Concatenate every ``(y_true, y_pred, blocks)`` record in one epoch, across animals.

    Legitimate because each record's predictions come from that animal's OWN frozen pre-stroke
    decoder scored on its own trials, so pooling counts positions, not models -- the same reason
    `_class_confusions` stores raw counts. Blocks are made unique per (animal, day) so a downstream
    block bootstrap cannot treat two animals' block 3 as one cluster.
    """
    if epoch == "pre":
        recs = [rec for _an, (pre, _bd) in per_animal.items() if (rec := pre) is not None]
        offs = list(range(len(recs)))
    else:
        recs, offs, k = [], [], 0
        for an, day in group_days_by_epoch(per_animal).get(epoch, []):
            r = per_animal[an][1].get(day)
            if r is not None:
                recs.append(r)
                offs.append(k)
            k += 1
    if not recs:
        return None
    y = np.concatenate([r[0] for r in recs])
    p = np.concatenate([r[1] for r in recs])
    b = np.concatenate([np.asarray(r[2]) + 10_000 * (i + 1) for i, r in zip(offs, recs)])
    return y, p, b


def behaviour_by_epoch(position_metrics, sessions_of, *, positions):
    """Per-position hit rate per epoch, POOLED ACROSS ANIMALS AND WEIGHTED BY SESSION.

    Priya, 2026-08-28: "weighted mean (ie just use each session's value)" -- so hits and trials are
    summed over every (animal, session) in the epoch and the rate is their ratio, exactly as
    `grant_figures.fig_behaviour` already computes its own baseline point. A session with more trials
    therefore counts for more, and an animal contributing six acute sessions counts for more than one
    contributing a single session. That is the requested weighting and it must be stated on the
    figure, because the alternative (mean of per-animal means) differs substantially in the acute
    panel where PS95 has one session and PS94 has six.

    ``position_metrics(animal, mmdd)`` and ``sessions_of(animal)`` are injected rather than imported
    so this is testable without the behaviour tree, and so the caller decides which store to read.
    Returns ``{epoch: {position: (rate, lo, hi, n_trials, n_sessions)}}``.
    """
    from wfield_local.grant_figures import _wilson

    acc = {e: {p: [0, 0, 0] for p in positions} for e in PANELS}   # hits, n, sessions
    for an in config.animals():
        for mmdd, day in sessions_of(an):
            e = epoch_of_day(an, int(day))
            if e not in acc:
                continue
            met = position_metrics(an, mmdd) or {}
            for pos in positions:
                m = met.get(pos)
                if not m or m[3] < 5:
                    continue
                acc[e][pos][0] += round(m[0] * m[3])
                acc[e][pos][1] += m[3]
                acc[e][pos][2] += 1
    out = {}
    for e, per_pos in acc.items():
        out[e] = {}
        for pos, (h, n, ns) in per_pos.items():
            if not n:
                continue
            r, lo, hi = _wilson(h, n)
            out[e][pos] = (r, lo, hi, n, ns)
    return out


def counts_by_epoch(per_animal: dict) -> dict:
    """``{epoch: confusion counts}``, pooled across animals. Raw counts, so a panel is a SUM."""
    from wfield_local.grant_figures import _counts
    return {e: _counts(pool_records(per_animal, e)) for e in PANELS}


# ---------------------------------------------------------------------------------------------
# SHOWING THE IMBALANCE RATHER THAN ASSERTING IT
# ---------------------------------------------------------------------------------------------
# Priya, 2026-08-28: "it can be made clear in the figure ... by using data points with transparency
# to allow overlap, with different colors for sessions from each mouse (ie 4 colors, 1 dot per
# session value)".
#
# This is the honest answer to session weighting. The pooled bar is weighted by session, so PS94
# dominates the acute panel (6 sessions) and PS95 the subacute (7) -- and a note saying so is read
# once and forgotten, while a column of dots is read every time the figure is looked at. One dot per
# SESSION VALUE means the reader can count the contributors and see that PS95's acute panel rests on
# a single session.
#
# IT DOES NOT APPLY TO THE HEATMAP PANELS. A confusion or crossnobis matrix has no axis to carry a
# per-session dot, so those state n and the per-animal session counts in the panel title instead.

#: Transparency for overlaid session points. Low enough that ~7 coincident dots are still separable.
POINT_ALPHA = 0.55
POINT_SIZE = 26


def session_points(ax, x, per_session, *, spread=0.16, size=POINT_SIZE, alpha=POINT_ALPHA,
                   zorder=5):
    """One dot per session value at ``x``, coloured by animal, spread deterministically.

    ``per_session`` is ``[(animal, value), ...]``. Colours come from `config.animal_color`, the
    single source of truth, rather than four colours chosen here -- the same animal must be the same
    colour in this figure and in every per-animal figure beside it.

    THE SPREAD IS DETERMINISTIC, NOT RANDOM. A jittered point cloud that moves between renders makes
    two versions of the same figure impossible to compare, and this project has already been bitten
    by figures that differed for reasons nobody could pin down. Points are laid out by their index
    within the group, so the same data always draws the same picture.

    Returns the number of points drawn, so a caller can assert the panel shows what it claims.
    """
    colors = config.animal_color()
    vals = [(a, v) for a, v in per_session if v is not None and np.isfinite(v)]
    if not vals:
        return 0
    n = len(vals)
    # centred, evenly spaced; a single point sits exactly on the tick
    offs = np.zeros(1) if n == 1 else np.linspace(-spread, spread, n)
    for (an, v), dx in zip(sorted(vals, key=lambda t: t[0]), offs):
        ax.scatter([x + dx], [v], s=size, color=colors.get(an, "k"), alpha=alpha,
                   edgecolors="none", zorder=zorder)
    return n


def animal_legend(ax, *, fontsize=None, loc="best", counts=None):
    """A four-entry animal legend, optionally annotated with each animal's session count.

    ``counts`` is ``{animal: n}``; when given the label reads ``PS95 (n=1)``, which is where the
    imbalance becomes unmissable rather than merely visible.
    """
    from matplotlib.lines import Line2D
    colors = config.animal_color()
    handles = [Line2D([], [], marker="o", ls="", color=colors[a], alpha=POINT_ALPHA,
                      markersize=6,
                      label=a if not counts else f"{a} (n={counts.get(a, 0)})")
               for a in sorted(colors)]
    ax.legend(handles=handles, fontsize=fontsize or FS_ANNOT - 1.5, loc=loc, frameon=False,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.25)


def per_session_values(per_animal, epoch, value_of):
    """``[(animal, value), ...]`` for one epoch, one value per SESSION -- the input `session_points`
    wants.

    ``value_of(animal, day, record)`` reduces a per-day record to the scalar the panel plots, so the
    same grouping serves accuracy, similarity and reliability without any of them re-deriving which
    sessions belong to which epoch.
    """
    out = []
    if epoch == "pre":
        for an, (pre, _bd) in sorted(per_animal.items()):
            if pre is not None:
                out.append((an, value_of(an, None, pre)))
        return out
    for an, day in group_days_by_epoch(per_animal).get(epoch, []):
        rec = per_animal[an][1].get(day)
        if rec is not None:
            out.append((an, value_of(an, day, rec)))
    return out


# ---------------------------------------------------------------------------------------------
# RENDERERS
# ---------------------------------------------------------------------------------------------


def confusion_row(counts, out, *, name, title, coverage=None, delta=True, chance=None,
                  annotate=False, cmap="viridis", labels=None):
    """The shared 3-epoch confusion panel: pre | acute | subacute [| delta].

    Serves figures 4, 5c and 5d, which differ only in which counts they are handed -- the LOSO
    matrix, the frozen-decoder matrix, and the same matrix as a difference. One renderer rather than
    three, for the reason `_pooled_bundle` was extracted: three copies of a panel that claim to show
    the same thing eventually stop doing so.

    ``counts`` is ``{epoch: MxM raw counts or None}``. Panels are ROW-NORMALISED for display only --
    the stored matrices stay raw counts so they remain addable, which is what makes epoch pooling a
    sum in the first place.

    ``delta`` adds a fourth panel, subacute minus acute in recall units, which is 5d. It is a
    difference of two ROW-NORMALISED matrices, not of counts: the epochs have very different n
    (acute 16 sessions, subacute 14, and neither balanced across animals), so a count difference
    would mostly report how many sessions each epoch happens to contain.

    ``annotate=False`` by default -- Priya, 2026-08-28, for the crossnobis panels: numbers in the
    boxes are unreadable at a quarter page and fight the colour they duplicate.

    ``coverage`` is `epoch_coverage(...)['per_epoch']`; when given, each post panel's title carries
    its per-animal session counts, because a heatmap has no axis to hang the session dots on.

    ``labels`` names the positions, normally `grant_figures.CONF_LABELS` shortened. It is a
    PARAMETER because this renderer has no business deciding what the rows of a matrix it was
    handed mean; the digit fallback exists for tests, and a real figure that shows 0-5 instead
    of far_L..far_R is unreadable rather than merely terse.
    """
    import matplotlib.pyplot as plt

    order = [e for e in PANELS if counts.get(e) is not None]
    if not order:
        return None
    cols = order + (["delta"] if delta and {"acute", "subacute"} <= set(order) else [])
    n = len(cols)
    # THE CANVAS IS THE PLACED SIZE. Every figure here is QUARTER_IN wide whatever the panel
    # count, so a point size written below is the point size the reader gets -- no scaling
    # arithmetic, and two of these side by side in the deck carry identical type.
    #
    # This was measured the other way first, and the measurement is why the rule is written
    # down. Sizing the canvas per panel (6.2in for three, 8.27in for four) left the four-panel
    # delta row rendering its ticks at 6.0pt where the three-panel row beside it rendered 8.0,
    # a 25% difference with NO overlap anywhere to give it away. Scaling the fonts up to
    # compensate then held the rendered size but bought it by growing type against panels that
    # had not grown -- 23 crowded tick labels. A fourth panel costs PANEL WIDTH; that is the
    # real price of a quarter page and it should be paid visibly rather than hidden in type.
    fig_w, fig_h = QUARTER_IN, 2.05 + 0.10 * (n > 3)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]
    labels = list(labels) if labels else None
    norm = {}
    for e in order:
        M = np.asarray(counts[e], float)
        rows = M.sum(1, keepdims=True)
        norm[e] = np.divide(M, rows, out=np.full_like(M, np.nan), where=rows > 0)
    for ax, key in zip(axes, cols):
        if key == "delta":
            D = norm["subacute"] - norm["acute"]
            lim = float(np.nanmax(np.abs(D))) or 1.0
            ax.imshow(np.ma.masked_invalid(D), cmap="RdBu_r", vmin=-lim, vmax=lim)
            # THE SCALE, IN THE TITLE. The row carries no colour bar -- at this width one would
            # cost a panel -- so without the limit printed the delta panel is a picture of signs
            # with no magnitude, and red at 0.03 looks exactly like red at 0.30.
            ttl = f"subacute - acute\nscale +/-{lim:.2f} recall"
        else:
            M = np.asarray(counts[key], float)
            ax.imshow(np.ma.masked_invalid(norm[key]), cmap=cmap, vmin=0, vmax=1)
            acc = float(np.nansum(np.diag(M)) / M.sum()) if M.sum() else float("nan")
            ttl = f"{key}\nn={int(M.sum())}, acc={acc:.2f}"
            if coverage and key in coverage:
                per = coverage[key]
                ttl += "\n" + " ".join(f"{a[-2:]}:{c}" for a, c in sorted(per.items()) if c)
        k = norm[order[0]].shape[0]
        labels = labels or [str(i) for i in range(k)]
        ax.set_title(ttl, fontsize=FS_ANNOT)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        # ROTATED, ALWAYS, and this was measured rather than assumed. Laying two-character
        # labels flat looks like the obvious saving and is not: rotated, a label's HORIZONTAL
        # extent is the type height (~0.11in at 8pt), where "cC" flat is ~0.13in. Flat labels
        # crowded on the three-panel row, which rotation had already passed clean.
        ax.set_xticklabels(labels, rotation=90, fontsize=FS_TICK - 1)
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=FS_TICK - 1)
        if annotate:
            for i in range(k):
                for j in range(k):
                    v = (norm[key][i, j] if key != "delta" else None)
                    if v is not None and np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5)
        if chance is not None and key != "delta":
            ax.set_xlabel(f"chance {chance:.2f}", fontsize=FS_TICK - 1)
    axes[0].set_ylabel("true position", fontsize=FS_LABEL - 1)
    fig.suptitle(title, fontsize=FS_ANNOT + 0.5, y=0.995)
    # ABSOLUTE INCH MARGINS, not tight_layout: imshow fixes an aspect, which makes the figure
    # "not compatible with tight_layout" -- one warning per panel into the nightly log -- and a
    # negotiated layout is not reproducible as panel counts change.
    #
    # `subplots_adjust` takes FRACTIONS, so constants there are not absolute at all -- the
    # gutter would grow with the canvas. These are inch targets divided by the canvas, which
    # is what keeps the y-label gutter and the title band the same physical size as the row
    # gains a panel. Every one of them exists to hold TEXT, and the text no longer scales.
    left_in, right_in = 0.53, 0.03
    top_in, bottom_in = 0.82, 0.41
    fig.subplots_adjust(left=left_in / fig_w, right=1 - right_in / fig_w,
                        top=1 - top_in / fig_h, bottom=bottom_in / fig_h, wspace=0.22)
    q = pathlib.Path(out) / f"{name}.png"
    fig.savefig(q, dpi=200)
    plt.close(fig)
    return q


#: Epoch bar colours. Deliberately a GREY RAMP, not four hues: the dots already spend the colour
#: budget on animal identity, and a second categorical palette beside it makes the reader ask which
#: colour system a given mark belongs to. Light-to-dark also reads as an ordering, which epochs are.
EPOCH_GREY = {"pre": "#c9c9c9", "acute": "#8a8a8a", "subacute": "#4a4a4a"}


def _value_and_ci(v):
    """``(value, lo, hi)`` from either a bare number or a `behaviour_by_epoch` tuple.

    Both shapes occur by design -- behaviour carries a Wilson interval, a decoding accuracy read off
    a confusion diagonal does not -- and a renderer that accepted only one would force its caller to
    invent the other, which is how a fabricated interval gets drawn.
    """
    if v is None:
        return None, None, None
    if isinstance(v, (tuple, list)):
        return (float(v[0]),
                float(v[1]) if len(v) > 1 and v[1] is not None else None,
                float(v[2]) if len(v) > 2 and v[2] is not None else None)
    return float(v), None, None


def bar_row(values, out, *, name, title, ylabel, positions, points=None, chance=None,
            counts=None, ylim=(0.0, 1.0), subtitle=None):
    """Per-position values as grouped bars, one group per position and one bar per epoch.

    Serves figure 1b (behaviour hit rate per spout position) and the per-position decoding
    accuracies, which differ only in what they are handed.

    ``values`` is ``{epoch: {position: value}}`` where a value is a bare number or a
    ``(v, lo, hi, ...)`` tuple; `behaviour_by_epoch` returns the latter and its Wilson interval is
    drawn as an error bar. ``points`` is ``{epoch: {position: [(animal, value), ...]}}`` and becomes
    one dot per SESSION -- the honest picture of the session weighting, since the bar itself is
    weighted by session and the epochs are not balanced across animals.

    ``subtitle`` carries the PER-EPOCH session counts. The legend cannot: it has one entry per
    animal, so the only count it can show is that animal's total across all three epochs -- and a
    total is precisely what hides the imbalance the dots exist to expose. PS95 contributing 19
    sessions overall says nothing about its contributing ONE acute session and SEVEN subacute.

    THE LEGEND SITS IN A RESERVED GUTTER, not inside the axes. Two legends -- three epochs and four
    animals -- placed on a 6.2in axes will overlap the bars or each other at some data range, and a
    legend that moves with the data is not a reproducible layout. An inch and a fifth of the canvas
    is spent so the figure is self-contained wherever it is placed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    epochs_present = [e for e in PANELS if values.get(e)]
    if not epochs_present or not positions:
        return None
    m, k = len(epochs_present), len(positions)

    fig_w, fig_h = QUARTER_IN, 2.60
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bw = 0.78 / m
    xs = np.arange(k, dtype=float)
    for j, e in enumerate(epochs_present):
        dx = (j - (m - 1) / 2.0) * bw
        vals, los, his = [], [], []
        for p in positions:
            v, lo, hi = _value_and_ci(values[e].get(p))
            vals.append(np.nan if v is None else v)
            los.append(np.nan if (v is None or lo is None) else v - lo)
            his.append(np.nan if (v is None or hi is None) else hi - v)
        ax.bar(xs + dx, vals, width=bw * 0.9, color=EPOCH_GREY.get(e, "#8a8a8a"),
               edgecolor="none", zorder=2, label=e)
        if np.any(np.isfinite(los)) or np.any(np.isfinite(his)):
            ax.errorbar(xs + dx, vals, yerr=[np.nan_to_num(los), np.nan_to_num(his)],
                        fmt="none", ecolor="0.25", elinewidth=0.8, capsize=1.5, zorder=3)
        for i, p in enumerate(positions):
            per = (points or {}).get(e, {}).get(p) or []
            # spread inside the bar, never across it: a dot that drifts under a neighbouring bar
            # is attributed to the wrong epoch by every reader who does not count.
            session_points(ax, xs[i] + dx, per, spread=bw * 0.30, size=POINT_SIZE * 0.55)

    if chance is not None:
        ax.axhline(chance, color="0.35", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(positions, fontsize=FS_TICK - 1)
    # THE CHANCE LEVEL GOES IN THE Y-LABEL, not on the line. Annotating the line inside the axes
    # printed grey text across the dark subacute bars wherever the bars were tall there -- and
    # `_overlaps` cannot see it, because text over its OWN axes is excluded by design (a panel
    # title legitimately sits over its own panel). So this one is unreachable by the detector and
    # has to be prevented by construction rather than caught.
    ax.set_ylabel(ylabel if chance is None else f"{ylabel} (chance {chance:.2f})",
                  fontsize=FS_LABEL - 1)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=FS_TICK - 1)
    ax.set_xlim(-0.6, k - 0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    colors = config.animal_color()
    handles = [Patch(facecolor=EPOCH_GREY.get(e, "#8a8a8a"), label=e) for e in epochs_present]
    handles += [Line2D([], [], marker="o", ls="", color=colors[a], alpha=POINT_ALPHA,
                       markersize=5, label=a if not counts else f"{a} ({counts.get(a, 0)})")
                for a in sorted(colors)]
    ax.legend(handles=handles, fontsize=FS_ANNOT - 2.0, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False, handletextpad=0.4,
              borderpad=0.15, labelspacing=0.32, handlelength=1.1)

    fig.suptitle(title, fontsize=FS_ANNOT + 0.5, y=0.985)
    if subtitle:
        fig.text(0.5, 0.90, subtitle, ha="center", va="top", fontsize=FS_ANNOT - 2.0,
                 color="0.30")
    # ABSOLUTE INCH MARGINS -- `subplots_adjust` takes fractions, so these are inch targets divided
    # by the canvas. The 1.24in right gutter is the legend's, and it is reserved rather than
    # negotiated so the axes is the same width on every figure in the family.
    left_in, right_in, top_in, bottom_in = 0.62, 1.24, 0.52 + 0.16 * bool(subtitle), 0.46
    fig.subplots_adjust(left=left_in / fig_w, right=1 - right_in / fig_w,
                        top=1 - top_in / fig_h, bottom=bottom_in / fig_h)
    q = pathlib.Path(out) / f"{name}.png"
    fig.savefig(q, dpi=200)
    plt.close(fig)
    return q

#: Spout position names by ANATOMY rather than by the rig. Priya, 2026-08-28.
#:
#: near/far replaces close/far, and ipsi/middle/contra replaces left/center/right. The second half
#: is the one that carries meaning: "the right spout" is a fact about the apparatus, "the
#: contralateral spout" is a fact about the lesion, and it is the second a reader of a stroke
#: figure needs. Abbreviated nI/nM/nC and fI/fM/fC -- "C" is unambiguous only because centre
#: became middle, which is half the reason that half of the rename is worth doing.
_ANATOMICAL = {"close_L": ("near ipsi", "nI"), "close_center": ("near middle", "nM"),
               "close_R": ("near contra", "nC"), "far_L": ("far ipsi", "fI"),
               "far_center": ("far middle", "fM"), "far_R": ("far contra", "fC")}

_SWAP = {"I": "C", "C": "I"}


def lesion_side(animals=None) -> str:
    """The cohort's lesion side, or a raise if it is not one side.

    Priya, 2026-08-28: derive it, "in case of a change in stroke location".
    """
    sides = {a: str((config.animals().get(a) or {}).get("stroke_laterality", "")).upper()[:1]
             for a in (animals or config.animals())}
    distinct = {v for v in sides.values() if v}
    if not distinct:
        raise ValueError("no stroke_laterality in animals.yaml: ipsi/contra cannot be named")
    if len(distinct) > 1:
        raise ValueError(
            f"mixed lesion sides {sides}: ipsi/contra cannot label a POOLED figure until the "
            "position axis is reflected per animal -- see `anatomical_labels`")
    return distinct.pop()


def anatomical_labels(order, short=True, animals=None):
    """Position labels as near/far x ipsi/middle/contra, DERIVED FROM THE LESION SIDE.

    IPSI AND CONTRA ARE PROPERTIES OF THE LESION, NOT OF THE RIG, so this refuses to guess. Every
    animal in this cohort is lesioned on the LEFT, which is why left maps to ipsi and right to
    contra uniformly here and the rename is a rename rather than a reflection of the spout axis.
    The data agrees rather than merely permitting it: the collapse is at far_R, the position
    contralateral to a left lesion, and it is at far_R in all four animals.

    A RIGHT-LESIONED ANIMAL WOULD INVERT THE MAPPING FOR THAT ANIMAL ALONE. A pooled figure that
    kept one label set would then average ipsi trials with contra trials under a single name --
    a wrong number wearing a correct-looking label, which is the failure class this repo keeps
    finding. So a mixed cohort raises here instead of mislabelling. The fix at that point is to
    reflect the position axis per animal BEFORE pooling, which is a real analysis change and has
    to be a deliberate one, not a silent consequence of adding a mouse.
    """
    left = lesion_side(animals) == "L"
    out = []
    for q in order:
        long, ab = _ANATOMICAL.get(str(q), (str(q), str(q)))
        if not left:
            long = (long.replace("ipsi", "\x00").replace("contra", "ipsi")
                        .replace("\x00", "contra"))
            ab = ab[:-1] + _SWAP.get(ab[-1], ab[-1])
        out.append(ab if short else long)
    return out
