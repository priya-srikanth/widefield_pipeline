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
        recs = []
        for _an, (pre, _bd) in sorted(per_animal.items()):
            if pre is None:
                continue
            recs.extend(pre if isinstance(pre, list) else [pre])
        recs = [r for r in recs if r is not None]
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

#: MORE transparent on the bar figures, and deliberately a different number from POINT_ALPHA. A
#: bar group can hold nineteen dots where a scatter panel holds seven, so the alpha that reads as
#: a separable cloud there reads as an opaque mass here -- and what it covers is the bar and its
#: error bar, which are the figure's actual claim. Priya, 2026-08-28.
BAR_POINT_ALPHA = 0.30

#: Dots sit ABOVE the bar and BELOW the error bar. Drawing them on top (the earlier zorder=5) hid
#: the interval behind the very points it was computed from, so raising transparency alone would
#: not have fixed it: at nineteen overlapping dots even a light alpha accumulates opaque.
BAR_POINT_Z = 2.5


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
            if pre is None:
                continue
            for r in (pre if isinstance(pre, list) else [pre]):
                out.append((an, value_of(an, None, r)))
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
    """The pooled epoch matrices: pre | acute | subacute, with each epoch's change beneath it.

    Serves figures 4, 5c and 5d, which differ only in which counts they are handed -- the LOSO
    matrix, the frozen-decoder matrix, and the same matrix as a difference. One renderer rather
    than three, for the reason `_pooled_bundle` was extracted: three copies of a panel that claim
    to show the same thing eventually stop doing so.

    ``counts`` is ``{epoch: MxM raw counts or None}``. Panels are ROW-NORMALISED for display only;
    the stored matrices stay raw counts so they remain addable, which is what makes epoch pooling
    a sum in the first place.

    TWO ROWS, NOT ONE LONG ROW. ``delta=True`` adds a second row holding acute - pre and
    subacute - pre, each placed DIRECTLY BENEATH the epoch it describes, so the eye reads down a
    column rather than hunting along five panels for which pair a difference refers to.

    THE BASELINE IS PRE-STROKE, NOT THE PREVIOUS EPOCH (Priya, 2026-08-28). subacute - acute
    answers "did it recover from its worst point"; subacute - pre answers "has it returned to
    baseline", which is the question a recovery figure is asked, and it is the one a reader
    assumes is being answered unless told otherwise. Both deltas therefore share the SAME
    reference and the same colour scale, so the two panels are directly comparable to each other
    -- which they are not when one is measured against pre and the other against acute.

    Differences are of ROW-NORMALISED matrices, not of counts: the epochs have very different n
    (acute 16 sessions, subacute 14, neither balanced across animals), so a count difference would
    mostly report how many sessions each epoch happens to contain.

    ``annotate=False`` by default -- Priya, for the crossnobis panels: numbers in the boxes are
    unreadable at a quarter page and fight the colour they duplicate. The colour bars carry the
    scale instead, one per row, since the two rows are on different scales and units.

    ``coverage`` is `epoch_coverage(...)['per_epoch']`; when given, each post panel's title carries
    its per-animal session counts, because a heatmap has no axis to hang the session dots on.

    ``labels`` names the positions. It is a PARAMETER because this renderer has no business
    deciding what the rows of a matrix it was handed mean; the digit fallback exists for tests, and
    a real figure showing 0-5 instead of the position names is unreadable rather than merely terse.
    """
    import matplotlib.pyplot as plt

    order = [e for e in PANELS if counts.get(e) is not None]
    if not order:
        return None
    norm = {}
    for e in order:
        M = np.asarray(counts[e], float)
        rows = M.sum(1, keepdims=True)
        norm[e] = np.divide(M, rows, out=np.full_like(M, np.nan), where=rows > 0)

    # which epochs get a delta panel: everything post-stroke that has a pre to be measured against
    deltas = [e for e in order if e != "pre"] if (delta and "pre" in order) else []
    ncol, nrow = max(len(order), 1), 1 + bool(deltas)
    k = norm[order[0]].shape[0]
    labels = list(labels) if labels else [str(i) for i in range(k)]

    # THE CANVAS IS THE PLACED SIZE. Every figure here is QUARTER_IN wide whatever the panel count,
    # so a point size written below is the point size the reader gets -- no scaling arithmetic, and
    # two of these side by side in the deck carry identical type. Measured the other way first:
    # sizing the canvas per panel left a four-panel row rendering ticks at 6.0pt beside a
    # three-panel row's 8.0, a 25% difference with no overlap anywhere to give it away.
    fig_w = QUARTER_IN
    left_in, gutter_in = 0.60, 0.72          # y labels; colour bars
    top_in = 0.86 + 0.10 * bool(coverage)
    # THE ROW GAP HOLDS A SET OF ROTATED TICK LABELS. Priya, 2026-08-28: label the top
    # row's x axis too -- a reader should not have to count across to the row below
    # to learn what a column is. That costs vertical space between the rows, and
    # 0.34in (which only had to clear a title) leaves the labels sitting on the
    # delta row's titles.
    row_gap_in, bottom_in = 0.72, 0.46
    panel_in = (fig_w - left_in - gutter_in - 0.14 * (ncol - 1)) / ncol
    fig_h = top_in + nrow * panel_in + (nrow - 1) * row_gap_in + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _axes(r, c):
        x0 = (left_in + c * (panel_in + 0.14)) / fig_w
        y0 = 1.0 - (top_in + (r + 1) * panel_in + r * row_gap_in) / fig_h
        return fig.add_axes([x0, y0, panel_in / fig_w, panel_in / fig_h])

    im_main = im_delta = None
    lim = 0.0
    for e in deltas:
        lim = max(lim, float(np.nanmax(np.abs(norm[e] - norm["pre"]))))
    lim = lim or 1.0

    for c, e in enumerate(order):
        ax = _axes(0, c)
        im_main = ax.imshow(np.ma.masked_invalid(norm[e]), cmap=cmap, vmin=0, vmax=1)
        M = np.asarray(counts[e], float)
        acc = float(np.nansum(np.diag(M)) / M.sum()) if M.sum() else float("nan")
        ttl = f"{e}\nn={int(M.sum())}, acc={acc:.2f}"
        if coverage and e in coverage:
            ttl += "\n" + " ".join(f"{a[-2:]}:{n}" for a, n in sorted(coverage[e].items()) if n)
        ax.set_title(ttl, fontsize=FS_ANNOT)
        _ticks(ax, k, labels, first=(c == 0), xlabels=True)
        if annotate:
            _annotate(ax, norm[e], k)
        if chance is not None and not deltas:
            ax.set_xlabel(f"chance {chance:.2f}", fontsize=FS_TICK - 1)
        if c == 0:
            ax.set_ylabel("true position", fontsize=FS_LABEL - 1)

    # THE LEFTMOST DRAWN COLUMN OF THIS ROW, not column 0. The delta row starts under `acute`,
    # so keying the y labels to column 0 left the entire row unlabelled -- six unnamed rows of a
    # signed matrix, which the overlap check cannot see because nothing collided.
    first_delta = min(order.index(e) for e in deltas) if deltas else None
    for e in deltas:
        c = order.index(e)
        ax = _axes(1, c)
        D = norm[e] - norm["pre"]
        im_delta = ax.imshow(np.ma.masked_invalid(D), cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"{e} - pre", fontsize=FS_ANNOT)
        _ticks(ax, k, labels, first=(c == first_delta), xlabels=True)
        if annotate:
            _annotate(ax, D, k)
        if chance is not None:
            ax.set_xlabel(f"chance {chance:.2f}", fontsize=FS_TICK - 1)
        if c == first_delta:
            ax.set_ylabel("true position", fontsize=FS_LABEL - 1)
    # The pre column carries no delta by construction: pre - pre is zero everywhere, and drawing
    # it would invite a reader to compare a panel of exact zeros against two real ones.

    # ONE COLOUR BAR PER ROW. The rows are in different units -- recall in [0, 1] above, a signed
    # change below -- so a single shared bar would be wrong for one of them, and a delta heatmap
    # with no scale is a picture of signs: red at 0.03 looks exactly like red at 0.30.
    _cbar(fig, im_main, fig_w, fig_h, left_in, gutter_in, panel_in, top_in, 0, row_gap_in,
          "recall")
    if im_delta is not None:
        _cbar(fig, im_delta, fig_w, fig_h, left_in, gutter_in, panel_in, top_in, 1, row_gap_in,
              "change in recall")

    fig.suptitle(title, fontsize=FS_ANNOT + 0.5, y=0.995)
    q = pathlib.Path(out) / f"{name}.png"
    fig.savefig(q, dpi=200)
    plt.close(fig)
    return q


def _ticks(ax, k, labels, *, first, xlabels):
    """Position ticks. Only the leftmost panel of a ROW carries the y labels -- six repeated sets
    of those is most of the ink on a quarter page, and a row shares one y axis by construction.

    Every row carries its own X labels, though: the columns are what a reader is comparing across,
    and making them count down to another row to find out what a column is defeats the point of
    putting the delta directly beneath its epoch."""
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    # ROTATED, ALWAYS, and measured rather than assumed: rotated, a label's horizontal extent is
    # the type height (~0.11in at 8pt), where "cC" laid flat is ~0.13in. Flat labels crowded a row
    # that rotation had already passed clean.
    ax.set_xticklabels(labels if xlabels else [], rotation=90, fontsize=FS_TICK - 1)
    ax.set_yticklabels(labels if first else [], fontsize=FS_TICK - 1)


def _annotate(ax, M, k):
    for i in range(k):
        for j in range(k):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.0)


def _cbar(fig, im, fig_w, fig_h, left_in, gutter_in, panel_in, top_in, row, row_gap_in, label):
    """A slim colour bar in the reserved right gutter, aligned to one row of panels."""
    if im is None:
        return
    x0 = (fig_w - gutter_in + 0.10) / fig_w
    y0 = 1.0 - (top_in + (row + 1) * panel_in + row * row_gap_in) / fig_h
    cax = fig.add_axes([x0, y0, 0.16 / fig_w, panel_in / fig_h])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FS_TICK - 2.5)
    cb.set_label(label, fontsize=FS_ANNOT - 1.5)


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


def _group_rule(fig, ax, xs, groups, *, pad_in=0.07, gap_in=0.055):
    """Draw the second x level BELOW the tick labels, positioned by MEASURING them.

    The offset used to be a guessed constant in axes fractions (-0.150), and it put the rule
    through the middle of "Ipsi"/"Middle"/"Contra". Nothing could catch it: `_overlaps` excludes
    tick labels from the chrome checks by design -- they legitimately sit close to their own axis
    -- and the rule is a Line2D, not text, so no text-vs-text test covers it either. A line through
    a label is invisible to every automated check this module has.

    So it is measured instead of guessed: draw once, take the lowest tick-label edge, and hang the
    rule a fixed number of INCHES below it. That holds whatever the font size, the label length or
    the figure height, none of which the old constant knew about.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    bottoms = []
    for t in ax.get_xticklabels():
        if not str(t.get_text()).strip():
            continue
        try:
            bottoms.append(inv.transform_bbox(t.get_window_extent(rend)).y0)
        except Exception:                                              # noqa: BLE001
            continue
    ax_h_in = max(1e-6, ax.get_position().height * fig.get_size_inches()[1])
    base = min(bottoms) if bottoms else -0.13
    y_rule = base - pad_in / ax_h_in
    y_text = y_rule - gap_in / ax_h_in
    tr = ax.get_xaxis_transform()
    for label, lo_i, hi_i in groups:
        ax.plot([xs[lo_i] - 0.42, xs[hi_i] + 0.42], [y_rule, y_rule], transform=tr,
                color="0.35", lw=0.9, clip_on=False, zorder=6)
        ax.text((xs[lo_i] + xs[hi_i]) / 2.0, y_text, label, transform=tr, ha="center", va="top",
                fontsize=FS_LABEL - 1, color="0.15", clip_on=False)
    return y_text


def bar_row(values, out, *, name, title, ylabel, positions, points=None, chance=None,
            counts=None, ylim=(0.0, 1.0), subtitle=None, groups=None, tick_labels=None,
            marks=None):
    """Per-position values as grouped bars, one group per position and one bar per epoch.

    Serves figure 1b (behaviour hit rate per spout position) and the per-position decoding
    accuracies, which differ only in what they are handed.

    ``values`` is ``{epoch: {position: value}}`` where a value is a bare number or a
    ``(v, lo, hi, ...)`` tuple; `behaviour_by_epoch` returns the latter and its Wilson interval is
    drawn as an error bar. ``points`` is ``{epoch: {position: [(animal, value), ...]}}`` and becomes
    one dot per SESSION -- the honest picture of the session weighting, since the bar itself is
    weighted by session and the epochs are not balanced across animals.

    ``positions`` are the KEYS into ``values`` and ``points``; ``tick_labels`` is what is printed
    under each bar, defaulting to the keys. THEY HAVE TO BE SEPARATE, and the reason is not
    stylistic: the two-level axis prints "Ipsi" under both the near and the far triple, so the
    display labels are not unique and cannot address a dict. Passing them as the keys silently
    looked up nothing and drew a figure with a correct axis and no bars at all.

    ``groups`` is ``[(label, first index, last index), ...]`` from `split_labels`, and draws a
    SECOND LEVEL on the x axis: a rule spanning each triple with "Near" / "Far" beneath it. The
    abbreviations are for heatmaps, where six labels have to fit under a panel an inch wide; a bar
    figure has the room to name the positions and should use it.

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
                        fmt="none", ecolor="0.15", elinewidth=1.1, capsize=2.0,
                        zorder=BAR_POINT_Z + 1)
        for i, p in enumerate(positions):
            per = (points or {}).get(e, {}).get(p) or []
            # spread inside the bar, never across it: a dot that drifts under a neighbouring bar
            # is attributed to the wrong epoch by every reader who does not count.
            session_points(ax, xs[i] + dx, per, spread=bw * 0.30, size=POINT_SIZE * 0.55,
                           alpha=BAR_POINT_ALPHA, zorder=BAR_POINT_Z)

    if marks:
        # ABOVE THE TALLER OF (bar, its dots), so a mark never lands on the data it refers to.
        for j, e in enumerate(epochs_present):
            dx = (j - (m - 1) / 2.0) * bw
            for i, q in enumerate(positions):
                mk = (marks.get(e) or {}).get(q)
                if not mk:
                    continue
                v, _lo, hi = _value_and_ci(values[e].get(q))
                pts = [y for _a, y in ((points or {}).get(e, {}).get(q) or [])
                       if y is not None and np.isfinite(y)]
                top = max([z for z in (v, hi) if z is not None] + pts) if (v is not None) else None
                if top is None:
                    continue
                ax.text(xs[i] + dx, top + 0.015 * (ylim[1] - ylim[0]), mk, ha="center",
                        va="bottom", fontsize=FS_ANNOT, color="0.10", zorder=7)
    if chance is not None:
        ax.axhline(chance, color="0.35", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels or positions, fontsize=FS_TICK - 1)
    # The second level is drawn AFTER the layout is set, below -- it has to measure the tick
    # labels, which do not have a position until the figure has been laid out once.
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
    handles += [Line2D([], [], marker="o", ls="", color=colors[a], alpha=BAR_POINT_ALPHA,
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
    left_in, right_in = 0.62, 1.24
    # the band grows with the subtitle's LINE COUNT; a flat constant clipped the
    # bootstrap line off the top the moment the stats line wrapped to two
    top_in = 0.52 + 0.15 * len(str(subtitle).split(chr(10))) * bool(subtitle)
    # the second x level needs its own band; without it the rule and its label fall off
    bottom_in = 0.46 + 0.34 * bool(groups)
    fig.subplots_adjust(left=left_in / fig_w, right=1 - right_in / fig_w,
                        top=1 - top_in / fig_h, bottom=bottom_in / fig_h)
    if groups:
        _group_rule(fig, ax, xs, groups)
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


def split_labels(order, animals=None):
    """``(minor labels, [(group label, first index, last index), ...])`` for a two-level x axis.

    The abbreviations earn their keep on a heatmap, where six tick labels have to fit under a
    panel an inch and a bit wide. A bar figure has room to say what they mean, so it says it:
    "Ipsi / Middle / Contra" under each tick, and "Near" / "Far" under a rule spanning each triple.
    Priya, 2026-08-28.

    DERIVED FROM THE SAME `anatomical_labels`, so the two levels cannot disagree with the
    abbreviations used elsewhere, and the near/far grouping is read off the long names rather than
    assumed to be the first three and the last three -- a reordered `CONF_LABELS` would otherwise
    put the rule under the wrong bars while every label on it stayed correct.
    """
    longs = anatomical_labels(order, short=False, animals=animals)
    minor = [x.split(" ", 1)[1].capitalize() if " " in x else x for x in longs]
    groups, start = [], 0
    for i, x in enumerate(longs):
        head = x.split(" ", 1)[0]
        last = (i == len(longs) - 1)
        nxt = None if last else longs[i + 1].split(" ", 1)[0]
        if last or nxt != head:
            groups.append((head.capitalize(), start, i))
            start = i + 1
    return minor, groups


# ---------------------------------------------------------------------------------------------
# UNCERTAINTY: animals -> sessions -> blocks
# ---------------------------------------------------------------------------------------------
# Priya, 2026-08-28: session-level, clustered by animal -- and then, "can it be by blocks nested
# within session?" It can, for anything built on `_collect_5c`, because those records carry the
# scheduler's own block ids as their third element.
#
# WHY THREE LEVELS AND NOT ONE. A Wilson interval over pooled trials answers "how precisely do we
# know this rate", treating 20,000 trials from four mice as 20,000 independent observations. It is
# the wrong question and the answer is far too narrow. The variance that matters here enters at
# three points: mice differ, a mouse's sessions differ, and within a session the ~6-trial position
# block is the unit the scheduler actually randomises -- trials inside one share a position and a
# moment in the session, so they are not independent either.
#
# THE ANIMAL DRAW IS THE OUTER ONE AND IT IS PAIRED. Pre and acute come from the SAME four mice, so
# a contrast resamples animals ONCE per draw and takes that animal's pre and acute sessions inside
# it. Resampling the two arms independently would discard the pairing the design provides and widen
# every interval for no reason.
#
# WHAT IT CANNOT BUY: n=4. Four animals is four, and the outer draw has only 35 distinct
# multisets. The interval is honest about between-animal variance but it is coarse, and no
# resampling scheme converts four mice into more.

def _blocks_of(rec):
    """``[(y, p), ...]`` grouped by block id -- the resampling unit inside one session."""
    y, p, b = np.asarray(rec[0]), np.asarray(rec[1]), np.asarray(rec[2])
    out = {}
    for k in np.unique(b):
        m = (b == k)
        out[int(k)] = (y[m], p[m])
    return list(out.values())


def _sessions_of(per_animal, animal, epoch):
    """Every session's blocks for one animal in one epoch: ``[[(y, p), ...], ...]``.

    A ``pre`` entry may be ONE record or a LIST of them. `_collect_5c` gives one -- a
    leave-one-session-out pool that is already an aggregate over that animal's pre-stroke days --
    while the behaviour tables give one record per session, and there the session level is real
    and must be resampled. Accepting both is what lets behaviour and decoding share this bootstrap
    instead of growing a second one that agrees with it until it doesn't.
    """
    pre, by_day = per_animal.get(animal, (None, {}))
    if epoch == "pre":
        if pre is None:
            return []
        if isinstance(pre, list):
            return [_blocks_of(r) for r in pre if r is not None]
        return [_blocks_of(pre)]
    days = [d for a, d in group_days_by_epoch(per_animal).get(epoch, []) if a == animal]
    return [_blocks_of(by_day[d]) for d in days if by_day.get(d) is not None]


def _draw(sessions, rng):
    """One hierarchical draw from one animal-epoch: sessions with replacement, then blocks."""
    if not sessions:
        return None
    picks = rng.integers(0, len(sessions), len(sessions))
    ys, ps = [], []
    for i in picks:
        blocks = sessions[i]
        if not blocks:
            continue
        for j in rng.integers(0, len(blocks), len(blocks)):
            y, p = blocks[j]
            ys.append(y)
            ps.append(p)
    if not ys:
        return None
    return np.concatenate(ys), np.concatenate(ps)


def contrast_ci(per_animal, epoch_a, epoch_b, stat, *, rng, n_boot=2000, alpha=0.05):
    """Interval on ``stat(epoch_a) - stat(epoch_b)``. See `contrast_draws` for the resampling.

    Returns ``(point, lo, hi, n_draws)`` at the requested ``alpha``.
    """
    got = contrast_draws(per_animal, epoch_a, epoch_b, stat, rng=rng, n_boot=n_boot)
    if got is None:
        return None
    point, d = got
    return (point, float(np.percentile(d, 100 * alpha / 2)),
            float(np.percentile(d, 100 * (1 - alpha / 2))), len(d))


def contrast_draws(per_animal, epoch_a, epoch_b, stat, *, rng, n_boot=2000):
    """``(point, draws)`` for ``stat(epoch_a) - stat(epoch_b)``: animals, then sessions, then blocks.

    ``stat(y, p) -> float | None`` reduces a pooled trial set (for instance accuracy at one
    position). ``point`` is computed on the REAL data, not on the bootstrap mean, so the number
    printed on a figure is the number in the data.

    THE DRAWS ARE RETURNED RATHER THAN AN INTERVAL because the same draws have to be summarised at
    more than one level -- an uncorrected 95% and a Bonferroni-corrected one across the twelve
    comparisons a per-position figure makes. Re-running the bootstrap per level would cost twice
    and, worse, give two intervals from two different resamples, so a corrected interval could come
    out NARROWER than the uncorrected one it is supposed to contain.

    The animal draw is shared between the two epochs, which keeps the contrast paired.
    """
    animals = sorted(per_animal)
    if not animals:
        return None
    real_a, real_b = _pooled(per_animal, epoch_a), _pooled(per_animal, epoch_b)
    if real_a is None or real_b is None:
        return None
    pa, pb = stat(*real_a), stat(*real_b)
    if pa is None or pb is None:
        return None
    sess = {(an, e): _sessions_of(per_animal, an, e)
            for an in animals for e in (epoch_a, epoch_b)}
    diffs = []
    for _ in range(n_boot):
        pick = [animals[i] for i in rng.integers(0, len(animals), len(animals))]
        ya, pa_, yb, pb_ = [], [], [], []
        for an in pick:
            da, db = _draw(sess[(an, epoch_a)], rng), _draw(sess[(an, epoch_b)], rng)
            if da is None or db is None:
                continue                       # this animal has no session in one arm
            ya.append(da[0]); pa_.append(da[1])
            yb.append(db[0]); pb_.append(db[1])
        if not ya or not yb:
            continue
        va = stat(np.concatenate(ya), np.concatenate(pa_))
        vb = stat(np.concatenate(yb), np.concatenate(pb_))
        if va is not None and vb is not None:
            diffs.append(va - vb)
    if len(diffs) < n_boot // 4:
        return None
    return float(pa - pb), np.asarray(diffs, float)


def _pooled(per_animal, epoch):
    """``(y, p)`` for one epoch across animals, from the real (unresampled) records."""
    rec = pool_records(per_animal, epoch)
    return None if rec is None else (np.asarray(rec[0]), np.asarray(rec[1]))


#: Marks for a contrast whose interval excludes zero. Two levels, because a per-position figure
#: makes TWELVE comparisons (six positions x two epochs) and a single asterisk that ignores this
#: is the commonest way a figure overstates itself.
MARK_UNCORRECTED, MARK_CORRECTED = "*", "**"


def contrast_marks(draws, *, n_comparisons, alpha=0.05):
    """``("", "*", "**")`` for one set of bootstrap draws.

    ``*`` the uncorrected 95% interval excludes zero; ``**`` it still excludes zero after a
    Bonferroni correction across ``n_comparisons``. BOTH ARE SHOWN rather than only the corrected
    one, because with four animals the corrected interval is very wide and reporting only it would
    turn every real effect into a blank -- and only the uncorrected one would call twelve
    comparisons one. The figure states which is which; the reader decides.

    Summarised from ONE set of draws, so the corrected interval necessarily contains the
    uncorrected one. Two bootstraps would not guarantee that.
    """
    if draws is None or not len(draws):
        return ""
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if not (lo > 0 or hi < 0):
        return ""
    a = alpha / max(1, n_comparisons)
    clo, chi = np.percentile(draws, [100 * a / 2, 100 * (1 - a / 2)])
    return MARK_CORRECTED if (clo > 0 or chi < 0) else MARK_UNCORRECTED


def contrast_panel(rows, out, *, name, title, ylabel, positions, tick_labels=None, groups=None,
                   subtitle=None, n_comparisons=None):
    """The companion panel: each epoch's change from pre-stroke, per position, with intervals.

    ``rows`` is ``{epoch: {position: (point, lo, hi, clo, chi)}}`` -- the uncorrected and the
    Bonferroni-corrected bounds from the same draws. The bar figure beside this one carries the
    marks; this carries the numbers, because a mark says only "not zero" and the size of the change
    is the thing a grant figure is actually claiming.

    ZERO IS DRAWN, not implied. A difference plot without its zero line invites the reader to
    compare two points to each other when the comparison that matters is each to no change.
    """
    import matplotlib.pyplot as plt

    epochs = [e for e in PANELS if rows.get(e)]
    if not epochs or not positions:
        return None
    k, m = len(positions), len(epochs)
    fig_w, fig_h = QUARTER_IN, 2.60
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    xs = np.arange(k, dtype=float)
    off = 0.78 / m
    for j, e in enumerate(epochs):
        dx = (j - (m - 1) / 2.0) * off
        for i, q in enumerate(positions):
            v = rows[e].get(q)
            if v is None:
                continue
            pt, lo, hi, clo, chi = v
            # the CORRECTED interval as a thin whisker behind the uncorrected one, so the figure
            # shows what the correction costs instead of only its verdict
            ax.plot([xs[i] + dx, xs[i] + dx], [clo, chi], color=EPOCH_GREY.get(e, "0.4"),
                    lw=0.9, solid_capstyle="butt", zorder=2)
            ax.plot([xs[i] + dx, xs[i] + dx], [lo, hi], color=EPOCH_GREY.get(e, "0.4"),
                    lw=2.6, solid_capstyle="butt", zorder=3)
            ax.plot([xs[i] + dx], [pt], marker="o", ms=4.0, color="0.12", zorder=4)
    ax.axhline(0.0, color="0.35", lw=0.9, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels or positions, fontsize=FS_TICK - 1)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL - 1)
    ax.tick_params(axis="y", labelsize=FS_TICK - 1)
    ax.set_xlim(-0.6, k - 0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=EPOCH_GREY.get(e, "0.4"), lw=2.6, label=f"{e} - pre")
               for e in epochs]
    handles += [Line2D([], [], color="0.4", lw=2.6, label="95%"),
                Line2D([], [], color="0.4", lw=0.9,
                       label=f"95% / {n_comparisons}" if n_comparisons else "corrected")]
    ax.legend(handles=handles, fontsize=FS_ANNOT - 2.0, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False, handletextpad=0.4, borderpad=0.15,
              labelspacing=0.32, handlelength=1.1)
    fig.suptitle(title, fontsize=FS_ANNOT + 0.5, y=0.985)
    if subtitle:
        fig.text(0.5, 0.90, subtitle, ha="center", va="top", fontsize=FS_ANNOT - 2.0,
                 color="0.30")
    left_in, right_in = 0.62, 1.24
    # the band grows with the subtitle's LINE COUNT; a flat constant clipped the
    # bootstrap line off the top the moment the stats line wrapped to two
    top_in = 0.52 + 0.15 * len(str(subtitle).split(chr(10))) * bool(subtitle)
    bottom_in = 0.46 + 0.34 * bool(groups)
    fig.subplots_adjust(left=left_in / fig_w, right=1 - right_in / fig_w,
                        top=1 - top_in / fig_h, bottom=bottom_in / fig_h)
    if groups:
        _group_rule(fig, ax, xs, groups)
    q = pathlib.Path(out) / f"{name}.png"
    fig.savefig(q, dpi=200)
    plt.close(fig)
    return q


def block_counts(per_animal) -> dict:
    """``{epoch: number of distinct blocks}`` -- the bootstrap's innermost resampling unit.

    Stated on a figure alongside N animals and n sessions because it is the unit the interval is
    actually built from, and because it is the one a reader cannot infer: sessions are countable
    from the panel titles, blocks are not. Where an epoch's interval looks surprisingly tight or
    wide, the block count is usually the explanation.
    """
    out = {}
    for e in PANELS:
        rec = pool_records(per_animal, e)
        out[e] = int(len(np.unique(np.asarray(rec[2])))) if rec is not None else 0
    return out


def stats_line(per_epoch, *, blocks=None, n_boot=None, notes=None, unit="sessions", width=104):
    """``N=4 animals, n=74 sessions -- pre 44 (92:11 ...) | acute 16 (...) | subacute 14 (...)``

    Priya, 2026-08-28: state N and n on every figure -- animals and total sessions -- and the other
    resampling unit where one is used.

    BOTH NUMBERS, ALWAYS, because they answer different questions and each misleads alone. "n=74
    sessions" sounds like a large study and is four mice; "N=4 animals" says nothing about how
    unevenly those mice are spread across the panels. The per-epoch breakdown is what makes the
    imbalance legible -- PS95 is one of sixteen acute sessions and seven of fourteen subacute --
    and it cannot be recovered from either total.

    ``per_epoch`` must count SESSIONS, not whatever a panel happens to be built from. The first
    version summed the accuracy figure's pre panel as 4 because `_collect_5c` returns one
    leave-one-session-out record per animal, and printed "n=34 sessions" for a cohort with 74. A
    count on a figure has to mean the thing it names.

    WRAPPED, because a single line of this does not fit 6.2in and matplotlib does not clip a
    centred `fig.text` -- it runs off BOTH edges, which is how the first render lost the "N=4
    animals" at one end and half the subacute breakdown at the other.
    """
    import textwrap

    parts, totals, animals = [], 0, set()
    for e in PANELS:
        per = per_epoch.get(e) or {}
        if not any(per.values()):
            continue
        animals |= {a for a, c in per.items() if c}
        n = sum(per.values())
        totals += n
        inner = " ".join(f"{a[-2:]}:{c}" for a, c in sorted(per.items()) if c)
        parts.append(f"{e} {n} ({inner})")
    head = f"N={len(animals)} animals, n={totals} {unit}"
    chunks = [head + " -- " + "   |   ".join(parts)]
    if blocks:
        got = " / ".join(f"{e} {blocks[e]:,}" for e in PANELS if blocks.get(e))
        chunks.append(f"bootstrap: animals -> sessions -> blocks"
                      f"{f', {n_boot:,} draws' if n_boot else ''}; blocks {got}")
    for note in (notes or []):
        chunks.append(note)
    out = []
    for c in chunks:
        out.extend(textwrap.wrap(c, width=width) or [c])
    return chr(10).join(out)


def mean_matrix_by_epoch(mats: dict) -> tuple[dict, dict]:
    """``({epoch: mean matrix}, {epoch: {animal: n sessions}})`` from a `_matrices_*` collector.

    A MEAN OVER SESSIONS, not a sum, and the difference from `counts_by_epoch` is not incidental.
    The confusion collectors return raw counts, so pooling them is addition and every trial counts
    once. These return matrices that have ALREADY been reduced -- a correlation, a split-half
    reliability, a crossnobis distance -- and there is no meaningful way to add two of those. The
    session becomes the unit, which is exactly the weighting Priya asked for ("weighted mean, ie
    just use each session's value") and exactly why the session dots matter: the acute panel is
    six PS94 sessions against one PS95 session.

    ``mats`` is ``{animal: {"PRE": M, day: M, ...}}``. NaN cells are skipped per cell rather than
    per matrix, so one position gated out in one session does not discard that session's other
    five.
    """
    out, cov = {}, {}
    for e in PANELS:
        stack, per = [], {}
        for an, by in sorted(mats.items()):
            if e == "pre":
                M = by.get("PRE")
                if M is not None:
                    stack.append(np.asarray(M, float))
                    per[an] = per.get(an, 0) + 1
                continue
            for key, M in by.items():
                if key == "PRE" or M is None:
                    continue
                if epoch_of_day(an, int(key)) != e:
                    continue
                stack.append(np.asarray(M, float))
                per[an] = per.get(an, 0) + 1
        if not stack:
            continue
        with np.errstate(invalid="ignore"):
            out[e] = np.nanmean(np.stack(stack), axis=0)
        cov[e] = per
    return out, cov


def matrix_row(mats, out, *, name, title, labels, cmap="viridis", vmin=None, vmax=None,
               unit="correlation", coverage=None, subtitle=None, delta=True, annotate=False):
    """pre | acute | subacute of an ALREADY-REDUCED matrix, with each epoch's change beneath it.

    The sibling of `confusion_row` for the `_matrices_*` family. It does not row-normalise -- these
    are correlations, reliabilities or distances, already on their own scale -- and it takes the
    scale from the data unless told, because a crossnobis distance and a correlation do not share
    a sensible fixed range.
    """
    import matplotlib.pyplot as plt

    order = [e for e in PANELS if mats.get(e) is not None]
    if not order:
        return None
    deltas = [e for e in order if e != "pre"] if (delta and "pre" in order) else []
    ncol, nrow = len(order), 1 + bool(deltas)
    k = np.asarray(mats[order[0]]).shape[0]
    labels = list(labels) if labels else [str(i) for i in range(k)]
    if vmin is None or vmax is None:
        allv = np.concatenate([np.asarray(mats[e], float).ravel() for e in order])
        allv = allv[np.isfinite(allv)]
        vmin = float(np.nanmin(allv)) if vmin is None else vmin
        vmax = float(np.nanmax(allv)) if vmax is None else vmax

    fig_w = QUARTER_IN
    left_in, gutter_in = 0.60, 0.78
    top_in = 0.86 + 0.10 * bool(coverage) + 0.15 * len(str(subtitle).split("\n")) * bool(subtitle)
    # THE ROW GAP HOLDS A SET OF ROTATED TICK LABELS. Priya, 2026-08-28: label the top
    # row's x axis too -- a reader should not have to count across to the row below
    # to learn what a column is. That costs vertical space between the rows, and
    # 0.34in (which only had to clear a title) leaves the labels sitting on the
    # delta row's titles.
    row_gap_in, bottom_in = 0.72, 0.46
    panel_in = (fig_w - left_in - gutter_in - 0.14 * (ncol - 1)) / ncol
    fig_h = top_in + nrow * panel_in + (nrow - 1) * row_gap_in + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _ax(r, c):
        x0 = (left_in + c * (panel_in + 0.14)) / fig_w
        y0 = 1.0 - (top_in + (r + 1) * panel_in + r * row_gap_in) / fig_h
        return fig.add_axes([x0, y0, panel_in / fig_w, panel_in / fig_h])

    lim = 0.0
    for e in deltas:
        d = np.asarray(mats[e], float) - np.asarray(mats["pre"], float)
        lim = max(lim, float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 0.0)
    lim = lim or 1.0

    im_a = im_d = None
    for c, e in enumerate(order):
        ax = _ax(0, c)
        M = np.asarray(mats[e], float)
        im_a = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=vmax)
        ttl = f"{e}\nmean diag {np.nanmean(np.diag(M)):.2f}"
        if coverage and e in coverage:
            ttl += "\n" + " ".join(f"{a[-2:]}:{n}" for a, n in sorted(coverage[e].items()) if n)
        ax.set_title(ttl, fontsize=FS_ANNOT)
        _ticks(ax, k, labels, first=(c == 0), xlabels=True)
        if annotate:
            _annotate(ax, M, k)
        if c == 0:
            ax.set_ylabel("true position", fontsize=FS_LABEL - 1)
    first_delta = min(order.index(e) for e in deltas) if deltas else None
    for e in deltas:
        c = order.index(e)
        ax = _ax(1, c)
        D = np.asarray(mats[e], float) - np.asarray(mats["pre"], float)
        im_d = ax.imshow(np.ma.masked_invalid(D), cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"{e} - pre", fontsize=FS_ANNOT)
        _ticks(ax, k, labels, first=(c == first_delta), xlabels=True)
        if annotate:
            _annotate(ax, D, k)
        if c == first_delta:
            ax.set_ylabel("true position", fontsize=FS_LABEL - 1)

    _cbar(fig, im_a, fig_w, fig_h, left_in, gutter_in, panel_in, top_in, 0, row_gap_in, unit)
    if im_d is not None:
        _cbar(fig, im_d, fig_w, fig_h, left_in, gutter_in, panel_in, top_in, 1, row_gap_in,
              f"change in {unit}")
    fig.suptitle(title, fontsize=FS_ANNOT + 0.5, y=0.995)
    if subtitle:
        fig.text(0.5, 1.0 - 0.30 / fig_h, subtitle, ha="center", va="top",
                 fontsize=FS_ANNOT - 2.0, color="0.30")
    q = pathlib.Path(out) / f"{name}.png"
    fig.savefig(q, dpi=200)
    plt.close(fig)
    return q


#: Where the collapsed pre-stroke baseline sits on a days-since-lesion axis. Left of day 1 with a
#: visible gap, and its tick reads "pre" rather than a day number, because it is not a day: it is
#: that animal's whole baseline summed into one point.
PRE_X = -2.0


def timecourse_panel(per_day, out, *, name, title, ylabel, positions, tick_labels=None,
                     subtitle=None, boundaries=None, chance=None, ylim=(0.0, 1.06)):
    """One small axes per position: value against DAYS SINCE LESION, one dot per animal per day.

    Priya, 2026-08-28: keep the time course but pool across animals, so each position and day
    carries four dots. This is the figure the epoch boundaries were DRAWN FROM, and showing it
    beside the pooled epoch bars is what lets a reader check the boundaries rather than take them.

    ``per_day`` is ``{position: {animal: {day: value}}}``; ``boundaries`` is
    ``{animal: (last acute day, first subacute day)}`` and is drawn as shaded spans, per animal
    where they differ.

    DAYS ARE EACH ANIMAL'S OWN, not calendar dates. The lesion dates differ -- PS94/PS95 on 0816,
    PS92/PS93 on 0817 -- so a calendar axis would put four different post-stroke days in one
    column and the epoch structure would dissolve. Day 0 is the last pre-stroke session.
    """
    import matplotlib.pyplot as plt

    positions = list(positions)
    ncol = len(positions)
    if not ncol:
        return None
    colors = config.animal_color()
    fig_w = QUARTER_IN * 2.0                 # a time axis needs width; this is a half-page figure
    left_in, right_in = 0.62, 1.05
    top_in = 0.52 + 0.15 * len(str(subtitle).split("\n")) * bool(subtitle)
    bottom_in, gap_in = 0.52, 0.16
    panel_w = (fig_w - left_in - right_in - gap_in * (ncol - 1)) / ncol
    fig_h = top_in + 1.55 + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    days_all = sorted({d for per in per_day.values() for by in per.values() for d in by})
    lo = min(days_all) if days_all else -1
    hi = max(days_all) if days_all else 1
    axes = []
    for i, q in enumerate(positions):
        ax = fig.add_axes([(left_in + i * (panel_w + gap_in)) / fig_w, bottom_in / fig_h,
                           panel_w / fig_w, 1.55 / fig_h])
        axes.append(ax)
        # THE EPOCH SPANS, drawn per animal only where they differ, so a reader can see that the
        # boundary is not one date but one rule applied to four animals.
        if boundaries:
            for an, (acute_hi, sub_lo) in sorted(boundaries.items()):
                ax.axvspan(0.5, acute_hi + 0.5, color="0.85", alpha=0.28, lw=0, zorder=0)
                ax.axvline(sub_lo - 0.5, color=colors.get(an, "0.5"), lw=0.7, alpha=0.5,
                           ls=":", zorder=1)
        ax.axvline(0.5, color="0.25", lw=0.9, zorder=2)          # the lesion
        for an, by in sorted(per_day.get(q, {}).items()):
            xs = sorted(by)
            ax.plot(xs, [by[d] for d in xs], "-", color=colors.get(an, "k"), lw=0.8,
                    alpha=0.45, zorder=3)
            ax.scatter(xs, [by[d] for d in xs], s=13, color=colors.get(an, "k"),
                       alpha=0.85, edgecolors="none", zorder=4)
        if chance is not None:
            ax.axhline(chance, color="0.4", lw=0.7, ls="--", zorder=2)
        # THE BASELINE TICK IS NAMED, not numbered. Leaving it as "-2" invites reading it as a
        # day two before the lesion, when it is every pre-stroke session at once.
        post = [d for d in days_all if d > 0]
        ticks = [PRE_X] + [d for d in post if d % 2 == 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["pre"] + [str(int(d)) for d in ticks[1:]])
        ax.set_title((tick_labels or positions)[i], fontsize=FS_ANNOT)
        ax.set_ylim(*ylim)
        ax.set_xlim(lo - 0.6, hi + 0.6)
        ax.tick_params(labelsize=FS_TICK - 2)
        if i:
            ax.set_yticklabels([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(ylabel, fontsize=FS_LABEL - 1)
    fig.text((left_in + (fig_w - left_in - right_in) / 2) / fig_w, 0.055,
             "days from lesion (each animal's own)", ha="center", fontsize=FS_LABEL - 1)
    animal_legend(axes[-1], fontsize=FS_ANNOT - 1.5, loc="upper left")
    axes[-1].get_legend().set_bbox_to_anchor((1.02, 1.0))
    fig.suptitle(title, fontsize=FS_ANNOT + 0.5, y=0.985)
    if subtitle:
        fig.text(0.5, 1.0 - 0.22 / fig_h, subtitle, ha="center", va="top",
                 fontsize=FS_ANNOT - 2.0, color="0.30")
    q_ = pathlib.Path(out) / f"{name}.png"
    fig.savefig(q_, dpi=200)
    plt.close(fig)
    return q_
