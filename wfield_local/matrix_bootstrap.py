"""Per-CELL bootstrap intervals for the matrix families.

Priya, 2026-09-12: "did we give up on having any bootstrapping of fig 10 best-match matrices?"

WHAT WAS AND WAS NOT ALREADY TESTED. Family 10's scalar arms carry the full nested bootstrap --
`epoch_10` (best-match accuracy, rank), `epoch_10b` (per position), and `epoch_10cdiag` (the
DIAGONAL of the destination matrix) all run through `_scalar_figure`, so "did position P still match
itself, and did that change" is tested with intervals and corrected marks. What has never carried
uncertainty is the OFF-DIAGONAL: *which* position P moved toward. That is the substitution claim,
and it rested on a fraction-of-sessions drawn as a colour, with no interval and no null.

This module closes that, and it is deliberately family-agnostic. `matrix_row` draws six families --
pattern correlation, split-half, crossnobis, the two row-centred crossnobis variants, and best-match
destination -- and none of them had per-cell intervals. The unit of resampling is the one every
other figure in the deck uses, so a cell interval here is comparable to a bar interval there rather
than being a second, subtly different bootstrap.

THE RESAMPLING IS ANIMALS THEN SESSIONS, AND THERE IS NO THIRD LEVEL BY CONSTRUCTION -- the same
argument `scalar_contrast_draws` states: these collectors return ONE MATRIX PER SESSION, so the
trial-level reduction has already happened inside the collector and there are no trials left to
resample. The animal draw is shared between the two epochs of a contrast, so a delta stays paired.

READ A CELL MARK AS A CLAIM ABOUT THAT CELL ALONE. A 6x6 is 36 comparisons; `n_comparisons` defaults
to the number of cells actually tested rather than to 1, and both the uncorrected and corrected
intervals are kept for the same reason `contrast_marks` keeps both -- with four animals the
corrected interval is wide enough that reporting only it would blank every real effect.
"""
from __future__ import annotations

import numpy as np

from wfield_local import epoch_figures as ef

#: Draws per interval. Matches the scalar families so the two are comparable.
N_BOOT = 2000

#: Marks, reusing `epoch_figures`' symbols so one legend serves the whole deck.
MARK_UNCORRECTED = ef.MARK_UNCORRECTED
MARK_CORRECTED = ef.MARK_CORRECTED


def by_epoch(mats):
    """``{epoch: {animal: [M, ...]}}`` from a `_matrices_*` collector's ``{animal: {"PRE"|day: M}}``.

    Mirrors `epoch_figures.scalar_by_epoch`'s grouping exactly -- same ``PRE`` key, same
    `epoch_of_day` assignment -- so a cell interval and a bar interval describe the same sessions.
    Non-finite matrices are dropped here rather than inside the draw, so a session missing one
    epoch cannot silently shrink a different one.
    """
    out = {}
    for an, by in (mats or {}).items():
        for key, M in (by or {}).items():
            if M is None:
                continue
            A = np.asarray(M, float)
            if A.ndim != 2 or not np.isfinite(A).any():
                continue
            e = "pre" if key == "PRE" else ef.epoch_of_day(an, int(key))
            if not e:
                continue
            out.setdefault(e, {}).setdefault(an, []).append(A)
    return out


def _stack_mean(chosen):
    """NaN-aware mean over a list of equal-shaped matrices, or None if nothing usable."""
    if not chosen:
        return None
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(chosen), axis=0)


def cell_draws(grouped, epoch, *, rng, n_boot=N_BOOT):
    """``(point KxK, draws (n_boot, K, K))`` for one epoch's per-cell mean. None if too thin.

    ``grouped`` is `by_epoch`'s output. Cells that are NaN in every draw stay NaN rather than
    becoming zero -- a position gated out of an epoch has no value, which is not the same as a
    value of nothing.
    """
    per_animal = (grouped.get(epoch) or {})
    animals = sorted(per_animal)
    if not animals:
        return None
    point = _stack_mean([M for an in animals for M in per_animal[an]])
    if point is None:
        return None
    out = []
    for _ in range(n_boot):
        pick = [animals[i] for i in rng.integers(0, len(animals), len(animals))]
        chosen = []
        for an in pick:
            sa = per_animal[an]
            chosen += [sa[i] for i in rng.integers(0, len(sa), len(sa))]
        M = _stack_mean(chosen)
        if M is not None:
            out.append(M)
    if len(out) < n_boot // 4:
        return None
    return point, np.stack(out)


def delta_draws(grouped, epoch, base="pre", *, rng, n_boot=N_BOOT):
    """``(point, draws)`` for ``epoch - base``, per cell, PAIRED on the animal draw.

    Only animals present in BOTH epochs contribute, matching `scalar_contrast_draws`: an animal
    with no pre-stroke panel cannot supply a change, and letting it into one side of the difference
    would compare different cohorts.
    """
    a_by, b_by = grouped.get(epoch) or {}, grouped.get(base) or {}
    animals = sorted(set(a_by) & set(b_by))
    if not animals:
        return None
    pa = _stack_mean([M for an in animals for M in a_by[an]])
    pb = _stack_mean([M for an in animals for M in b_by[an]])
    if pa is None or pb is None or pa.shape != pb.shape:
        return None
    out = []
    for _ in range(n_boot):
        pick = [animals[i] for i in rng.integers(0, len(animals), len(animals))]
        ca, cb = [], []
        for an in pick:
            sa, sb = a_by[an], b_by[an]
            ca += [sa[i] for i in rng.integers(0, len(sa), len(sa))]
            cb += [sb[i] for i in rng.integers(0, len(sb), len(sb))]
        A, B = _stack_mean(ca), _stack_mean(cb)
        if A is not None and B is not None and A.shape == B.shape:
            out.append(A - B)
    if len(out) < n_boot // 4:
        return None
    return pa - pb, np.stack(out)


def cell_ci(draws, alpha=0.05):
    """``(lo, hi)`` matrices at ``1 - alpha``, NaN-aware down the draw axis."""
    with np.errstate(invalid="ignore"):
        lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def cell_marks(draws, *, reference=0.0, n_comparisons=None, alpha=0.05, one_sided=False):
    """Per-cell ``""`` / ``"*"`` / ``"**"`` against ``reference``.

    ``reference`` is 0 for a delta and the CHANCE LEVEL for an absolute panel -- a best-match
    destination cell is a fraction of sessions, and the question there is not "is it non-zero" but
    "is it more than the 1/6 a coin would give".

    ``one_sided`` MARKS ONLY CELLS ABOVE THE REFERENCE, and the absolute row of a
    fraction-of-sessions panel needs it. Measured on the first render of family 10cs: two-sided
    against chance marked about thirty of thirty-six cells, because a destination that is never
    chosen sits at 0.00 with a tight interval and is therefore "significantly below 1/6" -- true,
    trivial, and true of almost every off-diagonal cell by construction. A mark that appears
    everywhere carries no information and actively hides the handful of cells that mean something.
    The substitution claim is "this destination is chosen MORE than chance", so that is what gets
    marked. The DELTA row stays two-sided: a cell can genuinely rise or fall.

    ``n_comparisons`` defaults to the number of cells with a usable interval, which is the honest
    family size: a 6x6 tested everywhere is 36 comparisons, and correcting as though it were one
    would be the mistake this argument exists to avoid.
    """
    lo, hi = cell_ci(draws, alpha)
    usable = np.isfinite(lo) & np.isfinite(hi)
    n = int(usable.sum()) if n_comparisons is None else int(n_comparisons)
    a = alpha / max(1, n)
    with np.errstate(invalid="ignore"):
        clo = np.nanpercentile(draws, 100 * a / 2, axis=0)
        chi = np.nanpercentile(draws, 100 * (1 - a / 2), axis=0)
    out = np.full(lo.shape, "", dtype=object)
    if one_sided:
        sig, csig = usable & (lo > reference), clo > reference
    else:
        sig = usable & ((lo > reference) | (hi < reference))
        csig = (clo > reference) | (chi < reference)
    out[sig] = MARK_UNCORRECTED
    out[sig & csig] = MARK_CORRECTED
    return out


def summarise(grouped, *, reference=0.0, delta_reference=0.0, seed=0, n_boot=N_BOOT,
              base="pre", panels=None, one_sided=True):
    """Everything a cell-interval figure and its sidecar need, for every epoch.

    Returns ``{epoch: {"point", "lo", "hi", "marks", "delta", "dlo", "dhi", "dmarks"}}`` with the
    delta keys absent on ``base``. One RNG threaded through all epochs, so a re-run reproduces and
    two panels of the same figure cannot disagree about which draws they came from.
    """
    rng = np.random.default_rng(seed)
    order = [e for e in (panels or ef.PANELS) if e in grouped]
    out = {}
    for e in order:
        got = cell_draws(grouped, e, rng=rng, n_boot=n_boot)
        if got is None:
            continue
        point, draws = got
        lo, hi = cell_ci(draws)
        rec = {"point": point, "lo": lo, "hi": hi,
               # ONE-SIDED on the absolute row: see `cell_marks`. Two-sided against chance marked
               # ~30 of 36 cells, because an unchosen destination sits at 0.00 and is trivially
               # "below 1/6".
               "marks": cell_marks(draws, reference=reference, one_sided=one_sided)}
        if e != base:
            d = delta_draws(grouped, e, base, rng=rng, n_boot=n_boot)
            if d is not None:
                dpoint, ddraws = d
                dlo, dhi = cell_ci(ddraws)
                rec.update(delta=dpoint, dlo=dlo, dhi=dhi,
                           dmarks=cell_marks(ddraws, reference=delta_reference))
        out[e] = rec
    return out


def write_cell_values(summary, q, *, labels=None, reference=None):
    """Sidecar for a cell-interval family: one row per (panel, row, col) with both intervals.

    Separate from `epoch_figures.write_matrix_values` because the shapes differ -- that one records
    a value and its delta, this one records four numbers and two marks per cell -- and collapsing
    them would drop the intervals, which are the entire reason this family exists.
    """
    import csv
    import pathlib

    from wfield_local import figure_layout as fl

    q = pathlib.Path(q)
    main = fl.sidecar(q, "_cells.csv")
    try:
        with open(main, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["panel", "row", "col", "value", "lo95", "hi95", "mark",
                        "delta_vs_pre", "delta_lo95", "delta_hi95", "delta_mark", "reference"])
            for e, rec in summary.items():
                P = np.asarray(rec["point"], float)
                names = ef._axis_labels(P.shape[0], labels, "r")
                cols = ef._axis_labels(P.shape[1], labels, "c")
                for i in range(P.shape[0]):
                    for j in range(P.shape[1]):
                        row = [e, names[i], cols[j], P[i, j], rec["lo"][i, j], rec["hi"][i, j],
                               rec["marks"][i, j]]
                        if "delta" in rec:
                            row += [rec["delta"][i, j], rec["dlo"][i, j], rec["dhi"][i, j],
                                    rec["dmarks"][i, j]]
                        else:
                            row += ["", "", "", ""]
                        row.append("" if reference is None else reference)
                        w.writerow(row)
    except Exception as ex:                                            # noqa: BLE001
        print(f"  [values] {main.name}: failed ({type(ex).__name__})", flush=True)
    return main


def figure(mats, out_dir, *, name, title, labels, unit, reference, cmap="viridis",
           vmin=None, vmax=None, subtitle=None, coverage=None, pre_sessions=None,
           seed=0, n_boot=N_BOOT, one_sided=True):
    """Draw a matrix family WITH per-cell marks, and write the interval sidecar beside it.

    Returns the figure path, or None when no epoch had enough sessions to resample.

    THE MARKS MEAN DIFFERENT THINGS ON THE TWO ROWS, and the subtitle has to say so. On the absolute
    row a mark is "this cell differs from ``reference``" -- chance, for a fraction-of-sessions
    panel. On the delta row it is "this cell CHANGED from pre-stroke", i.e. against zero. Reading a
    delta mark as "above chance" is the misreading this note exists to prevent.
    """
    grouped = by_epoch(mats)
    if not grouped:
        return None
    summary = summarise(grouped, reference=reference, delta_reference=0.0,
                        seed=seed, n_boot=n_boot, one_sided=one_sided)
    if len(summary) < 2:
        return None
    q = ef.matrix_row(
        {e: rec["point"] for e, rec in summary.items()}, out_dir,
        name=name, title=title, labels=labels, cmap=cmap, vmin=vmin, vmax=vmax, unit=unit,
        coverage=coverage, pre_sessions=pre_sessions, delta=True, annotate=False,
        cell_marks={e: rec["marks"] for e, rec in summary.items()},
        delta_cell_marks={e: rec["dmarks"] for e, rec in summary.items() if "dmarks" in rec},
        subtitle=subtitle)
    if q is not None:
        write_cell_values(summary, q, labels=labels, reference=reference)
    return q


def mark_note(reference, n_boot=N_BOOT):
    """The sentence every cell-marked panel must carry. One source, so no two can disagree."""
    return (f"PER-CELL marks from a nested animals->sessions bootstrap ({n_boot} draws), the same "
            f"unit the bar families use. TOP ROW: {MARK_UNCORRECTED} the 95% interval lies ABOVE "
            f"{reference:.3g} (the panel's reference), {MARK_CORRECTED} it still does after "
            f"Bonferroni across the tested cells. ONE-SIDED on purpose -- a destination that is "
            f"never chosen sits at 0.00 and is trivially 'below chance', which would mark almost "
            f"every off-diagonal cell and hide the few that mean something. BOTTOM ROW: TWO-SIDED "
            f"against ZERO, i.e. the cell CHANGED from pre-stroke in either direction. A "
            f"bottom-row mark is not a claim about chance.")
