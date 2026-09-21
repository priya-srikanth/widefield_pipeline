"""THE POOLED EPOCH RENDERER'S SHARED HELPERS.

Split out of `epoch_grant_figures` on 2026-09-21, alongside the five figure-family modules. These
are the helpers reached from MORE THAN ONE family -- scalar-figure layout, the session and pre
counts every panel states, the label shorteners, and the per-day seed. A helper used by one family
travels with it; one used across families has to live above all of them.

`_seed_for` is the reason several of these are shared rather than copied: a day's interval must
not depend on how many days preceded it in the run, or rendering a subset silently changes the
days that remain. One seed function, derived per day.
"""
from __future__ import annotations

import numpy as np

from wfield_local import epoch_figures as ef


def _session_counts(pre_override=None):
    """``{epoch: {animal: sessions}}`` from the epoch assignment itself.

    One definition of "how many sessions is this epoch", shared by every figure, rather than each
    one counting whatever it happens to hold.

    ``pre_override`` REPLACES THE PRE COUNT WITH WHAT THE PANEL ACTUALLY HOLDS, and only the pre
    count. The post-stroke epochs are stated from the assignment on purpose -- a family that drops
    a thin session should still say how many sessions the epoch HAS, with the drop visible in the
    block count. The pre panel is different: for the `stopped` class it is not a subset of the
    pre-stroke sessions but a small minority of them, because a well-trained pre-stroke animal
    barely quits. The first stopped render claimed "pre 44 (92:11 93:11 94:11 95:11)" over a panel
    built from FIFTEEN sessions, and the only visible sign was the block count falling from 3,688
    to 169 (Priya's stopped-class request, 2026-09-11). A count that is wrong by 3x is worse than
    no count.
    """
    from wfield_local import epochs
    out = {}
    for e, labels in epochs.labels_by_epoch().items():
        per = {}
        for lab in labels:
            an = lab.split("_")[0]
            per[an] = per.get(an, 0) + 1
        out[e] = per
    if pre_override:
        out["pre"] = dict(pre_override)
    return out
def _long_labels():
    """Full position names, title-cased: "Near Ipsi" ... "Far Contra".

    For panel TITLES, where there is room and where the two-level x axis is unavailable. Figure
    1c's six panels read "Ipsi / Middle / Contra" twice over, which is ambiguous the moment the
    Near and Far groups are not visibly bracketed together (Priya, 2026-08-28).
    """
    from wfield_local.grant_figures import CONF_LABELS
    return [x.title() for x in ef.anatomical_labels(CONF_LABELS, short=False)]
def _minor():
    """Per-tick labels for a bar figure: Ipsi / Middle / Contra."""
    from wfield_local.grant_figures import CONF_LABELS
    return ef.split_labels(CONF_LABELS)[0]
def _groups():
    """The second x level: a rule under each of Near and Far."""
    from wfield_local.grant_figures import CONF_LABELS
    return ef.split_labels(CONF_LABELS)[1]
def _short_labels():
    """Position labels for an axis: nI/nM/nC, fI/fM/fC.

    ANATOMY, NOT THE RIG -- and derived from `stroke_laterality` rather than hardcoded, so a
    right-lesioned animal raises instead of inheriting a label that would be backwards for it.
    """
    from wfield_local.grant_figures import CONF_LABELS
    return ef.anatomical_labels(CONF_LABELS)
def _totals(per_epoch):
    """``{animal: sessions across all epochs}`` -- the legend's count.

    Deliberately a different number from the subtitle's, and both are wanted: the legend says how
    many dots of a colour are on the figure, the subtitle says how they split across the panels.
    """
    out = {}
    for per in per_epoch.values():
        for an, c in (per or {}).items():
            out[an] = out.get(an, 0) + c
    return out
#: Bootstrap draws per contrast. Well past where the 95% percentile stops moving; the outer
#: animal draw has only 35 distinct multisets, so more draws buy resolution on the inner levels
#: and nothing whatever on the outer one.
N_BOOT = 2000
def _seed_for(align, variant, epoch, position) -> int:
    """A stable seed, for the same reason `grant_figures._seed` exists: `hash()` is salted per
    process, so seeding a bootstrap with it gives a different interval on every render."""
    from wfield_local.grant_figures import _seed
    return _seed("epoch-contrast", align, variant, epoch, position)
def _accuracy_at(y, p, code):
    """Accuracy at one true position CODE, on raw (unnamed) arrays."""
    if code is None:
        return None
    y, p = np.asarray(y), np.asarray(p)
    m = (y == code)
    if m.sum() < 5:
        return None
    return float(np.mean(y[m] == p[m]))
def _pre_counts(align, variant):
    """Per-animal PRE-STROKE SESSION counts, for the panel labels.

    The already-reduced families build ONE pre matrix per animal by averaging that animal's
    pre-stroke sessions leave-one-out, so a panel's own coverage reads 92:1 where the subtitle says
    92:11. See .
    """
    from wfield_local import grant_figures as _G
    return _G.pre_session_counts(align, variant)
def _scalar_figure(out_dir, *, name, title, ylabel, keys, values, points, tick_labels=None,
                   groups=None, chance=None, ylim=(0.0, 1.06), delta_name=None,
                   delta_title=None, delta_ylabel=None, notes=None, session_counts=None):
    """Bar row + marks + the epoch-minus-pre companion panel, for a per-session scalar family.

    ONE PATH FOR ALL OF THEM. 8g, 10, 10b and 11 differ only in which collector fills `values` and
    `points`, so the statistics, the marks and the companion panel are written once. Four copies
    would agree today and diverge the first time one of them gained a correction.

    ``notes`` EXTENDS the standard subtitle rather than replacing it, so a family with an unusual
    method can state it without any family losing the two lines every one of them needs (what the
    mean is over, and what the bootstrap resamples).

    **IT REWRITES ``values`` IN PLACE.** Every bar gets a bootstrap interval, and the interval is
    stored back as ``values[epoch][key] = (point, lo, hi)`` -- so a caller that reads its own
    ``values`` AFTER calling this gets a tuple where it put a float. That cost an afternoon: figure
    13n read ``vals[e]["state"]`` to compute a retention and got an array, and `np.isfinite` on an
    array is an array, so `if not np.isfinite(...)` raised "truth value ambiguous". Snapshot
    anything you need afterwards BEFORE the call. (Left in place rather than fixed by copying,
    because four existing families depend on the mutation to draw their intervals.)

    ``session_counts`` OVERRIDES the epoch assignment's counts. Every family here is built on
    TRIALS, so the assignment's per-epoch session counts describe them exactly; the behavioural-
    state family is built on one-second SEGMENTS that are not trials and whose sessions are a
    different set, and a subtitle that states the assignment's counts over that panel is simply
    wrong. The same correction `_position_bars` needed for the `stopped` class.
    """
    post = [e for e in ef.PANELS if e != "pre" and values.get(e)]
    if not post or "pre" not in values:
        # NO PRE, NO CONTRAST -- and say so, rather than draw bars with no marks and let a reader
        # assume the test was run and came back null.
        return ef.bar_row(values, out_dir, name=name, title=title,
                          subtitle=ef.stats_line(
                              session_counts if session_counts is not None
                              else _session_counts(), notes=list(notes or []) + [
                              _MEAN_NOTE, "no pre-stroke arm in this collector: no contrast drawn"]),
                          ylabel=ylabel, positions=keys, tick_labels=tick_labels, groups=groups,
                          points=points, counts=_totals(_session_counts()), chance=chance,
                          ylim=ylim)
    n_comp = sum(len(values[e]) for e in post)
    # EVERY BAR GETS ITS OWN INTERVAL, including pre. Same resampling as the contrasts, so a bar
    # and the mark above it cannot come from two different schemes.
    for e in list(values):
        for k in list(values[e]):
            got = ef.scalar_value_draws(
                points, e, k, rng=np.random.default_rng(_seed_for(name, "value", e, k)),
                n_boot=N_BOOT)
            ci = ef.with_ci(got)
            if ci is not None:
                values[e][k] = ci
    marks, rows, thin_marks = {}, {}, set()
    for e in post:
        marks[e], rows[e] = {}, {}
        for k in keys:
            if k not in values[e]:
                continue
            # A MARK NEEDS TWO ANIMALS. With one, the outer bootstrap level has no variance to
            # draw on and the interval is within-animal scatter wearing a star -- see
            # `ef.contrast_animals`. The BAR and its interval still appear; only the mark is
            # withheld, and the subtitle says how many epochs were affected.
            if ef.contrast_animals(points, e, "pre", k) < ef.MIN_ANIMALS_FOR_MARK:
                # PER (epoch, key). Tracking it per EPOCH silenced a sound bar because its
                # NEIGHBOUR was thin -- 12b's acute ENGAGED arm lost its mark to the STOPPED arm.
                thin_marks.add(f"{e}/{k}")
                continue
            got = ef.scalar_contrast_draws(
                points, e, "pre", k,
                rng=np.random.default_rng(_seed_for(name, "scalar", e, k)), n_boot=N_BOOT)
            if got is None:
                continue
            point, draws = got
            marks[e][k] = ef.contrast_marks(draws, n_comparisons=n_comp)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            a = 0.05 / max(1, n_comp)
            clo, chi = np.percentile(draws, [100 * a / 2, 100 * (1 - a / 2)])
            rows[e][k] = (point, float(lo), float(hi), float(clo), float(chi))
    counts = session_counts if session_counts is not None else _session_counts()
    thin_note = ([f"NO MARK on {', '.join(sorted(thin_marks))} (epoch/bar): fewer than "
                  f"{ef.MIN_ANIMALS_FOR_MARK} animals contribute, so the animal level of the "
                  f"bootstrap has no variance and a star would assert more than one animal can "
                  f"support. The bar and its interval are still shown"] if thin_marks else [])
    sub = ef.stats_line(counts, n_boot=N_BOOT, notes=thin_note + [
        _MEAN_NOTE,
        "bootstrap: animals -> sessions. No block level: these values are one number per session, "
        "so the trial reduction already happened inside the collector"] + list(notes or []))
    made = ef.bar_row(values, out_dir, name=name, title=title, subtitle=sub, marks=marks,
                      ylabel=ylabel, positions=keys, tick_labels=tick_labels, groups=groups,
                      points=points, counts=_totals(counts), chance=chance, ylim=ylim)
    if any(rows.values()):
        ef.contrast_panel(rows, out_dir, name=delta_name or f"{name}_delta",
                          title=delta_title or f"Change from pre-stroke -- {title}",
                          subtitle=sub, ylabel=delta_ylabel or f"{ylabel} - pre",
                          positions=keys, tick_labels=tick_labels, groups=groups,
                          n_comparisons=n_comp)
    return made
#: Stated on every already-reduced family, because it is the one thing that separates them from
#: the confusion figures: those pool by SUMMING raw counts (every trial once), these by AVERAGING
#: one value per session (every session once). Same figure shape, different weighting.
_MEAN_NOTE = ("pooled as a MEAN OVER SESSIONS: these values are already reduced per session, so a "
              "session is the unit and cannot be re-weighted by its trial count")
