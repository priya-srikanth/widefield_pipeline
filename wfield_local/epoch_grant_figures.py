"""Render the pooled cross-animal EPOCH figures.

Priya, 2026-08-28: pooled versions of the grant figures with three panels -- pre-stroke, acute and
subacute -- instead of a linear time axis, sized to be read at a quarter page or smaller. And:
*"I do NOT want this to have to re-create the wheel."*

SO THIS MODULE COMPUTES NOTHING. It calls `grant_figures`' existing collectors, groups their
per-day records by `epochs.epoch_of`, and hands the groups to `epoch_figures`' renderers. Every
population here is the SAME OBJECT the per-animal figures are drawn from, which is the only way two
figures captioned with the same trials can be relied on to contain them -- `tests/
test_epoch_figures.py` forbids this module from fitting a model or pooling sessions itself.

WHAT POOLING MEANS DIFFERS BY MEASURE, and the difference is not cosmetic:

  * `_collect_5c` returns TRIAL-LEVEL records `(y_true, y_pred, blocks)` scored by each animal's own
    frozen pre-stroke decoder. Pooling those is a CONCATENATION and the confusion matrix a SUM,
    because raw counts add. Every trial counts once, so a session with more trials counts for more.
  * `_matrices_*` return matrices that have already been reduced. Pooling those is a MEAN OVER
    SESSIONS -- which is exactly the weighting Priya asked for (*"weighted mean (ie just use each
    session's value)"*), and it is why the session dots matter: the acute panel is six PS94
    sessions against one PS95 session.

Run: ``python -m wfield_local.epoch_grant_figures [--only 1b 5c ...] [--output DIR]``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import config, epoch_figures as ef
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

#: (display name, alignment, trial class). The lick window admits only lick trials -- a trial with
#: no detected lick has no lick to align to -- which is `grant_figures._variants`' rule, restated
#: here as data because these figures name the class in their titles.
ARMS = (("ENL", "precue", "working", "ENL (pre-cue), lick + miss-while-working"),
        ("cue", "cue", "working", "post-cue, lick + miss-while-working"),
        ("lick", "lick", "lick", "post-lick, lick trials only"))

CHANCE = 1.0 / 6.0


def _session_counts():
    """``{epoch: {animal: sessions}}`` from the epoch assignment itself.

    One definition of "how many sessions is this epoch", shared by every figure, rather than each
    one counting whatever it happens to hold.
    """
    from wfield_local import epochs
    out = {}
    for e, labels in epochs.labels_by_epoch().items():
        per = {}
        for lab in labels:
            an = lab.split("_")[0]
            per[an] = per.get(an, 0) + 1
        out[e] = per
    return out


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


# --------------------------------------------------------------------------------- behaviour

def _behaviour_contrast(per_session, epoch, position, *, rng, n_boot):
    """``(point, draws)`` for the change in hit rate at one position, epoch minus pre.

    ``per_session`` is ``{epoch: {position: [(animal, hits, n), ...]}}``.

    ANIMALS THEN SESSIONS, and NOT trials-within-session. The trial level is deliberately absent
    rather than approximated: trials inside a position block share a position and a moment, so
    resampling them independently would treat correlated trials as independent and report an
    interval narrower than the data supports. Until the per-session trial tables are backfilled
    -- `spout_behavior` writes them from now on -- the session is the finest honest unit here, and
    the figure says so instead of implying it matched the imaging panels.

    Session-weighted within a draw (hits summed over n summed), which is the same weighting the
    bars use, so the interval describes the quantity actually plotted.
    """
    animals = sorted({a for e in (epoch, "pre")
                      for a, _h, _n in per_session.get(e, {}).get(position, [])})
    if not animals:
        return None

    def rate(rows):
        h = sum(x[1] for x in rows)
        n = sum(x[2] for x in rows)
        return (h / n) if n else None

    real_a = per_session.get(epoch, {}).get(position, [])
    real_b = per_session.get("pre", {}).get(position, [])
    pa, pb = rate(real_a), rate(real_b)
    if pa is None or pb is None:
        return None
    by = {(e, a): [r for r in per_session.get(e, {}).get(position, []) if r[0] == a]
          for e in (epoch, "pre") for a in animals}
    diffs = []
    for _ in range(n_boot):
        pick = [animals[i] for i in rng.integers(0, len(animals), len(animals))]
        ra, rb = [], []
        for an in pick:
            sa, sb = by[(epoch, an)], by[("pre", an)]
            if not sa or not sb:
                continue                       # animal absent from one arm
            ra += [sa[i] for i in rng.integers(0, len(sa), len(sa))]
            rb += [sb[i] for i in rng.integers(0, len(sb), len(sb))]
        va, vb = rate(ra), rate(rb)
        if va is not None and vb is not None:
            diffs.append(va - vb)
    if len(diffs) < n_boot // 4:
        return None
    return float(pa - pb), np.asarray(diffs, float)


def fig_behaviour(out_dir):
    """1b pooled: hit rate per spout position, per epoch, weighted by session."""
    from wfield_local.grant_figures import _position_metrics, _sessions, CONF_LABELS

    by = ef.behaviour_by_epoch(_position_metrics, lambda an: _sessions(an), positions=CONF_LABELS)
    if not any(by.values()):
        return None
    short = dict(zip(CONF_LABELS, _short_labels()))
    values = {e: {short[p]: v for p, v in per.items()} for e, per in by.items() if per}

    # ONE DOT PER SESSION, from the same store the bars are summed from -- not a second pass over
    # the behaviour tree, which could disagree with the bars it sits on.
    points = {e: {short[p]: [] for p in CONF_LABELS} for e in values}
    counts, per_sess = {}, {}
    for an in config.animals():
        for mmdd, day in _sessions(an):
            e = ef.epoch_of_day(an, int(day))
            if e not in points:
                continue
            met = _position_metrics(an, mmdd) or {}
            seen = False
            for p in CONF_LABELS:
                m = met.get(p)
                if m and m[3] >= 5:
                    points[e][short[p]].append((an, float(m[0])))
                    seen = True
            if seen:
                counts.setdefault(e, {})
                counts[e][an] = counts[e].get(an, 0) + 1
            for p in CONF_LABELS:
                m = met.get(p)
                if m and m[3] >= 5:
                    # hits and n, so a draw can re-weight by session exactly as the bars do
                    per_sess.setdefault(e, {}).setdefault(short[p], []).append(
                        (an, int(round(m[0] * m[3])), int(m[3])))
    post = [e for e in ef.PANELS if e != "pre" and values.get(e)]
    n_comp = sum(len(values[e]) for e in post)
    marks, rows = {}, {}
    for e in post:
        marks[e], rows[e] = {}, {}
        for p in CONF_LABELS:
            key = short[p]
            if key not in values[e]:
                continue
            got = _behaviour_contrast(per_sess, e, key,
                                      rng=np.random.default_rng(
                                          _seed_for("behaviour", "hit", e, p)),
                                      n_boot=N_BOOT)
            if got is None:
                continue
            point, draws = got
            marks[e][key] = ef.contrast_marks(draws, n_comparisons=n_comp)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            a = 0.05 / max(1, n_comp)
            clo, chi = np.percentile(draws, [100 * a / 2, 100 * (1 - a / 2)])
            rows[e][key] = (point, float(lo), float(hi), float(clo), float(chi))

    sub = ef.stats_line(
        counts, n_boot=N_BOOT,
        notes=["bootstrap: animals -> sessions (behaviour has no stored block ids yet; "
               "the imaging panels resample blocks within session as well)"])
    made = ef.bar_row(values, out_dir, name="epoch_1b_behaviour_by_position",
                      title="Behaviour: hit rate by spout position, pooled across animals "
                            "(one dot per session)",
                      # TOTALS in the legend, the PER-EPOCH breakdown in the subtitle. The legend
                      # answers "how many dots of this colour are on the figure at all"; the
                      # subtitle answers "how many are in the panel I am reading", which is the
                      # one that carries the imbalance.
                      subtitle=sub, counts=_totals(counts), marks=marks,
                      ylabel="hit rate", positions=_short_labels(), tick_labels=_minor(),
                      groups=_groups(), points=points,
                      # HEADROOM ABOVE 1.0 so a session at ceiling is a visible dot rather than a
                      # smear on the spine -- the near positions sit at ceiling in every epoch.
                      ylim=(0.0, 1.10))
    if any(rows.values()):
        ef.contrast_panel(
            rows, out_dir, name="epoch_1bdelta_behaviour_by_position",
            title="Change from pre-stroke in hit rate by spout position",
            subtitle=sub, ylabel="hit rate - pre", positions=_short_labels(),
            tick_labels=_minor(), groups=_groups(), n_comparisons=n_comp)
    return made


# ------------------------------------------------------------- decoding, from `_collect_5c`

def _named(record):
    """A record's ``(true, predicted)`` as POSITION NAMES.

    `_collect_5c` stores the NUMERIC position codes the decoder was fitted on, not the display
    names -- which is why `grant_figures._counts` maps through `POSITION_NAMES` before indexing
    `CONF_LABELS`. Comparing the raw codes to a name silently matches nothing: the first run of
    this figure produced empty panels and reported "wrote None", with no error anywhere.
    """
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES

    def nm(v):
        return np.array([POSITION_NAMES.get(int(x), str(x)) for x in np.asarray(v)])

    return nm(record[0]), nm(record[1])


#: Bootstrap draws per contrast. Well past where the 95% percentile stops moving; the outer
#: animal draw has only 35 distinct multisets, so more draws buy resolution on the inner levels
#: and nothing whatever on the outer one.
N_BOOT = 2000


def _seed_for(align, variant, epoch, position) -> int:
    """A stable seed, for the same reason `grant_figures._seed` exists: `hash()` is salted per
    process, so seeding a bootstrap with it gives a different interval on every render."""
    from wfield_local.grant_figures import _seed
    return _seed("epoch-contrast", align, variant, epoch, position)


def _code_of(position):
    """The NUMERIC code for a position name.

    `contrast_draws` gets its arrays straight from `pool_records`, which returns the records as
    stored -- the integer codes the decoder was fitted on, NOT display names. Comparing those to
    "close_L" matches nothing and returns an empty figure with no error, which is exactly the
    fault that produced three blank panels earlier today. Working in codes here keeps the mapping
    in one place instead of naming twenty thousand trials per bootstrap draw.
    """
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    for code, nm in POSITION_NAMES.items():
        if nm == position:
            return int(code)
    return None


def _accuracy_at(y, p, code):
    """Accuracy at one true position CODE, on raw (unnamed) arrays."""
    if code is None:
        return None
    y, p = np.asarray(y), np.asarray(p)
    m = (y == code)
    if m.sum() < 5:
        return None
    return float(np.mean(y[m] == p[m]))


def _accuracy_of(record, position=None):
    """Accuracy of one record, overall or at one true position. `None` when the class is absent."""
    if record is None:
        return None
    y, p = _named(record)
    if position is not None:
        m = (y == position)
        if m.sum() < 5:
            return None
        y, p = y[m], p[m]
    return float(np.mean(y == p)) if len(y) else None


def _per_position_accuracy(per_animal, out_dir, disp, align, variant, wname):
    """Per-position accuracy of the frozen pre-stroke decoder, by epoch."""
    from wfield_local.grant_figures import CONF_LABELS

    short = dict(zip(CONF_LABELS, _short_labels()))
    values, points = {}, {}
    for e in ef.PANELS:
        rec = ef.pool_records(per_animal, e)
        if rec is None:
            continue
        values[e] = {}
        points[e] = {}
        for q in CONF_LABELS:
            a = _accuracy_of(rec, q)
            if a is not None:
                values[e][short[q]] = a
            # one dot per session, from the SAME records the pooled bar sums
            pts = ef.per_session_values(per_animal, e, lambda _an, _d, r, q=q: _accuracy_of(r, q))
            points[e][short[q]] = [(an, v) for an, v in pts if v is not None]
    if not values:
        return None
    cov = ef.epoch_coverage(per_animal)
    per_epoch = dict(cov["per_epoch"])
    per_epoch["pre"] = {an: (1 if per_animal.get(an, (None, {}))[0] is not None else 0)
                        for an in per_animal}

    # THE CONTRAST AGAINST PRE-STROKE, per position, resampling animals -> sessions -> blocks.
    # Draws are taken ONCE per contrast and summarised at both an uncorrected and a corrected
    # level, so the corrected interval necessarily contains the uncorrected one -- two separate
    # bootstraps could not guarantee that and could print a corrected interval that was narrower.
    post = [e for e in ef.PANELS if e != "pre" and values.get(e)]
    n_comp = sum(len(values[e]) for e in post)          # every contrast this figure draws
    marks, rows = {}, {}
    for e in post:
        marks[e], rows[e] = {}, {}
        for q in CONF_LABELS:
            key = short[q]
            if key not in values[e]:
                continue
            got = ef.contrast_draws(
                per_animal, e, "pre",
                lambda y, pr, c=_code_of(q): _accuracy_at(y, pr, c),
                # SEEDED PER (window, class, epoch, position), stably: the same figure has to give
                # the same interval on every render, and `hash()` is salted per process.
                rng=np.random.default_rng(_seed_for(align, variant, e, q)), n_boot=N_BOOT)
            if got is None:
                continue
            point, draws = got
            marks[e][key] = ef.contrast_marks(draws, n_comparisons=n_comp)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            a = 0.05 / max(1, n_comp)
            clo, chi = np.percentile(draws, [100 * a / 2, 100 * (1 - a / 2)])
            rows[e][key] = (point, float(lo), float(hi), float(clo), float(chi))

    # SESSION COUNTS COME FROM THE EPOCH ASSIGNMENT, not from the panel's construction. The pre
    # panel here is four leave-one-session-out records -- one per animal -- so counting the panel
    # would print "pre 4" beside "acute 16" and imply the baseline rests on four sessions when it
    # rests on forty-four.
    sub = ef.stats_line(_session_counts(), blocks=ef.block_counts(per_animal), n_boot=N_BOOT,
                        notes=["pre panel is leave-one-session-out within each animal, "
                               "pooled across animals"])
    made = ef.bar_row(
        values, out_dir, name=f"epoch_acc_by_position_{align}_{variant}",
        title=f"Per-position decoding accuracy, {wname} -- pooled across animals",
        subtitle=sub, counts=_totals(per_epoch), marks=marks, points=points,
        ylabel="accuracy", positions=_short_labels(), tick_labels=_minor(), groups=_groups(),
        chance=CHANCE, ylim=(0.0, 1.10))
    if any(rows.values()):
        ef.contrast_panel(
            rows, out_dir, name=f"epoch_accdelta_by_position_{align}_{variant}",
            title=f"Change from pre-stroke in per-position accuracy, {wname}",
            subtitle=sub, ylabel="accuracy - pre", positions=_short_labels(),
            tick_labels=_minor(), groups=_groups(), n_comparisons=n_comp)
    return made


def _confusion_rows(per_animal, out_dir, disp, align, variant, wname):
    """5c/5d pooled: the frozen pre-stroke decoder's confusion per epoch, and the delta."""
    counts = ef.counts_by_epoch(per_animal)
    counts = {e: M for e, M in counts.items() if M is not None and np.asarray(M).sum()}
    if not counts:
        return []
    cov = ef.epoch_coverage(per_animal)["per_epoch"]
    made = []
    p = ef.confusion_row(
        counts, out_dir, name=f"epoch_5c_frozen_confusion_{align}_{variant}",
        title=f"Frozen PRE-stroke decoder, pooled across animals -- {wname}",
        coverage=cov, delta=True, chance=CHANCE, labels=_short_labels())
    if p:
        made.append(p)
    return made


# ------------------------------------------------------------------------------ the driver

def _epoch_arm(align, variant):
    """`_collect_5c`'s per-animal records, with anything outside an epoch reported not dropped."""
    from wfield_local.grant_figures import _collect_5c

    per_animal, _days = _collect_5c(align, variant)
    return per_animal


def _report(tag, path):
    """A figure that returned None is an ABSENCE, and it has to say so.

    `print(f"wrote {fn(...)}")` renders a None return as "wrote None", which scans as success in a
    log nobody reads closely -- and that is exactly how the per-position accuracy figure came back
    empty three times without anyone noticing. Same class as `_draw` swallowing a plotting error so
    a broken figure reports as zero missing.
    """
    if path:
        print(f"  wrote {path}", flush=True)
    else:
        print(f"  ?? {tag}: NO FIGURE -- the renderer found no data to draw", flush=True)


def main(argv=None) -> int:
    from wfield_local.console import use_utf8_stdout
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--only", nargs="+", default=None,
                    choices=("1b", "acc", "5c"))
    args = ap.parse_args(argv)
    out = args.output or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ("1b", "acc", "5c"))

    if "1b" in want:
        try:
            _report("1b", fig_behaviour(out))
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! 1b: {type(ex).__name__} {str(ex)[:160]}", flush=True)

    for disp, align, variant, wname in ARMS:
        if not (want & {"acc", "5c"}):
            break
        try:
            per_animal = _epoch_arm(align, variant)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! collect {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                  flush=True)
            continue
        cov = ef.epoch_coverage(per_animal)
        # UNASSIGNED DAYS ARE REPORTED, never folded into a neighbouring panel. A day between the
        # acute range and the first subacute day belongs to neither, and silently rounding it to
        # one would move a boundary Priya set.
        if cov["unassigned"]:
            print(f"  .. {align}/{variant} days in no epoch: {cov['unassigned']}", flush=True)
        print(f"  .. {align}/{variant} sessions per epoch: {cov['n']} {cov['per_epoch']}",
              flush=True)
        if "acc" in want:
            try:
                _report(f"acc {align}/{variant}",
                        _per_position_accuracy(per_animal, out, disp, align, variant, wname))
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! acc {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "5c" in want:
            try:
                for p in _confusion_rows(per_animal, out, disp, align, variant, wname):
                    _report(f"5c {align}/{variant}", p)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 5c {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
