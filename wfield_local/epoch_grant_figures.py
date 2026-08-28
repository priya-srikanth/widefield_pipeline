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

from wfield_local import config, epoch_figures as ef, epochs
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

def fig_behaviour_timecourse(out_dir):
    """1c: hit rate per position against DAYS SINCE LESION, one dot per animal per day.

    Priya, 2026-08-28: keep the time course but pool across animals, so each position and day
    carries four dots -- "helpful to show how I defined acute and subacute".

    THIS IS THE FIGURE THE BOUNDARIES WERE DRAWN FROM, which is why it belongs beside the pooled
    epoch bars rather than instead of them. The acute range is where far-contra accuracy sits below
    25% of that animal's own pre-stroke baseline; a reader can see that threshold being crossed and
    recrossed here, and can check the boundary instead of taking it.

    Days are each animal's OWN days since its lesion, not calendar dates: PS94/PS95 were lesioned
    on 0816 and PS92/PS93 on 0817, so a calendar axis would put four different post-stroke days in
    one column and dissolve the structure the figure exists to show.
    """
    from wfield_local.grant_figures import CONF_LABELS, _position_metrics, _sessions

    short = dict(zip(CONF_LABELS, _short_labels()))
    per_day = {short[p]: {} for p in CONF_LABELS}
    counts = {}
    # THE SAME SESSIONS THE EPOCH FIGURES USE. `_sessions` returns every registered session, so
    # taking it whole put the uncurated early-June dates on this axis and reported 61 pre sessions
    # where figure 1b reports 44 -- two figures side by side describing different cohorts.
    keep = {lab for labs in epochs.labels_by_epoch().values() for lab in labs}
    # ALL OF PRE-STROKE COLLAPSES TO ONE POINT, the same one figure 1b's pre bar is built from
    # (Priya, 2026-08-28). Two reasons beyond width: the June block sits near day -70 so a true
    # axis spends most of itself on empty space, and -- the point of this figure -- the acute
    # boundary is defined RELATIVE TO that collapsed baseline, so the baseline has to be on the
    # axis as the single number the rule actually uses rather than as a scatter the reader has to
    # average by eye.
    #
    # HITS AND TRIALS SUMMED, not rates averaged, which is how 1b pools: a session with more
    # trials at a position counts for more, and the point here must be the same number the bar is.
    early = {}
    for an in config.animals():
        for mmdd, day in _sessions(an):
            if f"{an}_{mmdd}" not in keep:
                continue
            e = ef.epoch_of_day(an, int(day))
            if e is None:
                continue
            met = _position_metrics(an, mmdd) or {}
            for p in CONF_LABELS:
                m = met.get(p)
                if m and m[3] >= 5:
                    if e == "pre":
                        h, n = early.setdefault(short[p], {}).setdefault(an, [0.0, 0])
                        early[short[p]][an] = [h + m[0] * m[3], n + m[3]]
                    else:
                        per_day[short[p]].setdefault(an, {})[int(day)] = float(m[0])
            counts.setdefault(e, {})
            counts[e][an] = counts[e].get(an, 0) + 1
    for q, by in early.items():
        for an, (h, n) in by.items():
            if n:
                per_day[q].setdefault(an, {})[ef.PRE_X] = float(h / n)
    if not any(per_day.values()):
        return None
    bounds = {a: (spec["acute"][1], spec["subacute_from"])
              for a, spec in epochs.EPOCH_SPEC.items()}
    return ef.timecourse_panel(
        per_day, out_dir, name="epoch_1c_behaviour_timecourse",
        title="Hit rate by spout position over days since lesion, one dot per animal per day",
        subtitle=ef.stats_line(counts, notes=[
            f"shaded = acute ({epochs.ACUTE_RULE}); dotted = each animal's first subacute day",
            "pre = that animal's whole pre-stroke baseline as one point, hits and trials summed "
            "-- the same number figure 1b's pre bar plots, and the one the acute rule is "
            "measured against"]),
        ylabel="hit rate", positions=_short_labels(), tick_labels=_minor(), boundaries=bounds)


def _behaviour_records():
    """``{animal: ([pre records], {day: record})}`` from the persisted per-trial tables.

    A behaviour session is encoded as the SAME ``(y_true, y_pred, blocks)`` record the decoding
    collectors return: ``y_true`` is the spout position, ``y_pred`` is that position on a hit and
    -1 on a miss. "Accuracy at position q" then reads out as the hit rate at q, and behaviour and
    decoding share one bootstrap -- animals, then sessions, then blocks -- rather than growing a
    second one that agrees with the first until somebody edits it.

    ENGAGED TRIALS ONLY, matching `_position_metrics`' `trials_engaged`, which is what the bars
    plot. Reward is auto-held after a miss run, so a sated animal's late misses are disengagement
    rather than spatial inaccuracy; including them would move the bars AND the interval.

    Returns ``{}`` when the tables are absent, so the caller can fall back to the session-level
    contrast instead of failing -- they only exist for sessions processed since 2026-08-28.
    """
    import pandas as pd

    from wfield_local.grant_figures import _sessions
    from wfield_local.paths import PathResolver

    root = Path(PathResolver().root("behavior_out")) / "sessions"
    out = {}
    for an in config.animals():
        pre, by_day = [], {}
        for mmdd, day in _sessions(an):
            d = root / an / f"2026{mmdd}"
            files = sorted(d.glob("*_trials.csv")) if d.exists() else []
            if not files:
                continue
            try:
                t = pd.read_csv(files[-1])
            except Exception:                                          # noqa: BLE001
                continue
            if not {"pos_idx", "hit", "block", "engaged"} <= set(t.columns):
                continue
            t = t[t["engaged"].astype(bool) & (t["block"] >= 0)]
            if t.empty:
                continue
            y = t["pos_idx"].to_numpy(int)
            hit = t["hit"].to_numpy(bool)
            rec = (y, np.where(hit, y, -1), t["block"].to_numpy(int))
            e = ef.epoch_of_day(an, int(day))
            if e == "pre":
                pre.append(rec)
            elif e is not None:
                by_day[int(day)] = rec
        if pre or by_day:
            out[an] = (pre, by_day)
    return out


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
    # PREFER THE TRIAL TABLES when they exist: they carry block ids, so behaviour gets the same
    # animals -> sessions -> blocks resampling the imaging panels use. Absent them, fall back to
    # the session-level contrast rather than fail -- the tables only exist for sessions processed
    # since 2026-08-28, and a half-backfilled tree must still produce a figure.
    blocks_recs = _behaviour_records()
    p_idx = _position_codes()
    used_blocks = {}          # per epoch: how many contrasts the blocks path actually served
    post = [e for e in ef.PANELS if e != "pre" and values.get(e)]
    n_comp = sum(len(values[e]) for e in post)
    marks, rows = {}, {}
    for e in post:
        marks[e], rows[e] = {}, {}
        for p in CONF_LABELS:
            key = short[p]
            if key not in values[e]:
                continue
            # TRY BLOCKS, FALL BACK PER CONTRAST. A non-empty `blocks_recs` is NOT evidence that
            # this contrast can be computed from it: a half-backfilled tree has every animal's
            # pre-stroke sessions and none of its post-stroke ones, so the blocks path returns
            # None for every acute and subacute contrast. Guarding on the dict being non-empty
            # would have silently dropped every mark from the figure -- which is what it did.
            got = None
            if blocks_recs:
                got = ef.contrast_draws(
                    blocks_recs, e, "pre",
                    lambda y, pr, c=int(p_idx[p]): _accuracy_at(y, pr, c),
                    rng=np.random.default_rng(_seed_for("behaviour", "hit", e, p)),
                    n_boot=N_BOOT)
                used_blocks[e] = used_blocks.get(e, 0) + (got is not None)
            if got is None:
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

    # THE SUBTITLE HAS TO NAME THE LEVEL THAT WAS ACTUALLY USED, per epoch, because a partly
    # backfilled tree can serve one epoch from blocks and another from sessions -- and a figure
    # whose panels were built two different ways has to say so rather than average the claim.
    if used_blocks and all(used_blocks.get(e) for e in post):
        nb = ef.block_counts({a: (v[0], v[1]) for a, v in blocks_recs.items()})
        sub = ef.stats_line(counts, blocks=nb, n_boot=N_BOOT)
    else:
        served = ", ".join(f"{e}: {'blocks' if used_blocks.get(e) else 'sessions'}" for e in post)
        sub = ef.stats_line(
            counts, n_boot=N_BOOT,
            notes=[f"bootstrap: animals -> sessions -> ({served}); a session-level arm means "
                   "that epoch's per-trial tables are not on disk yet"])
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


def _position_codes():
    """``{position name: numeric code}`` -- the behaviour table's `pos_idx` domain."""
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    return {nm: int(code) for code, nm in POSITION_NAMES.items()}


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




# ------------------------------------------------------- the already-reduced matrix families

#: (key, collector name, colour bar unit, colormap, fixed scale or None, title stem).
#: The scale is fixed only where the quantity has a natural range: a correlation does, a
#: crossnobis distance does not, and forcing one on it would compress every panel into a corner.
MATRIX_FAMILIES = (
    ("6", "_matrices_pattern", "pattern correlation", "viridis", (-1.0, 1.0),
     "Mean-pattern correlation against the pre-stroke reference"),
    ("7", "_matrices_splithalf", "split-half correlation", "viridis", (-1.0, 1.0),
     "Within-session split-half pattern similarity"),
    ("8", "_matrices_crossnobis", "crossnobis distance", "magma", None,
     "Crossnobis geometry, in pre-stroke units"),
)


def _matrix_family(key, collector, unit, cmap, scale, stem, out_dir, align, variant, wname):
    """One matrix family's pooled epoch row, from the collector the per-animal figures already use.

    NOTHING IS RECOMPUTED: `_matrices_pattern`, `_matrices_splithalf` and `_matrices_crossnobis`
    all bottom out in `_collect_7`, keyed by "PRE" or day, which is what an epoch is defined on.
    """
    from wfield_local import grant_figures as G

    mats, _days = getattr(G, collector)(align, variant)
    if not mats:
        return None
    pooled, cov = ef.mean_matrix_by_epoch(mats)
    if not pooled:
        return None
    vmin, vmax = (scale if scale else (None, None))
    sub = ef.stats_line(_session_counts(),
                        notes=["pooled as a MEAN OVER SESSIONS: these matrices are already "
                               "reduced, so a session is the unit and cannot be re-weighted "
                               "by its trial count"])
    return ef.matrix_row(
        pooled, out_dir, name=f"epoch_{key}_{collector.strip('_')}_{align}_{variant}",
        title=f"{stem} -- {wname}", labels=_short_labels(), cmap=cmap,
        vmin=vmin, vmax=vmax, unit=unit, coverage=cov, subtitle=sub, delta=True)




# --------------------------------------------------------- the per-day scalar families

def _fig_8g(out_dir, align, variant, wname):
    """8g pooled: per-position RDM row correlation against the pre-stroke geometry, by epoch.

    `_rdm_rows` gives ``{animal: {"PRE"|day: (per-position row r, whole r, n)}}`` -- one number per
    position per day, which is a bar figure once grouped.
    """
    from wfield_local import grant_figures as G
    from wfield_local.grant_figures import CONF_LABELS

    rows, _days = G._rdm_rows(align, variant)
    if not rows:
        return None
    short = dict(zip(CONF_LABELS, _short_labels()))
    idx = {short[q]: i for i, q in enumerate(CONF_LABELS)}

    def value_of(payload, key):
        try:
            per = payload[0]
            v = per.get(key) if isinstance(per, dict) else per[idx[key]]
        except Exception:                                              # noqa: BLE001
            return None
        return None if v is None else float(v)

    values, points = ef.scalar_by_epoch(rows, value_of, keys=_short_labels())
    if not values:
        return None
    return ef.bar_row(
        values, out_dir, name=f"epoch_8g_geometry_by_position_{align}_{variant}",
        title=f"Per-position RDM row correlation with the pre-stroke geometry -- {wname}",
        subtitle=ef.stats_line(_session_counts(), notes=[_MEAN_NOTE]),
        ylabel="row correlation", positions=_short_labels(), tick_labels=_minor(),
        groups=_groups(), points=points, counts=_totals(_session_counts()),
        ylim=(-0.2, 1.06))


def _fig_10(out_dir, align, variant, wname):
    """10 pooled: best-match accuracy and mean rank of the correct position, by epoch."""
    from wfield_local import grant_figures as G

    tables, _days = G._match_tables(align, variant)
    if not tables:
        return None
    by = {an: dict(per_day) for an, (_pre, _post, per_day) in tables.items()}
    made = []
    for key, i, lab, ylim in (("acc", 0, "best-match accuracy", (0.0, 1.06)),
                              ("rank", 1, "mean rank of the correct position", (0.0, 6.2))):
        values, points = ef.scalar_by_epoch(
            by, lambda p, _k, i=i: (None if p is None else float(p[i])), keys=[lab])
        if not values:
            continue
        p = ef.bar_row(
            values, out_dir, name=f"epoch_10_best_match_{key}_{align}_{variant}",
            title=f"Best-match {key} against the pre-stroke patterns -- {wname}",
            subtitle=ef.stats_line(_session_counts(), notes=[_MEAN_NOTE]),
            ylabel=lab, positions=[lab], points=points,
            counts=_totals(_session_counts()), ylim=ylim)
        if p:
            made.append(p)
    return made[0] if made else None


def _fig_11(out_dir, align, variant, wname):
    """11 pooled: the encoder's gain and shape terms, by epoch.

    `_enc_tables` gives ``{animal: {"PRE"|day: (raw, a, gain, per-position)}}``. `raw` and `gain`
    are the two scalars the per-animal figure plots against day; here they become two bars each.
    """
    from wfield_local import grant_figures as G

    tab, _days = G._enc_tables(align, variant)
    if not tab:
        return None
    LABELS = ["raw", "gain"]
    take = {"raw": 0, "gain": 2}

    def value_of(payload, key):
        try:
            v = payload[take[key]]
        except Exception:                                              # noqa: BLE001
            return None
        return None if v is None else float(v)

    values, points = ef.scalar_by_epoch(tab, value_of, keys=LABELS)
    if not values:
        return None
    return ef.bar_row(
        values, out_dir, name=f"epoch_11_encoder_gain_shape_{align}_{variant}",
        title=f"Encoder explained variance and gain -- {wname}",
        subtitle=ef.stats_line(_session_counts(), notes=[_MEAN_NOTE]),
        ylabel="value", positions=LABELS, points=points,
        counts=_totals(_session_counts()), ylim=(0.0, 1.30))


#: Stated on every already-reduced family, because it is the one thing that separates them from
#: the confusion figures: those pool by SUMMING raw counts (every trial once), these by AVERAGING
#: one value per session (every session once). Same figure shape, different weighting.
_MEAN_NOTE = ("pooled as a MEAN OVER SESSIONS: these values are already reduced per session, so a "
              "session is the unit and cannot be re-weighted by its trial count")

SCALAR_FAMILIES = (("8g", _fig_8g), ("10", _fig_10), ("11", _fig_11))


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
                    choices=("1b", "1c", "acc", "5c", "mat", "scal"))
    args = ap.parse_args(argv)
    out = args.output or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ("1b", "1c", "acc", "5c", "mat", "scal"))

    if "1b" in want:
        try:
            _report("1b", fig_behaviour(out))
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! 1b: {type(ex).__name__} {str(ex)[:160]}", flush=True)
    if "1c" in want:
        try:
            _report("1c", fig_behaviour_timecourse(out))
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! 1c: {type(ex).__name__} {str(ex)[:160]}", flush=True)

    #: The per-arm keys. DERIVED, not written out again: the guard below skips the whole arm loop
    #: when none of them is wanted, and listing them twice meant `--only scal` and `--only mat`
    #: broke out of the loop immediately and produced NOTHING, with no error and no report --
    #: an empty output directory and exit 0.
    ARM_KEYS = {"acc", "5c", "mat", "scal"}
    for disp, align, variant, wname in ARMS:
        if not (want & ARM_KEYS):
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
        if "scal" in want:
            for key, fn in SCALAR_FAMILIES:
                try:
                    _report(f"{key} {align}/{variant}", fn(out, align, variant, wname))
                except Exception as ex:                                # noqa: BLE001
                    print(f"  !! {key} {align}/{variant}: {type(ex).__name__} "
                          f"{str(ex)[:160]}", flush=True)
        if "mat" in want:
            for key, collector, unit, cmap, scale, stem in MATRIX_FAMILIES:
                try:
                    _report(f"{key} {align}/{variant}",
                            _matrix_family(key, collector, unit, cmap, scale, stem,
                                           out, align, variant, wname))
                except Exception as ex:                                # noqa: BLE001
                    print(f"  !! {key} {align}/{variant}: {type(ex).__name__} "
                          f"{str(ex)[:160]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
