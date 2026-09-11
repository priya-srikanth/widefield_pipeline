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


# --------------------------------------------------------------------------------- behaviour

def _licks_per_trial_by_day(short, keep):
    """``{position: {animal: {day: licks/trial}}}`` -- the OTHER half of the chronic rule.

    FROM THE COHORT TABLE `epoch_audit` ITSELF READS (`lpt__<position>` in
    `cohort/cohort_session_metrics.csv`, engaged-gated), not a second derivation. The chronic
    boundary is a conjunction -- far-contra hit rate AND licks/trial both flat, recovered and
    settled -- so a reader checking the dashed line needs the same numbers the rule saw, and a
    parallel computation that drifted would make the figure disagree with the boundary it draws.

    Pre-stroke collapses to one point like the hit-rate row, but trial-WEIGHTED
    (sum(lpt x n) / sum(n)) rather than a mean of session means, matching how the hit-rate
    baseline sums hits and trials.
    """
    import pandas as pd

    from wfield_local import epoch_audit
    from wfield_local.grant_figures import CONF_LABELS, _position_metrics, _sessions

    fp = epoch_audit._cohort_path()
    if not fp.exists():
        print("  ?? licks/trial: no cohort table, row omitted", flush=True)
        return {}
    df = pd.read_csv(fp)
    if "n_engaged" in df.columns:                      # same duplicate rule as epoch_audit
        df = df.sort_values("n_engaged").drop_duplicates(["animal", "date"], keep="last")
    lpt = {}
    for _i, r in df.iterrows():
        mmdd = str(int(r["date"]))[4:]
        for p in CONF_LABELS:
            v = r.get("lpt__" + p)
            if v is not None and pd.notna(v):
                lpt.setdefault((str(r["animal"]), mmdd), {})[p] = float(v)
    if not lpt:
        return {}

    out, early = {short[p]: {} for p in CONF_LABELS}, {}
    for an in config.animals():
        for mmdd, day in _sessions(an):
            if an + "_" + mmdd not in keep:
                continue
            e = ef.epoch_of_day(an, int(day))
            if e is None:
                continue
            met = _position_metrics(an, mmdd) or {}
            vals = lpt.get((an, mmdd)) or {}
            for p in CONF_LABELS:
                m, v = met.get(p), vals.get(p)
                # THE SAME >=5 ENGAGED-TRIAL FLOOR the hit-rate row uses, so the two rows describe
                # the same session set and a point present in one is present in the other.
                if v is None or not m or m[3] < 5:
                    continue
                if e == "pre":
                    s_, n_ = early.setdefault(short[p], {}).setdefault(an, [0.0, 0])
                    early[short[p]][an] = [s_ + v * m[3], n_ + m[3]]
                else:
                    out[short[p]].setdefault(an, {})[int(day)] = v
    for q, by in early.items():
        for an, (s_, n_) in by.items():
            if n_:
                out[q].setdefault(an, {})[ef.PRE_X] = float(s_ / n_)
    return out if any(out.values()) else {}


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
    lick_day = _licks_per_trial_by_day(short, keep)
    # `spec_for`, not EPOCH_SPEC: this is THE figure the boundaries are read off, so it must draw
    # the ones actually in force. Reading the stored dict here would show a chronic line the pooled
    # panels beside it disagree with, and the figure whose job is to let a reader check the
    # boundaries would be the one lying about them.
    bounds = {a: (s_["acute"][1], s_["subacute_from"], s_.get("chronic_from"))
              for a in epochs.EPOCH_SPEC for s_ in [epochs.spec_for(a)] if s_}
    # ONE PANEL PER ANIMAL, six positions overlaid (Priya, 2026-09-08). The epochs are per-animal,
    # so the per-POSITION layout had to draw all four animals' boundaries on every axes -- eight
    # vertical lines belonging to no particular trace. Per animal, a shaded span means that
    # animal's acute period and the recovery can be read straight off it.
    return ef.timecourse_by_animal(
        per_day, out_dir, name="epoch_1c_behaviour_timecourse",
        title="Hit rate and licks/trial by spout position over days since lesion, "
              "one panel per animal",
        subtitle=ef.stats_line(counts, notes=[
            (f"shaded = that animal's acute period ({epochs.ACUTE_RULE}); subacute is simply "
             f"everything after it, so the shading's right edge marks that onset. dashed = "
             f"its first chronic day where one exists ({epochs.CHRONIC_RULE}) -- only PS92 "
             "has stabilised, so only PS92 carries a dashed line"),
            "ENGAGED TRIALS ONLY (the `hit_rate` column, hits_engaged / trials_engaged): the "
            "gate is judged at the reference positions and backdated to the start of the run "
            "of misses that trips it, so far misses after a lesion stay in as the deficit "
            "rather than being deleted as disengagement",
            "colour = spout position (hue = side, lighter = far); pre = that animal's whole "
            "pre-stroke baseline as one point, hits and trials summed -- the same number figure "
            "1b's pre bar plots, and the one the acute rule is measured against"]),
        # LONG names in the LEGEND: "Ipsi" appearing twice would not say which distance it is.
        # THE AXIS NAMES THE POPULATION (Priya, 2026-09-08). Engaged and all-trial rates
        # diverge hard in the acute period -- PS94 8/23 far contra is 0.129 engaged against
        # 0.08 on all trials, on 62 of 100 trials -- so a bare "hit rate" is two figures.
        ylabel="hit rate (engaged trials)", positions=_short_labels(),
        tick_labels=_long_labels(),
        # THE SECOND HALF OF THE CHRONIC RULE. The dashed line requires far-contra hit rate
        # AND licks/trial both flat, recovered and settled; with only the hit-rate row drawn,
        # half of what placed that line was invisible. ylim from the data -- licks/trial is
        # not a rate and forcing 0-1 would flatten it against the axis.
        extra_rows=[(lick_day, "licks / trial (engaged)", None)] if lick_day else None,
        boundaries=bounds)


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

    WITHIN-ANIMAL: each session's delta is its own rate minus that animal's pre rate, and the epoch
    value is the SESSION-WEIGHTED MEAN of those (every session counts once). So the delta is normalised
    per animal even though it is still pooled by session -- the interval describes that quantity.
    """
    animals = sorted({a for e in (epoch, "pre")
                      for a, _h, _n in per_session.get(e, {}).get(position, [])})
    if not animals:
        return None

    def rate(rows):
        h = sum(x[1] for x in rows)
        n = sum(x[2] for x in rows)
        return (h / n) if n else None

    # WITHIN-ANIMAL, SESSION-WEIGHTED MEAN (Priya, 2026-08-29). Each epoch session's delta is its own
    # rate minus THAT animal's pre rate -- so PS94's acute is measured against PS94's pre, not against a
    # pooled pre that another animal dominates -- and every session counts once, so a six-session animal
    # weighs 6x (the "just use each session's value" weighting, applied to the within-animal delta).
    # ONLY the delta normalises within animal; the absolute/raw panel (behaviour_by_epoch) is unchanged.
    epoch_rows = per_session.get(epoch, {}).get(position, [])          # one (animal, hits, n) per session
    pre_rows = {a: [r for r in per_session.get("pre", {}).get(position, []) if r[0] == a]
                for a in animals}
    pre_rate = {a: rate(pre_rows[a]) for a in animals}

    def within_animal_mean(ep_rows, pre_by_animal):
        ds = [(h / n) - pre_by_animal[a] for a, h, n in ep_rows
              if n and pre_by_animal.get(a) is not None]
        return float(np.mean(ds)) if ds else None

    point = within_animal_mean(epoch_rows, pre_rate)
    if point is None:
        return None
    by_ep = {a: [r for r in epoch_rows if r[0] == a] for a in animals}
    diffs = []
    for _ in range(n_boot):                    # ANIMALS then SESSIONS; pre re-estimated per draw
        pick = [animals[i] for i in rng.integers(0, len(animals), len(animals))]
        ds = []
        for an in pick:
            ep, pr_rows = by_ep[an], pre_rows[an]
            if not ep or not pr_rows:
                continue                       # animal absent from one arm
            pr = rate([pr_rows[i] for i in rng.integers(0, len(pr_rows), len(pr_rows))])
            if pr is None:
                continue
            ds += [(ep[i][1] / ep[i][2]) - pr for i in rng.integers(0, len(ep), len(ep)) if ep[i][2]]
        if ds:
            diffs.append(float(np.mean(ds)))
    if len(diffs) < n_boot // 4:
        return None
    return point, np.asarray(diffs, float)


def fig_behaviour(out_dir):
    """1b pooled: hit rate per spout position, per epoch, weighted by session."""
    from wfield_local.grant_figures import _position_metrics, _sessions, CONF_LABELS

    by = ef.behaviour_by_epoch(_position_metrics, lambda an: _sessions(an), positions=CONF_LABELS)
    if not any(by.values()):
        return None
    short = dict(zip(CONF_LABELS, _short_labels()))
    # WILSON OVER POOLED TRIALS IS THE WRONG INTERVAL and was the only one any of these figures
    # had. It answers "how precisely do we know this pooled rate" while treating twenty thousand
    # trials from four mice as independent observations -- far too narrow, and narrowest exactly
    # where the clustering is worst. `behaviour_by_epoch` still returns it; the bootstrap below
    # replaces it wherever the per-trial tables allow, and only the point estimate is kept from it.
    values = {e: {short[p]: (v[0] if isinstance(v, (tuple, list)) else v)
                  for p, v in per.items()} for e, per in by.items() if per}

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
    # the bar intervals, from the same resampling as the marks, wherever the trial tables exist
    if blocks_recs:
        for e in list(values):
            for p in CONF_LABELS:
                key = short[p]
                if key not in values[e]:
                    continue
                got = ef.value_draws(
                    blocks_recs, e, lambda y, pr, c=int(p_idx[p]): _accuracy_at(y, pr, c),
                    rng=np.random.default_rng(_seed_for("behaviour", "value", e, p)),
                    n_boot=N_BOOT)
                ci = ef.with_ci(got)
                if ci is not None:
                    values[e][key] = ci
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


def _gap_at(y, p, code):
    """Refit-minus-frozen accuracy at one true position CODE, from a PAIRED record.

    ``p`` is the (n, 2) array `grant_figures._collect_5c(mode="paired")` returns -- column 0 the
    frozen pre-stroke decoder, column 1 the within-session refit -- so the difference is taken on
    the SAME trials and survives every level of the block bootstrap paired.

    Trials whose class the session could not TRAIN on are dropped from both arms together (the
    refit column carries `grant_figures.REFIT_UNAVAILABLE`), which keeps the pairing while refusing
    to score a decoder on a class it never saw. Without it the lick-aligned acute far-contralateral
    cell reads -0.42 -- the mouse not licking, presented as the code being gone.

    Positive = the session's own decoder reads a position the frozen one cannot: information
    PRESENT but unreadable by the pre-stroke model. Zero = both fail together, which is what a
    genuinely degraded code looks like.
    """
    from wfield_local.grant_figures import REFIT_UNAVAILABLE

    if code is None:
        return None
    y, p = np.asarray(y), np.asarray(p)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("_gap_at needs a paired record: use _collect_5c(..., mode='paired')")
    m = (y == code) & (p[:, 1] != REFIT_UNAVAILABLE)
    if m.sum() < 5:
        return None
    return float(np.mean(p[m, 1] == y[m]) - np.mean(p[m, 0] == y[m]))


def _gap_of(record, position):
    """`_gap_at` for one whole record, at one position NAME. `None` when the class is absent."""
    if record is None:
        return None
    return _gap_at(record[0], record[1], _code_of(position))


def _refit_at(y, p, code):
    """Accuracy of the REFIT column of a paired record at one true position code.

    Trials the session could not train on (`grant_figures.MIN_REFIT_CLASS`) are DROPPED, not scored
    as errors: the refit arm was never asked about them. Dropping them from both arms is what keeps
    this comparable to `_gap_at`, which drops the same trials.
    """
    from wfield_local.grant_figures import REFIT_UNAVAILABLE

    if code is None:
        return None
    y, p = np.asarray(y), np.asarray(p)
    m = (y == code) & (p[:, 1] != REFIT_UNAVAILABLE)
    if m.sum() < 5:
        return None
    return float(np.mean(p[m, 1] == y[m]))


def _refit_of(record, position):
    if record is None:
        return None
    return _refit_at(record[0], record[1], _code_of(position))


def _per_position_accuracy(per_animal, out_dir, disp, align, variant, wname):
    """Per-position accuracy of the frozen pre-stroke decoder, by epoch."""
    return _position_bars(
        per_animal, out_dir, align, variant, wname,
        name=f"epoch_acc_by_position_{align}_{variant}",
        title=f"Per-position decoding accuracy, {wname} -- pooled across animals",
        delta_name=f"epoch_accdelta_by_position_{align}_{variant}",
        delta_title=f"Change from pre-stroke in per-position accuracy, {wname}",
        ylabel="accuracy", delta_ylabel="accuracy - pre",
        stat_at=_accuracy_at, value_of=_accuracy_of,
        chance=CHANCE, ylim=(0.0, 1.10),
        notes=["pre panel is leave-one-session-out within each animal, pooled across animals"])


def _position_bars(per_animal, out_dir, align, variant, wname, *, name, title, delta_name,
                   delta_title, ylabel, delta_ylabel, stat_at, value_of, chance, ylim, notes,
                   reference=None):
    """ONE path for every per-position bar row built from decoder records.

    Frozen accuracy, refit accuracy and the paired frozen-minus-refit gap differ only in which
    statistic reduces a trial set, so the resampling, the marks, the Bonferroni correction across
    the twelve contrasts and the companion interval panel are written once. Three copies would
    agree today and diverge the first time one of them gained a correction -- the same argument
    `_scalar_figure` already makes for the scalar families.

    ``stat_at(y, p, code) -> float | None`` reduces a POOLED trial set at one position;
    ``value_of(record, position_name) -> float | None`` reduces ONE session for its dot.
    """
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
            # THE BAR'S OWN INTERVAL, from the same animals -> sessions -> blocks resampling the
            # marks use. Passing a bare number drew no error bar at all, silently: `bar_row` draws
            # one only where a value arrives as a (value, lo, hi) tuple.
            got = ef.value_draws(
                per_animal, e, lambda y, pr, c=_code_of(q): stat_at(y, pr, c),
                rng=np.random.default_rng(_seed_for(align, variant, e, f"{q}-value")),
                n_boot=N_BOOT)
            a = ef.with_ci(got) if got is not None else value_of(rec, q)
            if a is not None:
                values[e][short[q]] = a
            # one dot per session, from the SAME records the pooled bar sums
            pts = ef.per_session_values(per_animal, e, lambda _an, _d, r, q=q: value_of(r, q))
            points[e][short[q]] = [(an, v) for an, v in pts if v is not None]
    if not values:
        return None
    cov = ef.epoch_coverage(per_animal)
    per_epoch = dict(cov["per_epoch"])
    # A LIST OF PRE RECORDS IS A LIST OF SESSIONS. The frozen arm's pre entry is one
    # leave-one-session-out pool per animal and counts as one; the refit arm's is one record per
    # pre-stroke session, and counting that as one would print "pre 4" under a bar resting on 44.
    def _n_pre(pre):
        return 0 if pre is None else (len(pre) if isinstance(pre, list) else 1)

    per_epoch["pre"] = {an: _n_pre(per_animal.get(an, (None, {}))[0]) for an in per_animal}

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
                lambda y, pr, c=_code_of(q): stat_at(y, pr, c),
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
                        notes=list(notes))
    made = ef.bar_row(
        values, out_dir, name=name, title=title,
        subtitle=sub, counts=_totals(per_epoch), marks=marks, points=points,
        ylabel=ylabel, positions=_short_labels(), tick_labels=_minor(), groups=_groups(),
        chance=chance, ylim=ylim, reference=reference)
    if any(rows.values()):
        ef.contrast_panel(
            rows, out_dir, name=delta_name, title=delta_title,
            subtitle=sub, ylabel=delta_ylabel, positions=_short_labels(),
            tick_labels=_minor(), groups=_groups(), n_comparisons=n_comp)
    return made


def _frozen_vs_refit_matched(out_dir, align, variant, wname):
    """5rm: the same contrast with the two arms given the SAME AMOUNT of training data.

    Priya, 2026-09-10: "build the training set-matched version too". The unmatched pre bar is a
    training-set-size handicap -- ten pre-stroke sessions against four fifths of one -- so the raw
    gap cannot be read directly. Matching removes it, and what it exposes is that the baseline does
    not go to zero: it goes the OTHER WAY. At equal training-set size the refit arm is +0.090 BETTER
    pre-stroke (post-cue), because training within a session shares that session's own nuisance
    structure -- alignment, haemodynamics, arousal, the LocaNMF projection -- while the matched
    frozen model has to generalise across days.

    SO THE NO-LESION BASELINE IS BRACKETED RATHER THAN KNOWN: -0.073 unmatched (frozen has 10x the
    data) and +0.090 matched (frozen must cross sessions). Both bounds are real effects and neither
    is "the" answer, which is why both families are drawn and both are read as
    epoch-minus-pre. The headline survives either way -- far-contralateral has the largest
    lesion-attributable recoverable component, +0.196 unmatched and +0.158 matched.
    """
    return _frozen_vs_refit(out_dir, align, variant, wname, matched=True)


def _frozen_vs_refit(out_dir, align, variant, wname, *, matched=False):
    """5r: is the position code LOST, or present and MISREAD by the pre-stroke model?

    THE QUESTION THE FROZEN DECODER CANNOT ANSWER ALONE. A frozen decoder that fails post-stroke is
    consistent with two opposite readings -- the information is gone, or the information is there
    and the pre-stroke readout no longer points at it -- and every other figure in this set attacks
    that ambiguity indirectly (split-half reliability says the pattern is still repeatable;
    crossnobis says it moved; the confusions say where it went). This attacks it directly: refit a
    decoder WITHIN each session, on the same trials, with the same estimator and the same block
    grouping, and ask whether the position becomes decodable again.

        gap ~ 0, both low   -> the code is degraded. No model recovers it.
        gap > 0             -> the code is present and DISPLACED: only the frozen readout fails.

    THE PRE PANEL IS THE CONTROL AND IS NOT ZERO BY CONSTRUCTION. The frozen arm trains on ten
    pre-stroke sessions and the refit arm on one, so a gap exists at pre from training-set size
    alone, with no lesion involved. The post-stroke claim is the gap at that epoch MINUS the gap at
    pre, which is what the companion contrast panel plots -- reading the raw gap as the effect
    would charge the lesion for the handicap the design imposes.

    Trials are IDENTICAL between the two arms by construction (`mode="paired"` fills two columns of
    one record), so the difference is paired at the trial level and every bootstrap level inherits
    the pairing.
    """
    from wfield_local import grant_figures as G

    per_animal, _days = G._collect_5c(align, variant,
                                      "paired_matched" if matched else "paired")
    if not per_animal:
        return None
    key = "5rm" if matched else "5r"
    what = (" (training-set MATCHED)" if matched else "")
    note_arm = (["frozen arm trained on a size-matched random subset of pre-stroke BLOCKS",
                 "the pre gap is cross-session generalisation, not training-set size"]
                if matched else
                ["each session scored by a 5-fold block-CV decoder fitted on ITSELF",
                 "the pre gap is the training-set-size handicap, not an effect"])
    made = []
    p = _position_bars(
        per_animal, out_dir, align, variant, wname,
        name=f"epoch_{key}_refit_by_position_{align}_{variant}",
        title=f"Per-position accuracy of a WITHIN-SESSION refit decoder{what}, {wname}",
        delta_name=f"epoch_{key}delta_refit_by_position_{align}_{variant}",
        delta_title=f"Change from pre-stroke in refit decoding accuracy, {wname}",
        ylabel="accuracy (refit within session)", delta_ylabel="accuracy - pre",
        stat_at=_refit_at, value_of=_refit_of, chance=CHANCE, ylim=(0.0, 1.10),
        notes=note_arm + ["pre panel is one record per pre-stroke session, not a pooled LOSO "
                          "record"])
    if p:
        made.append(p)
    q = _position_bars(
        per_animal, out_dir, align, variant, wname,
        name=f"epoch_{key}gap_frozen_vs_refit_{align}_{variant}",
        title=f"Recoverable information: refit minus frozen accuracy{what}, {wname}",
        delta_name=f"epoch_{key}gapdelta_frozen_vs_refit_{align}_{variant}",
        delta_title=f"Change from pre-stroke in the refit-minus-frozen gap{what}, {wname}",
        ylabel="refit - frozen accuracy", delta_ylabel="gap - pre gap",
        stat_at=_gap_at, value_of=_gap_of,
        # A DIFFERENCE HAS NO CHANCE LEVEL and no natural range: zero is the reference and the
        # axis autoscales, because clipping a negative gap to a [0, 1] window would hide the
        # reading that matters most -- both decoders failing together.
        chance=None, reference=0.0, ylim=None,
        notes=["paired within trial: both arms score the SAME trials"] + note_arm[1:])
    if q:
        made.append(q)
    return made or None


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
    # WHICH POSITION DID IT MOVE TOWARD (Priya, 2026-09-09). Family 8 answers "did this position
    # move"; its rows are confounded by amplitude, because a pure gain change in P shifts P's
    # distance to every pre-stroke position equally and paints a uniform row. Row-centring removes
    # that term, so the off-diagonal contrast is substitution rather than gain. Needed because the
    # substitution claim otherwise rests on decoder confusions and best-match fraction, both of
    # which are LABEL-level -- they say which position the readout assigns, not which position the
    # pattern moved toward. Diverging: the scale is centred on zero, so RdBu_r not magma.
    ("8rc", "_matrices_crossnobis_rowcentred", "crossnobis distance, row-centred", "RdBu_r", None,
     "Crossnobis geometry, row-centred -- which position did it move TOWARD"),
    # WHERE THE BEST MATCH WENT (Priya, 2026-09-10): "could we add a version that plots, for each
    # spout position across epochs, where the best matches were -- so we could see if there is a
    # shift to one other position or if it's evenly distributed". Family 10b already reduces this
    # argmax to its diagonal; the whole matrix is the part that distinguishes SUBSTITUTION (mass on
    # one off-diagonal cell) from COLLAPSE (mass spread evenly). Fraction of sessions, so the scale
    # is a real 0-1 and fixing it is correct here where it is not for a distance.
    ("10c", "_matrices_best_match_destination", "fraction of sessions", "viridis", (0.0, 1.0),
     "Where each position's best pre-stroke match went"),
    # SCALE-FREE COMPANION TO 8rc (Priya, 2026-09-10). Row-centring removes amplitude from the row's
    # offset but leaves it multiplying the row's shape, so 8rc is comparable in DIRECTION across
    # epochs and not in MAGNITUDE. Dividing each row by its own SD makes it a z-profile and fixes
    # that, at the cost of discarding how far the position moved -- which 8 and 8rc still carry.
    ("8rz", "_matrices_crossnobis_rownorm", "row z-score", "RdBu_r", (-2.0, 2.0),
     "Crossnobis geometry, row-NORMALISED -- direction only, scale-free"),
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

    # THE DIAGONAL AS BARS, beside the matrix (Priya, 2026-08-28). The matrix carries the whole
    # structure -- where a position's pattern went, not only that it left -- and the diagonal
    # carries the headline: how much each position still resembles its own pre-stroke pattern.
    # One is not a summary of the other, which is why both are drawn: a diagonal that falls with a
    # flat off-diagonal is a code that faded, and the same diagonal with mass at a neighbour is a
    # code that MOVED. Same statistics as every other bar family here.
    from wfield_local.grant_figures import CONF_LABELS

    short = dict(zip(CONF_LABELS, _short_labels()))
    dpos = {short[q]: i for i, q in enumerate(CONF_LABELS)}

    def _diag(M, key):
        A = np.asarray(M, float)
        i = dpos[key]
        v = A[i, i] if i < A.shape[0] else np.nan
        return None if not np.isfinite(v) else float(v)

    dvals, dpoints = ef.scalar_by_epoch(mats, _diag, keys=_short_labels())
    if dvals:
        try:
            _scalar_figure(
                out_dir, name=f"epoch_{key}diag_{collector.strip('_')}_{align}_{variant}",
                title=f"{stem}: own-position value by epoch -- {wname}",
                ylabel=unit, keys=_short_labels(), values=dvals, points=dpoints,
                tick_labels=_minor(), groups=_groups(),
                # THE BARS INHERIT THE FAMILY'S SCALE DECISION. `MATRIX_FAMILIES` sets a fixed range
                # only where the quantity has one -- a correlation does, a crossnobis distance does
                # not -- and the matrix already honours that by passing vmin/vmax through as None.
                # The diagonal bars used to fall back to (-0.05, 1.10) instead, which is a
                # CORRELATION's bound applied to a distance: crossnobis values above 1.10 were drawn
                # clipped, bars and 95% intervals cut off at the axis (Priya, deck slides 590-592).
                # None here means autoscale, which is the same answer the matrix arrives at.
                ylim=(None if scale is None
                      else (min(-0.05, float(vmin)), float(vmax))),
                delta_name=f"epoch_{key}diagdelta_{collector.strip('_')}_{align}_{variant}",
                delta_title=f"{stem}: change from pre-stroke in the own-position value -- {wname}")
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {key}diag {align}/{variant}: {type(ex).__name__} {str(ex)[:120]}",
                  flush=True)
    return ef.matrix_row(
        pooled, out_dir, name=f"epoch_{key}_{collector.strip('_')}_{align}_{variant}",
        title=f"{stem} -- {wname}", labels=_short_labels(), cmap=cmap,
        vmin=vmin, vmax=vmax, unit=unit, coverage=cov, subtitle=sub, delta=True)




# --------------------------------------------------------- the per-day scalar families

def _scalar_figure(out_dir, *, name, title, ylabel, keys, values, points, tick_labels=None,
                   groups=None, chance=None, ylim=(0.0, 1.06), delta_name=None,
                   delta_title=None, delta_ylabel=None):
    """Bar row + marks + the epoch-minus-pre companion panel, for a per-session scalar family.

    ONE PATH FOR ALL OF THEM. 8g, 10, 10b and 11 differ only in which collector fills `values` and
    `points`, so the statistics, the marks and the companion panel are written once. Four copies
    would agree today and diverge the first time one of them gained a correction.
    """
    post = [e for e in ef.PANELS if e != "pre" and values.get(e)]
    if not post or "pre" not in values:
        # NO PRE, NO CONTRAST -- and say so, rather than draw bars with no marks and let a reader
        # assume the test was run and came back null.
        return ef.bar_row(values, out_dir, name=name, title=title,
                          subtitle=ef.stats_line(_session_counts(), notes=[
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
    marks, rows = {}, {}
    for e in post:
        marks[e], rows[e] = {}, {}
        for k in keys:
            if k not in values[e]:
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
    sub = ef.stats_line(_session_counts(), n_boot=N_BOOT, notes=[
        _MEAN_NOTE,
        "bootstrap: animals -> sessions. No block level: these values are one number per session, "
        "so the trial reduction already happened inside the collector"])
    made = ef.bar_row(values, out_dir, name=name, title=title, subtitle=sub, marks=marks,
                      ylabel=ylabel, positions=keys, tick_labels=tick_labels, groups=groups,
                      points=points, counts=_totals(_session_counts()), chance=chance, ylim=ylim)
    if any(rows.values()):
        ef.contrast_panel(rows, out_dir, name=delta_name or f"{name}_delta",
                          title=delta_title or f"Change from pre-stroke -- {title}",
                          subtitle=sub, ylabel=delta_ylabel or f"{ylabel} - pre",
                          positions=keys, tick_labels=tick_labels, groups=groups,
                          n_comparisons=n_comp)
    return made


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
    return _scalar_figure(
        out_dir, name=f"epoch_8g_geometry_by_position_{align}_{variant}",
        title=f"Per-position RDM row correlation with the pre-stroke geometry -- {wname}",
        ylabel="row correlation", keys=_short_labels(), values=values, points=points,
        tick_labels=_minor(), groups=_groups(), ylim=(-0.2, 1.10),
        delta_name=f"epoch_8gdelta_geometry_by_position_{align}_{variant}",
        delta_title=f"Change from pre-stroke in per-position row correlation -- {wname}")


def _fig_10(out_dir, align, variant, wname):
    """10 pooled: best-match accuracy and the correct position's rank, by epoch.

    BUILT ON `_matrices_pattern`, not `_match_tables`. Priya asked for a pre-stroke bar and
    `_match_tables` cannot give one: its per-day dict is post-stroke only, with pre held separately
    as a 6x6 of counts from which a mean rank cannot be recovered. `_matrices_pattern` is keyed
    "PRE"|day, so the baseline is a session like any other -- and it is the same source 10b reads,
    so the two figures cannot disagree about what a best match is.

    `_best_match` is `grant_figures`' own, used rather than reimplemented: it takes the WHOLE ROW,
    which is what separates "the code is gone" from "the code moved somewhere specific".
    """
    from wfield_local import grant_figures as G

    mats, _days = G._matrices_pattern(align, variant)
    if not mats:
        return None
    ACC, RANK = "best-match accuracy", "rank of correct position"

    def value_of(M, key):
        best, rank = G._best_match(np.asarray(M, float))
        ok = np.isfinite(rank)
        if not ok.any():
            return None
        if key == ACC:
            return float(np.mean(best[ok] == np.flatnonzero(ok)))
        return float(np.nanmean(rank))

    values, points = ef.scalar_by_epoch(mats, value_of, keys=[ACC, RANK])
    if not values:
        return None
    made = None
    for key, ylim, chance in ((ACC, (0.0, 1.10), 1.0 / 6.0), (RANK, (0.0, 6.4), None)):
        v = {e: {key: d[key]} for e, d in values.items() if key in d}
        pt = {e: {key: points[e][key]} for e in v}
        if not v:
            continue
        got = _scalar_figure(
            out_dir, name=f"epoch_10_best_match_{'acc' if key == ACC else 'rank'}_{align}_{variant}",
            title=f"Best match against the pre-stroke patterns -- {wname}",
            ylabel=key, keys=[key], values=v, points=pt, chance=chance, ylim=ylim,
            delta_name=(f"epoch_10delta_best_match_"
                        f"{'acc' if key == ACC else 'rank'}_{align}_{variant}"),
            delta_title=f"Change from pre-stroke in {key} -- {wname}")
        made = made or got
    return made


def _fig_10b(out_dir, align, variant, wname):
    """10b pooled: is the best-matching pre-stroke pattern the CORRECT position, per position?

    Read off `_matrices_pattern` -- ``{animal: {"PRE"|day: M}}`` where ``M[i, j]`` is the
    correlation of this session's pattern at position i with the pre-stroke pattern at j. The
    session scores 1 at position i when j = i is the argmax of that row.

    A ROW OF ALL-NaN SCORES NOTHING rather than scoring zero: a position gated out for too few
    trials has no best match, and counting that as "matched the wrong position" would turn missing
    data into evidence of reorganisation, which is the direction that would flatter the result.
    """
    from wfield_local import grant_figures as G
    from wfield_local.grant_figures import CONF_LABELS

    mats, _days = G._matrices_pattern(align, variant)
    if not mats:
        return None
    short = dict(zip(CONF_LABELS, _short_labels()))
    idx = {short[q]: i for i, q in enumerate(CONF_LABELS)}

    def value_of(M, key):
        i = idx[key]
        row = np.asarray(M, float)[i]
        if not np.isfinite(row).any():
            return None
        return float(np.nanargmax(row) == i)

    values, points = ef.scalar_by_epoch(mats, value_of, keys=_short_labels())
    if not values:
        return None
    return _scalar_figure(
        out_dir, name=f"epoch_10b_best_match_by_position_{align}_{variant}",
        title=f"Fraction of sessions whose best pre-stroke match is the correct position -- {wname}",
        ylabel="fraction of sessions", keys=_short_labels(), values=values, points=points,
        tick_labels=_minor(), groups=_groups(), chance=1.0 / len(CONF_LABELS), ylim=(0.0, 1.10),
        delta_name=f"epoch_10bdelta_best_match_by_position_{align}_{variant}",
        delta_title=f"Change from pre-stroke in best-match fraction -- {wname}")


def _fig_11(out_dir, align, variant, wname):
    """11 pooled: the encoder's explained variance and gain, by epoch.

    `_enc_tables` is keyed "PRE"|day, so the pre-stroke baseline is a bar here without any special
    handling -- unlike figure 10, whose collector holds pre separately and had to be rebuilt on
    `_matrices_pattern` to get one.
    """
    from wfield_local import grant_figures as G

    tab, _days = G._enc_tables(align, variant)
    if not tab:
        return None
    # THREE THINGS WERE CALLED "gain" IN THIS FAMILY and Priya, reasonably, could not tell them
    # apart: the explained variance AFTER one best rescale (this figure's second bar), the fitted
    # amplitude factor `a` (the 11amp figure), and the quantity 11pos calls "shape r2" -- which is
    # the SAME number as this figure's second bar, per position. Renamed to say what each is.
    LABELS = ["frozen EV", "EV after rescale"]
    take = {"frozen EV": 0, "EV after rescale": 2}

    def value_of(payload, key):
        try:
            v = payload[take[key]]
        except Exception:                                              # noqa: BLE001
            return None
        return None if v is None else float(v)

    values, points = ef.scalar_by_epoch(tab, value_of, keys=LABELS)
    if not values:
        return None
    made = _scalar_figure(
        out_dir, name=f"epoch_11_encoder_gain_shape_{align}_{variant}",
        title=(f"Encoder explained variance, before and after ONE best rescale -- {wname}"),
        ylabel="value", keys=LABELS, values=values, points=points, ylim=(0.0, 1.30),
        delta_name=f"epoch_11delta_encoder_gain_shape_{align}_{variant}",
        delta_title=f"Change from pre-stroke in encoder explained variance -- {wname}")

    # THE FITTED GAIN ITSELF, H11's middle panel, which the epoch version had been dropping.
    # `_enc_terms` returns (raw, a, gain, per) and this figure took only 0 and 2 -- transfer before
    # rescaling and after -- so `a`, the amplitude term the whole decomposition exists to isolate,
    # was computed nightly and never shown by epoch.
    #
    # ITS OWN FIGURE, NOT A THIRD BAR BESIDE raw AND gain. Those two are variance-explained on
    # [0, 1]; `a` is a RATIO around 1.0 and unbounded above. Sharing one axis would either squash
    # the r2 pair into the bottom third or clip `a` -- the same category error as the crossnobis
    # bars fixed earlier today, where a bound belonging to one quantity was applied to another.
    #
    # 1.0 IS THE MEANINGFUL VALUE, not zero: a = 1 means no amplitude change, a < 1 a weaker code,
    # a > 1 a stronger one. It is deliberately NOT passed as `chance=1.0` -- that would draw the
    # line but also append "(chance 1.00)" to the y-label, and a gain of 1 is not a chance level.
    def _amp(payload, key):
        try:
            v = payload[1]
        except Exception:                                              # noqa: BLE001
            return None
        return None if v is None or not np.isfinite(v) else float(v)

    avals, apoints = ef.scalar_by_epoch(tab, _amp, keys=["gain"])
    if avals:
        try:
            _scalar_figure(
                out_dir, name=f"epoch_11amp_encoder_amplitude_{align}_{variant}",
                title=(f"Fitted AMPLITUDE factor a by epoch (1.0 = no amplitude change; "
                       f"below 1 = smaller) -- {wname}"),
                ylabel="amplitude factor a", keys=["gain"], values=avals, points=apoints,
                ylim=None,
                delta_name=f"epoch_11ampdelta_encoder_amplitude_{align}_{variant}",
                delta_title=f"Change from pre-stroke in the fitted amplitude factor a -- {wname}")
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! 11amp {align}/{variant}: {type(ex).__name__} {str(ex)[:120]}", flush=True)

    # PER POSITION, the way the decoding families already are (Priya, 2026-09-07). `_enc_terms`
    # already returns `{position: shape r2 after the gain}` as its fourth term and this figure was
    # reading only terms 0 and 2, so the per-position tuning was computed every night and shown to
    # nobody.
    #
    # SHAPE r2 IS THE RIGHT PER-POSITION QUANTITY AND THE GAIN IS NOT. `_enc_terms` fits ONE gain
    # per session deliberately -- "a per-position gain would absorb the position-specific amplitude
    # loss that IS the deficit, and the decomposition would say nothing". Shape r2 asks the question
    # that survives that: with the session's amplitude change already divided out, is THIS position
    # still predicted by its pre-stroke pattern?
    #
    # AND IT IS THE PER-POSITION COMPLEMENT OF THE DECODING RECALL. Recall asks whether a position
    # stays DISCRIMINABLE from the other five; shape r2 asks whether its pattern is still the
    # pre-stroke one. A position can stay decodable on a changed pattern, so the two dissociating is
    # the measurement behind "decodable but re-geometried" -- read them side by side.
    short = dict(zip(G.CONF_LABELS, _short_labels()))

    def _shape_at(payload, key):
        try:
            per = payload[3] or {}
        except Exception:                                              # noqa: BLE001
            return None
        for q, sh in short.items():
            if sh == key:
                v = per.get(q)
                return None if v is None or not np.isfinite(v) else float(v)
        return None

    pvals, ppoints = ef.scalar_by_epoch(tab, _shape_at, keys=_short_labels())
    if pvals:
        try:
            _scalar_figure(
                out_dir, name=f"epoch_11pos_encoder_shape_{align}_{variant}",
                title=(f"Encoder EV after rescale (= shape r2), per position -- {wname}"),
                ylabel="EV after rescale", keys=_short_labels(),
                values=pvals, points=ppoints, tick_labels=_minor(), groups=_groups(),
                # r2 is bounded above by 1 and NOT below: a pattern unrelated to its reference goes
                # sharply negative, so a fixed floor would clip exactly the positions that lost
                # their tuning -- the ones the figure exists to find.
                ylim=None,
                delta_name=f"epoch_11posdelta_encoder_shape_{align}_{variant}",
                delta_title=f"Change from pre-stroke in per-position encoder shape r² -- {wname}")
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! 11pos {align}/{variant}: {type(ex).__name__} {str(ex)[:120]}", flush=True)
    return made


def _fig_9(out_dir, align, variant, wname):
    """9 pooled: the per-position change in pattern similarity, by epoch.

    THE ONE FAMILY THAT CARRIES ITS OWN BOOTSTRAP. Every other scalar family reads a collector and
    is reduced by the time it arrives here; `_delta_cis(..., full=True)` computes the day-minus-pre
    difference by resampling blocks within session, and returns a record per day holding the
    interval and the median for the mean diagonal and for each position separately.

    AND ITS VALUES ARE ALREADY DIFFERENCES, so there is no pre-stroke bar and no contrast against
    one -- pre minus pre is zero by construction, and drawing it would invite a reader to compare
    two real panels against a column of exact zeros. The question here is whether each epoch's
    change differs from ZERO, so the mark tests the value's own interval rather than a contrast.
    """
    from wfield_local import grant_figures as G
    from wfield_local.grant_figures import CONF_LABELS

    cis = G._delta_cis(align, variant, 10, G._mats_pattern, "9", full=True)
    if not cis:
        return None
    short = dict(zip(CONF_LABELS, _short_labels()))

    def value_of(rec, key):
        try:
            v = (rec.get("pos") or {}).get(_inv_short[key])
        except AttributeError:
            return None
        return None if not v else float(v[2])          # the median of the day's own draws

    _inv_short = {short[q]: q for q in CONF_LABELS}
    values, points = ef.scalar_by_epoch(cis, value_of, keys=_short_labels())
    if not values:
        return None

    # MARKS TEST THE VALUE AGAINST ZERO, not against a pre bar there is no room for.
    n_comp = sum(len(values[e]) for e in values)
    marks, rows = {}, {}
    for e in list(values):
        marks[e], rows[e] = {}, {}
        for k in list(values[e]):
            got = ef.scalar_value_draws(
                points, e, k, rng=np.random.default_rng(_seed_for("9", align, variant, f"{e}-{k}")),
                n_boot=N_BOOT)
            if got is None:
                continue
            point, draws = got
            values[e][k] = ef.with_ci(got)
            marks[e][k] = ef.contrast_marks(draws, n_comparisons=n_comp)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            a = 0.05 / max(1, n_comp)
            clo, chi = np.percentile(draws, [100 * a / 2, 100 * (1 - a / 2)])
            rows[e][k] = (point, float(lo), float(hi), float(clo), float(chi))

    sub = ef.stats_line(_session_counts(), n_boot=N_BOOT, notes=[
        _MEAN_NOTE,
        "values are ALREADY day-minus-pre differences, so there is no pre-stroke bar; the mark "
        "asks whether that epoch's change differs from zero"])
    made = ef.bar_row(
        values, out_dir, name=f"epoch_9_delta_trajectory_{align}_{variant}",
        title=f"Change in own-position pattern similarity from pre-stroke -- {wname}",
        subtitle=sub, marks=marks, ylabel="similarity - pre",
        positions=_short_labels(), tick_labels=_minor(), groups=_groups(), points=points,
        # AUTOSCALED, for the same reason as the crossnobis bars above: (-0.65, 0.35) was a window
        # fitted to the data as it stood when the figure was written, and a recovery trajectory that
        # moved outside it was drawn clipped rather than drawn larger. A change-from-baseline has no
        # natural range either.
        counts=_totals(_session_counts()), ylim=None)
    if any(rows.values()):
        ef.contrast_panel(
            rows, out_dir, name=f"epoch_9ci_delta_trajectory_{align}_{variant}",
            title=f"Change in own-position pattern similarity, with intervals -- {wname}",
            subtitle=sub, ylabel="similarity - pre", positions=_short_labels(),
            tick_labels=_minor(), groups=_groups(), n_comparisons=n_comp)
    return made


#: Stated on every already-reduced family, because it is the one thing that separates them from
#: the confusion figures: those pool by SUMMING raw counts (every trial once), these by AVERAGING
#: one value per session (every session once). Same figure shape, different weighting.
_MEAN_NOTE = ("pooled as a MEAN OVER SESSIONS: these values are already reduced per session, so a "
              "session is the unit and cannot be re-weighted by its trial count")

def _fig_11c(out_dir, align, variant, wname):
    """11c: the encoder's CEILING, and how much of its failure is the TEMPLATE being wrong.

    THE NUMBER THIS FIGURE EXISTS TO MAKE READABLE. Figure 11's `frozen EV` is an R^2, and acutely it
    is -0.388 post-cue: "worse than predicting the mean", and mute about what was achievable in that
    session. `EV after rescale` frees the AMPLITUDE only, and a large gap between the two is an
    amplitude story only when the rescaled value is HIGH -- a code that is simply gone also recovers
    a lot under rescaling. That ambiguity forced the withdrawal of the "half amplitude, half shape"
    reading on 2026-09-10.

    THREE BARS, ALL DIRECTLY MEASURED, all in the same units, all scoring the SAME half-session:

        ceiling          scored against the OTHER half of the same session -- how much position
                         structure the session has at all
        frozen (matched) scored against an equally sized draw from the pre-stroke pool -- the same
                         estimator and the same number of reference trials, differing only in WHICH
                         SESSIONS the reference came from
        frozen (all pre) scored against the whole pre-stroke pool: the encoder as it is actually
                         used elsewhere in this deck, and the only one of the three that is not
                         size-matched

    READ `ceiling - frozen (matched)`, NOT `ceiling - frozen (all pre)`. Only the first holds
    training-set size constant, so only the first isolates "the template came from other sessions".
    And read it against its own PRE value rather than against zero: at pre-stroke that difference is
    the cross-session generalisation cost with no lesion involved, and it is large -- 0.317 post-cue.

    WHY THE MATCHED ARM HAD TO BE ADDED. Without it the pre-cue window was uninterpretable: frozen EV
    0.330 against a ceiling of 0.090, a frozen arm beating its own ceiling, which is impossible for a
    real ceiling and was a fact about the construction rather than the data. Matching removes the
    asymmetry and the inversion goes with it, in all three windows.

    THE PRE-CUE WINDOW IS NOW SELF-CONSISTENT BUT STILL NEARLY SIGNAL-FREE: its ceiling is 0.117
    pre-stroke and hovers near zero afterwards, so the gap there is a ratio of two near-zero
    quantities. Consistent is not the same as informative.
    """
    from wfield_local import grant_figures as G

    raw_tab, _d1 = G._enc_tables(align, variant)
    cei_tab, _d2 = G._enc_ceiling_tables(align, variant)
    mat_tab, _d3 = G._enc_matched_tables(align, variant)
    if not cei_tab or not mat_tab:
        return None

    def _first(payload, _key):
        try:
            v = payload[0]
        except Exception:                                              # noqa: BLE001
            return None
        return None if v is None or not np.isfinite(v) else float(v)

    KEYS = ["ceiling", "frozen (matched)", "frozen (all pre)"]
    got = {}
    for key, tab in zip(KEYS, (cei_tab, mat_tab, raw_tab)):
        vals, pts = ef.scalar_by_epoch(tab, _first, keys=[key.split(" ")[0]])
        got[key] = (vals, pts, key.split(" ")[0])

    values, points = {}, {}
    for e in ef.PANELS:
        row, pt = {}, {}
        for key, (vals, pts, inner) in got.items():
            v = (vals.get(e) or {}).get(inner)
            if v is not None:
                row[key] = v
                pt[key] = (pts.get(e) or {}).get(inner, [])
        if row:
            values[e], points[e] = row, pt
    if not values:
        return None
    return _scalar_figure(
        out_dir, name=f"epoch_11c_encoder_ceiling_{align}_{variant}",
        title=f"Encoder ceiling vs the frozen template, training-set matched -- {wname}",
        ylabel="explained variance", keys=KEYS, values=values, points=points, ylim=None,
        # WRAPPED, because "frozen (matched)" and "frozen (all pre)" collide at this axis width and
        # the collision lands on the two bars a reader most needs to tell apart.
        tick_labels=["ceiling", "frozen\n(matched)", "frozen\n(all pre)"],
        delta_name=f"epoch_11cdelta_encoder_ceiling_{align}_{variant}",
        delta_title=f"Encoder ceiling and frozen template, change from pre-stroke -- {wname}")


SCALAR_FAMILIES = (("8g", _fig_8g), ("9", _fig_9), ("10", _fig_10),
                   ("10b", _fig_10b), ("11", _fig_11), ("11c", _fig_11c))


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
                    choices=("1b", "1c", "acc", "5c", "5r", "5rm", "mat", "scal"))
    args = ap.parse_args(argv)
    out = args.output or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ("1b", "1c", "acc", "5c", "5r", "5rm", "mat", "scal"))

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
    ARM_KEYS = {"acc", "5c", "5r", "5rm", "mat", "scal"}
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
        for _k, _fn in (("5r", _frozen_vs_refit), ("5rm", _frozen_vs_refit_matched)):
            if _k not in want:
                continue
            try:
                for q in (_fn(out, align, variant, wname) or []):
                    _report(f"{_k} {align}/{variant}", q)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! {_k} {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
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
