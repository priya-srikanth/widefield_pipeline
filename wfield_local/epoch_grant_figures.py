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
import warnings
from pathlib import Path

import numpy as np

from wfield_local import config, epoch_figures as ef, epochs
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

#: (display name, alignment, trial class). The lick window admits only lick trials -- a trial with
#: no detected lick has no lick to align to -- which is `grant_figures._variants`' rule, restated
#: here as data because these figures name the class in their titles.
#: (display key, alignment, trial class, caption). THE STOPPED ARMS ARE THE COMPLEMENT of the
#: `working` ones, not a subset: `flag_engagement`'s terminal quit period alone, which every other
#: arm removes (Priya, 2026-09-11: "the stopped class should basically be another frozen decoder /
#: encoder analysis set"). No lick-aligned stopped arm exists and none can -- a trial inside the
#: quit period is a non-response by construction, so there is no lick to align to.
ARMS = (("ENL", "precue", "working", "ENL (pre-cue), lick + miss-while-working"),
        ("cue", "cue", "working", "post-cue, lick + miss-while-working"),
        ("lick", "lick", "lick", "post-lick, lick trials only"),
        ("ENLstop", "precue", "stopped", "ENL (pre-cue), STOPPED trials only"),
        ("cuestop", "cue", "stopped", "post-cue, STOPPED trials only"))

CHANCE = 1.0 / 6.0


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
    from wfield_local import grant_figures as _G
    sub = ef.stats_line(_session_counts(_G.pre_session_counts(align, variant)),
                        blocks=ef.block_counts(per_animal), n_boot=N_BOOT, notes=list(notes))
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

    ONLY THE GAP IS DRAWN HERE. The refit-accuracy pair is 5r's and is identical under matching --
    see `_frozen_vs_refit`. And matching does not merely shrink the gap, it SHARPENS the
    dissociation; post-cue acute minus pre, per position:

                        nI       nM       nC       fI       fM       fC
        5r          +0.027  +0.132*  +0.168**  +0.034   +0.044  +0.196*
        5rm         +0.093   +0.039  +0.128*  -0.036  -0.132*  +0.158**

    Far-contralateral keeps a significant positive recoverable component once the handicap is
    removed, while FAR-MIDDLE turns significantly NEGATIVE: refitting buys less there than it bought
    before the lesion. That is "the code is degraded" as a positive finding rather than as an absent
    one, and it is invisible in the unmatched family, where far-middle is a flat +0.044.
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
    # THE MATCHED ARM DOES NOT DRAW THE REFIT-ACCURACY PAIR, because it cannot differ from 5r's.
    # `paired_matched` replaces the FROZEN model only; `_refit_pred` is called with the same X, y
    # and blocks, and `_seed_for` is keyed on (align, variant, epoch, position) rather than on the
    # figure name, so the bars, the intervals and the marks are identical by construction. Measured
    # 2026-09-11, post-cue acute-minus-pre, both families: -0.207* -0.141* -0.145* -0.339**
    # -0.363** -0.378** -- identical to three decimals AND in every mark. Drawing them twice put two
    # slides in the deck whose "(training-set MATCHED)" title promised a second analysis that did
    # not exist, and invited exactly the comparison this comment now forecloses. What matching CAN
    # move is the GAP, and it moves it a great deal -- see the second call below.
    if not matched:
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


def _fig_12_stopped(out_dir, align, variant, wname):
    """12: the trials the engagement gate THROWS AWAY -- does the position code survive quitting?

    Answer, measured: no, and it never did. This figure reports a NULL, which is why it sits at the
    end of the section rather than inside the argument.

    Priya, 2026-09-11: "do we already have a post-stroke vs pre-stroke 'stopped' trials pattern
    similarity analysis?" We did not. `flag_engagement` has only ever been a filter, and every
    figure in this deck is built on `~not_eng`; this is the complement.

    ``variant`` IS IGNORED AND MUST BE. The stopped set is defined by the gate, not by whether the
    animal licked -- a trial in the quit period is by construction a non-response -- so there is no
    lick/working distinction to make and drawing the same figure twice under two class labels would
    imply one. The figure renders once, on the `working` pass only.

    THREE PANELS PER EPOCH ROW, all correlated against the SAME pre-stroke ENGAGED template:

        PRE (stopped)   the CONTROL: how far stopping alone moves the pattern, with no lesion
        each epoch      the post-stroke stopped pattern against that same template

    THE CONTROL IS THE POINT. A post-stroke stopped pattern that no longer resembles the template
    is uninterpretable on its own, because a quitting animal is differently aroused, differently
    sated and differently postured whether or not it has a lesion. Only the difference between the
    two stopped columns is attributable to the lesion.

    AND THE CONTROL RESTS ON TWO ANIMALS. Post-cue, pre-stroke stopped trials number 6 (PS92), 40
    (PS93), 326 (PS94) and 495 (PS95); the first two are one session each and cannot carry a mean
    pattern. `_collect_stopped` returns None for them rather than a noisy one, and the subtitle
    names which animals the pre column rests on -- a four-animal-looking panel resting on two is
    the failure mode this whole section's per-animal counts exist to prevent.
    """
    if variant != "working":
        return None
    from wfield_local import grant_figures as G

    store, _days = G._collect_stopped(align)
    if not store:
        return None

    per_epoch, contributors = {}, {}
    pre_rows, pre_animals, pre_n, excluded = [], [], {}, []
    for an, rec in sorted(store.items()):
        ref = rec.get("WORK_REF")
        if not ref:
            continue
        n_tr, n_ss = rec.get("PRE_STOPPED_N", (0, 0))
        if rec.get("PRE_STOPPED"):
            pre_rows.append(G._corr_matrix(rec["PRE_STOPPED"], ref))
            pre_animals.append(an)
            pre_n[an] = n_ss
        else:
            excluded.append(f"{an} {n_tr}")
        for key, means in rec.items():
            if key in ("WORK_REF", "PRE_STOPPED", "PRE_STOPPED_N"):
                continue
            e = ef.epoch_of_day(an, int(key))
            if not e or e == "pre":
                continue
            per_epoch.setdefault(e, []).append(G._corr_matrix(means, ref))
            contributors.setdefault(e, {}).setdefault(an, 0)
            contributors[e][an] += 1

    mats = {}
    if pre_rows:
        mats["pre"] = G._nanmean_stack(pre_rows)
    for e, rows in per_epoch.items():
        mats[e] = G._nanmean_stack(rows)
    if len([m for m in mats.values() if m is not None]) < 2:
        return None

    cov = {e: dict(by) for e, by in contributors.items()}
    if pre_animals:
        # SESSIONS, not "1". The pre reference pools an animal's pre-stroke stopped trials into ONE
        # pattern, and reporting that as one contributing session -- which the first version did --
        # understates the baseline as badly as the stopped decoder arm's subtitle overstated it.
        cov["pre"] = dict(pre_n)
    n_pre = ", ".join(pre_animals) if pre_animals else "NONE"
    n_out = ("; excluded " + ", ".join(f"{x} trials" for x in excluded)) if excluded else ""

    # THE HEADLINE IS THE PRE PANEL, and it has to be stated or the delta row invites a reading it
    # cannot support. If the diagonal is at zero BEFORE the lesion then stopped trials carry no
    # recoverable position pattern at all, every post-stroke panel is a second near-zero number,
    # and their difference is a difference of noise however strongly the diverging colour map
    # renders it. Measured rather than asserted, so the sentence cannot outlive the result.
    _pre = mats.get("pre")
    _d = float(np.nanmean(np.diag(np.asarray(_pre, float)))) if _pre is not None else float("nan")
    verdict = ("READ THE PRE PANEL FIRST. Its diagonal is %.2f -- at zero before any lesion -- so "
               "stopped trials carry essentially NO position pattern in the first place, every "
               "post-stroke panel is a second near-zero number, and the change row below is a "
               "difference between two of them. Do not read structure into it. This is a NEGATIVE "
               "result about the quit period and it agrees with the stopped decoder arm, whose "
               "pre-stroke accuracy is 0.18-0.31 against a chance of 0.167."
               % _d) if np.isfinite(_d) and abs(_d) < 0.15 else ""
    return ef.matrix_row(
        mats, out_dir, name=f"epoch_12_stopped_pattern_{align}",
        title=(f"STOPPED trials: does the position pattern survive the animal quitting? -- "
               f"{wname}"),
        labels=_short_labels(), unit="pattern correlation", vmin=-1.0, vmax=1.0,
        coverage=cov, delta=True, annotate=False,
        subtitle=("Trials inside the terminal quit period ONLY -- the set every other figure in "
                  "this section removes. Every panel is correlated against the SAME reference: "
                  "that animal's pre-stroke ENGAGED mean pattern. The pre column is the CONTROL, "
                  "pre-stroke stopped trials against that template, and it measures how far "
                  "QUITTING ALONE moves the pattern with no lesion involved; only the difference "
                  f"between it and a post-stroke column is attributable to the lesion. It rests on "
                  f"{n_pre}{n_out} -- an animal needs four of the six positions and "
                  f"{G.MIN_STOPPED_REF} pre-stroke stopped trials to define the control, because a "
                  f"well-trained pre-stroke animal barely quits and a baseline built from one "
                  f"session is noise wearing the word 'control'. Cells with fewer than "
                  f"{G.MIN_STOPPED} stopped trials at that position in that session are absent, "
                  f"not zero. {verdict}"))


def _retained(acc, chance):
    """Fraction of the ABOVE-CHANCE range a score holds: ``(acc - chance) / (1 - chance)``.

    THE ONLY HONEST WAY TO PUT A SIX-WAY AND A THREE-WAY PROBLEM ON ONE AXIS. Position decoding has
    a chance of 1/6 and the behavioural-state decoder 1/3, so their raw accuracies are not
    comparable and their raw DROPS are not either: the same absolute fall means something different
    when the floor is 0.167 than when it is 0.333. Normalising asks the one question both can
    answer -- of the performance this readout had above chance, how much survived?
    """
    if acc is None:
        return None
    if np.ndim(acc):
        # A CALLER HANDED A CI TUPLE WHERE A POINT ESTIMATE BELONGS -- almost certainly because
        # `_scalar_figure` rewrote its `values` in place (see that function). Say which mistake it
        # is; the bare numpy error is "truth value of an array is ambiguous", which names neither
        # the variable nor the cause.
        raise TypeError(f"_retained needs a scalar accuracy, got {type(acc).__name__} of "
                        f"shape {np.shape(acc)} -- did _scalar_figure already overwrite it?")
    if not np.isfinite(acc):
        return None
    return float((acc - chance) / (1.0 - chance))


def _position_accuracy_by_epoch(align, variant):
    """``{epoch: BALANCED accuracy}`` of the FROZEN position decoder, from the 5c confusion counts.

    Read off the same matrices the 5c panel draws rather than recomputed, so the two figures can
    never disagree about the number this whole comparison hinges on.

    BALANCED -- the mean of the six row recalls -- and NOT the trial-weighted `trace / total`.
    The first version used the trial-weighted form and it made the comparison in 13n unfair in a
    way that was being disclosed as a caveat instead of fixed: the STATE arm is scored with
    `balanced_accuracy_score` because its class balance moves with epoch, so scoring position the
    other way put two different estimators on one axis. It also disagreed with the 5c panel's own
    printed accuracy (0.859 against the 0.89 on the figure), which is the row-recall mean.

    IT MATTERS HERE SPECIFICALLY because the post-stroke position sets are skewed by construction --
    PS93's are 49% far_center -- so a trial-weighted accuracy is pulled toward whichever positions
    the animal still attempts. The same reason `nolick_analysis` made balanced accuracy the headline
    on 2026-08-17.
    """
    from wfield_local import grant_figures as G

    per_animal, _d = G._collect_5c(align, variant, "frozen")
    if not per_animal:
        return {}
    out = {}
    for e, M in ef.counts_by_epoch(per_animal).items():
        if M is None:
            continue
        A = np.asarray(M, float)
        rows = A.sum(1)
        ok = rows > 0
        if ok.any():
            out[e] = float(np.mean(np.diag(A)[ok] / rows[ok]))
    return out


def _state_epoch_values(store, field, keys):
    """``(values, points)`` over epochs for one field of the state-decoder records.

    THE PRE COLUMN IS EVERY LEAVE-ONE-OUT RECORD, not one per animal: taking the first would throw
    away ten of each animal's eleven pre-stroke sessions and rest the baseline on four numbers.
    """
    vals, pts = {}, {}
    for e in ef.PANELS:
        row, pt = {}, {}
        for k in keys:
            got = []
            for an, rec in sorted(store.items()):
                src = (rec.get("PRE", []) if e == "pre" else
                       [r for day, r in rec.items()
                        if day != "PRE" and ef.epoch_of_day(an, int(day)) == e])
                for r in src:
                    v = (r.get(field) if field != "per_class"
                         else (r.get("per_class") or {}).get(k))
                    if v is not None and np.isfinite(v):
                        got.append((an, float(v)))
            if got:
                row[k] = float(np.mean([v for _a, v in got]))
                pt[k] = got
        if row:
            vals[e], pts[e] = row, pt
    return vals, pts


def _fig_12b_stopped_pooled(out_dir, align, variant, wname):
    """12b: do STOPPED trials still look like pre-stroke cortex? POOLED over positions.

    Priya, 2026-09-11, on why the per-position stopped arm is underpowered: "I more was thinking
    about mean pattern similarity or encoder similarity for ALL stopped trials (rather than per
    position)."

    THE POWER ARGUMENT, in numbers. The quit period is short by definition, and the per-position arm
    divides it six ways. Pooled across animals the post-cue stopped sets hold 867 trials pre-stroke,
    1,984 acute, 1,935 subacute and 359 chronic; split per position the chronic cell falls to 36-75
    trials, which is not a mean pattern. This uses ALL of a session's stopped trials for ONE
    measurement, and the question it asks -- does the cortical pattern during the quit period still
    resemble the pre-stroke one -- never needed the position axis.

    TWO BARS, and the second is what makes the first readable:

      similarity   corr(this session's pooled stopped pattern, that animal's pre-stroke ENGAGED
                   pooled pattern). Both stopped columns are scored against the SAME reference, so
                   the PRE-stroke stopped bar measures how far QUITTING ALONE moves the pattern with
                   no lesion involved, and only the difference between it and a post-stroke bar is
                   attributable to the lesion.
      reliability  the split-half correlation of the session's OWN stopped trials -- the ceiling
                   this measure can reach. A similarity of 0.4 against a ceiling of 0.45 and the
                   same 0.4 against a ceiling of 0.9 are opposite results, and the per-position
                   stopped arm had no ceiling at all.

    WHY IT IS NOT AN ENCODER. "Explained variance" in the encoder families is variance ACROSS
    POSITIONS -- `_enc_terms` centres a 6 x 380 matrix down the position axis and the denominator is
    the between-position sum of squares. Pool the positions away and that denominator is zero by
    construction, so an encoder EV has nothing left to explain. Correlation against a reference
    pattern is the measure that survives pooling, which is why this is the pattern-similarity family
    and not the encoder one.
    """
    if variant != "working":
        return None
    from wfield_local import grant_figures as G

    store, _days = G._collect_stopped_pooled(align)
    if not store:
        return None

    def _corr(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.size != b.size or not np.std(a) or not np.std(b):
            return None
        return float(np.corrcoef(a, b)[0, 1])

    VS_ENG = "vs pre-stroke ENGAGED"
    VS_STOP = "vs pre-stroke STOPPED"
    KEYS = [VS_STOP, VS_ENG]

    def _mean_of(by_sess, exclude=None):
        """Pooled pre-stroke stopped pattern, optionally leaving one session out."""
        v = [x for k, x in by_sess.items() if k != exclude]
        return np.mean(np.stack(v), axis=0) if v else None

    vals, pts, cov = {}, {}, {}
    for e in ef.PANELS:
        got = {k: [] for k in KEYS}
        for an, rec in sorted(store.items()):
            ref, by_sess = rec.get("REF"), (rec.get("PRE_BY_SESS") or {})
            if ref is None:
                continue
            if e == "pre":
                # LEAVE-ONE-SESSION-OUT on BOTH arms. A pre-stroke session scored against a pool
                # containing itself is scored partly against itself, and the whole pre bar -- which
                # is the control every post-stroke bar is read against -- would be too high.
                for mmdd, v in by_sess.items():
                    r = _corr(v, ref)
                    if r is not None:
                        got[VS_ENG].append((an, r))
                    other = _mean_of(by_sess, exclude=mmdd)
                    if other is not None:
                        r2 = _corr(v, other)
                        if r2 is not None:
                            got[VS_STOP].append((an, r2))
                continue
            full = _mean_of(by_sess)
            for day, v in rec.items():
                if day in ("REF", "PRE_BY_SESS", "PRE_STOPPED_N"):
                    continue
                if ef.epoch_of_day(an, int(day)) != e:
                    continue
                r = _corr(v, ref)
                if r is not None:
                    got[VS_ENG].append((an, r))
                if full is not None:
                    r2 = _corr(v, full)
                    if r2 is not None:
                        got[VS_STOP].append((an, r2))
        row = {k: float(np.mean([v for _a, v in got[k]])) for k in KEYS if got[k]}
        if row:
            vals[e] = row
            pts[e] = {k: got[k] for k in row}
            cov[e] = {a: sum(1 for x, _v in got[VS_ENG] if x == a)
                      for a in sorted({x for x, _v in got[VS_ENG]})}
    if len(vals) < 2:
        return None

    excl = [f"{an} {rec.get('PRE_STOPPED_N', 0)}" for an, rec in sorted(store.items())
            if not (rec.get("PRE_BY_SESS") or {})]
    has = sorted(an for an, rec in store.items() if (rec.get("PRE_BY_SESS") or {}))
    return _scalar_figure(
        out_dir, name=f"epoch_12b_stopped_pooled_similarity_{align}",
        title=("STOPPED trials, POOLED over positions: does the quit-period pattern still look "
               f"like pre-stroke cortex? -- {wname}"),
        ylabel="correlation with that pre-stroke reference",
        keys=KEYS, values=vals, points=pts,
        # SHORT, because two bar groups leave a narrow axis and "vs pre-stroke STOPPED" overran
        # into its neighbour. The full reference is named in the notes below, where there is room.
        tick_labels=["STOPPED", "ENGAGED"], ylim=(-0.4, 1.05),
        session_counts=cov,
        # TWO LINES. The method is in the docstring and the speaker note; a fourteen-line block
        # left the axes a sixth of the figure.
        notes=["a session's stopped trials pooled into ONE mean pattern, no position split. "
               "STOPPED = vs pre-stroke stopped (state-matched); ENGAGED = vs pre-stroke engaged, "
               "whose pre bar is the 'quitting alone' control. Pre is leave-one-session-out",
               "867 stopped trials pre, 1,984 acute, 1,935 subacute, 359 chronic"]
        + ([f"THE LEFT BAR RESTS ON {', '.join(has)} ONLY: "
            + ", ".join(f"{x} pre-stroke stopped trials" for x in excl)
            + " is too few to build a reference from, and a well-trained pre-stroke animal barely "
              "quits, so pooling cannot fix it"] if excl else []),
        delta_name=f"epoch_12bdelta_stopped_pooled_similarity_{align}",
        delta_title="Pooled stopped-trial similarity, change from pre-stroke")


def _long_of(q):
    """Position name -> the anatomical label the figures use ("far contra", not "far_R")."""
    from wfield_local.grant_figures import CONF_LABELS
    return dict(zip(CONF_LABELS, _long_labels()))[q]


def _fig_14pa_beta_maps_by_animal(out_dir, align, variant, wname):
    """14pa: the same maps PER ANIMAL, for the one position the deficit lives at.

    THE POOLED FIGURE AVERAGES FOUR ANIMALS AND THE PERMUTATION TEST RESTS ON FOUR, so neither can
    show whether a pattern REPLICATES -- and with n=4 replication across panels is stronger evidence
    than a p-value from a between-animal SE estimated on four numbers. Priya asked for these
    alongside the pooled view for exactly that reason.

    ONE POSITION PER FIGURE, far-contralateral by default: six positions x four animals x five
    epochs is 120 panels, which is a contact sheet rather than a figure. Far-contra is where the
    deficit is, and its neighbours are on the pooled panel.

    A ROW THAT LOOKS UNLIKE THE OTHER THREE IS THE POINT, not noise to be averaged away: PS94 and
    PS95 took 3 mW and show overt deficits while PS92 and PS93 were milder, so a severity-graded
    difference between rows is a finding and a random one is a warning.
    """
    if not ((variant == "working" and align in ("precue", "cue"))
            or (variant == "lick" and align == "lick")):
        return None
    from wfield_local import beta_maps as bm

    store, rel, ntr = bm.maps_by_epoch(align, variant)
    if not store:
        return None
    Q = "far_R"
    EPO = list(ef.PANELS)
    # ALL THREE POST-STROKE DELTAS, not just acute (Priya, 2026-09-12: "the pa map should have the
    # across-epoch deltas"). Acute-minus-pre alone shows the hit and hides the recovery, and
    # recovery is half of what the per-animal view is for -- a row that recovers and a row that
    # does not is exactly the between-animal difference this figure exists to expose.
    POST = ("acute", "subacute", "chronic")
    DCOLS = [f"{e} - pre" for e in POST]
    cells, titles = {}, {}
    for an, by_e in sorted(store.items()):
        per = {}
        for e in EPO:
            got = (by_e.get(e) or {}).get(Q)
            if not got:
                continue
            per[e] = np.mean(list(got.values()), axis=0)
            cells[(an, e)] = per[e]
            n_tr = sum((((ntr.get(an) or {}).get(e) or {}).get(Q) or {}).values())
            r = ((rel.get(an) or {}).get(e) or {}).get(Q)
            titles[(an, e)] = (f"{e}\n{len(got)} sess, n={n_tr}"
                               + (f"\nr={r:.2f}" if r is not None and np.isfinite(r) else ""))
        for e in POST:
            if "pre" in per and e in per:
                cells[(an, f"{e} - pre")] = per[e] - per["pre"]
                titles[(an, f"{e} - pre")] = f"{e.upper()} - PRE"
    if not cells:
        return None
    rows = [a for a in sorted(store) if any((a, e) in cells for e in EPO)]
    return ef.map_grid(
        cells, out_dir, name=f"epoch_14pa_beta_maps_by_animal_{align}_{variant}",
        title=(f"Far-CONTRALATERAL decoder map PER ANIMAL -- does the pattern replicate? "
               f"{wname}"),
        row_labels=rows, col_labels=EPO + DCOLS, panel_titles=titles, delta_cols=tuple(DCOLS),
        edges=bm.atlas_edges(),
        cbar_label=("cov(pixel, decoder output)\nred = MORE active on this position's\n"
                    "trials than on the average trial"),
        delta_label="change vs pre-stroke\n(orange-blue; expands if larger)",
        subtitle=("The pooled figure averages these four rows and the permutation test rests on "
                  "them, so neither can show REPLICATION -- which at n=4 is the stronger evidence. "
                  "Method identical to the pooled figure. Colour scale is per ANIMAL, so each row "
                  "is comparable across its own epochs and rows are not comparable to each other. "
                  "PS94 and PS95 took 3 mW and show overt deficits; PS92 and PS93 were milder, so a "
                  "severity-graded difference between rows is a finding and a random one is a "
                  "warning."))


#: How each reference is described on its own figure -- title, what a red pixel means, and the
#: sentence that says whether the six rows are independent. Kept as data rather than as branches
#: inside the renderer because the INDEPENDENCE claim is the one a reader must not have to infer.
_REF_TEXT = {
    "mean": dict(
        short="one vs REST",
        title="Position maps referenced to the MEAN OVER ALL TRIALS -- figure 14's reference, "
              "with no decoder in the path",
        cbar="activity minus the mean\nover ALL trials in the window",
        note=("THE SIX ROWS ARE NOT INDEPENDENT, and that is what this figure is for. The "
              "subtrahend is the mean over all trials, so a position that loses drive lowers the "
              "reference and hands every other position an increase it did not earn -- Priya, "
              "2026-09-12: \"in acute there may be less ss-ul/ll activity in far-center trials, "
              "which makes the near ipsi acute trial map look as though there is a relative "
              "*increase* in ss-ul/ll activity compared to pre-stroke.\" This is figure 14's "
              "reference rendered as a plain trial average, so the coupling can be seen without "
              "the decoder, the folds and the Haufe transform in the way. READ IT AGAINST THE "
              "QUIET-REFERENCED FIGURE: a rise that appears here and NOT there is the artefact."),
    ),
    "quiet": dict(
        short="one vs QUIET",
        title="Position maps referenced to the QUIET BASELINE -- one subtrahend per session, so "
              "the six positions are INDEPENDENT",
        cbar="activity minus that session's\nQUIET baseline (no running, no licking)",
        note=("THE SIX ROWS ARE INDEPENDENT. The subtrahend is one map per session -- the mean "
              "over that session's quiet frames, slow treadmill and no licking and buffered away "
              "from reward, as `quiet_periods` writes them -- and it is IDENTICAL for all six "
              "positions, so subtracting it cannot couple them. This answers \"is this "
              "position's cortex driven at all\", where the mean-referenced figure answers \"is "
              "it driven more than the others\". It is also the reference that does NOT sit "
              "inside the trial: figure 15's pre-cue baseline controls drift tightest but is "
              "blind to a sustained shift already present before the cue, and this one is not."),
    ),
}


def _fig_15rpa_reference_by_animal(out_dir, align, variant, wname):
    """15rpa: the quiet- and mean-referenced position maps PER ANIMAL, at far-contralateral.

    THE THIRD PER-ANIMAL PANEL, and it completes a set that now covers every map family: 14pa for
    the decoder maps, 15pa for the within-trial evoked maps, and this for the two reference maps.
    Each pooled figure averages four rows and each significance test estimates its spread from
    four animals, so none of them can show whether a pattern REPLICATES -- which at n=4 is the
    stronger evidence.

    BOTH REFERENCES, ONE PER FIGURE, so the pair can be read side by side at the animal level the
    way the pooled pair is read at the group level. That comparison is the whole point of the 15r
    family: same trials, same window, same floor, only the subtrahend differs, so a row that looks
    different between the two figures differs BECAUSE of the reference.

    WHAT TO LOOK FOR. Under the QUIET reference every animal's map should be broadly positive --
    task activity above rest -- and the epoch differences should be modest and consistent. Under
    the MEAN reference the common component is subtracted away, so what remains is
    position-specific and much noisier; a row that is dramatic there and flat under quiet is the
    one-vs-rest coupling showing itself in a single animal.
    """
    if not ((variant == "working" and align in ("precue", "cue"))
            or (variant == "lick" and align == "lick")):
        return None
    from wfield_local import beta_maps as bm
    from wfield_local import position_reference_maps as prm

    store, rel, ntr = prm.maps_by_epoch(align, variant)
    if not store:
        return None
    Q, EPO = "far_R", list(ef.PANELS)
    POST = ("acute", "subacute", "chronic")
    DCOLS = [f"{e} - pre" for e in POST]
    edges, out = bm.atlas_edges(), []

    for reference in prm.REFERENCES:
        txt = _REF_TEXT[reference]
        cells, titles = {}, {}
        for an, by_e in sorted(store.items()):
            per = {}
            for e in EPO:
                got = {k: d[reference] for k, d in ((by_e.get(e) or {}).get(Q) or {}).items()
                       if reference in d}
                if not got:
                    continue
                per[e] = np.mean(list(got.values()), axis=0)
                cells[(an, e)] = per[e]
                n_tr = sum((((ntr.get(an) or {}).get(e) or {}).get(Q) or {}).values())
                r = ((rel.get(an) or {}).get(e) or {}).get(Q, {}).get(reference)
                titles[(an, e)] = (f"{e}\n{len(got)} sess, n={n_tr}"
                                   + (f"\nr={r:.2f}" if r is not None and np.isfinite(r) else ""))
            for e in POST:
                if "pre" in per and e in per:
                    cells[(an, f"{e} - pre")] = per[e] - per["pre"]
                    titles[(an, f"{e} - pre")] = f"{e.upper()} - PRE"
        if not cells:
            continue
        rows = [a for a in sorted(store) if any((a, e) in cells for e in EPO)]
        out.append(ef.map_grid(
            cells, out_dir, name=f"epoch_15rpa_reference_{reference}_by_animal_{align}_{variant}",
            title=(f"Far-CONTRALATERAL, {txt['short']}, PER ANIMAL -- does it replicate? {wname}"),
            row_labels=rows, col_labels=EPO + DCOLS, panel_titles=titles,
            delta_cols=tuple(DCOLS), edges=edges,
            cbar_label=txt["cbar"], delta_label="change vs pre-stroke\n(same scale unless larger)",
            subtitle=(
                f"ONE OF A PAIR ({txt['short']}), at the animal level. The pooled 15r figures "
                f"average these rows and every significance test here estimates its spread from "
                f"four animals, so neither can show REPLICATION -- which at n=4 is the stronger "
                f"evidence. Same trials, window, class definition and 20-trial floor as figure 14; "
                f"only the subtrahend differs between this figure and its partner, so a row that "
                f"differs between them differs BECAUSE of the reference. NO significance is drawn "
                f"here: a per-animal panel has no between-animal spread to test, and a "
                f"within-animal test would be answering a different question from the pooled "
                f"figures. Colour scale is per ANIMAL, so each row is comparable across its own "
                f"epochs and rows are not comparable to each other. PS94 and PS95 took 3 mW and "
                f"show overt deficits; PS92 and PS93 were milder, so a severity-graded difference "
                f"between rows is a finding and a random one is a warning. {txt['note']}")))
    return [p for p in out if p]


def _fig_15r_reference_maps(out_dir, align, variant, wname):
    """15r: the SAME position maps under two different references -- the reference IS the claim.

    Priya, 2026-09-12: "does that average across all trials make sense? should we compare it to
    'quiet' instead?", then "maybe let's trial comparison the one vs rest vs one vs quiet".

    WHAT IS HELD FIXED AND WHAT IS VARIED. Trials, window, class definition and the 20-trial floor
    are figure 14's, exactly; the ONLY thing that changes between the two figures this returns is
    what gets subtracted. So a difference between them is the reference and cannot be anything
    else, which is the only way the comparison answers the question.

    NO DECODER ANYWHERE. These are trial averages. Figure 14 needs a fit, folds and class weights
    because it asks what DISTINGUISHES the positions; a reference question is about what happens on
    a position's trials, and a decoder in the path only adds a second thing that could explain a
    difference.

    THE THREE REFERENCES AND WHERE EACH LIVES:

        mean   here     minus the mean over ALL trials -- figure 14's, positions COUPLED
        quiet  here     minus the session's quiet baseline -- positions INDEPENDENT
        self   fig 15   minus that position's own pre-cue window -- positions INDEPENDENT

    Two independent references that disagree would localise the problem to what sits between them,
    which is the pre-cue window's own content: anticipation and locomotor state.
    """
    # SAME THREE ARMS AS FIGURE 14, which is what "the only difference is the reference" requires.
    # `stopped` is excluded for figure 14's reason: too few trials per position to fill six rows.
    if not ((variant == "working" and align in ("precue", "cue"))
            or (variant == "lick" and align == "lick")):
        return None
    from wfield_local import beta_maps as bm
    from wfield_local import position_reference_maps as prm
    from wfield_local.grant_figures import CONF_LABELS

    store, rel, ntr = prm.maps_by_epoch(align, variant)
    if not store:
        return None
    EPO = list(ef.PANELS)
    POST = ("acute", "subacute", "chronic")
    DCOLS = [f"{e} - pre" for e in POST]
    edges, out = bm.atlas_edges(), []

    for reference in prm.REFERENCES:
        txt = _REF_TEXT[reference]
        cells, titles, amp, contours, rows = {}, {}, {}, {}, []
        for q in CONF_LABELS:
            row, per = _long_of(q), {}
            for e in EPO:
                m, n_an, n_s = prm.pooled(store, reference, q, e)
                if m is None:
                    continue
                per[e] = m
                cells[(row, e)] = m
                n_tr = sum(sum((((ntr.get(an) or {}).get(e) or {}).get(q) or {}).values())
                           for an in store)
                rs = [((rel.get(an) or {}).get(e) or {}).get(q, {}).get(reference)
                      for an in store]
                rs = [r for r in rs if r is not None and np.isfinite(r)]
                titles[(row, e)] = (f"{e}\n{n_an} an, {n_s} sess, n={n_tr}"
                                    + (f"\nr={np.median(rs):.2f}" if rs else ""))
            if "pre" not in per:
                continue
            rows.append(row)
            base = float(np.sqrt(np.nanmean(per["pre"] ** 2)))
            if base > 0:
                amp[q] = {e: float(np.sqrt(np.nanmean(m ** 2))) / base for e, m in per.items()}
            pre_by = prm.by_animal(store, reference, q, "pre")
            for e in POST:
                if e not in per:
                    continue
                cells[(row, f"{e} - pre")] = per[e] - per["pre"]
                titles[(row, f"{e} - pre")] = f"{e.upper()} - PRE"
                post_by = prm.by_animal(store, reference, q, e)
                if not (pre_by and post_by):
                    continue
                try:
                    cm, lab = bm.significance_contour(pre_by, post_by)
                    print(f"  .. 15r {reference} {q} {e}: {lab}", flush=True)
                    if cm is not None and np.any(cm):
                        contours[(row, f"{e} - pre")] = cm
                except Exception as ex:                                # noqa: BLE001
                    print(f"  !! 15r sig {reference} {q} {e}: "
                          f"{type(ex).__name__} {str(ex)[:70]}", flush=True)
        if not cells:
            continue
        a_txt = ", ".join(f"{_long_of(q)} {amp[q].get('acute', float('nan')):.2f}"
                          for q in CONF_LABELS if q in amp)
        out.append(ef.map_grid(
            cells, out_dir, name=f"epoch_15r_reference_{reference}_{align}_{variant}",
            title=f"{txt['title']} -- {wname}",
            row_labels=rows, col_labels=EPO + DCOLS, panel_titles=titles,
            delta_cols=tuple(DCOLS), edges=edges, contours=contours,
            cbar_label=txt["cbar"], delta_label="change vs pre-stroke\n(SAME scale as the maps)",
            subtitle=(
                f"ONE OF A PAIR ({txt['short']}). Both figures use the SAME trials, the same "
                f"window, the same class definition and the same 20-trial floor as figure 14; the "
                f"only thing that differs is what is subtracted, so a difference between them is "
                f"the reference and nothing else. These are TRIAL AVERAGES -- no decoder, no "
                f"folds, no Haufe transform. {txt['note']} "
                "Colour scale is PER ROW, so a position is comparable across its own epochs and "
                "rows are not comparable to each other; the difference columns share their row's "
                "scale. THE THIN DARK OUTLINES ARE ALLEN CCF BOUNDARIES, not statistics. GREEN "
                "contours are bins significant under the animals-to-sessions bootstrap (2,000 "
                "draws, Bonferroni over 3,237 in-mask bins of an 8x downsampled grid) -- the same "
                "statistical object every bar family in this deck uses. "
                "r = split-half reliability of that epoch's mean map. "
                f"Acute amplitude relative to each position's own pre-stroke value: {a_txt}.")))
    return [p for p in out if p]


def _fig_15pa_evoked_maps_by_animal(out_dir, align, variant, wname):
    """15pa: figure 15's per-position evoked maps, PER ANIMAL, at the position that carries it.

    WHY THIS AND NOT ONLY 14pa. Figure 14's per-animal panels replicate beautifully and they
    replicate the wrong thing: once the `working` class was fixed, its acute far-contralateral map
    is near-uniformly negative in all four animals, which is the ENGAGEMENT collapse rather than a
    spatial code (see DECISIONS, 2026-09-12 late). The measure that stands is figure 15's --
    per-position `post-cue minus pre-cue`, a within-trial reference, positions INDEPENDENT -- so
    that is the one whose replication matters.

    FOUR ANIMALS IS THE CEILING ON THE PERMUTATION TEST, which is the other reason this exists. The
    pooled figure averages these rows and the test estimates its between-animal SE from four
    numbers; neither can show whether a pattern REPLICATES, and at n=4 replication across panels is
    the stronger evidence. A row that looks unlike the other three is a finding, not noise to be
    averaged away: PS94 and PS95 took 3 mW and show overt deficits while PS92 and PS93 were milder.

    ONE POSITION PER FIGURE. Six positions x four animals x five epochs is a contact sheet; this
    draws far-contralateral, where the deficit lives, and the pooled figure carries its neighbours.
    """
    if align != "cue" or variant != "working":
        return None
    from wfield_local import beta_maps as bm
    from wfield_local import position_evoked_maps as pem

    store, _counts = pem.maps_by_epoch()
    if not store:
        return None
    Q, EPO = "far_R", list(ef.PANELS)
    POST = ("acute", "subacute", "chronic")
    DCOLS = [f"{e} - pre" for e in POST]
    cells, titles = {}, {}
    for an, by_e in sorted(store.items()):
        per = {}
        for e in EPO:
            got = (by_e.get(e) or {}).get(Q)
            if not got:
                continue
            per[e] = np.mean(list(got.values()), axis=0)
            cells[(an, e)] = per[e]
            titles[(an, e)] = f"{e}\n{len(got)} sess"
        for e in POST:
            if "pre" in per and e in per:
                cells[(an, f"{e} - pre")] = per[e] - per["pre"]
                titles[(an, f"{e} - pre")] = f"{e.upper()} - PRE"
    if not cells:
        return None
    rows = [a for a in sorted(store) if any((a, e) in cells for e in EPO)]
    return ef.map_grid(
        cells, out_dir, name="epoch_15pa_evoked_maps_by_animal_cue",
        title="Far-CONTRALATERAL EVOKED map PER ANIMAL (post-cue minus pre-cue) -- does the "
              "INDEPENDENT measure replicate?",
        row_labels=rows, col_labels=EPO + DCOLS, panel_titles=titles, delta_cols=tuple(DCOLS),
        edges=bm.atlas_edges(),
        cbar_label="post-cue minus pre-cue\n(that position's OWN trials)",
        delta_label="change vs pre-stroke\n(SAME scale as the maps)",
        subtitle=(
            "THE PER-ANIMAL VIEW OF THE MEASURE THAT STANDS. Figure 14pa replicates too, but once "
            "the `working` class was fixed its acute far-contra map is near-uniformly negative in "
            "all four animals -- the ENGAGEMENT collapse, not a spatial code. Here each map is "
            "that position's OWN post-cue minus pre-cue, so the six positions are independent and "
            "a change cannot be inherited from another position's loss. Four animals is the "
            "ceiling on the permutation test and the pooled figure averages these rows, so "
            "neither can show REPLICATION -- which at n=4 is the stronger evidence. Colour scale "
            "is per ANIMAL: each row is comparable across its own epochs, rows are not comparable "
            "to each other. PS94 and PS95 took 3 mW and show overt deficits; PS92 and PS93 were "
            "milder, so a severity-graded difference between rows is a finding and a random one "
            "is a warning. Maps from `framemap_event_maps`; nothing recomputed."))


def _fig_15_evoked_maps(out_dir, align, variant, wname):
    """15: per-position EVOKED cortical maps -- the position-INDEPENDENT answer to "where".

    THE CONTROL THAT GATES FIGURE 14, not a complement to it. Priya, 2026-09-12: "in acute there may
    be less ss-ul/ll activity in far-center trials, which makes the near ipsi acute trial map look as
    though there is a relative *increase* in ss-ul/ll activity compared to pre-stroke." Figure 14's
    Haufe pattern is a covariance against the mean over ALL SIX positions, so one position losing
    drive lowers the reference and hands every other position an increase it did not earn. Four of
    its six rows cannot be read as written.

    HERE THE REFERENCE IS WITHIN TRIAL AND PER POSITION: each map is that position's own
    `post-cue mean - pre-cue mean`. Far-contralateral collapsing cannot leak into near-ipsilateral's
    map, because near-ipsilateral's map never looks at far-contralateral's trials.

    NOTHING IS RECOMPUTED. `framemap_event_maps` already writes these per session -- 123
    `*_spout_positions_1s_pre_post_delta_maps.npz` on the share, six positions x {pre, post, delta}
    as 540 x 640 Allen-aligned maps. This aggregates them by epoch.

    WHAT IT COSTS, so the two figures are not confused. The decoder pattern isolates what
    DISTINGUISHES positions but couples them; this keeps them independent but shows the WHOLE
    task-evoked response at that position -- cue, licking, movement, arousal -- not only the part
    carrying target identity. Disagreement is informative: a change here and not in figure 14 is a
    change in DRIVE that carries no position information; the reverse is a change in TUNING with no
    change in drive.

    ONE ALIGNMENT. The source maps are cue-referenced by construction (post-cue minus pre-cue), so
    there is nothing for a pre-cue or post-lick arm to be.
    """
    if align != "cue" or variant != "working":
        return None
    from wfield_local import beta_maps as bm
    from wfield_local import position_evoked_maps as pem
    from wfield_local.grant_figures import CONF_LABELS

    store, counts = pem.maps_by_epoch()
    if not store:
        return None
    EPO = list(ef.PANELS)
    cells, titles, amp, contours = {}, {}, {}, {}
    for q in CONF_LABELS:
        per = {}
        for e in EPO:
            # EACH ANIMAL'S OWN EPOCH MEAN FIRST, then the mean over animals -- so an animal with
            # more sessions cannot dominate, the rule every pooled family here uses.
            pa, ns = [], 0
            for an, by_e in store.items():
                got = (by_e.get(e) or {}).get(q)
                if got:
                    pa.append(np.mean(list(got.values()), axis=0))
                    ns += len(got)
            if pa:
                per[e] = np.mean(pa, axis=0)
                row = _long_of(q)
                cells[(row, e)] = per[e]
                titles[(row, e)] = f"{e}\n{len(pa)} an, {ns} sess"
        for e in ("acute", "subacute", "chronic"):
            if "pre" in per and e in per:
                cells[(_long_of(q), f"{e} - pre")] = per[e] - per["pre"]
                titles[(_long_of(q), f"{e} - pre")] = f"{e.upper()} - PRE"
        if "pre" in per:
            base = float(np.sqrt(np.nanmean(per["pre"] ** 2)))
            if base > 0:
                amp[q] = {e: float(np.sqrt(np.nanmean(m ** 2))) / base for e, m in per.items()}
            # THE PERMUTATION TEST IS BETTER POSED HERE THAN ON FIGURE 14, because these six maps
            # are INDEPENDENT: the null "this position's epoch label carries no information" is a
            # real null, where on a one-vs-rest map relabelling one position perturbs the reference
            # of the other five. Labels shuffled WITHIN animal; green contour = cluster mass above
            # the 95th percentile of the null.
            pre_by = {an: list(((by.get("pre") or {}).get(q) or {}).values())
                      for an, by in store.items()}
            pre_by = {a: v for a, v in pre_by.items() if v}
            for e in ("acute", "subacute", "chronic"):
                post_by = {an: list(((by.get(e) or {}).get(q) or {}).values())
                           for an, by in store.items()}
                post_by = {a: v for a, v in post_by.items() if v}
                if not (pre_by and post_by):
                    continue
                try:
                    cm, lab = bm.significance_contour(pre_by, post_by)
                    print(f"  .. 15e {q} {e}: {lab}", flush=True)
                    if cm is not None and np.any(cm):
                        contours[(_long_of(q), f"{e} - pre")] = cm
                except Exception as ex:                                # noqa: BLE001
                    print(f"  !! 15e sig {q} {e}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
    if not cells:
        return None
    rows = [_long_of(q) for q in CONF_LABELS if any((_long_of(q), e) in cells for e in EPO)]
    a_txt = ", ".join(f"{_long_of(q)} {amp[q].get('acute', float('nan')):.2f}"
                      for q in CONF_LABELS if q in amp)
    return ef.map_grid(
        cells, out_dir, name="epoch_15_evoked_maps_cue",
        title=("Per-position EVOKED maps: post-cue minus PRE-CUE, so the six positions are "
               "INDEPENDENT -- the control for figure 14"),
        row_labels=rows,
        col_labels=EPO + [f"{e} - pre" for e in ("acute", "subacute", "chronic")],
        panel_titles=titles,
        delta_cols=tuple(f"{e} - pre" for e in ("acute", "subacute", "chronic")),
        edges=bm.atlas_edges(), contours=contours,
        cbar_label="post-cue minus pre-cue\n(that position's OWN trials)",
        delta_label="change vs pre-stroke\n(SAME scale as the maps)",
        subtitle=(
            "READ THIS BEFORE FIGURE 14. Figure 14's decoder maps are one-vs-rest, centred on the "
            "mean over all six positions, so a position that loses drive lowers the reference and "
            "hands every other position an increase it did not earn -- four of its six rows cannot "
            "be read as written. Here each map is that position's OWN post-cue minus pre-cue, a "
            "WITHIN-TRIAL reference, so the positions are independent. "
            "THE COST: this shows the whole task-evoked response -- cue, licking, movement, "
            "arousal -- not only the part that carries target identity, which is what figure 14 "
            "isolates. A change visible here and absent there is a change in DRIVE without "
            "position information; the reverse is a change in TUNING without a change in drive. "
            "Maps from `framemap_event_maps`; nothing recomputed. Colour scale per ROW; the "
            "difference columns share their row's scale. THE THIN DARK OUTLINES ARE ALLEN CCF "
            "BOUNDARIES, not statistics. GREEN contours are bins where the change differs from "
            "zero under the SAME animals-to-sessions bootstrap the behaviour figures use (2,000 "
            "draws, resampled with replacement at both levels, Bonferroni over 3,237 in-mask bins "
            "of an 8x downsampled grid). It STEPS rather than curving because it is drawn on those "
            "bins: the maps carry no spatial detail finer than FWHM ~81 px, so a smooth "
            "full-resolution contour would claim a precision the data does not have. "
            f"Acute amplitude relative to each position's own pre-stroke value: {a_txt}."))


def _fig_14_beta_maps(out_dir, align, variant, wname):
    """14: WHERE the position code lives in cortex, and where it goes -- pooled decoder maps.

    Priya, 2026-09-12: "I want to start trying to answer *where* the displaced spout position codes
    move post-stroke", then "Could we do a beta weights mapping, as was done in this paper?"
    (Musall et al., Nat Neurosci 2022).

    EVERY OTHER FAMILY IN THIS SECTION ANSWERS "MOVED TOWARD WHAT", NOT "MOVED WHERE". Best-match
    destination collapses 380 features into one correlation per position pair and takes an argmax --
    a representational destination, not a location. Per component the joint basis gives 2-4
    components per Allen area, too thin to localise. This is the anatomical arm.

    METHOD, and the three departures from the paper are each measured -- see `wfield_local/beta_maps.py`:

      FEATURES   the temporal component matrix SVT, rank 100, NOT LocaNMF. `U @ A` is then a true
                 540 x 640 pixel map with no component-space intermediary, and it costs nothing:
                 SVT beats LocaNMF features by 0.06-0.09 balanced accuracy.
      PENALTY    L2, not the paper's L1. We are making a map, not selecting features, and L1's
                 choice among correlated predictors is free to change between days. Split-half of
                 the pre-stroke mean map: L1 0.688, L2 0.715, L2+Haufe 0.960.
      TRANSFORM  HAUFE, which the paper does not do and which is the single biggest factor above.
                 A decoder weight is a FILTER whose job includes cancelling correlated noise, so a
                 channel with NO signal can carry a large weight as a suppressor. `A = Cov(X) @ b`
                 makes it a PATTERN -- cov(channel, decoder output) -- which is the anatomical
                 question. Pattern and filter correlate at only r = 0.245 here.
      BALANCE    the lick arm only. `working` is uniform over positions by construction
                 (16.1-17.2% each) so pre-cue and post-cue need none; `lick` runs far-contra at
                 9.2-14.2% against ~19% near, and that skew IS the deficit.

    THE COLOUR SCALE IS PER ROW so each position is comparable across ITS OWN epochs, which is the
    comparison being made. Rows are not comparable to each other.

    `r` IN EACH PANEL IS THE SPLIT-HALF RELIABILITY of that epoch's mean map -- the ceiling any
    difference involving it can reach. IT IS NOT A CAVEAT TO DISCOUNT THE RESULT WITH: a split-half
    correlation of a near-absent signal is low BECAUSE the signal is near-absent. Far-contra acute
    is r = 0.53 AND 0.47 of its pre-stroke amplitude; those are one observation, not two. Far-middle
    falls to 0.48 amplitude while KEEPING r = 0.86, which is what shows the two are separable.

    THE RESULT, and it converges with the encoder from a completely different direction. Map
    amplitude relative to each position's own pre-stroke value:

        near ipsi 0.67   near middle 1.53   near contra 1.04
        far ipsi  1.09   far middle  0.48   far CONTRA  0.47      (acute)

    The two positions that lose more than half their map amplitude acutely are far-middle and
    far-CONTRA -- position-specific, in exactly the pair every other analysis implicates, and both
    recover by subacute (0.83 / 0.91). The encoder's fitted amplitude factor tells the same story in
    components rather than pixels: 0.286 acute against 0.749 pre-stroke, recovering to 0.745.
    """
    # THE THREE REAL ARMS: pre-cue and post-cue on `working`, and post-lick on `lick`. The two
    # `stopped` arms are excluded because these maps are fitted on the position label and the quit
    # period has too few trials per position to fit six classes -- that question is 12b's, pooled.
    if not ((variant == "working" and align in ("precue", "cue"))
            or (variant == "lick" and align == "lick")):
        return None
    from wfield_local import beta_maps as bm
    from wfield_local.grant_figures import CONF_LABELS

    store, rel, ntr = bm.maps_by_epoch(align, variant)
    if not store:
        return None

    EPO = [e for e in ef.PANELS]
    DELTA = "acute - pre"
    cells, titles, amp, contours = {}, {}, {}, {}
    for q in CONF_LABELS:
        per_epoch = {}
        for e in EPO:
            # EACH ANIMAL CONTRIBUTES ITS OWN EPOCH MEAN, then those are averaged -- so an animal
            # with more sessions cannot dominate the pooled map, the same rule the bar families use.
            per_animal, n_s, n_tr, rs = [], 0, 0, []
            for an, by_e in store.items():
                got = (by_e.get(e) or {}).get(q)
                if got:
                    per_animal.append(np.mean(list(got.values()), axis=0))
                    n_s += len(got)
                    n_tr += sum((((ntr.get(an) or {}).get(e) or {}).get(q) or {}).values())
                    r = ((rel.get(an) or {}).get(e) or {}).get(q)
                    if r is not None and np.isfinite(r):
                        rs.append(r)
            if per_animal:
                m = np.mean(per_animal, axis=0)
                per_epoch[e] = m
                row = _long_of(q)
                cells[(row, e)] = m
                rr = float(np.median(rs)) if rs else float("nan")
                # TRIALS, not just sessions. A cell built from 30 trials cannot be allowed to
                # look like one built from 521 (Priya: "can you include the n? the far R n may be
                # low"), and on the post-lick arm balancing makes exactly that happen acutely.
                titles[(row, e)] = (f"{e}\n{len(per_animal)} an, {n_s} sess, n={n_tr}"
                                    + (f"\nr={rr:.2f}" if np.isfinite(rr) else ""))
        if "pre" in per_epoch and "acute" in per_epoch:
            row = _long_of(q)
            cells[(row, DELTA)] = per_epoch["acute"] - per_epoch["pre"]
            titles[(row, DELTA)] = "ACUTE - PRE\ngreen = cluster p<0.05"
            # ONE DELTA COLUMN PER POST-STROKE EPOCH (Priya asked for all three): acute-minus-pre
            # alone shows the hit and not the recovery, and recovery is half this deck's claim.
            for _e in ("subacute", "chronic"):
                if _e in per_epoch:
                    cells[(row, f"{_e} - pre")] = per_epoch[_e] - per_epoch["pre"]
                    titles[(row, f"{_e} - pre")] = f"{_e.upper()} - PRE"
            # THE TEST, not the eye. 345,600 pixels makes an uncorrected threshold meaningless;
            # this shuffles epoch labels WITHIN animal and keeps clusters larger than 95% of those
            # obtainable by relabelling. Same statistic the panel draws.
            pre_by = {an: list(((by.get("pre") or {}).get(q) or {}).values())
                      for an, by in store.items()}
            post_by = {an: list(((by.get("acute") or {}).get(q) or {}).values())
                       for an, by in store.items()}
            pre_by = {a: v for a, v in pre_by.items() if v}
            post_by = {a: v for a, v in post_by.items() if v}
            try:
                cm, lab = bm.significance_contour(pre_by, post_by)
                print(f"  .. 14m {q} acute: {lab}", flush=True)
                if cm is not None and np.any(cm):
                    contours[(row, DELTA)] = cm
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 14m sig {q}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            base = float(np.sqrt(np.nanmean(per_epoch["pre"] ** 2)))
            if base > 0:
                amp[q] = {e: float(np.sqrt(np.nanmean(m ** 2))) / base
                          for e, m in per_epoch.items()}
    if not cells:
        return None

    rows = [_long_of(q) for q in CONF_LABELS if any((_long_of(q), e) in cells for e in EPO)]
    a_txt = ", ".join(f"{_long_of(q)} {amp[q].get('acute', float('nan')):.2f}"
                      for q in CONF_LABELS if q in amp)
    return ef.map_grid(
        cells, out_dir, name=f"epoch_14_beta_maps_{align}_{variant}",
        title=(f"WHERE the position code lives, and where it goes -- Haufe-transformed decoder "
               f"maps, {wname}"),
        row_labels=rows,
        col_labels=EPO + [DELTA, "subacute - pre", "chronic - pre"], panel_titles=titles,
        delta_cols=(DELTA, "subacute - pre", "chronic - pre"),
        edges=bm.atlas_edges(), contours=contours,
        cbar_label=("cov(pixel, decoder output)\nred = MORE active on this position's\n"
                    "trials than on the average trial"),
        delta_label="change vs pre-stroke\n(SAME scale as the maps)",
        subtitle=(
            "L2 logistic on the rank-100 SVT, beta Haufe-transformed to a PATTERN "
            "(A = Cov(X) beta) and rendered as U @ A -- full-resolution pixels, not components. "
            "Read the pattern for WHERE THE SIGNAL IS; the filter, which answers what the decoder "
            "USES, is a different map (they correlate at r = 0.245). "
            "Allen CCF boundaries overlaid; GREEN outlines on the difference columns are bins "
            "significant under the animals-to-sessions bootstrap the behaviour figures use (2,000 "
            "draws, Bonferroni over 3,237 bins of an 8x downsampled grid). "
            "Colour scale is PER ROW, so a position is comparable across its own epochs and rows "
            "are not comparable to each other. "
            "r = split-half reliability of that epoch's mean map, the ceiling a difference can "
            "reach -- and where a map is near-absent, low r IS the result rather than a reason to "
            "doubt it. "
            f"Acute map amplitude relative to each position's own pre-stroke value: {a_txt}."))


def _fig_13_state(out_dir, align, variant, wname):
    """13: DOES EVERYTHING DEGRADE, OR ONLY THE TARGET? The frozen behavioural-state decoder.

    Priya, 2026-09-11: "I'm more looking for evidence that not *all* decoding/encoding degrades
    post stroke, with running as an example."

    THE LESION IS VENTROLATERAL STRIATAL. No cortex is damaged anywhere in the field of view, the
    imaging window is the same one, the LocaNMF basis is the same. So if the frozen pre-stroke
    POSITION decoder collapses while a frozen pre-stroke BEHAVIOURAL-STATE decoder built from the
    identical features does not, the target deficit is SPECIFIC -- and the generic explanations a
    reader reaches for first (window clouding, haemodynamic drift, arousal, basis drift, "a lesion
    was made and everything got worse") all fail at once, because every one of them would degrade
    this readout too.

    IT HAD TO BE THE FROZEN ARM, which is a measurement and not a preference. Refit WITHIN a
    session, running-vs-quiet decodes at AUROC 0.99-1.00 and the three-way problem at macro-AUROC
    0.98-1.00. A ceiling cannot demonstrate preservation; a reader sees "the task was too easy to
    fail" and is right. The position claim rests on a frozen pre-stroke model failing on
    post-stroke data, so the control must be the same object carrying the same cross-session
    generalisation burden.

    METHOD IN FULL, because nothing else in this deck is built this way:

      UNIT      a ONE-SECOND window, not a trial. Trial-level labelling gives ~17 running trials
                per session, which decodes nothing; tiling the bouts gives 33,060 running and
                41,549 quiet one-second segments across the cohort. The length is set by QUIET and
                not chosen: quiet periods have a median of 1.10 s, so the 2 s window every other
                family here uses fits 17% of them while 1 s fits 58%.
      FEATURES  four 0.25 s sub-bins x 95 LocaNMF components = 380 columns, the SAME width as the
                trial-aligned arms and on the SAME joint basis. No per-segment baseline: a segment
                inside a running bout has no "before" that is not also running.
      CLASSES   quiet / running / licking, MUTUALLY EXCLUSIVE per Priya's rule -- running only if
                not also licking, licking only if not also running, quiet only inside a
                `behavior_events` quiet period (already buffered away from licks and rewards).
                Overlapping segments are DROPPED and counted, never assigned.
      LICKING   anchored at lick-bout ONSET and allowed to run past the bout end, because lick
                bouts have a median of 0.37 s and tiling strictly inside them would keep 22% of
                101,018 bouts and bias the class toward sustained licking. Running and quiet are
                tiled, at most 8 segments per period so no single long period dominates.
      MODEL     multinomial logistic on standardised features, frozen on ALL pre-stroke segments.
                The pre column is leave-one-session-out, as everywhere else here.
      SCORE     BALANCED accuracy against a chance of 1/3, never raw: quiet runs 3.4% of a
                pre-stroke session, 15.1% acutely and 0.7% chronically, and raw accuracy under a
                base rate that moves that much is not comparable across the epochs being compared.
      EXCLUDED  PS92 8/12, whose longest "running bout" is 2,441 s -- 41 minutes, 29% of the
                session, against a cohort maximum of 54 s. That is the crash+concat discontinuity
                (`docs/EXPERIMENT_ERRORS.md`) read as sustained locomotion.

    TWO CONFOUNDS CHECKED BEFORE THIS WAS DRAWN. Session TIME alone separates the classes at AUROC
    0.165-0.752, near chance, and restricting to the range where the classes overlap in time leaves
    the cortical score unchanged, so it is reading cortex rather than drift. And the animals RUN
    MORE after the lesion (5.1% of session acutely against 3.1% pre-stroke), so the state arm is
    not rescued by having more data at baseline than afterwards.

    THE LIMIT, stated here because it is easy to miss: licking windows are locked to a behavioural
    TRANSITION while running and quiet are sampled from inside sustained STATES, so a decoder could
    separate them partly on transient-versus-sustained rather than on which behaviour it is. This
    answers "does cortex still distinguish behavioural state at all", which is what the control
    needs; it is not a clean three-way contrast of matched epochs.
    """
    # ONE ALIGNMENT ONLY. These segments are not trials and have no cue to align to, so there is no
    # pre-cue/post-cue/post-lick distinction to make -- rendering the same figure under three arm
    # labels would imply three analyses where there is one.
    if variant != "working" or align != "cue":
        return None
    from wfield_local import locomotor_decoder as ld

    store, _days = ld.by_animal_day()
    if not store:
        return None

    vals, pts = _state_epoch_values(store, "balacc", ["state"])
    if not vals:
        return None
    # SNAPSHOTTED BEFORE `_scalar_figure` TOUCHES IT: that helper stores each bar's bootstrap
    # interval back into `values`, so these floats become (point, lo, hi) tuples the moment the
    # first figure is drawn. 13n needs the plain numbers.
    balacc = {e: float(row["state"]) for e, row in vals.items() if "state" in row}

    # THREE LINES ON THE CANVAS, NOT EIGHT. The full method is in this function's docstring and in
    # the deck speaker notes, which is where a reader who wants it goes; repeating it here wrapped
    # to thirteen lines and left the axes a fifth of the figure's height, so the one thing the
    # figure exists to show was the smallest thing on it. Keep what a reader CANNOT infer from the
    # title -- the unit, the class rule, and the fact that the model is frozen.
    NOTES = ["unit is a 1 s SEGMENT, not a trial: 4 x 0.25 s bins x 95 components, joint basis",
             "quiet / running / licking, mutually exclusive; frozen on ALL pre-stroke segments, "
             "pre column leave-one-session-out",
             "BALANCED accuracy: the class balance moves with epoch (quiet is 3.4% of a pre "
             "session, 15.1% acute, 0.7% chronic). Full method in the speaker notes"]
    counts = {e: {a: sum(1 for x, _v in pts[e]["state"] if x == a)
                  for a in sorted({x for x, _v in pts[e]["state"]})} for e in pts}
    made = []
    p = _scalar_figure(
        out_dir, name="epoch_13_state_decoder_cue",
        # NOT "preserved". The acute bar carries ** and chronic *, so the state decoder's fall IS
        # detectable -- it is 0.945 -> 0.875, small but not nothing. The claim this figure supports
        # is that it falls FAR LESS than position does, which is what 13n quantifies; a title
        # saying "preserved" would be contradicted by the marks on its own bars.
        title=("Does EVERYTHING degrade? Frozen pre-stroke BEHAVIOURAL-STATE decoder "
               "(quiet / running / licking), spout-position agnostic"),
        # `bar_row` APPENDS THE CHANCE LEVEL ITSELF, so naming it here too rendered
        # "balanced accuracy (chance 1/3) (chance 0.33)" down the side of the axes.
        ylabel="balanced accuracy", keys=["state"], values=vals, points=pts,
        tick_labels=["frozen state\ndecoder"], ylim=(0.0, 1.05), chance=1.0 / 3.0,
        notes=NOTES, session_counts=counts,
        delta_name="epoch_13delta_state_decoder_cue",
        delta_title="Frozen state decoder, change from pre-stroke")
    if p:
        made.append(p)

    pos = _position_accuracy_by_epoch(align, variant)
    if pos:
        cvals = {}
        for e in ef.PANELS:
            row = {}
            r_pos = _retained(pos.get(e), 1.0 / 6.0)
            if r_pos is not None:
                row["position (6-way)"] = r_pos
            if e in balacc:
                r_st = _retained(balacc[e], 1.0 / 3.0)
                if r_st is not None:
                    row["state (3-way)"] = r_st
            if row:
                cvals[e] = row
        if cvals:
            # DRAWN DIRECTLY, NOT THROUGH `_scalar_figure`, and this is not a shortcut. That
            # helper bootstraps every bar from its per-SESSION points and marks the contrasts; this
            # figure has no per-session points by construction -- each bar is ONE pooled number
            # derived from a pooled accuracy, and the position bar is not even per-session in
            # origin (it is the trace of a summed confusion matrix). Feeding it empty point lists
            # produced a numpy truth-value error, which was the right failure: the honest fix is
            # not to fake points but to draw the bars with NO marks and say in the subtitle that
            # there are none, rather than to emit error bars the data cannot support.
            q = ef.bar_row(
                cvals, out_dir, name="epoch_13n_state_vs_position_cue",
                title=("Of the performance each readout had ABOVE CHANCE, how much survived? "
                       "-- frozen decoders, post-cue"),
                ylabel="fraction of above-chance performance retained",
                positions=["position (6-way)", "state (3-way)"],
                tick_labels=["position\n(6-way)", "state\n(3-way)"], ylim=(0.0, 1.05),
                # THE COUNTS ARE THE STATE ARM'S, and they are the honest ones to print: the
                # position bar pools the same sessions minus PS92 8/12, which only this arm
                # excludes. Passing {} printed "N=0 animals, n=0 sessions" over a figure built
                # from eighty-nine.
                subtitle=ef.stats_line(counts, notes=[
                    "(accuracy - chance) / (1 - chance). Chance is 1/6 for the six-way position "
                    "decoder and 1/3 for the three-way state decoder, so raw accuracies -- and raw "
                    "DROPS -- are not comparable",
                    "BOTH BARS ARE BALANCED ACCURACY: mean of the row recalls for position, "
                    "sklearn balanced_accuracy_score for state. Both frozen on pre-stroke, pre "
                    "column leave-one-session-out",
                    "WHY NOT TRIAL-WEIGHTED for position (Priya asked): because THIS figure divides "
                    "by (1 - chance), and a trial-weighted chance level is not 1/6 -- it moves with "
                    "the trial mix, and the post-stroke mix is skewed by construction (PS93's "
                    "trials are 49% far_center, where always-guess-far_center scores 0.490). "
                    "Balanced accuracy has a null of exactly 1/6 however skewed either side is. The "
                    "trial-weighted accuracies are 0.859 / 0.525 / 0.749 / 0.833 -- they barely "
                    "differ here, so nothing in the reading turns on it; the 5c panel prints them",
                    "position 0.86 -> 0.43 acutely, losing 50%; state 0.92 -> 0.81, losing 11%; "
                    "RUNNING alone 0.98 -> 0.95, losing 3%",
                    "NO INTERVALS AND NO MARKS: each bar is one pooled number. The per-session "
                    "distribution is on the figures either side of this one"]))
            if q:
                made.append(q)

    # ---------------------------------------------------------------- confusion, per epoch
    # Priya, 2026-09-11: "can we add confusion matrices for the state decoder?" The per-class panel
    # gives the DIAGONAL -- how often each class is recalled -- and says nothing about where the
    # errors go, which is exactly the argument 5c makes for the position decoder. A quiet segment
    # misread as LICKING is a different failure from one misread as RUNNING: the first says the
    # post-stroke immobile animal looks task-engaged to the readout, the second that it looks like
    # it is moving.
    from wfield_local import locomotor_state as _ls

    conf = {}
    for e in ef.PANELS:
        acc = None
        for an, rec in sorted(store.items()):
            src = (rec.get("PRE", []) if e == "pre" else
                   [r for day, r in rec.items()
                    if day != "PRE" and ef.epoch_of_day(an, int(day)) == e])
            for r in src:
                M = r.get("confusion")
                if M is None:
                    continue
                acc = M.copy() if acc is None else acc + M
        if acc is not None and acc.sum():
            conf[e] = acc
    if len(conf) > 1:
        c = ef.confusion_row(
            conf, out_dir, name="epoch_13c_state_confusion_cue",
            title=("Frozen BEHAVIOURAL-STATE decoder, confusion by epoch -- "
                   "where do the errors go?"),
            coverage={e: dict(counts.get(e, {})) for e in conf}, delta=True,
            chance=1.0 / 3.0, labels=list(_ls.THREE_WAY))
        if c:
            made.append(c)

    CLS = ["quiet", "running", "licking"]
    pvals, ppts = _state_epoch_values(store, "per_class", CLS)
    if pvals:
        r = _scalar_figure(
            out_dir, name="epoch_13pos_state_decoder_by_class_cue",
            title="Frozen state decoder, RECALL PER CLASS -- which behavioural state changed?",
            ylabel="recall", keys=CLS, values=pvals, points=ppts, tick_labels=CLS,
            ylim=(0.0, 1.05), session_counts=counts, notes=[
                NOTES[0],
                "RUNNING IS THE CLEAN EXAMPLE: 0.98 / 0.95 / 0.94 / 0.97, flat at every epoch",
                "QUIET IS THE ONE THAT MOVES (0.86 -> 0.70 acutely) -- quiet goes from 3.4% of a "
                "pre-stroke session to 15.1% acutely, so a post-stroke animal sitting still may be "
                "in a genuinely different state. A finding about immobility, not a failed control"],
            delta_name="epoch_13posdelta_state_decoder_by_class_cue",
            delta_title="State decoder recall per class, change from pre-stroke")
        if r:
            made.append(r)
    return made or None


def _fig_10e_best_match_grid(out_dir, align, variant, wname):
    """10e: which pre-stroke position each position matches best -- PER ANIMAL, PER EPOCH.

    Priya, 2026-09-11, pointing at `grant_10_best_match`: "I want this best-match confusion matrices
    for epoch data". That grant figure gives each animal a PRE panel and one pooled POST panel; this
    is the same quantity cut by EPOCH instead, so a substitution can be watched appearing and
    resolving within an animal rather than only in the four-animal average that 10c draws.

    WHY PER ANIMAL MATTERS HERE SPECIFICALLY. The pooled 10c panel averages four animals whose
    epochs rest on very unequal session counts -- PS95 contributes one acute session and PS94 six --
    so a pooled off-diagonal cell can be one animal's whole story. The grid cannot hide that: each
    panel carries its own n, and a panel resting on one session looks like one session.

    FRACTIONS, NOT COUNTS, in every panel including pre. `grant_10` prints counts and has to warn in
    its own caption that the two panels' totals differ so only the percentage is comparable; here
    every cell is the fraction of that animal's sessions in that epoch whose best match was that
    column, which is the same construction everywhere and directly comparable across the grid.

    THE PRE PANEL IS A MEAN OF PER-SESSION ONE-HOTS, leave-one-session-out -- never the argmax of
    the averaged correlation matrix, which is a perfect identity for every animal and would draw a
    clean diagonal as the baseline. That error cost the best-match headline 0.02 when it was found
    on 2026-09-10; `_matrices_best_match_destination` carries the fix and this figure inherits it.
    """
    from wfield_local import grant_figures as G

    mats, _days = G._matrices_best_match_destination(align, variant)
    if not mats:
        return None
    store, _d2 = G._collect_7(align, variant, 10)

    grid, counts = {}, {}
    for an, by_key in mats.items():
        per_epoch, n_by = {}, {}
        if by_key.get("PRE") is not None:
            per_epoch["pre"] = np.asarray(by_key["PRE"], float)
            n_by["pre"] = len((store.get(an) or ({}, {}))[0])
        bucket = {}
        for key, M in by_key.items():
            if key == "PRE":
                continue
            e = ef.epoch_of_day(an, int(key))
            if e:
                bucket.setdefault(e, []).append(np.asarray(M, float))
        for e, stack in bucket.items():
            # nanmean over sessions: a row with no finite value in one session must not drag the
            # other sessions' vote toward zero, and `_matrices_best_match_destination` leaves such
            # a row NaN rather than one-hotting column 0. An ALL-NaN row across every session of an
            # epoch is legitimate (that position was never scorable) and numpy warns about it;
            # the warning is the expected case here, not a symptom.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_epoch[e] = np.nanmean(np.stack(stack), axis=0)
            n_by[e] = len(stack)
        if per_epoch:
            grid[an] = per_epoch
            counts[an] = n_by
    if not grid:
        return None

    made = []
    p = ef.matrix_grid_by_animal(
        grid, out_dir, name=f"epoch_10e_best_match_grid_{align}_{variant}",
        title=(f"Which PRE-STROKE position does each position match BEST, by animal and epoch "
               f"-- {wname}"),
        labels=_short_labels(),
        subtitle=("Cell = fraction of that animal's sessions in that epoch whose best pre-stroke "
                  "match was that column. Rows are the TRUE position, columns the pre-stroke "
                  "pattern matched. pre is leave-one-session-out and is NOT 100% -- read every "
                  "post-stroke panel against that animal's own pre panel, never against a perfect "
                  "diagonal. Argmax is invariant to a monotone change across a row, so the uniform "
                  "amplitude shifts that dominate the crossnobis families cannot move this."),
        unit="fraction of sessions", counts=counts, diag_label="self")
    if p:
        made.append(p)

    # THE DELTA GRID, as its own figure rather than as extra columns (Priya, 2026-09-11: "make sure
    # the best match matrices have delta matrices too"). `matrix_row` puts its deltas BENEATH, which
    # works because it has one row; this grid already spends its four rows on animals, and seven
    # columns at QUARTER_IN gives 0.6in panels that cannot carry the per-cell annotation the whole
    # figure is read through. Two figures at full size beats one at half.
    #
    # EACH ANIMAL AGAINST ITS OWN PRE, never against a pooled baseline or a perfect diagonal: the
    # leave-one-session-out pre panel is 92-100% here depending on the animal, and charging PS93 the
    # 8% its own baseline already misses would attribute it to the lesion.
    dgrid, dcounts = {}, {}
    for an, per_epoch in grid.items():
        base = per_epoch.get("pre")
        if base is None:
            continue
        d = {e: np.asarray(M, float) - np.asarray(base, float)
             for e, M in per_epoch.items() if e != "pre"}
        if d:
            dgrid[an] = d
            dcounts[an] = {e: n for e, n in counts[an].items() if e != "pre"}
    if dgrid:
        lim = max((float(np.nanmax(np.abs(M))) for by in dgrid.values() for M in by.values()
                   if np.isfinite(M).any()), default=1.0) or 1.0
        q = ef.matrix_grid_by_animal(
            dgrid, out_dir, name=f"epoch_10edelta_best_match_grid_{align}_{variant}",
            title=(f"Where each position's best match MOVED, by animal and epoch -- change from "
                   f"that animal's own pre-stroke -- {wname}"),
            labels=_short_labels(), cmap="RdBu_r", vmin=-lim, vmax=lim,
            subtitle=("Each panel is that epoch minus THAT ANIMAL's own leave-one-session-out pre "
                      "panel, so a perfect pre-stroke diagonal is not assumed and an animal whose "
                      "baseline already misses is not charged for it. BLUE on the diagonal = the "
                      "position stopped matching itself; RED off the diagonal in the SAME ROW names "
                      "where it went instead. A row that goes blue on the diagonal without any red "
                      "cell is a position that scattered rather than substituted."),
            unit="change in fraction of sessions", counts=dcounts,
            diag_label="self", diag_fmt="{:+.0%}")
        if q:
            made.append(q)
    return made or None


def _refit_confusion_rows(out_dir, align, variant, wname):
    """5cr: the WITHIN-SESSION REFIT decoder's confusion per epoch, and its change from pre.

    Priya, 2026-09-11: "can we add the refit confusion matrices to the deck". The 5r family reduces
    the refit decoder to a per-position ACCURACY -- the diagonal -- and the whole point of the
    frozen family's confusion panel is that the diagonal is not the interesting part: WHERE the
    errors go is. 5c answers that for the frozen decoder and nothing answered it for the refit one.

    WHAT THE PAIR SEPARATES, read against 5c panel for panel:

        5c off-diagonal, 5cr diagonal restored   the code is INTACT and the pre-stroke readout is
                                                 pointing at the wrong place -- displacement
        both off-diagonal, and in the SAME cells the code itself is confusable with that neighbour;
                                                 no readout recovers it -- degradation
        5cr off-diagonal in DIFFERENT cells      the within-session structure has reorganised rather
                                                 than simply weakened

    THE TWO PANELS ARE NOT ON THE SAME FOOTING and the difference is not cosmetic: the frozen arm
    trains on ten pre-stroke sessions and this one on four fifths of one, so its PRE panel is
    already worse than 5c's with no lesion involved. Read each family against ITS OWN pre column --
    which is what the delta row beneath does -- and never a 5cr cell against a 5c cell directly.

    SESSIONS THE REFIT COULD NOT BE FITTED ON ARE ABSENT, not zero: `_refit_pred` returns None for a
    session that cannot carry a 5-fold block split, so a thin epoch here rests on fewer sessions
    than the same epoch in 5c. The panel titles carry the per-animal session counts for that reason.
    """
    from wfield_local import grant_figures as G

    per_animal, _days = G._collect_5c(align, variant, "refit")
    if not per_animal:
        return []
    counts = ef.counts_by_epoch(per_animal)
    counts = {e: M for e, M in counts.items() if M is not None and np.asarray(M).sum()}
    if not counts:
        return []
    p = ef.confusion_row(
        counts, out_dir, name=f"epoch_5cr_refit_confusion_{align}_{variant}",
        title=(f"WITHIN-SESSION REFIT decoder, pooled across animals -- {wname}"),
        coverage=ef.epoch_coverage(per_animal)["per_epoch"], delta=True, chance=CHANCE,
        labels=_short_labels())
    return [p] if p else []




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
    the cross-session generalisation cost with no lesion involved, and it is large -- 0.276 post-cue.

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
        """THE AFTER-RESCALE SCORE, index 2 -- not the raw one. Priya: "what does negative variance
        explained mean though".

        It means the template's prediction is FURTHER from the data than predicting zero, i.e. than
        assuming no position tuning at all. And at these amplitudes that is almost entirely an
        amplitude statement: with PERFECT shape and only a scale mismatch,
        ``R2 = 1 - (1-a)^2 / a^2``, which is 0.000 at a = 0.5 and -2.13 at the a = 0.361 observed
        acutely. So a raw comparison between the ceiling and the matched arm is confounded: the
        ceiling's two halves have matched amplitude BY CONSTRUCTION (a = 0.75-0.89) while the matched
        frozen arm scores a shrunken post-stroke pattern against a full-amplitude pre-stroke template
        (a = 0.286 acutely in this arm, 0.361 in the unmatched one). The raw gap would be reporting
        amplitude and calling it template mismatch.

        The after-rescale score is amplitude-free on both sides and cannot go negative -- a = 0 is
        always available, which scores exactly 0 -- so this figure is about SHAPE and nothing else.
        The amplitude story is figure 11amp's, where it belongs.
        """
        try:
            v = payload[2]
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
        title=(f"Encoder SHAPE ceiling vs the frozen template, training-set matched "
               f"-- {wname}"),
        ylabel="EV after rescale (shape only)", keys=KEYS, values=values,
        points=points, ylim=(0.0, 1.05),
        # WRAPPED, because "frozen (matched)" and "frozen (all pre)" collide at this axis width and
        # the collision lands on the two bars a reader most needs to tell apart.
        tick_labels=["ceiling", "frozen\n(matched)", "frozen\n(all pre)"],
        delta_name=f"epoch_11cdelta_encoder_ceiling_{align}_{variant}",
        delta_title=f"Encoder ceiling and frozen template, change from pre-stroke -- {wname}")


def _fig_11cpos(out_dir, align, variant, wname):
    """11cpos: the encoder SHAPE ceiling and the frozen template, PER POSITION.

    11c pools the six positions into one number per epoch, which is the wrong shape for the question
    the lesion poses: far-contralateral is the position the deficit lives at, and a pooled ceiling
    averages it with five positions that barely moved. This is the same three quantities per
    position.

    TWO FIGURES. The first is the ceiling alone -- how much SHAPE each position's own trials can
    predict, which is a statement about that position's data and not about the template. The second
    is the CAPTURED FRACTION, `matched / ceiling`: of the shape a position's own trials can predict,
    how much does the pre-stroke template get? A fraction rather than a difference, because the
    positions do not share a ceiling and a raw gap of 0.2 means something different at a position
    whose ceiling is 0.3 than at one whose ceiling is 0.8.

    AFTER-RESCALE THROUGHOUT, for the reason `_fig_11c` gives at length: a raw score at these
    amplitudes is dominated by the encoder not being allowed to rescale, and the ceiling's halves
    have matched amplitude by construction while the frozen arm's reference does not.

    THE FRACTION IS CLIPPED FOR DISPLAY AND NOT FOR COMPUTATION. Where a ceiling is near zero the
    ratio is unstable and can exceed 1 or go negative; those cells are dropped rather than drawn,
    because a 300% "captured" bar is an artefact of a small denominator and reads as a result.
    """
    from wfield_local import grant_figures as G
    from wfield_local.grant_figures import CONF_LABELS

    cei_tab, _d1 = G._enc_ceiling_tables(align, variant)
    mat_tab, _d2 = G._enc_matched_tables(align, variant)
    if not cei_tab or not mat_tab:
        return None
    short = dict(zip(CONF_LABELS, _short_labels()))

    def _per(payload, key):
        try:
            per = payload[3] or {}
        except Exception:                                              # noqa: BLE001
            return None
        for q, sh in short.items():
            if sh == key:
                v = per.get(q)
                return None if v is None or not np.isfinite(v) else float(v)
        return None

    cvals, cpts = ef.scalar_by_epoch(cei_tab, _per, keys=_short_labels())
    mvals, _mp = ef.scalar_by_epoch(mat_tab, _per, keys=_short_labels())
    if not cvals:
        return None
    made = []
    p1 = _scalar_figure(
        out_dir, name=f"epoch_11cpos_encoder_ceiling_by_position_{align}_{variant}",
        title=f"Encoder SHAPE ceiling per position -- {wname}",
        ylabel="EV after rescale (shape)", keys=_short_labels(), values=cvals, points=cpts,
        tick_labels=_minor(), groups=_groups(), ylim=(0.0, 1.05),
        delta_name=f"epoch_11cposdelta_encoder_ceiling_by_position_{align}_{variant}",
        delta_title=f"Encoder shape ceiling per position, change from pre-stroke -- {wname}")
    if p1:
        made.append(p1)

    #: A per-session ceiling below this makes that session's captured FRACTION a ratio of noise.
    #: Applied PER SESSION, not to the pooled value: one session whose far-contra ceiling collapsed
    #: would otherwise drag a pooled ratio that the other fifteen support.
    MIN_CEILING = 0.10

    # THE FRACTION IS BUILT PER SESSION AND THEN POOLED, not computed from the pooled ceiling and
    # the pooled match. Priya, 2026-09-10: "why are there no sem or stats on the per position
    # fraction explained vs ceiling?" Because the first version divided one pooled number by
    # another, which has no distribution behind it and therefore no interval and no dots -- the one
    # bar family in this section that could not be argued with. A per-session ratio has both, and
    # goes through the same animals -> sessions bootstrap as every other bar here.
    #
    # A SYNTHETIC TABLE rather than a new collector: `scalar_by_epoch` wants one table whose payload
    # it can reduce, and the two arms are already keyed identically by (animal, PRE|day), so the
    # ratio can be formed session by session and handed back in the same shape. Slots 0-2 are NaN
    # because nothing reads them here; slot 3 is the per-position dict the accessor expects.
    frac_tab = {}
    for an, by_key in cei_tab.items():
        rec = {}
        for key, cpay in by_key.items():
            mpay = (mat_tab.get(an) or {}).get(key)
            if mpay is None:
                continue
            cper, mper = (cpay[3] or {}), (mpay[3] or {})
            row = {}
            for q in CONF_LABELS:
                c, m = cper.get(q), mper.get(q)
                if c is None or m is None or not np.isfinite(c) or not np.isfinite(m):
                    continue
                if c < MIN_CEILING:
                    continue
                row[q] = float(max(0.0, min(1.0, m / c)))
            if row:
                rec[key] = (np.nan, np.nan, np.nan, row)
        if len(rec) > 1:
            frac_tab[an] = rec

    if frac_tab:
        fvals, fpts = ef.scalar_by_epoch(frac_tab, _per, keys=_short_labels())
        if fvals:
            p2 = _scalar_figure(
                out_dir, name=f"epoch_11cfrac_encoder_captured_by_position_{align}_{variant}",
                title=(f"Of the shape a position's own trials can predict, how much does the "
                       f"PRE-STROKE template capture? -- {wname}"),
                ylabel="matched / ceiling", keys=_short_labels(), values=fvals, points=fpts,
                tick_labels=_minor(), groups=_groups(), ylim=(0.0, 1.10),
                delta_name=f"epoch_11cfracdelta_encoder_captured_by_position_{align}_{variant}",
                delta_title=f"Captured fraction per position, change from pre-stroke -- {wname}")
            if p2:
                made.append(p2)
    return made or None


SCALAR_FAMILIES = (("8g", _fig_8g), ("9", _fig_9), ("10", _fig_10),
                   ("10b", _fig_10b), ("11", _fig_11), ("11c", _fig_11c),
                   ("11cpos", _fig_11cpos))


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
    # `extend`, NOT the default `store`. With plain nargs="+" a REPEATED flag REPLACES the previous
    # value: `--only acc --only 13s` silently resolves to ["13s"] alone. That cost a 40-minute
    # render on 2026-09-11 -- five families were asked for, one ran, and the only symptom was a
    # figure still showing a number the code no longer produced. `extend` appends, which is what
    # repeating a flag reads as.
    ap.add_argument("--only", nargs="+", default=None, action="extend",
                    choices=("1b", "1c", "acc", "5c", "5cr", "5r", "5rm", "10e", "12s", "12b",
                             "13s", "14m", "15e", "15r", "mat", "scal"))
    args = ap.parse_args(argv)
    out = args.output or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ("1b", "1c", "acc", "5c", "5cr", "5r", "5rm", "10e", "12s", "12b", "13s", "14m", "15e",
                                 "15r", "mat", "scal"))
    # PRINTED, so "I asked for five families and one ran" is visible in the log rather than in a
    # stale figure three hours later.
    print(f"[epoch] rendering families: {' '.join(sorted(want))}", flush=True)

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
    ARM_KEYS = {"acc", "5c", "5cr", "5r", "5rm", "10e", "12s", "12b", "13s", "14m", "15e",
                "15r", "mat", "scal"}
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
        if "5cr" in want:
            try:
                for p in _refit_confusion_rows(out, align, variant, wname):
                    _report(f"5cr {align}/{variant}", p)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 5cr {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        # 12s IS DISPATCHED ON THE `working` ARMS ONLY, and the guard belongs here rather than
        # inside the figure. It draws the stopped trials of every arm against the ENGAGED template,
        # so it is one figure per ALIGNMENT, not one per (alignment, class) -- and returning None
        # from the other three arms made the renderer print "NO FIGURE" three times a render for a
        # case that is correct by construction. A warning that always fires is a warning nobody
        # reads.
        if "15r" in want:
            try:
                for p in (_fig_15r_reference_maps(out, align, variant, wname) or []):
                    _report(f"15r {align}/{variant}", p)
                for p in (_fig_15rpa_reference_by_animal(out, align, variant, wname) or []):
                    _report(f"15rpa {align}/{variant}", p)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 15r {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "15e" in want:
            try:
                _report(f"15e {align}/{variant}",
                        _fig_15_evoked_maps(out, align, variant, wname))
                _report(f"15pa {align}/{variant}",
                        _fig_15pa_evoked_maps_by_animal(out, align, variant, wname))
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 15e {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "14m" in want:
            try:
                _report(f"14m {align}/{variant}",
                        _fig_14_beta_maps(out, align, variant, wname))
                _report(f"14pa {align}/{variant}",
                        _fig_14pa_beta_maps_by_animal(out, align, variant, wname))
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 14m {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "13s" in want:
            try:
                for p in (_fig_13_state(out, align, variant, wname) or []):
                    _report(f"13s {align}/{variant}", p)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 13s {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        # ONE FIGURE PER ALIGNMENT. 12b pools a session's stopped trials whatever class the arm
        # names, so running it on the stopped arms too printed "NO FIGURE" twice a render for a
        # case that is correct by construction -- the same guard 12s needed.
        if "12b" in want and variant == "working":
            try:
                _report(f"12b {align}/{variant}",
                        _fig_12b_stopped_pooled(out, align, variant, wname))
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 12b {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "12s" in want and variant == "working":
            try:
                _report(f"12s {align}/{variant}",
                        _fig_12_stopped(out, align, variant, wname))
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 12s {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "10e" in want:
            try:
                for p in (_fig_10e_best_match_grid(out, align, variant, wname) or []):
                    _report(f"10e {align}/{variant}", p)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 10e {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
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
