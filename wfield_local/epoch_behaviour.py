"""POOLED EPOCH FIGURES — BEHAVIOUR

Licking accuracy and its time course, pooled across animals and stratified by epoch.

Split out of `epoch_grant_figures` on 2026-09-21, one module per figure family. THE GROUPING IS
FROM THE CALL GRAPH: every function here is reached from this family's entry points and from no
other. Anything shared with a sibling is in `epoch_kit`, so this imports from there and never
from `epoch_grant_figures` -- that direction would be a cycle.

Entry points, dispatched by `epoch_grant_figures.main`:
  - `fig_behaviour`
  - `fig_behaviour_timecourse`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wfield_local import config, epochs
from wfield_local import epoch_figures as ef
from wfield_local.epoch_kit import (  # noqa: F401
    _MEAN_NOTE,
    N_BOOT,
    _accuracy_at,
    _groups,
    _long_labels,
    _minor,
    _pre_counts,
    _scalar_figure,
    _seed_for,
    _session_counts,
    _short_labels,
    _totals,
)


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
    from wfield_local.grant_figures import CONF_LABELS, _position_metrics, _sessions

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
def _position_codes():
    """``{position name: numeric code}`` -- the behaviour table's `pos_idx` domain."""
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    return {nm: int(code) for code, nm in POSITION_NAMES.items()}
