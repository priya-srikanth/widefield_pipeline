"""One definition of the four things every per-session analysis in this repo re-implements.

WHY A MODULE, AND WHY NOW. Counted on 2026-09-20 across `scripts/rest_migration/` (74 modules,
19,208 lines): NINE copies of the nested animals->sessions bootstrap, SEVENTY repeats of the
curated-session filter, THIRTY-SEVEN direct lick loads, and a `fan_out`-plus-sort idiom written by
hand in each of the four modules that had been converted. `rest_by_position`'s docstring already
names the failure mode: *"The last time one quantity had two implementations in this repo -- the
flat map baseline against the encoder's time-local one -- they disagreed for months and the map
side was the wrong one."* And `parallel.py` exists because *"`grant_figures` grew a process pool
and the stages around it did not"*. The `_boot` copies were the same pattern, unresolved.

THEY WERE NOT YET DIVERGENT, WHICH IS THE ONLY REASON THIS IS A SAFE EXTRACTION. All seven copies
of the CI helper compute the point estimate as the mean of the FLAT pool and the interval from the
nested draw; all three copies of the paired-difference helper resample animals ONCE and reuse the
draw for both arms. (The 2026-09-21 engineering handoff asserted that `rest_coupling` differed on
the point estimate -- it does not, checked line by line before writing this. The real differences
were two empty-pool guards present in five copies and absent in two, and one copy returning the
animal count as a fourth element.) The copies always agree at first; this module is what keeps
that true.

WHAT IS DELIBERATELY NOT HERE. `wfield_local/matrix_bootstrap.py` and `epoch_figures` draw the
same animals-then-sessions resample but reduce to a MATRIX with NaN-aware means, and define the
point estimate per cell. Folding them in would mean one function with two reduction modes, which
is how a shared helper starts growing flags until the copies come back. They stay separate and
say so here so the next reader does not think they were missed.

ONE KNOWN WART, RECORDED RATHER THAN HIDDEN. `session_behavior` reaches UP into
`scripts/rest_migration/` for `session_trials`, `session_quit` and `near_codes` -- a library
importing from the scripts tree, which is backwards. It is done inside the function, so there is
no import cycle and nothing breaks, and it is not NEW coupling: the two worker functions this
replaces each imported those same three names across the tree already. Fixing it properly means
moving those three definitions down into `wfield_local`, which touches the modules that own them
and their callers, and that is a separate change with its own verification.

READ `DECISIONS.md` -- "FANNING THE ANALYSIS LOOPS OVER CORES, AND THE THREE THINGS IT CHANGED"
(2026-09-20) -- before changing anything below about ORDER. Half of this module exists to make an
ordering bug unrepresentable rather than merely documented.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

#: Draws per interval. 4000 in every rest_migration module that had its own copy, so adopting it
#: here changes no published number.
N_BOOT = 4000

#: Percentile bounds. A pair rather than an alpha, because every existing copy hard-coded
#: 2.5/97.5 and a derived version invites a caller to pass alpha=0.1 into a table whose other rows
#: are 95%.
PCTL = (2.5, 97.5)

#: Response window for scoring a trial as a hit, in seconds. 2.0 in `session_trials` call sites
#: throughout `rest_migration`.
RESP_S = 2.0


class Interval(NamedTuple):
    """A point estimate with a bootstrap interval, plus the n the interval was drawn over.

    FOUR FIELDS, NOT THREE. Most call sites index ``g[0] g[1] g[2]`` and one
    (`lick_bout_structure`) needed the animal count as ``g[3]``; a NamedTuple serves both without
    two return shapes. Do not unpack it positionally into exactly three names -- use the fields.
    """

    point: float
    lo: float
    hi: float
    n_animals: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the 95% interval clears zero -- the star in every table in this repo."""
        return (self.lo > 0.0) or (self.hi < 0.0)


# ---------------------------------------------------------------------------------------------
# 1. THE BOOTSTRAP
# ---------------------------------------------------------------------------------------------

def boot_ci(by_animal, rng, n_boot: int = N_BOOT):
    """Nested animals -> sessions bootstrap CI of the mean. ``Interval`` or None.

    ``by_animal`` is ``{animal: [value, ...]}``, one value per SESSION. There is no third level by
    construction: the per-session collector has already reduced over trials, so there are no
    trials left to resample.

    **THE POINT ESTIMATE IS THE MEAN OF THE FLAT POOL, NOT THE MEAN OF ANIMAL MEANS**, so an
    animal with fifteen sessions weighs more than one with three. That is what all seven copies
    did and what every published level in `DECISIONS.md` rests on -- but it is a CHOICE, and it is
    the one that produced a retracted result on 2026-09-20: a pooled-session mean of 4496 against
    4338 read as a change, where the paired within-animal difference said something else entirely.
    **If a quantity is a CHANGE, use `boot_delta`, which is animal-weighted by construction.
    `boot_ci` is for LEVELS.**

    THE DRAW ORDER IS LOAD-BEARING. Animals are sorted, then drawn; for each drawn animal its
    sessions are drawn. Reordering any of that under the same seed gives different numbers -- see
    the completion-order bug in `DECISIONS.md`, which moved a CI from [-22.9, -13.7] to
    [-23.1, -13.6] while leaving the point estimate exact. Any change here invalidates every
    interval in the deck. `tests/test_analysis_kit.py` pins the draw sequence against a literal
    copy of the pre-extraction code for this reason.
    """
    animals = sorted(by_animal)
    if not animals:
        return None
    # NO EARLY RETURN ON AN EMPTY POOL, DELIBERATELY. An all-empty `by_animal` still runs the loop
    # and returns None via the rescue check below, because that is how many draws the pre-
    # extraction code consumed. Callers share one `rng` across dozens of cells, so short-circuiting
    # here would not change THIS cell's answer -- it would shift every cell computed after it.
    flat = [v for a in animals for v in by_animal[a]]
    out = []
    for _ in range(n_boot):
        vals = []
        for a in (animals[i] for i in rng.integers(0, len(animals), len(animals))):
            sa = by_animal[a]
            vals += [sa[i] for i in rng.integers(0, len(sa), len(sa))]
        # GUARDED, WHICH TWO OF THE SEVEN COPIES WERE NOT. A draw is empty only if every animal it
        # picked has zero sessions; unguarded, `np.mean([])` returns NaN with a warning and one
        # NaN poisons every percentile. The unguarded copies were not wrong on today's data -- no
        # animal is empty -- they were one thin epoch away from silently returning NaN intervals.
        if vals:
            out.append(float(np.mean(vals)))
    if len(out) < n_boot // 4:
        return None
    o = np.asarray(out)
    return Interval(float(np.mean(flat)), float(np.percentile(o, PCTL[0])),
                    float(np.percentile(o, PCTL[1])), len(animals))


def boot_delta(post, pre, rng, n_boot: int = N_BOOT):
    """Paired animals -> sessions bootstrap of ``post - pre``. ``Interval`` or None.

    **THE ANIMALS ARE RESAMPLED ONCE AND THE SAME DRAW IS USED FOR BOTH ARMS**, which is what
    makes it paired: each animal is its own pre-stroke control, so between-animal variance -- the
    binding constraint at n=4 -- cancels instead of being counted twice.

    Only animals present in BOTH dicts contribute. An animal with no pre-stroke sessions cannot
    supply a change, and letting it into one side would compare different cohorts.

    **THE POINT ESTIMATE IS THE MEAN OVER ANIMALS OF EACH ANIMAL'S DIFFERENCE OF MEANS** --
    animals weighted equally, unlike `boot_ci`. The two conventions disagreeing is not an
    oversight: a LEVEL pooled over sessions describes the data collected, while a CHANGE has to be
    animal-weighted or one heavily-sampled animal decides it. PS95 supplies 7 of 15 pre-stroke
    sessions in one table, which is exactly how the 4496/4338 retraction happened.

    ONE INTERVAL CANNOT RESCUE A THIN PAIRING. With four animals, two of which may have a SINGLE
    pre-stroke session, the inner session draw has nothing to resample and the interval reflects
    the animal draw alone. `n_animals` comes back so a caller can say so, but **ground rule 8 --
    print the per-animal table BEFORE the bootstrap -- exists because that is not visible in the
    interval at all.**
    """
    animals = sorted(set(post) & set(pre))
    if not animals:
        return None
    obs = float(np.mean([float(np.mean(post[a])) - float(np.mean(pre[a])) for a in animals]))
    out = []
    for _ in range(n_boot):
        d = []
        for a in (animals[i] for i in rng.integers(0, len(animals), len(animals))):
            pa, qa = post[a], pre[a]
            d.append(float(np.mean([pa[i] for i in rng.integers(0, len(pa), len(pa))]))
                     - float(np.mean([qa[i] for i in rng.integers(0, len(qa), len(qa))])))
        if d:
            out.append(float(np.mean(d)))
    if len(out) < n_boot // 4:
        return None
    o = np.asarray(out)
    return Interval(obs, float(np.percentile(o, PCTL[0])), float(np.percentile(o, PCTL[1])),
                    len(animals))


def boot_delta_pairs(pairs, rng, n_boot: int = N_BOOT):
    """`boot_delta` for callers holding ``{animal: (post_values, pre_values)}``.

    `lick_bout_structure` builds its pairs that way because the delta is computed per quintile and
    threading two parallel dicts through five bins was worse. Same draw, same numbers -- this
    unzips the argument, it does not define a second bootstrap.
    """
    return boot_delta({a: v[0] for a, v in pairs.items()},
                      {a: v[1] for a, v in pairs.items()}, rng, n_boot)


# ---------------------------------------------------------------------------------------------
# 2. WHICH SESSIONS
# ---------------------------------------------------------------------------------------------

def curated_sessions(animals=None, *, phases=("pre", "post"), require_epoch=True, order="file"):
    """Session DICTS for the curated cohort. The filter repeated in seventy modules, once.

    ``config.pooled_labels()`` is the real definition of "pre-stroke plus post-stroke, nothing
    else" and the reasoning lives there -- read that docstring, not this one, for why
    `curated_dates()` cannot express this cohort. **It is CALLED, not re-derived**: the seventy
    copies each spelled it out as ``set(phase_labels("pre") + phase_labels("post"))``, which is
    `pooled_labels`' body, so this module would have been the seventy-first copy. All this adds is
    the two filters the copies also added: an epoch must resolve, and `--animals` may narrow.

    **``order="file"`` IS NOT A STYLE CHOICE AND THE DEFAULT IS NOT SORTED.** `config.load_sessions`
    does NOT return labels in sorted order (measured: 119 sessions, `labels != sorted(labels)`),
    and every one of the seventy copies filters `load_sessions()` in place. A serial module builds
    its bootstrap pool by iterating that list, so switching to sorted order changes the RNG draw
    sequence and moves published CIs -- the same failure as the fan-out completion-order bug, from
    the opposite direction. ``order="sorted"`` exists for new code and for anything already
    collecting sorted; **changing an existing caller's order needs a re-run and a DECISIONS
    entry, not a one-line edit.**
    """
    from wfield_local import config, epochs

    want = (set(config.pooled_labels()) if tuple(phases) == ("pre", "post")
            else {x for ph in phases for x in config.phase_labels(ph)})
    keep = [s for s in config.load_sessions()
            if s["label"] in want
            and (epochs.epoch_of(s["label"]) if require_epoch else True)
            and not (animals and config.animal_of(s["label"]) not in animals)]
    return sorted(keep, key=lambda s: s["label"]) if order == "sorted" else keep


def curated_labels(animals=None, **kw):
    """Just the labels from `curated_sessions`, same arguments and the same ordering caveat."""
    return [s["label"] for s in curated_sessions(animals, **kw)]


# ---------------------------------------------------------------------------------------------
# 3. WHAT A SESSION'S BEHAVIOUR IS
# ---------------------------------------------------------------------------------------------

def daq_rate(s) -> float:
    """DAQ sample rate, so a seconds-valued window can be compared against samples.

    Moved here from `channel_position_maps._daq_rate`, which `quit_prodrome` and
    `lick_bout_structure` were both reaching into across the `scripts/` tree -- a four-line h5
    attribute read is not a reason for one analysis script to import another's private helper.
    """
    import h5py

    with h5py.File(s["h5"], "r") as f:
        return float(f.attrs["sample_rate_hz"])


def lick_samples(s):
    """Lick onset samples for one session, with the discriminator settings the deck uses.

    THE FIVE MAGIC NUMBERS ARE THE POINT. ``(2.5, 1.0, (0.001, 0.020), 0.10)`` appears verbatim in
    thirty-seven places; they are a Schmitt trigger's high and low thresholds, an accepted pulse
    width, and a refractory period. This repo has already had to collapse three copies of a lick
    discriminator. Anyone tempted to tune them should change them HERE and re-run everything, not
    locally in one analysis.
    """
    from wfield_local.plot_lick_aligned_averages import _load_daq_events

    return _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)["lick_samples"]


class Behavior(NamedTuple):
    """One session's trial-level behaviour, gated and ready to measure.

    ``trials`` is EVERY trial and ``gated`` is the subset a per-trial measure may use. Both are
    returned because they answer different questions and the distinction has been got wrong here
    before: the licks-per-minute curves and the quit alignment deliberately keep every trial,
    because the quit is what they are measuring, while a per-trial average must not include the
    terminal quit period -- those trials have ~zero licks and are 0.040 of pre-stroke trials
    against 0.217 acutely, so an ungated per-trial curve has its composition tracking the
    independent variable.
    """

    session: dict
    trials: list
    gated: list
    quit: dict
    cue_samples: np.ndarray
    lick_samples: np.ndarray
    sample_rate: float
    near: set


def session_behavior(lab, *, gate=False, horizon_min=None, resp_s=RESP_S,
                     min_trials=60, min_gated=30):
    """Everything `quit_prodrome` and `lick_bout_structure` load per session. None if unusable.

    THIS WAS CHARACTER-IDENTICAL IN BOTH MODULES, including the two thresholds: a session needs 60
    trials to be worth scoring at all, and 30 SURVIVING the gate and the horizon to contribute a
    per-trial measure. Returning None rather than a short record is deliberate -- a partial result
    looks exactly like a legitimately short session downstream.

    ``gate`` drops trials at or after the animal's quit (skipped for a censored session, which by
    definition has no confirmed quit). ``horizon_min`` is ADMINISTRATIVE CENSORING: observe every
    session for the same wall-clock span so a rate is comparable across epochs of different
    session lengths.

    **THE CALLER PASSES THESE AS ARGUMENTS AND THE WORKER PASSES THEM IN ITS ITEM, NEVER THROUGH A
    MODULE GLOBAL.** `parallel.fan_out` uses the spawn start method, so a child re-imports its
    module fresh and never sees a global the parent assigned at runtime -- a gate set that way
    silently does nothing in every worker while working perfectly in a serial run.
    """
    from scripts.rest_migration.engagement_decomposition import near_codes, session_trials
    from scripts.rest_migration.quit_point import session_quit
    from wfield_local import config
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events

    s = next((x for x in config.load_sessions() if x["label"] == lab), None)
    if s is None:
        return None
    tr = session_trials(s, resp_s)
    if len(tr) < min_trials:
        return None
    q = session_quit(s, tr)
    if q is None:
        return None

    tt = sorted(tr, key=lambda r: float(r["elapsed_s"]))
    if gate and not q["censored"]:
        tt = [r for r in tt if float(r["elapsed_s"]) < float(q["quit_elapsed_s"])]
    if horizon_min:
        tt = [r for r in tt if float(r["elapsed_s"]) <= horizon_min * 60.0]
    if len(tt) < min_gated:
        return None

    # LOADED AFTER THE REJECTIONS, NOT BEFORE. These two reads come off MICROSCOPE and dominate the
    # per-session cost; a session that cannot contribute should not pay for them.
    cs = np.asarray(_load_cue_events(s["h5"])["cue_samples"], np.int64)
    return Behavior(session=s, trials=tr, gated=tt, quit=q, cue_samples=cs,
                    lick_samples=np.asarray(lick_samples(s), np.int64),
                    sample_rate=float(daq_rate(s)), near=near_codes())


# ---------------------------------------------------------------------------------------------
# 4. HOW THE WORK IS SPREAD
# ---------------------------------------------------------------------------------------------

def fan_sessions(items, worker, *, jobs=None, key=None, label="session", log=print):
    """`parallel.fan_out` with the sort built in. Returns ``(results, failures)``.

    **THE SORT IS THE WHOLE REASON THIS EXISTS** (CLAUDE.md ground rule 6). `fan_out` returns
    results in COMPLETION order, and keying them by label is not enough: dict insertion order is
    then completion order, every bootstrap pool is built by iterating that dict, and a seeded RNG
    drawing indices over a differently-ordered list gives different draws. Measured on
    `quit_prodrome`: the point estimate stayed exact at -18.7 licks/min while the CI moved from
    [-22.9, -13.7] to [-23.1, -13.6]. Small, and a CI that changes between runs is not
    reproducible. Each of the four converted modules re-derived this sort by hand; here it is not
    something a caller can forget.

    ``results`` is ``[(item, value), ...]`` sorted by ``key(item)``, defaulting to the item itself
    -- which is the label for a plain-label fan-out and ``(label, gate, horizon)`` for the modules
    whose options travel in the item, and sorts correctly either way because the label is first.
    Failures are returned unsorted and are never swallowed; a caller that writes a table from a
    partial result should refuse, because a short output file looks identical to a legitimately
    short one.
    """
    from wfield_local import parallel

    items = list(items)
    res, fail = parallel.fan_out(items, worker, jobs=jobs, label=label, log=log)
    return sorted(res, key=lambda kv: (key or (lambda x: x))(kv[0])), fail
