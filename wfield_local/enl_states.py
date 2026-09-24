"""ENL (pre-cue) activity on WORKING vs STOPPED trials — how much of it is sensory, how much plan.

Priya, 2026-09-24. The spout is physically in position during the ENL of every trial, so whatever
the cortex encodes about WHERE it is should be present whether or not the animal is going to move.
A trial the animal is still working is one where a plan is also being formed; a trial from the
terminal quit period is one where (pre-stroke) it is not. Subtracting them is therefore an attempt
to separate the sensory representation of position from the motor plan built on top of it, using
the animal's own behaviour as the switch rather than a task manipulation.

    success   a detected lick within `decode.max_rt_s`             -- `is_engaged`
    working   no lick, but the animal is still responding at the reference positions
    stopped   no lick, inside the FINAL non-recovering collapse    -- `engagement_gate`

**THE DEFAULT CONTRAST IS `working` AGAINST `stopped`, NOT `success` AGAINST `stopped`** (Priya,
2026-09-24). Both classes are non-hits, so outcome, reward and the movement itself are matched and
the only thing that differs is whether the animal was still working -- which is the variable of
interest. `--with-success` adds the success arm as a sensitivity check; it has far more trials and
reintroduces exactly the confounds the working/stopped pairing removes.

**PRE-STROKE THE INFERENCE HOLDS; POST-STROKE IT IS NOT UNDERWRITTEN, AND THAT IS NOT A DETAIL.**
Pre-stroke a no-lick trial "is essentially the sated/disengaged state"
(`precue_engagement_states`), so `stopped` really is "no plan". Post-stroke it may not be:
`grant_confusion` records that *nothing in the spout data proves the terminal run is satiety rather
than a late motor collapse*. If it is a motor collapse the plan may be intact and unexecuted, in
which case working - stopped is plan-minus-plan, the difference shrinks, and it would read as
"ENL is mostly sensory" for entirely the wrong reason. So:

  * the post-stroke contrast is COMPUTED AND REPORTED, never used to support "no plan was formed";
  * `precue_engagement_states` already built the independent witness -- a discriminator trained on
    the PRE-stroke lick/no-lick contrast, with post-stroke working and stopped pushed through it.
    `witness_verdict()` reads that result and stamps every post-stroke row with it, so the caveat
    travels with the number instead of living only in this docstring.

This is NOT the rolling per-trial gate that `POSTSTROKE_ENGAGEMENT_FILTERING = False` forbids.
That flag rejects adjudicating individual trials on a local dip in reference rate. `engagement_gate`
requires a non-recovering FINAL collapse, which `grant_confusion` states is "a much weaker claim
than adjudicating individual trials, and it is the same construct the miss-vs-stopped split rests on
throughout". Nothing here filters another analysis.

**THE TWO CLASSES CANNOT BE TIME-MATCHED, AND THAT LIMITS WHAT THE SUBTRACTION CAN CLAIM.**
`engagement_gate` marks ONE terminal run, so `working` (no-lick, engaged) exists only before the
onset and `stopped` (no-lick, not engaged) only after: working always precedes stopped, by
definition of the gate. `precue_engagement_states` removes elapsed time from its lick-vs-no-lick
contrast by drawing both classes from the same late window; copying that here leaves ZERO working
trials, which a smoke test caught before this ever touched data. `adjacent_window` does the best
available thing -- working trials from a same-length window ending at the onset -- and `time_gap`
reports the separation that remains. A working-minus-stopped difference is therefore always partly
a late-session difference, and the size of that residual is part of the result rather than a
footnote.

**WHICH IS WHY THE DECODING ARM, NOT THE SUBTRACTION, IS THE ONE THAT CAN CARRY A CLAIM ALONE.**
Decoding position from STOPPED ENL against a permutation null compares nothing across classes and
so inherits none of this: position readable from stopped ENL means the sensory representation
survives without a plan, full stop. The subtraction sizes the plan's contribution; the decode
establishes that the sensory part is there at all.

ONE UNIVERSE, ONE DEFINITION. The trials, positions and ordering come from the imaging pipeline's
own `_trial_features`/`classify_cues_with_backup`, never from the behaviour table -- the two disagree
on free rewards and on the response window. Features come from `locanmf_position_decoder`'s
`_build_signal` / `_window_feature` / `_bins_for`, so a trial here is byte-identical to the same
trial in the deck. Only the CATEGORISATION is new, which is the whole point of the module.

CLI::

    python -m wfield_local.enl_states --animal PS94            # per-position contrast
    python -m wfield_local.enl_states --animal PS94 --decode   # + position decoding on stopped ENL
    python -m wfield_local.enl_states --cohort --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local.locanmf_position_decoder import is_engaged
from wfield_local.precue_engagement_states import engagement_gate

# The feature layer borrows `_build_signal` / `_window_feature` / `_bins_for` /
# `_load_cue_events` from `locanmf_position_decoder` and `classify_cues_with_backup` from
# `behavior_position`, exactly as `nolick_decoder` does -- imported at the point of use rather
# than here, so this module's import list never claims a dependency it does not yet exercise.

#: The three states, in the order they are reported. Names match `behavior_clips.CATEGORIES` on
#: purpose -- the same words must mean the same trials in the clips someone watches and in the
#: numbers they are checking against.
STATES = ("success", "working", "stopped")

#: The contrast this module exists for. `success` is available but is not the default; see the
#: module docstring for why matching on outcome is the point.
DEFAULT_CONTRAST = ("working", "stopped")


def _cfg() -> dict:
    return config.defaults().get("enl_states") or {}


def align() -> str:
    """Always the pre-cue window. Stated as a function so a caller cannot quietly pass `cue`.

    The ENL is the pre-cue period by definition, and the existing state analysis
    (`precue_engagement_states.run_animal`) is `align="precue"` for the same reason. A cue-aligned
    window would include the cue response, which is the thing this contrast is trying to exclude.
    """
    return "precue"


def states_for(order, responded, positions, first_lick, rt, max_rt):
    """Per-trial state, in the imaging universe. Returns an array of `STATES` entries.

    Two DIFFERENT gates, and conflating them is the easy mistake:

      * `is_engaged` is per-trial and LICK-based -- did this trial get a response in time. It
        separates `success` from the no-lick trials.
      * `engagement_gate` is per-SESSION and reference-rate based -- is the animal still working at
        all. It separates `working` from `stopped` among those no-lick trials, and only on a final
        non-recovering collapse.

    `behavior_clips.categorise` composes the same two for the clip labels; this is that composition
    on the imaging pipeline's trials rather than the behaviour table's.
    """
    order = np.asarray(order)
    hit = np.array([is_engaged(fl, r, max_rt) for fl, r in zip(first_lick, rt)], bool)
    not_eng = engagement_gate(order, np.asarray(responded, bool), np.asarray(positions))
    out = np.full(order.size, "stopped", dtype=object)
    out[~hit & ~not_eng] = "working"
    out[hit] = "success"
    return out


def adjacent_window(state, order):
    """Working trials from the window immediately BEFORE the collapse. Returns a bool mask.

    **THE TWO CLASSES ARE TEMPORALLY DISJOINT BY CONSTRUCTION, and no matching can undo it.**
    `engagement_gate` marks one terminal run: everything from the backdated onset to the end of the
    session is not-engaged. `working` is no-lick AND engaged, so it can only occur BEFORE that
    onset; `stopped` is no-lick AND not-engaged, so it can only occur after. Working always precedes
    stopped, in every session, by definition of the gate.

    That kills the obvious control. `precue_engagement_states` removes elapsed time from its
    lick-vs-no-lick contrast by drawing both classes from the same late window, and a first draft of
    this module copied that -- which silently left ZERO working trials, because there are none in
    the stopped window to draw. Caught by a smoke test before it ever ran on data.

    What is actually available is ADJACENCY: take the working trials from a window of the same
    length ending at the collapse onset, so the two classes are as close in time as the gate allows.
    This SHRINKS the confound; it does not remove it, and `time_gap()` reports what is left so the
    reader can size it rather than assume it away.

    The consequence for reading the result is worth stating plainly: a working-minus-stopped
    difference is always partly a late-session difference. The decoding arm does not share this
    problem -- it asks whether position is readable WITHIN stopped trials against a permutation
    null, comparing nothing across classes -- which is why that arm, not this subtraction, is the
    one that can carry a claim on its own.
    """
    state, order = np.asarray(state, dtype=object), np.asarray(order)
    stopped = order[state == "stopped"]
    if stopped.size == 0:
        return np.ones(order.size, bool)
    onset = stopped.min()
    span = order.max() - onset + 1
    keep = (order >= onset - span) & (order < onset)
    return keep | (order >= onset)


def time_gap(state, order):
    """Trials between the last kept WORKING trial and the first STOPPED one, and class midpoints.

    Reported rather than assumed away: the classes cannot be time-matched (see `adjacent_window`),
    so the size of the remaining separation is part of the result. A cell where the two midpoints
    are hundreds of trials apart is not the same evidence as one where they are adjacent.
    """
    state, order = np.asarray(state, dtype=object), np.asarray(order)
    w, st = order[state == "working"], order[state == "stopped"]
    if w.size == 0 or st.size == 0:
        return {"gap": np.nan, "working_mid": np.nan, "stopped_mid": np.nan,
                "n_working": int(w.size), "n_stopped": int(st.size)}
    return {"gap": float(st.min() - w.max()),
            "working_mid": float(np.median(w)), "stopped_mid": float(np.median(st)),
            "n_working": int(w.size), "n_stopped": int(st.size)}


def contrast(df, states=DEFAULT_CONTRAST, by=("animal", "epoch", "position")):
    """Mean ENL amplitude per cell, and the difference between the two states.

    Returns one row per `by` cell with `n_<state>`, `mean_<state>` and `delta`. NO bootstrap here:
    at n=4 the per-animal table is printed BEFORE any resampling (CLAUDE.md rule 8), because the
    two traps that killed four results on 2026-09-19 -- a cell carried by one animal, and an animal
    with a single session -- are invisible once the numbers are pooled.
    """
    a, b = states
    g = (df[df.state.isin(states)]
         .groupby(list(by) + ["state"], dropna=False)["amp"]
         .agg(["size", "mean"]).unstack("state"))
    out = pd.DataFrame(index=g.index)
    for s in states:
        out[f"n_{s}"] = g[("size", s)] if ("size", s) in g.columns else 0
        out[f"mean_{s}"] = g[("mean", s)] if ("mean", s) in g.columns else np.nan
    out["delta"] = out[f"mean_{a}"] - out[f"mean_{b}"]
    return out.reset_index()


def witness_verdict(path=None):
    """What `precue_engagement_states` concluded about whether the gate shows up in cortex.

    Returns ``{"separates": bool|None, "source": str}``. ``None`` means the witness has not been
    run, which is NOT the same as a negative and is reported as "unknown" rather than defaulted.

    This exists so the post-stroke caveat travels ATTACHED TO THE NUMBER. A reader who sees only a
    working-minus-stopped difference will read it as plan-minus-no-plan; that reading is licensed
    only if a discriminator trained where engagement is not in question can also tell the two apart.
    """
    p = Path(path) if path else Path(_cfg().get("witness_json", "")) if _cfg().get("witness_json") else None
    if p is None or not p.is_file():
        return {"separates": None, "source": "not run"}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {"separates": d.get("separates"), "source": str(p)}


def stamp_caveat(df, witness=None):
    """Add the interpretation each row does and does not license."""
    w = witness or witness_verdict()
    pre = df["epoch"].astype(str).str.lower().eq("pre")
    df = df.copy()
    df["interpretation"] = np.where(
        pre,
        "plan-minus-no-plan (pre-stroke: no-lick is the sated state)",
        {True: "plan-minus-no-plan (witness: gate separates in cortex)",
         False: "DESCRIPTIVE ONLY (witness: gate does not separate)",
         None: "DESCRIPTIVE ONLY (witness not run; stopped may be late motor collapse)"
         }[w["separates"]])
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animal", action="append", default=None)
    ap.add_argument("--cohort", action="store_true")
    ap.add_argument("--with-success", action="store_true",
                    help="add the success arm as a sensitivity check (more trials, but outcome and "
                         "movement are no longer matched)")
    ap.add_argument("--no-late-matched", action="store_true",
                    help="do NOT restrict working trials to the stopped trials' window; a "
                         "difference then has elapsed time as a rival explanation")
    ap.add_argument("--decode", action="store_true",
                    help="also decode position from the STOPPED ENL against a permutation null")
    ap.add_argument("--out", default=None)
    ap.add_argument("--machine", default=None)
    ap.parse_args(argv)
    raise SystemExit("enl_states: analysis driver not wired yet -- see the module docstring for the "
                     "design it implements.")


if __name__ == "__main__":
    raise SystemExit(main())
