"""The post-stroke-capable analyses must all gate engagement the SAME way.

`flag_engagement` is position-BLIND: it judges one rolling response rate over all six spouts. After
a lesion that collapses because the animal cannot reach the far positions, so using it there labels
the motor deficit as disengagement -- the effect filed under the confound. Measured on the sessions
on disk 2026-09-08: it excludes 380 of PS94_0817's 643 trials and 224 of PS95_0817's 720, where the
reference gate excludes none.

It survives only for `nolick_decoder`-independent pre-stroke bookkeeping; any module that can see a
post-stroke session must use `precue_engagement_states.engagement_gate` (or `reference_engagement`,
which wraps it), so the rule has ONE definition with the non-recovery requirement and the backdating
to the start of the run of misses that trips it.
"""
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Modules that run over post-stroke sessions and therefore must not use the position-blind gate.
POSTSTROKE_CAPABLE = ("nolick_decoder.py", "poststroke_compare.py", "spout_behavior.py",
                      "behavior_clips.py", "position_coding_directions.py")


def test_poststroke_capable_modules_do_not_call_the_position_blind_gate():
    offenders = []
    for name in POSTSTROKE_CAPABLE:
        src = (REPO / "wfield_local" / name).read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue          # prose ABOUT the old gate is what this file is made of
            if "flag_engagement(" in stripped and "def flag_engagement" not in stripped:
                offenders.append(f"{name}:{i}: {stripped[:90]}")
    assert not offenders, "position-blind gate used where post-stroke data can reach:\n" + "\n".join(offenders)


def test_the_reference_positions_have_one_definition():
    """Two modules build the reference set in CODE space; they must not drift apart."""
    from wfield_local import poststroke_compare as pc
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    from wfield_local.precue_engagement_states import REFERENCE

    derived = {c for c, nm in POSITION_NAMES.items() if nm in REFERENCE}
    assert derived == set(pc.REFERENCE_POSITIONS)
    assert {POSITION_NAMES[c] for c in derived} == set(REFERENCE)


def test_the_shared_gate_backdates_to_the_start_of_the_run_that_trips_it():
    """The 2026-09-07 change: the trials leading into the collapse are disengaged too, not 'working'."""
    from wfield_local.precue_engagement_states import engagement_gate

    # all reference-position trials: 20 responses, then a terminal run of 20 misses
    n = 40
    responded = np.array([True] * 20 + [False] * 20, bool)
    positions = np.array(["close_L"] * n)
    not_eng = np.asarray(engagement_gate(np.arange(n), responded, positions), bool)
    assert not_eng[20:].all(), "the terminal collapse must be disengaged"
    assert not not_eng[:20].any(), "trials before the last success must stay engaged"
    # the gate starts AT the run, not at the point the rolling mean happened to cross
    assert int(np.flatnonzero(not_eng)[0]) == 20
