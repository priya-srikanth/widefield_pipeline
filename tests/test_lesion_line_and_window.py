"""Two things that were quietly wrong about post-stroke sessions (Priya, 2026-08-22).

1. THE LESION LINE. It was drawn before the first session whose phase is not "pre". PS92/PS93's
   8/17 session is EXCLUDED -- the 8/16 attempt did not take and their effective lesion followed the
   8/17 recording -- so "not pre" put the line between 8/14 and 8/17 and drew a session recorded
   BEFORE the lesion on the post side of it. PS94/PS95 were unaffected only because their 8/17 IS
   post, so the two rules coincide and the bug was invisible in half the cohort.

2. THE FROZEN DECODER'S ENGAGED CUT. _args hardcoded max_rt=2.0 while decode.max_rt_s moved to 3.5
   on 2026-08-21. A lick at 2.5 s is a hit the task rewarded, and this module was filing it as
   no-lick -- so "engaged" meant one thing in the per-session decoders (which read the config) and
   another in Section D.
"""
import inspect

from wfield_local import config


# ---------------------------------------------------------------- the lesion line

def _first_post(phases):
    return next((i for i, p in enumerate(phases) if p == "post"), None)


def _first_non_pre(phases):
    return next((i for i, p in enumerate(phases) if p != "pre"), None)


def test_an_excluded_session_sits_on_the_pre_side_of_the_line():
    """The 8/17 case: recorded before the effective lesion, so it belongs left of the line."""
    phases = ["pre", "pre", "excluded", "post", "post"]
    assert _first_post(phases) == 3
    assert _first_non_pre(phases) == 2, "the old rule put the line one session too early"


def test_the_two_rules_agree_when_nothing_is_excluded():
    """Why the bug was invisible for PS94/PS95."""
    phases = ["pre", "pre", "post", "post"]
    assert _first_post(phases) == _first_non_pre(phases)


def test_the_real_cohort_phases_move_the_line_for_ps92_and_ps93():
    for animal, shifts in (("PS92", True), ("PS93", True), ("PS94", False), ("PS95", False)):
        labs = sorted(s["label"] for s in config.load_sessions()
                      if config.animal_of(s) == animal)
        phases = [config.session_phase(animal, l.split("_")[1]) for l in labs]
        if "post" not in phases:
            continue
        moved = _first_post(phases) != _first_non_pre(phases)
        assert moved is shifts, (
            f"{animal}: expected the line to {'move' if shifts else 'stay'}; "
            f"phases around the lesion = {phases[-5:]}")


def test_both_figures_use_the_first_post_session():
    from wfield_local import hemispheric_dynamics, hemispheric_intensity

    for mod in (hemispheric_dynamics, hemispheric_intensity):
        src = inspect.getsource(mod)
        assert "first_non_pre" not in src, f"{mod.__name__} still places the lesion line by 'not pre'"
        assert "first_post" in src


# ---------------------------------------------------------------- the engaged cut

def test_the_frozen_decoder_uses_the_configured_response_window():
    from wfield_local.locanmf_frozen_decoder import _args

    assert _args().max_rt == float(config.defaults()["decode"]["max_rt_s"])


def test_the_frozen_decoder_cut_is_not_hardcoded():
    from wfield_local import locanmf_frozen_decoder as fd

    src = inspect.getsource(fd._args)
    assert "max_rt=2.0" not in src, (
        "max_rt is hardcoded again; it must come from decode.max_rt_s or Section D will cut at a "
        "different boundary than every other decoder")


def test_the_configured_window_matches_the_task():
    """3.5 s is the task's response window (gui_config timing.response_window), not a tuning knob."""
    assert float(config.defaults()["decode"]["max_rt_s"]) == 3.5


def test_no_production_module_hardcodes_the_engaged_cut():
    """The 3.5 s change reached only the modules that read config.

    On 2026-08-22 three still had `max_rt=2.0` baked into their own _args(): the frozen decoder,
    locanmf_cross_mouse and locanmf_rsa -- all three of which run in the nightly. Their figures sat
    in the same deck as the decoders', describing a different set of trials, with nothing on either
    to say so. The sweep/test modules are exempt: a parameter sweep pinning its own value is the
    point of it.
    """
    import re
    from pathlib import Path

    # Sweeps pin their own value by design. nolick_decoder and poststroke_compare use the 2.0 s
    # boundary to build the THREE-arm split (engaged / late_rewarded / undetected) and then
    # union the first two, which is the response window -- raising theirs would delete the late
    # arm. Both now say so in code; before 2026-08-22 that reason lived only in deck prose.
    EXEMPT = {"decoder_c_sweep.py", "encoder_bins_test.py", "filter_acausality_test.py",
              "postcue_window_test.py", "nolick_decoder.py", "poststroke_compare.py"}
    offenders = []
    for f in sorted(Path("wfield_local").glob("*.py")):
        if f.name in EXEMPT:
            continue
        src = f.read_text(encoding="utf-8")
        for m in re.finditer(r"max_rt\s*=\s*([0-9]+\.?[0-9]*)", src):
            offenders.append(f"{f.name}: max_rt={m.group(1)}")
    assert not offenders, (
        "engaged cut hardcoded in production module(s): " + "; ".join(offenders) +
        " -- it must come from decode.max_rt_s")
