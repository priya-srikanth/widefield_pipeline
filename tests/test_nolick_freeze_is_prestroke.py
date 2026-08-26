"""Freezing a "pre-stroke reference" must verify it IS pre-stroke.

Raised by the second analysis window, 2026-08-26: `nolick_reference_prestroke.json` is written by
`if not frozen.exists(): frozen.write_text(ref_path.read_text())`, and `ref_path` is built from
`from_list` -- every phase. So the freeze copies whatever the live reference happens to contain.

The existing artifact was checked and IS clean: its own `dates` field lists the 11 pre-stroke dates
(0606-0814) and the 44 session labels inside it are all pre-stroke. But that is an accident of WHEN
it was frozen, not a property of the code. The live `nolick_reference.json` today covers 19 dates
including 0817-0824. Delete the frozen file, or run the freeze for the first time on the other
analysis box, and you would mint a "pre-stroke reference" full of post-stroke sessions with nothing
to object -- `exists()` cannot see inside a file.

That is the same shape as the frozen-decoder contamination fixed earlier the same day: an artifact
whose NAME asserts a property that nothing checks.
"""
import inspect

from wfield_local import nightly_figs


def _freeze_block() -> str:
    src = inspect.getsource(nightly_figs.main)
    assert "nolick_reference_prestroke.json" in src, "freeze site moved; this test needs updating"
    return src


def test_the_freeze_checks_the_phase_before_writing():
    src = _freeze_block()
    i = src.index("nolick_reference_prestroke.json")
    window = src[i:i + 2500]
    assert "poststroke_dates()" in window, (
        "the freeze does not check whether the reference it is about to freeze covers post-stroke "
        "dates")


def test_a_contaminated_reference_is_refused_not_frozen():
    """Refusing is the only safe branch: writing it would preserve the contamination under a name
    that asserts the opposite, and every later reader would trust the name."""
    src = _freeze_block()
    i = src.index("nolick_reference_prestroke.json")
    window = src[i:i + 2500]
    assert "NOT freezing" in window, "no refusal path"
    # the write must be on the clean branch, i.e. guarded by the post-stroke check
    assert window.index("poststroke_dates()") < window.index("frozen.write_text"), (
        "the write happens before the phase check, so the check cannot prevent it")


def test_the_existing_artifact_is_actually_clean():
    """Pins the fact this was verified rather than assumed. If the on-disk reference ever stops
    being pre-stroke-only, that is a finding, not a test to relax."""
    import json
    from pathlib import Path

    from wfield_local import config
    p = Path(config.resolver().root("figures_working")) / "nolick_reference_prestroke.json"
    if not p.exists():                      # not every checkout has the artifact
        return
    dates = json.load(open(p)).get("dates") or []
    post = sorted(set(dates) & set(config.poststroke_dates()))
    assert not post, (
        f"the frozen pre-stroke no-lick reference covers post-stroke dates {post}; every no-lick "
        f"result read against it is comparing post-stroke data to a reference that contains it")
