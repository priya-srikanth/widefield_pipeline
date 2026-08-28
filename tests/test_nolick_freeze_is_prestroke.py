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


def test_the_freeze_BUILDS_prestroke_rather_than_copying_and_checking():
    """UPDATED 2026-08-28, and the reason is the point.

    The original mechanism copied the live reference and refused when it covered post-stroke dates.
    Refusing was right. But `from_list` now ALWAYS contains post-stroke dates, so the guard could
    only ever say no: every night logged a refusal, and the only pre-stroke reference in existence
    stayed the one written on 2026-08-19 that happened to be clean. A guard that can never pass is
    not a mechanism -- it is a permanently closed door with a sign on it.

    So the freeze now BUILDS with `phase="pre"`, and the artifact matches its `kind` field by
    construction rather than by luck. The check below survives as a second line, not the only one.
    """
    src = _freeze_block()
    i = src.index("nolick_reference_prestroke.json")
    window = src[i:i + 4200]
    assert 'phase="pre"' in window, "the freeze does not restrict the build to pre-stroke"
    assert "poststroke_dates()" in window, "the produced artifact is no longer phase-checked"


def test_a_contaminated_reference_is_still_refused_not_published():
    """The second check must act on what was PRODUCED, not on what was requested.

    Asking `build_reference` for pre-stroke and then trusting that it complied is the same category
    of mistake as `exists()` standing in for "is pre-stroke". The artifact is asked what it actually
    covers, and a bad one is renamed out of the way so nothing downstream reads it as a reference --
    renamed rather than deleted, matching the convention everywhere else here.
    """
    src = _freeze_block()
    i = src.index("nolick_reference_prestroke.json")
    window = src[i:i + 4200]
    assert 'pre_ref.get("dates"' in window, (
        "the check reads the caller's date list rather than the artifact's own")
    assert "REFUSED_contaminated" in window, "a contaminated artifact is not moved out of the way"
    assert window.index("poststroke_dates()") < window.index("froze the PRE-STROKE"), (
        "the success message can be reached without the phase check")


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
