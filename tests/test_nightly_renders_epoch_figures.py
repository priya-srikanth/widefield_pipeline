"""The nightly render must actually BUILD the figures the deck places.

Section I was written, the figures rendered by hand, and nothing wired the renderer into the
nightly run -- so every future night would have refreshed sections A-H while section I kept
placing whatever manual render happened to be on disk. The deck could not have reported it: the
files were PRESENT, and a completeness check only knows about absence.
"""
import pathlib
import re

SRC = (pathlib.Path(__file__).resolve().parents[1] / "wfield_local" / "nightly_figs.py"
       ).read_text(encoding="utf-8")


def _grant_block():
    i = SRC.index('cli("wfield_local.grant_figures"')
    return SRC[i:SRC.index("# build the refined ANALYSIS deck")]


def test_the_nightly_run_builds_the_pooled_epoch_figures():
    assert 'cli("wfield_local.epoch_grant_figures")' in SRC, (
        "nothing in the nightly run builds the epoch figures deck section I places")


def test_epoch_figures_run_after_the_collectors_they_read():
    """They recompute nothing -- running first would repeat all of `grant_figures`' collection."""
    b = _grant_block()
    assert b.index("wfield_local.grant_figures") < b.index("wfield_local.epoch_grant_figures")


def test_the_deck_is_built_after_the_epoch_figures():
    assert SRC.index('cli("wfield_local.epoch_grant_figures")') < SRC.index("build_analysis_deck(")


def test_a_subset_run_does_not_overwrite_the_pooled_deliverable():
    """`--only PS92` must not publish a one-animal figure over the four-animal one.

    `grant_figures.ANIMALS` is a fixed four-tuple, but the collectors under it honour
    WIDEFIELD_ONLY_ANIMALS -- so a subset run yields a correct figure of the WRONG THING at the
    path the deck reads.
    """
    b = _grant_block()
    call = b.index('cli("wfield_local.epoch_grant_figures")')
    guard = b.rindex("if only:", 0, call)
    assert guard > b.index("wfield_local.grant_figures"), "the epoch step is not guarded at all"
    # the guard's SKIP branch must not be the one that runs the renderer
    assert re.search(r"if only:.*?SKIPPED.*?else:\s*\n\s*cli\(\"wfield_local\.epoch_grant_figures\"\)",
                     b[guard:], re.S), "the subset guard does not skip the pooled render"


def test_skip_grant_says_both_sections_go_stale():
    """--skip-grant now leaves section I stale as well as H, and the log has to say so."""
    i = SRC.index("--skip-grant")
    msg = SRC[i:i + 400]
    assert "H and I" in msg or ("H" in msg and "I" in msg), (
        "the --skip-grant message still names only section H")
