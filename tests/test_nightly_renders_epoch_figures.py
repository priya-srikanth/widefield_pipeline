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


# --------------------------------------------------------------- the render loop's return contract
#
# THE SAME FAILURE SHAPE THIS FILE EXISTS FOR: outputs present on disk, nothing able to say they
# failed. `_frozen_vs_refit_overall` returned the single Path `ef.bar_row` gives instead of a list,
# so the render loop's `for q in (fn(...) or [])` raised
#     TypeError: 'WindowsPath' object is not iterable
# on EVERY align/variant -- after both figures and both sidecars had been written. The files looked
# complete, the numbers in them were correct, and all ten combinations were logged as failed.

EGF = (pathlib.Path(__file__).resolve().parents[1] / "wfield_local" / "epoch_grant_figures.py"
       ).read_text(encoding="utf-8")


def _registered_family_functions():
    """Names registered in the render loop's `(key, callable)` tables."""
    block = EGF[EGF.index("for _k, _fn in ((\"5r\""):]
    block = block[:block.index("\n        if \"scal\" in want")]
    return set(re.findall(r"\(\"[0-9a-z]+\", (_[A-Za-z_0-9]+)\)", block))


def test_every_family_the_render_loop_ITERATES_returns_a_list():
    """The loop does `for q in (fn(...) or [])`. A family returning a bare Path type-errors on every
    combination, which is invisible on disk because the figures are written first."""
    fns = _registered_family_functions()
    assert fns, "the render-loop registry moved; this test is no longer reading it"
    for name in sorted(fns):
        src = EGF[EGF.index(f"def {name}("):]
        src = src[:src.index("\ndef ", 1)]
        returns = re.findall(r"^    return (.+)$", src, re.M)
        assert returns, f"{name} has no top-level return"
        for r in returns:
            assert r == "None" or r.startswith("[") or r.startswith("made or None") \
                or r.startswith("[made]") or "_frozen_vs_refit(" in r or "_frozen_vs_refit_overall(" in r, (
                    f"{name} returns {r!r}; the render loop iterates it, so it must be a list "
                    f"(or None, or a delegation to a family that returns one)")


def test_the_two_new_pooled_families_are_declared_in_the_CLI_allowlist():
    """`--only` validates against an explicit tuple. A family wired into the loop but missing from
    that tuple cannot be rendered on its own, which is how the first 5ro run died before starting."""
    for key in ("5ro", "5rmo"):
        assert f'"{key}"' in EGF, f"{key} is not in the --only allowlist"


# --------------------------------------------------------------- stale sidecars from version skew

def test_a_sidecar_OLDER_than_its_figure_is_reported(tmp_path):
    """Two machines render into one directory. A checkout predating `write_values` overwrites the
    PNG and leaves the CSV, so the numbers describe a render nobody is looking at -- and because a
    CSV looks authoritative, it gets quoted."""
    import os
    from wfield_local import epoch_figures as ef

    png, csv_p = tmp_path / "f.png", tmp_path / "f.csv"
    csv_p.write_text("epoch,position,value\n", encoding="utf-8")
    png.write_bytes(b"x")
    os.utime(csv_p, (1, 1))                      # CSV older than the figure
    assert ef.stale_sidecars(tmp_path) == [("f.png", "f.csv")]


def test_a_sidecar_NEWER_than_its_figure_is_LEFT_ALONE(tmp_path):
    """The normal, correct state: the writer emits the figure first and the values second. A `<=`
    comparison here would delete every healthy pair in the directory."""
    import os
    from wfield_local import epoch_figures as ef

    png, csv_p = tmp_path / "g.png", tmp_path / "g.csv"
    png.write_bytes(b"x")
    csv_p.write_text("epoch,position,value\n", encoding="utf-8")
    os.utime(png, (1, 1))                        # figure older than the CSV
    assert ef.stale_sidecars(tmp_path) == []
    assert csv_p.exists()


def test_removal_is_OPT_IN_and_reporting_is_the_default(tmp_path):
    """A missing sidecar is recoverable by re-rendering; a wrongly deleted one is not."""
    import os
    from wfield_local import epoch_figures as ef

    png, csv_p = tmp_path / "h.png", tmp_path / "h.csv"
    csv_p.write_text("x\n", encoding="utf-8")
    png.write_bytes(b"x")
    os.utime(csv_p, (1, 1))
    ef.stale_sidecars(tmp_path)                  # default: report only
    assert csv_p.exists(), "the default must not delete"
    ef.stale_sidecars(tmp_path, remove=True)
    assert not csv_p.exists()


def test_a_figure_with_no_sidecar_at_all_is_not_flagged(tmp_path):
    """Most families have no CSV yet. Absence is not staleness."""
    from wfield_local import epoch_figures as ef

    (tmp_path / "i.png").write_bytes(b"x")
    assert ef.stale_sidecars(tmp_path) == []
