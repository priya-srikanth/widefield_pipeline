"""A logical-root override must reach EVERY consumer, not just the orchestrator.

FOUND 2026-08-28, while checking whether a third machine could run the pipeline.
`WIDEFIELD_FIGURES_WORKING` has been documented in `configs/paths.yaml` since 2026-08-11 as the way
to correct a machine whose profile resolves `figures_working` wrongly. It was read in exactly one
place -- `nightly_figs._default_out` -- while NINETEEN modules called
`PathResolver().root("figures_working")` directly and never saw it, among them every standalone
figure CLI and `joint_locanmf.BASIS_DIR`, which derives the joint-basis directory from that root.

So setting the documented variable fixed the nightly and silently did not fix anything run by hand,
nor where the joint basis was looked for. The failure it exists to prevent is on record: a box with
the imaging profile's mounts doing analysis work sent every figure to the other machine's
`C:/Users/sabatini/...` path and produced a deck with 80 slides and 287 missing figures, exit 0.

IT MATTERS FOR ANY NEW MACHINE. `detect_machine` knows three profiles and falls back to 'analysis',
so a third box adopts another machine's local paths by default and the env override is the supported
correction. A correction that works in one caller is not a correction.
"""
import os
import pathlib

import pytest

from wfield_local.paths import PathResolver


@pytest.fixture
def unset(monkeypatch):
    for k in list(os.environ):
        if k.startswith("WIDEFIELD_"):
            monkeypatch.delenv(k, raising=False)


def test_the_variable_name_is_derived_not_hardcoded(unset):
    """`figures_working` -> `WIDEFIELD_FIGURES_WORKING`, the name already documented in paths.yaml.
    Derived, so a new logical root gets an override without anyone adding a branch for it."""
    assert PathResolver.env_var("figures_working") == "WIDEFIELD_FIGURES_WORKING"
    assert PathResolver.env_var("labcams") == "WIDEFIELD_LABCAMS"


def test_the_override_wins_over_the_machine_profile(unset, monkeypatch):
    r = PathResolver(machine="analysis")
    before = r.root("figures_working")
    monkeypatch.setenv("WIDEFIELD_FIGURES_WORKING", "D:/elsewhere")
    assert PathResolver(machine="analysis").root("figures_working") == "D:/elsewhere"
    assert before != "D:/elsewhere", "the fixture machine already resolved to the override path"


def test_it_reaches_a_module_that_resolves_the_root_directly(unset, monkeypatch, tmp_path):
    """The actual regression. `joint_locanmf.BASIS_DIR` is computed from `figures_working`, and it
    is the one consumer where a wrong answer is not a missing figure but a WRONG BASIS -- the same
    session projected onto a different basis gives different features (256 columns vs 380, measured).
    """
    monkeypatch.setenv("WIDEFIELD_FIGURES_WORKING", str(tmp_path / "figs"))
    import importlib

    from wfield_local import joint_locanmf
    importlib.reload(joint_locanmf)
    assert pathlib.Path(joint_locanmf.BASIS_DIR) == tmp_path / "joint_bases", (
        "joint_locanmf did not honour the documented override")
    monkeypatch.delenv("WIDEFIELD_FIGURES_WORKING")
    importlib.reload(joint_locanmf)


def test_an_override_for_an_unavailable_root_still_works(unset, monkeypatch):
    """A root that is null for this machine raises without an override. That is exactly the case a
    new box needs to correct, so the override has to be consulted BEFORE the no-mount error."""
    r = PathResolver(machine="mac")
    with pytest.raises(RuntimeError):
        r.root("raw_labcams")
    monkeypatch.setenv("WIDEFIELD_RAW_LABCAMS", "/data/raw")
    assert PathResolver(machine="mac").root("raw_labcams") == "/data/raw"


def test_nightly_figs_no_longer_owns_the_override():
    """It kept its LOUD failure and gave up the override, or the two would drift apart again."""
    import inspect

    from wfield_local import nightly_figs
    src = inspect.getsource(nightly_figs._default_out)
    assert 'os.environ.get("WIDEFIELD_FIGURES_WORKING")' not in src, (
        "nightly_figs reads the override again; it belongs in PathResolver so all 19 callers see it")
    assert "SystemExit" in src, "the loud failure for a machine with no such root was lost"
