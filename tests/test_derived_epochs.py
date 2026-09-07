"""`chronic_from` is DERIVED each run and published through a file. What that has to guarantee.

Priya, 2026-09-07: "I'd like the pipeline to run the epoch definitions and just determine if we have
met 'chronic' criteria, order sessions into epochs appropriately, and analyze."

The standing objection to deriving a boundary was never that the rule might be wrong -- it is that a
boundary can move between two runs and silently redraw published panels. Deriving does not remove
that risk, so these tests are mostly about the machinery that answers it: the file that records what
each figure set was built on, the diff that says what moved, and the pin that reproduces an old one.
"""
import json
import subprocess
import sys

import pytest

from wfield_local import epoch_audit, epochs


@pytest.fixture
def derived(monkeypatch, tmp_path):
    """Opt INTO the derived path. The suite is pinned by default (see conftest)."""
    monkeypatch.delenv("WIDEFIELD_EPOCHS_PINNED", raising=False)
    monkeypatch.setenv("WIDEFIELD_EPOCH_BOUNDARIES", str(tmp_path / "epoch_boundaries.json"))
    epochs.clear_resolved()
    yield tmp_path / "epoch_boundaries.json"
    epochs.clear_resolved()


def test_a_derived_boundary_reassigns_sessions(derived):
    """The point of the whole change: deriving a boundary must actually move sessions between
    epochs, not merely be recorded somewhere."""
    lab = "PS93_0828"                                   # day 11, subacute under the stored spec
    assert epochs.EPOCH_SPEC["PS93"]["chronic_from"] is None
    epochs.clear_resolved()
    assert epochs.epoch_of(lab) == "subacute"
    epochs.save_boundaries({"PS93": {"chronic_from": 11}}, derived)
    epochs.clear_resolved()
    assert epochs.epoch_of(lab) == "chronic"


def test_the_file_is_what_a_subprocess_reads(derived):
    """THE MECHANISM THIS DESIGN RESTS ON. `grant_figures` and `epoch_grant_figures` run as
    SUBPROCESSES, so a boundary installed only in the parent would build the per-day figures on
    derived boundaries and the pooled ones on stored boundaries -- and every figure would render
    without complaint. Asserted with a real child process, because an in-process check cannot
    distinguish "published through the file" from "left in a module global"."""
    epochs.save_boundaries({"PS93": {"chronic_from": 11}}, derived)
    out = subprocess.run(
        [sys.executable, "-c",
         "from wfield_local import epochs; print(epochs.epoch_of('PS93_0828'))"],
        capture_output=True, text=True, check=False,
        env={**__import__("os").environ, "WIDEFIELD_EPOCH_BOUNDARIES": str(derived)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "chronic", out.stdout


def test_the_pin_reproduces_the_stored_spec(derived, monkeypatch):
    """`WIDEFIELD_EPOCHS_PINNED=1` rebuilds an older figure set under the boundaries it was
    published with, whatever is sitting in the boundaries file."""
    epochs.save_boundaries({"PS93": {"chronic_from": 11}}, derived)
    epochs.clear_resolved()
    assert epochs.epoch_of("PS93_0828") == "chronic"
    monkeypatch.setenv("WIDEFIELD_EPOCHS_PINNED", "1")
    epochs.clear_resolved()
    assert epochs.epoch_of("PS93_0828") == "subacute"
    assert epochs.resolved_source() == "pinned"


def test_the_merge_cannot_delete_a_stored_boundary(derived):
    """A derived file carries only `chronic_from`. If it were treated as a REPLACEMENT rather than
    an overlay, acute and subacute_from would vanish and `epoch_of` would return None for every
    post-stroke session -- emptying every pooled panel while rendering cleanly."""
    epochs.save_boundaries({"PS92": {"chronic_from": 11}}, derived)
    epochs.clear_resolved()
    spec = epochs.spec_for("PS92")
    assert spec["acute"] == (1, 5)
    assert spec["subacute_from"] == 7
    assert epochs.epoch_of("PS92_0818") == "acute"
    assert epochs.epoch_of("PS92_0824") == "subacute"


def test_a_truncated_file_falls_back_instead_of_crashing(derived):
    """A run killed mid-write must not take every figure in the deck down with it."""
    derived.parent.mkdir(parents=True, exist_ok=True)
    derived.write_text('{"boundaries": {"PS93": {"chron', encoding="utf-8")
    epochs.clear_resolved()
    assert epochs.load_boundaries(derived) is None
    assert epochs.epoch_of("PS93_0828") == "subacute", "must fall back to the stored spec"


def test_changes_names_every_boundary_that_moved():
    """The record that replaces the stored spec's guarantee. A stored boundary could not move
    without someone editing it; a derived one can, so the run has to say which and from what."""
    moved = epoch_audit.changes({"PS92": {"chronic_from": 11}, "PS93": {"chronic_from": 11}},
                                {"PS92": {"chronic_from": 11}, "PS93": {"chronic_from": None}})
    assert moved == ["PS93 chronic_from: None -> 11"]
    assert epoch_audit.changes({"PS92": {"chronic_from": 11}},
                               {"PS92": {"chronic_from": 11}}) == []


def test_the_written_file_records_the_rule_that_produced_it(derived):
    """A boundaries file that said only "11" would be unfalsifiable a month later. It carries the
    rule text and the constants, so a figure set can be checked against the rule it claims."""
    epochs.save_boundaries({"PS92": {"chronic_from": 11}}, derived)
    blob = json.loads(derived.read_text(encoding="utf-8"))
    assert blob["boundaries"]["PS92"]["chronic_from"] == 11
    assert "hit rate AND licks" in blob["chronic_rule"]
    assert blob["constants"]["K_SD"] == pytest.approx(epochs.CHRONIC_K_SD)
    assert blob["constants"]["LEVEL_MIN"] == epochs.CHRONIC_LEVEL_MIN


def test_unavailable_behaviour_writes_nothing(monkeypatch, derived):
    """A missing cohort table must leave the stored spec in force and the file untouched. Deriving
    "no animal is chronic" from absent data would restage the entire deck on a mount failure."""
    monkeypatch.setattr(epoch_audit, "audit",
                        lambda **k: {"available": False, "reason": "no table"})
    res = epoch_audit.resolve()
    assert res["available"] is False
    assert not derived.exists(), "nothing may be written when behaviour is unavailable"
    assert epochs.epoch_of("PS93_0828") == "subacute"


def test_the_live_derivation_still_matches_the_stored_spec(derived):
    """Today the derived boundaries and `EPOCH_SPEC` agree, and that is worth pinning: it means the
    switch to deriving changed no published panel on the day it landed. Expected to fail when an
    animal genuinely crosses -- at which point the stored spec is the stale one, and the fallback it
    provides (on a crash, or a missing cohort table) is what needs updating."""
    if not epoch_audit._cohort_path().exists():
        pytest.skip("no cohort behaviour table on this box")
    rep = epoch_audit.audit()
    derived_now = epoch_audit.derived_spec(rep)
    stored = {a: {"chronic_from": s["chronic_from"]} for a, s in epochs.EPOCH_SPEC.items()}
    assert derived_now == stored, (
        "derived boundaries have moved away from EPOCH_SPEC; update the stored fallback so a "
        f"crash or a missing cohort table does not silently revert them: {derived_now} vs {stored}")


def test_the_first_run_does_not_announce_that_everything_moved():
    """No previous file means nothing to have moved FROM. Reporting every animal as changed on the
    first run -- or after the file is deleted or the share remounted -- would be false, and would
    train the reader to skip the one message that matters on the night it is true."""
    assert epoch_audit.changes({"PS92": {"chronic_from": 11}}, None) == []
    # an EMPTY previous file is different from an absent one: it really did lose a boundary
    assert epoch_audit.changes({"PS92": {"chronic_from": 11}}, {}) == [
        "PS92 chronic_from: absent -> 11"]
