"""A pre-stroke reference must be reused verbatim, and must never move by accident.

Two failures this guards, both of which the repo has already met in other forms:

  SILENT DRIFT    Recomputing the reference every night means a post-stroke z score quoted in
                  tonight's deck is against a different band than the same z score in Friday's. The
                  no-lick arm was frozen for exactly this reason; the decoder band was not.
  SILENT STALENESS  The existing freeze is `if not frozen.exists()`. That cannot distinguish "current"
                  from "built from a session set that no longer exists" -- and on 2026-08-14 the flip
                  to the meegkit_hpfit SVTcorr changed the DATA under unchanged labels, so a
                  label-only check would not have caught it either.

The resolution is deliberately asymmetric: a HIT is silent and fast, a MISMATCH is loud and changes
nothing. Recomputing on mismatch would reintroduce the drift; trusting it blindly would score today's
data against inputs that are gone. Only an explicit `refreeze="<reason>"` may retire one, and it
renames rather than deletes.
"""
import json

import pytest

from wfield_local import prestroke_reference as pr


@pytest.fixture
def frozen_sig(monkeypatch):
    """Pin the signature so these tests do not depend on the live session table."""
    box = {"sig": "SIG_A"}
    monkeypatch.setattr(pr, "prestroke_signature", lambda **kw: box["sig"])
    monkeypatch.setattr(pr.config, "phase_labels", lambda *a, **k: ["PS92_0606", "PS93_0606"])
    return box


def test_first_call_computes_and_freezes(tmp_path, frozen_sig):
    p = tmp_path / "ref.json"
    calls = []
    payload, status = pr.load_or_freeze(p, lambda: calls.append(1) or {"band": 0.9},
                                        log=lambda *_: None)
    assert status == pr.STATUS_NEW and payload == {"band": 0.9} and len(calls) == 1
    doc = json.loads(p.read_text())
    assert doc["_freeze"]["signature"] == "SIG_A"
    assert doc["_freeze"]["freeze_version"] == pr.FREEZE_VERSION


def test_second_call_reuses_without_recomputing(tmp_path, frozen_sig):
    """The whole point: the expensive compute must NOT run again. This is where the hours go."""
    p = tmp_path / "ref.json"
    pr.load_or_freeze(p, lambda: {"band": 0.9}, log=lambda *_: None)

    def boom():
        raise AssertionError("recomputed a reference that was already frozen")

    payload, status = pr.load_or_freeze(p, boom, log=lambda *_: None)
    assert status == pr.STATUS_HIT and payload == {"band": 0.9}


def test_changed_inputs_are_reported_and_change_nothing(tmp_path, frozen_sig):
    """A stale reference must not be recomputed OR trusted -- the caller has to decide."""
    p = tmp_path / "ref.json"
    pr.load_or_freeze(p, lambda: {"band": 0.9}, log=lambda *_: None)
    frozen_sig["sig"] = "SIG_B"                      # a re-curation, or an upstream re-preprocess

    seen = []
    payload, status = pr.load_or_freeze(
        p, lambda: pytest.fail("recomputed on mismatch"), log=lambda m: seen.append(m))

    assert status == pr.STATUS_STALE
    assert payload == {"band": 0.9}, "the frozen numbers must come back unchanged"
    assert json.loads(p.read_text())["_freeze"]["signature"] == "SIG_A", "file was overwritten"
    assert any("DIFFERENT pre-stroke input set" in m for m in seen)


def test_refreeze_supersedes_by_renaming(tmp_path, frozen_sig):
    """Retiring a reference must keep the old numbers: an earlier deck was built against them."""
    p = tmp_path / "ref.json"
    pr.load_or_freeze(p, lambda: {"band": 0.9}, log=lambda *_: None)
    frozen_sig["sig"] = "SIG_B"

    payload, status = pr.load_or_freeze(p, lambda: {"band": 0.7},
                                        refreeze="added_two_prestroke_sessions",
                                        log=lambda *_: None)
    assert status == pr.STATUS_NEW and payload == {"band": 0.7}
    old = list(tmp_path.glob("ref.SUPERSEDED_added_two_prestroke_sessions_*.json"))
    assert len(old) == 1, "the superseded reference must be kept, with its reason in the name"
    assert json.loads(old[0].read_text())["payload"] == {"band": 0.9}
    assert json.loads(p.read_text())["payload"] == {"band": 0.7}


def test_a_parameter_change_is_a_different_reference(monkeypatch):
    """`extra` separates references that differ only by window/alignment/basis, so a cue-aligned
    band can never be served to a precue-aligned question."""
    monkeypatch.setattr(pr.config, "phase_labels", lambda *a, **k: [])
    monkeypatch.setattr(pr.config, "load_sessions", lambda *a, **k: [])
    a = pr.prestroke_signature(extra={"align": "cue"})
    b = pr.prestroke_signature(extra={"align": "precue"})
    assert a != b


def test_signature_tracks_the_inputs_not_just_the_labels(monkeypatch, tmp_path):
    """The 2026-08-14 scar: SVTcorr changed under identical labels."""
    svt = tmp_path / "SVTcorr.npy"
    svt.write_bytes(b"x" * 10)
    sess = [{"label": "PS92_0606", "mc": str(tmp_path)}]
    monkeypatch.setattr(pr.config, "phase_labels", lambda *a, **k: ["PS92_0606"])
    monkeypatch.setattr(pr.config, "load_sessions", lambda *a, **k: sess)
    monkeypatch.setattr(pr.config, "svtcorr_path", lambda mc: str(svt))
    monkeypatch.setattr(pr.config, "animal_of", lambda lab: lab.split("_")[0])

    before = pr.prestroke_signature()
    svt.write_bytes(b"y" * 999)                      # same label, different data
    assert pr.prestroke_signature() != before
