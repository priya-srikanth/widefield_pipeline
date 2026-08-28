"""The pre-stroke decoder and encoder as STORED OBJECTS.

WHY THIS EXISTS AT ALL. Until 2026-08-27 nothing in this repo persisted a fitted frozen model:
`pooled_frozen_loso` fitted `_pipe()` inline every call, `pooled_frozen_encoder` fitted its Ridge
inline, and every JSON on the server held result numbers with no weights behind them. That is why the
contamination of 2026-08-26 was possible -- a model refitted every run has no identity to
interrogate, so "what were you trained on?" had no answer on disk. These tests pin the identity.
"""
import json
import types

import pytest

from wfield_local import frozen_models as F


def spec(**kw):
    base = dict(animal="PS99", kind="decoder", align="cue", source="locanmf",
                train_labels=["PS99_0101", "PS99_0102"], basis_id="abc123", n_features=380)
    base.update(kw)
    return F.make_spec(base.pop("animal"), base.pop("kind"), **base)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Local dir under tmp, server off. Without this the tests would read and WRITE the real store."""
    monkeypatch.setattr(F, "local_dir", lambda: tmp_path / "frozen")
    monkeypatch.setattr(F, "server_dir", lambda: None)
    monkeypatch.setattr(F, "_session_sig", lambda lab: f"sig::{lab}")


# ---------------------------------------------------------------------------------------------
# IDENTITY -- what must change the id, and what must not.
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"align": "lick"}, {"source": "roi"}, {"basis_id": "def456"}, {"post_s": 1.0},
    {"train_labels": ["PS99_0101", "PS99_0102", "PS99_0103"]},
    {"train_labels": ["PS99_0101"]},
])
def test_things_that_must_change_the_id(kw):
    """A new PRE-STROKE session must mint a NEW model rather than move this one. The basis matters
    because ROI and joint features are different SPACES -- 264 columns against 380 -- and a weight
    vector over one is meaningless applied to the other."""
    assert F.spec_id(spec()) != F.spec_id(spec(**kw))


def test_the_id_ignores_training_session_ORDER():
    """Order is an accident of how the caller listed sessions; membership is the model."""
    a = spec(train_labels=["PS99_0101", "PS99_0102"])
    b = spec(train_labels=["PS99_0102", "PS99_0101"])
    assert F.spec_id(a) == F.spec_id(b)


def test_the_id_changes_when_an_INPUT_changes_under_an_unchanged_label(monkeypatch):
    """THE case that motivates signatures rather than label lists. On 2026-08-14 the switch to the
    meegkit_hpfit SVTcorr changed the underlying data while every label stayed identical; a model
    keyed on labels alone would have been served for data it was never fitted to."""
    before = F.spec_id(spec())
    monkeypatch.setattr(F, "_session_sig", lambda lab: f"REPROCESSED::{lab}")
    assert F.spec_id(spec()) != before


def test_trial_inclusion_is_NOT_part_of_the_identity():
    """Deliberate, and the thing most likely to be mis-assumed. Training always uses the pre-stroke
    ENGAGED trials; 'all', 'lick + miss-while-working' and 'lick only' select which POST-STROKE trials
    are pushed through the finished model. They are scoring-time populations, which is also what keeps
    the per-class confusions summable. So the spec carries no inclusion field at all."""
    assert not (set(spec()) & {"inclusion", "variant", "trials", "class"})


def test_decoder_and_encoder_are_different_models():
    assert F.spec_id(spec(kind="decoder")) != F.spec_id(spec(kind="encoder"))


# ---------------------------------------------------------------------------------------------
# THE FREEZE CONTRACT.
# ---------------------------------------------------------------------------------------------

def test_first_call_fits_and_stores_second_call_loads():
    calls = {"n": 0}

    def fit():
        calls["n"] += 1
        return {"full": "MODEL"}

    p1, s1 = F.load_or_fit(spec(), fit, log=lambda *a: None)
    p2, s2 = F.load_or_fit(spec(), fit, log=lambda *a: None)
    assert (s1, s2) == (F.STATUS_NEW, F.STATUS_HIT)
    assert p1 == p2 == {"full": "MODEL"}
    assert calls["n"] == 1, "a stored model was refitted"


def test_a_post_stroke_session_cannot_change_the_model():
    """THE property the whole change is for. The spec is built from the PRE-STROKE labels, so pooling
    more post-stroke days leaves the id -- and therefore the stored weights -- untouched."""
    sp = spec(train_labels=["PS99_0101", "PS99_0102"])
    F.load_or_fit(sp, lambda: {"full": "PRE-ONLY"}, log=lambda *a: None)
    again, status = F.load_or_fit(sp, lambda: {"full": "SHOULD NOT HAPPEN"}, log=lambda *a: None)
    assert status == F.STATUS_HIT and again == {"full": "PRE-ONLY"}


def test_a_changed_pre_stroke_set_is_REPORTED_and_the_old_model_is_left_alone():
    """A mismatch is never resolved silently. Refitting in place would move a reference that
    post-stroke results are already quoted against; reusing blindly would score today's data against
    a model built from data that is gone. So: new id, old one untouched, said out loud."""
    logged = []
    old = spec(train_labels=["PS99_0101", "PS99_0102"])
    F.load_or_fit(old, lambda: {"full": "OLD"}, log=logged.append)
    new = spec(train_labels=["PS99_0101", "PS99_0102", "PS99_0103"])
    payload, status = F.load_or_fit(new, lambda: {"full": "NEW"}, log=logged.append)

    assert status == F.STATUS_SPEC_CHANGED and payload == {"full": "NEW"}
    assert F.find(old)[0] is not None, "the old model was mutated or removed"
    assert F.load_or_fit(old, lambda: {"full": "X"}, log=lambda *a: None)[0] == {"full": "OLD"}
    said = " ".join(logged)
    assert "DIFFERENT pre-stroke input set" in said
    assert "PS99_0103" in said, "the message must name what changed, or it is not actionable"


def test_refreeze_supersedes_by_RENAMING_never_by_deleting():
    """The old weights are what an earlier deck was scored against, so they stay reachable, and the
    reason goes in the name -- a superseded model with no recorded reason is indistinguishable from a
    stray directory."""
    old = spec(train_labels=["PS99_0101", "PS99_0102"])
    F.load_or_fit(old, lambda: {"full": "OLD"}, log=lambda *a: None)
    new = spec(train_labels=["PS99_0101", "PS99_0103"])
    F.load_or_fit(new, lambda: {"full": "NEW"}, refreeze="recurated", log=lambda *a: None)

    names = [p.name for p in (F.local_dir() / "PS99").iterdir()]
    assert any(".SUPERSEDED_recurated_" in n for n in names), names
    assert any(n == F._slug(new) for n in names)


def test_the_manifest_records_what_it_was_trained_on():
    """A model whose provenance is only in this repo's history is the state the contamination hid in.
    It has to be readable off the artifact itself."""
    sp = spec()
    F.load_or_fit(sp, lambda: {"full": "M"}, meta={"n_engaged": 42}, log=lambda *a: None)
    d, _ = F.find(sp)
    man = json.loads((d / "manifest.json").read_text())
    assert man["spec"]["train_labels"] == ["PS99_0101", "PS99_0102"]
    assert man["spec"]["basis_id"] == "abc123"
    assert man["spec_id"] == F.spec_id(sp)
    assert man["meta"]["n_engaged"] == 42
    assert man["frozen_utc"]


def test_listing_reports_stored_models():
    F.load_or_fit(spec(), lambda: {"full": "M"}, log=lambda *a: None)
    F.load_or_fit(spec(align="lick"), lambda: {"full": "M2"}, log=lambda *a: None)
    got = F.listing("PS99")
    assert {r["align"] for r in got} == {"cue", "lick"}
    assert all(r["n_train"] == 2 and r["kind"] == "decoder" for r in got)


def test_an_unreadable_stored_model_refits_rather_than_raising():
    """Publishing and writing are file operations; a truncated artifact must not take down a night."""
    sp = spec()
    F.load_or_fit(sp, lambda: {"full": "M"}, log=lambda *a: None)
    d, _ = F.find(sp)
    (d / "model.joblib").write_bytes(b"not a pickle")
    payload, status = F.load_or_fit(sp, lambda: {"full": "REFIT"}, log=lambda *a: None)
    assert payload == {"full": "REFIT"} and status in (F.STATUS_NEW, F.STATUS_SPEC_CHANGED)


def test_a_half_written_directory_is_not_a_candidate():
    """`find` keys on manifest.json, so a directory that exists without one -- mid-publish -- must not
    shadow a good model or be served as one."""
    sp = spec()
    d = F.local_dir() / "PS99" / F._slug(sp)
    d.mkdir(parents=True)
    assert F.find(sp)[0] is None
    payload, status = F.load_or_fit(sp, lambda: {"full": "M"}, log=lambda *a: None)
    assert status == F.STATUS_NEW and payload == {"full": "M"}


def test_a_model_cannot_be_built_from_another_animals_sessions():
    """The animal is the FIRST field of the identity, so a pooled-across-animals training set would
    produce a model filed under one animal's name that had seen another's data -- the exact shape of
    label-asserts-a-property-nothing-verifies this module exists to end. Checked, not assumed."""
    with pytest.raises(ValueError, match="PS98"):
        F.make_spec("PS99", "decoder", align="cue", source="locanmf",
                    train_labels=["PS99_0101", "PS98_0101"])
