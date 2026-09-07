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
    # ALL THREE boundaries, not just chronic. Normalised because `acute` round-trips as a list.
    def _norm(spec):
        out = {}
        for a, s in spec.items():
            out[a] = {k: (list(v) if isinstance(v, (list, tuple)) else v)
                      for k, v in s.items() if k in ("acute", "subacute_from", "chronic_from")}
        return out

    assert _norm(derived_now) == _norm(epochs.EPOCH_SPEC), (
        "derived boundaries have moved away from configs/animals.yaml; promote them so a pinned "
        f"rebuild or a fresh clone reproduces today's epochs: {derived_now} vs {epochs.EPOCH_SPEC}")


def test_the_first_run_does_not_announce_that_everything_moved():
    """No previous file means nothing to have moved FROM. Reporting every animal as changed on the
    first run -- or after the file is deleted or the share remounted -- would be false, and would
    train the reader to skip the one message that matters on the night it is true."""
    assert epoch_audit.changes({"PS92": {"chronic_from": 11}}, None) == []
    # an EMPTY previous file is different from an absent one: it really did lose a boundary
    assert epoch_audit.changes({"PS92": {"chronic_from": 11}}, {}) == [
        "PS92 chronic_from: absent -> 11"]


# --- the boundaries and the rule now come from configs/, not from Python -------------------------

def test_the_spec_comes_from_animals_yaml():
    """CLAUDE.md rule 3: `configs/*.yaml` is the single source of truth, and the hardcoded
    per-animal dicts were retired for this reason. The epoch boundaries are per-animal facts about
    the experiment and belong beside `stroke_date`, not in a Python literal."""
    from wfield_local import config

    assert epochs.EPOCH_SPEC == config.epoch_spec()
    for animal, spec in epochs.EPOCH_SPEC.items():
        assert isinstance(spec["acute"], tuple), (
            f"{animal}: acute must be a TUPLE -- YAML yields a list, and a list compares unequal "
            f"to an identical tuple, which would surface as a failure far from the loader")
        assert isinstance(spec["subacute_from"], int)
        assert spec["chronic_from"] is None or isinstance(spec["chronic_from"], int)


def test_the_rule_constants_come_from_defaults_yaml():
    """The thresholds chosen on 2026-09-07 are tunable without a code edit, and the reasoning for
    each sits in the YAML beside the value it justifies."""
    from wfield_local import config

    ep = config.defaults()["epochs"]
    assert epochs.RULE_POSITION == ep["rule_position"]
    assert epochs.ACUTE_FRACTION == pytest.approx(ep["acute_fraction"])
    assert epochs.CHRONIC_K_SD == pytest.approx(ep["chronic"]["k_sd"])
    assert epochs.CHRONIC_K_RES == pytest.approx(ep["chronic"]["k_res"])
    assert epochs.CHRONIC_MIN_TAIL == ep["chronic"]["min_tail"]
    assert epochs.CHRONIC_LEVEL_MIN == {k: pytest.approx(v)
                                        for k, v in ep["chronic"]["level_min"].items()}


def test_the_rule_text_is_built_from_the_constants():
    """`CHRONIC_RULE` is stamped into `epoch_boundaries.json` and the deck's section I divider as
    the rule a figure set claims to be the output of. A hand-written copy would go stale the first
    time a threshold was tuned in YAML, and would then assert something false in a published deck."""
    assert f"{epochs.CHRONIC_K_RES:.4g}" in epochs.CHRONIC_RULE
    assert f"{100 * epochs.CHRONIC_LEVEL_MIN['hit']:.0f}%" in epochs.CHRONIC_RULE
    assert epochs.RULE_POSITION in epochs.CHRONIC_RULE


def test_a_stale_fallback_is_reported_with_the_yaml_to_paste():
    """When behaviour moves a boundary, `animals.yaml` does NOT follow -- the pipeline must not edit
    a version-controlled file both machines push. But a promotion that requires re-deriving the
    number by hand is one that does not happen, so the lines to paste are produced ready."""
    crossed = {a: {"chronic_from": s["chronic_from"]} for a, s in epochs.EPOCH_SPEC.items()}
    crossed["PS93"] = {"chronic_from": 11}                    # PS93 crosses
    stale = epoch_audit.stale_fallback(crossed)
    assert len(stale) == 1 and "PS93" in stale[0], stale
    paste = "\n".join(epoch_audit.promotion_yaml(crossed))
    assert "PS93:" in paste and "chronic_from: 11" in paste
    # only the animal that moved...
    assert "PS92" not in paste
    # ...and every value is YAML-spelled, comments included: a "was None" beside a
    # "chronic_from: null" invites writing Python's spelling into a YAML file, where it parses as
    # the truthy STRING "None" rather than an absent boundary.
    assert "None" not in paste, paste
    unset = " ".join(epoch_audit.promotion_yaml({"PS92": {"chronic_from": None}}))
    assert "null" in unset and "None" not in unset


def test_nothing_is_stale_today():
    """The committed fallback matches what behaviour currently implies. Expected to fail when an
    animal crosses -- which is the prompt to promote, not a defect."""
    spec = {a: {"chronic_from": s["chronic_from"]} for a, s in epochs.EPOCH_SPEC.items()}
    assert epoch_audit.stale_fallback(spec) == []


def test_the_fallback_precedence(derived, monkeypatch):
    """PINNED > derived > LAST DERIVED FILE > animals.yaml, in that order.

    The third step is the one that is easy to get wrong and was documented wrongly for an hour: a
    crashed or behaviour-less run keeps the LAST DERIVED boundaries rather than reverting to the
    hand-declared seed. Reverting would mean a mount failure silently restaged every pooled panel
    to whatever was last promoted, which could be months old.
    """
    lab = "PS93_0828"
    assert epochs.EPOCH_SPEC["PS93"]["chronic_from"] is None      # the declared seed

    # no artifact -> animals.yaml
    epochs.clear_resolved()
    assert epochs.epoch_of(lab) == "subacute"

    # a previous run derived it -> that file wins, even though nothing derives this run
    epochs.save_boundaries({"PS93": {"chronic_from": 11}}, derived)
    epochs.clear_resolved()
    assert epochs.epoch_of(lab) == "chronic", "a failed run must keep the last derived boundaries"

    # pinned beats everything
    monkeypatch.setenv("WIDEFIELD_EPOCHS_PINNED", "1")
    epochs.clear_resolved()
    assert epochs.epoch_of(lab) == "subacute"


def test_acute_and_subacute_have_no_other_source(derived):
    """`chronic_from` is optional in `animals.yaml` -- it is derived. `acute` and `subacute_from`
    are NOT: nothing computes them, so removing them from the YAML would leave every post-stroke
    session unassigned. Pinned here so nobody 'tidies up' the epochs block on the theory that the
    pipeline derives all of it."""
    epochs.save_boundaries({a: {"chronic_from": None} for a in epochs.EPOCH_SPEC}, derived)
    epochs.clear_resolved()
    for animal, spec in epochs.EPOCH_SPEC.items():
        assert spec["acute"] and spec["subacute_from"], animal
    assert epochs.epoch_of("PS92_0818") == "acute"
    assert epochs.epoch_of("PS92_0828") == "subacute"      # chronic cleared -> falls to subacute


# --- acute and subacute are derived too, as of 2026-09-07 ---------------------------------------

def test_acute_is_derived_and_reproduces_the_declared_ranges():
    """Priya: "we have clear derivation definitions for acute and subacute right? like, we don't
    have to hard-code them?" -- correct. The rule reproduces all four declared ranges exactly, which
    is why switching to deriving them moved no published panel."""
    if not epoch_audit._cohort_path().exists():
        pytest.skip("no cohort behaviour table on this box")
    hit, _lick = epoch_audit.load_far_position_tables()
    got = epochs.derive_acute_boundaries(hit)
    for animal, declared in epochs.EPOCH_SPEC.items():
        assert got[animal]["acute"] == declared["acute"], animal
        assert got[animal]["subacute_from"] == declared["subacute_from"], animal


def test_subacute_is_the_first_session_after_acute_not_the_next_day():
    """Subacute has NO rule of its own. `subacute_from` is the first RECORDED session after the
    acute prefix -- PS92's day 6 and PS94's day 8 were simply not run, and reading the boundary as
    `acute_hi + 1` would place it on a day with no data."""
    from wfield_local import config
    for animal, spec in epochs.EPOCH_SPEC.items():
        days = sorted(d for d in (epochs.days_since_stroke(l)
                                  for l in config.pooled_labels(animal)
                                  if epochs.epoch_of(l) != "pre") if d is not None)
        after = [d for d in days if d > spec["acute"][1]]
        assert spec["subacute_from"] == after[0], animal


def test_a_relapse_is_reported_not_folded_into_the_acute_range():
    """`(lo, hi)` cannot express "acute, recovered, acute again". A naive (first, last) over all
    below-threshold days would silently relabel the recovered sessions between them as acute."""
    from wfield_local import config
    animal = "PS92"
    pre = [l for l in config.phase_labels("pre") if config.animal_of(l) == animal]
    post = sorted((l for l in config.pooled_labels(animal) if epochs.epoch_of(l) != "pre"),
                  key=lambda x: x.split("_")[-1])
    hit = {l: {epochs.RULE_POSITION: 1.0} for l in pre}
    for i, l in enumerate(post):
        hit[l] = {epochs.RULE_POSITION: 0.01 if i < 2 else 0.9}
    hit[post[-1]] = {epochs.RULE_POSITION: 0.01}                # relapses at the very end
    got = epochs.derive_acute_boundaries(hit)[animal]
    assert got["acute"] == (epochs.days_since_stroke(post[0]),
                            epochs.days_since_stroke(post[1])), "range must be the PREFIX only"
    assert got["relapse"] == [epochs.days_since_stroke(post[-1])]
    assert got["subacute_from"] == epochs.days_since_stroke(post[2])


def test_an_underivable_boundary_is_omitted_not_written_as_none():
    """`spec_for` merges key-by-key, so an omitted key falls back to the declared value. Writing
    None instead would assert "there is no acute epoch" -- a much stronger claim than "behaviour
    could not tell me", and one that unassigns every early post-stroke session."""
    rep = {"chronic": {"PS92": {"derived_day": 11}},
           "acute_derived": {"PS92": {"acute": None, "subacute_from": None, "relapse": []}}}
    spec = epoch_audit.derived_spec(rep)
    assert spec["PS92"] == {"chronic_from": 11}, spec
    assert "acute" not in spec["PS92"] and "subacute_from" not in spec["PS92"]


def test_a_derived_acute_range_survives_the_json_round_trip(derived):
    """JSON has no tuple. `acute` goes out as a list and must come back as a tuple, or it compares
    unequal to an identical declared value and every consumer disagrees for no visible reason."""
    epochs.save_boundaries({"PS92": {"acute": [1, 3], "subacute_from": 4, "chronic_from": 11}},
                           derived)
    epochs.clear_resolved()
    spec = epochs.spec_for("PS92")
    assert spec["acute"] == (1, 3) and isinstance(spec["acute"], tuple)
    assert epochs.epoch_of("PS92_0821") == "subacute"     # day 4, acute under the declared (1,5)
