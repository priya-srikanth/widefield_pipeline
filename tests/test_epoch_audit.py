"""The epoch audit runs every nightly, reports, and never reassigns.

Both verifiers existed and neither was called by anything until 2026-09-07 -- `epochs` was
checkable in principle and unchecked in practice. These tests are mostly about the ways a wired-in
check can still fail to be a check: reading the wrong column, blocking the deck, or being unable to
say "I did not run" as distinct from "nothing is wrong".
"""
import pandas as pd
import pytest

from wfield_local import epoch_audit, epochs

POS = epochs.RULE_POSITION


def _csv(tmp_path, rows):
    d = tmp_path / "cohort"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "cohort_session_metrics.csv", index=False)
    return tmp_path


class _RV:
    def __init__(self, root):
        self._root = root

    def root(self, _name):
        return str(self._root)


def test_a_missing_table_is_not_agreement(tmp_path):
    """THE FAILURE THIS WHOLE MODULE GUARDS AGAINST. A box that has not run the behaviour stage
    must report "not checked", never "no disagreement" -- those look identical in a log and mean
    opposite things."""
    rep = epoch_audit.audit(rv=_RV(tmp_path))
    assert rep["available"] is False
    assert epoch_audit.disagreements(rep) == []          # nothing to report...
    text = " ".join(epoch_audit.report_lines(rep))
    assert "NOT CHECKED" in text                          # ...but it must SAY it did not check
    assert "agree" not in text.lower()


def test_it_reads_the_engaged_column_not_the_all_trial_one(tmp_path):
    """`hit__` is engagement-gated and `hitall__` is not, and they differ by up to 0.18 on PS94's
    post-stroke sessions. CHRONIC_RULE specifies engaged trials; reading the wrong column would
    audit a different quantity than the rule defines, and would still look like it worked."""
    rows = [{"animal": "PS92", "date": 20260606, "n_engaged": 300,
             f"hit__{POS}": 0.90, f"hitall__{POS}": 0.10, f"lpt__{POS}": 5.0}]
    hit, lick = epoch_audit.load_far_position_tables(rv=_RV(_csv(tmp_path, rows)))
    assert hit["PS92_0606"][POS] == 0.90
    assert lick["PS92_0606"][POS] == 5.0


def test_duplicate_sessions_on_one_date_take_the_larger(tmp_path):
    """A label is a DATE, and PS94 has two sessions on 0604. Without a rule the survivor depends on
    row order, so the audit could change verdict between runs on identical inputs."""
    rows = [{"animal": "PS94", "date": 20260604, "n_engaged": 100,
             f"hit__{POS}": 0.10, f"lpt__{POS}": 1.0},
            {"animal": "PS94", "date": 20260604, "n_engaged": 400,
             f"hit__{POS}": 0.90, f"lpt__{POS}": 9.0}]
    hit, _lick = epoch_audit.load_far_position_tables(rv=_RV(_csv(tmp_path, rows)))
    assert hit["PS94_0604"][POS] == 0.90
    # and the order it is written in must not matter
    hit2, _ = epoch_audit.load_far_position_tables(rv=_RV(_csv(tmp_path, rows[::-1])))
    assert hit2["PS94_0604"][POS] == 0.90


def test_a_disagreement_names_the_animal_and_the_boundary(tmp_path):
    """`disagreements` returns strings, not a boolean, so the nightly log says WHICH animal and
    WHICH boundary moved. A bare flag sends the reader back to a 20-hour log to find out."""
    from wfield_local import config
    rows = []
    for i, lab in enumerate(sorted(config.pooled_labels("PS94"), key=lambda s: s[-4:])):
        an, mmdd = lab.split("_")
        # PS94 looks fully recovered and perfectly stable from its first post-stroke session.
        # THE PRE-STROKE SESSIONS MUST SCATTER: every tolerance in the rule is a multiple of the
        # pre-stroke SD, so a flat baseline gives sd=0, no series can ever plateau, and the fixture
        # would "agree" for the wrong reason -- which is exactly how this test first passed.
        jitter = 0.03 * (1 if i % 2 else -1) if epochs.epoch_of(lab) == "pre" else 0.0
        rows.append({"animal": an, "date": int(f"2026{mmdd}"), "n_engaged": 500,
                     f"hit__{POS}": 0.90 + jitter, f"lpt__{POS}": 5.0 + 20 * jitter})
    rep = epoch_audit.audit(rv=_RV(_csv(tmp_path, rows)))
    bad = epoch_audit.disagreements(rep)
    assert any("PS94" in b and "chronic" in b for b in bad), bad
    # PS92 also disagrees here, and SHOULD: this fixture has no PS92 rows at all, and absent data
    # must never read as agreement (`test_a_missing_series_is_reported_not_guessed` in the epoch
    # tests pins the same contract one level down).
    assert any("PS92" in b for b in bad), bad
    # REPORTED, NOT APPLIED
    assert epochs.EPOCH_SPEC["PS94"]["chronic_from"] is None


def test_the_cli_exits_zero_even_when_behaviour_has_moved(monkeypatch, capsys):
    """A disagreement must NOT fail the step. `nightly_figs.cli` routes nonzero exits into
    FAILURES, and a run with failed steps refuses to publish the deck -- so a nonzero here would
    withhold an entire night's deck over the one outcome the audit exists to surface."""
    monkeypatch.setattr(epoch_audit, "audit", lambda **k: {
        "available": True, "position": POS, "n_sessions": 4, "csv": "x",
        "acute": {"PS94": {"agree": False, "disagreements": [
            {"label": "PS94_0825", "day": 9, "stored": "subacute", "derived": "acute"}]}},
        "chronic": {"PS94": {"agree": False, "derived_day": 9, "stored_day": None,
                             "hit": {"day": 9, "level": 0.95}, "licks": {"day": 9, "level": 0.9}}}})
    assert epoch_audit.main([]) == 0
    out = capsys.readouterr().out
    assert "DISAGREE" in out
    assert "NOT updated automatically" in out


def test_the_nightly_keeps_drift_out_of_the_failure_list(monkeypatch):
    """Drift goes to EPOCH_DRIFT and never to FAILURES, and a CRASH goes to FAILURES.

    The two halves are the point. Drift in FAILURES would block the deck; a crash NOT in FAILURES
    would let the audit silently stop running, which is how both verifiers came to be uncalled for
    ten days while reading as though they were wired in."""
    from wfield_local import nightly_figs as nf

    monkeypatch.setattr(nf, "FAILURES", [])
    monkeypatch.setattr(nf, "EPOCH_DRIFT", [])
    monkeypatch.setattr(epoch_audit, "audit", lambda **k: {"available": True, "position": POS,
                                                           "n_sessions": 1, "csv": "x",
                                                           "acute": {}, "chronic": {}})
    monkeypatch.setattr(epoch_audit, "disagreements", lambda r: ["PS93 chronic: 11 vs None"])
    nf._epoch_audit()
    assert nf.EPOCH_DRIFT == ["PS93 chronic: 11 vs None"]
    assert nf.FAILURES == [], "drift must not block the deck"

    def _boom(**k):
        raise RuntimeError("cohort table unreadable")

    monkeypatch.setattr(epoch_audit, "audit", _boom)
    nf._epoch_audit()
    assert any("epoch audit" in f for f in nf.FAILURES), "a crashed audit must be a failure"


@pytest.mark.skipif(not epoch_audit._cohort_path().exists(),
                    reason="no cohort behaviour table on this box")
def test_against_the_real_cohort_table_the_stored_spec_agrees():
    """The live check: today's behaviour still matches what `EPOCH_SPEC` says.

    This is expected to FAIL when an animal genuinely crosses a boundary, and that failure is the
    signal to edit `EPOCH_SPEC` -- deliberately not written as a tautology that can never fire."""
    rep = epoch_audit.audit()
    assert rep["available"] is True
    assert epoch_audit.disagreements(rep) == [], (
        "behaviour has moved away from EPOCH_SPEC; update the spec and rerun -- see "
        + "; ".join(epoch_audit.disagreements(rep)))
