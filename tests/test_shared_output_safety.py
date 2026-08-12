"""Safeguards against two machines clobbering each other's SHARED, cross-animal outputs.

Real incident (2026-08-11): the imaging box (PS92/PS93) and this helper box (PS94/PS95) split one
date. Each rebuilt the date's shared photobleach summary from its OWN in-memory session list, so the
last writer replaced the file with a single animal's results. The same shape of bug lives in
``preprocess_deck.build_decks``, whose stale-sibling prune deletes decks a partial run did not write.
"""
from __future__ import annotations

import json
from pathlib import Path

from wfield_local import photobleach, writeguard


# --------------------------------------------------------------------------- writeguard helpers
def test_covers_all():
    assert writeguard.covers_all(["PS92", "PS93", "PS94", "PS95"], ["PS92", "PS95"])
    assert writeguard.covers_all(["PS92", "PS92"], ["PS92"])
    assert not writeguard.covers_all(["PS94", "PS95"], ["PS92", "PS93", "PS94", "PS95"])


def test_warn_if_partial_aggregate_flags_only_partial(capsys):
    full = writeguard.warn_if_partial_aggregate("x.png", ["PS92", "PS93"], ["PS92", "PS93"])
    assert full is False and capsys.readouterr().out == ""

    partial = writeguard.warn_if_partial_aggregate(
        "x.png", ["PS94", "PS95"], ["PS92", "PS93", "PS94", "PS95"], what="photobleach summary")
    out = capsys.readouterr().out
    assert partial is True
    assert "PARTIAL photobleach summary" in out
    assert "'PS92', 'PS93'" in out or "PS92" in out       # names what is missing
    assert "last writer WINS" in out                       # states the consequence


# --------------------------------------------------------------------------- photobleach merge
def _write_record(out_dir: Path, label: str, pct415=-12.0, pct470=-2.0, with_norm=True):
    """Write a per-session record exactly as analyze() persists it."""
    rec = {"label": label, "dat": f"{label}.dat", "daq": f"{label}.h5", "n_frames": 1000,
           "dur_min": 60.0, "roi_px": 5000,
           "channels": {"415": {"median": 30000.0, "pct": pct415, "per_min": -1.0,
                                "start": 1.0, "end": 0.9},
                        "470": {"median": 29000.0, "pct": pct470, "per_min": -0.2,
                                "start": 1.0, "end": 0.98}}}
    if with_norm:
        rec["_norm"] = {"415": ([0.0, 60.0], [1.0, 0.88]), "470": ([0.0, 60.0], [1.0, 0.98])}
    (out_dir / f"photobleach_{label}.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec


def test_load_records_reads_per_session_files(tmp_path):
    _write_record(tmp_path, "PS92_0811")
    _write_record(tmp_path, "PS95_0811")
    recs = photobleach.load_records(tmp_path)
    assert sorted(recs) == ["PS92_0811", "PS95_0811"]
    assert "_norm" in recs["PS92_0811"], "the trend traces must survive the round-trip"


def test_load_records_ignores_the_aggregate_and_recovers_legacy_entries(tmp_path):
    # a legacy date: only the aggregate exists (no per-session records, and it carries no _norm)
    (tmp_path / photobleach.AGG_JSON).write_text(json.dumps(
        [{"label": "PS93_0811", "channels": {"415": {"pct": -9.0}}, "dur_min": 61.0}]),
        encoding="utf-8")
    _write_record(tmp_path, "PS94_0811")
    recs = photobleach.load_records(tmp_path)
    assert sorted(recs) == ["PS93_0811", "PS94_0811"], "legacy aggregate entries are recovered"
    assert "_norm" not in recs["PS93_0811"]      # degrades gracefully: drift bar, no trend line
    assert "_norm" in recs["PS94_0811"]


def test_per_session_record_wins_over_the_aggregate(tmp_path):
    (tmp_path / photobleach.AGG_JSON).write_text(json.dumps(
        [{"label": "PS94_0811", "channels": {"415": {"pct": -99.0}}}]), encoding="utf-8")
    _write_record(tmp_path, "PS94_0811", pct415=-12.0)
    recs = photobleach.load_records(tmp_path)
    assert recs["PS94_0811"]["channels"]["415"]["pct"] == -12.0


def test_corrupt_record_is_skipped_not_fatal(tmp_path):
    _write_record(tmp_path, "PS95_0811")
    (tmp_path / "photobleach_PS92_0811.json").write_text("{not json", encoding="utf-8")
    recs = photobleach.load_records(tmp_path)
    assert sorted(recs) == ["PS95_0811"]


def test_rebuild_summary_repairs_a_clobbered_aggregate(tmp_path):
    """THE REGRESSION TEST for the 2026-08-11 incident.

    Simulate: box A analysed PS92/PS93, box B analysed PS94/PS95, and box B's run() replaced the
    aggregate with only its own two. Rebuilding from the per-session records on disk must restore
    all four -- with no raw .dat access and no recomputation.
    """
    for lab in ("PS92_0811", "PS93_0811", "PS94_0811", "PS95_0811"):
        _write_record(tmp_path, lab)
    (tmp_path / photobleach.AGG_JSON).write_text(json.dumps(
        [{"label": "PS94_0811", "channels": {}}, {"label": "PS95_0811", "channels": {}}]),
        encoding="utf-8")

    n = photobleach.rebuild_summary(tmp_path)

    assert n == 4
    agg = json.loads((tmp_path / photobleach.AGG_JSON).read_text(encoding="utf-8"))
    assert sorted(r["label"] for r in agg) == ["PS92_0811", "PS93_0811", "PS94_0811", "PS95_0811"]
    assert (tmp_path / "photobleach_SUMMARY.png").exists()
    assert all("_norm" not in r for r in agg), "the aggregate stays lightweight (_norm is stripped)"


# --------------------------------------------------------------------------- deck prune guard
def test_build_decks_does_not_prune_when_the_run_is_partial(tmp_path, monkeypatch):
    """A run covering only PS94/PS95 must NOT delete the PS92/PS93 decks the other box owns."""
    from wfield_local import preprocess_deck as pd

    monkeypatch.setattr(pd.config, "animals",
                        lambda: {"PS92": {}, "PS93": {}, "PS94": {}, "PS95": {}})
    labcams = tmp_path / "labcams"
    (labcams / "20260811").mkdir(parents=True)
    for a in ("PS92", "PS93"):                       # the other machine's decks, already on disk
        (labcams / f"cross-session_preprocessing_{a}.pptx").write_bytes(b"other-machine deck")

    sessions = [{"label": f"{a}_0811", "mc": str(labcams / "20260811" / f"{a}_20260811" / "motion_corrected")}
                for a in ("PS94", "PS95")]
    pd.build_decks(str(labcams / "cross-session_preprocessing.pptx"), sessions=sessions,
                   labcams_root=str(labcams), xday_root=str(tmp_path / "xday"),
                   resolver=None, machine=None, max_sessions=10, verbose=False)

    for a in ("PS92", "PS93"):
        p = labcams / f"cross-session_preprocessing_{a}.pptx"
        assert p.exists(), f"{a}'s deck was deleted by a run that never covered {a}"
        assert p.read_bytes() == b"other-machine deck", f"{a}'s deck was overwritten"
