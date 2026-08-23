"""The imaging frame-sync audit.

Answering "have there been dropped frames, and are we syncing to ensure alignment?" needs the
distinction the audit exists to hold: the PCO camera is ON the DAQ clock (every exposure raises
pco_exposure, so alignment is an index), while the Blackfly behaviour cameras use the Arduino
heartbeat and a fitted match. Reporting one as if it were the other is the failure mode.
"""
import json

from wfield_local import sync_audit as sa


def _summary(tmp_path, date, sess, daq, dat, offset=0, skipped=0, clean=None):
    d = tmp_path / date / sess / "motion_corrected"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pco_cleanpairs_summary.json").write_text(json.dumps({
        "daq_pco_exposure_count": daq, "dat_physical_frame_count": dat,
        "chosen_exposure_offset": offset, "skipped_illuminated_frames": skipped,
        "clean_pairs": clean if clean is not None else dat // 2}))
    return d


def test_the_ordinary_plus_one_is_not_flagged(tmp_path):
    """69 of 80 real sessions sit at +1: the last exposure fires and the recording stops before
    that frame is written. Flagging it would make the audit useless."""
    _summary(tmp_path, "20260820", "PS94_20260820_094411", 373834, 373833)
    a = sa.audit(sa.collect(tmp_path))
    assert a["n"] == 1 and not a["bad_delta"] and not a["shifted"]


def test_an_exact_match_is_not_flagged(tmp_path):
    _summary(tmp_path, "20260604", "PS92_20260604_120000", 204696, 204696)
    assert not sa.audit(sa.collect(tmp_path))["bad_delta"]


def test_a_real_gap_is_flagged(tmp_path):
    _summary(tmp_path, "20260820", "PS93_20260820_140838", 434658, 430000)
    bad = sa.audit(sa.collect(tmp_path))["bad_delta"]
    assert len(bad) == 1 and bad[0]["delta"] == 4658


def test_a_shifted_frame_map_is_flagged_even_when_the_counts_agree(tmp_path):
    """The counts matching does NOT mean the alignment was free: a nonzero offset means the map had
    to be slid to line the trains up, and every real session so far has needed zero."""
    _summary(tmp_path, "20260820", "PS95_20260820_113748", 355885, 355884, offset=7)
    a = sa.audit(sa.collect(tmp_path))
    assert not a["bad_delta"], "counts are fine"
    assert len(a["shifted"]) == 1 and a["shifted"][0]["exposure_offset"] == 7


def test_the_blue_led_session_shape_is_not_a_sync_failure(tmp_path):
    """PS95_0813 skips 119438 illuminated frames because the 415 channel is missing for the first
    ~32 min (docs/EXPERIMENT_ERRORS.md). Its exposure and frame counts still agree exactly, so the
    audit must stay quiet -- a pairing problem is not an alignment problem."""
    _summary(tmp_path, "20260813", "PS95_20260813_120000", 532220, 532220,
             skipped=119438, clean=206391)
    a = sa.audit(sa.collect(tmp_path))
    assert not a["bad_delta"] and not a["shifted"]


def test_labels_survive_collection(tmp_path):
    _summary(tmp_path, "20260820", "PS92_20260820_162451", 430569, 430568)
    assert sa.collect(tmp_path)[0]["label"] == "PS92_0820"


def test_a_session_without_both_counts_is_skipped(tmp_path):
    d = tmp_path / "20260820" / "PS92_x" / "motion_corrected"
    d.mkdir(parents=True)
    (d / "pco_cleanpairs_summary.json").write_text(json.dumps({"daq_pco_exposure_count": 10}))
    assert sa.collect(tmp_path) == []
