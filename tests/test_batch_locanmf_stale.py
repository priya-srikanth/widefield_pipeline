"""A LocaNMF fit OLDER than the SVTcorr it read is stale, and stale counts as missing.

Existence of ``<label>_locanmf_summary.json`` was the whole "already done" test. After PS92_0922 was
re-preprocessed for the head-offset bug (2026-09-28, docs/EXPERIMENT_ERRORS.md) that test would have
kept the decomposition fitted to the misaligned data and every downstream number with it. The rule
now: compare the summary's mtime with its input's; older -> move the fit aside (never delete) and refit.
"""
from __future__ import annotations

import os

from wfield_local.batch_locanmf import set_aside_stale, stale_input


def _touch(p, mtime):
    p.write_text("x")
    os.utime(p, (mtime, mtime))


def test_a_fit_newer_than_its_input_is_current(tmp_path):
    svt, summ = tmp_path / "SVTcorr.npy", tmp_path / "PS92_0922_locanmf_summary.json"
    _touch(svt, 1_000_000.0)
    _touch(summ, 1_000_100.0)
    assert stale_input(summ, svt) is None


def test_a_fit_older_than_its_input_is_stale_by_the_gap(tmp_path):
    svt, summ = tmp_path / "SVTcorr.npy", tmp_path / "PS92_0922_locanmf_summary.json"
    _touch(summ, 1_000_000.0)
    _touch(svt, 1_000_600.0)                    # re-preprocessed ten minutes after the fit
    assert stale_input(summ, svt) == 600.0


def test_two_seconds_of_slack_absorb_copy2_and_smb_mtime_granularity(tmp_path):
    svt, summ = tmp_path / "SVTcorr.npy", tmp_path / "s.json"
    _touch(summ, 1_000_000.0)
    _touch(svt, 1_000_001.5)
    assert stale_input(summ, svt) is None


def test_no_input_to_compare_against_trusts_the_fit(tmp_path):
    summ = tmp_path / "s.json"
    _touch(summ, 1_000_000.0)
    assert stale_input(summ, None) is None
    assert stale_input(summ, tmp_path / "missing.npy") is None


def test_set_aside_renames_beside_and_keeps_every_file(tmp_path):
    out = tmp_path / "locanmf_affine8v1_hemo_meegkit_hpfit"
    out.mkdir()
    (out / "PS92_0922_locanmf_C.npy").write_bytes(b"C")
    (out / "PS92_0922_locanmf_summary.json").write_text("{}")
    moved = set_aside_stale(out)
    assert not out.exists(), "the live directory is freed for the refit"
    assert moved.parent == tmp_path and moved.name.startswith(out.name + "_stale_")
    assert sorted(p.name for p in moved.iterdir()) == ["PS92_0922_locanmf_C.npy", "PS92_0922_locanmf_summary.json"]
    assert (moved / "PS92_0922_locanmf_C.npy").read_bytes() == b"C"
