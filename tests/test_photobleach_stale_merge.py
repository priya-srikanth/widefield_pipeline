"""Merging must not republish a record whose raw no longer exists.

Merging itself is correct and stays: it is what stops two machines splitting a date from clobbering
each other's animals (the 8/11 summary once ended up with a single animal in it).

What it did not do is ask whether the records it carries forward are still true. On 2026-08-21 the
8/20 summary was rebuilt after PS92's re-upload, and the log said:

    [photobleach] merging 1 session(s) from disk: ['PS95_0820']

That record was PS95 at -146% drift, computed from an upload that had been quarantined hours
earlier and no longer existed anywhere. The summary rendered perfectly and published it as current.

A re-upload is the hard case: it has the SAME byte count as the file it replaces, because it is the
same recording. Only the mtime separates them.
"""
import json
import os

import pytest

from wfield_local import photobleach as pb


def _rec(dat, **kw):
    st = os.stat(dat)
    r = {"label": "PS95_0820", "dat": str(dat), "n_frames": 100,
         "dat_bytes": st.st_size, "dat_mtime": st.st_mtime}
    r.update(kw)
    return r


def test_a_record_whose_raw_is_gone_is_not_merged(tmp_path):
    """The exact 2026-08-21 case: the raw was quarantined, the record stayed behind."""
    dat = tmp_path / "raw.dat"
    dat.write_bytes(b"x" * 100)
    rec = _rec(dat)
    dat.unlink()
    assert pb._still_describes_its_raw(rec) is False


def test_a_record_matching_its_raw_is_merged(tmp_path):
    dat = tmp_path / "raw.dat"
    dat.write_bytes(b"x" * 100)
    assert pb._still_describes_its_raw(_rec(dat)) is True


def test_a_reupload_of_the_same_size_is_detected_by_mtime(tmp_path):
    """The hard case. A re-upload is the same recording, so the byte count is identical; the mtime
    is the only thing that moves. Size alone would call this record current."""
    dat = tmp_path / "raw.dat"
    dat.write_bytes(b"x" * 100)
    rec = _rec(dat)
    os.utime(dat, (rec["dat_mtime"] + 7200, rec["dat_mtime"] + 7200))
    assert rec["dat_bytes"] == os.stat(dat).st_size, "same size -- size cannot tell these apart"
    assert pb._still_describes_its_raw(rec) is False


def test_a_truncated_or_regrown_raw_is_detected_by_size(tmp_path):
    dat = tmp_path / "raw.dat"
    dat.write_bytes(b"x" * 100)
    rec = _rec(dat)
    dat.write_bytes(b"x" * 60)
    assert pb._still_describes_its_raw(rec) is False


def test_a_small_mtime_jitter_does_not_drop_a_good_record(tmp_path):
    """Filesystems and shares round mtimes. A one-second wobble must not discard a valid record."""
    dat = tmp_path / "raw.dat"
    dat.write_bytes(b"x" * 100)
    rec = _rec(dat)
    rec["dat_mtime"] = rec["dat_mtime"] - 1
    assert pb._still_describes_its_raw(rec) is True


def test_a_legacy_record_without_a_fingerprint_is_accepted_if_the_raw_exists(tmp_path):
    """Records predating this check carry no fingerprint. Existence is weaker than a match, but it
    is strictly better than accepting them blind, and it must not delete an entire history."""
    dat = tmp_path / "raw.dat"
    dat.write_bytes(b"x" * 100)
    old = {"label": "PS94_0812", "dat": str(dat), "n_frames": 100}
    assert pb._still_describes_its_raw(old) is True


def test_a_record_with_no_dat_field_is_left_alone(tmp_path):
    assert pb._still_describes_its_raw({"label": "PS94_0812"}) is True
