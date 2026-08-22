"""The share-write guard must actually fire.

A guard nobody has seen refuse anything is indistinguishable from one that does not work -- which is
the failure mode that produced most of this week's silent bugs. These call the guarded operations
with a share path and assert they raise, and with a tmp path and assert they do not.
"""
import os
import pathlib
import shutil

import pytest


SHARE = "N:/MICROSCOPE/Priya/Widefield/labcams/20260820/PS92_20260820_120000"


def test_opening_a_share_path_for_write_is_refused():
    with pytest.raises(AssertionError, match="live data share"):
        open(SHARE + "/x.npy", "wb")


def test_mkdir_on_a_share_is_refused():
    with pytest.raises(AssertionError, match="live data share"):
        pathlib.Path(SHARE + "/motion_corrected").mkdir(parents=True, exist_ok=True)


def test_makedirs_on_a_share_is_refused():
    with pytest.raises(AssertionError, match="live data share"):
        os.makedirs(SHARE + "/motion_corrected", exist_ok=True)


def test_copytree_into_a_share_is_refused(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(AssertionError, match="live data share"):
        shutil.copytree(str(src), SHARE + "/results")


def test_rmtree_on_a_share_is_refused():
    with pytest.raises(AssertionError, match="live data share"):
        shutil.rmtree(SHARE)


def test_the_unc_form_is_caught_too():
    """N: is a mapping; the same share reached by UNC must not slip past."""
    unc = "//research.files.med.harvard.edu/Neurobio/MICROSCOPE/Priya/x.npy"
    with pytest.raises(AssertionError, match="live data share"):
        open(unc, "wb")


def test_standby_is_caught_too():
    with pytest.raises(AssertionError, match="live data share"):
        open("M:/Widefield/labcams/x.bin", "wb")


def test_reads_are_still_allowed(tmp_path):
    """Only writing had the side effect; reading a config or resolving a path stays fine."""
    f = tmp_path / "a.txt"
    f.write_text("ok")
    assert f.read_text() == "ok"


def test_tmp_paths_are_untouched(tmp_path):
    d = tmp_path / "roots" / "labcams" / "20260820"
    d.mkdir(parents=True)
    (d / "x.npy").write_bytes(b"x")
    assert (d / "x.npy").stat().st_size == 1
