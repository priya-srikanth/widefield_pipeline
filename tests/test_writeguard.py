"""Tests for the MICROSCOPE/standby write-guard (wfield_local.writeguard)."""
import pytest

from wfield_local import writeguard
from wfield_local.writeguard import assert_writable, is_writable, WriteGuardError


# --- legitimate pipeline destinations (must ALWAYS pass; the nightly writes only here) ---
ALLOWED = [
    # analysis box (M: = MICROSCOPE) and imaging box (N: = MICROSCOPE), both under Priya
    r"M:\MICROSCOPE\Priya\Widefield\labcams\20260807\PS93_20260807_x\motion_corrected",
    "N:/MICROSCOPE/Priya/Widefield/labcams/20260807/PS93_x/motion_corrected/wfield_local_results",
    r"N:\MICROSCOPE\Priya\Widefield\labcams\xday\PS93_xall",
    "M:/MICROSCOPE/Priya/Widefield/labcams/locanmf_lick_pooled/cue_analysis/deck.pptx",
    r"\\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Priya\Widefield\labcams\x",
    # raw .dat archived to standby (contains /Priya/) — must NOT false-positive
    r"M:\collaborations\Priya\Widefield\labcams\20260807\PS93\motion_corrected\x_uint16.bin",
    r"\\standby.files.med.harvard.edu\hms\neurobio\sabatini\collaborations\Priya\Widefield\labcams\x.dat",
    # local drives — always allowed
    r"E:\labcams_data\20260807\PS93_x\raw_widefield_data\x_uint16.dat",
    r"D:\camera\20260807\cam1.csv",
    "C:/Users/sabatini/source/cue_lick/fig.png",
]

# --- foreign / path-bug targets on a recognized share (must be DENIED) ---
DENIED = [
    r"N:\MICROSCOPE\Rich\data\something",                 # another person's MICROSCOPE folder
    "M:/MICROSCOPE/SomeoneElse/x",
    r"\\research.files.med.harvard.edu\Neurobio\MICROSCOPE\x",   # share root (Priya segment dropped)
    "N:/MICROSCOPE/",
    # standby outside Priya
    r"\\standby.files.med.harvard.edu\hms\neurobio\sabatini\collaborations\OtherLab\x",
    r"\\standby.files.med.harvard.edu\hms\neurobio\sabatini\scratch\x",
]


@pytest.mark.parametrize("p", ALLOWED)
def test_allowed_paths_pass(p):
    assert_writable(p)          # must not raise
    assert is_writable(p)


@pytest.mark.parametrize("p", DENIED)
def test_denied_paths_raise(p):
    with pytest.raises(WriteGuardError):
        assert_writable(p)
    assert not is_writable(p)


def test_case_insensitive_and_empty():
    with pytest.raises(WriteGuardError):
        assert_writable("n:/microscope/RICH/data")   # case-insensitive match on the owner segment
    assert_writable("")                                # empty -> no-op (caller validates)
    assert_writable("M:/MICROSCOPE/priya/x")           # lowercase Priya also fine


def test_priya_substring_not_spoofed_by_similar_name():
    # a folder that merely starts with 'priya' (e.g. 'PriyaOld') must NOT count as the Priya subtree
    with pytest.raises(WriteGuardError):
        assert_writable("N:/MICROSCOPE/PriyaOld/x")
