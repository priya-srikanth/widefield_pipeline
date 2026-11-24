"""Lane E 11-24 (Priya): onsets / previous-lick lists on the movement-page path come from THE session lick table
(`session_licks.session_licks`: the fused four-camera train when selected), never from the cam4 licks_all.csv."""
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FIXED = ["iti_lick_timing_audit.py", "iti_dock_audit.py", "iti_lick_order.py", "lick_recovery_markers.py",
         "lick_code_subspace.py", "lick_cue_to_lick_extract.py", "lick_enl_angle_decode.py", "revup_neural.py",
         "revup_followup.py"]


@pytest.mark.parametrize("name", FIXED)
def test_no_licks_all_read(name):
    code = [ln for ln in (SCRIPTS / name).read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]
    assert not any("pd.read_csv(" in ln and "licks_all.csv" in ln for ln in code), name


def test_enlv_extract_onset_read_fixed():
    src = (SCRIPTS / "lick_enlv_extract.py").read_text(encoding="utf-8")
    assert 'usecols=["daq_on_s"])' not in src            # the onset-only read now goes through session_licks
