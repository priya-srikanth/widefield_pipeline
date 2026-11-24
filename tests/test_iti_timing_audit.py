"""iti_lick_timing_audit.onsets_and_gaps (lane E 11-24): onsets and previous licks come from THE session lick table
(fused four-camera train), never from cam4 licks_all.csv; no NaN gap."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import iti_lick_timing_audit as aud  # noqa: E402


def _table():
    # a fused table: lick 2 (t 10.3) was seen by a side camera only (cam4 never saw it); its peak is not a cam4 peak
    return pd.DataFrame({"daq_on_s": [9.95, 10.25, 12.90, 15.00], "daq_peak_s": [10.0, 10.3, 13.0, 15.1]})


def test_fused_only_lick_gets_onset_and_previous_lick_counts():
    t = np.array([10.3, 13.0, 15.1])
    to, gap = aud.onsets_and_gaps(t, _table(), behaviour_on=np.array([]))
    assert np.isfinite(gap).all()
    assert np.allclose(to, [10.25, 12.90, 15.00])
    assert gap[0] == pytest.approx(0.30)        # previous lick = the fused lick at 9.95
    assert gap[1] == pytest.approx(2.65)        # previous = the fused-only lick at 10.25 (cam4-only list would say 2.95)


def test_peak_float_roundtrip_is_tolerated_but_a_foreign_lick_refuses():
    t = np.array([13.0 + 2e-13])
    to, gap = aud.onsets_and_gaps(t, _table(), behaviour_on=np.array([]))
    assert np.isfinite(gap).all()
    with pytest.raises(ValueError, match="no row"):
        aud.onsets_and_gaps(np.array([11.0]), _table(), behaviour_on=np.array([]))
