"""session_poses guards (Priya, 2026-10-06, "build in the tests to ensure we don't double-count"): a lick listed
under two trials is refused; frames shared between two trials' windows are reported."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.session_poses import assert_no_double_events, read_index


def test_the_same_lick_under_two_trials_is_refused():
    with pytest.raises(ValueError, match="two trials"):
        assert_no_double_events([10.000, 10.002, 20.0], [1, 2, 2], what="licks")


def test_close_licks_in_one_trial_and_distinct_licks_pass():
    assert_no_double_events([10.000, 10.002, 10.160, 20.0], [1, 1, 1, 2])   # same trial: two real onsets
    assert_no_double_events([10.0, 10.1], [1, 2])                           # 100 ms apart: different licks


def test_read_index_reports_shared_frames(tmp_path, capsys):
    pd.DataFrame({"trial_k": [0, 0, 1, 1], "src_frame": [5, 6, 6, 7]}).to_csv(tmp_path / "windows_index.csv",
                                                                             index=False)
    assert len(read_index(tmp_path)) == 4
    assert "1 video frames" in capsys.readouterr().out
