

def test_match_daq_prefers_the_concat_h5_over_a_crash_segment():
    """A crashed day leaves THREE matching .h5: two segments and the joined one. Plain sorting picked
    the first segment -- 24 trials of PS92's 225-trial 8/12 -- which would silently score the day off
    a tenth of its data. sessions.yaml already declares the concat; discovery must agree."""
    from pathlib import Path

    from wfield_local.preprocess import _match_daq

    h5s = [Path("d/PS92_20260812_152628.h5"), Path("d/PS92_20260812_161728.h5"),
           Path("d/PS92_20260812_concat.h5")]
    assert _match_daq(h5s, "PS92", "20260812").name == "PS92_20260812_concat.h5"
    # and the ordinary single-file day is unchanged
    one = [Path("d/PS93_20260812_181632.h5")]
    assert _match_daq(one, "PS93", "20260812").name == "PS93_20260812_181632.h5"
    assert _match_daq(h5s, "PS94", "20260812") is None
