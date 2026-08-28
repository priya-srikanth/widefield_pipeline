

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


def test_discover_warns_and_skips_single_channel_mislabel(tmp_path, capsys):
    """A labcams single-channel MISLABEL (1_H_W) of a normal 2-channel session must NOT be dropped
    silently. On PS92 8/28 discovery returned "1 session(s)" with no reason, and the requested animal
    was quietly missing. Discovery now skips the 1-channel file (it is not a valid 2-channel movie by
    name) but PRINTS a warning naming the file + the rename that rescues it."""
    from wfield_local.preprocess import _discover

    date = "20260828"
    labcams = tmp_path / "labcams"
    bad = labcams / date / "PS92_20260828_143414" / "raw_widefield_data"
    good = labcams / date / "PS93_20260828_092518" / "raw_widefield_data"
    bad.mkdir(parents=True)
    good.mkdir(parents=True)
    (bad / "pco_edge_run000_00000000_1_460_480_uint16.dat").write_bytes(b"\x00" * 16)
    (good / "pco_edge_run000_00000000_2_460_480_uint16.dat").write_bytes(b"\x00" * 16)
    daq = tmp_path / "daq"
    daq.mkdir()

    out = _discover(date, str(labcams), str(daq))

    # only the genuine 2-channel session is discovered ...
    assert {s["animal"] for s in out} == {"PS93"}
    assert all(s["dims"] == "2_460_480" for s in out)
    # ... but the skipped one is announced, with the animal, its label, and the rescue rename.
    warn = capsys.readouterr().out
    assert "SKIPPED" in warn
    assert "PS92" in warn and "1_460_480" in warn and "2_460_480" in warn
