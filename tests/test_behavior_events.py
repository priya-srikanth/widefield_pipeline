"""Tests for the canonical DAQ behavior-event producer (wfield_local.behavior_events)."""
import numpy as np
import pytest

from wfield_local import behavior_events as be, config

h5py = pytest.importorskip("h5py")


def _write_daq(path, *, fs=5000.0, dur_s=6.0, lick_times=(1.0, 1.2, 2.0, 3.0), reward_times=(1.05, 3.05)):
    """Minimal DAQ .h5: high-rest lick_analog with brief dips, flat treadmill, reward pulses."""
    n = int(fs * dur_s)
    lick = np.full(n, 5.0, np.float32)
    for t in lick_times:                          # 2 ms contact dips to 0 V
        s = int(t * fs)
        lick[s:s + int(0.002 * fs)] = 0.0
    tread = np.full(n, 1.2587643276652853, np.float32)      # == offset_v -> zero speed (all slow)
    reward = np.zeros(n, np.float32)
    for t in reward_times:
        s = int(t * fs)
        reward[s:s + int(0.01 * fs)] = 5.0
    # digital sync line (bit0): a pulse every ~0.4 s
    sync_col = np.zeros(n, np.uint8)
    for s in range(int(0.5 * fs), n, int(0.4 * fs)):
        sync_col[s:s + int(0.01 * fs)] = 1
    packed = sync_col.astype(np.uint8)      # sync on bit0
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate_hz"] = fs
        g = f.create_group("analog")
        g.create_dataset("channel_names", data=[b"lick_analog", b"treadmill", b"reward_ttl"])
        g.create_dataset("samples", data=np.stack([lick, tread, reward], axis=1))
        d = f.create_group("digital")
        d.create_dataset("channel_names", data=[b"sync", b"cue"])
        d.create_dataset("packed_samples", data=packed[:, None])
    return path


def test_compute_events_counts(tmp_path):
    # 30 s session so rest survives the trial windows and the lick buffers
    h5 = _write_daq(tmp_path / "PS92_20260806_000000.h5", dur_s=30.0)
    ev = be.compute_events(h5)
    assert ev["fs"] == 5000.0 and ev["n_samples"] == 150000
    assert ev["lick_onsets"].size == 4              # 4 dips, all > 40 ms apart
    assert ev["reward_samples"].size == 2
    assert ev["running_starts"].size == 0           # flat treadmill -> no running
    assert ev["grooming_starts"].size == 0          # grooming off by default
    assert ev["quiet_starts"].size >= 1             # the rest tail after the buffers
    assert ev["sync_samples"].size >= 60            # ~0.4 s sync heartbeat over 30 s
    # v3 REDEFINED quiet/rest (trial-anchored, no reward buffer). The bump is what forces every
    # cached npz to recompute instead of serving the retired definition under the same array names.
    assert ev["schema_version"] == 3


def test_rest_and_quiet_are_the_same_arrays(tmp_path):
    """`rest_*` is the name the definition now carries; `quiet_*` stays for existing readers."""
    h5 = _write_daq(tmp_path / "PS92_20260806_000000.h5", dur_s=30.0)
    ev = be.compute_events(h5)
    assert ev["rest_starts"].tolist() == ev["quiet_starts"].tolist()
    assert ev["rest_stops"].tolist() == ev["quiet_stops"].tolist()


def test_rest_records_which_anchor_it_used(tmp_path):
    """A mask must state its own definition: three anchors are reachable and they differ.

    This fixture has a `cue` bit but no `trial_start` and no `spout_strobe`, which is the degenerate
    third case -- the trial can only be bounded by its own cue. It is legitimate, and it must be
    NAMED, because a session anchored this way is not comparable to one anchored on `trial_start`.
    """
    h5 = _write_daq(tmp_path / "PS92_20260806_000000.h5", dur_s=30.0)
    ev = be.compute_events(h5)
    assert "no trial_start" in str(ev["rest_anchor"])


def test_rest_excludes_trial_time(tmp_path):
    """The whole point of the redefinition: a sample inside a trial is never rest."""
    import numpy as np

    from wfield_local.quiet_periods import trial_exclusion

    h5 = _write_daq(tmp_path / "PS92_20260806_000000.h5", dur_s=30.0)
    ev = be.compute_events(h5)
    fs, n = float(ev["fs"]), int(ev["n_samples"])
    # rebuild the trial mask the same way `rest_mask` does, from the same cue times
    import h5py

    with h5py.File(h5, "r") as f:
        dn = [x.decode() for x in f["digital/channel_names"][:]]
        packed = f["digital/packed_samples"][:, 0]
    cue_bit = ((packed >> dn.index("cue")) & 1).astype(np.int8)
    cue_s = np.flatnonzero(np.diff(cue_bit, prepend=0) == 1) / fs
    in_trial, _note = trial_exclusion(n, fs, cue_s, None, None)
    rest = np.zeros(n, bool)
    for a, b in zip(ev["quiet_starts"], ev["quiet_stops"]):
        rest[int(a):int(b)] = True
    assert not (rest & in_trial).any(), "no rest sample may fall inside a trial window"


def test_min_ili_floor_applied_in_events(tmp_path):
    # two dips 20 ms apart (< 40 ms floor) collapse to one lick
    h5 = _write_daq(tmp_path / "PS93_20260806_000000.h5", lick_times=(1.0, 1.02, 2.0))
    ev = be.compute_events(h5)
    assert ev["lick_onsets"].size == 2              # the 20 ms double is floored out


def test_save_load_roundtrip(tmp_path):
    h5 = _write_daq(tmp_path / "PS94_20260806_000000.h5")
    ev = be.compute_events(h5)
    p = be.save_events(ev, tmp_path / "out" / "e.npz")
    back = be.load_events(p)
    assert np.array_equal(back["lick_onsets"], ev["lick_onsets"])
    assert back["fs"] == 5000.0 and isinstance(back["fs"], float)   # 0-d scalar unwrapped
    assert back["daq_h5"] == "PS94_20260806_000000.h5"
    assert be.load_events(tmp_path / "missing.npz") is None


def test_lick_onsets_s(tmp_path):
    h5 = _write_daq(tmp_path / "PS95_20260806_000000.h5", lick_times=(1.0, 2.0))
    ev = be.compute_events(h5)
    s = be.lick_onsets_s(ev)
    assert np.allclose(s, [1.0, 2.0], atol=0.002)


def test_get_or_compute_caches(tmp_path, monkeypatch):
    class _RV:
        def root(self, name):
            return str(tmp_path / "server")
    rv = _RV()
    # point the DAQ finder at our synthetic file
    daq_dir = tmp_path / "daqroot" / "20260806"
    daq_dir.mkdir(parents=True)
    _write_daq(daq_dir / "PS92_20260806_120000.h5")
    monkeypatch.setattr("wfield_local.spout_behavior._daq_h5_for",
                        lambda rv, a, d: (daq_dir / "PS92_20260806_120000.h5") if a == "PS92" else None)
    ev1 = be.get_or_compute(rv, "PS92", "20260806")
    assert ev1 is not None and be.events_path(rv, "PS92", "20260806").exists()
    # second call loads the cache (delete source -> still returns)
    (daq_dir / "PS92_20260806_120000.h5").unlink()
    ev2 = be.get_or_compute(rv, "PS92", "20260806")
    assert ev2 is not None and np.array_equal(ev1["lick_onsets"], ev2["lick_onsets"])
    assert be.get_or_compute(rv, "PS99", "20260806") is None       # no DAQ .h5 -> None


# ------------------------------------------------------------------------------------------------
# A RE-UPLOADED DAQ KEEPS ITS FILENAME, so the cached events' `daq_h5` field proves nothing.
#
# PS93_20260818_145123.h5 was re-recorded after a fault and re-uploaded under the same name on
# 2026-08-19. The events cached from the first copy stayed on disk and were still used: they recorded
# n_samples=38,005,000 against the replacement's 37,800,230, and one running bout began past the end
# of the file. Nothing failed. The frame map HAD been rebuilt from the corrected DAQ, so the running
# maps combined bout times from one recording with frame alignment from another -- the labelled
# "running" windows averaged 1.84 mm/s, below the 3 mm/s threshold that supposedly defined them, and
# the running-minus-quiet contrast collapsed to ~1/20 of every other session's.
# ------------------------------------------------------------------------------------------------

def test_cached_events_from_a_replaced_daq_are_rejected(tmp_path, capsys):
    import h5py
    import numpy as np

    from wfield_local import behavior_events as be

    h5 = tmp_path / "PS93_20260818_145123.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("analog/samples_int16", data=np.zeros((1000, 6), dtype=np.int16))

    assert be._matches_daq({"n_samples": 1000}, h5, "match")
    assert not be._matches_daq({"n_samples": 1041}, h5, "PS93_20260818")
    assert "STALE" in capsys.readouterr().out


def test_missing_or_unreadable_daq_does_not_invalidate_a_cache(tmp_path):
    """The guard must not turn an unrelated failure into a silent recompute of everything."""
    from wfield_local import behavior_events as be

    assert be._matches_daq({"n_samples": 123}, None, "no daq")
    assert be._matches_daq({"n_samples": 123}, tmp_path / "nope.h5", "absent")
    assert be._matches_daq({}, tmp_path / "nope.h5", "no n_samples recorded")
