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
    # 30 s session so quiet survives the 8 s post-reward buffer
    h5 = _write_daq(tmp_path / "PS92_20260806_000000.h5", dur_s=30.0)
    ev = be.compute_events(h5)
    assert ev["fs"] == 5000.0 and ev["n_samples"] == 150000
    assert ev["lick_onsets"].size == 4              # 4 dips, all > 40 ms apart
    assert ev["reward_samples"].size == 2
    assert ev["running_starts"].size == 0           # flat treadmill -> no running
    assert ev["grooming_starts"].size == 0          # grooming off by default
    assert ev["quiet_starts"].size >= 1             # the quiet tail after the buffers
    assert ev["sync_samples"].size >= 60            # ~0.4 s sync heartbeat over 30 s
    assert ev["schema_version"] == 2


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
