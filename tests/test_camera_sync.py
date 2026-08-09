"""Tests for the camera<->DAQ sync alignment (wfield_local.camera_sync + frame_sync matcher)."""
import h5py
import numpy as np
import pytest

from wfield_local import camera_sync as cs
from wfield_local.frame_sync import align_edge_sequences, _norm_to_01
from wfield_local.writeguard import WriteGuardError

FS = 1000.0        # DAQ Hz
FPS = 100.0        # camera
HIGH_S = 0.05      # sync pulse high-phase (5 cam frames / 50 DAQ samples wide)


def _pulses(n=200):
    rng = np.random.default_rng(0)
    return np.cumsum(rng.uniform(0.25, 0.65, n)) + 0.5     # irregular bounded ITI


def _make_daq(path, pulse_t):
    n = int((pulse_t.max() + 1.0) * FS)
    packed = np.zeros(n, np.uint8)
    for t in pulse_t:
        i = int(round(t * FS))
        packed[i:i + int(HIGH_S * FS)] |= 1                # bit0 = sync
    with h5py.File(path, "w") as h:
        h.attrs["sample_rate_hz"] = FS
        dg = h.create_group("digital")
        dg.create_dataset("channel_names", data=np.array([b"sync", b"cue"], dtype="S12"))
        dg.create_dataset("packed_samples", data=packed.reshape(-1, 1))


def _cam_arrays(pulse_t, epoch_ns=590_000_000_000, skip_pulse=None):
    n = int((pulse_t.max() + 1.0) * FPS)
    fid = np.arange(1000, 1000 + n)
    ts = (epoch_ns + np.arange(n) * (1e9 / FPS)).astype(np.int64)
    gpio = np.full(n, 12, np.int64)                        # 12 = 0b1100, bit0 low
    for k, t in enumerate(pulse_t):
        if k == skip_pulse:
            continue                                       # this pulse leaves NO edge (count -1)
        f = int(round(t * FPS))
        gpio[f:f + int(HIGH_S * FPS)] |= 1                 # bit0 -> 13
    return fid, ts, gpio


def _write_cam(path, fid, ts, gpio):
    with open(path, "w") as fh:
        for a, b, c in zip(fid, ts, gpio):
            fh.write(f"{a},{b},{c}\n")


def _build(tmp_path, fid, ts, gpio, pulse_t, cam="cam1"):
    h5 = tmp_path / "PS94_20260101_000000.h5"
    csv = tmp_path / f"{cam}_2026-01-01T00_00_00.csv"
    _make_daq(h5, pulse_t)
    _write_cam(csv, fid, ts, gpio)
    return cs.build_template(h5, csv)


def test_exact_parity(tmp_path):
    pt = _pulses(200)
    t = _build(tmp_path, *_cam_arrays(pt), pt)
    assert t["n_daq_edges"] == 200 and t["n_cam_edges"] == 200 and t["n_frame_drops"] == 0
    assert t["cam"] == "cam1" and t["quality_ok"]
    assert abs(t["slope_daqSec_per_camSec"] - 1.0) < 1e-3 and t["resid_ms_rms"] < 5.0
    # matched cam-edge times map onto their daq-edge times (<10 ms)
    mapped = cs.cam_seconds_to_daq_seconds(t, t["matched_cam_edge_sec"])
    assert np.abs(mapped - t["matched_daq_edge_sec"]).max() < 0.01


def test_dropped_frame_in_an_ITI_is_robust_and_reported(tmp_path):
    """A frame dropped between pulses (GPIO low): no edge lost, mapping rides through, gap reported."""
    pt = _pulses(200)
    fid, ts, gpio = _cam_arrays(pt)
    low = np.flatnonzero((gpio & 1) == 0)                  # an ITI frame, well away from any edge
    r = int(low[low.size // 2])
    fid, ts, gpio = np.delete(fid, r), np.delete(ts, r), np.delete(gpio, r)
    t = _build(tmp_path, fid, ts, gpio, pt, cam="cam2")
    assert t["n_frame_drops"] == 1          # frame_id contiguity flags the missing frame
    assert t["n_cam_edges"] == 200          # edge count UNCHANGED — the drop carried no edge
    assert t["quality_ok"] and t["resid_ms_rms"] < 5.0     # timestamp-based map is unaffected


def test_dropped_frame_during_a_pulse_keeps_the_edge(tmp_path):
    """A frame dropped mid-pulse (pulse spans 5 frames): the rising edge survives on adjacent frames."""
    pt = _pulses(200)
    fid, ts, gpio = _cam_arrays(pt)
    mid = np.flatnonzero((gpio[1:] & 1) & (gpio[:-1] & 1)) + 1   # high AND prev-high => not the leading edge
    r = int(mid[mid.size // 2])
    fid, ts, gpio = np.delete(fid, r), np.delete(ts, r), np.delete(gpio, r)
    t = _build(tmp_path, fid, ts, gpio, pt, cam="cam3")
    assert t["n_frame_drops"] == 1 and t["n_cam_edges"] == 200 and t["quality_ok"]


def test_dropped_sync_edge_uses_the_matcher(tmp_path):
    """A whole pulse missing on the camera (count mismatch) -> bounded-window matcher still aligns."""
    pt = _pulses(200)
    t = _build(tmp_path, *_cam_arrays(pt, skip_pulse=100), pt, cam="cam4")
    assert t["n_daq_edges"] == 200 and t["n_cam_edges"] == 199       # counts differ
    # bounded window skips ~2*window edges at each end, so ~118 of 200 match (real data: 12793/12875)
    assert t["n_matched"] >= 100 and t["quality_ok"] and t["resid_ms_rms"] < 5.0


def test_matcher_is_bounded_window_and_monotonic():
    s = np.cumsum(np.random.default_rng(1).uniform(0.25, 0.65, 200))
    s = _norm_to_01(s)
    i1, i2, d = align_edge_sequences(s, s)
    assert i1.size > 100 and np.array_equal(i1, i2)          # identical -> monotonic diagonal (~118 of 200)
    assert np.all(np.diff(i1) > 0) and np.all(np.diff(i2) > 0)
    with pytest.raises(ValueError):                          # too few edges (< window*4+5)
        align_edge_sequences(s[:50], s[:50])


def test_save_template_is_guarded():
    with pytest.raises(WriteGuardError):
        cs.save_template({"x": np.array([1])}, "N:/MICROSCOPE/Rich/data/x_daq_alignment.npz")
