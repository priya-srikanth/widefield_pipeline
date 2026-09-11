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


def test_template_path_dedicated_tree():
    from wfield_local.paths import PathResolver
    p = cs.template_path(PathResolver(machine="analysis"), "20260807", "PS94", "cam1")
    assert p.as_posix().endswith("Behavior_Cameras/Widefield/alignment_templates/cam1/PS94/20260807.npz")


def test_save_template_is_guarded():
    with pytest.raises(WriteGuardError):
        cs.save_template({"x": np.array([1])}, "N:/MICROSCOPE/Rich/data/x_daq_alignment.npz")


# --------------------------------------------------------------------- dropped-frame trimming
#
# PS92 2026-09-08 lost frames on all four cameras for the last ~18 of its 122 minutes -- perfect until
# t=6270 s, then 73-86% loss. Fitted over the whole recording that gave 6.4 / 7.7 / 9.7 / 13.6 ms
# residual (cam4 failing the gate) where its untroubled sibling PS93 managed 1.15 ms the same morning.
# The tail was corrupting the 86% of the session that was fine.
#
# Damage is placed MID-RECORDING in these fixtures, not in the tail: on this synthetic the matcher
# only ever pairs the first ~57% of edges, so tail damage never reaches the fit and would test
# nothing. A mid-session burst is the harder case anyway.


def _drop_window(fid, ts, gpio, t0, t1, p=0.4, seed=1):
    """Randomly delete a fraction of rows inside a time window, as a bandwidth-starved camera does.

    Rows VANISH while frame_id keeps counting -- that is what makes drops detectable at all -- so a
    pulse's rising edge is still seen but its onset is quantised late by however many frames were
    lost just before it.
    """
    rng = np.random.default_rng(seed)
    t = (ts - ts[0]) / 1e9
    hit = (t >= t0) & (t < t1)
    keep = ~(hit & (rng.random(len(fid)) < p))
    return fid[keep], ts[keep], gpio[keep]


def test_trimming_is_a_NO_OP_on_a_recording_with_no_drops(tmp_path):
    """The property that makes this safe to enable for every session at once: with nothing to trim,
    the fit is exactly what it was before. Both passes are gated on n_frame_drops."""
    pt = _pulses()
    fid, ts, gpio = _cam_arrays(pt)
    t = _build(tmp_path, fid, ts, gpio, pt)
    assert t["n_frame_drops"] == 0
    assert t["drop_trim_applied"] is False
    assert t["n_fit_edges"] == t["n_matched"]
    assert t["resid_ms_rms"] == t["resid_ms_rms_all"]


def test_a_dropout_does_not_set_the_alignment_for_the_clean_part(tmp_path):
    """The PS92 case: most anchors good, a minority mis-paired. The trimmed fit must come back to the
    precision the recording would have had without the dropout."""
    pt = _pulses()
    fid, ts, gpio = _cam_arrays(pt)
    clean = _build(tmp_path, fid, ts, gpio, pt)

    got = _build(tmp_path, *_drop_window(fid, ts, gpio, 25.0, 45.0, p=0.5), pt, cam="cam2")
    assert got["n_frame_drops"] > 0, "the fixture must actually drop frames"
    assert got["drop_trim_applied"] is True
    assert got["resid_ms_rms_all"] > 100.0, "untrimmed, this fit is an ITI out"
    assert got["resid_ms_rms"] < 0.1 * got["resid_ms_rms_all"], "the bad anchors are still voting"
    assert got["resid_ms_rms"] < 2 * clean["resid_ms_rms"], "should recover near clean precision"
    assert got["quality_ok"] is True


def test_it_recovers_even_when_the_untrimmed_fit_is_off_by_a_whole_ITI(tmp_path):
    """A heavier dropout puts the naive fit at ~760 ms -- an inter-pulse interval, i.e. the matcher
    paired the wrong pulses. Trimming still returns ~1.5 ms."""
    pt = _pulses()
    fid, ts, gpio = _cam_arrays(pt)
    got = _build(tmp_path, *_drop_window(fid, ts, gpio, 25.0, 45.0, p=0.7), pt, cam="cam3")
    assert got["resid_ms_rms_all"] > 100.0
    assert got["resid_ms_rms"] < 5.0
    assert got["quality_ok"] is True


def test_neither_pass_will_whittle_the_fit_down_to_a_short_lever_arm():
    """The guard that matters most, and the reason this cannot quietly launder a bad recording.

    Trimming below ``min_clean_frac`` buys precision on a short span and loses the thing the span was
    measuring -- a slope fitted over 5 s of a 2 h recording extrapolates badly to the other 7195.
    Both passes refuse and leave ``quality_ok`` to report the truth.
    """
    # residual pass: most anchors mis-paired, so there is no honest majority to fall back to
    mct = np.linspace(0, 100, 40)
    mdt = 1.0000003 * mct + 12.0
    rng = np.random.default_rng(3)
    bad = rng.choice(40, 30, replace=False)
    mdt[bad] += rng.uniform(0.3, 0.6, 30)
    keep = cs.robust_fit_mask(mct, mdt, np.ones(40, bool))
    assert keep.sum() >= 0.5 * 40, "must not trim past half the anchors"

    # drop-rate pass: when no bin is clean enough, keep everything rather than fit on a handful
    fid = np.arange(1000, 1000 + 500)
    ts = (590_000_000_000 + np.arange(500) * int(1e9 / FPS)).astype(np.int64)
    mask, _frac = cs.clean_edge_mask(fid, ts, np.arange(0, 500, 25),
                                     bin_s=1.0, max_drop_pct=-1.0, min_clean_frac=0.5)
    assert mask.all()


def test_every_matched_edge_is_STILL_SAVED_when_the_fit_is_trimmed(tmp_path):
    """Trimming changes which anchors the affine is fitted on, not what the file records. Someone
    re-fitting differently must not find the excluded edges already discarded."""
    pt = _pulses()
    fid, ts, gpio = _cam_arrays(pt)
    t = _build(tmp_path, *_drop_window(fid, ts, gpio, 25.0, 45.0, p=0.7), pt)
    assert len(t["matched_cam_edge_sec"]) == t["n_matched"]
    assert int(t["matched_in_fit"].sum()) == t["n_fit_edges"] < t["n_matched"]
    assert t["fit_cam_sec_lo"] < t["fit_cam_sec_hi"]


def test_the_residual_pass_is_what_catches_anchors_OUTSIDE_the_damaged_stretch():
    """Why the drop-rate mask alone is not enough, stated as a unit test.

    Losing one GPIO edge re-pairs the ITI matcher globally, so a mis-paired anchor can sit in a part
    of the recording that dropped nothing. Here every anchor is in a 'clean' bin and one is an ITI
    out; the drop mask keeps it and the residual pass removes it.
    """
    mct = np.linspace(0, 100, 40)
    mdt = 1.0000003 * mct + 12.0
    mdt[17] += 0.45                                   # one anchor mis-paired by ~an ITI
    seed = np.ones(mct.size, bool)                    # drop-rate mask saw nothing wrong
    keep = cs.robust_fit_mask(mct, mdt, seed)
    assert not keep[17]
    assert keep.sum() == mct.size - 1
