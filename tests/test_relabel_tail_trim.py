"""A DAT longer than the DAQ exposure record: where do the extra frames go?

The DAQ recorder and labcams are started and stopped BY HAND as two processes, so the camera can
be running before the DAQ starts (excess frames at the HEAD) or after it stops (at the TAIL). The
counts cannot tell which, and the choice decides the mapping of every frame in the session.

1a68c7c (2026-09-22) assumed the tail -- "camera outran the DAQ recorder" -- and the first session
it ever ran on was PS92_0922, where the DAQ had been started 2.466 s AFTER the camera. Every trial
window read 154 frames too early, and 4.6% of frames landed in the wrong channel with the wrong
mean subtracted, which looked like a whole-cortex anti-phase vascular oscillation
(docs/EXPERIMENT_ERRORS.md, 2026-09-22). These tests pin the rule that replaced the guess: place
the excess by the camlog's per-frame LED record, or by an explicit head offset, or REFUSE.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local.trim_illuminated_labcams import (camlog_led_ids, camlog_path_for, find_dat_head_offset,
                                                   load_daq_labels)


def _write_daq(path, n_exposures, hiccup_at=()):
    """A minimal DAQ .h5 with `n_exposures` pco pulses, LEDs alternating 415/470 per pulse.

    Each exposure is a narrow strobe with a gap before the next (as the real DAQ records), so an
    exposure's label window never bleeds into a neighbour's. ``hiccup_at`` repeats the previous LED
    on those pulses -- the alternation hiccups every real session has ~150 of, and the thing that
    makes a head offset visible in the labels at all.
    """
    import h5py

    block, lead = 6, 2
    nsamp = lead + n_exposures * block
    pco = np.zeros(nsamp, dtype=np.uint16)
    led415 = np.zeros(nsamp, dtype=np.int16)
    led470 = np.zeros(nsamp, dtype=np.int16)
    seq = []
    cur = 0
    for i in range(n_exposures):
        if i not in hiccup_at:
            cur = i % 2 if not seq else 1 - seq[-1]
        seq.append(cur)
        base = lead + i * block
        pco[base] = 1
        (led415 if cur == 0 else led470)[base:base + 2] = 3
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate_hz"] = 5000.0
        f.create_dataset("digital/channel_names", data=[b"pco_exposure"])
        f.create_dataset("digital/packed_samples", data=pco.reshape(-1, 1))
        f.create_dataset("analog/channel_names", data=[b"led415_ttl", b"led470_ttl"])
        f.create_dataset("analog/samples_int16", data=np.stack([led415, led470], axis=1))
        f.create_dataset("analog/int16_scale_volts_per_count", data=np.array([1.0, 1.0]))
        f.create_dataset("analog/int16_offset_volts", data=np.array([0.0, 0.0]))
    return np.array(seq)                      # 0 = 415, 1 = 470, per pulse


def _write_camlog(path, per_frame_bin):
    """A labcams camlog whose #LED lines carry the camera's own per-frame LED id (5 = one LED, 6 = the other)."""
    lines = ["# Camera: pco_edge log file", "# Log header:frame_id,timestamp"]
    for i, b in enumerate(per_frame_bin):
        lines.append(f"#LED:{5 if b == 0 else 6},{i + 1},{16.0 * (i + 1):.1f}")
        lines.append(f"{i + 1},2026-09-22 12:55:{4 + i * 0.016:09.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_longer_dat_with_no_camlog_and_no_head_offset_is_REFUSED(tmp_path):
    """The guess that put PS92_0922 out of register is no longer available."""
    h5 = tmp_path / "daq.h5"
    _write_daq(h5, n_exposures=6)
    with pytest.raises(ValueError, match="no camlog"):
        load_daq_labels(h5, physical_frame_count=8, offset=None, led_threshold=None)


def test_excess_frames_at_the_HEAD_are_placed_by_the_camlog(tmp_path):
    """The PS92_0922 case in miniature: the camera wrote 2 frames before the DAQ began.

    Even head offset, so the bulk parity is right and only a hiccup exposes it -- exactly why the
    real session passed every existing check.
    """
    h5, cam = tmp_path / "daq.h5", tmp_path / "pco.camlog"
    daq_seq = _write_daq(h5, n_exposures=40, hiccup_at=(10, 25))
    head = 2
    # camera frames: `head` frames the DAQ never saw, then one per DAQ pulse, in the DAQ's sequence
    cam_seq = np.r_[[1, 0], daq_seq]
    _write_camlog(cam, cam_seq)

    labels, meta = load_daq_labels(h5, physical_frame_count=40 + head, offset=None, led_threshold=None, camlog=cam)

    assert meta["dat_head_dropped_before_daq"] == head
    assert meta["dat_tail_dropped_beyond_daq"] == 0
    assert len(labels) == 40 + head, "labels are per DAT physical frame"
    assert (labels[:head] == 0).all(), "head frames are unmonitored -> 0 -> dropped by the pairing"
    assert labels[head] == (415 if daq_seq[0] == 0 else 470), "frame `head` is DAQ pulse 0"
    assert meta["camlog_agreement_at_head"] == 1.0
    assert meta["camlog_agreement_at_zero"] < 1.0, "a head offset must be VISIBLE at 0 or the test proves nothing"
    assert meta["labels_dark"] == 0, "head frames must not be counted as dark"


def test_excess_frames_at_the_TAIL_are_trimmed_when_the_camlog_says_so(tmp_path):
    """The genuine 'camera kept writing after the DAQ stopped' case still works -- by evidence, not assumption."""
    h5, cam = tmp_path / "daq.h5", tmp_path / "pco.camlog"
    daq_seq = _write_daq(h5, n_exposures=40, hiccup_at=(12,))
    cam_seq = np.r_[daq_seq, [1, 0]]                       # 2 trailing frames with no pulse
    _write_camlog(cam, cam_seq)

    labels, meta = load_daq_labels(h5, physical_frame_count=42, offset=None, led_threshold=None, camlog=cam)

    assert meta["dat_head_dropped_before_daq"] == 0
    assert meta["dat_tail_dropped_beyond_daq"] == 2
    assert (labels[40:] == 0).all() and (labels[:40] != 0).all()


def test_an_explicit_head_offset_overrides_everything(tmp_path):
    h5 = tmp_path / "daq.h5"
    _write_daq(h5, n_exposures=6)
    labels, meta = load_daq_labels(h5, physical_frame_count=8, offset=None, led_threshold=None, head_offset=2)
    assert meta["dat_head_dropped_before_daq"] == 2 and meta["dat_tail_dropped_beyond_daq"] == 0
    assert (labels[:2] == 0).all() and (labels[2:] != 0).all()


def test_daq_covering_the_whole_dat_drops_nothing_and_needs_no_camlog(tmp_path):
    """The ordinary night (DAQ >= DAT): unchanged, and a missing camlog is not an error."""
    h5 = tmp_path / "daq.h5"
    _write_daq(h5, n_exposures=6)
    labels, meta = load_daq_labels(h5, physical_frame_count=6, offset=None, led_threshold=None)
    assert len(labels) == 6
    assert meta["dat_head_dropped_before_daq"] == 0 and meta["dat_tail_dropped_beyond_daq"] == 0
    assert meta["labels_415"] == 3 and meta["labels_470"] == 3


def test_find_dat_head_offset_recovers_the_shift_and_is_polarity_free():
    """Which LED id is 415 is not assumed; hiccups are what make the offset identifiable."""
    rng = np.random.default_rng(0)
    seq = np.zeros(5000, dtype=int)
    for i in range(1, 5000):
        seq[i] = seq[i - 1] if rng.random() < 0.01 else 1 - seq[i - 1]
    daq_labels = np.where(seq == 0, 415, 470)
    for h in (0, 3, 154):
        cam = np.r_[rng.integers(0, 2, h), seq]
        for flip in (False, True):                                 # LED id 5 may be either wavelength
            ids = np.where(cam == (1 if flip else 0), 5, 6)
            best, at_best, at_zero = find_dat_head_offset(ids, daq_labels, max_offset=200)
            assert best == h, (h, flip, best)
            assert at_best == 1.0
            if h:
                assert at_zero < 0.99


def test_camlog_beside_the_dat_is_found_by_stripping_the_shape_suffix(tmp_path):
    dat = tmp_path / "pco_edge_run000_00000000_2_460_480_uint16.dat"
    dat.write_bytes(b"")
    assert camlog_path_for(dat) is None
    cam = tmp_path / "pco_edge_run000_00000000.camlog"
    _write_camlog(cam, [0, 1, 0, 1])
    assert camlog_path_for(dat) == cam
    assert camlog_led_ids(cam).tolist() == [5, 6, 5, 6]


def test_corrected_frame_samples_honour_the_head_offset(tmp_path):
    """The one consumer that turns a frame index into a DAQ time must subtract the head."""
    from wfield_local.framemap_event_maps import _corrected_frame_samples

    pco = np.arange(0, 2000, 10)                                    # 200 pulses, pulse k at sample 10k
    pairs0 = np.array([154, 156, 158])                              # DAT frames of three pairs
    old = tmp_path / "old.npz"
    np.savez(old, original_frame_index_ch0=pairs0)                 # a pre-2026-09-28 map: head implied 0
    new = tmp_path / "new.npz"
    np.savez(new, original_frame_index_ch0=pairs0, dat_head_offset=np.int64(154))
    assert _corrected_frame_samples(old, pco, 0).tolist() == [1540, 1560, 1580]
    assert _corrected_frame_samples(new, pco, 0).tolist() == [0, 20, 40], "frame 154 was exposed on pulse 0"
