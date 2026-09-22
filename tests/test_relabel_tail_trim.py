"""The relabel aligns DAT physical frames to DAQ pco-exposure TTLs. The DAQ recorder and labcams are
started/stopped by hand as two separate processes, so the camera can keep writing a few trailing
frames AFTER the DAQ recorder stops -- the DAT then has MORE physical frames than the DAQ has exposure
edges. On PS92 20260922 that overran by 153 frames (0.034%) and the relabel raised
"No valid DAQ exposure-label offset for DAT frame count", failing the whole nightly. Those tail frames
carry no exposure TTL and cannot be timestamped, so the correct behaviour is to drop them, not crash.
"""
from __future__ import annotations

import numpy as np

from wfield_local.trim_illuminated_labcams import load_daq_labels


def _write_daq(path, n_exposures):
    """A minimal DAQ .h5 with `n_exposures` pco pulses, LEDs alternating 415/470 per pulse.

    Each exposure is a narrow strobe with a gap before the next (as the real DAQ records), so an
    exposure's label window never bleeds into a neighbour's LED: pco high for one sample, the
    matching LED high across the two samples at the strobe. Result: load_daq_labels sees exactly
    `n_exposures` clean alternating labels.
    """
    import h5py

    block, lead = 6, 2
    nsamp = lead + n_exposures * block
    pco = np.zeros(nsamp, dtype=np.uint16)
    led415 = np.zeros(nsamp, dtype=np.int16)
    led470 = np.zeros(nsamp, dtype=np.int16)
    for i in range(n_exposures):
        base = lead + i * block
        pco[base] = 1                          # narrow exposure strobe
        led = led415 if i % 2 == 0 else led470
        led[base:base + 2] = 3                 # ~3 V, only at the strobe
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate_hz"] = 5000.0
        f.create_dataset("digital/channel_names", data=[b"pco_exposure"])
        f.create_dataset("digital/packed_samples", data=pco.reshape(-1, 1))
        f.create_dataset("analog/channel_names", data=[b"led415_ttl", b"led470_ttl"])
        f.create_dataset("analog/samples_int16", data=np.stack([led415, led470], axis=1))
        f.create_dataset("analog/int16_scale_volts_per_count", data=np.array([1.0, 1.0]))
        f.create_dataset("analog/int16_offset_volts", data=np.array([0.0, 0.0]))


def test_dat_frames_past_the_daq_record_are_trimmed_not_fatal(tmp_path):
    """Camera outran the DAQ recorder: DAT has 8 physical frames, DAQ recorded only 6 exposures.
    The 2-frame unmonitored tail is dropped (labels cover the 6 DAQ-confirmed frames) and reported
    in meta -- it must NOT raise."""
    h5 = tmp_path / "daq.h5"
    _write_daq(h5, n_exposures=6)

    labels, meta = load_daq_labels(h5, physical_frame_count=8, offset=None, led_threshold=None)

    assert len(labels) == 6
    assert meta["daq_pco_exposure_count"] == 6
    assert meta["dat_physical_frame_count"] == 8
    assert meta["dat_tail_dropped_beyond_daq"] == 2
    assert meta["labels_both"] == 0 and meta["labels_dark"] == 0
    # the six covered frames are a clean 415/470 alternation
    assert meta["labels_415"] == 3 and meta["labels_470"] == 3


def test_daq_covering_the_whole_dat_drops_nothing(tmp_path):
    """The ordinary night (DAQ >= DAT) is unchanged: no tail dropped."""
    h5 = tmp_path / "daq.h5"
    _write_daq(h5, n_exposures=6)

    labels, meta = load_daq_labels(h5, physical_frame_count=6, offset=None, led_threshold=None)

    assert len(labels) == 6
    assert meta["dat_tail_dropped_beyond_daq"] == 0
