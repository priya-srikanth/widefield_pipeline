"""The one place that reads the DAQ. These pin the distinctions that used to live in five copies.

Before 2026-08-19 the digital-bit unpack, the rising-edge scan, the 3-bit spout decode and the int16
analog scaling each existed in several modules. They agreed, but nothing MADE them agree, and the
failure mode is silent: a fix that reaches four of five copies yields figures that disagree with one
another while each looks correct alone.
"""
from __future__ import annotations

import numpy as np

from wfield_local import daq_io


# ------------------------------------------------------------------------------------------------
# The one genuine difference between the copies: whether a trace ALREADY HIGH at sample 0 counts.
# ------------------------------------------------------------------------------------------------

def test_a_digital_line_high_at_sample_zero_counts_as_an_edge():
    """The DAQ's digital lines idle low, so a high first sample means the pulse began exactly at the
    start of the recording. Dropping it loses a real cue/strobe/frame."""
    x = np.array([1, 1, 0, 0, 1, 0, 1])
    assert daq_io.rising_edges(x).tolist() == [0, 4, 6]


def test_a_derived_mask_high_at_sample_zero_does_not():
    """`quiet_periods` and `led_alternation_qc` threshold a CONTINUOUS signal. High at sample 0 there
    means the animal was already running (or the LED already on) when recording began -- that is not
    an onset, and counting it invents a bout with no beginning."""
    x = np.array([1, 1, 0, 0, 1, 0, 1])
    assert daq_io.rising_edges(x, include_first_sample=False).tolist() == [4, 6]


def test_the_threshold_form_matches_the_binary_form():
    sig = np.array([0.0, 3.0, 3.0, 0.1, 0.0, 4.0])
    assert (daq_io.rising_edges(sig, thr=0.5).tolist()
            == daq_io.rising_edges((sig > 0.5).astype(int)).tolist())


def test_the_two_callers_still_get_what_they_had():
    """quiet_periods and led_alternation_qc kept their own semantics through the consolidation."""
    from wfield_local.led_alternation_qc import _rising as led_rising
    from wfield_local.quiet_periods import _rising as quiet_rising

    assert quiet_rising(np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])).tolist() == [4, 7]
    assert led_rising(np.array([True, True, False, False, True, False, True])).tolist() == [4, 6]


# ------------------------------------------------------------------------------------------------
# The spout position code
# ------------------------------------------------------------------------------------------------

def test_strobe_codes_are_little_endian_over_the_three_bit_lines():
    names = ["spout_strobe", "spout_bit0", "spout_bit1", "spout_bit2"]
    bits = np.zeros((8, 4), dtype=np.uint8)
    bits[2, 1] = 1                      # code 1 at sample 2
    bits[4, 2] = 1                      # code 2 at sample 4
    bits[6, 1] = bits[6, 2] = bits[6, 3] = 1     # code 7 at sample 6
    got = daq_io.strobe_codes(bits, names, np.array([2, 4, 6]))
    assert got.tolist() == [1, 2, 7]
    assert got.dtype == np.int16


def test_strobe_codes_read_the_lines_at_the_strobe_not_around_it():
    """The firmware latches the code with the strobe. Sampling a neighbouring frame would silently
    pick up the NEXT trial's position on any trial where the spout had already begun moving."""
    names = ["spout_bit0", "spout_bit1", "spout_bit2"]
    bits = np.zeros((6, 3), dtype=np.uint8)
    bits[3, 0] = 1          # code 1 exactly at the strobe
    bits[4, 1] = 1          # a different code one sample later
    assert daq_io.strobe_codes(bits, names, np.array([3])).tolist() == [1]


# ------------------------------------------------------------------------------------------------
# Analog decoding
# ------------------------------------------------------------------------------------------------

def test_analog_channel_scales_int16_counts_into_volts(tmp_path):
    import h5py

    p = tmp_path / "d.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("analog/channel_names", data=[b"lick_analog", b"treadmill"])
        f.create_dataset("analog/samples_int16", data=np.array([[100, 0], [200, 0]], dtype=np.int16))
        f.create_dataset("analog/int16_scale_volts_per_count", data=np.array([0.01, 1.0]))
        f.create_dataset("analog/int16_offset_volts", data=np.array([0.5, 0.0]))
    with h5py.File(p, "r") as f:
        v = daq_io.analog_channel(f, "lick_analog")
        assert np.allclose(v, [1.5, 2.5])


def test_a_missing_channel_raises_unless_declared_optional(tmp_path):
    import h5py
    import pytest

    p = tmp_path / "d.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("analog/channel_names", data=[b"lick_analog"])
        f.create_dataset("analog/samples_int16", data=np.zeros((2, 1), dtype=np.int16))
        f.create_dataset("analog/int16_scale_volts_per_count", data=np.array([1.0]))
        f.create_dataset("analog/int16_offset_volts", data=np.array([0.0]))
    with h5py.File(p, "r") as f:
        with pytest.raises(ValueError, match="reward_ttl"):
            daq_io.analog_channel(f, "reward_ttl")
        # the trial table treats an absent reward line as missing, not fatal
        assert daq_io.analog_channel(f, "reward_ttl", required=False) is None
