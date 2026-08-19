"""Reading the DAQ recorder ``.h5``: the ONE place that unpacks digital bits and decodes analog.

WHY THIS EXISTS. The same handful of operations -- unpack the packed digital samples into a bit
matrix, find rising edges, assemble the 3-bit spout code at each strobe, scale an int16 analog
channel into volts -- had five independent implementations:

    daq_trials.decode                        (behaviour's trial table)
    plot_spout_trial_averages._load_daq_events   (cue maps)
    plot_lick_aligned_averages._load_daq_events  (lick maps)
    plot_frame_alignment_comparison._load_daq_events
    plot_running_activity_maps._decode_analog_channel

They agreed, but nothing made them agree, and the failure mode is silent: a fix to the strobe decode
or the lick floor reaching four of five copies produces figures that disagree with each other while
every one of them looks right on its own. That is the same shape as the date literals written down in
five places -- correct when written, divergent the moment one is edited.

The callers keep their own signatures. They are genuinely different jobs (cue maps do not want licks;
the alignment QC wants an arbitrary channel list) and collapsing them into one function with eight
flags would trade a real duplication for a worse abstraction. What is shared is the low-level
reading, and only that lives here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wfield_local import config

#: The three digital lines carrying the spout position code, LSB first.
SPOUT_BIT_CHANNELS = ("spout_bit0", "spout_bit1", "spout_bit2")


def rising_edges(x, thr: float | None = None, include_first_sample: bool = True) -> np.ndarray:
    """Sample indices where a trace goes low -> high, as int64.

    ``thr`` thresholds a continuous signal first (``x > thr``); omit it for an already-binary line.

    ``include_first_sample`` is the one real difference between the copies this replaced, and it is
    a parameter rather than a second function because it encodes a genuine distinction:

      True  (digital lines) -- a line found ALREADY HIGH at sample 0 counts as an edge there. The
            DAQ's digital lines idle low, so a high first sample means the pulse began exactly at
            the start of the recording and dropping it would lose a real event.
      False (derived masks) -- ``quiet_periods`` and ``led_alternation_qc`` threshold a continuous
            signal, where "already high at sample 0" means the animal was ALREADY running (or the
            LED already on) when recording began. That is not an onset, and counting it as one would
            invent a bout with no beginning.
    """
    b = (np.asarray(x) > thr) if thr is not None else np.asarray(x)
    b = b.astype(np.int8)
    if include_first_sample:
        return np.flatnonzero(np.diff(b, prepend=0) == 1).astype(np.int64)
    return (np.flatnonzero(np.diff(b) == 1) + 1).astype(np.int64)


def digital_bits(f) -> tuple[list[str], np.ndarray]:
    """``(channel_names, bits)`` from an open DAQ file; ``bits`` is (n_samples, n_channels) uint8."""
    names = [n.decode() for n in f["digital/channel_names"][:]]
    packed = f["digital/packed_samples"][:, 0]
    return names, np.unpackbits(packed[:, None], axis=1, bitorder="little")[:, : len(names)]


def analog_channel(f, channel_name: str, required: bool = True) -> np.ndarray | None:
    """One analog channel in VOLTS, from int16 counts when that is how it was stored.

    ``required=False`` returns None for an absent channel instead of raising -- some sessions lack
    ``reward_ttl``, and the trial table treats that as missing rather than fatal.
    """
    names = [n.decode() for n in f["analog/channel_names"][:]]
    if channel_name not in names:
        if required:
            raise ValueError(f"Analog channel {channel_name!r} not found. Available: {names}")
        return None
    i = names.index(channel_name)
    if "samples_int16" in f["analog"]:
        raw = f["analog/samples_int16"][:, i]
        scale = float(f["analog/int16_scale_volts_per_count"][i])
        offset = float(f["analog/int16_offset_volts"][i])
        return raw.astype(np.float32) * scale + offset
    return np.asarray(f["analog/samples"][:, i], dtype=np.float32)


def strobe_codes(bits, names, strobe_samples) -> np.ndarray:
    """The 3-bit spout position code latched at each strobe edge.

    The firmware emits the code AFTER moving the spout and BEFORE the cue, so sampling the three bit
    lines at the strobe edge is what defines a trial's position. A dead ``spout_bit1`` (Aug 2026)
    collapses this to four distinct codes; that is detected and repaired downstream by
    ``behavior_position.classify_cues_with_backup``, not here -- this function reports what the
    hardware said.
    """
    idx = {n: i for i, n in enumerate(names)}
    b = [bits[:, idx[c]] for c in SPOUT_BIT_CHANNELS]
    return (b[0][strobe_samples].astype(np.int16)
            + 2 * b[1][strobe_samples].astype(np.int16)
            + 4 * b[2][strobe_samples].astype(np.int16))


def lick_onsets(lick_volts, fs, thresh_upper, thresh_lower, lockout_s, refractory_s=0.0):
    """Lick onsets in SAMPLES, with the pipeline-wide physiological floor applied.

    ``min_ili_ms`` (40 ms) is read from config here rather than at each call site, because a floor
    that some callers apply and others do not is not a floor. See CLAUDE.md.
    """
    from wfield_local.lick_detection import detect_licks

    det = detect_licks(lick_volts, fs, thresh_upper=thresh_upper, thresh_lower=thresh_lower,
                       lockout_s=tuple(lockout_s), refractory_s=refractory_s,
                       min_ili_s=config.defaults()["lick_detection"]["min_ili_ms"] / 1000.0)
    return np.asarray(det["lick_onsets"], dtype=np.int64), det


def session_attrs(f) -> tuple[float, str]:
    """``(sample_rate_hz, created_at)`` -- the two file attributes every reader carries through."""
    return float(f.attrs["sample_rate_hz"]), str(f.attrs["created_at"])


def open_daq(h5_path: Path):
    """``h5py.File`` in read mode. Wrapped so h5py is imported in one place."""
    import h5py

    return h5py.File(h5_path, "r")
