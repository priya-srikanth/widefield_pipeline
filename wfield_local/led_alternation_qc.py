"""Which LED was on for each camera frame, from the DAQ — and where strict 470/415 alternation begins.

WHY THIS EXISTS. On 2026-08-13 PS95 was recorded with only the 470 nm LED armed for the first ~32 min;
415 alternation started later. That is not merely a period to exclude from analysis: the frame-pairing
step (``trim_illuminated_labcams`` / cleanpairs) infers frame identity from LED PARITY, so a ``.dat``
that is single-channel and then alternating can corrupt the frame map for the WHOLE session — a
half-frame offset propagating silently through everything downstream. The blue-only prefix has to be
split off BEFORE preprocessing, and this module says exactly where.

HOW. Every camera exposure raises DAQ digital ``pco_exposure``; the two LED TTLs (``led415_ttl``,
``led470_ttl``, analog) say which illumination was on. So for each exposure pulse we can read off which
LED fired, giving a per-frame label with no reliance on image content and no assumption of alternation.
Frame *i* of the ``.dat`` is exposure pulse *i*, so the answer is directly a ``.dat`` frame index.

The recommended split point starts on a **415** frame, because the pipeline pairs frames as
(415, 470) with ``functional_channel: 1`` — starting on the wrong parity would swap the channels for
the entire session, which looks like a plausible dataset rather than an error.

    python -m wfield_local.led_alternation_qc <daq.h5> [--plot-out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from wfield_local import daq_io

PCO_BIT_NAME, LED415, LED470 = "pco_exposure", "led415_ttl", "led470_ttl"


def _analog(f, name):
    names = [s.decode() for s in f["analog/channel_names"][:]]
    i = names.index(name)
    ds = f["analog/samples_int16"]
    x = ds[:, i].astype(np.float32)
    sc = np.asarray(f["analog/int16_scale_volts_per_count"])[i]
    of = np.asarray(f["analog/int16_offset_volts"])[i]
    return x * sc + of


def _digital_bit(f, name):
    names = [s.decode() for s in f["digital/channel_names"][:]]
    b = names.index(name)
    packed = f["digital/packed_samples"][:, 0]
    return ((packed >> b) & 1).astype(bool)


def _rising(b):
    # a mask that is already True at sample 0 is not an onset -- see daq_io.rising_edges
    return daq_io.rising_edges(b, include_first_sample=False)


def analyse(h5_path, thresh_v=2.5):
    """Per-frame LED identity and the alternation-onset frame index."""
    with h5py.File(str(h5_path), "r") as f:
        fs = float(f.attrs["sample_rate_hz"])
        pco = _digital_bit(f, PCO_BIT_NAME)
        v415 = _analog(f, LED415)
        v470 = _analog(f, LED470)

    starts = _rising(pco)
    ends = np.flatnonzero(pco[:-1] & (~pco[1:])) + 1
    if ends.size and starts.size and ends[0] < starts[0]:
        ends = ends[1:]
    n = min(starts.size, ends.size)
    starts, ends = starts[:n], ends[:n]

    # LED state DURING each exposure (max over the pulse: the TTL is high for the whole illumination)
    on415 = np.zeros(n, bool)
    on470 = np.zeros(n, bool)
    for i, (a, b) in enumerate(zip(starts, ends)):
        on415[i] = v415[a:b].max() > thresh_v
        on470[i] = v470[a:b].max() > thresh_v

    first415 = int(np.argmax(on415)) if on415.any() else -1
    out = {
        "h5": str(h5_path), "fs": fs, "n_frames": int(n),
        "duration_min": float(starts[-1] / fs / 60) if n else 0.0,
        "n_415": int(on415.sum()), "n_470": int(on470.sum()),
        "n_neither": int((~on415 & ~on470).sum()), "n_both": int((on415 & on470).sum()),
        "first_415_frame": first415,
        "first_415_time_s": float(starts[first415] / fs) if first415 >= 0 else None,
    }
    if first415 < 0:
        out["verdict"] = "NO 415 EXPOSURES AT ALL — session cannot be hemodynamically corrected"
        return out, on415, on470, starts

    # Is alternation strict after that point? Count ADJACENT REPEATS (lab[i]==lab[i+1]), not
    # mismatches against an ideal 0,1,0,1 phase. A single dropped or duplicated frame flips the phase
    # for everything after it, so the phase-comparison metric reports ~50% "violations" for one slip
    # and is useless for telling a healthy recording from a broken one. Repeats count the actual
    # events; `n_phase_slips` is the same number, named for what it does downstream.
    lab = np.where(on415, 0, np.where(on470, 1, -1))[first415:]
    repeats = np.flatnonzero(lab[:-1] == lab[1:])
    bad = int(repeats.size)
    out["alternating_after_first_415"] = bad == 0
    out["n_phase_slips_after"] = bad
    out["slip_frames"] = [int(first415 + i + 1) for i in repeats[:20]]
    out["longest_clean_run_frames"] = int(np.diff(np.r_[-1, repeats, lab.size - 1]).max())
    out["n_415_after"] = int((lab == 0).sum())
    out["n_470_after"] = int((lab == 1).sum())
    out["blue_only_prefix_frames"] = first415
    out["blue_only_prefix_min"] = float(starts[first415] / fs / 60)
    # the split must START on a 415 frame so pairs are (415, 470) as functional_channel=1 expects
    out["recommended_split_frame"] = first415
    out["recommended_split_parity_ok"] = bool(on415[first415])
    out["frames_kept_after_split"] = int(n - first415)
    out["verdict"] = ("clean alternation throughout" if first415 == 0 and bad == 0 else
                      f"SPLIT at frame {first415} ({out['blue_only_prefix_min']:.1f} min of "
                      f"single-channel prefix); {bad} phase slip(s) after")
    return out, on415, on470, starts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", nargs="+")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)

    allout = []
    for h in a.h5:
        out, on415, on470, starts = analyse(h)
        allout.append(out)
        print(f"\n=== {Path(h).name}")
        print(f"  {out['n_frames']} exposures over {out['duration_min']:.1f} min   "
              f"415={out['n_415']}  470={out['n_470']}  neither={out['n_neither']}  both={out['n_both']}")
        if out["first_415_frame"] >= 0:
            print(f"  first 415 exposure: frame {out['first_415_frame']} "
                  f"at {out['first_415_time_s']/60:.2f} min")
            print(f"  blue-only prefix: {out['blue_only_prefix_frames']} frames "
                  f"({out['blue_only_prefix_min']:.1f} min)")
            print(f"  alternation after: {out['alternating_after_first_415']}  "
                  f"{out['n_phase_slips_after']} phase slip(s); longest clean run "
                  f"{out['longest_clean_run_frames']} frames; "
                  f"415={out['n_415_after']} 470={out['n_470_after']} after the split")
            print(f"  -> SPLIT AT FRAME {out['recommended_split_frame']} "
                  f"(starts on 415: {out['recommended_split_parity_ok']}), "
                  f"keeping {out['frames_kept_after_split']} frames")
        print(f"  VERDICT: {out['verdict']}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(allout, indent=2, default=float))
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
