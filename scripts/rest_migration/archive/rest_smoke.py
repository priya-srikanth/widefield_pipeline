"""Smoke-test the REST mask on real sessions, against the retired definition and the predictions."""
from __future__ import annotations

import sys

import h5py
import numpy as np

from wfield_local import config, daq_io
from wfield_local import epoch_figures as ef
from wfield_local.grant_figures import _day
from wfield_local.lick_detection import detect_licks
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locomotor_state import MAX_SEGMENTS_PER_PERIOD, SEGMENT_S, segments
from wfield_local.quiet_periods import calibrate_treadmill, rest_mask, smooth_treadmill
from wfield_local.treadmill import bout_edges

PER_EPOCH = 2


def main():
    seg = config.defaults()["segmentation"]
    ld = config.defaults()["lick_detection"]
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    seen = {}
    print(f"{'session':<12}{'epoch':<10}{'anchor':<14}{'rest%':>7}{'bouts':>7}"
          f"{'med s':>7}{'segs':>7}{'capped':>8}")
    for s in SESSIONS:
        if s["label"] not in want or not s.get("h5"):
            continue
        an, mmdd = s["label"].split("_")
        d = _day(an, mmdd)
        if d is None:
            continue
        e = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
        if e is None or seen.get(e, 0) >= PER_EPOCH:
            continue
        try:
            with h5py.File(s["h5"], "r") as f:
                fs = float(f.attrs["sample_rate_hz"])
                dnames, bits = daq_io.digital_bits(f)
                tread_v = daq_io.analog_channel(f, "treadmill")
                lick_v = daq_io.analog_channel(f, "lick_analog")
            idx = {n: i for i, n in enumerate(dnames)}
            n = bits.shape[0]
            cue = daq_io.rising_edges(bits[:, idx["cue"]]) / fs
            ts = (daq_io.rising_edges(bits[:, idx["trial_start"]]) / fs
                  if "trial_start" in idx else np.empty(0))
            st = daq_io.rising_edges(bits[:, idx["spout_strobe"]]) / fs
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:60]}")
            continue
        seen[e] = seen.get(e, 0) + 1

        tr = seg["treadmill"]
        speed = smooth_treadmill(
            calibrate_treadmill(tread_v, tr["offset_v"], tr["volt_sec_per_rot"], tr["mm_per_rot"]),
            fs, tr["smoothing_sigma_s"])
        lk = detect_licks(lick_v, fs, thresh_upper=ld["thresh_upper"],
                          thresh_lower=ld["thresh_lower"],
                          lockout_s=tuple(ld["lockout_falling_edge_s"]),
                          min_ili_s=ld["min_ili_ms"] / 1000.0)
        rest, note = rest_mask(n, fs, speed, np.asarray(lk["lick_onsets"], np.int64),
                               cue, ts, st, session_dir=None)
        a, b = bout_edges(rest)
        dur = (b - a) / fs
        sgs, pid = segments(a, b, fs, seg_s=SEGMENT_S, cap=MAX_SEGMENTS_PER_PERIOD)
        per = np.bincount(pid, minlength=len(a)) if pid.size else np.zeros(len(a), int)
        capped = float(np.mean(per >= MAX_SEGMENTS_PER_PERIOD)) if len(a) else 0.0
        print(f"{s['label']:<12}{e:<10}{note:<14}{rest.mean() * 100:>6.1f}%{len(a):>7}"
              f"{(np.median(dur) if dur.size else 0):>7.2f}{sgs.size:>7}{capped:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
