"""REST BOUT DURATIONS per candidate definition -- the state decoder's window length depends on it.

WHY THIS IS NOT OPTIONAL, and why it is a DESIGN question rather than a re-render.
`docs/BEHAVIOURAL_STATE_CONTROL.md` section 2 is titled "The window length is set by QUIET, not
chosen", and it means it literally:

    | quiet periods | n = 14,017 | median 1.10 s | p75 1.60 | p95 12.48 | max 370 |

    "A 2 s window discards 83% of quiet."

The 1 s segment -- the unit of the entire state-decoder analysis -- exists because quiet's median
bout was 1.10 s. Every candidate definition here changes which frames are quiet, so it changes that
distribution, and the derivation has to be re-run rather than inherited. If rest bouts lengthen, a
2 s window may become affordable, which would be a change to the design and not merely to its
numbers. If they shorten, 1 s may no longer be affordable either.

SUBSAMPLED, deliberately. The full 92-session pass costs ~70 minutes and is dominated by reading
analog channels off the share; a bout-length DISTRIBUTION does not need every session, and a
balanced draw across epochs is the honest way to summarise it -- the fraction of frames that are
rest is strongly epoch-dependent under the old definition, so an unbalanced sample would report the
epoch mix as if it were a property of the definition.
"""
from __future__ import annotations

import sys
import time

import h5py
import numpy as np

from wfield_local import config
from wfield_local import epoch_figures as ef
from wfield_local.daq_io import rising_edges
from wfield_local.grant_figures import _day
from wfield_local.locanmf_cue_lick_analysis import SESSIONS, _load_cue_events
from wfield_local.quiet_periods import (
    calibrate_treadmill,
    detect_licks,
    idx2bool,
    set_short_bool_to_low,
    smooth_treadmill,
    widen_bool_sparse,
)

PER_EPOCH = 6           # sessions sampled per epoch
RESP_WIN_S, POST_CUE_S, PRE_STROBE_S = 3.5, 4.0, 1.0
VARIANTS = ("A_reward8", "B_reward4", "C_noreward", "D_cue", "E_ITI")


def _bouts(b):
    """Lengths, in samples, of the True runs of a boolean array."""
    b = np.asarray(b, bool)
    if not b.any():
        return np.empty(0, int)
    d = np.diff(np.concatenate([[0], b.view(np.int8), [0]]))
    return np.flatnonzero(d < 0) - np.flatnonzero(d > 0)


def main():
    t0 = time.time()
    seg = config.defaults()["segmentation"]
    q, ld = seg["quiet"], config.defaults()["lick_detection"]
    want = set(config.phase_labels("pre") + config.phase_labels("post"))

    # ---- a BALANCED draw, deterministic: the first PER_EPOCH registered sessions of each epoch
    chosen, seen = [], {}
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
        seen[e] = seen.get(e, 0) + 1
        chosen.append((s, e))
    print(f"[bouts] {len(chosen)} sessions: {seen}", flush=True)

    dur = {v: [] for v in VARIANTS}
    for i, (s, _e) in enumerate(chosen, 1):
        try:
            with h5py.File(s["h5"], "r") as f:
                fs = float(f.attrs["sample_rate_hz"])
                names = [x.decode() for x in f["analog/channel_names"][:]]
                sc = f["analog/int16_scale_volts_per_count"][:]
                of = f["analog/int16_offset_volts"][:]
                raw = f["analog/samples_int16"][:]      # ONE contiguous read; see quiet_variants.py
                n = f["digital/packed_samples"].shape[0]

            def ac(nm):
                j = names.index(nm)
                return raw[:, j].astype(np.float32) * sc[j] + of[j]

            tread_v, lick_v, reward_v = ac("treadmill"), ac("lick_analog"), ac("reward_ttl")
            del raw
            cue = _load_cue_events(s["h5"])
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
            continue

        def wid(b, buf):
            return widen_bool_sparse(b, int(buf[0] * fs), int(buf[1] * fs))

        tr = seg["treadmill"]
        speed = smooth_treadmill(
            calibrate_treadmill(tread_v, tr["offset_v"], tr["volt_sec_per_rot"], tr["mm_per_rot"]),
            fs, tr["smoothing_sigma_s"])
        lick = detect_licks(lick_v, fs, ld["thresh_upper"], ld["thresh_lower"],
                            tuple(ld["lockout_falling_edge_s"]), 0.10,
                            min_ili_s=ld["min_ili_ms"] / 1000.0)
        lick_b = idx2bool(np.asarray(lick["lick_onsets"], np.int64), n)
        reward_b = idx2bool(rising_edges(reward_v, thr=2.5, include_first_sample=False), n)

        cs = np.asarray(cue["cue_samples"], float)
        ss = np.asarray(cue["strobe_samples"], float)
        sr = float(cue["sample_rate_hz"])
        j = np.searchsorted(ss, cs, side="right") - 1
        ok = j >= 0
        cue_b, trial_b = np.zeros(n, bool), np.zeros(n, bool)
        for c0 in cs:
            a, b = int(max(0, c0 / sr * fs - 0.1 * fs)), int(min(n, c0 / sr * fs + POST_CUE_S * fs))
            if b > a:
                cue_b[a:b] = True
        for c0, st in zip(cs[ok], ss[j[ok]]):
            a = int(max(0, st / sr * fs - PRE_STROBE_S * fs))
            b = int(min(n, c0 / sr * fs + POST_CUE_S * fs))
            if b > a:
                trial_b[a:b] = True

        base = (~wid(~(speed < q["speed_mm_s"]), q["treadmill_buffer_s"])
                & ~wid(lick_b, q["lick_buffer_s"]))
        for k, v in (("A_reward8", base & ~wid(reward_b, [0.1, 8.0])),
                     ("B_reward4", base & ~wid(reward_b, [0.1, 4.0])),
                     ("C_noreward", base),
                     ("D_cue", base & ~cue_b),
                     ("E_ITI", base & ~trial_b)):
            v = set_short_bool_to_low(v, int(q["min_quiet_s"] * fs))
            dur[k].append(_bouts(v) / fs)
        print(f"  .. {i}/{len(chosen)} {s['label']} ({time.time() - t0:.0f}s)", flush=True)

    print("\nREST BOUT DURATION (s), pooled over the sampled sessions:")
    print(f"{'variant':<13}{'n bouts':>9}{'median':>9}{'p75':>8}{'p95':>8}{'max':>9}"
          f"{'>=1s':>8}{'>=2s':>8}")
    for v in VARIANTS:
        a = np.concatenate(dur[v]) if dur[v] else np.empty(0)
        if not a.size:
            print(f"{v:<13}{'--':>9}")
            continue
        print(f"{v:<13}{a.size:>9d}{np.median(a):>9.2f}{np.percentile(a, 75):>8.2f}"
              f"{np.percentile(a, 95):>8.2f}{a.max():>9.1f}"
              f"{float((a >= 1).mean()):>8.2f}{float((a >= 2).mean()):>8.2f}")
    print("\nThe last two columns are the fraction of bouts a 1 s / 2 s window fits inside --")
    print("the quantity BEHAVIOURAL_STATE_CONTROL.md used to reject 2 s (it discarded 83%).")
    print(f"\n[bouts] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
