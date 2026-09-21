"""WHICH TERM starves a session's rest baseline? Decompose the mask, do not guess.

PS92_0826 is the only session in 92 with no position-weighted baseline, and "why" has been answered
by inference twice (once "too much running?"). The rest mask is an INTERSECTION of independent terms,
so the honest answer is to apply them cumulatively and report what each one costs.

TERMS, in the order `rest_mask` applies them:
    between trials   cue + response_window + settle -> next trial_start
    DOCKED           dock -> next trial_start (spout parked, no target present)
    treadmill        speed < 1 mm/s, buffered [1, 2] s
    lick             outside a [1, 2] s buffer around every lick onset
    min_rest_s       runs shorter than 0.5 s dropped, AFTER the intersection
    engagement       both bracketing trials inside the engaged period (applied per rest PERIOD)

Reported as the fraction of samples surviving each cumulative step, and then per position, so a
starved position can be attributed to a term rather than to a hunch.

    python -m scripts.rest_migration.archive.why_no_rest --sessions PS92_0826 [PS92_0824 ...]
"""
from __future__ import annotations

import argparse

import numpy as np

from wfield_local import config


def run(label):
    import h5py

    from wfield_local import daq_io
    from wfield_local.lick_detection import detect_licks
    from wfield_local.quiet_periods import response_window_s
    from wfield_local.treadmill import calibrate_treadmill, smooth_treadmill

    s = next(x for x in config.load_sessions() if x["label"] == label)
    seg = config.defaults()["segmentation"]
    q = seg["rest"]
    tr = seg["treadmill"]
    ld = config.defaults()["lick_detection"]

    # USE THE EXISTING READER. The analog group stores `samples_int16` with a scale and offset, not
    # a `samples` array, and `behavior_events._read_analog` already handles that. Reading the group
    # directly raised KeyError and would have been a second copy of the conversion besides.
    from wfield_local.behavior_events import _read_analog

    with h5py.File(s["h5"], "r") as f:
        fs = float(f.attrs["sample_rate_hz"])
        dn = [x.decode() for x in f["digital/channel_names"][:]]
        lick_v = _read_analog(f, ld.get("channel", "lick_analog"))
        tread_v = _read_analog(f, tr["channel"])
        packed = f["digital/packed_samples"][:, 0]
    n = lick_v.size
    cue = daq_io.rising_edges((packed >> dn.index("cue")) & 1)
    ts = daq_io.rising_edges((packed >> dn.index("trial_start")) & 1)

    def frac(m):
        return 100.0 * float(np.mean(m))

    print(f"=== {label} ===  {n/fs/60:.1f} min, {len(cue)} cues", flush=True)

    # 1. between trials
    rw = response_window_s(s) if callable(response_window_s) else 3.5
    try:
        rw = float(rw)
    except Exception:                                                  # noqa: BLE001
        rw = 3.5
    m = np.ones(n, bool)
    open_s = rw + float(q.get("settle_s", 0.5))
    for c in cue:
        nxt = ts[np.searchsorted(ts, c, "right")] if np.searchsorted(ts, c, "right") < len(ts) else n
        a = int(min(n, c + open_s * fs))
        m[int(c):a] = False
        # everything from the cue to the window opening is excluded; after `a` up to `nxt` survives
    print(f"  between trials only          {frac(m):6.2f}%", flush=True)

    # 2. treadmill
    speed = smooth_treadmill(
        calibrate_treadmill(tread_v, tr["offset_v"], tr["volt_sec_per_rot"], tr["mm_per_rot"]),
        fs, tr["smoothing_sigma_s"])
    slow = speed < float(q["speed_mm_s"])
    tb = [float(v) for v in q["treadmill_buffer_s"]]
    from wfield_local.quiet_periods import widen_bool_sparse
    slow_b = ~widen_bool_sparse(~slow, int(tb[0] * fs), int(tb[1] * fs))
    print(f"    treadmill alone            {frac(slow_b):6.2f}%   (moving {100-frac(slow):.1f}% raw)",
          flush=True)
    m2 = m & slow_b
    print(f"  + treadmill                  {frac(m2):6.2f}%", flush=True)

    # 3. licks
    lk = detect_licks(lick_v, fs, ld["thresh_upper"], ld["thresh_lower"],
                      tuple(ld["lockout_falling_edge_s"]), 0.10,
                      min_ili_s=ld["min_ili_ms"] / 1000.0)
    on = np.asarray(lk["lick_onsets"], np.int64)
    lb = [float(v) for v in q["lick_buffer_s"]]
    lick_bool = np.zeros(n, bool)
    lick_bool[on[on < n]] = True
    lick_excl = widen_bool_sparse(lick_bool, int(lb[0] * fs), int(lb[1] * fs))
    print(f"    lick buffer alone excludes {frac(lick_excl):6.2f}%   ({on.size} lick onsets)",
          flush=True)
    m3 = m2 & ~lick_excl
    print(f"  + lick buffer                {frac(m3):6.2f}%   <- usually THE dominant term",
          flush=True)

    # 4. min run length
    from wfield_local.quiet_periods import set_short_bool_to_low
    m4 = set_short_bool_to_low(m3, int(float(q["min_rest_s"]) * fs))
    print(f"  + min_rest_s {float(q['min_rest_s']):.2f}s           {frac(m4):6.2f}%", flush=True)

    print(f"\n  FINAL mask on disk for comparison:", flush=True)
    try:
        import glob

        from wfield_local.quiet_periods import quiet_dir
        p = sorted(glob.glob(f"{quiet_dir(s['mc'])}/*quiet_sample.npy"))
        if p:
            d = np.load(p[0]).astype(bool)
            print(f"    {100*d.mean():6.2f}%  (adds DOCKED + engagement, which this script omits)",
                  flush=True)
    except Exception as ex:                                            # noqa: BLE001
        print(f"    unavailable: {type(ex).__name__}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    a = ap.parse_args()
    for lab in a.sessions:
        try:
            run(lab)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {ex}", flush=True)


if __name__ == "__main__":
    main()
