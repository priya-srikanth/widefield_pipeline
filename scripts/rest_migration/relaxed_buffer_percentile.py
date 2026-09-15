"""Can a LOWER PERCENTILE buy back what a RELAXED lick buffer gives away?

Priya, 2026-09-14: *"we relax the definition of the window a bit but make up for it by taking the
lower percentile."* The two halves are meant to work together -- a shorter buffer admits
lick-adjacent frames, and a percentile is what makes that affordable because it resists the
contamination those frames bring.

THE RISK THAT MAKES IT WORTH MEASURING RATHER THAN ARGUING. Licking is POSITION-DEPENDENT: near
positions are easier, so more hits, more reward, more consumption licking -- that is the measured
near/far survival gradient (all three far positions survive 15-18 points more than all three near,
at every epoch). So relaxing the buffer reintroduces POSITION-CORRELATED contamination into a
subtrahend that is supposed to be position-neutral, which is the specific thing the buffer exists to
prevent. A percentile trims a TAIL; it does not remove a systematic SHIFT.

THE MEASUREMENT, chosen because it asks the question directly rather than via maps. For each arm,
build a baseline PER POSITION from that position's rest frames, then report the SPREAD ACROSS THE SIX
positions, normalised by the session's own signal scale. A position-neutral subtrahend should have
small spread.

    relaxing widens the spread      -> contamination came in
    the percentile narrows it again -> the compensation works
    it does not                     -> the percentile trims the tail but not the shift

2x2 SO BUFFER AND ESTIMATOR NEVER MOVE TOGETHER -- the confound that invalidated `rest_vs_restw`
earlier today:

    A  median, [1, 2]     (current definition)
    B  median, [0.5, 1]   buffer effect ALONE
    C  p20,    [0.5, 1]   the proposal
    D  p20,    [1, 2]     percentile effect ALONE

Masks are recomputed IN MEMORY, never written, so no second variant lands on disk.

    python -m scripts.rest_migration.relaxed_buffer_percentile [--limit N] [--pct 20]
"""
from __future__ import annotations

import argparse

import numpy as np

from wfield_local import config

ARMS = (("A median [1,2]", (1.0, 2.0), None),
        ("B median [.5,1]", (0.5, 1.0), None),
        ("C p{p} [.5,1]", (0.5, 1.0), "pct"),
        ("D p{p} [1,2]", (1.0, 2.0), "pct"))


def _rest_mask_with_buffer(s, lick_buffer):
    """The rest mask recomputed with a different lick buffer, in memory. Returns a SAMPLE mask."""
    import copy

    import h5py

    from wfield_local import daq_io
    from wfield_local.behavior_events import _read_analog
    from wfield_local.docked_periods import behaviour_session_dir
    from wfield_local.lick_detection import detect_licks
    from wfield_local.plot_spout_trial_averages import _classify_cues
    from wfield_local.quiet_periods import rest_mask
    from wfield_local.treadmill import calibrate_treadmill, smooth_treadmill

    seg = config.defaults()["segmentation"]
    q = copy.deepcopy(seg["rest"])
    q["lick_buffer_s"] = list(lick_buffer)
    tr, ld = seg["treadmill"], config.defaults()["lick_detection"]

    with h5py.File(s["h5"], "r") as f:
        fs = float(f.attrs["sample_rate_hz"])
        dn = [x.decode() for x in f["digital/channel_names"][:]]
        lick_v = _read_analog(f, ld.get("channel", "lick_analog"))
        tread_v = _read_analog(f, tr["channel"])
        packed = f["digital/packed_samples"][:, 0]
    n = lick_v.size
    cue = daq_io.rising_edges((packed >> dn.index("cue")) & 1)
    ts = daq_io.rising_edges((packed >> dn.index("trial_start")) & 1)
    st = daq_io.rising_edges((packed >> dn.index("spout_strobe")) & 1)
    sync = daq_io.rising_edges((packed >> dn.index("sync")) & 1)
    speed = smooth_treadmill(
        calibrate_treadmill(tread_v, tr["offset_v"], tr["volt_sec_per_rot"], tr["mm_per_rot"]),
        fs, tr["smoothing_sigma_s"])
    lk = detect_licks(lick_v, fs, ld["thresh_upper"], ld["thresh_lower"],
                      tuple(ld["lockout_falling_edge_s"]), 0.10,
                      min_ili_s=ld["min_ili_ms"] / 1000.0)
    try:
        codes = np.asarray(_classify_cues(cue, st, daq_io.strobe_codes(packed, dn, st)))
    except Exception:                                                  # noqa: BLE001
        codes = None
    m, _note = rest_mask(n, fs, speed, np.asarray(lk["lick_onsets"], np.int64),
                         cue / fs, ts / fs, st / fs, params=q,
                         session_dir=behaviour_session_dir(s["label"]),
                         sync_s=sync / fs, position_codes=codes)
    return m, cue, codes, fs, packed, dn


def run(label, pct):
    from wfield_local.rest_by_position import frame_samples
    from wfield_local import daq_io

    s = next(x for x in config.load_sessions() if x["label"] == label)
    V = np.load(config.svtcorr_path(s["mc"]), mmap_mode="r")
    X = np.asarray(V[:, :], dtype=np.float64)
    T = X.shape[1]
    scale = float(np.sqrt((X ** 2).mean()))
    out = {}
    for name, buf, kind in ARMS:
        nm = name.format(p=pct)
        try:
            m, cue, codes, fs, packed, dn = _rest_mask_with_buffer(s, buf)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {label} {nm}: {type(ex).__name__} {str(ex)[:50]}", flush=True)
            return None
        pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
        f_of = frame_samples(s["mc"], s.get("fmdir"), s.get("regime"), pco)
        if f_of is None or codes is None:
            return None
        f_of = np.asarray(f_of)[:T]
        rest_fr = np.flatnonzero(m[np.clip(f_of, 0, m.size - 1)])
        # per-position rest frames, labelled by the bracketing trials (same rule as rest_by_position)
        per = {}
        cs = np.asarray(cue, np.int64)
        for fr in rest_fr:
            smp = f_of[fr]
            prev = np.searchsorted(cs, smp, "right") - 1
            nxt = prev + 1
            if prev < 0 or nxt >= len(codes) or codes[prev] != codes[nxt] or codes[prev] < 0:
                continue
            per.setdefault(int(codes[prev]), []).append(fr)
        per = {c: np.asarray(v) for c, v in per.items() if len(v) >= 50}
        if len(per) < 4:
            return None
        est = (lambda a: np.percentile(a, pct, axis=1)) if kind else (
            lambda a: np.median(a, axis=1))
        levels = np.stack([est(X[:, idx]) for c, idx in sorted(per.items())], 0)
        spread = float(np.sqrt(((levels - levels.mean(0)) ** 2).mean())) / scale
        frames = int(sum(v.size for v in per.values()))
        out[nm] = (spread, frames, len(per))
    line = f"  {label:12s}"
    for nm, (sp, fr, npos) in out.items():
        line += f"   {nm}: spread {sp:.4f} ({fr} fr, {npos}pos)"
    print(line, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--pct", type=float, default=20.0)
    a = ap.parse_args()
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    labs = [x["label"] for x in config.load_sessions() if x["label"] in want]
    if a.limit:
        labs = labs[: a.limit]
    rows = []
    for lab in labs:
        try:
            r = run(lab, a.pct)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
            continue
        if r:
            rows.append(r)
    if len(rows) < 4:
        print(f"\nonly {len(rows)} sessions -- a failed run, not a result")
        return
    keys = list(rows[0].keys())
    print(f"\nPOSITION SPREAD of the baseline ({len(rows)} sessions; lower = more position-neutral)")
    for k in keys:
        v = np.array([r[k][0] for r in rows if k in r])
        f = np.array([r[k][1] for r in rows if k in r], float)
        print(f"  {k:18s} median spread {np.median(v):.4f}   median frames {np.median(f):7.0f}")
    print("\nREAD: B vs A = what relaxing the buffer costs in position-neutrality.")
    print("      C vs B = whether the percentile buys it back.")
    print("      D vs A = what the percentile does on its own.")


if __name__ == "__main__":
    main()
