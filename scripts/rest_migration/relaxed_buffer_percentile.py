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

THE MASK IS CACHED PER BUFFER (2026-09-14). The first version recomputed it inside the arm loop, so
each session paid FOUR full DAQ reads for TWO distinct masks -- the estimator is the cheap half. That
alone was most of the ~10 min/session that limited the first run to four sessions.

ALSO REPORTED, because neutrality is not the only thing at stake: the per-position frame counts and
how many positions clear the PRODUCTION gate (`rest_by_position.MIN_FRAMES_PER_POSITION`, and the
six-position requirement). A relaxation that is neutrality-free is only worth taking if it actually
feeds the starved sessions.

    python -m scripts.rest_migration.relaxed_buffer_percentile [--limit N] [--every K] [--pct 20]
    python -m scripts.rest_migration.relaxed_buffer_percentile --sessions PS92_0826 PS94_0819
"""
from __future__ import annotations

import argparse

import numpy as np

from wfield_local import config

BUFFERS = ((1.0, 2.0), (0.5, 1.0))
ESTS = (("median", None), ("p{p}", "pct"))
MIN_POS_FRAMES = 50          # for the spread estimate itself
PROD_GATE = 200              # rest_by_position.MIN_FRAMES_PER_POSITION
PROD_POSITIONS = 6           # rest_by_position.MIN_POSITIONS_FOR_WEIGHTED


def _arm_name(est, buf, pct):
    return f"{est.format(p=pct)} [{buf[0]:g},{buf[1]:g}]"


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
    # THE REPAIRED CLASSIFIER, exactly as `rest_by_position._gather` uses it. Raw `_classify_cues`
    # reports what the hardware said, and a dead `spout_bit1` (Aug 2026) reads that bit low, so the
    # 3-bit code COLLAPSES 6 positions onto 4 (2->0, 3->1, 6->4, 7->5). Production was fixed for this
    # on 2026-09-14 and THIS SCRIPT WAS THEN WRITTEN WITH THE RAW CLASSIFIER ANYWAY -- so PS95_0806
    # reported "4 positions" and I read it as an unrecoverable hardware limit. It is recoverable:
    # `classify_cues_with_backup` repairs it from the behaviour log. (Priya, 2026-09-14.)
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.locanmf_position_decoder import _load_cue_events

    try:
        cue_ev = _load_cue_events(s["h5"])
        codes = np.asarray(classify_cues_with_backup(s, cue_ev, verbose=False))
        cs = np.asarray(cue_ev["cue_samples"], np.int64)
    except Exception as ex:                                            # noqa: BLE001
        print(f"  !! {s['label']}: repaired classifier failed ({type(ex).__name__}), "
              f"falling back to raw", flush=True)
        codes = np.asarray(_classify_cues(cue, st, daq_io.strobe_codes(packed, dn, st)))
        cs = np.asarray(cue, np.int64)
    m, _note = rest_mask(n, fs, speed, np.asarray(lk["lick_onsets"], np.int64),
                         cue / fs, ts / fs, st / fs, params=q,
                         session_dir=behaviour_session_dir(s["label"]),
                         sync_s=sync / fs, position_codes=codes)
    return m, cs, codes, fs, packed, dn


def _per_position(s, T, buf):
    """Rest FRAMES grouped by the position of the two trials bracketing them, for one buffer."""
    from wfield_local import daq_io
    from wfield_local.rest_by_position import frame_samples

    m, cue, codes, fs, packed, dn = _rest_mask_with_buffer(s, buf)
    pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
    f_of = frame_samples(s["mc"], s.get("fmdir"), s.get("regime"), pco)
    if f_of is None or codes is None:
        return None
    f_of = np.asarray(f_of)[:T]
    rest_fr = np.flatnonzero(m[np.clip(f_of, 0, m.size - 1)])
    per = {}
    cs = np.asarray(cue, np.int64)
    for fr in rest_fr:
        smp = f_of[fr]
        prev = np.searchsorted(cs, smp, "right") - 1
        nxt = prev + 1
        if prev < 0 or nxt >= len(codes) or codes[prev] != codes[nxt] or codes[prev] < 0:
            continue
        per.setdefault(int(codes[prev]), []).append(fr)
    return {c: np.asarray(v) for c, v in per.items()}


def run(label, pct):
    s = next(x for x in config.load_sessions() if x["label"] == label)
    V = np.load(config.svtcorr_path(s["mc"]), mmap_mode="r")
    X = np.asarray(V[:, :], dtype=np.float64)
    T = X.shape[1]
    scale = float(np.sqrt((X ** 2).mean()))

    # ONE mask per buffer, not one per arm.
    per_buf = {}
    for buf in BUFFERS:
        try:
            per_buf[buf] = _per_position(s, T, buf)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {label} buf {buf}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
            return None
        if per_buf[buf] is None:
            return None

    out = {}
    for buf in BUFFERS:
        raw = per_buf[buf]
        usable = {c: v for c, v in raw.items() if v.size >= MIN_POS_FRAMES}
        n_prod = sum(1 for v in raw.values() if v.size >= PROD_GATE)
        if len(usable) < 4:
            print(f"  .. {label} buf [{buf[0]:g},{buf[1]:g}]: only {len(usable)} usable positions",
                  flush=True)
            continue
        for est, kind in ESTS:
            f = ((lambda a: np.percentile(a, pct, axis=1)) if kind
                 else (lambda a: np.median(a, axis=1)))
            levels = np.stack([f(X[:, idx]) for c, idx in sorted(usable.items())], 0)
            spread = float(np.sqrt(((levels - levels.mean(0)) ** 2).mean())) / scale
            out[_arm_name(est, buf, pct)] = (
                spread, int(sum(v.size for v in usable.values())), len(usable), n_prod)
    if not out:
        return None
    parts = [f"{nm}: {sp:.4f} ({fr}fr {npos}p {nprod}/{PROD_POSITIONS}prod)"
             for nm, (sp, fr, npos, nprod) in out.items()]
    print(f"  {label:12s}  " + "   ".join(parts), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--every", type=int, default=1,
                    help="take every Kth curated session -- spreads the sample across animals/epochs "
                         "instead of taking the first N, which are all one date")
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--pct", type=float, default=20.0)
    a = ap.parse_args()
    if a.sessions:
        labs = list(a.sessions)
    else:
        want = set(config.phase_labels("pre") + config.phase_labels("post"))
        labs = [x["label"] for x in config.load_sessions() if x["label"] in want]
        labs = labs[:: max(1, a.every)]
        if a.limit:
            labs = labs[: a.limit]
    print(f"{len(labs)} sessions: {', '.join(labs)}\n", flush=True)
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
    keys = [_arm_name(e, b, a.pct) for b in BUFFERS for e, _ in ESTS]
    print(f"\nPOSITION SPREAD of the baseline ({len(rows)} sessions; lower = more position-neutral)")
    for k in keys:
        v = np.array([r[k][0] for r in rows if k in r])
        f = np.array([r[k][1] for r in rows if k in r], float)
        print(f"  {k:18s} n={v.size:3d}  median spread {np.median(v):.4f}   "
              f"median frames {np.median(f):7.0f}")

    # PAIRED, because every arm is measured on the same sessions. A median-of-medians hides a
    # consistent within-session shift; the paired delta does not.
    a_med, b_med = _arm_name("median", BUFFERS[0], a.pct), _arm_name("median", BUFFERS[1], a.pct)
    both = [r for r in rows if a_med in r and b_med in r]
    if both:
        d = np.array([r[b_med][0] - r[a_med][0] for r in both])
        g = np.array([r[b_med][1] / max(1, r[a_med][1]) for r in both])
        rng = np.random.default_rng(0)
        bs = np.array([np.median(rng.choice(d, d.size)) for _ in range(5000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"\nPAIRED B-A (median estimator, {len(both)} sessions)")
        print(f"  spread delta   median {np.median(d):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"worse in {int((d > 0).sum())}/{d.size}")
        print(f"  frame gain     median x{np.median(g):.3f}   range x{g.min():.3f}-x{g.max():.3f}")
        prod = [(r[a_med][3], r[b_med][3]) for r in both]
        gained = [p for p in prod if p[1] > p[0]]
        short_a = [p for p in prod if p[0] < PROD_POSITIONS]
        print(f"  production gate: {len(short_a)}/{len(prod)} sessions short of {PROD_POSITIONS} "
              f"positions under A; {len(gained)} gain a position under B")
    print("\nREAD: B vs A = what relaxing the buffer costs in position-neutrality.")
    print("      C vs B = whether the percentile buys it back.")
    print("      D vs A = what the percentile does on its own.")


if __name__ == "__main__":
    main()
