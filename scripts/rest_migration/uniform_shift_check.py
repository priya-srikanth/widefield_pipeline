"""THE THING POSITION SPREAD CANNOT SEE: does relaxing the lick buffer shift the baseline UNIFORMLY?

`relaxed_buffer_percentile` measured the SPREAD ACROSS POSITIONS of the rest baseline and found
`[0.5, 1]` free (paired -0.0019, better in 13/19) with +25% frames. But position spread is a
DIFFERENTIAL measure by construction: it compares the six positions to each other. A shorter buffer
that admits lick-adjacent frames EQUALLY at every position would raise all six baselines by the same
amount, leave the spread untouched, and still shift every `map - rest` amplitude -- because that
subtraction is `position_map - baseline`, so any delta in the baseline lands on every position.

The 2.5 h regeneration (94 masks + 111 events) is the cost of getting this wrong, so it is measured
before the config is touched, not after.

WHAT IS REPORTED, per session:

    d = baseline_B - baseline_A                     the shift itself, in component space
    ||d|| / ||evoked||                              the shift as a FRACTION of the measured signal,
                                                    where evoked = mean(post-cue frames) - baseline_A
    cos(d, evoked)                                  ALIGNMENT -- the half that magnitude alone misses

The alignment is the informative one and costs nothing extra. If `d` is ORTHOGONAL to the evoked
pattern it is estimation noise and a small magnitude is genuinely harmless. If `d` is ALIGNED with
it, the admitted frames carry task/lick-related activity, and then even a small magnitude subtracts
signal systematically -- the same failure the buffer exists to prevent, just uniform instead of
differential.

Component space is used directly rather than projecting through U: the SVD basis is orthonormal, so
norms and angles are preserved, and it avoids the U / U_atlas grid mismatch (460x480 vs 540x640) that
has already bitten once today.

NB the SIGN of any single component is arbitrary, so per-component signs are uninterpretable; norms
and the cosine between two vectors in the SAME session's basis are not.

    python -m scripts.rest_migration.uniform_shift_check [--limit N] [--every K] [--post 2.0]
"""
from __future__ import annotations

import argparse

import numpy as np

from wfield_local import config
from scripts.rest_migration.relaxed_buffer_percentile import BUFFERS, _rest_mask_with_buffer


def run(label, post_s):
    from wfield_local import daq_io
    from wfield_local.rest_by_position import frame_samples

    s = next(x for x in config.load_sessions() if x["label"] == label)
    V = np.load(config.svtcorr_path(s["mc"]), mmap_mode="r")
    X = np.asarray(V[:, :], dtype=np.float64)
    T = X.shape[1]
    scale = float(np.sqrt((X ** 2).mean()))

    base, frames, f_of, cs, fs = {}, {}, None, None, None
    for buf in BUFFERS:
        m, cs, _codes, fs, packed, dn = _rest_mask_with_buffer(s, buf)
        if f_of is None:
            pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
            f_of = frame_samples(s["mc"], s.get("fmdir"), s.get("regime"), pco)
            if f_of is None:
                return None
            f_of = np.asarray(f_of)[:T]
        idx = np.flatnonzero(m[np.clip(f_of, 0, m.size - 1)])
        if idx.size < 200:
            print(f"  .. {label}: only {idx.size} rest frames at {buf} -- skipped", flush=True)
            return None
        base[buf] = np.median(X[:, idx], axis=1)
        frames[buf] = idx.size

    d = base[BUFFERS[1]] - base[BUFFERS[0]]

    # EVOKED: pooled post-cue frames minus the arm-A baseline. This is the anchor -- the question is
    # not "is the shift small in absolute terms" but "is it small COMPARED TO WHAT WE MEASURE".
    post = []
    for c in cs:
        a = np.searchsorted(f_of, c, "left")
        b = np.searchsorted(f_of, c + post_s * fs, "right")
        if b > a:
            post.append(np.arange(a, min(b, T)))
    if not post:
        return None
    post = np.concatenate(post)
    evoked = X[:, post].mean(axis=1) - base[BUFFERS[0]]

    nd, ne = float(np.linalg.norm(d)), float(np.linalg.norm(evoked))
    cos = float(d @ evoked / (nd * ne)) if nd > 0 and ne > 0 else np.nan
    ratio = nd / ne if ne > 0 else np.nan
    print(f"  {label:12s}  ||d||/||evoked|| {ratio:6.3f}   cos {cos:+.3f}   "
          f"||d||/scale {nd / scale:7.4f}   frames {frames[BUFFERS[0]]}->{frames[BUFFERS[1]]}",
          flush=True)
    return dict(label=label, ratio=ratio, cos=cos, nd_scale=nd / scale,
                fa=frames[BUFFERS[0]], fb=frames[BUFFERS[1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--post", type=float, default=2.0)
    a = ap.parse_args()
    if a.sessions:
        labs = list(a.sessions)
    else:
        want = set(config.phase_labels("pre") + config.phase_labels("post"))
        labs = [x["label"] for x in config.load_sessions() if x["label"] in want]
        labs = labs[:: max(1, a.every)]
        if a.limit:
            labs = labs[: a.limit]
    print(f"{len(labs)} sessions\n", flush=True)
    rows = []
    for lab in labs:
        try:
            r = run(lab, a.post)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        if r:
            rows.append(r)
    if len(rows) < 4:
        print(f"\nonly {len(rows)} sessions -- a failed run, not a result")
        return
    ratio = np.array([r["ratio"] for r in rows])
    cos = np.array([r["cos"] for r in rows])
    print(f"\nUNIFORM SHIFT, {len(rows)} sessions")
    print(f"  ||d||/||evoked||   median {np.median(ratio):.3f}   "
          f"p90 {np.percentile(ratio, 90):.3f}   max {ratio.max():.3f}")
    print(f"  cos(d, evoked)     median {np.median(cos):+.3f}   "
          f"range {cos.min():+.3f} to {cos.max():+.3f}   positive in "
          f"{int((cos > 0).sum())}/{cos.size}")
    print("\nHOW TO READ IT:")
    print("  small ratio AND cos near 0   ->  the shift is estimation noise. Adopt [0.5,1].")
    print("  small ratio but cos near +-1 ->  the admitted frames carry the evoked pattern. Small")
    print("      now, but it is SIGNAL being subtracted, and it will not stay small in sessions")
    print("      that lick more. Do not adopt without a per-session bound.")
    print("  ratio comparable to 1        ->  a uniform shift the spread metric never saw.")


if __name__ == "__main__":
    main()
