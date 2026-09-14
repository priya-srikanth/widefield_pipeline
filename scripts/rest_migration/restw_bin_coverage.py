"""Does `restw`'s PER-POSITION time-local baseline have enough frames per bin to be estimated?

Priya, 2026-09-13: *"position-weighting will be for only the positions in 'time local'? I just worry
that that could be skewed for the positions in the block"*.

THE CONCERN, STATED PRECISELY. `session_restw_svt_timelocal` builds each position's OWN time-local
baseline -- bin the session into 12, median that position's rest frames per bin, interpolate. But a
position only HAS rest frames during its own BLOCKS, which occupy particular stretches of the
session. So for the bins where that position's blocks did not run, its baseline is not measured at
all; `_timelocal_from_mask` INTERPOLATES ACROSS THEM.

TWO WAYS THAT CAN GO WRONG:

1. NOISE. Each position holds ~1/6 of the rest frames, so each per-position per-bin median is
   estimated from ~1/6 the data of the pooled one. The buffer sweep already showed what small-n
   medians do -- they are noise-dominated and score like contamination when they are nothing of the
   sort. Six noisy baselines averaged is not obviously worse than one clean one, but it is not
   obviously equal either, and it has to be measured.

2. TIME-SKEW, which is Priya's point. If a position's blocks cluster early in the session, its
   baseline is MEASURED early and INTERPOLATED late; another position is the reverse. Averaging the
   six then blends estimates of unequal reliability at every moment, and the blend changes over the
   session. Block order is SHUFFLED across sessions at the cohort level (DECISIONS.md finding 9:
   mean normalised index 0.50 +- 0.006), but that is a statement about the COHORT and says nothing
   about whether a GIVEN session spreads a given position evenly.

WHAT THIS REPORTS, per session and per position: how many of the 12 bins actually contain that
position's rest frames, the frames-per-occupied-bin, and the mean NORMALISED SESSION TIME of that
position's rest frames (0.5 = evenly spread; far from 0.5 = clustered early or late).

RUN:  python -m scripts.rest_migration.restw_bin_coverage [--limit 6] [--bins 12]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    from wfield_local import config, joint_basis
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.position_reference_maps import REST_BASELINE_BINS
    from wfield_local.rest_by_position import rest_frames_by_position

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--bins", type=int, default=REST_BASELINE_BINS)
    a = ap.parse_args()

    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [x for x in SESSIONS if x["label"] in want]
    step = max(1, len(todo) // max(1, a.limit))
    todo = todo[::step][: a.limit]

    print(f"bins = {a.bins};  a per-position per-bin median needs >= 2*bins frames overall to be "
          f"formed at all")
    cov_all, skew_all, per_bin_all = [], [], []
    for s in todo:
        lab = s["label"]
        try:
            _u, v = joint_basis._load_session(s["mc"])
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:50]}")
            continue
        T = np.asarray(v).shape[1]
        per, info = rest_frames_by_position(s, T)
        if info.get("error") or not per:
            print(f"  !! {lab}: {info.get('error')}")
            continue
        edges = np.linspace(0, T, a.bins + 1)
        print(f"\n=== {lab}  ({info['n_periods']} labelled rest periods, T={T}) ===")
        print(f"    {'pos':>4}{'frames':>9}{'bins/12':>9}{'per bin':>10}{'mean t':>9}")
        for c in sorted(per):
            fr = np.asarray(per[c])
            occ = int(sum(1 for b in range(a.bins)
                          if ((fr >= edges[b]) & (fr < edges[b + 1])).any()))
            mt = float(fr.mean() / T)
            cov_all.append(occ / a.bins)
            skew_all.append(abs(mt - 0.5))
            per_bin_all.append(fr.size / max(1, occ))
            flag = "  <-- clustered" if abs(mt - 0.5) > 0.12 else ""
            print(f"    {c:>4}{fr.size:>9}{occ:>6}/{a.bins:<3}{fr.size / max(1, occ):>10.0f}"
                  f"{mt:>9.3f}{flag}")

    if not cov_all:
        print("\nNOTHING MEASURED -- a failed run, not a negative result.")
        return 1
    cov = np.array(cov_all)
    sk = np.array(skew_all)
    pb = np.array(per_bin_all)
    print(f"\n{'=' * 70}\nRESTW PER-POSITION BIN COVERAGE\n{'=' * 70}")
    print(f"position-sessions measured: {len(cov)}")
    print(f"bin coverage      mean {100 * cov.mean():.1f}%   min {100 * cov.min():.1f}%   "
          f"fully covered {int((cov >= 1.0).sum())}/{len(cov)}")
    print(f"frames per bin    mean {pb.mean():.0f}   min {pb.min():.0f}")
    print(f"time skew |t-0.5| mean {sk.mean():.3f}   max {sk.max():.3f}   "
          f"clustered (>0.12) {int((sk > 0.12).sum())}/{len(sk)}")
    print("\nHOW TO READ IT:")
    print("  coverage ~100% and low skew  ->  per-position time-local baselines are estimable and")
    print("                                   the current restw construction is sound")
    print("  coverage <100% or high skew  ->  positions are interpolated across bins they never")
    print("                                   occupied, and the six baselines have unequal")
    print("                                   reliability at any given moment. Prefer the ADDITIVE")
    print("                                   construction: ONE pooled time-local baseline (all")
    print("                                   rest frames, well estimated) PLUS the equally-")
    print("                                   weighted mean of per-position OFFSETS. That splits")
    print("                                   the temporal term from the composition term and")
    print("                                   needs no per-position-per-bin median at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
