"""Does the task really run a PERMUTED SWEEP of all six positions, and can we bin rest by sweep?

Priya, 2026-09-13: *"the task is designed such that the positions are balanced, ie it will sweep
through all 6 positions (each selected at random from the remaining positions out of 6), then
restarts. so we could do it in increments of every 6 blocks?"*

WHY THIS MATTERS MORE THAN IT SOUNDS. `restw` builds a time-local baseline PER POSITION, and its
per-position bin coverage is only 66% (min 17%, fully covered 2 of 30 position-sessions) with 16 of
30 positions clustered in time. A position that never occurs in a bin has its baseline INTERPOLATED
across it. Binning by TIME cannot fix that.

BINNING BY SWEEP FIXES IT BY CONSTRUCTION. If every 6 consecutive blocks contains each position
exactly once, then every sweep-bin is position-balanced with no estimate required -- coverage is
100% by design, and the composition bias `restw` exists to correct is ZERO within a bin.

THERE IS ALREADY INDEPENDENT EVIDENCE. `block_ids` measured 118 same-position adjacent block pairs
out of 4,216 = 2.80%. A permuted 6-sweep predicts a repeat ONLY at a sweep boundary, when one sweep
ends on X and the next begins on X: (1/6 of pairs are boundaries) x (1/6 chance) = 1/36 = 2.78%.
That is a match to two decimal places and is not a coincidence. This script checks it directly
rather than resting on the arithmetic.

HOW THE SWEEPS ARE PARSED. Not by slicing blocks into groups of six -- that assumes the recording
starts on a sweep boundary, and a session joined mid-sweep would be misaligned throughout. Instead:
accumulate blocks until all six positions have been seen, close the sweep, restart. That is the
natural parse of a permuted-sweep sequence and it self-synchronises after at most one partial sweep.

RUN:  python -m scripts.rest_migration.archive.sweep_structure [--limit 8]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def parse_sweeps(codes):
    """``[(start_block, stop_block)]`` -- greedy parse into sweeps covering all six positions.

    Returns half-open block index ranges. A trailing partial sweep is returned too, flagged by
    being shorter than six; callers decide whether to keep it.
    """
    out, seen, start = [], set(), 0
    for i, c in enumerate(codes):
        if c < 0:
            continue
        seen.add(int(c))
        if len(seen) == 6:
            out.append((start, i + 1))
            seen, start = set(), i + 1
    if start < len(codes):
        out.append((start, len(codes)))
    return out


def main() -> int:
    import h5py

    from wfield_local import config, daq_io
    from wfield_local.block_ids import block_ids, block_size_max_for
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS, _load_cue_events
    from wfield_local.plot_spout_trial_averages import _classify_cues

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()

    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [x for x in SESSIONS if x["label"] in want and x.get("h5")]
    step = max(1, len(todo) // max(1, a.limit))
    todo = todo[::step][: a.limit]

    n_sw, n_exact, n_blocks_all, rep_adj, adj_all = 0, 0, 0, 0, 0
    per_session = []
    for s in todo:
        lab = s["label"]
        try:
            cue = _load_cue_events(s["h5"])
            codes = np.asarray(_classify_cues(cue["cue_samples"], cue["strobe_samples"],
                                              cue["strobe_codes"]))
            blk = block_ids(codes, block_size_max_for(s))
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:55]}")
            continue
        # ONE POSITION PER BLOCK: collapse the trial sequence to a block-position sequence.
        order, pos = [], []
        for b in np.unique(blk):
            sel = blk == b
            if not sel.any():
                continue
            cs_ = codes[sel]
            cs_ = cs_[cs_ >= 0]
            if cs_.size:
                order.append(int(b))
                pos.append(int(np.bincount(cs_).argmax()))
        pos = np.asarray(pos)
        if pos.size < 12:
            print(f"  .. {lab}: only {pos.size} blocks")
            continue
        sweeps = parse_sweeps(pos)
        full = [x for x in sweeps if x[1] - x[0] >= 6]
        exact = sum(1 for x in full if x[1] - x[0] == 6)
        rep = int((pos[1:] == pos[:-1]).sum())
        n_sw += len(full)
        n_exact += exact
        n_blocks_all += pos.size
        rep_adj += rep
        adj_all += pos.size - 1
        per_session.append(len(full))
        lens = [x[1] - x[0] for x in full]
        print(f"  .. {lab:<12} {pos.size:>3} blocks  {len(full):>3} full sweeps  "
              f"exactly-6: {exact}/{len(full)}  sweep len mean {np.mean(lens):.2f}  "
              f"same-pos adjacent {rep}/{pos.size - 1} = {100 * rep / max(1, pos.size - 1):.1f}%",
              flush=True)

    print(f"\n{'=' * 74}\nIS IT A PERMUTED 6-SWEEP?\n{'=' * 74}")
    if not n_sw:
        print("NOTHING PARSED -- a failed run, not a negative result.")
        return 1
    print(f"sessions {len(per_session)};  blocks {n_blocks_all};  full sweeps {n_sw}")
    print(f"sweeps that are EXACTLY 6 blocks: {n_exact}/{n_sw} = {100 * n_exact / n_sw:.1f}%")
    print(f"same-position adjacent blocks:    {rep_adj}/{adj_all} = "
          f"{100 * rep_adj / max(1, adj_all):.2f}%   (a permuted 6-sweep predicts 1/36 = 2.78%)")
    print(f"\nsweeps per session: mean {np.mean(per_session):.1f}  "
          f"min {min(per_session)}  max {max(per_session)}")
    print("  -> that is the number of SWEEP-BINS available for a time-local baseline.")
    print("     Compare REST_BASELINE_BINS = 12: if the counts are close, binning by sweep costs")
    print("     no temporal resolution AND makes every bin position-balanced by construction.")
    print("\nIF exactly-6 is ~100%: the design holds, and a sweep-bin contains each position once,")
    print("     so per-position coverage is 100% by construction and the composition bias `restw`")
    print("     corrects is ZERO WITHIN A BIN. That is structurally better than any weighting.")
    print("IF exactly-6 is well below 100%: sweeps are being broken by aborted or dropped blocks,")
    print("     and the bin would carry a residual imbalance that still needs weighting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
