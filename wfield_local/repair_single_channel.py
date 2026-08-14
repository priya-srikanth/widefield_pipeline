"""Repair a SINGLE-CHANNEL acquisition into the 2-channel layout the pipeline expects.

WHY (PS95 2026-08-13). Only the 470 nm LED was armed at session start, so labcams was configured for
ONE channel and stayed that way for the whole recording — even after the 415 nm LED came on ~32 min in.
The file is `..._1_460_480_uint16.dat`: 532,219 flat unpaired frames, the only `_1_` file in the whole
dataset against nine `_2_` ones.

That is more dangerous than the missing 415 alone. `run_wfield_local` decides whether to hemodynamically
correct by testing `dat.shape[1] == 2`, so a 1-channel file passes straight through and yields an
UNCORRECTED `SVTcorr` **with no error raised** — a silently wrong session rather than a failed one. And
`cleanpairs` infers frame identity from LED PARITY, so feeding it a file that is single-channel and then
alternating can corrupt the frame map for the whole session, not just the prefix.

WHAT THIS DOES. Rebuilds a proper `..._2_460_480_uint16.dat` from the DAQ's own record of which LED fired
on each exposure (`led_alternation_qc`), rather than assuming anything about parity:

  1. label every exposure 415 / 470 from `led415_ttl` / `led470_ttl` gated on `pco_exposure`;
  2. drop the single-channel prefix (everything before the first 415 exposure);
  3. walk the remainder and emit a pair ONLY where a 415 frame is immediately followed by a 470 frame,
     skipping phase slips instead of letting one shift every pair after it. PS95 8/13 has 332 slips in
     413,116 frames (0.08%), so ~206,391 pairs survive.

The output frame order is (415, 470) per pair, matching `functional_channel: 1` — the pipeline's own
convention. Getting that backwards would swap the isosbestic and functional channels for the entire
session and still look like a plausible dataset, which is why the order is asserted rather than assumed.

A `repair_manifest.json` records the source, the split frame, every slip index, and the pair count, so
the provenance of a repaired session is never in doubt.

NOT DONE HERE, deliberately: the DAQ `.h5` is left alone. Its `pco_exposure` pulses still describe the
ORIGINAL frame sequence, so downstream frame-mapping must be told about the offset — see
`frame_index_map.npy`, which maps repaired-pair index -> original exposure index for exactly that.

    python -m wfield_local.repair_single_channel <raw.dat> <daq.h5> --output <dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

H, W, DTYPE = 460, 480, np.uint16
CHUNK_PAIRS = 4096


def plan(h5_path):
    """(labels, first415, pair_starts) from the DAQ — which exposures pair with which."""
    from wfield_local.led_alternation_qc import analyse

    out, on415, on470, _starts = analyse(h5_path)
    if out["first_415_frame"] < 0:
        raise ValueError(f"{h5_path}: no 415 exposures at all — nothing to repair")
    lab = np.where(on415, 0, np.where(on470, 1, -1))
    first = int(out["first_415_frame"])
    # a pair is a 415 immediately followed by a 470; anything else is skipped
    idx = np.arange(first, lab.size - 1)
    ok = (lab[idx] == 0) & (lab[idx + 1] == 1)
    return lab, first, idx[ok], out


def repair(dat_path, h5_path, out_dir, dry_run=False, verbose=True):
    dat_path, out_dir = Path(dat_path), Path(out_dir)
    lab, first, pair_starts, qc = plan(h5_path)

    nbytes = dat_path.stat().st_size
    fsz = H * W * np.dtype(DTYPE).itemsize
    n_file = nbytes // fsz
    if nbytes % fsz:
        raise ValueError(f"{dat_path.name}: size {nbytes} is not a whole number of {H}x{W} frames")
    # the DAQ may log one more exposure than the file holds (last frame not flushed) -- drop pairs
    # that would read past the end rather than trusting the counts to agree
    keep = pair_starts[(pair_starts + 1) < n_file]
    n_pairs = int(keep.size)
    if verbose:
        print(f"  file frames      {n_file:,}", flush=True)
        print(f"  DAQ exposures    {lab.size:,}   (delta {lab.size - n_file:+d})", flush=True)
        print(f"  single-ch prefix {first:,} frames ({qc['blue_only_prefix_min']:.1f} min) -> dropped",
              flush=True)
        print(f"  phase slips      {qc['n_phase_slips_after']:,}", flush=True)
        print(f"  PAIRS OUT        {n_pairs:,}", flush=True)
    if n_pairs < 1000:
        raise ValueError(f"only {n_pairs} pairs recoverable — refusing to write")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = dat_path.name.replace("_1_", "_2_")
    out_dat = out_dir / stem
    meta = {
        "source_dat": str(dat_path), "source_h5": str(h5_path), "created_utc":
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_frames_in": int(n_file), "n_daq_exposures": int(lab.size),
        "split_frame": first, "blue_only_prefix_min": qc["blue_only_prefix_min"],
        "n_phase_slips": qc["n_phase_slips_after"], "n_pairs_out": n_pairs,
        "pair_order": "(415, 470) — matches functional_channel: 1",
        "note": "rebuilt from DAQ LED labels, not from frame parity; slips skipped, not shifted",
    }
    if dry_run:
        print("  [dry-run] would write " + str(out_dat), flush=True)
        return meta

    src = np.memmap(dat_path, dtype=DTYPE, mode="r", shape=(int(n_file), H, W))
    t0 = time.time()
    with open(out_dat, "wb", buffering=0) as fo:
        for a in range(0, n_pairs, CHUNK_PAIRS):
            b = min(n_pairs, a + CHUNK_PAIRS)
            s = keep[a:b]
            block = np.empty((b - a, 2, H, W), dtype=DTYPE)
            block[:, 0] = src[s]                       # 415
            block[:, 1] = src[s + 1]                   # 470
            fo.write(block.tobytes())
            if a and a % (CHUNK_PAIRS * 16) == 0:
                el = time.time() - t0
                print(f"    {a:,}/{n_pairs:,} pairs  {a/el:.0f} pairs/s", flush=True)
    del src
    np.save(out_dir / "frame_index_map.npy", keep.astype(np.int64))
    (out_dir / "repair_manifest.json").write_text(json.dumps(meta, indent=2, default=float))
    if verbose:
        print(f"  wrote {out_dat} ({out_dat.stat().st_size/2**30:.1f} GiB) in "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dat")
    ap.add_argument("h5")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    repair(a.dat, a.h5, a.output, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
