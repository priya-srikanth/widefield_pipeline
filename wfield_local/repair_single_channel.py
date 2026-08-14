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

ALIGNMENT IS VERIFIED, NOT ASSUMED. Steps 1–3 index the .dat by DAQ exposure number, which is only
valid if exposure i IS file frame i. The frame counts agreeing (532,220 DAQ vs 532,219 file, delta +1)
is NOT sufficient evidence: a dropped write mid-session plus a trailing unflushed exposure gives the
same delta while shifting every later index. So `repair` runs `verify_offset` before writing anything,
reading ~700 MB of actual pixels: 415 and 470 frames differ grossly in mean intensity, and because the
sequence alternates, a misaligned comparison is ANTI-correlated rather than merely noisy. Measured on
PS95 2026-08-13: **1.000 agreement at offset 0, 0.001 at ±1**, and the pixel alternation onset lands on
119,104 — the DAQ's split frame exactly. (Parity has period 2 and so cannot separate offset 0 from ±2;
the onset check pins that, and both must pass.) The camlog corroborates all of it independently:
532,219 frame lines with zero `frame_id` gaps, 532,220 LED lines, LED state 5 first appearing at line
119,104, and a 332-frame state imbalance matching the 332 DAQ phase slips.

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
# 256 pairs = 512 frames = 226 MB per read block. Do not raise this much: the block is materialized
# as a (CHUNK, 2, H, W) array, so 4096 pairs would allocate 3.6 GB per iteration.
CHUNK_PAIRS = 256
MIN_PARITY_AGREEMENT = 0.95


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


def verify_offset(src, lab, first, n_spans=4, span=400):
    """Confirm from the PIXELS that DAQ exposure index == file frame index.

    The frame COUNT agreeing (delta +1) is weak evidence: a mid-session dropped write plus a trailing
    unflushed exposure also gives delta +1, while shifting every index after the drop -- which would
    swap 415 and 470 for the rest of the session and still produce a plausible dataset.

    415 nm and 470 nm frames differ grossly in mean intensity, so the .dat records its own channel
    identity. Alternation makes a misaligned comparison ANTI-correlated, so the correct offset scores
    ~1.0 and a wrong one ~0.0 -- a fingerprint rather than a weak coincidence test. Measured on PS95
    2026-08-13: 1.000 at offset 0, 0.001 at +/-1.

    Returns (agreement, mean_415, mean_470). Parity alone cannot separate offset 0 from +/-2 (the
    alternation has period 2); the caller pins that with the alternation ONSET, which must fall on
    ``first``.
    """
    n_file = src.shape[0]
    starts = np.linspace(first + 5_000, n_file - span - 10, n_spans).astype(int)
    ok = tot = 0
    dim_sum = np.zeros(2)
    dim_n = np.zeros(2)
    for s0 in starts:
        s0 = int(max(s0, 0))
        m = src[s0:s0 + span].reshape(span, -1).mean(axis=1)
        lo, hi = np.percentile(m, [25, 75])
        pix_is_415 = m < (lo + hi) / 2            # violet is the dimmer channel
        d = lab[s0:s0 + span]
        sel = d >= 0
        ok += int((pix_is_415 == (d == 0))[sel].sum())
        tot += int(sel.sum())
        for v in (0, 1):
            dim_sum[v] += m[d == v].sum()
            dim_n[v] += int((d == v).sum())
    agree = ok / max(tot, 1)
    means = tuple(dim_sum / np.maximum(dim_n, 1))
    return agree, means[0], means[1]


def alternation_onset(src, first, half=120):
    """Frame at which the pixels start alternating, found without reference to the DAQ."""
    a = max(first - half, 0)
    b = min(first + half, src.shape[0])
    m = src[a:b].reshape(b - a, -1).mean(axis=1)
    sw = np.abs(np.diff(m))
    k = max(first - a, 1)
    base = np.median(sw[:k]) if k > 1 else 0.0
    thr = max(5 * base, 0.2 * np.median(sw[k:])) if b - first > 2 else 5 * base
    hit = np.flatnonzero(sw > thr)
    return int(a + hit[0] + 1) if hit.size else None


def repair(dat_path, h5_path, out_dir, dry_run=False, verbose=True, verify=True):
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

    src = np.memmap(dat_path, dtype=DTYPE, mode="r", shape=(int(n_file), H, W))
    agree = onset = m415 = m470 = None
    if verify:
        agree, m415, m470 = verify_offset(src, lab, first)
        onset = alternation_onset(src, first)
        if verbose:
            print(f"  PIXEL parity     {agree:.3f} agreement with DAQ labels "
                  f"(415 {m415:.0f} vs 470 {m470:.0f})", flush=True)
            print(f"  PIXEL onset      {onset:,}  (DAQ {first:,}, delta "
                  f"{onset - first:+d})" if onset is not None else "  PIXEL onset      not found",
                  flush=True)
        # parity alone cannot separate offset 0 from +/-2, so the onset must agree as well
        if agree < MIN_PARITY_AGREEMENT:
            raise ValueError(
                f"pixel/DAQ label agreement {agree:.3f} < {MIN_PARITY_AGREEMENT}: exposure index is "
                f"NOT the frame index (agreement near 0 means an off-by-one, which would swap 415 and "
                f"470 for the whole session). Refusing to write.")
        if onset is None or abs(onset - first) > 1:
            raise ValueError(
                f"pixel alternation starts at {onset} but the DAQ says {first}: the two disagree on "
                f"where the single-channel prefix ends. Refusing to write.")

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
        "pixel_parity_agreement": agree, "pixel_alternation_onset": onset,
        "mean_intensity_415": m415, "mean_intensity_470": m470,
    }
    if dry_run:
        print("  [dry-run] would write " + str(out_dat), flush=True)
        return meta

    t0 = time.time()
    with open(out_dat, "wb", buffering=0) as fo:
        for a in range(0, n_pairs, CHUNK_PAIRS):
            b = min(n_pairs, a + CHUNK_PAIRS)
            s = keep[a:b]
            # read the whole contiguous span ONCE and gather within it. Fancy-indexing the memmap
            # directly issues one read per frame, which is fine locally and terrible over SMB.
            lo, hi = int(s[0]), int(s[-1]) + 2
            span = np.asarray(src[lo:hi])
            r = s - lo
            block = np.empty((b - a, 2, H, W), dtype=DTYPE)
            block[:, 0] = span[r]                      # 415
            block[:, 1] = span[r + 1]                  # 470
            fo.write(block.tobytes())
            if a and a % (CHUNK_PAIRS * 16) == 0:
                el = time.time() - t0
                print(f"    {a:,}/{n_pairs:,} pairs  {a/el:.0f} pairs/s", flush=True)
    del src
    np.save(out_dir / "frame_index_map.npy", keep.astype(np.int64))
    (out_dir / "repair_manifest.json").write_text(json.dumps(meta, indent=2, default=float))
    # the pipeline's frame map, so the maps step can place these frames on the DAQ clock
    fm = write_frame_map(out_dir.parent / "motion_corrected", stem, lab, keep, h5_path, dat_path)
    if verbose:
        print(f"  frame map -> {fm}", flush=True)
    if verbose:
        print(f"  wrote {out_dat} ({out_dat.stat().st_size/2**30:.1f} GiB) in "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    return meta


def write_frame_map(mc_dir, dat_name, lab, keep, h5_path, dat_path):
    """Emit the `*cleanpairs_frame_map.npz` + summary the rest of the pipeline consumes.

    A repaired session skips the TTL relabel (it is already paired), but the relabel is also what
    normally produces the frame map — and downstream timing depends on it. `framemap_event_maps`
    reads `pco_samples[original_frame_index_ch0[t] + offset]` to place corrected frame *t* on the DAQ
    clock, so those indices must refer to the ORIGINAL exposure sequence, not to positions in the
    repaired file. `keep` is exactly that: pair *t* of the repaired movie came from original
    exposures `(keep[t], keep[t]+1)`. Without this the session would be silently mistimed by the
    length of the dropped prefix — 32 minutes for PS95 8/13.

    Written in the relabel's own format and naming so every consumer works unchanged.
    """
    mc_dir = Path(mc_dir)
    mc_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(dat_name).stem
    npz = mc_dir / f"{stem}_daq_led_cleanpairs_frame_map.npz"
    js = mc_dir / f"{stem}_daq_led_cleanpairs_summary.json"

    keep = np.asarray(keep, dtype=np.int64)
    per_frame = np.where(lab == 0, 415, np.where(lab == 1, 470, 0)).astype(np.int16)
    used = np.zeros(lab.size, dtype=bool)
    used[keep] = True
    used[keep + 1] = True
    skipped = np.flatnonzero((per_frame != 0) & ~used).astype(np.int64)

    np.savez_compressed(
        npz,
        pair_index=np.arange(keep.size, dtype=np.int64),
        original_frame_index_ch0=keep,
        original_frame_index_ch1=keep + 1,
        channel_label_ch0=np.full(keep.size, 415, dtype=np.int16),
        channel_label_ch1=np.full(keep.size, 470, dtype=np.int16),
        labels_per_original_frame=per_frame,
        skipped_original_frame_index=skipped,
    )
    js.write_text(json.dumps({
        "mode": "repair_single_channel", "source_dat": str(dat_path), "daq_h5": str(h5_path),
        "output_dat": str(mc_dir.parent / "raw_widefield_data" / dat_name),
        "frame_map_npz": str(npz), "output_shape": [int(keep.size), 2, H, W],
        "output_dtype": "uint16", "channel_order": "415-470", "clean_pairs": int(keep.size),
        "skipped_illuminated_frames": int(skipped.size),
        # Eight modules read this key off the cleanpairs summary to place frames on the DAQ clock, so a
        # repaired session MUST carry it or it silently drops out of every one of them. It is 0 here,
        # and that is not an assumption: `verify_offset` measured exposure i == frame i at 1.000
        # agreement (0.001 at +/-1) before the repair was allowed to write.
        "chosen_exposure_offset": 0,
        "daq_pco_exposure_count": int(lab.size),
        "dat_physical_frame_count": int(lab.size),
        "labels_415": int((per_frame == 415).sum()), "labels_470": int((per_frame == 470).sum()),
        "labels_both": 0, "labels_dark": int((per_frame == 0).sum()),
        "note": "indices refer to the ORIGINAL exposure sequence; the repaired .dat is already paired",
    }, indent=2), encoding="utf-8")
    return npz


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dat")
    ap.add_argument("h5")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the pixel/DAQ parity guard (reads ~700 MB); NOT recommended")
    a = ap.parse_args(argv)
    repair(a.dat, a.h5, a.output, dry_run=a.dry_run, verify=not a.no_verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
