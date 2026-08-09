"""Run wfield motion correction on a labcams DAT file.

This wrapper avoids the installed wfield CLI's confusing output-path behavior
and writes motion-corrected data plus shift summaries into an explicit folder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from wfield.io import mmap_dat
# Sign-corrected motion correction (wfield 0.4.2 registration_upsample doubles drift
# due to a phase-correlation sign error). See wfield_local/motion_correct_fixed.py.
from wfield_local.motion_correct_fixed import motion_correct


class RelabeledDat:
    """Lazy view of the DAQ-relabeled cleanpairs movie, gathered from the RAW .dat on the fly.

    The TTL relabel is a pure frame GATHER: cleanpairs pair ``i`` = ``(raw[pairs[i,0]],
    raw[pairs[i,1]])`` (415, 470). Instead of materializing that raw-sized ``*_cleanpairs_*.dat``
    to disk just to feed motion correction, this presents the same ``(n_pairs, 2, H, W)`` array
    lazily: :meth:`__getitem__` gathers the requested frames from the raw memmap. ``motion_correct``
    only ever touches ``dat.shape`` and slice reads ``np.array(dat[a:b])`` (see
    ``motion_correct_fixed``), so this is a drop-in substitute — identical bytes into MC, no 148 GB
    intermediate written or read back.
    """

    def __init__(self, raw_flat, pairs):
        self._raw = raw_flat                      # memmap (Ntotal_frames, H, W), arrival order
        self._pairs = np.asarray(pairs)           # (n_pairs, 2) int indices into raw_flat
        self.dtype = raw_flat.dtype
        self.shape = (len(self._pairs), 2, raw_flat.shape[1], raw_flat.shape[2])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sl):
        # slice -> (k, 2, H, W); int -> (2, H, W). Fancy-indexing the raw memmap materializes
        # only the requested (chunk-sized) frames.
        return self._raw[self._pairs[sl]]


def _relabeled_from_summary(raw_dat: Path, summary: dict) -> RelabeledDat:
    """Build a :class:`RelabeledDat` from the raw .dat + a map-only relabel summary/frame_map."""
    fm = np.load(summary["frame_map_npz"])
    pairs = np.stack([fm["original_frame_index_ch0"], fm["original_frame_index_ch1"]], axis=1)
    h, w = int(summary["output_shape"][2]), int(summary["output_shape"][3])
    dt = np.dtype(summary["output_dtype"])
    ntotal = Path(raw_dat).stat().st_size // (h * w * dt.itemsize)
    raw_flat = np.memmap(raw_dat, mode="r", dtype=dt, shape=(int(ntotal), h, w))
    return RelabeledDat(raw_flat, pairs)


def _run_relabel_subprocess(dat: Path, daq_h5: Path, output: Path, mode: str, led_threshold,
                            map_only: bool = False) -> dict:
    """Run the TTL relabel in a SEPARATE process and return its summary dict.

    h5py and wfield cannot be imported in the same process on this rig (their
    HDF5 DLLs conflict), so the relabel (h5py) must run in its own interpreter,
    separate from motion correction (wfield). With ``map_only`` the subprocess writes only the
    frame_map + summary (no raw-sized cleanpairs .dat); the caller applies the map on the fly.
    """
    import json as _json
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-m", "wfield_local.trim_illuminated_labcams",
           str(dat), str(daq_h5), "--output-dir", str(output),
           "--label", "daq_led", "--mode", mode]
    if led_threshold is not None:
        cmd += ["--led-threshold", str(led_threshold)]
    if map_only:
        cmd += ["--map-only"]
    print("[pipeline] TTL relabel (separate process): " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=repo_root)
    summary_path = output / f"{Path(dat).stem}_daq_led_cleanpairs_summary.json"
    return _json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Motion-correct a wfield/labcams DAT.")
    parser.add_argument("dat", type=Path, help="DAT file ending in _N_H_W_dtype.dat")
    parser.add_argument("--output", type=Path, required=True, help="Output folder")
    parser.add_argument(
        "--mode",
        choices=("2d", "ecc"),
        default="2d",
        help="2d is translation-only and faster; ecc estimates rigid-body shifts.",
    )
    parser.add_argument("--chunksize", type=int, default=256)
    parser.add_argument("--nreference", type=int, default=60)
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=None,
        help="Optional short run for testing. Uses only the first N frames.",
    )
    # TTL-based LED relabeling BEFORE motion correction (and thus before SVD).
    # For trial-gated recordings the saved .dat channel split (frame parity) can
    # drift from the true 415/470 LED across trials; relabeling from the DAQ
    # makes channel 0 = 415, channel 1 = 470 deterministically.
    parser.add_argument(
        "--daq-h5",
        type=Path,
        default=None,
        help="DAQ recorder .h5. If given, relabel the DAT to DAQ-confirmed 415/470 "
        "pairs first, then motion-correct the relabeled DAT.",
    )
    parser.add_argument(
        "--relabel-mode",
        choices=("rescue", "acquire-enable"),
        default="acquire-enable",
        help="Relabel mode when --daq-h5 is given (default acquire-enable).",
    )
    parser.add_argument("--relabel-led-threshold", type=float, default=None)
    parser.add_argument(
        "--relabel-on-the-fly",
        action="store_true",
        help="Apply the DAQ relabel as a lazy gather from the raw .dat during motion correction "
        "instead of materializing the raw-sized cleanpairs .dat (identical bytes into MC; saves a "
        "~raw-sized write+readback). Only affects the --daq-h5 relabel path.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    dat_path = args.dat
    relabeled = False
    if args.daq_h5 is not None:
        relabeled = True
        on_the_fly = args.relabel_on_the_fly
        print(f"[pipeline] TTL relabel before motion correction (mode={args.relabel_mode}, "
              f"{'on-the-fly' if on_the_fly else 'materialized'})", flush=True)
        summary = _run_relabel_subprocess(
            args.dat, args.daq_h5, args.output, args.relabel_mode, args.relabel_led_threshold,
            map_only=on_the_fly)
        if on_the_fly:
            dat = _relabeled_from_summary(args.dat, summary)
            print(f"[pipeline] relabel applied on the fly from raw {args.dat} "
                  f"(no cleanpairs .dat written)", flush=True)
        else:
            dat_path = Path(summary["output_dat"])
            print(f"[pipeline] relabeled DAT -> {dat_path}", flush=True)
            dat = mmap_dat(str(dat_path), mode="r")
    else:
        dat = mmap_dat(str(dat_path), mode="r")

    if args.limit_frames is not None:
        dat = dat[: args.limit_frames]

    nframes, nchannels, height, width = dat.shape
    dtype = np.dtype(dat.dtype).name
    corrected_path = args.output / (
        f"motioncorrect_{nchannels}_{height}_{width}_{dtype}.bin"
    )

    print(f"Input: {dat_path}", flush=True)
    print(f"Input shape: {dat.shape}, dtype={dat.dtype}", flush=True)
    print(f"Output: {corrected_path}", flush=True)
    corrected = np.memmap(
        corrected_path,
        mode="w+",
        dtype=dat.dtype,
        shape=dat.shape,
    )

    (yshifts, xshifts), rshifts = motion_correct(
        dat,
        out=corrected,
        chunksize=args.chunksize,
        nreference=args.nreference,
        mode=args.mode,
        apply_shifts=True,
    )
    corrected.flush()
    del corrected

    shifts = np.rec.array(
        [yshifts, xshifts],
        dtype=[("y", "float32"), ("x", "float32")],
    )
    np.save(args.output / "motion_correction_shifts.npy", shifts)
    np.save(args.output / "motion_correction_rotation.npy", rshifts)

    summary = {
        "source_dat": str(dat_path),
        "original_dat": str(args.dat),
        "daq_h5": str(args.daq_h5) if args.daq_h5 is not None else None,
        "ttl_relabeled": relabeled,
        "relabel_mode": args.relabel_mode if relabeled else None,
        "motion_corrected_file": str(corrected_path),
        "shape": [int(nframes), int(nchannels), int(height), int(width)],
        "dtype": dtype,
        "mode": args.mode,
        "chunksize": args.chunksize,
        "nreference": args.nreference,
        "limit_frames": args.limit_frames,
        "y_shift_min_max": [float(np.nanmin(yshifts)), float(np.nanmax(yshifts))],
        "x_shift_min_max": [float(np.nanmin(xshifts)), float(np.nanmax(xshifts))],
        "rotation_min_max": [float(np.nanmin(rshifts)), float(np.nanmax(rshifts))],
    }
    (args.output / "motion_correction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Motion correction complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
