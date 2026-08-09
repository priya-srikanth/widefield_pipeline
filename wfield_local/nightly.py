"""One-command nightly, dispatched by machine (PathResolver auto-detect, or ``--machine``).

**Imaging box** (``preprocess`` -> ``preprocess_deck`` -> archive COPY):
  motion/SVD/xreg/push + maps + xall + intensity + photobleach, then rebuild the deck, then
  ``archive_day archive`` (+ ``verify``) — a size-verified COPY of raw ``.dat`` + motion-corrected
  ``.bin`` to M: standby and outputs to N:.

**Analysis / behavior-GPU box** (camera work FIRST, then LocaNMF analysis):
  ``camera_nightly`` — upload D: -> MICROSCOPE (camera videos/CSVs **AND behavior logs**) + dropped-frame
  QC + camera<->DAQ alignment templates — runs first because it needs no imaging output; THEN
  ``nightly_figs`` (LocaNMF decode/encode/RSA + deck), which has to wait for the imaging box's
  preprocessing/LocaNMF push.

**Never deletes anything.** E: cleanup (``archive_day clean --execute``) and D: cleanup stay MANUAL,
run only after byte-verify + check-in (ground rule 1). A failed step stops the chain, so a bad night
never loses local data.

    python -m wfield_local.nightly 20260808                      # this box's full nightly
    python -m wfield_local.nightly 20260808 --only PS94
    python -m wfield_local.nightly 20260808 --dry-run
    python -m wfield_local.nightly 20260808 --skip-camera        # analysis box: just the figs
    python -m wfield_local.nightly 0806-0808 --skip-archive      # imaging box: no archive copy

Each step runs as a ``python -m wfield_local.<step>`` subprocess (isolation; same pattern as the steps
themselves). The date grammar is shared with ``preprocess`` / ``nightly_figs``.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from wfield_local import config
from wfield_local.paths import PathResolver
from wfield_local.preprocess import list_raw_dates

REPO = Path(__file__).resolve().parents[1]


def _run(args, dry: bool) -> None:
    cmd = [sys.executable, "-m"] + [str(a) for a in args]
    print("\n$ " + " ".join(cmd), flush=True)
    if dry:
        return
    if subprocess.run(cmd, cwd=str(REPO)).returncode:
        raise SystemExit(f"[nightly] FAILED: {' '.join(str(a) for a in args)}")


def _available_dates(rv: PathResolver) -> list[str]:
    """YYYYMMDD dirs to resolve ranges / ``all`` against: raw acquisition (imaging) or camera dirs (analysis)."""
    if rv.machine == "imaging":
        return list_raw_dates(rv)
    root = Path(rv.root("behavior_cameras"))
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name))


def _imaging_nightly(dates, args, mach, only) -> int:
    _run(["wfield_local.preprocess", *dates, *only, *(["--dry-run"] if args.dry_run else []), *mach], dry=False)
    if not args.skip_deck:
        _run(["wfield_local.preprocess_deck", *mach], dry=args.dry_run)
    if not args.skip_archive:
        for d in dates:
            _run(["wfield_local.archive_day", "archive", "--date", d, *mach], dry=args.dry_run)
            _run(["wfield_local.archive_day", "verify", "--date", d, *mach], dry=args.dry_run)
        print("\n[nightly] E: cleanup is MANUAL (never automatic) — after you verify the copies + check in:")
        for d in dates:
            print(f"    python -m wfield_local.archive_day clean --date {d} --execute  # delete from E:")
    print(f"\nNIGHTLY {' '.join(dates)} (imaging: preprocess -> deck -> archive) DONE", flush=True)
    return 0


def _analysis_nightly(dates, args, mach, only) -> int:
    # camera work first — it needs no imaging output, so it runs while the imaging box is still preprocessing
    if not args.skip_camera:
        for d in dates:
            _run(["wfield_local.camera_nightly", d, *only, *(["--dry-run"] if args.dry_run else []), *mach], dry=False)
    # then the LocaNMF analysis + deck (this waits for the imaging box's preprocessing/LocaNMF push)
    if not args.skip_figs:
        frm = ["--from", args.from_dates] if args.from_dates else []
        _run(["wfield_local.nightly_figs", *dates, *only, *frm], dry=args.dry_run)
    print(f"\nNIGHTLY {' '.join(dates)} (analysis: camera -> LocaNMF figs) DONE", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dates", nargs="+", metavar="DATE", help="date(s), shared grammar (MMDD/YYYYMMDD/range/all)")
    ap.add_argument("--only", nargs="+", metavar="ANIMAL", help="restrict to these animals")
    ap.add_argument("--from", dest="from_dates", default=None, help="analysis: cross-session span for nightly_figs")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    ap.add_argument("--skip-camera", action="store_true", help="analysis: skip the camera upload/QC/alignment")
    ap.add_argument("--skip-figs", action="store_true", help="analysis: skip the LocaNMF figs")
    ap.add_argument("--skip-deck", action="store_true", help="imaging: skip the preprocess_deck rebuild")
    ap.add_argument("--skip-archive", action="store_true", help="imaging: skip the archive COPY")
    ap.add_argument("--machine", default=None, help="override machine (default: auto-detect)")
    args = ap.parse_args(argv)

    rv = PathResolver(machine=args.machine)
    dates = config.expand_dates(args.dates, width=8, available=_available_dates(rv))
    if not dates:
        print(f"[nightly] no dates resolved from {args.dates} on machine={rv.machine}")
        return 1
    mach = ["--machine", args.machine] if args.machine else []
    only = ["--only", *args.only] if args.only else []

    print(f"[nightly] machine={rv.machine}  dates={dates}", flush=True)
    if rv.machine == "imaging":
        return _imaging_nightly(dates, args, mach, only)
    return _analysis_nightly(dates, args, mach, only)


if __name__ == "__main__":
    raise SystemExit(main())
