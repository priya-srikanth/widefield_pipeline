"""Camera-side nightly orchestrator (analysis / behavior-GPU box).

Once a date's Blackfly recordings are on MICROSCOPE (`behavior_cameras/<date>`; the upload is a
separate step), this runs the two camera processing steps in one command:

  1. **Dropped-frame QC** (:mod:`wfield_local.dropframe_qc`) — per cam CSV, flag gaps in the monotonic
     frame_id + long timestamp deltas; writes ``dropped_frames_summary_<date>.{csv,txt}`` with the data.
  2. **Camera<->DAQ alignment templates** (:mod:`wfield_local.camera_sync`) — match each cam's GPIO sync
     train to the DAQ ``sync`` line and write ``alignment_templates/<cam>/<PSxx>/<date>.npz``.

Read-only on the camera CSVs + DAQ ``.h5``; writes only the summary + templates (both under Priya, so
writeguard-safe). Does NOT copy or delete anything. Mirrors the imaging box's ``preprocess`` orchestrator.

    python -m wfield_local.camera_nightly 20260807
    python -m wfield_local.camera_nightly 20260807 --only PS94        # one animal (both steps)
    python -m wfield_local.camera_nightly 20260807 --skip-dropframe   # just (re)build templates
"""
from __future__ import annotations

import argparse

from wfield_local import camera_sync, config, dropframe_qc
from wfield_local.paths import PathResolver


def run(date, rv, animals=None, do_dropframe=True, do_align=True) -> None:
    """Dropped-frame QC then alignment templates for ``date`` (both on MICROSCOPE)."""
    if do_dropframe:
        print("\n################ dropped-frame QC ################", flush=True)
        dropframe_qc.run(rv.resolve("behavior_cameras", date), date, animals=animals)
    if do_align:
        print("\n################ camera<->DAQ alignment templates ################", flush=True)
        camera_sync.run(date, rv, animals=animals)
    print(f"\nCAMERA NIGHTLY {date} DONE", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", metavar="YYYYMMDD")
    ap.add_argument("--only", nargs="+", metavar="ANIMAL", help="restrict to these animals, or 'all'")
    ap.add_argument("--skip-dropframe", action="store_true", help="skip the dropped-frame QC pass")
    ap.add_argument("--skip-align", action="store_true", help="skip the alignment-template pass")
    ap.add_argument("--machine", default=None, help="override machine (default: auto-detect)")
    args = ap.parse_args(argv)
    run(args.date, PathResolver(machine=args.machine), animals=config.normalize_animals(args.only),
        do_dropframe=not args.skip_dropframe, do_align=not args.skip_align)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
