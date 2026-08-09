"""Camera-side nightly orchestrator (analysis / behavior-GPU box).

One command per date runs the camera nightly, in order:

  0. **Upload** ``D:`` staging -> MICROSCOPE (size-verified, guarded; NEVER deletes ``D:``):
     ``D:\\camera\\<date>\\<PSxx>\\*`` -> ``Behavior_Cameras/Widefield/<date>/<PSxx>/`` and
     ``D:\\behavior_logs\\<PSxx>_<date>_*`` -> ``Behavior_logs/Widefield/<session>/``.
  1. **Dropped-frame QC** (:mod:`wfield_local.dropframe_qc`) -> ``dropped_frames_summary_<date>.{csv,txt}``.
  2. **Camera<->DAQ alignment templates** (:mod:`wfield_local.camera_sync`) ->
     ``alignment_templates/<cam>/<PSxx>/<date>.npz``.
  3. **Spout behavior figures** (:mod:`wfield_local.spout_behavior`) -> per-session behavior PNG +
     per-position metrics, and a refresh of the curated cross-session cohort summary.

Copies are idempotent + size-verified; a copy FAILURE stops the run before QC/align (never process a
partial upload). ``D:`` deletion is a separate manual step after byte-verification + check-in. Writes only
under MICROSCOPE/Priya (writeguard-checked). Mirrors the imaging box's ``preprocess`` orchestrator.

    python -m wfield_local.camera_nightly 20260807
    python -m wfield_local.camera_nightly 20260807 --only PS94        # one animal (all steps)
    python -m wfield_local.camera_nightly 20260807 --dry-run          # plan only, no writes
    python -m wfield_local.camera_nightly 20260807 --skip-copy        # data already on MICROSCOPE
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from wfield_local import camera_sync, config, dropframe_qc, spout_behavior, writeguard
from wfield_local.paths import PathResolver

BUF = 1 << 20
ANIMAL_RE = re.compile(r"PS\d+")


def _copy_one(src: str, dst: str, dry: bool) -> str:
    """Idempotent size-verified copy (skip if same-size dst exists; .part temp then atomic replace)."""
    if dry:
        return "dry"
    ssz = os.path.getsize(src)
    if os.path.exists(dst) and os.path.getsize(dst) == ssz:
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with open(src, "rb", buffering=0) as fi, open(tmp, "wb", buffering=0) as fo:
        while True:
            b = fi.read(BUF)
            if not b:
                break
            fo.write(b)
    os.replace(tmp, dst)
    return "ok" if os.path.getsize(dst) == ssz else "FAIL"


def _copy_tree(src_dir: Path, dst_dir: Path, dry: bool, res: dict, fails: list) -> None:
    """Copy every file under src_dir to dst_dir at the same relative path (guarded, size-verified)."""
    writeguard.assert_writable(dst_dir)                    # never write outside MICROSCOPE/Priya
    for f in sorted(x for x in src_dir.rglob("*") if x.is_file()):
        dst = str(dst_dir / f.relative_to(src_dir))
        st = _copy_one(str(f), dst, dry)
        res[st] = res.get(st, 0) + 1
        if st == "FAIL":
            fails.append(dst)


def upload(date, rv, animals=None, dry=False) -> tuple[dict, list]:
    """Copy the date's camera videos/CSVs AND behavior logs from ``D:`` to MICROSCOPE. Never deletes ``D:``."""
    res, fails = {"ok": 0, "skip": 0, "FAIL": 0, "dry": 0}, []
    aset = set(animals) if animals else None

    def _group(label, jobs):
        sub = {"ok": 0, "skip": 0, "FAIL": 0, "dry": 0}
        for src, dst in jobs:
            _copy_tree(src, dst, dry, sub, fails)
        for k, v in sub.items():
            res[k] += v
        print(f"[camera_nightly] {label}: {sub}", flush=True)

    # cameras: D:/camera/<date>/<PSxx>/*  ->  Behavior_Cameras/Widefield/<date>/<PSxx>/
    cam_stg = Path(rv.staging("camera_local"))
    cam_src = cam_stg / date if (cam_stg / date).is_dir() else cam_stg
    cam_dst = Path(rv.resolve("behavior_cameras", date))
    cam_jobs = [(a, cam_dst / a.name) for a in
                (sorted(p for p in cam_src.iterdir() if p.is_dir() and ANIMAL_RE.fullmatch(p.name))
                 if cam_src.is_dir() else [])
                if not aset or a.name in aset]
    _group("cameras", cam_jobs)

    # behavior logs: D:/behavior_logs/<PSxx>_<date>_<hhmmss>/*  ->  Behavior_logs/Widefield/<session>/
    log_stg = Path(rv.staging("behavior_logs_local"))
    log_dst = Path(rv.root("behavior_logs"))
    log_jobs = [(s, log_dst / s.name) for s in
                (sorted(p for p in log_stg.iterdir() if p.is_dir() and date in p.name)
                 if log_stg.is_dir() else [])
                if not aset or any(s.name.startswith(x) for x in aset)]
    _group("behavior logs", log_jobs)

    print(f"[camera_nightly] upload D:->MICROSCOPE total (D: untouched): {res}", flush=True)
    for f in fails[:10]:
        print(f"   FAIL {f}", flush=True)
    return res, fails


def run(date, rv, animals=None, do_copy=True, do_dropframe=True, do_align=True, do_behavior=True,
        dry=False) -> int:
    """Upload -> dropped-frame QC -> alignment templates -> behavior figs for ``date``.

    Returns 0 on success, 1 on copy fail (stops before any downstream step)."""
    if do_copy:
        print("\n################ upload D: -> MICROSCOPE (D: NOT deleted) ################", flush=True)
        _res, fails = upload(date, rv, animals, dry)
        if fails:
            print(f"[camera_nightly] {len(fails)} copy FAILURE(s) -> stopping before QC/align "
                  f"(nothing deleted).", flush=True)
            return 1
    if do_dropframe:
        print("\n################ dropped-frame QC ################", flush=True)
        if dry:
            print(f"[dry-run] dropframe_qc.run(behavior_cameras/{date})", flush=True)
        else:
            dropframe_qc.run(rv.resolve("behavior_cameras", date), date, animals=animals)
    if do_align:
        print("\n################ camera<->DAQ alignment templates ################", flush=True)
        if dry:
            print(f"[dry-run] camera_sync.run({date})", flush=True)
        else:
            camera_sync.run(date, rv, animals=animals)
    if do_behavior:
        print("\n################ spout behavior figures (+ curated cohort) ################", flush=True)
        spout_behavior.run(date, rv, animals=animals, cohort=True, from_spec="curated", dry=dry)
    print(f"\nCAMERA NIGHTLY {date} DONE", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", metavar="YYYYMMDD")
    ap.add_argument("--only", nargs="+", metavar="ANIMAL", help="restrict to these animals, or 'all'")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; copy nothing, write nothing")
    ap.add_argument("--skip-copy", action="store_true", help="data already on MICROSCOPE; skip the upload")
    ap.add_argument("--skip-dropframe", action="store_true", help="skip the dropped-frame QC pass")
    ap.add_argument("--skip-align", action="store_true", help="skip the alignment-template pass")
    ap.add_argument("--skip-behavior", action="store_true", help="skip the spout behavior figures")
    ap.add_argument("--machine", default=None, help="override machine (default: auto-detect)")
    args = ap.parse_args(argv)
    return run(args.date, PathResolver(machine=args.machine), animals=config.normalize_animals(args.only),
               do_copy=not args.skip_copy, do_dropframe=not args.skip_dropframe,
               do_align=not args.skip_align, do_behavior=not args.skip_behavior, dry=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
