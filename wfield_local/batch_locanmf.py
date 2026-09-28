"""Batch LocaNMF at fixed (chosen) params across many sessions -- kill-safe, resumable.

Runs ``run_locanmf`` once per session listed in a JSON manifest, all at the same
``r2_thresh``/``loc_thresh``/``maxrank``, each into its own output dir. Like
``sweep_locanmf.py`` it writes a session's outputs only on completion and SKIPS any
session whose ``<label>_locanmf_summary.json`` already exists -- so the batch can be
terminated anytime (e.g. to free the machine) and resumes where it stopped.

STALE COUNTS AS MISSING (2026-09-28). A summary OLDER than the ``svt`` it was fitted to is not
"done", it is a fit of data that no longer exists. Existence alone was the test, and it would have
kept PS92_0922's 2026-09-22 decomposition -- fitted to the frame-misaligned SVTcorr the head-offset
bug produced (docs/EXPERIMENT_ERRORS.md) -- after the session was re-preprocessed, with every
downstream decoder, encoder and RSA number for that session still built on it and nothing
reporting a problem. ``await_locanmf`` gates on the same existence test and hands the decision
here, and ``nightly_figs`` already applies this rule to its figures. The stale outputs are MOVED
aside (``<output>_stale_<timestamp>/``), never deleted: MICROSCOPE derived outputs are
read-only inputs to everyone else (CLAUDE.md rule 1), and the previous fit is what a before/after
comparison needs.

To avoid overwriting/deleting any existing MICROSCOPE outputs, point ``output`` at a
NEW per-session folder (``config.locanmf_dir_name()``, which NAMES the hemodynamic variant the
decomposition was fitted to -- see docs/PREPROCESSING_DECISION.md); the runner only creates/writes.

Manifest JSON = list of {"allen_dir","label","output"[, "svt"]}:
    python -m wfield_local.batch_locanmf --manifest sessions.json \
        --r2 0.95 --loc 80 --maxrank 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from wfield_local import config


def _stamp(msg):
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"


def stale_input(summary: Path, svt) -> float | None:
    """Seconds by which ``svt`` (the SVTcorr the fit read) post-dates ``summary``, or None if the
    fit is current. No ``svt`` in the manifest, or one that is not on disk, means nothing can be
    compared and the fit is trusted -- the pre-2026-09-28 behaviour."""
    if not svt:
        return None
    svt = Path(svt)
    if not svt.exists() or not summary.exists():
        return None
    # 2 s of slack: the push copies with copy2, and FAT/SMB mtimes are 2 s granular. A fit is only
    # stale if its input is CLEARLY newer than it.
    gap = svt.stat().st_mtime - summary.stat().st_mtime
    return gap if gap > 2.0 else None


def set_aside_stale(outdir: Path) -> Path:
    """Rename ``outdir`` to ``<outdir>_stale_<YYYYMMDD_HHMMSS>`` beside itself and return the new path.
    A rename, not a delete (rule 1); the caller then recreates ``outdir`` and refits into it."""
    target = outdir.with_name(f"{outdir.name}_stale_{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.rename(target)
    return target


def main() -> int:
    lp = config.defaults()["locanmf"]     # r2/loc/maxrank single source of truth (configs/defaults.yaml)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="JSON list of {allen_dir,label,output[,svt]}")
    ap.add_argument("--r2", type=float, default=lp["r2_thresh"])
    ap.add_argument("--loc", type=float, default=lp["loc_thresh"])
    ap.add_argument("--maxrank", type=int, default=lp["maxrank"])
    ap.add_argument("--mode", default="locanmf", choices=("locanmf", "snmf", "both"))
    ap.add_argument("--log", type=Path, default=None, help="master log (default: alongside manifest)")
    args = ap.parse_args()

    sessions = json.loads(args.manifest.read_text())
    master_path = args.log or args.manifest.with_suffix(".batch.log")
    master = open(master_path, "a", buffering=1)

    def log(msg):
        line = _stamp(msg)
        print(line, flush=True)
        master.write(line + "\n")

    log(f"batch start: {len(sessions)} sessions  params r2={args.r2} loc_thresh={args.loc} maxrank={args.maxrank}")
    done = 0
    for s in sessions:
        label = s["label"]
        outdir = Path(s["output"])
        summary = outdir / f"{label}_locanmf_summary.json"
        if summary.exists():
            stale_by = stale_input(summary, s.get("svt"))
            if stale_by is None:
                log(f"SKIP {label} (already complete: {summary})")
                done += 1
                continue
            moved = set_aside_stale(outdir)
            log(f"STALE {label}: {summary.name} is {stale_by:.0f} s OLDER than its input "
                f"{s.get('svt')} -> moved the previous fit to {moved}; refitting")
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "wfield_local.run_locanmf",
               "--allen-dir", str(s["allen_dir"]), "--label", label,
               "--output", str(outdir), "--mode", args.mode, "--maxrank", str(args.maxrank),
               "--loc-thresh", str(args.loc), "--r2-thresh", str(args.r2), "--device", "auto"]
        if s.get("svt"):
            cmd += ["--svt", str(s["svt"])]
        log(f"START {label} -> {outdir}")
        t0 = time.time()
        runlog = open(outdir / "run.log", "w", buffering=1)
        try:
            rc = subprocess.call(cmd, stdout=runlog, stderr=subprocess.STDOUT)
        except KeyboardInterrupt:
            runlog.close()
            log(f"INTERRUPTED during {label} after {time.time()-t0:.0f}s; {done} session(s) saved. Exiting.")
            master.close()
            return 130
        runlog.close()
        if rc == 0 and summary.exists():
            log(f"DONE {label}: exit 0 in {time.time()-t0:.0f}s")
            done += 1
        else:
            log(f"FAILED {label}: exit {rc} in {time.time()-t0:.0f}s (see {outdir/'run.log'})")

    log(f"batch complete: {done}/{len(sessions)} sessions have outputs")
    master.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
