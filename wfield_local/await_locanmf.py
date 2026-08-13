"""Poll MICROSCOPE for a night's LocaNMF inputs, then fully-automatically run LocaNMF, register the
sessions, and refresh the figs -- the analysis box's "wait for the imaging push" loop, packaged so it
runs standalone from PowerShell (no Claude Code session needed).

Each tick (default every 30 min) it:
  1. DETECTS ready sessions on MICROSCOPE: for each mouse, ``labcams/<YYYYMMDD>/PSxx_<YYYYMMDD>_*/
     motion_corrected/wfield_local_results/{SVTcorr.npy, allen_aligned_affine8v1/U_atlas.npy}`` present.
  2. For each ready mouse whose LocaNMF has NOT run yet (no LocaNMF output dir -- ``config.locanmf_dir_name()``): runs
     ``batch_locanmf`` (r2 0.95 / loc 80 / maxrank 20 from configs/defaults.yaml).
  3. REGISTERS any ready+unregistered mouse in ``configs/sessions.yaml`` (regime B if a
     ``*cleanpairs_frame_map.npz`` is present, else A; fmdir null), preserving the file's comments/format.
  4. If anything was newly registered: commit + push ``configs/sessions.yaml`` (rig procedure), then run
     ``nightly_figs`` for the date.
  5. Exits 0 once every requested mouse is registered; otherwise sleeps and re-checks.

Fully automatic by design (the user opted in). Escapes: ``--once`` (single pass), ``--no-push`` (commit
locally, don't push), ``--no-figs`` (skip the figs refresh), ``--no-locanmf`` (register only; assume LocaNMF
already ran), ``--dry-run`` (print the plan, touch nothing). GPU is required for the LocaNMF step.

    python -m wfield_local.await_locanmf 20260809                  # poll every 30 min until all 4 mice done
    python -m wfield_local.await_locanmf 20260809 --once --dry-run # one detection pass, no writes
    python -m wfield_local.await_locanmf 0809 --animals PS93 PS94  # just two mice
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from wfield_local import config
from wfield_local.paths import PathResolver

REPO = Path(__file__).resolve().parents[1]
SESSIONS_YAML = REPO / "configs" / "sessions.yaml"
TAG = "affine8v1"
ANIMALS_ALL = ["PS92", "PS93", "PS94", "PS95"]


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def log(m: str) -> None:
    print(f"[await {_stamp()}] {m}", flush=True)


def _ensure_conda_prefix() -> None:
    """Self-heal ``CONDA_PREFIX`` from the running interpreter when it is unset.

    The poller is meant to run detached (Start-Process / a bare PowerShell), and such a launch often
    carries no ``CONDA_PREFIX`` in its environment. The git ``pre-push`` hook resolves its Python from
    ``CONDA_PREFIX`` (to dodge the App Store shim), so without it every auto-register push dies with
    ``pre-push: no usable python with pytest`` and the sessions.yaml commit never reaches origin. The
    poller always runs in its own conda env, so ``sys.executable`` is that env's python -- derive the
    prefix from it and export it so the hook (and the git children that inherit os.environ) can find pytest."""
    if os.environ.get("CONDA_PREFIX"):
        return
    exe = Path(sys.executable)
    prefix = exe.parent.parent if exe.parent.name in ("bin", "Scripts") else exe.parent
    os.environ["CONDA_PREFIX"] = str(prefix)
    log(f"CONDA_PREFIX was unset -> derived {prefix} from the interpreter (for the git pre-push hook)")


# --------------------------------------------------------------------------- detection

def discover(rv: PathResolver, yyyymmdd: str, animals: list[str]) -> list[dict]:
    """Ready LocaNMF-input sessions for the date, one dict per mouse found.

    A mouse is INPUTS-READY when SVTcorr.npy + allen_aligned_<tag>/U_atlas.npy both exist under its
    session's motion_corrected/wfield_local_results. Also reports regime (frame_map present -> B), whether
    LocaNMF already ran (locanmf_<tag>_final present), whether it is already in sessions.yaml, and the
    root-relative mc/h5 fields needed to register it.
    """
    config._load.cache_clear()   # re-read sessions.yaml fresh: this long-running process may have just
                                 # registered a mouse (config caches YAML loads for the process lifetime)
    date_dir = Path(rv.resolve("labcams", yyyymmdd))
    registered = {s["label"] for s in config.load_sessions()}
    out: list[dict] = []
    for animal in animals:
        sess_dirs = sorted(glob.glob(str(date_dir / f"{animal}_{yyyymmdd}_*")))
        sess_dirs = [p for p in sess_dirs if Path(p).is_dir()]
        if not sess_dirs:
            continue
        sdir = Path(sess_dirs[-1])              # newest session for that mouse that day
        mc = sdir / "motion_corrected"
        res = mc / "wfield_local_results"
        allen = res / f"allen_aligned_{TAG}"
        svt = res / "SVTcorr.npy"
        u_atlas = allen / "U_atlas.npy"
        if not (svt.exists() and u_atlas.exists()):
            continue                            # inputs not fully pushed yet
        frame_map = bool(glob.glob(str(mc / "*cleanpairs_frame_map.npz")))
        locanmf_out = mc / f"locanmf_{TAG}_final"
        locanmf_done = locanmf_out.exists() and bool(glob.glob(str(locanmf_out / "*_C.npy")))
        # DAQ h5 for this mouse+date on MICROSCOPE (daq_recorder_output root, one date level)
        h5_glob = glob.glob(str(Path(rv.resolve("daq_recorder_output", yyyymmdd)) / f"{animal}_{yyyymmdd}_*.h5"))
        mmdd = yyyymmdd[4:8]
        out.append({
            "animal": animal, "mmdd": mmdd, "yyyymmdd": yyyymmdd, "session": sdir.name,
            "mc_dir": str(mc), "allen_dir": str(allen), "svt": str(svt),
            "locanmf_out": str(locanmf_out), "locanmf_done": locanmf_done,
            "regime": "B" if frame_map else "A",
            "registered": f"{animal}_{mmdd}" in registered,
            "mc_rel": f"{yyyymmdd}/{sdir.name}/motion_corrected",
            "h5_rel": (f"{yyyymmdd}/{Path(sorted(h5_glob)[0]).name}" if h5_glob else None),
        })
    return out


# --------------------------------------------------------------------------- sessions.yaml registration

def insert_session_entry(text: str, animal: str, mmdd: str, mc: str, h5: str,
                         regime: str, fmdir) -> str:
    """Return sessions.yaml `text` with a new `mmdd` entry appended at the end of `animal`'s block,
    preserving all comments/formatting. Idempotent: if `animal`+`mmdd` already present, returns unchanged.

    Assumes the canonical shape: ``sessions:`` then 2-space animal headers (``  PS92:``), 4-space quoted
    date keys (``    "0807":``), 6-space fields. New (chronologically-latest) dates go at the block end.
    """
    lines = text.splitlines(keepends=True)
    hdr = f"  {animal}:"
    ai = next((i for i, ln in enumerate(lines) if ln.rstrip("\n") == hdr), None)
    if ai is None:
        raise ValueError(f"animal header '{hdr}' not found in sessions.yaml")
    # block = from ai+1 up to the next 2-space animal header (or EOF)
    bj = len(lines)
    for i in range(ai + 1, len(lines)):
        if lines[i].startswith("  ") and not lines[i].startswith("   ") and lines[i].strip().endswith(":"):
            bj = i
            break
    block = lines[ai + 1:bj]
    if any(ln.rstrip("\n").strip() == f'"{mmdd}":' for ln in block):
        return text                              # already registered -> no-op
    # insertion point = after the last non-blank line of the block (before any trailing blanks)
    ins = bj
    while ins > ai + 1 and lines[ins - 1].strip() == "":
        ins -= 1
    fmdir_val = "null" if fmdir in (None, "null") else f'"{fmdir}"'
    entry = (
        f'    "{mmdd}":\n'
        f'      mc: "{mc}"\n'
        f'      h5: "{h5}"\n'
        f'      regime: {regime}\n'
        f'      fmdir: {fmdir_val}\n'
    )
    # guarantee the preceding line ends with a newline
    if ins > 0 and not lines[ins - 1].endswith("\n"):
        lines[ins - 1] = lines[ins - 1] + "\n"
    return "".join(lines[:ins]) + entry + "".join(lines[ins:])


def register(entries: list[dict], dry: bool) -> list[str]:
    """Insert each entry into sessions.yaml (skips ones already registered / missing an h5). Returns the
    list of ``PSxx_mmdd`` labels newly written."""
    text = SESSIONS_YAML.read_text(encoding="utf-8")
    written = []
    for e in entries:
        if e["registered"]:
            continue
        if not e["h5_rel"]:
            log(f"  !! {e['animal']} {e['mmdd']}: no DAQ .h5 on MICROSCOPE yet -> cannot register, will retry")
            continue
        new = insert_session_entry(text, e["animal"], e["mmdd"], e["mc_rel"], e["h5_rel"], e["regime"], None)
        if new != text:
            text = new
            written.append(f"{e['animal']}_{e['mmdd']}")
            log(f"  register {e['animal']} {e['mmdd']} (regime {e['regime']})")
    if written and not dry:
        SESSIONS_YAML.write_text(text, encoding="utf-8")
    return written


# --------------------------------------------------------------------------- actions

def _run(cmd: list[str], dry: bool, cwd: Path = REPO) -> int:
    log("$ " + " ".join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def run_locanmf(entry: dict, dry: bool) -> bool:
    """Run batch_locanmf for one session. Returns True on success (or dry-run)."""
    manifest = REPO / f".await_locanmf_{entry['animal']}_{entry['mmdd']}.json"
    # label MUST be "<animal>_<mmdd>" (e.g. PS92_0809): run_locanmf appends "_locanmf_{A,C,regions}.npy",
    # and the decoder/encoder/RSA load "<label>_locanmf_C.npy". A trailing "_locanmf" here doubled it
    # (PS92_0809_locanmf_locanmf_C.npy) so every downstream fig failed to find the LocaNMF output.
    label = f"{entry['animal']}_{entry['mmdd']}"
    spec = [{"allen_dir": entry["allen_dir"], "label": label,
             "output": str(Path(entry["mc_dir"]) / f"locanmf_{TAG}_final"), "svt": entry["svt"]}]
    log(f"  LocaNMF {entry['animal']} {entry['mmdd']} -> {spec[0]['output']}")
    if dry:
        log(f"    [dry-run] manifest={spec}")
        return True
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    try:
        rc = _run([sys.executable, "-u", "-m", "wfield_local.batch_locanmf", "--manifest", str(manifest)], dry=False)
    finally:
        manifest.unlink(missing_ok=True)
    return rc == 0


def commit_push(labels: list[str], no_push: bool, dry: bool) -> None:
    msg = f"sessions.yaml: auto-register {', '.join(labels)} (await_locanmf)"
    if _run(["git", "add", str(SESSIONS_YAML)], dry) or _run(["git", "commit", "-m", msg], dry):
        log("  !! git add/commit failed"); return
    if no_push:
        log("  committed locally (--no-push)"); return
    for attempt in range(3):
        _run(["git", "fetch", "origin"], dry)
        if _run(["git", "rebase", "origin/main"], dry):
            log("  !! rebase failed -- resolve manually; not pushing"); return
        if _run(["git", "push", "origin", "main"], dry) == 0:
            log("  pushed"); return
        log(f"  push rejected (attempt {attempt + 1}/3) -- re-fetch/rebase")
    log("  !! push failed after retries -- push manually")


# --------------------------------------------------------------------------- loop

def tick(rv, yyyymmdd, animals, args) -> set[str]:
    """One detection+action pass. Returns the set of animals now registered for the date."""
    ready = discover(rv, yyyymmdd, animals)
    have = {e["animal"] for e in ready}
    waiting = [a for a in animals if a not in have]
    log(f"date {yyyymmdd}: inputs ready {sorted(have) or 'none'} | waiting on {waiting or 'none'}")

    # 1. LocaNMF for ready-but-not-yet-run mice
    if not args.no_locanmf:
        for e in ready:
            if not e["locanmf_done"]:
                if run_locanmf(e, args.dry_run):
                    e["locanmf_done"] = True
    # 2. register newly-ready mice, then commit/push + figs if anything changed
    written = register(ready, args.dry_run)
    if written:
        commit_push(written, args.no_push, args.dry_run)
        if not args.no_figs:
            _run([sys.executable, "-u", "-m", "wfield_local.nightly_figs", yyyymmdd], args.dry_run)

    registered_now = {e["animal"] for e in discover(rv, yyyymmdd, animals) if e["registered"]}
    if args.dry_run:                             # dry-run never writes, so treat written as "would-register"
        registered_now |= set(a for a in animals for w in written if w.startswith(a))
    return registered_now


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", metavar="DATE", help="the night to await (MMDD or YYYYMMDD)")
    ap.add_argument("--animals", nargs="+", metavar="ANIMAL", default=None, help="subset (default: all four)")
    ap.add_argument("--interval-min", type=float, default=30.0, help="minutes between checks (default 30)")
    ap.add_argument("--once", action="store_true", help="single detection pass, then exit")
    ap.add_argument("--no-locanmf", action="store_true", help="register only; assume LocaNMF already ran")
    ap.add_argument("--no-figs", action="store_true", help="skip the nightly_figs refresh after registration")
    ap.add_argument("--no-push", action="store_true", help="commit sessions.yaml locally but do not push")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; write/commit/run nothing")
    ap.add_argument("--machine", default=None, help="override machine (default: auto-detect)")
    args = ap.parse_args(argv)

    _ensure_conda_prefix()   # so the git pre-push hook can find pytest even when launched detached
    rv = PathResolver(machine=args.machine)
    d = args.date if len(args.date) == 8 else None
    if d is None:                               # accept MMDD -> assume the imaging cohort year 2026
        d = f"2026{args.date}" if len(args.date) == 4 else args.date
    if len(d) != 8 or not d.isdigit():
        ap.error(f"date must be MMDD or YYYYMMDD, got {args.date!r}")
    animals = config.normalize_animals(args.animals) or ANIMALS_ALL

    log(f"awaiting LocaNMF inputs for {d}, animals={animals}, every {args.interval_min:g} min"
        + (" [DRY-RUN]" if args.dry_run else ""))
    while True:
        done = tick(rv, d, animals, args)
        remaining = [a for a in animals if a not in done]
        if not remaining:
            log(f"ALL DONE -- {d} registered for {animals}. Exiting.")
            return 0
        if args.once:
            log(f"--once: stopping. Still waiting on {remaining}.")
            return 0
        log(f"still waiting on {remaining}; next check in {args.interval_min:g} min")
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    raise SystemExit(main())
