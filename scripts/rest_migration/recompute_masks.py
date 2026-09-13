"""Build the REST masks for every registered session, into `quiet_<tag>_<variant>/`.

NOTHING IS OVERWRITTEN. The retired `quiet_<tag>/` masks stay exactly where they are -- they are the
provenance of every figure produced before this migration, and `{mc}/` is read-only under rule 1
anyway. This writes a sibling directory named for the variant, which is the same discipline
`docs/PREPROCESSING_DECISION.md` sets for the hemodynamic variants. Rolling back is one config line.

IT DRIVES `quiet_periods` RATHER THAN REIMPLEMENTING IT. The per-corrected-frame mapping is the
fiddly part (regime B needs the cleanpairs frame map and its chosen exposure offset; regime A takes
every second PCO edge), it is already written and tested there, and a second copy of it would be a
second thing to keep in step -- which is the exact failure this whole migration exists to fix.

RESUMABLE: a session whose output already exists is skipped, so an interrupted run costs only the
session it was on. Pass --force to rebuild anyway.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path


def _assert_package_matches_cwd():
    """Refuse to run if `wfield_local` came from a DIFFERENT checkout than the config being read.

    IN A GIT WORKTREE THIS IS A REAL HAZARD, not a theoretical one. The repo is installed editable
    against the MAIN checkout, and `python scripts/rest_migration/x.py` puts the SCRIPT's directory
    on `sys.path[0]` -- not the working directory -- so `import wfield_local` resolves through the
    editable install to the main tree while `configs/` is read from wherever the code that got
    imported lives. Running the same file as `python -m scripts.rest_migration.x` puts the CWD first
    and imports the worktree instead. The two produce DIFFERENT DEFINITIONS with no error: this
    script was refused as "variant is empty" under one form and ran under the other, which is how it
    was found. A script that WRITES masks must not be able to silently use another branch's code.
    """
    import wfield_local

    pkg = Path(wfield_local.__file__).resolve().parent.parent
    cwd = Path.cwd().resolve()
    if pkg != cwd:
        raise SystemExit(
            f"!! `wfield_local` was imported from {pkg}\n"
            f"   but the working directory is {cwd}.\n"
            f"   Invoke as:  python -m scripts.rest_migration.recompute_masks\n"
            f"   (running the file by path puts the script's own directory on sys.path, so the\n"
            f"    editable install wins and another checkout's definition would be written.)")


def main(argv=None) -> int:
    _assert_package_matches_cwd()
    from wfield_local import config
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.quiet_periods import quiet_dir, quiet_variant

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild sessions that already have masks")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    variant = quiet_variant()
    if not variant:
        print("!! segmentation.rest.variant is empty -- that is the RETIRED directory. Refusing to "
              "write, because this would overwrite the old masks in place.", flush=True)
        return 2

    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [s for s in SESSIONS if s["label"] in want and s.get("h5")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[rest] variant {variant!r}; {len(todo)} sessions", flush=True)

    t0, made, skipped, failed = time.time(), 0, 0, []
    for i, s in enumerate(todo, 1):
        out = Path(quiet_dir(s["mc"]))
        if not args.force and sorted(out.glob("*_quiet_frame.npy")):
            skipped += 1
            continue
        cmd = [sys.executable, "-m", "wfield_local.quiet_periods",
               "--daq-h5", str(s["h5"]), "--label", s["label"], "--output", str(out)]
        # REGIME B needs the frame map + the chosen exposure offset; regime A is raw//2. Same
        # resolution `locanmf_cue_lick_analysis` uses, so a session cannot be mapped one way here
        # and another way there.
        if s.get("regime") == "B":
            fmdir = s.get("fmdir") or s["mc"]
            fm = glob.glob(f"{fmdir}/*cleanpairs_frame_map.npz")
            summ = glob.glob(f"{fmdir}/*cleanpairs_summary.json")
            if not fm or not summ:
                failed.append((s["label"], "regime B without a cleanpairs frame map"))
                continue
            cmd += ["--frame-map", fm[0], "--cleanpairs-summary", summ[0]]
        if args.dry_run:
            print(f"  [{i}/{len(todo)}] DRY {s['label']} -> {out}", flush=True)
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            failed.append((s["label"], (r.stderr or r.stdout or "").strip()[-200:]))
            print(f"  [{i}/{len(todo)}] !! {s['label']}: {failed[-1][1][:120]}", flush=True)
            continue
        anchor = next((ln.split("anchored on", 1)[1].strip()
                       for ln in r.stdout.splitlines() if "anchored on" in ln), "?")
        made += 1
        # A MANIFEST BESIDE THE MASK, so a mask on disk states the definition that produced it
        # rather than relying on this script having been run with today's config.
        try:
            (out / "rest_manifest.json").write_text(json.dumps({
                "variant": variant,
                "params": config.defaults()["segmentation"]["rest"],
                "anchor": anchor,
                "label": s["label"],
                "source_h5": str(s["h5"]),
            }, indent=1), encoding="utf-8")
        except Exception as ex:                                        # noqa: BLE001
            print(f"     (manifest not written: {type(ex).__name__})", flush=True)
        print(f"  [{i}/{len(todo)}] {s['label']}  anchor={anchor}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n[rest] built {made}, skipped {skipped}, failed {len(failed)} "
          f"in {time.time() - t0:.0f}s", flush=True)
    for lab, why in failed:
        print(f"   !! {lab}: {why[:160]}", flush=True)
    # FAILURES ARE NOT FATAL TO THE RUN but they ARE reported and counted, because a session without
    # a rest mask silently loses its rest column downstream -- `quiet_frame_path` does not fall back
    # across definitions, by design.
    return 0


if __name__ == "__main__":
    sys.exit(main())
