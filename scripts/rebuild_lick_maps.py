"""Re-run ONLY the lick-dependent map steps for every registered session, with the ITI gate.

Priya, 2026-08-22: "rebuild the preprocessing decks with this change".

WHY NOT `python -m wfield_local.preprocess ... --skip-*`. That re-runs the CUE maps too, which this
change does not touch, at roughly the same cost per session as the lick ones -- about double the
wall clock to rewrite files with identical content. This takes `_maps_commands`' own command list,
so the arguments are built by the pipeline rather than by hand, and keeps only the steps that read
or produce the lick maps.

TWO THINGS TO KNOW BEFORE RUNNING THIS:
  * It writes into `motion_corrected/lick_aligned_<tag>/` on MICROSCOPE, which CLAUDE.md rule 1
    designates a READ-ONLY input for the analysis box (it is the imaging box's output tree). The
    files are DERIVED and regenerable, and the write stays inside MICROSCOPE/Priya, but this is the
    analysis box doing the imaging box's job.
  * The gate lives on this branch. Until it is on main and the imaging box has pulled, that box's
    next `preprocess` run will regenerate these maps WITHOUT it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from wfield_local import config
from wfield_local.paths import PathResolver
from wfield_local.preprocess import _maps_commands, discover_processed_sessions

#: the steps whose inputs or outputs are the lick maps. The cue map and the running-activity map are
#: untouched by the gate, so re-running them would only cost time.
LICK_STEPS = ("framemap_event_maps", "plot_lick_position_contrasts", "plot_lick_vs_cue_spout_maps")


def wanted(cmd: list[str]) -> bool:
    mod = next((c for c in cmd if c.startswith("wfield_local.")), "")
    name = mod.split(".")[-1]
    if name == "framemap_event_maps":
        return "lick" in cmd[cmd.index("--what") + 1] if "--what" in cmd else False
    return name in LICK_STEPS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=None, help="default: every registered date")
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rv = PathResolver()
    params = config.defaults()["preprocess"]
    dates = args.dates or sorted({s["label"].split("_")[1] for s in config.load_sessions()})
    dates = [d if len(d) == 8 else f"2026{d}" for d in dates]
    only = set(config.normalize_animals(args.animals) or [])

    todo = []
    for d in sorted(set(dates)):
        try:
            for s in discover_processed_sessions(d, rv):
                if only and s.get("animal") not in only:
                    continue
                todo.append((d, s))
        except Exception as ex:                                   # noqa: BLE001
            print(f"[rebuild] {d}: discovery failed ({type(ex).__name__} {str(ex)[:70]})", flush=True)

    print(f"[rebuild] {len(todo)} session(s) across {len(set(dates))} date(s)", flush=True)
    t0 = time.time()
    failed = []
    for i, (d, s) in enumerate(todo, 1):
        try:
            cmds = [c for c in _maps_commands(s, params, rv, allow_missing=False) if wanted(c)]
        except Exception as ex:                                   # noqa: BLE001
            print(f"[rebuild] {s.get('label', d)}: cannot build commands "
                  f"({type(ex).__name__} {str(ex)[:70]})", flush=True)
            failed.append(s.get("label", d))
            continue
        print(f"[rebuild] ({i}/{len(todo)}) {s.get('label', d)}: {len(cmds)} step(s)", flush=True)
        for c in cmds:
            if args.dry_run:
                print("   $ " + " ".join(str(x) for x in c[:3]) + " ...", flush=True)
                continue
            if subprocess.call([sys.executable, "-u", "-m", *[str(x) for x in c]],
                               cwd=str(Path(__file__).resolve().parent)):
                print(f"   !! failed: {c[0]}", flush=True)
                failed.append(f"{s.get('label', d)}:{c[0]}")
    dt = (time.time() - t0) / 60
    print(f"[rebuild] done in {dt:.1f} min; {len(failed)} failure(s)", flush=True)
    for f in failed:
        print("   failed:", f, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
