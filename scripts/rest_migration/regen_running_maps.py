"""Rebuild the rest/running SVD activity maps for the preprocessing deck, on the REST baseline.

NORMALLY AN IMAGING-BOX STEP. `preprocess.generate_maps` runs this; it is driven from the analysis
box here because every input is on MICROSCOPE and the rest masks were rebuilt there (Priya,
2026-09-13: "the imaging box transfers everything to standby. can you rebuild the decks?" then
"do it here anyway").

TWO CONSEQUENCES, stated because neither is obvious:
  * Output lands in `{mc}/running_activity_<tag>/`, which is INSIDE the imaging box's
    `motion_corrected/` tree. `writeguard` permits it (Priya's subtree) but that box owns the
    directory, and its next nightly will regenerate these files. That is fine -- it would produce
    the same maps from the same masks -- but it means this is a convenience, not a handover.
  * `plot_running_activity_maps` REFUSES a pre-v3 events npz, so `regen_events` must have run.
    Before that guard existed it would have silently built the rest panel on the retired
    reward-anchored definition.

ARGUMENTS ARE CONSTRUCTED EXACTLY AS `preprocess.py` CONSTRUCTS THEM, including the `<animal>_<mmdd>
_<tag>` label, so a map built here is indistinguishable from one the imaging box would build.
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="rebuild sessions that already have maps")
    args = ap.parse_args(argv)

    from wfield_local import config
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    tag = config.defaults()["preprocess"]["maps"]["tag"]
    rv = config.resolver()
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [s for s in SESSIONS if s["label"] in want and s.get("h5")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[running] {len(todo)} sessions, tag {tag}", flush=True)

    t0, made, skipped, failed = time.time(), 0, 0, []
    for i, s in enumerate(todo, 1):
        animal, mmdd = s["label"].split("_")
        lab = f"{animal}_{mmdd}_{tag}"
        mc = s["mc"]
        out = Path(f"{mc}/running_activity_{tag}")
        if not args.force and sorted(out.glob(f"{lab}_*.png")):
            skipped += 1
            continue
        res = f"{mc}/wfield_local_results"
        allen = f"{res}/allen_aligned_{tag}"
        events = f"{rv.root('behavior_out')}/events/{animal}/2026{mmdd}.npz"
        fm = sorted(glob.glob(f"{s.get('fmdir') or mc}/*cleanpairs_frame_map.npz"))
        if not fm:
            failed.append((s["label"], "no cleanpairs frame map"))
            continue
        cmd = [sys.executable, "-m", "wfield_local.plot_running_activity_maps",
               "--label", lab, "--events", events, "--wfield-results", res,
               "--allen-dir", allen, "--daq-h5", str(s["h5"]),
               "--frame-map", fm[0],
               "--cleanpairs-summary", fm[0].replace("_frame_map.npz", "_summary.json"),
               "--output", str(out)]
        if args.dry_run:
            print(f"  [{i}/{len(todo)}] DRY {lab}", flush=True)
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            failed.append((s["label"], (r.stderr or r.stdout or "").strip()[-200:]))
            print(f"  [{i}/{len(todo)}] !! {s['label']}: {failed[-1][1][:140]}", flush=True)
            continue
        made += 1
        note = "SKIPPED" if "SKIPPED" in r.stdout else "ok"
        print(f"  [{i}/{len(todo)}] {lab} {note} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n[running] built {made}, skipped {skipped}, failed {len(failed)} "
          f"in {time.time() - t0:.0f}s", flush=True)
    for lab, why in failed:
        print(f"   !! {lab}: {why[:160]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
