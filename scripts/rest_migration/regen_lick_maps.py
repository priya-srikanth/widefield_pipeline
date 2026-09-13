"""Rebuild the lick-aligned REST-normalised maps, so the preprocessing deck stops mixing baselines.

THE PROBLEM THIS FIXES. After `regen_running_maps` the preprocessing deck held 26 `quiet_running`
slides per animal on the NEW trial-anchored baseline sitting beside 26 `lick_quietnorm` slides that
were 512 hours old and built on the RETIRED reward-anchored one -- two definitions of the same word
inside one deck, with nothing on the slides saying so. That is precisely the failure the migration
exists to remove, so leaving it would have been worse than not rebuilding the deck at all.

`framemap_event_maps --what lick --quiet-frame <mask>` produces those maps. It is normally an
imaging-box `preprocess` step; it is driven here for the same reason and with the same caveat as
`regen_running_maps` -- output lands in the imaging box's `lick_aligned_<tag>/` and that box will
regenerate it on its next nightly, from the same masks, to the same answer.

ARGUMENTS MIRROR `preprocess.generate_maps` EXACTLY, including `--behavior-trials` (which
`classify_cues_with_backup` needs for the sessions whose DAQ strobe was degraded) and the tagged
`<animal>_<mmdd>_<tag>` label, so a map built here is indistinguishable from the imaging box's.
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
    args = ap.parse_args(argv)

    from wfield_local import config
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.quiet_periods import quiet_frame_path

    mp = config.defaults()["preprocess"]["maps"]
    tag, lick_post_s = mp["tag"], str(mp["lick_post_s"])
    rv = config.resolver()
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [s for s in SESSIONS if s["label"] in want and s.get("h5")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[lick] {len(todo)} sessions, tag {tag}, post {lick_post_s}s", flush=True)

    t0, made, failed = time.time(), 0, []
    for i, s in enumerate(todo, 1):
        animal, mmdd = s["label"].split("_")
        lab, mc = f"{animal}_{mmdd}_{tag}", s["mc"]
        # THE MASK COMES FROM THE RESOLVER, not a constructed path: it must be the variant the
        # config selects, and it must exist. A missing mask costs this session rather than
        # silently producing an un-normalised map.
        qf = quiet_frame_path(mc)
        if not qf:
            failed.append((s["label"], "no rest mask for the configured variant"))
            continue
        fm = sorted(glob.glob(f"{s.get('fmdir') or mc}/*cleanpairs_frame_map.npz"))
        if not fm:
            failed.append((s["label"], "no cleanpairs frame map"))
            continue
        bt = []
        hits = sorted(glob.glob(f"{rv.root('behavior_logs')}/{animal}_2026{mmdd}_*/trials.csv"))
        if s.get("behavior_trials"):
            bt = ["--behavior-trials", str(s["behavior_trials"])]
        elif hits:
            bt = ["--behavior-trials", hits[0]]
        cmd = [sys.executable, "-m", "wfield_local.framemap_event_maps",
               "--what", "lick", "--daq-h5", str(s["h5"]),
               "--wfield-results", f"{mc}/wfield_local_results",
               "--allen-dir", f"{mc}/wfield_local_results/allen_aligned_{tag}",
               "--frame-map", fm[0],
               "--cleanpairs-summary", fm[0].replace("_frame_map.npz", "_summary.json"),
               "--output", f"{mc}/lick_aligned_{tag}", "--label", lab,
               "--post-s", lick_post_s, "--quiet-frame", str(qf)] + bt
        if args.dry_run:
            print(f"  [{i}/{len(todo)}] DRY {lab}", flush=True)
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            failed.append((s["label"], (r.stderr or r.stdout or "").strip()[-200:]))
            print(f"  [{i}/{len(todo)}] !! {s['label']}: {failed[-1][1][:140]}", flush=True)
            continue
        made += 1
        print(f"  [{i}/{len(todo)}] {lab} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n[lick] built {made}, failed {len(failed)} in {time.time() - t0:.0f}s", flush=True)
    for lab, why in failed:
        print(f"   !! {lab}: {why[:160]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
