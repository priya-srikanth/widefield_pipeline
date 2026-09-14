"""Which sessions LOSE their RESTW column under the settled rest definition, and why -- by EPOCH.

THIS IS A RESULT, NOT BOOKKEEPING. `restw` needs at least `MIN_POSITIONS_FOR_WEIGHTED` positions each
holding at least `MIN_FRAMES_PER_POSITION` rest frames, and the docked + engagement terms cut hardest
exactly where the animal worked least. So the sessions that drop out are not a random sample: they
are expected to concentrate post-stroke, and a pooled figure that quietly loses its worst sessions
would report a deficit measured on the animals least affected by it.

Reported per epoch: how many sessions form a RESTW column, how many do not, and the reason -- too few
POSITIONS clearing the frame floor, or no rest mask at all. Also the per-position frame counts, since
"four positions cleared the floor" and "four positions cleared it by two frames" are different
situations and only the second is fragile.

    python -m scripts.rest_migration.restw_column_drops [--limit N]
"""
from __future__ import annotations

import argparse

import numpy as np

from wfield_local import config, epochs
from wfield_local.rest_by_position import (
    MIN_FRAMES_PER_POSITION, MIN_POSITIONS_FOR_WEIGHTED, rest_frames_by_position)

EPOCH_ORDER = ("pre", "acute", "subacute", "chronic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    # THE ANALYSED SET ONLY -- the same sessions `recompute_masks` builds for. Sessions outside it
    # (the excluded early-June ones) have no restdock mask because none was ever built, and counting
    # those as DROPS would make the drop rate meaningless: a session that was never in the analysis
    # is not a session the definition cost us.
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    rows = []
    sessions = [x for x in config.load_sessions() if x["label"] in want]
    print(f"[restw] analysed set: {len(sessions)} sessions", flush=True)
    if a.limit:
        sessions = sessions[: a.limit]
    for s in sessions:
        lab = s["label"]
        try:
            ep = epochs.epoch_of(lab)
        except Exception:                                              # noqa: BLE001
            ep = None
        # n_frames is only used to clip indices; the mask's own length bounds it, so a generous
        # value is safe and avoids loading SVTcorr just to count.
        try:
            per, info = rest_frames_by_position(s, 10**9, docked=False)
        except Exception as ex:                                        # noqa: BLE001
            rows.append((lab, ep, None, [], f"{type(ex).__name__}: {str(ex)[:60]}"))
            continue
        if info.get("error"):
            rows.append((lab, ep, None, [], info["error"]))
            continue
        counts = sorted(((int(c), int(np.asarray(v).size)) for c, v in per.items()),
                        key=lambda t: t[0])
        clearing = [c for c, n in counts if n >= MIN_FRAMES_PER_POSITION]
        ok = len(clearing) >= MIN_POSITIONS_FOR_WEIGHTED
        why = "" if ok else f"only {len(clearing)}/6 positions clear {MIN_FRAMES_PER_POSITION} frames"
        rows.append((lab, ep, ok, counts, why))
        print(f"  {lab:12s} {str(ep):9s} {'RESTW' if ok else 'DROP ':5s} "
              f"{len(clearing)}/6 clear  frames=" + ",".join(f"{c}:{n}" for c, n in counts),
              flush=True)

    print("\nPER-EPOCH RESTW COLUMN AVAILABILITY")
    print(f"{'epoch':10s} {'n':>4s} {'restw':>6s} {'drop':>5s} {'drop %':>7s}   "
          f"{'median positions clearing':>26s}")
    for ep in EPOCH_ORDER + (None,):
        sub = [r for r in rows if r[1] == ep]
        if not sub:
            continue
        good = [r for r in sub if r[2] is True]
        bad = [r for r in sub if r[2] is not True]
        med = np.median([sum(1 for _c, n in r[3] if n >= MIN_FRAMES_PER_POSITION) for r in sub])
        print(f"{str(ep):10s} {len(sub):4d} {len(good):6d} {len(bad):5d} "
              f"{100*len(bad)/len(sub):6.1f}%   {med:26.1f}")

    drops = [r for r in rows if r[2] is not True]
    if drops:
        print(f"\nSESSIONS WITHOUT A RESTW COLUMN ({len(drops)}):")
        for lab, ep, _ok, counts, why in drops:
            print(f"  {lab:12s} {str(ep):9s} {why}")
    else:
        print("\nNo session loses its RESTW column.")

    # FRAGILITY: a position clearing the floor by a hair is not the same as clearing it comfortably.
    thin = []
    for lab, ep, ok, counts, _why in rows:
        if ok is not True:
            continue
        m = min((n for _c, n in counts if n >= MIN_FRAMES_PER_POSITION), default=0)
        if m < 2 * MIN_FRAMES_PER_POSITION:
            thin.append((lab, ep, m))
    print(f"\nTHIN BUT PASSING (min contributing position under {2*MIN_FRAMES_PER_POSITION} "
          f"frames): {len(thin)}")
    for lab, ep, m in sorted(thin, key=lambda t: t[2])[:12]:
        print(f"  {lab:12s} {str(ep):9s} thinnest contributing position {m} frames")


if __name__ == "__main__":
    main()
