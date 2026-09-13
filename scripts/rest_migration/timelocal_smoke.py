"""Does the TIME-LOCAL rest reference actually differ from the flat one, and by how much?

WHY THIS RUNS BEFORE THE RENDER. Wiring the time-local baseline changes every REST-referenced map
in the deck, and a 92-session render is the wrong place to discover either failure mode:

    NO EFFECT      the two references agree to numerical noise, which would mean the drift the
                   change exists to remove is not in these maps -- and the render was wasted.
    TOO MUCH       the maps change by more than their own signal, which would mean the baseline is
                   eating the response rather than the drift (the failure a 12-bin baseline over a
                   ~2 s window could plausibly have).

The useful comparison is against the SIGNAL: report the RMS difference between the two references
as a fraction of the flat-referenced map's own RMS. A few percent is drift removal; approaching 1.0
is the baseline absorbing the response.

ALSO CHECKS THE ONE THING A NUMBER CANNOT: that the positions still differ from each other. If the
time-local baseline were subtracting a per-trial mean it would flatten the between-position
structure, and the between/within ratio is what would show it.

RUN:  python -m scripts.rest_migration.timelocal_smoke [--n 3]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="sessions to test")
    ap.add_argument("--align", default="cue")
    a = ap.parse_args()

    from wfield_local import beta_maps as bm
    from wfield_local import config
    from wfield_local import position_reference_maps as prm
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    mask = bm.stat_mask()
    want = list(config.phase_labels("pre"))
    todo = [s for s in SESSIONS if s["label"] in want][: a.n]
    t0 = time.time()
    ok = 0

    for s in todo:
        lab = s["label"]
        try:
            parts = prm.session_raw_maps(s, a.align)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        raw, flat, tl = parts["raw"], parts["quiet_flat"], parts["raw_rest"]
        if not raw:
            print(f"  .. {lab}: no positions"); continue
        if flat is None:
            print(f"  .. {lab}: no flat baseline to compare against"); continue
        if not tl:
            print(f"  !! {lab}: TIME-LOCAL REST MAPS ARE EMPTY -- the wiring did not fire")
            continue

        # The retired construction, rebuilt here so the two are compared on identical inputs.
        old = {q: m - flat for q, m in raw.items()}
        shared = [q for q in tl if q in old]
        d = [float(np.sqrt(np.mean((tl[q] - old[q])[mask] ** 2))) for q in shared]
        r = [float(np.sqrt(np.mean(old[q][mask] ** 2))) for q in shared]
        # between-position structure under the NEW reference
        ms = [tl[q] for q in shared]
        within = float(np.mean([np.sqrt(np.mean(m[mask] ** 2)) for m in ms]))
        gm = np.mean(ms, axis=0)
        between = float(np.mean([np.sqrt(np.mean((m - gm)[mask] ** 2)) for m in ms]))
        print(f"  .. {lab}: {len(shared)} positions   "
              f"|timelocal - flat| RMS {np.mean(d):.5f}  "
              f"= {np.mean(d) / max(1e-12, np.mean(r)):.1%} of the map's own RMS   "
              f"between/within {between / max(1e-12, within):.2f}   ({time.time() - t0:.0f}s)",
              flush=True)
        ok += 1

    print(f"\n{ok}/{len(todo)} sessions produced both references")
    if not ok:
        print("NOTHING WAS COMPARED -- this is a failed run, not a negative result.")
        return 1
    print("READ IT AS: a few % = drift removed; near 0% = the change does nothing; approaching "
          "100% = the baseline is eating the response. between/within should stay well above 0.")
    print(f"[done in {time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
