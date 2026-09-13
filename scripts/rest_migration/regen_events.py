"""Rebuild every session's canonical behavior-events npz on the REST definition (schema v3).

WHY IT HAS TO BE EXPLICIT. The v3 bump makes `behavior_events.get_or_compute` treat older files as
stale and recompute them on access, so consumers that go through that entry point heal themselves.
But `plot_running_activity_maps` -- the preprocessing deck's rest/running SVD maps -- loads the npz
DIRECTLY with `load_events`, and `nightly_figs` never calls `behavior_events` at all. Nothing else
in the analysis pipeline regenerates these, so they are rebuilt here, once, deliberately.

RECOMPUTING OVERWRITES; there is no variant mechanism for these files, unlike the imaging masks.
That is acceptable because they are derived and reproducible from an archived DAQ `.h5`, and each
npz stores the `params` block that produced it, so a file always states its own definition.
"""
from __future__ import annotations

import sys
import time


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from wfield_local import behavior_events as be
    from wfield_local import config

    rv = config.resolver()
    dates = sorted({x["label"].split("_")[-1] for x in config.load_sessions()})
    print(f"[events] {len(dates)} dates -> schema v{be.SCHEMA_VERSION}", flush=True)
    if args.dry_run:
        print("  " + " ".join(dates))
        return 0

    t0, ok, failed = time.time(), 0, []
    for i, mmdd in enumerate(dates, 1):
        try:
            be.run(f"2026{mmdd}", rv, force=True)
            ok += 1
        except Exception as ex:                                        # noqa: BLE001
            failed.append((mmdd, f"{type(ex).__name__}: {str(ex)[:120]}"))
            print(f"  !! {mmdd}: {failed[-1][1]}", flush=True)
        print(f"  [{i}/{len(dates)}] {mmdd} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n[events] {ok} dates rebuilt, {len(failed)} failed in {time.time() - t0:.0f}s",
          flush=True)
    for mmdd, why in failed:
        print(f"   !! {mmdd}: {why}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
