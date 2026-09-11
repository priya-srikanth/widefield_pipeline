"""Regenerate EVERY per-session behaviour figure, for all registered dates.

    PYTHONPATH=$(pwd) python scripts/rerun_behaviour_figures.py [--dates 20260805 ...] [--dry-run]

WHY THIS EXISTS. `spout_behavior` takes ONE date, because the nightly only ever has one. A change to
a per-session PANEL -- the engagement timeline gaining its WORKING / STOPPED annotation on
2026-09-10, the engagement gate changing before that -- is retroactive by nature: every session's
figure is now drawn by different code from the one beside it, and nothing in the pipeline notices,
because a figure that exists is not re-made. Looping by hand is how that was done twice; this is the
loop, kept, so the third time is a command rather than a paste.

IT WRITES TO THE SHARE (`behavior_out`), one date at a time, and it is SAFE TO INTERRUPT: each date
is independent and a re-run simply overwrites. It is NOT safe to run while the behaviour nightly is
running for one of the same dates — they would race on the same PNG paths.

Failures are reported per date and do not stop the sweep: an aborted session that cannot be scored
should not prevent the other fifty-one dates from being refreshed.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

from wfield_local import spout_behavior as sb
from wfield_local.paths import PathResolver


def all_dates(rv) -> list[str]:
    out = set()
    for p in sb.discover_sessions(rv, None, None):
        mo = re.search(r"_(\d{8})_", p.name)
        if mo:
            out.add(mo.group(1))
    return sorted(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=None, help="default: every registered date")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rv = PathResolver()
    dates = args.dates or all_dates(rv)
    print(f"{len(dates)} date(s) to refresh", flush=True)
    t0, failed = time.time(), []
    for i, d in enumerate(dates, 1):
        lead = f"[{i}/{len(dates)}] {d}"
        if args.dry_run:
            print(f"{lead}  (dry run)", flush=True)
            continue
        t1 = time.time()
        try:
            sb.main([d])
            print(f"{lead}  ok  {time.time() - t1:.0f}s", flush=True)
        except SystemExit as ex:                                       # argparse/main exit codes
            if ex.code not in (0, None):
                failed.append((d, f"exit {ex.code}"))
                print(f"{lead}  !! exit {ex.code}", flush=True)
        except Exception as ex:                                        # noqa: BLE001
            failed.append((d, f"{type(ex).__name__} {str(ex)[:90]}"))
            print(f"{lead}  !! {type(ex).__name__} {str(ex)[:90]}", flush=True)
    mins = (time.time() - t0) / 60.0
    print(f"\ndone in {mins:.1f} min; {len(dates) - len(failed)}/{len(dates)} ok", flush=True)
    for d, why in failed:
        print(f"  FAILED {d}: {why}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
