"""Publish this box's CURRENT-VERSION session cache to MICROSCOPE, so the nightly box inherits it.

WHY. Big pipeline changes happen on the dev box; the NIGHTLY analysis usually runs on the other one
(Priya, 2026-09-15). A `CACHE_VERSION` bump is therefore a TWO-MACHINE change made on one machine:
bumping 11 -> 12 for `restdock05` invalidated the nightly box's cache too, and its next run would
face the same cold rebuild this box just spent 16 h on. Copying the new entries over turns that into
a download.

WHICH ENTRIES. `CACHE_VERSION` is folded into the key HASH, not the filename, so v11 and v12 entries
are indistinguishable by name -- only by WHEN THEY WERE WRITTEN. This copies entries modified at or
after `--since` (default: start of today), which is exactly the set the current-version run produced.
That is a proxy, and a wrong `--since` silently ships stale entries, so the cutoff is PRINTED and the
count is reported against the live cache total.

NOT A DELETE, ANYWHERE. It only ever creates files under `MICROSCOPE/Priya/...` (rule 1, enforced by
`writeguard.assert_writable`). Orphaned older-version entries are left alone on both sides; they are
never read once the version moves, and pruning them is a separate decision.

RE-RUN IT AFTER A RENDER FINISHES. Entries written while a render is still going will be missed by a
copy taken mid-run; a second pass with the same `--since` picks up the remainder (existing files are
skipped unless --force).

    python -m scripts.rest_migration.publish_cache [--dest ...] [--since YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

DEFAULT_DEST = "N:/MICROSCOPE/Priya/Widefield/session_cache_v12"


def main():
    from wfield_local import session_cache as sc
    from wfield_local import writeguard

    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--since", default=None,
                    help="YYYY-MM-DD; default = start of today (local)")
    ap.add_argument("--force", action="store_true", help="recopy files already at the destination")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = Path(sc.CACHE_DIR)
    dest = Path(a.dest)
    if a.since:
        cutoff = time.mktime(time.strptime(a.since, "%Y-%m-%d"))
    else:
        n = time.localtime()
        cutoff = time.mktime((n.tm_year, n.tm_mon, n.tm_mday, 0, 0, 0, 0, 0, -1))
    print(f"source      {src}")
    print(f"destination {dest}")
    print(f"cutoff      {time.strftime('%Y-%m-%d %H:%M', time.localtime(cutoff))}  "
          f"(CACHE_VERSION {sc.CACHE_VERSION})")

    allf = [p for p in src.rglob("*") if p.is_file()]
    new = [p for p in allf if p.stat().st_mtime >= cutoff]
    tot = sum(p.stat().st_size for p in new)
    print(f"cache holds {len(allf)} entries; {len(new)} at/after the cutoff "
          f"({tot / 1e9:.2f} GB) -- the rest are older-version orphans, left alone")
    if not new:
        print("nothing to publish")
        return 0

    writeguard.assert_writable(dest)
    if not a.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    t0 = time.time()
    for i, p in enumerate(new, 1):
        rel = p.relative_to(src)
        out = dest / rel
        if out.exists() and not a.force and out.stat().st_size == p.stat().st_size:
            skipped += 1
            continue
        if a.dry_run:
            done += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        # TEMP FILE + REPLACE, never a direct write: a reader on the other box must never see a
        # half-copied pickle. A torn entry does not fail loudly at copy time -- it fails later,
        # inside whichever figure happens to need that session.
        tmp = out.with_suffix(out.suffix + ".part")
        shutil.copy2(p, tmp)
        os.replace(tmp, out)
        done += 1
        if done % 100 == 0:
            print(f"  [{i}/{len(new)}] {done} copied ({time.time() - t0:.0f}s)", flush=True)
    print(f"{'DRY ' if a.dry_run else ''}published {done}, skipped {skipped} already present, "
          f"in {time.time() - t0:.0f}s")
    print(f"\nOn the nightly box:  set WIDEFIELD_SESSION_CACHE={dest}")
    print("NB entries are PICKLES: the reading box must be on a compatible numpy "
          "(this repo runs numpy<2.1 on the imaging box and 2.2.6 here -- rule 6).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
