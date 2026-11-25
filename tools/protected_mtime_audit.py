"""POST-RUN MTIME AUDIT (CLAUDE.md rule 11: "After every chain, check the protected folders for files modified inside
the run window, and report any hit immediately").

Lists every file under the protected roots of `configs/protected_roots.yaml` (share globs expanded on this machine's
MICROSCOPE mount, plus the local frozen store and a WIDEFIELD_FROZEN_MODEL_DIR store) whose mtime is at or after
``--since`` (and before ``--until`` if given). Stat only, no hashing; ``--depth`` bounds the walk below each root
(default 6), ``--kinds`` narrows to some root kinds. Read-only: writes nothing unless ``--out`` names a report file.

Usage:
    python tools/protected_mtime_audit.py --since "2026-11-24 14:00" [--until "2026-11-24 18:00"] [--depth 4]
        [--kinds production patch frozen_store] [--exclude "*rebuild_20261123_fourcam*"] [--share-root M:/MICROSCOPE/Priya]
        [--out <new report.tsv>]
``--exclude`` (fnmatch on the full path, repeatable) drops the declared outputs of runs that legitimately write inside
a protected root (an active rebuild root's own run). A per-root count is printed first.
Exit code: 0 = no hit, 1 = hits (so a chain wrapper can stop on it), 2 = bad arguments.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_time(s: str) -> float:
    """'YYYY-mm-dd HH:MM[:SS]', 'HH:MM' (today) or a Unix timestamp -> epoch seconds (local time)."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(s, fmt))
        except ValueError:
            continue
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = dt.datetime.strptime(s, fmt).time()
            return dt.datetime.combine(dt.date.today(), t).timestamp()
        except ValueError:
            continue
    raise ValueError(f"cannot parse time {s!r}")


def scan(root: Path, t0: float, t1: float | None, depth: int):
    """Yield (mtime, path) of files under ``root`` modified in [t0, t1); stat only, ``depth`` levels below root."""
    if root.is_file():
        try:
            m = root.stat().st_mtime
        except OSError:
            return
        if m >= t0 and (t1 is None or m < t1):
            yield m, root
        return
    stack = [(str(root), 0)]
    while stack:
        d, lev = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if lev < depth:
                            stack.append((e.path, lev + 1))
                        continue
                    m = e.stat(follow_symlinks=False).st_mtime
                except OSError:
                    continue
                if m >= t0 and (t1 is None or m < t1):
                    yield m, Path(e.path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", default=None)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--kinds", nargs="+", default=None)
    ap.add_argument("--share-root", default=None, help="MICROSCOPE Priya root (default: the machine profile's)")
    ap.add_argument("--exclude", nargs="+", default=[], help="fnmatch patterns (full path) to leave out")
    ap.add_argument("--registry", default=None, help="registry yaml (default configs/protected_roots.yaml)")
    ap.add_argument("--out", default=None, help="also write the hits as TSV to this NEW file")
    a = ap.parse_args(argv)
    try:
        t0 = parse_time(a.since)
        t1 = parse_time(a.until) if a.until else None
    except ValueError as ex:
        print(ex, file=sys.stderr)
        return 2
    if a.registry:
        os.environ["WFIELD_PROTECTED_ROOTS_FILE"] = a.registry
    from wfield_local import protected_roots as pr
    roots = pr.roots_on_this_machine(a.share_root)
    if a.kinds:
        roots = [(k, r) for k, r in roots if k in a.kinds]
    t_start = time.time()
    hits = []
    import fnmatch
    excl = [x.replace("\\", "/").lower() for x in a.exclude]
    for kind, root in roots:
        for m, p in scan(root, t0, t1, a.depth):
            ps = str(p).replace("\\", "/").lower()
            if any(fnmatch.fnmatchcase(ps, x) for x in excl):
                continue
            hits.append((m, kind, root, p))
    hits.sort()
    win = f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(t0))} .. " + \
          (time.strftime('%Y-%m-%d %H:%M', time.localtime(t1)) if t1 else "now")
    print(f"[protected-audit] {len(roots)} roots, window {win}, depth {a.depth}: {len(hits)} file(s) modified "
          f"({time.time() - t_start:.1f} s)")
    from collections import Counter
    for (kind, root), n in sorted(Counter((k, str(r)) for _m, k, r, _p in hits).items()):
        print(f"  {n:6d}  {kind:<18} {root}")
    for m, kind, _root, p in hits:
        print(f"{time.strftime('%m-%d %H:%M:%S', time.localtime(m))}\t{kind}\t{p}")
    if a.out:
        from wfield_local import protected_roots as _pr
        _pr.check(a.out, "write the audit report to")
        with open(a.out, "x", encoding="utf-8") as fh:
            fh.write("mtime\tkind\troot\tpath\n")
            for m, kind, root, p in hits:
                fh.write(f"{dt.datetime.fromtimestamp(m).isoformat(timespec='seconds')}\t{kind}\t{root}\t{p}\n")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
