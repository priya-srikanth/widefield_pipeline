"""SPLICE a few sessions' rows into whole-cohort tables (lane E 11-24, Priya: "patch route" for PS95_0818 / 0820 after the
11-23 cam3 re-add). Nothing cohort-wide is fit on post-stroke sessions (frozen decoders / frame and z-scoring are pre-only
or within session), so a post session's rows can be rebuilt alone and swapped in.

For every table: OLD = the first existing copy among --old roots (e.g. the neural-tail run, then production); NEW = the
patch build's copy (built with WFIELD_ITI_ONLY_LABELS). The output (--out/<rel>) is OLD at the TEXT level with every row
of the --labels sessions replaced by NEW's rows of those sessions (in NEW's order, at the position of OLD's first such
row): every other line is copied verbatim, so the other sessions' rows are byte-identical (asserted). The headers must
be identical. Per-session folders (--dirs, e.g. cued_wide): OLD's files copied, the --labels sessions' files from NEW.
A manifest (--out/PATCH_MANIFEST.json, appended) records each table / folder, its sources, rows removed / added.

Usage: python scripts/patch_splice.py --out P --new F --old NT --old PROD --labels PS95_0818,PS95_0820
       [--tables rel ...] [--dirs rel ...]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import time
from pathlib import Path


def _read_lines(p: Path) -> list[str]:
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8", newline="") as f:
        return f.read().splitlines(keepends=True)


def _write_lines(p: Path, lines: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "wt", encoding="utf-8", newline="") as f:
        f.writelines(lines)


def _label_of(line: str, idx: int) -> str:
    return next(csv.reader(io.StringIO(line)))[idx]


def splice_lines(old: list[str], new: list[str], labels: set[str]) -> tuple[list[str], int, int]:
    """OLD lines with the ``labels`` rows replaced by NEW's rows of those labels (text level). Returns (lines,
    removed, added). Raises on a header mismatch or when NEW has none of the labels."""
    if old[0].rstrip("\r\n") != new[0].rstrip("\r\n"):
        raise ValueError(f"header differs:\n old {old[0][:200]}\n new {new[0][:200]}")
    idx = next(csv.reader(io.StringIO(old[0]))).index("label")
    add = [ln for ln in new[1:] if _label_of(ln, idx) in labels]
    if not add:
        raise ValueError(f"the new table has no rows of {sorted(labels)}")
    out, removed, placed = [old[0]], 0, False
    for ln in old[1:]:
        if _label_of(ln, idx) in labels:
            removed += 1
            if not placed:
                out.extend(add)
                placed = True
            continue
        out.append(ln)
    if not placed:
        out.extend(add)
    others_old = [ln for ln in old[1:] if _label_of(ln, idx) not in labels]
    others_new = [ln for ln in out[1:] if _label_of(ln, idx) not in labels]
    assert others_old == others_new, "other sessions' rows are not byte-identical"
    return out, removed, len(add)


def _sha(lines: list[str]) -> str:
    return hashlib.sha256("".join(lines).encode()).hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", action="append", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tables", nargs="*", default=[])
    ap.add_argument("--dirs", nargs="*", default=[])
    a = ap.parse_args(argv)
    out, new = Path(a.out), Path(a.new)
    labels = {x.strip() for x in a.labels.split(",") if x.strip()}
    olds = [Path(x) for x in a.old]
    man = []
    for rel in a.tables:
        src = next((o / rel for o in olds if (o / rel).exists()), None)
        if src is None:
            raise FileNotFoundError(f"{rel}: in none of {olds}")
        old_l, new_l = _read_lines(src), _read_lines(new / rel)
        lines, rem, add = splice_lines(old_l, new_l, labels)
        _write_lines(out / rel, lines)
        idx = next(csv.reader(io.StringIO(old_l[0]))).index("label")
        man.append({"table": rel, "old": str(src), "new": str(new / rel), "rows_removed": rem, "rows_added": add,
                    "other_rows_sha_old": _sha([ln for ln in old_l[1:] if _label_of(ln, idx) not in labels]),
                    "other_rows_sha_out": _sha([ln for ln in lines[1:] if _label_of(ln, idx) not in labels])})
        print(f"{rel}: {src.parent.name} -> {rem} rows of {sorted(labels)} replaced by {add} (others byte-identical)",
              flush=True)
    for rel in a.dirs:
        srcd = next((o / rel for o in olds if (o / rel).is_dir()), None)
        if srcd is None:
            raise FileNotFoundError(f"{rel}/: in none of {olds}")
        dst = out / rel
        dst.mkdir(parents=True, exist_ok=True)
        n_old = 0
        for f in sorted(srcd.iterdir()):
            if f.is_file() and f.stem not in labels:
                shutil.copy2(f, dst / f.name)
                n_old += 1
        put = []
        for x in sorted(labels):
            hit = sorted((new / rel).glob(f"{x}.*"))
            if not hit:
                raise FileNotFoundError(f"{new / rel}: no file of {x}")
            for f in hit:
                shutil.copy2(f, dst / f.name)
                put.append(f.name)
        man.append({"dir": rel, "old": str(srcd), "new": str(new / rel), "files_copied": n_old, "files_patched": put})
        print(f"{rel}/: {n_old} files from {srcd.parent.name}, patched {put}", flush=True)
    mf = out / "PATCH_MANIFEST.json"
    prev = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else []
    prev.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "labels": sorted(labels), "entries": man})
    mf.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
