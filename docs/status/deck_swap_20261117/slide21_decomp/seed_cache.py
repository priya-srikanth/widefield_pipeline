"""COPY (never move) the rebuild session-cache entries a variant may reuse into its own cache dir.

Usage: seed_cache.py <dest dir> <kind regex> [<kind regex> ...]
Reads the scan made by scan_cache.py (sc_kinds_files.csv); copies label__kind__key.pkl when the kind matches; skips a
file already present with the same size. Source = the rebuild session cache (read only)."""
import csv
import os
import re
import shutil
import sys
import time

SRC = "M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/session_cache"
HERE = os.path.dirname(os.path.abspath(__file__))
dest, pats = sys.argv[1], [re.compile(p) for p in sys.argv[2:]]
assert "decomp_slide21_20261117" in dest.replace("\\", "/"), dest
os.makedirs(dest, exist_ok=True)
n = b = 0
t0 = time.time()
with open(os.path.join(HERE, "sc_kinds_files.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if not any(p.fullmatch(r["kind"]) for p in pats):
            continue
        name = f"{r['label']}__{r['kind']}__{r['key']}"
        d = os.path.join(dest, name)
        if os.path.exists(d) and os.path.getsize(d) == int(r["bytes"]):
            continue
        shutil.copyfile(os.path.join(SRC, name), d)
        n += 1
        b += int(r["bytes"])
print(f"copied {n} files, {b/1e6:.0f} MB in {time.time()-t0:.0f} s -> {dest}")
