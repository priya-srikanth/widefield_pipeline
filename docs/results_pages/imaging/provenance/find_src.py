"""Recover page-figure -> source PNG pairs from the page_figs.py calls in the session transcripts."""
import glob
import json
import re

pairs = {}
for fn in glob.glob(r"C:/Users/SabatiniLab/.claude/projects/C--Users-SabatiniLab/*.jsonl"):
    for line in open(fn, encoding="utf8", errors="ignore"):
        if "page_figs" not in line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        txt = json.dumps(obj).replace("\\\\", "/")
        for m in re.finditer(r"\b([a-zA-Z0-9_]+)=([A-Za-z]:/[^\s\"']+?\.png)", txt):
            pairs.setdefault(m.group(1), set()).add(m.group(2))
for k in sorted(pairs):
    for v in sorted(pairs[k]):
        print(k, v)
