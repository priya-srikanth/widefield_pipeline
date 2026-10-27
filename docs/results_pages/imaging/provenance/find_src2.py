"""For every PNG on the imaging page, find the page_figs.py 'name=$VAR/path' argument that produced it (last one wins)."""
import glob
import json
import os
import re
from pathlib import Path

PAGE = Path(__file__).with_name("imaging_page")
names = {p.stem for p in PAGE.glob("*.png")}
src = {}
files = sorted(glob.glob(r"C:/Users/SabatiniLab/.claude/projects/C--Users-SabatiniLab/*.jsonl"), key=os.path.getmtime)
for fn in files:
    for line in open(fn, encoding="utf8", errors="ignore"):
        if "page_figs" not in line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        msg = obj.get("message", {})
        if msg.get("role") != "assistant":
            continue
        for c in msg.get("content", []):
            if not isinstance(c, dict) or c.get("type") != "tool_use":
                continue
            cmd = c.get("input", {}).get("command", "")
            if "page_figs" not in cmd:
                continue
            var = dict(re.findall(r"\b([A-Z][A-Z0-9_]*)=['\"]?([^'\";\s]+)['\"]?", cmd))
            for m in re.finditer(r"\b([a-zA-Z0-9_]+)=((?:\$\{?\w+\}?|[A-Za-z]:)?/?[^\s\"']+?\.png)", cmd):
                n, p = m.group(1), m.group(2)
                if n not in names:
                    continue
                p = re.sub(r"\$\{?(\w+)\}?", lambda v, var=var: var.get(v.group(1), v.group(0)), p)
                src[n] = p
for n in sorted(names):
    print(n, src.get(n, "?"))
