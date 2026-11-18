"""Read-only scan of a session cache dir: per kind -> n files, total MB, first/last mtime. Writes a CSV."""
import os
import sys
import time
from collections import defaultdict

src, out = sys.argv[1], sys.argv[2]
agg = defaultdict(lambda: [0, 0, 1e18, 0])
rows = []
with os.scandir(src) as it:
    for e in it:
        parts = e.name.split("__")
        if len(parts) < 3:
            continue
        st = e.stat()
        lab = parts[0]
        kind = "__".join(parts[1:-1])
        a = agg[kind]
        a[0] += 1
        a[1] += st.st_size
        a[2] = min(a[2], st.st_mtime)
        a[3] = max(a[3], st.st_mtime)
        rows.append((lab, kind, parts[-1], st.st_size, st.st_mtime))
f = lambda t: time.strftime("%m-%d_%H:%M", time.localtime(t))  # noqa: E731
with open(out, "w", encoding="utf-8") as fh:
    fh.write("kind,n,mb,first,last\n")
    for k, (n, b, t0, t1) in sorted(agg.items(), key=lambda kv: kv[1][3]):
        fh.write(f"{k},{n},{b/1e6:.1f},{f(t0)},{f(t1)}\n")
with open(out.replace(".csv", "_files.csv"), "w", encoding="utf-8") as fh:
    fh.write("label,kind,key,bytes,mtime\n")
    for r in rows:
        fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{f(r[4])}\n")
print(len(rows), "files", len(agg), "kinds")
