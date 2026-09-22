"""Does the reference-restricted gate leave residual satiety in the NON-reference positions?

THE WORRY (Priya, 2026-08-28). The gate judges engagement only at close_L/close_center. If effort
at the hard positions decays BEFORE the animal stops responding at the reference ones, those
late close_R / far misses are counted as engaged and attributed to the lesion rather than to
satiety -- a downward bias on exactly the positions the deficit is measured at.

THE TEST. Within ENGAGED trials only, hit rate by within-session quartile, split reference vs
non-reference. If the gate is clean the non-reference series is flat. If it slopes down while the
REFERENCE series stays flat and above threshold, the slope is residual satiety, and its size is
the bias.

Reference stays flat BY CONSTRUCTION where the gate fired, so the informative comparison is the
non-reference slope against the reference slope in the SAME bins: a shared decline is the session
ending, a non-reference-only decline is the bias.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from wfield_local import spout_behavior as sb
from wfield_local.paths import PathResolver
from wfield_local.precue_engagement_states import REFERENCE

rv = PathResolver()
dirs = sb.discover_sessions(rv, None, None)
print(f"{len(dirs)} session(s)\n", flush=True)

NQ = 4
acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # phase -> (grp,q) -> [hits, n]
per_session = []

for d in dirs:
    try:
        t = sb.load_trials(d, rv)
    except Exception as ex:
        print(f"  !! {d.name}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
        continue
    if t is None or not len(t):
        continue
    pos = t["pos_name"].to_numpy()
    resp = t["responded"].to_numpy().astype(bool)
    order = np.arange(len(t))
    try:
        not_eng, _info = sb.reference_engagement(resp, pos)
    except Exception as ex:
        print(f"  !! {d.name}: gate {type(ex).__name__} {str(ex)[:60]}", flush=True)
        continue
    eng = ~np.asarray(not_eng, bool)
    if eng.sum() < 40:
        continue

    animal, date = d.name.split("_")[0], d.name.split("_")[1]
    from wfield_local import config
    try:
        phase = config.session_phase(animal, date[4:])
    except Exception:
        phase = "?"

    idx = np.flatnonzero(eng)
    q = np.minimum((np.arange(len(idx)) * NQ) // max(1, len(idx)), NQ - 1)
    isref = np.isin(pos[idx], REFERENCE)
    hit = resp[idx]

    sess = {}
    for grp, m in (("reference", isref), ("non-reference", ~isref)):
        for qi in range(NQ):
            sel = m & (q == qi)
            if sel.sum():
                acc[phase][(grp, qi)][0] += int(hit[sel].sum())
                acc[phase][(grp, qi)][1] += int(sel.sum())
                sess[(grp, qi)] = float(hit[sel].mean())
    if all((g, i) in sess for g in ("reference", "non-reference") for i in (0, NQ - 1)):
        per_session.append((f"{animal}_{date[4:]}", phase,
                            sess[("reference", 0)] - sess[("reference", NQ - 1)],
                            sess[("non-reference", 0)] - sess[("non-reference", NQ - 1)]))

print(f"{'phase':10s} {'group':14s} " + " ".join(f"  Q{i+1}   " for i in range(NQ)) + "   Q1-Q4")
print("-" * 78)
for phase in ("pre", "post", "excluded", "?"):
    if phase not in acc:
        continue
    for grp in ("reference", "non-reference"):
        vals, ns = [], []
        for qi in range(NQ):
            h, n = acc[phase][(grp, qi)]
            vals.append(h / n if n else float("nan")); ns.append(n)
        drop = vals[0] - vals[-1]
        print(f"{phase:10s} {grp:14s} " + " ".join(f"{v:.3f} " for v in vals) + f"   {drop:+.3f}")
    print(f"{'':10s} {'n/bin':14s} " + " ".join(f"{n:5d} " for n in ns))
    print()

if per_session:
    import statistics as st
    print("PER-SESSION Q1-Q4 drop (paired, engaged trials only)")
    for phase in ("pre", "post"):
        rows = [(r, nr) for _l, p, r, nr in per_session if p == phase]
        if len(rows) < 3:
            continue
        dr = [r for r, _ in rows]; dn = [nr for _, nr in rows]
        diff = [b - a for a, b in rows]
        print(f"  {phase:9s} n={len(rows):3d}  reference {st.mean(dr):+.3f}  "
              f"non-reference {st.mean(dn):+.3f}  excess {st.mean(diff):+.3f} "
              f"(sd {st.pstdev(diff):.3f})")
