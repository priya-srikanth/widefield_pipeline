"""Slide-21 decomposition: the slide's frozen POSITION readout (scripts/rebuild_figs/fig_state_position_sessions.py) on
one epoch/data folder, printed as one compact row set (per-animal pre / acute levels, pre sessions per animal, change vs
own pre with Holm over the 6 tests, seed 20261102 -- the figure script's own load() and stats()).

    python slide21_readout_table.py <epoch/data dir> <label> [--drop-first-pre N]

--drop-first-pre N drops the first N pre-stroke session ordinals per animal before the stats (a SIDECAR-ONLY check: on
the OLD sidecars the first 3 pre ordinals are the June 06-06..08 sessions, assuming the collector's chronological
session order; the frozen model itself is NOT retrained, so this isolates only the change of the pre READOUT set).
Appends one JSON line per call to slide21_results.jsonl beside this file (or $SLIDE21_RESULTS)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "rebuild_figs"))
sys.path.insert(0, str(REPO))
import fig_state_position_sessions as F  # noqa: E402


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("label")
    ap.add_argument("--drop-first-pre", type=int, default=0)
    a = ap.parse_args(argv)
    import wfield_local
    print(f"[code] wfield_local from {wfield_local.__file__}")
    with contextlib.redirect_stdout(io.StringIO()):
        D = F.load(a.data_dir)
    if a.drop_first_pre:
        D = D[~((D.epoch == "pre") & (D.session_ordinal < a.drop_first_pre))].copy()
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        T = F.stats(D)
    _ = buf
    P = D[D.readout == "position"]
    lev = P.groupby(["animal", "epoch"], observed=True).retained.mean().unstack("epoch")
    npre = P[P.epoch == "pre"].groupby("animal").size()
    row = {"label": a.label, "data_dir": str(a.data_dir), "drop_first_pre": a.drop_first_pre,
           "n_pre": {k: int(v) for k, v in npre.items()},
           "pre": {k: round(float(v), 3) for k, v in lev["pre"].items()},
           "acute": {k: round(float(v), 3) for k, v in lev["acute"].items()},
           "subacute": {k: round(float(v), 3) for k, v in lev["subacute"].items()},
           "chronic": {k: round(float(v), 3) for k, v in lev["chronic"].items()}}
    for (r, e), (iv, p, ph) in T.items():
        row[f"{r}_{e}"] = {"delta": round(iv.point, 3), "lo": round(iv.lo, 3), "hi": round(iv.hi, 3),
                           "p": round(p, 4), "holm": round(ph, 4)}
    print(f"{a.label}: n_pre {row['n_pre']}")
    print("  pre   " + " ".join(f"{row['pre'][x]:.3f}" for x in F.ANIMALS))
    print("  acute " + " ".join(f"{row['acute'][x]:.3f}" for x in F.ANIMALS))
    for e in ("acute", "subacute", "chronic"):
        c = row[f"position_{e}"]
        print(f"  change {e:8s} {c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] Holm {c['holm']:.4f}")
    out = Path(os.environ.get("SLIDE21_RESULTS", Path(__file__).with_name("slide21_results.jsonl")))
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
