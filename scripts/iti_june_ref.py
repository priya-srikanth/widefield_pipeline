"""JUNE pre-stroke sessions as a NO-STROKE long-gap reference for the drift check (3.36; Priya 10-27: "can we try a
version of 3.36 with the jun pre sessions too").

The ITI-lick set is August-only (the baseline rule; `iti_licks.main` keeps dates >= 0801), so the drift curve of 3.36
spans only the 8-day August pre span and chronic gaps are extrapolated. June 6-8 (the curated June sessions, 3 per
animal; 3-D lick positions exist for all) are ~60 days before August, about the chronic gap, with no lesion between.
They are extracted with the SAME `iti_licks.extract` (same ROIs, drift removal, lick windows) into a SEPARATE folder
<E>/iti_licks/june_ref/sessions, so no August-baseline consumer (which globs <E>/iti_licks/sessions) can pick them up,
plus their frame-wise area SDs (as `iti_zscore_check.session_sd`) -> june_ref/session_sd.csv.
They are a REFERENCE ONLY: never in a training set, never in the pre baseline. Caveat: June is earlier in training (licks
and engagement differ), so June -> August drift includes learning, not elapsed time alone (conservative for the test).
Usage: python scripts/iti_june_ref.py [--jobs 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from iti_zscore_check import session_sd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

JUNE = il.OUT / "june_ref"
LABELS = [f"{a}_{d}" for a in ("PS92", "PS93", "PS94", "PS95") for d in ("0606", "0607", "0608")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    from wfield_local.writeguard import assert_writable
    assert_writable(JUNE)
    (JUNE / "sessions").mkdir(parents=True, exist_ok=True)
    todo = [(x, str(JUNE)) for x in LABELS if not (JUNE / "sessions" / f"{x}.npz").exists()]
    got, fail = ak.fan_sessions(todo, il.extract, jobs=a.jobs, label="june")
    for it, err in fail:
        print(f"  !! {it[0]}: {err}", flush=True)
    have = sorted(p.stem for p in (JUNE / "sessions").glob("*.npz"))
    got, fail = ak.fan_sessions([(x,) for x in have], session_sd, jobs=a.jobs, label="sd")
    for it, err in fail:
        print(f"  !! sd {it[0]}: {err}", flush=True)
    S = pd.DataFrame({r["label"]: r["sd"] for _it, r in got}).T.sort_index()
    S.to_csv(JUNE / "session_sd.csv")
    print(f"[june_ref] {len(have)} sessions, SDs for {len(S)} -> {JUNE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
