"""FRAME-LEVEL spread of each basis component per session (Priya 10-27: "frame-level scaling for SVD -- yes, seems
important esp. since we're analysing ITI").

The Allen-area features are z-scored by each area's SD over ALL imaging frames of the session (zscore_check/session_sd.csv).
The SVD / LocaNMF features were z-scored by their SD over the session's LICK windows, which the larger post-stroke lick
response inflates -- dividing part of a post-stroke change out. This computes the like-for-like frame-level SD: the same
drift-removed component traces that `iti_basis_decoder.features` samples at the licks, SD over every frame.
Output <E>/iti_licks/basis_<source>/frame_sd.csv (rows = session labels, columns = components).
Sessions run in parallel (analysis_kit.fan_sessions). Usage: python scripts/iti_basis_frame_sd.py [--sources svdbasis locanmf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402


def frame_sd(item) -> np.ndarray:
    """(label, source) -> SD over frames of each drift-removed component trace. Module-level (spawn)."""
    from iti_basis_decoder import features
    label, source = item
    R, _ft = features(label, source)
    return np.asarray(R, np.float64).std(0)


def main() -> int:
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["svdbasis", "locanmf"], choices=["svdbasis", "locanmf"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    for source in a.sources:
        d = il.OUT / f"basis_{source}"
        labels = sorted(p.stem for p in (d / "sessions").glob("*.npz"))
        res, fails = ak.fan_sessions([(x, source) for x in labels], frame_sd, jobs=a.jobs, label=f"frame SD {source}")
        if fails:
            print("FAILED:", [f[0] for f in fails])
        rows = {it[0]: v for it, v in res}
        T = pd.DataFrame.from_dict(rows, orient="index")
        T.index.name = "label"
        T.to_csv(d / "frame_sd.csv")
        print(f"{source}: {len(T)} sessions x {T.shape[1]} components -> {d / 'frame_sd.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
