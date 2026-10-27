"""ALLEN AREAS WITH THE ALM / tjM1 SPLIT (Priya 10-27: "could we do allen with our ALM/tjM1 split?").

The KEY-row area decoder uses whole MOs / MOp and leaves ALM / tjM1 out (they are the anterolateral ROI_FRAC of MOs /
MOp, `iti_licks.roi_masks`, so they overlap). The split basis replaces, per hemisphere, MOs -> ALM + MOs_rest and
MOp -> tjM1 + MOp_rest (disjoint). The session files hold the MOs and ALM means, so the rest mean is exact:
  rest = (mean_MOs - f * mean_ALM) / (1 - f),   f = n_ALM / n_MOs pixels in that session's masks.
This script writes, per session, f for the four pairs and the FRAME-level SD of the two rest areas per hemisphere (same
masks, drift removal and x1e3 as `iti_zscore_check.session_sd`) -> <E>/iti_licks/zscore_check/session_sd_split.csv.
Used by `iti_where_decoder` source "areas_split" and `iti_lick_order` (WFIELD_LICK_DECODER=areas_split).
Usage: python scripts/iti_area_split_sd.py [--jobs 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

F = il.OUT / "zscore_check" / "session_sd_split.csv"
PAIRS = (("MOs", "ALM"), ("MOp", "tjM1"))
HEMI = ("left", "right")


def one(item) -> dict:
    from wfield_local import combined_encoding as ce
    label = item[0]
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    out, cols = {}, []
    for src, dst in PAIRS:
        for h in HEMI:
            whole, sub = M[f"{src}_{h}"], M[f"{dst}_{h}"]
            out[f"f_{dst}_{h}"] = sub.sum() / whole.sum()
            M[f"{src}_rest_{h}"] = whole & ~sub
            cols.append(f"{src}_rest_{h}")
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    out.update(dict(zip(cols, R.std(0))))
    return {"label": label, "row": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    from wfield_local import analysis_kit as ak
    labels = sorted(p.stem for p in (il.OUT / "sessions").glob("*.npz"))
    got, fail = ak.fan_sessions([(x,) for x in labels], one, jobs=a.jobs, label="split sd")
    for it, err in fail:
        print(f"  !! {it[0]}: {err}", flush=True)
    S = pd.DataFrame({r["label"]: r["row"] for _it, r in got}).T.sort_index()
    S.to_csv(F)
    print(S.describe().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
