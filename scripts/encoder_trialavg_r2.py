"""TRIAL-AVERAGED R^2 of an encoder run (Priya 10-22: "is this overall variance we're talking about or single trial?"
-> "yes check").

The encoder's region R^2 is FRAME-level, held out, within a window: it scores every frame of every trial, so it mixes the
condition-locked part (each spout's average response, the only part task kernels can carry) with single-trial
variability (which movement terms can also carry). The trial-to-trial models score the single-trial part alone. This
script scores the third view, the CONDITION-LOCKED part alone: how well a model reproduces each spout's AVERAGE
response, from the per-position PETHs every run saves (`peth`, cue-aligned; `peth_lick`, first post-cue lick):

  R2_avg = 1 - sum_{spout, t in window} (data - pred)^2 / sum (data - mean_window)^2,   per area, per session,

offset free per area (the window mean of data and of the prediction removed, as the frame-level score), for the full
prediction and for each saved part alone (a part of ONE fit, so a part's R2 can be negative). Frozen-encoder variants
are separate fits, so `no_movement` vs `movement_only` folders give task-only vs movement-only trial-averaged R^2 directly.
Windows: cue 0 .. 1.5 s after the cue; lick 0 .. 1 s after the first post-cue lick.

Usage: python scripts/encoder_trialavg_r2.py --folder <run folder under imaging_encoding> [--folder ...]
Output: <E>/qc/trialavg_r2_<folder>.csv (per session x align x part x area) and a printed animal x epoch summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = {"peth": (0.0, 1.5), "peth_lick": (0.0, 1.0)}
AREAS = ("cortex", "SSp-m_left", "SSp-m_right", "MOs_left", "MOp_left")


def session_rows(js: Path) -> list[dict]:
    meta = json.loads(js.read_text(encoding="utf-8"))
    z = np.load(js.with_suffix(".npz"))
    parts, areas = meta["peth_parts"], meta["areas"]
    rows = []
    for key, (a0, a1) in WINDOWS.items():
        if key not in z.files:
            continue
        P, off = z[key], z["peth_off_s"]                      # (parts, positions, areas, off)
        w = (off >= a0) & (off < a1)
        D = P[parts.index("data")][..., w]                   # (positions, areas, t)
        Dc = D - np.nanmean(D, axis=(0, 2), keepdims=True)
        sst = np.nansum(Dc ** 2, axis=(0, 2))
        for pi, part in enumerate(parts):
            if part == "data":
                continue
            Pr = P[pi][..., w]
            Prc = Pr - np.nanmean(Pr, axis=(0, 2), keepdims=True)
            r2 = 1.0 - np.nansum((Dc - Prc) ** 2, axis=(0, 2)) / np.maximum(sst, 1e-12)
            for area in AREAS:
                if area in areas:
                    rows.append({"label": meta["label"], "animal": meta["animal"], "epoch": meta["epoch"],
                                 "align": "cue" if key == "peth" else "lick", "part": part, "area": area,
                                 "r2_avg": float(r2[areas.index(area)])})
    return rows


def main(argv=None) -> int:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", action="append", required=True)
    a = ap.parse_args(argv)
    E = oo.root(PathResolver()) / oc.RUN / "imaging_encoding"
    qc = E / "qc"
    assert_writable(qc)
    pd.set_option("display.width", 220)
    for folder in a.folder:
        sess = sorted((E / folder / "sessions").glob("*.json"))
        T = pd.DataFrame([r for js in sess for r in session_rows(js)])
        if T.empty:
            print(f"{folder}: no PETHs")
            continue
        T = T[T.label.str[-4:] >= "0801"]                    # August-only pre baseline (primary)
        slug = folder.replace("/", "_").replace("\\", "_")
        T.to_csv(qc / f"trialavg_r2_{slug}.csv", index=False, float_format="%.4f")
        S = (T[T.area == "cortex"].groupby(["align", "part", "animal", "epoch"]).r2_avg.mean()
             .unstack("epoch").reindex(columns=["pre", "acute", "subacute", "chronic"]))
        print(f"\n=== {folder}: trial-averaged R2, cortex (per animal, mean over sessions)")
        print(S.round(3).to_string())
        print(f"-> {qc / f'trialavg_r2_{slug}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
