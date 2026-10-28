"""Within-session z-scored features for the frozen where-decoder (Priya 10-26: "z-scored within session is ok"; NOT a
pre-lick baseline, because the inter-trial period carries position information).

The decoder's area traces already have each session's mean and slow drift removed (`iti_licks.extract`: adopted
hemo-corrected SVD -> area means -> drift basis regressed out -> x1e3), so a session-wide LEVEL change cannot reach the
decoder. A session-wide SCALE change can (F0 / expression, or the post-stroke rise in total variance, page 9.20): a
frozen linear readout then moves decoded means toward or away from its centre. Here every area is divided by its
frame-wise SD in that session (all frames after the same drift removal; NOT over licks, which would force every
session's mean lick feature to 0 and erase offsets by construction), in pre training and post scoring alike, and
the r, like-for-like offset and per-area readouts are rerun.
AREA BASIS (Priya 10-27 late: "use split bases, for consistency"): Allen areas WITH the ALM / tjM1 split by default
(iti_where_decoder.area_basis; the rest areas' frame SDs come from iti_area_split_sd.py); WFIELD_AREA_BASIS=plain = the
earlier set, outputs tagged _plain.

Steps: (1) per-session area SDs -> <E>/iti_licks/zscore_check/session_sd.csv (fanned over sessions, cached);
(2) same decoders with scaled features -> r per target (`iti_where_decoder.decode` logic) and per-lick predictions
(`iti_where_scatter.predictions`) -> zscore_check/where_preds_z<PRED_TAG>.csv (basis-specific cache name) ->
like-for-like offsets (iti_where_beyond_pull.py --z, converted 10-27 late), side by side with the unscaled run.
Usage: python scripts/iti_zscore_check.py [--jobs 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

OUTZ = il.OUT / "zscore_check"


def session_sd(item) -> dict:
    """Frame-wise SD of every decoder column in one session (module-level for spawn); same ROI masks and drift
    removal as `iti_licks.extract`."""
    from wfield_local import combined_encoding as ce
    label = item[0]
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    return {"label": label, "sd": dict(zip(cols, R.std(0)))}


def sds(jobs: int) -> pd.DataFrame:
    f = OUTZ / il.SD_NAME
    if f.exists():
        return pd.read_csv(f, index_col=0)
    from wfield_local import analysis_kit as ak
    labels = sorted(p.stem for p in (il.OUT / il.SESS_NAME).glob("*.npz"))
    got, fail = ak.fan_sessions([(x,) for x in labels], session_sd, jobs=jobs, label="sd")
    for it, err in fail:
        print(f"  !! {it[0]}: {err}", flush=True)
    S = pd.DataFrame({r["label"]: r["sd"] for _it, r in got}).T.sort_index()
    S.to_csv(f)
    return S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    OUTZ.mkdir(parents=True, exist_ok=True)
    sds(a.jobs)                                    # computes + caches the plain areas' frame SDs when absent
    import iti_where_beyond_pull as bp
    import iti_where_decoder as wd
    import iti_where_scatter as ws
    load0 = wd.load
    names, SDb = wd.area_meta()                 # basis feature names + frame SDs (split: + the rest areas SDs)

    def load_z(source, *a, **k):
        D = load0(source, *a, **k)
        for x, d in D.items():
            sd = SDb.loc[x, names].to_numpy(float)
            for st in ("iti", "trial"):
                X, C = d[st]
                d[st] = (X / sd[None, :], C)
        return D

    wd.load = load_z                            # restored below: in-process callers must not inherit the z-scoring
    try:
        Rz = wd.decode("areas")                 # (a) r per target, scaled vs unscaled
        P = ws.predictions()                    # (b) per-lick predictions with scaled features
    finally:
        wd.load = load0
    pd.set_option("display.width", 220)
    Rz.to_csv(OUTZ / il.tagged("where_decoder_z.csv"), index=False)
    R0 = pd.read_csv(il.OUT / il.tagged("where_decoder.csv"))
    R0 = R0[R0.source == "areas"]
    rows = []
    for tag, R in (("unscaled", R0), ("z-scored", Rz)):
        w = R.pivot_table(index=["set", "target", "animal", "epoch"], values="r").r.unstack("epoch")
        ch = (w[["acute", "subacute", "chronic"]].div(w["pre"], axis=0) - 1)
        rows.append(ch.groupby(["set", "target"]).mean().assign(version=tag))
    print("== frozen r, normalised change vs pre (mean of animals)")
    print(pd.concat(rows).reset_index().set_index(["set", "target", "version"]).sort_index().round(3).to_string())
    # (b) -> like-for-like offsets (was beyond-pull; 10-27 late)
    P.to_csv(il.pred_file("zscore_check/where_preds_z.csv"), index=False)
    if il.AREA_BASIS == "split":                          # its outputs are not basis-tagged
        bp.main(z=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
