"""Calibration table for the CALIBRATED executed bins of the binned contest (combined_encoding BINS_CALIB_ENV).

Per animal x spout: where that animal's 3-D tongue peak lands PRE-STROKE (median ml / ap / down over its curated August
pre sessions), plus the pooled within-spout SD per axis (the distance scale). One 'default' set per animal (all pre
sessions) and, for each pre session, a leave-that-session-out set (calib_for = <session>), so a pre session is never
binned with its own licks. Licks = the encoding model's usable licks (cam4 peak not interpolated, reach >= MIN_REACH_PX,
finite angle) with a 3-D peak.

Why (10-20): the tongue peak lands ~2 mm deeper than the spout table's position and laterally compressed (~0.4x close,
~0.6x far), so raw nearest-spout geometry put most close-spout licks in far bins (PS92 pre agreement 0.46).

Usage: PYTHONPATH=. python scripts/exec_bins_calibration.py <kinematics licks.csv.gz> <out.csv>
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def pre_licks(kin: str) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    K = pd.read_csv(kin, usecols=["session", "trial_id", "lick_idx", "side_ml", "out_ap", "down_dv"])
    cur = set(ak.curated_labels())
    out = []
    for sess in sorted(K.session.unique()):
        a, d = sess.split("_")
        lab = f"{a}_{d[4:]}"
        if lab not in cur or d < "20260806" or epochs.epoch_of(lab) != "pre":
            continue
        L = pd.read_csv(oo.session_dir(rv, oc.RUN, a, d) / "licks.csv")
        ok = ((~L.peak_interp.astype(bool)) & (L.protrusion_px >= ce.MIN_REACH_PX) & np.isfinite(L.peak_angle_deg))
        L = L[ok][["trial_id", "lick_idx", "position"]]
        J = L.merge(K[K.session == sess].drop_duplicates(["trial_id", "lick_idx"]), on=["trial_id", "lick_idx"]).dropna()
        J["position"] = [oc.pos_key(p, "L") for p in J.position]
        J["animal"], J["session"] = a, sess
        out.append(J.rename(columns={"side_ml": "ml", "out_ap": "ap", "down_dv": "down"}))
    return pd.concat(out, ignore_index=True)


def table(J: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for an, A in J.groupby("animal"):
        for calib_for in ["default", *sorted(A.session.unique())]:
            X = A if calib_for == "default" else A[A.session != calib_for]
            dev = X[["ml", "ap", "down"]] - X.groupby("position")[["ml", "ap", "down"]].transform("median")
            sd = dev.std()
            for p, g in X.groupby("position"):
                rows.append({"animal": an, "calib_for": calib_for, "position": p, "n": len(g),
                             "ml": g.ml.median(), "ap": g.ap.median(), "down": g.down.median(),
                             "sd_ml": sd.ml, "sd_ap": sd.ap, "sd_down": sd.down})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    kin, out = (argv or sys.argv[1:])[:2]
    J = pre_licks(kin)
    T = table(J)
    T.to_csv(out, index=False, float_format="%.4f")
    print(T[T.calib_for == "default"].round(2).to_string(index=False))
    print(f"-> {out} ({J.session.nunique()} pre sessions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
