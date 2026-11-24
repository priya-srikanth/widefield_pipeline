"""Calibration table for the CALIBRATED executed bins of the binned contest (combined_encoding BINS_CALIB_ENV).

Per animal x spout: where that animal's 3-D tongue peak lands PRE-STROKE (median ml / ap / down over its curated August
pre sessions), plus the pooled within-spout SD per axis (the distance scale). One 'default' set per animal (all pre
sessions) and, for each pre session, a leave-that-session-out set (calib_for = <session>), so a pre session is never
binned with its own licks. Licks = the encoding model's usable licks (cam4 peak not interpolated, reach >= MIN_REACH_PX,
finite angle) with a 3-D peak, of CAMERA-WORKING-PERIOD trials only (Priya 11-06, A20: `trial_classes.gate_kin_licks` --
first cue to the session's last camera lick, stopped runs (A2 + excise) out; a pre session with NO camera classes (no
video) contributes NOTHING, as gain_bio; ``WFIELD_KIN_GATE=legacy`` = every post-cue lick, the pre-11-06 rule). The
frozen encoder's snapshot table (exec_bins_calib_20261023.csv) is NOT rewritten by this change: a new table is a new
encoder input and needs its own decision.

Why (10-20): the tongue peak lands ~2 mm deeper than the spout table's position and laterally compressed (~0.4x close,
~0.6x far), so raw nearest-spout geometry put most close-spout licks in far bins (PS92 pre agreement 0.46).

FOUR-CAMERA (11-24, Priya OK): with a FUSED kinematics table (it has ``lick_key``, e.g. pose3d_4cam_3net_20261121_sync60/
kinematics_fused_postcue/licks.csv.gz) the licks are the encoder's own fourcam licks (`combined_encoding._fourcam_table`,
table == "licks") joined to the 3-D rows by ``lick_key`` (`session_licks.align_kin`), with the encoder's usable rule
for them (cam4 saw the lick: peak not interpolated, reach >= MIN_REACH_PX, finite angle -- the licks `exec_bins` bins).
Needs COMBINED_ENCODING_LICKS=fourcam (or the fused default) in the environment. Write it under a NEW name.

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
    from wfield_local import trial_classes as tc
    K = pd.read_csv(kin, usecols=lambda c: c in ("session", "trial_id", "lick_idx", "lick_key", "side_ml", "out_ap",
                                                 "down_dv"))
    fused = "lick_key" in K
    cur = set(ak.curated_labels())
    out = []
    for sess in sorted(K.session.unique()):
        a, d = sess.split("_")
        lab = f"{a}_{d[4:]}"
        if lab not in cur or d < "20260806" or epochs.epoch_of(lab) != "pre":
            continue
        L = pd.read_csv(oo.session_dir(rv, oc.RUN, a, d) / "licks.csv")
        ok = ((~L.peak_interp.astype(bool)) & (L.protrusion_px >= ce.MIN_REACH_PX) & np.isfinite(L.peak_angle_deg))
        Ks = tc.gate_kin_licks(K[K.session == sess])     # camera working period (A20), per pre session only
        if fused:
            J = _fused_licks(rv, a, d, L, sess, Ks)
        else:
            L = L[ok][["trial_id", "lick_idx", "position"]]
            J = L.merge(Ks.drop_duplicates(["trial_id", "lick_idx"]), on=["trial_id", "lick_idx"]).dropna()
        J["position"] = [oc.pos_key(p, "L") for p in J.position]
        J["animal"], J["session"] = a, sess
        out.append(J.rename(columns={"side_ml": "ml", "out_ap": "ap", "down_dv": "down"}))
    return pd.concat(out, ignore_index=True)


def _fused_licks(rv, a: str, d: str, L4: pd.DataFrame, sess: str, Ks: pd.DataFrame) -> pd.DataFrame:
    """One pre session's FOUR-CAMERA usable licks with their 3-D peak (see the module docstring): the fused post-cue
    licks, cam4's usable rule by lick_key (as `combined_encoding._load_behaviour` fourcam), the gated 3-D rows by
    lick_key. Columns trial_id, lick_idx, position, side_ml, out_ap, down_dv (complete rows only)."""
    from wfield_local import combined_encoding as ce
    from wfield_local import session_licks as SL
    A4 = ce._fourcam_table(rv, a, d)
    L = A4[A4["table"] == "licks"].reset_index(drop=True)
    k4 = L4.assign(lick_key=SL.lick_key(L4.trial_id, L4.t_ms)).drop_duplicates("lick_key").set_index("lick_key")
    J4 = k4.reindex(L.lick_key.to_numpy())
    interp = J4.peak_interp.astype(object).where(J4.peak_interp.notna(), True).astype(bool).to_numpy()
    ok = (~interp) & (J4.protrusion_px.to_numpy(float) >= ce.MIN_REACH_PX) & np.isfinite(J4.peak_angle_deg.to_numpy(float))
    X = SL.align_kin(L, Ks, ["side_ml", "out_ap", "down_dv"], sess)
    J = pd.concat([L[["trial_id", "lick_idx", "position"]], X], axis=1)[ok]
    return J.dropna()


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
