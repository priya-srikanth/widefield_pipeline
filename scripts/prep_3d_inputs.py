"""Cut the encoding window's 3-D inputs from the pose window's CURRENT tables (moved from scratch 10-23, when 3-D became
complete: 119 curated sessions, all four animals). Before this the encoding 3-D results ran on a 10-21 snapshot with
PS92 + PS93 only (ENCODING_FINDINGS status box).

Writes into <out_dir> (a local working folder, not MICROSCOPE):
  pose3d_kin_licks_snapshot_<tag>.csv.gz   frozen copy of <dlc>/session_poses/pose3d/kinematics/licks.csv.gz (per-lick
                                           3-D, trial licks; COMBINED_ENCODING_EXEC3D_TABLE)
  pose3d_spout_positions_<tag>.csv         per session x position, median measured 3-D spout (spout_ml / ap / dv) from the
                                           per-frame tables (COMBINED_ENCODING_TARGET3D_TABLE)
  exec_bins_calib_<tag>.csv                landing-zone calibration (scripts/exec_bins_calibration.py;
                                           COMBINED_ENCODING_BINS_CALIB)
  labels_3d_<tag>.txt                      curated August+ sessions with >= 50 % of usable cam4 licks matched in 3-D and
                                           >= 6 spout positions (the 3-D analyses' session list)
Usage: python scripts/prep_3d_inputs.py <out_dir> <tag>
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POSE3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d")
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
PRE_FROM = "20260806"          # August-only pre baseline (curated)
MIN_FRAC_3D, MIN_SPOUTS = 0.5, 6


def spout_table(dst: Path) -> pd.DataFrame:
    out = []
    for f in sorted(p for an in ANIMALS for p in POSE3D.glob(f"{an}_2026*_frames.csv.gz")):
        if f.name[5:13] < PRE_FROM:
            continue
        F = pd.read_csv(f, usecols=["position", "spout_ml", "spout_ap", "spout_dv"])
        m = F.groupby("position")[["spout_ml", "spout_ap", "spout_dv"]].median().reset_index()
        m.insert(0, "session", f.name[:13])
        out.append(m)
        print(f.name, len(F), flush=True)
    S = pd.concat(out, ignore_index=True)
    S.to_csv(dst, index=False, float_format="%.4f")
    return S


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    W, tag = Path(argv[0]), argv[1]
    W.mkdir(parents=True, exist_ok=True)
    snap = W / f"pose3d_kin_licks_snapshot_{tag}.csv.gz"
    shutil.copy(POSE3D / "kinematics" / "licks.csv.gz", snap)
    SP = spout_table(W / f"pose3d_spout_positions_{tag}.csv")
    cal = W / f"exec_bins_calib_{tag}.csv"
    subprocess.run([sys.executable, "scripts/exec_bins_calibration.py", str(snap), str(cal)], check=True)
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    K = pd.read_csv(snap, usecols=["session", "trial_id", "lick_idx", "side_ml", "out_ap", "down_dv", "reach_mm"])
    cur, avail = set(ak.curated_labels()), set(ce.sessions_available())
    keep, rows = [], []
    for sess in sorted(K.session.unique()):
        a, d = sess.split("_")
        lab = f"{a}_{d[4:]}"
        if d < PRE_FROM or lab not in cur or lab not in avail:
            continue
        L = pd.read_csv(oo.session_dir(rv, oc.RUN, a, d) / "licks.csv")
        ok = ((~L.peak_interp.astype(bool)) & (L.protrusion_px >= ce.MIN_REACH_PX)
              & np.isfinite(L.peak_angle_deg)).to_numpy()
        J = L[["trial_id", "lick_idx"]].merge(K[K.session == sess].drop_duplicates(["trial_id", "lick_idx"]), how="left")
        frac = (ok & J[["side_ml", "out_ap", "down_dv", "reach_mm"]].notna().all(1).to_numpy()).sum() / max(ok.sum(), 1)
        n_sp = int((SP.session == sess).sum())
        rows.append((lab, round(frac, 2), n_sp))
        if frac >= MIN_FRAC_3D and n_sp >= MIN_SPOUTS:
            keep.append(lab)
    R = pd.DataFrame(rows, columns=["label", "frac3d", "spouts"])
    print(R.to_string(index=False))
    print(R.assign(animal=R.label.str[:4], kept=R.label.isin(keep)).groupby("animal").kept.agg(["sum", "count"]))
    (W / f"labels_3d_{tag}.txt").write_text(" ".join(keep))
    print(len(keep), "sessions ->", W / f"labels_3d_{tag}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
