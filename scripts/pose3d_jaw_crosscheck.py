"""JAW cross-check, trial by trial: cam4 (2-D, v36 `cohort/tables/jaw_lateral/trials.csv`) vs 3-D (`pose3d_kinematics`
jaw_trials). Priya 10-19: "is jaw 3d also good?" -- run the jaw cross check.

    python -m scripts.pose3d_jaw_crosscheck

Matched per trial (same session, trial id, position): cam4 ``lr_wide`` (px, sideways at wide opening, + = ipsi) vs 3-D
``lr_wide_mm``; cam4 ``open_max`` (px) vs 3-D ``open_max_mm`` and its components (side / back / down). Per session: Pearson
r and slope (px per mm), and FAR_DIR (far_L - far_R) / 2 from each. Windows differ slightly by design (cam4 cue + 70 ..
3000 ms, 3-D cue + 70 .. trial stop), and the 3-D wide-opening threshold is 1.5 mm vs cam4's 50 px.
Output `session_poses/pose3d/crosscheck/jaw_{per_trial.csv.gz, agreement.csv}`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

V36 = "lp_r4_heatmap_20261013__clean_v36"


def _r_slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return np.nan, np.nan, int(ok.sum())
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), float(np.polyfit(x[ok], y[ok], 1)[0]), int(ok.sum())


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    J3 = pd.read_csv(d3 / "kinematics" / "jaw_trials.csv.gz", dtype={"date": str})
    J4 = pd.read_csv(Path(rv.root("orofacial")) / V36 / "cohort" / "tables" / "jaw_lateral" / "trials.csv",
                     dtype={"date": str})
    J4 = J4[J4.animal.isin(J3.animal.unique())]
    # trial ids: try as-is and +1 (0- vs 1-based); keep the shift that matches positions best per session
    out = []
    for s, g3 in J3.groupby("session"):
        a, d = s.split("_")
        g4 = J4[(J4.animal == a) & (J4.date == d)]
        best = None
        for sh in (0, 1):
            m = g3.merge(g4.assign(trial_id=g4.trial_id + sh), on="trial_id", suffixes=("", "_c4"))
            agree = (m.position == m.position_c4).mean() if len(m) else 0
            if best is None or agree > best[0]:
                best = (agree, sh, m)
        agree, sh, m = best
        m = m[m.position == m.position_c4]
        out.append(m.assign(id_shift=sh, pos_agree=agree))
        print(f"{s}: {len(m)} matched trials (trial id shift {sh}, position agreement {agree:.2f})")
    M = pd.concat(out, ignore_index=True)
    M.to_csv(d3 / "crosscheck" / "jaw_per_trial.csv.gz", index=False, float_format="%.4g")
    rows = []
    for s, g in M.groupby("session"):
        r1, s1, n1 = _r_slope(g.lr_wide_mm.to_numpy(float), g.lr_wide.to_numpy(float))
        r2, s2, n2 = _r_slope(g.open_max_mm.to_numpy(float), g.open_max.to_numpy(float))
        r3, _, _ = _r_slope(g.open_max_down_dv.to_numpy(float), g.open_max.to_numpy(float))
        r4, _, _ = _r_slope(g.open_max_back_ap.to_numpy(float), g.open_max.to_numpy(float))
        row = {"session": s, "epoch": g.epoch.iloc[0], "n": len(g), "r_lateral": r1, "px_per_mm_lateral": s1,
               "r_opening": r2, "px_per_mm_opening": s2, "r_cam4open_vs_3d_down": r3, "r_cam4open_vs_3d_back": r4}
        f4 = g.groupby("position").lr_wide.median()
        f3 = g.groupby("position").lr_wide_mm.median()
        if {"far_L", "far_R"} <= set(f4.index):
            row["FAR_DIR_cam4_px"] = (f4["far_L"] - f4["far_R"]) / 2
            row["FAR_DIR_3d_mm"] = (f3["far_L"] - f3["far_R"]) / 2
        rows.append(row)
    A = pd.DataFrame(rows)
    A.to_csv(d3 / "crosscheck" / "jaw_agreement.csv", index=False)
    pd.set_option("display.width", 220)
    print(A.round(2).to_string(index=False))
    ok = A.FAR_DIR_cam4_px.notna() & A.FAR_DIR_3d_mm.notna()
    if ok.sum() > 3:
        print(f"\nsession FAR_DIR, cam4 px vs 3-D mm: r = {np.corrcoef(A.FAR_DIR_cam4_px[ok], A.FAR_DIR_3d_mm[ok])[0, 1]:.2f} "
              f"over {int(ok.sum())} sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
