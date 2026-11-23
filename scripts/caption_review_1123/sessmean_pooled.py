"""Six-position session MEANS over licks (Priya 11-23): the gated per-session 3-D table (`kinematics_pooled_figs.
sessions3d` = `pose3d_summary_fig.gated_sessions`, camera working period) with the session MEAN of every lick measure
instead of the median, and the pooled change table the scalar family checks it against.

Writes, to a NEW folder (refuses to overwrite):
  pose3d_sessions.csv               per-session table, lick measures = session means (jaw measures unchanged, medians)
  pose3d_change_pooled_contra.csv   per-animal epoch means of the session changes + boot_delta (as `fig_3d`'s table;
                                    the descriptive interval is drawn from a fresh rng, seed 20261025, so it is not the
                                    draw-replayed interval of the figure script; the scalar family uses its own test)
Before writing, the MEDIAN path is rebuilt and compared with the production pose3d_sessions.csv (must match).

    python sessmean_pooled.py <kin_dir (licks.csv.gz, jaw_trials.csv.gz)> <production pooled dir> <out dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(kin: str, prod: str, out: str) -> None:
    kin, prod, out = Path(kin), Path(prod), Path(out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.pose3d_summary_fig import gated_sessions
    from scripts import kinematics_pooled_figs as K
    L = pd.read_csv(kin / "licks.csv.gz", dtype={"date": str})
    J = pd.read_csv(kin / "jaw_trials.csv.gz", dtype={"date": str})
    med = gated_sessions(L, J, "median")
    P0 = pd.read_csv(prod / "pose3d_sessions.csv", dtype={"date": str})
    cols = [c for c in P0.columns if c in med.columns and c not in ("session", "animal", "date", "epoch")]
    a = med.set_index("session").reindex(P0.session)[cols].to_numpy(float)
    b = P0.set_index("session")[cols].to_numpy(float)
    dev = np.nanmax(np.abs(a - b))
    nan_mis = int((np.isnan(a) != np.isnan(b)).sum())
    print(f"median rebuild vs production pose3d_sessions.csv: {len(P0)} sessions, max |diff| {dev:.3g}, NaN mismatches "
          f"{nan_mis}")
    assert dev < 1e-6 and nan_mis == 0, "the median path does not reproduce production; stop"
    S = gated_sessions(L, J, "mean")
    S.to_csv(out / "pose3d_sessions.csv", index=False)
    rows = []
    rng = np.random.default_rng(20261025)
    for m in ("midline_err_ipsi", "miss_mm", "reach_mm", "speed_peak"):
        long = [S[["animal", "date", "epoch", f"{m}|{p}"]].rename(columns={f"{m}|{p}": "v"}).assign(pos=p)
                for p in K.POS_3D if f"{m}|{p}" in S]
        Lm = pd.concat(long, ignore_index=True)
        P = K.pooled_change(Lm, "v", "pos", K.POS_3D, rng).assign(measure=m)
        if m in K.SIGNED_3D:
            for c in ("PS92", "PS93", "PS94", "PS95", "est"):
                P[c] = -P[c]
            P["lo"], P["hi"] = -P.hi, -P.lo
        rows.append(P)
    pd.concat(rows, ignore_index=True).to_csv(out / "pose3d_change_pooled_contra.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
