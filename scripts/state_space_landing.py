"""Join each first-lick trial of the per-trial area extracts (`iti_licks` sessions_roi8fr) to its 3-D first-lick
landing (pose3d kinematics `licks.csv.gz`, lick_idx 0), WITHOUT the 65 s/session behaviour reload: walk both tables in
trial order and pair a row with the next 3-D first lick at the same spout whose peak comes LAG_MS after the extract's
first-lick onset (validated against the exact cue-time join on PS94_0820: 3-D peak - onset = 39 +- 18 ms, -2..130).

Landing bin (Priya 11-07, like-for-like executed landings): nearest pre-stroke first-lick landing centre of the animal
(August pre sessions, per target spout, 3-D side / out / down), Mahalanobis with the pooled within-spout covariance --
the encoder page's landing bins in spirit (its softmax bins are pre centre +- 1 SD).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ITI = ("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
       "iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801/sessions_roi8fr")
LICKS = "M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d/kinematics/licks.csv.gz"
OUT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery/landing")
#: licks.csv.gz uses image-side names; all four lesions are LEFT, so L = ipsi, R = contra
SIDE = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
        "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}
LAG_MS = (-15.0, 160.0)
LOOKAHEAD = 6
XYZ = ["side_ml", "out_ap", "down_dv"]


def match(rt_ms, pos, k):
    """Extract rows (in trial order) -> row index into k (first licks sorted by trial), -1 = unmatched."""
    out = np.full(len(rt_ms), -1)
    j = 0
    kp, kt = k.pos.to_numpy(), k.t_peak_cam4.to_numpy(float)
    for i, (r, p) in enumerate(zip(rt_ms, pos)):
        for jj in range(j, min(j + LOOKAHEAD, len(k))):
            if kp[jj] == p and LAG_MS[0] <= kt[jj] - r <= LAG_MS[1]:
                out[i] = jj
                j = jj + 1
                break
    return out


def build():
    K = pd.read_csv(LICKS, usecols=["session", "trial_id", "lick_idx", "position", "contact", "t_peak_cam4", *XYZ,
                                    "epoch", "animal"])
    K = K[K.lick_idx == 0].copy()
    K["pos"] = K.position.map(SIDE)
    rows = []
    for f in sorted(glob.glob(ITI + "/PS9*.npz")):
        lab = Path(f).stem
        an, mmdd = lab.split("_")
        z = np.load(f)
        rt, pos = np.asarray(z["first_rt"], float) * 1000, np.asarray(z["first_pos"]).astype(str)
        k = K[K.session == f"{an}_2026{mmdd}"].sort_values("trial_id").reset_index(drop=True)
        m = match(rt, pos, k)
        d = pd.DataFrame({"label": lab, "animal": an, "row": np.arange(len(rt)), "pos": pos, "rt_ms": rt})
        for c in ["trial_id", "contact", *XYZ]:
            v = k[c].to_numpy()
            d[c] = np.where(m >= 0, v[np.maximum(m, 0)] if len(v) else np.nan, np.nan)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def landing_bins(D, epoch_of):
    """Nearest pre-stroke landing centre per animal (August pre first licks with 3-D), Mahalanobis."""
    D = D.copy()
    D["epoch"] = D.label.map(epoch_of)
    D["land"] = None
    for _an, g in D.groupby("animal"):
        pre = g[(g.epoch == "pre") & g[XYZ].notna().all(1)]
        cen = pre.groupby("pos")[XYZ].mean()
        res = np.concatenate([pre[pre.pos == p][XYZ].to_numpy() - cen.loc[p].to_numpy() for p in cen.index])
        ic = np.linalg.inv(np.cov(res.T))
        ok = g[XYZ].notna().all(1)
        X = g.loc[ok, XYZ].to_numpy()
        d2 = np.stack([np.einsum("ij,jk,ik->i", X - c, ic, X - c) for c in cen.to_numpy()], 1)
        D.loc[g.index[ok], "land"] = np.asarray(cen.index)[d2.argmin(1)]
    return D


if __name__ == "__main__":
    import sys
    from wfield_local import epochs
    OUT.mkdir(parents=True, exist_ok=True)
    D = landing_bins(build(), epochs.epoch_of)
    D.to_csv(OUT / "first_lick_landing.csv", index=False)
    truth = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if truth and truth.exists():
        t = np.load(truth)
        g = D[D.label == "PS94_0820"].sort_values("row")
        ok = (t >= 0)
        print("validation PS94_0820: rows", len(g), "| matched", int(g.trial_id.notna().sum()),
              "| exact trial agreement", float((g.trial_id.to_numpy()[ok] == t[ok]).mean()))
    print("matched share by epoch:\n", D.assign(m=D.trial_id.notna()).groupby("epoch").m.mean().round(3))
    print("on-target share (land == pos) by animal x epoch, far contra:\n",
          D[(D.pos == "far_contra") & D.land.notna()].assign(on=lambda x: x.land == x.pos)
          .groupby(["animal", "epoch"]).on.mean().unstack().round(2))
    print("far-contra landing bins, acute:\n",
          D[(D.pos == "far_contra") & (D.epoch == "acute")].land.value_counts())
