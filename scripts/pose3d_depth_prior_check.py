"""Held-out test of DEPTH PRIORS for cam1-ONLY tongue points (PS92; Priya 10-19: small licks cam4 never sees).

    python -m scripts.pose3d_depth_prior_check      # dlc env (aniposelib), ~20 min

Truth = both-camera 3-D (PnP, edge offsets, QC); cam4 hidden. q = X . U = depth along cam1's ray measured from the mouth
zero's foot point on that ray. (a) q = 0: the point on cam1's ray closest to the mouth zero (no data); (b) q learned from
cam1's 2-D position (KNN, k=100) on the animal's August pre sessions, leave-one-session-out; acute days scored with (b)
trained on pre ONLY -- tests whether a learned prior drags post-stroke licks toward the pre pattern. Errors by distance
from the mouth zero (small licks = the target).
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from scripts.apparatus_mouth import _rig
from scripts.pose3d import _cgroup
from wfield_local import o2_inference as oi
from wfield_local import pose3d as P3
from wfield_local import spout_frame as SF
from wfield_local.paths import PathResolver

rv = PathResolver()
PRE = ["20260805", "20260806", "20260807", "20260809", "20260810", "20260811", "20260813", "20260814"]
ACUTE = ["20260818", "20260819", "20260820"]
cg0 = _cgroup(rv).subset_cameras_names(["cam1", "cam4"])
rows = []
for d in PRE + ACUTE:
    c1 = oi.session_windows(rv, "lp_cam1_ctx_prod_20261019", "PS92", d, "cam1")
    c4 = oi.session_windows(rv, "lp_r4_prod_20261015", "PS92", d, "cam4")
    m1, m4 = SF.ruler_medians(c1[1], c1[0]), SF.ruler_medians(c4[1], c4[0])
    cg, _ = P3.pnp_group(cg0, {"cam1": m1, "cam4": m4}, _rig(rv, "PS92", d)["geometry"])
    i1, i4 = P3.pair(c1[0], c4[0])
    a, _ = P3.cleaned_xy(c1[1], c1[0], "tongue", "cam1", SF.from_medians_lr(m1).origin)
    b, _ = P3.cleaned_xy(c4[1], c4[0], "tongue", "cam4", SF.from_medians(m4).origin)
    a, b = a[i1], b[i4]
    X, rep = P3.triangulate(cg, a, b)
    X, _ = P3.qc_3d(X, "tongue", c1[0].trial_id.to_numpy()[i1], c1[0].t_ms.to_numpy(float)[i1])
    ok = np.isfinite(X).all(1) & (np.nanmax(rep, axis=1) < 15)
    C, U = P3.camera_rays(cg.cameras[0], a[ok])
    Xo = X[ok]
    q = (Xo * U).sum(1)
    rows.append(pd.DataFrame({"date": d, "x": a[ok, 0], "y": a[ok, 1], "q": q, "cu": U @ C,
                              "ux": U[:, 0], "uy": U[:, 1], "uz": U[:, 2], "cx": C[0], "cy": C[1], "cz": C[2],
                              "X": Xo[:, 0], "Y": Xo[:, 1], "Z": Xo[:, 2],
                              "dist": np.linalg.norm(Xo, axis=1)}))
    print(d, ok.sum())
T = pd.concat(rows, ignore_index=True)


def place(T, qhat):
    s = qhat - T.cu.to_numpy()
    return np.c_[T.cx + s * T.ux, T.cy + s * T.uy, T.cz + s * T.uz]


truth = T[["X", "Y", "Z"]].to_numpy()
T["err_a"] = np.linalg.norm(place(T, np.zeros(len(T))) - truth, axis=1)
T["err_b"] = np.nan
for d in PRE + ACUTE:
    tr = T[T.date.isin([p for p in PRE if p != d])]
    te = T.date == d
    knn = KNeighborsRegressor(n_neighbors=100).fit(tr[["x", "y"]].to_numpy(), tr.q.to_numpy())
    T.loc[te, "err_b"] = np.linalg.norm(place(T[te], knn.predict(T.loc[te, ["x", "y"]].to_numpy())) - truth[te.to_numpy()], axis=1)
T["bin"] = pd.cut(T.dist, [0, 1, 2, 3, 4, 10], labels=["0-1", "1-2", "2-3", "3-4", ">4"])
T["epoch"] = np.where(T.date.isin(PRE), "pre (LOSO)", "acute (trained on pre)")
pd.set_option("display.width", 200)
print(T.groupby(["epoch", "bin"], observed=True).agg(n=("dist", "size"), a_median=("err_a", "median"),
      a_p90=("err_a", lambda x: x.quantile(.9)), b_median=("err_b", "median"),
      b_p90=("err_b", lambda x: x.quantile(.9))).round(2).to_string())
