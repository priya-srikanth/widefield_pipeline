"""Held-out test of a REST-DEPTH ANCHOR for long one-camera JAW stretches (Priya 10-20: recover the centre-spout jaw).

    PYTHONPATH=. python -m scripts.pose3d_jaw_anchor_check      # dlc env (aniposelib), ~20 min

Why: at both centre positions the rod sits end-on in cam1's line of sight to the chin (cam1 keeps ~46-49 % of jaw frames
vs 98-100 % elsewhere), and at close_center cam4 also loses ~40 %. The one-camera fill (`pose3d.fill_single_view`) needs
a both-camera frame within 200 ms, so these long stretches have no 3-D jaw (far_center 38 %, close_center 48 % of frames).

Anchor: a frame seen by ONE camera is placed on that camera's ray at the CAMERA-FRAME DEPTH (z in camera coordinates) of
the trial's resting jaw (median both-camera 3-D jaw before the cue; the session's median rest if the trial has none).
cam4 faces the mouse, so its ray fixes ml and dv and the anchor supplies only ap; cam1 looks up at ~45 deg.

Test (truth = both-camera 3-D jaw, response window, side spouts where both cameras see the jaw): hide one camera, place
the point with the anchor, error per axis (ml / ap / dv) and 3-D; and the per-trial ``lr_wide`` (median sideways jaw over
frames opened > 1.5 mm, the jaw-steering readout) from anchored vs true points. Sessions: PS92 0807 / 0818 / 0915, PS93
0807 (pre / acute / chronic / second animal).
"""
import numpy as np
import pandas as pd

from scripts.apparatus_mouth import _rig
from scripts.pose3d import PROD, _cgroup
from wfield_local import o2_inference as oi
from wfield_local import pose3d as P3
from wfield_local import spout_frame as SF
from wfield_local.paths import PathResolver

SESS = ["PS92:20260807", "PS92:20260818", "PS92:20260915", "PS93:20260807"]
SIDE = ("close_L", "close_R", "far_L", "far_R")
OPEN_MM = 1.5


def anchored(cam, xy, z_rest):
    """Points on ``cam``'s rays through ``xy`` at camera-frame depth ``z_rest`` (per point)."""
    C, U = P3.camera_rays(cam, xy)
    R = cam.get_extrinsics_mat()[:3, :3]
    s = z_rest / (U @ R.T)[:, 2]
    return C + s[:, None] * U


def depth(cam, X):
    M = cam.get_extrinsics_mat()
    return (X @ M[:3, :3].T + M[:3, 3])[:, 2]


rv = PathResolver()
cg0 = _cgroup(rv).subset_cameras_names(["cam1", "cam4"])
rows, trials = [], []
for s in SESS:
    a, d = s.split(":")
    c1 = oi.session_windows(rv, PROD["cam1"], a, d, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], a, d, "cam4")
    ix1, f1 = c1[0].reset_index(drop=True), c1[1].reset_index(drop=True)
    ix4, f4 = c4[0].reset_index(drop=True), c4[1].reset_index(drop=True)
    m1, m4 = SF.ruler_medians(f1, ix1), SF.ruler_medians(f4, ix4)
    cg, _ = P3.pnp_group(cg0, {"cam1": m1, "cam4": m4}, _rig(rv, a, d)["geometry"])
    i1, i4 = P3.pair(ix1, ix4)
    j1, _ = P3.cleaned_xy(f1, ix1, "jaw", "cam1", SF.from_medians_lr(m1).origin)
    j4, _ = P3.cleaned_xy(f4, ix4, "jaw", "cam4", SF.from_medians(m4).origin)
    A, B = j1[i1], j4[i4]
    X, rep = P3.triangulate(cg, A, B)
    tid, t = ix4.trial_id.to_numpy()[i4], ix4.t_ms.to_numpy(float)[i4]
    X, _ = P3.qc_3d(X, "jaw", tid, t)
    pos, stop = ix4.position.to_numpy()[i4], ix4.stop_ms.to_numpy(float)[i4]
    both = np.isfinite(X).all(1) & (np.nanmax(rep, axis=1) < 15)
    F = pd.DataFrame({"trial_id": tid, "t": t, "pos": pos, "both": both})
    rest = F[F.both & (F.t < 0)].groupby("trial_id").groups
    sess_rest = np.nanmedian(X[both & (t < 0)], axis=0)
    Xr = np.tile(sess_rest, (len(F), 1))
    for k, idx in rest.items():
        Xr[(tid == k)] = np.median(X[np.asarray(idx)], axis=0)
    test = both & (t >= 70) & (t <= stop) & np.isin(pos, SIDE)
    for name, cam, xy in (("cam4 only", cg.cameras[1], B), ("cam1 only", cg.cameras[0], A)):
        P = anchored(cam, xy[test], depth(cam, Xr[test]))
        E = P - X[test]
        rows.append(pd.DataFrame({"session": f"{a}_{d}", "view": name, "pos": pos[test], "err_ml": E[:, 0],
                                  "err_ap": E[:, 1], "err_dv": E[:, 2], "err_3d": np.linalg.norm(E, axis=1)}))
        # per-trial lr_wide (mouse-left +), true vs anchored, on the TRUE wide-open frames
        T = pd.DataFrame({"trial_id": tid[test], "pos": pos[test], "ml_true": X[test, 0], "ml_hat": P[:, 0],
                          "open": np.linalg.norm(X[test] - Xr[test], axis=1)})
        W = T[T.open > OPEN_MM].groupby(["trial_id", "pos"]).agg(true=("ml_true", "median"), hat=("ml_hat", "median"),
                                                                 n=("open", "size"))
        W = W[W.n >= 3].reset_index()
        W["rest_ml"] = W.trial_id.map(pd.Series(Xr[:, 0], index=tid).groupby(level=0).first())
        trials.append(W.assign(session=f"{a}_{d}", view=name))
    print(s, "test frames", int(test.sum()))
R = pd.concat(rows, ignore_index=True)
W = pd.concat(trials, ignore_index=True)
pd.set_option("display.width", 200)


def ab(v):
    return v.abs().median()


print("\nper frame |error| (mm), median [p90 3-D]:")
print(R.groupby(["view", "session"]).agg(ml=("err_ml", ab), ap=("err_ap", ab), dv=("err_dv", ab), d3=("err_3d", "median"),
                                         d3_p90=("err_3d", lambda v: v.quantile(.9))).round(3).to_string())
print("\nper trial lr_wide (sideways jaw at wide opening, rel. to rest): anchored vs true")
for (v, s), g in W.groupby(["view", "session"]):
    tr, ht = -(g.true - g.rest_ml), -(g.hat - g.rest_ml)
    far = g.pos.isin(["far_L", "far_R"])
    fl, fr = (g.pos == "far_L").to_numpy(), (g.pos == "far_R").to_numpy()

    def fd(x, fl=fl, fr=fr):
        return (x[fl].median() - x[fr].median()) / 2
    print(f"  {v:9s} {s}: n={len(g)} r={np.corrcoef(tr, ht)[0, 1]:.3f} |diff| median {np.median(np.abs(tr - ht)):.3f} mm"
          f"; FAR_DIR true {fd(tr):.3f} vs anchored {fd(ht):.3f} mm (far trials {int(far.sum())})")
