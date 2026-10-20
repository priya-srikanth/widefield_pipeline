"""Held-out test of a DEPTH MODEL for one-camera jaw frames vs the rest-depth anchor (Priya 10-20: cam1 + anchor jaw sits
too HIGH when the jaw is open; "we should also know when jaw is opening (if tongue is out)").

    PYTHONPATH=. python -m scripts.pose3d_jaw_depth_model_check      # dlc env (aniposelib), ~20-30 min

Why: `pose3d.anchor_single_view` puts a one-camera jaw on the seen camera's ray at the trial's RESTING camera-frame
depth. An opening jaw moves down + back, partly along cam1's ~45 deg line of sight, so cam1 + anchor frames land too
close to cam1 (too high in cam4) and the opening is underestimated -- exactly on open-jaw frames.

Model (per session x seen camera): predict the jaw's camera-frame depth CHANGE from rest from (a) the seen camera's 2-D
jaw displacement from the trial's resting 2-D jaw (dx, dy, their squares and product) and (b) the TONGUE in that frame
(out / in from either camera's 2-D tongue; the seen camera's 2-D tongue distance from the resting jaw, px) -- tongue out
=> jaw open. Ordinary least squares, fitted on both-camera frames of OTHER trials (5-fold by trial, all positions); the
point = seen camera's ray at the predicted depth.

Test (as `pose3d_jaw_anchor_check`): truth = both-camera 3-D jaw, response window, side spouts; hide one camera; error per
axis, split by TRUE opening (closed < 1.5 mm <= open); opening bias (|estimate - rest| - |true - rest|); per-trial
lr_wide r and FAR_DIR. Sessions: PS92 0807 / 0818 / 0915, PS93 0807.
"""
import numpy as np
import pandas as pd

from scripts.apparatus_mouth import _rig
from scripts.pose3d import PROD, _cgroup
from wfield_local import o2_inference as oi
from wfield_local import pose3d as P3
from wfield_local import spout_frame as SF
from wfield_local.paths import PathResolver

# as `pose3d_jaw_anchor_check` (copied: importing that script would RUN it -- its body is module level)
SESS = ["PS92:20260807", "PS92:20260818", "PS92:20260915", "PS93:20260807"]
SIDE = ("close_L", "close_R", "far_L", "far_R")
OPEN_MM = 1.5


def anchored(cam, xy, z):
    """Points on ``cam``'s rays through ``xy`` at camera-frame depth ``z`` (per point)."""
    C, U = P3.camera_rays(cam, xy)
    R = cam.get_extrinsics_mat()[:3, :3]
    return C + (z / (U @ R.T)[:, 2])[:, None] * U


def depth(cam, X):
    M = cam.get_extrinsics_mat()
    return (X @ M[:3, :3].T + M[:3, 3])[:, 2]


def _trial_rest(v: np.ndarray, tid: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """Per-frame copy of each trial's pre-cue median of ``v`` (session median where a trial has none)."""
    out = np.tile(np.nanmedian(v[pre], axis=0), (len(v), 1))
    for k in np.unique(tid[pre]):
        m = tid == k
        out[m] = np.nanmedian(v[pre & m], axis=0)
    return out


def _features(xy, xy_rest, tongue_seen, tongue_any):
    d = xy - xy_rest
    tdist = np.linalg.norm(tongue_seen - xy_rest, axis=1)
    tout = np.isfinite(tongue_any).all(1).astype(float)
    tdist = np.where(np.isfinite(tdist), tdist, 0.0)
    return np.c_[np.ones(len(d)), d, d ** 2, d[:, 0] * d[:, 1], tout, tdist]


def main() -> int:
    rv = PathResolver()
    cg0 = _cgroup(rv).subset_cameras_names(["cam1", "cam4"])
    rows, trials = [], []
    rng = np.random.default_rng(0)
    for s in SESS:
        a, d = s.split(":")
        c1 = oi.session_windows(rv, PROD["cam1"], a, d, "cam1")
        c4 = oi.session_windows(rv, PROD["cam4"], a, d, "cam4")
        ix1, f1 = c1[0].reset_index(drop=True), c1[1].reset_index(drop=True)
        ix4, f4 = c4[0].reset_index(drop=True), c4[1].reset_index(drop=True)
        m1, m4 = SF.ruler_medians(f1, ix1), SF.ruler_medians(f4, ix4)
        cg, _ = P3.pnp_group(cg0, {"cam1": m1, "cam4": m4}, _rig(rv, a, d)["geometry"])
        i1, i4 = P3.pair(ix1, ix4)
        o1, o4 = SF.from_medians_lr(m1).origin, SF.from_medians(m4).origin
        j1, _ = P3.cleaned_xy(f1, ix1, "jaw", "cam1", o1)
        j4, _ = P3.cleaned_xy(f4, ix4, "jaw", "cam4", o4)
        g1, _ = P3.cleaned_xy(f1, ix1, "tongue", "cam1", o1)
        g4, _ = P3.cleaned_xy(f4, ix4, "tongue", "cam4", o4)
        A, B, TA, TB = j1[i1], j4[i4], g1[i1], g4[i4]
        X, rep = P3.triangulate(cg, A, B)
        tid, t = ix4.trial_id.to_numpy()[i4], ix4.t_ms.to_numpy(float)[i4]
        X, _ = P3.qc_3d(X, "jaw", tid, t)
        pos, stop = ix4.position.to_numpy()[i4], ix4.stop_ms.to_numpy(float)[i4]
        both = np.isfinite(X).all(1) & (np.nanmax(rep, axis=1) < 15)
        pre = both & (t < 0)
        Xr = _trial_rest(X, tid, pre)
        T_any = np.where(np.isfinite(TA).all(1, keepdims=True), TA, TB)       # tongue seen by either camera
        win = (t >= 70) & (t <= stop)
        test = both & win & np.isin(pos, SIDE)
        fit_ok = both & win
        trials_u = np.unique(tid[fit_ok])
        fold = dict(zip(trials_u, rng.permutation(len(trials_u)) % 5))
        fo = np.array([fold.get(k, -1) for k in tid])
        openv = np.linalg.norm(X - Xr, axis=1)
        for name, ci, xy, tg in (("cam4 only", 1, B, TB), ("cam1 only", 0, A, TA)):
            cam = cg.cameras[ci]
            xy_rest = _trial_rest(xy, tid, pre)
            F = _features(xy, xy_rest, tg, T_any)
            z, zr = depth(cam, X), depth(cam, Xr)
            P_model = np.full_like(X, np.nan)
            for k in range(5):
                tr = fit_ok & (fo != k) & np.isfinite(F).all(1)
                te = test & (fo == k)
                coef, *_ = np.linalg.lstsq(F[tr], (z - zr)[tr], rcond=None)
                P_model[te] = anchored(cam, xy[te], zr[te] + F[te] @ coef)
            P_anchor = np.full_like(X, np.nan)
            P_anchor[test] = anchored(cam, xy[test], zr[test])
            for meth, P in (("anchor", P_anchor), ("model", P_model)):
                E = P[test] - X[test]
                est_open = np.linalg.norm(P[test] - Xr[test], axis=1)
                rows.append(pd.DataFrame({"session": f"{a}_{d}", "view": name, "method": meth, "pos": pos[test],
                                          "state": np.where(openv[test] >= OPEN_MM, "open", "closed"),
                                          "tongue_out": np.isfinite(T_any[test]).all(1),
                                          "err_ml": E[:, 0], "err_ap": E[:, 1], "err_dv": E[:, 2],
                                          "err_3d": np.linalg.norm(E, axis=1),
                                          "open_bias": est_open - openv[test]}))
                Tt = pd.DataFrame({"trial_id": tid[test], "pos": pos[test], "ml_true": X[test, 0], "ml_hat": P[test, 0],
                                   "open": openv[test], "rest_ml": Xr[test, 0]})
                W = Tt[Tt.open > OPEN_MM].groupby(["trial_id", "pos"]).agg(
                    true=("ml_true", "median"), hat=("ml_hat", "median"), rest=("rest_ml", "first"), n=("open", "size"))
                trials.append(W[W.n >= 3].reset_index().assign(session=f"{a}_{d}", view=name, method=meth))
        print(s, "test frames", int(test.sum()), flush=True)
    R = pd.concat(rows, ignore_index=True)
    W = pd.concat(trials, ignore_index=True)
    pd.set_option("display.width", 220)

    def ab(v):
        return v.abs().median()
    print("\nper frame |error| (mm) median, by TRUE jaw state; open_bias = estimated - true opening (median, mm):")
    print(R.groupby(["view", "state", "method"]).agg(n=("err_3d", "size"), ml=("err_ml", ab), ap=("err_ap", ab),
                                                     dv=("err_dv", ab), d3=("err_3d", "median"),
                                                     d3_p90=("err_3d", lambda v: v.quantile(.9)),
                                                     open_bias=("open_bias", "median")).round(3).to_string())
    print("\nsame, open frames by tongue out / in:")
    print(R[R.state == "open"].groupby(["view", "tongue_out", "method"]).agg(
        n=("err_3d", "size"), d3=("err_3d", "median"), open_bias=("open_bias", "median")).round(3).to_string())
    print("\nper session 3-D error median (open frames):")
    print(R[R.state == "open"].pivot_table(index=["view", "session"], columns="method", values="err_3d",
                                           aggfunc="median").round(3).to_string())
    print("\nper trial lr_wide vs true, and FAR_DIR:")
    for (v, s, m), g in W.groupby(["view", "session", "method"]):
        tr, ht = -(g.true - g.rest), -(g.hat - g.rest)
        fl, fr = (g.pos == "far_L").to_numpy(), (g.pos == "far_R").to_numpy()
        fd_t = (tr[fl].median() - tr[fr].median()) / 2
        fd_h = (ht[fl].median() - ht[fr].median()) / 2
        print(f"  {v:9s} {s} {m:6s}: n={len(g)} r={np.corrcoef(tr, ht)[0, 1]:.3f}  FAR_DIR true {fd_t:+.3f} "
              f"est {fd_h:+.3f}")
    out = PathResolver().root("dlc")
    R.to_csv(f"{out}/session_poses/pose3d/jaw_depth_model_check.csv.gz", index=False, float_format="%.4g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
