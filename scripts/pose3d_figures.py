"""Pictures of the 3-D tongue (`wfield_local.pose3d`): trajectory maps, and a paired cam1 | cam4 | 3-D lick video.

    python -m scripts.pose3d_figures maps  PS92:20260807 PS92:20260818      # from session_poses/pose3d/ (run scripts.pose3d first)
    python -m scripts.pose3d_figures video PS92:20260807 [--per-position 1]  # recomputes the session (production O2 outputs)

``maps``: one column per session; rows = TOP (ml x ap, looking down, mouse facing up the page), SIDE (ap x dv, mouse's right
side) and FRONT (ml x dv, as seen from the spouts: mouse RIGHT on image LEFT). Thin lines = individual licks (up to 30 per
position, >= 5 frames, onset -> end), thick = their time-normalised mean; filled circles = commanded spout tips (= the
fitted frame), + = the mouth set point (origin), grey = nose and resting jaw medians.
``video``: per position the most typical lick (miss distance nearest that position's median, 15-45 frames); panels = cam1
and cam4 crops around the apparatus mouth (raw video; green circle = that camera's 2-D tongue, red x = the TRIANGULATED
point projected back, yellow ring = the target spout projected) and the 3-D path so far, seen from cam4's real position (``--view cam4 | cam1 | hybrid``,
from the calibration). 250 fps shown at 10 fps (25x slow).
dlc env (aniposelib). Output: `session_poses/pose3d/figures/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COL = {"far_L": "#1f77b4", "close_L": "#6baed6", "close_center": "#999999", "far_center": "#333333",
       "close_R": "#fc9272", "far_R": "#d62728"}
VIEWS = (("top", "ml", "ap", "ml (mm)  mouse left <-> mouse right", "ap (mm)  out of the mouth ->"),
         ("side", "ap", "dv", "ap (mm)  out of the mouth ->", "dv (mm)  up"),
         ("front", "ml", "dv", "ml (mm)  mouse right <-> mouse left (seen from the spouts)", "dv (mm)  up"))


def _out(rv) -> Path:
    d = Path(rv.root("dlc")) / "session_poses" / "pose3d" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lick_paths(S: pd.DataFrame, L: pd.DataFrame, min_frames: int = 5) -> list[tuple[str, np.ndarray]]:
    """[(position, (n, 3) ml/ap/dv path)] for every lick, cut from the frame table by trial and onset .. end."""
    g = {k: v for k, v in S.groupby("trial_id")}
    out = []
    for r in L[L.n_frames >= min_frames].itertuples():
        f = g.get(r.trial_id)
        if f is None:
            continue
        m = (f.t_ms >= r.t_onset_ms - 0.5) & (f.t_ms <= r.t_onset_ms + 4.0 * (r.n_frames - 1) + 0.5)
        P = f.loc[m, ["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
        if np.isfinite(P).all() and len(P) >= min_frames:
            out.append((r.position, P))
    return out


def _resample(P: np.ndarray, n: int = 25) -> np.ndarray:
    s = np.linspace(0, len(P) - 1, n)
    return np.stack([np.interp(s, np.arange(len(P)), P[:, k]) for k in range(3)], 1)


def cmd_maps(rv, sessions: list[str]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.apparatus_mouth import _rig
    from wfield_local import spout_world as SW
    d = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    fig, axes = plt.subplots(3, len(sessions), figsize=(5.2 * len(sessions), 14), squeeze=False)
    rng = np.random.default_rng(0)
    for j, s in enumerate(sessions):
        animal, date = s.split(":")
        S = pd.read_csv(d / f"{animal}_{date}_frames.csv.gz")
        L = pd.read_csv(d / f"{animal}_{date}_licks.csv")
        geo = _rig(rv, animal, date)["geometry"]
        paths = _lick_paths(S, L)
        rest = S[S.t_ms < 0]
        for i, (name, a, b, xl, yl) in enumerate(VIEWS):
            ax = axes[i, j]
            ka, kb = "ml ap dv".split().index(a), "ml ap dv".split().index(b)
            sx = -1.0 if name == "front" else 1.0                      # front: as seen from the spouts
            for p in SW.POSITIONS:
                mine = [P for q, P in paths if q == p]
                if not mine:
                    continue
                for k in rng.choice(len(mine), min(30, len(mine)), replace=False):
                    ax.plot(sx * mine[k][:, ka], mine[k][:, kb], color=COL[p], lw=0.5, alpha=0.25)
                M = np.mean([_resample(P) for P in mine], 0)
                ax.plot(sx * M[:, ka], M[:, kb], color=COL[p], lw=2.5, label=f"{p} (n={len(mine)})")
                c = SW.commanded_world(geo, p)
                ax.scatter(sx * c[ka], c[kb], s=140, color=COL[p], edgecolor="k", zorder=5)
            ax.scatter(0, 0, marker="+", s=200, color="k", zorder=6)
            for part, mk in (("nose", "^"), ("jaw", "s")):
                v = rest[[f"{part}_ml", f"{part}_ap", f"{part}_dv"]].median().to_numpy(float)
                if np.isfinite(v).all():
                    ax.scatter(sx * v[ka], v[kb], marker=mk, s=70, color="0.6", edgecolor="k", zorder=5)
                    ax.annotate(part, (sx * v[ka], v[kb]), textcoords="offset points", xytext=(6, 4), fontsize=8)
            ax.set_aspect("equal")
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.set_title(f"{animal} {date} -- {name}", fontsize=10)
            if i == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("3-D tongue tip per lick (cam1 + cam4), mouth-set-point frame; circles = commanded spout tips", fontsize=11)
    fig.tight_layout()
    f = _out(rv) / f"maps_{'_'.join(s.replace(':', '_') for s in sessions)}.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    return f


def _frame(cap, i: int) -> np.ndarray:
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, im = cap.read()
    if not ok:
        raise RuntimeError(f"frame {i} unreadable")
    return im


def view_angles(cg, world, which: str = "cam4") -> tuple[float, float]:
    """matplotlib (elev, azim) that puts the 3-D panel's viewer where a REAL camera sits (calibration extrinsics mapped
    into the session's world frame), so the panel matches that camera's image: ``cam4`` (front), ``cam1`` (below) or
    ``hybrid`` (the normalised mean of the two directions). Priya 10-19: a generic angle was confusing."""
    X0 = (world.R.T @ (-world.t)) / world.scale                              # the mouth (origin) in calibration space
    dirs = {}
    for c in cg.cameras:
        M = c.get_extrinsics_mat()
        C = -M[:3, :3].T @ M[:3, 3]                                           # camera centre, calibration space
        d = world.R @ (C - X0)
        dirs[c.get_name()] = d / np.linalg.norm(d)
    d = dirs["cam4"] if which == "cam4" else dirs["cam1"] if which == "cam1" else dirs["cam4"] + dirs["cam1"]
    d = d / np.linalg.norm(d)
    return float(np.degrees(np.arcsin(d[2]))), float(np.degrees(np.arctan2(d[1], d[0])))


def cmd_video(rv, session: str, per_position: int = 1, view: str = "cam4") -> Path:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.apparatus_mouth import _rig
    from scripts.pose3d import PROD, _cgroup, _june_factor
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d as P3
    from wfield_local import spout_world as SW
    animal, date = session.split(":")
    c1 = oi.session_windows(rv, PROD["cam1"], animal, date, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], animal, date, "cam4")
    if c1 is None or c4 is None:
        raise SystemExit(f"{session}: production outputs missing")
    if "video" in c1[0] or "video" in c4[0]:
        raise SystemExit(f"{session}: split day -- not supported by the video yet")
    rig = _rig(rv, animal, date)
    cg = _cgroup(rv).subset_cameras_names(["cam1", "cam4"])
    jf = _june_factor(date)
    S, info = P3.session_3d(cg, rig, c1[0], c1[1], c4[0], c4[1], june_factor=jf)
    cg = info["cg"]                                         # the session's cameras (PnP-posed by default)
    L = P3.lick_peaks(S, rig)
    w = info["world"]
    elev, azim = view_angles(cg, w, view)
    sh1, sh4 = info["shift_px"]
    K4 = cg.cameras[1].get_camera_matrix()
    pp4 = np.array([K4[0, 2], K4[1, 2]])

    def to_calib(W):                                        # world mm -> calibration space (inverse of to_world)
        return ((np.asarray(W, float) - w.t) @ w.R) / w.scale

    def proj(W):                                            # -> (cam1 px, cam4 RAW px)
        p = cg.project(to_calib(np.atleast_2d(W)))
        p1, p4 = p[0].copy(), p[1].copy()
        p1[:, 0] -= sh1                                     # undo the disparity shift (raw video coordinates)
        p4[:, 0] -= sh4
        p4 = p4 if not jf else pp4 + (p4 - pp4) / jf        # then the June scaling
        return p1, p4

    m1, m4 = np.asarray(info["mouth_px"]["cam1"], float), np.asarray(info["mouth_px"]["cam4"], float)
    m1[0] -= sh1
    m4[0] -= sh4
    if jf:
        m4 = pp4 + (m4 - pp4) / jf
    caps = {c: cv2.VideoCapture(str(oi.videos_for(rv, animal, date, c)[0])) for c in ("cam1", "cam4")}
    raw = {"cam1": c1[1], "cam4": c4[1]}
    ridx = {}                                               # video frame -> first row (a frame can sit in two windows)
    for cam, ix in (("cam1", c1[0]), ("cam4", c4[0])):
        r_ = pd.Series(np.arange(len(ix)), index=ix.src_frame.to_numpy())
        ridx[cam] = r_[~r_.index.duplicated()]
    H = 360
    f_out = _out(rv) / f"lick_paired_{animal}_{date}_{view}view.mp4"
    vw = None
    tgt = {p: SW.commanded_world(rig["geometry"], p) for p in SW.POSITIONS}
    fig = plt.figure(figsize=(4.4, 3.6), dpi=110)
    ax3 = fig.add_subplot(projection="3d")
    for p in SW.POSITIONS:
        g = L[(L.position == p) & L.n_frames.between(15, 45)]
        if g.empty:
            continue
        med = g.miss_mm.median()
        for r in g.iloc[np.argsort(np.abs(g.miss_mm.to_numpy() - med))[:per_position]].itertuples():
            F = S[(S.trial_id == r.trial_id) & (S.t_ms >= r.t_onset_ms - 40) & (S.t_ms <= r.t_onset_ms + 4 * r.n_frames + 40)]
            path = []
            for fr in F.itertuples():
                X = np.array([fr.tongue_ml, fr.tongue_ap, fr.tongue_dv])
                if np.isfinite(X).all():
                    path.append(X)
                panels = []
                for cam, src, mouth in (("cam1", fr.src1, m1), ("cam4", fr.src4, m4)):
                    im = _frame(caps[cam], src)
                    row = raw[cam].iloc[int(ridx[cam][src])] if src in ridx[cam].index else None
                    if row is not None and row.tongue_likelihood >= 0.4:
                        cv2.circle(im, (int(row.tongue_x), int(row.tongue_y)), 7, (0, 220, 0), 2, cv2.LINE_AA)
                    pa, pb = proj(tgt[p])
                    q = pa[0] if cam == "cam1" else pb[0]
                    cv2.circle(im, (int(q[0]), int(q[1])), 11, (0, 220, 255), 2, cv2.LINE_AA)
                    if np.isfinite(X).all():
                        pa, pb = proj(X)
                        q = pa[0] if cam == "cam1" else pb[0]
                        cv2.drawMarker(im, (int(q[0]), int(q[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
                    x0, y0 = int(mouth[0]) - 170, int(mouth[1]) - 170
                    x0, y0 = max(0, min(x0, im.shape[1] - 340)), max(0, min(y0, im.shape[0] - 340))
                    crop = cv2.resize(im[y0:y0 + 340, x0:x0 + 340], (H, H))
                    cv2.putText(crop, f"{cam} (raw view)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    panels.append(crop)
                ax3.cla()
                for q, c in tgt.items():
                    ax3.scatter(*c, s=60, color=COL[q], edgecolor="k", alpha=1.0 if q == p else 0.35)
                ax3.scatter(0, 0, 0, marker="+", s=80, color="k")
                if path:
                    A = np.array(path)
                    ax3.plot(A[:, 0], A[:, 1], A[:, 2], color=COL[p], lw=2)
                    ax3.scatter(*A[-1], s=30, color="r")
                ax3.set_xlim(-3.5, 3.5)
                ax3.set_ylim(-1, 6)
                ax3.set_zlim(-5, 1)
                ax3.set_xlabel("ml (+R)", fontsize=7)
                ax3.set_ylabel("ap (out)", fontsize=7)
                ax3.set_zlabel("dv (up)", fontsize=7)
                ax3.tick_params(labelsize=6)
                ax3.view_init(elev=elev, azim=azim)
                ax3.set_title(f"3-D, seen from {view}  |  {p}  t = {fr.t_ms - r.t_onset_ms:+.0f} ms", fontsize=8)
                fig.canvas.draw()
                im3 = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
                panels.append(cv2.resize(im3, (int(im3.shape[1] * H / im3.shape[0]), H)))
                frame = np.hstack(panels)
                bar = np.zeros((28, frame.shape[1], 3), np.uint8)
                cv2.putText(bar, f"{animal} {date}  trial {r.trial_id}  {p}  miss {r.miss_mm:.2f} mm  lat_err {r.lat_err_mm:+.2f} mm"
                            "   green = 2-D tongue, red x = 3-D point reprojected, yellow = target spout",
                            (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                frame = np.vstack([bar, frame])
                if vw is None:
                    vw = cv2.VideoWriter(str(f_out), cv2.VideoWriter_fourcc(*"mp4v"), 10, (frame.shape[1], frame.shape[0]))
                vw.write(frame)
            for _ in range(8):                                                      # pause between licks
                vw.write(frame)
    if vw is not None:
        vw.release()
    plt.close(fig)
    return f_out


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["maps", "video"])
    ap.add_argument("sessions", nargs="+", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--per-position", type=int, default=1)
    ap.add_argument("--view", default="cam4", choices=["cam4", "cam1", "hybrid"], help="video: the 3-D panel's viewpoint")
    a = ap.parse_args(argv)
    rv = PathResolver()
    if a.what == "maps":
        print(f"-> {cmd_maps(rv, a.sessions)}")
    else:
        for s in a.sessions:
            print(f"-> {cmd_video(rv, s, a.per_position, a.view)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
