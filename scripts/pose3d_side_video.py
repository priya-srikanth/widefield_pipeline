"""Four-camera 3-D check clip: cam1 / cam4 / cam2 / cam3 raw views with their 2-D keypoints and the FUSED 3-D point
reprojected, and the 3-D pose below (Priya 2026-11-04: "show me clips labeled in all 4 cameras with 3d keypoint views
below"). Pipeline = `pose3d_side.fuse_session` (11-04): cleaned cam1 / cam4 (production) AND cleaned cam2 / cam3 (`side_clean`),
water-drop rule, robust four-camera triangulation (worst view dropped beyond `REJECT_MM`), one-view fill from the
neighboring frames (`fill_n`) and the 3-D QC; spout: cam4 = the commanded tip, cam1's edge offset; nose: the side nose
moved onto the cam1 / cam4 definition with a per-session offset.

    conda activate dlc
    python -m scripts.pose3d_side_video PS93:20260814 [PS93:20260821 ...] [--per-position 1] [--t0 -100 --t1 700]

Trials = the side review-clip trials (2 per position; ``--per-position`` of them), cue + t0 .. t1 ms. Panels: filled dots =
that camera's cleaned keypoint USED in the 3-D point, open circles = a cleaned keypoint the fusion REJECTED (its ray
disagreed with the other cameras), crosses = the fused 3-D point projected into the camera (with that camera's offset for the part, so it lands on its own label definition). 3-D panels: seen from cam2's
side (profile: ap / dv) and the oblique front-right view; commanded spouts as rings (the trial's target filled), trails =
the last 120 ms of tongue and jaw. Also writes ``fused_<a>_<d>.csv.gz`` (per frame, per part: ml / ap / dv, views, rep px
per camera) and prints how often each part is 3-D with 2 vs 4 cameras. August sessions only (June side intrinsics open).
Output: `session_poses/pose3d_side/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ORDER = ["cam1", "cam4", "cam2", "cam3"]
BGR = {"nose": (0, 215, 255), "jaw": (60, 220, 60), "tongue": (60, 60, 255), "spout": (255, 200, 0)}
HEX = {"nose": "#ffd700", "jaw": "#3cdc3c", "tongue": "#ff3c3c", "spout": "#00c8ff"}
CROP = {"cam1": 380, "cam4": 380, "cam2": 280, "cam3": 280}
TILE = 300
TRAIL_MS = 120.0


class Reader:
    """Raw video frames, sequential when possible (one seek per jump)."""
    def __init__(self, path):
        import cv2
        self.cap, self.last = cv2.VideoCapture(str(path)), None

    def get(self, i: int):
        import cv2
        i = int(i)
        if self.last is None or i != self.last + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, im = self.cap.read()
        self.last = i
        return im if ok else None


def view_from(cam) -> tuple[float, float]:
    """matplotlib (elev, azim) putting the viewer at ``cam``'s center (a PnP-posed camera: world mm)."""
    M = cam.get_extrinsics_mat()
    C = -M[:3, :3].T @ M[:3, 3]
    d = C / np.linalg.norm(C)
    return float(np.degrees(np.arcsin(d[2]))), float(np.degrees(np.arctan2(d[1], d[0])))


def run(rv, session: str, per_position: int, t0: float, t1: float) -> Path:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.pose3d import _june_factor
    from scripts.pose3d_figures import OBLIQUE
    from scripts.pose3d_side_check import CLIPS, PREDS
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d_side as PS
    from wfield_local import spout_world as SW
    animal, date = session.split(":")
    if _june_factor(date):
        raise SystemExit(f"{session}: June -- side intrinsics not validated, not rendered")
    Z = PS.fuse_session(rv, animal, date, Path(PREDS), Path(rv.root("dlc")) / CLIPS)
    F, order, g, pts, rig, side = Z["F"], Z["order"], Z["g"], Z["pts"], Z["rig"], Z["side"]
    R = {p: (Z["parts"][p]["X"], None, Z["parts"][p]["views"]) for p in PS.PARTS}
    out = Path(rv.root("dlc")) / "session_poses" / "pose3d_side"
    out.mkdir(parents=True, exist_ok=True)

    # per-frame table + coverage: cam1 + cam4 measured vs the four-camera pipeline
    T = F[["trial_id", "position", "t_ms"]].copy()
    print(f"\n{session}: {len(T)} frames; px/mm vs cam4 {Z['stats']['scale']}; side nose offsets {Z['stats']['nose_offsets']}; "
          f"droplet frames {Z['stats']['droplet_frames']}")
    for part in PS.PARTS:
        P = Z["parts"][part]
        for k, ax in enumerate(("ml", "ap", "dv")):
            T[f"{part}_{ax}"] = P["X"][:, k]
        T[f"{part}_views"] = P["views"]
        for j, c in enumerate(order):
            T[f"{part}_used_{c}"] = P["used"][j]
            T[f"{part}_dist_mm_{c}"] = P["dist"][j]
        st = Z["stats"][part]
        print(f"  {part:6s} 3-D: cam1+cam4 {st['cam1+cam4']:5.1%} -> 4-camera {st['fused']:5.1%} (one-view fill "
              f"{st['one_view_fill']:4.1%}) | rejected views " + " ".join(f"{c} {st['rejected_' + c]}" for c in order)
              + " | median ray dist mm " + " ".join(f"{c} {np.nanmedian(P['dist'][j]):.2f}" for j, c in enumerate(order)))
    qc = {k: v for k, v in Z["stats"].items() if k.endswith(("_far", "_behind", "_spike"))}
    print(f"  3-D QC drops: {qc}")
    T.to_csv(out / f"fused_{animal}_{date}.csv.gz", index=False, float_format="%.3f")

    # drawing helpers
    gpart = {p: Z["parts"][p]["g"] for p in PS.PARTS}
    cams = {c.get_name(): c for c in g.cameras}
    mouth_px = {c: PS.project(cams[c], np.zeros((1, 3)))[0] for c in order}
    nose_w = np.nanmedian(R["nose"][0], axis=0)
    nose_px = {c: PS.project(cams[c], nose_w[None])[0] for c in order}
    # cam1 / cam4: centred on the mouth (the nose is far up the frame there); side views: between mouth and nose
    centre = {c: mouth_px[c] if c in ("cam1", "cam4") else (mouth_px[c] + nose_px[c]) / 2 for c in order}
    readers = {"cam1": Reader(oi.videos_for(rv, animal, date, "cam1")[0]),
               "cam4": Reader(oi.videos_for(rv, animal, date, "cam4")[0]),
               **{c: Reader(sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{c}_*.avi"))[0])
                  for c in PS.SIDE}}
    src = {"cam1": F.src1.to_numpy(), "cam4": F.src4.to_numpy(),
           **{c: side[c][0].src_frame.to_numpy()[F[f"i_{c}"].to_numpy()] for c in PS.SIDE}}
    tgt = {p: np.asarray(SW.commanded_world(rig["geometry"], p), float) for p in SW.POSITIONS}
    allX = np.vstack([R[p][0] for p in PS.PARTS] + [np.array(list(tgt.values()))])
    lo, hi = np.nanpercentile(allX, 1, axis=0) - 0.5, np.nanpercentile(allX, 99, axis=0) + 0.5
    views = [(*view_from(cams["cam2"]), "3-D seen from cam2 (profile)"), (OBLIQUE[0], OBLIQUE[1], "3-D oblique (front-right, above)")]
    fig = plt.figure(figsize=(2 * TILE / 100, 1.3 * TILE / 100), dpi=100)
    axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]

    trials = (F.drop_duplicates("trial_id").groupby("position").head(per_position)
              .sort_values("position").trial_id.tolist())
    f_out = out / f"video_4cam_{animal}_{date}.mp4"
    vw = None
    for tr in trials:
        rows = np.flatnonzero((F.trial_id.to_numpy() == tr) & (F.t_ms.to_numpy(float) >= t0) & (F.t_ms.to_numpy(float) <= t1))
        pos = str(F.position.iloc[rows[0]])
        for k in rows:
            tiles = []
            for j, c in enumerate(order):
                im = readers[c].get(src[c][k])
                if im is None:
                    im = np.zeros((450, 600, 3), np.uint8)
                for p in PS.PARTS:
                    q = pts[p][j, k]
                    if np.isfinite(q).all():
                        used = bool(Z["parts"][p]["used"][j, k])
                        cv2.circle(im, (int(round(q[0])), int(round(q[1]))), 4 if used else 6, BGR[p], -1 if used else 1,
                                   cv2.LINE_AA)
                    X = R[p][0][k]
                    if np.isfinite(X).all():
                        r_ = PS.project([cc for cc in gpart[p].cameras if cc.get_name() == c][0], X[None])[0]
                        cv2.drawMarker(im, (int(round(r_[0])), int(round(r_[1]))), BGR[p], cv2.MARKER_TILTED_CROSS, 13, 2,
                                       cv2.LINE_AA)
                h = CROP[c] // 2
                x0 = int(np.clip(centre[c][0] - h, 0, im.shape[1] - CROP[c]))
                y0 = int(np.clip(centre[c][1] - h, 0, im.shape[0] - CROP[c]))
                tile = cv2.resize(im[y0:y0 + CROP[c], x0:x0 + CROP[c]], (TILE, TILE))
                cv2.putText(tile, c, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            sel = (F.trial_id.to_numpy() == tr) & (F.t_ms.to_numpy(float) <= F.t_ms.iloc[k]) & \
                  (F.t_ms.to_numpy(float) >= F.t_ms.iloc[k] - TRAIL_MS)
            for ax, (el, az, name) in zip(axes, views):
                ax.cla()
                for q, c_ in tgt.items():
                    ax.scatter(*c_, s=40, facecolor=HEX["spout"] if q == pos else "none", edgecolor="0.4", lw=0.8)
                for p in PS.PARTS:
                    X = R[p][0]
                    if p in ("tongue", "jaw"):
                        A = X[sel]
                        A = A[np.isfinite(A).all(1)]
                        if len(A) > 1:
                            ax.plot(A[:, 0], A[:, 1], A[:, 2], color=HEX[p], lw=1.5)
                    if np.isfinite(X[k]).all():
                        ax.scatter(*X[k], s=36, color=HEX[p], edgecolor="k", lw=0.5)
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(lo[1], hi[1])
                ax.set_zlim(lo[2], hi[2])
                ax.set_box_aspect(hi - lo)
                ax.set_xlabel("ml (+R)", fontsize=7)
                ax.set_ylabel("ap (out)", fontsize=7)
                ax.set_zlabel("dv (up)", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.view_init(elev=el, azim=az)
                ax.set_title(name, fontsize=8)
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.03, top=0.92, wspace=0.0)
            fig.canvas.draw()
            im3 = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
            top = np.hstack(tiles)
            frame = np.vstack([top, cv2.resize(im3, (top.shape[1], int(top.shape[1] * im3.shape[0] / im3.shape[1])))])
            bar = np.zeros((40, frame.shape[1], 3), np.uint8)
            nv = "  ".join(f"{p} {int(R[p][2][k])}" for p in PS.PARTS)
            cv2.putText(bar, f"{animal} {date}  trial {tr}  {pos}  t = {F.t_ms.iloc[k]:+.0f} ms from cue   views: {nv}",
                        (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bar, "dot = keypoint used, ring = rejected, x = 4-camera 3-D reprojected | nose yellow, jaw green, tongue red, "
                        "spout blue | 3-D: rings = commanded spouts (target filled), trails 120 ms",
                        (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            frame = np.vstack([bar, frame])
            if vw is None:
                vw = cv2.VideoWriter(str(f_out), cv2.VideoWriter_fourcc(*"mp4v"), 25, (frame.shape[1], frame.shape[0]))
            vw.write(frame)
        for _ in range(12):
            vw.write(frame)
    if vw is not None:
        vw.release()
    plt.close(fig)
    print(f"-> {f_out}")
    return f_out


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--per-position", type=int, default=1)
    ap.add_argument("--t0", type=float, default=-100.0)
    ap.add_argument("--t1", type=float, default=700.0)
    a = ap.parse_args(argv)
    rv = PathResolver()
    for s in a.sessions:
        run(rv, s, a.per_position, a.t0, a.t1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
