"""Side-camera (cam2 / cam3) CLEANING QC: raw confident keypoints vs what `pose3d_side.side_clean` keeps, as a clip
(Priya 2026-11-04: QC the side cleaning before adopting it, as for cam1). The side thresholds are cam4's
`orofacial_clean` values scaled by each side camera's px / mm at the mouth (`pose3d_side.side_params`); this is the
check that the scaling removes the wrong points (jumps, a jaw riding the tongue, isolated frames) and keeps real motion.

    conda activate dlc
    python -m scripts.side_clean_qc PS93:20260814 PS93:20260821 [--per-position 1] [--t0 -100 --t1 700]

Per trial (the side review-clip trials): top = cam2 | cam3 raw views, ring = raw confident point (tongue >= 0.4, jaw >=
0.6), filled dot = the cleaned MEASURED point that enters the 3-D (no interpolated frames); bottom = the trial's jaw
(image y, px from its trial median) and tongue (distance from the projected mouth, px) per camera, raw gray vs cleaned
color, cursor = the current frame. Prints per camera x part: raw confident frames, kept, removed by cleaning. Writes
`session_poses/pose3d_side/side_clean_qc_<a>_<d>.mp4` + ``side_clean_qc_stats.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BGR = {"jaw": (60, 220, 60), "tongue": (60, 60, 255)}
TILE = 360
CROP = 280


def run(rv, session: str, per_position: int, t0: float, t1: float) -> tuple[Path, list[dict]]:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.pose3d_side_check import CLIPS, PREDS
    from scripts.pose3d_side_video import Reader
    from wfield_local import pose3d_side as PS
    animal, date = session.split(":")
    Z = PS.fuse_session(rv, animal, date, Path(PREDS), Path(rv.root("dlc")) / CLIPS)
    cams = {c.get_name(): c for c in Z["g"].cameras}
    raw, clean, mouth = {}, {}, {}
    stats = []
    for c in PS.SIDE:
        ix, flat = Z["side"][c]
        mouth[c] = PS.project(cams[c], np.zeros((1, 3)))[0]
        cl = PS.side_clean(flat, ix, mouth[c], Z["stats"]["scale"][c])
        for part in ("jaw", "tongue"):
            r = PS.confident_xy(flat, part)
            raw[(c, part)], clean[(c, part)] = r, cl[part]
            n_raw = int(np.isfinite(r).all(1).sum())
            n_keep = int((np.isfinite(r).all(1) & np.isfinite(cl[part]).all(1)).sum())
            stats.append({"session": f"{animal}_{date}", "cam": c, "part": part, "frames": len(r), "raw_confident": n_raw,
                          "kept": n_keep, "removed": n_raw - n_keep,
                          "removed_frac": round((n_raw - n_keep) / n_raw, 4) if n_raw else np.nan,
                          "scale_vs_cam4": round(Z["stats"]["scale"][c], 3)})
    ix2 = Z["side"]["cam2"][0].reset_index(drop=True)
    ix3 = Z["side"]["cam3"][0].reset_index(drop=True)
    readers = {c: Reader(sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{c}_*.avi"))[0]) for c in PS.SIDE}
    out = Path(rv.root("dlc")) / "session_poses" / "pose3d_side"
    f_out = out / f"side_clean_qc_{animal}_{date}.mp4"
    trials = ix2.drop_duplicates("trial_id").groupby("position").head(per_position).sort_values("position").trial_id.tolist()
    vw = None
    for tr in trials:
        rows = {c: np.flatnonzero((ixc.trial_id.to_numpy() == tr) & (ixc.t_ms.to_numpy(float) >= t0)
                                  & (ixc.t_ms.to_numpy(float) <= t1)) for c, ixc in (("cam2", ix2), ("cam3", ix3))}
        n = min(len(rows["cam2"]), len(rows["cam3"]))
        if n == 0:
            continue
        # trace panel, drawn once per trial; the cursor is added per frame
        fig, axs = plt.subplots(2, 2, figsize=(2 * TILE / 100, 1.6 * TILE / 100), dpi=100, sharex=True)
        tt = Z["side"]["cam2"][0].t_ms.to_numpy(float)[rows["cam2"][:n]]
        for col, c in enumerate(PS.SIDE):
            k = rows[c][:n]
            for row_, part in enumerate(("jaw", "tongue")):
                ax = axs[row_, col]
                R, C = raw[(c, part)][k], clean[(c, part)][k]
                if part == "jaw":
                    ref = np.nanmedian(R[:, 1]) if np.isfinite(R[:, 1]).any() else 0.0
                    yr, yc = R[:, 1] - ref, C[:, 1] - ref
                    ax.set_ylabel("jaw y (px, + down)", fontsize=7)
                else:
                    yr, yc = np.linalg.norm(R - mouth[c], axis=1), np.linalg.norm(C - mouth[c], axis=1)
                    ax.set_ylabel("tongue from mouth (px)", fontsize=7)
                ax.plot(tt, yr, ".", color="0.6", ms=3, label="raw confident")
                ax.plot(tt, yc, ".", color=np.array(BGR[part][::-1]) / 255, ms=3, label="cleaned (kept)")
                ax.set_title(f"{c} {part}", fontsize=8)
                ax.tick_params(labelsize=6)
                ax.axvline(0, color="k", lw=0.5)
        axs[1, 0].set_xlabel("ms from cue", fontsize=7)
        axs[1, 1].set_xlabel("ms from cue", fontsize=7)
        axs[0, 0].legend(fontsize=6, loc="upper left")
        fig.tight_layout()
        fig.canvas.draw()
        base = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
        xlim = axs[0, 0].get_xlim()
        bbox = [ax.get_position() for ax in axs[0]]
        plt.close(fig)
        H, W = base.shape[:2]
        for i in range(n):
            tiles = []
            for c in PS.SIDE:
                kk = rows[c][i]
                im = readers[c].get(Z["side"][c][0].src_frame.to_numpy()[kk])
                if im is None:
                    im = np.zeros((450, 600, 3), np.uint8)
                for part in ("jaw", "tongue"):
                    r, cc = raw[(c, part)][kk], clean[(c, part)][kk]
                    if np.isfinite(r).all():
                        cv2.circle(im, (int(round(r[0])), int(round(r[1]))), 7, BGR[part], 1, cv2.LINE_AA)
                    if np.isfinite(cc).all():
                        cv2.circle(im, (int(round(cc[0])), int(round(cc[1]))), 3, BGR[part], -1, cv2.LINE_AA)
                h = CROP // 2
                x0 = int(np.clip(mouth[c][0] - h, 0, im.shape[1] - CROP))
                y0 = int(np.clip(mouth[c][1] - h, 0, im.shape[0] - CROP))
                tile = cv2.resize(im[y0:y0 + CROP, x0:x0 + CROP], (TILE, TILE))
                cv2.putText(tile, c, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            panel = base.copy()
            frac = (tt[i] - xlim[0]) / (xlim[1] - xlim[0])
            for bb in bbox:
                x = int(round((bb.x0 + frac * bb.width) * W))
                cv2.line(panel, (x, 0), (x, H - 1), (0, 0, 0), 1)
            top = np.hstack(tiles)
            frame = np.vstack([top, cv2.resize(panel, (top.shape[1], int(top.shape[1] * H / W)))])
            bar = np.zeros((40, frame.shape[1], 3), np.uint8)
            cv2.putText(bar, f"{animal} {date}  trial {tr}  {ix2.position.iloc[rows['cam2'][0]]}  t = {tt[i]:+.0f} ms",
                        (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bar, "ring = raw confident, dot = kept by cleaning | jaw green, tongue red | traces: gray raw, "
                        "color cleaned", (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            frame = np.vstack([bar, frame])
            if vw is None:
                vw = cv2.VideoWriter(str(f_out), cv2.VideoWriter_fourcc(*"mp4v"), 25, (frame.shape[1], frame.shape[0]))
            vw.write(frame)
        for _ in range(12):
            vw.write(frame)
    if vw is not None:
        vw.release()
    return f_out, stats


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--per-position", type=int, default=1)
    ap.add_argument("--t0", type=float, default=-100.0)
    ap.add_argument("--t1", type=float, default=700.0)
    a = ap.parse_args(argv)
    rv = PathResolver()
    allstats = []
    for s in a.sessions:
        f, st = run(rv, s, a.per_position, a.t0, a.t1)
        allstats += st
        print(f"-> {f}")
    D = pd.DataFrame(allstats)
    out = Path(rv.root("dlc")) / "session_poses" / "pose3d_side" / "side_clean_qc_stats.csv"
    D.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(D.to_string(index=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
