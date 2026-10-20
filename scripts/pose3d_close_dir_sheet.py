"""Video sheet for the 3-D jaw CLOSE_DIR readout (handoff 10-20 to-do 1a: look before interpreting).

    python -m scripts.pose3d_close_dir_sheet [ANIMAL:YYYYMMDD ...] [--animal PS92] [--per 3]

CLOSE_DIR = (close_L - close_R) / 2 of the per-trial jaw ``lr_wide_mm`` (`pose3d_kinematics`; + = toward the target side).
3-D says it goes from -0.14 (August pre) to -0.27 mm (acute / chronic) in PS92: the jaw moves AWAY from a close side spout,
more so after the stroke. Unvalidated -- cam4's own jaw readout leaves the close spouts out (the rod covers part of the chin).

Per session (default: the August-pre, acute and chronic session of ``--animal`` whose CLOSE_DIR is nearest its epoch's
median) and per close side spout: the ``--per`` trials whose ``lr_wide_mm`` is nearest that position's session median
(typical, not extreme). Each row = one trial: cam4 and cam1 at REST (pre-cue frame nearest the trial's resting jaw) and at
the jaw's LARGEST OPENING (`pose3d_kinematics.session_jaw`'s frame); the 2-D jaw keypoint (cleaned LP) as a dot, a line
through its resting position (cam4: vertical = sideways in the image; cam1: along the camera's mouse left-right axis).
Frames come straight from each camera's video frame (``src1`` / ``src4`` in the 3-D frames = template-time pairing).
Writes `session_poses/pose3d/close_dir_sheet/<session>.png` + `picks.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TILE = 260
CROP4, CROP1 = (320, 300), (320, 300)


def _close_dir(S: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    from scripts.pose3d_kinematics import session_jaw
    J = session_jaw(S)
    jm = J.groupby("position").lr_wide_mm.median()
    cd = (jm.get("close_L", np.nan) - jm.get("close_R", np.nan)) / 2
    return float(cd), J


def _open_time(f: pd.DataFrame) -> tuple[float, float]:
    """(rest t_ms, open t_ms) of one trial, as `session_jaw`: rest = pre-cue frame nearest the pre-cue median jaw."""
    from scripts.pose3d_kinematics import JAW_WIN_MS
    J = f[["jaw_ml", "jaw_ap", "jaw_dv"]].to_numpy(float)
    t = f.t_ms.to_numpy(float)
    stop = float(f.stop_ms.iloc[0]) if np.isfinite(f.stop_ms.iloc[0]) else JAW_WIN_MS[1]
    pre = (t < 0) & (t >= -500) & np.isfinite(J).all(1)
    win = (t >= JAW_WIN_MS[0]) & (t <= stop) & np.isfinite(J).all(1)
    r = np.median(J[pre], axis=0)
    ip = np.flatnonzero(pre)[np.argmin(np.linalg.norm(J[pre] - r, axis=1))]
    iw = np.flatnonzero(win)[np.argmax(np.linalg.norm(J[win] - r, axis=1))]
    return float(t[ip]), float(t[iw])


def _pick_sessions(rv, d3: Path, animal: str) -> list[str]:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    from wfield_local import pose3d as P3
    bad = P3.failed_sessions(d3)
    rows = []
    for f in sorted(d3.glob(f"{animal}_*_frames.csv.gz")):
        s = f.name.replace("_frames.csv.gz", "")
        date = s.split("_")[1]
        ep = epochs.epoch_of(f"{animal}_{date[4:]}")
        if s in bad or s not in ak.curated_ids() or ep not in ("pre", "acute", "chronic") or (ep == "pre" and date < "20260801"):
            continue
        cd, _ = _close_dir(pd.read_csv(f).drop_duplicates(["trial_id", "t_ms"]))
        rows.append({"session": s, "epoch": ep, "close_dir": cd})
    T = pd.DataFrame(rows).dropna()
    print(T.groupby("epoch").close_dir.describe().round(3).to_string())
    out = []
    for _ep, g in T.groupby("epoch"):
        out.append(g.session.iloc[int(np.argmin(np.abs(g.close_dir - g.close_dir.median())))])
    return out


def sheet(rv, d3: Path, s: str, per: int, out: Path) -> list[dict]:
    import cv2

    from scripts.cam1_licks import PROD, _cam1_frame, _video
    from wfield_local import epochs
    from wfield_local import o2_inference as oi
    from wfield_local import spout_frame as SF
    animal, date = s.split("_")
    S = pd.read_csv(d3 / f"{s}_frames.csv.gz").drop_duplicates(["trial_id", "t_ms"])
    cd, J = _close_dir(S)
    c1 = oi.session_windows(rv, PROD["cam1"], animal, date, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], animal, date, "cam4")
    ix1, fl1 = c1[0].reset_index(drop=True), c1[1].reset_index(drop=True)
    ix4, fl4 = c4[0].reset_index(drop=True), c4[1].reset_index(drop=True)
    fr1, _, _ = _cam1_frame(fl1, ix1)
    o4 = SF.from_medians(SF.ruler_medians(fl4, ix4)).origin
    cams = {"cam4": (ix4, fl4, np.asarray(o4, float) + (0.0, 40.0), CROP4, np.array([0.0, 1.0])),
            "cam1": (ix1, fl1, np.asarray(fr1.origin, float) - 10.0 * np.asarray(fr1.ap_axis), CROP1,
                     np.asarray(fr1.ap_axis, float))}
    rows, strips = [], []
    for pos in ("close_L", "close_R"):
        g = J[(J.position == pos) & J.lr_wide_mm.notna()]
        if g.empty:
            continue
        med = float(g.lr_wide_mm.median())
        for r in g.iloc[np.argsort(np.abs(g.lr_wide_mm.to_numpy() - med))[:per]].itertuples():
            f = S[S.trial_id == r.trial_id]
            t_rest, t_open = _open_time(f)
            tiles = []
            for cam in ("cam4", "cam1"):
                ix, fl, centre, crop, along = cams[cam]
                h = ix[ix.trial_id == r.trial_id]
                cap = cv2.VideoCapture(str(_video(rv, animal, date, cam, ix, r.trial_id)))
                shot = {}
                for lab, t in (("rest", t_rest), ("open", t_open)):
                    src = int(f.loc[(f.t_ms - t).abs().idxmin(), "src4" if cam == "cam4" else "src1"])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, src)
                    ok, im = cap.read()
                    if not ok:
                        im = np.zeros((1024, 1280, 3), np.uint8)
                    k = h.index[np.argmin(np.abs(h.src_frame.to_numpy() - src))]
                    q = fl.iloc[k]
                    shot[lab] = (im, np.array([q.jaw_x, q.jaw_y], float) if q.jaw_likelihood >= 0.4 else None)
                rest_xy = shot["rest"][1]
                seen = [xy for _, xy in shot.values() if xy is not None]
                if seen:                                       # both jaw positions in view (the open jaw drops ~100 px)
                    centre = np.mean(seen, axis=0)
                for lab, t in (("rest", t_rest), ("open", t_open)):
                    im, xy = shot[lab]
                    if rest_xy is not None:                    # line through the resting jaw, along the camera's ap axis
                        p0, p1 = (rest_xy - 60 * along).astype(int), (rest_xy + 60 * along).astype(int)
                        cv2.line(im, tuple(p0), tuple(p1), (0, 255, 255), 1, cv2.LINE_AA)
                    if xy is not None:
                        cv2.circle(im, tuple(xy.astype(int)), 5, (255, 0, 255), -1, cv2.LINE_AA)
                    cx, cy = centre.astype(int)
                    x0, y0 = max(cx - crop[0] // 2, 0), max(cy - crop[1] // 2, 0)
                    tile = cv2.resize(im[y0:y0 + crop[1], x0:x0 + crop[0]], (TILE, int(TILE * crop[1] / crop[0])))
                    tile = cv2.resize(tile, (TILE, int(TILE * CROP4[1] / CROP4[0])))     # equal heights
                    cv2.putText(tile, f"{cam} {lab} {t:+.0f} ms", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(tile)
                cap.release()
            band = np.full((22, TILE * 4, 3), 30, np.uint8)
            cv2.putText(band, f"{pos}  trial {r.trial_id}  lr_wide {r.lr_wide_mm:+.2f} mm (+ = mouse left)  open "
                        f"{r.open_max_mm:.2f} mm  [{pos} median {med:+.2f}]", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            strips.append(np.vstack([band, np.hstack(tiles)]))
            rows.append({"session": s, "position": pos, "trial_id": r.trial_id, "lr_wide_mm": r.lr_wide_mm,
                         "pos_median_mm": med, "open_max_mm": r.open_max_mm, "t_rest_ms": t_rest, "t_open_ms": t_open})
    if strips:
        head = np.full((26, TILE * 4, 3), 0, np.uint8)
        cv2.putText(head, f"{s} ({epochs.epoch_of(f'{animal}_{date[4:]}')})  CLOSE_DIR {cd:+.2f} mm  (+ = toward the "
                    f"target side)   dot = jaw, line = resting jaw", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{s}.png"), np.vstack([head] + strips))
        print(f"-> {out / f'{s}.png'}  (CLOSE_DIR {cd:+.3f} mm)")
    return rows


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--animal", default="PS92")
    ap.add_argument("--per", type=int, default=3, help="trials per close side spout")
    a = ap.parse_args(argv)
    rv = PathResolver()
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    out = d3 / "close_dir_sheet"
    out.mkdir(exist_ok=True)
    sess = [x.replace(":", "_") for x in a.sessions] or _pick_sessions(rv, d3, a.animal)
    rows = []
    for s in sess:
        rows += sheet(rv, d3, s, a.per, out)
    pd.DataFrame(rows).to_csv(out / "picks.csv", index=False)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
