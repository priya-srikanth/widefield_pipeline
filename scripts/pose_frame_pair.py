"""Labeled cam1 + cam4 still pair at ONE timepoint (K08 Figure 2C request, 11-09): full native frames, no crop, with the
production pose model's four parts (tongue, jaw, spout, nose) marked and a small legend.

Source = the slide-29 machinery (`pose3d_figures.cmd_video`): production whole-session LP outputs on the share
(``o2/lp_r4_prod_20261015`` cam4, ``o2/lp_cam1_ctx_prod_20261019`` cam1; per frame ``pose`` (frames, 4, 3) = x, y,
likelihood for nose / jaw / tongue / spout, RAW image px), cut into trial windows on the DAQ clock by each camera's
alignment template, and paired across cameras on camera time (nearest within 2 ms, `pose3d.session_3d`: columns
src1 / src4 = the video frame index of each camera at that moment). Candidates = licks at the requested spout
(`pose3d.lick_peaks`), frame = the lick PEAK (tongue furthest out), smallest 3-D miss to the spout first, kept only
when all four parts are tracked (likelihood >= --min-lik) in BOTH cameras. The points are the model's 2-D predictions
drawn on the raw video (June sessions: cam4 LP coords divided by 0.8853 first, memory june-lp-coords-rescaled).

    dlc env:  python scripts/pose_frame_pair.py PS92:20260807 [--position far_R] [--n 4] [--out <dir>]
-> <out>/<animal>_<date>_<position>_t<trial>_{cam1,cam4}.png (+ _raw twins without marks) and candidates.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PARTS = ("tongue", "jaw", "spout", "nose")
BGR = {"tongue": (60, 60, 230), "jaw": (230, 160, 30), "spout": (40, 200, 240), "nose": (90, 200, 60)}
DEFAULT_OUT = "M:/MICROSCOPE/Priya/Claude outputs/k08_camframes"


def legend(im, scale: float) -> None:
    import cv2
    pad, row = int(10 * scale), int(26 * scale)
    w, h = int(150 * scale), row * len(PARTS) + pad
    x0, y0 = im.shape[1] - w - pad, pad
    sub = im[y0:y0 + h, x0:x0 + w]
    im[y0:y0 + h, x0:x0 + w] = (sub * 0.35).astype(im.dtype)
    for k, p in enumerate(PARTS):
        cy = y0 + pad + k * row + row // 2 - pad // 2
        cv2.circle(im, (x0 + pad + int(8 * scale), cy), int(7 * scale), BGR[p], -1, cv2.LINE_AA)
        cv2.putText(im, p, (x0 + pad + int(24 * scale), cy + int(6 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale,
                    (255, 255, 255), max(1, int(1.5 * scale)), cv2.LINE_AA)


def mark(im, row: pd.Series, scale: float, june: float | None, pp) -> None:
    import cv2
    for p in PARTS:
        x, y = float(row[f"{p}_x"]), float(row[f"{p}_y"])
        if june:                                             # June cam4 LP coords are x0.8853 about the principal point
            x, y = pp[0] + (x - pp[0]) / june, pp[1] + (y - pp[1]) / june
        c = (int(round(x)), int(round(y)))
        cv2.circle(im, c, int(9 * scale), (0, 0, 0), int(4 * scale), cv2.LINE_AA)
        cv2.circle(im, c, int(9 * scale), BGR[p], int(2.5 * scale), cv2.LINE_AA)
        cv2.circle(im, c, max(1, int(2 * scale)), BGR[p], -1, cv2.LINE_AA)
        org = (c[0] + int(12 * scale), c[1] - int(10 * scale))
        cv2.putText(im, p, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), int(4 * scale), cv2.LINE_AA)
        cv2.putText(im, p, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, BGR[p], max(1, int(1.5 * scale)), cv2.LINE_AA)


def main(argv=None) -> int:
    import cv2

    from scripts.apparatus_mouth import _rig
    from scripts.pose3d import PROD, _cgroup, _june_factor
    from scripts.pose3d_figures import _frame
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d as P3
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--position", default="far_R")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--min-lik", type=float, default=0.9)
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    a = ap.parse_args(argv)
    rv = PathResolver()
    animal, date = a.session.split(":")
    writeguard.assert_writable(a.out / "_")
    a.out.mkdir(parents=True, exist_ok=True)
    c1 = oi.session_windows(rv, PROD["cam1"], animal, date, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], animal, date, "cam4")
    if c1 is None or c4 is None:
        raise SystemExit(f"{a.session}: production outputs missing")
    rig = _rig(rv, animal, date)
    cg = _cgroup(rv).subset_cameras_names(["cam1", "cam4"])
    jf = _june_factor(date)
    S, info = P3.session_3d(cg, rig, c1[0], c1[1], c4[0], c4[1], june_factor=jf)
    pp4 = info["cg"].cameras[1].get_camera_matrix()[:2, 2]
    L = P3.lick_peaks(S, rig)
    L = L[L.position == a.position].sort_values("miss_mm")
    raw = {"cam1": c1[1], "cam4": c4[1]}
    ridx = {}
    for cam, ix in (("cam1", c1[0]), ("cam4", c4[0])):
        r_ = pd.Series(np.arange(len(ix)), index=ix.src_frame.to_numpy())
        ridx[cam] = r_[~r_.index.duplicated()]
    caps = {c: cv2.VideoCapture(str(oi.videos_for(rv, animal, date, c)[0])) for c in ("cam1", "cam4")}
    rows, used = [], set()
    for r in L.itertuples():
        if len(rows) >= a.n or r.trial_id in used:
            continue
        F = S[(S.trial_id == r.trial_id) & (S.t_ms == r.t_peak_ms)]
        if F.empty:
            continue
        fr = F.iloc[0]
        src = {"cam1": int(fr.src1), "cam4": int(fr.src4)}
        if any(src[c] not in ridx[c].index for c in src):
            continue
        prow = {c: raw[c].iloc[int(ridx[c][src[c]])] for c in src}
        if min(float(prow[c][f"{p}_likelihood"]) for c in src for p in PARTS) < a.min_lik:
            continue
        used.add(r.trial_id)
        stem = f"{animal}_{date}_{a.position}_t{r.trial_id}"
        for cam in ("cam1", "cam4"):
            im = _frame(caps[cam], src[cam])
            cv2.imwrite(str(a.out / f"{stem}_{cam}_raw.png"), im)
            scale = max(im.shape[:2]) / 900
            mark(im, prow[cam], scale, jf if cam == "cam4" else None, pp4)
            legend(im, scale)
            cv2.imwrite(str(a.out / f"{stem}_{cam}.png"), im)
        rows.append(dict(stem=stem, trial_id=r.trial_id, position=a.position, t_peak_ms=r.t_peak_ms,
                         miss_mm=r.miss_mm, cam1_frame=src["cam1"], cam4_frame=src["cam4"],
                         **{f"{c}_{p}_lik": float(prow[c][f"{p}_likelihood"]) for c in src for p in PARTS}))
        print(stem, f"miss {r.miss_mm:.2f} mm  cam1 frame {src['cam1']}  cam4 frame {src['cam4']}")
    pd.DataFrame(rows).to_csv(a.out / f"{animal}_{date}_{a.position}_candidates.csv", index=False, float_format="%.4g")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
