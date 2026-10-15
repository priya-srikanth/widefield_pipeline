"""Labelled video of the CLEANED tongue and jaw (orofacial_clean) on a few trials of a `session_poses` clip, to judge
the cleaning -- and, with several ``--model``s, the models -- by eye. Priya, 2026-10-15: "see if filling in mouth zero
makes sense"; "comparison slowed-down clips of the top models with cleaner data".

    PYTHONPATH=. python -m scripts.cleaned_label_clips <session_poses dir> --model "LP r4=<csv>" [--model ...] \
        --trials 508 12 47 125 87 515 [--fps 50] [--tag top6]

One panel per model (3 per row), the SAME frame in every panel. Markers (image coordinates):
  tongue  filled orange dot = measured; hollow orange ring = interpolated (gap <= 56 ms); grey X at the MOUTH = the
          mouth-zero fill (tongue in the mouth); nothing = no value
  jaw     filled cyan dot = measured; hollow cyan ring = interpolated; nothing = hidden (jaw.v34.hidden_fill: nan)
  raw     small white dot = the model's raw point where p >= 0.4 (tongue and jaw)
Writes <dir>/r4_eval/cleaned_labels_<tag or model>.mp4 (h264; slowed: --fps of 250-fps video). Source clip READ-ONLY.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    import cv2

    from scripts.session_poses import _ffmpeg
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poses_dir", type=Path)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CSV")
    ap.add_argument("--trials", type=int, nargs="+", required=True, help="trial ids")
    ap.add_argument("--win", type=float, nargs=2, default=(-0.5, 3.5), help="s around the cue")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    d = a.poses_dir
    idx = pd.read_csv(d / "windows_index.csv")
    # mouth = the spout-frame origin from round-3 DLC's spout, as every cleaned trace figure / the kinematics use
    ref = oc.read_pose(d / "windows_DLC.csv").iloc[:len(idx)]
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0]))
             for _, g in idx.iloc[:len(ref)].groupby("trial_k")]
    mouth = SF.from_medians(SF.position_medians(ref[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(),
                                                spans)).origin
    models = []
    n = len(idx)
    for spec in a.model:
        name, csv = spec.split("=", 1)
        P = oc.read_pose(Path(csv)).iloc[:len(idx)]
        n = min(n, len(P))
        models.append((name, P))
    idx = idx.iloc[:n]
    M = []
    for name, P in models:
        P = P.iloc[:n]
        flat = pd.DataFrame({f"{p}_{c}": P[f"{p}_{c}"].to_numpy(float) for p in ("tongue", "jaw")
                             for c in ("x", "y", "likelihood")})
        C = {}
        for part in ("tongue", "jaw"):
            c, pos, _ = oc.clean_windows(flat, idx, part, x0y0=tuple(mouth) if part == "tongue" else None)
            C[part] = (c.x_final[pos] + c.X0, c.y_final[pos] + c.Y0, c.fill_method[pos])
        M.append((name, P, C))
    rows = np.flatnonzero(idx.trial_id.isin(a.trials).to_numpy() & (idx.t_ms >= a.win[0] * 1000).to_numpy()
                          & (idx.t_ms <= a.win[1] * 1000).to_numpy())
    order = {t: i for i, t in enumerate(a.trials)}
    rows = sorted(rows, key=lambda r: (order[int(idx.trial_id.iloc[r])], r))
    cap = cv2.VideoCapture(str(d / "windows.mp4"))                       # READ-ONLY
    tag = a.tag or M[0][0].replace(" ", "_")
    out = d / "r4_eval" / f"cleaned_labels_{tag}.mp4"
    out.parent.mkdir(exist_ok=True)
    one = len(M) == 1
    W, H = (420, 360) if one else (300, 270)
    S = 2.0 if one else 1.5
    pw, ph = int(W * S), int(H * S) + 22
    ncol = 1 if one else min(3, len(M))
    nrow = int(np.ceil(len(M) / ncol))
    top = 44
    FW, FH = pw * ncol, ph * nrow + top
    FW += FW % 2
    FH += FH % 2
    ff = subprocess.Popen([_ffmpeg(), "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{FW}x{FH}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-crf", "18",
                           "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    cx, cy = int(mouth[0]), int(mouth[1]) + 60
    x0, y0 = max(cx - W // 2, 0), max(cy - H // 2, 0)
    last = -2
    for r in rows:
        if r != last + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r))
        ok, base = cap.read()
        last = r
        if not ok:
            break
        frame = np.zeros((FH, FW, 3), np.uint8)
        for k, (name, P, C) in enumerate(M):
            im = base.copy()
            for part in ("tongue", "jaw"):
                if P[f"{part}_likelihood"].iloc[r] >= 0.4:
                    cv2.circle(im, (int(P[f"{part}_x"].iloc[r]), int(P[f"{part}_y"].iloc[r])), 2, (255, 255, 255),
                               -1, cv2.LINE_AA)
            for part, col in (("tongue", (0, 165, 255)), ("jaw", (255, 255, 0))):
                x, y, fm = C[part][0][r], C[part][1][r], int(C[part][2][r])
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                pt = (int(round(x)), int(round(y)))
                if fm == oc.FILL_NONE:
                    cv2.circle(im, pt, 5, col, -1, cv2.LINE_AA)
                elif fm == oc.FILL_BASELINE:
                    cv2.drawMarker(im, pt, (170, 170, 170), cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
                else:
                    cv2.circle(im, pt, 6, col, 2, cv2.LINE_AA)
            crop = cv2.resize(im[y0:y0 + H, x0:x0 + W], (pw, int(H * S)), interpolation=cv2.INTER_CUBIC)
            lab = np.zeros((22, pw, 3), np.uint8)
            tf = {oc.FILL_NONE: "meas", oc.FILL_BASELINE: "mouth"}.get(int(C["tongue"][2][r]), "interp")
            jf = "hidden" if not np.isfinite(C["jaw"][1][r]) else {oc.FILL_NONE: "meas"}.get(int(C["jaw"][2][r]),
                                                                                             "interp")
            cv2.putText(lab, f"{name}   tongue {tf} | jaw {jf}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            rr, cc = divmod(k, ncol)
            frame[top + rr * ph: top + (rr + 1) * ph, cc * pw:(cc + 1) * pw] = np.vstack([lab, crop])
        g = idx.iloc[r]
        cv2.putText(frame, f"{d.name}  trial {int(g.trial_id)} {g.position}  {g.t_ms:+.0f} ms from cue  "
                    f"({a.fps / 250:.1f}x speed)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, "CLEANED: tongue orange (dot measured, ring interpolated, grey X = mouth fill) | jaw cyan "
                    "(dot measured, ring interpolated, none = hidden) | white = raw p>=0.4", (8, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{len(rows)} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
