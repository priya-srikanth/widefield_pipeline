"""Labelled video of the CLEANED tongue and jaw (orofacial_clean) on a few trials of a `session_poses` clip, to judge
the cleaning by eye -- Priya, 2026-10-15: "see if filling in mouth zero makes sense or if we should leave that blank
too; label the jaw cleaned trace without baseline fill".

    PYTHONPATH=. python -m scripts.cleaned_label_clips <session_poses dir> --model <csv> --name "LP r4" \
        --trials 508 12 47 125 87 515 [--fps 50]

Markers (image coordinates):
  tongue  filled orange dot = measured; hollow orange ring = interpolated (gap <= 56 ms); grey X at the MOUTH = the
          baseline fill ("tongue in the mouth"), drawn so its placement can be judged; nothing = no value
  jaw     filled cyan dot = measured; hollow cyan ring = interpolated; nothing = hidden (jaw.v34.hidden_fill: nan)
  raw     small white dot = the model's raw point where p >= 0.4 (tongue and jaw)
Writes <dir>/r4_eval/cleaned_labels_<name>.mp4 (h264; slowed: --fps of 250-fps video). Source clip READ-ONLY.
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--trials", type=int, nargs="+", required=True, help="trial ids")
    ap.add_argument("--win", type=float, nargs=2, default=(-0.5, 3.5), help="s around the cue")
    ap.add_argument("--fps", type=float, default=50.0)
    a = ap.parse_args(argv)
    d = a.poses_dir
    idx = pd.read_csv(d / "windows_index.csv")
    P = oc.read_pose(Path(a.model)).iloc[:len(idx)]
    idx = idx.iloc[:len(P)]
    # mouth = the spout-frame origin from round-3 DLC's spout, as every cleaned trace figure / the kinematics use
    ref = oc.read_pose(d / "windows_DLC.csv").iloc[:len(idx)]
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    mouth = SF.from_medians(SF.position_medians(ref[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(),
                                                spans)).origin
    flat = pd.DataFrame({f"{p}_{c}": P[f"{p}_{c}"].to_numpy(float) for p in ("tongue", "jaw")
                         for c in ("x", "y", "likelihood")})
    C = {}
    for part in ("tongue", "jaw"):
        c, pos, _ = oc.clean_windows(flat, idx, part, x0y0=tuple(mouth) if part == "tongue" else None)
        C[part] = (c.x_final[pos] + c.X0, c.y_final[pos] + c.Y0, c.fill_method[pos])
    rows = np.flatnonzero(idx.trial_id.isin(a.trials).to_numpy() & (idx.t_ms >= a.win[0] * 1000).to_numpy()
                          & (idx.t_ms <= a.win[1] * 1000).to_numpy())
    order = {t: i for i, t in enumerate(a.trials)}
    rows = sorted(rows, key=lambda r: (order[int(idx.trial_id.iloc[r])], r))
    cap = cv2.VideoCapture(str(d / "windows.mp4"))                       # READ-ONLY
    out = d / "r4_eval" / f"cleaned_labels_{a.name.replace(' ', '_')}.mp4"
    out.parent.mkdir(exist_ok=True)
    W, H, top = 420, 360, 44
    ff = subprocess.Popen([_ffmpeg(), "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W * 2}x{(H * 2) + top}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-crf",
                           "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    cx, cy = int(mouth[0]), int(mouth[1]) + 60
    x0, y0 = max(cx - W // 2, 0), max(cy - H // 2, 0)
    last = -2
    for r in rows:
        if r != last + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r))
        ok, im = cap.read()
        last = r
        if not ok:
            break
        for part in ("tongue", "jaw"):
            if P[f"{part}_likelihood"].iloc[r] >= 0.4:
                cv2.circle(im, (int(P[f"{part}_x"].iloc[r]), int(P[f"{part}_y"].iloc[r])), 2, (255, 255, 255), -1,
                           cv2.LINE_AA)
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
        crop = cv2.resize(im[y0:y0 + H, x0:x0 + W], (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)
        head = np.zeros((top, W * 2, 3), np.uint8)
        g = idx.iloc[r]
        tf = {oc.FILL_NONE: "measured", oc.FILL_BASELINE: "MOUTH FILL"}.get(int(C["tongue"][2][r]), "interpolated")
        jf = "hidden" if not np.isfinite(C["jaw"][1][r]) else {oc.FILL_NONE: "measured"}.get(int(C["jaw"][2][r]),
                                                                                              "interpolated")
        cv2.putText(head, f"{d.name}  {a.name}  trial {int(g.trial_id)} {g.position}  {g.t_ms:+.0f} ms   "
                    f"tongue: {tf}   jaw: {jf}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(head, "tongue: dot measured, ring interpolated, grey X mouth fill | jaw (cyan): dot measured, "
                    f"ring interpolated, none = hidden | white = raw p>=0.4 | {a.fps / 250:.1f}x speed", (8, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        ff.stdin.write(np.vstack([head, crop]).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{len(rows)} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
