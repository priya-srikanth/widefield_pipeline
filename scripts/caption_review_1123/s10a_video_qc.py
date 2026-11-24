"""S10A onsets video QC (caption review 11-23, ccB helper): short clips of example trials with every first-movement
candidate marked, so Priya can judge whether real movement starts there ("sniff test").

Selection (`select`, seed `SEED`, from `s10a_onsets_analysis.py`'s per-trial table; licked canonical trials, latency
<= 1.5 s, both an FR onset and a 3-D jaw onset before the first lick): epoch {pre, acute} x FR - 3-D jaw onset
{FR leads by > 100 ms, about equal (|d| <= 40 ms), jaw leads by > 40 ms} x the four animals, alternating far / near
spouts; plus acute trials where FR sets the first movement > 150 ms before the 3-D jaw; plus QC-flag trials where the
in-mouth lick (cam1) sets the first movement and the 3-D jaw shows nothing (no onset, no >= 0.20 mm rise within
+-60 ms).

Clip: cue - 0.5 s .. first lick + 0.3 s, every 250-fps frame, written at 62.5 fps (0.25x), front camera (cam4), below
camera (cam1) and one side camera (cam2), synced on the DAQ clock through the camera_sync templates
(`camera_sync.frame_daq_seconds`: each camera's frame nearest to the cam4 frame's DAQ time); raw videos are only read.
Overlay: time from the cue on every frame, camera names; under the video a strip with the FR lick-vigor z trace (own
pre-cue baseline, threshold z = 3) and the 3-D jaw displacement (mm, onset floor 0.20 mm), ticks for cue, FR onset,
3-D jaw onset, 2-D jaw onset, in-mouth lick (cam1) events, and the first lick (fused four-camera train), and a cursor.
Contact sheet per clip: cam4 + cam1 frames every 40 ms from FR onset - 200 ms to + 200 ms.

    python s10a_video_qc.py <s10a_onsets_dir> [--only-select] [--jobs 2]
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SEED = 1123
MOUSE = {"PS92": "#9", "PS93": "#10", "PS94": "#11", "PS95": "#12"}
CAMS = ("cam4", "cam1", "cam2")
CAMNAME = {"cam4": "front (cam4)", "cam1": "below (cam1)", "cam2": "side (cam2)"}
TILE_H = 300
OUT_FPS = 62.5
COL = {"cue": (0, 220, 255), "fr": (60, 200, 60), "jaw3d": (255, 140, 0), "jaw2d": (200, 120, 255),
       "mouth": (80, 80, 255), "lick": (255, 255, 255)}   # BGR
LABEL = {"cue": "cue", "fr": "FR onset", "jaw3d": "3-D jaw onset", "jaw2d": "2-D jaw onset",
         "mouth": "in-mouth lick (cam1)", "lick": "first lick"}


def select(W: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    X = W[np.isfinite(W.fr_minus_jaw3d) & (W.lat_ms <= 1500) & W.epoch.isin(["pre", "acute"])].copy()
    X["cat"] = np.select([X.fr_minus_jaw3d < -100, X.fr_minus_jaw3d.abs() <= 40, X.fr_minus_jaw3d > 40],
                         ["FR leads > 100 ms", "about equal", "jaw leads"], "")
    X["far"] = X.pos.str.startswith("far")
    picks = []
    for e in ("pre", "acute"):
        for cat in ("FR leads > 100 ms", "about equal", "jaw leads"):
            for i, an in enumerate(sorted(X.animal.unique())):
                far = (i % 2 == 0) if cat != "about equal" else (i % 2 == 1)
                g = X[(X.epoch == e) & (X.cat == cat) & (X.animal == an)]
                g2 = g[g.far == far]
                g = g2 if len(g2) else g
                if len(g):
                    r = g.iloc[rng.integers(len(g))].copy()
                    r["why"] = f"{e} / {cat}"
                    picks.append(r)
    taken = {(r.label, r.trial_id) for r in picks}
    A = X[(X.epoch == "acute") & (X.src_j3_all == "fr") & (X.fr_minus_jaw3d < -150)]
    A = A[[(a, b) not in taken for a, b in zip(A.label, A.trial_id)]]
    for an in sorted(A.animal.unique())[:4]:
        g = A[A.animal == an]
        r = g.iloc[rng.integers(len(g))].copy()
        r["why"] = "acute / FR sets first movement > 150 ms before 3-D jaw"
        picks.append(r)
    Q = W[W.qc_mouth_no_jaw.fillna(False).astype(bool) & (W.lat_ms <= 1500)]
    for an in sorted(Q.animal.unique()):
        g = Q[Q.animal == an]
        r = g.iloc[rng.integers(len(g))].copy()
        r["why"] = "QC flag / in-mouth lick sets first movement, no 3-D jaw movement"
        r["cat"] = "QC in-mouth without jaw"
        picks.append(r)
    S = pd.DataFrame(picks).reset_index(drop=True)
    S.insert(0, "clip", [f"clip{k + 1:02d}" for k in range(len(S))])
    return S


def _txt(im, s, org, scale=0.45, col=(255, 255, 255), thick=1):
    import cv2
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def _frames(rv, an: str, date: str, cam: str, t_lo: float, t_hi: float):
    """(DAQ times, frame indices, video path) of ``cam``'s frames covering t_lo .. t_hi (DAQ s)."""
    from wfield_local import camera_sync as cs
    for rec, tps in cs.templates_for(rv, date, an, cam).items():
        for tp in tps:
            fd = cs.frame_daq_seconds(rv, an, date, tp)
            if fd[0] <= t_lo and fd[-1] >= t_hi:
                k = np.flatnonzero((fd >= t_lo - 0.01) & (fd <= t_hi + 0.01))
                return fd, k, Path(rv.root("behavior_cameras")) / date / an / f"{cam}_{rec}.avi"
    return None


def _read(path: Path, idx: np.ndarray) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    lo, hi = int(idx.min()), int(idx.max())
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    out = {}
    want = set(int(i) for i in idx)
    for i in range(lo, hi + 1):
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = im
    cap.release()
    return out


def render(item) -> dict:
    row, odir, cache = item
    import cv2
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    an = row["animal"]
    date = "2026" + row["label"].split("_")[1]
    cue, fl = float(row["cue_s"]), float(row["first_lick_s"])
    t_lo, t_hi = cue - 0.5, fl + 0.3
    base = _frames(rv, an, date, "cam4", t_lo, t_hi)
    if base is None:
        return {"clip": row["clip"], "error": "no cam4 frames"}
    fd4, k4, p4 = base
    master = fd4[k4]
    vids = {"cam4": (k4, _read(p4, k4))}
    for cam in CAMS[1:]:
        r = _frames(rv, an, date, cam, t_lo, t_hi)
        if r is None:
            continue
        fd, _k, p = r
        j = np.clip(np.searchsorted(fd, master), 1, len(fd) - 1)
        j = np.where(np.abs(fd[j - 1] - master) < np.abs(fd[j] - master), j - 1, j)
        vids[cam] = (j, _read(p, j))
    # strip data
    d = pickle.loads((Path(cache) / f"{row['label']}.pkl").read_bytes())
    Z = np.load(Path(odir).parent / "s10a_frz_ccB1123.npz")
    m = np.flatnonzero((Z["label"] == row["label"]) & (Z["trial_id"] == row["trial_id"]))
    zl, zt = Z["lags"] * 1000, Z["z_own"][m[0]] if len(m) else None
    J = d["jaw3d"]
    jm = J["trial_id"] == int(row["trial_id"]) if J is not None else None
    jt, jd = (J["t_daq_ms"][jm], J["d_um"][jm] / 1000.0) if J is not None else (np.array([]), np.array([]))
    C = d["cam1"]
    ev = []
    if C is not None:
        e = C[C.keep_new & ~C.fused_matched]
        ev = ((e.daq_on_s.to_numpy(float) - cue) * 1000)
        ev = ev[(ev >= -500) & (ev <= (fl - cue) * 1000 + 300)]
    marks = {"cue": 0.0, "fr": row["fr_ms"], "jaw3d": row["jaw3d_ms"], "jaw2d": row["jaw2d_ms"],
             "lick": row["lat_ms"]}
    # layout
    tiles = {}
    for cam, (_j, fr) in vids.items():
        im0 = next(iter(fr.values()))
        h, w = im0.shape[:2]
        tiles[cam] = (int(round(w * TILE_H / h)), TILE_H)
    W_ = sum(v[0] for v in tiles.values())
    SH = 170
    t0ms, t1ms = -500.0, (fl - cue) * 1000 + 300
    L, Rm = 60, 20

    def X(t):
        return int(L + (t - t0ms) / (t1ms - t0ms) * (W_ - L - Rm))

    strip = np.zeros((SH, W_, 3), np.uint8)
    # FR z panel (top half) and 3-D jaw (bottom half)
    def plot(y0, y1, t, v, lo, hi, thr, col, name):
        cv2.rectangle(strip, (L, y0), (W_ - Rm, y1), (60, 60, 60), 1)
        yy = lambda u: int(y1 - (np.clip(u, lo, hi) - lo) / (hi - lo) * (y1 - y0))  # noqa: E731
        cv2.line(strip, (L, yy(thr)), (W_ - Rm, yy(thr)), (110, 110, 110), 1, cv2.LINE_AA)
        ok = np.isfinite(v) & (t >= t0ms) & (t <= t1ms)
        pts = np.array([[X(a), yy(b)] for a, b in zip(t[ok], v[ok])], np.int32)
        if len(pts) > 1:
            cv2.polylines(strip, [pts], False, col, 1, cv2.LINE_AA)
        _txt(strip, name, (4, y0 + 12), 0.36, col)
        _txt(strip, f"{thr:g}", (L - 26, yy(thr) + 4), 0.33, (150, 150, 150))
    if zt is not None:
        plot(8, 70, zl, zt.astype(float), -2, 12, 3.0, COL["fr"], "FR z")
    if len(jt):
        plot(80, 142, jt, jd.astype(float), 0, 1.0, 0.2, COL["jaw3d"], "jaw mm")
    for k, v in marks.items():
        if np.isfinite(v):
            cv2.line(strip, (X(v), 4), (X(v), 146), COL[k], 2 if k in ("cue", "lick") else 1, cv2.LINE_AA)
    for e in ev:
        cv2.drawMarker(strip, (X(e), 152), COL["mouth"], cv2.MARKER_TRIANGLE_UP, 8, 2)
    # legend line
    xl = 4
    for k in ("cue", "fr", "jaw3d", "jaw2d", "mouth", "lick"):
        v = marks.get(k, np.nan)
        s = LABEL[k] + ("" if k == "mouth" else (f" {v:.0f}" if np.isfinite(v) else " -"))
        _txt(strip, s, (xl, 166), 0.33, COL[k])
        xl += int(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0][0]) + 14
    import imageio_ffmpeg
    f_out = Path(odir) / f"{row['clip']}_ccB1123.mp4"
    H = TILE_H + SH
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                             "-pix_fmt", "bgr24", "-s", f"{W_}x{H}", "-r", f"{OUT_FPS}", "-i", "-",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(f_out)],
                            stdin=subprocess.PIPE)
    frames_out = {}
    title = (f"mouse {MOUSE[an]}  {row['epoch']}  day {row['day']}  {row['pos'].replace('close', 'near')}  "
             f"| {row['why']}")
    for n, tm in enumerate(master):
        im = np.zeros((H, W_, 3), np.uint8)
        x = 0
        for cam in CAMS:
            if cam not in vids:
                continue
            j, fr = vids[cam]
            src = fr.get(int(j[n]) if cam != "cam4" else int(j[n]))
            w, h = tiles[cam]
            if src is not None:
                im[:TILE_H, x:x + w] = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
            x += w
        clean = im[:TILE_H, : tiles["cam4"][0] + tiles.get("cam1", (0, 0))[0]].copy()
        x = 0
        for cam in CAMS:
            if cam in vids:
                _txt(im, CAMNAME[cam], (x + 6, 18), 0.45)
                x += tiles[cam][0]
        tms = (tm - cue) * 1000
        _txt(im, f"t = {tms:+.0f} ms from cue  (0.25x)", (6, TILE_H - 10), 0.55, (0, 255, 255))
        _txt(im, title, (6, 38), 0.4, (220, 220, 220))
        s = strip.copy()
        cv2.line(s, (X(tms), 0), (X(tms), 158), (0, 255, 255), 1)
        im[TILE_H:] = s
        for k, v in marks.items():
            if np.isfinite(v) and abs(tms - v) <= 12:
                _txt(im, LABEL[k].upper(), (W_ // 2 - 60, TILE_H - 40), 0.7, COL[k], 2)
        if len(ev) and np.min(np.abs(ev - tms)) <= 12:
            _txt(im, "IN-MOUTH LICK (cam1)", (W_ // 2 - 60, TILE_H - 64), 0.6, COL["mouth"], 2)
        proc.stdin.write(im.tobytes())
        frames_out[n] = (tms, clean)
    proc.stdin.close()
    proc.wait()
    # contact sheet: every 40 ms around the FR onset (or the first movement)
    c0 = row["fr_ms"] if np.isfinite(row["fr_ms"]) else row["mv_j3_all"]
    tl = np.array([v[0] for v in frames_out.values()])
    want = [c0 + dt for dt in range(-200, 201, 40)]
    sel = [int(np.argmin(np.abs(tl - t))) for t in want]
    cells = []
    for i in sel:
        tms, im = frames_out[i]
        im = im.copy()
        tag = ", ".join(LABEL[k] for k, v in marks.items() if np.isfinite(v) and abs(tms - v) <= 20 and k != "cue")
        if len(ev) and np.min(np.abs(ev - tms)) <= 20:
            tag = (tag + ", " if tag else "") + LABEL["mouth"]
        _txt(im, f"{tms:+.0f} ms", (6, 22), 0.6, (0, 255, 255))
        if tag:
            _txt(im, tag, (6, im.shape[0] - 10), 0.5, (255, 255, 255))
        cells.append(im)
    rows_ = [np.hstack(cells[i:i + 3] + [np.zeros_like(cells[0])] * (3 - len(cells[i:i + 3])))
             for i in range(0, len(cells), 3)]
    sheet = np.vstack(rows_)
    head = np.zeros((30, sheet.shape[1], 3), np.uint8)
    _txt(head, title + f"  | front + below, every 40 ms around FR onset ({c0:.0f} ms)", (6, 20), 0.55)
    cv2.imwrite(str(Path(odir) / f"{row['clip']}_contact_ccB1123.png"), np.vstack([head, sheet]))
    return {"clip": row["clip"], "frames": len(master), "cams": list(vids), "video": f_out.name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("odir", type=Path)
    ap.add_argument("--only-select", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--clips", nargs="*")
    a = ap.parse_args()
    W = pd.read_csv(a.odir / "s10a_trials_ccB1123.csv.gz")
    vq = a.odir / "video_qc"
    vq.mkdir(exist_ok=True)
    f_sel = vq / "selection_ccB1123.csv"
    if f_sel.exists():
        S = pd.read_csv(f_sel)
    else:
        S = select(W)
        S.to_csv(f_sel, index=False, float_format="%.10g")
    print(S[["clip", "animal", "epoch", "pos", "why", "fr_minus_jaw3d"]].to_string())
    if a.only_select:
        return
    if a.clips:
        S = S[S["clip"].isin(a.clips)]
    from wfield_local import parallel
    items = [(r.to_dict(), str(vq), str(a.odir / "cache_ccB1123")) for _, r in S.iterrows()
             if not (vq / f"{r['clip']}_ccB1123.mp4").exists()]
    res, fails = parallel.fan_out(items, render, jobs=a.jobs, label="clip")
    for _it, r in sorted(res, key=lambda x: x[1]["clip"]):
        print(r)
    for it, e in fails:
        print("FAILED", it[0]["clip"], e)


if __name__ == "__main__":
    main()
