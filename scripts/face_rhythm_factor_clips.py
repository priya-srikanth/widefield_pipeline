"""Video clips of each FaceRhythm factor at its strongest moments, for naming the components (fr env: decord, cv2).

    python scripts/face_rhythm_factor_clips.py --fdir <factor set> [--key PS92_20260806_cam4] [--comps 0 4]
                                               [--events 4] [--half-s 1.0] [--out DIR]

Per factor, the ``events`` strongest moments (``--window``: engaged-trial windows cue -2..+4 s by default; their
response part; between trials; large movement; anywhere) of its ABSOLUTE expression in one session (weights x FR normaliser D,
`face_rhythm_outputs.combined_series(absolute=True)`: 25 Hz in engaged trial windows, 5 Hz elsewhere), at least 10 s
apart, each +- ``half_s`` around the peak, every 2nd video frame at 50 fps (2.5x slow), one mp4 per factor:
  points  the TRACKED mesh (25 Hz `points` displacement + home), COLOUR = the direction the factor moves the point
          (Rich's convention: hue by the angle of its x / y spatial weights -- blue = lateral, red = vertical),
          SIZE / brightness = the point's weight in the factor x the factor's expression at that frame;
  strip   the factor's absolute expression over the clip with a cursor; the nearest cue's position and time from cue.
A 1 s title card per event names the factor (peak frequency; labels.json name if set). Video frame of 25 Hz bin j =
frames [10 j, 10 j + 10) (the runner's binning); DAQ <-> bin through the points output's ``t_daq_s``.
Output: ``<dlc>/face_rhythm/cam4/factor_clips/<set>/<key>_f<k>.mp4``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # the repo (wfield_local) from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))           # face_rhythm_factors

FPS_OUT, STRIDE, STRIP_H = 50, 2, 90


def _bundle_root(fdir: Path) -> Path:
    return fdir.parent.parent if (fdir.parent.parent / "tasks.tsv").exists() else \
        Path(r"M:\MICROSCOPE\Priya\DeepLabCut\Widefield\o2\fr_cam4_20261021")


def peaks(t: np.ndarray, a: np.ndarray, n: int, min_sep_s: float = 10.0, edge_s: float = 3.0) -> list[int]:
    """Indices of the ``n`` largest values of ``a``, at least ``min_sep_s`` apart and ``edge_s`` from the ends."""
    ok = np.isfinite(a) & (t > t[0] + edge_s) & (t < t[-1] - edge_s)
    out = []
    for i in np.argsort(-np.where(ok, a, -np.inf)):
        if not ok[i] or len(out) == n:
            break
        if all(abs(t[i] - t[j]) >= min_sep_s for j in out):
            out.append(int(i))
    return out


def point_colours(Wk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per point: BGR colour from the x / y weight angle (0 = lateral -> blue, 90 deg = vertical -> red) and the
    normalised weight magnitude (0..1)."""
    import cv2
    wx, wy = np.abs(Wk[0]), np.abs(Wk[1])
    ang = np.arctan2(wy, wx) / (np.pi / 2)                                   # 0 lateral .. 1 vertical
    hsv = np.stack([(120 * (1 - ang)).astype(np.uint8), np.full_like(wx, 255, np.uint8),
                    np.full_like(wx, 255, np.uint8)], 1)[None]
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0].astype(float)
    mag = np.hypot(wx, wy)
    return bgr, mag / max(mag.max(), 1e-12)


def main(argv=None) -> int:
    import cv2
    import decord
    import pandas as pd
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.paths import PathResolver
    from face_rhythm_factors import npz_memmap
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fdir", type=Path, required=True)
    ap.add_argument("--key", default=None, help="task key (default: the first projected pre-stroke task)")
    ap.add_argument("--comps", type=int, nargs="*", default=None)
    ap.add_argument("--events", type=int, default=4)
    ap.add_argument("--half-s", type=float, default=1.0)
    ap.add_argument("--window", default="trial", choices=("trial", "response", "between", "large", "any"),
                    help="where events may come from: engaged-trial windows cue -2..+4 s (trial, default -- the "
                         "trials output window), their response part 0..+4 s (response; for both: the "
                         "session-wide top of several factors is grooming), between trials outside large movement, "
                         "large movement (grooming + vigorous spontaneous licking: for labelling), anywhere")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    M = np.load(a.fdir / "master.npz")
    W, B, freqs, home = M["spatial"], M["spectral"], M["freqs"], M["home"].astype(float)
    R = W.shape[-1]
    names = fro.labels(a.fdir)
    proj = sorted((a.fdir / "proj").glob("*.npz"))
    key = a.key or next(p.stem for p in proj if str(np.load(p)["epoch"]) == "pre")
    bd = _bundle_root(a.fdir)
    T = pd.read_csv(bd / "tasks.tsv", sep="\t", dtype=str)
    r = T[T.key == key].iloc[0]
    vr = decord.VideoReader(str(Path(rv.root("behavior_cameras")) / r.date / r.animal / r.video))
    pf = bd / "out" / f"{key}_points.npz"
    disp = npz_memmap(pf, "disp")
    t25 = np.load(pf)["t_daq_s"]
    z = np.load(a.fdir / "proj" / f"{key}.npz")
    cues, cpos = z["cue_daq_s"], z["position"]
    t, X = fro.combined_series([a.fdir / "proj" / f"{key}.npz"], absolute=True)
    from wfield_local import face_rhythm_readouts as frr
    sc = np.sort(cues)
    k_ = np.clip(np.searchsorted(sc, t, side="right") - 1, 0, len(sc) - 1)
    k2 = np.clip(np.searchsorted(sc, t + 2.0, side="right") - 1, 0, len(sc) - 1)   # latest cue with cue - 2 <= t
    trial = (t - sc[k2] >= -2.0) & (t - sc[k2] < 4.0)
    resp = (t - sc[k_] >= 0) & (t - sc[k_] < 4.0)
    gm = np.interp(t, z["session_t_daq_s"], frr.large_movement(z["session_t_daq_s"], z["session_D"], cues)
                   .astype(float)) > 0.5
    allow = {"trial": trial, "response": resp, "between": ~trial & ~gm, "large": gm,
             "any": np.ones(len(t), bool)}[a.window]
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "factor_clips" / str(M["set"])
    out.mkdir(parents=True, exist_ok=True)
    H, Wd = vr[0].shape[:2]
    for k in (a.comps if a.comps is not None else range(R)):
        bgr, mag = point_colours(W[..., k])
        ev = peaks(t, np.where(allow, X[:, k], np.nan), a.events)
        top = np.nanmax(X[ev, k]) if ev else 1.0
        dst = out / f"{key}_f{k}_{a.window}.mp4"
        vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (Wd, H + STRIP_H))
        title = f"factor {k} ({names[k]}): peak {freqs[np.argmax(B[:, k])]:.1f} Hz  |  blue = lateral, red = vertical"
        for e_i, i in enumerate(ev):
            card = np.zeros((H + STRIP_H, Wd, 3), np.uint8)
            for li, txt in enumerate((title, f"{key}  event {e_i + 1}/{len(ev)}  t = {t[i]:.1f} s (DAQ)")):
                cv2.putText(card, txt, (10, 40 + 30 * li), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            for _ in range(FPS_OUT):
                vw.write(card)
            j0 = int(np.searchsorted(t25, t[i]))
            f_c = 10 * j0 + 5
            frames = list(range(max(f_c - int(a.half_s * 250), 0), min(f_c + int(a.half_s * 250), len(vr)), STRIDE))
            seg = (t >= t[i] - a.half_s) & (t <= t[i] + a.half_s)
            ts, xs = t[seg], X[seg, k]
            c = int(np.argmin(np.abs(cues - t[i])))
            for b0 in range(0, len(frames), 64):
                imgs = vr.get_batch(frames[b0:b0 + 64]).asnumpy()
                for fr_, img in zip(frames[b0:b0 + 64], imgs):
                    j = min(fr_ // 10, disp.shape[0] - 1)
                    tf = t25[j]
                    e_now = float(np.interp(tf, ts, xs)) / top if len(ts) > 1 else 1.0
                    P = home + np.nan_to_num(np.asarray(disp[j], float))
                    canvas = np.zeros((H + STRIP_H, Wd, 3), np.uint8)
                    canvas[:H] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    lvl = np.clip(mag * e_now, 0, 1)
                    for p in np.argsort(lvl):
                        if lvl[p] < 0.05:
                            cv2.circle(canvas, (int(P[p, 0]), int(P[p, 1])), 1, (90, 90, 90), -1)
                        else:
                            col = tuple(int(v) for v in bgr[p] * (0.35 + 0.65 * lvl[p]))
                            cv2.circle(canvas, (int(P[p, 0]), int(P[p, 1])), int(1 + 4 * lvl[p]), col, -1)
                    if len(ts) > 1:                                       # expression strip with a cursor
                        xx = ((ts - ts[0]) / max(ts[-1] - ts[0], 1e-9) * (Wd - 20) + 10).astype(int)
                        yy = (H + STRIP_H - 10 - xs / top * (STRIP_H - 30)).astype(int)
                        cv2.polylines(canvas, [np.column_stack([xx, yy]).astype(np.int32)], False, (0, 220, 255), 1)
                        cx = int((tf - ts[0]) / max(ts[-1] - ts[0], 1e-9) * (Wd - 20) + 10)
                        cv2.line(canvas, (cx, H + 5), (cx, H + STRIP_H - 5), (255, 255, 255), 1)
                    cv2.putText(canvas, f"f{k} {names[k]}  |  {tf - cues[c]:+.2f} s from cue ({cpos[c]})", (10, H + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    vw.write(canvas)
        vw.release()
        print(f"factor {k}: {len(ev)} events -> {dst}")
    (out / "README.txt").write_text(json.dumps({"set": str(M["set"]), "key": key, "colour": "hue = x/y weight angle "
                                                "(blue lateral, red vertical); size = weight x expression"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
