"""Blind review of the cam1 "jaw moved" call (encoding window via Priya 10-20 late: "blind-review the cam1 jaw-moved call
on acute no_try / jaw_only trials (far contra, close center); the 4 px floor is cam4's").

    python -m scripts.cam1_jaw_review [--per 12] [--seed 0]

Trials: ACUTE sessions with cam1 (PS92 / PS93), positions far_R (far contra) and close_center, NO lick (no cam4 post-cue
lick, no non-droplet cam1 event) -- the trials `trial_attempts` classes no_try / jaw_only. Stratified by the cam1 call
(`scripts/jaw_onsets.py` cam1_moved 1 vs 0, the same rule as `trial_attempts.cam1_jaw_moved`), ``--per`` per stratum x
position, shuffled; the call is NOT shown (key.csv holds it).

Each candidate: two rows of tiles -- cam1 (top) and cam4 (bottom, the frame nearest in DAQ time) -- at REST (pre-cue
frame nearest the baseline median) and at the cam1 jaw's LARGEST deflection in cue .. stop, plus -40 / +40 ms around
it. The cam1 jaw keypoint (likelihood >= 0.6) is ringed. Output `session_poses/cam1_licks/jaw_review/{candidates.json,
key.csv}`; the review page is built from candidates.json.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import pose_windows

V36 = "lp_r4_heatmap_20261013__clean_v36"
POS = ("far_R", "close_center")
CUT = 0.6
OFFSETS = (-40, 0, 40)
CROP, TILE = (300, 260), 230


def _cands(rv, per: int, rng) -> pd.DataFrame:
    root = Path(rv.root("orofacial")) / V36
    J = pd.read_csv(root / "imaging_encoding" / "qc" / "jaw_onsets.csv", dtype={"date": str})
    J = J[(J.epoch == "acute") & J.position.isin(POS) & J.cam1_moved.notna()]
    keep = []
    d1 = Path(rv.root("dlc")) / "session_poses" / "cam1_licks"
    for s, g in J.groupby("session"):
        a, d = s.split("_")
        L = pd.read_csv(root / "sessions" / a / d / "licks.csv", usecols=["trial_id"])
        E = pd.read_csv(d1 / f"{s}_events.csv", usecols=["trial_id", "droplet"])
        lick = set(L.trial_id) | set(E.loc[~E.droplet.astype(bool), "trial_id"])
        keep.append(g[~g.trial_id.isin(lick)])
    C = pd.concat(keep, ignore_index=True)
    picks = []
    for _key, g in C.groupby(["position", "cam1_moved"]):
        picks.append(g.sample(min(per, len(g)), random_state=int(rng.integers(1 << 30))))
    P = pd.concat(picks, ignore_index=True)
    print(C.groupby(["position", "cam1_moved"]).size().rename("available").to_string())
    return P


def _tile(cap, src, xy, label):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(src))
    ok, im = cap.read()
    if not ok:
        im = np.zeros((1024, 1280, 3), np.uint8)
    return im


def build(rv, P: pd.DataFrame, rng) -> tuple[list, list]:
    import cv2

    from scripts.cam1_licks import _video
    cands, keys = [], []
    for s, g in P.groupby("session"):
        a, d = s.split("_")
        d1 = Path(rv.root("dlc")) / "session_poses" / f"{s}_cam1_full"
        ix1 = pose_windows.index(d1)
        W = pose_windows.lp(d1)
        jx, jy = W[("jaw", "x")].to_numpy(float), W[("jaw", "y")].to_numpy(float)
        lk = W[("jaw", "likelihood")].to_numpy(float)
        y = np.where(lk >= CUT, jy, np.nan)
        w4 = np.load(Path(rv.root("orofacial")) / V36 / "sessions" / a / d / "windows.npz")
        ix4 = pd.DataFrame({k: w4[f"ix_{k}"] for k in ("trial_id", "t_daq_ms", "src_frame")})
        pj = list(map(str, w4["parts"])).index("jaw")
        jaw4 = w4["pose"][:, pj, :2].astype(float)                 # cam4 jaw px per window row (crop centre)
        cap1 = cv2.VideoCapture(str(_video(rv, a, d, "cam1", pd.DataFrame(), 0)))
        cap4 = cv2.VideoCapture(str(_video(rv, a, d, "cam4", pd.DataFrame(), 0)))
        for r in g.itertuples():
            h = ix1[ix1.trial_id == r.trial_id]
            t, k = h.t_daq_ms.to_numpy(float), h.index.to_numpy()
            yy = y[k]
            pre = (t >= -500) & (t < 0) & np.isfinite(yy)
            if pre.sum() < 2:
                continue
            base = np.nanmedian(yy[pre])
            stop = float(h.stop_ms.iloc[0]) if np.isfinite(h.stop_ms.iloc[0]) else 3500.0
            win = (t >= 0) & (t <= stop) & np.isfinite(yy)
            if not win.any():
                continue
            i_pk = np.flatnonzero(win)[int(np.argmax(np.abs(yy[win] - base)))]
            i_rest = np.flatnonzero(pre)[int(np.argmin(np.abs(yy[pre] - base)))]
            h4 = ix4[ix4.trial_id == r.trial_id]
            rows = []
            for cam, cap in (("cam1", cap1), ("cam4", cap4)):
                tiles = []
                for lab, i0, off in [("rest", i_rest, 0)] + [(f"{o:+d} ms", i_pk, o) for o in OFFSETS]:
                    tq = t[i0] + off
                    if cam == "cam1":
                        j = int(np.argmin(np.abs(t - tq)))
                        src, kk = int(h.src_frame.iloc[j]), k[j]
                        xy = (jx[kk], jy[kk]) if lk[kk] >= CUT else None
                    else:
                        j = int(np.argmin(np.abs(h4.t_daq_ms.to_numpy(float) - tq)))
                        src, xy = int(h4.src_frame.iloc[j]), None
                    im = _tile(cap, src, xy, lab)
                    if cam == "cam1":
                        c = np.array([np.nanmedian(jx[k][pre]), np.nanmedian(jy[k][pre])])
                        if xy is not None:
                            cv2.circle(im, (int(xy[0]), int(xy[1])), 10, (255, 0, 255), 1, cv2.LINE_AA)
                    else:
                        pre4 = h4.index[(h4.t_daq_ms.to_numpy(float) < 0) & (h4.t_daq_ms.to_numpy(float) >= -500)]
                        c = np.nanmedian(jaw4[pre4], axis=0) if len(pre4) else None
                    if c is None or not np.isfinite(c).all():
                        c = np.array([im.shape[1] / 2, im.shape[0] / 2])
                    x0, y0 = max(int(c[0]) - CROP[0] // 2, 0), max(int(c[1]) - CROP[1] // 2, 0)
                    tl = cv2.resize(im[y0:y0 + CROP[1], x0:x0 + CROP[0]], (TILE, int(TILE * CROP[1] / CROP[0])))
                    cv2.putText(tl, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(tl)
                rows.append(np.hstack(tiles))
            ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
            cid = f"j{len(cands):03d}"
            cands.append({"id": cid, "position": r.position, "img": base64.b64encode(buf.tobytes()).decode("ascii")})
            keys.append({"id": cid, "session": s, "trial_id": int(r.trial_id), "position": r.position,
                         "cam1_moved": r.cam1_moved, "cam4_moved": r.cam4_moved, "cam1_onset_ms": r.cam1_onset_ms,
                         "peak_t_ms": round(float(t[i_pk]), 1), "peak_defl_px": round(float(yy[i_pk] - base), 2)})
        cap1.release()
        cap4.release()
    order = rng.permutation(len(cands))
    return [cands[i] for i in order], [keys[i] for i in order]


#: VIDEO round (Priya 10-21: "jaw movements can be subtle so ... the frame review [is not] good ground truth (vs.
#: video)"): clips across the cam1 deflection range where the frame review was ambiguous, to pin the cam1 floor.
SIDE = ("far_R", "far_L", "close_L", "close_R")       # cam1 sees the jaw (not the centre spouts: rod end-on)
BANDS = ((8, 12), (12, 16), (16, 20), (20, 25))
CLIP_MS, CLIP_STEP = (-200, 400), 3                   # every 3rd cam1 frame (~12 ms)
VTILE, VCROP = 200, (300, 260)


def _peaks(rv, s: str, trials: set) -> pd.DataFrame:
    """Per trial: cam1 jaw |peak deflection| from the pre-cue median in cue .. stop (px) and its time (as `build`)."""
    d1 = Path(rv.root("dlc")) / "session_poses" / f"{s}_cam1_full"
    ix1 = pose_windows.index(d1)
    W = pose_windows.lp(d1)
    y = np.where(W[("jaw", "likelihood")].to_numpy(float) >= CUT, W[("jaw", "y")].to_numpy(float), np.nan)
    rows = []
    for tid, h in ix1[ix1.trial_id.isin(trials)].groupby("trial_id"):
        t, yy = h.t_daq_ms.to_numpy(float), y[h.index.to_numpy()]
        pre = (t >= -500) & (t < 0) & np.isfinite(yy)
        stop = float(h.stop_ms.iloc[0]) if np.isfinite(h.stop_ms.iloc[0]) else 3500.0
        win = (t >= 0) & (t <= stop) & np.isfinite(yy)
        if pre.sum() < 2 or not win.any():
            continue
        base = np.nanmedian(yy[pre])
        i = np.flatnonzero(win)[int(np.argmax(np.abs(yy[win] - base)))]
        rows.append({"session": s, "trial_id": int(tid), "defl_px": float(abs(yy[i] - base)), "peak_t_ms": float(t[i])})
    return pd.DataFrame(rows)


def video_round(rv, per: int, rng) -> tuple[list, list]:
    import cv2

    from scripts.cam1_licks import _video
    root = Path(rv.root("orofacial")) / V36
    J = pd.read_csv(root / "imaging_encoding" / "qc" / "jaw_onsets.csv", dtype={"date": str})
    J = J[(J.epoch == "acute") & J.position.isin(SIDE) & J.cam1_moved.notna()]
    pool = []
    d1c = Path(rv.root("dlc")) / "session_poses" / "cam1_licks"
    for s, g in J.groupby("session"):
        a, d = s.split("_")
        L = pd.read_csv(root / "sessions" / a / d / "licks.csv", usecols=["trial_id"])
        E = pd.read_csv(d1c / f"{s}_events.csv", usecols=["trial_id", "droplet"])
        lick = set(L.trial_id) | set(E.loc[~E.droplet.astype(bool), "trial_id"])
        nl = g[~g.trial_id.isin(lick)]
        if len(nl):
            pk = _peaks(rv, s, set(nl.trial_id))
            pool.append(pk.merge(nl[["trial_id", "position", "cam1_moved", "cam4_moved"]], on="trial_id"))
    Pool = pd.concat(pool, ignore_index=True)
    picks = []
    for lo, hi in BANDS:
        b = Pool[(Pool.defl_px >= lo) & (Pool.defl_px < hi)]
        print(f"band {lo}-{hi} px: {len(b)} no-lick trials")
        picks.append(b.sample(min(per, len(b)), random_state=int(rng.integers(1 << 30))).assign(band=f"{lo}-{hi}"))
    P = pd.concat(picks, ignore_index=True)
    cands, keys = [], []
    for s, g in P.groupby("session"):
        a, d = s.split("_")
        d1 = Path(rv.root("dlc")) / "session_poses" / f"{s}_cam1_full"
        ix1 = pose_windows.index(d1)
        w4 = np.load(root / "sessions" / a / d / "windows.npz")
        ix4 = pd.DataFrame({k: w4[f"ix_{k}"] for k in ("trial_id", "t_daq_ms", "src_frame")})
        pj = list(map(str, w4["parts"])).index("jaw")
        jaw4 = w4["pose"][:, pj, :2].astype(float)
        W = pose_windows.lp(d1)
        j1 = np.c_[W[("jaw", "x")].to_numpy(float), W[("jaw", "y")].to_numpy(float)]
        cap1 = cv2.VideoCapture(str(_video(rv, a, d, "cam1", pd.DataFrame(), 0)))
        cap4 = cv2.VideoCapture(str(_video(rv, a, d, "cam4", pd.DataFrame(), 0)))
        for r in g.itertuples():
            h = ix1[ix1.trial_id == r.trial_id]
            t = h.t_daq_ms.to_numpy(float)
            h4 = ix4[ix4.trial_id == r.trial_id]
            t4 = h4.t_daq_ms.to_numpy(float)
            pre = (t >= -500) & (t < 0)
            c1 = np.nanmedian(j1[h.index.to_numpy()[pre]], axis=0)
            pre4 = h4.index[(t4 >= -500) & (t4 < 0)]
            c4 = np.nanmedian(jaw4[pre4], axis=0) if len(pre4) else np.array([640.0, 512.0])
            sel = np.flatnonzero((t >= r.peak_t_ms + CLIP_MS[0]) & (t <= r.peak_t_ms + CLIP_MS[1]))[::CLIP_STEP]
            frames = []
            for k in sel:
                tiles = []
                for cap, src, cc in ((cap1, int(h.src_frame.iloc[k]), c1),
                                     (cap4, int(h4.src_frame.iloc[int(np.argmin(np.abs(t4 - t[k])))]), c4)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, src)
                    ok, im = cap.read()
                    if not ok:
                        im = np.zeros((1024, 1280, 3), np.uint8)
                    if not np.isfinite(cc).all():
                        cc = np.array([im.shape[1] / 2, im.shape[0] / 2])
                    x0, y0 = max(int(cc[0]) - VCROP[0] // 2, 0), max(int(cc[1]) - VCROP[1] // 2, 0)
                    tiles.append(cv2.resize(im[y0:y0 + VCROP[1], x0:x0 + VCROP[0]],
                                            (VTILE, int(VTILE * VCROP[1] / VCROP[0]))))
                fr = np.hstack(tiles)
                cv2.putText(fr, f"{t[k] - r.peak_t_ms:+.0f} ms", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), [cv2.IMWRITE_JPEG_QUALITY, 70])
                frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
            cid = f"v{len(cands):03d}"
            cands.append({"id": cid, "position": r.position, "frames": frames})
            keys.append({"id": cid, "session": s, "trial_id": int(r.trial_id), "position": r.position, "band": r.band,
                         "defl_px": round(r.defl_px, 2), "peak_t_ms": round(r.peak_t_ms, 1),
                         "cam1_moved": r.cam1_moved, "cam4_moved": r.cam4_moved})
        cap1.release()
        cap4.release()
    order = rng.permutation(len(cands))
    return [cands[i] for i in order], [keys[i] for i in order]


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=12, help="candidates per position x cam1 call (video: per band)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true", help="clip round across cam1 deflection bands (jaw_review_video/)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    rng = np.random.default_rng(a.seed)
    if a.video:
        cands, keys = video_round(rv, a.per, rng)
        out = Path(rv.root("dlc")) / "session_poses" / "cam1_licks" / "jaw_review_video"
        out.mkdir(parents=True, exist_ok=True)
        (out / "candidates.json").write_text(json.dumps(cands))
        pd.DataFrame(keys).to_csv(out / "key.csv", index=False)
        print(f"-> {out} ({len(cands)} clips)")
        return 0
    P = _cands(rv, a.per, rng)
    cands, keys = build(rv, P, rng)
    out = Path(rv.root("dlc")) / "session_poses" / "cam1_licks" / "jaw_review"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(json.dumps(cands))
    pd.DataFrame(keys).to_csv(out / "key.csv", index=False)
    print(f"-> {out} ({len(cands)} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
