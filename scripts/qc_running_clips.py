"""Side-camera clips with the SIGNED treadmill trace under each frame (Priya 10-23: "the cue-locked speed increases
could either be the treadmill 'rebound' ... or real walking ... let's look at a couple videos").
Events per session: (a) post-cue backward dips on still-before trials, (b) post-cue forward movement 1-3 s on
still-before trials, (c) reverse speed after running-bout ends (backlash). Located by camera TIMESTAMP via the
alignment template (as behavior_clips); source videos read-only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRE, POST, SLOW = 1.5, 3.5, 2.0          # s before / after the event; playback slowdown
OUT_W = 600


def events(B):
    t, v = (np.asarray(x, float) for x in B.analog["running"])
    cues = B.trials.cue_s.to_numpy(float)
    cues = cues[np.isfinite(cues)]
    off = np.arange(-3, 3, 0.02)
    seg = np.array([np.interp(c + off, t, v) for c in cues])
    still = np.abs(seg[:, off < 0]).max(1) < 0.5
    dip = seg[:, (off >= 0) & (off < 0.6)].min(1)
    fwd = seg[:, (off >= 1) & (off < 3)].max(1)
    ev = []
    for i in np.argsort(np.where(still, dip, np.inf))[:3]:
        ev.append(("cue_dip", cues[i], dip[i]))
    for i in np.argsort(np.where(still, -fwd, np.inf))[:3]:
        ev.append(("cue_forward", cues[i], fwd[i]))
    dt = np.median(np.diff(t))
    fast = np.abs(v) > 3
    ends = np.flatnonzero(fast[:-1] & ~fast[1:])
    mins = np.array([v[e:e + int(2 / dt)].min() if e + int(2 / dt) < len(v) else 0 for e in ends])
    for j in np.argsort(mins)[:3]:
        ev.append(("bout_end_reverse", t[ends[j]], mins[j]))
    return t, v, ev


def clip(cap, cam_ts, tpl, t, v, kind, t_ev, val, dest, title):
    import cv2
    fps = float(tpl["fps_cam"])
    cam_s = (t_ev - float(tpl["intercept_daqSec"])) / float(tpl["slope_daqSec_per_camSec"])
    f0 = int(np.searchsorted(cam_ts, (cam_s - PRE) * 1e9))
    n = int(round((PRE + POST) * fps))
    if f0 < 0 or f0 + n >= len(cam_ts):
        return 0
    m = (t >= t_ev - PRE) & (t <= t_ev + POST)
    tt, vv = t[m] - t_ev, v[m]
    lo, hi = min(-1.0, vv.min()), max(3.0, vv.max())
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    vw = None
    keep = max(1, int(round(fps / 30.0 / SLOW * 1.0)))     # ~30 fps output at 1/SLOW speed
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        if i % keep:
            continue
        # time of THIS frame from its own timestamp (drops-safe)
        t_rel = ((cam_ts[f0 + i] / 1e9) * float(tpl["slope_daqSec_per_camSec"]) + float(tpl["intercept_daqSec"])) - t_ev
        h, w = fr.shape[:2]
        sc = OUT_W / w
        f2 = cv2.resize(fr, (OUT_W, int(h * sc)), interpolation=cv2.INTER_AREA)
        ph = 170
        panel = np.full((ph, OUT_W, 3), 25, np.uint8)
        X = lambda x: int(10 + (OUT_W - 20) * (x + PRE) / (PRE + POST))          # noqa: E731
        Y = lambda y, ph=ph: int(ph - 20 - (ph - 40) * (y - lo) / (hi - lo))      # noqa: E731
        for yv, col in ((0, (120, 120, 120)), (1, (0, 165, 255)), (1.5, (0, 220, 255)), (3, (60, 60, 230)),
                        (-1, (200, 120, 120))):
            if lo <= yv <= hi:
                cv2.line(panel, (10, Y(yv)), (OUT_W - 10, Y(yv)), col, 1)
        pts = np.array([[X(a), Y(b)] for a, b in zip(tt[::4], vv[::4])], np.int32)
        cv2.polylines(panel, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(panel, (X(0), 5), (X(0), ph - 5), (80, 220, 80), 1)
        cv2.line(panel, (X(t_rel), 5), (X(t_rel), ph - 5), (0, 255, 255), 2)
        cv2.putText(panel, f"speed mm/s (signed; lines 0 / +1 / +1.5 / +3 / -1)  t={t_rel:+.2f}s", (10, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        out = np.vstack([f2, panel])
        cv2.putText(out, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        if vw is None:
            vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (out.shape[1], out.shape[0]))
        vw.write(out)
    if vw is not None:
        vw.release()
        return 1
    return 0


def main(labels, cam="cam2"):
    import cv2

    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    rv = PathResolver()
    dest = oo.root(rv) / oc.RUN / "imaging_encoding" / "qc" / "running_clips"
    assert_writable(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for lab in labels:
        an, date = config.animal_of(lab), "2026" + lab.split("_")[-1]
        tp = Path(rv.root("alignment_templates")) / cam / an / (date + ".npz")
        tpl = dict(np.load(tp, allow_pickle=True))
        vids = sorted((Path(rv.root("behavior_cameras")) / date / an).glob(cam + "_*.avi"))
        csvs = sorted((Path(rv.root("behavior_cameras")) / date / an).glob(cam + "_*.csv"))
        cam_ts = pd.read_csv(csvs[0], header=None, usecols=[1], dtype="int64").to_numpy().ravel()
        B = ce.load_behaviour(an, date, rv)
        t, v, ev = events(B)
        cap = cv2.VideoCapture(str(vids[0]))                # READ-ONLY
        try:
            for k, (kind, t_ev, val) in enumerate(ev):
                nm = f"{lab}_{cam}_{kind}_{k % 3 + 1}.mp4"
                ttl = f"{lab} {cam} {kind} (extreme {val:+.2f} mm/s) 0.5x; green = event (cue / bout end)"
                ok = clip(cap, cam_ts, tpl, t, v, kind, t_ev, val, dest / nm, ttl)
                print(nm, "ok" if ok else "SKIP (off recording)", flush=True)
        finally:
            cap.release()
    print("->", dest)


if __name__ == "__main__":
    main(sys.argv[1:] or ["PS94_0921", "PS95_0812"])
