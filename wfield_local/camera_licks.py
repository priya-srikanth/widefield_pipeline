"""LICKS ACROSS CAMERAS (Priya 11-18: "start drafting code for licks across cameras"): one lick table from cam4, cam1
and the side cameras cam2 / cam3. DRAFT -- the fusion rule and the side guards are decisions D2 / D3 of
docs/status/STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md; nothing reads this module yet.

    from wfield_local import camera_licks as CL
    Z = PS.fuse_session(rv, animal, date, preds_dir, clips_dir)          # four-camera session (side tongue cleaned)
    ev = {**CL.production_events(rv, animal, date), **{c: CL.side_events(Z, c) for c in CL.SIDE}}
    U = CL.fuse(ev)                                                      # one row per lick, which cameras saw it

WHAT EACH CAMERA CONTRIBUTES (no lick definition is re-made here):
  * cam4   the production v36 LP lick table (`licks_all.csv`; every context), as every LP consumer reads it.
  * cam1   the production cam1 2-D events (`cam1_licks`; ``_events_all.csv`` = the whole trial windows, or the post-cue
           ``_events.csv``), filtered by THE cam1 rule `cam1_licks.keep_mask` (droplets, far-from-mouth labels).
  * cam2 / cam3   NEW: `cam1_licks.detect` -- the same run-of-present-tongue-frames detector -- on the side tongue
           after the side cleaning and the side droplet rule (`pose3d_side.fuse_session`: `side_clean`,
           `droplet_mask`). Each side view gets its own mouth frame (`side_frame`): origin = the projected mouth zero,
           ``out`` = toward the projected spouts; the lips threshold and cam1's droplet rule (`cam1_licks.droplet_flags`:
           a near-static point away from the mouth opening) are rescaled by the camera's px / mm over cam1's
           (`pose3d_side.px_per_mm`), as is cam1's far-from-mouth rule (`cam1_licks.FAR_FROM_MOUTH_PX`: a label far
           behind the mouth line). The frame-level side droplet rule misses runs this short: on the r2b clips
           (PS93 0814, 11-18) four cam2-only 3-4 frame events sat static (range <= 1.3 px) ~7 mm out from the mouth.

FUSION (`fuse`): events of one trial whose spans overlap within `MERGE_TOL_MS` (the 12 ms gap that already bridges
frames of one lick, `pose3d.lick_events`) are ONE lick: span = union, peak from the first camera of `PEAK_PRIORITY`
that saw it (cam4's peak is the production one), ``cams`` / ``n_views`` / ``seen_<cam>``. A side event flagged by a
`GUARDS` rule (``static`` = the droplet rule, ``far`` = cam1's far-from-mouth rule, both rescaled) counts only when
another camera's event merges with it -- as cam1's rules apply to cam1-ONLY events (a static point another camera calls
a lick is the tongue held on the spout); a lick made of guarded events alone is ``guarded`` = True and `lick_train`
drops it. `lick_train` keeps licks
seen by >= ``min_views`` cameras: 1 = the union (today's cam4 + cam1-only rule extended to four cameras; the DEFAULT,
D2 as revised by Priya 11-21), 2 = two views (an option), and drops cam1-only licks >= `LONG_CAM1_ONLY_MS` (water drops).
Times are ms from the trial's cue (every production table carries ``trial_id`` + ``on_ms`` / ``off_ms`` on that clock).

SPLITTING BACK-TO-BACK LICKS (11-21 pilot, PS93 0814 / PS94 0817): chaining overlapping events merges licks that cam4
counts separately -- 310 fused PS93 licks held 2-4 cam4 licks, 218 of them because cam4's OWN v36 spans overlap (the
fusion chained them with no other camera) and the rest bridged by a side / cam1 event; in 3-D the four-camera track has
fewer gaps, so 62 runs held two cam4 peaks. `event_cuts` (rule a: the anchor cameras cam4, then cam1, set how many
licks) with ``S`` (rule b: the cut goes where the 3-D tongue returns closest to the mouth; no return inside one unbroken
run -> no split) is the default split for `fuse` and `pose3d.lick_peaks`; `mouth_cuts` (rule b alone, distance minima
by prominence) is kept for comparison: alone it agreed with the camera boundaries on ~30 % of cuts and added cuts
inside single licks. Numbers: docs/status/STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md section 0c.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SIDE = ("cam2", "cam3")
CAMS = ("cam4", "cam1", "cam2", "cam3")
#: whose peak a fused lick takes: cam4 (the production lick table), then cam1, then the side views
PEAK_PRIORITY = CAMS
#: events of one trial closer than this (ms between one's end and the next one's start) are one lick: the gap
#: `pose3d.lick_events` bridges inside one lick (12 ms = 3 frames at 250 fps)
MERGE_TOL_MS = 12.0
EVENT_COLS = ["trial_id", "position", "on_ms", "off_ms", "t_peak_ms", "cam"]


def side_frame(Z: dict, cam: str):
    """``to_frame(xy) -> (out, lat)`` px for camera ``cam`` of a `pose3d_side.fuse_session` result: origin = the mouth
    zero projected into the view, ``out`` = along the projected direction to the mean commanded spout (toward the
    spouts, as cam1's frame), ``lat`` = perpendicular in the image (sign = the image's, reported only)."""
    from wfield_local import pose3d_side as PS
    from wfield_local import spout_world as SW
    cam_obj = PS.cam_of(Z["g"], cam)
    geo = Z["rig"]["geometry"]
    tgt = np.mean([SW.commanded_world(geo, p) for p in SW.POSITIONS], axis=0)
    o, s = PS.project(cam_obj, np.zeros((1, 3)))[0], PS.project(cam_obj, tgt[None])[0]
    u = (s - o) / np.linalg.norm(s - o)
    v = np.array([-u[1], u[0]])

    def to_frame(xy):
        d = np.asarray(xy, float) - o
        return d @ u, d @ v
    return to_frame


def side_events(Z: dict, cam: str, span: str = "post") -> pd.DataFrame:
    """2-D lick events of side camera ``cam`` (`cam1_licks.detect` on its cleaned, droplet-free tongue points in the
    `side_frame`), with ``peak_dist_mm`` (px / `pose3d_side.px_per_mm`: a size scale at the mouth's depth, not 3-D)
    and ``cam``. ``span`` as `cam1_licks.detect` ("post" = cue .. stop, "all" = the whole clip / trial windows)."""
    from wfield_local import cam1_licks as C1
    from wfield_local import pose3d_side as PS
    F = Z["F"]
    xy = Z["pts"]["tongue"][Z["order"].index(cam)]
    out, lat = side_frame(Z, cam)(xy)
    ppm = PS.px_per_mm(PS.cam_of(Z["g"], cam))
    scale = ppm / PS.px_per_mm(PS.cam_of(Z["g"], "cam1"))
    lips = C1.LIPS_PX * scale
    t = F.t_ms.to_numpy(float)
    src = np.round(t / 4.0).astype(np.int64)                     # 4 ms frame grid: consecutive frames differ by 1
    E = C1.detect(out, lat, np.isfinite(xy).all(1), F.trial_id, t, src, F.stop_ms, F.position, lips_px=lips, span=span)
    E["peak_dist_mm"] = E.peak_dist_px / ppm
    E["static"] = C1.droplet_flags(E, np.ones(len(E), bool), scale=scale) if len(E) else []
    E["far"] = (E.med_out_px.to_numpy(float) < C1.FAR_FROM_MOUTH_PX * scale) if len(E) else []
    E["cam"] = cam
    return E


def production_events(rv, animal: str, date: str, span: str = "all") -> dict:
    """{"cam4": ..., "cam1": ...} production lick events of one session in `EVENT_COLS` (+ ``daq_on_s``, ``context``
    where the table has them). cam1 rows pass `cam1_licks.keep_mask`; cam4-matched cam1 rows are KEPT (fusion merges
    them with their cam4 lick, and ``seen_cam1`` then says cam1 saw it too). ``span`` "all" reads cam1's whole-window
    ``_events_all.csv``, "post" the post-cue ``_events.csv``. A missing table -> that camera absent."""
    from wfield_local import cam1_licks as C1
    from wfield_local import lp_licks
    sd, f4, f1 = lp_licks._inputs(f"{animal}_{date[4:]}", rv)
    if span == "all":
        f1 = f1.with_name(f1.name.replace("_events.csv", "_events_all.csv"))
    out = {}
    if Path(f4).exists():
        E4 = pd.read_csv(f4).rename(columns={"t_ms": "t_peak_ms"})
        out["cam4"] = _std(E4, "cam4")
    if Path(f1).exists():
        E1 = pd.read_csv(f1)
        out["cam1"] = _std(E1[C1.keep_mask(E1, label=f"{animal}_{date}")], "cam1")
    return out


def _std(E: pd.DataFrame, cam: str) -> pd.DataFrame:
    E = E.assign(cam=cam)
    extra = [c for c in ("daq_on_s", "context") if c in E]
    return E[EVENT_COLS + extra].reset_index(drop=True)


#: event flags that make an event count only when another camera's event merges with it (`side_events`)
GUARDS = ("static", "far")


def _guarded(E: pd.DataFrame) -> np.ndarray:
    g = np.zeros(len(E), bool)
    for c in GUARDS:
        if c in E:
            g |= E[c].astype(bool).to_numpy()
    return g


def within_frames(E: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """The events of ``E`` that overlap the frames of ``F`` (per trial, first .. last ``t_ms``): restricts a whole-session
    table to the clips a side model was run on, so every camera is compared over the same time."""
    from wfield_local import cam1_licks as C1
    W = F.groupby("trial_id").t_ms.agg(on_ms="min", off_ms="max").reset_index()
    return E[C1.overlaps(E, W)].reset_index(drop=True) if len(E) else E


#: cameras whose own lick boundaries split a fused lick / a 3-D run (`event_cuts`), in precedence order: cam4 (the
#: production lick table) always; cam1 only where cam4 does not call the time one lick
ANCHORS = ("cam4", "cam1")
#: tongue-return split (`mouth_cuts`, Priya 11-21: "shouldn't we also be able to tell based on the position the tongue
#: returns to (the 'mouth')?"): a local minimum of the 3-D tip-to-mouth distance inside a run, at least this
#: PROMINENT (mm below the lower of the two neighboring extension peaks), not an absolute distance, so small licks
#: at the lips still split. A comparison rule only (11-21 pilot: the median tongue return between two cam4 licks that
#: shared one 3-D run was 1.37 mm, 10 % below 0.77 mm; 0.75 split 72 of 310 merged fused licks): see the module doc.
MOUTH_PROMINENCE_MM = 0.75
#: physiological floor between lick peaks (`configs/defaults.yaml lick_detection.min_ili_ms`): a cut leaving two
#: peaks closer than this is not made
MIN_ILI_MS = 40.0


def event_cuts(events: dict, anchors: tuple = ANCHORS, S: pd.DataFrame | None = None) -> dict:
    """{trial_id: sorted cut times (ms from the cue)} at the boundaries between consecutive lick events of the
    ``anchors`` cameras: the midpoint between one event's end and the next one's start (a gap or an overlap), kept
    between their peaks. A later anchor's cut is dropped where it falls inside an event of an earlier anchor (cam4
    calls that span one lick).

    With ``S`` (a `pose3d` frames table) each cut moves to where the 3-D tongue is CLOSEST TO THE MOUTH between the two
    peaks (the anchor camera sets HOW MANY licks, the tongue's return sets WHERE one ends: Priya 11-21); a cut at the
    span midpoint can fall on a retracting tongue, and that piece's farthest frame then sits right after the cut, a
    sub-40 ms "lick" (PS93 0814: 84 such 3-D intervals vs 37 uncut). No 3-D frame between the peaks -> the midpoint."""
    cuts: dict = {}
    inside: dict = {}
    trk = _mouth_track(S) if S is not None else {}
    for cam in anchors:
        E = events.get(cam)
        if E is None or not len(E):
            continue
        for tid, g in E.sort_values(["trial_id", "t_peak_ms"]).groupby("trial_id"):
            on, off, pk = g.on_ms.to_numpy(float), g.off_ms.to_numpy(float), g.t_peak_ms.to_numpy(float)
            # consecutive events can OVERLAP (cam4 v36 spans: 225 of 9,994 on PS93 0814), so the cut is the middle of
            # the gap or of the overlap, kept between the two peaks (else the midpoint of the peaks)
            c = (off[:-1] + on[1:]) / 2.0
            bad = ~((c > pk[:-1]) & (c < pk[1:]))
            c[bad] = (pk[:-1][bad] + pk[1:][bad]) / 2.0
            ok = pk[1:] > pk[:-1]
            if tid in trk:
                c = _closest_to_mouth(trk[tid], pk[:-1], pk[1:], c)
            c = c[ok]
            spans = inside.get(tid)
            if spans is not None and len(c):
                c = c[~((spans[:, 0][None] <= c[:, None]) & (spans[:, 1][None] >= c[:, None])).any(1)]
            cuts[tid] = np.union1d(cuts.get(tid, np.array([])), c)
            sp = np.c_[on, off]
            inside[tid] = sp if spans is None else np.r_[spans, sp]
    return cuts


def _mouth_track(S: pd.DataFrame) -> dict:
    """{trial_id: (t_ms, tip-to-mouth distance mm)} over the frames with a 3-D tongue."""
    P = S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    ok = np.isfinite(P).all(1)
    t, tid, r = S.t_ms.to_numpy(float)[ok], S.trial_id.to_numpy()[ok], np.linalg.norm(P[ok], axis=1)
    out = {}
    if not len(t):
        return out
    b = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1], True])
    for x, y in zip(b[:-1], b[1:]):
        out[int(tid[x])] = (t[x:y], r[x:y])
    return out


#: a peak farther than this from the 3-D track (ms) is not on it (the cut then goes between the track and that peak)
EDGE_MS = 12.0


def _closest_to_mouth(tr, p0: np.ndarray, p1: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Move each cut ``c`` (between peaks p0 < p1) to the frame of minimum distance between the peaks -- only when the
    3-D track runs UNBROKEN from one peak to the other (the case that needs a cut). Where the track breaks between the
    peaks, the cut goes into the longest break (the gap already separates the licks; a minimum searched across it lands
    on the first lick's retracting tail and splits that off as a 3-4 frame "lick", PS93 0814). No frames -> ``c``."""
    t, r = tr
    c = c.copy()
    lo, hi = np.searchsorted(t, p0, "left"), np.searchsorted(t, p1, "right")
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a < 2:
            continue
        tt = t[a:b]
        steps = np.diff(tt)
        if (steps > 4.5).any():                                  # the track breaks: cut in the longest break
            k = int(np.argmax(steps))
            c[i] = (tt[k] + tt[k + 1]) / 2.0
        elif tt[0] - p0[i] > EDGE_MS:                            # the track starts after the first peak
            c[i] = (p0[i] + tt[0]) / 2.0
        elif p1[i] - tt[-1] > EDGE_MS:                           # ... or ends before the second
            c[i] = (tt[-1] + p1[i]) / 2.0
        else:
            c[i] = tt[int(np.argmin(r[a:b]))] - 0.5             # the minimum frame starts the next lick
    return c


def mouth_cuts(S: pd.DataFrame, prominence_mm: float = MOUTH_PROMINENCE_MM, min_ili_ms: float = MIN_ILI_MS,
               min_frames: int = 3) -> dict:
    """{trial_id: cut times} where the tongue RETURNS TOWARD THE MOUTH inside a run of consecutive 3-D tongue frames
    (``S`` = a `pose3d` frames table; distance = |tip| from the mouth zero): a local minimum at least ``prominence_mm``
    below both neighboring maxima (`scipy.signal.find_peaks` prominence). The cut is at the minimum's frame (it starts
    the next lick). A cut that leaves either side shorter than ``min_frames`` frames, or the two extension peaks closer
    than ``min_ili_ms``, is not made."""
    from scipy.signal import find_peaks
    t, tid = S.t_ms.to_numpy(float), S.trial_id.to_numpy()
    P = S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    have = np.isfinite(P).all(1)
    r = np.linalg.norm(P, axis=1)
    k = np.flatnonzero(have)
    cuts: dict = {}
    if not len(k):
        return cuts
    brk = np.r_[True, (tid[k[1:]] != tid[k[:-1]]) | (np.abs(t[k[1:]] - t[k[:-1]] - 4.0) > 0.5)]
    st = np.flatnonzero(brk)
    for a, b in zip(st, np.r_[st[1:], len(k)]):
        kk = k[a:b]
        if len(kk) < 2 * min_frames:
            continue
        rr = r[kk]
        mins, _ = find_peaks(-rr, prominence=prominence_mm)
        if not len(mins):
            continue
        keep, lo = [], 0
        for m in mins:                                  # left to right: each sub-lick keeps >= min_frames + its peak
            nxt = [x for x in mins if x > m]
            hi = nxt[0] if nxt else len(kk)
            if m - lo < min_frames or hi - m < min_frames:
                continue
            p_left = lo + int(np.argmax(rr[lo:m]))
            p_right = m + int(np.argmax(rr[m:hi]))
            if t[kk[p_right]] - t[kk[p_left]] < min_ili_ms:
                continue
            keep.append(m)
            lo = m
        if keep:
            tt = int(tid[kk[0]])
            cuts[tt] = np.union1d(cuts.get(tt, np.array([])), t[kk[keep]] - 0.5)
    return cuts


def fuse(events: dict, tol_ms: float = MERGE_TOL_MS, cuts: dict | None = None) -> pd.DataFrame:
    """One row per lick from per-camera event tables ({cam: events in `EVENT_COLS`}): trial_id, position, on_ms,
    off_ms (union of the merged spans), t_peak_ms (`PEAK_PRIORITY`), cams ("cam1+cam4"), n_views, seen_<cam> for every
    camera in `CAMS`, lick_idx (order in the trial). Events of one trial chain into one lick while each starts within
    ``tol_ms`` of the running end of the lick so far.

    ``cuts`` ({trial_id: ms}, `event_cuts` / `mouth_cuts`) SPLIT a chained lick (11-21 pilot: side / cam1 events
    bridged back-to-back cam4 licks into one): each event goes to the piece holding its peak, and its span is clipped
    to that piece, so a bridging event lends each lick only its own part. A cut with no event peak on one side of it
    inside the lick is ignored."""
    parts = [E[EVENT_COLS].assign(guarded=_guarded(E)) for E in events.values() if len(E)]
    if not parts:
        return pd.DataFrame(columns=EVENT_COLS[:-1] + ["cams", "n_views"] + [f"seen_{c}" for c in CAMS]
                            + ["guarded", "lick_idx"])
    A = pd.concat(parts, ignore_index=True).sort_values(["trial_id", "on_ms", "off_ms"], kind="mergesort")
    A = A.reset_index(drop=True)
    run_end = A.groupby("trial_id").off_ms.cummax()
    prev_end = run_end.groupby(A.trial_id).shift(1)
    A["lick"] = (prev_end.isna() | (A.on_ms > prev_end + tol_ms)).cumsum()
    if cuts:
        A = _apply_cuts(A, cuts)
    rank = {c: i for i, c in enumerate(PEAK_PRIORITY)}
    A["rank"] = A.cam.map(rank).fillna(len(rank))
    peak = A.sort_values(["lick", "rank"], kind="mergesort").groupby("lick").t_peak_ms.first()
    g = A.groupby("lick")
    U = g.agg(trial_id=("trial_id", "first"), position=("position", "first"), on_ms=("on_ms", "min"),
              off_ms=("off_ms", "max"))
    U["t_peak_ms"] = peak
    seen = pd.crosstab(A.lick, A.cam).reindex(columns=list(CAMS), fill_value=0) > 0
    for c in CAMS:
        U[f"seen_{c}"] = seen[c].to_numpy()
    U["n_views"] = seen.sum(1).to_numpy()
    U["guarded"] = g.guarded.all().to_numpy()
    U["cams"] = ["+".join(sorted(c for c in CAMS if r[c])) for _, r in seen.iterrows()]
    U = U.reset_index(drop=True)
    U["lick_idx"] = U.groupby("trial_id").cumcount()
    return U


def _apply_cuts(A: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    """Split each chained lick of ``A`` (column ``lick``) at the ``cuts`` of its trial: events by peak, spans clipped."""
    rows = []
    for _lk, g in A.groupby("lick", sort=True):
        c = cuts.get(g.trial_id.iloc[0])
        if c is None or not len(c):
            rows.append(g.assign(piece=0))
            continue
        c = np.asarray(c, float)
        c = c[(c > g.on_ms.min()) & (c < g.off_ms.max())]
        piece = np.searchsorted(c, g.t_peak_ms.to_numpy(float), side="right")
        used = np.unique(piece)
        if len(used) < 2:
            rows.append(g.assign(piece=0))
            continue
        bounds = np.r_[-np.inf, c, np.inf]
        g = g.assign(piece=piece)
        g["on_ms"] = np.maximum(g.on_ms.to_numpy(float), bounds[piece])
        g["off_ms"] = np.minimum(g.off_ms.to_numpy(float), bounds[piece + 1])
        rows.append(g)
    A = pd.concat(rows).sort_values(["lick", "piece", "on_ms"], kind="mergesort").reset_index(drop=True)
    A["lick"] = (A[["lick", "piece"]].diff().abs().sum(1) > 0).cumsum()
    return A.drop(columns="piece")


#: D2 as revised (Priya 11-21: "cam 1 DOES capture the tiny tongue at the lips licks so we shouldn't get rid of those,
#: and maybe we'll just use the cam1 rules to get rid of water?"): any camera's lick counts (cam1 after its own rules,
#: side views only when confirmed); a cam1-ONLY lick at least this long that no other camera saw is dropped -- the
#: wobbling water drops hanging from the spout that cam1's droplet rule misses (11-18 trial 540 census)
LONG_CAM1_ONLY_MS = 300.0


def lick_train(U: pd.DataFrame, min_views: int = 1, long_cam1_only_ms: float | None = LONG_CAM1_ONLY_MS) -> pd.DataFrame:
    """The fused licks seen by at least ``min_views`` cameras (D2 revised 11-21: 1 = the union, the default; 2 = two
    views kept as an option), licks made only of guarded events (`fuse`) removed, and cam1-only licks lasting
    >= ``long_cam1_only_ms`` removed (None = keep them)."""
    keep = (U.n_views >= min_views) & ~U.guarded.astype(bool)
    if long_cam1_only_ms is not None:
        keep &= ~((U.n_views == 1) & U.seen_cam1.astype(bool) & ((U.off_ms - U.on_ms) >= long_cam1_only_ms))
    return U[keep].reset_index(drop=True)


def view_table(U: pd.DataFrame) -> pd.DataFrame:
    """Licks per camera combination (the check of what the side views add): n, share, how many no camera of
    cam1 + cam4 saw (``side_only``) and how many are guarded-only (dropped by `lick_train`)."""
    side_only = ~U.seen_cam4 & ~U.seen_cam1
    T = U.groupby("cams").size().rename("n").to_frame()
    T["guarded"] = U[U.guarded.astype(bool)].groupby("cams").size().reindex(T.index, fill_value=0)
    T["share"] = T.n / T.n.sum()
    T["side_only"] = U[side_only].groupby("cams").size().reindex(T.index, fill_value=0)
    return T.sort_values("n", ascending=False)
