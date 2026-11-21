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
seen by >= ``min_views`` cameras: 1 = the union (today's cam4 + cam1-only rule extended to four cameras), 2 = two views.
Times are ms from the trial's cue (every production table carries ``trial_id`` + ``on_ms`` / ``off_ms`` on that clock).
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


def fuse(events: dict, tol_ms: float = MERGE_TOL_MS) -> pd.DataFrame:
    """One row per lick from per-camera event tables ({cam: events in `EVENT_COLS`}): trial_id, position, on_ms,
    off_ms (union of the merged spans), t_peak_ms (`PEAK_PRIORITY`), cams ("cam1+cam4"), n_views, seen_<cam> for every
    camera in `CAMS`, lick_idx (order in the trial). Events of one trial chain into one lick while each starts within
    ``tol_ms`` of the running end of the lick so far."""
    parts = [E[EVENT_COLS].assign(guarded=_guarded(E)) for E in events.values() if len(E)]
    if not parts:
        return pd.DataFrame(columns=EVENT_COLS[:-1] + ["cams", "n_views"] + [f"seen_{c}" for c in CAMS]
                            + ["guarded", "lick_idx"])
    A = pd.concat(parts, ignore_index=True).sort_values(["trial_id", "on_ms", "off_ms"], kind="mergesort")
    A = A.reset_index(drop=True)
    run_end = A.groupby("trial_id").off_ms.cummax()
    prev_end = run_end.groupby(A.trial_id).shift(1)
    A["lick"] = (prev_end.isna() | (A.on_ms > prev_end + tol_ms)).cumsum()
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


def lick_train(U: pd.DataFrame, min_views: int = 1) -> pd.DataFrame:
    """The fused licks seen by at least ``min_views`` cameras (D2: 1 = the union, 2 = two views), licks made only of
    guarded events (`fuse`) removed."""
    return U[(U.n_views >= min_views) & ~U.guarded.astype(bool)].reset_index(drop=True)


def view_table(U: pd.DataFrame) -> pd.DataFrame:
    """Licks per camera combination (the check of what the side views add): n, share, how many no camera of
    cam1 + cam4 saw (``side_only``) and how many are guarded-only (dropped by `lick_train`)."""
    side_only = ~U.seen_cam4 & ~U.seen_cam1
    T = U.groupby("cams").size().rename("n").to_frame()
    T["guarded"] = U[U.guarded.astype(bool)].groupby("cams").size().reindex(T.index, fill_value=0)
    T["share"] = T.n / T.n.sum()
    T["side_only"] = U[side_only].groupby("cams").size().reindex(T.index, fill_value=0)
    return T.sort_values("n", ascending=False)
