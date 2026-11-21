"""cam1 2-D LICK EVENTS (Priya 10-19: "let's add a cam1 2d lick detector") -- the incomplete licks at the lips that cam4
and the 3-D track miss.

Why a 2-D detector: a tongue tip seen by cam1 alone, away from any both-camera frame, has no 3-D point (the depth-prior
check, `scripts/pose3d_depth_prior_check.py`: 0.6-1.7 mm error on licks < 1 mm -- no usable depth), so the combined 3-D
track cannot contain it. Counting and sizing such licks needs no depth: cam1 (from below) sees the tongue leave the mouth.

    from wfield_local import cam1_licks as C1
    E = C1.detect(out_px, lat_px, present, trial_id, t_ms, src_frame, stop_ms, position)

An EVENT = a run of PRESENT cam1 tongue frames (after the cam1 cleaning: measured, or bridged by its <= 24 ms gap rule)
within one trial, consecutive video frames, cue (0 ms) .. trial stop, >= ``min_frames`` frames. Per event: on / off,
the PEAK = the frame farthest from the cam1 mouth (`spout_frame.from_medians_lr` origin), its distance / ``out``
(ventral, toward the spouts) / ``lat`` (+ = mouse's left) in px, and ``size_class``: ``lips`` when the peak never gets
more than ``lips_px`` below the mouth (``out``; `scripts/cam1_lick_aim.py`'s "dorsal runs", 10-18: a pale bump at the
lips, mouth nearly closed) else ``out``. The 20 px default is the 10-18 value -- PROPOSED, to be set from the blind review.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_FRAMES = 3
LIPS_PX = 20.0
CONTACT_TOL_MS = 20.0
#: WATER DROPLETS (Priya's blind review, 2 rounds, 10-19: every false cam1-only event was a droplet): a near-static point
#: (median distance from the event's median point < 3 px and total distance range < 12 px) whose median sits AWAY from
#: the mouth opening (``med_out_px`` > -35; real tongue tips at the lips sit at <= -44). 16 / 17 false caught, 0 / 70 real
#: lost. Applied to cam1-ONLY events: a static point that cam4 also calls a lick is the tongue held on the spout.
#: Droplets follow uncollected water (pre: 3 % of trials after a hit, 12 % after a miss, 15 % after 3 misses; most at
#: close_center) -- ``miss_prev3`` is kept per event as a covariate.
DROPLET_SPREAD_PX, DROPLET_RANGE_PX, DROPLET_MOUTH_OUT_PX = 3.0, 12.0, -35.0


def detect(out_px, lat_px, present, trial_id, t_ms, src_frame, stop_ms, position, *, min_frames: int = MIN_FRAMES,
           lips_px: float = LIPS_PX, span: str = "post") -> pd.DataFrame:
    """``span`` "post" = cue .. trial stop (the original post-cue events); "all" = every frame of the trial windows
    (strobe - 0.5 s .. stop + 3 s, cut at the next trial; encoding window via Priya 10-20: licks outside cue..stop),
    label with `orofacial_outputs.lick_context` (by the event's PEAK, 11-04)."""
    out_px, lat_px = np.asarray(out_px, float), np.asarray(lat_px, float)
    t, stop = np.asarray(t_ms, float), np.asarray(stop_ms, float)
    tid, src, pos = np.asarray(trial_id), np.asarray(src_frame), np.asarray(position)
    live = np.asarray(present, bool) & np.isfinite(out_px) & np.isfinite(lat_px)
    if span == "post":
        live &= (t >= 0) & (t <= stop)
    elif span != "all":
        raise ValueError(f"span must be 'post' or 'all', not {span!r}")
    k = np.flatnonzero(live)
    cols = ["trial_id", "position", "on_ms", "off_ms", "t_peak_ms", "n_frames", "peak_dist_px", "peak_out_px",
            "peak_lat_px", "max_out_px", "range_px", "spread_px", "med_out_px", "med_lat_px", "size_class", "lick_idx"]
    if not len(k):
        return pd.DataFrame(columns=cols)
    brk = np.r_[True, (tid[k[1:]] != tid[k[:-1]]) | (src[k[1:]] - src[k[:-1]] != 1)]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], len(k)]
    rows = []
    for a, b in zip(starts, ends):
        kk = k[a:b]
        if len(kk) < min_frames:
            continue
        d = np.hypot(out_px[kk], lat_px[kk])
        pk = kk[int(np.argmax(d))]
        mo = float(np.max(out_px[kk]))
        # movement within the event (review 10-19: acute false events = the label parked on a static spot for seconds)
        xy = np.c_[out_px[kk], lat_px[kk]]
        mxy = np.median(xy, 0)
        rows.append({"trial_id": int(tid[kk[0]]), "position": str(pos[kk[0]]), "on_ms": t[kk[0]], "off_ms": t[kk[-1]],
                     "t_peak_ms": t[pk], "n_frames": len(kk), "peak_dist_px": float(d.max()),
                     "peak_out_px": out_px[pk], "peak_lat_px": lat_px[pk], "max_out_px": mo,
                     "range_px": float(np.ptp(d)), "spread_px": float(np.median(np.hypot(*(xy - mxy).T))),
                     "med_out_px": float(mxy[0]), "med_lat_px": float(mxy[1]),
                     "size_class": "lips" if mo <= lips_px else "out"})
    E = pd.DataFrame(rows, columns=cols[:-1])
    E["lick_idx"] = E.groupby("trial_id").cumcount() if len(E) else []
    return E


def droplet_flags(E: pd.DataFrame, cam1_only, scale: float = 1.0) -> np.ndarray:
    """True for cam1-only events that look like a water droplet (see ``DROPLET_*``). ``scale`` = the camera's px / mm
    over cam1's, for the same rule on another view's events (`camera_licks.side_events`); 1 = cam1."""
    return (np.asarray(cam1_only, bool) & (E.spread_px.to_numpy(float) < DROPLET_SPREAD_PX * scale)
            & (E.range_px.to_numpy(float) < DROPLET_RANGE_PX * scale)
            & (E.med_out_px.to_numpy(float) > DROPLET_MOUTH_OUT_PX * scale))


#: FAR-FROM-MOUTH LABELS (Priya 11-17, approved): a cam1-only event whose median label position sits this far BEHIND the
#: mouth line (``med_out_px`` < -120 px) is a static tracking label, not a tongue. Her blind review scored 7 / 8
#: "no tongue: other" test events below it and 0 / 30 tongue events (docs/status/CAM1_EXTRA_LICKS_CHECK_20261117.md;
#: blast radius docs/status/CAM1_FARMOUTH_BLAST_RADIUS_20261117.md). Applied, like the droplet rule, to cam1-ONLY
#: events: a cam4-matched row is a cam4 lick whatever its cam1 label does. `restnl_licks.FAR_FROM_MOUTH_PX` is this.
FAR_FROM_MOUTH_PX = -120.0
_NO_FAR_COL: set = set()


def flag_col(x) -> np.ndarray:
    """A True/False event-flag column as bool; missing / NaN / anything but "true" (any case) = False."""
    return pd.Series(x).astype(str).str.lower().eq("true").to_numpy()


def keep_mask(E: pd.DataFrame, cam4_matched=None, *, label: str | None = None) -> np.ndarray:
    """THE cam1 LICK RULE (one definition, CLAUDE.md rule 9; Priya 11-17): ``bool[len(E)]``, True = this row of a cam1
    `*_events.csv` / `*_events_all.csv` table may count as a lick. A row is DROPPED when it is

      * a WATER DROPLET: the ``droplet`` column, or `droplet_flags` (on the cam1-only rows) when the column is absent;
      * a FAR-FROM-MOUTH label: cam1-only (not cam4-matched) with ``med_out_px`` < `FAR_FROM_MOUTH_PX`.

    cam4-matched rows are never dropped here (the droplet column is False on them by construction); callers that want
    cam1-ONLY licks still AND with ``~cam4_matched`` themselves. ``cam4_matched`` = the already-parsed flag (default:
    the table's column via `flag_col`). A table without ``med_out_px`` gets no far guard, printed once per ``label``."""
    m4 = flag_col(E["cam4_matched"]) if cam4_matched is None else np.asarray(cam4_matched, bool)
    if "droplet" in E:
        dr = flag_col(E["droplet"])
    else:
        dr = droplet_flags(E, ~m4)
    if "med_out_px" in E:
        far = ~m4 & (pd.to_numeric(E["med_out_px"], errors="coerce") < FAR_FROM_MOUTH_PX).to_numpy()
    else:
        far = np.zeros(len(E), bool)
        key = label or "<unlabelled cam1 table>"
        if key not in _NO_FAR_COL:
            _NO_FAR_COL.add(key)
            print(f"[cam1_licks] {key}: no med_out_px column -> no far-from-mouth guard", flush=True)
    return ~dr & ~far


def overlaps(E: pd.DataFrame, W: pd.DataFrame, on: str = "on_ms", off: str = "off_ms", tol_ms: float = 0.0) -> np.ndarray:
    """Per event of ``E``: does any window of ``W`` (same trial_id; ``on`` .. ``off``) overlap it, +- ``tol_ms``?"""
    win = {k: g[[on, off]].to_numpy(float) for k, g in W.groupby("trial_id")}
    res = []
    for r in E.itertuples():
        w = win.get(r.trial_id)
        res.append(bool(w is not None and ((w[:, 0] <= r.off_ms + tol_ms) & (w[:, 1] >= r.on_ms - tol_ms)).any()))
    return np.array(res, bool)


def contact_flags(E: pd.DataFrame, contacts: dict, tol_ms: float = CONTACT_TOL_MS) -> np.ndarray:
    """A DAQ spout-contact onset (``contacts`` = {trial_id: [ms from cue]}) within the event +- ``tol_ms``."""
    res = []
    for r in E.itertuples():
        c = np.asarray(contacts.get(r.trial_id, []), float)
        res.append(bool(((c >= r.on_ms - tol_ms) & (c <= r.off_ms + tol_ms)).any()))
    return np.array(res, bool)


def mm_per_px(lat_by_pos: dict, ml_by_pos: dict) -> float:
    """cam1 image scale at the spouts: |ml(far_L) - ml(far_R)| (commanded, mm) / |lat(far_L) - lat(far_R)| (image px,
    along the cam1 frame's left-right axis). The tongue at the lips sits at about the spout tips' depth; reach toward the
    camera is foreshortened, so mm values are a SIZE scale, not 3-D distances."""
    d_px = abs(lat_by_pos["far_L"] - lat_by_pos["far_R"])
    return float(abs(ml_by_pos["far_L"] - ml_by_pos["far_R"]) / d_px) if d_px > 0 else float("nan")


def cam1_frame(flat, ix):
    """(spout frame, ruler medians, ``to_frame(x, y) -> (out, lat)`` px) of a cam1 session: origin = the cam1 mouth (the
    L / R spout-line intersection, `spout_frame.from_medians_lr`, on the ENL no-lick ruler medians), ``out`` = ventral
    toward the spouts, ``lat`` = + mouse's left. Moved verbatim from `scripts/cam1_licks._cam1_frame` (11-18), which
    re-exports it."""
    from wfield_local import spout_frame as SF
    med = SF.ruler_medians(flat, ix)
    fr = SF.from_medians_lr(med)
    sgn = fr.mouse_left_sign or 1

    def to_frame(x, y):
        dx, dy = np.asarray(x, float) - fr.origin[0], np.asarray(y, float) - fr.origin[1]
        return -(dx * fr.ap_axis[0] + dy * fr.ap_axis[1]), sgn * (dx * fr.lr_axis[0] + dy * fr.lr_axis[1])
    return fr, med, to_frame


#: DETACHED, STATIONARY TONGUE LABELS (Priya 11-18: "would a rule about it being relatively stationary and 'jumping' from
#: mouth be helpful?" / "we would want to exclude these before cleaning / interpolating"). A real tongue leaves the
#: mouth and moves; a water drop hanging on the spout (PS93 0814 trial 540, four-camera review 11-18) or a parked label
#: appears away from the mouth and sits still. Per run of RAW confident cam1 tongue frames (one trial, likelihood >= the
#: cam1 tongue ``lk_thr``, gaps <= ``DETACH_MAX_GAP_MS`` joined, `tongue_runs`): MASKED when the run never comes back
#: within ``DETACH_OUT_PX`` of the mouth line (min ``out`` > it; ``out`` = ventral, toward the spouts) AND its median
#: distance from its own median point (``spread``) is < ``STATIC_SPREAD_PX``. Applied BEFORE `orofacial_clean`, so the
#: cleaning's 24 ms bridge cannot stitch a wobbling drop into a long event (trial 540: 556-748 ms "licks").
#: SET FROM (11-18, scratch `runs_table.py`; 76 sessions = every blind-reviewed session + 6 per animal across epochs):
#:   * blind-review labels (10-19, review2, review3, 11-17; null controls out): 0 / 118 tongue runs masked; no-tongue
#:     runs IN FRONT of the mouth 9 / 10 caught (the miss: a droplet wobbling 9 px); the 7 far behind are the far guard's
#:     (`FAR_FROM_MOUTH_PX`); the 7 that reach the mouth line are not separable by position;
#:   * trial 540: 1,099 / 1,124 drop frames masked;
#:   * 25 sessions: 0.30 % of confident tongue frames masked, 7.6 % of those inside a cam4 lick (on..off).
#: A run that overlaps a cam4 LP lick (production ``licks_all.csv`` on..off) is NEVER masked (Priya 11-18: "should we keep
#: those with a cam4 correlate?" -- yes): of 12 sampled masked runs inside a cam4 lick, 8 were the real tongue held at the
#: spout (label on the tongue in cam1, tongue in cam4), 2 grooming, 2 the label on a drop as the tongue retracted; 6 / 6
#: sampled masked runs WITHOUT a cam4 lick were a drop or the bare rod (scratch `masked_sheet.py`). As the droplet and
#: far-from-mouth rules, the mask then acts on cam1-ONLY evidence.
#: Rejected versions: raw runs without the bridge (35 % of masked frames inside cam4 licks: a lick whose label flickers
#: at the spout looked detached) and a range < 8 px test on bridged runs (missed trial 540: a long wobbling drop spans
#: 10-24 px).
DETACH_OUT_PX = 30.0
STATIC_SPREAD_PX = 6.0
#: raw runs are joined across gaps up to this (the cam1 cleaning's own bridge, 24 ms) before the test
DETACH_MAX_GAP_MS = 24.0


def tongue_runs(out, lat, ok, trial_id, t_ms, max_gap_ms: float = DETACH_MAX_GAP_MS) -> pd.DataFrame:
    """One row per run of confident cam1 tongue frames (one trial, gaps <= ``max_gap_ms`` joined): ``a`` / ``b`` (first /
    last frame row), n, min_out / max_out (px), range (ptp of the distance from the mouth), spread (median distance from
    the run's median point), dur_ms. ``out`` / ``lat`` = cam1 frame px (`cam1_frame`)."""
    out, lat = np.asarray(out, float), np.asarray(lat, float)
    t, tid = np.asarray(t_ms, float), np.asarray(trial_id)
    k = np.flatnonzero(np.asarray(ok, bool) & np.isfinite(out) & np.isfinite(lat))
    cols = ["a", "b", "trial_id", "n", "min_out", "max_out", "range", "spread", "dur_ms"]
    if not len(k):
        return pd.DataFrame(columns=cols)
    brk = np.r_[True, (tid[k[1:]] != tid[k[:-1]]) | (t[k[1:]] - t[k[:-1]] > max_gap_ms + 1.0)]
    starts = np.flatnonzero(brk)
    rows = []
    for i, j in zip(starts, np.r_[starts[1:], len(k)]):
        kk = k[i:j]
        xy = np.c_[out[kk], lat[kk]]
        med = np.median(xy, 0)
        rows.append((kk[0], kk[-1], tid[kk[0]], len(kk), out[kk].min(), out[kk].max(), float(np.ptp(np.hypot(*xy.T))),
                     float(np.median(np.hypot(*(xy - med).T))), t[kk[-1]] - t[kk[0]]))
    return pd.DataFrame(rows, columns=cols)


def detached_static(out, lat, ok, trial_id, t_ms, *, keep=None, detach_out_px: float = DETACH_OUT_PX,
                    spread_px: float = STATIC_SPREAD_PX, max_gap_ms: float = DETACH_MAX_GAP_MS) -> np.ndarray:
    """``bool[n]``: frames of a detached, stationary run (`tongue_runs`; see ``DETACH_OUT_PX``). ``ok`` = confident raw
    frames; ``keep`` = frames another camera vouches for (`keep_frames`): a run touching one is not masked."""
    R = tongue_runs(out, lat, ok, trial_id, t_ms, max_gap_ms)
    mask = np.zeros(len(np.asarray(out)), bool)
    okf = np.asarray(ok, bool)
    kp = np.zeros(len(okf), bool) if keep is None else np.asarray(keep, bool)
    for r in R[(R.min_out > detach_out_px) & (R.spread < spread_px)].itertuples():
        if not kp[r.a:r.b + 1].any():
            mask[r.a:r.b + 1] = okf[r.a:r.b + 1]
    return mask


def keep_frames(ix: pd.DataFrame, windows: pd.DataFrame) -> np.ndarray:
    """``bool[len(ix)]``: frames of ``ix`` (trial_id, t_ms) inside any window of ``windows`` (trial_id, on_ms, off_ms)."""
    tid, t = ix.trial_id.to_numpy(), ix.t_ms.to_numpy(float)
    out = np.zeros(len(t), bool)
    for tr, g in windows.groupby("trial_id"):
        k = np.flatnonzero(tid == tr)
        if len(k):
            w = g[["on_ms", "off_ms"]].to_numpy(float)
            out[k] = ((t[k, None] >= w[None, :, 0]) & (t[k, None] <= w[None, :, 1])).any(1)
    return out


def cam4_lick_windows(rv, animal: str, date: str) -> pd.DataFrame | None:
    """(trial_id, on_ms, off_ms) of every production cam4 LP lick of the session (``licks_all.csv``, the table
    `lp_licks` reads), or None when the session has none."""
    from pathlib import Path

    from wfield_local import lp_licks
    _sd, f4, _f1 = lp_licks._inputs(f"{animal}_{date[4:]}", rv)
    return pd.read_csv(f4, usecols=["trial_id", "on_ms", "off_ms"]) if Path(f4).exists() else None


_NO_KEEP: set = set()


def mask_detached(flat: pd.DataFrame, ix: pd.DataFrame, to_frame=None, keep_windows: pd.DataFrame | None = None,
                  label: str | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """(copy of ``flat`` with ``tongue_likelihood`` = 0 on `detached_static` frames, the mask): the cam1 tongue as the
    cleaning should see it (a masked frame is "not visible"). ``to_frame``: `cam1_frame`'s, else computed.
    ``keep_windows``: the cam4 lick windows (`cam4_lick_windows`); runs overlapping one are kept. Without them every
    detached, stationary run is masked, printed once per ``label``."""
    from wfield_local import orofacial_clean as OC
    to_frame = to_frame or cam1_frame(flat, ix)[2]
    out, lat = to_frame(flat.tongue_x.to_numpy(float), flat.tongue_y.to_numpy(float))
    ok = flat.tongue_likelihood.to_numpy(float) >= float(OC.params("tongue", "cam1")["lk_thr"])
    if keep_windows is None:
        key = label or "<unlabelled cam1 session>"
        if key not in _NO_KEEP:
            _NO_KEEP.add(key)
            print(f"[cam1_licks] {key}: no cam4 lick windows -> detached runs masked without the cam4 keep", flush=True)
        keep = None
    else:
        keep = keep_frames(ix, keep_windows)
    m = detached_static(out, lat, ok, ix.trial_id.to_numpy(), ix.t_ms.to_numpy(float), keep=keep)
    flat = flat.copy()
    flat.loc[m, "tongue_likelihood"] = 0.0
    return flat, m
