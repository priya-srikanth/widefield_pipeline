"""Clean tongue / jaw traces from DLC or Lightning Pose output: drop outliers and low-confidence frames, fill
short gaps, subtract a baseline, cut cue-aligned trials.

    python -m wfield_local.orofacial_clean <pose.csv> --session PS93:20260908            # per-frame .npz
    python -m wfield_local.orofacial_clean <pose.csv> --session PS93:20260908 --windows <index.csv>

PORTED 2026-10-01 (Priya: "port all relevant code from the stroke_orofacial_pipeline ... to make clean traces
with appropriate interpolation and dropping of outliers / low confidence frames"). Sources, read-only in
`../stroke_orofacial_pipeline/src/stroke_orofacial/dlc_kinematics/`:

  interpolation.py  "v5p3"  -> `clean_bodypart`. Six stages, transcribed function for function:
     1 x-range   NaN frames outside median(x of confident frames) +- x_range_pm      (`_stage_xrange`)
     2 lk gate   NaN frames below lk_thr                                            (`_stage_thr`)
     3 isolated  NaN confident frames with < n_neigh confident neighbours in +- neighbor_ms (`_stage_drop`)
     4 gap fill  PCHIP (linear fallback) over NaN runs <= max_gap_ms, bracketed by confident frames,
                 n_side context points each side                                  (`_interp_small_gaps`)
     5 baseline fill  remaining NaN -> (X0, Y0)
     6 subtract  x_final = x - X0, y_final = y - Y0
  jaw_cleanup_v34.py "v3.4" -> `jaw_v34`. On the interpolated jaw, at RAW frames only: Y-ceiling (positive
     side), bidirectional X-ceiling, frame outliers vs a local median across raw frames, isolated short raw
     clusters; then re-PCHIP x AND y from the surviving anchors over gaps <= 120 ms, baseline beyond.
  mean_sem_visual.py -> `snippet`, `stack`, `mean_sem` (NaN-padded windows; SEM over finite count).

WHAT CHANGED, ON PURPOSE (each also noted at its site):
  * Input is any DLC-format pose CSV/h5 (DLC round 3, Lightning Pose); cue frames come from OUR alignment
    templates (cam frame <-> DAQ) and the DAQ-derived trial table, not wavesurfer `sig_camIdx__idx_ws`.
  * Baseline: data-driven (their `force: false` path) -- they used per-animal fixed X0/Y0 from animals.yaml,
    which do not exist for this rig. Same pool rule: lowest 5 % of y among lk >= 0.95 frames 0-5 s after cues.
  * No centering step. Theirs subtracted a reference (spout midpoint for tongue, eye midpoint for jaw) before
    stage 1; stage 1 uses the bodypart's OWN median and stage 6 subtracts X0/Y0, so a constant centre cancels.
    A task/anatomical reference frame (spout axes, nose) belongs in a later geometry step, not in cleaning.
  * lk_thr 0.6 (our `dlc.train.pcutoff`) instead of their 0.5.
  * NOT YET PORTED: the tongue "v7 pre-clean" (`_preclean.py`: cluster classifier + x-side rules, 646 lines,
    thresholds tuned on old-rig tongue shapes) and the per-trial lick detector / angle features built on it.
    Port those once the px thresholds here are retuned on our view, since they sit on top of these traces.

LIGHTNING POSE NOTE: an LP point below its cutoff is the image centre (~338, 338 px), not a position (a flat
"occluded" heatmap's soft-argmax). Stage 2 removes it; never feed LP coordinates past this module unmasked.

Every px threshold in `orofacial_clean` (configs/defaults.yaml) is the OLD rig's and must be re-measured.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter

from wfield_local import config

FILL_NONE, FILL_PCHIP, FILL_LINEAR, FILL_BASELINE = 0, 1, 2, 3


def params(bodypart: str) -> dict:
    """`orofacial_clean.<bodypart>` merged with the shared keys (method, fallback, baseline)."""
    c = config.defaults()["orofacial_clean"]
    return {**c[bodypart], "interpolation_method": c["interpolation_method"],
            "interpolation_fallback": c["interpolation_fallback"], "baseline": c["baseline"]}


# --------------------------------------------------------------------------- input

def read_pose(path: Path) -> pd.DataFrame:
    """DLC-format pose file (3-row header CSV, or DLC .h5) -> flat ``{part}_x/_y/_likelihood`` columns.

    Same flattening as their `dlc_io.loaders.read_dlc`. Works for DLC and Lightning Pose output alike.
    """
    path = Path(path)
    d = pd.read_hdf(path) if path.suffix in (".h5", ".hdf5") else pd.read_csv(path, header=[0, 1, 2], index_col=0)
    lv = d.columns.nlevels
    parts, coords = d.columns.get_level_values(lv - 2), d.columns.get_level_values(lv - 1)
    out = pd.DataFrame({f"{p}_{c}": d.iloc[:, k].to_numpy(float) for k, (p, c) in enumerate(zip(parts, coords))})
    return out


def cue_frames(tpl: dict, cue_s) -> np.ndarray:
    """Camera frame (float) of each DAQ cue time, via the alignment template (same affine as `dlc_frames.frame_of`)."""
    fs = float(tpl["fs_daq"])
    return (np.asarray(cue_s, float) * fs - float(tpl["intercept_daqSample"])) / float(tpl["slope_daqSample_per_camFrame"])


# --------------------------------------------------------------------------- v5p3 stages (transcribed)

def _stage_xrange(x, y, lk, *, x_range_pm, x_range_source, lk_thr):
    finite_xy = np.isfinite(x) & np.isfinite(y)
    if x_range_source == "lk":
        m = finite_xy & np.isfinite(lk) & (lk >= lk_thr)
    elif x_range_source == "finite":
        m = finite_xy
    else:
        raise ValueError(f"x_range_source={x_range_source!r}")
    if not m.any():
        m = finite_xy
    xo, yo = x.astype(float).copy(), y.astype(float).copy()
    if not m.any():
        return xo, yo
    med = float(np.nanmedian(x[m]))
    bad = ~np.isfinite(xo) | (xo < med - x_range_pm) | (xo > med + x_range_pm)
    xo[bad] = np.nan
    yo[bad] = np.nan
    return xo, yo


def _stage_thr(x, y, lk, *, lk_thr):
    hi = np.isfinite(lk) & (lk >= lk_thr) & np.isfinite(x) & np.isfinite(y)
    xo, yo = x.astype(float).copy(), y.astype(float).copy()
    xo[~hi] = np.nan
    yo[~hi] = np.nan
    return xo, yo


def _stage_drop(x, y, *, n_neigh, neighbor_f):
    n = len(x)
    hi_idx = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    keep = np.zeros(n, bool)
    j0 = j1 = 0
    for ii in hi_idx:
        while j0 < hi_idx.size and hi_idx[j0] < ii - neighbor_f:
            j0 += 1
        while j1 < hi_idx.size and hi_idx[j1] <= ii + neighbor_f:
            j1 += 1
        if (j1 - j0 - 1) >= n_neigh:            # the point itself is always in its window
            keep[ii] = True
    xo, yo = np.full(n, np.nan), np.full(n, np.nan)
    xo[keep], yo[keep] = x[keep], y[keep]
    return xo, yo


def _interp_small_gaps(arr, hi_mask, max_gap_f, *, require_bracketed_by_hi, method, n_side,
                       min_total_context, fallback):
    out = arr.astype(float).copy()
    codes = np.zeros(len(out), np.int8)
    n = len(out)

    def context(le, re_):
        lp, i = [], le
        while i >= 0 and len(lp) < n_side:
            if hi_mask[i] and np.isfinite(out[i]):
                lp.append(i)
            i -= 1
        rp, i = [], re_
        while i < n and len(rp) < n_side:
            if hi_mask[i] and np.isfinite(out[i]):
                rp.append(i)
            i += 1
        return np.unique(np.array(lp[::-1] + rp, dtype=int))

    def linear(i_, j_, le, re_):
        out[i_:j_] = np.interp(np.arange(i_, j_), [le, re_], [out[le], out[re_]])
        codes[i_:j_] = FILL_LINEAR

    i = 0
    while i < n:
        if np.isfinite(out[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isfinite(out[j]):
            j += 1
        le, re_ = i - 1, j
        ok = (j - i) <= max_gap_f and le >= 0 and re_ < n and np.isfinite(out[le]) and np.isfinite(out[re_])
        if not ok or (require_bracketed_by_hi and not (hi_mask[le] and hi_mask[re_])):
            i = j
            continue
        if method == "linear":
            linear(i, j, le, re_)
        elif method == "pchip":
            ctx = context(le, re_)
            if le not in ctx or re_ not in ctx:
                ctx = np.unique(np.r_[ctx, [le, re_]]).astype(int)
            xc, yc = ctx.astype(float), out[ctx].astype(float)
            g = np.isfinite(yc)
            xc, yc = xc[g], yc[g]
            if xc.size < min_total_context or np.unique(xc).size < 2:
                if fallback == "linear":
                    linear(i, j, le, re_)
            else:
                try:
                    out[i:j] = PchipInterpolator(xc, yc, extrapolate=False)(np.arange(i, j, dtype=float))
                    codes[i:j] = FILL_PCHIP
                except Exception:                          # noqa: BLE001 -- theirs: fall back, never fail
                    if fallback == "linear":
                        linear(i, j, le, re_)
        else:
            raise ValueError(f"method={method!r}")
        i = j
    return out, codes


def pool_mask(n_frames: int, centers, fps: float, win_ms=(0.0, 5000.0)) -> np.ndarray:
    """Frames within ``win_ms`` of any cue frame (rounded), OR-ed -- their `_build_tone_aligned_pool_mask`."""
    w0, w1 = int(np.floor(win_ms[0] / 1000 * fps)), int(np.ceil(win_ms[1] / 1000 * fps))
    c = np.round(np.asarray(centers, float)).astype(int)
    m = np.zeros(n_frames, bool)
    for ci in c[(c >= 0) & (c < n_frames)]:
        lo, hi = max(0, ci + w0), min(n_frames - 1, ci + w1)
        if hi >= lo:
            m[lo:hi + 1] = True
    return m


def baseline_data_driven(x, y, lk, pool, *, y0_lk_thr, y0_percentile):
    """Y0 = median y of the lowest-percentile-y subset of confident pool frames; X0 = median x of the SAME subset.
    "Lowest y" = highest in the image (y grows downward): the tongue retracted / the jaw closed."""
    base = pool & np.isfinite(y) & np.isfinite(x) & np.isfinite(lk) & (lk >= y0_lk_thr)
    if not base.any():
        return float("nan"), float("nan")
    cut = float(np.nanpercentile(y[base], y0_percentile))
    low = base & (y <= cut)
    if not low.any():
        k = max(10, int(np.ceil(y0_percentile / 100 * base.sum())))
        idx = np.flatnonzero(base)[np.argsort(y[base])[:k]]
        low = np.zeros_like(base)
        low[idx] = True
    return float(np.nanmedian(x[low])), float(np.nanmedian(y[low]))


@dataclass
class Clean:
    """Per-frame stages + provenance for one bodypart (their `InterpolationResult`, minus wavesurfer fields)."""
    bodypart: str
    x_raw: np.ndarray
    y_raw: np.ndarray
    lk: np.ndarray
    x_drop: np.ndarray
    y_drop: np.ndarray
    x_interp: np.ndarray
    y_interp: np.ndarray
    x_final: np.ndarray
    y_final: np.ndarray
    fill_method: np.ndarray
    X0: float
    Y0: float
    fps: float
    extra: dict = field(default_factory=dict)

    @property
    def x_masked(self) -> np.ndarray:
        """x_final with baseline-filled frames as NaN -- what to PLOT or AVERAGE (theirs plotted the 0 fill)."""
        return np.where(self.fill_method == FILL_BASELINE, np.nan, self.x_final)

    @property
    def y_masked(self) -> np.ndarray:
        return np.where(self.fill_method == FILL_BASELINE, np.nan, self.y_final)


def clean_bodypart(df: pd.DataFrame, bodypart: str, centers, fps: float = 250.0, p: dict | None = None,
                   x0y0: tuple[float, float] | None = None) -> Clean:
    """v5p3 on one session (or one concatenation of windows -- see `clean_windows`)."""
    p = p or params(bodypart)
    x, y, lk = (df[f"{bodypart}_{c}"].to_numpy(float) for c in ("x", "y", "likelihood"))
    xr, yr = _stage_xrange(x, y, lk, x_range_pm=float(p["x_range_pm"]), x_range_source=str(p["x_range_source"]),
                           lk_thr=float(p["lk_thr"]))
    xt, yt = _stage_thr(xr, yr, lk, lk_thr=float(p["lk_thr"]))
    xd, yd = _stage_drop(xt, yt, n_neigh=int(p["n_neigh"]),
                         neighbor_f=max(int(round(float(p["neighbor_ms"]) / 1000 * fps)), 1))
    max_gap_f = max(int(round(float(p["max_gap_ms"]) / 1000 * fps)), 1)
    hi = np.isfinite(xd) & np.isfinite(yd)
    kw = dict(require_bracketed_by_hi=bool(p["require_bracketed_by_hilk"]), method=str(p["interpolation_method"]),
              n_side=int(p["interp_n_side"]), min_total_context=int(p.get("interp_min_total_context", 6)),
              fallback=str(p["interpolation_fallback"]))
    xi, xf = _interp_small_gaps(xd, hi, max_gap_f, **kw)
    yi, yf = _interp_small_gaps(yd, hi, max_gap_f, **kw)
    fm = np.where(xf != FILL_NONE, xf, yf).astype(np.int8)
    b = p["baseline"]
    if x0y0 is not None:
        X0, Y0 = map(float, x0y0)
    else:
        pm = pool_mask(len(x), centers, fps, tuple(b.get("pool_win_ms", (0.0, 5000.0))))
        X0, Y0 = baseline_data_driven(xd, yd, lk, pm, y0_lk_thr=float(b["y0_lk_thr"]),
                                      y0_percentile=float(b["y0_percentile"]))
        if not (np.isfinite(X0) and np.isfinite(Y0)):
            raise RuntimeError(f"{bodypart}: data-driven baseline pool is empty (no lk >= {b['y0_lk_thr']} frames "
                               f"within {b.get('pool_win_ms')} ms of a cue)")
    nan_left = ~(np.isfinite(xi) & np.isfinite(yi))
    fm[nan_left] = FILL_BASELINE
    xfill = np.where(np.isfinite(xi), xi, X0)
    yfill = np.where(np.isfinite(yi), yi, Y0)
    return Clean(bodypart, x, y, lk, xd, yd, xi, yi, xfill - X0, yfill - Y0, fm, X0, Y0, fps)


# --------------------------------------------------------------------------- jaw v3.4 (transcribed)

def _ms_f(ms: float, fps: float) -> int:
    return int(np.round(ms / 1000 * fps))


def _clusters(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


def _repchip(values, anchor_idx, n, max_gap_f):
    clean, fillable = np.zeros(n), np.zeros(n, bool)
    if anchor_idx.size >= 2:
        av = values[anchor_idx]
        v = PchipInterpolator(anchor_idx, av, extrapolate=False)(np.arange(n))
        prev = np.searchsorted(anchor_idx, np.arange(n), side="right") - 1
        nxt = prev + 1
        ok = (prev >= 0) & (nxt < len(anchor_idx))
        gap = np.where(ok, anchor_idx[np.clip(nxt, 0, len(anchor_idx) - 1)]
                       - anchor_idx[np.clip(prev, 0, len(anchor_idx) - 1)] - 1, np.iinfo(np.int64).max)
        fillable = ok & (gap <= max_gap_f) & np.isfinite(v)
        clean[fillable] = v[fillable]
        clean[anchor_idx] = av
    elif anchor_idx.size == 1:
        clean[anchor_idx] = values[anchor_idx]
    return clean, fillable


def jaw_session_baselines(x_final, y_final, centers, fps, win_ms=(-500.0, 0.0)):
    """(bmean_y, bmean_x): median over trials of the mean in ``win_ms`` around each cue (their v3.4 hoist)."""
    lo_f, hi_f, n = _ms_f(win_ms[0], fps), _ms_f(win_ms[1], fps), len(y_final)
    by, bx = [], []
    for c in np.round(np.asarray(centers, float)).astype(int):
        if not 0 <= c < n:
            continue
        sl = slice(max(c + lo_f, 0), min(c + hi_f + 1, n))
        yb, xb = y_final[sl][np.isfinite(y_final[sl])], x_final[sl][np.isfinite(x_final[sl])]
        if yb.size >= 2:
            by.append(float(np.mean(yb)))
        if xb.size >= 2:
            bx.append(float(np.mean(xb)))
    if not by:
        return None
    return float(np.median(by)), float(np.median(bx)) if bx else 0.0


def jaw_v34(c: Clean, centers, q: dict | None = None) -> Clean:
    """Apply the v3.4 cleanup to an interpolated jaw `Clean`; returns a new `Clean` (x/y_final re-PCHIP'd)."""
    q = q or params("jaw")["v34"]
    fps, n = c.fps, len(c.y_final)
    bl = jaw_session_baselines(c.x_final, c.y_final, centers, fps, tuple(q["baseline_win_ms"]))
    if bl is None:
        return c
    by, bx = bl
    raw = c.fill_method == FILL_NONE
    wy = raw & np.isfinite(c.y_final) & ((c.y_final - by) > float(q["y_ceiling_px"]))
    wx = raw & np.isfinite(c.x_final) & (np.abs(c.x_final - bx) > float(q["x_ceiling_px"]))
    after1 = raw & ~wy & ~wx
    wfo = np.zeros(n, bool)
    idx = np.flatnonzero(after1)
    if idx.size >= 3:
        win = 2 * max(_ms_f(float(q["frame_outlier_radius_ms"]), fps), 1) + 1
        dev = np.abs(c.y_final[idx] - median_filter(c.y_final[idx], size=win, mode="nearest"))
        wfo[idx[dev > float(q["frame_jump_thr_px"])]] = True
    cl = _clusters(raw & ~wy & ~wfo)                     # theirs: R2 clusters after R1-y + R3 (not R1-x)
    wiso = np.zeros(n, bool)
    rad = max(_ms_f(float(q["isolated_radius_ms"]), fps), 1)
    for k, (s, e) in enumerate(cl):
        if (e - s) > int(q["isolated_max_n"]):
            continue
        near = False
        for k2, (s2, e2) in enumerate(cl):
            if k2 == k or e2 <= s - rad:
                continue
            if s2 >= e + rad:
                break
            near = True
            break
        if not near:
            wiso[s:e] = True
    keep = raw & ~wy & ~wx & ~wfo & ~wiso
    anchors = np.flatnonzero(keep & np.isfinite(c.y_final))
    gap_f = max(_ms_f(float(q["max_gap_ms_pchip"]), fps), 1)
    yc, fillable = _repchip(c.y_final, anchors, n, gap_f)
    xc, _ = _repchip(c.x_final, anchors, n, gap_f)
    fm = np.full(n, FILL_BASELINE, np.int8)
    fm[fillable] = FILL_PCHIP
    fm[anchors] = FILL_NONE
    extra = {"v34_wiped_y": wy, "v34_wiped_x": wx, "v34_wiped_frame_outlier": wfo, "v34_wiped_isolated": wiso,
             "v34_bmean": (by, bx)}
    return Clean(c.bodypart, c.x_raw, c.y_raw, c.lk, c.x_drop, c.y_drop, c.x_interp, c.y_interp, xc, yc, fm,
                 c.X0, c.Y0, fps, {**c.extra, **extra})


# --------------------------------------------------------------------------- trials

def snippet(arr, c: int, pre_f: int, post_f: int) -> np.ndarray:
    """``arr[c-pre_f : c+post_f+1]``, NaN-padded out of range."""
    T, lo = pre_f + post_f + 1, c - pre_f
    out = np.full(T, np.nan)
    s0, s1 = max(lo, 0), min(c + post_f, len(arr) - 1)
    if s1 >= s0:
        out[s0 - lo: s0 - lo + (s1 - s0) + 1] = arr[s0:s1 + 1]
    return out


def stack(arr, centers, pre_f: int, post_f: int) -> np.ndarray:
    """(n_centers, T) of `snippet`."""
    if len(centers) == 0:
        return np.zeros((0, pre_f + post_f + 1))
    return np.stack([snippet(arr, int(round(c)), pre_f, post_f) for c in centers])


def mean_sem(A) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise NaN mean and SEM (std / sqrt(finite count)) -- their `_mean_sem`."""
    A = np.asarray(A, float)
    if A.size == 0:
        T = A.shape[1] if A.ndim == 2 else 0
        return np.full(T, np.nan), np.full(T, np.nan)
    d = np.sqrt(np.isfinite(A).sum(0))
    return np.nanmean(A, 0), np.nanstd(A, 0) / np.where(d == 0, np.nan, d)


def clean_windows(df: pd.DataFrame, index: pd.DataFrame, bodypart: str, fps: float = 250.0, sep: int = 300):
    """Clean a clip made of concatenated cue windows (`scripts.pose_cue_traces clip` index: trial_k, t_ms).

    Windows are joined with ``sep`` NaN frames (longer than any max gap, so nothing is filled across a window
    boundary) so session-level statistics (x-range median, baseline) pool over all trials, as on a full session.
    Returns (Clean on the padded array, row positions of the original frames, cue frame of each window).
    """
    parts, pos, cues, k = [], [], [], 0
    for _tk, g in index.groupby("trial_k", sort=True):
        rows = g.index.to_numpy()
        parts.append(df.iloc[rows].reset_index(drop=True))
        pos.append(np.arange(k, k + len(rows)))
        cues.append(k + int(np.argmin(np.abs(g.t_ms.to_numpy()))))
        k += len(rows)
        parts.append(pd.DataFrame(np.nan, index=range(sep), columns=df.columns))
        k += sep
    big = pd.concat(parts, ignore_index=True)
    c = clean_bodypart(big, bodypart, cues, fps)
    if bodypart == "jaw" and params("jaw")["v34"].get("enabled", True):
        c = jaw_v34(c, cues)
    return c, np.concatenate(pos), np.array(cues)


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pose", type=Path)
    ap.add_argument("--session", required=True, metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--cam", default="cam4")
    ap.add_argument("--windows", type=Path, default=None, help="index CSV if the pose file is a cue-window clip")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = a.session.split(":")
    df = read_pose(a.pose)
    res = {}
    if a.windows is not None:
        idx = pd.read_csv(a.windows).iloc[:len(df)]
        for bp in ("tongue", "jaw"):
            c, pos, _ = clean_windows(df, idx, bp)
            res[bp] = (c, pos)
    else:
        tpl = dict(np.load(Path(rv.root("alignment_templates")) / a.cam / animal / f"{date}.npz", allow_pickle=True))
        tr = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob(f"{animal}_{date}_*_trials.csv"))[-1]
        centers = cue_frames(tpl, pd.read_csv(tr)["cue_s"].dropna())
        for bp in ("tongue", "jaw"):
            c = clean_bodypart(df, bp, centers, float(tpl.get("fps_cam", 250.0)))
            if bp == "jaw" and params("jaw")["v34"].get("enabled", True):
                c = jaw_v34(c, centers)
            res[bp] = (c, np.arange(len(df)))
    out = a.out or a.pose.with_name(a.pose.stem + "_clean.npz")
    arrs = {}
    for bp, (c, pos) in res.items():
        for nm in ("x_raw", "y_raw", "lk", "x_final", "y_final", "fill_method"):
            arrs[f"{bp}_{nm}"] = getattr(c, nm)[pos]
        arrs[f"{bp}_X0Y0"] = np.array([c.X0, c.Y0])
        fm = c.fill_method[pos]
        print(f"{bp}: X0 {c.X0:.1f} Y0 {c.Y0:.1f} | raw {np.mean(fm == FILL_NONE) * 100:.1f}%  interp "
              f"{np.mean(np.isin(fm, (FILL_PCHIP, FILL_LINEAR))) * 100:.1f}%  baseline-fill {np.mean(fm == FILL_BASELINE) * 100:.1f}%")
    np.savez_compressed(out, **arrs)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
