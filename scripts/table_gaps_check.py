"""TABLE GAPS (Priya 11-21: "go ahead with your table gaps rule"): WHY do cam4 (front) and cam1 (below) have no lick
event for the side-only licks of the four-camera fusion, and does a TRACKING-based confirmation fix it?

The side-only guard check (`scripts/side_only_guard_check.py`, qc_phaseA/side_only_guard/) found that most real
side-only licks were TRACKED by cam1 / cam4 (tongue likelihood 1.0) but missing from their lick tables. This script
diagnoses EVERY side-only lick (all 14,696 of the existing fusion `camera_licks/lp_side_r2b_prod_20261118_3net_20261121`)
per camera, and writes the tables the README in ``qc_phaseA/table_gaps/`` reports.

READ ONLY on every existing output; writes only to ``<pose3d_4cam_3net_20261121>/qc_phaseA/table_gaps/``.

    python -m scripts.table_gaps_check diagnose [--jobs 6]   per side-only lick x camera: window, tracking, cleaning,
                                                             events (diag.csv.gz)
    python -m scripts.table_gaps_check summarize              cause tables by session / cause x blind verdict
    python -m scripts.table_gaps_check windows                head / tail outside the trial windows, per session x camera
    python -m scripts.table_gaps_check pick [--n 30]          blind sample of the licks the new rule CONFIRMS
    python -m scripts.table_gaps_check strips [--jobs 6]      blind strips of that sample (ids only)
    python -m scripts.table_gaps_check score                  join verdicts_blind_new.csv (written FIRST) to the rule
    python -m scripts.table_gaps_check compare                per session before / after (old vs gap-fixed fusion)

Camera naming: cam1 = below, cam4 = FRONT, cam2 / cam3 = side views.

CAUSES, per camera (cam4, cam1), first match wins (`classify`):
  d_no_table      the session has no lick table for that camera (PS93 0805: no behavior log, the chain skipped it)
  a_no_window     no frame of that camera inside the lick (outside the collected trial windows)
  e_cut_fragment  the camera HAS a kept event overlapping the lick (within `camera_licks.MERGE_TOL_MS`): the fusion's
                  split (`event_cuts`) put the side event's peak on the other side of a cut, so it became its own lick
  b_rule_dropped  cam1 only: an event overlaps but `cam1_licks.keep_mask` dropped it (droplet / far-from-mouth)
  c_not_tracked   fewer than `TRACK_MIN_FRAMES` frames with raw tongue likelihood >= the cleaning cutoff (0.4)
  b_detached      cam1 only: tracked, but the detached-static mask (`cam1_licks.mask_detached`, before cleaning) took it
  b_cleaning      tracked, but fewer than `TRACK_MIN_FRAMES` MEASURED frames survive the camera's cleaning
  b_min_frames    cam1 only: measured, but no run of >= `cam1_licks.MIN_FRAMES` consecutive present frames
  b_detector      cam4 only: measured, but the cam4 lick detector made no lick (sub-reason ``cam4_det``: small =
                  cleaned protrusion < the detector's ``min_peak_y_abs`` 20 px; near = a cam4 lick within 100 ms
                  (dedupe / minimum-interval rules); other)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

TAG = "_3net_20261121"
LICK_RUN = "lp_side_r2b_prod_20261118_3net_20261121"
V37 = "lp_r4_heatmap_20261013__clean_v37"
CAM1_DIR = "cam1_licks_v37"
#: "tracked" = at least this many frames inside the lick with raw tongue likelihood >= the cleaning cutoff
TRACK_MIN_FRAMES = 2
LK_CUT = 0.4                       # `orofacial_clean` tongue lk_thr, cam4 and cam1 alike
NEAR_MS = 100.0
CAM4_MIN_PEAK_PX = 20.0            # orofacial_kinematics.tongue.detector.min_peak_y_abs
CAMS2 = ("cam4", "cam1")


def _env() -> None:
    os.environ["WIDEFIELD_POSE3D_TAG"] = TAG
    os.environ["WIDEFIELD_POSE3D_SOURCE"] = "4cam"
    os.environ["WIDEFIELD_LICK_INPUT_RUN"] = V37
    os.environ["WIDEFIELD_CAM1_LICKS_DIR"] = CAM1_DIR
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


def _rv():
    from wfield_local.paths import PathResolver
    return PathResolver()


def dirs() -> dict:
    _env()
    from wfield_local import pose3d
    rv = _rv()
    d4 = pose3d.pose3d_dir(rv, source="4cam")
    assert d4.name == "pose3d_4cam_3net_20261121", d4
    out = d4 / "qc_phaseA" / "table_gaps"
    assert "MICROSCOPE" in str(out) and "Priya" in str(out), out
    dlc = Path(rv.root("dlc"))
    return {"4cam": d4, "out": out, "licks": dlc / "session_poses" / "camera_licks" / LICK_RUN,
            "guard": d4 / "qc_phaseA" / "side_only_guard", "week": d4 / "qc_phaseA" / "ps95_side_week",
            "v37": Path(rv.root("orofacial")) / V37 / "sessions", "cam1": dlc / "session_poses" / CAM1_DIR,
            "strips": out / "strips_new"}


# ------------------------------------------------------------------------------------------------ per-frame inputs
def _per_trial(tid: np.ndarray, t: np.ndarray) -> dict:
    """{trial_id: (sorted t, row index)}."""
    o = np.lexsort((t, tid))
    tid_s, t_s = tid[o], t[o]
    b = np.flatnonzero(np.r_[True, tid_s[1:] != tid_s[:-1], True])
    return {int(tid_s[a]): (t_s[a:c], o[a:c]) for a, c in zip(b[:-1], b[1:])}


def _rows(pt: dict, trial: int, on: float, off: float, pad: float = 2.0) -> np.ndarray:
    g = pt.get(int(trial))
    if g is None:
        return np.array([], int)
    t, r = g
    a, b = np.searchsorted(t, on - pad, "left"), np.searchsorted(t, off + pad, "right")
    return r[a:b]


def _frame_of(origin, ap_axis, lr_axis=None):
    """(out, lat) px of image points: out = toward the spouts (cam1's / cam4's mouth frame convention)."""
    o, ap = np.asarray(origin, float), np.asarray(ap_axis, float)
    lr = np.array([-ap[1], ap[0]]) if lr_axis is None else np.asarray(lr_axis, float)

    def f(x, y):
        dx, dy = np.asarray(x, float) - o[0], np.asarray(y, float) - o[1]
        return -(dx * ap[0] + dy * ap[1]), dx * lr[0] + dy * lr[1]
    return f


def cam4_frames(rv, an: str, date: str, sd: Path) -> dict | None:
    """cam4 per frame: trial_id, t_ms, raw tongue lk / out / spout distance, cleaned MEASURED (v37 clean.npz; None when
    the session has no v37 tables), and the v37 lick table. No v37 tables -> raw from the O2 bundle (tracking only)."""
    from wfield_local import spout_frame as SF
    if (sd / "windows.npz").exists():
        w = np.load(sd / "windows.npz", allow_pickle=False)
        parts = [str(p) for p in w["parts"]]
        pose = w["pose"]
        ix = pd.DataFrame({k[3:]: w[k] for k in ("ix_trial_id", "ix_t_ms", "ix_strobe_ms", "ix_position")})
        flat = pd.DataFrame({f"{p}_{c}": pose[:, k, j].astype(float) for k, p in enumerate(parts)
                             for j, c in enumerate(("x", "y", "likelihood"))})
        z = np.load(sd / "clean.npz", allow_pickle=False)
        meas = (z["tongue_fill"] == 0) & np.isfinite(z["tongue_x"])
        origin, ap = z["tongue_origin"], z["spout_ap_axis"]
        L = pd.read_csv(sd / "licks_all.csv", usecols=["trial_id", "on_ms", "off_ms", "t_ms"])
    else:
        from scripts.pose3d import PROD
        from wfield_local import o2_inference as oi
        got = oi.session_windows(rv, PROD["cam4"], an, date, "cam4")
        if got is None:
            return None
        ix, flat = got
        from scripts.pose3d import _june_factor
        f = _june_factor(date) or 1.0
        if f != 1.0:
            for p in ("tongue", "spout"):
                flat[f"{p}_x"], flat[f"{p}_y"] = flat[f"{p}_x"] * f, flat[f"{p}_y"] * f
        fr = SF.from_medians(SF.ruler_medians(flat, ix))
        origin, ap = fr.origin, fr.ap_axis
        meas, L = None, None
    med = SF.ruler_medians(flat, ix)
    to = _frame_of(origin, ap)
    x, y = flat.tongue_x.to_numpy(float), flat.tongue_y.to_numpy(float)
    out, lat = to(x, y)
    sp = np.array([med[p][:2] for p in med], float)
    sd_px = np.min(np.hypot(x[:, None] - sp[None, :, 0], y[:, None] - sp[None, :, 1]), axis=1) if len(sp) else \
        np.full(len(x), np.nan)
    return {"trial_id": ix.trial_id.to_numpy(), "t_ms": ix.t_ms.to_numpy(float),
            "lk": flat.tongue_likelihood.to_numpy(float), "out": out, "spout_d": sd_px, "meas": meas,
            "present": meas, "masked": None, "L": L}


def cam1_frames(rv, an: str, date: str, cam1_dir: Path) -> dict | None:
    """cam1 per frame exactly as `scripts.cam1_licks.run_session` builds its detector input: raw tongue, the
    detached-static mask (keep = the v37 cam4 lick windows), the cam1 cleaning; + the cam1 events_all table."""
    from scripts.pose3d import PROD
    from wfield_local import cam1_licks as C1
    from wfield_local import o2_inference as oi
    from wfield_local import orofacial_clean as oc
    c1 = oi.session_windows(rv, PROD["cam1"], an, date, "cam1")
    if c1 is None:
        return None
    ix, flat = c1[0].reset_index(drop=True), c1[1].reset_index(drop=True)
    fr, med, to_frame = C1.cam1_frame(flat, ix)
    flat_m, mask = C1.mask_detached(flat, ix, to_frame, keep_windows=C1.cam4_lick_windows(rv, an, date),
                                    label=f"{an}_{date}")
    cols = [c for c in flat_m.columns if c.startswith(("tongue_", "jaw_"))]
    c, pos, _ = oc.clean_windows(flat_m[cols], ix, "tongue", x0y0=tuple(fr.origin), cam="cam1")
    fm = c.fill_method[pos]
    x, y = flat.tongue_x.to_numpy(float), flat.tongue_y.to_numpy(float)
    out, _lat = to_frame(x, y)
    sp = np.array([med[p][:2] for p in med], float)
    sd_px = np.min(np.hypot(x[:, None] - sp[None, :, 0], y[:, None] - sp[None, :, 1]), axis=1)
    f1 = cam1_dir / f"{an}_{date}_events_all.csv"
    E = None
    if f1.exists():
        E = pd.read_csv(f1)
        E["keep"] = C1.keep_mask(E, label=f"{an}_{date}")
        E = E[["trial_id", "on_ms", "off_ms", "t_peak_ms", "keep"]]
    src = ix.src_frame.to_numpy()
    return {"trial_id": ix.trial_id.to_numpy(), "t_ms": ix.t_ms.to_numpy(float),
            "lk": flat.tongue_likelihood.to_numpy(float), "out": out, "spout_d": sd_px,
            "meas": (fm == oc.FILL_NONE) & np.isfinite(c.x_final[pos]), "present": fm != oc.FILL_BASELINE,
            "masked": mask, "src": src, "L": E}


def _max_run(rows: np.ndarray, ok: np.ndarray, src: np.ndarray | None) -> int:
    """Longest run of consecutive ``ok`` frames among ``rows`` (consecutive video frames when ``src`` is given)."""
    if not len(rows):
        return 0
    o = ok[rows]
    nxt = (np.diff(src[rows]) == 1) if src is not None else np.ones(len(rows) - 1, bool)
    best = cur = 0
    for i in range(len(rows)):
        if o[i] and (i == 0 or cur == 0 or nxt[i - 1]):
            cur += 1
        elif o[i]:
            cur = 1
        else:
            cur = 0
        best = max(best, cur)
    return best


def _events_near(L: pd.DataFrame | None, trial: int, on: float, off: float, tol: float, keep_only: bool):
    """(overlapping within tol, gap ms to the nearest event of the trial) for event table ``L``."""
    if L is None:
        return False, np.nan
    g = L[L.trial_id == trial]
    if keep_only and "keep" in g:
        g = g[g.keep.astype(bool)]
    if not len(g):
        return False, np.nan
    a, b = g.on_ms.to_numpy(float), g.off_ms.to_numpy(float)
    gap = np.maximum(0.0, np.maximum(a - off, on - b))
    return bool((gap <= tol).any()), float(gap.min())


def diag_worker(sid: str) -> pd.DataFrame:
    _env()
    from wfield_local import camera_licks as CL
    D = dirs()
    rv = _rv()
    an, date = sid.split("_")
    A = pd.read_csv(D["guard"] / "side_only_all.csv.gz")
    A = A[A.sid == sid].reset_index(drop=True)
    if not len(A):
        return A
    sd = D["v37"] / an / date
    S = pd.read_csv(D["4cam"] / f"{sid}_frames.csv.gz",
                    usecols=["trial_id", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv", "tongue_views",
                             "tongue_rep_cam1", "tongue_rep_cam4", "tongue_rep_cam2", "tongue_rep_cam3"])
    ptS = _per_trial(S.trial_id.to_numpy(), S.t_ms.to_numpy(float))
    fr = {"cam4": cam4_frames(rv, an, date, sd), "cam1": cam1_frames(rv, an, date, D["cam1"])}
    pt = {c: (_per_trial(f["trial_id"], f["t_ms"]) if f is not None else {}) for c, f in fr.items()}
    have_table = {"cam4": (sd / "licks_all.csv").exists(), "cam1": (D["cam1"] / f"{sid}_events_all.csv").exists()}
    rows = []
    X3 = S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    for r in A.itertuples():
        on, off, tr = float(r.on_ms), float(r.off_ms), int(r.trial_id)
        row = {"ex_id": r.ex_id}
        k3 = _rows(ptS, tr, on, off)
        row["n3d_frames"] = len(k3)
        for c in CAMS2:
            f = fr[c]
            row[f"{c}_table"] = have_table[c]
            if f is None:
                row[f"{c}_n_win"] = 0
                continue
            k = _rows(pt[c], tr, on, off)
            row[f"{c}_n_win"] = len(k)
            hi = f["lk"][k] >= LK_CUT
            row[f"{c}_n_raw"] = int(hi.sum())
            row[f"{c}_lk_max"] = float(np.nanmax(f["lk"][k])) if len(k) else np.nan
            row[f"{c}_out_med_raw"] = float(np.median(f["out"][k][hi])) if hi.any() else np.nan
            row[f"{c}_out_max_raw"] = float(np.max(f["out"][k][hi])) if hi.any() else np.nan
            row[f"{c}_spout_d_min_raw"] = float(np.min(f["spout_d"][k][hi])) if hi.any() else np.nan
            if f["masked"] is not None:
                row[f"{c}_n_masked"] = int((hi & f["masked"][k]).sum())
            if f["meas"] is not None:
                row[f"{c}_n_meas"] = int(f["meas"][k].sum())
                row[f"{c}_max_run_present"] = _max_run(k, f["present"], f.get("src"))
            if c == "cam4" and f["meas"] is not None and fr["cam4"] is not None:
                m = f["meas"][k]
                row["cam4_out_max_meas"] = float(np.max(f["out"][k][m])) if m.any() else np.nan
            row[f"{c}_n_used3d"] = int(np.isfinite(S[f"tongue_rep_{c}"].to_numpy(float)[k3]).sum()) if len(k3) else 0
            ov, gap = _events_near(f["L"], tr, on, off, CL.MERGE_TOL_MS, keep_only=True)
            row[f"{c}_ev_overlap"], row[f"{c}_ev_gap_ms"] = ov, gap
            if c == "cam1":
                ova, _g = _events_near(f["L"], tr, on, off, CL.MERGE_TOL_MS, keep_only=False)
                row["cam1_ev_overlap_any"] = ova
        if len(k3):
            u = np.isfinite(X3[k3]).all(1)
            row["n3d_tongue"] = int(u.sum())
            if u.any():
                d = np.linalg.norm(X3[k3][u], axis=1)
                j = int(np.argmax(d))
                row["reach3d_max_mm"] = float(d[j])
                row["tip3d_ap"], row["tip3d_dv"], row["tip3d_ml"] = (float(X3[k3][u][j, 1]), float(X3[k3][u][j, 2]),
                                                                     float(X3[k3][u][j, 0]))
        rows.append(row)
    return pd.DataFrame(rows)


def classify(R: pd.DataFrame, cam: str) -> pd.Series:
    """The per-camera cause (module docstring), first match wins."""
    def col(name, default=np.nan):
        return R[name] if name in R else pd.Series(default, index=R.index)
    out = pd.Series("", index=R.index, dtype=object)
    rules = [("d_no_table", ~col(f"{cam}_table", False).astype(bool)),
             ("a_no_window", col(f"{cam}_n_win", 0).fillna(0) == 0),
             ("e_cut_fragment", col(f"{cam}_ev_overlap", False).fillna(False).astype(bool))]
    if cam == "cam1":
        rules.append(("b_rule_dropped", col("cam1_ev_overlap_any", False).fillna(False).astype(bool)))
    rules.append(("c_not_tracked", col(f"{cam}_n_raw", 0).fillna(0) < TRACK_MIN_FRAMES))
    if cam == "cam1":
        rules.append(("b_detached", (col("cam1_n_raw", 0) - col("cam1_n_masked", 0)).fillna(0) < TRACK_MIN_FRAMES))
    rules.append(("b_cleaning", col(f"{cam}_n_meas", 0).fillna(0) < TRACK_MIN_FRAMES))
    if cam == "cam1":
        from wfield_local import cam1_licks as C1
        rules.append(("b_min_frames", col("cam1_max_run_present", 0).fillna(0) < C1.MIN_FRAMES))
        rules.append(("b_other", pd.Series(True, index=R.index)))
    else:
        rules.append(("b_detector", pd.Series(True, index=R.index)))
    for name, m in rules:
        out[(out == "") & m.to_numpy()] = name
    return out


def cam4_detector_sub(R: pd.DataFrame) -> pd.Series:
    small = R.cam4_out_max_meas < CAM4_MIN_PEAK_PX
    near = R.cam4_ev_gap_ms <= NEAR_MS
    return pd.Series(np.where(small, "small", np.where(near, "near", "other")), index=R.index)


#: the per-lick cause, combining the two cameras (first match wins)
def combined(R: pd.DataFrame) -> pd.Series:
    c4, c1 = R.cause_cam4, R.cause_cam1
    tracked = lambda c: c.str.startswith("b_")                       # noqa: E731
    out = np.select(
        [(c4 == "d_no_table") & (c1 == "d_no_table"),
         (c4 == "a_no_window") & (c1 == "a_no_window"),
         (c4 == "e_cut_fragment") | (c1 == "e_cut_fragment"),
         tracked(c4) | tracked(c1),
         (c4 == "c_not_tracked") & (c1 == "c_not_tracked")],
        ["d_no_table", "a_no_window", "e_cut_fragment", "b_tracked_not_called", "c_not_tracked"], "mixed")
    return pd.Series(out, index=R.index)


def diagnose(jobs: int) -> None:
    from wfield_local import parallel
    D = dirs()
    D["out"].mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(D["guard"] / "side_only_all.csv.gz", usecols=["sid"])
    sids = sorted(A.sid.unique())
    res, fail = parallel.fan_out(sids, diag_worker, jobs=jobs, label="session")
    for s, e in fail:
        print("FAILED", s, e)
    R = pd.concat([r for _s, r in sorted(res, key=lambda x: x[0]) if len(r)], ignore_index=True)
    R.to_csv(D["out"] / "diag.csv.gz", index=False, float_format="%.4g")
    print(len(R), "side-only licks diagnosed over", len(res), "sessions")


def _verdicts(D: dict) -> pd.DataFrame:
    """Blind verdicts of the side-only guard sample (62 judged) + the 41 PS95 side-week examples, by ex_id."""
    V = pd.read_csv(D["guard"] / "review.csv")[["ex_id", "verdict", "fragment"]].assign(set="guard_blind")
    W = pd.read_csv(D["week"] / "examples_review.csv")[["ex_id", "verdict"]].assign(set="side_week", fragment=False)
    V = pd.concat([V, W], ignore_index=True)
    V = V[V.verdict.isin(["real", "paw", "grooming", "water_drop", "other", "other_label"])]
    V["real"] = (V.verdict == "real") & ~V.fragment.astype(bool)
    return V


def load_diag(D: dict) -> pd.DataFrame:
    A = pd.read_csv(D["guard"] / "side_only_all.csv.gz")
    R = pd.read_csv(D["out"] / "diag.csv.gz")
    R = A.merge(R, on="ex_id", how="left")
    for c in CAMS2:
        R[f"cause_{c}"] = classify(R, c)
    R["cam4_det"] = np.where(R.cause_cam4 == "b_detector", cam4_detector_sub(R), "")
    R["cause"] = combined(R)
    return R


def summarize() -> None:
    D = dirs()
    R = load_diag(D)
    V = _verdicts(D)
    R = R.merge(V, on="ex_id", how="left")
    out = D["out"]
    pd.crosstab(R.sid, R.cause, margins=True).to_csv(out / "cause_by_session.csv")
    for c in CAMS2:
        pd.crosstab(R.sid, R[f"cause_{c}"], margins=True).to_csv(out / f"cause_{c}_by_session.csv")
    T = pd.crosstab(R.cause_cam4, R.cause_cam1, margins=True)
    T.to_csv(out / "cause_cam4_x_cam1.csv")
    J = R[R.verdict.notna()]
    VT = J.assign(v=np.where(J.real, "real", np.where(J.verdict == "real", "fragment", J.verdict)))
    pd.crosstab(VT.cause, VT.v, margins=True).to_csv(out / "cause_x_verdict.csv")
    pd.crosstab([VT.cause_cam4, VT.cause_cam1], VT.v, margins=True).to_csv(out / "cause_cams_x_verdict.csv")
    R.drop(columns=[c for c in ("verdict", "fragment", "set", "real") if c in R]).to_csv(
        out / "side_only_causes.csv.gz", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print("combined cause:\n", R.cause.value_counts().to_string())
    print("\nexcluding PS93 0805:\n", R[R.sid != "PS93_20260805"].cause.value_counts().to_string())
    for c in CAMS2:
        print(f"\n{c}:\n", R[f"cause_{c}"].value_counts().to_string())
    print("\ncam4 detector sub:\n", R[R.cause_cam4 == "b_detector"].cam4_det.value_counts().to_string())
    print("\ncam4 x cam1:\n", T.to_string())
    print("\ncause x verdict:\n", pd.crosstab(VT.cause, VT.v, margins=True).to_string())
    print("\ncam4/cam1 cause x verdict:\n", pd.crosstab([VT.cause_cam4, VT.cause_cam1], VT.v).to_string())


# ------------------------------------------------------------------------------------- (a) window coverage
def windows_worker(sid: str) -> dict:
    """cam4 v37 trial windows vs the session's sync span (first .. last DAQ sync pulse): seconds of the span outside
    every window -- head (first pulse .. first window), tail (last window .. last pulse), ITI gaps between windows."""
    _env()
    from wfield_local import o2_inference as oi
    D = dirs()
    an, date = sid.split("_")
    sd = D["v37"] / an / date
    row = {"sid": sid}
    if not (sd / "windows.npz").exists():
        return {**row, "error": "no v37 windows"}
    w = np.load(sd / "windows.npz", allow_pickle=False)
    d = np.sort(w["ix_daq_s"][np.isfinite(w["ix_daq_s"])])
    e = oi.daq_sync_edges_s(_rv(), an, date)
    a, b = float(e[0]), float(e[-1])
    inside = d[(d >= a) & (d <= b)]
    steps = np.diff(inside)
    gap = float(steps[steps > 1.5 / 249.81].sum() - (steps > 1.5 / 249.81).sum() / 249.81)
    return {**row, "sync_first_s": a, "sync_last_s": b, "span_s": b - a, "win_first_s": float(d[0]),
            "win_last_s": float(d[-1]), "head_s": max(0.0, float(d[0]) - a), "tail_s": max(0.0, b - float(d[-1])),
            "gaps_s": gap, "before_sync_s": max(0.0, a - float(d[0])), "after_sync_s": max(0.0, float(d[-1]) - b)}


def windows(jobs: int) -> None:
    from wfield_local import parallel
    D = dirs()
    sids = sorted(f"{p.parent.name}_{p.name}" for p in D["v37"].glob("PS*/2026*") if p.is_dir())
    res, fail = parallel.fan_out(sids, windows_worker, jobs=jobs, label="session")
    for s_, e in fail:
        print("FAILED", s_, e)
    W = pd.DataFrame([r for _s, r in sorted(res, key=lambda x: x[0])])
    W["uncovered_s"] = W[["head_s", "tail_s", "gaps_s"]].sum(1)
    W.to_csv(D["out"] / "window_coverage.csv", index=False, float_format="%.3f")
    print(W[["head_s", "tail_s", "gaps_s", "uncovered_s", "before_sync_s", "after_sync_s"]].describe().round(2))
    print(W.sort_values("uncovered_s").tail(8).to_string())


# --------------------------------------------------------------------------- blind check of the newly confirmed licks
GAPFIX = LICK_RUN + "_gapfix"
NEW_SEED = 2
N_NEW = 30


def _gapfix_dir() -> Path:
    return dirs()["licks"].parent / GAPFIX


def _blind_dirs() -> dict:
    """`side_only_guard_check.strip_worker`'s folders, pointed at ``table_gaps/blind_new`` (same strip drawing)."""
    D = dirs()
    out = D["out"] / "blind_new"
    return {**D, "out": out, "strips": out / "strips"}


def pick(n: int = N_NEW) -> None:
    """Blind sample of the licks the rule CONFIRMS by tracking (reason tracked_cam4 / tracked_cam1) in the gap-fixed
    fusion, round robin over animals, minus every lick already judged and minus PS93 0805 (its tables are rebuilt)."""
    from wfield_local import camera_licks as CL
    D = dirs()
    seen = set(_verdicts(D).ex_id)
    rows = []
    for f in sorted(_gapfix_dir().glob("PS*_fused.csv")):
        U = pd.read_csv(f)
        if "reason" not in U:
            continue
        sid = f.name[:13]
        # the FIRST version's confirmed-by-tracking set (side-only, not a fragment / guarded, cam4 or cam1 tracked >= 2
        # frames): the 11-21 sample was drawn from that run, where these were reason tracked_cam4 / tracked_cam1
        trk = (U.tracked_cam4 >= CL.TRACK_MIN_FRAMES) | (U.tracked_cam1 >= CL.TRACK_MIN_FRAMES)
        U = U[CL.side_only_mask(U) & trk & ~U.reason.isin(["cut_fragment", "guarded"])].copy()
        U["sid"], U["animal"] = sid, sid[:4]
        rows.append(U)
    A = pd.concat(rows, ignore_index=True)
    A["ex_id"] = [f"{r.sid}_t{int(r.trial_id)}_{int(r.on_ms)}" for r in A.itertuples()]
    A = A[~A.ex_id.isin(seen) & (A.sid != "PS93_20260805")].reset_index(drop=True)
    rng = np.random.default_rng(NEW_SEED)
    pools = {a: list(rng.permutation(np.flatnonzero(A.animal == a))) for a in sorted(A.animal.unique())}
    idx = []
    while len(idx) < n and any(pools.values()):
        for a in sorted(pools):
            if pools[a] and len(idx) < n:
                idx.append(pools[a].pop())
    P = A.loc[idx].iloc[rng.permutation(len(idx))].reset_index(drop=True)
    P["blind_id"] = [f"N{i + 1:02d}" for i in range(len(P))]
    B = _blind_dirs()
    B["out"].mkdir(parents=True, exist_ok=True)
    P.to_csv(B["out"] / "sample_key.csv", index=False, float_format="%.4g")
    print(len(A), "confirmed-by-tracking licks eligible;", len(P), "picked")
    print(P.groupby(["animal", "reason"]).size().to_string())


def strip_worker_new(sid: str) -> list:
    from scripts import side_only_guard_check as G
    G.dirs = _blind_dirs                 # same strips, our sample / folder (spawned process: patched here)
    return G.strip_worker(sid)


def strips(jobs: int) -> None:
    from wfield_local import parallel
    B = _blind_dirs()
    P = pd.read_csv(B["out"] / "sample_key.csv")
    res, fail = parallel.fan_out(sorted(P.sid.unique()), strip_worker_new, jobs=jobs, label="session")
    if fail:
        print("FAILED:", fail)
    print("strips:", sum(len(x) for _s, x in res), "of", len(P))


def score() -> None:
    """``blind_new/verdicts_blind_new.csv`` (blind_id, verdict, note; written BY EYE before this runs) joined to the key."""
    B = _blind_dirs()
    P = pd.read_csv(B["out"] / "sample_key.csv")
    V = pd.read_csv(B["out"] / "verdicts_blind_new.csv")
    J = P.merge(V, on="blind_id", how="left")
    J.to_csv(B["out"] / "review_new.csv", index=False, float_format="%.4g")
    print(pd.crosstab(J.reason, J.verdict, margins=True).to_string())
    print(pd.crosstab(J.animal, J.verdict, margins=True).to_string())


def validate() -> None:
    """The rule (gap-fixed fusion ``reason`` / ``keep``) on the earlier blind verdicts: 62 judged side-only guard
    licks + the 41 PS95 side-week examples (``real`` = a real lick that is not a fragment of a counted one)."""
    D = dirs()
    V = _verdicts(D)
    rows = []
    for sid in sorted(V.ex_id.str[:13].unique()):
        f = _gapfix_dir() / f"{sid}_fused.csv"
        U = pd.read_csv(f)
        U["ex_id"] = [f"{sid}_t{int(r.trial_id)}_{int(r.on_ms)}" for r in U.itertuples()]
        rows.append(U[["ex_id", "reason", "keep", "cams"]])
    J = V.merge(pd.concat(rows), on="ex_id", how="left")
    C = pd.read_csv(D["out"] / "side_only_causes.csv.gz", usecols=["ex_id", "cause"])
    J = J.merge(C, on="ex_id", how="left")
    J["v"] = np.where(J.real, "real", np.where(J.verdict == "real", "fragment", J.verdict))
    J.to_csv(D["out"] / "validation_blind.csv", index=False)
    print("unmatched:", int(J.reason.isna().sum()))
    print(pd.crosstab(J.reason, J.v, margins=True).to_string())
    for name, m in (("all", np.ones(len(J), bool)), ("without PS93 0805", ~J.ex_id.str.startswith("PS93_20260805"))):
        x = J[m]
        print(f"{name}: real kept {int((x.keep & x.real).sum())}/{int(x.real.sum())}; false kept "
              f"{int((x.keep & ~x.real).sum())}/{int((~x.real).sum())}; paws kept "
              f"{int((x.keep & (x.verdict == 'paw')).sum())}/{int((x.verdict == 'paw').sum())}")


# ------------------------------------------------------------------------------------------- before / after
def composition(U: pd.DataFrame) -> dict:
    """Kept licks by camera combination: all cameras (cam1 + cam4 + a side view), other multi-view, cam1-only,
    cam4-only, side-only."""
    c1, c4 = U.seen_cam1.astype(bool), U.seen_cam4.astype(bool)
    sd = U.seen_cam2.astype(bool) | U.seen_cam3.astype(bool)
    allc = c1 & c4 & sd
    return {"n": len(U), "all_cameras": int(allc.sum()), "other_multi": int(((U.n_views >= 2) & ~allc & (c1 | c4)).sum()),
            "cam1_only": int((c1 & ~c4 & ~sd).sum()), "cam4_only": int((c4 & ~c1 & ~sd).sum()),
            "side_only": int((~c1 & ~c4).sum())}


def _compare_one(f_old: Path, f_new: Path) -> dict:
    from wfield_local import camera_licks as CL
    Uo, Un = pd.read_csv(f_old), pd.read_csv(f_new)
    Ko, Kn = CL.lick_train(Uo), CL.lick_train(Un)
    row = {"sid": f_old.name[:13]}
    row.update({f"before_{k}": v for k, v in composition(Ko).items()})
    row.update({f"after_{k}": v for k, v in composition(Kn).items()})
    so = CL.side_only_mask(Un) & ~Un.guarded.astype(bool).to_numpy()
    for r in ("tracked_cam4", "tracked_cam1", "two_side_views", "cut_fragment", "unconfirmed_one_side_view"):
        row[f"side_only_{r}"] = int((so & (Un.reason == r).to_numpy()).sum())
    return row


def compare(new_dir: str | None = None, tag: str = "") -> None:
    D = dirs()
    new = Path(new_dir) if new_dir else _gapfix_dir()
    rows = []
    for f in sorted(new.glob("PS*_fused.csv")):
        fo = D["licks"] / f.name
        if fo.exists():
            rows.append(_compare_one(fo, f))
    C = pd.DataFrame(rows)
    C.to_csv(D["out"] / f"before_after_by_session{tag}.csv", index=False)
    tot = C.drop(columns="sid").sum()
    print(tot.to_string())
    print("\nsessions:", len(C), "| change in kept licks: median %.2f %%" % (100 * np.median(C.after_n / C.before_n - 1)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--jobs", type=int, default=6)
    sub.add_parser("summarize")
    w = sub.add_parser("windows")
    w.add_argument("--jobs", type=int, default=6)
    pk = sub.add_parser("pick")
    pk.add_argument("--n", type=int, default=N_NEW)
    st = sub.add_parser("strips")
    st.add_argument("--jobs", type=int, default=6)
    sub.add_parser("score")
    sub.add_parser("validate")
    cp = sub.add_parser("compare")
    cp.add_argument("--new", default=None, help="gap-fixed fusion folder (default the _gapfix one)")
    cp.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    if a.cmd == "diagnose":
        diagnose(a.jobs)
    elif a.cmd == "summarize":
        summarize()
    elif a.cmd == "windows":
        windows(a.jobs)
    elif a.cmd == "pick":
        pick(a.n)
    elif a.cmd == "strips":
        strips(a.jobs)
    elif a.cmd == "score":
        score()
    elif a.cmd == "validate":
        validate()
    elif a.cmd == "compare":
        compare(a.new, a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
