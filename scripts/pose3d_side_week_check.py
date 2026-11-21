"""PS95 SIDE-CAMERA WEEK CHECK (Priya 11-21, READY-file Q7): did the PS95 side cameras (cam2 / cam3) move or misbehave
between 2026-08-18 and 08-27, and are the side-only licks of that week real?  READ ONLY on every existing output and on
the raw videos; writes only to ``<pose3d_4cam_3net_20261121>/qc_phaseA/ps95_side_week/``.

Context: the four-camera run's per-session side gate (`pose3d_side.side_gate`) dropped cam3 for PS95 0818 and 0820, and
the fused lick train (`camera_licks`, ``lp_side_r2b_prod_20261118_3net_20261121``) has PS95 side-only licks at 3-10 % of
licks that week (cohort session median 0.42 %).

    python -m scripts.pose3d_side_week_check scan [--jobs 6]   per session x side camera: ENL landmark medians,
                                                               confidences, median ENL frame (frames/, scan.csv)
    python -m scripts.pose3d_side_week_check drift             registration vs the animal's 08-12 frame + spout-fit
                                                               similarity; contact sheets + drift plot
    python -m scripts.pose3d_side_week_check events            side-only lick features, every PS94 / PS95 session
    python -m scripts.pose3d_side_week_check examples [--jobs 4]   frame strips + traces of sampled side-only licks
    python -m scripts.pose3d_side_week_check split             side-only licks split by the near / long rule
                                                               (side_only_rule_split.csv; needs `events`)
    python -m scripts.pose3d_side_week_check table             the per-session table (sessions.csv)

The examples were judged BY EYE (examples_review.csv: real / paw / grooming / water_drop / other_label, written by hand
from the PNGs, 11-21). The split rule (NEAR_MM, LONG_MS) was read off those 41 judged examples AFTER looking -- a
post-hoc description of the separation, not a validated detector.

Definitions (nothing new is defined for the pipeline):
  * side points = `pose3d_side.side_windows` of the production side bundle (raw px), as `fuse_session` reads them.
  * ENL frame = `spout_frame.ruler_medians`' no-lick frame: strobe + 300 ms .. cue, tongue lk < 0.4, spout lk >= 0.9.
  * representative frame = the per-pixel MEDIAN of 15 ENL frames at ``far_center`` (the last ENL frame of 15 trials
    spread over the session, tongue lk < 0.1), grayscale. One single frame is kept beside it.
  * registration shift = `cv2.phaseCorrelate` of that median frame against the same animal's 08-12 median frame (Hann
    window); ``ncc_*`` = Pearson r of the two frames (central 80 %) before / after undoing the shift.
  * spout similarity = `cv2.estimateAffinePartial2D` from the 08-12 six spout medians to the session's (translation,
    rotation, scale); ``resid`` = rms px left after it. Nose / jaw ENL medians move with the head, the background with
    the camera, the spout medians with the camera AND the spout rig.
  * side-only lick = a fused lick (`camera_licks.fuse`) whose ``cams`` holds only cam2 / cam3 and that is not guarded
    (the ones `lick_train` keeps).
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
SIDE_BUNDLE = "lp_side_r2b_prod_20261118"
LICK_RUN = "lp_side_r2b_prod_20261118_3net_20261121"
SIDE = ("cam2", "cam3")
DATES = ("20260812", "20260813", "20260814", "20260817", "20260818", "20260819", "20260820", "20260821", "20260823",
         "20260825", "20260827", "20260831")
ANIMALS = ("PS95", "PS94")                     # PS94 = control: same days, same lesion date (08-16), same rig day
REF_DATE = "20260812"
REF_POS = "far_center"
N_MED = 15
WEEK = ("20260818", "20260827")
#: side-only examples: {session: n}. PS95 in the week (3 each), PS95 outside it, PS94 (control)
EXAMPLES = {**{f"PS95_{d}": 3 for d in ("20260818", "20260819", "20260820", "20260821", "20260823", "20260825",
                                        "20260827")},
            "PS95_20260812": 3, "PS95_20260814": 2, "PS95_20260831": 3, "PS95_20260903": 2,
            "PS94_20260818": 4, "PS94_20260819": 3, "PS94_20260823": 3}
SEED = 0
#: side-only licks whose side event peaks within NEAR_MM of the mouth and lasts >= LONG_MS: the 5 real examples all did
#: (2.3-3.1 mm, 28-76 ms), 35 of the 36 false positives did not (median 7.6 mm, 12 ms). Post hoc, see the docstring.
NEAR_MM = 4.0
LONG_MS = 20.0
POS_COLORS = {"close_L": "#1f77b4", "close_center": "#2ca02c", "close_R": "#d62728", "far_L": "#9467bd",
              "far_center": "#ff7f0e", "far_R": "#8c564b"}


def _env() -> None:
    from wfield_local import pose3d
    os.environ[pose3d.TAG_ENV] = TAG
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


def dirs() -> dict:
    from wfield_local import pose3d
    _env()
    d4 = pose3d.pose3d_dir(source="4cam")
    os.environ.pop(pose3d.SOURCE_ENV, None)
    assert d4.name == "pose3d_4cam_3net_20261121", d4
    out = d4 / "qc_phaseA" / "ps95_side_week"
    assert "MICROSCOPE" in str(out) and "Priya" in str(out), out
    licks = d4.parent / "camera_licks" / LICK_RUN
    return {"4cam": d4, "out": out, "frames": out / "frames", "ex": out / "examples", "licks": licks}


def _rv():
    from wfield_local.paths import PathResolver
    return PathResolver()


# --------------------------------------------------------------------------------------------------------- scan
def scan_worker(job: tuple) -> dict:
    """One session x side camera: ENL landmark medians, confidences, and the median ENL frame (written as PNG)."""
    _env()
    import cv2

    from wfield_local import o2_inference as oi
    from wfield_local import pose3d_side as PS
    from wfield_local import spout_frame as SF
    an, date, cam = job
    D = dirs()
    rv = _rv()
    got = PS.side_windows(rv, SIDE_BUNDLE, an, date, cam)
    if got is None:
        return {"animal": an, "date": date, "cam": cam, "missing": True}
    ix, flat = got
    t, sb = ix.t_ms.to_numpy(float), ix.strobe_ms.to_numpy(float)
    lk = {p: flat[f"{p}_likelihood"].to_numpy(float) for p in ("nose", "jaw", "tongue", "spout")}
    enl = (t >= sb + SF.ENL_SETTLE_MS) & (t < 0) & (lk["tongue"] < SF.RULER_TONGUE_MAX_LK) & (lk["spout"] >= SF.RULER_SPOUT_MIN_LK)
    row = {"animal": an, "date": date, "cam": cam, "width": int(ix.width.iloc[0]), "n_frames": len(ix),
           "n_enl": int(enl.sum()), "jaw_lk_med": float(np.median(lk["jaw"])), "tongue_lk_med": float(np.median(lk["tongue"])),
           "tongue_lk_frac04": float((lk["tongue"] >= 0.4).mean()), "nose_lk_med": float(np.median(lk["nose"])),
           "spout_lk_med_enl_any": float(np.median(lk["spout"][(t >= sb + SF.ENL_SETTLE_MS) & (t < 0)])),
           "jaw_lk_med_enl": float(np.median(lk["jaw"][enl])) if enl.any() else np.nan}
    for p, (x, y, n) in SF.ruler_medians(flat, ix).items():
        row[f"spout_{p}_x"], row[f"spout_{p}_y"], row[f"spout_{p}_n"] = x, y, n
    for part in ("nose", "jaw"):
        m = enl & (lk[part] >= 0.6)
        row[f"{part}_x"] = float(np.median(flat[f"{part}_x"].to_numpy(float)[m])) if m.any() else np.nan
        row[f"{part}_y"] = float(np.median(flat[f"{part}_y"].to_numpy(float)[m])) if m.any() else np.nan
    # representative frames: the last ENL frame of trials at REF_POS, tongue clearly absent
    cand = pd.DataFrame({"trial": ix.trial_id.to_numpy(), "t": t, "src": ix.src_frame.to_numpy()})[
        enl & (ix.position.astype(str).to_numpy() == REF_POS) & (lk["tongue"] < 0.1)]
    last = cand.sort_values("t").groupby("trial").tail(1).sort_values("trial")
    pick = last.iloc[np.unique(np.linspace(0, len(last) - 1, min(N_MED, len(last))).round().astype(int))] if len(last) else last
    vids = oi.videos_for(rv, an, date, cam)
    row["n_videos"] = len(vids)
    ims = []
    cap = cv2.VideoCapture(str(vids[0]))
    for f in pick.src.astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, im = cap.read()
        if ok:
            ims.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    cap.release()
    row["n_med_frames"] = len(ims)
    row["med_frame_src"] = ";".join(str(int(f)) for f in pick.src)
    if ims:
        D["frames"].mkdir(parents=True, exist_ok=True)
        med = np.median(np.stack(ims), axis=0).astype(np.uint8)
        cv2.imwrite(str(D["frames"] / f"{an}_{date}_{cam}_median.png"), med)
        cv2.imwrite(str(D["frames"] / f"{an}_{date}_{cam}_single.png"), ims[len(ims) // 2])
        row["img_mean"], row["img_std"] = float(med.mean()), float(med.std())
    return row


def scan(jobs: int) -> None:
    from wfield_local import parallel
    D = dirs()
    D["out"].mkdir(parents=True, exist_ok=True)
    items = [(a, d, c) for a in ANIMALS for d in DATES for c in SIDE]
    res, fail = parallel.fan_out(items, scan_worker, jobs=jobs, label="session-camera")
    if fail:
        raise SystemExit(f"scan failed: {fail}")
    T = pd.DataFrame([r for _, r in res]).sort_values(["animal", "cam", "date"])
    T.to_csv(D["out"] / "scan.csv", index=False, float_format="%.4g")
    print("wrote", D["out"] / "scan.csv", len(T))


# -------------------------------------------------------------------------------------------------------- drift
def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    h, w = a.shape
    y0, x0 = int(0.1 * h), int(0.1 * w)
    a, b = a[y0:h - y0, x0:w - x0].ravel(), b[y0:h - y0, x0:w - x0].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _spout_pts(r: pd.Series) -> tuple[list, np.ndarray]:
    ps = [p for p in POS_COLORS if np.isfinite(r.get(f"spout_{p}_x", np.nan))]
    return ps, np.array([[r[f"spout_{p}_x"], r[f"spout_{p}_y"]] for p in ps], float)


def drift() -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = dirs()
    T = pd.read_csv(D["out"] / "scan.csv", dtype={"date": str})
    G = pd.read_csv(D["4cam"] / "summary_production_lp_side_r2b_prod_20261118.csv")
    rows = []
    imgs = {}
    for r in T.itertuples():
        f = D["frames"] / f"{r.animal}_{r.date}_{r.cam}_median.png"
        imgs[(r.animal, r.date, r.cam)] = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) if f.exists() else None
    for _, r in T.iterrows():
        an, date, cam = r.animal, r.date, r.cam
        ref = T[(T.animal == an) & (T.cam == cam) & (T.date == REF_DATE)].iloc[0]
        a, b = imgs[(an, REF_DATE, cam)], imgs[(an, date, cam)]
        o = {"animal": an, "date": date, "cam": cam}
        if a is not None and b is not None:
            win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
            # copies: cv2.phaseCorrelate applies the window to its inputs IN PLACE in this OpenCV build (seen 11-21:
            # the 08-12 reference came back Hann-windowed); each frame z-scored so exposure changes do not register
            za, zb = ((v - v.mean()) / v.std() for v in (a, b))
            (dx, dy), resp = cv2.phaseCorrelate(za.copy(), zb.copy(), win)
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            bb = cv2.warpAffine(b, M, a.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
            o.update({"img_dx": dx, "img_dy": dy, "img_shift": float(np.hypot(dx, dy)), "img_resp": resp,
                      "ncc_raw": _ncc(a, b), "ncc_aligned": _ncc(a, bb)})
        p0, X0 = _spout_pts(ref)
        p1, X1 = _spout_pts(r)
        common = [p for p in p0 if p in p1]
        if len(common) >= 3:
            A = np.array([X0[p0.index(p)] for p in common], np.float32)
            B = np.array([X1[p1.index(p)] for p in common], np.float32)
            Ms, _ = cv2.estimateAffinePartial2D(A, B, method=cv2.LMEDS)
            if Ms is not None:
                s = float(np.hypot(Ms[0, 0], Ms[1, 0]))
                pred = A @ Ms[:, :2].T + Ms[:, 2]
                o.update({"spout_tx": float(Ms[0, 2] + (Ms[:, :2] @ A.mean(0) - A.mean(0))[0]),
                          "spout_ty": float(Ms[1, 2] + (Ms[:, :2] @ A.mean(0) - A.mean(0))[1]),
                          "spout_rot_deg": float(np.degrees(np.arctan2(Ms[1, 0], Ms[0, 0]))), "spout_scale": s,
                          "spout_resid_px": float(np.sqrt(np.mean(np.sum((pred - B) ** 2, 1)))),
                          "spout_mean_disp_px": float(np.mean(np.linalg.norm(B - A, axis=1))),
                          "spout_n_pos": len(common)})
        for part in ("nose", "jaw"):
            o[f"{part}_disp_px"] = float(np.hypot(r[f"{part}_x"] - ref[f"{part}_x"], r[f"{part}_y"] - ref[f"{part}_y"]))
            o[f"{part}_dx"], o[f"{part}_dy"] = r[f"{part}_x"] - ref[f"{part}_x"], r[f"{part}_y"] - ref[f"{part}_y"]
        g = G[G.session == f"{an}_{date}"]
        if len(g):
            o["gate_rms600"], o["gate_max600"] = float(g[f"side_rms600_{cam}"].iloc[0]), float(g[f"side_max600_{cam}"].iloc[0])
            o["gate_dropped"] = cam in str(g.side_dropped.iloc[0])
        rows.append(o)
    R = pd.DataFrame(rows)
    R.to_csv(D["out"] / "drift.csv", index=False, float_format="%.4g")
    print("wrote", D["out"] / "drift.csv")

    # contact sheets: one per camera; PS95 rows on top, PS94 below; spout medians (dots) vs the 08-12 medians (crosses)
    for cam in SIDE:
        for kind in ("median", "diff"):
            fig, axs = plt.subplots(4, 6, figsize=(18, 12.5))
            for ia, an in enumerate(ANIMALS):
                ref = T[(T.animal == an) & (T.cam == cam) & (T.date == REF_DATE)].iloc[0]
                a = imgs[(an, REF_DATE, cam)]
                for k, date in enumerate(DATES):
                    ax = axs[2 * ia + k // 6, k % 6]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    b = imgs[(an, date, cam)]
                    r = T[(T.animal == an) & (T.cam == cam) & (T.date == date)].iloc[0]
                    d = R[(R.animal == an) & (R.cam == cam) & (R.date == date)].iloc[0]
                    if b is None:
                        ax.text(0.5, 0.5, "no frame", ha="center", transform=ax.transAxes)
                        continue
                    if kind == "median":
                        ax.imshow(b, cmap="gray", vmin=0, vmax=255)
                        for p, c in POS_COLORS.items():
                            if np.isfinite(ref.get(f"spout_{p}_x", np.nan)):
                                ax.plot(ref[f"spout_{p}_x"], ref[f"spout_{p}_y"], "+", color=c, ms=7, mew=1.2)
                            if np.isfinite(r.get(f"spout_{p}_x", np.nan)):
                                ax.plot(r[f"spout_{p}_x"], r[f"spout_{p}_y"], "o", mfc="none", mec=c, ms=5, mew=1)
                        for part, c in (("nose", "#00e5ff"), ("jaw", "#ffd400")):
                            ax.plot(ref[f"{part}_x"], ref[f"{part}_y"], "+", color=c, ms=7, mew=1.2)
                            ax.plot(r[f"{part}_x"], r[f"{part}_y"], "o", mfc="none", mec=c, ms=5, mew=1)
                    else:
                        ax.imshow(b - a, cmap="RdBu_r", vmin=-60, vmax=60)
                    gate = "DROPPED" if bool(d.get("gate_dropped", False)) else "pass"
                    ax.set_title(f"{an} {date[4:6]}-{date[6:]}  shift {d.get('img_shift', np.nan):.1f} px  "
                                 f"r {d.get('ncc_raw', np.nan):.2f}\nspout {d.get('spout_mean_disp_px', np.nan):.1f} px  "
                                 f"gate rms {d.get('gate_rms600', np.nan):.2f} ({gate})",
                                 fontsize=7.5, color="#b00000" if gate == "DROPPED" else "black")
            what = ("per-pixel median of 15 no-lick ENL frames at far_center (spout parked, before the cue). "
                    "+ = 08-12 landmark medians, o = this session's (spout: one color per position; cyan nose, "
                    "yellow jaw).") if kind == "median" else "this session's median frame MINUS the animal's 08-12 median frame (gray levels)."
            fig.suptitle(f"{cam} (side view), one representative frame per day. Rows 1-2: PS95; rows 3-4: PS94 (control). "
                         f"{what}\nshift = phase-correlation shift vs 08-12; r = pixel correlation vs 08-12; spout = mean "
                         f"displacement of the six spout medians vs 08-12; gate = the four-camera side gate (rms600 <= 1.5, "
                         f"max <= 2.0).", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            f = D["out"] / f"contact_sheet_{cam}{'' if kind == 'median' else '_diff'}.png"
            fig.savefig(f, dpi=110)
            plt.close(fig)
            print("wrote", f)

    # drift plot
    S = sessions_share()
    fig, axs = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    x = {d: i for i, d in enumerate(DATES)}
    sty = {("PS95", "cam2"): ("#c0392b", "o", "-"), ("PS95", "cam3"): ("#c0392b", "s", "--"),
           ("PS94", "cam2"): ("#2c7fb8", "o", "-"), ("PS94", "cam3"): ("#2c7fb8", "s", "--")}
    for (an, cam), (c, m, ls) in sty.items():
        d = R[(R.animal == an) & (R.cam == cam)].sort_values("date")
        xx = [x[v] for v in d.date]
        axs[0].plot(xx, d.img_shift, ls=ls, marker=m, color=c, mfc="none", label=f"{an} {cam}")
        axs[1].plot(xx, d.ncc_raw, ls=ls, marker=m, color=c, mfc="none")
        axs[2].plot(xx, d.spout_mean_disp_px, ls=ls, marker=m, color=c, mfc="none")
        axs[3].plot(xx, d.gate_rms600, ls=ls, marker=m, color=c, mfc="none")
    for an, c in (("PS95", "#c0392b"), ("PS94", "#2c7fb8")):
        s = S[S.animal == an].set_index("date").reindex(list(DATES))
        axs[4].plot(range(len(DATES)), s.side_only_pct, marker="o", color=c, mfc="none", label=an)
    axs[3].axhline(1.5, color="gray", lw=0.8, ls=":")
    axs[4].axhline(0.42, color="gray", lw=0.8, ls=":")
    labels = ["registration shift vs 08-12 (px)", "pixel r vs 08-12", "spout medians: mean shift vs 08-12 (px)",
              "side gate rms600 (px)", "side-only licks (% of kept licks)"]
    for ax, lab in zip(axs, labels):
        ax.set_ylabel(lab, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvspan(x[WEEK[0]] - 0.5, x[WEEK[1]] + 0.5, color="#f3e1c0", alpha=0.5, lw=0)
    axs[0].legend(fontsize=7, ncol=4, frameon=False)
    axs[4].legend(fontsize=7, frameon=False)
    axs[-1].set_xticks(range(len(DATES)))
    axs[-1].set_xticklabels([f"{d[4:6]}-{d[6:]}" for d in DATES], fontsize=8)
    fig.suptitle("Side-camera view stability, PS95 vs PS94 (both lesioned 08-16). Shaded = 08-18..08-27. Reference = "
                 "each animal's 08-12 frame.\nDotted: gate threshold 1.5 (panel 4); cohort session median side-only "
                 "share 0.42 % (panel 5).", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(D["out"] / "drift_by_day.png", dpi=130)
    plt.close(fig)
    print("wrote", D["out"] / "drift_by_day.png")


# ------------------------------------------------------------------------------------------------------- events
def _side_only(U: pd.DataFrame) -> np.ndarray:
    return U.cams.astype(str).map(lambda s: set(s.split("+")) <= set(SIDE)).to_numpy() & ~U.guarded.astype(bool).to_numpy()


def sessions_share() -> pd.DataFrame:
    D = dirs()
    V = pd.read_csv(D["licks"] / "views.csv")
    so = V.side_only.to_numpy() > 0
    V["so_kept"] = np.where(so, V.side_only - V.guarded, 0)
    g = V.groupby("session")
    S = pd.DataFrame({"n_all": g.n.sum(), "n_guarded": g.guarded.sum(), "side_only_kept": g.so_kept.sum()}).reset_index()
    S["n_kept"] = S.n_all - S.n_guarded
    S["side_only_pct"] = 100 * S.side_only_kept / S.n_kept
    S["animal"], S["date"] = S.session.str[:4], S.session.str[5:]
    return S


def _match_side(U: pd.DataFrame, E: pd.DataFrame) -> pd.DataFrame:
    """For each fused lick of ``U``, the side event(s) inside its span: best (largest peak_dist) per lick."""
    rows = []
    Eg = {k: v for k, v in E.groupby("trial_id")}
    for _i, r in U.iterrows():
        e = Eg.get(r.trial_id)
        if e is None:
            rows.append({})
            continue
        m = e[(e.on_ms <= r.off_ms) & (e.off_ms >= r.on_ms)]
        if not len(m):
            rows.append({})
            continue
        b = m.sort_values("peak_dist_mm").iloc[-1]
        rows.append({"side_cam": b.cam, "side_n_frames": b.n_frames, "side_peak_dist_mm": b.peak_dist_mm,
                     "side_peak_out_px": b.peak_out_px, "side_range_px": b.range_px, "side_size_class": b.size_class,
                     "side_static": b.static, "side_far": b.far})
    return pd.DataFrame(rows, index=U.index)


def events() -> None:
    """Every fused lick of PS94 / PS95 sessions, flagged side-only, with the side event's features and the gap to the
    nearest OTHER fused lick of the same trial (ms). Summaries per session and side-only vs multi-view."""
    D = dirs()
    out = []
    cache = D["out"] / "fused_licks_ps94_ps95.csv.gz"
    for f in ([] if cache.exists() else sorted(D["licks"].glob("PS9[45]_2026*_fused.csv"))):
        sid = f.name[:13]
        U = pd.read_csv(f)
        E = pd.read_csv(D["licks"] / f"{sid}_side_events.csv")
        U["sid"], U["animal"], U["date"] = sid, sid[:4], sid[5:]
        U["side_only"] = _side_only(U)
        U = pd.concat([U, _match_side(U, E)], axis=1)
        gap = np.full(len(U), np.nan)
        for _, g in U.groupby("trial_id"):
            on, off = g.on_ms.to_numpy(float), g.off_ms.to_numpy(float)
            for j, i in enumerate(g.index):
                d = np.r_[on[np.arange(len(g)) != j] - off[j], on[j] - off[np.arange(len(g)) != j]]
                d = d[d >= 0]
                gap[U.index.get_loc(i)] = d.min() if len(d) else np.nan
        U["gap_other_ms"] = gap
        U["pre_cue"] = U.on_ms < 0
        U["dur_ms"] = U.off_ms - U.on_ms
        out.append(U)
    if out:
        A = pd.concat(out, ignore_index=True)
        A.to_csv(cache, index=False, float_format="%.4g")
    else:
        A = pd.read_csv(cache)
    K = A[~A.guarded.astype(bool)]
    so = K[K.side_only]
    mv = K[~K.side_only & K.seen_cam2.astype(bool)]
    summ = []
    for sid, g in K.groupby("sid"):
        s = g[g.side_only]
        summ.append({"session": sid, "n_kept": len(g), "n_side_only": len(s),
                     "so_cam2_only": int((s.cams == "cam2").sum()), "so_cam3_only": int((s.cams == "cam3").sum()),
                     "so_cam2_cam3": int((s.cams == "cam2+cam3").sum()),
                     "so_pre_cue_frac": float(s.pre_cue.mean()) if len(s) else np.nan,
                     "so_all_pre_cue_frac": float(g.pre_cue.mean()),
                     "so_dur_med_ms": float(s.dur_ms.median()) if len(s) else np.nan,
                     "so_peak_dist_med_mm": float(s.side_peak_dist_mm.median()) if len(s) else np.nan,
                     "so_gap_med_ms": float(s.gap_other_ms.median()) if len(s) else np.nan,
                     "so_within_50ms_frac": float((s.gap_other_ms < 50).mean()) if len(s) else np.nan,
                     "so_isolated_frac": float(s.gap_other_ms.isna().mean()) if len(s) else np.nan})
    pd.DataFrame(summ).to_csv(D["out"] / "side_only_by_session.csv", index=False, float_format="%.4g")
    c = pd.DataFrame({"side_only": so[["dur_ms", "side_n_frames", "side_peak_dist_mm", "gap_other_ms"]].describe().T["50%"],
                      "multi_view_with_cam2": mv[["dur_ms", "side_n_frames", "side_peak_dist_mm", "gap_other_ms"]].describe().T["50%"]})
    print(c)
    print("wrote", D["out"] / "side_only_by_session.csv")


# ----------------------------------------------------------------------------------------------------- examples
def pick_examples() -> pd.DataFrame:
    D = dirs()
    A = pd.read_csv(D["out"] / "fused_licks_ps94_ps95.csv.gz")
    rng = np.random.default_rng(SEED)
    rows = []
    for sid, n in EXAMPLES.items():
        g = A[(A.sid == sid) & A.side_only & ~A.guarded.astype(bool)]
        if not len(g):
            continue
        k = rng.choice(len(g), size=min(n, len(g)), replace=False)
        rows.append(g.iloc[np.sort(k)])
    P = pd.concat(rows, ignore_index=True)
    P["group"] = np.where(P.animal == "PS94", "PS94_control",
                          np.where(P.date.astype(str).between(*WEEK), "PS95_week", "PS95_outside"))
    P["ex_id"] = [f"{r.sid}_t{int(r.trial_id)}_{int(r.on_ms)}" for r in P.itertuples()]
    return P


def example_worker(sid: str) -> list:
    """Frame strips + traces for the picked side-only licks of one session (one `fuse_session`)."""
    _env()
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.pose3d import PROD
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d_side as PS
    D = dirs()
    rv = _rv()
    an, date = sid.split("_")
    P = pd.read_csv(D["out"] / "examples_picked.csv")
    P = P[P.sid == sid]
    c1 = oi.session_windows(rv, PROD["cam1"], an, date, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], an, date, "cam4")
    Z = PS.fuse_session(rv, an, date, side_bundle=SIDE_BUNDLE, c1=c1, c4=c4)
    F, order, pts, side = Z["F"], Z["order"], Z["pts"], Z["side"]
    cams = {c.get_name(): c for c in Z["g"].cameras}
    gate = Z["stats"]["side_gate"]
    X = Z["parts"]["tongue"]["X"]
    reach3d = np.linalg.norm(X, axis=1)
    raw = {c: (c1 if c == "cam1" else c4) for c in ("cam1", "cam4")}
    srcmap = {}
    for c in ("cam1", "cam4"):
        ix = raw[c][0]
        srcmap[c] = pd.Series(np.arange(len(ix)), index=ix.src_frame.to_numpy())
        srcmap[c] = srcmap[c][~srcmap[c].index.duplicated()]
    src = {"cam1": F.src1.to_numpy(), "cam4": F.src4.to_numpy(),
           **{c: np.where(F[f"i_{c}"].to_numpy() >= 0,
                          side[c][0].src_frame.to_numpy()[np.clip(F[f"i_{c}"].to_numpy(), 0, None)], -1) for c in SIDE}}

    def raw_lk(c: str, part: str, i: int) -> float:
        if c in SIDE:
            k = int(F[f"i_{c}"].iat[i])
            return float(side[c][1][f"{part}_likelihood"].iat[k]) if k >= 0 else np.nan
        s = src[c][i]
        if s < 0 or s not in srcmap[c].index:
            return np.nan
        return float(raw[c][1][f"{part}_likelihood"].iat[int(srcmap[c][s])])

    def raw_xy(c: str, part: str, i: int):
        if c in SIDE:
            k = int(F[f"i_{c}"].iat[i])
            fl = side[c][1]
        else:
            s = src[c][i]
            k = int(srcmap[c][s]) if (s >= 0 and s in srcmap[c].index) else -1
            fl = raw[c][1]
        if k < 0:
            return np.array([np.nan, np.nan]), np.nan
        return np.array([fl[f"{part}_x"].iat[k], fl[f"{part}_y"].iat[k]], float), float(fl[f"{part}_likelihood"].iat[k])

    mouth = {c: PS.project(cams[c], np.zeros((1, 3)))[0] for c in order}
    ppm = {c: PS.px_per_mm(cams[c]) for c in order}
    readers = {c: cv2.VideoCapture(str(oi.videos_for(rv, an, date, c)[0])) for c in order}
    show = ("cam2", "cam3", "cam1", "cam4")
    feats = []
    for r in P.itertuples():
        rows_t = np.flatnonzero(F.trial_id.to_numpy() == r.trial_id)
        tt = F.t_ms.to_numpy(float)[rows_t]
        on, off = float(r.on_ms), float(r.off_ms)
        win = rows_t[(tt >= on - 200) & (tt <= off + 200)]
        ev = rows_t[(tt >= on - 4) & (tt <= off + 4)]
        strip = rows_t[(tt >= on - 24) & (tt <= off + 24)]
        sel = strip[np.unique(np.linspace(0, len(strip) - 1, min(9, len(strip))).round().astype(int))] if len(strip) else strip
        f = {"ex_id": r.ex_id, "sid": sid, "group": r.group, "trial_id": int(r.trial_id), "position": r.position,
             "on_ms": on, "off_ms": off, "cams": r.cams, "pre_cue": bool(on < 0), "gap_other_ms": r.gap_other_ms,
             "side_peak_dist_mm": r.side_peak_dist_mm, "side_n_frames": r.side_n_frames,
             "gate_cam2": gate["cam2"]["pass"], "gate_cam3": gate["cam3"]["pass"],
             "reach3d_max_mm": float(np.nanmax(reach3d[ev])) if len(ev) and np.isfinite(reach3d[ev]).any() else np.nan}
        for c in show:
            lks = [raw_lk(c, "tongue", i) for i in ev]
            f[f"{c}_tongue_lk_max"] = float(np.nanmax(lks)) if np.isfinite(lks).any() else np.nan
            j = order.index(c)
            f[f"{c}_clean_frames"] = int(np.isfinite(pts["tongue"][j, ev]).all(1).sum())
            jl = [raw_lk(c, "jaw", i) for i in ev]
            f[f"{c}_jaw_lk_med"] = float(np.nanmedian(jl)) if np.isfinite(jl).any() else np.nan
        feats.append(f)
        # ---- figure: strip (4 cams x <=9 frames) + traces
        nf = max(len(sel), 1)
        fig = plt.figure(figsize=(1.75 * nf + 0.4, 1.75 * 4 + 4.4))
        gs = fig.add_gridspec(6, nf, height_ratios=[1, 1, 1, 1, 0.9, 0.9], hspace=0.25, wspace=0.04)
        for jc, c in enumerate(show):
            j = order.index(c)
            cs = 240 if c in SIDE else 220
            cx, cy = mouth[c]
            for jj, i in enumerate(sel):
                ax = fig.add_subplot(gs[jc, jj])
                ax.set_xticks([])
                ax.set_yticks([])
                vf = src[c][i]
                im = None
                if vf >= 0:
                    readers[c].set(cv2.CAP_PROP_POS_FRAMES, int(vf))
                    ok_, im = readers[c].read()
                if im is None:
                    ax.text(0.5, 0.5, "no frame", ha="center", fontsize=6, transform=ax.transAxes)
                    continue
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                x0 = int(np.clip(cx - cs / 2, 0, max(im.shape[1] - cs, 0)))
                y0 = int(np.clip(cy - cs / 2, 0, max(im.shape[0] - cs, 0)))
                ax.imshow(im[y0:y0 + cs, x0:x0 + cs])
                q, ql = raw_xy(c, "tongue", i)
                if np.isfinite(q).all() and ql >= 0.1:
                    ax.plot(q[0] - x0, q[1] - y0, "o", ms=5, mfc="none", mec="#ff3c3c", mew=0.8)
                qc = pts["tongue"][j, i]
                if np.isfinite(qc).all():
                    ax.plot(qc[0] - x0, qc[1] - y0, "o", ms=3.5, color="#ff3c3c")
                qj, jl = raw_xy(c, "jaw", i)
                if np.isfinite(qj).all() and jl >= 0.6:
                    ax.plot(qj[0] - x0, qj[1] - y0, "^", ms=4, color="#ffd400")
                ax.plot(cx - x0, cy - y0, "+", color="cyan", ms=6, mew=1)
                ax.text(2, 12, f"tg {ql:.2f}", color="w", fontsize=5.5)
                if jc == 0:
                    ax.set_title(f"{F.t_ms.iat[i] - on:+.0f} ms", fontsize=7)
                if jj == 0:
                    lab = c + (" (gate DROP)" if c in SIDE and not gate[c]["pass"] else "")
                    ax.set_ylabel(lab, fontsize=7)
        t = F.t_ms.to_numpy(float)[win] - on
        ax1 = fig.add_subplot(gs[4, :])
        ax2 = fig.add_subplot(gs[5, :], sharex=ax1)
        colc = {"cam1": "#1b9e77", "cam4": "#7570b3", "cam2": "#d95f02", "cam3": "#e7298a"}
        for c in show:
            j = order.index(c)
            d = np.linalg.norm(pts["tongue"][j, win] - mouth[c], axis=1) / ppm[c]
            ax1.plot(t, d, ".-", ms=2, lw=0.8, color=colc[c], label=f"{c} cleaned tongue - mouth (mm-scaled px)")
            ax2.plot(t, [raw_lk(c, "tongue", i) for i in win], lw=0.8, color=colc[c], label=c)
        ax1.plot(t, reach3d[win], "k-", lw=1.2, label="fused 3-D tongue reach (mm)")
        for ax in (ax1, ax2):
            ax.axvspan(0, off - on, color="#f3e1c0", alpha=0.6, lw=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax2.axhline(0.4, color="gray", lw=0.6, ls=":")
        ax1.set_ylabel("distance (mm)", fontsize=7)
        ax2.set_ylabel("raw tongue lk", fontsize=7)
        ax2.set_xlabel("ms from side-only lick onset (shaded = fused lick span)", fontsize=7)
        ax1.legend(fontsize=5.5, ncol=3, frameon=False, loc="upper left")
        ax2.legend(fontsize=5.5, ncol=4, frameon=False, loc="upper left")
        fig.suptitle(f"{r.ex_id} [{r.group}]  {r.position}  cams={r.cams}  {'PRE-cue (ITI)' if on < 0 else 'post-cue'}  "
                     f"dur {off - on:.0f} ms  side peak {r.side_peak_dist_mm:.2f} mm  gap to other lick "
                     f"{r.gap_other_ms:.0f} ms\nred dot = cleaned tongue (used for licks); open red = raw tongue label "
                     f"(lk >= 0.1; 'tg' = its likelihood); yellow = jaw (lk >= 0.6); cyan + = projected mouth zero",
                     fontsize=7.5)
        D["ex"].mkdir(parents=True, exist_ok=True)
        fig.savefig(D["ex"] / f"{r.group}__{r.ex_id}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    for cap in readers.values():
        cap.release()
    return feats


def examples(jobs: int) -> None:
    from wfield_local import parallel
    D = dirs()
    P = pick_examples()
    P.to_csv(D["out"] / "examples_picked.csv", index=False)
    print(P.groupby("group").size())
    res, fail = parallel.fan_out(sorted(P.sid.unique()), example_worker, jobs=jobs, label="session")
    if fail:
        print("FAILED:", fail)
    Fe = pd.DataFrame([f for _, fs in res for f in fs]).sort_values(["group", "ex_id"])
    Fe.to_csv(D["out"] / "examples_features.csv", index=False, float_format="%.4g")
    print("wrote", D["out"] / "examples_features.csv", len(Fe))


# -------------------------------------------------------------------------------------------------------- split
def split() -> None:
    D = dirs()
    A = pd.read_csv(D["out"] / "fused_licks_ps94_ps95.csv.gz")
    K = A[~A.guarded.astype(bool)]
    so = K[K.side_only].copy()
    so["near"] = (so.side_peak_dist_mm <= NEAR_MM) & (so.dur_ms >= LONG_MS)
    s = so.groupby("sid").agg(n_so=("near", "size"), n_near=("near", "sum"))
    s["near_close_center"] = so[so.near].groupby("sid").position.apply(lambda p: int((p == "close_center").sum()))
    s = s.join(K.groupby("sid").size().rename("n_kept"))
    s["so_pct"] = 100 * s.n_so / s.n_kept
    s["near_pct"] = 100 * s.n_near / s.n_kept
    s["far_or_short_pct"] = s.so_pct - s.near_pct
    s.to_csv(D["out"] / "side_only_rule_split.csv", float_format="%.4g")
    print(s.round(2).to_string())


# -------------------------------------------------------------------------------------------------------- table
def table() -> None:
    D = dirs()
    T = pd.read_csv(D["out"] / "scan.csv", dtype={"date": str})
    R = pd.read_csv(D["out"] / "drift.csv", dtype={"date": str})
    S = sessions_share()
    E = pd.read_csv(D["out"] / "side_only_by_session.csv")
    Sp = pd.read_csv(D["out"] / "side_only_rule_split.csv")
    rows = []
    for an in ANIMALS:
        for date in DATES:
            sid = f"{an}_{date}"
            o = {"session": sid, "animal": an, "date": date}
            s = S[S.session == sid]
            if len(s):
                o.update({"n_kept_licks": int(s.n_kept.iloc[0]), "side_only": int(s.side_only_kept.iloc[0]),
                          "side_only_pct": float(s.side_only_pct.iloc[0])})
            sp = Sp[Sp.sid == sid]
            if len(sp):
                o.update({"side_only_near_long_pct": float(sp.near_pct.iloc[0]),
                          "side_only_far_or_short_pct": float(sp.far_or_short_pct.iloc[0])})
            e = E[E.session == sid]
            if len(e):
                o.update({k: e[k].iloc[0] for k in ("so_cam2_only", "so_cam3_only", "so_cam2_cam3", "so_pre_cue_frac",
                                                    "so_dur_med_ms", "so_peak_dist_med_mm", "so_within_50ms_frac")})
            for cam in SIDE:
                t = T[(T.animal == an) & (T.date == date) & (T.cam == cam)]
                r = R[(R.animal == an) & (R.date == date) & (R.cam == cam)]
                if len(r):
                    r = r.iloc[0]
                    o.update({f"{cam}_gate_rms600": r.gate_rms600, f"{cam}_gate_max600": r.gate_max600,
                              f"{cam}_gate_dropped": bool(r.gate_dropped), f"{cam}_img_shift_px": r.img_shift,
                              f"{cam}_ncc_raw": r.ncc_raw, f"{cam}_spout_disp_px": r.spout_mean_disp_px,
                              f"{cam}_spout_scale": r.spout_scale, f"{cam}_spout_resid_px": r.spout_resid_px,
                              f"{cam}_jaw_disp_px": r.jaw_disp_px})
                if len(t):
                    t = t.iloc[0]
                    o.update({f"{cam}_jaw_lk_med": t.jaw_lk_med, f"{cam}_jaw_lk_med_enl": t.jaw_lk_med_enl,
                              f"{cam}_tongue_lk_med": t.tongue_lk_med, f"{cam}_tongue_lk_frac04": t.tongue_lk_frac04})
                # the CAMERA moved = the whole frame registers > 3 px off 08-12 (the spout medians also move with the
                # spout rig's day-to-day placement, 2-16 px in both animals, so they do not define a camera move)
                o[f"{cam}_view_moved"] = bool(o.get(f"{cam}_img_shift_px", 0) > 3)
            rows.append(o)
    out = pd.DataFrame(rows)
    out.to_csv(D["out"] / "sessions.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(out.to_string())
    print("wrote", D["out"] / "sessions.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["scan", "drift", "events", "examples", "split", "table"])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    _env()
    {"scan": lambda: scan(a.jobs), "drift": drift, "events": events, "examples": lambda: examples(a.jobs),
     "split": split, "table": table}[a.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
