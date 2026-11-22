"""Supplementary four-camera clips: PS94 FAR CONTRA licks in each epoch, with the 3-D tracking and the aim deviation
(Priya 11-22: "it would be nice to include 4-view clips as supplemental (ensure the nose is not cut off in the cam1 view
and the tongue isn't cut off in the cam4 view), with the 3d tracking, of a few PS94 example far contra licks from pre,
acute, subacute, and chronic (showing the aim deviation)").

Built on `scripts.pose3d_lick_clips` (production four-camera 3-D `pose3d_4cam_3net_20261121_sync60`, the adopted fused
train, the fusion re-run CHECKED against the production frames table before anything is drawn). Differences:
  * a manuscript layout: four views (cam4 front, cam1 below, cam2 / cam3 sides) with the cleaned 2-D keypoints and the
    production 3-D point reprojected; a TOP-DOWN panel (ml x ap, mouth zero at the origin) with the six commanded spouts
    (target filled), the tongue trail, the TARGET direction (dashed) and, from the first lick's peak extension on, that
    lick's direction (solid) with its aim error; a SIDE panel (ap x dv); a slim timeline (trial period, DAQ contacts,
    3-D licks, cursor).
  * crops that never clip: each view is padded before cropping (no clamping to the image edge), the cam1 crop includes
    the nose with a margin and extends AHEAD of it (mouth -> nose direction), and the cam4 crop includes every tongue
    point of the clip and the target spout with a margin.
  * H.264 Main profile, level 4.0, yuv420p, faststart (the High-profile clips of 11-22 did not open for Priya).

SELECTION RULE (stated, not chosen by eye). Pool = PS94 far_R (= far contra; PS94 lesion left) FIRST lick of each trial
in `kinematics/licks.csv.gz` (cam4 lick windows on the four-camera 3-D, the Fig 4 recipe), August-onward sessions only
(pre = August baseline), with >= 15 3-D frames and a finite aim error. Aim error = `midline_err_ipsi` (degrees out of the midline plane at the lick's peak extension, target
minus lick, + = toward ipsi). Per epoch: m = median aim error of the pool; session = the one whose
median is closest to m (sessions with >= 20 pool licks); its licks with <= 10 % single-view 3-D frames (so the
rendered lick is tracked from >= 2 views) are ranked by |aim error - m|, those whose fused licks in the clip window are all kept and
seen by >= 3 cameras (with >= 15 tongue frames) first; the first N with complete video in all four cameras are rendered.
Window: cue - 100 ms .. max(+700, first-lick peak + 450 ms).

    conda activate dlc   # MKL_THREADING_LAYER=SEQUENTIAL, PYTHONPATH=<worktree>
    python -m scripts.pose3d_supp_clips --out <folder> [--n 2] [--jobs 2]
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import pose3d_lick_clips as LC

ANIMAL, POSITION = "PS94", "far_R"      # default animal; POSITION = far contra for the left-lesioned PS94 / PS95
#: epoch colours (manuscript `vls_style.EP`, Paul Tol) -> BGR, for the epoch banner and title cards
EP_BGR = {"pre": (0xBB, 0xBB, 0xBB), "acute": (0x77, 0x33, 0xAA), "subacute": (0xAA, 0x77, 0x44),
          "chronic": (0x33, 0x88, 0x22)}
TITLE_S = 3.0          # title card length (s of video)


def far_contra(animal: str) -> str:
    from wfield_local import config
    lat = str((config.animals().get(animal) or {}).get("stroke_laterality", "L"))
    return "far_R" if lat == "L" else "far_L"
PRESET = "3net_20261121_sync60_gapfix60"
MAX_SINGLE_VIEW = 0.10   # candidate licks: <= 10 % of 3-D frames from one view (rendered licks well tracked)
KIN = LC.DLC / "session_poses" / "pose3d_4cam_3net_20261121_sync60" / "kinematics" / "licks.csv.gz"
KIN_FUSED = KIN.parent.parent / "kinematics_fused" / "licks.csv.gz"
EPOCHS = ("pre", "acute", "subacute", "chronic")
TILE = 300
W = 4 * TILE
SLOW = 10
MARGIN = 45            # px around every drawn point (source pixels)
NOSE_MARGIN = 90       # px around the nose, cam1 (plus the stretch ahead of it)
TONGUE_MARGIN = 60     # px around every tongue point and the target spout, cam4


def lick_pool(licks: str = "cam4", animal: str = ANIMAL) -> pd.DataFrame:
    """PS94 far_R FIRST licks per trial. ``cam4`` = `kinematics/licks.csv.gz` (cam4 lick windows on the four-camera 3-D),
    first = lowest lick_idx. ``fused`` (Priya 11-22: every figure on the four-camera fused lick list, kinematics only on
    licks with 3-D support) = `kinematics_fused/licks.csv.gz`, context post_cue, n_3d > 0, cam1-only licks out, first =
    the earliest fused lick after the cue (daq_peak_s)."""
    if licks == "fused":
        A = pd.read_csv(KIN_FUSED)
        A = A[(A.animal == animal) & (A.position == far_contra(animal)) & (A.context == "post_cue") & (A.n_3d > 0)
              & (A.cams.astype(str) != "cam1")]
        return A[A.daq_peak_s == A.groupby(["session", "trial_id"]).daq_peak_s.transform("min")]
    A = pd.read_csv(KIN)
    A = A[(A.animal == animal) & (A.position == far_contra(animal))]
    return A[A.lick_idx == A.groupby(["session", "trial_id"]).lick_idx.transform("min")]


def select(n: int, licks: str = "cam4", animal: str = ANIMAL) -> pd.DataFrame:
    """The rule in the module doc -> one row per epoch: session, epoch median, ranked candidate trials."""
    A = lick_pool(licks, animal)
    A = A[A.date.astype(str).str[4:] >= "0801"]
    A = A[(A.n_3d >= 15) & np.isfinite(A.midline_err_ipsi)]
    rows = []
    for ep in EPOCHS:
        g = A[A.epoch == ep]
        m = float(g.midline_err_ipsi.median())
        s = g.groupby("session").midline_err_ipsi.agg(["median", "size"])
        s = s[s["size"] >= 20]
        sess = (s["median"] - m).abs().sort_values(kind="mergesort").index[0]
        gs = g[(g.session == sess) & (g.share_single_view <= MAX_SINGLE_VIEW)]
        gs = gs.assign(d=(gs.midline_err_ipsi - m).abs()).sort_values(["d", "trial_id"])
        rows.append({"animal": animal, "position": far_contra(animal), "epoch": ep, "session": sess, "epoch_median": m, "epoch_n": len(g),
                     "session_median": float(s.loc[sess, "median"]), "session_n": int(s.loc[sess, "size"]),
                     "cands": [(int(r.trial_id), float(r.midline_err_ipsi), float(r.t_peak_3d), float(r.midline),
                                float(r.midline_target), float(r.reach_mm)) for r in gs.itertuples()]})
    R = pd.DataFrame(rows)
    R["pre_median"] = float(R.loc[R.epoch == "pre", "epoch_median"].iloc[0])
    return R


def _strict(T: dict, tr: int, t0: float, t1: float) -> bool:
    S, U = T["S"], T["U"]
    w = U[(U.trial_id == tr) & (U.off_ms >= t0) & (U.on_ms <= t1)]
    nt = int(((S.trial_id == tr) & (S.t_ms >= t0) & (S.t_ms <= t1) & (S.tongue_views != 0)).sum())
    return len(w) >= 1 and bool(w.keep.astype(bool).all()) and bool((w.n_views >= 3).all()) and nt >= 15


def run_epoch(item) -> dict:
    """Module level (fan_sessions pickles it). item = (row dict, n, out, prep_dir)."""
    row, n, out, prep_dir, sfx = item
    os.environ.update(LC.PRESETS[PRESET]["env"])
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = row["session"].split("_")
    session = f"{animal}:{date}"
    pr = LC.PRESETS[PRESET]
    T = LC.load_tables(LC.DLC / "session_poses" / pr["p3d"], LC.DLC / "session_poses" / "camera_licks" / pr["licks"],
                       animal, date)
    span = T["S"].groupby("trial_id").t_ms.agg(["min", "max"])
    cands = []
    for (tr, err, tpk, md, mdt, reach) in row["cands"]:
        t0, t1 = -100.0, max(700.0, float(np.ceil((tpk + 450.0) / 4.0) * 4.0))
        if tr not in span.index or span.loc[tr, "min"] > t0 or span.loc[tr, "max"] < t1:
            continue
        cands.append((tr, t0, t1, err, tpk, md, mdt, reach, _strict(T, tr, t0, t1)))
    cands = [c for c in cands if c[-1]] + [c for c in cands if not c[-1]]
    cands = cands[:12]
    # one clip spec per wanted trial: spec k takes the candidates from rank k on, skipping trials taken earlier is done
    # after prep (prep picks the first complete one per spec), so give each spec a disjoint candidate stripe
    specs = [{"name": f"{row['epoch']}_{k}", "kind": "trials",
              "cands": [(c[0], c[1], c[2], f"rank {i + 1}{'' if c[-1] else ' (relaxed: not all licks >= 3 cams)'}")
                        for i, c in enumerate(cands) if i % n == k]} for k in range(n)]
    f_prep = Path(prep_dir) / f"supp_{PRESET}_{animal}_{date}_{'_'.join(str(c[0]) for c in cands)}_n{n}.pkl"
    if f_prep.exists():
        P = pickle.loads(f_prep.read_bytes())
    else:
        P = LC.prep_session(rv, session, T, specs, pr["floor_ms"])
        f_prep.write_bytes(pickle.dumps(P))
    if not P["lick_check"][0] and not LC._lick_check_again(rv, session, T, P):
        raise RuntimeError(f"{session}: recomputed 3-D licks != production _licks.csv {P['lick_check'][1:]}")
    info = {c[0]: c for c in cands}
    files = render_supp(rv, P, T, row, info, Path(out), sfx)
    return {"rows": files["rows"], "checks": [{"session": session, **c} for c in P["checks"]],
            "video": files["video"], "stills": files["stills"]}


# ------------------------------------------------------------------------------------------------ drawing

def _crops(P: dict, clips: list, nose_px: dict, mouth_px: dict, tgt_px: dict) -> dict:
    """{cam: (cx, cy, size)} in SOURCE px; see module doc (no clipping, nose / tongue margins)."""
    out = {}
    for j, c in enumerate(P["order"]):
        boxes = []
        pts = [P["spout_px"][c]]
        for cl in clips:
            for p in LC.PARTS:
                q = cl["pts"][p][j][cl["used"][p][j]]
                pts += [q, cl["rep"][p][j]]
        A = np.vstack(pts)
        A = A[np.isfinite(A).all(1)]
        boxes.append((A.min(0) - MARGIN, A.max(0) + MARGIN))
        nz, mo = nose_px[c], mouth_px[c]
        if np.isfinite(nz).all():
            ahead = nz + 0.45 * (nz - mo) if (c == "cam1" and np.isfinite(mo).all()) else nz
            m = NOSE_MARGIN if c == "cam1" else MARGIN
            B = np.vstack([nz, ahead])
            boxes.append((B.min(0) - m, B.max(0) + m))
        if c == "cam4":
            T_ = [tgt_px[c][None]] + [cl["rep"]["tongue"][j] for cl in clips] + \
                 [cl["pts"]["tongue"][j] for cl in clips]
            T_ = np.vstack(T_)
            T_ = T_[np.isfinite(T_).all(1)]
            if len(T_):
                boxes.append((T_.min(0) - TONGUE_MARGIN, T_.max(0) + TONGUE_MARGIN))
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        size = int((hi - lo).max())
        out[c] = ((lo + hi) / 2, size)
    return out


def _cut(im: np.ndarray, centre, size: int) -> np.ndarray:
    """Square crop of ``size`` px at ``centre``; outside the image = black (never shifted / clipped)."""
    import cv2
    pad = size
    big = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x0 = int(round(centre[0] - size / 2)) + pad
    y0 = int(round(centre[1] - size / 2)) + pad
    return cv2.resize(big[y0:y0 + size, x0:x0 + size], (TILE, TILE), interpolation=cv2.INTER_AREA)


class H264Main(LC.H264):
    """As `pose3d_lick_clips.H264`, but Main profile / level 4.0 (opens in PowerPoint / Windows players)."""
    def __init__(self, path: Path, w: int, h: int, fps: int = LC.OUT_FPS):
        import imageio_ffmpeg
        self.p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                   "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-profile:v", "main", "-level", "4.0", "-preset", "medium", "-crf", "18",
                                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                                  stdin=subprocess.PIPE, creationflags=0x08000000)
        self.wh = (w, h)


def render_supp(rv, P: dict, T: dict, row: dict, info: dict, out: Path, sfx: str = "") -> dict:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.pose3d_side_video import BGR, HEX, Reader
    from wfield_local import behavior_events as BE
    from wfield_local import config, epochs
    from wfield_local.orofacial_cohort import pos_key
    animal, date = P["session"].split(":")
    pos_ = row["position"]
    ev = BE.get_or_compute(rv, animal, date)
    daq_s = BE.lick_onsets_s(ev) if ev is not None else np.array([])
    cues = LC.cue_daq_s(LC.PRESETS[PRESET]["env"]["WIDEFIELD_LICK_INPUT_RUN"], animal, date)
    day = epochs.days_since_stroke(f"{animal}_{date[4:]}")
    ep = row["epoch"]
    ep_lab = {"pre": "pre-stroke"}.get(ep, ep)
    day_lab = f"day {day:+d}" if day is not None else ""
    order = P["order"]
    clips = P["clips"]
    tgt = P["tgt"]
    tgt_px = {c: P["spout_px"][c][list(tgt).index(pos_)] for c in order}
    crops = _crops(P, clips, P["nose_px"], P["mouth_px"], tgt_px)
    readers = {c: Reader(P["videos"][c]) for c in order}
    view_order = [c for c in ("cam4", "cam1", "cam2", "cam3") if c in order]
    jidx = {c: order.index(c) for c in order}
    allpos = np.array(list(tgt.values()))
    T_ = tgt[pos_]
    fig, (axt, axs) = plt.subplots(1, 2, figsize=(W / 100, 3.2), dpi=100)
    F_ = cv2.FONT_HERSHEY_SIMPLEX
    out.mkdir(parents=True, exist_ok=True)
    (out / "stills").mkdir(exist_ok=True)
    f_vid = out / f"supp4cam_{animal}_{date}_{ep}_farcontra{sfx}.mp4"
    vw, rows, stills = None, [], []
    L3 = T["L"]
    for cl in clips:
        tr = cl["trial"]
        tr0, t0, t1, err, tpk, md, mdt, reach, strict = info[tr]
        c0 = 1000.0 * cues[tr]
        dq = daq_s * 1000.0 - c0
        dq = dq[(dq >= cl["t0"]) & (dq <= cl["t1"])]
        l3 = L3[(L3.trial_id == tr)]
        l3 = l3[(l3.t_onset_ms + 4.0 * (l3.n_frames - 1) >= cl["t0"]) & (l3.t_onset_ms <= cl["t1"])]
        Xt = cl["X"]["tongue"]
        k_pk = int(np.argmin(np.abs(cl["t_ms"] - tpk)))
        tip_pk = Xt[k_pk] if np.isfinite(Xt[k_pk]).all() else None
        n = len(cl["rows"])
        # panel limits: spouts, mouth zero and this clip's tongue / jaw points (+ 1 mm), not the session's range
        Q = np.vstack([allpos, np.zeros((1, 3)), cl["X"]["tongue"], cl["X"]["jaw"]])
        Q = Q[np.isfinite(Q).all(1)]
        lims = (Q.min(0) - 1.0, Q.max(0) + 1.0)
        for k in range(n):
            t = cl["t_ms"][k]
            tiles = []
            for c in view_order:
                j = jidx[c]
                s_ = cl["src"][c][k]
                im = readers[c].get(int(s_)) if np.isfinite(s_) else None
                if im is None:
                    im = np.zeros((600, 800, 3), np.uint8)
                size = crops[c][1]
                sc = max(1.0, size / TILE)
                for p in LC.PARTS:
                    r_ = cl["rep"][p][j, k]
                    if np.isfinite(r_).all():
                        cv2.drawMarker(im, (int(round(r_[0])), int(round(r_[1]))), BGR[p], cv2.MARKER_TILTED_CROSS,
                                       int(14 * sc), max(1, int(round(1.4 * sc))), cv2.LINE_AA)
                for p in LC.PARTS:
                    q = cl["pts"][p][j, k]
                    if np.isfinite(q).all():
                        xy = (int(round(q[0])), int(round(q[1])))
                        if bool(cl["used"][p][j, k]):
                            cv2.circle(im, xy, int(4 * sc), (0, 0, 0), -1, cv2.LINE_AA)
                            cv2.circle(im, xy, int(3 * sc), BGR[p], -1, cv2.LINE_AA)
                        else:
                            cv2.circle(im, xy, int(6 * sc), BGR[p], max(1, int(round(1.4 * sc))), cv2.LINE_AA)
                tile = _cut(im, crops[c][0], size)
                name = {"cam1": "cam1 (below)", "cam4": "cam4 (front)", "cam2": "cam2 (left side)",
                        "cam3": "cam3 (right side)"}[c]
                cv2.putText(tile, name, (6, 18), F_, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            # panels: top-down (ml x ap) and side (ap x dv), mouth zero at the origin
            sel = (cl["t_ms"] <= t) & (cl["t_ms"] >= t - LC.TRAIL_MS)
            selp = cl["trail_t"] >= t - LC.TRAIL_MS
            for ax, (i, jj, xl, yl) in ((axt, (0, 1, "ml (mm, + = right = contra)", "ap (mm, + = out)")),
                                        (axs, (1, 2, "ap (mm, + = out)", "dv (mm, + = up)"))):
                ax.cla()
                for q, cc in tgt.items():
                    ax.scatter(cc[i], cc[jj], s=60, facecolor=HEX["spout"] if q == pos_ else "none",
                               edgecolor="0.35", lw=0.9, zorder=2)
                ax.scatter(0, 0, s=25, marker="+", color="0.3", zorder=2)
                if ax is axt:
                    ax.plot([0, T_[0] * 1.15], [0, T_[1] * 1.15], ls="--", color=HEX["spout"], lw=1.2, zorder=1)
                for p in ("tongue", "jaw"):
                    A = np.vstack([cl["trail_X"][p][selp], cl["X"][p][sel]])
                    A = A[np.isfinite(A).all(1)]
                    if len(A) > 1:
                        ax.plot(A[:, i], A[:, jj], color=HEX[p], lw=1.6, zorder=3)
                for p in LC.PARTS[:3]:
                    Xp = cl["X"][p][k]
                    if np.isfinite(Xp).all():
                        ax.scatter(Xp[i], Xp[jj], s=34, color=HEX[p], edgecolor="k", lw=0.5, zorder=4)
                if ax is axt and tip_pk is not None and t >= tpk:
                    sc_ = 1.15 * np.linalg.norm(T_[:2]) / max(np.linalg.norm(tip_pk[:2]), 1e-6)
                    ax.plot([0, tip_pk[0] * sc_], [0, tip_pk[1] * sc_], color=HEX["tongue"], lw=1.4, zorder=3)
                    ax.scatter(tip_pk[0], tip_pk[1], s=50, marker="x", color=HEX["tongue"], zorder=5)
                    ax.text(0.02, 0.96, f"first lick: aim error {err:+.0f}\N{DEGREE SIGN} (+ = toward ipsi)",
                            transform=ax.transAxes, fontsize=8, va="top", color="#c00000")
                lo, hi = lims
                ax.set_xlim(lo[i], hi[i])
                ax.set_ylim(lo[jj], hi[jj])
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel(xl, fontsize=7)
                ax.set_ylabel(yl, fontsize=7)
                ax.tick_params(labelsize=6)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
            axt.set_title("top-down (mouth at +; dashed = direction of the target spout, solid = first lick)",
                          fontsize=7.5)
            axs.set_title("side (profile)", fontsize=7.5)
            fig.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.90, wspace=0.18)
            fig.canvas.draw()
            im3 = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
            # slim timeline
            st = np.full((68, W, 3), 18, np.uint8)
            a0, a1 = cl["t0"], cl["t1"]
            X = lambda tt, a0=a0, a1=a1: LC._x(np.clip(tt, a0, a1), a0, a1)   # noqa: E731
            cv2.putText(st, "trial", (6, 15), F_, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(st, "DAQ contact", (6, 31), F_, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(st, "3-D licks", (6, 47), F_, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.rectangle(st, (X(cl["t0"]), 5), (X(0), 17), (70, 70, 70), -1)
            cv2.rectangle(st, (X(0), 5), (X(cl["stop_ms"]), 17), (60, 120, 40), -1)
            for d in dq:
                cv2.line(st, (X(d), 22), (X(d), 34), (80, 80, 255), 2)
            for r in l3.itertuples():
                a, b = float(r.t_onset_ms), float(r.t_onset_ms) + 4.0 * (int(r.n_frames) - 1)
                first = abs(float(r.t_peak_ms) - tpk) <= 8.0
                cv2.rectangle(st, (X(a), 38), (max(X(b), X(a) + 1), 50), (60, 60, 255) if first else (150, 150, 230),
                              -1)
            xc = X(t)
            cv2.line(st, (xc, 2), (xc, 54), (0, 255, 255), 1)
            for tt in np.arange(np.ceil(cl["t0"] / 100) * 100, cl["t1"] + 1e-6, 100):
                cv2.putText(st, f"{tt:+.0f}", (X(tt) - 12, 64), F_, 0.33, (170, 170, 170), 1, cv2.LINE_AA)
            # header
            bar = np.zeros((70, W, 3), np.uint8)
            bar[:22] = EP_BGR[ep]
            cv2.putText(bar, f"{animal}   {ep_lab.upper()}   {day_lab}   far contra spout", (8, 16), F_, 0.55,
                        (0, 0, 0) if ep == "pre" else (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bar, f"{animal}  {date[:4]}-{date[4:6]}-{date[6:]}  {ep_lab}  {day_lab}  trial {tr}  spout far "
                        f"contra ({pos_}, {pos_key(pos_, str((config.animals().get(animal) or {}).get('stroke_laterality', 'L'))).replace('_', ' ')})   t = {t:+.0f} ms from cue   "
                        f"{SLOW}x slow motion", (8, 41), F_, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bar, f"first-lick aim error {err:+.0f} deg (epoch median {row['epoch_median']:+.0f}, n = "
                        f"{row['epoch_n']}; pre-stroke median {row['pre_median']:+.0f}) | dot = 2-D keypoint used, ring = rejected, x = 3-D point "
                        f"reprojected | nose yellow, jaw green, tongue red, spout blue", (8, 61), F_, 0.4,
                        (220, 220, 220), 1, cv2.LINE_AA)
            frame = np.vstack([bar, np.hstack(tiles), cv2.resize(im3, (W, int(W * im3.shape[0] / im3.shape[1]))), st])
            if frame.shape[0] % 2:
                frame = np.vstack([frame, np.zeros((1, W, 3), np.uint8)])
            if vw is None:
                vw = H264Main(f_vid, frame.shape[1], frame.shape[0])
            vw.write(frame)
            if k == k_pk:
                p_ = out / "stills" / f"{f_vid.stem}_t{tr}_peak_{t:+.0f}ms.png"
                cv2.imwrite(str(p_), frame)
                stills.append(str(p_))
        for _ in range(LC.HOLD_FRAMES):
            vw.write(frame)
        rows.append({"file": f_vid.name, "session": f"{animal}_{date}", "epoch": ep, "day": day, "trial": tr,
                     "position": pos_, "t0_ms": cl["t0"], "t1_ms": cl["t1"], "first_lick_peak_ms": round(tpk),
                     "aim_err_deg": round(err, 1), "lick_midline_deg": round(md, 1),
                     "target_midline_deg": round(mdt, 1), "reach_mm": round(reach, 2),
                     "epoch_median_deg": round(row["epoch_median"], 1), "pre_median_deg": round(row["pre_median"], 1), "epoch_n": row["epoch_n"],
                     "session_median_deg": round(row["session_median"], 1), "session_n": row["session_n"],
                     "rule": cl["rule"], "crop_px": "; ".join(f"{c} {crops[c][1]}" for c in view_order)})
    if vw is not None:
        vw.close()
    plt.close(fig)
    return {"rows": rows, "video": str(f_vid), "stills": stills, "frame_hw": frame.shape[:2]}


def concat(files: list, f_out: Path) -> None:
    import imageio_ffmpeg
    lst = f_out.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(f).as_posix()}'\n" for f in files))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                    str(lst), "-c:v", "libx264", "-profile:v", "main", "-level", "4.0", "-preset", "medium", "-crf",
                    "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(f_out)], check=True,
                   creationflags=0x08000000)
    lst.unlink()


def title_card(f_out: Path, hw: tuple, R: pd.DataFrame, ep: str, licks: str) -> Path:
    """A TITLE_S-second card before each epoch: animal, epoch, post-stroke day, spout, epoch-median aim error and its
    change from pre (frame size ``hw`` = the clips')."""
    import cv2

    from wfield_local import epochs
    r = R[R.epoch == ep].iloc[0]
    day = epochs.days_since_stroke(f"{r.animal}_{r.session.split('_')[1][4:]}")
    h, w = hw
    im = np.full((h, w, 3), 16, np.uint8)
    im[: int(0.22 * h)] = EP_BGR[ep]
    F_ = cv2.FONT_HERSHEY_SIMPLEX
    fg = (0, 0, 0) if ep == "pre" else (255, 255, 255)
    ep_lab = {"pre": "pre-stroke"}.get(ep, ep)
    cv2.putText(im, f"{r.animal}  -  {ep_lab}", (40, int(0.14 * h)), F_, 1.6, fg, 3, cv2.LINE_AA)
    lines = [f"post-stroke day {day:+d} (session {r.session.split('_')[1]})" if day is not None else r.session,
             f"far contra spout ({r.position}; lesion left)",
             f"first-lick aim error, epoch median {r.epoch_median:+.0f} deg (n = {r.epoch_n} first licks)",
             ("pre-stroke reference" if ep == "pre" else f"change from pre: {r.epoch_median - r.pre_median:+.0f} deg"),
             "+ = toward ipsi; lick list: " + ("four-camera fused, 3-D supported" if licks == "fused"
                                              else "cam4 lick windows"),
             "2 example trials (rule-selected near the epoch median), 10x slow motion"]
    for k, t in enumerate(lines):
        cv2.putText(im, t, (40, int(0.34 * h) + k * 46), F_, 0.85 if k < 4 else 0.62, (235, 235, 235),
                    2 if k < 4 else 1, cv2.LINE_AA)
    vw = H264Main(f_out, w, h)
    for _ in range(int(TITLE_S * LC.OUT_FPS)):
        vw.write(im)
    vw.close()
    return f_out


def _frame_hw(f) -> tuple:
    import re

    import imageio_ffmpeg
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(f)], capture_output=True,
                          text=True, creationflags=0x08000000).stderr
    w, h = map(int, re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})", info).groups())
    return h, w


def win_copy(f: Path) -> Path:
    """Maximally compatible copy for the default Windows players (Priya 11-22: the earlier clips, even Main-profile
    copies, opened only in VLC): Baseline profile, level 3.1, no B-frames, yuv420p, dimensions padded to a multiple of
    16, width <= 1920, constant 30 fps (frames repeated, so the duration and the slow-motion factor stay the same),
    avc1 tag, faststart -> ``<stem>_win.mp4``."""
    import re

    import imageio_ffmpeg
    f = Path(f)
    g = f.with_name(f.stem + "_win.mp4")
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(f)], capture_output=True,
                          text=True, creationflags=0x08000000).stderr
    w, h = map(int, re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})", info).groups())
    # level 3.1 allows 3600 macroblocks per frame (and 108000 / s, i.e. 3600 at 30 fps): scale down if needed
    s = min(1.0, 1920 / w, np.sqrt(3600 * 256 / (np.ceil(w / 16) * 16 * np.ceil(h / 16) * 16)))
    w2 = int(np.floor(w * s / 16) * 16)
    h2 = int(np.floor(h * s / 2) * 2)
    vf = (f"scale={w2}:{h2}:flags=lanczos,"
          "pad=ceil(iw/16)*16:ceil(ih/16)*16:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(f), "-an", "-vf", vf,
                    "-r", "30", "-vsync", "cfr", "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
                    "-bf", "0", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-tag:v", "avc1",
                    "-movflags", "+faststart", str(g)], check=True, creationflags=0x08000000)
    return g


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2, help="trials per epoch")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--animal", default=ANIMAL)
    ap.add_argument("--licks", choices=("cam4", "fused"), default="cam4",
                    help="lick list for the selection (fused -> outputs suffixed _fused)")
    ap.add_argument("--epochs", nargs="*", default=list(EPOCHS), help="epochs to render (others: the existing clip "
                    "of the same suffix, else the unsuffixed cam4-list clip, is used for the 4-epoch file)")
    ap.add_argument("--win-only", action="store_true", help="only write the _win.mp4 copies of the clips in --out")
    a = ap.parse_args(argv)
    if not str(a.out).replace("\\", "/").startswith("M:/MICROSCOPE/Priya/"):
        raise SystemExit("--out must be under M:/MICROSCOPE/Priya/")
    if a.win_only:
        for f in sorted(a.out.glob("supp4cam_*.mp4")):
            if not f.stem.endswith("_win"):
                print("win copy", win_copy(f))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    sfx = "" if a.licks == "cam4" else f"_{a.licks}"
    R = select(a.n, a.licks, a.animal)
    print(R[["epoch", "session", "epoch_median", "epoch_n", "session_median", "session_n"]].to_string())
    LC.PREP_DIR.mkdir(parents=True, exist_ok=True)
    items = [(r, a.n, str(a.out), str(LC.PREP_DIR), sfx) for r in R.to_dict("records") if r["epoch"] in a.epochs]
    res, fails = ak.fan_sessions(items, run_epoch, jobs=min(a.jobs, len(items)), key=lambda it: it[0]["session"],
                                  label="epoch")
    by_ep = {v["rows"][0]["epoch"]: v for _, v in res if v["rows"]}
    rows = [r for ep in EPOCHS if ep in by_ep for r in by_ep[ep]["rows"]]
    pd.DataFrame(rows).to_csv(a.out / f"clips{sfx}.csv", index=False)
    pd.DataFrame([c for ep in EPOCHS if ep in by_ep for c in by_ep[ep]["checks"]]).to_csv(
        a.out / f"checks_vs_production{sfx}.csv", index=False)
    vids = [by_ep[ep]["video"] for ep in EPOCHS if ep in by_ep]
    seq = []
    for ep in EPOCHS:
        if ep in by_ep:
            seq.append(by_ep[ep]["video"])
            continue
        old = (sorted(a.out.glob(f"supp4cam_{a.animal}_*_{ep}_farcontra{sfx}.mp4"))
               or sorted(a.out.glob(f"supp4cam_{a.animal}_*_{ep}_farcontra.mp4")))
        if old:
            seq.append(str(old[0]))
    if len(seq) == len(EPOCHS):
        # THE supplementary deliverable (Priya 11-22): one video per animal, epochs in order, a title card before
        # each epoch, the epoch banner in every frame (the per-epoch clips stay as building blocks)
        hw = _frame_hw(seq[0])
        parts = []
        for ep, v in zip(EPOCHS, seq):
            parts += [str(title_card(a.out / f"_title_{a.animal}_{ep}{sfx}.mp4", hw, R, ep, a.licks)), v]
        f4 = a.out / f"supp4cam_{a.animal}_farcontra_allepochs{sfx}.mp4"
        print("all-epochs file from", *seq, sep="\n  ")
        concat(parts, f4)
        for q in parts:
            if Path(q).name.startswith("_title_"):
                Path(q).unlink()
        vids.append(str(f4))
    for v in vids:
        print("win copy", win_copy(Path(v)))
    for ep in EPOCHS:
        if ep in by_ep:
            print(ep, by_ep[ep]["video"], *by_ep[ep]["stills"], sep="\n  ")
    for it, err in fails:
        print("FAILED", it[0]["epoch"], err)
    print(pd.DataFrame(rows).to_string())
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
