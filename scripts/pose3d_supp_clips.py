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
#: STYLE v2 (Priya 11-22, supplementary deliverable): manuscript animal numbers, no lab IDs / dates / trial numbers in any
#: frame; Fig 3C v26c framing (whole face in front + both sides, nose AND jaw in the bottom view) and keypoint style
#: (small filled dots, colour-coded word labels, the bottom-view nose reprojected from 3-D and marked as such)
ANIMAL_KEY = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/ANIMAL_KEY.csv")
V2_VIEWS = (("cam4", "front"), ("cam1", "bottom"), ("cam2", "left side"), ("cam3", "right side"))
COL_V2 = {"tongue": "#ff3b1f", "jaw": "#2b8cff", "spout": "#b45ff0", "nose": "#00e5ff"}      # fig3c COL
V2_MARGIN = 30         # px (source) around every keypoint of the epoch's clips, added to the v26c crop
#: word-label offsets (dx, dy, right-aligned) in tile px
V2_LABEL = {"nose": (7, -7, False), "jaw": (7, 13, False), "tongue": (-7, 14, True), "spout": (7, -9, False)}


def mouse_label(animal: str) -> str:
    K = pd.read_csv(ANIMAL_KEY, dtype=str)
    return str(K.set_index("lab_id").loc[animal, "manuscript_label"])


def _bgr(h: str) -> tuple:
    return int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)


def v2_tiles() -> tuple[int, dict]:
    """Tile height and widths (sum = W) from the Fig 3C v26c crop aspect ratios."""
    from scripts.fig3c_four_view_strip import CROPS_V26B
    asp = {c: CROPS_V26B[c][2] / CROPS_V26B[c][3] for c, _ in V2_VIEWS}
    th = int(W / sum(asp.values())) // 2 * 2
    tw = {c: int(round(th * a)) for c, a in asp.items()}
    tw["cam3"] += W - sum(tw.values())
    return th, tw


def _crops_v2(P: dict, clips: list) -> dict:
    """{cam: (x0, y0, cw, ch)} in SOURCE px: the Fig 3C v26c crop (same rig and cameras), grown to hold every keypoint
    of the epoch's clips (2-D and 3-D reprojected, incl. the bottom-view nose from 3-D) with ``V2_MARGIN``, at the v26c
    aspect ratio; parts outside the image are padded black."""
    from scripts.fig3c_four_view_strip import CROPS_V26B
    out = {}
    for c, _ in V2_VIEWS:
        j = P["order"].index(c)
        x0, y0, cw, ch = CROPS_V26B[c]
        pts = [P["spout_px"][c]]
        for cl in clips:
            for q in ("nose", "jaw", "tongue", "spout"):
                pts += [cl["pts"][q][j], cl["rep"][q][j]]
        A = np.vstack(pts)
        A = A[np.isfinite(A).all(1)]
        lo = np.minimum([x0, y0], A.min(0) - V2_MARGIN)
        hi = np.maximum([x0 + cw, y0 + ch], A.max(0) + V2_MARGIN)
        asp = cw / ch
        w_, h_ = hi - lo
        w_, h_ = max(w_, h_ * asp), max(h_, w_ / asp)
        cx, cy = (lo + hi) / 2
        out[c] = (cx - w_ / 2, cy - h_ / 2, w_, h_)
    return out


def _cut_rect(im: np.ndarray, box, size) -> np.ndarray:
    import cv2
    x0, y0, cw, ch = box
    pad = int(max(cw, ch))
    big = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    xa, ya = int(round(x0)) + pad, int(round(y0)) + pad
    return cv2.resize(big[ya:ya + int(round(ch)), xa:xa + int(round(cw))], size, interpolation=cv2.INTER_AREA)


def _txt(im, text, org, scale, col, right=False, thick=1):
    import cv2
    F_ = cv2.FONT_HERSHEY_SIMPLEX
    if right:
        org = (org[0] - cv2.getTextSize(text, F_, scale, thick)[0][0], org[1])
    cv2.putText(im, text, org, F_, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(im, text, org, F_, scale, col, thick, cv2.LINE_AA)


def epoch_title(ep: str, day) -> str:
    return "Pre-stroke" if ep == "pre" else f"Post-stroke day {day} ({ep})"


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


PRE_RULE_V2 = ("pre-stroke: first licks that CONTACTED the spout (DAQ), tracked in all four views (cams = all four, "
               "<= 10 % single-view 3-D frames); session = the pre session (>= 20 contacted first licks) whose 25th "
               "percentile of ipsilesional aim error is closest to the epoch's; trials ranked by distance from that 25th "
               "percentile (the better-aimed quarter of contacted licks)")
POST_RULE = ("post-stroke epochs: session = the one (>= 20 first licks) whose median aim error is closest to the epoch "
             "median; its first licks with <= 10 % single-view 3-D frames ranked by distance from the epoch median")


LAT_RULE = ("latency: first-lick peak time after the cue (t_peak_3d) within the interquartile range of that animal's "
            "first licks in the same epoch (pre-stroke: all pre-stroke far-contralesional first licks)")


def select(n: int, licks: str = "cam4", animal: str = ANIMAL, pre_rule: str = "median",
           lat_iqr: bool = False) -> pd.DataFrame:
    """The rule in the module doc -> one row per epoch: session, epoch median, ranked candidate trials.
    ``pre_rule`` = "contact_q25" (style v2, Priya 11-22): the pre epoch uses `PRE_RULE_V2` instead."""
    A = lick_pool(licks, animal)
    A = A[A.date.astype(str).str[4:] >= "0801"]
    A = A[(A.n_3d >= 15) & np.isfinite(A.midline_err_ipsi)]
    rows = []
    for ep in EPOCHS:
        g = A[A.epoch == ep]
        m = float(g.midline_err_ipsi.median())
        lq = np.percentile(g.t_peak_3d.dropna(), [25, 50, 75])
        in_lat = (g.t_peak_3d >= lq[0]) & (g.t_peak_3d <= lq[2]) if lat_iqr else pd.Series(True, index=g.index)
        if ep == "pre" and pre_rule == "contact_q25":
            c = g[g.contact.astype(str).isin(["True", "1", "1.0"])]
            q = float(np.percentile(c.midline_err_ipsi, 25))
            s = c.groupby("session").midline_err_ipsi.agg([lambda v: np.percentile(v, 25), "size"])
            s.columns = ["q25", "size"]
            s = s[s["size"] >= 20]
            sess = (s["q25"] - q).abs().sort_values(kind="mergesort").index[0]
            gs = c[(c.session == sess) & (c.share_single_view <= MAX_SINGLE_VIEW)
                   & (c.cams.astype(str) == "cam1+cam2+cam3+cam4") & in_lat.loc[c.index]]
            gs = gs.assign(d=(gs.midline_err_ipsi - q).abs()).sort_values(["d", "trial_id"])
            rows.append({"animal": animal, "position": far_contra(animal), "epoch": ep, "session": sess,
                         "epoch_median": m, "epoch_n": len(g), "session_median": float(s.loc[sess, "q25"]),
                         "session_n": int(s.loc[sess, "size"]), "rule": PRE_RULE_V2, "target_value": q,
                         "contact_frac": float(len(c) / len(g)), "lat_q25_50_75": lq.round(0).tolist(),
                         "cands": [(int(r.trial_id), float(r.midline_err_ipsi), float(r.t_peak_3d), float(r.midline),
                                    float(r.midline_target), float(r.reach_mm)) for r in gs.itertuples()]})
            continue
        s = g.groupby("session").midline_err_ipsi.agg(["median", "size"])
        s = s[s["size"] >= 20]
        sess = (s["median"] - m).abs().sort_values(kind="mergesort").index[0]
        gs = g[(g.session == sess) & (g.share_single_view <= MAX_SINGLE_VIEW) & in_lat]
        gs = gs.assign(d=(gs.midline_err_ipsi - m).abs()).sort_values(["d", "trial_id"])
        rows.append({"animal": animal, "position": far_contra(animal), "epoch": ep, "session": sess,
                     "epoch_median": m, "epoch_n": len(g), "rule": POST_RULE, "target_value": m,
                     "lat_q25_50_75": lq.round(0).tolist(),
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
    row, n, out, prep_dir, sfx = item[:5]
    style = item[5] if len(item) > 5 else "v1"
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
    files = render_supp(rv, P, T, row, info, Path(out), sfx, style)
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


def render_supp(rv, P: dict, T: dict, row: dict, info: dict, out: Path, sfx: str = "", style: str = "v1") -> dict:
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
    v2 = style in ("v2", "v3", "v5")      # v5 (11-24) renders as v3; v3 (11-22): v2 + cam1 nose only where cam1 tracked it + latency-IQR picks
    if v2:
        HEX = COL_V2
        crops2 = _crops_v2(P, clips)
        th, tw = v2_tiles()
        mouse = mouse_label(animal)
        ep_title = epoch_title(ep, day)
        from wfield_local import trial_classes as TC
        TT = TC.session_trials(f"{animal}_{date[4:]}").set_index("trial_id")
        rew_all = np.asarray(ev["reward_samples"], float) / float(ev["fs"]) if ev is not None else np.array([])
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
        if v2:
            # manuscript sign (+ = contralesional): the error vs the target direction, flipped from + = ipsilesional
            errc = -err
            strobe_ms = float(TT.loc[tr, "strobe_ms"]) if tr in TT.index else np.nan
            rw = (rew_all - cues[tr]) * 1000.0
            rw = rw[(rw >= -50.0) & (rw <= 1000.0)]
            rew_ms = float(rw[0]) if len(rw) else np.nan
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
                if v2:
                    s_ = cl["src"][c][k]
                    im = readers[c].get(int(s_)) if np.isfinite(s_) else None
                    if im is None:
                        im = np.zeros((600, 800, 3), np.uint8)
                    box = crops2[c]
                    tile = _cut_rect(im, box, (tw[c], th))
                    sx, sy = tw[c] / box[2], th / box[3]
                    for q in ("nose", "jaw", "spout", "tongue"):
                        xy2, rep3 = cl["pts"][q][j, k], False
                        if style in ("v3", "v5") and q == "nose" and c == "cam1":
                            # v3 (Priya 11-22): the bottom-view nose is ALWAYS the fused 3-D nose reprojected (smooth;
                            # never cam1's own point, never switching between the two), marked as Fig 3C v26c
                            if not np.isfinite(cl["rep"][q][j, k]).all():
                                continue
                            xy2, rep3 = cl["rep"][q][j, k], True
                        elif not (np.isfinite(xy2).all() and bool(cl["used"][q][j, k])):
                            # untracked (or rejected) in this view: the fused 3-D point reprojected, if any -- drawn
                            # only for the nose in the bottom view (cam1 has no nose output), as Fig 3C v26c
                            if q == "nose" and c == "cam1" and style == "v2" and np.isfinite(cl["rep"][q][j, k]).all():
                                xy2, rep3 = cl["rep"][q][j, k], True
                            else:
                                continue
                        x_, y_ = int(round((xy2[0] - box[0]) * sx)), int(round((xy2[1] - box[1]) * sy))
                        col = _bgr(COL_V2[q])
                        if rep3:
                            cv2.circle(tile, (x_, y_), 5, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(tile, (x_, y_), 3, col, -1, cv2.LINE_AA)
                        dx, dy, right = V2_LABEL[q]
                        _txt(tile, q + (" (3-D)" if rep3 else ""), (x_ + dx, y_ + dy), 0.4, col, right)
                    _txt(tile, dict(V2_VIEWS)[c], (6, 16), 0.48, (255, 255, 255))
                    tiles.append(tile)
                    continue
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
            for ax, (i, jj, xl, yl) in ((axt, (0, 1, "ml (mm, + = right = contralesional)", "ap (mm, + = out)")),
                                        (axs, (1, 2, "ap (mm, + = out)", "dv (mm, + = up)"))):
                ax.cla()
                for q, cc in tgt.items():
                    ax.scatter(cc[i], cc[jj], s=60, facecolor=HEX["spout"] if q == pos_ else "none",
                               edgecolor="0.35", lw=0.9, zorder=2)
                ax.scatter(0, 0, s=25, marker="+", color="0.3", zorder=2)
                if ax is axt:
                    ax.plot([0, T_[0] * 1.15], [0, T_[1] * 1.15], ls="--", color="#5a7fa8" if v2 else HEX["spout"],
                            lw=1.0 if v2 else 1.2, zorder=1, label="target direction" if v2 else None)
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
                    ax.plot([0, tip_pk[0] * sc_], [0, tip_pk[1] * sc_], color=HEX["tongue"], lw=1.6 if v2 else 1.4,
                            ls=":" if v2 else "-", zorder=3, label="first-lick direction" if v2 else None)
                    ax.scatter(tip_pk[0], tip_pk[1], s=50, marker="x", color=HEX["tongue"], zorder=5)
                    ax.text(0.02, 0.96, (f"first lick: {errc:+.0f}\N{DEGREE SIGN} vs target (- = ipsilesional)" if v2
                                         else f"first lick: aim error {err:+.0f}\N{DEGREE SIGN} (+ = toward ipsilesional)"),
                            transform=ax.transAxes, fontsize=8, va="top", color="#c00000")
                if ax is axt and v2:
                    ax.legend(loc="lower right", fontsize=6.5, frameon=False, handlelength=2.2)
                lo, hi = lims
                ax.set_xlim(lo[i], hi[i])
                ax.set_ylim(lo[jj], hi[jj])
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel(xl, fontsize=7)
                ax.set_ylabel(yl, fontsize=7)
                ax.tick_params(labelsize=6)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
            axt.set_title("top-down (mouth at +; dashed = target direction, " +
                          ("dotted = first-lick direction at peak extension)" if v2 else "solid = first lick)"),
                          fontsize=7.5)
            axs.set_title("side (profile)", fontsize=7.5)
            fig.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.90, wspace=0.18)
            fig.canvas.draw()
            im3 = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
            # slim timeline
            st = np.full((88 if v2 else 68, W, 3), 18, np.uint8)
            a0, a1 = cl["t0"], cl["t1"]
            X = lambda tt, a0=a0, a1=a1: LC._x(np.clip(tt, a0, a1), a0, a1)   # noqa: E731
            cv2.putText(st, "trial", (6, 15), F_, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(st, "spout contact" if v2 else "DAQ contact", (6, 31), F_, 0.4, (220, 220, 220), 1,
                        cv2.LINE_AA)
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
            if v2:
                # cue + water at 0 (tone and water together; the DAQ reward valve opens rew_ms after the cue)
                cv2.line(st, (X(0), 2), (X(0), 54), (0, 220, 255), 2)
                cv2.putText(st, "cue" if style == "v2" else "cue + water", (X(0) + 4, 30), F_, 0.36, (0, 220, 255), 1,
                            cv2.LINE_AA)
                if style == "v2" and np.isfinite(rew_ms) and abs(rew_ms) > 4.0:
                    cv2.drawMarker(st, (X(rew_ms), 8), (255, 200, 0), cv2.MARKER_TRIANGLE_DOWN, 10, 2, cv2.LINE_AA)
                if np.isfinite(strobe_ms):
                    txt = (f"<- spout arrived {strobe_ms / 1000:+.1f} s" if strobe_ms < cl["t0"] else "spout arrives")
                    if strobe_ms < cl["t0"]:
                        cv2.putText(st, txt, (LC.STRIP_L - 2, 16), F_, 0.33, (200, 200, 200), 1, cv2.LINE_AA)
                    else:
                        cv2.line(st, (X(strobe_ms), 2), (X(strobe_ms), 54), (200, 200, 200), 1)
                stp = cl["stop_ms"]
                if np.isfinite(stp) and stp > cl["t1"]:
                    tx = f"spout leaves {stp / 1000:+.1f} s ->"
                    cv2.putText(st, tx, (W - 14 - cv2.getTextSize(tx, F_, 0.33, 1)[0][0], 16), F_, 0.33,
                                (200, 200, 200), 1, cv2.LINE_AA)
                elif np.isfinite(stp):
                    cv2.line(st, (X(stp), 2), (X(stp), 54), (200, 200, 200), 1)
                cv2.putText(st, "ms from cue", (6, 64), F_, 0.33, (170, 170, 170), 1, cv2.LINE_AA)
                xl = 6
                # v3 (Priya 11-22): cue and water are functionally simultaneous -> ONE marker at 0 (the DAQ valve
                # opens 6-12 ms after the cue on ~88 % of trials, 110-130 ms on ~11 %, in every session checked;
                # auto_reward_delay = 0 in gui_config: controller latency, not a set delay)
                if style == "v2":
                    ev_lab = ((f"cue + water at 0 (valve {rew_ms:+.0f} ms)" if np.isfinite(rew_ms) else "cue at 0",
                               (0, 220, 255)), ("water valve (DAQ)", (255, 200, 0)))
                else:
                    ev_lab = (("cue + water (delivered together)", (0, 220, 255)),)
                for lab, col in (*ev_lab,
                                 ("trial period (spout in place)", (60, 120, 40)), ("spout contact", (80, 80, 255)),
                                 ("3-D lick (first lick bright)", (60, 60, 255)), ("time cursor", (0, 255, 255))):
                    cv2.rectangle(st, (xl, 74), (xl + 12, 83), col, -1)
                    cv2.putText(st, lab, (xl + 16, 83), F_, 0.34, (200, 200, 200), 1, cv2.LINE_AA)
                    xl += 26 + cv2.getTextSize(lab, F_, 0.34, 1)[0][0]
            # header
            if v2:
                bar = np.zeros((70, W, 3), np.uint8)
                bar[:24] = EP_BGR[ep]
                fg = (0, 0, 0) if ep == "pre" else (255, 255, 255)
                cv2.putText(bar, f"Mouse {mouse}   {ep_title}   far contralesional spout", (8, 17), F_, 0.58, fg, 1,
                            cv2.LINE_AA)
                ex = [c_["trial"] for c_ in clips].index(tr) + 1
                cv2.putText(bar, f"example lick {ex} of {len(clips)}   {SLOW}x slow motion   first-lick aim error vs "
                            f"target {errc:+.0f} deg (epoch median {-row['epoch_median']:+.0f}; " + ("pre-stroke reference"
                            if ep == "pre" else f"change from pre-stroke "
                            f"{-(row['epoch_median'] - row['pre_median']):+.0f}") + "; - = ipsilesional)",
                            (8, 42), F_, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                tt_ = f"t = {t:+.0f} ms"
                tw_ = cv2.getTextSize(tt_, F_, 0.62, 2)[0][0]
                cv2.putText(bar, tt_, (W - tw_ - 10, 18), F_, 0.62, fg, 2, cv2.LINE_AA)
                if 0.0 <= t < 200.0:          # CUE + WATER badge for 200 ms of real time (2 s of video at 10x)
                    bx1 = W - tw_ - 24
                    cv2.rectangle(bar, (bx1 - 150, 1), (bx1, 23), (0, 220, 255), -1)
                    cv2.rectangle(bar, (bx1 - 150, 1), (bx1, 23), (0, 0, 0), 1)
                    cv2.putText(bar, "CUE + WATER", (bx1 - 140, 18), F_, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(bar, ("dots = tracked keypoints used for the 3-D point (white-ringed = reprojected from "
                             "3-D)" if style == "v2" else "dots = keypoints tracked in each view and used for the 3-D "
                             "point; bottom-view nose = 3-D nose reprojected (white ring)") +
                            " | tongue red, jaw blue, spout purple, nose cyan | panels: 3-D, mouth at +",
                            (8, 62), F_, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            else:
                bar = np.zeros((70, W, 3), np.uint8)
                bar[:22] = EP_BGR[ep]
                cv2.putText(bar, f"{animal}   {ep_lab.upper()}   {day_lab}   far contra spout", (8, 16), F_, 0.55,
                            (0, 0, 0) if ep == "pre" else (255, 255, 255), 1, cv2.LINE_AA)
                lat = str((config.animals().get(animal) or {}).get("stroke_laterality", "L"))
                cv2.putText(bar, f"{animal}  {date[:4]}-{date[4:6]}-{date[6:]}  {ep_lab}  {day_lab}  trial {tr}  spout "
                            f"far contra ({pos_}, {pos_key(pos_, lat).replace('_', ' ')})   t = {t:+.0f} ms from cue   "
                            f"{SLOW}x slow motion", (8, 41), F_, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(bar, f"first-lick aim error {err:+.0f} deg (epoch median {row['epoch_median']:+.0f}, n = "
                            f"{row['epoch_n']}; pre-stroke median {row['pre_median']:+.0f}) | dot = 2-D keypoint used, "
                            f"ring = rejected, x = 3-D point reprojected | nose yellow, jaw green, tongue red, spout blue",
                            (8, 61), F_, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            row_t = np.hstack(tiles)
            frame = np.vstack([bar, row_t, cv2.resize(im3, (W, int(W * im3.shape[0] / im3.shape[1]))), st])
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
        if v2:
            rows.append({"file": f_vid.name, "session": f"{animal}_{date}", "epoch": ep, "day": day, "trial": tr,
                         "position": pos_, "t0_ms": cl["t0"], "t1_ms": cl["t1"], "first_lick_peak_ms": round(tpk),
                         "aim_err_vs_target_deg_pos_contra": round(-err, 1),
                         "epoch_median_vs_target_deg_pos_contra": round(-row["epoch_median"], 1),
                         "pre_median_vs_target_deg_pos_contra": round(-row["pre_median"], 1),
                         "epoch_change_from_pre_deg_pos_contra": round(-(row["epoch_median"] - row["pre_median"]), 1),
                         "epoch_n": row["epoch_n"], "rule": row["rule"], "rank": cl["rule"],
                         "cue_to_water_valve_ms": round(rew_ms, 1) if np.isfinite(rew_ms) else None,
                         "strobe_ms": round(strobe_ms, 1) if np.isfinite(strobe_ms) else None,
                         "stop_ms": round(cl["stop_ms"], 1), "style": style,
                         "crop_px": "; ".join(f"{c} {int(crops2[c][0])},{int(crops2[c][1])},{int(crops2[c][2])}x"
                                              f"{int(crops2[c][3])}" for c, _ in V2_VIEWS)})
            continue
        rows.append({"file": f_vid.name, "session": f"{animal}_{date}", "epoch": ep, "day": day, "trial": tr,
                     "position": pos_, "t0_ms": cl["t0"], "t1_ms": cl["t1"], "first_lick_peak_ms": round(tpk),
                     "aim_err_deg": round(err, 1), "lick_midline_deg": round(md, 1),
                     "target_midline_deg": round(mdt, 1), "reach_mm": round(reach, 2),
                     "epoch_median_deg": round(row["epoch_median"], 1), "pre_median_deg": round(row["pre_median"], 1), "epoch_n": row["epoch_n"],
                     "session_median_deg": round(row["session_median"], 1), "session_n": row["session_n"],
                     "rule": cl["rule"], "style": style,
                     "crop_px": "; ".join(f"{c} {crops[c][1]}" for c in view_order) if not v2 else
                     "; ".join(f"{c} {int(crops2[c][0])},{int(crops2[c][1])},{int(crops2[c][2])}x{int(crops2[c][3])}"
                               for c, _ in V2_VIEWS)})
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


def title_card(f_out: Path, hw: tuple, R: pd.DataFrame, ep: str, licks: str, style: str = "v1") -> Path:
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
    if style not in ("v2", "v3"):
        cv2.putText(im, f"{r.animal}  -  {ep_lab}", (40, int(0.14 * h)), F_, 1.6, fg, 3, cv2.LINE_AA)
    if style in ("v2", "v3"):
        cv2.putText(im, f"Mouse {mouse_label(r.animal)}  -  {epoch_title(ep, day)}", (40, int(0.14 * h)), F_, 1.5, fg,
                    3, cv2.LINE_AA)
        lines = ["far contralesional spout",
                 f"aim error vs target, epoch median {-r.epoch_median:+.0f} deg (n = {r.epoch_n} first licks)",
                 ("change from pre-stroke: reference epoch" if ep == "pre"
                  else f"change from pre-stroke: {-(r.epoch_median - r.pre_median):+.0f} deg"),
                 "- = ipsilesional; 3-D first-lick direction at peak extension; four-camera licks with 3-D support",
                 ("2 example licks that contacted the spout, from the better-aimed quarter (rule-selected)"
                  if ep == "pre" else "2 example licks, rule-selected near the epoch median")
                 + (", typical latency" if style == "v3" else "") + "; 10x slow motion"]
        for k, t in enumerate(lines):
            cv2.putText(im, t, (40, int(0.34 * h) + k * 46), F_, 0.85 if k < 3 else 0.62, (235, 235, 235),
                        2 if k < 3 else 1, cv2.LINE_AA)
        vw = H264Main(f_out, w, h)
        for _ in range(int(TITLE_S * LC.OUT_FPS)):
            vw.write(im)
        vw.close()
        return f_out
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


TITLE_S_V4 = 5.0       # v4 cards (Priya 11-24): held ~5 s, only the essentials (selection rules live in the legend)


def title_card_v4(f_out: Path, hw: tuple, mouse: str, ep: str, day, med_contra: float, chg_contra: float) -> Path:
    """v4 title card (Priya 11-24): animal + epoch, spout, epoch-median aim error, change from pre, sign + slow motion;
    no selection-rule / method text. Angles are + = contralesional (the mapping csv's *_pos_contra columns)."""
    import cv2
    h, w = hw
    im = np.full((h, w, 3), 16, np.uint8)
    im[: int(0.22 * h)] = EP_BGR[ep]
    F_ = cv2.FONT_HERSHEY_SIMPLEX
    fg = (0, 0, 0) if ep == "pre" else (255, 255, 255)
    cv2.putText(im, f"Mouse {mouse}  -  {epoch_title(ep, day)}", (40, int(0.14 * h)), F_, 1.5, fg, 3, cv2.LINE_AA)
    lines = ["far contralesional spout",
             f"aim error vs target, epoch median {med_contra:+.0f} deg",
             ("change from pre-stroke: reference epoch" if ep == "pre"
              else f"change from pre-stroke: {chg_contra:+.0f} deg"),
             "- = ipsilesional; 10x slow motion"]
    for k, t in enumerate(lines):
        cv2.putText(im, t, (40, int(0.34 * h) + k * 50), F_, 0.95, (235, 235, 235), 2, cv2.LINE_AA)
    vw = H264Main(f_out, w, h)
    for _ in range(int(TITLE_S_V4 * LC.OUT_FPS)):
        vw.write(im)
    vw.close()
    return f_out


def recard(deliv: Path, animal: str, src_style: str = "v3", style: str = "v4") -> Path:
    """Rebuild one animal's deliverable from the EXISTING ``blocks_<src_style>`` clips with v4 title cards (no clip
    re-render; same trials, same frames). Card numbers come from ``SuppVideo_mapping_<src_style>.csv``."""
    M = pd.read_csv(deliv / f"SuppVideo_mapping_{src_style}.csv")
    M = M[M.animal_lab_id == animal]
    blocks = deliv / f"blocks_{src_style}"
    parts = []
    for ep in EPOCHS:
        r = M[M.epoch == ep].iloc[0]
        v = blocks / r.block_file
        if not v.exists():
            raise SystemExit(f"missing block {v}")
        card = blocks / f"_title_{animal}_{ep}_{style}.mp4"
        parts += [str(title_card_v4(card, _frame_hw(v), str(r.mouse), ep, int(r.day),
                                    float(r.epoch_median_vs_target_deg_pos_contra),
                                    float(r.epoch_change_from_pre_deg_pos_contra))), str(v)]
    f4 = deliv / f"SuppVideo_mouse{mouse_label(animal)}_far_contralesional_{style}.mp4"
    if f4.exists():
        raise SystemExit(f"{f4} exists (new names only)")
    concat(parts, f4)
    for q in parts:
        if Path(q).name.startswith("_title_"):
            Path(q).unlink()
    return f4


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
    ap.add_argument("--style", choices=("v1", "v2", "v3", "v5"), default="v1",
                    help="v5 (Priya 11-24) = v3 clips, PRE examples by the same median rule as post-stroke, 5-s essentials-only "
                         "title cards; v2 = supplementary deliverable (manuscript animal numbers, Fig 3C v26c framing / keypoints); "
                         "per-epoch blocks go to --out/blocks_v2, the combined video + mapping csv to --out")
    ap.add_argument("--licks", choices=("cam4", "fused"), default="cam4",
                    help="lick list for the selection (fused -> outputs suffixed _fused)")
    ap.add_argument("--epochs", nargs="*", default=list(EPOCHS), help="epochs to render (others: the existing clip "
                    "of the same suffix, else the unsuffixed cam4-list clip, is used for the 4-epoch file)")
    ap.add_argument("--win-only", action="store_true", help="only write the _win.mp4 copies of the clips in --out")
    ap.add_argument("--recard", action="store_true", help="v4 (Priya 11-24): rebuild --animal's deliverable in --out "
                    "from the existing blocks_v3 clips with 5-s essentials-only title cards (+ _win copy)")
    a = ap.parse_args(argv)
    if not str(a.out).replace("\\", "/").startswith("M:/MICROSCOPE/Priya/"):
        raise SystemExit("--out must be under M:/MICROSCOPE/Priya/")
    if a.recard:
        f4 = recard(a.out, a.animal)
        print("v4", f4)
        print("win copy", win_copy(f4))
        return 0
    if a.win_only:
        for f in sorted(a.out.glob("supp4cam_*.mp4")):
            if not f.stem.endswith("_win"):
                print("win copy", win_copy(f))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    vx = a.style in ("v2", "v3", "v5")
    sfx = ("" if a.licks == "cam4" else f"_{a.licks}") + (f"_{a.style}" if vx else "")
    deliv = a.out
    if vx:
        a.out = a.out / f"blocks_{a.style}"
        a.out.mkdir(parents=True, exist_ok=True)
    R = select(a.n, a.licks, a.animal, pre_rule="contact_q25" if a.style in ("v2", "v3") else "median",
               lat_iqr=a.style in ("v3", "v5"))
    if a.style in ("v3", "v5"):
        R["rule"] = R["rule"] + "; " + LAT_RULE
    print(R[["epoch", "session", "epoch_median", "epoch_n", "session_median", "session_n"]].to_string())
    LC.PREP_DIR.mkdir(parents=True, exist_ok=True)
    items = [(r, a.n, str(a.out), str(LC.PREP_DIR), sfx, a.style) for r in R.to_dict("records")
             if r["epoch"] in a.epochs]
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
        from wfield_local import epochs
        # THE supplementary deliverable (Priya 11-22): one video per animal, epochs in order, a title card before
        # each epoch, the epoch banner in every frame (the per-epoch clips stay as building blocks)
        hw = _frame_hw(seq[0])
        parts = []
        for ep, v in zip(EPOCHS, seq):
            fc = a.out / f"_title_{a.animal}_{ep}{sfx}.mp4"
            if a.style == "v5":
                r = R[R.epoch == ep].iloc[0]
                day = epochs.days_since_stroke(f"{r.animal}_{r.session.split('_')[1][4:]}")
                card = title_card_v4(fc, hw, mouse_label(a.animal), ep, day, -float(r.epoch_median),
                                     -float(r.epoch_median - r.pre_median))
            else:
                card = title_card(fc, hw, R, ep, a.licks, a.style)
            parts += [str(card), v]
        f4 = (deliv / f"SuppVideo_mouse{mouse_label(a.animal)}_far_contralesional_{a.style}.mp4" if vx
              else a.out / f"supp4cam_{a.animal}_farcontra_allepochs{sfx}.mp4")
        print("all-epochs file from", *seq, sep="\n  ")
        concat(parts, f4)
        for q in parts:
            if Path(q).name.startswith("_title_"):
                Path(q).unlink()
        vids.append(str(f4))
        if vx:
            # provenance (NOT in the video): deliverable -> lab ID, sessions, trials, aim errors
            M = pd.DataFrame(rows).assign(deliverable=f4.name, mouse=mouse_label(a.animal), animal_lab_id=a.animal,
                                          block_file=lambda d: d.file)
            M = M[["deliverable", "mouse", "animal_lab_id", "epoch", "day", "session", "trial", "position",
                   "first_lick_peak_ms", "aim_err_vs_target_deg_pos_contra", "epoch_median_vs_target_deg_pos_contra",
                   "pre_median_vs_target_deg_pos_contra", "epoch_change_from_pre_deg_pos_contra", "epoch_n", "rule",
                   "rank", "cue_to_water_valve_ms", "strobe_ms", "stop_ms", "block_file", "crop_px"]]
            M["angle_definition"] = ("3-D midline angle of the tongue tip at the first lick's peak extension, "
                                     "asin(lateral / |r|) from the spout set point (= atan2(lateral, sagittal-plane "
                                     "extension)), MINUS the target spout's direction; + = contralesional. The on-video "
                                     "number is this raw error vs the target; the title cards also give the epoch "
                                     "median's change from the pre-stroke median (the manuscript's Fig 4D quantity is "
                                     "the change from each animal's pre-stroke MEAN, from the where-decoder aim table)")
            fm = deliv / f"SuppVideo_mapping_{a.style}.csv"
            if fm.exists():
                old = pd.read_csv(fm)
                # keep this animal's rows for epochs NOT rendered in this run (an --epochs re-run must not drop them;
                # bug 11-24: pre-only re-runs left the v5 mapping with the pre rows only)
                keep = ~((old.animal_lab_id == a.animal) & old.epoch.isin(M.epoch.unique()))
                M = pd.concat([old[keep], M], ignore_index=True)
                M["_o"] = M.epoch.map({e: i for i, e in enumerate(EPOCHS)})
                M = M.sort_values(["mouse", "_o", "rank"], kind="mergesort").drop(columns="_o")
            M.to_csv(fm, index=False)
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
