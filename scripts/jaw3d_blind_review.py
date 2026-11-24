"""BLIND check of the four-camera 3-D jaw call at close_center (Priya 11-22: blind-check the new 4-cam jaw calls at the
close center spout before any jaw-only counts are quoted).

Why: in the adopted whole-session run (`clean_v37_sync60`), `trial_classes.session_trials` judges the jaw at
close_center (neither cam4 nor cam1 sees it) from the 3-D jaw call of `jaw_onsets.csv` (``jaw3d_source`` == "4cam",
onset floor 0.10 mm, PROVISIONAL). Against the v36 class snapshot, 763 close_center trials moved no_response ->
jaw_only. This script samples them, renders RAW-video review sheets without the call, and scores the blind review.

    conda activate dlc      # MKL_THREADING_LAYER=SEQUENTIAL, PYTHONPATH=<worktree>
    python -m scripts.jaw3d_blind_review sample  --out <folder>          # key.csv (DO NOT OPEN before scoring)
    python -m scripts.jaw3d_blind_review render  --out <folder> --jobs 2 # sheets/, strips/ (pass 1)
    python -m scripts.jaw3d_blind_review render  --out <folder> --zoom   # strips_zoom/ (pass 2: native chin + diff)
    python -m scripts.jaw3d_blind_review unblind --out <folder>          # key + scores.csv + scores_zoom.csv -> summary

Result 11-22 (pass 2, primary; README.md in the output folder): A 23 yes / 12 unclear / 5 no, B 7 / 16 / 17, C 11 / 8 / 1;
the 5 clear false positives all have 3-D displacement <= 0.22 mm. Pass 1 (0.5x crops) was too insensitive (C 3 / 20 yes).

GROUPS (seeded `SEED`, stratified by animal x epoch, round-robin over the strata):
  A  close_center, new class jaw_only, v36 class no_response (the 3-D call made it jaw_only)       ``--n-a`` (40)
  B  close_center, new class no_response, 3-D call "not moved" (jaw3d_moved == 0; controls)        ``--n-b`` (40)
  C  close_L / close_R / far_L / far_R, new class jaw_only with a per-camera call (cam4 or cam1
     moved; the classifier blind-reviewed 10-21): positive reference                             ``--n-c`` (20)
Classes: the s1 snapshots ``cls`` (label, cue_idx), matched to the video trial_id by cue DAQ time (the LP run's
windows: median ``daq_s - t_ms``; nearest within 50 ms). Sessions whose cameras have more than one video (split days)
or whose side bundle is not done are skipped (stated in the README).

SHEET (one per trial, anonymous id; no position, animal, date or call shown): frames every 40 ms from cue - 200 ms to
the response-window end (min(cue + 3.5 s, trial stop)), in chunks of 16 frames (640 ms); rows cam2 (animal's left),
cam3 (animal's right), cam4 (front), cam1 (below). Frame choice: the production windows' DAQ time of each frame
(`o2_inference.session_windows`, alignment templates, sync span) -> the frame nearest each target time; cam4 / cam1
frames = the 3-D frames table's paired ``src4`` / ``src1`` (nearest ``t_daq_ms``). Crops: side views 1:1 raw px around
the trial's pre-cue median side jaw point (side model, likelihood >= 0.6), cam4 around its pre-cue jaw point (LP run
windows), cam1 a fixed region below the snout. ``strips/<id>.png`` = cam2 / cam3 / cam4 only (what is scored);
``sheets/<id>.png`` = all four views + the 3-D jaw displacement trace (dv, ap, ml from the trial's pre-cue median; no
threshold, no onset marker).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261122
LP = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis")
RUN_NEW = "lp_r4_heatmap_20261013__clean_v37_sync60"
RUN_OLD = "lp_r4_heatmap_20261013__clean_v36"
P3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60")
SIDE_BUNDLE = "lp_side_r2b_prod_20261118"
ENV = {"WIDEFIELD_WINDOWS_SPAN": "sync", "WIDEFIELD_RUN_LABEL": "60"}   # the sync60 production windows
STEP_MS, PRE_MS, CHUNK = 40.0, 200.0, 15
SIDE_CROP = (200, 160)        # raw px (w, h) around the side jaw point, scaled to TILE
CAM4_CROP = (300, 240)
CAM1_CROP, CAM1_CENTER = (260, 208), (300, 390)
TILE = (100, 80)
ZOOM_SIDE, ZOOM_CAM4, DIFF_GAIN = (100, 80), (150, 120), 2.5     # pass 2: side crops 1:1 raw px
JAW_CUT = 0.6
OTHER = ("close_L", "close_R", "far_L", "far_R")


# ------------------------------------------------------------------------------------------------ sampling

def _label_parts(label: str) -> tuple[str, str]:
    a, d = label.split("_")
    return a, d if len(d) == 8 else f"2026{d}"


def _trial_cues(animal: str, date: str) -> pd.DataFrame:
    from wfield_local import trial_classes as tc
    return tc._trial_times(LP / RUN_NEW / "sessions" / animal / date)


def _usable(rv, animal: str, date: str) -> str:
    from wfield_local import o2_inference as oi
    if not (P3D / f"{animal}_{date}_frames.csv.gz").exists():
        return "no 3-D frames"
    for c in ("cam1", "cam2", "cam3", "cam4"):
        if len(oi.videos_for(rv, animal, date, c)) != 1:
            return f"{c}: not one video"
    od = Path(rv.root("dlc")) / "o2" / SIDE_BUNDLE / "out" / f"{animal}_{date}"
    for c in ("cam2", "cam3"):
        if not any(od.glob(f"{c}_*_done.json")):
            return f"{c}: side bundle not done"
    return ""


def sample(out: Path, n_a: int, n_b: int, n_c: int) -> None:
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    q = "imaging_encoding/qc/s1_trial_classes_cue_stop.csv.gz"
    new = pd.read_csv(LP / RUN_NEW / q)
    old = pd.read_csv(LP / RUN_OLD / q, usecols=["label", "cue_idx", "cls"]).rename(columns={"cls": "cls_old"})
    M = new[["label", "cue_idx", "cue_s", "cls", "position"]].merge(old, on=["label", "cue_idx"], how="left")
    J = pd.read_csv(LP / RUN_NEW / "imaging_encoding/qc/jaw_onsets.csv", dtype={"date": str})
    A = M[(M.position == "close_center") & (M.cls == "jaw_only") & (M.cls_old == "no_response")]
    B = M[(M.position == "close_center") & (M.cls == "no_response")]
    C = M[M.position.isin(OTHER) & (M.cls == "jaw_only")]
    print(f"group A pool (no_response -> jaw_only at close_center): {len(A)}")
    rows, skipped = [], {}
    for lab in sorted(set(A.label) | set(B.label) | set(C.label)):
        animal, date = _label_parts(lab)
        why = _usable(rv, animal, date)
        if why:
            skipped[lab] = why
            continue
        tt = _trial_cues(animal, date)
        cue = tt.cue_daq.to_numpy(float)
        for g, D in (("A", A), ("B", B), ("C", C)):
            d = D[D.label == lab]
            for r in d.itertuples():
                k = int(np.nanargmin(np.abs(cue - r.cue_s)))
                if abs(cue[k] - r.cue_s) > 0.05:
                    continue
                rows.append(dict(group=g, label=lab, animal=animal, date=date, cue_idx=int(r.cue_idx),
                                 trial_id=int(tt.trial_id.iloc[k]), position=r.position, cls_new=r.cls,
                                 cls_old=r.cls_old, stop_ms=float(tt.stop_ms.iloc[k])))
    P = pd.DataFrame(rows)
    J["session"] = J.session.astype(str)
    P["session"] = P.animal + "_" + P.date
    P = P.merge(J.drop(columns=["animal", "date", "position"]), on=["session", "trial_id"], how="left")
    ok = (((P.group == "A") & (P.jaw3d_moved == 1) & (P.jaw3d_source == "4cam"))
          | ((P.group == "B") & (P.jaw3d_moved == 0) & (P.jaw3d_source == "4cam"))
          | ((P.group == "C") & ((P.cam4_moved == 1) | (P.cam1_moved == 1))))
    P = P[ok & P.epoch.notna()].reset_index(drop=True)
    print(P.groupby(["group", "animal", "epoch"]).size().unstack(fill_value=0).to_string())
    rng = np.random.default_rng(SEED)
    picks = []
    for g, n in (("A", n_a), ("B", n_b), ("C", n_c)):
        strata = {k: list(rng.permutation(v.index.to_numpy()))
                  for k, v in sorted(P[P.group == g].groupby(["animal", "epoch"]), key=lambda kv: kv[0])}
        keys = sorted(strata)
        got = []
        while len(got) < n and any(strata[k] for k in keys):
            for k in keys:
                if strata[k] and len(got) < n:
                    got.append(strata[k].pop(0))
        picks += got
    K = P.loc[picks].copy()
    K = K.iloc[rng.permutation(len(K))].reset_index(drop=True)
    K.insert(0, "id", [f"J{i + 1:03d}" for i in range(len(K))])
    out.mkdir(parents=True, exist_ok=False)
    keep = ["id", "group", "label", "session", "animal", "date", "epoch", "cue_idx", "trial_id", "position", "stop_ms",
            "cls_new", "cls_old", "jaw3d_moved", "jaw3d_onset_ms", "jaw3d_at_cue", "jaw3d_disp_mm", "jaw3d_rest_mm",
            "jaw3d_down_mm", "jaw3d_up_mm", "jaw3d_back_mm", "cam4_moved", "cam1_moved", "jaw_onset_ms",
            "jaw_onset_cam"]
    K[keep].to_csv(out / "key.csv", index=False)
    pd.Series(skipped, name="reason").rename_axis("label").to_csv(out / "skipped_sessions.csv")
    print(f"wrote {len(K)} trials ({K.group.value_counts().to_dict()}) -> {out / 'key.csv'}; "
          f"skipped {len(skipped)} sessions")


# ------------------------------------------------------------------------------------------------ six-spout sample (11-23)

VALID = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/ccB_caption_review_2026-11-23/"
             "jaw3d_qc/jaw3d_vs_2d_trials.csv.gz")
SEED6 = 20261123


def sample6(out: Path, n_per: int, n_anchor: int) -> None:
    """CC-B 11-23 (audit violation 1): DISAGREEMENT trials between the per-camera 2-D jaw call (production classes,
    jaw="camera") and the four-camera 3-D call at all six spouts (jaw="3d_all"), from
    `scripts/jaw3d_all_validate.py build`. Jaw-decided trials only (no contact, no lick / tongue). Groups:
      E  2-D jaw_only, 3-D no_response ("2-D only")          ``n_per`` per spout (5 spouts with a 2-D call)
      F  2-D no_response, 3-D jaw_only ("3-D only")          ``n_per`` per spout
      G  both jaw_only (anchor)  /  H  both no_response (anchor)   ``n_anchor`` each, any 2-D spout
    Stratified round-robin over animal x epoch inside each spout (seed `SEED6`); the ids are shuffled across groups."""
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    V = pd.read_csv(VALID, low_memory=False)
    V = V[V.epoch.isin(["pre", "acute", "subacute", "chronic"]) & ~V.frame_gated.astype(bool)
          & (V.position != "close_center") & (V.cls_detail_2d != "tongue_at_lips")]
    grp = {"E": (V.cls_2d == "jaw_only") & (V.cls_3d == "no_response"),
           "F": (V.cls_2d == "no_response") & (V.cls_3d == "jaw_only"),
           "G": (V.cls_2d == "jaw_only") & (V.cls_3d == "jaw_only"),
           "H": (V.cls_2d == "no_response") & (V.cls_3d == "no_response")}
    V["group"] = np.select(list(grp.values()), list(grp.keys()), "")
    V = V[V.group != ""].copy()
    V["session"] = V.label.map(lambda x: "_".join(_label_parts(x)))
    usable = {s: _usable(rv, *s.split("_")) for s in sorted(V.session.unique())}
    skipped = {s: w for s, w in usable.items() if w}
    V = V[V.session.map(lambda s: not usable[s])].reset_index(drop=True)
    rng = np.random.default_rng(SEED6)

    def pick(D: pd.DataFrame, n: int) -> list:
        strata = {k: list(rng.permutation(v.index.to_numpy()))
                  for k, v in sorted(D.groupby(["animal", "epoch"]), key=lambda kv: kv[0])}
        keys, got = sorted(strata), []
        while len(got) < n and any(strata[k] for k in keys):
            for k in keys:
                if strata[k] and len(got) < n:
                    got.append(strata[k].pop(0))
        return got
    picks = []
    for g in ("E", "F"):
        for pos in ("close_L", "close_R", "far_L", "far_center", "far_R"):
            picks += pick(V[(V.group == g) & (V.position == pos)], n_per)
    for g in ("G", "H"):
        picks += pick(V[V.group == g], n_anchor)
    K = V.loc[picks].copy()
    stops = {}
    for s in sorted(K.session.unique()):
        a, d = s.split("_")
        stops[s] = dict(zip(*_trial_cues(a, d)[["trial_id", "stop_ms"]].to_numpy().T))
    K["stop_ms"] = [float(stops[s].get(t, np.nan)) for s, t in zip(K.session, K.trial_id)]
    K = K.iloc[rng.permutation(len(K))].reset_index(drop=True)
    K.insert(0, "id", [f"Q{i + 1:03d}" for i in range(len(K))])
    K["date"] = K.session.str.split("_").str[1]
    out.mkdir(parents=True, exist_ok=False)
    keep = ["id", "group", "label", "session", "animal", "date", "epoch", "trial_id", "position", "pos", "stop_ms",
            "cls_2d", "cls_3d", "cam4_moved", "cam1_moved", "cam4_open_px", "cam1_open_px", "jaw3d_moved",
            "jaw3d_onset_ms", "jaw3d_disp_mm", "jaw3d_rest_mm", "jaw_onset_ms_2d"]
    K[keep].to_csv(out / "key.csv", index=False)
    pd.Series(skipped, name="reason").rename_axis("session").to_csv(out / "skipped_sessions.csv")
    pd.DataFrame({"id": sorted(K.id), "verdict": "", "confidence": "", "onset_ms": "", "note": ""}).to_csv(
        out / "scores_TO_FILL.csv", index=False)
    print(K.groupby(["group", "pos"]).size().to_string())
    print(f"wrote {len(K)} trials -> {out / 'key.csv'}; skipped {len(skipped)} sessions")


# ------------------------------------------------------------------------------------------------ rendering

def _frames_at(t_daq: np.ndarray, src: np.ndarray, targets: np.ndarray, tol_ms: float = 8.0) -> np.ndarray:
    o = np.argsort(t_daq)
    t, s = t_daq[o], src[o]
    k = np.clip(np.searchsorted(t, targets), 1, len(t) - 1)
    k = np.where(np.abs(t[k - 1] - targets) < np.abs(t[k] - targets), k - 1, k)
    return np.where(np.abs(t[k] - targets) <= tol_ms, s[k], -1).astype(int)


def _read(video: str, idx: np.ndarray) -> dict:
    import cv2
    want = sorted(set(int(i) for i in idx if i >= 0))
    got = {}
    if not want:
        return got
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, want[0])
    cur = want[0]
    for w in want:
        if w - cur > 300:
            cap.set(cv2.CAP_PROP_POS_FRAMES, w)
            cur = w
        while cur <= w:
            ok, im = cap.read()
            if not ok:
                break
            if cur == w:
                got[w] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            cur += 1
    cap.release()
    return got


def _crop(im, center, size, tile=TILE):
    import cv2
    if im is None:
        return np.full((tile[1], tile[0]), 40, np.uint8)
    w, h = size
    x0 = int(round(center[0] - w / 2))
    y0 = int(round(center[1] - h / 2))
    H, Wd = im.shape
    pad = np.zeros((h, w), np.uint8)
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(Wd, x0 + w), min(H, y0 + h)
    if xe > xs and ye > ys:
        pad[ys - y0:ye - y0, xs - x0:xe - x0] = im[ys:ye, xs:xe]
    return cv2.resize(pad, tile, interpolation=cv2.INTER_AREA)


def _jaw_center(flat: pd.DataFrame, rows: np.ndarray, rows_all: np.ndarray) -> tuple[float, float]:
    for r in (rows, rows_all):
        x, y = flat.jaw_x.to_numpy(float)[r], flat.jaw_y.to_numpy(float)[r]
        ok = flat.jaw_likelihood.to_numpy(float)[r] >= JAW_CUT
        if ok.sum() >= 5:
            return float(np.median(x[ok])), float(np.median(y[ok]))
    return np.nan, np.nan


def render_session(item) -> list[dict]:
    """Module-level worker: (session, key rows as records, out folder) -> one status dict per trial."""
    session, recs, out = item[:3]
    zoom = bool(item[3]) if len(item) > 3 else False
    os.environ.update(ENV)
    import cv2
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = session.split("_")
    S = pd.read_csv(P3D / f"{session}_frames.csv.gz",
                    usecols=["trial_id", "t_ms", "t_daq_ms", "src1", "src4", "jaw_ml", "jaw_ap", "jaw_dv"])
    w4 = np.load(LP / RUN_NEW / "sessions" / animal / date / "windows.npz", allow_pickle=True)
    parts4 = [str(p) for p in w4["parts"]]
    j4 = parts4.index("jaw")
    W4 = pd.DataFrame({"trial_id": w4["ix_trial_id"], "t_ms": w4["ix_t_ms"], "src": w4["ix_src_frame"],
                       "jaw_x": w4["pose"][:, j4, 0], "jaw_y": w4["pose"][:, j4, 1],
                       "jaw_likelihood": w4["pose"][:, j4, 2]})
    side = {}
    for c in ("cam2", "cam3"):
        got = oi.session_windows(rv, SIDE_BUNDLE, animal, date, c)
        if got is None:
            return [dict(id=r["id"], status=f"{c} windows missing") for r in recs]
        ix, flat = got
        from wfield_local import lp_sidecam as LS
        from wfield_local import pose3d_side as PS
        wd = PS.side_width(rv, animal, date, c)
        flat = flat.copy()
        flat["jaw_x"], flat["jaw_y"] = LS.to_raw(flat.jaw_x.to_numpy(float), flat.jaw_y.to_numpy(float), c, wd)
        side[c] = (ix, flat)
    vids = {c: str(oi.videos_for(rv, animal, date, c)[0]) for c in ("cam1", "cam2", "cam3", "cam4")}
    status = []
    for r in recs:
        tid = int(r["trial_id"])
        end = float(np.fmin(3500.0, r["stop_ms"])) if np.isfinite(r["stop_ms"]) else 3500.0
        tg = np.arange(-PRE_MS, end + 1e-6, STEP_MS)
        st = S[S.trial_id == tid]
        fr = {"cam4": _frames_at(st.t_daq_ms.to_numpy(float), st.src4.to_numpy(float), tg),
              "cam1": _frames_at(st.t_daq_ms.to_numpy(float), st.src1.to_numpy(float), tg)}
        center = {"cam1": CAM1_CENTER}
        g4 = W4[W4.trial_id == tid]
        r4 = g4.index.to_numpy()
        center["cam4"] = _jaw_center(W4, r4[(g4.t_ms.to_numpy() < 0) & (g4.t_ms.to_numpy() >= -500)], r4)
        for c, (ix, flat) in side.items():
            m = (ix.trial_id == tid).to_numpy()
            rows = np.flatnonzero(m)
            fr[c] = _frames_at(ix.t_daq_ms.to_numpy(float)[rows], ix.src_frame.to_numpy(float)[rows], tg)
            t = ix.t_ms.to_numpy(float)[rows]
            center[c] = _jaw_center(flat, rows[(t < 0) & (t >= -500)], rows)
        # side crops: centre a little above / forward of the jaw point so the mouth and lower jaw both show
        for c in ("cam2", "cam3"):
            if np.isfinite(center[c][0]):
                center[c] = (center[c][0] + (-30 if c == "cam2" else 30), center[c][1] - 20)
        if not np.isfinite(center["cam4"][0]):
            center["cam4"] = (340, 400)
        missing = {c: int((f < 0).sum()) for c, f in fr.items()}
        if zoom:
            fr.pop("cam1")
            ims = {c: _read(vids[c], fr[c]) for c in fr}
            (out / "strips_zoom").mkdir(exist_ok=True)
            cv2.imwrite(str(out / "strips_zoom" / f"{r['id']}.png"), _zoom_strip(r["id"], tg, fr, ims, center))
            status.append(dict(id=r["id"], status="ok", n_frames=len(tg), end_ms=end,
                               **{f"missing_{c}": v for c, v in missing.items()}))
            continue
        ims = {c: _read(vids[c], fr[c]) for c in fr}
        size = {"cam2": SIDE_CROP, "cam3": SIDE_CROP, "cam4": CAM4_CROP, "cam1": CAM1_CROP}
        tiles = {c: [_crop(ims[c].get(int(f)), center[c], size[c]) if np.isfinite(center[c][0]) and f >= 0
                     else None for f in fr[c]] for c in fr}
        for name, cams in (("strips", ("cam2", "cam3", "cam4")), ("sheets", ("cam2", "cam3", "cam4", "cam1"))):
            img = _layout(r["id"], tg, tiles, cams)
            if name == "sheets":
                img = np.vstack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), _trace(st, tg, img.shape[1])])
            (out / name).mkdir(exist_ok=True)
            cv2.imwrite(str(out / name / f"{r['id']}.png"), img)
        status.append(dict(id=r["id"], status="ok", n_frames=len(tg), end_ms=end,
                           **{f"missing_{c}": v for c, v in missing.items()},
                           **{f"center_{c}": f"{center[c][0]:.0f},{center[c][1]:.0f}" for c in center}))
    return status


def _zoom_strip(sid: str, tg: np.ndarray, fr: dict, ims: dict, center: dict) -> np.ndarray:
    """PASS 2 (zoom): side views at NATIVE resolution on the chin (``ZOOM_SIDE`` raw px around the side jaw point,
    shifted 10 px forward / up; tile = 1:1), each followed by a DIFFERENCE row = |frame - median of the pre-cue frames|
    x ``DIFF_GAIN`` (both 5 x 5 Gaussian-blurred first, to damp whiskers) (raw video only, no tracking or call drawn); cam4 ``ZOOM_CAM4`` around its jaw point."""
    tiles = {}
    for c in ("cam2", "cam3"):
        cx, cy = center[c]                       # pass-1 centre = jaw + (-/+30, -20): undo, then (-/+10, -10)
        cc = (cx + (20 if c == "cam2" else -20), cy + 10)
        crops = [(_crop(ims[c].get(int(f)), cc, ZOOM_SIDE) if np.isfinite(cx) and f >= 0 else None) for f in fr[c]]
        tiles[c] = crops
        import cv2
        sm = [cv2.GaussianBlur(x, (5, 5), 0).astype(float) if x is not None else None for x in crops]  # damp whiskers
        pre = [x for x, t in zip(sm, tg) if x is not None and t < 0]
        ref = np.median(np.stack(pre), axis=0) if pre else None
        tiles[f"{c} diff"] = [(np.clip(np.abs(x - ref) * DIFF_GAIN, 0, 255).astype(np.uint8)
                               if x is not None and ref is not None else None) for x in sm]
    tiles["cam4"] = [_crop(ims["cam4"].get(int(f)), center["cam4"], ZOOM_CAM4) if f >= 0 else None
                     for f in fr["cam4"]]
    return _layout(sid, tg, tiles, ("cam2", "cam2 diff", "cam3", "cam3 diff", "cam4"))


def _layout(sid: str, tg: np.ndarray, tiles: dict, cams: tuple) -> np.ndarray:
    import cv2
    tw, th = TILE
    lab_w, head, gap = 46, 26, 8
    n_chunks = int(np.ceil(len(tg) / CHUNK))
    W = lab_w + CHUNK * (tw + 2)
    blocks = []
    hdr = np.full((head, W), 255, np.uint8)
    cv2.putText(hdr, f"{sid}   frames every {STEP_MS:.0f} ms from cue - {PRE_MS:.0f} ms to response-window end",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
    blocks.append(hdr)
    for k in range(n_chunks):
        sl = slice(k * CHUNK, (k + 1) * CHUNK)
        tl = np.full((14, W), 255, np.uint8)
        for j, t in enumerate(tg[sl]):
            cv2.putText(tl, f"{t:+.0f}", (lab_w + j * (tw + 2) + 4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        0 if t < 0 else 0, 1, cv2.LINE_AA)
        blocks.append(tl)
        for c in cams:
            row = np.full((th + 2, W), 255, np.uint8)
            for li, word in enumerate(c.split()):
                cv2.putText(row, word, (4, th // 2 + 4 + 14 * li), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA)
            for j, im in enumerate(tiles[c][sl]):
                x = lab_w + j * (tw + 2)
                row[1:1 + th, x:x + tw] = im if im is not None else 40
                if tg[sl][j] < 0:                     # pre-cue frames: thin grey bar above the tile
                    row[0, x:x + tw] = 150
            blocks.append(row)
        blocks.append(np.full((gap, W), 255, np.uint8))
    return np.vstack(blocks)


def _trace(st: pd.DataFrame, tg: np.ndarray, width: int) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = st.t_daq_ms.to_numpy(float)
    pre = (t >= -500) & (t < 0)
    m = (t >= tg[0]) & (t <= tg[-1])
    fig, ax = plt.subplots(figsize=(width / 100, 1.9), dpi=100)
    for p, col in (("dv", "#1f77b4"), ("ap", "#d62728"), ("ml", "#7f7f7f")):
        y = st[f"jaw_{p}"].to_numpy(float)
        ax.plot(t[m], y[m] - np.nanmedian(y[pre]), color=col, lw=1, label=f"jaw {p}")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(tg[0], tg[-1])
    ax.set_xlabel("ms from cue (DAQ)")
    ax.set_ylabel("mm vs pre-cue")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=3)
    fig.tight_layout()
    fig.canvas.draw()
    im = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, 2::-1])   # RGBA -> BGR
    plt.close(fig)
    import cv2
    return cv2.resize(im, (width, im.shape[0]))


def render(out: Path, jobs: int, zoom: bool = False) -> None:
    from wfield_local import parallel
    K = pd.read_csv(out / "key.csv", dtype={"date": str})
    items = [(s, g[["id", "trial_id", "stop_ms"]].to_dict("records"), out, zoom)
             for s, g in K.groupby("session", sort=True)]
    res, fails = parallel.fan_out(items, render_session, jobs=jobs, label="session")
    res = sorted(res, key=lambda x: x[0][0])            # completion order -> session order (repo rule 6 / 9)
    fails = sorted(fails, key=lambda x: x[0][0])
    flat = [d for _it, r in res for d in r]
    for it, msg in fails:
        flat += [dict(id=r["id"], status=f"failed: {msg}") for r in it[1]]
    pd.DataFrame(flat).sort_values("id").to_csv(out / ("render_status_zoom.csv" if zoom else "render_status.csv"),
                                                index=False)
    print(f"rendered {sum(d.get('status') == 'ok' for d in flat)} / {len(K)} sheets")


# ------------------------------------------------------------------------------------------------ unblinding

def unblind(out: Path) -> None:
    """key.csv + scores.csv (pass 1) + scores_zoom.csv (pass 2, primary) -> unblinded.csv and the summary tables.
    ``onset_match``: the reviewer's onset within 250 ms of ``jaw3d_onset_ms`` (yes / unclear rows)."""
    K = pd.read_csv(out / "key.csv", dtype={"date": str})
    p1 = pd.read_csv(out / "scores.csv")[["id", "score"]].rename(columns={"score": "score_p1"})
    z = pd.read_csv(out / "scores_zoom.csv").rename(columns={"score": "score_zoom", "onset_ms": "rev_onset_ms",
                                                              "kind": "rev_kind"})
    D = K.merge(p1, on="id").merge(z[["id", "score_zoom", "confidence", "rev_onset_ms", "rev_kind", "note"]], on="id")
    ok = (D.score_zoom != "no") & D.jaw3d_onset_ms.notna() & D.rev_onset_ms.notna()
    D["onset_match"] = np.where(ok, (D.rev_onset_ms - D.jaw3d_onset_ms).abs() <= 250, np.nan)
    D.to_csv(out / "unblinded.csv", index=False)
    for col in ("score_zoom", "score_p1"):
        print(pd.crosstab(D.group, D[col], margins=True).to_string())
        print(pd.crosstab([D.group, D.animal], D[col]).to_string())
    a = D[D.group == "A"]
    print(pd.crosstab(pd.cut(a.jaw3d_disp_mm, [0, 0.2, 0.3, 0.5, 5]), a.score_zoom).to_string())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("sample", "sample6", "render", "unblind"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--zoom", action="store_true", help="render pass 2 (strips_zoom/: native-resolution chin + diff)")
    ap.add_argument("--n-a", type=int, default=40)
    ap.add_argument("--n-b", type=int, default=40)
    ap.add_argument("--n-c", type=int, default=20)
    ap.add_argument("--n-per", type=int, default=4, help="sample6: trials per spout and direction")
    ap.add_argument("--n-anchor", type=int, default=5, help="sample6: agreement anchors per kind")
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        sample(a.out, a.n_a, a.n_b, a.n_c)
    elif a.cmd == "sample6":
        sample6(a.out, a.n_per, a.n_anchor)
    elif a.cmd == "render":
        render(a.out, a.jobs, a.zoom)
    else:
        unblind(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
