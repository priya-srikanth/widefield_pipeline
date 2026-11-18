"""VIDEO REELS per AR-HMM state, for the human merge gate (Priya 11-17: "make the clips for merge QC"; Hillary
Tong's F11 / F12 workflow: the cosine shortlist proposes, the video decides).

    python -m wfield_local.arhmm_reels --out <arhmm out dir> --feat <features dir> --tag pooled_all_s1 [--jobs 6]

For every LIVE state (usage >= arhmm_states.DEAD_USAGE) it samples N_PER_STATE syllables -- one per session, cycling
over animals then epochs so every animal and stage is represented, each syllable at least half the state's median
dwell and at most MAX_SYL_S long -- and cuts the cam4 video from PAD_S before the syllable to PAD_S after it.
Writes, under ``<LP_analysis>/<run>/arhmm/v1/reels_<tag>/`` (or ``--dest``):
    state_XX/clip_NN_<label>.avi   each syllable (MJPG, 0.24x: every 2nd 250 fps frame at 30 fps), GREEN border while
                                   the syllable runs, grey before / after; caption = state, session, epoch, duration
    state_XX.avi                   the clips back to back (the reel to watch)
    contact_sheet.png              rows = states, columns = the mid-syllable frame of each clip (one-page overview,
                                   merge candidates read side by side)
    reels_manifest.csv             one row per clip (state, label, epoch, start / stop DAQ s, duration)

TIMING follows `behavior_clips` (read its docstring): DAQ seconds -> absolute camera seconds through the cam4
alignment template, then the camera ROW by searching its per-row timestamps (never the row-index affine, which
drifts after dropped frames); sessions whose template is missing / fails ``quality_ok`` are skipped, and a syllable
where the camera was dropping frames (`camera_sync.camera_healthy_at`) is not used. Source videos are opened
read-only (cv2.VideoCapture); the output folder passes `writeguard.assert_writable`.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

CAM = "cam4"
N_PER_STATE = 8
SPARE = 3                        # extra syllables drawn per state: a session without usable video (e.g. PS92_20260812,
                                 # crash + concat) is skipped, the reel keeps the first N_PER_STATE that cut cleanly
PAD_S = 0.5
MAX_SYL_S = 3.0
MIN_BINS = 3
OUT_PX, OUT_FPS, KEEP_EVERY = 360, 30.0, 2
TILE = 180
SEED = 0
STRIP_PX = 90                    # --wheel: treadmill strip under the video
WHEEL_FWD_MAX, WHEEL_BACK_MAX = 15.0, 2.0   # mm/s at the strip's top (forward, up) / bottom (backward, down)


def _recordings(rv, date: str, animal: str, cam: str) -> list:
    """(rec, template, video, timestamps csv, segment or None) for every usable recording of ``cam`` that day; a split /
    concatenated day (PS92 0812) has one template per recording, each valid inside segment_daq_sec."""
    from wfield_local import camera_sync as cs
    vdir = Path(rv.root("behavior_cameras")) / date / animal
    recs = []
    for rec, tpls in cs.templates_for(rv, date, animal, cam).items():
        vid, tsf = vdir / f"{cam}_{rec}.avi", vdir / f"{cam}_{rec}.csv"
        if not (vid.exists() and tsf.exists()):
            continue
        for tpl in tpls:
            if bool(tpl["quality_ok"]):
                seg = tpl.get("segment_daq_sec")
                recs.append((rec, tpl, vid, tsf, None if seg is None else (float(seg[0]), float(seg[1]))))
    return recs


class _SeqReader:
    """Sequential frame reader for a time-matched side camera: seeks once, then reads forward (seeking per frame in an
    MJPG avi is slow). An index already passed returns the last frame read."""

    def __init__(self, cap, f0: int):
        self.cap, self.i, self.last = cap, f0 - 1, None
        cap.set(1, f0)                                              # cv2.CAP_PROP_POS_FRAMES

    def get(self, idx: int):
        while self.i < idx:
            ok, fr = self.cap.read()
            if not ok:
                break
            self.i += 1
            self.last = fr
        return self.last


def _strip(t_rel, fwd, back, cues_rel, now_rel, width: int, t_span: tuple):
    """Treadmill strip: forward speed up (green), backward speed down (red), cue = yellow line, now = white line."""
    import cv2
    img = np.zeros((STRIP_PX, width, 3), np.uint8)
    mid = STRIP_PX // 2
    a, b = t_span

    def x(tt):
        return int(round((tt - a) / max(b - a, 1e-9) * (width - 1)))

    cv2.line(img, (0, mid), (width - 1, mid), (90, 90, 90), 1)
    pf = [(x(tt), int(mid - min(v, WHEEL_FWD_MAX) / WHEEL_FWD_MAX * (mid - 4))) for tt, v in zip(t_rel, fwd)]
    pb = [(x(tt), int(mid + min(v, WHEEL_BACK_MAX) / WHEEL_BACK_MAX * (mid - 4))) for tt, v in zip(t_rel, back)]
    if len(pf) > 1:
        cv2.polylines(img, [np.array(pf, np.int32)], False, (60, 200, 60), 2)
        cv2.polylines(img, [np.array(pb, np.int32)], False, (60, 60, 230), 2)
    for c in cues_rel:
        if a <= c <= b:
            cv2.line(img, (x(c), 0), (x(c), STRIP_PX - 1), (0, 220, 255), 2)
            cv2.putText(img, "cue", (x(c) + 3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.line(img, (x(now_rel), 0), (x(now_rel), STRIP_PX - 1), (255, 255, 255), 1)
    cv2.putText(img, f"fwd 0-{WHEEL_FWD_MAX:g}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 200, 60), 1, cv2.LINE_AA)
    cv2.putText(img, f"back 0-{WHEEL_BACK_MAX:g} mm/s", (4, STRIP_PX - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (60, 60, 230), 1, cv2.LINE_AA)
    return img


def syllables(z: np.ndarray, k: int, min_bins: int, max_bins: int) -> list[tuple[int, int]]:
    """(start, stop) bin runs of state ``k`` with min_bins <= length <= max_bins."""
    m = (z == k).astype(int)
    d = np.diff(np.r_[0, m, 0])
    st, sp = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a, b) for a, b in zip(st, sp) if min_bins <= b - a <= max_bins]


def select(Z: dict, live: list[int], dwell_med_bins: dict, epochs: dict, n: int = N_PER_STATE,
           seed: int = SEED) -> pd.DataFrame:
    """One syllable per session per state, cycling animals then epochs (seeded)."""
    from wfield_local import arhmm_model as am
    rng = np.random.default_rng(seed)
    rows = []
    max_bins = int(MAX_SYL_S * am.FS_HZ)
    for k in live:
        lo = max(MIN_BINS, int(0.5 * dwell_med_bins.get(k, MIN_BINS)))
        pool = {}
        for lab in sorted(Z):
            s = syllables(Z[lab], k, lo, max_bins)
            if s:
                pool.setdefault(lab.split("_")[0], {}).setdefault(epochs.get(lab, "?"), []).append((lab, s))
        order = []
        for an in sorted(pool):
            for ep in ("pre", "acute", "subacute", "chronic", "?"):
                for lab, s in pool[an].get(ep, []):
                    order.append((an, ep, lab, s))
        if not order:
            continue
        # round-robin over animals, then epochs within animal, a random session of that cell, a random syllable of it
        cells = {}
        for an, ep, lab, s in order:
            cells.setdefault(an, {}).setdefault(ep, []).append((lab, s))
        picked, used = [], set()
        ans = sorted(cells)
        ep_idx = {an: 0 for an in ans}
        while len(picked) < n and any(cells[an] for an in ans):
            for an in ans:
                eps = [e for e in ("pre", "acute", "subacute", "chronic", "?") if cells[an].get(e)]
                if not eps or len(picked) >= n:
                    continue
                ep = eps[ep_idx[an] % len(eps)]
                ep_idx[an] += 1
                cand = [(lab, s) for lab, s in cells[an][ep] if lab not in used]
                if not cand:
                    cells[an][ep] = []
                    continue
                lab, s = cand[rng.integers(len(cand))]
                a, b = s[rng.integers(len(s))]
                used.add(lab)
                picked.append({"state": k, "label": lab, "animal": an, "epoch": ep, "start_bin": int(a),
                               "stop_bin": int(b)})
        rows += picked
    return pd.DataFrame(rows)


def _session_job(item):
    """Cut every selected syllable of ONE session (module-level for the process pool). Returns manifest rows.
    The first camera of ``cams`` drives the clock (its frames, every KEEP_EVERY-th); other cameras show the frame
    nearest the same DAQ time through their own templates. ``wheel`` adds the treadmill strip (features run_fwd /
    run_back, 25 Hz) with the DAQ cues."""
    import cv2

    from wfield_local.camera_sync import camera_healthy_at
    from wfield_local.paths import PathResolver
    lab, sel, feat_dir, dest, cams, wheel = item
    rv = PathResolver()
    animal, date = lab.split("_")
    R = {c: _recordings(rv, date, animal, c) for c in cams}
    if not all(R.values()):
        miss = [c for c in cams if not R[c]]
        return [dict(r, status=f"no usable template / video / timestamps ({','.join(miss)})") for r in sel]
    F = np.load(Path(feat_dir) / "sessions" / f"{lab}.npz", allow_pickle=True)
    t = F["t_daq_s"]
    cues = np.array([])
    if wheel:
        cols = [str(c) for c in F["features"]]
        fwd, back = F["X"][:, cols.index("run_fwd")], F["X"][:, cols.index("run_back")]
        from wfield_local import analysis_kit as ak
        b = ak.session_behavior(f"{animal}_{date[4:]}")
        cues = np.sort(b.cue_samples / b.sample_rate) if b is not None else cues
    out = []
    caps, tss = {}, {}

    def opened(cam, rec, vid, tsf):
        if (cam, rec) not in caps:
            caps[(cam, rec)] = cv2.VideoCapture(str(vid))                  # READ-ONLY
            tss[(cam, rec)] = pd.read_csv(tsf, header=None, usecols=[1], dtype="int64").to_numpy().ravel()
        return caps[(cam, rec)], tss[(cam, rec)]

    try:
        for r in sel:
            t0, t1 = float(t[r["start_bin"]]), float(t[min(r["stop_bin"], len(t) - 1)])
            a_daq, b_daq = t0 - PAD_S, t1 + PAD_S
            win, bad = {}, None
            for cam in cams:
                hit = [x for x in R[cam] if x[4] is None or (x[4][0] <= a_daq and b_daq <= x[4][1])]
                if not hit:
                    bad = f"no {cam} recording covers the syllable"
                    break
                rec, tpl, vid, tsf, _seg = hit[0]
                cap, cam_ts = opened(cam, rec, vid, tsf)
                ic, sl = float(tpl["intercept_daqSec"]), float(tpl["slope_daqSec_per_camSec"])
                ca, cb = (a_daq - ic) / sl, (b_daq - ic) / sl
                if not camera_healthy_at(tpl, np.array([ca, cb])).all():
                    bad = f"{cam} dropping frames"
                    break
                f0, f1 = int(np.searchsorted(cam_ts, ca * 1e9)), int(np.searchsorted(cam_ts, cb * 1e9))
                if f0 <= 0 or f1 >= len(cam_ts) or f1 <= f0:
                    bad = f"{cam} window off the recording"
                    break
                win[cam] = (cap, cam_ts, ic, sl, f0, f1)
            if bad:
                out.append(dict(r, status=bad))
                continue
            cap, cam_ts, ic, sl, f0, f1 = win[cams[0]]
            side = {c: (_SeqReader(win[c][0], win[c][4]), win[c]) for c in cams[1:]}
            s0_ns, s1_ns = (t0 - ic) / sl * 1e9, (t1 - ic) / sl * 1e9
            W, H = OUT_PX * len(cams), OUT_PX + (STRIP_PX if wheel else 0)
            if wheel:
                m = (t >= a_daq) & (t <= b_daq)
                t_rel, fw, bk = t[m] - t0, fwd[m], back[m]
            sd = Path(dest) / f"state_{r['state']:02d}"
            sd.mkdir(parents=True, exist_ok=True)
            fn = sd / f"clip_{r['clip']:02d}_{lab}.avi"
            vw = cv2.VideoWriter(str(fn), cv2.VideoWriter_fourcc(*"MJPG"), OUT_FPS, (W, H))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            mid = None
            for i, fr_idx in enumerate(range(f0, f1)):
                ok, fr = cap.read()
                if not ok:
                    break
                if i % KEEP_EVERY:
                    continue
                inside = s0_ns <= cam_ts[fr_idx] <= s1_ns
                d_now = ic + sl * cam_ts[fr_idx] / 1e9                    # DAQ seconds of this frame
                panels = [cv2.resize(fr, (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)]
                for c, (rd, (_cp, ts2, ic2, sl2, _a, _b)) in side.items():
                    j = int(np.clip(np.searchsorted(ts2, (d_now - ic2) / sl2 * 1e9), 0, len(ts2) - 1))
                    f2 = rd.get(j)
                    p2 = (cv2.resize(f2, (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA) if f2 is not None
                          else np.zeros((OUT_PX, OUT_PX, 3), np.uint8))
                    cv2.putText(p2, c, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    panels.append(p2)
                img = np.hstack(panels)
                col = (60, 200, 60) if inside else (140, 140, 140)
                cv2.rectangle(img, (0, 0), (W - 1, OUT_PX - 1), col, 6)
                cv2.putText(img, f"state {r['state']}  {animal} {date[4:]} {r['epoch']}", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"syllable {t1 - t0:.2f} s" + ("  <<" if inside else ""), (10, OUT_PX - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if wheel:
                    img = np.vstack([img, _strip(t_rel, fw, bk, cues - t0, d_now - t0, W,
                                                 (-PAD_S, t1 - t0 + PAD_S))])
                vw.write(img)
                if inside and mid is None and cam_ts[fr_idx] >= (s0_ns + s1_ns) / 2:
                    mid = img
            vw.release()
            if mid is not None:
                cv2.imwrite(str(fn.with_suffix(".png")), mid)
            out.append(dict(r, status="ok", start_daq_s=t0, stop_daq_s=t1, dur_s=t1 - t0, file=str(fn)))
    finally:
        for c in caps.values():
            c.release()
    return out


def stitch(dest: Path, manifest: pd.DataFrame, n: int = N_PER_STATE) -> None:
    """state_XX.avi (clips back to back) + contact_sheet.png (rows = states)."""
    import cv2
    ok = manifest[manifest.status == "ok"].sort_values(["state", "clip"])
    rows = []
    for k, g in ok.groupby("state"):
        g = g.head(n)
        c0 = cv2.VideoCapture(g.file.iloc[0])
        size = (int(c0.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c0.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        c0.release()
        vw = cv2.VideoWriter(str(dest / f"state_{k:02d}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), OUT_FPS, size)
        tiles = []
        for f in g.file:
            cap = cv2.VideoCapture(f)
            while True:
                okr, fr = cap.read()
                if not okr:
                    break
                vw.write(fr)
            cap.release()
            png = Path(f).with_suffix(".png")
            if png.exists():
                im = cv2.imread(str(png))
                tiles.append(cv2.resize(im, (int(TILE * im.shape[1] / im.shape[0]), TILE)))
        vw.release()
        tw = tiles[0].shape[1] if tiles else TILE
        tiles += [np.zeros((TILE, tw, 3), np.uint8)] * (n - len(tiles))
        row = np.hstack(tiles[:n])
        lab = np.zeros((TILE, 70, 3), np.uint8)
        cv2.putText(lab, f"{k}", (12, TILE // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.hstack([lab, row]))
    if rows:
        cv2.imwrite(str(dest / "contact_sheet.png"), np.vstack(rows))


def reels_dir(tag: str) -> Path:
    """The default reel folder of a fitted model (``LP_analysis/<run>/arhmm/v1/reels_<tag>``); its ``state_labels.csv``
    holds the reviewed state labels and GROUPS that `behavior_states.state_groups` reads."""
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    return Path(oo.root(PathResolver())) / oc.RUN / "arhmm" / "v1" / f"reels_{tag}"


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import arhmm_states as st
    from wfield_local import epochs as ep
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    ap.add_argument("--out", required=True, help="AR-HMM output dir (model_<tag>.pkl, states_<tag>/)")
    ap.add_argument("--feat", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--n", type=int, default=N_PER_STATE)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--states", default=None, help="comma list: reel only these (may include states below DEAD_USAGE)")
    ap.add_argument("--cams", default=CAM, help="comma list; the first drives the clock (e.g. cam4,cam1)")
    ap.add_argument("--wheel", action="store_true", help="treadmill strip (forward / backward) + cue under each clip")
    a = ap.parse_args(argv)
    M, feats, Z = st.load(Path(a.out), a.tag, Path(a.feat))
    K = int(np.asarray(M["params"]["Ab"]).shape[0])
    u = st.usage(Z, K)
    live = [k for k in range(K) if u[k] >= st.DEAD_USAGE]
    dw = st.dwell_s(Z)
    from wfield_local import arhmm_model as am
    dmed = {int(k): float(v) * am.FS_HZ for k, v in dw["median"].items()}
    epochs = {lab: ep.epoch_of(lab) for lab in Z}
    want = [int(x) for x in a.states.split(",")] if a.states else live
    # extra states go AFTER the live ones, so the live states draw exactly the syllables of the default reel
    S = select(Z, live + [k for k in want if k not in live], dmed, epochs, a.n + SPARE)
    S = S[S.state.isin(want)].copy()
    S["clip"] = S.groupby("state").cumcount()
    dest = Path(a.dest) if a.dest else reels_dir(a.tag)
    assert_writable(dest)
    dest.mkdir(parents=True, exist_ok=True)
    cams = a.cams.split(",")
    items = [(lab, g.to_dict("records"), a.feat, str(dest), cams, a.wheel) for lab, g in S.groupby("label")]
    res, fails = ak.fan_sessions(items, _session_job, jobs=a.jobs, key=lambda it: it[0])
    for f in fails:
        print("  failed:", f)
    man = pd.DataFrame([r for _it, rows in res for r in rows])
    man.to_csv(dest / "reels_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    stitch(dest, man, a.n)
    print(man.status.value_counts().to_string())
    print(f"{S.state.nunique()} states -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
