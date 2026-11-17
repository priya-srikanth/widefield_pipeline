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
    """Cut every selected syllable of ONE session (module-level for the process pool). Returns manifest rows."""
    import cv2

    from wfield_local import camera_sync as cs
    from wfield_local.camera_sync import camera_healthy_at
    from wfield_local.paths import PathResolver
    lab, sel, feat_dir, dest = item
    rv = PathResolver()
    animal, date = lab.split("_")
    vdir = Path(rv.root("behavior_cameras")) / date / animal
    # every recording of the day with its template(s): an ordinary day has one; a split / concatenated day (PS92 0812:
    # two cam4 files after the acquisition crash) has one template per recording, each valid inside segment_daq_sec
    # (`camera_sync.templates_for`) -- a syllable is cut from the recording whose segment holds it
    recs = []
    for rec, tpls in cs.templates_for(rv, date, animal, CAM).items():
        vid, tsf = vdir / f"{CAM}_{rec}.avi", vdir / f"{CAM}_{rec}.csv"
        if not (vid.exists() and tsf.exists()):
            continue
        for tpl in tpls:
            if bool(tpl["quality_ok"]):
                seg = tpl.get("segment_daq_sec")
                recs.append((rec, tpl, vid, tsf, None if seg is None else (float(seg[0]), float(seg[1]))))
    if not recs:
        return [dict(r, status="no usable template / video / timestamps") for r in sel]
    t = np.load(Path(feat_dir) / "sessions" / f"{lab}.npz", allow_pickle=True)["t_daq_s"]
    out = []
    caps, tss = {}, {}
    try:
        for r in sel:
            t0, t1 = float(t[r["start_bin"]]), float(t[min(r["stop_bin"], len(t) - 1)])
            a_daq, b_daq = t0 - PAD_S, t1 + PAD_S
            hit = [x for x in recs if x[4] is None or (x[4][0] <= a_daq and b_daq <= x[4][1])]
            if not hit:
                out.append(dict(r, status="no recording covers the syllable"))
                continue
            rec, tpl, vid, tsf, _seg = hit[0]
            if rec not in caps:
                caps[rec] = cv2.VideoCapture(str(vid))                       # READ-ONLY
                tss[rec] = pd.read_csv(tsf, header=None, usecols=[1], dtype="int64").to_numpy().ravel()
            cap, cam_ts = caps[rec], tss[rec]
            ic, sl = float(tpl["intercept_daqSec"]), float(tpl["slope_daqSec_per_camSec"])
            ca, cb = (a_daq - ic) / sl, (b_daq - ic) / sl
            if not camera_healthy_at(tpl, np.array([ca, cb])).all():
                out.append(dict(r, status="camera dropping frames"))
                continue
            f0, f1 = int(np.searchsorted(cam_ts, ca * 1e9)), int(np.searchsorted(cam_ts, cb * 1e9))
            if f0 <= 0 or f1 >= len(cam_ts) or f1 <= f0:
                out.append(dict(r, status="window off the recording"))
                continue
            s0_ns, s1_ns = (t0 - ic) / sl * 1e9, (t1 - ic) / sl * 1e9
            sd = Path(dest) / f"state_{r['state']:02d}"
            sd.mkdir(parents=True, exist_ok=True)
            fn = sd / f"clip_{r['clip']:02d}_{lab}.avi"
            vw = cv2.VideoWriter(str(fn), cv2.VideoWriter_fourcc(*"MJPG"), OUT_FPS, (OUT_PX, OUT_PX))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            mid = None
            for i, fr_idx in enumerate(range(f0, f1)):
                ok, fr = cap.read()
                if not ok:
                    break
                if i % KEEP_EVERY:
                    continue
                inside = s0_ns <= cam_ts[fr_idx] <= s1_ns
                img = cv2.resize(fr, (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
                col = (60, 200, 60) if inside else (140, 140, 140)
                cv2.rectangle(img, (0, 0), (OUT_PX - 1, OUT_PX - 1), col, 6)
                cv2.putText(img, f"state {r['state']}  {animal} {date[4:]} {r['epoch']}", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"syllable {t1 - t0:.2f} s" + ("  <<" if inside else ""), (10, OUT_PX - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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
        vw = cv2.VideoWriter(str(dest / f"state_{k:02d}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), OUT_FPS,
                             (OUT_PX, OUT_PX))
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
                tiles.append(cv2.resize(cv2.imread(str(png)), (TILE, TILE)))
        vw.release()
        tiles += [np.zeros((TILE, TILE, 3), np.uint8)] * (n - len(tiles))
        row = np.hstack(tiles[:n])
        lab = np.zeros((TILE, 70, 3), np.uint8)
        cv2.putText(lab, f"{k}", (12, TILE // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.hstack([lab, row]))
    if rows:
        cv2.imwrite(str(dest / "contact_sheet.png"), np.vstack(rows))


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import arhmm_states as st
    from wfield_local import epochs as ep
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    ap.add_argument("--out", required=True, help="AR-HMM output dir (model_<tag>.pkl, states_<tag>/)")
    ap.add_argument("--feat", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--n", type=int, default=N_PER_STATE)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    M, feats, Z = st.load(Path(a.out), a.tag, Path(a.feat))
    K = int(np.asarray(M["params"]["Ab"]).shape[0])
    u = st.usage(Z, K)
    live = [k for k in range(K) if u[k] >= st.DEAD_USAGE]
    dw = st.dwell_s(Z)
    from wfield_local import arhmm_model as am
    dmed = {int(k): float(v) * am.FS_HZ for k, v in dw["median"].items()}
    epochs = {lab: ep.epoch_of(lab) for lab in Z}
    S = select(Z, live, dmed, epochs, a.n + SPARE)
    S["clip"] = S.groupby("state").cumcount()
    dest = Path(a.dest) if a.dest else Path(oo.root(PathResolver())) / oc.RUN / "arhmm" / "v1" / f"reels_{a.tag}"
    assert_writable(dest)
    dest.mkdir(parents=True, exist_ok=True)
    items = [(lab, g.to_dict("records"), a.feat, str(dest)) for lab, g in S.groupby("label")]
    res, fails = ak.fan_sessions(items, _session_job, jobs=a.jobs, key=lambda it: it[0])
    for f in fails:
        print("  failed:", f)
    man = pd.DataFrame([r for _it, rows in res for r in rows])
    man.to_csv(dest / "reels_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    stitch(dest, man, a.n)
    print(man.status.value_counts().to_string())
    print(f"{len(live)} states -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
