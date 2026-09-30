"""cam4 labelling round 4: the frames DLC round 3 gets wrong, each with consecutive CONTEXT around it.

    conda activate dlc
    python -m wfield_local.dlc_hard_frames --dry-run            # scan + print the picks, write nothing
    python -m wfield_local.dlc_hard_frames                      # + extract, manifest, sync (cam4), cam1 bursts
    python -m wfield_local.dlc_cam1_guide                        # the worksheet (cam1 + cam4 round 4)

WHY (Priya, 2026-09-30). Round 3 misses INCOMPLETE LICKS -- the tongue tip just between the lips, no spout
contact, so the lick sensor cannot find them -- and they matter most post-stroke. Measured on the round-3
review clips: of mouth openings of 12-30 px DLC finds a tongue on 4/9 (PS92 acute), 4/10 (PS93), 16/50
(PS95 chronic), against >=98 % for full openings. Also wanted: tricky jaw and spout frames, and erratic
high-confidence off-target tongue/jaw. Full reasoning: DECISIONS.md, 2026-09-30 (two entries).

HOW. The current network (with the production prior) is run at FULL frame rate over trial windows
(cue - 1.0 s .. cue + 3.5 s, two trials per position, per session) -- continuity-based rules need every
frame. Per session, from those predictions:

  incomplete_tongue  jaw-nose opening 12-30 px above the session's resting level (a peak), and no tongue
                     p > 0.6 within +-6 frames (24 ms)                                   context +-4
  erratic_tongue     confident (p > 0.8) tongue that jumps > 15 px from BOTH neighbours while the
                     neighbours agree, or a confident tongue with the mouth closed        context +-4
  erratic_jaw        the same spike rule for the jaw, or a confident jaw far lateral of its usual
                     offset from the nose (the "jaw on the tongue's edge" pattern)        context +-2
  tricky_spout       spout p in 0.1-0.6 (a local minimum), or moving > 5 px/frame at p < 0.9   none

Sessions: two per post-stroke epoch (acute, subacute, chronic), animals spread -- six in all, no pre-stroke;
none already in the cam4 labelled set, all with passing templates on
cam4 AND cam1 (the matched bursts). Picks within a session are >= 0.25 s apart.

TARGETS, CONTEXT, PROMOTED CONTEXT -- same rule as cam1 (`dlc_context_frames`): the centre is a target
(category ``round4``, phase = the rule name), neighbours are ``context``, and every ~3rd neighbour is
promoted to ``context_label`` (suggested, flexible). All-blank frames never reach training
(`dlc_train.drop_unlabelled`).

MATCHED cam1 BURSTS go to ``_frame_staging_unassigned/`` -- NOT `_frame_staging`, because
`dlc_project.sync_frames` copies every ``cam1_*`` folder there into the labelling project, i.e. onto the
student's worksheet. They exist so a moment labelled on cam4 is pairable later (multi-view LP needs the
image in every view; see DECISIONS). Nobody is asked to label them yet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import dlc_frames as DF
from wfield_local.dlc_context_frames import CONTEXT, CONTEXT_LABEL, PROMOTE_SPACING, burst
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

ROUND = "round4"
PARTS = ("nose", "jaw", "tongue", "spout")
#: kind -> (per-session cap, context half-width)
KINDS = {"incomplete_tongue": (4, 4), "erratic_tongue": (1, 4), "erratic_jaw": (1, 2), "tricky_spout": (1, 0)}
WIN_S = (-1.0, 3.5)
TRIALS_PER_POSITION = 2
MIN_SEP = 62                     # frames (0.25 s at 250 fps)
P_OK, P_SURE = 0.6, 0.8
SEED = 104
PER_EPOCH = 2
POST_EPOCHS = ("acute", "subacute", "chronic")


# --------------------------------------------------------------------------- the rules (pure)

def _spikes(x, y, p, jump=15.0, agree=8.0, p_min=P_SURE) -> np.ndarray:
    """Frames where a confident point jumps > ``jump`` px from both neighbours, which agree within ``agree``."""
    xy = np.c_[x, y]
    out = np.zeros(len(x), bool)
    if len(x) < 3:
        return out
    d_prev = np.linalg.norm(xy[1:-1] - xy[:-2], axis=1)
    d_next = np.linalg.norm(xy[1:-1] - xy[2:], axis=1)
    d_nb = np.linalg.norm(xy[2:] - xy[:-2], axis=1)
    conf = (p[1:-1] > p_min) & (p[:-2] > p_min) & (p[2:] > p_min)
    out[1:-1] = conf & (d_prev > jump) & (d_next > jump) & (d_nb < agree)
    return out


def candidates(P: dict, rest_open: float, lat_med: float, lat_mad: float) -> list[tuple[int, str, float]]:
    """``[(index, kind, score), ...]`` over ONE contiguous window of predictions.

    ``P[part]`` = (x, y, p) arrays. ``rest_open`` / ``lat_med`` / ``lat_mad`` are session-level (the resting
    jaw-nose opening, and the jaw's median lateral offset from the nose and its MAD), so a window is judged
    against the session, not against itself.
    """
    from scipy.signal import find_peaks

    nx, ny, npp = P["nose"]
    jx, jy, jp = P["jaw"]
    tx, ty, tp = P["tongue"]
    sx, sy, sp = P["spout"]
    n = len(jx)
    out: list[tuple[int, str, float]] = []
    opening = np.where(jp > P_OK, jy - ny, np.nan) - rest_open
    filled = pd.Series(opening).interpolate(limit=5).fillna(0.0).to_numpy()
    pk, _ = find_peaks(filled, height=12, distance=15, prominence=8)
    for i in pk:
        if filled[i] <= 30 and tp[max(0, i - 6): i + 7].max() <= P_OK:
            out.append((int(i), "incomplete_tongue", float(filled[i])))
    for i in np.flatnonzero(_spikes(tx, ty, tp)):
        out.append((int(i), "erratic_tongue", 100.0))
    closed = (tp > P_SURE) & (np.nan_to_num(opening, nan=99.0) < 4)
    run = np.convolve(closed.astype(int), np.ones(3, int), "same") == 3
    for i in np.flatnonzero(run):
        out.append((int(i), "erratic_tongue", float(tp[i])))
    for i in np.flatnonzero(_spikes(jx, jy, jp)):
        out.append((int(i), "erratic_jaw", 100.0))
    lat = np.abs((jx - nx) - lat_med)
    for i in np.flatnonzero((jp > P_SURE) & (lat > max(15.0, 4 * lat_mad))):
        out.append((int(i), "erratic_jaw", float(lat[i])))
    dsx = np.abs(np.diff(sx, prepend=sx[:1]))
    for i in range(1, n - 1):
        unsure = P_OK - 0.5 <= sp[i] <= P_OK and sp[i] <= sp[i - 1] and sp[i] <= sp[i + 1]
        if unsure or (dsx[i] > 5 and sp[i] < 0.9):
            out.append((i, "tricky_spout", float(P_OK - sp[i] + dsx[i] / 10)))
    return out


def select(cands: list[tuple[int, str, float]], caps: dict | None = None, min_sep: int = MIN_SEP) -> list[tuple[int, str, float]]:
    """Highest-scoring candidates per kind, in `KINDS` priority order, all >= ``min_sep`` apart.

    ``cands`` indices are SESSION frame numbers (already offset from their window).
    """
    caps = caps or {k: v[0] for k, v in KINDS.items()}
    chosen: list[tuple[int, str, float]] = []
    for kind in KINDS:
        pool = sorted((c for c in cands if c[1] == kind), key=lambda c: -c[2])
        k = 0
        for c in pool:
            if k >= caps.get(kind, 0):
                break
            if all(abs(c[0] - d[0]) >= min_sep for d in chosen):
                chosen.append(c)
                k += 1
    return chosen


def session_stats(windows: list[dict]) -> tuple[float, float, float]:
    """(resting opening, median lateral jaw offset, its MAD) over every confident frame of the session."""
    op, lat = [], []
    for P in windows:
        ok = (P["jaw"][2] > P_OK) & (P["nose"][2] > P_OK)
        op.append((P["jaw"][1] - P["nose"][1])[ok])
        lat.append((P["jaw"][0] - P["nose"][0])[ok])
    op, lat = np.concatenate(op), np.concatenate(lat)
    med = float(np.median(lat))
    return float(np.percentile(op, 20)), med, float(np.median(np.abs(lat - med)))


# --------------------------------------------------------------------------- rows (pure)

def rows_for(picks, meta: dict, tpl: dict, n_frames: int) -> list[dict]:
    """Target + context + promoted-context manifest rows for one session's picks."""
    fs = float(tpl["fs_daq"])
    slope, icept = float(tpl["slope_daqSample_per_camFrame"]), float(tpl["intercept_daqSample"])
    rows: list[dict] = []
    taken: set[int] = set()
    for f, kind, _score, trial in picks:
        half = KINDS[kind][1]
        ctx = [g for g in burst(f, half, n_frames) if g != f and g not in taken]
        # Every 3rd frame COUNTED FROM THE CENTRE (+-3), not `promote`'s greedy walk, which on a single
        # centred burst gives the lopsided -4/+3. Same 12 ms spacing, symmetric about the pick.
        sug = {g for g in ctx if (g - f) % PROMOTE_SPACING == 0}
        cue = trial["cue_s"]
        for g, cat, ph in [(f, ROUND, kind)] + [(g, CONTEXT_LABEL if g in sug else CONTEXT,
                                                   f"ctx_{kind}{g - f:+d}") for g in ctx]:
            if g in taken:
                continue
            taken.add(g)
            rows.append({**meta, "frame": int(g), "trial_id": int(trial["trial_id"]),
                         "position": str(trial["pos_name"]), "category": cat, "phase": ph,
                         "t_from_cue_s": round((g * slope + icept) / fs - cue, 4),
                         "image": f"img{int(g):07d}.png"})
    return rows


# --------------------------------------------------------------------------- inference (impure)

def pose_predictor(rv=None, iteration: int | None = None, batch: int = 16):
    """``predict(frames_bgr) -> (N, 4, 3)`` [x, y, p] per part, the current round's network, prior applied
    by the caller's `dlc_prior.apply` context."""
    import cv2
    import torch
    from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
    from deeplabcut.pose_estimation_pytorch.models import PoseModel

    from wfield_local import dlc_train
    from wfield_local.dlc_iti_frames import current_snapshot

    td, snapshot = current_snapshot(dlc_train.train_project(rv), iteration)
    print(f"[dlc_hard_frames] network: {td.parent.parent.name}/{snapshot.name}", flush=True)
    cfg = read_config_as_dict(str(td / "pytorch_config.yaml"))
    model = PoseModel.build(cfg["model"])
    model.load_state_dict(torch.load(snapshot, map_location="cpu", weights_only=True)["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    assert tuple(dlc_train.parts()) == PARTS, dlc_train.parts()
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def predict(frames) -> np.ndarray:
        out = []
        for k in range(0, len(frames), batch):
            a = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[k:k + batch]]).astype(np.float32) / 255.0
            t = torch.from_numpy(((a - mean) / std).transpose(0, 3, 1, 2)).to(device)
            with torch.inference_mode():
                out.append(model.get_predictions(model(t))["bodypart"]["poses"][:, 0].cpu().numpy())
        return np.concatenate(out)

    return predict


def predict_window(video: str, f0: int, f1: int, predict, chunk: int = 64) -> np.ndarray:
    """(n, 4, 3) poses for frames f0..f1-1, decoded sequentially and predicted ``chunk`` at a time -- a
    4.5 s window is ~1100 full frames (~1.5 GB) if held at once."""
    import cv2
    cap = cv2.VideoCapture(video)                      # READ-ONLY, always
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    out, buf = [], []
    for _ in range(f0, f1):
        ok, im = cap.read()
        if not ok:
            break
        buf.append(im)
        if len(buf) == chunk:
            out.append(predict(buf))
            buf = []
    if buf:
        out.append(predict(buf))
    cap.release()
    return np.concatenate(out) if out else np.empty((0, len(PARTS), 3))


def choose_sessions(rv=None, per_epoch: int = PER_EPOCH, epochs_=POST_EPOCHS) -> list[tuple[str, str, str, str]]:
    """(animal, date, sid, epoch): ``per_epoch`` sessions in each post-stroke epoch, animals spread
    (`dlc_frames._thin_epochs`), each the most central of that animal's epoch that is not already labelled
    on cam4 and has passing templates on cam4 AND cam1.

    Six sessions, post-stroke only (Priya, 2026-09-30: "for cam4, we can use fewer sessions"): the first
    draft took one session per animal x epoch plus two pre-stroke (13 sessions, ~220 frames to label).
    Incomplete licks are the post-stroke problem, so pre-stroke was the part to drop."""
    from wfield_local import dlc_project, epochs

    rv = rv or PathResolver()
    labelled = {p.name for p in (dlc_project.project_dir(rv) / "labeled-data").glob("cam4_*")}
    man = pd.read_csv(DF.staging_root(rv) / "frame_manifest.csv", dtype=str)
    labelled_dates = set(zip(man[man.video_stem.isin(labelled)].animal, man[man.video_stem.isin(labelled)].date))
    base = Path(rv.root("behavior_out")) / "sessions"
    by: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for an_dir in sorted(p for p in base.glob("PS*") if p.is_dir()):
        for sess in sorted(p for p in an_dir.iterdir() if p.is_dir()):
            an, date = an_dir.name, sess.name
            ep = epochs.epoch_of(f"{an}_{date[4:]}")
            if ep is None or (an, date) in labelled_dates:
                continue
            for t in sorted(sess.glob("*_trials.csv")):
                by.setdefault((an, ep), []).append((date, t.name[: -len("_trials.csv")]))
    out = []
    for (an, ep), sess in sorted(by.items()):
        if ep not in epochs_:
            continue
        pick = DF._pick_middle(sess, lambda ds, _an=an: DF.usable(_an, ds[0], ["cam4", "cam1"], rv))
        if pick is not None:
            out.append((an, pick[0], pick[1], ep))
    return DF._thin_epochs(out, per_epoch)


def scan_session(animal, date, sid, epoch, predict, rv=None) -> tuple[list[dict], dict]:
    rv = rv or PathResolver()
    tpl = dict(np.load(Path(rv.root("alignment_templates")) / "cam4" / animal / f"{date}.npz", allow_pickle=True))
    vid = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob("cam4_*.avi"))[0]
    t = pd.read_csv(Path(rv.root("behavior_out")) / "sessions" / animal / date / f"{sid}_trials.csv")
    t = t[np.isfinite(t["cue_s"].astype(float))]
    rng = np.random.default_rng(SEED)
    trials = pd.concat([g.sample(min(len(g), TRIALS_PER_POSITION), random_state=int(rng.integers(1 << 31)))
                        for _, g in t.groupby("pos_name")]).sort_values("cue_s")
    n_frames = int(tpl["n_cam_frames"])
    windows, spans = [], []
    for _, tr in trials.iterrows():
        f0 = max(0, DF.frame_of(tpl, tr.cue_s, WIN_S[0]))
        f1 = min(n_frames, DF.frame_of(tpl, tr.cue_s, WIN_S[1]))
        pose = predict_window(str(vid), f0, f1, predict)
        if len(pose) < 3:
            continue
        windows.append({p: (pose[:, k, 0], pose[:, k, 1], pose[:, k, 2]) for k, p in enumerate(PARTS)})
        spans.append((f0, tr))
    rest, lat_med, lat_mad = session_stats(windows)
    cands = []
    for P, (f0, tr) in zip(windows, spans):
        cands += [(f0 + i, k, s, tr) for i, k, s in candidates(P, rest, lat_med, lat_mad)]
    picks = select([c[:3] for c in cands])
    trial_of = {c[0]: c[3] for c in cands}
    picks = [(f, k, s, trial_of[f]) for f, k, s in picks]
    meta = {"animal": animal, "date": date, "cam": "cam4", "epoch": epoch, "video_stem": vid.stem}
    rows = rows_for(picks, meta, tpl, n_frames)
    for r in rows:
        r["_video"] = str(vid)
    info = {"stem": vid.stem, "frames_scanned": sum(len(P["jaw"][0]) for P in windows),
            "candidates": pd.Series([c[1] for c in cands]).value_counts().to_dict(),
            "picked": pd.Series([p[1] for p in picks]).value_counts().to_dict()}
    return rows, info


def matched_cam1(rows: list[dict], rv=None) -> list[dict]:
    """The cam1 frame at the same DAQ instant as every cam4 row (targets and context alike)."""
    rv = rv or PathResolver()
    out = []
    for (animal, date), grp in pd.DataFrame(rows).groupby(["animal", "date"]):
        t4 = dict(np.load(Path(rv.root("alignment_templates")) / "cam4" / animal / f"{date}.npz", allow_pickle=True))
        t1 = dict(np.load(Path(rv.root("alignment_templates")) / "cam1" / animal / f"{date}.npz", allow_pickle=True))
        vid1 = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob("cam1_*.avi"))[0]
        fs = float(t4["fs_daq"])
        for r in grp.to_dict("records"):
            t_daq = (r["frame"] * float(t4["slope_daqSample_per_camFrame"]) + float(t4["intercept_daqSample"])) / fs
            f1 = DF.frame_of(t1, t_daq, 0.0)
            out.append({**r, "cam": "cam1", "video_stem": vid1.stem, "frame": int(f1),
                        "image": f"img{int(f1):07d}.png", "_video": str(vid1)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sessions", nargs="*", metavar="ANIMAL:YYYYMMDD", help="override the session choice")
    a = ap.parse_args(argv)
    rv = PathResolver()
    from wfield_local import dlc_prior

    if a.sessions:
        from wfield_local.dlc_iti_frames import _parse_session
        sess = []
        for spec in a.sessions:
            animal, date, sid, _stem, epoch = _parse_session(spec, rv)
            sess.append((animal, date, sid, epoch))
    else:
        sess = choose_sessions(rv)
    print("sessions:", ", ".join(f"{s[0]}_{s[1]}({s[3]})" for s in sess), flush=True)
    predict = pose_predictor(rv)
    rows = []
    with dlc_prior.apply(rv, cam="cam4"):
        for animal, date, sid, epoch in sess:
            got, info = scan_session(animal, date, sid, epoch, predict, rv)
            print(f"{animal} {date} {epoch}: scanned {info['frames_scanned']} frames; candidates "
                  f"{info['candidates']}; picked {info['picked']}", flush=True)
            rows += got
    df = pd.DataFrame(rows)
    out = DF.staging_root(rv).parent / "round4_rows.csv"
    if a.dry_run:
        print(df.groupby(["video_stem", "category"]).size().unstack(fill_value=0).to_string())
        return 0
    assert_writable(out.parent)
    df.drop(columns=["_video"]).to_csv(out, index=False)
    n4 = DF.extract(rows, rv)
    DF.write_manifest(rows, rv)
    from wfield_local import dlc_project
    folders, images, _ = dlc_project.sync_frames(dlc_project.project_dir(rv), rv, cams=["cam4"])
    c1 = matched_cam1(rows, rv)
    un = DF.staging_root(rv).parent / "_frame_staging_unassigned"
    DF.write_manifest(c1, rv, dest=un / "frame_manifest.csv")
    n1 = _extract_to(c1, un)
    print(f"cam4: {n4} PNGs, synced {images} into {folders} folders; cam1 matched: {n1} PNGs -> {un} (not on any worksheet)")
    print(df.groupby(["video_stem", "category"]).size().unstack(fill_value=0).to_string())
    return 0


def _extract_to(rows: list[dict], root: Path) -> int:
    """`dlc_frames.extract`, but into ``root`` instead of `_frame_staging`."""
    import cv2
    written = 0
    for video, group in DF._by_video(rows):
        dest = root / group[0]["video_stem"]
        assert_writable(dest)
        dest.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        for r in sorted(group, key=lambda x: x["frame"]):
            out = dest / r["image"]
            if out.exists():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
            ok, im = cap.read()
            if ok:
                cv2.imwrite(str(out), im)
                written += 1
        cap.release()
    return written


if __name__ == "__main__":
    raise SystemExit(main())
