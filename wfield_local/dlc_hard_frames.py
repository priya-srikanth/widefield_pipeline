"""cam4 labelling round 4: the frames DLC round 3 gets wrong, each with consecutive CONTEXT around it.

    conda activate dlc
    python -m wfield_local.dlc_hard_frames scan      # DLC round 3 over the windows; poses cached per session
    python -m wfield_local.dlc_hard_frames clips     # the same windows -> one local lossless .avi per session
    #   WSL: ffmpeg -nostdin -> .mp4; litpose predict <occl model> <clip>.mp4
    #        --overrides dali.base.predict.sequence_length=16   (NOT the default 96: it filled the 8 GB card
    #        and the machine blue-screened twice on 2026-09-30, HYPERVISOR_ERROR)
    python -m wfield_local.dlc_hard_frames picks --lp-dir <folder of <animal>_<date>.csv>   # DLC + LP picks
    python -m wfield_local.dlc_hard_frames extract   # PNGs, manifest, sync (cam4), matched cam1 bursts
    python -m wfield_local.dlc_cam1_guide            # the worksheet (cam1 + cam4 round 4)

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
#: kind -> (per-session cap, context half-width). Picked from DLC round 3's predictions.
KINDS = {"incomplete_tongue": (4, 4), "erratic_tongue": (1, 4), "erratic_jaw": (1, 2), "tricky_spout": (1, 0)}
#: Picked from Lightning Pose's predictions on the SAME frames (Priya, 2026-09-30: "go ahead with selected
#: LP-focused frames"): LP's own erratic tongue/jaw (the two models fail differently), and frames where both
#: are confident but DISAGREE by > `DISAGREE_PX` -- the labeller adjudicates (this catches e.g. LP placing
#: the tongue tip off the distal end on a sideways lick, PS95 0907 frame 402).
#: Priority order = dict order: disagreements first (they are what caught the sideways-lick tip).
LP_KINDS = {"disagree_tongue": (1, 4), "disagree_jaw": (1, 2), "lp_erratic_tongue": (1, 4), "lp_erratic_jaw": (1, 2)}
HALF = {k: v[1] for k, v in {**KINDS, **LP_KINDS}.items()}
DISAGREE_PX = 15.0
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


def disagreements(A: dict, B: dict, px: float = DISAGREE_PX) -> list[tuple[int, str, float]]:
    """Frames where BOTH models are confident (p > 0.8) on tongue or jaw and their points are > ``px`` apart."""
    out = []
    for part in ("tongue", "jaw"):
        ax, ay, ap = A[part]
        bx, by, bp = B[part]
        d = np.hypot(ax - bx, ay - by)
        for i in np.flatnonzero((ap > P_SURE) & (bp > P_SURE) & (d > px)):
            out.append((int(i), f"disagree_{part}", float(d[i])))
    return out


def lp_candidates(P_lp: dict, P_dlc: dict, stats_lp) -> list[tuple[int, str, float]]:
    """LP-focused candidates for one window: LP's own erratic tongue/jaw + DLC-vs-LP disagreement."""
    own = [(i, "lp_" + k, s) for i, k, s in candidates(P_lp, *stats_lp) if k in ("erratic_tongue", "erratic_jaw")]
    return own + disagreements(P_lp, P_dlc)


def select(cands: list[tuple[int, str, float]], caps: dict | None = None, min_sep: int = MIN_SEP,
           taken=(), kinds=None) -> list[tuple[int, str, float]]:
    """Highest-scoring candidates per kind, in priority order, all >= ``min_sep`` apart and from ``taken``.

    ``cands`` indices are SESSION frame numbers (already offset from their window). ``kinds`` defaults to
    `KINDS` (the DLC pass); the LP pass passes `LP_KINDS` and the DLC picks as ``taken``.
    """
    kinds = kinds or KINDS
    caps = caps or {k: v[0] for k, v in kinds.items()}
    chosen: list[tuple[int, str, float]] = []
    fixed = [int(t) for t in taken]
    for kind in kinds:
        pool = sorted((c for c in cands if c[1] == kind), key=lambda c: -c[2])
        k = 0
        for c in pool:
            if k >= caps.get(kind, 0):
                break
            if all(abs(c[0] - d[0]) >= min_sep for d in chosen) and all(abs(c[0] - t) >= min_sep for t in fixed):
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
        half = HALF[kind]
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


def _session_io(animal, date, sid, rv):
    tpl = dict(np.load(Path(rv.root("alignment_templates")) / "cam4" / animal / f"{date}.npz", allow_pickle=True))
    vid = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob("cam4_*.avi"))[0]
    t = pd.read_csv(Path(rv.root("behavior_out")) / "sessions" / animal / date / f"{sid}_trials.csv")
    return tpl, vid, t[np.isfinite(t["cue_s"].astype(float))]


def windows_for(t: pd.DataFrame, tpl: dict) -> list[tuple[int, int, dict]]:
    """[(f0, f1, trial), ...]: the scanned windows. Seeded, so the DLC scan, the LP clip and a re-run all
    cover the SAME frames."""
    rng = np.random.default_rng(SEED)
    trials = pd.concat([g.sample(min(len(g), TRIALS_PER_POSITION), random_state=int(rng.integers(1 << 31)))
                        for _, g in t.groupby("pos_name")]).sort_values("cue_s")
    n = int(tpl["n_cam_frames"])
    out = []
    for _, tr in trials.iterrows():
        f0 = max(0, DF.frame_of(tpl, tr.cue_s, WIN_S[0]))
        f1 = min(n, DF.frame_of(tpl, tr.cue_s, WIN_S[1]))
        if f1 - f0 >= 3:
            out.append((f0, f1, {"cue_s": float(tr.cue_s), "trial_id": int(tr.trial_id), "pos_name": str(tr.pos_name)}))
    return out


def _as_parts(pose: np.ndarray) -> dict:
    return {p: (pose[:, k, 0], pose[:, k, 1], pose[:, k, 2]) for k, p in enumerate(PARTS)}


def save_poses(path: Path, scanned) -> None:
    """One npz per session from ``[(f0, trial, pose (n, 4, 3)), ...]``."""
    np.savez_compressed(path, f0=np.array([f0 for f0, _, _ in scanned]),
                        f1=np.array([f0 + len(p) for f0, _, p in scanned]),
                        trials=np.array([pd.Series(tr).to_json() for _, tr, _ in scanned]),
                        pose=np.concatenate([p for *_, p in scanned]) if scanned else np.empty((0, len(PARTS), 3)))


def load_poses(path: Path):
    """[(f0, trial, pose (n, 4, 3)), ...] from `save_poses`."""
    z = np.load(path, allow_pickle=False)
    out, k = [], 0
    for f0, f1, tj in zip(z["f0"], z["f1"], z["trials"]):
        n = int(f1 - f0)
        out.append((int(f0), pd.read_json(str(tj), typ="series").to_dict(), z["pose"][k:k + n]))
        k += n
    return out


def picks_from(wins_dlc, wins_lp=None) -> tuple[list, dict]:
    """Session picks: the DLC pass, then (if LP poses are given, same windows) the LP pass around them."""
    Pd = [_as_parts(p) for _, _, p in wins_dlc]
    stats_d = session_stats(Pd)
    cands = []
    for P, (f0, tr, _) in zip(Pd, wins_dlc):
        cands += [(f0 + i, k, s, tr) for i, k, s in candidates(P, *stats_d)]
    trial_of = {c[0]: c[3] for c in cands}
    picks = [(f, k, s, trial_of[f]) for f, k, s in select([c[:3] for c in cands])]
    info = {"frames": sum(len(p) for *_, p in wins_dlc),
            "dlc_candidates": pd.Series([c[1] for c in cands], dtype=str).value_counts().to_dict()}
    if wins_lp is not None:
        Pl = [_as_parts(p) for _, _, p in wins_lp]
        stats_l = session_stats(Pl)
        lc = []
        for A, B, (f0, tr, _) in zip(Pl, Pd, wins_dlc):
            n = min(len(A["jaw"][0]), len(B["jaw"][0]))
            A = {k: tuple(a[:n] for a in v) for k, v in A.items()}
            B = {k: tuple(b[:n] for b in v) for k, v in B.items()}
            lc += [(f0 + i, k, s, tr) for i, k, s in lp_candidates(A, B, stats_l)]
        lt = {c[0]: c[3] for c in lc}
        picks += [(f, k, s, lt[f]) for f, k, s in select([c[:3] for c in lc], kinds=LP_KINDS, taken=[p[0] for p in picks])]
        info["lp_candidates"] = pd.Series([c[1] for c in lc], dtype=str).value_counts().to_dict()
    info["picked"] = pd.Series([p[1] for p in picks], dtype=str).value_counts().to_dict()
    return picks, info


def scan_poses(animal, date, sid, predict, rv=None):
    """DLC round 3 over the session's windows -> [(f0, trial, pose), ...]."""
    rv = rv or PathResolver()
    tpl, vid, t = _session_io(animal, date, sid, rv)
    out = []
    for f0, f1, tr in windows_for(t, tpl):
        pose = predict_window(str(vid), f0, f1, predict)
        if len(pose) >= 3:
            out.append((f0, tr, pose))
    return out


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


#: Where per-session DLC poses, the session list and the picked rows live (on the DLC share).
def _cache(rv) -> Path:
    c = DF.staging_root(rv).parent / "round4_scan"
    assert_writable(c)
    c.mkdir(parents=True, exist_ok=True)
    return c


#: Local (not the share) scratch for the LP clips: a lossless window clip is ~1.5 GB per session.
LOCAL_CLIPS = Path.home() / "lp_clips" / "round4"


def _sessions(rv, specs=None):
    c = _cache(rv) / "sessions.csv"
    if specs:
        from wfield_local.dlc_iti_frames import _parse_session
        sess = [(a_, d_, s_, e_) for a_, d_, s_, _st, e_ in (_parse_session(s, rv) for s in specs)]
    elif c.exists():
        sess = [tuple(r) for r in pd.read_csv(c, dtype=str).itertuples(index=False)]
    else:
        sess = choose_sessions(rv)
    pd.DataFrame(sess, columns=["animal", "date", "sid", "epoch"]).to_csv(c, index=False)
    return sess


def cmd_scan(rv, sess) -> None:
    """DLC round 3 (+ prior) over every session's windows; poses cached per session as each finishes.

    On 2026-09-30 a first run was killed after 4 of 6 sessions with everything in memory; now each session
    is written when done and re-used on a re-run (delete its npz to force a re-scan)."""
    from wfield_local import dlc_prior
    predict = None
    with dlc_prior.apply(rv, cam="cam4"):
        for animal, date, sid, epoch in sess:
            npz = _cache(rv) / f"{animal}_{date}_dlc.npz"
            if npz.exists():
                print(f"{animal} {date}: DLC poses cached", flush=True)
                continue
            predict = predict or pose_predictor(rv)
            scanned = scan_poses(animal, date, sid, predict, rv)
            save_poses(npz, scanned)
            print(f"{animal} {date} {epoch}: {sum(len(p) for *_, p in scanned)} frames scanned", flush=True)


def cmd_clips(rv, sess) -> None:
    """The same windows, cut losslessly into one local .avi per session, for LP (`LOCAL_CLIPS`)."""
    import cv2
    LOCAL_CLIPS.mkdir(parents=True, exist_ok=True)
    for animal, date, sid, _ in sess:
        out = LOCAL_CLIPS / f"{animal}_{date}.avi"
        if out.exists() or (LOCAL_CLIPS / f"{animal}_{date}.mp4").exists():
            continue
        tpl, vid, _t = _session_io(animal, date, sid, rv)
        wins = load_poses(_cache(rv) / f"{animal}_{date}_dlc.npz")      # the frames DLC actually saw
        cap = cv2.VideoCapture(str(vid))                                 # READ-ONLY
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), 250.0, (w, h))
        for f0, _tr, pose in wins:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            for _ in range(len(pose)):
                ok, im = cap.read()
                wr.write(im if ok else np.zeros((h, w, 3), np.uint8))
        cap.release()
        wr.release()
        print(f"{animal} {date}: clip {sum(len(p) for *_, p in wins)} frames -> {out}", flush=True)


def _lp_windows(lp_csv: Path, wins_dlc) -> list:
    """Split an LP predictions CSV for a session clip back into the DLC windows (same order, same lengths)."""
    d = pd.read_csv(lp_csv, header=[0, 1, 2], index_col=0)
    d.columns = d.columns.droplevel(0)
    pose = np.stack([np.stack([d[p]["x"], d[p]["y"], d[p]["likelihood"]], 1) for p in PARTS], 1)
    out, k = [], 0
    for f0, tr, p in wins_dlc:
        out.append((f0, tr, pose[k:k + len(p)]))
        k += len(p)
    if k != len(pose):
        raise SystemExit(f"{lp_csv.name}: {len(pose)} LP rows vs {k} DLC frames -- clip and scan disagree")
    return out


def cmd_picks(rv, sess, lp_dir: Path | None) -> Path:
    """Picks for every session (DLC pass, plus the LP pass when ``lp_dir`` holds `<animal>_<date>.csv`)."""
    rows = []
    for animal, date, sid, epoch in sess:
        tpl, vid, _t = _session_io(animal, date, sid, rv)
        wins = load_poses(_cache(rv) / f"{animal}_{date}_dlc.npz")
        lp = None
        if lp_dir is not None:
            f = Path(lp_dir) / f"{animal}_{date}.csv"
            lp = _lp_windows(f, wins) if f.exists() else None
            if lp is None:
                print(f"{animal} {date}: no LP predictions at {f} -> DLC picks only", flush=True)
        picks, info = picks_from(wins, lp)
        meta = {"animal": animal, "date": date, "cam": "cam4", "epoch": epoch, "video_stem": vid.stem}
        got = rows_for(picks, meta, tpl, int(tpl["n_cam_frames"]))
        for r in got:
            r["_video"] = str(vid)
        rows += got
        print(f"{animal} {date} {epoch}: {info}", flush=True)
    out = _cache(rv) / "round4_rows.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).groupby(["video_stem", "category"]).size().unstack(fill_value=0).to_string())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["scan", "clips", "picks", "extract"])
    ap.add_argument("--sessions", nargs="*", metavar="ANIMAL:YYYYMMDD", help="override the session choice")
    ap.add_argument("--lp-dir", type=Path, default=None, help="picks: folder of LP CSVs named <animal>_<date>.csv")
    a = ap.parse_args(argv)
    rv = PathResolver()
    sess = _sessions(rv, a.sessions)
    print("sessions:", ", ".join(f"{s[0]}_{s[1]}({s[3]})" for s in sess), flush=True)
    if a.step == "scan":
        cmd_scan(rv, sess)
        return 0
    if a.step == "clips":
        cmd_clips(rv, sess)
        return 0
    if a.step == "picks":
        cmd_picks(rv, sess, a.lp_dir)
        return 0
    rows = pd.read_csv(_cache(rv) / "round4_rows.csv", dtype={"date": str}).to_dict("records")
    df = pd.DataFrame(rows)
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
