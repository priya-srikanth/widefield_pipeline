"""Pick the BETWEEN-TRIAL spout frames worth labelling: the spout in transit, and where the network is unsure.

    conda activate dlc
    python -m wfield_local.dlc_iti_frames --sessions PS95:20260907 PS93:20260824            # -> iti_rows.csv
    python -m wfield_local.dlc_iti_frames --sessions PS95:20260907 PS93:20260824 --extract  # + PNGs + manifest
    python -m wfield_local.dlc_spout_guide                                                    # the page for the labeller

WHY. Every trial-locked frame (`dlc_frames.phases`, `lick_offsets_s`) shows the spout AT one of its six
positions. It moves BETWEEN trials, and the network had never seen that: it found the retracted spout
in the right place at likelihood 0.58 -- under the 0.6 cutoff -- and dropped it on 11% of frames
(2026-09-26). The fix is frames from the gaps.

THE NETWORK CHOOSES THE FRAMES, NOT A TIME OFFSET. Sampling the ITI at a fixed delay mostly returns the
spout sitting still somewhere already covered. Instead every inter-trial gap that follows a POSITION
CHANGE is scanned at ~21 Hz (every 12th frame at 250 fps) with the current network, and two frames are
kept per gap: the largest jump in predicted spout x (the move itself) and the lowest spout likelihood
(where it cannot cope). Gaps are drawn stratified by the NEW position, two per position, so all six
are represented in every folder.

THE 2026-09-26 RUN. Sessions PS95_20260907 (chronic) and PS93_20260824 (subacute), response window
3.5 s, step 12, seed 92, two gaps per position, snapshot iteration-1/snapshot-best-060. It produced
the 47 frames in `SPOUT_FRAMES_GUIDE.html` (23 + 24, alongside the 24 already-labelled frames in each
folder); the rows, with the spout likelihood and x-jump at each pick, are kept in
`docs/dlc_iti_rows_20260926.csv`. The code ran from a scratch script that night and was recovered into
this module on 2026-09-28 -- the selection logic is unchanged, the GPU inference is behind an
injectable `predict` so the rule is testable without DeepLabCut.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import dlc_frames as DF
from wfield_local.paths import PathResolver

#: Response window after the cue; the gap scanned runs from cue + RW to the next trial_start.
RESPONSE_WINDOW_S = 3.5
#: Every STEP-th camera frame inside the gap (250 fps / 12 ~ 21 Hz).
STEP = 12
#: Gaps per (new) position, per session.
PER_POSITION = 2
#: A gap shorter than this holds no transit worth a frame.
MIN_GAP_S = 1.0
SEED = 92

PHASE_MOVE, PHASE_DOUBT = "iti_spout_move", "iti_low_conf"


# --------------------------------------------------------------------------- the rule (pure)

def position_change_gaps(pos) -> dict[str, list[int]]:
    """{new position: [i, ...]} where the gap after trial i leads to a DIFFERENT position pos[i+1]."""
    pos = np.asarray(pos).astype(str)
    out: dict[str, list[int]] = {}
    for i in np.flatnonzero(pos[1:] != pos[:-1]):
        out.setdefault(str(pos[i + 1]), []).append(int(i))
    return out


def pick_gaps(by_pos: dict[str, list[int]], rng, per_position: int = PER_POSITION) -> list[int]:
    """`per_position` gaps for every position, in position order, shuffled within a position by `rng`.

    Sorted keys and a seeded generator: the same trial table always yields the same gaps, so a
    re-run extracts the same frames rather than a second set beside the first.
    """
    pick: list[int] = []
    for _pnm, v in sorted(by_pos.items()):
        v = list(v)
        rng.shuffle(v)
        pick += v[:per_position]
    return pick


def choose_in_gap(xs, ps) -> tuple[int, int]:
    """(index of the transit frame, index of the doubt frame) within one scanned gap.

    Transit = largest |dx| between consecutive samples (the first sample's dx is 0 by construction,
    so a gap whose spout never moves picks its first frame). Doubt = lowest spout likelihood; a
    frame that failed to decode carries likelihood 0 and is therefore the doubt frame, which is
    what you want -- it is a frame the labeller should look at.
    """
    xs, ps = np.asarray(xs, float), np.asarray(ps, float)
    d = np.abs(np.diff(xs, prepend=xs[:1]))
    k_move = int(np.nanargmax(d)) if np.isfinite(d).any() else 0
    k_doubt = int(np.argmin(ps))
    return k_move, k_doubt


# --------------------------------------------------------------------------- the network

def current_snapshot(project: Path, iteration: int | None = None) -> tuple[Path, Path]:
    """(train dir, best snapshot) of the training project's CURRENT round -- or of `iteration`.

    The round counter lives in the project's `config.yaml`; the best snapshot is the
    `snapshot-best-<epoch>.pt` with the highest epoch in that round's train dir. Round 2 was
    iteration-1 / best-060 and round 3 (2026-09-28) iteration-2 / best-160: hardcoding either would
    quietly pick frames with a superseded network.
    """
    import yaml

    if iteration is None:
        iteration = int(yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8")).get("iteration", 0))
    td = next((project / "dlc-models-pytorch" / f"iteration-{iteration}").glob("*/train"))
    best = sorted(td.glob("snapshot-best-*.pt"), key=lambda q: int(q.stem.rsplit("-", 1)[1]))
    if not best:
        raise FileNotFoundError(f"no snapshot-best-*.pt under {td}")
    return td, best[-1]


def dlc_spout_predictor(rv=None, iteration: int | None = None):
    """A `predict(frame_bgr) -> (spout_x, spout_likelihood)` closure over the trained cam4 network
    of the project's current round (`current_snapshot`).

    Imports torch/deeplabcut lazily so the module (and its tests) load on a box without them.
    """
    import cv2
    import torch
    from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
    from deeplabcut.pose_estimation_pytorch.models import PoseModel

    from wfield_local import dlc_train

    td, snapshot = current_snapshot(dlc_train.train_project(rv), iteration)
    print(f"[dlc_iti_frames] network: {td.parent.parent.name}/{snapshot.name}", flush=True)
    cfg = read_config_as_dict(str(td / "pytorch_config.yaml"))
    model = PoseModel.build(cfg["model"])
    model.load_state_dict(torch.load(snapshot, map_location="cpu", weights_only=True)["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    sp = dlc_train.parts().index("spout")

    def predict(frame_bgr) -> tuple[float, float]:
        a = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(((a - mean) / std).transpose(2, 0, 1)[None]).to(device)
        with torch.inference_mode():
            p = model.get_predictions(model(t))["bodypart"]["poses"][0, 0].cpu().numpy()
        return float(p[sp, 0]), float(p[sp, 2])

    return predict


# --------------------------------------------------------------------------- one session

def scan_session(animal: str, date: str, sid: str, stem: str, epoch: str, predict, rv=None, *,
                 rw_s: float = RESPONSE_WINDOW_S, step: int = STEP, per_position: int = PER_POSITION,
                 min_gap_s: float = MIN_GAP_S, seed: int = SEED, read_frame=None) -> list[dict]:
    """Rows (manifest schema + `spout_p`, `spout_dx`) for one session's chosen ITI frames.

    `read_frame(video_path, frame_index) -> image | None` defaults to an OpenCV seek; injectable so
    the rule can be exercised without a video.
    """
    rv = rv or PathResolver()
    tpl = dict(np.load(Path(rv.root("alignment_templates")) / "cam4" / animal / f"{date}.npz",
                       allow_pickle=True))
    vid = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob("cam4_*.avi"))[0]
    t = pd.read_csv(Path(rv.root("behavior_out")) / "sessions" / animal / date / f"{sid}_trials.csv")
    return scan_trials(t, tpl, str(vid), animal, date, stem, epoch, predict, rw_s=rw_s, step=step,
                       per_position=per_position, min_gap_s=min_gap_s, seed=seed, read_frame=read_frame)


def scan_trials(t: pd.DataFrame, tpl: dict, video: str, animal: str, date: str, stem: str, epoch: str,
                predict, *, rw_s=RESPONSE_WINDOW_S, step=STEP, per_position=PER_POSITION,
                min_gap_s=MIN_GAP_S, seed=SEED, read_frame=None) -> list[dict]:
    if read_frame is None:
        read_frame = _cv2_reader()
    cue = t["cue_s"].to_numpy(float)
    start = t["trial_start_s"].to_numpy(float)
    pos = t["pos_name"].astype(str).to_numpy()
    n_frames = int(tpl["n_cam_frames"])
    fs = float(tpl["fs_daq"])
    slope, icept = float(tpl["slope_daqSample_per_camFrame"]), float(tpl["intercept_daqSample"])
    rows: list[dict] = []
    for i in pick_gaps(position_change_gaps(pos), np.random.default_rng(seed), per_position):
        t0, t1 = cue[i] + rw_s, start[i + 1]                 # the true inter-trial gap
        if t1 - t0 < min_gap_s:
            continue
        f0, f1 = DF.frame_of(tpl, t0, 0.0), DF.frame_of(tpl, t1, 0.0)
        fr = np.arange(f0, min(f1, n_frames), step)
        if len(fr) == 0:
            continue
        xs, ps = [], []
        for f in fr:
            im = read_frame(video, int(f))
            if im is None:
                xs.append(np.nan)
                ps.append(0.0)
                continue
            x, p = predict(im)
            xs.append(x)
            ps.append(p)
        xs, ps = np.array(xs, float), np.array(ps, float)
        d = np.abs(np.diff(xs, prepend=xs[:1]))
        k_move, k_doubt = choose_in_gap(xs, ps)
        for k, phase in ((k_move, PHASE_MOVE), (k_doubt, PHASE_DOUBT)):
            f = int(fr[k])
            rows.append({"animal": animal, "date": date, "cam": "cam4", "epoch": epoch, "video_stem": stem,
                         "frame": f, "trial_id": int(t["trial_id"].iloc[i + 1]), "position": str(pos[i + 1]),
                         "category": "iti", "phase": phase,
                         "t_from_cue_s": round((f * slope + icept) / fs - cue[i], 3),
                         "_group": f"iti{i}:{phase[4:]}", "image": f"img{f:07d}.png", "_video": video,
                         "spout_p": round(float(ps[k]), 3), "spout_dx": round(float(d[k]), 1)})
    return rows


def _cv2_reader():
    import cv2
    caps: dict[str, object] = {}

    def read_frame(video: str, frame: int):
        cap = caps.get(video)
        if cap is None:
            cap = caps[video] = cv2.VideoCapture(video)      # READ-ONLY, always
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, im = cap.read()
        return im if ok else None

    return read_frame


def dedupe(rows: list[dict]) -> pd.DataFrame:
    """One row per (video_stem, frame): a gap whose move and doubt frame coincide contributes once."""
    return pd.DataFrame(rows).drop_duplicates(subset=["video_stem", "frame"]).reset_index(drop=True)


# --------------------------------------------------------------------------- CLI

def _parse_session(spec: str, rv) -> tuple[str, str, str, str, str]:
    """'PS95:20260907' -> (animal, date, sid, video_stem, epoch), from the behaviour outputs on the share."""
    from wfield_local import dlc_train

    animal, date = spec.split(":")
    sess_dir = Path(rv.root("behavior_out")) / "sessions" / animal / date
    trials = sorted(sess_dir.glob(f"{animal}_{date}_*_trials.csv"))
    if not trials:
        raise SystemExit(f"no trial table under {sess_dir}")
    sid = trials[-1].name[: -len("_trials.csv")]
    vid = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob("cam4_*.avi"))
    if not vid:
        raise SystemExit(f"no cam4 video under {Path(rv.root('behavior_cameras')) / date / animal}")
    epoch = dlc_train.session_epochs(rv).get(f"{animal}_{date[4:]}", "")
    return animal, date, sid, vid[0].stem, epoch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True, metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--per-position", type=int, default=PER_POSITION)
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--iteration", type=int, default=None, help="network round (default: the project's current)")
    ap.add_argument("--out", type=Path, default=None, help="rows CSV (default: <staging>/iti_rows.csv)")
    ap.add_argument("--extract", action="store_true", help="also extract the PNGs and append the manifest")
    a = ap.parse_args(argv)
    rv = PathResolver()
    predict = dlc_spout_predictor(rv, a.iteration)
    rows: list[dict] = []
    for spec in a.sessions:
        animal, date, sid, stem, epoch = _parse_session(spec, rv)
        got = scan_session(animal, date, sid, stem, epoch, predict, rv, step=a.step,
                           per_position=a.per_position, seed=a.seed)
        print(f"{animal} {date}: {len(got)} candidate frames", flush=True)
        rows += got
    df = dedupe(rows)
    out = a.out or (DF.staging_root(rv) / "iti_rows.csv")
    df.to_csv(out, index=False)
    print(f"\n{len(df)} ITI frames -> {out}")
    print(df.groupby(["video_stem", "phase"]).size().to_string())
    print(f"spout likelihood at the picks: median {df.spout_p.median():.2f}, "
          f"{100 * (df.spout_p < 0.6).mean():.0f}% below the 0.6 cutoff (the novel ones)")
    if a.extract:
        recs = df.drop(columns=["spout_p", "spout_dx"]).to_dict("records")
        n = DF.extract(recs, rv)
        m = DF.write_manifest(recs, rv)
        print(f"extracted {n} new PNGs -> {DF.staging_root(rv)}; manifest appended -> {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
