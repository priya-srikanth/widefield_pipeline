"""Extract a stratified DeepLabCut LABELLING set from the widefield behavior cameras.

DLC's own ``extract_frames`` picks frames by k-means over appearance, which on a 1.5 M-frame
recording of a head-fixed mouse returns mostly the resting posture: the animal spends most of its
time not licking, so appearance clustering under-samples exactly the events the analysis is about.
This picks frames by EXPERIMENTAL DESIGN instead -- spout position x within-trial phase -- using the
cue times and the camera alignment template that ``behavior_clips`` already established.

**Coverage comes from spanning animals and EPOCHS, not from many frames of one session.** A network
labelled only on pre-stroke frames degrades precisely where the result lives; the 2pRAM cohort saw
post-stroke tongue tracking fall apart in its two severe animals, which is a tracking artefact
sitting on top of a real deficit and nearly impossible to separate afterwards. Hence a small
``per_session`` and a ``--cohort`` mode that walks one session per animal x epoch.

SOURCE VIDEOS ARE OPENED READ-ONLY AND NEVER WRITTEN, the same guarantee ``behavior_clips`` makes:
``cv2.VideoCapture`` is the only thing that touches them and ``assert_writable`` guards the output,
so a path bug lands as a refusal rather than as an edit to an irreplaceable recording.

FRAMES ARE WRITTEN AS PNG, never JPEG. Labels are placed on these images and inference then runs on
the raw video; a lossy intermediate would train the network on compression artefacts that the
inference frames do not have.

SELECTION IS DETERMINISTIC (``dlc.frames.seed``), keyed on animal+date+camera. Re-running extends
nothing and re-picks the same frames, so a second run cannot quietly grow a divergent labelling set
beside the one somebody has already spent hours annotating.

Layout, which is DLC's own so the project can adopt it without copying::

    <behavior_cameras>/dlc/labeled-data/<video-stem>/img<FRAME>.png
    <behavior_cameras>/dlc/labeled-data/frame_manifest.csv

The manifest carries the provenance of every frame -- animal, date, camera, epoch, trial, spout
position, trial category, phase, seconds from cue. A labelled frame whose provenance is unknown
cannot be audited later, and "which epochs is this network actually trained on?" is a question that
gets asked the first time tracking looks worse after the stroke.

CLI::

    python -m wfield_local.dlc_frames 20260907            # every session on one date
    python -m wfield_local.dlc_frames --cohort            # one session per animal x epoch
    python -m wfield_local.dlc_frames 20260907 --cam cam1 # a single camera
    python -m wfield_local.dlc_frames --cohort --dry-run  # what it would take, no decoding
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

MANIFEST_COLUMNS = ["animal", "date", "cam", "epoch", "video_stem", "frame", "trial_id",
                    "position", "category", "phase", "t_from_cue_s", "image"]


def _cfg() -> dict:
    return (config.defaults().get("dlc") or {})


def cameras() -> list[str]:
    return [str(c) for c in (_cfg().get("cameras") or ["cam4"])]


def bodyparts() -> list[str]:
    return [str(b) for b in (_cfg().get("bodyparts") or [])]


def phases() -> dict[str, float]:
    """``{phase name: seconds from the cue}``, ordered as declared in the YAML."""
    return {str(k): float(v) for k, v in (_cfg().get("frames", {}).get("phases") or {}).items()}


def per_session() -> int:
    return int(_cfg().get("frames", {}).get("per_session", 24))


def seed() -> int:
    return int(_cfg().get("frames", {}).get("seed", 92))


def out_root(rv=None) -> Path:
    """``<behavior_cameras>/dlc`` -- beside the recordings, like ``example_clips``."""
    rv = rv or PathResolver()
    return Path(rv.root("behavior_cameras")) / "dlc"


def _rng(animal: str, date: str, cam: str) -> np.random.Generator:
    """A generator whose stream depends only on the session, so cameras and dates are independent.

    Seeding once per run and drawing in iteration order would make every session's choice depend on
    which other sessions happened to be processed first -- so adding a date would silently re-pick
    frames for dates already labelled.
    """
    key = f"{animal}_{date}_{cam}_{seed()}".encode()
    return np.random.default_rng(np.frombuffer(key.ljust(32, b"\0")[:32], dtype=np.uint32))


def select_trials(trials: pd.DataFrame, n_cells: int, rng) -> pd.DataFrame:
    """One trial per spout position, rotating over trial CATEGORY so all three are represented.

    Positions are the design's own strata. Category (success / working / stopped, from
    ``behavior_clips.categorise``) is rotated rather than crossed: crossing would need 6 x 3 x
    n_phases frames per session, and the point of a small per-session budget is that breadth comes
    from epochs. Rotating still guarantees that failed and stopped trials -- where the mouth and
    tongue look least like the training distribution -- are in the labelled set.
    """
    from wfield_local.behavior_clips import CATEGORIES

    d = trials.copy()
    if "cat" not in d.columns:
        from wfield_local.behavior_clips import categorise
        d["cat"] = categorise(d)
    picks = []
    for i, pos in enumerate(sorted(d["pos_name"].dropna().astype(str).unique())):
        g = d[d["pos_name"].astype(str) == pos]
        if g.empty:
            continue
        # Rotate the PREFERRED category by position index; fall back through the rest so a session
        # missing a category still yields a frame for every position it ran.
        order = [CATEGORIES[(i + k) % len(CATEGORIES)] for k in range(len(CATEGORIES))]
        for cat in order:
            pool = g[g["cat"] == cat]
            if not pool.empty:
                picks.append(pool.iloc[int(rng.integers(len(pool)))])
                break
    out = pd.DataFrame(picks)
    return out.head(max(1, n_cells)) if not out.empty else out


def frame_of(tpl, cue_s: float, offset_s: float) -> int:
    """Camera frame index for ``offset_s`` from a cue at DAQ time ``cue_s``.

    The same affine ``behavior_clips`` cuts with: the alignment template maps DAQ samples to camera
    frames at ~1.2 ms residual, about a third of one frame at 250 fps.
    """
    fs, fps = float(tpl["fs_daq"]), float(tpl["fps_cam"])
    slope = float(tpl["slope_daqSample_per_camFrame"])
    icept = float(tpl["intercept_daqSample"])
    # Every term is coerced to a Python float first: `round` on a numpy float returns a numpy
    # float, which would flow into `cv2.CAP_PROP_POS_FRAMES` and the manifest as `2481.0`.
    return round((float(cue_s) * fs - icept) / slope + float(offset_s) * fps)


def plan_session(animal, date, sid, epoch, cam, rv=None) -> list[dict]:
    """The frames this session would contribute: one row per (trial, phase). No video is opened.

    Returns [] with a printed reason for any session that cannot be used, rather than raising --
    the cohort walk should report a missing template and carry on, not stop at the first gap.
    """
    rv = rv or PathResolver()
    tp = Path(rv.root("alignment_templates")) / cam / animal / (date + ".npz")
    if not tp.exists():
        print(f"[dlc_frames] {animal} {date} {cam}: no alignment template -> skip", flush=True)
        return []
    tpl = dict(np.load(tp, allow_pickle=True))
    if not bool(tpl["quality_ok"]):
        print(f"[dlc_frames] {animal} {date} {cam}: template quality_ok FALSE -> skip", flush=True)
        return []
    vids = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(cam + "_*.avi"))
    tpath = Path(rv.root("behavior_out")) / "sessions" / animal / date / (sid + "_trials.csv")
    if not vids or not tpath.exists():
        print(f"[dlc_frames] {animal} {date} {cam}: no video or trial table -> skip", flush=True)
        return []

    ph = phases()
    n_cells = max(1, per_session() // max(1, len(ph)))
    chosen = select_trials(pd.read_csv(tpath), n_cells, _rng(animal, date, cam))
    if chosen.empty:
        print(f"[dlc_frames] {animal} {date} {cam}: no usable trials -> skip", flush=True)
        return []

    stem, n_frames, rows = vids[0].stem, int(tpl["n_cam_frames"]), []
    for _, row in chosen.iterrows():
        for name, off in ph.items():
            f = frame_of(tpl, float(row["cue_s"]), off)
            if 0 <= f < n_frames:
                rows.append({"animal": animal, "date": date, "cam": cam, "epoch": epoch,
                             "video_stem": stem, "frame": f, "trial_id": int(row["trial_id"]),
                             "position": str(row["pos_name"]), "category": str(row["cat"]),
                             "phase": name, "t_from_cue_s": off,
                             "image": f"img{f:07d}.png", "_video": str(vids[0])})
    return rows


def extract(rows: list[dict], rv=None) -> int:
    """Decode and write the planned frames. Returns the number written.

    Seeks per frame rather than walking the file: a session contributes ~24 scattered frames out of
    1.5 M, so a sequential pass would decode 60000x more than it keeps. Rows are visited in
    ascending frame order so the seeks run forward through the stream.
    """
    import cv2

    written = 0
    for video, group in _by_video(rows):
        dest = out_root(rv) / "labeled-data" / group[0]["video_stem"]
        assert_writable(dest)
        dest.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))          # READ-ONLY, always
        if not cap.isOpened():
            print(f"[dlc_frames] could not open {video} -> skip", flush=True)
            continue
        try:
            for r in sorted(group, key=lambda x: x["frame"]):
                out = dest / r["image"]
                if out.exists():
                    continue                        # already extracted; never re-write over labels
                cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
                ok, frame = cap.read()
                if not ok:
                    print(f"[dlc_frames] {r['video_stem']} frame {r['frame']}: read failed",
                          flush=True)
                    continue
                cv2.imwrite(str(out), frame)        # PNG: lossless, per the module docstring
                written += 1
        finally:
            cap.release()
    return written


def _by_video(rows):
    """[(video path, rows)] grouped so each recording is opened once."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["_video"], []).append(r)
    return sorted(groups.items())


def write_manifest(rows: list[dict], rv=None, dest=None) -> Path:
    """Append to the manifest, de-duplicated on (video_stem, frame).

    APPEND, not overwrite: the labelling set is built up over several runs -- a date at a time, and
    a camera at a time -- and a manifest rewritten from one run's rows would drop the provenance of
    every frame extracted before it while the images stayed on disk.
    """
    dest = Path(dest) if dest else out_root(rv) / "labeled-data" / "frame_manifest.csv"
    assert_writable(dest.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if dest.exists():
        with open(dest, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
    seen = {(r["video_stem"], str(r["frame"])) for r in existing}
    fresh = [r for r in rows if (r["video_stem"], str(r["frame"])) not in seen]
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in existing + fresh:
            w.writerow({k: r.get(k, "") for k in MANIFEST_COLUMNS})
    return dest


def usable(animal: str, date: str, cams, rv=None) -> bool:
    """Does every requested camera have a passing alignment template for this session?

    ALL the requested cameras, not any: views that will be triangulated together have to be labelled
    on the same sessions, and a session with a template on one camera only cannot contribute a
    matched pair.
    """
    rv = rv or PathResolver()
    root = Path(rv.root("alignment_templates"))
    for cam in cams:
        tp = root / cam / animal / (date + ".npz")
        if not tp.exists():
            return False
        try:
            if not bool(dict(np.load(tp, allow_pickle=True))["quality_ok"]):
                return False
        except (OSError, KeyError, ValueError):
            return False
    return True


def _pick_middle(sessions, ok):
    """The most central session that ``ok`` accepts, searching outward from the middle.

    Plain "middle of the list" is what this replaces, and it was wrong in a way that hid itself: the
    camera alignment templates only start at 2026-06-06, while the `pre` epoch runs from 2026-05-27,
    so the median pre-stroke session for EVERY animal was an untemplated June date. The cohort walk
    duly skipped it and produced a labelling set with no pre-stroke frames at all -- the baseline the
    whole pre/post comparison rests on -- while reporting success. Falling back outward keeps the
    "representative of the epoch, not of its boundary" intent and still yields a session.
    """
    ordered = sorted(sessions)
    mid = len(ordered) // 2
    for i in sorted(range(len(ordered)), key=lambda j: (abs(j - mid), j)):
        if ok(ordered[i]):
            return ordered[i]
    return None


def cohort_sessions(rv=None, animals=None, cams=None) -> list[tuple[str, str, str, str]]:
    """One session per animal x epoch: ``[(animal, date, sid, epoch), ...]``.

    THE MIDDLE session of each epoch, not the first or last. A boundary session is the one most
    likely to be atypical of its epoch -- the first acute day is the animal at its worst, the last
    subacute day is nearly chronic -- and a labelling set should be representative of the epoch it
    is drawn from, not of the transition into it. See :func:`_pick_middle` for the usability
    fallback, which is not an optimisation.
    """
    from wfield_local import epochs

    rv = rv or PathResolver()
    cams = list(cams or cameras())
    base = Path(rv.root("behavior_out")) / "sessions"
    want = set(config.normalize_animals(animals) or [])
    by_epoch: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for an_dir in sorted(p for p in base.glob("PS*") if p.is_dir()):
        an = an_dir.name
        if want and an not in want:
            continue
        for sess in sorted(p for p in an_dir.iterdir() if p.is_dir()):
            date = sess.name
            ep = epochs.epoch_of(f"{an}_{date[4:]}")
            if ep is None:
                continue
            for t in sorted(sess.glob("*_trials.csv")):
                by_epoch.setdefault((an, ep), []).append((date, t.name[: -len("_trials.csv")]))
    out = []
    for (an, ep), sess in sorted(by_epoch.items()):
        pick = _pick_middle(sess, lambda ds, _an=an: usable(_an, ds[0], cams, rv))
        if pick is None:
            print(f"[dlc_frames] {an} {ep}: no session with templates on {'+'.join(cams)} -> "
                  f"THIS EPOCH WILL NOT BE LABELLED", flush=True)
            continue
        out.append((an, pick[0], pick[1], ep))
    return out


def date_sessions(date: str, rv=None, animals=None) -> list[tuple[str, str, str, str]]:
    """Every session on one date, as ``[(animal, date, sid, epoch), ...]``."""
    from wfield_local import epochs

    rv = rv or PathResolver()
    base = Path(rv.root("behavior_out")) / "sessions"
    want = set(config.normalize_animals(animals) or [])
    out = []
    for an_dir in sorted(p for p in base.glob("PS*") if p.is_dir()):
        an = an_dir.name
        if (want and an not in want) or not (an_dir / date).is_dir():
            continue
        ep = epochs.epoch_of(f"{an}_{date[4:]}")
        if ep is None:
            print(f"[dlc_frames] {an} {date}: no epoch -> skip", flush=True)
            continue
        for t in sorted((an_dir / date).glob("*_trials.csv")):
            out.append((an, date, t.name[: -len("_trials.csv")], ep))
    return out


def run(date=None, cohort=False, cams=None, rv=None, animals=None, dry=False) -> list[dict]:
    rv = rv or PathResolver()
    cams = list(cams or cameras())
    sessions = (cohort_sessions(rv, animals, cams) if cohort
                else date_sessions(date, rv, animals))
    if not sessions:
        print("[dlc_frames] no sessions matched", flush=True)
        return []
    rows = []
    for cam in (cams or cameras()):
        for animal, d, sid, epoch in sessions:
            rows += plan_session(animal, d, sid, epoch, cam, rv)
    if not rows:
        return rows
    span = sorted({(r["animal"], r["epoch"]) for r in rows})
    print(f"[dlc_frames] {len(rows)} frames from {len(sessions)} session(s), "
          f"{len({r['cam'] for r in rows})} camera(s), {len(span)} animal-epoch cell(s)", flush=True)
    if dry:
        for r in rows[:10]:
            print(f"  [dry] {r['animal']} {r['date']} {r['cam']} {r['epoch']} "
                  f"{r['position']}/{r['category']}/{r['phase']} frame {r['frame']}", flush=True)
        if len(rows) > 10:
            print(f"  [dry] ... and {len(rows) - 10} more", flush=True)
        return rows
    written = extract(rows, rv)
    manifest = write_manifest(rows, rv)
    print(f"[dlc_frames] wrote {written} new image(s); manifest -> {manifest}", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", metavar="YYYYMMDD", nargs="?", default=None)
    ap.add_argument("--cohort", action="store_true",
                    help="one session per animal x epoch instead of one date")
    ap.add_argument("--cam", action="append", default=None,
                    help="camera to extract (repeatable; default: dlc.cameras from defaults.yaml)")
    ap.add_argument("--animals", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    if not args.cohort and not args.date:
        ap.error("give a date or --cohort")
    rv = PathResolver(machine=args.machine)
    run(args.date, args.cohort, args.cam, rv, args.animals, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
