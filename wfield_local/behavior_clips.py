"""Annotated cam4 clips around the cue, for lab-meeting review.

Cuts 1 s before the cue to 3.5 s after it, played at 0.25x, with the trial's phase named on every
frame: ENL, Cue, Response. Written under
``<behavior_cameras>/example_clips/<animal>/<epoch>/<date>/``.

SOURCE VIDEOS ARE OPENED READ-ONLY AND NEVER WRITTEN. ``cv2.VideoCapture`` is the only thing that
touches them, and ``assert_writable`` guards the output directory, so a path bug lands as a refusal
rather than as an edit to an irreplaceable recording.

FRAME NUMBERS COME FROM THE ALIGNMENT TEMPLATE, never from a wall-clock guess. The trial table's
``cue_s`` IS the DAQ cue rising edge -- checked against the DAQ over 600 trials at a difference of
0.0000 s -- and ``camera_sync``'s affine maps DAQ samples to camera frames with ~1.2 ms residual,
about a third of one frame at 250 fps. The phase labels use that same map, so what the label says is
what the frame shows. A session whose template is missing or fails ``quality_ok`` is SKIPPED rather
than cut with a guessed offset: the 8/18 folder swap was found precisely because that flag refused.

THE WINDOW IS 3.5 s POST-CUE, NOT 3.0. The response window is ``timing.response_window`` = 3500 ms
in every session's gui_config.json, so a 3 s clip would end half a second before the window the
animal is scored on and would cut off late licks.

THE DENOMINATOR IS IN THE FILENAME (``far_R_working_1of1_trial0015.avi``). A cell holding one
qualifying trial and one sampled from a hundred otherwise produce indistinguishable files, and the
rare one is the likeliest to be picked up and shown. A manifest alongside can be separated from the
clip; a filename cannot.
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

CAM = "cam4"                     #: the front-facing camera (Priya, 2026-09-07)
PRE_S, POST_S = 1.0, 3.5
CUE_S = 0.100                    #: measured DAQ cue pulse width, 100.2 ms
OUT_PX, OUT_FPS, KEEP_EVERY = 480, 30.0, 2
PER_CELL = 5
CATEGORIES = ("success", "working", "stopped")
COL = {"ITI": (150, 150, 150), "ENL": (235, 200, 120), "Cue": (80, 220, 255),
       "Response": (120, 255, 120)}


def phase_at(t_rel, enl_len):
    """Phase name at ``t_rel`` seconds from cue, for a trial whose ENL ran ``enl_len`` seconds."""
    if t_rel < 0:
        return "ENL" if t_rel >= -enl_len else "ITI"
    return "Cue" if t_rel < CUE_S else "Response"


def categorise(d):
    """success / working / stopped per trial.

    ENGAGEMENT IS RECOMPUTED rather than read from the table's ``engaged`` column: the gate was
    backdated on 2026-09-07 to start at the first miss of the run that trips it, and a persisted
    trials.csv may predate that. Calling the function keeps one source of truth and makes the clips
    correct whether or not the behaviour analysis has been re-run yet.
    """
    from wfield_local.spout_behavior import reference_engagement

    hit = d["hit"].fillna(0).astype(int) == 1
    resp = d["responded"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    not_eng, _ = reference_engagement(resp, d["pos_name"].to_numpy().astype(str))
    eng = pd.Series(~not_eng, index=d.index)
    cat = pd.Series("stopped", index=d.index)
    cat[~hit & eng] = "working"
    cat[hit] = "success"
    return cat


def _clip(cap, tpl, row, dest, meta):
    """One clip. Returns frames written, or None if the window runs off the recording."""
    import cv2

    fs, fps = float(tpl["fs_daq"]), float(tpl["fps_cam"])
    slope = float(tpl["slope_daqSample_per_camFrame"])
    icept = float(tpl["intercept_daqSample"])
    cue_frame = (row["cue_s"] * fs - icept) / slope
    enl_len = float(row["cue_s"] - row["trial_start_s"])
    f0 = int(round(cue_frame - PRE_S * fps))
    n = int(round((PRE_S + POST_S) * fps))
    if f0 < 0 or f0 + n >= int(tpl["n_cam_frames"]):
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    vw, wrote = None, 0
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        if i % KEEP_EVERY:
            continue
        t_rel = (i / fps) - PRE_S
        f2 = cv2.resize(fr, (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
        ph = phase_at(t_rel, enl_len)
        cv2.putText(f2, ph, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, COL[ph], 2, cv2.LINE_AA)
        # BELOW the timeline, never through it: at 0.44 scale this line is ~10px tall and its
        # baseline sat inside the 5px bar, so the rule ran through the text.
        cv2.putText(f2, meta, (12, OUT_PX - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (215, 215, 215), 1, cv2.LINE_AA)
        cv2.putText(f2, "%+.2fs" % t_rel, (12, OUT_PX - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        x0, x1, y = 12, OUT_PX - 12, OUT_PX - 62
        for a, b, key in ((-PRE_S, 0.0, "ENL"), (0.0, CUE_S, "Cue"), (CUE_S, POST_S, "Response")):
            xa = int(x0 + (x1 - x0) * (a + PRE_S) / (PRE_S + POST_S))
            xb = int(x0 + (x1 - x0) * (b + PRE_S) / (PRE_S + POST_S))
            cv2.rectangle(f2, (xa, y), (max(xb, xa + 1), y + 5), COL[key], -1)
        xm = int(x0 + (x1 - x0) * (t_rel + PRE_S) / (PRE_S + POST_S))
        cv2.line(f2, (xm, y - 4), (xm, y + 9), (255, 255, 255), 1)
        if vw is None:
            vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS,
                                 (OUT_PX, OUT_PX))
        vw.write(f2)
        wrote += 1
    if vw is not None:
        vw.release()
    return wrote or None


def out_root(rv=None):
    """``<behavior_cameras>/example_clips`` -- a deliverable, beside the recordings it is cut from."""
    rv = rv or PathResolver()
    return Path(rv.root("behavior_cameras")) / "example_clips"


def _cell_counts(d, per_cell):
    """[(position, category, available, take), ...] for one session's trial table.

    A FLAT CAP HERE ON PURPOSE (Priya, 2026-09-07: "don't change the clip generation, just the
    deck"). What lands on disk is the LIBRARY -- five of every cell, so any later question can be
    answered from it -- and the deck is a curated VIEW that shows fewer. Pushing the curation down
    here would mean re-cutting video every time the presentation changes its mind.
    """
    out = []
    for pos, g in d.groupby("pos_name"):
        for cat in CATEGORIES:
            avail = int((g["cat"] == cat).sum())
            out.append((pos, cat, avail, min(avail, per_cell)))
    return out


def session_clips(animal, date, sid, epoch, rv=None, per_cell=PER_CELL, dry=False,
                  categories=None):
    """Clips for one session. Returns the number written (or that would be, under ``dry``).

    ``categories`` restricts which classes are cut. It exists so a cap change can be applied by
    re-cutting only the classes it touched: `success` is capped the same way before and after, so
    re-cutting it would spend an hour reproducing byte-identical files under identical names.
    """
    import cv2

    rv = rv or PathResolver()
    tp = Path(rv.root("alignment_templates")) / CAM / animal / (date + ".npz")
    if not tp.exists():
        print("[behavior_clips] %s %s: no %s template -> skip" % (animal, date, CAM), flush=True)
        return 0
    tpl = dict(np.load(tp, allow_pickle=True))
    if not bool(tpl["quality_ok"]):
        print("[behavior_clips] %s %s: template quality_ok FALSE -> skip" % (animal, date),
              flush=True)
        return 0
    vids = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(CAM + "_*.avi"))
    trials = Path(rv.root("behavior_out")) / "sessions" / animal / date / (sid + "_trials.csv")
    if not vids or not trials.exists():
        print("[behavior_clips] %s %s: no video or trial table -> skip" % (animal, date),
              flush=True)
        return 0
    d = pd.read_csv(trials)
    d["cat"] = categorise(d)
    dest_dir = out_root(rv) / animal / epoch / date
    cells = _cell_counts(d, per_cell)
    if categories:
        cells = [c for c in cells if c[1] in set(categories)]
    if dry:
        n = sum(t for _p, _c, _a, t in cells)
        print("[dry-run] %s %s (%s): would write %d clips -> %s"
              % (animal, date, epoch, n, dest_dir), flush=True)
        return n
    assert_writable(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(vids[0]))            # READ-ONLY, always
    made = 0
    try:
        for pos, cat, avail, take in cells:
            pool = d[(d["pos_name"] == pos) & (d["cat"] == cat)]
            for _, row in pool.head(take).iterrows():
                nm = "%s_%s_%dof%d_trial%04d.avi" % (pos, cat, take, avail, int(row["trial_id"]))
                meta = "%s %s %s | %s | %s | trial %d" % (animal, date, epoch, pos, cat,
                                                          int(row["trial_id"]))
                if _clip(cap, tpl, row, dest_dir / nm, meta):
                    made += 1
    finally:
        cap.release()
    print("[behavior_clips] %s %s (%s): %d clips -> %s" % (animal, date, epoch, made, dest_dir),
          flush=True)
    return made


def refile_stale_epochs(rv=None, dry=False):
    """Move clip folders whose EPOCH has moved since they were cut. Returns the moves made.

    THE DIRECTORY NAME IS A CLAIM, AND `epochs` RE-DERIVES THE TRUTH EACH RUN. `chronic_from` is
    computed from behaviour and republished nightly, so a boundary that shifts silently restales
    every clip cut before it: three PS92 sessions were sitting under `subacute/` on 2026-09-07 after
    the animal's chronic boundary landed, and the deck was labelling them from the folder.

    Cheap to do every night -- a rename, no video touched -- and it is the difference between a
    label that is checked and a label that was true once.
    """
    from wfield_local import epochs

    rv = rv or PathResolver()
    root = out_root(rv)
    if not root.is_dir():
        return []
    moves = []
    for an_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("PS")):
        for ep_dir in sorted(p for p in an_dir.iterdir()
                             if p.is_dir() and not p.name.startswith(".")):
            for d_dir in sorted(p for p in ep_dir.iterdir() if p.is_dir()):
                now = epochs.epoch_of("%s_%s" % (an_dir.name, d_dir.name[4:]))
                if now is None or now == ep_dir.name:
                    continue
                dest = an_dir / now / d_dir.name
                moves.append((str(d_dir), str(dest)))
                print("[behavior_clips] re-file %s %s: %s -> %s"
                      % (an_dir.name, d_dir.name, ep_dir.name, now), flush=True)
                if dry:
                    continue
                assert_writable(dest.parent)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    # both present: keep the newer cut rather than merging two epochs' worth
                    for f in d_dir.glob("*.avi"):
                        f.replace(dest / f.name)
                    try:
                        d_dir.rmdir()
                    except OSError:
                        pass
                else:
                    d_dir.replace(dest)
    return moves


def run(date, rv=None, animals=None, per_cell=PER_CELL, dry=False, categories=None) -> int:
    """Clips for every session on ``date``.

    INCREMENTAL BY DESIGN -- one date per night, roughly five minutes. Re-cutting all 44 sessions
    nightly would cost two to three hours to reproduce clips that have not changed.
    """
    from wfield_local import epochs

    rv = rv or PathResolver()
    refile_stale_epochs(rv, dry=dry)   # a boundary may have moved since the last cut
    base = Path(rv.root("behavior_out")) / "sessions"
    want = set(config.normalize_animals(animals) or [])
    total = 0
    for an_dir in sorted(p for p in base.glob("PS*") if p.is_dir()):
        an = an_dir.name
        if want and an not in want:
            continue
        sess = an_dir / date
        if not sess.is_dir():
            continue
        ep = epochs.epoch_of("%s_%s" % (an, date[4:]))
        if ep is None:
            # excluded (the 8/17 lesion attempt) or not yet phased -- a clip filed under no epoch
            # would be a clip nobody can interpret.
            print("[behavior_clips] %s %s: no epoch -> skip" % (an, date), flush=True)
            continue
        for t in sorted(sess.glob("*_trials.csv")):
            sid = t.name[: -len("_trials.csv")]
            total += session_clips(an, date, sid, ep, rv=rv, per_cell=per_cell, dry=dry,
                                   categories=categories)
    print("[behavior_clips] %s: %d clips" % (date, total), flush=True)
    return total


def write_manifest(rv=None, dest=None):
    """One row per clip: what it is, and how many trials the cell it was drawn from held."""
    rv = rv or PathResolver()
    root = out_root(rv)
    dest = Path(dest) if dest else root / "clip_manifest.csv"
    rows = []
    for p in sorted(root.rglob("*.avi")):
        try:
            an, epoch, date = p.parts[-4], p.parts[-3], p.parts[-2]
            stem = p.stem.rsplit("_trial", 1)[0]
            pos, cat, frac = stem.rsplit("_", 2)
            take, _, avail = frac.partition("of")
            rows.append(dict(animal=an, epoch=epoch, date=date, position=pos, category=cat,
                             clips_made=int(take), trials_available=int(avail), file=p.name))
        except (IndexError, ValueError):
            continue                       # not one of ours; leave it alone rather than guess
    if not rows:
        return None
    assert_writable(dest.parent)
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("[behavior_clips] manifest: %d clips -> %s" % (len(rows), dest), flush=True)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", nargs="?", metavar="YYYYMMDD")
    ap.add_argument("--only", nargs="+", metavar="ANIMAL", help="restrict to these animals")
    ap.add_argument("--per-cell", type=int, default=PER_CELL,
                    help="clips per position x category (default %d)" % PER_CELL)
    ap.add_argument("--manifest", action="store_true", help="rewrite the manifest and exit")
    ap.add_argument("--categories", nargs="+", choices=list(CATEGORIES),
                    help="restrict to these trial classes (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--machine", default=None)
    a = ap.parse_args(argv)
    rv = PathResolver(machine=a.machine)
    if a.manifest:
        write_manifest(rv)
        return 0
    if not a.date:
        ap.error("a date is required unless --manifest is given")
    run(a.date, rv, animals=a.only, per_cell=a.per_cell, dry=a.dry_run,
        categories=a.categories)
    if not a.dry_run:
        write_manifest(rv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
