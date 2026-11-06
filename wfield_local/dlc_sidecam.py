"""Side-view (cam2 + cam3) labelling plan: ONE shared model, cam3 flipped left-right, four parts.

    conda activate locanmf
    python -m wfield_local.dlc_sidecam pairs --dry-run   # cam2/cam3 frames at the cam4 round-4 moments
    python -m wfield_local.dlc_sidecam pairs             # extract + manifest + sync into the labelling project
    python -m wfield_local.dlc_context_frames cam2       # +-4 context around contact onset / end (and cam3)
    python -m wfield_local.dlc_sidecam jawseq [--dry-run] # jaw closed -> open runs in the unlabelled side folders
    python -m wfield_local.dlc_sidecam assign            # which camera labels which moment -> side_assignment.csv
    python -m wfield_local.dlc_sidecam_guide             # the worksheet, SIDECAM_GUIDE.html

WHY (Priya, 2026-10-13). Tracking stays nose / jaw / tongue / spout. The side views show what cam4 cannot: the
jaw opening in profile with no spout rod across the chin (cam4's close positions make the jaw side-ambiguous --
DECISIONS 2026-10-13 evening), and how far forward / down the tongue reaches (depth, close vs far). cam3 flipped
left-right looks like cam2 (same facing, same treadmill layout; cam3 sits a little more frontal), so ONE model
is trained on both, cam3 flipped at TRAINING / INFERENCE time. Labelling is on the raw images as recorded.
The four parts are midline or single structures, so flipping needs no left / right label swap.

LABEL EACH MOMENT IN ONE SIDE CAMERA, NOT BOTH (Priya: "can we label fewer frames than doubling"). With a shared
flipped model a cam2 frame and a cam3 frame teach it the same thing, so each target moment is assigned to cam2 OR
cam3, alternating within each session x spout position (both cameras get every position, animal and epoch), and
every 5th moment to BOTH (`OVERLAP_EVERY`) -- same-moment pairs to check cam2 / cam3 agreement and for a later
multi-camera combination. ~60 % of the doubled count.

CARRIED OVER FROM cam4 / cam1 (DECISIONS 2026-09-12, 2026-09-30, 2026-10-05):
  * SAME MOMENTS AS cam4. The side folders were extracted with `dlc.frames.anchor_cam: cam4`, so every side
    target is a moment cam4 also has; `pairs` adds the cam4 round-4 moments the same way (`dlc_hard_frames.
    matched_frames`, by DAQ time through each camera's alignment template).
  * TARGET vs CONTEXT. Targets are labelled completely (blank = hidden); context frames are for scrubbing and stay
    blank, or are labelled completely -- never partly. All-blank rows never reach training.
  * Paired round-4 moments carry cam4's context bursts as plain context (Priya 2026-10-05, for cam1: "there for
    context and do NOT need to be labeled"); cam4's suggested neighbours are not targets here.

JAW SEQUENCES (`jawseq`, Priya 2026-11-04: "include some jaw-closed through jaw-open frames on cam2 & 3 (for better
closed-jaw labeling; it's helpful for her to be able to see the jaw opening before / after it's closed to track the
jaw keypoint)"). The round-1 side model is least confident on cam2 (sometimes cam3) with the mouth closed before the
cue. In every side session whose cam2 and cam3 folders hold NO labels yet, one trial per camera (different spout
positions) whose jaw was CLOSED at the cue and then OPENED (`jaw_onsets.csv`: jaw_dir open, not onset_at_cue, onset
60-1500 ms after the cue; close mid excluded, the chin is hidden there): a continuous run of `JAW_HALF` frames either
side of the measured opening onset (DAQ time through the camera's alignment template), targets at
`JAW_TARGET_FRAMES` (closed -48 / -16 ms, opening +16, open +48), the rest context to scroll. Category `JAW`, phase
``jaw<ms>ms`` from onset. Each sequence is labelled in the camera it was extracted for (no cam2 / cam3 alternation:
the other camera has no frames there), so `assign` lists JAW targets as they are and leaves every other row's
assignment exactly as before.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wfield_local import dlc_frames as DF
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

SIDE = ("cam2", "cam3")
ROUND = "side_round4"          # category of the side frames at cam4 round-4 moments
CONTEXT = "context"
NOT_TARGET = {CONTEXT, "context_label"}
OVERLAP_EVERY = 5              # every 5th moment (per session x position) is labelled in BOTH side cameras
ASSIGN_NAME = "side_assignment.csv"
JAW = "jaw_seq"                # category of the jaw closed -> open targets (module docstring)
#: The 16 Part A (pilot) folders: the only ones whose ROUTINE cue-locked targets are listed. Elsewhere routine targets
#: are DEFERRED (Priya 2026-11-04: round 2 = Part B only, "hopefully this is our last round"): Part A already covers
#: every animal and epoch with them; round 2 lists only the hard moments (ROUND) and the jaw sequences (JAW). Deferred
#: frames stay on disk and in the assignment (``deferred`` True) -- a held-out set, or back in if round 2 is weak.
PILOT_STEMS = frozenset(f"{c}_{t}" for c in SIDE for t in (
    "2026-06-06T12_25_18", "2026-06-06T14_09_26", "2026-06-06T18_02_39", "2026-08-20T14_08_45", "2026-08-20T16_31_02",
    "2026-08-21T15_02_47", "2026-09-03T10_54_01", "2026-09-07T17_08_48"))
JAW_HALF = 16                  # frames either side of the jaw opening onset (250 fps -> +-64 ms)
JAW_TARGET_FRAMES = (-12, -4, 4, 12)
JAW_ONSET_MS = (60.0, 1500.0)  # opening onset after the cue: a cue response, not a jaw already moving


def moment_key(r) -> tuple:
    """The same physical moment across cameras: session + trial + phase + time from cue (the side frames were
    sampled at cam4's moments, so these fields match across cam2 / cam3 / cam4)."""
    return (r["animal"], str(r["date"]), str(r["trial_id"]), str(r["phase"]), round(float(r["t_from_cue_s"]), 3))


def assign(man: pd.DataFrame, overlap_every: int = OVERLAP_EVERY) -> pd.DataFrame:
    """One row per side TARGET frame with ``label`` True (this camera labels it) / False. Per session x position,
    moments sorted by time alternate cam2 / cam3; every ``overlap_every``-th goes to both. Deterministic."""
    t = man[man.cam.isin(SIDE) & ~man.category.isin(NOT_TARGET | {JAW})].copy()
    t["moment"] = [moment_key(r) for r in t.to_dict("records")]
    rows = []
    turn = 0     # ONE alternation counter carried across groups: restarting it per group gave cam2 every odd group's
    #              extra moment (2026-10-13: 189 vs 143 frames)
    for (_a, _d, _pos), g in t.groupby(["animal", "date", "position"], sort=True):
        moments = sorted(set(g.moment), key=lambda m: (m[4], m[2], m[3]))
        for k, m in enumerate(moments):
            if k % overlap_every == overlap_every - 1:
                who = {"cam2", "cam3"}
            else:
                who = {SIDE[turn % 2]}
                turn += 1
            for r in g[g.moment == m].to_dict("records"):
                rows.append({**{c: r[c] for c in ("animal", "date", "cam", "epoch", "video_stem", "image", "frame",
                                                  "trial_id", "position", "category", "phase", "t_from_cue_s")},
                             "moment_id": f"{m[0]}_{m[1]}_t{m[2]}_{m[3]}_{m[4]:+.3f}",
                             "both": len(who) == 2, "label": r["cam"] in who})
    return pd.DataFrame(rows)


def pair_rows_side(rv=None) -> list[dict]:
    """cam2 / cam3 frames at every cam4 round-4 row (targets and their context), re-labelled for the side
    worksheet: round-4 picks -> ``side_round4`` (phase ``pair:<kind>``), everything else -> plain context."""
    from wfield_local import dlc_hard_frames as HF
    rv = rv or PathResolver()
    man = pd.read_csv(DF.manifest_path(rv), dtype=str)
    c4 = man[man.cam == "cam4"]
    stems = set(c4.loc[c4.category == "round4", "video_stem"])
    src = c4[c4.video_stem.isin(stems) & c4.category.isin({"round4", CONTEXT, "context_label"})]
    src = src.assign(frame=src.frame.astype(int)).to_dict("records")
    out = []
    for cam in SIDE:
        for r in HF.matched_frames(src, rv, "cam4", cam):
            if r["category"] == "round4":
                r.update(category=ROUND, phase=f"pair:{r['phase']}")
            else:
                r["category"] = CONTEXT
            out.append(r)
    return out


def cmd_pairs(rv, dry: bool) -> int:
    from wfield_local import dlc_project
    rows = pair_rows_side(rv)
    df = pd.DataFrame(rows)
    print(df.groupby(["cam", "category"]).size().unstack(fill_value=0).to_string())
    if dry:
        return 0
    todo = [r for r in rows if not (DF.staging_root(rv) / r["video_stem"] / r["image"]).exists()]
    n = DF.extract(todo, rv) if todo else 0
    DF.write_manifest([{k: v for k, v in r.items() if k != "_video"} for r in rows], rv)
    proj = dlc_project.project_dir(rv)
    assert_writable(proj / "labeled-data")
    folders, images, _ = dlc_project.sync_frames(proj, rv, cams=list(SIDE))
    print(f"pairs: extracted {n} PNGs, synced {images} images into {folders} side folders")
    return n


def jaw_assignment(man: pd.DataFrame) -> pd.DataFrame:
    """JAW targets as assignment rows: labelled in their own camera (``label`` True, ``both`` False)."""
    j = man[man.cam.isin(SIDE) & (man.category == JAW)]
    cols = ["animal", "date", "cam", "epoch", "video_stem", "image", "frame", "trial_id", "position", "category",
            "phase", "t_from_cue_s"]
    rows = [{**{c: r[c] for c in cols}, "moment_id": f"{r['animal']}_{r['date']}_t{r['trial_id']}_{r['phase']}",
             "both": False, "label": True} for r in j.to_dict("records")]
    return pd.DataFrame(rows, columns=[*cols, "moment_id", "both", "label"])


def jaw_trials(animal: str, date: str, onsets: pd.DataFrame) -> pd.DataFrame:
    """Trials whose jaw was closed at the cue and then opened (module docstring), one per spout position, in a
    deterministic shuffled order (seeded per session)."""
    import zlib
    o = onsets[(onsets.animal == animal) & (onsets.date.astype(str) == date)]
    o = o[(o.jaw_dir == "open") & ~o.onset_at_cue.astype(str).str.lower().eq("true") & (o.position != "close_center")
          & o.jaw_onset_ms.between(*JAW_ONSET_MS)]
    o = o.sample(frac=1.0, random_state=zlib.crc32(f"{animal}_{date}".encode()) % 2**32)
    return o.drop_duplicates("position")


def jaw_rows(rv=None) -> list[dict]:
    """Planned JAW sequence rows (targets + context) for every side session whose cam2 AND cam3 folders hold no
    labels yet. One trial per camera, at different spout positions."""
    import numpy as np

    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.dlc_context_frames import untouched_folders
    rv = rv or PathResolver()
    onsets = pd.read_csv(oo.root(rv) / oc.RUN / "imaging_encoding" / "qc" / "jaw_onsets.csv", dtype={"date": str})
    man = pd.read_csv(DF.manifest_path(rv), dtype=str)
    free = {c: set(untouched_folders(c, rv)) for c in SIDE}
    side = man[man.cam.isin(SIDE)].drop_duplicates("video_stem")
    out = []
    for (animal, date), g in side.groupby(["animal", "date"], sort=True):
        stems = {r.cam: r.video_stem for r in g.itertuples()}
        if set(stems) != set(SIDE) or not all(stems[c] in free[c] for c in SIDE):
            continue
        picks = jaw_trials(animal, date, onsets)
        tfiles = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob("*_trials.csv"))
        trials = pd.concat([pd.read_csv(t) for t in tfiles])
        cue = dict(zip(trials.trial_id.astype(int), trials.cue_s.astype(float)))
        epoch = g.epoch.iloc[0]
        k = 0
        for r in picks.itertuples():
            if k == len(SIDE):
                break
            if int(r.trial_id) not in cue:
                continue
            cam, stem = SIDE[k], stems[SIDE[k]]
            tpl = dict(np.load(Path(rv.root("alignment_templates")) / cam / animal / f"{date}.npz", allow_pickle=True))
            t_on = cue[int(r.trial_id)] + float(r.jaw_onset_ms) / 1000.0
            f0, n = DF.frame_of(tpl, t_on, 0.0), int(tpl["n_cam_frames"])
            if f0 - JAW_HALF < 0 or f0 + JAW_HALF >= n:
                continue
            for d in range(-JAW_HALF, JAW_HALF + 1):
                ms = round(d * 1000.0 / float(tpl["fps_cam"]))
                out.append({"animal": animal, "date": date, "cam": cam, "epoch": epoch, "video_stem": stem,
                            "frame": f0 + d, "trial_id": int(r.trial_id), "position": r.position,
                            "category": JAW if d in JAW_TARGET_FRAMES else CONTEXT, "phase": f"jaw{ms:+d}ms",
                            "t_from_cue_s": round(float(r.jaw_onset_ms) / 1000.0 + ms / 1000.0, 4),
                            "image": f"img{f0 + d:07d}.png"})
            k += 1
        print(f"{animal} {date}: {k} jaw sequences ({len(picks)} eligible positions)", flush=True)
    return out


def cmd_jawseq(rv, dry: bool) -> int:
    from wfield_local import dlc_project
    rows = jaw_rows(rv)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no jaw sequences planned")
        return 0
    print(df.groupby(["cam", "category"]).size().unstack(fill_value=0).to_string())
    if dry:
        return 0
    man = pd.read_csv(DF.manifest_path(rv), dtype=str)
    have = set(zip(man.video_stem, man.frame.astype(int)))
    rows = [r for r in rows if (r["video_stem"], int(r["frame"])) not in have]   # an existing frame keeps its role
    recs = [{**r, "_video": str(sorted((Path(rv.root("behavior_cameras")) / r["date"] / r["animal"])
                                        .glob(f"{r['video_stem']}.avi"))[0])} for r in rows]
    n = DF.extract(recs, rv)
    DF.write_manifest(rows, rv)
    proj = dlc_project.project_dir(rv)
    assert_writable(proj / "labeled-data")
    folders, images, _ = dlc_project.sync_frames(proj, rv, cams=list(SIDE))
    print(f"jawseq: extracted {n} PNGs, synced {images} images into {folders} side folders")
    return n


def cmd_assign(rv) -> Path:
    man = pd.read_csv(DF.manifest_path(rv), dtype=str)
    a = pd.concat([assign(man), jaw_assignment(man)], ignore_index=True)
    a["deferred"] = ~a.category.isin({ROUND, JAW}) & ~a.video_stem.isin(PILOT_STEMS)
    out = DF.out_root(rv) / ASSIGN_NAME
    assert_writable(out.parent)
    a.to_csv(out, index=False)
    lab = a[a.label & ~a.deferred]
    print(f"deferred (routine targets outside Part A, not listed): {int((a.label & a.deferred).sum())} frames")
    print(f"{a.moment_id.nunique()} side target moments -> {len(lab)} frames to label "
          f"({int(lab.both.sum())} in both cameras) of {len(a)} side target frames on disk")
    print(lab.groupby(["cam", "category"]).size().unstack(fill_value=0).to_string())
    print(lab.groupby(["cam", "position"]).size().unstack(fill_value=0).to_string())
    print(f"-> {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["pairs", "jawseq", "assign"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rv = PathResolver()
    if a.step == "pairs":
        cmd_pairs(rv, a.dry_run)
    elif a.step == "jawseq":
        cmd_jawseq(rv, a.dry_run)
    else:
        cmd_assign(rv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
