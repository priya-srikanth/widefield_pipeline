"""Side-view (cam2 + cam3) labelling plan: ONE shared model, cam3 flipped left-right, four parts.

    conda activate locanmf
    python -m wfield_local.dlc_sidecam pairs --dry-run   # cam2/cam3 frames at the cam4 round-4 moments
    python -m wfield_local.dlc_sidecam pairs             # extract + manifest + sync into the labelling project
    python -m wfield_local.dlc_context_frames cam2       # +-4 context around contact onset / end (and cam3)
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


def moment_key(r) -> tuple:
    """The same physical moment across cameras: session + trial + phase + time from cue (the side frames were
    sampled at cam4's moments, so these fields match across cam2 / cam3 / cam4)."""
    return (r["animal"], str(r["date"]), str(r["trial_id"]), str(r["phase"]), round(float(r["t_from_cue_s"]), 3))


def assign(man: pd.DataFrame, overlap_every: int = OVERLAP_EVERY) -> pd.DataFrame:
    """One row per side TARGET frame with ``label`` True (this camera labels it) / False. Per session x position,
    moments sorted by time alternate cam2 / cam3; every ``overlap_every``-th goes to both. Deterministic."""
    t = man[man.cam.isin(SIDE) & ~man.category.isin(NOT_TARGET)].copy()
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
    man = pd.read_csv(DF.staging_root(rv) / "frame_manifest.csv", dtype=str)
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


def cmd_assign(rv) -> Path:
    man = pd.read_csv(DF.staging_root(rv) / "frame_manifest.csv", dtype=str)
    a = assign(man)
    out = DF.staging_root(rv) / ASSIGN_NAME
    assert_writable(out.parent)
    a.to_csv(out, index=False)
    lab = a[a.label]
    print(f"{a.moment_id.nunique()} side target moments -> {len(lab)} frames to label "
          f"({int(lab.both.sum())} in both cameras) of {len(a)} side target frames on disk")
    print(lab.groupby(["cam", "category"]).size().unstack(fill_value=0).to_string())
    print(lab.groupby(["cam", "position"]).size().unstack(fill_value=0).to_string())
    print(f"-> {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["pairs", "assign"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rv = PathResolver()
    if a.step == "pairs":
        cmd_pairs(rv, a.dry_run)
    else:
        cmd_assign(rv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
