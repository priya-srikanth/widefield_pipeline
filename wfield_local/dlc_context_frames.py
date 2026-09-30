"""CONTEXT frames: consecutive neighbours extracted around a labelling TARGET, to scrub through, not to label.

    conda activate locanmf
    python -m wfield_local.dlc_context_frames cam1 --dry-run     # what would be added, per folder
    python -m wfield_local.dlc_context_frames cam1               # extract + manifest + sync into the project
    python -m wfield_local.dlc_context_frames cam1 --promote 3   # label every ~3rd context frame too
    python -m wfield_local.dlc_cam1_guide                        # refresh the worksheet (lists targets)

WHY (Priya, 2026-09-30). Where the tongue tip is, or whether the jaw is showing, is often not decidable
from one frame -- but it is from a few frames either side, scrolling back and forth. The worst moments
are the START and END of a lick: the tongue emerging toward the spout and retracting from it. So the
context goes around spout CONTACT onset and contact END, measured per lick from the DAQ lick sensor,
not at a fixed offset (contact lasts ~65-95 ms pre-stroke and is expected to change post-stroke).

TARGET vs CONTEXT -- the rule everything else follows from:
  * a TARGET is a frame the labeller must label completely (every visible part; blank = occluded);
  * a CONTEXT frame is there to be looked at. Leave it blank -- or, if it is worth it, label it
    COMPLETELY like a target. Never partly: a frame with ANY label has its blanks read as occluded,
    by DLC always and by Lightning Pose under `uniform_heatmaps_for_nan_keypoints`.
  * An all-blank frame never reaches training: `dlc_train.drop_unlabelled` removes it before DLC
    sees it, and the Lightning Pose export must do the same (single-view) or mark it "not labelled"
    (`visible`=0, multi-view). See DECISIONS.md, 2026-09-30, "target vs context frames".

Context rows go into `frame_manifest.csv` with ``category == "context"`` and a phase naming the anchor
(``ctx_on-4`` = 4 frames before contact onset, ``ctx_off+2`` = 2 after contact end). The manifest is
the ONLY record of which frames are targets; the worksheet reads it. Filenames stay ``img%07d.png`` so
a burst sorts contiguously in napari's slider and so Lightning Pose's context model (which reads
frames t-2..t+2 by index) could use the same folders.

SCOPE: cam1 folders with NO labels yet (the student's untouched folders). Folders already labelled are
left exactly as they are. Existing targets are never moved -- they are the same DAQ instants as the
cam4 labelled frames, which is what makes the two views pairable (nose offset, multi-view LP).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local import dlc_frames as DF
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

CONTEXT = "context"
#: A context frame PROMOTED to a target (Priya, 2026-09-30: "include some labelling of the context
#: frames ... maybe every 3rd frame"). Kept distinct from the original targets so the provenance
#: survives; everything downstream (worksheet, training) treats it as a target -- label completely.
CONTEXT_LABEL = "context_label"
#: Promote a context frame when it is at least this many frames (12 ms) from every labelled frame.
PROMOTE_SPACING = 3
#: Frames either side of contact onset / contact end (250 fps -> +-16 ms).
HALF_LICK = 4
#: A manifest lick+0 target is matched to the DAQ onset nearest its frame time, within this.
MATCH_TOL_S = 0.010


# --------------------------------------------------------------------------- the rule (pure)

def burst(center: int, half: int, n_frames: int | None = None) -> list[int]:
    """center-half .. center+half, clipped to [0, n_frames)."""
    lo = max(0, int(center) - half)
    hi = int(center) + half + 1
    if n_frames is not None:
        hi = min(hi, int(n_frames))
    return list(range(lo, hi))


def pair_contacts(onsets, offsets) -> np.ndarray:
    """(N, 2) [onset, end] for each onset: the first offset after it, and before the next onset.

    An onset with no such offset (recording ends, or a glitch) gets NaN -- its end burst is skipped,
    not guessed.
    """
    on = np.asarray(onsets, float)
    off = np.sort(np.asarray(offsets, float))
    out = np.full((on.size, 2), np.nan)
    out[:, 0] = on
    nxt = np.r_[on[1:], np.inf]
    for i, t in enumerate(on):
        j = np.searchsorted(off, t, side="right")
        if j < off.size and off[j] < nxt[i]:
            out[i, 1] = off[j]
    return out


def contact_context(targets: pd.DataFrame, tpl: dict, contacts_s: np.ndarray, half: int = HALF_LICK) -> list[dict]:
    """Context rows for one folder.

    ``targets`` = that folder's manifest rows. For every lick target at contact onset (phase
    ``lick+0``), bursts go around the measured onset and the measured contact end; frames that are
    already targets are left out (they stay targets).
    """
    fs = float(tpl["fs_daq"])
    slope, icept = float(tpl["slope_daqSample_per_camFrame"]), float(tpl["intercept_daqSample"])
    n_frames = int(tpl["n_cam_frames"])
    have = set(targets["frame"].astype(int))
    rows: list[dict] = []
    seen: set[int] = set()
    for _, r in targets[targets["phase"].astype(str) == "lick+0"].iterrows():
        f0 = int(r["frame"])
        t0 = (f0 * slope + icept) / fs
        cue = t0 - float(r["t_from_cue_s"])
        k = int(np.argmin(np.abs(contacts_s[:, 0] - t0))) if len(contacts_s) else -1
        if k < 0 or abs(contacts_s[k, 0] - t0) > MATCH_TOL_S:
            print(f"[dlc_context_frames] {r['video_stem']} frame {f0}: no DAQ onset within "
                  f"{MATCH_TOL_S * 1000:.0f} ms -> skipped", flush=True)
            continue
        anchors = [("on", DF.frame_of(tpl, contacts_s[k, 0], 0.0))]
        if np.isfinite(contacts_s[k, 1]):
            anchors.append(("off", DF.frame_of(tpl, contacts_s[k, 1], 0.0)))
        for tag, fa in anchors:
            for f in burst(fa, half, n_frames):
                if f in have or f in seen:
                    continue
                seen.add(f)
                rows.append({**{c: r[c] for c in ("animal", "date", "cam", "epoch", "video_stem",
                                                   "trial_id", "position")},
                             "frame": int(f), "category": CONTEXT, "phase": f"ctx_{tag}{f - fa:+d}",
                             "t_from_cue_s": round((f * slope + icept) / fs - cue, 4),
                             "image": f"img{int(f):07d}.png"})
    return rows


def promote(targets, context, spacing: int = PROMOTE_SPACING) -> list[int]:
    """Context frames to label: walk them in order, keep one at least ``spacing`` frames from every
    frame already labelled (targets and earlier promotions).

    "Every 3rd frame" measured from the LABELLED frames rather than counted along the context list, so
    a promoted frame is never adjacent to an existing target (-4 and 0 are targets around each contact
    onset; counting along the list would label -3 next to -4 and add nothing).
    """
    labelled = sorted(int(t) for t in targets)
    out: list[int] = []
    for f in sorted(int(c) for c in context):
        if all(abs(f - t) >= spacing for t in labelled):
            out.append(f)
            labelled.append(f)
    return out


def promote_in_manifest(man_path: Path, stems, spacing: int = PROMOTE_SPACING, dry: bool = False) -> pd.DataFrame:
    """Re-mark the chosen context rows of ``stems`` as `CONTEXT_LABEL`. Returns the promoted rows.

    Idempotent: already-promoted rows count as labelled, so a second run adds nothing.
    """
    assert_writable(man_path.parent)
    man = pd.read_csv(man_path, dtype=str)
    hit = []
    for stem in stems:
        m = man[man.video_stem == stem]
        tg = m.loc[m.category != CONTEXT, "frame"].astype(int)
        cx = m.loc[m.category == CONTEXT, "frame"].astype(int)
        for f in promote(tg, cx, spacing):
            hit.append(m.index[(m.frame.astype(int) == f) & (m.category == CONTEXT)][0])
    if hit and not dry:
        man.loc[hit, "category"] = CONTEXT_LABEL
        man.to_csv(man_path, index=False)
    return man.loc[hit]


# --------------------------------------------------------------------------- DAQ (impure)

def contacts(animal: str, date: str, rv=None) -> np.ndarray:
    """(N, 2) [onset_s, end_s] spout contacts on the DAQ clock, from the pipeline's own detector.

    Same detector and parameters as every other lick consumer (`lick_detection.detect_licks` under
    `defaults.yaml lick_detection`); the onsets are the cleaned ones `daq_trials.decode` returns, and
    the ends are the detector's own downward crossings, which `decode` computes and discards.
    """
    from wfield_local import daq_io
    from wfield_local.lick_detection import detect_licks

    rv = rv or PathResolver()
    root = Path(rv.root("daq_recorder_output")) / date
    cands = sorted(root.glob(f"{animal}_{date}_*.h5"))
    if not cands:
        return np.empty((0, 2))
    p = config.defaults()["lick_detection"]
    with daq_io.open_daq(cands[0]) as f:
        fs, _ = daq_io.session_attrs(f)
        lick_v = daq_io.analog_channel(f, "lick_analog", required=False)
    if lick_v is None:
        return np.empty((0, 2))
    det = detect_licks(lick_v, fs, thresh_upper=p["thresh_upper"], thresh_lower=p["thresh_lower"],
                       lockout_s=tuple(p["lockout_falling_edge_s"]),
                       min_ili_s=p.get("min_ili_ms", 40) / 1000.0)
    return pair_contacts(det["lick_onsets"] / fs, det["offsets"] / fs)


def untouched_folders(cam: str, rv=None) -> list[str]:
    """Video stems whose live labelling folder has no CollectedData yet."""
    from wfield_local import dlc_project

    live = dlc_project.project_dir(rv) / "labeled-data"
    return sorted(p.name for p in live.glob(f"{cam}_*") if p.is_dir() and not any(p.glob("CollectedData_*")))


def plan(cam: str, rv=None, half: int = HALF_LICK) -> list[dict]:
    rv = rv or PathResolver()
    man = pd.read_csv(DF.staging_root(rv) / "frame_manifest.csv", dtype={"date": str})
    rows: list[dict] = []
    for stem in untouched_folders(cam, rv):
        t = man[(man.video_stem == stem) & (man.category != CONTEXT)]
        if t.empty:
            print(f"[dlc_context_frames] {stem}: not in the manifest -> skipped", flush=True)
            continue
        animal, date = t.animal.iloc[0], str(t.date.iloc[0])
        tpl = dict(np.load(Path(rv.root("alignment_templates")) / cam / animal / f"{date}.npz", allow_pickle=True))
        got = contact_context(t, tpl, contacts(animal, date, rv), half)
        n_licks = int((t.phase.astype(str) == "lick+0").sum())
        print(f"{stem}: {n_licks} licks -> {len(got)} context frames", flush=True)
        rows += got
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cam", choices=["cam1"])
    ap.add_argument("--half", type=int, default=HALF_LICK)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--promote", type=int, default=None, metavar="SPACING",
                    help="instead of extracting: mark context frames >= SPACING frames from every labelled "
                         "frame as targets (context_label), in folders with no labels yet")
    a = ap.parse_args(argv)
    rv = PathResolver()
    if a.promote:
        got = promote_in_manifest(DF.staging_root(rv) / "frame_manifest.csv", untouched_folders(a.cam, rv),
                                  a.promote, dry=a.dry_run)
        print(got.groupby("video_stem").size().to_string() if len(got) else "nothing to promote")
        print(f"\n{len(got)} context frames {'would be' if a.dry_run else ''} promoted to targets")
        return 0
    rows = plan(a.cam, rv, a.half)
    print(f"\n{len(rows)} context frames planned")
    if a.dry_run or not rows:
        return 0
    from wfield_local import dlc_project

    recs = [{**r, "_video": str(sorted((Path(rv.root("behavior_cameras")) / r["date"] / r["animal"])
                                        .glob(f"{r['video_stem']}.avi"))[0])} for r in rows]
    n = DF.extract(recs, rv)
    m = DF.write_manifest(rows, rv)
    proj = dlc_project.project_dir(rv)
    assert_writable(proj / "labeled-data")
    folders, images, _ = dlc_project.sync_frames(proj, rv, cams=[a.cam])
    print(f"extracted {n} PNGs; manifest -> {m}; synced {images} new images into {folders} {a.cam} folders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
