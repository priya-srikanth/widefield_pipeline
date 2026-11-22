"""Licks across cameras (`wfield_local.camera_licks`; Priya 11-18): what cam2 / cam3 add to the cam4 + cam1 lick train,
on the side REVIEW CLIPS now and on WHOLE SESSIONS once the side O2 inference exists. DRAFT (decisions D2 / D3 of
docs/status/STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md are open).

    conda activate dlc            # aniposelib (`pose3d_side.fuse_session` poses the cameras)
    python -m scripts.camera_licks clips [ANIMAL:YYYYMMDD ...] [--preds <video_preds dir>] [--jobs 3]
    python -m scripts.camera_licks session ANIMAL:YYYYMMDD [...] --bundle <side O2 bundle> [--jobs 3]

``session``: the same steps on the whole session (`pose3d_side.fuse_session(side_bundle=...)`: every cam1 + cam4 frame
kept, a side camera without that frame counts as not seeing it; production events NOT restricted). Writes
`session_poses/camera_licks/<bundle>/`. Untested until a whole-session side bundle exists (11-18: the r2b local test is
still running).

Per session with side predictions in ``--preds`` (default: the round-2 side model r2b on the review clips, `CLIP_SETS`;
cue -0.2 .. +1.5 s, 2 trials per position): `pose3d_side.fuse_session` (side tongue cleaned, side droplets removed), side events
(`camera_licks.side_events`), the production cam4 / cam1 events restricted to the same clip frames
(`camera_licks.within_frames`), fused (`camera_licks.fuse`). Writes `session_poses/camera_licks/clips_<model>/`:
``<a>_<d>_fused.csv`` (one row per lick), ``<a>_<d>_events.csv`` (every per-camera event), ``views.csv`` (licks per
camera combination per session) and prints the pooled view table. Fused licks are split with `camera_licks.event_cuts`
(anchor cameras + the 3-D tongue's return; 11-21 pilot). `camera_licks.lick_train` applies D2 (revised 11-21: any
camera, cam1-only >= 300 ms dropped) downstream.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PREDS = "C:/Users/SabatiniLab/lp_stage/lp_side_r2b_20261118/r2_full/models/side_r2b_r2_full/video_preds"
#: side review clip folders under the dlc root (the 10-27 pilot set, `pose3d_side_check.CLIPS`, and the 11-04 Part A set);
#: a session's clips are the first folder holding ``<a>_<d>/index.csv``
CLIP_SETS = ("inference_check_20261027_side_pilot/clips", "inference_check_20261104_partA_clips")
#: 11-22: was "%.4g" -- 4 significant digits kept 10 ms at 10-100 s from the cue (ITI licks) and 1 s of daq_on_s after
#: 1000 s, so the 11-21 fused / events tables cannot be key-joined or put on the DAQ clock. Folders written since carry
#: `session_licks.MANIFEST` (format 2); `session_licks.session_licks` refuses a fused folder without it.
FLOAT_FORMAT = "%.10g"


def _out(rv, preds: Path) -> Path:
    p = Path(rv.root("dlc")) / "session_poses" / "camera_licks" / f"clips_{preds.parent.name}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sessions(preds: Path) -> list[str]:
    names = {f.name.rsplit("_", 1)[0] for f in preds.glob("PS*_cam2.csv")}
    return sorted(n.replace("_", ":") for n in names if (preds / f"{n}_cam3.csv").exists())


def _clips(rv, animal: str, date: str) -> Path:
    for c in CLIP_SETS:
        d = Path(rv.root("dlc")) / c
        if (d / f"{animal}_{date}" / "index.csv").exists():
            return d
    raise FileNotFoundError(f"{animal}_{date}: no side review clips in {CLIP_SETS}")


def _bundle_out(rv, bundle: str, out_tag: str = "") -> Path:
    # 11-21: + the 3-D run tag (`pose3d.TAG_ENV`, e.g. _3net_20261121), so a production run never lands on the pilot's
    # camera_licks/<bundle>/ tables; no tag = the untagged folder, as before
    from wfield_local.pose3d import pose3d_tag
    # 11-21 table gaps: ``out_tag`` (e.g. _gapfix) = a NEW folder for a re-fused train; the side events stay read from
    # the run's own folder (`_saved_side`), never rewritten
    if out_tag and not re.fullmatch(r"_[A-Za-z0-9_]+", out_tag):
        raise ValueError(f"out_tag {out_tag!r}: expected '_' + letters / digits / '_'")
    p = Path(rv.root("dlc")) / "session_poses" / "camera_licks" / (bundle + pose3d_tag("4cam") + out_tag)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _saved_side(out: Path, animal: str, date: str) -> dict:
    """The side events `scripts.pose3d production --four-cam` wrote (``<s>_side_events.csv``), per camera, with the
    guard columns back to bool."""
    E = pd.read_csv(out / f"{animal}_{date}_side_events.csv")
    for c in ("static", "far"):
        if c in E:
            E[c] = E[c].astype("boolean").fillna(False).astype(bool)
    return {c: g.reset_index(drop=True) for c, g in E.groupby("cam")}


def run_session(s: str, preds: str | None = None, bundle: str | None = None, saved: bool = False,
                gapfix: bool = False, out_tag: str = "") -> dict:
    """One session: clips (``preds``) or the whole session (``bundle``). ``saved`` (11-21): the side events and the
    3-D frames of the four-camera production run (`pose3d.pose3d_dir`) instead of rebuilding the fused session."""
    from wfield_local import camera_licks as CL
    from wfield_local import pose3d_side as PS
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = s.split(":")
    if bundle and saved:
        from wfield_local.pose3d import pose3d_dir
        src = _bundle_out(rv, bundle)
        out = _bundle_out(rv, bundle, out_tag)
        ev = CL.production_events(rv, animal, date, span="all")
        ev.update(_saved_side(src, animal, date))
        S = pd.read_csv(pose3d_dir(rv) / f"{animal}_{date}_frames.csv.gz",
                        usecols=["trial_id", "t_ms", "stop_ms", "position", "tongue_ml", "tongue_ap", "tongue_dv"])
        U = CL.fuse(ev, cuts=CL.event_cuts(ev, S=S))
        if gapfix:                     # 11-21 table-gaps rule (`camera_licks.confirm_side_only`)
            T = CL.anchor_track_frames(rv, animal, date)
            U = CL.confirm_side_only(U, {c: CL.tracked_frames(U, T.get(c)) for c in CL.CONFIRM_CAMS})
        pd.concat(ev.values(), ignore_index=True).to_csv(out / f"{animal}_{date}_events.csv", index=False,
                                                         float_format=FLOAT_FORMAT)
        U.to_csv(out / f"{animal}_{date}_fused.csv", index=False, float_format=FLOAT_FORMAT)
        V = CL.view_table(U).reset_index()
        V.insert(0, "session", f"{animal}_{date}")
        res = {"views": V, "n_trials": int(S.trial_id.nunique()), "n_events": {c: len(E) for c, E in ev.items()}}
        if gapfix:
            Rz = U.groupby("reason").size().rename("n").reset_index()
            Rz.insert(0, "session", f"{animal}_{date}")
            res["reasons"] = Rz
        return res
    if bundle:
        Z = PS.fuse_session(rv, animal, date, side_bundle=bundle)
        ev = CL.production_events(rv, animal, date, span="all")
        out = _bundle_out(rv, bundle)
    else:
        Z = PS.fuse_session(rv, animal, date, Path(preds), _clips(rv, animal, date))
        ev = {c: CL.within_frames(E, Z["F"]) for c, E in CL.production_events(rv, animal, date, span="all").items()}
        out = _out(rv, Path(preds))
    ev.update({c: CL.side_events(Z, c, span="all") for c in CL.SIDE})
    # 11-21: split back-to-back licks at the anchor cameras' boundaries, placed where the 3-D tongue returns to the mouth
    U = CL.fuse(ev, cuts=CL.event_cuts(ev, S=PS.frames_table(Z)))
    pd.concat(ev.values(), ignore_index=True).to_csv(out / f"{animal}_{date}_events.csv", index=False,
                                                     float_format=FLOAT_FORMAT)
    U.to_csv(out / f"{animal}_{date}_fused.csv", index=False, float_format=FLOAT_FORMAT)
    V = CL.view_table(U).reset_index()
    V.insert(0, "session", f"{animal}_{date}")
    return {"views": V, "n_trials": int(Z["F"].trial_id.nunique()), "n_events": {c: len(E) for c, E in ev.items()}}


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import camera_licks as CL
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clips")
    c.add_argument("sessions", nargs="*", help="ANIMAL:YYYYMMDD (default: every session with cam2 + cam3 predictions)")
    c.add_argument("--preds", type=Path, default=Path(PREDS))
    c.add_argument("--jobs", type=int, default=3)
    w = sub.add_parser("session")
    w.add_argument("sessions", nargs="+", help="ANIMAL:YYYYMMDD")
    w.add_argument("--bundle", required=True, help="side O2 bundle name (`pose3d_side.side_windows`)")
    w.add_argument("--jobs", type=int, default=3)
    w.add_argument("--saved", action="store_true",
                   help="fuse from the four-camera production run's saved side events + 3-D frames (no rebuild)")
    w.add_argument("--gapfix", action="store_true",
                   help="table-gaps rule (11-21): a side-only lick counts only when cam2 AND cam3 saw it; split-off "
                        "fragments dropped; cam4 / cam1 tracking written (`camera_licks.confirm_side_only`; adds reason "
                        "/ keep); needs --saved and --out-tag")
    w.add_argument("--out-tag", default="", help="suffix of a NEW output folder, e.g. _gapfix")
    a = ap.parse_args(argv)
    if a.cmd == "session":
        if a.gapfix and not (a.saved and a.out_tag):
            ap.error("--gapfix needs --saved and --out-tag (never rewrite the existing fusion folder)")
        sessions, worker = a.sessions, _Worker(bundle=a.bundle, saved=a.saved, gapfix=a.gapfix, out_tag=a.out_tag)
        out = _bundle_out(PathResolver(), a.bundle, a.out_tag)
        from wfield_local import session_licks as SL
        if any(out.glob("*_fused.csv")) and not (out / SL.MANIFEST).exists():
            # 11-22: never mix full-precision tables into a folder of 4-significant-digit ones under one manifest
            ap.error(f"{out} holds fused tables written before the full-precision format (no {SL.MANIFEST}); "
                     "write a NEW folder (--out-tag)")
    else:
        sessions, worker = a.sessions or _sessions(a.preds), _Worker(preds=str(a.preds))
        out = _out(PathResolver(), a.preds)
    res, fail = ak.fan_sessions(sessions, worker, jobs=a.jobs)
    for s, err in fail:
        print(f"FAILED {s}: {err}")
    if not res:
        return 1
    V = pd.concat([r["views"] for _s, r in res], ignore_index=True)
    V.to_csv(out / "views.csv", index=False, float_format=FLOAT_FORMAT)
    if a.cmd == "session":
        # 11-22: the format manifest `session_licks` requires (full-precision tables); records how the folder was made
        import json

        from wfield_local import session_licks as SL
        from wfield_local.pose3d import pose3d_dir
        old = out / SL.MANIFEST
        prev = set(json.loads(old.read_text(encoding="utf-8")).get("sessions", [])) if old.exists() else set()
        SL.write_manifest(out, bundle=a.bundle, saved=a.saved, gapfix=a.gapfix, out_tag=a.out_tag,
                          pose3d_dir=str(pose3d_dir()), min_ili_ms=CL.MIN_ILI_MS,
                          sessions=sorted(prev | {s for s, _r in res}), failed=sorted(s for s, _e in fail))
    if any("reasons" in r for _s, r in res):
        Rz = pd.concat([r["reasons"] for _s, r in res if "reasons" in r], ignore_index=True)
        Rz.to_csv(out / "reasons.csv", index=False)
        print("\nPOOLED reason (table-gaps rule):\n", Rz.groupby("reason").n.sum().to_string())
    for s, r in res:
        print(f"{s}: {r['n_trials']} trials, events {r['n_events']}")
    P = V.groupby("cams")[["n", "side_only", "guarded"]].sum().sort_values("n", ascending=False)
    P["share"] = (P.n / P.n.sum()).round(3)
    print("\nPOOLED licks by camera combination (side_only = no cam4 / cam1 event; guarded = droplet / far-from-mouth only, dropped):")
    print(P.to_string())
    print(f"\nall licks {int(P.n.sum())}; guarded-only (dropped) {int(P.guarded.sum())}; side camera only and kept "
          f"{int(P.side_only.sum() - P.guarded.sum())}; cams in fusion order {CL.PEAK_PRIORITY}")
    return 0


class _Worker:
    """Module-level, picklable worker (`parallel.fan_out` spawns by name)."""

    def __init__(self, preds: str | None = None, bundle: str | None = None, saved: bool = False, gapfix: bool = False,
                 out_tag: str = ""):
        self.preds, self.bundle, self.saved, self.gapfix, self.out_tag = preds, bundle, saved, gapfix, out_tag

    def __call__(self, s: str) -> dict:
        return run_session(s, self.preds, self.bundle, self.saved, self.gapfix, self.out_tag)


if __name__ == "__main__":
    raise SystemExit(main())
