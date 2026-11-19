"""Licks across cameras on the side REVIEW CLIPS (`wfield_local.camera_licks`; Priya 11-18): what cam2 / cam3 add to the
cam4 + cam1 lick train, before the whole-session side inference exists. DRAFT (decisions D2 / D3 of
docs/status/STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md are open).

    conda activate dlc            # aniposelib (`pose3d_side.fuse_session` poses the cameras)
    python -m scripts.camera_licks clips [ANIMAL:YYYYMMDD ...] [--preds <video_preds dir>] [--jobs 3]

Per session with side predictions in ``--preds`` (default: the round-2 side model r2b on the review clips, `CLIP_SETS`;
cue -0.2 .. +1.5 s, 2 trials per position): `pose3d_side.fuse_session` (side tongue cleaned, side droplets removed), side events
(`camera_licks.side_events`), the production cam4 / cam1 events restricted to the same clip frames
(`camera_licks.within_frames`), fused (`camera_licks.fuse`). Writes `session_poses/camera_licks/clips_<model>/`:
``<a>_<d>_fused.csv`` (one row per lick), ``<a>_<d>_events.csv`` (every per-camera event), ``views.csv`` (licks per
camera combination per session) and prints the pooled view table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PREDS = "C:/Users/SabatiniLab/lp_stage/lp_side_r2b_20261118/r2_full/models/side_r2b_r2_full/video_preds"
#: side review clip folders under the dlc root (the 10-27 pilot set, `pose3d_side_check.CLIPS`, and the 11-04 Part A set);
#: a session's clips are the first folder holding ``<a>_<d>/index.csv``
CLIP_SETS = ("inference_check_20261027_side_pilot/clips", "inference_check_20261104_partA_clips")


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


def run_session(s: str, preds: str) -> dict:
    from wfield_local import camera_licks as CL
    from wfield_local import pose3d_side as PS
    from wfield_local.paths import PathResolver
    rv, preds = PathResolver(), Path(preds)
    animal, date = s.split(":")
    Z = PS.fuse_session(rv, animal, date, preds, _clips(rv, animal, date))
    ev = {c: CL.within_frames(E, Z["F"]) for c, E in CL.production_events(rv, animal, date, span="all").items()}
    ev.update({c: CL.side_events(Z, c, span="all") for c in CL.SIDE})
    U = CL.fuse(ev)
    out = _out(rv, preds)
    pd.concat(ev.values(), ignore_index=True).to_csv(out / f"{animal}_{date}_events.csv", index=False,
                                                     float_format="%.4g")
    U.to_csv(out / f"{animal}_{date}_fused.csv", index=False, float_format="%.4g")
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
    a = ap.parse_args(argv)
    sessions = a.sessions or _sessions(a.preds)
    res, fail = ak.fan_sessions(sessions, _Worker(str(a.preds)), jobs=a.jobs)
    for s, err in fail:
        print(f"FAILED {s}: {err}")
    if not res:
        return 1
    V = pd.concat([r["views"] for _s, r in res], ignore_index=True)
    V.to_csv(_out(PathResolver(), a.preds) / "views.csv", index=False, float_format="%.4g")
    for s, r in res:
        print(f"{s}: {r['n_trials']} trials, events {r['n_events']}")
    P = V.groupby("cams")[["n", "side_only", "static"]].sum().sort_values("n", ascending=False)
    P["share"] = (P.n / P.n.sum()).round(3)
    print("\nPOOLED licks by camera combination (side_only = no cam4 / cam1 event; static = static-only, dropped):")
    print(P.to_string())
    print(f"\nall licks {int(P.n.sum())}; static-only (dropped) {int(P.static.sum())}; side camera only and kept "
          f"{int(P.side_only.sum() - P.static.sum())}; cams in fusion order {CL.PEAK_PRIORITY}")
    return 0


class _Worker:
    """Module-level, picklable worker (`parallel.fan_out` spawns by name)."""

    def __init__(self, preds: str):
        self.preds = preds

    def __call__(self, s: str) -> dict:
        return run_session(s, self.preds)


if __name__ == "__main__":
    raise SystemExit(main())
