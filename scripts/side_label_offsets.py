"""Cross-camera keypoint DEFINITIONS from the student's own labels (Priya 2026-11-04: "plan to compare the student
labels"): at the same DAQ moment labeled in a side camera AND in cam1 + cam4, where does the side label sit relative to
the 3-D point her cam1 + cam4 labels triangulate to? Model-free counterpart of the side-model offsets fitted in
`pose3d_side` (side nose dv +0.3-1.2 mm, side tongue ap +0.1-0.2 mm), which may be the model's or the labels'.

    conda activate dlc
    python -m scripts.side_label_offsets PS92:20260820 PS93:20260820 PS94:20260903 PS95:20260821 PS95:20260907 \\
        [--clips <dir>] [--preds <dir>]

Moments: the side SIDECAM Part A frames (`dlc_sidecam.PILOT_STEMS`, the student's labels; Part B seeds are skipped until
edited) sharing (animal, date, trial, time from cue) with labeled cam4 and cam1 frames (`frame_manifest.csv`). Cameras
posed per session from the ENL no-lick spout medians (cam1 / cam4: the production windows; cam2 / cam3: the side model
on short cue clips, `scripts/side_review_clips.py cut --pre-ms 1000`), spout with cam1's edge offset. Per part x camera:
(a) the side label vs the cam1 + cam4 label point (ray-perpendicular vector, mm); (b) leave-one-camera-out over all
four cameras' labels where >= 3 are labeled. Writes `session_poses/pose3d_side/label_offsets{,_loo}.csv`.
August-onward sessions only (June side intrinsics open).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLIPS = "inference_check_20261104_partA_clips"
PREDS = "C:/Users/SabatiniLab/lp_stage/lp_side_crop2_20261027/models/side_crop2_heatmap_20261027/video_preds"
ORDER = ["cam1", "cam4", "cam2", "cam3"]


def labels(rv, man: pd.DataFrame, animal: str, date: str, pilot: set) -> pd.DataFrame:
    """One row per (camera, moment): the student's raw-px labels per part (NaN = not placed)."""
    from wfield_local import dlc_project
    live = dlc_project.project_dir(rv) / "labeled-data"
    m = man[(man.animal == animal) & (man.date == date)]
    rows = []
    for cam in ORDER:
        for stem in sorted(m.loc[m.cam == cam, "video_stem"].unique()):
            if cam in ("cam2", "cam3") and stem not in pilot:
                continue                                           # Part B / C: machine seeds until edited
            f = live / stem / "CollectedData_Priya.csv"
            if not f.is_file():
                continue
            d = pd.read_csv(f, header=[0, 1, 2], index_col=[0, 1, 2])
            d.index = [ix[2] for ix in d.index]
            d = d.droplevel(0, axis=1)
            mm = m[m.video_stem == stem].drop_duplicates("image").set_index("image")
            for img, r in d.iterrows():
                if img not in mm.index:
                    continue
                rec = {"cam": cam, "mkey": mm.loc[img, "mkey"], "position": mm.loc[img, "position"]}
                for p in ("nose", "jaw", "tongue", "spout"):
                    rec[f"{p}_x"], rec[f"{p}_y"] = r.get((p, "x"), np.nan), r.get((p, "y"), np.nan)
                rows.append(rec)
    return pd.DataFrame(rows)


def run(rv, session: str, clips: Path, preds: Path, man: pd.DataFrame, pilot: set) -> tuple[list, list]:
    from scripts.apparatus_mouth import _rig
    from scripts.pose3d import PROD, _cgroup, _june_factor
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d as P3
    from wfield_local import pose3d_side as PS
    from wfield_local import spout_frame as SF
    animal, date = session.split(":")
    if _june_factor(date):
        print(f"{session}: June skipped")
        return [], []
    c1 = oi.session_windows(rv, PROD["cam1"], animal, date, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], animal, date, "cam4")
    rig, cg = _rig(rv, animal, date), _cgroup(rv)
    S, info = P3.session_3d(cg, rig, c1[0], c1[1], c4[0], c4[1])
    side = PS.side_session(rv, animal, date, preds, clips, c4[0])
    meds = {"cam1": info["med1"], "cam4": info["med4"], **{c: SF.ruler_medians(side[c][1], side[c][0]) for c in PS.SIDE}}
    g, _ = PS.pose_all(cg, {c: meds[c] for c in ORDER}, rig["geometry"])
    order = [c.get_name() for c in g.cameras]
    L = labels(rv, man, animal, date, pilot)
    if L.empty:
        return [], []
    keys = sorted(set(L.mkey))
    rows, loo = [], []
    for part in PS.PARTS:
        gs = PS.shifted(g, PS.part_offsets(part) if part == "spout" else {})
        pts = np.full((len(order), len(keys), 2), np.nan)
        for j, cam in enumerate(order):
            Lc = L[L.cam == cam].drop_duplicates("mkey").set_index("mkey")
            for k, key in enumerate(keys):
                if key in Lc.index:
                    pts[j, k] = Lc.loc[key, [f"{part}_x", f"{part}_y"]].to_numpy(float)
        j1, j4 = order.index("cam1"), order.index("cam4")
        ref, _ = PS.triangulate_n(gs.subset_cameras_names(["cam1", "cam4"]), pts[[j1, j4]])
        for c in PS.SIDE:
            j = order.index(c)
            r = PS.ray_offset([cc for cc in gs.cameras if cc.get_name() == c][0], pts[j], ref)
            ok = np.isfinite(r).all(1)
            for k in np.flatnonzero(ok):
                rows.append({"session": f"{animal}_{date}", "part": part, "cam": c, "mkey": keys[k],
                             "ml": r[k, 0], "ap": r[k, 1], "dv": r[k, 2]})
        for j, cam in enumerate(order):                              # (b) leave one camera's label out
            others = [o for o in range(len(order)) if o != j]
            X, _ = PS.triangulate_n(gs.subset_cameras_names([order[o] for o in others]), pts[others])
            r = PS.ray_offset([cc for cc in gs.cameras if cc.get_name() == cam][0], pts[j], X)
            nv = np.isfinite(pts[others]).all(2).sum(0)
            for k in np.flatnonzero(np.isfinite(r).all(1) & (nv >= 2)):
                loo.append({"session": f"{animal}_{date}", "part": part, "cam": cam, "mkey": keys[k],
                            "ml": r[k, 0], "ap": r[k, 1], "dv": r[k, 2]})
    return rows, loo


def main(argv=None) -> int:
    from wfield_local import dlc_frames as DF
    from wfield_local.dlc_sidecam import PILOT_STEMS
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--clips", type=Path, default=None)
    ap.add_argument("--preds", type=Path, default=Path(PREDS))
    a = ap.parse_args(argv)
    rv = PathResolver()
    clips = a.clips or Path(rv.root("dlc")) / CLIPS
    man = pd.read_csv(DF.manifest_path(rv), dtype=str)
    man["mkey"] = (man.animal + "_" + man.date + "_t" + man.trial_id + "_"
                   + man.t_from_cue_s.astype(float).round(3).astype(str))
    R, O = [], []
    for s in a.sessions:
        r, o = run(rv, s, clips, a.preds, man, set(PILOT_STEMS))
        R += r
        O += o
    out = Path(rv.root("dlc")) / "session_poses" / "pose3d_side"
    D, E = pd.DataFrame(R), pd.DataFrame(O)
    D.to_csv(out / "label_offsets.csv", index=False, float_format="%.3f")
    E.to_csv(out / "label_offsets_loo.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220)

    def summ(X):
        g = X.groupby(["part", "cam"])
        t = g[["ml", "ap", "dv"]].median().round(2)
        t["n"] = g.size()
        t["med_len"] = g.apply(lambda v: np.median(np.linalg.norm(v[["ml", "ap", "dv"]].to_numpy(), axis=1))).round(2)
        t["sessions_same_sign_dv"] = g.apply(lambda v: int(np.sign(v.groupby("session").dv.median()).abs().sum()))
        return t
    print("(a) student's side label vs her cam1 + cam4 labels' 3-D point (mm, median ray-perpendicular vector):")
    print(summ(D).to_string())
    print("    per session (dv nose / ap tongue):")
    print(D[D.part.isin(["nose", "tongue"])].groupby(["part", "cam", "session"])[["ap", "dv"]].median().round(2).unstack(0).to_string())
    print("(b) leave-one-camera-out on the student's labels:")
    print(summ(E).to_string())
    print(f"-> {out}/label_offsets.csv, label_offsets_loo.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
