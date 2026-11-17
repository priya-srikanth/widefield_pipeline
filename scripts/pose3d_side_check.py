"""Side cameras in the 3-D pose: per-session PnP of cam2 / cam3 and the cross-camera OFFSET check (`wfield_local.pose3d_side`).

    conda activate dlc            # aniposelib
    python -m scripts.pose3d_side_check PS93:20260814 PS93:20260821 PS93:20260908 PS95:20260606 [--preds <dir>]

Side points = the side model's predictions on the review clips (`scripts/side_review_clips.py`, crop2 by default; the
clips are cut on each camera's template, cue -0.2 .. +1.5 s, 2 trials per position). Reference = the production cam1 +
cam4 3-D of the same trials (`scripts/pose3d` sources: `o2/lp_cam1_ctx_prod_20261019`, `o2/lp_r4_prod_20261015`).
Every camera posed per session from the parked spouts; side spout keypoint at the commanded point (``--side-edge-dv``
0) and one rod radius below it (-0.62 mm), both reported. Writes `session_poses/pose3d_side/`: ``offsets.csv`` (side
keypoint vs the cam1 + cam4 point, per session x camera x part), ``side_stereo.csv`` (cam2 + cam3 triangulated alone vs
cam1 + cam4), ``pnp.csv`` (spout reprojection px per camera) and prints the tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PREDS = "C:/Users/SabatiniLab/lp_stage/lp_side_crop2_20261027/models/side_crop2_heatmap_20261027/video_preds"
CLIPS = "inference_check_20261027_side_pilot/clips"
ROD_RADIUS_MM = 0.62


def run(s: str, preds: Path, rv) -> tuple[list, list, list]:
    from scripts.apparatus_mouth import _rig
    from scripts.pose3d import PROD, _cgroup, _june_factor
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d as P3
    from wfield_local import pose3d_side as PS
    from wfield_local import spout_frame as SF
    animal, date = s.split(":")
    c1 = oi.session_windows(rv, PROD["cam1"], animal, date, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], animal, date, "cam4")
    if c1 is None or c4 is None:
        print(f"{s}: missing cam1 / cam4 production output")
        return [], [], []
    rig, cg = _rig(rv, animal, date), _cgroup(rv)
    S, info = P3.session_3d(cg, rig, c1[0], c1[1], c4[0], c4[1], june_factor=_june_factor(date))
    side = PS.side_session(rv, animal, date, preds, Path(rv.root("dlc")) / CLIPS, c4[0])
    meds = {"cam1": info["med1"], "cam4": info["med4"],
            **{c: SF.ruler_medians(side[c][1], side[c][0]) for c in PS.SIDE}}
    june = int(side["cam2"][0].width.iloc[0]) != 600
    off_rows, st_rows, pnp_rows = [], [], []
    for dv in (0.0, -ROD_RADIUS_MM):
        g, err = PS.pose_all(cg, meds, rig["geometry"], side_edge_dv=dv)
        ref = info["cg"]
        same = all(np.allclose(PS.cam_of(g, c).get_extrinsics_mat(), PS.cam_of(ref, c).get_extrinsics_mat(), atol=1e-6)
                   for c in ("cam1", "cam4"))
        tag = {"session": f"{animal}_{date}", "side_edge_dv": dv, "june_side": june}
        pnp_rows.append({**tag, "cam1_cam4_pose_as_production": same,
                         **{f"spout_reproj_{c}_px": round(float(np.mean(list(e.values()))), 2) for c, e in err.items()},
                         **{f"n_pos_{c}": len(meds[c]) for c in meds}})
        o = PS.offsets_vs_reference(g, S, side)
        off_rows += [{**tag, **r} for r in o.to_dict("records")]
        st = PS.side_stereo_vs_reference(g, S, side)
        st_rows += [{**tag, **r} for r in st.to_dict("records")]
    return off_rows, st_rows, pnp_rows


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--preds", type=Path, default=Path(PREDS), help="side model video_preds (model-space csv per clip)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    O, T, N = [], [], []
    for s in a.sessions:
        o, t, n = run(s, a.preds, rv)
        O += o
        T += t
        N += n
    out = Path(rv.root("dlc")) / "session_poses" / "pose3d_side"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for name, rows in (("pnp", N), ("offsets", O), ("side_stereo", T)):
        df = pd.DataFrame(rows)
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.3f")
        print(f"\n== {name}\n{df.round(2).to_string(index=False)}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
