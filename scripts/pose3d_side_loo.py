"""Leave-one-camera-out offsets in the four-camera 3-D (`pose3d_side`): is a camera's keypoint a CONSTANT offset from the
other cameras' 3-D point (a labelling-definition difference -> build it in, like cam1's spout edge) or a variable one
(weight it down / leave it out)? Priya 2026-11-04: "should we build in cam1 offset for jaw / tongue".

    conda activate dlc
    python -m scripts.pose3d_side_loo PS93:20260814 PS93:20260821 PS93:20260908

Per session x camera x part: triangulate from the OTHER cameras (>= 2 views, the `fuse` offsets for spout / nose), then
the left-out camera's point vs that 3-D point -- median perpendicular offset to its ray (world ml / ap / dv mm), the
median and IQR of its length, and how much of the offset is constant: |median vector| / median |offset| (1 = every
frame shifted the same way; ~0 = scatter around zero). Tongue / jaw also split by mouth state (tongue out vs not, from
the fused tongue). Writes `session_poses/pose3d_side/loo_offsets.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["cam1", "cam4", "cam2", "cam3"]


def run(rv, session: str) -> list[dict]:
    from scripts.apparatus_mouth import _rig
    from scripts.pose3d import PROD, _cgroup, _june_factor
    from scripts.pose3d_side_check import CLIPS, PREDS
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d as P3
    from wfield_local import pose3d_side as PS
    from wfield_local import spout_frame as SF
    animal, date = session.split(":")
    if _june_factor(date):
        print(f"{session}: June skipped")
        return []
    c1 = oi.session_windows(rv, PROD["cam1"], animal, date, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], animal, date, "cam4")
    rig, cg = _rig(rv, animal, date), _cgroup(rv)
    S, info = P3.session_3d(cg, rig, c1[0], c1[1], c4[0], c4[1])
    side = PS.side_session(rv, animal, date, Path(PREDS), Path(rv.root("dlc")) / CLIPS, c4[0])
    meds = {"cam1": info["med1"], "cam4": info["med4"], **{c: SF.ruler_medians(side[c][1], side[c][0]) for c in PS.SIDE}}
    g, _ = PS.pose_all(cg, {c: meds[c] for c in ORDER}, rig["geometry"])
    order = [c.get_name() for c in g.cameras]
    F, pts = PS.four_camera_points(S, info, c1, c4, side, order)
    tongue_out = np.isfinite(PS.fuse(g, {"tongue": pts["tongue"]})["tongue"][0]).all(1)
    rows = []
    for part in PS.PARTS:
        gs = PS.shifted(g, PS.part_offsets(part))
        for j, cam in enumerate(order):
            others = [k for k in range(len(order)) if k != j]
            X, _rep = PS.triangulate_n(gs.subset_cameras_names([order[k] for k in others]), pts[part][others])
            c = [cc for cc in gs.cameras if cc.get_name() == cam][0]
            r = PS.ray_offset(c, pts[part][j], X)
            for state, m in (("all", np.ones(len(F), bool)), ("tongue_out", tongue_out), ("tongue_in", ~tongue_out)):
                if part in ("nose", "spout") and state != "all":
                    continue
                ok = m & np.isfinite(r).all(1)
                if ok.sum() < 20:
                    continue
                L = np.linalg.norm(r[ok], axis=1)
                med = np.median(r[ok], axis=0)
                rows.append({"session": f"{animal}_{date}", "part": part, "cam": cam, "state": state, "n": int(ok.sum()),
                             "ml": med[0], "ap": med[1], "dv": med[2], "med_len_mm": float(np.median(L)),
                             "iqr_len_mm": float(np.subtract(*np.percentile(L, [75, 25]))),
                             "constancy": float(np.linalg.norm(med) / np.median(L))})
    return rows


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", metavar="ANIMAL:YYYYMMDD")
    a = ap.parse_args(argv)
    rv = PathResolver()
    rows = [r for s in a.sessions for r in run(rv, s)]
    D = pd.DataFrame(rows)
    out = Path(rv.root("dlc")) / "session_poses" / "pose3d_side" / "loo_offsets.csv"
    D.to_csv(out, index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    print(D.round(2).to_string(index=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
