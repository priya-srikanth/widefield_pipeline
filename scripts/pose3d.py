"""Per-frame 3-D tongue / jaw / nose / spout (cam1 + cam4) and per-lick 3-D reach and aim (miss distance / lateral error against the target spout), per session (`wfield_local.pose3d`).

    python -m scripts.pose3d clips PS93:20260814 PS93:20260821 PS93:20260908      # test: cam1 clip (ctx-semi eval) + cam4 O2
    python -m scripts.pose3d production [ANIMAL:YYYYMMDD ...] [--jobs 4]           # both O2 production outputs

dlc env (aniposelib). cam4 = the LP r4 production output (`o2/lp_r4_prod_20261015`) in both modes; cam1 = the ctx-semi eval
predictions on the desktop trial clip (``clips``, `session_poses/<a>_<d>_cam1/windows_index.csv` rows) or the cam1
production output (`o2/lp_cam1_ctx_prod_20261019`). Writes to `session_poses/pose3d/`: ``<a>_<d>_frames.csv.gz`` (every
paired frame), ``<a>_<d>_licks.csv`` (one row per 3-D lick) and ``summary_<source>.csv`` (per session: pairing, how often
each part triangulates, reprojection px per part and camera, the world fit, lick counts, reach and aim by position).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROD = {"cam4": "lp_r4_prod_20261015", "cam1": "lp_cam1_ctx_prod_20261019"}
CAM1_CLIP_MODEL = "o2/lp_train_cam1_r2_20261018/models/mhcrnn_semi_640_b4/video_preds/{a}_{d}_cam1.csv"


def _cgroup(rv):
    from aniposelib.cameras import CameraGroup

    from scripts.apparatus_mouth import TOML
    hits = glob.glob(f"{rv.root('behavior_cameras')}/{TOML}")
    if not hits:
        raise SystemExit(f"calibration not found: {TOML}")
    return CameraGroup.load(hits[0])


def _june_factor(date: str) -> float | None:
    from wfield_local import config
    c = config.defaults().get("orofacial_camera") or {}
    return float(c["scale_factor"]) if str(date) < str(c.get("scale_before", "0")) else None


def _cam1_clip(rv, animal: str, date: str):
    from scripts.session_poses import read_index, session_dir
    from wfield_local import orofacial_clean as oc
    _, _, d = session_dir(rv, f"{animal}:{date}:cam1")
    f = Path(rv.root("dlc")) / CAM1_CLIP_MODEL.format(a=animal, d=date)
    if not f.exists():
        return None
    ix = read_index(d)
    P = oc.read_pose(f)
    if len(P) < len(ix):
        return None
    return ix.reset_index(drop=True), P.iloc[:len(ix)].reset_index(drop=True)


def run_session(s: str, source: str) -> dict:
    """One session -> files in session_poses/pose3d/ and its summary row (module level: fan_sessions pickles it)."""
    from scripts.apparatus_mouth import _rig
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d as P3
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = s.split(":")
    row = {"session": f"{animal}_{date}", "source": source}
    c1 = _cam1_clip(rv, animal, date) if source == "clips" else oi.session_windows(rv, PROD["cam1"], animal, date, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], animal, date, "cam4")
    if c1 is None or c4 is None:
        return {**row, "error": f"missing: {'cam1 ' if c1 is None else ''}{'cam4' if c4 is None else ''}".strip()}
    rig = _rig(rv, animal, date)
    jf = _june_factor(date)
    S, info = P3.session_3d(_cgroup(rv), rig, c1[0], c1[1], c4[0], c4[1], june_factor=jf)
    L = P3.lick_peaks(S, rig)
    out = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / f"{animal}_{date}_frames.csv.gz", index=False, float_format="%.3f")
    L.to_csv(out / f"{animal}_{date}_licks.csv", index=False)
    w = info["world"]
    row.update({**info.get("qc", {}), "method": info["method"], "june_cam4_scaled": bool(jf), "shift_cam1_px": round(info["shift_px"][0], 1),
                "shift_cam4_px": round(info["shift_px"][1], 1), "n_cam1": info["n_cam1"], "n_cam4": info["n_cam4"],
                "n_paired": info["n_paired"], "world_scale": round(w.scale, 3),
                "fit_resid_mm_max": round(max(w.residual_mm.values()), 3),
                "spout_reproj_cam1_px": round(info["reproj_spout_px"]["cam1"], 1),
                "spout_reproj_cam4_px": round(info["reproj_spout_px"]["cam4"], 1)})
    for part in P3.CLEANED + P3.STATIC:
        ok = np.isfinite(S[f"{part}_ml"].to_numpy(float))
        row[f"{part}_3d_frames"] = int(ok.sum())
        for cam in ("1", "4"):
            row[f"{part}_rep{cam}_med_px"] = round(float(np.nanmedian(S[f"{part}_rep{cam}"][ok])), 1) if ok.any() else np.nan
    row["licks_3d"] = len(L)
    if len(L):
        for p, g in L.groupby("position"):
            row[f"reach_mm_{p}"] = round(float(g.reach_mm.median()), 2)
            row[f"miss_mm_{p}"] = round(float(g.miss_mm.median()), 2)
            row[f"lat_err_mm_{p}"] = round(float(g.lat_err_mm.median()), 2)
    return row


def _worker(item):
    return run_session(*item)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["clips", "production"])
    ap.add_argument("sessions", nargs="*", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--jobs", type=int, default=4, help="sessions in parallel (whole-session npz: I/O and ~1 GB RSS each)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    sess = a.sessions
    if not sess and a.source == "production":
        have = [{p.name for p in (Path(rv.root("dlc")) / "o2" / b / "out").glob("PS*_2026*") if p.is_dir()}
                for b in PROD.values()]
        sess = [x.replace("_", ":", 1) for x in sorted(set.intersection(*have))]
    if not sess:
        raise SystemExit("no sessions")
    res, fails = ak.fan_sessions([(s, a.source) for s in sess], _worker, jobs=min(a.jobs, len(sess)))
    rows = [v for _, v in res] + [{"session": it[0].replace(":", "_"), "source": a.source, "error": e} for it, e in fails]
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(T.T.to_string(header=False))
    out = Path(rv.root("dlc")) / "session_poses" / "pose3d" / f"summary_{a.source}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(out, index=False)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
