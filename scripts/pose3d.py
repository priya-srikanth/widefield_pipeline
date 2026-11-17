"""Per-frame 3-D tongue / jaw / nose / spout (cam1 + cam4) and per-lick 3-D reach and aim (miss distance / lateral error against the target spout), per session (`wfield_local.pose3d`).

    python -m scripts.pose3d clips PS93:20260814 PS93:20260821 PS93:20260908      # test: cam1 clip (ctx-semi eval) + cam4 O2
    python -m scripts.pose3d production [ANIMAL:YYYYMMDD ...] [--jobs 4]           # both O2 production outputs
    python -m scripts.pose3d production ... --four-cam <side bundle | clips>         # + cam2 / cam3 (`pose3d_side`)

dlc env (aniposelib). cam4 = the LP r4 production output (`o2/lp_r4_prod_20261015`) in both modes; cam1 = the ctx-semi eval
predictions on the desktop trial clip (``clips``, `session_poses/<a>_<d>_cam1/windows_index.csv` rows) or the cam1
production output (`o2/lp_cam1_ctx_prod_20261019`). Writes to `session_poses/pose3d/`: ``<a>_<d>_frames.csv.gz`` (every
paired frame), ``<a>_<d>_licks.csv`` (one row per 3-D lick) and ``summary_<source>.csv`` (per session: pairing, how often
each part triangulates, reprojection px per part and camera, the world fit, lick counts, reach and aim by position).

FOUR CAMERAS (``--four-cam``, 2026-11-04): `pose3d_side.fuse_session` adds cam2 / cam3 (side model, cleaned with the
side parameters), robust four-camera triangulation, the water-drop rule, one-view fill and the 3-D QC; the per-frame
table has the same columns (`pose3d_side.frames_table`, + ``{part}_views`` and per-camera reprojection), so lick_peaks
runs unchanged. ``<side bundle>`` = the whole-session O2 output of the side model (model space, mapped to raw);
``clips`` = the side review clips (crop2 predictions; only their trials -- a test of the path). Writes to ``--out``
(default `session_poses/pose3d_4cam/`), never over the two-camera tables.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from wfield_local.pose3d import pose3d_dir

PROD = {"cam4": "lp_r4_prod_20261015", "cam1": "lp_cam1_ctx_prod_20261019"}
SIDE_BUNDLE_ENV = "WIDEFIELD_POSE3D_SIDE_BUNDLE"      # the side model's whole-session O2 bundle (4cam source)
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


def _four_cam_source() -> str:
    """The four-camera output folder: the env source when it is a 4cam variant (e.g. "4cam_clips" for review-clip tests),
    else "4cam"."""
    from wfield_local import pose3d as P3
    s = P3.pose3d_source()
    return s if s.startswith("4cam") else "4cam"


def run_session(s: str, source: str, four_cam: str | None = None, out_dir: str | None = None) -> dict:
    """One session -> files in session_poses/pose3d/ (or ``out_dir``) and its summary row (module level: fan_sessions
    pickles it). ``four_cam``: a side bundle name or "clips" (module docstring)."""
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
    if four_cam:
        if jf:
            return {**row, "error": "June: side-camera intrinsics not validated (four-camera mode skipped)"}
        from scripts.pose3d_side_check import CLIPS, PREDS
        from wfield_local import pose3d_side as PS
        Z = (PS.fuse_session(rv, animal, date, Path(PREDS), Path(rv.root("dlc")) / CLIPS, c1=c1, c4=c4)
             if four_cam == "clips" else PS.fuse_session(rv, animal, date, side_bundle=four_cam, c1=c1, c4=c4))
        S = PS.frames_table(Z)
        L = P3.lick_peaks(S, rig)
        out = Path(out_dir) if out_dir else pose3d_dir(rv, _four_cam_source())
        out.mkdir(parents=True, exist_ok=True)
        S.to_csv(out / f"{animal}_{date}_frames.csv.gz", index=False, float_format="%.3f")
        L.to_csv(out / f"{animal}_{date}_licks.csv", index=False)
        row.update({"four_cam": four_cam, "n_frames": len(S), "droplet_frames": Z["stats"]["droplet_frames"],
                    "fit_resid_mm_max": round(Z["stats"]["fit_resid_mm_max"], 3),
                    "tongue_rep1_med_px": round(Z["stats"]["tongue_rep1_med_px"], 1),
                    **{f"nose_off_{c}_dv": round(Z["stats"]["nose_offsets"][c][2], 2) for c in PS.SIDE},
                    **{f"tongue_off_{c}_ap": round(Z["stats"]["tongue_offsets"][c][1], 3) for c in PS.SIDE}})
        for part in PS.PARTS:
            st = Z["stats"][part]
            row[f"{part}_3d_frac"] = round(st["fused"], 4)
            row[f"{part}_cam14_frac"] = round(st["cam1+cam4"], 4)
            row[f"{part}_fill_frac"] = round(st["one_view_fill"], 4)
            for c in Z["order"]:
                row[f"{part}_rejected_{c}"] = st[f"rejected_{c}"]
        row["licks_3d"] = len(L)
        return row
    S, info = P3.session_3d(_cgroup(rv), rig, c1[0], c1[1], c4[0], c4[1], june_factor=jf)
    L = P3.lick_peaks(S, rig)
    out = Path(out_dir) if out_dir else pose3d_dir(rv, "2cam")
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
    ap.add_argument("--all", action="store_true",
                    help="every session with outputs, not just the curated cohort (`analysis_kit.curated_ids`)")
    ap.add_argument("--jobs", type=int, default=4, help="sessions in parallel (whole-session npz: I/O and ~1 GB RSS each)")
    ap.add_argument("--four-cam", default=None, metavar="SIDE_BUNDLE|clips",
                    help="add cam2 / cam3 (`pose3d_side`): an O2 side bundle name, or 'clips' (the side review clips)")
    ap.add_argument("--out", default=None, help="output folder (default session_poses/pose3d, or pose3d_4cam with --four-cam)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    import os

    from wfield_local import pose3d as P3
    if P3.pose3d_source() == "4cam" and not a.four_cam:   # the chain sets the source by env; 4cam_clips: pass --four-cam clips
        a.four_cam = os.environ.get(SIDE_BUNDLE_ENV)
        if not a.four_cam:
            raise SystemExit(f"{P3.SOURCE_ENV}=4cam needs --four-cam or {SIDE_BUNDLE_ENV} (the side model's O2 bundle)")
    sess = a.sessions
    if not sess and a.source == "production":
        have = [{p.name for p in (Path(rv.root("dlc")) / "o2" / b / "out").glob("PS*_2026*") if p.is_dir()}
                for b in PROD.values()]
        cur = set.intersection(*have) if a.all else set.intersection(*have) & ak.curated_ids()
        sess = [x.replace("_", ":", 1) for x in sorted(cur)]
    if not sess:
        raise SystemExit("no sessions")
    res, fails = ak.fan_sessions([(s, a.source, a.four_cam, a.out) for s in sess], _worker, jobs=min(a.jobs, len(sess)))
    rows = [v for _, v in res] + [{"session": it[0].replace(":", "_"), "source": a.source, "error": e} for it, e in fails]
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(T.T.to_string(header=False))
    base = Path(a.out) if a.out else pose3d_dir(rv, _four_cam_source() if a.four_cam else "2cam")
    out = base / f"summary_{a.source}{'_' + a.four_cam if a.four_cam else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():           # a SUBSET run replaces only its own sessions' rows (it used to overwrite the whole table)
        T = pd.concat([pd.read_csv(out), T], ignore_index=True).drop_duplicates("session", keep="last")
        T = T.sort_values("session").reset_index(drop=True)
    T.to_csv(out, index=False)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
