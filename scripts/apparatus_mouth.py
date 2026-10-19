"""The commanded (apparatus) MOUTH in cam1 and cam4, per session, from the triangulated spout positions
(`spout_world.apparatus_mouth`; Priya 2026-10-18 "build option 2"). Checks it is fixed to the head across days, and tests the
June cam4 correction ("we should be able to use june scaling to project into june as well").

    python -m scripts.apparatus_mouth clips  PS93:20260814 PS93:20260821 PS93:20260908      # trial clips, model predictions
    python -m scripts.apparatus_mouth labels [--animals PS92 PS93 PS94 PS95]                # labelled frames, both cameras
    python -m scripts.apparatus_mouth production [ANIMAL:YYYYMMDD ...] [--jobs 4]           # O2 production, every session

(dlc env: aniposelib.) ``production`` (10-19) = cam1 LP ctx-semi + cam4 LP r4 whole-session O2 outputs (`PROD`), every session
in both bundles; writes `apparatus_mouth_production.csv` (all variants) and `..._chosen.csv` (one row per session: Aug+ raw,
June cam4 x0.885 about the principal point -- `CHOSEN`).

Spout per position: ``clips`` = parked, no-lick ENL medians (`spout_frame.ruler_medians`; cam1 = DLC c1r2, cam4 = DLC r3 on the trial clip,
or the O2 whole-session LP r4 output where there is no cam4 clip -- June);
``labels`` = medians of the human spout labels per position (sessions labelled in BOTH cameras; lick frames, so the spout
may be pushed 2-5 px). June cam4 (dates < `orofacial_camera.scale_before`) is tried four ways: raw, and x
`orofacial_camera.scale_factor` about the image ORIGIN (v36 as applied), the calibration's principal point, and cam4's
spout-line intersection. Best = lowest reprojection error and fit residual. Reports, per session: the mouth in each camera,
its offset from that camera's nose (head-fixed: should be constant across days), reprojection px, fit residual mm and the
commanded / triangulated distance ratios. Calibration: `configs/paths.yaml` camera_calibration, the 09-11 rig toml.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

TOML = "Widefield_calibration_20260911_26mm_7x7_1200dpi_charuco/calibration_anipose.toml"


def _rig(rv, animal: str, date: str):
    from wfield_local import spout_world as SW
    hits = sorted(glob.glob(f"{rv.root('behavior_logs')}/{animal}_{date}_*/gui_config.json"))
    if not hits:
        raise SystemExit(f"no gui_config for {animal} {date}")
    return SW.session_rig(Path(hits[0]).parent)


def _cgroup(rv):
    from aniposelib.cameras import CameraGroup
    hits = glob.glob(f"{rv.root('behavior_cameras')}/{TOML}")
    if not hits:
        raise SystemExit(f"calibration not found: {TOML}")
    return CameraGroup.load(hits[0])


def _from_clips(rv, animal: str, date: str):
    from scripts.session_poses import read_index, session_dir
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    out = {}
    for cam, tag, pat in (("cam1", ":cam1", "dlc_r2/*.csv"), ("cam4", "", "windows_DLC.csv")):
        _, _, d = session_dir(rv, f"{animal}:{date}{tag}")
        f = glob.glob(str(d / pat))
        if not f and cam == "cam4":                       # no cam4 trial clip (June): the O2 whole-session output
            o2 = _cam4_from_o2(rv, animal, date)
            if o2 is None:
                return None
            out[cam] = o2
            continue
        if not f:
            return None
        idx = read_index(d)
        P = oc.read_pose(f[0])
        if len(P) < len(idx):
            return None
        P = P.iloc[:len(idx)]
        t = idx.t_ms.to_numpy(float)
        ok = (P.nose_likelihood >= 0.9).to_numpy() & (t < 0)
        out[cam] = (SF.ruler_medians(P, idx), np.array([np.median(P.nose_x[ok]), np.median(P.nose_y[ok])]))
    return out


def _col(d: pd.DataFrame, part: str, c: str) -> np.ndarray:
    s = d.loc[:, (slice(None), part, c)]
    return s.iloc[:, 0].to_numpy(float) if s.shape[1] else np.full(len(d), np.nan)


#: production bundles (configs: o2_inference.lp.model_rel / model_rel_cam1): cam4 LP r4, cam1 LP ctx-semi
PROD = {"cam4": "lp_r4_prod_20261015", "cam1": "lp_cam1_ctx_prod_20261019"}


def _cam4_from_o2(rv, animal: str, date: str, bundle: str = PROD["cam4"]):
    return _from_o2(rv, animal, date, bundle, "cam4")


def _from_o2(rv, animal: str, date: str, bundle: str, cam: str):
    """Parked spout medians + nose from an O2 whole-session output (RAW px: the June cam4 scaling is applied later, in the
    `evaluate` variants, never v36's). Split days: every video must be done; each is cut with its own recording's
    template, as `o2_inference.cmd_collect` does. None when the session is missing or incomplete."""
    import json
    from wfield_local import camera_sync as cs
    from wfield_local import o2_inference as oi
    from wfield_local import spout_frame as SF
    od = Path(rv.root("dlc")) / "o2" / bundle / "out" / f"{animal}_{date}"
    names = [v.name for v in oi.videos_for(rv, animal, date, cam)]
    vids = []
    for n in names:
        done = od / f"{Path(n).stem}_done.json"
        if not done.exists():
            return None
        npz = od / f"{Path(n).stem}_{json.loads(done.read_text(encoding='utf-8'))['scorer']}.npz"
        if not npz.exists():
            return None
        vids.append((n, np.load(npz)))
    if not vids:
        return None
    tpls = cs.templates_for(rv, date, animal, cam)
    rec = lambda name: Path(name).stem.split("_", 1)[1]           # noqa: E731  <cam>_<recording>.avi
    if any(rec(n) not in tpls for n, _ in vids):
        return None
    z = vids[0][1]
    videos = [(zz["pose"], t, n) for n, zz in vids for t in tpls[rec(n)]]
    parts = [str(x) for x in z["parts"]]
    ix, sel = oi.build_windows(videos[0][0], animal, date, rv, videos=videos if len(videos) > 1 else None, cam=cam)
    flat = pd.DataFrame({f"{p}_{c}": sel[:, k, j] for k, p in enumerate(parts) for j, c in enumerate(("x", "y", "likelihood"))})
    t = ix.t_ms.to_numpy(float)
    ok = (flat.nose_likelihood >= 0.9).to_numpy() & (t < 0)
    return SF.ruler_medians(flat, ix), np.array([np.median(flat.nose_x[ok]), np.median(flat.nose_y[ok])])


def _labels_by_session(rv):
    """{(animal, date): {cam: (spout medians per position, nose median)}} for sessions labelled in BOTH cameras."""
    from wfield_local import dlc_project
    from wfield_local.dlc_frames import manifest_path
    man = pd.read_csv(manifest_path(rv), dtype=str).drop_duplicates(["video_stem", "image"])
    live = dlc_project.project_dir(rv) / "labeled-data"
    acc: dict = {}
    for folder in sorted(live.glob("cam[14]_*")):
        csvs = sorted(folder.glob("CollectedData_*.csv"))
        if not csvs:
            continue
        d = pd.read_csv(csvs[0], header=[0, 1, 2], index_col=[0, 1, 2])
        img = [ix[2] for ix in d.index]
        m = man[man.video_stem == folder.name].set_index("image")
        if m.empty:
            continue
        cam = folder.name.split("_")[0]
        animal, date = m.animal.iloc[0], m.date.iloc[0]
        sx, sy, nx, ny = (_col(d, part, c) for part, c in (("spout", "x"), ("spout", "y"), ("nose", "x"), ("nose", "y")))
        pos = np.array([m.position.get(i, None) for i in img], dtype=object)
        med = {}
        for p in set(pos) - {None}:
            k = (pos == p) & np.isfinite(sx)
            if k.sum() >= 2:
                med[p] = (float(np.median(sx[k])), float(np.median(sy[k])), int(k.sum()))
        nose = np.array([np.nanmedian(nx), np.nanmedian(ny)])
        acc.setdefault((animal, date), {})[cam] = (med, nose)
    return {k: v for k, v in acc.items() if set(v) == {"cam1", "cam4"}}


def evaluate(rv, cg, animal: str, date: str, cams: dict, source: str) -> list[dict]:
    from wfield_local import config
    from wfield_local import spout_frame as SF
    from wfield_local import spout_world as SW
    rig = _rig(rv, animal, date)
    ocam = config.defaults().get("orofacial_camera") or {}
    june = str(date) < str(ocam.get("scale_before", "0"))
    m1, n1 = cams["cam1"]
    m4, n4 = cams["cam4"]
    variants = {"raw": m4}
    if june:
        f = float(ocam["scale_factor"])
        K = cg.subset_cameras_names(["cam4"]).cameras[0].get_camera_matrix()
        variants = {"raw": m4, "x0.885 @origin (v36)": SW.scale_about(m4, f, (0, 0)),
                    "x0.885 @principal pt": SW.scale_about(m4, f, (K[0, 2], K[1, 2])),
                    "x0.885 @cam4 mouth": SW.scale_about(m4, f, SF.from_medians(m4).origin)}
    rows = []
    for name, mv in variants.items():
        try:
            r = SW.apparatus_mouth({"cam1": m1, "cam4": mv}, rig["geometry"], cg)
        except ValueError as e:
            rows.append({"session": f"{animal}_{date}", "source": source, "cam4": name, "error": str(e)})
            continue
        w = r["world"]
        a1, a4 = r["mouth_px"]["cam1"], r["mouth_px"]["cam4"]
        rows.append({"session": f"{animal}_{date}", "source": source, "cam4": name, "n_pos": len(r["positions"]),
                     "cam1_mouth": f"({a1[0]:.0f},{a1[1]:.0f})", "cam1_mouth-nose": f"({a1[0] - n1[0]:+.0f},{a1[1] - n1[1]:+.0f})",
                     "cam4_mouth": f"({a4[0]:.0f},{a4[1]:.0f})", "cam4_mouth-nose": f"({a4[0] - n4[0]:+.0f},{a4[1] - n4[1]:+.0f})",
                     "cam4_spoutline-nose": "({:+.0f},{:+.0f})".format(*(SF.from_medians(m4).origin - n4)),
                     "reproj_cam1_px": round(r["reproj_px"]["cam1"], 1), "reproj_cam4_px": round(r["reproj_px"]["cam4"], 1),
                     "fit_resid_mm_max": round(max(w.residual_mm.values()), 3), "scale": round(w.scale, 3),
                     "dist_ratio_range": f"{w.distances.ratio.min():.3f}-{w.distances.ratio.max():.3f}"})
    return rows


#: the cam4 variant used downstream (10-19): Aug+ raw; June scaled about the cam4 PRINCIPAL POINT (D5 June test -- v36's
#: origin centre moves the cam1 mouth ~20 px; the scaling centre does not matter for 2-D measures, only here)
CHOSEN = ("raw", "x0.885 @principal pt")


def _production_one(s: str) -> list[dict]:
    """fan_sessions worker (module level: spawn pickles by name)."""
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = s.split(":")
    cams = {c: _from_o2(rv, animal, date, PROD[c], c) for c in ("cam1", "cam4")}
    miss = [c for c, v in cams.items() if v is None]
    if miss:
        return [{"session": f"{animal}_{date}", "source": "production", "error": f"no / incomplete O2 output: {miss}"}]
    return evaluate(rv, _cgroup(rv), animal, date, cams, "production")


def _production_sessions(rv) -> list[str]:
    """Sessions with an output folder in BOTH production bundles (completeness is checked per session)."""
    have = [{p.name for p in (Path(rv.root("dlc")) / "o2" / b / "out").glob("PS*_2026*") if p.is_dir()} for b in PROD.values()]
    return [s.replace("_", ":", 1) for s in sorted(set.intersection(*have))]


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["clips", "labels", "production"])
    ap.add_argument("--jobs", type=int, default=4, help="production: sessions in parallel (I/O-bound: whole-session npz)")
    ap.add_argument("sessions", nargs="*", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    cg = _cgroup(rv)
    rows = []
    if a.source == "clips":
        for s in a.sessions:
            animal, date = s.split(":")
            cams = _from_clips(rv, animal, date)
            if cams is None:
                print(f"{animal} {date}: predictions missing / incomplete -- skipped")
                continue
            rows += evaluate(rv, cg, animal, date, cams, "clips")
    elif a.source == "production":
        from wfield_local import analysis_kit as ak
        sess = a.sessions or _production_sessions(rv)
        if a.animals:
            sess = [s for s in sess if s.split(":")[0] in a.animals]
        res, fails = ak.fan_sessions(sess, _production_one, jobs=a.jobs)
        for s, err in fails:
            rows.append({"session": s.replace(":", "_"), "source": "production", "error": str(err)})
        rows += [r for _, rr in res for r in rr]
    else:
        for (animal, date), cams in sorted(_labels_by_session(rv).items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if a.animals and animal not in a.animals:
                continue
            rows += evaluate(rv, cg, animal, date, cams, "labels")
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    out = a.out or Path(rv.root("dlc")) / "session_poses" / f"apparatus_mouth_{a.source}.csv"
    T.to_csv(out, index=False)
    print(f"-> {out}")
    if a.source == "production" and "cam4" in T:                 # one row per session: the variant used downstream
        C = T[T.cam4.isin(CHOSEN)]
        C.to_csv(out.with_name(out.stem + "_chosen.csv"), index=False)
        print(f"-> {out.with_name(out.stem + '_chosen.csv')} ({len(C)} sessions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
