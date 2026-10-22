"""Move FaceRhythm ROIs between sessions and the MASTER face through the registration fields
(`scripts/face_rhythm_register.py`: ``register/fields/<s>.npy`` maps MASTER px -> SESSION px).

    <fr env python> scripts/face_rhythm_warp_rois.py to-master PS94_20260810      # a session's ROIs.h5 -> master draft
    <fr env python> scripts/face_rhythm_warp_rois.py to-sessions [--spacing 9]     # master ROIs.h5 -> every session
    <fr env python> scripts/face_rhythm_warp_rois.py to-sessions --cam cam2        # any camera (root <dlc>/face_rhythm/<cam>)
    <fr env python> scripts/face_rhythm_warp_rois.py carry PS93_20260821 PS95_20260810 --fields <old register>/fields
                                                    # a ROI drawn on an OLD master -> a new master session

to-master    the session's face / mask polygons -> master coordinates (numerical inverse of the field,
             `face_rhythm_overlay.inverse_points`), saved as the master's ROIs.h5 (a previous one is moved aside, never
             deleted) for checking / editing in `notebooks/face_rhythm_draw_roi.ipynb` (SESSION = the master, EDIT = True).
to-sessions  the master's POINTS (one mesh for the whole cohort: point i is the same face location in every session) and
             polygons -> each registered session (R(p), bilinear); masks re-rasterised from the warped polygons. JUNE is
             skipped (registration fell back to a similarity there; Priya 10-21: June not used by default). Each
             session's ``rois/<s>/ROIs.h5`` is written in the format the O2 runner reads (`scripts/o2_face_rhythm.py`).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"M:\MICROSCOPE\Priya\DeepLabCut\Widefield\face_rhythm\cam4")


def _save(path: Path, rois_obj, img, points):
    import face_rhythm as fr
    assert "MICROSCOPE" in str(path) and "Priya" in str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.with_name(f"ROIs_{time.strftime('%Y%m%d_%H%M%S', time.localtime(path.stat().st_mtime))}.h5")
        path.rename(old)
    fr.h5_handling.simple_save({"mask_images": rois_obj.mask_images, "roi_points": rois_obj.roi_points,
                                "exampleImage": img, "point_positions": np.asarray(points, np.float32)}, str(path))


def _sample(field, pts):
    from face_rhythm import helpers
    return helpers.remap_points(points=np.asarray(pts, np.float64), remappingIdx=field.astype(np.float64),
                                interpolation="linear")


def to_master(root: Path, session: str) -> Path:
    import face_rhythm as fr
    from face_rhythm_overlay import inverse_points
    reg = root / "register"
    master = json.loads((reg / "templates.json").read_text())["master"]
    src = fr.rois.ROIs(select_mode="file", path_file=str(root / "rois" / session / "ROIs.h5"), verbose=0)
    F = np.load(reg / "fields" / f"{session}.npy")
    polys = {k: inverse_points(F, np.asarray(v, float)).astype(np.float32) for k, v in src.roi_points.items()}
    img = np.load(root / "frames" / f"{master}.npz")["median"].astype(np.float32) / 255.0
    R = fr.rois.ROIs(select_mode="custom", exampleImage=img, coords_rois=polys, verbose=0)
    R.make_points(rois=[R[0], R[1]], point_spacing=9)
    out = root / "rois" / master / "ROIs.h5"
    _save(out, R, img, R.point_positions)
    print(f"{session} -> master {master}: {len(R.point_positions)} points -> {out}")
    return out


def carry(root: Path, src: str, dst: str, fields: Path, spacing: int = 9) -> Path:
    """The ROI saved on ``src`` (the OLD master) -> session ``dst`` through ``fields/<dst>.npy`` of the run in which
    ``src`` was the master (old master px -> dst px, sampled at the polygon vertices); points re-seeded where face AND
    mask cover, at ``spacing``. Used 10-21 when the master moved from PS93_20260821 (acute) to a pre-stroke face."""
    import face_rhythm as fr
    S = fr.rois.ROIs(select_mode="file", path_file=str(root / "rois" / src / "ROIs.h5"), verbose=0)
    F = np.load(Path(fields) / f"{dst}.npy")
    polys = {k: _sample(F, v).astype(np.float32) for k, v in S.roi_points.items()}
    img = np.load(root / "frames" / f"{dst}.npz")["median"].astype(np.float32) / 255.0
    R = fr.rois.ROIs(select_mode="custom", exampleImage=img, coords_rois=polys, verbose=0)
    R.make_points(rois=[R[0], R[1]], point_spacing=spacing)
    out = root / "rois" / dst / "ROIs.h5"
    _save(out, R, img, R.point_positions)
    print(f"{src} -> {dst}: {len(R.point_positions)} points, polygons {[len(v) for v in polys.values()]} -> {out}")
    return out


def to_sessions(root: Path, spacing: int | None) -> int:
    import face_rhythm as fr
    reg = root / "register"
    master = json.loads((reg / "templates.json").read_text())["master"]
    M = fr.rois.ROIs(select_mode="file", path_file=str(root / "rois" / master / "ROIs.h5"), verbose=0)
    pts = np.asarray(M.point_positions, float)
    if spacing:                                   # re-seed the master mesh at another spacing
        M.make_points(rois=[M[0], M[1]], point_spacing=spacing)
        pts = np.asarray(M.point_positions, float)
    n = 0
    for f in sorted((reg / "fields").glob("PS*.npy")):
        s = f.stem
        if s == master or s[9:11] == "06":        # the master keeps its own file; June skipped
            continue
        F = np.load(f)
        img = np.load(root / "frames" / f"{s}.npz")["median"].astype(np.float32) / 255.0
        polys = {k: _sample(F, v).astype(np.float32) for k, v in M.roi_points.items()}
        R = fr.rois.ROIs(select_mode="custom", exampleImage=img, coords_rois=polys, verbose=0)
        _save(root / "rois" / s / "ROIs.h5", R, img, _sample(F, pts))
        n += 1
    print(f"master {master}: {len(pts)} points -> {n} sessions")
    return n


def main(argv=None) -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["to-master", "to-sessions", "carry"])
    ap.add_argument("session", nargs="?", default=None)
    ap.add_argument("dst", nargs="?", default=None, help="carry: the target session")
    ap.add_argument("--fields", type=Path, default=None, help="carry: the fields/ folder of the OLD master's run")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--cam", default=None, help="cam4 | cam1 | cam2 | cam3: root = <dlc>/face_rhythm/<cam> (overrides --root)")
    ap.add_argument("--spacing", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cam:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from wfield_local import face_rhythm_cams as frc
        a.root = frc.default_root(a.cam)
    if a.cmd == "to-master":
        to_master(a.root, a.session)
    elif a.cmd == "carry":
        carry(a.root, a.session, a.dst, a.fields)
    else:
        to_sessions(a.root, a.spacing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
