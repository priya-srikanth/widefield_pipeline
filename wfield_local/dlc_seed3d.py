"""Seed cam1/cam2/cam3 labels by TRIANGULATION, not by asking a frontal network about a side view.

Priya, 2026-09-13: "but you have the anipose triangulation" -- and then "use the same logic to place
the spout, jaw, tongue, nose, and whiskers (if possible) seeds in Cams 1, 2, 3."

The donor is a frontal network. cam2 and cam3 sit ~55 deg off cam4 and 110 deg from each other, and
cam1 looks up from below; that is a VIEWPOINT shift, which the input-scale trick that rescued cam4
does nothing for. Geometry does not care about viewpoint. Scored on the calibration recording with
no labels at all -- triangulate a board corner from cam4+cam1 alone, project into the side view,
compare against where that camera actually saw it -- the route lands at 4.48 px median on cam2 and
7.71 px on cam3, inside DLC's own 6 px dot. Those are ChArUco corners, so treat them as a FLOOR:
real landmarks add human placement scatter and the tongue deforms between views.

TRIANGULATION IS ITS OWN VALIDATOR, which is what makes this safe to run on views the donor was
never meant for. Two independently-wrong predictions do not agree in 3-D: if cam1's "jaw" and
cam4's "jaw" are not the same physical point, their rays do not meet and the reprojection residual
is large. So the donor can be run everywhere, and the GEOMETRY -- not a likelihood the network
reports about its own guess -- decides what survives. A confidently wrong seed is worse than a blank
frame, because the labeller's eye anchors to whatever is already on screen.

THIS IS ALSO THE WHISKER TEST. Whether `L_whiskers_1` in cam4 and `L_whiskers_1` in cam2 are the
same whisker has been an open question (DECISIONS.md, "correspondence error is not correctable").
The residual answers it empirically: if they are different whiskers, the rays miss and the part
fails acceptance on its own. No judgement call required.

CLI::

    conda activate dlc
    python -m wfield_local.dlc_seed3d --dry-run      # report acceptance + residuals, write nothing
    python -m wfield_local.dlc_seed3d                # ... and write CollectedData for cam1/2/3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local.dlc_frames import bodyparts, cameras, out_root
from wfield_local.paths import PathResolver

SCORER = "Priya"
MIN_LIKELIHOOD = 0.30      # deliberately BELOW prelabel's 0.6: geometry is the better filter, and a
                           # point the network is unsure of but that two views agree on is a good
                           # seed, while a confident point the views disagree on is not.
MAX_RESIDUAL_PX = 12.0     # against a 2.83 px calibration floor and a 4.5-7.7 px measured route
MIN_VIEWS = 2


def calibration_path(rv=None, machine=None) -> Path:
    """The solved ``calibration_anipose.toml`` in the NEWEST calibration recording.

    Resolved through `dlc_calibration.find_calibration_dir` rather than a config key, so re-recording
    a calibration is picked up with no edit here -- and so this module cannot drift onto a different
    solve from the one the rest of the pipeline triangulates with.

    ROOTED ON ``behavior_cameras``, not on the ``camera_calibration`` path key. That key resolves to
    ``Behavior_Cameras`` while the recordings sit a level lower under ``Behavior_Cameras/Widefield``,
    so `find_calibration_dir`'s own default raises here (2026-09-13). Passing the root that
    `dlc_frames.out_root` already uses keeps this module on the same tree as the frames it is
    seeding, which is the tree that matters.
    """
    from wfield_local.dlc_calibration import find_calibration_dir

    rv = rv or PathResolver(machine=machine)
    return Path(find_calibration_dir(root=rv.root("behavior_cameras"))) / "calibration_anipose.toml"


def camera_group(rv=None, machine=None):
    p = calibration_path(rv, machine)
    if not p.exists():
        raise FileNotFoundError(
            f"No solved calibration at {p} -- run `python -m wfield_local.dlc_anipose` first. "
            "Seeding by reprojection is only as good as the solve behind it, so this refuses "
            "rather than falling back to an older one.")
    from aniposelib.cameras import CameraGroup
    return CameraGroup.load(str(p))


def manifest(rv=None) -> pd.DataFrame:
    return pd.read_csv(out_root(rv) / "labeled-data" / "frame_manifest.csv")


def moment_key(df: pd.DataFrame) -> pd.Series:
    """What makes a row the SAME INSTANT across cameras.

    The anchor (`dlc.frames.anchor_cam`) makes every view sample the same trials and the same licks,
    so a moment is identified by the BEHAVIOUR rather than by a frame number -- frame numbers differ
    between cameras even when the instant does not.
    """
    return (df.animal.astype(str) + "|" + df.date.astype(str) + "|"
            + df.trial_id.astype(str) + "|" + df.phase.astype(str))


def _donor_channel(bp: str) -> str | None:
    pc = (config.defaults().get("dlc") or {}).get("prelabel") or {}
    if bp == "spout":
        sf = list(pc.get("spout_from") or [])
        return str(sf[0]) if sf else "R_spout"
    return bp


def raw_predictions(cam: str, rv=None, workdir=None) -> pd.DataFrame:
    """Donor predictions for ``cam`` in ORIGINAL pixels, with LIKELIHOOD KEPT.

    `dlc_prelabel.to_labels` blanks anything under its threshold, which is right for seeding one
    view directly and wrong here: the whole point is to let geometry arbitrate, and a blanked point
    cannot be arbitrated.
    """
    from wfield_local import dlc_prelabel as pre
    import cv2

    index = pre.frame_index(cam, rv)
    if index.empty:
        return pd.DataFrame()
    work = Path(workdir or Path(pre.donor()["local_copy"]).parent / "seed3d")
    cfg_path = pre.ensure_local_donor()
    first = cv2.imread(index.iloc[0]["path"])
    h0, w0 = first.shape[:2]
    out = []
    for s, bps in sorted(pre.scale_groups(cam).items()):
        video = work / f"{cam}_scale{s:.2f}.avi"
        dw, dh = pre.build_video(index, s, video)
        pred = pre.predict(cfg_path, video, work / f"out_{cam}_{s:.2f}", expect=len(index))
        sx, sy = dw / w0, dh / h0
        for bp in bps:
            src = _donor_channel(bp)
            if src is None or (src, "likelihood") not in pred.columns:
                continue
            out.append(pd.DataFrame({
                "video_stem": index.video_stem.values, "image": index.image.values,
                "cam": cam, "bodypart": bp,
                "x": pred[(src, "x")].to_numpy() / sx,
                "y": pred[(src, "y")].to_numpy() / sy,
                "likelihood": pred[(src, "likelihood")].to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fit_moment(obs: dict, cg, order: list[str]):
    """``(xyz, residual_px, [views used])`` for one bodypart at one instant, or ``(None, ..., [])``.

    Drops the worst-fitting view and refits while at least MIN_VIEWS remain -- one bad view should
    cost that view, not the point. With exactly two views the residual is still meaningful: the
    least-squares point sits between two rays that do not meet, and how far it sits from each is
    exactly the disagreement being tested.
    """
    use = [c for c in order if c in obs]
    while len(use) >= MIN_VIEWS:
        pts = np.full((len(order), 1, 2), np.nan)
        for c in use:
            pts[order.index(c), 0] = obs[c]
        xyz = cg.triangulate(pts, undistort=True)
        if not np.isfinite(xyz[0]).all():
            return None, float("inf"), []
        rep = cg.project(xyz)
        err = {c: float(np.linalg.norm(rep[order.index(c), 0] - obs[c])) for c in use}
        worst = max(err, key=err.get)
        if err[worst] <= MAX_RESIDUAL_PX:
            return xyz[0], float(np.median(list(err.values()))), list(use)
        if len(use) == MIN_VIEWS:
            return None, float(err[worst]), []
        use.remove(worst)
    return None, float("inf"), []


def build(rv=None, workdir=None, cams=None, verbose=True):
    """Triangulate every (moment, bodypart) the donor offers, then reproject into ALL views.

    Returns ``(seeds, report)``. ``seeds`` is one row per (cam, image, bodypart) with reprojected
    x/y; ``report`` is per-attempt acceptance and residual, which is the number that says whether a
    part is trustworthy at all.
    """
    rv = rv or PathResolver()
    cams = list(cams or cameras())
    cg = camera_group(rv)
    order = [c.name for c in cg.cameras]

    man = manifest(rv)
    man["key"] = moment_key(man)
    look = {(r.cam, r.video_stem, r.image): r.key for r in man.itertuples()}

    preds = []
    for cam in cams:
        p = raw_predictions(cam, rv, workdir)
        if not p.empty:
            p["key"] = [look.get((c, s, i)) for c, s, i in zip(p.cam, p.video_stem, p.image)]
            preds.append(p[p.key.notna()])
        if verbose:
            print(f"[dlc_seed3d] {cam}: {0 if p.empty else len(p)} raw predictions", flush=True)
    if not preds:
        return pd.DataFrame(), pd.DataFrame()
    P = pd.concat(preds, ignore_index=True)
    P = P[P.likelihood >= MIN_LIKELIHOOD]

    # where each camera's frame for a moment lives, so a reprojection can be filed against an image
    slot = {(r.cam, r.key): (r.video_stem, r.image) for r in man.itertuples()}

    seeds, rep_rows = [], []
    for (key, bp), g in P.groupby(["key", "bodypart"], sort=False):
        obs = {r.cam: np.array([r.x, r.y]) for r in g.itertuples()}
        xyz, resid, used = fit_moment(obs, cg, order)
        rep_rows.append({"bodypart": bp, "n_views": len(obs), "accepted": xyz is not None,
                         "residual": resid, "used": len(used)})
        if xyz is None:
            continue
        proj = cg.project(np.asarray(xyz).reshape(1, 3))
        for cam in cams:
            where = slot.get((cam, key))
            if where is None or bp not in bodyparts(cam):
                continue                      # never invent a part a view cannot see
            xy = proj[order.index(cam), 0]
            seeds.append({"cam": cam, "video_stem": where[0], "image": where[1], "bodypart": bp,
                          "x": float(xy[0]), "y": float(xy[1]), "residual": resid,
                          "n_views": len(used)})
    return pd.DataFrame(seeds), pd.DataFrame(rep_rows)


def report(rep: pd.DataFrame) -> pd.DataFrame:
    """Per-bodypart acceptance and residual -- the table that decides what is trustworthy."""
    if rep.empty:
        return rep
    g = rep.groupby("bodypart")
    out = pd.DataFrame({
        "attempts": g.size(),
        "accepted": g.accepted.sum(),
        "accept_pct": (100 * g.accepted.mean()).round(1),
        "median_resid_px": g.apply(lambda t: t.loc[t.accepted, "residual"].median(),
                                   include_groups=False).round(2),
    })
    return out.sort_values("accept_pct", ascending=False)


def write(seeds: pd.DataFrame, rv=None, dry=False, force=False) -> list[Path]:
    """Write DLC ``CollectedData_Priya.{h5,csv}`` per folder, REFUSING to clobber existing labels.

    The refusal is the same one `dlc_prelabel` makes and for the same reason: a CollectedData file
    is the only record of manual work, and discarding a human's corrections is the one thing this
    pipeline must never do.
    """
    from wfield_local.writeguard import assert_writable

    root = out_root(rv) / "labeled-data"
    written = []
    for (cam, stem), g in seeds.groupby(["cam", "video_stem"]):
        dest = root / stem
        target = dest / f"CollectedData_{SCORER}.h5"
        if target.exists() and not force:
            print(f"[dlc_seed3d] {stem}: CollectedData exists -> LEFT ALONE", flush=True)
            continue
        bps = bodyparts(cam)
        imgs = sorted(g.image.unique())
        cols = pd.MultiIndex.from_tuples([(SCORER, b, c) for b in bps for c in ("x", "y")],
                                         names=["scorer", "bodyparts", "coords"])
        idx = pd.MultiIndex.from_tuples([("labeled-data", stem, i) for i in imgs])
        df = pd.DataFrame(np.nan, index=idx, columns=cols)
        for r in g.itertuples():
            if r.bodypart in bps:
                df.loc[("labeled-data", stem, r.image), (SCORER, r.bodypart, "x")] = r.x
                df.loc[("labeled-data", stem, r.image), (SCORER, r.bodypart, "y")] = r.y
        n = int(df.notna().sum().sum() // 2)
        if dry:
            print(f"  [dry] {stem}: {n} points over {len(imgs)} frames", flush=True)
            continue
        assert_writable(dest)
        df.to_hdf(target, key="df_with_missing", mode="w")
        df.to_csv(dest / f"CollectedData_{SCORER}.csv")
        print(f"[dlc_seed3d] {stem}: wrote {n} points over {len(imgs)} frames", flush=True)
        written.append(target)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", action="append", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    rv = PathResolver(machine=args.machine)
    seeds, rep = build(rv, args.workdir, args.cam)
    if rep.empty:
        print("[dlc_seed3d] nothing to do", flush=True)
        return 1
    print("\n[dlc_seed3d] per-bodypart acceptance (geometry, not the network's own confidence):",
          flush=True)
    print(report(rep).to_string(), flush=True)
    print(f"\n[dlc_seed3d] {len(seeds)} seed points across "
          f"{seeds.cam.nunique() if not seeds.empty else 0} camera(s)", flush=True)
    write(seeds, rv, args.dry_run, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
