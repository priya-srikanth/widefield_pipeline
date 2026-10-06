"""Multi-view Lightning Pose (cam1 + cam4) label export, synced unlabelled clips, calibration subset.

    conda activate dlc
    python -m wfield_local.lp_multiview export --out C:/Users/SabatiniLab/lp_stage/multiview-pilot-20261006
    python -m wfield_local.lp_multiview clips  --out C:/Users/SabatiniLab/lp_stage/multiview-pilot-20261006
    python -m wfield_local.lp_multiview calib  --out C:/Users/SabatiniLab/lp_stage/multiview-pilot-20261006-calib

WHAT LP 2.4.2 EXPECTS (read from the installed package, 2026-10-06, not guessed):
* One CSV per view (`data.csv_file` is a LIST, same order as `data.view_names`), DLC 3-row header
  (scorer / bodyparts / coords). With an explicit visibility flag the coords row is x, y, visible
  for EVERY keypoint, in that order (`utils/io.parse_label_csv` reshapes the columns to (N, K, 3)).
  Values must be 0 / 1 / 2: 0 = not labelled (all-zero heatmap, dropped from the loss by
  `HeatmapLoss.remove_nans`), 1 = occluded (uniform heatmap target, overrides any x, y),
  2 = visible (Gaussian; a visible-2 point with NaN x, y also becomes a zero heatmap).
* Rows are paired BY POSITION: every CSV has the same number of rows and row i must have the same
  image BASENAME in every view (`MultiviewHeatmapDataset.check_data_images_names`). There is no
  "missing view" row: an unpaired moment needs the other camera's image at the same instant, with
  every keypoint visible 0 there. So this export extracts that image from the video.
* Calibration (arm b) is AUTO-DISCOVERED: `<data_dir>/calibration.toml` (or
  `calibrations/<session>.toml`), with session = labeled-data folder name minus its last `_<view>`.
  Camera names in the toml must equal `view_names` in order. Its mere presence turns on 3-D
  augmentation (imgaug forced to `dlc-mv`), so arm (a) must use a data_dir WITHOUT the toml.
* Unlabelled videos for the temporal loss: `<video_dir>/<clip>_<view>.mp4`, one per view with the
  same clip name; DALI reads frame k of every view together, so the clips must be frame-synced.
  `clips` cuts them on the DAQ clock (cam4 frames consecutive, cam1 frame = the DAQ-matched one).

THE SPLIT RULE (DECISIONS 2026-10-05, Priya), per keypoint k, view v, moment m:
* view v has no labelled frame at m (unpaired)              -> 0 (not labelled)
* m is listed in `dlc.train.all_occluded` for view v         -> 1 (every part hidden)
* k labelled in v                                            -> 2
* k blank in v and labelled in SOME other labelled view      -> 0 (let the network infer it)
* k blank in EVERY labelled view at m                        -> 1 (teaches "absent")

PAIRING: a moment is a DAQ instant, keyed by its cam4-clock frame. cam1 frames map to cam4 through
the alignment templates (`dlc_frames.frame_of`, the same affine `dlc_hard_frames.matched_frames`
uses; ~1.2 ms residual against a 4 ms frame). A cam1 and a cam4 label are the same moment when they
map to the same cam4 frame (`--tol` frames, default 1, for the rounding of a ~1 ms residual).

LABEL SOURCES: the per-camera DLC TRAINING copies (`dlc_train.train_project`), i.e. exactly the
label sets the single-view models were trained on (cam1 round 1, 294 rows; cam4 rounds 1-3, 407
rows) -- never the labelling project, which holds rounds in progress. The `all_occluded` frames
were dropped from the training copy (all-blank) and are added back here as visible 1.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

VIEWS = ("cam1", "cam4")
ANCHOR = "cam4"
SCORER = "Priya"
NOT_LABELLED, OCCLUDED, VISIBLE = 0, 1, 2


# ------------------------------------------------------------------------------ the split rule

def split_visibility(labelled: dict, present: dict, all_occluded: dict | None = None) -> dict:
    """Per-view (n, K) visibility codes under the split rule.

    labelled[v]     (n, K) bool -- keypoint k has x, y in view v at moment i
    present[v]      (n,)   bool -- view v has a labelled frame at moment i (False = unpaired here)
    all_occluded[v] (n,)   bool -- moment i of view v is in `dlc.train.all_occluded`
    """
    views = list(labelled)
    all_occluded = all_occluded or {}
    n, K = np.asarray(labelled[views[0]]).shape
    lab = {v: np.asarray(labelled[v], bool) & np.asarray(present[v], bool)[:, None] for v in views}
    pres = {v: np.asarray(present[v], bool) for v in views}
    occ = {v: np.asarray(all_occluded.get(v, np.zeros(n, bool)), bool) & pres[v] for v in views}
    out = {}
    for v in views:
        others = np.zeros((n, K), bool)
        for w in views:
            if w != v:
                others |= lab[w]
        code = np.full((n, K), NOT_LABELLED, np.int64)
        code[pres[v][:, None] & lab[v]] = VISIBLE
        code[pres[v][:, None] & ~lab[v] & ~others] = OCCLUDED
        code[occ[v]] = OCCLUDED                       # every part checked hidden (wins over the rest)
        out[v] = code
    return out


def view_frame(xy: np.ndarray, vis: np.ndarray, index: list[str], keypoints: list[str],
               scorer: str = SCORER) -> pd.DataFrame:
    """LP multi-view CSV body for one view: columns (scorer, keypoint, x|y|visible), x/y NaN unless 2."""
    xy = np.array(xy, float, copy=True)
    xy[vis != VISIBLE] = np.nan
    cols, data = [], []
    for k, kp in enumerate(keypoints):
        cols += [(scorer, kp, "x"), (scorer, kp, "y"), (scorer, kp, "visible")]
        data += [xy[:, k, 0], xy[:, k, 1], vis[:, k].astype(float)]
    df = pd.DataFrame(np.column_stack(data) if data else np.empty((len(index), 0)), index=index,
                      columns=pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"]))
    for kp in keypoints:
        df[(scorer, kp, "visible")] = df[(scorer, kp, "visible")].astype(int)
    return df


def count_codes(vis: np.ndarray, keypoints: list[str]) -> dict:
    return {kp: {c: int((vis[:, k] == c).sum()) for c in (NOT_LABELLED, OCCLUDED, VISIBLE)}
            for k, kp in enumerate(keypoints)}


# ------------------------------------------------------------------------------ clocks / pairing

def map_frame(frame, src_tpl: dict, dst_tpl: dict) -> np.ndarray:
    """``dst`` camera frame at the DAQ instant of ``src`` frame(s) (`matched_frames`' affine)."""
    f = np.asarray(frame, float)
    t_daq = (f * float(src_tpl["slope_daqSample_per_camFrame"]) + float(src_tpl["intercept_daqSample"])) \
        / float(src_tpl["fs_daq"])
    fs = float(dst_tpl["fs_daq"])
    return np.rint((t_daq * fs - float(dst_tpl["intercept_daqSample"]))
                   / float(dst_tpl["slope_daqSample_per_camFrame"])).astype(np.int64)


def pair_moments(rows: dict, tol: int = 1) -> pd.DataFrame:
    """One row per moment from per-view label tables that carry (animal, date, anchor) columns.

    Exact anchor matches first, then nearest within ``tol`` frames (one-to-one). Columns of view v
    are suffixed ``_<v>``; ``delta`` = anchor(cam1) - anchor(cam4) for paired moments.
    """
    a, b = (rows[v].copy() for v in VIEWS)
    a["_ia"], b["_ib"] = np.arange(len(a), dtype=float), np.arange(len(b), dtype=float)
    pairs = []
    for key, ga in a.groupby(["animal", "date"]):
        gb = b[(b.animal == key[0]) & (b.date == key[1])]
        free_b = dict(zip(gb.anchor.to_numpy(), gb._ib.to_numpy())) if len(gb) else {}
        used_a = set()
        for d in range(0, tol + 1):
            for ia, fa in zip(ga._ia.to_numpy(), ga.anchor.to_numpy()):
                if ia in used_a:
                    continue
                for cand in ((fa,) if d == 0 else (fa - d, fa + d)):
                    if cand in free_b:
                        pairs.append((ia, free_b.pop(cand), int(fa - cand)))
                        used_a.add(ia)
                        break
    pa = {p[0] for p in pairs}
    pb = {p[1] for p in pairs}
    out = [dict(ia=ia, ib=ib, delta=d) for ia, ib, d in pairs]
    out += [dict(ia=float(ia), ib=np.nan, delta=np.nan) for ia in range(len(a)) if ia not in pa]
    out += [dict(ia=np.nan, ib=float(ib), delta=np.nan) for ib in range(len(b)) if ib not in pb]
    m = pd.DataFrame(out)
    left = a.add_suffix(f"_{VIEWS[0]}").rename(columns={f"_ia_{VIEWS[0]}": "ia"})
    right = b.add_suffix(f"_{VIEWS[1]}").rename(columns={f"_ib_{VIEWS[1]}": "ib"})
    m = m.merge(left, on="ia", how="left").merge(right, on="ib", how="left")
    for c in ("animal", "date", "anchor"):
        m[c] = m[f"{c}_{VIEWS[1]}"].where(m.ib.notna(), m[f"{c}_{VIEWS[0]}"])
    m["anchor"] = m.anchor.astype(np.int64)
    m["paired"] = m.ia.notna() & m.ib.notna()
    return m.sort_values(["animal", "date", "anchor"]).reset_index(drop=True)


def lp_image_path(animal: str, date: str, view: str, anchor: int) -> str:
    """``labeled-data/<animal>_<date>_<view>/img<cam4 frame>.png`` -- LP's <session>_<view> layout,
    and the SAME basename in every view (the anchor = the cam4-clock frame of the moment)."""
    return f"labeled-data/{animal}_{date}_{view}/img{int(anchor):07d}.png"


# ------------------------------------------------------------------------------ I/O (real data)

def _templates(rv, animal, date):
    root = Path(rv.root("alignment_templates"))
    return {v: dict(np.load(root / v / animal / f"{date}.npz", allow_pickle=True)) for v in VIEWS}


def all_occluded_list() -> set[str]:
    from wfield_local import config
    return {str(x) for x in ((config.defaults().get("dlc") or {}).get("train") or {}).get("all_occluded") or []}


def load_view(cam: str, rv) -> tuple[pd.DataFrame, list[str]]:
    """Training-copy labels for ``cam`` joined to the frame manifest, + the all_occluded frames."""
    from wfield_local.dlc_frames import staging_root
    from wfield_local.dlc_project import project_dir
    from wfield_local.dlc_train import train_project

    frames = []
    for d in sorted((train_project(rv, cam) / "labeled-data").iterdir()):
        h5 = sorted(d.glob("CollectedData_*.h5"))
        if h5:
            frames.append(pd.read_hdf(h5[0]))
    lab = pd.concat(frames)
    keypoints = list(dict.fromkeys(lab.columns.get_level_values("bodyparts")))
    rows = pd.DataFrame({"video_stem": [i[1] for i in lab.index], "image": [i[2] for i in lab.index]})
    for kp in keypoints:
        for c in ("x", "y"):
            rows[f"{kp}_{c}"] = lab.xs((kp, c), axis=1, level=("bodyparts", "coords")).iloc[:, 0].to_numpy()
    rows["all_occluded"] = False
    extra = [p for p in all_occluded_list() if p.startswith(cam + "_")]
    for p in extra:
        stem, img = p.split("/")
        if ((rows.video_stem == stem) & (rows.image == img)).any():
            rows.loc[(rows.video_stem == stem) & (rows.image == img), "all_occluded"] = True
        else:
            rows.loc[len(rows)] = {"video_stem": stem, "image": img, "all_occluded": True,
                                   **{f"{kp}_{c}": np.nan for kp in keypoints for c in "xy"}}
    man = pd.read_csv(staging_root(rv) / "frame_manifest.csv", dtype={"date": str})
    man = man[man.cam == cam][["video_stem", "image", "animal", "date", "frame", "category", "phase",
                               "trial_id", "position"]].drop_duplicates(["video_stem", "image"])
    rows = rows.merge(man, on=["video_stem", "image"], how="left")
    if rows.animal.isna().any():
        raise ValueError(f"{cam}: {int(rows.animal.isna().sum())} labelled frame(s) not in the manifest")
    rows["frame"] = rows.frame.astype(np.int64)
    rows["src_image"] = [str(project_dir(rv) / "labeled-data" / s / i) for s, i in zip(rows.video_stem, rows.image)]
    return rows.reset_index(drop=True), keypoints


def _video(rv, date, animal, cam) -> Path:
    return sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{cam}_*.avi"))[0]


def _extract(video: Path, frames_to_out: list[tuple[int, Path]]) -> int:
    import cv2
    cap = cv2.VideoCapture(str(video))            # READ-ONLY
    n = 0
    try:
        for f, out in sorted(frames_to_out):
            if out.exists():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, im = cap.read()
            if not ok:
                raise RuntimeError(f"{video.name} frame {f}: read failed")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), im)
            n += 1
    finally:
        cap.release()
    return n


def export(out: Path, rv=None, tol: int = 1) -> dict:
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable

    rv = rv or PathResolver()
    out = Path(out)
    assert_writable(out)
    rows, kps = {}, None
    for v in VIEWS:
        rows[v], k = load_view(v, rv)
        assert kps is None or k == kps, f"keypoint order differs: {kps} vs {k}"
        kps = k
    # every label onto the cam4 clock; and the other camera's frame at the same instant
    for v in VIEWS:
        r = rows[v]
        r["anchor"], r["other_frame"] = 0, 0
        for (animal, date), g in r.groupby(["animal", "date"]):
            tpl = _templates(rv, animal, date)
            other = [w for w in VIEWS if w != v][0]
            r.loc[g.index, "anchor"] = g.frame.to_numpy() if v == ANCHOR else map_frame(g.frame, tpl[v], tpl[ANCHOR])
            r.loc[g.index, "other_frame"] = map_frame(g.frame, tpl[v], tpl[other])
    m = pair_moments(rows, tol=tol)
    n = len(m)
    labelled, present, occl, xy = {}, {}, {}, {}
    for v in VIEWS:
        present[v] = m["ia" if v == VIEWS[0] else "ib"].notna().to_numpy()
        xy[v] = np.stack([np.stack([m[f"{kp}_x_{v}"].to_numpy(float), m[f"{kp}_y_{v}"].to_numpy(float)], -1)
                          for kp in kps], 1)
        labelled[v] = ~np.isnan(xy[v][..., 0])
        occl[v] = m[f"all_occluded_{v}"].eq(True).to_numpy()
    vis = split_visibility(labelled, present, occl)

    # images: copy the labelled ones, extract the other view's frame for unpaired moments
    to_extract: dict[tuple, list] = {}
    copied = 0
    for i, r in m.iterrows():
        for v in VIEWS:
            dst = out / lp_image_path(r.animal, r.date, v, r.anchor)
            if present[v][i]:
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(r[f"src_image_{v}"], dst)
                    copied += 1
            else:
                w = [x for x in VIEWS if x != v][0]
                f = int(r[f"other_frame_{w}"])
                to_extract.setdefault((r.date, r.animal, v), []).append((f, dst))
                m.loc[i, f"frame_{v}"] = f
    extracted = sum(_extract(_video(rv, d, a, v), lst) for (d, a, v), lst in to_extract.items())

    index = {v: [lp_image_path(a, d, v, f) for a, d, f in zip(m.animal, m.date, m.anchor)] for v in VIEWS}
    for v in VIEWS:
        view_frame(xy[v], vis[v], index[v], kps).to_csv(out / f"CollectedData_{v}.csv")
    keep = ["animal", "date", "anchor", "paired", "delta"] + [f"{c}_{v}" for v in VIEWS for c in
                                                               ("video_stem", "image", "frame", "category", "phase",
                                                                "all_occluded")]
    m[[c for c in keep if c in m.columns]].to_csv(out / "moments.csv", index=False)
    rep = {
        "moments": n, "paired": int(m.paired.sum()),
        "unpaired_only": {v: int((present[v] & ~m.paired.to_numpy()).sum()) for v in VIEWS},
        "labelled_rows": {v: int(present[v].sum()) for v in VIEWS},
        "pair_delta_frames": {str(k): int(c) for k, c in m.delta.dropna().astype(int).value_counts().items()},
        "visible_counts": {v: count_codes(vis[v], kps) for v in VIEWS},
        "visible_counts_paired_rows": {v: count_codes(vis[v][m.paired.to_numpy()], kps) for v in VIEWS},
        "one_view_only_on_paired": {kp: int(sum(((vis[v][:, k] == VISIBLE) & (vis[w][:, k] == NOT_LABELLED)
                                                  & m.paired.to_numpy()).sum()
                                                 for v, w in (VIEWS, VIEWS[::-1])))
                                    for k, kp in enumerate(kps)},
        "unpaired_by_category": {v: m.loc[present[v] & ~m.paired.to_numpy(), f"category_{v}"].value_counts().to_dict()
                                 for v in VIEWS},
        "images_copied": copied, "images_extracted": extracted, "tol_frames": tol, "keypoints": kps,
    }
    (out / "export_report.json").write_text(json.dumps(rep, indent=1, default=int), encoding="utf-8")
    return rep


# ------------------------------------------------------------------------------ synced clips

def clip_frames(f4_start: int, n: int, tpl4: dict, tpl1: dict) -> tuple[np.ndarray, np.ndarray]:
    """cam4 frames f4_start .. +n-1 and the DAQ-matched cam1 frame of each."""
    f4 = np.arange(f4_start, f4_start + n, dtype=np.int64)
    return f4, map_frame(f4, tpl4, tpl1)


def _write_clip(video: Path, frames: np.ndarray, dst: Path) -> None:
    import cv2
    import imageio_ffmpeg
    cap = cv2.VideoCapture(str(video))            # READ-ONLY
    lo, hi = int(frames.min()), int(frames.max())
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    buf = {}
    for f in range(lo, hi + 1):
        ok, im = cap.read()
        if not ok:
            raise RuntimeError(f"{video.name}: read failed at {f}")
        buf[f] = im
    cap.release()
    h, w = buf[lo].shape[:2]
    wr = imageio_ffmpeg.write_frames(str(dst), (w, h), fps=250, codec="libx264", quality=None,
                                     pix_fmt_in="bgr24", pix_fmt_out="yuv420p", macro_block_size=1,
                                     output_params=["-crf", "15", "-preset", "veryfast"])
    wr.send(None)
    for f in frames:
        wr.send(np.ascontiguousarray(buf[int(f)]))
    wr.close()


def clips(out: Path, rv=None, dur_s: float = 10.0, pre_s: float = 3.0, seed: int = 92) -> list[str]:
    """One frame-synced cam1/cam4 clip pair per labelled (animal, epoch), cut around a random cue."""
    from wfield_local.dlc_frames import frame_of, staging_root
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable

    rv = rv or PathResolver()
    vd = Path(out) / "videos"
    assert_writable(vd)
    vd.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(staging_root(rv) / "frame_manifest.csv", dtype=str)
    both = set(man[man.cam == "cam1"].date + man[man.cam == "cam1"].animal) & set(
        man[man.cam == "cam4"].date + man[man.cam == "cam4"].animal)
    mm = man[(man.cam == ANCHOR) & (man.date + man.animal).isin(both)]
    picked = (mm.drop_duplicates("video_stem")[["animal", "date", "video_stem", "epoch"]]
              .sort_values(["epoch", "animal"]).groupby(["animal", "epoch"], as_index=False).first())
    rng = np.random.default_rng(seed)
    made = []
    n = int(round(dur_s * 250))
    for r in picked.itertuples():
        tcsv = sorted((Path(rv.root("behavior_out")) / "sessions" / r.animal / r.date).glob("*_trials.csv"))
        if not tcsv:
            print(f"  {r.animal} {r.date}: no trials.csv -> skip")
            continue
        cue = pd.read_csv(tcsv[0])["cue_s"].to_numpy()
        mid = cue[(cue > 60) & (cue < cue.max() - 60)]
        if not len(mid):
            continue
        t0 = float(rng.choice(mid)) - pre_s
        tpl = _templates(rv, r.animal, r.date)
        f4, f1 = clip_frames(frame_of(tpl[ANCHOR], t0, 0.0), n, tpl[ANCHOR], tpl["cam1"])
        name = f"{r.animal}_{r.date}_t{int(t0)}"
        for v, fr in (("cam4", f4), ("cam1", f1)):
            dst = vd / f"{name}_{v}.mp4"
            if not dst.exists():
                _write_clip(_video(rv, r.date, r.animal, v), fr, dst)
        steps = np.diff(f1)
        print(f"  {name}: cam4 {f4[0]}-{f4[-1]}, cam1 {f1[0]}-{f1[-1]} (cam1 steps != 1: {int((steps != 1).sum())})")
        made.append(name)
    return made


# ------------------------------------------------------------------------------ calibration subset

def calib(out: Path, rv=None) -> Path:
    """cam1 + cam4 subset of the 09-11 anipose solve, camera order = VIEWS (LP asserts it)."""
    from aniposelib.cameras import CameraGroup

    from wfield_local.dlc_seed3d import calibration_path
    from wfield_local.writeguard import assert_writable

    src = calibration_path(rv)
    cg = CameraGroup.load(str(src))
    sub = cg.subset_cameras_names(list(VIEWS))
    assert list(sub.get_names()) == list(VIEWS), sub.get_names()
    dst = Path(out) / "calibration.toml"
    assert_writable(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sub.dump(str(dst))
    (Path(out) / "calibration_SOURCE.txt").write_text(str(src) + "\n", encoding="utf-8")
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "clips", "calib"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=int, default=1, help="pairing tolerance in cam4 frames (export)")
    ap.add_argument("--dur", type=float, default=10.0, help="clip length in s (clips)")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(json.dumps(export(a.out, tol=a.tol), indent=1, default=int))
    elif a.cmd == "clips":
        print(clips(a.out, dur_s=a.dur))
    else:
        print(calib(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
