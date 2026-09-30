"""Cue-aligned tongue / jaw x & y traces, trials overlaid: DLC vs Lightning Pose on IDENTICAL frames.

    # 1. cut the cue windows into one clip (+ a frame index), CPU only
    python -m scripts.pose_cue_traces clip PS93:20260908
    # 2. predict it with both models
    python -m scripts.pose_cue_traces dlc  PS93:20260908          # conda activate dlc (Windows GPU)
    #    LP, in WSL:  litpose predict <model_dir> <clip>.mp4 --overrides dali.base.predict.sequence_length=16
    # 3. plot
    python -m scripts.pose_cue_traces plot PS93:20260908 --lp <LP csv>

WHY (Priya, 2026-09-30): "LP looks better to me on PS93 unseen ... plot some cue-aligned tongue and jaw x & y
px location vs time traces trial-overlaid, to see how noisy DLC vs LP are."

STYLE is ported from `stroke_orofacial_pipeline/src/stroke_orofacial/dlc_kinematics/mean_sem_visual.py`
`_plot_trial_overlays_2x2` (thin per-trial lines, mean line in the same colour, black line at 0, px vs ms,
y NOT inverted). What differs, on purpose:
  * one moving spout, not L/R -> columns are the two MODELS (DLC | LP), rows are tongue Y, tongue X,
    jaw Y, jaw X, plus a row with the fraction of trials in which each part is confident;
  * NO cleaning: their pipeline drops low-likelihood/isolated points, PCHIP-fills gaps and baseline-fills
    the rest before plotting, which would hide exactly the noise this figure is for. Here a frame below
    likelihood 0.6 (`dlc.train.pcutoff`) is simply a gap; nothing is interpolated or filtered;
  * both models see the SAME frames: the windows are re-encoded once into one mp4 and both predict that
    file (DLC's own review CSVs are median-filtered; here neither is).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import dlc_frames as DF
from wfield_local.paths import PathResolver

WIN_S = (-0.5, 3.5)
PER_POSITION = 4
SEED = 908
PCUT = 0.6
PARTS = ("tongue", "jaw")
COLORS = {"DLC": "dodgerblue", "LP": "mediumvioletred"}     # stroke_orofacial side_colors, reused per model


def _out(rv, animal, date) -> Path:
    from wfield_local import dlc_project
    d = dlc_project.project_dir(rv).parent / "lp_vs_dlc_cue_traces" / f"{animal}_{date}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session(spec, rv):
    animal, date = spec.split(":")
    tpl = dict(np.load(Path(rv.root("alignment_templates")) / "cam4" / animal / f"{date}.npz", allow_pickle=True))
    vid = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob("cam4_*.avi"))[0]
    t = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob(f"{animal}_{date}_*_trials.csv"))[-1]
    return animal, date, tpl, vid, pd.read_csv(t)


def cmd_clip(spec, rv):
    import cv2
    animal, date, tpl, vid, t = _session(spec, rv)
    t = t[np.isfinite(t["cue_s"].astype(float))]
    rng = np.random.default_rng(SEED)
    tr = pd.concat([g.sample(min(len(g), PER_POSITION), random_state=int(rng.integers(1 << 31)))
                    for _, g in t.groupby("pos_name")]).sort_values("cue_s")
    out = _out(rv, animal, date)
    cap = cv2.VideoCapture(str(vid))                     # READ-ONLY
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = out / "cue_windows_raw.avi"
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"FFV1"), 250.0, (w, h))
    idx = []
    for k, (_, r) in enumerate(tr.iterrows()):
        f0 = DF.frame_of(tpl, r.cue_s, WIN_S[0])
        n = int(round((WIN_S[1] - WIN_S[0]) * 250))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for j in range(n):
            ok, im = cap.read()
            if not ok:
                break
            wr.write(im)
            idx.append({"trial_k": k, "trial_id": int(r.trial_id), "position": r.pos_name,
                        "src_frame": f0 + j, "t_ms": round((j / 250 + WIN_S[0]) * 1000, 3)})
    cap.release()
    wr.release()
    pd.DataFrame(idx).to_csv(out / "cue_windows_index.csv", index=False)
    print(f"{len(tr)} trials, {len(idx)} frames -> {raw}\n  now: ffmpeg -i {raw.name} -c:v libx264 -crf 12 -pix_fmt yuv420p cue_windows.mp4")


def cmd_dlc(spec, rv):
    import cv2
    from wfield_local import dlc_prior
    from wfield_local.dlc_hard_frames import PARTS as ALL, pose_predictor
    animal, date, *_ = _session(spec, rv)
    out = _out(rv, animal, date)
    predict = pose_predictor(rv)
    cap = cv2.VideoCapture(str(out / "cue_windows.mp4"))
    poses, buf = [], []
    with dlc_prior.apply(rv, cam="cam4"):
        while True:
            ok, im = cap.read()
            if ok:
                buf.append(im)
            if len(buf) == 64 or (not ok and buf):
                poses.append(predict(buf))
                buf = []
            if not ok:
                break
    P = np.concatenate(poses)
    cols = pd.MultiIndex.from_product([["DLC_round3"], ALL, ["x", "y", "likelihood"]], names=["scorer", "bodyparts", "coords"])
    pd.DataFrame(P.reshape(len(P), -1), columns=cols).to_csv(out / "cue_windows_DLC.csv")
    print(f"DLC: {len(P)} frames -> {out / 'cue_windows_DLC.csv'}")


def _load(csv):
    d = pd.read_csv(csv, header=[0, 1, 2], index_col=0)
    d.columns = d.columns.droplevel(0)
    return d


def median5(d: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """`deeplabcut.filterpredictions` defaults (filtertype='median', windowlength=5): a median filter on
    each part's x and y over the whole trace, likelihood untouched. Applied to BOTH models so the filtered
    comparison is like for like (DLC's review CSVs were filtered this way; LP's output is raw)."""
    from scipy.signal import medfilt
    out = d.copy()
    for part in {c[0] for c in d.columns}:
        for coord in ("x", "y"):
            out[(part, coord)] = medfilt(d[(part, coord)].to_numpy(float), window)
    return out


def cmd_plot(spec, rv, lp_csv, filtered: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    animal, date, *_ = _session(spec, rv)
    out = _out(rv, animal, date)
    idx = pd.read_csv(out / "cue_windows_index.csv")
    M = {"DLC": _load(out / "cue_windows_DLC.csv"), "LP": _load(lp_csv)}
    if filtered:
        M = {k: median5(v) for k, v in M.items()}
    tag = "median-5 filtered (DLC's filterpredictions default), both models" if filtered else "no cleaning"
    n = min(len(idx), *(len(m) for m in M.values()))
    t_ms = np.sort(idx.t_ms.unique())
    K = idx.trial_k.nunique()

    def stack(m, part, coord):
        a = m[part][coord].to_numpy()[:n].astype(float)
        a = np.where(m[part]["likelihood"].to_numpy()[:n] > PCUT, a, np.nan) if coord != "likelihood" else a
        A = np.full((K, len(t_ms)), np.nan)
        A[idx.trial_k[:n], np.searchsorted(t_ms, idx.t_ms[:n])] = a
        return A

    rows = [("tongue", "y"), ("tongue", "x"), ("jaw", "y"), ("jaw", "x")]
    fig, axs = plt.subplots(len(rows) + 1, 2, figsize=(12, 13), sharex=True)
    for c, name in enumerate(("DLC", "LP")):
        col = COLORS[name]
        for r, (part, coord) in enumerate(rows):
            ax = axs[r, c]
            A = stack(M[name], part, coord)
            for row in A:
                ax.plot(t_ms, row, lw=0.6, alpha=0.35, color=col)
            ax.plot(t_ms, np.nanmean(A, 0), lw=1, color="k", alpha=0.8, label="mean")
            ax.axvline(0, color="k", lw=1)
            ax.set_ylabel(f"{part.capitalize()} {coord.upper()} (px)")
            if r == 0:
                ax.set_title(f"{name} | {K} trials | points with p > {PCUT}, {tag}", fontsize=11)
        ax = axs[-1, c]
        for part, ls in (("tongue", "-"), ("jaw", "--")):
            L = stack(M[name], part, "likelihood")
            ax.plot(t_ms, np.nanmean(L > PCUT, 0), ls, color=col, lw=1, label=part)
        ax.axvline(0, color="k", lw=1)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction of trials\nconfident")
        ax.set_xlabel("Time relative to cue (ms)")
        ax.legend(fontsize=8, loc="upper left")
    for r in range(len(rows)):                           # same y-range per row across the two models
        lo = min(a.get_ylim()[0] for a in axs[r]); hi = max(a.get_ylim()[1] for a in axs[r])
        for a in axs[r]:
            a.set_ylim(lo, hi)
    fig.suptitle(f"{animal} cam4 | {date} | cue-aligned, trials overlaid | DLC round 3 vs LP occlusion (ep185) | "
                 f"image y increases downward", y=1.0, fontsize=12)
    fig.tight_layout()
    p = out / f"{animal}_{date}_cue_traces_DLC_vs_LP{'_median5' if filtered else ''}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"-> {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["clip", "dlc", "plot"])
    ap.add_argument("session", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--lp", type=Path, help="LP predictions CSV for the clip (plot)")
    ap.add_argument("--filtered", action="store_true", help="apply the median-5 filter to both models first")
    a = ap.parse_args(argv)
    rv = PathResolver()
    {"clip": lambda: cmd_clip(a.session, rv), "dlc": lambda: cmd_dlc(a.session, rv),
     "plot": lambda: cmd_plot(a.session, rv, a.lp, a.filtered)}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
