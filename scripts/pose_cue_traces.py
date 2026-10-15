"""Cue-aligned tongue / jaw x & y traces, trials overlaid: DLC vs Lightning Pose on IDENTICAL frames.

    # 1. cut the cue windows into one clip (+ a frame index), CPU only
    python -m scripts.pose_cue_traces clip PS93:20260908
    # 2. predict it with both models
    python -m scripts.pose_cue_traces dlc  PS93:20260908          # conda activate dlc (Windows GPU)
    #    LP, in WSL:  litpose predict <model_dir> <clip>.mp4 --overrides dali.base.predict.sequence_length=16
    # 3. plot
    python -m scripts.pose_cue_traces plot PS93:20260908 --lp <LP csv> [--filtered]
    python -m scripts.pose_cue_traces zoom PS93:20260908 --lp <LP csv>     # 4 trials, 0-1 s, both models per axis

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
#: Below-cutoff points get their OWN colours (Priya, 2026-10-01): drawn in the model colour they read as
#: tracking -- LP's are all the image centre (~338, 338 px; a flat "occluded" heatmap's soft-argmax), which
#: looked like a flat trace between licks.
LOWCONF_COLORS = {"DLC": "green", "LP": "grey"}


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


def cleaned_frame(d: pd.DataFrame, idx: pd.DataFrame, mouth=None, fill: bool = False) -> pd.DataFrame:
    """Tongue/jaw replaced by `orofacial_clean` output (baseline-subtracted px; baseline-filled frames -> NaN,
    likelihood 1 where a cleaned value exists, 0 where it does not), so `cmd_plot` draws the CLEAN traces."""
    from wfield_local import orofacial_clean as oc
    flat = pd.DataFrame({f"{p}_{c}": d[p][c].to_numpy(float) for p in ("tongue", "jaw")
                         for c in ("x", "y", "likelihood")})
    out = d.copy()
    for part in ("tongue", "jaw"):
        # TONGUE ZERO = the MOUTH (spout-frame origin) when given, as in `pose_kinematics_demo.run` (Priya 2026-10-01);
        # jaw keeps the data-driven baseline
        c, pos, _ = oc.clean_windows(flat, idx.iloc[:len(flat)], part,
                                     x0y0=tuple(mouth) if (part == "tongue" and mouth is not None) else None)
        # masked (default): baseline-filled frames -> NaN, only measured / interpolated points drawn. fill=True: the
        # stroke_orofacial convention -- the baseline fill drawn as the flat line at 0 (tongue in the mouth / jaw at rest)
        x, y = (c.x_final[pos], c.y_final[pos]) if fill else (c.x_masked[pos], c.y_masked[pos])
        out[(part, "x")], out[(part, "y")] = x, y
        out[(part, "likelihood")] = np.isfinite(y).astype(float)
    return out


def cmd_plot(spec, rv, lp_csv, filtered: bool = False, cleaned: bool = False):
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
    if cleaned:
        M = {k: cleaned_frame(v, idx) for k, v in M.items()}
        tag = "CLEANED (v5p3 + jaw v3.4), px rel. baseline"
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
        ax.set_ylabel("fraction of trials\n" + ("with a cleaned value" if cleaned else "confident"))
        ax.set_xlabel("Time relative to cue (ms)")
        ax.legend(fontsize=8, loc="upper left")
    for r in range(len(rows)):                           # same y-range per row across the two models
        lo = min(a.get_ylim()[0] for a in axs[r]); hi = max(a.get_ylim()[1] for a in axs[r])
        for a in axs[r]:
            a.set_ylim(lo, hi)
    fig.suptitle(f"{animal} cam4 | {date} | cue-aligned, trials overlaid | DLC round 3 vs LP occlusion (ep185) | "
                 f"image y increases downward", y=1.0, fontsize=12)
    fig.tight_layout()
    p = out / f"{animal}_{date}_cue_traces_DLC_vs_LP{'_median5' if filtered else ''}{'_cleaned' if cleaned else ''}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"-> {p}")


def cmd_zoom(spec, rv, lp_csv, n_trials: int = 4, t1_ms: float = 1000.0):
    """Both models on the SAME axes for a few trials, 0..t1 ms after the cue -- the trial overlay is too dense
    (24 x 1000 frames) to show single-frame jitter. Trials = those with the most frames where both models see
    the tongue. Also writes the LP-vs-DLC tongue distance (both confident) by spout position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    animal, date, *_ = _session(spec, rv)
    out = _out(rv, animal, date)
    idx = pd.read_csv(out / "cue_windows_index.csv")
    D, L = _load(out / "cue_windows_DLC.csv"), _load(lp_csv)
    n = min(len(idx), len(D), len(L))
    idx, D, L = idx.iloc[:n], D.iloc[:n], L.iloc[:n]
    both_t = (D["tongue"]["likelihood"].values > PCUT) & (L["tongue"]["likelihood"].values > PCUT)
    dist = np.hypot(D["tongue"]["x"].values - L["tongue"]["x"].values, D["tongue"]["y"].values - L["tongue"]["y"].values)
    by_pos = (pd.DataFrame({"position": idx.position, "d": np.where(both_t, dist, np.nan)})
              .groupby("position").d.agg(frames_both="count", median_px="median",
                                         p90_px=lambda s: s.quantile(0.9), frac_gt15=lambda s: (s > 15).mean()).round(2))
    by_pos.to_csv(out / "tongue_LP_vs_DLC_by_position.csv")
    print(by_pos.to_string())
    win = (idx.t_ms >= 0) & (idx.t_ms <= t1_ms)
    score = pd.Series(both_t & win.values).groupby(idx.trial_k.values).sum().sort_values(ascending=False)
    ks = list(score.index[:n_trials])
    rows = [("tongue", "y"), ("tongue", "x"), ("jaw", "y"), ("jaw", "x")]
    fig, axs = plt.subplots(len(rows), len(ks), figsize=(4 * len(ks), 10), sharex=True, squeeze=False)
    for c, k in enumerate(ks):
        m = (idx.trial_k == k).values & win.values
        t = idx.t_ms.values[m]
        pos = idx.position.values[m][0]
        for r, (part, coord) in enumerate(rows):
            ax = axs[r, c]
            for name, X in (("DLC", D), ("LP", L)):
                v = X[part][coord].values[m].astype(float)
                ok = X[part]["likelihood"].values[m] > PCUT
                ax.plot(t, np.where(ok, v, np.nan), lw=1, color=COLORS[name], label=name)
                ax.plot(t[~ok], v[~ok], ".", ms=2, color=LOWCONF_COLORS[name], alpha=0.5,
                        label=f"{name} p <= {PCUT}")
            ax.axvline(0, color="k", lw=1)
            if c == 0:
                ax.set_ylabel(f"{part.capitalize()} {coord.upper()} (px)")
            if r == 0:
                ax.set_title(f"trial {int(idx.trial_id.values[m][0])} | {pos}", fontsize=10)
        axs[-1, c].set_xlabel("Time relative to cue (ms)")
    axs[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{animal} cam4 | {date} | DLC round 3 (blue) vs LP occlusion (magenta), raw | line = p > {PCUT}, "
                 f"dots = below cutoff (DLC green, LP grey; LP below cutoff is always the image centre) | image y increases downward", y=1.0, fontsize=11)
    fig.tight_layout()
    p = out / f"{animal}_{date}_cue_traces_zoom_DLC_vs_LP.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"-> {p}")


def cmd_onlyone(spec, rv, lp_csv, per_group: int = 8, seed: int = 1):
    """Contact sheet of frames where only ONE model is confident (DLC-only / LP-only, tongue and jaw), with
    both models' points drawn, to judge who is right (Priya, 2026-10-01: "in some instances DLC is carrying
    valuable information that LP lost"). Filled marker = that model confident; hollow = below cutoff (LP's
    hollow marker is then the image centre and is not drawn)."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    animal, date, *_ = _session(spec, rv)
    out = _out(rv, animal, date)
    idx = pd.read_csv(out / "cue_windows_index.csv")
    D, L = _load(out / "cue_windows_DLC.csv"), _load(lp_csv)
    n = min(len(idx), len(D), len(L))
    rng = np.random.default_rng(seed)
    groups = []
    for part in ("jaw", "tongue"):
        d = D[part]["likelihood"].values[:n] > PCUT
        l_ = L[part]["likelihood"].values[:n] > PCUT
        for name, m in ((f"{part}: DLC only", d & ~l_), (f"{part}: LP only", ~d & l_)):
            cand = np.flatnonzero(m)
            pick = np.sort(rng.choice(cand, min(per_group, len(cand)), replace=False)) if len(cand) else []
            groups.append((name, part, int(m.sum()), pick))
    cap = cv2.VideoCapture(str(out / "cue_windows.mp4"))
    fig, axs = plt.subplots(len(groups), per_group, figsize=(2.2 * per_group, 2.5 * len(groups)), squeeze=False)
    for r, (name, part, total, pick) in enumerate(groups):
        for c in range(per_group):
            ax = axs[r, c]
            ax.axis("off")
            if c >= len(pick):
                continue
            i = int(pick[c])
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, im = cap.read()
            if not ok:
                continue
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)[250:640, 170:510])          # mouth region
            for mname, X, mk in (("DLC", D, "o"), ("LP", L, "X")):
                x, y, p = X[part]["x"].values[i], X[part]["y"].values[i], X[part]["likelihood"].values[i]
                if mname == "LP" and p <= PCUT:
                    continue                                   # LP below cutoff = image centre, meaningless
                ax.plot(x - 170, y - 250, mk, ms=7, mfc=COLORS[mname] if p > PCUT else "none",
                        mec=COLORS[mname], mew=1.5)
            ax.set_title(f"{idx.t_ms.values[i]:.0f} ms  D{D[part]['likelihood'].values[i]:.2f} "
                         f"L{L[part]['likelihood'].values[i]:.2f}", fontsize=7)
        axs[r, 0].text(-0.08, 0.5, f"{name}\n({total} fr)", transform=axs[r, 0].transAxes, ha="right",
                       va="center", fontsize=9)
    cap.release()
    fig.suptitle(f"{animal} {date}: frames where only one model is confident | DLC o blue, LP x magenta | "
                 f"filled = p > {PCUT}, hollow = below | title: ms from cue, D/L likelihood", fontsize=10)
    fig.tight_layout()
    p = out / f"{animal}_{date}_only_one_confident.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(f"-> {p}")


MODEL_COLORS = ["dodgerblue", "navy", "mediumvioletred", "darkorange", "seagreen", "black"]


def cmd_compare(poses_dir: Path, models: list[str], cleaned: bool = False, win_s=(-0.5, 3.5), tag: str = "",
                suffix: str = "", fill: bool = False, n_trials: int = 0):
    """Cue-aligned tongue / jaw x & y, trials overlaid, for ANY number of models on the SAME clip -- a
    `session_poses` directory (windows.mp4 + windows_index.csv) and ``name=csv`` pairs (Priya, 2026-10-13: "compare
    how the tongue and jaw cue-aligned trajectory plots look for all 4 (round 3/4, DLC/LP)"). Same rules as
    `cmd_plot`: below-cutoff points are gaps, nothing interpolated; ``cleaned`` swaps in `orofacial_clean` output.
    ``n_trials`` > 0 (Priya, 2026-10-15: "show a handful of trials for each model, so I can see whether they look
    plausible"): only that many trials, the SAME ones for every model (one per spout position first, fixed seed), each
    in its own colour, no mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    poses_dir = Path(poses_dir)
    idx = pd.read_csv(poses_dir / "windows_index.csv")
    M = {}
    for spec in models:
        name, csv = spec.split("=", 1)
        M[name] = _load(csv)
    mouth = None
    if cleaned:
        # ONE mouth for every model (from round-3 DLC's spout, as the kinematics pipeline does), so a model
        # difference in the cleaned tongue is tracking, not a different zero
        from wfield_local import orofacial_clean as oc
        from wfield_local import spout_frame as SF
        ref = oc.read_pose(poses_dir / "windows_DLC.csv").iloc[:len(idx)]
        spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0]))
                 for _, g in idx.iloc[:len(ref)].groupby("trial_k")]
        mouth = SF.from_medians(SF.position_medians(ref[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(),
                                                    spans)).origin
        M = {k: cleaned_frame(v, idx, mouth, fill=fill) for k, v in M.items()}
    n = min(len(idx), *(len(m) for m in M.values()))
    keep = (idx.t_ms[:n] >= win_s[0] * 1000) & (idx.t_ms[:n] <= win_s[1] * 1000)
    sub = idx[:n][keep]
    t_ms = np.sort(sub.t_ms.unique())
    K = int(idx.trial_k.max()) + 1
    show = np.arange(K)
    if n_trials:
        rng = np.random.default_rng(3)
        per = sub.groupby("trial_k").position.first()
        order = [int(rng.choice(per.index[per == p_])) for p_ in sorted(per.unique())]
        rest = [k for k in rng.permutation(per.index.to_numpy()) if k not in order]
        show = np.array((order + rest)[:n_trials])
        tlab = {k: f"t{int(idx.trial_id[idx.trial_k == k].iloc[0])} {per[k]}" for k in show}

    def stack(m, part, coord):
        a = m[part][coord].to_numpy()[:n].astype(float)
        if coord != "likelihood":
            a = np.where(m[part]["likelihood"].to_numpy()[:n] > PCUT, a, np.nan)
        A = np.full((K, len(t_ms)), np.nan)
        A[sub.trial_k, np.searchsorted(t_ms, sub.t_ms)] = a[keep.to_numpy()]
        return A

    rows = [("tongue", "y"), ("tongue", "x"), ("jaw", "y"), ("jaw", "x")]
    names = list(M)
    fig, axs = plt.subplots(len(rows) + 1, len(names), figsize=(4.2 * len(names), 13), sharex=True, squeeze=False)
    for c, name in enumerate(names):
        col = MODEL_COLORS[c % len(MODEL_COLORS)]
        for r, (part, coord) in enumerate(rows):
            ax = axs[r, c]
            A = stack(M[name], part, coord)
            if n_trials:
                for j, k in enumerate(show):
                    ax.plot(t_ms, A[k], lw=0.9, color=plt.cm.tab10(j % 10), label=tlab[k])
                if r == 0 and c == 0:
                    ax.legend(fontsize=7, loc="upper right")
            else:
                for row in A:
                    ax.plot(t_ms, row, lw=0.5, alpha=0.3, color=col)
                ax.plot(t_ms, np.nanmean(A, 0), lw=1, color="k", alpha=0.8)
            ax.axvline(0, color="k", lw=1)
            if c == 0:
                ref_name = ("mouth" if part == "tongue" else "baseline") if cleaned else ""
                ax.set_ylabel(f"{part.capitalize()} {coord.upper()} (px{' rel. ' + ref_name if cleaned else ''})")
            if r == 0:
                ax.set_title(f"{name}\n{len(show)} trials, p > {PCUT}", fontsize=10)
        ax = axs[-1, c]
        for part, ls in (("tongue", "-"), ("jaw", "--")):
            L = stack(M[name], part, "likelihood")
            ax.plot(t_ms, np.nanmean(L > PCUT, 0), ls, color=col, lw=1, label=part)
        ax.axvline(0, color="k", lw=1)
        ax.set_ylim(0, 1.02)
        if c == 0:
            ax.set_ylabel("fraction of trials\n" + ("with a cleaned value" if cleaned else "confident"))
        ax.set_xlabel("time from cue (ms)")
        ax.legend(fontsize=8, loc="upper left")
    for r in range(len(rows)):
        lo = min(a.get_ylim()[0] for a in axs[r]); hi = max(a.get_ylim()[1] for a in axs[r])
        for a in axs[r]:
            a.set_ylim(lo, hi)
    fig.suptitle(f"{poses_dir.name} cam4 | cue-aligned, trials overlaid | "
                 f"{('CLEANED, baseline fill drawn (flat 0)' if fill else 'CLEANED (orofacial_clean), baseline fill hidden') if cleaned else 'raw, no cleaning'} | image y increases downward"
                 + (f" | {tag}" if tag else ""), y=1.0, fontsize=12)
    fig.tight_layout()
    p = poses_dir / "r4_eval" / (f"cue_traces_compare{suffix}{'_cleaned' if cleaned else ''}{'_filled' if fill else ''}"
                                 f"{f'_{n_trials}trials' if n_trials else ''}.png")
    p.parent.mkdir(exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"-> {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["clip", "dlc", "plot", "zoom", "onlyone", "compare"])
    ap.add_argument("session", metavar="ANIMAL:YYYYMMDD", help="for `compare`: a session_poses directory")
    ap.add_argument("--lp", type=Path, help="LP predictions CSV for the clip (plot)")
    ap.add_argument("--filtered", action="store_true", help="apply the median-5 filter to both models first")
    ap.add_argument("--cleaned", action="store_true", help="plot orofacial_clean output (ported stroke_orofacial cleaning)")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=CSV",
                    help="compare: one model's predictions on the clip (repeat, in column order)")
    ap.add_argument("--suffix", default="", help="compare: added to the output filename (keep several comparisons)")
    ap.add_argument("--trials", type=int, default=0, help="compare: show only N trials (same for every model)")
    ap.add_argument("--fill", action="store_true", help="compare --cleaned: draw the baseline fill (flat 0) as well")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        cmd_compare(Path(a.session), a.model, cleaned=a.cleaned, suffix=a.suffix, fill=a.fill, n_trials=a.trials)
        return 0
    rv = PathResolver()
    {"clip": lambda: cmd_clip(a.session, rv), "dlc": lambda: cmd_dlc(a.session, rv),
     "plot": lambda: cmd_plot(a.session, rv, a.lp, a.filtered, a.cleaned), "zoom": lambda: cmd_zoom(a.session, rv, a.lp), "onlyone": lambda: cmd_onlyone(a.session, rv, a.lp)}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
