"""Does the video's pre-cue (ENL) position information come from the MOUSE or from the SPOUT?

    python -m scripts.enl_video_spout_check PS93:20260814 [--cams cam1 cam2 cam3 cam4] [--workers 4]

Priya, 2026-10-07: video + running alone decode spout position 96 % in the 2 s pre-cue window
(`enl_movement_removed_decode`), and removing video drops neural ENL decoding 0.40 -> 0.26. Is that the
animal (posture / head / whisking set per position: genuine, rightly removed) or the apparatus (spout still
settling after the strobe, holding vibration, whiskers on a spout whose location differs)? "we should be able
to do #3, using a mask around the spout position only for each trial's position, since the spout is not moving
during the ENL."

Per camera, ENL windows only (2 s ending at the cue, no DAQ lick inside, spout strobe at least 2 s before the
cue -- the spout is stationary there):
  SPOUT MASKS from the data: the mean ENL frame per position minus the mean over positions; pixels that differ
  strongly (|diff| > thr x robust SD of the difference image) are where THAT position's spout sits; dilated.
    union        the six positions' spout pixels OR-ed, applied to EVERY trial (no mask-induced position cue)
    per_position each trial masks only its own position's spout (Priya's version; the masked-pixel SET differs
                 by position, so a decoder could in principle read the mask -- compared against `union`)
  MOTION ENERGY per frame (grey, ds x area-averaged, |f_t - f_(t-1)|), averaged in decode.bins.precue bins;
  masked pixels set to the across-trial pixel mean (union: 0 variance -> no information).
  DECODE position (PCA 30 per bin fitted inside each CV fold -> logistic C 0.5, trial blocks as groups), per
  camera x {none, union, per_position}, all bins and per bin; null = labels shuffled with fixed predictions.
Outputs: session_poses/<a>_<d>_full/enl_video_spout_check.csv + enl_video_spout_masks.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]


def enl_trials(animal, date, rv, win_s=2.0):
    """Trials with a clean ENL window: (cue_s, pos_name, block id)."""
    from wfield_local import dlc_frames
    from wfield_local import trial_windows as TW
    b = TW.trial_bounds(animal, date, rv)
    sid = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob("*_trials.csv"))[-1].name[:-11]
    licks = np.sort(dlc_frames.lick_onsets(animal, date, sid, rv))
    ok = np.isfinite(b.cue_s) & np.isfinite(b.strobe_s) & (b.strobe_s <= b.cue_s - win_s)
    b = b[ok].copy()
    lo, hi = np.searchsorted(licks, b.cue_s - win_s), np.searchsorted(licks, b.cue_s)
    b = b[(hi - lo) == 0].reset_index(drop=True)
    b["block"] = (b.pos_name != b.pos_name.shift()).cumsum()
    return b


def _worker(cam, video, tpl, cues, win_s, ds, n_bins):
    """(per-trial binned motion energy (n_trial, n_bins, n_px), per-trial mean frame (n_trial, n_px), shape)."""
    import cv2

    from wfield_local import video_motion as VM
    cap = cv2.VideoCapture(str(video))
    fs, sl, ic = float(tpl["fs_daq"]), float(tpl["slope_daqSample_per_camFrame"]), float(tpl["intercept_daqSample"])
    me_out, mean_out, shape = [], [], None
    for c in cues:
        f0 = int(round(((c - win_s) * fs - ic) / sl))
        f1 = int(round((c * fs - ic) / sl))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0 - 1)
        prev, mes, acc = None, [], None
        for _ in range(f0 - 1, f1):
            ok, im = cap.read()
            if not ok:
                break
            g = VM.downsample(im, ds)
            shape = g.shape
            acc = g.astype(np.float64) if acc is None else acc + g
            if prev is not None:
                mes.append(np.abs(g - prev).ravel())
            prev = g
        mes = np.array(mes, np.float32)
        edges = np.linspace(0, len(mes), n_bins + 1).round().astype(int)
        me_out.append(np.stack([mes[a:b].mean(0) for a, b in zip(edges[:-1], edges[1:])]))
        mean_out.append((acc / (f1 - f0 + 1)).ravel().astype(np.float32))
    cap.release()
    return cam, np.array(me_out), np.array(mean_out), shape


def spout_masks(mean_frames, pos, shape, thr=6.0, dilate_px=2):
    """{position: bool mask (n_px)} where that position's mean ENL frame departs from the all-position mean."""
    import cv2
    grand = mean_frames.mean(0)
    out = {}
    for p in POS:
        m = pos == p
        if m.sum() < 3:
            continue
        d = mean_frames[m].mean(0) - grand
        mad = np.median(np.abs(d - np.median(d))) * 1.4826 + 1e-6
        mk = (np.abs(d) > thr * mad).reshape(shape).astype(np.uint8)
        mk = cv2.dilate(mk, np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8))
        out[p] = mk.ravel().astype(bool)
    return out


def decode(F, y, groups, n_pc=30, n_perm=1000):
    """F (n_trial, n_bins, n_px): PCA per bin inside each fold -> logistic. Returns (acc, null mean, p)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(F, y, groups):
        Xtr, Xte = [], []
        for b in range(F.shape[1]):
            k = min(n_pc, len(tr) - 1, F.shape[2])
            pca = PCA(k, random_state=0).fit(F[tr, b])
            Xtr.append(pca.transform(F[tr, b]))
            Xte.append(pca.transform(F[te, b]))
        Xtr, Xte = np.hstack(Xtr), np.hstack(Xte)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xtr), y[tr])
        pred[te] = clf.predict(sc.transform(Xte))
    acc = float((pred == y).mean())
    rng = np.random.default_rng(0)
    null = np.array([(pred == rng.permutation(y)).mean() for _ in range(n_perm)])
    return acc, float(null.mean()), float((np.sum(null >= acc) + 1) / (n_perm + 1))


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from concurrent.futures import ProcessPoolExecutor

    import matplotlib.pyplot as plt

    from scripts.session_poses import session_dir
    from wfield_local import config
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="animal:date")
    ap.add_argument("--cams", nargs="+", default=["cam1", "cam2", "cam3", "cam4"])
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--thr", type=float, default=6.0, help="spout-mask threshold (x robust SD)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    animal, date = a.session.split(":")[:2]
    win_s = float(config.defaults()["decode"]["precue_post_s"])
    n_bins = int(config.defaults()["decode"]["bins"]["precue"])
    tr = enl_trials(animal, date, rv, win_s)
    y, groups = tr.pos_name.to_numpy(), tr.block.to_numpy()
    print(f"{len(tr)} clean ENL trials: {tr.pos_name.value_counts().to_dict()}", flush=True)
    vids = {c: sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{c}_*.avi"))[0] for c in a.cams}
    tpls = {c: dict(np.load(Path(rv.root("alignment_templates")) / c / animal / f"{date}.npz", allow_pickle=True))
            for c in a.cams}
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_worker, c, str(vids[c]), tpls[c], tr.cue_s.to_numpy(), win_s, a.ds, n_bins) for c in a.cams]
        for fu in futs:
            cam, me, mf, shape = fu.result()
            res[cam] = (me, mf, shape)
            print(f"{cam}: motion energy {me.shape}", flush=True)
    out_dir = session_dir(rv, f"{animal}:{date}:full")[2]
    rows = []
    fig, axs = plt.subplots(2, len(a.cams), figsize=(4.2 * len(a.cams), 7), squeeze=False)
    for j, cam in enumerate(a.cams):
        me, mf, shape = res[cam]
        masks = spout_masks(mf, y, shape, thr=a.thr)
        union = np.any(np.stack(list(masks.values())), 0)
        axs[0, j].imshow(mf.mean(0).reshape(shape), cmap="gray")
        axs[0, j].imshow(np.ma.masked_where(~union.reshape(shape), union.reshape(shape)), cmap="autumn", alpha=0.6)
        axs[0, j].set_title(f"{cam}: union spout mask ({union.mean():.1%} of pixels)", fontsize=8)
        tint = np.zeros(shape + (3,))
        for k, p in enumerate(POS):
            if p in masks:
                tint[masks[p].reshape(shape)] = plt.get_cmap("tab10")(k)[:3]
        axs[1, j].imshow(tint)
        axs[1, j].set_title("per-position spout masks (colour = position)", fontsize=8)
        for ax in axs[:, j]:
            ax.axis("off")
        pixmean = me.mean(0, keepdims=True)
        variants = {"none": me}
        u = me.copy()
        u[:, :, union] = pixmean[:, :, union]
        variants["union"] = u
        pp = me.copy()
        for p, mk in masks.items():
            sel = y == p
            pp[np.ix_(sel, np.arange(me.shape[1]), np.flatnonzero(mk))] = pixmean[:, :, mk]
        variants["per_position"] = pp
        for vname, F in variants.items():
            acc, nm, pv = decode(F, y, groups)
            rows.append({"cam": cam, "mask": vname, "bins": "all", "acc": acc, "null": nm, "p": pv,
                         "masked_frac": {"none": 0.0, "union": float(union.mean())}.get(vname, np.nan)})
            print(f"  {cam} {vname:12s} all bins: acc {acc:.3f} (null {nm:.3f}, p={pv:.4f})", flush=True)
            for b in range(F.shape[1]):
                acc_b, nm_b, pv_b = decode(F[:, [b]], y, groups, n_perm=200)
                rows.append({"cam": cam, "mask": vname, "bins": f"bin{b}", "acc": acc_b, "null": nm_b, "p": pv_b})
            print("    per 0.5 s bin (earliest first): " + ", ".join(
                f"{r['acc']:.2f}" for r in rows[-F.shape[1]:]), flush=True)
    fig.suptitle(f"{animal} {date}: spout masks from the data (mean ENL frame per position vs all)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "enl_video_spout_masks.png", dpi=100)
    pd.DataFrame(rows).to_csv(out_dir / "enl_video_spout_check.csv", index=False)
    print("->", out_dir / "enl_video_spout_check.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
