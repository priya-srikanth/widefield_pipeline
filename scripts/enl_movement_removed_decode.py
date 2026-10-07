"""Is spout position decodable from the PRE-CUE (ENL) window after movement is regressed out?

    python -m scripts.enl_movement_removed_decode PS93:20260814:full [--video cam1 cam2 cam3 cam4] [--n-perm 1000]

Priya, 2026-10-07: "it's not surprising for video to capture all position-specific signal for the executed
position-specific task. what if we analyze only our ENL window (0-2 s before cue, licks excluded, as for other ENL
analyses)?" The whole-trial residual test (`movement_position_angle`) found almost nothing left once video was
in; there, the window spans cue and licking, where the target IS the movement. In the ENL window the spout is
already in position (after the strobe) and the animal is not licking, so position information there is
preparatory / sensory rather than executed.

SAME TRIALS AND FEATURES AS THE PROJECT'S ENL DECODERS: `locanmf_position_decoder._trial_features` with
align = precue (2 s ending at the cue, LICK-FREE -- slides earlier to the latest lick-free gap, bounded at the
spout strobe; no clean window -> dropped), `decode.bins.precue` sub-bins, engaged trials, position labels, trial
blocks as CV groups, logistic regression C = 0.5 (`decode` config). Only the SIGNAL fed in differs:

  raw          LocaNMF dF/F (the existing ENL decoder's input)
  resid_dlc    minus a cross-fitted movement model of DLC lick events + continuous tongue / jaw
  resid_run    ... + DAQ running speed
  resid_full   ... + video motion energy (blanked outside position strobe .. trial end)
  video_only   the movement regressors themselves (video PCs + running) as the decoder input -- how much
               position information the BODY carries in the ENL window (posture / whisking toward the spout)

Movement models are fit on the whole session (all frames) and removed with a cross-fitted residual
(`movement_encoding.cv_residual`: each trial block predicted by a model fit on the others), so the ENL window is
treated exactly like every other frame -- no window-specific fitting. Accuracy against the permutation null of the
fixed predictions (`nolick_analysis.permutation_null`) and the majority-class floor, for all components and for
the SSp / MO subsets.
Output: session_poses/<a>_<d>_full/enl_movement_removed_decode.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    import glob

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import scripts.movement_encoding_session as MS
    import scripts.movement_position_angle as MPA
    from scripts.session_poses import session_dir
    from wfield_local import config
    from wfield_local import locanmf_position_decoder as LPD
    from wfield_local import movement_encoding as ME
    from wfield_local import movement_inputs as MI
    from wfield_local import nolick_analysis as na
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="animal:date:full")
    ap.add_argument("--video", nargs="*", default=["cam1", "cam2", "cam3", "cam4"])
    ap.add_argument("--video-k", type=int, default=30)
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args(argv)
    rv = PathResolver()
    dp = config.defaults()["decode"]
    p = MI.params()
    grid = tuple(p["alpha_grid"])
    animal, date = a.session.split(":")[:2]
    label = f"{animal}_{date[4:]}"
    s = next(x for x in SESSIONS if x["label"] == label)
    Y, reg, ft = MS.imaging(label)                                   # (T, ncomp), footprint-scaled C
    P = MS.session_pieces(animal, date, rv, ft, float(p["min_reach_px"]), spec=a.session)
    mask = P["mask"]
    folds = ME.trial_block_folds(ft[mask], P["trial_starts_s"], int(p["n_folds"]))
    L = P["licks"]
    video = MS.video_signals(animal, date, rv, ft, a.video, a.video_k) if a.video else {}
    running = MS.running_signal(animal, date, rv)

    def residual(extra):
        P2 = {**P, "extra": extra}
        mov = MPA._design(ft, mask, events={"contact": P["contact_s"], "tongue_onset": L.on_s.to_numpy()},
                          video_signals=P["sig"], video_t_s=P["vt"], trial_starts_s=P["trial_starts_s"],
                          **MPA._extra_kw(P2))
        al = ME.choose_alphas(ME.standardise(mov.X)[0], Y[mask], mov.group, folds, grid=grid)
        R = np.full(Y.shape, np.nan)
        R[mask] = ME.cv_residual(mov, Y[mask], folds, al)
        print(f"   movement model {sorted(set(mov.group))}: alphas {al}", flush=True)
        return np.nan_to_num(R, nan=0.0)

    signals = {"raw": Y}
    print("resid_dlc ...", flush=True)
    signals["resid_dlc"] = residual({})
    print("resid_run ...", flush=True)
    signals["resid_run"] = residual(dict(running))
    if a.video:
        print("resid_full ...", flush=True)
        signals["resid_full"] = residual({**running, **video})
        # the movement regressors themselves, on the imaging frames (NaN outside strobe .. trial end -> 0 = mean)
        vb = np.column_stack([ME.bin_to_frames(t, v, ft) for t, v in [*video.values(), running["running"]]])
        vb = (vb - np.nanmean(vb, 0)) / np.where(np.nanstd(vb, 0) > 0, np.nanstd(vb, 0), 1.0)
        signals["video_only"] = np.nan_to_num(vb, nan=0.0)

    args = argparse.Namespace(align="precue", baseline="none", fs=31.23, pre_s=1.0, post_s=dp["precue_post_s"],
                              max_rt=dp["max_rt_s"], source="locanmf", bins=None, cv="block")
    names = {int(k): v for k, v in json.load(open(glob.glob(
        f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}
    rows = []
    for nm, sig in signals.items():
        X, y, g, *_rest, feat_reg = LPD._trial_features(
            s, args, signal=np.asarray(sig, float).T,
            feat_region=(np.asarray(reg) if nm != "video_only" else np.zeros(sig.shape[1], int)))
        groups = {"all": None} if nm == "video_only" else {"all": None, "SSp": ("SSp",), "MO": ("MOp", "MOs")}
        for gname, prefs in groups.items():
            cols = (np.arange(X.shape[1]) if prefs is None else np.array(
                [i for i in range(X.shape[1]) if any(names.get(int(feat_reg[i]), "").startswith(q)
                                                     for q in prefs)]))
            if cols.size == 0:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
            pred = cross_val_predict(clf, X[:, cols], y, cv=GroupKFold(min(5, np.unique(g).size)), groups=g)
            acc = accuracy_score(y, pred)
            nu = na.permutation_null(y, pred, n_perm=a.n_perm, labels=LPD.DISPLAY_ORDER)
            floor = na.majority_class_floor(y, labels=LPD.DISPLAY_ORDER)
            bal = na.balanced_accuracy(y, pred, LPD.DISPLAY_ORDER)
            rows.append({"session": label, "signal": nm, "areas": gname, "n_trials": len(y), "n_feat": len(cols),
                         "acc": acc, "null_mean": nu["raw_null_mean"], "null_ci_hi": nu["raw_null_ci"][1],
                         "p": nu["raw_p"], "bal_acc": bal, "bal_null_mean": nu["bal_null_mean"], "bal_p": nu["bal_p"],
                         "majority_floor": float(floor)})
            print(f"  {nm:11s} {gname:4s} n={len(y)} acc {acc:.3f} (null {nu['raw_null_mean']:.3f}, p={nu['raw_p']:.4f})"
                  f"  balanced {bal:.3f} (null {nu['bal_null_mean']:.3f}, p={nu['bal_p']:.4f})", flush=True)
    out = session_dir(rv, a.session)[2] / "enl_movement_removed_decode.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
