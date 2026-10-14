"""Single-camera EKS over a TWO-MODEL ensemble (e.g. round-4 LP heatmap + round-4 LP temporal-context) for one clip.

    conda activate eks
    PYTHONPATH=. python scripts/eks_two_model.py --member <a.csv> --member <b.csv> --out <eks.csv> [--smooth 10]

Priya, 2026-10-14 ("let's try both"): the heatmap model is the better tongue detector, the temporal-context model
removes the close-position jaw side-flips -- an ensemble smoother can take each where it is good. Members below a
likelihood cut (`--cut`, 0.4 = our cleaning threshold) get the occlusion variance floor, so a member that says
"hidden" (LP parks it at the image centre) does not pull the track; smooth_param is fixed
(10: the 2026-10-05 tests -- the auto fit over-smoothed the tongue to ~1/3 of each lick's excursion).
Output: a DLC-format csv (x, y, likelihood = EKS's score) that `pose_cue_traces compare` and
`model_lick_compare` read like any model.
"""
from __future__ import annotations

import argparse

import numpy as np

from wfield_local import pose_eks as PE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=float, default=10.0)
    ap.add_argument("--cut", type=float, default=0.4)
    a = ap.parse_args(argv)
    members = [PE.pose_from_lp_csv(m) for m in a.member]
    n = min(len(m) for m in members)
    members = [m[:n].copy() for m in members]
    # A member below --cut is saying "not visible", and LP parks that answer at the IMAGE CENTRE. The occlusion
    # variance floor alone did NOT stop it pulling the track (2026-10-14, Priya's lick review: EKS tongue points sat
    # up-right of LP r4's, toward the centre, at confidence ~0.5 = the mean of 1.0 and 0.0, and real licks were
    # lost). So on such frames its x, y become the mean of the CONFIDENT members (if any) -- it neither pulls nor
    # vetoes -- and its likelihood is left low for the floor. Frames where no member is confident are untouched.
    for k in range(members[0].shape[1]):
        conf = np.stack([m[:, k, 2] >= a.cut for m in members])                       # (M, n)
        xs = np.stack([m[:, k, 0] for m in members]); ys = np.stack([m[:, k, 1] for m in members])
        n_conf = conf.sum(0)
        mx = np.where(n_conf > 0, (xs * conf).sum(0) / np.maximum(n_conf, 1), np.nan)
        my = np.where(n_conf > 0, (ys * conf).sum(0) / np.maximum(n_conf, 1), np.nan)
        for j, m in enumerate(members):
            fill = ~conf[j] & (n_conf > 0)
            m[fill, k, 0], m[fill, k, 1] = mx[fill], my[fill]
    res, s = PE.smooth_singlecam(members, smooth_param=a.smooth, occluded_cut=a.cut)
    pose = res.pose.copy()
    # EKS returns a position on every frame; keep its own score as the likelihood so downstream cut-offs apply
    pose[:, :, 2] = np.clip(pose[:, :, 2], 0, 1)
    PE.pose_to_frame(pose, scorer="eks").to_csv(a.out)
    print(f"{n} frames, smooth_param {s.ravel().round(2).tolist()} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
