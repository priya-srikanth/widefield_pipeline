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
    members = [m[:n] for m in members]   # EKS needs a value everywhere: below --cut a member gets the occlusion
    #                                    variance floor (no pull on the track) instead of being blanked
    res, s = PE.smooth_singlecam(members, smooth_param=a.smooth, occluded_cut=a.cut)
    pose = res.pose.copy()
    # EKS returns a position on every frame; keep its own score as the likelihood so downstream cut-offs apply
    pose[:, :, 2] = np.clip(pose[:, :, 2], 0, 1)
    PE.pose_to_frame(pose, scorer="eks").to_csv(a.out)
    print(f"{n} frames, smooth_param {s.ravel().round(2).tolist()} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
