"""Why is the post-stroke PC2 transient so much larger at the CLOSE spouts? (Priya 11-07: "it's still curious to me that
PC2 amplitudes are so much higher for close spouts".) First-lick-aligned (WF_SS_ALIGN=lick, -1.0..0.7 s).

1. What PC2 is: the cohort PCA of state_space_trajectories (every animal's pre spout x time template, centered on the
   animal's own pre mean, stacked); loadings of PC1 / PC2 over the 62 areas.
2. Per trial: PC2 transient = mean PC2 score 0.1..0.3 s after the first lick minus -0.4..-0.1 s before it (same for PC1:
   0.1..0.5 s). Summarized per animal x spout x epoch.
3. The same trials' first lick (3-D, licks.csv.gz via the landing join's trial_id): reach_mm, speed_peak, latency
   (rt_ms), contact, and the number of further licks within 0.7 s of the first (a burst sums more lick responses).
   Within-animal trial-level association of the PC2 transient with each (Spearman), per epoch, close spouts.
Output: Widefield/state_space_recovery/pc2_lick/ (pc2_trials.csv, pc2_loadings.csv). Exploratory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("WF_SS_ALIGN", "lick")
sys.path.insert(0, str(Path(__file__).parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery")
OUT = ROOT / "pc2_lick"
KIN = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d/kinematics/licks.csv.gz")
T0 = -1.0                                                  # first lick-aligned bin start
idx = lambda a, b: slice(int(round((a - T0) / 0.1)), int(round((b - T0) / 0.1)))  # noqa: E731


def main():
    assert S.ALIGN == "lick"
    data, cols = S.load_sessions()
    mus, cent = {}, []
    for an, sess in data.items():
        T = R.template([s for s in sess if s[1] == "pre"])
        mus[an] = S.flat(T[np.isfinite(T).all(axis=(1, 2))]).mean(0)
        cent.append(S.flat(T[np.isfinite(T).all(axis=(1, 2))]) - mus[an])
    F = np.concatenate(cent)
    W = np.linalg.svd(F - F.mean(0), full_matrices=False)[2][:2].T                   # areas x 2
    load = pd.DataFrame({"area": cols, "pc1": W[:, 0], "pc2": W[:, 1]})
    # trials
    L = pd.read_csv(R.LANDING)
    K = pd.read_csv(KIN, usecols=["session", "trial_id", "lick_idx", "t_peak_cam4", "reach_mm", "speed_peak"])
    K["label"] = K.session.str.replace(r"_2026(\d{4})$", r"_\1", regex=True)
    first = K[K.lick_idx == 0].set_index(["label", "trial_id"])
    burst = (K.merge(first[["t_peak_cam4"]].rename(columns={"t_peak_cam4": "t0"}), left_on=["label", "trial_id"],
                     right_index=True)
             .assign(follow=lambda d: (d.lick_idx > 0) & (d.t_peak_cam4 - d.t0 <= 0.7))
             .groupby(["label", "trial_id"]).follow.sum())
    rows = []
    for an, sess in data.items():
        for lab, ep, _, Y, pos in sess:
            z = (Y - mus[an]) @ W                                                      # trials x bins x 2
            pc2 = z[:, idx(0.1, 0.3), 1].mean(1) - z[:, idx(-0.4, -0.1), 1].mean(1)
            pc1 = z[:, idx(0.1, 0.5), 0].mean(1) - z[:, idx(-0.4, -0.1), 0].mean(1)
            g = L[L.label == lab].set_index("row")
            for r in range(len(pos)):
                d = {"animal": an, "label": lab, "epoch": ep, "spout": pos[r], "pc1_tr": pc1[r], "pc2_tr": pc2[r]}
                if r in g.index:
                    tid = g.loc[r, "trial_id"]
                    d.update(rt_ms=g.loc[r, "rt_ms"], contact=g.loc[r, "contact"], trial_id=tid)
                    if (lab, tid) in first.index:
                        d.update(reach_mm=first.loc[(lab, tid), "reach_mm"], speed=first.loc[(lab, tid), "speed_peak"],
                                 n_follow=burst.get((lab, tid), np.nan))
                rows.append(d)
    D = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "pc2_trials.csv", index=False)
    load.to_csv(OUT / "pc2_loadings.csv", index=False)
    with pd.option_context("display.width", 230, "display.precision", 2, "display.max_rows", 200):
        lo = load.assign(a2=load.pc2.abs()).sort_values("a2", ascending=False)
        print("PC2 top loadings:\n", lo.head(12)[["area", "pc1", "pc2"]].to_string(index=False))
        print("PC1 sign / mean loading:", round(load.pc1.mean(), 3), " PC2 mean loading:", round(load.pc2.mean(), 3))
        for c in ("pc2_tr", "pc1_tr", "reach_mm", "speed", "n_follow", "rt_ms", "contact"):
            m = D.groupby(["spout", "epoch", "animal"])[c].mean().unstack("epoch")[R.EPOCHS]
            print(f"\n== {c} (animal mean of trials) ==")
            print(m.unstack("animal").reindex(S.POS).round(2).to_string())
        print("\n== within-animal Spearman, close spouts, trial level: PC2 transient vs ==")
        close = D[D.spout.str.startswith("close")]
        for ep in R.EPOCHS:
            out = []
            for c in ("reach_mm", "speed", "n_follow", "rt_ms", "pc1_tr"):
                rs = [g[["pc2_tr", c]].dropna().corr(method="spearman").iloc[0, 1] for _, g in
                      close[close.epoch == ep].groupby("animal") if g[c].notna().sum() > 20]
                out.append(f"{c} {np.round(rs, 2).tolist()}")
            print(f"  {ep:9s}", " | ".join(out))


if __name__ == "__main__":
    main()
