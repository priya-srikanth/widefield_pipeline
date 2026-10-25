"""ENCODING-MODEL confusion matrices from the relabel test (Priya 10-24: "compare neural encoding with real neural data
to see if post-stroke neural data looks like a different spout").

`wfield_local.frozen_relabel` re-predicts each spout q's trials with the frozen pre-stroke encoder AS IF they were spout
j (j's target columns + j's spout-locked templates; the executed licks / movement stay as recorded) and scores the
held-out SSE increase on q's own frames: dSSE(q -> j) > 0 = the true label fits better. Here, per session and true spout
q, the BEST-FITTING label (the j with the lowest SSE, q itself if no relabelling beats it; all areas jointly) gives a
one-hot row; rows averaged over an animal's sessions = the share of sessions whose cortex looks most like spout j.
Same layout and statistics as the landing / trajectory / decoder matrices: per-animal matrices (viridis 0-1) and epoch
minus pre within animal, mean of animals, nested bootstrap (animals -> sessions), Bonferroni over 36 (RdBu +-0.6).
Also the mean relabel margin per (q, j) (the dSSE itself, continuous) per animal x epoch.

Usage: python scripts/relabel_confusion.py [--tag v7g3_t3d] [--window lick]
Output: <E>/frozen_relabel_<tag>/relabel_confusion_<window>.csv + figures/relabel_confusion_*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from softmax_neural_vs_lick import boot_change  # noqa: E402  (same nested bootstrap as the decoder / lick matrices)

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
SP = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
LAB = ["cI", "cM", "cC", "fI", "fM", "fC"]
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]


def session_rows(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (session, true spout): one-hot best label + the dSSE vector (true label = 0)."""
    out = []
    for (lab, an, ep, q), g in d.groupby(["label", "animal", "epoch", "spout"]):
        if q not in SP:
            continue
        cost = {j: 0.0 for j in SP}
        cost.update(dict(zip(g["as"], g.dsse_x1e3)))
        if len(g) < 5:                          # this session lacks some spouts: no full relabel set
            continue
        best = min(SP, key=lambda j: cost[j])
        out.append({"label": lab, "animal": an, "epoch": ep, "spout": q, "best": best,
                    **{f"b_{j}": float(j == best) for j in SP}, **{f"m_{j}": cost[j] for j in SP}})
    return pd.DataFrame(out)


def session_mats(R: pd.DataFrame, prefix: str) -> dict:
    out = {}
    for (an, ep, _lab), g in R.groupby(["animal", "epoch", "label"]):
        M = np.full((6, 6), np.nan)
        for r, t in enumerate(SP):
            x = g[g.spout == t]
            if len(x):
                M[r] = x[[f"{prefix}{j}" for j in SP]].to_numpy(float)[0]
        out.setdefault((an, ep), []).append(M)
    return out


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v7g3_t3d")
    ap.add_argument("--window", default="lick", choices=("all", "cue", "lick"))
    a = ap.parse_args(argv)
    D = E / f"frozen_relabel_{a.tag}"
    fd = D / "figures"
    fd.mkdir(exist_ok=True)
    d = pd.read_csv(D / "relabel_by_session.csv")
    d = d[(d.window == a.window) & (d.area == "all_areas")]
    if d.empty:
        raise SystemExit("no all_areas rows")
    R = session_rows(d)
    R.to_csv(D / f"relabel_confusion_sessions_{a.window}.csv", index=False)
    S = session_mats(R, "b_")
    # per-animal best-label share matrices
    fig, axs = plt.subplots(4, 4, figsize=(13, 13))
    for i, an in enumerate(AN):
        for k, ep in enumerate(EP):
            ax = axs[i, k]
            if (an, ep) in S:
                M = np.nanmean(S[(an, ep)], 0)
                ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
                for r in range(6):
                    for c in range(6):
                        if np.isfinite(M[r, c]) and M[r, c] >= 0.1:
                            ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=6.5,
                                    color="k" if M[r, c] > 0.6 else "w")
                ax.set_title(f"{an} {ep} (n {len(S[(an, ep)])} sessions)", fontsize=9)
            ax.set_xticks(range(6)); ax.set_xticklabels(LAB, fontsize=7); ax.set_yticks(range(6)); ax.set_yticklabels(LAB, fontsize=7)
    fig.suptitle(f"NEURAL (encoding model): which spout's pre-stroke pattern best explains each spout's trials, {a.window} window.\n"
                 "Rows = true spout, columns = best-fitting label in the relabel test (executed licks fixed); share of sessions; rows sum to 1.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fd / f"relabel_confusion_matrices_{a.window}.png", dpi=120)
    plt.close(fig)
    # epoch minus pre, pooled
    rows = []
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    for k, ep in enumerate(("acute", "subacute", "chronic")):
        obs, sig, agree, n = boot_change(S, ep)
        ax = axs[k]
        ax.imshow(obs, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
        for r in range(6):
            for c in range(6):
                if sig[r, c]:
                    ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, ec="k", lw=2))
                if abs(obs[r, c]) >= 0.06 or sig[r, c]:
                    ax.text(c, r, f"{obs[r, c]:+.2f}\n{agree[r, c]}/{n}", ha="center", va="center", fontsize=6.5,
                            color="w" if abs(obs[r, c]) > 0.4 else "k")
                rows.append({"window": a.window, "epoch": ep, "target": SP[r], "as": SP[c], "change": obs[r, c],
                             "significant": bool(sig[r, c]), "same_sign": int(agree[r, c]), "n_animals": n})
        ax.set_xticks(range(6)); ax.set_xticklabels(LAB); ax.set_yticks(range(6)); ax.set_yticklabels(LAB)
        ax.set_title(f"{ep} minus pre", fontsize=10); ax.set_xlabel("best-fitting label (encoding model)")
        if k == 0:
            ax.set_ylabel("true spout")
    fig.suptitle(f"NEURAL (encoding model, relabel test): share of sessions best fit by each label, {a.window} window, epoch minus pre within animal, mean of 4.\n"
                 "Boxed = nested bootstrap excludes 0 (Bonferroni over 36); second line = animals with the same sign.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(fd / f"relabel_confusion_change_{a.window}.png", dpi=130)
    plt.close(fig)
    T = pd.DataFrame(rows)
    T.to_csv(D / f"relabel_confusion_{a.window}.csv", index=False)
    pd.set_option("display.width", 200)
    print(T[T.target == "far_contra"].round(2).to_string(index=False))
    print(R[R.spout == "far_contra"].groupby(["animal", "epoch"]).best.value_counts().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
