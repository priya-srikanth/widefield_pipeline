"""EXECUTED best bin vs NEURAL best match, trial by trial (Priya 10-24: "can we make a decoder match vs executed best
bin match confusion matrix?").

Every trial with 3-D licks (pre-stroke hits; post-stroke hits + failed attempts), any target spout: row = the spout
bin its licks fall in most (dominant softmax weight, width 1; landing or trajectory), column = the frozen pre-stroke
DECODER's best match (argmax of its six probabilities; LOSO for pre). Rows normalised per session (share of that
session's trials with executed bin r decoded as spout c), averaged over sessions per animal x epoch; epoch minus pre
within animal, mean of 4, nested bootstrap (Bonferroni over 36). The diagonal = "the cortex is read as the spout the
tongue went to", whatever the target was. Also the agreement rate (diagonal share of all trials) per animal x epoch,
split by whether the licks went to the TARGET (on-target) or elsewhere (off-target).

Usage: python scripts/exec_vs_neural_match.py [--source decoder|encoder]
Output: <E>/attempt_coding_pp/exec_vs_decoder_*.csv + figures/exec_vs_decoder_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from softmax_neural_vs_lick import SP, boot_change, lick_weights, path_weights  # noqa: E402

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
LAB = ["cI", "cM", "cC", "fI", "fM", "fC"]
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
NAME = {"decoder": "frozen pre-stroke decoder argmax",
        "encoder": "frozen pre-stroke encoder, best of the six relabellings (lowest lick-window SSE)"}


def mats(J: pd.DataFrame) -> dict:
    out = {}
    for (an, ep, _lab), g in J.groupby(["animal", "epoch", "label"]):
        M = np.full((6, 6), np.nan)
        for r, t in enumerate(SP):
            x = g[g.exec_bin == t]
            if len(x) >= 5:
                M[r] = [(x.dec == c).mean() for c in SP]
        out.setdefault((an, ep), []).append(M)
    return out


def encoder_trials(tag: str = "v7g3_t3d_trial") -> pd.DataFrame:
    """Per trial, the frozen pre-stroke ENCODER's best-matching label (lowest lick-window SSE over the parcels among the
    six relabellings, `frozen_relabel` trial_sse) -> columns label, animal, epoch, trial_id, spout, dec."""
    import json
    rows = []
    for js in sorted((E / f"frozen_relabel_{tag}" / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        for tid, v in (m.get("trial_sse") or {}).get("lick", {}).items():
            cost = {k: x for k, x in v.items() if k in SP}
            if len(cost) == 6:
                rows.append({"label": m["label"], "animal": m["animal"], "epoch": m["epoch"], "trial_id": int(tid),
                             "spout": v["spout"], "dec": min(cost, key=cost.get)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="decoder", choices=("decoder", "encoder"))
    src = ap.parse_args(argv).source
    D = E / "attempt_coding_pp"
    if src == "decoder":
        T = pd.concat([pd.read_csv(D / f"trials_lick_{an}.csv.gz") for an in AN])
        T = T[(T.method == "dom") & T.cls.isin(["pre_hit", "hit", "incomplete"])]
        T["dec"] = np.array(SP)[T[[f"pp_{p}" for p in SP]].to_numpy().argmax(1)]
    else:
        T = encoder_trials()
        print(f"encoder: {len(T)} trials with all six relabellings", flush=True)
    rows, agree = [], []
    for kind, W, pre in (("landing", lick_weights(1.0), "lw_"), ("trajectory", path_weights(1.0), "tw_")):
        J = T.merge(W, on=["label", "trial_id"], how="inner")
        J["exec_bin"] = np.array(SP)[J[[f"{pre}{p}" for p in SP]].to_numpy().argmax(1)]
        J["on_target"] = J.exec_bin == J.spout
        for (an, ep), g in J.groupby(["animal", "epoch"]):
            for ot, h in (("on-target", g[g.on_target]), ("off-target", g[~g.on_target])):
                if len(h) >= 10:
                    agree.append({"exec": kind, "animal": an, "epoch": ep, "licks": ot, "n": len(h),
                                  "neural_matches_executed": float((h.dec == h.exec_bin).mean()),
                                  "neural_matches_target": float((h.dec == h.spout).mean())})
        S = mats(J)
        fig, axs = plt.subplots(5, 4, figsize=(13, 16))
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
                ax.set_title(f"{an} {ep}", fontsize=9)
                ax.set_xticks(range(6)); ax.set_xticklabels(LAB, fontsize=7); ax.set_yticks(range(6)); ax.set_yticklabels(LAB, fontsize=7)
                if k == 0:
                    ax.set_ylabel("executed bin", fontsize=8)
        for k, ep in enumerate(("acute", "subacute", "chronic")):
            obs, sig, ag, n = boot_change(S, ep)
            ax = axs[4, k + 1]
            ax.imshow(obs, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
            for r in range(6):
                for c in range(6):
                    if sig[r, c]:
                        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, ec="k", lw=2))
                    if abs(obs[r, c]) >= 0.06 or sig[r, c]:
                        ax.text(c, r, f"{obs[r, c]:+.2f}\n{ag[r, c]}/{n}", ha="center", va="center", fontsize=6,
                                color="w" if abs(obs[r, c]) > 0.4 else "k")
                    rows.append({"exec": kind, "epoch": ep, "exec_bin": SP[r], "decoded": SP[c], "change": obs[r, c],
                                 "significant": bool(sig[r, c]), "same_sign": int(ag[r, c])})
            ax.set_title(f"{ep} minus pre (pooled)", fontsize=9)
            ax.set_xticks(range(6)); ax.set_xticklabels(LAB, fontsize=7); ax.set_yticks(range(6)); ax.set_yticklabels(LAB, fontsize=7)
            ax.set_xlabel(f"{src} best match", fontsize=8)
        axs[4, 0].set_visible(False)
        fig.suptitle(f"EXECUTED best bin ({kind}) vs NEURAL best match ({NAME[src]}), every trial with 3-D licks, any target.\n"
                     "Rows 1-4: per animal x epoch (share of a session's trials in that executed bin decoded as each spout; viridis 0-1).\n"
                     "Row 5: epoch minus pre within animal, mean of 4, nested bootstrap (boxed = Bonferroni over 36; second line = animals same sign).",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(D / "figures" / f"exec_vs_{src}_{kind}.png", dpi=115)
        plt.close(fig)
    R = pd.DataFrame(rows); A = pd.DataFrame(agree)
    R.to_csv(D / f"exec_vs_{src}_change.csv", index=False); A.to_csv(D / f"exec_vs_{src}_agreement.csv", index=False)
    pd.set_option("display.width", 200)
    print(A[A.exec == "landing"].round(2).to_string(index=False))
    print(R[(R.exec == "landing") & (R.exec_bin == R.decoded)].round(2).to_string(index=False))

    # agreement figure: off-target trials, does the decoder follow the executed bin or the target?
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for j, kind in enumerate(("landing", "trajectory")):
        ax = axs[j]
        a = A[(A.exec == kind) & (A.licks == "off-target")]
        for k, ep in enumerate(EP):
            for q, colr, dx in (("neural_matches_executed", "#252525", -0.15), ("neural_matches_target", "#d94801", 0.15)):
                v = a[a.epoch == ep]
                if not len(v):
                    continue
                ax.bar(k + dx, v[q].mean(), width=0.28, color=colr, alpha=0.85,
                       label={"neural_matches_executed": "decoded = where the licks went",
                              "neural_matches_target": "decoded = the target"}[q] if k == 1 else None)
                for _, rr in v.iterrows():
                    ax.scatter(k + dx, rr[q], marker=MK[rr.animal], s=18, facecolor="w", edgecolor="k", lw=0.7, zorder=5)
        ax.axhline(1 / 6, color="k", ls=":", lw=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(EP); ax.set_title(f"OFF-target trials (executed = {kind})", fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    axs[0].set_ylabel("share of trials")
    fig.suptitle(f"When the licks miss the target, does the cortex look like where they went or like the target? ({NAME[src]}; "
                 "markers = animals with >= 10 off-target trials; chance dotted)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(D / "figures" / f"exec_vs_{src}_offtarget.png", dpi=125)
    print(f"-> {D / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
