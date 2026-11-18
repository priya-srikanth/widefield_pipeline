"""Fig. 3b: GLM-HMM K5 (design v2) per-state response curves by spout, SPLIT BY EPOCH (Priya 11-17).
Rows = states; columns = P(contact), P(jaw-only), P(no response), P(pre-cue lick) -- observed rates weighted by the
trial's state posterior (does a state MEAN the same thing after the stroke? the emissions are shared across epochs) --
and P(state | spout) = mean posterior (which states the animals are in at each spout). Per animal (>= 5 effective
trials per animal x spout x epoch cell), mean +- SEM across animals. August-only pre. Spouts nI .. fC.
nM: jaw-only cannot be called (trial_classes JAW_UNJUDGED_POSITIONS). Scratch; locanmf env.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local.attempt_coding import LESION_NAME
from wfield_local.orofacial_cohort import EPOCH_LINE as ECOL
from wfield_local.orofacial_cohort import POS_ORDER

S = Path(sys.argv[1])
OUT = S / "poststroke_states"
EP = json.load(open("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/arhmm_20261117/inputs/epochs.json"))
EPOCHS = ["pre", "acute", "subacute", "chronic"]
K = 5
NAMES = {0: "0 engaged", 1: "1 engaged", 2: "2 engaged + running", 3: "3 partially engaged", 4: "4 disengaged"}
TICKS = [("n" if p.startswith("close") else "f") + p.split("_")[1][0].upper() for p in POS_ORDER]

d = pd.read_csv(S / "glmhmm_v1" / "states_pooled_all_K5_v2.csv.gz")
d = d[d.cls != "unknown"].copy()
d["animal"], d["spout"] = d.label.str[:4], d.position.map(LESION_NAME)
d["epoch"] = d.label.map(lambda lab: None if EP.get(lab) is None or (EP[lab] == "pre" and lab[-4:] < "0801")
                         else EP[lab])
d = d[d.epoch.notna()]
COLS = [("contact", d.cls.eq("contact").astype(float)), ("incomplete", d.cls.eq("incomplete").astype(float)),
        ("jaw-only", d.cls.eq("jaw_only").astype(float)),
        ("no response", d.cls.eq("no_response").astype(float)),
        ("pre-cue lick", d.precue_lick.eq(1).astype(float).where(d.precue_lick >= 0)), ("P(state | spout)", None)]


def cell(g, y, k):
    w = g[f"p_state{k}"].to_numpy()
    if y is None:                                            # occupancy
        return w.mean() if len(w) >= 5 else np.nan
    yy = y.loc[g.index].to_numpy()
    ok = np.isfinite(yy)
    return np.sum(w[ok] * yy[ok]) / w[ok].sum() if w[ok].sum() >= 5 else np.nan


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(POS_ORDER))
    fig, axs = plt.subplots(K, len(COLS), figsize=(20, 15), sharex=True)
    rows = []
    for k in range(K):
        for j, (name, y) in enumerate(COLS):
            ax = axs[k, j]
            for e in EPOCHS:
                A = np.full((d.animal.nunique(), len(POS_ORDER)), np.nan)
                for ia, (_a, ga) in enumerate(d[d.epoch == e].groupby("animal")):
                    for ip, sp in enumerate(POS_ORDER):
                        g = ga[ga.spout == sp]
                        if len(g):
                            A[ia, ip] = cell(g, y, k)
                n = np.sum(np.isfinite(A), 0)
                m = np.nanmean(A, 0)
                se = np.nanstd(A, 0, ddof=1) / np.sqrt(np.maximum(n, 1))
                ax.fill_between(x, m - se, m + se, color=ECOL[e], alpha=0.08, lw=0, zorder=2)
                ax.plot(x, m, "o-", color=ECOL[e], lw=1.6, ms=3.5, label=e, zorder=4)
                rows += [{"state": k, "output": name, "epoch": e, "spout": TICKS[i], "mean": m[i], "sem": se[i],
                          "n_animals": int(n[i])} for i in range(len(x))]
            ax.axvline(2.5, color="0.85", lw=0.8)
            ax.set_ylim(-0.02, 1.02)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(x, TICKS, fontsize=9)
            if k == 0:
                ax.set_title(name, fontsize=11)
            if j == 0:
                ax.set_ylabel(NAMES[k], fontsize=10, fontweight="bold")
    axs[0, -1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Fig. 3b. GLM-HMM K5 (design v2) per-state response by spout, split by epoch (August-only pre). Columns "
                 "1-5: observed rates weighted by state posterior (does the state mean the same thing post-stroke?); "
                 "column 6: P(state | spout)" + chr(10) + "mean +- SEM across animals (>= 5 effective trials per animal "
                 "x spout x epoch); spouts n / f = near / far, I / M / C = ipsi (ipsilesional) / mid / contra; nM: "
                 "jaw-only cannot be called -> no-lick nM trials default to no response", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig3b_glmhmm_state_curves_by_epoch.png", dpi=130)
    pd.DataFrame(rows).to_csv(OUT / "glmhmm_state_curves_by_epoch.csv", index=False)
    print("written")
