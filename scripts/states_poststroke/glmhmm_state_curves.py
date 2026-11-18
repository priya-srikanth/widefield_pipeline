"""GLM-HMM K5 (design v2) per-state response curves across the six spouts -- the analogue of the per-state
psychometric curves in Ashwood et al. 2022 (Priya 11-17). The fitted weights were not saved, so each curve is the
OBSERVED rate weighted by the trial's state posterior: sum_t p_k(t) y(t) / sum_t p_k(t), per animal, then mean +- SEM
across animals (thin lines = animals). Outputs: P(contact), P(incomplete), P(jaw-only), P(no response), P(pre-cue
lick), P(ITI lick), mean running fraction (pre-cue / ITI). Spouts in canonical order (orofacial_cohort.POS_ORDER;
L = ipsilesional). All epochs pooled (the emissions are shared across epochs); occupancy by epoch is the other figure.
Scratch; locanmf env.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local.attempt_coding import LESION_NAME
from wfield_local.orofacial_cohort import POS_ORDER

S = Path(sys.argv[1])
OUT = S / "poststroke_states"
d = pd.read_csv(S / "glmhmm_v1" / "states_pooled_all_K5_v2.csv.gz")
d = d[d.cls != "unknown"].copy()
d["animal"] = d.label.str[:4]
d["spout"] = d.position.map(LESION_NAME)
d["run"] = (d.run_frac_precue.fillna(0) + d.run_frac_iti.fillna(0)) / 2
OUTS = [("contact", d.cls.eq("contact")), ("incomplete", d.cls.eq("incomplete")), ("jaw-only", d.cls.eq("jaw_only")),
        ("no response", d.cls.eq("no_response")), ("pre-cue lick", d.precue_lick.eq(1).where(d.precue_lick >= 0)),
        ("ITI lick", d.iti_lick.eq(1).where(d.iti_lick >= 0)), ("running (frac)", d.run)]
NAMES = {0: "0 engaged", 1: "1 engaged", 2: "2 engaged + running", 3: "3 partially engaged", 4: "4 disengaged"}
# Okabe-Ito (colorblind-safe), avoiding the epoch palette (black / magenta / gold / teal)
SCOL = {0: "#0072B2", 1: "#56B4E9", 2: "#009E73", 3: "#E69F00", 4: "#D55E00"}
K = 5


def curves(y: pd.Series):
    """(K, animals, 6) posterior-weighted rate."""
    ok = y.notna()
    out = np.full((K, d.animal.nunique(), len(POS_ORDER)), np.nan)
    for ia, (_a, g) in enumerate(d[ok].groupby("animal")):
        yy = y[ok].loc[g.index].astype(float)
        for ip, sp in enumerate(POS_ORDER):
            m = (g.spout == sp).to_numpy()
            for k in range(K):
                w = g[f"p_state{k}"].to_numpy()[m]
                if w.sum() >= 5:                                  # >= 5 effective trials
                    out[k, ia, ip] = np.sum(w * yy.to_numpy()[m]) / w.sum()
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5))
    x = np.arange(len(POS_ORDER))
    rows = []
    for ax, (name, y) in zip(axs.ravel(), OUTS):
        C = curves(y)
        for k in range(K):
            for a_ in C[k]:
                ax.plot(x, a_, color=SCOL[k], lw=0.5, alpha=0.25, zorder=2)
            m = np.nanmean(C[k], 0)
            se = np.nanstd(C[k], 0, ddof=1) / np.sqrt(np.sum(np.isfinite(C[k]), 0))
            ax.fill_between(x, m - se, m + se, color=SCOL[k], alpha=0.08, lw=0, zorder=1)
            ax.plot(x, m, "o-", color=SCOL[k], lw=1.8, ms=4, label=NAMES[k], zorder=4)
            rows += [{"output": name, "state": k, "spout": sp, "mean": m[i], "sem": se[i]} for i, sp in enumerate(POS_ORDER)]
        ax.set_title(name, fontsize=10)
        ax.set_ylim(-0.02, 1.02 if name != "running (frac)" else None)
        ax.axvline(2.5, color="0.85", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        # near / far x ipsi / mid / contra, with the short codes nI .. fC (Priya 11-17)
        short = {"close": ("n", "near"), "far": ("f", "far")}
        ticks = [f"{short[p.split('_')[0]][0]}{p.split('_')[1][0].upper()}" for p in POS_ORDER]
        ax.set_xticks(x, ticks, fontsize=9)
    occ = d[[f"p_state{k}" for k in range(K)]].mean()
    ax = axs.ravel()[-1]
    ax.bar(range(K), occ.to_numpy(), color=[SCOL[k] for k in range(K)])
    ax.set_xticks(range(K), [str(k) for k in range(K)])
    ax.set_title("overall occupancy (all trials, all epochs)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    h, lab_ = axs[0, 0].get_legend_handles_labels()
    axs[1, 3].legend(h, lab_, frameon=False, fontsize=8, loc="upper right")   # empty corner of the occupancy panel
    for ax in axs[:, 0]:
        ax.set_ylabel("P (posterior-weighted)")
    fig.suptitle("Fig. 3. GLM-HMM K5 (design v2): per-state response curves by spout -- observed rates weighted by each "
                 "trial's state posterior\nper animal (thin), mean +- SEM across the 4 animals (thick); all epochs "
                 "pooled; spouts n = near, f = far; I / M / C = ipsi (ipsilesional, mouse left) / mid / contra" + chr(10)
                 + "nM: jaw-only cannot be called (cam4 chin hidden by the spout; cam1 failed review) -> no-lick nM "
                 "trials default to no response (trial_classes, jaw_judged = False)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_glmhmm_state_curves.png", dpi=150)
    pd.DataFrame(rows).to_csv(OUT / "glmhmm_state_curves.csv", index=False)
    print("written")
