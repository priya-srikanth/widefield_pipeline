"""Goal-question summary figures (10-24): binned contest per animal, attempt-class decoder, trial-type amplitudes."""
import sys
from pathlib import Path

import matplotlib
from wfield_local.orofacial_outputs import encoding_root as _encoding_root  # rebuild root (11-03)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E = _encoding_root()
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
TC, XC = "#6a3d9a", "#e66101"

# 1. binned contest
d = pd.read_csv(E / "movement_first_subset-nokin_exec-polar_licks-combined_bins/tables/region_measures.csv")
c = d[d.pos == "cortex"].copy()
fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
for r, w in enumerate(["cue", "lick"]):
    lo, hi = np.inf, -np.inf
    for k, an in enumerate(AN):
        ax = axes[r, k]
        a = c[c.animal == an]
        for col, colr, lab, dx in ((f"nokin:u:tgtbin|{w}", TC, "target bins", -0.08), (f"nokin:u:exbin|{w}", XC, "executed bins", 0.08)):
            m = [a[a.epoch == e][col].mean() for e in EP]
            for i, e in enumerate(EP):
                v = a[a.epoch == e][col].to_numpy()
                ax.scatter(np.full(len(v), i + dx), v, s=8, color=colr, alpha=0.35, lw=0)
                lo, hi = min(lo, np.nanmin(v) if len(v) else lo), max(hi, np.nanmax(v) if len(v) else hi)
            ax.plot(np.arange(4) + dx, m, "-o", color=colr, label=lab, lw=2, ms=6)
        ax.axhline(0, color="0.6", lw=0.8)
        n = [int((a.epoch == e).sum()) for e in EP]
        ax.set_title(f"{an}  (n {'/'.join(map(str, n))})" if r == 0 else "", fontsize=10)
        ax.set_xticks(range(4)); ax.set_xticklabels(EP)
        if k == 0:
            ax.set_ylabel(f"{w} window\nunique dR2 (R2 fraction)")
    for k in range(4):
        axes[r, k].set_ylim(max(lo, -45), min(hi, 80))
axes[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle("Binned target vs executed contest (O2, final inputs, width 1), cortex: unique contribution of each set "
             "beyond the other\nlines = epoch means, dots = sessions; chronic target > executed 4/4, acute lick 3/4, "
             "subacute mixed, pre no lead", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "fig1_binned_contest.png", dpi=140); plt.close(fig)

# 2. attempt-class decoder (far contra)
A = E / "attempt_coding"
CL = ["pre_hit", "hit", "incomplete", "noattempt", "reaction", "stopped"]
CC = {"pre_hit": "0.25", "hit": "#1b9e77", "incomplete": "#d95f02", "noattempt": "#7570b3", "reaction": "#e7298a",
      "stopped": "0.65"}
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for j, w in enumerate(["precue", "cue", "lick"]):
    s = pd.read_csv(A / f"summary_{w}.csv")
    s = s[(s.spout == "far_contra") & (s.method == "dom") & (((s.epoch == "acute") & (s.cls != "pre_hit")) | (s.cls == "pre_hit"))]
    ax = axes[0, j]
    for k, an in enumerate(AN):
        for i, cl in enumerate(CL):
            x = s[(s.animal == an) & (s.cls == cl)]
            if len(x) and x.n.iloc[0] >= 5:
                ax.bar(k * 7 + i, x.p_true.iloc[0], yerr=x.p_true_sem.iloc[0], color=CC[cl], width=0.85,
                       label=cl if k == 0 else None, error_kw={"lw": 0.8})
    ax.axhline(1 / 6, color="k", ls=":", lw=1)
    ax.set_xticks([k * 7 + 2.5 for k in range(4)]); ax.set_xticklabels(AN)
    ax.set_title({"precue": "ENL (pre-cue, before movement)", "cue": "cue window", "lick": "lick window"}[w], fontsize=10)
    if j == 0:
        ax.set_ylabel("frozen decoder P(true spout)\nfar contra (chance 1/6 dotted)")
    ax = axes[1, j]
    for k, an in enumerate(AN):
        t = pd.read_csv(A / f"trials_{w}_{an}.csv.gz")
        t = t[(t.method == "dom") & (t.spout == "far_contra") & ((t.epoch == "acute") | (t.cls == "pre_hit"))]
        for i, cl in enumerate(CL):
            x = t[t.cls == cl]
            if len(x) >= 5:
                ax.bar(k * 7 + i, x.pred.isin(["far_ipsi", "far_mid", "far_contra"]).mean(), color=CC[cl], width=0.85)
    ax.set_xticks([k * 7 + 2.5 for k in range(4)]); ax.set_xticklabels(AN); ax.set_ylim(0, 1)
    if j == 0:
        ax.set_ylabel("share decoded as ANY far spout")
axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
fig.suptitle("Item 1: far-contra trials on the frozen pre-stroke decoder, by what the animal did (acute; pre_hit = "
             "pre-stroke reference)\nfailed attempts (orange) lose the far-contra code but are read as a FAR spout "
             "(distance kept, side lost); no pre-cue difference", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(OUT / "fig2_attempt_decoder.png", dpi=140); plt.close(fig)

# 3. trial-type amplitudes (lick window, far contra)
h = pd.read_csv(E / "attempt_maps" / "hemispheres_lick.csv")
h = h[h.spout == "far_contra"].set_index(["animal", "epoch", "cls"])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for j, (ref, title) in enumerate([("pre_hit", "relative to PRE-STROKE HITS (touch + water)"),
                                  ("pre_incomplete", "relative to PRE-STROKE MISSES (no touch, no water)")]):
    ax = axes[j]
    for k, an in enumerate(AN):
        for i, (e, cl, colr, lab) in enumerate([("acute", "hit", "#1b9e77", "acute hits"),
                                                ("acute", "incomplete", "#d95f02", "acute failed attempts")]):
            try:
                r, rf = h.loc[(an, e, cl)], h.loc[(an, "pre", ref)]
            except KeyError:
                continue
            v = np.mean([r.ipsi / rf.ipsi, r.contra / rf.contra])
            ax.bar(k * 3 + i, v, color=colr, width=0.85, label=lab if k == 0 else None)
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_xticks([k * 3 + 0.5 for k in range(4)]); ax.set_xticklabels(AN); ax.set_title(title, fontsize=10)
axes[0].set_ylabel("lick-window amplitude ratio\n(mean of both hemispheres)")
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Item 4: how big is the far-contra response on failed attempts?\nSmaller than hits in 3/4, but vs "
             "touch-matched pre-stroke misses only 2/4 (PS92's reference = 7 trials)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(OUT / "fig3_attempt_amplitude.png", dpi=140); plt.close(fig)
print("ok")
