"""10-21 summary figures for results that had tables only (Priya: "note where i can find figures"):
  figures/behaviour/attempt_classes_far.png        attempt classes (canonical trials) + time to contact, far contra / far mid
  figures/behaviour/miss_then_correct.png           first-lick contact / miss-then-correct / no contact, lateral spouts
  figures/target_vs_execution/lick_templates.png    unique target vs executed template variance, cortex, per session
  figures/target_vs_execution/destination.png       10b destination class shares (target / execution / neither)
Rule 8: every panel per animal."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Users/SabatiniLab/Github/widefield_pipeline")
from wfield_local import epochs  # noqa: E402

E = "M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding"
Q = f"{E}/qc"
S = sys.argv[1]
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
COL = {"hit": "#2b8a3e", "incomplete": "#f08c00", "jaw_only": "#868e96", "no_try": "#c92a2a", "unknown": "#dee2e6"}
import os  # noqa: E402

for d in ("behaviour", "target_vs_execution"):
    os.makedirs(f"{E}/figures/{d}", exist_ok=True)

# 1. attempt classes + time to contact
A = pd.read_csv(f"{Q}/trial_attempts.csv")
A["epoch"] = A.label.map(epochs.epoch_of)
W = pd.read_csv(f"{Q}/trial_attempts_sessions.csv")
W["animal"], W["epoch"] = W.label.str.split("_").str[0], W.label.map(epochs.epoch_of)
fig, ax = plt.subplots(3, 4, figsize=(14, 9), sharey="row")
for r, pos in enumerate(("far_contra", "far_mid")):
    for c, an in enumerate(AN):
        g = A[(A.animal == an) & (A.pos == pos) & A.canonical]
        fr = g.groupby("epoch").cls.value_counts(normalize=True).unstack().reindex(EP).fillna(0)
        bottom = np.zeros(len(EP))
        for k in COL:
            v = fr[k].to_numpy() if k in fr else np.zeros(len(EP))
            ax[r, c].bar(EP, v, bottom=bottom, color=COL[k], label=k if (r, c) == (0, 0) else None)
            bottom += v
        ax[r, c].set_title(f"{an} {pos}: trial classes (canonical trials)", fontsize=8)
        ax[r, c].tick_params(labelsize=7)
for c, an in enumerate(AN):
    g = W[W.animal == an]
    for pos, ls in (("far_contra", "-"), ("far_mid", "--")):
        m = g.groupby("epoch")[f"ttc_ms_{pos}"].mean().reindex(EP)
        ax[2, c].plot(EP, m.to_numpy(), "o" + ls, label=pos)
    ax[2, c].set_title(f"{an}: time to contact (ms; none = 3500)", fontsize=8)
    ax[2, c].tick_params(labelsize=7)
ax[0, 0].legend(fontsize=7)
ax[2, 0].legend(fontsize=7)
fig.suptitle("Attempt classes (attempted = hit + incomplete; jaw_only = reaction, not attempt -- Priya's 10-21 review) "
             "and the time-to-contact deficit", fontsize=10)
fig.tight_layout()
fig.savefig(f"{E}/figures/behaviour/attempt_classes_far.png", dpi=130)
plt.close(fig)

# 2. miss-then-correct
T = pd.read_csv(f"{S}/spout_tuning_check.csv")
F = T[(T.test == "first_lick") & (T.area == "MOs bilateral")]
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for c, pos in enumerate(("close_contra", "close_ipsi", "far_contra", "far_ipsi")):
    for an in AN:
        g = F[(F.pos == pos) & (F.animal == an)].groupby("epoch").frac_miss_then_correct.mean().reindex(EP)
        ax[c].plot(EP, g.to_numpy(), "o-", label=an)
    ax[c].set_title(f"{pos}: first lick misses, a later lick touches", fontsize=8)
    ax[c].tick_params(labelsize=7)
ax[0].set_ylabel("fraction of analysed trials")
ax[0].legend(fontsize=7)
fig.suptitle("Miss-then-correct trials (lick window = first lick + 1 s; kept by both contact controls)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{E}/figures/behaviour/miss_then_correct.png", dpi=130)
plt.close(fig)

# 3. lick templates (cortex)
L = pd.read_csv(f"{Q}/lick_templates_polar_20261021/lick_templates.csv")
L = L[L.area == "cortex"]
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for c, an in enumerate(sorted(L.animal.unique())):
    g = L[L.animal == an]
    x = np.array([EP.index(e) for e in g.epoch]) + np.random.default_rng(0).uniform(-0.15, 0.15, len(g))
    ax[c].scatter(x - 0.05, g.u_tgt * 100, s=14, c="tab:blue", label="unique target")
    ax[c].scatter(x + 0.05, g.u_exec * 100, s=14, c="tab:red", label="unique executed (landing zone)")
    for k, col in (("u_tgt", "tab:blue"), ("u_exec", "tab:red")):
        m = g.groupby("epoch")[k].mean().reindex(EP) * 100
        ax[c].plot(range(4), m.to_numpy(), "-", color=col)
    ax[c].axhline(0, color="k", lw=0.5)
    ax[c].set_xticks(range(4), EP, fontsize=8)
    ax[c].set_title(f"{an}: pre-stroke lick templates, cortex (3-D)", fontsize=9)
ax[0].set_ylabel("% lick-window variance, held out")
ax[0].legend(fontsize=7)
if "neither" in L:                                              # ceiling run (free per-spout kernels)
    for an, mk in zip(sorted(L.animal.unique()), ("o", "s")):
        g = L[L.animal == an]
        sh = (g.neither / g.ceiling_gain).clip(-1, 1.5)
        x = np.array([EP.index(e) for e in g.epoch]) + np.random.default_rng(1).uniform(-0.15, 0.15, len(g))
        ax[2].scatter(x, sh, s=14, marker=mk, label=an)
        ax[2].plot(range(4), sh.groupby(g.epoch).median().reindex(EP).to_numpy(), "-")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xticks(range(4), EP, fontsize=8)
    ax[2].set_title("NEITHER share = matches no pre-stroke template\n(free kernels - both templates) / ceiling",
                    fontsize=8)
    ax[2].legend(fontsize=7)
fig.suptitle("Does post-stroke activity look like the pre-stroke TARGET or the pre-stroke EXECUTION template? "
             "(qc/lick_templates_polar_20261021)", fontsize=9)
fig.tight_layout()
fig.savefig(f"{E}/figures/target_vs_execution/lick_templates.png", dpi=130)
plt.close(fig)

# 4. destination
D = pd.read_csv(f"{Q}/best_match_destination.csv")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for c, (title, sub) in enumerate((("all spouts", D), ("far_contra", D[D.aimed == "far_contra"]),
                                  ("far_mid", D[D.aimed == "far_mid"]))):
    labels, vals = [], []
    for an in sorted(D.animal.unique()):
        for ep in EP[1:]:
            g = sub[(sub.animal == an) & (sub.epoch == ep)]
            if len(g):
                labels.append(f"{an}\n{ep[:3]}")
                vals.append([(g.cls == k).mean() for k in ("target", "execution", "neither")])
    v = np.array(vals)
    for k, (name, col) in enumerate((("target", "#1971c2"), ("execution", "#e03131"), ("neither", "#adb5bd"))):
        ax[c].bar(range(len(v)), v[:, k], bottom=v[:, :k].sum(1), color=col, label=name)
    ax[c].set_xticks(range(len(v)), labels, fontsize=7)
    ax[c].set_title(f"{title}", fontsize=8)
ax[0].legend(fontsize=7)
fig.suptitle("10b destination vs 3-D landing zone (PS92 / PS93): target = own spout; execution = where licks landed; "
             "neither = another spout", fontsize=9)
fig.tight_layout()
fig.savefig(f"{E}/figures/target_vs_execution/destination.png", dpi=130)
plt.close(fig)
print("ok")
