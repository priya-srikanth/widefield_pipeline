from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)
import sys
from pathlib import Path
import matplotlib
from wfield_local.orofacial_outputs import encoding_root as _encoding_root  # rebuild root (11-03)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
A = _encoding_root() / "attempt_coding"
OUT = Path(sys.argv[1])
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
CL = [("hit", "#1b9e77"), ("incomplete", "#d95f02"), ("noattempt", "#7570b3"), ("reaction", "#e7298a"), ("stopped", "0.6")]
fig, axes = plt.subplots(2, 4, figsize=(14, 9), sharey="row")
for r, w in enumerate(["cue", "lick"]):
    s = pd.read_csv(A / f"summary_{w}.csv")
    s = s[(s.spout == "far_contra") & (s.method == "dom")]
    for k, an in enumerate(AN):
        ax = axes[r, k]
        x = s[(s.animal == an) & (s.cls == "pre_hit")]
        if len(x):
            ax.axhline(x.p_true.iloc[0], color="0.2", lw=1.5, label="pre-stroke hits")
        for j, (cl, col) in enumerate(CL):
            xs, ys, es, ns = [], [], [], []
            for i, e in enumerate(EP[1:]):
                y = s[(s.animal == an) & (s.cls == cl) & (s.epoch == e)]
                if len(y) and y.n.iloc[0] >= 5:
                    xs.append(i + (j - 2) * 0.1); ys.append(y.p_true.iloc[0]); es.append(y.p_true_sem.iloc[0]); ns.append(int(y.n.iloc[0]))
            if xs:
                ax.errorbar(xs, ys, yerr=es, fmt="-o", color=col, ms=5, lw=1.5, capsize=2, label=cl)
                for xx, nn in zip(xs, ns):           # trial counts in a column under the axis, one row per class
                    ax.text(round(xx), -0.13 - 0.075 * j, str(nn), transform=ax.get_xaxis_transform(), ha="center",
                            va="top", fontsize=7, color=col)
        ax.axhline(1 / 6, color="k", ls=":", lw=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(EP[1:]); ax.set_ylim(0, 1)
        if r == 0:
            ax.set_title(an)
        if k == 0:
            ax.set_ylabel(f"{w} window\nP(true spout), far contra")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.955))
fig.suptitle("Item 1 across epochs: far-contra trials on the frozen pre-stroke decoder by trial type "
             "(classes with >= 5 trials; chance dotted, pre-stroke hits solid)\nnumbers under each axis = trials per class (same colours)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.subplots_adjust(hspace=0.75)
save_png_svg(fig, OUT / "fig2b_attempt_decoder_epochs.png", dpi=140)
print("ok")
