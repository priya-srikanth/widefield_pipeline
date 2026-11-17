"""PREVIEW of an LMM version of an epoch figure: per-session changes vs the animal's own pre (open markers, shape =
animal, color = epoch), LMM estimate (filled dot) + 95 % CI (t, df = n_animals - 1), animal bootstrap 95 % beside it
(gray; describes these animals), LMM stars in the top band (* p < .05, ** Holm within the figure).
usage: a3_lmm_preview.py <sessions.csv> <lmm.csv> <figure-name> <out.png> [ylabel]"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
sys.path.insert(0, ".")
from wfield_local import analysis_kit as ak, cat_points as cp, sig_marks as sm, epoch_figures as ef
sess, lmm, name, out = sys.argv[1:5]
from wfield_local import run_mode  # noqa: E402
RUN_TAG = run_mode.require_input_tag(sess, "a3_lmm_preview_fig")    # fifth review N5
ylab = sys.argv[5] if len(sys.argv) > 5 else "change vs own pre"
S = pd.read_csv(sess); H = pd.read_csv(lmm); H = H[H.figure == name]
EP = ["acute", "subacute", "chronic"]
cells = list(dict.fromkeys(S.position))
fig, ax = plt.subplots(figsize=(max(4, 1.9 * len(cells) + 1.5), 4.6))
for xi, cell in enumerate(cells):
    D = S[(S.position == cell) & np.isfinite(S.value)]
    pre = D[D.epoch == "pre"].groupby("animal").value.mean()
    pred = {a: list(g.value) for a, g in D[D.epoch == "pre"].groupby("animal")}
    for k, e in enumerate(EP):
        x = xi + (k - 1) * 0.27; col = ef.epoch_dot_color(e)
        po = D[(D.epoch == e) & D.animal.isin(pre.index)]
        if po.empty:
            continue
        rng = np.random.default_rng(0)
        for a, g in po.groupby("animal"):
            ax.scatter(x - 0.06 + rng.uniform(-0.03, 0.03, len(g)), g.value - pre[a], marker=cp.animal_marker(a),
                       facecolors="none", edgecolors=col, s=16, lw=0.8, alpha=0.6, zorder=2)
        postd = {a: list(g.value) for a, g in po.groupby("animal")}
        got = ak.paired_animal_draws(postd, pred, lambda v, _r=None: float(np.mean(v)), np.random.default_rng(1), 2000)
        if got[0] is not None:
            blo, bhi = np.percentile(got[1], [2.5, 97.5])
            ax.plot([x + 0.10] * 2, [blo, bhi], color="0.6", lw=3, solid_capstyle="butt", zorder=3)
        r = H[(H.position.astype(str) == str(cell)) & (H.epoch == e)]
        if len(r) and np.isfinite(r.iloc[0].get("beta", np.nan)):
            r = r.iloc[0]; tc = stats.t.ppf(0.975, max(1, r.n_animals - 1))
            ax.errorbar(x + 0.04, r.beta, yerr=tc * r.se, fmt="o", color=col, mec="k", ms=7, capsize=3,
                        elinewidth=1.4, zorder=4)
            if r.star_lmm in ("*", "**"):
                sm.star_top(ax, x + 0.04, r.star_lmm, color=col)
ax.axhline(0, color="0.5", lw=0.8, ls="--", zorder=1)
ax.set_xticks(range(len(cells))); ax.set_xticklabels([str(c).replace("_", "\n") for c in cells], fontsize=8)
ax.set_ylabel(ylab); ax.spines[["top", "right"]].set_visible(False)
h = [Line2D([], [], ls="none", marker="o", color=ef.epoch_dot_color(e), mec="k", ms=7, label=f"{e} (model estimate)")
     for e in EP]
h += [Line2D([], [], ls="-", lw=1.4, color="k", label="mixed model 95 % CI"),
      Line2D([], [], ls="-", lw=3, color="0.6", label="animal bootstrap 95 % (these 4 animals)")]
h += cp.animal_handles(["PS92", "PS93", "PS94", "PS95"])
fig.legend(handles=h, fontsize=7, ncol=5, loc="lower center", frameon=False)
ax.set_title(f"{name}  -  LMM version (PREVIEW)\nmixed model, separate between-animal spread per epoch; "
             "* p < .05, ** Holm across the figure's cells", fontsize=9)
sm.reserve_headroom(fig); fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.text(0.995, 0.005, f"input run tag: {RUN_TAG or 'default'}", ha="right", va="bottom", fontsize=6, color="0.5")
fig.savefig(out, dpi=130); print(out)
