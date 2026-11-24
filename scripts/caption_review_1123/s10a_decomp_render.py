"""S10A variants (caption review 11-23, ccB helper): the six-position latency decomposition in the manuscript's
`c2_latency_decomp.py` style (bars = mean over animals of per-animal medians, gray = cue -> first movement, striped =
first movement -> first lick, whiskers = SEM across animals, dots = per-animal totals in epoch colors), with a second
row that shows WHICH signal set the first movement: per spout x epoch, the share of licked trials (mean over animals)
set by each named category (3-D jaw, in-mouth lick (cam1), FaceRhythm vigor, other jaw rules) or by none (first
movement = the lick).

    python s10a_decomp_render.py <cells_<variant>.csv> <out_stem> "<title>"
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MC = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/manuscript_code/figures")
sys.path.insert(0, str(MC))
import vls_style as S  # noqa: E402

POS = [("close_ipsi", "near ipsi"), ("close_mid", "near mid"), ("close_contra", "near contra"),
       ("far_ipsi", "far ipsi"), ("far_mid", "far mid"), ("far_contra", "far contra")]
EPS = ["pre", "acute", "subacute", "chronic"]
EPL = {"pre": "pre", "acute": "ac", "subacute": "sub", "chronic": "chr"}
CW, CR = "#BDBDBD", "#F2F2F2"
CAT = {"jaw3d": ("3-D jaw", "#E69F00"), "jawcls": ("jaw (2-D; 3-D at near mid)", "#E69F00"),
       "jaw2d": ("jaw (2-D)", "#E69F00"), "mouth": ("in-mouth lick (cam1)", "#0072B2"),
       "cam1any": ("cam1 opening", "#0072B2"), "fr": ("FaceRhythm vigor", "#009E73"),
       "frpool": ("FaceRhythm vigor (pooled-pre baseline)", "#009E73"), "none": ("none (= first lick)", "#DDDDDD")}


def main(src: str, out: str, title: str = "") -> None:
    S.apply()
    mpl.rcParams["hatch.linewidth"] = 0.5
    mpl.rcParams["hatch.color"] = "0.45"
    d = pd.read_csv(src)
    cats = [c[len("share_"):] for c in d.columns if c.startswith("share_")]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.6, 2.9), sharex=True, gridspec_kw={"height_ratios": [2.0, 0.8]})
    x0, ticks, tl = 0, [], []
    for p, pl in POS:
        st = x0
        for e in EPS:
            g = d[(d.pos == p) & (d.epoch == e)]
            if g.empty:
                x0 += 1
                continue
            w, r = g.wait_med_ms.to_numpy(float), g.revup_med_ms.to_numpy(float)
            ax.bar(x0, w.mean(), 0.8, color=CW, edgecolor="0.3", lw=0.4)
            ax.bar(x0, r.mean(), 0.8, bottom=w.mean(), color=CR, edgecolor="0.3", lw=0.4, hatch="//////")
            if len(g) > 1:
                ax.errorbar(x0 - 0.15, w.mean(), yerr=w.std(ddof=1) / np.sqrt(len(w)), fmt="none", ecolor="0.15",
                            elinewidth=0.7, capsize=1.5)
                ax.errorbar(x0 + 0.15, w.mean() + r.mean(), yerr=r.std(ddof=1) / np.sqrt(len(r)), fmt="none",
                            ecolor="0.15", elinewidth=0.7, capsize=1.5)
            ax.scatter(np.full(len(g), x0) + np.linspace(-0.25, 0.25, len(g)), w + r, s=5, color=S.EP_DARK[e],
                       edgecolor="white", lw=0.3, zorder=4)
            bot = 0.0
            for c in cats:
                v = float(g[f"share_{c}"].mean()) * 100
                ax2.bar(x0, v, 0.8, bottom=bot, color=CAT[c][1], edgecolor="white", lw=0.3)
                bot += v
            ticks.append(x0)
            tl.append(EPL[e])
            x0 += 1
        ax.text((st + x0 - 1) / 2, 1.02, pl, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6.5)
        x0 += 0.6
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tl, fontsize=5.8)
    ax2.set_xlim(-0.7, x0 - 0.9)
    ax.set_ylabel("median latency (ms)")
    ax2.set_ylabel("first movement\nset by (%)", fontsize=6)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 50, 100])
    h = [plt.Rectangle((0, 0), 1, 1, fc=CW, ec="0.3", lw=0.4, label="cue → first movement"),
         plt.Rectangle((0, 0), 1, 1, fc=CR, ec="0.3", lw=0.4, hatch="//////", label="first movement → first lick")]
    ax.legend(handles=h, loc="upper left", fontsize=6, frameon=False)
    h2 = [plt.Rectangle((0, 0), 1, 1, fc=CAT[c][1], label=CAT[c][0]) for c in cats]
    ax2.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=len(h2), fontsize=5.8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=7, y=0.995)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.17, top=0.87, hspace=0.12)
    S.save(fig, out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
