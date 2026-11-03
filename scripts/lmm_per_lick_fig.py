"""Slide layout of the FaceRhythm per-lick LMM figure (Priya 11-03 via a8: "bottom plot is way too spread out; top plot is
too busy"): two rows of six factors grouped by preferred lick direction (row 1 right + centre, row 2 left), licks per
trial as a narrow panel at the right, short labels, faded session markers. Statistics are READ from the existing
``<stem>_lmm.csv`` (written by scripts/lmm_change_fig.py), so stars and intervals are unchanged.

    python -m scripts.lmm_per_lick_fig <sessions_long.csv> <stem>_lmm.csv --out <png>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.lmm_change_fig import draw_axis, legend_handles
    from wfield_local import sig_marks as sm
    from wfield_local import writeguard
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions")
    ap.add_argument("stats")
    ap.add_argument("--labels", required=True, help="file: comma list cell=label (label = 'factor / direction lick')")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    S = pd.read_csv(a.sessions)
    T = pd.read_csv(a.stats)
    T["star"] = T["star"].fillna("")
    lab = dict(kv.split("=", 1) for kv in Path(a.labels).read_text().strip().split(",") if "=" in kv)
    name = {c: v.split(" / ")[0] for c, v in lab.items()}
    pl = [c for c in lab if c.startswith("pl_")]
    row1 = [c for c in pl if c.startswith("pl_right_")] + [c for c in pl if c.startswith("pl_centre_")]
    row2 = [c for c in pl if c.startswith("pl_left_")]
    lpt = [c for c in lab if c.startswith("lpt_")]
    name.update({"lpt_right": "right\n(contra)", "lpt_left": "left\n(ipsi)"})
    out = Path(a.out)
    writeguard.assert_writable(out)
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0.55, wspace=0.12)
    axes = []
    for r, (cs, ttl) in enumerate(((row1, "factors per RIGHT (contra) lick, and per centre lick (last two)"),
                                   (row2, "factors per LEFT (ipsi) lick"))):
        ax = fig.add_subplot(gs[r, 0])
        draw_axis(ax, S, T, cs, {c: name[c].replace("_", "_\n", 1) if len(name[c]) > 12 else name[c] for c in cs},
                  dot_s=7, dot_alpha=0.3, label_size=7.5)
        ax.set_title(ttl, fontsize=9)
        ax.set_ylabel("log2 vs own pre", fontsize=8.5)
        axes.append(ax)
    ax = fig.add_subplot(gs[:, 1])
    draw_axis(ax, S, T, lpt, name, dot_s=7, dot_alpha=0.3, label_size=8)
    ax.set_title("licks per trial", fontsize=9)
    fam = T.groupby("family").p.apply(lambda q: int(np.isfinite(q).sum()))
    fig.suptitle("FaceRhythm per lick (FR 2.2): mixed model, separate between-animal spread per epoch (df = animals - 1)\n"
                 f"* p < .05, ** Holm within the factor panels ({fam.max()} tests) and within licks per trial "
                 f"({fam.min()})", fontsize=9)
    fig.legend(handles=legend_handles(S), fontsize=7, ncol=9, loc="lower center", frameon=False)
    sm.reserve_headroom(fig)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.17)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
