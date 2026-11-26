"""Encoder-part decomposition for ALL SIX spouts (CC-B follow-up 11-25): is the acute lick-part fall specific to far contra?

Input: the per-SESSION long table of scripts/lp_encoder_decompose.py (`lp_encoder_decompose_<folder>[_lick].csv`: per
session x position x region the evoked value of data / full / each part / residual). Works for the per-session fits
(parts task / instructed / uninstructed / pos_part) and for the frozen frame (parts timing / target / exec_* / lick /
other / contact / ... ): --mode persession | frozen picks the columns shown.

Per animal x spout x region x epoch: mean over sessions, change vs the animal's own (August) pre (table first), then
pooled = analysis_kit.boot_delta (nested animals -> sessions, seed 0) per part; * = 95% CI excludes 0, drawn in the top
band. Spout order: close ipsi, mid, contra, far ipsi, mid, contra (orofacial_cohort.POS_ORDER convention).

    python g4_decomp_allspouts.py <sessions.csv> <out dir> <label> --mode persession|frozen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wfield_local.analysis_kit import boot_delta  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
PLAB = ["close ipsi", "close mid", "close contra", "far ipsi", "far mid", "far contra"]
REG = ["SSp-m_left", "SSp-m_right", "cortex"]
RLAB = {"SSp-m_left": "SSp-m ipsilesional", "SSp-m_right": "SSp-m contralesional", "cortex": "cortex"}
PARTS = {"persession": [("data", "measured", "k"), ("instructed", "lick part (instructed)", "#e66101"),
                        ("pos_part", "target part", "#6a3d9a"), ("uninstructed", "uninstructed", "#5e7ea0"),
                        ("residual", "residual", "#1b9e77")],
         "frozen": [("data", "measured", "k"), ("lick", "lick part", "#e66101"), ("target", "target part", "#6a3d9a"),
                    ("other", "other (spontaneous)", "#5e7ea0"), ("residual", "residual", "#1b9e77")]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("label")
    ap.add_argument("--mode", choices=("persession", "frozen"), default="persession")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    parts = PARTS[a.mode]
    T = pd.read_csv(a.sessions)
    T = T[T.region.isin(REG) & T.pos.isin(POS)]
    cols = [p for p, _, _ in parts]
    rows, prow = [], []
    for reg in REG:
        for pos in POS:
            g = T[(T.region == reg) & (T.pos == pos)]
            A = g.groupby(["animal", "epoch"])[cols].mean()
            pre_data = {an: A.loc[(an, "pre"), "data"] if (an, "pre") in A.index else np.nan for an in AN}
            for ep in ("acute", "chronic"):
                for p, name, _ in parts:
                    vals = {an: A.loc[(an, ep), p] - A.loc[(an, "pre"), p] for an in AN
                            if (an, ep) in A.index and (an, "pre") in A.index}
                    rows.append(dict(region=reg, pos=pos, epoch=ep, part=name,
                                     **{an: vals.get(an, np.nan) * 1e3 for an in AN},
                                     **{f"{an}_of_pre": vals.get(an, np.nan) / pre_data[an] if pre_data.get(an) else np.nan for an in AN}))
                    post = {an: g[(g.animal == an) & (g.epoch == ep)][p].to_numpy(float) for an in AN}
                    pre = {an: g[(g.animal == an) & (g.epoch == "pre")][p].to_numpy(float) for an in AN}
                    com = [an for an in AN if len(post[an]) and len(pre[an])]
                    iv = boot_delta({an: post[an] for an in com}, {an: pre[an] for an in com}, np.random.default_rng(0))
                    prow.append(dict(region=reg, pos=pos, epoch=ep, part=name, point=iv.point * 1e3, lo=iv.lo * 1e3, hi=iv.hi * 1e3,
                                     n_animals=len(com), n_same_sign=int(sum(np.sign(vals[an]) == np.sign(iv.point) for an in com)),
                                     n_sessions_post=int(sum(len(post[an]) for an in com))))
    R, P = pd.DataFrame(rows), pd.DataFrame(prow)
    R.to_csv(a.out / "per_animal.csv", index=False, float_format="%.4f")
    P.to_csv(a.out / "pooled.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(f"== {a.label}: per animal, change vs own pre, x1e3 (table first)")
    print(R.set_index(["region", "pos", "epoch", "part"])[AN].round(2).to_string())
    print("== pooled (x1e3, 95% nested CI)")
    print(P.round(2).to_string(index=False))
    # ---- figure: rows = regions, cols = epochs; x = spouts, bars = parts
    fig, axs = plt.subplots(len(REG), 2, figsize=(15, 3.6 * len(REG)), sharey="row")
    nb = len(parts)
    for r, reg in enumerate(REG):
        for c, ep in enumerate(("acute", "chronic")):
            ax = axs[r, c]
            top = 0.0
            for j, pos in enumerate(POS):
                for k, (_p, name, col) in enumerate(parts):
                    x = j * (nb + 1) + k
                    q = P[(P.region == reg) & (P.pos == pos) & (P.epoch == ep) & (P.part == name)].iloc[0]
                    ax.bar(x, q.point, color=col, width=0.85, label=name if (j == 0) else None)
                    ax.plot([x, x], [q.lo, q.hi], color="0.3", lw=0.9)
                    rr = R[(R.region == reg) & (R.pos == pos) & (R.epoch == ep) & (R.part == name)].iloc[0]
                    for an in AN:
                        if np.isfinite(rr[an]):
                            ax.plot(x, rr[an], MK[an], mfc="none", mec="0.25", ms=3.5, zorder=5)
                    top = max(top, q.hi, *[rr[an] for an in AN if np.isfinite(rr[an])])
                    if q.lo > 0 or q.hi < 0:
                        ax.text(x, 0.95, "*", ha="center", va="bottom", fontsize=11, color="#6a3d9a", transform=ax.get_xaxis_transform())
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_xticks([j * (nb + 1) + (nb - 1) / 2 for j in range(6)]); ax.set_xticklabels(PLAB, fontsize=8)
            ax.set_title(f"{RLAB[reg]}, {ep} minus pre", fontsize=10)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if c == 0:
                ax.set_ylabel("evoked dF/F x1e3, epoch minus own pre")
            ax.margins(y=0.25)
    axs[0, 0].legend(fontsize=7.5, frameon=False, ncol=nb, loc="lower left")
    fig.suptitle(f"{a.label}: response change split into encoder parts, all six spouts; bars = pooled change (95% nested CI), "
                 "markers = animals (PS92 o, PS93 square, PS94 triangle, PS95 diamond), * = CI excludes 0", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out / "decomp_allspouts.png", dpi=120); plt.close(fig)
    print("->", a.out / "decomp_allspouts.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
