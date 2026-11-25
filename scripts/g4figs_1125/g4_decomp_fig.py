"""Figs 17.1 / 17.2 on the PER-SESSION G4 fits (the page's versions read the frozen pre-stroke frame, which is lane E's
wave-1 work): the SSp-m far-contra lick-window response split into the per-session encoder's parts.

Input: scripts/lp_encoder_decompose.py --align lick output (`lp_encoder_decompose_<folder>_lick_by_animal.csv`):
per animal x position x region x epoch the evoked (0..1 s after the first post-cue lick minus -0.3..-0.1 s) value of
data, full prediction, task / instructed / uninstructed parts, pos_part and residual, epoch minus the animal's own
August pre.

  fig17_2 (epochs): SSp-m far contra, bars = change from pre per part, per animal markers (shape = animal).
  fig17_1 (pooled): each part's change divided by the animal's own pre data response, mean of animals, markers =
  animals; the number under each bar = animals sharing the sign of the mean (as the page's 17.1). No bootstrap:
  `d_*` are per-animal values (one per animal and epoch), so there is nothing to resample within animal here.

    python g4_decomp_fig.py <by_animal.csv> <out dir> <label>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
EP = ["acute", "subacute", "chronic"]
PARTS = [("d_data", "measured", "k"), ("d_full", "predicted", "#b2abd2"), ("d_instructed", "lick part", "#e66101"),
         ("d_pos_part", "target part", "#6a3d9a"), ("d_uninstructed", "uninstructed", "#5e7ea0"),
         ("d_residual", "residual", "#1b9e77")]


def main() -> int:
    src, out, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(src)
    D = D[D.pos == "far_contra"]
    rows = []
    # ---- 17.2-style: SSp-m ipsi / contra, change from pre per part, per animal
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, reg, side in zip(axs, ["SSp-m_left", "SSp-m_right"], ["ipsilesional", "contralesional"]):
        g = D[D.region == reg]
        for k, ep in enumerate(EP):
            for j, (col, name, c) in enumerate(PARTS):
                x = k * (len(PARTS) + 1) + j
                vals = {a: g[(g.animal == a) & (g.epoch == ep)][col].mean() for a in AN}
                vals = {a: v for a, v in vals.items() if np.isfinite(v)}
                m = np.mean(list(vals.values())) * 1e3
                ax.bar(x, m, color=c, width=0.8, label=name if k == 0 else None)
                for a, v in vals.items():
                    ax.plot(x, v * 1e3, MK[a], mfc="none", mec="0.3", ms=4)
                rows.append(dict(region=reg, epoch=ep, part=name, mean_x1e3=m, n=len(vals),
                                 n_same_sign=int(sum(np.sign(v) == np.sign(m) for v in vals.values())),
                                 **{a: vals.get(a, np.nan) * 1e3 for a in AN}))
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks([k * (len(PARTS) + 1) + (len(PARTS) - 1) / 2 for k in range(3)]); ax.set_xticklabels(EP)
        ax.set_title(f"SSp-m {side}, far contra, lick window"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axs[0].set_ylabel("evoked dF/F x1e3, epoch minus own pre"); axs[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"{label}: far-contra SSp-m response change split into the PER-SESSION encoder parts (markers = PS92 o, PS93 s, PS94 ^, PS95 D)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / "fig17_2_decomp_epochs_persession.png", dpi=130); plt.close(fig)
    # ---- 17.1-style: pooled, each part's change / own pre data response
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    prow = []
    for ax, reg, side in zip(axs, ["SSp-m_left", "SSp-m_right"], ["ipsilesional", "contralesional"]):
        g = D[D.region == reg]
        for k, ep in enumerate(["acute", "chronic"]):
            for j, (col, name, c) in enumerate(PARTS):
                x = k * (len(PARTS) + 1) + j
                vals = {}
                for a in AN:
                    r = g[(g.animal == a) & (g.epoch == ep)]
                    if len(r) and np.isfinite(r[col].iloc[0]) and r.pre_data.iloc[0] != 0:
                        vals[a] = r[col].iloc[0] / r.pre_data.iloc[0]
                m = np.mean(list(vals.values()))
                ax.bar(x, m, color=c, width=0.8, label=name if k == 0 else None)
                for a, v in vals.items():
                    ax.plot(x, v, MK[a], mfc="none", mec="0.3", ms=4)
                ns = int(sum(np.sign(v) == np.sign(m) for v in vals.values()))
                ax.text(x, min(0, min(vals.values())) - 0.08, f"{ns}/{len(vals)}", ha="center", fontsize=7, color="0.3")
                prow.append(dict(region=reg, epoch=ep, part=name, mean_of_pre=m, n=len(vals), n_same_sign=ns,
                                 **{a: vals.get(a, np.nan) for a in AN}))
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks([k * (len(PARTS) + 1) + (len(PARTS) - 1) / 2 for k in range(2)]); ax.set_xticklabels(["acute", "chronic"])
        ax.set_title(f"SSp-m {side}, far contra, lick window"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axs[0].set_ylabel("change / own pre response"); axs[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"{label}: encoder parts, pooled: each part's change divided by the animal's own pre response, mean of animals; n = animals with the sign of the mean", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / "fig17_1_decomp_pooled_persession.png", dpi=130); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "fig17_2_table.csv", index=False, float_format="%.4f")
    pd.DataFrame(prow).to_csv(out / "fig17_1_table.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(pd.DataFrame(prow).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
