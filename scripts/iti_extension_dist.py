"""Distribution of lick EXTENSION (sagittal extension out of the mouth, and 3-D reach) per animal x epoch, with the
threshold used to define 'longer licks' for the angle-decoder r (Priya 10-27: "is longer than median fair? should we make
a single protrusion threshold? show me the protrusion distribution across epochs").
Threshold candidates drawn: per-animal PRE median (current; one fixed value per animal, applied to every epoch) and one
cohort-wide value (the pooled pre median). Prints, per animal x epoch, the fraction of licks above each threshold.
Output figures/iti_extension_dist.png, <E>/iti_licks/extension_thresholds.csv. Usage: python scripts/iti_extension_dist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    P = pd.read_csv(il.OUT / "where_angle_reach_preds.csv.gz")
    cohort = {v: P[P.epoch == "pre"][v].median() for v in ("act_sag", "act_reach")}
    rows = []
    fig, axs = plt.subplots(4, 4, figsize=(17, 13), squeeze=False)
    for i, (st, v, lab) in enumerate((("iti", "act_sag", "sagittal extension"), ("trial", "act_sag", "sagittal extension"),
                                      ("iti", "act_reach", "3-D reach"), ("trial", "act_reach", "3-D reach"))):
        for j, a in enumerate(AN):
            ax = axs[i, j]
            g = P[(P.set == st) & (P.animal == a)]
            own = g[g.epoch == "pre"][v].median()
            bins = np.linspace(0, np.nanpercentile(P[v], 99.5), 60)
            for e in EPS:
                x = g[g.epoch == e][v].dropna()
                if len(x) < 20:
                    continue
                ax.hist(x, bins=bins, density=True, histtype="step", lw=1.5, color=EPOCH_LINE[e])
                rows.append(dict(set=st, measure=v, animal=a, epoch=e, n=len(x), median=x.median(),
                                 frac_above_own_pre_median=float((x > own).mean()),
                                 frac_above_cohort_pre_median=float((x > cohort[v]).mean())))
            ax.axvline(own, color="k", lw=1.2, ls="--")
            ax.axvline(cohort[v], color="0.5", lw=1.2, ls=":")
            fr = {r["epoch"]: r["frac_above_own_pre_median"] for r in rows if r["set"] == st and r["measure"] == v
                  and r["animal"] == a}
            ax.text(0.98, 0.97, "\n".join(f"{e}: {100 * fr[e]:.0f}% above" for e in EPS if e in fr),
                    transform=ax.transAxes, ha="right", va="top", fontsize=7)
            ax.set_title(f"{a}, {'ITI' if st == 'iti' else 'cued'}: {lab}", fontsize=9)
            if i == 3 or i == 1:
                ax.set_xlabel("mm", fontsize=8)
    fig.legend(handles=[Line2D([], [], color=EPOCH_LINE[e], lw=2, label=e) for e in EPS]
               + [Line2D([], [], color="k", ls="--", label="animal's pre median (current threshold)"),
                  Line2D([], [], color="0.5", ls=":", label="cohort pre median (single threshold)")],
               loc="upper left", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("BEHAVIOUR: lick extension per animal x epoch (3-D, extension peak). Text = % of each epoch's licks above "
                 "the animal's pre median.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(il.OUT / "figures" / "iti_extension_dist.png", dpi=90)
    T = pd.DataFrame(rows)
    T.to_csv(il.OUT / "extension_thresholds.csv", index=False)
    pd.set_option("display.width", 200)
    print("cohort pre medians:", {k: round(v, 2) for k, v in cohort.items()})
    print(T.pivot_table(index=["measure", "set", "animal"], columns="epoch",
                        values=["median", "frac_above_own_pre_median", "frac_above_cohort_pre_median"])
          .reindex(columns=EPS, level=1).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
