"""SECONDARY: the ITI same-place read-out on stricter ITI lick sets (Priya 10-27: "keep all ITI licks, report 2 s quiet,
1 s far, both as secondary"; iti_lick_timing_audit.py: decoder ITI licks have no gap or event-distance rule and the ITI
is only ~3.1 s long).

Sets (WFIELD_ITI_FILTER in iti_lick_order; the frozen decoder is refit and rescored on each set):
  all        every ITI lick (PRIMARY)
  gap2       no lick onset in the preceding 2 s
  far1       >= 1 s after the response-window end and >= 1 s before the next spout arrival
  gap2far1   both
Read-out = like-for-like same-place LOCAL-LINEAR reference (lick_matched_readout.py --ref loclin; 10-27 late), change vs pre, nested bootstrap;
dots = animals (ANIMAL_MARKER); k/n = animals with the cohort sign. Decoder basis: BASIS (tag).
Output figures/lick_iti_filter_compare.png (figdata figdata/lick_iti_filter_compare.pkl; --plot-only redraws).
Usage: python scripts/lick_iti_filter_compare.py [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
import iti_licks as il  # noqa: E402

BASIS = "areas_splitr8"     # Allen + ALM 40 % / tjM1 50 % split (ROI v8, final)
SETS = [("", "all ITI licks (primary)"), ("_gap2", "2 s quiet before"), ("_far1", ">= 1 s from window end and spout arrival"),
        ("_gap2far1", "both")]
EP = ["acute", "subacute", "chronic"]
AN = ["PS92", "PS93", "PS94", "PS95"]
STEM = "lick_iti_filter_compare"


def compute() -> dict:
    rows = []
    for tag, lab in SETS:
        T = pd.read_csv(il.OUT / f"matched_loclin_{BASIS}{tag}_tests.csv")
        T = T[(T.scope == "all spouts") & (T.measure == "resid") & (T.set == "iti") & (T.cls == "all") & T.epoch.isin(EP)]
        P = pd.read_csv(il.OUT / f"lick_order_preds_{BASIS}{tag}.csv.gz", usecols=["set", "epoch", "label"])
        n = P[P.set == "iti"].groupby("epoch").size()
        for _, r in T.iterrows():
            rows.append(dict(set=lab, epoch=r.epoch, point=r.point, lo=r.lo, hi=r.hi, n_dir=r.n_dir, n=r.n,
                             n_licks=int(n.get(r.epoch, 0)), **{a: r.get(f"animal_{a}", np.nan) for a in AN}))
    return {"R": pd.DataFrame(rows)}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    R = F["R"]
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    cols = ["k", "#1b9e77", "#d95f02", "#7570b3"]
    w = 0.18
    for i, (_t, lab) in enumerate(SETS):
        g = R[R.set == lab].set_index("epoch").reindex(EP)
        x = np.arange(len(EP)) + (i - 1.5) * w
        ax.errorbar(x, g.point, yerr=[g.point - g.lo, g.hi - g.point], fmt="o", color=cols[i], ms=6, capsize=3,
                    label=lab, zorder=3)
        for j, ep in enumerate(EP):
            for a in AN:
                ax.plot(x[j] + 0.05, g.loc[ep, a], ANIMAL_MARKER[a], mfc="none", mec=cols[i], ms=4, alpha=0.7, zorder=2)
            ax.text(x[j], g.loc[ep, "hi"] + 0.03, f"{int(g.loc[ep, 'n_dir'])}/{int(g.loc[ep, 'n'])}",
                    ha="center", fontsize=7, color=cols[i])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(np.arange(len(EP)), EP)
    ax.set_ylabel("ITI read-out vs same-place pre licks,\nchange vs pre (mm; + = contralesional)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    ax.set_title("NEURAL, SECONDARY: ITI like-for-like read-out on stricter ITI lick sets (decoder refit on each set; "
                 "Allen + ALM/tjM1 split basis).\nPoint = cohort change vs pre, 95% nested bootstrap CI; open markers = "
                 "animals; k/n = animals with the cohort sign. PROVISIONAL", fontsize=8.5)
    fig.savefig(il.OUT / "figures" / f"{STEM}.png", dpi=110)
    plt.close(fig)


def main() -> int:
    if figdata.plot_only():
        F = figdata.load_fd(STEM)
    else:
        F = compute()
        figdata.save_fd(STEM, F)
    print(F["R"].round(3).to_string())
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
