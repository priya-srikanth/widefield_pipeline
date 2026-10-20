"""Presentation figure for the pre-cue SLOW-BAND caveat (Priya 10-19 / 10-20: handoff to-do 6).

Reads `imaging_encoding/qc/slowband_decode.csv` (`scripts/slowband_decode_check.py`, 46 sessions) and
`qc/position_block_summary.csv` (`scripts/position_block_durations.py`) and draws, per animal (rule 8: animals first):
  A  pre-cue spout decode from the adopted signal, with the >= 300 s component REMOVED, the >= 300 s component ALONE,
     and the >= 600 s component alone (pre-stroke sessions); chance = 1/6
  B  the same for the post-cue window (the slow band decodes the same pre and post cue -> no cue timing)
  C  the fraction of the above-chance pre-cue decode that survives without the slow band, per animal
  D  slow-band-alone decode vs the session's median position-block duration (longer blocks -> more slow-band decode)
Output: qc/slowband_caveat.png. The caveat in one line: the headline pre-cue code is ~85 % fast, but its magnitude
includes a slow block-level component, largest in the animals with the longest position blocks (PS92, PS93).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANCE = 1 / 6
VARIANTS = [("adopted", "adopted\nsignal"), ("fast300", "slow band\nremoved"), ("slow300", "slow band\nALONE (>=300 s)"),
            ("slow600", ">=600 s\nalone")]
ACOL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    q = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc"
    D = pd.read_csv(q / "slowband_decode.csv")
    B = pd.read_csv(q / "position_block_summary.csv")[["label", "median"]]
    D = D.merge(B, on="label", how="left")
    P = D[D["set"] == "pre"]
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.2))
    for ax, win, title in ((axs[0][0], "pre", "A  PRE-CUE spout decode (pre-stroke sessions)"),
                           (axs[0][1], "post", "B  POST-CUE spout decode (pre-stroke sessions)")):
        for vi, (v, _lab) in enumerate(VARIANTS):
            col = f"{v}_{win}"
            am = P.groupby("animal")[col].mean()
            ax.bar(vi, am.mean(), 0.6, color="0.85", edgecolor="0.4")
            for k, (a, val) in enumerate(am.items()):
                ax.plot(vi - 0.18 + 0.12 * k, val, "o", color=ACOL[a], ms=7, label=a if vi == 0 else None)
        ax.axhline(CHANCE, color="k", ls="--", lw=1)
        ax.text(len(VARIANTS) - 0.5, CHANCE + 0.01, "chance 1/6", ha="right", fontsize=8.5)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels([lab for _, lab in VARIANTS], fontsize=9)
        ax.set_ylabel("decode accuracy (6 spouts, block CV)")
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_ylim(0, max(0.9, float(P[[f"{v}_{win}" for v, _ in VARIANTS]].max().max()) + 0.05))
    axs[0][0].legend(fontsize=8.5, loc="upper right")
    # C: fraction of the above-chance pre-cue decode retained without the slow band
    ax = axs[1][0]
    P = P.assign(retained=(P.fast300_pre - CHANCE) / (P.adopted_pre - CHANCE))
    for k, (a, g) in enumerate(P.groupby("animal")):
        r = g.retained.clip(-0.5, 1.5)
        ax.plot(np.full(len(r), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(r)), r, ".", color=ACOL[a],
                alpha=0.6)
        ax.plot(k, (g.fast300_pre.mean() - CHANCE) / (g.adopted_pre.mean() - CHANCE), "D", color=ACOL[a], ms=9,
                mec="k")
    ax.axhline(1, color="0.5", lw=0.8)
    ax.set_xticks(range(P.animal.nunique()))
    ax.set_xticklabels(sorted(P.animal.unique()))
    ax.set_ylabel("above-chance pre-cue decode retained\nwithout the slow band (fraction)")
    ax.set_title("C  share of the pre-cue code that is FAST\n(diamond = animal mean ratio; dots = sessions)",
                 fontsize=10.5, loc="left")
    # D: slow-alone decode vs block duration
    ax = axs[1][1]
    for a, g in D.groupby("animal"):
        ax.plot(g["median"], g.slow300_pre, "o", color=ACOL[a], label=a, alpha=0.75)
    ok = D["median"].notna() & D.slow300_pre.notna()
    r = np.corrcoef(D.loc[ok, "median"], D.loc[ok, "slow300_pre"])[0, 1]
    ax.axhline(CHANCE, color="k", ls="--", lw=1)
    ax.set_xlabel("median position-block duration in the session (min)")
    ax.set_ylabel("pre-cue decode from the >=300 s component ALONE")
    ax.set_title(f"D  longer blocks -> more slow-band decode\n(r = {r:.2f}, all sessions)", fontsize=10.5,
                 loc="left")
    ax.legend(fontsize=8.5)
    fig.suptitle("Pre-cue spout decode: the SLOW-BAND caveat. The >=300 s component of the corrected signal carries "
                 "block-level position\nstructure (same before and after the cue); the headline pre-cue code survives "
                 "mostly without it, but its size includes it -- largest where position blocks are longest.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = q / "slowband_caveat.png"
    writeguard.assert_writable(q)
    fig.savefig(out, dpi=130)
    print(P.groupby("animal")[["adopted_pre", "fast300_pre", "slow300_pre", "slow600_pre", "adopted_post",
                               "fast300_post", "slow300_post"]].mean().round(3).to_string())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
