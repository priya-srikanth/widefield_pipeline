"""PER-ANIMAL per-spout Haufe maps (Priya 10-27: per-animal version of the per-spout maps; far contra + a comparison
spout). From the held-out per-session one-vs-rest maps of `iti_spout_pattern_stats.py` (cache spout_pattern_sessions.pkl,
cued licks): per animal x spout x epoch, the mean of that epoch's session maps (/ the animal's pre RMS for the spout);
similarity to the animal's pre map in each title. Rows = animal x spout (far contra, close contra), columns = epochs.
Output figures/cued_spout_animal_panels.png (cued licks; renamed 10-27 from iti_..., the ITI-family prefix). Usage: python scripts/iti_spout_animal_panels.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from iti_session_pattern_tests import _imaging_grid  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
SPOUTS = ["far_contra", "close_contra"]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SM = pickle.loads((il.OUT / "spout_pattern_sessions.pkl").read_bytes())
    to_img, ol = _imaging_grid()
    rows = [(a, sp) for a in AN for sp in SPOUTS]
    fig, axs = plt.subplots(len(rows), len(EPS), figsize=(2.6 * len(EPS), 2.3 * len(rows) + 1), squeeze=False)
    for i, (a, sp) in enumerate(rows):
        v = SM.get((a, sp), {})
        maps = {e: v[e].mean(0) for e in EPS if e in v}
        lim = np.nanpercentile(np.abs(np.concatenate(list(maps.values()))), 99) if maps else 1
        for j, e in enumerate(EPS):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if e not in maps:
                ax.axis("off")
                ax.set_title(f"{e}: no map (< 2 sessions)", fontsize=7)
                continue
            ax.imshow(to_img(maps[e]), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            sim = "" if e == "pre" else f", r to pre {np.corrcoef(maps[e], maps['pre'])[0, 1]:.2f}"
            ax.set_title(f"{e} (n={v[e].shape[0]}){sim}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{a}\n{sp.replace('_', ' ')}", fontsize=8.5)
    fig.suptitle("NEURAL: per-ANIMAL one-vs-rest Haufe maps for far contra and close contra (cued licks),\nmean of "
                 "held-out per-session maps, / the animal's pre RMS for the spout; one colour scale per row.\nr to pre = "
                 "pixel "
                 "correlation with the animal's own pre map. ipsi (lesion) = image LEFT. PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_size_inches()[1]))
    fig.savefig(il.OUT / "figures" / "cued_spout_animal_panels.png", dpi=90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
