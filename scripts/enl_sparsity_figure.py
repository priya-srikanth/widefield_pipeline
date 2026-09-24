"""The trial-count landscape for the ENL working-vs-stopped contrast -- what can and cannot be asked.

Priya, 2026-09-24: "we should plot the maps regardless, noting where there is sparse data."

This is the map that has to come first, because it decides which cells of every LATER map are worth
reading. It is drawn from `scripts/enl_state_counts.py`'s cached per-trial table, so it is real
counts over the 104 curated sessions rather than an estimate.

WHAT IT SHOWS AND WHY IT IS SHAPED THIS WAY. One panel per epoch, animals down, positions across,
cell = the number of STOPPED trials available after `enl_states.adjacent_window`. Stopped is the
scarce class in every cell -- `miss_working` is plentiful nearly everywhere -- so it alone sets
whether a cell is answerable, and plotting the pair would hide that behind the larger number.

NOTHING IS DROPPED. Cells below the floor are drawn with their count and hatched, not blanked:
"we could not test this" and "we tested this and found nothing" are different facts, and a figure
that omits the thin cells makes the first look like the second. The same reason `enl_decode`
reports an underpowered arm rather than filtering it.

    python -m scripts.enl_sparsity_figure --cache <csv> --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Below this many stopped trials a per-position cell is marked unreadable. Ten is the floor used
#: to pick the animals in `scripts/enl_state_counts.py`; it is a labelling threshold on this figure,
#: never a filter on the analysis.
FLOOR = 10

EPOCHS = ("pre", "acute", "subacute", "chronic")
POSITIONS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")


def figure(df, out, floor=FLOOR):
    """One panel per epoch; cell = stopped trials; hatched where below ``floor``."""
    adj = df[df.adjacent & (df.state == "stopped")]
    animals = sorted(df.animal.dropna().unique())
    grids = {}
    for ep in EPOCHS:
        g = (adj[adj.epoch == ep].groupby(["animal", "pos"]).size()
             .unstack("pos").reindex(index=animals, columns=list(POSITIONS)).fillna(0))
        grids[ep] = g.to_numpy(float)

    vmax = max(1.0, float(np.nanmax([g.max() for g in grids.values()])))
    fig, axes = plt.subplots(1, len(EPOCHS), figsize=(4.1 * len(EPOCHS), 3.0), constrained_layout=True)
    for ax, ep in zip(np.atleast_1d(axes), EPOCHS):
        g = grids[ep]
        ax.imshow(g, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                n = int(g[i, j])
                thin = n < floor
                if thin:
                    # hatch rather than blank: the cell is REPORTED as unreadable, not hidden
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               hatch="///", edgecolor="red", linewidth=0))
                ax.text(j, i, str(n), ha="center", va="center", fontsize=8,
                        color="red" if thin else ("white" if g[i, j] < vmax * .55 else "black"),
                        fontweight="bold" if thin else "normal")
        n_ok = int((g >= floor).all(axis=1).sum())
        ax.set_title(f"{ep}\n{n_ok}/{len(animals)} animals clear {floor}/position", fontsize=9)
        ax.set_xticks(range(len(POSITIONS)))
        ax.set_xticklabels([p.replace("_", "\n") for p in POSITIONS], fontsize=7)
        ax.set_yticks(range(len(animals)))
        ax.set_yticklabels(animals, fontsize=8)
    fig.suptitle("STOPPED trials available per cell (adjacent window)  —  red/hatched < "
                 f"{floor}, still shown", fontsize=10)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="CSV from scripts.enl_state_counts --cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--floor", type=int, default=FLOOR)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.cache)
    df = df[df.error.isna()] if "error" in df.columns else df
    if args.out:
        out = Path(args.out)
    else:
        from wfield_local.paths import PathResolver
        out = Path(PathResolver().root("figures_working")) / "enl_stopped_sparsity.png"
    p = figure(df, out, floor=args.floor)
    print(f"[enl_sparsity] -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
