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
import json
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


def imaging_counts(decode_results):
    """``{(animal, epoch, position): n}`` -- the stopped trials the DECODE ACTUALLY SCORED.

    THIS FIGURE WAS WRONG UNTIL 2026-09-24 and this is the fix. It was drawn from the behaviour
    table alone, which the counting script's own docstring calls "an upper bound on the imaging
    set" -- and the gap is not small: PS93 pre-stroke is **67 behaviour-side and 40 in the decode**,
    PS92 **10 and 6**. The docstring's guarantee is only that an EMPTY cell is empty in both, which
    is what it was written for (feasibility). It says nothing about a cell reading 12 that is really
    7, and this figure hatches at a floor of 10 -- so the overstatement lands exactly on the
    judgement the figure exists to support.

    Taken from `enl_decode`'s own output rather than recounted, so the figure cannot disagree with
    the analysis it is describing (rule 9). `recall_by_position` carries the per-position n of the
    scored set, which is the number that decides whether a cell was answerable.
    """
    out = {}
    for r in decode_results or []:
        rec = (((r.get("shared") or {}).get("stopped") or {}).get("recall_by_position") or {})
        for pos, cell in rec.items():
            out[(r.get("animal"), r.get("epoch"), pos)] = int(cell["n"])
    return out


def figure(df, out, floor=FLOOR, imaging=None):
    """One panel per epoch; cell = stopped trials; hatched where below ``floor``.

    With ``imaging`` the cell shows the DECODE's own count and the behaviour-table count beneath it,
    and the hatch follows the decode count. Without it the figure falls back to behaviour counts and
    says so in the title, because a figure that cannot tell you which universe it is counting is
    worse than one that admits it.
    """
    imaging = imaging or {}
    adj = df[df.adjacent & (df.state == "stopped")]
    animals = sorted(df.animal.dropna().unique())
    grids, img = {}, {}
    for ep in EPOCHS:
        g = (adj[adj.epoch == ep].groupby(["animal", "pos"]).size()
             .unstack("pos").reindex(index=animals, columns=list(POSITIONS)).fillna(0))
        grids[ep] = g.to_numpy(float)
        img[ep] = np.array([[imaging.get((a, ep, p), np.nan) for p in POSITIONS]
                            for a in animals], float)

    vmax = max(1.0, float(np.nanmax([g.max() for g in grids.values()])))
    fig, axes = plt.subplots(1, len(EPOCHS), figsize=(4.1 * len(EPOCHS), 3.0), constrained_layout=True)
    for ax, ep in zip(np.atleast_1d(axes), EPOCHS):
        g, gi = grids[ep], img[ep]
        # colour by the count that DECIDES -- the decode's own where it exists, behaviour otherwise
        shown = np.where(np.isfinite(gi), gi, g)
        ax.imshow(shown, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                beh, dec = int(g[i, j]), gi[i, j]
                have = np.isfinite(dec)
                n = int(dec) if have else beh
                thin = n < floor
                if thin:
                    # hatch rather than blank: the cell is REPORTED as unreadable, not hidden
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               hatch="///", edgecolor="red", linewidth=0))
                col = "red" if thin else ("white" if shown[i, j] < vmax * .55 else "black")
                ax.text(j, i, str(n), ha="center", va="center" if not have else "bottom",
                        fontsize=8, color=col, fontweight="bold" if thin else "normal")
                if have:
                    # the behaviour-table count beneath, so the GAP is visible rather than implied
                    ax.text(j, i + 0.06, f"({beh})", ha="center", va="top", fontsize=5.5,
                            color=col, alpha=0.75)
        n_ok = int((np.where(np.isfinite(gi), gi, g) >= floor).all(axis=1).sum())
        src = "decode" if np.isfinite(gi).any() else "BEHAVIOUR ONLY"
        ax.set_title(f"{ep}  [{src}]\n{n_ok}/{len(animals)} animals clear {floor}/position",
                     fontsize=9)
        ax.set_xticks(range(len(POSITIONS)))
        ax.set_xticklabels([p.replace("_", "\n") for p in POSITIONS], fontsize=7)
        ax.set_yticks(range(len(animals)))
        ax.set_yticklabels(animals, fontsize=8)
    any_img = any(np.isfinite(v).any() for v in img.values())
    fig.suptitle(
        "STOPPED trials per cell  —  red/hatched < "
        f"{floor}, still shown\n"
        + ("large = what the DECODE scored, (small) = behaviour table, which OVERSTATES it"
           if any_img else
           "BEHAVIOUR-TABLE COUNTS ONLY — an UPPER BOUND on the imaging set (PS93 pre: 67 vs 40). "
           "Pass --decode-json for the real counts"),
        fontsize=9.5)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="CSV from scripts.enl_state_counts --cache")
    ap.add_argument("--decode-json", action="append", default=None,
                    help="output of `enl_decode --out`, repeatable (one per epoch). WITHOUT it "
                         "the figure falls back to behaviour-table counts, which OVERSTATE the "
                         "imaging set, and says so in the title")
    ap.add_argument("--out", default=None)
    ap.add_argument("--floor", type=int, default=FLOOR)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.cache)
    df = df[df.error.isna()] if "error" in df.columns else df
    results = []
    for pth in args.decode_json or []:
        p_ = Path(pth)
        if not p_.is_file():
            print(f"[enl_sparsity] MISSING {p_} -- those cells fall back to behaviour", flush=True)
            continue
        results += json.loads(p_.read_text(encoding="utf-8"))
    img = imaging_counts(results)
    if not img:
        print("[enl_sparsity] no decode counts -- drawing BEHAVIOUR-TABLE UPPER BOUNDS", flush=True)
    if args.out:
        out = Path(args.out)
    else:
        from wfield_local.paths import PathResolver
        out = Path(PathResolver().root("figures_working")) / "enl_stopped_sparsity.png"
    p = figure(df, out, floor=args.floor, imaging=img)
    print(f"[enl_sparsity] -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
