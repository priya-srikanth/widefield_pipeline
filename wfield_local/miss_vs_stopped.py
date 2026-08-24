"""MISS-WHILE-WORKING vs STOPPED, per position, per session: is the plan there when the animal is trying?

THE QUESTION (Priya, 2026-08-23): on post-stroke trials with no lick, does the position code survive
-- and specifically on the trials where the animal is STILL WORKING the task, as opposed to the ones
after it has quit for the day?

WHY THE SPLIT IS THE WHOLE POINT. The two post-stroke failure modes are different phenomena and are
already separated by `position_coding_directions`:

  MISS WHILE WORKING  still working, fails to lick at THIS position. Position-specific, and 34-44%
                      of these trials are far_R.
  STOPPED             quit for the day, licks nowhere. Verified position-GENERAL: response ~0 at
                      every position, close included. Near-uniform across positions.

Pooling them is not a coarser version of this analysis -- it is a different one, because the two
classes differ in position composition by a total variation of 0.31-0.65 and ENL activity carries
position. A no-lick analysis that does not split them compares the spout, not the state.

WHAT THIS MODULE ADDS. Nothing is recomputed: `coding_direction.json` already stores every value per
position, per session, per class. This reads it and draws the ONE contrast the existing figures do
not put side by side -- the same position, the same session, miss against stopped -- because that
contrast is what distinguishes "the plan is intact and execution failed" from "there is no plan".

READ IT AS: 1.0 = that position's own PRE-STROKE pole; 0 = no position code. Miss clearly above zero
with stopped at zero, in the same animal and position, is the plan-intact signature.

Run: ``python -m wfield_local.miss_vs_stopped`` (after position_coding_directions).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wfield_local import config  # noqa: E402

#: Cells thinner than this are drawn hollow and excluded from the trend. A working animal rarely
#: misses at a position it can still reach, so the CLOSE cells are structurally thin -- PS95 has
#: close_center n=6 at +7.16, which is noise wearing a number.
MIN_N = 20

#: The positions this contrast can actually speak to. Close positions carry too few
#: miss-while-working trials to read, by the same structural asymmetry.
POSITIONS = ("far_L", "far_center", "far_R")

MISS, STOPPED = "poststroke_miss_working", "poststroke_stopped"


def series(rec, position, cls):
    """[(date, mean, sem, n)] for one position and class, post-stroke sessions only, date-sorted."""
    bs = rec["methods"]["dom_orth"]["positions"][position]["by_session"]
    out = []
    for lab in sorted(bs):
        date = lab.split("_")[-1]
        v = (bs[lab] or {}).get(cls) or {}
        if v.get("mean") is None:
            continue
        an = lab.split("_")[0]
        if config.session_phase(an, date) != "post":
            continue
        out.append((date, float(v["mean"]), float(v.get("sem") or np.nan), int(v.get("n") or 0)))
    return out


def fig_miss_vs_stopped(data, out_dir, window="ENL",
                        name="poststroke_miss_vs_stopped.png"):
    animals = [a for a in ("PS92", "PS93", "PS94", "PS95") if a in data]
    if not animals:
        return None
    fig, axes = plt.subplots(len(POSITIONS), len(animals), squeeze=False,
                             figsize=(3.3 * len(animals) + 1.0, 2.9 * len(POSITIONS) + 1.4),
                             sharey="row")
    for ri, pos in enumerate(POSITIONS):
        for ci, an in enumerate(animals):
            ax = axes[ri][ci]
            ax.axhline(0, color="k", lw=0.8)
            ax.axhline(1.0, color="tab:green", ls=":", lw=1.0)
            for cls, colour, mark, lbl in ((MISS, "tab:red", "o", "miss while WORKING"),
                                           (STOPPED, "tab:purple", "v", "STOPPED (quit)")):
                s = series(data[an], pos, cls)
                if not s:
                    continue
                x = np.arange(len(s))
                y = np.array([v for _, v, _, _ in s])
                e = np.array([se for _, _, se, _ in s])
                n = np.array([nn for _, _, _, nn in s])
                thin = n < MIN_N
                ax.errorbar(x[~thin], y[~thin], yerr=e[~thin], color=colour, marker=mark,
                            ms=5, lw=1.4, capsize=2, label=lbl)
                if thin.any():                     # drawn, but hollow and unjoined
                    ax.errorbar(x[thin], y[thin], yerr=e[thin], color=colour, marker=mark,
                                ms=5, lw=0, elinewidth=0.7, capsize=2, mfc="white", alpha=0.55)
                ax.set_xticks(x)
                ax.set_xticklabels([d for d, _, _, _ in s], rotation=45, ha="right", fontsize=7)
            if ri == 0:
                ax.set_title(an, fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{pos}\ncoding value", fontsize=9)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, loc="best")
            ax.tick_params(labelsize=7)
    fig.suptitle(
        f"Is the position code present when the animal is STILL WORKING but does not lick? "
        f"({window} window)\n"
        f"Same position, same session, two failure modes side by side. 1.0 (green dotted) = that "
        f"position's own PRE-STROKE pole; 0 = no position code. MISS WHILE WORKING is "
        f"position-specific; STOPPED is the animal having quit for the day and is position-GENERAL. "
        f"Miss above zero with stopped AT zero is the plan-intact / execution-failed signature. "
        f"Hollow points are n<{MIN_N} and are not joined -- a working animal rarely misses at a "
        f"position it can still reach, so those cells are structurally thin and must not be read.",
        fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = Path(out_dir) / name
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--window", default="ENL", choices=("ENL", "cue", "lick"))
    args = ap.parse_args(argv)
    out = args.output or Path(config.resolver().root("figures_working"))
    src = Path(out) / "coding_direction.json"
    if not src.exists():
        print(f"[miss_vs_stopped] {src} not found -- run position_coding_directions first",
              flush=True)
        return 1
    d = json.load(open(src))
    if args.window not in d:
        print(f"[miss_vs_stopped] window {args.window!r} not in {src.name}", flush=True)
        return 1
    p = fig_miss_vs_stopped(d[args.window], out, window=args.window)
    print(f"wrote {p}" if p else "[miss_vs_stopped] nothing to plot", flush=True)
    return 0 if p else 1


if __name__ == "__main__":
    raise SystemExit(main())
