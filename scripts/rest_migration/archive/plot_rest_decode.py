"""PICTURE of the rest-position decode -- the diagnostics were all text-only until now.

WHY THIS EXISTS. `rest_position_decode` and `rest_position_permutation` print tables and save
nothing. Every rest-migration conclusion on 2026-09-14/15 was therefore reported as numbers in a
chat window and a markdown table, with no figure anyone could look at -- which is exactly the
position the deck exists to avoid for every other result in this pipeline.

IT PARSES A SAVED RUN LOG rather than recomputing, because the decode is ~16 min of 8-thread work
and the nightly render owns the cores. That makes this script a REPORTER, not a measurement: it can
only be as right as the log it is handed, and it prints the run's own totals so a mismatch is
visible. `rest_position_decode` should learn to save a CSV; until it does, this is the bridge.

WHAT IT DRAWS
  left   per-session observed vs its own circular-shift null, one point per session, by epoch.
         The null is per-session and already contains drift-through-blocks, so the DISTANCE FROM
         THE DIAGONAL is the effect -- not the distance from 1/6.
  right  obs - null by epoch, animals overlaid, with the cohort mean -- and the reason the ANIMALS
         are drawn rather than the mean alone. The cohort reads 0.197 pre -> 0.097 acute -> 0.121
         subacute -> 0.245 chronic, which was described in chat as "the same acute dip as the task
         readout". THE FIGURE KILLED THAT: PS95 RISES at acute, so 3 of 4 animals dip, and the
         chronic rise is carried by PS92/PS93 with PS95 flat at baseline. The far-contra task
         deficit replicates 4/4; this does not. A pooled mean hid a reversal, which is the same
         lesson the composition test taught two days earlier.

THE DECODERS ARE PER SESSION -- within-session block-CV, `GroupKFold` by ~6-trial position block,
`LogisticRegression` fit inside each session's own folds. NOT a frozen pre-stroke model. Each epoch
therefore reads how much position information THAT SESSION'S OWN rest carries, not how much of the
pre-stroke code survives, so the chronic rise could be a DIFFERENT code. Only a frozen decoder
separates those (Priya, 2026-09-15).

    python -m scripts.rest_migration.archive.plot_rest_decode --log <path> [--out <dir>]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

EPOCHS = ("pre", "acute", "subacute", "chronic")
ECOL = {"pre": "#4C72B0", "acute": "#C44E52", "subacute": "#DD8452", "chronic": "#55A868"}
ACOL = {"PS92": "#4C72B0", "PS93": "#DD8452", "PS94": "#55A868", "PS95": "#C44E52"}

ROW = re.compile(
    r"\.\.\s+(PS\d+)_(\d{4})\s+\[(\w+)\]\s+n=\s*(\d+)\s+obs\s+([\d.]+)\s+null\s+([\d.]+)\s+p=([\d.]+)")


def parse(log: Path):
    rows = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ROW.search(line)
        if m:
            a, d, e, n, o, nu, p = m.groups()
            rows.append(dict(animal=a, date=d, epoch=e, n=int(n),
                             obs=float(o), null=float(nu), p=float(p)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("E:/cue_lick/rest_migration"))
    a = ap.parse_args()

    rows = parse(a.log)
    if len(rows) < 20:
        raise SystemExit(f"only parsed {len(rows)} sessions from {a.log} -- wrong log?")
    a.out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ---- left: observed vs its own null -------------------------------------------------------
    for e in EPOCHS:
        r = [x for x in rows if x["epoch"] == e]
        if not r:
            continue
        ax1.scatter([x["null"] for x in r], [x["obs"] for x in r], s=34, alpha=0.85,
                    color=ECOL[e], edgecolors="none", label=f"{e} (n={len(r)})")
    lim = max(max(x["obs"] for x in rows), max(x["null"] for x in rows)) * 1.08
    ax1.plot([0, lim], [0, lim], ls="--", lw=0.9, color="0.4", zorder=0)
    ax1.axhline(1 / 6, ls=":", lw=0.8, color="0.6", zorder=0)
    ax1.text(lim * 0.985, 1 / 6, " chance 1/6", fontsize=8, color="0.45", va="bottom", ha="right")
    above = sum(1 for x in rows if x["obs"] > x["null"])
    ax1.set(xlim=(0, lim), ylim=(0, lim), xlabel="circular-shift null (per session)",
            ylabel="observed decode accuracy")
    ax1.set_title(f"Position decoded FROM REST\n{above}/{len(rows)} sessions above their own null",
                  fontsize=11)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    # ---- right: obs - null by epoch, per animal + cohort mean ---------------------------------
    xs = np.arange(len(EPOCHS))
    for an in sorted({x["animal"] for x in rows}):
        ys = [np.mean([x["obs"] - x["null"] for x in rows
                       if x["epoch"] == e and x["animal"] == an] or [np.nan]) for e in EPOCHS]
        ax2.plot(xs, ys, "-o", ms=4.5, lw=1.0, alpha=0.65, color=ACOL.get(an, "0.5"), label=an)
    coh = [np.mean([x["obs"] - x["null"] for x in rows if x["epoch"] == e] or [np.nan])
           for e in EPOCHS]
    ax2.plot(xs, coh, "-o", ms=8, lw=2.4, color="0.15", label="cohort", zorder=5)
    ax2.axhline(0, ls="--", lw=0.9, color="0.4")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{e}\n(n={sum(1 for x in rows if x['epoch'] == e)})" for e in EPOCHS])
    ax2.set_ylabel("observed − null")
    ax2.set_title("Epoch trajectory -- the cohort mean dips acutely, but PS95 does NOT\n"
                  "(3 of 4 animals dip; chronic rise carried by PS92/PS93)", fontsize=11)
    ax2.legend(fontsize=8, frameon=False, ncol=2)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.suptitle("Rest carries position information (restdock05) -- read against the NULL, "
                 "not against chance: the null already contains drift-through-blocks",
                 fontsize=11.5, y=0.99)
    # THE CAVEAT TRAVELS ON THE FIGURE, not only in the handoff -- and the FIRST clause is the one
    # that was missing when this was described in chat as "the same acute dip as the task readout".
    # That was the COHORT MEAN talking. PS95 moves the other way, and between-animal replication is
    # the stronger claim than a pooled mean -- which is the whole lesson of the composition test.
    fig.text(0.5, 0.030,
             "DECODERS ARE PER SESSION (within-session block-CV, GroupKFold by ~6-trial position "
             "block) -- NOT a frozen pre-stroke model. So each epoch reads how much position "
             "information that session's own rest carries, not how much of the pre-stroke code "
             "survives; the chronic rise could be a DIFFERENT code, which only a frozen decoder "
             "separates.", ha="center", fontsize=8, color="0.35")
    fig.text(0.5, 0.006,
             "Acute dip NOT established: PS95 RISES at acute (3 of 4 animals dip), the chronic rise "
             "is carried by PS92/PS93 with PS95 flat, n=16 at both acute and chronic, and block "
             "composition across epochs and drift are unexamined. Parsed from a saved run log.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    p = a.out / "rest_position_decode_restdock05.png"
    fig.savefig(p, dpi=160)
    print(f"wrote {p}")
    for e in EPOCHS:
        r = [x for x in rows if x["epoch"] == e]
        if r:
            print(f"  {e:9s} n={len(r):3d}  obs {np.mean([x['obs'] for x in r]):.3f}  "
                  f"null {np.mean([x['null'] for x in r]):.3f}  "
                  f"diff {np.mean([x['obs'] - x['null'] for x in r]):.3f}  "
                  f"above {sum(1 for x in r if x['obs'] > x['null'])}/{len(r)}")


if __name__ == "__main__":
    main()
