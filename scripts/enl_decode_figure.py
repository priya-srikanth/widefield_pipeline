"""The ENL sensory-vs-plan result: does the pre-cue position code survive when the animal STOPS?

Priya, 2026-09-24. Draws `wfield_local.enl_decode --out`'s JSON. Nothing is recomputed here, so the
figure and the printed report cannot drift apart.

THREE PANELS, AND THE ORDER IS THE ARGUMENT.

  A  READOUT 4 -- one `success`-trained decoder, blocks held out across every arm, scoring all
     three. Because the decoder and its training set are identical across the bars, the only thing
     differing between them is which trials are being read.
  B  THE SURVIVING FRACTION -- panel A's `stopped` bar over its `miss_working` bar, both measured
     ABOVE THEIR OWN NULL. Both arms are no-lick trials, so outcome, reward and movement are matched
     and only engagement differs. This is the number the analysis exists to produce.
  C  WHERE IT LIVES -- the per-position recalls that panel A's `stopped` bar is the MEAN of. Drawn
     because "pooled across positions" invites the reading that positions were merged; they are not,
     the decoder is six-way throughout and panel A averages what panel C shows. Descriptive only:
     at 40-96 trials a cell these recalls are far too noisy to rank.

EVERY BAR CARRIES ITS OWN NULL, and that is not decoration. The nulls are NOT all 1/6: they are
permutation nulls with the decoder's predictions held fixed, so they move with prediction bias, and
`common_positions` can restrict an arm to fewer than six positions, which lifts its null outright
(PS92 pre-stroke: three shared positions, null 0.234). A single 1/6 chance line drawn across this
figure would be wrong for at least one bar and would flatter it.

NOTHING BELOW THRESHOLD IS BLANKED. An arm at its null is hatched and still drawn with its number; a
ratio that cannot be computed is labelled with the REASON rather than left empty. "We could not test
this" and "we tested this and found nothing" are different facts and only one is evidence of
absence -- the same rule `enl_decode.underpowered` and `enl_sparsity_figure` follow.

    python -m scripts.enl_decode_figure --json <enl_decode out.json> --out <png>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

#: Panel A's arms, left to right: the ceiling, the outcome-matched denominator, the test arm.
ARMS = ("success", "miss_working", "stopped")
ARM_COLOR = {"success": "#4C72B0", "miss_working": "#55A868", "stopped": "#C44E52"}

POSITIONS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")

#: Per-position cells below this many trials are hatched in panel C. A LABEL, never a filter -- the
#: recall is still drawn. Matches `scripts.enl_sparsity_figure.FLOOR`.
FLOOR = 10


def _rows(results):
    """Animals in file order, skipping cells `enl_decode` could not run at all."""
    return [r for r in results if r.get("animal") and not r.get("skipped") and "shared" in r]


def _panel_a(ax, rows):
    """Balanced accuracy per arm, each against ITS OWN permutation null."""
    w, x = 0.26, np.arange(len(rows))
    for k, arm in enumerate(ARMS):
        off = (k - 1) * w
        for i, r in enumerate(rows):
            d = r["shared"].get(arm, {})
            if "skipped" in d:
                ax.text(x[i] + off, 0.02, "n/a", ha="center", va="bottom", fontsize=6,
                        rotation=90, color="grey")
                continue
            above = d["above_null_balanced"]
            ci = r.get("bootstrap", {}).get("arms", {}).get(arm, {}).get("ci")
            ax.bar(x[i] + off, d["balanced_accuracy"], w, color=ARM_COLOR[arm],
                   edgecolor="black" if above else "red", linewidth=0.8,
                   hatch=None if above else "///",
                   yerr=None if not ci else [[max(0, d["balanced_accuracy"] - ci[0])],
                                             [max(0, ci[1] - d["balanced_accuracy"])]],
                   error_kw={"ecolor": "0.15", "capsize": 2.5, "lw": 1.1})
            # The arm's OWN null, as a tick over its own bar. Deliberately NARROWER than the bar:
            # at full width the three ticks of a group abut and read as one chance line drawn
            # across the figure, which is the single most misleading thing this panel could say.
            ax.plot([x[i] + off - w * 0.38, x[i] + off + w * 0.38],
                    [d["bal_null_mean"]] * 2, color="black", lw=1.6, zorder=5)
            top = max(d["balanced_accuracy"], ci[1] if ci else 0)     # clear the CI whisker
            ax.text(x[i] + off, top + 0.014, f"{d['n']}", ha="center", va="bottom", fontsize=6,
                    color="red" if not above else "black")
    ax.set_xticks(x)
    ax.set_xticklabels([_xlabel(r) for r in rows], fontsize=8)
    ax.set_ylabel("balanced accuracy (macro-recall)")
    ax.set_title("A  One success-trained decoder, blocks held out across all arms\n"
                 "black tick = that arm's own permutation null · hatched = at null · n above bar",
                 fontsize=9, loc="left")
    # Built by hand, NOT from the bars: an autogenerated legend takes its swatch from whichever bar
    # was drawn first, so PS92's at-null `stopped` bar would put a red hatch on the key and imply
    # every stopped arm is at null. The hatch means one arm's verdict, not one arm's identity.
    ax.legend(handles=[Patch(facecolor=ARM_COLOR[a], edgecolor="black", label=a) for a in ARMS]
              + [Patch(facecolor="white", edgecolor="red", hatch="///", label="at its own null")],
              fontsize=7, frameon=False, ncol=4, loc="upper right")
    ax.set_ylim(0, 0.83)


def _xlabel(r):
    """Animal, plus the position restriction when `common_positions` had to impose one."""
    drop = r.get("shared_positions", {}).get("dropped") or []
    return r["animal"] if not drop else f"{r['animal']}\n({6 - len(drop)}/6 pos)"


#: Panel B's two rungs: (result key, bootstrap key, label, colour). The spout is in position on
#: every arm, so each rung asks what the REMOVED component was contributing.
RUNGS = (("clean_ratio_working_vs_success", "miss_working_over_success",
          "miss_working / success\n(loses execution + reward)", "#55A868"),
         ("clean_ratio_vs_working", "stopped_over_miss_working",
          "stopped / miss_working\n(loses engagement; OUTCOME-MATCHED)", "#C44E52"))


def _panel_b(ax, rows):
    """The two rungs of the ladder, with bootstrap CIs and an explicit REASON where undefined."""
    x, w = np.arange(len(rows)), 0.34
    for k, (key, bkey, label, colour) in enumerate(RUNGS):
        off = (k - 0.5) * w
        for i, r in enumerate(rows):
            v = r.get(key, {}).get("value")
            num = r.get(key, {}).get("numerator", "stopped")
            if v is None or not r["shared"].get(num, {}).get("above_null_balanced"):
                ax.text(i + off, 0.06, f"n/a\n{num[:4]}\nat null", ha="center", va="bottom",
                        fontsize=6, color="red")
                continue
            ci = r.get("bootstrap", {}).get("ratios", {}).get(bkey, {}).get("ci")
            # the label goes on the first bar ACTUALLY DRAWN, not on i == 0: the stopped rung's
            # first animal is "n/a", so keying on the index drops that rung from the legend
            drawn = any(lab.get_label() == label for lab in ax.containers)
            ax.bar(i + off, v, w, color=colour, edgecolor="black", linewidth=0.8,
                   label=None if drawn else label,
                   yerr=None if not ci or not np.isfinite(ci[0]) else
                   [[max(0, v - ci[0])], [max(0, ci[1] - v)]],
                   error_kw={"ecolor": "0.15", "capsize": 2.5, "lw": 1.1})
            ax.text(i + off, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.text(-0.65, 1.02, "code fully intact", fontsize=6, color="grey", ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([r["animal"] for r in rows], fontsize=8)
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylabel("fraction of the rung above,\nboth above their own null")
    ax.set_title("B  The ladder — the spout is in position on EVERY arm, so each rung asks what the\n"
                 "removed component contributed. Bars are bootstrap 95% CIs; a CI crossing 1.0 "
                 "means\nthe rung is NOT resolvable, which is not the same as no drop",
                 fontsize=8.5, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.set_ylim(0, 1.75)


def _panel_c(ax, rows):
    """The per-position recalls panel A's `stopped` bar is the MEAN of. Descriptive only."""
    g = np.full((len(rows), len(POSITIONS)), np.nan)
    ns = np.zeros_like(g)
    for i, r in enumerate(rows):
        rec = r["shared"].get("stopped", {}).get("recall_by_position", {})
        for j, p in enumerate(POSITIONS):
            cell = rec.get(p)
            if cell:
                g[i, j], ns[i, j] = cell["recall"], cell["n"]
    hi = float(np.nanmax(g)) if np.isfinite(g).any() else 0.6
    ax.imshow(np.nan_to_num(g), cmap="magma", vmin=0, vmax=max(0.6, hi), aspect="auto")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            n = int(ns[i, j])
            if not np.isfinite(g[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="white"))
                ax.text(j, i, "no\ntrials", ha="center", va="center", fontsize=6, color="red")
                continue
            if n < FLOOR:                       # hatch, never blank
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///",
                                           edgecolor="red", linewidth=0))
            # `magma` is near-black at recall 0, so plain coloured text vanishes in exactly the
            # cells that matter most. Outline every label instead of picking a colour per cell.
            ax.text(j, i, f"{g[i, j]:.2f}\nn={n}", ha="center", va="center", fontsize=6.5,
                    color="#FFD400" if n < FLOOR else "white",
                    fontweight="bold" if n < FLOOR else "normal",
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])
    ax.set_xticks(range(len(POSITIONS)))
    ax.set_xticklabels([p.replace("_", "\n") for p in POSITIONS], fontsize=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["animal"] for r in rows], fontsize=8)
    ax.set_title(f"C  Per-position recall, STOPPED arm — panel A is each row's MEAN\n"
                 f"positions are NOT merged; the decoder is six-way. hatched < {FLOOR} trials.\n"
                 f"DESCRIPTIVE ONLY — far too noisy at these n to rank", fontsize=8.5, loc="left")


def figure(results, out):
    rows = _rows(results)
    if not rows:
        raise SystemExit("[enl_decode_figure] no scorable animal in the JSON")
    fig = plt.figure(figsize=(14.5, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15])
    _panel_a(fig.add_subplot(gs[0, :]), rows)
    _panel_b(fig.add_subplot(gs[1, 0]), rows)
    _panel_c(fig.add_subplot(gs[1, 1]), rows)

    r0 = rows[0]
    n_def = sum(1 for r in rows if r["shared"].get("stopped", {}).get("above_null_balanced"))
    n_sw = sum(1 for r in rows
               if r.get("bootstrap", {}).get("differences", {})
               .get("success_minus_miss_working", {}).get("p", 1) < 0.05)
    n_ws = sum(1 for r in rows
               if r.get("bootstrap", {}).get("differences", {})
               .get("miss_working_minus_stopped", {}).get("p", 1) < 0.05)
    fig.suptitle(
        f"Pre-cue position code without a motor plan  —  {r0.get('epoch')}-stroke, "
        f"align={r0.get('align')}, {r0.get('source')} basis (per-animal frozen)\n"
        f"success > miss_working in {n_sw}/{len(rows)} animals · "
        f"miss_working > stopped in only {n_ws}/{len(rows)} · "
        f"stopped above its own null in {n_def}/{len(rows)} (rule 8's bar is THREE)",
        fontsize=10.5)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="output of `enl_decode --out`")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    results = json.loads(Path(args.json).read_text(encoding="utf-8"))
    if args.out:
        out = Path(args.out)
    else:
        from wfield_local.paths import PathResolver
        out = Path(PathResolver().root("figures_working")) / "enl_decode_pre.png"
    print(f"[enl_decode_figure] -> {figure(results, out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
