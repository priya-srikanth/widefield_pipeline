"""Figures for the no-detected-lick reference.

Reads the frozen JSON written by `nolick_decoder.build_reference` and renders from it, rather than
recomputing. That keeps the picture and the frozen numbers the same object: once post-stroke
sessions arrive, a figure can be regenerated without any risk of it quietly reflecting a re-fit.

Two figures:

  * `nolick_reference_<source>.png` -- balanced accuracy per arm, pre-cue beside post-cue, with each
    arm's OWN permutation null drawn as the bar's baseline rather than a single 1/6 line. The nulls
    differ per arm (that is the entire point), so one shared chance line would misrepresent every
    bar but the engaged one.
  * `nolick_survival_<source>.png` -- the discriminating quantity: how much of the engaged code
    survives, pre-cue vs post-cue, per animal. The hypothesis lives on this figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402

ARMS = ("engaged", "late", "undetected")
COLORS = {"engaged": "tab:blue", "late": "tab:orange", "undetected": "tab:red"}
CH = 1 / 6


def _bars(ax, animal_res, title):
    x = np.arange(len(ARMS))
    w = 0.38
    for k, al in enumerate(("precue", "cue")):
        r = animal_res.get(al, {})
        vals, nulls, ns, ps = [], [], [], []
        for arm in ARMS:
            d = r.get(arm) or {}
            vals.append(d.get("balanced_accuracy", np.nan))
            nulls.append(d.get("bal_null_mean", np.nan))
            ns.append(d.get("n", 0))
            ps.append(d.get("bal_p", np.nan))
        off = (k - 0.5) * w
        hatch = None if al == "cue" else "//"
        ax.bar(x + off, vals, w, color=[COLORS[a] for a in ARMS], alpha=0.85 if al == "cue" else 0.5,
               hatch=hatch, edgecolor="k", linewidth=0.5,
               label="post-cue" if al == "cue" else "pre-cue (hatched)")
        # each arm's own null, as a short rule across its bar
        for xi, nl in zip(x + off, nulls):
            if np.isfinite(nl):
                ax.plot([xi - w / 2, xi + w / 2], [nl, nl], color="k", lw=1.6, zorder=5)
        for xi, v, n, p in zip(x + off, vals, ns, ps):
            if np.isfinite(v):
                star = "*" if np.isfinite(p) and p < 0.05 else ""
                ax.text(xi, v + 0.015, f"{n}{star}", ha="center", va="bottom", fontsize=7)
    ax.axhline(CH, color="grey", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(ARMS, fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=10)
    ax.set_ylabel("balanced accuracy (macro-recall)")


def figure(ref, out, source="locanmf"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    animals = sorted(ref["animals"])
    fig, axes = plt.subplots(1, len(animals), figsize=(4.2 * len(animals), 4.4), squeeze=False)
    for ax, an in zip(axes[0], animals):
        _bars(ax, ref["animals"][an], an)
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Position code on trials with NO DETECTED LICK — balanced accuracy vs each arm's own "
                 "permutation null (black rule); dotted = 1/6. n above bar, * = p<0.05",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p1 = out / f"nolick_reference_{source}.png"
    fig.savefig(p1, dpi=150); plt.close(fig)

    fig2, ax = plt.subplots(figsize=(1.6 * len(animals) + 2.4, 4.0))
    x = np.arange(len(animals)); w = 0.38
    for k, al in enumerate(("precue", "cue")):
        v = [((ref["animals"][an].get(al) or {}).get("compare") or {}).get("survival_ratio", np.nan)
             for an in animals]
        ax.bar(x + (k - 0.5) * w, v, w, label="pre-cue" if al == "precue" else "post-cue",
               color="tab:green" if al == "precue" else "tab:purple", edgecolor="k", linewidth=0.5)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.text(len(animals) - 0.5, 0.51, "0.5 = half the engaged code survives", fontsize=7,
            ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(animals)
    ax.set_ylabel("survival ratio (no-detected-lick / engaged, above chance)")
    ax.set_title("PRE-cue surviving while POST-cue collapses = plan formed, movement failed",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig2.tight_layout()
    p2 = out / f"nolick_survival_{source}.png"
    fig2.savefig(p2, dpi=150); plt.close(fig2)
    print(f"wrote {p1.name} + {p2.name}", flush=True)
    return [p1, p2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    ref = json.loads(Path(a.reference).read_text())
    figure(ref, a.output, source=ref.get("source", "locanmf"))


if __name__ == "__main__":
    main()
