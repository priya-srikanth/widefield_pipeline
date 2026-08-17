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


def figures(ref, out):
    """One pair of figures per BASIS, plus the agreement panel.

    The reference is computed in two independent bases on purpose, so the figures must not quietly
    show one of them: a reader who sees a single panel cannot tell whether the result is about the
    cortex or about the parcellation.
    """
    paths = []
    for bkey, block in ref.get("by_basis", {}).items():
        if block.get("animals"):
            paths += figure({"animals": block["animals"]}, out, source=bkey)
            paths += [figure_per_session({"animals": block["animals"]}, out, source=bkey)]
    paths += [figure_agreement(ref, out)]
    return [p for p in paths if p is not None]


def figure_per_session(ref, out, source="roi"):
    """Survival ratio per SESSION, pre-cue vs post-cue, one row per animal.

    The pooled number can only say what an animal does on average, and the post-stroke comparison is
    made one session at a time. This is the same quantity resolved to the unit it will be used at.
    Sessions whose undetected arm is tiny are drawn faint, because a survival ratio computed on nine
    trials should not look as solid as one computed on ninety.
    """
    animals = sorted(ref["animals"])
    rows = [(an, sorted((ref["animals"][an].get("cue") or {}).get("per_session", {})))
            for an in animals]
    rows = [(an, labs) for an, labs in rows if labs]
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 1, figsize=(11.0, 2.5 * len(rows)), squeeze=False)
    for ax, (an, labs) in zip(axes[:, 0], rows):
        x = np.arange(len(labs)); w = 0.38
        for k, al in enumerate(("precue", "cue")):
            ps = (ref["animals"][an].get(al) or {}).get("per_session", {})
            v, ns = [], []
            for lab in labs:
                d = ps.get(lab, {})
                v.append((d.get("compare") or {}).get("survival_ratio", np.nan))
                ns.append((d.get("nolick_pooled") or {}).get("n", 0))
            alphas = [min(1.0, 0.25 + n / 40.0) for n in ns]
            for xi, vi, a in zip(x + (k - 0.5) * w, v, alphas):
                ax.bar(xi, vi, w, color="tab:green" if al == "precue" else "tab:purple",
                       alpha=a, edgecolor="k", linewidth=0.4)
            if al == "cue":
                for xi, n in zip(x + (k - 0.5) * w, ns):
                    ax.text(xi, 0.02, str(n), ha="center", va="bottom", fontsize=6, rotation=90)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([l[-4:] for l in labs], fontsize=7)
        ax.set_ylabel("survival ratio"); ax.set_title(an, fontsize=9, loc="left")
    axes[0, 0].set_title(f"{rows[0][0]}   (green = pre-cue, purple = post-cue; opacity ~ n of the "
                         f"no-detected-lick arm, printed at the base)", fontsize=8, loc="left")
    fig.suptitle(f"Per-session survival of the position code without a detected lick "
                 f"[{source} basis]", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = Path(out) / f"nolick_per_session_{source}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"wrote {p.name}", flush=True)
    return p


def figure_agreement(ref, out):
    """Do the two bases reach the same verdict per animal? Rendered as text, because the honest
    answer is a sentence and a bar chart of agreement would be theatre."""
    cons = ref.get("consensus") or {}
    if not cons:
        return None
    animals = sorted(cons)
    fig, ax = plt.subplots(figsize=(11.5, 0.9 + 0.85 * len(animals)))
    ax.axis("off")
    ax.set_title("Two-basis agreement — Allen-ROI vs joint-LocaNMF", fontsize=11, loc="left")
    for i, an in enumerate(animals):
        v = cons[an]
        agree = isinstance(v, str)
        txt = v if agree else ("bases DISAGREE: " + "; ".join(
            f"{k}={str(x)[:60]}" for k, x in (v or {}).get("DISAGREEMENT", {}).items())
            if isinstance(v, dict) else "not computed")
        ax.text(0.01, 1 - (i + 0.6) / (len(animals) + 0.5), f"{an}   {txt}",
                transform=ax.transAxes, fontsize=8.5, va="center",
                color=("black" if agree else "firebrick"), wrap=True)
    fig.tight_layout()
    p = Path(out) / "nolick_basis_agreement.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


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
    if "by_basis" in ref:
        figures(ref, a.output)
    else:                                   # older single-basis reference
        figure(ref, a.output, source=ref.get("source", "locanmf"))


if __name__ == "__main__":
    main()
