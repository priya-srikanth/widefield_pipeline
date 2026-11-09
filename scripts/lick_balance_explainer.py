"""HOW THE BALANCE IS CALCULATED -- a worked example on real data for the page (Priya 10-28: "replace the balance figures
with this new balance; build in an explanation of how it's calculated").

One animal (EXAMPLE_ANIMAL), cued FIRST post-cue licks, the primary definition (lick_recovery_primary):
  A. each lick's response = mean dF/F 0.1..0.5 s after the lick minus -1.0..-0.5 s before it, summed over the 5 pooled
     areas (ALM, tjM1, SSp-m, SSp-n, SSs) of one hemisphere; pre-stroke licks vs lick angle with the straight-line
     reference (here ONE line on the pooled response for display; the analysis fits one line per area and hemisphere
     and lick order, and sums the predictions -- printed check: the display gain vs the analysis gain);
  B. an acute and a chronic session's licks on the same axes, against the pre line;
  C. per session, gain = sum(observed) / sum(predicted), contralesional and ipsilesional; balance = log2 contra gain -
     log2 ipsi gain, over all of the animal's sessions in time, with the far-contra aim.
Output figure lick_balance_explainer.png (figdata + --plot-only).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_subsets as bs  # noqa: E402
import lick_recovery_primary as rp  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

EXAMPLE_ANIMAL = "PS94"
DN = rp.CUED_WIN


def compute() -> dict:
    P = pd.read_csv(lo.preds_path())
    P = P[(P.animal == EXAMPLE_ANIMAL) & (P.set == "trial")].reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    L = pd.concat([bs._session(x) for x in sorted(P.label.unique())], ignore_index=True)
    P = P.merge(L, on=["set", "label", "k"], how="left", validate="one_to_one")
    P = P[P.order == 0].copy()
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    for s in ("c", "i"):
        P[f"pool_{s}"] = P[[f"{DN}|{a}|{s}" for a in rp.POOL]].sum(1)
    pre = P[P.epoch == "pre"].dropna(subset=["angle", "pool_c", "pool_i"])
    lines = {s: np.polyfit(pre.angle, pre[f"pool_{s}"], 1) for s in ("c", "i")}
    S = pd.read_csv(il.OUT_W / "balsub_sessions.csv")
    S = S[(S.set == rp.lrm.C1SET) & (S.animal == EXAMPLE_ANIMAL)].copy()
    for s in ("c", "i"):
        S[f"gain_{s}"] = (S[[f"{DN}|{a}|obs_{s}" for a in rp.POOL]].sum(1)
                          / S[[f"{DN}|{a}|pred_{s}" for a in rp.POOL]].sum(1))
    S["balance"] = np.log2(S.gain_c) - np.log2(S.gain_i)
    S = S.sort_values("day")
    post = S[S.epoch != "pre"]
    ex = {"acute": post[post.epoch == "acute"].label.iloc[0], "chronic": post[post.epoch == "chronic"].label.iloc[-1]}
    chk = []
    for ep, lab in ex.items():
        q = P[P.label == lab].dropna(subset=["angle", "pool_c", "pool_i"])
        for s in ("c", "i"):
            gd = q[f"pool_{s}"].sum() / np.polyval(lines[s], q.angle).sum()
            ga = float(S.loc[S.label == lab, f"gain_{s}"].iloc[0])
            chk.append({"epoch": ep, "label": lab, "side": s, "gain_display": gd, "gain_analysis": ga})
    chk = pd.DataFrame(chk)
    print("== display (one pooled line) vs analysis (per-area lines) gains\n" + chk.round(3).to_string(index=False))
    return {"P": P[["label", "epoch", "angle", "pool_c", "pool_i"]], "lines": lines, "S": S[["label", "epoch", "day",
            "aim", "gain_c", "gain_i", "balance"]], "ex": ex, "chk": chk}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    P, lines, S, ex, chk = F["P"], F["lines"], F["S"], F["ex"], F["chk"]
    fig, axs0 = plt.subplots(1, 5, figsize=(27, 5.8), gridspec_kw={"width_ratios": [0.95, 1, 1, 0.8, 1.2]})
    # step 0: the areas pooled, on the Allen map (image left = ipsilesional)
    from frozen_area_maps import outline
    from iti_area_maps import split_image
    img, nm = split_image(il.labels()[0])
    rgb = np.ones((*img.shape, 3))
    rgb[img < 0] = 1.0
    for k, n in enumerate(nm):
        base = n.rsplit("_", 1)[0]
        if base in rp.POOL:
            rgb[img == k] = (0.77, 0.31, 0.32) if n.endswith("_right") else (0.30, 0.45, 0.69)
        elif (img == k).any():
            rgb[img == k] = (0.93, 0.93, 0.93)
    ax = axs0[0]
    ax.imshow(rgb, interpolation="nearest")
    ax.imshow(np.where(outline(img) > 0, 0.0, np.nan), cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
    for a in rp.POOL:
        for side in ("left", "right"):
            if f"{a}_{side}" in nm:
                yy, xx = np.nonzero(img == nm.index(f"{a}_{side}"))
                if len(yy):
                    ax.text(xx.mean(), yy.mean(), a, ha="center", va="center", fontsize=7, color="w", fontweight="bold")
    ax.set_title("step 0: the 5 pooled areas (ALM, tjM1, SSp-m, SSp-n, SSs)\nblue = ipsilesional (lesion side, image left), "
                 "red = contralesional", fontsize=9.5)
    ax.axis("off")
    axs = axs0[1:]
    xs = np.linspace(P.angle.quantile(0.01), P.angle.quantile(0.99), 50)
    for j, (s, lab) in enumerate((("c", "CONTRALESIONAL"), ("i", "IPSILESIONAL"))):
        ax = axs[j]
        pre = P[P.epoch == "pre"]
        ax.plot(pre.angle, pre[f"pool_{s}"], ".", color="0.75", ms=3, label="pre-stroke first licks")
        ax.plot(xs, np.polyval(lines[s], xs), "k-", lw=2, label="pre reference line")
        for ep, lb in ex.items():
            q = P[P.label == lb]
            ax.plot(q.angle, q[f"pool_{s}"], "o", color=EPOCH_LINE[ep], ms=4, alpha=0.75, label=f"{ep} session {lb}")
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_xlabel("lick angle (deg)", fontsize=9)
        ax.set_ylabel("lick response, 5 areas summed\n(dF/F x1e3, 0.1-0.5 s minus -1..-0.5 s)", fontsize=9)
        ax.set_title(f"step 1-2: {lab} hemisphere\nresponse of each first post-cue lick vs its angle", fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax = axs[2]
    x = np.arange(len(ex))
    for k, (s, col, lab) in enumerate((("c", "#C44E52", "contralesional gain"), ("i", "#4C72B0", "ipsilesional gain"))):
        v = [S.loc[S.label == lb, f"gain_{s}"].iloc[0] for lb in ex.values()]
        ax.bar(x + (k - 0.5) * 0.36, v, width=0.34, color=col, label=lab)
        for xi, vv in zip(x, v, strict=True):
            ax.text(xi + (k - 0.5) * 0.36, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=8)
    top = max(S.loc[S.label.isin(list(ex.values())), ["gain_c", "gain_i"]].to_numpy().max(), 1.2)
    for xi, lb in zip(x, ex.values(), strict=True):
        b = S.loc[S.label == lb, "balance"].iloc[0]
        ax.text(xi, top * 1.12, f"{'implied dir.' if rp.BAL_DISPLAY == 'direction' else 'balance'}\n{rp.bal_disp(b):+.2f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xticks(x, [f"{ep}\n{lb}" for ep, lb in ex.items()], fontsize=8)
    ax.set_ylim(0, top * 1.35)
    ax.set_title("step 3-4: gain = sum observed / sum predicted\nbalance = log2(contra gain) - log2(ipsi gain)"
                 + ("\nshown as implied direction = -balance" if rp.BAL_DISPLAY == "direction" else ""), fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax = axs[3]
    ax.plot(S.day, rp.bal_disp(S.balance), "k-", lw=0.8, alpha=0.5)
    for ep, h in S.groupby("epoch"):
        ax.plot(h.day, rp.bal_disp(h.balance), "o", color="0.75" if ep == "pre" else EPOCH_LINE[ep], ms=6, label=ep)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("day relative to stroke", fontsize=9)
    ax.set_ylabel(rp.BAL_LABEL, fontsize=8.5)
    ax2 = ax.twinx()
    ax2.plot(S.day, S.aim, "s--", color="#8172B3", ms=4, lw=0.8, alpha=0.8, label="far-contra first-lick aim")
    ax2.set_ylabel("far-contra first-lick aim (deg vs pre)", fontsize=9, color="#8172B3")
    ax.set_title(f"{EXAMPLE_ANIMAL}: {rp.BAL_SHORT} per session over recovery (gray = pre, left-out)\nand the behavioral "
                 "recovery axis it tracks", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="lower right")
    for a in axs:
        a.spines["top"].set_visible(False)
    d = chk.groupby("side")[["gain_display", "gain_analysis"]].apply(lambda t: (t.gain_display - t.gain_analysis).abs().max())
    fig.suptitle(f"HOW THE BALANCE IS CALCULATED (worked example, {EXAMPLE_ANIMAL}, cued first post-cue licks)\n"
                 "1: each lick's response (orofacial areas + SSs of one hemisphere).  2: what PRE-stroke licks at the same "
                 "angle predict (reference line).  3: gain = observed / predicted, summed over the session's licks, per "
                 "hemisphere.\n4: balance = log2(contra gain) - log2(ipsi gain); 0 = same contra / ipsi relation as before the "
                 "stroke, < 0 = contralesional relatively weaker.  Display: one pooled line per hemisphere = the sum of the "
                 f"analysis' per-area lines (gain difference {float(d.max()):.2f}). PROVISIONAL"
                 + ("\n5: shown as implied direction = -(contra - ipsi); + = looks like a more contralesional lick (in the "
                    "pre-stroke code each hemisphere responds more to licks toward the opposite side, so a weaker "
                    "contralesional balance reads like a more contralesional lick; same sign as lick angle, aim and the "
                    "decoder offset)" if rp.BAL_DISPLAY == "direction" else ""), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.81 if rp.BAL_DISPLAY == "direction" else 0.84))
    save_png_svg(fig, il.OUT_W / "figures" / "lick_balance_explainer.png", dpi=95)
    plt.close(fig)


def trial_set_sentence() -> str:
    """The lick set of the worked example (Priya 11-09): one animal's FIRST post-cue licks of the KEY set."""
    return bs.set_sentence(("trial_1",), full=False, inside=False, subset=False,
                           extra=[f"animal {EXAMPLE_ANIMAL} only", "lick angle finite"])


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    if plot_only():
        plot(load_fd("lick_balance_explainer"))
        return 0
    F = compute()
    save_fd("lick_balance_explainer", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
