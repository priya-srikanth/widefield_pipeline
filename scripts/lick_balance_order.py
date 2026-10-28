"""Hemispheric BALANCE over LICK ORDER, by epoch (Priya 10-28: "is it possible to plot contra/ipsi ratio over lick
number across epochs").

Per lick (the frozen KEY-row decoder's lick set, `iti_lick_order.preds_path()`): balance = X_right - X_left in the
decoder's own window (mean 0..0.2 s from the lick peak; the decoder inputs, x1e3 dF/F) per pair (ALM, tjM1, SSp-m,
SSp-n) and the ORofacial pool (mean of the four). A per-lick contra / ipsi RATIO is not defined (the decoder inputs carry no
per-lick baseline and can be ~0 or negative), so the balance is a difference; + = more contralesional.
Two references, both per animal from its PRE licks of the same set:
  spout   balance minus the animal's pre mean at the same spout (as 13.2's 'vs pre at the spout')
  angle   balance minus the animal's pre linear fit on the lick angle (asin(side / reach) of the 3-D tongue vector), i.e.
          the balance AT THE SAME LICK ANGLE (removes the change of landing with lick order, 13.2)
Lick order = position in the cued sequence (0 = first post-cue lick) or in the ITI bout (0 = bout start), capped at 9.
Curves: per animal x epoch x order mean, then mean +- SEM over animals; bars = orders where an epoch differs from pre
(the 14.1 per-bin test with order as the bin: paired nested bootstrap animals > sessions, BH-FDR across the 10 orders).
Cells (animal x epoch x order) with < MIN_CELL licks are dropped.
Outputs <OUT_W>/balance_order_licks.csv.gz, balance_order_sig.csv; figure lick_balance_order.png (figdata + --plot-only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_reference as lbr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

PAIRS = [*lbr.PAIRS, "orofacial"]
CAP = lo.ORDER_CAP
MIN_CELL = 20                          # licks per animal x epoch x order cell
SETS = {"trial": "CUED (order after the cue)", "iti": "ITI (order within the bout)"}
REFS = {"spout": "vs the animal's pre mean at the SPOUT", "angle": "at the same LICK ANGLE (vs the pre balance-angle fit)"}


def licks() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), lbr._session_bal)
    if fail:
        raise RuntimeError(f"balance failed for {len(fail)} sessions: {fail[:5]}")
    B = pd.concat([v for _it, v in res], ignore_index=True)
    P = P.merge(B, on=["set", "label", "k"], how="left", validate="one_to_one")
    P["bal_orofacial"] = P[lbr.BAL].mean(1)
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    P["ord_c"] = np.minimum(P.order, CAP)
    out = []
    for (_st, _a), g in P.groupby(["set", "animal"]):
        g = g.copy()
        pre = g[g.epoch == "pre"]
        for a in PAIRS:
            b = f"bal_{a}"
            g[f"spout|{a}"] = g[b] - g.pos.map(pre.groupby("pos")[b].mean())
            ok = pre[[b, "angle"]].notna().all(1)
            if ok.sum() > 20:
                k1, k0 = np.polyfit(pre.angle[ok], pre[b][ok], 1)
                g[f"angle|{a}"] = g[b] - (k0 + k1 * g.angle)
            else:
                g[f"angle|{a}"] = np.nan
        out.append(g)
    return pd.concat(out, ignore_index=True)


def compute() -> dict:
    P = licks()
    cols = ["set", "label", "animal", "epoch", "pos", "order", "ord_c", "angle", *[f"{r}|{a}" for r in REFS for a in PAIRS]]
    P[cols].to_csv(il.OUT_W / "balance_order_licks.csv.gz", index=False)
    curves, sig, recs = {}, {}, []
    for st in SETS:
        g = P[P.set == st]
        for r in REFS:
            for a in PAIRS:
                v = f"{r}|{a}"
                # an animal x epoch x order cell needs >= MIN_CELL licks (the acute ITI orders 5-8 were one PS95 session
                # of 5 licks at +24), else it is dropped from the curve AND the test
                gc = g.groupby(["epoch", "animal", "ord_c"])[v]
                keep = gc.transform("count") >= MIN_CELL
                g_ = g[keep]
                A = g_.groupby(["epoch", "animal", "ord_c"])[v].mean().reset_index()
                curves[(st, r, a)] = {ep: (A[A.epoch == ep].groupby("ord_c")[v].mean(),
                                           A[A.epoch == ep].groupby("ord_c")[v].sem()) for ep in il.EP4}
                L = (g_.dropna(subset=[v]).groupby(["label", "animal", "epoch", "ord_c"])[v].mean().reset_index()
                     .rename(columns={"ord_c": "lag"}))
                S_ = il.sig_compute(L, value=v, tag={"set": st, "ref": r, "pair": a})
                sig[(st, r, a)] = S_
                recs += S_["recs"]
    R = pd.DataFrame(recs)
    R.to_csv(il.OUT_W / "balance_order_sig.csv", index=False)
    pd.set_option("display.width", 220)
    print("== significant orders (epoch vs pre)\n" + R[R.sig_bh].groupby(["set", "ref", "pair", "epoch"]).lag
          .apply(lambda s: list(s.astype(int))).to_string())
    return {"curves": curves, "sig": sig}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    rows = [(st, r) for st in SETS for r in REFS]
    fig, axs = plt.subplots(len(rows), len(PAIRS), figsize=(24, 4.2 * len(rows)), squeeze=False)
    for i, (st, r) in enumerate(rows):
        for j, a in enumerate(PAIRS):
            ax = axs[i, j]
            for ep in il.EP4:
                mu, se = F["curves"][(st, r, a)][ep]
                ax.plot(mu.index, mu, "o-", color=col[ep], ms=3, lw=1.3)
                ax.fill_between(mu.index, mu - se, mu + se, color=col[ep], alpha=0.13, lw=0)
            il.sig_draw(ax, F["sig"][(st, r, a)], col)
            il._headroom([ax])
            ax.axhline(0, color="0.6", lw=0.7, ls=":")
            ax.set_xticks(range(CAP + 1), [*map(str, range(CAP)), f"{CAP}+"], fontsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(a + (" (ALM + tjM1 + SSp-m + SSp-n)" if a == "orofacial" else ""), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{SETS[st]}\nbalance {REFS[r]}\n(X_right - X_left, x1e3 dF/F)", fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("lick order", fontsize=8.5)
    fig.legend(handles=[Line2D([], [], color=col[e], lw=2, marker="o", ms=4, label=e) for e in il.EP4],
               loc="lower center", ncol=4, fontsize=9.5, frameon=False)
    fig.suptitle("Hemispheric BALANCE (contralesional minus ipsilesional activity in the decoder's 0..0.2 s window; + = more "
                 "contralesional) over LICK ORDER, by epoch; mean +- SEM over animals.\nRows: cued / ITI x reference (vs the "
                 "animal's pre mean at the spout; at the same lick angle). Bars above the curves: orders where that epoch "
                 "differs from pre (paired nested bootstrap animals > sessions, BH-FDR across the 10 orders); cells < 20 licks per animal dropped. PROVISIONAL",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(il.OUT_W / "figures" / "lick_balance_order.png", dpi=95)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_balance_order"))
        return 0
    F = compute()
    save_fd("lick_balance_order", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
