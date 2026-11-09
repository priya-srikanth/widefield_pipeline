"""Balance over LICK ORDER with the PRIMARY balance (Priya 10-29: replace 'balance over lick number' with the updated
balance). Per session x lick order (0..8, 9+ pooled), from lick_balance_subsets (balsub_order.csv): balance =
log2(sum observed / sum predicted, contralesional) - log2(same, ipsilesional), pooled over ALM, tjM1, SSp-m, SSp-n, SSs,
response 0.1..0.5 s after the lick minus -1..-0.5 s, prediction = the animal's PRE line on lick angle FOR THAT LICK ORDER
(pre sessions leave-one-session-out, so pre ~ 0 with its noise). Cued = every post-cue lick in the decoder's lick set (all
orders; the 1500-lick cap applies, random); ITI = every ITI lick. Cells with < MIN_N licks dropped.
Per animal x epoch mean over sessions, then mean of animals; per order, epoch vs pre: nested bootstrap animals >
sessions (analysis_kit.boot_delta), BH-FDR over orders. Outputs <OUT_W>/balorder_primary.csv; figure
lick_balance_order_primary.png. The figure shows the balance under lick_recovery_primary.BAL_DISPLAY (default: implied
direction = -balance, Priya 10-29; the CSV keeps the hemispheric sign). --plot-only redraws from balorder_primary.csv
(no bootstrap, CSV untouched).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import plot_only  # noqa: E402
from lick_recovery_primary import BAL_DISPLAY, BAL_LABEL, bal_disp  # noqa: E402

MIN_N = 10
EPS = ["pre", "acute", "subacute", "chronic"]
SETS = ["cued", "ITI"]


def trial_set_sentence() -> str:
    """The lick sets of lick_balance_order_primary (Priya 11-09): every KEY cued / ITI lick by lick order (balsub_order.csv
    of lick_balance_subsets' default run), session x order cells with >= MIN_N licks."""
    import lick_balance_subsets as lbs
    cell = f"session x lick-order cells (orders 0-8, 9+ pooled) with >= {MIN_N} licks"
    return lbs.set_sentence(("trial", "iti"), full=False, inside=False, extra=[cell])


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    O = pd.read_csv(il.OUT_W / "balsub_order.csv")
    O = O[O.set.isin(SETS) & (O.n >= MIN_N)].copy()
    ok = (O.obs_c > 0) & (O.pred_c > 0) & (O.obs_i > 0) & (O.pred_i > 0)
    O["bal"] = np.where(ok, np.log2(O.obs_c.clip(lower=1e-12) / O.pred_c.clip(lower=1e-12))
                        - np.log2(O.obs_i.clip(lower=1e-12) / O.pred_i.clip(lower=1e-12)), np.nan)
    O = O.dropna(subset=["bal"])
    if plot_only():                                      # redraw from the stored table: no bootstrap, CSV untouched
        R = pd.read_csv(il.OUT_W / "balorder_primary.csv")
    else:
        R = _stats(O)
    _plot(O, R)
    return 0


def _stats(O: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(0)
    rows = []
    for (st, o), g in O.groupby(["set", "ord_c"]):
        ref = {a: h[h.epoch == "pre"].bal.to_numpy() for a, h in g.groupby("animal")}
        for ep in EPS[1:]:
            post = {a: h[h.epoch == ep].bal.to_numpy() for a, h in g.groupby("animal")}
            com = [a for a in sorted(post) if len(post[a]) and len(ref.get(a, []))]
            if len(com) < 3:
                continue
            iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, rng)
            pa = {a: float(post[a].mean() - ref[a].mean()) for a in com}
            rows.append({"set": st, "ord_c": o, "epoch": ep, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                         "p": float(getattr(iv, "p", np.nan)), "n_animals": len(com),
                         "n_same_sign": int(sum(np.sign(v) == np.sign(iv.point) for v in pa.values())),
                         **{f"d_{a}": v for a, v in pa.items()}})
    R = pd.DataFrame(rows)
    # BH over orders within set x epoch, on the bootstrap p if available, else CI-excludes-0
    R["sig"] = (R.lo > 0) | (R.hi < 0)
    if R.p.notna().any():
        for _k, g in R.groupby(["set", "epoch"]):
            p = g.p.to_numpy()
            o_ = np.argsort(p)
            q = p[o_] * len(p) / (np.arange(len(p)) + 1)
            q = np.minimum.accumulate(q[::-1])[::-1]
            qq = np.empty_like(q)
            qq[o_] = q
            R.loc[g.index, "q_bh"] = qq
        R["sig"] = R.q_bh < 0.05
    R.to_csv(il.OUT_W / "balorder_primary.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal: epoch minus pre balance by lick order (rule 8)\n" + R.set_index(["set", "epoch", "ord_c"])[
        [c for c in R.columns if c.startswith("d_")] + ["delta", "lo", "hi", "sig"]].round(3).to_string())
    return R


def _plot(O: pd.DataFrame, R: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    O = O.assign(bal=bal_disp(O.bal))                    # DISPLAY sign only (lick_recovery_primary.BAL_DISPLAY)
    R = R.assign(sig=R.sig.astype(str).str.lower().isin(["true", "1"]))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for j, st in enumerate(SETS):
        ax = axs[j]
        g = O[O.set == st]
        A = g.groupby(["animal", "epoch", "ord_c"]).bal.mean().reset_index()
        nan_ = A.groupby(["epoch", "ord_c"]).animal.transform("nunique")
        A = A[nan_ >= 3]                                 # orders where >= 3 animals contribute (display)
        for ep in EPS:
            a_ = A[A.epoch == ep]
            for _an, h in a_.groupby("animal"):
                ax.plot(h.ord_c, h.bal, "-", color="0.6" if ep == "pre" else EPOCH_LINE[ep], lw=0.5, alpha=0.35)
            m = a_.groupby("ord_c").bal.mean()
            ax.plot(m.index, m.values, "o-", color="0.35" if ep == "pre" else EPOCH_LINE[ep], lw=2, ms=4, label=ep)
            if ep != "pre":
                s_ = R[(R.set == st) & (R.epoch == ep) & R.sig]
                for _, r in s_.iterrows():
                    ax.plot(r.ord_c, m.get(r.ord_c, np.nan), "o", mfc="none", mec="k", ms=10, mew=1.2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(10), [str(k) for k in range(9)] + ["9+"])
        ax.set_ylim(*sorted(bal_disp(np.array([-1.2, 0.8]))))
        ax.set_xlabel("lick order (0 = first post-cue lick / ITI bout start)" if st == "cued" else "lick order within the ITI bout")
        what = "balance" if BAL_DISPLAY == "hemisphere" else "implied direction"
        ax.set_title(f"{st} licks: primary {what} by lick order (mean of animals; thin = animals;\nringed = epoch differs "
                     "from pre, nested bootstrap, BH over orders)", fontsize=9.5)
        ax.legend(fontsize=8, frameon=False)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].set_ylabel(BAL_LABEL + "\n(0 = as pre at the same angle & order)", fontsize=8.5)
    shown = "\nShown as " + BAL_LABEL.replace("\n", " ") if BAL_DISPLAY == "direction" else ""
    fig.suptitle("Hemispheric BALANCE over lick order, PRIMARY definition (gain ratio, ALM + tjM1 + SSp-m + SSp-n + SSs, "
                 "0.1-0.5 s after each lick, referenced to pre licks at the same angle AND order). PROVISIONAL" + shown,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_png_svg(fig, il.OUT_W / "figures" / "lick_balance_order_primary.png", dpi=95)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
