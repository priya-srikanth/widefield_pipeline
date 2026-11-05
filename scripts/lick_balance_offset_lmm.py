"""Does the session's hemispheric BALANCE track its read-out OFFSET? (Priya 10-28: check 3 of "is hemispheric balance
interesting or just a bad decoder": "sure let's try 3, then 1, then 2").

Per post-stroke session x set (ITI, cued), two session means over the same licks (the frozen KEY-row decoder's lick set,
`iti_lick_order.preds_path()`):
  offset   the like-for-like read-out offset (decoded minus pre licks that landed at the same place; mm, + =
           contralesional; `iti_lick_order.per_lick` -> lfl) -- already relative to pre
  balance  the decoder-window balance X_right - X_left at the same lick ANGLE and ORDER (`lick_balance_order`, reference
           'angord': 0 for pre by construction), per pair and the orofacial pool
Models (statsmodels MixedLM, random intercept per animal, ML; sessions as observations -- anti-conservative at 4
animals, so the per-animal correlations are printed first, rule 8):
  offset ~ balance + log(day)                      the balance slope with recovery time held
  offset ~ balance_within + balance_animal + log(day)   balance split into the animal's mean and the session's
                                                   deviation from it: balance_within = the WITHIN-animal slope (is a
                                                   session with a bigger balance shift a session with a bigger offset?)
Sign expectation: the frozen decoder's pre slope of read-out on balance is NEGATIVE (more contralesional-hemisphere
activity -> more ipsilesional read-out), so if the balance drives the offset the slope should be negative.
Outputs <OUT_W>/baloff_sessions.csv, baloff_models.csv; figure lick_balance_offset.png (figdata + --plot-only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_order as lbo  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

PAIRS = lbo.PAIRS
SETMAP = {"trial": "cued", "iti": "ITI"}


def sessions() -> pd.DataFrame:
    P = lo.per_lick(pd.read_csv(lo.preds_path()))
    off = P.groupby(["set", "label", "animal", "epoch"]).lfl.mean().rename("offset").reset_index()
    B = pd.read_csv(il.OUT_W / "balance_order_licks.csv.gz", usecols=["set", "label", *[f"angord|{a}" for a in PAIRS]])
    bal = B.groupby(["set", "label"]).mean().reset_index().rename(columns={f"angord|{a}": f"bal_{a}" for a in PAIRS})
    S = off.merge(bal, on=["set", "label"], how="inner")
    day = pd.read_csv(il.OUT_W / "aim_deficit_sessions.csv", usecols=["label", "day"]).drop_duplicates("label")
    S = S.merge(day, on="label", how="left")
    S["set"] = S.set.map(SETMAP)
    return S


def compute() -> dict:
    import statsmodels.formula.api as smf
    S = sessions()
    S.to_csv(il.OUT_W / "baloff_sessions.csv", index=False)
    P = S[(S.epoch != "pre") & (S.day > 0)].copy()
    P["logday"] = np.log(P.day)
    pd.set_option("display.width", 220)
    rows, an = [], []
    for st, G in P.groupby("set"):
        for a in PAIRS:
            b = f"bal_{a}"
            g = G.dropna(subset=["offset", b, "logday"]).copy()
            for animal, h in g.groupby("animal"):                      # rule 8: per animal first
                r = np.corrcoef(h[b], h.offset)[0, 1] if len(h) > 3 else np.nan
                an.append({"set": st, "pair": a, "animal": animal, "n": len(h), "r": r,
                           "slope": np.polyfit(h[b], h.offset, 1)[0] if len(h) > 3 else np.nan})
            g["bal"] = g[b]
            g["bal_animal"] = g.groupby("animal").bal.transform("mean")
            g["bal_within"] = g.bal - g.bal_animal
            for name, f in (("balance + log(day)", "offset ~ bal + logday"),
                            ("within + between + log(day)", "offset ~ bal_within + bal_animal + logday")):
                try:
                    m = smf.mixedlm(f, g, groups=g["animal"]).fit(reml=False)
                    term = "bal" if "bal +" in f or f.endswith("bal") else "bal_within"
                    rows.append({"set": st, "pair": a, "model": name, "n": len(g), "slope": m.fe_params[term],
                                 "lo": m.conf_int().loc[term, 0], "hi": m.conf_int().loc[term, 1], "p": m.pvalues[term],
                                 "logday": m.fe_params["logday"], "p_logday": m.pvalues["logday"]})
                except Exception as ex:
                    rows.append({"set": st, "pair": a, "model": f"{name} FAILED {ex}", "n": len(g)})
    A, M = pd.DataFrame(an), pd.DataFrame(rows)
    print("== per animal (rule 8): within-animal correlation of session offset with session balance (post sessions)")
    print(A.pivot_table(index=["set", "pair"], columns="animal", values="r").round(2).to_string())
    print("== mixed models (slope of offset on balance, mm per balance unit)\n" + M.round(4).to_string(index=False))
    M.to_csv(il.OUT_W / "baloff_models.csv", index=False)
    return {"S": S, "M": M, "A": A}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    S, M, A = F["S"], F["M"], F["A"]
    fig, axs = plt.subplots(2, len(PAIRS), figsize=(23, 9), squeeze=False)
    for i, st in enumerate(("ITI", "cued")):
        for j, a in enumerate(PAIRS):
            ax = axs[i, j]
            g = S[(S.set == st) & (S.epoch != "pre")]
            for (ep, an_), h in g.groupby(["epoch", "animal"]):
                ax.plot(h[f"bal_{a}"], h.offset, ANIMAL_MARKER.get(an_, "o"), color=EPOCH_LINE[ep], ms=5, alpha=0.8,
                        mec="k", mew=0.3)
            m = M[(M.set == st) & (M.pair == a) & (M.model == "within + between + log(day)")]
            r = A[(A.set == st) & (A.pair == a)].set_index("animal").r
            ttl = f"{a}, {st}"
            if len(m) and "slope" in m and np.isfinite(m.slope.iloc[0]):
                pv = m.p.iloc[0]
                s_ = f"within-animal slope {m.slope.iloc[0]:+.3f} (p {pv:.2g})"
                ttl += "\n" + (rf"$\mathbf{{{s_.replace(' ', '~')}}}$" if pv < 0.05 else s_)
            ttl += "\nper-animal r: " + " / ".join(f"{r.get(x, np.nan):+.2f}" for x in ("PS92", "PS93", "PS94", "PS95"))
            ax.set_title(ttl, fontsize=8.5)
            ax.axhline(0, color="0.7", lw=0.6)
            ax.axvline(0, color="0.7", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 1:
                ax.set_xlabel("session balance at the same angle & order\n(X_right - X_left vs pre, x1e3 dF/F)", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{st}: session like-for-like offset\n(mm, + = contralesional)", fontsize=8.5)
    h = [Line2D([], [], color=EPOCH_LINE[e], marker="o", lw=0, label=e) for e in ("acute", "subacute", "chronic")]
    h += [Line2D([], [], color="0.3", marker=ANIMAL_MARKER.get(x, "o"), lw=0, label=x) for x in ("PS92", "PS93", "PS94", "PS95")]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=9, frameon=False)
    fig.suptitle("Does a session's hemispheric BALANCE track its read-out OFFSET? One point per post-stroke session; mixed model "
                 "offset ~ balance_within + balance_animal + log(day) + (1 | animal) (sessions as observations, "
                 "anti-conservative);\nwithin-animal slope bold when p < 0.05. Expected sign if the balance drives the "
                 "offset: NEGATIVE (more contralesional activity -> more ipsilesional read-out). PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(il.OUT_W / "figures" / "lick_balance_offset.png", dpi=100)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_balance_offset"))
        return 0
    F = compute()
    save_fd("lick_balance_offset", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
