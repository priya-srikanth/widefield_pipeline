"""Does the CONTRA / IPSI hemispheric BALANCE "explain" the like-for-like read-out offset? (Priya 10-28: "then let's
regress whether contra/ipsi balance explains lick offset (regress on it)").

Same partial-linear reference as 12.16 (`lick_covariate_reference.references`: Gaussian kernel on landing + linear
covariate terms fitted on PRE licks only, Robinson double residual, pre sessions leave-one-out; with no covariates it is
exactly the 12.1 like-for-like reference), with the covariates = the lick's hemispheric BALANCE in the decoder's own
feature window: per pair (ALM, tjM1, SSp-m, SSp-n) the mean over lags 0..0.2 s from the lick peak of
(X_right - X_left) of the session file's per-lick traces (`iti_X` / `tr_X`, the decoder's inputs; x1e3 dF/F).
References: landing only; + all four balances; + each pair alone; + the mean balance over the four pairs (ONE number:
a uniform hemispheric shift).
READ WITH CARE -- partly circular: the frozen decoder is a linear read-out of these same area features, and its lateral
weights are largely an ipsi-minus-contra difference (11.13), so balance terms measured in the same frames reconstruct
part of the read-out by construction. What is informative is (a) how much of the offset ONE uniform balance number
carries (a global hemispheric shift) versus the four pair-wise balances, and (b) which pair carries it. A large share
explained says the offset IS a change of the hemispheric balance at a fixed landing place; it does not say why.
Outputs <OUT_W>/balref_{betas,sessions,tests,explained}.csv; figure lick_balance_reference.png (figdata + --plot-only).
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
import lick_covariate_reference as cv  # noqa: E402
import likeforlike as lfl  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

PAIRS = ["ALM", "tjM1", "SSp-m", "SSp-n"]
BAL = [f"bal_{a}" for a in PAIRS]
W = (il.LAGS_D >= 0.0) & (il.LAGS_D <= 0.2)
REFS = {"landing": [], "bal_all4": BAL, **{f"bal_{a}": [f"bal_{a}"] for a in PAIRS}, "bal_mean": ["bal_mean"]}
RLAB = {"landing": "landing only", "bal_all4": "+ 4 pair balances", "bal_mean": "+ mean balance (1 number)",
        **{f"bal_{a}": f"+ {a} balance" for a in PAIRS}}


def _session_bal(label: str) -> pd.DataFrame:
    acts = cv._acts(label)
    d = il._load(label)
    cols = list(d["cols"])
    out = []
    for st, (xk, sk) in {"iti": ("iti_X", "iti_side"), "trial": ("tr_X", "tr_side")}.items():
        act = np.asarray(acts.get(st, []), float)
        if not len(act):
            continue
        m = il.subseq_match(d[sk].astype(float), act)
        X = d[xk].astype(float)[:, W, :].mean(1)
        R = pd.DataFrame({"set": st, "label": label, "k": np.arange(len(act))})
        for a in PAIRS:
            b = X[:, cols.index(f"{a}_right")] - X[:, cols.index(f"{a}_left")]
            R[f"bal_{a}"] = np.where(m >= 0, b[np.maximum(m, 0)], np.nan)
        out.append(R)
    return pd.concat(out, ignore_index=True)


def compute() -> dict:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    res, fail = ak.fan_sessions(sorted(P.label.unique()), _session_bal)
    if fail:
        raise RuntimeError(f"balance failed for {len(fail)} sessions: {fail[:5]}")
    C = pd.concat([v for _it, v in res], ignore_index=True)
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    P["bal_mean"] = P[BAL].mean(1)
    print(f"balance attached to {P[BAL].notna().all(1).mean():.4f} of predictions")
    ok = np.isfinite(P[["act", "dec"]].to_numpy(float)).all(1)
    sg, _cvt = lfl.choose(P[ok], "act", "dec", lfl.SIGMA_MM, "nw")
    P["resid_landing_all"] = np.nan
    P.loc[ok, "resid_landing_all"] = P.dec[ok] - lfl.ref_table(P[ok], "act", "dec", sg, "nw").ref
    allc = [*BAL, "bal_mean"]
    P["common"] = np.isfinite(P[["act", "dec", *allc]].to_numpy(float)).all(1)
    # reuse 12.16's reference / session / test code with this covariate set (module globals swapped for the call)
    cv.REFS, cv.ALLCOV = REFS, allc
    cv.VALS = [f"resid_{r}" for r in REFS] + [f"expl_{r}" for r in REFS if r != "landing"]
    Q = P[P.common]
    R, B = cv.references(Q, sg)
    P = P.join(R)
    chk = lfl.ref_table(Q, "act", "dec", sg, "nw")
    d1 = np.nanmax(np.abs((Q.dec - chk.ref) - P.loc[Q.index, "resid_landing"]))
    print(f"landing-only reference vs likeforlike nw on the common set: max |diff| = {d1:.2e} (sigma {sg} mm)")
    S = cv.sessions(P)
    T = cv.tests(S)
    X = cv.explained(T)
    for f, df in (("betas", B), ("sessions", S), ("tests", T), ("explained", X)):
        df.to_csv(il.OUT_W / f"balref_{f}.csv", index=False)
    pd.set_option("display.width", 220)
    x = X[X.scope == "all spouts|all"]
    print("== share of the like-for-like offset change explained (1 - change(ref) / change(landing)), all licks")
    print(x.pivot_table(index=["set", "ref"], columns="epoch", values="share").reindex(columns=cv.EP).round(2).to_string())
    print("== per animal share, bal_all4 and bal_mean")
    print(x[x.ref.isin(["bal_all4", "bal_mean"])][["set", "ref", "epoch", "change_landing", "change_ref", "share",
                                                   *[c for c in x if c.startswith("share_PS")]]].round(2).to_string(index=False))
    print("== pre-lick MSE: landing only vs + covariates (held-out); betas per SD")
    print(B.round(4).to_string(index=False))
    return dict(T=T, X=X, B=B, sigma=sg, d1=d1)


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T, X = F["T"], F["X"]
    refs = list(REFS)
    cols = {"landing": "0.25", "bal_all4": "#7b3294", "bal_mean": "#d95f0e", "bal_ALM": "#1b9e77", "bal_tjM1": "#2c7fb8",
            "bal_SSp-m": "#e7298a", "bal_SSp-n": "#66a61e"}
    fig, axs = plt.subplots(2, 2, figsize=(17, 10), squeeze=False)
    for i, st in enumerate(("iti", "trial")):
        t = T[(T.scope == "all spouts|all") & (T.set == st)]
        ax = axs[i, 0]
        for k, ref in enumerate(refs):
            q = t[t.measure == f"resid_{ref}"].set_index("epoch").reindex(cv.EP)
            x = np.arange(3) + (k - (len(refs) - 1) / 2) * 0.11
            ax.errorbar(x, q.point, yerr=[q.point - q.lo, q.hi - q.point], fmt="o", color=cols[ref], ms=5, capsize=2,
                        label=RLAB[ref])
        ax.axhline(0, color="0.6", lw=0.7, ls="--")
        ax.set_xticks(range(3), cv.EP)
        ax.set_ylabel(f"{cv.SETS[st]}: like-for-like offset,\nchange vs pre (mm, + = contralesional)", fontsize=9)
        ax.set_title("offset after matching landing (+ balance covariates)", fontsize=9.5)
        if i == 0:
            ax.legend(fontsize=7.5, frameon=False, ncol=2)
        ax = axs[i, 1]
        x_ = X[(X.scope == "all spouts|all") & (X.set == st) & (X.ref != "landing")]
        # a share is only meaningful where there is an offset to explain: blank epochs whose landing-only change CI
        # includes 0 (cued subacute / chronic, where the change is ~0 and the ratio runs to +-4)
        base = t[t.measure == "resid_landing"].set_index("epoch").reindex(cv.EP)
        live = ((base.lo > 0) | (base.hi < 0)).to_numpy()
        for k, ref in enumerate([r for r in refs if r != "landing"]):
            q = x_[x_.ref == ref].set_index("epoch").reindex(cv.EP)
            q = q.where(pd.Series(live, index=q.index), np.nan)
            xx = np.arange(3) + (k - 2.5) * 0.12
            ax.bar(xx, q.share, width=0.11, color=cols[ref], label=RLAB[ref])
            for a_, mk in zip(("PS92", "PS93", "PS94", "PS95"), "osD^"):
                if f"share_{a_}" in q:
                    ax.plot(xx, q[f"share_{a_}"].clip(-1, 2), mk, color="k", ms=3, mfc="none", lw=0)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.axhline(1, color="0.6", lw=0.7, ls=":")
        ax.set_xticks(range(3), cv.EP)
        ax.set_ylim(-1, 2)
        ax.set_ylabel("share of the offset change explained\n(1 = all; bars = animal-weighted, symbols = animals)",
                      fontsize=9)
        ax.set_title("how much of the offset the hemispheric balance accounts for\n(blank = no significant offset to "
                     "explain)", fontsize=9.5)
    fig.suptitle("Does the CONTRA / IPSI BALANCE explain the like-for-like read-out offset? Partial-linear reference "
                 "(as 12.16): landing kernel + balance terms fitted on PRE licks; balance = X_right - X_left in the decoder's "
                 "0..0.2 s window per pair.\nPARTLY CIRCULAR: the decoder is a linear read-out of the same features, so "
                 "balance terms reconstruct part of it by construction; informative = one uniform balance number vs the "
                 "four pairs, and which pair. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png_svg(fig, il.OUT_W / "figures" / "lick_balance_reference.png", dpi=100)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_balance_reference"))
        return 0
    F = compute()
    save_fd("lick_balance_reference", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
