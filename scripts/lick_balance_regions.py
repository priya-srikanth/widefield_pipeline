"""Is the balance -> read-out-offset link SPECIFIC to orofacial cortex? (Priya 10-28: "should we see if balance in other
brain regions is equally predictive?", "yes run the head-to-head balance model").

Per lick (the frozen KEY-row decoder's lick set): balance = X_right - X_left in the decoder's 0..0.2 s window, per region
GROUP (mean over the group's matched area pairs; only areas with both a left and a right ROI -- VISp / VISal / VISl have
no left ROI and are left out):
  orofacial   ALM, tjM1, SSp-m, SSp-n            (the pool of 12.18 / 14.14)
  visual      VISam, VISpm, VISa, VISrl
  RSP         RSPd, RSPagl, RSPv
  barrel      SSp-bfd
  limb/trunk  SSp-ul, SSp-ll, SSp-tr
  ACAd        ACAd
  non-orofacial  mean of the five control groups
Reference as 14.14 'angord': per animal and lick order, the pre-stroke line of balance on lick angle; the residual is the
balance at the same angle and order (0 for pre). Session means (post sessions) against the session's like-for-like offset.
  (1) per region: offset ~ bal_within + bal_animal + log(day) + (1 | animal)    (as 12.18)
  (2) HEAD-TO-HEAD: offset ~ oro_within + nonoro_within + oro_animal + nonoro_animal + log(day) + (1 | animal)
  (3) does the region track lick direction pre-stroke at all: per animal, the pre slope of balance on angle (per deg) and
      its share of variance (R2) on pre licks
Per-animal correlations first (rule 8); MixedLM with sessions as observations is anti-conservative at 4 animals.
Outputs <OUT_W>/balreg_sessions.csv, balreg_models.csv, balreg_pre_slopes.csv; figure lick_balance_regions.png.
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

GROUPS = {"orofacial": ["ALM", "tjM1", "SSp-m", "SSp-n"], "visual": ["VISam", "VISpm", "VISa", "VISrl"],
          "RSP": ["RSPd", "RSPagl", "RSPv"], "barrel": ["SSp-bfd"], "limb/trunk": ["SSp-ul", "SSp-ll", "SSp-tr"],
          "ACAd": ["ACAd"]}
CONTROLS = ["visual", "RSP", "barrel", "limb/trunk", "ACAd"]
ALLG = [*GROUPS, "non-orofacial"]
MIN_CELL = 20
CAP = lo.ORDER_CAP


def _session(label: str) -> pd.DataFrame:
    acts = lbr.cv._acts(label)
    d = il._load(label)
    cols = list(d["cols"])
    out = []
    for st, (xk, sk) in {"iti": ("iti_X", "iti_side"), "trial": ("tr_X", "tr_side")}.items():
        act = np.asarray(acts.get(st, []), float)
        if not len(act):
            continue
        m = il.subseq_match(d[sk].astype(float), act)
        X = d[xk].astype(float)[:, lbr.W, :].mean(1)
        R = pd.DataFrame({"set": st, "label": label, "k": np.arange(len(act))})
        for g, areas in GROUPS.items():
            ok = [a for a in areas if f"{a}_right" in cols and f"{a}_left" in cols]
            b = np.mean([X[:, cols.index(f"{a}_right")] - X[:, cols.index(f"{a}_left")] for a in ok], 0)
            R[f"bal_{g}"] = np.where(m >= 0, b[np.maximum(m, 0)], np.nan)
        out.append(R)
    return pd.concat(out, ignore_index=True)


def licks() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), _session)
    if fail:
        raise RuntimeError(f"balance failed for {len(fail)} sessions: {fail[:5]}")
    P = P.merge(pd.concat([v for _i, v in res], ignore_index=True), on=["set", "label", "k"], how="left",
                validate="one_to_one")
    P["bal_non-orofacial"] = P[[f"bal_{g}" for g in CONTROLS]].mean(1)
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    P["ord_c"] = np.minimum(P.order, CAP)
    P = lo.per_lick(P)                                   # adds lfl (like-for-like offset)
    slopes, out = [], []
    for (st, an), g in P.groupby(["set", "animal"]):
        g = g.copy()
        pre = g[g.epoch == "pre"]
        for grp in ALLG:
            b = f"bal_{grp}"
            ok = pre[[b, "angle"]].notna().all(1)
            if ok.sum() > 50:
                s1, s0 = np.polyfit(pre.angle[ok], pre[b][ok], 1)
                r2 = np.corrcoef(pre.angle[ok], pre[b][ok])[0, 1] ** 2
                slopes.append({"set": st, "animal": an, "group": grp, "pre_slope": s1, "pre_r2": r2})
            g[f"res_{grp}"] = np.nan
            for k_, pk in pre[ok].groupby("ord_c"):
                if len(pk) >= MIN_CELL and pk.angle.std() > 0:
                    a1, a0 = np.polyfit(pk.angle, pk[b], 1)
                    mk = g.ord_c == k_
                    g.loc[mk, f"res_{grp}"] = g.loc[mk, b] - (a0 + a1 * g.loc[mk, "angle"])
        out.append(g)
    return pd.concat(out, ignore_index=True), pd.DataFrame(slopes)


def compute() -> dict:
    import statsmodels.formula.api as smf
    P, SL = licks()
    S = P.groupby(["set", "label", "animal", "epoch"]).agg(offset=("lfl", "mean"),
                                                           **{f"r_{g}": (f"res_{g}", "mean") for g in ALLG}).reset_index()
    day = pd.read_csv(il.OUT_W / "aim_deficit_sessions.csv", usecols=["label", "day"]).drop_duplicates("label")
    S = S.merge(day, on="label", how="left")
    S["set"] = S.set.map({"trial": "cued", "iti": "ITI"})
    S.to_csv(il.OUT_W / "balreg_sessions.csv", index=False)
    SL["set"] = SL.set.map({"trial": "cued", "iti": "ITI"})
    SL.to_csv(il.OUT_W / "balreg_pre_slopes.csv", index=False)
    pd.set_option("display.width", 230)
    print("== (3) pre-stroke balance vs lick angle, per animal: slope per deg / R2\n"
          + SL.pivot_table(index=["set", "group"], columns="animal", values=["pre_slope", "pre_r2"]).round(3).to_string())
    Q = S[(S.epoch != "pre") & (S.day > 0)].copy()
    Q["logday"] = np.log(Q.day)
    rows, an = [], []
    for st, G in Q.groupby("set"):
        for grp in ALLG:
            v = f"r_{grp}"
            g = G.dropna(subset=["offset", v, "logday"]).copy()
            for a_, h in g.groupby("animal"):
                an.append({"set": st, "group": grp, "animal": a_,
                           "r": np.corrcoef(h[v], h.offset)[0, 1] if len(h) > 3 else np.nan})
            g["w"] = g[v] - g.groupby("animal")[v].transform("mean")
            g["b"] = g.groupby("animal")[v].transform("mean")
            try:
                m = smf.mixedlm("offset ~ w + b + logday", g, groups=g["animal"]).fit(reml=False)
                rows.append({"set": st, "model": "single", "term": grp, "slope": m.fe_params["w"],
                             "lo": m.conf_int().loc["w", 0], "hi": m.conf_int().loc["w", 1], "p": m.pvalues["w"],
                             "llf": m.llf, "n": len(g)})
            except Exception as ex:
                rows.append({"set": st, "model": "single", "term": f"{grp} FAILED {ex}"})
        # (2) head-to-head
        g = G.dropna(subset=["offset", "r_orofacial", "r_non-orofacial", "logday"]).copy()
        for v, nm in (("r_orofacial", "o"), ("r_non-orofacial", "c")):
            g[f"{nm}w"] = g[v] - g.groupby("animal")[v].transform("mean")
            g[f"{nm}b"] = g.groupby("animal")[v].transform("mean")
        try:
            m = smf.mixedlm("offset ~ ow + cw + ob + cb + logday", g, groups=g["animal"]).fit(reml=False)
            ci = m.conf_int()
            for t, lab in (("ow", "orofacial (with non-orofacial in)"), ("cw", "non-orofacial (with orofacial in)")):
                rows.append({"set": st, "model": "head-to-head", "term": lab, "slope": m.fe_params[t], "lo": ci.loc[t, 0],
                             "hi": ci.loc[t, 1], "p": m.pvalues[t], "llf": m.llf, "n": len(g)})
            r_oc = np.corrcoef(g.ow, g.cw)[0, 1]
            rows.append({"set": st, "model": "head-to-head", "term": "corr(oro, non-oro) within animal", "slope": r_oc,
                         "n": len(g)})
        except Exception as ex:
            rows.append({"set": st, "model": "head-to-head", "term": f"FAILED {ex}"})
    M, A = pd.DataFrame(rows), pd.DataFrame(an)
    print("== per animal (rule 8): within-animal r of session offset with session balance (same angle & order)\n"
          + A.pivot_table(index=["set", "group"], columns="animal", values="r").round(2).to_string())
    print("== models\n" + M.round(4).to_string(index=False))
    M.to_csv(il.OUT_W / "balreg_models.csv", index=False)
    return {"S": S, "M": M, "A": A, "SL": SL}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M, A, SL = F["M"], F["A"], F["SL"]
    an_mk = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    fig, axs = plt.subplots(2, 3, figsize=(20, 9.5), squeeze=False)
    for i, st in enumerate(("cued", "ITI")):
        # (a) per-region within-animal r (animals) + LMM slope significance
        ax = axs[i, 0]
        for j, g in enumerate(ALLG):
            a = A[(A.set == st) & (A.group == g)]
            for _k, r in a.iterrows():
                ax.plot(j + {"PS92": -0.15, "PS93": -0.05, "PS94": 0.05, "PS95": 0.15}[r.animal], r.r,
                        an_mk[r.animal], color="#1f77b4" if g == "orofacial" else "0.45", ms=6, mfc="none")
            m = M[(M.set == st) & (M.model == "single") & (M.term == g)]
            if len(m) and m.p.iloc[0] < 0.05:
                ax.text(j, 1.02, "*", ha="center", fontsize=14, fontweight="bold")
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xticks(range(len(ALLG)), [g.replace("/", "/\n") for g in ALLG], fontsize=8)
        ax.set_ylim(-1.05, 1.15)
        ax.set_ylabel(f"{st}: within-animal r\n(session balance vs read-out offset)", fontsize=9)
        ax.set_title("each region alone (symbols = animals; * = within-animal LMM slope p < 0.05)", fontsize=9)
        # (b) head-to-head
        ax = axs[i, 1]
        h = M[(M.set == st) & (M.model == "head-to-head") & M.term.str.contains("in\\)")]
        for j, (_k, r) in enumerate(h.iterrows()):
            ax.errorbar(j, r.slope, yerr=[[r.slope - r.lo], [r.hi - r.slope]], fmt="o", color="k", capsize=3)
            s_ = f"p = {r.p:.2g}"
            ax.text(j, r.hi, (rf"$\mathbf{{{s_.replace(' ', '~')}}}$" if r.p < 0.05 else s_), ha="center", va="bottom",
                    fontsize=9)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xticks(range(len(h)), [t.replace(" (", "\n(") for t in h.term], fontsize=8.5)
        ax.set_xlim(-0.6, len(h) - 0.4)
        rc = M[(M.set == st) & (M.term == "corr(oro, non-oro) within animal")]
        ax.set_title("HEAD-TO-HEAD: both balances in one model (within-animal slopes)\n"
                     + (f"within-animal corr(orofacial, non-orofacial balance) = {rc.slope.iloc[0]:.2f}" if len(rc) else ""),
                     fontsize=9)
        ax.set_ylabel("offset slope (mm per balance unit)", fontsize=9)
        # (c) pre-stroke direction tracking
        ax = axs[i, 2]
        sl = SL[SL.set == st]
        for j, g in enumerate(ALLG):
            s = sl[sl.group == g]
            for _k, r in s.iterrows():
                ax.plot(j + {"PS92": -0.15, "PS93": -0.05, "PS94": 0.05, "PS95": 0.15}[r.animal], r.pre_r2,
                        an_mk[r.animal], color="#1f77b4" if g == "orofacial" else "0.45", ms=6, mfc="none")
        ax.set_xticks(range(len(ALLG)), [g.replace("/", "/\n") for g in ALLG], fontsize=8)
        ax.set_ylabel("pre-stroke R2 of balance on lick angle", fontsize=9)
        ax.set_title("does the region's balance track lick direction before the stroke?", fontsize=9)
        for a_ in axs[i]:
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)
    fig.suptitle("Is the balance -> read-out-offset link specific to OROFACIAL cortex? Balance = X_right - X_left (decoder's "
                 "0..0.2 s window) at the same lick angle and order, session means, post-stroke sessions;\nmixed models "
                 "offset ~ balance (within + between animal) + log(day) + (1 | animal), sessions as observations "
                 "(anti-conservative). Symbols: PS92 o, PS93 s, PS94 ^, PS95 D. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(il.OUT_W / "figures" / "lick_balance_regions.png", dpi=100)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_balance_regions"))
        return 0
    F = compute()
    save_fd("lick_balance_regions", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
