"""Which AREAS should the balance pool? (Priya 10-28: "yes to held out windows and subset analysis").

Areas: ALM, tjM1, SSp-m, SSp-n (the orofacial pool), SSs and SSp-un -> all 63 non-empty subsets. Four FIXED definitions
from the sweep / nested check (lick_balance_variants, lick_balance_nested):
  ratio_020_035  angle-matched gain ratio, baseline-subtracted 0.2..0.35 s (confirmed for the first cued lick)
  ratio_010_020  same, 0.1..0.2 s
  raw_010_020    raw-level difference 0.1..0.2 s
  raw_075_100    raw-level difference 0.75..1.0 s
Every lick value is referenced to the PRE line on lick angle per animal x lick set x lick order (pre sessions
leave-one-pre-session-out; lick_recovery_markers._residuals) PER AREA AND SIDE; a subset's pool is the mean over its
areas, and because the reference is linear, the pooled residual = the mean of the area residuals, so subsets are built
from per-area session sums: difference = mean over areas of the session-mean referenced (contra - ipsi); ratio =
log2(sum_areas obs contra / sum_areas predicted contra) - log2(same, ipsi).
Lick sets / targets: cued first lick, cued (all) -> far-contra first-lick aim; ITI first after >= 2 s quiet, ITI (all)
-> ITI's own axis (iti_ang). Per subset x definition: leave-one-animal-out cvR2; NESTED: the subset (fixed definition),
or subset x definition, chosen by LOAO on 3 animals, tested on the 4th, against a null repeating the whole selection on
within-animal shuffled sessions (N_NULL).
Outputs <OUT_W>/balsub_sessions.csv, balsub_loao.csv, balsub_nested.csv; figure lick_balance_subsets.png.
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_nested as bn  # noqa: E402
import lick_balance_reference as lbr  # noqa: E402
import lick_balance_regions as lbg  # noqa: E402
import lick_recovery_markers as lrm  # noqa: E402

AREAS = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs", "SSp-un"]
DEFS = {"ratio_020_035": ("bs", 0.2, 0.35), "ratio_010_020": ("bs", 0.1, 0.2),
        # Priya 10-28: "for ITI we may want it shifted sooner? what's the harm in 0-0.5"
        "ratio_000_050": ("bs", 0.0, 0.5), "ratio_010_035": ("bs", 0.1, 0.35), "ratio_010_050": ("bs", 0.1, 0.5),
        "raw_010_020": ("raw", 0.1, 0.2), "raw_075_100": ("raw", 0.75, 1.0)}
SETS = {"trial_1": lrm.C1SET, "trial": "cued", "iti_q": lrm.QSET, "iti": "ITI"}
TARGET = {lrm.C1SET: "aim", "cued": "aim", lrm.QSET: "iti_ang", "ITI": "iti_ang"}
N_NULL = 200
# REFERENCE covariate for the pre line (env WFIELD_BAL_REF; Priya 10-28 "is angle better than lateral distance"):
# "angle" (default, asin(side / reach), deg) or "lateral" (side_k, mm: the landing's sideways position). Outputs of a
# non-default reference carry the suffix _<ref>.
REF = os.environ.get("WFIELD_BAL_REF", "angle")
SFX = "" if REF == "angle" else f"_{REF}"


def _wm(lo_, hi_):
    L = il.LAGS_D
    return (L >= lo_ - 1e-9) & ((L < hi_ - 1e-9) if hi_ < L.max() else (L <= hi_ + 1e-9))


def _session(label: str) -> pd.DataFrame:
    acts = lbr.cv._acts(label)
    d = il._load(label)
    cols = list(d["cols"])
    base_m = _wm(*il.BASE_RESP)
    out = []
    for st, (xk, sk) in {"iti": ("iti_X", "iti_side"), "trial": ("tr_X", "tr_side")}.items():
        act = np.asarray(acts.get(st, []), float)
        if not len(act):
            continue
        m = il.subseq_match(d[sk].astype(float), act)
        ok, mm = m >= 0, np.maximum(m, 0)
        X = d[xk].astype(float)
        R = {"set": st, "label": label, "k": np.arange(len(act))}
        for a in AREAS:
            for side, h in (("c", "right"), ("i", "left")):
                tr = X[:, :, cols.index(f"{a}_{h}")]
                base = tr[:, base_m].mean(1)
                for dn, (kind, lo_, hi_) in DEFS.items():
                    v = tr[:, _wm(lo_, hi_)].mean(1) - (base if kind == "bs" else 0)
                    R[f"{dn}|{a}|{side}"] = np.where(ok, v[mm], np.nan)
        out.append(pd.DataFrame(R))
    return pd.concat(out, ignore_index=True)


def sessions() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), _session)
    if fail:
        raise RuntimeError(f"failed: {fail[:5]}")
    P = P.merge(pd.concat([v for _i, v in res], ignore_index=True), on=["set", "label", "k"], how="left",
                validate="one_to_one")
    P["angle"] = (np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1))) if REF == "angle"
                  else P.side_k.astype(float))         # column name kept: lrm._residuals fits on 'angle'
    P["ord_c"] = np.minimum(P.order, lbg.CAP)
    vs = []
    for dn in DEFS:
        for a in AREAS:
            P[f"{dn}|{a}|d"] = P[f"{dn}|{a}|c"] - P[f"{dn}|{a}|i"]
            vs += [f"{dn}|{a}|{s}" for s in (("c", "i") if dn.startswith("ratio") else ("d",))]
    gres, gfail = ak.fan_sessions(sorted(P[P.set == "iti"].label.unique()), lrm._gaps)
    if gfail:
        raise RuntimeError(f"gaps failed: {gfail[:5]}")
    Q = P[P.set == "iti"].merge(pd.concat([v for _i, v in gres], ignore_index=True), on=["set", "label", "k"],
                                how="left", validate="one_to_one")
    Q = Q[Q.gap >= il.GAP_MAIN].assign(set="iti_q").drop(columns="gap")
    Q1 = P[(P.set == "trial") & (P.order == 0)].assign(set="trial_1")
    P = lrm._residuals(pd.concat([P, Q, Q1], ignore_index=True), vs)
    rows = []
    for (st, lab, an, ep), g in P.groupby(["set", "label", "animal", "epoch"]):
        rec = {"set": SETS[st], "label": lab, "animal": an, "epoch": ep}
        for dn in DEFS:
            for a in AREAS:
                if dn.startswith("ratio"):
                    for s in ("c", "i"):
                        obs, r_ = g[f"{dn}|{a}|{s}"], g[f"res_{dn}|{a}|{s}"]
                        okk = obs.notna() & r_.notna()
                        rec[f"{dn}|{a}|obs_{s}"] = obs[okk].sum()
                        rec[f"{dn}|{a}|pred_{s}"] = (obs[okk] - r_[okk]).sum()
                else:
                    rec[f"{dn}|{a}|diff"] = g[f"res_{dn}|{a}|d"].mean()
        rows.append(rec)
    S = pd.DataFrame(rows)
    R = pd.read_csv(il.OUT_W / "recmark_sessions.csv", usecols=["label", "day", "aim", "iti_ang"]).drop_duplicates("label")
    return S.merge(R, on="label", how="left")


def subset_values(S: pd.DataFrame) -> pd.DataFrame:
    """Session value of every subset x definition."""
    subs = [c for k in range(1, len(AREAS) + 1) for c in itertools.combinations(AREAS, k)]
    out = {}
    for dn in DEFS:
        for sub in subs:
            key = f"{dn}:{'+'.join(sub)}"
            if dn.startswith("ratio"):
                g = []
                for s in ("c", "i"):
                    o = S[[f"{dn}|{a}|obs_{s}" for a in sub]].sum(1)
                    p = S[[f"{dn}|{a}|pred_{s}" for a in sub]].sum(1)
                    g.append(np.where((o > 0) & (p > 0), np.log2(o.clip(lower=1e-12) / p.clip(lower=1e-12)), np.nan))
                out[key] = g[0] - g[1]
            else:
                out[key] = S[[f"{dn}|{a}|diff" for a in sub]].mean(1).to_numpy()
    return pd.concat([S[["set", "label", "animal", "epoch", "day", "aim", "iti_ang"]], pd.DataFrame(out, index=S.index)],
                     axis=1)


def main() -> int:
    S = sessions()
    S.to_csv(il.OUT_W / f"balsub_sessions{SFX}.csv", index=False)
    if "--sessions-only" in sys.argv:
        return 0
    V = subset_values(S)
    keys = [c for c in V.columns if ":" in c]
    rng = np.random.default_rng(0)
    L, N = [], []
    for st, yv in TARGET.items():
        G = V[(V.set == st) & (V.epoch != "pre") & (V.day > 0)].dropna(subset=[yv]).copy()
        y, grp = G[yv].to_numpy(), G.animal.to_numpy()
        r_days = bn._loao_r2(np.log(G.day.to_numpy()), y, grp)
        for k in keys:
            x = G[k].to_numpy(float)
            if np.isfinite(x).mean() < 0.9:
                continue
            x = np.where(np.isfinite(x), x, np.nanmean(x))
            dn, sub = k.split(":")
            per_an = {a: np.corrcoef(x[grp == a], y[grp == a])[0, 1] for a in np.unique(grp)}
            L.append({"set": st, "def": dn, "subset": sub, "n_areas": sub.count("+") + 1, "loao": bn._loao_r2(x, y, grp),
                      "days": r_days, **{f"r_{a}": v for a, v in per_an.items()}})
        pool = "ALM+tjM1+SSp-m+SSp-n+SSs"
        ratio_keys = [f"{dn}:{pool}" for dn in DEFS if dn.startswith("ratio")]
        for fam, kk in [(f"subsets | {dn}", [k for k in keys if k.startswith(dn + ":")]) for dn in DEFS] + \
                       [("subsets x definitions", keys), ("WINDOW only, pool oro+SSs (ratio)", ratio_keys)]:
            C = G[kk]
            C = C.loc[:, C.notna().mean() >= 0.9]
            X = C.to_numpy(float)
            X = np.where(np.isfinite(X), X, np.nanmean(X, 0))
            r, chosen = bn.nested(X, list(C.columns), y, grp)
            null = []
            for _ in range(N_NULL):
                Xs = X.copy()
                for a in np.unique(grp):
                    ix = np.flatnonzero(grp == a)
                    Xs[ix] = X[rng.permutation(ix)]
                null.append(bn.nested(Xs, list(C.columns), y, grp)[0])
            null = np.asarray(null)
            N.append({"set": st, "family": fam, "nested_loao": r, "days": r_days, "null95": float(np.quantile(null, 0.95)),
                      "p_null": float((1 + np.sum(null >= r)) / (1 + len(null))),
                      "chosen": " | ".join(f"{a}:{c}" for a, c in zip(np.unique(grp), chosen, strict=True))})
            print(f"[{st}] {fam}: nested {r:.3f} (days {r_days:.3f}, null95 {N[-1]['null95']:.3f}, p {N[-1]['p_null']:.3g})"
                  f"\n    {N[-1]['chosen']}", flush=True)
    L, N = pd.DataFrame(L), pd.DataFrame(N)
    L.to_csv(il.OUT_W / f"balsub_loao{SFX}.csv", index=False)
    N.to_csv(il.OUT_W / f"balsub_nested{SFX}.csv", index=False)
    pd.set_option("display.width", 230)
    print("== top subsets per set x definition (LOAO; per-animal r = rule 8)")
    print(L.sort_values("loao", ascending=False).groupby(["set", "def"]).head(4).round(3).to_string(index=False))
    # single areas and the orofacial pool, for reference
    ref = L[(L.n_areas == 1) | (L.subset == "ALM+tjM1+SSp-m+SSp-n")]
    print("== single areas + orofacial pool\n" + ref.pivot_table(index=["set", "subset"], columns="def", values="loao")
          .round(2).to_string())
    plot(L, N)
    return 0


def plot(L: pd.DataFrame, N: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = list(TARGET)
    fig, axs = plt.subplots(2, len(sets), figsize=(5.2 * len(sets), 10), squeeze=False)
    cl = dict(zip(DEFS, ["#4C72B0", "#64B5CD", "#C44E52", "#DD8452"], strict=True))
    for j, st in enumerate(sets):
        ax = axs[0, j]
        t = L[L.set == st]
        for dn in DEFS:
            u = t[t["def"] == dn].sort_values("loao", ascending=False).reset_index(drop=True)
            ax.plot(np.arange(len(u)), u.loao, "-", color=cl[dn], lw=1.2, label=dn)
            for name, mk in (("ALM+tjM1+SSp-m+SSp-n", "o"), ("ALM+tjM1+SSp-m+SSp-n+SSs+SSp-un", "s")):
                ix = u.index[u.subset == name]
                if len(ix):
                    ax.plot(ix, u.loao.iloc[ix], mk, mfc="none", mec=cl[dn], ms=8)
        ax.axhline(t.days.iloc[0], color="0.4", ls="--", lw=1, label="days since stroke")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("63 subsets, ranked within definition", fontsize=8)
        ax.set_title(f"{st} -> {TARGET[st]}: LOAO cvR² of every area subset\n(circle = orofacial 4, square = all 6)",
                     fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        best = t.sort_values("loao", ascending=False).head(5)
        ax.text(0.98, 0.98, "\n".join(f"{r['def']}: {r.subset} {r.loao:.2f}" for _, r in best.iterrows()),
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        ax = axs[1, j]
        n = N[N.set == st]
        x = np.arange(len(n))
        ax.bar(x, n.nested_loao, color=[cl.get(f.split(" | ")[-1], "#8172B3") for f in n.family])
        ax.plot(x, n.null95, "k_", ms=16, mew=1.5)
        ax.axhline(n.days.iloc[0], color="0.4", ls="--", lw=1)
        for xi, (_, r) in enumerate(n.iterrows()):
            ax.text(xi, max(r.nested_loao, 0) + 0.03, f"p {r.p_null:.2g}", ha="center", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1, 1)
        ax.set_xticks(x, [f.replace("subsets | ", "") for f in n.family], rotation=30, ha="right", fontsize=7.5)
        ax.set_title(f"{st}: NESTED (subset chosen on 3 animals, tested on the 4th)", fontsize=9)
        for a_ in axs[:, j]:
            for s_ in ("top", "right"):
                a_.spines[s_].set_visible(False)
    fig.suptitle("Which areas should the balance pool? All 63 subsets of ALM, tjM1, SSp-m, SSp-n, SSs, SSp-un for four fixed "
                 "definitions (top: each subset's leave-one-animal-out cvR², ranked);\nbottom: the subset (or subset x "
                 "definition) CHOSEN on 3 animals and tested on the 4th, bar = null 95% from repeating the selection on "
                 "shuffled sessions. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(il.OUT_W / "figures" / f"lick_balance_subsets{SFX}.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
