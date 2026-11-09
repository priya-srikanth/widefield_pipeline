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
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

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

from wfield_local import mask_keys  # noqa: E402

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
# REFERENCE METHOD (env WFIELD_BAL_REFMETHOD; Priya 10-28 "still using lick angle gaussian?"): "line" (default, the pre
# straight line on the covariate) or "gauss" (as the decoder's like-for-like offset: Gaussian-weighted mean of the pre
# licks of the same animal x set x order at the same LATERAL landing place, sigma GAUSS_SIGMA mm; pre sessions
# leave-one-session-out; NaN where the summed weight < GAUSS_MINW, i.e. no same-place pre licks -- no extrapolation).
REFMETHOD = os.environ.get("WFIELD_BAL_REFMETHOD", "line")
GAUSS_SIGMA, GAUSS_MINW = 0.1, 2.0
# INSIDE-RANGE control (env WFIELD_BAL_INSIDE=1; 10-28: acute licks fall beyond the pre angle range, where the line
# is EXTRAPOLATED): post-stroke licks outside the animal's pre 2.5..97.5 % range of the covariate (per set) are dropped.
INSIDE = os.environ.get("WFIELD_BAL_INSIDE", "") == "1"
# KINEMATICS-MATCHED reference (env WFIELD_BAL_KIN=1; 10-28 test 2: is the balance just altered execution?): the pre
# reference is a multiple regression on lick angle + 3-D reach + peak tongue speed + elevation instead of angle alone.
KIN = os.environ.get("WFIELD_BAL_KIN", "") == "1"
KIN_COVS = ["angle", "reach_mm", "speed_peak", "elev_deg"]
# MO-rest candidates (env WFIELD_BAL_MO=1; Priya 10-28 "MOp / MOs rest weren't worth including?"): MOs_rest and MOp_rest
# added to the areas; definitions limited to the primary + two references to keep the 255-subset search tractable.
MO = os.environ.get("WFIELD_BAL_MO", "") == "1"
# FULL first-lick set (env WFIELD_BAL_FULL=1; Priya 10-29 "do the re-extraction"): the cued first-lick set ('trial_1')
# from lick_firstlick_extract.py (every first post-cue lick, ~3x the capped set) instead of the decoder's capped licks.
FULL = os.environ.get("WFIELD_BAL_FULL", "") == "1"
if MO:
    AREAS = [*AREAS, "MOs_rest", "MOp_rest"]
    DEFS = {k: DEFS[k] for k in ("ratio_010_050", "ratio_020_035", "raw_010_020")}
SFX = ("" if REF == "angle" else f"_{REF}") + ("" if REFMETHOD == "line" else f"_{REFMETHOD}") + ("_inside" if INSIDE else "") + ("_kin" if KIN else "") + ("_mo" if MO else "") + ("_full" if FULL else "")


def _residuals_multi(P: pd.DataFrame, vs: list[str], covs: list[str]) -> pd.DataFrame:
    """As lrm._residuals with a MULTIPLE regression on `covs` (per animal x set x lick order, pre licks; pre sessions
    leave-one-session-out). Licks with a missing covariate get NaN."""
    out = []
    for (_st, _an), g in P.groupby(["set", "animal"]):
        g = g.copy()
        res = np.full((len(g), len(vs)), np.nan)
        okc = g[covs].notna().all(1).to_numpy()
        for _o, go in g.groupby("ord_c"):
            ix = np.flatnonzero(g.index.isin(go.index))
            pre = go[(go.epoch == "pre") & go[covs].notna().all(1)]
            if len(pre) < max(lbg.MIN_CELL, 3 * len(covs)):
                continue
            Xall = np.column_stack([np.ones(len(go)), go[covs].to_numpy(float)])
            V = go[vs].to_numpy(float)
            for lab in [None, *pre.label.unique()]:
                tr = pre[pre.label != lab] if lab else pre
                if len(tr) < max(lbg.MIN_CELL, 3 * len(covs)):
                    continue
                Xt = np.column_stack([np.ones(len(tr)), tr[covs].to_numpy(float)])
                Vt = tr[vs].to_numpy(float)
                B = np.full((Xt.shape[1], len(vs)), np.nan)
                for j in range(len(vs)):
                    okv = np.isfinite(Vt[:, j])
                    if okv.sum() > Xt.shape[1] + 2:
                        B[:, j] = np.linalg.lstsq(Xt[okv], Vt[okv, j], rcond=None)[0]
                sel = (go.label == lab).to_numpy() if lab else (go.epoch != "pre").to_numpy()
                rows = ix[sel]
                res[rows] = V[sel] - Xall[sel] @ B
        res[~okc] = np.nan
        g[[f"res_{v}" for v in vs]] = res
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _residuals_gauss(P: pd.DataFrame, vs: list[str]) -> pd.DataFrame:
    """value - Gaussian-weighted mean of the same-animal x set x order PRE licks at the same lateral place (side_k, mm)."""
    out = []
    for (_st, _an, _o), g in P.groupby(["set", "animal", "ord_c"]):
        g = g.copy()
        pre = g[g.epoch == "pre"].dropna(subset=["side_k"])
        V = pre[vs].to_numpy(float)
        xp = pre.side_k.to_numpy(float)
        labp = pre.label.to_numpy()
        res = np.full((len(g), len(vs)), np.nan)
        for lab in g.label.unique():
            ix = np.flatnonzero((g.label == lab).to_numpy())
            x = g.side_k.to_numpy(float)[ix]
            keep = labp != lab                                       # pre sessions: leave own session out
            if not keep.any():
                continue
            Wt = np.exp(-0.5 * ((x[:, None] - xp[None, keep]) / GAUSS_SIGMA) ** 2)
            Vk = V[keep]
            okv = np.isfinite(Vk)
            num = Wt @ np.where(okv, Vk, 0.0)
            den = Wt @ okv.astype(float)
            ref = np.where(den >= GAUSS_MINW, num / np.maximum(den, 1e-12), np.nan)
            res[ix] = g[vs].to_numpy(float)[ix] - ref
        g[[f"res_{v}" for v in vs]] = res
        out.append(g)
    return pd.concat(out, ignore_index=True)


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
        cx = cols
        if MO:                                              # MOs_rest / MOp_rest (exact, iti_licks.with_rest)
            X, cx = il.with_rest(X, cols, label, basis="split")
        R = {"set": st, "label": label, "k": np.arange(len(act))}
        for a in AREAS:
            for side, h in (("c", "right"), ("i", "left")):
                tr = X[:, :, cx.index(f"{a}_{h}")]
                base = tr[:, base_m].mean(1)
                for dn, (kind, lo_, hi_) in DEFS.items():
                    v = tr[:, _wm(lo_, hi_)].mean(1) - (base if kind == "bs" else 0)
                    R[f"{dn}|{a}|{side}"] = np.where(ok, v[mm], np.nan)
        out.append(pd.DataFrame(R))
    return pd.concat(out, ignore_index=True)


def _full_session(label: str) -> pd.DataFrame:
    """Every first post-cue lick of a session from <OUT_W>/firstlick_full (no 1500 cap), the same features as _session."""
    f = il.OUT_W / "firstlick_full" / f"{label}.npz"
    if not f.exists():
        return pd.DataFrame()
    z = np.load(f, allow_pickle=True)
    mask_keys.check_obj(z, f)                     # stale-read guard (11-09)
    X = z["X"].astype(float)
    cx = list(z["cols"])
    if MO:
        X, cx = il.with_rest(X, cx, label, basis="split")
    base_m = _wm(*il.BASE_RESP)
    R = {"set": "trial_1", "label": label, "animal": str(z["animal"]), "epoch": str(z["epoch"]), "pos": z["pos"],
         "order": 0, "ord_c": 0, "side_k": z["side_ml"], "reach_mm": z["reach_mm"], "speed_peak": z["speed_peak"],
         "elev_deg": z["elev_deg"]}
    for a in AREAS:
        for side, h in (("c", "right"), ("i", "left")):
            tr = X[:, :, cx.index(f"{a}_{h}")]
            base = tr[:, base_m].mean(1)
            for dn, (kind, lo_, hi_) in DEFS.items():
                R[f"{dn}|{a}|{side}"] = tr[:, _wm(lo_, hi_)].mean(1) - (base if kind == "bs" else 0)
    return pd.DataFrame(R)


def sessions() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k", "speed_peak", "elev_deg"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    if il.ITI_SUBSET:                                      # docked-spout etc. subset (10-29); k = cache index, kept
        P = P[il.iti_subset_keep(P)].reset_index(drop=True)
        print(f"ITI subset filter {il.ITI_SUBSET}: {int((P.set == 'iti').sum())} ITI licks kept", flush=True)
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
    if FULL:                                               # every first post-cue lick (lick_firstlick_extract.py)
        labs = sorted(P[P.set == "trial"].label.unique())
        fres, ffail = ak.fan_sessions(labs, _full_session)
        if ffail:
            raise RuntimeError(f"full first-lick load failed: {ffail[:5]}")
        Q1 = pd.concat([v for _i, v in fres], ignore_index=True)
        Q1["angle"] = (np.degrees(np.arcsin(np.clip(Q1.side_k / Q1.reach_mm, -1, 1))) if REF == "angle"
                       else Q1.side_k.astype(float))
        print(f"== FULL first-lick set: {len(Q1)} licks (capped set: "
              f"{int(((P.set == 'trial') & (P.order == 0)).sum())})", flush=True)
    else:
        Q1 = P[(P.set == "trial") & (P.order == 0)].assign(set="trial_1")
    P = pd.concat([P, Q, Q1], ignore_index=True)
    P = (_residuals_gauss(P, vs) if REFMETHOD == "gauss" else _residuals_multi(P, vs, KIN_COVS) if KIN
         else lrm._residuals(P, vs))
    if INSIDE:
        rng_ = P[P.epoch == "pre"].groupby(["set", "animal"]).angle.quantile([0.025, 0.975]).unstack()
        key = pd.MultiIndex.from_arrays([P.set, P.animal])
        lo_ = rng_[0.025].reindex(key).to_numpy(float)
        hi_ = rng_[0.975].reindex(key).to_numpy(float)
        out_ = (P.epoch != "pre").to_numpy() & ((P.angle.to_numpy() < lo_) | (P.angle.to_numpy() > hi_))
        print(f"== inside-range control: dropping {int(out_.sum())} of {int((P.epoch != 'pre').sum())} post licks",
              flush=True)
        print(P.assign(out=out_)[P.epoch != "pre"].groupby(["set", "epoch"]).out.mean().round(3).to_string(), flush=True)
        P = P[~out_].copy()
    # PER LICK ORDER sums for the primary definition and pool (Priya 10-29: rebuild 'balance over lick order' with the
    # updated balance): per set x session x lick order, sum of observed / predicted contra and ipsi over the pool areas
    prim, pool = "ratio_010_050", ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]
    if prim in DEFS:
        orows = []
        for (st, lab, an, ep, o), g in P.groupby(["set", "label", "animal", "epoch", "ord_c"]):
            rec = {"set": SETS[st], "label": lab, "animal": an, "epoch": ep, "ord_c": o, "n": len(g)}
            for s in ("c", "i"):
                ob = g[[f"{prim}|{a}|{s}" for a in pool]].sum(1)
                rs = g[[f"res_{prim}|{a}|{s}" for a in pool]].sum(1)
                ok = ob.notna() & rs.notna()
                rec[f"obs_{s}"], rec[f"pred_{s}"] = ob[ok].sum(), (ob[ok] - rs[ok]).sum()
            orows.append(rec)
        pd.DataFrame(orows).to_csv(il.OUT_W / f"balsub_order{SFX}.csv", index=False)
    rows = []
    for (st, lab, an, ep), g in P.groupby(["set", "label", "animal", "epoch"]):
        rec = {"set": SETS[st], "label": lab, "animal": an, "epoch": ep, "n_licks": len(g)}
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
    cl = dict(zip(DEFS, ["#4C72B0", "#64B5CD", "#C44E52", "#DD8452", "#8172B3", "#55A868", "#937860", "#DA8BC3"]))
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
    save_png_svg(fig, il.OUT_W / "figures" / f"lick_balance_subsets{SFX}.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
