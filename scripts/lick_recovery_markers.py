"""Which per-session NEURAL measure tracks behavioural RECOVERY? Orofacial response SIZE vs hemispheric BALANCE, with the
cross-validated marker tests that back 14.16 (Priya 10-28: "yes let's do response size vs recovery"; "where are the LOAO
and LOSO cvR2 for the balance vs recovery figures"; "add pre references as grey cloud").

Recovery axis = far-contra FIRST-cued-lick aim vs the animal's pre landing (deg, - = ipsilesional; aim_deficit_sessions.csv
farcontra_first, key figure 1A). Hemispheres: _right = contralesional, _left = ipsilesional (configs/animals.yaml).

Predictors (session means; post-stroke sessions for every model):
  A. DECODER-WINDOW measures, per lick X in the decoder's 0..0.2 s window (x1e3 dF/F, as the balance of 12.18 / 14.16),
     referenced like the balance: per animal x lick set x lick order, the PRE line on lick angle; the residual is the
     value at the same angle and order (pre sessions scored LEAVE-ONE-PRE-SESSION-OUT, so the pre cloud is a real noise
     floor and not 0 by construction). Orofacial pool = ALM, tjM1, SSp-m, SSp-n:
       contra = mean of the 4 contralesional areas, ipsi = mean of the 4 ipsilesional, size = (contra + ipsi) / 2,
       balance = contra - ipsi (reproduces balreg_sessions r_orofacial). size and balance are the SAME two numbers
       rotated: size = what both sides share, balance = how they differ.
     Per area the same four; non-orofacial pool (visual, RSP, barrel, limb / trunk, ACAd pairs) balance and size.
  ITI FIRST AFTER QUIET: the ITI licks with >= 2 s (il.GAP_MAIN) since the previous lick onset, as the other ITI
     analyses define a bout start (Priya 10-28), with their own pre reference; a third lick set in every model.
  CUED FIRST LICK: only the first post-cue lick of each trial (order 0), own pre reference (Priya 10-28: "are we also
     using first post-cue lick"); 'cued' = every post-cue lick, as 14.16.
  B. RESPONSE AUC (the 14.x response measure: 0..1 s after the lick, baseline -1.0..-0.5 s; cued = first cued lick,
     ITI = bout starts; cue response 0..0.5 s after the cue), orofacial pool contra / ipsi / both, each as the change vs
     the animal's own pre mean (fraction), area_auc_sessions_split{_cued}.csv.
Models per predictor and lick set:
  per-animal r (rule 8, printed first); concurrent MixedLM aim ~ within + between + log(day) + (1 | animal);
  LOAO = leave-one-ANIMAL-out cvR2 (OLS aim ~ predictor, pooled SSE / SST around the overall mean; the 14.16 numbers),
  with a within-animal shuffle null (2000 permutations of the predictor across each animal's sessions) and the error per
  held-out animal; LOSO = leave-one-SESSION-out cvR2 with the animal UNKNOWN (pooled OLS) and KNOWN (animal intercepts).
  days = log(day) as the predictor.
Outputs <OUT_W>/recmark_sessions.csv (all sessions, incl. pre), recmark_models.csv, recmark_cv.csv,
recmark_cv_animals.csv; figures lick_recovery_markers_cv.png, lick_recovery_markers_size.png,
lick_recovery_markers_areas.png (figdata + --plot-only).
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
import lick_balance_regions as lbg  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

ORO = ["ALM", "tjM1", "SSp-m", "SSp-n"]
CTRL = {g: lbg.GROUPS[g] for g in lbg.CONTROLS}
AN = ["PS92", "PS93", "PS94", "PS95"]
QSET = "ITI first after quiet"
YAX = ("iti_ang", "iti_ang90")
C1SET = "cued first lick"
SETS3 = ("cued", C1SET, "ITI", QSET)
N_SHUF = 2000
# predictors shown in the summary figure (column, label)
KEY = [("days", "days since stroke"), ("oro_bal", "oro BALANCE"), ("oro_balz", "oro balance, z-scored avg"),
       ("oro_balw", "oro balance, pre-fit weights (own animal)"),
       ("oro_balwx", "oro balance, pre-fit weights (other animals)"), ("oro_size", "oro SIZE (both sides)"),
       ("oro_contra", "oro contra"), ("oro_ipsi", "oro ipsi"), ("non_bal", "non-oro balance"), ("non_size", "non-oro size"),
       ("auc_both", "AUC 0-1 s, both"), ("auc_contra", "AUC contra"), ("auc_ipsi", "AUC ipsi"),
       ("cue_both", "cue AUC, both")]


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
        pick = lambda v, m=m: np.where(m >= 0, v[np.maximum(m, 0)], np.nan)  # noqa: E731
        for a in ORO:
            R[f"{a}_contra"] = pick(X[:, cols.index(f"{a}_right")])
            R[f"{a}_ipsi"] = pick(X[:, cols.index(f"{a}_left")])
        # non-orofacial pool exactly as lick_balance_regions: per group the mean over its matched pairs, then the mean of
        # the five groups
        gc, gi = [], []
        for areas in CTRL.values():
            ok = [a for a in areas if f"{a}_right" in cols and f"{a}_left" in cols]
            gc.append(np.mean([X[:, cols.index(f"{a}_right")] for a in ok], 0))
            gi.append(np.mean([X[:, cols.index(f"{a}_left")] for a in ok], 0))
        R["non_contra"] = pick(np.mean(gc, 0))
        R["non_ipsi"] = pick(np.mean(gi, 0))
        out.append(R)
    return pd.concat(out, ignore_index=True)


def _gaps(label: str) -> pd.DataFrame:
    """ITI licks of one session -> time since the previous lick ONSET (s), for the 'first lick after a quiet period' set
    (Priya 10-28; >= il.GAP_MAIN = 2 s, the rule of the other ITI analyses). Module-level for spawn."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "t"])
    C = C[(C.label == label) & (C.set == "iti")]
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_on_s", "daq_peak_s"])
    A = A[np.isfinite(A.daq_on_s)].sort_values("daq_on_s").reset_index(drop=True)
    gap = pd.Series(np.r_[np.inf, np.diff(A.daq_on_s.to_numpy(float))], index=A.daq_peak_s.to_numpy(float))
    gap = gap[~gap.index.duplicated()]
    return C.assign(gap=gap.reindex(C.t.to_numpy(float)).to_numpy(float)).drop(columns="t")


def _residuals(P: pd.DataFrame, vs: list[str]) -> pd.DataFrame:
    """Per animal x set x lick order: value minus the PRE line on angle; pre sessions leave-one-pre-session-out."""
    out = []
    for (_st, _an), g in P.groupby(["set", "animal"]):
        g = g.copy()
        pre = g[g.epoch == "pre"]
        for v in vs:
            g[f"res_{v}"] = np.nan
            for k_, pk in pre.dropna(subset=[v, "angle"]).groupby("ord_c"):
                mk = (g.ord_c == k_).to_numpy()
                if len(pk) >= lbg.MIN_CELL and pk.angle.std() > 0:
                    a1, a0 = np.polyfit(pk.angle, pk[v], 1)
                    post = mk & (g.epoch != "pre").to_numpy()
                    g.loc[post, f"res_{v}"] = g.loc[post, v] - (a0 + a1 * g.loc[post, "angle"])
                for lab in pk.label.unique():                     # pre: refit without this session
                    q = pk[pk.label != lab]
                    sel = mk & (g.label == lab).to_numpy()
                    if len(q) >= lbg.MIN_CELL and q.angle.std() > 0:
                        b1, b0 = np.polyfit(q.angle, q[v], 1)
                        g.loc[sel, f"res_{v}"] = g.loc[sel, v] - (b0 + b1 * g.loc[sel, "angle"])
        out.append(g)
    return pd.concat(out, ignore_index=True)


def sessions() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), _session)
    if fail:
        raise RuntimeError(f"failed for {len(fail)} sessions: {fail[:5]}")
    P = P.merge(pd.concat([v for _i, v in res], ignore_index=True), on=["set", "label", "k"], how="left",
                validate="one_to_one")
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    P["ord_c"] = np.minimum(P.order, lbg.CAP)
    vs = []
    for a in ORO:
        P[f"{a}_size"] = (P[f"{a}_contra"] + P[f"{a}_ipsi"]) / 2
        P[f"{a}_bal"] = P[f"{a}_contra"] - P[f"{a}_ipsi"]
        vs += [f"{a}_{s}" for s in ("contra", "ipsi", "size", "bal")]
    P["oro_contra"] = P[[f"{a}_contra" for a in ORO]].mean(1)
    P["oro_ipsi"] = P[[f"{a}_ipsi" for a in ORO]].mean(1)
    P["oro_size"] = (P.oro_contra + P.oro_ipsi) / 2
    P["oro_bal"] = P.oro_contra - P.oro_ipsi
    P["non_size"] = (P.non_contra + P.non_ipsi) / 2
    P["non_bal"] = P.non_contra - P.non_ipsi
    vs += ["oro_contra", "oro_ipsi", "oro_size", "oro_bal", "non_contra", "non_ipsi", "non_size", "non_bal"]
    # two other ways to combine the four area balances (Priya 10-28): Z-SCORED average (each area's balance / its pre-lick
    # SD, per animal x set, so the four count equally) and a PRE-FIT WEIGHTED combination (per animal x set, OLS of lick
    # angle on the four balances over pre licks; weights frozen and applied to every lick = the balance axis that best
    # tracks direction pre-stroke, in deg). Weights use all pre licks, so pre sessions' values carry a small leak.
    P["oro_balz"], P["oro_balw"] = np.nan, np.nan
    bcols = [f"{a}_bal" for a in ORO]
    for (_st, _an), g in P.groupby(["set", "animal"]):
        pre = g[g.epoch == "pre"].dropna(subset=[*bcols, "angle"])
        sd = pre[bcols].std()
        P.loc[g.index, "oro_balz"] = (g[bcols] / sd).mean(1)
        for a, b in zip(ORO, bcols, strict=True):     # per-area z-scored balance (Priya 10-28: SSp-n scale differences)
            P.loc[g.index, f"{a}_balz"] = g[b] / sd[b]
        Xp = np.column_stack([np.ones(len(pre)), pre[bcols].to_numpy()])
        coef = np.linalg.lstsq(Xp, pre.angle.to_numpy(), rcond=None)[0]
        P.loc[g.index, "oro_balw"] = g[bcols].to_numpy() @ coef[1:]
    # FAIR pre-fit weights for the cross-animal test (Priya 10-28 "do the pre-fit weighted balance"): per lick set and
    # animal, OLS of lick angle on the four per-area Z-SCORED balances over the OTHER animals' pre licks (z puts the
    # animals on one scale); the frozen weights are applied to this animal's licks. The animal's own data never enter
    # its weights, so leave-one-animal-out stays clean.
    P["oro_balwx"] = np.nan
    zcols = [f"{a}_balz" for a in ORO]
    for st_ in P.set.unique():
        Ps = P[P.set == st_]
        for an in Ps.animal.unique():
            tr = Ps[(Ps.animal != an) & (Ps.epoch == "pre")].dropna(subset=[*zcols, "angle"])
            Xt = np.column_stack([np.ones(len(tr)), tr[zcols].to_numpy()])
            cw = np.linalg.lstsq(Xt, tr.angle.to_numpy(), rcond=None)[0]
            idx = Ps.index[Ps.animal == an]
            P.loc[idx, "oro_balwx"] = P.loc[idx, zcols].to_numpy() @ cw[1:]
    vs += ["oro_balz", "oro_balw", "oro_balwx", *[f"{a}_balz" for a in ORO]]
    # ITI RECOVERY AXIS of its own (Priya 10-28; a different question from the cued aim): per session the ITI lick
    # angle vs the animal's pre sessions, signed so that NEGATIVE = ipsilesional like the aim (sign taken from pre
    # far-contra cued licks): iti_ang = session mean, iti_ang90 = how far contralesionally ITI licks reach (90th pct).
    sgn = float(np.sign(P[(P.set == "trial") & (P.epoch == "pre") & (P.pos == "far_contra")].angle.mean()))
    It = P[P.set == "iti"].assign(sa=lambda d: sgn * d.angle)
    Ia = It.groupby(["label", "animal", "epoch"]).sa.agg(iti_ang="mean", iti_ang90=lambda s: np.nanpercentile(s, 90))
    Ia = Ia.reset_index()
    for c in ("iti_ang", "iti_ang90"):
        Ia[c] = Ia[c] - Ia.animal.map(Ia[Ia.epoch == "pre"].groupby("animal")[c].mean())
    ITI_AX = Ia[["label", "iti_ang", "iti_ang90"]]
    gres, gfail = ak.fan_sessions(sorted(P[P.set == "iti"].label.unique()), _gaps)
    if gfail:
        raise RuntimeError(f"gaps failed for {len(gfail)} sessions: {gfail[:5]}")
    G = pd.concat([v for _i, v in gres], ignore_index=True)
    Q = P[P.set == "iti"].merge(G, on=["set", "label", "k"], how="left", validate="one_to_one")
    Q = Q[Q.gap >= il.GAP_MAIN].assign(set="iti_q")        # own pre reference (fit on the same kind of licks)
    print(f"== ITI first-after-quiet licks (gap >= {il.GAP_MAIN} s): {len(Q)} of {int((P.set == 'iti').sum())} ITI licks; "
          f"order 0 share {(Q.order == 0).mean():.3f}")
    Q1 = P[(P.set == "trial") & (P.order == 0)].assign(set="trial_1")   # first post-cue lick only, own reference
    P = _residuals(pd.concat([P, Q.drop(columns="gap"), Q1], ignore_index=True), vs)
    S = P.groupby(["set", "label", "animal", "epoch"])[[f"res_{v}" for v in vs]].mean().reset_index()
    S.columns = [c.replace("res_", "") for c in S.columns]
    S["set"] = S.set.map({"trial": "cued", "iti": "ITI", "iti_q": QSET, "trial_1": C1SET})
    # B: response AUCs (session level), change vs the animal's own pre mean
    parts = []
    for st, fn in (("cued", "area_auc_sessions_split_cued.csv"), ("ITI", "area_auc_sessions_split.csv")):
        T = pd.read_csv(il.OUT_W / fn)
        T = T[T.area.isin([f"{a}_{h}" for a in ORO for h in ("right", "left")])].copy()
        T["side"] = np.where(T.area.str.endswith("_right"), "contra", "ipsi")
        out = None
        for col, nm in (("resp", "auc"), ("cue", "cue")):
            if col not in T:
                continue
            W = T.groupby(["label", "animal", "epoch", "side"])[col].mean().unstack("side")
            W["both"] = W[["contra", "ipsi"]].mean(1)
            W = W.reset_index()
            for c in ("contra", "ipsi", "both"):
                pm = W[W.epoch == "pre"].groupby("animal")[c].mean()
                W[f"{nm}_{c}"] = W[c] / W.animal.map(pm) - 1
            W = W[["label", *[f"{nm}_{c}" for c in ("contra", "ipsi", "both")]]]
            out = W if out is None else out.merge(W, on="label", how="outer")
        parts.append(out.assign(set=st))
        if st == "ITI":                      # the ITI AUC is already bout starts after >= 2 s without licking
            parts.append(out.assign(set=QSET))
        if st == "cued":                     # the cued AUC is already the first post-cue lick
            parts.append(out.assign(set=C1SET))
    S = S.merge(pd.concat(parts, ignore_index=True), on=["set", "label"], how="left")
    A = pd.read_csv(il.OUT_W / "aim_deficit_sessions.csv", usecols=["label", "day", "farcontra_first"])
    S = S.merge(A, on="label", how="left").rename(columns={"farcontra_first": "aim"})
    S = S.merge(ITI_AX, on="label", how="left")
    return S


def _ols_pred(Xtr, ytr, Xte):
    Xtr = np.column_stack([np.ones(len(Xtr)), Xtr])
    Xte = np.column_stack([np.ones(len(Xte)), Xte])
    return Xte @ np.linalg.lstsq(Xtr, ytr, rcond=None)[0]


def cv_r2(g: pd.DataFrame, cols: list[str], by: str, animal_known: bool = False) -> tuple[float, np.ndarray]:
    """Leave-one-`by`-out cvR2 of aim (pooled SSE / SST around the overall mean), and the held-out predictions."""
    y = g.aim.to_numpy()
    X = g[cols].to_numpy(float) if cols else np.zeros((len(g), 0))
    if animal_known:
        X = np.column_stack([X, pd.get_dummies(g.animal, drop_first=True).to_numpy(float)])
    pred = np.full(len(g), np.nan)
    for k in g[by].unique():
        te = (g[by] == k).to_numpy()
        pred[te] = _ols_pred(X[~te], y[~te], X[te])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), pred


N_BLOCKS, GAP = 4, 1


def cv_blocked(g: pd.DataFrame, cols: list[str]) -> float:
    """WITHIN-ANIMAL BLOCKED cvR2 (Priya 10-28: replaces leave-one-epoch-out, which tested extrapolation to an aim range
    the model never saw). Each animal's sessions, in day order, are cut into N_BLOCKS contiguous blocks; each block is
    held out in turn, its GAP neighbouring sessions on each side are dropped from training, and the model is fit on the
    rest (that animal's other sessions + the other animals) WITH animal intercepts. Pooled SSE / SST around the mean."""
    g = g.sort_values(["animal", "day"]).reset_index(drop=True)
    pos = g.groupby("animal").cumcount().to_numpy()
    n_a = g.groupby("animal").animal.transform("size").to_numpy()
    blk = np.floor(pos * N_BLOCKS / n_a).astype(int)
    y = g.aim.to_numpy()
    X = np.column_stack([g[cols].to_numpy(float), pd.get_dummies(g.animal, drop_first=True).to_numpy(float)])
    pred = np.full(len(g), np.nan)
    an = g.animal.to_numpy()
    for a in np.unique(an):
        for b in range(N_BLOCKS):
            te = (an == a) & (blk == b)
            if not te.any():
                continue
            lo_, hi_ = pos[te].min() - GAP, pos[te].max() + GAP
            excl = (an == a) & (pos >= lo_) & (pos <= hi_)
            pred[te] = _ols_pred(X[~excl], y[~excl], X[te])
    return 1 - np.nansum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def models(S: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf
    preds = [c for c in S.columns if c not in ("set", "label", "animal", "epoch", "day", "aim", "logday", *YAX)]
    rng = np.random.default_rng(0)
    rows, cvr, cva, an = [], [], [], []
    for st, G in S[(S.epoch != "pre") & (S.day > 0)].groupby("set"):
        G = G.dropna(subset=["aim"]).sort_values(["animal", "day"]).copy()
        G["logday"] = np.log(G.day)
        G["days"] = G.logday
        G["sess"] = np.arange(len(G))
        for v in ["days", *preds]:
            g = G.dropna(subset=[v]).copy()
            if g.animal.nunique() < 4 or len(g) < 20:
                continue
            for a_, h in g.groupby("animal"):
                an.append({"set": st, "pred": v, "animal": a_, "n": len(h), "r": np.corrcoef(h[v], h.aim)[0, 1]})
            if v != "days":
                g["w"] = g[v] - g.groupby("animal")[v].transform("mean")
                g["b"] = g.groupby("animal")[v].transform("mean")
                try:
                    m = smf.mixedlm("aim ~ w + b + logday", g, groups=g["animal"]).fit(reml=False)
                    rows.append({"set": st, "pred": v, "n": len(g), "slope": m.fe_params["w"],
                                 "lo": m.conf_int().loc["w", 0], "hi": m.conf_int().loc["w", 1], "p": m.pvalues["w"]})
                except Exception as ex:                                    # noqa: BLE001
                    rows.append({"set": st, "pred": v, "n": len(g), "note": f"FAILED {ex}"})
            # same sessions for every comparison: days evaluated on this predictor's sessions
            r_loao, pr = cv_r2(g, [v], "animal")
            r_days, pd_ = cv_r2(g, ["days"], "animal")
            null = []
            if v != "days":
                for _ in range(N_SHUF):
                    gs = g.copy()
                    gs[v] = gs.groupby("animal")[v].transform(lambda s: rng.permutation(s.to_numpy()))
                    null.append(cv_r2(gs, [v], "animal")[0])
            null = np.asarray(null)
            cvr.append({"set": st, "pred": v, "n": len(g), "loao": r_loao, "loao_days_same_sessions": r_days,
                        "loao_both": cv_r2(g, ["days", v], "animal")[0] if v != "days" else np.nan,
                        "null95": np.quantile(null, 0.95) if len(null) else np.nan,
                        "p_shuffle": (1 + np.sum(null >= r_loao)) / (1 + len(null)) if len(null) else np.nan,
                        "loso_unknown": cv_r2(g, [v], "sess")[0],
                        # within-animal BLOCKED (replaces leave-one-epoch-out, Priya 10-28; cv_blocked)
                        "loblk": cv_blocked(g, [v]),
                        "loblk_days_same_sessions": cv_blocked(g, ["days"]),
                        "loblk_plus_days": cv_blocked(g, ["days", v]) if v != "days" else np.nan,
                        "loso_known": cv_r2(g, [v], "sess", animal_known=True)[0],
                        "loso_known_plus_days": cv_r2(g, ["days", v], "sess", animal_known=True)[0] if v != "days"
                        else np.nan,
                        "loso_known_days_same_sessions": cv_r2(g, ["days"], "sess", animal_known=True)[0]})
            for a_ in AN:
                te = (g.animal == a_).to_numpy()
                if te.any():
                    cva.append({"set": st, "pred": v, "animal": a_, "mae": np.mean(np.abs(g.aim[te] - pr[te])),
                                "mae_days": np.mean(np.abs(g.aim[te] - pd_[te]))})
    return {"M": pd.DataFrame(rows), "CV": pd.DataFrame(cvr), "CVA": pd.DataFrame(cva), "AN": pd.DataFrame(an)}


def compute() -> dict:
    S = sessions()
    S.to_csv(il.OUT_W / "recmark_sessions.csv", index=False)
    # self-check: the balance reproduces 14.16's session values (post sessions; same reference)
    B = pd.read_csv(il.OUT_W / "balrec_sessions.csv", usecols=["set", "label", "r_orofacial"])
    chk = S.merge(B, on=["set", "label"])
    dmax = float(np.nanmax(np.abs(chk.oro_bal - chk.r_orofacial)))
    print(f"== self-check: oro_bal vs balrec r_orofacial, {len(chk)} sessions, max |diff| = {dmax:.2e}")
    F = models(S)
    # the ITI recovery axes (ITI lick sets only); the cued-aim results keep y = "aim"
    for k in ("M", "CV", "CVA", "AN"):
        F[k]["y"] = "aim"
    for yax in YAX:
        Fi = models(S[S.set.isin(["ITI", QSET])].assign(aim=lambda d, yax=yax: d[yax]))
        for k in ("M", "CV", "CVA", "AN"):
            F[k] = pd.concat([F[k], Fi[k].assign(y=yax)], ignore_index=True)
    F["M"].to_csv(il.OUT_W / "recmark_models.csv", index=False)
    F["CV"].to_csv(il.OUT_W / "recmark_cv.csv", index=False)
    F["CVA"].to_csv(il.OUT_W / "recmark_cv_animals.csv", index=False)
    pd.set_option("display.width", 240)
    print("== per animal (rule 8): within-animal r of predictor with aim\n" + F["AN"].pivot_table(
        index=["set", "pred"], columns="animal", values="r").round(2).to_string())
    print("== concurrent mixed models (within-animal slope, with log(day))\n" + F["M"].round(4).to_string(index=False))
    print("== cross-validated R2\n" + F["CV"].round(3).to_string(index=False))
    print("== error per held-out animal (deg): predictor / days\n" + F["CVA"].pivot_table(
        index=["set", "pred"], columns="animal", values=["mae", "mae_days"]).round(1).to_string())
    return {"S": S, **F, "selfcheck": dmax}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    YY = {k: (F[k][F[k].y == "aim"] if "y" in F[k] else F[k]) for k in ("CV", "CVA", "M", "AN")}
    CV, CVA, M, S = YY["CV"], YY["CVA"], YY["M"], F["S"]
    fig_dir = il.OUT_W / "figures"
    # 1) cross-validated summary
    fig, axs = plt.subplots(4, 3, figsize=(20, 20), gridspec_kw={"width_ratios": [1.25, 1.25, 0.9]})
    for i, st in enumerate(SETS3):
        c = CV[CV.set == st].set_index("pred")
        ks = [(k, lab) for k, lab in KEY if k in c.index]
        x = np.arange(len(ks))
        ax = axs[i, 0]
        v = [c.loc[k, "loao"] for k, _ in ks]
        ax.bar(x, v, color=["0.55" if k == "days" else "#C44E52" if "bal" in k else "#4C72B0" for k, _ in ks])
        ax.plot(x, [c.loc[k, "null95"] for k, _ in ks], "k_", ms=14, mew=1.5, label="shuffle null, 95th pct")
        for xi, (k, _) in enumerate(ks):
            p = c.loc[k, "p_shuffle"]
            if np.isfinite(p):
                ax.text(xi, max(v[xi], 0) + 0.03, f"p {p:.3g}", ha="center", fontsize=6.5,
                        fontweight="bold" if p < 0.05 and v[xi] > 0 else "normal")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(max(min(-0.6, np.nanmin(v) - 0.05), -1.5), 1)
        ax.set_xticks(x, [lab for _, lab in ks], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{st}: leave-one-ANIMAL-out cvR² of the far-contra aim (unseen animal)", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax = axs[i, 1]
        w = 0.21
        for j, (col, lab, cl) in enumerate((("loblk", "within-animal blocked (4 blocks, 1-session gap)", "#64B5CD"),
                                            ("loso_unknown", "leave-one-session-out, animal unknown", "#DD8452"),
                                            ("loso_known", "leave-one-session-out, animal known", "#55A868"),
                                            ("loso_known_plus_days", "... animal known + days", "#8172B3"))):
            ax.bar(x + (j - 1.5) * w, [c.loc[k, col] for k, _ in ks], width=w * 0.95, color=cl, label=lab)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-0.6, 1)
        ax.set_xticks(x, [lab for _, lab in ks], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{st}: within-animal BLOCKED and leave-one-SESSION-out cvR²", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
        ax = axs[i, 2]
        sel = [k for k in ("oro_bal", "oro_size", "auc_both") if k in c.index]
        a = CVA[(CVA.set == st)]
        t0 = a[a.pred == "oro_bal"].set_index("animal").reindex(AN)
        ax.bar(np.arange(4) - 0.3, t0.mae_days, width=0.19, color="0.55", label="days since stroke")
        for j, k in enumerate(sel):
            t_ = a[a.pred == k].set_index("animal").reindex(AN)
            ax.bar(np.arange(4) + (j - 0.5) * 0.2, t_.mae, width=0.19, label=dict(KEY)[k],
                   color=["#C44E52", "#4C72B0", "#64B5CD"][j])
        ax.set_xticks(range(4), AN)
        ax.set_ylabel("error in the held-out animal (deg)")
        ax.set_title(f"{st}: error per held-out animal (LOAO)", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        for a_ in axs[i]:
            for s_ in ("top", "right"):
                a_.spines[s_].set_visible(False)
    fig.suptitle("Which session measure estimates RECOVERY (far-contra first-lick aim)? BALANCE (contra - ipsi, red) vs SIZE "
                 "(blue) of the orofacial lick response, vs days since stroke (grey). Post-stroke sessions.\nDecoder-window "
                 "measures (0-0.2 s, same lick angle & order as pre): oro balance / size / contra / ipsi, non-oro; AUC = the 14.x "
                 "response (0-1 s, / animal's pre - 1). OLS aim ~ measure; pooled R². Shuffle = measure permuted "
                 "across each animal's sessions (2000x). PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(fig_dir / "lick_recovery_markers_cv.png", dpi=95)
    plt.close(fig)
    # 2) scatter: size vs aim and balance vs aim, with the pre cloud
    fig, axs = plt.subplots(4, 3, figsize=(18, 20.5))
    AN_ = YY["AN"]
    for i, st in enumerate(SETS3):
        for j, (k, lab) in enumerate((("oro_bal", "orofacial BALANCE (contra - ipsi)"),
                                      ("oro_size", "orofacial SIZE (mean of both sides)"),
                                      ("auc_both", "orofacial response AUC 0-1 s, both sides (/ pre - 1)"))):
            ax = axs[i, j]
            g = S[S.set == st]
            pre = g[g.epoch == "pre"]
            ax.plot(pre[k], pre.aim, "o", color="0.8", ms=6, mec="0.6", mew=0.4, zorder=1)
            for (ep, a_), h in g[g.epoch != "pre"].groupby(["epoch", "animal"]):
                ax.plot(h[k], h.aim, ANIMAL_MARKER.get(a_, "o"), color=EPOCH_LINE[ep], ms=6, mec="k", mew=0.3, zorder=2)
            m = M[(M.set == st) & (M.pred == k)]
            r = AN_[(AN_.set == st) & (AN_.pred == k)].set_index("animal").r
            c = CV[(CV.set == st) & (CV.pred == k)]
            ttl = f"{st}: {lab}"
            if len(m) and "p" in m and np.isfinite(m.p.iloc[0]):
                s_ = f"within-animal slope {m.slope.iloc[0]:+.2f}, p {m.p.iloc[0]:.2g} (with log day)"
                ttl += "\n" + (rf"$\mathbf{{{s_.replace(' ', '~')}}}$" if m.p.iloc[0] < 0.05 else s_)
            ttl += "\nper-animal r " + " / ".join(f"{r.get(x, np.nan):+.2f}" for x in AN)
            if len(c):
                ttl += f"; LOAO cvR² {c.loao.iloc[0]:.2f} (days {c.loao_days_same_sessions.iloc[0]:.2f})"
            ax.set_title(ttl, fontsize=8.5)
            ax.axhline(0, color="0.7", lw=0.6)
            ax.axvline(0, color="0.7", lw=0.6)
            ax.set_xlabel(lab + " vs pre", fontsize=8.5)
            ax.set_ylabel("far-contra first-lick aim (deg vs pre)", fontsize=8.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    h = [Line2D([], [], color="0.8", marker="o", mec="0.6", lw=0, label="pre (left-out reference)")]
    h += [Line2D([], [], color=EPOCH_LINE[e], marker="o", lw=0, label=e) for e in ("acute", "subacute", "chronic")]
    h += [Line2D([], [], color="0.3", marker=ANIMAL_MARKER.get(x, "o"), lw=0, label=x) for x in AN]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=9, frameon=False)
    fig.suptitle("Recovery vs orofacial BALANCE and SIZE, one point per session (grey = pre sessions scored "
                 "leave-one-pre-session-out: the noise floor; not in the fits). Mixed models on post-stroke sessions, random "
                 "intercept per animal, log(day). PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(fig_dir / "lick_recovery_markers_size.png", dpi=95)
    plt.close(fig)
    # 3) per area: concurrent slope p and LOAO cvR2 for contra / ipsi / size / balance
    fig, axs = plt.subplots(1, 4, figsize=(30, 5.5))
    for i, st in enumerate(SETS3):
        ax = axs[i]
        c = CV[CV.set == st].set_index("pred")
        m = M[M.set == st].set_index("pred")
        kinds = ["contra", "ipsi", "size", "bal", "balz"]
        for j, kd in enumerate(kinds):
            ks = [f"{a}_{kd}" for a in [*ORO, "oro"]]
            vals = [c.loc[k, "loao"] if k in c.index else np.nan for k in ks]
            xs = np.arange(len(ks)) + (j - 2) * 0.17
            ax.bar(xs, vals, width=0.16, label={"bal": "balance", "balz": "balance, z-scored per area"}.get(kd, kd),
                   color=["#937860", "#8C8C8C", "#4C72B0", "#C44E52", "#F28E8E"][j])
            for xi, k in zip(xs, ks, strict=True):
                if k in m.index and "p" in m and np.isfinite(m.loc[k, "p"]) and m.loc[k, "p"] < 0.05:
                    ax.text(xi, max(c.loc[k, "loao"], 0) + 0.02, "*", ha="center", fontsize=10)
        ax.axhline(c.loc["days", "loao"], color="0.4", ls="--", lw=1, label="days")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1, 1)
        ax.set_xticks(range(len(ORO) + 1), [*ORO, "orofacial pool"])
        ax.set_title(f"{st}: LOAO cvR² per area (* = within-animal mixed-model slope p < 0.05)", fontsize=10)
        ax.legend(fontsize=8, frameon=False, ncol=5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("Per area: does each side, both sides together (size) or the difference (balance) estimate recovery in an "
                 "unseen animal? Decoder-window measures at the same lick angle & order. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(fig_dir / "lick_recovery_markers_areas.png", dpi=95)
    plt.close(fig)
    # 4) ITI's own recovery axes (ITI lick sets): LOAO per measure + balance vs axis scatter with the pre cloud
    if "y" not in F["CV"] or not F["CV"].y.isin(YAX).any():
        return
    CVy, ANy, My = F["CV"], F["AN"], F["M"]
    ylab = {"iti_ang": "mean ITI lick angle vs pre (deg; - = ipsilesional)",
            "iti_ang90": "contralesional reach of ITI licks, 90th pct angle vs pre (deg)"}
    fig, axs = plt.subplots(len(YAX), 3, figsize=(21, 6.5 * len(YAX)), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for i, yax in enumerate(YAX):
        for j, st in enumerate(("ITI", QSET)):
            ax = axs[i, j]
            c = CVy[(CVy.y == yax) & (CVy.set == st)].set_index("pred")
            ks = [(k, lab) for k, lab in KEY if k in c.index]
            x = np.arange(len(ks))
            v = [c.loc[k, "loao"] for k, _ in ks]
            ax.bar(x, v, color=["0.55" if k == "days" else "#C44E52" if "bal" in k else "#4C72B0" for k, _ in ks])
            ax.plot(x, [c.loc[k, "null95"] for k, _ in ks], "k_", ms=14, mew=1.5)
            for xi, (k, _) in enumerate(ks):
                p = c.loc[k, "p_shuffle"]
                if np.isfinite(p):
                    ax.text(xi, max(v[xi], 0) + 0.03, f"p {p:.3g}", ha="center", fontsize=6.5,
                            fontweight="bold" if p < 0.05 and v[xi] > 0 else "normal")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylim(max(min(-0.6, np.nanmin(v) - 0.05), -1.5), 1)
            ax.set_xticks(x, [lab for _, lab in ks], rotation=35, ha="right", fontsize=8)
            ax.set_title(f"{st} balance etc. -> {yax}: leave-one-ANIMAL-out cvR²", fontsize=10)
        ax = axs[i, 2]
        g = S[S.set == QSET]
        pre = g[g.epoch == "pre"]
        ax.plot(pre.oro_bal, pre[yax], "o", color="0.82", mec="0.6", mew=0.4, ms=6, zorder=0)
        for (ep, a_), h in g[g.epoch != "pre"].groupby(["epoch", "animal"]):
            ax.plot(h.oro_bal, h[yax], ANIMAL_MARKER.get(a_, "o"), color=EPOCH_LINE[ep], ms=6, mec="k", mew=0.3)
        r = ANy[(ANy.y == yax) & (ANy.set == QSET) & (ANy.pred == "oro_bal")].set_index("animal").r
        rd_ = ANy[(ANy.y == yax) & (ANy.set == QSET) & (ANy.pred == "days")].set_index("animal").r
        m = My[(My.y == yax) & (My.set == QSET) & (My.pred == "oro_bal")]
        ttl = (f"{QSET}: orofacial balance vs {yax}\nper-animal r " + " / ".join(f"{r.get(x_, np.nan):+.2f}" for x_ in AN)
               + "\n(days vs axis: " + " / ".join(f"{rd_.get(x_, np.nan):+.2f}" for x_ in AN) + ")")
        if len(m) and "p" in m and np.isfinite(m.p.iloc[0]):
            ttl += f"; within-animal slope p {m.p.iloc[0]:.2g}"
        ax.set_title(ttl, fontsize=8.5)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("orofacial balance vs pre (first ITI lick after quiet)", fontsize=8.5)
        ax.set_ylabel(ylab[yax], fontsize=8.5)
        for a_ in axs[i]:
            for s_ in ("top", "right"):
                a_.spines[s_].set_visible(False)
    fig.suptitle("ITI licks against an ITI RECOVERY AXIS of their own (a different question from the cued aim): per session, "
                 "the ITI lick angle vs the animal's pre sessions (mean; and the 90th percentile toward contralesional).\n"
                 "Same measures and cross-validation as the cued figure; grey = pre sessions (not fitted). PROVISIONAL",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "lick_recovery_markers_iti_axis.png", dpi=95)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_recovery_markers"))
        return 0
    if "--models-only" in sys.argv:          # re-fit the models on the saved session table (no per-session pass)
        F = load_fd("lick_recovery_markers")
        F.update(models(F["S"]))
        for k, f in (("M", "recmark_models"), ("CV", "recmark_cv"), ("CVA", "recmark_cv_animals")):
            F[k].to_csv(il.OUT_W / f"{f}.csv", index=False)
        print(F["CV"][["set", "pred", "loao", "loblk", "loblk_days_same_sessions", "loso_unknown"]].round(3).to_string(index=False))
        save_fd("lick_recovery_markers", F)
        plot(F)
        return 0
    F = compute()
    save_fd("lick_recovery_markers", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
