"""What tracks ITI-lick RECOVERY? (Priya 10-28: "I really want to think about how to predict / what is correlating with ITI
lick recovery -- the decoder still works on it, so it's somewhere"; "proceed with your proposed ITI next steps").

ITI recovery axis = iti_ang: session mean ITI lick angle vs the animal's pre sessions, negative = ipsilesional
(lick_recovery_markers; -12.6 / -6.0 / -3.8 deg acute / subacute / chronic). Lick sets: ITI (all ITI licks) and ITI first
lick after >= 2 s without licking. Post-stroke sessions in every model; per-animal r printed first (rule 8).

A. DECODER OFFSET, COVERAGE CONTROL. The like-for-like offset (read-out minus same-place pre licks) tracks iti_ang within
   animals beyond log(day) (-17 deg / mm, 4/4 negative). Sessions with more ipsilesional licking put more licks where few
   pre licks landed, where the reference is thin and decoder shrinkage could inflate the offset. Session offset
   recomputed from (i) all licks, (ii) licks INSIDE the animal's pre landing range (likeforlike 'outside' == 0), (iii)
   licks inside the central 10-90 % of the animal's pre landings of that set. Same models.
B. WHERE: per decoder feature (every area x hemisphere, raw 0..0.2 s window = the decoder's input) the per-lick value at the
   same lick angle & order as pre (lick_recovery_markers._residuals), session mean; within-animal partial r with iti_ang
   (log day removed) per animal, and its mean -> an area map.
C. LEARNED RECOVERY DIRECTION: ridge regression of iti_ang on all features' session means (z-scored within the training
   animals), leave-one-ANIMAL-out with the ridge penalty chosen by inner leave-one-animal-out, against days and a
   within-animal shuffle null (N_SHUF); also leave-one-epoch-out.
Outputs <OUT_W>/itirec_offset_sessions.csv, itirec_feature_sessions.csv, itirec_where.csv, itirec_learned.csv; figure
lick_iti_recovery_search.png (figdata + --plot-only).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_reference as lbr  # noqa: E402
import lick_balance_regions as lbg  # noqa: E402
import lick_recovery_markers as lrm  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

AN = lrm.AN
N_SHUF = 500
ALPHAS = np.logspace(-2, 4, 13)
# MODE (env WFIELD_RECSEARCH_SET, read at import so spawned workers see it; Priya 10-28 "can we do that for cued licks
# too"): "" = ITI licks vs ITI's own axis (iti_ang); "cued" = cued licks (all, and the first post-cue lick) vs the
# far-contra first-lick aim.
MODE = os.environ.get("WFIELD_RECSEARCH_SET", "")
CUED = MODE == "cued"
SETS = {"trial": "cued", "trial_1": lrm.C1SET} if CUED else {"iti": "ITI", "iti_q": lrm.QSET}
RAW, SUB = ("trial", "trial_1") if CUED else ("iti", "iti_q")
XK, SK = ("tr_X", "tr_side") if CUED else ("iti_X", "iti_side")
YSRC = "aim" if CUED else "iti_ang"
YLAB = "far-contra first-lick aim (deg vs pre)" if CUED else "ITI lick angle vs pre (deg)"
TAG = "cuedrec" if CUED else "itirec"
FIGN = "lick_cued_recovery_search" if CUED else "lick_iti_recovery_search"
WHAT = "cued recovery (far-contra aim)" if CUED else "ITI recovery"


def _session(label: str) -> pd.DataFrame:
    acts = lbr.cv._acts(label)
    d = il._load(label)
    act = np.asarray(acts.get(RAW, []), float)
    if not len(act):
        return pd.DataFrame()
    m = il.subseq_match(d[SK].astype(float), act)
    X = d[XK].astype(float)[:, lbr.W, :].mean(1)
    R = pd.DataFrame(np.where((m >= 0)[:, None], X[np.maximum(m, 0)], np.nan), columns=[f"f|{c}" for c in d["cols"]])
    return pd.concat([pd.DataFrame({"set": RAW, "label": label, "k": np.arange(len(act))}), R], axis=1)


def licks() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    P = lo.per_lick(P)                                     # lfl + lfl_outside, both sets (reference per set x animal)
    P = P[P.set == RAW].copy()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), _session)
    if fail:
        raise RuntimeError(f"failed for {len(fail)} sessions: {fail[:5]}")
    P = P.merge(pd.concat([v for _i, v in res], ignore_index=True), on=["set", "label", "k"], how="left",
                validate="one_to_one")
    if not CUED:
        gres, gfail = ak.fan_sessions(sorted(P.label.unique()), lrm._gaps)
        if gfail:
            raise RuntimeError(f"gaps failed: {gfail[:5]}")
        P = P.merge(pd.concat([v for _i, v in gres], ignore_index=True), on=["set", "label", "k"], how="left",
                    validate="one_to_one")
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    P["ord_c"] = np.minimum(P.order, lbg.CAP)
    # central 10-90 % of the animal's pre landings (ITI)
    q = P[P.epoch == "pre"].groupby("animal").act.quantile([0.1, 0.9]).unstack()
    P["central"] = (P.act >= P.animal.map(q[0.1])) & (P.act <= P.animal.map(q[0.9]))
    Q = (P[P.order == 0] if CUED else P[P.gap >= il.GAP_MAIN]).assign(set=SUB)
    return pd.concat([P, Q], ignore_index=True)


def compute() -> dict:
    P = licks()
    feats = [c for c in P.columns if c.startswith("f|")]
    P = lrm._residuals(P, feats)
    R = pd.read_csv(il.OUT_W / "recmark_sessions.csv", usecols=["label", "day", YSRC]).drop_duplicates("label")
    R = R.rename(columns={YSRC: "yrec"})
    # A: offset coverage control
    rows = []
    for (st, lab, an, ep), g in P.groupby(["set", "label", "animal", "epoch"]):
        rows.append({"set": SETS[st], "label": lab, "animal": an, "epoch": ep,
                     "off_all": g.lfl.mean(), "off_inside": g.lfl[g.lfl_outside == 0].mean(),
                     "off_central": g.lfl[g.central].mean(), "frac_outside": (g.lfl_outside != 0).mean(),
                     "frac_central": g.central.mean()})
    O = pd.DataFrame(rows).merge(R, on="label", how="left")
    O.to_csv(il.OUT_W / f"{TAG}_offset_sessions.csv", index=False)
    lrm.N_SHUF = N_SHUF
    FA = {k: [] for k in ("M", "CV", "AN")}
    for st in SETS.values():
        T = O[O.set == st].drop(columns=["frac_outside", "frac_central"]).rename(columns={"yrec": "aim"})
        F = lrm.models(T)
        for k in FA:
            FA[k].append(F[k])
    FA = {k: pd.concat(v, ignore_index=True) for k, v in FA.items()}
    pd.set_option("display.width", 230)
    print("== A. fraction of post licks outside the pre range / outside the central 10-90 %:\n"
          + O[O.epoch != "pre"].groupby(["set", "epoch"])[["frac_outside", "frac_central"]].mean().round(3).to_string())
    print("== A. per animal r with iti_ang (rule 8)\n" + FA["AN"].pivot_table(index=["set", "pred"], columns="animal",
                                                                               values="r").round(2).to_string())
    print("== A. models\n" + FA["CV"][["set", "pred", "loao", "p_shuffle", "loblk", "loso_unknown", "loso_known"]]
          .merge(FA["M"][["set", "pred", "slope", "p"]], on=["set", "pred"], how="left").round(3).to_string(index=False))
    # B: where -- per feature, within-animal partial r with iti_ang (log day removed), per animal
    S = P.groupby(["set", "label", "animal", "epoch"])[[f"res_{f}" for f in feats]].mean().reset_index()
    S.columns = [c.replace("res_f|", "") for c in S.columns]
    S["set"] = S.set.map(SETS)
    S = S.merge(R, on="label", how="left")
    S.to_csv(il.OUT_W / f"{TAG}_feature_sessions.csv", index=False)
    cols = [f[2:] for f in feats]
    W = []
    for st, G in S[(S.epoch != "pre") & (S.day > 0)].groupby("set"):
        G = G.dropna(subset=["yrec"])
        for an, h in G.groupby("animal"):
            ld = np.log(h.day.to_numpy())
            Z = np.column_stack([np.ones(len(h)), ld])
            ry = h.yrec.to_numpy() - Z @ np.linalg.lstsq(Z, h.yrec.to_numpy(), rcond=None)[0]
            for c in cols:
                x = h[c].to_numpy(float)
                if np.isfinite(x).sum() < 6:
                    continue
                okk = np.isfinite(x)
                rx = x[okk] - Z[okk] @ np.linalg.lstsq(Z[okk], x[okk], rcond=None)[0]
                W.append({"set": st, "animal": an, "feature": c, "partial_r": np.corrcoef(rx, ry[okk])[0, 1]})
    W = pd.DataFrame(W)
    Wm = W.groupby(["set", "feature"]).partial_r.agg(["mean", lambda s: int((np.sign(s) == np.sign(s.mean())).sum())])
    Wm.columns = ["mean_partial_r", "n_same_sign"]
    Wm = Wm.reset_index()
    W.to_csv(il.OUT_W / f"{TAG}_where.csv", index=False)
    print(f"== B. features most associated with {WHAT} within animals (partial r, log day removed)\n"
          + Wm.reindex(Wm.mean_partial_r.abs().sort_values(ascending=False).index).groupby("set").head(10)
          .round(2).to_string(index=False))
    # C: learned direction, nested ridge, LOAO
    rng = np.random.default_rng(0)
    L = []
    for st, G in S[(S.epoch != "pre") & (S.day > 0)].groupby("set"):
        G = G.dropna(subset=["yrec"]).copy()
        X = G[cols].to_numpy(float)
        X = np.where(np.isfinite(X), X, np.nanmean(X, 0))
        y = G.yrec.to_numpy()
        grp = G.animal.to_numpy()
        ep = G.epoch.to_numpy()
        r_loao, pred = _ridge_cv(X, y, grp)
        ld = np.log(G.day.to_numpy())[:, None]
        r_days = 1 - np.sum((y - _ols_cv(ld, y, grp)) ** 2) / np.sum((y - y.mean()) ** 2)
        null = []
        for _ in range(N_SHUF // 5):                       # nested ridge is costlier: 100 shuffles
            Xs = X.copy()
            for a in np.unique(grp):
                ix = np.flatnonzero(grp == a)
                Xs[ix] = X[rng.permutation(ix)]
            null.append(_ridge_cv(Xs, y, grp)[0])
        null = np.asarray(null)
        mae = {a: float(np.mean(np.abs(y[grp == a] - pred[grp == a]))) for a in np.unique(grp)}
        L.append({"set": st, "n": len(y), "n_features": X.shape[1], "loao": r_loao, "loao_days": r_days,
                  "null95": float(np.quantile(null, 0.95)), "p_shuffle": float((1 + np.sum(null >= r_loao)) / (1 + len(null))),
                  **{f"mae_{a}": v for a, v in mae.items()}})
    L = pd.DataFrame(L)
    L.to_csv(il.OUT_W / f"{TAG}_learned.csv", index=False)
    print("== C. learned direction (nested ridge, all features)\n" + L.round(3).to_string(index=False))
    return {"O": O, "FA": FA, "W": W, "Wm": Wm, "L": L, "S": S}


def _ols_cv(X, y, grp):
    pred = np.full(len(y), np.nan)
    for g in np.unique(grp):
        te = grp == g
        pred[te] = lrm._ols_pred(X[~te], y[~te], X[te])
    return pred


def _ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    ym = y.mean()
    b = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return lambda Xn: ym + ((Xn - mu) / sd) @ b


def _ridge_cv(X, y, grp, inner=None):
    """Outer leave-one-`grp`-out; ridge penalty chosen by inner leave-one-group-out (inner groups = `inner` or `grp`)."""
    inner = grp if inner is None else inner
    pred = np.full(len(y), np.nan)
    for g in np.unique(grp):
        te = grp == g
        Xtr, ytr, itr = X[~te], y[~te], inner[~te]
        best, best_a = np.inf, ALPHAS[0]
        for a in ALPHAS:
            err = 0.0
            for h in np.unique(itr):
                t2 = itr == h
                if t2.all():
                    continue
                err += np.sum((ytr[t2] - _ridge_fit(Xtr[~t2], ytr[~t2], a)(Xtr[t2])) ** 2)
            if err < best:
                best, best_a = err, a
        pred[te] = _ridge_fit(Xtr, ytr, best_a)(X[te])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), pred


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from frozen_area_maps import outline, paint
    from iti_area_maps import split_image
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    O, FA, Wm, L = F["O"], F["FA"], F["Wm"], F["L"]
    img, nm = split_image(il.labels()[0])
    ol = outline(img)
    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.25, 0.75])
    for i, st in enumerate(SETS.values()):
        # A: offset vs iti_ang, all vs inside / central
        ax = fig.add_subplot(gs[i, 0])
        g = O[O.set == st]
        pre = g[g.epoch == "pre"]
        ax.plot(pre.off_all, pre.yrec, "o", color="0.82", mec="0.6", mew=0.4, ms=6)
        for (ep, a_), h in g[g.epoch != "pre"].groupby(["epoch", "animal"]):
            ax.plot(h.off_all, h.yrec, ANIMAL_MARKER.get(a_, "o"), color=EPOCH_LINE[ep], ms=6, mec="k", mew=0.3)
        an = FA["AN"][(FA["AN"].set == st)].pivot_table(index="pred", columns="animal", values="r")
        ttl = f"{st}: decoder offset vs {WHAT}"
        for pr, lab in (("off_all", "all licks"), ("off_inside", "inside pre range"), ("off_central", "central 10-90%")):
            if pr in an.index:
                m = FA["M"][(FA["M"].set == st) & (FA["M"].pred == pr)]
                c = FA["CV"][(FA["CV"].set == st) & (FA["CV"].pred == pr)]
                ttl += (f"\n{lab}: r " + "/".join(f"{an.loc[pr, a]:+.2f}" for a in AN if a in an.columns)
                        + (f", p {m.p.iloc[0]:.2g}" if len(m) and "p" in m else "")
                        + (f", LOAO {c.loao.iloc[0]:.2f}" if len(c) else ""))
        ax.set_title(ttl, fontsize=7.5)
        ax.set_xlabel("session like-for-like offset (mm, all licks)", fontsize=8)
        ax.set_ylabel(YLAB, fontsize=8)
        # B: map of within-animal partial r
        ax = fig.add_subplot(gs[i, 1])
        w = Wm[Wm.set == st].set_index("feature")
        vals = w.mean_partial_r.to_dict()
        lim = max(0.3, float(np.nanmax(np.abs(list(vals.values())))))
        im = ax.imshow(paint(img, nm, vals), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
        for k_, row in w.iterrows():
            if row.n_same_sign == 4 and k_ in nm:
                yy, xx = np.nonzero(img == nm.index(k_))
                if len(yy):
                    ax.text(xx.mean(), yy.mean(), "4/4", ha="center", va="center", fontsize=6)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(f"{st}: where does activity at the same lick angle track {WHAT}? within-animal partial r "
                     "with it\n(log day removed; mean of animals; '4/4' = same sign in all animals; + = more activity, "
                     "more contralesional licking)", fontsize=8)
        # C: learned direction
        ax = fig.add_subplot(gs[i, 2])
        lr = L[L.set == st]
        if len(lr):
            r = lr.iloc[0]
            ax.bar([0, 1], [r.loao_days, r.loao], color=["0.55", "#8172B3"])
            ax.plot([1], [r.null95], "k_", ms=18, mew=1.5)
            ax.set_xticks([0, 1], ["days\n(LOAO)", "learned\n(LOAO)"], fontsize=8)
            ax.set_title(f"{st}: learned recovery direction, ridge on {int(r.n_features)} features\nLOAO {r.loao:.2f} "
                         f"(shuffle p {r.p_shuffle:.2g}; bar = null 95%)", fontsize=8.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1, 1)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    h = [Line2D([], [], color="0.82", marker="o", lw=0, label="pre")]
    h += [Line2D([], [], color=EPOCH_LINE[e], marker="o", lw=0, label=e) for e in ("acute", "subacute", "chronic")]
    fig.legend(handles=h, loc="lower left", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"What tracks {WHAT} ({YLAB})?\nA: the decoder offset, with a "
                 "coverage control; B: which area x hemisphere, at the same lick angle & order, tracks it within animals "
                 "beyond time; C: a learned multivariate direction, tested on unseen animals. Image left = ipsilesional. "
                 "PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_png_svg(fig, il.OUT_W / "figures" / f"{FIGN}.png", dpi=90)
    plt.close(fig)


def trial_set_sentence() -> str:
    """The licks of this driver's figure (Priya 11-09; MODE from WFIELD_RECSEARCH_SET): ITI = every ITI lick of the KEY
    decoder cache + its first licks after >= il.GAP_MAIN s quiet; cued = every post-cue lick + the first post-cue lick."""
    from wfield_local import trial_set_text as tst
    extra = ["the KEY decoder cache's licks", "3-D lateral landing, angle and features present",
             "coverage control rows: all licks / inside the animal's pre landing range / its central 10-90 %"]
    if CUED:
        cap = (f"iti_licks.extract post-cue set: every trial, no engagement / camera-class gate, at most "
               f"{int(il.N_TRIAL_MAX)} per session (random draw, seed 0)")
        return tst.cued_licks(extra=[cap, *extra, f"subset [{lrm.C1SET}]: lick order 0 (first post-cue lick)"])
    return tst.iti_licks(filt=lo.ITI_FILTER, extra=[*extra, f"subset [{lrm.QSET}]: no licks_all onset (any context) "
                                                             f"in the preceding {il.GAP_MAIN:g} s"])


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())
    if plot_only():
        F = load_fd(FIGN)
        for k in ("O", "S"):                       # figdata saved before the target column was renamed
            F[k] = F[k].rename(columns={"iti_ang": "yrec"})
        plot(F)
        return 0
    F = compute()
    save_fd(FIGN, F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
