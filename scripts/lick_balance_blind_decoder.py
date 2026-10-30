"""BALANCE-BLIND frozen lateral lick decoders (Priya 10-28: is the post-stroke like-for-like offset just a hemispheric
BALANCE shift that a lateralised decoder misreads, or is something patterned beyond balance changing?).

Same licks, same KEY-row decoder recipe as `iti_lick_order.predictions` (Allen areas + ALM / tjM1 split, lags 0..0.2 s
mean, divided by the session's frame-level SD, StandardScaler + RidgeCV(alphas 1e-1..1e5) fitted on the animal's PRE
licks of the set, pre sessions leave-one-session-out, post licks scored by the all-pre fit), but the per-lick feature
vector is first made BLIND to the hemispheric balance:
  orig        unchanged (must reproduce the cached lick_order_preds exactly -- checked)
  unif        UNIFORM BALANCE REMOVED: b = mean over the 21 matched pairs of (X_right - X_left) / 2; b subtracted from the
              paired _right features, added to the paired _left ones. sum_k (R_k - L_k) is then identically 0, so no
              linear read-out (after any per-feature rescaling) can see the uniform lateral difference; pair sums and the
              pattern of pair differences around their mean are kept. The 3 unpaired areas (VISp / VISal / VISl right;
              no left ROI) are left untouched: subtracting b from them would WRITE the balance into them (V - b).
  hemi        HEMISPHERE-NORMALISED: each hemisphere's own per-lick mean over its areas subtracted from its areas (removes
              each hemisphere's overall level, hence also the uniform balance; keeps within-hemisphere patterns).
  sensitivity unif_paired = unif with the 3 unpaired areas DROPPED (the only leak left in unif: an unpaired right area
              against the pair sums is a crude balance proxy); unif_raw = unif applied in dF/F before the SD scaling.
  regional    (follow-up 10-28) oro = the 4 orofacial pairs (ALM, tjM1, SSp-m, SSp-n) set to their per-lick (R+L)/2, so
              their left-right difference is zero, all else untouched; oro_mo = + MOs_rest / MOp_rest pairs; oro_raw =
              oro in dF/F; allpair = EVERY matched pair set to its mean (only hemispheric sums + the 3 unpaired areas
              left); nonoro = the complement of oro (orofacial pairs keep their difference, all other pairs zeroed).
  subsets     no transform, feature subset only: sub_visrsp = every VIS* / RSP* feature (both sides, incl. unpaired
              VISp / VISal / VISl right); sub_orofac = ALM, tjM1, MOs_rest, MOp_rest, SSp-m, SSp-n, both sides.
WHERE: after the per-session frame-SD scaling, before the StandardScaler. The SD-scaled features ARE the decoder's input,
so "uniform" is defined in the units its weights act on (an area with a large dF/F swing does not dominate b). The
pre-fit StandardScaler is a per-feature rescale and cannot reintroduce a direction that is identically zero.
Score: like-for-like offset (`likeforlike.offset`, Gaussian reference on landing, sigma by held-out pre error) per lick;
session mean over all licks (>= MIN); change vs pre per epoch, per animal and pooled (`analysis_kit.boot_delta`, as
`iti_lick_order.tests`). Decoding quality: r(decoded, actual) on held-out (LOSO) pre licks, per animal.
Outputs <OUT_W>/balblind_{preds.csv.gz,sessions,tests,quality}.csv, <OUT_W>/figures/balblind_offset.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_first_lick as fl  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402
import likeforlike as lfl  # noqa: E402

AN, EP, EPS, SETS, POS, MIN = lo.AN, lo.EP, lo.EPS, lo.SETS, lo.POS, lo.MIN
OROF = ["ALM", "tjM1", "SSp-m", "SSp-n"]
OROF_MO = [*OROF, "MOs_rest", "MOp_rest"]
VARIANTS = ["orig", "unif", "hemi", "unif_paired", "unif_raw", "oro", "oro_mo", "oro_raw", "allpair", "nonoro", "sub_visrsp", "sub_orofac"]
SUBS = ["orig", "sub_visrsp", "sub_orofac"]
MAIN = ["orig", "unif", "hemi", "oro", "allpair", "nonoro"]
VLAB = {"orig": "original", "unif": "uniform balance removed", "hemi": "hemisphere-normalised",
        "unif_paired": "uniform removed, unpaired dropped", "unif_raw": "uniform removed in dF/F",
        "oro": "orofacial pair diffs zeroed", "oro_mo": "orofacial + MOs/MOp_rest diffs zeroed",
        "oro_raw": "orofacial diffs zeroed in dF/F", "allpair": "ALL pair diffs zeroed", "nonoro": "non-orofacial pair diffs zeroed",
        "sub_visrsp": "VIS + RSP areas only", "sub_orofac": "orofacial areas only (ALM, tjM1, MOs/MOp_rest, SSp-m/n)"}
VCOL = {"orig": "#444444", "unif": "#2c7fb8", "hemi": "#d95f02", "oro": "#7b3294", "allpair": "#1b9e77", "nonoro": "#e7298a", "sub_visrsp": "#66a61e", "sub_orofac": "#a6761d"}
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}


def _order_one(x: str):
    """One session -> (iti bout index, cued lick_idx) for its ITI / cued licks (module-level for fan_sessions)."""
    z = il._load(x)
    ti, tt = z["iti_t"].astype(float), z["tr_t"].astype(float)
    return {"iti": lo._bout_idx(x, ti), "trial": fl._lick_idx(x, tt), "iti_t": ti}


def _pairs(names):
    R = {n[:-6]: i for i, n in enumerate(names) if n.endswith("_right")}
    L = {n[:-5]: i for i, n in enumerate(names) if n.endswith("_left")}
    com = sorted(set(R) & set(L))
    return (np.array([R[k] for k in com]), np.array([L[k] for k in com]),
            np.array(sorted(R.values())), np.array(sorted(L.values())), com)


def transform(X: np.ndarray, names, how: str) -> np.ndarray:
    """Per-lick balance-blind transform of features X (n, f) in `names` order (module docstring)."""
    if how == "orig":
        return X
    ir, il_, rall, lall, _ = _pairs(names)
    X = X.copy()
    if how in ("unif", "unif_paired", "unif_raw"):
        b = ((X[:, ir] - X[:, il_]) / 2).mean(1, keepdims=True)
        X[:, ir] -= b
        X[:, il_] += b
        if how == "unif_paired":
            X = X[:, np.sort(np.r_[ir, il_])]
        return X
    if how.startswith("sub_"):
        keep = [i for i, n in enumerate(names) if (n.startswith(("VIS", "RSP")) if how == "sub_visrsp"
                                                   else n.rsplit("_", 1)[0] in OROF_MO)]
        return X[:, keep]
    if how in ("oro", "oro_mo", "oro_raw", "allpair", "nonoro"):
        _ir, _il, _r, _l, com = _pairs(names)
        sel = {"oro": OROF, "oro_raw": OROF, "oro_mo": OROF_MO}.get(how)
        keep = [k in (sel or OROF) for k in com]
        use = np.array(keep) if sel else (np.ones(len(com), bool) if how == "allpair" else ~np.array(keep))
        m = (X[:, ir[use]] + X[:, il_[use]]) / 2
        X[:, ir[use]] = m
        X[:, il_[use]] = m
        return X
    if how == "hemi":
        X[:, rall] -= X[:, rall].mean(1, keepdims=True)
        X[:, lall] -= X[:, lall].mean(1, keepdims=True)
        return X
    raise ValueError(how)


def predictions() -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    from wfield_local import analysis_kit as ak
    from wfield_local import orofacial_cohort as oc
    D, names, SD = wd.area_basis(basis="split")
    ir, il_, rall, lall, com = _pairs(names)
    print(f"{len(names)} features; {len(com)} matched pairs; unpaired: "
          f"{[names[i] for i in np.setdiff1d(np.arange(len(names)), np.r_[ir, il_])]}", flush=True)
    res, fail = ak.fan_sessions(sorted(D), _order_one, jobs=8, label="lick order", log=lambda m: None)
    if fail:
        raise RuntimeError(f"order index failed for {[f[0] for f in fail][:5]}: {fail[0][1]}")
    ORD = dict(res)
    rows = []
    for st in SETS:
        for a in AN:
            S = {}
            for x, d in sorted(D.items()):
                if d["animal"] != a:
                    continue
                X, C = d[st]
                t = ORD[x]["iti_t"] if st == "iti" else il._load(x)["tr_t"].astype(float)
                idx = ORD[x][st]
                pos = np.array([q if q in POS or "_" not in q else oc.pos_key(q, "L") for q in d[f"{st}_pos"].astype(str)])
                ok = np.isfinite(C[:, 0]) & np.isfinite(idx) & np.isin(pos, POS)
                if st == "iti":
                    ok &= lo.iti_keep(x, t)
                if ok.sum() >= 20:
                    sd = SD.loc[x, names].to_numpy(float)
                    S[x] = (d["epoch"], X[ok], sd, C[ok, 0], idx[ok], pos[ok])
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            for how in VARIANTS:
                F = {}
                for x, (_ep, Xr, sd, _y, _idx, _pos) in S.items():
                    F[x] = (transform(Xr, names, how[:-4]) / sd[None, :] if how.endswith("_raw")
                            else transform(Xr / sd[None, :], names, how))
                fits = {}
                for held in [*pre, None]:
                    trn = [q for q in pre if q != held]
                    Xt, yt = np.vstack([F[q] for q in trn]), np.concatenate([S[q][3] for q in trn])
                    sc = StandardScaler().fit(Xt)
                    fits[held] = (sc, RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xt), yt))
                for x, (ep, _Xr, _sd, y, idx, pos) in S.items():
                    sc, m = fits[x if x in pre else None]
                    rows.append(pd.DataFrame({"variant": how, "set": st, "animal": a, "label": x, "epoch": ep,
                                              "pos": pos, "order": idx, "act": y, "dec": m.predict(sc.transform(F[x])),
                                              "alpha": m.alpha_ if x not in pre else fits[x][1].alpha_}))
            print(f"  {st} {a}: {len(S)} sessions ({len(pre)} pre)", flush=True)
    return pd.concat(rows, ignore_index=True)


def score(P: pd.DataFrame):
    """Per variant: like-for-like offset per lick, session means, change vs pre (boot_delta), pre held-out r."""
    from wfield_local import analysis_kit as ak
    sess, tests, qual = [], [], []
    for how, g in P.groupby("variant", sort=False):
        g = g.reset_index(drop=True)
        g["lfl"], _out = lfl.offset(g, "act", "dec", lfl.SIGMA_MM, f"lateral (mm) [{how}]")
        S = (g.groupby(["set", "animal", "label", "epoch"]).lfl
             .agg(lambda s: s.mean() if s.notna().sum() >= MIN else np.nan).rename("lfl").reset_index())
        S["variant"] = how
        sess.append(S)
        for (st, a), h in g[g.epoch == "pre"].groupby(["set", "animal"]):
            qual.append(dict(variant=how, set=st, animal=a, r_pre=float(np.corrcoef(h.act, h.dec)[0, 1]),
                             r_pre_within_session=float(np.mean([np.corrcoef(q.act, q.dec)[0, 1]
                                                                 for _l, q in h.groupby("label")])),
                             n=len(h), alpha_median=float(h.alpha.median())))
        for st, gs in S.groupby("set"):
            prd = {a: h.lfl.dropna().to_numpy() for a, h in gs[gs.epoch == "pre"].groupby("animal")}
            prd = {a: v for a, v in prd.items() if len(v)}
            pm = gs[gs.epoch == "pre"].groupby("animal").lfl.mean()
            for ep in EP:
                post = {a: h.lfl.dropna().to_numpy() for a, h in gs[gs.epoch == ep].groupby("animal") if a in prd}
                post = {a: v for a, v in post.items() if len(v)}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0)) if post else None
                am = (gs[gs.epoch == ep].groupby("animal").lfl.mean() - pm).dropna()
                tests.append(dict(variant=how, set=st, epoch=ep, point=getattr(ci, "point", np.nan),
                                  lo=getattr(ci, "lo", np.nan), hi=getattr(ci, "hi", np.nan),
                                  n_pos=int((am > 0).sum()), n=len(am), **{f"animal_{k}": v for k, v in am.items()}))
    return pd.concat(sess, ignore_index=True), pd.DataFrame(tests), pd.DataFrame(qual)


def plot(T: pd.DataFrame, Q: pd.DataFrame, MAIN=MAIN, name="balblind_offset", title="refit on balance-blind features") -> Path:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, st in zip(axs, SETS):
        for j, how in enumerate(MAIN):
            r = Q[(Q.variant == how) & (Q.set == st)].r_pre.mean()
            for i, ep in enumerate(EP):
                row = T[(T.variant == how) & (T.set == st) & (T.epoch == ep)]
                if row.empty:
                    continue
                row = row.iloc[0]
                xc = i + (j - (len(MAIN) - 1) / 2) * 0.14
                ax.errorbar(xc, row.point, yerr=[[row.point - row.lo], [row.hi - row.point]], fmt="o", ms=7,
                            color=VCOL[how], capsize=3, label=f"{VLAB[how]} (pre r {r:.2f})" if i == 0 else None, zorder=3)
                for a in AN:
                    v = row.get(f"animal_{a}", np.nan)
                    if np.isfinite(v):
                        ax.scatter(xc + 0.04, v, marker=MARK[a], s=12, facecolor="none", edgecolor=VCOL[how], lw=0.8,
                                   zorder=2)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(len(EP)), EP)
        ax.set_title(f"{SETS[st]} licks")
        ax.legend(fontsize=7, loc="upper right", frameon=False)
    axs[0].set_ylabel("like-for-like offset change vs pre (mm)\n+ = read more contralesional")
    for a, mk in MARK.items():
        axs[1].scatter([], [], marker=mk, facecolor="none", edgecolor="0.3", label=a)
    axs[1].legend(fontsize=7, loc="upper right", frameon=False)
    fig.suptitle(f"Frozen pre-stroke lateral decoder {title} (NEURAL): mean of animals, 95% nested "
                 "bootstrap CI; open symbols = animals", fontsize=9)
    fig.tight_layout()
    out = il.OUT_W / "figures" / f"{name}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    return out


def main() -> int:
    pd.set_option("display.width", 240)
    fp = il.OUT_W / "balblind_preds.csv.gz"
    if il.ITI_SUBSET:                # subset run (dock, ...; 10-29): the cached refits of the full set, ITI rows subset
        P = pd.read_csv(fp if fp.exists() else il.OUT / "balblind_preds.csv.gz")
        P = pd.concat([il.subset_preds(g.reset_index(drop=True), v) for v, g in P.groupby("variant", sort=False)],
                      ignore_index=True)
    elif fp.exists() and "--rerun" not in sys.argv:
        P = pd.read_csv(fp)
    else:
        P = predictions()
        P.to_csv(fp, index=False)
    # the orig variant must reproduce the cached KEY-row predictions exactly
    C = il.subset_preds(pd.read_csv(lo.preds_path()))
    O = P[P.variant == "orig"].reset_index(drop=True)
    same = len(C) == len(O) and np.allclose(C.dec.to_numpy(), O.dec.to_numpy(), atol=1e-8)
    print(f"orig vs cached {lo.preds_path().name}: n {len(O)} / {len(C)}, identical decoded: {same}"
          + ("" if same else f" (max |diff| {np.nanmax(np.abs(C.dec.to_numpy() - O.dec.to_numpy())) if len(C) == len(O) else 'n/a'})"))
    S, T, Q = score(P)
    for n, df in (("sessions", S), ("tests", T), ("quality", Q)):
        df.to_csv(il.OUT_W / f"balblind_{n}.csv", index=False)
    print("\n== PER ANIMAL (rule 8): like-for-like offset change vs pre (mm), animal mean of session means")
    acols = [f"animal_{a}" for a in AN]
    print(T[["set", "variant", "epoch", *acols]].pivot_table(index=["set", "variant"], columns="epoch", values=acols)
          .round(3).to_string())
    print("\n== pre held-out r(decoded, actual) per animal (LOSO pre licks)")
    print(Q.pivot_table(index=["set", "variant"], columns="animal", values="r_pre").round(3).to_string())
    print(Q.pivot_table(index=["set", "variant"], columns="animal", values="r_pre_within_session").round(3).to_string())
    print("\n== POOLED: nested bootstrap (boot_delta), mean of animals")
    T2 = T.merge(Q.groupby(["variant", "set"]).r_pre.mean().rename("r_pre_mean").reset_index(), on=["variant", "set"])
    print(T2[["set", "epoch", "variant", "point", "lo", "hi", "n_pos", "n", "r_pre_mean"]].round(3).to_string(index=False))
    print("figure:", plot(T, Q))
    print("figure:", plot(T, Q, SUBS, "balblind_subsets", "refit on AREA SUBSETS"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
