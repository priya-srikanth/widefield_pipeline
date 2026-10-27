"""LICK ORDER and the lateral read-out, ITI and cued (Priya 10-27: "first-lick ITI, first-lick post-cue, subsequent bout
licks ITI, subsequent post-cue -- or a correlation of lick order and offset?"). Extends `iti_first_lick.py` (cued only).

Order index per lick:
  cued  `lick_idx` of the 3-D lick table (0 = first post-cue lick of the trial; joined on the exact DAQ peak time)
  ITI   index within its lick BOUT: all of the session's licks (licks_all.csv onsets, every context) split at gaps
        >= BOUT_GAP (1 s, the extraction's bout rule); 0 = the bout's first lick. An ITI lick that continues a cued bout
        past the response window therefore has index > 0 -- it is a subsequent lick, not a self-initiated launch.
Decoder: the KEY-row lateral where-decoder per set (Allen areas, lags 0..0.2 s, features z-scored within session with
zscore_check/session_sd.csv, ridge, trained on that set's August pre licks, pre sessions leave-one-session-out). Per
lick: landing = actual - the animal's pre mean at the spout (ITI: the previous trial's spout); offset beyond the pull =
decoded - (a + b x actual), a, b = the animal's pre fit for the set. + = contralesional.
Two read-outs:
  (1) four classes (ITI first / ITI later / cued first / cued later): session means, change vs pre (paired nested
      bootstrap), and first - later per session.
  (2) lick-order slope: per session, OLS slope of the value on min(order, ORDER_CAP), spout means removed (mm per lick);
      change vs pre. Both for all spouts and far contra.
Outputs <E>/iti_licks/lick_order_{preds.csv.gz,sessions.csv,tests.csv}, figures/lick_order_iti_cued{,_slope}.png.
Usage: python scripts/iti_lick_order.py [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_first_lick as fl  # noqa: E402
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
EPS = ["pre", *EP]
POS = fl.POS
ORDER_CAP = 9
MIN = 8
SETS = {"iti": "ITI", "trial": "cued"}
# DECODER SOURCE (Priya 10-27: "let's use the SVD basis"; Allen / LocaNMF = basis-robustness checks): env WFIELD_LICK_DECODER = "areas" (Allen areas, z-scored by
# zscore_check/session_sd.csv; the KEY row until 10-27) or "svd" (the animal's pre-stroke 200-component SVD basis,
# iti_basis_decoder extraction; each component z-scored by its SD over that session's licks, ITI + cued together)
import os  # noqa: E402

SOURCE = os.environ.get("WFIELD_LICK_DECODER", "areas")
SRC_TAG = "" if SOURCE == "areas" else f"_{SOURCE}"


def preds_path() -> Path:
    return il.OUT / f"lick_order_preds{SRC_TAG}.csv.gz"
VALS = ("landing", "decoded", "beyond")
VLAB = {"landing": "landing vs pre at the spout (BEHAVIOUR)", "decoded": "decoded position vs pre at the spout (NEURAL)",
        "beyond": "offset beyond the pull (NEURAL)"}
VCOL = {"landing": "0.45", "decoded": "#2c7fb8", "beyond": "#7b3294"}


def _bout_idx(label: str, t: np.ndarray) -> np.ndarray:
    """Index of each lick (exact DAQ peak time t) within its bout of the session's licks (gap >= BOUT_GAP)."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_on_s", "daq_peak_s"])
    A = A[np.isfinite(A.daq_on_s)].sort_values("daq_on_s").reset_index(drop=True)
    on = A.daq_on_s.to_numpy(float)
    start = np.r_[True, np.diff(on) >= il.BOUT_GAP]
    bid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    idx = np.arange(len(on)) - first[bid]
    m = pd.Series(idx, index=A.daq_peak_s.to_numpy(float))
    m = m[~m.index.duplicated()]
    return m.reindex(np.asarray(t, float)).to_numpy(float)


def predictions() -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    from iti_decoder_contrib import feature_names
    from wfield_local import orofacial_cohort as oc
    SD = pd.read_csv(il.OUT / "zscore_check" / "session_sd.csv", index_col=0)
    names = feature_names()
    D = wd.load({"areas": "areas", "svd": "svdbasis", "svdraw": "svdbasis", "locanmf": "locanmf"}[SOURCE])   # basis robustness (Priya 10-27)
    sd_svd = {x: np.vstack([d["iti"][0], d["trial"][0]]).std(0) for x, d in D.items()} if SOURCE != "areas" else {}
    rows = []
    for st in SETS:
        for a in AN:
            S = {}
            for x, d in sorted(D.items()):
                if d["animal"] != a:
                    continue
                X, C = d[st]
                t = il._load(x)["iti_t" if st == "iti" else "tr_t"].astype(float)
                idx = _bout_idx(x, t) if st == "iti" else fl._lick_idx(x, t)
                pos = np.array([q if q in POS or "_" not in q else oc.pos_key(q, "L") for q in d[f"{st}_pos"].astype(str)])
                ok = np.isfinite(C[:, 0]) & np.isfinite(idx) & np.isin(pos, POS)
                if ok.sum() >= 20:
                    # svdraw = SVD basis UNSCALED (check: lick-based z-scoring partly divides out the larger post-lick response)
                    sd = (SD.loc[x, names].to_numpy(float) if SOURCE == "areas" else np.ones(X.shape[1]) if SOURCE == "svdraw"
                          else np.where(sd_svd[x] > 0, sd_svd[x], 1))
                    S[x] = (d["epoch"], X[ok] / sd[None, :], C[ok, 0], idx[ok], pos[ok])
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            fits = {}
            for held in [*pre, None]:
                trn = [q for q in pre if q != held]
                Xt, yt = np.vstack([S[q][1] for q in trn]), np.concatenate([S[q][2] for q in trn])
                sc = StandardScaler().fit(Xt)
                fits[held] = (sc, RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xt), yt))
            for x, (ep, X, y, idx, pos) in S.items():
                sc, m = fits[x if x in pre else None]
                rows.append(pd.DataFrame({"set": st, "animal": a, "label": x, "epoch": ep, "pos": pos, "order": idx,
                                          "act": y, "dec": m.predict(sc.transform(X))}))
    return pd.concat(rows, ignore_index=True)


def per_lick(P: pd.DataFrame) -> pd.DataFrame:
    """landing (vs the animal's pre mean at the spout) and offset beyond the pull, per lick."""
    out = []
    for (_st, _a), g in P.groupby(["set", "animal"]):
        g = g.copy()
        pre = g[g.epoch == "pre"]
        b, c = np.polyfit(pre.act, pre.dec, 1)
        g["landing"] = g.act - g.pos.map(pre.groupby("pos").act.mean())
        g["beyond"] = g.dec - (c + b * g.act)
        g["decoded"] = g.dec - g.pos.map(pre.groupby("pos").dec.mean())     # read-out vs the animal's pre at the spout
        out.append(g)
    P = pd.concat(out, ignore_index=True)
    P["cls"] = np.where(P.order == 0, "first", "later")
    P["ord_c"] = np.minimum(P.order, ORDER_CAP)
    return P


def sessions(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, g0 in (("all spouts", P), ("far contra", P[P.pos == "far_contra"])):
        for (st, a, lab, ep), h in g0.groupby(["set", "animal", "label", "epoch"]):
            r = dict(scope=scope, set=st, animal=a, label=lab, epoch=ep)
            for cls in ("first", "later"):
                q = h[h.cls == cls]
                for v in VALS:
                    r[f"{v}_{cls}"] = q[v].mean() if len(q) >= MIN else np.nan
                r[f"n_{cls}"] = len(q)
            for v in VALS:
                r[f"{v}_diff"] = r[f"{v}_first"] - r[f"{v}_later"]
                if len(h) >= 2 * MIN and h.ord_c.nunique() > 2:
                    yv = h[v] - h.groupby("pos")[v].transform("mean")
                    xv = h.ord_c - h.groupby("pos").ord_c.transform("mean")
                    r[f"{v}_slope"] = float((xv * yv).sum() / (xv ** 2).sum()) if (xv ** 2).sum() > 0 else np.nan
                else:
                    r[f"{v}_slope"] = np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def tests(S: pd.DataFrame) -> pd.DataFrame:
    out = []
    cols = [f"{v}_{k}" for v in VALS for k in ("first", "later", "diff", "slope")]
    for (scope, st), g in S.groupby(["scope", "set"]):
        for col in cols:
            prd = {a: h[col].dropna().to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            prd = {a: v for a, v in prd.items() if len(v)}
            pm = g[g.epoch == "pre"].groupby("animal")[col].mean()
            out.append(dict(scope=scope, set=st, measure=col, epoch="pre", level=g[g.epoch == "pre"][col].mean(),
                            point=np.nan, lo=np.nan, hi=np.nan, n_dir=np.nan, n=len(pm.dropna())))
            for ep in EP:
                post = {a: h[col].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in prd}
                post = {a: v for a, v in post.items() if len(v)}
                if not post:
                    continue
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                am = (g[g.epoch == ep].groupby("animal")[col].mean() - pm).dropna()
                out.append(dict(scope=scope, set=st, measure=col, epoch=ep, level=g[g.epoch == ep][col].mean(),
                                point=ci.point, lo=ci.lo, hi=ci.hi, n_dir=int((am * np.sign(ci.point) > 0).sum()),
                                n=len(am), **_lmm(g, col, ep)))
    return pd.DataFrame(out)


_LMM = {}


def _lmm(g: pd.DataFrame, col: str, ep: str) -> dict:
    """Epoch-model mixed fit (wfield_local.mixed_models, the shared definition: y ~ C(epoch), sessions as observations,
    random-effects ladder per animal) for one session table x measure; cached per (table, measure)."""
    from wfield_local import mixed_models as mx
    key = (g.scope.iloc[0], g.set.iloc[0], col)          # one session table per scope x set
    if key not in _LMM:
        L = mx.add_time_covariates(g[["label", col]].rename(columns={col: "y"}).dropna())
        T = mx.lmm_table(L.assign(grp=0), ["grp"], models=("epoch",), print_animals=False)
        _LMM[key] = {} if T.empty else {t.split("T.")[-1].rstrip("]"): (c, p_) for t, c, p_ in zip(T.term, T.coef, T.p)}
    c, p_ = _LMM[key].get(ep, (np.nan, np.nan))
    return {"mixed_coef": c, "mixed_p": p_}


def _star(r) -> str:
    """* = bootstrap CI excludes 0; + dagger = mixed model p < 0.05 (Priya 10-27)."""
    b = "*" if np.isfinite(r.lo) and (r.lo > 0 or r.hi < 0) else ""
    return b + ("†" if np.isfinite(r.get("mixed_p", np.nan)) and r.mixed_p < 0.05 else "")


def plot_classes(T: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    CL = {("iti", "first"): "#1f6fb4", ("iti", "later"): "#9cc3e6", ("trial", "first"): "#c0392b",
          ("trial", "later"): "#eba59d"}
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.8), squeeze=False)
    for i, scope in enumerate(("all spouts", "far contra")):
        for j, (val, lab) in enumerate((("landing", "landing (BEHAVIOUR)"), ("beyond", "offset beyond the pull (NEURAL)"),
                                        ("beyond_diff", "first - later, beyond the pull (NEURAL)"))):
            ax = axs[i, j]
            keys = [("iti", "diff"), ("trial", "diff")] if val.endswith("diff") else list(CL)
            for k, (st, cls) in enumerate(keys):
                m = val if val.endswith("diff") else f"{val}_{cls}"
                t = T[(T.scope == scope) & (T.set == st) & (T.measure == m)].set_index("epoch").reindex(EP)
                x = np.arange(1, 4) + (k - (len(keys) - 1) / 2) * 0.16
                col = CL[(st, "first")] if cls == "diff" else CL[(st, cls)]
                ax.errorbar(x, t.point, yerr=[t.point - t.lo, t.hi - t.point], fmt="o", color=col, capsize=2.5, ms=5)
                for xx, (_e, r) in zip(x, t.iterrows()):
                    if np.isfinite(r.point):
                        ax.text(xx, r.hi, f"{r.point:+.2f}{_star(r)}\n{int(r.n_dir)}/{int(r.n)}", ha="center",
                                va="bottom", fontsize=6, color=col, fontweight="bold" if "*" in _star(r) else "normal")
            ax.axhline(0, color="0.7", lw=0.6, ls=":")
            ax.set_xticks([1, 2, 3], EP)
            ax.set_title(f"{scope}: {lab}\nchange vs pre (mm)", fontsize=9)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    h = [Line2D([], [], color=CL[k], marker="o", lw=0, label=f"{SETS[k[0]]} {k[1]} lick" + ("s" if k[1] == "later" else ""))
         for k in CL]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("FIRST vs LATER licks, ITI (first = bout start, gap >= 1 s) and cued (first = first post-cue lick): "
                 "lateral, change vs pre,\nfrozen KEY-row decoder per set (z-scored Allen areas); 95% paired nested "
                 "bootstrap CI, * = CI excludes 0 (bold),\n\u2020 = mixed (LMM) p < 0.05 (second opinion, anti-conservative at n=4), k/n animals. + = contralesional. PROVISIONAL", fontsize=9.5)
    _layout(fig, bottom=0.05)
    fig.savefig(il.OUT / "figures" / f"lick_order_iti_cued{SRC_TAG}.png", dpi=100)
    plt.close(fig)


def _layout(fig, bottom: float) -> None:
    """tight_layout leaving room for the suptitle (0.2 in per line + 0.3 in) and a legend row at the bottom."""
    t = fig._suptitle.get_text() if fig._suptitle is not None else ""
    top = 1 - (0.2 * (t.count(chr(10)) + 1) + 0.3) / fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, bottom, 1, top))


def plot_slope(P: pd.DataFrame, T: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    fig, axs = plt.subplots(2, 5, figsize=(21, 8.6), squeeze=False)
    for i, st in enumerate(SETS):
        for j, val in enumerate(VALS):
            ax = axs[i, j]
            g = P[P.set == st]
            # per animal x epoch x order mean, then mean +/- SEM over animals
            A = g.groupby(["epoch", "animal", "ord_c"])[val].mean().reset_index()
            for ep in EPS:
                q = A[A.epoch == ep].groupby("ord_c")[val]
                mu, se = q.mean(), q.sem()
                ax.plot(mu.index, mu, "o-", color=col[ep], ms=3.5, lw=1.4)
                ax.fill_between(mu.index, mu - se, mu + se, color=col[ep], alpha=0.15, lw=0)
            ax.axhline(0, color="0.7", lw=0.6, ls=":")
            ax.set_xticks(range(ORDER_CAP + 1), [*map(str, range(ORDER_CAP)), f"{ORDER_CAP}+"])
            ax.set_xlabel("lick order (0 = " + ("bout start)" if st == "iti" else "first post-cue lick)"), fontsize=8)
            ax.set_ylabel(f"{val} (mm, + = contralesional)", fontsize=8)
            ax.set_title(f"{SETS[st]}: {VLAB[val]}\nmean +/- SEM over animals", fontsize=8.5)
        for j, scope in enumerate(("all spouts", "far contra")):
            ax = axs[i, 3 + j]
            for k, val in enumerate(VALS):
                t = T[(T.scope == scope) & (T.set == st) & (T.measure == f"{val}_slope")].set_index("epoch").reindex(EP)
                x = np.arange(1, 4) + (k - 1) * 0.22
                c = VCOL[val]
                ax.errorbar(x, t.point, yerr=[t.point - t.lo, t.hi - t.point], fmt="o", color=c, capsize=2.5)
                for xx, (_e, r) in zip(x, t.iterrows()):
                    if np.isfinite(r.point):
                        ax.text(xx, r.hi, f"{r.point:+.3f}{_star(r)}\n{int(r.n_dir)}/{int(r.n)}", ha="center",
                                va="bottom", fontsize=5.8, color=c, fontweight="bold" if "*" in _star(r) else "normal")
            pre = T[(T.scope == scope) & (T.set == st) & (T.epoch == "pre")].set_index("measure")
            ax.text(0.99, 0.01, "pre slopes (mm/lick): " + ", ".join(
                f"{v} {pre.level.get(f'{v}_slope', np.nan):+.3f}" for v in VALS), transform=ax.transAxes, fontsize=6.5,
                va="bottom", ha="right", color="0.3")
            ax.axhline(0, color="0.7", lw=0.6, ls=":")
            ax.set_xticks([1, 2, 3], EP)
            ax.set_xlim(0.5, 3.5)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))     # room for the pre text and the labels
            ax.set_ylabel("change in slope vs pre (mm per lick)", fontsize=8)
            ax.set_title(f"{SETS[st]}, {scope}: lick-order SLOPE\n(spout means removed), change vs pre", fontsize=8.5)
        for ax in axs[i]:
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    # two legends, each under the columns it belongs to
    fig.legend(handles=[Line2D([], [], color=col[e], lw=2, label=e) for e in EPS], loc="lower center",
               bbox_to_anchor=(0.30, 0.0), ncol=4, fontsize=8.5, frameon=False, title="columns 1-3: epoch", title_fontsize=8)
    fig.legend(handles=[Line2D([], [], color=VCOL[v], marker="o", lw=0, label=f"{v} slope") for v in VALS],
               loc="lower center", bbox_to_anchor=(0.80, 0.0), ncol=3, fontsize=8.5, frameon=False,
               title="columns 4-5: measure", title_fontsize=8)
    fig.suptitle("LICK ORDER vs lateral landing and read-out, ITI (row 1, order within the lick bout) and cued (row 2, "
                 "order after the cue); frozen KEY-row decoder per set.\nSlope = per-session OLS of the value on "
                 f"min(order, {ORDER_CAP}), spout means removed; 95% paired nested bootstrap CI of the change vs pre, "
                 "* = CI excludes 0 (bold), † = mixed (LMM) p < 0.05 (second opinion, anti-conservative at n=4), k/n "
                 "animals. PROVISIONAL", fontsize=9.5)
    _layout(fig, bottom=0.08)
    fig.savefig(il.OUT / "figures" / f"lick_order_iti_cued_slope{SRC_TAG}.png", dpi=100)
    plt.close(fig)


def main() -> int:
    f = preds_path()
    if f.exists():
        P = pd.read_csv(f)
    else:
        P = predictions()
        P.to_csv(f, index=False)
    P = per_lick(P)
    print("share first (order 0) by set x epoch:\n",
          P.groupby(["set", "epoch"]).order.apply(lambda s: (s == 0).mean()).round(3).unstack().to_string())
    S = sessions(P)
    S.to_csv(il.OUT / f"lick_order_sessions{SRC_TAG}.csv", index=False)
    pd.set_option("display.width", 240)
    print("== per animal (rule 8): beyond-pull offset, first / later / slope")
    for scope in ("all spouts", "far contra"):
        for m in ("beyond_first", "beyond_later", "beyond_slope"):
            print(f"-- {scope} {m}")
            print(S[S.scope == scope].pivot_table(index=["set", "animal"], columns="epoch", values=m)
                  .reindex(columns=EPS).round(3).to_string())
    T = tests(S)
    T.to_csv(il.OUT / f"lick_order_tests{SRC_TAG}.csv", index=False)
    print(T[T.epoch != "pre"].round(3).to_string(index=False))
    plot_classes(T)
    plot_slope(P, T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
