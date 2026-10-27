"""FIRST cued lick vs LATER licks: is the cortical read-out of the first lick (launched before any feedback) held further
toward the target than later licks? (handoff 10-27: "cued group = all post-cue licks; first-lick version = the cleaner
intention test").

The KEY-row lateral where-decoder (Allen areas, lags 0..0.2 s, features z-scored within session, trained on August pre
cued licks, pre leave-one-session-out) is refit here with each cued lick's within-trial index (`lick_idx` of the 3-D lick
table, 0 = first post-cue lick; joined on the exact DAQ peak time -> licks_all.csv row -> lick_uid). Cued licks are the
extraction's subsample (<= 1500 / session), so every class keeps its share.
Per animal x lick class (first / later) x spout x epoch: landing shift (actual - the animal's pre mean for that spout and
class), decoded shift (same), offset beyond the pull (decoded shift - b x landing shift; b = the animal's pre slope,
fit on all pre cued licks). Change vs pre = paired nested bootstrap over sessions; first - later difference per session.
Outputs <E>/iti_licks/first_lick_{preds.csv.gz,sessions.csv,tests.csv}, figures/iti_first_lick.png.
Usage: python scripts/iti_first_lick.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
EPS = ["pre", *EP]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
MIN = 8


def _lick_idx(label: str, t: np.ndarray) -> np.ndarray:
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_peak_s"])
    uid = pd.Series(np.arange(len(A)), index=A.daq_peak_s.to_numpy(float))
    uid = uid[~uid.index.duplicated()]
    K = _K()
    K = K[(K.session == f"{an}_{date}") & (K.table == "licks")].drop_duplicates("lick_uid").set_index("lick_uid")
    u = uid.reindex(np.asarray(t, float)).to_numpy()
    out = np.full(len(t), np.nan)
    ok = np.isfinite(u)
    out[ok] = K.lick_idx.reindex(u[ok].astype(int)).to_numpy(float)
    return out


_KC = {}


def _K():
    if "K" not in _KC:
        from wfield_local.paths import PathResolver
        f = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz"
        _KC["K"] = pd.read_csv(f, usecols=["session", "table", "lick_uid", "lick_idx"])
    return _KC["K"]


def predictions() -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    from iti_decoder_contrib import feature_names
    from wfield_local import orofacial_cohort as oc
    SD = pd.read_csv(il.OUT / "zscore_check" / "session_sd.csv", index_col=0)
    names = feature_names()
    D = wd.load("areas")
    rows = []
    for a in AN:
        S = {}
        for x, d in sorted(D.items()):
            if d["animal"] != a:
                continue
            X, C = d["trial"]
            t = il._load(x)["tr_t"].astype(float)
            idx = _lick_idx(x, t)
            pos = np.array([q if q in POS or "_" not in q else oc.pos_key(q, "L") for q in d["trial_pos"].astype(str)])
            ok = np.isfinite(C[:, 0]) & np.isfinite(idx) & np.isin(pos, POS)
            if ok.sum() >= 20:
                S[x] = (d["epoch"], X[ok] / SD.loc[x, names].to_numpy(float)[None, :], C[ok, 0], idx[ok], pos[ok])
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
            rows.append(pd.DataFrame({"animal": a, "label": x, "epoch": ep, "pos": pos, "lick_idx": idx,
                                      "act": y, "dec": m.predict(sc.transform(X))}))
    return pd.concat(rows, ignore_index=True)


def sessions(P: pd.DataFrame) -> pd.DataFrame:
    P = P.copy()
    P["cls"] = np.where(P.lick_idx == 0, "first", "later")
    rows = []
    for a, g in P.groupby("animal"):
        pre = g[g.epoch == "pre"]
        b, _c = np.polyfit(pre.act, pre.dec, 1)
        ref = pre.groupby(["cls", "pos"])[["act", "dec"]].mean()
        for (lab, ep, cls, sp), h in g.groupby(["label", "epoch", "cls", "pos"]):
            if len(h) < MIN or (cls, sp) not in ref.index:
                continue
            da = h.act.mean() - ref.loc[(cls, sp), "act"]
            dd = h.dec.mean() - ref.loc[(cls, sp), "dec"]
            rows.append(dict(animal=a, label=lab, epoch=ep, cls=cls, spout=sp, n=len(h), landing=da, decoded=dd,
                             beyond=dd - b * da))
    return pd.DataFrame(rows)


def tests(S: pd.DataFrame) -> pd.DataFrame:
    out = []
    for scope, g0 in (("all spouts", S), ("far contra", S[S.spout == "far_contra"])):
        g1 = g0.groupby(["animal", "label", "epoch", "cls"])[["landing", "decoded", "beyond"]].mean().reset_index()
        w = g1.pivot_table(index=["animal", "label", "epoch"], columns="cls", values="beyond").dropna().reset_index()
        w["first_minus_later"] = w["first"] - w["later"]
        for val, src in (("landing", None), ("decoded", None), ("beyond", None), ("first_minus_later", w)):
            for cls in (("first", "later") if src is None else ("diff",)):
                g = g1[g1.cls == cls] if src is None else src
                col = val
                prd = {a: h[col].dropna().to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
                for ep in EP:
                    post = {a: h[col].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in prd}
                    post = {a: v for a, v in post.items() if len(v)}
                    if not post:
                        continue
                    ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                    am = (g[g.epoch == ep].groupby("animal")[col].mean()
                          - g[g.epoch == "pre"].groupby("animal")[col].mean()).dropna()
                    out.append(dict(scope=scope, measure=val, cls=cls, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                                    n_dir=int((am * np.sign(ci.point) > 0).sum()), n=len(am)))
    return pd.DataFrame(out)


def plot(T: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.4), squeeze=False)
    for i, scope in enumerate(("far contra", "all spouts")):
        for j, val in enumerate(("landing", "decoded", "beyond")):
            ax = axs[i, j]
            for k, (cls, col) in enumerate((("first", "#c0392b"), ("later", "0.4"))):
                t = T[(T.scope == scope) & (T.measure == val) & (T.cls == cls)].set_index("epoch").reindex(EP)
                x = np.arange(1, 4) + (k - 0.5) * 0.18
                ax.errorbar(x, t.point, yerr=[t.point - t.lo, t.hi - t.point], fmt="o-", color=col, capsize=3,
                            label=f"{cls} lick" + (" (lick_idx 0)" if cls == "first" else "s"))
                for xx, (_e, r) in zip(x, t.iterrows()):
                    if np.isfinite(r.point):
                        ax.text(xx, r.hi, f"{r.point:+.2f}\n{int(r.n_dir)}/{int(r.n)}", ha="center", va="bottom",
                                fontsize=6.5, color=col)
            ax.axhline(0, color="0.7", lw=0.6, ls=":")
            ax.set_xticks([1, 2, 3], EP)
            ax.set_title(f"{scope}: {val} shift vs pre (mm)", fontsize=9)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    axs[0, 0].legend(fontsize=8, frameon=False)
    d = T[(T.measure == "first_minus_later")]
    txt = "; ".join(f"{r.scope} {r.epoch}: {r.point:+.2f} [{r.lo:+.2f},{r.hi:+.2f}] {r.n_dir}/{r.n}" for r in d.itertuples())
    fig.suptitle("FIRST cued lick (launched before feedback) vs LATER licks: landing (BEHAVIOUR), decoded and offset "
                 "beyond the pull (NEURAL), lateral, change vs each class's pre mean at the spout;\nfrozen KEY-row "
                 "decoder (z-scored Allen areas); 95% paired nested bootstrap CI, k/n animals. + = contralesional.\n"
                 f"Beyond-pull first - later, change vs pre: {txt}. PROVISIONAL", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(il.OUT / "figures" / "iti_first_lick.png", dpi=100)
    plt.close(fig)


def main() -> int:
    f = il.OUT / "first_lick_preds.csv.gz"
    if f.exists():
        P = pd.read_csv(f)
    else:
        P = predictions()
        P.to_csv(f, index=False)
    print("lick_idx share first:", P.groupby("epoch").lick_idx.apply(lambda s: (s == 0).mean()).round(3).to_dict())
    S = sessions(P)
    S.to_csv(il.OUT / "first_lick_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): far contra, beyond-pull offset by class")
    fc = S[S.spout == "far_contra"]
    print(fc.pivot_table(index=["animal", "cls"], columns="epoch", values="beyond").reindex(columns=EPS).round(2).to_string())
    T = tests(S)
    T.to_csv(il.OUT / "first_lick_tests.csv", index=False)
    print(T.round(3).to_string(index=False))
    plot(T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
