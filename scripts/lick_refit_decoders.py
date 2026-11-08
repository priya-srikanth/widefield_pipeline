"""CAN post-stroke lick position still be decoded? Decoders REFIT per epoch and per session, beside the FROZEN pre decoder
(Priya 10-27 late: "we're going to want to re-do the per session / per-epoch decoders now too right, to show they can
be fit"). Same features and targets as the KEY where-decoder (iti_where_decoder.decode, z-scored like iti_zscore_check):
Allen areas with the ALM / tjM1 split (ROI v8; WFIELD_AREA_BASIS), lags 0..0.2 s, each area / its frame-wise SD in the
session; ridge (RidgeCV) on standardized features.

Per animal x lick set (ITI / cued) x target (lateral, sagittal, angle, reach, 2-D):
  frozen         trained on the animal's PRE licks (pre sessions leave-one-session-out) and applied to every session
                 (= zscore_check/where_decoder_z.csv, recomputed here so all three rows share one code path)
  epoch refit    trained on the OTHER sessions of the SAME epoch (leave-one-session-out within the epoch); pre = the
                 frozen pre scores by construction. Needs >= 2 sessions in the epoch.
  session refit  trained and scored within ONE session: 5 contiguous blocks of licks (time order, no shuffling, so
                 neighboring licks never straddle train and test), r over the pooled held-out predictions.
  pre, matched n the epoch-refit control for training size: pre leave-one-session-out with the training licks randomly
                 subsampled to the mean training size of that animal's epoch refit (fewer post sessions = fewer
                 training licks, which alone lowers r).
Reading: frozen r falls while refit r stays near the (matched-n) pre level -> the position code is still there but its
read-out moved; refit r falls too -> less position information in these features.
Per session r; change vs pre = paired nested bootstrap (analysis_kit.boot_delta); per-animal values first (rule 8).
Outputs <E>/iti_licks/refit_decoders_sessions.csv, refit_decoders_tests.csv, figures/lick_refit_decoders.png;
figdata/lick_refit_decoders.pkl (--plot-only redraws from it).
Usage: python scripts/lick_refit_decoders.py [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
SETS = {"iti": "ITI", "trial": "cued"}
TARGETS = ("lateral", "sagittal", "angle", "reach", "2-D")
SHOWN = ("lateral", "angle", "reach")
KINDS = ("frozen", "epoch refit", "session refit", "pre, matched n")
KCOL = {"frozen": "#7b3294", "epoch refit": "#2c7fb8", "session refit": "#41ab5d", "pre, matched n": "0.45"}
NFOLD = 5
MIN_LICKS = 40
STEM = "lick_refit_decoders"
ALPHAS = np.logspace(-1, 5, 13)


def load_z() -> dict:
    """Session -> {epoch, animal, iti / trial: (X z-scored, Y), iti_idx / trial_idx}; Y columns lateral, sagittal,
    angle, reach; *_idx = the kept licks' rows in the session file (licks with every coordinate finite)."""
    names, SDb = wd.area_meta()
    D = wd.load("areas")
    out = {}
    for x, d in D.items():
        sd = SDb.loc[x, names].to_numpy(float)
        rec = {"epoch": d["epoch"], "animal": d["animal"]}
        for st in SETS:
            X, C = d[st]
            ok = np.isfinite(C).all(1)
            Y = np.column_stack([C[ok, 0], C[ok, 3], np.degrees(np.arctan2(C[ok, 0], C[ok, 3])), C[ok, 2]])
            rec[st] = (X[ok] / sd[None, :], Y)
            rec[f"{st}_idx"] = np.flatnonzero(ok)       # row of each kept lick in the session file (iti_t / tr_t order)
        out[x] = rec
    return out


def _fit(X: np.ndarray, Y: np.ndarray):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    return sc, [RidgeCV(alphas=ALPHAS).fit(Xs, Y[:, k]) for k in range(Y.shape[1])]


def _pred(fit, X: np.ndarray) -> np.ndarray:
    sc, ms = fit
    Xs = sc.transform(X)
    return np.column_stack([m.predict(Xs) for m in ms])


def _scores(P: np.ndarray, Y: np.ndarray, sd2: np.ndarray) -> dict:
    r = {t: float(np.corrcoef(P[:, k], Y[:, k])[0, 1]) for k, t in enumerate(TARGETS[:4])}
    r["2-D"] = wd._vec_r(P[:, :2] / sd2, Y[:, :2] / sd2)
    return r


def _contiguous_cv(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    P = np.full(Y.shape, np.nan)
    for idx in np.array_split(np.arange(len(Y)), NFOLD):
        trn = np.setdiff1d(np.arange(len(Y)), idx)
        P[idx] = _pred(_fit(X[trn], Y[trn]), X[idx])
    return P


def compute() -> pd.DataFrame:
    D = load_z()
    rng = np.random.default_rng(0)
    rows = []
    for st in SETS:
        for a in AN:
            S = {x: (d["epoch"], *d[st]) for x, d in sorted(D.items()) if d["animal"] == a and len(d[st][1]) >= MIN_LICKS}
            by = {ep: [x for x, v in S.items() if v[0] == ep] for ep in EP}
            if len(by["pre"]) < 3:
                continue
            sd2 = np.vstack([S[q][2] for q in by["pre"]])[:, :2].std(0)
            frozen = {h: _fit(np.vstack([S[q][1] for q in by["pre"] if q != h]),
                              np.vstack([S[q][2] for q in by["pre"] if q != h])) for h in [*by["pre"], None]}
            ntrain = {}
            for x, (ep, X, Y) in S.items():
                base = dict(set=st, animal=a, label=x, epoch=ep, n=len(Y))
                rec = {"frozen": _scores(_pred(frozen[x if ep == "pre" else None], X), Y, sd2),
                       "session refit": _scores(_contiguous_cv(X, Y), Y, sd2)}
                if ep == "pre":
                    rec["epoch refit"] = rec["frozen"]
                elif len(by[ep]) >= 2:
                    oth = [q for q in by[ep] if q != x]
                    Xt, Yt = np.vstack([S[q][1] for q in oth]), np.vstack([S[q][2] for q in oth])
                    ntrain.setdefault(ep, []).append(len(Yt))
                    rec["epoch refit"] = _scores(_pred(_fit(Xt, Yt), X), Y, sd2)
                for kind, sc_ in rec.items():
                    rows += [dict(base, kind=kind, target=t, r=v) for t, v in sc_.items()]
            # training-size control: pre LOSO with training licks subsampled to each epoch's mean refit size
            for ep, ns in ntrain.items():
                m = int(np.mean(ns))
                for x in by["pre"]:
                    Xt = np.vstack([S[q][1] for q in by["pre"] if q != x])
                    Yt = np.vstack([S[q][2] for q in by["pre"] if q != x])
                    if len(Yt) > m:
                        k = rng.choice(len(Yt), m, replace=False)
                        Xt, Yt = Xt[k], Yt[k]
                    _ep, X, Y = S[x]
                    rows += [dict(set=st, animal=a, label=x, epoch="pre", n=len(Y), kind="pre, matched n", vs=ep,
                                  target=t, r=v) for t, v in _scores(_pred(_fit(Xt, Yt), X), Y, sd2).items()]
            print(f"[{st} {a}] sessions {[len(by[e]) for e in EP]}, epoch-refit training licks "
                  f"{ {e: int(np.mean(v)) for e, v in ntrain.items()} }", flush=True)
    return pd.DataFrame(rows)


def tests(R: pd.DataFrame) -> pd.DataFrame:
    """Per set x target x kind x post epoch: change in r vs pre (frozen / refits: vs the frozen pre LOSO; the matched-n
    row: the epoch-refit r vs the pre LOSO trained on the same number of licks)."""
    out = []
    for (st, tg), g in R.groupby(["set", "target"]):
        for kind in ("frozen", "epoch refit", "session refit"):
            for ep in EP[1:]:
                post = g[(g.kind == kind) & (g.epoch == ep)]
                if kind == "epoch refit":
                    ref = g[(g.kind == "pre, matched n") & (g.vs == ep)]
                    ref_lab = "pre, matched n"
                else:
                    ref = g[(g.kind == kind) & (g.epoch == "pre")]
                    ref_lab = f"pre ({kind})"
                com = [a for a in AN if (post.animal == a).any() and (ref.animal == a).any()]
                if not com:
                    continue
                ci = ak.boot_delta({a: post[post.animal == a].r.dropna().to_numpy() for a in com},
                                   {a: ref[ref.animal == a].r.dropna().to_numpy() for a in com}, np.random.default_rng(0))
                am = {a: float(post[post.animal == a].r.mean() - ref[ref.animal == a].r.mean()) for a in com}
                out.append(dict(set=st, target=tg, kind=kind, epoch=ep, ref=ref_lab,
                                pre_r=float(np.mean([ref[ref.animal == a].r.mean() for a in com])),
                                post_r=float(np.mean([post[post.animal == a].r.mean() for a in com])),
                                delta=ci.point, lo=ci.lo, hi=ci.hi, n_down=int(sum(v < 0 for v in am.values())),
                                n=len(com), **{f"animal_{a}": am.get(a, np.nan) for a in AN}))
    return pd.DataFrame(out)


def figure_data(R: pd.DataFrame, T: pd.DataFrame) -> dict:
    panels = {}
    for st in SETS:
        for tg in SHOWN:
            g = R[(R.set == st) & (R.target == tg)]
            lines = {}
            for kind in ("frozen", "epoch refit", "session refit"):
                am = g[g.kind == kind].groupby(["animal", "epoch"]).r.mean()
                mu = [am.xs(e, level="epoch").mean() if e in am.index.get_level_values("epoch") else np.nan for e in EP]
                se = [am.xs(e, level="epoch").std(ddof=1) / np.sqrt(am.xs(e, level="epoch").size)
                      if e in am.index.get_level_values("epoch") and am.xs(e, level="epoch").size > 1 else np.nan for e in EP]
                lines[kind] = (mu, se)
            mn = g[g.kind == "pre, matched n"].groupby(["vs", "animal"]).r.mean().groupby("vs").mean()
            t = T[(T.set == st) & (T.target == tg)]
            txt = {(r.kind, r.epoch): f"{r.delta:+.2f}{' *' if (r.lo > 0 or r.hi < 0) else ''} {r.n_down}/{r.n}"
                   for r in t.itertuples()}
            panels[(st, tg)] = {"lines": lines, "matched": {e: float(mn.get(e, np.nan)) for e in EP[1:]}, "txt": txt}
    return {"panels": panels}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, len(SHOWN), figsize=(4.6 * len(SHOWN), 8.6), squeeze=False)
    x = np.arange(len(EP))
    for i, st in enumerate(SETS):
        for j, tg in enumerate(SHOWN):
            ax = axs[i, j]
            P = F["panels"].get((st, tg))
            if P is None:
                ax.set_axis_off()
                continue
            for k, kind in enumerate(("frozen", "epoch refit", "session refit")):
                mu, se = P["lines"][kind]
                ax.errorbar(x + (k - 1) * 0.06, mu, yerr=se, fmt="o-", color=KCOL[kind], lw=1.6, ms=5, capsize=2,
                            label=kind)
                for e in range(1, len(EP)):
                    t = P["txt"].get((kind, EP[e]))
                    if t:
                        ax.text(e, 0.03 + 0.065 * k, t, color=KCOL[kind], fontsize=7, ha="center",
                                transform=ax.get_xaxis_transform())
            mx = [e for e in range(1, len(EP)) if np.isfinite(P["matched"].get(EP[e], np.nan))]
            ax.plot(mx, [P["matched"][EP[e]] for e in mx], "_", color=KCOL["pre, matched n"], ms=16, mew=2.5,
                    label="pre, matched n (epoch-refit reference)")
            ax.set_xticks(x, EP)
            ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
            ax.set_title(f"{SETS[st]} licks: {tg.upper()}", fontsize=10)
            if j == 0:
                ax.set_ylabel("decoded vs actual, r (held-out)", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7.5, frameon=False, loc="upper right")
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("Can lick position still be DECODED after the stroke? Frozen pre decoder vs decoders REFIT per epoch "
                 "(leave-one-session-out within the epoch) and per session (5 contiguous blocks).\n" + il.basis_label() +
                 " (ROI v8), z-scored, lags 0-0.2 s. Mean +/- SEM over animals; text = change vs pre (frozen / session "
                 "refit: vs their pre; epoch refit: vs pre trained on the SAME number of licks, gray bars), 95 % paired "
                 "nested bootstrap, * = CI excludes 0, k/n animals lower. PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_png_svg(fig, il.OUT / "figures" / f"{STEM}.png", dpi=100)
    plt.close(fig)


def main() -> int:
    il.split_only(STEM)
    if plot_only():
        plot(load_fd(STEM))
        return 0
    R = compute()
    R.to_csv(il.OUT / "refit_decoders_sessions.csv", index=False)
    T = tests(R)
    T.to_csv(il.OUT / "refit_decoders_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print("== per animal (rule 8) + nested bootstrap")
    print(T.round(3).to_string(index=False))
    F = figure_data(R, T)
    save_fd(STEM, F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
