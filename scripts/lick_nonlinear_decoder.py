"""NON-LINEAR read-out of lick position (overnight queue 5f, approved LOW item; SVD_GAUSS queue C "non-linear decoder"):
does a non-linear decoder recover the post-stroke loss of the frozen read-out, or remove the like-for-like offset?

Same frame as the KEY frozen where-decoder (iti_where_decoder.decode / lick_refit_decoders "frozen"): Allen areas with
the ALM / tjM1 split (ROI v8), each area / its frame-wise SD in the session (zscore_check), lags 0..0.2 s averaged; per
animal x lick set (ITI / cued); trained on the animal's AUGUST PRE licks, pre sessions scored leave-one-session-out,
every post session scored by the decoder fit on all pre sessions (frozen, never refit). Targets lateral (mm) and angle
(deg, atan2(lateral, sagittal)). Two decoders on IDENTICAL licks and folds:
  linear      the KEY decoder itself (StandardScaler + RidgeCV per target, lick_refit_decoders._fit), so its r reproduces
              refit_decoders_sessions.csv "frozen" exactly (checked at run time)
  non-linear  KERNEL RIDGE with an RBF kernel, Nystroem approximation (N_LANDMARK training licks as landmarks, fixed seed;
              exact kernel ridge on ~10^4 pre licks is a 10^4 x 10^4 solve per fit). Features standardized on the training
              licks. Hyperparameters (RBF width gamma = c / n_features, ridge alpha = a * n_train) chosen by INNER
              leave-one-session-out over the TRAINING pre sessions only (nested: a held-out pre session never informs its
              own hyperparameters), pooled held-out MSE, separately per target.
Why kernel ridge: it is the smooth, convex, ridge-regularized generalization of the linear decoder (the same loss and
penalty in an RBF feature space; small gamma -> ~linear), so a difference from the linear row is the non-linearity, not
a change of loss, optimizer or regularization family. Boosting / an MLP add tuning knobs and randomness without a reason
to expect a smoother read-out of 45 area signals.
Readouts per session: r (decoded vs actual), r / animal's pre mean (model's own pre), and the like-for-like same-place
offset (likeforlike.offset, Gaussian reference = method "nw", sigma by held-out pre error, chosen per model x target).
Change vs pre per model and the PAIRED model contrast (non-linear - linear per session, then post - pre): nested
bootstrap (analysis_kit.boot_delta); per-animal values first (rule 8). Held-out pre r per model + per-animal difference.
Outputs <OUT_W>/nonlinear_decoder_{preds.csv.gz,hyper.csv,sessions.csv,tests.csv,pre.csv}, figures/lick_nonlinear_decoder.png,
figdata/lick_nonlinear_decoder.pkl (--plot-only redraws from it). OUT_W = iti_licks.OUT_W (env WFIELD_ITI_OUT_W).
Usage: python scripts/lick_nonlinear_decoder.py [--plot-only] [--animals PS94 ...] [--jobs 8]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import likeforlike as lfl  # noqa: E402
import lick_refit_decoders as lr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
SETS = {"iti": "ITI", "trial": "cued"}
TARGETS = {"lateral": 0, "angle": 2}            # column of lick_refit_decoders.load_z Y
UNIT = {"lateral": "mm", "angle": "deg"}
SIGMAS = {"lateral": lfl.SIGMA_MM, "angle": lfl.SIGMA_DEG}
MODELS = ("linear", "nonlinear")
MCOL = {"linear": "#7b3294", "nonlinear": "#e66101", "nonlinear - linear": "0.3"}
N_LANDMARK = 1000
GAMMA_C = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)      # gamma = c / n_features (standardised features); small = ~linear
ALPHA_A = np.logspace(-9, 0, 10)                # alpha = a * n_train
MIN_OFF = 8                                     # licks with a reference per session for a session offset
STEM = "lick_nonlinear_decoder"


# ------------------------------------------------------------------------------------------------ kernel ridge
def _nys(X: np.ndarray, gamma: float):
    from sklearn.kernel_approximation import Nystroem
    return Nystroem(kernel="rbf", gamma=gamma, n_components=min(N_LANDMARK, len(X)), random_state=0).fit(X)


def _ridge_path(F: np.ndarray, Y: np.ndarray, alphas: np.ndarray) -> tuple:
    """Ridge (centered, unpenalized intercept) for every alpha at once from one eigendecomposition. -> (mu_f, mu_y, [W])."""
    mu_f, mu_y = F.mean(0), Y.mean(0)
    Fc = F - mu_f
    s, V = np.linalg.eigh(Fc.T @ Fc)
    FtY = V.T @ (Fc.T @ (Y - mu_y))
    return mu_f, mu_y, [V @ (FtY / (np.maximum(s, 0) + a)[:, None]) for a in alphas]


def _krr_fit(X: np.ndarray, Y: np.ndarray, gamma: float, alpha: float) -> tuple:
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    ny = _nys(Xs, gamma)
    mu_f, mu_y, (W,) = _ridge_path(ny.transform(Xs), Y, np.array([alpha]))
    return sc, ny, mu_f, mu_y, W


def _krr_pred(fit: tuple, X: np.ndarray) -> np.ndarray:
    sc, ny, mu_f, mu_y, W = fit
    return (ny.transform(sc.transform(X)) - mu_f) @ W + mu_y


def _krr_select(S: dict, trn: list[str], k: int) -> tuple[float, float, np.ndarray]:
    """Inner leave-one-session-out over the training sessions `trn` for target column k: pooled held-out MSE per
    (gamma, alpha). Returns (gamma, alpha, mse grid (gamma x alpha))."""
    from sklearn.preprocessing import StandardScaler
    nf = S[trn[0]][1].shape[1]
    sse = np.zeros((len(GAMMA_C), len(ALPHA_A)))
    nn = 0
    for h in trn:
        it = [q for q in trn if q != h]
        Xt, Yt = np.vstack([S[q][1] for q in it]), np.vstack([S[q][2] for q in it])[:, [k]]
        Xh, Yh = S[h][1], S[h][2][:, [k]]
        sc = StandardScaler().fit(Xt)
        Xts, Xhs = sc.transform(Xt), sc.transform(Xh)
        for i, c in enumerate(GAMMA_C):
            ny = _nys(Xts, c / nf)
            mu_f, mu_y, Ws = _ridge_path(ny.transform(Xts), Yt, ALPHA_A * len(Yt))
            Fh = ny.transform(Xhs) - mu_f
            for j, W in enumerate(Ws):
                sse[i, j] += float(((Fh @ W + mu_y - Yh) ** 2).sum())
        nn += len(Yh)
    mse = sse / nn
    i, j = np.unravel_index(np.argmin(mse), mse.shape)
    return GAMMA_C[i] / nf, float(ALPHA_A[j]), mse


def _unit(item):
    """One (set, animal): frozen linear + nested kernel-ridge decoders. Module-level for the spawn pool."""
    st, a, S = item
    t0 = time.time()
    pre = [x for x, v in S.items() if v[0] == "pre"]
    cols = list(TARGETS.values())
    # STORED (Priya 11-03): the whole frozen reference -- linear KEY decoders, nested kernel-ridge fits and their
    # hyper-parameter rows -- through frozen_models.load_or_fit (kind lick_nonlinear; spec fingerprints the training
    # rows). WIDEFIELD_ITI_FROZEN=0 fits in memory exactly as before.
    from wfield_local import frozen_models, frozen_readouts

    def _fit_all():
        lin_, nl_, hyp_ = _fit_reference(st, a, S, pre, cols)
        return {"lin": lin_, "nl": nl_, "hyp": hyp_}
    if frozen_models.family_on(frozen_readouts.ITI_FROZEN_ENV):
        spec = frozen_models.make_spec(a, "lick_nonlinear", align=f"{st}", source="lick_refit_z", train_labels=pre,
                                       post_s=0.0, n_features=int(np.asarray(S[pre[0]][1]).shape[1]),
                                       train_data={q: (S[q][1], S[q][2]) for q in pre}, lp_only=True)
        spec["targets"] = {k: int(v) for k, v in TARGETS.items()}
        spec["grid"] = [[float(g) for g in GAMMA_C], [float(x) for x in ALPHA_A]]
        P, status = frozen_models.load_or_fit(spec, _fit_all, meta={"script": "lick_nonlinear_decoder.py"})
        print(f"  [frozen] {st} {a}: lick_nonlinear {status}", flush=True)
    else:
        P = _fit_all()
    lin, nl, hyp = P["lin"], P["nl"], P["hyp"]
    rows = []
    for x, (ep, X, Y) in S.items():
        key = x if ep == "pre" else None
        Pl = lr._pred(lin[key], X)
        Pn = np.column_stack([_krr_pred(f, X)[:, 0] for f in nl[key]])
        R = pd.DataFrame({"set": st, "animal": a, "label": x, "epoch": ep, "k": np.arange(len(Y))})
        for j, (tg, k) in enumerate(TARGETS.items()):
            R[f"act_{tg}"] = Y[:, k]
            R[f"linear_{tg}"] = Pl[:, j]
            R[f"nonlinear_{tg}"] = Pn[:, j]
        rows.append(R)
    print(f"[{st} {a}] {len(pre)} pre sessions, {sum(len(v[2]) for v in S.values())} licks, {time.time() - t0:.0f} s",
          flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(hyp)


def _fit_reference(st, a, S, pre, cols):
    """The pre-LOSO / all-pre linear + kernel-ridge fits of one (set, animal) (moved out of `_unit` unchanged)."""
    lin, nl, hyp = {}, {}, []
    for held in [*pre, None]:
        trn = [q for q in pre if q != held]
        Xt, Yt = np.vstack([S[q][1] for q in trn]), np.vstack([S[q][2] for q in trn])
        lin[held] = lr._fit(Xt, Yt[:, cols])                 # the KEY decoder (RidgeCV per target)
        nl[held] = []
        for tg, k in TARGETS.items():
            g, a_, mse = _krr_select(S, trn, k)
            nl[held].append(_krr_fit(Xt, Yt[:, [k]], g, a_ * len(Yt)))
            ig, ia = np.unravel_index(np.argmin(mse), mse.shape)
            hyp.append(dict(set=st, animal=a, held_out=held or "(all pre: post sessions)", target=tg,
                            gamma_c=GAMMA_C[ig], alpha_a=a_, inner_mse=float(mse.min()),
                            edge=bool(ig in (0, len(GAMMA_C) - 1) or ia in (0, len(ALPHA_A) - 1)), n_train=len(Yt)))
    return lin, nl, hyp


def predictions(animals: list[str] | None, jobs: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    D = lr.load_z()
    items = []
    for st in SETS:
        for a in AN:
            if animals and a not in animals:
                continue
            S = {x: (d["epoch"], *d[st]) for x, d in sorted(D.items())
                 if d["animal"] == a and len(d[st][1]) >= lr.MIN_LICKS}
            if sum(v[0] == "pre" for v in S.values()) >= 3:
                items.append((st, a, S))
    res, fail = ak.fan_sessions(items, _unit, jobs=jobs, key=lambda it: (it[0], it[1]), label="decoder unit",
                                 log=lambda m: None)
    if fail:
        raise RuntimeError(f"{len(fail)} decoder units failed: {fail[0][1]}")
    return (pd.concat([v[0] for _it, v in res], ignore_index=True),
            pd.concat([v[1] for _it, v in res], ignore_index=True))


# ------------------------------------------------------------------------------------------------ session readouts
def offsets(P: pd.DataFrame, models=MODELS, targets=tuple(TARGETS)) -> pd.DataFrame:
    """Per lick: like-for-like same-place offset (Gaussian reference, likeforlike method "nw") per model x target."""
    out = pd.DataFrame(index=P.index)
    for m in models:
        for tg in targets:
            Q = P[["set", "animal", "label", "epoch"]].assign(act=P[f"act_{tg}"], dec=P[f"{m}_{tg}"])
            res, _out = lfl.offset(Q, "act", "dec", SIGMAS[tg], f"{m} {tg}", method="nw")
            out[f"off_{m}_{tg}"] = res
    return out


def sessions(P: pd.DataFrame, models=MODELS, targets=tuple(TARGETS)) -> pd.DataFrame:
    """Per set x session x model x target: r, offset (mean over licks with a reference), r_norm (/ animal's pre mean)."""
    rows = []
    for (st, a, lab, ep), g in P.groupby(["set", "animal", "label", "epoch"]):
        for m in models:
            for tg in targets:
                off = g[f"off_{m}_{tg}"]
                rows.append(dict(set=st, animal=a, label=lab, epoch=ep, model=m, target=tg, n=len(g),
                                 r=float(np.corrcoef(g[f"{m}_{tg}"], g[f"act_{tg}"])[0, 1]),
                                 offset=float(off.mean()) if off.notna().sum() >= MIN_OFF else np.nan,
                                 off_cov=float(off.notna().mean())))
    S = pd.DataFrame(rows)
    pm = S[S.epoch == "pre"].groupby(["set", "animal", "model", "target"]).r.mean()
    S["r_norm"] = S.r / [pm.get(k, np.nan) for k in zip(S.set, S.animal, S.model, S.target)]
    return S


def _boot_rows(S: pd.DataFrame, val: str, label: str, keys: dict) -> list[dict]:
    """Change vs pre of one session measure (S has animal, epoch, val): nested paired bootstrap + per animal."""
    out = []
    pre = {a: h[val].dropna().to_numpy() for a, h in S[S.epoch == "pre"].groupby("animal")}
    pre = {a: v for a, v in pre.items() if len(v)}
    pm = S[S.epoch == "pre"].groupby("animal")[val].mean()
    for ep in EP[1:]:
        post = {a: h[val].dropna().to_numpy() for a, h in S[S.epoch == ep].groupby("animal") if a in pre}
        post = {a: v for a, v in post.items() if len(v)}
        if not post:
            continue
        ci = ak.boot_delta(post, pre, np.random.default_rng(0))
        if ci is None:
            continue
        am = (S[S.epoch == ep].groupby("animal")[val].mean() - pm).dropna()
        out.append(dict(**keys, model=label, value=val, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                        n_dir=int((am * np.sign(ci.point) > 0).sum()), n=len(am),
                        **{f"animal_{a}": am.get(a, np.nan) for a in AN}))
    return out


def tests(S: pd.DataFrame, models=MODELS, contrasts=(("nonlinear", "linear"),), group=("set", "target")) -> pd.DataFrame:
    """Change vs pre per model, and the PAIRED contrast (model A - model B per session, then post - pre)."""
    rows = []
    vals = [v for v in ("r", "r_norm", "offset") if v in S]
    for k, g in S.groupby(list(group)):
        keys = dict(zip(group, k))
        W = g.pivot_table(index=["animal", "label", "epoch"], columns="model", values=vals)
        for val in vals:
            for m in models:
                if (val, m) in W.columns:
                    rows += _boot_rows(W[(val, m)].rename(val).reset_index(), val, m, keys)
            for a_, b_ in contrasts:
                if (val, a_) in W.columns and (val, b_) in W.columns:
                    d = (W[(val, a_)] - W[(val, b_)]).rename(val).reset_index()
                    rows += _boot_rows(d, val, f"{a_} - {b_}", keys)
    return pd.DataFrame(rows)


def pre_levels(S: pd.DataFrame, models=MODELS, contrasts=(("nonlinear", "linear"),), group=("set", "target")
               ) -> pd.DataFrame:
    """Held-out PRE r per model (animal means; mean of animals) and the paired difference per animal + nested CI of the
    LEVEL of the per-session difference (analysis_kit.boot_ci, flat pool)."""
    rows = []
    for k, g in S[S.epoch == "pre"].groupby(list(group)):
        keys = dict(zip(group, k))
        W = g.pivot_table(index=["animal", "label"], columns="model", values="r")
        for m in models:
            if m in W:
                am = W[m].groupby("animal").mean()
                rows.append(dict(**keys, model=m, point=float(am.mean()), lo=np.nan, hi=np.nan,
                                 **{f"animal_{a}": am.get(a, np.nan) for a in AN}))
        for a_, b_ in contrasts:
            if a_ in W and b_ in W:
                d = (W[a_] - W[b_]).dropna()
                am = d.groupby("animal").mean()
                ci = ak.boot_ci({a: d.xs(a, level="animal").to_numpy() for a in am.index}, np.random.default_rng(0))
                rows.append(dict(**keys, model=f"{a_} - {b_}", point=float(am.mean()), lo=ci.lo, hi=ci.hi,
                                 n_dir=int((am * np.sign(am.mean()) > 0).sum()), n=len(am),
                                 **{f"animal_{a}": am.get(a, np.nan) for a in AN}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figure
def figure_data(S: pd.DataFrame, T: pd.DataFrame, PL: pd.DataFrame, H: pd.DataFrame) -> dict:
    lines = {}
    for (st, tg, m), g in S.groupby(["set", "target", "model"]):
        for val in ("r", "r_norm"):
            am = g.groupby(["animal", "epoch"])[val].mean().unstack("epoch").reindex(columns=EP)
            lines[(st, tg, m, val)] = (am.mean().to_numpy(), (am.std(ddof=1) / np.sqrt(am.count())).to_numpy(),
                                       am.to_dict("index"))
    return dict(lines=lines, T=T, PL=PL, H=H, animals=sorted(S.animal.unique()))


def _txt(r) -> str:
    sig = np.isfinite(r.lo) and (r.lo > 0 or r.hi < 0)
    return f"{r.point:+.2f}{'*' if sig else ''} {int(r.n_dir)}/{int(r.n)}"


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    T, PL = F["T"], F["PL"]
    fig, axs = plt.subplots(2, 4, figsize=(19, 9.2), squeeze=False)
    x = np.arange(len(EP))
    for i, st in enumerate(SETS):
        for j, tg in enumerate(TARGETS):
            ax = axs[i, j]                                         # r / pre per model
            for q, m in enumerate(MODELS):
                L = F["lines"].get((st, tg, m, "r_norm"))
                if L is None:
                    continue
                mu, se, _am = L
                ax.errorbar(x + (q - 0.5) * 0.1, mu, yerr=se, fmt="o-", color=MCOL[m], lw=1.6, ms=5, capsize=2, label=m)
                for e in range(1, len(EP)):
                    t = T[(T.set == st) & (T.target == tg) & (T.model == m) & (T.value == "r_norm") & (T.epoch == EP[e])]
                    if len(t):
                        ax.text(e, 0.03 + 0.06 * q, _txt(t.iloc[0]), color=MCOL[m], fontsize=7, ha="center",
                                transform=ax.get_xaxis_transform())
            for e in range(1, len(EP)):
                t = T[(T.set == st) & (T.target == tg) & (T.model == "nonlinear - linear") & (T.value == "r_norm")
                      & (T.epoch == EP[e])]
                if len(t):
                    ax.text(e, 0.15, "NL-L " + _txt(t.iloc[0]), color=MCOL["nonlinear - linear"], fontsize=7,
                            ha="center", transform=ax.get_xaxis_transform())
            pl = PL[(PL.set == st) & (PL.target == tg)].set_index("model")
            pre_s = ", ".join(f"{m} {pl.loc[m, 'point']:.3f}" for m in MODELS if m in pl.index)
            if "nonlinear - linear" in pl.index:
                d = pl.loc["nonlinear - linear"]
                pre_s += f"\nNL - L {d.point:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}] {int(d.n_dir)}/{int(d.n)}"
            ax.axhline(1, color="0.7", lw=0.8, ls="--")
            ax.set_xticks(x, EP)
            ax.set_ylim(min(0.0, ax.get_ylim()[0]), max(1.25, ax.get_ylim()[1]))
            ax.set_title(f"{SETS[st]} licks: {tg.upper()}, frozen r / animal's pre\nheld-out pre r: {pre_s}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("r / animal's pre mean (model's own pre)", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper right")
            ax = axs[i, 2 + j]                                     # like-for-like offset change
            labs = []
            for q, m in enumerate([*MODELS, "nonlinear - linear"]):
                t = T[(T.set == st) & (T.target == tg) & (T.model == m) & (T.value == "offset")].set_index("epoch")
                xs = np.arange(1, 4) + (q - 1) * 0.22
                for e, xx in zip(EP[1:], xs):
                    if e not in t.index:
                        continue
                    r = t.loc[e]
                    ax.errorbar(xx, r.point, yerr=[[r.point - r.lo], [r.hi - r.point]], fmt="o", color=MCOL[m], ms=6,
                                capsize=2.5, mfc="w" if m == "nonlinear - linear" else MCOL[m])
                    for a in F["animals"]:
                        v = r.get(f"animal_{a}", np.nan)
                        if np.isfinite(v):
                            ax.plot(xx - 0.07, v, ANIMAL_MARKER.get(a, "o"), mfc="none", mec=MCOL[m], ms=4, mew=0.8)
                    labs.append((xx, _txt(r), MCOL[m]))
            lo_, hi_ = ax.get_ylim()
            ax.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))                # headroom for the text strip (inside the axes)
            for xx, s_, c_ in labs:
                ax.text(xx, 0.985, s_, color=c_, fontsize=6.3, ha="center", va="top", rotation=90,
                        transform=ax.get_xaxis_transform())
            ax.axhline(0, color="0.6", lw=0.7, ls="--")
            ax.set_xticks([1, 2, 3], EP[1:])
            ax.set_xlim(0.5, 3.6)
            ax.set_title(f"{SETS[st]} licks: {tg.upper()} like-for-like offset\nchange vs pre ({UNIT[tg]}, + = read more "
                         f"contralesional)", fontsize=8.5)
            for a_ in axs[i]:
                for s_ in ("top", "right"):
                    a_.spines[s_].set_visible(False)
    h = [Line2D([], [], color=MCOL[m], marker="o", lw=0, label=m, mfc="w" if " - " in m else MCOL[m])
         for m in [*MODELS, "nonlinear - linear"]]
    h += [Line2D([], [], color="0.4", marker=ANIMAL_MARKER[a], lw=0, mfc="none", label=a) for a in F["animals"]]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=8.5, frameon=False)
    fig.suptitle("Does a NON-LINEAR read-out recover the post-stroke loss or remove the offset? Frozen pre-stroke "
                 "decoders on identical licks and folds:\nLINEAR = the KEY ridge decoder; NON-LINEAR = RBF kernel ridge "
                 f"(Nystroem, {N_LANDMARK} landmarks; width + penalty by inner leave-one-pre-session-out). "
                 f"{il.basis_label()} (ROI v8), z-scored, lags 0-0.2 s.\nLeft: r / animal's pre (mean +/- SEM over "
                 "animals). Right: change vs pre of the like-for-like same-place offset (Gaussian pre reference, "
                 "sigma by held-out pre error).\nText: change vs pre, 95 % paired nested bootstrap, * = CI excludes 0, "
                 "k/n animals in that direction; NL-L = paired contrast (non-linear minus linear). PROVISIONAL",
                 fontsize=9.5)
    import iti_lick_order as lo
    lo._layout(fig, bottom=0.05)
    fig.subplots_adjust(top=1 - (0.17 * 5 + 0.25 + 0.55) / fig.get_size_inches()[1])   # suptitle + 3-line panel titles
    out = il.OUT_W / "figures" / f"{STEM}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, out, dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def check_linear(S: pd.DataFrame) -> float:
    """The linear row must reproduce the KEY frozen r of lick_refit_decoders (refit_decoders_sessions.csv, if present)."""
    f = il.OUT / "refit_decoders_sessions.csv"
    if not f.exists():
        print("check: no refit_decoders_sessions.csv to compare with")
        return np.nan
    R = pd.read_csv(f)
    R = R[(R.kind == "frozen") & R.target.isin(list(TARGETS))]
    M = S[S.model == "linear"].merge(R[["set", "label", "target", "r"]], on=["set", "label", "target"],
                                     suffixes=("", "_key"))
    d = float(np.nanmax(np.abs(M.r - M.r_key))) if len(M) else np.nan
    print(f"check: linear r vs refit_decoders_sessions.csv frozen r, {len(M)} session x target rows, max |diff| = {d:.2e}")
    return d


def trial_set_sentence(animals=None) -> str:
    """THE lick set of this driver's figure (Priya 11-09): the where-decoder licks of lick_refit_decoders.load_z
    (every coordinate finite), sessions with >= lr.MIN_LICKS, animals with >= 3 such pre sessions."""
    extra = ["every 3-D coordinate finite", "animals with >= 3 such pre sessions"]
    if animals:
        extra.append("animals " + ", ".join(animals) + " only")
    return lr.where_sets(lr.MIN_LICKS, extra=extra)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--animals", nargs="*")
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence(None if a.plot_only else a.animals))
    il.split_only(STEM)
    if plot_only():
        plot(load_fd(STEM))
        return 0
    P, H = predictions(a.animals, a.jobs)
    P = P.join(offsets(P))
    S = sessions(P)
    chk = check_linear(S)
    T = tests(S)
    PL = pre_levels(S)
    pd.set_option("display.width", 250)
    print("== hyperparameters (gamma_c x 1/n_features, alpha_a x n_train; edge = on the grid edge):")
    print(H.groupby(["set", "target"])[["gamma_c", "alpha_a", "edge"]].agg(["median", "min", "max"]).to_string())
    print("== held-out PRE r per model and the per-animal difference:")
    print(PL.round(4).to_string(index=False))
    print("== change vs pre (per animal, rule 8) + nested bootstrap:")
    print(T.round(4).to_string(index=False))
    cov = S.groupby(["set", "model", "target", "epoch"]).off_cov.mean().unstack("epoch")
    print("== share of licks with a like-for-like reference (mean of sessions):\n" + cov.round(3).to_string())
    W = il.OUT_W
    W.mkdir(parents=True, exist_ok=True)
    P.to_csv(W / "nonlinear_decoder_preds.csv.gz", index=False)
    H.to_csv(W / "nonlinear_decoder_hyper.csv", index=False)
    S.to_csv(W / "nonlinear_decoder_sessions.csv", index=False)
    T.to_csv(W / "nonlinear_decoder_tests.csv", index=False)
    PL.to_csv(W / "nonlinear_decoder_pre.csv", index=False)
    F = figure_data(S, T, PL, H)
    F["check_linear"] = chk
    save_fd(STEM, F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
