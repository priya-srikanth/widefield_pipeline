"""PIXEL MIXED MODELS on the per-session Haufe LEVEL maps, with TFCE (overnight 10-27 backlog 5e; spec
docs/status/SVD_GAUSS_SUPERSEDE_QUEUE_2026-10-27.md part D: "per pixel, time model (day-1 step map, recovery-per-e-fold
map) and behavior model (aim-deficit map), animal fixed intercepts solved for all pixels at once (descriptive per
mixed_models.py), significance by session-label permutation (pre/post within animal for the step; post days within
animal for recovery) + TFCE").

MAPS. The per-session self-fit Haufe maps of `lick_session_level_maps.py` exactly as it builds them (`patterns` read from
its cache session_levels/patterns_gap2_enc.pkl, `build_maps`: amplitude KEPT, each map / the animal's mean pre-session
RMS of the all-lick map of that set and target; sessions with decoder fit share < MIN_FIT dropped; drawn pixels = stat
mask on every 2nd row / column). Maps: LATERAL (ITI + cued x all + first licks) and the per-spout one-vs-rest maps (cued
+ ITI, all licks; canonical spout order). WFIELD_MAP_SCALE=frame (z-scored features) is honored as there (outputs "_z").

MODELS (one design matrix per map, shared by every pixel; closed-form OLS, all pixels at once). Rows = sessions, the
same covariates as `wfield_local.mixed_models` / `iti_mixed_models.py` (`add_time_covariates`: August pre only, post =
day >= 1, lday = log(days since stroke) with day 1 = 0; `add_deficit` with `iti_recovery_measure.aim_deficit()` =
animal's August-pre mean - session FIRST-lick weighted signed aim, degrees, + = aim shifted ipsilesional, 0 for pre;
WFIELD_AIM=all = all-lick sensitivity, outputs "_aimall"):
  TIME       map ~ animal + post + post:lday       -> STEP (post: level at day 1 vs pre) and RECOVERY (post:lday: change
                                                      per e-fold of days since stroke)
  BEHAVIOR  map ~ animal + post + post:deficit    -> AIM (post:deficit: change per degree of aim deficit, drawn per
                                                      10 deg) and POST_REC (post: change left at deficit 0 = full
                                                      behavioral recovery); mirrors mixed_models' behavior model
Animal = FIXED intercepts (4 dummies, no grand intercept): descriptive per mixed_models.py ("a per-pixel fixed-intercept
OLS can be drawn as a descriptive companion only"); random effects are not fitted per pixel. Sessions are weighted
equally (an animal with more sessions weighs more), unlike change_z's equal animal weights.
STATISTIC per pixel: t = coefficient / its OLS SE, SE = s * sqrt((X'X)^-1_jj), s = the residual SD (df n - p) floored at
its 10th percentile over pixels (floor fixed across permutations, as `change_z` / `level_z`). Why t and not the
per-animal-effect / pooled-SD form of iti_session_cluster_tests: the slopes (recovery, aim) have no per-animal
mean-difference form, and s IS that module's s_pooled generalized -- the within-animal session SD (around each animal's
fitted course instead of its group mean) pooled over all animals, df-weighted. For the step with no slope term t would
be change_z up to the animal weighting.
NULLS (session labels permuted WITHIN animal, the same permutation for every pixel, so the maps' spatial correlation is
kept; s recomputed per draw):
  step      the (post, lday) rows re-dealt among the animal's sessions (pre / post labels permuted, with their days)
  recovery  lday re-dealt among the animal's POST sessions (post labels kept: H0 no trend given the step)
  aim       deficit re-dealt among the animal's POST sessions
  post_rec  the (post, deficit) rows re-dealt among the animal's sessions
TFCE (E 0.5, H 2) on the t image, FWER p = share of N_PERM null draws whose max |TFCE| over cortex reaches the pixel's:
`iti_session_cluster_tests._fwer` (one definition, rule 9; DECISIONS 10-27 late "pixel map stats = FWER max-statistic").
Solid contour = FWER p < 0.05; dashed = also >= 3 animals' OWN coefficient (the same model fit to that animal alone)
shares the sign (the per-animal guard, as every other map family; no animal gate on drawing).
RULE 8 (printed before the tests): sessions per animal (pre / post, day range, deficit range) and, per map x term, each
animal's own coefficient map: its RMS, r with the pooled coefficient map, and the share of pixels with the pooled sign.
Inference is over sessions within THESE animals (fixed effects), as the level / change tests; serial correlation between
neighboring sessions is not modeled (t somewhat optimistic; the permutation null re-deals whole sessions, so it carries
the same caveat only through exchangeability within animal).
Outputs (<OUT_W> = il.OUT unless WFIELD_ITI_OUT_W redirects): <OUT_W>/session_levels/pixel_lmm_{tests,per_animal,
sessions}.csv, pixel_lmm_maps.npz (coef / t / tfce / p / agree per map x term on the half-resolution image + idx_img);
figures/lick_session_pixel_lmm_{time,behavior}.png (lateral rows) and lick_session_pixel_lmm_{time,behavior}_spout_
{cued,iti}.png; figure data figdata/<stem>.pkl. --plot-only redraws from the figure data alone.
Usage: [WFIELD_MAP_SCALE=frame] [WFIELD_AIM=all] python scripts/lick_session_pixel_lmm.py [--plot-only] [--perm N]
       [--jobs N (<= 8)] [--animals PS92,PS93] [--maps lat|spout|all] [--timing]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_recovery_measure as rm  # noqa: E402
import lick_session_level_maps as lm  # noqa: E402
from figdata import CHANGE_CMAP, PRE_CMAP, grid_fns, imaging_grid, load_fd, plot_only, save_fd, sym_lim  # noqa: E402
from iti_session_cluster_tests import _fwer, _img_fn  # noqa: E402

N_PERM = 1000
MAX_JOBS = 8
TAG = lm.ZS + rm.AIM_TAG
OUTD = il.OUT_W / f"session_levels{lm.ZS}"
FIG = il.OUT_W / "figures"
AIM_PER = 10.0                        # the aim slope is drawn per 10 deg of deficit
MODELS = {"time": ("step", "recovery"), "behaviour": ("post_rec", "aim")}
TERM = {"step": ("time", "post"), "recovery": ("time", "slope"), "post_rec": ("behaviour", "post"),
        "aim": ("behaviour", "slope")}
TERM_LAB = {"step": "day-1 STEP vs pre", "recovery": "RECOVERY per e-fold of days",
            "post_rec": "POST at full behavioral recovery (deficit 0)", "aim": f"AIM DEFICIT slope, per {AIM_PER:g} deg"}
SCALE_OF = {"step": 1.0, "recovery": 1.0, "post_rec": 1.0, "aim": AIM_PER}
LAT_KEYS = [f"{st}/{sub}/lat" for st in lm.SETS for sub in lm.SUBS]


def spout_keys(st: str) -> list[str]:
    return [f"{st}/all/sp_{p}" for p in lm.POS]


def stem(name: str) -> str:
    return f"lick_session_pixel_lmm_{name}{TAG}"


# ------------------------------------------------------------------------------------------------ the model


def design(cov: pd.DataFrame, model: str) -> np.ndarray:
    """Design matrix (sessions x [animal dummies (sorted), post, post:slope]); slope = lday (time) | deficit
    (behavior). cov has animal, post, lday, deficit."""
    an = sorted(cov.animal.unique())
    D = (cov.animal.to_numpy()[:, None] == np.array(an)[None, :]).astype(float)
    x = cov["lday" if model == "time" else "deficit"].to_numpy(float)
    post = cov.post.to_numpy(float)
    return np.column_stack([D, post, post * x])


def ols_t(X: np.ndarray, Y: np.ndarray, floor: float | None = None):
    """Closed-form OLS for all pixels at once: (B (p, pix), t (p, pix), s (pix,)). s = residual SD (df n - p), floored."""
    G = np.linalg.inv(X.T @ X)
    B = G @ X.T @ Y
    R = Y - X @ B
    s = np.sqrt((R ** 2).sum(0) / (len(X) - X.shape[1]))
    if floor is not None:
        s = np.maximum(s, floor)
    return B, B / (s[None, :] * np.sqrt(np.diag(G))[:, None]), s


def _shuffle(cov: pd.DataFrame, term: str, rng) -> pd.DataFrame:
    """One null draw of the covariates (module docstring NULLS), within animal; rows in their fixed (label) order."""
    c = cov.copy()
    for _a, ix in sorted(c.groupby("animal").indices.items()):
        if term in ("step", "post_rec"):
            cols = ["post", "lday"] if term == "step" else ["post", "deficit"]
            c.iloc[ix, [c.columns.get_loc(k) for k in cols]] = c.iloc[rng.permutation(ix)][cols].to_numpy()
        else:
            col = "lday" if term == "recovery" else "deficit"
            px = ix[c.post.to_numpy()[ix] == 1]
            c.iloc[px, c.columns.get_loc(col)] = c.iloc[rng.permutation(px)][col].to_numpy()
    return c


class Cell(tuple):
    """One fan-out unit (key, term, Y, cov, idx_img, n_perm); repr = 'key|term' so the progress log stays one line."""

    def __repr__(self) -> str:
        return f"{self[0]}|{self[1]}"


def coefs(Y: np.ndarray, cov: pd.DataFrame, term: str):
    """(pooled coefficient vector, {animal: its own coefficient vector}) of `term`, in drawn units (aim per AIM_PER deg);
    per animal = the same model fit to that animal's sessions alone (skipped when rank-deficient)."""
    model, which = TERM[term]
    j = -2 if which == "post" else -1
    pooled = np.linalg.lstsq(design(cov, model), Y, rcond=None)[0][j]
    per = {}
    for a, ix in sorted(cov.groupby("animal").indices.items()):
        Xa = design(cov.iloc[ix], model)
        if len(ix) > Xa.shape[1] and np.linalg.matrix_rank(Xa) == Xa.shape[1]:
            per[a] = np.linalg.lstsq(Xa, Y[ix], rcond=None)[0][j] * SCALE_OF[term]
    return pooled * SCALE_OF[term], per


def test_term(item):
    """(key, term, Y (sessions, drawn pixels), cov (label-sorted), idx_img, n_perm) -> result dict (t, TFCE, FWER p).
    Module-level (spawn)."""
    import zlib
    key, term, Y, cov, idx_img, n_perm = item
    model, which = TERM[term]
    j = -2 if which == "post" else -1
    rng = np.random.default_rng(zlib.crc32(f"{key}|{term}|pixel_lmm".encode()))
    img = _img_fn(idx_img)
    X = design(cov, model)
    floor = float(np.quantile(ols_t(X, Y)[2], 0.1))
    _B, t, _s = ols_t(X, Y, floor)
    T, p = _fwer(t[j], lambda: ols_t(design(_shuffle(cov, term, rng), model), Y, floor)[1][j], img, n_perm, "tfce")
    return dict(key=key, term=term, model=model, t=img(t[j]), tfce=T, p=p, n=cov.animal.nunique(), n_sess=len(cov),
                df=len(cov) - X.shape[1], floor=floor,
                n_sess_animal={a: (int((g.post == 0).sum()), int((g.post == 1).sum())) for a, g in cov.groupby("animal")})


# ------------------------------------------------------------------------------------------------ data


def load_maps(animals: list[str] | None, keys: list[str]):
    """(L from lick_session_level_maps.build_maps, covariates per label, idx_img, ol, pre level maps)."""
    from wfield_local import mixed_models as mx
    P = lm.patterns(MAX_JOBS)
    if animals:
        P = {x: r for x, r in P.items() if r["animal"] in animals}
    grid = imaging_grid()
    to_img, ol = grid_fns(grid)
    g = to_img(np.arange(len(grid["pix"]), dtype=float))
    ok = np.isfinite(g)
    sel = g[ok].astype(int)
    idx_img = np.full(g.shape, -1)
    idx_img[ok] = np.arange(ok.sum())
    L = lm.build_maps(P, sel)
    C = mx.add_time_covariates(pd.DataFrame({"label": sorted(P)}))
    C = mx.add_deficit(C, rm.aim_deficit())
    C = C.set_index("label")
    data = {}
    for k in keys:
        rows, Ys = [], []
        for a in sorted(L):
            v = L[a].get(k)
            if v is None:
                continue
            for i, x in enumerate(v["label"]):
                if x in C.index and np.isfinite(C.loc[x, "deficit"]):
                    rows.append(x)
                    Ys.append(v["M"][i])
        if not rows:
            continue
        o = np.argsort(rows)                              # defined order: by label (animal first)
        cov = C.loc[[rows[i] for i in o], ["animal", "epoch", "day", "post", "lday", "deficit"]].reset_index()
        Y = np.vstack([Ys[i] for i in o])
        pre = [Y[(cov.animal == a).to_numpy() & (cov.post == 0).to_numpy()] for a in sorted(cov.animal.unique())]
        pre = [m.mean(0) for m in pre if len(m)]
        data[k] = dict(Y=Y, cov=cov, pre=_img_fn(idx_img)(np.mean(pre, 0)) if pre else None, n_pre_an=len(pre))
    return data, idx_img, ol


def sessions_table(data: dict) -> pd.DataFrame:
    rows = []
    for k, d in data.items():
        for a, g in d["cov"].groupby("animal"):
            post = g[g.post == 1]
            rows.append(dict(key=k, animal=a, n_pre=int((g.post == 0).sum()), n_post=len(post),
                             day_min=float(post.day.min()) if len(post) else np.nan,
                             day_max=float(post.day.max()) if len(post) else np.nan,
                             lday_sd=float(post.lday.std()) if len(post) > 1 else np.nan,
                             deficit_min=float(post.deficit.min()) if len(post) else np.nan,
                             deficit_max=float(post.deficit.max()) if len(post) else np.nan))
    return pd.DataFrame(rows)


def per_animal_checks(data: dict) -> tuple[dict, pd.DataFrame]:
    """RULE 8, before any permutation: per map x term the pooled coefficient and each animal's own (`coefs`) -> ({(key,
    term): (pooled, {animal: vec})}, table of r with the pooled map over pixels, share of pixels with the pooled sign,
    RMS)."""
    Cf, rows = {}, []
    for k, d in sorted(data.items()):
        for term in TERM:
            pooled, per = coefs(d["Y"], d["cov"], term)
            Cf[(k, term)] = (pooled, per)
            for a, v in per.items():
                rows.append(dict(key=k, term=term, animal=a, rms=float(np.sqrt((v ** 2).mean())),
                                 r_pooled=float(np.corrcoef(v, pooled)[0, 1]),
                                 same_sign=float((np.sign(v) == np.sign(pooled)).mean())))
            rows.append(dict(key=k, term=term, animal="POOLED", rms=float(np.sqrt((pooled ** 2).mean())), r_pooled=1.0,
                             same_sign=1.0))
    return Cf, pd.DataFrame(rows)


def tests_table(R: dict) -> pd.DataFrame:
    rows = []
    for (k, term), r in R.items():
        cx = np.isfinite(r["p"])
        sig = (np.nan_to_num(r["p"], nan=1) < 0.05) & cx
        st, sub, tg = k.split("/")
        rows.append(dict(key=k, set=lm.SETLAB[st], sub=sub, target=tg, model=r["model"], term=term,
                         n_animals=r["n"], n_sessions=r["n_sess"], df=r["df"], n_perm=r["n_perm"],
                         frac_sig=float(sig.sum() / cx.sum()), frac_sig_pos=float((sig & (r["coef"] > 0)).sum() / cx.sum()),
                         frac_sig_neg=float((sig & (r["coef"] < 0)).sum() / cx.sum()),
                         frac_sig_agree3=float((sig & (np.nan_to_num(r["agree"]) >= 3)).sum() / cx.sum()),
                         min_p=float(np.nanmin(r["p"])), coef_rms=float(np.sqrt(np.nanmean(r["coef"] ** 2))),
                         max_abs_t=float(np.nanmax(np.abs(r["t"]))), s_floor=r["floor"]))
    return pd.DataFrame(rows)


def run(data: dict, idx_img: np.ndarray, n_perm: int, jobs: int) -> dict:
    from wfield_local import analysis_kit as ak
    items = [Cell((k, term, d["Y"], d["cov"], idx_img, n_perm)) for k, d in sorted(data.items())
             for m in MODELS.values() for term in m]
    print(f"{len(items)} map x term tests x {n_perm} permutations on {jobs} workers", flush=True)
    res, fails = ak.fan_sessions(items, test_term, jobs=jobs, key=lambda it: (it[0], it[1]), label="pixel lmm",
                                 log=lambda m: print(m[:110], flush=True))
    if fails:
        raise SystemExit(f"tests failed: {[(f[0][:2], str(f[1])[:200]) for f in fails[:5]]}")
    out = {}
    for _it, r in res:
        r["n_perm"] = n_perm
        out[(r["key"], r["term"])] = r
    return out


# ------------------------------------------------------------------------------------------------ figures


def _rowlab(k: str) -> str:
    st, sub, tg = k.split("/")
    if tg.startswith("sp_"):
        return f"{lm.SETLAB[st]} {tg[3:].replace('_', ' ')}"
    return f"{lm.SETLAB[st]} lateral, {sub} licks"


def _head(model: str, fam: str, n_perm: int, aim: str) -> str:
    what = ("LATERAL-position maps; rows = lick set x subset (FIRST = ITI: no lick in the preceding 2 s; cued: first "
            "post-cue lick)" if fam == "lat" else f"{lm.SETLAB[fam]} per-spout one-vs-rest maps (all licks), canonical "
            "spout order")
    if model == "time":
        mdl = ("TIME model per pixel: map ~ animal (fixed intercepts) + post + post:log(days since stroke) (day 1 = 0; "
               "as iti_mixed_models). Col 2 = day-1 STEP vs pre; col 3 = RECOVERY per e-fold of days (sign opposite to "
               "the step = return toward pre).")
        null = ("null: pre / post labels (with their days) re-dealt within animal (step); post days re-dealt within "
                "animal (recovery)")
    else:
        mdl = (f"BEHAVIOR model per pixel: map ~ animal (fixed intercepts) + post + post:aim deficit (deficit = animal's "
               f"pre - session {aim} weighted signed aim, deg, + = ipsilesional; 0 for pre). Col 2 = POST term = change "
               f"left at full behavioral recovery (deficit 0); col 3 = change per {AIM_PER:g} deg of aim deficit.")
        null = ("null: (pre / post, deficit) re-dealt within animal (post term); deficits re-dealt among the animal's "
                "post sessions (aim slope)")
    paras = [f"NEURAL: per-SESSION self-fit Haufe maps -- {what}; amplitude kept (/ animal's mean pre-session RMS of the "
             f"all-lick map). Col 1 = pre LEVEL (mean of animals, purple-orange, own scale).", mdl,
             f"SOLID = TFCE FWER p < 0.05 on t = coefficient / OLS SE ({null}; {n_perm} x, max |TFCE| over cortex). "
             "DASHED = also >= 3 animals' own coefficient (same model, that animal alone) shares the sign. Sessions = "
             "observations within these 4 animals (fixed effects; descriptive per mixed_models). "
             "Positive (red) = more active for licks landing more CONTRALESIONAL (lateral) / to THAT spout, per unit of "
             "the term. ipsi (lesion) = image LEFT. PROVISIONAL"]
    return "\n".join(textwrap.fill(q, 160) for q in paras)


def plot(D: dict) -> None:
    """D = R (results of this figure's maps x terms), pre ({key: pre level image}), keys, model, fam, ol, n_perm, aim,
    out (png name)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R, keys, model, ol = D["R"], [k for k in D["keys"] if any((k, t) in D["R"] for t in MODELS[D["model"]])], D["model"], D["ol"]
    terms = MODELS[model]
    fig, axs = plt.subplots(len(keys), 3, figsize=(11.5, 2.75 * len(keys) + 1.6), squeeze=False, layout="constrained")
    for i, k in enumerate(keys):
        lim_c = sym_lim([R[(k, t)]["coef"] for t in terms if (k, t) in R])
        im_l = im_c = None
        for j in range(3):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                pre = D["pre"].get(k)
                if pre is not None:
                    lim = sym_lim([pre])
                    im_l = ax.imshow(pre, cmap=PRE_CMAP, vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                r0 = R.get((k, terms[0]), {})
                npre = D.get("npre", {}).get(k, ("?", "?"))
                ax.set_title(f"pre (level): {npre[0]} pre sess / {npre[1]} an\nmodel: {r0.get('n_sess', '?')} sess / "
                             f"{r0.get('n', '?')} an", fontsize=7.5)
                ax.set_ylabel(_rowlab(k), fontsize=8.5)
                continue
            t = terms[j - 1]
            r = R.get((k, t))
            if r is None:
                ax.set_title(f"{t}: no data", fontsize=7.5)
                continue
            im_c = ax.imshow(r["coef"], cmap=CHANGE_CMAP, vmin=-lim_c, vmax=lim_c, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            fs = lm._sig_contours(ax, r)
            cx = np.isfinite(r["p"])
            sig = np.nan_to_num(r["p"], nan=1) < 0.05
            ax.set_title(f"{TERM_LAB[t]}\nsig {fs:.0f}% (+{100 * (sig & (r['coef'] > 0)).sum() / cx.sum():.0f} / "
                         f"-{100 * (sig & (r['coef'] < 0)).sum() / cx.sum():.0f}), min p {np.nanmin(r['p']):.3f}",
                         fontsize=7.5)
        if im_l is not None:
            cb = fig.colorbar(im_l, ax=axs[i, 0], fraction=0.06, pad=0.02)
            cb.set_label("pre LEVEL / pre RMS", fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
        if im_c is not None:
            cb = fig.colorbar(im_c, ax=axs[i, 1:].tolist(), fraction=0.03, pad=0.01)
            cb.set_label("coefficient / pre RMS (per unit of the term)", fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
    fig.suptitle(_head(model, D["fam"], D["n_perm"], D["aim"]), fontsize=7.5)
    FIG.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, FIG / D["out"], dpi=95)
    plt.close(fig)


FIGS = [(m, fam) for m in MODELS for fam in ("lat", "trial", "iti")]


def fig_name(model: str, fam: str) -> str:
    return stem(model if fam == "lat" else f"{model}_spout_{lm.SETLAB[fam].lower()}") + ".png"


def fam_keys(fam: str) -> list[str]:
    return LAT_KEYS if fam == "lat" else spout_keys(fam)


# ------------------------------------------------------------------------------------------------ main


def _arg(name: str, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> int:
    if plot_only():
        for m, fam in FIGS:
            try:
                D = load_fd(fig_name(m, fam))
            except SystemExit as e:
                print(e)
                continue
            plot(D)
        return 0
    t0 = time.time()
    n_perm = int(_arg("--perm", N_PERM))
    jobs = min(int(_arg("--jobs", MAX_JOBS)), MAX_JOBS)
    animals = _arg("--animals")
    animals = sorted(animals.split(",")) if animals else None
    which = _arg("--maps", "all")
    keys = (LAT_KEYS if which in ("lat", "all") else []) + \
           ([k for st in ("trial", "iti") for k in spout_keys(st)] if which in ("spout", "all") else [])
    data, idx_img, ol = load_maps(animals, keys)
    print(f"maps loaded: {len(data)} keys, {time.time() - t0:.0f} s; aim lick set = {rm.AIM}; scale = "
          f"{lm.SCALE or 'unscaled'}", flush=True)
    pd.set_option("display.width", 240)
    S = sessions_table(data)
    print("\n== RULE 8: sessions per animal in the models (pre / post, post day range, aim deficit range, deg)")
    print(S[S.key.isin(LAT_KEYS[:1] + [k for k in data if "sp_" in k][:1])].round(2).to_string(index=False))
    if "--timing" in sys.argv:                            # one cell, serial: seconds per permutation
        k = sorted(data)[0]
        t1 = time.time()
        test_term((k, "step", data[k]["Y"], data[k]["cov"], idx_img, n_perm))
        dt = time.time() - t1
        n_cells = len(data) * 4
        print(f"TIMING: one cell ({k}, step, {n_perm} perm) {dt:.1f} s -> {dt / n_perm:.3f} s / perm; full run "
              f"{n_cells} cells x {N_PERM} perm / {jobs} workers ~ {dt / n_perm * N_PERM * n_cells / jobs / 60:.1f} min "
              "(+ load)")
        return 0
    Cf, A = per_animal_checks(data)
    print("\n== RULE 8: per-animal coefficient maps (same model, that animal alone) vs the pooled map -- r over pixels "
          "(then share of pixels with the pooled sign)")
    print(A.pivot_table(index=["key", "term"], columns="animal", values="r_pooled", sort=False).round(2).to_string())
    print("\n== RULE 8: share of pixels where the animal's own coefficient has the pooled sign")
    print(A.pivot_table(index=["key", "term"], columns="animal", values="same_sign", sort=False).round(2).to_string())
    R = run(data, idx_img, n_perm, jobs)
    img = _img_fn(idx_img)
    for kk, r in R.items():
        pooled, per = Cf[kk]
        r["coef"] = img(pooled)
        agree = (np.sign(np.vstack(list(per.values()))) == np.sign(pooled)[None, :]).sum(0) if per else 0 * pooled
        r["agree"] = img(agree.astype(float))
        r["n_agree_animals"] = len(per)
    T = tests_table(R)
    print("\n== TFCE FWER tests: fraction of cortex p < 0.05 (+ / -), with >= 3 animals agreeing, min p")
    print(T[["key", "term", "n_animals", "n_sessions", "df", "frac_sig", "frac_sig_pos", "frac_sig_neg",
             "frac_sig_agree3", "min_p", "coef_rms", "max_abs_t"]].round(3).to_string(index=False))
    OUTD.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUTD / f"pixel_lmm_tests{TAG}.csv", index=False)
    A.to_csv(OUTD / f"pixel_lmm_per_animal{TAG}.csv", index=False)
    S.to_csv(OUTD / f"pixel_lmm_sessions{TAG}.csv", index=False)
    arr = {"idx_img": idx_img}
    for (k, term), r in R.items():
        for f in ("coef", "t", "tfce", "p", "agree"):
            arr[f"{k.replace('/', '__')}__{term}__{f}"] = r[f].astype(np.float32)
    np.savez_compressed(OUTD / f"pixel_lmm_maps{TAG}.npz", **arr)
    for m, fam in FIGS:
        ks = fam_keys(fam)
        # figure data = only what the figure draws (t / TFCE maps live in the npz), float32
        Rf = {kk: {f: (v[f].astype(np.float32) if f in ("coef", "p", "agree") else v[f])
                   for f in ("coef", "p", "agree", "n", "n_sess", "model")}
              for kk, v in R.items() if kk[0] in ks and kk[1] in MODELS[m]}
        if not Rf:
            continue
        D = dict(R=Rf, pre={k: data[k]["pre"] for k in ks if k in data}, keys=ks, model=m, fam=fam, ol=ol,
                 n_perm=n_perm, aim=rm.AIM_LABEL[rm.AIM], out=fig_name(m, fam),
                 npre={k: (int((data[k]["cov"].post == 0).sum()), data[k]["n_pre_an"]) for k in ks if k in data})
        save_fd(D["out"], D)
        plot(D)
    print(f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
