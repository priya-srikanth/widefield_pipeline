"""Statistics for the Haufe decoder-pattern maps (3.33; Priya 10-27: "can 3.33 have stats?").

3.33 shows one pattern per animal x epoch (decoder refit on the epoch's pooled licks), so there is nothing to resample.
Here every SESSION gets its own pattern, from a decoder that never saw it: the epoch's ridge decoder (pre-stroke SVD
basis, lags 0..0.2 s, as `iti_decoder_patterns`) is refit leaving that session out, applied to the session's licks,
and the session pattern is A_s = Cov(X_s, yhat_s) / Var(yhat_s), mapped to pixels (W @ A_s) and divided by the
animal's pooled pre pattern RMS (as 3.33). SIGNIFICANCE (DECISIONS 2026-10-27 late, Priya "2 a yes"): the decoder
page's FAMILY-WISE MAX-STATISTIC, from the shared machinery of `iti_session_cluster_tests.py` (rule 9: `change_z` /
`level_z`, `test_one` / `test_level` with stat "max", `max_fwer`), per map x epoch x kind over the cortex pixels drawn
on the figure (every 2nd row / column, `drawn_index`):
  change   epoch - pre: per animal (mean epoch - mean pre sessions) / within-animal session SD pooled over animals and
           groups, summed over animals / sqrt(k); null = session labels permuted between pre and the epoch within
           each animal (`N_PERM`, same permutation for every pixel); FWER p = share of draws whose max |z| over pixels
           reaches the pixel's.
  level    epoch map != 0: `level_z`, null = sign flip of whole session maps within animal. CAVEAT: a HELD-OUT pattern
           (Cov(X_s, X_s w) / Var) is not sign-symmetric under no information, so the level test (like the bootstrap
           level CI it replaces) says the map is non-zero, not that the session carries position.
SOLID outline = FWER p < 0.05; DASHED = also >= 3 animals (of 4) share the sign (an outline only; rule 8). The OLD
per-pixel test is kept for comparison in the CSV (frac_sig_fdr*: nested bootstrap animals > sessions, `N_BOOT` draws,
two-sided p, Benjamini-Hochberg q 0.05 over the stat mask; frac_sig_maxboot: the bootstrap-standardised max-statistic
drawn as the solid line before 10-27 late). Similarity to pre: per-animal r(epoch session-mean map, pre map) with a
session-bootstrap 95% CI (within animal), printed per animal.
Outputs <E>/iti_licks/decoder_pattern_stats.csv (per epoch / axis / set: method, fraction of the drawn cortex FWER-
significant (frac_sig, frac_sig_agree, min_p_fwer), the old FDR fractions, per-animal similarity CI),
figures/lick_decoder_pattern_stats_{lat,sag}.png, and the figure data figdata/lick_decoder_pattern_stats_{lat,sag}.pkl
(the test results for that axis + the stats table + the imaging grid). Usage:
python scripts/iti_decoder_pattern_stats.py [--plot-only] [--perm N]   (--plot-only redraws from figdata: no tests)
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_decoder_patterns as dp  # noqa: E402
import iti_licks as il  # noqa: E402
from figdata import (CHANGE_CMAP, CMAP_NOTE, PRE_CMAP, grid_fns, imaging_grid, levels_name, load_fd,  # noqa: E402
                     plot_only, save_fd, sym_lim)
import iti_where_decoder as wd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN, EP, AXES = dp.AN, dp.EP, dp.AXES
N_BOOT = 1000
N_PERM = 1000                                                # permutation draws of the FWER max-statistic
Q = 0.05
AGREE_MIN = 3                                                # outline: >= 3 of 4 animals share the sign (Priya 10-27)


def _ridge(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    return sc, RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(X), y)


def session_maps():
    """{(animal, set, axis): {epoch: (n_sessions, pixels) array}} of held-out session patterns / pre RMS."""
    D = wd.load("svdbasis")
    out = {}
    for a in AN:
        W = np.asarray(pickle.loads((dp.SPATIAL / f"spatial_{a}.pkl").read_bytes())["P"], np.float32)
        for st in ("iti", "trial"):
            S = {}
            for x, d in sorted(D.items()):
                if d["animal"] != a:
                    continue
                X, C = d[st]
                ok = np.isfinite(C[:, [0, 3]]).all(1)
                if ok.sum() >= 20:
                    S[x] = (d["epoch"], X[ok], C[ok])
            by = {ep: sorted(x for x, v in S.items() if v[0] == ep) for ep in EP}
            if len(by["pre"]) < 3:
                continue
            for ax_, k in AXES.items():
                pre_rms = float(np.sqrt(np.mean((W @ dp.pattern(*dp._stack(S, by["pre"], k))) ** 2)))
                res = {}
                for ep in EP:
                    if len(by[ep]) < 2:
                        continue
                    rows = []
                    for x in by[ep]:
                        sc, m = _ridge(*dp._stack(S, [q for q in by[ep] if q != x], k))
                        X = S[x][1]
                        yh = m.predict(sc.transform(X))
                        A = ((X - X.mean(0)).T @ (yh - yh.mean())) / (len(yh) - 1) / yh.var(ddof=1)
                        rows.append((W @ A.astype(np.float32)) / pre_rms)
                    res[ep] = np.vstack(rows)
                out[(a, st, ax_)] = res
                print(f"{a} {st} {ax_}: " + ", ".join(f"{e} {v.shape[0]}" for e, v in res.items()), flush=True)
    return out


def _boot(per_animal: dict, rng, paired_ref: dict | None = None) -> np.ndarray:
    """(N_BOOT, pixels) of the mean-of-animals map (minus the paired pre map if given), animals > sessions."""
    an = sorted(per_animal)
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(an, size=len(an), replace=True)
        acc = 0
        for a in pick:
            M = per_animal[a]
            v = M[rng.integers(0, len(M), len(M))].mean(0)
            if paired_ref is not None:
                R = paired_ref[a]
                v = v - R[rng.integers(0, len(R), len(R))].mean(0)
            acc = acc + v
        draws.append(acc / len(pick))
    return np.asarray(draws)


def _maxstat(B: np.ndarray, point: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """FWER (max-statistic) significance, as the decoder page's Haufe maps: standardise each bootstrap draw per pixel
    (centred on the bootstrap mean, / bootstrap SD), take the max |t| over pixels per draw, threshold = its (1 - alpha)
    quantile; a pixel is significant if |point| / SD exceeds it."""
    sd = B.std(0)
    sd = np.where(sd > 0, sd, np.inf)
    tmax = np.abs((B - B.mean(0)) / sd).max(1)
    return np.abs(point) / sd > np.quantile(tmax, 1 - alpha)


def method_text(n_perm: int) -> str:
    """The significance method, as recorded in the CSV `method` column and the figure subtitle."""
    return (f"FWER max-statistic (iti_session_cluster_tests change_z / level_z; max |z| over the drawn cortex pixels; "
            f"change: session labels permuted within animal, level: session sign-flip; {n_perm} draws)")


def fwer_maps(cells: dict, n_perm: int, prefix: str, sel: np.ndarray, npix: int) -> dict:
    """FWER max-statistic per cell, via the SHARED tests (iti_session_cluster_tests.test_cell, stat "max"; rule 9).
    cells = {cell key (..., epoch, kind): data} with data {animal: (pre (n0, npix), epoch (n1, npix))} for "change" and
    {animal: (n, npix)} for "level". Tested on the drawn pixels `sel`; returns {cell key: dict(z, p)} as full npix
    vectors (NaN off `sel`). Cells fan out over <= 8 workers (results re-keyed, so completion order cannot matter)."""
    from iti_session_cluster_tests import test_cell

    from wfield_local import parallel
    items, back = [], {}
    for ck, data in cells.items():
        ep, kind = ck[-2], ck[-1]
        key = prefix + "|" + "|".join(str(x) for x in ck[:-2])
        if kind == "change":
            d = {a: (np.asarray(v[0][:, sel]), np.asarray(v[1][:, sel])) for a, v in sorted(data.items())}
        else:
            d = {a: np.asarray(v[:, sel]) for a, v in sorted(data.items())}
        items.append((kind, (key, ep, d, np.arange(len(sel)), n_perm, "max")))
        back[(key, ep, kind)] = ck
    res, fails = parallel.fan_out(items, test_cell, jobs=parallel.default_jobs(8), label="map cell",
                                  log=lambda m: print(m[:110], flush=True))
    res = sorted(res, key=lambda kv: repr((kv[0][0], kv[0][1][:2])))   # + kind: no ties            # completion order -> cell order
    if fails:
        raise RuntimeError(f"FWER cells failed: {[f[0][1][:2] for f in fails]}")
    out = {}
    for _it, r in res:
        z, p = np.full(npix, np.nan), np.full(npix, np.nan)
        z[sel], p[sel] = r["z"], r["p"]
        out[back[(r["key"], r["epoch"], r["kind"])]] = dict(z=z, p=p)
    return out


def old_fdr(per_animal: dict, rng, ref: dict | None) -> dict:
    """The OLD per-pixel test, kept for comparison (until 10-27 late): nested bootstrap (animals > sessions), two-sided
    p, BH q 0.05 (sig_fdr), and the bootstrap-standardised max-statistic (sig_maxboot)."""
    am = np.vstack([per_animal[a].mean(0) - (ref[a].mean(0) if ref else 0) for a in sorted(per_animal)])
    point = am.mean(0)
    B = _boot(per_animal, rng, ref)
    p = ak.boot_p_columns(B)                                 # shared p (decision 9, Priya 11-05): floored at 1/N
    return dict(am=am, point=point, sig_fdr=ak.bh(p, Q), sig_maxboot=_maxstat(B, point))


def merge_fwer(res: dict, F: dict, sel: np.ndarray, alpha: float = 0.05) -> dict:
    """Per cell: sig = FWER p < alpha (PRIMARY) and the summary fractions (over the drawn pixels for FWER, over the
    stat mask for the old FDR columns, as before)."""
    summ = {}
    for ck, r in res.items():
        f = F[ck]
        r["z"], r["p_fwer"] = f["z"], f["p"]
        r["sig"] = np.nan_to_num(f["p"], nan=1.0) < alpha
        agr = r["agree"] >= min(AGREE_MIN, r["n"])
        summ[ck] = dict(frac_sig=float(r["sig"][sel].mean()), frac_sig_agree=float((r["sig"] & agr)[sel].mean()),
                        min_p_fwer=float(np.nanmin(f["p"])), frac_sig_fdr=float(r["sig_fdr"].mean()),
                        frac_sig_fdr_agree=float((r["sig_fdr"] & agr).mean()),
                        frac_sig_maxboot=float(r["sig_max"].mean()))
    return summ


def test(SM: dict, n_perm: int = N_PERM):
    rng = np.random.default_rng(0)
    res, cells, rows, meta = {}, {}, [], {}
    for st in ("iti", "trial"):
        for ax_ in AXES:
            have = {a: SM[(a, st, ax_)] for a in AN if (a, st, ax_) in SM}
            pre = {a: v["pre"] for a, v in have.items()}
            for ep in EP:
                lvl = {a: v[ep] for a, v in have.items() if ep in v}
                for kind, ref in (("level", None), ("change", pre)):
                    if kind == "change" and ep == "pre":
                        continue
                    lvl_ = {a: m for a, m in lvl.items() if a in pre} if kind == "change" else lvl
                    o = old_fdr(lvl_, rng, ref)
                    agree = (np.sign(o["am"]) == np.sign(o["point"])[None, :]).sum(0)
                    ck = (st, ax_, ep, kind)
                    res[ck] = dict(point=o["point"], agree=agree, n=len(o["am"]), sig_fdr=o["sig_fdr"],
                                   sig_max=o["sig_maxboot"])
                    cells[ck] = {a: (pre[a], lvl_[a]) for a in lvl_} if kind == "change" else dict(lvl_)
                    meta[ck] = dict(set=st, axis=ax_, epoch=ep, kind=kind, n_animals=len(o["am"]))
            for a, v in have.items():                              # within-animal similarity to pre, session bootstrap
                P = v["pre"].mean(0)
                for ep in EP[1:]:
                    if ep not in v:
                        continue
                    M = v[ep]
                    bs = [np.corrcoef(M[rng.integers(0, len(M), len(M))].mean(0), P)[0, 1] for _ in range(200)]
                    rows.append(dict(set=st, axis=ax_, epoch=ep, kind="similarity", animal=a,
                                     r=float(np.corrcoef(M.mean(0), P)[0, 1]), lo=float(np.percentile(bs, 2.5)),
                                     hi=float(np.percentile(bs, 97.5))))
    from iti_session_cluster_tests import drawn_index
    npix = len(next(iter(res.values()))["point"])
    sel = drawn_index(grid_fns(imaging_grid())[0], npix)
    summ = merge_fwer(res, fwer_maps(cells, n_perm, "decoder_pattern", sel, npix), sel)
    stat_rows = [dict(**meta[ck], method=method_text(n_perm), n_perm=n_perm, **summ[ck]) for ck in res]
    return res, pd.DataFrame(stat_rows + rows)


def figdata(res: dict, T: pd.DataFrame, n_perm: int = N_PERM) -> dict:
    """{axis: figure data} -- everything `plot` draws for lick_decoder_pattern_stats_<axis>.png (+ the method)."""
    grid = imaging_grid()
    return {ax_: dict(axis=ax_, res={k: v for k, v in res.items() if k[1] == ax_}, T=T, grid=grid,
                      method=method_text(n_perm)) for ax_ in AXES}


def n_perm_arg() -> int:
    return int(sys.argv[sys.argv.index("--perm") + 1]) if "--perm" in sys.argv else N_PERM


def plot(D: dict, levels: bool = False) -> None:
    """D = one axis's figure data (`figdata`). Main figure: pre LEVEL map (PRE_CMAP, own scale) + acute / subacute /
    chronic - pre CHANGE maps (CHANGE_CMAP, one scale per row). levels=True: the supplementary <stem>_levels.png with
    the four epoch LEVEL maps on one scale per row (PRE_CMAP), same outlines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res, T, ax_ = D["res"], D["T"], D["axis"]
    fw = "method" in D                                  # FWER max-statistic figure data (10-27 late)
    to_img, ol = grid_fns(D["grid"])
    axname = {"lat": "LATERAL position (+ = contralesional)", "sag": "SAGITTAL extension (+ = further out)"}
    cols = list(EP) if levels else ["pre", "acute - pre", "subacute - pre", "chronic - pre"]
    fig, axs = plt.subplots(2, len(cols), figsize=(3.0 * len(cols) + 1.2, 6.9), squeeze=False, layout="constrained")
    for i, st in enumerate(("iti", "trial")):
        # the level scale as before (99th percentile over the four epoch levels of the row); pre in the main figure
        # gets its own scale, the change columns keep the row's shared level-derived scale
        lim = np.nanpercentile(np.abs(np.concatenate([res[(st, ax_, e, "level")]["point"]
                                                      for e in EP if (st, ax_, e, "level") in res])), 99)
        lim_pre = sym_lim([res[(st, ax_, "pre", "level")]["point"]]) if (st, ax_, "pre", "level") in res else lim
        im_l = im_c = None
        for j, c in enumerate(cols):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            key = (st, ax_, c, "level") if c in EP else (st, ax_, c.split(" ")[0], "change")
            if key not in res:
                ax.axis("off")
                continue
            r = res[key]
            if key[3] == "level":
                lv = lim if levels else lim_pre
                im_l = ax.imshow(to_img(r["point"]), cmap=PRE_CMAP, vmin=-lv, vmax=lv, interpolation="nearest")
            else:
                im_c = ax.imshow(to_img(r["point"]), cmap=CHANGE_CMAP, vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            # FWER figure data (10-27 late): solid = r["sig"] (max-statistic FWER); older figure data: solid = the
            # bootstrap max-statistic, dashed = BH (r["sig"] was BH then)
            solid = r["sig"] if fw else r.get("sig_max", np.zeros_like(r["sig"]))
            s_img = to_img((r["sig"] & (r["agree"] >= min(AGREE_MIN, r["n"]))).astype(float), 0)
            ax.contour(s_img, levels=[0.5], colors="k", linewidths=0.7, linestyles="--")
            ax.contour(to_img(solid.astype(float), 0), levels=[0.5], colors="k", linewidths=1.3)
            g = T[(T.set == st) & (T.axis == ax_) & (T.epoch == key[2]) & (T.kind == key[3])]
            fs = float(g.frac_sig.iloc[0]) if len(g) else np.nan
            fa = float(g.frac_sig_agree.iloc[0]) if len(g) else np.nan
            name = f"{c} (level)" if key[3] == "level" else c
            ax.set_title(f"{name} (n={r['n']})\n{'FWER ' if fw else ''}sig {100 * fs:.0f}% of cortex ({100 * fa:.0f}% "
                         f"also >=3/{r['n']} agree)", fontsize=7.5)
            if j == 0:
                ax.set_ylabel("ITI licks" if st == "iti" else "cued licks", fontsize=9)
        if levels:
            if im_l is not None:
                fig.colorbar(im_l, ax=axs[i, :].tolist(), fraction=0.02, pad=0.01,
                             label="epoch LEVEL: pattern / animal's pre RMS")
        else:
            if im_l is not None:
                fig.colorbar(im_l, ax=axs[i, 0], fraction=0.06, pad=0.02, label="pre LEVEL (pattern / pre RMS)")
            if im_c is not None:
                fig.colorbar(im_c, ax=axs[i, 1:].tolist(), fraction=0.025, pad=0.01,
                             label="CHANGE epoch - pre (pattern / pre RMS)")
    import textwrap
    head = (f"SUPPLEMENTARY to lick_decoder_pattern_stats_{ax_}.png: epoch LEVEL maps (not changes), one scale per row "
            "(purple-orange). " if levels else f"STATS for the decoder pattern maps -- {CMAP_NOTE}. ")
    paras = [head + f"NEURAL: Haufe pattern of the {axname[ax_]} decoder, one held-out pattern per SESSION (epoch decoder "
             "refit leaving the session out), / animal's pre RMS, mean of animals.",
             (f"SOLID outline = FWER p < 0.05 by the {D['method']}; DASHED = also >= 3 animals (of 4) share the sign. "
              "Old per-pixel BH-FDR result kept in the CSV for comparison." if fw else
              f"SOLID outline = max-statistic FWER 0.05 over cortex (as the decoder page); DASHED = 95% nested bootstrap "
              f"CI (animals > sessions, {N_BOOT} draws) excludes 0 after Benjamini-Hochberg q 0.05 AND >= 3 animals (of "
              "4) agree in sign.") + ("" if levels else " Change columns = epoch - pre, paired by animal."),
             "ipsi (lesion) = image LEFT. Unscaled features. PROVISIONAL"]
    fig.suptitle("\n".join(textwrap.fill(q, 150) for q in paras), fontsize=9)
    out = f"lick_decoder_pattern_stats_{ax_}.png"
    save_png_svg(fig, il.OUT / "figures" / (levels_name(out) if levels else out), dpi=105, bbox_inches="tight")
    plt.close(fig)


def plot_all(D: dict) -> None:
    """Main figure + the supplementary all-levels figure."""
    plot(D)
    plot(D, levels=True)


def main() -> int:
    if plot_only():
        for ax_ in AXES:
            plot_all(load_fd(f"lick_decoder_pattern_stats_{ax_}"))
        return 0
    cache = il.OUT / "decoder_pattern_sessions.pkl"
    if cache.exists():
        SM = pickle.loads(cache.read_bytes())
    else:
        SM = session_maps()
        cache.write_bytes(pickle.dumps(SM, protocol=pickle.HIGHEST_PROTOCOL))
    n_perm = n_perm_arg()
    res, T = test(SM, n_perm)
    T.to_csv(il.OUT / "decoder_pattern_stats.csv", index=False)
    pd.set_option("display.width", 220)
    sim = T[T.kind == "similarity"]
    print("== per animal (rule 8): similarity of the session-mean epoch map to pre, within-animal 95% CI")
    print(sim.assign(s=sim.apply(lambda r: f"{r.r:.2f} [{r.lo:.2f},{r.hi:.2f}]", axis=1))
          .pivot_table(index=["set", "axis", "animal"], columns="epoch", values="s", aggfunc="first")[EP[1:]].to_string())
    print(f"\n== fraction of cortex significant: FWER max-statistic ({n_perm} permutations; frac_sig, frac_sig_agree = "
          "also >= 3 animals same sign) beside the OLD per-pixel bootstrap BH q 0.05 (frac_sig_fdr*)")
    print(T[T.kind != "similarity"][["set", "axis", "epoch", "kind", "n_animals", "frac_sig", "frac_sig_agree", "min_p_fwer",
                                     "frac_sig_fdr", "frac_sig_fdr_agree"]].round(3).to_string(index=False))
    for ax_, D in figdata(res, T, n_perm).items():
        save_fd(f"lick_decoder_pattern_stats_{ax_}", D)
        plot_all(D)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
