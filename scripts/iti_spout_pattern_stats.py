"""Statistics for the PER-SPOUT Haufe maps (3.35; Priya 10-27: "can we add stats to 3.35 too").

Same construction as `iti_decoder_pattern_stats.py`, for the one-vs-rest spout decoders of `iti_spout_patterns.py`
(cued licks, the animal's pre-stroke SVD basis, lags 0..0.2 s): each SESSION gets a held-out pattern (the epoch's
one-vs-rest ridge decoder refit leaving that session out, applied to the session's licks, A_s = Cov(X_s, yhat_s) /
Var(yhat_s)), mapped to pixels and divided by the RMS of the animal's pooled pre map for that spout (as 3.35).
SIGNIFICANCE (DECISIONS 2026-10-27 late): the FWER max-statistic of `iti_decoder_pattern_stats.py` (shared machinery of
`iti_session_cluster_tests.py`: change_z / level_z, session-label permutation within animal for change, session
sign-flip for level, max |z| over the drawn cortex pixels per spout x epoch x kind; the level-test caveat for held-out
patterns applies). SOLID = FWER p < 0.05; DASHED = also >= 3 animals (of 4) agree in sign (outline only). The OLD per-
pixel nested bootstrap + BH q 0.05 is kept in the CSV (frac_sig_fdr*, frac_sig_maxboot) for comparison. Per animal x
spout: similarity of the session-mean epoch map to the pre map with a within-animal session-bootstrap 95% CI.
--iti: the same for ITI (self-initiated) licks labelled by the PRECEDING trial's spout (`iti_pos`; Priya 10-27: "map
distinguishing ITI licks based on preceding trial's spout location").
Outputs <E>/iti_licks/spout_pattern_stats[_iti].csv, figures/lick_cued_spout_pattern_stats.png / lick_iti_spout_pattern_stats.png. Usage:
python scripts/iti_spout_pattern_stats.py [--iti] [--plot-only] [--perm N]
Figure data: figdata/lick_{cued,iti}_spout_pattern_stats.pkl (test results, stats table, imaging grid, set); --plot-only
redraws from it alone (no tests). The CSV records the method (column `method`).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import (CHANGE_CMAP, CMAP_NOTE, PRE_CMAP, grid_fns, imaging_grid, levels_name, load_fd,  # noqa: E402
                     plot_only, save_fd, sym_lim)
import iti_where_decoder as wd  # noqa: E402
from iti_decoder_pattern_stats import (AGREE_MIN, N_BOOT, N_PERM, _ridge, fwer_maps, merge_fwer,  # noqa: E402
                                       method_text, n_perm_arg, old_fdr)
from iti_decoder_patterns import SPATIAL, pattern  # noqa: E402
from iti_spout_patterns import AN, EP, POS  # noqa: E402


def session_maps(st: str = "trial"):
    """{(animal, spout): {epoch: (n_sessions, pixels)}} of held-out one-vs-rest session patterns / pre RMS."""
    from wfield_local import orofacial_cohort as oc
    D = wd.load("svdbasis")
    out = {}
    for a in AN:
        W = np.asarray(pickle.loads((SPATIAL / f"spatial_{a}.pkl").read_bytes())["P"], np.float32)
        S = {}
        for x, d in sorted(D.items()):
            if d["animal"] != a:
                continue
            X, _C = d[st]
            P = np.array([p if p in POS or "_" not in p else oc.pos_key(p, "L") for p in d[f"{st}_pos"]])
            ok = np.isin(P, POS)
            if ok.sum() >= 30:
                S[x] = (d["epoch"], X[ok], P[ok])
        by = {ep: sorted(x for x, v in S.items() if v[0] == ep) for ep in EP}
        if len(by["pre"]) < 3:
            continue
        for sp in POS:
            Xp = np.vstack([S[x][1] for x in by["pre"]])
            yp = np.concatenate([(S[x][2] == sp).astype(float) for x in by["pre"]])
            if not 0 < yp.sum() < len(yp):
                continue
            rms = float(np.sqrt(np.mean((W @ pattern(Xp, yp).astype(np.float32)) ** 2)))
            res = {}
            for ep in EP:
                rows = []
                for x in by[ep]:
                    trn = [q for q in by[ep] if q != x]
                    if not trn:
                        continue
                    Xt = np.vstack([S[q][1] for q in trn])
                    yt = np.concatenate([(S[q][2] == sp).astype(float) for q in trn])
                    if not 0 < yt.sum() < len(yt):
                        continue
                    sc, m = _ridge(Xt, yt)
                    X = S[x][1]
                    yh = m.predict(sc.transform(X))
                    if yh.var() <= 0:
                        continue
                    A = ((X - X.mean(0)).T @ (yh - yh.mean())) / (len(yh) - 1) / yh.var(ddof=1)
                    rows.append((W @ A.astype(np.float32)) / rms)
                if len(rows) >= 2:
                    res[ep] = np.vstack(rows)
            out[(a, sp)] = res
            print(f"{a} {sp}: " + ", ".join(f"{e} {v.shape[0]}" for e, v in res.items()), flush=True)
    return out


def test(SM: dict, n_perm: int = N_PERM, st: str = "trial"):
    rng = np.random.default_rng(0)
    res, cells, rows, meta = {}, {}, [], {}
    for sp in POS:
        have = {a: SM[(a, sp)] for a in AN if (a, sp) in SM and "pre" in SM[(a, sp)]}
        pre = {a: v["pre"] for a, v in have.items()}
        for ep in EP:
            lvl = {a: v[ep] for a, v in have.items() if ep in v}
            for kind in ("level", "change"):
                if kind == "change" and ep == "pre":
                    continue
                ref = pre if kind == "change" else None
                o = old_fdr(lvl, rng, ref)
                agree = (np.sign(o["am"]) == np.sign(o["point"])[None, :]).sum(0)
                ck = (sp, ep, kind)
                res[ck] = dict(point=o["point"], agree=agree, n=len(o["am"]), sig_fdr=o["sig_fdr"],
                               sig_max=o["sig_maxboot"])
                cells[ck] = {a: (pre[a], lvl[a]) for a in lvl} if kind == "change" else dict(lvl)
                meta[ck] = dict(spout=sp, epoch=ep, kind=kind, n_animals=len(o["am"]))
        for a, v in have.items():
            P = v["pre"].mean(0)
            for ep in EP[1:]:
                if ep not in v:
                    continue
                M = v[ep]
                bs = [np.corrcoef(M[rng.integers(0, len(M), len(M))].mean(0), P)[0, 1] for _ in range(200)]
                rows.append(dict(spout=sp, epoch=ep, kind="similarity", animal=a, r=float(np.corrcoef(M.mean(0), P)[0, 1]),
                                 lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
    from iti_session_cluster_tests import drawn_index
    npix = len(next(iter(res.values()))["point"])
    sel = drawn_index(grid_fns(imaging_grid())[0], npix)
    summ = merge_fwer(res, fwer_maps(cells, n_perm, f"spout_pattern_{st}", sel, npix), sel)
    stat_rows = [dict(**meta[ck], method=method_text(n_perm), n_perm=n_perm, **summ[ck]) for ck in res]
    return res, pd.DataFrame(stat_rows + rows)


def fig_name(st: str) -> str:
    return "lick_iti_spout_pattern_stats.png" if st == "iti" else "lick_cued_spout_pattern_stats.png"


def plot(D: dict, levels: bool = False) -> None:
    """D = figure data: res (test results), T (stats table), st (set), grid (imaging grid). Main figure: pre LEVEL map
    (PRE_CMAP, own scale) + acute / subacute / chronic - pre CHANGE maps (CHANGE_CMAP). levels=True: the supplementary
    <stem>_levels.png with the four epoch LEVEL maps on one shared scale (PRE_CMAP), same outlines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res, T, st = D["res"], D["T"], D["st"]
    fw = "method" in D                                  # FWER max-statistic figure data (10-27 late)
    to_img, ol = grid_fns(D["grid"])
    cols = list(EP) if levels else ["pre", "acute - pre", "subacute - pre", "chronic - pre"]
    fig, axs = plt.subplots(len(POS), len(cols), figsize=(2.75 * len(cols) + 1.4, 2.5 * len(POS) + 1.2), squeeze=False,
                            layout="constrained")
    # level scale as before (all epoch levels of the figure); the change columns keep it, pre (main) gets its own
    lim = np.nanpercentile(np.abs(np.concatenate([r["point"] for k, r in res.items() if k[2] == "level"])), 99)
    lim_pre = sym_lim([r["point"] for k, r in res.items() if k[1] == "pre" and k[2] == "level"])
    im_l = im_c = None
    for i, sp in enumerate(POS):
        for j, c in enumerate(cols):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            key = (sp, c, "level") if c in EP else (sp, c.split(" ")[0], "change")
            if key not in res:
                ax.axis("off")
                continue
            r = res[key]
            if key[2] == "level":
                lv = lim if levels else lim_pre
                im_l = ax.imshow(to_img(r["point"]), cmap=PRE_CMAP, vmin=-lv, vmax=lv, interpolation="nearest")
            else:
                im_c = ax.imshow(to_img(r["point"]), cmap=CHANGE_CMAP, vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            # FWER figure data: solid = r["sig"] (max-statistic FWER); older figure data: solid = bootstrap max-stat
            solid = r["sig"] if fw else r.get("sig_max", np.zeros_like(r["sig"]))
            ax.contour(to_img((r["sig"] & (r["agree"] >= min(AGREE_MIN, r["n"]))).astype(float), 0), levels=[0.5], colors="k",
                       linewidths=0.7, linestyles="--")
            ax.contour(to_img(solid.astype(float), 0), levels=[0.5], colors="k", linewidths=1.3)
            g = T[(T.spout == sp) & (T.epoch == key[1]) & (T.kind == key[2])]
            sim = T[(T.spout == sp) & (T.epoch == key[1]) & (T.kind == "similarity")]
            sub = f"\nsim to pre {sim.r.mean():.2f}" if key[1] in EP[1:] and len(sim) else ""
            name = f"{c} (level)" if key[2] == "level" else c
            ax.set_title(f"{name} (n={r['n']}) {'FWER ' if fw else ''}sig {100 * float(g.frac_sig.iloc[0]):.0f}% "
                         f"({100 * float(g.frac_sig_agree.iloc[0]):.0f}% >=3/{r['n']}){sub}" if fw else
                         f"{name} (n={r['n']}) sig {100 * float(g.frac_sig_agree.iloc[0]):.0f}%{sub}", fontsize=7)
            if j == 0:
                ax.set_ylabel(sp.replace("_", " "), fontsize=9)
    if levels:
        if im_l is not None:
            fig.colorbar(im_l, ax=axs.ravel().tolist(), fraction=0.02, pad=0.01, shrink=0.5,
                         label="epoch LEVEL: pattern / animal's pre RMS (spout)")
    else:
        if im_l is not None:
            fig.colorbar(im_l, ax=axs[:, 0].tolist(), fraction=0.06, pad=0.02, shrink=0.5,
                         label="pre LEVEL (pattern / pre RMS)")
        if im_c is not None:
            fig.colorbar(im_c, ax=axs[:, 1:].ravel().tolist(), fraction=0.025, pad=0.01, shrink=0.5,
                         label="CHANGE epoch - pre (pattern / pre RMS)")
    import textwrap
    what = ("ITI (self-initiated) licks labelled by the PRECEDING trial's spout" if st == "iti" else "cued licks")
    head = (f"SUPPLEMENTARY to {fig_name(st)}: epoch LEVEL maps (not changes), one shared scale (purple-orange). "
            if levels else ("ITI licks by PRECEDING spout" if st == "iti" else "STATS for the per-spout maps")
            + f" -- {CMAP_NOTE}. ")
    paras = [head + f"NEURAL: per-spout one-vs-rest Haufe maps ({what}), one held-out pattern per SESSION, / animal's pre "
             "RMS for the spout, mean of animals.",
             (f"SOLID outline = FWER p < 0.05 by the {D['method']}; DASHED = also >= 3 animals (of 4) agree in sign; "
              "'sig %' = solid fraction of cortex (dashed fraction in brackets). Old per-pixel BH-FDR kept in the CSV."
              if fw else
              f"SOLID outline = max-statistic FWER 0.05 (as the decoder page); DASHED = 95% nested bootstrap CI (animals > "
              f"sessions, {N_BOOT} draws) excludes 0 after BH q 0.05 AND >= 3 animals (of 4) agree in sign; 'sig %' = the "
              "dashed fraction of cortex."),
             ("" if levels else "Change columns = epoch - pre, paired by animal; ")
             + "sim = r(session-mean epoch map, pre map), mean of animals. ipsi (lesion) = image LEFT. PROVISIONAL"]
    fig.suptitle("\n".join(textwrap.fill(q, 120) for q in paras), fontsize=9)
    out = fig_name(st)
    save_png_svg(fig, il.OUT / "figures" / (levels_name(out) if levels else out), dpi=95, bbox_inches="tight")
    plt.close(fig)


def plot_all(D: dict) -> None:
    """Main figure + the supplementary all-levels figure."""
    plot(D)
    plot(D, levels=True)


def main(argv=None) -> int:
    st = "iti" if "--iti" in (argv if argv is not None else sys.argv[1:]) else "trial"
    sfx = "_iti" if st == "iti" else ""
    if plot_only():
        plot_all(load_fd(fig_name(st)))
        return 0
    cache = il.OUT / f"spout_pattern_sessions{sfx}.pkl"
    if cache.exists():
        SM = pickle.loads(cache.read_bytes())
    else:
        SM = session_maps(st)
        cache.write_bytes(pickle.dumps(SM, protocol=pickle.HIGHEST_PROTOCOL))
    n_perm = n_perm_arg()
    res, T = test(SM, n_perm, st)
    T.to_csv(il.OUT / f"spout_pattern_stats{sfx}.csv", index=False)
    pd.set_option("display.width", 220)
    sim = T[T.kind == "similarity"]
    print("== per animal (rule 8): similarity of the session-mean epoch map to pre, within-animal 95% CI")
    print(sim.assign(s=sim.apply(lambda r: f"{r.r:.2f} [{r.lo:.2f},{r.hi:.2f}]", axis=1))
          .pivot_table(index=["spout", "animal"], columns="epoch", values="s", aggfunc="first")
          .reindex(columns=EP[1:]).to_string())
    print(f"\n== fraction of cortex significant: FWER max-statistic ({n_perm} permutations; frac_sig_agree = also >= 3 "
          "animals same sign) beside the OLD per-pixel bootstrap BH q 0.05 (frac_sig_fdr*)")
    print(T[T.kind != "similarity"][["spout", "epoch", "kind", "n_animals", "frac_sig", "frac_sig_agree", "min_p_fwer",
                                     "frac_sig_fdr", "frac_sig_fdr_agree"]].round(3).to_string(index=False))
    D = dict(res=res, T=T, st=st, grid=imaging_grid(), method=method_text(n_perm))
    save_fd(fig_name(st), D)
    plot_all(D)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
