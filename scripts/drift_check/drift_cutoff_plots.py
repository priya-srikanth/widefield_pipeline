"""Figures + summary tables for the drift-cutoff check (--plot-only path; reads only the CSVs the check wrote).

    python scripts/drift_check/drift_cutoff_check.py plot
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
EPOCHS = ("pre", "acute", "subacute", "chronic")
EP_COL = {"pre": "#4d4d4d", "acute": "#d62728", "subacute": "#ff7f0e", "chronic": "#1f77b4"}
AN_MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
ARMS = ("pos10", "pos15", "pos20", "pos30", "blk10", "blk20", "cos10", "cos20")
POS_ORDER = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
BANDS = (">43", "20-43", "10-20", "5-10", "2-5", "1-2", "<1")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, out: Path, name: str):
    from wfield_local import orofacial_cohort as oc
    oc.save_png_svg(fig, out / name, dpi=140)
    print(f"  -> {out / name}")


def _cells(D):
    return [(a, e) for a in ANIMALS for e in EPOCHS if ((D.animal == a) & (D.epoch == e)).any()]


def _scatter_cells(ax, D, col, cells, jitter=0.12, ref=None):
    rng = np.random.default_rng(0)
    for i, (a, e) in enumerate(cells):
        v = D[(D.animal == a) & (D.epoch == e)][col].to_numpy(float)
        ax.scatter(i + rng.uniform(-jitter, jitter, v.size), v, s=16, facecolors="none", edgecolors=EP_COL[e],
                   marker=AN_MK[a], lw=0.9)
        if v.size:
            ax.plot([i - 0.3, i + 0.3], [np.nanmedian(v)] * 2, color=EP_COL[e], lw=2)
    if ref is not None:
        for r in np.atleast_1d(ref):
            ax.axhline(r, color="0.5", ls="--", lw=0.7)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{a}\n{e[:3]}" for a, e in cells], fontsize=7)


def fig_blocks(out: Path):
    plt = _plt()
    B = pd.read_csv(out / "blocks.csv")
    S = pd.read_csv(out / "sessions.csv")
    cells = _cells(B)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
    ax = axs[0]
    for i, (a, e) in enumerate(cells):
        v = B[(B.animal == a) & (B.epoch == e)].dur_s.to_numpy() / 60
        ax.boxplot([v], positions=[i], widths=0.5, showfliers=False, medianprops=dict(color=EP_COL[e]))
        mx = B[(B.animal == a) & (B.epoch == e)].groupby("label").dur_s.max() / 60
        ax.scatter(np.full(mx.size, i) + np.random.default_rng(i).uniform(-.15, .15, mx.size), mx, s=14,
                   facecolors="none", edgecolors=EP_COL[e], marker=AN_MK[a], lw=0.8)
    for c in (10, 15, 20, 30):
        ax.axhline(c, color="0.6", ls=":", lw=0.7)
        ax.text(len(cells) - 0.4, c, f" {c} min", fontsize=7, va="center", color="0.4")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{a}\n{e[:3]}" for a, e in cells], fontsize=7)
    ax.set_ylabel("block duration (min)")
    ax.set_title("Fig 1a. Block durations (box = all blocks, 5-95% whiskers off; markers = each session's longest)")
    _scatter_cells(axs[1], S, "task_span_min", cells)
    axs[1].set_ylabel("task span (min)")
    axs[1].set_title("Fig 1b. Session length")
    _scatter_cells(axs[2], S, "peak_period_min", cells)
    axs[2].set_ylabel("peak period (min)")
    axs[2].set_title("Fig 1c. Dominant block-alternation period\n(position indicators, spectral peak 1-60 min)")
    fig.tight_layout()
    _save(fig, out, "fig1_block_durations.png")
    plt.close(fig)


def fig_order(out: Path):
    plt = _plt()
    TR = pd.read_csv(out / "trial_times.csv")
    O = pd.read_csv(out / "spout_order.csv")
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    for ax, a in zip(axs, ANIMALS):
        M = TR[TR.animal == a].groupby(["label", "epoch", "pos"]).t_norm.mean().reset_index()
        for j, p in enumerate(POS_ORDER):
            g = M[M.pos == p]
            for e in EPOCHS:
                v = g[g.epoch == e].t_norm.to_numpy()
                ax.scatter(j + np.random.default_rng(j).uniform(-.2, .2, v.size), v, s=12, facecolors="none",
                           edgecolors=EP_COL[e], marker=AN_MK[a], lw=0.8)
            ax.plot([j - .3, j + .3], [g.t_norm.mean()] * 2, color="k", lw=2)
        ax.axhline(0.5, color="0.6", ls="--", lw=0.7)
        o = O[(O.animal == a) & (O.epoch == "all")].iloc[0]
        ax.set_title(f"{a}: Kendall W {o.kendall_W:.2f} (p {o.friedman_p:.2g})\ntime-only LOSO decode "
                     f"{o.time_only_loso_acc:.2f} (chance .17)", fontsize=8)
        ax.set_xticks(range(6))
        ax.set_xticklabels([p.replace("_", "\n") for p in POS_ORDER], fontsize=7)
    axs[0].set_ylabel("mean normalised time in session (per session)")
    fig.suptitle("Fig 2. Spout order vs time in session (markers = sessions; bar = mean over sessions)")
    fig.tight_layout()
    _save(fig, out, "fig2_spout_order.png")
    plt.close(fig)


def fig_residual(out: Path, M: pd.DataFrame):
    plt = _plt()
    cells = _cells(M)
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharey="row")
    cols = plt.cm.viridis(np.linspace(0, 0.95, len(BANDS)))
    for r, sig in enumerate(("bm", "comp")):
        for c, fr in enumerate(("all", "oot")):
            ax = axs[r, c]
            bottom = np.zeros(len(cells))
            for b, col in zip(BANDS, cols):
                v = np.array([M[(M.animal == a) & (M.epoch == e)][f"v3_{sig}_{fr}_{b}"].median() for a, e in cells])
                ax.bar(range(len(cells)), v, bottom=bottom, color=col, label=b + " min" if r == 0 and c == 0 else None)
                bottom += v
            blk = np.array([M[(M.animal == a) & (M.epoch == e)][f"v3_{sig}_{fr}_block"].median() for a, e in cells])
            ax.scatter(range(len(cells)), blk, marker="_", s=200, color="r", zorder=5,
                       label="block band (per animal)" if r == 0 and c == 0 else None)
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels([f"{a}\n{e[:3]}" for a, e in cells], fontsize=7)
            ax.set_title(f"Fig 3{'abcd'[2 * r + c]}. {'brain-mean' if sig == 'bm' else 'all components (pixel-total)'}"
                         f", {'all frames' if fr == 'all' else 'out-of-trial frames'}", fontsize=9)
            ax.set_ylabel("fraction of variance (median session)")
    axs[0, 0].legend(fontsize=7, ncol=4, frameon=False)
    fig.suptitle("Fig 3. What is left after hpfit: variance by band (DCT periods over the whole record)")
    fig.tight_layout()
    _save(fig, out, "fig3_residual_bands.png")
    plt.close(fig)


def fig_arms(out: Path, M: pd.DataFrame, arms=ARMS, name="fig4_arms.png", title=""):
    plt = _plt()
    cells = _cells(M)
    panels = [("ret_comp_block", "a. block-band retention (all comps)", [0.98, 1.0]),
              ("ret_comp_cycle", "a'. cycle-band retention (all comps)", [0.98, 1.0]),
              ("ret_bm_block", "a''. block-band retention (brain-mean)", [0.98, 1.0]),
              ("bmean_posvar_ratio_roi_median", "b. between-position variance of block means (ROI median)", [1.0]),
              ("bmean_r_roi_min", "b'. block-mean r after vs before (worst ROI)", [1.0]),
              ("ret_comp_10-43", "c. 10-43 min power kept (all comps; 1 - removed)", [1.0]),
              ("dprecue", "d. pre-cue (ENL) decode: arm - base", [0.0]),
              ("dcue", "d'. cue-window decode: arm - base", [0.0])]
    D = M.copy()
    for arm in arms:
        D[f"{arm}_dprecue"] = D[f"dec_precue_{arm}"] - D["dec_precue_base"]
        D[f"{arm}_dcue"] = D[f"dec_cue_{arm}"] - D["dec_cue_base"]
    fig, axs = plt.subplots(len(panels), len(arms), figsize=(3.2 * len(arms) + 3, 3.0 * len(panels)), sharey="row",
                            squeeze=False)
    for r, (key, title, ref) in enumerate(panels):
        for c, arm in enumerate(arms):
            ax = axs[r, c]
            _scatter_cells(ax, D, f"{arm}_{key}", cells, ref=ref)
            if r == 0:
                ax.set_title(arm, fontsize=11)
            if c == 0:
                ax.set_ylabel(title, fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(title or "Fig 4. Slow cosine removal stacked on hpfit (pos = position-protected, cos = unprotected): "
                 "retention, block means, removed variance, decoding\n(sessions; shape = animal, colour = epoch; "
                 "bar = median). Block-band retention is 1 by construction when the band is faster than the cutoff.")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out, name)
    plt.close(fig)


def tables(out: Path):
    M = pd.read_csv(out / "measure_sessions.csv")
    agg = {}
    cols3 = [c for c in M.columns if c.startswith("v3_")]
    T3 = M.groupby(["animal", "epoch"])[cols3].median()
    T3.to_csv(out / "item3_residual_bands_animal_epoch.csv", float_format="%.4g")
    agg["item3"] = T3
    rows = []
    for (a, e), g in M.groupby(["animal", "epoch"]):
        for arm in ARMS:
            rows.append(dict(
                animal=a, epoch=e, arm=arm, n=len(g),
                blockband_ret_comp_median=g[f"{arm}_ret_comp_block"].median(),
                blockband_ret_comp_min=g[f"{arm}_ret_comp_block"].min(),
                blockband_ret_bm_median=g[f"{arm}_ret_bm_block"].median(),
                blockband_ret_bm_min=g[f"{arm}_ret_bm_block"].min(),
                band12_ret_comp_min=g[f"{arm}_ret_comp_1-2"].min(),
                cycle_ret_comp_median=g[f"{arm}_ret_comp_cycle"].median(),
                cycle_ret_comp_min=g[f"{arm}_ret_comp_cycle"].min(),
                cycle_ret_bm_min=g[f"{arm}_ret_bm_cycle"].min(),
                bmean_r_roi_median=g[f"{arm}_bmean_r_roi_median"].median(),
                bmean_r_roi_min=g[f"{arm}_bmean_r_roi_min"].min(),
                bmean_var_ratio_roi=g[f"{arm}_bmean_var_ratio_roi_median"].median(),
                bmean_posvar_ratio_roi=g[f"{arm}_bmean_posvar_ratio_roi_median"].median(),
                bmean_posvar_ratio_roi_min=g[f"{arm}_bmean_posvar_ratio_roi_median"].min(),
                bmean_posvar_ratio_bm=g[f"{arm}_bmean_posvar_ratio_bm"].median(),
                removed_10_43_comp=1 - g[f"{arm}_ret_comp_10-43"].median(),
                removed_10_43_bm=1 - g[f"{arm}_ret_bm_10-43"].median(),
                removed_gt43_comp=1 - g[f"{arm}_ret_comp_>43"].median(),
                removed_total_comp=g[f"{arm}_removed_comp_total"].median(),
                removed_total_bm=g[f"{arm}_removed_bm_total"].median(),
                precue_base=g.dec_precue_base.mean(), precue_arm=g[f"dec_precue_{arm}"].mean(),
                d_precue=(g[f"dec_precue_{arm}"] - g.dec_precue_base).mean(),
                d_precue_neg=int(((g[f"dec_precue_{arm}"] - g.dec_precue_base) < 0).sum()),
                cue_base=g.dec_cue_base.mean(), cue_arm=g[f"dec_cue_{arm}"].mean(),
                d_cue=(g[f"dec_cue_{arm}"] - g.dec_cue_base).mean()))
    T4 = pd.DataFrame(rows)
    T4["epoch"] = pd.Categorical(T4.epoch, EPOCHS, ordered=True)
    T4 = T4.sort_values(["arm", "animal", "epoch"])
    T4.to_csv(out / "item4_arms_animal_epoch.csv", index=False, float_format="%.4g")
    agg["item4"] = T4
    P = []
    for arm in ARMS:
        g = M
        P.append(dict(arm=arm, n=len(g), d_precue=(g[f"dec_precue_{arm}"] - g.dec_precue_base).mean(),
                      d_cue=(g[f"dec_cue_{arm}"] - g.dec_cue_base).mean(),
                      blockband_ret_comp_min=g[f"{arm}_ret_comp_block"].min(),
                      posvar_ratio_median=g[f"{arm}_bmean_posvar_ratio_roi_median"].median(),
                      removed_10_43_comp=1 - g[f"{arm}_ret_comp_10-43"].median()))
    TP = pd.DataFrame(P)
    TP.to_csv(out / "item4_arms_pooled.csv", index=False, float_format="%.4g")
    agg["pooled"] = TP
    (out / "figdata").mkdir(exist_ok=True)
    with open(out / "figdata" / "drift_cutoff_tables.pkl", "wb") as f:
        pickle.dump(agg, f)
    return M


def fig_replacement(out: Path):
    plt = _plt()
    D = pd.read_csv(out / "replacement_vs_stacked.csv")
    D["epoch"] = D["epoch"].astype(str)
    cuts = sorted(D.cutoff_min.unique())
    panels = [("r2_comp_rep_vs_stk", "R2 replacement vs stacked\n(all components, variance-weighted)"),
              ("r_bm_rep_vs_stk", "r brain-mean,\nreplacement vs stacked"),
              ("r_roi_rep_vs_stk_min", "r worst ROI,\nreplacement vs stacked"),
              ("slow5_power_rep_over_stk_bm", ">= 5 min power, brain-mean:\nreplacement / stacked"),
              ("dprecue", "pre-cue decode:\nreplacement - stacked"),
              ("dcue", "cue decode:\nreplacement - stacked")]
    D["dprecue"] = D.dec_precue_rep - D.dec_precue_stk
    D["dcue"] = D.dec_cue_rep - D.dec_cue_stk
    fig, axs = plt.subplots(1, len(panels), figsize=(20, 3.8))
    for ax, (k, t) in zip(axs, panels):
        for i, c in enumerate(cuts):
            g = D[D.cutoff_min == c]
            for _, r in g.iterrows():
                ax.scatter(i + np.random.default_rng(int(r.label[-4:])).uniform(-.15, .15), r[k], s=22,
                           facecolors="none", edgecolors=EP_COL.get(r.epoch, "k"), marker=AN_MK[r.animal])
        ax.set_xticks(range(len(cuts)))
        ax.set_xticklabels([f"{c} min" for c in cuts])
        ax.set_title(t, fontsize=8)
    fig.suptitle("Fig 5. Replacement (polynomial -> position-protected cosines) vs stacked (hpfit + same cosines); "
                 "8 sessions, shape = animal, colour = epoch")
    fig.tight_layout()
    _save(fig, out, "fig5_replacement_vs_stacked.png")
    plt.close(fig)


def main(out: Path) -> int:
    out = Path(out)
    if (out / "blocks.csv").exists():
        fig_blocks(out)
        fig_order(out)
    if (out / "measure_sessions.csv").exists():
        M = tables(out)
        fig_residual(out, M)
        fig_arms(out, M, arms=("pos10", "pos15", "pos20", "pos30", "cos10", "cos20"))
        fig_arms(out, M, arms=("blk10", "blk20"), name="fig4b_block_protected_literal.png",
                 title="Fig 4b. The literal per-BLOCK protection (ill-posed): it ADDS slow power and destroys decoding")
    if (out / "replacement_vs_stacked.csv").exists():
        fig_replacement(out)
    return 0
