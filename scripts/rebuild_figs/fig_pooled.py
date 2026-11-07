"""POOLED cross-animal, within-animal-normalized versions of the per-animal results-page figures (Priya 10-24): goal
page Figs 1p / 2c / 4p / 5p / 5c / 6p / 10p / 12p (`fig1p_contest_pooled.png` ... `fig12p_residual_trend_pooled.png`).

Each quantity is a CHANGE from the animal's own reference (pre, or the pre-stroke null), pooled with the paired
animals -> sessions bootstrap (`analysis_kit.boot_delta`, animal-weighted) where session values exist; animal dots on
every panel (rule 8).

PROVENANCE: recreated 2026-11-04 (RECONCILIATION section 7) from scratchpad_scripts_backup_20261104/S1_cdd1e016/
  fig_pooled.py. The plotting / bootstrap code is the original's, unchanged (wrapped in draw(); every bootstrap is
  seeded as before). Changes: inputs are THIS run's routed outputs (no hard-coded adopted-run M: root, no wf_attempt
  sys.path), --out / --plot-only / figdata (`_common`; the figdata = the input tables, so --plot-only re-runs the
  same seeded bootstraps on them).
INPUTS (this run): <encoding>/movement_first_subset-nokin_exec-polar_licks-combined_bins{,_bw0.5,_bw2}/tables/
  region_measures.csv; attempt_coding.out_dir()/summary_{lick,cue}.csv; <encoding>/qc/lp_encoder_decompose_
  frozen_encoder_gate-attempted_v7g3_t3d_joint_lick_by_animal.csv; <encoding>/allsession_encoder_v7g3_t3d_ref/
  {allsession_changes_by_session, random_splits_<an>}.csv; frozen_encoder.out_root()/frozen_encoder_gate-attempted_
  v7g3_t3d/residual_trend_sessions.csv.

    PYTHONPATH=<wf_rebuild> python scripts/rebuild_figs/fig_pooled.py [--out DIR] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)
from wfield_local.frozen_encoder import design_tag as _design_tag  # frame tag of the FROZEN_LICKS_ALL design (8.51)

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402
from fig_layer2 import allsession_dir  # noqa: E402

from wfield_local.analysis_kit import boot_delta  # noqa: E402

NAME = "pooled"
AN = ["PS92", "PS93", "PS94", "PS95"]
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
POST = ["acute", "subacute", "chronic"]
ECOL = {"acute": "#d95f02", "subacute": "#e6ab02", "chronic": "#1b9e77"}
BINS = "movement_first_subset-nokin_exec-polar_licks-combined_bins"


def compute() -> dict:
    """Every input table of this run (routed writers' folders)."""
    from wfield_local import attempt_coding as ac
    from wfield_local import frozen_encoder as fe
    from wfield_local import run_mode
    E = run_mode.encoding_out_root()
    L2 = allsession_dir(_design_tag("v7g3_t3d_ref"))
    return {
        "bins": {f: pd.read_csv(E / f"{BINS}{f}" / "tables" / "region_measures.csv") for f in ("", "_bw0.5", "_bw2")},
        "attempt_summary": {w: pd.read_csv(Path(ac.out_dir()) / f"summary_{w}.csv") for w in ("lick", "cue")},
        "decompose": pd.read_csv(E / "qc" / _design_tag("lp_encoder_decompose_frozen_encoder_gate-attempted_v7g3_t3d"
                                                         "_joint_lick_by_animal.csv")),
        "layer2": pd.read_csv(L2 / "allsession_changes_by_session.csv", keep_default_na=False, na_values=[""]),
        "random_splits": {a: pd.read_csv(L2 / f"random_splits_{a}.csv") for a in AN},
        "residual_trend": pd.read_csv(fe.out_root() / _design_tag("frozen_encoder_gate-attempted_v7g3_t3d")
                                      / "residual_trend_sessions.csv"),
        "src": {"encoding": str(E), "allsession": str(L2)},
    }


def rng():
    return np.random.default_rng(0)


def sess(df, col, ep):
    return {a: df[(df.animal == a) & (df.epoch == ep)][col].dropna().to_numpy(float) for a in AN
            if len(df[(df.animal == a) & (df.epoch == ep)][col].dropna())}


def pooled_bars(ax, xs, ivs, per_an, color, dx=0.0, width=0.6, label=None):
    """Bar = pooled change, whisker = 95% nested CI, markers = each animal's change."""
    for x, iv, pa in zip(xs, ivs, per_an):
        if iv is None:
            continue
        c = color[x] if isinstance(color, dict) else color
        ax.bar(x + dx, iv.point, width=width, color=c, alpha=0.8, label=label if x == xs[0] else None)
        ax.errorbar(x + dx, iv.point, yerr=[[iv.point - iv.lo], [iv.hi - iv.point]], color="k", capsize=3, lw=1.2)
        for a, v in pa.items():
            ax.scatter(x + dx + {"PS92": -0.12, "PS93": -0.04, "PS94": 0.04, "PS95": 0.12}[a] * width / 0.6, v,
                       marker=MK[a], s=18, facecolor="w", edgecolor="k", lw=0.8, zorder=5)
        star = "*" if (iv.lo > 0 or iv.hi < 0) else ""
        ax.text(x + dx, max(iv.hi, max(pa.values(), default=iv.hi)), star, ha="center", va="bottom", fontsize=12)
    ax.axhline(0, color="k", lw=0.7)


def animal_legend(fig, y=None, ci=True):
    """Animal markers (+ the CI whisker) in one row UNDER the panels, so it never meets a two-line title."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=MK[a], ls="", mfc="w", mec="k", label=a) for a in AN]
    if ci:
        h.append(Line2D([], [], color="k", lw=1.2, label="95% nested CI (animals -> sessions)"))
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))


def per_animal_delta(post, pre):
    return {a: post[a].mean() - pre[a].mean() for a in sorted(set(post) & set(pre))}


def draw(fd: dict, OUT: Path) -> list:
    TXT = []
    # ---------------------------------------------------------------- 1p: binned contest (change from pre)
    d = fd["bins"][""].copy()
    c = d[d.pos == "cortex"].copy()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    for j, w in enumerate(["lick", "cue"]):  # noqa: B007 (the original's loop, unchanged)
        c[f"t_{w}"] = c[f"nokin:u:tgtbin|{w}"]
        c[f"x_{w}"] = c[f"nokin:u:exbin|{w}"]
        c[f"d_{w}"] = c[f"t_{w}"] - c[f"x_{w}"]
    for j, w in enumerate(["lick", "cue"]):
        ax = axs[j]
        for q, colr, dx, lab in (("t", "#6a3d9a", -0.2, "target bins"), ("x", "#e66101", 0.2, "executed bins")):
            ivs, pas = [], []
            for ep in POST:
                post, pre = sess(c, f"{q}_{w}", ep), sess(c, f"{q}_{w}", "pre")
                ivs.append(boot_delta(post, pre, rng())); pas.append(per_animal_delta(post, pre))
            pooled_bars(ax, list(range(3)), ivs, pas, colr, dx=dx, width=0.38, label=lab)
        ax.set_xticks(range(3)); ax.set_xticklabels(POST); ax.set_title(f"{w} window: unique dR2, change from pre")
        ax.set_ylabel("unique dR2 (R2 fraction), epoch - pre"); ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax = axs[2]
    for k, w in enumerate(["lick", "cue"]):
        ivs, pas = [], []
        for ep in POST:
            post, pre = sess(c, f"d_{w}", ep), sess(c, f"d_{w}", "pre")
            iv = boot_delta(post, pre, rng()); ivs.append(iv); pas.append(per_animal_delta(post, pre))
            TXT.append(f"1p contest {w} {ep}: {iv.point:+.4f} [{iv.lo:+.4f}, {iv.hi:+.4f}]")
        pooled_bars(ax, list(range(3)), ivs, pas, ["#3f007d", "#9e9ac8"][k], dx=(-0.2, 0.2)[k], width=0.38, label=f"{w} window")
    ax.set_xticks(range(3)); ax.set_xticklabels(POST); ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("contest: (target - executed), change from pre"); ax.set_ylabel("R2 fraction")
    animal_legend(fig)
    fig.suptitle("NEURAL, pooled: binned target vs executed contest (cortex, width 1). Within-animal change from pre, mean of animals; * = CI excludes 0", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9)); save_png_svg(fig, OUT / "fig1p_contest_pooled.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 2c: decoder, share of the pre-stroke code retained
    CL = [("hit", "#1b9e77"), ("incomplete", "#d95f02"), ("noattempt", "#7570b3"), ("reaction", "#e7298a"), ("stopped", "0.6")]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for j, w in enumerate(["lick", "cue"]):
        s = fd["attempt_summary"][w].copy()
        s = s[(s.spout == "far_contra") & (s.method == "dom")]
        ax = axs[j]
        for i, ep in enumerate(POST):
            for k, (cl, col) in enumerate(CL):
                vals = {}
                for a in AN:
                    ph = s[(s.animal == a) & (s.cls == "pre_hit")]
                    y = s[(s.animal == a) & (s.cls == cl) & (s.epoch == ep)]
                    if len(ph) and len(y) and y.n.iloc[0] >= 5:
                        vals[a] = (y.p_true.iloc[0] - 1 / 6) / (ph.p_true.iloc[0] - 1 / 6)
                if len(vals) >= 2:
                    x = i * 6 + k
                    m = np.mean(list(vals.values()))
                    ax.bar(x, m, color=col, width=0.85, label=cl if i == 0 else None)
                    for a, v in vals.items():
                        ax.scatter(x, v, marker=MK[a], s=18, facecolor="w", edgecolor="k", lw=0.8, zorder=5)
                    ax.text(x, -0.62, f"{len(vals)}", ha="center", fontsize=7, color=col)
                    TXT.append(f"2c {w} {ep} {cl}: mean {m:+.2f} (n animals {len(vals)})")
        ax.axhline(1, color="0.3", lw=1, ls="--"); ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks([i * 6 + 2 for i in range(3)]); ax.set_xticklabels(POST); ax.set_ylim(-0.7, 1.4)
        ax.set_title(f"{w} window"); ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper right")
    axs[0].set_ylabel("share of the pre-stroke far-contra code retained\n(P(true) - 1/6) / (pre hits - 1/6)")
    animal_legend(fig, ci=False)
    fig.suptitle("NEURAL, pooled: frozen pre-stroke decoder on far-contra trials by trial type, normalized to each animal's pre-stroke hits\n"
                 "(1 = as well coded as pre-stroke hits, 0 = chance); bar = mean of animals with >= 5 trials (count under each bar)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86)); save_png_svg(fig, OUT / "fig2c_decoder_pooled.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 4p: decomposition as a fraction of each animal's pre response
    q = fd["decompose"].copy()
    q = q[q.pos == "far_contra"]
    PARTS = [("d_data", "measured", "0.2"), ("d_full", "predicted", "#9e9ac8"), ("d_lick", "of which lick", "#d95f02"),
             ("d_target", "of which target", "#6a3d9a"), ("d_residual", "residual", "#1b9e77")]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for r, reg in enumerate(["SSp-m_left", "SSp-m_right"]):
        ax = axs[r]
        for i, ep in enumerate(POST):
            for k, (cn, lab, col) in enumerate(PARTS):
                vals = {a: float(x[cn].iloc[0] / x.pre_data.iloc[0]) for a in AN
                        if len(x := q[(q.animal == a) & (q.epoch == ep) & (q.region == reg)])}
                x0 = i * 6 + k
                m = np.mean(list(vals.values()))
                ax.bar(x0, m, color=col, width=0.85, label=lab if i == 0 else None)
                for a, v in vals.items():
                    ax.scatter(x0, v, marker=MK[a], s=18, facecolor="w", edgecolor="k", lw=0.8, zorder=5)
                same = sum(np.sign(v) == np.sign(m) for v in vals.values())
                ax.text(x0, ax.get_ylim()[0] if False else -1.05, f"{same}/{len(vals)}", ha="center", fontsize=6.5)
                if cn in ("d_data", "d_full", "d_residual"):
                    TXT.append(f"4p {reg} {ep} {lab}: {m:+.2f} of pre ({same}/{len(vals)} same sign)")
        ax.axhline(0, color="k", lw=0.7); ax.set_xticks([i * 6 + 2 for i in range(3)]); ax.set_xticklabels(POST)
        ax.set_title(reg.replace("_left", " LEFT (ipsilesional)").replace("_right", " RIGHT (contralesional)"))
        ax.set_ylim(-1.15, 1.25)
    axs[0].set_ylabel("change from pre / animal's pre-stroke response"); axs[0].legend(fontsize=8, frameon=False, ncol=5, loc="upper left")
    animal_legend(fig, ci=False)
    fig.suptitle("NEURAL, pooled: frozen-encoder decomposition (far contra, lick window), each part's change divided by the animal's own "
                 "pre-stroke response,\nmean of animals; numbers under bars = animals with the sign of the mean", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86)); save_png_svg(fig, OUT / "fig4p_decomp_pooled.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 5p: Layer 2 change vs the late-pre null, pooled
    W = fd["layer2"].copy()
    _c = [c for c in W.columns if c.startswith("change")]
    W[_c] = W[_c] / 1e3                     # stored x1e3 -> R2 fraction
    W = W[W.fit == "combined"]
    AR = ["cortex", "SSp-m_left", "SSp-m_right"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for j, ar in enumerate(AR):
        a_ = W[W.area == ar]
        ivs, pas = [], []
        for ep in POST:
            post, pre = sess(a_, "change_ref", ep), sess(a_, "change_ref", "pre_late")
            iv = boot_delta(post, pre, rng()); ivs.append(iv); pas.append(per_animal_delta(post, pre))
            TXT.append(f"5p {ar} {ep} - pre_late: {iv.point:+.4f} [{iv.lo:+.4f}, {iv.hi:+.4f}] n={iv.n_animals}")
        pooled_bars(axs[j], list(range(3)), ivs, pas, dict(enumerate(ECOL[e] for e in POST)))
        axs[j].set_xticks(range(3)); axs[j].set_xticklabels(POST); axs[j].set_title(ar)
    axs[0].set_ylabel("change_ref minus late-pre null\nheld-out R2 gain (fraction)")
    animal_legend(fig)
    fig.suptitle("NEURAL, pooled: Layer 2 kernel change per epoch minus each animal's late-pre (drift-null) change, mean of animals; * = CI excludes 0", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88)); save_png_svg(fig, OUT / "fig5p_layer2_pooled.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 5c: random-split null, pooled in null-SD units
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for j, ar in enumerate(AR):
        Z = {}
        for a in AN:
            T = fd["random_splits"][a].copy()
            M = T[T.area == ar].groupby(["split", "epoch"]).change_ref.mean().unstack("epoch")
            mu, sd = M["pseudo"].median(), M["pseudo"].std()
            Z[a] = (M - mu) / sd
        ax = axs[j]
        pooled = {}
        for k, ep in enumerate(["pseudo", *POST]):
            cols = [Z[a][ep].to_numpy() for a in AN if ep in Z[a] and not (ep == "acute" and a == "PS95")]
            n = min(len(v) for v in cols)
            pv = np.mean([v[:n] for v in cols], 0)              # split i of each animal pooled (random pairing)
            pooled[ep] = pv
            col = "0.55" if ep == "pseudo" else ECOL[ep]
            ax.scatter(np.full(n, k) + np.random.default_rng(k).uniform(-0.15, 0.15, n), pv, s=12, color=col, alpha=0.75, lw=0)
            ax.plot([k - 0.3, k + 0.3], [np.median(pv)] * 2, color="k", lw=1.5)
            for a in AN:
                if ep in Z[a] and not (ep == "acute" and a == "PS95"):
                    ax.scatter(k + 0.38, Z[a][ep].median(), marker=MK[a], s=18, facecolor="w", edgecolor="k", lw=0.8)
        labs = ["random pre\nsplit (null)"]
        for ep in POST:
            p = float((pooled["pseudo"][:len(pooled[ep])] >= pooled[ep]).mean())
            labs.append(f"{ep}\nP={p:.2f}")
            TXT.append(f"5c {ar} {ep}: pooled z median {np.median(pooled[ep]):.1f}, P={p:.2f}")
        ax.axhspan(np.quantile(pooled["pseudo"], 0.05), np.quantile(pooled["pseudo"], 0.95), color="0.88", zorder=0)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8); ax.set_title(ar)
    axs[0].set_ylabel("change_ref in units of each animal's null SD\n(0 = null median)")
    animal_legend(fig)
    fig.suptitle("NEURAL, pooled: Layer 2 random-split null. Each animal's values in units of its own null spread, then the mean over animals per split\n"
                 "(dots = 20 pooled splits; open markers = each animal's median; PS95 acute excluded: one session). P = share of pooled null splits >= the epoch.", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86)); save_png_svg(fig, OUT / "fig5c_layer2_random_splits_pooled.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 6p: Layer 2 per group, pooled
    G = ["lick", "contact", "target", "exec_within", "exec_locked", "timing", "other"]
    a_ = W[W.area == "cortex"]
    fig, ax = plt.subplots(figsize=(14, 4.4))
    for k, ep in enumerate(["acute", "chronic"]):
        ivs, pas = [], []
        for g in G:
            post, pre = sess(a_, f"change_{g}", ep), sess(a_, f"change_{g}", "pre_late")
            iv = boot_delta(post, pre, rng()); ivs.append(iv); pas.append(per_animal_delta(post, pre))
            TXT.append(f"6p {g} {ep} - pre_late: {iv.point:+.4f} [{iv.lo:+.4f}, {iv.hi:+.4f}]")
        pooled_bars(ax, list(range(len(G))), ivs, pas, ECOL[ep], dx=(-0.2, 0.2)[k], width=0.38, label=ep)
    ax.set_xticks(range(len(G))); ax.set_xticklabels(G); ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylabel("unique change of one group minus late-pre null\nheld-out R2 (fraction), cortex")
    animal_legend(fig, y=0.93)
    fig.suptitle("NEURAL, pooled: Layer 2 unique change per kernel group minus the late-pre null, mean of animals; * = CI excludes 0", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.85)); save_png_svg(fig, OUT / "fig6p_layer2_groups_pooled.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 10p: width sweep, contest change from pre, pooled
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    for j, w in enumerate(["lick", "cue"]):
        ax = axs[j]
        for k, (f, bw) in enumerate([("_bw0.5", 0.5), ("", 1), ("_bw2", 2)]):
            dd = fd["bins"][f].copy()
            cc = dd[dd.pos == "cortex"].copy()
            cc["d"] = cc[f"nokin:u:tgtbin|{w}"] - cc[f"nokin:u:exbin|{w}"]
            ivs, pas = [], []
            for ep in POST:
                post, pre = sess(cc, "d", ep), sess(cc, "d", "pre")
                iv = boot_delta(post, pre, rng()); ivs.append(iv); pas.append(per_animal_delta(post, pre))
                TXT.append(f"10p {w} width {bw} {ep}: {iv.point:+.4f} [{iv.lo:+.4f}, {iv.hi:+.4f}]")
            pooled_bars(ax, list(range(3)), ivs, pas, ["#cbc9e2", "#9e9ac8", "#54278f"][k], dx=(-0.27, 0, 0.27)[k], width=0.26,
                        label=f"width {bw:g}")
        ax.set_xticks(range(3)); ax.set_xticklabels(POST); ax.set_title(f"{w} window"); ax.legend(fontsize=8, frameon=False, loc="upper left")
    axs[0].set_ylabel("contest (target - executed) unique dR2 (R2 fraction)\nchange from pre")
    animal_legend(fig)
    fig.suptitle("NEURAL, pooled: softmax width sweep of the contest, within-animal change from pre, mean of animals; * = CI excludes 0", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88)); save_png_svg(fig, OUT / "fig10p_width_pooled.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 12p: residual trend in pre-SD units, pooled
    R = fd["residual_trend"].copy()
    R = R[R.window == "lick"]
    ROWS = [("cortex", ["cortex"]), ("MOs", ["MOs_left", "MOs_right"]), ("MOp", ["MOp_left", "MOp_right"]),
            ("SSp-m", ["SSp-m_left", "SSp-m_right"]), ("SSp-n", ["SSp-n_left", "SSp-n_right"])]
    fig, axs = plt.subplots(len(ROWS), 4, figsize=(15, 3.5 * len(ROWS)), sharey=True, gridspec_kw={"width_ratios": [3, 1, 3, 1]})
    for r, (fam, regs) in enumerate(ROWS):
        if len(regs) == 1:
            axs[r, 2].set_visible(False); axs[r, 3].set_visible(False)
        for j, reg in enumerate(regs):
            ax, sx = axs[r, 2 * j], axs[r, 2 * j + 1]
            x_ = R[R.region == reg].copy()
            for a in AN:
                g = x_[x_.animal == a].sort_values("day")
                p = g[g.epoch == "pre"].far_contra
                g = g.assign(z=(g.far_contra - p.mean()) / p.std())
                x_.loc[g.index, "z"] = g.z
                ax.plot(g.day, g.z, color="0.75", lw=0.8, zorder=1)
                ax.scatter(g.day, g.z, marker=MK[a], s=12, facecolor="w", edgecolor="0.4", lw=0.7, zorder=2)
            bins = np.arange(-12, 56, 4)
            x_["bin"] = pd.cut(x_.day, bins)
            pm = x_.groupby(["bin", "animal"], observed=True).z.mean().groupby(level=0, observed=True).agg(["mean", "count"])
            pm = pm[pm["count"] >= 3]
            ax.plot([b.mid for b in pm.index], pm["mean"], color="k", lw=2.2, label="mean of animals (4-day bins, >= 3 animals)")
            ax.axvline(0, color="r", ls=":", lw=1); ax.axhline(0, color="k", lw=0.6)
            side = "" if reg == "cortex" else (" (ipsilesional)" if reg.endswith("_left") else " (contralesional)")
            ax.set_title(reg + side, fontsize=10)
            if r == len(ROWS) - 1 or reg == "cortex":
                ax.set_xlabel("days since stroke")
            if r == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper left")
            ivs, pas = [], []
            for ep in POST:
                post, pre = sess(x_, "z", ep), sess(x_, "z", "pre")
                iv = boot_delta(post, pre, rng()); ivs.append(iv); pas.append(per_animal_delta(post, pre))
                TXT.append(f"12p {reg} {ep}: {iv.point:+.2f} pre-SD [{iv.lo:+.2f}, {iv.hi:+.2f}]")
            pooled_bars(sx, list(range(3)), ivs, pas, dict(enumerate(ECOL[e] for e in POST)))
            sx.set_xticks(range(3)); sx.set_xticklabels(["acute", "sub-\nacute", "chronic"], fontsize=8)
            sx.set_title("change from pre\n(pooled, 95% CI)", fontsize=8.5)
        axs[r, 0].set_ylabel(f"{fam}: far-contra residual\n(animal's pre-stroke SD)")
    animal_legend(fig)
    fig.suptitle("NEURAL, pooled: frozen-encoder residual (far contra, lick window) per session vs day, each animal in units of its own pre-stroke spread "
                 "(0 = pre mean); thick line = mean of animals.\nSide panels: each epoch's pooled change from pre with 95% nested CI (animals -> sessions); "
                 "markers = animals; * = CI excludes 0", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)); save_png_svg(fig, OUT / "fig12p_residual_trend_pooled.png", dpi=115); plt.close(fig)

    return TXT


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    fd = _common.load_figdata(a.out, NAME) if a.plot_only else compute()
    if not a.plot_only:
        _common.save_figdata(a.out, NAME, fd)
    TXT = draw(fd, a.out)
    (a.out / "pooled_numbers.txt").write_text("\n".join(TXT))
    print("\n".join(TXT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
