"""Figures for the state-space recovery results page (Priya 11-07, "draft a results page").

Reads only the tables the analysis scripts wrote under Widefield/state_space_recovery/ (no imaging reload):
  sim/sim_detection_n{0,15}.csv                       state_space_sim.py
  trials{,_on_target,_landing}/state_space_trials_*.csv  state_space_recovery_trials.py [--condition]
  landing/balance_cells*.csv                          state_space_balance_cells.py
  path_tests/{drift_direction,repertoire_before_mean,trial_alpha}.csv   state_space_path_tests.py [--trial-alpha]
Page conventions: sessions = open markers (shape = animal, color = epoch), pooled = filled marker + 95 % CI in the epoch
color, no lines across epochs, stars in the top band, no top spine, no figure numbers inside images.
Writes PNGs to Widefield/state_space_recovery/page_figs/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import state_space_sim as S  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402  (PNG + svg/ twin, Priya 11-06)

ROOT = S.ROOT
# WF_SS_ALIGN as in state_space_sim: inputs from <dir>_<align>, figures named <name>_<align> (cue = no suffix)
ALIGN = S.ALIGN
TAG = "" if ALIGN == "cue" else f"_{ALIGN}"
# WF_SS_BASIS (Priya 11-08 "render and add the SVD basis version of section 3-7 figures" / "and LocaNMF"): inputs from
# <dir><align><basis>, figures named <name><align><basis>, the basis written top left. Allen = no suffix (unchanged).
BTAG = S.btag()
ALIGN_LABEL = S.ALIGN_LABEL
POST = ["acute", "subacute", "chronic"]
EPOCHS = ["pre"] + POST
POS = S.POS                               # canonical spout order
POS_LABEL = {p: p.replace("_", " ") for p in POS}
COND = [("target", "trials", "by target (all trials)", "k"),
        ("on_target", "trials_on_target", "on-target trials only", "#2166ac"),
        ("landing", "trials_landing", "grouped by landing", "#e08214")]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.spines.top": False, "axes.spines.right": False})


def sessions(sub):
    """Session table of a trials run; `sub` may carry "{TAG}" to place the alignment tag itself (variant runs)."""
    name = (sub.format(TAG=TAG) if "{TAG}" in sub else f"{sub}{TAG}") + BTAG
    f = ROOT / name / "state_space_trials_sessions.csv"
    if BTAG and not f.exists():                 # a basis run without this variant: the series is left out
        return None
    return pd.read_csv(f)


def deltas(df, col, key=("animal",)):
    """Session value minus its animal's mean pre value (pre sessions included: the null spread)."""
    pre = df[df.epoch == "pre"].groupby(list(key))[col].mean()
    d = df.copy()
    d["d"] = d[col] - d.set_index(list(key)).index.map(pre).to_numpy()
    return d


def change(df, col, rng):
    """animal-weighted boot_delta per post epoch: {epoch: (point, lo, hi, n_up, n_animals)}."""
    pre = {a: g[g.epoch == "pre"][col].dropna().tolist() for a, g in df.groupby("animal")}
    out = {}
    for ep in POST:
        post = {a: g[g.epoch == ep][col].dropna().tolist() for a, g in df.groupby("animal")}
        post = {a: v for a, v in post.items() if v and pre.get(a)}
        iv = kit.boot_delta(post, pre, rng) if post else None
        up = sum(np.mean(post[a]) > np.mean(pre[a]) for a in post)
        out[ep] = (iv.point, iv.lo, iv.hi, up, len(post)) if iv is not None else (np.nan,) * 3 + (0, 0)
    return out


# STARS (Priya 11-07: "LMM can be back up, not primary"): PRIMARY "*" = the animal-weighted nested bootstrap 95 % CI of
# the change excludes 0 (boot_delta, top row of the band); BACKUP "\u2020" (second row) = analysis_kit.epoch_change_lmm
# (value ~ C(epoch) + (1 | animal) + (1 | animal:epoch), t on df = animals - 1) p < .05, uncorrected. Panels register
# the LMM p-values; `flush_marks` draws the daggers once per figure.
_MARKS: list = []


def lmm_p(df, col):
    """{epoch: p} from epoch_change_lmm on the session values of `col` (NaN where the model cannot test)."""
    d = df[["animal", "epoch", col]].dropna()
    if d.empty:
        return {}
    r = kit.epoch_change_lmm(d, value=col)
    return dict(zip(r.epoch, r.p))


def mark(ax, x, p, color, row=1, fontsize=10):
    _MARKS.append((ax, x, p, color, row, fontsize))


def flush_marks(fig):
    for ax, x, p, color, row, fs in (m for m in _MARKS if m[0].figure is fig):
        if np.isfinite(p) and p < 0.05:
            sig_marks.star_top(ax, x, "\u2020", color=color, row=row, fontsize=fs)
    _MARKS[:] = [m for m in _MARKS if m[0].figure is not fig]


def scatter_sessions(ax, d, x0, jit=0.13, alpha=0.55, size=16, color=None):
    rng = np.random.default_rng(1)
    for an, g in d.groupby("animal"):
        xs = x0 + rng.uniform(-jit, jit, len(g))
        cols = [color or EPOCH_LINE[e] for e in g.epoch]
        ax.scatter(xs, g.d, marker=ANIMAL_MARKER[an], facecolors="none", edgecolors=cols, s=size, lw=0.8, alpha=alpha,
                   zorder=2)


def epoch_panel(ax, df, col, title, rng, unit=""):
    """x = epoch; faded session dots (minus own pre mean) + pooled change and 95 % bootstrap CI; * = CI excludes 0
    (primary), dagger = LMM p < .05 (backup)."""
    d = deltas(df, col).dropna(subset=["d"])
    ch = change(df, col, rng)
    ps = lmm_p(df, col)
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    for i, ep in enumerate(EPOCHS):
        scatter_sessions(ax, d[d.epoch == ep], i)
        if ep == "pre":
            continue
        pt, lo, hi, up, n = ch[ep]
        ax.errorbar(i + 0.3, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", color=EPOCH_LINE[ep], ms=6, lw=2, capsize=0,
                    zorder=4)
        if lo > 0 or hi < 0:
            sig_marks.star_top(ax, i + 0.3, "*", color=EPOCH_LINE[ep])
        mark(ax, i + 0.3, ps.get(ep, np.nan), EPOCH_LINE[ep])
        ax.annotate(f"{up}/{n}", (i + 0.3, 0), xytext=(0, -14), textcoords="offset points", ha="center",
                    fontsize=7, color=EPOCH_LINE[ep], annotation_clip=False)
    ax.set_xticks(range(4), EPOCHS)
    ax.set_xlim(-0.5, 3.6)
    ax.set_title(title)
    ax.set_ylabel(f"change vs own pre{unit}")


def legend_animals_epochs(fig, y=0.995):
    hs = [plt.Line2D([], [], marker=ANIMAL_MARKER[a], ls="", mfc="none", mec="k", label=a) for a in sorted(ANIMAL_MARKER)]
    hs += [plt.Line2D([], [], marker="o", ls="", mfc=EPOCH_LINE[e], mec=EPOCH_LINE[e], label=e) for e in EPOCHS]
    fig.legend(handles=hs, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, y))


def save(fig, out, name):
    if TAG and name in ("ss_repertoire", "ss_drift", "ss_path_over_days"):     # titles without the alignment
        fig.text(0.0, 1.0, ALIGN_LABEL[ALIGN], fontsize=9, fontweight="bold", va="bottom")
    if BTAG:
        fig.text(0.0, 1.035, S.BASIS_LABEL[S.BASIS], fontsize=9.5, fontweight="bold", va="bottom", color="#7570b3"
                 if S.BASIS == "svd" else "#1b9e77")
    flush_marks(fig)
    sig_marks.reserve_headroom(fig)
    save_png_svg(fig, out / f"{name}{TAG}{BTAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- figures

SIM_ROWS = [("none", 0, "no change (false-positive rate)"), ("rot12", 10, "rotation, main plane 10°"),
            ("rot12", 20, "rotation, main plane 20°"), ("rot12", 30, "rotation, main plane 30°"),
            ("rot", 20, "rotation, random plane 20°"), ("rot", 45, "rotation, random plane 45°"),
            ("gain_rand", 0.15, "per-area gains, log SD .15"), ("gain_rand", 0.3, "per-area gains, log SD .30"),
            ("gain_ipsi", 0.8, "ipsilesional hemisphere x.8"), ("gain_ipsi", 0.6, "ipsilesional hemisphere x.6"),
            ("amp", 1.25, "global amplitude x1.25"), ("amp", 1.5, "global amplitude x1.5"),
            ("shift", 0.25, "shared shift .25"), ("shift", 0.5, "shared shift .5"),
            ("newdim", 0.05, "new dimension, 5 % var"), ("newdim", 0.15, "new dimension, 15 % var"),
            ("spread", 0.5, "trial spread +.5 SD"), ("spread", 1.0, "trial spread +1 SD")]
SIM_COLS = [("r", "pattern r"), ("amp", "amplitude"), ("rot_gain", "rotation\nbeyond gain"),
            ("out_var", "outside\npre space"), ("out_resid", "outside,\nbeyond gain"), ("gain_sd", "per-area\ngain SD"),
            ("shared_dist", "shared\nshift"), ("spread", "spread\n(raw)"), ("spread_norm", "spread\n(amp-free)")]


def fig_sim(out, sub="sim", name="ss_sim_detection", what=""):
    """``sub`` = the sim output folder (per basis: sim_lick_svd, sim_lick_lnmfbm; running list B9 / B15)."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
    for ax, n, ttl in zip(axs, (0, 15), ("all trials", "15 trials per spout (acute-like)")):
        t = pd.read_csv(ROOT / sub / f"sim_detection_n{n}.csv")
        M = np.array([[t[(t.plant == p) & np.isclose(t.level, lv)][c].iloc[0] for c, _ in SIM_COLS]
                      for p, lv, _ in SIM_ROWS])
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=7,
                        color="w" if M[i, j] < 0.6 else "k")
        ax.set_xticks(range(len(SIM_COLS)), [c[1].replace("\n", " ") for c in SIM_COLS], fontsize=7.5, rotation=40,
                      ha="right")
        ax.set_yticks(range(len(SIM_ROWS)), [r[2] for r in SIM_ROWS])
        ax.set_title(ttl)
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axs, shrink=0.6, label="share of planted pre sessions detected (outside the null's 95 %)")
    fig.suptitle("Method check: known changes planted into pre-stroke sessions, scored session vs session" + what, y=0.99)
    if name == "ss_sim_detection":
        save(fig, out, name)
    else:                                       # basis runs: the folder already names the alignment
        flush_marks(fig)
        save_png_svg(fig, out / f"{name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


MAIN = [("r", "pattern r with pre\n(Δ correlation)"), ("amp", "amplitude ratio\n(Δ ratio; pre-pre ≈ 1)"), ("out_resid_t", "outside pre space, beyond gains\n(Δ fraction of variance)"),
        ("rot_gain", "rotation beyond gain\n(Δ held-out R²)"), ("shift_snr", "shared shift\n(Δ squared distance / noise expectation)"),
        ("spread_raw", "trial spread, raw\n(Δ ratio to a pre session)"), ("spread_norm", "trial spread / amplitude²\n(Δ ratio)"),
        ("dist_pre", "distance from pre sessions\n(Δ, pre-pre median = 1)"), ("step", "day-to-day step\n(Δ, pre median = 1)")]


def fig_change(out, rng):
    df = sessions("trials")
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    for ax, (c, t) in zip(axs.ravel(), MAIN):
        epoch_panel(ax, df, c, t, rng)
    legend_animals_epochs(fig)
    fig.suptitle("Single-trial state space, session vs each pre session, within spout "
                 f"(target grouping; {ALIGN_LABEL[ALIGN]})", y=1.02)
    fig.tight_layout()
    save(fig, out, "ss_change_vs_pre")


COND_CONTACT = [("target", "trials", "contact + incomplete (primary)", "k"),
                ("contact", "trials_contact", "contact only (robustness)", "#2166ac"),
                ("incomplete", "trials_incomplete", "incomplete only (descriptive)", "#e08214")]
COLS_CONTACT = [("r", "pattern r with pre\n(Δ correlation)"), ("amp", "amplitude ratio\n(Δ ratio; pre-pre ≈ 1)"), ("out_resid_t", "outside pre space, beyond gains\n(Δ fraction of variance)"),
                ("shift_snr", "shared shift\n(Δ squared distance / noise expectation)"), ("alpha", "position pre (0) -> chronic (1)\n(Δ fraction of pre -> chronic distance)"),
                ("chronic_like", "share of chronic-like trials\n(Δ proportion of trials)")]


COND_XFORM = [("raw", "trials", "raw areas (primary)", "k"),
              ("z", "trials{TAG}_xz", "areas z-scored", "#7570b3"),
              ("gsr", "trials{TAG}_xgsr", "cortex-wide mode removed", "#1b9e77"),
              ("spout", "trials{TAG}_xspout", "spout-specific part only", "#d95f02")]
# BASIS COMPARISON (Priya 11-08 "figures?" of "one cohort SVD basis reads like the Allen areas"): the same readouts in
# the three spaces; each basis gets its own per-animal top-8 pre space. LocaNMF = per-animal joint components with the
# glue / bulb components dropped (_lnmfbm); SVD = one cohort basis on the glue-free brain mask (_svd).
COND_BASIS = [("allen", "trials", "Allen areas (62; primary)", "k"),
              ("lnmf", "trials{TAG}_lnmfbm", "LocaNMF components (per animal, glue dropped)", "#1b9e77"),
              ("svd", "trials{TAG}_svd", "cohort SVD (100 components, glue-free mask)", "#7570b3")]
_TITLES = dict(MAIN + COLS_CONTACT)
COLS_XFORM = [(k, _TITLES[k]) for k in ("r", "amp", "rot_gain", "out_resid_t", "shift_snr", "alpha")]


def fig_like(out, rng, cond=None, cols=None, name="ss_like_for_like",
             title="Like for like: trials grouped by target, on-target only, or by landing"):
    """Readouts under several trial selections (series colors); a series without a readout (incomplete-only has no
    session-pair readouts: too few incomplete trials per spout in pre sessions) is left out of that panel."""
    cond = cond or COND
    cols = cols or [("r", "pattern r with pre\n(Δ correlation)"), ("amp", "amplitude ratio\n(Δ ratio; pre-pre ≈ 1)"), ("rot_gain", "rotation beyond gain\n(Δ held-out R²)"),
                    ("out_resid_t", "outside pre space, beyond gains\n(Δ fraction of variance)"), ("shift_snr", "shared shift\n(Δ squared distance / noise expectation)"),
                    ("dist_pre", "distance from pre sessions\n(Δ, pre-pre median = 1)")]
    COND_ = cond
    dfs = {k: sessions(sub) for k, sub, _, _ in COND_}
    COND_ = [c for c in COND_ if dfs[c[0]] is not None]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.8))
    for ax, (c, t) in zip(axs.ravel(), cols):
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        for k, (key, _, lab, colr) in enumerate(COND_):
            df = dfs[key]
            if c not in df or df[c].notna().sum() == 0:
                continue
            d = deltas(df, c).dropna(subset=["d"])
            ch = change(df, c, rng)
            ps = lmm_p(df, c)
            for i, ep in enumerate(POST):
                x = i + (k - (len(COND_) - 1) / 2) * (0.78 / len(COND_))
                scatter_sessions(ax, d[d.epoch == ep], x, jit=0.05, alpha=0.25, size=9, color=colr)
                pt, lo, hi, up, n = ch[ep]
                ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", color=colr, ms=5, lw=1.8, zorder=4,
                            label=lab if i == 0 else None)
                if lo > 0 or hi < 0:
                    sig_marks.star_top(ax, x, "*", color=colr, row=0, fontsize=10)
                mark(ax, x, ps.get(ep, np.nan), colr, row=1, fontsize=9)
        ax.set_xticks(range(3), POST)
        ax.set_title(t)
        ax.set_ylabel("change vs own pre")
    axs[0, 0].legend(handles=[plt.Line2D([], [], marker="o", color=colr, lw=1.8, label=lab) for _, _, lab, colr in COND_],
                     frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle(f"{title} ({ALIGN_LABEL[ALIGN]})", y=1.0)
    fig.tight_layout()
    save(fig, out, name)


# Balance area sets (Priya 11-09: "motor version as well (ALM, tjM1)"): table / figure suffix -> (title, short label,
# areas). Tables from state_space_balance_{cells,angle}.py [--motor].
BAL_AREAS = {"": ("Orofacial", "orofacial", "SSp-m, SSp-n, SSs"), "_motor": ("Motor", "motor", "ALM, tjM1")}


def fig_balance(out, rng, sfx=""):
    title, _, areas = BAL_AREAS[sfx]
    tgt = pd.read_csv(ROOT / "landing" / f"balance_cells_target{sfx}.csv")
    cel = pd.read_csv(ROOT / "landing" / f"balance_cells_sessions{sfx}.csv")
    on = cel[cel.target == cel.land]
    ang = pd.read_csv(ROOT / "landing" / f"balance_angle_sessions{sfx}.csv").rename(columns={"bal_matched": "bal"})
    fig, axs = plt.subplots(3, 6, figsize=(16, 9.6), sharey="row")
    rows = ((tgt, "all trials, by target"), (on, "landed on target only"),
            (ang, "matched for target + landing"))
    for row, (df, lab) in enumerate(rows):
        for j, p in enumerate(POS):
            ax = axs[row, j]
            g = df[df.target == p].rename(columns={"bal": "b"})
            epoch_panel(ax, g, "b", f"{POS_LABEL[p]}\n{lab}", rng)
            if j:
                ax.set_ylabel("")
            else:
                ax.set_ylabel("balance change vs own pre\nlog2 contra / ipsi")
            ax.tick_params(axis="x", labelsize=7)
    legend_animals_epochs(fig, y=1.02)
    fig.suptitle(f"{title} hemispheric balance ({areas}; 0-1 s after the cue), per spout: by target, "
                 "landed on target vs matched for lateral landing", y=1.05)
    fig.tight_layout()
    save(fig, out, f"ss_balance_by_spout{sfx}")


def balance_range(cel):
    """Per session: range (max - min) of the on-target balance across the five spouts other than far contra (acutely
    far contra has almost no on-target first licks, so it is left out in every epoch for a fair comparison)."""
    on = cel[cel.target == cel.land]
    w = on.pivot_table(index=["animal", "epoch", "label"], columns="target", values="bal")
    five = [p for p in POS if p != "far_contra"]
    w = w.reindex(columns=five).dropna()
    return (w.max(1) - w.min(1)).rename("rng").reset_index()


def fig_balance_cross(out, rng, sfx=""):
    _, short, areas = BAL_AREAS[sfx]
    c = pd.read_csv(ROOT / "landing" / f"balance_cells{sfx}.csv")
    cel = pd.read_csv(ROOT / "landing" / f"balance_cells_sessions{sfx}.csv")
    cats = [("pre", "far_contra", "far_contra", "pre: aimed far contra,\nlanded far contra"),
            ("pre", "far_ipsi", "far_ipsi", "pre: aimed far ipsi,\nlanded far ipsi"),
            ("pre", "far_contra", "far_ipsi", "pre: aimed far contra,\nlanded far ipsi"),
            ("acute", "far_ipsi", "far_ipsi", "acute: aimed far ipsi,\nlanded far ipsi"),
            ("acute", "far_contra", "far_ipsi", "acute: aimed far contra,\nlanded far ipsi"),
            ("chronic", "far_contra", "far_contra", "chronic: aimed far contra,\nlanded far contra")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [2.6, 1]})
    epoch_panel(ax2, balance_range(cel), "rng", "spread of the balance across spouts\n(on target, 5 spouts without "
                "far contra)", rng)
    ax2.set_ylabel("change in range (max - min), log2")
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    for i, (ep, t, ld, _) in enumerate(cats):
        for k, an in enumerate(sorted(ANIMAL_MARKER)):
            v = c[(c.animal == an) & (c.epoch == ep) & (c.target == t) & (c.land == ld)]
            if len(v):
                ax.scatter(i + (k - 1.5) * 0.09, v.bal.iloc[0], marker=ANIMAL_MARKER[an], facecolors="none",
                           edgecolors=EPOCH_LINE[ep], s=40, lw=1.2)
    ax.set_xticks(range(len(cats)), [x[3] for x in cats], fontsize=8)
    ax.set_ylabel(f"balance, log2 contra / ipsi\n({short}: {areas}; 0-1 s after the cue)")
    ax.set_title("same landing, different target (first licks pooled over an animal's sessions, n >= 5)")
    legend_animals_epochs(fig, y=1.08)
    fig.tight_layout()
    save(fig, out, f"ss_balance_target_vs_landing{sfx}")


def fig_path(out):
    df = sessions("trials")
    cols = [("alpha", "position on the pre (0) -> chronic (1) line"),
            ("chronic_like", "share of trials nearer the chronic template"),
            ("dist_pre", "distance from pre sessions (pre-pre median = 1)"),
            ("step", "step from the previous session (pre median = 1)")]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (c, t) in zip(axs, cols):
        for an, g in df.groupby("animal"):
            ax.scatter(g.day, g[c], marker=ANIMAL_MARKER[an], facecolors="none",
                       edgecolors=[EPOCH_LINE[e] for e in g.epoch], s=22, lw=0.9)
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        if c in ("dist_pre", "step"):
            ax.axhline(1, color="0.6", lw=0.8)
        ax.set_title(t)
        ax.set_xlabel("days from stroke")
    legend_animals_epochs(fig, y=1.1)
    fig.tight_layout()
    save(fig, out, "ss_path_over_days")


def fig_cloud(out):
    t = pd.read_csv(ROOT / f"path_tests{TAG}{BTAG}" / "trial_alpha.csv")
    bins = np.linspace(-4, 5, 46)                                   # trials outside are dropped, not piled at the edge
    ctr = 0.5 * (bins[1:] + bins[:-1])
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.6), sharey=True)
    groups = [("all animals", t)] + [(an, g) for an, g in t.groupby("animal")]
    for ax, (ttl, g) in zip(axs, groups):
        for ep in EPOCHS:
            e = g[g.epoch == ep]
            for _, s in e.groupby("label"):
                h, _ = np.histogram(s.alpha, bins, density=True)
                ax.plot(ctr, h, color=EPOCH_LINE[ep], lw=0.4, alpha=0.25)
            if len(e):
                h, _ = np.histogram(e.alpha, bins, density=True)
                ax.plot(ctr, h, color=EPOCH_LINE[ep], lw=2.4, label=ep)
                ax.axvline(e.alpha.median(), color=EPOCH_LINE[ep], lw=1, ls=":")
        for x in (0, 1):
            ax.axvline(x, color="0.5", lw=0.8)
        ax.set_title(ttl)
        ax.set_xlabel("trial position, pre (0) -> chronic (1)")
    axs[0].set_ylabel("density (thin = sessions)")
    axs[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Single-trial clouds along the pre -> chronic line, within spout: does the cloud slide, or do "
                 f"chronic-like trials appear first? ({ALIGN_LABEL[ALIGN]})", y=1.03)
    fig.tight_layout()
    save(fig, out, "ss_trial_clouds")


def fig_repertoire(out, rng):
    r = pd.read_csv(ROOT / f"path_tests{TAG}{BTAG}" / "repertoire_before_mean.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    for an, g in r.groupby("animal"):
        ax.scatter(g.expected_slid, g.observed, marker=ANIMAL_MARKER[an], facecolors="none",
                   edgecolors=[EPOCH_LINE[e] for e in g.epoch], s=28, lw=1)
    lim = [0, max(r.observed.max(), r.expected_slid.max()) * 1.05]
    ax.plot(lim, lim, color="0.5", lw=0.8)
    ax.set_xlabel("expected if the pre cloud slid to the session's position")
    ax.set_ylabel("observed share of chronic-like trials")
    ax.set_title("each session: observed vs slid cloud")
    ax = axs[1]
    ax.axhline(0, color="0.6", lw=0.8)
    for i, ep in enumerate(("acute", "subacute")):
        g = r[r.epoch == ep].assign(d=lambda x: x.excess)
        scatter_sessions(ax, g, i)
        iv = kit.boot_ci({a: h.excess.tolist() for a, h in g.groupby("animal")}, rng)
        ax.errorbar(i + 0.3, iv.point, yerr=[[iv.point - iv.lo], [iv.hi - iv.point]], fmt="o", color=EPOCH_LINE[ep],
                    ms=6, lw=2)
        if iv.lo > 0 or iv.hi < 0:
            sig_marks.star_top(ax, i + 0.3, "*", color=EPOCH_LINE[ep])
    ax.set_xticks([0, 1], ["acute", "subacute"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel("excess chronic-like share (observed - slid)")
    ax.set_title("excess by epoch")
    legend_animals_epochs(fig, y=1.08)
    fig.tight_layout()
    save(fig, out, "ss_repertoire")


def fig_drift(out):
    p = pd.read_csv(ROOT / f"trials{TAG}{BTAG}" / "state_space_trials_pairs.csv")
    dd = pd.read_csv(ROOT / f"path_tests{TAG}{BTAG}" / "drift_direction.csv")
    fig, axs = plt.subplots(1, 5, figsize=(18, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    for ax, (an, g) in zip(axs, p.groupby("animal")):
        for ep in ("pre", "chronic"):
            h = g[(g.ep_a == ep) & (g.ep_b == ep)]
            ax.scatter(h.gap, h.dist, marker=ANIMAL_MARKER[an], facecolors="none", edgecolors=EPOCH_LINE[ep], s=18,
                       lw=0.8, label=f"within {ep}")
            if len(h) > 3:
                sl, ic = np.polyfit(h.gap, h.dist, 1)
                xs = np.array([0, h.gap.max()])
                ax.plot(xs, ic + sl * xs, color=EPOCH_LINE[ep], lw=1.6)
                ax.text(0.97, 0.05 if ep == "pre" else 0.15, f"{ep} {sl:+.3f}/day", transform=ax.transAxes,
                        ha="right", color=EPOCH_LINE[ep], fontsize=8)
        ax.set_title(an)
        ax.set_xlabel("days between the two sessions")
    axs[0].set_ylabel("distance between sessions\n(pre-pre median = 1, noise-corrected)")
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = axs[4]
    ax.axhline(0, color="0.6", lw=0.8)
    items = [("cos_drift_chronic", "cos(pre drift,\nchronic - pre)", "chronic"),
             ("share_chronic_along_drift", "share of chronic - pre\nalong the drift", "chronic"),
             ("cos_drift_acute", "cos(pre drift,\nacute - pre)", "acute")]
    for i, (c, _, ep) in enumerate(items):
        for k, row in enumerate(dd.itertuples()):
            ok = row.drift_split_half_cos >= 0.5
            ax.scatter(i + (k - 1.5) * 0.1, getattr(row, c), marker=ANIMAL_MARKER[row.animal],
                       facecolors="none", edgecolors=EPOCH_LINE[ep], s=40, lw=1.2, alpha=1 if ok else 0.35)
    ax.set_xticks(range(3), [x[1] for x in items], fontsize=8)
    ax.set_title("pre-stroke drift direction (faded = unreliable,\nsplit-half cos < .5: PS95)")
    fig.tight_layout()
    save(fig, out, "ss_drift")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "page_figs"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    figs = {"sim": lambda: fig_sim(out), "change": lambda: fig_change(out, rng), "like": lambda: fig_like(out, rng),
            "xform": lambda: fig_like(out, rng, COND_XFORM, COLS_XFORM, "ss_xform",
                                      "Dimensionality variants: the same readouts on raw, z-scored, cortex-wide-mode-removed "
                                      "and spout-specific activity"),
            "basis": lambda: fig_like(out, rng, COND_BASIS, COLS_XFORM, "ss_basis",
                                      "Does the basis matter? The same readouts in Allen areas, LocaNMF components and "
                                      "one cohort SVD basis"),
            "simb": lambda: [fig_sim(out, f"sim{TAG}_{b}", f"ss_sim_detection{TAG}_{b}", f" ({lab}; {ALIGN_LABEL[ALIGN]})")
                             for b, lab in (("svd", "cohort SVD basis"), ("lnmfbm", "LocaNMF components"))
                             if (ROOT / f"sim{TAG}_{b}").exists()],
            "contact": lambda: fig_like(out, rng, COND_CONTACT, COLS_CONTACT, "ss_contact_sets",
                                       "Contact + incomplete (primary) vs contact-only vs incomplete-only first licks"),
            "balance": lambda: fig_balance(out, rng), "cross": lambda: fig_balance_cross(out, rng),
            "balance_motor": lambda: fig_balance(out, rng, "_motor"),
            "cross_motor": lambda: fig_balance_cross(out, rng, "_motor"),
            "path": lambda: fig_path(out), "cloud": lambda: fig_cloud(out),
            "repertoire": lambda: fig_repertoire(out, rng), "drift": lambda: fig_drift(out)}
    for k, f in figs.items():
        if not a.only or k in a.only:
            f()


if __name__ == "__main__":
    main()
