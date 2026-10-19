"""Figures for the combined task + movement encoding model (`wfield_local.combined_encoding`), Musall et al. 2019 Fig. 4-5
style, plus the spout-position and executed-vs-intended readouts (Priya 2026-10-18: "generate figures that make the most
sense, including musall-style as well as figures comparing across spout positions and the executed vs intended").

    python scripts/combined_encoding_figures.py [--run lp_r4_heatmap_20261013__clean_v36] [--order movement_first]
                                                [--baseline all|late]

Averaging: within animal first (mean over the animal's sessions in the epoch), then the mean over animals; error bars /
bands = SEM over animals, dots = animals (CLAUDE.md rule 8). Maps: per-session block R^2 = 1 - SSE / SST on the shared
Allen grid (2 x 2 px blocks), then the same averaging; Allen outlines from the atlas. Stats on changes vs pre:
`orofacial_stats.session_change_tests` (session Mann-Whitney + BH-FDR, animal-weighted bootstrap) on the cortex values.

Measures (see the module docstring): alone = cvR^2 of a single-variable model (upper bound); dR2 = full minus the
variable shuffled in the full orthogonalised model (movement_first: shared variance credited to movement -> position's
dR2 is conservative; position shuffle = label permutation, so only position SPECIFICITY is removed). Fig. 5:
task-independent = cvR^2(task + v) - cvR^2(task); task-aligned = alone(v) - task-independent.

Outputs: <LP_analysis>/<run>/imaging_encoding/<order>/figures{,_allpre}/ (+ svg/) and tables{,_allpre}/; AUGUST-ONLY pre baseline is PRIMARY (Priya 2026-10-18); 'all' = June-including sensitivity -> *_allpre.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VARS = ("cue", "arrival", "pos_enl", "pos_cue", "licks", "exec_angle", "tongue", "jaw", "nose", "running")
GROUPS = ("task", "instructed", "uninstructed")
GROUP_OF = {"cue": "task", "arrival": "task", "pos_enl": "task", "pos_cue": "task", "licks": "instructed",
            "exec_angle": "instructed",
            "tongue": "instructed", "jaw": "instructed", "nose": "uninstructed", "running": "uninstructed"}
#: group colours: distinct from the epoch line colours (k / magenta / goldenrod / teal, `orofacial_cohort.EPOCH_LINE`)
#: and the animal colours (blue / red / orange / green)
GCOL = {"task": "#6a3d9a", "instructed": "#8c510a", "uninstructed": "#5e7ea0"}
WINDOWS = ("all", "enl", "cue", "lick")
WIN_DESC = {"all": "all frames", "enl": "pre-cue 2 s (ENL)", "cue": "cue 0-1 s", "lick": "+-0.2 s of a lick peak"}
EPOCHS = ("pre", "acute", "subacute", "chronic")
#: TARGET vs EXECUTED (Priya 10-18: "most invested in the target vs executed angle question"). Order-free (raw columns)
#: R2 gains over the model with NO position information, in the model with and without tongue x / y:
#:   category = 6 spout categories (pos_enl + pos_cue)        -> {base}full                  - {base}drop:position
#:   target   = the spout's angle + angle^2 (cue + ENL blocks) -> {base}angle:position        - {base}drop:position
#:   executed = each lick's absolute angle + angle^2           -> {base}lickangle:position    - {base}drop:position
#:   target_lick = the SAME lick kernel / licks scaled by the spout angle -> {base}licktarget:position - {base}drop:position
#:              (the clean comparison: executed vs target_lick differ only in which angle scales the kernel)
#: base "" = raw:full / drop:position (tongue x / y in; x / y is itself a direction signal), "notxy:" = tongue
#: protrusion + speed only. The executed model replaces exec_angle (within-position deviation) by the absolute angle,
#: which subsumes it, so its gain is what WHERE THE TONGUE WENT adds beyond the within-position deviation.
TVX = {"xy": {"category": "raw:full", "target": "angle:position", "target_lick": "licktarget:position",
              "executed": "lickangle:position", "none": "drop:position"},
       "notxy": {"category": "notxy:full", "target": "notxy:angle:position", "target_lick": "notxy:licktarget:position",
                 "executed": "notxy:lickangle:position", "none": "notxy:drop:position"}}
TVX_KEYS = ("category", "target", "target_lick", "executed")
TVX_COL = {"category": "#1b9e77", "target": "#6a3d9a", "target_lick": "#b2abd2", "executed": "#d95f02"}
FOCUS = ("cortex", "MOs", "MOp", "SSp-m", "SSp-ul", "SSp-bfd", "SSp-n", "ACAd", "VISp")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10})
    return plt


def _title(fig, title: str, note: str, width: int = 140, y: float = 0.995) -> None:
    """Suptitle wrapped to the figure (Priya 10-18: titles must not run off the figure)."""
    import textwrap
    fig.suptitle("\n".join(textwrap.wrap(title, width) + textwrap.wrap(note, width)), fontsize=11, y=y, va="top")


def _save(fig, path: Path) -> None:
    from wfield_local.orofacial_cohort import save_png_svg
    save_png_svg(fig, path, dpi=130)
    _plt().close(fig)


def load(run: str, order: str, baseline: str, late_from: str):
    """Region table (long), per-session map dicts, PETH / SSE arrays and session metadata."""
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    d = oo.root(PathResolver()) / run / "imaging_encoding" / order / "sessions"
    metas, R, sess = [], [], {}
    for jf in sorted(d.glob("PS*_*.json")):
        m = json.loads(jf.read_text(encoding="utf-8"))
        if m["epoch"] not in EPOCHS:
            continue
        if baseline == "late" and m["epoch"] == "pre" and m["date"][4:8] < late_from:
            continue
        lab = m["label"]
        r = pd.read_csv(d / f"{lab}_regions.csv")
        r["area_b"] = r.area.str.replace("_left", "").str.replace("_right", "")
        r = r.assign(animal=m["animal"], date=m["date"], epoch=m["epoch"], day=m["day"])
        R.append(r)
        metas.append(m)
        sess[lab] = d / f"{lab}.npz"
    if not metas:
        raise SystemExit(f"no sessions in {d}")
    return pd.concat(R, ignore_index=True), pd.DataFrame(metas), sess


def region_wide(R: pd.DataFrame) -> pd.DataFrame:
    """One row per session x area (bilateral areas averaged): r2 per (model, window) + derived dR2 / Fig. 5 columns."""
    g = R.groupby(["animal", "date", "epoch", "day", "area_b", "model", "window"], as_index=False).r2.mean()
    W = g.pivot_table(index=["animal", "date", "epoch", "day", "area_b"], columns=["model", "window"], values="r2")
    out = pd.DataFrame(index=W.index)
    for w in WINDOWS:
        out[f"full|{w}"] = W[("full", w)]
        for v in (*VARS, *GROUPS, "position"):
            if ("alone:" + v, w) in W:
                out[f"alone:{v}|{w}"] = W[("alone:" + v, w)]
            if ("shuf:" + v, w) in W:
                out[f"dR2:{v}|{w}"] = W[("full", w)] - W[("shuf:" + v, w)]
            if ("drop:" + v, w) in W:
                out[f"dR2drop:{v}|{w}"] = W[("full", w)] - W[("drop:" + v, w)]
        for v in (*[x for x in VARS if GROUP_OF[x] != "task"], "instructed", "uninstructed"):
            if ("task+:" + v, w) in W:
                ind = W[("task+:" + v, w)] - W[("alone:task", w)]
                out[f"tind:{v}|{w}"] = ind
                out[f"talign:{v}|{w}"] = W[("alone:" + v, w)] - ind
        for st, mods in TVX.items():
            if (mods["none"], w) not in W:
                continue
            for k in TVX_KEYS:
                if (mods[k], w) in W:
                    out[f"tvx:{st}:{k}|{w}"] = W[(mods[k], w)] - W[(mods["none"], w)]
    return out.reset_index().rename(columns={"area_b": "pos"})            # `pos` = area, for session_change_tests


def _animal_stats(D: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per epoch: mean over animals of per-animal means, SEM over animals, the animal values."""
    a = D.groupby(["epoch", "animal"])[col].mean().reset_index()
    s = a.groupby("epoch")[col].agg(["mean", "sem", "count"])
    s["vals"] = a.groupby("epoch")[col].apply(list)
    return s.reindex([e for e in EPOCHS if e in s.index])


# ------------------------------------------------------------------------------------------------ figures

def fig4_bars(Wd: pd.DataFrame, path: Path, title: str, epoch: str = "pre") -> None:
    """Musall Fig. 4a-style: per variable and group, single-variable cvR^2 (open) vs unique dR2 (filled), cortex, one
    row per window; animal dots."""
    plt = _plt()
    D = Wd[(Wd.pos == "cortex") & (Wd.epoch == epoch)]
    names = [*VARS, *GROUPS]
    fig, axs = plt.subplots(len(WINDOWS), 1, figsize=(14, 3.0 * len(WINDOWS) + 1.4), squeeze=False)
    for i, w in enumerate(WINDOWS):
        ax = axs[i][0]
        for k, v in enumerate(names):
            col = GCOL[GROUP_OF.get(v, v)]
            for off, key, fill, hatch in ((-0.27, f"alone:{v}|{w}", "white", None), (0.0, f"dR2:{v}|{w}", col, None),
                                          (0.27, f"dR2drop:{v}|{w}", "white", "////")):
                if key not in D:
                    continue
                a = D.groupby("animal")[key].mean()
                ax.bar(k + off, a.mean(), 0.25, color=fill, edgecolor=col, lw=1.4, hatch=hatch,
                       yerr=a.sem() if len(a) > 1 else None, error_kw={"lw": 1, "capsize": 2})
                ax.plot(np.full(len(a), k + off) + np.linspace(-0.06, 0.06, len(a)), a.to_numpy(), "o", ms=3,
                        color="0.25", zorder=4)
        full = D.groupby("animal")[f"full|{w}"].mean()
        ax.axhline(full.mean(), color="k", ls="--", lw=1, label=f"full model {full.mean():.3f}")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(len(VARS) - 0.5, color="0.7", lw=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n if n in GROUPS else n for n in names], fontsize=9, rotation=0)
        ax.set_ylabel(f"cvR$^2$ / $\\Delta$R$^2$\n{w}: {WIN_DESC[w]}", fontsize=9)
        ax.legend(fontsize=10.5, frameon=False, loc="upper right")
        ax.tick_params(labelsize=10)
    n = D.groupby("animal").date.nunique().to_dict()
    _title(fig, title, f"Musall Fig. 4-style, cortex, {epoch.upper()} sessions {n}: OPEN = single-variable cvR$^2$ "
                 "(alone; own ridge penalty), FILLED = unique $\\Delta$R$^2$, shuffled in the full movement-first model "
                 "(variance shared with later groups credited to it), HATCHED = drop $\\Delta$R$^2$, full minus a refit "
                 "without it, unorthogonalised (shared variance credited to nobody); colour = group (task green, instructed orange, uninstructed purple); dots = animals, bars = "
                 "mean +- SEM over animals")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def _grid_from(npz: Path, keys: list[str]) -> dict:
    """{key: (H, W) block R^2 map, NaN off-brain} for one session (key = 'model|window'; 'dR2:v|w' derived)."""
    z = np.load(npz)
    mk = [str(k) for k in z["map_keys"]]
    M = z["maps"].astype(np.float64)
    H, Wd_ = (int(x) for x in z["shape"])
    flat = z["flat"]
    idx = {k: i for i, k in enumerate(mk)}
    out = {}
    for key in keys:
        name, w = key.split("|")
        tot = M[idx[f"TOTAL|{w}"]]
        ok = tot > 0

        den = np.where(ok, tot, 1.0)
        full_r2 = np.where(ok, 1 - M[idx[f"full|{w}"]] / den, np.nan)
        if name.startswith("dR2drop:"):
            v = full_r2 - np.where(ok, 1 - M[idx[f"drop:{name[8:]}|{w}"]] / den, np.nan)
        elif name.startswith("dR2:"):
            v = full_r2 - np.where(ok, 1 - M[idx[f"shuf:{name[4:]}|{w}"]] / den, np.nan)
        else:
            v = np.where(ok, 1 - M[idx[f"{name}|{w}"]] / den, np.nan)
        g = np.full(H * Wd_, np.nan)
        g[flat] = v
        out[key] = g.reshape(H, Wd_)
    return out


def cohort_maps(meta: pd.DataFrame, sess: dict, keys: list[str]) -> dict:
    """{epoch: {key: map}} within animal first, then over animals; also 'n' = sessions per animal."""
    acc: dict = {}
    for _, m in meta.iterrows():
        G = _grid_from(sess[m.label], keys)
        for k, g in G.items():
            acc.setdefault((m.epoch, m.animal, k), []).append(g)
    out = {}
    for e in EPOCHS:
        an = sorted({a for (ee, a, _k) in acc if ee == e})
        if not an:
            continue
        out[e] = {k: np.nanmean([np.nanmean(acc[(e, a, k)], 0) for a in an if (e, a, k) in acc], 0) for k in keys}
        out[e]["_n"] = {a: len(acc[(e, a, keys[0])]) for a in an}
    return out


def _outline(meta: pd.DataFrame):
    """Allen area boundaries on the 2 x 2 block grid (from one session's atlas; the grid is shared)."""
    from wfield_local import config
    for lab in meta.label:
        s = next((x for x in config.load_sessions() if x["label"] == lab), None)
        if s is None:
            continue
        f = Path(s["mc"]) / "wfield_local_results" / "allen_aligned_affine8v1" / "allen_area_atlas_native_grid.npy"
        if f.exists():
            a = np.load(f)[::2, ::2]
            b = np.zeros(a.shape, bool)
            b[:-1] |= a[:-1] != a[1:]
            b[:, :-1] |= a[:, :-1] != a[:, 1:]
            return np.where(b, 1.0, np.nan)
    return None


def _map_ax(ax, M, vmin, vmax, cmap, outline, title):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    if outline is not None and outline.shape == M.shape:
        ax.imshow(outline, cmap="Greys", vmin=0, vmax=1.2, alpha=0.55, interpolation="nearest")
    ax.set_title(title, fontsize=10.5)
    ax.set_axis_off()
    return im


def fig4_maps(meta, sess, path: Path, title: str, window: str, outline, epoch: str = "pre") -> None:
    """Musall Fig. 4b-style pixel maps: row 1 single-variable cvR^2, row 2 unique dR2; columns = variables + groups + full."""
    plt = _plt()
    names = [*VARS, *GROUPS]
    keys = ([f"alone:{v}|{window}" for v in names] + [f"dR2:{v}|{window}" for v in names]
            + [f"dR2drop:{v}|{window}" for v in names] + [f"full|{window}"])
    C = cohort_maps(meta[meta.epoch == epoch], sess, keys)
    if epoch not in C:
        return
    M = C[epoch]
    fig, axs = plt.subplots(3, len(names) + 1, figsize=(1.9 * (len(names) + 1) + 1.5, 7.6), squeeze=False)
    fig.subplots_adjust(top=0.80, wspace=0.05, hspace=0.12)
    va = np.nanpercentile(np.concatenate([M[k].ravel() for k in keys if k.startswith("alone")]), 99)
    vd = np.nanpercentile(np.concatenate([M[k].ravel() for k in keys if k.startswith("dR2")]), 99)
    for j, v in enumerate(names):
        imp = _map_ax(axs[2][j], M[f"dR2drop:{v}|{window}"], 0, vd, "viridis", outline, "")
    axs[2][-1].set_axis_off()
    axs[2][0].text(-0.15, 0.5, "drop $\\Delta$R$^2$", transform=axs[2][0].transAxes, rotation=90, va="center",
                   ha="right")
    fig.colorbar(imp, ax=list(axs[2][:-1]), shrink=0.7, pad=0.01)
    for j, v in enumerate(names):
        ima = _map_ax(axs[0][j], M[f"alone:{v}|{window}"], 0, va, "magma", outline, v)
        imd = _map_ax(axs[1][j], M[f"dR2:{v}|{window}"], 0, vd, "viridis", outline, "")
    imf = _map_ax(axs[0][-1], M[f"full|{window}"], 0, np.nanpercentile(M[f"full|{window}"], 99), "magma", outline,
                  "FULL")
    axs[1][-1].set_axis_off()
    axs[0][0].text(-0.15, 0.5, "alone cvR$^2$", transform=axs[0][0].transAxes, rotation=90, va="center", ha="right")
    axs[1][0].text(-0.15, 0.5, "unique $\\Delta$R$^2$", transform=axs[1][0].transAxes, rotation=90, va="center",
                   ha="right")
    fig.colorbar(ima, ax=list(axs[0][:-1]), shrink=0.7, pad=0.01)
    fig.colorbar(imd, ax=list(axs[1][:-1]), shrink=0.7, pad=0.01)
    fig.colorbar(imf, ax=axs[0][-1], shrink=0.7, pad=0.02)
    _title(fig, title, f"Musall Fig. 4-style maps, {epoch.upper()} sessions {M['_n']}, window {window} "
                 f"({WIN_DESC[window]}): per-pixel (2x2 block) cvR$^2$ of each single-variable model (top) and its unique "
                 "$\\Delta$R$^2$ in the full movement-first model (bottom); within animal, then mean over animals")
    _save(fig, path)


def fig_position_maps(meta, sess, path: Path, title: str, outline) -> None:
    """Position-specific dR2 maps across recovery: pos_enl (ENL window), pos_cue (cue window), exec_angle (lick
    window); row pairs = level and change vs pre."""
    plt = _plt()
    rows = [("dR2:pos_enl|enl", "spout position, ENL (pre-cue 2 s)"), ("dR2:pos_cue|cue", "spout position, cue 0-1 s"),
            ("dR2:pos_cue|lick", "spout position, around licks"), ("dR2:exec_angle|lick", "EXECUTED angle (within "
                                                                                          "position), licks")]
    keys = [k for k, _ in rows]
    C = cohort_maps(meta, sess, keys)
    eps = [e for e in EPOCHS if e in C]
    fig, axs = plt.subplots(len(rows), 2 * len(eps) - 1, figsize=(2.0 * (2 * len(eps) - 1) + 1.5, 2.1 * len(rows) + 1.4),
                            squeeze=False)
    for i, (k, lab) in enumerate(rows):
        v = max(np.nanpercentile(C[e][k], 99) for e in eps)
        dv = max(np.nanpercentile(np.abs(C[e][k] - C["pre"][k]), 99) for e in eps[1:]) if len(eps) > 1 else v
        for j, e in enumerate(eps):
            im = _map_ax(axs[i][j], C[e][k], 0, v, "viridis", outline, f"{e} ({sum(C[e]['_n'].values())} s)" if i == 0
                         else "")
        fig.colorbar(im, ax=list(axs[i][:len(eps)]), shrink=0.85, pad=0.01)
        for j, e in enumerate(eps[1:]):
            im2 = _map_ax(axs[i][len(eps) + j], C[e][k] - C["pre"][k], -dv, dv, "RdBu_r", outline,
                          f"{e} - pre" if i == 0 else "")
        if len(eps) > 1:
            fig.colorbar(im2, ax=list(axs[i][len(eps):]), shrink=0.85, pad=0.01)
        axs[i][0].text(-0.1, 0.5, lab, transform=axs[i][0].transAxes, rotation=90, va="center", ha="right", fontsize=10.5)
    _title(fig, title, "Position-specific unique $\\Delta$R$^2$ maps across recovery (left: level; right: change vs "
                 "pre). Position = label permutation (only position SPECIFICITY removed; shared arrival / cue response "
                 "kept); movement-first -> what movement cannot explain. Executed angle = lick kernel x the lick's "
                 "within-position angle deviation")
    _save(fig, path)


def fig_dR2_by_epoch(Wd: pd.DataFrame, ST: pd.DataFrame, path: Path, title: str, area: str = "cortex",
                     prefix: str = "dR2") -> None:
    """dR2 (filled) and alone (open) per epoch, one panel per variable / group, one row per window; animal dots; stars
    = change vs pre significant by both tests."""
    plt = _plt()
    names = [*VARS, *GROUPS]
    D = Wd[Wd.pos == area]
    fig, axs = plt.subplots(len(WINDOWS), len(names), figsize=(1.75 * len(names) + 1, 2.3 * len(WINDOWS) + 1.4),
                            squeeze=False, sharex=True)
    for i, w in enumerate(WINDOWS):
        for j, v in enumerate(names):
            ax = axs[i][j]
            col = GCOL[GROUP_OF.get(v, v)]
            key = f"{prefix}:{v}|{w}"
            if key not in D:
                ax.set_axis_off()
                continue
            s = _animal_stats(D, key)
            x = np.arange(len(s))
            ax.errorbar(x, s["mean"], yerr=s["sem"], color=col, marker="o", ms=4, lw=1.4, capsize=2)
            for xi, vals in zip(x, s["vals"]):
                ax.plot(np.full(len(vals), xi) + np.linspace(-0.12, 0.12, len(vals)), vals, ".", color="0.5", ms=3)
            ax.axhline(0, color="0.7", lw=0.7)
            if ST is not None and len(ST):
                for xi, e in zip(x, s.index):
                    r = ST[(ST.metric == key) & (ST.pos == area) & (ST.epoch == e)]
                    if len(r) and r.agree.iloc[0] == "both":
                        ax.text(xi, ax.get_ylim()[1], "*", ha="center", va="top", fontsize=13, color="k")
            ax.set_xticks(x)
            ax.set_xticklabels([e[:3] for e in s.index], fontsize=10)
            ax.tick_params(labelsize=9.5)
            if i == 0:
                ax.set_title(v, fontsize=9, color=col)
            if j == 0:
                ax.set_ylabel(f"$\\Delta$R$^2$, {w}", fontsize=10.5)
    _title(fig, title, f"Unique $\\Delta$R$^2$ by epoch, {area}: mean +- SEM over animals (within animal first), dots = "
                 "animals; * = change vs pre significant by BOTH the session Mann-Whitney (BH-FDR) and the animal-"
                 "weighted bootstrap")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, path)


def fig5_task_aligned(Wd: pd.DataFrame, path: Path, title: str, window: str = "all", area: str = "cortex") -> None:
    """Musall Fig. 5-style: for each movement variable / group, task-aligned vs task-independent explained variance,
    stacked, per epoch."""
    plt = _plt()
    names = [*[v for v in VARS if GROUP_OF[v] != "task"], "instructed", "uninstructed"]
    D = Wd[Wd.pos == area]
    fig, axs = plt.subplots(1, len(names), figsize=(2.0 * len(names) + 1, 4.2), squeeze=False, sharey=True)
    for j, v in enumerate(names):
        ax = axs[0][j]
        ka, ki = f"talign:{v}|{window}", f"tind:{v}|{window}"
        if ka not in D:
            ax.set_axis_off()
            continue
        sa, si = _animal_stats(D, ka), _animal_stats(D, ki)
        x = np.arange(len(sa))
        ax.bar(x, si["mean"], 0.6, color=GCOL[GROUP_OF.get(v, v)], label="task-independent")
        ax.bar(x, sa["mean"], 0.6, bottom=si["mean"], color="0.75", label="task-aligned")
        ax.errorbar(x, si["mean"], yerr=si["sem"], fmt="none", ecolor="k", lw=0.8, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([e[:3] for e in sa.index], fontsize=10.5)
        ax.set_title(v, fontsize=9)
        ax.tick_params(labelsize=10)
    axs[0][0].set_ylabel(f"cvR$^2$ ({area}, {window})", fontsize=9)
    axs[0][-1].legend(fontsize=10, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _title(fig, title, "Musall Fig. 5-style: each movement variable's explained variance split into TASK-ALIGNED "
                 "(grey; also explained by the task variables) and TASK-INDEPENDENT (coloured; cvR$^2$(task + v) - "
                 "cvR$^2$(task)); mean over animals, bar = SEM of the task-independent part")
    fig.tight_layout(rect=(0, 0, 0.92, 0.88))
    _save(fig, path)


def fig_time_resolved(meta, sess, path: Path, title: str, areas=("cortex", "MOs", "SSp-m"), kind: str = "shuf") -> None:
    """Cue-aligned time-resolved unique dR2 = (SSE_shuf - SSE_full) / SST per frame, per epoch; rows = areas."""
    plt = _plt()
    first = np.load(next(iter(sess.values())))
    t = first["peth_off_s"]
    trm = json.loads(Path(str(next(iter(sess.values()))).replace(".npz", ".json")).read_text())["tr_models"]
    A = json.loads(Path(str(next(iter(sess.values()))).replace(".npz", ".json")).read_text())["areas"]
    shufs = [m for m in trm if m.startswith(kind + ":")]
    acc: dict = {}
    for _, m in meta.iterrows():
        z = np.load(sess[m.label])
        sse, sst = z["sse"], z["sst"]
        for ar in areas:
            ai = [i for i, a in enumerate(A) if a == ar or a.rsplit("_", 1)[0] == ar]
            if not ai:
                continue
            ss_t = sst[ai].sum(0)
            full = sse[trm.index("full")][ai].sum(0)
            for sm in shufs:
                d = (sse[trm.index(sm)][ai].sum(0) - full) / np.where(ss_t > 0, ss_t, np.nan)
                acc.setdefault((ar, sm, m.epoch, m.animal), []).append(d)
    fig, axs = plt.subplots(len(areas), len(shufs), figsize=(2.4 * len(shufs) + 1, 2.3 * len(areas) + 1.4),
                            squeeze=False, sharex=True)
    for i, ar in enumerate(areas):
        for j, sm in enumerate(shufs):
            ax = axs[i][j]
            for e in EPOCHS:
                an = [a for (aa, s_, ee, a) in acc if aa == ar and s_ == sm and ee == e]
                if not an:
                    continue
                per = np.array([np.nanmean(acc[(ar, sm, e, a)], 0) for a in sorted(set(an))])
                mu = np.nanmean(per, 0)
                from wfield_local.orofacial_cohort import EPOCH_LINE
                ax.plot(t, mu, color=EPOCH_LINE[e], lw=1.3, label=e)
                if len(per) > 1:
                    se = np.nanstd(per, 0, ddof=1) / np.sqrt(len(per))
                    ax.fill_between(t, mu - se, mu + se, color=EPOCH_LINE[e], alpha=0.15, lw=0)
            ax.axvline(0, color="0.6", lw=0.7)
            ax.axhline(0, color="0.8", lw=0.7)
            if i == 0:
                ax.set_title(sm.split(":", 1)[1], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{ar}\n$\\Delta$R$^2$(t)", fontsize=10.5)
            if i == len(areas) - 1:
                ax.set_xlabel("s from cue", fontsize=10.5)
            ax.tick_params(labelsize=9.5)
    axs[0][-1].legend(fontsize=10, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _title(fig, title, "Time-resolved unique $\\Delta$R$^2$ around the cue (held-out SSE per frame, all engaged trials): "
                 "(SSE with the variable / group shuffled - SSE full) / SST; within animal, then mean +- SEM over "
                 "animals")
    fig.tight_layout(rect=(0, 0, 0.93, 0.92))
    _save(fig, path)


def fig_peth_partition(meta, sess, path: Path, title: str, epoch: str, areas=("cortex", "MOs", "SSp-m", "SSp-ul")) -> None:
    """Musall Fig. 5-style PETH partition: data, full prediction and its split by group, cue-aligned, per position."""
    plt = _plt()
    from wfield_local.orofacial_cohort import POS_ORDER, pos_disp
    js = {lab: json.loads(Path(str(f).replace(".npz", ".json")).read_text()) for lab, f in sess.items()}
    acc: dict = {}
    t = None
    for _, m in meta[meta.epoch == epoch].iterrows():
        z = np.load(sess[m.label])
        J = js[m.label]
        t = z["peth_off_s"]
        P = z["peth"]                                   # (parts, positions, areas, t)
        for ar in areas:
            ai = [i for i, a in enumerate(J["areas"]) if a == ar or a.rsplit("_", 1)[0] == ar]
            if not ai:
                continue
            for pi, p in enumerate(J["pos_order"]):
                for ci, part in enumerate(J["peth_parts"]):
                    acc.setdefault((ar, p, part, m.animal), []).append(np.nanmean(P[ci, pi, ai], 0))
    if t is None:
        return
    parts = ("data", "full", "task", "instructed", "uninstructed")
    sty = {"data": ("k", "-", 1.6), "full": ("0.5", "--", 1.2), "task": (GCOL["task"], "-", 1.2),
           "instructed": (GCOL["instructed"], "-", 1.2), "uninstructed": (GCOL["uninstructed"], "-", 1.2)}
    fig, axs = plt.subplots(len(areas), len(POS_ORDER), figsize=(2.6 * len(POS_ORDER) + 1, 2.2 * len(areas) + 1.4),
                            squeeze=False, sharex=True, sharey="row")
    for i, ar in enumerate(areas):
        for j, p in enumerate(POS_ORDER):
            ax = axs[i][j]
            for part in parts:
                an = sorted({a for (aa, pp, pt, a) in acc if aa == ar and pp == p and pt == part})
                if not an:
                    continue
                mu = np.nanmean([np.nanmean(acc[(ar, p, part, a)], 0) for a in an], 0)
                c, ls, lw = sty[part]
                ax.plot(t, mu, color=c, ls=ls, lw=lw, label=part)
            ax.axvline(0, color="0.7", lw=0.7)
            if i == 0:
                from wfield_local.orofacial_cohort import _pos_color
                ax.set_title(pos_disp(p), fontsize=9, color=_pos_color(p, "L"))
            if j == 0:
                ax.set_ylabel(f"{ar}\n(SVD units)", fontsize=10.5)
            if i == len(areas) - 1:
                ax.set_xlabel("s from cue", fontsize=10.5)
            ax.tick_params(labelsize=9.5)
    axs[0][-1].legend(fontsize=10, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    n = meta[meta.epoch == epoch].groupby("animal").date.nunique().to_dict()
    _title(fig, title, f"PETH partition, {epoch.upper()} sessions {n}: data (black), the FULL model's prediction "
                 "(grey dashed) and its split by group (the full model fitted once; its prediction summed over each "
                 "group's regressors, no refit), cue-aligned, per spout position; within animal, then mean over animals")
    fig.tight_layout(rect=(0, 0, 0.93, 0.9))
    _save(fig, path)


def fig_exec_vs_intended(Wd: pd.DataFrame, ST: pd.DataFrame, path: Path, title: str) -> None:
    """Executed (within-position lick angle) vs intended (spout position) unique dR2: per epoch, cortex, and per area in
    pre (bilateral)."""
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 2.4]})
    D = Wd[Wd.pos == "cortex"]
    for ax, (w, pairs) in zip(axs[:2], (("lick", (("dR2:exec_angle|lick", "executed angle"),
                                                   ("dR2:pos_cue|lick", "spout position"))),
                                        ("cue", (("dR2:exec_angle|cue", "executed angle"),
                                                 ("dR2:pos_cue|cue", "spout position"))))):
        for k, (key, lab) in enumerate(pairs):
            s = _animal_stats(D, key)
            x = np.arange(len(s)) + (k - 0.5) * 0.25
            ax.errorbar(x, s["mean"], yerr=s["sem"], marker="os"[k], ms=5, lw=1.4, capsize=2,
                        color=("#d95f02", "#1b9e77")[k], label=lab)
            for xi, vals in zip(x, s["vals"]):
                ax.plot(np.full(len(vals), xi), vals, ".", color="0.55", ms=3)
        ax.axhline(0, color="0.7", lw=0.7)
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(list(s.index), fontsize=10.5)
        ax.set_title(f"cortex, {w} window ({WIN_DESC[w]})", fontsize=9)
        ax.set_ylabel("unique $\\Delta$R$^2$", fontsize=9)
        ax.legend(fontsize=10.5, frameon=False)
    ax = axs[2]
    P = Wd[(Wd.epoch == "pre") & (Wd.pos != "cortex")]
    areas = (P.groupby("pos")["dR2:pos_cue|lick"].mean().sort_values(ascending=False).index[:18])
    x = np.arange(len(areas))
    for k, (key, lab, c) in enumerate((("dR2:exec_angle|lick", "executed angle", "#d95f02"),
                                       ("dR2:pos_cue|lick", "spout position", "#1b9e77"))):
        a = P[P.pos.isin(areas)].groupby(["pos", "animal"])[key].mean().groupby("pos")
        ax.bar(x + (k - 0.5) * 0.38, a.mean().reindex(areas), 0.36, yerr=a.sem().reindex(areas), color=c, label=lab,
               error_kw={"lw": 0.8, "capsize": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(areas, rotation=60, fontsize=10, ha="right")
    ax.set_title("PRE sessions, by area (bilateral), lick window: top areas by spout-position $\\Delta$R$^2$", fontsize=9)
    ax.legend(fontsize=10.5, frameon=False)
    _title(fig, title, "EXECUTED vs INTENDED: unique $\\Delta$R$^2$ of the lick's executed angle (within-position "
                 "deviation, lick-onset kernel) vs the spout position (cue kernel, label permutation), movement-first. "
                 "Mean +- SEM over animals, dots = animals")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, path)


def fig_target_vs_executed(Wd: pd.DataFrame, path: Path, title: str, area: str = "cortex") -> None:
    """Rows: tongue x / y in, out. Columns: cue / lick / all windows. Per epoch, the R2 gained over the no-position model
    by the 6 categories, the TARGET angle, the EXECUTED (absolute lick) angle; last column: executed - target per animal."""
    plt = _plt()
    D = Wd[Wd.pos == area]
    wins = [w for w in ("cue", "lick", "all") if f"tvx:xy:target|{w}" in D]
    if not wins:
        print("  target-vs-executed: variant models missing (old-model outputs?) -- skipped")
        return
    fig, axs = plt.subplots(2, len(wins) + 1, figsize=(4.3 * (len(wins) + 1), 8.2), squeeze=False)
    for r, (st, lab) in enumerate((("xy", "tongue x / y IN"), ("notxy", "tongue x / y OUT (protrusion + speed only)"))):
        for c, w in enumerate(wins):
            ax = axs[r][c]
            for k, key in enumerate(TVX_KEYS):
                col = f"tvx:{st}:{key}|{w}"
                if col not in D:
                    continue
                s = _animal_stats(D, col)
                x = np.arange(len(s)) + (k - 1.5) * 0.16
                ax.errorbar(x, s["mean"], yerr=s["sem"], marker="os^D"[k], ms=5, lw=1.4, capsize=2,
                            color=TVX_COL[key], label=key)
                for xi, vals in zip(x, s["vals"]):
                    ax.plot(np.full(len(vals), xi), vals, ".", color="0.55", ms=3)
            ax.axhline(0, color="0.7", lw=0.7)
            ax.set_xticks(range(len(s)))
            ax.set_xticklabels(list(s.index))
            ax.set_title(f"{lab}\n{w} window ({WIN_DESC[w]})", fontsize=10.5)
            ax.set_ylabel("R$^2$ gain over no-position model")
            if r == 0 and c == 0:
                ax.legend(frameon=False)
        ax = axs[r][-1]
        for k, w in enumerate(wins):
            a, b = f"tvx:{st}:executed|{w}", f"tvx:{st}:target_lick|{w}"         # same kernel, same licks
            if a not in D or b not in D:
                continue
            s = _animal_stats(D.assign(_d=D[a] - D[b]), "_d")
            x = np.arange(len(s)) + (k - (len(wins) - 1) / 2) * 0.2
            ax.errorbar(x, s["mean"], yerr=s["sem"], marker="o", ms=5, lw=1.4, capsize=2,
                        color=("0.15", "#8c510a", "#5e7ea0")[k], label=w)
            for xi, vals in zip(x, s["vals"]):
                ax.plot(np.full(len(vals), xi), vals, ".", color="0.6", ms=3)
        ax.axhline(0, color="0.7", lw=0.7)
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(list(s.index))
        ax.set_title(f"{lab}\nexecuted - target, lick kernel (> 0 = tongue)", fontsize=10.5)
        ax.legend(frameon=False, title="window")
    _title(fig, title, f"TARGET vs EXECUTED angle, {area}: order-free R$^2$ gain over the model without position, for "
                 "the 6 spout categories, the target (spout) angle + angle$^2$ on the cue / ENL blocks ('target') or on the lick "
                 "kernel ('target_lick', the same licks as 'executed'), and each lick's absolute executed angle + "
                 "angle$^2$ on the lick kernel. Mean +- SEM over animals, dots = animals.")
    fig.tight_layout(h_pad=2.0)
    _save(fig, path)


def fig_target_vs_executed_sessions(Wd: pd.DataFrame, path: Path, title: str, area: str = "cortex",
                                    st: str = "notxy") -> None:
    """Every SESSION as a dot (animal colour), animal means as large markers, cohort mean of animal means as the black
    line (Priya 10-19: "are these figures showing all sessions or only each animal mean" -> show both).
    Rows: lick / cue window; columns: category, target (lick kernel), executed (lick kernel), executed - target."""
    from wfield_local import config
    plt = _plt()
    D = Wd[Wd.pos == area].copy()
    wins = [w for w in ("lick", "cue") if f"tvx:{st}:target_lick|{w}" in D]
    if not wins:
        print("  target-vs-executed (sessions): variant models missing -- skipped")
        return
    acol = config.animal_color()
    animals = sorted(D.animal.unique())
    keys = [("category", "6 spout categories"), ("target_lick", "TARGET (lick kernel)"),
            ("executed", "EXECUTED (lick kernel)"), ("diff", "executed - target (lick kernel)")]
    fig, axs = plt.subplots(len(wins), len(keys), figsize=(4.4 * len(keys), 3.9 * len(wins) + 1.2), squeeze=False)
    eps = [e for e in EPOCHS if e in set(D.epoch)]
    for r, w in enumerate(wins):
        D["diff"] = D[f"tvx:{st}:executed|{w}"] - D[f"tvx:{st}:target_lick|{w}"]
        for c, (k, lab) in enumerate(keys):
            col = "diff" if k == "diff" else f"tvx:{st}:{k}|{w}"
            ax = axs[r][c]
            means = []
            for ei, e in enumerate(eps):
                for ai, an in enumerate(animals):
                    v = D[(D.epoch == e) & (D.animal == an)][col].dropna().to_numpy()
                    if not len(v):
                        continue
                    xo = ei + (ai - (len(animals) - 1) / 2) * 0.16
                    ax.plot(xo + np.random.default_rng(ei * 10 + ai).uniform(-0.04, 0.04, len(v)), v, "o", ms=3,
                            color=acol.get(an, "0.5"), alpha=0.45, mec="none")
                    ax.plot(xo, v.mean(), "D", ms=7, color=acol.get(an, "0.5"), mec="k", mew=0.6,
                            label=an if (r == 0 and c == 0 and ei == 0) else None)
                am = D[D.epoch == e].groupby("animal")[col].mean()
                means.append(am.mean())
            ax.plot(range(len(eps)), means, "-", color="k", lw=2, zorder=5, label="mean of animals" if (r == 0 and c == 0) else None)
            ax.axhline(0, color="0.7", lw=0.7)
            ax.set_xticks(range(len(eps)))
            ax.set_xticklabels(eps)
            ax.set_title(f"{lab}\n{w} window ({WIN_DESC[w]})", fontsize=10.5)
            if c == 0:
                ax.set_ylabel("R$^2$ gain over no-position model")
    axs[0][0].legend(frameon=False, fontsize=9, loc="upper left")
    n = D.groupby(["animal", "epoch"]).date.nunique().unstack("epoch").reindex(columns=eps).fillna(0).astype(int)
    _title(fig, title, f"{area}, tongue x/y {'OUT' if st == 'notxy' else 'IN'}: small dots = sessions, diamonds = animal "
                 "means, black line = mean of animal means. Sessions per animal x epoch: "
                 + "; ".join(f"{a} " + "/".join(str(x) for x in n.loc[a]) for a in n.index) + f" ({'/'.join(eps)}).",
           width=120)
    fig.tight_layout()
    _save(fig, path)


def main(argv=None) -> int:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.orofacial_stats import session_change_tests
    from wfield_local.paths import PathResolver
    oc._repo_on_path()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--order", default="movement_first")
    ap.add_argument("--baseline", choices=("all", "late"), default="late", help="AUGUST-ONLY pre baseline is PRIMARY (Priya 2026-10-18); 'all' = June-including sensitivity -> *_allpre")
    a = ap.parse_args(argv)
    base = oo.root(PathResolver()) / a.run / "imaging_encoding" / a.order
    sfx = "_allpre" if a.baseline == "all" else ""
    figs, tabs = base / f"figures{sfx}", base / f"tables{sfx}"
    writeguard.assert_writable(base)
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    R, meta, sess = load(a.run, a.order, a.baseline, oc.LATE_PRE_FROM)
    Wd = region_wide(R)
    Wd.to_csv(tabs / "region_measures.csv", index=False, float_format="%.5g")
    title = (f"Combined task + movement encoding model ({a.order}) -- {a.run} -- baseline: "
             f"{'LATE (August) pre only' if a.baseline == 'late' else 'all curated pre'} -- sessions: "
             f"{meta.groupby('animal').label.nunique().to_dict()}")
    print(title)
    pd.set_option("display.width", 220)
    print("per-animal cortex dR2 by epoch (rule 8: animals before the pooled test):")
    for key in ("dR2:position|all", "dR2drop:position|all", "tvx:notxy:target|lick", "tvx:notxy:executed|lick",
                "dR2:pos_enl|enl", "dR2:pos_cue|cue", "dR2:pos_cue|lick", "dR2:exec_angle|lick", "dR2:instructed|all",
                "dR2:uninstructed|all", "full|all"):
        if key not in Wd:
            continue
        t_ = Wd[Wd.pos == "cortex"].groupby(["animal", "epoch"])[key].mean().unstack("epoch")
        print(f"  {key}\n" + t_.reindex(columns=list(EPOCHS)).round(4).to_string())
    metrics = [c for c in Wd.columns if c.split(":")[0] in ("dR2", "dR2drop", "alone", "full", "tind", "talign", "tvx")]
    ST = session_change_tests(Wd[Wd.pos.isin(FOCUS)], metrics)
    ST.to_csv(tabs / "stats_vs_pre.csv", index=False, float_format="%.4g")
    outline = _outline(meta)
    fig4_bars(Wd, figs / "fig4_bars_pre.png", title)
    for w in ("all", "enl", "cue", "lick"):
        fig4_maps(meta, sess, figs / f"fig4_maps_pre_{w}.png", title, w, outline)
    fig_position_maps(meta, sess, figs / "position_dR2_maps_by_epoch.png", title, outline)
    fig_dR2_by_epoch(Wd, ST, figs / "dR2_by_epoch_cortex.png", title)
    for ar in ("MOs", "SSp-m"):
        fig_dR2_by_epoch(Wd, ST, figs / f"dR2_by_epoch_{ar}.png", title, area=ar)
    for w in ("all", "cue"):
        fig5_task_aligned(Wd, figs / f"fig5_task_aligned_{w}.png", title, window=w)
    fig_time_resolved(meta, sess, figs / "time_resolved_dR2.png", title)
    for e in EPOCHS:
        fig_peth_partition(meta, sess, figs / f"peth_partition_{e}.png", title, e)
    fig_exec_vs_intended(Wd, ST, figs / "executed_vs_intended.png", title)
    for ar in ("cortex", "MOs", "SSp-m"):
        fig_target_vs_executed(Wd, figs / f"target_vs_executed_{ar}.png", title, area=ar)
        fig_target_vs_executed_sessions(Wd, figs / f"target_vs_executed_sessions_{ar}.png", title, area=ar)
    sig = ST[(ST.agree == "both") & (ST.pos == "cortex")]
    print(f"cortex changes vs pre significant by both tests: {len(sig)}")
    print(sig[["metric", "epoch", "mean_change", "n_animals"]].to_string(index=False))
    print(f"figures -> {figs}\ntables -> {tabs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
