"""LMM VERSION of a change-vs-pre figure (Priya 2026-11-03: hidden deck slides beside the bootstrap versions).

    python -m scripts.lmm_change_fig <sessions.csv> --out <png> [--title T] [--ylabel Y] [--cells a,b,c]
        [--labels "a=close ipsi,b=..."] [--panel-by COL]

Input: one row per SESSION, columns ``position`` (the figure's cell: spout, measure, ...), ``epoch`` (pre / acute /
subacute / chronic), ``animal``, ``value`` -- the `<name>_sessions.csv` sidecar every epoch bar family writes; other
figures write the same four columns. Optional ``panel`` column (or --panel-by) splits the cells into panels sharing one
Holm family.

Drawn per cell and epoch:
  * open markers  = each session's change from ITS ANIMAL's pre mean (shape = animal, colour = epoch);
  * filled dot    = `analysis_kit.epoch_change_lmm` estimate, whisker = its 95 % t interval (df = n_animals - 1);
  * grey bar      = the animal bootstrap 95 % interval (`paired_animal_draws`): a description of THESE animals;
  * stars (top band) = `analysis_kit.lmm_stars`: * p < .05, ** Holm across ALL the figure's cells x epochs.
Writes <png> and <png stem>_lmm.csv (the statistics). Canonical spout order when the cells are spouts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EP = ("acute", "subacute", "chronic")
SPOUTS = [("close_ipsi", "nI", "close_L"), ("close_mid", "nM", "close_center"), ("close_contra", "nC", "close_R"),
          ("far_ipsi", "fI", "far_L"), ("far_mid", "fM", "far_center"), ("far_contra", "fC", "far_R")]
SPOUT_LABEL = {a: lab for names, lab in zip(SPOUTS, ["close\nipsi", "close\nmid", "close\ncontra", "far\nipsi",
                                                      "far\nmid", "far\ncontra"]) for a in names}


def _order(cells):
    rank = {a: i for i, names in enumerate(SPOUTS) for a in names}
    return sorted(cells, key=lambda c: (rank.get(str(c), 99), list(cells).index(c)))


def statistics(S: pd.DataFrame, cells, family=None) -> pd.DataFrame:
    """`epoch_change_lmm` per cell; stars Holm across the whole figure, or within each ``family[cell]`` group (e.g. one
    row of a multi-row figure that was already its own correction family)."""
    from wfield_local import analysis_kit as ak
    out = []
    for c in cells:
        T = ak.epoch_change_lmm(S[S.position == c])
        T.insert(0, "position", c)
        out.append(T)
    T = pd.concat(out, ignore_index=True)
    T["family"] = T.position.map(family) if family else "all"
    T["star"] = ""
    for _f, ix in T.groupby("family").groups.items():
        T.loc[ix, "star"] = ak.lmm_stars(T.loc[ix, "p"].to_numpy())
    return T


def draw_axis(ax, S: pd.DataFrame, T: pd.DataFrame, cs, labels: dict, *, dot_s: float = 16, dot_alpha: float = 0.6,
              label_size: float = 8) -> None:
    """One panel: for each cell in ``cs`` and each post epoch, the session changes (open markers), the animal
    bootstrap bar (grey) and the LMM estimate + CI (filled dot) from ``T``, stars in the top band."""
    from wfield_local import analysis_kit as ak
    from wfield_local import cat_points as cp
    from wfield_local import epoch_figures as ef
    from wfield_local import sig_marks as sm
    for xi, c in enumerate(cs):
        D = S[(S.position == c) & np.isfinite(S.value)]
        pre = D[D.epoch == "pre"].groupby("animal").value.mean()
        pred = {a: list(g.value) for a, g in D[D.epoch == "pre"].groupby("animal")}
        for k, e in enumerate(EP):
            x, col = xi + (k - 1) * 0.27, ef.epoch_dot_color(e)
            po = D[(D.epoch == e) & D.animal.isin(pre.index)]
            if po.empty:
                continue
            rng = np.random.default_rng(0)
            for a, g in po.groupby("animal"):
                ax.scatter(x - 0.06 + rng.uniform(-0.03, 0.03, len(g)), g.value - pre[a], marker=cp.animal_marker(a),
                           facecolors="none", edgecolors=col, s=dot_s, lw=0.8, alpha=dot_alpha, zorder=2)
            got = ak.paired_animal_draws({a: list(g.value) for a, g in po.groupby("animal")}, pred,
                                         lambda v, _r=None: float(np.mean(v)), np.random.default_rng(1), 2000)
            if got[0] is not None and len(got[1]) and len(got[2]) >= 2:   # same condition as the LMM test
                lo, hi = np.percentile(got[1], [2.5, 97.5])
                ax.plot([x + 0.10] * 2, [lo, hi], color="0.6", lw=3, solid_capstyle="butt", zorder=3)
            r = T[(T.position == c) & (T.epoch == e)]
            if len(r) and np.isfinite(r.iloc[0].beta):
                r = r.iloc[0]
                ax.errorbar(x + 0.04, r.beta, yerr=[[r.beta - r.lo], [r.hi - r.beta]], fmt="o", color=col,
                            mec="k", ms=7, capsize=3, elinewidth=1.4, zorder=4)
                if isinstance(r.star, str) and r.star:
                    sm.star_top(ax, x + 0.04, r.star, color=col)
    ax.axhline(0, color="0.5", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(range(len(cs)))
    ax.set_xticklabels([labels.get(str(c), SPOUT_LABEL.get(str(c).split(":")[-1], str(c))) for c in cs],
                       fontsize=label_size)
    ax.set_xlim(-0.6, len(cs) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)


def legend_handles(S: pd.DataFrame) -> list:
    from matplotlib.lines import Line2D

    from wfield_local import cat_points as cp
    from wfield_local import epoch_figures as ef
    h = [Line2D([], [], ls="none", marker="o", color=ef.epoch_dot_color(e), mec="k", ms=7, label=e) for e in EP]
    h += [Line2D([], [], ls="-", lw=1.4, color="k", label="mixed model, 95 % CI"),
          Line2D([], [], ls="-", lw=3, color="0.6", label="animal bootstrap 95 % (these 4 animals)")]
    return h + cp.animal_handles(sorted(S.animal.unique()))


def draw(S: pd.DataFrame, T: pd.DataFrame, cells, out: Path, *, title: str, ylabel: str, labels: dict, panels=None, ncols: int = 0,
         figsize=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import sig_marks as sm
    groups = panels or [("", cells)]
    nc = ncols or len(groups)
    nr = -(-len(groups) // nc)
    if nr == 1:
        widths = [max(1.6, 1.9 * len(cs)) for _, cs in groups]
        fig, axes = plt.subplots(1, nc, figsize=figsize or (max(sum(widths) + 1.2, 10.0), 4.8), squeeze=False,
                                 gridspec_kw={"width_ratios": widths})
    else:                       # a grid of equal panels (one Holm family each, typically)
        w = max(1.6, 1.9 * max(len(cs) for _, cs in groups))
        fig, axes = plt.subplots(nr, nc, figsize=figsize or (w * nc + 1.2, 3.6 * nr + 1.2), squeeze=False)
    flat = list(axes.ravel())
    for ax in flat[len(groups):]:
        ax.set_visible(False)
    for ax, (pname, cs) in zip(flat, groups):
        draw_axis(ax, S, T, cs, labels)
        ax.set_title(pname, fontsize=9)
    for row in axes:
        row[0].set_ylabel(ylabel)
    hs = legend_handles(S)
    narrow = fig.get_figwidth() < 10                                    # a one-graph slide (a8 11-03: --figsize 6,4.5)
    ncol = 3 if narrow else max(2, min(len(hs), int(fig.get_figwidth() / 2.4)))   # fits the width (34, 11-03)
    fig.legend(handles=hs, fontsize=6.5 if narrow else 7, ncol=ncol, loc="lower center", frameon=False,
               columnspacing=1.0 if narrow else 2.0)
    leg_rows = -(-len(hs) // ncol)
    fam = T.groupby("family").p.apply(lambda q: int(np.isfinite(q).sum()))
    scope = (f"Holm across {fam.iloc[0]} tests" if len(fam) == 1
             else f"Holm within each panel ({', '.join(str(v) for v in fam.unique())} tests)")
    how = ("mixed model, per-epoch animal spread, df = animals - 1" if narrow else
           "change vs own pre: mixed model, separate between-animal spread per epoch (df = animals - 1)")
    fig.suptitle(f"{title}\n{how}; * p < .05, ** {scope}", fontsize=8.5 if narrow else 9)     # two lines
    sm.reserve_headroom(fig)
    bottom = min(0.35, (0.22 * leg_rows + 0.1) / fig.get_figheight())   # legend band, in figure fraction
    top = 1 - 0.55 / fig.get_figheight()                                # two-line suptitle band (34, 11-03)
    fig.tight_layout(rect=(0, bottom, 1, top))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import writeguard
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--ylabel", default="change vs own pre")
    ap.add_argument("--cells", default=None, help="comma list (default: all, canonical spout order)")
    ap.add_argument("--labels", default="", help='"cell=label,cell=label"')
    ap.add_argument("--panel-by", default=None, help="column naming each cell's panel")
    ap.add_argument("--holm-by-panel", action="store_true", help="each panel is its own Holm family")
    ap.add_argument("--ncols", type=int, default=0, help="panels per row (default: all in one row)")
    ap.add_argument("--figsize", default=None, help="W,H inches for a multi-row grid (e.g. 13,5.5 for a slide)")
    a = ap.parse_args(argv)
    S = pd.read_csv(a.sessions)
    S["position"] = S.position.astype(str)
    cells = a.cells.split(",") if a.cells else _order(list(dict.fromkeys(S.position)))
    S = S[S.position.isin(cells)]
    labels = dict(kv.split("=", 1) for kv in a.labels.split(",") if "=" in kv)
    panels = None
    if a.panel_by:
        pm = S.drop_duplicates("position").set_index("position")[a.panel_by]
        panels = [(p, [c for c in cells if pm.get(c) == p]) for p in dict.fromkeys(pm[cells])]
    out = Path(a.out)
    writeguard.assert_writable(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    T = statistics(S, cells, family=dict(pm) if (panels and a.holm_by_panel) else None)
    T.to_csv(out.with_name(out.stem + "_lmm.csv"), index=False)
    draw(S, T, cells, out, title=a.title or Path(a.sessions).stem.replace("_sessions", ""), ylabel=a.ylabel,
         labels=labels, panels=panels, ncols=a.ncols,
         figsize=tuple(float(v) for v in a.figsize.split(",")) if a.figsize else None)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
