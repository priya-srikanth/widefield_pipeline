"""Render one LMM hidden-slide figure (Priya 2026-11-03) from a per-session CSV, via scripts/lmm_change_fig.

    python -m scripts.lmm_hidden.render <sessions.csv> --name <figure_name> [--outdir DIR] [lmm_change_fig options]

Writes into --outdir (default grant_figures/lmm_hidden_20261103):
  <name>.png, <name>.svg            the LMM version of the figure (stars = analysis_kit.lmm_stars)
  <name>_sessions.csv               the input rows (position, epoch, animal, value [, panel])
  <name>_lmm.csv                    the statistics table (epoch_change_lmm per cell, family, star)
  <name>_draw.json                  the drawing arguments, so `redraw` can re-lay-out the figure WITHOUT re-fitting
The extractors in this folder write the sessions CSV; this script is the one place that draws.

SLIDE LAYOUT (deck owner a8, 11-03): the drawing itself is `lmm_change_fig.draw` on main (with its own suptitle band
since 9a820fbb). This module keeps main's layout unless the figure is resized / footnoted or main's band leaves less
than GAP_PX, and then re-lays-out the finished figure before it is saved (`_slide_layout`): figure size, a clear gap of at least
GAP_PX (at the saved 150 dpi) between the two-line suptitle and the first panel titles / stars (the deck crops the title
at that gap), the legend and an optional footnote line at the bottom. `redraw` re-draws any subset of a figure's panels
from its saved _sessions.csv / _lmm.csv (no re-fit; the stars and the Holm family stay the WHOLE figure's) and checks the
drawn estimates and stars against the _lmm.csv.

Capture mode (one-off, used to recover the drawing arguments of figures rendered before _draw.json existed): with
LMM_HIDDEN_CAPTURE=<dir> set, `render` writes <dir>/<name>_draw.json, does NOT fit or draw, and returns the published
<dir>/<name>_lmm.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import _paths  # noqa: E402

#: default output folder; LMM_HIDDEN_OUTDIR overrides (a8 11-17, rebuild re-render)
OUTDIR = _paths.env_path("LMM_HIDDEN_OUTDIR", "M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/lmm_hidden_20261103")
GAP_PX = 30          # >= 25 px asked; measured at the saved dpi
DPI = 150            # lmm_change_fig.draw saves at 150 dpi


def _read_T(path) -> pd.DataFrame:
    T = pd.read_csv(path)
    T["position"] = T.position.astype(str)
    T["star"] = T.star.fillna("").astype(str)
    return T


def _scope_text(T: pd.DataFrame) -> str:
    """The suptitle's Holm phrase, exactly as lmm_change_fig.draw words it, but from the WHOLE figure's table."""
    fam = T.groupby("family").p.apply(lambda q: int(np.isfinite(q).sum()))
    return (f"Holm across {fam.iloc[0]} tests" if len(fam) == 1
            else f"Holm within each panel ({', '.join(str(v) for v in fam.unique())} tests)")


def _suptitle(title: str, T_full: pd.DataFrame) -> str:
    return (f"{title}\nchange vs own pre: mixed model, separate between-animal spread per epoch (df = animals - 1); "
            f"* p < .05, ** {_scope_text(T_full)}")


def _gap(fig, st, axes) -> float:
    r = fig.canvas.get_renderer()
    return st.get_window_extent(r).y0 - max(a.get_tightbbox(r).y1 for a in axes)


def _slide_layout(fig, *, figsize=None, footnote: str | None = None, suptitle: str | None = None) -> None:
    """Re-lay-out a finished lmm_change_fig figure ONLY where needed: if the figure is resized (single-row figures,
    whose size main's draw sets itself) or gets a footnote, lay it out again (legend + footnote at the bottom, title gap
    >= GAP_PX); otherwise keep main's layout (its suptitle band, 9a820fbb) and only push the panels down if its gap is
    still < GAP_PX. Records which path ran in fig._wf_layout."""
    fig.set_dpi(DPI)
    st = fig._suptitle
    if st is not None and suptitle:
        st.set_text(suptitle)
    axes = [a for a in fig.get_axes() if a.get_visible()]
    resize = bool(figsize) and tuple(round(v, 2) for v in fig.get_size_inches()) != tuple(round(v, 2) for v in figsize)
    if not (resize or footnote):
        fig._wf_layout = "main"
        if st is not None and _gap(fig, st, axes) < GAP_PX:
            fig._wf_layout = "main + gap push"
            H = fig.get_figheight() * DPI
            for _ in range(12):
                g = _gap(fig, st, axes)
                if g >= GAP_PX:
                    break
                fig.subplots_adjust(top=fig.subplotpars.top - (GAP_PX + 4 - g) / H)
        fig._wf_title_gap_px = _gap(fig, st, axes) if st is not None else None
        return
    fig._wf_layout = "slide relayout"
    if resize:
        fig.set_size_inches(*figsize)
    for ax in axes:                     # slide legibility: long y labels fit one panel row, crowded x labels shrink
        ax.yaxis.label.set_fontsize(8)
        if len(ax.get_xticks()) > 8:
            ax.tick_params(axis="x", labelsize=6.5)
    if st is not None:
        st.set_y(0.995)
        st.set_va("top")
    leg = fig.legends[0] if fig.legends else None
    if leg is not None and resize:      # main fits the legend columns to the width it drew at; refit to the new width
        n = len(leg.legend_handles)
        leg.set_ncols(max(2, min(n, int(fig.get_figwidth() / 2.4))))
    foot = fig.text(0.005, 0.004, footnote, fontsize=7, ha="left", va="bottom", style="italic") if footnote else None
    r = fig.canvas.get_renderer()
    H = fig.get_figheight() * DPI
    foot_h = (foot.get_window_extent(r).height + 4) if foot else 0
    if leg is not None:
        leg.set_bbox_to_anchor((0.5, foot_h / H), transform=fig.transFigure)
    leg_h = leg.get_window_extent(r).height if leg is not None else 0
    st_h = st.get_window_extent(r).height if st is not None else 0
    bottom = (foot_h + leg_h + 6) / H
    top = 1 - (st_h + GAP_PX + 8) / H
    fig.tight_layout(rect=(0, bottom, 1, top))
    for _ in range(12):           # tight_layout is approximate: measure and push until the gaps are real
        r = fig.canvas.get_renderer()
        boxes = [a.get_tightbbox(r) for a in axes]
        ax_top = max(b.y1 for b in boxes)
        ax_bot = min(b.y0 for b in boxes)
        st_bot = st.get_window_extent(r).y0 if st is not None else H
        leg_top = (leg.get_window_extent(r).y1 if leg is not None else 0) + 4
        gap = st_bot - ax_top
        need_top = gap < GAP_PX or gap > GAP_PX + 12      # a clear gap, but no wasted band either
        need_bot = ax_bot < leg_top
        if not (need_top or need_bot):
            break
        sp = fig.subplotpars
        kw = {}
        if need_top:
            kw["top"] = min(sp.top - (GAP_PX + 4 - gap) / H, 0.999)
        if need_bot:
            kw["bottom"] = sp.bottom + (leg_top - ax_bot + 3) / H
        fig.subplots_adjust(**kw)
    fig._wf_title_gap_px = _gap(fig, st, axes) if st is not None else None


LOWHIT_LABEL = "< 20 hits: shown, not tested"


def _draw_shown(fig, S, shown, cells, panels) -> None:
    """Rows kept OUT of the model (animal x epoch cells below analysis_kit.MIN_HITS hits; Priya 11-17 "show the low-hit
    cells but mark as <20 hits"): each session's change from its animal's pre mean (pre from the fitted rows), drawn as a
    hollow gray diamond (the hollow hit diamond of decoder_retention_sessions) at the session-marker position of
    lmm_change_fig.draw_axis. No estimate, bar or star uses them."""
    from scripts import lmm_change_fig as L
    groups = panels or [("", cells)]
    axes = [a for a in fig.get_axes() if a.get_visible()]
    for ax, (_p, cs) in zip(axes, groups):
        for xi, c in enumerate(cs):
            pre = S[(S.position == c) & (S.epoch == "pre")].groupby("animal").value.mean()
            D = shown[(shown.position == c) & shown.animal.isin(pre.index)]
            for k, e in enumerate(L.EP):
                g = D[D.epoch == e]
                if g.empty:
                    continue
                x = xi + (k - 1) * 0.27 - 0.06
                rng = np.random.default_rng(7)
                ax.scatter(x + rng.uniform(-0.03, 0.03, len(g)), g.value.to_numpy() - pre[g.animal].to_numpy(),
                           marker="D", s=22, facecolors="none", edgecolors="0.55", lw=0.9, zorder=2)


def _draw(S, T_draw, cells, out: Path, *, spec: dict, panels=None, ncols=0, figsize=None, footnote=None,
          T_full=None, shown=None) -> dict:
    """lmm_change_fig.draw, with `_slide_layout` applied just before its savefig; returns the drawn-number check."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.container import ErrorbarContainer
    from matplotlib.figure import Figure

    from scripts import lmm_change_fig as L
    orig = Figure.savefig
    got = {}

    def savefig(fig, fname, *a, **k):
        if shown is not None and len(shown):
            _draw_shown(fig, S, shown, cells, panels)
        _slide_layout(fig, figsize=figsize, footnote=footnote,
                      suptitle=_suptitle(spec.get("title") or out.stem, T_full if T_full is not None else T_draw))
        betas, stars = [], []
        for ax in fig.get_axes():
            for c in ax.containers:
                if isinstance(c, ErrorbarContainer):
                    betas.append(float(np.asarray(c.lines[0].get_ydata())[0]))
            stars += [t.get_text() for t in ax.texts if t.get_text() in ("*", "**")]
        got.update(betas=sorted(betas), stars=sorted(stars), gap=fig._wf_title_gap_px, layout=fig._wf_layout)
        return orig(fig, fname, *a, **k)

    Figure.savefig = savefig
    orig_lh = L.legend_handles
    if shown is not None and len(shown):
        from matplotlib.lines import Line2D
        L.legend_handles = lambda S_: orig_lh(S_) + [Line2D([], [], ls="none", marker="D", mfc="none", mec="0.55",
                                                            ms=5, label=LOWHIT_LABEL)]
    try:
        L.draw(S, T_draw, cells, out, title=spec.get("title") or out.stem, ylabel=spec.get("ylabel", "change vs own pre"),
               labels=spec.get("labels") or {}, panels=panels, ncols=ncols, figsize=figsize)
    finally:
        Figure.savefig = orig
        L.legend_handles = orig_lh
    # the check: every drawn estimate is a beta of the table and every star is the table's star, nothing else
    D = T_draw[T_draw.position.isin(cells) & np.isfinite(T_draw.beta)]
    exp_b = sorted(D.beta.astype(float))
    exp_s = sorted(s for s in D.star if s)
    ok = (len(exp_b) == len(got["betas"]) and np.allclose(exp_b, got["betas"], rtol=0, atol=1e-12)
          and exp_s == got["stars"])
    return dict(file=out.name, estimates=len(got["betas"]), stars=len(got["stars"]),
                double=got["stars"].count("**"), numbers_match=bool(ok), title_gap_px=round(float(got["gap"] or -1), 1),
                layout=got["layout"])


def _panels(S, cells, panel_by):
    if not panel_by:
        return None
    pm = S.drop_duplicates("position").set_index("position")[panel_by]
    return [(p, [c for c in cells if pm.get(c) == p]) for p in dict.fromkeys(pm[cells])]


def render(sessions: Path, name: str, outdir: Path = OUTDIR, *, title: str = "", ylabel: str = "change vs own pre",
           cells=None, labels: dict | None = None, panel_by: str | None = None, holm_by_panel: bool = False,
           ncols: int = 0, figsize=None, shown_only: pd.DataFrame | None = None) -> pd.DataFrame:
    from scripts import lmm_change_fig as L
    from wfield_local import writeguard
    name, title = _paths.out_name(name), _paths.out_title(title or name)   # all-ITI re-render of a docked figure
    S = pd.read_csv(sessions)
    S["position"] = S.position.astype(str)
    cells = list(cells) if cells else L._order(list(dict.fromkeys(S.position)))
    spec = dict(name=name, title=title or name, ylabel=ylabel, cells=cells, labels=labels or {}, panel_by=panel_by,
                holm_by_panel=holm_by_panel, ncols=ncols, figsize=list(figsize) if figsize else None)
    cap = os.environ.get("LMM_HIDDEN_CAPTURE")
    if cap:
        writeguard.assert_writable(cap)
        Path(cap, f"{name}_draw.json").write_text(json.dumps(spec, indent=1))
        return _read_T(Path(cap, f"{name}_lmm.csv"))
    outdir = Path(outdir)
    writeguard.assert_writable(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    S = S[S.position.isin(cells)]
    panels = _panels(S, cells, panel_by)
    fam = dict(S.drop_duplicates("position").set_index("position")[panel_by]) if (panels and holm_by_panel) else None
    T = L.statistics(S, cells, family=fam)
    dst = outdir / f"{name}_sessions.csv"
    if Path(sessions).resolve() != dst.resolve():
        shutil.copyfile(sessions, dst)
        if _paths.docked_as() and "_docked" in Path(sessions).name:
            Path(sessions).unlink()          # the extractor wrote it under the docked name; keep only the renamed copy
    T.to_csv(outdir / f"{name}_lmm.csv", index=False)
    shown = None
    if shown_only is not None and len(shown_only):
        shown = shown_only.assign(position=shown_only.position.astype(str))
        shown.to_csv(outdir / f"{name}_shown_only.csv", index=False)
        spec["shown_only"] = f"{name}_shown_only.csv"
    (outdir / f"{name}_draw.json").write_text(json.dumps(spec, indent=1))
    for ext in (".png", ".svg"):
        _draw(S, T, cells, outdir / f"{name}{ext}", spec=spec, panels=panels, ncols=ncols,
              figsize=tuple(figsize) if figsize else None, shown=shown)
    return T


def redraw(name: str, outdir: Path = OUTDIR, *, out_name: str | None = None, panels: list | None = None,
           ncols: int | None = None, figsize=None, footnote: str | None = None, title_suffix: str = "") -> list[dict]:
    """Re-draw <name> (or the listed PANELS of it, as <out_name>) from its saved _sessions / _lmm / _draw files.
    No re-fit: the stars are the saved ones, i.e. the WHOLE figure's Holm family."""
    from wfield_local import writeguard
    outdir = Path(outdir)
    writeguard.assert_writable(outdir)
    spec = json.loads((outdir / f"{name}_draw.json").read_text())
    if title_suffix:
        spec["title"] = f"{spec.get('title') or name}{title_suffix}"
    S = pd.read_csv(outdir / f"{name}_sessions.csv")
    S["position"] = S.position.astype(str)
    T = _read_T(outdir / f"{name}_lmm.csv")
    cells = [c for c in spec["cells"] if c in set(S.position)]
    allp = _panels(S, cells, spec.get("panel_by"))
    if panels is not None:
        if allp is None:
            raise ValueError(f"{name} has no panels to select")
        missing = set(panels) - {p for p, _ in allp}
        if missing:
            raise ValueError(f"{name}: unknown panels {missing}")
        allp = [pc for p in panels for pc in allp if pc[0] == p]
        cells = [c for _, cs in allp for c in cs]
    Sd = S[S.position.isin(cells)]
    sh = None
    if spec.get("shown_only"):
        sh = pd.read_csv(outdir / spec["shown_only"])
        sh["position"] = sh.position.astype(str)
    nc = spec.get("ncols", 0) if ncols is None else ncols
    out = []
    for ext in (".png", ".svg"):
        out.append(_draw(Sd, T, cells, outdir / f"{out_name or name}{ext}", spec=spec, panels=allp, ncols=nc,
                         figsize=figsize, footnote=footnote, T_full=T, shown=sh))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions")
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--title", default="")
    ap.add_argument("--ylabel", default="change vs own pre")
    ap.add_argument("--cells", default=None)
    ap.add_argument("--labels", default="")
    ap.add_argument("--panel-by", default=None)
    ap.add_argument("--holm-by-panel", action="store_true")
    ap.add_argument("--ncols", type=int, default=0)
    a = ap.parse_args(argv)
    labels = dict(kv.split("=", 1) for kv in a.labels.split(",") if "=" in kv)
    T = render(Path(a.sessions), a.name, Path(a.outdir), title=a.title, ylabel=a.ylabel,
               cells=a.cells.split(",") if a.cells else None, labels=labels, panel_by=a.panel_by,
               holm_by_panel=a.holm_by_panel, ncols=a.ncols)
    print(T.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
