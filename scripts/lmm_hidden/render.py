"""Render one LMM hidden-slide figure (Priya 2026-11-03) from a per-session CSV, via scripts/lmm_change_fig.

    python -m scripts.lmm_hidden.render <sessions.csv> --name <figure_name> [--outdir DIR] [lmm_change_fig options]

Writes into --outdir (default grant_figures/lmm_hidden_20261103):
  <name>.png, <name>.svg            the LMM version of the figure (stars = analysis_kit.lmm_stars)
  <name>_sessions.csv               the input rows (position, epoch, animal, value [, panel])
  <name>_lmm.csv                    the statistics table (epoch_change_lmm per cell, family, star)
The extractors in this folder write the sessions CSV; this script is the one place that draws.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

OUTDIR = Path("M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/lmm_hidden_20261103")


def render(sessions: Path, name: str, outdir: Path = OUTDIR, *, title: str = "", ylabel: str = "change vs own pre",
           cells=None, labels: dict | None = None, panel_by: str | None = None, holm_by_panel: bool = False,
           ncols: int = 0) -> pd.DataFrame:
    from scripts import lmm_change_fig as L
    from wfield_local import writeguard
    outdir = Path(outdir)
    writeguard.assert_writable(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(sessions)
    S["position"] = S.position.astype(str)
    cells = list(cells) if cells else L._order(list(dict.fromkeys(S.position)))
    S = S[S.position.isin(cells)]
    panels, fam = None, None
    if panel_by:
        pm = S.drop_duplicates("position").set_index("position")[panel_by]
        panels = [(p, [c for c in cells if pm.get(c) == p]) for p in dict.fromkeys(pm[cells])]
        fam = dict(pm) if holm_by_panel else None
    T = L.statistics(S, cells, family=fam)
    dst = outdir / f"{name}_sessions.csv"
    if Path(sessions).resolve() != dst.resolve():
        shutil.copyfile(sessions, dst)
    T.to_csv(outdir / f"{name}_lmm.csv", index=False)
    for ext in (".png", ".svg"):
        L.draw(S, T, cells, outdir / f"{name}{ext}", title=title or name, ylabel=ylabel, labels=labels or {},
               panels=panels, ncols=ncols)
    return T


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
