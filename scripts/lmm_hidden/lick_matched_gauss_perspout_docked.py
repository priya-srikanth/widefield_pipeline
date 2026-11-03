"""LMM version of goal page Fig 13.6 DOCKED (lick_matched_gauss_perspout_docked.png; deck o_perspout_all).

Source: scripts/lick_matched_readout.py --ref gauss (figure 3, per spout), run under WFIELD_ITI_FILTER=dock (docked-spout
ITI licks; cued rows unchanged), outputs in <iti_licks>/itidock_main/; the page PNG is that figure with the dock banner
(scripts/iti_dock_page_figures.py). Per-session table: itidock_main/matched_gauss_sessions.csv.
Value: session mean of `resid` (decoded lateral minus the animal's PRE licks of any spout Gaussian-weighted on landing
distance; mm, + = contralesional) per spout (cued = target spout, ITI = previous trial's spout), sessions with >= 8 licks.
Cells (24): rows {ITI all, ITI FIRST lick (bout start), cued all, cued FIRST lick} x 6 spouts (canonical order);
panel = row. Published test: paired nested bootstrap (analysis_kit.boot_delta), * = 95 % CI excludes 0, uncorrected
(matched_gauss_tests.csv, measure resid). Holm family here = all 72 cell x epoch tests.
    python -m scripts.lmm_hidden.lick_matched_gauss_perspout_docked [--root <itidock_main>] [--outdir DIR]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.lmm_hidden import _lick_matched_common as cm
from scripts.lmm_hidden import render

NAME = "lick_matched_gauss_perspout_docked"
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
ROWS = [("iti", "all", "ITI"), ("iti", "first", "ITI FIRST"), ("trial", "all", "cued"),
        ("trial", "first", "cued FIRST")]
PANEL = {"ITI": "ITI, all licks", "ITI FIRST": "ITI, FIRST lick (bout start)", "cued": "cued, all licks",
         "cued FIRST": "cued, FIRST lick"}


def build(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    S = pd.read_csv(root / "matched_gauss_sessions.csv")
    T = pd.read_csv(root / "matched_gauss_tests.csv")
    T = T[T.measure == "resid"]
    rows, tests, cells = [], [], []
    for st, cls, rl in ROWS:
        for p in POS:
            c = f"{rl}:{p}"
            cells.append(c)
            g = S[(S.set == st) & (S.scope == p) & (S.cls == cls)]
            rows.append(pd.DataFrame(dict(position=c, panel=PANEL[rl], epoch=g.epoch, animal=g.animal, label=g.label,
                                          n_licks=g.n, value=g.resid)))
            tests.append(T[(T.set == st) & (T.scope == p) & (T.cls == cls)].assign(position=c))
    R = pd.concat(rows, ignore_index=True).dropna(subset=["value"])
    return R, pd.concat(tests, ignore_index=True), cells


def main() -> int:
    a = cm.args(__doc__)
    R, Tp, cells = build(Path(a.root))
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = outdir / f"{NAME}_sessions.csv"
    R.to_csv(src, index=False)
    T = render.render(src, NAME, outdir, title="Fig 13.6 DOCKED: like-for-like (Gaussian same-place) read-out per spout",
                      ylabel="read-out vs pre licks at the same place\n(mm, + = contralesional)", cells=cells,
                      panel_by="panel", ncols=2)
    cm.report(cm.verify(R, Tp), T, R)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
