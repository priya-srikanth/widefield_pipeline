"""LMM version of goal page Fig 12.1 DOCKED (lick_matched_gauss_readout_docked.png; deck crop o_gauss_readout_allspouts).

Source: scripts/lick_matched_readout.py --ref gauss, run under WFIELD_ITI_FILTER=dock (docked-spout ITI licks; cued
rows unchanged), outputs in <iti_licks>/itidock_main/; the page PNG is that figure with the dock banner
(scripts/iti_dock_page_figures.py). Per-session table: itidock_main/matched_gauss_sessions.csv.
Value: session mean of `resid` = decoded lateral position minus the animal's PRE licks (any spout) Gaussian-weighted on
landing distance (same place; pre sessions leave-one-session-out), mm, + = contralesional; sessions with >= 8 licks in
the cell. Pre rows are levels (near 0 by construction), so the change vs own pre is formed by the LMM as published.
Cells (12): set {ITI, cued} x {all spouts, far contra} x {all, first, later licks}; panel = set.
Published test: paired nested animals->sessions bootstrap (analysis_kit.boot_delta), * = 95 % CI excludes 0, uncorrected,
per cell x epoch (matched_gauss_tests.csv, measure resid). Holm family here = all 36 cell x epoch tests.
    python -m scripts.lmm_hidden.lick_matched_gauss_readout_docked [--root <itidock_main>] [--outdir DIR]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.lmm_hidden import _lick_matched_common as cm
from scripts.lmm_hidden import _paths, render

NAME = "lick_matched_gauss_readout_docked"
SETS = {"iti": "ITI", "trial": "cued"}
PANELS = [("all spouts", "all", "all spouts\nall licks"), ("all spouts", "first", "all spouts\nFIRST lick"),
          ("all spouts", "later", "all spouts\nlater licks"), ("far_contra", "all", "far contra\nall licks"),
          ("far_contra", "first", "far contra\nFIRST lick"), ("far_contra", "later", "far contra\nlater licks")]


def cid(st: str, scope: str, cls: str) -> str:
    return f"{SETS[st]}|{scope}|{cls}"


def build(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    S = pd.read_csv(root / "matched_gauss_sessions.csv")
    T = pd.read_csv(root / "matched_gauss_tests.csv")
    T = T[T.measure == "resid"]
    rows, tests, cells, labels = [], [], [], {}
    for st in SETS:
        for scope, cls, lab in PANELS:
            c = cid(st, scope, cls)
            cells.append(c)
            labels[c] = lab
            g = S[(S.set == st) & (S.scope == scope) & (S.cls == cls)]
            rows.append(pd.DataFrame(dict(position=c, panel=SETS[st], epoch=g.epoch, animal=g.animal, label=g.label,
                                          n_licks=g.n, value=g.resid)))
            t = T[(T.set == st) & (T.scope == scope) & (T.cls == cls)]
            tests.append(t.assign(position=c))
    R = pd.concat(rows, ignore_index=True).dropna(subset=["value"])
    return R, pd.concat(tests, ignore_index=True), cells, labels


def main() -> int:
    a = cm.args(__doc__)
    R, Tp, cells, labels = build(Path(a.root))
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = outdir / f"{_paths.out_name(NAME)}_sessions.csv"
    R.to_csv(src, index=False)
    T = render.render(src, NAME, outdir, title="Fig 12.1 DOCKED: like-for-like (Gaussian same-place) lick read-out",
                      ylabel="read-out vs pre licks at the same place\n(mm, + = contralesional)", cells=cells,
                      labels=labels, panel_by="panel", ncols=1)
    cm.report(cm.verify(R, Tp), T, R)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
