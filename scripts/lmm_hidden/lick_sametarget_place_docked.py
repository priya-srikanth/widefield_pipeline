"""LMM version of goal page Fig 12.15 DOCKED (lick_sametarget_place_docked.png; deck o_sametarget_2x2).

Source: scripts/lick_same_target_place.py, run under WFIELD_ITI_FILTER=dock (docked-spout ITI licks; cued rows
unchanged), outputs in <iti_licks>/itidock_main/; the page PNG is that figure with the dock banner
(scripts/iti_dock_page_figures.py). Per-session table: itidock_main/sametarget_sessions.csv (scope all spouts, cls all).
Values (session means over licks, NaN when < 8 licks carry the value; mm, + = contralesional):
  any_all   read-out minus pre licks of ANY spout at the same place (all licks)
  any_cov   the same, only licks that have a same-target same-place pre reference ("covered")
  tt        read-out minus pre licks of the SAME TARGET at the same place (same licks as any_cov)
  tt_minus  tt - any_cov per lick (what matching the target removes)
Cells (8): set {ITI, cued} x the four values; panel = set. The fifth column of the published figure (coverage %, a
descriptive line plot with no test) is not a change-vs-pre cell and is left out.
Published test: paired nested bootstrap (analysis_kit.boot_delta), * = 95 % CI excludes 0, uncorrected
(sametarget_tests.csv, scope all spouts, cls all). Holm family here = all 24 cell x epoch tests.
    python -m scripts.lmm_hidden.lick_sametarget_place_docked [--root <itidock_main>] [--outdir DIR]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.lmm_hidden import _lick_matched_common as cm
from scripts.lmm_hidden import _paths, render

NAME = "lick_sametarget_place_docked"
SETS = {"iti": "ITI", "trial": "cued"}
VALS = {"any_all": "1. any spout\nsame place\n(all licks)", "any_cov": "2. any spout\nsame place\n(covered)",
        "tt": "3. SAME TARGET\nsame place\n(covered)", "tt_minus": "4. = 3 - 2\n(target\ncomposition)"}


def build(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    S = pd.read_csv(root / "sametarget_sessions.csv")
    S = S[(S.scope == "all spouts") & (S.cls == "all")]
    T = pd.read_csv(root / "sametarget_tests.csv")
    T = T[(T.scope == "all spouts") & (T.cls == "all")]
    rows, tests, cells, labels = [], [], [], {}
    for st in SETS:
        g = S[S.set == st]
        for v, lab in VALS.items():
            c = f"{SETS[st]}|{v}"
            cells.append(c)
            labels[c] = lab
            rows.append(pd.DataFrame(dict(position=c, panel=SETS[st], epoch=g.epoch, animal=g.animal, label=g.label,
                                          n_licks=g.n, value=g[v])))
            tests.append(T[(T.set == st) & (T.measure == v)].assign(position=c))
    R = pd.concat(rows, ignore_index=True).dropna(subset=["value"])
    return R, pd.concat(tests, ignore_index=True), cells, labels


def main() -> int:
    a = cm.args(__doc__)
    R, Tp, cells, labels = build(Path(a.root))
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = outdir / f"{_paths.out_name(NAME)}_sessions.csv"
    R.to_csv(src, index=False)
    T = render.render(src, NAME, outdir, title="Fig 12.15 DOCKED: same-target same-place check of the like-for-like offset",
                      ylabel="read-out vs pre licks at the same place\n(mm, + = contralesional)", cells=cells,
                      labels=labels, panel_by="panel", ncols=2)
    cm.report(cm.verify(R, Tp), T, R)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
