"""LMM hidden-slide version for deck shape `iti_ramp_row1` (crop lr_iti_ramp_row1.png = row 1 of goal page Fig 14.1
`lick_iti_ramp_wide_docked.png`: dF/F around self-initiated DOCKED-spout ITI bout-start licks, -2..+3 s, 7 key ROIs,
spouts pooled; per-bin BH-FDR bars above the curves).

DEFAULT RUN (11-03, the hidden companion of PI-deck slide 94, ITI row): the main row is a CURVE figure whose own test is
per time bin (BH-FDR); its pre-specified per-session summary is the response AUC, so the slide's 0..1.0 s window
applies, spouts pooled. Value (`_pooled_resp`, = wf_attempt lick_cued_ramp.compute ITI summary on the wide traces): the
docked ITI bout starts (gap >= 2 s, dock flag), session PETH (>= 10 bout starts) baseline -1.0..-0.5 s, AUC 0..1.0 s /
the animal's pre mean. It must reproduce the docked run's session table itidock_main/iti_ramp_auc_sessions_wide.csv
(resp_auc_n) row for row and itidock_main/ramp_window_tests.csv (iti, auc_0_1_n) cell for cell before any model is
fitted. SECONDARY (Priya 11-03): peak 0..1.5 s / pre (peak_n), same check. Outputs iti_ramp_row1_lmm_w0-1* (primary)
and iti_ramp_row1_lmm_peak* (secondary); one panel, ** = Holm across 7 ROIs x 3 epochs = 21 tests.

`--perspout` = the earlier per-spout SUPPLEMENT (files without a suffix), described below:

Rule followed (fixed before any model was run): DECISIONS.md 2026-11-03 "Curve figures: mixed model on PRE-SPECIFIED
per-session summaries", row "lick-ramp time courses (wf_attempt: cued_ramp_summary_row2 / iti_ramp_row1, ...)":
summary per session x spout = the response AUC 0-0.5 s already used on lick_cued_ramp_perspout; sign / unit as that
figure. Stars: epoch_change_lmm + lmm_stars, ** = Holm across the spout x epoch cells of EACH summary panel (one panel
per ROI -> holm_by_panel=True).

Same traces as the curves: <iti_licks>/cued_wide/<label>.npz key `iti` (bout starts), kept as lick_cued_ramp._sessions
under WFIELD_ITI_FILTER=dock: gap >= 2.0 s AND flagged `dock` in <iti_licks>/iti_dock_audit.csv (iti_licks.
bout_subset_keep, 3 ms match). Spout = the PREVIOUS trial's spout (<iti_licks>/bout_prev_pos.csv), as the ITI per-spout
figure (lick_iti_ramp_perspout). Value (scripts/lmm_hidden/_curve_auc.table): per trial minus the mean over
-1.0..-0.5 s (the perspout baseline, BASE_SPOUT, used for cued AND ITI per-spout AUCs; the row-1 curves themselves use
-1.5..-1.0 s -- the pre-specified summary is kept as is); session mean at that spout (>= 5 bout starts); AUC 0..0.5 s;
/ the animal's mean pre AUC at that spout and ROI (pre = 1).
Verification: the same code WITHOUT the dock filter reproduces the published all-ITI per-spout AUC test
(perspout_auc_tests.csv, set iti, window 0-0.5, 4 ROIs); no docked per-spout AUC table was published.

    python -m scripts.lmm_hidden.curve_iti_ramp_row1 [--root <iti_licks>] [--outdir DIR] [--perspout]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import _curve_auc as ca
from scripts.lmm_hidden import _pooled_resp as pr
from scripts.lmm_hidden import lick_cued_ramp_perspout as lcp
from scripts.lmm_hidden import _paths, render

NAME = "iti_ramp_row1_lmm"


def perspout(root: Path, outdir: Path) -> int:
    """The per-spout AUC 0-0.5 s SUPPLEMENT (files without a suffix, 11-03)."""
    # verification on the all-ITI set (published), then the docked set (the figure's)
    A = ca.table(root, key="iti", rois=lcp.ROIS)
    V = lcp.animal_change(A).merge(ca.published(root, "iti"), on=["position", "epoch"], how="outer")
    ok = np.isclose(V.change, V.delta, atol=1e-6)
    print(f"verification all-ITI vs perspout_auc_tests.csv (iti, 0-0.5): {int(ok.sum())}/{len(V)} cells match")
    if not ok.all():
        print(V[~ok].round(4).to_string(index=False))
    D = ca.table(root, key="iti", rois=ca.ROIS7, subset="dock")
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / f"{_paths.out_name(NAME)}_sessions.csv"
    D.to_csv(f, index=False)
    cells = [f"{r}:{p}" for r in ca.ROIS7 for p in lcp.POS]
    T = render.render(f, NAME, outdir, title="Docked-spout ITI bout-start licks, response AUC 0..0.5 s per previous-trial "
                      "spout / the animal's pre AUC at that spout (pre-specified curve summary), change vs own pre "
                      "(LMM; ** Holm within each ROI panel)",
                      ylabel="AUC 0..0.5 s / pre at spout,\nchange vs pre", cells=cells, panel_by="panel",
                      holm_by_panel=True, ncols=4)
    ca.summary(D, T)
    C = lcp.animal_change(D).merge(T[["position", "epoch", "beta", "p", "star"]], on=["position", "epoch"], how="left")
    C.to_csv(outdir / f"{NAME}_verify.csv", index=False)
    pd.set_option("display.width", 220)
    print(C.round(4).to_string(index=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(_paths.iti_root(lcp.ROOT)))
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    ap.add_argument("--perspout", action="store_true", help="the per-spout AUC 0-0.5 s supplement instead")
    a = ap.parse_args(argv)
    root, outdir = Path(a.root), Path(a.outdir)
    if a.perspout:
        return perspout(root, outdir)
    pr.run(root, outdir, key="iti", subset="dock", which="iti_dock", name=NAME,
           what="Docked-spout ITI bout-start licks (slide 94 ITI row, Fig 14.1 docked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
