"""LMM hidden-slide version for deck shape `iti_ramp_row1` (crop lr_iti_ramp_row1.png = row 1 of goal page Fig 14.1
`lick_iti_ramp_wide_docked.png`: dF/F around self-initiated DOCKED-spout ITI bout-start licks, -2..+3 s, 7 key ROIs,
spouts pooled; per-bin BH-FDR bars above the curves).

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

    python -m scripts.lmm_hidden.curve_iti_ramp_row1 [--root <iti_licks>] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import _curve_auc as ca
from scripts.lmm_hidden import lick_cued_ramp_perspout as lcp
from scripts.lmm_hidden import render

NAME = "iti_ramp_row1_lmm"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(lcp.ROOT))
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    a = ap.parse_args(argv)
    root, outdir = Path(a.root), Path(a.outdir)
    # verification on the all-ITI set (published), then the docked set (the figure's)
    A = ca.table(root, key="iti", rois=lcp.ROIS)
    V = lcp.animal_change(A).merge(ca.published(root, "iti"), on=["position", "epoch"], how="outer")
    ok = np.isclose(V.change, V.delta, atol=1e-6)
    print(f"verification all-ITI vs perspout_auc_tests.csv (iti, 0-0.5): {int(ok.sum())}/{len(V)} cells match")
    if not ok.all():
        print(V[~ok].round(4).to_string(index=False))
    D = ca.table(root, key="iti", rois=ca.ROIS7, subset="dock")
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / f"{NAME}_sessions.csv"
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


if __name__ == "__main__":
    raise SystemExit(main())
