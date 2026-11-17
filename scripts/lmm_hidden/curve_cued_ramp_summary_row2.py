"""LMM hidden-slide version for deck shape `cued_ramp_summary_row2` (crop lr_cued_summary_row2.png = row 2 of goal page
Fig 14.23 `lick_cued_ramp_summary.png`: FIRST cued lick, per-session lick response, 7 key ROIs).

DEFAULT RUN (11-03, the hidden companion of PI-deck slide 94, cued row): the main slide's OWN quantity -- per session,
spouts pooled, response AUC 0..1.0 s after the first cued lick (session PETH baseline -1.0..-0.5 s) / the animal's pre
mean, 7 ROIs (`_pooled_resp`, = wf_attempt lick_cued_ramp.compute; RESP_W = (0, 1.0) is the default since Priya 10-28
evening). It must reproduce cued_ramp_auc_sessions.csv row for row and ramp_window_tests.csv (cued, auc_0_1_n) cell for
cell before any model is fitted. SECONDARY (Priya 11-03): peak 0..1.5 s / pre (ramp_window_tests peak_n), same check.
Outputs cued_ramp_summary_row2_lmm_w0-1* (primary) and cued_ramp_summary_row2_lmm_peak* (secondary); one panel,
** = Holm across the 7 ROIs x 3 epochs = 21 tests.

`--perspout` = the earlier per-spout SUPPLEMENT (files without a suffix), described below:

Rule followed (fixed before any model was run): DECISIONS.md 2026-11-03 "Curve figures: mixed model on PRE-SPECIFIED
per-session summaries", row "lick-ramp time courses (wf_attempt: cued_ramp_summary_row2 / iti_ramp_row1, ...)":
summary per session x spout = the response AUC 0-0.5 s already used on lick_cued_ramp_perspout; sign / unit as that
figure. Stars: epoch_change_lmm + lmm_stars, ** = Holm across the spout x epoch cells of EACH summary panel (one panel
per ROI -> holm_by_panel=True).

NOTE: the published row is NOT a curve figure: it plots per-session AUC 0..1.0 s / the animal's pre, spouts POOLED
(nested-bootstrap stars). Per the DECISIONS row, this version applies the pre-specified per-spout summary to the SAME
traces (first cued lick, <iti_licks>/cued_wide/<label>.npz key `lick`) and the SAME 7 ROIs; it is not the row's own
scalar.

Value (scripts/lmm_hidden/_curve_auc.table, == lick_cued_ramp_perspout.table generalised to 7 ROIs; checked identical
on its 4 ROIs): per trial minus the mean over -1.0..-0.5 s; session mean at that target spout (>= 5 licks); AUC
0..0.5 s; / the animal's mean pre AUC at that spout and ROI (pre = 1).
Source: wf_attempt scripts/lick_cued_ramp.py (`_sessions`, `_pe`, `_spout_auc_tests`, run of 10-28).

    python -m scripts.lmm_hidden.curve_cued_ramp_summary_row2 [--root <iti_licks>] [--outdir DIR] [--perspout]
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

NAME = "cued_ramp_summary_row2_lmm"


def perspout(root: Path, outdir: Path) -> int:
    """The per-spout AUC 0-0.5 s SUPPLEMENT (files without a suffix, 11-03)."""
    D = ca.table(root, key="lick", rois=ca.ROIS7)
    ca.check_cued(root, D)
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / f"{_paths.out_name(NAME)}_sessions.csv"
    D.to_csv(f, index=False)
    cells = [f"{r}:{p}" for r in ca.ROIS7 for p in lcp.POS]
    T = render.render(f, NAME, outdir, title="First cued lick, response AUC 0..0.5 s per spout / the animal's pre AUC at "
                      "that spout (pre-specified curve summary), change vs own pre (LMM; ** Holm within each ROI panel)",
                      ylabel="AUC 0..0.5 s / pre at spout,\nchange vs pre", cells=cells, panel_by="panel",
                      holm_by_panel=True, ncols=4)
    ca.summary(D, T)
    V = lcp.animal_change(D).merge(ca.published(root, "lick"), on=["position", "epoch"], how="left")
    V = V.merge(T[["position", "epoch", "beta", "p", "star"]], on=["position", "epoch"], how="left")
    m = V.delta.notna()
    print(f"verification (4 ROIs in perspout_auc_tests.csv, set lick, 0-0.5): "
          f"{int(np.isclose(V.change[m], V.delta[m], atol=1e-6).sum())}/{int(m.sum())} cells match the boot_delta point")
    V.to_csv(outdir / f"{NAME}_verify.csv", index=False)
    pd.set_option("display.width", 220)
    print(V.round(4).to_string(index=False))
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
    pr.run(root, outdir, key="lick", subset="", which="lick", name=NAME,
           what="First cued lick (slide 94 cued row, Fig 14.23 row 2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
