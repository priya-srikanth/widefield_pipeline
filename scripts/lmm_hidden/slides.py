"""Slide versions of the LMM hidden-slide figures (deck owner a8, 11-03). NO RE-FIT: everything is re-drawn from the
saved <name>_sessions.csv / <name>_lmm.csv / <name>_draw.json in the output folder (`render.redraw`).

    python -m scripts.lmm_hidden.slides [--outdir DIR] [--only NAME ...]

1. Every figure is re-drawn IN PLACE (same file names, identical statistics) with the slide title gap
   (render.GAP_PX between the two-line suptitle and the first panel titles; the deck crops the title there).
2. Four figures also get panel subsets at ~13 x 5.5 in for 16:9 slides (SPLITS below). The stars are the saved ones,
   i.e. the Holm family of the WHOLE figure, as published -- never re-corrected per image; a footnote line says so.
Every image is checked: the drawn LMM estimates and stars must equal the _lmm.csv rows of the cells it shows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.lmm_hidden import render

FIGS = ["rec_wlc_primary", "rec_kn_epoch", "lick_cued_ramp_perspout", "fig9i3_cls_pooled", "key_figure_BC",
        "lick_matched_gauss_readout_docked", "lick_cued_first_lick", "lick_sametarget_place_docked",
        "lick_order_iti_cued_slope_docked", "lick_lateral_ratio_auc_docked", "decoder_retention_pooled",
        "recovery_timing_pooled_epoch_lick", "spec_groups", "lick_matched_gauss_perspout_docked",
        "cued_ramp_summary_row2_lmm", "iti_ramp_row1_lmm",
        # slide 94 companions (11-03): spouts pooled, AUC 0-1 s (primary) and peak (secondary); one panel, no split
        "cued_ramp_summary_row2_lmm_w0-1", "iti_ramp_row1_lmm_w0-1", "iti_ramp_row1_lmm_w0-0.5", "cued_ramp_summary_row2_lmm_peak",
        "iti_ramp_row1_lmm_peak"]

SLIDE = (13.0, 5.5)
_AREAS = ["ALM", "tjM1", "SSp-m", "SSp-n", "orofacial"]
_SPEC_A = ["RESIDUAL by ROI, cue window (pre-SD units)", "RESIDUAL by ROI, lick window (pre-SD units)"]
_SPEC_B = ["RESIDUAL, cue window (pre-SD units)", "RESIDUAL, lick window (pre-SD units)", "LAYER 2 (R2, vs pre_late)",
           "LAYER 2 by ROI (R2, vs pre_late)"]
# name -> [(suffix, panels, ncols, title suffix)]
SPLITS = {
    **{c: [("_a", ["ALM_right", "tjM1_right", "SSp-m_right", "cortex"], 2, " [a: right-hemisphere ROIs + cortex]"),
           ("_b", ["ALM_left", "tjM1_left", "SSp-m_left"], 3, " [b: left-hemisphere ROIs]")]
       for c in ("cued_ramp_summary_row2_lmm", "iti_ramp_row1_lmm")},
    "lick_lateral_ratio_auc_docked": [(f"_{s}", [f"{a}, {s}" for a in _AREAS], 3, f" [{s} licks]")
                                      for s in ("cued", "cue", "iti")],
    "spec_groups": [("_a", _SPEC_A, 2, " [a: residual by ROI, cue and lick windows]"),
                    ("_b", _SPEC_B, 2, " [b: area groups incl. orofacial - control, and Layer 2]")],
}


def footnote(name: str, outdir: Path) -> str:
    T = render._read_T(outdir / f"{name}_lmm.csv")
    fam = T.groupby("family").p.apply(lambda q: int(q.notna().sum()))
    if len(fam) == 1:
        how = f"the Holm family of the WHOLE figure ({fam.iloc[0]} tests)"
    else:
        how = f"Holm within each panel ({', '.join(str(v) for v in fam.unique())} tests), as in the whole figure"
    return (f"Panels split across images: ** uses {how} -- {name}, as published; not re-corrected per image. "
            "Same statistics as the full figure (no re-fit).")


def run(outdir: Path, only=None) -> pd.DataFrame:
    rows = []
    for name in FIGS:
        if only and name not in only:
            continue
        for r in render.redraw(name, outdir):
            rows.append(dict(figure=name, **r))
        for suf, panels, nc, tsuf in SPLITS.get(name, []):
            base = name[:-4] if name.endswith("_lmm") else name
            out = f"{base}_lmm{suf}" if name.endswith("_lmm") else f"{name}{suf}"
            for r in render.redraw(name, outdir, out_name=out, panels=panels, ncols=nc, figsize=SLIDE,
                                   footnote=footnote(name, outdir), title_suffix=tsuf):
                rows.append(dict(figure=name, **r))
        print(name, "done", flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    R = run(Path(a.outdir), a.only)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))
    bad = R[~R.numbers_match | (R.title_gap_px < 25)]
    print(f"{len(R)} images; numbers match the _lmm.csv in {int(R.numbers_match.sum())}; "
          f"title gap >= 25 px in {int((R.title_gap_px >= 25).sum())}")
    return 1 if len(bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
