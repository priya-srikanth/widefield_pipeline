"""ITI 0..0.5 s LMM re-render (11-17): slide splits of lick_lateral_ratio_auc_docked_iti05 (as slides.SPLITS; no re-fit,
stars = the whole figure's Holm family), notitle crops, and the slide-111 stack (cued session-z unchanged from the 11-17
render + the new ITI session-z). Run from wf_lmm_main with LMM_HIDDEN_OUTDIR = the iti05 folder."""
import os
from pathlib import Path

import pandas as pd

from scripts.lmm_hidden import render, slides
from scripts.lmm_hidden.crop import drop_title, stack

out = Path(os.environ["LMM_HIDDEN_OUTDIR"])
parent = out.parent
name = "lick_lateral_ratio_auc_docked_iti05"
rows = []
for suf, panels, nc, tsuf in slides.SPLITS["lick_lateral_ratio_auc_docked"]:
    dst = out / f"{name}{suf}.png"
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    for r in render.redraw(name, out, out_name=f"{name}{suf}", panels=panels, ncols=nc, figsize=slides.SLIDE,
                           footnote=slides.footnote(name, out), title_suffix=tsuf):
        rows.append(dict(figure=name, **r))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.to_string(index=False))
print("all numbers match:", bool(R.numbers_match.all()), "min title gap px:", R.title_gap_px.min())
for n in ("iti_ramp_row1_lmm_w0-0.5_iti05", "iti_ramp_row1_lmm_w0-0.5_sessz_iti05",
          *[f"{name}_{s}" for s in ("cued", "cue", "iti")]):
    drop_title(out / f"{n}.png", out / f"{n}_notitle.png")
    print("notitle", n)
stack([parent / "cued_ramp_summary_row2_lmm_w0-1_sessz_notitle.png", out / "iti_ramp_row1_lmm_w0-0.5_sessz_iti05_notitle.png"],
      out / "lick_lmm_sessz_stack_iti05.png")
print("stack written")
