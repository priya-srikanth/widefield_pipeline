"""PI deck slide 156 (Priya 11-18: "can the y axis be a bit lower, esp for limb (doesn't have to match orofacial)?"):
swap fig:q_fr for the native 2x2 from `quiet_spikes_v2_fr.py --plot-only` (rows orofacial / limb with their own y scale,
top = 1.1 x the row's 99th-percentile session; the one session above it, PS92_0828, drawn at the top edge with its
value). Same run folder and tables as the 11-18 c1guard swap (no recompute). Exact-match note edits (fail loudly).
usage: python s156_ylim_swap.py <deck_in> <deck_out>"""
import shutil
import sys
from pathlib import Path

from pptx import Presentation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apply_specs import find_shape, set_para_text, swap_picture  # noqa: E402

T = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
NEW = f"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/{T}/cohort1_quiet_spikes/v2_quiet_lp_c1guard_k375_20261117/"
SWAP = "M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs/"
SRC = NEW + "fr_regressed_own/quiet_events_face_regressed_2x2.png"
IMG = SWAP + "s156_quiet_fr_2x2_c1guard_ylim.png"
NOTES = [
    ("Stars in the figure = the mixed model (FDR), as drawn by the script.",
     "Stars in the figure = the mixed model (FDR), as drawn by the script. Orofacial and limb rows have their own y scale; "
     "the one session above it, PS92_0828 (chronic, day 11), is drawn at the top edge with its value (orofacial 80 / 92, "
     "limb 60 / 67 per min). 6.0 of its 8.4 quiet minutes come after the last cue (task ended at 120.2 min, imaging ran "
     "to 126.8 min; no quit detected, so the tail counts as engaged); task-period quiet alone is 2.4 min (56 / 69 per min)."),
    ("fr_regressed_own/quiet_events_face_regressed.png (scripts/cohort2_qc/quiet_spikes_v2_fr.py --jobs 3,",
     "fr_regressed_own/quiet_events_face_regressed_2x2.png (scripts/cohort2_qc/quiet_spikes_v2_fr.py --jobs 3, then "
     "--plot-only for the 2x2,"),
    ("deck image M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs/"
     "s156_quiet_fr_2x2_c1guard.png, cut by docs/status/deck_swap_20261117/quiet_c1guard/c1guard_crops.py, same geometry "
     "as the previous image)",
     f"deck image {IMG}, native 2x2 render, swap docs/status/deck_swap_20261117/quiet_c1guard/s156_ylim_swap.py)"),
]


def main(deck_in, deck_out):
    if not Path(IMG).exists():
        shutil.copyfile(SRC, IMG)
    P = Presentation(deck_in)
    s = P.slides[155]
    t = next(x.text_frame.text for x in s.shapes if x.has_text_frame and x.text_frame.text.strip())
    assert t.startswith("Quiet orofacial events"), t
    paras = s.notes_slide.notes_text_frame.paragraphs
    for old, new in NOTES:
        hit = [p for p in paras if old in p.text]
        if len(hit) != 1:
            raise SystemExit(f"notes text not found once ({len(hit)}): {old[:60]!r}")
        set_para_text(hit[0], hit[0].text.replace(old, new))
    swap_picture(s, find_shape(s, "fig:q_fr"), IMG)
    P.save(deck_out)
    print("slide 156: picture swapped,", len(NOTES), "note edits ->", deck_out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
