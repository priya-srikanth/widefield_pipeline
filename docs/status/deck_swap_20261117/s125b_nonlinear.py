"""Add a HIDDEN supporting slide right after slide 125: the non-linear-decoder version of the combined like-for-like
figure (Priya 11-17 "update page, decks, repo with these"). Duplicates slide 125's shapes (title, rule, bullets, picture,
caption), swaps the picture, rewrites the texts and notes. usage: python s125b_nonlinear.py <deck_in> <deck_out> <image>"""
import copy
import sys

from PIL import Image
from pptx import Presentation

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from apply_specs import set_para_text, swap_picture, write_notes  # noqa: E402

src, dst, img = sys.argv[1:4]
P = Presentation(src)
base = P.slides[124]
assert base.shapes[0].text_frame.text.startswith("Decoded vs landed"), base.shapes[0].text_frame.text
new = P.slides.add_slide(base.slide_layout)
for sh in list(new.shapes):
    sh._element.getparent().remove(sh._element)
for sh in base.shapes:
    new.shapes._spTree.append(copy.deepcopy(sh._element))
# the copied picture still points at slide 125's image rId: give it this slide's own image
pic = [sh for sh in new.shapes if sh.shape_type == 13][0]
swap_picture(new, pic, img)
Image.open(img).verify()
T = {sh.name: sh for sh in new.shapes if sh.has_text_frame}
set_para_text(T["TextBox 1"].text_frame.paragraphs[0], "Same with a non-linear decoder")
b = T["TextBox 3"].text_frame.paragraphs
set_para_text(b[0], "•  Frozen non-linear (RBF kernel ridge) decoder, same licks, curves and tests as the previous slide")
set_para_text(b[1], "•  Same result: ITI every epoch, every lick set; cued acute + first lick; still pulled to the middle")
set_para_text(T["TextBox 5"].text_frame.paragraphs[0],
              "Mean decoded lateral position (non-linear decoder) for licks landing at each place (Gaussian σ 0.1 mm), "
              "mean ± SEM of 4 animals, dashed = 3; bottom row = only licks where the animal's own pre curve exists; "
              "bars = bins that differ from pre (BH)")
block = {
    "Window": "lick peak, decoder lags 0..0.2 s",
    "Licks": "LP video licks (cam4) with 3-D landing (cam1 + cam4)",
    "Trial period": "cued: inside a trial's response window; ITI: between trials (all; bout start; spout docked, "
                    "WFIELD_ITI_FILTER=dock)",
    "Trial types": "cued all / cued first lick / ITI all / ITI first lick of a bout / docked-spout ITI licks, all spouts",
    "Gate": "OROFACIAL_GATE=attempted, WIDEFIELD_TRIAL_GATE=camera",
    "Baseline": "August pre licks of the same set landing at the same place (Gaussian-weighted local line, sigma 0.1 mm), "
                "pre LOSO",
    "Correction": "paired nested bootstrap per 0.25-mm landing bin, Benjamini-Hochberg across bins [traced: render logs "
                  "panels_nonlinear_first_20261117/logs/nl_inside.log, panels_nonlinear_dock_20261117/logs/nl_inside.log, "
                  "panels.log]",
}
body = [
    "Encoding page Fig 12.2 S2 [#f-lick_matched_panels_nonlinear] (goal page v319). Fig 12.2 (previous slide) with the "
    "NON-LINEAR decoder: RBF kernel ridge on the same features (z-scored Allen areas with the ALM / tjM1 split, lags "
    "0-0.2 s), frozen pre-stroke, leave-one-pre-session-out; its linear refit reproduces the KEY decoder to 0.03 mm.",
    "Session means, change vs pre (mm; 4/4 unless stated): ITI all +0.34 / +0.27 / +0.28; ITI first lick of a bout "
    "+0.39 / +0.28 / +0.28; docked +0.28 / +0.25 (3/4) / +0.30; cued all +0.24 / +0.13 / +0.02; cued first +0.36 / "
    "+0.25 / +0.18. Strict row: significant bins identical to the top row except docked acute (7 -> 6 of 8). Cued shifts "
    "slightly larger than with the linear decoder (as before); ITI essentially the same.",
    "Why the curves are still compressed toward the middle: any decoder fitted to minimize prediction error shrinks toward "
    "the mean when the activity carries position only partly (for least squares the decoded-vs-true slope equals R2); "
    "ridge regularization adds a little. The comparison is with pre licks landing at the same place, read by the same "
    "frozen decoder, so the compression cancels in the change vs pre.",
    "Licks: LP video licks (cam4 detector) with 3-D tongue position from cam1 + cam4 pose; no DAQ spout-contact input.",
    "FIGURE SOURCE: Encoding (goal) page Fig 12.2 S2 (https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d, v319) · render "
    "M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
    "iti_licks/panels_nonlinear_first_20261117/<tag>/figures/lick_matched_panels.png (WFIELD_LICK_DECODER=nonlinear "
    "scripts/lick_matched_readout.py --ref loclin --first-curves --inside-curves; scripts/lick_matched_panels.py) · deck "
    "image: presentations/_source_PI_overview_2026-11-17/swap_inputs/s125b_lick_matched_panels_nonlinear_crop.png",
    "LICK SOURCE: Fig 12.2 S2 [licks: LP video, cam4 (3-D)].",
]
if new.notes_slide.notes_text_frame is None:      # a fresh notes page here has no body placeholder: copy 125's
    new.notes_slide.shapes._spTree.append(copy.deepcopy(base.notes_slide.notes_placeholder._element))
write_notes(new, block, body)
# move the new (last) slide to right after 125 and hide it (supporting slide)
lst = P.slides._sldIdLst
el = lst[-1]
lst.remove(el)
lst.insert(125, el)
new._element.set("show", "0")
P.save(dst)
print("added hidden slide 126:", P.slides[125].shapes[0].text_frame.text, "| total", len(P.slides))
