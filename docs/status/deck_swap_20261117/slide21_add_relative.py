"""Slide 21 (Priya 11-17 "can we add the per-animal pre-normalized versions?"): stack the normalized-to-own-pre figure under
the absolute one. Legend rows cut from the second copy (same legend as the top); caption moved down and extended."""
import os
import shutil

from PIL import Image
from pptx import Presentation
from pptx.util import Emu

F = "M:/MICROSCOPE/Priya/Widefield/presentations/Overview_2026-11-06.pptx"
B = F.replace(".pptx", ".bak_pre_a8_s21rel_20261117.pptx")
REL = ("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/rebuild_figs/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_"
       "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/rel_own_pre_20261117/"
       "state_position_sessions_rel_own_pre.png")
OUT = "M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs/s021b_state_position_rel_own_pre.png"
if not os.path.exists(OUT):
    im = Image.open(REL)
    im.crop((0, 90, im.width, im.height)).save(OUT)
if not os.path.exists(B):
    shutil.copy2(F, B)
P = Presentation(F)
s = P.slides[20]
title = s.shapes[0].text_frame.text
assert title.startswith("Frozen decoders: position fails acutely"), title
pic = next(sh for sh in s.shapes if sh.name == "fig:state_position_sessions")
cap = next(sh for sh in s.shapes if sh.name == "TextBox 5")
assert not any(sh.name == "fig:state_position_rel_own_pre" for sh in s.shapes), "already added"
x0, x1, y0, ycap = 4434840, 4434840 + 7498079, 1143000, 6858000 - 640080 - 91440
a1 = Image.open(pic.image.filename if False else OUT.replace("s021b_state_position_rel_own_pre", "s021_state_position")).size
a2 = Image.open(OUT).size
r1, r2 = a1[1] / a1[0], a2[1] / a2[0]
W = min(x1 - x0, int((ycap - y0 - 45720) / (r1 + r2)))
left = x0 + (x1 - x0 - W) // 2
pic.left, pic.top, pic.width, pic.height = Emu(left), Emu(y0), Emu(W), Emu(int(W * r1))
p2 = s.shapes.add_picture(OUT, Emu(left), Emu(y0 + int(W * r1) + 45720), Emu(W), Emu(int(W * r2)))
p2.name = "fig:state_position_rel_own_pre"
cap.top, cap.height = Emu(ycap), Emu(640080)
para = cap.text_frame.paragraphs[0]
new = ("Top: above-chance retained; bottom: each session divided by its animal's own pre mean (pre = 1). One open marker "
       "per session (shape = animal, color = epoch); filled dot + bar = mean of animals ± SEM; frozen pre-stroke decoders "
       "(state: bal_em centering), cue window; change vs own pre, boot_delta p, Holm over 6: * < .05, ** < .01")
para.runs[0].text = new
for r in para.runs[1:]:
    r._r.getparent().remove(r._r)
tf = s.notes_slide.notes_text_frame
ps = tf.paragraphs
idx = next(i for i, p in enumerate(ps) if p.text.startswith("RELATIVE to each animal"))
ps[idx].runs[0].text = ps[idx].text.replace("Figure: rebuild_figs/<tag>/rel_own_pre_20261117/",
                                            "Bottom picture: rebuild_figs/<tag>/rel_own_pre_20261117/")
for r in ps[idx].runs[1:]:
    r._r.getparent().remove(r._r)
P.save(F)
print("ok", W, r1, r2)
