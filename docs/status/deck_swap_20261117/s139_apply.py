"""Slide 139 (Priya 11-17): title "Cued ipsi-contra balance places an unseen animal on recovery"; keep the caveat, add the
cued-angle result and the ITI / docked result as their own bullets; layout 3a = bullets on top full width, 4-column
figure full width below, caption at the bottom. Notes + Data: block from specs/s139_cuedang.json (apply_specs helpers)."""
import copy
import json
import os
import shutil
import sys

from PIL import Image
from pptx import Presentation
from pptx.util import Emu, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_specs as A  # noqa: E402

F = "M:/MICROSCOPE/Priya/Widefield/presentations/Overview_2026-11-06.pptx"
B = F.replace(".pptx", ".bak_pre_a8_s139_20261117.pptx")
SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs", "s139_cuedang.json")
TITLE = "Cued ipsi-contra balance places an unseen animal on recovery"
BULLETS = [
    "•  Balance after the first lick (orofacial + SSs, angle-matched gain ratio) vs lick direction",
    "•  Cued: places an unseen animal on its far-contra aim (R² .66 vs .23 for days) and on its own cued lick angle "
    "(.68 vs .08): it tracks recovery of lick direction",
    "•  State marker, not a forecast: it does not predict the next session's change",
    "•  Caveat: for cued licks it largely re-describes the landing error (activity stays target-like while the lick "
    "lands ipsilesionally)",
    "•  ITI: all-ITI balance places an unseen animal on its ITI lick angle (.44 vs .17); on docked-spout licks only, "
    "the link is lost (R² ≤ .05)",
]

if not os.path.exists(B):
    shutil.copy2(F, B)
spec = json.load(open(SPEC, encoding="utf-8"))
spec = spec[0] if isinstance(spec, list) else spec
P = Presentation(F)
s = P.slides[138]
title = s.shapes[0].text_frame.text
assert A.norm("Ipsi-contra balance places an unseen animal")[:30] in A.norm(title), title
W, H = P.slide_width, P.slide_height
sh = {x.name: x for x in s.shapes}
# title
A.set_para_text(sh["TextBox 1"].text_frame.paragraphs[0], TITLE)
# bullets: reuse paragraph formatting, full width on top
tb = sh["TextBox 3"]
tf = tb.text_frame
ps = tf.paragraphs
size = next((r.font.size for r in ps[0].runs if r.font.size), Pt(16))
first = ps[0]._p
for p in ps[1:]:
    p._p.getparent().remove(p._p)
from pptx.text.text import _Paragraph  # noqa: E402
A.set_para_text(_Paragraph(first, tf), BULLETS[0])
prev = first
for t in BULLETS[1:]:
    q = copy.deepcopy(first)
    prev.addnext(q)
    A.set_para_text(_Paragraph(q, tf), t)
    prev = q
for p in tf.paragraphs:
    for r in p.runs:
        r.font.size = Pt(13)
left, width = Emu(457200), Emu(11277295)
tb.left, tb.top, tb.width, tb.height = left, Emu(1080000), width, Emu(1900000)
# caption at the bottom, full width
cap = sh["TextBox 5"]
cap_new = next(e["new"] for e in spec["slide_text"] if e["shape_name"] == "TextBox 5")
A.set_para_text(cap.text_frame.paragraphs[0], cap_new)
cap.left, cap.width, cap.top, cap.height = left, width, Emu(H - 600000), Emu(520000)
# figure: full width between bullets and caption
pic = sh["fig:lick_recovery_primary"]
img = spec["pictures"][0]["new_image"]
iw, ih = Image.open(img).size
top, bottom = 3050000, H - 640000
h = bottom - top
w = int(h * iw / ih)
if w > width:
    w = int(width)
    h = int(w * ih / iw)
A.swap_picture(s, pic, img, box=[int(left + (width - w) // 2), top, w, h])
# notes + Data: block
A.write_notes(s, spec["data_block"], spec["notes_body"])
P.save(F)
print("ok", size, w, h)
