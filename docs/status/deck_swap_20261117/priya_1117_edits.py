"""Priya 11-17 decisions: slide 22 kept as a pre-rebuild record (notes line); slide 12 wording for the degenerate
mixed-model cell (per-session version queued). Slide 157 rewrite is applied by apply_specs.py (status set ready)."""
import copy
import os
import shutil

from pptx import Presentation
from pptx.text.text import _Paragraph

F = "M:/MICROSCOPE/Priya/Widefield/presentations/Overview_2026-11-06.pptx"
B = F.replace(".pptx", ".bak_pre_a8_priya1117_20261117.pptx")
if not os.path.exists(B):
    shutil.copy2(F, B)
P = Presentation(F)


def title(s):
    return s.shapes[0].text_frame.text


def set_text(p, t):
    p.runs[0].text = t
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)


s22 = P.slides[21]
assert title(s22).startswith("PS94 state: bal_em restores running"), title(s22)
tf = s22.notes_slide.notes_text_frame
first = tf.paragraphs[0]._p
if not tf.paragraphs[0].text.startswith("PRE-REBUILD RECORD"):
    p = copy.deepcopy(first)
    first.addprevious(p)
    set_text(_Paragraph(p, tf), "PRE-REBUILD RECORD (Priya 11-17: keep as is). Image and numbers are the 11-03 render on "
             "pre-rebuild inputs (session_z + bal_em frozen state models, pre-rebuild feature cache). Not re-run: the "
             "rebuild's frozen store holds only the bal_em state models. The Data: lines below were matched to this "
             "picture by file name in the 11-16 pass and may describe the rebuild step rather than this 11-03 run.")

s12 = P.slides[11]
assert title(s12).startswith("Mixed-model stars: frozen decoder by spout, cue"), title(s12)
tb = next(sh for sh in s12.shapes if sh.name == "TextBox 3")
para = tb.text_frame.paragraphs[0]
assert "acute far contra not tested" in para.text, para.text
set_text(para, "•  acute far ipsi *, acute far mid *, acute close ipsi *, subacute far contra *; acute far contra: "
         "mixed model degenerate (all 4 animals −.61 to −.67, no between-animal variance; see the bootstrap on the "
         "previous slide)")
P.save(F)
print("ok")
