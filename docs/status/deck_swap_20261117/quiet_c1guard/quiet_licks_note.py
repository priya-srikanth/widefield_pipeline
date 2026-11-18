"""Quiet-event slides: extend the notes' "Licks:" data-block line with the cam1 far-from-mouth note (Priya 11-17: keep
k = 3.75 and the current figures; the guard re-run changes every per-animal rate and model term by < 2 %,
QUIET_C1GUARD_OLD_NEW_20261117.md). Idempotent. usage: python quiet_licks_note.py <deck_in> <deck_out>"""
import sys

from pptx import Presentation

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from apply_specs import set_para_text  # noqa: E402

SLIDES = {112: "Backup: post-stroke gain", 154: "Quiet example", 155: "Quiet traces", 156: "Quiet orofacial events",
          157: "Holds at matched stillness", 158: "Pre-cue event", 203: "Backup: quiet event maps"}
NOTE = ("; cam1-only far-from-mouth labels (med_out < -120 px) were still counted as licks in this render -- removing them "
        "(rule of 11-17) changes every per-animal quiet-event rate and model term by < 2 % (re-run, same k 3.75)")

P = Presentation(sys.argv[1])
for n, title in SLIDES.items():
    s = P.slides[n - 1]
    t = next(x.text_frame.text for x in s.shapes if x.has_text_frame and x.text_frame.text.strip())
    assert t.startswith(title), (n, t)
    ps = [p for p in s.notes_slide.notes_text_frame.paragraphs if p.text.startswith("Licks:")]
    assert len(ps) == 1, (n, len(ps))
    if NOTE not in ps[0].text:
        set_para_text(ps[0], ps[0].text.rstrip(". ") + NOTE)
    print(n, "ok |", ps[0].text[:90])
P.save(sys.argv[2])
