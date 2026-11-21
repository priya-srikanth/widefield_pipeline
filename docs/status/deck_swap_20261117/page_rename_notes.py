"""Deck NOTES only: cite the renamed results pages (Priya 11-18 OK, via 3c: imaging page -> "Spout (Target) Coding After
Stroke", goal / encoding page -> "Movement (Execution) Coding After Stroke"; URLs unchanged). Run-level replacement keeps
formatting; slide text is not touched. usage: python page_rename_notes.py <deck_in> <deck_out>"""
import re
import sys

from pptx import Presentation

RULES = [(re.compile(r"[Ee]ncoding \(goal\) page"), "Movement (Execution) page"),
         (re.compile(r"[Ee]ncoding page"), "Movement (Execution) page"),
         (re.compile(r"\b[Gg]oal page"), "Movement (Execution) page"),
         (re.compile(r"[Ii]maging page"), "Spout (Target) page")]
LEFT = re.compile(r"([Ii]maging|[Ee]ncoding|[Gg]oal)\)? page")


def main(src, dst):
    P = Presentation(src)
    n, left = 0, []
    for i, s in enumerate(P.slides, 1):
        if not s.has_notes_slide or s.notes_slide.notes_text_frame is None:
            continue
        tf = s.notes_slide.notes_text_frame
        for p in tf.paragraphs:
            for r in p.runs:
                t = r.text
                for rx, new in RULES:
                    t, k = rx.subn(new, t)
                    n += k
                if t != r.text:
                    r.text = t
        if LEFT.search(tf.text):
            left.append(i)
    P.save(dst)
    print(f"{n} replacements; slides with a remaining old name (split across runs?): {left}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
