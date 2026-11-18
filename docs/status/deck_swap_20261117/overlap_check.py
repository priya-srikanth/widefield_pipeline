"""Flag slides where a non-empty text box overlaps a picture (bounding boxes) -- Priya 11-17 "ensure text isn't obscured
by or obscuring figures". Text-box frames can be larger than their text, so flagged slides are then checked by eye
(export_slides.ps1 on a scratch copy). usage: python overlap_check.py <deck> [slides comma list | all]"""
import sys

from pptx import Presentation


def boxes(s):
    pics, texts = [], []
    for sh in s.shapes:
        b = (sh.left or 0, sh.top or 0, (sh.left or 0) + (sh.width or 0), (sh.top or 0) + (sh.height or 0))
        if sh.shape_type == 13:
            pics.append((sh.name, b))
        elif sh.has_text_frame and sh.text_frame.text.strip():
            texts.append((sh.name, b, sh.text_frame.text.strip()[:40]))
    return pics, texts


def inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


P = Presentation(sys.argv[1])
want = None if len(sys.argv) < 3 or sys.argv[2] == "all" else {int(x) for x in sys.argv[2].split(",")}
for i, s in enumerate(P.slides, 1):
    if want and i not in want:
        continue
    pics, texts = boxes(s)
    for pn, pb in pics:
        if pb[2] > P.slide_width + 9144 or pb[3] > P.slide_height + 9144:
            print(f"{i}\tPIC OFF SLIDE\t{pn}")
        for tn, tb, tt in texts:
            a = inter(pb, tb)
            if a > 0:
                frac = a / max(1, (tb[2] - tb[0]) * (tb[3] - tb[1]))
                if frac > 0.02:
                    print(f"{i}\t{frac:.2f}\t{tn} x {pn}\t{tt}")
