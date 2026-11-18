"""Two deck edits (Priya 11-18): (1) slide 27 picture -> the 15f render with session markers in panel C (3c redraw, same
file name, stats unchanged), caption + one notes line; (2) slide 2 agenda bullet and slide 10 divider "Cortical spout
coding" -> "Spout (target) coding" (page renamed "Spout (Target) Coding After Stroke").
usage: python s27c_rename_1118.py <deck_in> <deck_out>"""
import sys
from pathlib import Path

from pptx import Presentation

sys.path.insert(0, str(Path(__file__).parent))
from apply_specs import find_shape, set_para_text, swap_picture  # noqa: E402
from s27_rest15f import SRC, make_crop  # noqa: E402

CROP = Path("M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs/"
            "s27_rest15f_floor30_pct5_sessC_crop.png")


def main(src, dst):
    img = make_crop(SRC, CROP)
    P = Presentation(src)
    s = P.slides[26]
    assert s.shapes[0].text_frame.text.startswith("Lick-free rest decoder"), s.shapes[0].text_frame.text
    swap_picture(s, find_shape(s, "fig:rest15f_floor30_pct5"), str(img), box=[365762, 2150000, 11460171, 4200000])
    cap = find_shape(s, "TextBox 5").text_frame.paragraphs[0]
    set_para_text(cap, "A: accuracy per session (chance 1/6); B: retained per animal; C: change in above-chance fraction "
                       "vs own pre (sessions faded, animals large), animal-weighted mean, 95% CI (* excludes 0)")
    tf = s.notes_slide.notes_text_frame
    ps = [p for p in tf.paragraphs if p.text.startswith("Licks: rest lick-free")]
    assert len(ps) == 1
    import copy
    new = copy.deepcopy(ps[0]._p)
    ps[0]._p.addprevious(new)
    from pptx.text.text import _Paragraph
    set_para_text(_Paragraph(new, tf), "Panel C (redrawn 11-18, Priya: 'add sessions to c'): large open markers = animal "
                  "means; faded small open markers = sessions (session above-chance fraction minus its animal's own pre "
                  "mean; pre sessions show the no-stroke spread); stats unchanged (CSVs identical). Picture: " + img.as_posix())
    for i, old, new_t in ((1, "•  Cortical spout coding", "•  Spout (target) coding"),
                          (9, "1 · Cortical spout coding", "1 · Spout (target) coding")):
        hits = [p for sh in P.slides[i].shapes if sh.has_text_frame for p in sh.text_frame.paragraphs if p.text == old]
        assert len(hits) == 1, (i + 1, len(hits))
        set_para_text(hits[0], new_t)
    P.save(dst)
    print("saved", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
