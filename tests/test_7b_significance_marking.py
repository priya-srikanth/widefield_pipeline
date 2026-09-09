"""7b's disattenuated panel marks significance with a BOX, not a printed interval.

The interval used to be drawn under the value as "[lo,hi]" at 6.2pt. Twelve day columns in a
~3.5in panel is a ~0.29in cell and that string needs about twice that, so neighbours overlapped
into unreadable runs like "[0.21,0.0948,0.0873,...]" -- numbers present on the figure and not
readable off it (Priya, 2026-09-08).
"""
import inspect
import re

from wfield_local import grant_figures as G

SRC = inspect.getsource(G.fig_reliability_verdict)


def test_the_interval_is_no_longer_printed_into_the_cell():
    assert "[{band[0]:.2f},{band[1]:.2f}]" not in SRC, \
        "the CI is being printed into the cell again; it does not fit"
    # and the label is back to one readable size rather than shrinking to fit two lines
    assert "fontsize=6.2" not in SRC, "6.2pt was the two-line size; the label should be 7.5"


def test_significance_is_marked_by_a_box_against_the_PRE_column():
    assert "add_patch" in SRC and "Rectangle" in SRC, "no box is drawn"
    # the comparator must be column 0 (PRE), not zero -- against zero nearly every cell would
    # qualify and the mark would carry no information
    assert re.search(r"band\[1\]\s*<\s*dis\[i,\s*0\]", SRC), \
        "the box must compare the CI upper bound against the PRE column"


def test_the_caption_explains_the_box_and_what_an_unboxed_cell_means():
    assert "BOXED cell" in SRC
    # an unboxed cell must not be readable as 'no change'
    assert "is NOT 'no change'" in SRC
