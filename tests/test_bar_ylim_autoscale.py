"""A quantity with no natural range must not be drawn against a fixed window.

WHY (2026-09-07, Priya from deck slides 590-592 and 602-604). The section I bars were clipped:
`epoch_8diag_matrices_crossnobis_*` fell back to (-0.05, 1.10) and `epoch_9_delta_trajectory_*`
was pinned at (-0.65, 0.35). 1.10 is a CORRELATION's bound applied to a crossnobis DISTANCE --
`MATRIX_FAMILIES` records `None` for that family precisely because "a crossnobis distance does
not [have a natural range], and forcing one on it would compress every panel into a corner". The
matrix honoured that; the diagonal bars did not.

THE ERROR BARS GO FIRST. A mean can sit comfortably inside a fixed window while its 95% interval
and its individual session dots run off the top, so the span has to be taken over intervals and
points, not bar heights.
"""
import numpy as np

from wfield_local import epoch_figures as ef


def test_span_covers_intervals_and_points_not_just_bars():
    lo, hi = ef._data_span({"e": {"p": (0.5, 0.2, 1.9)}}, {"e": {"p": [("PS92", 2.4)]}})
    assert lo <= 0.2, "the interval's lower end must be inside the span"
    assert hi >= 2.4, "a session dot above every bar must still be inside the span"


def test_span_survives_empty_and_nonfinite():
    assert ef._data_span({}, {}) == (0.0, 1.0)
    lo, hi = ef._data_span({"e": {"p": (np.nan, None, None)}}, {})
    assert np.isfinite(lo) and np.isfinite(hi)


def test_the_crossnobis_diagonal_inherits_the_family_scale():
    """The family says None; the bars must not substitute a correlation's bound."""
    import inspect

    from wfield_local import epoch_grant_figures as eg
    src = inspect.getsource(eg._matrix_family)
    assert "ylim=(None if scale is None" in src, (
        "the diagonal bars must autoscale when the family has no fixed scale, as the matrix does")
    assert "1.10" not in src.split("ylim=(None if scale is None")[1][:200], (
        "the correlation fallback is back in the diagonal bars' ylim")


def test_the_delta_trajectory_is_not_pinned_to_a_hand_fitted_window():
    """AST, not a text search: the comment explaining the fix quotes the old window verbatim.

    A grep for "(-0.65, 0.35)" matches the explanation as readily as a regression -- the same trap
    that made an earlier guard in this repo fire on its own documentation. Resolve the actual
    keyword argument instead.
    """
    import ast
    import inspect
    import textwrap

    from wfield_local import epoch_grant_figures as eg

    tree = ast.parse(textwrap.dedent(inspect.getsource(eg._fig_9)))
    ylims = [kw.value for n in ast.walk(tree) if isinstance(n, ast.Call)
             for kw in n.keywords if kw.arg == "ylim"]
    assert ylims, "no ylim argument found in _fig_9"
    for v in ylims:
        assert isinstance(v, ast.Constant) and v.value is None, (
            "a change-from-baseline has no natural range; a hand-fitted window clips the "
            "trajectory it exists to show as soon as recovery moves outside it")
