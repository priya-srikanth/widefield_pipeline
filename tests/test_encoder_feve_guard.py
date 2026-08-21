"""The FEVE degenerate-collapse guard, and the log line that reports it.

The pooled FEVE figures are rebuilt every night. A run in which most sessions fail to load
overwrites a 64-region heatmap with a one-region strip and renders it perfectly happily, which is
how slide 21 came to read as empty (Priya, 2026-08-20). The guard refuses that overwrite.

On 2026-08-21 the guard fired correctly in the nightly -- and the log said:

    !! FEVE region axis collapsed to 0 region(s) from 58 session(s) -- REFUSING to overwrite ...
    wrote locanmf_encoder_feve_by_region_pooled.png

The figure on disk was intact. Only the report was wrong, because the builder returned the path it
had refused to write and the caller printed "wrote" unconditionally. A guard whose refusal reads as
a success is worse than no guard: it trains you to ignore the warning above it.
"""
import pytest

from wfield_local import locanmf_position_encoder as enc


# ---------------------------------------------------------------- the guard itself

def test_a_collapsed_region_axis_is_refused():
    res = {f"PS9{i % 4 + 2}_08{i:02d}": object() for i in range(58)}
    assert enc._degenerate_feve(res, regs=[], out_path="x.png", verbose=False) is True


def test_a_nearly_collapsed_axis_is_refused():
    res = {"PS94_0812": object(), "PS95_0813": object()}
    assert enc._degenerate_feve(res, regs=["MOp"], out_path="x.png", verbose=False) is True


def test_a_full_region_axis_is_written():
    res = {"PS94_0812": object()}
    regs = [f"r{i}" for i in range(enc.MIN_FEVE_REGIONS)]
    assert enc._degenerate_feve(res, regs, out_path="x.png", verbose=False) is False


def test_no_sessions_at_all_is_not_a_collapse():
    """Nothing loaded is a different failure, and not one this guard should claim to have caught."""
    assert enc._degenerate_feve({}, regs=[], out_path="x.png", verbose=False) is False


# ---------------------------------------------------------------- the report

class _Q:
    name = "locanmf_encoder_feve_by_region_pooled.png"


def test_a_refusal_never_reports_as_a_write():
    """The 2026-08-21 bug, pinned."""
    msg = enc._feve_log(enc.fig_region_feve_pooled, None)
    assert not msg.startswith("wrote"), msg
    assert "kept existing" in msg
    assert "locanmf_encoder_feve_by_region_pooled.png" in msg


def test_a_refusal_names_the_right_figure():
    msg = enc._feve_log(enc.fig_region_feve_sessions, None)
    assert "locanmf_encoder_feve_by_region_sessions.png" in msg
    assert "pooled" not in msg


def test_a_real_write_still_reports_the_basename():
    msg = enc._feve_log(enc.fig_region_feve_pooled, _Q())
    assert msg == "wrote locanmf_encoder_feve_by_region_pooled.png"


@pytest.mark.parametrize("f", [enc.fig_region_feve_pooled, enc.fig_region_feve_sessions])
def test_both_builders_are_covered_by_the_namer(f):
    assert enc._feve_name(f).endswith(".png")
