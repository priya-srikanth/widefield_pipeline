"""Deck notes that QUOTE a figure's sidecar instead of hard-coding the number.

Priya, 2026-09-12: "yes have notes reference the sidecar."

171 decimal statistics were hand-copied into 33 deck notes with nothing linking them to the figures
they describe, so a re-render moved the figure and left the caption asserting the old value. These
pin the four properties that make the replacement trustworthy: it resolves from the real CSV, it
fails visibly rather than silently, it never averages an ambiguous match, and `SELF` gives each
glob-placed arm its own numbers.
"""
from __future__ import annotations

import pytest

from wfield_local import deck_values as dv


@pytest.fixture()
def sidecar(tmp_path):
    (tmp_path / "epoch_x_cue_lick.csv").write_text(
        "epoch,position,point,lo95,hi95\n"
        "acute,frozen,-0.0896,-0.1632,-0.0274\n"
        "acute,gap,0.0397,-0.0144,0.0984\n"
        "chronic,frozen,-0.0608,-0.0539,0.0363\n"
        "pre,frozen,,,\n", encoding="utf-8")
    (tmp_path / "epoch_x_precue_working.csv").write_text(
        "epoch,position,point,lo95,hi95\n"
        "acute,frozen,-0.5000,-0.6,-0.4\n", encoding="utf-8")
    return tmp_path


def test_a_token_resolves_from_the_real_sidecar(sidecar):
    r = dv.Resolver([sidecar])
    got = r.resolve("deficit {{epoch_x_cue_lick: epoch=acute, position=frozen -> point:+.3f}}")
    assert got == "deficit -0.090"
    assert r.report() == []


def test_the_format_spec_is_optional(sidecar):
    r = dv.Resolver([sidecar])
    assert r.resolve("{{epoch_x_cue_lick: epoch=acute, position=gap -> point}}") == "0.0397"


def test_an_empty_cell_reads_as_not_applicable(sidecar):
    """A pre row has no delta; printing an empty string would look like a formatting bug."""
    r = dv.Resolver([sidecar])
    assert r.resolve("{{epoch_x_cue_lick: epoch=pre, position=frozen -> point}}") == "n/a"


@pytest.mark.parametrize("token,fragment", [
    ("{{nope: epoch=acute -> point}}", "sidecar missing"),
    ("{{epoch_x_cue_lick: epoch=zzz, position=gap -> point}}", "has no row"),
    ("{{epoch_x_cue_lick: epoch=acute, position=gap -> nocol}}", "has no column"),
])
def test_a_failure_is_visible_and_never_raises(sidecar, token, fragment):
    """The deck must still build; a caption that cannot find its number must say so on the slide."""
    r = dv.Resolver([sidecar])
    assert fragment in r.resolve(token)
    assert r.report(), "every failure is logged for the build"


def test_an_ambiguous_selector_is_a_failure_not_a_mean(sidecar):
    """Averaging two rows would invent a number that appears in no sidecar."""
    r = dv.Resolver([sidecar])
    out = r.resolve("{{epoch_x_cue_lick: epoch=acute -> point}}")
    assert "ambiguous" in out and "2 rows" in out


def test_an_ambiguous_selector_that_agrees_is_fine(sidecar):
    """Two rows with the SAME value are not ambiguous -- the answer is unique even if the match is not."""
    r = dv.Resolver([sidecar])
    assert r.resolve("{{epoch_x_cue_lick: position=frozen -> point}}").startswith("[[?")
    assert r.resolve("{{epoch_x_cue_lick: epoch=acute, position=frozen -> lo95}}") == "-0.1632"


def test_SELF_gives_each_arm_its_own_numbers(sidecar):
    """One glob-placed caption serves five arms; a hard-coded stem would print one arm's numbers on all."""
    r = dv.Resolver([sidecar])
    tok = "{{SELF: epoch=acute, position=frozen -> point:+.3f}}"
    assert r.resolve(tok, self_stem="epoch_x_cue_lick") == "-0.090"
    assert r.resolve(tok, self_stem="epoch_x_precue_working") == "-0.500"


def test_SELF_without_a_figure_is_reported(sidecar):
    r = dv.Resolver([sidecar])
    assert "no figure" in r.resolve("{{SELF: epoch=acute -> point}}")
    assert r.report()


def test_text_without_tokens_is_returned_untouched(sidecar):
    r = dv.Resolver([sidecar])
    s = "ordinary prose with a { brace and a 0.123 number"
    assert r.resolve(s) is s


def test_stem_of_strips_the_extension():
    assert dv.stem_of("epoch_9_delta_cue_working.png") == "epoch_9_delta_cue_working"
    assert dv.stem_of(None) is None


def test_each_sidecar_is_read_once_per_build(sidecar):
    """A deck resolves hundreds of tokens; re-reading the CSV for each would be the slow way."""
    r = dv.Resolver([sidecar])
    for _ in range(5):
        r.resolve("{{epoch_x_cue_lick: epoch=acute, position=gap -> point}}")
    assert list(r._cache) == ["epoch_x_cue_lick"]
