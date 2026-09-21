"""The `|| text` fallback: an absence the note's author declared is not a build warning.

Added 2026-09-17 for the `epoch_14` beta-map note, which is glob-placed across the cue, pre-cue and
lick arms and CANNOT resolve on the lick arm. That arm has no `Far Contra / acute - pre` row at all,
because the 20-trial floor refuses the cell -- the animal does not lick far-contralateral acutely,
and that refusal IS the deficit the family is describing. Before the fallback existed the published
slide carried a literal `[[? ... has no row ...]]` marker and every deck build reported the same
unresolved sidecar, which is a permanent warning for correct behaviour and trains its reader to
ignore warnings.

THE SCOPE IS THE DESIGN. `||` covers a missing ROW only. A missing FILE means a family was never
rendered and a missing COLUMN means the sidecar's schema changed under the note; no author can
legitimately anticipate either, so both stay loud regardless of what the token says. These tests
pin that boundary, because widening it later would silently re-enable the failure mode
`deck_values` exists to prevent.
"""
from __future__ import annotations

import pytest

from wfield_local import deck_values as dv


@pytest.fixture
def sidecar(tmp_path):
    """One stats sidecar with a Far Contra row and NO Far Middle row -- the real asymmetry."""
    (tmp_path / "epoch_x_cue_lick_stats.csv").write_text(
        "row,col,amplitude_vs_pre,sig_bins,n_bins\n"
        "Far Contra,acute - pre,2.1771,971,2022\n", encoding="utf-8")
    return tmp_path


def test_a_fallback_is_used_when_the_row_is_absent(sidecar):
    r = dv.Resolver([sidecar])
    out = r.resolve("amplitude {{SELF_stats: row=Far Middle, col=acute - pre -> "
                    "amplitude_vs_pre:.2f || REFUSED (<20 trials)}}.",
                    self_stem="epoch_x_cue_lick")
    assert out == "amplitude REFUSED (<20 trials)."


def test_a_fallback_is_not_counted_as_a_miss(sidecar):
    """THE WHOLE POINT. An absence declared in the token is not a build warning."""
    r = dv.Resolver([sidecar])
    r.resolve("{{SELF_stats: row=Far Middle, col=acute - pre -> amplitude_vs_pre:.2f || n/a}}",
              self_stem="epoch_x_cue_lick")
    assert r.report() == []


def test_a_fallback_is_ignored_when_the_row_exists(sidecar):
    """It must never shadow a real number, or the two arms that DO resolve print the excuse."""
    r = dv.Resolver([sidecar])
    out = r.resolve("{{SELF_stats: row=Far Contra, col=acute - pre -> amplitude_vs_pre:.2f "
                    "|| REFUSED}}", self_stem="epoch_x_cue_lick")
    assert out == "2.18"
    assert r.report() == []


def test_a_fallback_does_NOT_excuse_a_missing_sidecar_file(sidecar):
    """A missing FILE means the family was never rendered. `||` must not hide that."""
    r = dv.Resolver([sidecar])
    out = r.resolve("{{epoch_nope: row=A, col=B -> x || fine}}")
    assert "sidecar missing" in out
    assert r.report(), "a missing file must still be reported"


def test_a_fallback_does_NOT_excuse_a_missing_column(sidecar):
    """A missing COLUMN means the schema changed under the note -- always a defect."""
    r = dv.Resolver([sidecar])
    out = r.resolve("{{SELF_stats: row=Far Contra, col=acute - pre -> no_such_field || fine}}",
                    self_stem="epoch_x_cue_lick")
    assert "has no column" in out
    assert r.report(), "a missing column must still be reported"


def test_a_format_spec_still_works_alongside_a_fallback(sidecar):
    """The spec group had to stop swallowing the pipe; this is that regression guard."""
    r = dv.Resolver([sidecar])
    assert r.resolve("{{SELF_stats: row=Far Contra, col=acute - pre -> amplitude_vs_pre:+.3f "
                     "|| x}}", self_stem="epoch_x_cue_lick") == "+2.177"


def test_a_token_without_a_fallback_still_marks_and_counts(sidecar):
    """Default behaviour unchanged: silence is still not an option."""
    r = dv.Resolver([sidecar])
    out = r.resolve("{{SELF_stats: row=Far Middle, col=acute - pre -> amplitude_vs_pre}}",
                    self_stem="epoch_x_cue_lick")
    assert "has no row" in out
    assert r.report()


def test_the_epoch_14_note_actually_carries_a_fallback():
    """The fix must be WIRED, not merely available -- the defect this file closes was a real
    unresolved reference reported by the 2026-09-17 deck build, not a hypothetical."""
    from conftest import deck_source

    src = deck_source()          # the epoch_14 legend moved to deck_registry on 2026-09-21
    i = src.find("row=Far Contra, col=acute - pre")
    assert i > 0, "the epoch_14 token is gone; this guard needs rewriting"
    assert "||" in src[i:i + 400], "the epoch_14 far-contra token lost its fallback"
