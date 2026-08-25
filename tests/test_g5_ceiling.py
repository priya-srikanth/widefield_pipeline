"""G5 must draw the pre-stroke ceiling, and must still draw at all without it.

`r_pre_loo` is written by `poststroke_compare.pattern_similarity` only from 2026-08-25 onward, so
every `section_g.json` written before then lacks the field. A figure that crashed on those would take
the whole section-G render down with it; a figure that silently omitted the band would invite the
comparison with 1.0 the band exists to prevent. Both cases are pinned here.
"""
import numpy as np
import pytest

pytest.importorskip("matplotlib")

from wfield_local import plot_poststroke as pp

POS = pp.POS


def _sim(with_ceiling):
    rng = np.random.default_rng(0)
    out = {}
    for lab in ("PS94_0817", "PS94_0818"):
        d = {}
        for p in POS:
            e = {"r": float(rng.uniform(0.1, 0.8)), "n_pre": 200, "n_post": 90}
            if with_ceiling:
                e["r_pre_loo"] = 0.82
                e["n_pre_sessions"] = 6
            d[p] = e
        out[lab] = d
    return out


def _ceiling_lines(ax):
    """Horizontal segments at a constant y that are not the axhline at 0 -- i.e. the ceiling band."""
    n = 0
    for ln in ax.get_lines():
        y = ln.get_ydata()
        if len(y) == 2 and np.allclose(y[0], y[1]) and not np.allclose(y[0], 0.0):
            n += 1
    return n


def test_g5_renders_with_the_ceiling_present(tmp_path):
    """Smoke: the figure builds when the field is there. Its VALUE is checked below."""
    p = pp.fig_similarity(_sim(True), tmp_path, name="g5.png")
    assert p and p.exists() and p.stat().st_size > 0


def test_ceiling_is_absent_on_old_json_without_crashing(tmp_path):
    """A section_g.json written before 2026-08-25 has no r_pre_loo. The figure must still render."""
    p = pp.fig_similarity(_sim(False), tmp_path, name="g5_old.png")
    assert p and p.exists(), "G5 must survive a JSON that predates the ceiling field"


def test_ceiling_value_reaches_the_axes(tmp_path, monkeypatch):
    """The band is at r_pre_loo, not at 1.0 and not at the bar height."""
    captured = {}
    real = pp.plt.subplots

    def spy(*a, **k):
        fig, axes = real(*a, **k)
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(pp.plt, "subplots", spy)
    pp.fig_similarity(_sim(True), tmp_path, name="g5_val.png")
    ax = captured["axes"][0][0]
    ys = [ln.get_ydata()[0] for ln in ax.get_lines()
          if len(ln.get_ydata()) == 2 and np.allclose(*ln.get_ydata())]
    assert any(abs(y - 0.82) < 1e-9 for y in ys), f"ceiling 0.82 not drawn; found {ys}"
    assert not any(abs(y - 1.0) < 1e-9 for y in ys), "the ceiling must not be 1.0"
