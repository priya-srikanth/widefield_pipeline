"""grant_kit._fit_stamp / _thin_crowded_ticks (redraw pass 11-09) and the computed engaged-selection label."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from wfield_local import grant_confusion, grant_kit  # noqa: E402


@pytest.fixture(autouse=True)
def _default_rc():
    """Matplotlib DEFAULTS for every test here, whatever was imported before.

    These tests measure text extents, so they depend on the global rcParams. The third-party
    `wfield` package restyles matplotlib AT IMPORT (wfield/viz.py: ``style.use(mplstyledict)`` --
    Arial, 12 pt, ...), and any test module that imports `wfield_local.cross_day_align` (e.g.
    test_cross_day_roi) pulls it in during collection. In a full run that moved the stamp's
    overlap partner and test_stamp_gets_its_own_band failed only there (11-09).
    """
    with matplotlib.rc_context():
        matplotlib.rcdefaults()          # leaves the backend alone (rcsetup STYLE_BLACKLIST)
        yield

DAYS = [1, 2, 3, 4, 5, 7, 9, 11, 15, 18, 22, 25, 29, 32, 36, 39, 43, 46, 50]


def _fig():
    fig, axs = plt.subplots(1, 3, figsize=(9, 4))
    for ax in axs:
        ax.plot(range(len(DAYS)), range(len(DAYS)))
        ax.set_xticks(range(len(DAYS)))
        ax.set_xticklabels([f"d{d}" for d in DAYS], fontsize=8)
        ax.set_xlabel("days from lesion")
    fig.text(0.5, 0.004, "Post-stroke sessions used (from the session config): PS92 20,  PS93 20,  PS94 20,  PS95 20"
             "      positions: c = close, f = far;  L / C / R = left / center / right, padded out to the right edge",
             ha="center", va="bottom", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig, axs


def test_stamp_gets_its_own_band():
    fig, _ = _fig()
    st = fig.text(0.995, 0.002, "LP licks: video lick attempts\nline 2\nline 3\nline 4", ha="right", va="bottom",
                  fontsize=6, color="#8a2be2")
    # the stamp may be either member of the pair -- the order follows fig.texts, not importance
    assert any("LP licks" in a or "LP licks" in b for a, b, _c in grant_kit._overlaps(fig))
    grant_kit._fit_stamp(fig, [st])
    assert not [r for r in grant_kit._overlaps(fig) if "LP licks" in r[0] or "LP licks" in r[1]]
    plt.close(fig)


def test_no_stamp_is_a_noop():
    fig, axs = _fig()
    pos = [a.get_position().bounds for a in axs]
    grant_kit._fit_stamp(fig, [])
    assert [a.get_position().bounds for a in axs] == pos
    plt.close(fig)


def test_day_ticks_thinned_last_day_kept():
    fig, axs = _fig()
    grant_kit._thin_crowded_ticks(fig)
    shown = [t.get_text() for t in axs[0].get_xticklabels() if t.get_visible()]
    assert shown[0] == "d1" and shown[-1] == "d50" and len(shown) < len(DAYS)
    assert not [r for r in grant_kit._overlaps(fig) if "xtick" in r[0]]
    plt.close(fig)


def test_position_ticks_never_thinned():
    fig, ax = plt.subplots(figsize=(1.2, 1.2))
    ax.plot(range(6), range(6))
    ax.set_xticks(range(6))
    ax.set_xticklabels(["cL", "cC", "cR", "fL", "fC", "fR"], fontsize=10)
    grant_kit._thin_crowded_ticks(fig)
    assert all(t.get_visible() for t in ax.get_xticklabels())
    plt.close(fig)


def test_engaged_selection_label_follows_lick_source(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    assert grant_confusion.engaged_selection_label().startswith("LP-attempted trials")
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "daq")
    assert grant_confusion.engaged_selection_label() == "DAQ contact trials"
