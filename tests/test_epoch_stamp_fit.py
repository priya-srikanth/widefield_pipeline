"""`epoch_figures._fit_stamp` (11-17, image page Fig 2.3): the run / trial-set footer is wrapped to the canvas and the
canvas grows under the figure when the footer would sit on the content; a figure it does not touch is unchanged."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wfield_local import epoch_figures as ef  # noqa: E402


def _fig(bottom):
    fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=100)
    ax.bar(range(6), [0.9, 0.8, 0.9, 0.8, 0.8, 0.2])
    ax.set_xticks(range(6))
    ax.set_xticklabels(["Ipsi", "Middle", "Contra"] * 2)
    fig.subplots_adjust(bottom=bottom)
    ax.text(1, -0.2, "Near", transform=ax.get_xaxis_transform(), va="top")
    return fig, ax


def test_long_multiline_stamp_is_wrapped_and_moved_off_the_content():
    fig, ax = _fig(0.12)
    long = "mode " + "x" * 30 + "_" + "y" * 60 + "; " + " ".join(["word"] * 60)
    st = fig.text(0.995, 0.002, long + "\n" + "camera gate: partial coverage", ha="right", va="bottom",
                  fontsize=6, color=ef._STAMP_COLOR)
    h0 = fig.get_size_inches()[1]
    ax_h0 = ax.get_position().height * h0
    added = ef._fit_stamp(fig)
    assert added > 0
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = st.get_window_extent(r)
    assert bb.x0 >= 0                                                   # nothing clipped at the left edge
    assert bb.y1 < ax.get_tightbbox(r).y0                               # off the tick labels and the group label
    assert abs(ax.get_position().height * fig.get_size_inches()[1] - ax_h0) < 1e-6   # axes size in inches kept
    plt.close(fig)


def test_figure_with_room_is_left_alone():
    fig, ax = _fig(0.45)
    fig.text(0.995, 0.002, "short stamp", ha="right", va="bottom", fontsize=6, color=ef._STAMP_COLOR)
    size0, pos0 = tuple(fig.get_size_inches()), tuple(ax.get_position().bounds)
    assert ef._fit_stamp(fig) == 0.0
    assert tuple(fig.get_size_inches()) == size0 and tuple(ax.get_position().bounds) == pos0
    plt.close(fig)


def test_no_stamp_no_change():
    fig, _ax = _fig(0.12)
    assert ef._fit_stamp(fig) == 0.0
    plt.close(fig)
