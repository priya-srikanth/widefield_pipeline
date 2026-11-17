"""Shared figure conventions for the FaceRhythm outputs (Priya 10-22: "increase font size, ensure no overlap of titles or
legends with each other or figures, and ensure methods are documented").

* `style()`      a context with readable fonts (11 pt titles / labels, 10 pt ticks, 9 pt legends). Use it around
                 every FR figure: ``with frf.style(): fig, axs = frf.subplots(...)``.
* `subplots()`   `plt.subplots` with CONSTRAINED layout, which reserves room for titles, suptitles, colorbars and
                 legends placed outside the axes (``legend_outside``) -- tight_layout overlapped them.
* `save()`       writes ``<dir>/<name>.png`` and records the figure's METHODS in ``<dir>/README.md`` (one section per
                 figure, replaced on every save; the repo's convention of a README describing each output folder, as
                 `orofacial_outputs.README`). Titles stay SHORT; how the numbers were made goes in the methods text.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import contextlib
import re
from pathlib import Path

RC = {"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
      "legend.fontsize": 9, "figure.titlesize": 13, "axes.titleweight": "normal"}


@contextlib.contextmanager
def style():
    import matplotlib
    with matplotlib.rc_context(RC):
        yield


def subplots(*args, **kw):
    import matplotlib.pyplot as plt
    kw.setdefault("layout", "constrained")
    return plt.subplots(*args, **kw)


def face_background(ax, home, mask=None) -> None:
    """Every mesh point (or those in ``mask``) as a small light-gray dot, under the marked points (Priya 11-07: show the
    full face on the significant-point maps)."""
    import numpy as np
    home = np.asarray(home)
    if mask is not None:
        home = home[np.asarray(mask, bool)]
    ax.scatter(home[:, 0], home[:, 1], s=1.2, color="0.85", lw=0, zorder=0)


def sign_rings(ax, home, mask, values, s: float = 7, lw: float = 0.5) -> None:
    """Rings around the marked points, dark red where the change is up and dark blue where it is down (Priya 11-07:
    outlines in the change's color instead of black)."""
    import numpy as np
    home, mask, values = np.asarray(home), np.asarray(mask, bool), np.asarray(values, float)
    for sel, col in ((mask & (values > 0), "#67001f"), (mask & (values <= 0), "#053061")):
        if sel.any():
            ax.scatter(home[sel, 0], home[sel, 1], s=s, facecolors="none", edgecolors=col, linewidths=lw)


def sig_cmap():
    """RdBu_r without its pale middle (Priya 11-07: a marked point near white reads as the gray background); marked
    points are never near 0, so nothing is lost."""
    import matplotlib
    import numpy as np
    from matplotlib.colors import ListedColormap
    base = matplotlib.colormaps["RdBu_r"]
    return ListedColormap(np.vstack([base(np.linspace(0.0, 0.33, 128)), base(np.linspace(0.67, 1.0, 128))]),
                          name="RdBu_r_nomid")


def legend_outside(ax, **kw) -> None:
    """Legend to the right of ``ax`` (constrained layout makes room for it)."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=False, **kw)


def save(fig, path: Path, methods: str, title: str | None = None, dpi: int = 110) -> Path:
    """Save the PNG and write its methods section into the folder's README.md (created / updated in place)."""
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, path, dpi=dpi)
    plt.close(fig)
    readme = path.parent / "README.md"
    head = f"## {path.name}"
    block = f"{head}\n\n{(title + chr(10) + chr(10)) if title else ''}{methods.strip()}\n"
    text = readme.read_text(encoding="utf-8") if readme.exists() else \
        (f"# {path.parent.name}\n\nFigures written by widefield_pipeline (FaceRhythm); one section per figure, "
         "regenerated with it. Do not edit by hand.\n")
    pat = re.compile(rf"^## {re.escape(path.name)}\n.*?(?=^## |\Z)", re.S | re.M)
    text = pat.sub(block + "\n", text) if pat.search(text) else text.rstrip() + "\n\n" + block
    readme.write_text(text, encoding="utf-8")
    return path
