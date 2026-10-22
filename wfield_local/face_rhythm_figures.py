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


def legend_outside(ax, **kw) -> None:
    """Legend to the right of ``ax`` (constrained layout makes room for it)."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=False, **kw)


def save(fig, path: Path, methods: str, title: str | None = None, dpi: int = 110) -> Path:
    """Save the PNG and write its methods section into the folder's README.md (created / updated in place)."""
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
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
