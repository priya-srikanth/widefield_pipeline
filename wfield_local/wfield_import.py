"""Import the third-party `wfield` package WITHOUT its matplotlib style.

`wfield/__init__.py` imports `wfield.viz`, which runs ``plt.matplotlib.style.use(mplstyledict)`` at import time
(Arial, font 12, tick labels 8, savefig.dpi 500, a black-first color cycle, ...). Any `from wfield.<sub> import ...`
triggers it, so before this helper any process that imported e.g. `wfield_local.cross_day_align` or
`wfield_local.allen_register` (pulled in at run time by `area_basis.roi_masks` for the v8 ROI rim) drew every later
figure in wfield's style instead of ours (Priya 11-09: "stop it, use our own figure styling").

Every import of `wfield` / `wfield.*` in this repo goes through `no_style_leak()` (or `import_wfield`):

    from wfield_local.wfield_import import no_style_leak
    with no_style_leak():
        from wfield.allen import load_allen_landmarks

The import runs inside `matplotlib.rc_context()`, so rcParams are restored on exit (the backend is excluded by
matplotlib itself). wfield is cached in `sys.modules` after the first import, so later imports never re-run
`style.use`. Never edit site-packages to fix this. Test: tests/test_wfield_style_leak.py.
"""
from __future__ import annotations

import contextlib
import importlib


@contextlib.contextmanager
def no_style_leak():
    """Run the enclosed `wfield` import(s) with matplotlib.rcParams saved and restored afterwards."""
    import matplotlib
    with matplotlib.rc_context():
        yield


def import_wfield(name: str = "wfield"):
    """`importlib.import_module(name)` for a `wfield` (sub)module, inside `no_style_leak()`."""
    with no_style_leak():
        return importlib.import_module(name)
