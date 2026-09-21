"""THE RESTRICTION GLOBALS MUST BE SET WHERE THEY ARE READ.

`_ONLY_WINDOW` / `_ONLY_VARIANT` gate which alignment and trial class a render process produces.
They were module globals in `grant_figures`, assigned with `global` from `main` and
`_render_unit`, and read by `_windows` / `_variants`. Those readers moved to `grant_kit` on
2026-09-21.

**A module global assigned in one module and read in another is two different variables.** Had the
`global` statements stayed behind, they would have bound names in `grant_figures` that nothing
reads, `grant_kit._ONLY_WINDOW` would have stayed `None`, and every worker would have rendered
EVERY alignment instead of the single unit it was handed. Nothing raises. The render still
"succeeds"; it is just wrong, and roughly three times slower, and two units end up writing the
same file -- which `_run_parallel`'s collision check would have reported as a mysterious bug
somewhere else entirely.

So this file asserts the wiring, not the arithmetic: that the setter is the only route, that the
driver uses it, and that no `global _ONLY_*` has come back.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from wfield_local import grant_kit


@pytest.fixture(autouse=True)
def _reset():
    """Leave the process unrestricted. These are module state; a leak would skew another test."""
    yield
    grant_kit.set_only()


def test_the_setter_actually_reaches_the_readers():
    """The whole point: `set_only` must change what `_windows` / `_variants` return."""
    grant_kit.set_only()
    assert len(grant_kit._windows()) == 3, "unrestricted must give every alignment"
    assert grant_kit._variants("cue") == ("lick", "working")

    grant_kit.set_only("cue", "lick")
    assert [w[1] for w in grant_kit._windows()] == ["cue"]
    assert grant_kit._variants("cue") == ("lick",)
    assert grant_kit.only() == ("cue", "lick")

    grant_kit.set_only()
    assert len(grant_kit._windows()) == 3, "set_only() with no arguments must clear the restriction"


def test_the_lick_window_still_admits_only_the_lick_class():
    """Unrestricted, and independent of the split: a trial with no detected lick has no lick to
    align to, so a 'miss trial, lick-aligned' panel is undefined rather than weak."""
    grant_kit.set_only()
    assert grant_kit._variants("lick") == ("lick",)


def test_the_driver_sets_them_through_grant_kit_not_by_global():
    """Source-level, because the failure is silent -- a `global` here binds a name nobody reads.

    Checked on the AST rather than by substring so a `global` inside a docstring or a comment
    cannot satisfy it, and so this keeps working if the functions are renamed.
    """
    from wfield_local import grant_figures

    src = Path(inspect.getfile(grant_figures)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    bad = [f"{n.name}: global {', '.join(g.names)}"
           for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
           for g in ast.walk(n) if isinstance(g, ast.Global)
           if any(x.startswith("_ONLY_") for x in g.names)]
    assert not bad, (
        f"{bad} -- these are read in grant_kit, so a `global` here binds a SECOND variable that "
        f"nothing reads and every worker renders every alignment")

    for fn in (grant_figures._render_unit, grant_figures.main):
        assert "set_only(" in inspect.getsource(fn), (
            f"{fn.__name__} must restrict the process through grant_kit.set_only")


def test_the_kit_does_not_reach_back_into_grant_figures():
    """It is the SHARED layer. An import back up would make the split circular and meaningless.

    ON THE IMPORT STATEMENTS, not on the text. The first version of this grepped the source for
    `grant_figures` and failed on the module docstring, which says where the code came from --
    a guard that cannot tell a dependency from a sentence about one.
    """
    tree = ast.parse(Path(inspect.getfile(grant_kit)).read_text(encoding="utf-8"))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add(n.module or "")
    assert not any("grant_figures" in m for m in imported), (
        f"grant_kit must not depend on grant_figures; it imports {sorted(imported)}")
