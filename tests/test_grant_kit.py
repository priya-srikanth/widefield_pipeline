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


def test_every_name_that_moved_is_still_reachable_on_grant_figures():
    """The re-export must not shrink to "what grant_figures still calls".

    THAT IS EXACTLY HOW IT BROKE. The list was first built from the names `grant_figures` still
    references, which dropped `_runs_to_blocks`, `_delta_diag_ci` and `_delta_diag_one` -- used
    only by their kit siblings now -- and three test modules that read them off `grant_figures`
    failed. Tests and sibling scripts address these by their historical home, so moving one is a
    rename of a public-ish name whether or not it starts with an underscore.

    Pinned against the PRE-SPLIT revision, so the property is "nothing became unreachable", not a
    hand-maintained list that drifts.
    """
    import ast
    import subprocess

    from wfield_local import grant_figures

    old = subprocess.run(["git", "show", "ae7b70a:wfield_local/grant_figures.py"],
                         capture_output=True, text=True, check=False,
                         cwd=Path(__file__).resolve().parents[1])
    if old.returncode != 0:
        import pytest
        pytest.skip("pre-split revision not in this checkout")
    tree = ast.parse(old.stdout)
    was = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            was |= {x.id for x in n.targets if isinstance(x, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            was.add(n.target.id)

    # THE TWO RESTRICTION GLOBALS ARE EXEMPT, AND MUST STAY EXEMPT. Re-exporting them would bind
    # `grant_figures._ONLY_WINDOW` to whatever the kit held at import time and then never update
    # it -- a stale copy that reads like the real thing, which is the precise bug the setter
    # exists to prevent. Reach them through `grant_kit.only()`.
    EXEMPT = {"_ONLY_WINDOW", "_ONLY_VARIANT"}
    gone = sorted(n for n in was
                  if not hasattr(grant_figures, n) and not n.startswith("__") and n not in EXEMPT)
    assert not gone, (
        f"{gone} were defined in grant_figures before the split and are now reachable from "
        f"neither it nor its kit re-export")
    for n in EXEMPT:
        assert not hasattr(grant_figures, n), (
            f"{n} is re-exported onto grant_figures: that is a SNAPSHOT taken at import time, "
            f"not a view of the kit's value, and it will silently go stale the moment "
            f"set_only() is called. Use grant_kit.only().")
