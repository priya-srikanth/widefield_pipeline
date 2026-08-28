"""A `__main__` guard must come after every function it can reach.

FOUND 2026-08-28: `locanmf_frozen_decoder` ran `main()` from a guard at line 899 while
`write_animal_confusion_grid` was defined at 911. `python -m wfield_local.locanmf_frozen_decoder
--loso` therefore died with NameError -- after writing the decoder JSON and before the encoder ran,
leaving a half-finished output set.

It survived because nothing exercised the path: `nightly_figs` IMPORTS these modules and calls their
functions in-process, which executes the whole body first. Only the CLI was broken, and the CLI is
what a by-hand re-run uses. A module-scope check is cheap and catches the whole class.
"""
import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "wfield_local"
MODULES = sorted(p for p in PKG.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_toplevel_def_after_the_main_guard(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guard = None
    for node in tree.body:
        if isinstance(node, ast.If):
            t = node.test
            if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                    and t.left.id == "__name__"):
                guard = node.lineno
    if guard is None:
        pytest.skip("no __main__ guard")
    late = [n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.lineno > guard]
    assert not late, (
        f"{path.name}: {late} defined AFTER the __main__ guard at line {guard}. Running this module "
        f"as a script executes the guard before those definitions exist, so any of them reached "
        f"from main() raises NameError -- while importing the module works, which is how this hides")
