"""`--help` prints the MODULE DOCSTRING, and on the rig it prints it to a cp1252 console.

`RawDescriptionHelpFormatter` sends the whole docstring to stdout, so one character outside the
Windows console codepage turns `--help` into a `UnicodeEncodeError` traceback -- on the box where a
new person is most likely to type it, and about a module that is otherwise fine. Caught 2026-09-22
when `dlc_train --help` died on a `→` in its own docstring; every other module in the repo was
already clean, which is why this is a convention worth pinning rather than a one-off fix.
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

import wfield_local

#: EVERY module with an argparse CLI, in BOTH trees. `scripts/` was missing and the gap was real:
#: `scripts.cd_migration --help` and `scripts.cd_cross_animal_figure --help` both raised
#: UnicodeEncodeError on 2026-09-25 because their module docstrings carried a Greek delta, and this
#: test -- which exists for exactly that failure -- could not see them. A guard that covers one of
#: two directories is a guard that will be trusted and be wrong.
_ROOTS = (Path(wfield_local.__file__).parent, Path(wfield_local.__file__).parent.parent / "scripts")
MODULES = sorted(
    p for root in _ROOTS for p in root.rglob("*.py")
    if root.exists() and "argparse" in p.read_text(encoding="utf-8")
)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_module_docstring_encodes_on_a_windows_console(path):
    # Parsing compiles the file, so a pre-existing W605 elsewhere in the repo would surface
    # here as a DeprecationWarning about a module this test is not about. `ruff` is the
    # gate for that; this test is only about the docstring's ENCODING.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    bad = sorted({c for c in doc if not _encodable(c)})
    assert not bad, (
        f"{path.name} docstring has {bad!r} (" +
        ", ".join(f"U+{ord(c):04X}" for c in bad) +
        ") which cp1252 cannot encode, so `--help` raises UnicodeEncodeError on the rig. "
        "Use an ASCII equivalent: -> for an arrow.")


def _encodable(c: str) -> bool:
    try:
        c.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


# ------------------------------------------------------------------------------------------
# The SECOND way `--help` dies, and the reason it is here rather than in a file of its own: it is
# the same failure -- a module that is otherwise fine, whose `--help` raises for someone trying to
# find out what it does.
#
# argparse %-EXPANDS every `help=` string (`_expand_help` does `help % params`), so a lone `%` in
# one is a format specification. `nightly_figs` carried "the grant render is ~18% of a run"; `% o`
# is a space-flagged octal conversion, and `nightly_figs --help` raised
# `TypeError: %o format: an integer is required, not dict` -- for every argument, not just that one,
# because formatting happens over the whole help text at once.
#
# It had been broken for as long as that sentence existed. The docstring test above could not catch
# it: the docstring encoded fine, and the break is in an ARGUMENT's help, expanded at render time.
# Found 2026-09-24 while adding `--skip-enl`.
#
# STATIC, not by rendering. Building a real parser means calling each module's `main()`, which
# parses `sys.argv` and in several modules touches the network on import of its dependencies. The
# AST pass reads the literal that argparse will be handed, which is where the bug lives.
# ------------------------------------------------------------------------------------------

def _help_literals(tree):
    """Every string literal passed as ``help=`` to an ``add_argument`` call."""
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg == "help" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                out.append((node.lineno, kw.value.value))
    return out


def _bad_percent(text: str):
    """Positions of a `%` argparse would treat as a conversion. `%%` and `%(name)s` are fine."""
    bad, i = [], 0
    while i < len(text):
        if text[i] != "%":
            i += 1
            continue
        if text[i + 1:i + 2] == "%":            # an escaped literal percent
            i += 2
            continue
        if text[i + 1:i + 2] == "(":            # argparse's own %(default)s style
            i += 2
            continue
        bad.append(i)
        i += 1
    return bad


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_argument_help_has_no_unescaped_percent(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = [(ln, t) for ln, t in _help_literals(tree) if _bad_percent(t)]
    assert not bad, "\n".join(
        f"{path.name}:{ln} help= has an unescaped '%': {t[:90]!r}" for ln, t in bad) + (
        "\n\nargparse %-expands help strings, so a lone '%' is a format spec and `--help` raises "
        "TypeError for the WHOLE parser. Write '%%' for a literal percent sign.")


def test_the_percent_check_catches_the_bug_it_was_written_for():
    """`~18% of a run` is the literal that broke `nightly_figs --help`."""
    assert _bad_percent("the grant render is ~18% of a run")
    assert not _bad_percent("the grant render is ~18%% of a run")
    assert not _bad_percent("defaults to %(default)s")
    assert not _bad_percent("no percent here at all")
