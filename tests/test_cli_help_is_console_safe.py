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

MODULES = sorted(
    p for p in Path(wfield_local.__file__).parent.glob("*.py")
    if "argparse" in p.read_text(encoding="utf-8")
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
