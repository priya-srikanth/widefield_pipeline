"""PRINTED REPORTS MUST BE ASCII. The Windows console is cp1252 and kills the step otherwise.

This is the third time in one session that a Greek delta reached a `print` and raised
UnicodeEncodeError -- not mangling a character, ABORTING the analysis step mid-run, after the
expensive part had already been paid for. `tests/test_cli_help_is_console_safe.py` guards argparse
help strings for the same reason; this guards the report builders, which are the other thing these
modules print.

FIGURES ARE NOT COVERED and must not be: matplotlib renders to a file, not to the console, so a
figure keeps its typography. The rule is about what goes to stdout.
"""
from __future__ import annotations

import importlib

import pytest

#: (module, callable name) for every report builder that is printed by a CLI.
REPORTS = [
    ("scripts.cd_migration", "report"),
    ("scripts.cd_cross_animal_figure", "table"),
    ("scripts.cd_geometry_figure", "table"),
    ("wfield_local.cd_overlap_null", "report"),
]


@pytest.mark.parametrize(("mod_name", "fn_name"), REPORTS, ids=lambda v: str(v).split(".")[-1])
def test_report_source_has_no_non_ascii_in_printed_strings(mod_name, fn_name):
    """The STRING LITERALS the report builder assembles must be encodable as cp1252.

    Checked on the source rather than by calling it, because calling needs real data: a report
    builder is fed dumps off the share and cannot run in a unit test. The literals are what carry
    the risk -- every failure so far has been a typographic character typed into a format string.
    """
    import ast
    import inspect
    import textwrap

    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    # THE FUNCTION'S OWN DOCSTRING IS EXCLUDED: it is read in an editor, never printed, so it may
    # keep its typography. MODULE docstrings are a different matter -- argparse prints those as the
    # CLI description -- and `test_cli_help_is_console_safe` owns that check for both trees.
    doc = ast.get_docstring(tree.body[0], clean=False) if tree.body else None
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if doc is not None and node.value == doc:
                continue
            try:
                node.value.encode("cp1252")
            except UnicodeEncodeError:
                offenders = {c for c in node.value if ord(c) > 127}
                bad.append((node.lineno, "".join(sorted(offenders)), node.value[:70]))
    assert not bad, (
        f"{mod_name}.{fn_name} builds text containing characters the cp1252 console cannot "
        f"encode:\n"
        + "\n".join(f"  line {ln}: {chars!r} in {txt!r}" for ln, chars, txt in bad)
        + "\n\nprint() to a cp1252 console raises UnicodeEncodeError and ABORTS the step. Use ASCII "
          "in anything printed; figures may keep their typography.")
