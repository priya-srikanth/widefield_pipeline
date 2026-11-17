"""Importing our modules that pull in the third-party `wfield` package must leave matplotlib.rcParams unchanged.

`wfield/viz.py` runs ``style.use(mplstyledict)`` at import (Arial, 12 pt, savefig.dpi 500, ...); before 11-09 that
restyled every later figure of any process importing e.g. `wfield_local.cross_day_align`. Every import now goes
through `wfield_local.wfield_import.no_style_leak` (Priya 11-09). Each check runs in a FRESH interpreter: inside
pytest, wfield may already be in sys.modules, and then an import would prove nothing.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_PROBE = r"""
import sys, json, matplotlib
matplotlib.use("Agg")
before = {k: repr(v) for k, v in matplotlib.rcParams.items() if k != "backend"}
assert "wfield" not in sys.modules
{body}
assert "wfield" in sys.modules, "probe did not import wfield"
after = {k: repr(v) for k, v in matplotlib.rcParams.items() if k != "backend"}
print(json.dumps(sorted(k for k in before if before[k] != after.get(k))))
"""

#: every module of ours that imports wfield at MODULE level (keep in sync: test_every_wfield_import_is_wrapped)
TOP_LEVEL = ["allen_register", "apply_allen_transform", "cross_day_align", "plot_alignment_before_after",
             "plot_alignment_landmark_overlays", "plot_allen_reference_landmarks", "motion_correct_fixed",
             "run_wfield_motion", "run_wfield_local"]


def _changed(body: str) -> list:
    import json
    r = subprocess.run([sys.executable, "-c", _PROBE.replace("{body}", body)], capture_output=True, text=True,
                       cwd=str(REPO), timeout=300)
    if r.returncode != 0 and ("No module named 'cv2'" in r.stderr or "No module named 'wfield'" in r.stderr):
        pytest.skip(r.stderr.strip().splitlines()[-1])
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_bare_wfield_import_does_leak():
    """The control: without the helper, importing wfield DOES change rcParams (else the tests below are vacuous)."""
    assert "font.size" in _changed("import wfield.allen")


@pytest.mark.parametrize("mod", TOP_LEVEL)
def test_our_module_import_leaves_rcparams_unchanged(mod):
    assert _changed(f"import wfield_local.{mod}") == []


def test_import_wfield_helper_and_area_basis_path():
    """`import_wfield` itself, and the run-time path `area_basis.roi_masks` takes (allen_register constants)."""
    assert _changed("from wfield_local.wfield_import import import_wfield\nimport_wfield('wfield.local_nmf')\n"
                    "from wfield_local.allen_register import BREGMA_OFFSET, RESOLUTION") == []


def _wfield_imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "wfield" or a.name.startswith("wfield."):
                    yield node
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and (
                node.module == "wfield" or node.module.startswith("wfield.")):
            yield node


def test_every_wfield_import_is_wrapped():
    """Static guard: every `import wfield...` / `from wfield... import` in wfield_local/, scripts/ and tests/ sits
    inside a ``with no_style_leak():`` block (string imports go through `import_wfield`)."""
    bad = []
    for d in ("wfield_local", "scripts", "tests"):
        for p in (REPO / d).rglob("*.py"):
            if "archive" in p.parts:
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            wrapped = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.With) and any(
                        isinstance(i.context_expr, ast.Call) and getattr(i.context_expr.func, "id", "") == "no_style_leak"
                        for i in node.items):
                    wrapped |= {id(n) for n in ast.walk(node)}
            bad += [f"{p.relative_to(REPO)}:{n.lineno}" for n in _wfield_imports(tree) if id(n) not in wrapped]
    assert not bad, f"unwrapped wfield imports (use wfield_local.wfield_import): {bad}"
