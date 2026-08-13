"""No module may carry a machine-specific absolute path as a RUNTIME default.

This bug class has bitten the pipeline four times, each time silently and each time found only by its
symptoms:

  * behavior_position.BEH_ROOT = "M:/MICROSCOPE/..."  -> on any other machine the dead-spout_bit1
    repair never fired, leaving 4-of-6 position labels and ~1/3 of cues mislabelled in every cached
    decode/encode/RDM result (session_cache v2).
  * session_cache CACHE_DIR = "C:/Users/sabatini/..." -> a second cache under a nonexistent user.
  * nightly_figs.DEFAULT_OUT = "C:/Users/sabatini/..." -> every figure step raised FileNotFoundError,
    the deck built with 0 figures / 287 missing, and the run still exited 0.
  * archive_day.DEFAULTS m_raw = "M:/collaborations/Priya/..." -> the standby destination of a tool
    that COPIES THEN DELETES did not exist on the helper box, which maps standby at Priya level.

They all share one shape: a path literal that happens to be correct on the machine it was written on.
Mounts belong in configs/paths.yaml and must be resolved through PathResolver, so a machine that lacks
a root fails loudly instead of writing somewhere that is not its own.

Docstrings, comments and CLI usage EXAMPLES are fine -- they document, they do not execute. What is
checked here is executable code: module-level assignments and argparse defaults.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

WFIELD = Path(__file__).resolve().parents[1] / "wfield_local"

# A Windows drive-letter path, or a user-home path, appearing in executable code.
DRIVE = re.compile(r"^[A-Za-z]:[/\\]")

# Modules allowed to hold a literal, with the reason. Keep this list SHORT and justified.
ALLOWED = {
    "paths.py",            # the resolver itself: _SIGNATURE is how a machine is auto-detected
    "session_cache.py",    # last-resort fallback when config is unavailable (e.g. in tests)
    "joint_locanmf.py",    # ditto
    "nightly_figs.py",     # ditto
    "allen_register.py",   # guarded by an os.path.isdir() probe with a ~ fallback
}


def _string_literals_in_code(path: Path):
    """(lineno, value) for string literals that are NOT docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
               and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value


def test_no_machine_specific_paths_in_executable_code():
    offenders = []
    for py in sorted(WFIELD.glob("*.py")):
        if py.name in ALLOWED:
            continue
        for lineno, val in _string_literals_in_code(py):
            if DRIVE.match(val) or val.startswith("~/") or "Users/sabatini" in val:
                # a bare drive root inside a longer help/usage string is documentation, not a default;
                # require the literal to BE a path, not merely contain one
                if len(val) < 4 or " " in val.strip():
                    continue
                offenders.append(f"{py.name}:{lineno}: {val!r}")
    assert not offenders, (
        "machine-specific path literal in executable code — add a logical root to configs/paths.yaml "
        "and resolve it via config.resolver().root(...):\n  " + "\n  ".join(offenders))
