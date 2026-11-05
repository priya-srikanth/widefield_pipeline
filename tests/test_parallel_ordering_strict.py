"""Stricter parallel-ordering checks (T10, third review), AST-based, beside tests/test_parallel_results_are_ordered.py.

1. Every raw `fan_out` call (any alias: `fan_out`, `parallel.fan_out`, `from ... import fan_out as fo`) whose result is
   assigned must SORT THAT RESULT VARIABLE (`sorted(<var>` / `<var>.sort(`) within the next 14 lines; `fan_sessions`
   sorts internally. Docstrings are never scanned (AST, not text). `as_completed(` must be followed by a sort too.
2. A module-level name that main() -- or a function main() calls directly -- MUTATES (a `global` rebind, a subscript or
   attribute store) must not be READ by a worker function passed to fan_out / fan_sessions / map_collect.collect,
   unless that worker re-applies it (its own `global` rebind). Spawned workers re-import the module and never see it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 14
FAN = ("fan_out",)
WORKER_CALLS = ("fan_out", "fan_sessions", "collect")


def _files():
    for base in ("wfield_local", "scripts"):
        for p in sorted((ROOT / base).rglob("*.py")):
            if "__pycache__" in p.parts or p.name in ("parallel.py", "analysis_kit.py") or "archive" in p.parts:
                continue
            yield p


def _fan_aliases(tree) -> set:
    names = set(FAN)
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name in FAN and a.asname:
                    names.add(a.asname)
    return names


def _call_name(c: ast.Call) -> str | None:
    f = c.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def _unsorted_sites():
    bad = []
    for p in _files():
        src = p.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        alias = _fan_aliases(tree)
        for n in ast.walk(tree):
            if not isinstance(n, ast.Assign) or not isinstance(n.value, ast.Call):
                continue
            nm = _call_name(n.value)
            if nm not in alias and nm != "as_completed":
                continue
            tg = n.targets[0]
            var = tg.elts[0] if isinstance(tg, ast.Tuple) and tg.elts else tg
            if not isinstance(var, ast.Name):
                continue
            window = "\n".join(lines[n.lineno - 1: n.end_lineno + WINDOW])
            v = var.id
            if f"sorted({v}" not in window and f"{v}.sort(" not in window:
                bad.append(f"{p.relative_to(ROOT).as_posix()}:{n.lineno} ({v})")
    return bad


def test_every_fan_out_result_variable_is_sorted():
    assert not _unsorted_sites(), _unsorted_sites()


def _mutated_by(fn: ast.FunctionDef, module_names: set) -> set:
    out = set()
    for g in ast.walk(fn):
        if isinstance(g, ast.Global):
            out |= set(g.names) & module_names
        if isinstance(g, (ast.Assign, ast.AugAssign)):
            for t in (g.targets if isinstance(g, ast.Assign) else [g.target]):
                base = t
                while isinstance(base, (ast.Subscript, ast.Attribute)):
                    base = base.value
                if base is not t and isinstance(base, ast.Name) and base.id in module_names:
                    out.add(base.id)
    return out


def _runtime_state_cases():
    cases = []
    for p in _files():
        src = p.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        funcs = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
        main = funcs.get("main")
        if main is None:
            continue
        module_names = {t.id for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                        for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
        callees = {_call_name(c) for c in ast.walk(main) if isinstance(c, ast.Call)} & set(funcs)
        mutated = _mutated_by(main, module_names)
        for c in callees:
            mutated |= _mutated_by(funcs[c], module_names)
        if not mutated:
            continue
        workers = set()
        for c in ast.walk(tree):
            if isinstance(c, ast.Call) and _call_name(c) in WORKER_CALLS and len(c.args) >= 2:
                w = c.args[1]
                if isinstance(w, ast.Name) and w.id in funcs:
                    workers.add(w.id)
        # a mutator ALSO called at module level runs at import, i.e. in every spawned worker too: not a leak
        at_import = {_call_name(n.value) for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)}
        at_import_names = set().union(*[_mutated_by(funcs[c], module_names) for c in at_import if c in funcs]) \
            if at_import else set()
        for w in workers:
            reads = {n.id for n in ast.walk(funcs[w]) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            w_calls = {_call_name(c) for c in ast.walk(funcs[w]) if isinstance(c, ast.Call)} & set(funcs)
            reapplied = {n for g in ast.walk(funcs[w]) if isinstance(g, ast.Global) for n in g.names} | at_import_names
            for c in w_calls:
                reapplied |= _mutated_by(funcs[c], module_names)
            leak = (mutated & reads) - reapplied
            if leak:
                cases.append(f"{p.relative_to(ROOT).as_posix()}: worker {w} reads {sorted(leak)} set at runtime by main")
    return cases


def test_no_runtime_state_reaches_a_worker_unseen():
    assert not _runtime_state_cases(), _runtime_state_cases()


@pytest.mark.parametrize("bad", ["""
from wfield_local.parallel import fan_out as fo
def main():
    res, fail = fo(items, w)
    other = sorted(items)
    use(res)
""", '''
"""docstring: res, fail = fan_out(items, w) then sorted(res"""
def main():
    res, fail = fan_out(items, w)
    use(res)
'''])
def test_the_detector_catches_aliases_and_unrelated_sorts(bad, tmp_path, monkeypatch):
    f = tmp_path / "scripts" / "x.py"
    f.parent.mkdir()
    f.write_text(bad)
    monkeypatch.setattr(__import__(__name__), "ROOT", tmp_path)
    assert _unsorted_sites()
