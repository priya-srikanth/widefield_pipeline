"""CLAUDE.md rule 11 (Priya 11-24): READ RESOLVERS ARE NEVER WRITE PATHS.

~15:13 on 11-24, `iti_where_scatter` / `iti_where_3axis` saved recomputed caches through `iti_licks.pred_file`, a
redirect-aware READ getter that resolved to a patch folder, and overwrote it. This static check fails when a write
call receives a path built by a read getter, directly (``df.to_csv(il.pred_file("x.csv"))``) or through a local name
assigned from one in the same function (``f = il.in_path_x("x.csv"); ...; np.save(f, a)``). Writers use an explicit
out-path: ``il.pred_file_out`` / ``il.OUT_X`` / ``il.out_x()``.

Read getters: `READ_GETTERS`. Write calls: DataFrame.to_* writers, np.save*, savetxt, savefig, save_png_svg,
open(..., "w"/"a"/"x"/"+"), Path.write_text / write_bytes / mkdir / touch / open("w"), pickle.dump / joblib.dump,
shutil copies and moves (destination), os.replace / os.rename (destination).

A deliberate read-modify-write of the SAME file through a getter is still a write through a read path: give it an
out-path. `ALLOWED` lists reviewed exceptions as (file, line text) with the reason; keep it empty if you can.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READ_GETTERS = {"pred_file", "in_path_x", "in_path", "resolve_cache", "in_path_w", "read_path"}
_FRAME_WRITERS = {"to_csv", "to_parquet", "to_pickle", "to_json", "to_excel", "to_feather", "to_hdf", "to_stata"}
_NP_WRITERS = {"save", "savez", "savez_compressed", "savetxt"}
_PATH_WRITERS = {"write_text", "write_bytes", "mkdir", "touch"}
#: (relative file, stripped source line) -> reason. Reviewed exceptions only.
ALLOWED: dict = {}


def _calls_getter(node: ast.AST, tainted: set) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if name in READ_GETTERS:
                return True
        if isinstance(n, ast.Name) and n.id in tainted:
            return True
    return False


def _mode_writes(call: ast.Call, pos: int) -> bool:
    m = call.args[pos] if len(call.args) > pos else next((k.value for k in call.keywords if k.arg == "mode"), None)
    return isinstance(m, ast.Constant) and isinstance(m.value, str) and any(c in m.value for c in "wax+")


def _write_targets(call: ast.Call) -> list[ast.AST]:
    """The path expression(s) a call writes to ([] = not a write call)."""
    f = call.func
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    if isinstance(f, ast.Attribute):
        a = f.attr
        if a in _FRAME_WRITERS:
            return [call.args[0]] if call.args else [kw[k] for k in ("path_or_buf", "path") if k in kw]
        if a in _NP_WRITERS and isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy"):
            return [call.args[0]] if call.args else [kw["file"]] if "file" in kw else []
        if a == "savefig":
            return [call.args[0]] if call.args else [kw["fname"]] if "fname" in kw else []
        if a in _PATH_WRITERS:
            return [f.value]
        if a == "open" and _mode_writes(call, 0):                       # Path(...).open("w")
            return [f.value]
        if a in ("dump",) and isinstance(f.value, ast.Name) and f.value.id in ("pickle", "joblib") and len(call.args) > 1:
            return [call.args[1]]
        if a in ("copy", "copy2", "copyfile", "copytree", "move") and isinstance(f.value, ast.Name) \
                and f.value.id == "shutil" and len(call.args) > 1:
            return [call.args[1]]
        if a in ("replace", "rename") and isinstance(f.value, ast.Name) and f.value.id == "os" and len(call.args) > 1:
            return [call.args[1]]
        if a == "save_png_svg" and len(call.args) > 1:
            return [call.args[1]]
    elif isinstance(f, ast.Name):
        if f.id == "open" and call.args and _mode_writes(call, 1):
            return [call.args[0]]
        if f.id == "save_png_svg" and len(call.args) > 1:
            return [call.args[1]]
    return []


def _scan_scope(body_nodes, rel: str, out: list) -> None:
    tainted: set = set()
    for stmt in body_nodes:
        for n in ast.walk(stmt):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not stmt:
                continue
            if isinstance(n, ast.Assign) and _calls_getter(n.value, set()):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        tainted.add(t.id)
            elif isinstance(n, ast.Assign):
                for t in n.targets:                                      # re-assigned from a clean value: untaint
                    if isinstance(t, ast.Name) and t.id in tainted and not _calls_getter(n.value, tainted):
                        tainted.discard(t.id)
            if isinstance(n, ast.Call):
                for tgt in _write_targets(n):
                    if _calls_getter(tgt, tainted):
                        out.append((rel, n.lineno))


def violations(paths=None) -> list[tuple[str, int, str]]:
    files = paths or [*sorted((ROOT / "scripts").rglob("*.py")), *sorted((ROOT / "wfield_local").rglob("*.py"))]
    found = []
    for f in files:
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        rel = f.relative_to(ROOT).as_posix() if f.is_relative_to(ROOT) else str(f)
        hits: list = []
        funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for fn in funcs:
            _scan_scope(fn.body, rel, hits)
        _scan_scope([s for s in tree.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                   ast.ClassDef))], rel, hits)
        lines = src.splitlines()
        for r, ln in sorted(set(hits)):
            text = lines[ln - 1].strip()
            if (r, text) not in ALLOWED:
                found.append((r, ln, text))
    return found


def test_no_write_through_a_read_resolver():
    bad = violations()
    assert not bad, ("writes through a READ resolver (CLAUDE.md rule 11; use pred_file_out / OUT_X):\n"
                     + "\n".join(f"  {r}:{ln}: {t}" for r, ln, t in bad))


def test_the_checker_catches_the_11_24_shapes(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text(
        "import numpy as np\n"
        "def a(R, il):\n"
        "    R.to_csv(il.pred_file('where_preds.csv'), index=False)\n"
        "def b(il, x):\n"
        "    p = il.in_path_x('t.npy')\n"
        "    np.save(p, x)\n"
        "def c(il, fig, save_png_svg):\n"
        "    save_png_svg(fig, il.in_path('figures') / 'f.png')\n"
        "    with open(il.in_path('x.txt'), 'w') as fh:\n"
        "        fh.write('x')\n"
        "def ok(il, R):\n"
        "    R.to_csv(il.pred_file_out('where_preds.csv'))\n"
        "    T = R.read_csv(il.pred_file('where_preds.csv'))\n"
        "    p = il.in_path('a.csv')\n"
        "    p = il.OUT_X / 'a.csv'\n"
        "    T.to_csv(p)\n", encoding="utf-8")
    got = sorted(ln for _r, ln, _t in violations([f]))
    assert got == [3, 6, 8, 9], got
