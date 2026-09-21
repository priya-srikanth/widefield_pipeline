"""Nothing may consume a `fan_out` result in completion order, and a frozen model's identity
must not depend on how its training set was handed over.

WHY THESE TWO THINGS TOGETHER. They are the same failure wearing different clothes: a container
whose order is incidental decides something that ought to be a property of the data. It has now
bitten this repo three times --

  1. `fan_out` completion order feeding a bootstrap pool (2026-09-20). The point estimate stayed
     exact at -18.7 licks/min while the CI moved from [-22.9, -13.7] to [-23.1, -13.6].
  2. `for lab in {set comprehension}` in `rest_coupling._contrast_by_animal` (2026-09-21). Python
     randomises string hashing per process, so the module disagreed with ITSELF between two runs
     of identical code over identical data.
  3. `animals = list(results)` in the frozen decoder's summary figures, and three JSON artefacts
     written from completion-ordered dicts (2026-09-21). No number moved; the FIGURE and the FILE
     BYTES did, which defeats the point of a frozen model -- you freeze it so you can diff it.

None of the three was caught by a test, and none would have been caught by a test that checked
values, because in two of the three the values were right. CLAUDE.md ground rule 9.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

#: How many lines after a `fan_out(` call the ordering has to be established in. Every real
#: consumer does it within a few lines of the call; a longer window would let a sort somewhere
#: else in the function count as covering this one.
WINDOW = 14

#: Call sites that legitimately do NOT sort, each with the reason. Empty on purpose -- add an
#: entry only when the result genuinely cannot reach an output, and say why here rather than in a
#: comment at the call site, so this list is the one place to read.
ALLOWED: dict[str, str] = {}


def _fan_out_sites():
    """``(path, lineno, source_window)`` for every fan-out call in the tree.

    BOTH SPELLINGS. `fan_sessions` is `fan_out` plus the sort, and matching only the raw
    `fan_out(` made this file quietly stop covering every module converted onto the kit -- which
    is how a guard turns into decoration. It showed up as the runtime-global test reporting
    SKIPPED rather than passing: **an empty parametrize list is not a pass.** `parallel.py` and
    `analysis_kit.py` are excluded because they DEFINE the two wrappers.
    """
    out = []
    for p in sorted(list((ROOT / "wfield_local").rglob("*.py"))
                    + list((ROOT / "scripts").rglob("*.py"))):
        if "__pycache__" in p.parts or p.name in ("parallel.py", "analysis_kit.py"):
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, ln in enumerate(lines):
            if (re.search(r"\b(fan_out|fan_sessions)\s*\(", ln)
                    and not ln.lstrip().startswith("#")):
                out.append((p.relative_to(ROOT).as_posix(), i + 1,
                            "\n".join(lines[i:i + WINDOW])))
    return out


def test_there_are_fan_out_sites_to_check():
    """A guard on the guard: a rename of `fan_out` would otherwise make this file vacuously pass."""
    assert len(_fan_out_sites()) >= 12


@pytest.mark.parametrize("site", _fan_out_sites(), ids=lambda s: f"{s[0]}:{s[1]}")
def test_every_fan_out_result_is_ordered_before_use(site):
    path, lineno, window = site
    key = f"{path}:{lineno}"
    if key in ALLOWED:
        pytest.skip(ALLOWED[key])
    # `fan_sessions` sorts internally (and `input_order` is a sort key), so either the call
    # goes through the kit or the caller sorts for itself.
    assert "sorted(" in window or "fan_sessions(" in window, (
        f"{key} consumes a `fan_out` result without sorting it within {WINDOW} lines.\n"
        "`fan_out` returns units in COMPLETION order, so whatever is built here is built in a "
        "different order on every run. If it feeds a seeded RNG the numbers move; if it feeds a "
        "figure or a JSON dump the bytes move. Use `analysis_kit.fan_sessions`, which sorts, or "
        "sort at the point of collection.\n\n" + window)


def test_the_kit_sorts_and_says_so():
    """`fan_sessions` is the intended answer, so it must actually contain the sort."""
    src = (ROOT / "wfield_local" / "analysis_kit.py").read_text(encoding="utf-8")
    fn = next(f for f in ast.parse(src).body
              if isinstance(f, ast.FunctionDef) and f.name == "fan_sessions")
    assert "sorted(" in ast.get_source_segment(src, fn)


# --------------------------------------------------------------------------------------------
# A frozen model's IDENTITY
# --------------------------------------------------------------------------------------------

def test_spec_id_does_not_depend_on_how_the_training_set_was_passed():
    """A list, the same list reversed, and a set must all name the SAME frozen model.

    `make_spec` sorts `train_labels` and `spec_id` hashes with `sort_keys=True`. If either were
    dropped, a pooled fit whose label list arrived in a different order would miss the store, refit,
    and publish a SECOND model for the same training data under a different id -- two frozen
    references where the repo believes there is one.
    """
    from wfield_local import frozen_models as fm

    labs = ["PS92_0812", "PS92_0808", "PS92_0814"]
    kw = dict(align="cue", source="roi")
    a = fm.make_spec("PS92", "decoder", train_labels=labs, **kw)
    b = fm.make_spec("PS92", "decoder", train_labels=list(reversed(labs)), **kw)
    c = fm.make_spec("PS92", "decoder", train_labels=set(labs), **kw)
    assert fm.spec_id(a) == fm.spec_id(b) == fm.spec_id(c)
    assert a["train_labels"] == sorted(labs)
    assert list(a["train_sigs"]) == sorted(a["train_sigs"])


def test_spec_id_is_not_a_python_hash():
    """`spec_id` must survive a process restart, which `hash()` does not.

    Checked structurally rather than by spawning an interpreter: the id is a sha1 over a
    `sort_keys=True` JSON dump, and neither ingredient is affected by `PYTHONHASHSEED`.
    """
    import inspect

    from wfield_local import frozen_models as fm

    src = inspect.getsource(fm.spec_id)
    assert "sha1" in src and "sort_keys=True" in src
    assert re.search(r"\bhash\s*\(", src) is None


def test_the_frozen_estimators_are_deterministic_across_blas_threads():
    """The decoder's `lbfgs` and the encoder's `Ridge` must give BITWISE identical coefficients
    however many threads BLAS is using.

    This is what makes "frozen" mean anything on a box where the local model store may be empty:
    only two models are published on MICROSCOPE, so most runs REFIT rather than hit the cache, and
    the guarantee then rests on the fit itself. `pin_blas` fixes the thread count for rendering
    (`test_render_is_machine_independent`); this asserts the fits do not need it.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from wfield_local import parallel

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 32))
    y = rng.integers(0, 6, 300)
    P = np.eye(6)[y]

    seen = set()
    for threads in (1, 2, 4):
        parallel.pin_blas(threads)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=0.5)).fit(X, y)
        seen.add((clf[-1].coef_.tobytes(), Ridge(alpha=1.0).fit(P, X).coef_.tobytes()))
    parallel.pin_blas()
    assert len(seen) == 1, "the frozen estimators changed with the BLAS thread count"


# --------------------------------------------------------------------------------------------
# The OTHER spawn trap: a module global set at runtime does not reach a worker
# --------------------------------------------------------------------------------------------

def _modules_that_fan_out_and_mutate_a_global():
    """``(path, {global names assigned inside main})`` for every module that does both."""
    out = []
    for path, _lineno, _win in _fan_out_sites():
        src = (ROOT / path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
        main = next((f for f in tree.body
                     if isinstance(f, ast.FunctionDef) and f.name == "main"), None)
        if main is None:
            continue
        names = {n for g in ast.walk(main) if isinstance(g, ast.Global) for n in g.names}
        if names:
            out.append((path, names, tree))
    return out


@pytest.mark.parametrize("case", _modules_that_fan_out_and_mutate_a_global(),
                         ids=lambda c: c[0])
def test_a_runtime_global_reaches_the_workers(case):
    """A global that `main` sets is invisible to every spawned worker, and the failure is SILENT.

    `channel_position_maps --late` is the live example: it sets `WIN_START_S = LATE_START_S` as a
    module global, deliberately, because the window is consumed four call levels down inside
    `_win_avg_base`. `fan_out` spawns, so each child re-imports the module with `WIN_START_S` back
    at 0.0 -- fanning that loop out without re-applying it in the worker would have written
    EARLY-window numbers into `*_late.csv`. No error, no warning, a plausible table, the wrong
    answer.

    The rule from CLAUDE.md ground rule 6 is that OPTIONS TRAVEL IN THE ITEM. This checks the
    observable consequence: any global `main` assigns must also be assigned by some other
    module-level function, i.e. by the worker, from what it was handed.
    """
    path, names, tree = case
    reassigned = {n
                  for f in tree.body
                  if isinstance(f, ast.FunctionDef) and f.name != "main"
                  for g in ast.walk(f) if isinstance(g, ast.Global)
                  for n in g.names}
    missed = sorted(names - reassigned)
    assert not missed, (
        f"{path}: main() sets the module global(s) {missed} at runtime and the module fans work "
        "out over spawned processes, which re-import the module and never see them. Pass the "
        "value in the ITEM and re-apply it at the top of the worker, the way "
        "`channel_position_maps.session_maps` does with WIN_START_S.")
