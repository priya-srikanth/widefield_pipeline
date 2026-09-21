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
    """``(path, lineno, source_window)`` for every `parallel.fan_out(` call in the tree."""
    out = []
    for p in sorted(list((ROOT / "wfield_local").rglob("*.py"))
                    + list((ROOT / "scripts").rglob("*.py"))):
        if "__pycache__" in p.parts or p.name == "parallel.py":
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, ln in enumerate(lines):
            if re.search(r"\bfan_out\s*\(", ln) and not ln.lstrip().startswith("#"):
                out.append((p.relative_to(ROOT).as_posix(), i + 1,
                            "\n".join(lines[i:i + WINDOW])))
    return out


def test_there_are_fan_out_sites_to_check():
    """A guard on the guard: a rename of `fan_out` would otherwise make this file vacuously pass."""
    assert len(_fan_out_sites()) >= 6


@pytest.mark.parametrize("site", _fan_out_sites(), ids=lambda s: f"{s[0]}:{s[1]}")
def test_every_fan_out_result_is_ordered_before_use(site):
    path, lineno, window = site
    key = f"{path}:{lineno}"
    if key in ALLOWED:
        pytest.skip(ALLOWED[key])
    assert "sorted(" in window, (
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
