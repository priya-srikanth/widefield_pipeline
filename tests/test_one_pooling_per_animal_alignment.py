"""`grant_figures` pools each (animal, alignment) ONCE, through `_pooled_bundle`.

WHY THIS EXISTS (2026-08-28). `_pooled_bundle` was extracted to be the one shared pooling, and its
own docstring records the reason: *"extracted because it was character-identical in
`fig_pattern_similarity` and `fig_pattern_similarity_per_session`, and a third and fourth copy is
how two figures that claim to describe the same trials quietly stop doing so."*

The extraction happened. The callers were never migrated. Four sites went on rebuilding the same
pooling by hand, and `_collect_5c` -- lru-cached on (align, variant) -- rebuilt it once PER VARIANT,
so five cache keys times four animals came to roughly twenty poolings where twelve serve the whole
render. That is most of why the grant stage ran six hours.

This is not only a speed property, which is why it is pinned by a test rather than left to a
comment. Four hand-written recipes for "the pre-stroke and post-stroke trials of one animal at one
alignment" agree today by inspection; they agree tomorrow only by luck. The moment one grows an
argument the others do not, two figures captioned with the same population are drawn from different
trials, and nothing in the output says so -- the same failure class as the frozen decoder that
carried a stale basis for eight days behind a name asserting it did not.

WHAT IS ALLOWED. `_pooled_bundle` itself calls `pool_sessions`. Nothing else in the module may.
A genuinely different pooling -- a different `post_s`, a different `nolick_ref`, a subset of
sessions -- is a legitimate reason to call it directly, and the way to add one is to widen the
bundle's key rather than to open a second recipe. If you are here because this test failed on such
a change, extend `_pooled_bundle` and its key; do not add the call back.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "wfield_local" / "grant_figures.py"
ALLOWED = {"_pooled_bundle"}


def _enclosing_function(tree):
    """{lineno: name of the innermost enclosing function} for every line in the module."""
    owner = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                # innermost wins: walk order is outer-first, so only overwrite with a tighter span
                prev = owner.get(ln)
                if prev is None or (node.end_lineno - node.lineno) < prev[1]:
                    owner[ln] = (node.name, node.end_lineno - node.lineno)
    return {ln: v[0] for ln, v in owner.items()}


def _calls_to(tree, func_name):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name == func_name:
                out.append(node.lineno)
    return out


def test_pool_sessions_is_called_only_by_the_bundle():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    owner = _enclosing_function(tree)
    sites = [(owner.get(ln, "<module>"), ln) for ln in _calls_to(tree, "pool_sessions")]
    offenders = sorted((fn, ln) for fn, ln in sites if fn not in ALLOWED)
    assert not offenders, (
        "these call pool_sessions directly instead of using _pooled_bundle: "
        + ", ".join(f"{a} (line {ln})" for a, ln in offenders)
        + ". Widen the bundle's cache key rather than opening a second recipe -- see this "
          "module's docstring.")


def test_the_four_migrated_figures_go_through_the_bundle():
    """The specific sites that used to rebuild it. Named, so a revert is loud rather than slow.

    A blanket "nobody calls pool_sessions" would also pass if these functions were deleted or
    stopped pooling altogether, which is not the property being pinned.
    """
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    owner = _enclosing_function(tree)
    users = {owner.get(ln) for ln in _calls_to(tree, "_pooled_bundle")}
    for fn in ("fig_confusion_pre_post_working", "_collect_5c",
               "fig_pattern_similarity_per_session", "fig_pattern_similarity", "_collect_7"):
        assert fn in users, f"{fn} no longer obtains its trials from _pooled_bundle"


def test_the_bundle_carries_the_numeric_labels_the_decoders_fit_on():
    """`YE`/`YU` as well as `en`/`un`.

    Dropping them is what kept `fig_confusion_pre_post_working` and `_collect_5c` on their own
    recipes: both fit and score on the integer position codes, and refitting on the display names
    would be a different call to the optimiser rather than a rename.
    """
    from wfield_local import grant_figures

    src = ast.parse(SRC.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(src)
              if isinstance(n, ast.FunctionDef) and n.name == "_pooled_bundle")
    keys = {k.value for n in ast.walk(fn) if isinstance(n, ast.Dict)
            for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    for k in ("XE", "YE", "GE", "XU", "YU", "GU", "en", "un", "BE", "BU",
              "kept", "pre_i", "e_pre", "not_eng"):
        assert k in keys, f"_pooled_bundle no longer returns {k!r}"
    assert hasattr(grant_figures, "_pooled_bundle")
