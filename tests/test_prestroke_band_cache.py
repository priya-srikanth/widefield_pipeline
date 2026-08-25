"""The pre-stroke band is memoized per session, and the POSITION SET is part of the cache key.

WHY NOT JUST FREEZE IT. Freezing the band the way `nolick_reference_prestroke.json` is frozen looks
like the obvious saving and is wrong here. On the lick-only arm `recoding_test` deliberately re-scores
every pre-stroke session over the POST-stroke session's preserved positions, because PS94 8/17 has
engaged trials at 4 positions where its pre-stroke sessions have 6 -- an unmatched comparison pits a
4-way problem (chance 0.25) against 6-way ones (0.167). The module's own docstring calls that "the
same trial-composition error that produced a spurious 'PS94 neural deficit' headline earlier in this
project". A frozen band would silently stop being position-matched and reintroduce it.

So the saving has to be a cache keyed on the position set, not a freeze. These tests pin that, plus
the asymmetry that makes it worth doing: the pre-stroke rows are the reference and repeat nightly, so
they are cached; the post-stroke row is the thing under examination and is always recomputed.
"""
import inspect

from wfield_local import poststroke_compare as pc


def test_post_stroke_rows_are_never_cached(monkeypatch):
    """The session under examination must always be recomputed -- it is the result, not a reference."""
    calls = []
    monkeypatch.setattr(pc, "accuracy_score", lambda *a, **k: calls.append(1) or 0.5)
    monkeypatch.setattr(pc, "cross_val_predict", lambda *a, **k: [0])
    monkeypatch.setattr(pc, "GroupKFold", lambda n: None)
    monkeypatch.setattr(pc, "_pipe", lambda: None)

    def boom(*a, **k):
        raise AssertionError("a post-stroke row was routed through the cache")

    monkeypatch.setattr("wfield_local.session_cache.cached", boom)
    out = pc._within_accuracy("PS94_0817", [[0]], [0], [0], 2, True, pos_key=(0, 1, 2))
    assert out == 0.5 and len(calls) == 1


def test_pre_stroke_rows_go_through_the_cache(monkeypatch):
    seen = {}
    monkeypatch.setattr(pc, "accuracy_score", lambda *a, **k: 0.9)
    monkeypatch.setattr(pc, "cross_val_predict", lambda *a, **k: [0])
    monkeypatch.setattr(pc, "GroupKFold", lambda n: None)
    monkeypatch.setattr(pc, "_pipe", lambda: None)
    monkeypatch.setattr("wfield_local.config.load_sessions",
                        lambda *a, **k: [{"label": "PS92_0606", "mc": "x"}])

    def fake(session, kind, compute, params=None, verbose=True):
        seen["kind"] = kind
        return compute()

    monkeypatch.setattr("wfield_local.session_cache.cached", fake)
    got = pc._within_accuracy("PS92_0606", [[0]], [0], [0], 2, False, pos_key=(0, 1, 2, 3, 4, 5))
    assert got == 0.9 and "within_acc__" in seen["kind"]


def test_a_different_position_set_is_a_different_key(monkeypatch):
    """The whole safety property: a 4-position band must not be served for a 6-position question."""
    keys = []
    monkeypatch.setattr(pc, "accuracy_score", lambda *a, **k: 0.9)
    monkeypatch.setattr(pc, "cross_val_predict", lambda *a, **k: [0])
    monkeypatch.setattr(pc, "GroupKFold", lambda n: None)
    monkeypatch.setattr(pc, "_pipe", lambda: None)
    monkeypatch.setattr("wfield_local.config.load_sessions",
                        lambda *a, **k: [{"label": "PS92_0606", "mc": "x"}])
    monkeypatch.setattr("wfield_local.session_cache.cached",
                        lambda session, kind, compute, params=None, verbose=True:
                        keys.append(kind) or compute())

    pc._within_accuracy("PS92_0606", [[0]], [0], [0], 2, False, pos_key=(0, 1, 2, 3, 4, 5))
    pc._within_accuracy("PS92_0606", [[0]], [0], [0], 2, False, pos_key=(0, 1, 4, 5))
    assert len(set(keys)) == 2, f"position set is not in the cache key: {keys}"


def test_fold_count_is_in_the_key(monkeypatch):
    """A session with fewer blocks gets fewer folds; that is a different estimate, not the same one."""
    keys = []
    monkeypatch.setattr(pc, "accuracy_score", lambda *a, **k: 0.9)
    monkeypatch.setattr(pc, "cross_val_predict", lambda *a, **k: [0])
    monkeypatch.setattr(pc, "GroupKFold", lambda n: None)
    monkeypatch.setattr(pc, "_pipe", lambda: None)
    monkeypatch.setattr("wfield_local.config.load_sessions",
                        lambda *a, **k: [{"label": "PS92_0606", "mc": "x"}])
    monkeypatch.setattr("wfield_local.session_cache.cached",
                        lambda session, kind, compute, params=None, verbose=True:
                        keys.append(kind) or compute())
    pc._within_accuracy("PS92_0606", [[0]], [0], [0], 5, False, pos_key=(0, 1))
    pc._within_accuracy("PS92_0606", [[0]], [0], [0], 3, False, pos_key=(0, 1))
    assert len(set(keys)) == 2


def test_an_unregistered_session_computes_rather_than_guessing(monkeypatch):
    """No session record means no trustworthy cache identity; compute instead of inventing one."""
    monkeypatch.setattr(pc, "accuracy_score", lambda *a, **k: 0.42)
    monkeypatch.setattr(pc, "cross_val_predict", lambda *a, **k: [0])
    monkeypatch.setattr(pc, "GroupKFold", lambda n: None)
    monkeypatch.setattr(pc, "_pipe", lambda: None)
    monkeypatch.setattr("wfield_local.config.load_sessions", lambda *a, **k: [])
    monkeypatch.setattr("wfield_local.session_cache.cached",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("used the cache")))
    assert pc._within_accuracy("PS99_9999", [[0]], [0], [0], 2, False, pos_key=(0,)) == 0.42


def test_the_call_site_keys_on_the_matched_position_set():
    """`scored` is the set both arms are matched on; deriving a second expression would let the cache
    key and the actual scoring drift apart."""
    src = inspect.getsource(pc.recoding_test)
    assert "pos_key=tuple(sorted(scored))" in src.replace("\n", "").replace(" ", "").replace(
        "pos_key=tuple(sorted(scored))", "pos_key=tuple(sorted(scored))") or \
        "tuple(sorted(scored))" in src, "the call site must key on `scored`"
