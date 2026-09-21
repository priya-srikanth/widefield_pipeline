"""A frozen-model lookup that misses because the STORE MOVED must say so.

WHY. On 2026-09-18 commit 889c5e0 gave the analysis desktop its own machine profile, which set
`figures_working` for it, which moved `frozen_models.local_dir()` off the 48 models fitted on
2026-08-28. `find()` returned None, `load_or_fit` refitted exactly as designed, and nothing
reported anything for two weeks. The models were still valid -- same training labels, same input
signatures, same `freeze_version` -- they were simply no longer being looked for.

The distinction the code was missing: an EMPTY store means "nothing frozen yet"; an ABSENT one
means "you are not looking where the models are". Only the second is a bug, and only the second
can silently disable the SPEC-CHANGED warning, because `siblings()` has nothing to compare a new
training set against.
"""
from __future__ import annotations

from pathlib import Path

from wfield_local import frozen_models as fm

ROOT = Path(__file__).resolve().parents[1]


def _fresh():
    fm._ANNOUNCED.clear()


def test_a_missing_local_root_is_reported(tmp_path, monkeypatch):
    _fresh()
    gone = tmp_path / "not_created"
    monkeypatch.setattr(fm, "local_dir", lambda: gone)
    said = []
    assert fm.warn_if_store_moved(said.append) is True
    assert "DOES NOT EXIST" in said[0]


def test_an_empty_but_present_store_is_not_reported(tmp_path, monkeypatch):
    """An empty store is the normal first run and must stay quiet, or the warning becomes noise."""
    _fresh()
    here = tmp_path / "frozen_models"
    here.mkdir()
    monkeypatch.setattr(fm, "local_dir", lambda: here)
    said = []
    assert fm.warn_if_store_moved(said.append) is False
    assert said == []


def test_it_warns_once_per_process(tmp_path, monkeypatch):
    """`load_or_fit` is called once per (animal, kind, align, source) -- 48 times in a sweep."""
    _fresh()
    monkeypatch.setattr(fm, "local_dir", lambda: tmp_path / "nope")
    said = []
    assert fm.warn_if_store_moved(said.append) is True
    assert fm.warn_if_store_moved(said.append) is False
    # Count the WARNING, not the lines: one warning may add a second line pointing at the legacy
    # store, and on the analysis desktop that store really exists.
    assert sum("DOES NOT EXIST" in s for s in said) == 1


def test_it_points_at_the_legacy_store_when_one_is_there(tmp_path, monkeypatch):
    """The actionable half: name the directory the models are actually in, and say do not delete."""
    _fresh()
    legacy = tmp_path / "wf_local" / "frozen_models" / "PS92" / "decoder_cue_roi_abc123"
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(fm, "local_dir", lambda: tmp_path / "nope")
    monkeypatch.setattr(fm, "legacy_local_dir",
                        lambda: tmp_path / "wf_local" / "frozen_models")
    said = []
    fm.warn_if_store_moved(said.append)
    joined = "\n".join(said)
    assert "1 model(s) are sitting in" in joined
    assert "publish_basis" in joined
    assert "Do NOT delete" in joined


def test_load_or_fit_calls_it_on_a_miss(tmp_path, monkeypatch):
    """The warning is worthless if nothing invokes it. `load_or_fit` must, on every miss."""
    _fresh()
    monkeypatch.setattr(fm, "local_dir", lambda: tmp_path / "nope")
    monkeypatch.setattr(fm, "server_dir", lambda: None)
    monkeypatch.setattr(fm, "_write", lambda *a, **k: None)
    spec = {"animal": "PS92", "kind": "decoder", "align": "cue", "source": "roi",
            "basis_id": None, "train_labels": []}
    said = []
    _payload, status = fm.load_or_fit(spec, lambda: {"full": None}, log=said.append)
    assert status == fm.STATUS_NEW
    assert any("DOES NOT EXIST" in s for s in said)


def test_the_legacy_path_is_derived_from_the_basis_fallback_not_repeated():
    """One literal, in the one module `test_no_hardcoded_machine_paths` sanctions for it.

    Writing `C:/wf_local/frozen_models` here a second time both tripped that guard and created a
    pointer that would go stale the moment `_basis_dir`'s fallback moved. Deriving it means the
    two cannot disagree.
    """
    from wfield_local import joint_locanmf

    assert fm.legacy_local_dir() == joint_locanmf.FALLBACK_BASIS_DIR.parent / "frozen_models"
    assert fm.legacy_local_dir().name == "frozen_models"
    # That the literal itself stays out of executable code is `test_no_hardcoded_machine_paths`'s
    # job, and it owns the docstring-vs-code distinction. A crude `"wf_local" not in source` check
    # here failed on the INCIDENT NARRATIVE in `warn_if_store_moved`'s docstring -- which is
    # documentation, and exactly what that guard is careful to exempt. One fact, one test.
