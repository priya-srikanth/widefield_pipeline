"""`paths.cohorts()` is memoised on cohorts.yaml's (path, mtime, size) (A10, Priya 2026-11-06).

Pinned: (1) every cohort accessor gives EXACTLY what the old re-parse-every-call code gave, for every cohort and
with/without WIDEFIELD_COHORT; (2) an edited file is re-read in the same process; (3) a caller mutating its result
cannot change the next caller's (the old code handed out a fresh parse each time)."""
import os
import shutil

import pytest
import yaml

from wfield_local import config, paths


def _fresh():
    """The pre-memo implementation: parse the file on every call."""
    with open(paths.CONFIG_DIR / "cohorts.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _old_active():
    cfg = _fresh()
    return os.environ.get(paths.COHORT_ENV, "").strip() or cfg["default"]


@pytest.mark.parametrize("env", [None, "cohort1", "cohort2"])
def test_memo_equals_fresh_parse(monkeypatch, env):
    if env is None:
        monkeypatch.delenv(paths.COHORT_ENV, raising=False)
    else:
        monkeypatch.setenv(paths.COHORT_ENV, env)
    ref = _fresh()
    for _ in range(3):                                   # cold, then warm hits
        assert paths.cohorts() == ref
        assert paths.active_cohort() == _old_active()
        for name in ref["cohorts"]:
            assert paths.cohort_spec(name) == dict(ref["cohorts"][name], name=name)
        assert paths.cohort_spec() == dict(ref["cohorts"][_old_active()], name=_old_active())
        for d in ("20260606", "20260806", "20261028", "20261029", "20270105", "20250101"):
            d_ = int(d)
            want = [n for n, c in ref["cohorts"].items()
                    if int(c["first"]) <= d_ and (c.get("last") is None or d_ <= int(c["last"]))]
            assert paths.cohort_for_date(d) == (want[0] if want else None)
        for a in ("PS92", "PS93", "PS118", "PS122", "PS999"):
            want = next((n for n, c in ref["cohorts"].items() if a in (c.get("animals") or [])), None)
            assert paths.cohort_of_animal(a) == want
        assert config.date8("0105") == config.date8("0105")


def test_unknown_cohort_still_raises(monkeypatch):
    monkeypatch.setenv(paths.COHORT_ENV, "cohort_nope")
    with pytest.raises(ValueError):
        paths.active_cohort()


def test_results_are_private_copies():
    a = paths.cohorts()
    a["default"] = "MUTATED"
    a["cohorts"]["cohort1"]["animals"].append("PSX")
    s = paths.cohort_spec("cohort1")
    s["animals"].append("PSY")
    s["first"] = 0
    assert paths.cohorts() == _fresh()
    assert paths.cohort_spec("cohort1") == dict(_fresh()["cohorts"]["cohort1"], name="cohort1")


def test_edited_file_is_reread(monkeypatch, tmp_path):
    shutil.copy(paths.CONFIG_DIR / "cohorts.yaml", tmp_path / "cohorts.yaml")
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv(paths.COHORT_ENV, raising=False)
    assert paths.active_cohort() == "cohort1"
    p = tmp_path / "cohorts.yaml"
    st = os.stat(p)
    txt = p.read_text(encoding="utf-8").replace("default: cohort1", "default: cohort2")
    p.write_text(txt, encoding="utf-8")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))   # coarse-mtime filesystems: force a change
    assert paths.active_cohort() == "cohort2"
    assert paths.cohorts()["default"] == "cohort2"


def test_same_mtime_different_size_is_reread(monkeypatch, tmp_path):
    shutil.copy(paths.CONFIG_DIR / "cohorts.yaml", tmp_path / "cohorts.yaml")
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path)
    p = tmp_path / "cohorts.yaml"
    st = os.stat(p)
    assert "PS999" not in paths.cohort_spec("cohort2")["animals"]
    p.write_text(p.read_text(encoding="utf-8").replace("animals: [PS118,", "animals: [PS999, PS118,"),
                 encoding="utf-8")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))                   # same mtime, size differs
    assert "PS999" in paths.cohort_spec("cohort2")["animals"]
