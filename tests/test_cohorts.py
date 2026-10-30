"""Cohorts are kept apart (configs/cohorts.yaml, 2026-10-29): cohort1 = PS92-95 unchanged by default, cohort2 =
PS118 / PS120-122 only when selected, and cohort-level outputs never share a folder."""
import pytest

from wfield_local import config, paths
from wfield_local.paths import PathResolver


@pytest.fixture
def cohort(monkeypatch):
    def use(name):
        if name is None:
            monkeypatch.delenv(paths.COHORT_ENV, raising=False)
        else:
            monkeypatch.setenv(paths.COHORT_ENV, name)
    return use


def test_default_is_cohort1_and_unchanged(cohort):
    cohort(None)
    assert config.cohort() == "cohort1"
    assert set(config.animals()) == {"PS92", "PS93", "PS94", "PS95"}
    assert {config.animal_of(s) for s in config.load_sessions()} == {"PS92", "PS93", "PS94", "PS95"}
    assert config.cohort_year() == "2026"
    rv = PathResolver(machine="analysis")
    assert rv.root("cohort_out") == rv.root("labcams")          # cohort1 outputs stay where they always were
    assert not rv.root("figures_working").endswith("/cohort1")


def test_cohort2_sees_only_its_animals(cohort):
    cohort("cohort2")
    assert set(config.animals()) == {"PS118", "PS120", "PS121", "PS122"}
    assert not {config.animal_of(s) for s in config.load_sessions()} & {"PS92", "PS93", "PS94", "PS95"}
    assert config.date_policy()["cross_session_exclude"] == []   # cohort1's June exclusions do not carry over
    assert config.stroke_cutoff() is None                        # cohort1's lesion dates do not leak in


def test_cohort_outputs_are_separate(cohort):
    cohort("cohort2")
    rv = PathResolver(machine="analysis")
    for name in paths.COHORT_SCOPED_ROOTS:
        assert rv.root(name).endswith("/cohort2"), name
    assert rv.root("cohort_out") == rv.root("labcams") + "/cohort2"
    assert not rv.root("labcams").endswith("/cohort2")          # SESSION data root is shared, never scoped


def test_cohorts_yaml_matches_animal_files(cohort):
    for name, spec in paths.cohorts()["cohorts"].items():
        cohort(name)
        assert set(config.animals()) == set(spec["animals"]), name
        for a in spec["animals"]:
            assert config.cohort_of(a) == name


def test_cohort_for_date():
    assert paths.cohort_for_date("20260806") == "cohort1"
    assert paths.cohort_for_date("20261029") == "cohort2"
    assert paths.cohort_for_date("20250101") is None


def test_use_cohort_for_dates_refuses_a_span(cohort):
    cohort(None)
    with pytest.raises(SystemExit):
        config.use_cohort_for_dates(["20261005", "20261029"])


def test_preprocess_discovery_sees_three_digit_ids_but_only_the_active_cohort(cohort):
    from wfield_local import preprocess
    assert preprocess.ANIMAL_RE.search("PS118_20261029_120000").group(1) == "PS118"
    cohort("cohort2")
    rows = [dict(animal=a, sess=f"{a}_20261029_x") for a in ("PS118", "PS122", "PS94", "PS104", None)]
    kept = {r["animal"] for r in preprocess._in_cohort(rows, "20261029")}
    assert kept == {"PS118", "PS122", None}


def test_no_reference_yet_skips_cross_registration(cohort):
    from wfield_local import preprocess
    cohort("cohort2")
    params = config.defaults()["preprocess"]
    assert preprocess.reference_for("PS118", params, PathResolver(machine="analysis")) is None


def test_await_locanmf_registers_into_the_active_cohorts_file(cohort):
    import yaml
    from wfield_local import await_locanmf as aw
    cohort("cohort2")
    path = aw._sessions_yaml()
    assert path.as_posix().endswith("configs/cohort2/sessions.yaml")
    text = aw.insert_session_entry(path.read_text(encoding="utf-8"), "PS118", "1029",
                                   "20261029/PS118_20261029_120000/motion_corrected",
                                   "20261029/PS118_20261029_120001.h5", "B", None)
    raw = yaml.safe_load(text)["sessions"]
    assert set(raw["PS118"]) == {"1029"} and raw["PS120"] is None    # bare headers stay valid
