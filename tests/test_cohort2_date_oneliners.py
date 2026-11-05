"""Cohort-2 date one-liners (code review 11-04, Phase 1 item 7). cohort1 results must not move; cohort2's 5-character
animals and New-Year sessions must come out right.

- `analysis_kit.curated_ids` took the date as ``lab[5:]``, which is one character late for ``PS118_1105``.
- `nightly_figs` sorted MMDD strings, which puts cohort2's January sessions before its October ones.
- `preprocess` built the YYYYMMDD by hand from `cohort_year()`, which gives 2026 for a January 2027 session."""
import pytest

from wfield_local import analysis_kit as ak
from wfield_local import config, paths


@pytest.fixture
def cohort(monkeypatch):
    def use(name):
        if name is None:
            monkeypatch.delenv(paths.COHORT_ENV, raising=False)
        else:
            monkeypatch.setenv(paths.COHORT_ENV, name)
    return use


def test_curated_ids_unchanged_for_cohort1(cohort):
    cohort(None)
    old = {f"{lab.split('_')[0]}_{config._to8(lab[5:])}" for lab in ak.curated_labels()}   # the pre-11-05 formula
    assert ak.curated_ids() == old
    assert len(old) > 50


def test_curated_ids_handles_five_character_animals(cohort, monkeypatch):
    cohort("cohort2")
    monkeypatch.setattr(ak, "curated_labels", lambda: ["PS118_1105", "PS121_0105"])
    assert ak.curated_ids() == {"PS118_20261105", "PS121_20270105"}


def test_full_date_sort_matches_mmdd_sort_for_cohort1(cohort):
    cohort(None)
    mmdd = sorted({s["label"][-4:] for s in config.load_sessions()})
    assert sorted(mmdd, key=config.date8) == mmdd                          # nightly_figs order unchanged for cohort1


def test_full_date_sort_crosses_new_year_for_cohort2(cohort):
    cohort("cohort2")
    labs = ["PS118_0105", "PS118_1103", "PS118_1230"]
    assert sorted(labs, key=config.date8) == ["PS118_1103", "PS118_1230", "PS118_0105"]
    assert config.date8("0105") == "20270105" != f"{config.cohort_year()}0105"   # preprocess's old hand-built year
