"""ONE August-only baseline cut (code review 11-04 item 2, Priya 11-05: "curated august sessions from config").

`configs/cohorts.yaml <cohort>.baseline_pre_from` is the definition; `config.baseline_pre_from()` reads it and
`config.early_pre(x)` tests a session against it by `config.date_key` (not an MMDD string compare). These tests pin
that (1) the constants that still carry the value agree with it, (2) the CURATED set an August-only filter keeps is
the same under every spelling the repo used before (0801, 0806, month == 08), so the sweep moved no session, and
(3) the comparison is cohort- and year-aware.
"""
import pytest

from wfield_local import analysis_kit as ak
from wfield_local import config


def test_cohort1_cut_is_0806_and_cohort2_has_none(monkeypatch):
    assert config.baseline_pre_from() == "0806"           # Priya 11-05: the first curated August day
    assert config.early_pre("PS92_0805")                  # the excluded wonky day is never baseline
    monkeypatch.setenv("WIDEFIELD_COHORT", "cohort2")
    assert config.baseline_pre_from() is None
    assert config.early_pre("PS118_1105") is False


def _pre_set(monkeypatch, value):
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", value)
    return config.phase_labels("pre")


def test_run_mode_values_select_the_cohort_cut_s_pre_set(monkeypatch):
    """WIDEFIELD_PRE_FROM stays "0801" (it names output folders, _pre0801_); it must pick the SAME pooled pre set as
    the cohort cut. Values are compared by the sessions they select, not as strings."""
    import yaml

    from wfield_local import nightly_figs, orofacial_cohort, run_mode
    cut = config.baseline_pre_from()
    assert orofacial_cohort.LATE_PRE_FROM == cut
    prof = yaml.safe_load(open(config.CONFIG_DIR / "render_profiles" / "cohort1_rebuild_20261103.yaml",
                               encoding="utf-8"))
    want = _pre_set(monkeypatch, cut)
    for v in (nightly_figs.IMAGING_PRE_FROM, run_mode.PRIMARY_PRE_FROM, prof["env"]["WIDEFIELD_PRE_FROM"]):
        assert _pre_set(monkeypatch, v) == want, v
    monkeypatch.delenv("WIDEFIELD_PRE_FROM")
    assert want == [x for x in config.phase_labels("pre") if not config.early_pre(x)]


def test_every_old_spelling_keeps_the_same_curated_sessions():
    cur = ak.curated_labels()
    new = [x for x in cur if not config.early_pre(x)]
    assert new == [x for x in cur if x[-4:] >= "0806"]
    assert new == [x for x in cur if x[-4:] >= "0801"]   # same set ONLY because date_policy drops 0805
    assert new == [x for x in cur if x[-4:-2] != "06"]
    early = [x for x in cur if config.early_pre(x)]
    assert early and {x[-4:] for x in early} == {"0606", "0607", "0608"}
    assert all(config.session_phase(x.split("_")[0], x.split("_")[1]) == "pre" for x in early)


def test_early_pre_accepts_labels_mmdd_and_yyyymmdd():
    assert config.early_pre("PS92_0607") and config.early_pre("0607") and config.early_pre("20260607")
    assert not config.early_pre("PS92_0806") and not config.early_pre("PS93_1005")
    with pytest.raises(ValueError):
        config.early_pre("PS92_0806_pix")                 # a file stem must be cut to its label first


def test_the_compare_is_year_aware(monkeypatch):
    """A cohort that crosses New Year: an MMDD string compare says 0105 < 1105; date_key says it is LATER."""
    monkeypatch.setenv("WIDEFIELD_COHORT", "cohort2")
    from wfield_local import paths
    real = paths.cohort_spec
    monkeypatch.setattr(paths, "cohort_spec", lambda name=None: {**real(name), "baseline_pre_from": "1201"})
    assert config.early_pre("PS118_1105")
    assert not config.early_pre("PS118_0105")             # 2027-01-05, after the 2026-12-01 cut
    assert "0105" < "1201"                                # what the old spelling would have concluded
