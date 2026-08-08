"""Tests for the config loader (reads the real configs/*.yaml)."""
from wfield_local import config


def test_sessions_structure():
    ss = config.load_sessions()
    assert len(ss) == 37
    for s in ss:
        assert set(s) >= {"label", "mc", "h5", "regime", "fmdir"}
        assert s["label"][:4].startswith("PS") and s["label"][4] == "_"
        assert s["label"][5:].isdigit() and len(s["label"][5:]) == 4
        assert isinstance(s["mc"], str) and isinstance(s["h5"], str)


def test_dates_are_strings_not_octal():
    dp = config.date_policy()
    for key in ("all_registered", "cross_session", "cross_session_exclude"):
        for d in dp[key]:
            assert isinstance(d, str) and d.isdigit() and len(d) == 4, (key, d)


def test_animal_color_all_four_incl_ps93():
    ac = config.animal_color()
    assert set(ac) == {"PS92", "PS93", "PS94", "PS95"}
    assert ac["PS93"] == "tab:red"  # PS93 present (the two legacy copies omit it)


def test_curated_cross_session_policy():
    assert config.cross_session_dates() == ["0606", "0607", "0608", "0806", "0807"]
    assert set(config.date_policy()["cross_session_exclude"]) == {
        "0601", "0602", "0603", "0604", "0605", "0805"}


def test_behavior_trials_override_survives_roundtrip():
    ss = {s["label"]: s for s in config.load_sessions()}
    ps93 = ss["PS93_0805"]  # the cam1-recovered session
    assert "behavior_trials" in ps93 and ps93["behavior_trials"].endswith("ps93_reviewed_trials.csv")


def test_module_sessions_are_config_driven():
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS, ANIMAL_COLOR
    assert len(SESSIONS) == 37
    assert ANIMAL_COLOR["PS93"] == "tab:red"
    assert not hasattr(__import__("wfield_local.locanmf_cue_lick_analysis", fromlist=["x"]), "L")


def test_load_sessions_animal_and_date_subset():
    a = config.load_sessions(animals=["PS93"])
    assert a and all(s["label"].startswith("PS93") for s in a) and len(a) < 37
    d = config.load_sessions(dates=["0807"])
    assert d and all(s["label"].endswith("0807") for s in d)
    ad = config.load_sessions(animals=["PS93"], dates=["0807"])
    assert [s["label"] for s in ad] == ["PS93_0807"]


def test_load_sessions_env_filter_and_explicit_override(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_ONLY_ANIMALS", "PS92, PS95")   # comma/space tolerant
    assert {s["label"][:4] for s in config.load_sessions()} == {"PS92", "PS95"}
    # explicit arg wins over the env var
    assert all(s["label"].startswith("PS93") for s in config.load_sessions(animals=["PS93"]))
