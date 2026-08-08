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


def test_expand_dates_shared_grammar():
    avail = ["0605", "0606", "0607", "0608", "0806", "0807"]
    # single, either width, comma/space list
    assert config.expand_dates("0807", available=avail) == ["0807"]
    assert config.expand_dates("20260807", available=avail) == ["0807"]
    assert config.expand_dates("0806 0807", available=avail) == ["0806", "0807"]
    assert config.expand_dates("0806,0807", available=avail) == ["0806", "0807"]
    assert config.expand_dates(["0806", "0807"], available=avail) == ["0806", "0807"]
    # width=8 output (preprocessing form)
    assert config.expand_dates("0807", width=8, available=avail) == ["20260807"]
    # range intersects the available set (so it respects month boundaries / gaps)
    assert config.expand_dates("0605-0608", available=avail) == ["0605", "0606", "0607", "0608"]
    assert config.expand_dates("0606-0806", available=avail) == ["0606", "0607", "0608", "0806"]
    # 'all' expands to the available set; sorted + de-duplicated
    assert config.expand_dates("all", available=avail) == sorted(set(avail))
    assert config.expand_dates("0807 0807 0806", available=avail) == ["0806", "0807"]
    # explicit single passes through even if not in available (a freshly-acquired date)
    assert config.expand_dates("0810", available=avail) == ["0810"]


def test_expand_dates_all_needs_available():
    import pytest
    with pytest.raises(ValueError):
        config.expand_dates("all")
    with pytest.raises(ValueError):
        config.expand_dates("junk", available=["0807"])


def test_deep_merge_pure():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 9}, "c": 4}
    out = config._deep_merge(base, over)
    assert out == {"a": {"x": 1, "y": 9}, "b": 3, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}   # base untouched


def test_defaults_session_override_noop():
    # no active overrides in session_overrides.yaml -> defaults(session) == defaults()
    assert config.defaults("PS93_0807") == config.defaults()
    assert config.defaults() is config.defaults()    # cached same object when no session


def test_defaults_blocks_are_the_single_source():
    d = config.defaults()
    assert d["locanmf"]["r2_thresh"] == 0.95 and d["locanmf"]["loc_thresh"] == 80 and d["locanmf"]["maxrank"] == 20
    dec = d["decode"]
    assert dec["aligns"] == ["lick", "cue", "precue"]
    assert (dec["lick_post_s"], dec["cue_post_s"], dec["precue_post_s"]) == (2.0, 2.0, 1.0)
    assert dec["max_rt_s"] == 2.0 and dec["cv"] == "block" and dec["baseline"] == "none" and dec["chance"] == 0.167


def test_normalize_animals():
    assert config.normalize_animals(None) is None
    assert config.normalize_animals([]) is None
    assert config.normalize_animals("all") is None
    assert config.normalize_animals(["PS93", "all"]) is None   # 'all' anywhere => no filter
    assert config.normalize_animals("PS93 PS94") == ["PS93", "PS94"]
    assert config.normalize_animals(["PS93", "PS94"]) == ["PS93", "PS94"]


def test_load_sessions_env_filter_and_explicit_override(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_ONLY_ANIMALS", "PS92, PS95")   # comma/space tolerant
    assert {s["label"][:4] for s in config.load_sessions()} == {"PS92", "PS95"}
    # explicit arg wins over the env var
    assert all(s["label"].startswith("PS93") for s in config.load_sessions(animals=["PS93"]))
