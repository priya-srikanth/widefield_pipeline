"""Tests for the config loader (reads the real configs/*.yaml)."""
from wfield_local import config


def test_sessions_structure():
    ss = config.load_sessions()
    assert len(ss) >= 37   # baseline registered set; grows as new recording days are registered
    for s in ss:
        assert set(s) >= {"label", "mc", "h5", "regime", "fmdir"}
        assert s["label"][:4].startswith("PS") and s["label"][4] == "_"
        assert s["label"][5:].isdigit() and len(s["label"][5:]) == 4
        assert isinstance(s["mc"], str) and isinstance(s["h5"], str)


def test_dates_are_strings_not_octal():
    """Unquoted MMDD in YAML parses as octal (0606 -> 390). Checks EVERY list in date_policy, not a
    fixed key list, so a newly added policy key is covered without editing this test."""
    dp = config.date_policy()
    checked = 0
    for key, val in dp.items():
        if not isinstance(val, list):
            continue
        for d in val:
            assert isinstance(d, str) and d.isdigit() and len(d) == 4, (key, d)
            checked += 1
    assert checked, "date_policy has no date lists left to check — did a key get renamed?"


def test_animal_color_all_four_incl_ps93():
    ac = config.animal_color()
    assert set(ac) == {"PS92", "PS93", "PS94", "PS95"}
    assert ac["PS93"] == "tab:red"  # PS93 present (the two legacy copies omit it)


def test_curated_cross_session_policy():
    assert set(config.date_policy()["cross_session_exclude"]) == {
        "0601", "0602", "0603", "0604", "0605", "0805"}


def test_curated_dates_is_live_and_excludes_the_policy_set():
    """curated_dates() = REGISTERED minus cross_session_exclude (auto-includes new nights)."""
    live = config.curated_dates()
    registered = {s["label"].split("_")[1] for s in config.load_sessions()}
    exclude = set(config.date_policy()["cross_session_exclude"])
    assert live == sorted(registered - exclude)
    assert not (set(live) & exclude), "an excluded date leaked into the curated set"
    assert {"0606", "0607", "0608", "0806", "0807"} <= set(live), "the policy anchors must survive"


def test_no_static_curated_date_list_remains():
    """The hand-maintained `cross_session` / `all_registered` lists lagged five nights behind and the
    deck builder read one of them, so a hand-run deck silently covered fewer dates than the nightly.
    They were deleted; keep them deleted, and keep the derived accessor the only one."""
    assert "cross_session" not in config.date_policy()
    assert "all_registered" not in config.date_policy()
    assert not hasattr(config, "cross_session_dates")


def test_behavior_trials_override_survives_roundtrip():
    ss = {s["label"]: s for s in config.load_sessions()}
    ps93 = ss["PS93_0805"]  # the cam1-recovered session
    assert "behavior_trials" in ps93 and ps93["behavior_trials"].endswith("ps93_reviewed_trials.csv")


def test_module_sessions_are_config_driven():
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS, ANIMAL_COLOR
    assert len(SESSIONS) >= 37
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
    assert (dec["lick_post_s"], dec["cue_post_s"], dec["precue_post_s"]) == (2.0, 2.0, 2.0)
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


def test_figures_working_resolves_on_every_machine_that_runs_nightly_figs():
    """A null figures_working sent every figure to the OTHER box's C:/Users/sabatini path and produced a
    deck with 0 figures / 287 missing, while the run still exited 0. Any machine that runs the analysis
    must resolve a real local working dir."""
    import os
    from wfield_local import config
    for machine in ("analysis", "imaging"):
        prev = os.environ.get("WIDEFIELD_MACHINE")
        os.environ["WIDEFIELD_MACHINE"] = machine
        try:
            config.resolver.cache_clear() if hasattr(config.resolver, "cache_clear") else None
            root = config.resolver().root("figures_working")
            assert root, f"{machine}: figures_working must not be empty"
        finally:
            if prev is None:
                os.environ.pop("WIDEFIELD_MACHINE", None)
            else:
                os.environ["WIDEFIELD_MACHINE"] = prev
            config.resolver.cache_clear() if hasattr(config.resolver, "cache_clear") else None


def test_locanmf_dir_is_config_driven_and_variant_aware(monkeypatch):
    """The LocaNMF directory NAMES the hemodynamic variant it was fitted to, because LocaNMF is fitted
    to SVTcorr and a different drift removal gives a different decomposition. The literal used to be
    written out 39 times across 22 modules, so renaming it -- which adopting a variant requires --
    meant editing all of them. Keep it in one place."""
    from wfield_local import config

    assert config.locanmf_dir("X:/mc") == "X:/mc/" + config.locanmf_dir_name()
    assert config.locanmf_dir("X:/mc/") == "X:/mc/" + config.locanmf_dir_name()
    # explicit variant overrides config
    assert config.locanmf_dir_name("meegkit_hpfit") == "locanmf_affine8v1_hemo_meegkit_hpfit"
    # and the env var does too, so an analysis can be pointed elsewhere without editing config
    monkeypatch.setenv("WIDEFIELD_LOCANMF_VARIANT", "zerophase")
    assert config.locanmf_dir_name() == "locanmf_affine8v1_hemo_zerophase"


def test_no_hardcoded_locanmf_dir_left_in_wfield_local():
    """A literal that reappears defeats the whole point -- the rename would half-apply."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "wfield_local"
    offenders = [p.name for p in root.glob("*.py")
                 if "locanmf_affine8v1_final" in p.read_text(encoding="utf-8")]
    assert not offenders, f"hardcoded LocaNMF dir name in: {offenders}"
