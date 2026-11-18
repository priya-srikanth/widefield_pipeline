"""rest15f_restnl_1117 --score-min-post / --match-pct (Priya 11-18: floor 30 + central 90 % primary): both are named in
the outputs and reach the fit workers in the item; rest_frozen_decoder_nl keeps its own defaults."""
from __future__ import annotations

import inspect

import pytest

r15 = pytest.importorskip("scripts.rest_migration.rest15f_restnl_1117")


def test_settings_name_outputs_and_the_old_settings_keep_the_old_names(monkeypatch):
    monkeypatch.setenv(r15.LP_ENV, "c1all")
    r15._apply_mode()
    assert (r15.SCORE_MIN_POST_DEFAULT, r15.MATCH_PCT_DEFAULT) == (30, 5.0)
    assert r15.nl.MIN_PERIODS == 40
    assert r15.floor_tag((40, 10)) == "" and r15.floor_tag(None) == ""
    assert r15.floor_tag((30, 10)) == "_floor30"
    assert r15.floor_tag((30, 5)) == "_floor30_pct5"
    assert r15.out_name((40, 10)) == "rest15f_restnlc1all_20261117"
    assert r15.out_name((30, 10)) == "rest15f_restnlc1all_floor30_20261118"
    assert r15.out_name((30, 5)) == "rest15f_restnlc1all_floor30_pct5_20261118"
    assert r15.stem("all", (30, 5)) == f"rest15f_{r15.MASK}_allpre_durgap_floor30_pct5"
    txt = " ".join(r15.suptitle_text("all", "pre 34", (30, 5)).split())
    assert ">= 30 matched periods post-stroke, >= 40 pre" in txt and "central 90% of pre" in txt
    assert all(len(line) <= r15.TITLE_WRAP for line in r15.suptitle_text("all", "pre 34", (30, 5)).splitlines())


def test_apply_cfg_sets_the_nl_globals(monkeypatch):
    monkeypatch.setattr(r15.nl, "SCORE_MIN_POST", r15.nl.SCORE_MIN_POST)
    monkeypatch.setattr(r15.nl, "PCT", r15.nl.PCT)
    r15._apply_cfg((30, 5))
    assert r15.nl.SCORE_MIN_POST == 30 and r15.nl.PCT == 5.0


def test_settings_travel_in_the_fit_item():
    src = inspect.getsource(r15._fit_job)
    assert "an, pre, out, cfg = item" in src and "_apply_cfg(cfg)" in src
    assert "(an, a.pre, str(out), cfg)" in inspect.getsource(r15.main)
