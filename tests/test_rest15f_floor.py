"""rest15f_restnl_1117 --score-min-post (Priya 11-18: floor 30 primary): the floor is named in the outputs and reaches the
fit workers in the item."""
from __future__ import annotations

import inspect

import pytest

r15 = pytest.importorskip("scripts.rest_migration.rest15f_restnl_1117")


def test_floor_names_outputs_and_40_keeps_the_old_names(monkeypatch):
    monkeypatch.setenv(r15.LP_ENV, "c1all")
    r15._apply_mode()
    assert r15.SCORE_MIN_POST_DEFAULT == 30
    assert r15.nl.MIN_PERIODS == 40                                   # rest_frozen_decoder_nl's own default unchanged
    assert r15.floor_tag(40) == "" and r15.floor_tag(30) == "_floor30"
    assert r15.out_name(40) == "rest15f_restnlc1all_20261117"
    assert r15.out_name(30) == "rest15f_restnlc1all_floor30_20261118"
    assert r15.stem("all", 30) == f"rest15f_{r15.MASK}_allpre_durgap_floor30"
    assert r15.stem("all", 40) == f"rest15f_{r15.MASK}_allpre_durgap"
    txt = r15.suptitle_text("all", "pre 34", 30)
    assert ">= 30 matched periods post-stroke, >= 40 pre" in " ".join(txt.split())
    assert all(len(line) <= r15.TITLE_WRAP for line in txt.splitlines())


def test_floor_travels_in_the_fit_item():
    src = inspect.getsource(r15._fit_job)
    assert "an, pre, out, floor = item" in src and "nl.SCORE_MIN_POST = int(floor)" in src
    assert "(an, a.pre, str(out), floor)" in inspect.getsource(r15.main)
