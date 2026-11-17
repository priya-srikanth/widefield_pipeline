"""The session-cache KIND carries the lick-detector part (``daqlk``) in EVERY run; output names do not change (11-06).

Before: `run_mode.lickdet_part` switched on only with another output part (the pooling-only ``pre`` included), so an
untagged run keyed ``tf-cue-x`` and a pre-only run ``tf-cue-x-daqlk`` for identical content (one detector since
session_cache v14). Now `session_cache.effective_kind` uses `run_mode.session_kind_tag` (detector part always on);
`run_mode.tag` / `cache_tag` / `keyed_name` / `lp_licks.output_dir` keep the old rule, pinned byte-for-byte here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from wfield_local import config, lp_licks, render_profile, run_mode, session_cache

_OWNED = render_profile.OWNED_PREFIXES

PROFILE = "cohort1_rebuild_20261103"
PROFILE_TAG = ("hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-"
               "incomplete-stopA2jexrt3500c70_daqlk")


@pytest.fixture
def clean_env(monkeypatch):
    import os
    for k in list(os.environ):
        if k.startswith(_OWNED):
            monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_untagged_and_pre_only_share_one_kind_with_the_detector_part(clean_env):
    untagged = session_cache.effective_kind("tf-cue-x")
    clean_env.setenv("WIDEFIELD_PRE_FROM", "0801")
    pre_only = session_cache.effective_kind("tf-cue-x")
    assert untagged == pre_only == "tf-cue-x-daqlk"
    assert untagged.endswith("-" + config.LICK_DETECTOR_TAG)


def test_output_names_unchanged_untagged(clean_env):
    assert run_mode.tag() == "" and run_mode.cache_tag() == "" and not lp_licks.output_tag()
    assert Path(lp_licks.output_dir("X:/out")) == Path("X:/out")
    assert run_mode.keyed_name("trials_cue_PS92.csv.gz", cache=True) == "trials_cue_PS92.csv.gz"
    assert run_mode.keyed_name("x.json") == "x.json"
    assert run_mode.subdir("X:/c", cache=True) == Path("X:/c") and run_mode.subdir("X:/c") == Path("X:/c")


def test_output_names_unchanged_pre_only(clean_env):
    clean_env.setenv("WIDEFIELD_PRE_FROM", "0801")
    assert run_mode.tag() == "pre0801_daqlk" == lp_licks.output_tag()
    assert run_mode.cache_tag() == "daqlk"
    assert Path(lp_licks.output_dir("X:/out")) == Path("X:/pre0801_daqlk/out")
    assert run_mode.keyed_name("trials_cue_PS92.csv.gz", cache=True) == "trials_cue_PS92.csv__daqlk.gz"


def test_tagged_kind_identical_to_cache_tag(clean_env):
    """Any run with an output part: the kind tag is exactly the old cache tag (no tagged key moves)."""
    clean_env.setenv("WIDEFIELD_PRE_FROM", "0801")
    clean_env.setenv(config.LOCANMF_PROJECT_ENV, "1")
    assert "lnmfproj" in run_mode.tag()
    assert run_mode.session_kind_tag() == run_mode.cache_tag()
    assert session_cache.effective_kind("tf-cue-x") == "tf-cue-x-" + run_mode.cache_tag()


def test_render_profile_tag_and_kind_unchanged(clean_env):
    env: dict = {}
    d = render_profile.apply(PROFILE, env)
    for k, v in env.items():
        clean_env.setenv(k, v)
    assert d["notes"]["expected_run_tag"] == PROFILE_TAG
    assert run_mode.tag() == PROFILE_TAG == lp_licks.output_tag()
    assert Path(lp_licks.output_dir("X:/out")) == Path("X:/") / PROFILE_TAG / "out"
    assert run_mode.session_kind_tag() == run_mode.cache_tag()
    assert run_mode.cache_tag().startswith(PROFILE_TAG.replace("_pre0801", "") + "_csig-")
