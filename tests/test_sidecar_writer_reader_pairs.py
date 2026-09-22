"""A WRITER AND ITS READER MUST RESOLVE THE SAME PATH.

The 2026-09-21 move of sidecars into ``data/`` changed `find_sidecar` on the READ side while five
writers kept building the flat path by hand. Both halves worked; they simply addressed different
files. For a day the pipeline wrote fresh numbers where nothing looked and served numbers from two
days earlier, and nothing anywhere raised so much as a warning -- the stale file existed, so it
resolved.

A test that only checks the writer, or only the reader, cannot see that. Each case here writes
with the real writer and reads with the real reader, so the two are asserted AGAINST EACH OTHER
rather than against a path spelled out in the test -- which would just be a third opinion, free to
be wrong in the same way.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from wfield_local import figure_layout as fl

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rotation_maps = _load("rotation_maps", "scripts/rest_migration/rotation_maps.py")


# ------------------------------------------------------------------ figure_layout's own contract

def test_what_sidecar_for_writes_is_what_find_sidecar_for_reads(tmp_path):
    p = fl.sidecar_for(tmp_path, "fig", "_sessions.csv")
    p.write_text("x\n", encoding="utf-8")
    assert fl.find_sidecar_for(tmp_path, "fig", "_sessions.csv") == p


def test_the_dir_and_png_spellings_agree(tmp_path):
    """`sidecar_for(dir, stem, sfx)` and `sidecar(dir/stem.png, sfx)` are the same rule."""
    assert (fl.sidecar_for(tmp_path, "fig", ".csv")
            == fl.sidecar(tmp_path / "fig.png", ".csv"))
    assert (fl.find_sidecar_for(tmp_path, "fig", ".csv")
            == fl.find_sidecar(tmp_path / "fig.png", ".csv"))


def test_a_sidecar_written_before_the_move_is_still_found(tmp_path):
    """MICROSCOPE keeps originals and a colleague's copy may never be migrated."""
    flat = tmp_path / "fig.csv"
    flat.write_text("pre-migration\n", encoding="utf-8")
    assert fl.find_sidecar_for(tmp_path, "fig", ".csv") == flat


def test_the_new_location_WINS_over_a_flat_leftover(tmp_path):
    """This preference is the whole reason a flat writer is dangerous rather than merely untidy."""
    (tmp_path / "fig.csv").write_text("flat\n", encoding="utf-8")
    new = fl.sidecar_for(tmp_path, "fig", ".csv")
    new.write_text("data dir\n", encoding="utf-8")
    assert fl.find_sidecar_for(tmp_path, "fig", ".csv").read_text(encoding="utf-8") == "data dir\n"


def test_find_returns_None_rather_than_a_path_that_is_not_there(tmp_path):
    assert fl.find_sidecar_for(tmp_path, "absent", ".csv") is None


# --------------------------------------------------------- rotation_maps: the --replot round trip

def _cache(n=2):
    return [{"arm": "cue", "contrast": "far-near", "position": i, "animal": "PS92",
             "z": np.arange(4, dtype=np.float32) + i, "sig": np.ones(4, bool)} for i in range(n)]


def test_the_rotation_cache_is_read_back_from_where_it_was_written(tmp_path):
    """`--replot` must find today's cache, not the one before the move."""
    rows = [{"arm": "cue", "animal": "PS92", "position": 0}]
    p = rotation_maps.save_cache(_cache(), rows, tmp_path, sha="deadbeef")
    assert p.parent.name == fl.DATA_DIR, "the writer must not put it flat"
    cache, back, sha = rotation_maps.load_cache(tmp_path)
    assert sha == "deadbeef" and back == rows and len(cache) == 2


def test_the_rotation_cache_written_before_the_move_still_replots(tmp_path):
    """A cache from an older run sits flat; refusing to see it costs a 30-minute recompute."""
    rows = [{"arm": "cue", "animal": "PS92", "position": 0}]
    p = rotation_maps.save_cache(_cache(), rows, tmp_path, sha="old")
    flat = tmp_path / p.name
    p.replace(flat)
    cache, back, sha = rotation_maps.load_cache(tmp_path)
    assert sha == "old" and len(cache) == 2


def test_an_absent_rotation_cache_is_reported_not_invented(tmp_path):
    assert rotation_maps.load_cache(tmp_path) == (None, None, None)
    assert rotation_maps.load_draws(tmp_path) == []


def test_the_rotation_draws_round_trip(tmp_path):
    draws = [{"arm": "cue", "animal": "PS92", "contrast": "far-near", "position": 1,
              "cos_obs": 0.4, "cos_null": np.zeros(3, np.float32),
              "cos_boot": np.ones(2, np.float32)}]
    p = rotation_maps.save_draws(draws, tmp_path)
    assert p.parent.name == fl.DATA_DIR
    back = rotation_maps.load_draws(tmp_path)
    assert len(back) == 1 and back[0]["position"] == 1
    assert back[0]["cos_null"].shape == (3,) and back[0]["cos_boot"].shape == (2,)


def test_the_rotation_region_table_goes_to_data(tmp_path):
    p = rotation_maps.write_csv([{"animal": "PS92", "region": "M1", "cos": 0.3}], tmp_path)
    assert p.parent.name == fl.DATA_DIR and p.exists()


@pytest.mark.parametrize("mod, rel", [
    ("transfer_arms", "scripts/rest_migration/transfer_arms.py"),
    ("rest_frozen_decoder", "scripts/rest_migration/rest_frozen_decoder.py"),
    ("shared_position_projection", "scripts/rest_migration/shared_position_projection.py"),
    ("channel_position_maps", "scripts/rest_migration/channel_position_maps.py"),
])
def test_no_fixed_writer_builds_a_flat_sidecar_path_again(mod, rel):
    """A backstop, not the main guard -- the round trips above are.

    This one is source text, so it goes blind the moment somebody spells the join a new way; it is
    here only to catch the literal reintroduction of the pattern that caused the incident.
    """
    src = (ROOT / rel).read_text(encoding="utf-8")
    bad = [ln.strip() for ln in src.splitlines()
           if ("out_dir /" in ln or "a.out /" in ln or "out /" in ln)
           and any(x in ln for x in ('.csv"', ".csv'", '.npz"', '.svg"'))]
    assert not bad, f"{mod} builds a sidecar path by hand again:\n  " + "\n  ".join(bad)
