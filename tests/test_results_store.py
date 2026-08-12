"""A figure must be a VIEW of a saved result, never the only copy of it.

Motivating incident (2026-08-12): investigating why PS94_0811 looked like an outlier in the
session x session RSA required recomputing every RDM from SVTcorr, because locanmf_rsa emitted only
a .png. The numbers behind a figure have to survive it.
"""
from __future__ import annotations

import numpy as np

from wfield_local import results_store as rs


def test_roundtrip_scalars_and_arrays(tmp_path):
    payload = {"labels": ["PS94_0810", "PS94_0811"],
               "session_rsa": np.array([[1.0, 0.14], [0.14, 1.0]]),
               "rdm": {"PS94_0811": np.arange(36, dtype=float).reshape(6, 6)},
               "reliability": {"PS94_0811": 0.889},
               "nested": {"deep": {"arr": np.array([1.5, 2.5])}}}
    p = rs.save(tmp_path, "rsa_sessions", "0606-0811", payload, meta={"basis": "locanmf"})
    assert p.endswith(".json")

    got = rs.load(tmp_path, "rsa_sessions", "0606-0811")
    assert got["labels"] == payload["labels"]
    assert np.allclose(got["session_rsa"], payload["session_rsa"])
    assert np.allclose(got["rdm"]["PS94_0811"], payload["rdm"]["PS94_0811"])
    assert got["reliability"]["PS94_0811"] == 0.889
    assert np.allclose(got["nested"]["deep"]["arr"], [1.5, 2.5]), "nested arrays must round-trip"


def test_meta_records_how_the_numbers_were_made(tmp_path):
    """The basis matters: a LocaNMF RDM and an ROI RDM are not comparable, so it must be recorded."""
    rs.save(tmp_path, "rsa_sessions", "t", {"x": 1}, meta={"basis": "locanmf", "align": "lick"})
    got = rs.load(tmp_path, "rsa_sessions", "t")
    assert got["_meta"]["basis"] == "locanmf"
    assert got["_meta"]["align"] == "lick"
    assert got["_meta"]["name"] == "rsa_sessions" and got["_meta"]["schema"] == rs.SCHEMA
    assert got["_meta"]["written"]


def test_load_missing_returns_none(tmp_path):
    assert rs.load(tmp_path, "never_written", "tag") is None


def test_no_npz_written_when_there_are_no_arrays(tmp_path):
    rs.save(tmp_path, "plain", None, {"a": 1, "b": [1, 2, 3]})
    d = tmp_path / rs.SUBDIR
    assert (d / "plain.json").exists()
    assert not (d / "plain.npz").exists()
    assert rs.load(tmp_path, "plain")["b"] == [1, 2, 3]


def test_numpy_scalars_are_json_safe(tmp_path):
    rs.save(tmp_path, "np", None, {"f": np.float64(0.5), "i": np.int64(7)})
    got = rs.load(tmp_path, "np")
    assert got["f"] == 0.5 and got["i"] == 7


def test_listing_reports_what_exists_and_how(tmp_path):
    rs.save(tmp_path, "rsa_sessions", "0606-0811", {"x": 1}, meta={"basis": "roi"})
    rs.save(tmp_path, "decoder", "0811", {"y": 2}, meta={"align": "cue"})
    lst = rs.listing(tmp_path)
    assert set(lst) == {"rsa_sessions_0606-0811.json", "decoder_0811.json"}
    assert lst["rsa_sessions_0606-0811.json"]["basis"] == "roi"


def test_tagged_and_untagged_do_not_collide(tmp_path):
    rs.save(tmp_path, "x", None, {"v": "untagged"})
    rs.save(tmp_path, "x", "0606-0811", {"v": "tagged"})
    assert rs.load(tmp_path, "x")["v"] == "untagged"
    assert rs.load(tmp_path, "x", "0606-0811")["v"] == "tagged"
