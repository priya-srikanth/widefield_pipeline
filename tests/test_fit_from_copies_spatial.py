"""A --fit-from encoder run copies the source run's spatial basis into its own fit/ (11-08).

The pixel readers read <run>/fit/spatial_<animal>.pkl from the run they plot; --fit-from runs kept the basis only in
the source run, so v_pix_frAsplit_new / v_pixmus_unique_noFR_cue failed on 11-07.
"""
from __future__ import annotations

import inspect

from wfield_local import frozen_encoder as fe


def test_copies_missing_files_byte_for_byte(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    (src / "spatial_PS92.pkl").write_bytes(b"basis92")
    (src / "spatial_PS93.pkl").write_bytes(b"basis93")
    got = fe.copy_fit_spatial(src, dst, ["PS92", "PS93", "PS94"])      # PS94 has no source: skipped
    assert [p.name for p in got] == ["spatial_PS92.pkl", "spatial_PS93.pkl"]
    assert (dst / "spatial_PS92.pkl").read_bytes() == b"basis92"
    assert not (dst / "spatial_PS94.pkl").exists()


def test_existing_copy_is_left_alone(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    (src / "spatial_PS92.pkl").write_bytes(b"new")
    (dst / "spatial_PS92.pkl").write_bytes(b"old")
    assert fe.copy_fit_spatial(src, dst, ["PS92"]) == []
    assert (dst / "spatial_PS92.pkl").read_bytes() == b"old"


def test_main_calls_it_on_the_fit_from_reuse_path():
    src = inspect.getsource(fe.main) if hasattr(fe, "main") else inspect.getsource(fe)
    assert "copy_fit_spatial(src_fit, out / \"fit\"" in src
