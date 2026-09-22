"""The stray-sidecar audit must fail on the thing it exists to catch, and refuse an empty run.

`scripts/check_figure_layout` was written after five writers spent a day putting fresh CSVs flat
beside their figures while every reader resolved the `data/` copy from two days earlier. The tool
is only worth having if it FAILS on that arrangement, so each case below builds the arrangement
and asserts the exit code -- three other checks written this week passed on an empty measurement
before anyone noticed, which is the failure this file is guarding against.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_figure_layout", ROOT / "scripts" / "check_figure_layout.py")
cfl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfl)


def _png(d: pathlib.Path, stem: str) -> pathlib.Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stem}.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


def _aged(p: pathlib.Path, text: str, when: float) -> pathlib.Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    os.utime(p, (when, when))
    return p


def test_a_flat_sidecar_beside_a_figure_is_a_stray(tmp_path):
    _png(tmp_path, "fig_a")
    (tmp_path / "fig_a.csv").write_text("x\n", encoding="utf-8")
    strays, shadows, n_dirs, n_png = cfl.scan(tmp_path)
    assert [p.name for p in strays] == ["fig_a.csv"]
    assert shadows == [] and n_dirs == 1 and n_png == 1
    assert cfl.main(["prog", str(tmp_path)]) == 1


def test_a_flat_copy_NEWER_than_the_one_readers_get_is_a_stale_shadow(tmp_path):
    """The severe case: both files exist, so nothing is missing and nothing errors."""
    _png(tmp_path, "fig_b")
    now = time.time()
    _aged(tmp_path / "data" / "fig_b.csv", "old numbers\n", now - 86400)
    _aged(tmp_path / "fig_b.csv", "fresh numbers\n", now)
    strays, shadows, _, _ = cfl.scan(tmp_path)
    assert strays == []
    assert [(f.name, t.name) for f, t in shadows] == [("fig_b.csv", "fig_b.csv")]
    assert cfl.main(["prog", str(tmp_path)]) == 1


def test_a_flat_copy_OLDER_than_the_data_copy_is_only_a_stray(tmp_path):
    _png(tmp_path, "fig_c")
    now = time.time()
    _aged(tmp_path / "fig_c.csv", "leftover\n", now - 86400)
    _aged(tmp_path / "data" / "fig_c.csv", "current\n", now)
    strays, shadows, _, _ = cfl.scan(tmp_path)
    assert [p.name for p in strays] == ["fig_c.csv"] and shadows == []


def test_the_svg_twin_is_looked_for_in_svg_not_data(tmp_path):
    _png(tmp_path, "fig_d")
    now = time.time()
    _aged(tmp_path / "svg" / "fig_d.svg", "<svg/>", now - 86400)
    _aged(tmp_path / "fig_d.svg", "<svg/>", now)
    _, shadows, _, _ = cfl.scan(tmp_path)
    assert [t.parent.name for _, t in shadows] == ["svg"]


def test_a_correctly_laid_out_tree_passes(tmp_path):
    _png(tmp_path, "fig_e")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fig_e.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "svg").mkdir()
    (tmp_path / "svg" / "fig_e.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "retired").mkdir()
    (tmp_path / "retired" / "old.csv").write_text("x\n", encoding="utf-8")
    strays, shadows, n_dirs, n_png = cfl.scan(tmp_path)
    assert (strays, shadows, n_dirs, n_png) == ([], [], 1, 1)
    assert cfl.main(["prog", str(tmp_path)]) == 0


def test_it_recurses_into_figure_subdirectories(tmp_path):
    """`grant_figures/epoch/` is where the real failure was; a non-recursive scan misses it."""
    _png(tmp_path, "top")
    _png(tmp_path / "epoch", "nested")
    (tmp_path / "epoch" / "nested.csv").write_text("x\n", encoding="utf-8")
    strays, _, n_dirs, n_png = cfl.scan(tmp_path)
    assert [p.name for p in strays] == ["nested.csv"] and (n_dirs, n_png) == (2, 2)


@pytest.mark.parametrize("argv, why", [
    (["prog"], "no directory given"),
    (["prog", "!!missing!!"], "a path that does not exist"),
])
def test_it_refuses_rather_than_reporting_a_pass_on_nothing(argv, why, tmp_path):
    if len(argv) > 1:
        argv = ["prog", str(tmp_path / "does_not_exist")]
    assert cfl.main(argv) == 2, why


def test_a_directory_with_no_png_is_refused_not_passed(tmp_path):
    """Pointed at the wrong place, the tool must say so instead of printing zero problems."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "orphan.csv").write_text("x\n", encoding="utf-8")
    assert cfl.main(["prog", str(tmp_path)]) == 2


def test_every_not_sidecar_exemption_still_earns_its_place():
    """An exemption is only valid while no FIGURE owns that stem.

    If somebody later adds `exclusion_mask_painted.png`, its sidecars become real sidecars and the
    exemption starts hiding them. Asserted against the real tree, skipped when it is not mounted.
    """
    from wfield_local.paths import PathResolver
    try:
        d = pathlib.Path(PathResolver().root("labcams")) / "grant_figures"
    except Exception:                                                  # noqa: BLE001
        pytest.skip("labcams root not resolvable here")
    if not d.is_dir():
        pytest.skip(f"{d} not mounted")
    assert cfl.NOT_SIDECARS, "an empty exemption list would make this test vacuous"
    for stem in cfl.NOT_SIDECARS:
        owners = sorted(p.name for p in d.rglob(f"{stem}*.png"))
        assert not owners, f"{stem!r} is exempted but these figures own it: {owners}"


# ----------------------------------------------------------------------- the --fix repair mode

def test_repair_retires_the_copy_readers_were_getting_and_promotes_the_fresh_one(tmp_path):
    """The whole point: after the fix, readers resolve the FRESH numbers, and the old ones
    survive under `retired/` so a figure already quoted somewhere can still be explained."""
    _png(tmp_path, "fig_f")
    now = time.time()
    _aged(tmp_path / "data" / "fig_f.csv", "old\n", now - 86400)
    _aged(tmp_path / "fig_f.csv", "fresh\n", now)
    strays, shadows, _, _ = cfl.scan(tmp_path)
    cfl.repair(strays, shadows, dry_run=False)
    assert (tmp_path / "data" / "fig_f.csv").read_text(encoding="utf-8") == "fresh\n"
    assert not (tmp_path / "fig_f.csv").exists()
    retired = list((tmp_path / "data" / "retired").glob("fig_f.csv.SUPERSEDED_*"))
    assert len(retired) == 1 and retired[0].read_text(encoding="utf-8") == "old\n"
    assert cfl.scan(tmp_path)[:2] == ([], []), "repair must leave the tree clean"


def test_repair_moves_a_plain_stray_into_data(tmp_path):
    _png(tmp_path, "fig_g")
    (tmp_path / "fig_g.csv").write_text("x\n", encoding="utf-8")
    strays, shadows, _, _ = cfl.scan(tmp_path)
    cfl.repair(strays, shadows, dry_run=False)
    assert (tmp_path / "data" / "fig_g.csv").read_text(encoding="utf-8") == "x\n"
    assert cfl.scan(tmp_path)[:2] == ([], [])


def test_repair_retires_a_stray_that_LOST_to_the_data_copy(tmp_path):
    """Flat is older: the `data/` copy is current and must not be clobbered by the leftover."""
    _png(tmp_path, "fig_h")
    now = time.time()
    _aged(tmp_path / "fig_h.csv", "leftover\n", now - 86400)
    _aged(tmp_path / "data" / "fig_h.csv", "current\n", now)
    strays, shadows, _, _ = cfl.scan(tmp_path)
    cfl.repair(strays, shadows, dry_run=False)
    assert (tmp_path / "data" / "fig_h.csv").read_text(encoding="utf-8") == "current\n"
    assert len(list((tmp_path / "data" / "retired").glob("fig_h.csv.SUPERSEDED_*"))) == 1


def test_dry_run_is_the_default_and_moves_nothing(tmp_path):
    _png(tmp_path, "fig_i")
    (tmp_path / "fig_i.csv").write_text("x\n", encoding="utf-8")
    strays, shadows, _, _ = cfl.scan(tmp_path)
    cfl.repair(strays, shadows)
    assert (tmp_path / "fig_i.csv").exists() and not (tmp_path / "data").exists()


def test_repair_is_idempotent(tmp_path):
    _png(tmp_path, "fig_j")
    (tmp_path / "fig_j.csv").write_text("x\n", encoding="utf-8")
    assert cfl.main(["prog", str(tmp_path), "--fix"]) == 1
    assert cfl.main(["prog", str(tmp_path)]) == 0
    assert cfl.main(["prog", str(tmp_path), "--fix"]) == 0


def test_an_unknown_flag_is_refused_rather_than_ignored(tmp_path):
    """`--dryrun` must not silently become a real move."""
    _png(tmp_path, "fig_k")
    (tmp_path / "fig_k.csv").write_text("x\n", encoding="utf-8")
    assert cfl.main(["prog", str(tmp_path), "--fix", "--dryrun"]) == 2
    assert (tmp_path / "fig_k.csv").exists(), "nothing may move when the flags are refused"


def test_an_office_lock_file_is_not_a_stray(tmp_path):
    """`~$deck.pptx` appears while somebody has the deck open; a clean tree must stay clean."""
    _png(tmp_path, "fig_l")
    (tmp_path / "~$deck.pptx").write_bytes(b"lock")
    assert cfl.scan(tmp_path)[0] == []
    assert cfl.main(["prog", str(tmp_path)]) == 0


def test_ignore_glob_exempts_a_family_and_needs_an_argument(tmp_path):
    """`analysis_figures/*.json` are a publish SOURCE, not sidecars -- see the module docstring."""
    _png(tmp_path, "fig_m")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "fig_m.csv").write_text("x\n", encoding="utf-8")
    assert [p.name for p in cfl.scan(tmp_path)[0]] == ["fig_m.csv", "summary.json"]
    assert [p.name for p in cfl.scan(tmp_path, ["*.json"])[0]] == ["fig_m.csv"]
    assert cfl.main(["prog", str(tmp_path), "--ignore"]) == 2, "a bare --ignore must be refused"
