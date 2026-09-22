"""`rest_frozen_decoder --from-csv` rebuilds both figures from the per-session table alone.

Verified against the real 2026-09-22 run: the three-panel figure and the confusion figure both
come back BYTE-IDENTICAL without scoring a single session, and three consecutive re-derivations
leave the inputs byte-identical too.

Two things here are less obvious than they look.

**Only `_sessions.csv` is an input.** The cohort table is DERIVED from it inside `main`, so
loading the cohort CSV as well produced eight epoch rows where the figure expected four. It was
caught by matplotlib complaining about mismatched axes -- luck, since the two had to disagree in
LENGTH for anything to notice. A derived artefact is not a second source of truth.

**`_read_conf` must index by CODE, never by position.** The live path fixes the code order up
front precisely so per-epoch matrices can be summed; reading them back positionally would
transpose a row in exactly the case the fixed order exists to prevent -- a session missing a
position.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "rest_frozen_decoder", ROOT / "scripts" / "rest_migration" / "rest_frozen_decoder.py")
rfd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rfd)

CODES = [0, 1, 2, 3, 4, 5]


def _write_conf(path, cells):
    lines = ["epoch,true,pred,count"]
    lines += [f"{e},{t},{p},{c}" for e, t, p, c in cells]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_confusion_counts_are_placed_by_code_not_by_row_order(tmp_path):
    """The file is deliberately read out of order, so a positional reader would disagree."""
    q = tmp_path / "c.csv"
    _write_conf(q, [("pre", 5, 0, 7), ("pre", 0, 5, 3), ("pre", 2, 2, 11)])
    conf = rfd._read_conf(q, ["pre"], CODES)
    m = conf["pre"]
    assert m[5, 0] == 7 and m[0, 5] == 3 and m[2, 2] == 11
    assert m.sum() == 21, "no cell may be written twice or dropped"
    assert m.dtype == np.int64


def test_an_epoch_the_run_does_not_have_is_ignored_not_invented(tmp_path):
    q = tmp_path / "c.csv"
    _write_conf(q, [("pre", 0, 0, 5), ("chronic", 1, 1, 9)])
    conf = rfd._read_conf(q, ["pre"], CODES)
    assert set(conf) == {"pre"} and conf["pre"][0, 0] == 5


def test_an_unknown_code_is_an_error_not_a_silent_drop(tmp_path):
    """A code this run does not know about means the file and the run disagree about the axes."""
    q = tmp_path / "c.csv"
    _write_conf(q, [("pre", 0, 99, 5)])
    with pytest.raises(SystemExit, match="99"):
        rfd._read_conf(q, ["pre"], CODES)


def test_an_epoch_with_no_rows_is_a_zero_matrix_not_a_missing_key(tmp_path):
    q = tmp_path / "c.csv"
    _write_conf(q, [("pre", 0, 0, 5)])
    conf = rfd._read_conf(q, ["pre", "acute"], CODES)
    assert conf["acute"].shape == (6, 6) and conf["acute"].sum() == 0


def test_the_skipped_count_round_trips(tmp_path):
    """It is in the figure's suptitle, so getting it wrong is a one-glyph difference -- exactly
    the shape of the CHRONIC_RULE caption artefact on 2026-09-22."""
    assert rfd._load_meta(tmp_path, "s") is None
    q = rfd._save_meta(tmp_path, "s", 7)
    assert q.parent.name == "data", "the meta sidecar must not land flat"
    assert rfd._load_meta(tmp_path, "s") == 7


def test_zero_skipped_is_recorded_rather_than_left_absent(tmp_path):
    """Absent means 'unknown, a run that predates this'; 0 means 'none were skipped'. They are
    different, and the run prints a warning for one of them."""
    rfd._save_meta(tmp_path, "s", 0)
    assert rfd._load_meta(tmp_path, "s") == 0


def test_from_csv_refuses_without_the_per_session_table(tmp_path, monkeypatch, capsys):
    """It must not fall through to a full decode, which is the cost the flag exists to avoid."""
    monkeypatch.setattr("sys.argv", ["rest_frozen_decoder", "--from-csv", "--out", str(tmp_path)])
    assert rfd.main() == 1
    assert "REFUSING" in capsys.readouterr().out


def test_placement_is_by_code_even_when_codes_are_not_0_to_n(tmp_path):
    """With `CODES = [0..5]` a positional reader and a code-based one agree by coincidence, so
    the test above cannot tell them apart. A non-identity code list can."""
    q = tmp_path / "c.csv"
    _write_conf(q, [("pre", 30, 10, 4), ("pre", 10, 30, 6)])
    m = rfd._read_conf(q, ["pre"], [10, 20, 30])["pre"]
    assert m.shape == (3, 3)
    assert m[2, 0] == 4 and m[0, 2] == 6, "codes must be looked up, not used as indices"
    assert m.sum() == 10
