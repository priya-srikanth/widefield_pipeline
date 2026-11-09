"""iti_where_decoder under a SUBSET ITI filter draws only filtered data (bug B1, Priya 11-09).

Before: with WFIELD_ITI_FILTER=dock (or dock01, gap2dock, far1s, notdock, notdock_n) main / --plot-only decoded / read
EVERY ITI lick (where_decoder.csv under OUT) and saved that figure into the subset folder. Now those paths refuse; --z
draws the subset r table of iti_dock_where_preds.py and is allowed only when that copy exists under OUT_W.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def wd(monkeypatch, tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import iti_licks as il
        import iti_where_decoder as wd
    finally:
        sys.path.remove(str(REPO / "scripts"))
    monkeypatch.setattr(il, "OUT", tmp_path / "out")
    monkeypatch.setattr(il, "OUT_W", tmp_path / "out" / "itifilter_dock")
    return wd, il


def test_no_filter_never_refuses(wd, monkeypatch):
    wd_, il = wd
    monkeypatch.setattr(il, "ITI_SUBSET", "")
    assert wd_.subset_guard(False) is None and wd_.subset_guard(True) is None


def test_filter_refuses_unfiltered_draws(wd, monkeypatch):
    wd_, il = wd
    monkeypatch.setattr(il, "ITI_SUBSET", "dock")
    with pytest.raises(SystemExit, match="unfiltered figure"):
        wd_.subset_guard(False)
    with pytest.raises(SystemExit, match="unfiltered figure"):        # plot() itself, non-z (main / --plot-only)
        wd_.plot(None, None)


def test_filter_z_needs_the_subset_copy(wd, monkeypatch):
    wd_, il = wd
    monkeypatch.setattr(il, "ITI_SUBSET", "dock")
    with pytest.raises(SystemExit, match="missing"):
        wd_.subset_guard(True)
    f = Path(il.OUT_W) / "zscore_check" / il.tagged("where_decoder_z.csv")
    f.parent.mkdir(parents=True)
    f.write_text("source,set\n")
    assert wd_.subset_guard(True) == f
    # the same table under OUT (the unfiltered primary) never counts as the subset copy
    monkeypatch.setattr(il, "OUT_W", il.OUT)
    with pytest.raises(SystemExit):
        wd_.subset_guard(True)


def test_cli_plot_only_refuses_before_writing(tmp_path):
    # PYTHONPATH = THIS checkout first: the installed (editable) wfield_local may point at another checkout
    pp = os.pathsep.join([str(REPO), str(REPO / "scripts")] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    env = dict(os.environ, WFIELD_ITI_FILTER="dock", WFIELD_ITI_OUT_W=str(tmp_path / "w"), MPLBACKEND="Agg",
               PYTHONPATH=pp)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "iti_where_decoder.py"), "--plot-only"], env=env,
                       cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "unfiltered figure" in r.stderr
    assert not any((tmp_path / "w").rglob("*.png"))
