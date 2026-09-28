"""The spatial prior is switched on PER PART (`dlc.prior.parts`), not all-or-nothing.

2026-09-28: on for tongue and jaw; nose and spout unmasked until a full-session run has checked their
boxes. `apply()` is the one entry point, so an inference path never has to know which.
"""
from __future__ import annotations

import pytest

from wfield_local import dlc_prior

BOX = {"nose": (0, 1, 0, 1), "jaw": (0, 2, 0, 2), "tongue": (0, 3, 0, 3), "spout": (0, 4, 0, 4)}


def _cfg(monkeypatch, **prior):
    monkeypatch.setattr(dlc_prior, "_cfg", lambda: prior)
    monkeypatch.setattr(dlc_prior, "boxes",
                        lambda rv=None, cam=None, padding=None, frame=(680, 680): dict(BOX))


def test_off_means_no_boxes(monkeypatch):
    _cfg(monkeypatch, enabled=False, parts=["tongue"])
    assert dlc_prior.active() is None


def test_parts_filters_the_boxes(monkeypatch):
    _cfg(monkeypatch, enabled=True, parts=["tongue", "jaw"])
    assert set(dlc_prior.active()) == {"tongue", "jaw"}


def test_no_parts_means_every_part(monkeypatch):
    _cfg(monkeypatch, enabled=True)
    assert set(dlc_prior.active()) == set(BOX)


def test_a_part_that_is_not_trained_is_refused(monkeypatch):
    _cfg(monkeypatch, enabled=True, parts=["whisker"])
    with pytest.raises(SystemExit, match="whisker"):
        dlc_prior.active()


def test_the_repo_config_is_on_for_tongue_and_jaw_only():
    assert dlc_prior.enabled() is True
    assert dlc_prior.parts() == ["tongue", "jaw"]
