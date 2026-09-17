"""A reference needs THREE things to land together, and twice only two did.

`position_reference_maps.REFERENCES` is iterated by `maps_by_epoch`, dispatched by
`reference_maps`, and captioned by `epoch_grant_figures._REF_TEXT`. Miss any one and the failure is
quiet in a specific, expensive way:

* **`restw`, 2026-09-13** -- named in REFERENCES ahead of its builder. `reference_maps` raises on an
  unknown name, so EVERY 15r render went down for twenty minutes.
* **`precue`, 2026-09-13** -- added to REFERENCES with no `_REF_TEXT` entry. The KeyError fired
  per-arm AFTER the earlier references were written, so the render **reported exit 0** and produced
  two references where three were asked for. It was found by counting files (21 against an expected
  39), not by the exit status.

The second is the dangerous one: a render that succeeds while silently producing less than it was
asked for looks exactly like a render that worked. These tests are cheap and make the registry
self-enforcing, so `raw` (2026-09-17) and everything after it cannot repeat either failure.
"""
from __future__ import annotations

import pytest

from wfield_local.epoch_grant_figures import _REF_TEXT
from wfield_local.position_reference_maps import REFERENCES, reference_maps


@pytest.mark.parametrize("ref", REFERENCES)
def test_every_reference_has_a_caption(ref):
    """THE `precue` FAILURE. A missing key raises mid-render, after earlier files are written."""
    assert ref in _REF_TEXT, (
        f"{ref!r} is in REFERENCES with no _REF_TEXT entry -- 15r will raise per-arm AFTER "
        f"writing the earlier references and still exit 0")


@pytest.mark.parametrize("ref", REFERENCES)
def test_every_caption_carries_the_fields_the_render_reads(ref):
    for field in ("short", "title", "cbar", "note"):
        assert field in _REF_TEXT[ref], f"_REF_TEXT[{ref!r}] is missing {field!r}"
        assert str(_REF_TEXT[ref][field]).strip(), f"_REF_TEXT[{ref!r}][{field!r}] is empty"


@pytest.mark.parametrize("ref", REFERENCES)
def test_every_reference_dispatches(ref):
    """THE `restw` FAILURE. A name in the tuple with no branch takes down every render."""
    parts = {k: {0: 1.0} for k in ("raw", "raw_rest", "raw_restw", "raw_precue")}
    parts["trial_mean"] = 0.0
    try:
        reference_maps(parts, ref)
    except ValueError as ex:                                           # pragma: no cover
        pytest.fail(f"{ref!r} is in REFERENCES but reference_maps rejects it: {ex}")
    except Exception:                                                  # noqa: BLE001
        # Any other error is about THIS fixture's shape, not a missing branch, which is all this
        # test claims to check.
        pass


def test_no_caption_without_a_reference():
    """The mirror: a caption for a name no longer in REFERENCES is a retired entry that will
    mislead the next reader into thinking the family is still produced."""
    orphans = sorted(set(_REF_TEXT) - set(REFERENCES))
    assert not orphans, f"_REF_TEXT has entries for non-references: {orphans}"
