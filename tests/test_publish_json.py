"""The analysis JSON must reach MICROSCOPE, and a FROZEN artifact must never be overwritten there.

2026-08-25: `_publish_figs` globs `*.png`, so all 125 JSON artifacts in `figures_working` lived only
on the box that produced them. Two matter beyond convenience:

  coding_direction.json            the numbers behind section G9 and the miss-while-working vs
                                   stopped result recorded in DECISIONS
  nolick_reference_prestroke.json  a reference the pipeline FREEZES on purpose -- nightly_figs:
                                   "written once ... and never overwritten -- a reference that moves
                                   after the comparison data arrive is not a reference"

A frozen reference held only on local disk is unrecoverable: regenerating it over today's grown
session set yields a DIFFERENT reference, which is the exact thing the freeze exists to prevent. And
with two analysis boxes each holding their own copy, publishing with a plain last-writer-wins copy
would let whichever box ran second silently redefine the reference the other one's results were
computed against. That is why divergence is REPORTED and the destination left alone, rather than
resolved by overwriting.
"""
import json

import pytest

from wfield_local import nightly_figs as nf


class _RV:
    """Minimal resolver stand-in: keeps the copy off the real MICROSCOPE share."""

    def __init__(self, dst):
        self._dst = str(dst)

    def root(self, name):
        assert name == "cue_analysis_out"
        return self._dst


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(nf.writeguard, "assert_writable", lambda p: None, raising=False)
    src, dst = tmp_path / "work", tmp_path / "share"
    src.mkdir()
    return src, dst, _RV(dst)


def _write(p, obj):
    p.write_text(json.dumps(obj))


def test_json_is_published_and_grouped(dirs):
    src, dst, rv = dirs
    _write(src / "coding_direction.json", {"a": 1})
    _write(src / "spatial_reorganisation_cue_all.json", {"b": 2})
    _write(src / "something_unrecognised.json", {"c": 3})
    (src / "a_figure.png").write_bytes(b"not json")

    res = nf._publish_json(src, rv, log=lambda *_: None)

    assert res["copied"] == 3
    assert (dst / "analysis_json" / "coding_directions" / "coding_direction.json").exists()
    assert (dst / "analysis_json" / "spatial" / "spatial_reorganisation_cue_all.json").exists()
    # An unrecognised artifact must be FILED, never dropped -- silently skipping the one file nobody
    # thought to route is how an artifact goes missing without any count changing.
    assert (dst / "analysis_json" / "other" / "something_unrecognised.json").exists()
    assert not (dst / "analysis_json" / "other" / "a_figure.png").exists()


def test_a_frozen_reference_is_never_overwritten(dirs):
    """The whole point. Two boxes, two different frozen references, and the published one wins."""
    src, dst, rv = dirs
    published = dst / "analysis_json" / "references" / "nolick_reference_prestroke.json"
    published.parent.mkdir(parents=True)
    _write(published, {"frozen": "FIRST BOX", "computed": "2026-08-19"})

    _write(src / "nolick_reference_prestroke.json", {"frozen": "SECOND BOX", "computed": "2026-08-25"})
    seen = []
    res = nf._publish_json(src, rv, log=lambda m: seen.append(m))

    assert res["copied"] == 0, "a frozen artifact was overwritten"
    assert res["frozen_conflicts"] == ["nolick_reference_prestroke.json"]
    assert json.loads(published.read_text())["frozen"] == "FIRST BOX"
    assert any("diverged" in m for m in seen), "divergence must be reported, not silent"


def test_an_identical_frozen_reference_is_a_quiet_skip(dirs):
    """Agreement is the normal case and must not cry wolf -- a guard that fires on a healthy run
    gets ignored, which is how the joint-basis rank check briefly became worthless."""
    src, dst, rv = dirs
    payload = {"frozen": "same on both boxes"}
    published = dst / "analysis_json" / "references" / "nolick_reference_prestroke.json"
    published.parent.mkdir(parents=True)
    _write(published, payload)
    _write(src / "nolick_reference_prestroke.json", payload)

    res = nf._publish_json(src, rv, log=lambda *_: None)

    assert res["frozen_conflicts"] == []
    assert res["skipped"] == 1 and res["copied"] == 0


def test_a_frozen_reference_is_created_when_absent(dirs):
    """Never-overwrite must not become never-publish; the first box to run still seeds it."""
    src, dst, rv = dirs
    _write(src / "nolick_reference_prestroke.json", {"frozen": "first"})
    res = nf._publish_json(src, rv, log=lambda *_: None)
    assert res["copied"] == 1
    assert (dst / "analysis_json" / "references" / "nolick_reference_prestroke.json").exists()


def test_ordinary_json_still_updates(dirs):
    """Only the frozen set is protected -- coding_direction.json legitimately changes every night."""
    src, dst, rv = dirs
    d = dst / "analysis_json" / "coding_directions" / "coding_direction.json"
    d.parent.mkdir(parents=True)
    _write(d, {"old": True})
    _write(src / "coding_direction.json", {"new": True, "padding": "x" * 50})

    nf._publish_json(src, rv, log=lambda *_: None)
    assert json.loads(d.read_text()) == {"new": True, "padding": "x" * 50}
