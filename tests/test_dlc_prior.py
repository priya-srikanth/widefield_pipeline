"""Tests for the cam4 inference spatial prior (wfield_local.dlc_prior).

The prior's whole value is that it changes NOTHING on ordinary frames and rescues the rare
confident-wrong one, so the tests that matter are the ones pinning each half of that.
"""
import pytest

from wfield_local import dlc_prior

torch = pytest.importorskip("torch")
# DeepLabCut lives ONLY in the `dlc` env (README "Per-machine environments"), so these skip rather
# than error on the analysis box -- same rule as the other dlc test modules. Importing it inside the
# fixture instead would raise at collection and report as an ERROR, which reads like a broken test.
_sp = pytest.importorskip(
    "deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor")


class _FakePredictor:
    """Stands in for DLC's ``HeatmapPredictor``: returns the argmax of whatever heatmap it is given.

    Deliberately NOT the real class — the point under test is the masking, and using DLC's decoder
    here would make the test depend on locref refinement and sigmoid scaling that have nothing to do
    with it. The patch target is monkeypatched onto this in `_patched`.
    """

    def forward(self, stride, outputs):
        hm = outputs["heatmap"]
        n, k, h, w = hm.shape
        flat = hm.reshape(n, k, -1).argmax(-1)
        return {"poses": torch.stack([(flat % w) * stride, (flat // w) * stride], -1).float()}


@pytest.fixture
def patchable(monkeypatch):
    monkeypatch.setattr(_sp.HeatmapPredictor, "forward", _FakePredictor.forward, raising=False)
    return _sp


def _heatmap(peaks, k=2, h=16, w=16):
    """``peaks`` = list of (channel, cell_y, cell_x, value)."""
    hm = torch.zeros(1, k, h, w)
    for c, y, x, v in peaks:
        hm[0, c, y, x] = v
    return {"heatmap": hm}


def test_the_prior_lets_the_second_peak_win_rather_than_dropping_the_frame(patchable):
    """The spout-on-fur case: a HIGHER peak outside the box, a real one inside.

    Rejecting after the fact would score a dropout. Masking before the argmax recovers the true
    point, which is the entire reason this is done at the heatmap and not on the output table.
    Measured on nine real frames 2026-09-26: (103-109, 527) at p 0.69-0.83 -> (345-374, 481-486).
    """
    box = {"a": (0, 40, 0, 40), "b": (0, 128, 0, 128)}
    out = _heatmap([(0, 1, 1, 9.0), (0, 10, 10, 5.0)])        # cell 1,1 = px 8,8 (inside)
    with dlc_prior.masking(box, bodyparts=["a", "b"]):
        poses = patchable.HeatmapPredictor().forward(8, out)["poses"]
    assert tuple(poses[0, 0].tolist()) == (8.0, 8.0), "the in-box peak should win"

    out = _heatmap([(0, 1, 1, 5.0), (0, 10, 10, 9.0)])        # the BIGGER peak is now outside
    with dlc_prior.masking(box, bodyparts=["a", "b"]):
        poses = patchable.HeatmapPredictor().forward(8, out)["poses"]
    assert tuple(poses[0, 0].tolist()) == (8.0, 8.0), "outside peak must be suppressed, not chosen"


def test_without_the_prior_the_spurious_peak_wins(patchable):
    """The control: the same heatmap, unmasked, picks the impossible point. Without this the test
    above would pass just as well against a prior that did nothing at all."""
    out = _heatmap([(0, 1, 1, 5.0), (0, 10, 10, 9.0)])
    poses = patchable.HeatmapPredictor().forward(8, out)["poses"]
    assert tuple(poses[0, 0].tolist()) == (80.0, 80.0)


def test_the_patch_is_removed_even_when_the_body_raises(patchable):
    """A left-behind monkeypatch would silently change every later run in the same process."""
    before = patchable.HeatmapPredictor.forward
    with pytest.raises(ValueError):
        with dlc_prior.masking({"a": (0, 40, 0, 40), "b": (0, 128, 0, 128)}, bodyparts=["a", "b"]):
            raise ValueError("boom")
    assert patchable.HeatmapPredictor.forward is before


def test_a_channel_count_mismatch_refuses_instead_of_masking_the_wrong_part(patchable):
    """Boxes are applied BY POSITION, so a list that disagrees with the head would mask each part
    with another part's box and return plausible, wrong coordinates. That must raise."""
    out = _heatmap([(0, 1, 1, 9.0)], k=2)
    with pytest.raises(SystemExit):
        with dlc_prior.masking({"a": (0, 40, 0, 40)}, bodyparts=["a"]):
            patchable.HeatmapPredictor().forward(8, out)


def test_a_partial_box_set_refuses_rather_than_masking_some_parts(patchable):
    with pytest.raises(SystemExit):
        with dlc_prior.masking({"a": (0, 40, 0, 40)}, bodyparts=["a", "b"]):
            pass


def test_the_offset_moves_the_box_into_crop_coordinates(patchable):
    """Boxes are full-frame; a cropped heatmap is not. Getting this wrong masks the wrong region and
    deletes real peaks silently, so it is pinned rather than trusted."""
    box = {"a": (100, 140, 100, 140), "b": (0, 1e4, 0, 1e4)}
    out = _heatmap([(0, 1, 1, 9.0)])                      # cell 1,1 -> px 8,8 in CROP coords
    with dlc_prior.masking(box, bodyparts=["a", "b"]):    # no offset: 8,8 is outside 100-140
        p_no = patchable.HeatmapPredictor().forward(8, out)["poses"]
    with dlc_prior.masking(box, bodyparts=["a", "b"], offset=(100, 100)):
        p_off = patchable.HeatmapPredictor().forward(8, out)["poses"]
    assert tuple(p_off[0, 0].tolist()) == (8.0, 8.0), "with the offset the peak is inside the box"
    # Without the offset the real peak is masked away and the argmax falls on the first surviving
    # cell inside the (wrongly placed) box -- px 104, i.e. the true detection is LOST. That is the
    # failure this parameter exists to prevent, so it is asserted exactly rather than as "not 8".
    assert tuple(p_no[0, 0].tolist()) == (104.0, 104.0)


def test_an_empty_box_refuses_rather_than_returning_a_corner(patchable):
    """A box that covers no heatmap cell would mask everything and make the argmax meaningless —
    it would still return a coordinate, which is what makes this worth refusing."""
    out = _heatmap([(0, 1, 1, 9.0)])
    with pytest.raises(SystemExit):
        with dlc_prior.masking({"a": (5000, 5001, 5000, 5001), "b": (0, 1e4, 0, 1e4)},
                               bodyparts=["a", "b"]):
            patchable.HeatmapPredictor().forward(8, out)


def test_pad_is_read_from_config_not_hardcoded():
    assert dlc_prior.pad() > 0
