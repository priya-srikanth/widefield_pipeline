"""The mirror test must not call noise a relocation.

"More right-hemisphere activity post-stroke" has two very different readings: the right hemisphere
doing more of its own thing, or the LEFT hemisphere's pattern having MOVED across the midline. Only
the second deserves the word TRANSFER, and it is the stronger and more interesting claim, so the bar
for saying it has to be set deliberately rather than by whichever correlation happens to be larger.

The three cases below are the three real ones this criterion has already got wrong or right, kept as
regression tests because each version of the rule looked reasonable until a number was put through it.
"""
from __future__ import annotations

from wfield_local.spatial_reorganisation import MIRROR_MARGIN, _flag_mirror


def test_a_hair_of_difference_is_not_a_relocation():
    """PS94 8/18 far_center, the case that broke the previous rule.

    `mirror_r > normal_r` with no margin flagged TRANSFER on normal +0.677 against mirror +0.682 --
    a 0.005 correlation difference, and the pre-stroke baseline difference at that position is
    -0.015, so essentially nothing moved at all. Reporting it would have put a spurious midline
    transfer in the deck for two of four animals.
    """
    r = _flag_mirror({"normal_r": 0.677, "mirror_r": 0.682,
                      "mirror_minus_normal": 0.005, "pre_mirror_minus_normal": -0.015})
    assert not r["transfer"]
    assert not r["reduced_asymmetry"]


def test_a_real_relocation_is_still_called_one():
    """The rule has to keep the power to say yes, or it is just a way of never reporting anything."""
    r = _flag_mirror({"normal_r": 0.20, "mirror_r": 0.70,
                      "mirror_minus_normal": 0.50, "pre_mirror_minus_normal": -0.10})
    assert r["transfer"]
    assert r["reduced_asymmetry"]


def test_less_asymmetric_is_not_relocated():
    """PS94 close_R under the FIRST rule, which asked only for a shift of >0.15 toward mirror.

    Normal +0.718 against mirror +0.323: the pattern had not gone anywhere, it had become less
    asymmetric. Those are different claims about the brain and only one of them is transfer. The
    weaker claim keeps its own flag rather than being silently promoted.
    """
    r = _flag_mirror({"normal_r": 0.718, "mirror_r": 0.323,
                      "mirror_minus_normal": -0.395, "pre_mirror_minus_normal": -0.60})
    assert not r["transfer"], "mirror is far BELOW normal; the pattern stayed where it was"
    assert r["reduced_asymmetry"]


def test_transfer_implies_reduced_asymmetry_never_the_reverse():
    """Transfer is the strictly stronger claim, so it must be a subset. Guards against a future
    edit that loosens one flag without noticing it has crossed the other."""
    cases = [
        {"normal_r": 0.1, "mirror_r": 0.9, "mirror_minus_normal": 0.8,
         "pre_mirror_minus_normal": 0.0},
        {"normal_r": 0.9, "mirror_r": 0.1, "mirror_minus_normal": -0.8,
         "pre_mirror_minus_normal": -0.9},
        {"normal_r": 0.5, "mirror_r": 0.5, "mirror_minus_normal": 0.0,
         "pre_mirror_minus_normal": 0.0},
    ]
    for c in cases:
        r = _flag_mirror(dict(c))
        if r["transfer"]:
            assert r["reduced_asymmetry"], f"transfer without reduced_asymmetry: {c}"


def test_the_margin_is_actually_applied():
    """A difference just under the margin must not flag; just over must. Pins the boundary so the
    constant cannot be quietly zeroed."""
    base = {"normal_r": 0.2, "mirror_r": 0.9, "pre_mirror_minus_normal": 0.0}
    under = _flag_mirror({**base, "mirror_minus_normal": MIRROR_MARGIN - 0.01})
    over = _flag_mirror({**base, "mirror_minus_normal": MIRROR_MARGIN + 0.01})
    assert not under["transfer"]
    assert over["transfer"]
