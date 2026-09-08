"""A printed board is a physical object that cannot be corrected later, so the checks are physical.

Every one of these is a way a board could be printed, used for a whole recording session, and only
then found to be unusable -- at which point the recording is unusable too.
"""
from __future__ import annotations

import pytest

from wfield_local import dlc_board as db


def test_the_configured_board_fits_its_paper():
    """A board that overflows gets silently shrunk by the print dialog -- the one failure a printed
    ruler can catch but nothing upstream would."""
    spec = db.board_spec()
    w, h = db.check_fits(spec)
    pw, ph = db.PAPER[spec["paper"]]
    assert w < pw and h < ph


def test_an_oversized_board_is_REFUSED_not_scaled():
    with pytest.raises(ValueError, match="will not fit"):
        db.check_fits({"squares_x": 9, "squares_y": 7, "square_mm": 40.0, "marker_mm": 30.0,
                       "dictionary": "DICT_4X4_50", "paper": "a4"})


def test_the_ruler_and_mount_tab_are_accounted_for_in_the_fit():
    """They share the page. A fit check that ignored them would pass a board that then overlaps the
    tab, and tape over a marker is a missing ID -- not merely a smaller board."""
    tall = {"squares_x": 5, "squares_y": 7, "square_mm": 40.0, "marker_mm": 30.0,
            "dictionary": "DICT_4X4_50", "paper": "a4"}
    board_h = tall["squares_y"] * tall["square_mm"]                       # 280 mm
    assert board_h < db.PAPER["a4"][1], "fits the sheet on its own"
    with pytest.raises(ValueError):
        db.check_fits(tall)                                               # but not with the extras


def test_the_marker_must_sit_inside_its_square():
    with pytest.raises(ValueError, match="smaller than square_mm"):
        db.check_fits({"squares_x": 5, "squares_y": 7, "square_mm": 10.0, "marker_mm": 10.0,
                       "dictionary": "DICT_4X4_50", "paper": "a4"})


def test_the_board_clears_the_decode_threshold_where_the_OLD_one_failed():
    """The whole point of the redesign, in the units that caused it.

    cam2 imaged the 08-05 board at ~1.2 mm markers and got 2.1 px per code cell, i.e. ~10.5 px/mm at
    that working distance. The new board has to clear ~3 there with margin.
    """
    px_per_mm = 2.1 * 6 / 1.2                      # back out cam2's scale from the measurement
    got = db.px_per_bit_at(db.board_spec(), px_per_mm)
    assert got > 3.0, f"only {got} px/bit on cam2 -- no better than the board it replaces"
    assert got > 8.0, f"{got} px/bit leaves no margin for a board held further away"


def test_the_board_does_not_outgrow_the_rig():
    """cam4's field is the snout alone. A full-page board would be ~20x the current one and cam4
    would see two squares of it, which trades one unusable camera for another."""
    spec = db.board_spec()
    w, _ = db.check_fits(spec)
    assert w < 4 * db.CAGE_PLATE_MM, "wider than ~4 cage plates stops being a rig-scale target"
    assert db.n_markers(spec) >= 20, "too few markers to constrain a pose well"


def test_the_marker_count_matches_the_dictionary():
    """DICT_4X4_50 holds 50 markers; a board needing more would silently repeat IDs, and a repeated
    ID is a pose solution that can land in two places."""
    assert db.n_markers(db.board_spec()) <= 50
    assert db.n_markers(db.board_spec("a3")) <= 50


def test_the_dictionary_matches_the_board_already_in_use():
    """dlc_calibration hard-codes DICT_4X4_50 from measurement; a new board in another dictionary
    would read as no board at all."""
    from wfield_local.dlc_calibration import ARUCO_DICT

    assert db.board_spec()["dictionary"] == ARUCO_DICT


def test_the_mount_tab_is_at_least_a_cage_plate_deep():
    assert db.MOUNT_TAB_MM >= db.CAGE_PLATE_MM


# --------------------------------------------------------------------- working distance

#: cam4 (tight, 680 px snout view) and cam2 (wide, whole animal), at the distance the 08-05 board
#: was held: 30.4 px and 13.1 px across the same 1.2 mm marker.
CAM4_PXMM, CAM4_PX, CAM2_PXMM = 25.0, 680, 10.9


def test_the_OLD_board_had_no_usable_window_at_the_distance_it_was_used():
    """The failure, in one number. Its window is 0.18-0.73x the distance it was actually held at --
    it excludes 1.0, so there was no way to record a usable calibration with it where it was."""
    _lo, hi = db.working_window({"squares_x": 9, "squares_y": 7, "square_mm": 1.6,
                                 "marker_mm": 1.2}, CAM4_PXMM, CAM4_PX, CAM2_PXMM)
    assert hi < 1.0, "the board could only work held much closer than it was"


def test_the_new_board_window_contains_a_workable_standoff():
    lo, hi = db.working_window(db.board_spec(), CAM4_PXMM, CAM4_PX, CAM2_PXMM)
    assert lo < 1.5, "needing >1.5x standoff starts to leave the rig"
    assert hi / lo > 3.0, "less than 3x of latitude is hard to hold by hand"


def test_making_the_board_bigger_MOVES_the_window_it_does_not_widen_it():
    """Why "just print it huge" is not the answer.

    Both bounds scale linearly with the board, so the ratio is invariant: a 4x board has the same
    ~4x of latitude, just centred 4x further away -- which may be outside the enclosure. Size picks
    WHERE you stand, not how forgiving the setup is.
    """
    small = db.working_window(db.board_spec(), CAM4_PXMM, CAM4_PX, CAM2_PXMM)
    big = db.working_window({"squares_x": 7, "squares_y": 5, "square_mm": 42.0, "marker_mm": 32.0},
                            CAM4_PXMM, CAM4_PX, CAM2_PXMM)
    assert big[0] > small[0] and big[1] > small[1], "a bigger board must be held further away"
    assert abs((big[1] / big[0]) - (small[1] / small[0])) < 0.2, "the latitude is size-invariant"


def test_a_board_with_no_window_at_all_is_reported_as_such():
    """If the tight camera cannot see 3 squares before the wide one stops decoding, no distance
    works and the pair cannot be calibrated with that board at any standoff."""
    lo, hi = db.working_window({"squares_x": 5, "squares_y": 5, "square_mm": 200.0,
                                "marker_mm": 1.0}, CAM4_PXMM, CAM4_PX, CAM2_PXMM)
    assert hi == 0.0 and lo == float("inf")
