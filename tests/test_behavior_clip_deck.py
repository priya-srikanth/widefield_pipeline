"""The clip deck's one job is clips that PLAY.

Everything else it checks -- slide count, clips per position, poster frames, file size -- was
passing for the entire life of the module while every embedded clip was undecodable by PowerPoint.
These tests are about the gap that let that happen.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# ------------------------------------------------------------- codec (2026-09-17)

def test_the_verifier_REJECTS_the_encoding_that_shipped_for_years(tmp_path):
    """Every deck built before 2026-09-17 embedded MPEG-4 Part 2 in an AVI, which PowerPoint cannot
    decode -- and every check the module made passed anyway: right slide count, right clip count,
    right poster frames, plausible file size. None of them asked whether the bytes could be PLAYED.

    So the test that matters is not "does the verifier pass on a good deck" but "does it FAIL on the
    bad one". Built here with the exact encoder the module used to use.
    """
    import zipfile

    import cv2
    import numpy as np

    from wfield_local import behavior_clip_deck as bcd

    if bcd._ffmpeg() is None:
        pytest.skip("needs ffmpeg to probe the stream")

    bad = tmp_path / "old_style.avi"
    vw = cv2.VideoWriter(str(bad), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64))
    assert vw.isOpened()
    rng = np.random.default_rng(0)
    for _ in range(15):
        vw.write(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    vw.release()

    deck = tmp_path / "bad.pptx"
    with zipfile.ZipFile(deck, "w") as z:
        z.write(bad, "ppt/media/media1.avi")

    n, problems = bcd.verify_playable(deck)
    assert n == 1
    assert problems, "the verifier passed a deck PowerPoint cannot play"
    assert any("not .mp4" in p or "not H.264" in p for p in problems), problems


def test_the_verifier_notices_an_empty_deck():
    """A deck with no video at all is a different failure from a deck of unplayable video, and
    silently reporting 'fine' for it would hide a build that placed nothing.
    """
    import zipfile

    from wfield_local import behavior_clip_deck as bcd

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        deck = Path(td) / "empty.pptx"
        with zipfile.ZipFile(deck, "w") as z:
            z.writestr("ppt/slides/slide1.xml", "<p/>")
        n, problems = bcd.verify_playable(deck)
    assert n == 0 and problems == ["no video embedded at all"]
