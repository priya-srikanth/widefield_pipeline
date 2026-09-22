"""The guide is read by someone who did not build any of this, so the failure modes are social.

The one that matters: sending the labeller to the TRAINING copy. `dlc_train.stage()` overwrites that
copy's labels from the labelling project on every run, so a correction made there is destroyed by the
next retrain -- silently, because the file is well-formed either way. Everything else here is about
the page not going stale or not rendering.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from wfield_local import dlc_review_guide as rg
from wfield_local import dlc_train as dt

from tests.test_dlc_train import PARTS, _labels, _preds, _write_preds, projects  # noqa: F401


@pytest.fixture
def built(projects, tmp_path):  # noqa: F811 -- pytest fixture requested by name
    """A project with three flagged frames, one of them a whole lick, plus the crops."""
    cv2 = pytest.importorskip("cv2")
    src, _train = projects
    proj = dt.stage()
    truth = dt.labels(proj)
    # rows 0..3 are one session's four lick frames (see the manifest in the dlc_train fixture)
    _write_preds(proj, _preds(truth, {
        (0, "tongue"): (300.0, 0.0, 0.01),     # DELETE
        (1, "jaw"): (20.0, 0.0, 0.95),         # REPLACE  }- same lick
        (2, "jaw"): (20.0, 0.0, 0.95),         # REPLACE  }
    }))
    for f in (proj / "labeled-data").rglob("img*.png"):
        cv2.imwrite(str(f), np.full((680, 680, 3), 90, np.uint8))
    dt.review_images(proj)
    dest = tmp_path / "CORRECTION_GUIDE.html"
    # `src` (the LABELLING project) comes back too, so tests that need it do not have to
    # re-request the imported `projects` fixture -- a pytest-idiomatic parameter that ruff
    # reads as shadowing the import (F811).
    return proj, rg.build(proj, dest=dest), dest.read_text(encoding="utf-8"), src


def test_the_guide_sends_the_labeller_to_the_LABELLING_project_not_the_training_copy(built):
    """The one mistake that destroys work rather than wasting time."""
    proj, path, s, src = built
    assert "dlc_project --cam cam4 --label --folder" in s
    assert "dlc_train" not in re.sub(r"dlc_review_guide|<code>python -m wfield_local\.dlc_train", "", s)
    # and it warns about the trap by name
    assert "training" in s and "overwritten" in s


def test_every_flagged_frame_appears_with_its_picture(built):
    proj, path, s, src = built
    d = dt.classify(proj)
    assert len(d) == 3
    for _, r in d.iterrows():
        assert r.image in s, f"{r.image} missing from the guide"
    assert s.count("data:image/png;base64,") == len(d)
    assert "background:var(--sunk);border-radius:6px" not in s, "a crop failed to embed"


def test_frames_of_one_lick_are_grouped_so_they_are_done_together(built):
    """Four offsets are one ~80 ms protrusion; fixing one in isolation makes it disagree with its
    neighbours, which is why `dlc.frames` pruning keeps a lick whole in the first place."""
    proj, path, s, src = built
    assert 'class="lick"' in s
    assert "same tongue flick" in s


def test_the_guide_names_the_folder_for_every_session_that_has_work(built):
    proj, path, s, src = built
    for stem in dt.classify(proj).session.unique():
        assert f"--folder {stem}" in s, f"no command for {stem}"


def test_the_page_is_well_formed_and_standalone(built):
    import html.parser

    proj, path, s, src = built
    assert s.startswith("<!doctype html>") and s.rstrip().endswith("</html>")
    assert "<style>" in s and "http://" not in s and "https://" not in s   # no external fetches

    class P(html.parser.HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack, self.bad = [], []

        def handle_starttag(self, t, a):
            if t not in ("img", "br", "meta", "input", "hr", "link"):
                self.stack.append(t)

        def handle_endtag(self, t):
            if self.stack and self.stack[-1] == t:
                self.stack.pop()
            elif t in self.stack:
                self.bad.append(t)
                self.stack.remove(t)

    p = P()
    p.feed(s)
    assert not p.stack and not p.bad, f"unclosed={p.stack[:3]} mismatched={p.bad[:3]}"


def test_a_clean_round_produces_a_page_that_says_so_rather_than_a_stale_list(projects, tmp_path):  # noqa: F811
    """The guide is regenerated every round, so 'nothing to fix' must render as nothing to fix --
    not as last round's list, which would send someone to move points already moved."""
    proj = dt.stage()
    _write_preds(proj, _preds(dt.labels(proj), {}))
    dest = tmp_path / "g.html"
    rg.build(proj, dest=dest)
    s = dest.read_text(encoding="utf-8")
    assert "Nothing to fix" in s
    assert "data:image/png;base64," not in s


def test_the_guide_does_not_send_her_back_to_ask_about_the_tongue_landmark(built):
    """Settled 2026-09-22, so the page applies the rule rather than deferring to it.

    It used to say "ask Priya once, write the answer down". Leaving that in after the decision was
    made would stall the one part the whole exercise is about, waiting on a conversation that has
    already happened.
    """
    proj, path, s, src = built
    assert "DECIDE" in s and "landmark" in s
    for stale in ("ask Priya", "Do not decide the tongue landmark", "has not been settled"):
        assert stale not in s, f"{stale!r} outlived the decision"


# ------------------------------------------------------------------- it is read on a MacBook

def test_the_guide_gives_a_PASTEABLE_MAC_PATH_for_every_session(built):
    """On her Mac she has napari + napari-deeplabcut and NOT DeepLabCut and NOT this repo, so
    `python -m wfield_local.dlc_project --label` is not a command she can run. The Mac route is
    File -> Open Folder..., which needs a PATH, not a command line."""
    proj, path, s, src = built
    for stem in dt.classify(proj).session.unique():
        want = f"{rg.MAC_ROOT}/widefield-Priya-2026-09-08/labeled-data/{stem}"
        assert want in s, f"no pasteable Mac path for {stem}"


def test_the_mac_route_is_primary_and_the_rig_route_is_tucked_away(built):
    proj, path, s, src = built
    assert "conda activate label" in s and "napari" in s
    assert "Open Folder" in s and "Cmd</kbd>+<kbd>Shift</kbd>+<kbd>G" in s
    # the rig command still exists, but only inside a collapsed <details>
    assert "dlc_project --cam cam4 --label" in s
    for m in re.finditer(r"dlc_project --cam cam4 --label", s):
        assert "<details>" in s[:m.start()][-400:], "rig command is not inside a <details>"


def test_no_windows_only_instructions_survive(built):
    """A Windows path or `Anaconda Prompt` on this page is an instruction she cannot follow."""
    proj, path, s, src = built
    for bad in ("Anaconda Prompt", r"C:\Users", "conda activate dlc"):
        assert bad not in s, f"{bad!r} is a rig-only instruction"
    assert "Ctrl</kbd>+<kbd>S" not in s, "Ctrl+S is the Windows shortcut; on a Mac it is Cmd+S"


def test_it_warns_about_the_two_folders_that_look_identical(built):
    """`_frame_staging` holds copies of the same images under the same names and only fails at SAVE
    time; a doubled `Neurobio-1` mount is the usual cause of 'that path does not exist'."""
    proj, path, s, src = built
    assert "_frame_staging" in s
    assert "Neurobio-1" in s


def test_the_guide_states_the_REAL_per_part_training_counts(built):
    """The first draft said the network "learned this landmark from your other ~190 frames" for
    every part. True of nose (192) and spout (187); for the TONGUE it is 71, because the tongue is
    only out for a fraction of the time -- a ~3x overstatement on the one part most flagged frames
    are about, biased toward trusting the network over the labeller's own eyes.
    """
    proj, path, s, src = built
    counts = rg._training_counts(proj)
    assert counts, "no per-part counts"
    for bp, (n_train, n_all) in counts.items():
        assert f"<td>{n_train}</td><td>{n_all}</td>" in s, f"{bp} counts missing from the table"
    assert "~190" not in s, "the blanket frame count is back"
    # a part the network has barely seen must be called out, not averaged away
    weak = [bp for bp, (n, _) in counts.items() if n < 100]
    if weak:
        assert "trust your own eyes first" in s


def test_the_guide_does_not_claim_a_single_spacing_between_lick_frames(built):
    """`dlc.frames.lick_offsets_s` is -16/0/+32/+64 ms, so consecutive frames are 16, 32 and 32 ms
    apart and a group may hold 2, 3 or 4 of them. "16 ms apart" is wrong on most pairs."""
    proj, path, s, src = built
    assert "16 ms apart" not in s and "16&nbsp;ms apart" not in s
    assert "80&nbsp;ms" in s or "80 ms" in s


def test_the_zoom_scale_reference_is_one_we_actually_measured(built):
    """It used to say 12 px is "about a fifth of the tongue's width" -- a number nobody measured.
    The nose-to-jaw distance is in the labels and is 174 px."""
    proj, path, s, src = built
    assert "fifth of the tongue" not in s
    assert "174" in s and "680" in s


# ------------------------------------------------------------------ new frames added to label

def test_no_new_frames_section_when_everything_is_labelled(built):
    """The section is dormant until frames are actually extracted -- it must not appear empty."""
    proj, path, s, src = built
    assert "New frames to label" not in s


def _add_blank_frames(live, stem, names):
    cv2 = pytest.importorskip("cv2")
    for n in names:
        cv2.imwrite(str(live / "labeled-data" / stem / n), np.full((680, 680, 3), 70, np.uint8))


def test_new_unlabelled_frames_are_listed_with_their_folder_path(built, tmp_path):
    """After extraction the new frames are blank images sitting in folders she already knows, so
    the page has to name them and repeat the pasteable path."""
    proj, path, s, src = built
    _add_blank_frames(src, "cam4_pre_a", ["img0000900.png", "img0000901.png"])

    dest = tmp_path / "g2.html"
    rg.build(proj, dest=dest)
    s2 = dest.read_text(encoding="utf-8")

    assert "New frames to label (2)" in s2
    assert "img0000900.png" in s2 and "img0000901.png" in s2
    assert f"{rg.MAC_ROOT}/widefield-Priya-2026-09-08/labeled-data/cam4_pre_a" in s2
    assert "new frames to check" in s2               # flagged in the headline too
    assert "They are <b>blank</b>" in s2             # nothing seeded them yet


def test_new_frames_section_says_place_ONLY_the_four_trained_parts(built, tmp_path):
    """The main guide asks for ten parts on cam4. Here it is four, because the whisker points are
    donor output rather than human labels and more of them trains nothing."""
    proj, path, s, src = built
    _add_blank_frames(src, "cam4_pre_a", ["img0000902.png"])
    dest = tmp_path / "g3.html"
    rg.build(proj, dest=dest)
    s3 = dest.read_text(encoding="utf-8")

    for bp in PARTS:
        assert f"<code>{bp}</code>" in s3
    assert "Leave the whiskers and the eyes <b>empty</b>" in s3
    assert "If a part is not visible, leave it blank" in s3
    assert "Shift</kbd>+<kbd>&rarr;" in s3            # how to FIND them


def test_a_frame_whose_tongue_alone_is_blank_is_not_called_unlabelled(projects):  # noqa: F811
    """The tongue is legitimately absent on most frames. Treating 'tongue missing' as 'not done'
    would send her back through work she has finished, every round, forever."""
    src, _ = projects
    dt.stage()
    live = src
    h5 = live / "labeled-data" / "cam4_pre_a" / f"CollectedData_{dt.SCORER}.h5"
    df = pd.read_hdf(h5)
    df.loc[df.index[0], (dt.SCORER, "tongue", slice(None))] = np.nan
    df.to_hdf(h5, key="df_with_missing", mode="w")

    pend = rg.pending_frames(live, "cam4", PARTS)
    assert df.index[0][2] not in pend.get("cam4_pre_a", []), "a tongue-less frame was called pending"


def test_a_frame_with_a_row_but_no_points_at_all_IS_pending(projects):  # noqa: F811
    """DLC writes an all-NaN row for a frame nobody opened; that one genuinely needs labelling."""
    src, _ = projects
    dt.stage()
    h5 = src / "labeled-data" / "cam4_pre_a" / f"CollectedData_{dt.SCORER}.h5"
    df = pd.read_hdf(h5)
    df.loc[df.index[1], :] = np.nan
    df.to_hdf(h5, key="df_with_missing", mode="w")

    pend = rg.pending_frames(src, "cam4", PARTS)
    assert df.index[1][2] in pend.get("cam4_pre_a", [])


def test_SEEDED_frames_are_not_described_as_blank(built, tmp_path):
    """A frame `seed_pending` filled in carries the NETWORK's guesses, above `pcutoff` only.

    Telling the labeller to "place the four parts" would be wrong, and worse, it would read as
    though the points already on the frame were somebody's work -- the exact confusion that makes a
    seed indistinguishable from a label in the first place. The files cannot tell them apart, which
    is why `seed_pending` records what it seeded and the guide is handed that list separately.
    """
    proj, path, s, src = built
    _add_blank_frames(src, "cam4_pre_a", ["img0000910.png"])
    rec = proj / "_seed" / "seeded_frames.csv"
    rec.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"session": "cam4_pre_a", "image": "img0000910.png"}]).to_csv(rec, index=False)

    dest = tmp_path / "g4.html"
    rg.build(proj, dest=dest)
    s4 = dest.read_text(encoding="utf-8")

    assert "They are <b>blank</b>" not in s4
    assert "put there by the network" in s4
    assert "A blank tongue does not mean" in s4
    assert "img0000910.png" in s4


# --------------------------------------------------- finding the frame, and knowing when to delete

def test_every_listed_frame_carries_its_SLIDER_POSITION(built):
    """A filename is how you check you are on the right frame; it is not how you get there.

    napari opens a folder as a stack and moves through it with the slider and D/A, so without a
    position the only way to reach `img0674392.png` is to scroll and squint.
    """
    proj, path, s, src = built
    d = dt.classify(proj)
    assert len(d)
    assert s.count('class="pos">slider') == len(d)
    for _, r in d.iterrows():
        pos = rg.folder_positions(src, r.session)[r.image]
        assert f"slider <b>{pos}</b>" in s, f"{r.image} has no position"


def test_folder_positions_match_sorted_order_which_is_numeric_order(built, projects):  # noqa: F811 -- pytest fixture requested by name
    """`dlc_frames` zero-pads the frame number to seven digits, so lexicographic == numeric. If
    that ever stops being true the positions silently point at the wrong frames."""
    src, _ = projects
    stem = "cam4_pre_a"
    pos = rg.folder_positions(src, stem)
    names = sorted((src / "labeled-data" / stem).glob("img*.png"))
    assert pos == {q.name: i for i, q in enumerate(names)}
    nums = [int(n.stem.replace("img", "")) for n in names]
    assert nums == sorted(nums), "filename order is not frame-number order"


def test_the_guide_explains_how_to_jump_to_a_position(built):
    proj, path, s, src = built
    assert "Getting to a frame without scrolling" in s
    assert "counting from <b>0</b>" in s
    assert "Shift</kbd>+<kbd>&rarr;" in s


def test_ADD_does_not_override_a_deliberate_blank(built):
    """She may have left a part blank because the LANDMARK was not visible -- the tongue is out but
    the tip is behind the spout. The network cannot know that, so ADD is its opinion, not a
    correction, and the guide has to say so or she will place points she does not believe."""
    proj, path, s, src = built
    assert "opinion, not a correction" in s
    assert "then blank was the right answer and it stays blank" in s


def test_the_guide_says_when_NOT_to_delete(built):
    """Deleting a merely-misplaced point loses the information in how far it moved, and deleting
    out of uncertainty throws away a usable label."""
    proj, path, s, src = built
    assert "When to delete a point, and when not to" in s
    assert "Do NOT delete</b> a point that is merely in the wrong place" in s
    assert "Do NOT delete</b> a point just because you are unsure" in s
    assert "A wrong point is taught as the truth" in s


def test_the_guide_warns_that_the_save_only_writes_the_SELECTED_layer(built):
    """With the image layer selected napari saves the photograph and does not complain."""
    proj, path, s, src = built
    assert "only writes the SELECTED layer" in s
    assert "Deletions need saving too" in s
