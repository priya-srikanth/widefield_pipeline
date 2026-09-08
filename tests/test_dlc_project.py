"""The project holds the only copy of hand-placed labels, so the tests are about not losing them."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wfield_local import dlc_project as dp


def _frames(root, stem, n=3, labelled=False, parts=("nose", "jaw")):
    if labelled:
        pytest.importorskip("tables", reason="the .h5 round-trip needs pytables (in the dlc env)")
    d = root / "labeled-data" / stem
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"img{i:07d}.png").write_bytes(b"png")
    if labelled:
        cols = pd.MultiIndex.from_tuples([(dp.SCORER, bp, c) for bp in parts for c in ("x", "y")],
                                         names=["scorer", "bodyparts", "coords"])
        idx = pd.MultiIndex.from_tuples([("labeled-data", stem, f"img{i:07d}.png")
                                         for i in range(n)])
        df = pd.DataFrame(np.arange(n * len(parts) * 2, dtype=float).reshape(n, -1),
                          columns=cols, index=idx)
        df.to_hdf(d / f"CollectedData_{dp.SCORER}.h5", key="df_with_missing", mode="w")
    return d


@pytest.fixture
def share(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "out_root", lambda rv=None: tmp_path)
    return tmp_path


def test_hand_corrected_labels_are_never_overwritten_by_a_refresh(share, tmp_path):
    """The single most damaging thing this module could do.

    Once a human has corrected labels, the copy on the share is the STALE one -- a refresh that
    copied over it would run the work backwards and there is no other record of it.
    """
    _frames(share, "cam4_x", labelled=True)
    proj = tmp_path / "proj"
    dp.write_config(proj)
    dp.sync_frames(proj)

    corrected = proj / "labeled-data" / "cam4_x" / f"CollectedData_{dp.SCORER}.h5"
    corrected.write_bytes(b"corrected by hand")
    dp.sync_frames(proj)                                   # refresh again
    assert corrected.read_bytes() == b"corrected by hand"


def test_a_refresh_adds_NEW_frames_without_touching_old_ones(share, tmp_path):
    _frames(share, "cam4_x", n=2)
    proj = tmp_path / "proj"
    dp.write_config(proj)
    assert dp.sync_frames(proj)[1] == 2

    (proj / "labeled-data" / "cam4_x" / "img0000000.png").write_bytes(b"EDITED")
    _frames(share, "cam4_x", n=4)                          # two more arrive
    _folders, images, _ = dp.sync_frames(proj)
    assert images == 2, "only the new ones"
    assert (proj / "labeled-data" / "cam4_x" / "img0000000.png").read_bytes() == b"EDITED"


def test_seeded_labels_are_padded_to_the_projects_FULL_bodypart_list(share, tmp_path):
    """dlc_prelabel writes only what a camera can see -- ten on cam4 -- but a DLC project has ONE
    bodypart list and its readers expect every column. NaN says 'unplaced' in DLC's own language;
    a short table is a shape mismatch waiting to surface somewhere inside DLC."""
    _frames(share, "cam4_x", labelled=True, parts=("nose", "jaw"))
    proj = tmp_path / "proj"
    dp.write_config(proj)
    dp.sync_frames(proj)

    got = pd.read_hdf(proj / "labeled-data" / "cam4_x" / f"CollectedData_{dp.SCORER}.h5")
    from wfield_local.dlc_frames import bodyparts
    assert [c[1] for c in got.columns][::2] == bodyparts(), "columns must be the project's union"
    assert got[(dp.SCORER, "nose", "x")].notna().all(), "seeded values survive the padding"
    assert got[(dp.SCORER, "L_eye", "x")].isna().all(), "padded parts are unplaced, not zero"


def test_the_project_bodyparts_are_the_union_of_the_views(share, tmp_path):
    import yaml

    from wfield_local.dlc_frames import bodyparts, cameras
    proj = tmp_path / "proj"
    cfg = yaml.safe_load(dp.write_config(proj).read_text(encoding="utf-8"))
    assert cfg["bodyparts"] == bodyparts()
    for cam in cameras():
        assert set(bodyparts(cam)) <= set(cfg["bodyparts"])


def test_project_path_is_rewritten_for_the_machine_it_is_opened_on(share, tmp_path):
    """DLC stores an ABSOLUTE path. MICROSCOPE is N: on the imaging box and M: on the analysis box,
    so a project created on one and opened on the other points at nothing."""
    import yaml
    proj = tmp_path / "proj"
    dp.write_config(proj)
    p = proj / "config.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["project_path"] = "M:/somewhere/else"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    dp.write_config(proj)
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["project_path"] == \
        str(proj).replace("\\", "/")


def test_a_refresh_preserves_the_training_iteration(share, tmp_path):
    """Resetting `iteration` would make DLC write a new network's files over the previous one's."""
    import yaml
    proj = tmp_path / "proj"
    dp.write_config(proj)
    p = proj / "config.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["iteration"] = 3
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    dp.write_config(proj)
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["iteration"] == 3


def test_folders_without_frames_are_not_adopted(share, tmp_path):
    (share / "labeled-data" / "cam4_empty").mkdir(parents=True)
    _frames(share, "cam4_real")
    assert [p.name for p in dp.frame_dirs()] == ["cam4_real"]


def test_only_the_requested_cameras_are_synced(share, tmp_path):
    _frames(share, "cam4_a")
    _frames(share, "cam2_b")
    assert [p.name for p in dp.frame_dirs(cams=["cam4"])] == ["cam4_a"]
    assert len(dp.frame_dirs()) == 2
