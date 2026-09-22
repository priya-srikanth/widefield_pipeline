"""The network is cheap to retrain; the labels are not, and a WRONG network is worse than none.

So these tests are about the three ways this module could quietly produce a network that looks
trained and is not what anyone believes:

  * training on donor SEEDS because a column had values (`dlc.train.bodyparts` is the contract),
  * reporting a test RMSE measured partly on the training set (the session hold-out),
  * running backwards over the hand-placed labels (the copy direction is the opposite of
    `dlc_project`'s, which is exactly the kind of thing a later reader "fixes").

One test pins a DeepLabCut implementation detail rather than our own code, deliberately — see
`test_DLC_drops_bodyparts_outside_the_config_list`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wfield_local import dlc_train as dt

PARTS = ["nose", "jaw", "tongue", "spout"]
UNION = PARTS + ["L_whiskers_1", "R_whiskers_1", "L_eye"]


def _labels(stem, n, parts=UNION, start=0.0):
    cols = pd.MultiIndex.from_tuples([(dt.SCORER, bp, c) for bp in parts for c in ("x", "y")],
                                     names=["scorer", "bodyparts", "coords"])
    idx = pd.MultiIndex.from_tuples([("labeled-data", stem, f"img{i:07d}.png") for i in range(n)])
    vals = start + np.arange(n * len(parts) * 2, dtype=float).reshape(n, -1)
    return pd.DataFrame(vals, columns=cols, index=idx)


def _folder(proj, stem, n=4, parts=UNION):
    pytest.importorskip("tables", reason="the .h5 round-trip needs pytables (in the dlc env)")
    d = proj / "labeled-data" / stem
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"img{i:07d}.png").write_bytes(b"png")
    _labels(stem, n, parts).to_hdf(d / f"CollectedData_{dt.SCORER}.h5", key="df_with_missing",
                                   mode="w")
    return d


@pytest.fixture
def projects(tmp_path, monkeypatch):
    """A labelling project with four cam4 sessions over two epochs, and a manifest naming them."""
    src = tmp_path / "widefield-Priya-2026-09-08"
    (src / "labeled-data").mkdir(parents=True)
    (src / "config.yaml").write_text("Task: widefield\n", encoding="utf-8")
    stems = ["cam4_pre_a", "cam4_pre_b", "cam4_acute_a", "cam4_acute_b"]
    for s in stems:
        _folder(src, s)
    _folder(src, "cam1_pre_a")                      # another view: must never be staged for cam4

    staging = tmp_path / "_frame_staging"
    staging.mkdir()
    # The REAL manifest shape (dlc_frames writes one row per frame), because `classify` groups by
    # lick and that needs `phase` and `frame`. A four-column stand-in passed the epoch tests and
    # crashed `classify` on a missing `image` column.
    ep = dict(zip(stems, ["pre", "pre", "acute", "acute"]))
    phases = ["lick-16", "lick+0", "lick+32", "lick+64"]          # one whole lick per session
    man = [{"animal": "PS92", "date": "20260606", "cam": "cam4", "epoch": ep[s], "video_stem": s,
            "frame": str(1000 + off), "trial_id": "1", "position": "far_R", "category": "success",
            "phase": ph, "t_from_cue_s": "0.4", "image": f"img{i:07d}.png"}
           for s in stems
           for i, (ph, off) in enumerate(zip(phases, [-4, 0, 8, 16]))]
    pd.DataFrame(man).to_csv(staging / "frame_manifest.csv", index=False)

    monkeypatch.setattr(dt.dlc_project, "project_dir", lambda rv=None: src)
    monkeypatch.setattr(dt, "staging_root", lambda rv=None: staging)
    monkeypatch.setattr(dt, "train_cfg", lambda: {"bodyparts": PARTS, "cam": "cam4",
                                                 "tag": "orofacial", "training_fraction": 0.5,
                                                 "seed": 42})
    return src, dt.train_project()


# ------------------------------------------------------------------ training on seeds, not labels

def test_only_the_DECLARED_bodyparts_are_staged_however_full_the_other_columns_are(projects):
    """The whole reason this module exists.

    Every column in the fixture is fully populated -- as cam4's whisker columns genuinely are, from
    `dlc_prelabel`. "Has values" therefore cannot be the criterion, and `dlc.train.bodyparts` is.
    """
    src, dest = projects
    dt.stage()
    staged = pd.read_hdf(dest / "labeled-data" / "cam4_pre_a" / f"CollectedData_{dt.SCORER}.h5")
    assert list(dict.fromkeys(staged.columns.get_level_values("bodyparts"))) == PARTS

    # and the donor seeds were NOT merely blanked -- the columns are gone, so nothing downstream
    # can decide to use them
    assert "L_whiskers_1" not in set(staged.columns.get_level_values("bodyparts"))


def test_parts_refuses_to_guess_when_nothing_is_declared(monkeypatch):
    monkeypatch.setattr(dt, "train_cfg", lambda: {"bodyparts": []})
    with pytest.raises(SystemExit, match="nothing has been declared as refined"):
        dt.parts()


def test_a_declared_part_that_is_not_in_the_labels_is_a_refusal(projects):
    """Not a silently-dropped column: a typo in the contract must not train a smaller network."""
    with pytest.raises(SystemExit, match="absent from the labels"):
        dt.stage(["nose", "tongue_tip"])


def test_only_the_requested_camera_is_staged(projects):
    src, dest = projects
    dt.stage()
    assert sorted(p.name for p in (dest / "labeled-data").iterdir()) == [
        "cam4_acute_a", "cam4_acute_b", "cam4_pre_a", "cam4_pre_b"]


# ------------------------------------------------------------------------------- copy direction

def test_labels_are_re_synced_FORWARD_and_the_training_copy_is_overwritten(projects):
    """The opposite of `dlc_project`'s rule, and the reason it is opposite.

    In `dlc_project` the human edit is DOWNSTREAM of the copy, so overwriting would destroy work.
    Here it is UPSTREAM: the labelling project is the only source of truth, and refusing to
    overwrite would train on a stale snapshot of it -- silently, since a stale label file is
    perfectly well-formed.
    """
    src, dest = projects
    dt.stage()
    staged = dest / "labeled-data" / "cam4_pre_a" / f"CollectedData_{dt.SCORER}.h5"
    _labels("cam4_pre_a", 4, PARTS, start=1000.0).to_hdf(staged, key="df_with_missing", mode="w")

    dt.stage()                                                  # re-stage
    fresh = pd.read_hdf(staged)
    assert fresh.to_numpy().max() < 1000.0, "a stale training copy survived a re-stage"


def test_staging_never_writes_into_the_LABELLING_project(projects):
    src, dest = projects
    before = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
    dt.stage()
    after = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
    assert before == after, "the labelling project was modified"


def test_the_training_project_is_a_SEPARATE_directory_from_the_labelling_one(projects):
    src, dest = projects
    assert dest != src and src not in dest.parents and dest.name.startswith(src.name)


# ---------------------------------------------------------------------------- the session split

def test_no_session_is_split_across_train_and_test(projects):
    """The claim the test RMSE rests on.

    Four frames of one lick are four views of one pose (`dlc.frames.lick_offsets_s`: 99-100% of
    within-onset pairs are closer in appearance than the 5th percentile of between-onset pairs), so
    a uniform frame split reports an error measured partly on the training set.
    """
    src, dest = projects
    proj = dt.stage()
    train_idx, test_idx = dt.split(proj)
    stems = np.array([ix[1] for ix in dt.labels(proj).index])
    assert not (set(stems[train_idx]) & set(stems[test_idx]))
    assert len(train_idx) and len(test_idx)
    assert len(train_idx) + len(test_idx) == len(stems)


def test_the_hold_out_spans_epochs(projects):
    """Otherwise `per_epoch_error` has nothing to contrast and the post-stroke question is unasked.

    At training_fraction 0.5 over two epochs the hold-out is two sessions, and the round-robin must
    take one from each rather than both from whichever epoch sorted first.
    """
    proj = dt.stage()
    _, test_idx = dt.split(proj)
    epochs = dt.session_epochs()
    held = {epochs[ix[1]] for ix in dt.labels(proj).index[test_idx]}
    assert held == {"pre", "acute"}


def test_the_split_is_deterministic_under_the_seed(projects):
    proj = dt.stage()
    assert dt.split(proj) == dt.split(proj)


def test_a_split_that_would_hold_out_everything_is_a_refusal(projects, monkeypatch):
    """Silently training on nothing is the failure this replaces."""
    monkeypatch.setattr(dt, "train_cfg", lambda: {"bodyparts": PARTS, "cam": "cam4",
                                                  "tag": "orofacial", "training_fraction": 0.0,
                                                  "seed": 42})
    proj = dt.stage()
    with pytest.raises(SystemExit, match="degenerate"):
        dt.split(proj)


def test_split_indices_index_the_order_DLC_will_build(projects):
    """`labels()` must reproduce `merge_annotateddatasets`' concatenation order exactly.

    The indices we hand `create_training_dataset` are positions into ITS table, so if the two
    orders ever diverge the split silently permutes -- train frames scored as test, and a number
    that is wrong rather than absent. DLC's order is: read each folder's h5, `pd.concat`,
    `sort_index` (trainingsetmanipulation.py::merge_annotateddatasets).
    """
    proj = dt.stage()
    ours = dt.labels(proj)
    theirs = pd.concat([pd.read_hdf(h5) for h5 in dt._label_files(proj, "cam4")]).sort_index()
    pd.testing.assert_index_equal(ours.index, theirs.index)


# --------------------------------------------------------------------------- the donor mapping

def test_conversion_array_maps_the_refined_parts_onto_donor_head_channels(monkeypatch):
    """`spout <- R_spout` is the one that is not identity, and it is a measurement, not a guess:
    R clears 0.6 on 53% of frames against L's 23% (`dlc.donor.conversion`)."""
    monkeypatch.setattr(dt, "donor_bodyparts", lambda: [
        "nose", "jaw", "tongue", "L_whiskers_1", "L_whiskers_2", "L_whiskers_3",
        "R_whiskers_1", "R_whiskers_2", "R_whiskers_3", "L_eye", "R_eye", "L_spout", "R_spout"])
    arr, bps = dt.conversion(PARTS)
    assert bps == PARTS
    assert arr == [0, 1, 2, 12]


def test_an_unmappable_part_gives_up_the_decoder_rather_than_inventing_a_channel(monkeypatch):
    """DLC requires one entry per project bodypart when `with_decoder=True`, so a partial mapping
    is not something it accepts -- the only alternatives are a re-initialised head or a fabricated
    channel index, and the second would put a part's weights on an unrelated feature."""
    monkeypatch.setattr(dt, "donor_bodyparts", lambda: ["nose", "jaw"])
    assert dt.conversion(["nose", "jaw", "tongue"]) is None


# ------------------------------------------------------------------------------ DLC's own behaviour

def test_DLC_drops_bodyparts_outside_the_config_list():
    """Pinned because we RELY on it and DeepLabCut does not document it.

    `merge_annotateddatasets` ends with
    ``AnnotationData.reindex(cfg["bodyparts"], axis=1, level="bodyparts")``. That call is what makes
    a four-part `config.yaml` sufficient on its own; `subset_labels` is belt-and-braces on top of it
    (it makes the staged files self-describing). If a DLC upgrade changed reindex to PAD the missing
    parts instead of dropping the extra ones, the whisker seeds would silently re-enter training
    and nothing else here would notice.
    """
    df = _labels("cam4_pre_a", 3, UNION)
    out = df.reindex(PARTS, axis=1, level=df.columns.names.index("bodyparts"))
    assert list(dict.fromkeys(out.columns.get_level_values("bodyparts"))) == PARTS
    assert out.shape[1] == 2 * len(PARTS)


def test_subset_labels_agrees_with_what_DLC_would_have_done():
    df = _labels("cam4_pre_a", 3, UNION)
    pd.testing.assert_frame_equal(
        dt.subset_labels(df, PARTS),
        df.reindex(PARTS, axis=1, level=df.columns.names.index("bodyparts")))


# ----------------------------------------------------------------------------------- the audit

def test_the_audit_counts_PLACED_points_not_rows(projects):
    """`parts()` is a declaration; this is the measurement that stands next to it.

    A blank is masked out of the loss rather than taught as absence, so "how many frames is this
    part actually learned from" is a different number from the frame count -- and for the tongue it
    is the number that matters (out for ~70 ms per lick, so ~93 of 240 cam4 frames).
    """
    src, dest = projects
    d = src / "labeled-data" / "cam4_pre_a"
    df = _labels("cam4_pre_a", 4, UNION)
    df.loc[:, (dt.SCORER, "tongue", slice(None))] = np.nan
    df.iloc[0, df.columns.get_indexer([(dt.SCORER, "tongue", "x"), (dt.SCORER, "tongue", "y")])] = 5.0
    df.to_hdf(d / f"CollectedData_{dt.SCORER}.h5", key="df_with_missing", mode="w")

    a = dt.audit(dt.stage())
    assert a.loc["cam4_pre_a", "frames"] == 4
    assert a.loc["cam4_pre_a", "tongue"] == 1
    assert a.loc["cam4_pre_a", "nose"] == 4


def test_the_staged_config_names_the_refined_subset_as_the_projects_bodyparts(projects):
    """DLC reads `nbodyparts` from here (`auxiliaryfunctions.get_bodyparts`), so this IS the
    network's head size -- and it is the field `dlc_project.write_config` would overwrite with the
    twelve-part union if the two projects were one."""
    import yaml

    proj = dt.stage()
    cfg = yaml.safe_load((proj / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["bodyparts"] == PARTS
    assert cfg["project_path"] == str(proj).replace("\\", "/")


# ------------------------------------------------------------------------- the per-epoch table

def test_per_epoch_error_separates_train_from_test(projects, monkeypatch):
    """`evaluate_network` predicts on EVERY labelled image, train and test alike.

    Pooling them makes each epoch's error depend on what share of that epoch's sessions happened to
    be held out -- an artefact of the split reported as a property of the epoch, which is the same
    class of mistake the table exists to catch.
    """
    proj = dt.stage()
    truth = dt.labels(proj)
    pred = truth.copy()
    _, test_idx = dt.split(proj)
    pred.iloc[:, :] = truth.to_numpy() + 1.0          # 1 px off everywhere ...
    pred.iloc[test_idx, :] = truth.to_numpy()[test_idx] + 10.0   # ... and 10 px on the held-out

    ev = proj / "evaluation-results-pytorch"
    ev.mkdir(parents=True, exist_ok=True)
    pred.to_hdf(ev / "preds.h5", key="df_with_missing", mode="w")

    out = dt.per_epoch_error(proj)
    assert set(out["split"]) == {"train", "test"}
    # sqrt(2) px per point on train (1 px in x and y), sqrt(200) on test
    assert np.allclose(out.loc[out["split"] == "train", "rmse_px"], np.sqrt(2.0))
    assert np.allclose(out.loc[out["split"] == "test", "rmse_px"], np.sqrt(200.0))
    # pooling would have reported something between the two for every epoch
    assert out["n"].sum() == len(truth) * len(PARTS)


# ------------------------------------------------------- the derived project must not shadow the real one

def test_the_training_project_does_not_become_the_LABELLING_project(tmp_path, monkeypatch):
    """The 2026-09-21 bug, and the most damaging thing in this module.

    `dlc_project.project_dir` globbed `widefield-Priya-*` and returned the LAST match. A sibling
    training project named `widefield-Priya-2026-09-08-orofacial` sorts after the real one, so
    `project_dir()` returned the four-bodypart derived copy -- and `dlc_project --label` would have
    opened the GUI on a directory that `stage()` overwrites on the next retrain. Two guards now: the
    date-shaped name match here, and `dlc_train.train_project`'s subdirectory.
    """
    from wfield_local import dlc_project as dp

    monkeypatch.setattr(dp, "out_root", lambda rv=None: tmp_path)
    real = tmp_path / "widefield-Priya-2026-09-08"
    real.mkdir()
    assert dp.project_dir() == real

    # the exact name that broke it, plus the subdirectory form actually used now
    (tmp_path / "widefield-Priya-2026-09-08-orofacial").mkdir()
    (tmp_path / "training" / "widefield-Priya-2026-09-08-orofacial").mkdir(parents=True)
    assert dp.project_dir() == real, "a derived project shadowed the labelling project"


def test_train_project_lives_outside_the_labelling_glob(projects):
    """The second, independent guard: even with a `*` glob, a subdirectory is not a sibling."""
    src, dest = projects
    assert dest.parent.name == "training"
    assert dest.parent.parent == src.parent
    assert not list(src.parent.glob("widefield-Priya-*-orofacial"))


# ------------------------------------------------------------------------- the with_decoder flip

def test_weight_init_keeps_with_decoder_OFF_so_DLC_does_not_take_the_superanimal_path(monkeypatch):
    """`create_training_dataset` branches on `with_decoder` ALONE.

    True routes to `make_super_animal_finetune_config`, which raises
    "`WeightInitialization.dataset` is required for fine-tuning SuperAnimal models" -- our donor is a
    custom project -- and would build the model config from SuperAnimal templates, discarding DLC's
    resnet_50 + aug_default defaults (the two-sided `affine.scaling` we rely on). So the array is
    carried with the decoder off, and `pytorch_updates` flips it at train time, where
    `PoseModel.build` applies it for any snapshot.
    """
    pytest.importorskip("deeplabcut")
    monkeypatch.setattr(dt, "donor_bodyparts", lambda: PARTS + ["L_spout", "R_spout"])
    monkeypatch.setattr(dt, "donor_snapshot", lambda: __file__)
    monkeypatch.setattr(dt, "_cfg", lambda: {"donor": {"conversion": {
        "nose": "nose", "jaw": "jaw", "tongue": "tongue", "spout": "R_spout"}}})

    wi = dt.weight_init(PARTS)
    assert wi.with_decoder is False, "would route into DLC's SuperAnimal fine-tune branch"
    assert list(wi.conversion_array) == [0, 1, 2, 5]
    assert dt.pytorch_updates(PARTS)["train_settings.weight_init.with_decoder"] is True


def test_no_decoder_flip_when_the_donor_cannot_be_mapped(monkeypatch):
    monkeypatch.setattr(dt, "donor_bodyparts", lambda: ["nose"])
    monkeypatch.setattr(dt, "_cfg", lambda: {"donor": {"conversion": {"nose": "nose"}}})
    assert "train_settings.weight_init.with_decoder" not in dt.pytorch_updates(PARTS)


def test_per_epoch_error_reads_DLCs_FOUR_level_prediction_columns(projects):
    """The 2026-09-21 crash, after a full 200-epoch run had already finished.

    Labels are `(scorer, bodyparts, coords)`; `evaluate_network` writes
    `(scorer, individuals, bodyparts, coords)` -- DLC inserts `individuals` even for a
    single-animal project -- plus a `likelihood` coord. A positional `.loc[:, (slice(None), bp,
    ["x","y"])]` put the bodypart name on `individuals` and raised `KeyError('nose')`.
    """
    proj = dt.stage()
    truth = dt.labels(proj)
    cols = pd.MultiIndex.from_tuples(
        [("DLC_Resnet50_x", "individual_1", bp, c) for bp in PARTS for c in ("x", "y", "likelihood")],
        names=["scorer", "individuals", "bodyparts", "coords"])
    pred = pd.DataFrame(0.0, index=truth.index, columns=cols)
    for bp in PARTS:                                   # 3 px off in x, 4 in y -> 5 px error
        pred[("DLC_Resnet50_x", "individual_1", bp, "x")] = dt._xy(truth, bp)[:, 0] + 3.0
        pred[("DLC_Resnet50_x", "individual_1", bp, "y")] = dt._xy(truth, bp)[:, 1] + 4.0
        pred[("DLC_Resnet50_x", "individual_1", bp, "likelihood")] = 0.99

    ev = proj / "evaluation-results-pytorch" / "iteration-0"
    ev.mkdir(parents=True, exist_ok=True)
    pred.to_hdf(ev / "DLC_Resnet50_x_snapshot_best-90.h5", key="df_with_missing", mode="w")

    out = dt.per_epoch_error(proj)
    assert out is not None and set(out["bodypart"]) == set(PARTS)
    assert np.allclose(out["rmse_px"], 5.0)


def test_xy_handles_both_column_depths():
    three = _labels("cam4_pre_a", 3, PARTS)
    four = three.copy()
    four.columns = pd.MultiIndex.from_tuples(
        [(s, "individual_1", b, c) for s, b, c in three.columns],
        names=["scorer", "individuals", "bodyparts", "coords"])
    assert np.array_equal(dt._xy(three, "jaw"), dt._xy(four, "jaw"))
    assert dt._xy(three, "jaw").shape == (3, 2)


def test_worst_frames_ranks_a_stray_label_first(projects):
    """The 2026-09-21 finding: one misclick produced a 53 px epoch RMSE.

    `cam4_2026-08-26T12_25_41/img1796416.png` had the tongue at x=619 on a 680 px frame while every
    other tongue in that session sat at x=338-373. It is a TRAIN frame, so the network had every
    chance to fit it and returned likelihood 0.012 instead -- which is the signature this report
    exists to surface, and which no DLC call produces (`extract_outlier_frames` needs videos and
    looks for temporal jumps, not wrong labels).
    """
    proj = dt.stage()
    truth = dt.labels(proj)
    cols = pd.MultiIndex.from_tuples(
        [("DLC_x", "individual_1", bp, c) for bp in PARTS for c in ("x", "y", "likelihood")],
        names=["scorer", "individuals", "bodyparts", "coords"])
    pred = pd.DataFrame(0.0, index=truth.index, columns=cols)
    for bp in PARTS:
        pred[("DLC_x", "individual_1", bp, "x")] = dt._xy(truth, bp)[:, 0] + 1.0
        pred[("DLC_x", "individual_1", bp, "y")] = dt._xy(truth, bp)[:, 1]
        pred[("DLC_x", "individual_1", bp, "likelihood")] = 0.95
    # one frame where the network puts the tongue 300 px from the label, and is unsure
    pred.iloc[0, pred.columns.get_loc(("DLC_x", "individual_1", "tongue", "x"))] += 300.0
    pred.iloc[0, pred.columns.get_loc(("DLC_x", "individual_1", "tongue", "likelihood"))] = 0.01

    ev = proj / "evaluation-results-pytorch"
    ev.mkdir(parents=True, exist_ok=True)
    pred.to_hdf(ev / "preds.h5", key="df_with_missing", mode="w")

    out = dt.worst_frames(proj, n=3)
    assert out.iloc[0]["bodypart"] == "tongue"
    assert out.iloc[0]["err_px"] > 300
    assert out.iloc[0]["likelihood"] < 0.05
    assert out.iloc[0]["image"] == truth.index[0][2]
    assert out.iloc[1]["err_px"] < 2.0, "only the injected frame should stand out"


# ------------------------------------------------------------------------ the review classification

def _preds(truth, tweaks):
    """Predictions == labels, then per (row, part): (dx, dy, likelihood)."""
    cols = pd.MultiIndex.from_tuples(
        [("DLC_x", "individual_1", bp, c) for bp in PARTS for c in ("x", "y", "likelihood")],
        names=["scorer", "individuals", "bodyparts", "coords"])
    pred = pd.DataFrame(0.0, index=truth.index, columns=cols)
    for bp in PARTS:
        xy = dt._xy(truth, bp)
        pred[("DLC_x", "individual_1", bp, "x")] = xy[:, 0]
        pred[("DLC_x", "individual_1", bp, "y")] = xy[:, 1]
        pred[("DLC_x", "individual_1", bp, "likelihood")] = 0.95
    for (row, bp), (dx, dy, lik) in tweaks.items():
        for c, v in (("x", dx), ("y", dy)):
            pred.iloc[row, pred.columns.get_loc(("DLC_x", "individual_1", bp, c))] += v
        pred.iloc[row, pred.columns.get_loc(("DLC_x", "individual_1", bp, "likelihood"))] = lik
    return pred


def _write_preds(proj, pred):
    ev = proj / "evaluation-results-pytorch"
    ev.mkdir(parents=True, exist_ok=True)
    pred.to_hdf(ev / "preds.h5", key="df_with_missing", mode="w")


def test_classify_separates_the_three_faults_by_CONFIDENCE_not_by_error(projects):
    """The three faults take three different actions, so error size alone cannot rank them.

    A label 100 px from its session's median is a long protrusion, not a mistake -- the tongue moves.
    A distance-from-median rule flagged 13 real cam4 frames of which 12 were fine (2026-09-21). What
    separates them is whether the network disagrees, and how sure it is.
    """
    proj = dt.stage()
    truth = dt.labels(proj)
    pred = _preds(truth, {
        (0, "tongue"): (300.0, 0.0, 0.01),    # nothing there      -> DELETE
        (1, "jaw"): (20.0, 0.0, 0.95),        # confident elsewhere -> REPLACE
        (2, "tongue"): (20.0, 0.0, 0.50),     # unsure              -> DECIDE
    })
    _write_preds(proj, pred)

    d = dt.classify(proj)
    got = {(r.verdict, r.bodypart): r for _, r in d.iterrows()}
    assert ("DELETE", "tongue") in got and got[("DELETE", "tongue")].image == truth.index[0][2]
    assert ("REPLACE", "jaw") in got and got[("REPLACE", "jaw")].image == truth.index[1][2]
    assert ("DECIDE", "tongue") in got
    # the same 20 px error lands in two different verdicts, on likelihood alone
    assert round(got[("REPLACE", "jaw")].err_px) == round(got[("DECIDE", "tongue")].err_px) == 20


def test_classify_finds_an_unplaced_part_the_network_is_confident_about(projects):
    """ADD: `evaluate_network` scores only what was labelled, so a MISSED part is invisible to it."""
    src, _ = projects
    d0 = pd.read_hdf(src / "labeled-data" / "cam4_pre_a" / f"CollectedData_{dt.SCORER}.h5")
    d0.loc[d0.index[0], (dt.SCORER, "tongue", slice(None))] = np.nan
    d0.to_hdf(src / "labeled-data" / "cam4_pre_a" / f"CollectedData_{dt.SCORER}.h5",
              key="df_with_missing", mode="w")

    proj = dt.stage()
    truth = dt.labels(proj)
    _write_preds(proj, _preds(truth, {}))
    d = dt.classify(proj)
    add = d[d.verdict == "ADD"]
    assert len(add) == 1 and add.iloc[0].bodypart == "tongue"
    assert not np.isfinite(add.iloc[0].err_px)


def test_a_label_far_from_the_session_median_is_NOT_flagged_when_the_network_agrees(projects):
    """The false-positive this classification exists to avoid: the tongue genuinely travels."""
    proj = dt.stage()
    truth = dt.labels(proj)
    _write_preds(proj, _preds(truth, {}))       # network agrees everywhere
    assert dt.classify(proj).empty


def test_review_images_writes_one_annotated_crop_per_flagged_frame(projects):
    cv2 = pytest.importorskip("cv2")
    proj = dt.stage()
    truth = dt.labels(proj)
    _write_preds(proj, _preds(truth, {(0, "tongue"): (40.0, 0.0, 0.02)}))
    for f in (proj / "labeled-data").rglob("img*.png"):     # real PNGs, not the b"png" stub
        cv2.imwrite(str(f), np.full((680, 680, 3), 128, np.uint8))

    dest = dt.review_images(proj)
    out = sorted(p.name for p in dest.glob("*.png"))
    assert len(out) == 1 and out[0].startswith("DELETE_tongue_")

    # a second round must not leave the first round's crops behind to be read as current
    _write_preds(proj, _preds(truth, {}))
    dt.review_images(proj)
    assert not list(dest.glob("*.png"))
