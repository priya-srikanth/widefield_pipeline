"""The SUPPLEMENTAL successful-lick selection (Priya 11-07): WIDEFIELD_WORKING_TRIALS=in_working_contact = in_working AND
class == contact only, same camera working period / end cut. Pins: parsing + aliases; its tag (and so every frozen spec,
cache key and output folder) differs from production and the robustness arms; production's tag is unchanged; the
snapshot table narrows to contact rows; the rebuild's live `in_sel` follows it; the render-profile arm's run tag."""
import numpy as np
import pandas as pd
import pytest

from wfield_local import trial_classes as tc
from wfield_local import working_trials as wt

#: the PRODUCTION tag, pinned byte for byte (frozen specs of the 11-03 late models)
PROD_TAG = "sel-camera-cue_stop-endlick-cls-contact-incomplete"


@pytest.fixture
def clean(monkeypatch):
    for k in (wt.WORKING_TRIALS_ENV, tc.GATE_ENV):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.mark.parametrize("v", ["in_working_contact", "contact", "contact_only", " Contact_Only ",
                               "in_working_contact_only"])
def test_selection_parses_contact_aliases(clean, v):
    clean.setenv(wt.WORKING_TRIALS_ENV, v)
    assert wt.selection() == wt.IN_WORKING_CONTACT
    assert wt.is_camera() and wt.working_classes() == frozenset({"contact"})


def test_production_and_robustness_parsing_unchanged(clean):
    assert wt.selection() == wt.IN_WORKING                          # unset = production
    for v, want in (("in_working", wt.IN_WORKING), ("licks", wt.IN_WORKING), ("in_working_jaw", wt.IN_WORKING_JAW),
                    ("quit_tail", wt.QUIT_TAIL)):
        clean.setenv(wt.WORKING_TRIALS_ENV, v)
        assert wt.selection() == want
    clean.setenv(wt.WORKING_TRIALS_ENV, "nonsense")
    with pytest.raises(ValueError):
        wt.selection()


def test_tags_are_distinct_and_production_is_pinned(clean):
    assert wt.tag("in_working") == PROD_TAG
    assert wt.tag() == PROD_TAG                                     # env unset
    assert wt.tag("in_working_jaw") == "sel-camera-cue_stop-endlick"
    ct = wt.tag("in_working_contact")
    assert ct == "sel-camera-cue_stop-endlick-cls-contact"
    assert len({ct, wt.tag("in_working"), wt.tag("in_working_jaw")}) == 3
    assert wt.WORKING_CLASSES[wt.IN_WORKING] == frozenset({"contact", "incomplete"})
    assert wt.WORKING_CLASSES[wt.IN_WORKING_CONTACT] == frozenset({"contact"})


def test_trial_classes_follow_the_selection(clean):
    """The rebuild's live path (`in_sel`, the selection tag in frozen specs / cache keys / output folders)."""
    clean.setenv(tc.GATE_ENV, "camera")
    prod_part = tc.sel_cls_part()
    prod_miss = tc.sel_classes("miss_working")
    prod_jaw = tc.jaw_counts_in_stop()
    assert prod_part == "contact-incomplete" and prod_miss == frozenset({"incomplete"}) and prod_jaw
    assert tc.sel_class_text() == "contact + incomplete"
    clean.setenv(wt.WORKING_TRIALS_ENV, "in_working_contact")
    assert tc.sel_cls_part() == "contact"
    assert tc.sel_classes("working") == frozenset({"contact"})
    assert tc.sel_classes("miss_working") == prod_miss             # the no-lick arm stays production's
    assert tc.jaw_counts_in_stop() == prod_jaw                     # same stop rule / working period
    assert tc.sel_class_text() == "contact"
    cls = np.array(["contact", "incomplete", "jaw_only", "no_response"], dtype=object)
    assert tc.in_sel(cls, "working").tolist() == [True, False, False, False]
    tag = tc.selection_tag("camera", "cue_stop", end="lick", cls=tc.sel_cls_part())
    assert tag == "sel-camera-cue_stop-endlick-cls-contact" and tag != tc.selection_tag(
        "camera", "cue_stop", end="lick", cls="contact-incomplete")
    clean.setenv(wt.WORKING_TRIALS_ENV, "in_working_jaw")
    assert tc.sel_cls_part() is None


@pytest.fixture
def snap(tmp_path, clean):
    rng = np.random.default_rng(3)
    rows = []
    for lab in ("PSX_0806", "PSX_0901"):
        for k in range(50):
            cls = rng.choice(["contact", "incomplete", "jaw_only", "no_response"], p=[.5, .3, .1, .1])
            rows.append(dict(label=lab, cue_idx=k, cls=cls, position="far_R"))
    for k in range(10):
        rows.append(dict(label="PSX_0809", cue_idx=k, cls="no_video", position=""))
    D = pd.DataFrame(rows)
    D["working_pl_lick"] = tc.in_working(D.cls, D.cue_idx, session=D.label, position=D.position, end="lick")
    f = tmp_path / "snap.csv.gz"
    D.to_csv(f, index=False)
    clean.setenv(wt.SNAPSHOT_ENV, str(f))
    wt._table.cache_clear()
    yield D
    wt._table.cache_clear()


def test_table_narrows_to_contact_rows(snap):
    p = str(wt.snapshot_path())
    Tc, Tp = wt._table(p, wt.IN_WORKING_CONTACT), wt._table(p, wt.IN_WORKING)
    assert Tc["PSX_0809"] == {} and Tp["PSX_0809"] == {}
    for lab, g in snap[snap.cls != "no_video"].groupby("label"):
        got = np.array([Tc[lab][k] for k in g.cue_idx])
        exp = g.working_pl_lick.to_numpy(bool) & (g.cls == "contact").to_numpy()
        assert np.array_equal(got, exp) and got.any()
        assert not got[g.cls.to_numpy() != "contact"].any()
        prod = np.array([Tp[lab][k] for k in g.cue_idx])
        assert np.array_equal(got, prod & (g.cls == "contact").to_numpy())   # a strict subset of production
        assert prod[g.cls.to_numpy() == "incomplete"].any()


def test_row_masks_under_the_contact_selection(snap):
    kept = ["PSX_0806", "PSX_0901"]
    idx = {lab: (np.arange(0, 30), np.arange(30, 50)) for lab in kept}
    GE, GU = np.repeat([0, 1], 30), np.repeat([0, 1], 20)
    we, wu, have = wt.row_masks(kept, idx, GE, GU, sel=wt.IN_WORKING_CONTACT)
    assert have == {0, 1}
    cls_e = np.concatenate([snap[snap.label == lab].set_index("cue_idx").cls.reindex(range(30)) for lab in kept])
    assert not we[cls_e != "contact"].any() and we.any()


def test_profile_arm_has_its_own_run_tag(monkeypatch):
    from wfield_local import render_profile, run_mode
    d = render_profile.load("cohort1_rebuild_20261103", arm="contactOnly")
    prod = render_profile.load("cohort1_rebuild_20261103")
    assert d["env"][wt.WORKING_TRIALS_ENV] == "in_working_contact"
    for k in list(__import__("os").environ):
        if k.startswith(render_profile.OWNED_PREFIXES):
            monkeypatch.delenv(k, raising=False)
    for k, v in d["env"].items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(tc, "class_source_signature", lambda *a, **k: "sig")
    assert run_mode.tag() == d["notes"]["expected_run_tag"]
    assert d["notes"]["expected_run_tag"] == prod["notes"]["expected_run_tag"].replace(
        "-cls-contact-incomplete-", "-cls-contact-")
    assert run_mode.keyed_name("section_g.json") != "section_g.json"
    assert "cls-contact-stop" in run_mode.keyed_name("section_g.json")
