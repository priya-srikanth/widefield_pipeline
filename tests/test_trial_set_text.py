"""Every figure states its trial set (Priya 11-09) + the jaw-only class panel of the working-map families.

Pins: the sentence for each variant under the run's switches (`trial_set_text.sentence`), that a figure saved through
the shared save path carries it (and only when a trial set is active, once), that a map bundle / figdata records it and
a replot restores it (stale "lick + miss-while-working" titles rewritten), the jaw-only map selector (exactly the
jaw_only rows), that the existing position-map cache kinds do not move, and the pooled jaw-only 15r panel.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wfield_local import trial_set_text as tst  # noqa: E402

CAMERA_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_WORKING_TRIALS": "in_working",
              "WIDEFIELD_TRIAL_STOPPED": "both", "WIDEFIELD_STOP_LICKS": "excise", "WIDEFIELD_STOP_TAIL": "lick_trial",
              "WIDEFIELD_TRIAL_END": "lick", "WIDEFIELD_LICK_SOURCE": "lp", "WIDEFIELD_GATE_SOURCE": "daq"}


@pytest.fixture
def camera(monkeypatch):
    for k, v in CAMERA_ENV.items():
        monkeypatch.setenv(k, v)
    yield


# ------------------------------------------------------------------------------------------------ the sentences
def test_working_sentence_production_camera_run(camera):
    assert tst.sentence("working") == (
        "trials: camera contact + incomplete (LP licks), within the working period (first-last camera lick), "
        "stopped runs (A2 + lick excise) and quit tail removed; excluded: jaw-only, no response, unknown")


def test_period_supplement_sentence(camera):
    assert tst.sentence("working_period") == (
        "trials: all trials in the working period (first-last camera lick): camera contact + incomplete + jaw-only + "
        "no response (LP licks); excluded: stopped runs (A2 + lick excise), stopped-licking, quit tail, unknown "
        "(no video / unmatched cue)")
    assert tst.arm_caption("cue", "working_period") == (
        "post-cue, ALL working-period trials (contact + incomplete + jaw-only + no response)")
    assert tst.variant_from_name("epoch_15rpa_position_RESTNLref_by_animal_cue_working_period") == (
        "cue", "working_period")


def test_every_variant_names_its_classes(camera):
    s = {v: tst.sentence(v) for v in tst.VARIANTS}
    assert s["jaw_only"].startswith("trials: camera jaw-only")
    assert "excluded: contact, incomplete, no response, unknown" in s["jaw_only"]
    assert "within the working period" in s["jaw_only"]
    assert s["miss_working"].startswith("trials: no-lick trials") and "camera class incomplete" in s["miss_working"]
    assert "jaw-only" in s["miss_working"].split("excluded:")[1]
    assert s["stopped"].startswith("trials: camera stopped") and "A2" in s["stopped"]
    assert s["lick"].startswith("trials: lick trials (LP licks") and "excluded: no-lick trials" in s["lick"]
    for v, txt in s.items():
        assert "miss-while-working" not in txt, v                # the stale DAQ wording never appears under camera


def test_working_selection_switches_the_sentence(camera, monkeypatch):
    monkeypatch.setenv("WIDEFIELD_WORKING_TRIALS", "in_working_jaw")
    s = tst.sentence("working")
    assert s.startswith("trials: camera contact + incomplete + jaw-only (LP licks)")
    assert "excluded: no response, unknown" in s
    monkeypatch.setenv("WIDEFIELD_WORKING_TRIALS", "in_working_contact")
    s = tst.sentence("working")
    assert s.startswith("trials: camera contact (LP licks)") and "excluded: incomplete, jaw-only" in s


def test_lick_source_and_stop_rule_are_read(camera, monkeypatch):
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "daq")
    assert "(DAQ licks)" in tst.sentence("working")
    assert "a DAQ spout contact" in tst.sentence("lick")
    monkeypatch.setenv("WIDEFIELD_TRIAL_STOPPED", "long")
    assert "stopped runs (A3 + lick excise)" in tst.sentence("working")


def test_daq_gate_wording(camera, monkeypatch):
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "daq")
    assert tst.sentence("working") == ("trials: lick + no-lick trials outside the terminal quit period "
                                       "(LP licks; DAQ quit-tail rule)")
    assert tst.short_label("working") == "lick + miss-while-working (DAQ quit-tail rule)"
    monkeypatch.setenv("WIDEFIELD_GATE_SOURCE", "lp")
    assert "quit = no DAQ contact AND no LP attempt" in tst.sentence("working")
    with pytest.raises(ValueError):
        tst.sentence("jaw_only")                                 # a camera class: no DAQ-rule set exists
    assert tst.safe_sentence("jaw_only") is None


def test_captions_and_stale_title(camera):
    assert tst.arm_caption("cue", "working") == "post-cue, camera working trials (contact + incomplete)"
    assert tst.arm_caption("precue", "working", enl_window=1.0).startswith("ENL (pre-cue) 1 s lick-free, ")
    assert tst.arm_caption("cue", "lick") == "post-cue, LICK TRIALS ONLY"
    assert tst.arm_caption("cue", "jaw_only") == "post-cue, JAW-ONLY camera trials"
    assert (tst.refresh_title("X -- post-cue, lick + miss-while-working | g", "working")
            == "X -- post-cue, camera working trials (contact + incomplete) | g")
    assert tst.variant_from_name("epoch_15r_position_RESTNLref_cue_working_bundle") == ("cue", "working")
    assert tst.variant_from_name("epoch_15r_position_RESTNLref_precue_jaw_only") == ("precue", "jaw_only")
    assert tst.variant_from_name("frozen_refit_pooled") == (None, None)


# ------------------------------------------------------------------------------------------------ on the figure
def _texts(fig):
    return [t.get_text() for t in fig.texts]


def test_saved_figure_carries_the_sentence_once(camera, tmp_path):
    from wfield_local import figure_layout as fl
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1])
    with tst.active("working", "cue"):
        fl.save_png_svg(fig, tmp_path / "a.png", dpi=50)
        fl.save_png_svg(fig, tmp_path / "b.png", dpi=50)       # idempotent: one stamp per figure
    hits = [t for t in fig.texts if t.get_gid() == tst.GID]
    assert len(hits) == 1
    assert hits[0].get_text().replace("\n", " ") == tst.sentence("working")
    assert (tmp_path / "a.png").exists() and (tmp_path / "svg" / "a.svg").exists()
    assert tst.current() is None                                  # the context does not leak
    plt.close(fig)


def test_no_active_trial_set_adds_nothing(camera, tmp_path):
    from wfield_local import figure_layout as fl
    fig, ax = plt.subplots(figsize=(3, 2))
    fl.save_png_svg(fig, tmp_path / "c.png", dpi=50)
    assert fig.texts == []
    plt.close(fig)


def test_stamp_prepends_to_the_run_stamp(camera):
    fig = plt.figure(figsize=(8, 3))
    fig.text(0.995, 0.002, "LP licks: run stamp", color=tst.RUN_STAMP_COLOR)
    t = tst.stamp(fig, "trials: X")
    assert len(fig.texts) == 1 and t.get_text() == "trials: X\nLP licks: run stamp" and t.get_gid() == tst.GID
    plt.close(fig)


@pytest.fixture
def fake_run_stamp(monkeypatch):
    """`lp_licks.stamp` without the run's coverage table (it reads M:): a plain purple run stamp."""
    from wfield_local import lp_licks
    monkeypatch.setattr(lp_licks, "stamp", lambda fig: fig.text(0.995, 0.002, "RUN STAMP", color=tst.RUN_STAMP_COLOR))


def test_grant_kit_save_carries_it(camera, fake_run_stamp, tmp_path):
    from wfield_local import grant_kit
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([0, 1], [1, 2])
    with tst.active("jaw_only", "cue"):
        grant_kit._save(fig, tmp_path / "g.png", dpi=50)
    hits = [t.get_text() for t in fig.texts if t.get_gid() == tst.GID]
    assert len(hits) == 1 and hits[0].endswith("\nRUN STAMP")
    assert hits[0][: -len("\nRUN STAMP")].replace("\n", " ") == tst.sentence("jaw_only")
    plt.close(fig)


def test_map_bundle_records_it_and_replot_restores_it(camera, fake_run_stamp, tmp_path):
    from wfield_local import epoch_figures as ef
    cells = {("far contra", "pre"): np.ones((6, 5)), ("far contra", "acute"): np.zeros((6, 5))}
    name = "epoch_15r_position_RESTNLref_cue_working"
    with tst.active("working", "cue"):
        q = ef.map_grid(cells, tmp_path, name=name, title="T -- post-cue, lick + miss-while-working",
                        row_labels=["far contra"], col_labels=["pre", "acute"])
    meta = json.loads((tmp_path / "data" / f"{name}_bundle.json").read_text(encoding="utf-8"))
    assert meta["trial_set"] == tst.sentence("working")
    # a bundle written BEFORE 11-09 has no trial_set: the replot names the class from the stem and fixes the title
    meta.pop("trial_set")
    (tmp_path / "data" / f"{name}_bundle.json").write_text(json.dumps(meta), encoding="utf-8")
    p = ef.replot_map(q, tmp_path, edges=None)
    m2 = json.loads((tmp_path / "data" / f"{p.stem}_bundle.json").read_text(encoding="utf-8"))
    assert m2["trial_set"] == tst.sentence("working")
    assert "miss-while-working" not in m2["kw"]["title"]
    assert m2["kw"]["title"] == "T -- post-cue, camera working trials (contact + incomplete)"


# ------------------------------------------------------------------------------------------------ the jaw-only class
def test_jaw_only_rows_select_exactly_jaw_only():
    from wfield_local import position_reference_maps as prm
    ce = ["contact", "jaw_only", "incomplete", "stopped", "jaw_only"]
    cn = ["no_response", "jaw_only", "after_end", "unknown", "stopped_licking"]
    ke, kn = prm.jaw_only_rows(ce, cn)
    assert ke.tolist() == [False, True, False, False, True]
    assert kn.tolist() == [False, True, False, False, False]


def test_period_rows_are_the_judged_working_period_classes():
    from wfield_local import position_reference_maps as prm
    from wfield_local import trial_classes as tc
    ce = ["contact", "jaw_only", "incomplete", "stopped", "no_response", "unknown"]
    cn = ["no_response", "jaw_only", "after_end", "unknown", "stopped_licking", "early_run"]
    ke, kn = prm.class_rows(ce, cn, prm.CLASS_VARIANTS[prm.WORKING_PERIOD])
    assert ke.tolist() == [True, True, True, False, True, False]
    assert kn.tolist() == [True, True, False, False, False, False]
    # every out-of-period label is excluded by construction
    assert not set(prm.CLASS_VARIANTS[prm.WORKING_PERIOD]) & set(tc.OUT_OF_WORKING_PERIOD)


def test_jaw_only_xy_takes_both_arms(camera, monkeypatch):
    from wfield_local import position_reference_maps as prm
    from wfield_local import trial_classes as tc
    table = {"e": np.array(["contact", "jaw_only", "incomplete"], object),
             "n": np.array(["jaw_only", "no_response"], object)}
    monkeypatch.setattr(tc, "row_classes", lambda label, idx: table[label_arm(idx)])

    def label_arm(idx):
        return "e" if len(idx) == 3 else "n"
    X, y = np.arange(9.).reshape(3, 3), np.array([1, 2, 3])
    Xn, yn = np.arange(6.).reshape(2, 3) + 100, np.array([4, 5])
    Xo, yo = prm._jaw_only_xy("S", X, y, Xn, yn, np.arange(3), np.arange(2))
    assert yo.tolist() == [2, 4] and Xo[:, 0].tolist() == [3.0, 100.0]
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "daq")
    with pytest.raises(ValueError):
        prm._jaw_only_xy("S", X, y, Xn, yn, np.arange(3), np.arange(2))


def test_jaw_branch_is_outside_the_map_digest():
    """The jaw-only branch of `_working_xy` is hook-marked, so `map_collect.code_digest` (every existing prm kind, and
    restnl_mask's digest over the same parts) hashes exactly the pre-11-09 source."""
    from wfield_local import gate_source
    from wfield_local import position_reference_maps as prm
    src = gate_source.strip_hooks(inspect.getsource(prm._working_xy))
    assert "jaw" not in src.lower()
    assert any("CLASS_VARIANTS" in ln and gate_source.HOOK_MARK in ln
               for ln in inspect.getsource(prm._working_xy).splitlines())
    assert prm._class_xy not in prm._raw_maps_parts() and prm.class_rows not in prm._raw_maps_parts()


@pytest.mark.parametrize("variant", ["working", "lick", "miss_working", "stopped"])
def test_existing_prm_kinds_unchanged_and_jaw_kind_new(variant, monkeypatch):
    from wfield_local import map_collect, session_cache
    from wfield_local import position_reference_maps as prm
    seen = {}
    monkeypatch.setattr(session_cache, "cached", lambda s, kind, fn, params=None, verbose=False: seen.setdefault(
        "kind", kind))
    old_params = {"post_s": 2.0, "min_trials": prm.MIN_TRIALS_PER_CLASS, "rest_bins": prm.REST_BASELINE_BINS,
                  "refs": prm.REFERENCES}
    code = prm._raw_maps_code()
    prm.session_raw_maps_cached({"mc": "Z:/nowhere", "label": "PS00_0000"}, "cue", variant=variant)
    assert seen["kind"] == map_collect.cache_kind("prm", "cue", variant, old_params, code)
    for new in ("jaw_only", "working_period"):
        seen.clear()
        prm.session_raw_maps_cached({"mc": "Z:/nowhere", "label": "PS00_0000"}, "cue", variant=new)
        assert seen["kind"].startswith(f"prm-cue-{new}-")
        assert seen["kind"] != map_collect.cache_kind("prm", "cue", new, old_params, code)   # keyed on its code


def test_jaw_session_floor_bypass_is_hook_only():
    """The jaw-only per-session maps keep every position with >= 1 trial (floor applied after epoch pooling), through
    hook-marked lines of `session_raw_maps` only -- the digest of the existing kinds hashes the pre-11-09 source."""
    from wfield_local import gate_source
    from wfield_local import position_reference_maps as prm
    raw = inspect.getsource(prm.session_raw_maps)
    assert any("SESSION_FLOOR_FREE" in ln and gate_source.HOOK_MARK in ln for ln in raw.splitlines())
    assert "SESSION_FLOOR_FREE" not in gate_source.strip_hooks(raw)
    assert prm.SESSION_FLOOR_FREE == frozenset({"jaw_only"}) and "working_period" not in prm.SESSION_FLOOR_FREE


def test_pooled_by_epoch_trial_weighted_with_floor():
    from wfield_local import position_reference_maps as prm
    m = np.ones((2, 2))
    store = {"A": {"acute": {"far_R": {"A_1": {"rest": 1 * m}, "A_2": {"rest": 4 * m}}}},
             "B": {"acute": {"far_R": {"B_1": {"rest": 9 * m}}}}}
    ntr = {"A": {"acute": {"far_R": {"A_1": 15, "A_2": 5}}}, "B": {"acute": {"far_R": {"B_1": 12}}}}
    maps, ns = prm.pooled_by_epoch(store, ntr, "rest", "far_R", "acute", floor=20)
    assert ns == {"A": 20, "B": 12}                     # every animal with trials is counted ...
    assert set(maps) == {"A"}                           # ... only the pooled cell >= floor is drawn
    assert np.allclose(maps["A"], (15 * 1 + 5 * 4) / 20)   # trial-weighted, not the mean of session means (2.5)
    assert prm.pooled_by_epoch(store, ntr, "rest", "far_R", "pre") == ({}, {})
    assert prm.JAW_EPOCH_FLOOR == 20


def test_jaw_epoch_cells_floor_titles_and_delta(monkeypatch):
    from wfield_local import beta_maps as bm
    from wfield_local import epoch_maps as em
    monkeypatch.setattr(bm, "brain_mask", lambda: None)
    m = np.ones((2, 2))
    work = {a: {"pre": {"far_R": {f"{a}_w": {"rest": 2 * m}}}} for a in ("A", "B")}
    store = {"A": {"pre": {"far_R": {"A_p": {"rest": 2 * m}}}, "acute": {"far_R": {"A_a": {"rest": 6 * m}}}},
             "B": {"pre": {"far_R": {"B_p": {"rest": 2 * m}}}, "acute": {"far_R": {"B_a": {"rest": 10 * m}}}}}
    ntr = {"A": {"pre": {"far_R": {"A_p": 25}}, "acute": {"far_R": {"A_a": 30}}},
           "B": {"pre": {"far_R": {"B_p": 3}}, "acute": {"far_R": {"B_a": 40}}}}
    cells, titles, rows, dcols = em.jaw_epoch_cells(store, ntr, work, "rest", ["far_R", "close_L"],
                                                    ["pre", "acute", "subacute"], floor=20)
    r = em._long_of("far_R")
    assert rows == [r]                                  # close_L has no jaw-only trials: no row
    assert np.allclose(cells[(r, "pre")], 1.0)          # A only (B pre n=3 < 20), / working pre scale 2
    assert np.allclose(cells[(r, "acute")], (3.0 + 5.0) / 2)
    assert "1 an, n=25" in titles[(r, "pre")] and "n < 20: B 3" in titles[(r, "pre")]
    assert "2 an, n=70" in titles[(r, "acute")]
    assert dcols == ["acute - pre"] and np.allclose(cells[(r, "acute - pre")], 2.0)   # within-animal: A only
    assert (r, "subacute") not in cells and (r, "subacute") not in titles


def test_pooled_jaw_only_panel(camera, monkeypatch, tmp_path):
    from wfield_local import beta_maps as bm
    from wfield_local import epoch_figures as ef
    from wfield_local import epoch_maps as em
    from wfield_local import position_reference_maps as prm
    m = np.ones((4, 4))
    work = {"PS94": {"pre": {"far_R": {"PS94_0801": {"rest": 2 * m}}}}}
    jaw = {"PS94": {"acute": {"far_R": {"PS94_0817": {"rest": 4 * m}, "PS94_0818": {"rest": 8 * m}}}}}
    ntr = {"PS94": {"acute": {"far_R": {"PS94_0817": 12, "PS94_0818": 12}}}}
    monkeypatch.setattr(prm, "maps_by_epoch", lambda align, v: (jaw, {}, ntr) if v == "jaw_only" else (work, {}, {}))
    monkeypatch.setattr(prm, "render_references", lambda: ["rest"])
    monkeypatch.setattr(bm, "atlas_edges", lambda *a, **k: None)
    monkeypatch.setattr(bm, "excluded_mask", lambda: None)
    monkeypatch.setattr(bm, "brain_mask", lambda: None)
    got = []
    monkeypatch.setattr(ef, "map_grid", lambda cells, out, **kw: got.append(dict(cells=cells, **kw)) or out / kw["name"])
    out = em.jaw_only_panels(tmp_path, "cue", tst.arm_caption("cue", "jaw_only"))
    assert [p.name for p in out] == ["epoch_15r_position_RESTref_cue_jaw_only",
                                     "epoch_15rpa_position_RESTref_by_animal_cue_jaw_only"]
    pooled, per_an = got
    assert "JAW-ONLY camera trials" in pooled["title"] and pooled["row_labels"] == [em._long_of("far_R")]
    # two sessions of 12 (each under the old per-session floor of 20) pool to 24 >= 20: the cell exists
    assert np.allclose(pooled["cells"][(em._long_of("far_R"), "acute")], 3.0)   # (12*4 + 12*8) / 24 / scale 2
    assert "n=24" in pooled["panel_titles"][(em._long_of("far_R"), "acute")]
    assert not pooled["delta_cols"]
    assert per_an["row_labels"] == ["PS94"] and np.allclose(per_an["cells"][("PS94", "acute")], 6.0)
    assert per_an["panel_titles"][("PS94", "acute")] == "acute\nn=24"
    # ONE render draws primary + period supplement + jaw-only: supplement_panels = the period 15r / 15rpa + jaw
    calls = []
    monkeypatch.setattr(em, "_fig_15r_reference_maps",
                        lambda out, al, v, w: calls.append(("15r", v, w, tst.current()[0])) or [out / f"r_{v}.png"])
    monkeypatch.setattr(em, "_fig_15rpa_reference_by_animal",
                        lambda out, al, v, w: calls.append(("15rpa", v, w, tst.current()[0])) or [])
    got.clear()
    made = em.supplement_panels(tmp_path, "cue", lambda v: tst.arm_caption("cue", v))
    assert made[0] == tmp_path / "r_working_period.png" and len(made) == 3
    assert calls == [("15r", "working_period", tst.arm_caption("cue", "working_period"), "working_period"),
                     ("15rpa", "working_period", tst.arm_caption("cue", "working_period"), "working_period")]
    assert tst.current() is None
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "daq")
    assert em.jaw_only_panels(tmp_path, "cue", "x") == []        # camera classes: nothing under the DAQ rule
    assert em.supplement_panels(tmp_path, "cue", str) == []
    assert em.supplement_panels(tmp_path, "lick", str) == []


# ------------------------------------------------------------------------------------------------ driver-level declare
def test_declare_reaches_a_plain_savefig(camera, tmp_path):
    """One declaration per driver: a figure written with a PLAIN ``fig.savefig`` / ``plt.savefig`` carries it too."""
    tst.declare("lick")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1])
    fig.savefig(tmp_path / "plain.png", dpi=40)
    plt.savefig(tmp_path / "plain2.png", dpi=40)                  # pyplot path: the same Figure.savefig
    hits = [t for t in fig.texts if t.get_gid() == tst.GID]
    assert len(hits) == 1 and hits[0].get_text().replace("\n", " ") == tst.sentence("lick")
    plt.close(fig)


def test_hook_is_byte_identical_without_a_set(camera, tmp_path):
    tst.install_savefig_hook()
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1])
    fig.savefig(tmp_path / "a.png", dpi=40, metadata={"Software": None})
    from matplotlib.figure import Figure
    Figure.savefig.__wrapped__(fig, tmp_path / "b.png", dpi=40, metadata={"Software": None})
    assert fig.texts == []
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
    plt.close(fig)


def test_narrower_activation_wins_and_none_hides_the_declaration(camera):
    tst.declare(text="events: X")
    assert tst.current_sentence() == "events: X"
    with tst.active("working"):
        assert tst.current_sentence() == tst.sentence("working")
    with tst.active(None):
        assert tst.current() is None                              # a replot of a bundle that named no set
    assert tst.current_sentence() == "events: X"
    tst.declare()
    assert tst.current() is None


def test_declaration_is_visible_from_a_worker_thread(camera):
    import threading
    tst.declare("working")
    got = []
    th = threading.Thread(target=lambda: got.append(tst.current_sentence()))
    th.start()
    th.join()
    assert got == [tst.sentence("working")]


# ------------------------------------------------------------------------------------------------ event / frame sets
def test_event_sentences(monkeypatch):
    monkeypatch.delenv("WFIELD_ITI_FILTER", raising=False)
    assert tst.iti_licks() == (
        "events: ITI licks (3-D whole-session lick table, context iti: after a trial's response window and before the "
        "next spout arrival; no cue); excluded: post-cue (trial) licks")
    b = tst.iti_bout_starts(2.0, (-2.0, 0.75))
    assert "no lick onset in the preceding 2 s" in b and "within [-2, +0.75] s of the onset" in b
    assert b.endswith("excluded: later licks of a bout, post-cue (trial) licks")
    assert tst.cued_licks(first_only=True).startswith("events: FIRST post-cue lick of each trial")
    u = tst.union(tst.iti_licks(), tst.cued_licks())
    assert u.startswith("events: ITI licks") and "  AND  post-cue licks" in u and u.count("events:") == 1
    assert tst.frames("RESTnl rest").startswith("frames: RESTnl rest")
    assert tst.events("x", keep=["", "k"], excluded=["a", "b"]) == "events: x; k; excluded: a, b"


@pytest.mark.parametrize("name", sorted(tst.ITI_FILTER_TEXT))
def test_iti_filter_from_the_env(monkeypatch, name):
    monkeypatch.setenv("WFIELD_ITI_FILTER", name)
    assert f"; ITI filter {name}: {tst.ITI_FILTER_TEXT[name]};" in tst.iti_licks()
    monkeypatch.setenv("WFIELD_ITI_FILTER", "nonsense")             # the drivers ignore an unknown name too
    assert "ITI filter" not in tst.iti_licks()


def test_filter_names_match_the_drivers():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "iti_licks.py").read_text(encoding="utf-8")
    assert "ITI_SUBSET_FILTERS = " + repr(tst.ITI_SUBSET_FILTERS).replace("'", '"') in src
    src2 = (Path(__file__).resolve().parents[1] / "scripts" / "iti_lick_order.py").read_text(encoding="utf-8")
    assert '{"gap2": gap2, "far1": far1, "gap2far1": gap2 & far1}' in src2
    assert set(tst.ITI_REFIT_FILTERS) | set(tst.ITI_SUBSET_FILTERS) == set(tst.ITI_FILTER_TEXT)
