"""Stale-read guards, part 2 (11-09): the remaining ITI-family extracts, their readers and the pooled caches found by
the repo sweep after the glue-mask fix. Same rule as tests/test_mask_keyed_caches.py (stale = refused; no digest =
refused only in a rebuild root). The digest is faked; nothing reads the share."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture
def masks(monkeypatch):
    from wfield_local import mask_keys, rebuild_root

    class M:
        def set(self, sig):
            monkeypatch.setattr(mask_keys, "signature", lambda: sig)

        def rebuild(self):
            monkeypatch.setattr(rebuild_root, "active", lambda: True)
    monkeypatch.delenv(rebuild_root.ENV, raising=False)
    m = M()
    m.set("glue_out")
    return m


def _stale_and_legacy(masks, f, call, **fields):
    np.savez(f, mask_sig="glue_in", **fields)
    with pytest.raises(RuntimeError, match="masks"):
        call()
    np.savez(f, mask_sig="glue_out", **fields)
    assert call()["cached"]
    np.savez(f, **fields)
    assert call()["cached"]
    masks.rebuild()
    with pytest.raises(RuntimeError, match="before the 11-09"):
        call()


def test_firstlick_full(masks, tmp_path, monkeypatch):
    import lick_firstlick_extract as m
    monkeypatch.setattr(m, "OUT", tmp_path)
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.extract("L"))


def test_lick_correction(masks, tmp_path, monkeypatch):
    import lick_correction as m
    monkeypatch.setattr(m, "SESS", tmp_path)
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.extract("L"))


def test_enlv_licks(masks, tmp_path, monkeypatch):
    import lick_enlv_extract as m
    monkeypatch.setattr(m, "SESS", tmp_path)
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.one(("L", None, None)))


def test_enlv_regress(masks, tmp_path, monkeypatch):
    import lick_enlv_extract as m
    monkeypatch.setattr(m, "SESS_REG", tmp_path)
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.regress(("L", None)), version=m.REG_VERSION)


def test_revup(masks, tmp_path, monkeypatch):
    import revup_neural as m
    monkeypatch.setattr(m, "SESS", tmp_path)
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.session_worker("L"), version=m.VERSION)


def test_precue_restw_traces(masks, tmp_path, monkeypatch):
    import lick_precue_restw_traces as m
    monkeypatch.setattr(m.il, "OUT_W", tmp_path)
    (tmp_path / "c").mkdir()
    _stale_and_legacy(masks, tmp_path / "c" / "L.npz", lambda: m._session("L", "c", "src"), version=m.VERSION)


def test_prelick_hemisphere_tuning(masks, tmp_path, monkeypatch):
    import prelick_hemisphere_tuning as m
    monkeypatch.setattr(m, "SESS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x"])
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.extract(("L", "PS92_20260810", None, None)))


def test_prelick_svd_tuning(masks, tmp_path, monkeypatch):
    import prelick_svd_tuning as m
    monkeypatch.setattr(m, "FEAT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x"])
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.extract("L"))


def test_prelaunch_aim_tracking(masks, tmp_path, monkeypatch):
    import prelaunch_aim_tracking as m
    monkeypatch.setattr(m, "FEAT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x"])
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.extract(("L", None, None, None)))


def test_basis_svdbasis(masks, tmp_path, monkeypatch):
    import iti_basis_decoder as m
    monkeypatch.setattr(m.il, "OUT", tmp_path)
    (tmp_path / "basis_svdbasis" / "sessions").mkdir(parents=True)
    _stale_and_legacy(masks, tmp_path / "basis_svdbasis" / "sessions" / "L.npz",
                      lambda: m.extract(("L", "svdbasis")))


def test_cued_wide(masks, tmp_path, monkeypatch):
    import lick_cued_ramp as m
    monkeypatch.setattr(m.il, "OUT_W", tmp_path)
    (tmp_path / m.WIDE).mkdir()
    _stale_and_legacy(masks, tmp_path / m.WIDE / "L.npz", lambda: m.wide_session("L"),
                      iti=np.zeros(1), pos=np.array(["a"]), maxdiff=0.0, maxdiff_iti=0.0)


def test_sidecar_rule(masks, tmp_path):
    from wfield_local import mask_keys
    f = tmp_path / "pool.pkl"
    f.write_bytes(b"x")
    mask_keys.check_sidecar(f)                                   # legacy, default root
    mask_keys.write_sidecar(f)
    mask_keys.check_sidecar(f)
    masks.set("glue_new")
    with pytest.raises(RuntimeError, match="glue_out"):
        mask_keys.check_sidecar(f)
    mask_keys.sidecar(f).unlink()
    masks.rebuild()
    with pytest.raises(RuntimeError, match="before the 11-09"):
        mask_keys.check_sidecar(f)


def test_sd_tables_are_sidecar_checked(masks, tmp_path, monkeypatch):
    import iti_licks as il
    from wfield_local import mask_keys
    f = tmp_path / "sd.csv"
    f.write_text("label,a\nL,1\n", encoding="utf-8")
    mask_keys.write_sidecar(f, "glue_in")
    monkeypatch.setattr(il, "_TAB", {})
    with pytest.raises(RuntimeError, match="glue_in"):
        il._table(f)
    mask_keys.write_sidecar(f)
    assert il._table(f).loc["L", "a"] == 1


SOURCES = {
    "arrival_to_cue_traces": 2, "stopped_trial_traces": 1, "lick_correction": 1, "lick_enlv_extract": 2,
    "lick_firstlick_extract": 1, "revup_neural": 1, "lick_precue_restw_traces": 1, "lick_cued_ramp": 1,
    "prelick_hemisphere_tuning": 1, "prelick_svd_tuning": 2, "prelaunch_aim_tracking": 1, "iti_basis_decoder": 1,
    "within_spout_deviation": 1, "lick_template_test": 1, "spout_unique_maps": 1}


@pytest.mark.parametrize("script,n", sorted(SOURCES.items()))
def test_writers_stamp(script, n):
    src = (REPO / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    assert src.count("mask_keys.stamp()") >= n


@pytest.mark.parametrize("script", ["iti_area_maps", "lick_balance_contact", "lick_balance_imaging_control",
                                    "lick_balance_subsets", "lick_interhemispheric", "lick_cue_to_lick_extract",
                                    "iti_where_decoder", "lick_code_subspace", "lick_enlv_balance_offset",
                                    "within_lick_correction", "within_spout_deviation", "lick_template_test",
                                    "spout_unique_maps"])
def test_readers_check(script):
    assert "mask_keys.check" in (REPO / "scripts" / f"{script}.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("script", ["iti_zscore_check", "iti_area_split_sd", "iti_june_ref", "iti_decoder_pattern_stats",
                                    "iti_spout_pattern_stats", "lick_session_level_maps", "lick_acad_midline_control",
                                    "partial_refit"])
def test_pooled_caches_write_sidecars(script):
    assert "mask_keys.write_sidecar(" in (REPO / "scripts" / f"{script}.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the ENL pre-cue caches (11-09)
def test_precue_enl_level_cache(masks, tmp_path, monkeypatch):
    import json

    import lick_precue_restw as m
    monkeypatch.setattr(m.il, "OUT_W", tmp_path)
    monkeypatch.setattr(m, "enl_cache", lambda: "c")
    monkeypatch.setattr(m, "_enl_meta", lambda: {"w": 1.0})
    (tmp_path / "c").mkdir()
    _stale_and_legacy(masks, tmp_path / "c" / "L.npz", lambda: m.session_worker_enl("L"), version=m.ENL_VERSION,
                      meta=json.dumps({"w": 1.0}))


def test_precue_enl_traces_cache(masks, tmp_path, monkeypatch):
    import json

    import lick_precue_restw as pr
    import lick_precue_restw_traces as m
    monkeypatch.setattr(m.il, "OUT_W", tmp_path)
    monkeypatch.setattr(pr, "enl_cache", lambda: "src")
    monkeypatch.setattr(m, "enl_cache_t", lambda: "c")
    for d in ("src", "c"):
        (tmp_path / d).mkdir()
    meta, cue = json.dumps({"w": 1.0}), np.array([1.0, 2.0])
    np.savez(tmp_path / "src" / "L.npz", mask_sig="glue_out", meta=meta, cue=cue)   # a current source entry
    _stale_and_legacy(masks, tmp_path / "c" / "L.npz", lambda: m.session_worker_enl("L"), version=m.ENL_VERSION_T,
                      meta=meta, cue=cue)


def test_precue_enl_source_cache_refused(masks, tmp_path, monkeypatch):
    """The traces worker refuses a stale level-script ENL cache before using its trials / windows."""
    import lick_precue_restw as pr
    import lick_precue_restw_traces as m
    monkeypatch.setattr(m.il, "OUT_W", tmp_path)
    monkeypatch.setattr(pr, "enl_cache", lambda: "src")
    (tmp_path / "src").mkdir()
    np.savez(tmp_path / "src" / "L.npz", mask_sig="glue_in", meta="{}", cue=np.zeros(1))
    with pytest.raises(RuntimeError, match="masks"):
        m.session_worker_enl("L")


def test_precue_enl_readers_check():
    for s, n_stamp, n_check in (("lick_precue_restw", 2, 6), ("lick_precue_restw_traces", 2, 6)):
        src = (REPO / "scripts" / f"{s}.py").read_text(encoding="utf-8")
        assert src.count("mask_keys.stamp()") >= n_stamp
        assert src.count("mask_keys.check_obj(") >= n_check
