"""FROZEN encoder spontaneous-lick design v7g3_t3d_spont (FROZEN_LICKS_ALL=split; Priya 11-04, REBUILD_CODE 8.51):
the combined encoder's three lick variables (licks / licks_pre / licks_iti), the same credit / penalty groups and
one-kernel-per-lick routing, the per-trial lick timing rule, the design tag guard, and v7g3_t3d unchanged.
Synthetic, no share. The design is read at IMPORT (module constants, like FROZEN_LICKSPOUT / FROZEN_DESIGN), so the
split design is a separately loaded copy of the module; the package's own `frozen_encoder` stays v7g3_t3d."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from wfield_local import combined_encoding as ce
from wfield_local import frozen_encoder as fe

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_encoder_licks_all as tla  # noqa: E402


def _load(monkeypatch, mode: str | None):
    """A fresh copy of frozen_encoder under FROZEN_LICKS_ALL=mode (the package module is left untouched)."""
    if mode is None:
        monkeypatch.delenv(fe.LICKS_ALL_ENV, raising=False)
    else:
        monkeypatch.setenv(fe.LICKS_ALL_ENV, mode)
    for k in ("FROZEN_DESIGN", "FROZEN_LICKSPOUT", "FROZEN_LICKLAND", "FROZEN_FR", "FROZEN_FR_SPLIT", "FROZEN_FR_LAGS"):
        monkeypatch.delenv(k, raising=False)
    spec = importlib.util.spec_from_file_location(f"_fe_{mode}", fe.__file__)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def fs(monkeypatch):
    monkeypatch.delenv(ce.LICKS_ALL_ENV, raising=False)
    return _load(monkeypatch, "split")


def test_v7g3_t3d_design_unchanged(monkeypatch):
    """Unset (and "none") = the v7g3_t3d layout: no spontaneous variables, variants, run-tag part or tag rewrite."""
    for mode in (None, "none"):
        m = _load(monkeypatch, mode)
        assert not m.SPONT and m.design_name() == "v7g3_t3d"
        assert not set(m.SPONT_VARS) & set(m.FGROUP) and "no_spont" not in m.VARIANTS
        assert m.FGROUP == fe.FGROUP and m.FPEN == fe.FPEN and m.VARIANTS == fe.VARIANTS
        assert "licksall" not in m._run_tag() and m._run_tag() == fe._run_tag()
        assert m.design_tag("gate-attempted_v7g3_t3d_frA") == "gate-attempted_v7g3_t3d_frA"
    monkeypatch.setenv(fe.LICKS_ALL_ENV, "shared")
    with pytest.raises(ValueError, match="FROZEN_LICKS_ALL"):
        _load(monkeypatch, "shared")


def test_spont_groups_and_penalties_are_the_encoders(fs):
    """licks_pre / licks_iti: the encoder's SPLIT_LICK_VARS, UNINSTRUCTED (= frozen "other", with nose / running / push),
    penalty group = the encoder's PEN_GROUP_OF ("lick"); the cued lick kernel and contact keep their groups."""
    assert fs.SPONT and fs.design_name() == "v7g3_t3d_spont"
    assert set(fs.SPONT_VARS) == set(ce.SPLIT_LICK_VARS)
    for v, g in ce.SPLIT_LICK_VARS.items():
        assert fs.FGROUP[v] == fs.SPONT_GROUP[g] == "other"
        assert fs.FPEN[v] == ce.PEN_GROUP_OF[v] == "lick"
    other = {v for v, g in fs.FGROUP.items() if g == "other"}
    assert other == {"nose_dir:spl", "nose_dir:dev", "nose_mag", "running", "push", "licks_pre", "licks_iti"}
    for v in fe.FGROUP:                                   # every v7g3_t3d variable keeps its group / penalty
        assert fs.FGROUP[v] == fe.FGROUP[v] and fs.FPEN[v] == fe.FPEN[v]
    assert {"no_spont", "novar_licks_pre", "novar_licks_iti"} <= set(fs.VARIANTS)
    assert set(fs.VARIANTS) - set(fe.VARIANTS) == {"no_spont", "novar_licks_pre", "novar_licks_iti"}
    assert fs._run_tag() == fe._run_tag() + "|licksall-split"     # never a v7g3_t3d fit's statistics


def test_raw_blocks_route_every_lick_like_the_encoder(fs, monkeypatch):
    """frozen raw_blocks under split = the encoder's split raw blocks: lick_onset = post-cue kernel, contact = the
    whole-window contacts, licks_pre / licks_iti = the encoder's own variables; the encoder env is set only inside."""
    x, B = tla._B()
    ft = x["imaging"][1]
    out = fs.raw_blocks(B, ft)
    assert ce.LICKS_ALL_ENV not in __import__("os").environ
    monkeypatch.setenv(ce.LICKS_ALL_ENV, "split")
    ref = ce.raw_blocks(B, ft)
    monkeypatch.delenv(ce.LICKS_ALL_ENV)
    k = ref["licks"].shape[1] // 2
    post, pre, iti = ce.split_licks(B)
    np.testing.assert_array_equal(out["lick_onset"], ce._event_cols(post, ft, ce.LAGS["lick"]))
    np.testing.assert_array_equal(out["lick_onset"], ref["licks"][:, :k])
    np.testing.assert_array_equal(out["contact"], ce._event_cols(B.context_contact_s, ft, ce.LAGS["lick"]))
    for v in fs.SPONT_VARS:
        np.testing.assert_array_equal(out[v], ref[v], err_msg=v)
    assert set(out) == {v.split(":")[0] for v in fs.FGROUP} - {"push"}
    # every lick in exactly one kernel (the encoder's split_licks)
    assert len(np.intersect1d(post, pre)) == len(np.intersect1d(post, iti)) == len(np.intersect1d(pre, iti)) == 0
    c = fs.spont_counts(B)
    assert (c["n_post"], c["n_pre"], c["n_iti"]) == (len(post), len(pre), len(iti))
    assert c["n_contacts"] == len(B.context_contact_s)


def test_v7g3_t3d_raw_blocks_byte_identical(monkeypatch):
    """The v7g3_t3d path: frozen raw_blocks = split_blocks(encoder raw_blocks) with LICKS_ALL unset, exactly, even on a
    Behavior carrying the context licks (they are not modeled)."""
    m = _load(monkeypatch, None)
    monkeypatch.delenv(ce.LICKS_ALL_ENV, raising=False)
    x, B = tla._B()
    ft = x["imaging"][1]
    got = m.raw_blocks(B, ft)
    ref = fe.split_blocks(ce.raw_blocks(B, ft))
    assert set(got) == set(ref) and not set(m.SPONT_VARS) & set(got)
    for v in ref:
        np.testing.assert_array_equal(got[v], ref[v], err_msg=v)


def test_inherited_encoder_licks_all_refused(fs, monkeypatch):
    """A frame render never inherits the O2 bundle's COMBINED_ENCODING_LICKS_ALL, in either design."""
    x, B = tla._B()
    ft = x["imaging"][1]
    m0 = _load(monkeypatch, None)
    for mod in (fs, m0):
        monkeypatch.setenv(ce.LICKS_ALL_ENV, "split")
        with pytest.raises(ValueError, match="COMBINED_ENCODING_LICKS_ALL|licks-all"):
            mod.raw_blocks(B, ft)
        monkeypatch.delenv(ce.LICKS_ALL_ENV)
    with pytest.raises(ValueError, match="licks_pre"):     # split design fed v7g3_t3d raw blocks
        fs.split_blocks(ce.raw_blocks(B, ft))


def test_no_spont_and_novar_drop_exactly_the_spontaneous_columns(fs):
    col_var = np.array(["cue", "cue", "licks_pre", "licks_pre", "lick_onset:dev", "licks_iti", "running"])
    C, src = fs._variant_transform("no_spont", col_var, np.eye(len(col_var)))
    assert list(col_var[src]) == ["cue", "cue", "lick_onset:dev", "running"] and C.shape == (7, 4)
    _C, src = fs._variant_transform("novar_licks_pre", col_var, np.eye(len(col_var)))
    assert "licks_pre" not in col_var[src] and "licks_iti" in col_var[src]
    _C, src = fs._variant_transform("no_other", col_var, np.eye(len(col_var)))   # drop:uninstructed incl. spont licks
    assert set(col_var[src]) == {"cue", "lick_onset:dev"}


def test_design_tag_and_guard(fs, monkeypatch):
    assert fs.design_tag("gate-attempted_v7g3_t3d_frA_ls_basis") == "gate-attempted_v7g3_t3d_spont_frA_ls_basis"
    assert fs.design_tag("v7g3_t3d_ref") == "v7g3_t3d_spont_ref"
    assert fs.design_tag("allsession_encoder_v7g3_t3d_frA") == "allsession_encoder_v7g3_t3d_spont_frA"
    assert fs.design_tag("v7g3_t3d_spont_ref") == "v7g3_t3d_spont_ref"                # idempotent
    fs.check_tag("gate-attempted_v7g3_t3d_spont")
    with pytest.raises(SystemExit, match="spont"):
        fs.check_tag("gate-attempted_v7g3_t3d")                                       # never into a v7g3_t3d folder
    m0 = _load(monkeypatch, None)
    m0.check_tag("gate-attempted_v7g3_t3d")
    with pytest.raises(SystemExit):
        m0.check_tag("gate-attempted_v7g3_t3d_spont_frA")                             # and back
    assert m0.design_tag("gate-attempted_v7g3_t3d") == "gate-attempted_v7g3_t3d"


def test_main_refuses_a_mismatched_tag(fs, tmp_path):
    with pytest.raises(SystemExit, match="spont"):
        fs.main(["--tag", "gate-attempted_v7g3_t3d", "--out-root", str(tmp_path), "PS93_0814"])
    assert not any(tmp_path.iterdir())


def test_te_design_refuses_spont(monkeypatch):
    monkeypatch.setenv("FROZEN_DESIGN", "te")
    monkeypatch.setenv(fe.LICKS_ALL_ENV, "split")
    spec = importlib.util.spec_from_file_location("_fe_te_spont", fe.__file__)
    m = importlib.util.module_from_spec(spec)
    with pytest.raises(ValueError, match="te"):
        spec.loader.exec_module(m)


def test_first_lick_timing(fs, monkeypatch):
    """v7g3_t3d: the first camera lick in cue .. stop (the session npz rule, unchanged). Split design: the encoder's
    per-trial LICK TIMING (`trial_first_lick`): close spout LP late / missing -> the DAQ contact."""
    m0 = _load(monkeypatch, None)
    x, B = tla._B()
    E = B.trials[B.trials.engaged].reset_index(drop=True)
    lk = np.sort(np.concatenate([B.lick_on_s, B.extra_lick_on_s]))
    old = np.full(len(E), np.nan)
    for i, (c, s) in enumerate(zip(E.cue_s, E.stop_s)):
        j = np.searchsorted(lk, c)
        if j < len(lk) and lk[j] < s:
            old[i] = lk[j]
    t0, src0 = m0.first_lick_s(B, E)
    np.testing.assert_array_equal(t0, old)
    assert src0 is None
    t1, src1 = fs.first_lick_s(B, E)
    tm = ce.trial_first_lick(E.cue_s, E.stop_s, E.pos, lk, B.contact_s)
    np.testing.assert_array_equal(t1, tm["first_s"])
    assert (src1 == tm["src"].astype(str)).all() and "daq" in set(src1)
    close = np.char.startswith(E.pos.to_numpy().astype(str), "close")
    d = src1 == "daq"
    assert (~np.isfinite(old[d]) | close[d]).all()        # a far spout keeps a late LP onset


def test_profile_default_and_robustness_arm():
    from wfield_local import render_profile
    p = render_profile.load("cohort1_rebuild_20261103")
    assert p["env"]["FROZEN_LICKS_ALL"] == "split"
    a = render_profile.load("cohort1_rebuild_20261103", arm="frozenNoSpont")
    assert a["env"]["FROZEN_LICKS_ALL"] == "none"
    assert a["notes"]["expected_run_tag"] == p["notes"]["expected_run_tag"]   # kept apart by the FRAME tag


def test_stats_design_guard(fs, monkeypatch):
    m0 = _load(monkeypatch, None)
    old = [{"run_mode": m0._run_tag()}]
    new = [{"run_mode": fs._run_tag()}]
    assert m0.stats_design_ok(old) and not m0.stats_design_ok(new)
    assert fs.stats_design_ok(new) and not fs.stats_design_ok(old)
