"""The STORED frozen pre-stroke decoders are the DEFAULT (11-02, Priya: "we should cache frozen decoders as we always
do"; CLAUDE.md rule 10). Pinned here, for the two families that used to refit them in memory on every render:

  - epoch acc / 5c (`grant_confusion._collect_5c` mode "frozen" -> `_stored_epoch_decoder`), opt-out
    WIDEFIELD_EPOCH_FROZEN=0;
  - rest decoder 12.1 (`scripts/rest_migration/rest_frozen_decoder.main` -> `frozen_rest_payload`), opt-out
    WIDEFIELD_REST_FROZEN=0.

Each default must reach `frozen_models.load_or_fit`, each opt-out must not, and the model the store would freeze must
be the in-memory model bit for bit (same estimator, same rows), which is what makes a store-and-load render reproduce
an in-memory one wherever the training set is the same. No data or store is touched: `load_or_fit` is replaced by a
recorder that calls the fit it is handed.
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


class _Recorder:
    """Stands in for `frozen_models.load_or_fit`: records the spec, runs the fit (a store miss), returns its payload."""

    def __init__(self):
        self.calls = []

    def __call__(self, spec, fit, *, meta=None, refreeze=None, log=print):
        payload = fit()
        self.calls.append({"spec": spec, "meta": meta, "payload": payload})
        return payload, "frozen-new"


def _same_model(a, b):
    sa, sb = a[-1], b[-1]                                   # the LogisticRegression step
    np.testing.assert_array_equal(sa.coef_, sb.coef_)
    np.testing.assert_array_equal(sa.intercept_, sb.intercept_)
    np.testing.assert_array_equal(a[0].mean_, b[0].mean_)
    np.testing.assert_array_equal(a[0].scale_, b[0].scale_)


# ------------------------------------------------------------------------------------------------ epoch acc / 5c
def _bundle(seed=0, n_sess=4, n_per=60, n_feat=12):
    rng = np.random.default_rng(seed)
    kept = [f"PS92_08{10 + i:02d}" for i in range(n_sess)] + ["PS92_0901"]
    G = np.repeat(np.arange(len(kept)), n_per)
    Y = np.tile(np.arange(6), len(G) // 6 + 1)[:len(G)]
    X = rng.normal(size=(len(G), n_feat)) + 0.8 * Y[:, None] * rng.normal(size=(1, n_feat))
    pre_i = set(range(n_sess))
    return {"XE": X, "YE": Y, "GE": G, "kept": kept, "pre_i": pre_i,
            "e_pre": np.isin(G, list(pre_i)), "basis_id": "testbasis000"}


@pytest.fixture
def daq_env(monkeypatch):
    for k in ("WIDEFIELD_EPOCH_FROZEN", "WIDEFIELD_LICK_SOURCE", "WIDEFIELD_LP_FROZEN", "WIDEFIELD_ENL_WINDOW_S",
              "WIDEFIELD_REST_FROZEN"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_epoch_default_goes_through_load_or_fit(daq_env):
    from wfield_local import frozen_models, grant_confusion
    rec = _Recorder()
    daq_env.setattr(frozen_models, "load_or_fit", rec)
    assert grant_confusion.epoch_frozen_on()
    bd = _bundle()
    out = grant_confusion._stored_epoch_decoder("PS92", "cue", bd)
    assert len(rec.calls) == 1 and out is rec.calls[0]["payload"]
    spec = rec.calls[0]["spec"]
    assert spec["kind"] == "epoch_decoder" and spec["align"] == "cue" and spec["animal"] == "PS92"
    assert spec["train_labels"] == sorted(bd["kept"][i] for i in bd["pre_i"])   # pre only, never a post session
    assert "lick_source" not in spec                                            # a DAQ spec, byte-identical ids


@pytest.mark.parametrize("off", ["0", "false", "no", "off", " OFF "])
def test_epoch_opt_out_fits_in_memory(daq_env, off):
    from wfield_local import frozen_models, grant_confusion
    rec = _Recorder()
    daq_env.setattr(frozen_models, "load_or_fit", rec)
    daq_env.setenv("WIDEFIELD_EPOCH_FROZEN", off)
    assert not grant_confusion.epoch_frozen_on()
    assert grant_confusion._stored_epoch_decoder("PS92", "cue", _bundle()) is None
    assert rec.calls == []


@pytest.mark.parametrize("val", ["", "1", "yes", "true"])
def test_epoch_unset_or_old_opt_in_is_on(daq_env, val):
    from wfield_local import grant_confusion
    daq_env.setenv("WIDEFIELD_EPOCH_FROZEN", val)
    assert grant_confusion.epoch_frozen_on()


def test_epoch_stored_fit_is_the_in_memory_fit(daq_env):
    """full = `_pipe().fit(XE[e_pre], YE[e_pre])` and loso[label] = the fit without that session, bit for bit."""
    from wfield_local import frozen_models, grant_confusion
    from wfield_local.locanmf_frozen_decoder import _pipe
    rec = _Recorder()
    daq_env.setattr(frozen_models, "load_or_fit", rec)
    bd = _bundle(seed=3)
    pay = grant_confusion._stored_epoch_decoder("PS92", "lick", bd)
    XE, YE, GE, e_pre = bd["XE"], bd["YE"], bd["GE"], bd["e_pre"]
    _same_model(pay["full"], _pipe().fit(XE[e_pre], YE[e_pre]))
    assert sorted(pay["loso"]) == sorted(bd["kept"][i] for i in bd["pre_i"])
    for i in sorted(bd["pre_i"]):
        tr = e_pre & (GE != i)
        _same_model(pay["loso"][bd["kept"][i]], _pipe().fit(XE[tr], YE[tr]))


def test_collect_5c_frozen_mode_uses_the_store():
    """The acc family's collector asks the store in mode "frozen" and only falls back to a fit when opted out."""
    from wfield_local import grant_confusion
    src = inspect.getsource(grant_confusion._collect_5c)
    assert 'if mode in ("frozen", "paired") else None)' in src
    assert 'clf = stored["full"] if stored else _pipe().fit(XE[e_pre], YE[e_pre])' in src
    assert 'stored["loso"].get(kept[i]) if stored else None' in src


# ------------------------------------------------------------------------------- 5c paired (5r / 5ro) and 5b (11-02)
def _full_bundle(seed=0, n_sess=4, n_per=60, n_feat=12, n_post=2):
    """A `_pooled_bundle`-shaped dict: engaged rows (E) for n_sess pre + n_post post sessions, unengaged rows (U) for
    the post sessions with half of them in the terminal quit period."""
    rng = np.random.default_rng(seed)
    kept = [f"PS92_08{10 + i:02d}" for i in range(n_sess)] + [f"PS92_09{1 + i:02d}" for i in range(n_post)]
    GE = np.repeat(np.arange(len(kept)), n_per)
    YE = np.tile(np.arange(6), len(GE) // 6 + 1)[:len(GE)]
    W = rng.normal(size=(1, n_feat))
    XE = rng.normal(size=(len(GE), n_feat)) + 0.8 * YE[:, None] * W
    GU = np.repeat(np.arange(n_sess, len(kept)), 12)
    YU = np.tile(np.arange(6), len(GU) // 6 + 1)[:len(GU)]
    XU = rng.normal(size=(len(GU), n_feat)) + 0.3 * YU[:, None] * W
    pre_i = set(range(n_sess))
    return {"XE": XE, "YE": YE, "GE": GE, "XU": XU, "YU": YU, "GU": GU, "kept": kept, "pre_i": pre_i,
            "e_pre": np.isin(GE, list(pre_i)), "not_eng": np.arange(len(GU)) % 2 == 1,
            "BE": GE * 100 + np.arange(len(GE)) // 6, "BU": 10000 + GU * 100 + np.arange(len(GU)) // 6,
            "basis_id": "testbasis000"}


def _collect(gc, monkeypatch, mode, variant="working", seed=5):
    bd = _full_bundle(seed=seed)
    monkeypatch.setattr(gc, "ANIMALS", ("PS92",))
    monkeypatch.setattr(gc, "_pooled_bundle", lambda an, align: bd)
    return gc._collect_5c.__wrapped__("cue", variant, mode)


def _same_records(a, b):
    assert a[1] == b[1]
    for an in a[0]:
        pa, pb = a[0][an][0], b[0][an][0]
        for ra, rb in (zip(pa, pb) if isinstance(pa, list) else [(pa, pb)]):
            for x, y in zip(ra, rb):
                np.testing.assert_array_equal(x, y)
        assert sorted(a[0][an][1]) == sorted(b[0][an][1])
        for d in a[0][an][1]:
            for x, y in zip(a[0][an][1][d], b[0][an][1][d]):
                np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("variant", ["working", "lick"])
def test_collect_5c_paired_stored_equals_in_memory(daq_env, variant):
    """Mode "paired" (5r / 5ro / recovery_trajectory) asks the store, and its records -- frozen AND refit columns,
    pre LOSO list and every post day -- equal the opt-out in-memory records EXACTLY."""
    from wfield_local import frozen_models
    from wfield_local import grant_confusion as gc
    rec = _Recorder()
    daq_env.setattr(frozen_models, "load_or_fit", rec)
    got = _collect(gc, daq_env, "paired", variant)
    assert len(rec.calls) == 1 and rec.calls[0]["spec"]["kind"] == "epoch_decoder"
    assert "lick_source" not in rec.calls[0]["spec"]
    daq_env.setenv("WIDEFIELD_EPOCH_FROZEN", "0")
    ref = _collect(gc, daq_env, "paired", variant)
    assert len(rec.calls) == 1                                       # the opt-out never reached the store
    assert got[0]["PS92"][1], "no post-stroke day was scored"
    _same_records(got, ref)
    some = next(iter(got[0]["PS92"][1].values()))
    assert some[1].ndim == 2 and some[1].shape[1] == 2               # (frozen, refit) columns


def test_collect_5c_frozen_column_of_paired_is_the_frozen_mode(daq_env):
    """The paired frozen column IS mode "frozen" (same stored model), so 5r and acc can never disagree on it."""
    from wfield_local import frozen_models
    from wfield_local import grant_confusion as gc
    daq_env.setattr(frozen_models, "load_or_fit", _Recorder())
    fz = _collect(gc, daq_env, "frozen")
    pr = _collect(gc, daq_env, "paired")
    for d, (y, p, b) in fz[0]["PS92"][1].items():
        yp, pp, bp = pr[0]["PS92"][1][d]
        np.testing.assert_array_equal(y, yp)
        np.testing.assert_array_equal(b, bp)
        np.testing.assert_array_equal(p, pp[:, 0])


def test_collect_5c_refit_does_not_ask_the_store(daq_env):
    """No frozen column (refit): in memory by design."""
    from wfield_local import frozen_models
    from wfield_local import grant_confusion as gc
    rec = _Recorder()
    daq_env.setattr(frozen_models, "load_or_fit", rec)
    daq_env.setattr(frozen_models, "find", lambda spec: (None, None))
    _collect(gc, daq_env, "refit")
    assert rec.calls == []


def test_collect_5c_matched_draws_stored_in_one_seed_keyed_spec(daq_env):
    """Rebuild 11-03 (item h): 5rm's size-matched draws go to the store under ONE spec per animal x align x class
    (kind matched_frozen); stored, replayed and in-memory records are identical."""
    from wfield_local import frozen_models
    from wfield_local import grant_confusion as gc
    rec = _Recorder()
    daq_env.setattr(frozen_models, "load_or_fit", rec)
    daq_env.setattr(frozen_models, "find", lambda spec: (None, None))
    stored = _collect(gc, daq_env, "paired_matched")
    assert len(rec.calls) == 1 and rec.calls[0]["spec"]["kind"] == "matched_frozen"
    assert "seed" in rec.calls[0]["spec"]["align"] and rec.calls[0]["payload"]
    payload = rec.calls[0]["payload"]
    daq_env.setattr(frozen_models, "find", lambda spec: ("hit", "local"))      # replay from the stored payload
    daq_env.setattr(frozen_models, "_read", lambda d: payload)
    replay = _collect(gc, daq_env, "paired_matched")
    assert len(rec.calls) == 1                                                  # nothing refitted / rewritten
    daq_env.setenv("WIDEFIELD_EPOCH_FROZEN", "0")
    mem = _collect(gc, daq_env, "paired_matched")
    assert len(rec.calls) == 1
    _same_records(stored, mem)
    _same_records(replay, mem)


def _render_5b(gc, monkeypatch, out):
    bd = _full_bundle(seed=11)
    monkeypatch.setattr(gc, "ANIMALS", ("PS92",))
    monkeypatch.setattr(gc, "_pooled_bundle", lambda an, align: bd)
    monkeypatch.setattr(gc, "_fig_root", lambda: Path(out) / "no_such_root")   # no coding_direction.json -> recompute
    gc.fig_confusion_pre_post_working(out)
    import pandas as pd
    return {al: pd.read_csv(Path(out) / "data" / f"grant_5b_confusion_working_{al}.csv") for al in ("precue", "cue")}


def test_5b_stored_equals_in_memory(daq_env, tmp_path):
    """5b's pre (LOSO) / post_all / post_working counts, and the per-session split, are identical under the stored
    default and the opt-out; the default reaches the store once per (animal, window) with the acc family's spec."""
    from wfield_local import frozen_models
    from wfield_local import grant_confusion as gc
    rec = _Recorder()
    daq_env.setattr(frozen_models, "load_or_fit", rec)
    got = _render_5b(gc, daq_env, tmp_path / "stored")
    assert [c["spec"]["align"] for c in rec.calls] == ["precue", "cue"]
    assert all(c["spec"]["kind"] == "epoch_decoder" and "lick_source" not in c["spec"] for c in rec.calls)
    daq_env.setenv("WIDEFIELD_EPOCH_FROZEN", "0")
    ref = _render_5b(gc, daq_env, tmp_path / "memory")
    assert len(rec.calls) == 2
    for al in ("precue", "cue"):
        a = got[al][got[al].panel != "decoder"].reset_index(drop=True)
        b = ref[al][ref[al].panel != "decoder"].reset_index(drop=True)
        assert set(a.panel) == {"pre", "post_all", "post_working", "post_working_session"}
        import pandas as pd
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        assert got[al][got[al].panel == "decoder"]["true"].tolist() == ["store"]
        assert ref[al][ref[al].panel == "decoder"]["true"].tolist() == ["memory"]
        # the per-session split sums to the drawn post_working panel
        lab = list(gc.CONF_LABELS)
        ps = a[a.panel == "post_working_session"].groupby("true", sort=False)[lab].sum()
        pw = a[a.panel == "post_working"].set_index("true")[lab]
        np.testing.assert_array_equal(ps.loc[pw.index].to_numpy(), pw.to_numpy())


def test_5b_uses_the_stored_epoch_decoder():
    from wfield_local import grant_confusion as gc
    src = inspect.getsource(gc.fig_confusion_pre_post_working)
    assert 'frozen = _stored_epoch_decoder(an, align, bd, tag="5b")' in src
    assert 'clf = frozen["full"] if frozen else _pipe().fit(XE[e_pre], YE[e_pre])' in src
    assert 'frozen["loso"].get(bd["kept"][i]) if frozen else None) or _pipe().fit(XE[tr], YE[tr])' in src


@pytest.mark.parametrize("lp_frozen", [None, "lp", "daq"])
def test_lp_arm_spec_is_lp_trained_by_default_and_daq_trained_on_request(daq_env, lp_frozen):
    """Under WIDEFIELD_LICK_SOURCE=lp the 5r / 5b frozen model is the acc family's: since 11-03 (Priya) the LP-TRAINED
    spec (lick_source=lp, the LP-kept pre sessions) by default -- "lp", the old opt-in name, is the same spec -- and the
    DAQ-trained one (no lick_source; every pre session of the animal) with WIDEFIELD_LP_FROZEN=daq. Never one id."""
    from wfield_local import config, frozen_models
    from wfield_local import grant_confusion as gc
    specs = {}
    for lf in (None, "lp", "daq"):
        rec = _Recorder()
        daq_env.setattr(frozen_models, "load_or_fit", rec)
        daq_env.setenv("WIDEFIELD_LICK_SOURCE", "lp")
        if lf:
            daq_env.setenv("WIDEFIELD_LP_FROZEN", lf)
        else:
            daq_env.delenv("WIDEFIELD_LP_FROZEN", raising=False)
        _collect(gc, daq_env, "paired")
        specs[lf] = rec.calls[0]["spec"]
    s = specs[lp_frozen]
    if lp_frozen == "daq":
        assert "lick_source" not in s
        assert s["train_labels"] == sorted(x for x in config.phase_labels("pre") if config.animal_of(x) == "PS92")
    else:
        assert s["lick_source"] == "lp"
        assert s["train_labels"] == sorted(_full_bundle()["kept"][i] for i in range(4))
    assert frozen_models.spec_id(specs[None]) == frozen_models.spec_id(specs["lp"])
    assert frozen_models.spec_id(specs[None]) != frozen_models.spec_id(specs["daq"])


# ------------------------------------------------------------------------------------- siblings() grouping (11-02)
def _put(root, spec, stamp):
    import json
    from wfield_local import frozen_models
    d = Path(root) / spec["animal"] / frozen_models._slug(spec)
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"spec": spec, "frozen_utc": stamp}), encoding="utf-8")
    return d


def _spec(**kw):
    s = {"freeze_version": 1, "animal": "PS92", "kind": "epoch_decoder", "align": "precue", "source": "locanmf",
         "basis_id": None, "post_s": 2.0, "zscore": True, "alpha": None, "n_features": 10,
         "train_labels": ["PS92_0810", "PS92_0811"], "train_sigs": {}}
    s.update(kw)
    return s


def test_siblings_group_by_window_and_lick_source(daq_env, tmp_path):
    """A 1.0 s model, a 2.0 s model and an LP-trained model of the same kind / align / source are NOT each other's
    siblings (no false SPEC-CHANGED); a changed training set at the same window and source still is."""
    from wfield_local import frozen_models
    daq_env.setenv("WIDEFIELD_FROZEN_MODEL_DIR", str(tmp_path))
    daq_env.setattr(frozen_models, "server_dir", lambda: None)
    s2 = _spec()
    s1 = _spec(post_s=1.0)
    slp = _spec(post_s=1.0, lick_source="lp")
    s2b = _spec(train_labels=["PS92_0810", "PS92_0811", "PS92_0812"])
    for i, s in enumerate((s2, s1, slp)):
        _put(tmp_path, s, f"2026-11-02T0{i}:00:00")
    names = lambda sp: [r["dir"].name for r in frozen_models.siblings(sp)]       # noqa: E731
    assert names(s2) == [frozen_models._slug(s2)]
    assert names(s1) == [frozen_models._slug(s1)]
    assert names(slp) == [frozen_models._slug(slp)]
    assert names(s2b) == [frozen_models._slug(s2)]                    # same window / source: still a sibling
    # the ids themselves are unchanged by the grouping and all differ
    assert len({frozen_models.spec_id(s) for s in (s2, s1, slp, s2b)}) == 4


def test_load_or_fit_status_new_across_window_changed_within(daq_env, tmp_path):
    from wfield_local import frozen_models
    daq_env.setenv("WIDEFIELD_FROZEN_MODEL_DIR", str(tmp_path))
    daq_env.setattr(frozen_models, "server_dir", lambda: None)
    _put(tmp_path, _spec(), "2026-11-02T00:00:00")
    logs = []
    _p, st = frozen_models.load_or_fit(_spec(post_s=1.0), lambda: {"m": 1}, log=logs.append)
    assert st == frozen_models.STATUS_NEW and not any("DIFFERENT" in x for x in logs)
    _p, st = frozen_models.load_or_fit(_spec(train_labels=["PS92_0810"]), lambda: {"m": 2}, log=logs.append)
    assert st == frozen_models.STATUS_SPEC_CHANGED
    _p, st = frozen_models.load_or_fit(_spec(post_s=1.0), lambda: {"m": 3}, log=logs.append)
    assert st == frozen_models.STATUS_HIT and _p == {"m": 1}


# ------------------------------------------------------------------------------------------------ rest decoder 12.1
def _rest_data(seed=0, n_pre=4, n=120, n_feat=8):
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_pre + 1):
        lab = f"PS94_08{10 + i:02d}" if i < n_pre else "PS94_0905"
        y = np.tile(np.arange(6), n // 6)
        X = rng.normal(size=(n, n_feat)) + 0.7 * y[:, None] * rng.normal(size=(1, n_feat))
        g = np.repeat(np.arange(n // 6), 6)
        dur = rng.uniform(0.5, 5.0, n)
        gap = rng.uniform(0.2, 6.0, n)
        data[lab] = ("pre" if i < n_pre else "acute", X, y, g, dur, gap, 1.0)
    return data


def _rf():
    return pytest.importorskip("scripts.rest_migration.rest_frozen_decoder")


def test_rest_default_on_and_opt_out(daq_env):
    rf = _rf()
    assert rf.rest_frozen_on()
    for off in ("0", "false", "no", "off"):
        daq_env.setenv("WIDEFIELD_REST_FROZEN", off)
        assert not rf.rest_frozen_on()


def test_rest_page_spec_is_the_nl_scripts_spec():
    """The page run (4 bins, duration + lick-gap matched at 10 %, gate, 40 / 5, restdock05) names the same model as
    `rest_frozen_decoder_nl --mask rest`, so the two scripts share one stored reference per animal."""
    rf = _rf()
    assert rf.rest_frozen_align(4, True, True, 10.0, True, 40, 5, "restdock05") == "rest-rest-b4-dur10-gap10-gate"
    nl_src = (ROOT / "scripts" / "rest_migration" / "rest_frozen_decoder_nl.py").read_text(encoding="utf-8")
    assert 'align=f"rest-{mask}-b{BINS}-dur{PCT:g}-gap{PCT:g}-gate"' in nl_src
    assert 'source=f"joint_{basis_id}"' in nl_src and "post_s=0.0, zscore=True, alpha=0.1" in nl_src
    # any change to what selects the training rows changes the identity
    assert rf.rest_frozen_align(4, False, True, 10.0, True, 40, 5, "restdock05") != "rest-rest-b4-dur10-gap10-gate"
    assert rf.rest_frozen_align(4, True, True, 10.0, False, 40, 5, "restdock05").endswith("-nogate")
    assert rf.rest_frozen_align(4, True, True, 10.0, True, 30, 5, "restdock05").endswith("-min30x5")


@pytest.mark.parametrize("match", [(True, True), (False, False), (True, False)])
def test_rest_default_goes_through_load_or_fit_and_matches_in_memory(daq_env, match):
    """`frozen_rest_payload` reaches `load_or_fit`, and its full / LOSO models are `main`'s in-memory `_fit_frozen`
    on `training_pool` rows with the same duration / lick-gap windows, bit for bit."""
    from wfield_local import frozen_models
    rf = _rf()
    rec = _Recorder()
    daq_env.setattr(frozen_models, "load_or_fit", rec)
    data = _rest_data(seed=7)
    pre_labs = [k for k, v in data.items() if v[0] == "pre"]
    pre_dur = np.concatenate([data[k][4] for k in pre_labs])
    pre_gap = np.concatenate([data[k][5] for k in pre_labs])
    lo_s, hi_s = (float(np.percentile(pre_dur, 10)), float(np.percentile(pre_dur, 90))) if match[0] else (None, None)
    glo, ghi = (float(np.percentile(pre_gap, 10)), float(np.percentile(pre_gap, 90))) if match[1] else (None, None)
    pay, status, spec = rf.frozen_rest_payload("PS94", "basisxyz", data, pre_labs, (lo_s, hi_s, glo, ghi), bins=4,
                                               pct=10.0, gate=True, min_periods=40, min_per_class=5,
                                               variant="restdock05")
    assert status == "frozen-new" and len(rec.calls) == 1 and spec["kind"] == "rest_frozen"
    assert spec["train_labels"] == sorted(pre_labs)

    def in_memory(lab):                                  # main's own row selection, verbatim
        train = rf.training_pool(pre_labs, lab)
        Xt = np.concatenate([data[k][1] for k in train])
        yt = np.concatenate([data[k][2] for k in train])
        if lo_s is not None:
            dt = np.concatenate([data[k][4] for k in train])
            m_tr = (dt >= lo_s) & (dt <= hi_s)
            Xt, yt = Xt[m_tr], yt[m_tr]
        if glo is not None:
            gt = np.concatenate([data[k][5] for k in train])
            if lo_s is not None:
                gt = gt[m_tr]
            m_g = (gt >= glo) & (gt <= ghi)
            Xt, yt = Xt[m_g], yt[m_g]
        return rf._fit_frozen(Xt, yt)

    _same_model(pay["full"], in_memory("PS94_0905"))
    for lab in pre_labs:
        _same_model(pay["loso"][lab], in_memory(lab))


def test_rest_main_asks_the_store_by_default():
    rf = _rf()
    src = inspect.getsource(rf.main)
    assert "if rest_frozen_on():" in src and "frozen_rest_payload(" in src
    assert 'model = payload["loso"].get(lab) if ep == "pre" else payload["full"]' in src


def test_rest_digested_sources_unchanged():
    """`rest_frozen_decoder_nl._code_key` hashes `_collect` + `_frame_samples`: both must stay source-identical to the
    commit that keyed the collected-feature cache (7c53214a), or every cached session would be recollected."""
    rf = _rf()
    try:
        old = subprocess.run(["git", "-C", str(ROOT), "show", "7c53214a:scripts/rest_migration/rest_frozen_decoder.py"],
                             capture_output=True, text=True, check=True, encoding="utf-8").stdout
    except Exception:                                                    # noqa: BLE001
        pytest.skip("git history not available")
    for fn in (rf._collect, rf._frame_samples):
        assert inspect.getsource(fn) in old, fn.__name__


# ================================================================================ pass 2 (11-02): more families stored
def _pool(seed=0, n_sess=4, n_post=2, n_per=60, n_feat=10, animal="PS92"):
    """Pooled engaged rows (X, y int, session label per row) for n_sess pre + n_post post sessions."""
    rng = np.random.default_rng(seed)
    labs = [f"{animal}_08{10 + i:02d}" for i in range(n_sess)] + [f"{animal}_09{1 + i:02d}" for i in range(n_post)]
    S = np.repeat(np.array(labs, dtype=object), n_per)
    Y = np.tile(np.arange(6), len(S) // 6 + 1)[:len(S)].astype(int)
    W = rng.normal(size=(1, n_feat))
    X = rng.normal(size=(len(S), n_feat)) + 0.8 * Y[:, None] * W
    return X, Y, S, labs[:n_sess], labs


# ------------------------------------------------------------------------------------------- nolick reference (1)
@pytest.fixture
def pass2_env(daq_env):
    for k in ("WIDEFIELD_NOLICK_FROZEN", "WIDEFIELD_ATTEMPT_FROZEN", "WIDEFIELD_DRIFT_FROZEN",
              "WIDEFIELD_CODING_FROZEN", "WIDEFIELD_CD_FROZEN", "WIDEFIELD_SECTION_G_FROZEN",
              "WIDEFIELD_ITI_FROZEN", "WIDEFIELD_ENCODER_FROZEN"):
        daq_env.delenv(k, raising=False)
    return daq_env


def test_family_on_opt_out_values(pass2_env):
    from wfield_local import frozen_models
    for v in ("", "1", "yes"):
        pass2_env.setenv("WIDEFIELD_X_FROZEN", v)
        assert frozen_models.family_on("WIDEFIELD_X_FROZEN")
    for v in ("0", "false", "no", "off", " OFF "):
        pass2_env.setenv("WIDEFIELD_X_FROZEN", v)
        assert not frozen_models.family_on("WIDEFIELD_X_FROZEN")


def _nolick_args(seed=0):
    X, Y, S, pre, labs = _pool(seed)
    m_pre = np.isin(S, pre)
    sess = [{"label": lab} for lab in labs]
    return X, Y, S, m_pre, set(pre), sess


def test_nolick_stored_equals_in_memory(pass2_env):
    """full = `_pipe().fit(XE[m_pre], YE[m_pre])`; the stored LOSO predictions = `cross_val_predict(_pipe(), ...,
    LeaveOneGroupOut)` EXACTLY; the default reaches the store once, the opt-out never."""
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

    from wfield_local import frozen_models
    from wfield_local import nolick_decoder as nd
    from wfield_local.locanmf_frozen_decoder import _pipe
    rec = _Recorder()
    pass2_env.setattr(frozen_models, "load_or_fit", rec)
    X, Y, S, m_pre, pre, sess = _nolick_args(seed=4)
    common = [("SSp", 0), ("MOp", 0)]
    pay = nd._stored_nolick_reference("PS92", "precue", 2.0, None, X, Y, S, m_pre, pre, sess, common)
    assert len(rec.calls) == 1
    spec = rec.calls[0]["spec"]
    assert spec["kind"] == "nolick_reference" and spec["source"] == "roi" and "lick_source" not in spec
    assert spec["train_labels"] == sorted(pre) and spec["align"].startswith("precue-rt2-f")
    _same_model(pay["full"], _pipe().fit(X[m_pre], Y[m_pre]))
    ref = cross_val_predict(_pipe(), X[m_pre], Y[m_pre], cv=LeaveOneGroupOut(), groups=S[m_pre])
    got = nd._loso_predict(pay["loso"], X[m_pre], S[m_pre], Y.dtype)
    np.testing.assert_array_equal(got, ref)
    assert got.dtype == ref.dtype
    pass2_env.setenv("WIDEFIELD_NOLICK_FROZEN", "0")
    assert nd._stored_nolick_reference("PS92", "precue", 2.0, None, X, Y, S, m_pre, pre, sess, common) is None
    assert len(rec.calls) == 1


def test_nolick_spec_distinguishes_cut_basis_window_and_features(pass2_env):
    from types import SimpleNamespace

    from wfield_local import frozen_models
    from wfield_local import nolick_decoder as nd
    tr = ["PS92_0810", "PS92_0811"]
    ids = {frozen_models.spec_id(nd.nolick_spec("PS92", al, rt, b, tr, com, 8, None))
           for al in ("precue", "cue") for rt in (2.0, None)
           for b in (None, SimpleNamespace(basis_id="abc123def456"))
           for com in ([("SSp", 0)], [("SSp", 0), ("MOp", 0)])}
    assert len(ids) == 16
    assert frozen_models.spec_id(nd.nolick_spec("PS92", "cue", 2.0, None, tr, [], 8, None)) != \
        frozen_models.spec_id(nd.nolick_spec("PS92", "cue", 2.0, None, tr + ["PS92_0812"], [], 8, None))


def test_nolick_lp_arm_lp_trained_by_default(pass2_env):
    from wfield_local import config, frozen_models
    from wfield_local import nolick_decoder as nd
    X, Y, S, m_pre, pre, _sess = _nolick_args()
    real = [s for s in config.load_sessions() if s["label"].startswith("PS92")]
    rp = [s["label"] for s in real if config.session_phase("PS92", s["label"].split("_")[-1]) == "pre"]
    specs = {}
    for lf in ("daq", None):
        rec = _Recorder()
        pass2_env.setattr(frozen_models, "load_or_fit", rec)
        pass2_env.setenv("WIDEFIELD_LICK_SOURCE", "lp")
        if lf:
            pass2_env.setenv("WIDEFIELD_LP_FROZEN", lf)
        else:
            pass2_env.delenv("WIDEFIELD_LP_FROZEN", raising=False)
        nd._stored_nolick_reference("PS92", "cue", 2.0, None, X, Y, S, m_pre, pre, real, [])
        specs[lf] = rec.calls[0]["spec"]
    assert "lick_source" not in specs["daq"] and specs["daq"]["train_labels"] == sorted(rp)      # opt-in cross-check
    assert specs[None]["lick_source"] == "lp" and specs[None]["train_labels"] == sorted(pre)    # LP-trained default
    assert frozen_models.spec_id(specs[None]) != frozen_models.spec_id(specs["daq"])


def test_nolick_analyse_animal_uses_the_store():
    from wfield_local import nolick_decoder as nd
    src = inspect.getsource(nd.analyse_animal)
    assert "stored = _stored_nolick_reference(" in src
    assert 'clf = stored["full"] if stored is not None else _pipe().fit(XE[m_pre], YE[m_pre])' in src
    assert 'pred_e = _loso_predict(stored["loso"], XE[m_pre], SE[m_pre], YE.dtype)' in src


# ------------------------------------------------------------------------------------- attempt_coding axes (2)
def _same_axes(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        for x, y in zip(a[k], b[k]):
            np.testing.assert_array_equal(x, y)


def _attempt_rows(seed=0):
    from wfield_local import attempt_coding as ac
    rng = np.random.default_rng(seed)
    kept = ["PS92_0810", "PS92_0811", "PS92_0812", "PS92_0901"]
    G = np.repeat(np.arange(4), 90)
    sp = np.tile(np.array(ac.SPOUTS), 60)[:len(G)]
    mu = {P: rng.normal(size=12) for P in ac.SPOUTS}
    X = np.vstack([mu[p] for p in sp]) + rng.normal(size=(len(G), 12))
    ref = G < 3
    return X, G, sp, ref, kept


def test_attempt_axes_stored_equal_in_memory(pass2_env):
    from wfield_local import attempt_coding as ac
    from wfield_local import frozen_models
    rec = _Recorder()
    pass2_env.setattr(frozen_models, "load_or_fit", rec)
    X, G, sp, ref, kept = _attempt_rows(seed=2)
    pay = ac.stored_axes("PS92", "precue", 1.0, X, G, sp, ref, kept, "basis0", verbose=False)
    spec = rec.calls[0]["spec"]
    assert spec["kind"] == "attempt_axes" and spec["lick_source"] == "lp"        # LP-only by nature
    assert spec["train_labels"] == kept[:3] and spec["align"] == "precue-dom"
    for meth in ac.METHODS:
        _same_axes(pay[meth]["full"], ac.fit_axes(X[ref], sp[ref], meth))
        for s in range(3):
            _same_axes(pay[meth]["loso"][kept[s]], ac.fit_axes(X[ref & (G != s)], sp[ref & (G != s)], meth))
    pass2_env.setenv("WIDEFIELD_ATTEMPT_FROZEN", "0")
    assert ac.stored_axes("PS92", "precue", 1.0, X, G, sp, ref, kept, "basis0", verbose=False) is None
    assert len(rec.calls) == 1


def test_attempt_run_animal_uses_the_store():
    from wfield_local import attempt_coding as ac
    src = inspect.getsource(ac.run_animal)
    assert "axes = stored_axes(animal, align, post_s, X, G, sp, ref, kept" in src
    assert 'W = axes[meth]["full"] if axes else fit_axes(X[ref], sp[ref], meth)' in src
    assert 'axes[meth]["loso"][kept[s]] if axes else fit_axes(X[tr], sp[tr], meth)' in src


# ------------------------------------------------------------------------------------- june_august_drift (3)
def _drift_bundle(seed=0):
    rng = np.random.default_rng(seed)
    kept = ["PS92_0606", "PS92_0607", "PS92_0608", "PS92_0810", "PS92_0811", "PS92_0812", "PS92_0901"]
    GE = np.repeat(np.arange(len(kept)), 48)
    YE = np.tile(np.arange(6), len(GE) // 6).astype(int)
    W = rng.normal(size=(1, 9))
    XE = rng.normal(size=(len(GE), 9)) + 0.7 * YE[:, None] * W
    pre_i = set(range(6))
    return {"XE": XE, "YE": YE, "GE": GE, "kept": kept, "pre_i": pre_i, "e_pre": np.isin(GE, list(pre_i)),
            "basis_id": "testbasis000"}


def test_drift_stored_equals_in_memory_for_every_training_set(pass2_env):
    """Every training set `decoder_rows.score` uses (June LOSO, all June, Aug LOSO, all Aug) is served by the
    month's stored payload, and each equals `_pipe().fit(XE[e_pre & isin(GE, train)], ...)` bit for bit."""
    jad = pytest.importorskip("scripts.june_august_drift")
    from wfield_local import frozen_models
    from wfield_local.locanmf_frozen_decoder import _pipe
    rec = _Recorder()
    pass2_env.setattr(frozen_models, "load_or_fit", rec)
    jad._DRIFT_MODELS.clear()
    bd = _drift_bundle(seed=6)
    J, A = [0, 1, 2], [3, 4, 5]
    sets = [[j for j in J if j != i] for i in J] + [J] + [[j for j in A if j != i] for i in A] + [A]
    for tr_s in sets:
        tr = bd["e_pre"] & np.isin(bd["GE"], tr_s)
        _same_model(jad._stored_month_model("PS92", bd, tr_s), _pipe().fit(bd["XE"][tr], bd["YE"][tr]))
    assert [c["spec"]["align"] for c in rec.calls] == ["cue-june", "cue-aug"]          # one fit per month
    assert rec.calls[0]["spec"]["train_labels"] == bd["kept"][:3]
    assert rec.calls[1]["spec"]["train_labels"] == bd["kept"][3:6]
    assert all(c["spec"]["kind"] == "drift_decoder" for c in rec.calls)
    assert frozen_models.spec_id(rec.calls[0]["spec"]) != frozen_models.spec_id(rec.calls[1]["spec"])
    with pytest.raises(ValueError):
        jad._stored_month_model("PS92", bd, [0, 3])                                    # spans months
    jad._DRIFT_MODELS.clear()
    pass2_env.setenv("WIDEFIELD_DRIFT_FROZEN", "0")
    assert jad._stored_month_model("PS92", bd, J) is None and len(rec.calls) == 2


def test_drift_lp_arm_lp_trained_by_default(pass2_env):
    jad = pytest.importorskip("scripts.june_august_drift")
    from wfield_local import config, frozen_models
    bd = _drift_bundle()
    pass2_env.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    specs = {}
    for lf in (None, "daq"):
        rec = _Recorder()
        pass2_env.setattr(frozen_models, "load_or_fit", rec)
        if lf:
            pass2_env.setenv("WIDEFIELD_LP_FROZEN", lf)
        jad.drift_payload("PS92", bd, "aug", log=lambda *_: None)
        specs[lf] = rec.calls[0]["spec"]
    want = [x for x in config.phase_labels("pre") if config.animal_of(x) == "PS92" and x.split("_")[-1][:2] == "08"]
    assert "lick_source" not in specs["daq"] and specs["daq"]["train_labels"] == sorted(want)
    assert specs[None]["lick_source"] == "lp" and specs[None]["train_labels"] == bd["kept"][3:6]


def test_drift_decoder_rows_uses_the_store():
    jad = pytest.importorskip("scripts.june_august_drift")
    src = inspect.getsource(jad.decoder_rows)
    assert "stored = _stored_month_model(an, bd, train_sessions)" in src
    assert "(stored if stored is not None else _pipe().fit(XE[tr], YE[tr])).predict(X)" in src


# ------------------------------------------------------------------------------- position_coding_directions (4)
def _cd_pool(seed=0, n_pre=3, n_post=2, n_per=150, n_feat=10):
    from wfield_local import position_coding_directions as pcd
    rng = np.random.default_rng(seed)
    kept = [f"PS92_08{10 + i:02d}" for i in range(n_pre)] + [f"PS92_09{1 + i:02d}" for i in range(n_post)]
    GE = np.repeat(np.arange(len(kept)), n_per)
    pos = np.array(pcd.BY_SEVERITY)
    en = np.tile(pos, len(GE) // len(pos) + 1)[:len(GE)]
    mu = {P: rng.normal(size=n_feat) for P in pos}
    XE = np.vstack([mu[p] for p in en]) + rng.normal(size=(len(GE), n_feat))
    GU = np.repeat(np.arange(len(kept)), 40)
    XU = rng.normal(size=(len(GU), n_feat)) * 1.3
    pre_i = set(range(n_pre))
    return XE, en, np.isin(GE, list(pre_i)), GE, XU, np.isin(GU, list(pre_i)), pre_i, kept


def _old_coding_loop(XE, en, e_pre, GE, XU, u_pre, pre_i, methods):
    """The in-memory construction of `run_animal` at 8debc35e, VERBATIM in what it fits (pinned copy)."""
    from wfield_local.position_coding_directions import (BY_SEVERITY, direction, engagement_axis, orthogonalise,
                                                         poles)
    e_axis = engagement_axis(XE[e_pre], XU[u_pre]) if (e_pre.sum() and u_pre.sum()) else None
    out = {}
    for meth in methods:
        base_m, do_orth = (meth[:-5], True) if meth.endswith("_orth") else (meth, False)
        W, W_loso, cos, PW, PWL = {}, {}, {}, {}, {}
        for D in BY_SEVERITY:
            Xp, Xn = XE[e_pre & (en == D)], XE[e_pre & (en != D)]
            if len(Xp) < 20 or len(Xn) < 20:
                continue
            w_raw = direction(Xp, Xn, base_m)
            cos[D] = float(w_raw @ e_axis) if e_axis is not None else None
            w = orthogonalise(w_raw, e_axis) if do_orth else w_raw
            p0, p1 = poles(Xp, Xn, w)
            W[D] = (w, p0, p1)
            for i in pre_i:
                tr = e_pre & (GE != i)
                Ap, An = XE[tr & (en == D)], XE[tr & (en != D)]
                if len(Ap) < 20 or len(An) < 20:
                    continue
                wi = direction(Ap, An, base_m)
                if do_orth:
                    wi = orthogonalise(wi, e_axis)
                W_loso[(D, i)] = (wi, *poles(Ap, An, wi))
        for A in BY_SEVERITY:
            for B in BY_SEVERITY:
                if A == B:
                    continue
                XA, XB = XE[e_pre & (en == A)], XE[e_pre & (en == B)]
                if len(XA) < 20 or len(XB) < 20:
                    continue
                wab = direction(XA, XB, base_m)
                if do_orth and e_axis is not None:
                    wab = orthogonalise(wab, e_axis)
                a0, a1 = poles(XA, XB, wab)
                PW[f"{A}|{B}"] = (wab, a0, a1)
                for i in pre_i:
                    tr = e_pre & (GE != i)
                    Ap, Bp = XE[tr & (en == A)], XE[tr & (en == B)]
                    if len(Ap) < 20 or len(Bp) < 20:
                        continue
                    wi = direction(Ap, Bp, base_m)
                    if do_orth:
                        wi = orthogonalise(wi, e_axis)
                    r0, r1 = poles(Ap, Bp, wi)
                    PWL[(f"{A}|{B}", i)] = (wi, r0, r1)
        out[meth] = (W, W_loso, cos, PW, PWL)
    return e_axis, out


def _same_triples(a, b):
    assert sorted(a, key=str) == sorted(b, key=str)
    for k in a:
        for x, y in zip(a[k], b[k]):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("methods", [("dom", "dom_orth"), ("lr", "lr_orth")])
def test_coding_reference_equals_the_old_in_memory_loop(pass2_env, methods):
    from wfield_local import frozen_models
    from wfield_local import position_coding_directions as pcd
    rec = _Recorder()
    pass2_env.setattr(frozen_models, "load_or_fit", rec)
    XE, en, e_pre, GE, XU, u_pre, pre_i, kept = _cd_pool(seed=1)
    ref = pcd.stored_reference("PS92", "precue", 2.0, "basis0", XE, en, e_pre, GE, XU, u_pre, pre_i, kept, methods)
    spec = rec.calls[0]["spec"]
    assert spec["kind"] == "coding_directions" and spec["align"] == f"precue-{'+'.join(methods)}"
    assert spec["train_labels"] == kept[:3] and "lick_source" not in spec
    e_old, old = _old_coding_loop(XE, en, e_pre, GE, XU, u_pre, pre_i, methods)
    np.testing.assert_array_equal(ref["e_axis"], e_old)
    for meth in methods:
        W, WL, cos, PW, PWL = old[meth]
        m = ref["methods"][meth]
        _same_triples(m["W"], W)
        _same_triples({(D, kept.index(lab)): v for (D, lab), v in m["W_loso"].items()}, WL)
        _same_triples(m["PW"], PW)
        _same_triples({(k, kept.index(lab)): v for (k, lab), v in m["PW_loso"].items()}, PWL)
        assert {D: float(m["w_raw"][D] @ ref["e_axis"]) for D in m["W"]} == cos
    pass2_env.setenv("WIDEFIELD_CODING_FROZEN", "0")
    mem = pcd.stored_reference("PS92", "precue", 2.0, "basis0", XE, en, e_pre, GE, XU, u_pre, pre_i, kept, methods)
    assert len(rec.calls) == 1
    for meth in methods:
        for part in ("W", "W_loso", "PW", "PW_loso"):
            _same_triples(mem["methods"][meth][part], ref["methods"][meth][part])


def test_coding_confusion_decoder_stored_equals_in_memory(pass2_env):
    from wfield_local import frozen_models
    from wfield_local import position_coding_directions as pcd
    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER
    rec = _Recorder()
    pass2_env.setattr(frozen_models, "load_or_fit", rec)
    rng = np.random.default_rng(3)
    GE = np.repeat(np.arange(5), 72)
    YE = np.tile(np.array(DISPLAY_ORDER), 60)
    XE = rng.normal(size=(len(GE), 8)) + 0.9 * YE[:, None] * rng.normal(size=(1, 8))
    XU, YU = XE[:50] + 0.1, YE[:50]
    e_pre, pre_i, kept = GE < 3, {0, 1, 2}, [f"PS92_08{10 + i}" for i in range(5)]

    def _mask(cls, pos=None):
        return (~e_pre) if cls == "poststroke_lick" else np.arange(len(XU)) % 2 == (cls == "poststroke_stopped")

    pay = pcd.stored_confusion_decoder("PS92", "cue", 2.0, "basis0", XE, YE, GE, e_pre, pre_i, kept)
    assert rec.calls[0]["spec"]["kind"] == "coding_decoder"
    old_pipe = pcd._confusion_pipe
    full = old_pipe().fit(XE[e_pre], YE[e_pre])
    _same_model(pay["full"], full)
    for i in sorted(pre_i):
        tr = e_pre & (GE != i)
        _same_model(pay["loso"][kept[i]], old_pipe().fit(XE[tr], YE[tr]))
    a = pcd._class_confusions(XE, YE, GE, XU, YU, e_pre, pre_i, _mask, kept, models=pay)
    b = pcd._class_confusions(XE, YE, GE, XU, YU, e_pre, pre_i, _mask, kept)
    assert a == b
    est = full[-1]
    assert est.C == 0.5 and est.max_iter == 2000                     # the module's own settings, not `_pipe`'s
    pass2_env.setenv("WIDEFIELD_CODING_FROZEN", "0")
    assert pcd.stored_confusion_decoder("PS92", "cue", 2.0, "b", XE, YE, GE, e_pre, pre_i, kept) is None


def test_coding_lp_arm_lp_trained_by_default(pass2_env):
    from wfield_local import config, frozen_models
    from wfield_local import position_coding_directions as pcd
    XE, en, e_pre, GE, XU, u_pre, pre_i, kept = _cd_pool()
    pass2_env.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    specs = {}
    for lf in (None, "daq"):
        rec = _Recorder()
        pass2_env.setattr(frozen_models, "load_or_fit", rec)
        if lf:
            pass2_env.setenv("WIDEFIELD_LP_FROZEN", lf)
        pcd.stored_reference("PS92", "cue", 2.0, "b", XE, en, e_pre, GE, XU, u_pre, pre_i, kept, ("dom",))
        specs[lf] = rec.calls[0]["spec"]
    assert "lick_source" not in specs["daq"]
    assert specs["daq"]["train_labels"] == sorted(x for x in config.phase_labels("pre") if config.animal_of(x) == "PS92")
    assert specs[None]["lick_source"] == "lp" and specs[None]["train_labels"] == kept[:3]


def test_coding_run_animal_uses_the_store():
    from wfield_local import position_coding_directions as pcd
    src = inspect.getsource(pcd.run_animal)
    assert "models=stored_confusion_decoder(animal, align, post_s, basis.basis_id" in src
    assert "ref = stored_reference(animal, align, post_s, basis.basis_id" in src
    assert "direction(" not in src                                    # no direction is fitted in run_animal itself


# ------------------------------------------------------------------------------------------- cd_trajectories (5)
def _cd_traj_inputs(seed=0, ncomp=9, T=40):
    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER
    rng = np.random.default_rng(seed)
    y = np.tile(np.array(DISPLAY_ORDER), 40)
    mu = {int(p): rng.normal(size=ncomp) for p in DISPLAY_ORDER}
    X = np.vstack([mu[int(p)] for p in y]) + rng.normal(size=(len(y), ncomp))
    G = [(rng.normal(size=(ncomp, T)), 50 + 10 * i) for i in range(3)]
    Rb = {int(p): [rng.normal(size=ncomp) for _ in range(3)] for p in DISPLAY_ORDER}
    drop = np.zeros(ncomp, bool)
    drop[[1, 4]] = True
    return X, y, G, Rb, drop, np.arange(-10, T - 10) / 10.0


def _old_cd_fit(Xall, yall, Gpre_chunks, Rb, drop, reference, method, t):
    """`analyse_animal`'s in-memory fit at 8debc35e (pinned copy of the inline block)."""
    from wfield_local import cd_trajectories as cdt
    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER
    stats = cdt.component_stats(Xall) if cdt.STANDARDISE else None
    cim = None
    if Gpre_chunks:
        Gpre = cdt.pooled_mean(Gpre_chunks, stats)
        if drop is not None:
            Gpre = np.asarray(Gpre, float).copy()
            Gpre[np.asarray(drop, bool)] = 0.0
        cim = cdt.condition_independent_modes(Gpre, t)
    if reference == "contrast":
        dirs, surviving = cdt.fit_directions(Xall, yall, DISPLAY_ORDER, method=method, stats=stats, cim=cim,
                                             with_surviving=True, drop=drop)
    else:
        base = {k_: np.mean(np.stack(v_), 0) for k_, v_ in Rb.items()}
        dirs = cdt.directions_vs_rest(Xall, yall, base, DISPLAY_ORDER, stats=stats, cim=cim, drop=drop)
        surviving = {}
    return stats, cim, dirs, surviving


def _eq_nested(a, b):
    if isinstance(a, dict):
        assert sorted(a, key=str) == sorted(b, key=str)
        for k in a:
            _eq_nested(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _eq_nested(x, y)
    elif a is None:
        assert b is None
    else:
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("reference,orth,method", [("contrast", True, "dom"), ("contrast", False, "dom"),
                                                    ("rest", True, "dom"), ("contrast", True, "lr")])
def test_cd_reference_stored_equals_the_old_inline_fit(pass2_env, reference, orth, method):
    from types import SimpleNamespace

    from wfield_local import cd_trajectories as cdt
    from wfield_local import frozen_models
    rec = _Recorder()
    pass2_env.setattr(frozen_models, "load_or_fit", rec)
    X, y, G, Rb, drop, t = _cd_traj_inputs(seed=5)
    Gc = G if orth else None
    basis = SimpleNamespace(basis_id="b" * 12, ncomp=X.shape[1])
    kw = dict(method=method, reference=reference, use=("success",), orth=orth, mask_occluded=True, win_n=20,
              pre_n=10, post_frames=30, post_s=2.0, verbose=False,
              train_data=(X, y, Gc, Rb, drop))            # FREEZE_VERSION 2 training-data fingerprint (rebuild 11-03)
    got = cdt.stored_cd_reference("PS92", "precue", basis, ["PS92_0810", "PS92_0811"], lambda: cdt.fit_cd_reference(
        X, y, Gc, Rb, drop, reference=reference, method=method, t=t), **kw)
    assert len(rec.calls) == 1 and rec.calls[0]["spec"]["kind"] == "cd_directions"
    old = _old_cd_fit(X, y, Gc, Rb, drop, reference, method, t)
    _eq_nested([got["stats"], got["cim"], got["dirs"], got["surviving"]], list(old))
    pass2_env.setenv("WIDEFIELD_CD_FROZEN", "0")
    mem = cdt.stored_cd_reference("PS92", "precue", basis, ["PS92_0810"], lambda: cdt.fit_cd_reference(
        X, y, Gc, Rb, drop, reference=reference, method=method, t=t), **kw)
    assert len(rec.calls) == 1
    _eq_nested([mem["stats"], mem["cim"], mem["dirs"], mem["surviving"]], list(old))


def test_cd_spec_distinguishes_every_variant():
    from wfield_local import cd_trajectories as cdt
    base = dict(method="dom", reference="contrast", use=("success",), orth=True, mask_occluded=True, win_n=20,
                pre_n=10, post_frames=30)
    vals = {cdt.cd_spec_align("precue", **base)}
    for k, v in (("method", "lr"), ("reference", "rest"), ("use", ("success", "miss_working")), ("orth", False),
                 ("mask_occluded", False), ("win_n", 10), ("pre_n", 5), ("post_frames", 40)):
        vals.add(cdt.cd_spec_align("precue", **{**base, k: v}))
    vals.add(cdt.cd_spec_align("cue", **base))
    assert len(vals) == 10


def test_cd_analyse_animal_uses_the_store():
    from wfield_local import cd_trajectories as cdt
    src = inspect.getsource(cdt.analyse_animal)
    assert "cref = stored_cd_reference(" in src and "fit_cd_reference(" in src
    assert 'stats, cim, dirs, surviving = cref["stats"], cref["cim"], cref["dirs"], cref["surviving"]' in src
    assert "fit_directions(" not in src and "condition_independent_modes(Gpre" not in src


# ------------------------------------------------------- the ENL window reaches every "precue-*" spec (fix, 11-02)
@pytest.mark.parametrize("align", ["precue", "precue-rt2-fabc", "precue-dom+dom_orth", "precue-keep1-2-seed0"])
def test_precue_specs_carry_the_enl_window(pass2_env, align):
    """A WIDEFIELD_ENL_WINDOW_S=1.0 run must never resolve to the 2.0 s model: found by the nolick ENL 1.0 render,
    where "precue-rt2-..." specs ignored the window and the 1.0 s run HIT the 2.0 s models (precue cells differed)."""
    from wfield_local import frozen_models
    tr = ["PS92_0810", "PS92_0811"]
    a = frozen_models.make_spec("PS92", "k", align=align, source="locanmf", train_labels=tr, train_data=None)
    pass2_env.setenv("WIDEFIELD_ENL_WINDOW_S", "1.0")
    b = frozen_models.make_spec("PS92", "k", align=align, source="locanmf", train_labels=tr, train_data=None)
    assert a["post_s"] == 2.0 and b["post_s"] == 1.0
    assert frozen_models.spec_id(a) != frozen_models.spec_id(b)


def test_non_precue_specs_ignore_the_enl_window(pass2_env):
    from wfield_local import frozen_models
    tr = ["PS92_0810"]
    for align in ("cue", "cue-june", "lick-dom", "iti_licks.frozen_decode-x", "rest-rest-b4"):
        a = frozen_models.make_spec("PS92", "k", align=align, source="s", train_labels=tr, train_data=None)
        pass2_env.setenv("WIDEFIELD_ENL_WINDOW_S", "1.0")
        b = frozen_models.make_spec("PS92", "k", align=align, source="s", train_labels=tr, train_data=None)
        pass2_env.delenv("WIDEFIELD_ENL_WINDOW_S")
        assert frozen_models.spec_id(a) == frozen_models.spec_id(b), align


# --------------------------------------------------------------------------------- section G (poststroke_compare) (6)
def _g_pool(seed=11):
    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER
    rng = np.random.default_rng(seed)
    K = len(DISPLAY_ORDER)
    kept = ["PS99_0601", "PS99_0602", "PS99_0603", "PS99_0604", "PS99_0817", "PS99_0818"]
    per = 72
    YE = np.array([DISPLAY_ORDER[i % K] for i in range(per * len(kept))])
    GE = np.repeat(np.arange(len(kept)), per)
    XE = rng.normal(size=(YE.size, 10)) + YE[:, None] * 0.4
    nu = 60 * len(kept)
    YU = np.array([DISPLAY_ORDER[i % K] for i in range(nu)])
    GU = np.repeat(np.arange(len(kept)), nu // len(kept))
    XU = rng.normal(size=(nu, 10)) + YU[:, None] * 0.2
    return {"XE": XE, "YE": YE, "GE": GE, "BE": [np.zeros(per, int)] * len(kept),
            "XU": XU, "YU": YU.astype(int), "GU": GU, "kept": kept, "pre_i": {0, 1, 2, 3}, "post_i": {4},
            "align": "cue", "source": "roi", "common": [("SSp", 0)], "_frozen_cache": {}}


def _g_run(monkeypatch, fn, off, **kw):
    from wfield_local import frozen_models
    from wfield_local import poststroke_compare as pc
    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER
    rec = _Recorder()
    monkeypatch.setattr(frozen_models, "load_or_fit", rec)
    if off:
        monkeypatch.setenv("WIDEFIELD_SECTION_G_FROZEN", "0")
    else:
        monkeypatch.delenv("WIDEFIELD_SECTION_G_FROZEN", raising=False)
    keep = list(DISPLAY_ORDER[:4])
    return getattr(pc, fn)(_g_pool(), keep, **kw), rec


@pytest.mark.parametrize("fn,kw,kind", [("decode_matched", {"post_all_trials": False}, "g_lickonly_decoder"),
                                         ("looks_like_which", {}, "g_lick_disc"),
                                         ("fits_engaged_distribution", {"n_boot": 300}, "g_lick_disc_folds")])
def test_section_g_stored_equals_in_memory(pass2_env, fn, kw, kind):
    """Each section G reference (lick-only arm, lick discriminator, the advancing-RNG fold discriminators + the
    bootstrap that continues from their generator) gives the SAME record stored and opted out, exactly."""
    got, rec = _g_run(pass2_env, fn, False, **kw)
    assert [c["spec"]["kind"] for c in rec.calls] == [kind]
    spec = rec.calls[0]["spec"]
    assert spec["train_labels"] == ["PS99_0601", "PS99_0602", "PS99_0603", "PS99_0604"]
    assert "keep" in spec["align"] and spec["align"].startswith("cue-") and "lick_source" not in spec
    ref, rec2 = _g_run(pass2_env, fn, True, **kw)
    assert rec2.calls == []
    assert got == ref


def test_section_g_specs_key_on_preserved_set_seed_and_columns(pass2_env):
    from wfield_local import frozen_models
    from wfield_local import poststroke_compare as pc
    rec = _Recorder()
    pass2_env.setattr(frozen_models, "load_or_fit", rec)
    for keep, common in (([1, 2, 3, 4], [("SSp", 0)]), ([1, 2, 3], [("SSp", 0)]), ([1, 2, 3, 4], [("MOp", 0)])):
        d = _g_pool()
        d["common"] = common
        pc._g_store(d, "g_lick_disc", f"{pc._keep_tag(keep)}-seed0", lambda: "m")
        pc._g_store(d, "g_lick_disc", f"{pc._keep_tag(keep)}-seed1", lambda: "m")
        pc._g_store(d, "g_lick_disc", f"{pc._keep_tag(keep)}-seed1", lambda: "m")     # memoised on the pool
    assert len(rec.calls) == 6
    assert len({frozen_models.spec_id(c["spec"]) for c in rec.calls}) == 6
    # an unkeyable pool (no identity recorded) fits in memory, as `frozen` does
    assert pc._g_store({"kept": ["PS99_0601"], "pre_i": {0}, "_frozen_cache": {}}, "g_lick_disc", "x", lambda: 1) is None


def test_section_g_sites_use_the_store():
    from wfield_local import poststroke_compare as pc
    assert '_g_store(\n            d, "g_lickonly_decoder", _keep_tag(pos)' in inspect.getsource(pc.decode_matched)
    for fn in (pc.looks_like_which, pc.undetected_state_split):
        assert '_g_store(d, "g_lick_disc", f"{_keep_tag(keep)}-seed{seed}"' in inspect.getsource(fn)
    src = inspect.getsource(pc.fits_engaged_distribution)
    assert '_g_store(d, "g_lick_disc_folds"' in src and 'rng.set_state(folds["rng_after"])' in src


# ------------------------------------------------------------------------------- ITI / cued-lick read-outs (7)
def _iti_data(seed=0, k=None):
    rng = np.random.default_rng(seed)
    labs = ["PS92_0810", "PS92_0811", "PS92_0812", "PS92_0813", "PS92_0901"]
    X = {q: rng.normal(size=(40 + 5 * i, 7)) for i, q in enumerate(labs)}
    W = rng.normal(size=(7,) if k is None else (7, k))
    Y = {q: X[q] @ W + rng.normal(size=(len(X[q]),) if k is None else (len(X[q]), k)) for q in labs}
    return labs[:4], X, Y


def _old_iti_fits(pre, X, Y, alphas, k=None):
    """The inline construction every ITI script carried at 8debc35e (pinned copy)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    fits = {}
    for held in [*pre, None]:
        trn = [q for q in pre if q != held]
        Xt = np.vstack([X[q] for q in trn])
        Yt = np.concatenate([Y[q] for q in trn]) if k is None else np.vstack([Y[q] for q in trn])
        sc = StandardScaler().fit(Xt)
        fits[held] = (sc, RidgeCV(alphas=alphas).fit(sc.transform(Xt), Yt) if k is None else
                      [RidgeCV(alphas=alphas).fit(sc.transform(Xt), Yt[:, j]) for j in range(k)])
    return fits


def _same_ridge(a, b):
    np.testing.assert_array_equal(a[0].mean_, b[0].mean_)
    np.testing.assert_array_equal(a[0].scale_, b[0].scale_)
    ma, mb = (a[1], b[1]) if isinstance(a[1], list) else ([a[1]], [b[1]])
    for x, y in zip(ma, mb):
        np.testing.assert_array_equal(x.coef_, y.coef_)
        assert x.intercept_ == y.intercept_ and x.alpha_ == y.alpha_


@pytest.mark.parametrize("k", [None, 3])
def test_iti_helper_equals_the_old_inline_fits(pass2_env, k):
    from wfield_local import frozen_models, frozen_readouts
    rec = _Recorder()
    pass2_env.setattr(frozen_models, "load_or_fit", rec)
    pre, X, Y = _iti_data(seed=2, k=k)
    al = np.logspace(-1, 5, 13)
    got = frozen_readouts.pre_loso_ridge("PS92", "iti_licks.frozen_decode-direction", pre, X, Y, al, k=k)
    spec = rec.calls[0]["spec"]
    assert spec["kind"] == "iti_readout" and spec["lick_source"] == "lp"          # LP-only by nature
    assert spec["train_order"] == pre and spec["n_targets"] == k and spec["loso"] is True
    old = _old_iti_fits(pre, X, Y, al, k)
    assert list(got) == list(old) == [*pre, None]
    for h in old:
        _same_ridge(got[h], old[h])
    pass2_env.setenv("WIDEFIELD_ITI_FROZEN", "0")
    mem = frozen_readouts.pre_loso_ridge("PS92", "x", pre, X, Y, al, k=k)
    assert len(rec.calls) == 1
    for h in old:
        _same_ridge(mem[h], old[h])


def test_iti_spec_carries_tag_alphas_order_and_data(pass2_env):
    from wfield_local import frozen_models, frozen_readouts
    rec = _Recorder()
    pass2_env.setattr(frozen_models, "load_or_fit", rec)
    pre, X, Y = _iti_data()
    al = np.logspace(-1, 4, 11)
    frozen_readouts.pre_loso_ridge("PS92", "a-w0..0.2:x", pre, X, Y, al)
    frozen_readouts.pre_loso_ridge("PS92", "b", pre, X, Y, al)
    frozen_readouts.pre_loso_ridge("PS92", "b", pre, X, Y, np.logspace(0, 5, 11))
    frozen_readouts.pre_loso_ridge("PS92", "b", pre[::-1], X, Y, al)
    frozen_readouts.pre_loso_ridge("PS92", "b", pre, X, {q: v + 1e-9 for q, v in Y.items()}, al)
    frozen_readouts.pre_loso_ridge("PS92", "b", pre, X, Y, al, loso=False)
    ids = [frozen_models.spec_id(c["spec"]) for c in rec.calls]
    assert len(set(ids)) == len(ids) == 6
    assert ":" not in rec.calls[0]["spec"]["align"] and rec.calls[0]["spec"]["tag"] == "a-w0..0.2:x"
    assert list(rec.calls[-1]["payload"]) == [None]


ITI_SCRIPTS = ("iti_licks", "iti_basis_decoder", "iti_contact_check", "iti_first_lick", "iti_lick_order",
               "iti_nmatch_check", "iti_where_decoder", "iti_where_3axis", "iti_where_angle_reach", "iti_where_scatter",
               "iti_decoder_contrib", "lick_gain_prediction")


@pytest.mark.parametrize("name", ITI_SCRIPTS)
def test_iti_scripts_use_the_one_helper(name):
    src = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
    assert "frozen_readouts.pre_loso_ridge(" in src
    assert "for held in [*pre, None]" not in src
    assert "RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xp)" not in src
