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
    assert 'stored = _stored_epoch_decoder(an, align, bd) if mode == "frozen" else None' in src
    assert 'clf = stored["full"] if stored else _pipe().fit(XE[e_pre], YE[e_pre])' in src
    assert 'stored["loso"].get(kept[i]) if stored else None' in src


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
