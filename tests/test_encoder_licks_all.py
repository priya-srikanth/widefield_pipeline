"""LICKS_ALL=split as the production encoder (Priya 11-04, option B) and the lick-free "enl" scoring window: every lick
in exactly one kernel, one bundle serving split and unset runs, the decoders' lick-free ENL search. Synthetic, no share."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wfield_local import combined_encoding as ce
from wfield_local import enl_window


def _B(n=60, seed=0):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_encoder_trials as te
    x = te._inputs(n_trials=n, seed=seed)
    B = x["behaviour"]
    T = B.trials
    rng = np.random.default_rng(seed + 1)
    cue, st, sp = T.cue_s.to_numpy(), T.strobe_s.to_numpy(), T.stop_s.to_numpy()
    pre = np.sort(np.concatenate([s + rng.uniform(0.1, 2.9, 2) for s in st[::2]]))       # pre-cue licks, half the trials
    iti = np.sort(sp[:-1] + 1.0)                                                             # one ITI lick per gap
    post_extra = cue[5::9] + 2.5                                                             # pre-cue-table rows, post-cue
    # cam1-only onsets: two post-cue, one in the next ENL duplicating a cam4 pre-cue lick, one new pre-cue, one ITI
    ex = np.sort(np.r_[cue[1] + 0.6, cue[2] + 0.9, pre[0] + 0.01, st[7] + 1.55, sp[9] + 1.6])
    B.extra_lick_on_s = ex
    B.context_lick_on_s = {"pre_cue": pre, "iti": iti, "post_extra": post_extra}
    whole = np.sort(np.r_[B.contact_s, st[::3] + 2.0])                                       # + pre-cue contacts
    B.context_contact_s = whole
    return x, B


def test_split_every_lick_in_exactly_one_kernel():
    _x, B = _B()
    post, pre, iti = ce.split_licks(B)
    ctx = B.context_lick_on_s
    allin = np.sort(np.concatenate([B.lick_on_s, ctx["pre_cue"], ctx["iti"], ctx["post_extra"], B.extra_lick_on_s]))
    got = np.sort(np.concatenate([post, pre, iti]))
    assert len(got) == len(allin) - 1                              # exactly the one cam1 duplicate dropped
    assert len(np.intersect1d(post, pre)) == len(np.intersect1d(post, iti)) == len(np.intersect1d(pre, iti)) == 0
    assert set(np.round(ctx["post_extra"], 6)) <= set(np.round(post, 6))     # post-cue rows of the other tables
    assert set(np.round(B.lick_on_s, 6)) <= set(np.round(post, 6))           # licks.csv: post-cue, incl. 0 .. 70 ms
    ctxs = ce.lick_context_s(np.r_[pre, iti, post], B.trials)
    assert (ctxs[:len(pre)] == "pre_cue").all() and (ctxs[len(pre):len(pre) + len(iti)] == "iti").all()
    assert (ctxs[len(pre) + len(iti):] == "post_cue").all()


def test_split_kernels_and_unset_from_the_same_behaviour(monkeypatch):
    """Split adds a pre-cue and an ITI lick kernel and uses the whole-window contacts; the SAME Behaviour without
    LICKS_ALL gives exactly the unset design (the pickle keeps the cue-table contacts)."""
    x, B = _B()
    ft = x["imaging"][1]
    monkeypatch.delenv(ce.LICKS_ALL_ENV, raising=False)
    plain = ce.raw_blocks(B, ft)
    B0 = ce.Behaviour(*[getattr(B, f) for f in ce.Behaviour.__dataclass_fields__])          # no context attributes
    ref = ce.raw_blocks(B0, ft)
    for v in ref:
        np.testing.assert_array_equal(plain[v], ref[v], err_msg=v)
    monkeypatch.setenv(ce.LICKS_ALL_ENV, "split")
    sp = ce.raw_blocks(B, ft)
    k = ref["licks"].shape[1] // 2                                       # licks = [post-cue lick, contact]
    assert sp["licks"].shape[1] == 2 * k
    post, pre, iti = ce.split_licks(B)
    np.testing.assert_array_equal(sp["licks"][:, :k], ce._event_cols(post, ft, ce.LAGS["lick"]))
    np.testing.assert_array_equal(sp["licks"][:, k:], ce._event_cols(B.context_contact_s, ft, ce.LAGS["lick"]))
    np.testing.assert_array_equal(sp["licks_pre"], ce._event_cols(pre, ft, ce.LAGS["lick"]))   # their own variables
    np.testing.assert_array_equal(sp["licks_iti"], ce._event_cols(iti, ft, ce.LAGS["lick"]))
    assert set(sp) - set(ref) == {"licks_pre", "licks_iti"}
    for v in ref:
        if v != "licks":
            np.testing.assert_array_equal(sp[v], ref[v], err_msg=v)
    monkeypatch.setenv(ce.LICKS_ALL_ENV, "score")                         # scoring only: the unset design
    sc = ce.raw_blocks(B, ft)
    for v in ref:
        np.testing.assert_array_equal(sc[v], ref[v], err_msg=v)


def test_enl_window_is_the_decoders_search(monkeypatch):
    x, B = _B()
    ft = x["imaging"][1]
    E = B.trials[B.trials.engaged].reset_index(drop=True)
    ridx = np.arange(len(ft))
    for src, length in (("lp", 1.0), ("daq", 2.0)):
        monkeypatch.setattr(ce, "enl_spec", lambda s=src, w=length: {"source": s, "length_s": w, "lickfree": True})
        rows, info = ce.enl_lickfree(B, E, ft, ridx)
        licks = ce.enl_licks(B, src)
        assert info["kind"] == "lickfree" and info["source"] == src and info["length_s"] == length
        assert info["n_fixed"] + info["n_slid"] + info["n_no_window"] == len(E)
        lf = np.searchsorted(ft, licks)
        assert not np.isin(np.flatnonzero(rows), lf).any()                    # no lick frame inside the window
        fs = 1 / np.median(np.diff(ft))
        n = int(round(length * fs))
        for c, s_ in zip(E.cue_s, E.strobe_s):                               # each trial's window = the decoders' start
            c0 = int(np.searchsorted(ft, c))
            w0 = enl_window.precue_window_start(c0, c0 - (c - s_) * fs, lf, n)
            if w0 is not None:
                assert rows[w0:w0 + n].all()
    # inputs without pre-cue licks: the fixed window, said so
    B0 = ce.Behaviour(*[getattr(B, f) for f in ce.Behaviour.__dataclass_fields__])
    rows, info = ce.enl_lickfree(B0, E, ft, ridx)
    assert rows is None and info["kind"] == "fixed2s" and "unset" in info["reason"]


def test_decoder_reexports_the_moved_search():
    src = (Path(__file__).resolve().parents[1] / "wfield_local" / "locanmf_position_decoder.py").read_text()
    assert "from wfield_local.enl_window import lickfree_window, precue_window_start" in src
    assert "def precue_window_start" not in src and "def lickfree_window" not in src
    assert enl_window.precue_window_start(100, 10, np.array([95]), 20) == 75      # slid to the latest clean gap
    assert enl_window.precue_window_start(100, 10, np.array([50]), 20) == 80      # clean fixed window kept
    assert enl_window.precue_window_start(30, 10, np.array([15, 25]), 20) is None


@pytest.fixture(scope="module")
def runs():
    import os
    keys = (ce.TRIALS_ENV, ce.SMOKE_ENV, ce.ONLY_ENV, ce.LICKS_ALL_ENV)
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update({ce.TRIALS_ENV: "off", ce.ONLY_ENV: "drop:"})
    os.environ.pop(ce.SMOKE_ENV, None)
    out = {}
    try:
        for mode in ("", "split"):
            if mode:
                os.environ[ce.LICKS_ALL_ENV] = mode
            else:
                os.environ.pop(ce.LICKS_ALL_ENV, None)
            x, _B_ = _B(n=48)
            x["behaviour"] = _B_
            out[mode or "unset"] = ce.run_session("PS93_0814", inputs=x)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def test_run_session_enl_rows(runs):
    u, s = runs["unset"], runs["split"]
    for o in (u, s):
        info = o["res"]["enl_window"]
        assert info["kind"] == "lickfree" and info["n_no_window"] >= 0
        assert {"enl", "enl_fixed2s"} <= set(o["region"].window)
    a = u["region"].set_index(["model", "window", "area"]).r2
    b = s["region"].set_index(["model", "window", "area"]).r2
    assert not np.allclose(a.loc[(slice(None), "enl", slice(None))], a.loc[(slice(None), "enl_fixed2s", slice(None))])
    assert not np.allclose(a.loc[("full", "all")], b.loc[("full", "all")])          # split is another model


def test_licks_all_modes_and_folder(monkeypatch):
    monkeypatch.setenv(ce.LICKS_ALL_ENV, "bogus")
    x, B = _B()
    with pytest.raises(ValueError, match="LICKS_ALL"):
        ce.raw_blocks(B, x["imaging"][1])
    from wfield_local.paths import PathResolver
    monkeypatch.setenv(ce.LICKS_ALL_ENV, "split")
    assert ce.out_dir(PathResolver(), "movement_first").name.endswith("_licksall-split")
    monkeypatch.setenv(ce.LICKS_ALL_ENV, "score")
    assert "_licksall" not in ce.out_dir(PathResolver(), "movement_first").name


def _o2():
    spec = importlib.util.spec_from_file_location("o2e", Path(__file__).resolve().parents[1] / "scripts" / "o2_encoding.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bundle_default_split_and_submit_override():
    m = _o2()
    env = {}
    m.set_licks_all("split", env)
    assert env == {ce.LICKS_ALL_ENV: "split"}
    env = {"X": "1"}
    m.set_licks_all("none", env)
    assert env == {"X": "1"}
    with pytest.raises(SystemExit, match="LICKS_ALL"):
        m.set_licks_all("split", {ce.LICKS_ALL_ENV: "shared"})
    r = m.RUNNER
    assert 'os.environ.pop("COMBINED_ENCODING_LICKS_ALL", None)' in r and r.index('"LICKS_ALL" in os.environ') < r.index(
        "run_session")
    runs = {t: extra for t, _o, _only, extra in m.FULL_RUNS}
    assert "LICKS_ALL=none" in runs["main_nolicksall"]


def test_split_variables_groups_and_models(monkeypatch, runs):
    """licks_pre / licks_iti: credit UNINSTRUCTED, penalty "lick"; their drop models (+ both) are fitted."""
    monkeypatch.setenv(ce.LICKS_ALL_ENV, "split")
    g = ce.group_of()
    assert g["licks_pre"] == g["licks_iti"] == "uninstructed" and g["licks"] == "instructed"
    assert ce.PEN_GROUP_OF["licks_pre"] == ce.PEN_GROUP_OF["licks_iti"] == ce.PEN_GROUP_OF["licks"] == "lick"
    monkeypatch.delenv(ce.LICKS_ALL_ENV)
    assert ce.group_of() is ce.GROUP_OF
    m = set(runs["split"]["region"].model)
    assert {"drop:licks_pre", "drop:licks_iti", "drop:licks_pre+licks_iti"} <= m
    assert not {"drop:licks_pre", "drop:licks_pre+licks_iti"} & set(runs["unset"]["region"].model)


def test_enl_source_override(monkeypatch):
    monkeypatch.delenv("WIDEFIELD_LICK_SOURCE", raising=False)
    monkeypatch.delenv("WIDEFIELD_ENL_WINDOW_S", raising=False)
    monkeypatch.delenv(ce.ENL_SOURCE_ENV, raising=False)
    assert ce.enl_spec()["source"] == "daq" and ce.enl_spec()["length_s"] == 2.0          # the decoders' default
    monkeypatch.setenv(ce.ENL_SOURCE_ENV, "lp")
    assert ce.enl_spec()["source"] == "lp" and ce.enl_spec()["length_s"] == 1.0
    monkeypatch.setenv(ce.ENL_SOURCE_ENV, "video")
    with pytest.raises(ValueError):
        ce.enl_spec()
