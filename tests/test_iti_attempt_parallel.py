"""Serial == parallel, EXACTLY, for the two loops fanned out on 2026-11-06 (CLAUDE.md rules 6 / 9).

1. `scripts/iti_licks.frozen_decode_lags` (the `--key` path; 6.2 h serial): unit = one (animal, lag), both targets.
2. `wfield_local.attempt_coding._cue_classes` (the per-session DAQ cue + LP class load inside `run_animal`).

Synthetic inputs only: the expensive loaders are monkeypatched, the frozen store is switched off
(WIDEFIELD_ITI_FROZEN=0, inherited by spawned workers), nothing reads M:. Equality is asserted with
`pd.testing.assert_frame_equal(check_exact=True)` / `np.array_equal`, never a tolerance (rule 9).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_env_and_modules():
    """A script imported here runs module-level `os.environ.setdefault`s and stays in sys.modules; both leaked into
    later tests once (WIDEFIELD_PRE_FROM). Snapshot both and restore them, whatever the test did."""
    env = dict(os.environ)
    mods = set(sys.modules)
    path = list(sys.path)
    yield
    os.environ.clear()
    os.environ.update(env)
    # SCRIPT modules only (iti_licks, fr_musall, ...): a dropped `wfield_local.*` submodule would leave the package
    # attribute pointing at the old object while pickle / importlib find a new one ("not the same object")
    for m in set(sys.modules) - mods:
        if m != "wfield_local" and not m.startswith("wfield_local."):
            del sys.modules[m]
    sys.path[:] = path


def _iti(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    for m in ("iti_licks", "fr_musall"):
        sys.modules.pop(m, None)
    import iti_licks as il
    return il


def _synthetic_D(il, seed=0, n_lags=None):
    """{label: session dict} for three animals x (4 pre + 2 post) sessions, plus one too-small session and one animal
    with too few pre sessions (both must be skipped exactly as before)."""
    rng = np.random.default_rng(seed)
    nl = len(il.LAGS_D) if n_lags is None else n_lags
    cols = np.array(["a_left", "a_right", "b_left", "b_right"])
    D = {}
    for an in ("PSA", "PSB", "PSC"):
        npre = 4 if an != "PSC" else 2
        for k, ep in enumerate(["pre"] * npre + ["acute", "chronic"]):
            lab = f"{an}_{k:04d}"
            d = {"animal": np.array(an), "epoch": np.array(ep), "cols": cols}
            for pre_, n in (("iti", 40 + 3 * k), ("tr", 30 + 2 * k)):
                d[f"{pre_}_X"] = rng.normal(size=(n, nl, len(cols))).astype(np.float32)
                d[f"{pre_}_side"] = rng.normal(size=n)
                d[f"{pre_}_pos"] = rng.integers(0, 3, n)
            D[lab] = d
    D["PSA_9999"] = {"animal": np.array("PSA"), "epoch": np.array("acute"), "cols": cols,
                     "iti_X": np.zeros((5, nl, 4), np.float32), "iti_side": np.zeros(5), "iti_pos": np.zeros(5, int),
                     "tr_X": np.zeros((5, nl, 4), np.float32), "tr_side": np.zeros(5), "tr_pos": np.zeros(5, int)}
    return D


def _setup_iti(monkeypatch, il, n_lags=5):
    monkeypatch.setenv("WIDEFIELD_ITI_FROZEN", "0")              # in-memory fits: no store, no M:
    monkeypatch.setattr(il, "AN", ["PSA", "PSB", "PSC"])
    monkeypatch.setattr(il, "LAGS_D", np.round(np.linspace(-0.1, 0.1, n_lags), 3))
    # the split-basis rest areas read a table on M:; the features themselves are not what is under test
    monkeypatch.setattr(il, "area_features", lambda X, cols, label, *a, **k: X[..., [0, 1, 3]] * 2.0 - X[..., [2]])


class _ShuffledFanOut:
    """`parallel.fan_out` stand-in: runs in-process but returns results in a SCRAMBLED completion order."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def __call__(self, items, worker, *, jobs=None, label="unit", log=print):
        self.calls += 1
        from wfield_local import parallel
        parallel.pin_blas()                                     # as the real fan_out does, in the CALLING process
        res = [(it, worker(it)) for it in items]
        return [res[i] for i in self.rng.permutation(len(res))], []


def test_frozen_decode_lags_parallel_order_is_serial_order(monkeypatch):
    il = _iti(monkeypatch)
    _setup_iti(monkeypatch, il)
    from wfield_local import parallel
    D = _synthetic_D(il, n_lags=5)
    for set_name in ("iti", "trial"):
        serial = il.frozen_decode_lags(D, set_name, jobs=1)
        fake = _ShuffledFanOut(seed=3)
        monkeypatch.setattr(parallel, "fan_out", fake)
        par = il.frozen_decode_lags(D, set_name, jobs=4)
        assert fake.calls == 2                                  # one pool per animal with >= 3 pre sessions
        assert len(serial) == 2 * 5 * 2 * 6                     # animals x lags x targets x sessions (PSC, 9999 out)
        pd.testing.assert_frame_equal(serial, par, check_exact=True)


def test_frozen_decode_lags_serial_matches_the_pre_change_loop(monkeypatch):
    """jobs=1 reproduces a verbatim copy of the loop as it was before 11-06."""
    il = _iti(monkeypatch)
    _setup_iti(monkeypatch, il)
    D = _synthetic_D(il, n_lags=5)

    def old(D, set_name):                                       # frozen copy of the pre-11-06 source
        from wfield_local import frozen_readouts
        kx, ky, kp = ("iti_X", "iti_side", "iti_pos") if set_name == "iti" else ("tr_X", "tr_side", "tr_pos")
        rows = []
        for a in il.AN:
            S = {x: d for x, d in D.items() if str(d["animal"]) == a and len(d[ky]) >= 20}
            pre = [x for x, d in S.items() if str(d["epoch"]) == "pre"]
            if len(pre) < 3:
                continue
            Y = {tg: {x: (S[x][ky] if tg == "direction" else il._within(S[x][ky], S[x][kp])) for x in S}
                 for tg in ("direction", "relative")}
            for li, lag in enumerate(il.LAGS_D):
                X = {x: il.area_features(S[x][kx][:, li, :].astype(float), S[x]["cols"], x) for x in S}
                for tg in ("direction", "relative"):
                    fits = frozen_readouts.pre_loso_ridge(a, f"iti_licks.frozen_decode_lags-{set_name}-{tg}-lag{lag:g}",
                                                          pre, X, Y[tg], np.logspace(-1, 4, 11))
                    for x in S:
                        sc, m = fits[x if x in pre else None]
                        pr = m.predict(sc.transform(X[x]))
                        rows.append({"label": x, "animal": a, "epoch": str(S[x]["epoch"]), "set": set_name,
                                     "target": tg, "lag": float(lag),
                                     "r": float(np.corrcoef(pr, Y[tg][x])[0, 1]) if np.std(pr) > 0 else np.nan})
        return pd.DataFrame(rows)

    for set_name in ("iti", "trial"):
        pd.testing.assert_frame_equal(old(D, set_name), il.frozen_decode_lags(D, set_name, jobs=1), check_exact=True)


def test_frozen_decode_lags_specs_are_distinct_per_unit(monkeypatch):
    """Store safety: every (set, target, lag) call names its own frozen spec, so concurrent units never write the same
    model directory. Checked on the sanitized align (the directory name part) the store actually keys on."""
    il = _iti(monkeypatch)
    _setup_iti(monkeypatch, il, n_lags=41)
    monkeypatch.setattr(il, "LAGS_D", np.round(np.arange(-1.0, 1.0001, 0.05), 3))   # the production lag grid
    from wfield_local import frozen_readouts
    tags = []
    real = frozen_readouts.fit_all

    def spy(animal, tag, pre, X, Y, alphas, **k):
        tags.append((animal, "".join(c if (c.isalnum() or c in "._+-") else "-" for c in str(tag))))
        return real(list(pre), X, Y, np.asarray(alphas, float))

    monkeypatch.setattr(frozen_readouts, "pre_loso_ridge", spy)
    D = _synthetic_D(il, n_lags=41)
    for set_name in ("iti", "trial"):
        il.frozen_decode_lags(D, set_name, jobs=1)
    assert len(tags) == 2 * 2 * 41 * 2
    assert len(set(tags)) == len(tags)


def test_frozen_decode_lags_real_spawn_two_workers(monkeypatch):
    """The real process pool (2 workers, spawn): the module-level worker and `_LagUnit` pickle, and the frame is
    identical to the serial one."""
    il = _iti(monkeypatch)
    _setup_iti(monkeypatch, il, n_lags=3)
    D = _synthetic_D(il, n_lags=3)
    serial = il.frozen_decode_lags(D, "iti", jobs=1)
    par = il.frozen_decode_lags(D, "iti", jobs=2)
    pd.testing.assert_frame_equal(serial, par, check_exact=True)


# ---------------------------------------------------------------------------------------------- attempt_coding
def _fake_cue_one(calls):
    def one(lab, max_rt):
        calls.append(lab)
        if lab.endswith("_unreg"):
            return None
        if lab.endswith("_bad"):
            raise RuntimeError(f"{lab}: no LP tables")
        r = np.random.default_rng(int(lab.split("_")[1]) + int(max_rt * 10))
        n = 20 + int(lab.split("_")[1]) % 7
        return r.choice(np.array(["hit", "incomplete", "no_try", ""]), n).astype(str), r.integers(0, 500, n)
    return one


def _same_cc(a, b):
    assert list(a) == list(b)                                   # same labels, same ORDER
    for k in a:
        assert a[k][0].dtype == b[k][0].dtype and a[k][1].dtype == b[k][1].dtype
        assert np.array_equal(a[k][0], b[k][0]) and np.array_equal(a[k][1], b[k][1])


def test_attempt_prefetched_cue_classes_equal_serial(monkeypatch):
    """`_prefetch_sessions` (scrambled completion order) + `_cue_classes(prefetched=)` == the serial `_cue_classes`:
    same dict, same key order, same arrays; unregistered sessions dropped as before."""
    from wfield_local import attempt_coding as ac
    from wfield_local import parallel
    calls = []
    monkeypatch.setattr(ac, "_cue_class_one", _fake_cue_one(calls))
    monkeypatch.setattr(ac, "_WARM_FEAT", {("PSX", "cue"): lambda s, args: None})   # no basis, no projection
    from wfield_local import locanmf_cue_lick_analysis as lcla
    labels = ["PSX_0810", "PSX_0811", "PSX_0812_unreg", "PSX_0901", "PSX_0902", "PSX_0903"]
    monkeypatch.setattr(lcla, "SESSIONS", [{"label": x} for x in labels])
    serial = ac._cue_classes(labels, 3.5)
    monkeypatch.setattr(parallel, "fan_out", _ShuffledFanOut(seed=7))
    pre = ac._prefetch_sessions("PSX", labels, "precue", 1.0, "cue", 3.5, jobs=4)
    calls.clear()
    par = ac._cue_classes(labels, 3.5, prefetched=pre)
    assert calls == []                                          # every session came from the workers
    _same_cc(serial, par)
    # a subset of labels (run_animal passes `kept`, a subset of pre + post) keeps ITS order
    sub = ["PSX_0901", "PSX_0810"]
    _same_cc(ac._cue_classes(sub, 3.5), ac._cue_classes(sub, 3.5, prefetched=pre))


def test_attempt_worker_failure_falls_back_to_serial_and_raises_the_same(monkeypatch):
    """A session whose load failed in a worker is recomputed in the parent, so the serial error surfaces unchanged."""
    from wfield_local import attempt_coding as ac
    from wfield_local import locanmf_cue_lick_analysis as lcla
    from wfield_local import parallel
    calls = []
    monkeypatch.setattr(ac, "_cue_class_one", _fake_cue_one(calls))
    monkeypatch.setattr(ac, "_WARM_FEAT", {("PSX", "cue"): lambda s, args: (_ for _ in ()).throw(OSError("cold"))})
    labels = ["PSX_0810", "PSX_0811_bad"]
    monkeypatch.setattr(lcla, "SESSIONS", [{"label": x} for x in labels])
    monkeypatch.setattr(parallel, "fan_out", _ShuffledFanOut(seed=1))
    pre = ac._prefetch_sessions("PSX", labels, "precue", 1.0, "cue", 3.5, jobs=2)
    assert set(pre) == {"PSX_0810"}                             # warm failure is not a cue failure
    with pytest.raises(RuntimeError, match="no LP tables"):
        ac._cue_classes(labels, 3.5, prefetched=pre)
    with pytest.raises(RuntimeError, match="no LP tables"):
        ac._cue_classes(labels, 3.5)


def test_attempt_prefetch_worker_is_spawn_picklable():
    import pickle

    from wfield_local import attempt_coding as ac
    item = ("PS92", "PS92_0810", "precue", 1.0, "cue", 3.5, 24)
    assert pickle.loads(pickle.dumps(ac._prefetch_one)) is ac._prefetch_one
    assert pickle.loads(pickle.dumps(item)) == item


def test_attempt_run_animal_serial_path_unchanged_by_default():
    """jobs defaults to 1 (no prefetch, no pool) for library callers; the CLI passes --jobs."""
    import inspect

    from wfield_local import attempt_coding as ac
    assert inspect.signature(ac.run_animal).parameters["jobs"].default == 1
    src = inspect.getsource(ac.run_animal)
    assert "if jobs is not None and jobs > 1 else None" in src
    assert "cc = _cue_classes(kept, _args(\"locanmf\", align, post_s).max_rt, prefetched=pre_cc)" in src


# ---------------------------------------------------------------------------------------------- BLAS state
def test_parent_blas_kept_restores_threads_and_env(monkeypatch):
    from wfield_local import parallel
    for v in parallel._BLAS_VARS:
        monkeypatch.delenv(v, raising=False)
    before = parallel.blas_threads()
    with parallel.parent_blas_kept() as n:
        assert n == before
        parallel.pin_blas(1)
        assert parallel.blas_threads() in (1, None)
    assert parallel.blas_threads() == before
    assert all(v not in os.environ for v in parallel._BLAS_VARS)


def test_frozen_decode_lags_leaves_the_parent_unpinned(monkeypatch):
    """The rest of key_figures must run at the parent's own BLAS count, as it did before the fan-out."""
    il = _iti(monkeypatch)
    _setup_iti(monkeypatch, il, n_lags=2)
    from wfield_local import parallel
    monkeypatch.setattr(parallel, "fan_out", _ShuffledFanOut(seed=0))
    before = parallel.blas_threads()
    il.frozen_decode_lags(_synthetic_D(il, n_lags=2), "iti", jobs=2)
    assert parallel.blas_threads() == before
