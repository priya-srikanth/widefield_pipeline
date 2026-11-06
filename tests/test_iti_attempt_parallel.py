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
    for m in set(sys.modules) - mods:
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
    model directory. Checked on the sanitised align (the directory name part) the store actually keys on."""
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
