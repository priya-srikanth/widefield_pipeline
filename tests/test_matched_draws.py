"""MANY MATCHED DRAWS PER SCORED SESSION (Priya 2026-11-25) and the LEGACY PIN.

The training-set MATCHED frozen-vs-refit family (5rm / 5rmgap / 5rmgapdelta / 5rmo / 5rmodelta, `recovery_trajectory`
matched, `scripts/gap_time_variance.py`) scored each session with ONE size-matched random subset of pre-stroke blocks,
seeded by the session's TRIAL COUNT: a one-trial change in the trial set gave a completely different matched model
(per-session far-contra values moved by up to 0.4 between lineages). `WIDEFIELD_MATCHED_DRAWS=<N>` averages N draws
seeded by the session LABEL and the draw index (`grant_confusion.matched_draw_seed`).

Pinned here:
  * THE LEGACY RULE IS UNCHANGED (CLAUDE.md rule 9: exact equality, no tolerance). Unset / "legacy" gives the same
    spec (align string AND spec id), the same payload keys (pool | n_target | seed) and the same records as the code
    before this change. The expected values were produced by the unmodified queue head eed09e0f on this fixture.
  * the label seed: pinned values, different per draw, the same in another process, independent of the trial count;
  * a multi-draw record's frozen column read through `paired_frozen_hits` is the mean over draws of each draw's
    accuracy, on the same trials as the refit column; stored, replayed and in-memory records are identical; the
    draws go to a NEW spec;
  * no reader of the matched records compares column 0 with the labels directly.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

#: produced by eed09e0f (before the change) on `_full_bundle(seed=5)`, window "cue"
LEGACY = {
    "working": dict(
        align="cue-working-seed2520690741", spec_id="6b0a366929d7", digest="aa6311aa125ea85b",
        keys=["post|53|2586691379", "post|53|2586691489", "pre-PS92_0810|48|2580690921", "pre-PS92_0811|48|2580691031",
              "pre-PS92_0812|48|2580691141", "pre-PS92_0813|48|2580691251"]),
    "lick": dict(
        align="cue-lick-seed3013139426", spec_id="6b7a9c4a4fb2", digest="c5d51fe020d7a73d",
        keys=["post|48|3073140046", "post|48|3073140156", "pre-PS92_0810|48|3073139606", "pre-PS92_0811|48|3073139716",
              "pre-PS92_0812|48|3073139826", "pre-PS92_0813|48|3073139936"]),
}


class _Recorder:
    """Stands in for `frozen_models.load_or_fit`: records the spec, runs the fit (a store miss), returns its payload."""

    def __init__(self):
        self.calls = []

    def __call__(self, spec, fit, *, meta=None, refreeze=None, log=print):
        payload = fit()
        self.calls.append({"spec": spec, "meta": meta, "payload": payload})
        return payload, "frozen-new"


def _full_bundle(seed=5, n_sess=4, n_per=60, n_feat=12, n_post=2):
    """The `_pooled_bundle`-shaped fixture of tests/test_frozen_default.py (copied: the pin must not move with it)."""
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


@pytest.fixture
def env(monkeypatch):
    for k in ("WIDEFIELD_EPOCH_FROZEN", "WIDEFIELD_LICK_SOURCE", "WIDEFIELD_LP_FROZEN", "WIDEFIELD_ENL_WINDOW_S",
              "WIDEFIELD_REST_FROZEN", "WIDEFIELD_MATCHED_DRAWS"):
        monkeypatch.delenv(k, raising=False)
    from wfield_local import frozen_models
    from wfield_local import grant_confusion as gc
    rec = _Recorder()
    monkeypatch.setattr(frozen_models, "load_or_fit", rec)
    monkeypatch.setattr(frozen_models, "find", lambda spec: (None, None))
    monkeypatch.setattr(gc, "ANIMALS", ("PS92",))
    monkeypatch.rec = rec
    return monkeypatch


def _collect(monkeypatch, variant="working", bd=None):
    from wfield_local import grant_confusion as gc
    bd = _full_bundle() if bd is None else bd
    monkeypatch.setattr(gc, "_pooled_bundle", lambda an, align: bd)
    return gc._collect_5c_cached.__wrapped__("cue", variant, "paired_matched")


def _records(out):
    pre, by_day = out[0]["PS92"]
    return list(pre) + [by_day[d] for d in sorted(by_day)]


def _matched_calls(rec):
    return [c for c in rec.calls if c["spec"]["kind"] == "matched_frozen"]


# ------------------------------------------------------------------------------------------------- the seed rule
def test_label_seed_is_pinned_and_differs_per_draw_and_label():
    from wfield_local import grant_confusion as gc
    s = [gc.matched_draw_seed("PS93_0818", d) for d in range(200)]
    assert s[:3] == PINNED_SEEDS
    assert len(set(s)) == 200 and all(0 <= x < 2**63 for x in s)
    assert gc.matched_draw_seed("PS93_0819", 0) != s[0] and gc.matched_draw_seed("PS92_0818", 0) != s[0]
    # the draw index is an integer, however it is spelled
    assert gc.matched_draw_seed("PS93_0818", np.int64(1)) == s[1]


#: blake2b("matched-draws|labelseed1|PS93_0818|<d>") >> 1 for d = 0, 1, 2
PINNED_SEEDS = [2545868578640363924, 1275051521005769678, 8183115333295846646]


def test_label_seed_is_the_same_in_another_process():
    """Not Python's salted `hash`: a second interpreter with a different PYTHONHASHSEED returns the same seeds."""
    import os
    code = ("from wfield_local import grant_confusion as gc;"
            "print([gc.matched_draw_seed('PS93_0818', d) for d in range(3)])")
    for hs in ("1", "4242"):
        e = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(ROOT))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, cwd=str(ROOT),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert out.returncode == 0, out.stderr[-600:]
        assert out.stdout.strip().splitlines()[-1] == str(PINNED_SEEDS)


def test_env_values(monkeypatch):
    from wfield_local import grant_confusion as gc
    monkeypatch.delenv(gc.MATCHED_DRAWS_ENV, raising=False)
    assert gc.matched_draws() is None and gc.matched_tag() == ""
    for v, want in (("legacy", None), (" LEGACY ", None), ("1", 1), ("50", 50)):
        monkeypatch.setenv(gc.MATCHED_DRAWS_ENV, v)
        assert gc.matched_draws() == want
    monkeypatch.setenv(gc.MATCHED_DRAWS_ENV, "50")
    assert gc.matched_tag() == "draws50-labelseed1"
    assert gc.matched_spec_align("PS92", "cue", "working") == "cue-working-draws50-labelseed1"
    for bad in ("0", "-3", "many", "2.5"):
        monkeypatch.setenv(gc.MATCHED_DRAWS_ENV, bad)
        with pytest.raises(ValueError, match="WIDEFIELD_MATCHED_DRAWS"):
            gc.matched_draws()


def _drop_one_trial(bd, sess=4):
    """The bundle without the LAST engaged trial of one session (a one-trial change in the trial set)."""
    i = int(np.flatnonzero(bd["GE"] == sess)[-1])
    keep = np.ones(len(bd["GE"]), bool)
    keep[i] = False
    out = dict(bd)
    for k in ("XE", "YE", "GE", "BE", "e_pre"):
        out[k] = bd[k][keep]
    return out


def _seeds_of(payload, tag="post"):
    return sorted(int(k.split("|")[2]) for k in payload if k.split("|")[0] == tag)


def test_label_seeds_do_not_move_with_the_trial_count_and_the_legacy_seeds_do(env):
    """THE DEFECT AND THE FIX. One trial fewer in one post session: every legacy seed of that session changes (a
    different matched model), every label seed stays (the same block order, cut one trial earlier at most)."""
    from wfield_local import grant_confusion as gc
    bd = _full_bundle()
    _collect(env, bd=bd)
    _collect(env, bd=_drop_one_trial(_full_bundle()))
    a, b = (_seeds_of(c["payload"]) for c in _matched_calls(env.rec))
    assert len(set(a) & set(b)) == 1 and a != b            # legacy: the changed session's seed moved, the other stayed
    env.rec.calls.clear()
    env.setenv(gc.MATCHED_DRAWS_ENV, "4")
    _collect(env, bd=bd)
    _collect(env, bd=_drop_one_trial(_full_bundle()))
    a, b = (_seeds_of(c["payload"]) for c in _matched_calls(env.rec))
    assert a == b and len(a) == 8                          # 2 post sessions x 4 draws, identical seeds
    want = sorted(gc.matched_draw_seed(lab, d) for lab in ("PS92_0901", "PS92_0902") for d in range(4))
    assert a == want


def test_same_seed_draws_nested_training_sets_when_the_target_moves():
    """`_matched_frozen` adds whole blocks in the seeded order until n_target: with the SAME seed a smaller target
    trains on a subset of the larger target's rows -- the model changes by at most the last block(s), not wholesale."""
    from wfield_local import grant_confusion as gc
    rng = np.random.default_rng(0)
    blk = np.repeat(np.arange(40), 6)
    y = np.tile(np.arange(6), 40)
    X = rng.normal(size=(len(y), 5))
    seen = []

    class _Pipe:
        def fit(self, Xk, yk):
            seen.append({tuple(r) for r in np.round(Xk, 12)})
            self.classes_ = np.unique(yk)
            return self

    seed = gc.matched_draw_seed("PS93_0818", 0)
    for n in (120, 119, 100):
        gc._matched_frozen(_Pipe, X, y, blk, n, np.random.default_rng(seed))
    assert seen[1] <= seen[0] and seen[2] <= seen[1]
    assert len(seen[0]) == 120 and len(seen[2]) == 102     # whole blocks of 6


# ------------------------------------------------------------------------------------------------ the legacy pin
def _digest(records):
    h = hashlib.sha256()
    for r in records:
        for a in r:
            h.update(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("setting", [None, "legacy"])
@pytest.mark.parametrize("variant", ["working", "lick"])
def test_legacy_single_draw_is_exactly_the_code_before_the_change(env, variant, setting):
    """N = 1 with the old seed rule (the default): the spec (align string, spec id), the payload keys and every record
    equal what eed09e0f produced on this fixture -- so the stored single-draw entries still hit and replay."""
    from wfield_local import frozen_models
    from wfield_local import grant_confusion as gc
    if setting:
        env.setenv(gc.MATCHED_DRAWS_ENV, setting)
    out = _collect(env, variant)
    want = LEGACY[variant]
    (call,) = _matched_calls(env.rec)
    assert call["spec"]["align"] == want["align"]
    assert frozen_models.spec_id(call["spec"]) == want["spec_id"]
    assert sorted(call["payload"]) == want["keys"]
    recs = _records(out)
    assert all(np.asarray(r[1]).dtype.kind == "i" and np.asarray(r[1]).shape[1] == 2 for r in recs)
    assert out[0]["PS92"][0].labels == ["PS92_0810", "PS92_0811", "PS92_0812", "PS92_0813"] and out[1] == [15, 16]
    assert _digest(recs) == want["digest"]


def test_collector_cache_key_separates_legacy_from_many_draws(env):
    """`_collect_5c` keys its lru cache on the run: the matched setting is part of it (only for paired_matched)."""
    from wfield_local import grant_confusion as gc
    seen = []
    env.setattr(gc, "_collect_5c_cached", lambda align, variant, mode, run: seen.append((mode, run)) or ({}, []))
    gc._collect_5c("cue", "working", "paired_matched")
    env.setenv(gc.MATCHED_DRAWS_ENV, "20")
    gc._collect_5c("cue", "working", "paired_matched")
    gc._collect_5c("cue", "working", "paired")
    env.delenv(gc.MATCHED_DRAWS_ENV)
    gc._collect_5c("cue", "working", "paired")
    assert seen[1][1] == ("draws20-labelseed1",) + seen[0][1]            # prepended; the legacy key is unchanged
    assert not any("labelseed" in str(x) for x in seen[0][1])
    assert seen[2][1] == seen[3][1]                        # the unmatched family does not move with the setting


# ---------------------------------------------------------------------------------------------- many draws
@pytest.mark.parametrize("variant", ["working", "lick"])
def test_many_draws_record_is_the_mean_over_draws_on_the_refit_trials(env, variant):
    from wfield_local import grant_confusion as gc
    legacy = _records(_collect(env, variant))
    env.rec.calls.clear()
    env.setenv(gc.MATCHED_DRAWS_ENV, "5")
    cap = []
    env.setattr(gc, "MATCHED_DRAW_CAPTURE", cap)
    out = _collect(env, variant)
    recs = _records(out)
    (call,) = _matched_calls(env.rec)
    assert call["spec"]["align"] == f"cue-{variant}-draws5-labelseed1"          # a NEW spec, never the legacy one
    assert call["spec"]["align"] != LEGACY[variant]["align"]
    assert len(call["payload"]) == 5 * len(recs) == 30 and len(cap) == len(recs)
    for r, old, c in zip(recs, legacy, cap):
        y, p, blk = r
        # same trials, blocks and REFIT column as the one-draw record: only the frozen column changed
        np.testing.assert_array_equal(y, old[0])
        np.testing.assert_array_equal(blk, old[2])
        np.testing.assert_array_equal(p[:, 1], old[1][:, 1])
        assert p.dtype.kind == "f" and p.shape == (len(y), 2) and (p[:, 0] >= gc.MATCHED_FRAC_BASE).all()
        hits = gc.paired_frozen_hits(y, p)
        assert c["hits"].shape == (5, len(y))
        np.testing.assert_allclose(hits, c["hits"].mean(0), rtol=0, atol=1e-14)
        # the session accuracy = the mean over draws of each draw's accuracy, overall and at one position
        np.testing.assert_allclose(hits.mean(), np.mean([h.mean() for h in c["hits"]]), rtol=0, atol=1e-13)
        m = y == 3
        np.testing.assert_allclose(hits[m].mean(), np.mean([h[m].mean() for h in c["hits"]]), rtol=0, atol=1e-13)
        # a reader that still compares column 0 with the labels gets exactly 0, never a plausible number
        assert not (p[:, 0] == y).any()
        assert c["seeds"] == [gc.matched_draw_seed(c["label"], d) for d in range(5)]


def test_many_draws_stored_replayed_and_in_memory_records_are_identical(env):
    from wfield_local import frozen_models
    from wfield_local import grant_confusion as gc
    env.setenv(gc.MATCHED_DRAWS_ENV, "3")
    stored = _records(_collect(env))
    (call,) = _matched_calls(env.rec)
    n_calls = len(env.rec.calls)
    payload = call["payload"]
    env.setattr(frozen_models, "find", lambda spec: ("hit", "local"))
    env.setattr(frozen_models, "_read", lambda d: payload)
    replay = _records(_collect(env))
    assert len(_matched_calls(env.rec)) == 1                                    # nothing refitted / rewritten
    env.setenv("WIDEFIELD_EPOCH_FROZEN", "0")
    mem = _records(_collect(env))
    assert len(_matched_calls(env.rec)) == 1 and n_calls >= 1
    assert len(stored) == len(replay) == len(mem) == 6
    for a, b, c in zip(stored, replay, mem):
        for x, y, z in zip(a, b, c):
            np.testing.assert_array_equal(x, y)
            np.testing.assert_array_equal(x, z)


def test_one_label_seeded_draw_is_a_zero_one_column(env):
    from wfield_local import grant_confusion as gc
    env.setenv(gc.MATCHED_DRAWS_ENV, "1")
    for y, p, _b in _records(_collect(env)):
        assert set(np.unique(gc.paired_frozen_hits(y, p))) <= {0.0, 1.0}


def test_only_labels_hook_scores_only_those_sessions(env):
    from wfield_local import grant_confusion as gc
    env.setenv(gc.MATCHED_DRAWS_ENV, "2")
    env.setattr(gc, "MATCHED_ONLY_LABELS", {"PS92_0902"})
    out = _collect(env)
    pre, by_day = out[0]["PS92"]
    assert pre is None and sorted(by_day) == [16]


# ------------------------------------------------------------------------------------------------- the reducers
def _old_gap(y, p, code):
    m = (y == code) & (p[:, 1] != -1)
    return float(np.mean(p[m, 1] == y[m]) - np.mean(p[m, 0] == y[m]))


def _old_overall(y, p):
    m = p[:, 1] != -1
    fz = float((p[m, 0] == y[m]).mean())
    rf = float((p[m, 1] == y[m]).mean())
    return fz, rf, rf - fz


def test_reducers_unchanged_on_one_draw_records_and_average_draws_on_many():
    from wfield_local import epoch_accuracy as ea
    from wfield_local import grant_confusion as gc
    from wfield_local import recovery_trajectory as rt
    rng = np.random.default_rng(3)
    n, nd = 240, 7
    y = np.tile(np.arange(6), n // 6)
    blk = np.arange(n) // 6
    refit = np.where(rng.random(n) < 0.7, y, (y + 1) % 6)
    refit[y == 5] = -1                                                           # a class the refit could not train on
    draws = np.array([np.where(rng.random(n) < 0.5, y, (y + 2) % 6) for _ in range(nd)])
    one = np.column_stack([draws[0], refit])
    # ONE DRAW (integer labels): exactly the formulas before the change
    for code in range(5):
        assert ea._gap_at(y, one, code) == _old_gap(y, one, code)
    assert ea._gap_at(y, one, 5) is None
    fz, rf, gap = _old_overall(y, one)
    assert (ea._overall_at(y, one, "frozen"), ea._overall_at(y, one, "refit"), ea._overall_at(y, one, "gap")) == (fz, rf, gap)
    assert rt._session_pair((y, one, blk)) == (fz, rf, int((refit != -1).sum()))
    assert rt._pre_pair([(y, one, blk)]) == (fz, rf)
    # MANY DRAWS: the mean over draws of the one-draw statistic, same trial count
    hits = (draws == y).sum(0)
    many = np.column_stack([gc.matched_frozen_column(hits, nd), refit.astype(float)])
    per = [np.column_stack([d, refit]) for d in draws]
    for code in range(5):
        np.testing.assert_allclose(ea._gap_at(y, many, code), np.mean([_old_gap(y, q, code) for q in per]),
                                   rtol=0, atol=1e-13)
    np.testing.assert_allclose(ea._overall_at(y, many, "frozen"), np.mean([_old_overall(y, q)[0] for q in per]),
                               rtol=0, atol=1e-13)
    assert ea._overall_at(y, many, "refit") == rf
    np.testing.assert_allclose(ea._overall_at(y, many, "gap"), np.mean([_old_overall(y, q)[2] for q in per]),
                               rtol=0, atol=1e-13)
    f2, r2, n2 = rt._session_pair((y, many, blk))
    assert r2 == rf and n2 == int((refit != -1).sum())
    np.testing.assert_allclose(f2, np.mean([_old_overall(y, q)[0] for q in per]), rtol=0, atol=1e-13)
    # pooled one-draw + many-draw rows read per trial
    both = gc.paired_frozen_hits(np.concatenate([y, y]), np.vstack([one.astype(float), many]))
    np.testing.assert_array_equal(both[:n], (draws[0] == y).astype(float))
    np.testing.assert_allclose(both[n:], hits / nd, rtol=0, atol=1e-14)


#: every module that reads `_collect_5c(..., "paired_matched")` records (list them here when one is added)
MATCHED_READERS = ("wfield_local/epoch_accuracy.py", "wfield_local/recovery_trajectory.py",
                   "scripts/gap_time_variance.py")
_DIRECT = re.compile(r"\[[^\]\n]*,\s*0\]\s*==|==\s*\w+\[[^\]\n]*,\s*0\]")


def test_no_matched_reader_compares_the_frozen_column_with_the_labels_directly():
    """Column 0 of a multi-draw matched record is an encoded fraction: it is read ONLY through `paired_frozen_hits`."""
    for rel in MATCHED_READERS:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert not _DIRECT.search(src), f"{rel} compares column 0 of a paired record directly; use paired_frozen_hits"
    asked = sorted(p.relative_to(ROOT).as_posix()
                   for d in ("wfield_local", "scripts") for p in (ROOT / d).rglob("*.py")
                   if "/archive/" not in p.as_posix()
                   and re.search(r"[\"']paired_matched[\"']", p.read_text(encoding="utf-8", errors="replace"))
                   # june_august_drift only NAMES the mode in its docstring: it has its own draws, seeds and store spec
                   and p.name not in ("grant_confusion.py", "matched_draws_pilot.py", "june_august_drift.py"))
    assert asked == sorted(MATCHED_READERS), asked


# ------------------------------------------------------------------------------------------ the freeze allow-list
def test_freeze_only_lets_a_render_mint_only_the_listed_kinds(monkeypatch, tmp_path):
    """WIDEFIELD_FREEZE_ONLY: a miss of a listed kind / align prefix fits and stores; any other miss raises (the
    matched-draws render must find every epoch decoder it reads already stored)."""
    from wfield_local import frozen_models as fm
    monkeypatch.delenv(fm.LOAD_ONLY_ENV, raising=False)
    monkeypatch.setenv(fm.FREEZE_ONLY_ENV, "matched_frozen:cue-working-draws250")
    ok = {"kind": "matched_frozen", "align": "cue-working-draws250-labelseed1"}
    assert fm.freeze_allowed(ok)
    assert not fm.freeze_allowed({"kind": "matched_frozen", "align": "cue-working-seed684029838"})
    assert not fm.freeze_allowed({"kind": "epoch_decoder", "align": "cue"})
    monkeypatch.setenv(fm.FREEZE_ONLY_ENV, "matched_frozen")
    assert fm.freeze_allowed({"kind": "matched_frozen", "align": "lick-lick-seed1"})
    monkeypatch.delenv(fm.FREEZE_ONLY_ENV)
    assert fm.freeze_allowed({"kind": "epoch_decoder", "align": "cue"})
    # through load_or_fit, on an empty tmp store: the disallowed miss raises before any fit
    monkeypatch.setenv("WIDEFIELD_FROZEN_MODEL_DIR", str(tmp_path / "store"))
    monkeypatch.setenv(fm.ALLOW_FREEZE_ENV, "1")   # main: load-only is the default; the allow-list narrows an allowed freeze
    monkeypatch.setenv(fm.FREEZE_ONLY_ENV, "matched_frozen")
    spec = fm.make_spec("PS92", "epoch_decoder", align="cue", source="locanmf", train_labels=["PS92_0810"],
                        post_s=2.0, n_features=3, train_data=(np.zeros((4, 3)), np.arange(4)))
    called = []
    with pytest.raises(RuntimeError, match="WIDEFIELD_FREEZE_ONLY"):
        fm.load_or_fit(spec, lambda: called.append(1) or {"m": 1})
    assert not called and not (tmp_path / "store").exists()
