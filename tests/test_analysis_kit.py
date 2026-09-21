"""`analysis_kit` must reproduce the nine copies it replaces, draw for draw.

WHY PINNING THE RNG SEQUENCE IS THE RIGHT TEST HERE. The documented failure mode of this repo's
bootstraps is not a wrong formula -- it is a DIFFERENT DRAW ORDER under the same seed, which moves
a CI while leaving the point estimate exact and therefore survives every eyeball check. It cost a
reproducibility bug on 2026-09-20 ([-22.9, -13.7] -> [-23.1, -13.6]). `STATUS_2026-09-21` says the
verification bar for a refactor here is "run before, run after, diff the per-session output"; for
the bootstrap specifically this is STRONGER and takes milliseconds instead of ninety minutes,
because a literal copy of the pre-extraction source is checked in below and compared bit for bit.

THE COPIES BELOW ARE FROZEN AND MUST NOT BE "TIDIED". They are the 2026-09-20 sources of
`rest_coupling._boot`, `evoked_hrf_latency._boot` and `rest_coupling._boot_diff`, preserved
verbatim including the guards one of them lacks. Editing them to match the kit would make this
test assert that the kit equals itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import analysis_kit as ak

N_BOOT = 4000


# --------------------------------------------------------------------------------------------
# The frozen pre-extraction sources. Do not edit.
# --------------------------------------------------------------------------------------------

def _boot_guarded(by_animal, rng, n_boot=N_BOOT):
    """Verbatim `rest_coupling._boot` / `quit_point._boot` / `nvc_evoked._boot_ci` as of d4bfb47."""
    animals = sorted(by_animal)
    if not animals:
        return None
    flat = [v for a in animals for v in by_animal[a]]
    o = []
    for _ in range(n_boot):
        vals = []
        for a in (animals[i] for i in rng.integers(0, len(animals), len(animals))):
            sa = by_animal[a]
            vals += [sa[i] for i in rng.integers(0, len(sa), len(sa))]
        if vals:
            o.append(float(np.mean(vals)))
    if len(o) < n_boot // 4:
        return None
    o = np.asarray(o)
    return float(np.mean(flat)), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def _boot_unguarded(by, rng, n_boot=N_BOOT):
    """Verbatim `quit_prodrome._boot` / `lick_bout_structure._boot` as of d4bfb47."""
    A = sorted(by)
    if not A:
        return None
    flat = [x for k in A for x in by[k]]
    o = []
    for _ in range(n_boot):
        vals = []
        for k in (A[i] for i in rng.integers(0, len(A), len(A))):
            sa = by[k]
            vals += [sa[i] for i in rng.integers(0, len(sa), len(sa))]
        o.append(float(np.mean(vals)))
    o = np.asarray(o)
    return float(np.mean(flat)), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def _boot_diff_frozen(post_by_animal, pre_by_animal, rng, n_boot=N_BOOT):
    """Verbatim `rest_coupling._boot_diff` / `evoked_hrf_latency._boot_diff` as of d4bfb47."""
    animals = sorted(set(post_by_animal) & set(pre_by_animal))
    if not animals:
        return None
    obs = float(np.mean([float(np.mean(post_by_animal[x])) - float(np.mean(pre_by_animal[x]))
                         for x in animals]))
    o = []
    for _ in range(n_boot):
        d = []
        for x in (animals[i] for i in rng.integers(0, len(animals), len(animals))):
            pa, qa = post_by_animal[x], pre_by_animal[x]
            d.append(float(np.mean([pa[i] for i in rng.integers(0, len(pa), len(pa))]))
                     - float(np.mean([qa[i] for i in rng.integers(0, len(qa), len(qa))])))
        o.append(float(np.mean(d)))
    o = np.asarray(o)
    return obs, float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


# --------------------------------------------------------------------------------------------
# Fixtures shaped like the real cohort: four animals, wildly unequal session counts.
# --------------------------------------------------------------------------------------------

def _cohort(seed=7):
    """PS92-PS95 with 15/3/8/5 sessions. UNBALANCED ON PURPOSE -- a balanced fixture cannot tell a
    flat-pool mean from a mean of animal means, which is the distinction this module documents."""
    g = np.random.default_rng(seed)
    return {"PS92": list(g.normal(10, 2, 15)), "PS93": list(g.normal(14, 2, 3)),
            "PS94": list(g.normal(11, 2, 8)), "PS95": list(g.normal(9, 2, 5))}


def _pre_post(seed=11):
    g = np.random.default_rng(seed)
    post = {"PS92": list(g.normal(12, 2, 9)), "PS93": list(g.normal(15, 2, 4)),
            "PS94": list(g.normal(13, 2, 6)), "PS95": list(g.normal(10, 2, 7))}
    pre = {"PS92": list(g.normal(10, 2, 15)), "PS93": list(g.normal(14, 2, 1)),
           "PS94": list(g.normal(11, 2, 8)), "PS96": list(g.normal(9, 2, 5))}
    return post, pre


# --------------------------------------------------------------------------------------------
# 1. Bit-for-bit against the frozen sources
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("frozen", [_boot_guarded, _boot_unguarded])
def test_boot_ci_reproduces_the_pre_extraction_draws(frozen):
    by = _cohort()
    want = frozen(by, np.random.default_rng(20260919))
    got = ak.boot_ci(by, np.random.default_rng(20260919))
    assert got is not None and want is not None
    # EXACT, not approximate. A tolerance here would pass the very bug this guards against: the
    # completion-order CI shift was 0.2 licks/min on a -18.7 estimate, well inside any sane rtol.
    assert (got.point, got.lo, got.hi) == want


def test_boot_delta_reproduces_the_pre_extraction_draws():
    post, pre = _pre_post()
    want = _boot_diff_frozen(post, pre, np.random.default_rng(20260920))
    got = ak.boot_delta(post, pre, np.random.default_rng(20260920))
    assert got is not None and want is not None
    assert (got.point, got.lo, got.hi) == want


def test_the_two_frozen_copies_agreed_with_each_other():
    """The guarded and unguarded copies must be equivalent on data where no animal is empty.

    If this ever fails, the extraction was NOT safe and the divergence has to be resolved
    deliberately rather than by adopting whichever copy was extracted.
    """
    by = _cohort()
    a = _boot_guarded(by, np.random.default_rng(3))
    b = _boot_unguarded(by, np.random.default_rng(3))
    assert a == b


# --------------------------------------------------------------------------------------------
# 2. The two weighting conventions, pinned because they DISAGREE
# --------------------------------------------------------------------------------------------

def test_boot_ci_point_is_the_flat_pool_mean_not_the_mean_of_animal_means():
    by = _cohort()
    flat = float(np.mean([v for a in sorted(by) for v in by[a]]))
    per_animal = float(np.mean([np.mean(by[a]) for a in sorted(by)]))
    assert abs(flat - per_animal) > 0.1, "fixture is too balanced to distinguish the conventions"
    assert ak.boot_ci(by, np.random.default_rng(0)).point == pytest.approx(flat)


def test_boot_delta_point_is_animal_weighted():
    """The complement of the test above, and the one the 4496/4338 retraction turned on."""
    post, pre = _pre_post()
    shared = sorted(set(post) & set(pre))
    want = float(np.mean([np.mean(post[a]) - np.mean(pre[a]) for a in shared]))
    assert ak.boot_delta(post, pre, np.random.default_rng(0)).point == pytest.approx(want)


def test_boot_delta_uses_only_animals_present_in_both_arms():
    post, pre = _pre_post()
    assert "PS95" in post and "PS95" not in pre and "PS96" in pre
    assert ak.boot_delta(post, pre, np.random.default_rng(0)).n_animals == 3


def test_boot_delta_pairs_is_the_same_bootstrap():
    post, pre = _pre_post()
    shared = sorted(set(post) & set(pre))
    pairs = {a: (post[a], pre[a]) for a in shared}
    want = ak.boot_delta({a: post[a] for a in shared}, {a: pre[a] for a in shared},
                         np.random.default_rng(5))
    assert ak.boot_delta_pairs(pairs, np.random.default_rng(5)) == want


# --------------------------------------------------------------------------------------------
# 3. The guards
# --------------------------------------------------------------------------------------------

def test_an_empty_animal_does_not_poison_the_interval():
    """The reason the guarded copy is the one that was adopted.

    `defaultdict(list)` creates a key on READ, so a cell nobody contributed to arrives here as an
    animal with no sessions. Unguarded, `np.mean([])` is NaN and one NaN makes every percentile
    NaN -- a table of NaNs reads as "no data" rather than as a bug.
    """
    by = dict(_cohort())
    by["PS96"] = []
    got = ak.boot_ci(by, np.random.default_rng(1))
    assert got is not None
    assert np.isfinite([got.point, got.lo, got.hi]).all()


def test_all_animals_empty_returns_none_without_short_circuiting_the_rng():
    """None, AND the same number of draws the frozen code consumed.

    Callers share one `rng` across dozens of table cells. Returning early would leave this cell's
    answer correct and silently change every cell computed after it.
    """
    by = {"PS92": [], "PS93": []}
    r1, r2 = np.random.default_rng(2), np.random.default_rng(2)
    assert ak.boot_ci(by, r1) is None
    assert _boot_guarded(by, r2) is None
    assert r1.integers(0, 10 ** 9) == r2.integers(0, 10 ** 9)


def test_no_animals_returns_none():
    assert ak.boot_ci({}, np.random.default_rng(0)) is None
    assert ak.boot_delta({}, {}, np.random.default_rng(0)) is None


def test_excludes_zero_is_the_star_in_the_tables():
    assert ak.Interval(1.0, 0.2, 1.8, 4).excludes_zero
    assert ak.Interval(-1.0, -1.8, -0.2, 4).excludes_zero
    assert not ak.Interval(1.0, -0.2, 1.8, 4).excludes_zero
    assert not ak.Interval(0.0, 0.0, 1.0, 4).excludes_zero


# --------------------------------------------------------------------------------------------
# 4. Ordering -- the two traps this module exists to close
# --------------------------------------------------------------------------------------------

def test_fan_sessions_sorts_results_that_arrive_in_completion_order(monkeypatch):
    """The bug from `DECISIONS.md`, made unrepresentable.

    `fan_out` is stubbed rather than run: the point under test is the SORT, and spawning a real
    pool inside the suite would test the standard library.
    """
    from wfield_local import parallel

    labels = ["PS92_0812", "PS95_0903", "PS93_0820", "PS94_0827"]

    def _shuffled(items, worker, **kw):
        return [(it, worker(it)) for it in reversed(list(items))], []

    monkeypatch.setattr(parallel, "fan_out", _shuffled)
    res, fail = ak.fan_sessions(labels, str, log=lambda *a, **k: None)
    assert [it for it, _v in res] == sorted(labels)
    assert fail == []


def test_fan_sessions_sorts_on_the_label_when_options_travel_in_the_item(monkeypatch):
    """`quit_prodrome` and `lick_bout_structure` fan over ``(label, gate, horizon)`` tuples,
    because spawn does not carry a runtime global into a worker. Sorting a tuple whose first
    element is the label is the same order as sorting the labels."""
    from wfield_local import parallel

    items = [("PS95_0903", True, 90.0), ("PS92_0812", True, 90.0), ("PS93_0820", True, 90.0)]
    monkeypatch.setattr(parallel, "fan_out",
                        lambda it, w, **kw: ([(x, w(x)) for x in reversed(list(it))], []))
    res, _ = ak.fan_sessions(items, lambda x: x[0], log=lambda *a, **k: None)
    assert [it[0] for it, _v in res] == ["PS92_0812", "PS93_0820", "PS95_0903"]


def test_curated_sessions_preserves_load_sessions_order_by_default(monkeypatch):
    """**THE DEFAULT IS NOT SORTED, AND THAT IS NOT AN OVERSIGHT.**

    `config.load_sessions` returns an unsorted list (measured: 119 sessions, order != sorted), and
    all seventy hand-written copies of this filter preserve it. A serial module builds its
    bootstrap pool by iterating the result, so quietly sorting here would change the draw sequence
    and move published CIs -- the completion-order bug from the other direction.
    """
    from wfield_local import config, epochs

    sess = [{"label": x} for x in ["PS95_0903", "PS92_0812", "PS93_0820", "PS94_0827"]]
    monkeypatch.setattr(config, "load_sessions", lambda *a, **k: sess)
    monkeypatch.setattr(config, "phase_labels", lambda ph, *a, **k:
                        ["PS95_0903", "PS92_0812"] if ph == "pre" else ["PS93_0820", "PS94_0827"])
    monkeypatch.setattr(config, "animal_of", lambda lab: lab.split("_")[0])
    monkeypatch.setattr(epochs, "epoch_of", lambda lab: "pre")

    assert ak.curated_labels() == ["PS95_0903", "PS92_0812", "PS93_0820", "PS94_0827"]
    assert ak.curated_labels(order="sorted") == sorted(x["label"] for x in sess)
    assert ak.curated_labels(["PS93", "PS94"]) == ["PS93_0820", "PS94_0827"]


def test_curated_sessions_drops_labels_with_no_epoch(monkeypatch):
    from wfield_local import config, epochs

    sess = [{"label": x} for x in ["PS92_0812", "PS93_0820"]]
    monkeypatch.setattr(config, "load_sessions", lambda *a, **k: sess)
    monkeypatch.setattr(config, "phase_labels", lambda ph, *a, **k:
                        [x["label"] for x in sess] if ph == "pre" else [])
    monkeypatch.setattr(config, "animal_of", lambda lab: lab.split("_")[0])
    monkeypatch.setattr(epochs, "epoch_of", lambda lab: "" if lab == "PS93_0820" else "pre")

    assert ak.curated_labels() == ["PS92_0812"]
    assert ak.curated_labels(require_epoch=False) == ["PS92_0812", "PS93_0820"]


def test_input_order_restores_the_serial_loops_order_not_alphabetical(monkeypatch):
    """The key that makes a serial->parallel conversion provably behaviour-preserving.

    `load_sessions` order is not sorted, so collecting a fan-out alphabetically reorders every
    pool the loop builds and moves the CIs while leaving the point estimates exact -- invisible in
    review, and the same signature as the three ordering bugs in `DECISIONS.md`.
    """
    from wfield_local import parallel

    labels = ["PS95_0903", "PS92_0812", "PS94_0827", "PS93_0820"]   # deliberately unsorted
    monkeypatch.setattr(parallel, "fan_out",
                        lambda it, w, **kw: ([(x, w(x)) for x in reversed(list(it))], []))

    res, _ = ak.fan_sessions(labels, str, key=ak.input_order(labels), log=lambda *a, **k: None)
    assert [it for it, _v in res] == labels

    items = [(lab, True, 90.0) for lab in labels]
    res, _ = ak.fan_sessions(items, lambda x: x[0], key=ak.input_order(labels),
                             log=lambda *a, **k: None)
    assert [it[0] for it, _v in res] == labels

    # and without it, the default is alphabetical -- which is the thing to be deliberate about
    res, _ = ak.fan_sessions(labels, str, log=lambda *a, **k: None)
    assert [it for it, _v in res] == sorted(labels) != labels
