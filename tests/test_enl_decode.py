"""`enl_decode` -- the properties the sensory-vs-plan claim rests on.

The sharp one is `test_shared_decoder_holds_blocks_out_of_every_arm`. Readout 4 trains on `success`
and scores `stopped` and `miss_working` with the same fitted model, and those arms come from the SAME
sessions and the same position blocks -- a stopped trial and a success trial in one block share a
spout position and sit seconds apart. If the fold that scores a stopped trial trained on success
trials from its own block, the ratio would report memorisation, and it would report it as the answer
to the question the analysis exists to ask.
"""
import numpy as np
import pytest

from wfield_local import enl_decode as ed


def _arm(y, g, X):
    return {"X": np.asarray(X, float), "y": np.asarray(y), "g": np.asarray(g)}


def test_pool_arms_offsets_block_groups_across_sessions():
    """Block ids restart per session; pooling them unchanged leaks SESSION into the CV folds."""
    one = {a: _arm([0, 1], [0, 1], np.zeros((2, 3))) for a in ("miss_working", "stopped", "success")}
    pooled = ed.pool_arms([one, one])
    for arm in ("miss_working", "stopped", "success"):
        assert sorted(pooled[arm]["g"].tolist()) == [0, 1, 2, 3]


def test_ratio_is_undefined_when_the_denominator_sits_at_its_null():
    """A fraction of an effect that is not established is not a number.

    The first version tested ``den - 1/6 > 0`` and so returned 100 for an arm AT chance, because
    ``0.167 - 1/6`` is a tiny positive rather than zero.
    """
    at_null = {"balanced_accuracy": 0.167, "bal_null_mean": 0.1666, "above_null_balanced": False}
    good = {"balanced_accuracy": 0.40, "bal_null_mean": 0.167, "above_null_balanced": True}
    val, note = ed.ratio(good, at_null, den_name="miss_working")
    assert val is None
    assert "miss_working" in note
    val, note = ed.ratio(at_null, good)
    assert val == pytest.approx((0.167 - 0.1666) / (0.40 - 0.167), rel=1e-6)
    assert "NOT above its own null" in note        # computed, but flagged as unestablished


def test_underpowered_separates_could_not_test_from_found_nothing():
    y = np.tile(np.arange(6), 10)                                   # 60 trials, all six positions
    under, why = ed.underpowered(y)
    assert under and "60 trials" in why
    y = np.repeat(np.arange(5), 200)                                # 1000 trials, one position absent
    under, why = ed.underpowered(y)
    assert under and "5/6 positions" in why
    assert ed.underpowered(np.tile(np.arange(6), 200)) == (False, "")


def test_shared_profile_is_what_both_arms_can_supply():
    """The elementwise MINIMUM share: a target above an arm's own share collapses its draw."""
    rich = np.tile(np.arange(6), 100)
    poor = np.concatenate([np.zeros(100, int), np.tile(np.arange(6), 5)])
    tf = ed.shared_profile([rich, poor])
    assert tf.sum() == pytest.approx(1.0)
    assert tf.min() > 0


def test_shared_decoder_holds_blocks_out_of_every_arm():
    """Features carry BLOCK IDENTITY and the label is a function of the block.

    So a fold that trained on a trial's own block would score it perfectly, and a fold that did not
    cannot do better than chance. Correct hold-out therefore pins every arm near chance -- including
    the two arms that are only ever SCORED, which is the leakage path a `GroupKFold` on the training
    arm alone would leave open.
    """
    rng = np.random.default_rng(0)
    nblk, sig = 12, np.eye(12) * 40.0                      # one unmistakable signature per block

    def build(n):
        g = rng.integers(0, nblk, n)
        return _arm(g % 6, g, sig[g] + rng.normal(scale=0.1, size=(n, nblk)))

    pooled = {"success": build(1200), "miss_working": build(300), "stopped": build(300)}
    preds, cover = ed.shared_decoder(pooled["success"], pooled)
    assert preds is not None
    for arm in ("success", "miss_working", "stopped"):
        assert cover[arm] == pytest.approx(1.0)            # every block is held out in some fold
        acc = float((np.asarray(preds[arm], int) == pooled[arm]["y"]).mean())
        assert acc < 0.45, f"{arm} scored {acc:.3f} -- its own block was in the training set"


def test_shared_decoder_recovers_a_label_that_is_not_block_confounded():
    """The companion to the leakage test: hold-out must not destroy a real, generalising code."""
    rng = np.random.default_rng(1)
    tmpl = rng.normal(size=(6, 20)) * 3.0

    def build(n):
        g = rng.integers(0, 12, n)
        y = rng.integers(0, 6, n)
        return _arm(y, g, tmpl[y] + rng.normal(size=(n, 20)))

    pooled = {"success": build(1200), "miss_working": build(300), "stopped": build(300)}
    preds, _ = ed.shared_decoder(pooled["success"], pooled)
    for arm in ("success", "miss_working", "stopped"):
        acc = float((np.asarray(preds[arm], int) == pooled[arm]["y"]).mean())
        assert acc > 0.8, f"{arm} scored only {acc:.3f}"


def test_shared_decoder_reports_coverage_below_one():
    """An arm whose blocks are absent from the training arm is SCORED ON A SUBSET -- say so."""
    rng = np.random.default_rng(2)
    train = _arm(rng.integers(0, 6, 600), rng.integers(0, 8, 600), rng.normal(size=(600, 5)))
    # blocks 8-11 exist only here, so no fold of the training arm ever holds them out
    far = _arm(rng.integers(0, 6, 200), rng.integers(4, 12, 200), rng.normal(size=(200, 5)))
    _preds, cover = ed.shared_decoder(train, {"stopped": far})
    assert 0.0 < cover["stopped"] < 1.0


def test_shared_decoder_declines_a_training_arm_it_cannot_split():
    one_class = _arm(np.zeros(50, int), np.arange(50) % 5, np.zeros((50, 3)))
    assert ed.shared_decoder(one_class, {"stopped": one_class}) == (None, None)
    one_block = _arm(np.arange(50) % 6, np.zeros(50, int), np.zeros((50, 3)))
    assert ed.shared_decoder(one_block, {"stopped": one_block}) == (None, None)


def test_common_positions_is_the_intersection_not_the_union():
    rich = _arm(np.tile(np.arange(6), 50), np.arange(300) % 8, np.zeros((300, 3)))
    thin = _arm(np.array([0, 0, 2, 5]), np.array([0, 1, 2, 3]), np.zeros((4, 3)))
    assert ed.common_positions([rich, thin]) == [0, 2, 5]
    # DISPLAY_ORDER is a display order, NOT sorted, and the label set handed to `evaluate_arm` has
    # to keep it -- `recall_by_position` and the matched draw are both reported against it.
    assert ed.common_positions([rich, rich]) == list(ed.na.DISPLAY_ORDER)


def test_readout4_restricts_both_arms_to_the_positions_they_share():
    """PS92 pre-stroke has SIX stopped trials and its balanced null printed as 0.333, not 1/6.

    A three-way macro-recall divided by a six-way one is not a ratio of one quantity, so the shared
    label set has to be enforced -- and the restriction has to be VISIBLE in the result.
    """
    rng = np.random.default_rng(4)
    tmpl = rng.normal(size=(6, 20)) * 3.0

    def build(n, keep):
        g = rng.integers(0, 16, n)
        y = rng.choice(keep, n)
        return _arm(y, g, tmpl[y] + rng.normal(size=(n, 20)))

    pooled = {"success": build(1500, np.arange(6)), "miss_working": build(300, np.arange(6)),
              "stopped": build(60, np.array([1, 3, 4]))}
    r = ed.analyse(pooled, n_perm=200, do_transfer=False)
    assert r["shared_positions"]["used"] and len(r["shared_positions"]["dropped"]) == 3
    for arm in ("success", "miss_working", "stopped"):
        d = r["shared"][arm]
        assert d["positions_used"] == 3
        # every scored trial is at a shared position, so the null is ~1/3 for BOTH arms alike
        assert d["bal_null_mean"] == pytest.approx(1 / 3, abs=0.08)
    assert r["shared"]["success"]["n"] < r["shared"]["success"]["n_total"]    # 6-way arm was subset


def test_readout4_leaves_a_full_six_position_cell_unrestricted():
    rng = np.random.default_rng(5)
    tmpl = rng.normal(size=(6, 20)) * 3.0

    def build(n):
        g = rng.integers(0, 16, n)
        y = rng.integers(0, 6, n)
        return _arm(y, g, tmpl[y] + rng.normal(size=(n, 20)))

    r = ed.analyse({"success": build(1500), "miss_working": build(300), "stopped": build(400)},
                   n_perm=200, do_transfer=False)
    assert r["shared_positions"]["dropped"] == []
    assert r["shared"]["stopped"]["positions_used"] == 6
    assert r["shared"]["stopped"]["bal_null_mean"] == pytest.approx(1 / 6, abs=0.03)


def _scored(y, g, pred):
    return {"y": np.asarray(y), "g": np.asarray(g), "pred": np.asarray(pred),
            "n_total": len(y), "coverage": 1.0}


def test_bootstrap_resampling_unit_is_the_block_not_the_trial():
    """`decode_ci` records why: ~6 correlated trials per block, so trial resampling is too narrow."""
    rng = np.random.default_rng(6)
    g = np.repeat(np.arange(20), 30)                       # 20 blocks, 600 trials
    y = rng.integers(0, 6, 600)
    sc = {"success": _scored(y, g, y.copy()), "stopped": _scored(y, g, y.copy())}
    out = ed.bootstrap_arms(sc, {"success": 1 / 6, "stopped": 1 / 6}, list(range(6)), n_boot=200)
    assert out["n_effective"] == 20                        # NOT 600


def test_bootstrap_does_not_separate_two_arms_that_are_the_same():
    rng = np.random.default_rng(7)
    g = np.repeat(np.arange(24), 20)
    y = rng.integers(0, 6, 480)
    pred = np.where(rng.random(480) < 0.5, y, rng.integers(0, 6, 480))
    sc = {"success": _scored(y, g, pred), "miss_working": _scored(y, g, pred.copy())}
    out = ed.bootstrap_arms(sc, {"success": 1 / 6, "miss_working": 1 / 6}, list(range(6)),
                            n_boot=400)
    d = out["differences"]["success_minus_miss_working"]
    assert d["p"] > 0.5 and d["ci"][0] <= 0 <= d["ci"][1]


def test_bootstrap_separates_a_strong_arm_from_a_chance_arm():
    rng = np.random.default_rng(8)
    g = np.repeat(np.arange(24), 20)
    y = rng.integers(0, 6, 480)
    strong = np.where(rng.random(480) < 0.75, y, rng.integers(0, 6, 480))
    sc = {"success": _scored(y, g, strong), "stopped": _scored(y, g, rng.integers(0, 6, 480))}
    out = ed.bootstrap_arms(sc, {"success": 1 / 6, "stopped": 1 / 6}, list(range(6)), n_boot=400)
    d = out["differences"]["success_minus_stopped"]
    assert d["p"] < 0.01 and d["ci"][0] > 0
    assert out["arms"]["success"]["ci"][0] > out["arms"]["stopped"]["ci"][1]


def test_bootstrap_flags_draws_where_the_denominator_collapses():
    """A ratio whose denominator keeps falling to its null is not a measurement -- say how often."""
    rng = np.random.default_rng(9)
    g = np.repeat(np.arange(24), 20)
    y = rng.integers(0, 6, 480)
    sc = {"success": _scored(y, g, rng.integers(0, 6, 480)),      # denominator AT chance
          "stopped": _scored(y, g, rng.integers(0, 6, 480))}
    out = ed.bootstrap_arms(sc, {"success": 1 / 6, "stopped": 1 / 6}, list(range(6)), n_boot=400)
    assert out["ratios"]["stopped_over_success"]["frac_undefined"] > 0.2


def test_analyse_flags_a_stopped_arm_with_no_position_code():
    """The negative control -- the one that matters. Noise in `stopped`, signal elsewhere."""
    rng = np.random.default_rng(3)
    tmpl = rng.normal(size=(6, 20)) * 3.0

    def build(n, strength):
        g = rng.integers(0, 16, n)
        y = rng.integers(0, 6, n)
        return _arm(y, g, strength * tmpl[y] + rng.normal(size=(n, 20)))

    pooled = {"success": build(1500, 1.0), "miss_working": build(300, 1.0),
              "stopped": build(400, 0.0)}
    r = ed.analyse(pooled, n_perm=200, do_transfer=True)
    assert r["shared"]["success"]["above_null_balanced"]
    assert not r["shared"]["stopped"]["above_null_balanced"]
    for key in ("clean_ratio_vs_working", "clean_ratio_vs_success"):
        assert "NOT above its own null" in r[key]["note"]
