"""Consolidation A3 (Priya 11-03): every change-vs-pre is animal-weighted, paired, and the point describes the same
animals as the draws -- `analysis_kit.paired_animal_draws`, used by boot_delta, epoch_figures.scalar_contrast_draws /
contrast_draws, matrix_bootstrap.delta_draws and epoch_behaviour._behaviour_contrast."""
import numpy as np

from wfield_local import analysis_kit as ak
from wfield_local import epoch_behaviour as eb
from wfield_local import epoch_figures as ef
from wfield_local import matrix_bootstrap as mb

# A has 6 sessions changing by +1, B has 1 session changing by -1; C has no pre (cannot supply a change).
POST = {"A": [1.0] * 6, "B": [-1.0], "C": [5.0]}
PRE = {"A": [0.0] * 3, "B": [0.0]}


def test_animal_weighted_and_same_animals():
    point, draws, animals = ak.paired_animal_draws(POST, PRE, ak._mean_of, np.random.default_rng(0), 200)
    assert animals == ["A", "B"]                  # C left out of the point AND the draws
    assert point == 0.0                           # (+1 + -1) / 2, not the session-weighted (6 - 1) / 7
    assert set(np.round(draws, 9)) <= {-1.0, 0.0, 1.0}


def test_scalar_contrast_is_boot_delta():
    pts = {"acute": {"k": [(a, v) for a, vs in POST.items() for v in vs]},
           "pre": {"k": [(a, v) for a, vs in PRE.items() for v in vs]}}
    point, draws = ef.scalar_contrast_draws(pts, "acute", "pre", "k", rng=np.random.default_rng(1), n_boot=200)
    iv, d2 = ak.boot_delta({a: POST[a] for a in "AB"}, PRE, np.random.default_rng(1), 200, draws=True)
    assert point == iv.point == 0.0
    assert np.array_equal(draws, d2)


def test_contrast_draws_stat_per_animal(monkeypatch):
    # per_animal[an] = (pre record, {day: record}); record = (y, p, block). stat = mean(y == p) (accuracy).
    def rec(acc, n=20):
        y = np.zeros(n, int)
        p = np.where(np.arange(n) < acc * n, 0, 1)
        return (y, p, np.arange(n) // 5)
    per_animal = {"A": ([rec(0.5)], {d: rec(1.0, 200) for d in range(1, 7)}),     # +0.5, many trials
                  "B": ([rec(0.5)], {7: rec(0.0)})}                               # -0.5, few trials
    monkeypatch.setattr(ef, "group_days_by_epoch",
                        lambda pa: {"acute": [(an, d) for an, (_p, by) in pa.items() for d in by]})
    got = ef.contrast_draws(per_animal, "acute", "pre", lambda y, p: float(np.mean(y == p)),
                            rng=np.random.default_rng(2), n_boot=200)
    assert got is not None and abs(got[0]) < 1e-12     # animals weigh equally; trial-pooled would be ~ +0.49


def test_matrix_delta_animal_weighted():
    g = {"acute": {"A": [np.ones((2, 2))] * 6, "B": [-np.ones((2, 2))]},
         "pre": {"A": [np.zeros((2, 2))], "B": [np.zeros((2, 2))]}}
    point, _d = mb.delta_draws(g, "acute", "pre", rng=np.random.default_rng(3), n_boot=200)
    assert np.allclose(point, 0.0)


def test_behaviour_contrast_animal_weighted():
    per = {"acute": {"far": [("A", 10, 10)] * 6 + [("B", 0, 10)]},
           "pre": {"far": [("A", 5, 10), ("B", 5, 10)]}}
    point, _d = eb._behaviour_contrast(per, "acute", "far", rng=np.random.default_rng(4), n_boot=200)
    assert abs(point) < 1e-12                     # (+0.5 + -0.5) / 2; session-weighted was (6*0.5 - 0.5) / 7
