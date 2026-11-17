"""glm_hmm: recovers a known multi-output 2-state model from simulated sessions (before it touches real data)."""
import numpy as np
from scipy.special import softmax

from wfield_local import glm_hmm as gh


def _simulate(seed=0, n_seq=12, T=300):
    rng = np.random.default_rng(seed)
    D, n_cls = 3, [3, 2, 3]
    W = [np.stack([rng.normal(0, 1.5, (c - 1, D)) for _ in range(2)]) for c in n_cls]
    W[0][0, :, 0], W[0][1, :, 0] = [-2.0, -2.0], [1.5, 2.0]        # state 0 = responds, state 1 = does not
    A = np.array([[0.97, 0.03], [0.05, 0.95]])
    seqs, zs = [], []
    for _ in range(n_seq):
        X = np.c_[np.ones(T), rng.integers(0, 2, (T, D - 1))].astype(float)
        z = np.zeros(T, int)
        for t in range(1, T):
            z[t] = rng.choice(2, p=A[z[t - 1]])
        Y = np.zeros((T, len(n_cls)), int)
        for j, Wj in enumerate(W):
            for t in range(T):
                p = softmax(np.r_[0.0, Wj[z[t]] @ X[t]])
                Y[t, j] = rng.choice(len(p), p=p)
        Y[rng.random(T) < 0.03, 0] = -1                              # unobserved outcomes
        seqs.append((X, Y))
        zs.append(z)
    return seqs, zs, n_cls, A


def test_recovers_states_and_transitions():
    seqs, zs, n_cls, A = _simulate()
    m = gh.GLMHMM(2, n_cls=n_cls).fit(seqs, n_restarts=3)
    post = m.posteriors(seqs)
    z = np.concatenate(zs)
    g = np.vstack(post)
    acc = max((g.argmax(1) == z).mean(), (g.argmax(1) == 1 - z).mean())
    assert acc > 0.9, acc
    diag = np.sort(np.diag(m.A))
    assert np.all(diag > 0.9), m.A


def test_more_states_never_lose_training_likelihood_much():
    seqs, _zs, n_cls, _A = _simulate(seed=1, n_seq=6, T=200)
    l1 = gh.GLMHMM(1, n_cls=n_cls).fit(seqs, n_restarts=1).ll
    l2 = gh.GLMHMM(2, n_cls=n_cls).fit(seqs, n_restarts=3).ll
    assert l2 > l1


def test_shared_offsets_keep_states_common_across_animals():
    """Design v2: two animals with very different baseline rates (a SHARED offset column) and the same two states.
    With the offset shared across states, the states must be the common engaged / disengaged pair, not the animals."""
    rng = np.random.default_rng(3)
    n_cls, T = [3, 2], 300
    Wst = [np.array([[[-2.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]).repeat(2, axis=1), np.array([[[-1.0, 0.5, 0.0]],
                                                                                        [[1.0, 0.5, 0.0]]])]
    off = [np.array([[1.5], [1.5]]), np.array([[2.0]])]           # animal-2 shift per output (C-1 rows)
    A = np.array([[0.97, 0.03], [0.05, 0.95]])
    seqs, zs = [], []
    for s in range(12):
        animal = s % 2
        Xs = np.c_[np.ones(T), rng.integers(0, 2, (T, 2))].astype(float)
        X = np.c_[np.full(T, float(animal)), Xs]                     # column 0 = shared animal offset
        z = np.zeros(T, int)
        for t in range(1, T):
            z[t] = rng.choice(2, p=A[z[t - 1]])
        Y = np.zeros((T, 2), int)
        for j in range(2):
            for t in range(T):
                p = softmax(np.r_[0.0, Wst[j][z[t]] @ Xs[t] + off[j][:, 0] * animal])
                Y[t, j] = rng.choice(len(p), p=p)
        seqs.append((X, Y))
        zs.append(z)
    m = gh.GLMHMM(2, n_cls=n_cls, n_shared=1).fit(seqs, n_restarts=3)
    g = np.vstack(m.posteriors(seqs)).argmax(1)
    z = np.concatenate(zs)
    acc = max((g == z).mean(), (g == 1 - z).mean())
    assert acc > 0.85, acc
    assert m.B is not None and m.B[1][0, 0] > 1.0, m.B                 # the animal shift is learned as the shared term
