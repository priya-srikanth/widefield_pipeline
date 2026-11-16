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
