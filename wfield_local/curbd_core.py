"""CURBD core: a data-constrained multi-region RNN trained with FORCE / recursive least squares, and the Current-Based
Decomposition of its activity into region -> region currents.

ATTRIBUTION. Perich MG, Arlt C, Soares S, Young ME, Mosher CP, Minxha J, Carter E, Rutishauser U, Rudebeck PH,
Harvey CD, Rajan K (2020/2021), "Inferring brain-wide interactions using data-constrained recurrent neural network
models", bioRxiv 10.1101/2020.12.18.423348. Reference implementation: github.com/rajanlab/CURBD (`curbd.py`,
`trainMultiRegionRNN` / `computeCURBD`, written by M. G. Perich), which is licensed **GNU GPL-3.0**. This module is a
port of that algorithm (same update rule, parameter names and defaults), so it is a GPL-3.0 derivative: keep this
notice, and if this repository is ever distributed outside the lab, this file (or the repo) must carry GPL-3.0 terms.

DELIBERATE DEVIATIONS FROM THE REFERENCE (all documented, none change the algorithm):
  * no plotting / printing inside the loop; a seeded ``numpy.random.Generator`` instead of the global RNG.
  * RESETS (``reset_idx``, data-sample indices = segment starts) set the state to ``arctanh(data)`` so the rate
    equals the data at the reset; the reference sets H = data (the inverse nonlinearity it accepts is unused).
  * the error of each RLS update compares the rate with the data sample AT THE SAME TIME; the reference python port
    compares the rate at t = k*dtData with sample k-1 (its `iLearn` starts at 0 after one data step).
  * the learning step is skipped at the first sample after a reset (the state was just set to the data).
  * returns the per-pass pVar of the training passes AND of the free (no-learning) passes; the current decomposition
    uses the rates of the LAST free pass, as in the reference (its RNN array is that of the final run).

pVar (reference definition) = 1 - ||A - R||^2 / (N T var(A)) with A the scaled data and R the model rates sampled at
the data times.
"""
from __future__ import annotations

import time

import numpy as np


def scale_data(Y: np.ndarray, clip_sd: float = 4.0) -> tuple[np.ndarray, dict]:
    """(N, T) activity -> tanh range. Each unit is z-scored, clipped at +-clip_sd and divided by clip_sd, then capped at
    +-0.999 (the reference divides the whole matrix by its max and caps at 0.999; per-unit z-scoring keeps one high-
    variance unit from flattening the others)."""
    mu = Y.mean(1, keepdims=True)
    sd = Y.std(1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.clip((Y - mu) / sd, -clip_sd, clip_sd) / clip_sd
    return np.clip(Z, -0.999, 0.999), {"mu": mu.ravel(), "sd": sd.ravel(), "clip_sd": clip_sd}


def _noise(rng, n, nt, dt, tau_wn, amp):
    amp_wn = np.sqrt(tau_wn / dt)
    iwn = amp_wn * rng.standard_normal((n, nt))
    out = np.ones((n, nt))
    a = np.exp(-dt / tau_wn)
    for t in range(1, nt):
        out[:, t] = iwn[:, t] + (out[:, t - 1] - iwn[:, t]) * a
    return amp * out


def train(A: np.ndarray, dt_data: float, dt_factor: int = 4, g: float = 1.5, tau_rnn: float = 0.2,
          tau_wn: float = 0.1, amp_wn: float = 0.01, n_train: int = 30, n_free: int = 1, P0: float = 1.0,
          reset_idx=None, seed: int = 0, J_init: np.ndarray | None = None) -> dict:
    """Train J so the RNN rates follow ``A`` (N, T, already in (-1, 1), see `scale_data`).

    Dynamics (Euler, step dt = dt_data / dt_factor): tau dH/dt = -H + J tanh(H) + noise. At every data sample, RLS
    (Sussillo & Abbott 2009) updates every row of J with the shared inverse-correlation estimate P.
    ``J_init`` given = no training passes run on top unless n_train > 0 (used to free-run a trained J on other data).
    """
    rng = np.random.default_rng(seed)
    N, T = A.shape
    dt = dt_data / dt_factor
    resets = np.zeros(T, bool)
    resets[0] = True
    if reset_idx is not None:
        resets[np.asarray(reset_idx, int)] = True
    J = (g * rng.standard_normal((N, N)) / np.sqrt(N)) if J_init is None else np.array(J_init, float)
    J0 = J.copy()
    noise = _noise(rng, N, T * dt_factor, dt, tau_wn, amp_wn)
    P = P0 * np.eye(N)
    H0 = np.arctanh(A)                                   # reset states
    R = np.zeros((N, T))
    pv_train, pv_free = [], []
    var = A.var()
    leak = dt / tau_rnn
    t0 = time.perf_counter()
    for run in range(n_train + n_free):
        learn = run < n_train
        H = H0[:, 0].copy()
        for k in range(T):
            if resets[k]:
                H = H0[:, k].copy()
                r = np.tanh(H)
                R[:, k] = r
                continue
            base = (k - 1) * dt_factor
            for s in range(dt_factor):
                r = np.tanh(H)
                H += leak * (-H + J @ r + noise[:, base + s])
            r = np.tanh(H)
            R[:, k] = r
            if learn:
                err = r - A[:, k]
                kk = P @ r
                c = 1.0 / (1.0 + r @ kk)
                P -= c * np.outer(kk, kk)
                J -= c * np.outer(err, kk)
        pv = 1.0 - np.mean((A - R) ** 2) / var
        (pv_train if learn else pv_free).append(float(pv))
    return {"J": J, "J0": J0, "R": R, "pvar_train": pv_train, "pvar_free": pv_free, "seconds": time.perf_counter() - t0,
            "params": {"dt_data": dt_data, "dt_factor": dt_factor, "g": g, "tau_rnn": tau_rnn, "tau_wn": tau_wn,
                       "amp_wn": amp_wn, "n_train": n_train, "n_free": n_free, "P0": P0, "seed": seed}}


def free_run(J: np.ndarray, A: np.ndarray, dt_data: float, reset_idx=None, seed: int = 0, **kw) -> dict:
    """Run a trained J on data ``A`` without learning (held-out pVar: resets to the data at segment starts only)."""
    return train(A, dt_data, n_train=0, n_free=1, reset_idx=reset_idx, seed=seed, J_init=J, **kw)


def currents(J: np.ndarray, R: np.ndarray, region_of: np.ndarray, n_regions: int) -> dict:
    """CURBD: current into target region a from source region b = J[a_units, b_units] @ R[b_units] (N_a, T).
    Returns per (target, source): mean |current| (over target units and time), the signed mean, and the RMS."""
    mabs = np.zeros((n_regions, n_regions))
    msgn = np.zeros((n_regions, n_regions))
    rms = np.zeros((n_regions, n_regions))
    for a in range(n_regions):
        ia = region_of == a
        for b in range(n_regions):
            ib = region_of == b
            C = J[np.ix_(ia, ib)] @ R[ib]
            mabs[a, b] = np.abs(C).mean()
            msgn[a, b] = C.mean()
            rms[a, b] = np.sqrt((C ** 2).mean())
    return {"mean_abs": mabs, "mean_signed": msgn, "rms": rms}
