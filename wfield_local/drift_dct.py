"""Block-protected slow cosine drift removal (drift-cutoff check, 2026-11-03).

The adopted drift step (`hemo_variants` `meegkit_hpfit`) fits an order-10 robust polynomial on out-of-trial frames
(strobe-0.25 s -> cue+4 s masked out); its 50% cutoff is ~43 min. This module is the candidate REPLACEMENT for that
step (Priya 11-03): on the SAME out-of-trial mask, regress each temporal trace on

    [orthonormal DCT-II cosines with period >= min_period_s]  +  [one indicator per position block]

and subtract ONLY the cosine part (DC excluded). The block indicators take up each block's mean, so a block-level
(spout position) offset cannot be absorbed by the drift term -- the constraint "do NOT remove block-level position
information". `protect=False` gives the unprotected contrast (cosines only).

A BLOCK here is a run of consecutive trials at the same spout position (the DAQ position code from
`classify_cues_with_backup`, the decoder's own source). Block k spans from its first trial's strobe - 0.25 s (the
trial mask's leading edge) to the next block's start; the last block ends at its last cue + 4 s (the mask's trailing
edge). Frames before the first block / after the last are the reference level (no indicator), which keeps the DC
cosine identifiable.

The fit is least squares with shared frame weights: OLS, then `robust_iter` passes that drop mask frames whose
residual energy (summed over traces) is an outlier (> median + 3 x 1.4826 MAD of its square root). This is an
APPROXIMATION of meegkit's per-sample IRLS, chosen so one solve serves every component.

MEASURED 2026-11-03 (PS92_0820): the per-BLOCK protection is ILL-POSED. With one free offset per ~1.5-min block, a
slow cosine is nearly collinear with the block indicators and is identified only by within-block slopes (ITI frames);
the fit extrapolates those slopes into large slow cosines (10-43 min power x2-6 AFTER "removal", decoding to chance).
`protect="position"` (one indicator per spout POSITION = union of its blocks; 6 regressors) protects the between-
position means -- the information the constraint is about -- while leaving the cosines identified across blocks.

Because the cosines are exact orthonormal DCT-II vectors, the subtraction changes ONLY DCT coefficients 1..kmax
(periods >= min_period_s): band power faster than the cutoff is untouched by construction.
"""
from __future__ import annotations

import numpy as np

TRIAL_PRE_S, TRIAL_POST_S = 0.25, 4.0           # = filter_acausality_test.MASK_SPEC["strobedetrend"] + TRIAL_POST_S


def kmax_for(n: int, fs: float, min_period_s: float) -> int:
    """Highest DCT index whose period 2n/(k fs) is >= min_period_s."""
    return int(np.floor(2.0 * n / (float(min_period_s) * fs)))


def dct_periods_s(n: int, fs: float) -> np.ndarray:
    """Period (s) of each orthonormal DCT-II coefficient k = 0..n-1 (inf for k = 0)."""
    k = np.arange(n, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(k > 0, 2.0 * n / (np.maximum(k, 1) * fs), np.inf)


def dct_basis(n: int, fs: float, min_period_s: float) -> np.ndarray:
    """(n, kmax+1) orthonormal DCT-II vectors k = 0..kmax (column 0 = DC); scipy.fft.dct(norm='ortho') convention."""
    k = np.arange(kmax_for(n, fs, min_period_s) + 1)
    t = (np.arange(n) + 0.5) / n
    B = np.cos(np.pi * np.outer(t, k)) * np.sqrt(2.0 / n)
    B[:, 0] = 1.0 / np.sqrt(n)
    return B


def block_segments(cue: dict, codes, csmp, n_frames: int, fs: float) -> list[dict]:
    """Position blocks in FRAME units (see module docstring). `cue` = plot_spout_trial_averages._load_daq_events,
    `codes` = classify_cues_with_backup(session, cue) (per cue; < 0 = unusable), `csmp` = DAQ sample per frame."""
    cs = np.asarray(cue["cue_samples"], float)
    ss = np.asarray(cue["strobe_samples"], float)
    sr = float(cue["sample_rate_hz"])
    codes = np.asarray(codes)
    j = np.searchsorted(ss, cs, side="right") - 1
    st = np.where(j >= 0, ss[np.clip(j, 0, max(len(ss) - 1, 0))], cs - 3.0 * sr)
    st = np.where(cs - st > 60 * sr, cs - 3.0 * sr, st)          # no strobe within 60 s: assume the median 3 s lead
    csmp = np.asarray(csmp, float)
    inside = (cs >= csmp[0]) & (cs <= csmp[-1])
    ok = inside & (codes >= 0)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    f_start = np.searchsorted(csmp, st[idx]) - int(round(TRIAL_PRE_S * fs))
    f_cue = np.searchsorted(csmp, cs[idx])
    c = codes[idx]
    runs, i = [], 0
    while i < c.size:
        k = i
        while k + 1 < c.size and c[k + 1] == c[i]:
            k += 1
        runs.append((i, k))
        i = k + 1
    segs = []
    for r, (i, k) in enumerate(runs):
        a = max(0, int(f_start[i]))
        b = int(f_start[runs[r + 1][0]]) if r + 1 < len(runs) else int(f_cue[k] + round(TRIAL_POST_S * fs))
        b = min(n_frames, max(b, a + 1))
        segs.append(dict(start=a, stop=b, code=int(c[i]), n_trials=k - i + 1, first_cue=int(idx[i]),
                         last_cue=int(idx[k])))
    return segs


def block_matrix(segs, n_frames: int, rows=None, by: str = "block") -> np.ndarray:
    """0/1 indicators, one column per block (`by="block"`) or per position code (`by="position"`, union of that
    position's blocks); `rows` = frame indices to evaluate (default all frames)."""
    rows = np.arange(n_frames) if rows is None else np.asarray(rows)
    keys = list(range(len(segs))) if by == "block" else sorted({sg["code"] for sg in segs})
    M = np.zeros((rows.size, len(keys)))
    for j, sg in enumerate(segs):
        lo, hi = np.searchsorted(rows, [sg["start"], sg["stop"]])
        M[lo:hi, j if by == "block" else keys.index(sg["code"])] = 1.0
    return M


def fit_cosine_drift(X: np.ndarray, mask: np.ndarray, fs: float, min_period_s: float, segs=None,
                     protect="block", robust_iter: int = 2, min_block_frames: int = 10):
    """Fit X (K, T) on the mask frames. Returns (beta_cos (K, kmax+1) incl. DC, B (T, kmax+1), info).
    `protect`: "block" (True) | "position" | None/False (cosines only)."""
    protect = "block" if protect is True else (protect or None)
    X = np.asarray(X, float)
    K, T = X.shape
    m = np.asarray(mask, bool)
    m = m[:T] if m.size >= T else np.pad(m, (0, T - m.size))
    rows = np.flatnonzero(m)
    B = dct_basis(T, fs, min_period_s)
    nc = B.shape[1]
    D = B[rows]
    n_blk = 0
    if protect and segs:
        Bl = block_matrix(segs, T, rows, by=protect)
        keep = Bl.sum(0) >= min_block_frames
        D = np.hstack([D, Bl[:, keep]])
        n_blk = int(keep.sum())
    Y = X[:, rows].T                                              # (Nm, K)
    w = np.ones(rows.size, bool)
    for it in range(robust_iter + 1):
        beta = np.linalg.lstsq(D[w], Y[w], rcond=None)[0]          # (p, K)
        if it == robust_iter:
            break
        e = np.sqrt(((Y - D @ beta) ** 2).sum(1))
        med = np.median(e[w])
        mad = 1.4826 * np.median(np.abs(e[w] - med))
        w = e <= med + 3.0 * max(mad, 1e-12)
    info = dict(n_cos=nc, n_blocks_fit=n_blk, n_blocks=len(segs or []), mask_frames=int(rows.size),
                frac_kept_robust=float(w.mean()), kmax=nc - 1)
    return beta[:nc].T, B, info


def remove(X, mask, fs, min_period_s, segs=None, protect="block", robust_iter=2):
    """X minus its fitted cosine drift (DC excluded). Same shape/dtype handling as hemo_variants.remove_drift."""
    beta, B, _ = fit_cosine_drift(X, mask, fs, min_period_s, segs, protect, robust_iter)
    return np.asarray(X, float) - beta[:, 1:] @ B[:, 1:].T
