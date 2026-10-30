"""Find and repair OTHER-LED frames in a motion-corrected (T, 2, H, W) movie BEFORE the SVD (2026-10-30).

WHY. Some frames carry the other LED's light, though the DAQ labels their exposure correctly
(docs/COHORT2_QC_2026-10-29.md §0b-0e): whole frames, or a BAND OF ROWS (rolling-shutter signature). 25-93 per channel
per session on cohort2 10-29, 10-170 on every cohort1 session. In RiboL1, where 470 is ~2x brighter than 415, each
one is a +-50-110% single-frame spike: it collapsed the hemo coefficient (0.1-0.4 instead of ~1.5) and loaded up to
half the energy of some SVD components. Interpolating them in SVT afterwards fixes the temporal side, but the SVD
basis is built from frames that include them. So they are repaired in the movie itself, before `approximate_svd`.

DETECTION (one streaming pass; frames block-averaged BLOCK x BLOCK). For each pair t and slot c:
  own   = median of the same slot over t-W..t+W (t excluded)      -> what this frame should look like
  other = median of the other slot over t-2..t+2                   -> what the other LED's light looks like now
  d = x / own - 1,  P = other / own - 1                            (per block-pixel, brain only)
In each of NB row bands, least squares d ~ alpha + beta * P over the band's pixels:
  normal frame:  alpha ~ 0, beta ~ 0
  swapped band:  alpha ~ 0, beta ~ 1       (the band holds the other LED's light INSTEAD of its own)
  dual exposure: alpha ~ 1, beta ~ 1       (both LEDs' light ADDED; rare)
P is not flat even when the two LEDs are equally bright (vessels vs parenchyma, two illumination fields), so beta is
measurable in GCaMP6s sessions too. A band is BAD when beta > BETA_MIN and beta is > Z robust SDs from that band's
session median. A frame is bad if any band is.

REPAIR (`RepairedDat`, a read-only view; the .bin on disk is never touched):
  * both slots of a pair are FULL swaps (>= 90% of bands bad, alpha ~ 0)  -> the two slots are exchanged;
  * any other bad frame (single swap, partial / rolling-shutter band, dual exposure) -> replaced by the mean of its
    nearest good same-slot neighbours. It is DROPPED in effect, but the frame grid stays regular, so the frame map,
    event timing and the SVD's shape are unchanged.
`frames_average` is corrected by the same edits (exact: old mean + sum(repaired - original) / T).

The view has no ``filename`` attribute on purpose: wfield `approximate_svd` then reads it through ``dat[a:b]`` slices
(both the binning and the projection pass) instead of re-opening the raw file.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

BLOCK = 8
NB = 16
WIN = 5
BETA_MIN = 0.08            # PS122 has 5-18% admixtures (beta ~0.1, ~10 robust SD) that still shift a frame ~3.5%
Z = 8.0
STRONG = 0.5               # strongest-band beta above this: an unambiguous swap, no isolation test
ISOLATE = 0.4              # weak candidate kept only if both neighbours' beta < ISOLATE x its own
CHUNK = 2000


def _ds(x: np.ndarray) -> np.ndarray:
    """(n, 2, H, W) -> (n, 2, H // BLOCK, W // BLOCK) block means, float32."""
    n, c, H, W = x.shape
    h, w = H // BLOCK, W // BLOCK
    return x[:, :, :h * BLOCK, :w * BLOCK].astype(np.float32).reshape(n, c, h, BLOCK, w, BLOCK).mean((3, 5))


def _brain_ds(frames_average: np.ndarray, func: int) -> np.ndarray:
    im = _ds(np.asarray(frames_average, np.float32)[None])[0, func]
    dark = min(im[:2, :2].mean(), im[:2, -2:].mean(), im[-2:, :2].mean(), im[-2:, -2:].mean())
    return im > dark + 0.2 * (np.percentile(im, 99.5) - dark)


def _band_fit(d, P, brain, edges):
    """Vectorised least squares d ~ alpha + beta * P per row band over brain pixels.

    d, P: (n, h, w). Returns (n, NB, 2) = (alpha, beta); NaN where the band has < 8 brain pixels or no P spread."""
    n = d.shape[0]
    out = np.full((n, NB, 2), np.nan, np.float32)
    for b in range(NB):
        m = brain[edges[b]:edges[b + 1]]
        N = int(m.sum())
        if N < 8:
            continue
        y = d[:, edges[b]:edges[b + 1]][:, m].astype(np.float64)      # (n, N)
        x = P[:, edges[b]:edges[b + 1]][:, m].astype(np.float64)
        sx, sy = x.sum(1), y.sum(1)
        sxx, sxy = (x * x).sum(1), (x * y).sum(1)
        den = N * sxx - sx * sx
        ok = den > 1e-9 * N * N
        beta = np.where(ok, (N * sxy - sx * sy) / np.where(ok, den, 1), np.nan)
        alpha = (sy - beta * sx) / N
        out[:, b, 0], out[:, b, 1] = alpha, beta
    return out


def detect(dat, frames_average, func: int = 1, verbose: bool = True) -> dict:
    """Stream the movie once; return per pair x slot x band (alpha, beta) and the bad-frame classification."""
    T = len(dat)
    fa_ds = _ds(np.asarray(frames_average, np.float32)[None])[0]
    brain = _brain_ds(frames_average, func)
    edges = np.linspace(0, brain.shape[0], NB + 1).astype(int)
    coef = np.full((T, 2, NB, 2), np.nan, np.float32)
    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        lo, hi = max(0, s - WIN), min(T, e + WIN)
        X = np.maximum(_ds(np.asarray(dat[lo:hi])), 1.0)              # (n, 2, h, w)
        n = len(X)
        pad = np.pad(X, ((WIN, WIN), (0, 0), (0, 0), (0, 0)), mode="edge")
        own = np.median(np.stack([pad[WIN + k:WIN + k + n] for k in range(-WIN, WIN + 1) if k != 0]), 0)
        oth = np.median(np.stack([pad[WIN + k:WIN + k + n] for k in range(-2, 3)]), 0)[:, ::-1]
        i0, i1 = s - lo, e - lo
        for c in (0, 1):
            d = X[i0:i1, c] / own[i0:i1, c] - 1
            P = oth[i0:i1, c] / own[i0:i1, c] - 1
            coef[s:e, c] = _band_fit(d, P, brain, edges)
        if verbose and (e == T or (s // CHUNK) % 10 == 0):
            print(f"[swap_repair] detect {e}/{T} pairs", flush=True)
    beta, alpha = coef[..., 1], coef[..., 0]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN bands (no brain pixels)
        med = np.nanmedian(beta, 0)
        rsd = np.nanmedian(np.abs(beta - med), 0) * 1.4826 + 1e-6
    with np.errstate(invalid="ignore"):
        bad_band = (beta > BETA_MIN) & ((beta - med) / rsd > Z)
    bad = bad_band.any(2)                                   # (T, 2)
    # A swap is ONE exposure. A WEAK candidate (strongest band beta < STRONG) is kept only if it is isolated in time:
    # the same slot's neighbours t-1 / t+1 must sit well below it. PS122 10-29 showed weak (beta ~.05-.1) two-frame
    # events with a small global rise -- physiology / hemodynamics, not admixture -- which this rejects.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bmed = np.nanmax(beta, 2)                           # (T, 2): strongest band (partial swaps are banded)
    nb = np.maximum(np.r_[bmed[:1] * 0, bmed[:-1]], np.r_[bmed[1:], bmed[:1] * 0])
    weak = bad & (bmed < STRONG)
    bad &= ~(weak & ~(nb < ISOLATE * bmed))
    bad_band &= bad[..., None]
    frac = bad_band.sum(2) / np.maximum(np.isfinite(beta).sum(2), 1)   # over bands that have brain pixels
    full = bad & (frac >= 0.9) & (np.nanmedian(np.abs(alpha), 2) < 0.25)
    dual = bad & (np.nanmedian(alpha, 2) > 0.5)
    exchange = full[:, 0] & full[:, 1]
    return dict(alpha=alpha, beta=beta, bad_band=bad_band, bad=bad, full=full, dual=dual, exchange=exchange,
                fa_ds=fa_ds, brain_ds=brain, func=np.array(func))


class RepairedDat:
    """Read-only (T, 2, H, W) view of ``dat`` with the detected frames repaired. Deliberately has no ``filename``."""

    def __init__(self, dat, plan: dict):
        self._dat = dat
        self.shape = tuple(dat.shape)
        self.dtype = dat.dtype
        self.ndim = len(self.shape)
        self._exchange = np.flatnonzero(plan["exchange"])
        bad = plan["bad"].copy()
        bad[plan["exchange"]] = False                       # exchanged pairs are fixed, not interpolated
        self._bad = bad
        self._ex_set = set(self._exchange.tolist())
        self._badpairs = np.flatnonzero(bad.any(1))
        self._good = [np.flatnonzero(~bad[:, c]) for c in (0, 1)]

    def __len__(self):
        return self.shape[0]

    def _replacement(self, t: int, c: int) -> np.ndarray:
        g = self._good[c]
        k = np.searchsorted(g, t)
        nbr = [g[j] for j in (k - 1, k) if 0 <= j < len(g)]
        return np.mean([np.asarray(self._dat[j, c], np.float32) for j in nbr], 0)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self[int(key):int(key) + 1][0]
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("RepairedDat supports contiguous slices only (what approximate_svd uses)")
        start, stop, _ = key.indices(self.shape[0])
        blk = np.array(self._dat[start:stop], copy=True)
        for t in self._exchange[(self._exchange >= start) & (self._exchange < stop)]:
            blk[t - start] = blk[t - start, ::-1].copy()
        for t in self._badpairs[(self._badpairs >= start) & (self._badpairs < stop)]:
            for c in (0, 1):
                if self._bad[t, c]:
                    blk[t - start, c] = np.round(self._replacement(int(t), c)).astype(blk.dtype)
        return blk


def corrected_average(dat, rep: RepairedDat, frames_average: np.ndarray) -> np.ndarray:
    """frames_average after the repairs, exactly: old + sum(repaired - original) / T over the edited pairs only."""
    fa = np.asarray(frames_average, np.float64).copy()
    T = len(dat)
    for t in sorted(set(rep._exchange.tolist()) | set(rep._badpairs.tolist())):
        fa += (np.asarray(rep[t], np.float64) - np.asarray(dat[t], np.float64)) / T
    return fa


def summary(plan: dict) -> dict:
    bad, full, dual, ex = plan["bad"], plan["full"], plan["dual"], plan["exchange"]
    T = bad.shape[0]
    return {"pairs": int(T), "bad_415_slot": int(bad[:, 0].sum()), "bad_470_slot": int(bad[:, 1].sum()),
            "full_swaps": int(full.sum()), "partial": int((bad & ~full & ~dual).sum()), "dual_exposure": int(dual.sum()),
            "pairs_exchanged": int(ex.sum()), "frames_interpolated": int((bad & ~ex[:, None]).sum()),
            "pct_frames_repaired": float(100 * bad.sum() / (2 * T)),
            "params": {"block": BLOCK, "bands": NB, "win": WIN, "beta_min": BETA_MIN, "z": Z}}


def save(plan: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    s = summary(plan)
    np.savez_compressed(out_dir / "swap_repair.npz", **{k: v for k, v in plan.items() if isinstance(v, np.ndarray)})
    (out_dir / "swap_repair.json").write_text(json.dumps(s, indent=2))
    return s
