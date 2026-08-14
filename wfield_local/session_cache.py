"""Disk cache for expensive PER-SESSION analysis results (decode recall/EV, RDMs, crossnobis, hemisphere).

The cross-day figures (cross-mouse, within-animal, RSA) recompute per-session results for EVERY session on
every nightly run, and each per-session compute (LocaNMF load + block-CV decode/encode) is the slow pole.
This memoizes each per-session result to disk so only NEW or CHANGED sessions recompute; the rest load
instantly.

Cache key (per session, per `kind`): the mtime+size of the session's LocaNMF component file
(`{label}_locanmf_C.npy` — rewritten whenever LocaNMF is re-run, e.g. the PS93 8/5 recovered-positions
rerun), the DAQ h5, the optional `behavior_trials` override CSV, plus a repr of the analysis params and
CACHE_VERSION. Any of those changing → automatic invalidation.

IMPORTANT: mtimes do NOT capture changes to the COMPUTE CODE. **Bump CACHE_VERSION whenever you change the
logic of a cached function** (per_session / rdm / crossnobis / hemisphere), so stale results are discarded.

Bypass with env `WIDEFIELD_NO_CACHE=1` (always recompute, don't read/write). Relocate with
`WIDEFIELD_SESSION_CACHE=<dir>`. Clearing = delete the cache dir (safe; it just forces a recompute).
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

CACHE_VERSION = 6  # bump when any cached function's computation changes
# v6 (2026-08-14): locanmf.output_dir_name flipped to the meegkit_hpfit decomposition after the
# 52/52 refit. LocaNMF-SOURCE cached results were computed against the zerophase components --
# a different decomposition entirely (~15 fewer components per session), not just different
# numbers. mtimes do not see a config change, so without this they would be served unchanged.
# v5 (2026-08-14): analyses now read the ADOPTED meegkit_hpfit SVTcorr instead of the zerophase one
# (configs/defaults.yaml hemo.variant, resolved by config.svtcorr_path). Every cached decode/encode/RDM
# was computed on zerophase data. The cache keys on input mtimes, which do NOT see a config change, so
# without this bump the old zerophase-derived results would keep being served under the new methodology
# -- silently, and indistinguishably from a real re-run.
# v4 (2026-08-12): _crossnobis_rdm now estimates the noise covariance from the HELD-OUT folds (those not
# supplying either pattern in the cross-fold product) instead of from every trial, so the whitening matrix
# is independent of the data it whitens. Residuals are also taken within (fold, position) cells rather than
# within position, matching the quantity actually whitened. Cached "crossnobis" entries predate both.
# v3 (2026-08-12): _crossnobis_rdm now noise-whitens with the Ledoit-Wolf shrunk INVERSE COVARIANCE
# instead of per-feature variance alone. Every cached "crossnobis" entry was computed with the old
# diagonal whitening and is not comparable (mean sibling RSA: ROI +0.258 -> +0.817, LocaNMF +0.528 ->
# +0.767). No input mtime changed, so the key cannot see it -> forced invalidation.
# v2 (2026-08-11): behavior_position.BEH_ROOT was a hardcoded "M:/MICROSCOPE/..." (the analysis box's
# mount), so on any other machine the dead-spout_bit1 repair silently did not fire and the 8/5-8/6
# sessions kept 4-of-6 position labels (~1/3 of cues mislabelled). Now resolved via PathResolver.
# Position labels feed every cached quantity (decode, encode, RDM, crossnobis, hemisphere), and the
# cache key cannot see it (no input mtime changed), so any cache written on an affected machine holds
# wrong values -> forced invalidation.

def _default_cache_dir() -> Path:
    """Cache location for THIS machine.

    The old default was the analysis box's "C:/Users/sabatini/source/..." path, which on any other
    machine silently created a cache under a nonexistent user instead of reusing the real one -- the
    same hardcoded-other-box-path bug that broke behavior_position's BEH_ROOT and nightly_figs'
    DEFAULT_OUT. Derive it from the machine's own working figure root when one exists."""
    env = os.environ.get("WIDEFIELD_SESSION_CACHE")
    if env:
        return Path(env)
    try:
        from wfield_local import config
        return Path(config.resolver().root("figures_working")).parent / ".widefield_session_cache"
    except Exception:                              # noqa: BLE001 - config may be unavailable in tests
        return Path("C:/Users/sabatini/source/.widefield_session_cache")


CACHE_DIR = _default_cache_dir()


def _disabled() -> bool:
    return bool(os.environ.get("WIDEFIELD_NO_CACHE"))


def _stat_sig(path) -> str:
    if not path:
        return "-"
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "MISSING"


def _params_str(params) -> str:
    if params is None:
        return ""
    if hasattr(params, "__dict__"):            # argparse.Namespace / SimpleNamespace
        return repr(sorted(vars(params).items()))
    return str(params)


def session_signature(session, params=None) -> str:
    """Hash of the inputs that determine a per-session result: LocaNMF C.npy + h5 + behavior_trials
    mtimes/sizes, the params, and CACHE_VERSION.

    The LocaNMF directory NAMES the hemodynamic variant it was fitted to, so switching variants
    changes this path and therefore the signature -- cached results from one drift-removal cannot be
    served for another. That is the intended behaviour, not a coincidence.
    """
    # imported lazily, like _default_cache_dir above: config may be unavailable in unit tests
    from wfield_local import config

    mc, lab = session["mc"], session["label"]
    parts = [
        f"v{CACHE_VERSION}", lab, _params_str(params),
        _stat_sig(f"{config.locanmf_dir(mc)}/{lab}_locanmf_C.npy"),
        _stat_sig(session.get("h5", "")),
        _stat_sig(session.get("behavior_trials", "")),
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def cached(session, kind, compute, params=None, verbose=True):
    """Return `compute()` for (session, kind), memoized to disk under a signature of the session's inputs.
    `compute` is a zero-arg callable producing a picklable result."""
    if _disabled():
        return compute()
    lab = session["label"]
    sig = session_signature(session, params)
    fp = CACHE_DIR / f"{lab}__{kind}__{sig}.pkl"
    if fp.exists():
        try:
            with open(fp, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass  # corrupt/partial -> recompute below
    res = compute()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(res, fh)
    os.replace(tmp, fp)  # atomic publish
    for old in CACHE_DIR.glob(f"{lab}__{kind}__*.pkl"):  # prune superseded signatures
        if old != fp:
            try:
                old.unlink()
            except OSError:
                pass
    if verbose:
        print(f"  [cache] computed {lab}/{kind}", flush=True)
    return res
