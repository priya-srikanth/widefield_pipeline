"""LP_licks: video (Lightning Pose) licks as the imaging pipeline's lick source (Priya, 2026-10-21: "this will be a
separate family - LP_licks ... for lick-window analyses, let's make versions gated on LP lick post-cue").

WHY. Every imaging analysis that aligns to a lick or gates on one took the lick from the DAQ lick sensor, i.e. SPOUT
CONTACT. Post-stroke most far-contra attempts never touch the spout (acute far contra: 0-12 contact trials per session
against 21-78 attempted), so the lick-aligned far-contra cell was EMPTY in every animal and the "no-lick" arm held
trials on which the animal licked. Pre-stroke the two definitions nearly coincide (LP attempt without contact <= 3 % of
trials in every animal x position except PS93 far_ipsi 27 % / far_mid 14% -- its orofacial deficit; LP onset = tongue
out, 10-60 ms before contact).

DEFINITIONS (Priya's choices, 10-21):
  LP lick        cam4 LP lick onsets (`licks_all.csv`, every context, DAQ s) + cam1-only events (`cam1_licks/<a>_<d>
                 _events.csv`: not cam4-matched, not water droplets) -- "combined", the same set `combined_encoding`
                 uses under COMBINED_ENCODING_LICKS=combined. cam1 missing (PS95 pre-stroke) -> cam4 alone.
  post-cue gate  the trial's attempt class (`scripts/trial_attempts.py`) in ATTEMPT_CLASSES = hit / incomplete, with
                 the first LP lick (trial_attempts `combined`, else `cam4`; ms from the cue) inside max_rt. jaw_only is
                 a reaction, not an attempt (blind review 10-21). The attempt class is the authority: no_try /
                 jaw_only / unknown are NO-LICK trials. `unknown` = no tongue lick but the jaw could not be scored --
                 1176 of its 1181 trials are close_center, where the jaw is masked, and none has a counted lick
                 (checked 10-21); dropping them would delete close_center's no-lick arm. Only trials with NO attempts
                 row (no video / untracked / unmatched cue) are DROPPED (marked out of coverage).
  alignment      lick-aligned windows start at that first LP lick.

HOW IT ENTERS. `_trial_features` (and `nolick_decoder.categorize`) read `lick_source()`; under "lp" the DAQ lick
sample train is replaced by `session_lp(...)["lick_samples"]`, edited so that the first lick after each cue IS the LP
gate: on an attempted trial every LP lick in [cue, first LP lick) is removed and the first LP lick is inserted if the
tables lack it; on a non-attempted trial every LP lick in [cue, cue + max_rt] is removed. Licks elsewhere (pre-cue,
later in the trial, ITI) are the raw LP set. Trials are matched to DAQ cues by CUE TIME (LP `cue_daq` vs DAQ cue
sample / rate: all trials within 20 ms, median ~1 ms on the three sessions checked 10-21), never by trial id -- ids
are offset in some sessions.

The DAQ path is untouched when the switch is unset. Results go to separate LP_licks outputs and cache keys
(`spec_token`), so LP and DAQ numbers can never be served for each other.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

#: Selects the imaging lick source: "daq" (default, spout contact) or "lp" (this module).
LICK_SOURCE_ENV = "WIDEFIELD_LICK_SOURCE"
LICK_SOURCES = ("daq", "lp")
ATTEMPT_CLASSES = ("hit", "incomplete")
#: LP cue <-> DAQ cue match tolerance (s).
MATCH_TOL_S = 0.02
#: Bump when the definition above changes (enters every LP cache key).
LP_VERSION = 1


class LPUnavailable(RuntimeError):
    """The session has no LP lick tables (no video, or not tracked): it has no LP_licks version."""


def lick_source() -> str:
    v = os.environ.get(LICK_SOURCE_ENV, "daq").strip().lower() or "daq"
    if v not in LICK_SOURCES:
        raise ValueError(f"{LICK_SOURCE_ENV}={v!r}; expected one of {LICK_SOURCES}")
    return v


def is_lp() -> bool:
    return lick_source() == "lp"


def _run_dir(rv=None) -> Path:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    return oo.root(rv or PathResolver()) / oc.RUN


def attempts_path(rv=None) -> Path:
    return _run_dir(rv) / "imaging_encoding" / "qc" / "trial_attempts.csv"


def spec_token(rv=None) -> str:
    """Cache-key token for the LP source: the definition version + the attempts table's identity (a rewrite of
    trial_attempts.csv -- the jaw_onsets reruns -- invalidates every LP entry)."""
    p = attempts_path(rv)
    st = p.stat() if p.exists() else None
    return f"lp{LP_VERSION}:{st.st_mtime_ns if st else 0}:{st.st_size if st else 0}"


def output_dir(default: Path) -> Path:
    """Renderer output under the LP source: <parent>/LP_licks/<name> (grant_figures/epoch -> grant_figures/LP_licks/
    epoch; grant_figures -> grant_figures/LP_licks). Unchanged under DAQ."""
    default = Path(default)
    if not is_lp():
        return default
    if default.name == "grant_figures":
        return default / "LP_licks"
    return default.parent / "LP_licks" / default.name


STAMP = "LP licks: video lick (cam4 + cam1) attempts, gate = hit / incomplete within max_rt; lick-aligned = LP onset"


def stamp(fig) -> None:
    """Corner label on every figure rendered under the LP source, so a copied PNG still says what it is."""
    if is_lp():
        fig.text(0.995, 0.002, STAMP, ha="right", va="bottom", fontsize=6, color="#8a2be2")


_H5_LABEL: dict = {}


def label_for_h5(h5_path) -> str:
    """Session label of a DAQ .h5 (the shared lick loader is called with the path only)."""
    from wfield_local import config
    key = str(Path(h5_path).resolve()).lower()
    if key not in _H5_LABEL:
        for s in config.load_sessions():
            if s.get("h5"):
                _H5_LABEL[str(Path(s["h5"]).resolve()).lower()] = s["label"]
    if key not in _H5_LABEL:
        raise LPUnavailable(f"no registered session for {h5_path}")
    return _H5_LABEL[key]


def cache_tag(rv=None) -> str:
    """Short suffix `session_cache.cached` appends to every kind under the LP source."""
    import hashlib
    return "lp" + hashlib.sha1(spec_token(rv).encode()).hexdigest()[:8]


_ATTEMPTS: dict = {}


def _attempts(rv=None) -> pd.DataFrame:
    p = attempts_path(rv)
    key = (str(p), p.stat().st_mtime_ns if p.exists() else 0)
    if key not in _ATTEMPTS:
        if not p.exists():
            raise LPUnavailable(f"no attempts table {p}")
        _ATTEMPTS.clear()
        _ATTEMPTS[key] = pd.read_csv(p, usecols=["label", "trial_id", "cls", "cam4", "combined"])
    return _ATTEMPTS[key]


def lp_lick_times(label: str, rv=None) -> tuple[np.ndarray, pd.DataFrame]:
    """(all LP lick onsets in DAQ s, sorted; per-trial cue table [trial_id -> cue_daq]) for one session."""
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = rv or PathResolver()
    an, d = label.split("_")
    sd = oo.session_dir(rv, oc.RUN, an, "2026" + d)
    f4 = sd / "licks_all.csv"
    if not (sd / "windows.npz").exists() or not f4.exists():
        raise LPUnavailable(f"{label}: no LP lick tables in {sd}")
    cue_t = oc._cue_table(sd).set_index("trial_id")
    t = pd.read_csv(f4, usecols=["daq_on_s"]).daq_on_s.to_numpy(float)
    parts = [t[np.isfinite(t)]]
    f1 = Path(rv.root("dlc")) / "session_poses" / "cam1_licks" / f"{an}_2026{d}_events.csv"
    if f1.exists():
        E = pd.read_csv(f1)
        only = ~E.cam4_matched.fillna(False).astype(bool) & ~E.droplet.fillna(False).astype(bool)
        E = E[only & E.trial_id.isin(cue_t.index)]
        t1 = E.trial_id.map(cue_t.cue_daq).to_numpy(float) + E.on_ms.to_numpy(float) / 1000.0
        parts.append(t1[np.isfinite(t1)])
    return np.sort(np.concatenate(parts)), cue_t


def edit_train(licks, cue_s, drop, attempted, first_s, max_rt) -> np.ndarray:
    """The LP lick train (s, sorted) edited so the first lick after each kept cue IS the LP post-cue gate.

    Attempted with the first LP lick inside max_rt: LP licks in [cue, first) removed, the first inserted if missing.
    Otherwise (not attempted, or first after max_rt): LP licks in [cue, cue + max_rt] removed -- a later first lick
    survives, so `nolick_decoder`'s late_rewarded arm still sees it. Dropped cues and every other lick untouched."""
    licks = np.asarray(licks, float)
    keep = np.ones(len(licks), bool)
    add = []
    for kk in np.flatnonzero(~np.asarray(drop, bool)):
        c0 = cue_s[kk]
        if attempted[kk] and first_s[kk] <= max_rt:
            t0 = c0 + first_s[kk]
            keep &= ~((licks >= c0) & (licks < t0 - 1e-4))
            if not np.any(np.abs(licks - t0) <= 1e-3):
                add.append(t0)
        else:
            keep &= ~((licks >= c0) & (licks <= c0 + max_rt))
    return np.sort(np.concatenate([licks[keep], np.asarray(add, float)]))


def session_lp(label: str, cue_samples, sr: float, max_rt: float, rv=None) -> dict:
    """The LP lick train for `_trial_features`, on the DAQ sample clock.

    Returns ``lick_samples`` (int, sorted), ``drop`` (bool per DAQ cue: no attempts row / unknown class / unmatched ->
    exclude the trial; an `unknown` class is a no-lick trial, not a drop), ``attempted`` (bool per DAQ cue), ``first_s`` (first LP lick minus cue, s; NaN if none) and
    counts for the log."""
    cue_s = np.asarray(cue_samples, float) / float(sr)
    licks, cue_t = lp_lick_times(label, rv)
    A = _attempts(rv)
    A = A[A.label == label]
    if A.empty:
        raise LPUnavailable(f"{label}: not in {attempts_path(rv).name}")
    A = A.set_index("trial_id")
    A = A[A.index.isin(cue_t.index)]
    tcue = cue_t.loc[A.index, "cue_daq"].to_numpy(float)
    # each attempts trial -> its DAQ cue, by time
    j = np.clip(np.searchsorted(cue_s, tcue), 1, max(len(cue_s) - 1, 1))
    cand = np.stack([j - 1, j])
    dt = np.abs(cue_s[cand] - tcue)
    k = cand[np.argmin(dt, 0), np.arange(len(tcue))]
    ok = dt.min(0) <= MATCH_TOL_S
    n = len(cue_s)
    drop = np.ones(n, bool)
    attempted = np.zeros(n, bool)
    first_s = np.full(n, np.nan)
    lat = A.combined.fillna(A.cam4).to_numpy(float) / 1000.0
    cls = A.cls.astype(str).to_numpy()
    for kk, good, c, la in zip(k, ok, cls, lat):
        if not good:
            continue
        drop[kk] = False
        if c in ATTEMPT_CLASSES and np.isfinite(la):
            attempted[kk] = True
            first_s[kk] = la
    t_all = edit_train(licks, cue_s, drop, attempted, first_s, max_rt)
    return {"lick_samples": np.round(t_all * sr).astype(np.int64), "drop": drop, "attempted": attempted,
            "first_s": first_s, "n_cues": n, "n_matched": int((~drop).sum()),
            "n_attempted": int((attempted & (first_s <= max_rt)).sum())}
