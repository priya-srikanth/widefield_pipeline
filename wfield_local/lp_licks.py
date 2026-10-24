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
                 uses under COMBINED_ENCODING_LICKS=combined. cam1 missing -> cam4 alone. CAM1 IS NOT FINISHED
                 (10-21: events for 74 / 108 curated sessions; PS95 none, PS94 chronic 2 / 8, PS93 pre 7 / 8): those
                 sessions are cam4-only LP until cam1 lands -- revisit then; the cache tag includes the cam1 folder,
                 so new cam1 files recompute every LP entry.
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


_CAM1_SIG: dict = {}


def _cam1_signature(rv=None) -> str:
    """Digest of the cam1 lick-event folder (names, sizes, mtimes). CAM1 IS NOT FINISHED (Priya 10-21: "not all cam1
    analysis is done so we'll have to revisit"): 74 / 108 curated sessions have events on 10-21 (PS95 none, PS94 chronic
    2 / 8), and a session without them falls back to cam4. When cam1 lands, every LP entry must recompute -- the
    per-session sidecar sees its own file, but the session_cache tag would not without this. Memoised per process (one
    listing of the share per render)."""
    from wfield_local.paths import PathResolver
    import hashlib
    d = Path((rv or PathResolver()).root("dlc")) / "session_poses" / "cam1_licks"
    if str(d) not in _CAM1_SIG:
        h = hashlib.sha1()
        for f in sorted(d.glob("*_events.csv")) if d.exists() else []:
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns};".encode())
        _CAM1_SIG[str(d)] = h.hexdigest()[:10]
    return _CAM1_SIG[str(d)]


def spec_token(rv=None) -> str:
    """Cache-key token for the LP source: the definition version + the attempts table's identity (a rewrite of
    trial_attempts.csv -- the jaw_onsets reruns -- invalidates every LP entry) + the cam1 event folder's identity."""
    p = attempts_path(rv)
    st = p.stat() if p.exists() else None
    return f"lp{LP_VERSION}:{st.st_mtime_ns if st else 0}:{st.st_size if st else 0}:cam1-{_cam1_signature(rv)}"


#: Pre-cue (ENL) lick-free window length, s (Priya 10-21, after `scripts/enl_lp_census.py`: "1.0 s, plus DAQ 1.0 s").
#: With LP licks a 2 s lick-free window keeps only 48 % of PS93's pre-stroke far_L / far_center trials and ~66 % of
#: acute far_R in PS92 / PS93 -- a position- and epoch-dependent selection; at 1.0 s every animal x position x epoch
#: keeps >= 95 %. Unset = 1.0 under LP, the existing 2.0 s under DAQ; set under DAQ for the MATCHED DAQ arm
#: (contact-free 1.0 s), so LP vs DAQ differ only in the lick definition. Applies to align == "precue" only, where
#: `post_s` IS the window that ends at the cue.
ENL_WINDOW_ENV = "WIDEFIELD_ENL_WINDOW_S"
LP_ENL_WINDOW_S = 1.0


def enl_window_s() -> float | None:
    v = os.environ.get(ENL_WINDOW_ENV, "").strip()
    if v:
        return float(v)
    return LP_ENL_WINDOW_S if is_lp() else None


def apply_enl_window(args):
    """``args`` with post_s replaced by the ENL window for the precue alignment (a copy; others returned as is). Called
    where features are KEYED and where they are built, so the cache key always sees the window actually used."""
    import copy
    w = enl_window_s()
    if w is None or getattr(args, "align", None) != "precue" or getattr(args, "post_s", None) == w:
        return args
    a = copy.copy(args)
    a.post_s = float(w)
    return a


def output_tag() -> str | None:
    """Output-folder tag of a non-default run: "LP_licks" (LP source, ENL 1.0 s unless overridden -> "LP_licks_ENL<w>s"),
    "DAQ_ENL<w>s" (DAQ source with a non-default ENL window); None for the default DAQ run."""
    w = enl_window_s()
    if is_lp():
        return "LP_licks" if w == LP_ENL_WINDOW_S else f"LP_licks_ENL{w:g}s"
    return None if w is None else f"DAQ_ENL{w:g}s"


def output_dir(default: Path) -> Path:
    """Renderer output for a non-default run: <parent>/<tag>/<name> (grant_figures/epoch -> grant_figures/LP_licks/epoch;
    grant_figures -> grant_figures/LP_licks). Unchanged for the default DAQ run."""
    default = Path(default)
    tag = output_tag()
    if tag is None:
        return default
    if default.name == "grant_figures":
        return default / tag
    return default.parent / tag / default.name


def stamp(fig) -> None:
    """Corner label on every figure of a non-default run, so a copied PNG still says what it is."""
    tag = output_tag()
    if tag is None:
        return
    w = enl_window_s()
    txt = (("LP licks: video lick (cam4 + cam1) attempts, gate = hit / incomplete within max_rt; lick-aligned = LP "
            "onset; ") if is_lp() else "DAQ contact licks; ") + f"ENL lick-free window {w:g} s"
    fig.text(0.995, 0.002, txt, ha="right", va="bottom", fontsize=6, color="#8a2be2")


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


def _inputs(label: str, rv):
    """(session dir, cam4 licks_all, cam1 events) paths for one session."""
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    an, d = label.split("_")
    sd = oo.session_dir(rv, oc.RUN, an, "2026" + d)
    return sd, sd / "licks_all.csv", Path(rv.root("dlc")) / "session_poses" / "cam1_licks" / f"{an}_2026{d}_events.csv"


_TIMES: dict = {}


def lp_lick_times(label: str, rv=None) -> tuple[np.ndarray, pd.DataFrame]:
    """(all LP lick onsets in DAQ s, sorted; per-trial cue table [trial_id -> cue_daq]) for one session.

    EFFICIENCY (Priya 10-21: "are you ... using cache and sidecars"): one session is asked for up to three times per
    process (shared lick loader, `_trial_features`, `categorize`) and by every worker of every render, and the inputs
    are CSVs on the share. Memoised in-process, and kept as a LOCAL sidecar (`result_cache`, keyed on the size + mtime
    of windows.npz / licks_all.csv / the cam1 events and this module's source), so the share is read once per input
    change."""
    from wfield_local import result_cache, session_cache
    from wfield_local.paths import PathResolver
    rv = rv or PathResolver()
    sd, f4, f1 = _inputs(label, rv)
    if not (sd / "windows.npz").exists() or not f4.exists():
        raise LPUnavailable(f"{label}: no LP lick tables in {sd}")
    ins = [sd / "windows.npz", f4, f1]
    key = (label, tuple(result_cache.signature(ins, "")))
    if key not in _TIMES:
        import sys
        ver = f"lp{LP_VERSION}-" + result_cache.code_version(sys.modules[__name__])
        if len(_TIMES) > 64:
            _TIMES.clear()
        _TIMES[key] = result_cache.cached(session_cache.CACHE_DIR / "lp_licks", f"{label}__times", ins, ver,
                                          lambda: _read_times(label, sd, f4, f1))
    return _TIMES[key]


def _read_times(label, sd, f4, f1):
    from wfield_local import orofacial_cohort as oc
    cue_t = oc._cue_table(sd).set_index("trial_id")
    t = pd.read_csv(f4, usecols=["daq_on_s"]).daq_on_s.to_numpy(float)
    parts = [t[np.isfinite(t)]]
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


_LP: dict = {}


def session_lp(label: str, cue_samples, sr: float, max_rt: float, rv=None) -> dict:
    """Memoised per process on (label, cue train, sr, max_rt, attempts identity) -- see `_session_lp`."""
    import hashlib
    cs = np.ascontiguousarray(np.asarray(cue_samples, np.int64))
    key = (label, hashlib.sha1(cs.tobytes()).hexdigest(), float(sr), float(max_rt), spec_token(rv))
    if key not in _LP:
        if len(_LP) > 64:
            _LP.clear()
        _LP[key] = _session_lp(label, cs, sr, max_rt, rv)
    return _LP[key]


def _session_lp(label: str, cue_samples, sr: float, max_rt: float, rv=None) -> dict:
    """The LP lick train for `_trial_features`, on the DAQ sample clock.

    Returns ``lick_samples`` (int, sorted), ``drop`` (bool per DAQ cue: no attempts row / unknown class / unmatched ->
    exclude the trial; an `unknown` class is a no-lick trial, not a drop), ``attempted`` (bool per DAQ cue), ``first_s`` (first LP lick minus cue, s; NaN if none),
    ``cls`` / ``trial_id`` (the attempts class and trial_id per DAQ cue; "" / -1 when unmatched -- `attempt_coding` reads them) and
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
    cue_cls = np.full(n, "", dtype=object)          # attempts class per DAQ cue ("" = no attempts row matched)
    cue_tid = np.full(n, -1, int)                   # trial_attempts trial_id per DAQ cue (-1 = none)
    lat = A.combined.fillna(A.cam4).to_numpy(float) / 1000.0
    cls = A.cls.astype(str).to_numpy()
    for kk, good, c, la, tid in zip(k, ok, cls, lat, A.index.to_numpy(int)):
        if not good:
            continue
        drop[kk] = False
        cue_cls[kk], cue_tid[kk] = c, tid
        if c in ATTEMPT_CLASSES and np.isfinite(la):
            attempted[kk] = True
            first_s[kk] = la
    t_all = edit_train(licks, cue_s, drop, attempted, first_s, max_rt)
    return {"lick_samples": np.round(t_all * sr).astype(np.int64), "drop": drop, "attempted": attempted,
            "first_s": first_s, "cls": cue_cls, "trial_id": cue_tid, "n_cues": n, "n_matched": int((~drop).sum()),
            "n_attempted": int((attempted & (first_s <= max_rt)).sum())}
