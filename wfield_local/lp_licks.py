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

ENGAGEMENT GATE: a separate switch, WIDEFIELD_GATE_SOURCE=lp (11-01, `wfield_local/gate_source.py`), makes the engaged /
working / stopped gate and the S28 quit count DAQ contact OR an LP attempt as a response. NB under this module's
WIDEFIELD_LICK_SOURCE=lp alone, the imaging gate and the S28 quit are already LP-only (they read this train), while the
rest-baseline gates stay DAQ-only; see that module. Its runs get a "_gateLP" output tag here.
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
LP_VERSION = 2          # 2 (11-04): first-lick timing falls back to the DAQ contact at the close spouts
#: LP-vs-DAQ timing tolerance, ms (Priya 11-04; wf_attempt docs/status/rebuild_20261103/lp_missed_contact/
#: LP_MISSED_CONTACT_20261104.md): a first LP lick ONSET more than this after the trial's first DAQ contact (in the
#: response window, cue + 70 ms .. min(cue + max_rt, stop)) is late -- at a CLOSE spout the contact time is used instead
#: (LP missed the lick; ``timing_source`` = "daq"), at a FAR spout the LP time is kept and the trial is flagged
#: ``grooming_suspect`` (a paw on the spout: 7 / 8 paw contacts caught, 0 false alarms). Classes are unchanged. Enters
#: `spec_token`, so every LP cache key / lick-aligned frame carries it.
LP_DAQ_TOL_MS = 50.0
CLOSE_POSITIONS = ("close_ipsi", "close_mid", "close_contra", "close_L", "close_center", "close_R")


class LPUnavailable(RuntimeError):
    """The session has no LP lick tables (no video, or not tracked): it has no LP_licks version."""


def lick_source() -> str:
    v = os.environ.get(LICK_SOURCE_ENV, "daq").strip().lower() or "daq"
    if v not in LICK_SOURCES:
        raise ValueError(f"{LICK_SOURCE_ENV}={v!r}; expected one of {LICK_SOURCES}")
    return v


def is_lp() -> bool:
    return lick_source() == "lp"


def refuse_lp_folder(path, what: str) -> None:
    """An LP_licks output folder is written ONLY by an LP run (fifth review N5b): a render block that forgot to set
    WIDEFIELD_LICK_SOURCE=lp would otherwise put DAQ-gated figures into an LP-named folder. SystemExit."""
    from pathlib import Path as _P
    if any(seg.startswith("LP_licks") for seg in _P(path).parts) and not is_lp():
        raise SystemExit(f"{what}: refusing to write {path} -- an LP_licks folder, but {LICK_SOURCE_ENV} is not 'lp' "
                         "(set it in the render block: render_profile block_env)")


def _run_dir(rv=None) -> Path:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    return oo.root(rv or PathResolver()) / oc.RUN


def _enc_dir(rv=None) -> Path:
    """The adopted run's imaging_encoding tree (`orofacial_outputs.encoding_root`: inside the rebuild root under one)."""
    from wfield_local import orofacial_outputs as oo
    return oo.encoding_root(rv=rv)


def attempts_path(rv=None) -> Path:
    """The per-trial attempts table: THIS run's (`run_mode.attempts_dir`, coordinator 11-04) -- in a camera-gated
    rebuild run the routed table on the run's camera classes, never the pre-rebuild one; the untagged qc by default."""
    from wfield_local import run_mode
    if run_mode.tag():
        return run_mode.attempts_dir() / "trial_attempts.csv"
    return _enc_dir(rv) / "qc" / "trial_attempts.csv"


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
    return (f"lp{LP_VERSION}t{LP_DAQ_TOL_MS:g}:{st.st_mtime_ns if st else 0}:{st.st_size if st else 0}"
            f":cam1-{_cam1_signature(rv)}")


def first_lick_timing(pos, lp_ms, contact_ms, stop_ms, tol_ms: float = LP_DAQ_TOL_MS) -> pd.DataFrame:
    """Per trial (Priya 11-04, `LP_DAQ_TOL_MS`): ``first_lick_ms`` (the lick time lick-aligned frames use, ms from the
    cue), ``timing_source`` ("lp" | "daq" | "" = no time) and ``grooming_suspect``. ``lp_ms`` = first LP lick onset
    (combined, else cam4), ``contact_ms`` = first DAQ contact from cue + 70 ms; only a contact inside the response
    window (<= min(max_rt, stop), `trial_classes.response_end_ms`) is used. CLOSE spouts: LP missing or > tol after the
    contact -> the contact time ("daq"). FAR spouts: the LP time, flagged when > tol after the contact."""
    from wfield_local import trial_classes as TC
    lp = np.asarray(lp_ms, float)
    ct = np.asarray(contact_ms, float)
    ct = np.where((ct >= TC.POSTCUE_START_MS) & (ct <= TC.response_end_ms(stop_ms)), ct, np.nan)
    close = np.isin(np.asarray(pos, dtype=object), CLOSE_POSITIONS)
    has_c = np.isfinite(ct)
    late = has_c & (~np.isfinite(lp) | (lp > ct + tol_ms))
    use_daq = close & late
    t = np.where(use_daq, ct, lp)
    src = np.where(use_daq, "daq", np.where(np.isfinite(lp), "lp", ""))
    groom = ~close & has_c & np.isfinite(lp) & (lp > ct + tol_ms)
    return pd.DataFrame({"first_lick_ms": t, "timing_source": src, "grooming_suspect": groom})


#: Pre-cue (ENL) lick-free window length, s (Priya 10-21, after `scripts/enl_lp_census.py`: "1.0 s, plus DAQ 1.0 s").
#: With LP licks a 2 s lick-free window keeps only 48 % of PS93's pre-stroke far_L / far_center trials and ~66 % of
#: acute far_R in PS92 / PS93 -- a position- and epoch-dependent selection; at 1.0 s every animal x position x epoch
#: keeps >= 95 %. Unset = 1.0 under LP, the existing 2.0 s under DAQ; set under DAQ for the MATCHED DAQ arm
#: (contact-free 1.0 s), so LP vs DAQ differ only in the lick definition. Applies to align == "precue" only, where
#: `post_s` IS the window that ends at the cue.
ENL_WINDOW_ENV = "WIDEFIELD_ENL_WINDOW_S"
LP_ENL_WINDOW_S = 1.0


def enl_window_s() -> float | None:
    """The ENL window OVERRIDE in force (WIDEFIELD_ENL_WINDOW_S, else 1.0 under LP), or None for the default DAQ run.
    None means "the config window" -- `enl_post_s` resolves it; this form exists for keys that must stay unchanged on
    the default run (`frozen_models.make_spec`, `map_collect.cache_kind`)."""
    v = os.environ.get(ENL_WINDOW_ENV, "").strip()
    if v:
        return float(v)
    return LP_ENL_WINDOW_S if is_lp() else None


def enl_post_s(source: str | None = None) -> float:
    """THE pre-cue (ENL) window, s -- the ONE definition every ENL analysis reads (Priya 11-04: "every ENL analysis
    follows the lick source"). WIDEFIELD_ENL_WINDOW_S if set; else 1.0 s under the LP source (`LP_ENL_WINDOW_S`); else
    the DAQ window `decode.precue_post_s` (2.0 s). ``source`` forces a lick source instead of reading
    WIDEFIELD_LICK_SOURCE -- cohort 2 has no LP and passes "daq". Never read `precue_post_s` directly for an ENL
    window: under LP it is the wrong number (the 11-04 audit found six scripts doing so)."""
    src = lick_source() if source is None else str(source).strip().lower()
    if src not in LICK_SOURCES:
        raise ValueError(f"lick source {source!r}; expected one of {LICK_SOURCES}")
    v = os.environ.get(ENL_WINDOW_ENV, "").strip()
    if v:
        return float(v)
    if src == "lp":
        return LP_ENL_WINDOW_S
    from wfield_local import config
    return float(config.defaults()["decode"]["precue_post_s"])


def is_precue(align) -> bool:
    """True for the pre-cue alignment and its variant names ("precue-rt2-..", see `frozen_models.make_spec`)."""
    return align == "precue" or str(align).startswith("precue-")


def post_s_for(align: str, source: str | None = None) -> float:
    """``decode.{align}_post_s``, except the pre-cue window, which is `enl_post_s` -- for code that resolves a window
    per alignment (`position_coding_directions`, `cd_trajectories`, ...)."""
    if is_precue(align):
        return enl_post_s(source)
    from wfield_local import config
    return float(config.defaults()["decode"].get(f"{align}_post_s", 2.0))


#: DAQ-clock sample rate used when a caller has cue / lick TIMES (s) rather than samples (`gate_licks_s`): 10 us.
_GATE_SR = 100_000.0


def gate_licks_s(label: str, cue_s, daq_lick_s, max_rt: float, source: str | None = None, rv=None):
    """``(lick_s, keep)``: the lick train an ENL lick-free gate must use -- the SAME source as `enl_post_s` -- and a
    per-cue keep mask. DAQ: ``daq_lick_s`` sorted, every cue kept. LP: `session_lp`'s train (the edited LP train
    `_trial_features` uses) on these cues, and cues without an LP verdict dropped. For scripts that gate on cue / lick
    TIMES (DAQ s) instead of through `_trial_features`; ``cue_s`` must be finite."""
    src = lick_source() if source is None else str(source).strip().lower()
    cue_s = np.asarray(cue_s, float)
    if src != "lp":
        return np.sort(np.asarray(daq_lick_s, float)), np.ones(len(cue_s), bool)
    lp = session_lp(label, np.round(cue_s * _GATE_SR).astype(np.int64), _GATE_SR, float(max_rt), rv)
    return np.asarray(lp["lick_samples"], float) / _GATE_SR, ~np.asarray(lp["drop"], bool)


#: ms grid on which `enl_window_starts_s` runs the decoders' frame-based search (licks occupy one tick: 1 ms).
_SLIDE_TICK_S = 1e-3


def enl_window_starts_s(cue_s, strobe_s, lick_s, win_s: float | None = None):
    """Per trial, the start (DAQ s) of its LICK-FREE ENL window, NaN = no lick-free window -> drop the trial.

    THE DECODERS' OWN RULE, not a copy (Priya 11-04: "reuse that exact function"):
    `locanmf_position_decoder.precue_window_start` -- the fixed window ending at the cue if it holds no lick, else the
    latest lick-free gap in [strobe, cue] long enough for the window, else drop -- run on a 1 ms grid, since it works
    in integer frames. ``lick_s`` must come from the window's source (`gate_licks_s`); ``win_s`` defaults to
    `enl_post_s()`. For scripts that build ENL windows on times instead of through `_trial_features`. The search is
    `enl_window.precue_window_start` itself (the decoders re-export it; the combined encoder calls it on frames): one
    function, imported from its home so a time-based caller does not pull in the decoder's sklearn imports."""
    from wfield_local.enl_window import precue_window_start
    w = enl_post_s() if win_s is None else float(win_s)
    k = 1.0 / _SLIDE_TICK_S
    L = np.sort(np.round(np.asarray(lick_s, float) * k))
    wn = int(round(w * k))
    out = np.full(len(np.atleast_1d(cue_s)), np.nan)
    for i, (c, st) in enumerate(zip(np.atleast_1d(cue_s), np.atleast_1d(strobe_s))):
        if not np.isfinite(c):
            continue
        r = precue_window_start(int(round(c * k)), (st * k) if np.isfinite(st) else np.nan, L, wn)
        if r is not None:
            out[i] = r / k
    return out


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
        tag = "LP_licks" if w == LP_ENL_WINDOW_S else f"LP_licks_ENL{w:g}s"
    else:
        tag = None if w is None else f"DAQ_ENL{w:g}s"
    from wfield_local import gate_source
    if gate_source.is_lp():             # LP-inclusive gate: its own folder, never the plain LP_licks / default one
        tag = f"{tag or 'DAQ'}_gateLP"
    # RUN MODE (rebuild 2026-11-03, `run_mode`): hemo variant, LocaNMF projection, PRE_FROM, trial selection, map refs,
    # in-memory refits, the LP arm's DAQ-trained cross-check. Each adds a part, so such a run is routed by
    # `output_dir` and REFUSED by `writeguard` outside its folder. The default run adds nothing (None, as before).
    from wfield_local import run_mode
    extra = run_mode.parts()
    if extra:
        tag = "_".join(([tag] if tag else []) + extra)
    return tag


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
            "onset; ") if is_lp() else "DAQ contact licks; ") + (f"ENL lick-free window {w:g} s" if w is not None
                                                                  else "ENL lick-free window 2 s")
    from wfield_local import gate_source
    if gate_source.is_lp():
        txt += "; engagement gate / quit = DAQ contact OR LP attempt"
    from wfield_local import run_mode
    if run_mode.tag():
        txt += f"; mode {run_mode.tag()}"
    from wfield_local import trial_classes as _tc
    if _tc.camera_gate_on():                  # N6 (Priya 11-03): name the partially video-covered sessions
        try:
            txt += "\n" + _tc.coverage_note()
        except Exception as ex:                       # noqa: BLE001 - a caption must not kill a figure; say why
            if output_tag():
                # coordinator 11-04: in a tagged (rebuild) run a missing coverage table is an ERROR, never a quiet
                # "unavailable" caption on every camera-gated figure
                raise
            txt += f"\ncamera gate: coverage note unavailable ({type(ex).__name__})"
        from wfield_local import spout_scaling
        if spout_scaling.spout_scaling() and spout_scaling.FALLBACK_USED:          # D-B3: the fallbacks actually used
            txt += ("\nbal_em centre: camera-working rows; contact-row fallback for "
                    + ", ".join(sorted(spout_scaling.FALLBACK_USED)))
        txt += "\n" + _tc.design_note()                                       # review 7A (main's 11-03 design)
        try:                                                                      # review 7A NB3 / 8A: from the classes
            _ex = _tc.working_excluded_sessions()
        except Exception as ex:                                                   # noqa: BLE001 - say why, never die
            _ex = [f"unavailable ({type(ex).__name__})"]
        if _ex:
            txt += "\nworking selection excludes (no camera working rows): " + ", ".join(_ex)
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
        A = pd.read_csv(p)
        if "cls_raw" in A:
            # a camera-gated table: `cls` carries the run's session-level classes (stopped / early_run / after_end);
            # the LP source needs the TRIAL-level class (the camera gate applies the run's rules itself)
            A["cls"] = A["cls_raw"]
        for c in ("pos", "stop_ms", "contact"):
            if c not in A:
                raise LPUnavailable(f"{p}: no {c!r} column (the first-lick DAQ fallback needs it; rebuild the table)")
        A = A[["label", "trial_id", "cls", "cam4", "combined", "pos", "stop_ms", "contact"]].copy()
        tm = first_lick_timing(A.pos, A.combined.fillna(A.cam4), A.contact, A.stop_ms)
        A["first_lick_ms"], A["timing_source"] = tm.first_lick_ms.to_numpy(), tm.timing_source.to_numpy()
        A["grooming_suspect"] = tm.grooming_suspect.to_numpy()
        _ATTEMPTS[key] = A
    return _ATTEMPTS[key]


def _inputs(label: str, rv):
    """(session dir, cam4 licks_all, cam1 events) paths for one session."""
    from wfield_local import orofacial_cohort as oc
    from wfield_local import config, orofacial_outputs as oo
    an, d = label.split("_")
    sd = oo.session_dir(rv, oc.RUN, an, config.date8(d))
    return sd, sd / "licks_all.csv", Path(rv.root("dlc")) / "session_poses" / "cam1_licks" / f"{an}_{config.date8(d)}_events.csv"


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
    # first lick time: LP, or at a close spout the DAQ contact when LP missed / was late (`first_lick_timing`, 11-04)
    lat = A.first_lick_ms.to_numpy(float) / 1000.0
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
