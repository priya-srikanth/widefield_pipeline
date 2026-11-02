"""LP-INCLUSIVE ENGAGEMENT GATE -- an OPTION (Priya 2026-11-01: "should we do it anyway for consistency? most of our
final analyses will be based on LP licks" -> yes, for the LP arm; the DAQ arms stay as they are).

    WIDEFIELD_GATE_SOURCE = daq (default) | lp

WHAT IT CHANGES. Only who counts as "still responding" for the ONE engagement rule,
`precue_engagement_states.engagement_gate` (reference-spout, non-recovering terminal collapse, backdated), i.e. the
engaged / working vs stopped split and the S28 quit point. It never moves a trial between the lick and no-lick arms
(that is WIDEFIELD_LICK_SOURCE's job, `lp_licks`), never changes a hit rate, a latency or a lick count.

  daq  every gate exactly as before. `gate_responded` returns its input OBJECT untouched (no copy, no I/O), so the
       default path is byte-identical by construction; pinned by `tests/test_lp_gate_option.py`.
  lp   a trial is a RESPONSE for the gate if it has
         (a) the caller's own response (unchanged: the engaged arm, the DAQ hit, ...), OR
         (b) a DAQ spout CONTACT in [cue, min(cue + window, next cue)] -- read with `_load_daq_events(...,
             allow_lp=False)`, so it is contact even inside an LP_licks render, OR
         (c) an LP ATTEMPT: trial_attempts class in `lp_licks.ATTEMPT_CLASSES` = hit / incomplete, matched to the DAQ cue
             by cue time (`lp_licks.session_lp`), with the first LP lick (combined, else cam4) <= window after the cue.
             jaw_only / no_try / unknown are NOT responses (jaw_only = a reaction, not an attempt; Priya's 10-21 blind
             review). A cue with no LP verdict (no attempts row, unmatched, or a session without video -- PS93_0813)
             gets (a) | (b) only.
       window = the caller's own response window: decode.max_rt_s (3.5 s) for the imaging gates, the session's
       response window (3.5 s) for the rest-baseline gates, `resp_s` (2.0 s) for the S28 quit rows.

WHERE IT IS WIRED (one helper, `gate_responded`, called at each place a gate's `responded` vector is built):
  precue_engagement_states._session_tables  -> position_coding_directions._gate_all (lick / miss_working / stopped),
                                               attempt_coding, grant_kit, position_axes, the precue-state analysis
  beta_maps._quit_mask                      -> 14 / 15r working + miss_working / stopped maps, attempt_maps,
                                               position_reference_maps, hemo_map_control
  nolick_decoder.categorize (session engagement column) and poststroke_compare.poststroke_engagement
  rest_by_position._engaged_trials, rest_engagement.engaged_by_cue (the rest-baseline gates)
  engagement_decomposition.session_trials   adds a `gate_hit` column ONLY under the switch, and
  quit_point.session_quit                   uses it when present (-> analysis_kit.session_behavior, S28 quit tests)
NOT wired (behaviour-pipeline / orofacial gates, judged on the behaviour trial table, not the imaging one):
  spout_behavior.reference_engagement / flag_engagement (behaviour figures, behavior_clips) and
  orofacial_cohort.engagement_keep (reads the behaviour table's `engaged` column; its own "attempted" gate options
  already exist). Scripts that call engagement_gate directly (disengagement_tests, enl_state_counts,
  stopped_trial_traces, channel_position_maps) are DAQ-arm diagnostics and are not wired either.

WHAT THE LP ARM'S GATE WAS BEFORE THIS SWITCH (found 11-01, worth knowing when reading LP_licks numbers): under
WIDEFIELD_LICK_SOURCE=lp the shared lick loader (`plot_lick_aligned_averages._load_daq_events`) already returns the LP
train, so the imaging gate (engaged arm = LP-attempted arm), `nolick_decoder.categorize` and `session_trials` (hence
the S28 quit) were already LP-ONLY -- no DAQ contact -- while the rest-baseline gates (`rest_engagement`,
`rest_by_position`, which read the canonical DAQ events / lick channel directly) stayed DAQ-ONLY. The switch makes all of
them the same DAQ-OR-LP rule.

CACHE KEYS -- the gate source enters a key ONLY when it is "lp" (the pattern of `lp_licks.cache_tag` and
`frozen_models.LP_FROZEN_ENV`), so every existing DAQ / LP_licks cache entry and every frozen-model spec id is
unchanged:
  * `session_cache.cached` appends `cache_tag()` to the kind under the switch -- every session-cached gate consumer
    (map_collect pixel maps "bm-*" / "prm-*", categorize, CD arms, ...). Exempt: trial features ("tf-*"), which are built
    before any gate is applied, so they stay shared and warm.
  * map_collect's source digests (`beta_maps._session_maps_code`, `position_reference_maps._raw_maps_code`) hash the
    SOURCE of `_quit_mask`, `precue_engagement_states`, `rest_engagement`, `rest_by_position`. Every hook line carries
    the marker `HOOK_MARK` and `map_collect.code_digest` drops marked lines before hashing, so the hooks do not move
    those digests (they are the identity when the switch is off; under the switch the kind carries `cache_tag()`,
    whose GATE_VERSION is bumped when the rule changes). A hook must therefore be WHOLE inserted lines, each marked.
  * in-process lru_caches need nothing (the env is fixed per process, and `parallel` children inherit it).
  * `grant_kit._boot_cached` is keyed on the data it consumes, so a changed gate changes its key by itself.
  * FROZEN MODELS: not touched. Every frozen decoder is trained on pre-stroke trial FEATURES, which no gate selects
    ("working" / "lick only" choose which post-stroke trials are scored), so the spec deliberately does not carry the
    gate source -- a gate-LP render reuses the same frozen references.
  * outputs: `lp_licks.output_tag()` gains a "_gateLP" suffix (LP_licks_gateLP, DAQ_gateLP), so `writeguard` refuses a
    gate-LP run's write into the plain LP_licks or default folders, and `lp_licks.stamp` says so on every figure.

HOW TO RUN AN LP-ARM RENDER WITH IT:  WIDEFIELD_LICK_SOURCE=lp WIDEFIELD_GATE_SOURCE=lp <the usual render command>.
Measured effect (curated PS92-95, August-only pre): docs/status/LP_GATE_OPTION_20261101.md.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

#: Selects what counts as a RESPONSE for the engagement gate / quit point.
GATE_SOURCE_ENV = "WIDEFIELD_GATE_SOURCE"
GATE_SOURCES = ("daq", "lp")
#: Bump when the gate-response rule (or a hook) changes: enters `cache_tag`, the only key that sees the hooks.
GATE_VERSION = 1
#: Trailing marker on every hook line in a source-digested module (see CACHE KEYS above).
HOOK_MARK = "# [gate-source hook]"


def source() -> str:
    v = os.environ.get(GATE_SOURCE_ENV, "daq").strip().lower() or "daq"
    if v not in GATE_SOURCES:
        raise ValueError(f"{GATE_SOURCE_ENV}={v!r}; expected one of {GATE_SOURCES}")
    return v


def is_lp() -> bool:
    return source() == "lp"


def cache_tag(rv=None) -> str:
    """Suffix `session_cache.cached` appends to every kind under WIDEFIELD_GATE_SOURCE=lp (never otherwise). Carries
    GATE_VERSION and the LP inputs' identity (`lp_licks.spec_token`: attempts table + cam1 folder), so a rewrite of the LP
    tables starts fresh gate-LP entries."""
    from wfield_local import lp_licks
    return "glp" + hashlib.sha1(f"g{GATE_VERSION}:{lp_licks.spec_token(rv)}".encode()).hexdigest()[:8]


_DAQ_TRAIN: dict = {}
_EXTRA: dict = {}


def _daq_contact_samples(h5) -> tuple[np.ndarray, float]:
    """(sorted DAQ CONTACT lick samples, sample rate): always contact, even inside an LP_licks render. Memoised per
    process (one .h5 read per session)."""
    from wfield_local.plot_lick_aligned_averages import _load_daq_events
    key = str(h5)
    if key not in _DAQ_TRAIN:
        if len(_DAQ_TRAIN) > 32:
            _DAQ_TRAIN.clear()
        lk = _load_daq_events(h5, "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10, allow_lp=False)
        _DAQ_TRAIN[key] = (np.sort(np.asarray(lk["lick_samples"], float)), float(lk["sample_rate_hz"]))
    return _DAQ_TRAIN[key]


def extra(label: str, window_s: float, cue_samples=None, sr=None, rv=None) -> dict:
    """Per DAQ cue, the responses the LP-inclusive gate adds: ``{"daq", "lp", "lp_cov", "any"}`` bool arrays.

    Indexed over ``cue_samples`` if given -- pass the caller's own train so indices line up; ``sr`` defaults to the
    session's DAQ rate (cue samples are DAQ samples) -- else the session's DAQ cue train
    (`plot_spout_trial_averages._load_daq_events`, the train `_trial_features` indexes).

      daq     a DAQ contact in [cue, min(cue + window_s, next cue)] (`session_trials`' hit rule)
      lp      LP attempt: class hit / incomplete, cue-time matched, first LP lick <= window_s after the cue
      lp_cov  the cue has an LP verdict (False everywhere for a session without LP tables)
      any     daq | lp
    """
    from wfield_local import config, lp_licks
    s = next((x for x in config.load_sessions() if x["label"] == label), None)
    if s is None:
        raise KeyError(f"gate_source.extra: no registered session {label}")
    if cue_samples is None:
        from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue
        cue = _load_cue(s["h5"])
        cue_samples, sr = cue["cue_samples"], cue["sample_rate_hz"]
    if sr is None:
        sr = _daq_contact_samples(s["h5"])[1]
    cs = np.ascontiguousarray(np.asarray(cue_samples, np.int64))
    key = (label, hashlib.sha1(cs.tobytes()).hexdigest(), float(sr), float(window_s), lp_licks.spec_token(rv))
    if key in _EXTRA:
        return _EXTRA[key]
    licks, sr_l = _daq_contact_samples(s["h5"])
    if abs(sr_l - float(sr)) > 1e-6 * max(sr_l, 1.0):
        licks = licks * (float(sr) / sr_l)        # caller's cues on a nominal rate: same clock, rescale the licks
    csf = cs.astype(float)
    end = np.minimum(csf + float(window_s) * float(sr), np.append(csf[1:], np.inf))
    daq = np.searchsorted(licks, end, "right") > np.searchsorted(licks, csf, "left")
    lp = np.zeros(cs.size, bool)
    cov = np.zeros(cs.size, bool)
    try:
        L = lp_licks.session_lp(label, cs, float(sr), float(config.defaults()["decode"]["max_rt_s"]), rv)
        cov = ~np.asarray(L["drop"], bool)
        fs = np.asarray(L["first_s"], float)
        lp = cov & np.asarray(L["attempted"], bool) & np.isfinite(fs) & (fs <= float(window_s))
    except lp_licks.LPUnavailable as ex:
        print(f"  [gate LP] {label}: no LP verdicts ({str(ex)[:60]}) -- gate = DAQ contact only", flush=True)
    out = {"daq": daq, "lp": lp, "lp_cov": cov, "any": daq | lp}
    if len(_EXTRA) > 64:
        _EXTRA.clear()
    _EXTRA[key] = out
    return out


def gate_responded(label: str, order, responded, window_s: float | None = None, cue_samples=None, sr=None):
    """``responded`` for `engagement_gate` under the gate source -- THE one place the LP-inclusive rule is applied.

    Unset / "daq": returns ``responded`` ITSELF (same object; no copy, no I/O). "lp": ``responded | extra.any[order]``,
    where ``order`` holds DAQ cue indices (every gate caller's trial order is a cue index); out-of-range indices add
    nothing. ``window_s`` defaults to decode.max_rt_s.
    """
    if not is_lp():
        return responded
    from wfield_local import config
    w = float(config.defaults()["decode"]["max_rt_s"]) if window_s is None else float(window_s)
    ex = extra(label, w, cue_samples=cue_samples, sr=sr)["any"]
    order = np.asarray(order, np.int64)
    ok = (order >= 0) & (order < ex.size)
    add = np.zeros(order.size, bool)
    add[ok] = ex[order[ok]]
    return np.asarray(responded, bool) | add


def strip_hooks(src: str) -> str:
    """``src`` without its marked hook lines -- what `map_collect.code_digest` hashes (see CACHE KEYS)."""
    if HOOK_MARK not in src:
        return src
    return "".join(ln for ln in src.splitlines(keepends=True) if HOOK_MARK not in ln)
