"""RESTnl -- the LICK-FREE ("no-lick") FLAT rest baseline, as a per-session MAP for the epoch map families.

Priya 11-01: RESTnl is the PRIMARY rest reference for the imaging-page rest-referenced maps (figures 7.1-7.11,
7.16-7.20, 8.1, 8.2); the page's old REST (time-local, DAQ-contact lick exclusion) is kept as a supplement. Rule `wr`,
FLAT (verified appropriate, docs/status/RESTNL_MAP_SHORTCUT_CHECK_20261101.md in the wf_attempt worktree), pre =
August-only for the RESTnl outputs.

THE DEFINITION (ported from wf_attempt `scripts/restw_strict_check.py` mask `strict` and `scripts/restnl.py` rule `wr`;
the goal page's RESTnl, so both pages subtract the same quantity):

  MASK    the restdock05 rest sample mask (`rest_by_position._session_daq`, the RESTW mask: docked, not running, DAQ
          contact licks excluded [0.5, 1.0] s, min_rest_s) MINUS every DAQ lick onset (the mask's own detector and
          config criteria) and EVERY cam4 Lightning-Pose lick of `licks_all.csv` (contact or not, any context;
          `daq_on_s`), each widened [0.5 s before, 2.0 s after]; the mask's min_rest_s floor re-applied after each
          intersection, as `quiet_periods.rest_mask` does last.
  FRAMES  `rest_by_position.rest_frames_by_position(docked=False, engaged_only=True)` -- the module's own period
          labelling and engagement gate -- fed the lick-free mask by in-process substitution of its loader (the same
          substitution `restw_strict_check` makes; restored in a finally). Its file is NOT edited: it is inside
          `position_reference_maps._raw_maps_code()`, and editing it would invalidate every `prm` map cache.
  LEVEL   per position the per-component MEDIAN of the SVT over that position's rest frames; rule `wr` = equal-weight
          mean over the positions with >= 50 frames, renormalised over those present, >= 4 of 6 required (`strict` =
          all 6 with >= 50 frames, kept as an option). One (K,) vector b_nl per session.
  MAP     base_map = U_atlas @ b_nl on the maps' own grid (`joint_basis._load_session`: same U, same NaN zeroing, same
          MAP_SHAPE). Because the baseline is FLAT and the map window features are linear means with no per-trial
          baseline, RESTnl[q] = raw[q] - base_map EXACTLY for every position q, alignment and lick gate -- so the
          RESTnl reference is formed after collection from the cached `raw` column (`position_reference_maps.
          reference_maps(parts, "restnl")`), and ONE baseline per session serves all 7 `prm` cache kinds. Verified
          on the RESTW column (check doc section 2: flat to <= 1.3e-7 across positions and arms).

CACHE: one small npz per session in its OWN directory (`cache_dir()`, beside the session cache, never inside it), keyed
on the source of this module + the parameters + the input files (session_cache.session_signature + the rest mask +
licks_all.csv). Not a session_cache `prm` kind; CACHE_VERSION is not touched.

SESSIONS WITHOUT licks_all.csv (no behaviour video, e.g. PS93_0813) get NO RESTnl column -- never a DAQ-only fallback
under the same name. `build()` lists them.

    python -m wfield_local.restnl_mask [--labels ...] [--jobs 4] [--measure] [--validate]
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

NAME = "RESTnl"
BUF_S = (0.5, 2.0)                 # buffer around every lick: before, after (s)
MIN_FR = 50                        # frames a position needs to contribute a level
MIN_POS_WR = 4                     # rule wr: positions (of 6) with >= MIN_FR frames
RULES = ("wr", "strict")
RULE = "wr"
VERSION = 1
N_POS = 6
LICK_SOURCE_TEXT = "DAQ spout-contact licks + cam4 Lightning-Pose video licks (licks_all.csv)"
DEF = ("RESTnl (lick-free rest, flat): restdock05 rest minus every DAQ contact lick and every cam4 LP video lick "
       "[0.5 s before, 2.0 s after], engagement-gated; per-position median, equal-weight mean over positions with "
       ">= 50 frames (>= 4 of 6); one flat baseline per session")


# ------------------------------------------------------------------------------------------------ mask helpers
def widen(idx, n: int, nb: int, na: int) -> np.ndarray:
    """Boolean of length n, True within [i - nb, i + na] of every event sample i (= quiet_periods.widen_bool_sparse)."""
    idx = np.asarray(idx, np.int64)
    idx = idx[(idx >= 0) & (idx < n)]
    d = np.zeros(n + 1, np.int32)
    np.add.at(d, np.maximum(idx - nb, 0), 1)
    np.add.at(d, np.minimum(idx + na + 1, n), -1)
    return np.cumsum(d[:n]) > 0


def floor_runs(b: np.ndarray, n: int) -> np.ndarray:
    """The rest mask's own min-run floor (`quiet_periods.set_short_bool_to_low`)."""
    from wfield_local.quiet_periods import set_short_bool_to_low
    return set_short_bool_to_low(b, n)


def daq_lick_onsets(s: dict, fs: float) -> np.ndarray:
    """DAQ lick onsets (samples) with the rest mask's own detector and config criteria."""
    import h5py

    from wfield_local import config
    from wfield_local.lick_detection import detect_licks
    lk = config.defaults()["lick_detection"]
    with h5py.File(s["h5"], "r") as f:
        nm = [x.decode() for x in f["analog/channel_names"][:]]
        i = nm.index(lk.get("channel", "lick_analog"))
        if "samples_int16" in f["analog"]:
            sc = float(f["analog/int16_scale_volts_per_count"][i])
            of = float(f["analog/int16_offset_volts"][i])
            lv = f["analog/samples_int16"][:, i].astype(np.float32) * sc + of
        else:
            lv = np.asarray(f["analog/samples"][:, i], np.float32)
    det = detect_licks(lv, fs, thresh_upper=lk["thresh_upper"], thresh_lower=lk["thresh_lower"],
                       lockout_s=tuple(lk["lockout_falling_edge_s"]), min_ili_s=lk.get("min_ili_ms", 0) / 1000.0)
    return np.sort(np.asarray(det["lick_onsets"], np.int64))


def licks_all_path(label: str) -> Path:
    """The cam4 whole-session LP lick table of a session (may not exist)."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, mmdd = label.split("_")[0], label.split("_")[1]
    return oo.session_dir(PathResolver(), oc.RUN, an, config.date8(mmdd)) / "licks_all.csv"


def session_of(label: str) -> dict:
    from wfield_local import config
    return next(x for x in config.load_sessions() if x["label"] == label)


def lickfree_mask(s: dict):
    """``(mask, daq_parts, info)``: the lick-free rest SAMPLE mask and the DAQ pieces rest_frames_by_position needs.

    Raises FileNotFoundError when the session has no licks_all.csv (no cam4 video -> no RESTnl)."""
    import pandas as pd

    from wfield_local import analysis_kit as ak
    from wfield_local import config
    from wfield_local import rest_by_position as rbp
    lp = licks_all_path(s["label"])
    if not lp.exists():
        raise FileNotFoundError(f"no licks_all.csv ({lp})")
    rest0, cs_s, codes, ts, fs_samp, sync = rbp._session_daq(s)
    fs = ak.daq_rate(s)
    eng = rbp._engaged_trials(s, cs_s, codes, fs)
    n = rest0.shape[0]
    nb, na = int(BUF_S[0] * fs), int(BUF_S[1] * fs)
    minrun = int(float(config.defaults()["segmentation"]["rest"]["min_rest_s"]) * fs)
    daq_on = daq_lick_onsets(s, fs)
    cam4 = pd.read_csv(lp, usecols=["daq_on_s"]).daq_on_s.to_numpy(float)
    cam4 = cam4[np.isfinite(cam4)]
    m_daq2 = floor_runs(rest0 & ~widen(daq_on, n, nb, na), minrun)
    m = floor_runs(m_daq2 & ~widen(np.round(cam4 * fs).astype(np.int64), n, nb, na), minrun)
    info = {"n_daq_licks": int(len(daq_on)), "n_cam4_licks": int(len(cam4)), "n_rest0_samples": int(rest0.sum()),
            "n_daq2_samples": int(m_daq2.sum()), "n_lickfree_samples": int(m.sum()), "fs": float(fs),
            "n_trials": int(len(eng)), "n_trials_engaged": int(np.sum(eng))}
    return m, (cs_s, codes, ts, fs_samp, sync, eng), info


def frames_by_position(s: dict, n_frames: int, mask, daq_parts) -> dict:
    """``{code: frame indices}`` from `rest_by_position.rest_frames_by_position(docked=False, engaged_only=True)` with
    the lick-free mask substituted for its loader (in-process, restored in a finally; the module file is untouched)."""
    from wfield_local import rest_by_position as rbp
    cs_s, codes, ts, fs_samp, sync, eng = daq_parts
    od, oe = rbp._session_daq, rbp._engaged_trials
    try:
        rbp._engaged_trials = lambda *_a, **_k: eng
        rbp._session_daq = lambda _s: (mask, cs_s, codes, ts, fs_samp, sync)
        per, info = rbp.rest_frames_by_position(s, n_frames, docked=False, engaged_only=True)
    finally:
        rbp._session_daq, rbp._engaged_trials = od, oe
    if info.get("error"):
        raise RuntimeError(f"{s['label']}: rest_frames_by_position: {info['error']}")
    return per


def position_medians(V, per) -> tuple[np.ndarray, np.ndarray]:
    """``(posmed (6, K) NaN where a position has no frames, nfr (6,))``."""
    V = np.asarray(V)
    K, T = V.shape
    pm = np.full((N_POS, K), np.nan)
    nfr = np.zeros(N_POS, int)
    for c in range(N_POS):
        idx = np.asarray(per.get(c, []), np.int64)
        idx = idx[idx < T]
        nfr[c] = idx.size
        if idx.size:
            pm[c] = np.median(V[:, idx], axis=1)
    return pm, nfr


def combine(posmed, nfr, rule: str = RULE) -> tuple[np.ndarray, bool, list]:
    """``(b (K,), ok, positions used)`` under rule wr (default) or strict."""
    if rule not in RULES:
        raise ValueError(f"rule {rule!r} not in {RULES}")
    use = np.asarray(nfr) >= MIN_FR
    if rule == "strict":
        ok = bool(use.all())
    else:
        ok = bool(use.sum() >= MIN_POS_WR)
    b = np.asarray(posmed)[use].mean(0) if use.any() else np.full(np.asarray(posmed).shape[1], np.nan)
    ok = ok and bool(np.isfinite(b).all())
    return b, ok, [int(c) for c in np.flatnonzero(use)]


# ------------------------------------------------------------------------------------------------ cache
def cache_dir() -> Path:
    """OWN directory beside the session cache (never inside it)."""
    from wfield_local import session_cache
    c = Path(session_cache.CACHE_DIR)
    from wfield_local import gate_source as _gs  # [gate-source hook] LP-inclusive rest gate (11-02, lp-final)
    if _gs.is_lp():  # [gate-source hook] its own baselines, so a gate-LP build never prunes / reads the default ones
        return c.with_name(c.name + "_restnl_" + _gs.cache_tag())  # [gate-source hook]
    return c.with_name(c.name + "_restnl")


def _code() -> str:
    src = inspect.getsource(sys.modules[__name__])
    from wfield_local.gate_source import strip_hooks  # [gate-source hook] marked lines leave the digest unchanged
    src = strip_hooks(src)  # [gate-source hook]
    return hashlib.sha1(src.encode()).hexdigest()[:12]


def _helper_code() -> str:
    """Digest of the CROSS-MODULE code a RESTnl baseline depends on (review N8, 11-03): the rest frames and their gate
    (rest_by_position, rest_engagement, engagement_gate), the quiet / rest masks and response window (quiet_periods,
    analysis_kit) -- the same roots the prm map digest hashes (`position_reference_maps._raw_maps_code`)."""
    from wfield_local import map_collect
    from wfield_local.position_reference_maps import _raw_maps_parts
    # the CLOSURES of the prm digest (its cross-module helper set, sufficiency-tested), not the prm map functions
    return map_collect.code_digest(*[x for x in _raw_maps_parts() if hasattr(x, "source")])


def _key(s: dict) -> str:
    from wfield_local import map_collect, session_cache
    from wfield_local.quiet_periods import quiet_dir
    import glob
    qs = sorted(glob.glob(f"{quiet_dir(s['mc'])}/*quiet_sample.npy"))
    ua = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/U_atlas.npy")
    # + the defaults.yaml keys and helper code the baseline reads (review N8): a yaml / helper edit now rebuilds it
    parts = [f"v{VERSION}", _code(), _helper_code(), repr(sorted(map_collect.map_yaml_subset().items())),
             repr((BUF_S, MIN_FR, MIN_POS_WR)), session_cache.session_signature(s),
             map_collect.stat_sig(qs[0] if qs else None), map_collect.stat_sig(ua[0] if ua else None),
             map_collect.stat_sig(licks_all_path(s["label"]))]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _stem(label: str) -> str:
    """``<label>__restnl`` (default) or ``<label>__restnl-<run_mode.cache_tag>`` (rebuild 2026-11-03): another hemo
    variant / selection / projection run writes its OWN name, so the prune in `build_session` (by stem) can never delete
    the adopted baselines. Default names unchanged."""
    from wfield_local import run_mode
    t = run_mode.cache_tag()
    return f"{label}__restnl" + (f"-{t}" if t else "")


def cache_path(label: str, s: dict | None = None) -> Path:
    s = s or session_of(label)
    return cache_dir() / f"{_stem(label)}__{_key(s)}.npz"


def build_session(label: str, measure: bool = False) -> dict:
    """Compute one session's RESTnl baseline (and, with ``measure``, the flat-vs-time-local measurement) and write
    the cache npz. Returns a summary dict (no arrays)."""
    from wfield_local import joint_basis
    from wfield_local.position_reference_maps import MAP_SHAPE
    t0 = time.perf_counter()
    s = session_of(label)
    fp = cache_path(label, s)
    out = {"label": label}
    if fp.exists() and not measure:
        z = np.load(fp)
        return {"label": label, "cached": True, "ok_wr": bool(z["ok_wr"]), "nfr_pos": z["nfr_pos"].tolist()}
    try:
        mask, parts, info = lickfree_mask(s)
    except FileNotFoundError as ex:
        return {"label": label, "excluded": str(ex)}
    t_mask = time.perf_counter() - t0
    u, v = joint_basis._load_session(s["mc"])
    K, T = v.shape
    per = frames_by_position(s, T, mask, parts)
    pm, nfr = position_medians(v, per)
    b, ok, used = combine(pm, nfr, "wr")
    bs, oks, _u = combine(pm, nfr, "strict")
    base = (u @ np.asarray(b, np.float32)).reshape(MAP_SHAPE).astype(np.float32)
    t_all = time.perf_counter() - t0
    out.update(info)
    out.update({"cached": False, "ok_wr": ok, "ok_strict": oks, "positions_used": used, "nfr_pos": nfr.tolist(),
                "K": int(K), "T": int(T), "t_mask_s": round(t_mask, 2), "t_total_s": round(t_all, 2)})
    cache_dir().mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, version=VERSION, label=label, b_nl=np.asarray(b, float), b_strict=np.asarray(bs, float),
                        ok_wr=ok, ok_strict=oks, posmed=pm, nfr_pos=nfr, positions_used=np.asarray(used, int),
                        base_map=base, info=json.dumps(info))
    os.replace(tmp, fp)
    for old in cache_dir().glob(f"{_stem(label)}__*.npz"):         # prune superseded keys of THIS mode's cache only
        if old != fp:
            try:
                old.unlink()
            except OSError:
                pass
    if measure:
        try:
            out["measure"] = measure_flat_vs_timelocal(s, u, v, per, b)
        except Exception as ex:                                        # noqa: BLE001
            out["measure_error"] = f"{type(ex).__name__} {str(ex)[:120]}"
    return out


def baseline(label: str) -> dict | None:
    """The cached RESTnl baseline of one session (``{"b_nl", "base_map", "ok_wr", "nfr_pos", ...}``), or None when it
    has not been built, the session has no licks_all.csv, or rule wr fails. Never computes (call `build`)."""
    try:
        fp = cache_path(label)
    except StopIteration:
        return None
    if not fp.exists():
        return None
    z = np.load(fp)
    if not bool(z["ok_wr"]):
        return None
    return {k: z[k] for k in z.files}


def baseline_or_build(label: str) -> dict | None:
    """`baseline`, BUILDING it first when its cache file is absent (T4, third review: a key change -- yaml / helper code
    -- makes every old baseline invisible; it must be rebuilt, never read as "no baseline"). None only for the real
    reasons: the session is excluded (no licks_all) or rule wr fails."""
    d = baseline(label)
    if d is not None:
        return d
    try:
        fp = cache_path(label)
    except StopIteration:
        return None
    if not fp.exists():
        build_session(label)
    return baseline(label)


def base_maps(labels) -> tuple[dict, list]:
    """``({label: base_map}, [labels without a usable RESTnl baseline])`` -- what `maps_by_epoch` attaches. Missing
    baselines are BUILT (`baseline_or_build`)."""
    got, miss = {}, []
    for lab in labels:
        d = baseline_or_build(lab)
        if d is None:
            miss.append(lab)
        else:
            got[lab] = np.asarray(d["base_map"], np.float32)
    return got, miss


# ------------------------------------------------------------------------------------------------ flat vs time-local
def measure_flat_vs_timelocal(s, u, v, per, b_nl, nb: int = 12) -> dict:
    """The check doc's chk2 measure (section 4), on one session: does a FLAT baseline differ from the page's TIME-LOCAL
    REST in what the REST-referenced maps see? Uses the cue/working contact `prm` cache entry for raw / raw_rest /
    raw_restw (read through `session_raw_maps_cached`). Values x1e-3 dF/F over the brain mask."""
    from wfield_local import position_reference_maps as prm
    from wfield_local.quiet_periods import quiet_frame_path
    from wfield_local.rest_engagement import engaged_frame_mask
    for k in ("WIDEFIELD_LICK_SOURCE", "WIDEFIELD_ENL_WINDOW_S"):
        if os.environ.get(k):
            raise RuntimeError(f"measure needs the contact arm: unset {k}")
    d = prm.session_raw_maps_cached(s, "cue", post_s=2.0, variant="working")
    if not d.get("raw_rest"):
        return {"skipped": "no REST maps"}
    T = v.shape[1]
    brain = np.abs(u).sum(1) > 0
    sc = 1e3

    def pm(bb):
        return (u @ np.asarray(bb, np.float32).reshape(-1)).reshape(-1)

    def rms(x):
        return float(np.sqrt(np.mean(np.asarray(x)[brain] ** 2)) * sc)

    q = np.load(quiet_frame_path(s["mc"])).astype(bool)
    qm = np.zeros(T, bool)
    L = min(len(q), T)
    qm[:L] = q[:L]
    e, _ = engaged_frame_mask(s, T)
    qm &= e
    edges = np.linspace(0, T, nb + 1)

    def binmaps(mask):
        qi = np.flatnonzero(mask)
        outm, nn = [], []
        for k in range(nb):
            sel = qi[(qi >= edges[k]) & (qi < edges[k + 1])]
            nn.append(int(sel.size))
            outm.append(pm(np.median(v[:, sel], 1)) if sel.size >= 20 else np.full(u.shape[0], np.nan))
        return np.stack(outm), nn

    r = {}
    BM, nbin = binmaps(qm)
    nlmask = np.zeros(T, bool)
    for c in per:
        nlmask[np.asarray(per[c])[np.asarray(per[c]) < T]] = True
    BN, nbin_nl = binmaps(nlmask)
    r["rest_frames_per_bin"], r["restnl_frames_per_bin"] = nbin, nbin_nl
    for tag, B in (("rest", BM), ("restnl", BN)):
        Bb = B[:, brain]
        okb = np.isfinite(Bb).all(1)
        if okb.sum() >= 2:
            sdb = np.nanstd(Bb[okb], 0)
            rg = np.nanmax(Bb[okb], 0) - np.nanmin(Bb[okb], 0)
            r[f"{tag}_bin_sd_med"] = float(np.median(sdb) * sc)
            r[f"{tag}_bin_sd_p99"] = float(np.percentile(sdb, 99) * sc)
            r[f"{tag}_bin_range_med"] = float(np.median(rg) * sc)
            bm = np.nanmean(Bb, 1)
            okk = np.isfinite(bm)
            r[f"{tag}_slope"] = float(np.polyfit(np.arange(nb)[okk] / (nb - 1), bm[okk], 1)[0] * sc)
            r[f"{tag}_n_empty_bins"] = int((~okb).sum())
    flat_rest = pm(np.median(v[:, qm], 1))
    qs = [x for x in d["raw_rest"] if x in d["raw"]]
    R = {x: d["raw_rest"][x].reshape(-1) for x in qs}
    allq = np.stack([R[x] for x in qs])
    con = allq - allq.mean(0)
    fvt = np.stack([(d["raw"][x].reshape(-1) - R[x]) - flat_rest for x in qs])
    common = fvt.mean(0)
    base_nl = pm(b_nl)
    base_w = None
    if d.get("raw_restw") and qs[0] in d["raw_restw"]:
        base_w = d["raw"][qs[0]].reshape(-1) - d["raw_restw"][qs[0]].reshape(-1)
    for i, x in enumerate(qs):
        r[f"{x}_rest_map_rms"] = rms(R[x])
        r[f"{x}_contrast_rms"] = rms(con[i])
        r[f"{x}_flat_minus_tl_rms"] = rms(fvt[i])
        r[f"{x}_posdep_rms"] = rms(fvt[i] - common)
        # what the map itself becomes under RESTnl, and how far that is from REST
        nlm = d["raw"][x].reshape(-1) - base_nl
        r[f"{x}_restnl_map_rms"] = rms(nlm)
        r[f"{x}_restnl_minus_rest_rms"] = rms(nlm - R[x])
    r["flat_minus_tl_common_rms"] = rms(common)
    if base_w is not None:
        r["restnl_minus_restw_base_rms"] = rms(base_nl - base_w)
        r["restnl_minus_restw_base_p99"] = float(np.percentile(np.abs(base_nl - base_w)[brain], 99) * sc)
    r["positions"] = qs
    return r


# ------------------------------------------------------------------------------------------------ driver
def _job(item):
    label, measure = item
    try:
        return build_session(label, measure=measure)
    except Exception as ex:                                            # noqa: BLE001
        return {"label": label, "error": f"{type(ex).__name__} {str(ex)[:160]}"}


def default_labels() -> list:
    """Every curated cohort-1 session (June + August pre and post) -- the drift figure needs June."""
    from wfield_local import analysis_kit as ak
    if os.environ.get("WIDEFIELD_PRE_FROM"):
        raise SystemExit("unset WIDEFIELD_PRE_FROM: baselines are built for every curated session (June included)")
    return ak.curated_labels()


def build(labels=None, jobs: int = 4, measure: bool = False):
    """Fan `build_session` over sessions (sorted results, rule 6/9). Returns the list of summaries."""
    from wfield_local import analysis_kit as ak
    labels = list(labels or default_labels())
    res, fails = ak.fan_sessions([(x, measure) for x in labels], _job, jobs=min(int(jobs), 4), key=lambda it: it[0],
                                 label="restnl")
    out = [r for _it, r in res]
    for it, err in fails:
        out.append({"label": it[0], "error": str(err)[:200]})
    return sorted(out, key=lambda r: r["label"])


def validate(rows) -> dict:
    """Frame counts vs wf_attempt's restw_check `strict__nfr_pos` (the same mask and frame selection -> must be EQUAL)."""
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local import run_mode
    # restw_strict_check writes <iti_licks.OUT>/restw_check = run_mode.subdir(<imaging_encoding>/iti_licks) (11-04)
    src = run_mode.subdir(oo.encoding_root(oc.RUN, PathResolver()) / "iti_licks") / "restw_check" / "sessions"
    eq, ne, absent = [], [], []
    for r in rows:
        if "nfr_pos" not in r:
            continue
        f = src / f"{r['label']}.npz"
        if not f.exists():
            absent.append(r["label"])
            continue
        z = np.load(f)
        theirs = np.asarray(z["strict__nfr_pos"], int).tolist()
        (eq if theirs == list(r["nfr_pos"]) else ne).append((r["label"], list(r["nfr_pos"]), theirs))
    return {"equal": len(eq), "differ": ne, "not_in_restw_check": absent, "src": str(src)}


def main(argv=None) -> int:
    import argparse

    from wfield_local import parallel
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--measure", action="store_true", help="also run the flat-vs-time-local measurement per session")
    ap.add_argument("--out", type=Path, default=None, help="JSON summary path (default: <cache_dir>/build_summary.json)")
    a = ap.parse_args(argv)
    parallel.pin_blas()
    t0 = time.time()
    rows = build(a.labels, a.jobs, a.measure)
    v = validate(rows)
    excl = [r["label"] for r in rows if r.get("excluded")]
    err = [(r["label"], r["error"]) for r in rows if r.get("error")]
    fail_wr = [r["label"] for r in rows if "ok_wr" in r and not r["ok_wr"]]
    print(f"[restnl] {len(rows)} sessions in {(time.time() - t0) / 60:.1f} min; excluded (no licks_all.csv): {excl}; "
          f"rule wr fails: {fail_wr}; errors: {err}", flush=True)
    print(f"[restnl] frame counts vs restw_check strict__nfr_pos: {v['equal']} equal, {len(v['differ'])} differ "
          f"{v['differ'][:5]}; {len(v['not_in_restw_check'])} not in restw_check", flush=True)
    outp = a.out or (cache_dir() / ("build_summary_measure.json" if a.measure else "build_summary.json"))
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"rows": rows, "validate": v, "excluded": excl, "fail_wr": fail_wr, "errors": err},
                               indent=1, default=str), encoding="utf-8")
    print(f"[restnl] -> {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
