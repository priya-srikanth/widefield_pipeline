"""Quiet-period detection for baseline (F0) selection.

Builds a per-sample (and per-corrected-frame) "quiet" mask: times when the animal
is NOT running and NOT licking (and not in a peri-reward window). These frames are a
behavior-controlled baseline for event-triggered ΔF/F — e.g. intersect with the
pre-cue ENL window, or pool as an F0, instead of relying on a true inter-trial quiet
period (which trial-triggered acquisition doesn't record).

Ported from the stroke_orofacial_pipeline (`spout_behavior/bouts.py::find_quiet_bouts`):
    quiet = slow-treadmill  AND  not-near-lick  AND  not-near-reward  [AND not grooming]
with each exclusion widened by a time buffer, then short quiet runs dropped.

Adapted for THIS rig:
- ONE spout, so the stroke pipeline's grooming detector (bilateral two-spout
  conjunction) does not apply. Grooming here would rely only on single-spout
  "long-touch" contact, but with close spouts a TRUE long lick can produce a long
  deflection -> long-touch is an unreliable grooming proxy. So grooming is OFF by
  default; enable with --grooming only as an experiment (see caveat below).
- The quiet exclusions use running (treadmill) + licking (lick_analog) + reward.

!! TUNE LATER: the defaults below (running/quiet speed, min durations, time buffers,
lick thresholds) are starting points carried over from the stroke pipeline. Revisit
"running" speed/duration and the lick/reward/treadmill buffers for this rig and task
once we have ground-truth (e.g. DLC/FaceRhythm movement) to validate against. The
stroke `min_quiet_duration` was 10 s (for long rest bouts); here it defaults small so
the mask is usable as a per-frame baseline cleanliness flag within short ENL windows.

Vendored boolean helpers (idx2bool / widen_bool_sparse / set_short_bool_to_low) are
MIT-licensed ports from bnpm (© 2021 RichieHakim) via the stroke pipeline.

Runs in the wfield CPU env (numpy + scipy + h5py; no torch/GPU).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wfield_local import daq_io

from wfield_local import config
from wfield_local.treadmill import calibrate_treadmill, smooth_treadmill
from wfield_local.lick_detection import detect_licks


# --- vendored boolean helpers (MIT, bnpm © 2021 RichieHakim; via stroke pipeline) ---
def idx2bool(idx: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=bool)
    idx = np.asarray(idx, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < length)]
    out[idx] = True
    return out


def widen_bool_sparse(b: np.ndarray, n_before: int, n_after: int) -> np.ndarray:
    b = np.asarray(b).astype(bool); n = b.shape[0]
    out = b.copy()
    for i in np.flatnonzero(b):
        out[max(0, i - n_before):min(n, i + n_after + 1)] = True
    return out


def set_short_bool_to_low(b: np.ndarray, n: int) -> np.ndarray:
    """Zero True runs shorter than n samples (runs of length == n are kept)."""
    b = np.asarray(b).astype(bool).copy()
    if b.size == 0:
        return b
    pad = np.concatenate(([False], b, [False])).astype(np.int8)
    d = np.diff(pad)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)  # exclusive
    for s, e in zip(starts, ends):
        if (e - s) < n:
            b[s:e] = False
    return b


#: Which quiet-mask variant every consumer reads, from `segmentation.quiet.variant`. An empty
#: string means the original `quiet_<tag>` directory; anything else selects `quiet_<tag>_<variant>`.
#:
#: WHY A VARIANT RATHER THAN AN OVERWRITE (2026-09-12). The quiet definition changed -- the 8 s
#: post-reward buffer was a carryover from the stroke_orofacial pipeline, whose task runs on an 8 s
#: post-tone window; ours has a 3.5 s response window, and Priya's framing is that quiet should be
#: "non-running ITI frames", anchored on the CUE rather than on the reward. Rewriting the existing
#: masks would have made every figure built on them unreproducible and un-diffable at once, across
#: the QUIET reference, the state decoder, the position encoder and deck sections A-C. So this
#: follows the rule `docs/PREPROCESSING_DECISION.md` already sets for the hemodynamic variants:
#: nothing overwrites the original, every alternative gets its own directory beside it, and a
#: manifest says which definition produced it. Retiring a variant is then a config edit, and the
#: comparison stays available -- which matters here, because the change moves a lot of results and
#: "did it move because of this?" has to remain answerable.
def response_window_s(session_dir=None):
    """The session's real response window (s), or the configured default.

    READ PER SESSION, because it is a task setting that can be retuned -- `daq_trials` already does
    this and the two must not disagree. Every session to date ran 3500 ms.
    """
    from wfield_local import config as _cfg

    default = float(_cfg.defaults()["decode"]["max_rt_s"])
    if session_dir is None:
        return default, "default"
    try:
        from wfield_local.daq_trials import response_window_s as _rw

        return _rw(session_dir, default)
    except Exception:                                                  # noqa: BLE001
        return default, "default"


def trial_exclusion(n, fs, cue_s, trial_start_s, strobe_s, *, params=None, session_dir=None):
    """Boolean, True where a sample is INSIDE a trial and so may not be rest.

    THE TRIAL, NOT THE REWARD (2026-09-12). The retired definition excluded a fixed 8 s after every
    reward, which made the category track the animal's PERFORMANCE: a post-stroke mouse that misses
    more has less of its session buffered out, and "quiet" measured 4.4% of frames pre-stroke
    against 17.1% acutely. Anchoring on the trial removes that coupling by construction, because the
    spout moves and the cue plays whether or not the animal succeeds. Measured acute/pre: 3.89 -> 1.10.

    A trial occupies ``[trial_start, cue + response_window + settle]``:

    * IT OPENS AT `trial_start`, NOT AT THE STROBE. Firmware `startTrial` moves the spout and only
      THEN pulses the strobe, so the strobe fires AFTER the movement -- a window ending at the
      strobe contains the spout's travel. `trial_start` precedes the strobe on 16,607 of 16,607
      trials by a median 0.925 s (p1 0.623, p99 1.043), which IS the travel time. Where a session
      has no `trial_start` bit, the opening falls back to ``strobe - strobe_fallback_s`` and the
      caller is told, because that is a different definition and must not pass silently.
    * IT CLOSES AT cue + response_window + settle. See `settle_s` in the config for why 0.5 s: the
      settle guards nothing else misses -- reward arrives 6 ms after the cue, so the trial window
      already contains it -- and 0.5 s leaves 2.02 s of rest per trial against 1.52 s at 1.0 s.

    Returns ``(mask, note)``; `note` names the opening actually used.
    """
    from wfield_local import config as _cfg

    p = dict(params or _cfg.defaults()["segmentation"]["rest"])
    rw, _src = response_window_s(session_dir)
    settle = float(p.get("settle_s", 0.5))

    cue = np.asarray(cue_s, float)
    ts = np.asarray(trial_start_s if trial_start_s is not None else [], float)
    if ts.size:
        opens, note = ts, "trial_start"
    else:
        # NO trial_start BIT. Fall back to the strobe pulled back by the measured travel time, and
        # say so -- this is a DIFFERENT definition and a session using it must be identifiable.
        opens = np.asarray(strobe_s, float) - float(p.get("strobe_fallback_s", 1.0))
        note = f"strobe-{float(p.get('strobe_fallback_s', 1.0)):g}s (no trial_start bit)"

    mask = np.zeros(int(n), bool)
    closes = cue + rw + settle
    # ALL TIMES ARE SECONDS; samples are seconds * fs. This read `a / sr * fs` with `sr` defaulting
    # to `fs`, which is algebraically just `a` -- so a time in seconds was used directly as a sample
    # index and only the first few thousand samples of each session were ever marked in-trial. The
    # symptom was rest measuring 62% of an acute session, against 36% for the variant that applies
    # NO trial exclusion at all: an impossible number, which is why the smoke test existed.
    j = np.searchsorted(opens, cue, side="right") - 1
    for k in range(cue.size):
        a = opens[j[k]] if j[k] >= 0 else cue[k] - rw
        i0 = int(max(0, round(a * fs)))
        i1 = int(min(n, round(closes[k] * fs)))
        if i1 > i0:
            mask[i0:i1] = True
    return mask, note


def rest_mask(n, fs, speed, lick_onsets, cue_s, trial_start_s, strobe_s, *,
              params=None, session_dir=None):
    """``(mask, note)`` -- the REST mask. THE SINGLE DEFINITION every consumer resolves to.

    Two modules used to compute this independently -- this one from argparse literals and
    `behavior_events` from the config -- with a comment in the second claiming they agreed. They
    could not be made to disagree loudly, only quietly, and editing the config moved one of them.
    Both now call this.

    REST = inside no trial, AND slow treadmill (buffered), AND away from licking. There is no reward
    term: reward is simultaneous with the cue, so the trial window already contains it, and a reward
    term would couple the category to how often the animal earned water.
    """
    from wfield_local import config as _cfg

    p = dict(params or _cfg.defaults()["segmentation"]["rest"])
    in_trial, note = trial_exclusion(n, fs, cue_s, trial_start_s, strobe_s, params=p,
                                     session_dir=session_dir)

    def wid(b, buf):
        return widen_bool_sparse(b, int(buf[0] * fs), int(buf[1] * fs))

    slow = np.asarray(speed) < float(p["speed_mm_s"])
    rest = (~in_trial
            & ~wid(~slow, p["treadmill_buffer_s"])
            & ~wid(idx2bool(np.asarray(lick_onsets, np.int64), int(n)), p["lick_buffer_s"]))
    rest = set_short_bool_to_low(rest, int(float(p["min_rest_s"]) * fs))
    return rest, note


def quiet_variant():
    """The configured variant string, or "" for the original masks."""
    return str(config.defaults()["segmentation"]["rest"].get("variant", "") or "")


def quiet_dir(mc, variant=None, tag=None):
    """Directory holding this session's quiet masks for the selected variant."""
    if tag is None:
        tag = config.defaults()["preprocess"]["maps"]["tag"]
    v = quiet_variant() if variant is None else str(variant or "")
    return f"{mc}/quiet_{tag}" + (f"_{v}" if v else "")


def quiet_frame_path(mc, variant=None, tag=None, fallback=False):
    """Path to the per-corrected-frame quiet mask, or None.

    RESOLVES IN ONE PLACE. Six modules used to glob `{mc}/quiet_affine8v1/*quiet_frame.npy`
    independently -- `position_reference_maps`, `locanmf_position_encoder`,
    `locanmf_cue_lick_analysis`, `preprocess`, `roi_activity` and the lick-aligned pair -- so
    selecting a different definition would have meant editing six literals and hoping none was
    missed.

    **`fallback` DEFAULTS TO FALSE, AND THAT IS THE WHOLE POINT.** An earlier version of this
    defaulted to True, reasoning that a partially recomputed cohort should "degrade per session
    rather than per figure". That is backwards: falling back means a pooled map averages two
    DIFFERENT DEFINITIONS of its own subtrahend, with nothing on the figure to say so -- which is
    precisely the failure the variant directories exist to prevent, reintroduced one level down. A
    session without the selected variant loses its rest column instead, the same way a session with
    no mask at all already does, and the count is reported.
    """
    import glob

    for v in ([variant] if variant is not None else [quiet_variant()]) + ([""] if fallback else []):
        hits = sorted(glob.glob(f"{quiet_dir(mc, v, tag)}/*quiet_frame.npy"))
        if hits:
            return hits[0]
    return None


def quiet_variant_used(mc, variant=None, tag=None):
    """``""`` or the variant name actually found for this session -- None if neither exists."""
    import glob

    for v in ([variant] if variant is not None else [quiet_variant()]) + [""]:
        if sorted(glob.glob(f"{quiet_dir(mc, v, tag)}/*quiet_frame.npy")):
            return v
    return None


def quiet_baseline_svt(svt: np.ndarray, quiet_frame: np.ndarray) -> np.ndarray:
    """Mean SVT (K,) over the quiet frames -> the quiet-period baseline in SVD space.

    Subtract `U @ quiet_baseline_svt(...)` from an event-aligned map to express
    activity relative to the not-running/not-licking quiet baseline instead of the
    session mean. `quiet_frame` is the per-corrected-frame mask from this module;
    it is clipped/padded to the SVT length.
    """
    T = svt.shape[1]
    qf = np.asarray(quiet_frame).astype(bool)
    qf = qf[:T] if qf.size >= T else np.pad(qf, (0, T - qf.size))
    idx = np.flatnonzero(qf)
    if idx.size == 0:
        raise ValueError("no quiet frames in mask")
    return np.asarray(svt[:, idx]).mean(axis=1).astype(np.float32)


def _rising(sig: np.ndarray, thr: float = 0.5) -> np.ndarray:
    # an animal ALREADY running at sample 0 has no bout onset to report -- see daq_io.rising_edges
    return daq_io.rising_edges(sig, thr=thr, include_first_sample=False)


def _runs_at_least(contact: np.ndarray, n: int) -> np.ndarray:
    """Keep only True runs of length >= n (the long-touch violations)."""
    return set_short_bool_to_low(contact, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daq-h5", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label", required=True)
    # per-corrected-frame mapping (regime B = cleanpairs frame-map; else regime A raw//2)
    ap.add_argument("--frame-map", type=Path, default=None)
    ap.add_argument("--cleanpairs-summary", type=Path, default=None)
    ap.add_argument("--offset", type=int, default=None)
    # EVERY DEFAULT BELOW COMES FROM configs/defaults.yaml (rule 3, single source of truth).
    #
    # THEY USED TO BE LITERALS HERE, and that was a live bug rather than a style problem. This
    # module writes the per-corrected-frame `*_quiet_frame.npy` that the QUIET reference reads;
    # `behavior_events` computes the SAME quantity from `segmentation.quiet` in the config, and its
    # own comment claims the two agree. `preprocess.py` invokes this module passing none of these
    # flags, so it always used the literals. Editing the config therefore moved the events npz and
    # left the per-frame masks untouched -- two definitions of "quiet" under one name, differing
    # silently, with nothing in either output saying which had been used. Found 2026-09-12 while
    # changing the reward buffer.
    seg = config.defaults()["segmentation"]
    tr, qd = seg["treadmill"], seg["rest"]
    ap.add_argument("--tread-channel", default=tr["channel"])
    ap.add_argument("--offset-v", type=float, default=tr["offset_v"])
    ap.add_argument("--volt-sec-per-rot", type=float, default=tr["volt_sec_per_rot"])
    ap.add_argument("--mm-per-rot", type=float, default=tr["mm_per_rot"])
    ap.add_argument("--smoothing-sigma-s", type=float, default=tr["smoothing_sigma_s"])
    ap.add_argument("--quiet-speed", type=float, default=qd["speed_mm_s"],
                    help="mm/s; below = 'slow'")
    ap.add_argument("--treadmill-buffer", type=float, nargs=2,
                    default=tuple(qd["treadmill_buffer_s"]))
    # licking
    ap.add_argument("--lick-channel", default="lick_analog")
    ld = config.defaults()["lick_detection"]
    ap.add_argument("--lick-thresh-upper-v", type=float, default=ld["thresh_upper"])
    ap.add_argument("--lick-thresh-lower-v", type=float, default=ld["thresh_lower"])
    ap.add_argument("--lockout-s", type=float, nargs=2, default=tuple(ld["lockout_falling_edge_s"]))
    ap.add_argument("--refractory-s", type=float, default=0.10)
    ap.add_argument("--lick-buffer", type=float, nargs=2, default=tuple(qd["lick_buffer_s"]))
    # reward
    ap.add_argument("--reward-channel", default=seg["reward"]["channel"])
    ap.add_argument("--reward-thresh-v", type=float, default=seg["reward"]["thresh_v"])
    # NO reward buffer: rest is now anchored on the TRIAL, and reward is simultaneous
    # with the cue, so the trial window already contains it.
    # grooming (OFF by default; unreliable with one close spout)
    ap.add_argument("--grooming", action="store_true",
                    help="EXPERIMENTAL: exclude single-spout long-touch as grooming. "
                         "Caveat: a true long lick at close spouts also looks long.")
    ap.add_argument("--groom-contact-thresh-v", type=float, default=2.5)
    ap.add_argument("--groom-max-long-s", type=float, default=0.4)
    ap.add_argument("--groom-buffer", type=float, nargs=2, default=(6.0, 6.0))
    # output mask shaping
    ap.add_argument("--min-quiet-s", type=float, default=qd["min_rest_s"],
                    help="drop quiet runs shorter than this (stroke rest-bout default was 10 s)")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.daq_h5, "r") as f:
        fs = float(f.attrs["sample_rate_hz"])
        an = [s.decode() for s in f["analog/channel_names"][:]]
        di = [s.decode() for s in f["digital/channel_names"][:]]
        sc = f["analog/int16_scale_volts_per_count"][:]; of = f["analog/int16_offset_volts"][:]

        def ac(name):
            i = an.index(name)
            return f["analog/samples_int16"][:, i].astype(np.float32) * sc[i] + of[i]

        tread_v = ac(args.tread_channel)
        lick_v = ac(args.lick_channel)
        reward_v = ac(args.reward_channel)
        packed = f["digital/packed_samples"][:, 0]
    n = len(packed)
    pco = _rising((packed >> di.index("pco_exposure")) & 1)

    def wid(b, buf):
        return widen_bool_sparse(b, int(buf[0] * fs), int(buf[1] * fs))

    # running / slow treadmill
    speed = smooth_treadmill(calibrate_treadmill(tread_v, args.offset_v, args.volt_sec_per_rot, args.mm_per_rot),
                             fs, args.smoothing_sigma_s)
    slow = speed < args.quiet_speed

    # licks (onsets) + reward
    lick = detect_licks(lick_v, fs, args.lick_thresh_upper_v, args.lick_thresh_lower_v,
                        tuple(args.lockout_s), args.refractory_s,
                        min_ili_s=config.defaults()["lick_detection"]["min_ili_ms"] / 1000.0)
    lick_onsets = np.asarray(lick["lick_onsets"], dtype=np.int64)
    lick_bool = idx2bool(lick_onsets, n)
    reward_bool = idx2bool(_rising(reward_v, args.reward_thresh_v), n)

    # ONE DEFINITION, shared with `behavior_events` -- see `rest_mask`.
    cue_e = _rising((packed >> di.index("cue")) & 1) if "cue" in di else np.empty(0, int)
    ts_e = (_rising((packed >> di.index("trial_start")) & 1) if "trial_start" in di
            else np.empty(0, int))
    st_e = (_rising((packed >> di.index("spout_strobe")) & 1) if "spout_strobe" in di
            else np.empty(0, int))
    quiet, rest_note = rest_mask(n, fs, speed, lick_onsets, cue_e / fs, ts_e / fs, st_e / fs,
                                 params=qd, session_dir=args.daq_h5.parent)
    print(f"  .. rest anchored on {rest_note}", flush=True)

    groom_bool = np.zeros(n, dtype=bool)
    if args.grooming:
        contact = lick_v < args.groom_contact_thresh_v
        groom_bool = _runs_at_least(contact, int(args.groom_max_long_s * fs))
        quiet = set_short_bool_to_low(quiet & ~wid(groom_bool, args.groom_buffer),
                                      int(args.min_quiet_s * fs))



    # ---- map to corrected frames ----
    if args.frame_map is not None:
        from wfield_local.framemap_event_maps import _offset_from_summary
        offset = args.offset if args.offset is not None else _offset_from_summary(args.cleanpairs_summary)
        fm = np.load(args.frame_map)
        frame_samples = pco[np.clip(fm["original_frame_index_ch0"] + offset, 0, len(pco) - 1)]
        regime = "B(frame-map)"
    else:
        npairs = len(pco) // 2
        frame_samples = pco[np.arange(npairs) * 2]
        regime = "A(raw//2)"
    quiet_frame = quiet[np.clip(frame_samples, 0, n - 1)]

    np.save(args.output / f"{args.label}_quiet_sample.npy", quiet)
    np.save(args.output / f"{args.label}_quiet_frame.npy", quiet_frame)
    summary = {
        "label": args.label, "daq_h5": str(args.daq_h5), "fs": fs, "n_samples": int(n),
        "regime": regime, "n_frames": int(quiet_frame.size),
        "frac_quiet_sample": float(quiet.mean()), "frac_quiet_frame": float(quiet_frame.mean()),
        "n_licks": int(lick_onsets.size), "grooming_used": bool(args.grooming),
        # THE DEFINITION TRAVELS WITH THE MASK. `reward_buffer` is gone from this list because the
        # definition no longer has one -- rest is bounded by the TRIAL, and reward is simultaneous
        # with the cue, so the trial window already contains it. `rest_anchor` records whether the
        # trial's opening came from `trial_start` or from the strobe fallback, which are different
        # definitions and must be distinguishable on a mask that is already written.
        "rest_anchor": rest_note,
        "params": {k: getattr(args, k) for k in (
            "quiet_speed", "min_quiet_s", "treadmill_buffer", "lick_buffer",
            "smoothing_sigma_s", "lick_thresh_upper_v", "lick_thresh_lower_v", "refractory_s",
            "groom_max_long_s", "groom_buffer")},
        "rest_params": dict(qd),
        "tune_later": "running/rest speed, durations, and the lick/treadmill buffers are "
                      "starting points (stroke-pipeline defaults); revisit for this rig/task.",
        "grooming_caveat": "single-spout long-touch is an unreliable grooming proxy (a true long "
                           "lick at close spouts also looks long); OFF by default.",
    }
    (args.output / f"{args.label}_quiet_periods_summary.json").write_text(json.dumps(summary, indent=2, default=list))

    # ---- QC: a 120 s window of speed/lick/reward with quiet shaded ----
    t = np.arange(n) / fs
    w = slice(0, min(n, int(120 * fs)))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t[w], speed[w], lw=0.6, color="0.3", label="speed (mm/s)")
    ax.axhline(args.quiet_speed, color="green", lw=0.6, ls="--")
    for i in lick_onsets[(lick_onsets >= w.start) & (lick_onsets < (w.stop or n))]:
        ax.axvline(t[i], color="orange", lw=0.3, alpha=0.5)
    qw = quiet[w]
    ax.fill_between(t[w], 0, 1, where=qw, transform=ax.get_xaxis_transform(),
                    color="cyan", alpha=0.25, label="quiet")
    ax.set_xlabel("s"); ax.set_ylabel("mm/s"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{args.label} quiet periods (first 120 s) | quiet={quiet.mean()*100:.1f}% samples, "
                 f"{quiet_frame.mean()*100:.1f}% frames | grooming={'on' if args.grooming else 'off'}")
    fig.tight_layout(); fig.savefig(args.output / f"{args.label}_quiet_periods.png", dpi=130); plt.close(fig)

    print(json.dumps(summary, indent=2, default=list), flush=True)
    print(f"[{args.label}] quiet: {quiet.mean()*100:.1f}% samples, {quiet_frame.mean()*100:.1f}% frames ({regime})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
