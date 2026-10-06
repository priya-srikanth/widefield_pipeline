"""What made each DAQ spout contact: a DLC lick, something without the tongue, or grooming.

Priya, 2026-10-06, on the 83 PS93 0814 contacts no DLC lick accounted for: "could these be grooming? look again at
the stroke_orofacial_pipeline. we can't use the simultaneous 2-spout contact rule, but we may be able to try some
others. however for close contacts it's theoretically possible that if water contacts the mouth the lick line gets
pulled down for a prolonged period with or without real licks".

stroke_orofacial (`spout_behavior/bouts.py find_grooming_bouts`) marked grooming where (a) both spouts were touched
within +-20 ms or (b) a touch lasted > 400 ms, then widened +-5 s and filled good gaps < 10 s. (a) needs two spouts;
(b) alone would flag a water bridge at a close spout during real licking. On PS93 0814: lick contacts last median
67 ms (p90 85, max 168); the one clear grooming bout (paws at the face in cam4 / cam1 / cam2) gave the longest contact
of the session, 323 ms, with no tongue; no contact exceeded 400 ms.

Classes (`classify`), per contact:
  lick         onset inside a kept DLC lick's rise start .. fall end (+- span_pad_ms) -- `tongue_kinematics`
               contact.match "span"
  no_tongue    not in a lick, and the cleaned tongue is not out (protrusion <= lip + tongue_out_px) within
               +-tongue_win_ms: lip / chin on a close spout, a water bridge, a paw, or a tongue the camera cannot see
               (cam4 only for now; cam1 can rescue hidden tongues once whole-session cam1 poses exist)
  tongue_other not in a kept lick but the tongue is out (a lick the detector rejected, or an edge case)
and GROOMING (`grooming_periods`): a contact lasting >= long_touch_ms during which the tongue is out on less than
long_touch_max_tongue_frac of the frames (a water bridge DURING licking keeps the tongue out, so it is spared),
widened by groom_widen_s. Thresholds are first guesses from one session -- check them across sessions
(`scripts/contact_survey.py`) before relying on them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = {"span_pad_ms": 8.0, "tongue_out_px": 10.0, "tongue_win_ms": 12.0, "long_touch_ms": 250.0,
            "long_touch_max_tongue_frac": 0.5, "groom_widen_s": [5.0, 5.0]}


def params(overrides: dict | None = None) -> dict:
    from wfield_local import config
    p = {**DEFAULTS, **(config.defaults().get("contact_classes", {}) or {})}
    return {**p, **(overrides or {})}


def contact_durations_ms(lick_v: np.ndarray, fs: float, onsets_s: np.ndarray, thresh_upper: float,
                         max_s: float = 5.0) -> np.ndarray:
    """How long the lick line stays below ``thresh_upper`` (the onset threshold) after each onset, ms."""
    below = np.asarray(lick_v) < float(thresh_upper)
    out = np.full(len(onsets_s), np.nan)
    n = int(max_s * fs)
    for k, t in enumerate(np.asarray(onsets_s, float)):
        i = int(round(t * fs))
        seg = below[i:i + n]
        if seg.size:
            j = np.argmax(~seg) if (~seg).any() else seg.size
            out[k] = j / fs * 1000.0
    return out


def session_durations(animal: str, date: str, onsets_s: np.ndarray, rv=None) -> np.ndarray:
    """Contact durations for one session's DAQ onsets (the lick_analog channel, configs `lick_detection`)."""
    from wfield_local import config, daq_io
    from wfield_local.paths import PathResolver
    rv = rv or PathResolver()
    h5 = sorted((Path(rv.root("daq_recorder_output")) / date).glob(f"{animal}_{date}_*.h5"))[0]
    with daq_io.open_daq(h5) as f:
        fs, _ = daq_io.session_attrs(f)
        v = daq_io.analog_channel(f, "lick_analog")
    return contact_durations_ms(v, float(fs), onsets_s, config.defaults()["lick_detection"]["thresh_upper"])


def in_spans(t: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Is each time inside any [lo, hi] span (spans need not be sorted)?"""
    t = np.asarray(t, float)
    o = np.argsort(lo)
    lo, hi = np.asarray(lo, float)[o], np.asarray(hi, float)[o]
    hi_run = np.maximum.accumulate(hi) if len(hi) else hi
    k = np.searchsorted(lo, t, side="right") - 1
    return (k >= 0) & (hi_run[np.clip(k, 0, max(len(hi_run) - 1, 0))] >= t) if len(lo) else np.zeros(len(t), bool)


def tongue_out_fraction(t0: np.ndarray, t1: np.ndarray, vt: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Fraction of video frames in each [t0, t1] with the tongue out (``out`` per frame at times ``vt``)."""
    o = np.argsort(vt)
    vt, out = np.asarray(vt, float)[o], np.asarray(out, bool)[o]
    a, b = np.searchsorted(vt, t0), np.searchsorted(vt, t1, side="right")
    return np.array([out[i:j].mean() if j > i else np.nan for i, j in zip(a, b)])


def classify(contact_s: np.ndarray, dur_ms: np.ndarray, lick_lo_s: np.ndarray, lick_hi_s: np.ndarray,
             vt: np.ndarray, tongue_out: np.ndarray, p: dict | None = None) -> pd.DataFrame:
    """One row per contact: class (lick / no_tongue / tongue_other), duration, tongue-out fraction, long_touch,
    grooming_touch. ``lick_lo_s / lick_hi_s`` = each kept lick's rise start / fall end (DAQ s); ``tongue_out`` per
    video frame at ``vt``."""
    p = params(p)
    c = np.asarray(contact_s, float)
    pad, w = p["span_pad_ms"] / 1000.0, p["tongue_win_ms"] / 1000.0
    is_lick = in_spans(c, np.asarray(lick_lo_s) - pad, np.asarray(lick_hi_s) + pad)
    near = tongue_out_fraction(c - w, c + w, vt, tongue_out)
    dur = np.asarray(dur_ms, float)
    during = tongue_out_fraction(c, c + np.nan_to_num(dur, nan=0.0) / 1000.0, vt, tongue_out)
    cls = np.where(is_lick, "lick", np.where(near > 0, "tongue_other", "no_tongue"))
    long_touch = dur >= p["long_touch_ms"]
    groom = long_touch & ~(during >= p["long_touch_max_tongue_frac"])
    return pd.DataFrame({"t_s": c, "class": cls, "dur_ms": dur, "tongue_out_near": near, "tongue_out_during": during,
                         "long_touch": long_touch, "grooming_touch": groom})


def grooming_periods(table: pd.DataFrame, p: dict | None = None) -> list[tuple[float, float]]:
    """[(t0, t1), ...] DAQ s: each grooming touch widened by groom_widen_s, overlapping periods merged."""
    p = params(p)
    before, after = p["groom_widen_s"]
    g = table[table.grooming_touch]
    iv = sorted((t - before, t + d / 1000.0 + after) for t, d in zip(g.t_s, g.dur_ms))
    out: list[list[float]] = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]
