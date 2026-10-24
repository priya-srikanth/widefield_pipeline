"""Licks for the FaceRhythm readouts: the LP (v36) whole-session lick table, bout classes, first-lick latency.

`lick_table` is defined in `face_rhythm_validation` (where the factors were named against licking) and re-exported
here so analysis code has one place to import lick handling from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wfield_local.face_rhythm_validation import lick_table  # noqa: F401  (re-export)

#: bout classes of `bout_classes` (Priya 10-24: why is close_R asymmetry > 0 before a bout starts?)
BOUT_CLASSES = ("first", "later", "leadin", "carry", "first_late")


def bout_classes(L: pd.DataFrame, leadin: float = 0.5, late_s: float = 0.6) -> dict[str, np.ndarray]:
    """Boolean masks over ``L`` (a `lick_table`, sorted by time) of the bout ONSETS (first lick of a v36 bout, inside
    an engaged trial's cue -2..+4 s) in each class:
      first       the trial's first bout
      later       every later bout of the trial
      leadin      >= ``leadin`` s without any lick before onset (no carry-over possible)
      carry       a lick < ``leadin`` s before onset
      first_late  first bouts starting >= ``late_s`` after the cue (a pre-onset window that is post-cue, pre-lick)"""
    t = L.daq_on_s.to_numpy(float)
    gap = np.r_[np.inf, np.diff(t)]                                    # time since the previous lick (any trial)
    onset = (L.n_in_bout == 1).to_numpy() & np.isfinite(L.t_from_cue.to_numpy(float))
    on = L[onset]
    first_bout = on.groupby("cue_idx").bout_idx.transform("min")
    first = np.zeros(len(L), bool)
    first[np.flatnonzero(onset)[(on.bout_idx == first_bout).to_numpy()]] = True
    return {"first": first, "later": onset & ~first, "leadin": onset & (gap >= leadin),
            "carry": onset & (gap < leadin), "first_late": first & (L.t_from_cue.to_numpy(float) >= late_s)}


def first_lick_latency(L: pd.DataFrame, n_cues: int) -> np.ndarray:
    """Per cue (index into the SORTED engaged cues the table was built with): time of the first lick at or after the
    cue (s), NaN when the trial has none inside cue 0..+4 s."""
    L = L[(L.t_from_cue >= 0) & (L.cue_idx >= 0)]
    lat = np.full(n_cues, np.nan)
    f = L.groupby("cue_idx").t_from_cue.min()
    lat[f.index.to_numpy(int)] = f.to_numpy()
    return lat
