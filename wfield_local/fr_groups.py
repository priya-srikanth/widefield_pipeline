"""GROUPED FaceRhythm regressors for the encoding model (Priya 10-22: "FR components will also have to be dropped or
grouped with respect to licking (eg can pool all licking related factors as 1 'factor' amplitude for directionless lick
info, pool side-specific licking-related factors for direction info)").

Raw FR factors are side-specific (left / right / central licking), so entering them one by one gives the model lick
DIRECTION through FR, and "position beyond FR" becomes a lower bound by construction. Grouped:

  lick      SUM of every lick-locked factor (absolute expression, so amplitudes add): how much licking, whichever
            spout. Directionless only because it is balanced across directions -- dropping one direction's factor
            (e.g. central f4) would make the sum read low on those trials and carry direction (10-22 discussion)
  left / centre / right   side-pooled lick factors: lick DIRECTION through FR
  other     the remaining factors, one trace each (includes lick-INITIATION factors: not "non-lick")

Modes (the env value "<grouping>:<mode>", COMBINED_ENCODING_FR_GROUPS, read by `face_rhythm_outputs.fr_signals`):
  A = lick + left + centre + right + other   (everything FR says, direction included)
  B = lick + other                            (directionless FR: position beyond B = position beyond FR's lick amount)
Side information through FR = A - B.

GROUPINGS keys name a factor SET's labelling; factor numbers shift between fits, so a grouping is only valid for the
set it names (checked).
"""
from __future__ import annotations

import os

import numpy as np

ENV = "COMBINED_ENCODING_FR_GROUPS"

#: pilot_v1 (set pilot_pre_PS92_PS93; Priya's PROVISIONAL labels from the clips, 10-22 FR handoff), settled 10-22 by
#: lick-locking checks (lick-triggered average with a PRE-LICK baseline, -0.2 .. +0.4 s, SD units; lick-rate r with and
#: without running speed partialled out; PS92_0806 / 0812, PS93_0806 / 0812) and the FR window's 15-session identity
#: check (docs/FACE_RHYTHM_FINDINGS.md sec 4):
#:   lick amplitude  f0 (.31-.64), f4 (.62-1.39), f5 (.35-.74), f9 (.40-1.00): lick-locked in every check
#:   other           f2 -- elevated DURING licking (lick-rate r +.04-.24) but not locked to the licks (.12-.20; a first
#:                   check with a random-time baseline read the tonic elevation as locking) and its FR-window peak is
#:                   at RUNNING onset (lick-rate r -0.24 in their readouts); f1 / f3 / f6 -- lick INITIATION (peak
#:                   0.25-0.4 s before the first licks of a bout; f6 locks .43-.61 but lick-rate r ~ 0), so "other"
#:                   is NOT "non-lick"; f7 (running onset, 1.3-1.8), f8. Priya: "ok" (10-22).
#: Factor numbers belong to this fit only: master_pre_v1 renumbers them -- never carry this mapping over.
GROUPINGS = {
    "pilot_v1": {"set": "pilot_pre_PS92_PS93",
                 "lick": (0, 4, 5, 9), "left": (0, 5), "centre": (4,), "right": (9,), "other": (1, 2, 3, 6, 7, 8)},
}
MODES = {"A": ("lick", "left", "centre", "right", "other"), "B": ("lick", "other")}


def spec() -> tuple[str, str] | None:
    """(grouping, mode) from the environment, or None."""
    v = os.environ.get(ENV)
    if not v:
        return None
    g, _, m = v.partition(":")
    if g not in GROUPINGS or m not in MODES:
        raise ValueError(f"{ENV}={v!r}: expected <{'|'.join(GROUPINGS)}>:<{'|'.join(MODES)}>")
    return g, m


def group(sig: dict, grouping: str, mode: str, set_name: str | None = None) -> dict:
    """Raw factor traces {"fr_<k>": (t, values)} -> grouped {"fr_lick" | "fr_left" | ... | "fr_other_<k>": (t, values)}.
    A sum is NaN wherever any of its factors is (blanked frames stay blanked)."""
    G = GROUPINGS[grouping]
    if set_name is not None and set_name != G["set"]:
        raise ValueError(f"FR grouping {grouping!r} is for set {G['set']!r}, not {set_name!r} (factor numbers shift)")
    names = list(sig)
    idx = {int(n.rsplit("_", 1)[1]): n for n in names if n.rsplit("_", 1)[-1].isdigit()}
    need = {k for part in ("lick", "left", "centre", "right", "other") for k in G[part]}
    missing = sorted(need - set(idx))
    if missing:
        raise ValueError(f"FR grouping {grouping!r}: factors {missing} not in {names}")
    t = sig[names[0]][0]
    out = {}
    for part in MODES[mode]:
        if part == "other":
            for k in G["other"]:
                out[f"fr_other_{k}"] = (t, np.asarray(sig[idx[k]][1], float))
        else:
            out[f"fr_{part}"] = (t, np.sum([np.asarray(sig[idx[k]][1], float) for k in G[part]], axis=0))
    return out
