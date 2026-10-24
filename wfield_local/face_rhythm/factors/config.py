"""Constants of the factor stages (one place; read by every stage and by `plan`)."""
from __future__ import annotations

EPS = 1e-12
TRIAL_DS = 2                                        # 25 -> 12.5 Hz for the fit rows only (default; plan --trial-ds)
CHUNK = 256                                         # rows per einsum chunk
PRECUE_S, RESP_S = (-2.0, 0.0), (0.0, 4.0)          # movement-power windows (s from cue)
BUSY_S = (-3.0, 6.0)                                # "between trials" = outside this around every engaged cue
EDGE_BINS = 10                                      # 2 s at 5 Hz kept out of the between-trial pool at each end
ARRIVAL_S = (-1.0, 2.0)                             # PRE-CUE rows: 5 Hz bins around each trial's position strobe (spout
PRECUE_EXT_S = -4.0                                 # arrival) + from cue - 4 s, both before the 25 Hz grid (cue - 2 s)
POINTS_CHUNK = 20000                                # 25 Hz bins per chunk when centring / projecting displacements
SIDECAR_MODULES = ("face_rhythm_timing", "face_rhythm_multiview")    # copied next to fr_factors.py by `plan` (O2)
EPOCH_GROUPS = ("pre", "acute", "subacute", "chronic", "preA", "preB")
