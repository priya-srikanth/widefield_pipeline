"""Item 9 of PROMPT_engagement_movement_both_cohorts.md: `manuscript_code/data/fr_pre_treadmill.npz` for Supplementary S13,
on EXACTLY the sessions and trials of `fr_pre_timecourses_abs.npz` (2026-11-20).

The FaceRhythm amplitude timecourses come from the frozen-master per-session task files
(`face_rhythm_outputs.factor_dir(PathResolver(), BUNDLE, SET)/proj/*.npz`, master_pre_v1: engaged trials, cue -2..+4 s,
25 Hz, 151 samples; enumerated by `face_rhythm.common.sessions(fdir, 'proj')`, sorted by file name). Here, for each
PRE-stroke proj session in that order and each spout position, the cue-aligned FORWARD treadmill speed (mm/s;
`behavior_states.load_session`, the states page's loader, units and sign) is sampled at each of that session's trials'
cue_daq_s + t_ms and averaged over the session's trials at that position -> (sessions x 151) per position, layout as
the FaceRhythm file; `sessions_<pos>` keys hold the session labels; the session counts are asserted equal to the
FaceRhythm file's. Supersedes review/c2_treadmill_2026-11-20/superseded/fr_pre_treadmill_collector_sessions_v1.npz
(collector sessions, 31 not 32).

Usage: MC=<manuscript_code> python scripts/c2_s13_treadmill_from_fr.py

Repo copy (2026-11-20) of review/c2_treadmill_2026-11-20/c2_s13_treadmill_from_fr.py; only the import path changed (+ lint-only fixes).
Output (refuses to overwrite): $MC/data/fr_pre_treadmill.npz (MC = vls_stroke_paper/manuscript_code).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this repo (the review copy used $WF_REPO)
import numpy as np  # noqa: E402
from wfield_local import behavior_states as B  # noqa: E402
from wfield_local import face_rhythm_outputs as fro  # noqa: E402
from wfield_local.face_rhythm import common as frc  # noqa: E402
from wfield_local.paths import PathResolver  # noqa: E402

MC = Path(os.environ["MC"]); OUT = MC / "data" / "fr_pre_treadmill.npz"
if OUT.exists():
    sys.exit(f"refusing to overwrite {OUT}")
POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]
FR = np.load(MC / "data" / "fr_pre_timecourses_abs.npz")
fdir = fro.factor_dir(PathResolver(), fro.BUNDLE, fro.SET)
T = frc.sessions(fdir, "proj"); T = T[T.epoch == "pre"]
per = {p: [] for p in POS}; names = {p: [] for p in POS}
for r in T.itertuples():
    with np.load(r.file) as z:
        t_ms = z["t_ms"].astype(float); cues = z["cue_daq_s"].astype(float); pos = z["position"].astype(str)
    assert np.allclose(t_ms, FR["t_ms"]), "time grid differs from the FaceRhythm file"
    d = str(r.date); lab = f"{r.animal}_{d if len(d) == 8 else '2026' + d[-4:]}"   # FR labels PS92_0806 -> PS92_20260806
    t, fwd, _back, _z = B.load_session(lab, None)
    for p in POS:
        m = pos == p
        if not m.any():
            continue
        X = np.stack([np.interp(c + t_ms / 1000.0, t, fwd, left=np.nan, right=np.nan) for c in cues[m]])
        per[p].append(np.nanmean(X, 0)); names[p].append(lab)
    print("done", lab, flush=True)
arrs = {p: np.asarray(per[p], np.float32) for p in POS}
cmp = {p: (arrs[p].shape[0], FR[p].shape[0]) for p in POS}
print("sessions per position (treadmill, FaceRhythm):", cmp, flush=True)
assert all(a == b for a, b in cmp.values()), "session counts differ from the FaceRhythm file"
np.savez(OUT, t_ms=FR["t_ms"], **arrs, **{f"sessions_{p}": np.asarray(names[p]) for p in POS})
print("wrote", OUT, flush=True)
