"""Treadmill extras, six-position cohort (PROMPT_engagement_movement_both_cohorts.md items 7 and 9; 2026-11-20).

Item 9 -> `manuscript_code/data/fr_pre_treadmill.npz` (new file, next to fr_pre_timecourses_abs.npz, same layout):
  t_ms (151: -2000..+4000 ms, the FaceRhythm grid) and, per spout position (close_L, close_center, close_R, far_L,
  far_center, far_R), a (sessions x 151) array of cue-aligned FORWARD treadmill speed (mm/s): per August pre-stroke
  session, the mean over that position's working trials (inside the lick working period,
  `trial_classes.camera_engaged_period`), wheel trace from `behavior_states.load_session` (the states page's loader,
  units and sign), linearly interpolated at cue + t_ms. Extra keys `sessions_<position>` list the session labels (the
  FaceRhythm file has none, so the session match is checked by count only).
Item 7 (treadmill): the push / walking session readouts and tests on the FIRST N MIN of each session (t_min <= N, N =
  behavior_states.COMMON_SESSION_MIN = 95, as Fig 1.5), next to the unmatched ones; session lengths reported.

Usage: MC=<manuscript_code> python scripts/c2_treadmill_extras.py <out_dir>

Repo copy (2026-11-20) of review/c2_treadmill_2026-11-20/c2_treadmill_extras.py; only the import path changed (+ lint-only fixes).
Outputs (refuses to overwrite): <out_dir>/treadmill_tests_primary_first95.csv, treadmill_session_readouts_first95.csv,
session_lengths_min.csv, and $MC/data/fr_pre_treadmill.npz (item 9; superseded for S13 by c2_s13_treadmill_from_fr.py,
which uses the FaceRhythm sessions; this script's npz is review/.../superseded/fr_pre_treadmill_collector_sessions_v1.npz).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this repo (the review copy used $WF_REPO)
import numpy as np  # noqa: E402
from wfield_local import behavior_states as B  # noqa: E402
from wfield_local import orofacial_stats as st  # noqa: E402
from wfield_local import trial_classes as tc  # noqa: E402

OUT = Path(sys.argv[1]); MC = Path(os.environ["MC"])
NPZ = MC / "data" / "fr_pre_treadmill.npz"
files = [NPZ, OUT / "treadmill_tests_primary_first95.csv", OUT / "treadmill_session_readouts_first95.csv",
         OUT / "session_lengths_min.csv"]
if any(p.exists() for p in files):
    sys.exit("refusing to overwrite: " + ", ".join(str(p) for p in files if p.exists()))
POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]
t_ms = np.load(MC / "data" / "fr_pre_timecourses_abs.npz")["t_ms"].astype(float)

# ---- item 7: time-matched push / walk
C = B.load_collected(); T = C["trials"]
T = T[T.label.map(B.session_epoch).notna()].copy(); T["epoch"] = T.label.map(B.session_epoch)
L = T.groupby(["label", "animal", "epoch"]).t_min.max().rename("length_min").reset_index()
L.to_csv(OUT / "session_lengths_min.csv", index=False)
N = B.COMMON_SESSION_MIN
print(f"session length (min): min {L.length_min.min():.0f}, 10th pct {L.length_min.quantile(.1):.0f}, median "
      f"{L.length_min.median():.0f}; sessions shorter than N={N:g}: {int((L.length_min < N).sum())} of {len(L)} "
      f"(kept whole); animals per epoch with sessions: {L.groupby('epoch').animal.nunique().to_dict()}", flush=True)
S = B.per_session(T[~(T.t_min > N)], "lick_period")
S["pos"] = "all"; S["date"] = S.label.str.split("_").str[1]
S.to_csv(OUT / "treadmill_session_readouts_first95.csv", index=False)
st.session_change_tests(S, ["push_cue_minus_control", "walk_cue"], fdr_scope="all").to_csv(
    OUT / "treadmill_tests_primary_first95.csv", index=False)

# ---- item 9: per-position pre-stroke forward speed traces for S13
labels = sorted({lab for lab in T.label.unique() if B.session_epoch(lab) == "pre"})
per = {p: [] for p in POS}; names = {p: [] for p in POS}
for lab in labels:
    TR = tc.session_trials(lab)
    if TR is None or not len(TR):
        continue
    TR = TR.sort_values("trial_id").reset_index(drop=True)
    t, fwd, _back, _z = B.load_session(lab, None)
    working = np.asarray(tc.camera_engaged_period(lab, TR.cue_daq.to_numpy(float)), bool)
    for p in POS:
        cues = TR.cue_daq.to_numpy(float)[working & (TR.position.to_numpy() == p)]
        if not len(cues):
            continue
        X = np.stack([np.interp(c + t_ms / 1000.0, t, fwd, left=np.nan, right=np.nan) for c in cues])
        per[p].append(np.nanmean(X, 0)); names[p].append(lab)
    print("done", lab, flush=True)
arrs = {p: np.asarray(per[p], np.float32) for p in POS}
np.savez(NPZ, t_ms=t_ms.astype(np.float32), **arrs, **{f"sessions_{p}": np.asarray(names[p]) for p in POS})
fr = np.load(MC / "data" / "fr_pre_timecourses_abs.npz")
print("sessions per position, treadmill vs FaceRhythm:", {p: (arrs[p].shape[0], fr[p].shape[0]) for p in POS}, flush=True)
