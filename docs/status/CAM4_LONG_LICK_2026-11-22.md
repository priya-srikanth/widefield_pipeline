# cam4 "long lick" check (2026-11-22)

Priya 11-22: measure how often one cam4 lick holds two licks before changing the split rule. Trigger: the clip note on
PS95 0821 far_L trial 259 ("cam4 calls one long lick where the other cameras, 3-D and DAQ see two; the fused train
follows cam4"). Read-only; scripts `scripts/cam4_long_lick_qc.py` (counts) and `scripts/cam4_long_lick_strips.py`
(frame strips). Outputs: `session_poses/pose3d_4cam_3net_20261121_sync60/qc_cam4_long_lick_20261122/`
(`cam4_windows.csv.gz` one row per cam4 lick, `rates_*.csv`, `reach_first_sensitivity.csv`, `strips/`).

Population: all 601,073 cam4 post-cue licks (cue .. stop, `licks.csv` of the adopted v37_sync60 run) of 125 sessions.
For each lick's `on_ms .. off_ms` window (what `pose3d_kinematics` measures for Fig 4B / 4E) we counted the 3-D lick
events whose peak falls inside, the DAQ contact onsets inside +- 20 ms (distinct only if >= 60 ms apart), and the
fused licks inside.

## Results
- **Two licks in one cam4 window (3-D AND DAQ both say two): 24 licks of 601,073 (0.004 %).** 3-D alone says two
  0.09 %. The fused train splits 11 of the 24. Not worth changing the split rule or rebuilding the lick list.
- **The trial-259 example is not a merge in the production tables.** cam4 has two licks there (peaks 352 and 488 ms),
  but their windows OVERLAP (316-440 and 404-548), which the clip overlay draws as one bar; the fused train has two
  licks (split at 434). Overlapping cam4 windows: 3.0 % of licks (12 % of first licks); the reach maximum falls in the
  shared part for only 0.03 %.
- **DAQ double contacts inside one cam4 window: 0.94 %** (close spouts 1.1-1.8 %, far 0.1-0.5 %; first licks 3-6 %),
  nearly always with ONE 3-D protrusion: contact chatter / a re-touch during one lick, not a second lick.
- **The real problem is the other way: short cam4 windows that miss the tongue's peak.** 6.5 % of the cam4 licks in
  the 3-D kinematics table (38,405 of 591,449) have no 3-D lick peak inside their window (median window 24 ms vs
  84 ms; median reach 3.48 vs 4.55 mm; 97.8 % of them have a fused lick there whose span covers the whole
  protrusion). The reach of such a lick is measured on a fragment of the protrusion (example strip PS95 0809 t229:
  window 232-240 ms on the rising edge, true peak ~5.4 mm at ~250 ms, reported 2.99 mm).
  More common in PRE first licks (15.8 % of first licks per session x position on average) than post (5.9-8.1 %).
- **Fig 4B sensitivity** (first-lick reach, session x position medians, August-on): dropping those fragment windows
  raises the median by 0.06 mm on average (pre +0.10, acute +0.04, subacute +0.02, chronic +0.05), so the change from
  pre shrinks: acute 4.143 -> 4.259 (+0.116) becomes 4.247 -> 4.297 (+0.050); chronic +0.292 -> +0.235; subacute
  +0.296 -> +0.216 (means of session medians, not the bootstrapped stat). Fig 4E speed: fragment windows also read
  slower (median 154 vs 181 mm/s).
- ILI / lick-rate readouts: the merge case is negligible (24 licks). Fragment windows are still licks (counts are
  right); only their kinematics are wrong.

## Recommendation
Keep the cam4 lick LIST (no split-rule change, no fused rebuild). Fix the MEASUREMENT SPAN in
`scripts/pose3d_kinematics.py --licks cam4`: measure each cam4 lick over the span of its matching fused lick (or the
3-D tongue-out run containing the cam4 peak) instead of the cam4 `on_ms .. off_ms`, or drop windows with no 3-D peak.
Rebuild cost: `kin_cam4` ~10 min, then the phase chain (pose3d_phase, mouth0, curves, from-peak, aligned sessions;
~30 min) and `c2_lick3d_session_summary`, then the scalar family / Fig 4 re-render. Needs Priya's OK (changes Fig 4B /
4E / 4G-I numbers).
