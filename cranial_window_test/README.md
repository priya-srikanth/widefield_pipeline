# cranial_window_test — PS104 one-off analysis

**PS104** (VGLUT1-iCre × RiboL1-jGCaMP8s) cranial-window test session: 415/470 imaging with
random water rewards, **no DAQ / no behavior task**. Explicitly **NOT part of the PS92–95
cohort analysis** (kept separate; excluded from cross-day/cross-session comparisons).

Scripts (standalone; run from the repo root in the `wfield` env):
- `_ps104_window.py` — crop to the cranial-window ROI, per-channel ΔF/F over time, hemo correction
- `_ps104_event.py` — candidate motor / licking-bout event detection + spatial ΔF/F maps
