# recovery_gui — dead-strobe spout-position recovery toolkit

When a DAQ spout-strobe bit dies, the cue/lick maps lose positions. For **PS93 2026-08-05**
the strobe bit1 was dead **and** the behavior log was empty, so true spout positions were
recovered from the **cam1 head-on video** (spout x-position per DAQ-sync-aligned frame) and
human-verified (279/279, 0 corrections). Full method + provenance: `../STROBE_BIT1_RECOVERY.md`.

Scripts (standalone; run from the repo root in the `wfield` env):
- `_ps93_autodetect.py` — DAQ↔cam1 sync map, spout-x detect, 2-cluster classify, validation scatter
- `_ps93_verify.py` — reference / closest-call montage
- `_ps93_gui.py` — interactive per-trial review GUI (accept/override), saves reviewed CSV
- `_ps93_apply.py` — write synthetic trials.csv + regenerate cue/lick/quiet maps
- `_ps93_gui_refs.py` — per-position reference frames (calibration)

NB: paths/session IDs are PS93-8/5-specific — parameterize before reusing on another session.
