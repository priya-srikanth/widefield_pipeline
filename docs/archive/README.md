# docs/archive — retired, kept for reference

Historical one-offs and superseded specs. **Not maintained** — they record what happened at a point in
time and are preserved for provenance, not as current instructions. For how the pipeline runs today, see the
`runbooks/`, root `README.md`, and `CLAUDE.md`.

- `GPU_LOCANMF_KICKOFF.md` — paste-ready instructions used to first stand up LocaNMF on the NVIDIA/GPU box
  (custom Windows build of `localnmf`/`cuhals`). Still the reference for the GPU install recipe; the clone
  path/URL in it is pre-migration.
- `GPU_LOCANMF_RUNLOG.md` — run log of executing the kickoff on the RTX 4060 box (2026-06-04): env recipe +
  deltas from the idealized instructions.
- `MOTION_CORRECTION_SIGN_BUG.md` — the wfield 0.4.2 motion-correction sign-bug writeup + remediation. The
  fix is now the standard path (`run_wfield_motion` → sign-fixed 2D); this is the triage record.
- `LOCANMF_XSESSION_DECK_SPEC.md` — spec for the retired cross-session PowerPoint deck
  (`locanmf_xsession_deck.py`), superseded by `locanmf_analysis_deck.py` →
  `spout_position_analysis_summary.pptx`.
- `MIGRATION.md` — record of the 2026-08-08 split of this repo out of `Widefield_DAQ_recorder` + the
  imaging-box migration prompt.
