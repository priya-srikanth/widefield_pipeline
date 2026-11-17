# State-space recovery results page

Published: https://claude.ai/artifact/9sGELLFEwADEt9k64fXTxe (draft, exploratory; owner Priya).

- `index.html` -- the page source (also backed up at `M:/MICROSCOPE/Priya/Widefield/state_space_recovery/page/`).
  Republish this file to keep the URL; pass every new or changed PNG in the Artifact `files` map.
- `fig_ids.csv` -- permanent figure id (`f-<image stem>`, use these for links) -> current number N.k, title, image.
  Regenerate after any renumbering: `python docs/results_pages/state_space/make_fig_ids.py`.
- Figures: `scripts/state_space_page_figs.py` (WF_SS_ALIGN=cue|lick), `state_space_trajectories.py`,
  `state_space_noise.py`, `state_space_contact.py` -> `M:/.../state_space_recovery/page_figs/`.
- Conventions: left TOC (figure lists collapsible per section), "Where things stand" table with a figures column,
  N.k numbers, a title above every image, methods box (`#methods`), cue before lick, units in panel titles.
- Plan, numbers, decisions: `docs/STATE_SPACE_RECOVERY_PLAN.md`.
