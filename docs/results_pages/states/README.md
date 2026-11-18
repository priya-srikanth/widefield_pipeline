# Behavioral states results page

Live page: https://claude.ai/artifact/ChbBM9K12WWRohx6nU6rCX (first published 11-17). Findings: `docs/FINDINGS.md`
section 9; handoff: `docs/status/STATUS_2026-11-16_ENCDESIGN_TIM_STATES_HANDOFF.md`.

Rebuild:
1. `python -m wfield_local.behavior_states collect` -- writes the tables to `LP_analysis/<run>/arhmm/v1/poststroke_<tag>/`.
2. `python scripts/states_page_figs.py` -- writes the figures to `.../page_figs/`.
3. Copy the PNGs next to `index.html` (the page references bare file names).
4. Run `scripts/page_stable_ids.py index.html` and then
   `scripts/page_file_names.py index.html --prefix LP_analysis/<run>/arhmm/v1/poststroke_pooled_all_s1/page_figs`.
5. Republish with every changed PNG in the Artifact `files` map.

Conventions follow `docs/templates/README.md`:
- figures numbered N.k, with stable `f-<image stem>` ids (`fig_ids.csv`);
- epoch colors from `orofacial_cohort.EPOCH_LINE`;
- sessions as open markers, pooled values as mean with 95 % CI;
- `*` in the top band = paired bootstrap change vs own pre.
