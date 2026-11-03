# Handoff 2026-11-03 — kinematics page (behaviour split) + repo consolidation audit

Window: sabatinilab-e3. Owns the **kinematics page** and the **repo consolidation review**. Read this first, then
`docs/status/REPO_AUDIT_2026-11-03.md` (the audit) and DECISIONS 2026-11-03 (trial classes).

## 0. State at hand-off

| thread | state | next step |
|---|---|---|
| Kinematics page | **v23** live, https://claude.ai/artifact/DKcKRJKbhBoratxxRc6qZv | behaviour / kinematics split (§2), after Priya answers the FaceRhythm question |
| Repo audit | audit + independent verification DONE (`REPO_AUDIT_2026-11-03.md`, `REPO_AUDIT_VERIFY_2026-11-03.md`); draft plan `REPO_CONSOLIDATION_PLAN_2026-11-03.md` | Priya approves the plan, Tier A decisions first (A1: S28 quit point uses a 2.0 s "responded" window vs the task 3.5 s) |
| Code changes from the audit | **NONE made** | wait for (1) Priya's approval of the plan, (2) the cohort-1 rebuild window (sabatinilab-34, swap-mask + lag hemo pass) to land, so we don't collide |

## 1. Kinematics page: what changed today (v14 → v23)

Source `M:/MICROSCOPE/Priya/Widefield/kin_page_src/kin_page/` (index.html + PNGs + fig_ids.csv). After any edit run
`scripts/page_stable_ids.py <index.html>`; publish with the `files` map for new / changed PNGs.
**Several windows write this folder** (the camera-orientation / + = contra window replaced index.html under me once):
always rebuild from the LIVE page (Artifact read) or the current index.html, never a cached copy.

- **Section 6 rewritten, every engaged trial.** Classes contact / incomplete / jaw-only / no response come from ONE table,
  `session_poses/cam1_licks/trial_classes/sessions.csv`, built by `jaw_only_incomplete_figs.classify` (5b's, 07385b37).
  `jaw_pass_qc` is the jaw-MOVED call, not a QC gate (DECISIONS 2026-11-03).
  - Fig 6.1 = raw fraction per class, 6.2 = change vs own pre (`scripts/trial_class_raw_share_fig.py --all-engaged [--change]`).
  - 6.3 = over days, 6.4–6.5 = jaw / tongue traces (5b's re-rendered PNGs; corner a/s/t = animals/sessions/trials).
  - 6.6 = PS94 all-trial jaw.
  - Far contra acute: contact .05, incomplete .60, jaw-only .23, **no response .13** (pre .012, chronic .002).
- **Copied in from the neural pages (11-03, as published, nothing re-run; captions name the source):**
  - Figs 1.7–1.8: hit rate over days and by spout (coding page 0.1, 0.2).
  - Fig 1.9: aim deficit per session (encoder 16.10). Fig 1.10: inter-trial lick rate (encoder 10.6).
  - Figs 4.12–4.19: where licks land, trajectory matching (encoder section 5).
- PI deck (sabatinilab-a8) uses tc_share_all*.png and the re-rendered trial-class PNGs.

## 2. Decided / pending with Priya

- **Behaviour vs kinematics split (Priya 11-03):**
  - NEW **behaviour page** = DAQ-contact measures: hit rate (1.7, 1.8), engagement / quitting (responded fraction, all
    trials vs engaged), contact-based lick timing / rate.
  - **Kinematics page** = camera-based: camera lick timing / rate, aim, paths, landing, jaw, AND the section-6 trial
    classes (Priya: kinematics page).
  - Section 1's DAQ-contact rows move to the behaviour page, cross-linked. NOT STARTED.
- **PENDING:** does the FaceRhythm page (https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1, source
  `M:/.../Widefield/fr_page_src/`, last updated 10-27) merge into kinematics or stay separate with a link? Priya had
  forgotten it existed. Ask before the split.
- **Encoder page trim (Priya: yes):** ask the encoder-page window to replace encoder 5.2, 5.5–5.8, 10.6 and 16.10
  with links to the kinematics copies. Keep 5.1 / 5.3 / 5.4 (landing labels its sections 6–7 use) and 16.6 / 16.9
  (recovery axis). NOT YET SENT.
- **Response window (Priya):** read it from the behaviour log, never hard-code it; she believes it is 3.5 s everywhere.
  The verifier was asked where it is recorded and what the 2.0 s sites really are.

## 3. Repo audit: top findings (unverified until VERIFY.md)

Full table: REPO_AUDIT §0. Highest risk:
1. **Two trial classifiers**, differing on every input: `scripts/trial_attempts.py:114` (encoder attempt classes) vs
   `jaw_only_incomplete_figs.classify`. 5b is adding a "working / camera_lick" GATES helper beside classify for
   imaging decoding: a 4th "responded"-type definition; its docstring will name the other three.
2. **`jaw_pass_qc` read as a gate with `.astype(bool)`** (NaN → True) in `cam1_incomplete_licks.py:56`,
   `orofacial_cohort.py:501/526/546`, `cam1_licks.py:110/179`.
3. **Session-weighted bootstraps**, with the point estimate including animals the draws lack:
   `epoch_figures.scalar_contrast_draws`, `contrast_draws`, `matrix_bootstrap`, `epoch_behaviour`.
   Compare with `analysis_kit.boot_delta`.
4. **Branches diverged:** main vs analysis/attempt-coding, 60 shared modules differ. The engagement gate can count LP
   attempts on the branch (`WIDEFIELD_GATE_SOURCE=lp`); dates are built as "2026"+mmdd; animal id is `label[:4]`
   (breaks PS118).
5. **Lick / statistics drift:**
   - DAQ lick params 1.0 + 100 ms refractory (imaging) vs 0.5 / none (config);
   - response window 2.0 vs 3.5 s;
   - 5 hand-written BH implementations;
   - ~17 ad-hoc LMMs.
- Resolved: "August pre" ≥0801 vs ≥0806 changes no current number (no 0805 files reach those readers).
  `rest_engagement` ≈ `rest_by_position` in practice.

Plan principle (to propose): one function per shared quantity, homed in an EXISTING module, imported everywhere; a
"where each quantity is defined" map in ARCHITECTURE.md; Priya approves before any refactor.

## 4. Swapped-frame question (answered, for the record)

hpfit and its swap-mask / lag variants run POST-SVD on SVT → new SVTcorr. U is unchanged and no .dat is needed.
`hemo_meegkit_hpfit_lag_refitT_swapmask` is built for all 120 curated cohort-1 sessions (11-02, per sabatinilab-34).
The pre-SVD repair applies to RiboL1 only.

## 5. Pitfalls

- The Artifact publish `edits` option is refused here (hook) → edit with Edit / a script, then publish.
- Publish conflicts: merge onto the version the refusal hands back; check the M: source wasn't overwritten by another window.
- Commit only own files, staged by name. The clone has other windows' uncommitted edits → `git rebase --autostash`.
- Kinematics page convention: one figure per idea, stars in the top band (sig_marks.star_top), sessions as open
  markers (shape = animal, colour = epoch).
