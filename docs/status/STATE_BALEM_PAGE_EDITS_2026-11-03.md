# Imaging page, section 3: edits for the bal_em production state decoder (2026-11-03)

**Not published.** This session's account cannot see the imaging page
(https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV). Whoever can see it: re-read the live page first, then apply the
edits below to **section 3 only** (header block, the two tables, Figs 3.1-3.5) and bump the version line.

Why: on 11-03 Priya adopted bal_em as the production frozen state decoder ("ok bal_em it is"; DECISIONS 2026-11-03
"bal_em adopted as the production state decoder"; commit 9547a773). The 13s epoch family was re-rendered. The session_z
outputs are in `labcams/grant_figures/epoch/_superseded_session_z_20261103/` (+ `data/`, `svg/`).

All numbers below were read from the re-rendered sidecars (`labcams/grant_figures/epoch/data/epoch_13*`) and from
`labcams/state_decoder_ps94_check/production_balem/` (`check_log.txt`, `animal_epoch.csv`, `boot_delta.csv`).

---

## 0. Status tag

Keep "neural · stands". Add to the status row: "state decoder = bal_em (11-03); specificity ratio now holds at every
epoch".

## 1. New correction block (insert ABOVE the existing "CORRECTION (11-02, late)" block)

> **CORRECTION (11-03): bal_em is now the production state decoder** (widefield_pipeline commit 9547a773; DECISIONS.md
> "2026-11-03 — bal_em adopted as the production state decoder"). Training: each pre-stroke session is centred at the
> equal-weight mean of its three class means (training labels). Scoring: the frozen model's posteriors -> EM re-estimate
> of the session's class mix (Saerens et al. 2002) -> soft class means of the raw features -> centre at their equal-weight
> mean, production SD, 3 passes. **No test-session labels are used.** Stored models: new spec ids PS92 96f8d3543d94,
> PS93 084308f266df, PS94 8bfbb26d78de, PS95 57866a0b31e3. The session_z models are kept. Epoch family 13 was re-rendered
> 11-03 ~11:05; the session_z outputs are in `grant_figures/epoch/_superseded_session_z_20261103/`.
> - Pooled above-chance performance retained, state: **.972 / .927 / .950 / .922** (pre / acute / subacute / chronic).
>   It was .967 / .903 / .865 / .872 under session_z (11-02) and .976 / .872 / .820 / .838 unscaled. Position is unchanged
>   at .863 / .428 / .696 / .761.
> - So **state loses 4.6 / 2.3 / 5.1 % against position's 50 / 19 / 12 % (~11x / 8x / 2.3x)**. The ratio now holds at
>   every epoch; under session_z, chronic state and position losses were similar.
> - PS94 running recall is **.99 -> .97 / .96 / .92** (session_z .98 -> .84 / .54 / .49).
> - The cost is small: PS92 acute balanced accuracy .935 -> .914 (quiet recall .88 -> .80), and PS94 chronic quiet
>   .95 -> .89.
> - Change vs pre (Fig 3.4, boot_delta animals -> sessions, Holm over 6): position -.41 / -.19 / -.09 (all **); state
>   -.033 (*) / -.026 (ns) / -.055 (**).
> - A minimum-mass fallback (training-mix weights when an EM class share is < 2 / 5 / 10 %) was tested on the forced-mix
>   stress test and kept OFF: it does not reduce the drop when a state is absent.
>
> The 11-02 block below is kept as the record; its state numbers are superseded.

## 2. Table "frozen decoder, above-chance (normalised)"

Add a row ABOVE the "(z-scored, 11-02)" row, and relabel that row:

| frozen decoder, above-chance (normalised) | pre | acute | subacute | chronic |
|---|---|---|---|---|
| position | .863 | .428 | .696 | .761 |
| **behavioural state (bal_em, 11-03, production)** | **.972** | **.927** | **.950** | **.922** |
| behavioural state (session_z, 11-02; superseded) | .967 | .903 | .865 | .873 |
| behavioural state, pre-fix (unscaled, 10-15 render) | .976 | .872 | .820 | .838 |

## 3. Paragraph after that table: replace its first sentence through "...never that state is flat."

> Acutely position keeps 0.50 of its pre-stroke above-chance accuracy and state 0.95 (session_z 0.93, pre-fix ~0.89).
> Running alone (the running-class recall of this same decoder) goes 0.99 -> 0.97 acutely, keeping ~98 %, then 0.98
> subacute and 0.97 chronic. Per animal acute running is .96 / 1.00 / .97 / 1.00 (PS95 one session). Change vs pre
> (`boot_delta`, animals -> sessions): -.010 [-.039, .007] / -.002 [-.019, .013] / -.019 [-.055, .010]; no interval
> excludes 0. State still falls slightly at the cohort level. Raw balanced accuracy minus pre (Fig 3.2): acute -0.022
> [95 % -0.047, -0.002; Bonferroni -0.053, +0.000], subacute -0.018 [-0.041, +0.000; -0.045, +0.003], chronic -0.037
> [-0.072, -0.010; -0.081, -0.007]; only chronic excludes 0 after Bonferroni. The residual loss is now mostly the QUIET
> class (13pos: quiet .98 -> .90 / .95 / .92), not running. So the claim is the ratio: position loses about eleven times
> more of its above-chance performance acutely, and still ~2x more chronically. It is never that state is flat.

Delete the sentence "By chronic the two losses are of similar size (position −.09, state −.11, the state loss carried by
PS94)." That sentence was true under session_z only.

## 4. Per-animal table ("Per animal (added 10-25; corrected 11-02)")

Retitle it "Per animal (added 10-25; corrected 11-02, 11-03): state decoder above chance, (accuracy − 1/3) / (2/3); bal_em
(session_z in brackets)". Replace the rows:

| animal | pre | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | .947 (.936) | .871 (.902) | .932 (.835) | .892 (.838) |
| PS93 | .975 (.972) | .941 (.932) | .966 (.961) | .963 (.957) |
| PS94 | .981 (.976) | .955 (.871) | .901 (.716) | .845 (.669) |
| PS95 | .986 (.984) | .992 (.988), 1 session | .983 (.939) | .967 (.966) |

Replace the paragraph under it with:

> **Corrected 11-03 (bal_em).** PS94's state loss is mostly gone: chronic -.136 vs pre (session_z -.31, unscaled
> -.53). The other three animals change by -.055 (PS92) / -.012 (PS93) / -.019 (PS95) chronic. PS92 acute is the one
> cell that is LOWER than under session_z (.871 vs .902; quiet recall .88 -> .80). PS94's states were always separable
> within session (refit .97-.99). What the session_z decoder lost was PS94's running class, and the label-free
> class-balanced centring recovers it (Fig 3.5). So the state readout is nearly intact in all four animals, while
> position falls by half acutely. Session means, cue window, engaged trials; pre = leave one session out (n = 11).
> Source: `epoch_13_state_decoder_cue_sessions.csv` (11-03 re-render); table
> `imaging_page_src/s_per_animal_regen_20261103_balem/state_table_balem.py`. Earlier values are in brackets (session_z)
> and in the 11-02 table record.

## 5. Figures

| fig | image to swap in | old image -> keep as |
|---|---|---|
| 3.1 `d_state.png` | `labcams/grant_figures/epoch/epoch_13n_state_vs_position_cue.png` (11-03; `_nolines` version beside it) | `imaging_page_src/imaging_page/_superseded_session_z_20261103/d_state.png` |
| 3.2 `d_state_delta.png` | `labcams/grant_figures/epoch/epoch_13delta_state_decoder_cue.png` (or `..._nolines.png`, session-dot style, redrawn 11-03 from the new sidecars) | `.../_superseded_session_z_20261103/d_state_delta.png` |
| 3.3 `s_per_animal.png` | `imaging_page_src/s_per_animal_regen_20261103_balem/s_per_animal_nolines.png` (state panel = bal_em; template panels unchanged, checked to 0 difference) | `.../_superseded_session_z_20261103/s_per_animal.png` |
| 3.4 `state_position_sessions.png` | `presentations/_source_PI_overview_2026-11-02/build_inputs/state_position_sessions_balem.png` | the old `state_position_sessions.png` is still beside it |
| 3.5 `state_ps94_diagnostic.png` | `presentations/_source_PI_overview_2026-11-02/build_inputs/state_ps94_balem.png` | the old `state_ps94_diagnostic.png` is still beside it |

### Fig 3.1: title and re-render line

- **Title:** "Frozen position vs behavioural-state decoders, cue window: position loses ~half its above-chance accuracy
  acutely, state ~5 % (bal_em, 11-03)".
- **Replace the "Re-rendered 11-02 (late)..." line with:** "Re-rendered 11-03 with the production bal_em state decoder
  (commit 9547a773; `labcams/grant_figures/epoch/epoch_13n_state_vs_position_cue.png`, ~11:05). State .972 / .927 /
  .950 / .922 (session_z .967 / .903 / .865 / .873; pre-fix .976 / .872 / .820 / .838). Position unchanged at .863 /
  .428 / .696 / .761. Session_z image: `_superseded_session_z_20261103/d_state.png`."
- **In the caption body:** after "State = logistic (C = 1.0) classifying 1 s segments ...", add: "Scaling: bal_em. Each
  training session is centred at the equal-weight mean of its class means. Each scored session is centred at the
  equal-weight mean of EM-estimated soft class means (no test labels), with the per-session SD."

### Fig 3.2: title and re-render line

- **Title:** "Frozen behavioural-state decoder, change from pre: a small fall (−.02 to −.04 balanced accuracy); only
  chronic survives Bonferroni (bal_em, 11-03)".
- **Replace the "Re-rendered 11-02 (late)..." line with:** "Re-rendered 11-03 (bal_em). Raw balanced accuracy minus
  pre: acute −0.022 [95 % −0.047, −0.002; Bonferroni −0.053, +0.000], subacute −0.018 [−0.041, +0.000; −0.045,
  +0.003], chronic −0.037 [−0.072, −0.010; −0.081, −0.007]. Session_z: −0.043 / −0.068 / −0.063 (all Bonferroni). Pre-fix:
  −0.069 / −0.104 / −0.092."

### Fig 3.3: title, re-draw line and reading

- **Title:** "Per animal, frozen encoder and frozen state decoder, cue window: template amplitude and captured shape fall
  in all four; state stays ≥ .85 in all four (bal_em, 11-03)".
- **Replace the "Right panel re-drawn 11-02 (late)..." line with:** "Right panel re-drawn 11-03 with the bal_em state
  decoder: per animal from the re-rendered `epoch_13_state_decoder_cue_sessions.csv` (PS94 chronic .845, session_z .669,
  pre-fix .447; PS92 .892, PS93 .963, PS95 .967). Left and middle panels unchanged (session means checked equal to the
  stored tables). Drawn in the 11-03 session-dot style by `scripts/imaging_page/s_per_animal.py --src
  imaging_page_src/s_per_animal_regen_20261103_balem`."
- **Reading, right panel:** "PS94 falls most (chronic −.14 vs pre), PS92 −.06, PS93 / PS95 within .02."

### Fig 3.4: title, re-render line and stats / reading

- **Title:** "Frozen position vs behavioural-state decoders, cue window, one marker per session: position loses .41 of
  its above-chance performance acutely, state .03; state stays ≥ .85 per animal (bal_em, 11-03)".
- **Replace the "Re-rendered 11-02 (late)..." line with:** "Re-rendered 11-03 with the bal_em state decoder
  (`build_inputs/state_position_sessions_balem.png`, from the 11-03 sessions sidecar). The state panel and its stats
  changed; position did not. Session_z values (record): state −.044 (*) / −.104 (**) / −.110 (**), mean of animals .97
  → .92 → .86 → .86, chronic per animal .84 / .96 / .67 / .97."
- **Stats sentence:** "Position −.41 / −.19 / −.09 (all **), state −.033 (*) / −.026 (ns, Holm p .063) / −.055 (**)".
- **Reading:** "mean of animals: position .86 → .45 → .67 → .77, state .97 → .94 → .95 → .92. State per animal chronic
  .89 / .96 / .85 / .97: the PS94 decline is now small (its running class is recovered, Fig 3.5)."
- **Script line:** "`fig_state_position_sessions.py state_position_sessions_balem`".

### Fig 3.5: title, caption and status

- **Title:** "PS94 state diagnostic: the three states stay separable (refit .98–.99); the plain session z-score lost
  PS94's RUNNING class and bal_em (production since 11-03) restores it".
- **Decoders sentence:**
  - bal_em (orange) = the stored `scaling="session_bal_em"` models, scored by the production path (`score_animal`; label-free
    EM equal-weight centre, production SD, 3 passes).
  - session_z (blue) = the stored 11-02 models (previous production).
  - Within-session refit (grey) is unchanged.
- **Measured, row 3:** "PS94 pooled confusion of the **bal_em** decoder per epoch".
- **Reading:** "PS94 balanced accuracy: bal_em .99 / .97 / .93 / .90, session_z .98 / .91 / .81 / .78, refit .98 / .99 /
  .98 / .98. PS94 running recall: bal_em .99 / .97 / .96 / .92 vs session_z .98 / .84 / .54 / .49. The cost is chronic
  quiet .95 → .89 (read as running .08) and licking .90 → .88. PS92 / PS93 / PS95 are .91–.99 under bal_em. Mechanism
  unchanged: PS94 runs ~2x faster and has ~4x more running segments post-stroke, which pulled the all-segment session
  mean. The EM-estimated class mix lets the centre follow the class means without test labels."
- **Delete:** "Whether the state decoder should use the class-balanced z-score is open (Pending)." It is DECIDED.
- **Source:** `build_inputs/state_ps94_balem.png / .csv / _confusions.csv`. Script: `fig_state_ps94_balem.py`, same folder.

## 6. Known caption leftovers in the re-rendered PNGs (not page text, for whoever next edits `epoch_state.py`)

`epoch_13pos` still prints "any post-stroke fall is carried mostly by one animal". The running bars are now nearly flat
(.99 / .97 / .98 / .97), and the residual decline is QUIET (.98 → .90 ** / .95 * / .92 *). That phrase is a literal in
`epoch_state._fig_13_state`; it is harmless, but out of date. The 13n subtitle is measured and correct ("state 0.97 ->
0.93 acutely, losing 5%; RUNNING-class recall 0.99 -> 0.97, losing 2%").
