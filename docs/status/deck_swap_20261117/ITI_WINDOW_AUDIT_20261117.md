# ITI response window 0..0.5 s: audit and re-render, 2026-11-17

This audit covers cohort 1 (PS92-95), rebuild tag `..._daqlk`.

Decision: DECISIONS.md, top entry "2026-11-17 -- ITI response window 0..0.5 s; cued stays 0..1.0 s". Priya: "we should probably just change ITI AUC and balance results to the 0-0.5s".

Background: `ITI_NEG_AUC_DIAGNOSTIC.md` (this folder). The ITI bout-start response is a short transient followed by a dip below baseline. The dip depends on epoch: the share of tjM1 sessions with a negative 0.5-1 s AUC is 26 / 12 / 0 / 0 % (all-ITI) and 43 / 30 / 18 / 9 % (docked) for pre / acute / subacute / chronic.
- So over 0..1 s the **pre** integral is pulled down, and every post-stroke ITI increase measured against it is inflated.
- The normalized ITI AUCs (AUC / own pre) become unstable when that pre denominator is near zero.

**Bottom line.**
- Every ITI increase shrinks at 0..0.5 s, typically by 1/3 to 1/2. None reverses.
- Several ITI cells **gain** a star, because the 0..1 s CIs were inflated by the near-zero pre integrals. One example is docked contralesional tjM1 subacute: +2.29 with the CI including 0 at 0..1 s, +0.85* at 0..0.5 s.
- No PI-deck title claim changes. These change:
  - the ITI caveat on slide 135 (Fig 14.13 matched-angle balance);
  - the hidden LMM backups 108 / 111 / 138 (they go from 0 to 5, and from 0 to 2, LMM stars);
  - the session-z share on 110 (~2/3 → ~1/2);
  - goal-page Fig 14.29's "chronic orofacial only".

## 1. Code: every place an ITI response AUC / ratio / balance used 0..1.0 s

The new single definition is in `scripts/iti_licks.py`: `CUED_RESP_WIN = (0.0, 1.0)`, `ITI_RESP_WIN = (0.0, 0.5)`, plus the helpers `win_label` / `win_tag`. Each comment cites Priya 11-17 and the DECISIONS entry. Status codes: **F** = fixed in code (this commit); **U** = unaffected (reason given).

| file | what used 0..1 s on ITI | status |
|---|---|---|
| `scripts/lick_cued_ramp.py` | `RESP_W` for the ITI summary (`SPEC_ITI`, `iti_ramp_auc_sessions_wide.csv` resp_auc / resp_auc_n, `iti_ramp_session_tests_wide.csv`, `lick_iti_ramp_summary_wide.png`) | **F**: `ITI_RESP_W` = 0..0.5 s. The 0..1 s values are kept as resp_auc_cw / resp_auc_cw_n plus `iti_ramp_session_tests_wide_cw.csv` (secondary). Labels and suptitle are built from the constants. The window-sensitivity title now names each set's main window. `RIN` now reads caches from OUT whenever a `WFIELD_ITI_OUT_W` folder lacks them. Cued `RESP_W` = `CUED_RESP_WIN`, unchanged. |
| `scripts/lick_lateral_ratio.py` | primary `WINDOWS[0]` = 0..1 s for every row, including ITI | **F**: `SET_WIN` (cued / cue 0..1, ITI 0..0.5). The figures `lick_lateral_ratio_{auc,side_auc,rings}.png` draw each row on its own window. The one-window twins are `_0-1.png` (the old primary) and `_0-0.5.png`. Both windows are still computed for every test / model. |
| `scripts/iti_area_maps.py` | `RESP_W` for ITI `resp` (area_auc_sessions{,_split}.csv, maps, groups, compare) | **F**: ITI uses `ITI_RESP_W`, cued keeps `RESP_W`. Labels come from the constants. `WOUT` honours `WFIELD_ITI_OUT_W` (before, a non-subset re-render would have written to production OUT). |
| `scripts/lick_balance_angle_lmm.py` | `resp` = `logr_0-1` for cued and ITI (Fig 14.13) | **F**: `RESP_COL` per set (ITI `logr_0-0.5`). Inputs are read through `il.in_path`. |
| `scripts/lick_slide95_sessz.py` | ITI scalar `resp_auc` over `lcr.RESP_W` (tables, LMM input; not drawn) | **F**: the ITI rows use `lcr.ITI_RESP_W`. |
| `scripts/lick_vigor_perspout.py` | main PNG = 0..1 s for cued and ITI rows | **F**: `lick_vigor_perspout.png` is drawn per set (cued 0..1, ITI 0..0.5). The 0..1 version is now `_0-1.png`. |
| `scripts/lick_balance_imaging_control.py` | `W = "0-1"` for every set (`balimg_control`, secondary) | **F**: `WSET` per set via `_per_set`, and the label is fixed. Not re-rendered: `balimg_control.png` is on no page or deck (Fig 14.19 shows `balimg_primary_control`, 0.1..0.5 s). |
| `scripts/lick_recovery_markers.py` | ITI `auc_*` = `iti_area_maps` resp; labels "AUC 0-1 s" | **F**: the values follow `iti_area_maps`; labels / docstring updated. |
| `wf_lmm_main scripts/lmm_hidden/_pooled_resp.py` | `RESP_W` 0..1 for the ITI pooled AUC; check column `auc_0_1_n` | **F**: `resp_w(key)` (ITI 0..0.5); `measures(key)` gives the file suffix `_w0-0.5`; the check uses `auc_0_0.5_n`. |
| `wf_lmm_main .../lick_lateral_ratio_auc_docked.py` | `WIN = logr_0-1` for all rows | **F**: `SET_WIN` (ITI row `logr_0-0.5`). |
| `wf_lmm_main .../curve_iti_ramp_row1.py`, `slides.py` | docstring / figure list | **F** (text only) |
| `scripts/lick_balance_variants.py`, `lick_balance_nested.py` | sweep over 0..0.2 / 0..0.5 / 0..1 (no default window; the per-area AUC only at 0..1) | **U** by design: an explicit window sweep, not a primary result. The `lick_balance_nested` inline image's candidate pool already contains 0..0.5. |
| `scripts/iti_dock_page_figures.py` | copier with the pre-11-16 page numbers | **U** (no computation). Re-copy from the new folders if 3c uses it. |
| `scripts/iti_hemo_check.py` | reproduction check against `area_auc_sessions.csv` resp (500 ms) | **U**; the check agrees again after the switch. |
| Already 0..0.5 s for ITI | `iti_licks.py` (resp_auc), `lick_iti_ramp_animals.py`, lmm_hidden `_curve_auc.py` / `lick_cued_ramp_perspout.py`, `iti_mixed_models.py`, the per-spout ITI curves (/ pre 0..0.5 s) | **U** |
| Different window | Fig 14.18 recovery marker (`lick_recovery_primary.py` `ratio_010_050` = 0.1..0.5 s), `lick_balance_imaging_control_primary.py` (0.1..0.5), `lick_balance_maps_svd.py` (0..0.2), head-to-head (resp02 = 0..0.2 s, scratch `prelick_test/head_to_head`), Fig 14.20 `rest_level_vs_aim` (RESTnl rest level; no response AUC) | **U**: confirmed for 3c. |
| Cued only | `specificity_control`, `residual_roi`, `frozen_area_maps`, `within_spout_deviation`, `kinematic_normality`, `lick_precue_restw`, `curve_cued_ramp_summary_row2` | **U** |
| Tests | no test pins the 0..1 s window (`tests/test_trial_set_*` check sentences / paths only) | **U** |
| Scratch, not in the repo | `prelick_test/prelick_balance_test.py` (`resp10` = 0..1 s, including ITI) | Not a page or deck input; noted only. |

## 2. Outputs: re-rendered on the rebuild (new folders, nothing overwritten)

Root (iti_licks): `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/<folder>/<tag>/`, with logs in `<folder>/logs/` (`# env` heads).
- The env was replayed from render_daqlk `l_cued_ramp_g` / `l_lateral_ratio_g`.
- Docked runs add `WFIELD_ITI_FILTER=dock`, as in the 11-17 docked logs.
- Code: wf_cleanup2 56f4d0db plus the edits in this commit.

| folder | run | ITI set | feeds |
|---|---|---|---|
| `lateral_ratio_iti05_20261117` | `lick_lateral_ratio --plot-only` (seeded from the rebuild figdata / tables, which hold both windows); `lick_balance_angle_lmm` (compute) | all ITI | Fig 14.11 rings, the 14.9 / 14.10 all-ITI twins, Fig 14.13 (resp + dec, current shared-y code); slide 135 caveat |
| `lateral_ratio_dock_iti05_20261117` | `lick_lateral_ratio --plot-only` (seeded from `lateral_ratio_dock_20261117`) | docked | Fig 14.9 / 14.10 docked; slide 203; LMM 136-138 input |
| `cued_ramp_iti05_20261117` | `lick_cued_ramp` (compute) | all ITI | Fig 14.23 `lick_iti_ramp_summary_wide`, Fig 14.25 `lick_ramp_windows_iti` (title only) |
| `itidock_iti05_20261117` | `lick_cued_ramp` (compute) | docked | slide 105 notes; LMM 108 reproduction tables; docked Fig 14.23 twin |
| `vigor_iti05_20261117` | `lick_vigor_perspout --plot-only` (seeded) | all ITI | Fig 14.8 |
| `area_maps_dock_iti05_20261117` | `iti_area_maps` (main, `--split`, `--split --spout far_contra`) | docked | Fig 14.29 (split + farcontra + levels), 14.30 peth (curves, unchanged), 14.31 compare, 14.32 groups |
| `area_maps_iti05_20261117` | `iti_area_maps --split`; `lick_recovery_markers` | all ITI | inline image after Fig 14.18 (`lick_recovery_markers_size`, `_cv`) |
| `slide95_sessz_iti05_20261117` | `lick_slide95_sessz` | docked | slide 110 notes; slide 111 LMM input |

LMM companions: wf_lmm_main 1ff6171c plus edits. Folder: `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/<tag>/lmm_hidden_20261117/iti05_20261117/`, logs in `logs/`, all names suffixed `_iti05`.
- `iti_ramp_row1_lmm_w0-0.5_iti05*`: reproduces `itidock_iti05` `iti_ramp_auc_sessions_wide.csv` 714/714 rows and `ramp_window_tests` 21/21.
- `iti_ramp_row1_lmm_w0-0.5_sessz_iti05*`
- `lick_lateral_ratio_auc_docked_iti05*`: 315/315 bootstrap points reproduced; splits `_cued` / `_cue` / `_iti`.
- `lick_lmm_sessz_stack_iti05.png`
- The `iti_ramp_row1_lmm_peak_iti05*` files repeat the peak, which is unchanged.

Not re-rendered:
- `balimg_control` (on no page or deck).
- The all-ITI non-split `iti_area_maps` (the page uses the docked and split versions).
- `lick_balance_variants` / `nested` (window sweep).
- The pipeline decks: no auto-built deck section reads these stems. `deck_registry` "Fig 14.23" is the cued summary and is unaffected; its number is stale (now 14.24).

## 3. 0..1.0 s vs 0..0.5 s: change vs own pre (boot_delta; * = 95 % CI excludes 0)

**Docked ITI response AUC / own pre** (acute / subacute / chronic). Source: `itidock_iti05_20261117/iti_ramp_session_tests_wide{,_cw}.csv`.

| ROI | 0..1.0 s | 0..0.5 s | flag |
|---|---|---|---|
| tjM1 contra | +2.44* / +2.29 / +2.76* | +1.25* / +0.85* / +1.14* | shrinks ~1/2; subacute **gains** * |
| SSp-m contra | +1.67* / +1.14 / +1.58* | +1.23* / +0.78* / +1.00* | shrinks; subacute gains * |
| SSp-m ipsi | +0.59* / +0.47 / +0.89* | +0.61* / +0.44* / +0.70* | subacute gains * |
| tjM1 ipsi | +0.70 / +0.75 / +1.48* | +0.60* / +0.45 / +0.84* | chronic shrinks; acute gains * |
| ALM contra | +0.38* / +0.32 / +0.51 | +0.46* / +0.30 / +0.36 | unchanged pattern |
| cortex | +0.96* / +0.54 / +0.59 | +0.67* / +0.53* / +0.52 | acute shrinks; subacute gains * |
| ALM ipsi | n.s. | n.s. | - |

**All-ITI response AUC / own pre** (Fig 14.23). Source: `cued_ramp_iti05_20261117/...`.
- tjM1 contra: +1.89* / +1.22* / +1.53* becomes +1.26* / +0.74* / +0.88*.
- SSp-m contra: +1.78* / +1.01* / +1.20* becomes +1.31* / +0.74* / +0.78*.
- cortex chronic: +0.53 n.s. becomes +0.48*.
- ALM ipsi: acute and subacute gain *.
- Every cell that was * stays *; all shrink by 25-45 %.

**Normalized per-spout ITI AUC** (diagnostic check). Docked tjM1 contra far-ipsi acute is +11.17 [-0.04, +33.87] at 0..1 s and +1.02* [+0.20, +2.30] at 0..0.5 s; close-ipsi acute -12.1 vs +1.20.
- This matches the diagnostic.
- These per-spout ITI tests are printed but not drawn. The ITI per-spout curves and the LMM per-spout supplement already used 0..0.5 s.

**ITI lateral ratio log2(contra/ipsi), docked** (slide 203, Fig 14.9 docked; both windows from the 11-17 docked tables).
- Pooled acute ALM / tjM1 / SSp-m / SSp-n / orofacial: +0.50* / +0.88* / +0.88* / +0.65* / +0.67* becomes +0.33* / +0.50* / +0.44* / +0.56* / +0.43*.
- Subacute: all five stay *.
- Chronic: ALM only (+0.18* → +0.13*).
- Stars across the ITI cells: 78 of 209 → 92 of 210. Most losses are chronic close-ring cells (e.g. ALM close contra chronic +0.35* → +0.14 n.s.); gains are acute close-ring cells that were NaN or n.s. (e.g. SSp-n close contra acute NaN → +0.63*).

**Docked LR tests** (0..1 → 0..0.5):

| pair | spout x epoch | side x epoch | distance x epoch |
|---|---|---|---|
| ALM | .021 → .084 | | |
| tjM1 | **degenerate (LR 0, p 1.0; the "failed" fit) → .016** | .40 → .042 | |
| SSp-m | .098 → .050 | | |
| SSp-n | .24 → .081 | .020 → .0006 | |
| orofacial | .016 → .005 | | .63 → .037 |

- The pre spout / side effects all become stronger.
- The ALM distance x epoch fit failed at 0..1 s ("Singular matrix") and fits at 0..0.5 s (p .19).

**ITI lateral ratio, all-ITI** (Fig 14.11 / twins).
- Pooled acute: tjM1 +0.36* → +0.24*, orofacial +0.28* → +0.21*.
- Chronic pooled: ALM +0.12* → +0.09*; the others are n.s. either way.
- Stars: 72 → 83 of 210.
- Spout x epoch: tjM1 .22 → .014, SSp-n .24 → .0011, SSp-m .031 → .0011, orofacial .027 → .0011.

**Fig 14.13, ITI balance at matched lick angle** (resp, LMM, post vs pre):

| pair | 0..1.0 s | 0..0.5 s |
|---|---|---|
| ALM | +0.068 (p .14) | **+0.064 (p .005)**; subacute p .016, chronic p .021 |
| tjM1 | +0.104 (p .080) | +0.017 (p .58); chronic 3/4 animals (PS92 opposite) |
| orofacial | **+0.145 (p .010)** | +0.040 (p .16) |
| SSp-n chronic | +0.003 | +0.106 (p .039) |

- The pre slope on angle is now significant in every pair, including ALM; at 0..1 s, ALM p was .65.
- The FINDINGS 708-711 sentence ("not significant in ALM / tjM1") flips for ALM.

**ITI LMM backups:**
- Slide 108 (docked AUC / pre): 0 → 5 of 21 starred (SSp-m contra / ipsi acute and chronic, tjM1 ipsi chronic). Contralesional tjM1 is p .055 / .20 / .067.
- Slide 111 (session-z): 0 → 2 of 21 (cortex acute p .034, SSp-m ipsi chronic p .016).
- Slides 136-138 (balance): whole figure 71/313 → 72/312, 0 ** either way.
  - Cued / cue rows: identical stars.
  - ITI row: 13 → 14. Chronic ALM close / far contra and subacute ALM / orofacial close contra drop; acute tjM1 far mid, SSp-n far contra, SSp-n all, orofacial close ipsi and subacute tjM1 far contra appear.

**Session-z (slide 110), docked ITI contra tjM1 / SSp-m:**
- raw +1.25* / +0.85* / +1.14* and +1.23* / +0.78* / +1.00*;
- z +0.65* / +0.52* / +0.68* and +0.68* / +0.46* / +0.59*.
- So ~1/2 is kept (it was ~2/3 at 0..1 s), and it is significant in every epoch.

**Area maps.**
- All-ITI split: areas with CI excluding 0 go from 34 / 28 / 4 to 42 / 45 / 10 (acute / subacute / chronic).
  - Chronic at 0..1 s: tjM1 and SSp-m both sides.
  - Chronic at 0..0.5 s adds MOp_rest / MOs_rest / SSp-ll / SSp-tr / SSp-ul / ACAd (right).
- Docked (Fig 14.29): 40 / 7 / 10. The chronic set adds contralesional limb / trunk S1 to tjM1, SSp-m, SSp-n and MOp_rest. There was no docked 0..1 s rebuild render to compare.
- So "chronic orofacial only" does **not** hold at 0..0.5 s: chronic = orofacial + contralesional limb / trunk S1.
- Groups (Fig 14.32, docked):
  - orofacial resp / pre +0.62* / +0.40* / +0.53*;
  - visual + RSP +1.72* / +1.75* / +1.38;
  - orofacial minus visual is subacute -1.36*.
  - The 0..1 s all-ITI version had visual + RSP at -15 / -3.8 / -1.5, an artifact of the near-zero pre denominator.

**Recovery markers** (inline image after 14.18): ITI response-AUC LOAO cvR² is -1.13 (both sides, 0..1 s) and -0.36 (0..0.5 s). It still fails, so the conclusion is unchanged: response size does not estimate recovery.

**Vigor control** (Fig 14.8): the ITI rows are on 0..0.5 s (the `_0-0.5` twin already existed). Same conclusion as the page's 0..0.5 panel.

## 4. Pages (owners relay)

Encoding / goal page (owner 3c), current numbering (after the 11-16 insert):

| Fig | image | new path (`<iti_licks>/<folder>/<tag>/figures/`) | conclusion |
|---|---|---|---|
| 14.8 | `lick_vigor_perspout.png` | `vigor_iti05_20261117/.../lick_vigor_perspout.png` (ITI rows 0..0.5) | unchanged |
| 14.9 | `lick_lateral_ratio_auc_docked.png` (+ `_0-0.5_docked` twin) | `lateral_ratio_dock_iti05_20261117/.../lick_lateral_ratio_auc.png` (twins `_0-1`, `_0-0.5`) | magnitudes down; tjM1 spout x epoch now testable (p .016) |
| 14.10 | `lick_lateral_ratio_side_auc_docked.png` (+ `side_auc_0-0.5`) | `lateral_ratio_dock_iti05_20261117/.../lick_lateral_ratio_side_auc.png` | side x epoch now p < .05 in tjM1, SSp-n, orofacial |
| 14.11 | `lick_lateral_ratio_rings.png` (+ `_0-0.5`) | `lateral_ratio_iti05_20261117/.../lick_lateral_ratio_rings.png` | ITI row only |
| 14.13 | `lick_balance_angle_resp.png` (shared-y redraw 11-16; `_dec` unchanged in value) | `lateral_ratio_iti05_20261117/.../lick_balance_angle_resp.png` (+ `_dec`) | **changes**: ITI matched-angle shift significant in ALM, not in orofacial / tjM1 |
| 14.23 | `lick_iti_ramp_summary_wide.png` | `cued_ramp_iti05_20261117/.../lick_iti_ramp_summary_wide.png` (docked twin `itidock_iti05_20261117/...`) | shrinks; all stars kept (cortex chronic gains) |
| 14.25 | `lick_ramp_windows_iti.png` | `cued_ramp_iti05_20261117/.../lick_ramp_windows_iti.png` | title text only |
| 14.29 | `lick_iti_area_maps_split_docked.png` (+ farcontra, levels) | `area_maps_dock_iti05_20261117/.../lick_iti_area_maps_split{,_farcontra}{,_levels}.png` | **changes**: chronic not orofacial-only |
| 14.31 / 14.32 | `lick_iti_area_compare_docked`, `lick_iti_area_groups_docked` | `area_maps_dock_iti05_20261117/.../lick_iti_area_{compare,groups}.png` | groups: control areas up as much as orofacial |
| inline after 14.18 | `lick_recovery_markers_size` | `area_maps_iti05_20261117/.../lick_recovery_markers_size.png` | unchanged (fails) |
| 14.1, 14.2, 14.4, 14.5-14.7, 14.12, 14.14-14.22, 14.24, 14.26-14.28, 14.30 | - | - | unaffected: curves, cued, or other windows. 14.18 (0.1-0.5 s), 14.20 (RESTnl) and head-to-head (resp02) are confirmed unaffected. |

Status-table text on the goal page quoting ITI 0..1 s (for 3c):
- "ITI ... chronically it stays up mainly in orofacial motor cortex (contralesional tjM1)": still true for tjM1, but chronic now also includes limb / trunk S1 contra.
- "The hemispheric BALANCE ..." row (14.9 / 14.13).

Other pages (imaging, kinematics, FR, state-space): no provenance row and no caption uses these producers. They are unaffected.

## 5. PI deck (`Overview_2026-11-06.pptx`, scratch copy 12:17)

Specs: `specs/iti05.json`, 9 slides, built by `specs_iti05_build.py`.

| slide | uses | change | conclusion_changed |
|---|---|---|---|
| 105 | ITI row (curves) + notes ITI AUC 0..1 | notes numbers + Window; image unchanged | no (title is cued) |
| 108 (H) | LMM docked ITI AUC 0..1 | new image, title "0-0.5 s", 5/21 starred | yes (backup) |
| 109 (H) | peak 0..1.5 s | none | no |
| 110 (H) | session-z; ITI AUC numbers in notes | bullet ~2/3 → ~1/2, notes | yes (share) |
| 111 (H) | LMM session-z stack | new stack, bullets | yes (ITI 0 → 2 stars) |
| 113 / 114 | **cued** per-spout AUC 0..0.5 s (no ITI content) | none from this decision | no |
| 135 | cued row; notes ITI caveat (Fig 14.13) | caveat rewritten; status needs_priya | caveat yes, title no |
| 136 / 137 (H) | LMM balance cued / cue rows (Holm family includes ITI) | new image (same stars), 71/313 → 72/312 | no |
| 138 (H) | LMM balance ITI row | new image + bullets (13 → 14) | yes (post-acute cells) |
| 203 | full Fig 14.9 docked | new image (ITI row 0..0.5), notes | no |

Disagreement with the diagnostic's slide list: slides 113 / 114 contain **no ITI result**. Both are the first cued lick per target spout, and their window is 0..0.5 s.
- That per-spout cued window (`SPOUT_WINDOWS[0]`) predates this decision.
- It is a cued-window inconsistency (cued main figures are 0..1 s), not an ITI one.
- They are left unchanged; flagged for Priya.

## 6. Docs quoting ITI 0..1 s numbers (needs update by the owning window)

- `docs/FINDINGS.md:217-223` (ITI response larger acutely, 41 areas; chronic 5 areas): at 0..0.5 s the all-ITI split is 42 / 45 / 10 areas, and chronic adds limb / trunk S1. The line "the chronic ITI effect grows with longer windows" is consistent with the dip.
- `docs/FINDINGS.md:592-596` (ITI chronically mainly contralesional tjM1): still true, plus limb / trunk S1.
- `docs/FINDINGS.md:708-711` (Fig 14.13 ITI shift ALM / tjM1 n.s., +0.07 / +0.10): now ALM +0.06 p .005, tjM1 +0.02 n.s., orofacial +0.04 n.s.
- `docs/FINDINGS.md:945` (0-1 s response AUC fails as a recovery marker): unchanged; ITI fails at 0..0.5 s too (-0.36).
- `docs/FINDINGS.md:1044-1050` (docked ITI response increase, "subacute lost"): at 0..0.5 s subacute is **not** lost (tjM1 contra +0.85*, SSp-m contra +0.78*).
- `docs/ENCODING_FINDINGS.md:1406-1409` (whole cortex 41/45 areas; chronic MOp / SSp-m / tjM1): see FINDINGS 217.
- `docs/status/STATUS_2026-10-26_ENCODING_HANDOFF.md:1774-1776` ("one window for both"): superseded by DECISIONS 2026-11-17.

## 7. Addendum (Priya 11-17, "yes update 113 and 114 to match"): cued per-spout AUC moves to 0..1.0 s

The cued per-spout first-lick / cue response AUC now uses `iti_licks.CUED_RESP_WIN`. It was `SPOUT_WINDOWS[0]` = 0..0.5 s; 0..0.5 s is kept as a secondary. DECISIONS 2026-11-17 has a one-line addendum.

- **Code (wf_cleanup2, `scripts/lick_cued_ramp.py`):** `spout_windows(key)` puts the main window first (cued / cue `CUED_RESP_WIN`, ITI `ITI_RESP_WIN`). It feeds the per-spout AUC tests, the `lick_c*_perspout_windows` rows, and the per-spout curve normaliser (pre AUC over all spouts, now on the main window: 0..1.0 s cued / cue, 0..0.5 s ITI, so the ITI scale is unchanged).
- **Code (wf_lmm_main):** `lick_cued_ramp_perspout.WIN = il.CUED_RESP_WIN` and `WIN_ITI = il.ITI_RESP_WIN`, imported from `scripts.iti_licks` after merging origin/main into the branch. `_curve_auc.win(key)` keeps the ITI per-spout supplement on 0..0.5 s.
- **Render:** `iti_licks/cued_ramp_cued01_20261117/<tag>/` (log `logs/cued_ramp.log`; env replayed from `l_cued_ramp_g`; cached `cued_wide` traces reused, nothing recomputed upstream).
  - `perspout_auc_tests.csv` is identical (648 rows, max |diff| 0) to the rebuild table, which already held both windows.
  - The per-spout curves are rescaled. Significant per-bin bars: lick 2187 → 2178, cue 2282 → 2301; ITI is identical.
- **LMM render:** `lmm_hidden_20261117/cued01_20261117/lick_cued_ramp_perspout_cued01*`. It reproduces the window-0-1 bootstrap points 72/72.
- **Comparison, cued per-spout AUC / pre at spout** (contralesional tjM1, acute / subacute / chronic; 0..0.5 → 0..1.0 s):

  | spout | 0..0.5 s | 0..1.0 s |
  |---|---|---|
  | close ipsi | +.57* / +.47* / +.35* | +.72* / +.46* / +.37* |
  | close mid | +.61* / +.50* / +.37* | +.74* / +.47* / +.34* |
  | close contra | +.75* / +.47* / +.37* | +.91* / +.43* / +.33* |
  | far ipsi | +.56* / +.49* / +.63* | +.47* / +.36* / +.55* |
  | far mid | +.27* / +.52* / +.64* | +.13 / +.49* / +.64* |
  | far contra | +.20 / +.29 / +.52* | -.12 / +.22 / +.48* |

  - Cells with *, all 4 ROIs: lick 60 → 49 of 72; cue 55 → 54.
  - Losses are acute far mid (tjM1, SSp-m, cortex) and subacute / chronic far ipsi and ALM chronic close spouts. Full table: `iti05_scripts/cmp_cued01.txt`.
  - LMM (slide 114): 31 → 22 of 72 starred, 0 ** either way. No acute cell stays starred; none are added.
- **Conclusion:** the slide 113 title "up, except far contra" holds and sharpens. Acute far contra is now below pre (n.s.) in every ROI, acute far mid is n.s., close spouts are larger, and chronic far contra stays up. Slide 114 (hidden backup) changes: there is no acute mixed-model star.
- **Specs:** `specs/cued01.json` (slides 113, 114; "old" strings from a fresh deck copy, 13:35). The iti05 specs were already applied to the deck by then.
- **Goal page (3c):**
  - Fig 14.7 → `cued_ramp_cued01_20261117/<tag>/figures/lick_cued_ramp_perspout.png`. The cue-aligned twin `lick_cued_ramp_cue_perspout.png` and the window rows `lick_cued_perspout_windows.png` / `lick_cue_perspout_windows.png` (0..1.0 s now first) are in the same folder.
  - Fig 14.8 vigor: its cued rows are already on 0..1.0 s in `vigor_iti05_20261117` (y unit = the pre 0..0.5 s AUC, as before).
  - The 14.7 caption numbers change as in the table above.
