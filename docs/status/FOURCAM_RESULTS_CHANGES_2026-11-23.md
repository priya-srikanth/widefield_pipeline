# Result changes from the four-camera 3-D re-analysis (running review, from 2026-11-23)

Priya 11-23: "we should review any changes in results that come with the new 3d-based analysis." This is the ONE
place those changes are collected. Each row compares the old basis (v36 / cam4 2-D / pre-cam3-on) with the new
four-camera basis: numbers, stars gained or lost, and whether Priya has to decide. Add a row when a family finishes;
never edit an earlier row's numbers (add a new dated row if a rerun changes them again).

Comparison tool for imaging tables: `C:/Users/SabatiniLab/lp_stage/imaging_fourcam_20261123/tools/compare_tables.py`.
Inventory of what is re-run: `FOURCAM_REANALYSIS_INVENTORY_2026-11-23.md`. Rule: every six-position analysis uses the
four-camera 3-D kinematics (only exception: the 2-D front-camera protrusion check for cohort-1 comparison).

## Status by family

| Family | Old basis | New basis | Status | Changes | Priya decides? |
|---|---|---|---|---|---|
| 3-D kinematics scalars (S11, Fig 4) | 2,000 draws, cam3 dropped for PS95 0818/0820 | 4,000 draws, cam3 on | done 11-23 | no star flips (NOTE_CC_B_cam3on, NOTE_CC_B_boot4000) | no |
| Lick-phase cluster tests (Fig 4G/H, S11) | 500 permutations | 5,000 permutations | done 11-23 | phase axis: dist far contra acute GAINED (q .054 -> .039); down0 close contra acute LOST (q .041 -> .059). ms axis: dist and down0 close ipsi subacute LOST. Fig 4G/4H unchanged; S11 B/C re-render (NOTE_CC_B_phase_perm5000) | done (re-rendered by CC-A) |
| Trial classes after cam3-on | pre-cam3-on jaw calls | cam3-on 3-D jaw | done 11-23 | 12 trials change class, all #12 08-20 (8 no_response -> jaw_only, 4 back); cls_final unchanged | no |
| S12D miss-minus-hit correction | regression (step_b) | binned check | done 11-23 | binned version shows NO change in any epoch (acute -0.4 [-1.8, 1.2] vs regression +2.7 **). Alternative (miss gain vs pre, hit gain control) running | YES |
| S10A latency decomposition | 2-D jaw (cam4/cam1) + cam1 + FR vigor | 3-D jaw at all spouts; cam1 (in-mouth lick) and FR as separate categories | running | acute move -> lick lengthening depends on the FR criterion (131 vs 60 ms with jaw only); FR threshold (z 3, K 3) uncalibrated; checks + video QC running | YES (after QC) |
| Six-position reach / speed session value | session median, >= 20 licks | session mean | running | pending | after results |
| Imaging F1 epoch families | v36 rebuild | rebuild_20261123_fourcam | running | pending (compare_tables vs v36) | after results |
| Imaging F2 AC freeze / F4 RESTnl maps | v36 rebuild | fourcam root | F2 freeze, F4 contact done; LP-gate versions running | pending comparison | after results |
| Encoder frame (11-22 fourcam) | morning fused-mode frame | fourcam frame, cam1-only dropped | done 11-22, KEPT (manuscript lineage, Priya: no redo) | see NOTE_CC_4cam | no |
| Page-lineage spont_frA frame | v36 rebuild frame | fourcam, local run | waiting (smoke test, RAM) | pending | after results |
| AR-HMM states | VERSION 3 features | VERSION 4 re-infer, saved fits | done 11-23 | 15.5 % of 40-ms frames / 8.5 % of syllable groups relabeled, even across epochs; Priya: keep re-infer | done |
| GLM-HMM (S15) | v1 | v1_floor020 default | done; select-K for v1_floor020 to run | pending select-K | no |
| FaceRhythm lick-aligned maps | cam4 v36 lick list | fused four-camera list | NOT YET RE-RUN (flagged by caption review) | pending | after audit |
| Motion energy / TIM / GLM-HMM inputs | ? | 3-D / fused licks | audit running (AUDIT_sixpos_3d_kinematics.md) | pending | after audit |

## Two-tone cohort (not 3-D, same review because the rules changed with it)
| Change | Status | Result |
|---|---|---|
| In-mouth frames = 0 for every measure (item 1) | done 11-23 | angle only changes; 3 angle clusters gained, 0 lost; Fig 2H loses 9 dot bins |
| All-lick measures as session means (item 19) | done 11-23 | 3 stars gained (2E mod contra chronic; 2D mild contra subacute; 2D mod ipsi subacute), 0 lost |
| xy pair merged into the scalar family (item 20) | done 11-23 | 1 star gained (S6D mild contra acute) |
| Fig 2E over all licks (no 5-lick cap) | running | pending |
