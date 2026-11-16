# Headline table on the CORRECTED outputs (LP lick class, rest fix, D7, section G camera groups), 2026-11-09

Read-only. No renders, no repo edits, no M: writes. Numbers were read from sidecars and step logs. Per-animal values
come first, then the pooled estimate as the source reports it. A "computed mean" is a plain mean of per-animal
values that I took; it has no CI.

**OLD** = the 11-08 headline table value (cohort-1 rebuild, DAQ lick source). For rows that table did not cover, OLD =
the DAQ / pre-fix output that the page used. **NEW** = the 11-08/11-09 re-run on fixed code.

Path shorthands:
- `$ROOT` = `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103`
- `$TAG` = `hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`
- `$EPd` = `$ROOT/labcams/grant_figures/$TAG/epoch/data` (DAQ render; also the valid home of every cue/working output)
- `$EPL` = `$ROOT/labcams/grant_figures/LP_licks_gateLP_$TAG/epoch/data` (LP epoch arms + matched-pre lick half)
- `$LPF` = `$ROOT/labcams/grant_figures/LPfinal/trained_lp/LP_licks_gateLP_$TAG/epoch/data` (LP pre-cue decoders, Fig 2.8)
- `$LPFM` = `$ROOT/labcams/grant_figures/LPfinal/LP_licks_gateLP_${TAG}_refs-mean+rest+restw+precue+raw+restnl/epoch/data`
- `$SGo` = `$ROOT/figures_working/$TAG/section_g__$TAG.json` (section G r4, DAQ grouping)
- `$SGn` = `$ROOT/figures_working/LP_licks_gateLP_$TAG/section_g__LP_licks_gateLP_$TAG.json` (section G LP + G4-only merge)
- `$DRd` / `$DRL` = `$ROOT/labcams/grant_figures/map_drift_prepre/{,LP_licks_}hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk_refs-rest+precue+restnl/drift_vs_stroke_panels.csv`
- `$LOG` = `C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/render_daqlk/logs`

Common settings unless a row says otherwise:
- Animals: cohort 1, PS92-95. Epochs are per animal. Sessions: pre 8/8/8/8 (DAQ) or 8/**7**/8/8 (LP; PS93_0813 has no video
  and drops out), acute 5/4/6/1, subacute 2/3/5/7, chronic 12/12/8/11.
- Gate: camera trial gate (cue_stop, end = last camera lick, A2 relabel). Working = camera contact + incomplete.
- Pre: August pre only (0806-0814).
- Baseline: each animal's own pre, with pre sessions scored leave-one-session-out.
- Epoch decoder statistics: bootstrap animals -> sessions -> blocks, 2,000 draws, 95% CI. "corr" = the
  18-comparison corrected CI the sidecar reports.
- LP lick class = camera lick attempts (contact + incomplete, first camera lick within max_rt). The LP lick-window
  decoders are set 1 (epoch arm; Priya 11-09 accepted the near-identical set 2 in grant).

Flags:
- **UNCHANGED**: within noise, or not re-run because the output does not depend on the corrected input. The reason is
  given in the row.
- **CHANGED-SAME-CONCLUSION**: the numbers moved but the stated claim holds.
- **CONCLUSION-CHANGES**: the stated claim no longer holds as written.
- **NOT-YET-AVAILABLE**: the output is missing or still running.
- Two extra labels, used only where none of the four fits: **NEW (no comparator)** and **CHANGED, claim not located**.

---

## 1. CONCLUSION-CHANGES (for Priya first)

| # | Claim (OLD) | OLD | NEW | What it becomes |
|---|---|---|---|---|
| 1 | **C1. Frozen decoder, LICK window, far-contra acute loss** (11-08: loses corrected significance; rests on 6 sessions) | DAQ: -.367 [-.666, -.066], corr [-.858, +.041]. Per animal (session means): PS92 -.37 (2 sess), PS93 -.54 (3), PS94 -.33 (1), PS95 none. Absolute fC accuracy: pre .956, acute .531 | LP: per animal **-.82 (5 sess) / -.73 (4) / -.66 (6) / -.60 (1), 4/4**. Pooled **-.686 [-.789, -.571], corr [-.833, -.510]**. Absolute: pre .909, acute .193, sub .655, chronic .722. Far middle acute -.336, corr [-.515, -.162] (per animal -.18 / -.32 / -.49 / -.38; DAQ was -.165, corr crossed 0). Far ipsi acute -.241, corr [-.509, -.055] (DAQ corr crossed 0). Subacute fC -.289, corr [-.497, -.105] (DAQ -.160). Chronic fC -.188, corr [-.337, -.031] (DAQ -.202; per animal -.17 / -.24 / -.32 / -.03) | The acute far-contra lick-window loss is significant after correction, in 4/4 animals and 16 acute sessions. It is now larger than the cue-window loss (-.618). The acute far-middle and far-ipsi lick-window losses are also significant after correction. Sources: `$EPd` vs `$EPL/epoch_accdelta_by_position_lick_lick{,_sessions}.csv`, `epoch_acc_by_position_lick_lick.csv`. Window: 0-2 s from the first lick (DAQ vs camera). Trials: lick class. Frozen pre-trained decoder, bal_em. |
| 2 | **Section G, lick-only arm: post-lick frozen accuracy acutely near pre** (r4) | DAQ lick-only, acute, per animal .870 / .647 / .826 / .941 (n trials 1429 / 1116 / 1609 / 442). **Far-contra recall undefined in every acute session** (no DAQ far-contra lick trials). Chronic .825 / .683 / .777 / .870 | LP lick-only, acute, per animal **.652 / .458 / .599 / .780** (n 1785 / 1459 / 2158 / 533), far-contra recall **.08 / .14 / .43 / .23**. Subacute .833 / .661 / .781 / .836. Chronic .837 / .685 / .802 / .856 (unchanged) | The acute lick-window accuracy was high because the far-contra trials were missing. With camera licks it drops acutely in 4/4, and acute far-contra recall is .08-.43. Chronic is unchanged. Source: `$SGo` vs `$SGn`, `arms.lickonly["post-lick"]`; my per-animal session means. No pre reference or retained ratio for LP yet (section 3). |
| 3 | **D6 matched-pre (mix-matched) control, LICK half** (table only; 11-08 P6 failed, m_mixsym_cue_l on DAQ rows) | DAQ lick rows, acute excess per animal: far contra PS92 +.03 / PS93 -.04 / PS94 +.01 / PS95 none. Trials 15 / 35 / 13 / 1: the cell is effectively empty. Far ipsi -.03 / -.14 / +.09 / +.39 (computed mean +.08, 2/4 negative). Far mid -.15 / .00 / -.07 / +.01 | LP lick rows, acute excess: **far contra -.16 / -.35 / -.21 / -.29 (computed mean -.252, 4/4)** (trials 272 / 232 / 265 / 39). **Far ipsi -.15 / -.14 / -.24 / -.25 (-.196, 4/4)**. **Far mid -.24 / -.09 / -.18 / -.07 (-.146, 4/4)**. Near acute -.06 to -.09 (3/4). Chronic: near ipsi +.140 (0/4 negative). Matches the working half (far contra -.257, far ipsi -.210, far mid -.148) | The lick half now shows the same acute far loss beyond the mix-matched refit as the working half (Holm .0045 each). Before, the lick-half acute far-contra cell was empty and the other far cells were mixed. Sources: `$EPd/refit_matched_pre_cue_lick_sessions.csv` vs `$EPL/refit_matched_pre_cue_lick_sessions.csv` (and `_cue_working_` for the working half); `$LOG/m_mixsym_cue_l.log`, `m_matchedpre_cue_lick_lp.log`. Cue window. No bootstrap for the lick half: means are computed. |
| 4 | **Pre-cue (ENL) far-contra map amplitude RISES acutely** (page: "the pre-cue far-contra rise becomes 4/4" under RESTnl) | DAQ ENL (2.0 s, DAQ lick-free), acute far contra / pre per animal: RESTnl **1.40 / 1.05 / 1.00 / 1.08**, REST 1.40 / 1.18 / .59 / 1.06, RAW .59 / .42 / 1.58 / 1.22. Drift panel, acute - late pre: REST 1.13 (46 sig bins), RESTnl 1.23 (208 bins) | LP ENL (1.0 s, LP lick-free), acute far contra / pre: RESTnl **.83 / .61 / .73 / .85**, REST .70 / .61 / .32 / .79, RESTW .72 / .61 / .55 / .78, RAW .67 / .70 / 1.68 / 1.14. Drift panel, acute - late pre: REST **.66 (6 bins)**, RESTnl **.86 (55 bins)** | With the lick-source fix the pre-cue far-contra amplitude goes DOWN acutely: below pre in 4/4 under RESTnl, REST and RESTW. RAW is down in 2/4 (PS94 and PS95 up). The "rise" no longer holds. Sources: `$LOG/g5_restnl_maps_contact.log` (11-06) vs `$LOG/g5_restnl_maps_lp_r2.log`. The LPfinal sidecar `$LPFM/lpfinal_amplitude_per_animal_precue_working.csv` gives identical numbers. Drift: `$DRd` vs `$DRL`. Pre-normalized nested bootstrap (animals -> sessions) with maxstat for the bins. |
| 5 | **Fig 9.1: the CD shared mode "grows" (magnitude above its trial-matched pre-vs-pre ceiling)**, page "12/12 cells" | Acute magnitude / ceiling: PS92 1.01 / 1.13 / 1.00 (cue / lick / pre-cue), PS93 1.38 / 1.41 / 1.38, PS94 1.19 / 1.23 / 1.20, PS95 1.37 / 1.41 / 1.37. Share of null draws not below the observed scale ("frac no worse"): acute cells at <= .05 in **10/12** (all except PS92 cue .61 and pre-cue .66) | Acute magnitude / ceiling: **PS92 .84 / .95 / .82 (below ceiling; frac no worse 1.00 / .96 / 1.00)**, PS93 1.19 / 1.29 / 1.19 (.06 / .00 / .07), **PS94 1.00 / 1.06 / 1.02 (.65 / .02 / .35)**, PS95 1.24 / 1.28 / 1.22 (.00 / .00 / .00). Acute cells at <= .05: **5/12**. Subacute: magnitude above in 4/4 (frac .00, except PS94 cue .02 and pre-cue .05), as before. Overlap (rotation) below the ceiling in every animal, epoch and window, as before | "Rotates" holds. "Grows" holds from the subacute epoch on, but **acutely only in PS93 and PS95 (plus PS94 in the lick window)**. PS92's acute magnitude is now below its ceiling. Sources: `$LOG/g7_cd_geometry_r2.log`, `cd_overlap_null.log` (DAQ) vs `g7_cd_geometry_lp.log`, `cd_overlap_null_lp.log`. Gate: lick (success trials = DAQ contact vs LP attempts). The frac-no-worse values are not p-values (module docstring). |
| 6 | **Pre-cue (ENL) frozen decoder, far MIDDLE acute loss** (DAQ render: significant after correction; not a separate 11-08 row) | DAQ ENL 2 s: per animal -.06 / -.29 / -.26 / -.17. Pooled -.192 [-.285, -.088], corr [-.316, -.036] | LP ENL 1 s (LP-trained, LPfinal store): per animal -.05 / -.20 / -.02 / -.15. Pooled **-.103 [-.194, -.021], corr [-.241, +.011]** | The acute far-middle pre-cue loss loses corrected significance and is near 0 in PS92 and PS94. Far contra still holds (row S6). Sources: `$EPd` vs `$LPF/epoch_accdelta_by_position_precue_working{,_sessions}.csv`. |
| 7 | **Per-animal acute cue-window far-contra map amplitude, RESTW, 4/4 about halved** (FINDINGS 3.5: .50 / .45 / .53 / .40). This was P8, not rendered in the 11-08 table | FINDINGS .50 / .45 / .53 / .40 | RESTW **.38 / .46 / .67 / .98** (PS95: 1 acute session). RESTnl .43 / .50 / .75 / .98. REST .36 / .46 / .58 / .97. RAW .14 / .13 / .40 / .81 | 3/4, not 4/4 (PS95 shows no acute drop). This is **not caused by the lick fix**: cue/working is source-independent, and the 11-06 DAQ-contact run and the 11-08 LP run give identical numbers. It is a rebuild-vs-FINDINGS difference that the 11-08 table could not show. Sources: `$LOG/g5_restnl_maps_lp_r2.log` (= `g5_restnl_maps_contact.log`), `$LPFM/lpfinal_amplitude_per_animal_cue_working.csv`. |
| 8 | **Section G, G1 "do post-stroke no-lick trials look like pre-stroke licking trials?"** (r4: DAQ engaged vs DAQ none) | Estimable in 46 post sessions, **4 animals**. Acute: 9 engaged-like / 7 undetected-like. Post undetected classified engaged-like (acute session means): .47 / .48 / .69 / .85. Controls .70 / .57 / .86 / .83 | Camera WORKING vs NO_RESPONSE (b83f376d): estimable in **11 sessions, 2 animals (PS94, PS95)**. Every other session reports "too few no-lick trials to discriminate". Acute: 6/6 engaged-like. PS94 .77 (control .84), PS95 .65 (control .67) | G1 cannot be estimated in PS92 or PS93 under the camera classes (camera no_response trials are rare). The cohort-level G1 reading is not supported (2/4 animals). Source: `$SGo` vs `$SGn`, `looks_like_which`. |

---

## 2. Full table

Columns: ID / claim; OLD (source); NEW (source); window / lick source / trials / gate / baseline / correction; n;
flag.

### 2a. Frozen and refit decoders (epoch family)

| ID | Claim | OLD | NEW | Settings | n | Flag |
|---|---|---|---|---|---|---|
| C1 | Lick-window far-contra acute loss | see section 1 #1 | see section 1 #1 | 0-2 s from first lick. DAQ contact -> LP attempts. Frozen decoder (LP-trained, set 1). bal_em. Corrected CI over 18 comparisons | LP: pre 31 (PS93 7), acute 16, sub 17, chronic 43 | **CONCLUSION-CHANGES** |
| L-abs | Frozen decoder accuracy by window, far contra, absolute (Fig 2.4) | lick (DAQ): pre .956, acute .531, sub .773, chronic .754 | lick (LP): pre .909, acute .193 [.102, .336], sub .655, chronic .722. Cue and pre-cue: see S1 / S6 | `$EPd` / `$EPL/epoch_acc_by_position_lick_lick.csv` | as C1 | CHANGED-SAME-CONCLUSION (pattern acute < sub < chronic < pre holds; acute is much deeper) |
| W1 | Far MIDDLE acute loss, cue | -.318 [-.444, -.208], corr [-.526, -.167]; -.19 / -.27 / -.36 / -.47 | not re-run: cue/working is lick-source independent (camera working rows; `scratchpad/epoch_trim.md` §1). 11-08 value stands | cue 0-2 s, camera working, frozen | 4 / 107 | UNCHANGED |
| W2 | Near positions acute, cue | -.202 / -.095 / -.191 | same (not re-run, source independent) | cue | 4 | UNCHANGED |
| W3 | Near middle chronic, cue | -.145, corr [-.310, +.014] | same | cue | 4 | UNCHANGED |
| S1 | Frozen position decoder retained, cue (Fig 3.4) | .840 / .465 / .727 / .758 | same (cue/working; not re-run) | cue | 4 | UNCHANGED |
| S2 | State decoder (bal_em) loses little | .972 -> .942 / .949 / .923 | same. The 13s state decoder is cue-only, and its licking class is DAQ behavior_events by design (`epoch_trim.md`) | cue | 4 | UNCHANGED |
| S3 | Position falls far more than state | ~14x / 5.6x / 2x | same | derived | 4 | UNCHANGED |
| S4 | Far-contra acute loss, cue | -.618 [-.695, -.531]; -.67 / -.63 / -.62 / -.61 | same | cue | 4 | UNCHANGED |
| S5 | Far contra chronic, cue | -.154, corr [-.317, -.041] | same | cue | 4 | UNCHANGED |
| S6 | Pre-cue far-contra acute loss | DAQ ENL 2 s: per animal -.47 / -.37 / -.24 / -.11. Pooled -.289 [-.428, -.145], corr [-.480, -.061]. Chronic -.114, corr [-.238, -.014] | LP ENL 1 s, LP-trained: per animal **-.23 / -.24 / -.24 / -.12 (4/4)**. Pooled **-.201 [-.270, -.116], corr [-.293, -.063]**. Chronic -.095, corr [-.205, -.010] (-.07 / -.07 / -.19 / -.05) | pre-cue, ENL window; DAQ 2 s vs LP 1 s lick-free; camera working; `$EPd` vs `$LPF` | 4 / 107 | CHANGED-SAME-CONCLUSION |
| L-fM | Pre-cue far-middle acute loss | -.192, corr sig | -.103, corr crosses 0 | as S6 | 4 | **CONCLUSION-CHANGES** (section 1 #6) |
| L-gap | Pre-cue refit-minus-frozen gap, far contra (Fig 1.1 / 5rgap) | DAQ: acute per animal .47 / .26 / .11 / -.10; pooled +.181, corr crosses 0. Chronic .09 / .06 / .20 / .10; pooled +.106 [.034, .183], corr [-.011, .226] | LP (D7 code): acute .23 / .11 / .10 / -.14; pooled +.080, corr [-.170, .263]. Chronic .13 / .05 / .17 / .09; pooled +.111 [.041, .182], **corr [.009, .217]** | pre-cue; refit floor = 10 trials (D7); `$EPd` vs `$EPL/epoch_5rgapdelta_frozen_vs_refit_precue_working*.csv` | 4 | CHANGED-SAME-CONCLUSION (acute gap halves; chronic now just clears correction) |
| C4 | Refit-over-frozen gap G keeps rising (imaging LMM) | matched +.027, z 1.75, q .096 | LMM not re-run. Its decoder input, `recovery_trajectory_{matched,unmatched}_cue_working.csv` (`$LOG/g5_recovery_cue_r.log`, D7 code, 11-08 22:40), has a per-animal table **identical** to the 11-07 run: matched acute G .190 / .123 / .199 / .092, final G .142 / .275 / .249 / .205, slope/day +.0006 / +.0045 / +.0023 / +.0009 | cue/working; LMM `$LOG/b_imaging_lmm_r.log` (11-08 09:53) | 4 / 76 post | UNCHANGED (inputs identical) |
| S19 | Frozen decoder tracks deficit; not back at recovery | frozen -.097, z -3.35; F at recovery +.065 | same (same input as C4) | cue | 4 | UNCHANGED |
| C5 | Spread tracks deficit | z -1.8, q .071 | not re-run; cue window, source independent | cue | 4 | UNCHANGED |
| S20 | Coding ceiling | chronic +.040; PS94 .060 -> .045 | not re-run; cue (11c cue_working) | cue | 4 | UNCHANGED |
| S7 | Far-contra code on chronic hits ~70-77 % of pre | lick -.288, cue -.269, pre-cue -.273 | not re-run. `d_decoder_retention_r2` already ran with `WIDEFIELD_LICK_SOURCE=lp` (11-07 14:21, `$LOG/d_decoder_retention_r2.log` header) | LP attempt classes | 4 | UNCHANGED |
| S8 | Failed attempts ~no far-contra code | acute failed lick -.10 / -.13 / .17 / .19 | same (same step) | LP | 4 | UNCHANGED |
| S9 | Behavior back, cortex gap opens | gap -.294 [-.431, -.144] | same. `d_recovery_timing_r2` already ran with LICK_SOURCE=lp | LP | 4 | UNCHANGED |
| C7 | Per-animal half-recovery day | not estimable in 3/4 | same (same step; aim table still missing, P4) | LP | 4 | UNCHANGED |
| W4 | Chronic crossnobis vs June->August drift | .63/.93, .96/1.15, .42/.43, .61/.63 | not re-run; `g5_june_aug` is cue-window frozen (cue/working rows) | cue | 4 | UNCHANGED |
| S21 | Frozen chronic deficit within drift | .04/.13, .09/.30, .23/.21, .13/.19 | same | cue | 4 | UNCHANGED |
| S18 | Template amplitude / capture | .96->1.20 ... | not re-run (cue) | cue | 4 | UNCHANGED |
| N1 | Matched-pre control, WORKING half (Fig 3.4 control) | acute far ipsi / mid / contra -.221 / -.148 / -.259 (Holm .0045 each); per animal far contra -.15 / -.34 / -.23 / -.31; chronic near ipsi +.152 | not re-run (working = camera classes, source independent). `$ROOT/rebuild_figs/$TAG/refit_matched_pre_stats.csv` | cue, working, refit floor of that run | 4 / 799 pairs | UNCHANGED |
| D6-L | Matched-pre, LICK half | section 1 #3 | section 1 #3 | cue, lick class DAQ -> LP, LPfinal store | 4 / 799 pairs (LP) | **CONCLUSION-CHANGES** |

### 2b. Section G (poststroke readouts; OLD = r4, DAQ grouping; NEW = LP, camera working vs no_response, b83f376d; G4 from the 90a5e25f G4-only re-run)

The PS93_0813 unit is skipped under LP (no video). That is 76 post units, vs 88 in r4, which also had 12 June units.
These are my per-animal session means from the json. No bootstrap.

| ID | Readout | OLD | NEW | Settings | n | Flag |
|---|---|---|---|---|---|---|
| G-cue | Frozen post-cue accuracy, all trials | acute .468 / .314 / .436 / .611; chronic .844 / .675 / .771 / .782 | acute .467 / .314 / .436 / .610; chronic .844 / .675 / .771 / .782 | cue, all trials | 4 / 76 | UNCHANGED |
| G-pre | Frozen pre-cue accuracy, all trials | acute .349 / .258 / .345 / .344; sub .465 / .287 / .462 / .369; chronic .380 / .385 / .534 / .388 | acute .231 / .190 / .285 / .298; sub .301 / .250 / .408 / .325; chronic .268 / .291 / .420 / .360 | pre-cue, DAQ ENL 2 s -> LP 1 s | 4 / 76 | Lower in every epoch and animal. Without a LP pre reference the retained ratio cannot be judged (section 3). Treat as NOT-YET-AVAILABLE for the claim |
| G-lick | Lick-only post-lick accuracy | section 1 #2 | section 1 #2 | 0-2 s from first lick; DAQ lick trials -> camera attempts | 4 / 76 | **CONCLUSION-CHANGES** |
| G1 | No-lick trials look like licking trials? | section 1 #8 | section 1 #8 | train pre engaged vs no-lick (DAQ none -> camera no_response) | 4 -> 2 animals | **CONCLUSION-CHANGES** |
| G3 | Impaired no-lick positions | UNDECIDABLE (too few no-lick trials) in 88/88 units | UNDECIDABLE in 76/76 | precue / cue | 4 | UNCHANGED |
| G4 | Post trials by camera class, read by the pre working-vs-no-response boundary (fraction "working-like", spout-balanced, cue window, acute) | Old frame negative = DAQ `none`. No no_response rows scored (n = 0 everywhere). contact .84 / .67 / .87 / .88; incomplete .25 / .19 / .26 / .68; jaw_only .30 / .27 / .09 / .21 | Negative = camera no_response. contact .85 / .68 / .82 / .93; incomplete .16 / **.47** / .28 / .66; jaw_only .45 / .29 / .04 / .26; **no_response — / — / .01 / .18** (pre reference .15 / .08 / .09 for PS93-95; no PS92 / PS93 post rows); pre-cue no_response .58 / .58 (PS94 / PS95 acute) | cue and pre-cue; camclass/z-contact; spout-balanced, cells < 10 rows excluded | 4 (no_response post: 2) | CHANGED, claim not located (I found no stated G4 numbers on the page or FINDINGS). PS93 incomplete moves .19 -> .47; no_response is now scorable in 2/4 |

### 2c. Coding directions (Fig 9.1)

| ID | Claim | OLD | NEW | Settings | n | Flag |
|---|---|---|---|---|---|---|
| 9.1-ov | Shared mode rotates (overlap below ceiling) | acute overlap / ceiling .54-.85, frac no worse <= .07 in every cell | acute .51-.87, frac <= .05 in every cell; sub .63-.89 | cue / lick / pre-cue, gate lick, K = 2-3 | 4 | UNCHANGED |
| 9.1-mag | Shared mode grows (magnitude above ceiling) | section 1 #5 | section 1 #5 | same | 4 | **CONCLUSION-CHANGES** (acute) |

### 2d. Maps (rest-referenced, RESTnl) and drift

| ID | Claim | OLD | NEW | Settings | n | Flag |
|---|---|---|---|---|---|---|
| W6 | Acute far-contra map loss, cue, REST vs late pre | .58 (968 / 2609 bins) | .5797 (968 bins), identical in `$DRL` | cue/working, REST, pre-normalized maxstat | 4 | UNCHANGED |
| P8 / 7.x | Per-animal acute cue far-contra amplitude | FINDINGS RESTW .50 / .45 / .53 / .40 | .38 / .46 / .67 / .98 (not lick-related) | cue/working | 4 (PS95 1 session) | **CONCLUSION-CHANGES** (section 1 #7) |
| M-pre | Pre-cue far-contra amplitude acute | RESTnl 1.40 / 1.05 / 1.00 / 1.08 | .83 / .61 / .73 / .85 | ENL 2 s DAQ -> 1 s LP | 4 | **CONCLUSION-CHANGES** (section 1 #4) |
| M-lick | Lick-window far-contra acute amplitude (page: LP lick .72, 3/4) | DAQ rebuild: empty (no acute far-contra DAQ lick trials) | RESTnl .52 / .58 / .88 / 1.06; REST .46 / .53 / .67 / 1.05; RESTW .48 / .53 / .79 / 1.06 (3/4 below pre; PS95 not) | lick 0-2 s, LP attempts; `$LOG/g5_restnl_maps_lp_r2.log` = `$LPFM/lpfinal_amplitude_per_animal_lick_lick.csv` | 4 (PS94 5 acute sessions) | CHANGED-SAME-CONCLUSION (3/4, as on the page) |
| S22 | Drift: far contra beyond drift (cue); far mid (cue) not separable; chronic far-middle PRE-CUE loss beyond drift | cue fC acute .58 vs drift .96; cue fM .90 vs .80; pre-cue fM chronic REST .52 (623 bins) vs drift .98; RESTnl .43 (743) vs .95 | cue rows identical. Pre-cue fM chronic: REST **.32 (1228 bins)** vs drift 1.10; RESTnl .54 (981) vs 1.05. Pre-cue fC acute: REST 1.13 -> .66, RESTnl 1.23 -> .86 | `$DRd` (contact, 11-07) vs `$DRL` (LP with June supplement, 11-09 02:01); drift = LATE - EARLY pre; pre-normalized bootstrap, maxstat | 4 | CHANGED-SAME-CONCLUSION |
| D-lick | Lick-window far-contra acute map vs drift (new: LP with June) | DAQ: no acute far-contra row | REST **.67 (722 bins)** vs drift 1.04; RESTnl .75 (289) vs 1.05; PRECUE ref .58 (654) vs 1.25. Far mid acute 1.08 / 1.18 (not beyond drift) | lick, LP | 4 | NEW (no comparator); same direction as the cue claim |
| 12.1 | Frozen decoder on rest, retained (REST) | .52 / .18 / .23 / .33 acute; chronic .71 / .59 / .47 / .59 | identical after the rest fix (`$LOG/g5_rest_frozen_r.log` vs `g5_rest_frozen.log`) | rest periods, DAQ-contact exclusion | 4 (PS93_0813 skipped) | UNCHANGED |
| 12.1b | Same, RESTnl, August pre | pooled acute d -.109 [-.160, -.052]; chronic -.090 [-.128, -.051] | identical (`g5_rest_frozen_nl_r.log`) | RESTnl mask (DAQ + cam4 licks) | per-animal block lists PS94 / PS95 (53 scored, 6 skipped) | UNCHANGED |

### 2e. Encoder family (own camera-lick definition; not re-run by design, `scratchpad/rerun_settings_check.md` §0)

These rows use COMBINED_ENCODING_LICKS = combined (cam4 + cam1), OROFACIAL_GATE = attempted on the camera attempts
table, and ENL 1.0 s LP. That is already the documented lick class, so none of these steps took
WIDEFIELD_LICK_SOURCE. The 11-08 numbers stand.

| ID | Claim | 11-08 value (= current) | Flag |
|---|---|---|---|
| C2 | Chronic residual anatomy (tjM1 + SSp-m, not ALM) | tjM1 cue +4.7 / +6.8, lick +7.5 / +9.3 x1e3, 4/4 | UNCHANGED |
| C3 | Orofacial-SPECIFIC residual, chronic cue | no-FR +0.97 [-0.27, 2.56], 3/4; FR-A +0.71 [-0.41, 2.02], 3/4 (n.s.) | UNCHANGED |
| C6 | Frozen target-unique R2 stays down | no-FR lick -0.0096 [-0.020, +0.004], 1/4 up | UNCHANGED |
| W5 | Acute orofacial excess, Layer 2 | FR-A +0.023 [-0.0005, 0.049] | UNCHANGED |
| W7 | Acute contra SSp-m drop | -.59 / -.56 / -.12 / +.13 | UNCHANGED |
| S10-S17 | Decomposition, residual SSp-m, global part, matched residual, Layer 2, spout package, executed part, self-fit | as 11-08 | UNCHANGED |
| P1 | Partial refit, chronic, contra SSp-m (SSp-m_right) (FINDINGS S4d: target +0.034, executed +0.054, +0.064 specific, 4/4) | now available (`$LOG/b_partial_frA.log`, 11-08 13:53). Target cue per animal .073 / .018 / .046 / .050, pooled **+.042 [.019, .065], 4/4 up**; target lick +.053 [.027, .085], 4/4. Executed cue .120 / .050 / .038 / .024, pooled +.051 [.022, .091], 4/4; executed lick +.059 [.026, .106], 4/4. Executed - target cue +.010 [-.018, .034] (.047 / .032 / -.009 / -.026), lick +.006 [-.021, .029] | CHANGED-SAME-CONCLUSION for target and executed. I could not match the "+0.064 specific" quantity to a row in the log |
| P12 | Within-lick correction summary | now `.done` (`e_wlc_summarise`, 11-08 13:05, 5db29b07); per-animal tables only; no verdict line to compare | NEW (no comparator) |

### 2f. ITI / quiet / gain-bio / balance families

| ID | Claim | OLD | NEW | Flag |
|---|---|---|---|---|
| S23-S25 | ITI offset / lateral / decoder r | as 11-08 | not re-run (own video licks) | UNCHANGED |
| S26 | Quiet orofacial events up acutely | rate_both +0.74 / +0.45 / +0.21 | not re-run (`gb_quiet_lp`, QS lick source daq+lp) | UNCHANGED |
| P5 | Quiet-event gain stats (11-08: `gb_stats` FAILED) | FINDINGS x5.1 / x2.5 / x1.4 (rate, face-regressed) | `gb_stats_r` done 11-08 15:57 (rest fix; R5 = restdock05 + engaged). **Per-animal** R5 corr gain (unadjusted) acute / sub / chronic: PS92 1.25 / 2.45 / 1.50, PS93 1.60 / 1.33 / 1.48, PS94 1.29 / 1.44 / 1.49, PS95 1.32 / 1.04 / 1.10 (`gb_summarise_r.log`). LMM lgain corr_hp acute +.329 [.089, .569], p .022, Holm .13 (sub +.282, chronic +.402, Holm .13). The log prints `*` at raw p < .05: none survives Holm in A_gain. oro_post r415_hp acute +.227, Holm .017 | NEW (no comparator): a different quantity from x5.1. Stars here are LMM-based, not boot_delta |
| P9 | Cued orofacial BALANCE predicts unseen-animal recovery (FINDINGS cvR2 .50 vs days .23) | 11-08: no producing step | `l_recov_markers_r2` (11-08 23:28): cued oro_bal LOAO **.489** vs days .226 (shuffle p .000). Primary `l_recov_primary_r2`: cued first lick -> aim LOAO .656 vs days .226. Per animal r .69 / .89 / .93 / .75 (days r .92 / .89 / .87 / .93) | UNCHANGED vs FINDINGS |
| ENL / ENLv | ENL-slide, ENL target, ENLv balance / curves | never ran before (chains parked) | done 11-09 (`l_enl_target_r2`, `x_enlv_balance_r4`, `x_enlv_curves_r4`). Example: `l_enl_target_r2`, frozen position fc_lat_p, lick acute -1.08 [-1.32, -.85], 4/4 (-1.37 / -1.23 / -.91 / -.83); pre-cue acute -.21 [-.27, -.14], 4/4 | NEW (no comparator) |

---

## 3. NOT-YET-AVAILABLE

| Item | Why | Paths checked |
|---|---|---|
| C5b: pre-cue (ENL) spread tracks the deficit (LMM) | `b_imaging_lmm_r` ran 11-08 09:53. That is before D7 (b27a7acd, 12:33) and before the LP ENL arm. It reads the pre-cue `*_sessions_long.csv` from the DAQ epoch folder (2 s DAQ ENL), and the LP ENL arm wrote only 5r + 14 families. No LP pre-cue spread input exists | `$LOG/b_imaging_lmm_r.log`; `wf_render_v10/scripts/imaging_lmm.py` inputs; `$EPL` listing |
| Imaging LMM re-run in general (C4 / C5 / S19 / S20) | not re-run. C4 / S19 marked UNCHANGED because the recovery-trajectory input is identical; C5 / S20 because they are cue-window | `$LOG/b_imaging_lmm_r.*` (no newer run) |
| Section G LP figures and `section_g_retained.csv` (retained vs pre for G-pre / G-lick; G2b per position; G6) | only `g5_g4_fig_redraw` ran on LP. No `g7_section_g_figs` on the LP json. `$ROOT/figures_working/data/section_g_retained.csv` is the r4 (DAQ) version | `$ROOT/figures_working/LP_licks_gateLP_$TAG/` (json + `section_g_g4_camclass.png` only); `$LOG/g7_section_g_figs*` |
| Cue-window refit families after D7 (Fig 1.2 5r cue, Fig 4.2 5rmgap, 5rmo deck sidecar) | the LP cue arm ran only `scal mat`. The DAQ-folder 5r / 5rgap cue outputs are dated 11-05 (pre-D7). The D7 re-run of recovery_trajectory came out identical, which suggests little change, but I did not verify this for 5r | `$EPd/epoch_5rgap*_cue_working*` (mtime 11-05 19:16) |
| Interhemispheric / cross-area / coupling on RESTnl (P10 coupling; Figs 14.18 S10 / S17 / S18) | code on `feat/interhemi-restnl-1109` (helper). The v11 restdock05 chain was marked `.done NOT RUN` | `$LOG/l_interhemi_rest_r5.done`, `l_interhemi_r5.done`, `l_xa_r5.done`, `l_cv_sm_r5.done` |
| v11 steps (queue_daqlk11) | running: `b_rsplit_PS92_r5` (log 06:08, no .done), `l_nl_order_r5` (claim, no .done); the rest of B5E5 / L5R not started (fig_rsplit, fig_resid_per_animal, exec_vs_*, z_fig_pooled, e_mm, e_lmm_*, e_iti_lick_rate, e_pat_*, e_fpa_*, enlv target code, ITI prelick, code subspace) | `$LOG/*_r5.*` |
| Later LP lick decoders (pcd, nolick, pre-cue states, joint xsession on LP) | not queued yet (running list 11-08 17:45) | `$ROOT/figures_working/$TAG/joint_xsession_*`, `precue_states_*` (DAQ only) |
| Fig 9.2 c_migration ("far contra drifts toward far center"), grant LP lick units (5c / 5d / 9 / 10 / 10b), Haufe lick maps (Figs 14.3 / 14.6 / 14.9) | the outputs exist (`g6_grant_lp` 04:05; `$EPL/epoch_14*_lick_lick*`), but **I did not extract their numbers** in this pass | `$ROOT/labcams/grant_figures/LP_licks_gateLP_$TAG/grant_*`; `$EPL/epoch_14*` |
| P2 frozen encoder trial-to-trial variance on contact trials | still no producing step | `$LOG/t_te_readout.*` (`done_nopartial`, `failed_nocalib`) |
| P4 aim recovery link / aim half-recovery | `e_rlink_aim_r` FAILED (aim table missing); not re-run | `$LOG/e_rlink_aim_r.failed` |
| P7 goal summary figure | `d_fig_goal_summary` still RESERVED_HOLD, not run | `$LOG/d_fig_goal_summary.*` |
| P11 rest carries spout position (observed / null 1.634) | no producing step found | - |
| Contact-only section G arm (supplement) | deferred by Priya 11-09 | - |

## Counts

- CONCLUSION-CHANGES: 8 rows (C1, L-fM, D6-L, G-lick, G1, 9.1-mag, P8, M-pre). These are the 8 items in section 1.
  C1 also carries the acute far-middle and far-ipsi lick-window cells, which now pass correction.
- CHANGED-SAME-CONCLUSION: 6 (L-abs, S6, L-gap, M-lick, S22, P1).
- UNCHANGED: 44. 2a: 20 (W1-W4, S1-S5, S7-S9, S18-S21, C4, C5, C7, N1). 2b: 2 (G-cue, G3). 2c: 1 (9.1-ov).
  2d: 3 (W6, 12.1, 12.1b). 2e: 13 (C2, C3, C6, W5, W7, S10-S17). 2f: 5 (S23, S24, S25, S26, P9).
  **37 of the 44 were not re-run**: cue-window, encoder-family or already-LP outputs that do not depend on the
  corrected inputs. C4 and S19 count here because their LMM was not re-run, though its input was re-run and came out
  identical. The other 7 were re-run and came out identical (G-cue, G3, 9.1-ov, W6, 12.1, 12.1b) or match FINDINGS (P9).
- NOT-YET-AVAILABLE: 14 (the 13 items in section 3, plus G-pre, whose retained ratio needs the LP section G figures).
- Extra labels: NEW (no comparator) 4 (D-lick, P5, P12, ENL/ENLv); CHANGED, claim not located 1 (G4).
