# Licks inside "stopped" time: census, 2026-11-04 (read-only)

Priya, 11-04: by definition there should be NO licks in stopped time. This census checks whether the stop rule respects that, and where it does not.

The census uses the per-trial raw classes of round 2 (`STOP_RULE_GUARD_CENSUS_R2_20261104.md`), under profile `cohort1_rebuild_20261103` and window cue_stop. There are 107 cohort-1 sessions and 56,979 trials. The arms are reproduced exactly: A0 has 5925 stopped trials, A2 has 5611 and A3 > 10 has 5175. A2 is primary.

No repo code was edited and nothing was committed. Nothing was written to M:. One process was used. The scripts are in the scratchpad folder `C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/slc/`: `gather.py`, `analyze.py` and `q2.py`-`q8.py`.

## Summary

**1. The stop rule never lets a cue-response lick into stopped time.** This was verified on every trial.
- Category (iii), cue + 70 ms .. max_rt: 0 camera licks and 0 DAQ contacts in any stopped trial, under A0 and A2.
- Category (iv), max_rt .. stop: also 0. Today those licks count toward the class, so a trial holding one is never no_response.
- Category (ii), 0-70 ms: 1 camera lick in the whole set, and 0 DAQ contacts.

**2. Stopped time does contain licks: in the pre-cue period and in the ITI.**

| | runs (any lick in span) | camera licks | DAQ contacts | trials with a pre-cue lick (camera / DAQ) | ITIs with a lick (camera / DAQ) |
|---|---|---|---|---|---|
| A2 mid-session runs | **88 / 200** | 698 | 421 | 131 / 86 | 58 / 7 |
| A0 mid-session runs | 111 / 237 | 838 | 490 | 159 / 98 | 81 / 11 |
| quit tail (same in A0 and A2) | **19 / 59** | 114 | 80 | 23 / 17 | 12 / 4 |

- 273 of the 5611 A2 stopped trials (4.9 %) are touched by a lick: 233 in mid-session runs and 40 in the tail. "Touched" means a lick in the trial's own pre-cue / 0-70 ms, or in the ITI on either side of it.
- Per epoch, touched trials / A2 stopped trials:
  - pre: 50 / 819
  - acute: 141 / 2220
  - subacute: 51 / 1706
  - chronic: 31 / 866
- The pre-cue licks sit at a median of **−2.8 s** before the cue (10-90 %: −4.2 to −2.1 s). Many are real bouts: up to 16 camera licks with matching DAQ contacts in one pre-cue period.
- Every DAQ contact in stopped time has a camera lick too. "Runs with a DAQ contact" is a strict subset of "runs with a camera lick" (55 of 88 for A2).
- cam1-only events add nothing here: they are post-cue only.

**3. The pending max_rt / 70 ms rule barely moves anything.**
- 15 trials lose their lick class:
  - 11 whose only lick or contact is after max_rt;
  - 4 whose only camera lick peaks at 0-70 ms.
- Under A2, 2 of those 15 become stopped. They lengthen two existing acute PS93 runs by 4 trials and 1 trial.
- **0 new stop runs; A2 stopped trials 5611 → 5616; the tail is unchanged (1313).**
- 18 more trials go from contact to incomplete (DAQ contact only after max_rt, camera lick inside it).
- Every cohort hit rate moves by ≤ .004.

**4. A "break" rule enforcing no licks in stopped time costs pre almost nothing and acute about one point.**
- Rule: any camera lick (by peak) or DAQ contact anywhere in the span breaks the run at that point, and runs are re-formed with ≥ 6 and A2.
  - A2 stopped trials: 5611 → 5339 (−272: 32 pre, 153 acute).
  - Pre pooled hit: .956 → .954.
  - **Acute pooled hit: .667 → .656** (deficit −.289 → −.298).
  - Acute close contra: .911 → .886. Acute close ipsi: .933 → .922.
- The restored trials are no_response / jaw_only trials next to non-cue licking, so most of them are probably disengagement. They lower the acute close-spout hit rates, as A1-trimmed did in round 2.
- The break rule alone still leaves 343 pre-cue licks at the START of re-formed runs, because the trial after a pre-cue lick opens the next run.
- A strict form also takes the lick-touched trial out of the run, which leaves zero licks in stopped time. It costs:
  - pre pooled .956 → .953;
  - acute pooled .667 → .652;
  - acute close ipsi .933 → .911.
- **The alternative that leaves the hit rates unchanged** is to EXCISE the 273 lick-touched trials from "stopped", putting them in neither class. "Stopped" is not in the hit denominator, so every hit rate stays at the A2 value, and stopped time is then lick-free.

**5. Rest-period consistency.**
- All 200 A2 mid-session runs lie inside the camera working period. That period runs from the first to the last camera-lick TRIAL (`camera_engaged_period`); no run starts before the first lick trial. So RESTnl draws docked rest frames from inside stopped runs.
- RESTnl already removes every DAQ and cam4 lick with a [−0.5, +2.0] s buffer, so the licks found here are cut out of rest. Rest and stopped disagree only on the label, not the frames.
- The quit tail is outside the working period, so it gives no rest.
- In 19 of 59 sessions there are camera licks after the last lick TRIAL: pre-cue or ITI licks in the tail. The "last camera lick" therefore means the last lick trial in the class rule, but a later lick event exists.
- Starting the tail after the last lick EVENT instead would move 269 tail trials (1313 → 1044) back into the working period.

**Decisions needed (Priya):**
- (a) Which categories break, or excise, stopped time? Pre-cue / ENL licks are 85-90 % of the camera licks in stopped spans, so the choice is mainly about them.
- (b) Break and re-form (lowers acute close-spout hit about 1-2 points), or excise the touched trials (hit rates unchanged)?
- (c) Should the quit tail start after the last lick EVENT, not the last lick trial?

## Sources (whole-session tables, file:line in `C:/Users/SabatiniLab/Github/wf_rebuild`)

- **cam4 LP licks, whole session:** `<orofacial>/lp_r4_heatmap_20261013__clean_v36/sessions/<animal>/<date>/licks_all.csv`.
  - Built by `wfield_local/orofacial_outputs.py:323-393` (`all_licks_session`) and written at `:404`.
  - It combines `licks.csv` (post-cue), `licks_precue.csv` and `licks_iti.csv`, with `context` and ABSOLUTE DAQ times `daq_on_s` / `daq_peak_s`.
  - Read the same way by `lp_licks.py:278-327` (`_inputs`, `_read_times`) and `restnl_mask.py:103-110, 137`.
  - The class itself reads only `licks.csv` (on_ms in [0, stop]): `trial_classes.py:198-211`.
  - Category assignment here uses `daq_peak_s` (lick PEAK).
- **cam1-only events:** `<dlc>/session_poses/cam1_licks/<animal>_<date>_events.csv`.
  - Used by `trial_classes.py:212-223`: not cam4_matched, not in_3d, not a droplet; `jaw_only_trial` rows = "tongue seen".
  - Also used by `lp_licks.py:321-326`, which keeps in_3d rows.
  - **These are POST-CUE only:** on_ms ≥ 0 in all 107 sessions. They are not a whole-session table, so cam1 cannot see pre-cue or ITI licks. Times = cue_daq + `t_peak_ms`.
- **DAQ contacts, whole session:**
  - Function: `dlc_frames.lick_onsets` (`wfield_local/dlc_frames.py:220-254`), from `daq_trials.decode`. Its cache is `<dlc root>/lick_onsets/<animal>_<date>.npz` (`:182-183`).
  - Consumers:
    - `trial_classes._daq_contact_onsets` (`trial_classes.py:142-148`) and `trial_contact_times` (`:151-160`);
    - `scripts/pose_kinematics_demo.py:28-37` (`daq_contacts_ms`, cue + 70 ms .. stop);
    - the class column `n_daq_contacts` (`trial_classes.py:232-235`, `CONTACT_FROM_CUE_S` `:86`).
  - NB `lick_onsets` WRITES that cache on a miss (`_store_licks`, `:213-217`). To keep M: untouched, this census read the npz directly: all 107 sessions are present and the params digest matches.
  - Check: the DAQ contacts in cue + 70 ms .. stop agree with `n_daq_contacts > 0` on 100 % of trials, and no contact trial lacks one.
- **Trial bounds:** `windows.npz` via `trial_classes._trial_times` (`:131-139`): cue_daq, strobe_ms and stop_ms.
  - The span of a stopped run is strobe(first trial) .. stop(last trial), including the ITIs between its trials.
  - Strobe was present for every trial, so the cue − 3 s fallback was never used.
  - Events before the first strobe or after the last stop are unassigned: 110 cam4 licks and 871 DAQ contacts, cohort-wide.
- **Categories, per event relative to the trial holding it:**
  - (i) strobe .. cue;
  - (ii) cue .. cue + 70 ms;
  - (iii) cue + 70 ms .. min(cue + 3.5 s, stop);
  - (iv) cue + 3.5 s .. stop;
  - (v) stop .. next strobe.
  - The ITI after the LAST trial of a run is outside the span.
- **Pending rule, as modelled:**
  - contact = DAQ contact in (iii);
  - incomplete = camera lick PEAK in (iii), or cam1 "tongue seen" (left untimed, as today);
  - jaw / no_response / unknown unchanged.
- **Rest:** `trial_classes.camera_engaged_period` (`:562-580`) and `_lick_span` (`:583-591`). The RESTnl lick sources are DAQ + cam4 `licks_all` onsets, buffer (0.5, 2.0) s: `restnl_mask.py:11-15, 52, 59, 137`.

## T1. Licks inside stopped spans, A2 mid-session runs, per animal x epoch

i = trials with a pre-cue lick, ii = trials with a 0-70 ms lick, v = ITIs (inside the span) with a lick. Categories (iii) and (iv) are 0 everywhere, so they are omitted.

| animal | epoch | runs | trials | runs w/ any | runs w/ cam | runs w/ DAQ | cam i | cam ii | cam v | DAQ i | DAQ v | cam licks | DAQ contacts | jaw_only in runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PS92 | pre | 1 | 17 | 1 | 1 | 1 | 2 | 0 | 4 | 2 | 1 | 21 | 16 | 7 |
| PS92 | acute | 15 | 536 | 11 | 11 | 9 | 20 | 0 | 9 | 16 | 1 | 113 | 75 | 192 |
| PS92 | subacute | 1 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| PS93 | pre | 2 | 18 | 2 | 2 | 0 | 2 | 0 | 3 | 0 | 0 | 7 | 0 | 10 |
| PS93 | acute | 21 | 572 | 13 | 13 | 7 | 22 | 0 | 9 | 11 | 0 | 141 | 98 | 216 |
| PS93 | subacute | 7 | 164 | 6 | 6 | 2 | 8 | 1 | 7 | 2 | 1 | 22 | 10 | 78 |
| PS93 | chronic | 2 | 15 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 0 | 12 |
| PS94 | pre | 4 | 113 | 3 | 3 | 3 | 12 | 0 | 3 | 8 | 1 | 47 | 36 | 53 |
| PS94 | acute | 38 | 700 | 21 | 21 | 13 | 34 | 0 | 8 | 24 | 2 | 155 | 94 | 253 |
| PS94 | subacute | 16 | 272 | 3 | 3 | 2 | 3 | 0 | 0 | 2 | 0 | 7 | 4 | 103 |
| PS94 | chronic | 6 | 107 | 3 | 3 | 3 | 5 | 0 | 2 | 5 | 1 | 35 | 14 | 43 |
| PS95 | pre | 16 | 317 | 3 | 3 | 1 | 3 | 0 | 3 | 1 | 0 | 12 | 4 | 142 |
| PS95 | acute | 1 | 12 | 1 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 6 |
| PS95 | subacute | 44 | 1074 | 12 | 12 | 8 | 11 | 0 | 5 | 8 | 0 | 94 | 45 | 368 |
| PS95 | chronic | 26 | 373 | 8 | 8 | 6 | 7 | 0 | 4 | 7 | 0 | 39 | 25 | 192 |
| **all** | pre | 23 | 465 | 9 | 9 | 5 | 19 | 0 | 13 | 11 | 2 | 87 | 56 | 212 |
| **all** | acute | 75 | 1820 | 46 | 46 | 29 | 78 | 0 | 26 | 51 | 3 | 411 | 267 | 667 |
| **all** | subacute | 68 | 1518 | 21 | 21 | 12 | 22 | 1 | 12 | 12 | 1 | 123 | 59 | 552 |
| **all** | chronic | 34 | 495 | 12 | 12 | 9 | 12 | 0 | 7 | 12 | 1 | 77 | 39 | 247 |
| **all** | all | 200 | 4298 | 88 | 88 | 55 | 131 | 1 | 58 | 86 | 7 | 698 | 421 | 1678 |

A0 mid-session runs (full table in `stopped_licks_runs_by_animal_epoch.csv`):

| epoch | runs | runs w/ any | cam licks | DAQ contacts | jaw_only in runs |
|---|---|---|---|---|---|
| pre | 23 | 9 | 87 | 56 | 212 |
| acute | 104 | 66 | 547 | 335 | 802 |
| subacute | 75 | 23 | 126 | 59 | 580 |
| chronic | 35 | 13 | 78 | 40 | 255 |
| all | 237 | 111 | 838 | 490 | 1849 |

Jaw_only trials make up 39 % of A2 mid-run trials (1678 / 4298) and 38 % of the tail (504 / 1313). They are fine by definition, since there is no tongue.

## T2. Quit tail (after the last camera-lick trial; identical in every arm)

| animal | epoch | sessions with a tail | tail trials | tails w/ any | cam i | cam v | DAQ i | DAQ v | cam licks | DAQ contacts | jaw_only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PS92 | pre | 3 | 5 | 1 | 1 | 1 | 1 | 1 | 4 | 2 | 2 |
| PS92 | acute | 5 | 147 | 3 | 6 | 3 | 5 | 2 | 35 | 31 | 52 |
| PS93 | pre | 2 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| PS93 | acute | 3 | 83 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 |
| PS93 | subacute | 1 | 41 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 25 |
| PS93 | chronic | 3 | 92 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 53 |
| PS94 | pre | 6 | 201 | 4 | 5 | 2 | 3 | 0 | 14 | 7 | 69 |
| PS94 | acute | 6 | 168 | 4 | 8 | 2 | 7 | 1 | 43 | 30 | 55 |
| PS94 | subacute | 5 | 97 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 40 |
| PS94 | chronic | 6 | 77 | 2 | 1 | 1 | 0 | 0 | 2 | 0 | 26 |
| PS95 | pre | 6 | 112 | 1 | 1 | 0 | 1 | 0 | 12 | 10 | 45 |
| PS95 | acute | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PS95 | subacute | 4 | 50 | 2 | 1 | 1 | 0 | 0 | 2 | 0 | 15 |
| PS95 | chronic | 8 | 202 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 86 |
| **all** | all | 59 | 1313 | 19 | 23 | 12 | 17 | 4 | 114 | 80 | 504 |

The other 48 sessions end on a lick trial, so they have no tail. In the tail, categories (ii)-(iv) are 0, as they must be, since these trials come after the last lick trial.

## T3. Pending max_rt rule (peak in cue + 70 ms .. min(cue + 3.5 s, stop))

Class changes from today's raw classes, all 56,979 trials:

| today → pending | trials |
|---|---|
| contact → incomplete | 18 |
| contact → jaw_only / no_response | 6 / 1 |
| incomplete → jaw_only / no_response | 4 / 4 |
| everything else | unchanged |

- **15 trials lose their lick class.**
  - 11 have their only lick or contact after max_rt.
  - 4 have their only camera lick peaking at 0-70 ms (onset before the cue).
- They are spread over all four animals and over pre (3), acute (8), subacute (3) and chronic (1).
- Under pending A2 they end up as 8 jaw_only, 5 no_response and **2 stopped**.
- The 2 stopped trials extend two acute PS93 runs:
  - PS93_0819: 440-466 → 436-466;
  - PS93_0821: 436-444 → 435-444.
- **New stop runs: 0. Runs lost: 0.**
- Pending A2 stopped trials are 5616 (+5); pending A0 5931 (+6); the tail is unchanged.
- So the pending rule creates **2 stopped trials that hold a post-max_rt lick**, plus 1 that holds a 0-70 ms lick.
- Cohort hit rates move by ≤ .004 in every cell.
  - Pre pooled .956 → .956; acute pooled .667 → .667.
  - Largest shifts: subacute far contra .741 → .737; pre far ipsi .898 → .896.
- Full tables: `stopped_licks_hit_cohort.csv`, rule = `pend`.

## T4. Enforcing "no licks in stopped time": break rule vs A2

- **Break rule:**
  - A lick or contact in trial k's (i) or (ii) breaks the run before k.
  - One in (iv), or in the ITI after k, breaks it after k.
  - Runs are then re-formed with ≥ 6 and the A2 both-sides test.
- **Strict:** in addition, the trial whose own pre-cue / 0-70 ms holds the lick is not a run member.
- **Tail-event:** the quit tail starts after the last lick EVENT, not after the last lick trial.

| variant | stopped | mid runs | mid trials | tail | pre stopped | acute stopped | pre pooled hit | acute pooled hit | acute pooled deficit | residual licks in stopped spans |
|---|---|---|---|---|---|---|---|---|---|---|
| A2 (today) | 5611 | 200 | 4298 | 1313 | 819 | 2220 | .956 | .667 | −.289 | 698 cam + 421 DAQ (mid) |
| A2 + break, categories i-v, cam + DAQ | 5339 | 223 | 4026 | 1313 | 787 | 2067 | .954 | .656 | −.298 | 343 pre-cue, all in a run's first trial; 1 at 0-70 ms |
| A2 + break, i-v, + tail-event | 5313 | 234 | 4269 | 1044 | 783 | 2045 | .954 | .654 | −.300 | |
| A2 + break, i, ii, iv only (ITI does not break) | 5381 | 215 | 4068 | 1313 | 789 | 2074 | .954 | .657 | −.298 | |
| A2 + break, ii, iv only (post-cue only) | 5611 | 201 | 4298 | 1313 | 819 | 2220 | .956 | .667 | −.289 | splits 1 run |
| A2 + break, DAQ contacts only | 5465 | 209 | 4152 | 1313 | 802 | 2132 | .955 | .661 | −.294 | |
| A2 + break, camera licks only | 5339 | 223 | 4026 | 1313 | 787 | 2067 | .954 | .656 | −.298 | = cam + DAQ |
| **A2 + strict break** | 5255 | 218 | 3942 | 1313 | 769 | 2024 | .953 | .652 | −.301 | **0** (mid) |
| A2 + strict break + tail-event | 5228 | 229 | 4184 | 1044 | 764 | 2002 | .953 | .651 | −.302 | |
| **A2 + excise touched trials** (neither class) | 5338 | n/a | n/a | n/a | 769 | 2079 | **.956** | **.667** | **−.289** | 0 |
| A3 > 10 (today) | 5175 | 135 | 3862 | 1313 | 786 | 2068 | .954 | .655 | −.299 | |
| A3 > 10 + break i-v | 4715 | 137 | 3402 | 1313 | 715 | 1812 | .950 | .637 | −.314 | |

- Camera-only and cam + DAQ give identical results, because DAQ contacts in stopped time always come with camera licks.
- The excise counts are 5611 − 273 trials. Its hit rates are unchanged because "stopped" is not in the denominator.

### T4a. Cohort-mean hit rate per spout, pre and acute (canonical order)

| epoch | variant | close ipsi | close mid | close contra | far ipsi | far mid | far contra | pooled |
|---|---|---|---|---|---|---|---|---|
| pre | A2 | .980 | .979 | .981 | .898 | .937 | .958 | .956 |
| pre | A2 + break | .977 | .979 | .979 | .897 | .934 | .957 | .954 |
| pre | A2 + strict break | .976 | .979 | .977 | .894 | .934 | .956 | .953 |
| acute | A2 | .933 | .915 | .911 | .707 | .452 | .052 | .667 |
| acute | A2 + break | .922 | .904 | .886 | .697 | .441 | .051 | .656 |
| acute | A2 + strict break | .911 | .902 | .881 | .696 | .435 | .051 | .652 |
| subacute | A2 | .944 | .954 | .941 | .875 | .896 | .741 | .893 |
| subacute | A2 + break | .932 | .952 | .935 | .874 | .889 | .739 | .888 |
| chronic | A2 | .978 | .980 | .982 | .955 | .981 | .965 | .973 |
| chronic | A2 + break | .976 | .980 | .979 | .950 | .980 | .963 | .972 |

Acute deficit vs pre:

| variant | close ipsi | close mid | close contra | far ipsi | far mid | far contra | pooled |
|---|---|---|---|---|---|---|---|
| A2 | −.047 | −.065 | −.070 | −.191 | −.485 | −.906 | −.289 |
| A2 + break | −.055 | −.075 | −.093 | −.201 | −.494 | −.905 | −.298 |
| A2 + strict break | −.065 | −.076 | −.097 | −.199 | −.499 | −.905 | −.301 |

### T4b. Per animal, pre and acute: hit, A2 → A2 + break → A2 + strict break

The full table with denominators and stopped counts is `stopped_licks_hit_pre_acute_A2_vs_break.csv`.

| epoch | animal | close ipsi | close mid | close contra | far ipsi | far mid | far contra | pooled |
|---|---|---|---|---|---|---|---|---|
| pre | PS92 | .992 → .986 → .986 | .994 = | .988 → .986 → .986 | .953 = | .983 = | .972 = | .980 → .979 → .979 |
| pre | PS93 | .976 = | .980 = | .998 = | .717 → .715 → .715 | .852 → .845 → .845 | .972 → .968 → .968 | .918 → .916 → .916 |
| pre | PS94 | .989 → .979 → .976 | .980 = | .984 → .982 → .976 | .984 → .984 → .980 | .968 → .964 → .963 | .948 = | .976 → .973 → .971 |
| pre | PS95 | .965 = | .964 → .961 → .961 | .952 → .950 → .949 | .938 → .937 → .930 | .945 = | .939 → .939 → .938 | .951 → .950 → .948 |
| acute | PS92 | .915 → .894 → .866 | .912 → .902 → .899 | .942 → .926 → .918 | .805 → .793 → .791 | .690 → .682 → .663 | .046 → .045 → .045 | .723 → .711 → .703 |
| acute | PS93 | .905 → .892 → .886 | .924 → .898 → .894 | .961 → .915 → .906 | .539 → .535 → .535 | .775 → .745 → .739 | .124 → .122 → .122 | .705 → .687 → .684 |
| acute | PS94 | .921 → .909 → .901 | .881 → .874 → .874 | .800 → .763 → .758 | .648 → .620 → .619 | .243 → .237 → .237 | .030 → .029 → .029 | .596 → .580 → .578 |
| acute | PS95 | .992 = | .942 = | .942 = | .838 = | .100 = | .009 = | .646 → .645 → .644 |

**Reading:**
- In pre, no cell moves more than .012 (PS94 close ipsi). Most pre cells move ≤ .007, and every pre pooled rate moves ≤ .005.
- In acute, three of four animals lose 1-5 points at the close spouts, the spouts the stroke barely affects. PS95 has almost no acute stop runs, so it does not move.
- So the break rule mainly moves disengaged trials back into the denominator. It is the A1-trimmed pattern from round 2.
- The far contra deficit is unchanged (−.905).

## T5. Rest-period consistency

- **Inside the working period.** 0 of 200 A2 mid-session runs start before the session's first camera-lick trial, and by construction none reaches past the last one. Every mid-session stopped run therefore lies inside `camera_engaged_period`, which gates the rest baseline.
- **Frames.** RESTnl removes every DAQ contact and every cam4 `licks_all` onset, each with a (0.5, 2.0) s buffer. The licks counted here are therefore already cut out of rest, and the rest frames that come from stopped runs are lick-free.
  - The one gap would be cam1-only licks, which RESTnl does not use. They are 0 in stopped spans, because cam1 events are post-cue only and any post-cue cam1 lick makes the trial incomplete.
  - The inconsistency is in the LABEL: a stopped run is called a lick-free stop, but 88 of 200 contain non-cue licking.
- **End of the working period.** Both the rest gate and the quit tail end at the last camera-lick TRIAL. 19 of 59 tails contain later pre-cue / ITI licks (114 camera, 80 DAQ), so the last lick EVENT is later than the period end in those sessions.
  - Moving the end to the last event would put 269 tail trials back into the working period.
  - Those 269 trials split into new runs or isolated lapses: T4 rows "tail-event".
- **Under the pending rule** the last lick trial moves in 0 sessions (tail 1313 → 1313), so the rest period is unchanged.

## T6. Five example runs with licks inside (A2, mid-session)

Times are seconds from that trial's cue, using the lick peak for the camera. Trial index i is in cue order. The full list is `stopped_licks_examples.csv`.

1. **PS93_0821 (acute), trials 341-349 (9 trials: 6 jaw_only, 3 no_response; far_mid x5, close_contra x4).**
   - Pre-cue (i) bouts in three consecutive close_contra trials:
     - i = 346: 16 camera licks −5.48 .. −2.75 and 16 DAQ contacts −5.50 .. −3.00;
     - i = 347: 6 camera licks −3.58 .. −2.67 and 5 DAQ contacts;
     - i = 348: 8 camera licks −3.81 .. −2.54 and 6 DAQ contacts.
   - The animal licks the spout before each cue, then does not respond to the cue.
   - Under the break rule, no run of ≥ 6 is left and all 9 trials return to the working period.
2. **PS92_0809 (pre), trials 423-439 (17 trials: 10 no_response, 7 jaw_only).**
   - i = 423: pre-cue (i) at −2.56 and −2.38 (DAQ −2.59), then an ITI (v) lick at +4.77.
   - i = 436: ITI lick +6.76 (camera and DAQ).
   - i = 437 (close_ipsi): pre-cue (i) bout of 14 camera licks −4.36 .. −2.44 with 14 DAQ contacts, then ITI licks +3.73 and +6.41.
   - i = 438: ITI lick +3.78.
   - The break rule keeps 424-436 (13 trials) as stopped.
3. **PS93_0822 (subacute), trials 440-511 (72 trials).**
   - i = 440: ITI (v) licks at +5.26 and +5.97.
   - i = 448: pre-cue (i) lick −1.99.
   - i = 496: pre-cue (i) lick −0.24 and the only (ii) lick in the dataset, peaking at +0.00 s.
   - i = 503: pre-cue (i) lick −0.43.
   - The break rule splits the run into 441-447, 448-495, 496-502 and 503-511, all still ≥ 6, so the stopped trial count barely changes.
4. **PS92_0820 (acute), trials 443-552 (110 trials, 72 no_response / 38 jaw_only).**
   - i = 480 (close_ipsi): pre-cue −2.31 (camera and DAQ).
   - i = 492: ITI licks +4.94, +5.30 and +6.17.
   - i = 493: pre-cue −2.04 and −1.76.
   - i = 498: pre-cue −2.57 and −2.41 (camera and DAQ).
   - i = 530: pre-cue −2.77 (camera and DAQ).
   - The break rule gives 443-479, 480-492, 498-529 and 530-552. Trials 493-497 (5) return to the working period.
5. **PS95_0806 (pre), trials 470-522 (53 trials).**
   - Single ITI (v) licks after i = 486 (+5.45) and i = 507 (+4.90), with no DAQ contact.
   - The break rule gives 470-486, 487-507 and 508-522, all ≥ 6, so nothing is restored.

## Files (this folder)

- `stopped_licks_runs.csv`: every run and tail, with lick and contact counts per category and source. The columns `rule` (cur / pend) and `variant` (A0, A2, A3gt10, A2_brk_all, A2_brk_strict, A2_brk_all_tailevt) say which run definition it came from.
- `stopped_licks_runs_by_animal_epoch.csv`: T1 / T2 for A0 and A2 (runs with licks per category).
- `stopped_licks_variant_totals.csv`: T4 totals for every variant, today's rule and pending.
- `stopped_licks_hit_cohort.csv` and `stopped_licks_hit_by_animal_epoch_spout.csv`: hit rate and deficit vs pre, every variant and both rules.
- `stopped_licks_hit_pre_acute_A2_vs_break.csv`: T4b.
- `stopped_licks_examples.csv`: T6 lick times.
- `stopped_licks_checks.csv`: per-session checks.
  - DAQ contacts in cue + 70 ms .. stop agree with `n_daq_contacts` on 100 % of trials.
  - No-lick trials with a (iii) / (iv) camera peak or DAQ contact: 0.
  - By lick ONSET there are 2 such trials; in_3d cam1 events would add 11. Both are outside today's class sources, so they are not in the counts above.
