# Engagement findings (09-20) re-tested on CAMERA trial classes -- 2026-11-05

**Question.** Do the three 09-20 engagement findings (DECISIONS "WHERE THE TWO THREADS LANDED" + "THE WITHIN-SESSION
DECLINE IS PARTLY FATIGUE AFTER ALL") still hold when the quit / gate is defined by camera classes instead of DAQ licks?
The findings are: (1) quitting is time-driven; (2) a within-session decline that is not the quit tail; (3) no
within-session motor fatigue. The acute ILI deficit is included as well.

**Camera rules** (CAMERA_CLASS_INVENTORY_20261104 §3h / answers). Source table: rebuild
`qc/sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70/trial_attempts.csv` (11-04 19:16). No `daqlk`
trial_attempts table exists yet; only the hv-encoder folders carry `_daqlk`. Classes were matched to DAQ cues by cue time
(20 ms), the same way trial_classes.cue_classes does it.
- Camera quit = the first trial after the last camera lick (stop_kind "tail"). A quit is only called when the tail is
  >= 6 trials and starts >= 300 s after the first cue.
- `cam_tail` gate = all camera-classified trials minus the tail.
- `cam_work` gate = cam_tail minus the mid-session stop runs (>= 6 no_response / jaw_only).

**Sessions and methods.**
- Pre = August only. Epochs are pinned.
- 107 camera sessions. PS92_0908 uses covered trials only (17.5 % uncovered).
- DAQ-only row: PS93_0813, a pre session with no video. Its near Q5-Q1 = +1.5, near ILI = 176.8 ms. It is dropped from
  the camera rows. The June sessions have no camera classes.
- Statistics are the original ones (quit_prodrome / lick_bout_structure / the 09-19 within-animal slope), with the
  original 2.0 s response window. Measures are still DAQ licks; only the trial selection / quit changes. Changes use
  boot_delta; the slope CI uses lick_bout_structure's TEST-2 draw.

**Reproduced first.**
- Recomputed live on the 120 sessions: licks/trial Q5-Q1 near pre -3.8 / acute -8.3 / subacute -10.7 / chronic -3.3.
  Gap change at subacute is -6.8 near / -5.8 far.
- Decel acute near: +10.1 [+0.9, +19.3]. ILI acute: +12.5 [+4.2, +20.7]. All match the documented values.
- From the 09-19 CSV, the quit slope is +0.72 [+0.33, +1.09] for licks and -0.13 [-0.43, +0.15] for time (documented
  +0.72 [+0.36, +1.06] / -0.13 [-0.43, +0.12]).

## F1 quitting is time-driven (slope on log early rate; budget = 0, clock = +1)
| set | n quits | PS92 / 93 / 94 / 95 licks slope | pooled licks | pooled time |
|---|---|---|---|---|
| DAQ 09-19 CSV | 45 | +0.76 / +1.86 / +0.84 / +0.27 | +0.72 [+0.33, +1.09] | -0.13 [-0.43, +0.15] |
| DAQ, Aug, camera sessions | 48 | +1.05 / +1.60 / +1.18 / +0.22 | +0.87 [+0.38, +1.27] | -0.05 [-0.39, +0.26] |
| CAMERA quit | 40 | +0.67 / +1.45 / +0.89 / +0.83 | +0.91 [+0.40, +1.34] | -0.06 [-0.32, +0.19] |
| CAMERA quit + camera licks | 40 | +0.35 / +0.91 / +1.23 / +0.82 | +1.03 [+0.72, +1.35] | -0.06 [-0.24, +0.15] |

The camera quit comes a median +12.2 min after the DAQ quit (IQR +6.1 to +21.1; 38 sessions where both call one).
**Verdict: HOLDS (slightly stronger; 4/4 animals positive).**

## F2 decline not the quit tail (licks/trial, (Q5-Q1) minus own pre)
| set | subacute near | subacute far | acute near | per animal, subacute near |
|---|---|---|---|---|
| DAQ, Aug | -6.7 [-10.9, -2.9]* | -6.7 [-11.3, -2.9]* | -4.4 n.s. | -12.0 / -5.8 / -2.9 / -6.3 |
| CAM tail | -8.7 [-12.2, -5.0]* | -7.4 [-11.6, -4.0]* | -6.6 [-13.8, +0.6] | -12.0 / -5.8 / -8.1 / -8.9 |
| CAM work (stop runs out too) | -7.0 [-11.0, -3.4]* | -6.8 [-11.4, -3.0]* | -4.6 n.s. | -12.0 / -5.8 / -3.9 / -6.4 |

**Verdict: HOLDS.** It is not the camera tail, and it is not the mid-session stop runs either. It is 4/4 animals at
subacute.

## F3 no within-session motor fatigue
- The median ILI never rises within a session. Pooled ILI Q5-Q1 is <= 0 in every epoch x near / far cell in all sets
  (CAM tail near: pre -4.2, acute -6.8, subacute -9.1, chronic -2.5). **HOLDS.**
- The within-bout deceleration steepening at acute (the 09-20 "fatigue after all" support):

| set | acute near | per animal |
|---|---|---|
| DAQ 120 (June in pre) | +10.1 [+0.9, +19.3]* | +13.4 / +12.5 / +14.7 / -0.4 |
| DAQ, Aug pre | +6.6 [-2.6, +15.4] | +8.8 / +7.2 / +14.1 / -3.6 |
| CAM tail | +7.3 [-3.3, +17.4] | +4.8 / +16.3 / +13.8 / -5.8 |
| CAM work | +5.8 [-3.1, +15.4] | +5.9 / +5.0 / +15.9 / -3.7 |

**Verdict: WEAKENS.** The steepening is 3/4 animals and its CI crosses zero. It is lost to the August-only baseline,
not to the camera classes.

## Acute ILI deficit (near median ILI minus own pre)
| set | acute | subacute | chronic | per animal, acute |
|---|---|---|---|---|
| DAQ 120 (June in pre) | +12.5 [+4.2, +20.7]* | +1.7 n.s. | -5.8* | +4.1 / +16.5 / +6.8 / +22.4 |
| DAQ, Aug = CAM tail = CAM work | +18.8 [+11.8, +25.8]* | +8.0 [+1.7, +14.1]* | +0.5 n.s. | +11.9 / +20.6 / +13.7 / +29.0 |

**Verdict: CHANGES (baseline, not camera).** The acute deficit is larger (4/4 animals). It has NOT recovered by
subacute (+8.0*); it recovers by chronic. The camera gate makes no difference.

## Caveats
- The tag is stopA2jexrt3500c70 (pre-daqlk). The lick-detector switch changes 0 hits and restores 0.58 % of licks, so
  the effect here should be negligible. It is not re-run.
- Licks are still the DAQ discriminator of 09-20, with a 2.0 s hit window (the session window is now 3.5 s).
- The camera quit needs a >= 6-trial tail, which is my threshold (matched to the stop-run rule). A tail is "everything
  after the last camera lick" (A2), so a short tail would otherwise always count as a quit.
- The 10-07 lick_bout CSV (from-csv) gives a decline-pacing test of -0.10 / +0.83, against -0.48 / +0.50 documented on
  09-20. That pacing claim has already drifted on DAQ data and was not re-tested here.
- Outputs: M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/engagement_camera_1105/
  (engagement_camera_report.txt, f1_quit_records.csv, f2f3_session_summary.csv, fig1-3, repro_daq/).
