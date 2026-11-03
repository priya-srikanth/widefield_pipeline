# Jaw-only trials after the session's last contact: same as the others? (11-03, e3)

Question (Priya 11-03): the 807 jaw-only trials after a session's last DAQ contact (5b) — keep them in "working" for
uniformity, unless they look different from the earlier jaw-only trials or don't happen before the stroke.

Data: `trial_classes.session_trials` (cue_stop) on every trial of 120 curated sessions (56,979 trials;
working = contact / incomplete / jaw_only = 52,523; jaw_only = 3,412). Jaw measures from v36 `jaw_onsets.csv`.
"After" = trial_id after the session's last contact trial.

**Counts.**
- 807 after-last-contact jaw-only trials = 1.5 % of working. **768 of them (95 %) are already outside the engagement
  gate** (orofacial_cohort.engagement_keep); only 39 are engaged trials.
- They happen **before the stroke too**: 150 pre trials in 15 / 31 pre sessions.
- Share of working by epoch: pre 0.9 % · **acute 3.8 %** · subacute 1.9 % · chronic 1.1 %. Acute has about 4x as
  many as pre, because acute sessions have more jaw-only trials and longer sated tails.
- Per animal × epoch:

  | animal | pre | acute | subacute | chronic |
  |---|---|---|---|---|
  | PS92 | 2 | 161 | 0 | 0 |
  | PS93 | 13 | 39 | 36 | 68 |
  | PS94 | 83 | 74 | 95 | 46 |
  | PS95 | 52 | 6 | 27 | 105 |

- After the last contact, the classes are about 35–44 % jaw_only, 39–47 % no_response, 2–7 % incomplete and
  13–17 % unknown, in every epoch.

**Do they look different?** Yes: smaller and later, in the same way in every epoch, including pre.
- Pooled medians (before → after the last contact), 4 / 4 epochs:
  - cam4 opening smaller: pre 9.5 → 7.3 px, acute 5.9 → 4.1, subacute 6.4 → 5.3, chronic 10.7 → 6.1;
  - onset later: pre 606 → 1168 ms, acute 955 → 1013, subacute 715 → 917, chronic 300 → 802.
  - Mann-Whitney p < 0.03 for both measures in every epoch.
- Per animal × epoch, the 12 cells with at least 10 trials on each side:
  - cam4 opening smaller in 10 / 12 (median −2 px);
  - cam1 opening smaller in 9 / 12;
  - onset later in 9 / 12 (typically +200 ms);
  - 3-D displacement mixed (smaller in 5 / 12).
- Spout position mix: about the same before and after.

**Reading.** These are weak, late jaw movements of an animal that has stopped working (sated). They are not attempts
like the earlier jaw-only trials. The difference exists pre-stroke too, so it is not a stroke effect. But their share
of "working" is about 4x higher acutely, so leaving them in adds more sated trials to acute than to pre.

**Recommendation (e3):** do NOT count them as working.
- Primary gate: per-trial working (contact / incomplete / jaw_only) AND before the session's last contact. This is
  5b's "quit-tail AND jaw-or-lick" option.
- Keep the plain per-trial gate as the robustness row (5b already runs both).
- Kinematics trial-class figures: unchanged (they are engagement-gated, so only the 39 engaged trials are involved).

Tables: scratchpad `a4/quit_tail_alltrials.csv` (per trial). Not in the repo; regenerate from
`trial_classes.session_trials` + `jaw_onsets.csv` if needed.
