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

**Which cutoff (5b, 11-03).** Priya worried that a last-CONTACT cutoff would cut real acute attempts when a session's
last blocks are far contra / far mid. 5b compared three anchors on the s1 snapshot (11e8a846; working trials dropped):

| cutoff | acute dropped | incomplete licks dropped | flag in `s1_trial_classes_cue_stop.csv.gz` |
|---|---|---|---|
| after the last contact | 315 (4.2 %), 86 of them far_R | 94 | `working_pl_contact` |
| **after the last camera lick (contact or incomplete)** | **131 (1.8 %)** | **0 by construction** | `working_pl_lick` |
| after the last close ipsi / mid lick | 302 (4.1 %) | 80 (chronic 482 dropped) | `working_pl_close` |

So her worry is confirmed for the last-contact cutoff. With the **last camera lick**, only jaw-only trials after the final
lick of any kind go. That is 504 trials (1.0 % of working; pre 0.8 %, acute 1.8 %, subacute 1.0 %, chronic 0.8 %, so
acute is about 2x pre instead of 4x). 128 of them are pre-stroke, in 15 / 31 pre sessions; only 28 are engaged trials.
They still look different from the earlier jaw-only trials:
- cam4 opening smaller in all four epochs (p < 0.04), and smaller in 9 / 12 animal × epoch cells;
- onset later in 9 / 12 cells (pooled: pre 615 → 1168 ms, chronic 392 → 802 ms; acute and subacute not significant).

**Recommendation (e3, with 5b):** primary "working" = contact / incomplete / jaw_only AND not after the session's last
camera lick (`working_pl_lick`). Keep the plain per-trial gate as the robustness row; the section 1 decoders already
compute all three arms. If adopted, the rule moves from `scripts/s1_trial_class_snapshot.py` into
`wfield_local.trial_classes` as a gate. It's Priya's decision; nothing has changed yet. The kinematics trial-class
figures are engagement-gated and unaffected (28 trials).

Tables: scratchpad `a4/quit_tail_alltrials.csv` (per trial). Not in the repo; regenerate from
`trial_classes.session_trials` + `jaw_onsets.csv` if needed.
