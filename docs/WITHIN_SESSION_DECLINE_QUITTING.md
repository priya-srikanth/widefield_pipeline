# Within-session decline of aim and latency vs quitting (cohort1, 2026-11-07)

Branch `analysis/within-session-decline-quitting-20261107`. Script `scripts/within_session_decline.py`
(extract / analyze / plot). Outputs `M:/MICROSCOPE/Priya/Widefield/within_session_decline/` (`trials.csv.gz`,
`tables/`, `figures/`). Cohort1 only (PS92-95), August-only pre, curated sessions. **Exploratory, n = 4 animals.**

Priya (11-07), on the state-space observation that first-lick latency slows and far-contra landing drifts
ipsilesionally within post-stroke sessions: *"this is important; i wonder how this relates to the animals 'quitting'
after stroke. can we look into this"*.

## 0. Decisions needed from Priya first

(filled in at the end, section 6)

## 1. What was already established (not re-tested here)

| claim | number | source |
|---|---|---|
| Acute orofacial motor deficit in the lick RHYTHM (median ILI) | +12.5 ms [+4.6, +19.9], 4/4, flat across the session (already in Q1), recovered by subacute | STATUS_2026-09-20, DECISIONS 09-20 |
| ILI does not rise within a session | 0 / 16 epoch x position cells | same |
| BUT "no within-session motor fatigue" was revised the same day | within-bout deceleration +15-25 ms at every epoch incl. chronic (4/4); it steepens across the session acutely (+10.1 [+0.9, +19.1]); bouts shorten across the session subacute / chronic near; the decline in licks / trial is paced about half by licks and half by the clock | DECISIONS "THE WITHIN-SESSION DECLINE IS PARTLY FATIGUE AFTER ALL" (09-20), `lick_bout_structure` |
| Within-session ENGAGEMENT decline (licks / trial), not the quit tail | Q5-Q1 pre -3.8, acute -8.3, subacute -10.7, chronic -3.5; steepening vs own pre significant SUBACUTE (-6.8 near / -5.8 far) | STATUS_2026-09-20 |
| The QUIT is time-paced, not lick-budget-paced | log(quit licks) on log(early rate) +0.72 [+0.36, +1.06]; log(quit time) -0.13 | `quit_point` |
| Quit prodrome in LICK RATE | -18.7 licks/min [-22.9, -13.7] in the 10 min before, vs time-matched controls | `quit_prodrome` |
| Animals quit sooner early post | KM restricted mean -15.1 min [-25.4, -5.6] 4/4, -77 trials 4/4; chronic ~0 | disengagement tests (S28) addendum |
| Satiety at the quit | collected water at quit-or-end early post -290 ul (Holm .001, 4/4) -> quits at LESS water; on equal-length (80 min) data -105 ul [-225, +10] 3/4 -> satiety not rejected; weight -5 % | disengagement P1 + addendum |
| Fatigue at the quit | P3 n.s.: bouts shorten before quits, tongue does not slow (within-bout ILI) | disengagement P3 |
| Frustration (far-contra block lapses) | +0.037 n.s. | disengagement P2 |
| Quit state in cortex | no prodrome in cortex / face / running; cue response falls AT the quit | disengagement P4 |
| Latency tracks aim within animals | r -0.72 to -0.92 | revup_behaviour (FINDINGS) |

**What was NOT established:** (i) whether the 3-D AIM (lateral landing) or the first-lick LATENCY degrade across a
session -- every earlier within-session test used DAQ lick counts / intervals, never where the tongue lands or how
soon the first lick comes; (ii) whether aim / latency show a prodrome before the quit (the S28 prodrome tests looked at
lick rate, within-bout ILI, cortex, face, running); (iii) whether a session's aim drift predicts how early it quits;
(iv) whether the drift is specific to the impaired (far-contra) direction; (v) a time vs effort vs water separation for
aim / latency (the 09-20 pacing test was for licks / trial and for the quit).

**Reconciling with "no within-session motor fatigue".** That line (CLAUDE.md pointer to STATUS_2026-09-20) is stale:
it was about the lick-rhythm PERIOD (median ILI), which a brainstem CPG holds fixed, and DECISIONS revised it the same
day. Aim (where the first lick lands) and latency (how soon the first lick starts) are different measures from ILI:
a slower start and a drifting aim can coexist with a fixed cycle period. Nothing in the earlier work contradicts the
new observation; the question is what paces it.

## 2. What was done here

(see sections 3-5; written as the analysis ran)
