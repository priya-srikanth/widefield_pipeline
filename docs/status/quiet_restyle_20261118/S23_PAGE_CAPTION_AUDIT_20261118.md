# Movement page section 23: caption audit, old (pre-rebuild captions) vs new (sync-end run), 2026-11-18

Four read-only audits of the kept section-23 figures (23.3, 23.7-23.10, 23.12-23.14, 23.16, 23.19, 23.20, 23.22, 23.24) against
a8's sync-end re-renders. NOTE: these audits used the scripts' LMM stars. The page uses a8's later `*_boot_stars.csv`
(nested bootstrap = *, LMM q < .05 = dagger). Calls that changed under the bootstrap: 23.24 limb 4-7 Hz stays starred;
23.3 whole-cortex acute sync starred; 23.20 orofacial ACF half-time starred; 23.12 acute proxy frequency and chronic
regressed rate dagger only; 23.16 chronic cued induced dagger only, subacute ITI induced starred. The "PS94 movement
events flip" in the 23.7 audit is an artifact of pooled counts over unequal session numbers (not on the page).
23.8 was re-run after Priya's quit = quit..last sync pulse (code 9701d914); its final numbers are in the page caption
and quit_state/still_up_rates_eng_vs_quit.csv, not in the audit below.
Priya OK'd the conclusion changes 11-18 (page v325: 11 figures; v326: 23.7 + 23.8).

# Section 23 kept figures: old (pre-rebuild captions) vs new (sync-end run) — 2026-11-18


<!-- out_A.md -->
# Section 23 OLD (caption) vs NEW (sync-end run): Figs 23.3, 23.24, 23.9, 23.10

R = .../cohort1_quiet_spikes/v2_quiet_lp_c1guard_syncend_20261118
Every new log header shows: QS_ENGAGED_END=lastsync, QS_LICK_SOURCE=daq+lp, FROZEN_LICKS_ALL=split, code wf_qstyle1118 67d49b99.
Sessions: 108 (pre 32 / acute 16 / subacute 17 / chronic 43). 415-nm spectra exist for 104 of them, so all sync LMMs use n = 104 (quiet_sync_4to8.log L29-35; lmm_sync.csv column n).
Acute sessions per animal: PS92 5, PS93 4, PS94 6, PS95 1 (quiet_sync_4to8.log L32-35).

Applies to all four captions (method):
- The captions say the lick definition includes "cam1-only labels far from the mouth (med_out < -120 px), which this pre-rebuild render counts as licks". The new run is the c1guard run with FROZEN_LICKS_ALL=split, so that clause and "Output root v2_quiet_lp/" no longer describe it.
- "Engaged" in the new run ends at min(quit, last sync edge) (QS_ENGAGED_END=lastsync). The captions do not say this.
- "PS93 08-13 has no video (DAQ only)" still holds (quiet_fluct.log L26 warning: no cam4 LP table).
- The captions define stars as LMM vs pre with FDR, and the new pngs do the same. These four outputs contain no bootstrap-CI ("boot_delta") stars, so under the 11-07 convention (bootstrap stars, LMM daggers) the bootstrap stars are MISSING for all four figures.

---

## Fig 23.3: synchronization index (quiet_sync_index.log; quiet_f0_noise.log)

| quantity | old (caption) | new | source |
|---|---|---|---|
| acute orofacial sync change (LMM) | +0.68, q < .001 | +0.690, q < .001 | quiet_sync_index.log L159 |
| acute orofacial per animal | - | +0.587 / +0.815 / +0.701 / +0.594 (4/4 positive) | L50-53 |
| acute whole cortex | +0.30 | +0.243, p .029, q .051 (q .036 in the 4to8 FDR family) | L143; quiet_sync_4to8.log L201 |
| acute limb (bilateral) | +0.04, n.s. | +0.111, q .53, n.s. | L147 |
| ordering orofacial > cortex > limb | yes | holds (0.69 > 0.24 > 0.11) | - |
| acute orofacial ipsi - contra (withdrawn) | +0.13, 4/4 | +0.124 (LMM p .023, q .023); per animal +0.088 / +0.189 / +0.066 / +0.353, 4/4 | L175; quiet_f0_noise.log L492-498 |
| fast-band laterality agreement (4-8 Hz, T22 "C") | holds in only 2/4 | 2/4 (PS92 -0.031, PS93 +0.167, PS94 -0.037, PS95 +0.162) | quiet_f0_noise.log L508-514 |
| limb shifts the same way (4-8 Hz) | yes | limb ipsi - contra 4-8 Hz +0.130, q .033 (f0 family) / .042 (4to8 family) | f0 L811; 4to8 L219 |
| 4-8 Hz keeps X % of laterality | 57 % | 33 % (+0.041, q .44) | quiet_sync_4to8.log L279 |
| 4-7 Hz keeps X % | 41 % | 30 % (+0.038, q .53) | 4to8 L284 |
| "both n.s." | yes | yes | - |
| limb shifts equally (re-check) | yes | limb 4-8 +0.130 (q .042), larger than orofacial +0.041 | 4to8 L219 |
| 415 nm 4-8 Hz lower ipsilesionally | yes | orofacial -0.066 q .003; limb -0.076 q < .001 | 4to8 L270, L264 |
| non-calcium 7-11 Hz peak lower on lesion side | yes | 415 nm 7-11 Hz orofacial ipsi - contra acute -0.086, q < .001, 4/4 negative | f0 L838, L535-538 |
| "no ipsilesional F0 drop" | yes | still true. Acute F470 orofacial ipsi - contra is a RISE: +0.035, q < .001 (per animal +0.070 / +0.042 / +0.005 / -0.002); F470/F415 +0.017, q < .001 | f0 L727, L155-161, L793 |
| orofacial 1-4 Hz power rise | yes | acute log10 change +0.743 / +0.880 / +0.461 / +0.745, 4/4 | quiet_sync_index.log L82-85 |
| 415 unchanged in 1-4 Hz (Fig 23.11); no ipsi bias (Fig 23.13) | - | not in my outputs (other figures) | - |
| n | n = 4 | 4 animals, 104 sessions in the LMM | lmm_sync.csv |
| per-session marker counts | - | not printed per measure (session counts as above) | - |

**Conclusion changes:**
- None of the headline claims change. The orofacial sync rise holds (+0.69, q < .001, 4/4). The withdrawn laterality is still 4/4 in the 8-15 Hz index, and the fast bands keep only 2/4. 415 nm 4-8 Hz power is still lower ipsilesionally, and limb shifts as much as orofacial.
- Whole cortex acute is now +0.243 with q .051 in its own family, so it gets no star in the new png. The caption gives no star or q for cortex, so I cannot tell whether a star was lost.
- Retention numbers fall: "4-8 Hz keeps 57 % ... 4-7 Hz 41 %" becomes 33 % and 30 %. This only strengthens the withdrawal. The caption text must change.
- New chronic results, not mentioned in the caption: whole cortex chronic +0.425, q < .001 (**), and limb chronic +0.275, q .038. The png shows ** on whole cortex chronic and * on limb contra chronic.

**Method:** the n, window and lick source in the caption should be updated as noted in the general section above.

**PNG (sync_index):** five panels: orofacial ipsi, orofacial contra, limb ipsi, limb contra, whole cortex. Session markers, animal-mean bars, LMM stars in the top band. The png has NO ipsi - contra contrast panel and no panel letters. The caption says "The ipsi - contra difference (+0.13, 4/4) shown here", but the difference is not shown in this png; it is only in f0_noise row 2 panel A and sync_4to8 panel A. The f0_noise png (T22 check) has three rows: F0, T20 variants A-D plus 8-15-above-floor and 415 7-11 Hz, and noise-floor diagnostics. It matches the "F0 / noise-floor check" description.

---

## Fig 23.24: sync laterality with fast bands (quiet_sync_4to8.log)

| quantity | old (caption) | new | source |
|---|---|---|---|
| A 8-15 Hz orofacial ipsi - contra acute | +0.141, q .002 | +0.124, q .036 | 4to8 L210 |
| A per animal PS92/93/94/95 | +0.07 / +0.23 / +0.11 / +0.29, 4/4 | +0.088 / +0.189 / +0.066 / +0.353, 4/4 | L42-45 |
| B 4-8 Hz | +0.080, q .11 | +0.041, q .44 | L225 |
| B % of 8-15 | 57 % | 33 % | L279 |
| B per animal | -0.03 / +0.23 / +0.03 / +0.16 | -0.031 / +0.167 / -0.037 / +0.162 | L50-53 |
| "clear in PS93 and PS95 only" | yes | holds. PS94 changes sign from +0.03 to -0.04; still not clear | - |
| PS95 one acute session | yes | yes (n = 1) | L35 |
| C 4-7 Hz | +0.058, q .30 | +0.038, q .53 | L240 |
| C % | 41 % | 30 % | L284 |
| C k/4 | 2/4 | 2/4 (-0.018 / +0.172 / -0.045 / +0.166) | L58-61 |
| D floor-subtracted 4-8 | +0.051, q .37 | **-0.046**, q .87 (n = 83 sessions) | L255 |
| D % | 36 % | **-37 %** | L289 |
| D per animal | - | -0.081 / +0.144 / -0.237 / NaN (PS95) | L66-69 |
| limb ipsi - contra 4-8 | +0.099, q .015 | +0.130, q .042 | L219 |
| limb ipsi - contra 4-7 | +0.091, q .018 | +0.098, **q .073** | L234 |
| orofacial minus limb laterality (two bands) | -0.025 / -0.038, n.s. | 4-8: -0.082, q .44; 4-7: -0.047, q .53; 8-15: -0.070, q .27 (all n.s.) | L228, L243, L213 |
| 415 4-8 orofacial ipsi - contra | -0.062, q .015 | -0.066, q .003 | L270 |
| 415 4-8 limb ipsi - contra | -0.071, q < .001 | -0.076, q < .001 | L264 |
| DAQ-only frames 4-8 Hz | +0.064 = 48 % | **MISSING** (only daq+lp was run; no DAQ-only output in R) | - |
| bilateral orofacial 4-8 Hz index acute | +0.54, q < .001 | +0.330, **q .004** | L222 |
| n | n = 4 | 4 animals, 104 sessions (D: 83 / 58 / 52 / 50) | L29, lmm.csv |

**Conclusion changes:**
- The floor-subtracted 4-8 Hz value changes sign: "floor-subtracted 4-8 Hz +0.051, q .37 = 36 %" becomes -0.046 (-37 %). It is n.s. both times. This supports the withdrawal, but the sentence must be rewritten.
- The limb 4-7 Hz result loses its significance: "limb ipsi - contra shifts as much (4-8 Hz +0.099, q .015; 4-7 Hz +0.091, q .018)" now reads 4-7 Hz q .073. The png has no star there (panel C, limb row). Limb 4-8 Hz keeps its star (q .042). The qualitative claim gets stronger: limb 4-8 Hz +0.130 is about 3 times orofacial +0.041.
- "orofacial minus limb laterality is ~0 in every band (-0.025 / -0.038, n.s.)": still n.s. in every band, but 4-8 Hz is now -0.082, which is harder to call "~0". Suggest saying "n.s. in every band".
- "8-15 Hz +0.141, q .002" becomes +0.124, q .036. The star is still there (one * in the png, possibly ** before) and the result is still 4/4.
- "4-8 Hz +0.080, q .11 = 57 %" becomes +0.041, q .44 = 33 %. "4-7 Hz ... = 41 %" becomes 30 %.
- "+0.54, q < .001" becomes +0.330, q .004. Still a rise, still starred, but smaller.
- The reading "no neural ipsilesional synchronization in any band; the T20 laterality withdrawal stands" still holds, and more strongly.

**Method:** the DAQ-only comparison was not re-run (MISSING). The window and lick wording should change as in the general section. "August pre" matches (WIDEFIELD_PRE_FROM=0801; README).

**PNG:** columns A 1-4/8-15, B 1-4/4-8, C 1-4/4-7, D floor-subtracted 4-8, E 415 nm 4-8. Rows: orofacial ipsi - contra, limb ipsi - contra, orofacial minus limb. This matches the caption. The png subtitle already prints the new 33 % / 30 % / -37 %. Stars:
- orofacial row: A acute *, E acute **
- limb row: A acute **, B acute *, E acute *** and subacute ***
- orofacial minus limb row: A chronic *

---

## Fig 23.9: quiet power spectrum (quiet_fluct.log; quiet_spectrum_bands.csv)

| quantity | old (caption) | new | source |
|---|---|---|---|
| acute orofacial peak frequency | ~2-3 Hz | peak plateau about 2-4 Hz in the geometric mean, read off the png | quiet_spectrum.png bottom row |
| peak ratio, ipsilesional | x4 | geometric-mean curve peaks at about x8 (png). Printed value: **MISSING**; the log prints band ratios only. 1-3 Hz band per animal: 7.16 / 10.16 / 4.01 / 7.50 | png; quiet_fluct.log "power ratio" table, orofacial_left 1-3 Hz; spectrum_bands.csv |
| peak ratio, contralesional | x3 | about x6 at ~3.5 Hz (png). Printed value: **MISSING**. 1-3 Hz per animal: 7.20 / 5.50 / 3.63 / 4.47 | same, orofacial_right |
| broad rise below 1 Hz | x2-3 | ipsi 0.1-0.5 Hz: 3.13 / 2.18 / 2.72 / 2.19; 0.5-1 Hz: 2.90 / 2.85 / 3.20 / 3.45. Contra 0.1-0.5: 2.79 / 1.91 / 2.35 / 1.39; 0.5-1: 3.81 / 2.25 / 2.56 / 2.27. png thick curve about x2-2.5 | quiet_fluct.log band table |
| ~9 Hz peak in every epoch | yes | not clearly visible in the new orofacial median PSDs (flat floor about 1e-3 above ~7 Hz). The f0_noise orofacial PSD shows at most a faint bump at 8-9 Hz. The fitted "7-11 Hz peak frequency" by animal: pre 9.19 / 9.59 / 9.62 / 8.37; acute 9.35 / 6.25 / 7.58 / 8.50 | quiet_spectrum.png; quiet_f0_noise.log L482-488 |
| events per session | - | printed in the log (not in the caption) | quiet_fluct.log |

**Conclusion changes:**
- The amplitude is larger: "Acute orofacial power peaks at ~2-3 Hz (x4 ipsilesional, x3 contralesional)" now reads about x8 ipsi and about x6 contra from the png. Every animal's 1-3 Hz band ratio is at least 3.6, and the ipsi > contra ordering holds. Exact peak values are not printed (MISSING), so take numbers from the png or compute them before editing.
- The "~9 Hz peak in every epoch = probably heartbeat residue" claim cannot be confirmed by eye in the new spectrum png. Check this before keeping the sentence.
- New, not in the caption: the subacute orofacial curve also rises (about x5 peak, ipsi). Limb ipsi chronic has one large thin-line outlier near 1 Hz (one animal's chronic curve reaches about x19). The caption says nothing about limb.

**Method:** Welch 4-s, quiet runs >= 4 s, engaged, median over sessions: all match the fluct README and the png title. Window and lick wording as in the general section.

**PNG:** four columns (orofacial ipsi / contra, limb ipsi / contra). Top row: median PSD per epoch. Bottom row: ratio vs pre, thin lines = animals, thick = geometric mean. This matches the caption. No panel letters.

---

## Fig 23.10: per-pixel SD maps (quiet_fluct.log; quiet_sd_maps_significance.csv)

The caption has no effect-size numbers. Its numeric and method items:

| quantity | old (caption) | new | source |
|---|---|---|---|
| test | bootstrap max-statistic FWER p < 0.05, 8x8-px bins, eroded mask | same (2,000 draws, nested animals -> sessions) | quiet_sd_maps_significance.csv |
| pooled significant bins | - | acute 1316 / subacute 1241 / chronic 1338 of 2022 bins | csv rows 12-14 |
| per-animal acute bins | - | PS92 929, PS93 1918, PS94 686, **PS95 0: "< 2 sessions in acute or pre: not tested"** | csv rows 0-11 |
| n | n = 4 | 4 animals, 108 sessions | log |

**Conclusion changes:** none can be checked numerically, because the caption makes no quantitative claim. The caption should say that PS95 acute is untested (1 session); the png already marks it "not tested (1 session)".

**PNG:**
- Rows: PS92, PS93, PS94, PS95, pooled. Columns: pre SD, then acute, subacute and chronic log2(SD / pre), with black outlines. This matches the caption.
- The pooled acute increase is lateral and anterior with a posterior band, and much of the cortex is outlined. Chronic PS92 shows a whole-cortex increase.
- The caption says "black on white halo"; I cannot clearly see a white halo at this resolution.

---

## MISSING items
1. Fig 23.24: DAQ-only frames, 4-8 Hz +0.064 = 48 %. No DAQ-only run exists in R.
2. Fig 23.9: printed peak ratios "x4 ipsi, x3 contra". The new outputs print band ratios only (1-3 Hz etc.); the geometric-mean peak was read off the png (about x8 / x6).
3. Fig 23.9: a printed confirmation of the "~9 Hz peak in every epoch".
4. All four figures: bootstrap-CI stars. Only LMM / FDR stars are produced.
5. Fig 23.3: old cortex and limb q-values and stars were never stated, so I cannot tell whether stars were gained or lost.

<!-- out_B.md -->
# Section 23 OLD vs NEW (sync-end run) -- Figs 23.7, 23.8, 23.20, 23.19

R = .../cohort1_quiet_spikes/v2_quiet_lp_c1guard_syncend_20261118 (code wf_qstyle1118 67d49b99, 2026-11-18)
OLD = .../labcams/cohort1_quiet_spikes/v2_quiet_lp (read only for map significance, to compare contours)

Run settings for all four logs: QS_ENGAGED_END=lastsync, QS_LICK_SOURCE=daq+lp, FROZEN_LICKS_ALL=split, WIDEFIELD_PRE_FROM=0801, c1guard root.
"(derived)" = a ratio or percent I worked out from two printed numbers. Nothing was re-run.

## Method mismatches common to all four captions (item 4)
- The caption says "Output root v2_quiet_lp/" and "cohort1_quiet_spikes/v2_quiet/". The new root is v2_quiet_lp_c1guard_syncend_20261118.
- The caption says the render counts "cam1-only labels far from the mouth (med_out < -120 px) ... as licks". The new run is the **c1guard** run, so that sentence describes the old render and should be removed or rewritten.
- The engaged period now ends at min(quit, last sync edge). The captions say nothing about this; add it.
- The lick source matches (DAQ + LP, env QS_LICK_SOURCE=daq+lp). Note that each script's README.txt still says "no DAQ contact lick". That is a stale docstring. The env line in the log is what actually ran.
- Session n: quit_state, width and steinmetz have 108 sessions (pre 32 / acute 16 / subacute 17 / chronic 43, counted from the session CSVs). **eta_split has 107 sessions**. PS93_0813 is dropped because it has no face video, so it cannot be split into still and move. spread_sessions.csv has 30 pre sessions with events. The caption's "PS93 08-13 has no video (DAQ only)" is true for 23.8 quit_state, 23.19 and 23.20. For 23.7 / 23.8 maps the session is excluded entirely, so the caption should say that instead.

---------------------------------------------------------------------
## Fig 23.7 (f-q_selection): still vs movement UP events, pre vs acute
Source: R/logs/quiet_eta_split.log ("T1 SPREAD" table, "event counts" table), R/eta_split/map_significance.csv

| quantity | old (caption) | new | source |
|---|---|---|---|
| spread, still, pre PS92/93/94/95 | 0.41 / 0.41 / 0.31 / 0.41 | 0.48 / 0.36 / 0.33 / 0.35 | eta_split.log T1 SPREAD, rows "still PSxx", col pre |
| spread, still, acute PS92/93/94/95 | 0.37 / 0.32 / 0.30 / 0.37 | 0.33 / 0.32 / 0.25 / 0.34 | same, col acute |
| animals with lower acute spread | 4/4 | 4/4 (drops -0.15 / -0.04 / -0.08 / -0.01, derived) | same |
| "small" | small | mostly small. PS92 drop is now larger (-0.15). PS95 is ~0 | same |
| still event count ratio acute/pre PS92/93/94/95 | x9 / x5 / x11 / x1.3 | x6.4 / x9.7 / x8.6 / x1.2 (derived: 818/128, 823/85, 1326/155, 410/341) | eta_split.log "event counts", still rows |
| movement event count ratio acute/pre | x4 / x5 / x1.6 / x0.3 | x1.75 / x7.9 / **x0.90** / x0.27 (derived: 70/40, 309/39, 236/261, 61/226) | same, move rows |
| height bins | 4-6 + 6-9 x pre SD, equal weight | h 0-1 (= 4-6, 6-9), matches | log header + png title |
| movement threshold | FR speed > engaged-quiet 90th pct, +/-0.3 s | matches (README) | eta_split/README.txt |
| map significance -0.3 s (still) | "weaker lateral lead-in" (old run: 0 of 2022 bins) | 26 bins, **SUPPRESSED (edge enrichment 2.1x)** | map_significance.csv |
| map significance 0 s (still) | (old: 110 bins) | 8 bins (edge 0.17x, "small") | same |
| map significance +0.2 s (still) | "less lateral recruitment" (old: 223 bins) | 596 bins (edge 0.59x) | same |
| map significance +0.5 / +1.0 s (still) | (old 0 / 0) | 5 (suppressed) / 18 (small) | same |
| n animals in test | n = 4 | 4 | same |

Caveat on the count ratios: the old caption's ratios may have been rate or per-session ratios. The new log prints only raw pooled counts, and acute has fewer sessions than pre. A rate-based ratio is **MISSING** from the new outputs.

Conclusion changes:
- **PS94 movement events flip direction**: they went up (x1.6) and now go down (x0.90, derived from raw counts). Affected sentence: "movement events x4 / x5 / x1.6 / x0.3". PS92 movement drops from x4 to x1.75 and PS93 still events rise from x5 to x9.7, so the per-animal ordering changes. The headline "still events increase in all four, PS95 least" still holds.
- The spread drop is 4/4 in both runs, so there is **no change in sign or k**. The magnitudes changed, so the caption should state the new per-animal values.
- "the residual acute difference is a weaker lateral lead-in at -0.3 s": the new map does show lateral blue in DELTA at -0.3 s, but its contour is SUPPRESSED as a rim artifact, and the old run had 0 significant bins. This was never a tested claim. Make it descriptive or drop it.
- "less lateral recruitment at +0.2 s": still supported and stronger (596 bins). The contour now covers a broad anterior / orofacial-to-lateral band, so it is not only lateral SSp.
- "Pre-stroke still events are already confined to anterolateral orofacial cortex": still true by eye in the pre row.
- No stars or daggers are attached to this figure, so none are gained or lost.

PNG check: T1_up_still shows rows pre engaged (4 an.) / acute eng (4 an.) / DELTA at -0.3, 0, +0.2, +0.5, +1.0 s, height-matched, as the caption describes. T1_up_move and T3_down_still are the same layout for movement and DOWN events. They match the caption. The caption body only describes the still panel; move and DOWN are the supporting panels.

---------------------------------------------------------------------
## Fig 23.8 (f-q_quit_map): pre engaged vs pre quit, still UP events
Which script made the OLD image: the page has `src="q_quit_map.png"`. The local copy C:/Users/SabatiniLab/Github/wf_attempt/_goal_page/q_quit_map.png (dated Nov 1) **is the quiet_eta_split.py T4 figure**: it is titled "T4: orofacial UP events without face movement: pre ENGAGED vs pre QUIT period", with rows "pre engaged (3 an.)" / "pre quit (3 an.)" / DELTA at -0.3 to +1.0 s. So the caption does describe a T4 still pre-engaged-vs-pre-quit ETA map, and the new equivalent is R/eta_split/T4_up_still_pre_eng_vs_pre_quit.png.
Note: FIGURE_PROVENANCE_20261116_rerun.csv and MASTER_MANIFEST list the source as quiet_eta_fr;quiet_fluct, which is wrong. The provenance registry should say quiet_eta_split.py (T4).

| quantity | old (caption) | new | source |
|---|---|---|---|
| animals in map | 3 (PS93-95; "PS92 has no pre quit period") | **4 an.** in the png rows. PS92 now has pre quit still UP events (PS92_0807 n=3, 0811 n=4, 0814 n=4 at h0). The significance test still reports animals = 3 | png row labels; spread_sessions.csv; map_significance.csv |
| "same map; DELTA only a few bins" | few bins (old run: 25 bins at 0 s only) | **210 bins at 0 s (edge 1.24x), 46 at +0.2 s, 20 at +1.0 s (small), 1 at -0.3 s (suppressed)** | map_significance.csv T4 rows |
| still event rate engaged, /min PS93/94/95 | 0.13 / 0.43 / 0.50 | **MISSING** (no per-state rate is printed or tabulated) | -- |
| still event rate quit, /min PS93/94/95 | 8.4 / 10.5 / 3.9 | **MISSING** | -- |
| acute engaged-quiet rate | 2-6 /min | **MISSING** (still-only rate not printed. steinmetz rate_up_hz acute = 0.155-0.40 Hz is all-height, all UP events, so not comparable) | -- |
| spread T4 pre, eng vs quit PS92/93/94/95 | not in caption | eng 0.48 / 0.36 / 0.33 / 0.35. quit 0.24 / 0.19 / 0.19 / 0.21 | eta_split.log "T4 SPREAD" |
| "pre quit data are sparse" | sparse | pre quit-period quiet minutes (median) 0.0 / 0.2 / 2.4 / 7.5. Sessions with >= 60 s quit quiet: 1 / 2 / 5 / 7 | quiet_quit_state.log tables 1-2 |
| quit_state session n | -- | 108 done | quiet_quit_state.log |

Conclusion changes:
- **"Same map in both states (DELTA: only a few bins)" is no longer true as written.** At 0 s, 210 of 2022 bins are significant, outlining orofacial / anterolateral cortex bilaterally plus posterior patches. There are also bins at +0.2 s and +1.0 s. DELTA is mostly blue at 0 / +0.2 s, so the engaged and quit maps differ: by eye the pre-quit 0 s peak is broader and spreads more laterally than pre-engaged. The quit spread index is about half the engaged one (0.19-0.24 vs 0.33-0.48), so quit events are more confined. Their shape is similar but the maps are not statistically the same.
- **"PS92 has no pre quit period" / "(PS93-95)" is no longer true.** PS92 contributes a few events. The figure says 4 animals and the test uses 3, and this needs reconciling before the caption says "n = 3" or "n = 4".
- The rate numbers cannot be checked because they are MISSING. One route is per-state n from spread_sessions.csv divided by eng_min / quit_min in quit_state_sessions.csv, but that would be a recompute and I did not do it.

PNG check: 3 rows (pre engaged 4 an., pre quit 4 an., DELTA) x 5 lags, which matches the caption's description of an engaged-vs-quit still map. The DELTA panel now has substantial contours, which contradicts "only a few bins".

---------------------------------------------------------------------
## Fig 23.20 (f-q_width): event width
Source: R/logs/quiet_event_width.log (per-animal tables lines 138-204, LMM rows lines 206-245), R/width/lmm_width.csv. LMM n = 108 (acf 107).

| quantity | old (caption) | new | source |
|---|---|---|---|
| FWHM orofacial acute vs pre | -0.037 s (~10 % narrower) | **-0.069 s** (SE 0.013, q < 1e-4). Pre level 0.342, so **~20 %** (derived) | log line 227 (coef), 226 (pre_level) |
| FWHM orofacial per animal pre -> acute | -- | 0.323->0.271, 0.348->0.261, 0.362->0.302, 0.366->0.255 (4/4 narrower) | log lines 145-148 |
| FWHM height-matched orofacial acute | -- | -0.086 s, q < 1e-4 (4/4) | line 235 |
| FWHM limb acute | -- | -0.061 s, q < 1e-4 | line 223 |
| decay acute ("all decay -18 %") | -18 % | orofacial -0.048 s on 0.162 = **~-30 %** (derived), q < 1e-4, 4/4. limb -0.043 on 0.197 = ~-22 %, q = 0.0003, 4/4 | lines 214-219, 145-204 |
| rise acute ("unchanged") | unchanged | **orofacial -0.015 s, p = 0.022, q = 0.026. limb -0.016 s, q = 0.014. 4/4 animals lower in both columns** (orofacial 0.148->0.133, 0.150->0.125, 0.143->0.141, 0.160->0.130). PNG shows "*" at acute for both rise panels | lines 239-243. Rise table (log ~line 162). PNG |
| ACF half-time acute ("unchanged") | unchanged | orofacial -0.031 s, p = 0.026, **q = 0.070** (no star). limb -0.021, q = 0.14. Per animal orofacial 3/4 lower (PS94 0.206->0.210) | lines 207, 211. acf table |
| subacute | (not in caption) | FWHM orofacial -0.042 q = 0.003. decay orofacial -0.035 q = 0.0001 (** in png) | lines 220, 228 |
| "NOT robust to movement correction (Fig 23.21)" | -- | not checked here (23.21 is not in this set) | -- |

Conclusion changes:
- **Rise is no longer unchanged.** It now has an LMM star (q < .05) at acute in both orofacial and limb, and 4/4 animals are lower. Affected sentence: "acute events ~10 % narrower (FWHM orofacial -0.037 s), all decay (-18 %), rise unchanged". Decay still dominates the narrowing (about -30 % vs -10 % for rise).
- **The effect size roughly doubled**: FWHM went from -0.037 s to -0.069 s, so "~10 %" becomes "~20 %". Decay went from -18 % to about -22 % (limb) and -30 % (orofacial).
- The ACF stays non-significant (q = 0.07), but it is now borderline and 3/4 animals are lower. "Threshold-free ACF unchanged" still holds at q < .05. Consider "not significantly changed".
- Stars: the png stars are LMM-FDR stars (the title says "stars = LMM epoch vs August pre, FDR"). That breaks the page convention, where "*" = bootstrap CI excluding 0 and the LMM gets a dagger. The squares are bootstrap means +/- 95 % CI, but no boot_delta star is computed. **MISSING: bootstrap-delta stars for this figure.**

PNG check: 2 rows (orofacial, limb) x 5 panels (FWHM, FWHM 4-8 x SD, rise, decay, ACF half-time), with sessions as open markers by epoch. This matches the caption's measures. The star semantics need the fix noted above.

---------------------------------------------------------------------
## Fig 23.19 (f-q_steinmetz): epileptiform criteria
Source: R/logs/quiet_steinmetz_check.log (tables), R/steinmetz_check/steinmetz_check_sessions.csv (108 sessions)

| quantity | old (caption) | new (acute, PS92/93/94/95) | source |
|---|---|---|---|
| acute UP/DOWN ratio | ~1 | 0.79 / 1.08 / 0.97 / 0.67 (~1, PS95 lower) | log up_down_ratio, col acute |
| IEI 0.3-0.5 s obs/Poisson | at or above Poisson | 2.04 / 1.20 / 1.44 / 2.85 (all >= 1) | log iei_0.3-0.5 |
| IEI 0.5-1 s obs/Poisson | (no refractory gap) | 1.89 / 1.29 / 1.14 / 2.18 (all > 1) | log iei_0.5-1 |
| IEI CV | 0.67-0.8 | **0.83-0.95** (0.935 / 0.950 / 0.828 / 0.911) | log iei_cv |
| heights | 2.4-3.1 % | **1.6-2.6** (1.605 / 2.275 / 2.043 / 2.581, med_h) | log med_h |
| share > 10 % | (implied ~0) | 0 in all acute | log share_gt10 |
| rate | 0.10-0.18 Hz | **0.155-0.397 Hz** (0.226 / 0.397 / 0.198 / 0.155) | log rate_up_hz |
| bilateral symmetry | "matches" | **MISSING** (not in this script. README points to T11 / T2. The T2 log2 ipsi/contra table in quiet_eta_split.log is a different measure) | -- |
| "shared with pre-stroke events" (rate) | -- | pre rate 0.025-0.078 Hz, 3-8x lower than acute | log rate_up_hz pre |

Conclusion changes:
- The core conclusion still holds: the events fail the interictal criteria, because UP/DOWN is ~1, there is no refractory gap (IEI at or above Poisson) and heights are well under 10 %.
- The numbers have shifted. The CV is now 0.83-0.95, closer to Poisson. Heights are 1.6-2.6, lower than 2.4-3.1, so the events are even further below the Steinmetz threshold. The rate is 0.16-0.40 Hz, higher and now overlapping Steinmetz's 0.1-0.5 Hz more fully.
- Affected sentence: "IEI CV 0.67-0.8; heights 2.4-3.1 %. Only rate (0.10-0.18 Hz) and bilateral symmetry match, both shared with pre-stroke events". **"Rate shared with pre-stroke events" does not hold in the new numbers.** Pre rates are 0.025-0.078 Hz, below the 0.1-0.5 Hz band, while acute rates are inside it. That sentence needs rewriting. (The old caption may have meant the qualitative property. Check it with Priya.)
- No stars, daggers or k/4 counts are in this caption.
- Note: PS94 chronic IEI columns are NaN, because n_iei is too small in 7/8 sessions. This does not affect the acute statements.

PNG check: 4 panels (UP/DOWN count, IEI 0.3-0.5 obs/Poisson, IEI 0.5-1 obs/Poisson, IEI CV), sessions as markers with a median bar. This matches the caption's criteria. Height and rate are not plotted; they are only in the log.

<!-- out_C.md -->
# Section 23 OLD (caption) vs NEW (sync-end run) -- Figs 23.12, 23.13, 23.14, 23.22

R = M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/hv-meegkit_..._daqlk/cohort1_quiet_spikes/v2_quiet_lp_c1guard_syncend_20261118
All four logs: code wf_qstyle1118 67d49b99, env QS_ENGAGED_END=lastsync, QS_LICK_SOURCE=daq+lp, FROZEN_LICKS_ALL=split, exit 0.
"derived" = simple arithmetic I did on printed log numbers (differences / means), not a recompute.

---------------------------------------------------------------------------------------------------
## Fig 23.12 -- breathing (log R/logs/quiet_breathing.log; tables R/breathing/*.csv)

### n -- BIG CHANGE (not in caption, but affects everything)
- extract: **68 done, 40 failed** of 108 (log l.136; td pass l.286 same). Failures are "RuntimeError: only N quiet segments" (N = 0-18; minimum used appears to be ~20, kept sessions have min 21 segments, l.327) plus PS93_0813 "no FaceRhythm task".
- Sessions kept per animal x epoch (pre, acute, subacute, chronic): PS92 [2,4,1,0], PS93 [1,4,3,2], PS94 [7,5,5,7], PS95 [8,1,7,11] (l.330). Totals: pre 18/32, acute 14/16, subacute 16/17, chronic 20/43.
- PS92 has NO chronic sessions; PS93 has 1 pre session; PS95 has 1 acute session. Most PS92/PS93 sessions are lost.
- Old n per epoch: MISSING (caption does not state it). Likely cause: shorter sync-end engaged period + c1guard licks leave fewer >= 4-s quiet runs.

| quantity | old (caption) | new (log) | source |
|---|---|---|---|
| proxy frequency acute | +0.35 Hz, q .22 (n.s.) | **+0.604 Hz, p .031, q .031** | quiet_breathing.log l.765 (proxy_freq face acute) |
| proxy frequency, per animal acute vs pre | -- | PS92 3.63->4.56, PS93 4.25->4.38, PS94 3.61->3.95, PS95 3.25->3.00 (3/4 up, derived) | l.333-336 |
| orofacial coherence with proxy "mostly < 0.1" | < 0.1 | still true: animal means 0.004-0.040 (l.413-416); fig panel (a) max one point ~0.135 at 0.75 Hz | l.410-416, png (a) |
| band mean 1.5-4 Hz coherence (coh_corr, orofacial) acute | -0.020, q .015 | **-0.011, p .070, q .117 (n.s.)**; subacute -0.0135, q .033 | lmm_breathing.csv / log l.~658-660 (coh_corr orofacial) |
| coherence at proxy's own frequency (coh_resp, orofacial) acute | -0.020, q .19 | -0.010, p .39, q .48 | l.696 |
| cortical peak frequency acute (peak_freq, orofacial) | -1.3 Hz, q 2e-4 | **-0.64 Hz, p .0035, q .0087** | l.756 |
| cortical peak "~3 Hz acutely" | ~3 Hz | acute animal means 2.81, 3.06, 3.35, 2.50 Hz | l.373-376 |
| cortical peak tracks proxy across sessions? | "does not track" | orofacial peak_freq: LMM slope +0.20 Hz/Hz (se .097), **p .037**; per animal rho -0.22 (p .63), -0.29 (.42), **+0.51 (p .011, PS94)**, -0.13 (.51); ratio_peak_freq LMM -0.28, p .19 | l.~616-625 (FREQUENCY TRACKING table) |
| 1-4 Hz power acute raw -> regressed | +0.52 -> +0.53 | +0.558 (q<1e-4) -> +0.571 (q<1e-4) | l.801, l.813 (td_logP_raw / td_logP_res orofacial acute); png suptitle "+0.56 vs +0.57" |
| event rate acute raw -> regressed (log10) | +0.63 -> +0.67 | +0.645 -> +0.692 (both q<1e-4) | l.729, l.741 (lograte_raw / lograte_res orofacial acute); png "+0.65 vs +0.69" |
| ipsi / contra alike (regressed rate acute) | alike | ipsi (orofacial_left) +0.731, contra (orofacial_right) +0.630 -- still alike | l.744, l.747 |
| chronic regressed 1-4 Hz power | +0.06, q .33 (n.s.) | +0.043, p .52, q .60 (n.s.) | l.815 |
| chronic regressed event rate | +0.18, p .003 | **+0.123, p .034, q .046** (weaker, still < .05) | l.743 |
| chronic regressed rate per animal (post - pre, log10) | -- | PS92 none, PS93 +0.237, PS94 -0.032, PS95 +0.235 (2/3 up) | l.~603-614 table (d) |
| frac of increase left after regression (acute) | -- | td power 0.997-1.037, rate 0.93-1.15 per animal | table (d) |
| bootstrap-CI stars for any of these | (caption gives none) | MISSING -- breathing script prints / plots only LMM stars ("stars = LMM epoch vs August pre, FDR q", png suptitle) | png suptitle |

### Conclusion changes
1. "Proxy frequency acute +0.35 Hz (q .22 -- the direction Priya saw, not significant)." -> NEW: +0.60 Hz, q .031 = **significant (gains a dagger / LMM star; png shows "*" over acute in panel (b) proxy peak frequency)**. 3/4 animals up (PS95 down, and PS95 has only 1 acute session).
2. "Orofacial coherence with the proxy is low ... slightly DOWN acutely (band mean 1.5-4 Hz -0.020, q .015 ...)" -> NEW acute -0.011, q .117 = **no longer significant acutely** (only subacute q .033 remains, png star at subacute). Direction unchanged (down).
3. "The cortical peak moves DOWN to ~3 Hz acutely (-1.3 Hz, q 2e-4) and does not track the proxy across sessions." -> shift halved (-0.64 Hz) but still q .009; the "does not track" part is weakened: pooled LMM tracking slope +0.20, p .037, driven by PS94 (rho .51, p .011); other 3 animals negative/n.s.
4. "chronically ... the rate stays up (+0.18, p .003)" -> +0.12, p .034 / q .046; only 2/3 animals with chronic data up (PS94 slightly down; PS92 has no chronic session).
5. Main conclusion "Regressing the proxy out ... leaves the increase" -- UNCHANGED (stronger: +0.56->+0.57, +0.65->+0.69).

### Method mismatches
- n: 40/108 sessions dropped (above). Caption is silent on n; with pre = 2 / 1 sessions for PS92 / PS93 the per-animal pre is fragile.
- Lick text: caption says cam1-only labels far from the mouth "which this pre-rebuild render counts as licks"; new run uses FROZEN_LICKS_ALL=split (c1guard dir) -- caption sentence is outdated. README.txt in breathing/ still says "no DAQ contact lick" but env QS_LICK_SOURCE=daq+lp (README text stale, run used DAQ+LP).
- Engaged end: new = min(quit, last sync edge) (QS_ENGAGED_END=lastsync); caption has no engaged-end statement.
- Output root: caption v2_quiet_lp/ -> new v2_quiet_lp_c1guard_syncend_20261118/.
- Stars on the png are LMM stars, not bootstrap stars (house rule: * = bootstrap CI, dagger = LMM).

### PNG
Panels match the caption content: (a) coherence vs Hz, (b) proxy velocity spectrum, reference orofacial PSD, (e) frequency tracking scatter; row 2 (b) proxy peak freq, (c) coherence at peak / 1.5-4 Hz / best-PC; row 3 (d) raw vs regressed 1-4 Hz power and event rate. Panel letters repeat ((b),(c) twice) and stars are LMM.

---------------------------------------------------------------------------------------------------
## Fig 23.13 -- laterality, amplitude, recovery (log R/logs/quiet_hemi_amp_recovery.log; R/hemi_amp_recovery/*.csv)

n: T11 108 sessions (t11_sessions.csv), T8 LMM n = 107 (t8_lmm_ipsi_minus_contra.csv), T14 76 post sessions (t14_sessions.csv).

| quantity | old (caption) | new (log) | source |
|---|---|---|---|
| T8 orofacial ipsi - contra rate change vs pre | +0.02 (n.s.) | acute +0.055 (q .36), subacute +0.012 (q .84), chronic +0.036 (q .62) -- all n.s. (caption's epoch for +0.02 unclear) | log l.55-57 |
| T8 limb contra > ipsi acutely | q .01 | **coef -0.141, p .038, q .077 (n.s. after FDR)** | log l.51; t8_lmm csv |
| T8 limb animals contra > ipsi acutely | PS92 / PS94 | PS92 (ipsi -0.09 vs contra 0.40), PS94 (0.10 vs 0.20) -- same 2 animals; PS93, PS95 ipsi > contra | log l.30-33 |
| T11 median event heights | 2.4-3.1 % dF/F | **1.33-2.58 %** (animal x epoch medians of med_h_pos; acute 1.61-2.58) | log l.59-65 |
| T11 events > 10 % share | < 1 % | pooled: pre 4/1540 (0.26 %), acute 0/5079, subacute 0/1335, chronic 4/1712 (png legend); per animal max 1.08 % (PS94 chronic), all others <= 0.62 % | log l.107-113; png |
| T11 > 10 % RARER acutely | rarer | acute 0-0.05 % vs pre 0.18-0.62 %, 4/4 lower -- holds | log l.110-113 |
| T11 acute orofacial events co-occur with limb LESS | less (more local) | limb_coinc_pos pre -> acute: PS92 .071->.151 UP, PS93 .024->.152 UP, PS94 .236->.092 down, PS95 .095->.066 down (**2/4**); limb_coinc_neg 4/4 UP (.203->.262, .153->.256, .055->.117, .030->.167) | log l.91-105 |
| T14 LP: within-animal r rate vs far-contra recovery | r -0.72, 4/4 | **-0.736, p_perm < .001 (0.000), 4/4** (-0.59, -0.84, -0.77, -0.74) | log l.117 |
| T14 beyond log-day trend | -0.19 (n.s.) | -0.177, p .132 (n.s.); signs PS92 +0.34, PS93 -0.41, PS94 -0.07, PS95 -0.57 (3/4 neg) | log l.118 |
| T14 DAQ-frames comparison | was -0.23, p .04 | MISSING (no DAQ-only sync-end run) | -- |
| T15 ordering across animals | none (n = 4) | Spearman acute rate vs half-recovery day rho -0.40; vs lesion % rho -0.40 (n.s.) -- holds | log l.120-127 |
| T15 per animal values | -- | acute rate x pre 5.12/7.78/6.60/6.11; half-recovery day 4.7/3.9/7.0/7.9; lesion 46.7/45.0/67.7/81.9 % | log l.121-125 |
| stars / bootstrap CI for T8 | -- | MISSING (only LMM q printed; png stars labeled "LMM, FDR"; none drawn) | png |

### Conclusion changes
1. "limb contra > ipsi acutely (q .01, PS92 / PS94)" -> **q .077, no longer significant** (p .038 raw). Same two animals, direction unchanged.
2. "T11: median heights 2.4-3.1 % dF/F" -> **1.3-2.6 %** (number change; claim "small vs 10 % interictal" only gets stronger). Possibly a different summarization (old range basis unknown) -- worth checking before editing.
3. "acute orofacial events co-occur with limb events less (more local)" -> **NOT supported**: UP events 2/4 animals less, DOWN events 4/4 MORE limb-coincident acutely. Sentence should be dropped or rewritten.
4. "events > 10 % are < 1 % and RARER acutely" -- holds (pooled 0/5079 acute).
5. T14 (r -0.74 4/4; detrended -0.18 n.s.) and T15 -- unchanged.

### Method mismatches
- Caption lick text ("cam1-only labels ... which this pre-rebuild render counts as licks") outdated: run uses FROZEN_LICKS_ALL=split (c1guard). Engaged end now lastsync.
- Caption names output roots v2_quiet_lp/ and v2_quiet/ -> now v2_quiet_lp_c1guard_syncend_20261118/.
- "(was -0.23, p .04 with DAQ frames)" refers to an old DAQ-only run with no new counterpart.

### PNG
Matches: T8 orofacial and limb ipsi/contra bars, T11 height histogram (10 % line) and limb-synchrony proxy, T14 scatter (r -0.74 / -0.18 in title), T15 per-animal scatter. No panel letters; no stars drawn on T8 panels (consistent with q > .05).

---------------------------------------------------------------------------------------------------
## Fig 23.14 -- recovery link (log R/logs/quiet_recovery_link.log; R/recovery_link/recovery_link.csv)

n: 76 post-stroke sessions (recovery_link_sessions.csv: PS92 5/2/12, PS93 4/3/12, PS94 6/5/8, PS95 1/7/11 acute/subacute/chronic). Permutation 5000 draws (README).

| quantity | old (caption) | new (log) | source |
|---|---|---|---|
| LP event rate raw | -0.72 (4/4) | -0.739, p < .001, 4/4 | log l.46 |
| LP event rate detrended | -0.20 (n.s.) | -0.182, p .125, 3/4 same sign (n.s.) | log l.47 |
| synchronization (orofacial) raw | -0.64 (4/4), "DAQ-only run" | -0.570, p < .001, 4/4 | log l.30 |
| synchronization detrended | n.s. | **-0.412, p_perm < .001 (0.000), 3/4 (PS94 +0.04)** | log l.31 |
| 1-4 Hz power raw (ipsi / contra) | -0.56 / -0.59 (4/4) | -0.725 / -0.647, both p < .001, 4/4 | log l.36, l.38 |
| 1-4 Hz power detrended (ipsi / contra) | n.s. | **-0.404, p .0006 / -0.368, p .0014; 3/4 (PS94 +)** | log l.37, l.39 |
| 8-15 Hz power raw | +0.47 (4/4) | **ipsi -0.005, p .97, 2/4; contra -0.099, p .42, 3/4 (negative)** | log l.40, l.42 |
| 8-15 Hz detrended | n.s. | ipsi +0.164, p .18; contra +0.138, p .25 (n.s.) | log l.41, l.43 |
| sync ipsi - contra | -- | raw -0.148 p .22; detrended -0.023 p .86 | log l.32-33 |
| 0.5-1 Hz power | -- | raw -0.506 p < .001 4/4; detrended -0.305 p .010 3/4 | log l.44-45 |

### Conclusion changes
1. "DAQ-only run: synchronisation -0.64, 1-4 Hz -0.56 / -0.59, 8-15 Hz +0.47 raw (4/4), all n.s. detrended." ->
   - **8-15 Hz flips**: +0.47 (4/4) -> about 0 / slightly negative (2/4 and 3/4 neg), n.s.
   - **"all n.s. detrended" is no longer true**: sync (-0.41, p < .001), 1-4 Hz ipsi (-0.40, p .0006), contra (-0.37, p .0014), 0.5-1 Hz (-0.30, p .010) all survive log-day removal (each 3/4 animals; PS94 opposite sign).
2. "Markers of the recovery time course, not session-level predictors." -> challenged for synchronization and 1-4 Hz power (they now carry within-animal recovery information beyond log(day), 3/4 animals). Still true for event rate (detrended n.s.).
3. Event-rate numbers -- unchanged.
Caveat: the old sync / power numbers came from a DAQ-only-frames run; the new numbers come from DAQ+LP frames on the sync-end run, so this is not like-for-like. A DAQ-only sync-end counterpart is MISSING.
A "RuntimeWarning: Mean of empty slice" (log l.26-27, in the band-power log10) means at least one session x band had no frames (NaN, dropped).

### Method mismatches
- Caption labels sync / power as "DAQ-only run"; new run has QS_LICK_SOURCE=daq+lp (the caption's DAQ / LP split no longer applies; all measures now from one LP-quiet run).
- Lick text (cam1-only labels counted as licks) outdated (FROZEN_LICKS_ALL=split); engaged end = lastsync; output root changed.
- "far-contra hit = DAQ spout contact (trial_attempts.csv)": consistent with README (recovery_timing.series).

### PNG
Six panels: event rate, sync orofacial, sync ipsi-contra, 1-4 Hz ipsi, 8-15 Hz ipsi, 0.5-1 Hz -- with r / p in titles matching the log. Caption's "1-4 Hz ipsi / contra" -- only ipsi plotted (contra in table only). No panel letters.

---------------------------------------------------------------------------------------------------
## Fig 23.22 -- effort (log R/logs/attempt_effort.log; <parent of R>/attempt_effort/contrast_vs_hit.csv, attempt_effort_trials.csv)

n: 107 sessions, 0 failed (PS93_0813 not in list); 57,048 trials. Caption has no numbers; checks of the qualitative claims (class minus hit within session, animal means, log "WITHIN-SESSION class minus hit" table = contrast_vs_hit.csv):

| claim | new values (pre / acute) | holds? |
|---|---|---|
| jaw-only run more than hits (run_win) | pre +0.52, -0.02, +0.23, +0.10; acute +0.34, +0.20, +0.13, 0.00 (PS92, PS93, PS94, PS95) | yes, 3/4 each epoch (PS93 pre / PS95 acute not) |
| ... equally pre-stroke | animal mean pre +0.21 vs acute +0.16 (derived) | yes |
| running starts before the cue (run_delta jaw - hit ~ 0 / negative) | pre +0.07, -0.08, -0.08, -0.03; acute +0.04, +0.03, 0.00, 0.00 | yes (window - baseline adds ~nothing) |
| lick-no-contact barely run more (run_win incomplete - hit) | pre +0.22, +0.04, +0.02, +0.07; acute +0.05, +0.01, +0.05, 0.00 | mostly (PS92 pre +0.22; PS94 subacute +0.29) |
| limb cortex lower on every failed class than hits | limb_resp class - hit: incomplete pre -3.45, -2.78, **+0.10 (PS94)**, -2.68; all other cells pre / acute negative | 23/24 animal x class x (pre, acute) cells lower; PS94 incomplete pre is the exception |
| ... by the same amount pre and acute | animal means (derived) incomplete -2.20 / -2.22; jaw_only -2.99 / -3.19; **no_try -3.17 / -4.11** (PS93 -1.99 -> -4.01, PS92 -3.51 -> -4.50) | yes for incomplete and jaw-only; no_try is ~0.9 % dF/F more negative acutely (3/4 animals) |

### Conclusion changes
- None of the main claims flips. Two qualifications: (i) "limb cortex is lower on every failed class" has one exception (PS94 lick-no-contact pre, +0.10); (ii) "by the same amount pre and acute" holds for lick-no-contact and jaw-only but not for no-try (more negative acutely). Old per-cell values MISSING (caption gives none), so it is unknown whether these were already true in the old run.
- Note PS95 acute = all zeros in running (only 0 values printed for acute; few acute trials) -- PS95 contributes nothing to the acute running claims.

### Method mismatches
- Window cue -> +3.5 s, DAQ bouts 3 mm/s, baseline cue-2..cue: matches README. Limb / cortex window is cue..+1.5 s minus cue-1..0 (README, png titles); caption says only "after the cue" -- fine but less specific.
- Caption "Scripts / outputs: ... cohort1_quiet_spikes/v2_quiet/" -> new output is cohort1_quiet_spikes/attempt_effort/ under the rebuild root (one level above the run folder, so it is not namespaced by the syncend run).
- Log env carries QS_ENGAGED_END=lastsync / daq+lp, but trial classes come from trial_attempts.csv; whether the attempt table itself was rebuilt with the new lick split is not stated in the log (MISSING).
- Caption class "lick-no-contact" = "incomplete" in log / png legend.

### PNG
Eight panels (all spouts top, far contra bottom): running fraction, running minus pre-cue baseline, limb cortex dF/F, whole cortex dF/F, by class (hit / incomplete / jaw_only / no_try) x epoch -- matches the caption. No panel letters, no stars (descriptive), legend says "incomplete" not "lick-no-contact".

---------------------------------------------------------------------------------------------------
## MISSING (not in new outputs)
- 23.12: old per-epoch session n (to compare to the 68/108 kept); bootstrap-CI stars (script makes LMM stars only).
- 23.13: "(was -0.23, p .04 with DAQ frames)" -- no DAQ-only sync-end run; T8 bootstrap stars (LMM only); basis of the old "2.4-3.1 %" range.
- 23.14: DAQ-only-frames counterparts for sync / 1-4 Hz / 8-15 Hz (new run is DAQ+LP).
- 23.22: old class-minus-hit values (caption has no numbers); whether trial_attempts.csv used the new lick split.

<!-- out_D.md -->
# Fig 23.16 (f-q_lick_lfo): OLD caption vs NEW sync-end run

R = .../cohort1_quiet_spikes/v2_quiet_lp_c1guard_syncend_20261118
Logs: R/logs/quiet_lick_lfo.log (LL), R/logs/quiet_state_response.log (SR)

## Images
- The page shows one image: `q_lick_lfo.png` (title: "[licks: LP video onsets (cam4 + cam1) + unmatched DAQ contacts] Lick-locked low-frequency activity does not fall (no Ramanathan-type task LFO loss)").
- New pngs that match it:
  - **R/lick_lfo/lick_lfo_epochs.png**: the main per-epoch panels (evoked, ITPC, induced, quiet power; cued and ITI rows). This is most likely the image to swap in for q_lick_lfo.png.
  - R/lick_lfo/lick_lfo_recovery.png: the within-animal ITPC vs far-contra recovery link.
  - R/lick_lfo/lick_lfo_traces.png: per-animal traces and ITPC time courses (supporting).
  - The state-response sentence (T21) goes with R/state_response/state_response_prediction.png. That png is not part of this figure on the page.

## 1. Numbers (old caption | new | source)

All values are LMM coefficients, epoch vs August pre, on orofacial cortex. Amplitude and power are log10 ratios; ITPC is raw.

| quantity | old (caption) | new | source |
|---|---|---|---|
| cued evoked, acute | +0.06, n.s. | +0.056, p .147, q .174 (n.s.) | LL l.229 |
| cued evoked, subacute | +0.16 | **+0.130**, q .0009 | LL l.230 |
| cued evoked, chronic | +0.19 | **+0.165**, q <.0001 | LL l.231 |
| ITI evoked, "every epoch" | +0.17 each epoch | **+0.122 / +0.103 / +0.129** (acute / subacute / chronic), all q <=.0002 | LL l.232-234 |
| cued induced, acute | +0.22 | +0.213, q <.0001 | LL l.235 |
| cued ITPC, acute | -0.01 (unchanged) | -0.006, p .64, q .85 (unchanged) | LL l.241 |
| cued ITPC, chronic | +0.10, q .004 | **+0.070**, q <.0001 (PPC chronic is +0.110, q <.0001, l.249) | LL l.243 |
| recovery link, cued ITPC, log-day removed | +0.35, p .002, 3/4 | **+0.38, p_perm .001, 3/4** (PS92 +0.72, PS93 +0.78, PS94 +0.28, PS95 -0.24; n = 19 post sessions per animal) | LL l.291; recovery_link.csv row 3 |
| same, DAQ licks only | +0.28 | MISSING (no DAQ-only rerun in R) | - |
| lick-count matched, hits only | +0.27, p .025 | MISSING | - |
| lick-count matched, without far-contra trials | +0.43, p .001 | MISSING | - |
| lick-count matched, hits at other spouts | +0.39 | MISSING | - |
| matched variants with DAQ licks | +0.27 / +0.37 / +0.38 | MISSING | - |
| PS95 opposite throughout | opposite | PS95 r = -0.24 on the main link (still opposite); matched variants MISSING | LL l.291 |
| T21: share of the acute orofacial lick response explained by pre-lick power, causal power | 15-21 % | **14-20 %** (cued 13.7 %, ITI 20.2 %) | SR l.520+ "summed over animals": acute orofacial causal |
| T21: same, zero-phase power | up to 42-47 % | **33-38 %** (linear 33.5 / 33.6 %; log10 38.3 / 37.1 %) | SR summed table, acute orofacial linear / log10 |
| T21: same, DAQ licks | 14-22 % | MISSING | - |
| ITPC near ceiling | ~0.8 | cued median ITPC 0.77 / 0.77 / 0.83 / 0.85 (pre / acute / subacute / chronic); ITI 0.65-0.69 | LL l.213-220 |
| n | PROVISIONAL, n = 4 | 4 animals, 108 sessions (pre 32, acute 16, subacute 17, chronic 43); ITI LMM n = 107 | LL l.137, l.229-261 |

Context the caption does not quote:
- ITI induced power: acute +0.209 (q .023), subacute +0.140 (n.s.), chronic +0.166 (q .059).
- Cued ITPC subacute: +0.037, q .007.
- Time model: cued ITPC post:lday +0.024, q <.0001.
- Quiet 1-4 Hz psd: acute +0.70, subacute +0.39, chronic +0.06 (n.s.).

## 2. MISSING
- DAQ-licks recovery link (+0.28).
- All lick-count-matched recovery variants (+0.27 p .025, +0.43 p .001, +0.39) and their DAQ versions (+0.27 / +0.37 / +0.38). No output in R/lick_lfo, R/recovery_link or any log in R/logs contains this analysis.
- PS95 "opposite throughout" for the matched variants.
- T21 DAQ-licks range (14-22 %).

## 3. Conclusion changes
- **No direction flips.**
- **Significance (LMM) is unchanged for every quoted term:**
  - cued evoked: acute n.s.; subacute and chronic significant.
  - ITI evoked: significant in every epoch.
  - cued induced acute: significant.
  - ITPC: acute n.s.; chronic significant.
- **Stars in the new epochs png** (these are LMM FDR stars per the png header, not bootstrap):
  - cued evoked: subacute ***, chronic ***; acute none.
  - cued ITPC: subacute **, chronic ***.
  - cued induced: acute ***, subacute **, chronic ***.
  - ITI evoked: ***, ***, ***.
  - ITI induced: acute * only.
  - quiet power: acute ***, subacute ***.
  - All of these agree with the caption's claims.
- **Recovery link:** 3/4 is unchanged; PS95 is still the odd animal; p is .001 (was .002).
- **Magnitude changes worth rewording** (no change in the claim):
  - "ITI +0.17 every epoch" is now about +0.10 to +0.13. Still up in every epoch, but +0.17 is wrong.
  - Cued evoked subacute / chronic: +0.13 / +0.16 (was +0.16 / +0.19).
  - "ITPC ... up chronically (+0.10, q .004)": ITPC chronic is now +0.07 (q <.0001). The new PPC is +0.11. Check which measure the old +0.10 came from.
  - T21: "up to 42-47 % with zero-phase power" is now 33-38 %. Causal power: 14-20 %. The "only a minority" claim holds and is slightly stronger.
- **Title claim** ("Lick-locked low-frequency activity does not fall (no Ramanathan-type task LFO loss)") still holds: evoked, induced and ITPC are up or unchanged, never down.

## 4. Method mismatches between caption and run
- **Lick source.** The run used QS_LICK_SOURCE=daq+lp (LP video + unmatched DAQ), matching the caption.
  - Exception: PS93_0813 has no cam4 LP table, so that session gets no video licks (LL l.26).
  - The lick_lfo README still describes "first DAQ lick onset" events. That is stale text; the env and png axis say LP + unmatched DAQ.
- **cam1 far-label licks.** The caption says the run counts "cam1-only labels far from the mouth (med_out < -120 px), which this pre-rebuild render counts as licks".
  - The new run is the `c1guard` variant with FROZEN_LICKS_ALL=split, which presumably drops those labels.
  - That sentence, and "pre-rebuild render", need rewriting.
- **Engaged window.** The engaged period now ends at min(quit, last sync edge) (QS_ENGAGED_END=lastsync). The caption doesn't state an engaged-end rule.
- **Gate.** WIDEFIELD_TRIAL_GATE=camera, TRIAL_WINDOW=cue_stop. The caption's "cued first lick in the response window" is consistent.
- **Output path.** The caption names `cohort1_quiet_spikes/v2_quiet_lp/`. The new path is `v2_quiet_lp_c1guard_syncend_20261118/lick_lfo` (+ `state_response`).
- **Stars.** The epochs png marks LMM-FDR stars ("stars = mixed model epoch vs August pre, FDR q"), not bootstrap-CI stars. That goes against the "bootstrap primary" convention. The caption itself states no stars.
- **Session counts.** Per-animal session counts (lick_lfo_traces.png) are pre 8 / acute 5,4,6,1 / subacute 2,3,5,7 / chronic 12,12,8,11. PS95 has only 1 acute session.

## 5. PNG check
- **lick_lfo_epochs.png:** matches the caption. It shows 2 rows (cued, ITI) x evoked / ITPC / induced / quiet power, with ratio to the animal's pre, session markers (shape = animal, color = epoch) and pooled mean + CI. Minor issue: the cued ITPC y-axis shows a duplicated "1.1" tick label.
- **lick_lfo_recovery.png:** matches the caption. Panels: ITPC across post days, far-contra recovery, and detrended scatter (mean +0.38, p .001; PS95 -0.24).
- **lick_lfo_traces.png:** matches as a supporting figure. It shows per-animal band-passed responses and ITPC time courses with the 0-0.5 s window shaded. Cued ITPC sits around 0.8-0.9.
- **state_response_prediction.png:** matches the T21 sentence. Orofacial bilateral acute summed ratios: causal 0.14 / 0.20, linear 0.33 / 0.34, log10 0.38 / 0.37.
