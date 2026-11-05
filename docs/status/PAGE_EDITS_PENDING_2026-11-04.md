# Pending results-page edits from the 11-03 / 11-04 work

Every method change, check and finding from 11-03 / 11-04 that has to reach the results pages. Priya decided (11-04) that
no interim swaps are made: everything below is applied in ONE pass after sabatinilab-34's cohort-1 rebuild renders.
Owners: imaging / kinematics / behaviour / FaceRhythm pages = sabatinilab-e0; encoding page = sabatinilab-34. The
PI-deck window (sabatinilab-a8) keeps this list.

Placement:

- MAIN means a figure or claim in the section body.
- COLLAPSIBLE means a collapsible supplementary block: a methods check / robustness figure, closed by default.

Captions follow Priya's rule: rewrite a wrong caption cleanly, and never add CORRECTION notes on the page.

## A. Method changes: every affected caption / methods line is rewritten in the rebuild pass

| # | change (decision) | source | pages / sections | placement |
|---|---|---|---|---|
| A1 | Working = contact + incomplete camera licks, cut at the session's last camera lick; jaw-only and no-response are their own classes (11-03 late) | DECISIONS 11-03 late; JAW_ONLY_CHECKS_2026-11-03.md | imaging §0 (definition box, 2 places), §1-4, 6, 7 (e0 list); encoding (34) | methods line MAIN; the jaw-only check COLLAPSIBLE |
| A2 | August-only pre baseline for the whole imaging render | DECISIONS 11-03 | imaging (all epoch figures) | methods line |
| A3 | Spout bal_em, now superseded by the A4 centre | SPOUT_BALEM_PAGE_EDITS_2026-11-03.md | imaging §3 | rewrite under A4 |
| A4 | Frozen-decoder / CD / pattern centre = Wt: the DAQ-label balanced centre on all working rows. The engagement axis stays z (11-04) | DECISIONS 11-04 "centre = Wt" | imaging §2-4 (decoders, CDs, patterns); encoding where frozen decoders appear | methods line MAIN; the centre checks COLLAPSIBLE (B3) |
| A5 | Section D scored on working rows; section G grouped by camera class; G9 / coding-direction classes on camera classes with bias guards (11-04) | bcdg_1104 (4a37cbc8); DECISIONS 11-04 | imaging §2-4, section G figures | MAIN figures change; methods line |
| A6 | Camera classes everywhere, including the engagement measures (quit point = last camera lick; stop rule A2 "both sides"; DAQ = robustness) | 34 CAMERA_CLASS_INVENTORY_20261104.md, STOP_RULE_GUARD_CENSUS_R2_20261104.md | behaviour page (hit rate, Fig 0.2, engagement); imaging §1; encoding | MAIN: hit-rate figures change. COLLAPSIBLE: the stop-rule comparison A0 / A2 / A3 > 10 |
| A7 | Cue + 70 ms floor kept; 0-70 ms licks excluded from both tables (camera licks by peak); contacts after max_rt don't count | CUE70_CHECK_2026-11-04.md; 34 CONTACT_DAQ_MISMATCH_20261104.md | kinematics (lick tables), imaging lick-aligned, encoding | methods line; the latency histogram COLLAPSIBLE |
| A8 | Lick class for lick-aligned arms = contact + incomplete with LP timing; DAQ fallback > 50 ms at close spouts | 34 census; running list | imaging lick-aligned, encoding lick chain | methods line |
| A9 | ENL windows follow the lick source (LP 1.0 s sliding lick-free / DAQ 2.0 s), encoder ENL scoring included | 34 rebuild/enl-window | imaging precue figures, encoding, quiet events | methods line |
| A10 | Every Allen-ROI analysis on the ALM / tjM1 split basis; orofacial group = ALM, tjM1, SSp-m, SSp-n | 34 SPLIT_BASIS_INVENTORY_20261104.md | imaging, encoding, quiet events | MAIN: panel layouts change; methods line |
| A11 | Section-1 table without jaw-only trials (camlickpl). The 1104 table was built at ENL 2.0 s; Priya decided LP 1.0 s for the s1 refit (2.0 s as a sensitivity row), so e0 re-runs at 1.0 s in the rebuild pass (~25 min each for Allen / LocaNMF; --suffix ready) | e0, s1_refit_*_1104 | imaging §1 table + deck slide 9 | MAIN |
| A12 | State discriminators: camera-class (working vs none) as the primary G4 readout, z frame on contact rows | G4_CAMCLASS_MIXSHIFT_2026-11-04.md | imaging section G (G4) | MAIN readout; the mix-shift test COLLAPSIBLE |
| A13 | Encoder spontaneous licking: LICKS_ALL = split. Three lick variables: licks = cued (instructed); licks_pre and licks_iti (uninstructed). "drop:licks" now means cued only; the instructed / uninstructed group contents change; the Musall maps and unique-R2 figures change. Caveat: the tongue / jaw traces stay instructed over the whole trial | 34 rebuild log 8.43, encoder-classes | encoding | MAIN (methods + variance-partition figures) |
| A14 | Encoder per-class outputs: every cue trial is scored (non-working by the full-data fit); per-class PETH / cvR2 / trial-to-trial R2. Class comparisons use trial-to-trial R2, not cvR2 | 34 | encoding | new per-class figure, MAIN or COLLAPSIBLE (Priya to choose) |
| A13b | The FROZEN encoder also gets pre-cue and ITI lick terms (Priya 11-04; new frozen ids; the lick-kernel-free version is a robustness row). Its all-frames target columns move | 34; B12_PRECUE_POSITION_CLAIMS_20261104.md (B) | encoding Figs 1.1, 2.1 (+ deck slide), 2.10 / 2.12, 20.3-20.7 | MAIN figures + methods line |
| A15 | 3-D spout targets = the commanded positions (the old measured targets sat 1.02 mm behind in ap); affects the encoder's polar targets | e0 bddf02e7, pose3d_spout_commanded | encoding (+ kinematics 3-D) | methods line |
| A16 | Stopped trials with pre-cue / ITI licks are excised from both stopped and working; the quit tail is labelled "quit", separate from mid-session stops | 34 STOPPED_LICKS_CENSUS_20261104.md | behaviour / imaging per-class tables; quiet events | methods line |
| A8+ | At FAR spouts, LP > 50 ms late is flagged grooming_suspect and the LP time is kept | 34 LP_MISSED_CONTACT_20261104.md | lick-aligned methods | methods line |

## B. Robustness checks: COLLAPSIBLE supplementary blocks (one per check, closed by default)

| # | check | source doc | suggested home | key figure / table |
|---|---|---|---|---|
| B1 | Jaw-only trials read like no-response trials, not attempts | JAW_ONLY_CHECKS_2026-11-03.md | imaging §2 (decoders) | per-class accuracy, cue / precue |
| B2 | G4 by camera class + mix-shift (z frame stable; working-row frames drift) | G4_CAMCLASS_MIXSHIFT_2026-11-04.md | imaging section G | bias by frame |
| B3 | Frozen-decoder centre: mix-shift and direct comparison (W / Wt / Ct / Cto / N / Z) | CENTRE_MIXSHIFT_2026-11-04.md | imaging §2-3 | headline-by-frame table |
| B4 | Rest-centred frame rejected (the rest state moves post-stroke) | REST_CENTRE_2026-11-04.md | same block as B3 | rest bias per animal |
| B5 | Reference-spout anchor and intercept-only refit | REFSPOUT_CENTRE_2026-11-04.md | same block as B3 | the Iref caveat per B6 |
| B6 | Injected-effect validation: Wt is conservative (~2/3 recovery of a spout-specific loss; immune to offsets / rest); WtF / WtR | INJECTED_EFFECT_2026-11-04.md (§7) | same block as B3 | frame x effect bias table |
| B7 | Post-stroke SD rise = global gain; F0 rose, so dF/F isn't inflated | SD_RISE_2026-11-04.md | imaging §2 + the lick-response section | variance split, F0 table |
| B8 | Lick response raw vs session-z (acute cued increase is the global gain; subacute / chronic tjM1 / SSp-m and ITI survive) | SD_RISE_2026-11-04.md §6; 34 _sessz renders | encoding / imaging lick-response section (34) | raw vs z panel; MAIN text reworded only if Priya decides |
| B9 | Cue + 70 ms latency check | CUE70_CHECK_2026-11-04.md | kinematics / lick methods | first-contact histogram |
| B10 | Stop-rule guard comparison | 34 STOP_RULE_GUARD_CENSUS_R2_20261104.md | behaviour page | arms table |
| B11 | Contact vs incomplete accuracy per spout (new panel, sections D and G) | 34 bcdg-port | imaging §2 / section G | MAIN if Priya wants it on the slide, else COLLAPSIBLE |
| B12 | Split vs no-licksall encoder (tag main_nolicksall). FINDING: pre-cue position credit (drop:pos_enl, LP ENL window) FALLS once spontaneous licks are modelled (PS93_0814 .0072 -> .0029, PS93_0818 .0044 -> .0037; per-animal numbers after O2). A conclusion-change watch for every encoder pre-cue position claim | 34 | encoding | COLLAPSIBLE (+ caption changes if it holds) |
| B13 | ENL scoring window: lick-free sliding vs enl_fixed2s | 34 | encoding | COLLAPSIBLE |
| B14 | LP missed contacts: 56 real close-spout misses (spout occlusion, droplet filter, gate M), 8 grooming | 34 LP_MISSED_CONTACT_20261104.md | kinematics / lick methods | COLLAPSIBLE |
| B15 | Contact vs DAQ mismatch: 23 trials, all contacts after max_rt; unjudged close-mid 3.1 % | 34 CONTACT_DAQ_MISMATCH_20261104.md | behaviour / kinematics methods | COLLAPSIBLE |
| B16 | Stopped-time licks census | 34 STOPPED_LICKS_CENSUS_20261104.md | behaviour page (same block as B10) | COLLAPSIBLE |

Encoding-page caption fixes in the pass (34): the window "enl" readers (combined_encoding_figures, encoding_key_figures,
encoding_mixed, fr_encoding_summary) and the lick variable lists (curbd_pilot).

## C. Methods text (one methods paragraph, imaging page)

- The cross-session normalisation paragraph:
  - the frame choice (Wt) and its bias direction (conservative for a spout-specific loss; blind to a proportional
    global gain);
  - the bracket (N, WtF);
  - the rejected frames, with reasons.
- Draft text and likely reviewer questions: docs/LITERATURE_CROSS_SESSION_NORMALISATION_2026-11-04.md (§5).

## NOT in this pass (e0)

These land with the next 3-D re-run, after the round-2 side model, as their own kinematics-page pass:

- the spout-position ENL no-lick rule for the v36 2-D chain (docs/status/SPOUT_POSITION_AUDIT_2026-11-04.md);
- the four-camera 3-D (cam2 / cam3).

## Status

All items are HELD until the rebuild renders (Priya 11-04).

- e0 holds imaging §1-4, 6, 7 + SPOUT_BALEM §7 + the s1 table.
- 34 holds the encoding page and renders the B8 / B11 figures.
- After the render: 34 sends the per-slide / per-figure old -> new list. e0 and 34 apply A + B in one pass; a8 swaps
  the deck.
