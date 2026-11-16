# Encoder / ITI drift cutoff check (2026-11-03)

Question: the encoder's own slow-drift removal (`combined_encoding.DRIFT_MIN_PERIOD_S` = 600 s, cosine basis projected
out of Y and every design column on the fitted frames; reused by the ITI scripts) -- is 10 min too short, i.e. does it
strip real position information that sits at the scheduler's six-position CYCLE period (p95 22 / 17 / 10.6 / 8.4 min,
PS92 / 93 / 94 / 95; DRIFT_CUTOFF_CHECK_20261103.md)? Priya 11-03: keep the encoder / ITI drift steps; check the cutoff.

Data: `hemo_meegkit_hpfit_lag_refitT_swapmask` (WIDEFIELD_HEMO_VARIANT), no new drift step in the shared signal.
Outputs `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/encoder_drift_cutoff_20261103/`. Own session cache
`C:/Users/SabatiniLab/.widefield_session_cache_encdrift`, own input cache `C:/Users/SabatiniLab/wfield_cache_encdrift`,
own frozen-model dir; published encoder outputs and stored frozen models untouched.

## Progress log

- 2026-11-03: encoder worktree `wf_swapmask_enc`, new branch `analysis/encoder-drift-cutoff` off e80dbef7 (= 9f458196,
  the published `frozen_encoder_gate-attempted_v7g3_t3d` commit, + hemo / out-root overrides). Added env override
  `COMBINED_ENCODING_DRIFT_MIN_PERIOD_S` (unset = 600 s; test `tests/test_encoder_drift_env.py`), commit 25e294d2; the
  input-cache key ignores it (130b2c8b; the inputs do not depend on the drift basis). Note: the frozen encoder fits on
  the TRIAL frames (strobe - 0.5 s .. stop + 1 s of every trial), not literally every frame; the basis spans those.
  Launched frozen encoder at 600 s (reproduction of the 11-02 swapmask run) then 1500 s, 6 workers
  (`chain_enc.sh`, logs `C:/Users/SabatiniLab/encdrift_logs/`).
- 2026-11-03: item 1 script `scripts/encoder_drift_retention.py` (enc worktree, b5b0840e). Band component = DCT band-pass
  of Y on the full frame grid (the drift check's definition), restricted to the procedure's rows; retention =
  ||(I - P_Q) y_b||^2 / ||y_b||^2. (A first version that built the band basis on the GAPPED encoder rows mixed slow power
  into the band and was dropped.) One-session probe PS92_0820 (acute; encoder rows = 59% of frames): ITI procedure
  cycle retention .72 / .90 / .99 / 1.00 / 1.00 at 600 / 900 / 1200 / 1500 / 2400 s (block band 1.00 throughout), but the
  ENCODER procedure leaks: cycle .71 / .78 / .82 / .88 / .95, block .96 / .97 / .98 / .98 / .995 -- on gapped trial rows
  (engaged trials only, span shorter than the session) the slow cosines are not orthogonal to faster bands, so even a
  40-min basis removes ~5% of the cycle band. Full 107-session run launched (3 workers).
- 2026-11-03: ITI worktree `wf_iti_drift`, branch `analysis/iti-drift-cutoff` off attempt-coding b950581b. Same env
  override + `hemo_dir()` (b6d53b44). Driver `scripts/iti_drift_check.py` runs `iti_licks.extract` UNCHANGED once per
  cutoff per session (one imaging load) into `<OUT>/iti/drift<c>/sessions_roi8fr/`; reproduction check: adopted hemo at
  600 s reproduces the published `sessions_roi8fr/PS92_0820.npz` EXACTLY (every array, max |diff| 0). Readouts =
  section-14 key-figure definitions (ITI response / build-up, cued / cue response, per spout, contra / ipsi log ratio,
  balance proxy) + a pre-cue LEVEL per spout (no within-trial baseline). Launched on 107 sessions x 600 / 1200 / 1500 s,
  2 workers.
- 2026-11-03: free RAM fell to ~4-6 GB (encoder stats pass ~4-5 GB per worker + other sessions' jobs). An attempted
  Stop-Process on the ITI PIDs was blocked (the worker PIDs had been picked by parent PID, which breaks the rule);
  the ITI run was then stopped with TaskStop on its own background task. 6 / 107 sessions were done at all cutoffs,
  and their extracts are kept. Coordinator: resume with 1 worker after the encoder 600 s run, or once free RAM stays
  > 8 GB. 12:46: restarted with 2 workers at the coordinator's request (free 23 GB; task bdecxf9fn, log
  `iti_drift2.log`; it skips the extracted sessions). The earlier gated restart (task bn4gnwuuw) was stopped with TaskStop before it launched.
- 2026-11-03 ~13:20: ITEM 1 DONE (107 / 107, 0 failed; `retention_sessions.csv`, `retention_animal_epoch.csv`). Cortex
  CYCLE-band retention, MIN over sessions per animal x epoch, at 600 / 900 / 1200 / 1500 / 2400 s:
  - ITI procedure (all frames): 1500 s keeps 1.000 in every cell except PS92 pre (.753, ONE session, PS92_0812 = the
    crash + concat session, whose time axis has a gap; .949 at 2400 s). At 600 s: PS92 .59 / .72 / .18 / .70
    (pre / acute / subacute / chronic), PS93 .87-.92, PS94 .87-.99, PS95 1.00. 1200 s: >= .985 everywhere except PS92
    pre (.750, 0812) and PS92 subacute (.626) / chronic (.934).
  - ENCODER procedure (engaged-trial rows; median row share 92 / 89 / 76 / 72% of frames, PS92 / 93 / 94 / 95): on gapped
    rows the slow cosines also take part of the faster bands, so NO cutoff reaches >= .98 in every cell. At 1500 s the
    medians are >= .96 in every cell (lowest PS92 acute .961, PS94 acute .970, PS93 acute .974, PS95 subacute .978; MIN cells: PS92 pre .764 (0812), PS95
    subacute .782 (PS95_0820), PS92 acute .881, PS94 acute .920, PS93 acute / subacute .937 / .938); at 2400 s the MIN
    cells are still .82-.99. At 600 s: median PS92 subacute .225 (both sessions), PS92 acute .754, pre .850. The BLOCK band
    under the encoder procedure is also below 1 at 600 s (MIN .77 PS95 subacute, .81 PS92 pre, .84 PS93 acute) and only
    partly recovers with longer cutoffs (1500 s: .87 / .93 / .94).
- 2026-11-03 13:26: frozen encoder 600 s DONE (107 sessions). REPRODUCTION: identical to the 11-02 swapmask run in every
  one of 3424 cells (joint / unique target / unique spout x window x area x session), max |diff| 0 (`encoding/compare/`).
  1500 s run started 13:26 (HEAD 424b83a1; encoder code identical to 130b2c8b, only scripts added).
- 2026-11-03 14:20: frozen encoder 1500 s DONE (107). Headline cells 600 -> 1500 (cue window, frame R2, change from pre,
  mean of animals [nested 95% CI], n up):
  - target unique cortex, chronic: -.0132 [-.022, -.005] 0/4 -> -.0122 [-.021, -.004] 0/4 (per animal PS92-95
    -.023 / -.007 / -.008 / -.016 -> -.021 / -.004 / -.009 / -.016); acute -.014 -> -.013, 0/4 both.
  - spout unique SSp-m contra - ipsi, chronic: +.0177 [.005, .031] 4/4 -> +.0214 [.008, .035] 4/4 (per animal
    .013 / .019 / .020 / .018 -> .024 / .016 / .028 / .018); lick window chronic +.019 [-.000, .037] 4/4 -> +.024
    [.006, .040] 4/4 (CI now excludes 0).
  - joint R2 cortex: pre R2 per animal drops .004-.016 at 1500; changes from pre move <= .017 EXCEPT PS92 subacute
    +.040 -> -.141 (cue) / +.071 -> -.102 (lick), which turns the pooled subacute cue cell -.012 -> -.051.
  - SIDE EFFECT (item 4): the drift failure comes back in exactly the session that set 600 s. PS92_0824 frozen cortex
    R2 (all frames) .438 -> .189 and cue .586 -> .247; PS92_0826 (the other PS92 subacute) .461 -> .353. Every other
    session moves -.019 .. +.019 (median -.002). No session's all-window frozen R2 goes negative (min .189); the
    cue / lick negative cells are the same three sessions at both cutoffs (PS94_0817 / 0818 lick, PS95_0914).
  Launched: per-session encoder (block CV) on PS92_0824 / 0826, PS94_0917, PS92_0812, PS95_0820 at 600 / 1200 / 1500 /
  no-drift (1e7 s) (`selfit.log`, 2 workers), and frozen encoder 1200 then 900 s (4 workers).
- 2026-11-03 ~14:30: RAM fell to 1 GB within minutes of launching the per-session fits (2 workers) together with the
  1200 / 900 frozen-encoder chain (4 workers). I stopped both of my tasks with TaskStop (bhf1gq4bi, bp4ntn4gc); free RAM
  went back to 46 GB, so the per-session encoder fit takes far more than the docstring's 3-4 GB. The ITI run (2 workers)
  was left running. Relaunched the per-session fit alone: PS92_0824 x 600 / 1500 / no-drift, 1 worker (bb2ld2cdq),
  with a 10 GB RAM watch. The 1200 / 900 frozen runs wait until it finishes.
- 2026-11-03: per-session encoder (block CV, full model) PS92_0824: cortex held-out R2 (all frames) .418 / .147 / -.072 at
  600 / 1500 / no drift basis (cue window .479 / .129 / .051; SSp-m contra .595 / .365 / .072). So 1500 s is most of the
  way back to the original -0.22-type failure (the no-drift value here is -.07 on the swap-mask data). Running 0824 at
  900 / 1200 and PS92_0826 at 600-1500 (1 worker).
- 2026-11-03: PS92_0824 per-session encoder, cortex held-out R2 (all / cue / lick) at 600 / 900 / 1200 / 1500 / none:
  all .418 / .332 / .267 / .147 / -.072; cue .479 / .401 / .324 / .129 / .051; lick .426 / .342 / .293 / .098 / -.036
  (SSp-m contra all .595 / .550 / .500 / .365 / .072). The fit degrades steadily with every longer cutoff; there is no
  plateau between 600 and 1500.
- 2026-11-03 ~15:40: RAM fell to 3-5 GB with the frozen 1200 s chain (3 workers) + per-session fit (PS92_0826) + ITI
  (2 workers) + other sessions' jobs. TaskStop on the chain task (b7wcul09z) reported success, but its bash (17020)
  and frozen_encoder (1624) processes kept running and the log kept growing. I did NOT kill them: their PIDs come only
  from a process listing, which the rules forbid. TaskStop on the per-session fit (bocezdilf) did end its process, and
  free RAM went back to 22 GB. So the 1200 s run continues, and the chain will then run 900 s. PS92_0826 per-session
  fits are still to do (0824 900 / 1200 were done before the stop).
- 2026-11-03: PS92_0826 per-session encoder, cortex held-out R2 (all / cue / lick) at 600 / 1200 / 1500 / none:
  all .445 / .375 / .325 / .253; cue .422 / .372 / .368 / .244; lick .408 / .385 / .357 / .249 (SSp-m contra all
  .630 / .601 / .592 / .524). The same direction as 0824, smaller.
- 2026-11-03 16:50: frozen encoder 1200 s DONE; the 900 s run started. At 1200 s the headline cells sit between 600 and
  1500: target unique cortex chronic (cue) -.0129 [-.022, -.004] 0/4; spout contra - ipsi chronic (cue) +.0216
  [.009, .035] 4/4, (lick) +.0242 [.006, .040] 4/4. PS92_0824 frozen R2 (all) .438 / .252 / .189 at 600 / 1200 / 1500;
  PS92_0826 .461 / .395 / .353. Every other session moves by <= .02. PS92 subacute joint-R2 change (cue) +.040 /
  -.079 / -.141.
- 2026-11-03 ~17:45: ITEM 3 (ITI) DONE: 107 / 107 sessions x 600 / 1200 / 1500 s, 0 failed (`iti/iti_drift_sessions.csv`,
  `iti_drift_animal_epoch.csv`, `iti_drift_pooled.csv`). Every baseline-subtracted section-14 readout is unchanged: ITI
  response / build-up, cued / cue response, per spout, contra / ipsi log ratio, balance proxy. Max |change of the pooled
  epoch - pre delta| between 600 and 1500 is .014, against a typical |delta| of 5.5 (dF/F x1e3). No n_up count changes.
  (E.g. ITI response tjM1 contra acute +12.601 -> +12.602, 4/4; build-up chronic -3.086 -> -3.086; balance cued chronic
  +.178 -> +.178; far-contra contra / ipsi log ratio acute +.590 -> +.590.) A slow basis is ~constant over a 2 s
  peri-event window, so the in-trial baseline removes it.
  Only the pre-cue LEVEL (no in-trial baseline; a proxy for the Fig 14.21 level vs RESTnl) moves. Far-contra, change
  from pre, 600 -> 1500: cortex acute +5.13 [.86, 10.4] 4/4 -> +5.00 [1.01, 9.9] 4/4; subacute -1.84 (1/4 up) -> -.99
  (3/4); chronic -3.48 -> -3.78; tjM1 contra acute +4.46 -> +4.51 (3/4), subacute +.74 -> +1.59; ALM contra acute
  +5.01 (3/4) -> +5.60 (4/4), subacute +1.36 -> +2.77. No CI changes side; acute signs are stable. The subacute /
  chronic level cells move by up to ~1-1.4 units, mostly PS92 / PS93 (PS92 cortex subacute 4.97 -> 6.42; PS93 pre level
  -3.21 -> -4.49).
- 2026-11-03 18:04: frozen encoder 900 s DONE; the chain that TaskStop had not ended finished on its own. Every
  process I started has now exited. Final compare `encoding/compare/` (600 / 900 / 1200 / 1500 + the 11-02 ref).

## Result (2026-11-03)

Headline cells, chronic - pre, cue window, frame R2, mean of animals [nested 95% CI], n up, at 600 / 900 / 1200 / 1500:
- target unique cortex: -.0132 [-.022,-.005] / -.0131 / -.0129 / -.0122 [-.021,-.004]; 0/4 at every cutoff.
- spout unique SSp-m contra - ipsi: +.0177 [.005,.031] / +.0202 / +.0216 / +.0214 [.008,.035]; 4/4 at every cutoff
  (lick window +.019 [-.000,.037] -> +.024 [.006,.040], 4/4). The rise is mostly PS92 (.013 -> .024) and PS94.
- joint R2 cortex pre per animal moves <= .016. Changes from pre move <= .02, except PS92 subacute (+.040 / -.010 /
  -.079 / -.141), carried by PS92_0824 / 0826.
Side effects: PS92_0824 frozen R2 .438 / .338 / .252 / .189; per-session block-CV cortex R2 .418 / .332 / .267 /
.147 (no drift basis -.072). PS92_0826 frozen .461 / .428 / .395 / .353; per-session .445 / - / .375 / .325. All other
sessions move by <= .02. No frozen all-window R2 goes negative at any cutoff.
RECOMMENDATION: keep 600 s for the encoder. No cutoff gives cycle retention >= .98 in every animal x epoch under the
encoder's gapped trial rows (even 2400 s: MIN .82-.99). Each step above 600 s brings back the PS92_0824 drift failure
that set 600 s. The headline conclusions are unchanged at every cutoff tested (target 0/4, spout contra - ipsi 4/4
with CI > 0). For the ITI scripts the cutoff does not matter for any baseline-subtracted readout. 1500 s would keep
>= .98 of the cycle band in every cell except PS92_0812 (time gap), but it moves the pre-cue LEVEL proxy in the
subacute / chronic cells and would make ITI and encoder inconsistent. So 600 s there too, with 1500 s as a reported
sensitivity row for level-type readouts (Fig 14.21) and for PS92.
