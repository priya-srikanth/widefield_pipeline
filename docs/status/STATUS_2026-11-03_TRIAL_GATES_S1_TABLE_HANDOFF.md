# Handoff 2026-11-03 — camera trial gates, jaw-only / incomplete figures, imaging section 1 table (window 5b)

Window `sabatinilab-5b`. Peers this touches: **e3** (kinematics page, owns it), **34** (cohort 1 swap-mask rebuild,
frozen decoders, epoch families; owns `_class_select` and the shared gates), **a8** (PI deck; its helper may
republish the imaging page with re-plotted figures).

## UPDATE 11-04 early — jaw-only OUT of working (Priya decision 11-03 late, via a8)

New definition: working = contact + incomplete camera licks, cut at the session's last camera lick; jaw-only becomes
its own class, like no-response (`docs/status/JAW_ONLY_CHECKS_2026-11-03.md`, a8 running list 11-03 AWAY). a8 is
implementing it, then ONE combined render; page edits come to this window (e0) after that render. **HOLD all page
edits (SPOUT_BALEM §7 included) until a8 sends figure paths.**

Imaging page v51 (scratchpad `imgpage/imaging_page.html`) sections to redo after the render. Rewrite the captions
cleanly, with no correction notes:
- the definition "jaw moved or any camera lick" is written out at lines 126, 127 (§0 box) and 154 (§1); change all three to the new wording;
- figures whose numbers come from working trials: §1 (Fig 1-1, 1-2, S1 table), §2 (2-1, 2-2), §3 (3-1, 3-3),
  §4 (4-1, 4-2), §6 (6-1..6-3), §7 (7-1..7-4); replace the numbers with the re-rendered ones;
- the LP-attempt decoder figures (2-3 and the §4 jaw-only mention) already exclude jaw-only trials; no change.
Kinematics / behaviour / FaceRhythm pages: no "working = jaw or lick" wording found (jaw-only is already its own class there).

## UPDATE 11-03 night (3) — PUBLISHING FOR OTHER WINDOWS (Priya: this window may publish any page)

- **Imaging page v51** (id 1793740803-1eef, on v50) for a8: `docs/status/STATE_BALEM_PAGE_EDITS_2026-11-03.md` (section 3 ->
  bal_em production state decoder; Figs 3.1-3.5 swapped to *_balem images; Closed / Pending updated; the new block is a plain
  note, not a CORRECTION label, per Priya's rewrite rule) + the imaging rows of `FIGURE_RULES_AUDIT_2026-11-03.md`
  (*_nolines images for 2.1-2.8 except 2.4b [no render], 4.2, 6.1-6.3, 9.1, 9.6-9.8, 10.1, 12.1; captions say faded open
  markers). Script: scratchpad `imgpage/apply_a8_1103.py` (asserts every replacement). Old images stay in the artifact,
  unreferenced. NOT done: 2.4b, 5.2 (no source), FaceRhythm page (asked a8), SPOUT_BALEM (a8 holds it until A3).
- **FaceRhythm page v36** (id 1793745454-bdeb, on 1793125970-97fe) for a8: the audit's 27 FaceRhythm rows, images uploaded as
  `fig/<name>_rules.png` (old files kept, unreferenced), a "Plot (figure rules)" note per caption (session markers; fit-level
  for 7.10 / 7.11 / 8.16 / 8.18). Script: scratchpad `frpage/apply_fr_rules_1103.py`.
- Account switched back to hhmi after these publishes; further publishing needs the HMS account again.
- **QUEUED for e3 (needs the HMS account):** kinematics page = `kin_page_src/kin_page/index.html` (v26 + 13 collapsed LMM
  toggles + FR cross-link; new files `lmm/<stem>_lmm.png`, 13 stems listed in e3's message); FaceRhythm page = 3 LMM toggles
  + kinematics cross-link, new files `lmm/{peak_side_headline,per_lick_stats,asym_aligned_change}_lmm.png`. The FR M: source
  (10-27) is OLDER than live v36: refresh it from live v36 first, re-run `kin_page_src/kin_page/add_lmm_toggles_1103.py fr`
  (idempotent), diff, publish. Send e3 the version numbers.
  **e3 11-03 (later): HOLD the kinematics publish.** e3 is splitting the page (DAQ-contact figures move to a new behaviour page, cross-linked), so `kin_lmm/` is superseded; publish kinematics + the new behaviour page from M: only when e3 says the sources are ready. FaceRhythm (`fr_lmm/`) still goes ahead. Account: the tool needs a Claude Code restart after switching to HMS (the switch alone left it on the old sign-in, list empty).
  **e3 11-03 (split READY) — publish order once the tool works:** (1) BEHAVIOUR, new page "Task Behaviour After Stroke", icon chart, from `M:/.../behaviour_page_src/behaviour_page/index.html`; local files `epoch_1f_engagement_by_epoch.png`, `lmm/epoch_1f_engagement_by_epoch_lmm.png`, `lmm/behaviour_hits_by_spout_lmm.png`; copy `b_timecourse.png` + `behaviour_hits_by_spout.png` server-side from the kinematics artifact (v26); NOT `epoch_1g_*`, `svg/` or `data/`. (2) KINEMATICS from `M:/.../kin_page_src/kin_page/index.html` (backups `index_pre_split_1103.html`, `index_v26_pre_lmm.html`), after replacing the 2 `href="#behaviour-page"` (data-behaviour-link) with the behaviour URL; files = 12 `lmm/<stem>_lmm.png`. (3) FACERHYTHM `fr_lmm/` staged + optionally the behaviour link in xlink-kin. Read each live page in full first; send e3 the three versions + the behaviour URL.
  **+ e3 (later):** kinematics files map also gets `p_phase_contra.png` (Fig 9.7 replaced by the 3-D phase figure with test bars; caption rewritten in index.html; same name, so it replaces the published file).
  **11-04:** sidebar figure toggles (first figure per section, rest behind a ▸ N button, as on the encoder / FaceRhythm / imaging pages) added to KINEMATICS v28 (1793762111-ab7c) and BEHAVIOUR v2 (1793762112-2bde), script scratchpad `add_toc_toggles_1104.py <page> <localStorage prefix>` (asserts once-only); e3 asked to fold it into the M: sources. Sharing checked 11-04: encoder, kinematics, FaceRhythm, behaviour = organization; IMAGING page still private (Priya to share). Publishing images: use the explicit {path: absolute source} `files` map (root + list form was refused).
  **11-03 21:55 ALL THREE PUBLISHED:** behaviour v1 (1793760190-da2d) https://claude.ai/artifact/M1xhZBWSPS84aELTSFaMMK; FaceRhythm v37 (1793760943-8fe2; 3 LMM toggles + kinematics / behaviour links); kinematics v27 (1793760972-f7dd; e3 split + 12 LMM toggles + new Fig 9.7 p_phase_contra.png, behaviour links set). The "Read permission rule" refusals were the `root` + list form of `files`; the explicit {published path: absolute source} map published fine. e3 told. (The block below is superseded.)
  **11-03 21:45 PUBLISH STATE:** BEHAVIOUR page published https://claude.ai/artifact/M1xhZBWSPS84aELTSFaMMK v1 (1793760190-da2d; b_timecourse / behaviour_hits_by_spout copied from kinematics v26). KINEMATICS and FACERHYTHM NOT published: every image upload after that was refused ("blocked by a Read permission rule"; p_phase_contra.png, then lmm/*.png); waiting on Priya to approve. Staged: scratchpad `kin_split/` (M: source with the 2 behaviour links set to the URL, 12 lmm PNGs + p_phase_contra.png; live still v26 1793740846-76c9) and `fr_lmm/` (behaviour link added to xlink-kin; live still v36 1793745454-bdeb).
  **STAGED 11-03** (scratchpad, session-local): `fr_lmm/index.html` (+ `fr_lmm/lmm/*.png`, `fr_lmm/fig/*_rules.png`) = v36 +
  toggles + xlink-kin; `kin_lmm/kin_page.html` (+ `kin_lmm/lmm/*.png`) = e3's source. In both the lmm-supp-style block was moved
  from the publish wrapper's head into the body (the wrapper head is replaced on publish). Before publishing: confirm live is
  still FR v36 / kin v26, publish with the lmm/ files map.
- **Kinematics page v25 -> v26** for e3 (source `kin_page_src/kin_page/index.html`, verified as a section-6-only diff against
  live; images uploaded by name).

## UPDATE 11-03 night (2) — SECTION 1 PUBLISHED (imaging page v50)

Priya moved Claude Code to the HMS account; the Artifact tool works again after a restart (this window is now
`sabatinilab-e0`). Published https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV **version 50** (id 1793735906-0006),
built on the live version 1793703965-38a8 (read in full first; a8's 11-03 Fig 3.5 and Pending item kept). Changes,
section 1 only, plus its summary row:
- summary row: ranges .43–.89 / .26–.56, "acute post-cue drop 4 / 4", input = camera trial classes;
- input line: camera-defined working trials to the last camera lick (Figs 1.1–1.3 still DAQ / engagement);
- per-animal table (Allen split + LocaNMF) + robustness table from `s1_update_20261103.html`;
- **Fig 1.4 swapped** to a8's `decoding_refit_sessions_campllick.png` (`fig_decoding_sessions_campllick_1103.py`;
  per-animal means equal the table; new file uploaded, old `decoding_refit_sessions.png` left unreferenced), caption
  rewritten;
- takeaway opening rewritten (old "up in PS92 / PS95" sentence removed); Fig 1.2 sentence (b); "Not the table above"
  notes in Figs 1.1 / 1.2 updated.
Published source: scratchpad `imgpage/imaging_page.html` (session-local). Other windows told (a8, e3, 34) and offered
publishing of their pages from here (Priya: this window may publish any window's pages). **Section 1 is done.**

## UPDATE 11-03 night — CUTOFF DECIDED: LAST CAMERA LICK IS PRIMARY (supersedes the block below where they differ)

**Priya 11-03: "last camera lick cutoff sounds good".** Primary trial set for section 1 = camera working up to the
session's last camera lick (`working_pl_lick` in the snapshot; columns `split_campllick_*` in `s1_refit_basis.csv`,
`trials == "camworkpl_lick"` in `s1_refit_locanmf.csv`). No re-run needed: every arm was already computed.
`p_img_s1_table_20261103.py` now builds the table from it; no cutoff, last contact, last close-spout lick, camera lick
gate, plain Allen and the DAQ gates are robustness rows. Blocks rebuilt (`s1_update_20261103.html`, `_numbers.md`).

**Numbers on the primary set** (Allen split; pre-cue pre -> acute; post-cue pre -> acute -> chronic): PS92 .36 -> .35,
.77 -> .59 -> .85; PS93 .40 -> .31, .64 -> .43 -> .82; PS94 .55 -> .42, .85 -> .62 -> .89; PS95 .34 -> .42,
.82 -> .75 -> .84. Ranges post-cue 0.43–0.89, pre-cue 0.26–0.56 (were .44–.88 / .27–.57 without the cutoff); no cell
moves by more than .02; conclusions unchanged (acute post-cue drop 4 / 4, back by chronic 4 / 4, pre-cue up PS95,
down PS93 / PS94, flat PS92; acute sessions decoded 5 / 4 / 6 / 1). LocaNMF agrees within .04.

**Told:** e3 (move `working_pl_lick` into `wfield_local.trial_classes` as the end-of-session rule; until then the
snapshot is the one definition), a8 (slide 9 ranges .43–.89 / .26–.56; trial line adds "up to the last camera lick").
**Done after:** e3 put the rule in `wfield_local.trial_classes` (`in_working(..., end="lick")`, b506c1e3; anchors
lick / contact / close / none); the snapshot script now calls it (0 mismatches vs the snapshot on 57,601 rows, all
anchors), so there is one definition. a8 re-plotted slide 9 on `split_campllick_*` (checked by export).
**Still open:** publishing (account access, below).

## UPDATE 11-03 late — SECTION 1 READY TO PUBLISH

**State:** both decoders done on the final classes (snapshot rebuilt at 3b7d08a4 = A4 + the close_center amendment;
working / camera_lick flags identical cue-for-cue to the 28f2a632 snapshot, only 1,121 unknown -> no_response).
The page blocks are built and committed (6418c9fe): `docs/results_pages/imaging/s1_update_20261103.html` (per-animal
table: Allen split + LocaNMF on camera-defined working trials; robustness table; Fig 1.2 sentence (b); takeaway) from
`p_img_s1_table_20261103.py`; deck numbers in `s1_update_20261103_numbers.md` (sent to a8 for slide 9).

**NOT PUBLISHED: the results pages are not visible to the current claude.ai account** (this window and a8 both; the
pages belong to the account they were made under). The Artifact tool is off in this session. To publish: sign in to
the owning account (or share the page with edit access), read the LIVE imaging page, replace section 1's per-animal
`<h4>` + table, the Fig 1.2 sentence ("A decoder trained on the acute session itself still loses the far spouts, so the
acute post-cue loss of section 2 is not only a frozen-decoder mismatch.") and the first sentences of the takeaway with
the blocks in the HTML file, and republish. Rewrite cleanly, no "CORRECTION" notes.

**Findings (per animal first):** post-cue 0.44–0.88, pre-cue 0.27–0.57 (chance .17); acute post-cue drop in 4 / 4
(PS95 now included; it went UP on DAQ licking trials); back at or above pre by chronic in 4 / 4; pre-cue no consistent
acute change (up PS95, down PS93 / PS94, flat PS92). Acute sessions with a decode 5 / 4 / 6 / 1 (was 4 / 4 / 4 / 1).
Robustness (mean of animals): trial gate, end-of-session cutoff, split vs plain, Allen vs LocaNMF move no cell by more
than a few hundredths; DAQ licking trials sit higher post-cue (acute .72 vs .60) because each window contains the lick.

**End-of-session cutoff (Priya 11-03: "report both, no primary yet"; her worry: a last-CONTACT cut drops real acute
far-spout attempts — confirmed):** three anchors in the snapshot (`working_pl_contact / _lick / _close`). Last CONTACT
drops 4.2 % of acute working trials incl. 94 incomplete licks; last CLOSE-spout lick 4.1 % and far-spout contacts
when a session ends on far blocks; **last CAMERA LICK drops 0.8–1.8 % and no attempt** (only jaw-only trials after the
final lick of any kind). e3 (QUIT_TAIL_JAW_ONLY_2026-11-03.md) recommends `working_pl_lick` as primary; Priya to decide.
Section 1 shows plain camera-working (table) and the last-camera-lick row (robustness table).

## UPDATE 08:53 — earlier state

**Running now** (camera lick source `WIDEFIELD_LICK_SOURCE=lp`, ENL pinned 2.0 s; gates read from the A4 snapshot):
* Allen table `wf_attempt/scripts/s1_refit_basis_table.py --jobs 6` -> `imaging_encoding/LP_licks_ENL2s/qc/s1_refit_basis.csv`
  (13 / 108 at 08:53, ~25 min). Arms per feature set (plain_orig / plain_v8 / split): **camwork** (DEFAULT),
  **camlick**, quit-tail working, LP licking.
* LocaNMF table `scripts/s1_refit_locanmf_table.py` -> `imaging_encoding/LP_licks_ENL2s/qc/s1_refit_locanmf.csv`
  (precue done, cue running). Arms: camwork / camlick / quittail.
* DAQ robustness rows already done: `imaging_encoding/qc/s1_refit_basis.csv` (quit-tail working + DAQ licking, all
  three feature sets) and `qc/s1_refit_locanmf_figrule.csv` (checked 0 mismatches vs Figs 1.1 / 1.2).
* PS93_0813 (no video) fails cleanly in the LP runs (LPUnavailable): expected; it stays in the DAQ rows (Priya: keep it
  wherever only contact is needed).

**Decided since the first version of this handoff (Priya, 11-03, partly via e3 / a8):**
* **A4 — one trial classification for every analysis**: `wfield_local/trial_classes.py` (e3, 28f2a632; jaw = cam4 OR
  cam1, DAQ contact primary, `window` param). The GATES / session_classes / cue_classes / in_gate names in
  `jaw_only_incomplete_figs` are unchanged; `classify()` is gone. e3 OWNS the A4 / A5 consolidation edits; this window
  no longer edits the class rules.
* **Window = cue_stop for BOTH the ENL and the cue columns** ("working" = responded to THIS trial's cue; the ENL window
  is lick-free by its own rule).
* **close_center no-lick, no-contact, no-tongue = NO_RESPONSE** (not unknown), with a `jaw_judged` column for a
  sensitivity drop (e3 23c8f775, local; on main with A1 shortly). Does NOT change working or camera_lick, so the running
  decodes stay valid.

**Class snapshot** (new, main): `scripts/s1_trial_class_snapshot.py` -> `imaging_encoding/qc/s1_trial_classes_cue_stop.csv.gz`
+ `.json` (commit 28f2a632, clean). One row per DAQ cue of the 108 curated imaging sessions: cls, working,
camera_lick. Both decoders INDEX it (the branch has no `trial_classes.py`; no copy of the rules anywhere). Shares:
contact .821, jaw_only .060, no_response .058, incomplete .040, unknown .021 (close_center no-lick; -> no_response
under 23c8f775).
Caveat measured: 807 jaw-only trials fall after the session's last contact lick (~1.6 % of the working set, mostly
the sated tail). e3 passes "quit-tail AND jaw-or-lick" to Priya as an option; nothing changed.

**Next (in order):**
1. When e3 says 23c8f775 is on main: re-run the snapshot (`python -m scripts.s1_trial_class_snapshot`, ~2 min) and
   ASSERT every cue's working / camera_lick flag equals the 28f2a632 snapshot (keep a copy of the old one first,
   `cp -n`). If any flag differs, re-run both decoders.
2. When both runs finish: assemble the section 1 table = PRIMARY camera-working on split Allen + LocaNMF (pre-cue and
   post-cue, per animal x epoch, August pre); robustness = camera lick gate, plain Allen, DAQ quit-tail working, DAQ
   licking (the old table). Per-animal FIRST (rule 8).
3. Imaging page section 1 (live page; rebuild from live immediately before publishing — a8's figure-rules helper also
   republishes it): table, Fig 1.2 sentence (b), takeaway, the "acute 4 / 4 / 4 / 1" counts (now 5 / 4 / 6 / 1 with
   working trials). No "CORRECTION" notes.
4. Send a8 the slide 9 numbers + source file; tell 34 the section 1 scripts / outputs for the rebuild inventory.

**Commits since the first version:** main 6553eb39 (snapshot + LocaNMF camera arms), b1d8377d, aca69271; branch
analysis/attempt-coding e7997933, 79da1615, a9355d14 (local, not pushed).

**Pitfall:** the repo writeguard refuses an LP-source run's write outside an `LP_licks_*` directory — use
`lp_licks.output_dir(out)` (both scripts do).

## Decisions needed from Priya

None open (the post-quit jaw-only option is with e3 -> Priya). Decided today:

* **"Working" trials (11-03):** for ENL and cue windows, working = **jaw moved OR any camera-seen lick**
  (contact, incomplete or jaw-only), per trial, no session-level rule. The DAQ quit-tail rule
  (`grant_kit._class_select("working")`) becomes a robustness row. **Camera lick gate** = contact + incomplete.
  **Lick-aligned windows always use the lick gate.** This is the DEFAULT for the cohort 1 rebuild (window 34
  switches it there, once).
* **End-of-session cutoff (11-03 night):** working trials stop at the session's **last camera lick** (contact or
  incomplete); jaw-only trials after it are dropped (0.8–1.8 % of working; no attempt lost). Last contact / last
  close-spout lick / no cutoff = robustness rows.
* **Imaging page section 1:** table re-run on the **split Allen basis** (ALM / tjM1 + MOs / MOp rests; the 10-27
  primary basis), with a **LocaNMF** row; the Fig 1.2 sentence **softened (option b)**; publish **now** on the current
  signal; the cohort 1 rebuild re-renders it later with the same scripts.

## Done today (commits)

| what | where | commit |
|---|---|---|
| Jaw-only / incomplete / contact trial classes + figures (kinematics page section 6) | `scripts/jaw_only_incomplete_figs.py` (main) | 0bc154cb, c0ce2627 |
| Denominator fix: ALL engaged trials (jaw_pass_qc is the jaw-MOVED call, not QC; caught by e3) | same | 07385b37 |
| **Shared camera gates**: `GATES`, `session_classes`, `cue_classes`, `in_gate` beside `classify` | same | aca69271 |
| Section 1 LocaNMF refit per-session table (`--fig-rule` = Figs 1.1 / 1.2 rule) | `scripts/s1_refit_locanmf_table.py` (main) | aca69271 |
| Section 1 Allen table, three feature sets (plain_orig / plain_v8 / split) x licking + quit-tail working | `scripts/s1_refit_basis_table.py` (branch analysis/attempt-coding, wf_attempt; local, not pushed — the v8 ROIs exist only there) | 9f6c7a27, a9355d14 |

### Shared gates — what consumers need to know (`scripts.jaw_only_incomplete_figs`)

* `session_classes(label)` -> every trial (no engagement filter): trial_id, position, cue_daq, cls, has_cam1.
* `cue_classes(table, cue_s)` -> class per DAQ cue, matched by cue TIME within 20 ms (never trial id).
* `in_gate(cls, "working" | "camera_lick")`.
* Coverage, 108 curated imaging sessions from 08-06: **107 with v36** (PS93_0813 has no behaviour video —
  EXPERIMENT_ERRORS 2026-08-13; it drops out of camera-gated rows only), **all 107 with cam1**, >= 99.7 % of DAQ
  cues matched except **PS92_0908 (82.5 %**, video covers 321 / 389 trials; the rest are "unknown").
* cam4-only sessions (none today) would be classified without cam1 (no cam1-only licks, jaw-only not cam1-corrected).
* "unknown" (no jaw call and no lick, or no video for the trial) is in NEITHER gate.
* Share of trials in the gates (mean over sessions): working pre .97 / acute .81 / subacute .88 / chronic .97;
  camera lick .94 / .65 / .78 / .95.
* Docstring relates it to the other "responded" definitions (`rest_engagement`, `rest_by_position._engaged_trials`,
  `nolick_decoder.categorize`, WIDEFIELD_GATE_SOURCE=lp) until those are consolidated (e3's 11-03 repo audit).

## Section 1 table — state

* **LocaNMF row** (`imaging_encoding/qc/s1_refit_locanmf_figrule.csv`, checked: **0 mismatches** against the 5r records
  over 152 post-stroke days + pre lists). Post-cue acute drop in **4 / 4** (incl. PS95, which goes UP on plain Allen);
  pre-cue no consistent acute change. `s1_refit_locanmf.csv` = working-trial pre (quit-tail rule) — superseded by
  the camera-working rule, re-run pending.
* **Allen rows** (`imaging_encoding/qc/s1_refit_basis.csv`): RUNNING at handoff (08:09, 56 / 108 sessions),
  quit-tail working + licking. `plain_orig` licking reproduces the current table exactly (PS92_0806 .263 / .567).
* The three acute sessions without a value in the current table (PS92_0818, PS94_0817, PS94_0820) had **no
  far-contra licking trial** (PS94_0817 none at far mid either); the decoder needs all 6 positions. Working trials
  include the misses, so they decode (PS92_0818 split post-cue .55).
* Adding no-lick trials lowers post-cue accuracy in EVERY epoch (PS92_0806 .57 -> .49): the post-cue window of a
  licking trial contains the lick. Read the working table as its own level, not against the old numbers.

## Next steps (in order)

1. Switch both table scripts' working arm to the shared camera gate (`cue_classes` + `in_gate`), keep the quit-tail
   arm as robustness; add the camera-lick arm (licking trials = contact + incomplete). Re-run (Allen ~25 min, LocaNMF
   ~5 min).
2. Imaging page (https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV) section 1 ONLY: new table (split + LocaNMF
   primary; plain + quit-tail + DAQ-licking robustness), Fig 1.2 sentence (b): "a decoder refit on the acute session
   still loses the far spouts, so part of the acute loss is not a frozen-decoder mismatch; fewer acute trials and
   changed licking may contribute", takeaway. **Rebuild from the LIVE page immediately before publishing**
   (a8's helper republishes it); rewrite captions cleanly, no "CORRECTION" notes (Priya 11-03 rule).
3. Tell **a8** the new section 1 numbers: deck slide 9 quotes them (".55–.91 post / .28–.55 pre", "acute pre-cue up
   in PS92 / PS95, down in PS93 / PS94", acute 4 / 4 / 4 / 1 sessions) from `slowband_decode_all.csv`.
4. Tell **34** the gate import path is live (aca69271, `scripts.jaw_only_incomplete_figs`).

## Pitfalls hit today

* `epochs.epoch_of` costs ~35 ms a call: per-row lookup over ~700 k trace rows ran for hours. Look up per session.
* `D.align` in pandas is the `DataFrame.align` METHOD, not the column: `D.align == x` is always False. Use `D["align"]`.
* A process pool run from `python -c exec(...)` without a `__main__` guard breaks on Windows spawn
  ("BrokenProcessPool"), not an OOM.
* Window 34's `git commit -am` in wf_attempt swept up this window's uncommitted file (undone, nothing lost). Commit
  explicit paths only; commit before long runs.
