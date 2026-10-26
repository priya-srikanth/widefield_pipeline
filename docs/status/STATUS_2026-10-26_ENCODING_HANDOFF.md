# Encoding handoff — 2026-10-26 (updated ~19:35) — READ FIRST

Supersedes the RUNNING / TO-DO lists of `STATUS_2026-10-22_ENCODING_HANDOFF.md` §000z. Window = session `sabatinilab-d4`
(pose window `sabatinilab-5b`, FaceRhythm `sabatinilab-e3`). Worktree `C:/Users/SabatiniLab/Github/wf_attempt` (branch
analysis/attempt-coding). Old scratchpad (this session only; a NEW session can read it by absolute path but cannot
publish from it): `S = C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad`.
Python: `C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe`, run from the worktree with
`PYTHONPATH=$(cygpath -w $PWD)` (another `wfield_local` install shadows it). ITI outputs: `<E>/iti_licks/` =
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/` (figures in `figures/`).

## 0. START HERE (19:35)

### 0a. PUBLISH THE PAGE — exact steps for a new session

The Artifact tool was switched off in this session from ~14:15, so everything since v103 is LOCAL ONLY.
- Live: https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d = **v103** (12:00). Everything after it exists only locally.
- **Local source (newest, complete): `C:/Users/SabatiniLab/Github/wf_attempt/_goal_page/index.html`** (+ its 190 PNGs;
  untracked, listed in `.git/info/exclude`; 63 MB). Backup copy: `M:/MICROSCOPE/Priya/Widefield/encoding_goal_page_src/`.
  The old `S/goal_page/` was identical at 19:30.
- What the local page adds over v103:
  - a TITLE above every figure (`<div class="ftitle">`, also used by the sidebar; source `_goal_page/fig_titles.csv`);
  - section-3 captions open with "Decoder: frozen / per-session" and "Measured: r vs dF/F";
  - 3.9 caption carries the paired ITI vs in-trial test;
  - new figures: 1.37 TE partial refit, 3.11 lick rate, 3.14 basis decoder, 3.15 where decoder, 3.16 lateral x bases,
    3.26 orofacial vs control PETHs; supplements 3.17 / 3.18 = the where decoder (every target) in the joint LocaNMF
    and pre-stroke SVD bases (`iti_where_decoder_locanmf.png`, `iti_where_decoder_svdbasis.png`);
  - corrected whole-cortex claim; 3.23 kinematics control with a face-adjusted row;
  - all section-3 figures redrawn: frame / measure labels, stars above data, SEM bands on 3.7 / 3.19 / 3.22,
    3.15 split in two.
- **Step 1 — bring in the last redrawn images** (only if the redraw chain finished after 19:30; check `S/relabel.log`
  ends with `relabel3 done`):
  ```bash
  F="M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/figures"
  G=/c/Users/SabatiniLab/Github/wf_attempt/_goal_page
  for n in iti_basis_decoder iti_area_maps iti_area_peth iti_area_groups iti_area_compare iti_where_decoder \
           iti_where_decoder_bases iti_where_decoder_locanmf iti_where_decoder_svdbasis iti_frozen_timecourse iti_frozen_summary iti_ramp_sessions iti_ramp_summary \
           iti_ramp_peth iti_ramp_peth_isolated iti_kinematics_control; do cp "$F/$n.png" $G/; done
  ```
- **Step 2 — read the live page first.** Call Artifact with `action: "read"` and the URL. A publish to an artifact the
  session has not read is refused. Confirm the live page is our v103: its 3.x sections end at the whole-cortex
  supplement, and it has no `ftitle` divs. The local file is a strict superset of v103: no one edited the live page
  after 12:00.
- **Step 3 — publish** with ALL changed images in `files` (publishing index.html alone does NOT upload PNGs; v98-v102
  showed 9 broken figures that way):
  - `action: "publish"`, `url`: the page URL above;
  - `file_path`: `C:/Users/SabatiniLab/Github/wf_attempt/_goal_page/index.html`;
  - `root`: `C:/Users/SabatiniLab/Github/wf_attempt/_goal_page`;
  - `files`: every PNG in `_goal_page` modified after 2026-10-26 12:00, each mapped to itself. Generate the map with:
    ```bash
    cd /c/Users/SabatiniLab/Github/wf_attempt/_goal_page && find . -maxdepth 1 -name "*.png" -newermt "2026-10-26 12:00" \
      | sed 's|^\./||' | awk 'BEGIN{printf "{"} {printf "%s\"%s\": \"%s\"", (NR>1?", ":""), $0, $0} END{print "}"}'
    ```
    About 25 files. Expect at least: te_partial_refit_lines, iti_area_compare, iti_basis_decoder, iti_where_decoder,
    iti_where_decoder_bases, iti_lick_rate, iti_direction_decoding, iti_frozen_timecourse, iti_frozen_summary,
    iti_ramp_sessions, iti_ramp_summary, iti_binned_sweep, iti_binned_summary, iti_recovery_marker,
    iti_kinematics_control (+ the step-1 files).
  - `label`: e.g. "Titles, ITI labels, where + refit".
  - If the refusal hands back the live version, check it is v103 and publish the local file again. It is the merge,
    since local = v103 + our edits.
  - Then open the published page and spot-check 3.7, 3.15, 3.16, 3.23 and 1.37 for images.
- Numbering: **Priya may quote LIVE (v103) numbers.** Local numbers moved after several inserts: her "3.17" on
  10-26 evening was the kinematics control, which is 3.23 locally. Map with `_goal_page/fig_map.csv` (last renumber)
  and `docs/status/GOAL_PAGE_FIG_MAP_2026-10-26.csv` (pre-10-26 → current).

### 0b. RUNNING (background in THIS session; may die with it)
- **Redraw chain** `S/run_relabel.sh`, log `S/relabel.log`, started 18:45. Steps:
  1. plot-direction / key / binned / recovery marker: DONE;
  2. basis decoder `--summarise` was running at 19:30;
  3. then `iti_area_maps.py`, `iti_area_maps.py --compare`, `iti_where_decoder.py` (full, ~25 min);
  4. then `iti_licks.py --key` again (3.8 / 3.9 / 3.20 with the star fix);
  5. then `iti_licks.py --ramp` (SEM bands on 3.19 / 3.22).

  If the log does not end with `relabel3 done`, rerun the missing steps from the worktree with the PYTHONPATH above.
  Each step redraws from saved session data; nothing is lost. Then do 0a step 1.

- **Decoded vs actual landing scatter** (Priya ~19:40: "plot the actual 3d (or 2d) decoded vs true lick positions for
  given spout positions per epoch per position"; NEW `scripts/iti_where_scatter.py`, started ~19:45):
  - Same frozen decoder as 3.15: one ridge per axis (lateral, outward), PER LICK.
  - Saves `<E>/iti_licks/where_preds.csv` (one row per lick).
  - Figures `iti_where_scatter_{trial,iti}.png`: rows = all animals + each animal, columns = epochs; per spout, an arrow
    from the actual mean to the decoded mean, with 1-SD ellipses.
  - In-trial licks are grouped by target spout; ITI licks by the spout nearest their actual landing.
  - Ridge shrinks predictions toward the centre, so read each epoch against pre.
  - `--plot-only` redraws from the CSV.
  - When done: look at the figures, add them to the page as a KEY / supplement pair after 3.16, then renumber.
- **Helper agents launched ~19:45 (results may land in this session):**
  - (1) TITLE VERIFICATION from the code for figures whose model frame was inferred: 3.1-3.6, 1.15-1.16, 1.18, 1.21-1.22,
    1.24-1.28, 1.44-1.47, 7.1-7.6, 9.13-9.14, plus a spot-check of sections 5 / 6 / 8. Output
    `_goal_page/title_verification.csv` (proposed_title column). Apply the proposed titles to `_goal_page/index.html`
    (edit the `<div class="ftitle">` text inside ONE figure: see pitfall 0f), then republish.
  - (2) PUBLISH attempt by a subagent (in case the Artifact tool is available to it). If it reports the tool is off
    too, the page is still v103: use section 0a from a new session.

### 0c. WAITING ON OTHERS
- **sabatinilab-5b:** "go" for the kinematics_all swap sent 15:50; Priya runs the rename lines 5b gives her. When the
  table is live, re-run `python -m wfield_local.stray_licks`. Move the old `<E>/stray_licks/sessions` aside first; it
  skips sessions already done. These are derived outputs, so no approval is needed for that rename. Then update
  ENCODING_FINDINGS 2.6.

### 0d. TO DO (in order)
1. **Per-time-bin significance bars on the time-course figures** (Priya 19:30: "for the post-lick response curves and
   decoder curves -- significance testing and mark with lines (colored per epoch) where in the time bins curves
   differ from pre"). Plan, agreed in chat:
   - one helper in `scripts/iti_licks.py`, e.g. `sig_bars(ax, X_by_animal_epoch, lags, colors)`;
   - for each post epoch, `ak.boot_delta_columns(post, pre, rng)` with `{animal: array (n_sessions, n_lags)}`. This is
     the same paired animals → sessions draw as every dot-plot star, vectorised over time bins;
   - then `ak.boot_p_columns(draws)` → `ak.bh(p, q=0.05)` across the bins of that curve;
   - draw contiguous significant runs as thick horizontal segments in the epoch colour (`EPOCH_LINE`) in a strip below
     the curves, one row per post epoch (x in data, y in axes: blended transform);
   - say "paired nested bootstrap per time bin, BH-FDR q = 0.05 across bins" in the suptitle and caption.

   Figures and their data:
   - 3.7 left / middle: per-session decoder per lag, `decode_sessions.csv` (label, animal, epoch, set, target, lag, r)
     → redraw via `--plot-direction`.
   - 3.8: frozen decoder per lag, `decode_frozen_lags_sessions.csv` → the `key_figures` block (`--key`).
   - 3.14: basis decoder time courses (`iti_basis_decoder.py --summarise`).
   - 3.19 / 3.22: `ramp_figure`, animal-mean PETH normalised by pre amplitude. Test on per-session PETHs (same
     normalisation) → `--ramp`.
   - 3.20: `iti_ramp_sessions.png`, raw + normalised rows, per-session PETHs → `--key`.
   - 3.26 orofacial vs control PETHs (`iti_area_maps.py --compare`); optionally 3.25.

   Then update captions and republish.
2. TE partial-refit line figure (page 1.37): the title still names only the target null. Add the landing − target
   lick-kernel contrast, which is significant but not drawn (`boot.csv`).
3. 3.9 caption: add the lick-count control line (matched 40 licks / session: same numbers; `iti_nmatch_check.py`).
4. stray_licks re-run (0c).
5. Older open items: PRE-level TE group-kernel maps (replace 1.20-1.21); epoch-kernel drift-RATE test (8.18);
   every-lick window (Priya to confirm first-lick primary); June→August drift ceiling; lesion burden /
   leave-PS93-out; share deletions / nokin retirement (Priya decides).
6. **Push**: origin has everything through 66a287e. Local commits after it (31423a1, a73908c, d384d9b, 270885e, + this
   handoff) need a cherry-pick push:
   ```bash
   git fetch origin; git worktree add --detach ../wf_push_tmpNN origin/main; cd ../wf_push_tmpNN
   git cherry-pick <commits>; git push origin HEAD:main   # retry: fetch + rebase origin/main
   ```
   The pre-push tests take ~7–14 min. Never rebase the local branch, never force-push. `S/push28.sh` is the template.

### 0e. Results since 13:50 (PROVISIONAL; ENCODING_FINDINGS 8.17 / 8.20, FINDINGS 4d vii-x)
- **ITI lick rate (BEHAVIOUR):**
  - PS92-94 drop acutely and stay 14-28 % below pre chronically.
  - Only PS95 ends above pre: about 4×, from near-zero pre ITI licking.
  - Rate tracks far-contra hit rate within animal (r +0.67; +0.51 with the log-day trend removed, PS94 ~0).
  - So rate is not the cause of the decoder's ITI loss.
- **Where decoder:**
  - The ITI loss is LATERAL, in every basis (−19 to −20 %, 4/4); outward position is weakly decoded and unchanged.
  - In-trial: outward drops acutely and recovers; angle ends above pre (+7 to +11 %).
- **ITI vs in-trial, paired test (3.9):** chronic −0.062 [−0.110, −0.014], 3/4 animals (PS93 0.00). Say "in three of
  four animals", not "confidently".
- **TE partial refit (1.37):**
  - Landing-labelled minus target-labelled lick kernels: lower chronically in every area (contra SSp-m −0.019, cortex
    −0.024; 0/4 up).
  - Acute target advantage falls (−0.126), back by chronic.
  - Same without PS95_0907.
- **Kinematics control + FACE (3.23 row 4):** the larger post-lick response survives regressing out reach + speed +
  post-onset FaceRhythm (tjM1 contra chronic +0.74 [+0.22, +1.26], SSp-m +0.63 [+0.14, +1.11]); unchanged from reach
  + speed alone.
- **Decoder r vs AUC:**
  - r = information about lick position (decoded vs actual lateral tongue-tip position).
  - AUC = amount of activity (dF/F integrated over a window); not ROC.
  - They can move independently: post-stroke AUC is up, frozen ITI r is down.

### 0f. NEW PITFALLS (evening)
- **The Artifact tool can be switched off mid-session.** Keep the page source somewhere a new session can publish from
  (now `_goal_page/` in the worktree). Pass every changed PNG in `files`.
- **Regex over the page:** `<figure id="...">.*?X` with re.S spans ACROSS figures. This hit Fig 3.1, whose title and
  caption were overwritten, then restored. Match inside one figure: `<figure id="(fig[^"]+)">(?:(?!</figure>).)*?X`.
- **Escapes in Bash:** a `\\n` inside a Bash heredoc reaches Python as a real newline (the tool layer unescapes it).
  Build backslashes with `chr(92) + "n"`, write the script with the Write tool, or use Edit.
- **`renumber_sections.py`** defaults to `goal_page/index.html` next to the script. In `_goal_page/`, pass the
  index.html path explicitly.
- **Background jobs:** a script launched with `&` inside a run_in_background shell gives no completion notice (wait on
  a log line instead). Define every variable (`S=...`) inside chain scripts; one chain ran 8 steps with `$S` empty.
- **Editing a script while a chain runs:** a step already started keeps the old code. 3.8 / 3.9 were redrawn before the
  star fix, hence the queued `--key` rerun.
- **Old-design vs new-design figure names:** `partial_refit_lines.png` = OLD design (1.38);
  `te_partial_refit_lines.png` = TE design (1.37).

# EARLIER 10-26 (history; superseded by section 0)

## RUNNING as of 16:20 (history)

### NOW (16:20) -- read this block first
- **DONE 15:46: TE partial refit** (exit 0; ENCODING_FINDINGS 8.20 top, FINDINGS 4d x; LOCAL page 1.37 as
  `te_partial_refit_lines.png` -- NOT `partial_refit_lines.png`, which is the OLD design's figure, now 1.38). Result:
  landing-labelled minus target-labelled lick kernels lower chronically in every area (0/4 up); acute target advantage
  falls. Same without PS95_0907. To do: retitle the line figure for the TE nulls + draw the landing - target contrast.
- **"go" SENT to sabatinilab-5b 15:50; waiting for them to report the table live.** Was: AFTER the refit exits: message sabatinilab-5b "go" for the kinematics_all swap (their regenerated table in
  `session_poses/pose3d/kinematics_all__tmp_20261026/`: only daq_peak_s differs, diff checked; Priya APPROVED the rename
  10-26 and runs the lines 5b gives her). Then re-run `python -m wfield_local.stray_licks` (patched 1d6e227; move the old
  `<E>/stray_licks/sessions` aside first -- it skips sessions already done; derived outputs, Priya's OK not needed to
  rename) and update ENCODING_FINDINGS 2.6.
- **DONE 14:35: frozen decoder in other bases** (`iti_basis_decoder.py`): result holds in joint LocaNMF and the pre-stroke
  SVD basis (ENCODING_FINDINGS 8.17 top, FINDINGS 4d vii); figure added to the LOCAL page after 3.13 (unpublished).
- **WHERE decoder** (Priya 10-26: angle / distance together, "or lateral location instead of angle"):
  `scripts/iti_where_decoder.py` DONE 15:00 (ENCODING_FINDINGS 8.17 top, FINDINGS 4d viii); on the LOCAL page as 3.15
  (unpublished) (it picks up the basis
  folders if present and raises if a basis file does not line up). Targets lateral (side_ml), outward (out_ap), 2-D
  vector r, angle / reach polar check; prints reach spread per epoch. -> page next to 3.9.
- **ITI lick RATE vs recovery** (Priya 10-26: "does inter-trial licking correlate with recovery (and exceed pre-stroke
  levels)"): `scripts/iti_lick_rate.py` DONE 15:40 (ENCODING_FINDINGS 8.17 top, FINDINGS 4d ix; LOCAL page 3.11; `--plot-only`
  redraws from the CSVs). Answer: dips with the deficit and returns, exceeds pre only in PS95. Rate = ITI licks / ITI minutes (next strobe - stop, clipped 0..60 s), no-contact rate, per trial, contact share;
  change vs pre + within-animal correlation with far-contra behaviour (raw, log-day detrended). Earlier count check:
  fewer ITI licks post-stroke in PS92-94 (so exceeding pre is unlikely in counts; rate is the fair test).
- **Page publish pending**: files = iti_area_compare.png, iti_basis_decoder.png, iti_where_decoder.png, iti_lick_rate.png, iti_direction_decoding.png (3.7 now with SEM bands), iti_where_decoder_bases.png (new 3.16) + every section-3 PNG redrawn 10-26 evening (frame + measure labels, stars above data, SEM on 3.19 / 3.22); every figure now has a title (.ftitle, fig_titles.csv in the scratchpad),
  te_partial_refit_lines.png
  (all copied into S/goal_page; renumbered 16:00, map in GOAL_PAGE_FIG_MAP_2026-10-26.csv).
- Pushed to origin through e9656c8; LOCAL commits since (1d6e227 stray fix, b60d363, ff61daa, d47d32c, + this) need a
  cherry-pick push (memory note: never rebase this branch; cherry-pick onto origin/main in a temp worktree).


1. **DONE 11:05 (areas 08:49, basis 10:10, self-fit 11:05; first readouts ENCODING_FINDINGS 8.20, page 1.34 / 1.35,
   `scripts/te_readout.py`). TE partial refit now running (`S/partial_te.log`).** 1. **Encoder redesign, FROZEN_DESIGN=te** (`S/run_te.sh`, log `S/te_chain.log`; started 06:34):
   frozen AREA run `frozen_encoder_gate-attempted_v7g3_t3d_frA_te` (log `S/frozen_te.log`, `--no-drop-spout`, default
   variants) → PIXEL basis run `..._frA_te_basis` (FROZEN_PIX_MUSALL=1; `S/frozen_te_basis.log`) → self-fit
   (`session_vs_frozen.py --tag ..._frA_te --variants joint no_target no_execution no_contact no_task no_movement`, `S/svf_te.log`).
   Design (DECISIONS 2026-10-26; code `wfield_local/frozen_encoder.py` TE_BLOCK + `te_lick_parts`): TARGET = pos_enl,
   pos_cue + lick-locked kernels labelled by the trial's target spout (post-cue, pre-cue/ENL, cam1-only licks; never ITI);
   EXECUTION = unsplit 3-D tongue/jaw, exec angle, all FaceRhythm, lick-locked kernels by 3-D LANDING (+ unknown), ITI
   licks included; CONTACT = every DAQ-detected lick of the session (Priya: contacts just outside trials count); nuisance
   timing / other. Frames = whole engaged session incl. ITI (14 % of frames pre, 29 % acute). Smoke PS92_0806 / 0819 OK.
2. **DONE 07:55 -- ITI chain v4** (wide decoding lags -1..+1 s): `S/iti_full4.log` (re-extract + summary) → `S/iti_key4.log` (key
   figures) → `S/iti_ctrl4.log` (controls) → `S/iti_binned.log` (binned frozen-decoder sweep). Outputs `<E>/iti_licks/`
   (session npz backups of earlier versions: `sessions_v1..v3`, figures `figures_v3`; derived, deletable).
   At 06:59: TE frozen area run in its pre-statistics pass (17/31 pre sessions); ITI v4 extraction DONE (summary
   running, then key / controls / binned); whole-cortex maps (job 4) STARTED; old partial refit 64/101.
3. **DONE 08:31 (results ENCODING_FINDINGS 8.19 top; figures in its partial_refit/figures) -- Lick-kernel partial refit, OLD design** (`S/run_partial_ls.sh`, 7 workers, `S/partial_ls7.log`; 62/101 at 06:56).
   Superseded for the contest by the TE design, but let it finish: then `run_partial.sh`-style `--plot-only` with
   `--tag gate-attempted_v7g3_t3d_frA_ls`.
4. **DONE (re-run 08:30 on RAW dF/F AUC; per-area normalisation dropped) -- Whole-cortex supplement** (Priya 10-26: "non-orofacial per-area activity maps ... supplemental figure with VIS
   areas"): `scripts/iti_area_maps.py`, queued to start when the ITI re-extraction finishes (`S/run_area.sh`, log
   `S/iti_area.log`) -> `iti_area_maps.png` (per-area response / build-up AUC change from pre on the Allen atlas,
   significant areas outlined), `iti_area_peth.png` (non-orofacial PETHs: VISp, VISl, VISam, RSPd, SSp-bfd / ul / ll / tr,
   ACAd, PL), `iti_area_groups.png` (orofacial vs visual + RSP, session dots) -> page supplement (section 3).
5. **DONE (all 10-26 work through 13:00 on origin)** -- Push of the 10-26 commits: the first attempt hit a rebase CONFLICT in DECISIONS.md (another session appended a
   10-25 FaceRhythm entry at the same place); resolved by keeping both entries (theirs first), rebase completed, push
   retrying (task in this session). CONFIRM on origin: `git log origin/main --oneline | grep "Handoff 2026-10-26"`.

## TO DO (in order, with why)

A0. **DONE v98.** **Whole-cortex supplement → page** (running job 4): say in the caption whether the larger response / weaker
   build-up are orofacial-specific or cortex-wide (orofacial vs visual + RSP test).
A. **DONE v98 (results ENCODING_FINDINGS 8.17 top, FINDINGS 4d iii-iv).** **ITI v4 outputs → page**: frozen time courses now -1..+1 s (3.8), refreshed 3.9 / 3.12 / 3.13; binned decoder
   (`iti_binned_sweep.png` = configuration sweep, boxed = chosen on PRE only; `iti_binned_summary.png` = one number per
   session at the chosen configuration, raw + normalised, all areas / orofacial / visual+RSP). Priya asked for this
   ("binned time dynamics ... single-number decoder comparison across sessions").
B. **(i) + (ii) DONE 09:00** (ENCODING_FINDINGS 8.17 top, page 3.9 caption; `scripts/iti_contact_check.py`): no-contact
   in-trial licks recover too; paired ITI - in-trial gap widens chronically -0.062 [-0.110, -0.014]. (iii) DONE 09:30 (`iti_recovery_marker.py`,
   page 3.10): in-trial tracks the recovery time course (+0.62; spread-controlled +0.53), ITI does not;
   beyond the time trend +0.16 n.s.
   B. **3.9 decisive checks** (Priya: "3.9 looks really good — is it as important as I think?"): (i) in-trial licks
   WITHOUT spout contact through the frozen decoder (does in-trial recovery ride on contact?); (ii) session-paired
   nested test of the ITI − in-trial frozen difference (per animal chronic gap: PS92 -0.11, PS94 -0.16, PS95 -0.24,
   PS93 -0.01). (iii) **Recovery marker** (Priya: "perhaps a marker of recovery?"): within animal, does the in-trial
   frozen decoder track that day's behaviour (far-contra hit rate / pre, `recovery_timing.series`) across post sessions,
   and the ITI one not? Chronic behaviour has no between-animal spread, so it is a within-animal time-course test.
C. **Frozen unique + self-fit DONE (8.20).** Pixel CHANGE maps of the group parts DONE (page 1.36, `frozen_pixel_maps.py` now shows `execution`). Still open:
   PRE-level group-kernel maps (each group's evoked part per spout in pre; replace 1.20-1.21);
   pixel unique maps; TE partial refit readout when it lands (`partial_refit.py --plot-only --tag ..._frA_te`
   with FROZEN_DESIGN=te). C. **TE readouts** when run 1 lands: unique(target) vs unique(execution) (frozen, epoch − pre, nested; per area +
   pixel); self-fit remapping; GROUP-KERNEL maps = each group's summed contribution averaged around the event per spout
   (from the basis run's per-group parts, PART_VARIANTS) — replaces page 1.20–1.21 single-beta maps and the contests
   1.33–1.36. **TE partial refit: code DONE (61743c3) and QUEUED** (`S/run_partial_te.sh`, log `S/partial_te.log`; waits
   for `done` in te_chain.log and `exit` in partial_ls7.log, then 6 workers; ~11 design builds x ~2 min per session ->
   ~6 h local, no O2). Sets target / target_spout / target_lick / execution / exec_land / contact / target_land; nulls:
   ONE block permutation of trial spouts fed to raw_blocks AND te_lick_parts (`tpos_perm`), landing weights circularly
   shifted along the landed licks (`land_roll`, 10-90 %). Smoke PS92_0819: target null changes only pos_enl / pos_cue /
   lick_onset@tgt:*, landing null only lick_onset@land:*, defaults rebuild identically (label agreement 0.11).
   Decoder / encoder signals = per-session adopted SVD x U_atlas averaged per Allen area (NOT a shared LocaNMF basis;
   Priya asked 10-26) -- registration drift cannot explain ITI-down / in-trial-recovered (same sessions, same features).
D. **Pose window TOLD 11:35 (sabatinilab-5b): it regenerates the table into kinematics_all__tmp_20261026/, diffs vs live (only daq_peak_s may differ), sends me the diff; swap = rename on M: -> needs Priya, and only after the TE partial refit ends (~16:30). stray_licks PATCHED (1d6e227, lick_uid join); re-run after the refit (memory).** Tell the pose window: `pose3d/kinematics_all/licks.csv.gz` daq_peak_s was rounded to 4 significant digits (writer fixed
   c2aa922; regenerates on its next kinematics step); `stray_licks.py` (ENCODING_FINDINGS 2.6) must be re-run with exact
   times (lick_uid → licks_all.csv).
E0. **Frozen decoder in other bases RUNNING** (Priya 10-26: LocaNMF + the pre-stroke SVD basis; `scripts/iti_basis_decoder.py`,
   log `S/iti_basis.log`): joint LocaNMF bases are per animal, fit 08-15 on 11 PRE sessions (06-06..08-14), every session
   PROJECTED from current SVTcorr; SVD basis = te_basis fit/spatial_<animal>.pkl. Decoder trained on August pre only.
   Per-animal basis is fine (decoders are per animal; no cross-animal basis needed). -> page next to 3.9.
E. Open from before: epoch-kernel drift-RATE test (8.18; within-chronic r per day vs pre rate); the narrow every-lick
   window (Priya to confirm first-lick primary); June→August drift ceiling; lesion burden / leave-PS93-out; share
   deletions / nokin retirement (Priya decides).

## Results of 10-26 (all PROVISIONAL; details ENCODING_FINDINGS 8.17–8.19, FINDINGS 4d / 4e)

- **Self-initiated ITI licks (KEY, page 3.7–3.28):** within-session direction decoding intact; FROZEN pre-stroke decoder
  down every epoch for ITI licks (-0.135 r chronic, -22 %), largely recovers for cued in-trial licks (-20 % acute → -9 %
  chronic). Each lick evokes a larger response (~2× acutely, +0.6–0.8 chronic, also isolated licks; post-stroke bouts are
  SHORTER); slow build-up before the lick weaker chronically in tjM1 / SSp-m (3/4). Not lick kinematics (post-stroke ITI
  licks slightly slower; kinematics-adjusted / range-matched responses unchanged) and not face movement (face-adjusted
  build-up still weaker). "Relative" direction for ITI = relative to the PREVIOUS trial's spout (spout retracted in ITI).
- **Lick-count control (Priya: "is this a confound")**: ITI licks are fewer post-stroke (PS92-94), but the frozen
  decoder on test sets matched to 40 licks per session gives the same numbers (ITI -0.136 / -0.115 / -0.123,
  in-trial -0.144 / -0.080 / -0.064); more licks -> slightly lower r within epoch. `scripts/iti_nmatch_check.py`
  (uses the complete v3 extraction). Not on the page yet: add one line to the 3.9 caption.
- **FaceRhythm lick factor (1.16 / 1.17):** chronic trial-by-trial unique gain in bilateral orofacial motor cortex with
  LOWER trial-to-trial variance (-40 % / -44 %) → tighter coupling, not more variability.
- **Block-permutation nulls** reproduce the single-shift partial-refit contrasts (1.33 lines).
- **Landing vs target lick kernels, lagged FR, self-fit with lick kernels** (8.19; 1.35 / 1.36 / 9.21): landing kernels do
  not absorb the chronic target term; no shift toward cortex leading the face; remapped spout information is shared
  between target and lick-locked spout terms (old design — superseded by TE).
- **Epoch kernel fits** (8.18; 1.20 / 1.21): lick kernels diverge steadily from early pre but not yet beyond drift; the
  single-lag lick-kernel maps are leftover terms (collinear with templates / kinematics) → replaced by TE group kernels.

## PITFALLS (10-26)

- Heredocs / `python -` with `\n` inside string literals STILL break (hit twice more): write patch scripts with the Write
  tool; for small fixes use the Edit tool (Edit needs a prior Read of that file in the session).
- The whole-session 3-D table rounds daq_peak_s (above); `t_peak_3d` (<= 6.9 s) is fine.
- ITI lick `position` = the trial WINDOW's label (tail = trial just finished, lead = UPCOMING trial); use the previous
  trial's spout (`prev_pos`).
- `_pix_musall_variant` must accept `novar_*` (fixed); `frozen_pixel_musall.py --layout novar` uses the real variable bases.
- `summarise_musall` rewrites `frozen_musall_r2*.csv` with only the listed variants: re-score extra variants into a
  SEPARATE tag (copy fit/templates.pkl, pre_stats.pkl, spatial_*.pkl), as `..._frA_ls_basis_novar`.
- Editing `iti_licks.py` while its chain runs breaks later steps if array shapes change (LAGS_D): wait for the chain.
- TE runs need FROZEN_LICKSPOUT / LICKLAND / FR_SPLIT / FR_LAGS unset (TE raises otherwise); frozen_encoder backups:
  `S/frozen_encoder_backup3.py` (before TE).
- Push: pre-push tests ~8–14 min; races with other sessions retry automatically (temp worktree loop).
- PAGE IMAGES: publishing index.html alone does NOT upload new / changed PNGs (v98-v102 had 9 broken figures). Always
  pass every new or regenerated image in the Artifact `files` map (root = goal_page) on the same publish.
