# Handoff — 2026-10-19: combined encoding model (target vs executed, drift, credit rules), two-band test, O2 path

**START HERE for the encoding model / imaging x orofacial work.** Supersedes
`STATUS_2026-10-18_EVENING_V36_BASELINE_ENCODING_HANDOFF.md` (still the reference for LP run v36, the August-only baseline
rule, the orofacial deck and the 10-18 pitfalls). Reasoning and numbers: `DECISIONS.md` 2026-10-18 (night) and 2026-10-19.
Scratchpad (logs, quick scripts): `C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\`.

## 1. What we are doing and why

Priya's question: **does cortex track the TARGET (where the spout is) or the EXECUTED movement (where the tongue went),
and how does that change after the VLS stroke?** The vehicle is the Musall-style combined encoding model
(`wfield_local/combined_encoding.py`: task + instructed + uninstructed movement, ridge, cross-validated, per session) on
LP run v36, August-only pre baseline, engaged trials only.

Along the way:
- **Slow drift** made one session's fit negative (PS92_0824, R2 -0.22). Cause: an early-session hemodynamic-correction
  residual of within-session bleaching (raw channels normal; not across-day bleaching). The adopted preprocessing keeps
  drift slower than ~20 min BY DESIGN (trial-contrast analyses are blind to it); the encoding model now projects drift
  (>= 600 s) out of signal and design (FWL). Churchland lab / wfield never see this: they high-pass at 0.1 Hz, the filter
  retired 08-13.
- **Two-band correction** (a second 415 -> 470 coefficient for the slow band) fixes the artefact at the source but was NOT
  adopted: blood-flow correction identical, post-cue decode unchanged, pre-cue decode lower (26/32 sessions).
- That led to: **the slow band (>= 5 min) carries part of the pre-cue position decode** (0.41 -> 0.35 without it; the slow
  component alone decodes 0.29-0.32 in PS92 / PS93, whose position blocks are longer). Quote pre-cue numbers with that
  caveat. Smooth signals decode BELOW chance under block CV (neighbouring blocks differ) -- not neutral for the decoder.
- **Credit rules:** our movement_first SHUFFLE dR2 credits instructed with everything it shares with task (Musall left
  task RAW). Lead with the order-free DROP dR2; `--order musall` added for direct comparison.
- **Fair target vs executed:** the original base kept jaw / nose / tongue kinematics that already carry executed direction
  and distance, so executed looked redundant. Redesigned: `nodir:*` (fair for angle) and `nokin:*` (fair for angle AND
  distance: base = task timing + lick timing + running). Nose is not executed movement (posture / orienting).

## 2. Results so far

- **Main run (original base, provisional for target vs executed):** cortex tracks the target in every animal and epoch
  (lick window, unique target 0.045 / 0.035 / 0.037 / 0.049 pre / acute / subacute / chronic vs executed 0.007 / 0.003 /
  0.012 / 0.011; alone 0.056 vs 0.018 pre); acute drop in unique target in 3/4 animals (PS93 0.051 -> 0.018, back to 0.068
  chronic; PS94 the exception). Position coding rises chronically. Figures `<v36>/imaging_encoding/movement_first/figures/`
  (index in DECISIONS 10-19; per-session view `target_vs_executed_sessions_<area>.png`; per-animal maps in the scratchpad
  `PS9x_encoding_maps.png`).
- **Video pilot:** video absorbs ~80 % of position's unique share in the lick window (0.06 -> 0.015) and all of executed;
  ~a fifth of target coding survives. Pre-cue: use the residual DECODE (movement-removed ENL decode, 10-12/13), not R2.
- **PETH partitions** make sense: lick-locked response -> instructed; uninstructed not cue-locked; far-spout amplitude
  lower because licking is (fewer licks, fewer contacts, later).
- **Misdirected far-contra licks** are numerous enough for a plan-vs-execution test (acute 88-807 per animal; pre-stroke
  matched misses 300-600 per animal).

## 3. RUNNING at handoff

| job | where / log | then |
|---|---|---|
| `nokin:*` subset, 119 sessions, 4 workers, below-normal priority, started 07:08 | `encoding_nokin.log`; out `imaging_encoding/movement_first_subset-nokin/` | target vs executed per animal (fair contest), `target_vs_executed_sessions`-style figure for it |
| chain AFTER nokin (`queue_after_nokin.ps1`): nodir remainder (`--skip-done`) -> `--only trmodels` -> `--order musall --only shuf:` | `after_nokin_queue.log` | **intended to move to O2** -- stop this chain (command §6) once the O2 bundle is checked |
| O2 bundle build, all sessions, 3 workers | `o2_bundle.log`; bundle `DeepLabCut/Widefield/o2/enc_v36_20261019/` | then Priya: transfer + submit (COMMANDS.md in the bundle) |
| O2 equivalence check, PS94_0806 (bundle vs share, `--only drop:task`) | `o2_check.log` | must report max diff < 1e-4 before trusting O2 |
| single-trial smoke, PS92_0811 (`--only trmodels`) | task output (scratchpad `enc_trial_test.py`) | trial-to-trial SSp-m readout sanity |
| (other session) `litpose predict` on the GPU | not ours | -- |

## 4. To-dos (priority)

1. **O2:** check passes -> Priya transfers the bundle and submits (nodir, trmodels, musall_shuf; later main / video
   reruns); stop the local post-nokin chain; `collect` each tag.
2. **nokin result** (fair target vs executed), per animal first (rule 8), lick + cue windows, unique shares from
   `nokin:target+exec` / `nokin:kin+target+exec`; then nodir for angle alone.
3. **Single-trial readout** (trmodels): trial-to-trial dR2 per group / position; figure beside the time-resolved ones.
4. **Musall-order shuffle figures** (musall_shuf) beside drop / movement_first.
5. **Misdirected-lick decode:** lick-aligned features; decoder / templates on contacted licks; apply to acute far-contra
   misdirected licks; controls = pre-stroke far-contra misses, no-contact licks at matched angles, lick number / latency.
6. **Pre-cue caveat:** decide how to present the headline pre-cue decode given the slow-band share (e.g. report the
   >= 300 s-removed version beside it; or a per-trial-baseline variant).
7. **Optional:** lick-onset-only base (removes the contact-rate leak); PS92 subacute imaging dip with / without PS92_0824.
8. Older lists (10-18 evening §4: orofacial findings rewrite on v36, contour tightening, ...) still stand.

## 5. Pitfalls (new)

- **Shuffle dR2 under an orthogonalisation order credits shared variance to the EARLIER group** -- always state the credit
  rule; lead with the drop (order-free) dR2. Musall left task raw.
- **The base model decides what "unique" means**: kinematics in the base absorb executed direction AND distance (tongue
  protrusion / speed, jaw opening / speed); contact events carry position.
- **Slow drift**: fatal for continuous-time models with contiguous-block CV; remove it inside the model (projection), never
  by re-adding a 0.1 Hz filter.
- **Smooth signals decode BELOW chance under block CV** (neighbouring position blocks differ); a slow component is not
  neutral for the decoder.
- **Variance explained != information**: video takes shared variance; use residual decoding for movement-null questions.
- **Cohort figures are means of animal means** (PS95 acute = 1 session; PS92 subacute = 2 incl. 0824); stats are
  session-level.
- **Bash heredocs on this box** mangle `\n` / quotes in embedded Python and text (`'` inside a quoted heredoc broke one):
  write patch files with the Write tool.
- **sed replacement `\n` inserts a real newline** (broke two f-strings) -- use the Edit tool.
- **Stopping processes is denied to Claude**: give Priya command-line-matched stop commands wrapped in `@()`.
- **PowerShell `*>` logs are UTF-16** -- read with `iconv -f UTF-16LE`.
- **Low priority needs setting on workers too** (children inherit the parent's class only when spawned after it is set).

## 6. Commands

```bash
# local model runs (locanmf env); --only = model-name prefixes or 'trmodels'; outputs -> <order>[_video-..][_subset-..]/
python -m wfield_local.combined_encoding --all --jobs 4 --run lp_r4_heatmap_20261013__clean_v36 [--order musall] [--only nokin:] [--video cam1,cam4] [--skip-done]
python scripts/combined_encoding_figures.py --run lp_r4_heatmap_20261013__clean_v36
# diagnostics
python scripts/encoding_drift_check.py ; python scripts/position_block_durations.py ; python scripts/slowband_decode_check.py
python scripts/misdirected_lick_counts.py
# O2
python scripts/o2_encoding.py bundle --name enc_v36_20261019 --user ps150 --jobs 3
python scripts/o2_encoding.py check --name enc_v36_20261019 PS94_0806 --only drop:task
python scripts/o2_encoding.py collect --name enc_v36_20261019 --tag <TAG>
```
Stop the local post-nokin chain (Priya, PowerShell):
```powershell
$w = @(Get-CimInstance Win32_Process -Filter "name='powershell.exe'" | Where-Object { $_.CommandLine -match 'queue_after_nokin' }); $w | ForEach-Object { "stopping $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force }
```
