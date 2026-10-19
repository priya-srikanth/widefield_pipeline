# Handoff — 2026-10-19 evening: encoding model results (fair target vs executed, lick-matched, trial-to-trial), misdirected licks, per-spout maps

**START HERE for the encoding model / imaging x orofacial work.** Supersedes
`STATUS_2026-10-19_ENCODING_O2_HANDOFF.md` (still the reference for the O2 path, drift removal, credit rules, two-band
test, slow-band pre-cue caveat). Reasoning and numbers: `DECISIONS.md` 2026-10-19, 2026-10-19 (late morning),
2026-10-19 (afternoon). Scratchpad (logs, launchers): `C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\`.
VLS-stroke cohort PS92-PS95, lesions LEFT; LP run v36; AUGUST-ONLY pre baseline; engaged trials only.

## 1. The question and where it stands

Priya's question: does cortex track the TARGET (spout) or the EXECUTED lick, and how does that change after the stroke?
Today the fair contest ran on O2 for all 119 sessions, plus a lick-matched control, trial-to-trial variance, the
Musall-order shuffle, and a three-round misdirected-lick analysis.

**Current reading (provisional until the trial-to-trial contest is in):**
- PRE: cortex's lick-locked spatial signal follows the spout, not the tongue -- unique target .017-.030 vs unique executed
  ~0 (lick-matched), although 60-87 % of the executed predictors are independent of the target ones (not collinearity).
- POST: the executed lick gains its own representation (survives lick-matching), largest acutely / subacutely, when the
  tongue misses far-contra targets most (far-contra bias +23 -> +48..70 deg acutely). It rides on SYSTEMATIC per-spout
  structure: within-position lick-to-lick executed-angle deviation explains ~0 trial-to-trial variance (angle only;
  the full executed set on trial-to-trial is running tonight).
- CHRONIC: in PS92 / PS93 position coding genuinely strengthens (6-category ceiling x2-3, survives lick-matching) with the
  balance back toward target (target .51 / .43 of the ceiling vs executed .15 / .12); the extra signal is more
  categorical than a smooth angle / distance code. PS94: no real chronic gain once lick-matched, stays executed-weighted
  -- the animal with the least behavioural recovery. PS95 chronic (matched) pending.
- "Target" = where the spout is (context / touch / posture / plan); it does NOT isolate intention. The misdirected-lick
  test (aimed at intention) found no per-lick plan signature once confounds were matched.

## 2. Findings and decisions today (details in DECISIONS)

| topic | result | status |
|---|---|---|
| O2 bundle `enc_v36_20261019` | bundle == share (4.6e-7); O2 == desktop (1e-5, CSV rounding); ~10 min / session | adopted; full reruns on O2 |
| fair contest `nokin:*` (no kinematics) | target > executed pre 4/4; executed up subacute (+0.012) / chronic (+0.010), 4/4, both tests; target up chronic (+0.013) | `movement_first_subset-nokin/` |
| relative shares (unique / 6-category ceiling) | acute / subacute shift toward executed (3/4, 4/4); chronic back toward target in 3/4; PS94 stays executed-weighted | `tables/relative_shares.csv` |
| lick-matched nokin (`_matched`) | chronic ceiling + target rise REAL in PS92 / PS93; PS94's was more data; executed rise survives in all | 100/107 + O2 tasks collecting; PS95 chronic pending |
| target / executed divergence | 60-87 % independent; pre-stroke angle COMPRESSION at lateral spouts (geometry); acute far-contra ipsi shift, recovers chronically | DECISIONS afternoon |
| executed-angle (within-position) rise in the shuffle maps | NOT robust: drop ~0 everywhere; spread not increased; came from shuffle in 2/4 animals | `qc/exec_angle_spread.csv` |
| unique (drop) position maps | ~= label-permutation maps; acute drop, chronic rise contralesional | `position_dR2drop_maps_by_epoch.png` |
| per-spout unique maps (one-vs-rest `shufpos`) | **BUG found**: the relabelling re-permuted EVERY trial -> six identical maps; FIXED (minimal swap, b67e027); every shufpos-derived per-position number since 10-18 invalid until rerun | old maps renamed `INVALID_shufpos_bug_*`; rerun queued |
| misdirected licks | decoder fails acutely (17 contacted far-contra licks); templates: every "plan-like" lean traced to touch / reach / lick number / previous lick; matched first attempts ~0, same as pre-stroke misses -> NO plan signature | `cohort/tables/misdirected_licks/template_*.csv` |
| trial-to-trial variance (`_subset-trmodels`) | 27-42 % explained, all movement; cortex = uninstructed, SSp-m = licking; acute more (licking) | `time_resolved_dR2_drop_TRIAL.png` |
| Musall-order shuffle (`musall_subset-shuf`) | instructed .10 -> .19 acute -> .08 chronic; position .036 -> .062 chronic; uninstructed inflated (slow-variable shuffle) | tables there |
| shared group penalty | split penalties (task / lick / orofacial / uninstructed / video) give IDENTICAL R2 -> NOT the cause; lick-window drop with traces = property of the traces | split kept in code; NO rerun needed |
| video in the lick window | NOT used (Priya 10-19: video re-describes the licks); video only for the pre-cue movement-null readout | the earlier "80 % posture" reading withdrawn |

## 3. RUNNING at handoff (updated ~18:45)

| job | where / log | then |
|---|---|---|
| local lick-matched nokin (started 14:01), last 2 sessions PS95_0928 / 1001; 105/107 done (O2 + local combined) | box, `matched_local.log`; out `movement_first_subset-nokin_matched/` | rerun `combined_encoding_figures.py --order movement_first_subset-nokin_matched` at 107/107; compare with unmatched (`tables/relative_shares.csv` in both); DECISIONS line incl. PS95 chronic |
| O2 `nokin_matched` array (56867176, `-p short,medium`), trickling (fair-share ~0.07) | O2; Priya rsyncs `out/` back | `o2_encoding.py collect --name enc_v36_20261019 --tag nokin_matched` (identical to local; overwrites) -- can be cancelled once local hits 107 |
| FIXED per-spout rerun (`run_shufpos_now.ps1`, started 18:33, 4 workers, PID 26308) | box, `shufpos_now.log`; out `movement_first_subset-shufpos/` | `combined_encoding_figures.py --order movement_first_subset-shufpos` -> `figures/position_each_maps_{enl,cue,lick}.png` (readout hook committed 50fcb84); rough ETA ~3 h |
| overnight trial-to-trial nokin (`run_nokin_tr.ps1`), waits for the matched run, then runs ALONGSIDE the per-spout rerun (RAM ok: 4 + 4 workers) | box, `nokin_tr_queue.log` / `nokin_tr.log`; out `movement_first_subset-nokin_tr/` (SUFFIX `_tr`) | `combined_encoding_figures.py --order movement_first_subset-nokin_tr` -> `tables/tr_contest_0-1.5s.csv` = full executed set (angle, angle^2, reach) on TRIAL-TO-TRIAL vs all variance; ~4-5 h |
| chained per-spout launcher (`run_shufpos.ps1`, waits for nokin_tr) | box, `shufpos_queue.log` | redundant now: `--skip-done` will find every session done and exit |
| (other session) cam1 LP on O2 gpu_quad, tasks 21-137 est. start 10-20 19:40 | not ours | -- |

Bundle code on the share is b67e027 (split penalties, MATCH, NOKIN_TR, fixed shufpos); O2 alternative for the per-spout
rerun: rsync the bundle, `TAG=shufpos ONLY=shufpos-` (not needed if the local run finishes).
Stop commands (Priya, PowerShell; Claude cannot stop processes): nokin runs = python whose command line matches
`combined_encoding` AND `only nokin:`; per-spout = `combined_encoding` AND `shufpos-`; launchers = powershell matching
`run_nokin_tr` / `run_shufpos` -> `taskkill /PID <pid> /T /F`.

## 4. To-dos (priority)

1. **Trial-to-trial contest** (tonight's run): unique executed vs target on trial-to-trial variance -- Priya's test
   ("higher would be more executed tracking"). Then decide what to present as the target-vs-executed result.
2. **Finish lick-matched** (PS95 chronic) + DECISIONS line; update the relative-shares conclusions.
3. **Tracking-noise check**: measurement error in the executed angle biases every executed test DOWN (worst on trial-to-
   trial). Peak-averaged angle (a few frames around the peak) and / or the 3-D tongue estimates (POSE3D session);
   rerun the contest with the cleaner angle.
4. **Toward "intended"**: (a) pre-cue -> lick-window carry-over decode (does the pre-movement spout code persist into
   the lick window?); (b) touch-free contest (no-contact licks / early pre-touch window).
5. Per-spout maps from the FIXED shufpos rerun (`--order movement_first_subset-shufpos`); review with Priya;
   check which earlier per-position readouts used shufpos (DECISIONS 10-18 "per-position readouts") and redo them.
6. Pre-cue slow-band caveat presentation (10-19 morning handoff to-do 6); orofacial findings rewrite on v36 (10-18).

## 5. Pitfalls (new today)

- **O2 fair-share** is ~0.07 after today (470 encoding tasks + cam1 GPU training): new arrays sit in `(None)` /
  `(Priority)`. Request what is used (`--time=1:00:00 --mem=24G` for encoding), allow `-p short,medium`, use the box for
  small runs. `sshare -U -l`, `sprio -u ps150`, `squeue -u ps150 --start`.
- **Overlapping arrays write a session twice** (a session in two concurrent submits) -> trailing junk in the JSON;
  `collect` repairs it. Do not resubmit the full range while an earlier array still runs.
- **Scheduler-cancelled tasks** (CANCELLED+ after 2 s, no log): find with `sacct ... | grep -v COMPLETED`, resubmit
  those task ids.
- **Bundle build is network-bound** (1 GbE share link saturated), not CPU-bound; the model runs read their inputs once
  per session, so they barely compete with it.
- **Brain / encoding maps: ipsi = image LEFT** (left hemisphere = ipsilesional); only cam1 face video has ipsi = image right.
- **Shuffle dR2 of slow or event variables is inflated** (circular shift of running / lick-kernel events): shuffle-running
  in the lick window exceeded the full R2. Lead with drop dR2.
- **"Pre-lick" window = before VISIBLE protrusion**: includes initiation and, in bouts, the previous lick -- not a
  context control. **No-contact licks at a spout are SHORT** (reach 100-120 vs 165-210 px) and later in the trial -- not a
  touch-free control. Match first attempts on lick number and reach.
- **Within-session decoders fail where the deficit is**: acute far-contra has ~17 contacted licks per session.
- **Unique shares are bounded by independence AND by measurement noise**: tracking error in the executed angle counts as
  "independent" variance cortex cannot encode -> biases against executed.
- **Executed angle is on a compressed scale** at lateral spouts (pre-stroke bias ~+-30 deg toward the midline).
- **Do not use video in the lick window** for target / executed questions (it re-describes the licks).
- **Bash heredocs / sed mangle Python**: patch with the Edit tool (two patches failed again today).

## 6. Commands

```bash
# collect an O2 tag, then readouts (subset folders route to the contest / trial-to-trial readouts automatically)
python scripts/o2_encoding.py collect --name enc_v36_20261019 --tag <nokin|nodir|trmodels|musall_shuf|nokin_matched>
python scripts/combined_encoding_figures.py --order movement_first_subset-nokin            # + _matched, _tr, -nodir, -trmodels
python scripts/combined_encoding_figures.py --order musall_subset-shuf
python scripts/o2_encoding.py match --name enc_v36_20261019                                  # per-animal lick targets
python scripts/exec_angle_spread_check.py ; python scripts/misdirected_lick_decode.py         # spread check; templates
# local subset run into its own folder
COMBINED_ENCODING_SUFFIX=_tr python -m wfield_local.combined_encoding --jobs 4 --only nokin: --skip-done <labels>
```
O2 submit pattern (login node): `sbatch -p short,medium --array=1-119%60 --time=1:00:00 --mem=24G
--export=ALL,TAG=<tag>,ORDER=movement_first,ONLY=<prefixes>,VIDEO=,MATCH=<match.json|empty> $B/run_enc.sbatch`;
transfer node: `rsync -ah <share bundle>/ <scratch bundle>/` before, `rsync -ah <scratch>/out/ <share>/out/` after.
