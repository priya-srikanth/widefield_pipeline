# Overnight 11-05 -> 11-06: for Priya (decisions / questions first, then findings, then what ran)

Detailed log: `docs/status/RUNNING_LIST_2026-11-01.md` (11-05 entries). This page is the short version, updated through
the night.

## A. Decisions / questions waiting for you

1. **Slide 102 (gain_bio) port: five choices** (port done, branch `port/gain-bio-20261105`, not pushed):
   (1) copy the existing 1.6 GB FaceRhythm cache into the rebuild root (else ~0.5-1 TB re-read)? rec: yes;
   (2) quiet events: DAQ or the unified 1 s LP version? rec: LP 1 s;
   (3) task window: all cues before quit, or camera working rows only? rec: working rows;
   (4) the "415 is about half the gain" comparison mixes the new corrected signal with raw 470 / 415 -- keep and check?;
   (5) rest frames use the stored rest masks built with the OLD lick detector (affects all rest analyses) -- note as a
   limitation, or rebuild the masks?
2. **FR pixel maps: pre-stroke drift** (FR_PIXEL_CALIBRATION_20261105.md). The false-positive check is fine (4.6-4.9 %
   pooled), but late-pre differs from early-pre in 10 / 14 maps (4/4 animals same sign). unique_position cue chronic:
   all 232 significant bins share the pre-drift sign, 90 are drift-significant alone -> part of that "post-stroke change"
   continues a pre trend. Options: add a drift-control panel (late-vs-early pre, or epoch vs late pre); rewrite captions
   "max-statistic per panel" (they say Bonferroni, which is wrong everywhere it appears); 250 null splits + per-map
   seeds. Which?
3. **Spont co-variation (optional)**: add `no_spont` to the pixmusall twins / frAsplit_basis frames (cheap)? A "spont alone"
   Musall row needs a small code change + two frame re-renders -- only if you want a claim about spontaneous licks.

4. **(from a8, code cleanup) Bare hand runs and the pre-stroke cut.** Should a module run by hand with no profile and no
   WIDEFIELD_PRE_FROM default to the August-only cut (instead of pooling June + August)? It changes run tags and needs
   every June block (june_august_drift, restnl_mask, map_drift_prepre, reembedding) to pass an explicit "all". My rec:
   not before the replacement pass; for now only a loud WARNING when a cohort-1 run pools June; then the full switch with
   tests. (nightly_figs already defaults to August without a profile.)

5. **PS93 gets no pre-cue engagement discriminator even at the 15-trial floor** (only 12 late no-response trials before the
   stroke; PS92 has 16, right at the floor). Accept PS93 as missing, or relax the late window for PS93 only?
6. **chronic_stability's behaviour row** reads the 1b hit-rate table, which is now the camera working period. Keep it on
   camera (consistent), or keep that row on DAQ?
7. **Local disk cleanup:** the encoder input cache on C: grows 6-10 GB/h (85 GB free at 23:30); ~17 GB of 11-04 imaging
   entries are superseded. OK to delete superseded LOCAL cache entries (derived, reproducible)? A disk watcher now
   halts the render queue if C: < 20 GB free.

8. **PI deck fluoroscopy wording** (a8): slide 161 bullet 2 says "about half the bolus"; with the new barium-amount
   measure it is closer to "about a third of the barium" (severe day 1 amount ~0.35x pre). Reword?

## B. Findings (early; nothing on pages changed)

- Mixed models, binned contest, LOG deficit (now primary): cue window -- spout identity tracks recovery (-8.1, q .015)
  and landing weakly (-3.15, q .004); lick window -- landing tracks recovery (-6.8, q < .001), spout identity does not
  (-4.5, q .55) though it is above pre post-stroke (+10.5, q .042). Log fits better than linear wherever there is a
  relation (dAIC -8 to -20). Figures: `.../imaging_encoding/<_daqlk tag>/figures/mixed_models/` (open markers, shape =
  animal, colour = epoch).

## C. What ran / is running (summary)

- Render `_daqlk`: lanes G, 2 (A B D E F Z), T, D, E, X. 11 / ~227 steps done at 22:34; per-step times 2-4x the
  estimates (network-bound at ~92 MB/s). Revised forecast: ~weekend unless the local input cache helps (measuring).

## D. Overnight updates (newest last)
- ~23:00: PI deck fluoroscopy update (your request): new all_graphs (10 panels, + G bolus density, H barium amount)
  to replace slide 159 + notes with the density / amount numbers; density_method as a HIDDEN supporting slide after 159.
  The deck is a8's (PI-deck window, has the fluoro scripts / backups) -> coordinating with a8 so only one of us edits;
  will not edit while PowerPoint has it open.
- ~23:30: **Decisions follow-up done** (branch fix/decisions-followup-20261105, wf_decisions2, unpushed): precue
  discriminator floor 15 (PS92 / PS94 / PS95 get one; PS93 not -- A5); 1b hit rate on the camera working period (DAQ =
  robustness figure; 1c stays DAQ so epoch boundaries stay pinned); iti_lick_rate + within_lick_correction cut to the working
  period; **trial table rebuilt under the new lick detector: IDENTICAL** to the 11-04 table (0 class / hit changes over
  57,048 trials) -> the O2 encoder runs need nothing.
- ~23:30: **Local input cache: not worth it mid-render** (RENDER_IO_LOCAL_CACHE_20261105.md): most heavy steps already read
  locally; the remaining network reads are ~1-2.5 h over 15+ h of queue (< 10 % gain); steps are CPU / RAM bound. Option
  for later: point the per-session result cache at a local copy (no code change). Disk risk found -> watcher started (A7).
- ~23:30: decisions push rejected twice by the same 14 pre-push tests (pass when run alone) -> likely test-order pollution
  from the new tests importing scripts; diagnosing, will fix the tests (not bypass).
- ~23:40: **PI deck fluoroscopy update DONE by a8** (192 slides / 85 hidden): slide 161 = the new 10-panel all_graphs,
  notes with density / amount + sitting / gap / transit stats; NEW hidden slide 162 "How bolus density is measured"
  (density_method). Backup backup_before_fluoro_density_1105.pptx; hidden states asserted unchanged. Wording question = A8.
