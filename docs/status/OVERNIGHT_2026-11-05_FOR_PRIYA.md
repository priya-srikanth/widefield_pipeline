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

