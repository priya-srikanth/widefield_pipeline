# For Priya to review (overnight 10-26 -> 10-27, encoding window sabatinilab-d4)

Live page: https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (version noted per item). Newest at the bottom of each
section. Everything here is PROVISIONAL until you have looked at it.

## Key results to look at
1. **Page v119: sagittal figures + new KEY row** ("Where things stand", 3.15-3.22). Lateral r is lost, sagittal r is
   not (3.15) -> the loss is direction-specific, not a measurement problem. KEY row now quotes z-scored features for
   both axes (unscaled in brackets); the figures themselves are still unscaled (3.22 links the z version).

2. **Spread-matched r** (`where_spread_matched[_z]_tests.csv`, FINDINGS 4d): the LATERAL r loss survives matching each
   session's lick distribution to pre (ITI -0.11 to -0.12, cued -0.12 -> -0.05, 4/4 every cell); sagittal ~unchanged.
   Retires the "narrower spread" caveat for the lateral loss (matching is partial in PS95 ITI: SD 0.79 of pre).

3. **As aim recovers, the ITI contralesional offset fades -- now 4/4 with the z-scored decoder**
   (`figures/iti_recovery_link_angle_z.png`, one panel per animal over post-stroke days: black = far-contra cued lick
   angle / pre, red = decoded ITI lateral offset). Time trend removed: -0.48 (PS92 -0.35, PS93 -0.44, PS94 -0.62, PS95
   -0.51; p 2e-4). Cued sagittal offset also tracks recovery (-0.57, 4/4): "read further" acutely -> "read closer" as
   aim returns. PS95 angle course: fast rise to ~0.95 by day 9, ~0.85 days 11-20, ~0.95 later (your "dip").

4. **Far-contra pattern test** (biology question about the chronic far-contra ipsi reading; FINDINGS 4d): the
   chronic far-contra pattern is the same pattern at lower gain in PS92 / PS94 / PS95 (cos 0.85-0.97, gain 0.57-0.84),
   degraded with partial drift toward close contra in PS93 -- but far ipsi loses gain as much (0.25-0.93), so the
   across-spout contrast shrinks broadly while only far contra is read ipsi. Not resolved; worth discussing.

5. **The ITI lateral r loss is not elapsed-time drift** (`figures/iti_where_drift.png`): single-pre-session decoders
   barely lose lateral r over the 8-day pre span; post sessions sit below that curve (ITI chronic -0.13, 4/4; cued
   lateral loss gone chronically; sagittal ~0).

6. **Haufe pattern maps of the lick-position decoder** (`figures/iti_decoder_patterns_lat.png`, per animal
   `iti_decoder_patterns_animals_lat.png`): pre-stroke lateral map = crossed push-pull in lateral orofacial S1/M1
   (more lesioned-side activity for contralesional licks). Preserved chronically in PS92 / PS93 / PS95 (similarity
   0.76-0.90); PS94's ITI map is genuinely different (0.19, reliability 0.99) -- the least-recovered animal remapped.
   PS93 / PS95-ITI pre maps are mostly one-signed (less clear push-pull) -- worth a look.

## Questions for you
1. Recovery ranking: by far-contra cued lick ANGLE / pre, chronic PS92 1.01, PS93 0.77, PS94 0.13, PS95 0.89. You
   rank PS95 best (fast early recovery, dip, then improvement) -- is there a measure you want used for the
   between-animal ranking (e.g. area under the recovery curve, days to criterion)?
2. Should figures 3.15-3.22 be redrawn on z-scored features too (KEY row already uses them)? ~15 min chain.
3. Drift check: the August-only pre span is 8 days, so chronic gaps (~39 days) are extrapolated. June pre sessions
   would give 2-month pre -> pre gaps -- OK to use them for THIS drift reference only (August-only stays the baseline)?
4. sabatinilab-e3 (PI deck) asked to be told when the decoder and activity-map figures are final: not told yet.

## Issues / caveats
1. The chronic cued far-contra "read ipsilesional" interpretation (scaled-down command vs changed strategy vs blurred
   pattern) is open; the "smaller decoded size" argument was WITHDRAWN (measured from the mouth on forward only).
