# RESTnl baseline: does varying position coverage change any map result? (2026-11-02)

Read-only sensitivity check of the RESTnl map baseline (rule `wr`: equal-weight mean of the per-position SVT medians
over positions with >= 50 lick-free rest frames, >= 4 of 6). Nothing on M: or in the RESTnl cache was modified; no
figures were made, nothing was rendered to the page, nothing committed.

- Script (untracked): `wf_restnl/scripts/restnl_position_coverage.py` (`posmaps`, `collect --arm`, `analyse`).
- Tables: `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/restnl_position_coverage_check/`:
  `headline_far_contra_acute.csv`, `ratios_by_animal_all.csv` (every arm x position x epoch x animal x baseline),
  `all_cells_vs_wr50_summary.csv`, `q1_drop_one_position_shift_sessions.csv`, `coverage_by_session.csv`, `meta.json`.
- Intermediate maps (brain pixels only) are in the session scratchpad (`restnl_cov/`), not on M:.

## VERDICT

1. **No headline conclusion changes.** Across 5 alternative baselines, the far-contra acute counts stay
   the same: pre-cue rise 4/4 (7.1, 7.5), cue loss 4/4 (7.7, 7.10) and LP lick loss 3/4 (7.18; PS95 is the
   exception). The pooled far-contra acute values move by <= .023 under every alternative except `fixC`
   (far contra dropped for ALL animals, including PS94/PS95, whose coverage is complete). `fixC` moves the
   pre-cue pooled values by -.058 (7.1) and -.078 (7.5); both stay far above 1.
2. **What the coverage gap costs is PS93's pre-cue amplitude, about ±.07.** Filling in the missing far-contra
   level raises PS93 7.1 from 1.17 to 1.24 and 7.5 from 1.24 to 1.32-1.35. Imputation and the 25-frame
   threshold agree. The cue and lick windows barely move (PS93 7.7 .52 -> .515-.53).
3. **Recommendation: keep rule `wr`** (threshold 50) as primary. Optionally state the imputed / 25-frame values
   as the sensitivity. Do **not** adopt a fixed position set: the strict intersection leaves PS93 with 3
   positions, and it moves non-headline cells by up to .23 pooled (7.5 close_L subacute). That is more
   movement than the problem it solves.
4. Cells within about ±.07 of 1 are fragile under ANY baseline-composition change (mostly PS93 subacute /
   chronic). Do not read a direction into them; none of them is a headline.

## 0. Coverage (all 107 RESTnl August sessions of PS92-95; position codes verified)

Position index = trial strobe code (`rest_by_position` groups frames by `codes`; `POSITION_NAMES`):
0 close_center, 1 close_L, 2 close_R, 3 far_center, 4 far_L, **5 far_R = far contra** (the column the amplitude
tables call far contra). Your index-5 assumption is correct.

Sessions that do not use all 6 positions under rule `wr` (frames per position in brackets, 0..5). PS94 and PS95
use all 6 in every session.

| session | frames [0..5] | dropped (< 50) | at 25 frames |
|---|---|---|---|
| PS92_0811 (pre) | 75 150 62 676 **22** 753 | far_L | still dropped |
| PS92_0812 (pre) | 162 150 105 84 **0** 229 | far_L | still dropped |
| PS92_0826 | 126 60 161 308 371 **41** | far contra | recovered |
| PS93_0807 (pre) | 225 384 353 237 312 **33** | far contra | recovered |
| PS93_0809 (pre) | 119 476 581 160 117 **31** | far contra | recovered |
| PS93_0810 (pre) | 335 569 493 234 355 **36** | far contra | recovered |
| PS93_0811 (pre) | 130 481 649 107 184 **23** | far contra | still dropped |
| PS93_0812 (pre) | 568 998 1632 **33** 391 **0** | far_center + far contra (4 used) | far_center recovered |
| PS93_0820 | 399 735 **0** 392 375 444 | close_R | still dropped |
| PS93_0828 | 753 1136 945 209 188 **16** | far contra | still dropped |

So PS93 is missing far contra in 5 of its 6 August pre sessions with RESTnl (0806 and 0814 have it; 0813 has no
RESTnl at all) and in 0828. A missing position is not a random gap: these are rests at a spout the animal kept
licking at.

## 1. Q1: how far does dropping one position move the baseline MAP? (sessions with all 6 positions)

Shift = RMS over the brain mask of `U(b_full - b_drop_i)` = `(P_i - mean of the other 5) / 6`. It is set against
the far-contra RESTnl response map RMS (contact arm, same metric as the ratio tables) in two forms:
(a) that session's own far-contra map, and (b) the animal's PRE-mean far-contra map, which is the denominator of
every ratio. The shift in absolute terms is a median of 0.6-1.2 e-3 dF/F (dropping far contra: PS92 .84, PS93 1.19,
PS94 .78, PS95 .57 e-3).

Median shift / animal pre-mean far-contra map (b):

| window | dropped | PS92 | PS93 | PS94 | PS95 |
|---|---|---|---|---|---|
| cue (7.7) | close_center | .035 | .045 | .040 | .021 |
| | close_L | .046 | .042 | .042 | .034 |
| | close_R | .037 | .049 | .043 | .031 |
| | far_center | .056 | .065 | .040 | .026 |
| | far_L | .040 | .049 | .043 | .030 |
| | **far contra** | .038 | **.073** | .048 | .029 |
| pre-cue (7.1) | close_center | .120 | .137 | .096 | .126 |
| | close_L | .158 | .128 | .101 | .210 |
| | close_R | .129 | .147 | .104 | .191 |
| | far_center | .194 | .198 | .096 | .159 |
| | far_L | .137 | .149 | .105 | .185 |
| | **far contra** | .131 | **.222** | .116 | .179 |

Against the session's own map (a): cue 2-7 %, pre-cue 10-21 %; far contra is the largest for PS93 (cue .070,
pre-cue .191). Reference amplitudes (median, e-3): cue-window far-contra map 13-22, pre-cue 5-7 (PS95 pre mean 3.2).

**Reading:** for the cue and lick windows a one-position drop is a 3-7 % perturbation of the map. For the pre-cue
(ENL) window, whose map is about 3x smaller, it is 10-22 %. Far contra is the most influential position for
PS93, and PS93 is the animal that drops it. So the pre-cue figures are the place to look, and section 2 confirms
it. A shift of this size need not change a ratio: a pre-vs-post ratio changes only when the drop is unequal
between pre and post sessions, which is exactly PS93's case (far contra missing in 5 of 6 pre sessions, only 0828
post).

## 2. Q2 / Q3: headline far-contra acute / pre ratios under alternative baselines (per animal first)

Baselines compared (each session's base map rebuilt as the mean of `U @ posmed[c]` over the set; `wr50` reproduces
the cached `base_map` to 2e-6 relative and the page's RESTnl maps to 1.5e-8 absolute):

- `wr50`: current rule `wr` (>= 50 frames, >= 4 of 6). **Reproduces RESTNL_MAPS_20261101 section 6 exactly.**
- `wr25`: same rule at a threshold of >= 25 frames (Q3).
- `fixA`: a fixed set per animal = positions with >= 50 frames in EVERY one of that animal's August sessions.
  PS92 {close_center, close_L, close_R, far_center}; PS93 {close_center, close_L, far_L}; PS94 and PS95 all 6.
- `fixB`: no far contra, in every session of the animals that ever drop it (PS92 and PS93); PS94 and PS95 stay
  on `wr50`. Any other occasional drop still follows `wr`. This is "the 5 non-far-contra positions where far
  contra is the binding constraint".
- `fixC`: no far contra for all 4 animals.
- `impute` (Q2b: simple, so done): each missing position c = this session's mean over its used set U, plus the
  offset of c from the mean over U, averaged over the nearest earlier and the nearest later session of the same
  animal that have c and all of U. The offset is taken in map space (shared Allen grid). The baseline is then
  the mean over all 6 positions. Neighbors: PS93 pre far contra from 0806 and 0814 (both pre); PS93_0828 from
  0826 and 0901; PS92_0826 from 0824 and 0828; PS92 far_L from 0810 and 0813; PS93 close_R from 0819 and 0821.

| fig | baseline | PS92 | PS93 | PS94 | PS95 | count | pooled | d pooled |
|---|---|---|---|---|---|---|---|---|
| 7.1 contact pre-cue (>1) | **wr50** | 2.00 | 1.17 | 1.17 | 1.63 | 4/4 | **1.80** | |
| | wr25 | 2.00 | 1.24 | 1.17 | 1.63 | 4/4 | 1.81 | +.013 |
| | fixA | 1.90 | 1.25 | 1.17 | 1.63 | 4/4 | 1.77 | -.023 |
| | fixB | 1.96 | 1.17 | 1.17 | 1.63 | 4/4 | 1.80 | +.001 |
| | fixC | 1.96 | 1.17 | 1.14 | 1.47 | 4/4 | 1.74 | -.058 |
| | impute | 2.02 | 1.24 | 1.17 | 1.63 | 4/4 | 1.79 | -.008 |
| 7.5 ENL 1 s pre-cue (>1) | **wr50** | 2.26 | 1.24 | 1.13 | 1.50 | 4/4 | **2.04** | |
| | wr25 | 2.26 | 1.32 | 1.13 | 1.50 | 4/4 | 2.06 | +.023 |
| | fixA | 2.18 | 1.33 | 1.13 | 1.50 | 4/4 | 2.02 | -.017 |
| | fixB | 2.25 | 1.22 | 1.13 | 1.50 | 4/4 | 2.04 | -.006 |
| | fixC | 2.25 | 1.22 | 1.08 | 1.36 | 4/4 | 1.96 | -.078 |
| | impute | 2.30 | 1.35 | 1.13 | 1.50 | 4/4 | 2.06 | +.016 |
| 7.7 contact cue (<1) | **wr50** | .59 | .52 | .60 | .43 | 4/4 | **.53** | |
| | wr25 | .59 | .52 | .60 | .43 | 4/4 | .53 | .000 |
| | fixA | .58 | .55 | .60 | .43 | 4/4 | .53 | +.003 |
| | fixB | .59 | .54 | .60 | .43 | 4/4 | .53 | +.005 |
| | fixC | .59 | .54 | .59 | .42 | 4/4 | .52 | .000 |
| | impute | .59 | .52 | .60 | .43 | 4/4 | .52 | -.002 |
| 7.10 LP cue (<1) | **wr50** | .56 | .53 | .60 | .43 | 4/4 | **.52** | |
| | wr25 | .56 | .53 | .60 | .43 | 4/4 | .52 | .000 |
| | fixA | .55 | .55 | .60 | .43 | 4/4 | .53 | +.003 |
| | fixB | .56 | .55 | .60 | .43 | 4/4 | .53 | +.005 |
| | fixC | .56 | .55 | .59 | .42 | 4/4 | .52 | .000 |
| | impute | .56 | .52 | .60 | .43 | 4/4 | .52 | -.003 |
| 7.18 LP lick (<1) | **wr50** | .50 | .55 | .82 | 1.07 | 3/4 | **.72** | |
| | wr25 | .50 | .55 | .82 | 1.07 | 3/4 | .72 | .000 |
| | fixA | .49 | .57 | .82 | 1.07 | 3/4 | .73 | +.004 |
| | fixB | .50 | .57 | .82 | 1.07 | 3/4 | .73 | +.005 |
| | fixC | .50 | .57 | .81 | 1.06 | 3/4 | .72 | -.001 |
| | impute | .50 | .54 | .82 | 1.07 | 3/4 | .72 | -.002 |

(Unrounded values: `headline_far_contra_acute.csv`.)

**No 4/4 or 3/4 count changes, and no pooled headline value moves by more than .05** except under `fixC`, which
perturbs animals with complete coverage. The pre-cue rise of PS94, the closest per-animal call (1.13-1.17), is
untouched by every coverage fix; it is moved only by `fixC` (1.08-1.14, still > 1).

### Beyond the headline (all 6 positions x 3 post epochs x 5 figures = 90 pooled cells, 360 per-animal cells)

| baseline | max abs d pooled (cell) | pooled cells abs d > .05 | per-animal cells crossing 1 |
|---|---|---|---|
| wr25 | .070 (7.5 close_L chronic) | 3 / 90 (all 7.5 close_L) | 3 (all within .05 of 1) |
| fixA | .232 (7.5 close_L subacute) | 13 / 90 | 4 |
| fixB | .098 (7.5 close_L subacute) | 7 / 90 (7.1 / 7.5 subacute) | 8 (PS93 subacute) |
| fixC | .105 (7.5 close_L acute) | 5 / 90 | 13 |
| **impute** | **.035** (7.5 close_L chronic) | **0 / 90** | 7 (all PS93, all within .07 of 1) |

The cells that cross 1 are PS93 subacute / chronic at |ratio - 1| <= .07 (for example 7.7 far contra subacute
.95 -> 1.05 under fixB / fixC). `fixB` and `fixC` remove far contra from EVERY PS93 session, not only the deficient
ones, so they change the composition of the baseline itself. That is a different baseline rather than a
coverage repair, and the cells it moves are the near-1 subacute ones. Imputation, which repairs only the missing
sessions, keeps every pooled cell within .035.

Far contra, subacute / chronic (cue, 7.7): PS93 subacute .95 (wr50) / .92 (impute) / 1.05 (fixB); chronic 1.20 /
1.16 / 1.23. The PS94 chronic RESTnl drop (.71, flagged in RESTNL_MAPS section 4 as a flat-baseline offset) is
unchanged by every coverage variant except fixC (.74). It is not a coverage effect.

### Q3: is the 50-frame threshold the binding issue?

Only partly. At 25 frames, PS93 0807 / 0809 / 0810 and PS92_0826 recover far contra (31-41 frames), and PS93_0812
recovers far_center. PS93_0811 (23), 0828 (16) and 0812 (0) keep no far contra, and PS92 0811/0812 far_L (22 / 0)
and PS93_0820 close_R (0) stay missing. Sessions with 0 frames cannot be fixed by any threshold. The 25-frame
result agrees with the imputation on the cell that matters (PS93 7.1 1.238 vs 1.239; 7.5 1.317 vs 1.345). Every
headline count and pooled value is unchanged at 25 (pooled moves <= .023). A median over 25-49 frames is noisier,
so lowering the threshold trades a composition bias for noise; it is not clearly better than imputing.

## 3. Recommendation

**Keep rule `wr` at 50 frames as the primary RESTnl baseline.** No headline result changes under any coverage
repair, and the pooled values move <= .023.

Add one caveat sentence to the RESTnl captions / write-up: "PS93 lacks lick-free far-contra rest in 5 of 6 August
pre sessions, so its baseline there averages 5 positions; imputing the missing level (or a 25-frame threshold)
raises PS93's acute pre-cue far-contra ratio from 1.17 to 1.24 (7.1) and from 1.24 to 1.32-1.35 (7.5); cue / lick
values change by <= .03, and no count changes."

If a single alternative is wanted as a sensitivity column, use **imputation**. It is the only variant that repairs
just the deficient sessions, keeps all 90 pooled cells within .035, and agrees with the independent 25-frame
check. Do not use a fixed position set: the intersection is 3 positions for PS93, and both fixed variants move
unrelated near-1 cells more than the coverage gap does.

## Method notes / provenance

- Per-position maps `U_atlas @ posmed[c]` per session from the RESTnl cache npz (positions with 0 frames have no
  median). Rebuilding `wr50` from them reproduces the cached `base_map` (2e-6 relative). Raw per-position maps
  came from `position_reference_maps.maps_by_epoch` on the warm `prm` caches. The env was that of
  `restnl_maps.setup_env` (August-only pre, legacy digest pin, WIDEFIELD_MAP_JOBS = 4); the arms were run
  sequentially. PS93_0813 has no RESTnl and is excluded, as on the page.
- Ratio = RMS(animal epoch-mean map) / RMS(animal pre-mean map), brain mask; pooled = `within_animal_pooled`
  (each animal normalized by its own pre RMS, intersection of animals). This is the same definition as
  `restnl_maps.amplitude_table` / the 15r line.
- Run time about 1 h 15 min (posmaps about 5 min with 4 workers; map collection about 1 h for 5 aligns).
