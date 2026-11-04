# Jaw-only trials in "working"? — checks, 2026-11-03

**Question (Priya, 11-03).** Since 26fd5b44 the epoch families' working = `trial_classes.in_working` =
contact | incomplete | jaw_only (cue -> stop, cut at the session's last camera lick), on the pre-stroke training set AND
the post-stroke scored set, with an August-only pre. Should jaw_only trials be in working (train + scored), or reported as
their own class like no_response? The four camera classes (contact, incomplete, jaw_only, none) stay either way.

Nothing in production changed. No frozen model was stored or published; every model below is an in-memory check fit
from the production primitives. Priya decides.

Everything (scripts, trial-level predictions, tables, logs):
`M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/jaw_only_checks_1103/`
- `jaw_checks.py`: joint LocaNMF basis, all arms (T1, within-class, T2, decomposition). One worker per animal x window.
- `jaw_checks_roi.py`: ROI basis, T1 / within-class / T2 (no decomposition: the endpoints are joint-basis numbers).
- `verify_endpoints.py` -> `verify_endpoints_log.txt`; `summarise.py [roi]` -> `summarise_log.txt`,
  `summarise_roi_log.txt`; tables in `tables/`, `tables_roi/`; predictions in `out/pred_<animal>_<window>[_roi].csv.gz`.

## Design

- **Same decoder as production.** Joint-basis rows from `pool_sessions` (production z), the A4 snapshot classes
  (`working_trials.row_masks` / `row_classes`), bal_em training frame (each training session centred on the equal-weight
  mean of its true class means, SD over its training rows), label-free EM scoring centre (`spout_scaling.score_centre`),
  `_pipe()`, LOSO over pre sessions. This is `grant_kit._frozen_frame`, the decoder behind the PS95 .975 -> .642 number.
  The ROI basis (the section-G pooled decoder's features, production ROI pool = August pre + post) gives the same answers
  (`summarise_roi_log.txt`), so it is listed only where it differs.
- **Both windows, equal priority.** Cue = `align="cue"`, 2 s post-cue. Precue = the production precue spec the 11-03
  precue frozen models use: `align="precue"`, the 2 s pre-cue analysis window (`decode.precue_post_s` = 2.0), trials
  with a lick-free 2 s pre-cue window (`precue_lickfree_check_s`), same pool and classes. Precue is BEFORE the cue, so a
  class difference there is state / anticipation before the movement, not the movement.
- **Reproduced first.** From one pool (June + August pre) the old cell (DAQ-lick engaged rows, all pre; post scored =
  engaged + no-lick outside the quit tail; centre on engaged rows) and the new cell (in_working, August, both sides)
  match the previous helper's production numbers (`spout_balem_production/checkA/A_{old,new}_{cue,precue}.csv`) in
  every session of every animal, both windows: n identical, max |accuracy difference| 1.1e-16
  (`verify_endpoints_log.txt`). PS95 close ipsi acute: .975 old, .642 new, as in `summarise_A_log.txt`.
- **Close mid** is excluded from every jaw / none score (the jaw cannot be judged there; no_response is the default).
- **Thin cells.** A spout with < 5 trials of a class is flagged and left out of balanced accuracy; "k" = spouts that
  count. PS92 (18 jaw-only trials) and PS93 (25) are too thin for any jaw-only arm; PS94 (96) and PS95 (261) carry it.
- jaw_only(inW) = jaw_only before the last camera lick (the in_working ones); none(<=end) likewise.

## Counts (pre-stroke August, held-out rows; cue; precue within 1-2 trials)

| animal | contact | incomplete | jaw_only(inW) | jaw cI / cC / fI / fM / fC | none(<=end) |
|---|---|---|---|---|---|
| PS92 | 2841 | 23 | 18 (5 sessions) | 3 / 3 / 4 / 0 / 8 | 33 |
| PS93 | 3404 | 269 | 25 (6) | 7 / 0 / 9 / 6 / 3 | 28 |
| PS94 | 4140 | 26 | 96 (6) | 17 / 16 / 21 / 25 / 17 | 94 |
| PS95 | 4692 | 65 | 261 (8) | 45 / 46 / 50 / 54 / 66 | 229 |

Pre-stroke incomplete is thin except PS93 (mostly far spouts). Full table with every class: `summarise_log.txt` §0.

## Test 1 — do jaw-only trials carry the spout like attempts?

**Cross-class** (decoder trained on contact+incomplete, scored per held-out class; balanced accuracy, close mid out;
chance .167; [block-bootstrap CI]):

| animal | window | contact | incomplete | jaw_only | none |
|---|---|---|---|---|---|
| PS94 | cue | .933 | .465 (k=2) | **.206** [.13, .28] | **.195** [.08, .33] |
| PS94 | precue | .700 | .435 (k=2) | **.387** [.30, .47] | **.351** [.23, .48] |
| PS95 | cue | .938 | .361 | **.286** [.22, .35] | **.249** [.17, .32] |
| PS95 | precue | .496 | .280 | **.247** [.19, .31] | **.336** [.26, .43] |
| PS93 | cue / precue | .760 / .564 | .257 / .263 | .524 / .463 (k=3, n=25) | .433 / .350 (k=2) |
| PS92 | cue / precue | .858 / .576 | thin | thin (k=1) | .219 / .383 |

Pooled jaw - none (paired `boot_delta`, per-session raw accuracy): cue +.12 [-.01, +.25], precue +.02 [-.10, +.16];
jaw - contact cue -.54 [-.71, -.32], precue -.25 [-.37, -.12]. The positive cue jaw - none is carried by the thin
animals (PS93 jaw .52 on 25 trials); in PS94 and PS95 jaw and none are within .04 of each other. ROI: same pattern
(PS94 cue .245 vs .181, PS95 .240 vs .266).

Confusion direction: jaw-only errors go to neighbouring spouts at .52-.58 against .44-.47 if errors were uniform, the
same as no-response errors (.51-.55) and well below contact errors (.88-.89 in PS94 / PS95). Jaw-only predictions are
spread over the whole 6x6 (`summarise_log.txt` §1).

**Within-class** (train AND test on one class, LOSO by session, 5 spouts, chance .20; null = 20 within-session label
permutations; control = contact+incomplete subsampled to the same n per session x spout, seeds 0-4):

| animal | window | jaw within | null 95th | c+i control | none within | null 95th | c+i control |
|---|---|---|---|---|---|---|---|
| PS94 | cue | .204 | .247 | .670 | .225 | .257 | .503 |
| PS94 | precue | .215 | .223 | .382 | .265 | .232 | .326 |
| PS95 | cue | .260 (p=.048) | .231 | .792 | .145 | .229 | .710 |
| PS95 | precue | .275 | .263 | .310 | .191 | .228 | .316 |

PS92 / PS93: too thin (k <= 3 spouts with >= 5 trials; no arm above its null). Post-stroke none (subacute + chronic,
secondary, not a substitute): PS94 .188 / .213 and PS95 .250 / .308 against controls .72 / .45 and .86 / .39.

**2x2 reading per animal (jaw_only):** PS94 cue = absent (within at null, cross at chance); PS94 precue = cross only;
PS95 cue and precue = "present, same code" by the letter of the rule, but at .26-.29 both ways, beside a count-matched
lick control of .79 (cue) — a trace, not a code. PS92 / PS93 too thin. none gets the same readings (PS94 precue none is
the one "present + same code").

## Test 2 — does adding jaw-only to training change the reference?

A = train contact+incomplete; B = + jaw_only(inW); all scored on IDENTICAL held-out contact+incomplete trials. B minus
A, overall accuracy:

| animal | cue: B own / B in A's frame / count-matched own / cm A frame | precue (same order) |
|---|---|---|
| PS92 | -.002 / -.001 / .000 / .000 | -.005 / -.005 / -.004 / -.003 |
| PS93 | +.001 / .000 / .000 / -.001 | -.002 / -.002 / -.003 / -.003 |
| PS94 | +.001 / .000 / .000 / -.002 | -.001 / +.004 / -.001 / +.001 |
| PS95 | -.009 / -.006 / -.010 / -.009 | +.003 / -.002 / -.003 / -.004 |

Pooled B own - A: cue -.002 [-.033, +.030], precue -.001 [-.021, +.020]. Per spout |B - A| <= .02 everywhere
(PS95 cue cI -.014, cM -.020). Count-matching and the frame make no difference, so the pre-stroke reference is not
moved by jaw-only training trials, in either window, on either basis (ROI: all |d| <= .009).

## Decomposition of old -> new (post-stroke, frozen accuracy)

One factor at a time, cue, close ipsi (cI), animal x epoch session means (full grid for every spout, both windows,
and alternative orders in `summarise_log.txt` §3):

| step | PS92 acute | PS93 acute | PS94 acute | PS95 acute | animal mean acute / subacute / chronic |
|---|---|---|---|---|---|
| OLD (DAQ-lick train, all pre; old scored; centre engaged) | .849 | .582 | .750 | **.975** | .789 / .866 / .953 |
| + August-only | .833 | .598 | .729 | .975 | .784 / .818 / .915 |
| + train contact+incomplete (centre c+i rows) | .800 | .549 | .597 | .833 | .695 / .804 / .919 |
| + train jaw_only too (centre in_working rows) | .773 | .584 | .600 | .642 | .650 / .812 / .926 |
| + score contact+incomplete only | .799 | .667 | .617 | .639 | .680 / .829 / .930 |
| NEW (+ score jaw_only) | .734 | .570 | .578 | **.642** | .631 / .777 / .916 |
| jaw as its own class: train c+i, score c+i, centre c+i | .829 | .629 | .619 | .832 | .727 / .824 / .924 |

**Model vs centre (§3b).** The PS95 acute drop is NOT jaw-only trials being scored at close ipsi (there is 1 such trial;
non-jaw close-ipsi trials alone read .639). It comes from what the frozen model is and which of the session's rows
its label-free EM centre is computed from. PS95 acute cue cI, model (rows) x centre rows (columns):

| model | engaged (old) | contact+incomplete | in_working (new) |
|---|---|---|---|
| DAQ-lick, all pre | .975 | .933 | .850 |
| DAQ-lick, August | .975 | .832 | .700 |
| contact+incomplete, August | – | .833 | .675 |
| in_working, August (new) | – | .723 | .642 |

- **Centre rows, all four animals:** centring the acute session on in_working rows (which add its jaw-only trials —
  PS95 acute has 107, all at far mid / far contra) lowers acute cI against centring on contact+incomplete, for every
  model: A model PS92 .829 -> .735, PS93 .629 -> .557, PS94 .619 -> .559, PS95 .832 -> .675.
- **Model, PS95 only:** at the c+i centre, B (with jaw) vs A (without) is .723 vs .833 in PS95, within .02 elsewhere.
  It is invisible on the pre-stroke reference (Test 2) and shows up only on an out-of-distribution acute session.
- **Baseline:** June+August vs August-only matters only for the DAQ-lick model (June has no camera classes): at the c+i
  centre PS95 .933 vs .832; under the old engaged centre, none.
- Far contra acute (animal mean): old .219, new .191, "c+i both sides" .187; close mid acute .635 / .762 / .738.
  Overall acute .539 / .516 / .592; pre panel .882 / .841 / .856.
- Precue: the same direction, smaller: acute cI old .559, new .526, c+i both sides .567; PS95 acute cI .500 / .500 / .513.
- Only the cue cI NEW - OLD acute is clear of zero (pooled -.158 [-.301, -.004]); every single step's interval spans 0.
  PS95 acute is ONE session.

## What it implies (not a recommendation)

- **Reading: jaw_only ~ none, i.e. non-attempts, in the two animals that can answer it (PS94, PS95), at cue and
  precue, on both bases.** Lick-trained readout of jaw-only trials is .21 / .29 (cue) against none .20 / .25 and
  contact .93 / .94; the within-class decoders find at most a trace (.20-.28, at or just above the permutation null,
  vs .67-.79 for count-matched licks at cue); errors are not toward neighbours. This matches Priya's 10-21 blind
  review (0 / 24 jaw-only trials were attempted licks). PS92 / PS93 cannot answer it (18 / 25 trials).
- At precue every non-lick class (jaw, none, incomplete) sits at .25-.39 against .50-.70 for contact trials. Before the
  cue the trials the animal will lick on already carry more spout signal: a state / anticipation difference, not the
  movement, and it is shared by jaw-only and no-response trials alike.
- **Option 1, keep jaw_only in working (current production):** training reference unchanged (Test 2), but each
  post-stroke session's EM centre includes its jaw-only rows, which are non-attempt-like and concentrated at the
  impaired far spouts acutely; that moves every spout's reading. Acute cI animal mean .631 (PS95 .642); pre panel .841
  (jaw-only trials score ~.25 there).
- **Option 2, jaw_only as its own class like no_response (train c+i, score c+i, centre c+i):** acute cI .727 (PS95
  .832), far contra acute .187, overall acute .592, pre panel .856. It is NOT the old number: .975 -> .832 in PS95 is the
  move from the DAQ-lick set (centre on engaged rows) to the camera classes and August-only, independent of jaw-only.
  The jaw-only trials would then be reported separately, as no_response is.
- An in-between option (train c+i, score in_working) is listed in the table above ("train c+i, score inw": acute cI .672);
  it keeps jaw-only in the scored set and its centre, so it inherits Option 1's centring effect.

## Limits

- n = 4, and the jaw-only question rests on two animals; PS95 acute is a single session.
- Pre-stroke incomplete trials are thin (except PS93), so "jaw vs incomplete" is weak; contact is the lick comparison.
- The within-class null has 20 permutations (p floor .048).
- Close mid jaw-only cannot exist (unjudged); jaw-only vs none at close mid is not testable.
- Joint basis is the production basis for these figures; ROI agrees on T1 / T2 / within-class, and was not run for the
  decomposition.
