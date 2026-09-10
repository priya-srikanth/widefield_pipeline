# scripts/ — kept one-off / recovery utilities

Not part of the nightly pipeline (that is `python -m wfield_local.preprocess` /
`preprocess_deck` on the imaging box and `nightly_figs` on the analysis box — see
`wfield_local/README.md`). These are occasional-use tools kept because they may be
needed again; they are not imported by the package.

- `_qc_from_standby.py` — regenerate motion-correction QC figures (with bottom example
  images) for dates whose raw movies were already cleaned off `E:`. Reads the corrected
  `.bin` **and** raw `.dat` from **M: standby**, and writes QC on **N:**. Auto-discovers
  sessions. Usage: `python scripts/_qc_from_standby.py 20260605 20260606 ...`.

- `rebuild_lick_maps.py` — re-run ONLY the lick-dependent map steps of `preprocess` for a set of
  sessions (after a change to lick detection or the ITI lick gate), instead of the whole night.

## The position-axis null and manifold controls

Three read-only analyses that back the 2026-08-23 DECISIONS.md entries on how a post-stroke coding
axis should be judged. None writes anything; each prints a table. They are kept as scripts rather
than folded into `wfield_local.position_axes` because they are CONTROLS on that module's verdicts —
run when the verdict rule changes, not nightly.

- `axis_drift_null.py` — how much a position axis moves between PRE-STROKE sessions, bucketed by
  gap (1-3 d / 4-10 d / the June-August ~60 d natural experiment). Establishes that axes are stable
  within days and drift over months, which is why the 60-day rate is the WRONG null for a 3-9 day
  pre-to-post gap.
- `axis_holdout_null.py` — **the null actually used.** Pooled-vs-held-out TWO pre-stroke sessions:
  the identical operation to the post-stroke comparison with no lesion in it. Also reports the
  effect of excluding PS95 8/13 (a known-degraded session still in the curated set).
- `axis_manifold.py` — on- vs off-manifold (Sadtler 2014 / Oby 2019): post-stroke activity variance
  on the pre-stroke PCA manifold against a cross-validated pre-stroke ceiling, and the fraction of
  each coding axis lying inside it.

## RUNNING THESE FROM A GIT WORKTREE — read this before trusting a result

`python scripts/foo.py` puts **`scripts/`** on `sys.path`, not the working directory, so
`import wfield_local` falls through to the EDITABLE INSTALL, which points at the main checkout
(`C:/Users/SabatiniLab/Github/widefield_pipeline`). From a worktree that silently runs the MAIN
copy of the package against your worktree's script — no error, just the wrong code. `python -m
wfield_local.x` does not have this problem (`-m` puts the cwd first), which is why the nightly
never hit it.

Set the path explicitly when running a script from a worktree:

    PYTHONPATH=$(pwd) python scripts/engagement_axis_balance.py

Symptom when it bites: a `TypeError` about an argument the function visibly accepts, or worse,
a result computed by an older version of a function you just edited.

### Enabling the hooks from a worktree without disturbing the other checkouts

`bash scripts/setup-hooks.sh` sets `core.hooksPath` in the SHARED config, so it turns the hooks on
for the main checkout and every worktree at once — including any other session mid-commit. To opt
one worktree in on its own:

    git config extensions.worktreeConfig true
    git config --worktree core.hooksPath .githooks

The flag only permits per-worktree overrides; it changes nothing by itself. Verify with
`cat .git/worktrees/<name>/config.worktree` — `hooksPath` should appear there and NOT in
`.git/config`.

Worth knowing before turning them on anywhere: `pre-commit` lints only STAGED files, so the ~700
pre-existing ruff errors across the package do not block you until you touch one of those files
(worst offenders: `tests/test_spout_behavior.py` 35, `wfield_local/allen_register.py` 26).
`wfield_local/locanmf_analysis_deck.py` was cleaned on 2026-08-24 for exactly this reason — its 30
errors would have blocked every deck edit. `pre-push` runs the full suite, ~25 s.

## Reproducing the preliminary-data numbers

`docs/PRELIM_DATA_VLS_STROKE.md` quotes exact per-position per-epoch values that the grant paragraph
rests on. These two scripts regenerate them. A document that quotes numbers with no committed way to
recompute them is a document nobody can check — which is the only reason they are here rather than in
a scratchpad.

- `prelim_numbers_crossnobis.py` — the GAIN vs MOVE decomposition of the crossnobis own-position
  distance, per position per epoch, plus the far-contra row that carries the "moved toward
  ipsilesional targets" claim. Row-centring is an exact split (`raw_diag = rowmean + rc_diag`); the
  script checks that identity numerically and prints the residual, so a collector change that
  invalidates the document's tables shows up as a number instead of silently.
- `prelim_numbers_frozen_vs_refit.py` — frozen / within-session-refit / paired gap per position per
  epoch, and the fraction of each position's frozen deficit that refitting recovers. The pre-stroke
  gap is the training-set-size handicap and not an effect; the fraction is suppressed where the
  deficit is too small for a ratio to mean anything.

Both read the same collectors the epoch figures use, so they cannot drift from the figures without
the figures drifting too.
