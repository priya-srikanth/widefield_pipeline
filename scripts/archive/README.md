# Archived probes — controls and nulls that shaped the analysis

**Evidence, not dead code.** Companion to
[`scripts/rest_migration/archive/`](../rest_migration/archive/README.md), which holds the
rest-baseline probes; this directory holds the ones that were never part of that migration.

Filed here 2026-09-22, under Priya's rule: *"anything that is a prior test that guided the current
analysis can be archived (as evidence of how we made these decisions), anything that is a test that
was cosmetic or did not end up contributing to the current analysis pipeline decisions can be
removed"*.

**Nothing was removed, and the reason is a measurement rather than caution.** All eleven candidates
were checked against `DECISIONS.md` and the status docs, and **every one has its result recorded
there** — including the two that are not cited by filename (`axis_drift_null` and `axis_manifold`
each have a dedicated `DECISIONS.md` section: "THE DRIFT NULL, AND THE THREE WRONG NULLS BEFORE IT"
and the Sadtler/Golub/Oby on-manifold argument). Not one turned out to be cosmetic. Deleting any of
them would leave a conclusion in `DECISIONS.md` with no working behind it.

## What is here

| module | the question it answered |
|---|---|
| `axis_composition_null.py` | can OUTCOME COMPOSITION alone rotate a position axis? The control the outcome-blind arm needs, built inside pre-stroke where there is no lesion |
| `axis_drift_null.py` | how much does a position axis move between PRE-STROKE sessions? Uses the June–August gap as a natural experiment, so a post-stroke cosine has a baseline rate to beat |
| `axis_manifold.py` | does post-stroke activity leave the pre-stroke subspace, or rearrange inside it? Different recovery prognoses (Oby 2019) |
| `orth_vs_raw.py` | does a result survive the engagement projection? Established that the projection is **not** a neutral cleanup — position axes sit at \|cos\| 0.61–0.89 to the engagement axis |
| `nolick_fraction.py` | how much of each position's post-stroke sample is placed at the cue rather than a lick? Found that at the impaired positions **the lick window IS the cue window**, so the two are not independent replications |
| `position_by_position.py` | the per position × window × block reduction the results are discussed in |
| `gate_bias_probe.py` | does the reference-restricted engagement gate leave residual satiety at the non-reference positions? |
| `basis_residual.py` | **a control that does not work at this rank, kept so it is not rebuilt** — the module says so in its own first line. Subspace overlap 0.996–0.997 makes the residual uninformative |
| `lick_bout_decoder.py` | do extra lick-bout onsets buy real readout, or just enlarge n? CV grouped by trial, so the comparison is honest rather than flattering |
| `_qc_from_standby.py` | one-off: regenerate motion-correction QC for dates whose movies had been cleaned from E: |

## What did NOT come here

`scripts/figure_label_audit.py` **stays live.** It is a TOOL, not a probe — it checks that a figure
which varies by a parameter says so on the figure, a bug class that has produced two real errors and
that no unit test can catch, because in both cases the data was right and only the label was wrong.
The same distinction `rest_migration/archive` draws for `publish_cache` and the `regen_*` pair.

## Before adding anything here

Apply the criteria in
[`../rest_migration/archive/README.md`](../rest_migration/archive/README.md) — in particular
**criterion 3: check what a module WRITES, not just who calls it.** A module nothing imports can
still render a figure the deck registers, and archiving it leaves every test green while the
nightly quietly loses the ability to regenerate a published figure. All ten above were checked
against `wfield_local/deck_registry.py`; none writes a registered figure.

## Archiving a file moves it out from under the repo's guards

`tests/test_entrypoint_after_defs.py` and `tests/test_engaged_cut_comes_from_config.py`
parametrise over `wfield_local/*.py`. Moving `basis_residual.py` and `lick_bout_decoder.py` here
dropped them from both, silently — the tests stay green because the parametrisation is simply
shorter. Same for `ruff`, which `pyproject.toml` now `force-exclude`s from both archive
directories.

That is the right outcome for frozen evidence, and it is the wrong outcome for anything still
run. **So the question to ask before moving a file here is not "does anything import it" but
"does anything still RUN it" —** if the answer is yes, the guards it leaves behind were load-
bearing, and their silence afterwards will not tell you.
