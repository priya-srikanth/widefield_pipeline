"""Which components sit in territory the map analyses refuse to test, and dropping them cleanly.

The failure this file guards is the one that actually happened while measuring it: `Basis.A` carries
NaN at off-brain pixels, so the first run of the measurement reported "0 components affected" for all
four animals. **A count of NaNs and a count of zeros look identical in the output** -- nothing raised,
nothing warned, and "the glue covers no component" reads exactly like "there is no glue". The same
shape of bug is already recorded on `paint_exclusion.region_breakdown`, which returned an empty list
for the same reason.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import cd_trajectories as cdt
from wfield_local import component_exclusion as cex


class _FakeBasis:
    """Enough of `joint_locanmf.Basis` for these functions: footprints and a count."""

    def __init__(self, A):
        self.A = A
        self.ncomp = A.shape[2]
        self.basis_id = "fake"


def _footprints(H=8, W=10, spec=()):
    """One component per (row_slice, col_slice, value) in `spec`."""
    A = np.zeros((H, W, len(spec)), np.float32)
    for k, (rs, cs, v) in enumerate(spec):
        A[rs, cs, k] = v
    return A


def test_a_component_entirely_inside_the_mask_is_flagged_and_one_outside_is_not(monkeypatch):
    ex = np.zeros((8, 10), bool)
    ex[:, :5] = True                                    # left half excluded
    monkeypatch.setattr(cex, "excluded_pixels", lambda *a, **k: ex)
    b = _FakeBasis(_footprints(spec=[(slice(None), slice(0, 5), 1.0),      # all inside
                                     (slice(None), slice(5, 10), 1.0)]))   # all outside
    f = cex.mass_fraction(b, "PS99")
    assert f[0] == pytest.approx(1.0)
    assert f[1] == pytest.approx(0.0)
    assert list(cex.occluded(b, "PS99")) == [True, False]


def test_a_NaN_footprint_does_not_read_as_ZERO_MASS(monkeypatch):
    """THE BUG THIS FILE EXISTS FOR. Off-brain pixels are NaN in a real basis; reducing over them
    gives NaN, and a NaN compared against a threshold is False -- so an unguarded implementation
    reports every component as clean and the measurement silently says the glue covers nothing."""
    ex = np.zeros((8, 10), bool)
    ex[:, :5] = True
    monkeypatch.setattr(cex, "excluded_pixels", lambda *a, **k: ex)
    A = _footprints(spec=[(slice(None), slice(0, 5), 1.0)])
    A[0, 0, 0] = np.nan                                 # one NaN anywhere poisons a naive sum
    b = _FakeBasis(A)
    f = cex.mass_fraction(b, "PS99")
    assert np.isfinite(f[0]) and f[0] > 0.9
    assert bool(cex.occluded(b, "PS99")[0]) is True


def test_an_all_zero_component_counts_as_BAD_not_as_clean(monkeypatch):
    """A component with no mass at all cannot be shown to be cortex, and "cannot be shown" must not
    default to "is". It also cannot contribute to a direction, so dropping it costs nothing."""
    ex = np.zeros((8, 10), bool)
    monkeypatch.setattr(cex, "excluded_pixels", lambda *a, **k: ex)
    b = _FakeBasis(_footprints(spec=[(slice(0, 1), slice(0, 1), 0.0)]))
    assert not np.isfinite(cex.mass_fraction(b, "PS99")[0])
    assert bool(cex.occluded(b, "PS99")[0]) is True


def test_a_mask_of_the_wrong_shape_RAISES_rather_than_silently_matching_nothing(monkeypatch):
    monkeypatch.setattr(cex, "excluded_pixels", lambda *a, **k: np.zeros((4, 4), bool))
    b = _FakeBasis(_footprints())
    with pytest.raises(ValueError, match="does not match"):
        cex.mass_fraction(b, "PS99")


def test_the_threshold_is_applied_to_MASS_not_to_pixel_overlap(monkeypatch):
    """A component that touches the glue with a sliver of its footprint but carries its weight
    elsewhere is NOT occluded. Counting pixels instead of mass would flag it."""
    ex = np.zeros((8, 10), bool)
    ex[:, 0] = True
    monkeypatch.setattr(cex, "excluded_pixels", lambda *a, **k: ex)
    A = np.zeros((8, 10, 1), np.float32)
    A[:, 0, 0] = 0.01                  # wide but faint inside the mask
    A[:, 5, 0] = 10.0                  # its actual mass is outside
    assert cex.mass_fraction(_FakeBasis(A), "PS99")[0] < 0.01


def test_summarise_names_the_count_so_a_masked_run_says_that_it_is_one(monkeypatch):
    ex = np.zeros((8, 10), bool)
    ex[:, :5] = True
    monkeypatch.setattr(cex, "excluded_pixels", lambda *a, **k: ex)
    b = _FakeBasis(_footprints(spec=[(slice(None), slice(0, 5), 1.0),
                                     (slice(None), slice(5, 10), 1.0)]))
    txt = cex.summarise(b, "PS99")
    assert "1/2" in txt and "dropped" in txt


# ------------------------------------------------------------------ the drop, inside the CD fit


def test_a_dropped_component_gets_EXACTLY_zero_weight():
    """Not "small" -- zero. A projection is `w @ sig`, so anything nonzero still reads the component,
    and a robustness arm that half-includes what it claims to exclude is worse than none."""
    rng = np.random.default_rng(31)
    ncomp = 12
    tmpl = rng.normal(size=(6, ncomp))
    y = rng.integers(0, 6, 900)
    X = tmpl[y] + rng.normal(scale=0.3, size=(900, ncomp))
    drop = np.zeros(ncomp, bool)
    drop[[2, 7, 9]] = True
    dirs = cdt.fit_directions(X, y, list(range(6)), drop=drop)
    for _p, (w, _p0, _p1) in dirs.items():
        assert np.all(w[drop] == 0.0)
        assert np.abs(w[~drop]).sum() > 0


def test_dropping_a_PURE_NOISE_component_barely_changes_the_direction():
    """The sanity check on the machinery: a component carrying no position information should cost
    almost nothing when removed, whatever its variance."""
    rng = np.random.default_rng(32)
    ncomp = 12
    tmpl = rng.normal(size=(6, ncomp))
    tmpl[:, 4] = 0.0                                   # component 4 has NO position signal ...
    y = rng.integers(0, 6, 1500)
    X = tmpl[y] + rng.normal(scale=0.3, size=(1500, ncomp))
    X[:, 4] += rng.normal(scale=5.0, size=1500)        # ... but plenty of variance
    drop = np.zeros(ncomp, bool)
    drop[4] = True
    a = cdt.fit_directions(X, y, list(range(6)), stats=cdt.component_stats(X))
    b = cdt.fit_directions(X, y, list(range(6)), stats=cdt.component_stats(X), drop=drop)
    kept = [(b[q][2] - b[q][1]) / (a[q][2] - a[q][1]) for q in a]
    assert min(kept) > 0.95


def test_a_drop_mask_of_the_wrong_length_RAISES():
    rng = np.random.default_rng(33)
    y = rng.integers(0, 6, 400)
    X = rng.normal(size=(400, 10))
    with pytest.raises(ValueError, match="drop mask"):
        cdt.fit_directions(X, y, list(range(6)), drop=np.zeros(9, bool))


def test_the_masked_arm_gets_its_OWN_result_tag():
    """A masked run and an unmasked one are two analyses of the same animal and alignment; sharing a
    tag would make the second overwrite the first and remove the comparison from the dumps."""
    a = cdt.result_tag("PS95", "precue", orth=True, mask_occluded=False)
    b = cdt.result_tag("PS95", "precue", orth=True, mask_occluded=True)
    assert a != b and b.endswith("cortexonly")


def test_DROPPING_IS_THE_DEFAULT_for_the_CD(monkeypatch):
    """Priya, 2026-09-24: *"i think for the CD analyses we should drop the masked components"*.

    Pinned in two places, because a default that lives only in `argparse` is not the default for a
    caller that imports the function -- and `_render_animal` reads the item dict with a fallback,
    which is the third place it could silently revert.
    """
    import inspect

    sig = inspect.signature(cdt.analyse_animal)
    assert sig.parameters["mask_occluded"].default is True
    src = inspect.getsource(cdt._render_animal)
    assert 'item.get("mask_occluded", False)' not in src
    assert 'item.get("mask_occluded", True)' in src


def test_the_CLI_rejects_anything_but_drop_or_keep_and_defaults_to_drop(capsys):
    """`keep` must remain REACHABLE: it is the arm every figure on the share before 2026-09-24 was,
    so the comparison has to stay runnable rather than becoming archaeology."""
    import inspect

    with pytest.raises(SystemExit):
        cdt.main(["--occluded", "nonsense"])
    capsys.readouterr()
    src = inspect.getsource(cdt.main)
    assert '"--occluded", default="drop", choices=("drop", "keep")' in src


def test_a_masked_dump_is_not_served_to_an_unmasked_lookup(tmp_path):
    from tests.test_cd_geometry_figure import _result

    cdt.save_result(_result(), tmp_path, mask_occluded=True)
    assert cdt.load_result(tmp_path, "PS95", "precue", orth=True) is None
    got = cdt.load_result(tmp_path, "PS95", "precue", orth=True, mask_occluded=True)
    assert got is not None


# ------------------------------------------------------------------ the deck knows about them


def test_every_CD_figure_the_module_can_emit_HAS_A_DECK_ENTRY():
    """`deck_figure_coverage` cannot see a family the registry was never told about -- it reports
    figures the deck EXPECTS and is silent about ones it does not know exist. That is how 33 of 75
    figures ended up unreferenced. So the pairing is asserted here instead: build the filename the
    way `_render_animal` builds it, for every (layout, align) the nightly renders, and require a
    `CD_FIGURES` pattern to match it.
    """
    import fnmatch

    from wfield_local import deck_registry as reg

    tag = "_".join(["dom", "contrast", "lick", "orth", "cortexonly"])
    emitted = [f"cd_{lay}_PS95_{al}_{tag}.png"
               for lay in cdt.LAYOUTS for al in ("precue", "cue", "lick")]
    # THE SECOND GATE, pre-cue and cue only -- the lick alignment cannot take it.
    wtag = "_".join(["dom", "contrast", "lick_or_working", "orth", "cortexonly"])
    emitted += [f"cd_{lay}_PS95_{al}_{wtag}.png"
                for lay in cdt.LAYOUTS for al in ("precue", "cue")]
    emitted.append("cd_cim_geometry_lick.png")
    for name in emitted:
        assert any(fnmatch.fnmatch(name, pat) for pat, _t, _l in reg.CD_FIGURES), name


def test_no_CD_deck_pattern_is_dead():
    """The other direction: a pattern matching nothing the module emits is a slide that will never
    appear, and it would sit there looking like coverage."""
    import fnmatch

    from wfield_local import deck_registry as reg

    tag = "_".join(["dom", "contrast", "lick", "orth", "cortexonly"])
    emitted = [f"cd_{lay}_{an}_{al}_{tag}.png"
               for lay in cdt.LAYOUTS for al in ("precue", "cue", "lick")
               for an in ("PS92", "PS93", "PS94", "PS95")]
    emitted += ["cd_cim_geometry_lick.png", "cd_cim_geometry_lick_cortexonly.png",
                "cd_cim_geometry_lick_cortexonly_vs_ceiling.png"]
    # the POOLED families, which are one figure per alignment rather than per animal
    emitted += [f"cd_xanimal_{lay}_{al}_contrast_lick_cortexonly.png"
                for lay in ("perposition", "cross") for al in ("precue", "cue", "lick")]
    emitted += [f"cd_migration_{al}_contrast_{g}_cortexonly.png"
                for al in ("precue", "cue", "lick") for g in ("lick", "lick_or_working")]
    emitted += [f"cd_{lay}_PS95_{al}_dom_contrast_lick_or_working_orth_cortexonly.png"
                for lay in cdt.LAYOUTS for al in ("precue", "cue")]
    emitted += [f"cd_cim_geometry_{g}_cortexonly{sfx}.png"
                for g in ("lick", "lick_or_working") for sfx in ("", "_vs_ceiling")]
    for pat, _t, _l in reg.CD_FIGURES:
        assert any(fnmatch.fnmatch(n, pat) for n in emitted), pat


def test_the_CD_notes_carry_the_ARTEFACT_WARNING_not_just_the_method():
    """A caveat that lives only in a status document does not travel with the figure. The naive
    reading of a one-vs-rest CD panel is wrong, so every trajectory slide's note has to say so."""
    from wfield_local import deck_registry as reg

    # SCOPED TO THE TRAJECTORY PANELS. The derived families (pooled delta, migration) are checked
    # separately below for THEIR OWN caveats -- requiring the trajectory caveat verbatim on a
    # migration matrix would be cargo-culting the string rather than the reasoning.
    for pat, _ttl, legend in reg.CD_FIGURES:
        if not (pat.startswith("cd_epochs_") or pat.startswith("cd_cross_")):
            continue
        if "lick_or_working" in pat:
            # the second gate's panels must say WHAT THE GATE IS, since that is the only thing
            # distinguishing them from the success-only panels beside them on the deck
            assert "MISS-WHILE-WORKING" in legend, pat
            assert "NO LICK-ALIGNED VERSION" in legend, pat
        assert "0.289" in legend, pat                 # the measured near-cancellation
        assert "artefact" in legend.lower() or "artifact" in legend.lower(), pat
        assert "surviving fraction" in legend or "survived" in legend, pat
        assert "Z-SCORED" in legend, pat              # the standardisation, measured
        assert "HAEMODYNAMICS ARE SLOW" in legend, pat


def test_the_geometry_note_states_the_CHANCE_LEVEL_and_the_MISSING_NULL():
    """0.6 against 1.0 reads as a deficit; against K/n it is ~30x chance. And the matched null has
    not run, so the note must say the number has no scale beyond chance yet."""
    from wfield_local import deck_registry as reg

    leg = next(le for pat, _t, le in reg.CD_FIGURES if "geometry" in pat)
    assert "K/n" in leg and "chance" in leg.lower()
    assert "NO MATCHED NULL" in leg.upper()
    assert "SCALE-INVARIANT" in leg.upper()           # why row B exists at all


def test_the_DERIVED_families_carry_THEIR_OWN_caveats():
    """Each derived figure has one way it can be misread, and its note has to close that one.

    A pooled delta invites "the band includes zero so nothing happened", when pre-stroke is zero BY
    CONSTRUCTION and the difference carries both epochs' noise. A migration matrix invites reading
    off-diagonal mass as movement, when the six directions are not orthogonal and neighbouring
    positions resemble each other already. Neither caveat is the trajectory panels' caveat, so they
    are asserted separately rather than by requiring one string everywhere.
    """
    from wfield_local import deck_registry as reg

    notes = dict((pat, leg) for pat, _t, leg in reg.CD_FIGURES)
    xa = notes["cd_xanimal_perposition_*.png"]
    assert "WITHIN ANIMAL" in xa
    assert "BY CONSTRUCTION" in xa            # pre-stroke is zero by definition, not measurement
    assert "boot_delta" in xa and "sessions within each animal" in xa
    assert "2026-09-20" in xa                 # why the point estimate is animal-weighted

    mig = notes["cd_migration_*.png"]
    assert "NOT ORTHOGONAL" in mig
    assert "BOLD IS THE DIAGONAL" in mig      # Priya had to ask what bold meant
    assert "BOXED" in mig and "HATCHED" in mig
    assert "0.33-0.43" in mig                 # the measured residual correlation, not an assertion
