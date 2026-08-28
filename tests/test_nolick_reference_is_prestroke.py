"""The no-lick PRE-STROKE reference must be pre-stroke by construction, and its model too.

THE ARTIFACT'S OWN `kind` FIELD SAYS "pre-stroke". Two separate things were not.

1  THE MODEL. `analyse_animal` fitted `clf = _pipe().fit(XE, YE)` over every pooled session and ran
   the engaged arm's LOSO over all of them too. This module exists, in its own docstring, as "the
   pre-stroke reference for reading post-stroke failed trials" -- and the model reading them was
   being trained on them. Same class as `pooled_frozen_loso` (fixed 2026-08-26) and `ood_control`
   (fixed 2026-08-28); this is the third instance.

2  THE ARTIFACT. `nightly_figs` built it from `from_list`, which is ALL phases, then refused to
   freeze the result when post-stroke dates were present. The refusal was right, but `from_list`
   now ALWAYS contains post-stroke dates -- so the guard could only ever say no. Every night logged
   a refusal and the only pre-stroke reference in existence was the one written on 2026-08-19 that
   happened to be clean. A guard that can never pass is not a mechanism. `phase="pre"` restricts the
   dates, so the artifact matches its name, and the guard goes back to being a cheap second check.

POOLING IS UNCHANGED and still spans every session -- it is what reconciles the feature columns and
makes post-stroke rows comparable at all. Only the TRAINING rows are restricted. That is the same
distinction that made the frozen decoder's fix safe.
"""
import inspect

from wfield_local import config, nolick_decoder as N


def test_build_reference_accepts_a_phase():
    assert "phase" in inspect.signature(N.build_reference).parameters


def test_phase_pre_drops_post_stroke_dates():
    """The filter itself, without running the (very expensive) analysis."""
    src = inspect.getsource(N.build_reference)
    assert 'config.phase_labels(phase)' in src, "phase does not resolve through phase_labels"
    assert "set(dates) & keep" in src, "dates are not actually intersected with the phase"


def test_the_artifact_records_which_phase_it_is():
    """A reference whose scope is only in the calling code is the state the contamination hid in."""
    src = inspect.getsource(N.build_reference)
    assert '"phase": phase' in src, "the artifact does not record its own phase"


def test_training_is_restricted_to_prestroke():
    src = inspect.getsource(N.analyse_animal)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "_pipe().fit(XE, YE)" not in code, "still trains on every pooled session"
    assert "_pipe().fit(XE[m_pre], YE[m_pre])" in code, "the pre-stroke-only fit is missing"


def test_the_engaged_arm_is_the_prestroke_band():
    """The engaged arm IS the reference band. Mixing post-stroke sessions into it answers a
    different question -- and it is the number the other arms are compared against."""
    code = "\n".join(ln.split("#", 1)[0]
                     for ln in inspect.getsource(N.analyse_animal).splitlines())
    assert "groups=SE[m_pre]" in code, "the engaged LOSO still spans all phases"
    assert "(YE[m_pre] == c).mean()" in code, "the position-matching target still spans all phases"


def test_it_refuses_when_there_is_no_prestroke_reference_to_build():
    """Fewer than two pre-stroke sessions is not a thin reference, it is not a reference."""
    code = inspect.getsource(N.analyse_animal)
    assert "len(labs_pre) < 2" in code
    assert '"skipped"' in code


def test_the_result_says_what_it_trained_on():
    code = inspect.getsource(N.analyse_animal)
    for field in ('"training_phase"', '"pre_labels"', '"post_labels"'):
        assert field in code, f"the result does not record {field}"


def test_the_nightly_builds_the_frozen_one_with_phase_pre():
    """The guard alone could only ever refuse. The freeze has to BUILD pre-stroke."""
    from wfield_local import nightly_figs

    src = inspect.getsource(nightly_figs)
    assert 'phase="pre"' in src, "the nightly still freezes whatever the live build produced"
    assert "REFUSED_contaminated" in src, "the second check no longer renames a bad artifact"


def test_pre_stroke_dates_exist_to_build_from():
    """Pins the CONDITION: if the cohort ever had no pre-stroke sessions the tests above would pass
    vacuously while the reference could not be built at all."""
    assert len(config.phase_labels("pre")) >= 8
