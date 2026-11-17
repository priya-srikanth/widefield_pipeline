"""Priya 11-06 ("relax the late window for PS93 only"): the engagement discriminator's late window is the final quarter
(`LATE_FRACTION` 0.75) for every animal and for the DAQ gate; under the CAMERA gate PS93 alone takes the per-animal
override in configs/defaults.yaml (`precue_engagement_states.camera_late_fraction`, 0.67: no_response 12 -> 15)."""
import inspect

from wfield_local import config
from wfield_local import precue_engagement_states as pes


def test_override_is_ps93_only_and_from_config():
    over = config.defaults()["precue_engagement_states"]["camera_late_fraction"]
    assert over == {"PS93": 0.67}


def test_every_other_animal_unchanged():
    assert pes.LATE_FRACTION == 0.75
    for a in ("PS92", "PS94", "PS95", "PS118", "PS120", "PS121", "PS122"):
        assert pes.late_fraction(a, camera=True) == 0.75, a
    for a in ("PS92", "PS93", "PS94", "PS95"):
        assert pes.late_fraction(a, camera=False) == 0.75, a     # the DAQ gate never takes an override


def test_ps93_camera_window():
    assert pes.late_fraction("PS93", camera=True) == 0.67


def test_run_animal_uses_the_per_animal_fraction():
    src = inspect.getsource(pes.run_animal)
    assert "frac = late_fraction(animal, camera)" in src
    assert 'late = t["order"] > frac * span' in src
    assert "LATE_FRACTION" not in src                           # no path left on the global constant
    assert '"late_fraction": frac' in src                       # recorded in the output JSON
