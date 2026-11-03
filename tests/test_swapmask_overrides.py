"""The 2026-11-02 cohort-1 swap-mask re-run added env overrides so a sensitivity run on another hemo variant reads its own
inputs and writes its own folders. With the variables UNSET every default must be exactly what it was."""
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENV = ("WIDEFIELD_HEMO_VARIANT", "QS_OUT_ROOT", "RESTNL_OUT_ROOT", "FROZEN_OUT_ROOT")


@pytest.fixture
def clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_encoder_inputs_default_is_adopted(clean_env):
    from wfield_local import combined_encoding as ce
    from wfield_local.hemo_variants import ADOPTED
    assert ADOPTED == "meegkit_hpfit"
    assert ce.hemo_dir() == "hemo_meegkit_hpfit"


def test_encoder_inputs_override(clean_env):
    from wfield_local import combined_encoding as ce
    clean_env.setenv("WIDEFIELD_HEMO_VARIANT", "meegkit_hpfit_lag_refitT_swapmask")
    assert ce.hemo_dir() == "hemo_meegkit_hpfit_lag_refitT_swapmask"
    clean_env.setenv("WIDEFIELD_HEMO_VARIANT", "zerophase")
    with pytest.raises(ValueError):
        ce.hemo_dir()


def _load_script(name, subdir):
    sys.path.insert(0, str(REPO / subdir))
    try:
        if name in sys.modules:
            return importlib.reload(sys.modules[name])
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(REPO / subdir))


def test_quiet_out_default_and_override(clean_env):
    m = _load_script("quiet_area_spikes", "scripts/cohort2_qc")
    assert m.OUT == Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort1_quiet_spikes")
    clean_env.setenv("QS_OUT_ROOT", "C:/tmp/qs")
    m = _load_script("quiet_area_spikes", "scripts/cohort2_qc")
    assert m.OUT == Path("C:/tmp/qs")
    clean_env.delenv("QS_OUT_ROOT")
    _load_script("quiet_area_spikes", "scripts/cohort2_qc")       # leave the module in its default state


def test_restnl_out_default_and_override(clean_env):
    from wfield_local import lp_licks
    from wfield_local.paths import PathResolver
    m = _load_script("restnl_maps", "scripts")
    want = lp_licks.output_dir(Path(PathResolver().root("cohort_out")) / "grant_figures" / "epoch_restnl")
    assert m.out_dir() == want
    clean_env.setenv("RESTNL_OUT_ROOT", "C:/tmp/rn/epoch_restnl")
    assert m.out_dir() == lp_licks.output_dir(Path("C:/tmp/rn/epoch_restnl"))


def test_frozen_encoder_root_default_and_override(clean_env):
    from wfield_local import frozen_encoder as fe
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    assert fe.out_root() == oo.root(PathResolver()) / oc.RUN / "imaging_encoding"
    clean_env.setenv("FROZEN_OUT_ROOT", "C:/tmp/fe")
    assert fe.out_root() == Path("C:/tmp/fe")
