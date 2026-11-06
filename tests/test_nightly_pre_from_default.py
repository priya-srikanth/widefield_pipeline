"""nightly_figs defaults cohort 1 to the August-only pre baseline ONLY outside a render profile (Priya 11-05, rebuild ->
main merge): under a profile (WIDEFIELD_REBUILD_ROOT set) the profile owns WIDEFIELD_PRE_FROM, including the blocks that
deliberately unset it (june_august_drift, 12.1b, map_drift_prepre, restnl_mask)."""
from pathlib import Path

from wfield_local import nightly_figs

SRC = Path(nightly_figs.__file__).read_text(encoding="utf-8")


def test_constant_is_august():
    assert nightly_figs.IMAGING_PRE_FROM == "0801"


def test_default_is_gated_on_cohort1_and_no_profile():
    assert 'if config.cohort() == "cohort1" and not rebuild_root.active():' in SRC
    i = SRC.index("and not rebuild_root.active():")
    assert 'os.environ.setdefault("WIDEFIELD_PRE_FROM", IMAGING_PRE_FROM)' in SRC[i:i + 200]
