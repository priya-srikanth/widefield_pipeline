"""wfield_local.input_provenance (lane E 11-24): figures state the lick set / 3-D table their inputs were BUILT on."""
import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from wfield_local import input_provenance as ip  # noqa: E402

SIG = "fused:lp_side_r2b_prod_20261118_3net_20261121_sync60_gapfix60:a9d0dcaa2a|kinematics_fused:68514379:1795378063329284000"
SIG2 = "fused:lp_side_r2b_prod_20261118_3net_20261121_sync60_gapfix60:a9d0dcaa2a|kinematics_fused:68516098:1795413371586449000"


@pytest.fixture(autouse=True)
def _clean():
    ip.reset()
    yield
    ip.reset()


def test_describe_names_run_hash_table_size():
    d = ip.describe(SIG)
    assert "lp_side_r2b_prod_20261118_3net_20261121_sync60_gapfix60" in d and "a9d0dcaa2a" in d
    assert "kinematics_fused/licks.csv.gz" in d and "68,514,379 B" in d


def test_legacy_and_footer_counts():
    ip.record(SIG)
    ip.record(SIG)
    ip.record(None)
    assert ip.seen() == {SIG: 2, ip.LEGACY: 1}
    f = ip.footer_text()
    assert f.startswith("inputs read:") and "[2 session file(s)]" in f and "legacy" in f


def test_footer_empty_without_inputs():
    assert ip.footer_text() == ""


def test_stamp_keys_and_guard():
    k = ip.stamp_keys(SIG)
    assert k["lick_sig"] == SIG and "pose3d_run" in k
    ip.check_same_source(k["lick_sig"], SIG, "x")          # same source: accepted
    ip.check_same_source(None, SIG, "x")                   # pre-11-24 extract without a key: accepted
    with pytest.raises(RuntimeError, match="re-extract"):
        ip.check_same_source(SIG, SIG2, "x")                # another 3-D table version: refused
    with pytest.raises(RuntimeError):
        ip.check_same_source(SIG, None, "x")                # legacy session file vs a fused extract: refused


def test_stamp_adds_the_footer_to_a_figure():
    import matplotlib.pyplot as plt

    from wfield_local import trial_set_text as tst
    ip.record(SIG)
    fig = plt.figure()
    t = tst.stamp(fig, "a trial set")
    assert t is not None and "a trial set" in t.get_text() and "a9d0dcaa2a" in t.get_text()
    plt.close(fig)
