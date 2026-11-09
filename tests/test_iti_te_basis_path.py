"""The ITI svdbasis readers point at the TE encoder basis, which is never a `_spont` folder.

910c408b (11-04) passed the ITI encoder tag through `frozen_encoder.design_tag`, which is right for the main encoder run
(`encoder_run()`, rendered under FROZEN_LICKS_ALL=split) but also renamed the TE basis to `..._spont_frA_te_basis`. The TE
runs are rendered with FROZEN_LICKS_ALL=none, so that folder never exists: on 11-07 every svdbasis item of
l_basis_extract failed. Run in a subprocess because `frozen_encoder` reads FROZEN_LICKS_ALL at import.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_PROBE = """
import sys
sys.path.insert(0, r"{scripts}")
import iti_licks as il
print(il.encoder_run().name)
print(il.te_basis_run().name)
"""


def _names(licks_all: str) -> list[str]:
    env = dict(os.environ, FROZEN_LICKS_ALL=licks_all, MPLBACKEND="Agg")
    out = subprocess.run([sys.executable, "-c", _PROBE.format(scripts=REPO / "scripts")], env=env, cwd=REPO,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-2:]


def test_te_basis_never_spont_under_split_design():
    main, te = _names("split")
    assert main == "frozen_encoder_gate-attempted_v7g3_t3d_spont_frA"
    assert te == "frozen_encoder_gate-attempted_v7g3_t3d_frA_te_basis"


def test_te_basis_unchanged_without_split_design():
    main, te = _names("none")
    assert main == "frozen_encoder_gate-attempted_v7g3_t3d_frA"
    assert te == "frozen_encoder_gate-attempted_v7g3_t3d_frA_te_basis"


def test_readers_use_the_te_helper():
    for f in ("iti_basis_decoder.py", "iti_decoder_patterns.py"):
        src = (REPO / "scripts" / f).read_text(encoding="utf-8")
        assert 'SPATIAL = il.te_basis_run() / "fit"' in src, f
        assert 'encoder_run("_te_basis")' not in src, f


_PROBE_SPATIAL = """
import os, sys
sys.path.insert(0, r"{scripts}")
from wfield_local import frozen_encoder as fe
fe.out_root = lambda: __import__("pathlib").Path(os.environ["PROBE_OUT_ROOT"])   # never the share
import iti_decoder_patterns as dp
import iti_basis_decoder as bd
print(dp.SPATIAL.as_posix())
print(bd.SPATIAL.as_posix())
"""


def test_te_basis_spatial_path_under_split(tmp_path):
    """11-08 (e_pat_stats_maps): under the render env (FROZEN_LICKS_ALL=split) the readers' SPATIAL is the TE run's fit/."""
    env = dict(os.environ, FROZEN_LICKS_ALL="split", MPLBACKEND="Agg", PROBE_OUT_ROOT=str(tmp_path))
    env.pop("FROZEN_DESIGN", None)
    out = subprocess.run([sys.executable, "-c", _PROBE_SPATIAL.format(scripts=REPO / "scripts")], env=env, cwd=REPO,
                         capture_output=True, text=True, check=True)
    paths = out.stdout.strip().splitlines()[-2:]
    want = (tmp_path / "frozen_encoder_gate-attempted_v7g3_t3d_frA_te_basis" / "fit").as_posix()
    for p in paths:
        assert "_spont" not in p
        assert p.endswith("frozen_encoder_gate-attempted_v7g3_t3d_frA_te_basis/fit")
        assert p == want


def test_te_tag_derived_from_te_design():
    """The TE tag comes from `tag_for_design` under the TE lick design (none), independent of the reader's env."""
    from wfield_local import frozen_encoder as fe
    assert fe.TE_LICKS_ALL == "none"
    assert fe.tag_for_design("gate-attempted_v7g3_t3d_frA", spont=False) == "gate-attempted_v7g3_t3d_frA"
    assert fe.tag_for_design("gate-attempted_v7g3_t3d_frA", spont=True) == "gate-attempted_v7g3_t3d_spont_frA"
    assert fe.design_tag("gate-attempted_v7g3_t3d_frA") == fe.tag_for_design("gate-attempted_v7g3_t3d_frA", fe.SPONT)


def test_missing_te_basis_refuses_loudly(tmp_path):
    import pytest
    sys.path.insert(0, str(REPO / "scripts"))
    import iti_licks as il
    fit = tmp_path / "frozen_encoder_gate-attempted_v7g3_t3d_frA_te_basis" / "fit"
    with pytest.raises(SystemExit) as e:
        il.te_basis_spatial("PS92", fit)
    assert "TE basis missing" in str(e.value) and str(fit) in str(e.value) and "t_frame_te_basis" in str(e.value)
    fit.mkdir(parents=True)
    (fit / "spatial_PS92.pkl").write_bytes(b"x")
    assert il.te_basis_spatial("PS92", fit) == fit / "spatial_PS92.pkl"
