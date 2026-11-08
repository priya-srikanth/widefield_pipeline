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
