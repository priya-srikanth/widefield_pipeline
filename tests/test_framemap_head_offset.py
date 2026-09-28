"""Every frame-map -> DAQ-time conversion must go through `frame_samples_from_map`, which honours
`dat_head_offset`. Ten call sites indexed `pco[original_frame_index_ch0 + offset]` by hand (2026-09-28)
and would have placed PS92_0922 2.466 s late after its frame map had been fixed."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from wfield_local.framemap_event_maps import _corrected_frame_samples, frame_samples_from_map

REPO = Path(__file__).resolve().parents[1]


class _FM(dict):
    @property
    def files(self):
        return list(self.keys())


def test_helper_subtracts_the_head_and_clips():
    pco = np.arange(0, 2000, 10)                                      # pulse k at sample 10k
    fm = _FM(original_frame_index_ch0=np.array([154, 156, 158]), dat_head_offset=np.int64(154))
    assert frame_samples_from_map(fm, pco, 0).tolist() == [0, 20, 40]
    assert frame_samples_from_map(fm, pco, 1).tolist() == [10, 30, 50]
    old = _FM(original_frame_index_ch0=np.array([0, 2, 4]))         # pre-2026-09-28 map: head implied 0
    assert frame_samples_from_map(old, pco, 0).tolist() == [0, 20, 40]
    huge = _FM(original_frame_index_ch0=np.array([10_000]))
    assert frame_samples_from_map(huge, pco, 0).tolist() == [1990], "clipped, as every hand-rolled site did"


def test_corrected_frame_samples_delegates(tmp_path):
    pco = np.arange(0, 2000, 10)
    p = tmp_path / "m.npz"
    np.savez(p, original_frame_index_ch0=np.array([154, 156]), dat_head_offset=np.int64(154))
    assert _corrected_frame_samples(p, pco, 0).tolist() == [0, 20]


def test_no_module_indexes_the_frame_map_into_pco_by_hand():
    pat = re.compile(r"pco\w*\[\s*np\.clip\(\s*\w+\[\s*[\"']original_frame_index_ch0|pco\w*\[\s*\w+\[\s*[\"']original_frame_index_ch0")
    offenders = []
    for f in list((REPO / "wfield_local").glob("*.py")) + list((REPO / "scripts").rglob("*.py")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(txt.splitlines(), 1):
            code = ("return" in line or "=" in line) and not line.strip().startswith("#")   # a docstring mention is not a call
            if code and pat.search(line) and "frame_samples_from_map" not in line:
                offenders.append(f"{f.relative_to(REPO)}:{i}")
    assert not offenders, "index the frame map through frame_samples_from_map: " + ", ".join(offenders)
