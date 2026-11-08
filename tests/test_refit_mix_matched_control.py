"""Mix-matched refit control: exact per-position counts from whole blocks; short pre sessions are skipped."""
import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rmmc", REPO / "scripts" / "refit_mix_matched_control.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def _session(per_pos=(30, 30, 30, 30, 30, 30), block=6):
    y, b, k = [], [], 0
    for q, n in enumerate(per_pos):
        for s in range(0, n, block):
            m = min(block, n - s)
            y += [q] * m
            b += [k] * m
            k += 1
    return {"y": np.array(y), "blk": np.array(b)}


def test_exact_counts_from_whole_blocks():
    P = _session()
    want = {0: 13, 1: 30, 5: 2}
    rows = M.matched_rows(P, want, np.random.default_rng(0))
    got = {q: int((P["y"][rows] == q).sum()) for q in want}
    assert got == want
    # at most one partial block per position
    for q in want:
        r = rows[P["y"][rows] == q]
        sizes = [int((P["blk"][r] == b).sum()) for b in np.unique(P["blk"][r])]
        assert sum(s < 6 for s in sizes) <= 1


def test_short_pre_session_is_skipped():
    assert M.matched_rows(_session((30, 30, 30, 30, 30, 4)), {5: 5}, np.random.default_rng(0)) is None


def test_recall_per_position():
    r = M.recall([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    assert r == {0: (2, 0.5), 1: (3, 2 / 3)}
