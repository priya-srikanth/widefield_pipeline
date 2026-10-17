"""Pins `behavior_position.best_trial_offset` -- extracted 2026-10-17 from `classify_cues_with_backup` so the
orofacial engagement gate could share it -- against a FROZEN copy of the loop it replaced. Exact equality, not a
tolerance (CLAUDE.md rule 9): the position repair feeds every dead-strobe session's labels."""
import numpy as np

from wfield_local.behavior_position import best_trial_offset


def _frozen_loop(daq, b, good):
    """Verbatim pre-extraction loop from classify_cues_with_backup (2026-10-17)."""
    N = len(daq); rng = max(20, abs(len(b) - N) + 10)
    best = (-1.0, 0, None)
    for off in range(-rng, rng + 1):                  # behavior[j] -> daq[j+off]
        out = np.full(N, -1, dtype=np.int16)
        k = np.arange(len(b)) + off; ok = (k >= 0) & (k < N)
        out[k[ok]] = b[ok]
        val = (out >= 0) & np.isin(daq, good) & np.isin(out, good)
        if val.sum() < 20:
            continue
        sc = float((daq[val] == out[val]).mean())
        if sc > best[0]:
            best = (sc, off, out)
    return best


def _blocks(rng, n):
    """Position sequence in scheduler-like blocks (runs of 3-8 at one of 6 positions)."""
    out = []
    while len(out) < n:
        out += [int(rng.integers(6))] * int(rng.integers(3, 9))
    return np.array(out[:n], dtype=np.int16)


def test_matches_frozen_loop_on_dead_bit_and_offset_cases():
    rng = np.random.default_rng(0)
    for trial in range(40):
        n = int(rng.integers(60, 400))
        true = _blocks(rng, n + 40)
        lead = int(rng.integers(0, 30))                     # log starts before / after the DAQ
        daq = true[lead:lead + n].copy()
        b = true[: n + int(rng.integers(-10, 30))].copy()
        good = list(range(6))
        if trial % 2:                                       # dead spout_bit1: two positions mis-coded
            dead = rng.choice(6, 2, replace=False)
            daq[np.isin(daq, dead)] = int(rng.integers(6))
            good = sorted(set(range(6)) - set(int(x) for x in dead))
        N = len(daq); r = max(20, abs(len(b) - N) + 10)
        new = best_trial_offset(daq, b, r, valid_pos=good, min_n=20)
        old = _frozen_loop(daq, b, good)
        assert new[0] == old[0] and new[1] == old[1]
        assert (new[2] is None and old[2] is None) or np.array_equal(new[2], old[2])


def test_recovers_known_offset():
    rng = np.random.default_rng(1)
    true = _blocks(rng, 300)
    daq, b = true[25:], true                                # log has 25 extra leading trials
    sc, off, _ = best_trial_offset(daq, b, 60)
    assert sc == 1.0 and off == -25
