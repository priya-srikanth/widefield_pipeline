"""Relabel test: the cyclic shifts map every spout to every other spout exactly once."""
from wfield_local import frozen_relabel as fr


def test_shifts_cover_every_pair_once():
    order = ["close_contra", "close_ipsi", "close_mid", "far_contra", "far_ipsi", "far_mid"]
    pairs = [(q, j) for rl in fr.shifts(order) for q, j in rl.items()]
    assert len(pairs) == 30 and len(set(pairs)) == 30
    assert all(q != j for q, j in pairs)
    assert all(sorted(rl.values()) == sorted(order) for rl in fr.shifts(order))      # each shift is a permutation
