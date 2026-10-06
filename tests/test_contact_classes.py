"""contact_classes: a contact inside a lick window is a lick; outside, it is no_tongue / tongue_other by the
tongue's visibility; a LONG touch without the tongue is grooming, but a long touch during licking (water bridge,
tongue out) is not (Priya, 2026-10-06)."""
from __future__ import annotations

import numpy as np

from wfield_local import config
from wfield_local import contact_classes as CC


def _p():
    return dict(CC.DEFAULTS)


def test_classes_and_grooming(monkeypatch):
    monkeypatch.setattr(config, "defaults", lambda session=None: {})
    vt = np.arange(0, 20, 0.004)
    out = np.zeros(len(vt), bool)
    out[(vt > 1.0) & (vt < 1.2)] = True           # lick 1
    out[(vt > 5.0) & (vt < 5.6)] = True           # licking through a long water-bridge touch
    contacts = np.array([1.1, 3.0, 5.05, 9.0, 1.25])
    dur = np.array([60.0, 80.0, 400.0, 350.0, 50.0])
    t = CC.classify(contacts, dur, np.array([1.0, 5.0]), np.array([1.2, 5.6]), vt, out, _p())
    assert list(t["class"]) == ["lick", "no_tongue", "lick", "no_tongue", "no_tongue"]   # 1.25 s: past the window + pad
    assert list(t.grooming_touch) == [False, False, False, True, False]   # 5.05 s: long but tongue out -> spared
    g = CC.grooming_periods(t, _p())
    assert len(g) == 1 and abs(g[0][0] - 4.0) < 1e-9 and abs(g[0][1] - (9.35 + 5.0)) < 1e-9


def test_contact_durations_from_the_lick_line():
    fs = 1000.0
    v = np.full(3000, 4.0)
    v[1000:1067] = 0.2                             # a 67 ms touch
    d = CC.contact_durations_ms(v, fs, np.array([1.0]), thresh_upper=2.5)
    assert abs(d[0] - 67.0) < 1.5


def test_in_spans_handles_unsorted_and_overlapping_spans():
    assert list(CC.in_spans(np.array([0.5, 2.5, 4.0]), np.array([2.0, 0.0]), np.array([3.0, 1.0]))) == [True, True, False]
