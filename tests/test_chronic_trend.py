"""The chronic-window trend test must call a flat series flat and a drifting one drifting."""
from wfield_local import chronic_trend, epochs


def _table(animal, pre_labels, post_pairs, position="far_R"):
    """Build a {label: {position: value}} table from explicit pre/post values."""
    t = {lab: {position: v} for lab, v in pre_labels}
    t.update({lab: {position: v} for lab, v in post_pairs})
    return t


def test_flat_series_plateaus(monkeypatch):
    days = [11, 14, 18, 22, 25]
    pre = [(f"X_060{i}", 1.0 + 0.01 * (i % 2)) for i in range(1, 6)]
    # flat RELATIVE TO the pre-stroke SD (~0.005 here): total drift must stay under it
    post = [(f"X_08{20 + k}", 0.900 + 0.0005 * (k % 2)) for k, _d in enumerate(days)]
    monkeypatch.setattr(epochs, "EPOCH_SPEC", {"X": {"chronic_from": 11}}, raising=False)
    monkeypatch.setattr(epochs, "_normalised_series",
                        lambda *_a, **_k: ([v for _l, v in pre],
                                           list(zip(days, [v for _l, v in post]))))
    r = chronic_trend.chronic_trend(_table("X", pre, post), "X", "far_R")
    assert r["n_chronic"] == 5
    assert r["flat"] and r["settled"] and r["plateaued"]


def test_drifting_series_is_not_flat(monkeypatch):
    days = [11, 14, 18, 22, 25]
    pre = [1.0, 1.01, 0.99, 1.0, 1.0]                       # tight baseline -> small SD
    post = [0.50, 0.60, 0.70, 0.80, 0.90]                   # large monotone climb
    monkeypatch.setattr(epochs, "EPOCH_SPEC", {"X": {"chronic_from": 11}}, raising=False)
    monkeypatch.setattr(epochs, "_normalised_series",
                        lambda *_a, **_k: (pre, list(zip(days, post))))
    r = chronic_trend.chronic_trend({}, "X", "far_R")
    assert not r["flat"], "a 0.4 climb against a ~0.007 baseline SD must not read as flat"
    assert not r["plateaued"]
    assert r["slope_per_day"] > 0
    assert r["ci_excludes_zero"], "a monotone climb should have a CI excluding zero"


def test_no_chronic_epoch_is_reported_not_guessed(monkeypatch):
    monkeypatch.setattr(epochs, "EPOCH_SPEC", {"X": {"chronic_from": None}}, raising=False)
    r = chronic_trend.chronic_trend({}, "X", "far_R")
    assert "no chronic epoch" in r["note"]


def test_too_few_chronic_sessions_is_reported(monkeypatch):
    monkeypatch.setattr(epochs, "EPOCH_SPEC", {"X": {"chronic_from": 11}}, raising=False)
    monkeypatch.setattr(epochs, "_normalised_series",
                        lambda *_a, **_k: ([1.0, 1.0, 1.0], [(11, 0.9), (14, 0.9)]))
    r = chronic_trend.chronic_trend({}, "X", "far_R")
    assert "chronic sessions" in r["note"]
