"""Per-session sidecars can name each dot's session, so positions pair within a session even with a missing cell."""
import pandas as pd
import pytest

from wfield_local import epoch_figures as ef


def test_sessions_column_written_and_parallel(tmp_path):
    q = tmp_path / "fig.png"
    points = {"acute": {"nI": [("PS94", 0.5), ("PS94", 0.6)], "fC": [("PS94", 0.4)]}}
    sessions = {"acute": {"nI": ["day1", "day2"], "fC": ["day2"]}}
    ef._write_sessions_csv(points, q, sessions)
    d = pd.read_csv(ef.sidecar(q, "_sessions.csv"))
    assert list(d.columns) == ["epoch", "position", "animal", "value", "session"]
    w = d.pivot_table(index="session", columns="position", values="value")
    assert w.loc["day2", "fC"] == 0.4 and pd.isna(w.loc["day1", "fC"])


def test_without_sessions_the_file_is_unchanged(tmp_path):
    q = tmp_path / "fig.png"
    ef._write_sessions_csv({"pre": {"nI": [("PS92", 0.9)]}}, q)
    assert ef.sidecar(q, "_sessions.csv").read_text().splitlines() == ["epoch,position,animal,value",
                                                                       "pre,nI,PS92,0.9"]


def test_mismatched_sessions_refused(tmp_path):
    with pytest.raises(ValueError):
        ef._write_sessions_csv({"pre": {"nI": [("PS92", 0.9)]}}, tmp_path / "f.png", {"pre": {"nI": []}})


def test_per_session_values_keys(monkeypatch):
    monkeypatch.setattr(ef, "group_days_by_epoch", lambda pa: {"acute": [("PS94", 3)]})
    pa = {"PS94": ([("r0",), ("r1",)], {3: ("r3",)})}
    pre = ef.per_session_values(pa, "pre", lambda an, d, r: r[0], with_session=True)
    assert pre == [("PS94", "r0", "pre0"), ("PS94", "r1", "pre1")]
    assert ef.per_session_values(pa, "acute", lambda an, d, r: r[0], with_session=True) == [("PS94", "r3", "day3")]
    assert ef.per_session_values(pa, "acute", lambda an, d, r: r[0]) == [("PS94", "r3")]
