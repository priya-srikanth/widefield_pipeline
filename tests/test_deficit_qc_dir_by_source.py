"""`run_mode.deficit_qc_dir(name)` routes by the deficit's SOURCE table (D9 queue audit 11-08): trial-attempts deficits to
the routed attempts folder, `contact` (position_coding_trend_recovery_sessions.csv) to the run's tagged encoding qc."""
from pathlib import Path

from wfield_local import mixed_models as mm
from wfield_local import run_mode


def test_routes_by_source_table(monkeypatch):
    monkeypatch.setattr(run_mode, "attempts_dir", lambda: Path("A"))
    monkeypatch.setattr(run_mode, "encoding_out_root", lambda: Path("ENC"))
    for name, (csv, *_rest) in mm.DEFICITS.items():
        want = Path("A") if csv == "trial_attempts_sessions.csv" else Path("ENC") / "qc"
        assert run_mode.deficit_qc_dir(name) == want, name
    assert run_mode.deficit_qc_dir("contact") == Path("ENC") / "qc"
    assert run_mode.deficit_qc_dir() == run_mode.deficit_qc_dir(mm.DEFICIT) == Path("A")
