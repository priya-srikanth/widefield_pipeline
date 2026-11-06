"""The GAIN_BIO chain brought into the repo (`scripts/gain_bio/`, 11-05; deck slide 102 `fig4_gain_sessions.png`):
each script imports and answers --help, carries no machine / adopted-run path in its CODE (provenance docstrings may
name the old share folder), resolves its output root and its quiet-events input to THIS run's routed folders (under
the rebuild root when one is set), never to the original decoder_mix_checks/gain_bio_1104 folder, and the figure
draws from synthetic figdata (--plot-only).

Priya's 11-06 answers, pinned exactly: (2) the quiet-events input is the LP version (v2_quiet_lp = quiet_spikes_v2 under
QS_LICK_SOURCE=daq+lp); (3) TASK windows open only at camera-working-period cues; (4) every 415 comparison is like with
like (415 HP against raw 470 HP; corrected HP against raw 470 HP as the correction check)."""
from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
GB = REPO / "scripts" / "gain_bio"
SCRIPTS = ["gain_bio", "summarise_gain_bio", "gain_stats", "fig_sessions"]
BAD = ("M:/", "C:/Users", "widefield_pipeline", "decoder_mix_checks", "cohort1_quiet_spikes/v2_quiet", "sd_rise_1104",
       "scratchpad", "hemo_meegkit")


@pytest.fixture(autouse=True)
def _path(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "cohort2_qc"))
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    monkeypatch.syspath_prepend(str(GB))
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_", "GAIN_BIO_", "QS_")):
            monkeypatch.delenv(k, raising=False)


def _code_strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(n) is not None}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


@pytest.mark.parametrize("name", SCRIPTS + ["gb_common"])
def test_no_hard_coded_paths_in_code(name):
    hits = [s for s in _code_strings(GB / f"{name}.py") if any(b in s for b in BAD)]
    assert not hits, (name, hits)


@pytest.mark.parametrize("name", SCRIPTS)
def test_imports_and_answers_help(name, capsys):
    m = importlib.import_module(name)
    with pytest.raises(SystemExit) as e:
        m.main(["--help"])
    assert e.value.code == 0
    assert "--out" in capsys.readouterr().out
    src = (GB / f"{name}.py").read_text(encoding="utf-8")
    assert "PROVENANCE" in src and "hemo_{ADOPTED}" not in src.split('"""', 2)[2]


def test_output_and_quiet_input_follow_the_rebuild_root(monkeypatch, tmp_path):
    import gb_common
    from quiet_area_spikes import _out_root

    from wfield_local import lp_licks
    monkeypatch.setenv("WIDEFIELD_REBUILD_ROOT", str(tmp_path / "rb"))
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    tag = lp_licks.output_tag()
    assert tag and tag.endswith("daqlk")
    out, q = gb_common.out_root(), gb_common.quiet_dir()
    rb = (tmp_path / "rb").resolve()
    assert out.resolve().is_relative_to(rb) and tag in out.parts and out.name == "gain_bio", out
    assert q == _out_root() / gb_common.QUIET_SUBDIR and q.resolve().is_relative_to(rb) and tag in q.parts, q
    assert not out.exists()                                     # resolving a path writes nothing
    monkeypatch.setenv("GAIN_BIO_OUT", str(tmp_path / "given"))
    monkeypatch.setenv("GAIN_BIO_QUIET_DIR", str(tmp_path / "q"))
    assert gb_common.out_root() == tmp_path / "given" and gb_common.quiet_dir() == tmp_path / "q"


def test_default_run_never_writes_to_the_original_folder():
    import gb_common
    out = gb_common.out_root().as_posix()
    assert "decoder_mix_checks" not in out and out.endswith("/gain_bio")


def test_fig4_draws_from_figdata(monkeypatch, tmp_path):
    import _common
    import fig_sessions as fs
    rng = np.random.default_rng(0)
    ep = fs.EP
    labs = [(f"{an}_{md}", an, e) for an in ("PS92", "PS93", "PS94", "PS95")
            for e, md in zip(ep, ("0810", "0819", "0901", "0925"))]
    U = pd.DataFrame([dict(label=lab, animal=an, epoch=e, set="R5", sig=sg, gain=rng.uniform(0.8, 1.6),
                           ipsi_contra=rng.normal(0, .05), oro_post=rng.normal(0, .1))
                      for lab, an, e in labs for sg in ("corr", "corr_hp", "r470_hp", "r415_hp")])
    M = pd.DataFrame([dict(label=lab, animal=an, epoch=e, measure=ms, sig="corr", gain=rng.uniform(0.8, 1.6))
                      for lab, an, e in labs for ms in ("R5_chunks", "still_far")])
    SS = pd.DataFrame([dict(label=lab, animal=an, epoch=e, sig="corr", oro=rng.normal(.2, .1), post=rng.normal(.2, .1))
                       for lab, an, e in labs])
    TR = pd.DataFrame([dict(sig="corr", quantity="lgain", mean_slope_per10d=-0.007, p=0.73)])
    LM = {k: pd.DataFrame([dict(sig=sg, quantity=q, epoch=e, star="*" if e == "acute" else "")
                           for sg in ("corr", "corr_hp", "r470_hp", "r415_hp")
                           for q in ("lgain", "ipsi_contra", "oro_post", "oro", "post") for e in ep[1:]])
          for k in ("A_gain", "D_space", "E_oro_post_parts")}
    H = pd.DataFrame(dict(epoch=ep[1:], beta_415=[.15, .18, .14], beta_470=[.3, .3, .28], ratio_415_470=[.5, .6, .5]))
    _common.save_figdata(tmp_path, fs.NAME, {"U": U, "M": M, "SS": SS, "TR": TR, "LM": LM, "H": H})
    monkeypatch.setenv("GAIN_BIO_OUT", str(tmp_path))          # main() exports --out; restored at teardown
    assert fs.main(["--out", str(tmp_path), "--plot-only"]) == 0
    assert (tmp_path / "figs" / "fig4_gain_sessions.png").exists()


def test_gain_bio_refuses_an_empty_quiet_folder(monkeypatch, tmp_path):
    """No quiet-events sessions -> SystemExit before anything is written (the original silently gave R5far = 0)."""
    import gain_bio
    (tmp_path / "q" / "sessions").mkdir(parents=True)
    monkeypatch.setenv("GAIN_BIO_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("GAIN_BIO_QUIET_DIR", str(tmp_path / "q"))
    monkeypatch.setenv("QS_FR_BASIS", "own")                    # main() sets it; restored at teardown
    with pytest.raises(SystemExit, match="no quiet-events v2 sessions"):
        gain_bio.main([])
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------------------ Priya 11-06 answers
def test_quiet_input_is_the_lp_version(monkeypatch, tmp_path):
    """(2): the default quiet folder is exactly the folder quiet_spikes_v2 writes under QS_LICK_SOURCE=daq+lp."""
    import gb_common
    monkeypatch.setenv("WIDEFIELD_REBUILD_ROOT", str(tmp_path / "rb"))
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    monkeypatch.setenv("QS_LICK_SOURCE", gb_common.QUIET_LICK_SOURCE)
    for m in ("quiet_spikes_v2", "quiet_area_spikes"):           # both bind their roots at import
        monkeypatch.delitem(sys.modules, m, raising=False)
    try:
        v2 = importlib.import_module("quiet_spikes_v2").V2
    finally:
        for m in ("quiet_spikes_v2", "quiet_area_spikes"):
            sys.modules.pop(m, None)
    assert gb_common.QUIET_LICK_SOURCE == "daq+lp" and gb_common.QUIET_SUBDIR == "v2_quiet_lp"
    assert gb_common.quiet_dir() == v2 and v2.name == "v2_quiet_lp"
    assert (tmp_path / "rb").resolve() in v2.resolve().parents


@pytest.fixture
def cue_classes(monkeypatch):
    """A synthetic session: 6 DAQ cues with fixed camera classes (trial_classes' loaders replaced)."""
    from wfield_local import trial_classes as tc
    full = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    cls = np.array(["contact", "no_response", "incomplete", "stopped", "jaw_only", "after_end"], dtype=object)
    state = {"cls": cls}
    monkeypatch.setattr(tc, "daq_cue_seconds", lambda label: full)
    monkeypatch.setattr(tc, "cue_classes_for", lambda label, cue_s, window=None: state["cls"])
    return full, state


@pytest.mark.parametrize("rows, want", [
    ("period", [1, 1, 1, 0, 1, 0]),        # camera working period: every class but the out-of-period ones
    ("working", [1, 0, 1, 0, 0, 0]),       # camera working rows (in_working): contact | incomplete
    ("all", [1, 1, 1, 1, 1, 1]),           # the 11-04 run
])
def test_task_cues_exact(cue_classes, rows, want):
    """(3): which cues open a TASK window, per rule; cue times matched within CUE_MATCH_S (here 5 ms off)."""
    import gb_common
    full, _ = cue_classes
    keep, note = gb_common.task_cues("PS92_0820", full + 0.005, rows)
    assert keep.dtype == bool and keep.tolist() == [bool(x) for x in want], (rows, keep, note)


def test_task_cues_default_is_the_working_period_and_no_video_keeps_nothing(cue_classes, monkeypatch):
    import gb_common
    full, state = cue_classes
    assert gb_common.TASK_ROWS == "period" and gb_common.task_rows() == "period"
    assert gb_common.task_cues("x", full)[0].tolist() == [True, True, True, False, True, False]
    state["cls"] = np.array(["unknown"] * 6, dtype=object)              # no behaviour video: never a DAQ fallback
    for rows in ("period", "working"):
        keep, note = gb_common.task_cues("x", full, rows)
        assert not keep.any() and "no camera classes" in note
    monkeypatch.setenv("GAIN_BIO_TASK_ROWS", "bogus")
    with pytest.raises(ValueError):
        gb_common.task_rows()


def test_task_mask_exact():
    """TASK frames = [i, i + w) from the first frame at or after each KEPT cue; a cue too close to the end drops."""
    import gain_bio
    ft = np.arange(20) * 0.1                                             # 10 Hz, 20 frames
    cue_s = np.array([0.05, 0.5, 1.0, 1.75])                             # -> frames 1, 5, 10, 18
    m, n = gain_bio.task_mask(ft, cue_s, np.array([True, False, True, True]), 3, 20)
    assert n == 2 and np.flatnonzero(m).tolist() == [1, 2, 3, 10, 11, 12]
    m, n = gain_bio.task_mask(ft, cue_s, np.zeros(4, bool), 3, 20)
    assert n == 0 and not m.any()


def test_415_compared_like_with_like():
    """(4): 415 is set against raw 470 with the same high-pass; corrected only against raw 470 (both high-passed)."""
    import fig_sessions
    import gain_stats
    import gb_common
    import summarise_gain_bio as sg
    assert (gb_common.NONCA, gb_common.CA_REF, gb_common.CORR_CHECK) == ("r415_hp", "r470_hp", ("corr_hp", "r470_hp"))
    assert gb_common.GAIN_SIGS == ("corr_hp", "r470_hp", "r415_hp")
    assert all(k.endswith("_hp") for k in gb_common.GAIN_SIGS)
    assert tuple(k for k, _ in fig_sessions.A_SIGS) == gb_common.GAIN_SIGS
    W = pd.DataFrame({"corr": [2.0], "corr_hp": [1.5], "r470_hp": [1.25], "r415_hp": [1.1]})
    R = sg.ratio_table(W)
    assert R["415/470"].iloc[0] == 1.1 / 1.25 and R["corr/470"].iloc[0] == 1.5 / 1.25
    assert "415/corr_hp" not in R.columns
    L = pd.DataFrame([dict(sig=s_, quantity="lgain", epoch=e, beta=b)
                      for s_, e, b in [("corr_hp", "acute", .4), ("r470_hp", "acute", .32), ("r415_hp", "acute", .16),
                                       ("r470_hp", "chronic", .2), ("r415_hp", "chronic", .1),
                                       ("r470_hp", "subacute", 0.0), ("r415_hp", "subacute", .05)]])
    H = gain_stats.ratio_415(L)
    assert H.epoch.tolist() == ["acute", "subacute", "chronic"]
    assert H.ratio_415_470.iloc[0] == .16 / .32 and H.ratio_415_470.iloc[2] == .1 / .2
    assert np.isnan(H.ratio_415_470.iloc[1])
