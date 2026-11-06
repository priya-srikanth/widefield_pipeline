"""The GAIN_BIO chain brought into the repo (`scripts/gain_bio/`, 11-05; deck slide 102 `fig4_gain_sessions.png`):
each script imports and answers --help, carries no machine / adopted-run path in its CODE (provenance docstrings may
name the old share folder), resolves its output root and its quiet-events input to THIS run's routed folders (under
the rebuild root when one is set), never to the original decoder_mix_checks/gain_bio_1104 folder, and the figure
draws from synthetic figdata (--plot-only)."""
from __future__ import annotations

import ast
import importlib
import os
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
    assert q == _out_root() / "v2_quiet" and q.resolve().is_relative_to(rb) and tag in q.parts, q
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
                      for lab, an, e in labs for sg in ("corr", "r415_hp")])
    M = pd.DataFrame([dict(label=lab, animal=an, epoch=e, measure=ms, sig="corr", gain=rng.uniform(0.8, 1.6))
                      for lab, an, e in labs for ms in ("R5_chunks", "still_far")])
    SS = pd.DataFrame([dict(label=lab, animal=an, epoch=e, sig="corr", oro=rng.normal(.2, .1), post=rng.normal(.2, .1))
                       for lab, an, e in labs])
    TR = pd.DataFrame([dict(sig="corr", quantity="lgain", mean_slope_per10d=-0.007, p=0.73)])
    LM = {k: pd.DataFrame([dict(sig=sg, quantity=q, epoch=e, star="*" if e == "acute" else "")
                           for sg in ("corr", "r415_hp") for q in ("lgain", "ipsi_contra", "oro_post", "oro", "post")
                           for e in ep[1:]])
          for k in ("A_gain", "D_space", "E_oro_post_parts")}
    _common.save_figdata(tmp_path, fs.NAME, {"U": U, "M": M, "SS": SS, "TR": TR, "LM": LM})
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
