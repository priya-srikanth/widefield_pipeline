"""Priya 11-06, A27 + A30 (fix/priya1106-cam4-style). Synthetic inputs only, no share reads.

  A27 the cam4 orofacial cohort (`orofacial_cohort`, `orofacial_jaw_lateral` and every reader of their tables) runs on
      the CAMERA WORKING PERIOD (`trial_classes.working_period_ids`, one definition); WFIELD_KIN_GATE=legacy keeps the
      pre-11-06 behavior engaged gate (`engagement_keep`, unchanged); the folders carry the gate name
      (prepost_working / prepost_legacygate / prepost_alltrials), so old and new tables never mix.
  A30 pose3d_summary_fig session dots = OPEN markers, shape = animal, color = epoch; pose3d_phase hides the top spine
      on every 2-D panel.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ("pose3d_summary_fig", "pose3d_phase")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Clean run-mode env; scripts importable by name; restore sys.path / os.environ and drop ONLY the script modules."""
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_", "OROFACIAL_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    yield
    for m in SCRIPTS:
        sys.modules.pop(m, None)


from wfield_local import orofacial_cohort as oc  # noqa: E402
from wfield_local import trial_classes as tc  # noqa: E402

CLS = np.array(["contact", "contact", "incomplete", "jaw_only", "no_response", "stopped", "after_end", "unknown"],
               dtype=object)


@pytest.fixture
def session(monkeypatch):
    """PS92_0810 with the classes above (working period = trials 1-5); PS93_0813 with no video (all unknown)."""
    full = np.arange(len(CLS)) * 10.0
    T = pd.DataFrame({"trial_id": np.arange(len(CLS)) + 1, "position": ["far_R"] * len(CLS), "cue_daq": full + 0.004,
                      "cls": ["raw"] * len(CLS)})
    monkeypatch.setattr(tc, "session_trials", lambda lab, *a, **k: T if lab.startswith("PS92") else None)
    monkeypatch.setattr(tc, "daq_cue_seconds", lambda lab: full)
    monkeypatch.setattr(tc, "cue_classes_for", lambda lab, cs, w=None: CLS.copy() if lab.startswith("PS92")
                        else np.array(["unknown"] * len(cs), dtype=object))
    monkeypatch.setattr(oc, "ENGAGED_ONLY", True)
    monkeypatch.setattr(oc, "BASELINE", "late")
    return T


# ------------------------------------------------------------------------------------------------ A27 gate
def test_default_gate_is_the_camera_working_period(session, monkeypatch, tmp_path):
    monkeypatch.setattr(oc, "engagement_keep", lambda *a, **k: pytest.fail("behavior gate used under the default"))
    assert oc.cohort_gate() == "working"
    keep, lab = oc.cohort_keep("PS92", "20260810", tmp_path)
    assert keep == {1, 2, 3, 4, 5} == tc.working_period_ids("PS92_0810")       # one definition (rule 9)
    assert lab.startswith("working")
    keep, lab = oc.cohort_keep("PS93", "20260813", tmp_path)                    # no camera classes: contributes nothing
    assert keep == set() and lab.startswith("none")


def test_legacy_gate_is_the_unchanged_behaviour_gate(session, monkeypatch, tmp_path):
    sentinel = ({1, 2, 9}, "reference")
    seen = []
    monkeypatch.setattr(oc, "engagement_keep", lambda a, d, sd, gate=None: seen.append((a, d, sd, gate)) or sentinel)
    monkeypatch.setenv(tc.KIN_GATE_ENV, "legacy")
    assert oc.cohort_gate() == "legacy"
    assert oc.cohort_keep("PS92", "20260810", tmp_path) is sentinel
    assert seen == [("PS92", "20260810", tmp_path, None)]           # the env-default OROFACIAL_GATE, as before 11-06
    # engagement_keep itself is untouched: same gates, same default (the encoders / trial_attempts read it)
    assert oc.GATES == ("reference", "attempted", "canonical_attempted")


def test_all_trials_control_is_ungated(session, monkeypatch, tmp_path):
    monkeypatch.setattr(oc, "ENGAGED_ONLY", False)
    assert oc.cohort_gate() == "alltrials"
    assert oc.cohort_keep("PS92", "20260810", tmp_path) == (None, "off (all trials)")


def test_folders_carry_the_gate_and_never_collide(session, monkeypatch, tmp_path):
    from wfield_local import orofacial_outputs as oo
    from wfield_local import rebuild_root
    monkeypatch.setattr(oo, "root", lambda rv: tmp_path / "LP_analysis")
    monkeypatch.setattr(rebuild_root, "root", lambda: None)
    base = tmp_path / "LP_analysis" / oc.RUN / "cohort"
    assert oc.cohort_base(rv=object()) == base                                  # default run: the old place
    w = oc.cohort_dir("tables", rv=object())
    assert w == base / "tables" / "prepost_working"
    assert oc.cohort_dir("figures", "jaw_lateral", rv=object()) == base / "figures" / "jaw_lateral_working"
    monkeypatch.setenv(tc.KIN_GATE_ENV, "legacy")
    lg = oc.cohort_dir("tables", rv=object())
    assert lg == base / "tables" / "prepost_legacygate"
    assert oc.cohort_dir("tables", baseline="all", rv=object()).name == "prepost_legacygate_allpre"
    monkeypatch.setattr(oc, "ENGAGED_ONLY", False)
    al = oc.cohort_dir("tables", rv=object())
    assert al.name == "prepost_alltrials"
    names = {w.name, lg.name, al.name}
    assert len(names) == 3 and "prepost" not in names        # the plain pre-11-06 folder is never written again
    with pytest.raises(ValueError):
        oc.cohort_dir("nope")


def test_cohort_tree_is_routed_under_a_rebuild_root(session, monkeypatch, tmp_path):
    from wfield_local import rebuild_root
    monkeypatch.setattr(rebuild_root, "root", lambda: tmp_path / "rb")
    assert oc.cohort_dir("tables", rv=object()) == tmp_path / "rb" / "LP_analysis" / oc.RUN / "cohort" / "tables" / \
        "prepost_working"


def test_provenance_line_and_titles(session, monkeypatch):
    d = oc.data_line()
    assert d.startswith("Data: ") and f"Trials: {tc.KIN_TRIALS_TEXT['working']}." in d and "August" in d
    assert oc.trials_text() == tc.KIN_TRIALS_TEXT["working"]
    monkeypatch.setenv(tc.KIN_GATE_ENV, "legacy")
    assert tc.KIN_TRIALS_TEXT["legacy"] in oc.data_line()


def test_producers_and_readers_use_the_one_gate_and_folder():
    """Static: the per-session workers call `cohort_keep` (not the behavior gate) and every reader of the cohort tables
    resolves them through `cohort_dir` (no hard-coded tables/prepost left)."""
    src = (REPO / "wfield_local" / "orofacial_cohort.py").read_text(encoding="utf-8")
    body = src[src.index("def session_metrics"):src.index("def collect(")]
    assert "cohort_keep(animal, date, sd)" in body and "engagement_keep(" not in body
    jl = (REPO / "wfield_local" / "orofacial_jaw_lateral.py").read_text(encoding="utf-8")
    assert "oc.cohort_keep(animal, date, sd)" in jl and "oc.engagement_keep(" not in jl
    assert 'oc.cohort_dir("tables", "jaw_lateral"' in jl
    main = src[src.index("def main(argv=None)"):]
    assert 'cohort_dir("tables", run=a.run' in main and "data_line.txt" in main
    readers = ("scripts/disengagement_tests.py", "scripts/imaging_lmm.py", "scripts/kinematics_pooled_figs.py",
               "scripts/lmm_curve_summaries.py", "scripts/orofacial_cover_sensitivity.py",
               "scripts/orofacial_robustness_table.py", "scripts/pose3d_jaw_crosscheck.py", "wfield_local/orofacial_deck.py")
    for r in readers:
        s = (REPO / r).read_text(encoding="utf-8")
        assert "cohort_dir(" in s or "cohort_base(" in s, r
        assert not re.search(r'"tables"\s*/\s*"(prepost|jaw_lateral)"', s), r


# ------------------------------------------------------------------------------------------------ A30 style
def test_summary_fig_session_dots_are_open_animal_epoch():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pose3d_summary_fig as sf
    from matplotlib.colors import to_rgba
    from wfield_local.cat_points import animal_marker
    assert sf.EP_COL is oc.EPOCH_LINE
    fig, ax = plt.subplots()
    pc = sf.session_dots(ax, [1, 2], [0.1, 0.2], "PS93", "acute")
    pj = sf.session_dots(ax, [-20], [0.3], "PS93", "pre", june=True)
    assert len(pc.get_facecolors()) == 0 or np.allclose(pc.get_facecolors()[:, 3], 0)      # open
    assert np.allclose(pc.get_edgecolors()[0], to_rgba(oc.EPOCH_LINE["acute"]))
    assert np.allclose(pj.get_edgecolors()[0], to_rgba(sf.JUNE_EDGE))
    ref = ax.scatter([0], [0], marker=animal_marker("PS93"))
    assert np.allclose(pc.get_paths()[0].vertices, ref.get_paths()[0].vertices)            # shape = animal (square)
    plt.close(fig)
    src = (REPO / "scripts" / "pose3d_summary_fig.py").read_text(encoding="utf-8")
    fig_src = src[src.index("def figure("):src.index("def main(")]
    assert "ax.scatter(" not in fig_src                     # every session dot goes through session_dots


def test_summary_figure_renders_open_dots(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pose3d_summary_fig as sf
    monkeypatch.setattr(sf, "_x", lambda animal, date: float(int(str(date)[4:]) - 820))
    dates = ["20260610", "20260815", "20260825", "20260905"]
    eps = ["pre", "pre", "acute", "chronic"]
    SE = pd.DataFrame([{"animal": "PS92", "subset": "all", "position": p, "date": d, "epoch": e, "miss_med": 1.0 + i}
                       for p in sf.POS for i, (d, e) in enumerate(zip(dates, eps))])
    K = pd.DataFrame([{"animal": "PS92", "date": d, "epoch": e, "jaw_far_dir_mm": 0.1 * i}
                      for i, (d, e) in enumerate(zip(dates, eps))])
    figs, real_close = [], plt.close
    monkeypatch.setattr(plt, "close", lambda f=None: figs.append(f))
    sf.figure("PS92", SE, K, tmp_path / "s.png", "20260801")
    assert (tmp_path / "s.png").exists() and figs
    cols = [c for ax in figs[0].get_axes() for c in ax.collections if len(c.get_offsets())]
    assert cols and all(len(c.get_facecolors()) == 0 or np.allclose(c.get_facecolors()[:, 3], 0) for c in cols)
    assert all(not ax.spines["top"].get_visible() for ax in figs[0].get_axes() if ax.get_visible())
    for f in figs:
        real_close(f)


def test_phase_figures_have_no_top_spine(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pose3d_phase as ph
    fig = plt.figure()
    a2 = fig.add_subplot(1, 2, 1)
    a3 = fig.add_subplot(1, 2, 2, projection="3d")
    ph._save(fig, tmp_path / "x.png")
    assert not a2.spines["top"].get_visible() and a3.name == "3d" and (tmp_path / "x.png").exists()
    plt.close(fig)
    src = (REPO / "scripts" / "pose3d_phase.py").read_text(encoding="utf-8")
    # every figure via _save, which writes the PNG + its svg/ twin through save_png_svg
    assert "fig.savefig(" not in src and src.count("save_png_svg(fig, path") == 1
    assert src.count("    _save(fig, path)\n") == 4
