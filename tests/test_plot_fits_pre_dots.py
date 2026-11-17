"""Pins the house style of `mixed_models.plot_fits` (open session markers, shape = animal, color = epoch) and the
pre-session dots of the decoder trajectory (Priya 11-06): pre rows go to a SEPARATE sidecar, the post-only series CSV
is byte-identical to what it was, the pre rows carry their own label / day, and `imaging_lmm.decoder_table` reads
them. Synthetic only: no share access, no real figure."""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb

from wfield_local import mixed_models as mm
from wfield_local import recovery_trajectory as rt
from wfield_local.orofacial_cohort import EPOCH_LINE


def _long(with_pre=True):
    rows = []
    rng = np.random.default_rng(0)
    for an in ("PS92", "PS93", "PS94"):
        days = ([-8, -5, -2] if with_pre else []) + [1, 3, 9, 20, 40]
        for d in days:
            ep = "pre" if d <= 0 else ("acute" if d <= 5 else ("subacute" if d < 15 else "chronic"))
            post = float(d > 0)
            rows.append({"label": f"{an}_{d}", "animal": an, "day": float(d), "epoch": ep, "post": post,
                         "lday": np.log(max(d, 1)) if d > 0 else 0.0, "source": "s", "measure": "m",
                         "y": rng.normal(), "deficit": 0.0 if d <= 0 else 0.3})
    return pd.DataFrame(rows)


def _T(model):
    terms = ({"time": ("pre_level", "post", "post:lday"), "time_post": ("post_level_day1", "lday")}[model])
    return pd.DataFrame([{"model": model, "source": "s", "measure": "m", "term": t, "coef": 0.1, "se": 0.1, "z": 1.0,
                          "p": 0.3, "q": 0.3, "re": "~1", "n": 24} for t in terms])


def _draw(monkeypatch, tmp_path, L, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close("all")
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)     # keep the figure to inspect its artists
    mm.plot_fits(L, _T(model), [("m", {"source": "s", "measure": "m"})], model, tmp_path / "f.png", "t", "v")
    assert (tmp_path / "f.png").exists()
    return plt.figure(plt.get_fignums()[-1])


@pytest.mark.parametrize("model", ["time", "time_post"])
def test_session_markers_are_OPEN_and_coloured_by_EPOCH(monkeypatch, tmp_path, model):
    from matplotlib.collections import PathCollection
    fig = _draw(monkeypatch, tmp_path, _long(), model)
    ax = fig.axes[0]
    cols = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert cols, "no session dots drawn"
    seen = set()
    for c in cols:
        fc = c.get_facecolors()
        assert len(fc) == 0 or np.allclose(fc[:, 3], 0), "session markers must be OPEN (facecolor none)"
        ec = {tuple(np.round(e[:3], 3)) for e in c.get_edgecolors()}
        assert len(ec) == 1
        match = [ep for ep, col in EPOCH_LINE.items() if tuple(np.round(to_rgb(col), 3)) in ec]
        assert match, f"edge color {ec} is not an epoch color"
        seen.add(match[0])
    assert seen == {"pre", "acute", "subacute", "chronic"}
    assert not ax.spines["top"].get_visible()


def test_post_only_fit_leaves_the_pre_dots_OUT_of_the_line(monkeypatch, tmp_path):
    """time_post: pre rows are drawn (pre color, negative days) but the fitted curve starts at day 1."""
    fig = _draw(monkeypatch, tmp_path, _long(), "time_post")
    ax = fig.axes[0]
    pre_rgb = tuple(np.round(to_rgb(EPOCH_LINE["pre"]), 3))
    pre_x = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                            if tuple(np.round(c.get_edgecolors()[0][:3], 3)) == pre_rgb])
    assert len(pre_x) == 9 and (pre_x < 0).all()
    fit = [ln for ln in ax.get_lines() if ln.get_label() == "fit"]
    assert fit and np.min(fit[0].get_xdata()) >= 1
    assert "not in the post-only fit" in fig._supxlabel.get_text()


# ------------------------------------------------------------------ the pre-session sidecar

def _row(an, day, fz, rf, label=None):
    r = rt._pair_row(an, day, "acute" if day > 0 else "pre", (fz, rf, 100), 0.8, 0.85, False)
    return {"label": label, **r} if label else r


#: FROZEN COPY of `write_csv` before the pre sidecar (2026-11-06), so "byte-identical" is pinned, not eyeballed
def _old_write_csv(rows, dest):
    cols = ["animal", "matched", "day", "epoch", "n_trials", "frozen", "refit", "gap",
            "pre_frozen", "pre_refit", "pre_gap", "F_deficit", "G_reorg", "refit_deficit"]
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, '') for c in cols} for r in rows)
    return dest


def test_post_only_CSV_is_byte_identical_and_pre_rows_go_to_their_own_file(monkeypatch, tmp_path):
    post = [_row("PS92", 2, 0.5, 0.7), _row("PS92", 9, 0.7, 0.8)]
    pre = [_row("PS92", -3, 0.82, 0.86, "PS92_0809"), _row("PS92", 0, 0.78, 0.84, "PS92_0812")]
    monkeypatch.setattr(rt, "series", lambda *a, matched=False, **k: [dict(r, matched=int(matched)) for r in post])
    monkeypatch.setattr(rt, "pre_points", lambda *a, **k: {"PS92": [(0.0, 0.0)]})
    monkeypatch.setattr(rt, "figure", lambda *a, **k: [])
    monkeypatch.setattr(rt, "figure_by_animal", lambda *a, **k: [])
    outs = {}
    for name, pre_rows in (("with", pre), ("without", [])):
        monkeypatch.setattr(rt, "pre_series", lambda *a, _p=pre_rows, **k: list(_p))
        rt.run(out_dir=tmp_path / name)
        outs[name] = {p.name: p.read_bytes() for p in (tmp_path / name).rglob("*.csv")}
    for tag in ("unmatched", "matched"):
        stem = f"recovery_trajectory_{tag}_cue_working"
        ref = _old_write_csv([dict(r, matched=int(tag == "matched")) for r in post], tmp_path / f"ref_{tag}.csv")
        assert outs["with"][stem + ".csv"] == outs["without"][stem + ".csv"] == ref.read_bytes()
        assert stem + "_pre.csv" in outs["with"], "the (F, G) pre cloud is still written under its own name"
        assert stem + rt.PRE_SESSIONS_SUFFIX in outs["with"]
        assert stem + rt.PRE_SESSIONS_SUFFIX not in outs["without"]
    P = pd.read_csv(tmp_path / "with" / "data" / ("recovery_trajectory_unmatched_cue_working" + rt.PRE_SESSIONS_SUFFIX))
    assert list(P.columns) == ["label", *rt.COLS]
    assert list(P.label) == ["PS92_0809", "PS92_0812"] and (P.epoch == "pre").all()
    assert np.allclose(P.F_deficit, 0.8 - P.frozen) and np.allclose(P.refit_deficit, 0.85 - P.refit)


def test_pre_series_places_each_HELD_OUT_record_at_its_own_day(monkeypatch):
    from tests.test_recovery_trajectory import _rec
    from wfield_local import epochs
    from wfield_local import grant_figures as G
    from wfield_local.grant_confusion import PreRecords
    recs = PreRecords([_rec(300, 0.9, 0.85, seed=1), _rec(300, 0.7, 0.8, seed=2)])
    recs.labels += ["PS92_0807", "PS92_0811"]
    unl = [_rec(300, 0.9, 0.85, seed=3)]                       # no labels: skipped, never guessed
    monkeypatch.setattr(G, "_collect_5c", lambda *a, **k: ({"PS92": (recs, {}), "PS93": (unl, {})}, []))
    monkeypatch.setattr(epochs, "days_since_stroke", lambda lab: {"PS92_0807": -6, "PS92_0811": -2}[lab])
    rows = rt.pre_series()
    assert [(r["label"], r["day"]) for r in rows] == [("PS92_0807", -6), ("PS92_0811", -2)]
    pre_f, pre_r = rt._pre_pair(recs)
    assert np.isclose(np.mean([r["frozen"] for r in rows]), pre_f)          # equal-n records: mean = pooled
    assert all(np.isclose(r["F_deficit"], pre_f - r["frozen"]) for r in rows)
    assert all(np.isclose(r["G_reorg"], r["gap"] - (pre_r - pre_f)) for r in rows)


def test_PreRecords_is_still_a_plain_list_to_existing_readers():
    from wfield_local.grant_confusion import PreRecords
    p = PreRecords()
    assert isinstance(p, list) and not p and (p or None) is None and p.labels == []


def test_decoder_table_reads_the_pre_sidecar(monkeypatch, tmp_path):
    import importlib
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    il = importlib.import_module("scripts.imaging_lmm")
    d = tmp_path / "epoch"
    (d / "data").mkdir(parents=True)
    stem = "recovery_trajectory_unmatched_cue_working"
    rt.write_csv([_row("PS92", 2, 0.5, 0.7)], d / "data" / f"{stem}.csv")
    rt.write_csv([_row("PS92", -3, 0.82, 0.86, "PS92_0809"), _row("PS92", -1, 0.8, 0.8, "PS92_0899")],
                 d / "data" / f"{stem}{rt.PRE_SESSIONS_SUFFIX}", cols=["label", *rt.COLS])
    days = {"PS92_0809": -3, "PS92_0814": 2}
    monkeypatch.setattr(ak, "curated_labels", lambda **k: sorted(days))
    monkeypatch.setattr(epochs, "days_since_stroke", lambda lab: days.get(lab))
    D = il.decoder_table(d)
    assert set(D.label) == {"PS92_0814", "PS92_0809"}, "the uncurated pre label is dropped"
    assert set(D.source) == {"decoder_unmatched"}
    assert (D[D.label == "PS92_0809"].measure.nunique()) == len(il.DECODER_METRICS)
