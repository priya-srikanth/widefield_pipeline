"""The remaining four `--from-csv` arms: what each one had to persist, and what it must refuse.

Each was verified against a live run by byte-comparing the outputs -- that is the real evidence
and it lives in `DECISIONS.md`. These tests hold the parts a re-run cannot re-check cheaply: the
round trips, the refusals, and the one place two implementations could drift apart.

THE FOUR NEEDED DIFFERENT THINGS, which is the point worth remembering when adding a fifth:

    evoked_hrf_latency    nothing new -- it draws no figure, so its per-session CSV is the whole
                          input
    reference_family_roi  nothing new -- it already persisted the region vectors; the work was
                          to stop calling `maps_by_epoch`, which is the entire cost
    lick_bout_structure   a PER-TRIAL table. Its CSV is a session summary and every panel is a
                          per-quintile mean over trials
    quit_prodrome         the cumulative-lick SERIES and the per-trial triples. Both CSVs are
                          per-session summaries and neither can redraw a panel

So the question to ask of a new one is not "is there a CSV" but "is every number the FIGURES use
in it" -- and for three of these four the honest answer was no.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "rest_migration" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ehl = _load("evoked_hrf_latency")
lbs = _load("lick_bout_structure")
rfr = _load("reference_family_roi")
qp = _load("quit_prodrome")


# ------------------------------------------------------------------ evoked_hrf_latency

def test_evoked_hrf_latency_refuses_without_its_csv(tmp_path, capsys):
    assert ehl.main(["--from-csv", "--out", str(tmp_path)]) == 1
    assert "REFUSING" in capsys.readouterr().out


def test_the_tables_are_one_implementation_not_two():
    """Both paths call `_tables`. Two copies would agree today and drift later, and the whole
    claim `--from-csv` makes is that the numbers are the same ones."""
    src = (ROOT / "scripts" / "rest_migration" / "evoked_hrf_latency.py").read_text(
        encoding="utf-8")
    assert src.count("def _tables(") == 1
    assert src.count("_tables(rows, a.seed)") == 2, "the live path and --from-csv must share it"


# ------------------------------------------------------------------ reference_family_roi

def _vectors_npz(tmp_path, n=4, nreg=3):
    import json
    meta = [{"family": "cue", "animal": f"PS9{2 + i % 2}", "epoch": "pre" if i < 2 else "acute",
             "position": str(i % 2), "session": f"s{i}"} for i in range(n)]
    V = np.arange(n * nreg, dtype=np.float32).reshape(n, nreg)
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    np.savez_compressed(d / "epoch_15k_region_vectors_cue.npz", meta=json.dumps(meta),
                        regions=json.dumps([f"r{j}" for j in range(nreg)]),
                        sids=np.arange(nreg), V=V)
    return meta, V


def test_region_vectors_round_trip_in_stored_order(tmp_path):
    meta, V = _vectors_npz(tmp_path)
    got, sids, labels = rfr.load_vectors(tmp_path, "cue")
    assert [d["session"] for d in got] == [m["session"] for m in meta], "order must survive"
    assert labels == ["r0", "r1", "r2"] and list(sids) == [0, 1, 2]
    for i, d in enumerate(got):
        assert np.array_equal(d["v"], V[i].astype(np.float64))


def test_a_missing_vectors_npz_is_reported_not_invented(tmp_path):
    assert rfr.load_vectors(tmp_path, "cue") == (None, None, None)


def test_meta_and_vectors_of_different_lengths_is_an_error(tmp_path):
    """Pairing them by position when they disagree would attach every vector to the wrong cell,
    and every number downstream would still look plausible."""
    import json
    d = tmp_path / "data"
    d.mkdir()
    np.savez_compressed(d / "epoch_15k_region_vectors_cue.npz",
                        meta=json.dumps([{"family": "cue", "animal": "PS92", "epoch": "pre",
                                          "position": "0", "session": "s0"}]),
                        regions=json.dumps(["r0"]), sids=np.arange(1),
                        V=np.zeros((3, 1), np.float32))
    with pytest.raises(SystemExit, match="refusing"):
        rfr.load_vectors(tmp_path, "cue")


# ------------------------------------------------------------------ lick_bout_structure

def test_the_quintile_filter_is_one_function_not_two():
    """`per_session_q` was a byte-for-byte copy of `quint`. That filter decides which sessions
    enter a quintile, so two copies are two places for a published CI to drift."""
    src = (ROOT / "scripts" / "rest_migration" / "lick_bout_structure.py").read_text(
        encoding="utf-8")
    assert "per_session_q = quint" in src
    assert src.count("def per_session_q(") == 0


def test_lick_bout_structure_needs_BOTH_tables(tmp_path, capsys):
    """The summary alone cannot redraw a panel, so having only it must refuse rather than draw
    something thinner and call it the figure."""
    d = tmp_path / "data"
    d.mkdir()
    (d / "epoch_26_lick_bout_structure.csv").write_text(
        "label,animal,epoch,early_lpm,base_licks,cross_time_s,cross_licks,n_trials\n"
        "PS92_0601,PS92,pre,1.0,2.0,3.0,4.0,5\n", encoding="utf-8")
    assert lbs.main(["--from-csv", "--out", str(tmp_path)]) == 1
    assert "REFUSING" in capsys.readouterr().out


# ------------------------------------------------------------------ quit_prodrome

def _sess(n_trials=(3, 2)):
    out = {}
    for i, nt in enumerate(n_trials):
        lab = f"PS9{2 + i}_060{i}"
        out[lab] = {
            "label": lab, "animal": f"PS9{2 + i}", "epoch": "pre" if i == 0 else "acute",
            "t": np.arange(nt, dtype=float) * 10.0,
            "c": np.arange(nt, dtype=float) * 3.0,
            "quit_s": float("nan") if i else 120.0, "censored": bool(i), "end_s": 999.0 + i,
            "per_pos": {"near": [(1.0, 2.0, 0.1)] * (i + 1), "far": []},
            "per_ili": {"near": [], "far": [(5.0, 90.0, 0.5)] * (2 - i)},
        }
    return out


def test_quit_prodrome_state_round_trips_including_the_empty_lists(tmp_path):
    """RAGGED BY CONSTRUCTION: a session contributes a `near` triple only when it had one, and an
    empty list must come back empty rather than as a row of zeros."""
    sess = _sess()
    assert qp.save_state(sess, tmp_path) is not None
    back = qp.load_state(tmp_path)
    assert list(back) == list(sess), "session order must survive"
    for lab, v in sess.items():
        b = back[lab]
        assert b["animal"] == v["animal"] and b["epoch"] == v["epoch"]
        assert b["censored"] is v["censored"] and b["end_s"] == v["end_s"]
        assert np.array_equal(b["t"], v["t"]) and np.array_equal(b["c"], v["c"])
        for key in ("per_pos", "per_ili"):
            for grp in ("near", "far"):
                assert len(b[key][grp]) == len(v[key][grp])
                for got, want in zip(b[key][grp], v[key][grp]):
                    assert tuple(got) == pytest.approx(want)


def test_a_censored_session_keeps_its_NaN_quit_time(tmp_path):
    """NaN is the measurement here -- it means 'never quit' -- so it must not read back as 0."""
    qp.save_state(_sess(), tmp_path)
    back = qp.load_state(tmp_path)
    vals = [v["quit_s"] for v in back.values()]
    assert np.isnan(vals[1]) and vals[0] == 120.0


def test_a_missing_state_npz_is_reported_not_invented(tmp_path):
    assert qp.load_state(tmp_path) is None


def test_counts_that_disagree_with_the_arrays_are_an_error(tmp_path):
    """The sessions are sliced out of one flat array by their counts. If those disagree, every
    session after the first is silently offset -- plausible numbers, wrong sessions."""
    import json
    qp.save_state(_sess(), tmp_path)
    q = tmp_path / "data" / "epoch_23_quit_prodrome_state.npz"
    with np.load(q, allow_pickle=False) as f:
        d = dict(f)
    meta = json.loads(str(d["meta"]))
    meta[0]["n_t"] = int(meta[0]["n_t"]) - 1          # one point short
    d["meta"] = json.dumps(meta)
    np.savez_compressed(q, **d)
    with pytest.raises(SystemExit, match="mis-sliced"):
        qp.load_state(tmp_path)
