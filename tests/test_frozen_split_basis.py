"""FROZEN ENCODER AREA BASIS (Priya 11-26: Allen-area frozen targets default to the ALM / tjM1 split; plain = robustness;
existing frames unchanged). Synthetic grid, no share: the split masks are `area_basis` ROI version 1 (no v8 rim, so no
brain-mask read); the production v8 path is the same function with the rim on.

Pinned: an existing (plain) frame resolves, tags and reads exactly as before; a NEW frame defaults to split with the split
names; the run tag (the frame's identity) differs between the two; the bundle reduction keeps the split pixels; the
readers' Bonferroni count / primary rows / named-area guard accept both."""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wfield_local import area_basis as ab
from wfield_local import frozen_encoder as fe

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(fe.AREA_BASIS_ENV, raising=False)
    monkeypatch.setattr(ab, "ROI_VERSION", "1")          # v1 masks: no v8 rim -> no share brain-mask read


def _grid():
    """40 x 40 brain, both hemispheres: MOs (rows 0-12), MOp (13-25), SSp-m (26-39); stat mask = all but a 1-px border."""
    H = W = 40
    flat = np.arange(H * W)
    r, c = flat // W, flat % W
    left = c < W // 2
    band = np.where(r < 13, 0, np.where(r < 26, 1, 2))
    base = np.array([1, 2, 3])[band]
    parea = np.where(left, base, base + 10)
    names = {1: "MOs_left", 2: "MOp_left", 3: "SSp-m_left", 11: "MOs_right", 12: "MOp_right", 13: "SSp-m_right"}
    pstat = (r > 0) & (r < H - 1) & (c > 0) & (c < W - 1)
    rng = np.random.default_rng(0)
    Ub = rng.standard_normal((H * W, 5)).astype(np.float32)
    return Ub, flat, parea, names, (H, W), pstat


def _plain_ra(Ub, parea, names, pstat):
    """The pre-11-26 Allen-area target, verbatim (`session_design`'s plain branch)."""
    areas = sorted({int(a) for a in np.unique(parea[pstat]) if a != 0})
    Ra = np.stack([Ub[(parea == a) & pstat].mean(0) for a in areas] + [Ub[pstat].mean(0)], 1)
    return Ra, [names.get(a, str(a)) for a in areas] + ["cortex"]


# ------------------------------------------------------------------------------------------------ identity / default
def test_run_tag_records_the_basis_and_plain_is_unchanged(monkeypatch):
    legacy = fe._run_tag()
    assert fe._AREA_TAG not in legacy and fe.area_basis() == "plain"
    monkeypatch.setenv(fe.AREA_BASIS_ENV, "plain")
    assert fe._run_tag() == legacy                       # plain adds NOTHING: old run tags / statistics match
    monkeypatch.setenv(fe.AREA_BASIS_ENV, "split")
    assert fe._run_tag() == legacy + "|areas-split" != legacy
    assert fe.area_basis_of_run_mode(fe._run_tag()) == "split" and fe.area_basis_of_run_mode(legacy) == "plain"
    monkeypatch.setenv(fe.AREA_BASIS_ENV, "allen")
    with pytest.raises(ValueError):
        fe.area_basis()


def _frame(tmp_path, run_mode):
    (tmp_path / "fit").mkdir(parents=True)
    (tmp_path / "fit" / "pre_stats.pkl").write_bytes(pickle.dumps([{"label": "PS92_0806", "run_mode": run_mode}]))
    return tmp_path


def test_new_frame_defaults_split_existing_frame_keeps_its_basis(tmp_path, monkeypatch):
    assert fe.resolve_area_basis(None, tmp_path / "new") == "split" == fe.AREA_BASIS_NEW
    assert fe.resolve_area_basis("plain", tmp_path / "new") == "plain"          # robustness frame on request
    old = _frame(tmp_path / "old", "spline_thr|k0=1.5|tongue3d|licksall-split")
    assert fe.frame_area_basis(old) == "plain" and fe.resolve_area_basis(None, old) == "plain"
    with pytest.raises(SystemExit, match="never re-fitted in another basis"):
        fe.resolve_area_basis("split", old)
    monkeypatch.setenv(fe.AREA_BASIS_ENV, "split")
    with pytest.raises(SystemExit):
        fe.resolve_area_basis(None, old)
    monkeypatch.delenv(fe.AREA_BASIS_ENV)
    new = _frame(tmp_path / "split", "spline_thr|k0=1.5|areas-split")
    assert fe.resolve_area_basis(None, new) == "split"
    assert fe.adopt_frame_area_basis(old) == "plain" and fe.area_basis() == "plain"
    monkeypatch.delenv(fe.AREA_BASIS_ENV)
    assert fe.adopt_frame_area_basis(new) == "split" and fe._run_tag().endswith("|areas-split")


def test_main_refuses_to_refit_a_plain_frame_in_split(tmp_path):
    tag = fe.design_tag("gate-attempted_v7g3_t3d")
    _frame(tmp_path / f"frozen_encoder_{tag}", "linear")
    before = sorted(p.name for p in (tmp_path / f"frozen_encoder_{tag}").rglob("*"))
    with pytest.raises(SystemExit, match="plain-basis frame"):
        fe.main(["--tag", tag, "--out-root", str(tmp_path), "--target", "split", "--reuse-stats"])
    assert sorted(p.name for p in (tmp_path / f"frozen_encoder_{tag}").rglob("*")) == before


# ------------------------------------------------------------------------------------------------------- targets
def test_split_targets_names_and_values():
    Ub, flat, parea, names, shape, pstat = _grid()
    am = fe.area_mask_record("PS92_0806", parea, pstat, names, flat, shape)
    assert am["basis"] == "split" and am["names"] == ab.session_area_masks(parea, pstat, names, flat, shape)["split"]
    Ra, nm = fe.split_area_targets("PS92_0806", Ub, flat, pstat, am)
    assert nm[-1] == "cortex" and not ({"MOs_left", "MOs_right", "MOp_left", "MOp_right"} & set(nm))
    assert {f"{n}_{h}" for n in ("ALM", "tjM1", "MOs_rest", "MOp_rest") for h in ("left", "right")} <= set(nm)
    for k, n in enumerate(nm[:-1]):
        np.testing.assert_array_equal(Ra[:, k], Ub[np.isin(flat, am["pix"][n])].mean(0))
    np.testing.assert_array_equal(Ra[:, -1], Ub[pstat].mean(0))     # cortex = the stat mask, as plain frames
    # the split pieces tile their parent (v1: the masks are on the stat mask): pixel-weighted ALM + MOs_rest = MOs
    Rp, nmp = _plain_ra(Ub, parea, names, pstat)
    n_alm, n_rest = len(am["pix"]["ALM_left"]), len(am["pix"]["MOs_rest_left"])
    mos = (n_alm * Ra[:, nm.index("ALM_left")] + n_rest * Ra[:, nm.index("MOs_rest_left")]) / (n_alm + n_rest)
    np.testing.assert_allclose(mos, Rp[:, nmp.index("MOs_left")], atol=1e-6)


def test_bundled_masks_are_required_on_replayed_imaging():
    Ub, flat, parea, names, shape, pstat = _grid()
    am = fe.area_mask_record("PS92_0806", parea, pstat, names, flat, shape)
    with fe.bundled_area_masks({"PS92_0806": am}):
        assert fe.area_mask_record("PS92_0806", None, None, None, None, None) is am
        with pytest.raises(RuntimeError, match="no split area_masks"):
            fe.area_mask_record("PS92_0807", parea, pstat, names, flat, shape)
    assert fe.area_mask_record("PS92_0807", parea, pstat, names, flat, shape)["names"] == am["names"]   # context left


def test_bundle_reduction_keeps_the_split_pixels(monkeypatch):
    import o2_partial as op
    Ub, flat, parea, names, shape, pstat = _grid()
    v = (np.zeros((3, 5)), np.arange(3.0), Ub, flat, parea, names, shape, pstat)
    plain = op._reduce("load_imaging", v, "PS92_0806")[1]
    assert plain[7].all() and len(plain[3]) == int(pstat.sum())     # plain bundles: stat pixels only, as before
    monkeypatch.setenv(fe.AREA_BASIS_ENV, "split")
    red = op._reduce("load_imaging", v, "PS92_0806")[1]
    am = fe.area_mask_record("PS92_0806", parea, pstat, names, flat, shape)
    Ra_full, nm = fe.split_area_targets("PS92_0806", Ub, flat, pstat, am)
    Ra_red, nm_r = fe.split_area_targets("PS92_0806", red[2], red[3], red[7], am)
    assert nm == nm_r
    np.testing.assert_array_equal(Ra_full, Ra_red)
    with pytest.raises(RuntimeError, match="missing from the imaging"):           # a plain-reduced bundle: refused
        fe.split_area_targets("PS92_0806", plain[2], plain[3], plain[7],
                              {**am, "pix": {**am["pix"], "ALM_left": np.array([0])}})


def test_o2_expected_run_tag_carries_the_split(monkeypatch):
    spec = importlib.util.spec_from_file_location("o2f_split", REPO / "scripts" / "o2_frozen.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setenv("COMBINED_ENCODING_POSE", "xyz3d")
    plain = m.expected_run_tag("spont_frA")
    monkeypatch.setenv(fe.AREA_BASIS_ENV, "split")
    assert m.expected_run_tag("spont_frA") == plain + "|areas-split"
    assert m.expected_run_tag("spont_basis") == m.expected_run_tag("spont_basis")   # spatial frames: no area basis
    assert "|areas-" not in m.expected_run_tag("spont_basis")


# ------------------------------------------------------------------------------------------------------- readers
def _allen_areas():
    base = ["SSp-m", "SSp-n", "MOp", "MOs", "VISp", "RSPd"]
    return [f"{a}_{h}" for a in base for h in ("left", "right")]


def _old_bonf(B, keys, ref=None):
    import encoder_split_rows as esr
    ref = B if ref is None else ref
    n = ref[~ref.area.isin(esr.SPLIT_AREAS)].groupby(list(keys)).area.nunique()
    return np.minimum(B.p.to_numpy(float) * B.set_index(list(keys)).index.map(n).to_numpy(float), 1.0)


def _split_frame(allen):
    import encoder_split_rows as esr
    return [a for a in allen if not a.startswith(("MOs_", "MOp_"))] + list(esr.SPLIT_AREAS)


def test_bonferroni_plain_unchanged_split_frame_counts_every_row():
    import encoder_split_rows as esr
    rng = np.random.default_rng(1)
    allen = _allen_areas()
    split_frame = _split_frame(allen)

    def tab(areas):
        return pd.DataFrame([{"q": q, "window": w, "area": a, "p": rng.uniform(0, 0.05)}
                             for q in ("residual", "residual_z") for w in ("cue", "lick") for a in areas])
    P, PT, S = tab(allen), tab(allen + list(esr.SPLIT_AREAS)), tab(split_frame)
    for T in (P, PT):                                                    # plain frame, plain + twin rows: unchanged
        np.testing.assert_array_equal(esr.bonf_allen(T, ["q", "window"]), _old_bonf(T, ["q", "window"]))
    sp = PT[PT.area.isin(esr.SPLIT_AREAS)]
    np.testing.assert_array_equal(esr.bonf_allen(sp, ["q", "window"], ref=P), _old_bonf(sp, ["q", "window"], ref=P))
    # split-basis frame (Priya 11-26): over EVERY area row tested (ALM / tjM1 / rests count): 16 here, 45 on the atlas
    np.testing.assert_allclose(esr.bonf_allen(S, ["q", "window"]), np.minimum(S.p * len(split_frame), 1.0))
    assert esr.bonf_count(split_frame) == len(split_frame) == len(allen) + 4
    assert esr.bonf_count(allen) == esr.bonf_count(allen + list(esr.SPLIT_AREAS)) == len(allen)
    # primary rows: plain + twin -> the Allen rows only (as before); split frame -> every row
    np.testing.assert_array_equal(esr.primary_rows(PT.area), ~PT.area.isin(esr.SPLIT_AREAS).to_numpy())
    assert esr.primary_rows(S.area).all()
    assert esr.is_split_frame(split_frame) and not esr.is_split_frame(allen + list(esr.SPLIT_AREAS))
    assert not esr.is_split_frame(allen)


def test_split_area_list_side_by_side_and_plain_unchanged():
    import encoder_split_rows as esr
    allen = _allen_areas() + ["cortex"]
    want = ["cortex", "SSp-m_right", "MOp_right", "MOp_left", "MOs_right"]
    assert fe.split_area_list(want, allen) == want                               # plain: unchanged
    assert fe.split_area_list(want, allen + list(esr.SPLIT_AREAS)) == want      # plain + twin rows: unchanged
    assert fe.split_area_list(want, _split_frame(allen)) == [
        "cortex", "SSp-m_right", "tjM1_right", "MOp_rest_right", "tjM1_left", "MOp_rest_left", "ALM_right",
        "MOs_rest_right"]


# --------------------------------------------------------------------------------------- readers on fake frames
def _write_frame(root, folder, areas, variants=("joint",), seed=0):
    import json
    rng = np.random.default_rng(seed)
    for v in variants:
        d = root / folder / v / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        for lab, ep in (("PS92_0806", "pre"), ("PS92_0901", "acute")):
            r2 = {w: {m: rng.uniform(0, 1, len(areas)).tolist() for m in ("frame", "trial", "avg")}
                  for w in ("cue", "lick")}
            (d / f"{lab}.json").write_text(json.dumps({"label": lab, "animal": "PS92", "epoch": ep, "areas": areas,
                                                       "r2_musall": r2, "n_trials": 100}))
    return root / folder


def _old_session_r2(E, folder, variant, AREAS, measure="frame", areas=None):
    """frozen_fr_compare.session_r2 before 11-26, verbatim."""
    import glob
    import json
    rows = []
    for f in sorted(glob.glob(str(E / folder / variant / "sessions" / "*.json"))):
        m = json.loads(Path(f).read_text())
        for w in ("cue", "lick"):
            fr = m["r2_musall"][w][measure]
            for a in (AREAS if areas is None else areas):
                if a in m["areas"]:
                    rows.append({"label": m["label"], "animal": m["animal"], "epoch": m["epoch"], "window": w,
                                 "area": a, "r2": fr[m["areas"].index(a)]})
    return pd.DataFrame(rows)


def _old_table(sdir, AREAS, measure="frame", areas=None):
    """partial_refit.table before 11-26, verbatim."""
    import json
    areas = AREAS if areas is None else areas
    rows = []
    for f in sorted(sdir.glob("*.json")):
        m = json.loads(f.read_text())
        for v, ww in m["scores"].items():
            for w, dd in ww.items():
                for key, vals in dd.items():
                    if key not in (measure,) and not key.startswith("spout:"):
                        continue
                    for a in (m["areas"] if areas == "all" else areas):
                        if a in m["areas"]:
                            rows.append((m["label"], m["animal"], m["epoch"], v, w, key, a, vals[m["areas"].index(a)]))
    return pd.DataFrame(rows, columns=["label", "animal", "epoch", "variant", "window", "measure", "area", "r2"])


def _old_frozen_r2(run, variant, AREAS, measure="frame", areas=None):
    """session_vs_frozen.frozen_r2 before 11-26, verbatim."""
    import json
    rows = []
    for f in sorted((run / variant / "sessions").glob("*.json")):
        m = json.loads(f.read_text())
        for w, dd in m["r2_musall"].items():
            for a in (m["areas"] if areas == "all" else (areas or AREAS)):
                if a in m["areas"]:
                    rows.append({"label": m["label"], "window": w, "area": a, "variant": variant,
                                 "frozen": dd[measure][m["areas"].index(a)]})
    return pd.DataFrame(rows)


def _same(a, b):
    pd.testing.assert_frame_equal(a, b)
    assert a.to_csv().encode() == b.to_csv().encode()


def test_session_r2_plain_byte_identical_split_side_by_side(tmp_path, monkeypatch):
    import frozen_fr_compare as ffc
    monkeypatch.setattr(ffc, "E", tmp_path)
    allen = _allen_areas() + ["cortex"]
    _write_frame(tmp_path, "plain", allen)
    _write_frame(tmp_path, "twin", allen + list(__import__("encoder_split_rows").SPLIT_AREAS))
    _write_frame(tmp_path, "split", _split_frame(allen))
    for folder in ("plain", "twin"):
        for areas in (None, ["cortex", "MOp_right", "MOs_left"]):
            _same(ffc.session_r2(folder, "joint", areas=areas), _old_session_r2(tmp_path, folder, "joint", ffc.AREAS,
                                                                                areas=areas))
    S = ffc.session_r2("split", "joint")
    assert list(dict.fromkeys(S.area)) == ["cortex", "SSp-m_right", "SSp-m_left", "tjM1_right", "MOp_rest_right",
                                           "tjM1_left", "MOp_rest_left"]
    assert all(a in ffc.ALAB for a in S.area)


def test_partial_refit_and_svf_tables_plain_identical_split_expanded(tmp_path):
    import json

    import partial_refit as pr
    import session_vs_frozen as svf
    allen = _allen_areas() + ["cortex"]
    for kind, areas in (("plain", allen), ("split", _split_frame(allen))):
        sd = tmp_path / kind / "sessions"
        sd.mkdir(parents=True)
        rng = np.random.default_rng(3)
        for lab in ("PS92_0806", "PS92_0901"):
            sc = {"joint": {w: {"frame": rng.uniform(0, 1, len(areas)).tolist()} for w in ("cue", "lick")}}
            (sd / f"{lab}.json").write_text(json.dumps({"label": lab, "animal": "PS92", "epoch": "pre", "areas": areas,
                                                        "scores": sc}))
    for areas in (None, "all"):
        kw = {} if areas is None else {"areas": areas}
        _same(pr.table(tmp_path / "plain" / "sessions", **kw), _old_table(tmp_path / "plain" / "sessions", pr.AREAS,
                                                                          areas=areas))
    S = pr.table(tmp_path / "split" / "sessions")
    assert list(dict.fromkeys(S.area)) == fe.split_area_list(pr.AREAS, _split_frame(allen))
    run_p = _write_frame(tmp_path, "frozen_encoder_plain", allen)
    run_s = _write_frame(tmp_path, "frozen_encoder_split", _split_frame(allen))
    for areas in (None, "all", ["SSp-m_right", "MOs_left"]):
        _same(svf.frozen_r2(run_p, "joint", areas=areas), _old_frozen_r2(run_p, "joint", svf.AREAS, areas=areas))
    Fs = svf.frozen_r2(run_s, "joint")
    assert list(dict.fromkeys(Fs.area)) == fe.split_area_list(svf.AREAS, _split_frame(allen))
    assert len(svf.frozen_r2(run_s, "joint", areas="all")) == 2 * 2 * len(_split_frame(allen))


def test_contest_lines_unique_table_split_frame(tmp_path, monkeypatch):
    import contest_lines as cl
    allen = _allen_areas() + ["cortex"]
    monkeypatch.setattr(cl, "E", tmp_path)
    for tag, areas in (("p", allen), ("s", _split_frame(allen))):
        (tmp_path / f"frozen_encoder_{tag}").mkdir()
        pd.DataFrame([{"label": "PS92_0806", "animal": "PS92", "epoch": "pre", "align": "cue", "area": a,
                       "measure": "frame", "variant": v, "r2": 0.5} for a in areas for v in ("joint", "no_target")]) \
            .to_csv(tmp_path / f"frozen_encoder_{tag}" / "frozen_musall_r2.csv", index=False)
    P = cl.unique_table("p", {"target": "target"})
    assert set(P.area) == set(cl.AREAS)
    S = cl.unique_table("s", {"target": "target"})
    assert set(S.area) == set(fe.split_area_list(cl.AREAS, _split_frame(allen)))


def test_specificity_orofacial_group_on_split_frames():
    import specificity_control as sc
    allen = _allen_areas() + ["VISp_left", "VISp_right"]
    assert sc.groups_for(allen) is sc.GROUPS                                         # plain: unchanged
    g = sc.groups_for(_split_frame(allen))
    assert g["orofacial"] == [f"{a}_{h}" for a in ("SSp-m", "SSp-n", "ALM", "tjM1") for h in ("left", "right")]
    assert g["control"] == sc.CONTROL
    rows = [{"animal": "PS92", "label": lab, "epoch": ep, "window": "cue", "area": a, "residual": 1.0 + k,
             "residual_z": float(k)} for lab, ep in (("PS92_0806", "pre"), ("PS92_0901", "acute"))
            for k, a in enumerate(_split_frame(allen))]
    T = pd.DataFrame(rows)
    G = sc.group_means(T, ("residual_z", "residual"))
    oro = G[G.area == "orofacial"].set_index("label").residual_z.sort_index()
    want = T[T.area.isin(g["orofacial"])].groupby("label").residual_z.mean().sort_index()
    np.testing.assert_allclose(oro.to_numpy(), want.to_numpy())
