"""ALM / tjM1 on the slide-94 / slide-139 area maps (Priya 11-06): session_vs_frozen --roi-targets (self-fit ROI targets,
Allen fits unchanged), the split painting of session_vs_frozen --maps / remap_checks / frozen_peth_trial_maps --split,
the Allen-only Bonferroni count, and reembedding_test's map (split ROIs painted on their own territory, values unchanged).
Synthetic inputs only (no imaging); the real-data identity checks are in docs/status/ALM_TJM1_ROWS_PLAN_20261027.md."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
AN = ["PS92", "PS93", "PS94", "PS95"]


@pytest.fixture
def scripts():
    """scripts/ on sys.path for the test; afterwards sys.path, os.environ and the SCRIPT modules imported are restored /
    dropped (wfield_local.* stays: dropping it breaks pickling)."""
    env, path, before = dict(os.environ), list(sys.path), set(sys.modules)
    sys.path.insert(0, str(SCRIPTS))
    yield
    for m in set(sys.modules) - before:
        f = getattr(sys.modules.get(m), "__file__", None) or ""
        if f and Path(f).resolve().parent == SCRIPTS:
            del sys.modules[m]
    sys.path[:] = path
    os.environ.clear()
    os.environ.update(env)


# ------------------------------------------------------------------------------------------------ synthetic atlas
def _img():
    """10 x 10: left half SSp-m_left (0), right half MOs_left (1), a -1 border column."""
    img = np.full((10, 10), -1)
    img[:, 1:5], img[:, 5:9] = 0, 1
    return img, ["SSp-m_left", "MOs_left"]


def _split_image(img, nm):
    """esr.split_image's contract on the synthetic atlas: MOs_left -> ALM_left (top) + MOs_rest_left (bottom)."""
    img2, nm2 = img.copy(), list(nm) + ["ALM_left", "MOs_rest_left"]
    par = img == nm.index("MOs_left")
    img2[par & (np.arange(10)[:, None] < 5)] = len(nm2) - 2
    img2[par & (np.arange(10)[:, None] >= 5)] = len(nm2) - 1
    return img2, nm2


@pytest.fixture
def atlas(scripts, monkeypatch):
    import encoder_split_rows as esr
    import frozen_area_maps as fam
    calls = []
    real_paint = fam.paint

    def paint(img, nm, values):
        calls.append((list(nm), dict(values)))
        return real_paint(img, nm, values)
    monkeypatch.setattr(fam, "area_image", lambda label: _img())
    monkeypatch.setattr(fam, "paint", paint)
    monkeypatch.setattr(esr, "split_image", _split_image)
    return calls


def _split_painted(calls):
    return [c for c in calls if "ALM_left" in c[0] and "ALM_left" in c[1] and "MOs_rest_left" in c[1]]


# ------------------------------------------------------------------------------------------------ Bonferroni
def test_bonf_allen_counts_allen_areas_only(scripts):
    import encoder_split_rows as esr
    B = pd.DataFrame({"area": ["SSp-m_left", "MOs_left", "ALM_left", "MOs_rest_left", "SSp-m_left"],
                      "window": ["cue", "cue", "cue", "cue", "lick"], "p": [0.01, 0.02, 0.01, 0.4, 0.03]})
    np.testing.assert_array_equal(esr.bonf_allen(B, ["window"]), [0.02, 0.04, 0.02, 0.8, 0.03])
    S = B[B.area.isin(esr.SPLIT_AREAS)]                   # split rows alone, counted against the Allen table
    np.testing.assert_array_equal(esr.bonf_allen(S, ["window"], ref=B), [0.02, 0.8])
    A = B[~B.area.isin(esr.SPLIT_AREAS)]                  # no split rows: = the old nunique(area) count
    old = np.minimum(A.p * A.groupby("window").area.transform("nunique"), 1.0).to_numpy()
    np.testing.assert_array_equal(esr.bonf_allen(A, ["window"]), old)


# ------------------------------------------------------------------------------------------------ self-fit
def _fake_design(fe, extra=None):
    rng = np.random.default_rng(3)
    groups = {}
    for v, g in fe.FGROUP.items():
        groups.setdefault(g, v)
    vars_ = list(groups.values())[:3]
    col_var = np.array([v for v in vars_ for _ in range(2)])
    n_tr, per = 60, 20
    n = n_tr * per
    X = rng.normal(size=(n, len(col_var)))
    A = X @ rng.normal(size=(len(col_var), 3)) + rng.normal(size=(n, 3))
    Rd = A if extra is None else np.column_stack([A, A[:, :len(extra["names"])]])   # ROI = copies of the Allen targets
    names = ["SSp-m_left", "MOs_left", "cortex"] + ([] if extra is None else list(extra["names"]))
    pos = np.array(["close_ipsi", "far_contra", "close_mid"] * (n_tr // 3))
    first = np.arange(n_tr) * per + 6
    return {"X": X, "Rd": Rd, "area_names": names, "col_var": col_var, "frame_trial": np.repeat(np.arange(n_tr), per),
            "epoch": "chronic", "day": 30, "n_trials": n_tr, "pos_order": sorted(set(pos)),
            "pos_counts": {p: int((pos == p).sum()) for p in set(pos)}, "cue_f": np.arange(n_tr) * per + 2,
            "first_f": first, "fs": 10.0, "pos": pos}, groups


def test_self_fit_roi_targets_keep_allen_identical(scripts, monkeypatch, tmp_path):
    from wfield_local import frozen_encoder as fe
    import session_vs_frozen as svf
    d0, groups = _fake_design(fe)
    (tmp_path / "fit").mkdir()
    np.savez(tmp_path / "fit" / "PS92_joint.npz", col_var=d0["col_var"])
    monkeypatch.setattr(fe, "_templates_of", lambda *a, **k: {})
    monkeypatch.setattr(fe, "session_design", lambda label, templates, extra_targets=None: _fake_design(fe, extra_targets)[0])
    drop = next(f"no_{g}" for g in list(groups)[1:3] if f"no_{g}" in fe._DROP)
    variants = ("joint", drop)
    extra = {"names": ["ALM_left", "MOs_rest_left"], "P": None}
    base = svf.self_fit_job(("PS92_0901", str(tmp_path), variants))
    roi = svf.self_fit_job(("PS92_0901", str(tmp_path), variants, extra))
    assert base["areas"] == ["SSp-m_left", "MOs_left", "cortex"]
    assert roi["areas"] == base["areas"] + extra["names"] and roi["roi_targets"] == extra["names"]
    assert roi["alphas"] == base["alphas"]
    for v in variants:
        for w, dd in base["r2_musall"][v].items():
            for k, vals in dd.items():
                assert roi["r2_musall"][v][w][k][:3] == vals          # Allen R2 EXACTLY unchanged
                # ROI columns = copies of the first two Allen targets, fitted with the held penalties -> same R2
                np.testing.assert_allclose(roi["r2_musall"][v][w][k][3:], vals[:2], rtol=0, atol=1e-10)


# ------------------------------------------------------------------------------------------------ svf --maps
def _remap_tables(split: bool):
    rows_s, rows_f = [], []
    areas = ["SSp-m_left", "MOs_left", "cortex"] + (["ALM_left", "MOs_rest_left"] if split else [])
    rng = np.random.default_rng(0)
    for an in AN:
        for ep, ns in (("pre", 3), ("acute", 2), ("chronic", 2)):
            for k in range(ns):
                lab = f"{an}_{ep}{k}"
                for w in ("cue", "lick"):
                    for ar in areas:
                        for v in ("joint", "no_task"):
                            # Allen values do not depend on `split` (same generator order for them)
                            s_, f_ = rng.normal(0.3, 0.05), rng.normal(0.25, 0.05)
                            if ar in ("ALM_left", "MOs_rest_left") and not split:
                                continue
                            rows_s.append({"label": lab, "animal": an, "epoch": ep, "n_trials": 100, "window": w,
                                           "area": ar, "variant": v, "self": s_ + (0.1 if ep != "pre" else 0)})
                            rows_f.append({"label": lab, "window": w, "area": ar, "variant": v, "frozen": f_})
    return pd.DataFrame(rows_s), pd.DataFrame(rows_f)


def _allen_only(S, F):
    keep = lambda t: t[~t.area.isin(["ALM_left", "MOs_rest_left"])].reset_index(drop=True)  # noqa: E731
    return keep(S), keep(F)


def test_svf_maps_paint_split_rows_allen_unchanged(atlas, monkeypatch, tmp_path):
    import session_vs_frozen as svf
    S, F = _remap_tables(split=True)
    S0, F0 = _allen_only(S, F)
    outs = {}
    for kind, (s_, f_, split) in {"allen": (S0, F0, None), "split": (S, F, (Path("pix"), Path("fit")))}.items():
        Fa, Fs = f_[~f_.area.isin(["ALM_left", "MOs_rest_left"])], f_[f_.area.isin(["ALM_left", "MOs_rest_left"])]
        monkeypatch.setattr(svf, "self_r2", lambda sdir, measure="frame", areas=None, s_=s_: s_)
        monkeypatch.setattr(svf, "frozen_r2", lambda run, v, measure="frame", areas=None, Fa=Fa: Fa[Fa.variant == v])
        monkeypatch.setattr(svf, "frozen_split_r2", lambda pix, fit, v, measure="frame", Fs=Fs: Fs[Fs.variant == v])
        out = tmp_path / kind
        (out / "sessions").mkdir(parents=True)
        atlas.clear()
        svf.area_maps(tmp_path, out, ["joint", "no_task"], split=split)
        outs[kind] = pd.read_csv(out / "remap_area.csv")
        assert (out / "figures" / "remap_area_maps_cue.png").exists()
        assert bool(_split_painted(atlas)) == (kind == "split")
    a, s = outs["allen"], outs["split"]
    key = ["q", "window", "area", "epoch"]
    s_allen = s[~s.area.isin(["ALM_left", "MOs_rest_left"])].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a.sort_values(key).reset_index(drop=True), s_allen[a.columns], check_exact=True)
    sp = s[s.area.isin(["ALM_left", "MOs_rest_left"])]
    assert len(sp) and np.allclose(sp.p_bonf, np.minimum(sp.p * 2, 1.0))      # 2 Allen areas (cortex dropped)


# ------------------------------------------------------------------------------------------------ remap_checks check 2
def test_remap_checks_per_animal_split_painting(atlas, tmp_path):
    import remap_checks as rc
    S, F = _remap_tables(split=True)
    G = S.merge(F, on=["label", "window", "area", "variant"])
    U = G[G.variant == "joint"].assign(q="full", u=lambda d: d["self"] - d["frozen"])
    rc.per_animal_maps(U, tmp_path, "tag", split=None)
    assert not _split_painted(atlas)
    atlas.clear()
    rc.per_animal_maps(U, tmp_path, "tag", split=(Path("pix"), Path("fit")))
    assert _split_painted(atlas)
    assert (tmp_path / "figures" / "remap_per_animal_lick.png").exists()


# ------------------------------------------------------------------------------------------------ slide 94 --split
def test_peth_trial_maps_split_companions(atlas, monkeypatch, tmp_path):
    import frozen_peth_trial_maps as fpt
    tag = "gate-attempted_v7g3_t3d_spont_frA"
    run = tmp_path / f"frozen_encoder_{tag}"
    pix = tmp_path / "pix"
    for v in fpt.VARIANTS:
        (run / v / "sessions").mkdir(parents=True)
        (pix / v / "sessions").mkdir(parents=True)

    def fake(areas):
        def table(*args):
            v = args[-1]
            rng = np.random.default_rng(sum(map(ord, v)))
            rows = []
            for an in AN:
                for ep, ns in (("pre", 3), ("acute", 2), ("chronic", 2)):
                    for k in range(ns):
                        for w in ("cue", "lick"):
                            for meas in ("avg", "trial"):
                                for ar in areas:
                                    rows.append((f"{an}_{ep}{k}", an, ep, w, meas, ar, rng.normal(0.3, 0.05)))
            return pd.DataFrame(rows, columns=["label", "animal", "epoch", "window", "measure", "area", "r2"])
        return table
    monkeypatch.setattr(fpt, "E", tmp_path)
    monkeypatch.setattr(fpt, "table", fake(["SSp-m_left", "MOs_left"]))
    monkeypatch.setattr(fpt, "split_table", fake(["ALM_left", "MOs_rest_left"]))
    assert fpt.main(["--tag", tag, "--out", str(tmp_path / "a")]) == 0
    assert not _split_painted(atlas)
    atlas.clear()
    assert fpt.main(["--tag", tag, "--out", str(tmp_path / "b"), "--pix-run", str(pix), "--fit-run", "x"]) == 0
    assert _split_painted(atlas)
    # the Allen outputs are byte-identical with and without --split
    assert (tmp_path / "a" / "peth_trial_area.csv").read_bytes() == (tmp_path / "b" / "peth_trial_area.csv").read_bytes()
    for w in ("cue", "lick"):
        assert (tmp_path / "b" / "figures" / f"peth_trial_maps_{w}_split.png").exists()
        assert not (tmp_path / "a" / "figures" / f"peth_trial_maps_{w}_split.png").exists()
    sp = pd.read_csv(tmp_path / "b" / "peth_trial_area_split.csv")
    assert set(sp.area) == {"ALM_left", "MOs_rest_left"} and np.allclose(sp.p_bonf, np.minimum(sp.p * 2, 1.0))


# ------------------------------------------------------------------------------------------------ reembedding map
def test_reembedding_map_paints_split_rois_values_unchanged(atlas):
    import reembedding_test as rt
    img, nm = _img()
    names = ["SSp-m_left", "ALM_left", "MOs_rest_left"]
    v = np.array([0.1, 0.7, 0.2])
    img_s, nm_s = rt.split_map(img, nm, names)
    M = rt.area_values_map(img_s, nm_s, names, v)
    for n, x in zip(names, v):                            # each read-out area on its own territory, its own value
        assert np.all(M[img_s == nm_s.index(n)] == x)
    assert not np.isin(0.9, M)                            # nothing summed back into MOs (0.7 + 0.2)
    i0, n0 = rt.split_map(img, nm, ["SSp-m_left", "MOs_left"])        # no split basis: plain atlas
    assert i0 is img and n0 is nm
