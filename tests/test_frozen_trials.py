"""FROZEN encoder per-trial outputs (Priya 11-05, REBUILD_CODE 8.52): `<label>_trials.npz` in the combined encoder's
layout (`combined_encoding._TrialScorer`), readable by `wfield_local.encoder_trials`; classes joined from the attempts
table; the post hoc --trials-only mode. Synthetic, no share."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wfield_local import combined_encoding as ce
from wfield_local import encoder_trials as et
from wfield_local import frozen_encoder as fe

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_encoder_trials as tet  # noqa: E402


def _design(seed=0):
    """A frozen-style session: engaged trials, their frame rows (strobe - 0.5 .. stop + 1 s), cue / first-lick frames."""
    x = tet._inputs(n_trials=48, seed=seed)
    B = x["behaviour"]
    ft = x["imaging"][1]
    T = B.trials
    E = T[T.engaged & np.isfinite(T.cue_s)].reset_index(drop=True)
    rows = np.zeros(len(ft), bool)
    for a, b in zip(E.strobe_s - 0.5, E.stop_s + 1.0):
        rows |= (ft >= a) & (ft <= b)
    ridx = np.flatnonzero(rows)
    ftr = ft[ridx]
    lk = np.sort(np.asarray(B.lick_on_s, float))
    fl, _src = fe.first_lick_s(B, E)
    first_f = np.full(len(E), -1)
    for i in np.flatnonzero(np.isfinite(fl)):
        first_f[i] = int(np.searchsorted(ftr, fl[i]))
    rng = np.random.default_rng(seed + 3)
    K = 4
    Rd = rng.normal(size=(len(ridx), K))
    d = {"E": E, "ft": ft, "ridx": ridx, "cue_f": np.searchsorted(ftr, E.cue_s.to_numpy(float)), "first_f": first_f,
         "lk_all": lk, "contact_s": np.asarray(B.contact_s, float), "fs": float(1.0 / np.median(np.diff(ft)))}
    return d, Rd


def _attempts(E, label="PS93_0814"):
    cls = np.where(np.arange(len(E)) % 4 == 0, "incomplete", "hit")
    return pd.DataFrame({"label": label, "trial_id": E.trial_id, "cls": cls, "cls_raw": cls, "attempted": True,
                         "canonical": True})


def _write(tmp_path, monkeypatch, mode="full"):
    monkeypatch.setenv(fe.TRIALS_ENV, mode)
    d, Rd = _design()
    off = np.arange(int(ce.PETH_S[0] * d["fs"]), int(ce.PETH_S[1] * d["fs"]) + 1)
    fits = {"joint": 0, "no_target": 0, "no_lick": 0, "joint|drop:close_mid": 0, "lick_only": 0}
    tv = fe.trial_variants(fits)
    assert tv == ["joint", "no_target", "no_lick"]                # TRIAL_VARIANTS order; drop-spout / alone left out
    rng = np.random.default_rng(7)
    preds = {v: Rd + rng.normal(scale=0.5 + i, size=Rd.shape) for i, v in enumerate(tv)}
    sc = fe._trial_scorer(d, Rd, off, len(tv))
    for i, v in enumerate(tv):
        sc.model(i, "full" if v == "joint" else v, preds[v], None)
    A = tmp_path / "trial_attempts.csv"
    _attempts(d["E"]).to_csv(A, index=False)
    from wfield_local import lp_licks
    monkeypatch.setattr(lp_licks, "attempts_path", lambda *a, **k: A)
    info = fe.write_trials("PS93_0814", tmp_path, d, Rd, {"areas": ["a", "b", "c", "cortex"]}, sc, tv, "loso")
    return d, Rd, off, preds, tv, info


def test_trials_file_layout_and_sse_as_the_encoder(tmp_path, monkeypatch):
    d, Rd, off, preds, tv, info = _write(tmp_path, monkeypatch)
    p = Path(info["path"])
    assert p == tmp_path / "joint" / "sessions" / "PS93_0814_trials.npz" and info["MB"] > 0
    z = et.load(p)
    assert list(z["tr_models"]) == ["full", "no_target", "no_lick"] and list(z["tr_variants"]) == tv
    E = d["E"]
    n = len(E)
    for a in ("cue", "lick"):
        assert z[f"sse_bins_{a}"].shape[:2] == (3, n) and z[f"sst_bins_{a}"].shape[0] == n
    # binned SSE of trial j = the encoder's definition: (data - prediction)^2 over its aligned frames, per bin
    idx, ok = ce.trial_rows(z["t_cue_row"], z["t_row_lo"], z["t_row_hi"], z["off"])
    b = z["bin_of_off"]
    j, q = 5, int(b[len(b) // 2])
    m = ok[j] & (b == q)
    want = ((Rd[idx[j][m]] - preds["no_target"][idx[j][m]]) ** 2).sum(0)
    np.testing.assert_allclose(z["sse_bins_cue"][1, j, q], want, rtol=1e-5)
    np.testing.assert_allclose(z["rd_mean"], Rd.mean(0))
    # classes joined, held-out flag, the encoder's frames (data + joint held-out cv)
    assert (z["table"].cls == np.where(np.arange(n) % 4 == 0, "incomplete", "hit")).all()
    assert z["t_heldout"].all() and (z["t_frozen_from"] == "loso").all() and z["t_fitted"].all()
    assert list(z["parts"]) == ["data", "cv"] and z["frames"].dtype == np.float16
    np.testing.assert_allclose(z["frames"][1].astype(float), preds["joint"], atol=2e-2 * np.abs(preds["joint"]).max())


def test_reader_class_summary_works_on_the_frozen_file(tmp_path, monkeypatch):
    _d, _Rd, _off, _p, _tv, info = _write(tmp_path, monkeypatch)
    z = et.load(info["path"])
    S = et.class_summary(z, by="cls", area=-1)
    assert list(S.cls) == ["hit", "incomplete"] and S.n.sum() == len(z["table"])
    assert np.isfinite(S.cvR2).all() and np.isfinite(S.trial_R2).all()


def test_sse_mode_has_no_frames(tmp_path, monkeypatch):
    _d, _Rd, _off, _p, _tv, info = _write(tmp_path, monkeypatch, mode="sse")
    z = et.load(info["path"])
    assert "frames" not in z and "sse_bins_cue" in z and "sse_trial_bins_lick" in z
    S = et.class_summary(z, by="cls", area=-1)
    assert np.isfinite(S.trial_R2).all()                            # per-class trial R2 without frames


def test_trials_mode_env(monkeypatch):
    monkeypatch.delenv(fe.TRIALS_ENV, raising=False)
    assert fe.trials_mode() == "sse"
    monkeypatch.setenv(fe.TRIALS_ENV, "off")
    assert fe.trials_mode() == "off" and fe.trial_variants({"no_target": 0}) == []
    monkeypatch.setenv(fe.TRIALS_ENV, "bogus")
    with pytest.raises(ValueError):
        fe.trials_mode()


def test_trials_only_needs_saved_statistics(tmp_path, monkeypatch):
    monkeypatch.delenv(fe.LICKS_ALL_ENV, raising=False)
    with pytest.raises(SystemExit, match="saved statistics"):
        fe.main(["--trials-only", "--tag", "gate-attempted_v7g3_t3d", "--out-root", str(tmp_path), "PS93_0814"])
    with pytest.raises(SystemExit, match="Allen-area"):
        fe.main(["--trials-only", "--target", "basis", "--tag", "gate-attempted_v7g3_t3d", "--out-root", str(tmp_path)])
    assert not list((tmp_path / "frozen_encoder_gate-attempted_v7g3_t3d").rglob("*.npz"))


def test_load_trial_parts_skips_trials_files(tmp_path, monkeypatch):
    _d, _Rd, _off, _p, _tv, info = _write(tmp_path, monkeypatch)
    assert fe.load_trial_parts(tmp_path).empty                       # the trials file is not a session file


def test_binned_class_trial_r2_equals_the_frame_path(tmp_path, monkeypatch):
    """Mode full: the per-class trial-to-trial R2 from the frames (`class_full_fine`) = the binned terms written for
    mode sse (trial means removed within spout x class), every class, both alignments."""
    _d, _Rd, _off, _p, _tv, info = _write(tmp_path, monkeypatch, mode="full")
    z = et.load(info["path"])
    zb = {k: v for k, v in z.items() if k != "frames"}
    for al in ("cue", "lick"):
        a = et.class_summary(z, by="cls", area=-1, align=al)
        b = et.class_summary(zb, by="cls", area=-1, align=al)
        np.testing.assert_allclose(a.trial_R2, b.trial_R2, rtol=1e-4)
        np.testing.assert_allclose(a.cvR2, b.cvR2)
    assert z["sse_trial_bins_cue"].shape == z["sst_trial_bins_cue"].shape == z["sst_bins_cue"].shape
