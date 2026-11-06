"""fr_pixel_significance: caption wording follows beta_maps.CORRECTION (max-statistic, not Bonferroni), and the
SUPPLEMENTAL --drift-control (late pre minus early pre, same test, epoch tests read back; Priya 11-06)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

fps = pytest.importorskip("fr_pixel_significance")
SCRIPTS = Path(fps.__file__).resolve().parent


def test_correction_label_follows_the_code():
    from wfield_local import beta_maps as bm
    assert bm.CORRECTION == "maxstat"
    lab = fps.correction_label()
    assert "max-statistic" in lab and "per panel" in lab and "Bonferroni" not in lab
    assert "Bonferroni" in fps.correction_label("bonferroni")


@pytest.mark.parametrize("name", ["fr_pixel_maps_sig.py", "variance_maps.py", "fr_direction.py"])
def test_fr_pixel_family_captions_do_not_say_bonferroni(name):
    """Every FR pixel-map caption draws the max-statistic test (`fr_pixel_significance.test`), so none may say Bonferroni."""
    assert "Bonferroni" not in (SCRIPTS / name).read_text(encoding="utf-8")


def test_main_figure_caption_has_no_bonferroni_literal():
    src = (SCRIPTS / "fr_pixel_significance.py").read_text(encoding="utf-8")
    body = src.split("def main(", 1)[1]
    assert "Bonferroni" not in body and "correction_label()" in body


def test_early_late_is_date_order_first_half_vs_rest():
    m = {lab: np.full((2, 2), i, float) for i, lab in enumerate(["PS92_0806", "PS92_0807", "PS92_0809", "PS92_0810",
                                                                  "PS93_0806", "PS93_0807", "PS93_0808"])}
    items = {"PS92": [(k, m[k]) for k in ("PS92_0810", "PS92_0806", "PS92_0809", "PS92_0807")],     # shuffled on purpose
             "PS93": [(k, m[k]) for k in ("PS93_0808", "PS93_0806", "PS93_0807")],
             "PS94": [("PS94_0806", m["PS92_0806"])]}                                              # one session: dropped
    early, late = fps.early_late(items)
    assert set(early) == set(late) == {"PS92", "PS93"}
    assert [x[0, 0] for x in early["PS92"]] == [0, 1] and [x[0, 0] for x in late["PS92"]] == [2, 3]
    assert [x[0, 0] for x in early["PS93"]] == [4] and [x[0, 0] for x in late["PS93"]] == [5, 6]       # 3 -> 1 v 2


def test_drift_overlap_and_animal_agreement():
    ep_sig = np.array([[True, True], [True, False]])
    ep_m = np.array([[0.2, 0.2], [-0.1, 0.3]])
    dr_sig = np.array([[True, False], [True, True]])
    dr_m = np.array([[0.05, 0.01], [0.02, 0.1]])
    ov = fps.drift_overlap(ep_sig, ep_m, dr_sig, dr_m)
    assert ov == dict(epoch_sig=3, epoch_sig_drift_same_sign=2, epoch_sig_drift_sig_same_sign=1)
    assert np.isnan(fps.drift_overlap(None, None, dr_sig, dr_m)["epoch_sig"])
    per = [dr_m, dr_m * 2, np.where(np.eye(2, dtype=bool), dr_m, -dr_m)]
    assert fps.animals_agree(dr_sig, per, dr_m) == 2                     # (1, 0) flips sign in one animal
    assert fps.bin_mean(np.arange(16.).reshape(4, 4), (2, 2), 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]


def test_drift_control_end_to_end(tmp_path, monkeypatch):
    """Synthetic: a late-pre rise in one corner, a chronic rise in the same corner. The drift test finds the corner (up),
    reads the chronic test back and counts its overlap; CSVs + the *_driftcontrol figure are written."""
    import combined_encoding_figures as cef
    monkeypatch.setattr(cef, "_outline", lambda meta: None)
    rng = np.random.default_rng(0)
    H = W = 64
    mask = np.ones((H, W), bool)
    corner = np.zeros((H, W), bool)
    corner[:32, :32] = True
    pre_days = ["0806", "0807", "0808", "0809", "0810", "0811"]
    rows, maps = [], {}
    for an in ("PS92", "PS93", "PS94", "PS95"):
        for k, d in enumerate(pre_days):
            lab = f"{an}_{d}"
            rows.append((lab, an, "pre"))
            base = 0.4 * corner * (k >= 3)
            maps[lab] = {(q, "cue"): base + rng.normal(0, 0.05, (H, W)) for q in fps.QS}
        for d in ("0920", "0922", "0924"):
            lab = f"{an}_{d}"
            rows.append((lab, an, "chronic"))
            maps[lab] = {(q, "cue"): 0.6 * corner + rng.normal(0, 0.05, (H, W)) for q in fps.QS}
    meta = pd.DataFrame(rows, columns=["label", "animal", "epoch"])
    out = tmp_path / "pixel_significance"
    (out / "figures").mkdir(parents=True)
    # the main run's chronic test, written the way main() writes it
    pre = {an: [np.where(mask, maps[r.label][("unique_position", "cue")], np.nan) for r in g.itertuples()]
           for an, g in meta[meta.epoch == "pre"].groupby("animal")}
    post = {an: [np.where(mask, maps[r.label][("unique_position", "cue")], np.nan) for r in g.itertuples()]
            for an, g in meta[meta.epoch == "chronic"].groupby("animal")}
    mm = np.nanmean([np.nanmean(post[a], 0) - np.nanmean(pre[a], 0) for a in pre], 0)
    for F, f in fps.FACTORS.items():
        sig, _n = fps.test(pre, post, f, mask)
        np.savez_compressed(out / f"sig_unique_position_cue_chronic_{F}px.npz", mean=mm.astype(np.float32), sig=sig)
    a = SimpleNamespace(windows=["cue"], zscore=False, mode="A")
    assert fps.drift_control(meta, maps, mask, out, a, lambda q, w: None) == 0
    S = pd.read_csv(out / "sig_bins_driftcontrol.csv")
    r = S[(S.q == "unique_position") & (S.px == 16) & (S.epoch == "chronic")].iloc[0]
    assert r.drift_up > 0 and r.drift_down == 0 and r.drift_sig_all_animals_same_sign == r.drift_sig
    assert r.sessions_early_late == "3v3/3v3/3v3/3v3"
    assert bool(r.epoch_test_found) and r.epoch_sig > 0
    assert r.epoch_sig_drift_same_sign == r.epoch_sig and 0 < r.epoch_sig_drift_sig_same_sign <= r.epoch_sig
    assert not S[(S.q == "full_fr") & (S.epoch == "chronic")].epoch_test_found.any()     # no main run for it: NaN
    assert np.isnan(S[(S.q == "full_fr") & (S.epoch == "chronic")].epoch_sig).all()
    assert (out / "figures" / "fr_pixel_significance_cue_modeA_driftcontrol.png").exists()
    assert (out / "sig_unique_position_cue_driftpre_16px.npz").exists()
    T = pd.read_csv(out / "session_means_driftcontrol.csv")
    assert len(T) == len(fps.QS) * len(meta) and T.days.notna().all()
