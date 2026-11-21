"""`behavior_states`: GLM-HMM roles come from the states' own profiles (no hard-coded indices), the syllable groups keep
the display order, and the push / walking thresholds come from configs/defaults.yaml."""
import numpy as np
import pandas as pd

from wfield_local import behavior_states as bs


def _table(profile, n=400, seed=0):
    """Synthetic GLM-HMM trial table: state k emits 'contact' with prob attempt_k, else 'no_response'."""
    rng = np.random.default_rng(seed)
    K = len(profile)
    rows = []
    for k, att in enumerate(profile):
        for _ in range(n):
            p = np.full(K, 0.01)
            p[k] = 1 - 0.01 * (K - 1)
            r = {"cls": "contact" if rng.random() < att else "no_response", "run_frac_precue": 0.0,
                 "run_frac_iti": 0.0, "trial_id": len(rows), "label": "PS92_20260806"}
            r.update({f"p_state{j}": p[j] for j in range(K)})
            rows.append(r)
    return pd.DataFrame(rows)


def test_roles_follow_profiles_not_indices():
    # disengaged first, engaged last: roles must follow the profiles
    T = _table([0.05, 0.5, 0.99, 0.98])
    roles = bs.glmhmm_roles(T)
    assert roles == {0: "disengaged", 1: "partial", 2: "engaged", 3: "engaged"}
    eng = bs.glmhmm_engaged(T)["PS92_20260806"]
    arg = T[[f"p_state{k}" for k in range(4)]].to_numpy().argmax(1)
    assert eng == set(T.trial_id[np.isin(arg, [2, 3])])


def test_thresholds_come_from_config():
    from wfield_local import config
    seg = config.defaults()["segmentation"]
    assert bs.walk_threshold() == float(seg["running"]["thresh_speed_mm_s"])
    pp = bs.push_params()
    assert pp["thresh"] == float(seg["push"]["thresh_back_mm_s"])
    assert pp["cue"][0] < 0 < pp["cue"][1] and pp["control"][1] <= pp["cue"][0]


def test_group_display_order(tmp_path, monkeypatch):
    from wfield_local import arhmm_reels
    d = tmp_path / "reels"
    d.mkdir()
    pd.DataFrame({"state": [2, 5, 9, 4], "group": ["lick_contra", "quiet", "new_group", "walk_slow"]}).to_csv(
        d / "state_labels.csv", index=False)
    monkeypatch.setattr(arhmm_reels, "reels_dir", lambda tag: d)
    G = bs.state_groups("x")
    assert list(G) == ["quiet", "lick_contra", "walk_slow", "new_group"]       # display order, unknown last
    assert G["quiet"] == [5]


# --- 11-18 additions: state merge, colors, cue-window statistics, walk window, outcome rows, session time ---

def _epoch_of(label):
    return {"0806": "pre", "0807": "pre", "0820": "acute", "0821": "acute"}.get(label.split("_")[1][-4:])


def test_glmhmm_table_merges_posteriors(tmp_path, monkeypatch):
    from wfield_local import glm_hmm
    rows = []
    for i, (lab, cls) in enumerate([("PS92_20260806", "contact"), ("PS93_20260820", "no_response"),
                                    ("PS94_20260806", "unknown"), ("PS95_20260101", "contact")]):
        p = np.array([.1, .2, .3, .25, .15]) if i % 2 else np.array([.4, .4, .1, .05, .05])
        rows.append({"trial_id": i, "position": "close_L", "cls": cls, "label": lab,
                     **{f"p_state{k}": p[k] for k in range(5)}})
    pd.DataFrame(rows).to_csv(tmp_path / "states_pooled_all_K5_v2.csv.gz", index=False)
    monkeypatch.setattr(glm_hmm, "out_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(bs, "session_epoch", _epoch_of)
    monkeypatch.setattr(bs, "laterality", lambda: {})
    T = bs.glmhmm_table(K=5, design="v2")
    assert list(T.trial_id) == [0, 1]                          # 'unknown' class and the no-epoch session dropped
    assert "p_state1" not in T and np.allclose(T.p_state0, [.8, .3])
    assert np.allclose(T.filter(like="p_state").sum(1), 1)     # still a distribution
    R = bs.glmhmm_table(K=5, design="v2", merge=False)
    assert {"p_state0", "p_state1"} <= set(R) and np.allclose(R.p_state0, [.4, .1])


def test_merged_state_label_and_stable_colors():
    assert bs.glmhmm_state_label(0) == "0+1" and bs.glmhmm_state_label(3) == "3"
    full = {0: "engaged", 1: "engaged", 2: "engaged", 3: "partial", 4: "disengaged"}
    merged = {0: "engaged", 2: "engaged", 3: "partial", 4: "disengaged"}
    cf, cm = bs.glmhmm_state_colors(full), bs.glmhmm_state_colors(merged)
    assert all(cm[k] == cf[k] for k in merged)                 # a state keeps its color with or without the merge
    assert len(set(cf.values())) == 5


def test_cue_window_stats_finds_the_changed_window(monkeypatch):
    monkeypatch.setattr(bs, "session_epoch", _epoch_of)
    t = np.round(np.arange(-2, 4.5, 0.1), 2)
    early = (t >= 0) & (t < 1)
    rng = np.random.default_rng(1)
    per = {}
    for a in ("PS92", "PS93", "PS94", "PS95"):
        for day, ep in (("20260806", "pre"), ("20260807", "pre"), ("20260820", "acute"), ("20260821", "acute")):
            v = 1 + 0.01 * rng.standard_normal((1, t.size))
            if ep == "acute":
                v[0, early] *= 2                                   # doubled in the early window only
            per[f"{a}_{day}"] = v
    S, A = bs.cue_window_stats(per, [0], ["x"], t, scale="log2")
    s = S[S.epoch == "acute"].set_index("window")
    assert abs(s.loc["early", "point"] - 1) < 0.05 and s.loc["early", "sig"] in ("*", "**")
    assert abs(s.loc["pre-cue", "point"]) < 0.05 and s.loc["late", "n_animals"] == 4
    assert S[S.epoch == "chronic"].point.isna().all()          # no chronic sessions -> no estimate, no mark
    assert set(A.animal) == {"PS92", "PS93", "PS94", "PS95"}


def test_walk_window_from_config_after_cue():
    from wfield_local import config
    a, b = bs.walk_window()
    assert [a, b] == [float(x) for x in config.defaults()["segmentation"]["walk"]["window_s"]] and 0 <= a < b


def test_trial_outcome_rows():
    assert bs.trial_outcome("contact") == bs.trial_outcome("incomplete") == "working"
    assert bs.trial_outcome("jaw_only") == "jaw_only" and bs.trial_outcome("no_response") == "no_response"
    assert bs.trial_outcome("unknown") is None


def test_add_session_time_keeps_rows_and_index():
    T = pd.DataFrame({"label": ["a", "a", "b"], "trial_id": [2, 1, 1]}, index=[10, 11, 12])
    trials = pd.DataFrame({"label": ["a", "a", "b"], "trial_id": [1.0, 2.0, 1.0], "t_min": [5., 6., 7.]})
    out = bs.add_session_time(T, trials)
    assert list(out.index) == [10, 11, 12] and list(out.t_min) == [6., 5., 7.]
