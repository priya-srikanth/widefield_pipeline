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
