"""Check (11-18): are contralesional lick runs shorter because licking switches between lick groups more often?
Engaged trials (GLM-HMM), response window cue .. cue + min(stop, 2 s): lick-group switches per second of lick frames and
the mean run length of the TARGET-side lick group, by spout; change vs own pre via orofacial_cohort.epoch_tables.
Uses behavior_states for every definition. Scratch; locanmf env."""
import numpy as np
import pandas as pd

from wfield_local import analysis_kit as ak
from wfield_local import behavior_states as bs
from wfield_local import orofacial_cohort as oc


def one(item):
    from wfield_local import trial_classes as tc
    lab, eng, groups = item
    T = tc.session_trials(lab)
    t, _f, _b, z = bs.load_session(lab)
    gn = list(groups)
    g = np.where(z >= 0, bs._group_lut(groups)[np.maximum(z, 0)], -1)
    lick = [gn.index(x) for x in bs.LICK_GROUPS]
    lat = bs.laterality()[lab.split("_")[0]]
    rows = []
    for r in T.itertuples():
        if r.trial_id not in eng:
            continue
        a = r.cue_daq
        b = a + min((r.stop_ms if np.isfinite(r.stop_ms) else 2000) / 1000.0, bs.RESP_MAX_S)
        w = g[np.searchsorted(t, a):np.searchsorted(t, b)]
        lw = w[np.isin(w, lick)]
        if len(lw) < 5:
            continue
        runs = [q[0] for q in np.split(lw, np.flatnonzero(np.diff(lw) != 0) + 1)]
        pos = oc.pos_key(r.position, lat)
        side = {"ipsi": "lick_ipsi", "mid": "lick_central", "contra": "lick_contra"}[pos.split("_")[1]]
        k = gn.index(side)
        tr = [len(q) for q in np.split(w, np.flatnonzero(np.diff(w) != 0) + 1) if q[0] == k]
        rows.append({"label": lab, "pos": pos, "switch_per_s": (len(runs) - 1) / (len(lw) / 25.0),
                     "target_run_s": np.mean(tr) / 25.0 if tr else np.nan,
                     "target_share": float(np.mean(lw == k))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    T = bs.glmhmm_table()
    eng = bs.glmhmm_engaged(T)
    groups = bs.state_groups()
    labs = sorted(eng)
    res, fail = ak.fan_sessions([(lab, eng[lab], groups) for lab in labs], one, key=lambda it: it[0])
    D = pd.concat([r for _i, r in res if r is not None and len(r)])
    D["animal"], D["epoch"] = D.label.str.split("_").str[0], D.label.map(bs.session_epoch)
    D = D[D.epoch.notna()]
    for pos in ("close_contra", "far_contra", "close_ipsi", "far_ipsi", "close_mid", "far_mid"):
        S = D[D.pos == pos].groupby(["label", "animal", "epoch"])[["switch_per_s", "target_run_s", "target_share"]].mean().reset_index()
        _A, DD = oc.epoch_tables(S, metrics=["switch_per_s", "target_run_s", "target_share"])
        DD["sig"] = (DD.lo > 0) | (DD.hi < 0)
        print("##", pos)
        for m, gg in DD.groupby("metric", sort=False):
            print("  ", m.ljust(14), " ".join(f"{r.epoch[:3]} {r.delta:+.3f}[{r.lo:+.3f},{r.hi:+.3f}]{'*' if r.sig else ' '}"
                                         for r in gg.itertuples()))
