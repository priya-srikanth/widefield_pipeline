"""How much SPOUT information does each frozen-encoder regressor group carry? (Priya 10-22: "ensure our variable groups
appropriately include or exclude general directionless movement vs direction-specific movement vs no movement".)

The frozen groups (`wfield_local.frozen_encoder.FGROUP`) are defined by what each signal measures; this MEASURES it.
Per pre-stroke session: each trial's regressor values averaged over the post-lick window (first post-cue lick -0.1 ..
+1.0 s; trials without a lick: cue .. +1.1 s), per group and per variable, then a multinomial logistic decoder of the
spout (six classes, chance 1/6) with block cross-validation (GroupKFold over ~6-trial position blocks, as the
imaging decoders). "Directionless" movement should sit near chance; direction-specific movement and task near ceiling.
Accuracy above chance in a directionless variable = target information it carries indirectly (e.g. wider jaw
opening for far spouts), which the encoder will credit to that group.

Usage (same env as the frozen run): python scripts/frozen_group_leakage.py [LABELS ...] [--jobs 4]
Default labels: two August pre sessions per animal. Output: <E>/qc/frozen_group_leakage.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def one(label: str) -> list[dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from wfield_local import frozen_encoder as fe
    d = fe.session_design(label)
    X, fs = d["X"], d["fs"]
    pos = d["pos"]
    start = np.where(d["first_f"] >= 0, d["first_f"] - int(round(0.1 * fs)), d["cue_f"])
    span = int(round(1.1 * fs))
    feats = np.stack([np.nanmean(X[max(s, 0):max(s, 0) + span], 0) for s in start])     # (trials, columns)
    blocks = np.cumsum(np.r_[1, pos[1:] != pos[:-1]])                                   # position blocks
    sets = {**{f"group:{g}": [v for v in fe.FGROUP if fe.FGROUP[v] == g] for g in fe.FGROUPS},
            **{f"var:{v}": [v] for v in fe.FGROUP}}
    rows = []
    for name, vs in sets.items():
        m = np.isin(d["col_var"], vs)
        F = feats[:, m]
        F = F[:, np.nanstd(F, 0) > 1e-12]
        if F.shape[1] == 0:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        k = min(5, len(np.unique(blocks)))
        acc = cross_val_score(clf, np.nan_to_num(F), pos, groups=blocks, cv=GroupKFold(k)).mean()
        rows.append({"label": label, "animal": d["animal"], "set": name, "n_cols": int(F.shape[1]),
                     "n_trials": len(pos), "acc": float(acc), "chance": 1.0 / len(np.unique(pos))})
    return rows


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import config, epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    labels = a.labels
    if not labels:
        pre = sorted(x for x in ce.sessions_available() if x[-4:] >= "0801" and epochs.epoch_of(x) == "pre")
        labels = []
        for an in sorted({config.animal_of(x) for x in pre}):
            mine = [x for x in pre if config.animal_of(x) == an]
            labels += [mine[0], mine[len(mine) // 2]] if len(mine) > 1 else mine
    got, fail = ak.fan_sessions(labels, one, jobs=a.jobs)
    for it, err in fail:
        print("FAILED", it, err)
    T = pd.DataFrame([r for _, rr in got for r in rr])
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc"
    assert_writable(out)
    T.to_csv(out / "frozen_group_leakage.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(T.pivot_table(index="set", columns="animal", values="acc").round(2).to_string())
    print(f"chance {T.chance.iloc[0]:.3f}\n-> {out / 'frozen_group_leakage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
