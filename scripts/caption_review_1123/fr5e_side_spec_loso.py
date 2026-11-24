"""Fig 5E caption review (Priya 11-23, item 24): sensitivity of side specificity to leave-one-session-out templates.

`face_rhythm_session_shape_tests.load_sessions` (the producer of the Fig 5E table) builds each animal's movement templates
from ALL its pre-stroke sessions, so a pre-stroke session is scored against a template that contains itself. This
script recomputes side_spec for the pre-stroke sessions with the scored session left out of the template (post-stroke
sessions are unchanged: they are never in the template), then repeats the side_spec cells of the 24-test lick family
(within-animal session-label shuffle, 10,000 shuffles, seed 0, two-sided; Holm over the 24 tests with the other 18
p values taken from the stored session_scalars.csv). Nothing in the producer's output folder is written.

    python fr5e_side_spec_loso.py <stored session_scalars.csv> <out_dir> [n_perm]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(stored: str, out_dir: str, n_perm: str = "10000") -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import face_rhythm_cohort_null as N
    import face_rhythm_session_shape_tests as st
    from wfield_local import analysis_kit as ak
    from wfield_local.face_rhythm import common as frc
    from wfield_local import face_rhythm_outputs as fro
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    fdir = frc.default_set_dir(set_name=fro.SET)
    print("set dir:", fdir)
    T, names, _home = st.load_sessions(fdir)
    S = frc.sessions(fdir, "proj")
    sp = {}
    for t in S[S.epoch == "pre"].itertuples():
        with np.load(t.file, allow_pickle=True) as z:
            have = list(z["power_positions"].astype(str))
            a = np.full((6, z["speed_response"].shape[1]), np.nan)
            for i, p in enumerate(frc.POSITIONS):
                if p in have:
                    a[i] = z["speed_response"][have.index(p)]
            sp[(t.animal, t.key)] = a
    T["side_spec_all"] = T.side_spec
    loso = []
    for r in T.itertuples():
        if r.unit not in st.OWN_SIDE or r.epoch != "pre":
            loso.append(r.side_spec)
            continue
        acc = [v for (an, k), v in sp.items() if an == r.animal and k != r.key]
        m = np.nanmean(acc, 0)
        loso.append(N.side_spec(r.load, m - np.nanmean(m, 0, keepdims=True), st.OWN_SIDE[r.unit]))
    T["side_spec"] = loso
    keep = T[T.unit.isin(st.LICK)][["key", "animal", "epoch", "unit", "recovered", "side_spec_all", "side_spec"]]
    keep.rename(columns={"side_spec": "side_spec_loso"}).to_csv(out / "side_spec_per_session_all_vs_loso.csv",
                                                                index=False)
    pre = keep[keep.epoch == "pre"]
    print(f"pre sessions: side_spec all-template mean {pre.side_spec_all.mean():.3f}, LOSO mean "
          f"{pre.side_spec.mean():.3f}; median per-session diff {np.median(pre.side_spec_all - pre.side_spec):.3f}")
    rows = []
    for tag, col in (("all_pre_template (as Fig 5E)", "side_spec_all"), ("LOSO_pre_template", "side_spec")):
        rng = np.random.default_rng(0)
        for ep in frc.POST_EPOCHS:
            for nm in st.LICK:
                d = T[(T.unit == nm) & T.recovered & T.epoch.isin(["pre", ep])]
                groups = [(g[col].to_numpy(float), (g.epoch == "pre").to_numpy()) for _, g in d.groupby("animal")]
                obs, p, n_an = st.shuffle_test(groups, "value", int(n_perm), rng, "two")
                rows.append(dict(template=tag, epoch=ep, unit=nm, delta=obs, p=p, n_animals=n_an))
    R = pd.DataFrame(rows)
    Rs = pd.read_csv(stored)
    other = Rs[(Rs.family == "lick pair") & (Rs.measure != "side_spec")]
    R["p_holm24"] = np.nan
    for _tag, g in R.groupby("template"):
        pv = np.concatenate([g.p.to_numpy(), other.p.to_numpy()])
        R.loc[g.index, "p_holm24"] = ak.holm(pv)[:len(g)]
    st_ss = Rs[Rs.measure == "side_spec"][["epoch", "unit", "delta", "p", "p_holm"]].rename(
        columns={"delta": "stored_delta", "p": "stored_p", "p_holm": "stored_p_holm"})
    R = R.merge(st_ss, on=["epoch", "unit"], how="left")
    R["stars"] = np.where(R.p_holm24 < .01, "**", np.where(R.p_holm24 < .05, "*", ""))
    R.to_csv(out / "side_spec_tests_all_vs_loso.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
