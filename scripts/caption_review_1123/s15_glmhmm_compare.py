"""CC-B overnight 11-23, item 3b: six-position GLM-HMM (S15E / H / J, Results l.178) OLD (production v1_floor020) vs NEW
(refit on the adopted jaw rule, WIDEFIELD_JAW_CALL=3d_mouth + the cap jaw table; same design v2 / K5 / seeds / options).

    python scripts/caption_review_1123/s15_glmhmm_compare.py --old-gh DIR --new-gh DIR --old-eng DIR --new-eng DIR \
        --live-tests CSV --out-gh DIR --out-eng DIR --tag jaw3dmouth_ccB1123

Reads the two posterior tables (states_pooled_all_K5_v2.csv.gz: every trial's class = the GLM-HMM outcome input), the
two params json, and the two engagement folders (s4 / s5 / s6 of scripts/c2_engagement_tim). Writes (refuses to
overwrite), each file name ending in _<tag>.csv:
  out-gh/glmhmm_outcome_input_changes_<tag>.csv     outcome input old -> new per epoch x spout (fit set = August on)
  out-gh/glmhmm_state_agreement_<tag>.csv           per epoch: same argmax state / same role, old vs new
  out-gh/glmhmm_state_confusion_<tag>.csv           argmax state old x new (trial counts)
  out-gh/glmhmm_state_profiles_<tag>.csv            per state (old, new): role, P(lick) (= the figure's p_attempt mean
                                                    over spouts), P(jaw-only), P(no response), occupancy, self-transition
  out-gh/glmhmm_roles_<tag>.csv                     role-collapsed occupancy / P(lick) / transitions (S15E schematic;
                                                    same computation as make_engmove_4cam_mefirst.schematic)
  out-eng/glmhmm_grouped_occupancy_by_epoch_<tag>.csv  epoch x role session-mean occupancy (match all, as the live
                                                    table), old / new
  out-eng/glmhmm_occupancy_tests_old_vs_new_<tag>.csv  F1 occupancy tests (first95): live 4,000-draw table vs new
  out-eng/jaw_only_by_glmhmm_role_old_vs_new_<tag>.csv  s6 role shares (production model), old / new, near middle
                                                    excluded (other_spouts) and included (all)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = ["pre", "acute", "subacute", "chronic"]
OM = {"contact": "attempt", "incomplete": "attempt", "jaw_only": "jaw_only", "no_response": "no_response"}
ROLES = ["engaged", "partial", "disengaged"]


def write(df: pd.DataFrame, f: Path) -> None:
    if f.exists():
        raise SystemExit(f"refusing to overwrite {f}")
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    print("wrote", f)


def states(gh: Path) -> pd.DataFrame:
    from wfield_local import behavior_states as B
    T = pd.read_csv(gh / "states_pooled_all_K5_v2.csv.gz")
    T["epoch"] = T.label.map(B.session_epoch)
    pc = [f"p_state{k}" for k in range(5)]
    T["argmax"] = T[pc].to_numpy().argmax(1)
    return T


def roles_unmerged(gh: Path) -> dict:
    """The production role rule (behavior_states.glmhmm_roles on the MERGED 0+1 table), mapped back to the 5 states."""
    import os
    from wfield_local import behavior_states as B
    os.environ["GLMHMM_OUT"] = str(gh)
    G = B.glmhmm_table()
    r = B.glmhmm_roles(G)
    out = {}
    for k in range(5):
        out[k] = r[0] if k in B.GLMHMM_MERGE[0] else r[k]
    return out


def profiles(gh: Path, eng: Path, T: pd.DataFrame, tag: str) -> pd.DataFrame:
    d = json.loads((gh / "params_pooled_all_K5_v2.json").read_text())
    E = pd.read_csv(eng / "glmhmm_emission_by_state_spout.csv").groupby(["state", "role_merged"])[
        ["p_attempt", "p_jaw_only", "p_no_response"]].mean().reset_index()
    A = np.array(d["A"])
    rows = []
    for _, e in E.iterrows():
        k = int(e.state)
        w = T[f"p_state{k}"].to_numpy(float)
        rows.append({"model": tag, "state": k, "role": e.role_merged, "p_lick": e.p_attempt, "p_jaw_only": e.p_jaw_only,
                     "p_no_response": e.p_no_response, "occupancy": float(w.mean()), "self_transition": A[k, k],
                     "emp_attempt": float(np.average(T.cls.isin(["contact", "incomplete"]), weights=w)),
                     "emp_jaw_only": float(np.average(T.cls.eq("jaw_only"), weights=w)),
                     "emp_no_response": float(np.average(T.cls.eq("no_response"), weights=w)), "ll": d["ll"]})
    return pd.DataFrame(rows)


def role_schematic(P: pd.DataFrame, gh: Path, tag: str) -> pd.DataFrame:
    A = np.array(json.loads((gh / "params_pooled_all_K5_v2.json").read_text())["A"])
    roles, occ, pl = dict(zip(P.state, P.role)), dict(zip(P.state, P.occupancy)), dict(zip(P.state, P.p_lick))
    rows = []
    for ra in ROLES:
        sa = [k for k, r in roles.items() if r == ra]
        w = np.array([occ[k] for k in sa])
        w = w / w.sum()
        r = {"model": tag, "role_from": ra, "states": "+".join(map(str, sa)), "occupancy": sum(occ[k] for k in sa),
             "p_lick": float((np.array([pl[k] for k in sa]) * w).sum())}
        for rb in ROLES:
            sb = [k for k, x in roles.items() if x == rb]
            r[f"T_to_{rb}"] = float((w[:, None] * A[np.ix_(sa, sb)]).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("old-gh", "new-gh", "old-eng", "new-eng", "live-tests", "out-gh", "out-eng"):
        ap.add_argument(f"--{k}", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    t = a.tag
    O, N = states(a.old_gh), states(a.new_gh)
    m = O.merge(N, on=["label", "trial_id"], suffixes=("_old", "_new"), how="outer", indicator=True)
    print("trial match", m._merge.value_counts().to_dict())
    assert (m.position_old == m.position_new).all()
    m = m[m._merge == "both"].copy()
    m["epoch"] = m.epoch_old
    m["y_old"] = m.cls_old.map(OM).fillna("unobserved")
    m["y_new"] = m.cls_new.map(OM).fillna("unobserved")
    # 1. outcome input changes
    ch = m.groupby(["epoch", "position_old", "y_old", "y_new"], dropna=False).size().rename("n_trials").reset_index()
    ch = ch.rename(columns={"position_old": "position"})
    ch["changed"] = ch.y_old != ch.y_new
    write(ch, a.out_gh / f"glmhmm_outcome_input_changes_{t}.csv")
    # 2. state / role agreement
    ro, rn = roles_unmerged(a.old_gh), roles_unmerged(a.new_gh)
    m["role_old"], m["role_new"] = m.argmax_old.map(ro), m.argmax_new.map(rn)
    rows = []
    for ep, g in [("all_fit", m)] + [(e, m[m.epoch == e]) for e in EPS]:
        for sub, h in (("all", g), ("outcome_changed", g[g.y_old != g.y_new]), ("outcome_same", g[g.y_old == g.y_new])):
            rows.append({"epoch": ep, "trials": sub, "n_trials": len(h), "same_state": float((h.argmax_old == h.argmax_new).mean()),
                         "same_role": float((h.role_old == h.role_new).mean()),
                         **{f"{r}_old_to_other": int(((h.role_old == r) & (h.role_new != r)).sum()) for r in ROLES},
                         **{f"{r}_new_from_other": int(((h.role_new == r) & (h.role_old != r)).sum()) for r in ROLES}})
    write(pd.DataFrame(rows), a.out_gh / f"glmhmm_state_agreement_{t}.csv")
    cf = pd.crosstab(m.argmax_old, m.argmax_new).reset_index().rename(columns={"argmax_old": "state_old"})
    cf.columns = [str(c) if c == "state_old" else f"new_{c}" for c in cf.columns]
    write(cf, a.out_gh / f"glmhmm_state_confusion_{t}.csv")
    # 3. profiles + role schematic
    P = pd.concat([profiles(a.old_gh, a.old_eng, O, "old"), profiles(a.new_gh, a.new_eng, N, "new")])
    write(P, a.out_gh / f"glmhmm_state_profiles_{t}.csv")
    R = pd.concat([role_schematic(P[P.model == "old"], a.old_gh, "old"), role_schematic(P[P.model == "new"], a.new_gh, "new")])
    write(R, a.out_gh / f"glmhmm_roles_{t}.csv")
    # 4. grouped occupancy by epoch (session mean, match all; the live table's definition)
    occ = []
    for tag_, eng in (("old", a.old_eng), ("new", a.new_eng)):
        E = pd.read_csv(eng / "sessions_engagement.csv")
        for match in ("all", "first95"):
            e = E[(E.model == "production") & (E["match"] == match)]
            g = e.groupby("epoch")[ROLES].mean().reindex(EPS).reset_index()
            occ.append(g.assign(model=tag_, match=match, n_sessions=e.groupby("epoch").size().reindex(EPS).values))
    occ = pd.concat(occ).rename(columns={"partial": "running_not_trying", "disengaged": "disengaged_still"})
    write(occ, a.out_eng / f"glmhmm_grouped_occupancy_by_epoch_{t}.csv")
    # 5. F1 occupancy tests
    cols = ["metric", "epoch", "n_sessions", "n_pre_sessions", "boot_change", "boot_lo", "boot_hi", "p", "q_fdr", "star"]
    Lt = pd.read_csv(a.live_tests)
    Lt = Lt[Lt.family.str.startswith("F1 ") & (Lt["match"] == "first95")][cols]
    Nt = pd.read_csv(a.new_eng / "tests_ALL.csv")
    Nt = Nt[Nt.family.str.startswith("F1 ") & (Nt["match"] == "first95")][cols]
    tt = Lt.merge(Nt, on=["metric", "epoch"], suffixes=("_old", "_new"))
    tt["star_changed"] = tt.star_old != tt.star_new
    write(tt, a.out_eng / f"glmhmm_occupancy_tests_old_vs_new_{t}.csv")
    # 6. role share (s6, production model rows)
    S = []
    for tag_, eng in (("old", a.old_eng), ("new", a.new_eng)):
        s = pd.read_csv(eng / "jaw_only_by_glmhmm_role.csv")
        S.append(s[s.model == "production"].assign(version=tag_))
    S = pd.concat(S)
    k = ["match", "class", "epoch", "spout"]
    sw = S[S.version == "old"].drop(columns=["model", "version"]).merge(
        S[S.version == "new"].drop(columns=["model", "version"]), on=k, suffixes=("_old", "_new"))
    write(sw, a.out_eng / f"jaw_only_by_glmhmm_role_old_vs_new_{t}.csv")
    pd.set_option("display.width", 250)
    print(tt.round(4).to_string(index=False))
    print(sw[(sw.spout != "close_center") & (sw.epoch == "pooled")].round(3).to_string(index=False))
    print(R.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
