"""Score cam1_licks.mask_detached (bridged version) on the blind-review labels and its blast radius (scratch, 11-18)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "C:/Users/SabatiniLab/Github/wf_docbranches")
sys.path.insert(0, sys.argv[1])


def one(s, L):
    from scripts.cam1_licks import PROD
    from wfield_local import cam1_licks as C1, lp_licks, o2_inference as oi
    from wfield_local import orofacial_clean as OC
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    a, d = s.split("_")
    c1 = oi.session_windows(rv, PROD["cam1"], a, d, "cam1")
    if c1 is None:
        return None
    ix, flat = c1[0].reset_index(drop=True), c1[1].reset_index(drop=True)
    _f, m = C1.mask_detached(flat, ix)
    ok = flat.tongue_likelihood.to_numpy(float) >= float(OC.params("tongue", "cam1")["lk_thr"])
    tid, t = ix.trial_id.to_numpy(), ix.t_ms.to_numpy(float)
    lab = []
    for r in L.itertuples():
        k = np.flatnonzero((tid == r.trial_id) & (np.abs(t - r.t_ms) <= 8) & ok)
        lab.append({"id": r.id, "n_ok": len(k), "masked": bool(m[k].any()) if len(k) else np.nan,
                    "masked_all": bool(m[k].all()) if len(k) else np.nan})
    sd, f4, _ = lp_licks._inputs(f"{a}_{d[4:]}", rv)
    Lk = pd.read_csv(f4, usecols=["trial_id", "on_ms", "off_ms"])
    in4 = np.zeros(len(t), bool)
    for tr, g in Lk.groupby("trial_id"):
        k = np.flatnonzero(tid == tr)
        for on, off in zip(g.on_ms, g.off_ms):
            in4[k[(t[k] >= on) & (t[k] <= off)]] = True
    blast = {"session": s, "conf": int(ok.sum()), "masked": int(m.sum()), "masked_in_cam4_lick": int((m & in4).sum()),
             "conf_in_cam4_lick": int((ok & in4).sum())}
    if s == "PS93_20260814":
        k = tid == 540
        blast["t540_conf"], blast["t540_masked"] = int((ok & k).sum()), int((m & k).sum())
    return {"lab": lab, "blast": blast}


class W:
    def __init__(self, L): self.L = L
    def __call__(self, s): return one(s, self.L[self.L.session == s])


if __name__ == "__main__":
    from drop_features import labels
    from wfield_local import analysis_kit as ak
    L = labels()
    B = pd.read_csv(f"{sys.argv[1]}/mask_blast.csv").session.tolist()
    S = sorted(set(L.session) | set(B))
    res, fail = ak.fan_sessions(S, W(L), jobs=6)
    for s, e in fail:
        print("FAIL", s, e)
    lab = pd.DataFrame([x for _s, v in res if v for x in v["lab"]])
    O = L.merge(lab, on="id")
    O = O[~O.stratum.str.contains("null") & O.answer.isin(["lips", "out", "none", "droplet", "other"]) & (O.n_ok > 0)]
    O["y"] = np.where(O.answer.isin(["lips", "out"]), "tongue", "no_tongue")
    print(pd.crosstab(O.y, O.masked).to_string())
    print(pd.crosstab(O.answer, O.masked).to_string())
    Bk = pd.DataFrame([v["blast"] for _s, v in res if v and v["blast"]["session"] in B or (v and v["blast"]["session"] == "PS93_20260814")])
    Bk.to_csv(f"{sys.argv[1]}/mask_blast_v2.csv", index=False)
    tot = Bk[["conf", "masked", "masked_in_cam4_lick", "conf_in_cam4_lick"]].sum()
    print(f"\nblast ({len(Bk)} sessions): masked {tot.masked} / {tot.conf} ({tot.masked / tot.conf:.2%}); inside a cam4 lick "
          f"{tot.masked_in_cam4_lick} ({tot.masked_in_cam4_lick / max(tot.masked, 1):.1%} of masked; "
          f"{tot.masked_in_cam4_lick / tot.conf_in_cam4_lick:.3%} of cam4-lick frames)")
    print(Bk[Bk.session == "PS93_20260814"].to_string(index=False))
