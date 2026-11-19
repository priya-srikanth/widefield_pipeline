"""Every bridged cam1 tongue run (cam1_licks.tongue_runs) of the labeled + blast sessions, annotated (scratch, 11-18)."""
import sys
import pandas as pd
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
    to_frame = C1.cam1_frame(flat, ix)[2]
    out, lat = to_frame(flat.tongue_x.to_numpy(float), flat.tongue_y.to_numpy(float))
    ok = flat.tongue_likelihood.to_numpy(float) >= float(OC.params("tongue", "cam1")["lk_thr"])
    tid, t = ix.trial_id.to_numpy(), ix.t_ms.to_numpy(float)
    R = C1.tongue_runs(out, lat, ok, tid, t)
    R["t0"], R["t1"] = t[R.a.to_numpy(int)], t[R.b.to_numpy(int)]
    sd, f4, _ = lp_licks._inputs(f"{a}_{d[4:]}", rv)
    Lk = pd.read_csv(f4, usecols=["trial_id", "on_ms", "off_ms"])
    win = {k: g[["on_ms", "off_ms"]].to_numpy(float) for k, g in Lk.groupby("trial_id")}
    R["cam4"] = [bool(w is not None and ((w[:, 0] <= r.t1) & (w[:, 1] >= r.t0)).any())
                 for r, w in ((r, win.get(r.trial_id)) for r in R.itertuples())]
    R["answer"], R["stratum"] = None, None
    for e in L.itertuples():
        hit = (R.trial_id == e.trial_id) & (R.t0 <= e.t_ms + 8) & (R.t1 >= e.t_ms - 8)
        R.loc[hit, "answer"], R.loc[hit, "stratum"] = e.answer, e.stratum
    R["session"] = s
    return R


class W:
    def __init__(self, L): self.L = L
    def __call__(self, s): return one(s, self.L[self.L.session == s])


if __name__ == "__main__":
    from drop_features import labels
    from wfield_local import analysis_kit as ak
    L = labels()
    B = pd.read_csv(f"{sys.argv[1]}/mask_blast.csv").session.tolist()
    res, fail = ak.fan_sessions(sorted(set(L.session) | set(B)), W(L), jobs=6)
    for s, e in fail:
        print("FAIL", s, e)
    R = pd.concat([v for _s, v in res if v is not None], ignore_index=True)
    R["blast"] = R.session.isin(B)
    R.to_csv(f"{sys.argv[1]}/runs.csv.gz", index=False)
    print(len(R), "runs")
