"""Raw (pre-cleaning) cam1 tongue features of every blind-reviewed cam1 event: is the point detached from the mouth and
stationary? Scratch evaluation for the frame-level drop mask (Priya 11-18)."""
import sys
import numpy as np, pandas as pd

CUT = 0.4
D = "M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/cam1_licks"
G = "M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/restnl_cam1_compare_20261117/cam1_extra_check/review"


def labels():
    R = []
    for r in ("review", "review2", "review3"):
        s = pd.read_csv(f"{D}/{r}/scored.csv")
        R.append(pd.DataFrame({"src": r, "id": s.id, "session": s.session, "trial_id": s.trial_id, "t_ms": s.t_ms,
                               "stratum": s.stratum, "answer": s.answer}))
    k, a = pd.read_csv(f"{G}/key.csv"), pd.read_csv(f"{G}/scored_priya_20261117.csv")
    k = k.merge(a, on="id")
    lab = k.label.str.split("_")
    k["session"] = lab.str[0] + "_2026" + lab.str[1]
    R.append(pd.DataFrame({"src": "x1117", "id": k.id, "session": k.session, "trial_id": k.trial_id, "t_ms": k.t_peak_ms,
                           "stratum": k.stratum, "answer": k.answer}))
    return pd.concat(R, ignore_index=True)


def session(s, rows):
    sys.path.insert(0, "C:/Users/SabatiniLab/Github/wf_docbranches")
    from scripts.cam1_licks import PROD, _cam1_frame
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    a, d = s.split("_")
    c1 = oi.session_windows(rv, PROD["cam1"], a, d, "cam1")
    if c1 is None:
        return []
    ix, flat = c1[0].reset_index(drop=True), c1[1].reset_index(drop=True)
    _fr, _med, to_frame = _cam1_frame(flat, ix)
    out, lat = to_frame(flat.tongue_x.to_numpy(float), flat.tongue_y.to_numpy(float))
    ok = flat.tongue_likelihood.to_numpy(float) >= CUT
    tid, t = ix.trial_id.to_numpy(), ix.t_ms.to_numpy(float)
    res = []
    for r in rows.itertuples():
        k = np.flatnonzero((tid == r.trial_id) & (np.abs(t - r.t_ms) <= 60) & ok)
        if not len(k):
            res.append({"id": r.id, "found": False}); continue
        c = k[np.argmin(np.abs(t[k] - r.t_ms))]
        a0 = c
        while a0 - 1 >= 0 and ok[a0 - 1] and tid[a0 - 1] == tid[c] and abs(t[a0] - t[a0 - 1] - 4) < 1:
            a0 -= 1
        b0 = c
        while b0 + 1 < len(ok) and ok[b0 + 1] and tid[b0 + 1] == tid[c] and abs(t[b0 + 1] - t[b0] - 4) < 1:
            b0 += 1
        kk = np.arange(a0, b0 + 1)
        xy = np.c_[out[kk], lat[kk]]
        dist = np.hypot(*xy.T)
        med = np.median(xy, 0)
        res.append({"id": r.id, "found": True, "run_n": len(kk), "onset_out": out[a0], "onset_dist": dist[0],
                    "min_out": out[kk].min(), "max_out": out[kk].max(), "min_dist": dist.min(),
                    "spread": float(np.median(np.hypot(*(xy - med).T))), "range": float(np.ptp(dist)),
                    "step0": float(np.hypot(*(xy[1] - xy[0]))) if len(kk) > 1 else np.nan,
                    "max_step": float(np.max(np.hypot(*np.diff(xy, axis=0).T))) if len(kk) > 1 else np.nan})
    return res


class W:
    def __init__(self, L): self.L = L
    def __call__(self, s): return session(s, self.L[self.L.session == s])


if __name__ == "__main__":
    sys.path.insert(0, "C:/Users/SabatiniLab/Github/wf_docbranches")
    from wfield_local import analysis_kit as ak
    L = labels()
    res, fail = ak.fan_sessions(sorted(L.session.unique()), W(L), jobs=6)
    for s, e in fail:
        print("FAIL", s, e)
    F = pd.DataFrame([x for _s, v in res for x in v])
    O = L.merge(F, on="id", how="left")
    O.to_csv(f"{sys.argv[1]}/drop_features.csv", index=False)
    print(O.answer.value_counts().to_string())
