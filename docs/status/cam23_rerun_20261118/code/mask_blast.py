"""Blast radius of cam1_licks.mask_detached (scratch, 11-18): per session, confident cam1 tongue frames masked, how many
sit inside a cam4 lick (licks_all on..off: cam4 sees a tongue then -> a wrong mask), and trial 540 of PS93 0814."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "C:/Users/SabatiniLab/Github/wf_docbranches")


def one(s):
    from scripts.cam1_licks import PROD
    from wfield_local import cam1_licks as C1, lp_licks, o2_inference as oi, epochs
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
    sd, f4, _ = lp_licks._inputs(f"{a}_{d[4:]}", rv)
    L = pd.read_csv(f4, usecols=["trial_id", "on_ms", "off_ms"])
    tid, t = ix.trial_id.to_numpy(), ix.t_ms.to_numpy(float)
    in4 = np.zeros(len(t), bool)
    for tr, g in L.groupby("trial_id"):
        k = np.flatnonzero(tid == tr)
        for on, off in zip(g.on_ms, g.off_ms):
            in4[k[(t[k] >= on) & (t[k] <= off)]] = True
    post = (t >= 0) & (t <= ix.stop_ms.to_numpy(float))
    r = {"session": s, "epoch": epochs.epoch_of(f"{a}_{d[4:]}"), "conf": int(ok.sum()), "masked": int(m.sum()),
         "masked_post": int((m & post).sum()), "masked_in_cam4_lick": int((m & in4).sum()),
         "conf_in_cam4_lick": int((ok & in4).sum()), "trials_masked_post": int(len(np.unique(tid[m & post])))}
    if s == "PS93_20260814":
        k = tid == 540
        r["t540_conf"], r["t540_masked"] = int((ok & k).sum()), int((m & k).sum())
    return r


if __name__ == "__main__":
    from wfield_local import analysis_kit as ak, config
    S = []
    for a in ("PS92", "PS93", "PS94", "PS95"):
        labs = [l for l in ak.curated_labels() if l.startswith(a)]
        S += [f"{a}_{config.date8(l.split('_')[1])}" for l in np.array(labs)[np.linspace(0, len(labs) - 1, 6).astype(int)]]
    S = sorted(set(S + ["PS93_20260814"]))
    res, fail = ak.fan_sessions(S, one, jobs=6)
    for s, e in fail:
        print("FAIL", s, e)
    R = pd.DataFrame([v for _s, v in res if v])
    R.to_csv(f"{sys.argv[1]}/mask_blast.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))
    tot = R[["conf", "masked", "masked_post", "masked_in_cam4_lick", "conf_in_cam4_lick"]].sum()
    print(f"\nmasked {tot.masked} / {tot.conf} confident frames ({tot.masked / tot.conf:.2%}); inside a cam4 lick "
          f"{tot.masked_in_cam4_lick} / {tot.conf_in_cam4_lick} ({tot.masked_in_cam4_lick / tot.conf_in_cam4_lick:.3%})")
