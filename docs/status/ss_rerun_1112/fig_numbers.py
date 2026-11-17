"""Caption numbers of SS Figs 3.3 / 3.4 (response size), 3.7 / 3.8 (noise), 4.5 (contact size) glue-in vs glue-free:
change vs own pre (log2), boot_delta (animals -> sessions), per spout x epoch; per-animal deltas. Read-only."""
import numpy as np
import pandas as pd
from wfield_local import analysis_kit as kit

R = "M:/MICROSCOPE/Priya/Widefield/state_space_recovery"
B = R + "/_superseded_glue_20261109/allen_outputs_pre_v20u_20261112"
EP = ["acute", "subacute", "chronic"]


def cell(t, col, ep, spout=None):
    if spout is not None:
        t = t[t.spout == spout]
    post = {a: g[col].dropna().to_numpy() for a, g in t[t.epoch == ep].groupby("animal")}
    pre = {a: g[col].dropna().to_numpy() for a, g in t[t.epoch == "pre"].groupby("animal")}
    iv = kit.boot_delta(post, pre, np.random.default_rng(0))
    an = sorted(set(post) & set(pre))
    per = [post[a].mean() - pre[a].mean() for a in an]
    if iv is None:
        return "n/a"
    lo, hi = iv.lo, iv.hi
    return f"{iv.point:+.2f}[{lo:+.2f},{hi:+.2f}]{'*' if lo > 0 or hi < 0 else ''} {sum(p > 0 for p in per)}/{len(per)}"


rows = []
for d, sub in (("cue", ""), ("lick", "_lick")):
    for f, cols in ((f"trajectories{sub}/resp_size.csv", ["lsize"]), (f"noise{sub}/noise_spouts.csv", None)):
        for src, root in (("glue-in", B), ("glue-free", R)):
            t = pd.read_csv(f"{root}/{f}")
            cc = cols or [c for c in ("base_noise", "resp_noise", "size2") if c in t]
            for c in cc:
                if c != "lsize":
                    t[c] = np.log2(t[c].clip(lower=1e-12))
                for sp in sorted(t.spout.unique()):
                    for ep in EP:
                        rows.append(dict(align=d, table=f.split("/")[0], measure=c, spout=sp, epoch=ep, src=src, v=cell(t, c, ep, sp)))
x = pd.DataFrame(rows).pivot_table(index=["align", "table", "measure", "spout", "epoch"], columns="src", values="v", aggfunc="first")
x = x[["glue-in", "glue-free"]]
x["star_change"] = [("*" in str(a)) != ("*" in str(b)) for a, b in zip(x["glue-in"], x["glue-free"])]
x.to_csv("C:/Users/SabatiniLab/Github/wf_attempt/docs/status/ss_rerun_1112/fig_numbers.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(x[x.star_change].to_string())
print("n cells", len(x), "star changes", int(x.star_change.sum()))
sel = x.reset_index()
print(sel[(sel.measure == "lsize") & (sel.epoch.isin(["acute", "chronic"]))].to_string(index=False))
