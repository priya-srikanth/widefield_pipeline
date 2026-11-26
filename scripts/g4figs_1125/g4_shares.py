"""Unique target / executed / drop-position shares by epoch, main (thr_push) vs main_spline (read-only; from
g4_running_compare_ccB1124/shares/shares_main_vs_spline.csv). August-only pre; per animal first (mean over sessions),
then mean of animals; pooled = boot_delta change vs own pre (95% nested CI)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wfield_local import config
from wfield_local.analysis_kit import boot_delta
S = pd.read_csv(r"M:/MICROSCOPE/Priya/Widefield/run_records/2026-11-24/g4_running_compare_ccB1124/shares/shares_main_vs_spline.csv")
S = S[~((S.epoch == "pre") & S.label.map(lambda l: config.early_pre("2026" + l.split("_")[1])))]
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
EP = ["pre", "acute", "subacute", "chronic"]; AN = ["PS92", "PS93", "PS94", "PS95"]
rows, prow = [], []
for (tag, w), g in S[S.window.isin(["all", "cue", "lick", "enl"])].groupby(["tag", "window"]):
    for q in ("uniq_target", "uniq_exec", "drop_position", "full"):
        A = g.groupby(["animal", "epoch"])[q].mean().unstack("epoch").reindex(columns=EP)
        for an, r in A.iterrows():
            rows.append(dict(tag=tag, window=w, quantity=q, animal=an, **r.to_dict()))
        rows.append(dict(tag=tag, window=w, quantity=q, animal="mean", **A.mean().to_dict()))
        for ep in EP[1:]:
            post = {a: g[(g.animal == a) & (g.epoch == ep)][q].to_numpy(float) for a in AN}
            pre = {a: g[(g.animal == a) & (g.epoch == "pre")][q].to_numpy(float) for a in AN}
            iv = boot_delta({a: v for a, v in post.items() if len(v)}, {a: v for a, v in pre.items() if len(v)}, np.random.default_rng(0))
            prow.append(dict(tag=tag, window=w, quantity=q, epoch=ep, point=iv.point, lo=iv.lo, hi=iv.hi,
                             n_up=int(sum(post[a].mean() > pre[a].mean() for a in AN if len(post[a]) and len(pre[a])))))
T = pd.DataFrame(rows); P = pd.DataFrame(prow)
T.to_csv(OUT / "shares_by_animal_epoch.csv", index=False, float_format="%.5f")
P.to_csv(OUT / "shares_change_vs_pre.csv", index=False, float_format="%.5f")
n = S[S.tag == "main"].drop_duplicates("label").groupby(["animal", "epoch"]).size().unstack().reindex(columns=EP)
print("sessions per animal x epoch (August-only pre):\n", n)
pd.set_option("display.width", 220)
for w in ("all", "lick", "cue"):
    print(f"\n== window {w}")
    print(T[(T.window == w)].set_index(["tag", "quantity", "animal"])[EP].mul(1000).round(2).to_string())
    print(P[P.window == w].assign(**{c: P[P.window == w][c] * 1000 for c in ("point", "lo", "hi")}).round(2).to_string(index=False))
