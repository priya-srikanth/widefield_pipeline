"""Hemisphere gain on a hemisphere-MATCHED area set (Priya 11-13: "yes we should compute on a matched set").
hemi_gain_t = mean log gain (vs the multi-session pre template) over left (ipsilesional) areas minus right areas.
The glue mask leaves 21 left / 24 right areas; matched = the 21 areas present on both sides (drops VISp, VISal, VISl
right). Recomputed from the saved per-area gain_t_<area> columns (no re-run); check: the full set reproduces the
saved hemi_gain_t. Per-pair breakdown = log(g_left) - log(g_right) per area pair, change vs own pre.
Same statistic as state_space_recovery_trials.summarize: kit.boot_delta, rng seed 0, n_up = animals above own pre."""
from pathlib import Path
import numpy as np
import pandas as pd
from wfield_local import analysis_kit as kit

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery")
EPOCHS = ["acute", "subacute", "chronic"]


def delta(df, col):
    rng = np.random.default_rng(0)
    pre = {a: g[g.epoch == "pre"][col].dropna().tolist() for a, g in df.groupby("animal")}
    out = []
    for ep in EPOCHS:
        post = {a: g[g.epoch == ep][col].dropna().tolist() for a, g in df.groupby("animal")}
        post = {a: v for a, v in post.items() if v}
        iv = kit.boot_delta(post, pre, rng)
        up = sum(np.mean(post[a]) > np.mean(pre[a]) for a in post if pre.get(a))
        per = {a: round(np.mean(post[a]) - np.mean(pre[a]), 3) for a in sorted(post) if pre.get(a)}
        star = "*" if (iv.lo > 0 or iv.hi < 0) else ""
        out.append(dict(readout=col, epoch=ep, delta=round(iv.point, 3), lo=round(iv.lo, 3), hi=round(iv.hi, 3),
                        star=star, n_up=f"{up}/{len(post)}", **{f"d_{a}": v for a, v in per.items()}))
    return out


rows = []
for align, sub in (("cue", "trials"), ("lick", "trials_lick")):
    df = pd.read_csv(ROOT / sub / "state_space_trials_sessions.csv")
    g = {c[7:]: c for c in df if c.startswith("gain_t_")}
    L = {a[:-5] for a in g if a.endswith("_left")}
    R = {a[:-6] for a in g if a.endswith("_right")}
    def lg(a, df=df, g=g):
        return np.log(np.clip(df[g[a]], 1e-3, None))
    full = np.mean([lg(a + "_left") for a in L], 0) - np.mean([lg(a + "_right") for a in R], 0)
    err = np.nanmax(np.abs(full - df.hemi_gain_t))
    print(f"[{align}] left {len(L)} right {len(R)} matched {len(L & R)}; right-only {sorted(R - L)}; "
          f"left-only {sorted(L - R)}; full-set reproduce max err {err:.2e}")
    assert err < 1e-9
    M = sorted(L & R)
    df["hemi_full"] = full
    df["hemi_matched"] = np.mean([lg(a + "_left") - lg(a + "_right") for a in M], 0)
    for a in M:
        df[f"pair_{a}"] = lg(a + "_left") - lg(a + "_right")
    for col in ["hemi_full", "hemi_matched"] + [f"pair_{a}" for a in M]:
        for r in delta(df, col):
            rows.append(dict(align=align, **r))
res = pd.DataFrame(rows)
out = Path(__file__).parent / "hemi_matched.csv"
res.to_csv(out, index=False)
with pd.option_context("display.width", 250, "display.max_rows", 500):
    print(res[res.readout.isin(["hemi_full", "hemi_matched"])].to_string(index=False))
    print(res[res.readout.str.startswith("pair_") & (res.epoch != "acute")].sort_values(["align", "epoch", "delta"])
          .to_string(index=False))
