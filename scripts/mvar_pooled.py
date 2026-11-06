"""POOLED sensory / motor MVAR (Priya 10-30, after the 18-region MVAR failed its gate). PROVISIONAL.

Design + gate pre-registered in docs/status/MVAR_POOLED_PREREG_20261030.md (written before any pooled fit) -- read it.
Six nodes (S_i, S_c, M_i, M_c, R_i, R_c = sensory / motor / RSPd pooled per hemisphere, i = ipsilesional = left),
lags up to 8 samples (~0.77 s). The fitting, nested CV and gate are mvar_regions' code UNCHANGED; this module only
swaps in the node series and the lag range (patched into mvar_regions at import, so spawned workers see it too).

    python scripts/mvar_pooled.py fit                 # August pre sessions, all modes
    python scripts/mvar_pooled.py gate
    python scripts/mvar_pooled.py fixed               # only if >= 3 / 4 animals pass on raw
    python scripts/mvar_pooled.py summarise
    python scripts/mvar_pooled.py plot | --plot-only
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import mvar_regions as mr  # noqa: E402

cp = mr.cp
NODES = ["S_i", "S_c", "M_i", "M_c", "R_i", "R_c"]
HEM = {"i": "left", "c": "right"}
MEMBERS = {f"S_{h}": [f"{a}_{HEM[h]}" for a in mr.SENS] for h in "ic"} | \
          {f"M_{h}": [f"{a}_{HEM[h]}" for a in mr.MOT] for h in "ic"} | {f"R_{h}": [f"RSPd_{HEM[h]}"] for h in "ic"}
OUT = mr.OUT.parent / "mvar_pooled"
NR18 = len(cp.REGIONS)

# ---- patch mvar_regions (module globals are looked up at call time)
mr.OUT = OUT
mr.REGIONS = NODES
mr.NR = len(NODES)
mr.PS = (1, 2, 3, 4, 6, 8)
mr.PMAX = 8


def load_runs(lab, mode="raw"):
    """List of (T_run, 6) node arrays: each region z-scored over the session, then averaged into its node."""
    d = np.load(mr.DATA / f"{lab}.npz")
    Y = d["res_one"].astype(np.float64).T                 # (T, 18)
    dt = float(d["dt"])
    st = list(d["run_starts"]) + [len(Y)]
    Y = Y - Y.mean(0)
    if mode == "gsr":                                     # leave-self-out global removal on the 18 regions, BEFORE pooling
        S = Y.sum(1, keepdims=True)
        for a in range(NR18):
            g = (S[:, 0] - Y[:, a]) / (NR18 - 1)
            Y[:, a] = Y[:, a] - (g @ Y[:, a]) / (g @ g) * g
    Y = Y / (Y.std(0) + 1e-12)
    ix = {r: i for i, r in enumerate(cp.REGIONS)}
    X = np.stack([Y[:, [ix[r] for r in MEMBERS[n]]].mean(1) for n in NODES], 1)
    runs = [X[a:b] for a, b in zip(st[:-1], st[1:])]
    if mode == "dec":
        runs = [r + mr.TAU * np.gradient(r, dt, axis=0) for r in runs]
    return runs, dt


mr.load_runs = load_runs


def session_fit(job):
    return mr.session_fit(job)


def fixed_job(job):
    return mr.fixed_job(job)


# ------------------------------------------------------------------------------------------------------------ readouts
def group_values(W) -> dict:
    n = {k: i for i, k in enumerate(NODES)}

    def w(t, s):                                          # W[target, source]
        return float(W[n[t], n[s]])
    v = {}
    for hs, ht in itertools.product("ic", "ic"):
        v[f"S{hs}->M{ht}"] = w(f"M_{ht}", f"S_{hs}")
        v[f"M{ht}->S{hs}"] = w(f"S_{hs}", f"M_{ht}")
        v[f"dir S{hs}<->M{ht}"] = v[f"S{hs}->M{ht}"] - v[f"M{ht}->S{hs}"]
        v[f"R{hs}->M{ht}"] = w(f"M_{ht}", f"R_{hs}")
        v[f"M{ht}->R{hs}"] = w(f"R_{hs}", f"M_{ht}")
    v["S->M intra"] = (v["Si->Mi"] + v["Sc->Mc"]) / 2
    v["S->M inter"] = (v["Si->Mc"] + v["Sc->Mi"]) / 2
    v["M->S intra"] = (v["Mi->Si"] + v["Mc->Sc"]) / 2
    v["M->S inter"] = (v["Mc->Si"] + v["Mi->Sc"]) / 2
    v["dir intra"] = v["S->M intra"] - v["M->S intra"]
    v["dir inter"] = v["S->M inter"] - v["M->S inter"]
    v["R->M"] = np.mean([v[f"R{a}->M{b}"] for a, b in itertools.product("ic", "ic")])
    v["M->R"] = np.mean([v[f"M{b}->R{a}"] for a, b in itertools.product("ic", "ic")])
    v["dir R<->M"] = v["R->M"] - v["M->R"]
    v["S->M all"] = (v["S->M intra"] + v["S->M inter"]) / 2
    v["S->M minus R->M"] = v["S->M all"] - v["R->M"]
    v["mean |W| offdiag"] = float(np.abs(W[~np.eye(len(W), dtype=bool)]).mean())
    return v


KEY = ["S->M intra", "S->M inter", "M->S intra", "M->S inter", "dir intra", "dir inter", "dir Si<->Mi", "dir Sc<->Mc",
       "dir Si<->Mc", "dir Sc<->Mi", "R->M", "dir R<->M", "S->M minus R->M", "mean |W| offdiag"]


def noise_floor() -> None:
    """Split-half difference of each primary readout over pre sessions (reported, not gated)."""
    import pandas as pd
    rows = []
    for lab in mr.all_labels(pre_only=True):
        for mode in mr.MODES:
            f = OUT / "sessions" / f"{lab}_{mode}.npz"
            if not f.exists():
                continue
            s = mr.load_sess(lab, mode)
            a, b, full = group_values(s["W_h1"]), group_values(s["W_h2"]), group_values(s["W"])
            rows += [{"animal": lab[:4], "label": lab, "mode": mode, "metric": k, "full": full[k], "h1": a[k], "h2": b[k],
                      "half_diff": a[k] - b[k]} for k in KEY]
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "tables" / "readout_halves.csv", index=False)
    g = D.groupby(["mode", "metric"]).agg(pre_mean=("full", "mean"), between_sess_sd=("full", "std"),
                                          half_sd=("half_diff", lambda x: float(np.sqrt(np.mean(x ** 2) / 2))))
    print("\n=== readout noise floor (pre): split-half SD of a single-session estimate vs between-session SD ===")
    print(g.round(4).to_string())


def summarise() -> None:
    import pandas as pd

    from wfield_local import analysis_kit as ak
    rows = []
    for lab in mr.all_labels():
        for mode in mr.MODES:
            for fit, sub in (("fixed", "fixed"), ("cv", "sessions")):
                f = OUT / sub / f"{lab}_{mode}.npz"
                if f.exists():
                    rows.append({"label": lab, "animal": lab[:4], "epoch": mr.epoch(lab), "mode": mode, "fit": fit,
                                 **group_values(np.load(f)["W"])})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "tables" / "group_weights_sessions.csv", index=False)
    pd.set_option("display.width", 250)
    A, T = [], []
    for (mode, fit), g in D.groupby(["mode", "fit"], sort=True):
        for m in KEY:
            pre = {a: g[(g.animal == a) & (g.epoch == "pre")][m].tolist() for a in mr.AN}
            for ep in mr.EPS:
                for a in mr.AN:
                    v = g[(g.animal == a) & (g.epoch == ep)][m]
                    A.append({"mode": mode, "fit": fit, "metric": m, "epoch": ep, "animal": a, "n": len(v),
                              "mean": v.mean(), "delta": v.mean() - np.mean(pre[a]) if len(v) and pre[a] else np.nan})
                if ep == "pre":
                    continue
                post = {a: v for a in mr.AN if (v := g[(g.animal == a) & (g.epoch == ep)][m].tolist())}
                iv = ak.boot_delta(post, {a: v for a, v in pre.items() if v}, np.random.default_rng(0))
                dl = [np.mean(post[a]) - np.mean(pre[a]) for a in post if pre.get(a)]
                T.append({"mode": mode, "fit": fit, "metric": m, "epoch": ep, "delta": iv.point if iv else np.nan,
                          "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan, "n_animals": len(dl),
                          "n_down": int(sum(x < 0 for x in dl)), "n_up": int(sum(x > 0 for x in dl)),
                          "pre_mean": float(np.mean([np.mean(v) for v in pre.values() if v]))})
    A, T = pd.DataFrame(A), pd.DataFrame(T)
    A.to_csv(OUT / "tables" / "group_weights_animal.csv", index=False)
    T.to_csv(OUT / "tables" / "group_weights_tests.csv", index=False)
    for mode in mr.MODES:
        a = A[(A["mode"] == mode) & (A.fit == "fixed")]
        print(f"\n=== PER-ANIMAL (mode {mode}, fixed lambda/p): delta vs Aug pre ===")
        print(a.pivot_table(index=["metric", "animal"], columns="epoch", values="delta").reindex(columns=mr.EPS[1:])
              .round(4).to_string())
    t = T.copy()
    t["ci_excl0"] = (t.lo > 0) | (t.hi < 0)
    print("\n=== boot_delta (paired nested) ===")
    print(t.round(4).to_string())


# ------------------------------------------------------------------------------------------------------------- figures
def plot() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    td, fd = OUT / "tables", OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    G = pd.read_csv(td / "gate_animals.csv")
    crit = [("halves_med", "halves\n(median)", 0.7), ("halves_min", "halves\n(worst)", 0.5),
            ("sessions_med", "sessions", 0.7), ("ridge_up_med", "lambda x10", 0.7), ("ridge_dn_med", "lambda /10", 0.7),
            ("halves_pair_med", "halves,\npair-specific", 0.5)]
    modes = [m for m in mr.MODES if m in set(G["mode"])]
    fig, axs = plt.subplots(1, len(modes), figsize=(4.6 * len(modes), 3.8), sharey=True, squeeze=False)
    for ax, mode in zip(axs[0], modes):
        g = G[G["mode"] == mode].set_index("animal")
        x = np.arange(len(crit))
        for k, an in enumerate(mr.AN):
            if an in g.index:
                ax.plot(x + (k - 1.5) * 0.12, [g.loc[an, c] for c, _, _ in crit], "o", ms=5,
                        label=f"{an} {'pass' if g.loc[an, 'PASS'] else 'fail'}")
        for i, (_, _, thr) in enumerate(crit):
            ax.hlines(thr, i - 0.35, i + 0.35, color="k", ls="--", lw=1)
        ax.set_xticks(x, [c[1] for c in crit], fontsize=7)
        ax.set_title(f"{mode}{' (gated)' if mode == 'raw' else ' (not gated)'}: {int(g.PASS.sum())}/4 pass", fontsize=9)
        ax.set_ylim(-0.2, 1.05)
    axs[0, 0].set_ylabel("Pearson r of off-diagonal W")
    axs[0, -1].legend(fontsize=7, loc="lower left", frameon=False)
    fig.suptitle("Pooled 6-node MVAR stability gate (pre-registered; >= 3 / 4 animals needed on raw)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "pooled_gate.png", dpi=150)
    plt.close(fig)
    f = td / "group_weights_animal.csv"
    if not f.exists():
        print("no post-gate tables; gate figure only")
        return
    A = pd.read_csv(f)
    T = pd.read_csv(td / "group_weights_tests.csv")
    show = ["dir intra", "dir inter", "S->M intra", "S->M inter", "M->S intra", "R->M", "dir R<->M", "S->M minus R->M"]
    fig, axs = plt.subplots(len(modes), 1, figsize=(12, 3.2 * len(modes)), squeeze=False)
    for ax, mode in zip(axs[:, 0], modes):
        for i, m in enumerate(show):
            for j, ep in enumerate(mr.EPS[1:]):
                xx = i + (j - 1) * 0.25
                t = T[(T["mode"] == mode) & (T.fit == "fixed") & (T.metric == m) & (T.epoch == ep)]
                if len(t):
                    t = t.iloc[0]
                    ax.errorbar(xx, t.delta, yerr=[[t.delta - t.lo], [t.hi - t.delta]], fmt="s", color=mr.COL[ep],
                                ms=5, label=ep if i == 0 else None)
                a = A[(A["mode"] == mode) & (A.fit == "fixed") & (A.metric == m) & (A.epoch == ep)]
                ax.plot(np.full(len(a), xx + 0.08), a.delta, ".", color="grey", ms=4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(show)), show, fontsize=8)
        ax.set_ylabel(f"{mode}: change vs pre\n(lag-summed W, z units)", fontsize=8)
    axs[0, 0].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Pooled MVAR: change vs August pre (fixed per-animal lambda / p; squares = boot_delta, dots = animals)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "pooled_change.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "gate", "fixed", "summarise", "plot"])
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--modes", nargs="+", default=list(mr.MODES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only or a.cmd == "plot":
        return plot()
    from wfield_local import analysis_kit as ak
    jobs = min(4, a.jobs)
    labs = a.labels or mr.all_labels(pre_only=not a.all)
    t0 = time.time()
    if a.cmd == "fit":
        todo = [(x, m) for x in labs for m in a.modes if (mr.DATA / f"{x}.npz").exists()
                and not (OUT / "sessions" / f"{x}_{m}.npz").exists()]
        res, fail = ak.fan_sessions(todo, session_fit, jobs=jobs, label="fit")
        print(f"fitted {len(res)}, failed {fail}")
    elif a.cmd == "gate":
        mr.gate()
        noise_floor()
    elif a.cmd == "fixed":
        import pandas as pd
        G = pd.read_csv(OUT / "tables" / "gate_animals.csv")
        if G[G["mode"] == "raw"].PASS.sum() < 3:
            sys.exit("gate FAILED on raw (< 3 animals): no pre vs post fits (pre-registered)")
        par = {(r.mode, r.animal): (r.lam_geomed, int(r.p_mode)) for r in G.itertuples()}
        todo = [(x, m) + par[(m, x[:4])] for x in mr.all_labels() for m in a.modes if (mr.DATA / f"{x}.npz").exists()]
        res, fail = ak.fan_sessions(todo, fixed_job, jobs=jobs, label="fixed")
        print(f"fixed fits {len(res)}, failed {fail}")
    elif a.cmd == "summarise":
        summarise()
    print(f"[{a.cmd}] wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
