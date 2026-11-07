"""LICK-VIGOR control for the PER-SPOUT lick responses of 14.7 (Priya 10-28: "should we try regressing on lick vigor";
"proceed ... starting with lick vigor control").

Question: are the per-spout differences of 14.7 -- cued far-contra licks not larger acutely (PS92 / PS93 below pre), ITI
licks larger after every previous spout -- explained by HOW the tongue moved? Per lick: response = dF/F AUC 0..0.5 s
after lick onset (baseline -1.0..-0.5 s, = 14.7) / the animal's pre AUC over all spouts (= 14.7's scale). Covariates =
the lick's own kinematics, z-scored against the animal's PRE licks of the same set:
  VIGOR   3-D reach, peak tongue speed, outward (AP) and height (DV) of the tip, cam4 lick duration, retraction speed
  +LANDING the same plus the lateral tip position (side_ml) = where the lick landed
Per animal, a linear fit on PRE licks only (all spouts); each lick's response minus the fitted covariate part (+ the pre
intercept) -- as 14.13 (`iti_licks._pre_adjust`). Then per spout, session means (>= MIN_SPOUT licks), change vs pre at
that spout (paired nested bootstrap, `analysis_kit.boot_delta`), unadjusted / vigor / vigor+landing on the SAME scale.
Vigor + landing asks: same movement, same place -- is the response still different? (For cued far-contra licks that
miss ipsilesionally, adjusting for landing removes the part of the response that follows where the lick went.)
Lick kinematics: cam4 licks_all.csv (onset, duration, retraction) + pose3d kinematics_all (reach, speed, 3-D tip),
matched to each lick by DAQ onset time (ITI: exact, as 14.13; cued first licks: nearest cam4 onset within MATCH_S).
Cued first-lick times are rebuilt once per session (`iti_licks._first_lick_times`, slow) and cached in
<OUT_W>/cued_first_lick_times.csv (`--times-cache`).
Outputs <OUT_W>/vigor_perspout_licks.csv, vigor_perspout_tests.csv; figure lick_vigor_perspout.png (figdata +
--plot-only). Usage: python scripts/lick_vigor_perspout.py --times-cache [--jobs N]; python scripts/lick_vigor_perspout.py
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_cued_ramp as cr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

TIMES = "cued_first_lick_times.csv"
MATCH_S = 0.03
BASE = cr.BASE_SPOUT
ROIS = cr.SPOUT_ROIS
VIG = ["reach", "speed", "out_ap", "down_dv", "dur_ms", "retract"]
MODELS = {"none": [], "vigour": VIG, "vigour+landing": [*VIG, "side_ml"]}
MIN_PRE = 30
# response windows (Priya 10-28: longer AUC; "did you re-do the vigor regression"): 0..0.5, 0..1.0, late 0.5..1.0 s
VWIN = ((0.0, 0.5), (0.0, 1.0), (0.5, 1.0))
RW = [(r, f"{w0:g}-{w1:g}") for w0, w1 in VWIN for r in ROIS]
EP = ["acute", "subacute", "chronic"]


# ------------------------------------------------------------------------------------------------ cued first-lick times
def _times_worker(label: str) -> dict:
    d = il._load(label)
    cue, fl = il._first_lick_times(label, d["first_rt"])
    ok = np.isfinite(fl) & np.isfinite(cue)                 # = the rows of cued_wide/<label>.npz (wide_session)
    return {"label": label, "t": fl[ok]}


def times_cache(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    res, fails = ak.fan_sessions(il.labels(), _times_worker, jobs=jobs)
    rows = [pd.DataFrame({"label": x, "row": np.arange(len(v["t"])), "t": v["t"]}) for x, v in res]
    pd.concat(rows, ignore_index=True).to_csv(il.OUT_W / TIMES, index=False, float_format="%.17g")
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")


# ------------------------------------------------------------------------------------------------------- per-lick table
_K = {}


def _kin(label: str, t: np.ndarray, exact: bool) -> pd.DataFrame:
    """Kinematics of licks at DAQ onset times t: exact onset match (ITI bout starts are cam4 onsets) or nearest cam4
    onset within MATCH_S (cued first licks come from the behavior lick table)."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv",
                    usecols=["daq_on_s", "on_ms", "off_ms", "max_retract_velocity_y_px_per_s"])
    if "K" not in _K:
        p = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz"
        _K["K"] = pd.read_csv(p, usecols=["session", "lick_uid", "reach_mm", "speed_peak", "side_ml", "out_ap", "down_dv"])
    K = _K["K"]
    K = K[K.session == f"{an}_{date}"].drop_duplicates("lick_uid").set_index("lick_uid")
    on = A.daq_on_s.to_numpy(float)
    t = np.asarray(t, float)
    if exact:
        uid = pd.Series(np.arange(len(A)), index=on)
        u = uid[~uid.index.duplicated()].reindex(t).to_numpy()
    else:
        o = np.argsort(on)
        j = np.clip(np.searchsorted(on[o], t), 1, len(on) - 1)
        jj = np.where(np.abs(on[o][j - 1] - t) <= np.abs(on[o][j] - t), j - 1, j)
        u = np.where(np.abs(on[o][jj] - t) <= MATCH_S, o[jj], np.nan)
    out = pd.DataFrame(np.nan, index=range(len(t)), columns=["reach", "speed", "side_ml", "out_ap", "down_dv", "dur_ms",
                                                             "retract"])
    ok = np.isfinite(u)
    ui = u[ok].astype(int)
    out.loc[ok, ["reach", "speed", "side_ml", "out_ap", "down_dv"]] = \
        K.reindex(ui)[["reach_mm", "speed_peak", "side_ml", "out_ap", "down_dv"]].to_numpy(float)
    out.loc[ok, "dur_ms"] = (A.off_ms.to_numpy(float) - A.on_ms.to_numpy(float))[ui]
    out.loc[ok, "retract"] = A.max_retract_velocity_y_px_per_s.to_numpy(float)[ui]
    return out


def _auc(X: np.ndarray, w0: float, w1: float) -> np.ndarray:
    """(licks, lags, rois) -> per-lick AUC w0..w1 s after baseline (BASE) subtraction."""
    X = X - np.nanmean(X[:, (cr.LAGS_W >= BASE[0]) & (cr.LAGS_W <= BASE[1]), :], 1, keepdims=True)
    w = (cr.LAGS_W >= w0) & (cr.LAGS_W <= w1)
    return np.trapezoid(X[:, w, :], cr.LAGS_W[w], axis=1)


def lick_table() -> pd.DataFrame:
    TT = pd.read_csv(il.OUT_W / TIMES, float_precision="round_trip")
    TT = {x: g.sort_values("row").t.to_numpy(float) for x, g in TT.groupby("label")}
    rows = []
    for x in il.labels():
        z = np.load(il.OUT_W / cr.WIDE / f"{x}.npz")
        d = il._load(x)
        cols = list(z["cols"])
        ri = [cols.index(r) for r in ROIS]
        keep = z["iti_gap"] >= il.GAP_MAIN
        bt = np.asarray(d["bout_t"], float)
        sets = {"cued": (z["lick"].astype(float), z["pos"].astype(str), TT.get(x, np.zeros(0)), False),
                "iti": (z["iti"].astype(float)[keep], cr.bout_prev_pos(x, bt)[keep], bt[keep], True)}
        for st, (X, pos, t, exact) in sets.items():
            if len(X) != len(t):
                raise ValueError(f"{x} {st}: {len(X)} traces vs {len(t)} lick times")
            if not len(X):
                continue
            L = pd.DataFrame({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "set": st, "pos": pos})
            for w0, w1 in VWIN:
                A = _auc(X[:, :, ri], w0, w1)
                for k, r in enumerate(ROIS):
                    L[f"auc_{r}@{w0:g}-{w1:g}"] = A[:, k]
            kin = _kin(x, t, exact)
            rows.append(pd.concat([L, kin], axis=1))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------------------------------- compute
def compute() -> dict:
    from wfield_local import analysis_kit as ak
    L = lick_table()
    pd.set_option("display.width", 220)
    cov = L.groupby(["set", "epoch"])[["reach", "dur_ms"]].apply(lambda g: g.notna().all(1).mean()).unstack()
    print("== share of licks with kinematics (reach + duration), set x epoch\n" + cov.reindex(columns=il.EP4).round(3)
          .to_string())
    out = []
    for _st, G in L.groupby("set"):
        G = G.copy()
        # scale: the animal's mean PRE AUC over all spouts (14.7's scale), and z-scores vs the animal's pre licks
        # every window in units of the animal's pre 0..0.5 s AUC (a later window's own pre AUC can be ~0 for ITI licks)
        for r, w in RW:
            G[f"y_{r}@{w}"] = G[f"auc_{r}@{w}"] / G.animal.map(G[G.epoch == "pre"].groupby("animal")[f"auc_{r}@0-0.5"].mean())
        for c in [*VIG, "side_ml"]:
            mu = G[G.epoch == "pre"].groupby("animal")[c].mean()
            sd = G[G.epoch == "pre"].groupby("animal")[c].std()
            G[f"z_{c}"] = (G[c] - G.animal.map(mu)) / G.animal.map(sd)
        full = G[[f"z_{c}" for c in MODELS["vigour+landing"]]].notna().all(1)
        G = G[full].reset_index(drop=True)                  # one common lick set for all three models
        for r, w in RW:
            for m, cols in MODELS.items():
                y = f"y_{r}@{w}"
                G[f"{m}|{r}@{w}"] = G[y] if not cols else il._pre_adjust(G, y, [f"z_{c}" for c in cols], MIN_PRE)
        out.append(G)
    L = pd.concat(out, ignore_index=True)
    L.to_csv(il.OUT_W / "vigor_perspout_licks.csv", index=False)
    print("== vigor per spout (z vs the animal's pre licks of the set; session means, mean over sessions)")
    S0 = L.groupby(["set", "pos", "label", "animal", "epoch"])[["z_reach", "z_speed", "z_dur_ms", "z_side_ml"]].mean()
    print(S0.groupby(["set", "pos", "epoch"]).mean().unstack("epoch").reindex(columns=il.EP4, level=1).round(2).to_string())
    tests = []
    for (st, p), G in L.groupby(["set", "pos"]):
        n = G.groupby("label").size()
        S = G[G.label.isin(n[n >= cr.MIN_SPOUT].index)].groupby(["label", "animal", "epoch"]).mean(numeric_only=True)\
            .reset_index()
        for r, w in RW:
            for m in MODELS:
                v = f"{m}|{r}@{w}"
                ref = {a: g[v].dropna().to_numpy() for a, g in S[S.epoch == "pre"].groupby("animal")}
                for ep in EP:
                    post = {a: g[v].dropna().to_numpy() for a, g in S[S.epoch == ep].groupby("animal")}
                    com = sorted(a for a in set(post) & set(ref) if len(post[a]) and len(ref[a]))
                    if len(com) < 3:
                        continue
                    iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
                    pa = {a: post[a].mean() - ref[a].mean() for a in com}
                    tests.append({"set": st, "spout": p, "roi": r, "window": w, "model": m, "epoch": ep, "delta": iv.point,
                                  "lo": iv.lo, "hi": iv.hi, "sig": bool(iv.lo > 0 or iv.hi < 0), "n_animals": len(com),
                                  **{f"d_{a}": pa.get(a, np.nan) for a in ("PS92", "PS93", "PS94", "PS95")}})
    T = pd.DataFrame(tests)
    T.to_csv(il.OUT_W / "vigor_perspout_tests.csv", index=False)
    T["cell"] = [f"{d:+.2f}{'*' if s else ''}" for d, s in zip(T.delta, T.sig)]
    print("== change vs pre at the spout (units of the animal's all-spout pre AUC), none / vigor / vigor+landing")
    print(T[T.roi.isin(["tjM1_right", "SSp-m_right"])].pivot_table(index=["set", "roi", "spout", "window"], columns=["epoch", "model"],
          values="cell", aggfunc="first").reindex(columns=EP, level=0).reindex(columns=list(MODELS), level=1).to_string())
    print("== per animal (rule 8), far contra, tjM1_right")
    print(T[(T.spout == "far_contra") & (T.roi == "tjM1_right")][["set", "window", "model", "epoch", "d_PS92", "d_PS93", "d_PS94",
                                                                     "d_PS95"]].round(2).to_string(index=False))
    return {"T": T}


# ---------------------------------------------------------------------------------------------------------------- plot
MCOL = {"none": "0.25", "vigour": "#2c7fb8", "vigour+landing": "#d95f0e"}
PROIS = ["tjM1_right", "SSp-m_right"]


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import iti_lick_order as lo
    T = F["T"]
    if "window" not in T:
        T = T.assign(window="0-0.5")
    pos = list(lo.POS)
    for wn in [f"{w0:g}-{w1:g}" for w0, w1 in VWIN]:
        rows = [(st, r) for st in ("cued", "iti") for r in PROIS]
        fig, axs = plt.subplots(len(rows), len(EP), figsize=(18, 13), sharey="row", squeeze=False)
        for i, (st, r) in enumerate(rows):
            for j, ep in enumerate(EP):
                ax = axs[i, j]
                for k, m in enumerate(MODELS):
                    t = T[(T.set == st) & (T.roi == r) & (T.window == wn) & (T.epoch == ep) & (T.model == m)].set_index("spout").reindex(pos)
                    x = np.arange(len(pos)) + (k - 1) * 0.22
                    ax.errorbar(x, t.delta, yerr=[t.delta - t.lo, t.hi - t.delta], fmt="none", ecolor=MCOL[m], capsize=2)
                    sig = t.sig.fillna(False).astype(bool).to_numpy()
                    ax.plot(x[sig], t.delta[sig], "o", color=MCOL[m], ms=6, zorder=3)
                    ax.plot(x[~sig], t.delta[~sig], "o", mec=MCOL[m], mfc="white", ms=6, zorder=3)
                ax.axhline(0, color="0.6", lw=0.7, ls="--")
                ax.set_xticks(range(len(pos)), [p.replace("_", "\n") for p in pos], fontsize=8)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if i == 0:
                    ax.set_title(ep, fontsize=11)
                if j == 0:
                    what = "CUED first lick (target spout)" if st == "cued" else "ITI bout start (PREVIOUS trial's spout)"
                    ax.set_ylabel(f"{what}\n{r}: epoch MINUS pre at the spout\n(0 = no change; units = animal's pre AUC)",
                                  fontsize=8.5)
        fig.legend(handles=[Line2D([], [], color=MCOL[m], marker="o", lw=0, label={"none": "unadjusted",
                   "vigour": "vigor regressed out", "vigour+landing": "vigor + landing regressed out"}[m]) for m in MODELS]
                   + [Line2D([], [], color="0.4", marker="o", mfc="white", lw=0, label="hollow = CI includes 0")],
                   loc="lower center", ncol=4, fontsize=9.5, frameon=False)
        fig.suptitle(f"LICK-VIGOR CONTROL for the per-spout responses (14.7): response AUC {wn.replace('-', '..')} s after lick onset, in units of the pre 0..0.5 s AUC (baseline "
                     "-1.0..-0.5 s), epoch MINUS pre AT THAT SPOUT (0 = no change; 1 = one whole pre response, all-spout average), mean of animals with 95% paired nested bootstrap CI.\nVigour = "
                     "3-D reach, peak speed, outward and height of the tip, lick duration, retraction speed; + landing = also "
                     "the lateral tip position; per animal, linear fit on PRE licks only (as 14.13);\none common lick set (licks "
                     "with every covariate) for all three. _right = contralesional. PROVISIONAL", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        save_png_svg(fig, il.OUT_W / "figures" / ("lick_vigor_perspout.png" if wn == "0-1" else f"lick_vigor_perspout_{wn}.png"), dpi=100)
        plt.close(fig)


def main() -> int:
    if "--times-cache" in sys.argv:
        times_cache(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8)
        return 0
    if plot_only():
        plot(load_fd("lick_vigor_perspout"))
        return 0
    F = compute()
    save_fd("lick_vigor_perspout", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
