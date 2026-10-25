"""Trial-to-trial vs PETH variance WITH FaceRhythm (Priya 10-25: "what about PETH vs trial-by-trial variance?").

Source: O2 run fr_A_tr (enc_v36_full_20261024; FR models added to `combined_encoding.TR_MODELS`), collected to
`movement_first_fr-master_pre_v1_frg-A_subset-fr_exec-polar_licks-combined_tr/`. Per session x area x time (+-3 s):
  sse / sst              frame error / total (all frames of the trials)
  sse_trial / sst_trial  the same after removing each spout's trial-averaged response from data AND prediction
                         (variability AROUND the condition mean = trial-to-trial)
and the EXACT split frame = trial-to-trial + condition mean, so PETH error = sse - sse_trial, PETH total = sst - sst_trial.
Per window (cue 0 .. 1.5 s from the cue; lick -0.1 .. 1.0 s from the first post-cue lick, `*_lick`), summed over time:
  R2_frame = 1 - SSE / SST;  R2_trial = 1 - SSE_tr / SST_tr;  R2_peth = 1 - (SSE - SSE_tr) / (SST - SST_tr)
for models full (no FR), fr:full, fr:alone, fr:drop:<g>. Quantities (R2 as a fraction): full_fr, alone_fr,
unique_fr = fr:full - full, uniquefr_<g> = fr:full - fr:drop:<g> (g = task, instructed, uninstructed, position).

Usage: python scripts/fr_trial_peth.py
Output: <E>/fr_encoding_summary/trial_peth/ (sessions.csv, per_animal.csv, bootstrap.csv, figures/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E  # noqa: E402

D = E / "movement_first_fr-master_pre_v1_frg-A_subset-fr_exec-polar_licks-combined_tr" / "sessions"
WIN = {"cue": ("", 0.0, 1.5), "lick": ("_lick", -0.1, 1.0)}
G = ["task", "instructed", "uninstructed", "position"]
Q = ["full_fr", "alone_fr", "unique_fr"] + [f"uniquefr_{g}" for g in G]
MEAS = ["frame", "trial", "peth"]
EPS = ["acute", "subacute", "chronic"]
AREAS = ["cortex", "SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left", "MOs_right", "MOs_left", "SSp-n_right",
         "SSp-n_left", "SSp-ul_right", "SSp-ul_left"]


def session_rows(jf: Path) -> list[dict]:
    from wfield_local import config, epochs
    m = json.loads(jf.read_text())
    lab = m["label"]
    if lab[-4:] < "0801":
        return []
    z = np.load(jf.with_suffix(".npz"))
    off = z["peth_off_s"]
    ix = {k: i for i, k in enumerate(m["tr_models"])}
    rows = []
    for w, (suf, t0, t1) in WIN.items():
        tm = (off >= t0) & (off <= t1)
        sse = z[f"sse{suf}"][:, :, tm].sum(-1)
        sst = z[f"sst{suf}"][:, tm].sum(-1)
        sse_t = z[f"sse_trial{suf}"][:, :, tm].sum(-1)
        sst_t = z[f"sst_trial{suf}"][:, tm].sum(-1)
        r2 = {"frame": 1 - sse / sst, "trial": 1 - sse_t / sst_t, "peth": 1 - (sse - sse_t) / (sst - sst_t)}
        for ai, ar in enumerate(m["areas"]):
            for meas, R in r2.items():
                r = {"label": lab, "animal": config.animal_of(lab), "epoch": epochs.epoch_of(lab), "window": w,
                     "area": ar, "measure": meas, "peth_share": float((sst[ai] - sst_t[ai]) / sst[ai]),
                     "full_fr": R[ix["fr:full"], ai], "alone_fr": R[ix["fr:alone"], ai],
                     "unique_fr": R[ix["fr:full"], ai] - R[ix["full"], ai]}
                for g in G:
                    r[f"uniquefr_{g}"] = R[ix["fr:full"], ai] - R[ix[f"fr:drop:{g}"], ai]
                rows.append(r)
    return rows


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import analysis_kit as ak
    out = E / "fr_encoding_summary" / "trial_peth"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    T = pd.DataFrame([r for jf in sorted(D.glob("PS*_*.json")) for r in session_rows(jf)])
    T.to_csv(out / "sessions.csv", index=False)
    print(f"{T.label.nunique()} sessions", flush=True)
    A = T.groupby(["window", "area", "measure", "animal", "epoch"])[Q + ["peth_share"]].mean().reset_index()
    A.to_csv(out / "per_animal.csv", index=False)
    pd.set_option("display.width", 250)

    # pre-stroke levels, cortex
    lv = A[(A.area == "cortex") & (A.epoch == "pre")].groupby(["window", "measure"])[Q + ["peth_share"]].mean()
    print("== pre-stroke level, cortex, mean of animals (R2 fraction; peth_share = PETH variance / frame variance)")
    print(lv.round(3).to_string())

    # epoch changes: per animal + nested bootstrap
    rows = []
    for (w, ar, meas), t in T[T.area.isin(AREAS)].groupby(["window", "area", "measure"]):
        for q in Q:
            pre = {a: t[(t.epoch == "pre") & (t.animal == a)][q].dropna().to_numpy() for a in AN}
            for ep in EPS:
                post = {a: t[(t.epoch == ep) & (t.animal == a)][q].dropna().to_numpy() for a in AN}
                com = [a for a in AN if len(pre[a]) and len(post[a])]
                if not com:
                    continue
                iv, d = ak.boot_delta({a: post[a] for a in com}, {a: pre[a] for a in com}, np.random.default_rng(0),
                                      draws=True)
                per = {a: float(post[a].mean() - pre[a].mean()) for a in com}
                rows.append({"window": w, "area": ar, "measure": meas, "q": q, "epoch": ep, "point": iv.point,
                             "lo": iv.lo, "hi": iv.hi, "p": float(ak.boot_p(d)), "n_up": sum(v > 0 for v in per.values()),
                             "n": len(com), **per})
    B = pd.DataFrame(rows)
    B.to_csv(out / "bootstrap.csv", index=False)
    key = B[(B.epoch == "chronic") & B.q.isin(["uniquefr_position", "uniquefr_task", "uniquefr_uninstructed", "unique_fr"])
            & B.area.isin(["cortex", "SSp-m_right", "SSp-m_left"])]
    print("\n== chronic - pre, key quantities (point [CI], p uncorrected, animals up)")
    print(key[["window", "area", "q", "measure", "point", "lo", "hi", "p", "n_up", "n"] + AN].round(3).to_string(index=False))

    # figure 1: pre-stroke level per measure, cortex
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharey="row")
    labs = ["full | FR", "FR alone", "FR unique", "task | FR", "instr. | FR", "uninstr. | FR", "position | FR"]
    for i, w in enumerate(WIN):
        for j, meas in enumerate(MEAS):
            ax = axs[i, j]
            t = A[(A.area == "cortex") & (A.epoch == "pre") & (A.window == w) & (A.measure == meas)]
            for k, q in enumerate(Q):
                ax.bar(k, t[q].mean(), color="#1b9e77" if "fr" in q.split("_")[-1] or q == "alone_fr" else "#7570b3",
                       width=0.65)
                ax.scatter(np.full(len(t), k) + np.linspace(-0.15, 0.15, len(t)), t[q], s=12, c="k", zorder=5)
            ax.set_xticks(range(len(Q))); ax.set_xticklabels(labs, rotation=30, fontsize=8, ha="right")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{w} window, {meas}" + (" (trial-to-trial)" if meas == "trial" else " (trial-averaged)" if meas == "peth" else ""),
                         fontsize=9)
            if j == 0:
                ax.set_ylabel("held-out R2 (fraction), cortex, pre")
    fig.suptitle("NEURAL: per-session encoder WITH FaceRhythm (mode A), pre-stroke cortex: frame vs trial-to-trial vs PETH "
                 "(trial-averaged) R2.\nunique = fr:full minus the model without the group; FR unique = fr:full minus full. "
                 "Bars = mean of animals, dots = animals.", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.14, hspace=0.6, wspace=0.08)
    fig.savefig(out / "figures" / "fr_trial_peth_pre_levels.png", dpi=115)
    plt.close(fig)

    # figure 2: chronic - pre, trial vs PETH, cortex / contra SSp-m / ipsi SSp-m
    qs = ["unique_fr", "uniquefr_uninstructed", "uniquefr_task", "uniquefr_position"]
    ql = ["FR unique", "uninstructed | FR", "task | FR", "position | FR"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharey="row")
    for i, w in enumerate(WIN):
        for j, ar in enumerate(["cortex", "SSp-m_right", "SSp-m_left"]):
            ax = axs[i, j]
            for k, q in enumerate(qs):
                for m_, (meas, col) in enumerate((("trial", "#d95f02"), ("peth", "#1b9e77"))):
                    r = B[(B.window == w) & (B.area == ar) & (B.measure == meas) & (B.q == q) & (B.epoch == "chronic")]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    x = k + (m_ - 0.5) * 0.36
                    ax.bar(x, r.point, width=0.34, color=col, label=meas if (k == 0 and i == 0 and j == 0) else None)
                    ax.plot([x, x], [r.lo, r.hi], color="k", lw=1)
                    ax.scatter(np.full(4, x), [r.get(a, np.nan) for a in AN], s=10, c="w", edgecolors="k", zorder=5)
            ax.set_xticks(range(len(qs))); ax.set_xticklabels(ql, fontsize=8, rotation=20)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{w} window, {ar.replace('_right', ' contralesional').replace('_left', ' ipsilesional')}", fontsize=9)
            if j == 0:
                ax.set_ylabel("chronic - pre, R2 (fraction)")
    axs[0, 0].legend(["trial-to-trial", "PETH (trial-averaged)"], fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("NEURAL: chronic minus pre, per-session encoder WITH FaceRhythm: trial-to-trial (orange) vs PETH (green) "
                 "unique R2.\nBars = mean of animals [95% nested bootstrap CI], dots = animals.", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.1, hspace=0.45, wspace=0.08)
    fig.savefig(out / "figures" / "fr_trial_peth_chronic.png", dpi=115)
    plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
