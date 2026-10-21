"""Mixed-effects time course of the POSITION encoding measures (Priya 10-20: "what about mixed model for position /
distance? are we using random intercept and random slope per animal").

Outcomes per session x area x window (dR2 x1e3, cortex + both hemispheres of the focus areas):
  position       full - drop:position                  (movement_first; order-free drop, the lead measure)
  side           full - shufside:position               (labels permuted WITHIN distance -> side information)
  distance       full - shufdist:position               (labels permuted WITHIN side -> distance information)
  spout:<p>      full - droppos-<p>:position            (movement_first_subset-droppos-; per-spout unique)
  far-close      mean(far spouts) - mean(close spouts)  of the per-spout values (does distance coding change?)
  contra-ipsi    mean(contra spouts) - mean(ipsi spouts)

Models (wfield_local/mixed_models.py -- the shared definition): `time` y ~ post + post:lday and `behaviour`
y ~ post + post:deficit (far-contra contact deficit), random intercept + step + slope per animal with the module's
fallback ladder; FDR (orofacial_stats.bh_fdr) within model x term x window; per-animal epoch means printed first.
Output: <run>/imaging_encoding/qc/encoding_mixed.csv. Curated sessions from 08-06 (August pre baseline).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

FOCUS = ("cortex", "SSp-m_left", "SSp-m_right", "MOs_left", "MOs_right", "MOp_left", "MOp_right", "SSp-ul_left",
         "SSp-ul_right")
WINDOWS = ("enl", "cue", "lick", "cue2lick", "cuestop", "firstlick")   # extra windows where a run has them



def _regions(folder: Path, labels: set) -> pd.DataFrame:
    fs = [f for f in glob.glob(str(folder / "sessions" / "*_regions.csv")) if Path(f).stem[:-8] in labels]
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return R[R.area.isin(FOCUS) & R.window.isin(WINDOWS)]


def outcomes(E: Path, labels: set) -> pd.DataFrame:
    rows = []
    R = _regions(E / "movement_first", labels)
    P = R.pivot_table(index=["label", "window", "area"], columns="model", values="r2")
    for name, m in (("position", "drop:position"), ("side", "shufside:position"), ("distance", "shufdist:position")):
        if m in P:
            rows.append(((P["full"] - P[m]) * 1e3).rename("y").reset_index().assign(measure=name))
    D = _regions(E / "movement_first_subset-droppos-", labels)
    Q = D.pivot_table(index=["label", "window", "area"], columns="model", values="r2")
    sp = {}
    for c in Q.columns:
        if c.startswith("droppos-") and c.endswith(":position"):
            p = c[len("droppos-"):-len(":position")]
            sp[p] = (Q["full"] - Q[c]) * 1e3
            rows.append(sp[p].rename("y").reset_index().assign(measure=f"spout:{p}"))
    if sp:
        mean = lambda keys: pd.concat([sp[k] for k in keys if k in sp], axis=1).mean(1)   # noqa: E731
        far, close = [k for k in sp if k.startswith("far")], [k for k in sp if k.startswith("close")]
        contra, ipsi = [k for k in sp if k.endswith("contra")], [k for k in sp if k.endswith("ipsi")]
        rows.append((mean(far) - mean(close)).rename("y").reset_index().assign(measure="far-close"))
        rows.append((mean(contra) - mean(ipsi)).rename("y").reset_index().assign(measure="contra-ipsi"))
    # ANGLE contest (Priya 10-20: "try this for ... position (or angle) analysis"): nokin target / executed measures,
    # from the cam4 contest folder and, when present, the 3-D polar contest (combined licks, soft bins)
    for tag, fold in (("cam4", "movement_first_subset-nokin"),
                      ("3d", "movement_first_subset-nokin_exec-polar_licks-combined_bins")):
        if not (E / fold / "sessions").exists():
            continue
        N = _regions(E / fold, labels).pivot_table(index=["label", "window", "area"], columns="model", values="r2")
        for name, (hi, lo) in (("ceiling", ("category", "none")), ("u_target", ("target+exec", "exec")),
                               ("u_exec", ("target+exec", "target")), ("u_exec_strict", ("category+exec", "category")),
                               ("u_tgtbin", ("tgtbin+exbin", "exbin")), ("u_exbin", ("tgtbin+exbin", "tgtbin"))):
            if f"nokin:{hi}" in N and f"nokin:{lo}" in N:
                rows.append(((N[f"nokin:{hi}"] - N[f"nokin:{lo}"]) * 1e3).rename("y").reset_index()
                            .assign(measure=f"{tag}:{name}"))
    from wfield_local import mixed_models as mm
    L = mm.add_time_covariates(pd.concat(rows, ignore_index=True).dropna(subset=["y"]))
    return mm.add_deficit(L, mm.behaviour_deficit(E / "qc" / "position_coding_trend_recovery_sessions.csv"))


FIG_PANELS = (("cue", "cam4:ceiling"), ("cue", "cam4:u_target"), ("cue", "position"),
              ("lick", "distance"), ("lick", "cam4:u_target"), ("lick", "position"))
ANIMAL_COL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}


def figures(L: pd.DataFrame, T: pd.DataFrame, out: Path, area: str = "cortex") -> None:
    """Session values vs behavioural deficit (and vs days since stroke) with the fitted fixed effects, per panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    for model, xcol, xlab in (("behaviour", "deficit", "far-contra contact deficit (0 = pre level, 1 = total loss)"),
                              ("time", "day", "days since stroke (pre sessions at their negative days)")):
        fig, ax = plt.subplots(2, 3, figsize=(13, 7.5), squeeze=False)
        for k, (win, meas) in enumerate(FIG_PANELS):
            a_ = ax.flat[k]
            G = L[(L.window == win) & (L.measure == meas) & (L.area == area)]
            if G.empty:
                a_.axis("off")
                continue
            for an, g in G.groupby("animal"):
                for post, mk in ((0, "o"), (1, "^")):
                    h = g[g.post == post]
                    a_.scatter(h[xcol], h.y, s=14, marker=mk, color=ANIMAL_COL.get(an, "0.4"), alpha=0.8,
                               label=f"{an} {'post' if post else 'pre'}" if k == 0 else None)
            c = T[(T.model == model) & (T.window == win) & (T.measure == meas) & (T.area == area)].set_index("term")
            if {"pre_level", "post"} <= set(c.index):
                slope = "post:deficit" if model == "behaviour" else "post:lday"
                pre, step, b = c.coef["pre_level"], c.coef["post"], c.coef.get(slope, np.nan)
                if model == "behaviour":
                    xx = np.linspace(min(0, G.deficit.min()), max(1, G.deficit.max()), 50)
                    a_.plot(xx, pre + step + b * xx, "k-", lw=1.5, label="fit: post" if k == 0 else None)
                else:
                    xx = np.linspace(1, max(G.day.max(), 2), 80)
                    a_.plot(xx, pre + step + b * np.log(xx), "k-", lw=1.5, label="fit: post" if k == 0 else None)
                a_.axhline(pre, color="k", ls=":", lw=1, label="fit: pre level" if k == 0 else None)
                a_.set_title(f"{meas} | {win} window\npost {step:+.1f} (z {c.z['post']:.1f}), {slope} {b:+.1f} "
                             f"(z {c.z.get(slope, np.nan):.1f}); re {c.re.iloc[0]}", fontsize=8)
            a_.axhline(0, color="0.8", lw=0.5)
            a_.set_xlabel(xlab, fontsize=7)
            a_.set_ylabel(f"{area} dR2 x1e3", fontsize=7)
            a_.tick_params(labelsize=7)
        ax.flat[0].legend(fontsize=6, ncol=2)
        fig.suptitle(f"Position / angle coding vs {'behavioural recovery' if model == 'behaviour' else 'time'} "
                     f"({area}; mixed model, animal random effects; o pre, ^ post sessions)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"encoding_mixed_{model}_{area}.png", dpi=150)
        plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    a = ap.parse_args(argv)
    E = oo.root(PathResolver()) / a.run / "imaging_encoding"
    labels = {x for x in ak.curated_labels() if x[-4:] >= "0806"}
    L = outcomes(E, labels)
    from wfield_local import mixed_models as mm
    by = ["measure", "window", "area"]
    T = mm.lmm_table(L[L.area == "cortex"], by, print_animals=True)           # rule 8 table: cortex only (readable)
    T = pd.concat([T, mm.lmm_table(L[L.area != "cortex"], by, print_animals=False)], ignore_index=True)
    from wfield_local.orofacial_stats import bh_fdr                       # FDR over ALL areas within model x term x window
    for _, idx in T.groupby(["model", "term", "window"]).groups.items():
        T.loc[idx, "q"] = bh_fdr(T.loc[idx, "p"].to_numpy())
    T.to_csv(E / "qc" / "encoding_mixed.csv", index=False, float_format="%.5g")
    mm.summary_table(T, ["window", "area", "measure"]).to_csv(E / "qc" / "encoding_mixed_summary.csv")
    pd.set_option("display.width", 230)
    for model in ("time", "behaviour"):
        for win in WINDOWS:
            S = T[(T.model == model) & (T.window == win) & (T.area == "cortex") & T.term.notna()]
            if S.empty:
                continue
            print(f"\n=== [{model}] {win} window, cortex: dR2 x1e3; q = FDR over measures x areas")
            print(S.pivot_table(index="measure", columns="term", values=["coef", "z", "q"]).round(3).to_string())
    print("\n  random-effect structure used:", T.groupby("re").size().to_dict())
    for area in ("cortex", "SSp-m_left", "SSp-m_right"):
        figures(L, T, E / "qc" / "figures", area)
    print(f"\n-> {E / 'qc' / 'encoding_mixed.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
