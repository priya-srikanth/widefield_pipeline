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
y ~ post + post:deficit (mixed_models.DEFICITS; default ttc), random intercept + step + slope per animal with the module's
fallback ladder; FDR (orofacial_stats.bh_fdr) within model x term x window; per-animal epoch means printed first.
Extra windows (cue2lick = cue .. first lick, cuestop = cue .. trial stop, firstlick = first post-cue lick -0.1 .. +1 s)
come from the O2 `_win` runs: position from movement_first_subset-drop_win, cam4 contest from ..._nokin_win.
Output: <run>/imaging_encoding/qc/encoding_mixed.csv. Curated sessions from 08-06 (August pre baseline).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

FOCUS = ("cortex", "SSp-m_left", "SSp-m_right", "MOs_left", "MOs_right", "MOp_left", "MOp_right", "SSp-ul_left",
         "SSp-ul_right")
WINDOWS = ("enl", "cue", "lick", "cue2lick", "cuestop", "firstlick")   # extra windows where a run has them
EXTRA = ("cue2lick", "cuestop", "firstlick")    # only in the O2 `_win` runs (drop_win: position; nokin_win: contest)



def _regions(folder: Path, labels: set) -> pd.DataFrame:
    fs = [f for f in glob.glob(str(folder / "sessions" / "*_regions.csv")) if Path(f).stem[:-8] in labels]
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return R[R.area.isin(FOCUS) & R.window.isin(WINDOWS)]


def outcomes(E: Path, labels: set, deficit: str | None = None) -> pd.DataFrame:
    rows = []
    R = _regions(E / "movement_first", labels)
    P = R.pivot_table(index=["label", "window", "area"], columns="model", values="r2")
    for name, m in (("position", "drop:position"), ("side", "shufside:position"), ("distance", "shufdist:position")):
        if m in P:
            rows.append(((P["full"] - P[m]) * 1e3).rename("y").reset_index().assign(measure=name))
    if (E / "movement_first_subset-drop_win" / "sessions").exists():   # extra windows: position only (ONLY=drop:)
        W = _regions(E / "movement_first_subset-drop_win", labels)
        W = W[W.window.isin(EXTRA)].pivot_table(index=["label", "window", "area"], columns="model", values="r2")
        if "drop:position" in W:
            rows.append(((W["full"] - W["drop:position"]) * 1e3).rename("y").reset_index().assign(measure="position"))
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
    for tag, fold, wins in (("cam4", "movement_first_subset-nokin", WINDOWS),
                            ("cam4", "movement_first_subset-nokin_win", EXTRA),        # extra windows (O2 _win run)
                            ("3d", "movement_first_subset-nokin_exec-polar_licks-combined_bins", WINDOWS)):
        if not (E / fold / "sessions").exists():
            continue
        N = _regions(E / fold, labels)
        N = N[N.window.isin(wins)].pivot_table(index=["label", "window", "area"], columns="model", values="r2")
        for name, (hi, lo) in (("ceiling", ("category", "none")), ("u_target", ("target+exec", "exec")),
                               ("u_exec", ("target+exec", "target")), ("u_exec_strict", ("category+exec", "category")),
                               ("u_tgtbin", ("tgtbin+exbin", "exbin")), ("u_exbin", ("tgtbin+exbin", "tgtbin"))):
            if f"nokin:{hi}" in N and f"nokin:{lo}" in N:
                rows.append(((N[f"nokin:{hi}"] - N[f"nokin:{lo}"]) * 1e3).rename("y").reset_index()
                            .assign(measure=f"{tag}:{name}"))
    from wfield_local import mixed_models as mm
    L = mm.add_time_covariates(pd.concat(rows, ignore_index=True).dropna(subset=["y"]))
    return mm.add_deficit(L, mm.deficit(E / "qc", deficit or mm.DEFICIT))


FIG_PANELS = (("cue", "cam4:ceiling"), ("cue", "cam4:u_target"), ("cue", "position"),
              ("lick", "distance"), ("lick", "cam4:u_target"), ("lick", "position"))
def figures(L: pd.DataFrame, T: pd.DataFrame, out: Path, area: str = "cortex", sfx: str = "") -> None:
    """Session values vs behavioural deficit and vs days, fitted fixed effects (mixed_models.plot_fits)."""
    from wfield_local import mixed_models as mm
    panels = [(f"{meas} | {win} window", {"window": win, "measure": meas, "area": area}) for win, meas in FIG_PANELS]
    for model in ("behaviour", "time"):
        mm.plot_fits(L, T, panels, model, out / f"encoding_mixed_{model}_{area}{sfx}.png",
                     f"Position / angle coding vs {'behavioural recovery' if model == 'behaviour' else 'time'} ({area})",
                     f"{area} dR2 x1e3")


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--deficit", default=None, help="behavioural index (mixed_models.DEFICITS); default = primary, "
                    "others write *_<name> outputs")
    a = ap.parse_args(argv)
    E = oo.root(PathResolver()) / a.run / "imaging_encoding"
    labels = {x for x in ak.curated_labels() if x[-4:] >= "0806"}
    from wfield_local import mixed_models as mm
    dname = a.deficit or mm.DEFICIT
    sfx = "" if dname == mm.DEFICIT else f"_{dname}"
    L = outcomes(E, labels, dname)
    by = ["measure", "window", "area"]
    T = mm.lmm_table(L[L.area == "cortex"], by, print_animals=True)           # rule 8 table: cortex only (readable)
    T = pd.concat([T, mm.lmm_table(L[L.area != "cortex"], by, print_animals=False)], ignore_index=True)
    from wfield_local.orofacial_stats import bh_fdr                       # FDR over ALL areas within model x term x window
    for _, idx in T.groupby(["model", "term", "window"]).groups.items():
        T.loc[idx, "q"] = bh_fdr(T.loc[idx, "p"].to_numpy())
    T.to_csv(E / "qc" / f"encoding_mixed{sfx}.csv", index=False, float_format="%.5g")
    mm.summary_table(T, ["window", "area", "measure"]).to_csv(E / "qc" / f"encoding_mixed_summary{sfx}.csv")
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
        figures(L, T, E / "figures" / "mixed_models", area, sfx)
    print(f"\n-> {E / 'qc' / 'encoding_mixed.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
