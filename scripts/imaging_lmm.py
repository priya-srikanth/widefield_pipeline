"""Session-level mixed models for the imaging / behaviour tables that already carry one value per session (Priya
2026-10-20: "take a close look at the current imaging analyses - decide where lmm approach would make sense and prepare
code to do so ... avoiding redundancy"). Models and conventions: wfield_local/mixed_models.py (the shared definition).

WHICH ANALYSES, AND WHY THESE (survey 10-20, DECISIONS):
  trend      qc/position_coding_trend_recovery_sessions.csv (scripts/position_coding_trend_recovery.py): the position
             ceiling (category - none) and the per-spout spread, pre + post sessions -> time + behaviour models
  decoder    grant_figures/epoch/data/recovery_trajectory_{unmatched,matched}_cue_working.csv
             (wfield_local/recovery_trajectory.py): frozen / refit decoder recall, gap, F_deficit, G_reorg -- POST rows
             only, already expressed against the animal's pre -> post-only models y ~ lday and y ~ deficit
             (random intercept + slope per animal when estimable)
  behaviour  cohort/tables/prepost/session_metrics.csv (wfield_local/orofacial_cohort.py): response rate, contact,
             latency, pre-cue licking, ipsi bias ... -> time model only (the deficit is itself behaviour)
  epoch      grant_figures/epoch/data/*_sessions_long.csv (scripts/epoch_sessions_long.py, other window 10-21): the epoch
             scalar families 8g / 10 / 10b / 11 / 11amp / 11pos / 11c per session (pre sessions scored leave-one-out,
             August pre) -> time / behaviour / epoch models; source = `epoch:<family>`, measure = "<measure> | <arm> |
             <position>". CD window projections (family cd): PRE ROWS ARE IN-SAMPLE (directions fitted on them), so
             CD gets the post-only models only (y ~ lday, y ~ deficit), never a step from pre.
ALREADY ON THE MODULE: the encoding measures (scripts/encoding_mixed.py) and the within-spout slopes
(scripts/within_spout_mixed.py).
NOT HERE: pixel maps keep beta_maps.significance_contour / vs_zero_contour.

Output: <run>/imaging_encoding/qc/imaging_lmm.csv + imaging_lmm_summary.csv.
Usage: PYTHONPATH=. python scripts/imaging_lmm.py [--sources trend,decoder,behaviour,epoch]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BEHAVIOUR_METRICS = ("resp_rate", "contact_frac", "latency_contact_ms", "latency_ms", "precue_rate_hz",
                     "precue_trial_frac", "iti_rate_hz", "ipsi_bias_deg", "angle_gain_far", "reach_frac", "protrusion_px")
DECODER_METRICS = ("frozen", "refit", "gap", "F_deficit", "G_reorg", "refit_deficit")


def _long(S: pd.DataFrame, cols, source: str) -> pd.DataFrame:
    cols = [c for c in cols if c in S]
    L = S.melt(id_vars=["label"], value_vars=cols, var_name="measure", value_name="y").dropna(subset=["y"])
    return L.assign(source=source)


def trend_table(E: Path) -> pd.DataFrame:
    S = pd.read_csv(E / "qc" / "position_coding_trend_recovery_sessions.csv")
    return _long(S, ("ceiling", "spread", "spread_enl"), "trend")


def behaviour_table(run_root: Path) -> pd.DataFrame:
    S = pd.read_csv(run_root / "cohort" / "tables" / "prepost" / "session_metrics.csv")
    return _long(S, BEHAVIOUR_METRICS, "behaviour")


def decoder_table() -> pd.DataFrame:
    """Post-stroke rows of the recovery trajectory, with the session label recovered from (animal, day)."""
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    from wfield_local import figure_layout as fl
    from wfield_local.paths import PathResolver
    d = Path(PathResolver().root("labcams")) / "grant_figures" / "epoch"
    lab_of = {(x[:4], epochs.days_since_stroke(x)): x for x in ak.curated_labels(order="sorted")}
    out = []
    for tag in ("unmatched", "matched"):
        p = fl.sidecar_for(d, f"recovery_trajectory_{tag}_cue_working", ".csv", mkdir=False)
        if not p.exists():
            continue
        R = pd.read_csv(p)
        R["label"] = [lab_of.get((a, int(dd))) for a, dd in zip(R.animal, R.day)]
        out.append(_long(R.dropna(subset=["label"]), DECODER_METRICS, f"decoder_{tag}"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def epoch_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(families with LOO pre rows, CD rows) in the long format, from the `*_sessions_long.csv` files."""
    from wfield_local import orofacial_cohort as oc
    from wfield_local.paths import PathResolver
    d = Path(PathResolver().root("labcams")) / "grant_figures" / "epoch" / "data"
    fs = sorted(d.glob("*_sessions_long.csv"))
    if not fs:
        return pd.DataFrame(), pd.DataFrame()
    R = pd.concat([pd.read_csv(f, dtype={"family": str}) for f in fs], ignore_index=True)
    pos = R.position.fillna("").astype(str).map(lambda p: oc.pos_key(p, "L") if p.split("_")[-1] in ("L", "R", "center") else p)
    R = R.assign(source="epoch:" + R.family, y=R.value,
                 measure=R.measure + " | " + R.arm + " / " + R.variant.astype(str) + np.where(pos != "", " | " + pos, ""))
    R = R.dropna(subset=["y"])[["label", "source", "measure", "y"]]
    return R[R.source != "epoch:cd"], R[R.source == "epoch:cd"]


def post_only_table(L: pd.DataFrame) -> pd.DataFrame:
    """Post-stroke-only sources: y ~ lday (time) and y ~ deficit (behaviour), random intercept (+ slope)."""
    from wfield_local import mixed_models as mm
    from wfield_local.orofacial_stats import bh_fdr
    rows = []
    for model, term in (("time_post", "lday"), ("behaviour_post", "deficit")):
        for (src, meas), G in L.groupby(["source", "measure"]):
            G = G.dropna(subset=[term])
            if len(G) < mm.MIN_POST or G.animal.nunique() < mm.MIN_ANIMALS:
                continue
            r, re = mm.fit_lmm(G, f"y ~ {term}", term, ladder=(f"~{term}", "~1"))
            for row in mm.tidy(r, re, ["Intercept", term], len(G), term):
                rows.append({"model": model, "source": src, "measure": meas,
                             **{**row, "term": "post_level_day1" if row["term"] == "pre_level" and model == "time_post"
                                else ("post_level_full_recovery" if row["term"] == "pre_level" else row["term"])}})
    T = pd.DataFrame(rows)
    if not T.empty:
        T["q"] = np.nan
        for _, idx in T.groupby(["model", "term"]).groups.items():
            T.loc[idx, "q"] = bh_fdr(T.loc[idx, "p"].to_numpy())
    return T


def figures(L: pd.DataFrame, T: pd.DataFrame, figs: Path, sfx: str = "") -> None:
    """One figure per source x model (mixed_models.plot_fits)."""
    from wfield_local import mixed_models as mm
    sets = [("trend", ("ceiling", "spread", "spread_enl"), ("time", "behaviour"),
             "position ceiling / spread (trend table)"),
            ("behaviour", ("resp_rate", "contact_frac", "latency_ms", "ipsi_bias_deg", "angle_gain_far",
                           "precue_rate_hz"), ("time",), "behaviour (orofacial session metrics)")]
    sets += [(f"decoder_{t}", DECODER_METRICS, ("time_post", "behaviour_post"),
              f"frozen / refit decoder ({t}; post-stroke only)") for t in ("unmatched", "matched")]
    ep = [f"{m} | {arm}" for arm in ("cue / working", "lick / lick") for m in ("frozen EV", "EV after rescale")]
    sets += [("epoch:11", ep, ("time", "behaviour"), "epoch family 11: frozen pre-stroke encoder EV (LOO pre)"),
             ("epoch:10", [f"best-match accuracy | {a}" for a in ("cue / working", "lick / lick")],
              ("time", "behaviour"), "epoch family 10: best-match accuracy"),
             ("epoch:10b", [f"best match is own position | cue / working | {p}" for p in ("fC", "fM", "fI")],
              ("time", "behaviour"), "epoch family 10b: far positions best-match own"),
             ("epoch:11c", [f"{m} | {a}" for a in ("cue / working", "lick / lick") for m in ("ceiling", "frozen (matched)")],
              ("time", "behaviour"), "epoch family 11c: encoder ceiling vs frozen (matched)")]
    for src, meas, models, title in sets:
        for model in models:
            if not ((T.source == src) & (T.model == model)).any():
                continue
            panels = [(m, {"source": src, "measure": m}) for m in meas]
            mm.plot_fits(L, T, panels, model, figs / f"imaging_lmm_{src.replace(':', '-')}_{model}{sfx}.png", title, "value")


def main(argv=None) -> int:
    from wfield_local import mixed_models as mm
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--sources", default="trend,decoder,behaviour,epoch")
    ap.add_argument("--deficit", default=None, help="behavioural index (mixed_models.DEFICITS); non-default -> *_<name>")
    a = ap.parse_args(argv)
    run_root = oo.root(PathResolver()) / a.run
    E = run_root / "imaging_encoding"
    dname = a.deficit or mm.DEFICIT
    sfx = "" if dname == mm.DEFICIT else f"_{dname}"
    deficit = mm.deficit(E / "qc", dname)
    src = a.sources.split(",")
    tabs, L, D, C = [], None, None, None
    prepost = [f(E if f is trend_table else run_root) for name, f in (("trend", trend_table),
                                                                      ("behaviour", behaviour_table)) if name in src]
    if prepost:
        L = mm.add_time_covariates(pd.concat(prepost, ignore_index=True))
        L = mm.add_deficit(L, deficit)
        L.loc[L.source == "behaviour", "deficit"] = np.nan          # behaviour vs a behavioural deficit = circular
        tabs.append(mm.lmm_table(L, ["source", "measure"], ("time", "behaviour", "epoch"), fdr_by=("source",)))
    if "epoch" in src:
        F, C = epoch_tables()
        if not F.empty:
            Le = mm.add_deficit(mm.add_time_covariates(F), deficit)
            tabs.append(mm.lmm_table(Le, ["source", "measure"], ("time", "behaviour", "epoch"), fdr_by=("source",),
                                     print_animals=False))
            L = Le if L is None else pd.concat([L, Le], ignore_index=True)
        if not C.empty:                                     # CD: pre rows in-sample -> post-only models
            C = mm.add_time_covariates(C)
            C = C[C.post == 1].assign(deficit=lambda x: x.label.map(deficit))
            tabs.append(post_only_table(C))
    if "decoder" in src:
        D = decoder_table()
        if not D.empty:
            D = mm.add_time_covariates(D)
            D = D[D.post == 1].assign(deficit=lambda x: x.label.map(deficit))
            print("  decoder (post-stroke only), per-animal epoch means\n"
                  + mm.animal_epoch_table(D, ["source", "measure"]).round(4).to_string())
            tabs.append(post_only_table(D))
    T = pd.concat([t for t in tabs if not t.empty], ignore_index=True)
    T.to_csv(E / "qc" / f"imaging_lmm{sfx}.csv", index=False, float_format="%.5g")
    mm.summary_table(T, ["source", "measure"]).to_csv(E / "qc" / f"imaging_lmm_summary{sfx}.csv")
    pd.set_option("display.width", 230)
    for model in T.model.unique():
        S = T[T.model == model]
        print(f"\n=== {model}\n" + S.pivot_table(index=["source", "measure"], columns="term",
                                                values=["coef", "z", "q"]).round(3).to_string())
    figures(pd.concat([x for x in (L, D, C) if x is not None and not x.empty], ignore_index=True), T,
            E / "figures" / "mixed_models", sfx)
    print(f"\n-> {E / 'qc' / 'imaging_lmm.csv'}, {E / 'figures' / 'mixed_models'}/imaging_lmm_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
