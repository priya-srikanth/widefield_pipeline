"""Per-animal headline tables: FINAL LP arm vs the CURRENT LP-arm versions (rule 8: per animal before pooled).

Maps: amplitude ratio RMS(epoch mean map) / RMS(pre mean map), per animal (`restnl_maps.amplitude_table`), far contra.
    final   = lp_licks.output_dir(<LPfinal>/epoch) under lp_final_render.setup()/data/lpfinal_amplitude_per_animal_...csv
    current = <grant_figures>/LP_licks/epoch_restnl/data/restnl_amplitude_per_animal_<align>_<variant>.csv (11-01:
              LP licks, gate as before = LP-only imaging gate + DAQ-only rest gate, Aug pre; refs restnl/rest/restw/raw)
Decoders: far-contra accuracy per animal x epoch.
    final   = <LPfinal>/trained_<daq|lp>/<tag>/epoch/data/lpfinal_enl_trials_acc_per_animal.csv
    current = <LPfinal>/LP_licks/compare/current_LParm_enl1s_acc_per_animal.csv (plain LP_licks, June + Aug pre,
              in-memory fit = what the page's LP numbers are)
Writes <LPfinal>/compare/lpfinal_vs_current_*.csv and prints markdown tables.

    python scripts/lp_final_compare.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

AN = ["PS92", "PS93", "PS94", "PS95"]
EPOCHS = ["acute", "subacute", "chronic"]


def roots():
    from wfield_local.paths import PathResolver
    g = Path(PathResolver().root("cohort_out")) / "grant_figures"
    return g, g / "LPfinal"


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def final_maps_dir(lf):
    """The FINAL maps folder exactly as `lp_final_render` routes it (review M10, 11-03): its `setup()` env (LP source +
    LP gate + PRE_FROM + MAP_REFS) through `lp_licks.output_dir`, so the run-mode parts (pre0801, refs-...) the rebuild
    added are followed instead of reading the pre-rebuild LP_licks_gateLP/epoch file."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import lp_final_render
    from wfield_local import lp_licks
    with lp_final_render.scoped():                # the render env in this block only; the process env is restored
        return lp_licks.output_dir(lf / "epoch")


def maps(g, lf):
    rows = []
    fdir = final_maps_dir(lf)
    print(f"(final maps from {fdir})", flush=True)
    for align, var in (("precue", "working"), ("cue", "working"), ("lick", "lick")):
        f_new = fdir / "data" / f"lpfinal_amplitude_per_animal_{align}_{var}.csv"
        f_old = g / "LP_licks" / "epoch_restnl" / "data" / f"restnl_amplitude_per_animal_{align}_{var}.csv"
        if not f_new.exists():
            print(f"(maps {align}: final table missing)")
            continue
        new = pd.read_csv(f_new)
        old = pd.read_csv(f_old) if f_old.exists() else pd.DataFrame(columns=new.columns)
        for ref in sorted(set(new.reference)):
            for e in EPOCHS:
                r = {"align": align, "reference": ref, "epoch": e}
                for tag, d in (("final", new), ("current", old)):
                    s = d[(d.reference == ref) & (d.position == "far_R") & (d.epoch == e)].set_index("animal").ratio
                    for an in AN:
                        r[f"{an}_{tag}"] = float(s.get(an, float("nan")))
                    r[f"n<1_{tag}"] = int((s < 1).sum())
                rows.append(r)
    return pd.DataFrame(rows)


def decoders(lf):
    cur = lf / "LP_licks" / "compare" / "current_LParm_enl1s_acc_per_animal.csv"
    C = pd.read_csv(cur) if cur.exists() else None
    rows = []
    for f in sorted(lf.glob("trained_*/*/epoch/data/lpfinal_enl_trials_acc_per_animal.csv")):
        T = pd.read_csv(f)
        arm = f.parents[2].name
        for e in ["pre"] + EPOCHS:
            r = {"trained": T.trained.iloc[0], "arm": arm, "window_s": T.window_s.iloc[0], "epoch": e}
            for an in AN:
                x = T[(T.animal == an) & (T.epoch == e)]
                r[an] = float(x.acc_far_R.iloc[0]) if len(x) else float("nan")
                r[f"{an}_n"] = int(x.n_far_R.iloc[0]) if len(x) else 0
            rows.append(r)
    if C is not None:
        for e in ["pre"] + EPOCHS:
            r = {"trained": "lp (in-memory, current page)", "arm": "LP_licks (all pre)", "window_s": 1.0, "epoch": e}
            for an in AN:
                x = C[(C.animal == an) & (C.epoch == e)]
                r[an] = float(x.acc_far_R.iloc[0]) if len(x) else float("nan")
                r[f"{an}_n"] = int(x.n_far_R.iloc[0]) if len(x) else 0
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    g, lf = roots()
    out = lf / "compare"
    out.mkdir(parents=True, exist_ok=True)
    M = maps(g, lf)
    M.to_csv(out / "lpfinal_vs_current_maps_farcontra.csv", index=False)
    D = decoders(lf)
    D.to_csv(out / "lpfinal_vs_current_decoders_farcontra.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(M.round(2).to_string())
        print(D.round(3).to_string())
    print(md(M.round(2)))
    print(md(D[["trained", "arm", "window_s", "epoch"] + AN + [f"{a}_n" for a in AN]].round(3)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
