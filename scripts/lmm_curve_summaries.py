"""LMM versions of the CURVE figures (Priya 11-03): each session's curve reduced to the summaries PRE-SPECIFIED in
DECISIONS 2026-11-03 ("Curve figures: mixed model on PRE-SPECIFIED per-session summaries"), then
`scripts.lmm_change_fig` (epoch_change_lmm, Holm within each summary panel). Writes <stem>_long.csv (the per-session
summaries) and <stem>_lmm.png / _lmm_lmm.csv into --out.

    python -m scripts.lmm_curve_summaries --out M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/lmm_hidden_20261103
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from wfield_local.pose3d import pose3d_dir

POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
P3 = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra", "far_L": "far_ipsi",
      "far_center": "far_mid", "far_R": "far_contra"}


def _flip(df: pd.DataFrame) -> pd.Series:
    """+1 / -1 per row so a stored + = image-right (= ipsi for a LEFT lesion) lateral reads + = contra."""
    from wfield_local import config
    lat = {a: str((v or {}).get("stroke_laterality", "L")) for a, v in config.animals().items()}
    return df.animal.map(lambda a: -1.0 if lat.get(a, "L") == "L" else 1.0)


def cam4(base: Path) -> dict:
    from wfield_local import orofacial_cohort as oc
    out = {}
    L = oc.baseline_filter(pd.read_csv(base / "lick_phase.csv", dtype={"date": str}))
    ph = L[(L.set == "postcue_norm") & L.u.between(0.1, 0.9)]
    s = ph.groupby(["animal", "date", "epoch", "pos"]).lat_out_px.mean().reset_index()
    out["l_phase_all"] = (s.assign(value=s.lat_out_px * _flip(s)), "lateral tongue tip over lick phase 0.1-0.9, "
                          "in-mouth excluded (kin 2.5)", "lateral - own pre (px; + = contra)")
    tm = L[(L.set == "postcue") & L.t_ms.between(-72, 72)]
    s = tm.groupby(["animal", "date", "epoch", "pos"]).lat_out_px.mean().reset_index()
    out["p_paths_2d"] = (s.assign(value=s.lat_out_px * _flip(s)), "2-D path: lateral tip -72..+72 ms around the "
                         "lick peak, in-mouth excluded (kin 4.2)", "lateral - own pre (px; + = contra)")
    C = oc.baseline_filter(pd.read_csv(base / "cue_aligned_traces.csv", dtype={"date": str}))
    C = C[C.t_ms.between(70, 1500)]
    for stem, var, lab in (("l_cue_excl", "lat_out_px", "in-mouth excluded (kin 2.1)"),
                           ("l_cue_incl", "lat_px", "in-mouth included (kin 2.2)")):
        s = C[C["var"] == var].groupby(["animal", "date", "epoch", "pos"]).value.mean().reset_index()
        out[stem] = (s.assign(value=s.value * _flip(s)), f"cue-aligned lateral tongue, cue +70 to +1500 ms, {lab}",
                     "lateral - own pre (px; + = contra)")
    return out


def pose3d(curves: Path) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import kinematics_pooled_figs as K
    C = pd.read_csv(curves)
    C = C[(C.kind == "norm") & (C.lickset == "all") & (C["cov"] >= 0.5)]
    C = C[(C.epoch != "pre") | (C.date.astype(str).str[-4:] >= "0801")]
    lo, hi = (0.1, 0.9) if C.x.max() <= 1.0 + 1e-9 else (10, 90)          # phase stored as 0-1 or 0-100
    C = C[C.x.between(lo, hi)]
    g = ["animal", "session", "epoch", "position"]
    out = {}

    def summ(m, how, panel):
        s = C[C.measure == m].groupby(g).v.agg(how).reset_index()
        s["value"] = s.v * K._sg(True, m)
        s["pos"] = s.position.map(P3)
        s["panel"] = panel
        s["position"] = m + ":" + s.pos
        return s
    out["p_paths"] = (pd.concat([summ("side", "mean", "tongue side offset (mm; + = contra)"),
                                 summ("j_down", "max", "jaw-opening peak (mm)")]),
                      "3-D paths: mean tongue side offset and jaw-opening peak over phase 0.1-0.9 (kin 4.11)",
                      "change vs own pre")
    out["p_phase"] = (pd.concat([summ("mid", "mean", "midline angle (deg; + = contra)"),
                                 summ("dist", "mean", "3-D distance from mouth zero (mm)")]),
                      "3-D phase: mean midline angle and 3-D distance over phase 0.1-0.9 (kin 9.7)",
                      "change vs own pre")
    return out


def main(argv=None) -> int:
    from scripts import lmm_change_fig as lcf
    from wfield_local import orofacial_cohort as oc
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    rv = PathResolver()
    base = Path(rv.root("orofacial")) / oc.RUN / "cohort" / "tables" / "prepost"
    jobs = {**cam4(base), **pose3d(pose3d_dir(rv) / "phase" / "curves.csv.gz")}
    for stem, (D, title, ylab) in jobs.items():
        D = D[D.epoch.isin(["pre", "acute", "subacute", "chronic"])]
        two = "panel" in D
        if not two:
            D = D.assign(position=D.pos)
        f = out / f"{stem}_long.csv"
        D[["position", "epoch", "animal", "value"] + (["panel"] if two else [])].to_csv(f, index=False)
        cells = ([f"{m}:{p}" for m in dict.fromkeys(D.position.str.split(":").str[0]) for p in POS] if two else POS)
        args = [str(f), "--out", str(out / f"{stem}_lmm.png"), "--title", title, "--ylabel", ylab,
                "--cells", ",".join(c for c in cells if c in set(D.position))]
        if two:
            args += ["--panel-by", "panel", "--holm-by-panel", "--ncols", "1", "--figsize", "13,5.5"]   # stacked (a8 11-03)
        lcf.main(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
