"""Cohort-2 (six-position) first-lick curves on COHORT 1'S LICK-PHASE AXIS (Priya 2026-11-19, PROMPT_c2_phase_aligned.md):
VLS manuscript Fig 4H/I and S9 trajectory panels. Ported 2026-11-19 from the manuscript's
`manuscript_code/analysis/c2_phase_from_peak.py` (v41, run there with ``--sig aligned``) so the numbers are made in the
pipeline (PROMPT_pipeline_sync_v41 item 8); the arithmetic is unchanged. New here: the per-cluster table
``<out>/c2_phase_first_<measure>_phase_sessions_clusters.csv`` (one row per observed cluster, with p and its BH q),
the same information the 'sig' rows carry bin by bin.

Phase definition = stroke_orofacial_pipeline build_first_lick_centered_phase_rows(ili_source="synthetic_only")
(dlc_kinematics/tongue_pertrial.py, Addendum #15): for each lick, window = peak_t +- ILI/2 and
phase = (t - (peak_t - ILI/2)) / ILI in [0, 1], peak at 0.5, where ILI = the ANIMAL'S pre-stroke median inter-lick
interval (one fixed value per animal). In-mouth frames count as 0 (the side0 / down0 / mid0 / dist curves of
`scripts/pose3d_phase_mouth0.py`).

ILI, cohort 1's definition (tongue_pertrial.compute_pre_stroke_median_ili_ms): intervals between adjacent tongue peaks
in the same bout (bout = consecutive peaks <= bout.max_ili_ms = 300 ms), pooled over all of the animal's pre-stroke
sessions, median over intervals. Cohort-2 counterpart: adjacent 3-D extension peaks (t_peak_3d) of consecutive licks
(lick_idx) in the same trial, intervals in (0, 300] ms, pooled over the animal's August-pre sessions (`config.early_pre`
excludes June), median over intervals. (v41: PS92 156, PS93 164, PS94 160, PS95 144 ms.)

Per session: the peak-aligned curve (x = ms from the 3-D extension peak, 4 ms grid, -152..+152) restricted to
|x| <= ILI/2, mapped to phase and linearly interpolated onto cohort 1's phase bins (bin centres read from cohort 1's
phenotype_pooled_phase_bin_mean_sem.csv); a bin whose neighbouring samples are missing stays missing.
Change curves: session minus its animal's own August-pre mean curve (on the phase bins). Summary per position x epoch x
bin: mean and SEM ACROSS SESSIONS (SD ddof=1 / sqrt(n_sessions)), n_sessions, n_animals ('mean' = change, 'raw').
'sig' rows: `orofacial_stats` cluster test on the phase-binned change curves, labels shuffled within animal
(--sig mask = uncorrected cluster p < 0.05, `cluster_permutation`; --sig aligned = per-cluster permutation p,
`cluster_permutation_p`, with BH-FDR within each measure across positions x epochs, q < 0.05 -- the manuscript's rule).
Filters as `scripts/pose3d_phase_curves.py --across sessions`: coverage >= 0.5, early-pre sessions dropped,
sign + = contralesional for mid/side measures.

Outputs (refuses to overwrite; default out = `<pose3d>/phase/vls_manuscript/`):
  c2_phase_first_<measure>_phase_sessions.csv  (columns as *_sessions.csv; 'sig' rows also carry p and q with --sig aligned)
  c2_phase_first_<measure>_phase_sessions_clusters.csv  (--sig aligned only)
  c2_pre_ili_ms.csv, _c2_phase_check_roundtrip.csv (check 2 below)

Curves (Priya 2026-11-20: mouth0 is the default, both kept): ``--curves-source mouth0`` (default,
`<pose3d>/phase/vls_manuscript/curves_mouth0_first_peak.csv.gz`, in-mouth = 0) or ``--curves-source pipeline``
(`<pose3d>/phase/curves.csv.gz`, in-mouth excluded for side / down / out; pass its measure names, e.g. side down mid
dist). ``--curves`` overrides either. ``--out-suffix S`` inserts S before `.csv` in every output name.

    python scripts/pose3d_phase_from_peak.py [--out-dir DIR] [--sig mask|aligned] [--measures side0 down0 mid0 dist]
        [--curves-source mouth0|pipeline | --curves CSV] [--out-suffix S] [--c1-bins CSV] [--licks CSV]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local import orofacial_stats as OS
from wfield_local.orofacial_cohort import CLUSTER_PERM
from wfield_local.pose3d import pose3d_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose3d_phase_curves import CURVES_SOURCES, curves_path  # noqa: E402

#: cohort 1's phase bins (stroke_orofacial_pipeline phase-aligned trajectories, 3-tier v3)
C1_BINS = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/stats/results/phenotype_pooled/"
               "phase_aligned_trajectories/3_tier/v3/phenotype_pooled_phase_bin_mean_sem.csv")
BOUT_MAX_ILI_MS = 300.0                                  # cohort 1 configs/defaults.yaml bout.max_ili_ms
EPOCHS = ("acute", "subacute", "chronic")
SIGN_FLIP = ("mid", "mid0", "side", "side0")             # stored + = ipsilesional -> + = contralesional


def august_pre(df):
    return (df.epoch == "pre") & ~df.date.astype(str).map(config.early_pre)


def pre_ili_ms(licks_csv):
    """Per animal: median interval between 3-D extension peaks of consecutive licks in a trial, (0, 300] ms,
    August-pre sessions pooled (cohort 1's compute_pre_stroke_median_ili_ms rule)."""
    L = pd.read_csv(licks_csv, usecols=["animal", "session", "date", "epoch", "trial_id", "lick_idx", "t_peak_3d"],
                    float_precision="round_trip")
    L = L[august_pre(L)].dropna(subset=["t_peak_3d"]).sort_values(["session", "trial_id", "lick_idx"])
    g = L.groupby(["session", "trial_id"])
    d = pd.DataFrame({"animal": L.animal, "ili": g.t_peak_3d.diff(), "step": g.lick_idx.diff()})
    d = d[(d.step == 1) & (d.ili > 0) & (d.ili <= BOUT_MAX_ILI_MS)]
    return (d.groupby("animal").ili.agg(ili_ms="median", n_intervals="size")
            .join(L.groupby("animal").session.nunique().rename("n_sessions")).reset_index())


def phase_bins(c1_csv=C1_BINS):
    # round_trip: the exact double of each decimal. pandas' default ("high") parser is off by an ulp on some values
    # (0.006944444444444444 -> 0.0069444444444444 under pandas 2.3.3 here), which moved every phase bin by ~1e-17
    b = pd.read_csv(c1_csv, usecols=["phase_bin", "phase_bin_center"], float_precision="round_trip")
    b = b.drop_duplicates("phase_bin").sort_values("phase_bin")
    return b.phase_bin_center.to_numpy(float)


def interp_nan(xq, x, y):
    """Linear interpolation that returns NaN where either bracketing sample is NaN (no bridging over gaps)."""
    j = np.clip(np.searchsorted(x, xq), 1, len(x) - 1)
    i = j - 1
    w = (xq - x[i]) / (x[j] - x[i])
    out = y[i] * (1 - w) + y[j] * w
    out[(xq < x[0]) | (xq > x[-1])] = np.nan
    return out


def to_phase(pv, ili_of, centres):
    """sessions x ms -> sessions x phase bins, each row with its animal's ILI."""
    xs = pv.columns.to_numpy(float)
    out = np.full((len(pv), len(centres)), np.nan)
    for r, (an, vals) in enumerate(zip(pv.index.get_level_values(0), pv.to_numpy(float))):
        ili = ili_of[an]
        assert abs(((0 - (-ili / 2)) / ili) - 0.5) < 1e-12            # check 1: the peak (x = 0) maps to phase 0.5
        keep = np.abs(xs) <= ili / 2 + 1e-9
        out[r] = interp_nan(centres * ili - ili / 2, xs[keep], vals[keep])   # bin centre -> ms, then sample
    return pd.DataFrame(out, index=pv.index, columns=centres)


def sig_rows(ch, keep, sel, an, mode):
    """Cluster test on the phase-binned change curves. mask: boolean per bin (uncorrected). aligned: list of clusters
    (start bin, stop bin exclusive, p, the cluster dict) with permutation p (BH applied later across the measure)."""
    sub = ch.to_numpy(float)[keep]
    ok = np.isfinite(sub).all(0)
    if mode == "mask":
        mask = np.zeros(ch.shape[1], bool)
        if ok.sum() >= 3:
            mask[ok] = OS.cluster_permutation(sub[:, ok], sel[keep], an[keep], n_perm=CLUSTER_PERM)
        return mask
    if ok.sum() < 3:
        return []
    idx = np.flatnonzero(ok)
    cl = OS.cluster_permutation_p(sub[:, ok], sel[keep], an[keep], n_perm=CLUSTER_PERM)
    return [(idx[c["start"]], idx[c["stop"] - 1] + 1, c["p"], c) for c in cl]     # back to full-bin indices


def build(curves, ili, centres, meas, mode):
    C = pd.read_csv(curves, float_precision="round_trip")
    C = C[(C.kind == "peak") & (C.lickset == "first") & C.measure.isin(meas) & (C["cov"] >= 0.5)]
    C = C[(C.epoch != "pre") | august_pre(C)]
    ili_of = ili.set_index("animal").ili_ms.to_dict()
    res, cls, chk = {}, {}, []
    for m in meas:
        rows, clusters = [], []
        sg = -1.0 if m in SIGN_FLIP else 1.0
        for p, g in C[C.measure == m].groupby("position"):
            pvm = g.pivot_table(index=["animal", "session", "epoch"], columns="x", values="v") * sg
            pv = to_phase(pvm, ili_of, centres)
            an = pv.index.get_level_values(0).to_numpy()
            ep = pv.index.get_level_values(2).to_numpy()
            # check 2 (per animal): pre raw mean on phase -> back to ms == the ms-domain pre mean inside the window
            for a in np.unique(an):
                s = (an == a) & (ep == "pre")
                if not s.any():
                    continue
                ph = pv[s].mean(0).to_numpy()
                ms = pvm[s].mean(0)
                w = ms.index[np.abs(ms.index) <= ili_of[a] / 2 - 4]
                back = interp_nan(np.asarray(w, float), centres * ili_of[a] - ili_of[a] / 2, ph)
                chk.append({"measure": m, "position": p, "animal": a,
                            "max_abs_diff": float(np.nanmax(np.abs(back - ms[w].to_numpy()))),
                            "range": float(np.nanmax(ms[w]) - np.nanmin(ms[w]))})
            base = pv[ep == "pre"].groupby(level=0).mean()
            ch = pd.DataFrame(pv.to_numpy(float) - base.reindex(an).to_numpy(float), index=pv.index, columns=pv.columns)
            for e in ("pre",) + EPOCHS:
                sel = ep == e
                if not sel.any():
                    continue
                for row, src in (("mean", ch), ("raw", pv)):
                    s = src[sel]
                    ns = s.notna().sum(0).to_numpy()
                    na = s.notna().groupby(level=0).any().sum(0).to_numpy()
                    rows.append(pd.DataFrame({"row": row, "position": p, "measure": m, "epoch": e, "x": centres,
                                              "v": s.mean(0).to_numpy(),
                                              "sem": s.std(0, ddof=1).to_numpy() / np.sqrt(np.maximum(ns, 1)),
                                              "n_sessions": ns, "n_animals": na}))
                if e == "pre" or sel.sum() < 2:
                    continue
                keep = (ep == "pre") | sel
                r = sig_rows(ch, keep, sel, an, mode)
                if mode == "mask":
                    rows.append(pd.DataFrame({"row": "sig", "position": p, "measure": m, "epoch": e, "x": centres,
                                              "v": r}))
                else:
                    clusters += [{"position": p, "epoch": e, "start": s0, "end": e0, "p": pv_,
                                  "phase_start": centres[s0], "phase_end": centres[e0 - 1], "n_bins": int(e0 - s0),
                                  "sign": c["sign"], "mass": c["mass"], "p_raw": c["p_raw"],
                                  "t_threshold": c["t_threshold"]} for s0, e0, pv_, c in r]
                    rows.append(pd.DataFrame({"row": "sig", "position": p, "measure": m, "epoch": e, "x": centres,
                                              "v": False}))
        out = pd.concat(rows, ignore_index=True)
        cdf = pd.DataFrame(clusters)
        if mode == "aligned":
            # BH-FDR within the measure, across its positions x epochs (one family per measure)
            out["p"] = np.nan
            out["q"] = np.nan
            out["v"] = out["v"].astype(object)            # 'sig' v is True/False as in *_sessions.csv
            if clusters:
                cdf["q"] = OS.bh_fdr(cdf.p.to_numpy())
                cdf["sig_aligned"] = cdf.q < OS.ALPHA
                for _, c in cdf.iterrows():
                    hit = (out.row == "sig") & (out.position == c.position) & (out.epoch == c.epoch)
                    bins = hit & out.x.isin(centres[int(c.start):int(c.end)])
                    out.loc[bins, ["p", "q"]] = [c.p, c.q]
                    if c.q < OS.ALPHA:
                        out.loc[bins, "v"] = True
        res[m] = out
        cls[m] = cdf.assign(measure=m) if len(cdf) else cdf
    return res, cls, pd.DataFrame(chk)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=None, help="default <pose3d>/phase/vls_manuscript")
    ap.add_argument("--sig", choices=["mask", "aligned"], default="aligned")
    ap.add_argument("--measures", nargs="+", default=["side0", "down0", "mid0", "dist"])
    ap.add_argument("--curves-source", choices=sorted(CURVES_SOURCES), default="mouth0",
                    help="mouth0 (default; in-mouth = 0) or pipeline (<pose3d>/phase/curves.csv.gz, in-mouth excluded)")
    ap.add_argument("--curves", type=Path, default=None, help="explicit curves file (overrides --curves-source)")
    ap.add_argument("--out-suffix", default="", help="inserted before .csv in every output name")
    ap.add_argument("--c1-bins", type=Path, default=C1_BINS)
    ap.add_argument("--licks", type=Path, default=None, help="default <pose3d>/kinematics/licks.csv.gz")
    a = ap.parse_args(argv)
    d3 = pose3d_dir()
    out_dir = a.out_dir or d3 / "phase" / "vls_manuscript"
    curves = a.curves or curves_path(d3, a.curves_source)
    sfx = a.out_suffix
    outs = {m: out_dir / f"c2_phase_first_{m}_phase_sessions{sfx}.csv" for m in a.measures}
    cl_outs = {m: out_dir / f"c2_phase_first_{m}_phase_sessions_clusters{sfx}.csv" for m in a.measures} \
        if a.sig == "aligned" else {}
    ili_out = out_dir / f"c2_pre_ili_ms{sfx}.csv"
    chk_out = out_dir / f"_c2_phase_check_roundtrip{sfx}.csv"
    clash = [p for p in [*outs.values(), *cl_outs.values(), ili_out, chk_out] if p.exists()]
    if clash:
        sys.exit("refusing to overwrite: " + ", ".join(map(str, clash)))
    ili = pre_ili_ms(a.licks or d3 / "kinematics" / "licks.csv.gz")
    print(ili.to_string(index=False), flush=True)
    centres = phase_bins(a.c1_bins)
    print(f"{len(centres)} phase bins from cohort 1", flush=True)
    res, cls, chk = build(curves, ili, centres, a.measures, a.sig)
    out_dir.mkdir(parents=True, exist_ok=True)
    ili.to_csv(ili_out, index=False)
    for m, df in res.items():
        df.to_csv(outs[m], index=False)
        print("wrote", outs[m], flush=True)
        if m in cl_outs:
            cls[m].to_csv(cl_outs[m], index=False)
    chk.to_csv(chk_out, index=False)
    print("check 2 (per animal, phase -> ms round trip): max |diff| / range =",
          float((chk.max_abs_diff / chk.range.replace(0, np.nan)).max()), flush=True)


if __name__ == "__main__":
    main()
