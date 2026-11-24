"""Fig 5F (FaceRhythm lick-factor peak by spout side) restricted to trial CLASSES (Priya 11-22: "which trials are
included ... report whether the result holds on contact trials only").

Same computation as `scripts/face_rhythm_peak.py --split side` (per session and frozen factor: trial-AVERAGED absolute
expression on the FR trial grid, cue -2..+4 s at 25 Hz; PEAK of that average over cue 0..+4 s; split videos combined
trial-weighted; log2 vs the animal's own pre-stroke sessions computed on the SAME trial subset; epoch change by
`face_rhythm.stats.epoch_change` with the same seed and group order; Holm over the 12 factors per group x epoch), but
each session's trials are first restricted to a class subset from the final trial-class table
(`imaging_encoding/qc/s1_trial_classes_cue_stop.csv.gz`; tongue_at_lips is already jaw_only there), matched to the FR
grid by session label and nearest cue time (<= 50 ms).

    python scripts/face_rhythm_peak_trialclass.py [--subsets all contact attempted]
-> <dlc>/face_rhythm/cam4/readouts/<set>/peak_trialclass_20261122/: peak_sessions_side_<subset>.csv,
   peak_epoch_stats_side_<subset>.csv, trial_mix_side.csv (class fractions of the grid trials per side x epoch),
   class_match.csv (match rate per session). ``all`` reproduces the published peak_epoch_stats_side.csv (plumbing check).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wfield_local.face_rhythm import common as frc  # noqa: E402
from wfield_local.face_rhythm import stats as frs  # noqa: E402

CLASS_TABLE = ("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding/"
               "qc/s1_trial_classes_cue_stop.csv.gz")
SUBSETS = {"all": None, "contact": ("contact",), "attempted": ("contact", "incomplete")}
MATCH_S = 0.05
#: ``--gate working`` (11-23, four-camera rule): grid trials must also be inside the camera working period (final class
#: not in `trial_classes.OUT_OF_WORKING_PERIOD`, trials without video dropped)
NOT_WP = ("stopped", "stopped_licking", "early_run", "after_end", "unknown", "no_video")


def classes_for(t, C: pd.DataFrame, cue: np.ndarray, col: str = "cls") -> np.ndarray:
    """Class of each FR grid trial (nearest cue of the session's class rows within MATCH_S; '' = unmatched)."""
    c = C[C.label == t.label].sort_values("cue_s")
    out = np.full(len(cue), "", dtype=object)
    if c.empty:
        return out
    cs = c.cue_s.to_numpy(float)
    i = np.clip(np.searchsorted(cs, cue), 1, len(cs) - 1)
    j = np.where(np.abs(cs[i - 1] - cue) <= np.abs(cs[i] - cue), i - 1, i)
    ok = np.abs(cs[j] - cue) <= MATCH_S
    out[ok] = c[col].to_numpy()[j[ok]]
    return out


def peaks_subset(fdir: Path, window, C: pd.DataFrame, keep, gate: str = "none") -> tuple[pd.DataFrame, list, list]:
    rows, match, mix = [], [], []
    for t in frc.sessions(fdir, "proj").itertuples():
        with np.load(t.file) as z:
            X = z["trials"].astype(float) * z["trials_D"][..., None]
            tms, pos = z["t_ms"] / 1000.0, z["position"].astype(str)
            cue = z["cue_daq_s"].astype(float)
        cls = classes_for(t, C, cue)
        match.append({"label": t.label, "key": t.key, "n_grid": len(cue), "n_matched": int((cls != "").sum())})
        sel = np.ones(len(cue), bool) if keep is None else np.isin(cls, keep)
        if gate == "working":
            fin = classes_for(t, C, cue, "cls_final")
            sel &= (fin != "") & ~np.isin(fin, NOT_WP)
            match[-1]["n_working"] = int(((fin != "") & ~np.isin(fin, NOT_WP)).sum())
        groups = frc.position_groups(pos, "side")
        for g, m in groups.items():
            for k in ("contact", "incomplete", "jaw_only", "no_response", "unknown", "no_video", ""):
                mix.append({"label": t.label, "animal": t.animal, "epoch": t.epoch, "group": g, "cls": k or "unmatched",
                            "n": int((m & (cls == k)).sum())})
        w = (tms >= window[0]) & (tms <= window[1])
        for g, m0 in groups.items():
            m = m0 & sel
            if m.sum() < 3:
                continue
            sub = np.nanmean(X[m], 0)[w]
            k = np.nanargmax(sub, 0)
            rows += [{"label": t.label, "key": t.key, "animal": t.animal, "epoch": t.epoch, "group": g, "factor": j,
                      "peak": float(sub[k[j], j]), "n_trials": int(m.sum())} for j in range(X.shape[-1])]
    S = pd.DataFrame(rows)
    S = (S.assign(wp=S.peak * S.n_trials).groupby(["label", "animal", "epoch", "group", "factor"], as_index=False)
         .agg(wp=("wp", "sum"), n_trials=("n_trials", "sum")))
    S["peak"] = S.wp / S.n_trials
    S = S.drop(columns="wp")
    pre = S[S.epoch == "pre"].groupby(["animal", "group", "factor"]).peak
    S = S.join(pre.mean().rename("pre_mean"), on=["animal", "group", "factor"])
    S["peak_vs_pre"] = S.peak / S.pre_mean
    S["log2_vs_pre"] = frs.log2_vs_pre(S, "peak", ["animal", "group", "factor"])
    return S, match, mix


def stats(S: pd.DataFrame, names) -> pd.DataFrame:
    rng = np.random.default_rng(0)                                   # as face_rhythm_peak.cohort
    animals = sorted(S.animal.unique())
    rows = []
    for (g, k), d in S.groupby(["group", "factor"]):
        for r in frs.epoch_change(d, "log2_vs_pre", rng):
            per = {an: r[an] for an in animals if an in r}
            rows.append({"group": g, "factor": k, "name": names[k], "epoch": r["epoch"], "mean_vs_pre": r["mean"],
                         "sem": r["sem"], "delta": r["delta"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "n_animals": r["n_animals"], "excludes_0": r["excludes_0"], "n_below_pre": r["n_below"],
                         "n_above_pre": len(per) - r["n_below"], "p_boot": r["p_boot"], **per})
    return frs.add_holm(pd.DataFrame(rows), ["group", "epoch"])


def main(argv=None) -> int:
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    ap.add_argument("--window", type=float, nargs=2, default=(0.0, 4.0))
    ap.add_argument("--class-table", default=CLASS_TABLE)
    ap.add_argument("--gate", default="none", choices=("none", "working"),
                    help="working: grid trials also inside the camera working period (four-camera rule, 11-23)")
    ap.add_argument("--out-name", default="peak_trialclass_20261122", help="output folder under the readouts set")
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = fro.factor_dir(rv, fro.BUNDLE, fro.SET)
    out = frc.readouts_dir(fro.SET, rv) / a.out_name
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    names = frc.factor_names(fdir)
    C = pd.read_csv(a.class_table, usecols=["label", "cue_s", "cls", "cls_final"])
    for s in a.subsets:
        f = out / f"peak_epoch_stats_side_{s}.csv"
        if a.out_name != "peak_trialclass_20261122" and f.exists():
            raise SystemExit(f"refusing to overwrite {f}")
        S, match, mix = peaks_subset(fdir, tuple(a.window), C, SUBSETS[s], a.gate)
        S["name"] = [names[k] for k in S.factor]
        S.to_csv(out / f"peak_sessions_side_{s}.csv", index=False, float_format="%.6g")
        stats(S, names).to_csv(out / f"peak_epoch_stats_side_{s}.csv", index=False, float_format="%.6g")
        if s == a.subsets[0]:
            pd.DataFrame(match).to_csv(out / "class_match.csv", index=False)
            pd.DataFrame(mix).to_csv(out / "trial_mix_side.csv", index=False)
        print("written", s, flush=True)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
