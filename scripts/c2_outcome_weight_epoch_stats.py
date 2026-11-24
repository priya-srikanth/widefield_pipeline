"""Six-position cohort (PS92-95): trial-outcome change (manuscript Fig 4B) and body weight by epoch (Fig 3D).

Ported unchanged from the manuscript's own scripts (VLS manuscript v41, `figures/trial_class_distribution.py` cohort-2
part and `figures/weights_by_epoch.py` cohort-2 part), so the numbers come from this repo; the two-tone halves of both
are in stroke_orofacial_pipeline (`scripts/manuscript/run_trial_outcome_stats.py`, `run_weight_epoch_lmm.py`).

Trial outcomes (`outcomes`):
  input  s1_trial_classes_cue_stop.csv.gz (LP v36 imaging_encoding/qc; camera classes contact / incomplete / jaw_only /
         no_response; no_video, unknown and position 'unknown' dropped);
  trials in each session's WORKING PERIOD (cue index <= the session's last contact or incomplete lick);
  pre = August sessions from 0806 to the stroke-day session, 0817 excluded for PS92/93 (lasered before the effective
  stroke); epochs from `config.epoch_spec()` (days since stroke; days between the acute and sub-acute epochs dropped);
  per session the fraction of each class; per position x post epoch: paired nested bootstrap (animals resampled with
  replacement, then sessions within each animal's pre and post sets; 4,000 draws; numpy default_rng(0), draw order as
  the manuscript); change = mean over animals of (post mean - pre mean); '*' 95% CI excludes 0, '**' the
  Bonferroni-level CI over the 18 position x epoch cells (2.5/18 % tails) excludes 0.
  The draw scheme is the manuscript's (not `analysis_kit.boot_delta`), kept so the stars are unchanged.

Body weight (`weight`):
  input  configs/animal_weights.csv (pre-session weight); sessions from 20260806; % of each animal's mean pre-stroke
  (day <= 0) weight; epochs: acute <= acute last day, chronic >= chronic_from, days between counted as sub-acute;
  statsmodels MixedLM, REML, random intercept per animal: pct ~ C(epoch, pre); each post epoch vs pre (3 contrasts),
  BH-FDR; * / ** / *** at q 0.05 / 0.01 / 0.001.

Writes (refuses to overwrite) under --out-dir (default M:/MICROSCOPE/Priya/Widefield/manuscript_stats):
  c2_trial_outcome_boot_stats.csv, c2_trial_outcome_fractions.csv, c2_weight_epoch_lmm.csv, c2_weight_epoch_days.csv

Usage: python scripts/c2_outcome_weight_epoch_stats.py [--out-dir DIR] [--classes CSV.GZ] [--weights CSV]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from wfield_local import config  # noqa: E402

SRV = Path("M:/MICROSCOPE/Priya")
# 11-22 (plan item 3 / runbook bug: steps 5 / 6 / 8 passed only --out-dir, so every input stayed v36 / two-camera):
# the DEFAULT follows the adopted run (`orofacial_cohort.RUN`, routed like the producers); pass it explicitly anyway
from wfield_local import orofacial_outputs as _oo  # noqa: E402
CLASSES_CSV = _oo.encoding_root() / "qc" / "s1_trial_classes_cue_stop.csv.gz"
WEIGHTS_CSV = REPO / "configs/animal_weights.csv"
OUT_DIR = SRV / "Widefield/manuscript_stats"
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
CLS = ["complete", "incomplete", "jaw_only", "no_resp"]
EPS = ["pre", "acute", "subacute", "chronic"]
FIRST_PRE = "0806"                                   # August pre-stroke sessions only (manuscript convention)
EXCL = {"PS92": {"0817"}, "PS93": {"0817"}}          # lasered before the effective stroke (animals.yaml note)
N_BOOT = 4000
N_CELLS = 18                                         # 6 positions x 3 post epochs (Bonferroni level)
YEAR = 2026


def _bounds():
    """{animal: (acute last day, sub-acute first day, chronic first day)} from configs/animals.yaml."""
    sp = config.epoch_spec()
    return {a: (sp[a]["acute"][1], sp[a]["subacute_from"], sp[a]["chronic_from"]) for a in ANIMALS}


def _stroke(a):
    s = str(config.stroke_date(a))[-4:]          # MMDD within the cohort year
    return dt.date(YEAR, int(s[:2]), int(s[2:]))


def load_outcomes(path, bounds) -> pd.DataFrame:
    """Working-period trials with class, animal, mmdd, epoch (manuscript filters)."""
    c2 = pd.read_csv(path)
    c2 = c2[c2.cls.isin(["contact", "incomplete", "jaw_only", "no_response"]) & (c2.position != "unknown")].copy()
    last = c2[c2.cls.isin(["contact", "incomplete"])].groupby("label").cue_idx.max()
    c2 = c2[c2.cue_idx <= c2.label.map(last)].copy()
    c2["animal"] = c2.label.str[:4]
    c2["mmdd"] = c2.label.str[5:9]
    c2 = c2[~c2.apply(lambda r: r.mmdd in EXCL.get(r.animal, set()), axis=1) & (c2.mmdd >= FIRST_PRE)]

    def ep(a, m):
        d = (dt.date(YEAR, int(m[:2]), int(m[2:])) - _stroke(a)).days
        al, sf, cf = bounds[a]
        if d <= 0:
            return "pre"
        if d <= al:
            return "acute"
        if d < sf:
            return None
        return "subacute" if d < cf else "chronic"

    c2["epoch"] = [ep(a, m) for a, m in zip(c2.animal, c2.mmdd)]
    c2 = c2[c2.epoch.notna()].copy()
    c2["cls"] = c2.cls.replace({"contact": "complete", "no_response": "no_resp"})
    return c2


def outcome_boot(c2: pd.DataFrame) -> pd.DataFrame:
    """Paired nested bootstrap of each class fraction, post epoch vs pre, per position (manuscript draw order)."""
    rng = np.random.default_rng(0)
    f = (c2.groupby(["position", "animal", "label", "epoch"]).cls.value_counts(normalize=True)
         .unstack().reindex(columns=CLS).fillna(0).reset_index())
    rows = []
    for p_, d in f.groupby("position"):
        for e in EPS[1:]:
            an = sorted(set(d[d.epoch == e].animal) & set(d[d.epoch == "pre"].animal))
            if not an:
                continue
            by = {a: (d[(d.animal == a) & (d.epoch == "pre")][CLS].values,
                      d[(d.animal == a) & (d.epoch == e)][CLS].values) for a in an}
            point = np.mean([by[a][1].mean(0) - by[a][0].mean(0) for a in an], axis=0)
            bs = np.empty((N_BOOT, len(CLS)))
            for b in range(N_BOOT):
                aa = rng.choice(an, len(an))
                v = []
                for a in aa:
                    p0, p1 = by[a]
                    v.append(p1[rng.integers(0, len(p1), len(p1))].mean(0) - p0[rng.integers(0, len(p0), len(p0))].mean(0))
                bs[b] = np.mean(v, axis=0)
            for i, c in enumerate(CLS):
                lo, hi = np.percentile(bs[:, i], [2.5, 97.5])
                lo2, hi2 = np.percentile(bs[:, i], [100 * 0.025 / N_CELLS, 100 - 100 * 0.025 / N_CELLS])
                star = "**" if (lo2 > 0 or hi2 < 0) else "*" if (lo > 0 or hi < 0) else ""
                rows.append(dict(cohort=2, position=p_, epoch=e, cls=c, delta=point[i], lo=lo, hi=hi,
                                 lo_bonf=lo2, hi_bonf=hi2, star=star, n_animals=len(an)))
    return pd.DataFrame(rows)


def outcome_fractions(c2: pd.DataFrame) -> pd.DataFrame:
    """Bar heights: mean over animals of each animal's per-epoch class fractions."""
    out = []
    for (p, e), g in c2.groupby(["position", "epoch"]):
        pa = g.groupby("animal").cls.value_counts(normalize=True).unstack().reindex(columns=CLS).fillna(0)
        out.append(dict(position=p, epoch=e, n_animals=len(pa), **pa.mean(axis=0).to_dict()))
    return pd.DataFrame(out)


def weights(path, bounds) -> pd.DataFrame:
    """Animal-days with % of pre-stroke weight and epoch (gap days counted as sub-acute)."""
    w = pd.read_csv(path, dtype={"date": str})
    w = w[w.date >= f"{YEAR}{FIRST_PRE}"].copy()
    w["day"] = [(dt.datetime.strptime(d, "%Y%m%d").date() - _stroke(a)).days for a, d in zip(w.animal, w.date)]
    w["pct"] = 100 * w.pre_weight_g / w.animal.map(w[w.day <= 0].groupby("animal").pre_weight_g.mean())
    w["epoch"] = ["pre" if d <= 0 else "acute" if d <= bounds[a][0] else "subacute" if d < bounds[a][2] else "chronic"
                  for a, d in zip(w.animal, w.day)]
    return w


def weight_lmm(w: pd.DataFrame) -> pd.DataFrame:
    """MixedLM pct ~ epoch (random intercept animal); each post epoch vs pre, BH over the three."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    x = w.copy()
    x["epoch"] = pd.Categorical(x.epoch, EPS)
    r = smf.mixedlm("pct ~ C(epoch, Treatment('pre'))", x, groups=x.animal).fit(reml=True)
    rows = []
    for e in EPS[1:]:
        k = f"C(epoch, Treatment('pre'))[T.{e}]"
        rows.append(("all", e, r.params[k], r.conf_int().loc[k, 0], r.conf_int().loc[k, 1], r.pvalues[k]))
    t = pd.DataFrame(rows, columns=["group", "epoch", "diff_pct", "ci_lo", "ci_hi", "p"])
    t["q"] = multipletests(t.p, method="fdr_bh")[1]
    t["stars"] = ["***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "" for q in t.q]
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--classes", type=Path, default=CLASSES_CSV)
    ap.add_argument("--weights", type=Path, default=WEIGHTS_CSV)
    a = ap.parse_args(argv)
    outs = {k: a.out_dir / f"{k}.csv" for k in ("c2_trial_outcome_boot_stats", "c2_trial_outcome_fractions",
                                               "c2_weight_epoch_lmm", "c2_weight_epoch_days")}
    clash = [str(p) for p in outs.values() if p.exists()]
    if clash:
        sys.exit(f"refusing to overwrite {', '.join(clash)}")
    b = _bounds()
    c2 = load_outcomes(a.classes, b)
    t2 = outcome_boot(c2)
    fr = outcome_fractions(c2)
    w = weights(a.weights, b)
    lm = weight_lmm(w)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    t2.to_csv(outs["c2_trial_outcome_boot_stats"], index=False)
    fr.to_csv(outs["c2_trial_outcome_fractions"], index=False)
    lm.to_csv(outs["c2_weight_epoch_lmm"], index=False)
    w.to_csv(outs["c2_weight_epoch_days"], index=False)
    print(t2[t2.star != ""].to_string(index=False))
    print(lm.round(4).to_string(index=False))
    for p in outs.values():
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
