"""Per-animal body weight over the experiment, aligned to each animal's stroke date.

Data = `configs/animal_weights.csv` (animal, date YYYYMMDD, pre_weight_g), the PRE-session body
weight imported from the lab weight sheet by `scripts/import_animal_weights.py`. Days-relative-to-
stroke and epoch are DERIVED here from `configs/animals.yaml` so the stroke date and the epoch
boundaries stay single-sourced (`config.stroke_date`, `epoch_figures.epoch_of_day`); a weigh-in on
the stroke date is day 0 (baseline / `pre`, per the animals.yaml convention).

Three views (all PER ANIMAL -- this cohort has no phenotype-severity split to pool over, unlike
stroke_orofacial):
  1. raw weight (g) over days-since-stroke         -> animal_weights.png
  2. weight as % of each animal's pre-stroke mean  -> animal_weights_pct_pre.png
  3. % of pre by epoch (pre/acute/subacute/chronic)-> animal_weights_by_epoch.png

The pre-stroke BASELINE for normalization is the mean pre-session weight over each animal's
pre-stroke RECORDING window (first pre-stroke session day .. stroke day), not the earlier
free-feeding acclimation weigh-ins -- so 100% is the task-ready pre-stroke weight the deficit is
measured against.

    from wfield_local import animal_weights as aw
    df = aw.load()            # animal, date, day_since_stroke, epoch, pre_weight_g, pct_pre
    aw.render(out_dir)        # all three figures + the derived CSV

CLI::

    python -m wfield_local.animal_weights [--out DIR] [--start first_session|all|<int>]
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local.epochs import days_since_stroke
from wfield_local.epoch_figures import epoch_of_day

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
EPOCHS = ("pre", "acute", "subacute", "chronic")
CSV = Path(__file__).resolve().parent.parent / "configs" / "animal_weights.csv"


def _stroke_date(animal: str) -> dt.date | None:
    v = (config.animals().get(animal) or {}).get("stroke_date")
    if v in (None, "", "null"):
        return None
    v = str(v)
    return dt.datetime.strptime(v, "%Y%m%d").date() if len(v) == 8 else None


def first_session_day(animal: str) -> int | None:
    """Day-since-stroke of the animal's earliest CURATED pre-stroke recording session."""
    days = [d for d in (days_since_stroke(l) for l in config.pooled_labels(animal)) if d is not None]
    return min(days) if days else None


def _epoch_for(animal: str, day, first_session: int | None) -> str | None:
    """pre/acute/subacute/chronic within the RECORDING window; None for pre-recording acclimation."""
    if pd.isna(day):
        return None
    day = int(day)
    if first_session is not None and day < first_session:
        return None                        # acclimation / free-feeding, before the experiment
    if day <= 0:
        return "pre"                       # stroke day is baseline; epoch_of_day leaves day 0 as None
    return epoch_of_day(animal, day)


def load(animal: str | None = None) -> pd.DataFrame:
    """Tidy weights with day_since_stroke, epoch, and % of each animal's pre-stroke baseline."""
    df = pd.read_csv(CSV, dtype={"animal": str, "date": str})
    df["pre_weight_g"] = pd.to_numeric(df["pre_weight_g"], errors="coerce")
    df = df.dropna(subset=["pre_weight_g"])
    recs = []
    for an, sub in df.groupby("animal"):
        sd = _stroke_date(an)
        fsd = first_session_day(an)
        sub = sub.copy()
        sub["day_since_stroke"] = [
            (dt.datetime.strptime(d, "%Y%m%d").date() - sd).days if sd else np.nan for d in sub["date"]]
        sub["epoch"] = [_epoch_for(an, d, fsd) for d in sub["day_since_stroke"]]
        # baseline = mean pre weight over the pre-stroke RECORDING window [first_session .. 0]
        base_mask = sub["day_since_stroke"].between(fsd if fsd is not None else -10**9, 0)
        baseline = sub.loc[base_mask, "pre_weight_g"].mean()
        sub["baseline_g"] = baseline
        sub["pct_pre"] = sub["pre_weight_g"] / baseline * 100.0
        recs.append(sub)
    out = pd.concat(recs, ignore_index=True).sort_values(["animal", "date"]).reset_index(drop=True)
    if animal:
        out = out[out["animal"] == animal].reset_index(drop=True)
    return out


def _colors():
    try:
        return config.animal_color()
    except Exception:                                                # noqa: BLE001
        return {}


def _x0(df, start):
    if start == "all":
        return int(df["day_since_stroke"].min())
    if start == "first_session":
        fs = [d for d in (first_session_day(a) for a in ANIMALS) if d is not None]
        return min(fs) if fs else int(df["day_since_stroke"].min())
    return int(start)


def _timecourse(df, col, ylabel, title, out_png, x0, hline=None):
    colors = _colors()
    fig, ax = plt.subplots(figsize=(10, 5))
    for an in ANIMALS:
        d = df[(df["animal"] == an) & (df["day_since_stroke"] >= x0)]
        if d.empty:
            continue
        c = colors.get(an)
        ax.plot(d["day_since_stroke"], d[col], "-o", ms=2.5, lw=1.2, label=an, color=c)
    ax.axvline(0, color="firebrick", ls="--", lw=1, zorder=0)
    ax.text(0, ax.get_ylim()[1], " stroke", color="firebrick", fontsize=8, va="top")
    if hline is not None:
        ax.axhline(hline, color="0.6", ls=":", lw=1, zorder=0)
    ax.set_xlabel("days since lesion (0 = stroke)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _by_epoch(df, out_png):
    """Mean % of pre-stroke baseline per epoch, one line per animal (x = epoch)."""
    colors = _colors()
    fig, ax = plt.subplots(figsize=(7, 5))
    xi = {e: i for i, e in enumerate(EPOCHS)}
    for an in ANIMALS:
        d = df[df["animal"] == an]
        xs, ys, es = [], [], []
        for e in EPOCHS:
            v = d.loc[d["epoch"] == e, "pct_pre"]
            if len(v):
                xs.append(xi[e]); ys.append(v.mean())
                es.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=1.4, capsize=3, label=an,
                        color=colors.get(an))
    ax.axhline(100, color="0.6", ls=":", lw=1, zorder=0)
    ax.set_xticks(range(len(EPOCHS))); ax.set_xticklabels(EPOCHS)
    ax.set_xlim(-0.3, len(EPOCHS) - 0.7)
    ax.set_ylabel("body weight (% of pre-stroke mean)", fontsize=10)
    ax.set_title("Body weight by epoch, per animal (mean ± SEM of weigh-ins)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render(out_dir, start="first_session", log=print):
    df = load()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x0 = _x0(df, start)
    _timecourse(df, "pre_weight_g", "pre-session body weight (g)",
                "Body weight over the experiment, per animal",
                out_dir / "animal_weights.png", x0)
    _timecourse(df, "pct_pre", "body weight (% of pre-stroke mean)",
                "Body weight (normalized to pre-stroke mean), per animal",
                out_dir / "animal_weights_pct_pre.png", x0, hline=100)
    _by_epoch(df, out_dir / "animal_weights_by_epoch.png")
    df.to_csv(out_dir / "animal_weights_rel_stroke.csv", index=False)
    # per-animal baseline, for the record
    for an in ANIMALS:
        b = df.loc[df["animal"] == an, "baseline_g"].iloc[0] if (df["animal"] == an).any() else float("nan")
        log(f"  {an}: pre-stroke baseline {b:.2f} g")
    log(f"[weights] wrote 3 figures + animal_weights_rel_stroke.csv to {out_dir}  (x from day {x0})")
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output dir (default: behavior_summary/weights on the share)")
    ap.add_argument("--start", default="first_session", help="first_session | all | <int day>")
    a = ap.parse_args(argv)
    if a.out:
        out_dir = Path(a.out)
    else:
        from wfield_local.paths import PathResolver
        out_dir = Path(PathResolver().root("behavior_out")) / "weights"
    render(out_dir, start=a.start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
