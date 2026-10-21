"""Post-stroke lesion burden: ipsilesional / contralesional striatal (caudoputamen) area per section.

The widefield-cohort (PS92-95) version of the VLS stroke methods paper's Figure 1F ("Caudoputamen
ipsi/contra % area"). Same quantity and the same drawing as the paper's notebook
(`Documents/data/Immuno/Stroke area quantification plots.ipynb`): per section,
100 * ipsi / contra area; faded points = sections, black = animal mean +/- SEM across sections.
Two deliberate differences: points take the animal's colour from `configs/animals.yaml` (the paper
used matplotlib's cycle), and the jitter is seeded so a re-render does not move the points.

Data = `configs/histology_striatum_area.csv`, imported from the QuPath project by
`scripts/import_histology_areas.py` (ipsi = the smaller outline; see that docstring).

    from wfield_local import histology_area as ha
    df = ha.load()                 # one row per section, + ipsi_contra_pct
    ha.per_animal(df)              # n, mean, sem, min, max per animal
    ha.render(out_dir)             # PNG + SVG + per-section CSV + _meta.csv

CLI::

    python -m wfield_local.histology_area [--out DIR]     # default <microscope>/Widefield/histology

Also copy the PNG/SVG/CSV into the QuPath project folder (Priya, 2026-10-21); see docs/HISTOLOGY.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wfield_local import config

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "configs" / "histology_striatum_area.csv"
# Sections dropped from the plot and statistics, keyed by (animal, image, slide_section), each with a reason.
# Kept in the output CSV with included=False so the exclusion stays visible.
EXCLUSIONS = REPO / "configs" / "histology_section_exclusions.csv"
STEM = "histology_striatum_ipsi_contra_pct_area"
TITLE = "Caudoputamen ipsi/contra % area"
YLABEL = "Ipsi/contra % area"


def load(path=DATA, exclusions=EXCLUSIONS, include_excluded: bool = False) -> pd.DataFrame:
    """One row per section with ``ipsi_contra_pct``; excluded sections dropped unless ``include_excluded``."""
    df = pd.read_csv(path)
    df["ipsi_contra_pct"] = 100 * df.ipsi_area_um2 / df.contra_area_um2
    df["included"], df["exclude_reason"] = True, ""
    if exclusions is not None and Path(exclusions).exists():
        ex = pd.read_csv(exclusions)
        for _, e in ex.iterrows():
            m = (df.animal == e.animal) & (df.image == e.image) & (df.slide_section == e.slide_section)
            if m.sum() != 1:
                raise ValueError(f"exclusion {e.animal} {e.image} #{e.slide_section} matches {m.sum()} sections")
            df.loc[m, ["included", "exclude_reason"]] = [False, e.reason]
    return df if include_excluded else df[df.included].reset_index(drop=True)


def per_animal(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("animal").ipsi_contra_pct
    return pd.DataFrame({"n_sections": g.size(), "mean": g.mean(), "sem": g.sem(ddof=1),
                         "min": g.min(), "max": g.max()})


def plot(df: pd.DataFrame, out_stem: Path, seed: int = 0) -> Path:
    colors = config.animal_color()
    animals = sorted(df.animal.unique())
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, an in enumerate(animals):
        y = df.loc[df.animal == an, "ipsi_contra_pct"].to_numpy()
        ax.scatter(i + rng.normal(0, 0.04, len(y)), y, s=20, alpha=0.6, color=colors.get(an), zorder=3)
        sem = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0.0
        ax.errorbar(i, y.mean(), yerr=sem, fmt="o", color="black", capsize=4, linewidth=2,
                    markersize=6, zorder=5)
    ax.set_xticks(range(len(animals)), animals, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel(YLABEL, fontsize=12)
    ax.set_title(TITLE, fontsize=16, pad=12)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return png


def render(out_dir, data=DATA, log=print) -> Path:
    out_dir = Path(out_dir)
    full = load(data, include_excluded=True)
    for _, r in full[~full.included].iterrows():
        log(f"[histology] EXCLUDED {r.animal} section {r.section} ({r.image} #{r.slide_section}, "
            f"{r.ipsi_contra_pct:.1f}%): {r.exclude_reason}")
    df = full[full.included].reset_index(drop=True)
    summ = per_animal(df)
    log("[histology] per-animal ipsi/contra % area (mean +/- SEM across sections):")
    log(summ.round(1).to_string())
    png = plot(df, out_dir / STEM)
    full.to_csv(out_dir / f"{STEM}.csv", index=False, float_format="%.4f")
    try:
        from wfield_local import figure_meta
        figure_meta.write_meta(png, source=str(Path(data).resolve().relative_to(REPO)) if Path(data).resolve().is_relative_to(REPO) else str(data), quantity="100 * ipsi / contra caudoputamen area per section",
                               ipsi_rule="smaller outline of each section (flipped mounts)",
                               excluded=[f"{r.animal} {r.image} #{r.slide_section}" for _, r in full[~full.included].iterrows()],
                               n_sections={a: int(n) for a, n in summ.n_sections.items()},
                               mean_pct={a: round(float(v), 2) for a, v in summ["mean"].items()},
                               sem_pct={a: round(float(v), 2) for a, v in summ["sem"].items()},
                               ylim=(0, 100))
    except Exception as e:                       # noqa: BLE001 -- a sidecar must not cost the figure
        log(f"[histology] meta sidecar skipped: {e}")
    log(f"[histology] wrote {png.name} (+ .svg, .csv) to {out_dir}")
    return png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output dir (default: <microscope>/Widefield/histology)")
    ap.add_argument("--data", type=Path, default=DATA)
    a = ap.parse_args(argv)
    if a.out:
        out = Path(a.out)
    else:
        from wfield_local.paths import PathResolver
        out = Path(PathResolver().root("microscope")) / "Widefield" / "histology"
    render(out, data=a.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
