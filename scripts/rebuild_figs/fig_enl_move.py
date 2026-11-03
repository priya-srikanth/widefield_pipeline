"""Imaging results page Fig 11.2 (`d_enl_move.png`): within-session PRE-CUE (ENL, lick-free) spout-position decoding
before / after removing movement, across recovery, per animal (x = days from stroke, symlog).

PROVENANCE: the plotting script is LOST (imaging page provenance prov_s5_13.md: "temp enl_traj.py (outside repo) from
  scripts/enl_movement_multisession.py csv"); only its output survives: labcams/enl_movement_multisession/
  enl_movement_trajectory.png (10-13 10:09; the page's d_enl_move.png is that file after the page copy step
  docs/results_pages/imaging/page_figs.py). RECREATED 2026-11-03 (cohort-1 rebuild, step 3) from that image, its page
  caption and the CSV, and checked pixel by pixel against it (see SESSIONS below for what had to be inferred).

INPUT: <labcams>/enl_movement_multisession/enl_movement_multisession.csv, one row per session x arm, written by
  scripts/enl_movement_multisession.py (refit within each session: logistic C 0.5, all LocaNMF components, 4 x 0.5 s
  bins, GroupKFold by position block; window [cue - 2 s, cue], lick-free, engaged trials; chance 1/6). Arms drawn:
    raw        no removal                                                  (black)
    null_3     top-3 movement-potent dimensions projected out               (blue, "top-3 movement dims removed")
    residual   running + 4-camera video motion energy regressed out in-fold (red, "running + video regressed out")
  THE FIGURE WAS DRAWN FROM THE 10-13 STATE OF THE CSV, which survives as enl_movement_multisession.csv.bak_20261015
  (same timestamp as the PNG). On 10-15 PS92_0812 was re-run (22 -> 210 trials, raw 0.0 -> .395) into the live CSV, so
  the live CSV adds one PS92 session to the figure. Default --csv = the .bak (reproduces the page); pass the live CSV
  (or the rebuild's) to redraw on newer data. Default paths = M:/MICROSCOPE/Priya/Widefield/labcams/... as
  enl_movement_multisession.py resolves them on this box (PathResolver root "labcams"; the rebuild routes the script's
  output to root "cohort_out" -- give --csv).
SESSIONS (inferred from the caption, confirmed against the image): curated sessions (`analysis_kit.curated_labels`;
  drops PS9x_0602 / 0603 / 0605 / 0805, which are in the CSV) with n_trials >= --min-trials (60; drops the 22-trial
  PS92_0812 of the 10-13 CSV). Points sorted by day, joined by lines per arm and animal.
PLOT PARAMETERS INFERRED from the surviving PNG (pixel-identical at these values, 1980 x 440 at dpi 110): figsize
  18 x 4, 1 x 4 panels sharey, symlog x with linthresh 5 (linscale 1), markers "o-" ms 3 lw 1, chance 1/6 dotted 0.6
  grey, stroke line at day 0 (0.6 grey, lw 0.8), legend fontsize 7 in the first panel, suptitle fontsize 10.
  With the live (10-15) CSV the only change is PS92_0812 (day -5, raw .395) appearing (0.08 % of pixels).
REBUILD: enl_movement_multisession.py reads per-session LocaNMF C, which ignores the hemo variant (config.py:161-164,
  decision D1) and is skip-if-exists per session: run it into a NEW output (or prune) and pass its CSV here.
OUT: <out>/enl_movement_trajectory.png (+ enl_movement_trajectory.csv = the plotted points)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

NAME = "enl_movement_trajectory"
CSV_DEF = Path("M:/MICROSCOPE/Priya/Widefield/labcams/enl_movement_multisession/"
               "enl_movement_multisession.csv.bak_20261015")
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
XSCALE = {"linthresh": 5.0}
MS = 3
CHANCE_KW = {"color": "0.6", "ls": ":"}
ARMS = (("raw", "k", "raw"), ("null_3", "tab:blue", "top-3 movement dims removed"),
        ("residual", "tab:red", "running + video regressed out"))


def load(csv: Path, min_trials: int) -> pd.DataFrame:
    from wfield_local.analysis_kit import curated_labels
    d = pd.read_csv(csv)
    d = d[d.animal.isin(ANIMALS) & d.arm.isin([a for a, _c, _l in ARMS]) & (d.n_trials >= min_trials)
          & d.label.isin(set(curated_labels()))]
    return d[["label", "animal", "epoch", "day", "arm", "n_trials", "acc", "null_mean", "p"]].sort_values(
        ["animal", "arm", "day"]).reset_index(drop=True)


def draw(d: pd.DataFrame, png: Path) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, an in zip(axs, ANIMALS):
        g = d[d.animal == an]
        for arm, c, lab in ARMS:
            s = g[g.arm == arm].sort_values("day")
            ax.plot(s.day, s.acc, "o-", color=c, ms=MS, lw=1, label=lab)
        ax.axhline(1 / 6, **CHANCE_KW)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xscale("symlog", **XSCALE)
        ax.set_title(an)
        ax.set_xlabel("days from stroke")
    axs[0].set_ylabel("ENL position decoding accuracy")
    axs[0].legend(fontsize=7)
    fig.suptitle("Pre-cue (ENL, lick-free) spout-position decoding before / after removing movement "
                 "(running + 4-camera video)", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--csv", type=Path, default=CSV_DEF, help="enl_movement_multisession CSV")
    ap.add_argument("--min-trials", type=int, default=60)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        d = _common.load_figdata(a.out, NAME)["d"]
    else:
        d = load(a.csv, a.min_trials)
        d.to_csv(a.out / f"{NAME}.csv", index=False)
        _common.save_figdata(a.out, NAME, {"d": d, "src": str(a.csv), "min_trials": a.min_trials})
        print(d[d.arm == "raw"].groupby(["animal", "epoch"]).size().unstack().to_string())
    draw(d, a.out / f"{NAME}.png")
    print("->", a.out / f"{NAME}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
