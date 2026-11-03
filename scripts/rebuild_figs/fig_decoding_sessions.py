"""Imaging results page Fig 1.4 (`decoding_refit_sessions.png`; PI-deck slide 9): within-session REFIT spout decoder
accuracy per session, pre-cue and post-cue windows (the per-session view of the imaging page section-1 table).

PROVENANCE: recreated 2026-11-03 (cohort-1 rebuild, step 3) from the share-only
  M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-02/fig_decoding_sessions.py (11-02 20:12).
  Computation, table check, layout and jitter seeds unchanged. Changes: argparse (--out / --plot-only / --src), no
  sys.path insert of widefield_pipeline (uses the repo on PYTHONPATH), output folder no longer build_inputs/,
  figdata pickle (<out>/data/decoding_refit_sessions_figdata.pkl).

INPUT (read as is, nothing re-run): LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/qc/
  slowband_decode_all.csv (scripts/slowband_decode_check.py --all, 10-20): adopted_pre = pre-cue window, adopted_post
  = post-cue (cue) window, ADOPTED SVTcorr; decoder = filter_acausality_test.decode (multinomial logistic on Allen-ROI
  features, refit per session, GroupKFold by ~6-trial position blocks; chance 1/6). Default path = the original's
  hard-coded M: path (kept; --src overrides). The columns are --pre-col / --post-col so the rebuild can point the
  figure at a new hemo variant's columns (manifest M00003: "script loads hemo_{ADOPTED} directly; parameterise").
  NOTE (original): three curated acute sessions have EMPTY decode rows in the source (PS92_0818, PS94_0817,
  PS94_0820), so acute n = 4 / 4 / 4 / 1, not the curated 5 / 4 / 6 / 1.
  The TABLE check (the section-1 table values) is printed only; it is a check of THIS input, so expect mismatches
  when the rebuild regenerates the CSV.
REBUILD: regenerate slowband_decode_all.csv (slowband_decode_check.py --all) under the new variant.

Cohort 1 only (PS92-95). Curated sessions (`analysis_kit.curated_labels`). Pre = August-only (>= 0806).
Epoch = `epoch_figures.epoch_of_day` on days from `config.stroke_date` (cross-checked against the source epoch column).
OUT: <out>/decoding_refit_sessions.png / .csv
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

NAME = "decoding_refit_sessions"
SRC = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/qc/"
           "slowband_decode_all.csv")
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
EPOCHS = ("pre", "acute", "subacute", "chronic")
CHANCE = 1 / 6
PANELS = (("precue_acc", "Pre-cue window (no lick yet)"), ("postcue_acc", "Post-cue window"))
# the slide table (session means per animal x epoch), checked below
TABLE = {"precue_acc": {"PS92": (.361, .389, .340, .402), "PS93": (.405, .348, .281, .405),
                        "PS94": (.549, .430, .468, .530), "PS95": (.326, .474, .304, .362)},
         "postcue_acc": {"PS92": (.751, .669, .770, .822), "PS93": (.680, .551, .664, .821),
                         "PS94": (.860, .769, .819, .869), "PS95": (.841, .906, .802, .853)}}


def _day(animal: str, mmdd: str) -> int:
    from wfield_local import config
    s = config.stroke_date(animal)
    return (date(2026, int(mmdd[:2]), int(mmdd[2:])) - date(2026, int(s[:2]), int(s[2:]))).days


def load(src: Path, pre_col: str, post_col: str) -> pd.DataFrame:
    from wfield_local.analysis_kit import curated_labels
    from wfield_local.epoch_figures import epoch_of_day
    D = pd.read_csv(src)
    D["animal"] = D.label.str.split("_").str[0]
    D = D[D.animal.isin(ANIMALS) & D.label.isin(set(curated_labels()))].copy()
    D["date"] = D.label.str[-4:]
    D["day"] = [_day(a, m) for a, m in zip(D.animal, D.date)]
    D["epoch_fig"] = [epoch_of_day(a, d) for a, d in zip(D.animal, D.day)]
    bad = D[D.epoch_fig != D.epoch]
    if len(bad):
        print(f"  WARNING epoch mismatch vs source:\n{bad[['label', 'day', 'epoch_fig', 'epoch']]}")
    D["epoch"] = D.epoch_fig
    D = D[D.epoch.isin(EPOCHS)]
    D = D[(D.epoch != "pre") | (D.date >= "0806")]                       # August-only pre baseline
    empty = D[D[pre_col].isna() & D[post_col].isna()]
    if len(empty):
        print("  curated sessions with NO decode in the source (dropped):", ", ".join(empty.label))
    D = D.drop(index=empty.index)
    D = D.rename(columns={pre_col: "precue_acc", post_col: "postcue_acc"})
    D = D[["animal", "date", "label", "day", "epoch", "precue_acc", "postcue_acc"]]
    D["epoch"] = pd.Categorical(D.epoch, EPOCHS, ordered=True)
    return D.sort_values(["animal", "day"]).reset_index(drop=True)


def check_table(S: pd.DataFrame) -> None:
    worst = 0.0
    for col, rows in TABLE.items():
        M = S.groupby(["animal", "epoch"], observed=True)[col].mean().unstack("epoch").reindex(columns=EPOCHS)
        for a, vals in rows.items():
            for e, v in zip(EPOCHS, vals):
                d = abs(M.loc[a, e] - v)
                worst = max(worst, d)
                if d > 0.0015:
                    print(f"  TABLE MISMATCH {col} {a} {e}: data {M.loc[a, e]:.3f} vs table {v:.3f}")
    print(f"table check: max |data - table| = {worst:.4f}")


def draw(S: pd.DataFrame, png: Path) -> None:
    from wfield_local.orofacial_cohort import EPOCH_LINE
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    rng = np.random.default_rng(0)
    off = {a: o for a, o in zip(ANIMALS, (-0.24, -0.12, 0.0, 0.12))}
    mean_dx = 0.27                                                       # cohort mean sits right of the sessions
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (col, title) in zip(axs, PANELS):
        ax.axhline(CHANCE, color="0.45", ls="--", lw=1.1, zorder=1)
        for (a, e), g in S.groupby(["animal", "epoch"], observed=True):
            v = g[col].dropna()
            x = EPOCHS.index(e) + off[a] + rng.uniform(-0.035, 0.035, len(v))
            ax.scatter(x, v, marker=MARK[a], s=26, facecolors="none", edgecolors=EPOCH_LINE[e], linewidths=1.0,
                       alpha=0.85, zorder=2)
        M = S.groupby(["animal", "epoch"], observed=True)[col].mean().unstack("epoch").reindex(columns=EPOCHS)
        mu, se = M.mean(), M.std(ddof=1) / np.sqrt(M.notna().sum())
        xs = np.arange(len(EPOCHS)) + mean_dx
        ax.errorbar(xs, mu, yerr=se, color="0.15", ls="none", elinewidth=2.2, marker="o", ms=7, mfc="0.15", mec="0.15",
                    capsize=4, capthick=1.6, zorder=3)
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_xticks(np.arange(len(EPOCHS)))
        ax.set_xticklabels(["Pre", "Acute", "Subacute", "Chronic"], fontsize=11)
        ax.set_xlim(-0.5, len(EPOCHS) - 0.4)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.text(len(EPOCHS) - 0.42, CHANCE - 0.015, "chance", ha="right", va="top", fontsize=11, color="0.4")
    axs[0].set_ylabel("Decoding accuracy (6 spouts)", fontsize=12)
    h = [Line2D([], [], ls="none", marker=MARK[a], mfc="none", mec="0.3", ms=7, label=a) for a in ANIMALS]
    h += [Line2D([], [], ls="none", marker="o", ms=7, mfc="0.15", mec="0.15", label="Mean of animals ± SEM")]
    h += [Line2D([], [], ls="none", marker="o", mfc="none", mec=EPOCH_LINE[e], mew=1.6, ms=7,
                 label=e.capitalize()) for e in EPOCHS]
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    fig.savefig(png, dpi=200, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--src", type=Path, default=SRC, help="slowband_decode_all.csv")
    ap.add_argument("--pre-col", default="adopted_pre", help="pre-cue accuracy column of --src")
    ap.add_argument("--post-col", default="adopted_post", help="post-cue accuracy column of --src")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        S = _common.load_figdata(a.out, NAME)["S"]
    else:
        S = load(a.src, a.pre_col, a.post_col)
        check_table(S)
        S[["animal", "date", "epoch", "precue_acc", "postcue_acc", "label", "day"]].to_csv(
            a.out / f"{NAME}.csv", index=False, float_format="%.4f")
        _common.save_figdata(a.out, NAME, {"S": S, "src": str(a.src), "cols": (a.pre_col, a.post_col)})
        pd.set_option("display.width", 200)
        print(S.groupby(["animal", "epoch"], observed=True).size().unstack("epoch").to_string())
    draw(S, a.out / f"{NAME}.png")
    print(f"-> {a.out / f'{NAME}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
