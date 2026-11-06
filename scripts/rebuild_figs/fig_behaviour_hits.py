"""Imaging results page Fig 0.2 (`behaviour_hits_by_spout.png`; also the PI-deck behaviour opener, slide 4):
hit rate per spout in the CAMERA WORKING PERIOD, by epoch, one open marker per session, cohort 1 (PS92-95); the DAQ
quit-tail (engaged-gated) hit rate is drawn as the labelled robustness figure `behaviour_hits_by_spout_daq_quittail.png`.

HIT-RATE PERIOD (Priya 11-05, DECISIONS_PENDING_MERGE_2026-11-04 (f) 2; supersedes RECONCILIATION D8 for the hit
rate): PRIMARY = the camera working period, first cue -> the session's last camera lick, minus the stopped runs; the
DAQ quit-tail rule is the labelled robustness row. The camera hit rate comes from the run's trial-attempts table
(`scripts/trial_attempts.py` under the camera trial gate, `run_mode.attempts_dir()`): column `frac_hit_<pos>` =
contact trials / working-period trials (`canonical`: every trial not stopped / stopped_licking / early_run / after_end
/ unknown), i.e. Priya's 11-04 camera-class answer 2: numerator = contact, denominator = contact + incomplete +
jaw_only + no_response. A table built without the camera gate (no `frac_stopped_*` columns) is REFUSED.

PROVENANCE: recreated 2026-11-03 (cohort-1 rebuild, step 3) from the share-only
  M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-02/fig_behaviour_hits.py (11-02 20:23).
  Computation, layout, jitter seeds and styling unchanged. Changes: argparse (--out / --plot-only / --metrics),
  output folder no longer the deck's build_inputs/, figdata pickle (<out>/data/behaviour_hits_by_spout_figdata.pkl).
  The original imported whatever `wfield_local` was on the path (it was run from widefield_pipeline); this one uses
  the repo it is run from (PYTHONPATH = the rebuild worktree).

INPUT (robustness row): behavior_summary/cohort/cohort_session_metrics.csv, column hit__<pos> = engaged-gated hit rate (a DAQ lick in
the session's real 3.5 s response window; `spout_behavior`, trials from the DAQ strobe). Default path is the
original's hard-coded M: path (kept; override with --metrics). Sessions = analysis_kit.curated_labels, pre restricted
to AUGUST (August-only baseline, Priya 10-18). Epoch = epoch_figures.epoch_of_day on days from config.stroke_date.
Lesion LEFT in all four (asserted), so contralesional = R spouts.
REBUILD: regenerate cohort_session_metrics.csv (spout_behavior --cohort) if the engagement gate changes (decision D8);
curated_labels / epoch_of_day come from the configs.

PLOT: one open marker per SESSION (shape = animal, colour = epoch), filled dot + bar beside it = mean of the 4
animals' session means +- SEM over animals; no connecting lines, no close / far divider; canonical spout order
(orofacial_cohort.POS_ORDER).
OUT: <out>/behaviour_hits_by_spout.png + .csv (camera working period, PRIMARY) and
     <out>/behaviour_hits_by_spout_daq_quittail.png + .csv (DAQ quit-tail rule, ROBUSTNESS); per animal x epoch x spout,
     with n sessions. The same session filter (curated, August-only pre) and epochs for both.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import datetime as dt
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

NAME = "behaviour_hits_by_spout"
#: the labelled robustness row (DAQ quit-tail / engagement-gated hit rate; Priya 11-05 (f) 2)
NAME_DAQ = NAME + "_daq_quittail"
#: the camera working-period hit rate in the run's trial_attempts_sessions.csv (`trial_attempts.session_summary`)
CAMERA_COL = "frac_hit_{pos}"

def _src() -> Path:
    """The behaviour pipeline's cohort table, through the path resolver (behavior_out; read in place)."""
    from wfield_local import config
    return Path(config.resolver().root("behavior_out")) / "cohort" / "cohort_session_metrics.csv"


SRC = _src()
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
POS = {"close_ipsi": "close_L", "close_mid": "close_center", "close_contra": "close_R",
       "far_ipsi": "far_L", "far_mid": "far_center", "far_contra": "far_R"}       # lesion LEFT (animals.yaml)
LABEL = {"close_ipsi": "close\nipsi", "close_mid": "close\nmid", "close_contra": "close\ncontra",
         "far_ipsi": "far\nipsi", "far_mid": "far\nmid", "far_contra": "far\ncontra"}
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}


def attempts_src() -> Path:
    """The run's per-session trial-attempts table (camera classes under the camera trial gate)."""
    from wfield_local import run_mode
    return run_mode.attempts_dir() / "trial_attempts_sessions.csv"


def camera_metrics(src) -> pd.DataFrame:
    """The camera working-period hit rate per session in the DAQ table's layout (animal, date, hit__<DAQ pos name>),
    from the run's trial_attempts_sessions.csv (path or frame). REFUSES a table without the camera run classes."""
    W = src if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    if not any(c.startswith("frac_stopped_") for c in W.columns):
        raise SystemExit(f"{src if not isinstance(src, pd.DataFrame) else 'attempts table'}: not a camera-gated "
                         "trial-attempts table (no frac_stopped_* columns) -- build it with scripts/trial_attempts.py "
                         "under WIDEFIELD_TRIAL_GATE=camera (the render profile)")
    out = pd.DataFrame({"animal": W.label.str[:4], "date": "2026" + W.label.str[5:]})
    for p, c in POS.items():
        col = CAMERA_COL.format(pos=p)
        out[f"hit__{c}"] = W[col].astype(float) if col in W else np.nan
    return out


def compute(src, metrics: pd.DataFrame | None = None):
    """Per animal x epoch x spout hit rate from the DAQ cohort table ``src``, or from ``metrics`` (`camera_metrics`)."""
    from wfield_local import analysis_kit as ak, config
    from wfield_local.epoch_figures import epoch_of_day
    for a in AN:
        assert config.animals()[a].get("stroke_laterality") == "L", a
    cur = {f"{lab[:4]}_{lab[5:]}" for lab in ak.curated_labels()}
    M = pd.read_csv(src, dtype={"date": str}) if metrics is None else metrics.copy()
    M["key"] = M.animal + "_" + M.date.str[4:]
    M = M[M.animal.isin(AN) & M.key.isin(cur)].copy()
    sd = {a: dt.datetime.strptime("2026" + config.stroke_date(a), "%Y%m%d") for a in AN}
    M["day"] = [(dt.datetime.strptime(d, "%Y%m%d") - sd[a]).days for a, d in zip(M.animal, M.date)]
    M["epoch"] = [epoch_of_day(a, int(d)) for a, d in zip(M.animal, M.day)]
    M = M[M.epoch.notna() & ~((M.epoch == "pre") & M.date.map(config.early_pre))]     # August-only pre
    rows = []
    for (a, e), g in M.groupby(["animal", "epoch"]):
        for p, c in POS.items():
            v = g[f"hit__{c}"].astype(float)
            rows.append(dict(animal=a, epoch=e, spout=p, hit=v.mean(), n_sessions=int(v.notna().sum())))
    T = pd.DataFrame(rows)
    keep = ["animal", "date", "key", "day", "epoch"] + [f"hit__{c}" for c in POS.values()]
    return {"T": T, "M": M[keep].copy(), "src": str(src)}


YLABEL = {NAME: "hit rate, camera working period", NAME_DAQ: "hit rate, engaged trials (DAQ quit-tail rule)"}
TITLE = {NAME: None, NAME_DAQ: "ROBUSTNESS: DAQ quit-tail period (primary = camera working period, Fig 0.2)"}


def draw(fd: dict, png: Path, name: str = NAME) -> None:
    from wfield_local.orofacial_cohort import EPOCH_LINE, POS_ORDER
    T, M = fd["T"], fd["M"]
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    x = np.arange(len(POS_ORDER))
    off = dict(zip(EP, np.linspace(-0.3, 0.3, len(EP))))
    rng = np.random.default_rng(0)
    for e in EP:
        P = T[T.epoch == e].pivot(index="animal", columns="spout", values="hit")[list(POS_ORDER)]
        m, se = P.mean(), P.std(ddof=1) / np.sqrt(P.notna().sum())
        G = M[M.epoch == e]
        for k, pos in enumerate(POS_ORDER):
            for a, g in G.groupby("animal"):
                v = g[f"hit__{POS[pos]}"].astype(float).dropna().to_numpy()
                jx = x[k] + off[e] - 0.035 + rng.uniform(-0.03, 0.03, len(v))
                ax.scatter(jx, v, marker=MARK[a], s=14, facecolor="none", edgecolor=EPOCH_LINE[e], lw=0.7, alpha=0.75,
                           zorder=2)
        ax.errorbar(x + off[e] + 0.045, m, yerr=se, color=EPOCH_LINE[e], mfc=EPOCH_LINE[e], mec="0.15", ls="none",
                    marker="o", ms=6, capsize=2.5, elinewidth=1.4, zorder=3, label=e)
    ax.set_xticks(x, [LABEL[p] for p in POS_ORDER], fontsize=12)
    ax.set_xlim(-0.55, len(POS_ORDER) - 0.45)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(YLABEL[name], fontsize=12)
    if TITLE[name]:
        ax.set_title(TITLE[name], fontsize=11, pad=28)
    ax.tick_params(axis="y", labelsize=11)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2 = [plt.Line2D([], [], ls="", marker=MARK[a], mfc="none", mec="0.4", label=a) for a in AN]
    ax.legend(h1 + h2, l1 + AN, ncol=8, fontsize=10,
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    save_png_svg(fig, png, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--metrics", type=Path, default=SRC, help="cohort_session_metrics.csv (the DAQ robustness row)")
    ap.add_argument("--attempts", type=Path, default=None,
                    help="trial_attempts_sessions.csv of the camera run (default: run_mode.attempts_dir())")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in (NAME, NAME_DAQ):                 # the primary first, then the labelled robustness row
        if a.plot_only:
            fd = _common.load_figdata(a.out, name)
        else:
            if name == NAME:
                src = a.attempts or attempts_src()
                fd = compute(src, metrics=camera_metrics(src))
            else:
                fd = compute(a.metrics)
            fd["period"] = "camera_working" if name == NAME else "daq_quit_tail"
            T = fd["T"]
            T.to_csv(a.out / f"{name}.csv", index=False)
            print(f"== {name} ({fd['period']})")
            print(T.pivot_table(index=["animal", "epoch"], columns="spout", values="hit")[list(POS)].round(2)
                  .reindex(EP, level=1).to_string())
            print(T.groupby(["animal", "epoch"]).n_sessions.max().unstack()[EP].to_string())
            _common.save_figdata(a.out, name, fd)
        draw(fd, a.out / f"{name}.png", name)
        print("wrote", a.out / f"{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
