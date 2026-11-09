"""Refit recall vs matched pre (D6, Priya 11-08): is the post-stroke drop in WITHIN-SESSION refit recall a trial-mix effect?

INPUT: `scripts/refit_matched_pre.py` output in the run's epoch data folder (nothing re-run here):
  PRIMARY     refit_matched_pre_cue_working_sessions.csv   cue window, WORKING trials (contact + incomplete camera licks,
              cut at the last camera lick), DAQ licks (daqlk run)
  SUPPORTING  refit_matched_pre_cue_lick_sessions.csv      same, lick trials only: TABLE ONLY (Priya 11-08, option 1).
              From the LP run's epoch data folder (`--lick-dir`; LP licks = every camera-seen lick), while the working
              half stays in the original (DAQ-lick) folder (`--data-dir`). `--plot-only --lick-dir D --table-out DIR`
              rebuilds only the table (+ caption) from D and redraws the figure from its figdata.
              Acute far contra holds 1-12 lick trials per session; there both a post refit and its matched pre control
              score ~0, so the excess is ~0 by construction there -- "can't tell", not "no loss". Not plotted.
QUANTITY, per session x position: excess = refit recall of the session minus that of the animal's OWN pre-stroke
sessions, both subsampled to the same position mix and size (whole blocks), refit within their own session, mean over
pre partners and draws. No trial floor. Pre sessions are scored the same way against the animal's OTHER pre sessions
(the null; its mean is ~0 by construction, its spread is the noise floor).
STATS: change vs pre per position x post epoch = `analysis_kit.boot_delta` (paired animals -> sessions, 4000 draws) of
post-session excess against pre-session excess; `boot_p`; Holm over the 18 tests (6 positions x 3 epochs). Stars in
the top band. Per-animal table printed first (ground rule 8).
OUT: <out>/refit_matched_pre.png (+ svg), refit_matched_pre_stats.csv, refit_matched_pre_lick_table.csv, refit_matched_pre_caption.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402

NAME = "refit_matched_pre"
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
EPOCHS = ("pre", "acute", "subacute", "chronic")
ROWS = (("close", "Near"), ("far", "Far"))
SIDES = ("ipsi", "mid", "contra")
CAPTION = (
    "Within-session refit decoder: recall at each spout position in a session, minus the animal's own pre-stroke "
    "sessions, after matching both to the same position mix and number of trials (whole blocks; each refit within its "
    "own session; mean over every pre partner and 5 draws; no trial floor). Cue window, working trials (contact + "
    "incomplete camera licks, cut at the last camera lick), DAQ licks. Pre = each pre session scored the same way "
    "against the animal's other pre sessions (mean ~0 by construction; its spread is the noise floor). Open markers = "
    "sessions (shape = animal, color = epoch); filled = mean of animals +- SEM. Stars: change vs pre, paired animals -> "
    "sessions bootstrap (4000 draws), Holm over 18 tests; * < .05, ** < .01, *** < .001. Lick-trial version: table only "
    "(acute far contra has 1-12 lick trials per session).")
#: appended when the lick table comes from the LP run (`--lick-dir`)
CAPTION_LICK_LP = (" Lick-trial table: LP licks (every camera-seen lick; LP run), same cue window and matching; a "
                   "summary without statistics (mean excess, trials, sessions per animal x position x epoch).")


def lesion_key(df: pd.DataFrame) -> pd.Series:
    from wfield_local import config
    from wfield_local.orofacial_cohort import pos_key
    lat = {a: str((v or {}).get("stroke_laterality", "L")) for a, v in config.animals().items()}
    return pd.Series([pos_key(p, lat[a]) for a, p in zip(df.animal, df.position)], index=df.index)


def load(src: Path, variant: str) -> pd.DataFrame:
    D = pd.read_csv(src / f"refit_matched_pre_cue_{variant}_sessions.csv")
    D["pos"] = lesion_key(D)
    D["epoch"] = pd.Categorical(D.epoch, EPOCHS, ordered=True)
    return D.dropna(subset=["excess"]).sort_values(["animal", "pos", "epoch", "label"]).reset_index(drop=True)


def per_animal(D: pd.DataFrame) -> pd.DataFrame:
    t = D.groupby(["pos", "animal", "epoch"], observed=True).excess.mean().unstack("epoch").reindex(columns=EPOCHS)
    return t.reindex([f"{r}_{s}" for r, _ in ROWS for s in SIDES], level=0)


def stats(D: pd.DataFrame) -> pd.DataFrame:
    from wfield_local.analysis_kit import boot_delta, boot_p, holm
    rng = np.random.default_rng(20261108)
    rows = []
    for r, _ in ROWS:
        for s in SIDES:
            g = D[D.pos == f"{r}_{s}"]
            pre = {a: list(x.excess) for a, x in g[g.epoch == "pre"].groupby("animal")}
            for e in EPOCHS[1:]:
                post = {a: list(x.excess) for a, x in g[g.epoch == e].groupby("animal", observed=True)}
                iv, draws = boot_delta(post, pre, rng, draws=True)
                rows.append(dict(pos=f"{r}_{s}", epoch=e, delta=iv.point if iv else np.nan,
                                 lo=iv.lo if iv else np.nan, hi=iv.hi if iv else np.nan,
                                 n_animals=iv.n_animals if iv else 0, p=boot_p(draws)))
    T = pd.DataFrame(rows)
    T["p_holm"] = holm(T.p.to_numpy())
    return T


def lick_table(D: pd.DataFrame) -> pd.DataFrame:
    g = D.groupby(["animal", "pos", "epoch"], observed=True)
    return pd.concat([g.excess.mean(), g.n.sum().rename("trials_sum"), g.n.min().rename("trials_min_session"),
                      g.size().rename("sessions")], axis=1).reset_index()


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def draw(D: pd.DataFrame, T: pd.DataFrame, png: Path) -> None:
    from wfield_local.orofacial_cohort import EPOCH_LINE
    from wfield_local.sig_marks import reserve_headroom, star_top
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    rng = np.random.default_rng(0)
    off = {a: o for a, o in zip(ANIMALS, (-0.24, -0.12, 0.0, 0.12))}
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for i, (r, rname) in enumerate(ROWS):
        for j, s in enumerate(SIDES):
            ax, key = axs[i, j], f"{r}_{s}"
            g = D[D.pos == key]
            ax.axhline(0, color="0.45", ls="--", lw=1.1, zorder=1)
            for (a, e), x in g.groupby(["animal", "epoch"], observed=True):
                xs = EPOCHS.index(e) + off[a] + rng.uniform(-0.035, 0.035, len(x))
                ax.scatter(xs, x.excess, marker=MARK[a], s=22, facecolors="none", edgecolors=EPOCH_LINE[e],
                           linewidths=1.0, alpha=0.85, zorder=2)
            M = g.groupby(["animal", "epoch"], observed=True).excess.mean().unstack("epoch").reindex(columns=EPOCHS)
            mu, se = M.mean(), M.std(ddof=1) / np.sqrt(M.notna().sum())
            ax.errorbar(np.arange(len(EPOCHS)) + 0.27, mu, yerr=se, color="0.15", ls="none", elinewidth=2.0,
                        marker="o", ms=6, mfc="0.15", mec="0.15", capsize=3, capthick=1.4, zorder=3)
            for _, t in T[T.pos == key].iterrows():
                st = _stars(t.p_holm)
                if st:
                    star_top(ax, EPOCHS.index(t.epoch), st, color=EPOCH_LINE[t.epoch], fontsize=13)
            ax.set_title(f"{rname} {s}", fontsize=12, pad=6)
            ax.set_xticks(np.arange(len(EPOCHS)))
            ax.set_xticklabels(["Pre", "Acute", "Sub.", "Chronic"], fontsize=10)
            ax.set_xlim(-0.5, len(EPOCHS) - 0.4)
        axs[i, 0].set_ylabel("Refit recall vs own pre\n(mix + size matched)", fontsize=11)
    reserve_headroom(fig)
    h = [Line2D([], [], ls="none", marker=MARK[a], mfc="none", mec="0.3", ms=7, label=a) for a in ANIMALS]
    h += [Line2D([], [], ls="none", marker="o", ms=7, mfc="0.15", mec="0.15", label="Mean of animals ± SEM")]
    h += [Line2D([], [], ls="none", marker="o", mfc="none", mec=EPOCH_LINE[e], mew=1.6, ms=7,
                 label=e.capitalize()) for e in EPOCHS]
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.5, h_pad=1.5)
    save_png_svg(fig, png, dpi=200, facecolor="white")
    plt.close(fig)


def trial_set_sentence() -> str:
    """The figure's trials (the WORKING sessions CSV; the lick-trial table has no figure)."""
    return _common.working_trials(extra=["cue window; each session and its pre controls subsampled to common position "
                                         "counts (whole blocks)"])


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--data-dir", type=Path, default=None, help="epoch data folder (default: this run's)")
    ap.add_argument("--lick-dir", type=Path, default=None,
                    help="epoch data folder holding refit_matched_pre_cue_lick_sessions.csv (the LP run's; default "
                         "--data-dir)")
    ap.add_argument("--table-out", type=Path, default=None,
                    help="where the lick table + caption go (default --out)")
    a = ap.parse_args(argv)
    _common.declare(trial_set_sentence())          # THE TRIAL SET of every figure (Priya 11-09)
    a.out.mkdir(parents=True, exist_ok=True)
    tab = a.table_out or a.out
    tab.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        fd = _common.load_figdata(a.out, NAME)
        D, T = fd["D"], fd["T"]
        if a.lick_dir:
            lick_table(load(a.lick_dir, "lick")).to_csv(tab / f"{NAME}_lick_table.csv", index=False,
                                                        float_format="%.4g")
            (tab / f"{NAME}_caption.txt").write_text(CAPTION + CAPTION_LICK_LP + "\n", encoding="utf-8")
    else:
        src = a.data_dir or _common.epoch_data_dir()
        D = load(src, "working")
        pd.set_option("display.width", 200)
        print("per-animal mean excess (working trials), position x animal x epoch:")
        print(per_animal(D).round(3).to_string())
        T = stats(D)
        print("\nchange vs pre, boot_delta animals -> sessions, Holm over 18:")
        print(T.round(4).to_string(index=False))
        T.to_csv(a.out / f"{NAME}_stats.csv", index=False, float_format="%.4g")
        L = lick_table(load(a.lick_dir or src, "lick"))
        L.to_csv(tab / f"{NAME}_lick_table.csv", index=False, float_format="%.4g")
        (tab / f"{NAME}_caption.txt").write_text(CAPTION + (CAPTION_LICK_LP if a.lick_dir else "") + "\n",
                                                 encoding="utf-8")
        _common.save_figdata(a.out, NAME, {"D": D, "T": T, "src": str(src)})
    draw(D, T, a.out / f"{NAME}.png")
    print(f"-> {a.out / f'{NAME}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
