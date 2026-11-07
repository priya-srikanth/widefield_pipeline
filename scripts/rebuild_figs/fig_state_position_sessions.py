"""Imaging results page Fig 3.4 (`state_position_sessions.png`; PI-deck slide 13): FROZEN position (6-way) vs
behavioral-state (3-way) decoders, per session, as the fraction of above-chance performance kept.

PROVENANCE: recreated 2026-11-03 (cohort-1 rebuild, step 3) from the share-only
  M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-02/fig_state_position_sessions.py
  (11-02 20:20; its 21:39 re-render after the per-session z-scored state decoder, commit abda8c9c, is the page image).
  Computation (ordinal pairing, cross-checks, boot_delta + Holm, seeds), layout and jitter unchanged. Changes:
  argparse (--out / --plot-only / --data-dir), no sys.path insert of widefield_pipeline (uses the repo on PYTHONPATH),
  output folder no longer build_inputs/, figdata pickle (<out>/data/state_position_sessions_figdata.pkl).

Normalized: retained = (balanced accuracy - chance) / (1 - chance); chance 1/6 position, 1/3 state.

INPUTS (figdata sidecars of the epoch render, read as is, nothing re-run), in --data-dir (default = the original's
hard-coded M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/data):
  STATE     epoch_13_state_decoder_cue_sessions.csv      per-session balanced accuracy of the frozen 3-way
            (rest / running / licking, 1 s segments) decoder, `epoch_state._fig_13_state` <- locomotor_decoder.
            PROVENANCE CORRECTION (review M9, 11-03): the share sidecar was REWRITTEN 11-03 11:05 by main's bal_em state
            decoder (9547a773, "bal_em is the PRODUCTION frozen state decoder"; merged into the rebuild e74a4579). On
            the current sidecar this script reproduces build_inputs/state_position_sessions_balem.{csv,png} exactly,
            NOT the 11-02 21:39 session_z image (CSV max |diff| .307, 1.2 % pixels). The page / slide 13 image is the
            11-02 21:39 session_z version until the bal_em page edits (docs/status/STATE_BALEM_PAGE_EDITS_2026-11-03.md)
            land; the rebuild render uses the merged state spec (scaling session_bal_em + scaling_params + train_data).
  POSITION  per-session, per-true-position recall of the FROZEN pre-stroke position decoder (cue window, working
            trials), derived as refit - gap from
              epoch_5r_refit_by_position_cue_working_sessions.csv   (refit recall)
              epoch_5rgap_frozen_vs_refit_cue_working_sessions.csv  (refit - frozen recall, same trials)
            both from `_collect_5c(mode="paired")` (pre = one record PER pre session, LOSO frozen model).
            Cross-check only: epoch_acc_by_position_cue_working_sessions.csv (mode="frozen", the 13n source).
            Rows are paired by within-(epoch, animal, position) ordinal; position and state sessions are NOT matched
            one-to-one (different collectors).
  13n check: epoch_13n_state_vs_position_cue.csv (printed only).
REBUILD: re-render epoch families 13, 5r, 5rgap, acc (and 13n for the check) and point --data-dir at their data/.

STATS: change vs pre per readout x post epoch, `analysis_kit.boot_delta` (paired animals -> sessions, 4000 draws),
`boot_p`, Holm over the 6 tests; stars in the reserved top band (`sig_marks.star_top` + `reserve_headroom`).
OUT: <out>/state_position_sessions.png / .csv
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

NAME = "state_position_sessions"
SRC = _common.epoch_data_dir()   # R3: this run's epoch data (rebuild root under the profile)
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
EPOCHS = ("pre", "acute", "subacute", "chronic")
POSITIONS = ("nI", "nM", "nC", "fI", "fM", "fC")
PANELS = (("position", "Spout position (6-way)", 1 / 6), ("state", "Behavioral state (3-way)", 1 / 3))
# slide table (pooled) and imaging-page 13n sidecar
TABLE = {"position": (.86, .43, .70, .76), "state": (.98, .87, .82, .84)}


def _by_ordinal(src: Path, f: str) -> pd.DataFrame:
    d = pd.read_csv(src / f)
    d["k"] = d.groupby(["epoch", "animal", "position"]).cumcount()
    w = d.set_index(["epoch", "animal", "k", "position"])["value"].unstack("position")
    assert not w.isna().any().any(), f"{f}: missing per-position values"
    return w[list(POSITIONS)]


def load(src: Path) -> pd.DataFrame:
    refit = _by_ordinal(src, "epoch_5r_refit_by_position_cue_working_sessions.csv")
    gap = _by_ordinal(src, "epoch_5rgap_frozen_vs_refit_cue_working_sessions.csv")
    assert refit.index.equals(gap.index)
    frozen = refit - gap
    # cross-check against the frozen-mode sidecar (the 13n source)
    fz = _by_ordinal(src, "epoch_acc_by_position_cue_working_sessions.csv")
    post = fz.drop(index="pre", level="epoch")
    print(f"check post-stroke frozen recall vs frozen-mode sidecar: max |diff| = "
          f"{np.abs((post - frozen.loc[post.index]).values).max():.1e}")
    pre_d = (fz.loc["pre"].droplevel("k") - frozen.loc["pre"].groupby("animal").mean()).abs().values.max()
    print(f"check pre per-animal recall (frozen-mode pooled LOSO) vs session mean: max |diff| = {pre_d:.3f}")
    P = frozen.mean(axis=1).rename("balanced_acc").reset_index()
    P["readout"] = "position"
    S = pd.read_csv(src / "epoch_13_state_decoder_cue_sessions.csv").rename(columns={"value": "balanced_acc"})
    S["k"] = S.groupby(["epoch", "animal"]).cumcount()
    S["readout"] = "state"
    D = pd.concat([P, S[["epoch", "animal", "k", "balanced_acc", "readout"]]], ignore_index=True)
    chance = D.readout.map({"position": 1 / 6, "state": 1 / 3})
    D["chance"] = chance
    D["retained"] = (D.balanced_acc - chance) / (1 - chance)
    D = D.rename(columns={"k": "session_ordinal"})
    D["epoch"] = pd.Categorical(D.epoch, EPOCHS, ordered=True)
    return D.sort_values(["readout", "animal", "epoch", "session_ordinal"]).reset_index(drop=True)


def check(D: pd.DataFrame, src: Path) -> None:
    side = pd.read_csv(src / "epoch_13n_state_vs_position_cue.csv")
    for r, _t, _c in PANELS:
        g = D[D.readout == r]
        sess = g.groupby("epoch", observed=True).retained.mean().reindex(EPOCHS)
        anim = g.groupby(["animal", "epoch"], observed=True).retained.mean().unstack("epoch").reindex(columns=EPOCHS)
        sc = side[side.position.str.startswith(r)].set_index("epoch").value.reindex(EPOCHS)
        print(f"\n{r}: table   " + " ".join(f"{v:.2f}" for v in TABLE[r]))
        print(f"{r}: 13n sidecar   " + " ".join(f"{v:.3f}" for v in sc))
        print(f"{r}: mean over sessions   " + " ".join(f"{v:.3f}" for v in sess))
        print(f"{r}: mean of animals   " + " ".join(f"{v:.3f}" for v in anim.mean()))
        print(anim.round(3).to_string())


def stats(D: pd.DataFrame) -> dict:
    from wfield_local.analysis_kit import boot_delta, boot_p, holm
    rng = np.random.default_rng(20261102)
    res = {}
    for r, _t, _c in PANELS:
        g = D[D.readout == r]
        pre = {a: list(x.retained) for a, x in g[g.epoch == "pre"].groupby("animal")}
        for e in EPOCHS[1:]:
            post = {a: list(x.retained) for a, x in g[g.epoch == e].groupby("animal")}
            iv, draws = boot_delta(post, pre, rng, draws=True)
            res[(r, e)] = (iv, boot_p(draws))
    keys = list(res)
    ph = holm([res[k][1] for k in keys])
    out = {}
    for k, p_adj in zip(keys, ph):
        iv, p = res[k]
        out[k] = (iv, p, p_adj)
        print(f"  {k[0]:8s} {k[1]:8s} delta {iv.point:+.3f} [{iv.lo:+.3f}, {iv.hi:+.3f}]  p {p:.4f}  Holm {p_adj:.4f}")
    return out


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def draw(D: pd.DataFrame, T: dict, png: Path) -> None:
    from wfield_local.orofacial_cohort import EPOCH_LINE
    from wfield_local.sig_marks import reserve_headroom, star_top
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    rng = np.random.default_rng(0)
    off = {a: o for a, o in zip(ANIMALS, (-0.24, -0.12, 0.0, 0.12))}
    mean_dx = 0.27
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (r, title, _c) in zip(axs, PANELS):
        g = D[D.readout == r]
        ax.axhline(0, color="0.45", ls="--", lw=1.1, zorder=1)
        for (a, e), s in g.groupby(["animal", "epoch"], observed=True):
            x = EPOCHS.index(e) + off[a] + rng.uniform(-0.035, 0.035, len(s))
            ax.scatter(x, s.retained, marker=MARK[a], s=26, facecolors="none", edgecolors=EPOCH_LINE[e],
                       linewidths=1.0, alpha=0.85, zorder=2)
        M = g.groupby(["animal", "epoch"], observed=True).retained.mean().unstack("epoch").reindex(columns=EPOCHS)
        mu, se = M.mean(), M.std(ddof=1) / np.sqrt(M.notna().sum())
        xs = np.arange(len(EPOCHS)) + mean_dx
        ax.errorbar(xs, mu, yerr=se, color="0.15", ls="none", elinewidth=2.2, marker="o", ms=7, mfc="0.15",
                    mec="0.15", capsize=4, capthick=1.6, zorder=3)
        for e in EPOCHS[1:]:
            st = _stars(T[(r, e)][2])
            if st:
                star_top(ax, EPOCHS.index(e), st, color=EPOCH_LINE[e], fontsize=13)
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_xticks(np.arange(len(EPOCHS)))
        ax.set_xticklabels(["Pre", "Acute", "Subacute", "Chronic"], fontsize=11)
        ax.set_xlim(-0.5, len(EPOCHS) - 0.4)
    lo = min(-0.05, float(np.floor(D.retained.min() * 10) / 10))
    axs[0].set_ylim(lo, 1.05)
    axs[0].set_yticks(np.arange(np.ceil(lo * 4) / 4, 1.01, 0.25))
    axs[0].set_ylabel("Above-chance performance retained", fontsize=12)
    axs[1].text(len(EPOCHS) - 0.42, 0.015, "chance", ha="right", va="bottom", fontsize=11, color="0.4")
    reserve_headroom(fig)
    h = [Line2D([], [], ls="none", marker=MARK[a], mfc="none", mec="0.3", ms=7, label=a) for a in ANIMALS]
    h += [Line2D([], [], ls="none", marker="o", ms=7, mfc="0.15", mec="0.15", label="Mean of animals ± SEM")]
    h += [Line2D([], [], ls="none", marker="o", mfc="none", mec=EPOCH_LINE[e], mew=1.6, ms=7,
                 label=e.capitalize()) for e in EPOCHS]
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    save_png_svg(fig, png, dpi=200, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--data-dir", type=Path, default=SRC, help="grant_figures/epoch/data folder with the sidecars")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        fd = _common.load_figdata(a.out, NAME)
        D, T = fd["D"], fd["T"]
    else:
        D = load(a.data_dir)
        check(D, a.data_dir)
        print("\nchange vs pre (fraction retained), boot_delta animals -> sessions:")
        T = stats(D)
        D[["readout", "animal", "epoch", "session_ordinal", "balanced_acc", "chance", "retained"]].to_csv(
            a.out / f"{NAME}.csv", index=False, float_format="%.4f")
        _common.save_figdata(a.out, NAME, {"D": D, "T": T, "src": str(a.data_dir)})
        pd.set_option("display.width", 200)
        print("\nsessions per animal x epoch:")
        print(D.groupby(["readout", "animal", "epoch"], observed=True).size().unstack("epoch").to_string())
    draw(D, T, a.out / f"{NAME}.png")
    print(f"-> {a.out / f'{NAME}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
