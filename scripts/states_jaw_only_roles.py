"""Which GLM-HMM ROLE do JAW-ONLY trials fall into (Priya 11-20)? Cohort 1 (six-position, PS92-PS95), behavior only.

For each trial class (attempt = contact + incomplete / jaw-only / no response; the GLM-HMM's own outcome levels) the
share of its trials whose most probable state is engaged / partial / disengaged, per epoch and pooled, for three
models:
  production  pooled K5, design v2 (`behavior_states.glmhmm_table`, states 0 + 1 merged; roles `glmhmm_roles`)
  ipsimid     K5 refit warm-started from production, outcome observed on ipsilesional + middle spouts only
  init        K5 refit warm-started from production, outcome = response initiation (any jaw opening = responded)
The refits are the review C2 control fits (`s3_glmhmm_refit.py` in ``--refit-dir``; roles from each refit's own state
profiles by the `glmhmm_roles` rule). Every trial of the analysis sessions (the GLM-HMM's trial set), near-middle spout
dropped for every class (`behavior_states.JAW_BLIND_POSITIONS`); sessions with >= `ROLE_SHARE_MIN_TRIALS` trials of a
class enter that class's share; animal-level = mean over the animal's sessions, then over animals. Whole sessions
(primary) and the first `COMMON_SESSION_MIN` min (length-matched).

Test: share of jaw-only trials in each role, each post epoch vs pre, `orofacial_stats.session_change_tests` (session
MWU + BH within metric, and `analysis_kit.boot_delta` vs each animal's own pre); "*" = q < .05 AND the CI excludes 0.

    PYTHONPATH=<repo> python scripts/states_jaw_only_roles.py [--refit-dir DIR]

Writes to `behavior_states.out_dir()/page_figs/`: jaw_only_role_share.png (+ svg), jaw_only_role_share_epochs.png,
glmhmm_role_share_by_class{,_per_session,_tests}.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402
from wfield_local import behavior_states as bs  # noqa: E402
from wfield_local import cat_points as cp  # noqa: E402
from wfield_local import orofacial_cohort as oc  # noqa: E402
from wfield_local import orofacial_stats as ost  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402

REFIT_DIR = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/review/"
                 "c2_treadmill_2026-11-20/c2_engagement_tim")
MODELS = {"production": "production K5 v2", "ipsimid": "refit: outcome on ipsi + mid spouts only",
          "init": "refit: any jaw opening = response"}
CLASS_NAMES = {"attempt": "attempt", "jaw_only": "jaw-only", "no_response": "no response"}
EPS = list(oc.EPOCH_ORDER)
ECOL = oc.EPOCH_LINE
ROLE_COL = {r: bs.ROLE_COLORS[r][0] for r in bs.ROLES}


def trial_table(refit_dir: Path) -> pd.DataFrame:
    """The production GLM-HMM trial table + role_production / role_ipsimid / role_init + t_min."""
    T = bs.glmhmm_table()
    T["role_production"] = bs.glmhmm_role_column(T)
    for v in ("ipsimid", "init"):
        R = pd.read_csv(refit_dir / f"glmhmm_refit_{v}_K5.csv.gz", usecols=["label", "trial_id", "role"])
        T = T.merge(R.rename(columns={"role": f"role_{v}"}), on=["label", "trial_id"], how="left", validate="1:1")
        miss = T[f"role_{v}"].isna().mean()
        if miss > 0.001:
            raise RuntimeError(f"{v}: {miss:.1%} of production trials have no refit role")
    return bs.add_session_time(T, bs.load_collected()["trials"])


def tables(T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per, summ = [], []
    for match, D in (("whole", T), ("first95", T[~(T.t_min > bs.COMMON_SESSION_MIN)])):
        for m in MODELS:
            S = bs.role_share_sessions(D, f"role_{m}").assign(model=m, match=match)
            per.append(S)
            summ.append(bs.role_share_summary(S).assign(model=m, match=match))
    return pd.concat(per, ignore_index=True), pd.concat(summ, ignore_index=True)


def tests(P: pd.DataFrame) -> pd.DataFrame:
    """Jaw-only share in each role, post epoch vs pre (whole sessions), per model."""
    out = []
    for m in MODELS:
        S = P[(P.model == m) & (P.match == "whole") & (P.cls == "jaw_only") & (P.n_trials >= bs.ROLE_SHARE_MIN_TRIALS)]
        S = S.assign(date=S.label.str.split("_").str[1], pos="all")
        r = ost.session_change_tests(S, [f"share_{k}" for k in bs.ROLES])
        out.append(r.assign(model=m))
    R = pd.concat(out, ignore_index=True)
    R["star"] = R.headline_sig & R.boot_sig
    return R


def fig_stacked(M: pd.DataFrame, od: Path) -> None:
    """Rows = models; columns = pooled + epochs; one stacked bar per trial class (animal-mean role shares)."""
    cols = ["pooled", *EPS]
    fig, axs = plt.subplots(len(MODELS), len(cols), figsize=(2.3 * len(cols), 2.5 * len(MODELS)), sharey=True,
                            squeeze=False)
    x = np.arange(len(CLASS_NAMES))
    for i, m in enumerate(MODELS):
        for j, ep in enumerate(cols):
            ax = axs[i, j]
            d = M[(M.model == m) & (M.epoch == ep) & (M.match == "whole")].set_index("cls").reindex(list(CLASS_NAMES))
            bottom = np.zeros(len(x))
            for r in bs.ROLES:
                v = d[r].to_numpy(float)
                ax.bar(x, v, 0.7, bottom=bottom, color=ROLE_COL[r], edgecolor="white", lw=0.5,
                       label=r if (i == 0 and j == 0) else None)
                bottom += np.nan_to_num(v)
            for k, c in enumerate(CLASS_NAMES):
                ax.text(x[k], 1.02, f"{int(d.loc[c, 'n_sessions'])}", ha="center", va="bottom", fontsize=6, color="0.4")
            ax.set_xticks(x, [n.replace(" ", "\n") for n in CLASS_NAMES.values()], fontsize=7)
            ax.set_ylim(0, 1.12)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(ep, fontsize=9, color="k" if ep == "pooled" else ECOL[ep])
        axs[i, 0].set_ylabel(f"{m}\nshare of trials in role", fontsize=8)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.25), ncol=3)
    fig.suptitle("GLM-HMM role of each trial class (mean over animals of session shares; near-mid spout excluded; "
                 "grey number = sessions)", fontsize=9)
    oc._tidy(fig)
    fig.tight_layout()
    save_png_svg(fig, od / "jaw_only_role_share.png")
    plt.close(fig)
    print("  wrote jaw_only_role_share")


def fig_epochs(P: pd.DataFrame, R: pd.DataFrame, od: Path) -> None:
    """Jaw-only share in each role by epoch: sessions (open markers) + pooled 95 % CI; '*' = q < .05 AND CI excl. 0."""
    fig, axs = plt.subplots(len(MODELS), len(bs.ROLES), figsize=(3.3 * len(bs.ROLES), 2.7 * len(MODELS)),
                            squeeze=False, sharey=True)
    xs = {e: k for k, e in enumerate(EPS)}
    rng_seed = 0
    for i, m in enumerate(MODELS):
        S = P[(P.model == m) & (P.match == "whole") & (P.cls == "jaw_only") & (P.n_trials >= bs.ROLE_SHARE_MIN_TRIALS)]
        for j, r in enumerate(bs.ROLES):
            ax = axs[i, j]
            col = f"share_{r}"
            for e in EPS:
                G = S[S.epoch == e]
                cp.session_dots(ax, xs[e], G[col].to_numpy(float), G.animal.to_numpy(), ECOL[e], jitter=0.07,
                                seed=xs[e] * 97)
                by = {a: g[col].tolist() for a, g in G.groupby("animal", sort=True)}
                iv = ak.boot_ci(by, np.random.default_rng(rng_seed)) if by else None
                if iv is not None:
                    cp.summary_point(ax, xs[e], iv.point, np.array([[iv.point - iv.lo], [iv.hi - iv.point]]), ECOL[e],
                                     label=e if (i == 0 and j == 0) else None)
                t = R[(R.model == m) & (R.metric == col) & (R.epoch == e)]
                if len(t) and bool(t.star.iloc[0]):
                    sig_marks.star_top(ax, xs[e], "*", color=ECOL[e], fontsize=12)
            ax.set_xticks(range(len(EPS)), EPS, fontsize=7)
            ax.set_xlim(-0.6, len(EPS) - 0.4)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"jaw-only trials in the {r} role", fontsize=9)
        axs[i, 0].set_ylabel(f"{m}\nshare (per session)", fontsize=8)
    cp.add_animal_entries(axs[0, 0], sorted(P.animal.unique()))
    axs[0, 0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.suptitle("Jaw-only trials: GLM-HMM role by epoch (sessions with >= "
                 f"{bs.ROLE_SHARE_MIN_TRIALS} jaw-only trials; near-mid excluded)", fontsize=9)
    oc._tidy(fig)
    sig_marks.reserve_headroom(fig)
    fig.tight_layout()
    save_png_svg(fig, od / "jaw_only_role_share_epochs.png")
    plt.close(fig)
    print("  wrote jaw_only_role_share_epochs")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    ap.add_argument("--refit-dir", type=Path, default=REFIT_DIR)
    a = ap.parse_args(argv)
    od = bs.out_dir() / "page_figs"
    od.mkdir(parents=True, exist_ok=True)
    T = trial_table(a.refit_dir)
    P, M = tables(T)
    R = tests(P)
    P.to_csv(od / "glmhmm_role_share_by_class_per_session.csv", index=False)
    M.to_csv(od / "glmhmm_role_share_by_class.csv", index=False)
    R.to_csv(od / "glmhmm_role_share_by_class_tests.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    show = M[M.match == "whole"][["model", "cls", "epoch", "n_animals", "n_sessions", "n_trials", *bs.ROLES]]
    print(show.round(3).to_string(index=False))
    print(R[["model", "metric", "epoch", "n_sessions", "n_animals", "mean_change", "p", "q_fdr", "boot_change",
             "boot_lo", "boot_hi", "star"]].round(3).to_string(index=False))
    fig_stacked(M, od)
    fig_epochs(P, R, od)
    print("out:", od)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
