"""S12F (six-position, cohort 2): within- vs between-session recovery of far-contralesional 3-D first-lick angle on
session thirds, change from each animal's own pre-stroke value (Priya 11-23; the same analysis as the two-tone S8E,
`thirds_recovery_core.py`, an identical copy of stroke_orofacial_pipeline
scripts/manuscript/recovery_mechanics/thirds_recovery_core.py).

Input: `t1_sessions_firstangle.csv` written by s12f_firstangle.py (item 31): engaged far-contralesional trials, first
post-cue lick 3-D angle (Fig 4D definition), sessions with >= 24 scored trials split into thirds; pre = August
pre-stroke sessions, post = post-stroke days 1-14 (period 'window'). Bootstrap: wfield_local.analysis_kit.boot_delta
(paired animals -> sessions, 4,000 draws, rng seed 0 per quantity).

    python s12f_thirds_boot.py <manuscript_code/figures> <t1_sessions_firstangle.csv> <out_dir> <fig_stem>
Refuses to overwrite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thirds_recovery_core as TC  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
NUM = dict(zip(AN, ["#9", "#10", "#11", "#12"]))


def boot(post: dict, pre: dict):
    from wfield_local import analysis_kit as ak
    iv = ak.boot_delta(post, pre, np.random.default_rng(0))
    return (iv.point, iv.lo, iv.hi) if iv is not None else (np.nan, np.nan, np.nan)


def load(path: str) -> pd.DataFrame:
    S = pd.read_csv(path).dropna(subset=["W_eL"])
    S = S[S.period.isin(["pre", "window"])]
    return pd.DataFrame({"animal": S.animal, "label": S.label, "epoch": S.epoch, "day": S.day,
                         "post": (S.period == "window").astype(int), "early": S.early_eL, "mid": S.mid_eL,
                         "late": S.late_eL, "M": S.M_eL})


def render(figdir: str, S: pd.DataFrame, stem: str):
    """Same 5.6 x 1.35 in page as C2R_E_daycourse_r3_lp (main.tex scale / trim unchanged); same drawing as S8E."""
    sys.path.insert(0, figdir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import vls_style as St
    St.apply()
    plt.rcParams.update({'font.size': 6.8, 'axes.labelsize': 6.8, 'axes.titlesize': 6.8, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'legend.fontsize': 6.3, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2,
                         'ytick.major.size': 2})
    fig, axs = plt.subplots(1, 4, figsize=(5.6, 1.35))
    for ax, a in zip(axs, AN):
        TC.draw_animal(ax, S[S.animal == a], lambda r: St.EP[r.epoch], "o", pre_label_fs=6.3)
        ax.set_title(NUM[a], pad=2)
        ax.set_xlabel("sessions (thirds)", labelpad=8)
    axs[0].set_ylabel("Δ first-lick\nangle (deg)")
    fig.subplots_adjust(left=0.1, right=0.995, bottom=0.24, top=0.86, wspace=0.32)
    for x in ("pdf", "svg", "png"):
        fig.savefig(f"{stem}.{x}", dpi=300 if x == "png" else None)
    plt.close(fig)


def main(figdir: str, t1: str, out: str, stem: str):
    o = Path(out)
    if o.exists() and any(o.iterdir()):
        sys.exit(f"refusing to write into non-empty {o}")
    for x in ("pdf", "svg", "png"):
        if Path(f"{stem}.{x}").exists():
            sys.exit(f"refusing to overwrite {stem}.{x}")
    o.mkdir(parents=True, exist_ok=True)
    S = load(t1)
    S.to_csv(o / "thirds_sessions_sixpos.csv", index=False)
    R, A = TC.analyze(S, "six-position (n = 4)", boot, NUM)
    R.to_csv(o / "thirds_boot_lmm_sixpos.csv", index=False)
    A.to_csv(o / "thirds_per_animal_sixpos.csv", index=False)
    TC.pairs(S, None).to_csv(o / "pairs_anygap_sixpos.csv", index=False)
    G = TC.gap_table(S, "six-position (n = 4)")
    G.to_csv(o / "pair_gaps_sixpos.csv", index=False)
    print(G.pivot_table(index="post", columns="gap", values="n_pairs", fill_value=0))
    print(S[S.post == 1].groupby("animal").day.agg(["min", "max", "size"]))
    with open(o / "tables_sixpos.md", "w", encoding="utf-8") as f:
        f.write("| Quantity | Pre | Post | Stroke effect, bootstrap [95% CI] | Mixed model [95% CI], p "
                "| Per animal: pre -> post (effect) | Same sign |\n|---|---|---|---|---|---|---|\n")
        for _, r in R.iterrows():
            f.write(TC.fmt_row(r) + "\n")
    print((o / "tables_sixpos.md").read_text(encoding="utf-8"))
    print(R[["quantity", "lmm_converged", "lmm_n_obs", "n_units_pre", "n_units_post"]].to_string())
    render(figdir, S, stem)


if __name__ == "__main__":
    main(*sys.argv[1:5])
