"""Does the shared increase follow BEHAVIORAL RECOVERY within animal? (Priya 11-09 "great keep going" on the open
question "whether it tracks behavioral recovery (far-contra aim) within animal").

Per session: the region-mean change beyond licking (`state_space_shift_behavior.py` sessions.csv, "res | <readout>",
minus RESTnl, units of the animal's pre-template pixel RMS) and the observed change ("obs | ..."), joined to the
behavioral recovery axis of Fig 6.3: far-contra FIRST-cued-lick aim vs the animal's pre landing (deg, - = ipsilesional;
`aim_deficit_sessions.csv` farcontra_first). Post-stroke sessions, as `state_space_step_vs_gradual`:
  per-animal r (printed first, rule 8); within-animal slope (animal-demeaned) with a within-animal shuffle null (aim
  permuted within animal, 2,000); MixedLM value ~ aim + (1 | animal) and value ~ aim + C(epoch) + (1 | animal) (does aim
  explain the value WITHIN epochs, or only through the epoch step?).
Outputs: <state_space_recovery>/shift_recovery{tag}/coupling.csv, page_figs/ss_shift_recovery{tag}.png. EXPLORATORY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_chronic_maps as C  # noqa: E402
import state_space_sim as S  # noqa: E402
import state_space_step_vs_gradual as SV  # noqa: E402

from wfield_local.figure_layout import save_png_svg  # noqa: E402
from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE  # noqa: E402

OUT = S.ROOT / f"shift_recovery{S.tag()}"
FIGS = S.ROOT / "page_figs"
READS = ["orofacial contra | before the lick", "orofacial contra | early (0..0.3 s)", "orofacial contra | late (0.3..0.7 s)",
         "orofacial ipsi | before the lick", "orofacial ipsi | early (0..0.3 s)", "orofacial ipsi | late (0.3..0.7 s)",
         "posterior-medial contra | early (0..0.3 s)", "posterior-medial ipsi | early (0..0.3 s)"]


def main():
    d = pd.read_csv(S.ROOT / f"shift_behavior{S.tag()}" / "sessions.csv")
    aim = pd.read_csv(SV.AIM)[["label", "farcontra_first"]].rename(columns={"farcontra_first": "aim"})
    d = d.merge(aim, on="label", how="left")
    names = {}
    for k in ("res", "obs"):
        for i, r in enumerate(READS):
            names[f"{k}{i}"] = f"{k} | {r}"
            d[f"{k}{i}"] = d[f"{k} | {r}"]
    post = d[d.epoch != "pre"]
    rng = np.random.default_rng(0)
    rows = []
    for c, full in names.items():
        r_an = {an: round(float(g[[c, "aim"]].dropna().corr().iloc[0, 1]), 2) for an, g in post.groupby("animal")}
        b, p = SV.shuffle_p(post, c, rng)
        b1, p1 = SV.mixed(post, c, "{c} ~ aim")
        b2, p2 = SV.mixed(post, c, "{c} ~ aim + C(epoch)")
        rows.append({"readout": full, "r_by_animal": r_an, "within_slope_per_deg": b, "shuffle_p": p, "lmm_b": b1,
                     "lmm_p": p1, "lmm_b_within_epoch": b2, "lmm_p_within_epoch": p2,
                     "n_sessions": int(post[[c, "aim"]].dropna().shape[0])})
        print(f"  {full}: r {r_an} | within slope {b:+.4f}/deg p={p:.3f} | lmm {b1:+.4f} p={p1:.3f} | "
              f"+epoch {b2:+.4f} p={p2:.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "coupling.csv", index=False)

    fig, axs = plt.subplots(2, 4, figsize=(19, 9), squeeze=False)
    for ax, (i, r) in zip(axs.ravel(), enumerate(READS)):
        c = f"res{i}"
        pre = d[d.epoch == "pre"]
        ax.scatter(pre.aim, pre[c], color="0.75", s=10, zorder=0)
        for an, g in post.groupby("animal"):
            ax.scatter(g.aim, g[c], marker=ANIMAL_MARKER[an], facecolors="none",
                       edgecolors=[EPOCH_LINE[e] for e in g.epoch], s=22, lw=0.9)
            gg = g.dropna(subset=[c, "aim"])
            if len(gg) > 2:
                bb = np.polyfit(gg.aim, gg[c], 1)
                xx = np.array([gg.aim.min(), gg.aim.max()])
                ax.plot(xx, np.polyval(bb, xx), color="0.35", lw=0.8)
        q = R[R.readout == f"res | {r}"].iloc[0]
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_title(f"{r}\nwithin-animal slope {q.within_slope_per_deg:+.3f}/deg, shuffle p={q.shuffle_p:.3f}\n"
                     f"within epoch (MixedLM + epoch) p={q.lmm_p_within_epoch:.3f}", fontsize=8.5)
        ax.set_xlabel("far-contra aim vs own pre (deg, - = ipsilesional)")
        ax.set_ylabel("change beyond licking vs own pre\n(pre pixel RMS)")
    hs = [plt.Line2D([], [], marker=ANIMAL_MARKER[a], ls="", mfc="none", mec="k", label=a) for a in sorted(ANIMAL_MARKER)]
    hs += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=EPOCH_LINE[e], label=e) for e in EPOCH_LINE if e != "pre"]
    hs += [plt.Line2D([], [], marker="o", ls="", color="0.75", label="pre sessions")]
    fig.legend(handles=hs, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(C._wrap(
        "Does the shared increase follow behavioral recovery? Per session: region-mean change beyond licking (minus "
        "RESTnl, cohort SVD, first-lick-aligned; Fig 10.21's residual) against far-contra first-cued-lick aim vs own pre "
        "(Fig 6.3's recovery axis). Post-stroke sessions: lines = per-animal fits; within-animal slope with a "
        "within-animal shuffle (2,000); 'within epoch' = MixedLM value ~ aim + C(epoch) + (1 | animal)."), y=1.06,
        fontsize=9.5)
    fig.tight_layout()
    save_png_svg(fig, FIGS / f"ss_shift_recovery{S.tag()}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
