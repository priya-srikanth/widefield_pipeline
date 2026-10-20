"""Target vs executed with the cam4 and 3-D executed sets, side by side (Priya 10-20: "did you make 3d figures").

Reads the four `movement_first_subset-nokin_exec-<mode>` runs (same licks in every arm; DECISIONS 2026-10-19 night,
2026-10-20 morning) and draws unique target / unique executed per mode, epoch and area, for ALL variance (lick window)
and TRIAL-TO-TRIAL variance (cue 0-1.5 s, around each position's mean). Animal dots. Output: the xyz2 run's figures/.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

#: (output-folder suffix after `movement_first_subset-nokin_`, label). 3dxyz RETIRED (coordinate mismatch, 10-20);
#: polar = the coordinate-matched angular contest, run with combined licks + binned contest (10-20 queue)
MODES = [("exec-cam4m", "cam4\nangle/reach"), ("exec-3d", "3-D\nangle/reach"),
         ("exec-xyz2", "3-D xyz vs\nspout xyz"), ("exec-polar_licks-combined_bins", "3-D polar vs\nspout polar")]
AREAS = [("cortex", "cortex"), ("MOs", "MOs"), ("SSp-m", "SSp-m")]
NEED = ["nokin:none", "nokin:target", "nokin:exec", "nokin:target+exec"]


def _lick_table(E: Path) -> pd.DataFrame:
    from wfield_local import epochs
    rows = []
    for m, _ in MODES:
        for f in glob.glob(str(E / f"movement_first_subset-nokin_{m}" / "sessions" / "*_regions.csv")):
            R = pd.read_csv(f)
            lab = Path(f).stem[:-8]
            for area, _ in AREAS:
                sub = R[(R.window == "lick") & (R.area.str.startswith(area)) & ~R.area.str.contains("ul|bfd|n_|tr")]
                for a_, g in sub.groupby("area"):
                    r = g.set_index("model").r2
                    if set(NEED) <= set(r.index):
                        rows.append({"mode": m, "label": lab, "animal": lab[:4], "epoch": epochs.epoch_of(lab),
                                     "area": area, "sub": a_, "u_target": r["nokin:target+exec"] - r["nokin:exec"],
                                     "u_exec": r["nokin:target+exec"] - r["nokin:target"]})
    D = pd.DataFrame(rows)
    return D.groupby(["mode", "label", "animal", "epoch", "area"])[["u_target", "u_exec"]].mean().reset_index()


def _trial_table(E: Path) -> pd.DataFrame:
    out = []
    for m, _ in MODES:
        f = E / f"movement_first_subset-nokin_{m}" / "tables" / "tr_contest_0-1.5s.csv"
        if f.exists():
            T = pd.read_csv(f)
            T = T[T.variance == "trial"]
            out.append(T.assign(mode=m))
    return pd.concat(out) if out else pd.DataFrame()


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    E = oo.root(PathResolver()) / oc.RUN / "imaging_encoding"
    L, T = _lick_table(E), _trial_table(E)
    panels = [("lick window, all variance", L), ("cue 0-1.5 s, TRIAL-TO-TRIAL variance", T)]
    fig, axs = plt.subplots(2 * len(panels), len(AREAS), figsize=(4.6 * len(AREAS), 3.1 * 2 * len(panels) + 1.2),
                            squeeze=False)
    for pi, (ptitle, D) in enumerate(panels):
        for ei, ep in enumerate(("pre", "acute")):
            for ai, (area, alab) in enumerate(AREAS):
                ax = axs[2 * pi + ei][ai]
                if D.empty:
                    ax.set_axis_off()
                    continue
                for mi, (m, _mlab) in enumerate(MODES):
                    g = D[(D["mode"] == m) & (D.epoch == ep) & (D.area == area)]
                    if g.empty:
                        continue
                    for k, (col, c) in enumerate((("u_target", "#6a3d9a"), ("u_exec", "#d95f02"))):
                        x = mi + (k - 0.5) * 0.36
                        v = g[col].to_numpy() * 1000
                        ax.bar(x, np.nanmean(v), 0.34, color=c, alpha=0.85,
                               label=("unique target" if col == "u_target" else "unique executed")
                               if (mi == 0 and ai == 0 and ei == 0 and pi == 0) else None)
                        ax.plot(np.full(len(v), x), v, ".", color="k", ms=3, alpha=0.5)
                ax.axhline(0, color="0.5", lw=0.6)
                ax.set_xticks(range(len(MODES)))
                ax.set_xticklabels([ml for _, ml in MODES] if (2 * pi + ei) == 2 * len(panels) - 1 else [],
                                   fontsize=7.5)
                ax.set_title(f"{alab} -- {ep} ({ptitle})", fontsize=9)
                if ai == 0:
                    ax.set_ylabel(r"unique $\Delta$R$^2$ x1000")
    axs[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Target vs executed with the cam4 and 3-D executed sets -- PS92 (8 pre, 3 acute sessions; identical "
                 "licks in every arm; dots = sessions)\n'3-D endpoint vs spout 3-D' is the coordinate-matched contest "
                 "(both sets = 3-D ml / ap / dv). One animal: decides what to run on the cohort, not the conclusion.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = E / "movement_first_subset-nokin_exec-xyz2" / "figures" / "exec_sets_compared_PS92.png"
    writeguard.assert_writable(out.parent)
    fig.savefig(out, dpi=130)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
