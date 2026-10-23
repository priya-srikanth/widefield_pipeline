"""QC (Priya 10-23): SIGNED treadmill speed vs cortex-mean dF/F per session -- forward > 1.5 mm/s and backward < -0.3 mm/s shaded, cues marked; whole session + 3-min zoom + signed speed histogram + cortex level by signed speed bin. Usage: python scripts/qc_running_speed.py [labels...] (default PS92_0811 PS95_0812 PS94_0921); unset COMBINED_ENCODING_LICKS for sessions without cam1 licks."""
import sys

import numpy as np


def main(labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    out = oo.root(rv) / oc.RUN / "imaging_encoding" / "frozen_encoder_gate-attempted_v6thr1" / "figures"
    fig, axs = plt.subplots(len(labels) * 2, 2, figsize=(15, 3.1 * len(labels) * 2),
                            gridspec_kw={"width_ratios": [4, 1]}, squeeze=False)
    for i, lab in enumerate(labels):
        an, date = config.animal_of(lab), "2026" + lab.split("_")[-1]
        V, ft, Ub, _flat, _pa, _nm, _sh, pstat = ce.load_imaging(lab)
        B = ce.load_behaviour(an, date, rv)
        tr, vr = (np.asarray(x, float) for x in B.analog["running"])
        spd = np.interp(ft, tr, vr)                       # SIGNED (forward +)
        ctx = (V @ Ub[pstat].mean(0)).astype(float) * 1e2
        cues = B.trials.cue_s.to_numpy(float)
        cues = cues[np.isfinite(cues)]
        t0 = ft[0]
        frac1, frac3, fracneg = np.mean(spd > 1.5), np.mean(spd > 3), np.mean(spd < -0.3)
        mid = ft[0] + 0.4 * (ft[-1] - ft[0])
        for row, (lo, hi, ttl) in enumerate(((ft[0], ft[-1], "whole session"), (mid, mid + 180, "3-min zoom"))):
            ax = axs[2 * i + row][0]
            m = (ft >= lo) & (ft <= hi)
            tt = (ft[m] - t0) / 60
            ax.plot(tt, spd[m], color="k", lw=0.5)
            y0, y1 = min(spd[m].min() * 1.05, -1), spd[m].max() * 1.05
            ax.fill_between(tt, y0, y1, where=spd[m] > 1.5, color="tab:orange", alpha=0.3, lw=0, label="> +1.5 mm/s")
            ax.fill_between(tt, y0, y1, where=spd[m] < -0.3, color="tab:purple", alpha=0.35, lw=0,
                            label="< -0.3 mm/s (backward)")
            ax.axhline(1.5, color="tab:orange", lw=0.8, ls="--")
            ax.axhline(0, color="0.5", lw=0.6)
            ax.axhline(-0.3, color="tab:purple", lw=0.8, ls="--")
            ax.set_ylabel("speed (mm/s)")
            ax2 = ax.twinx()
            ax2.plot(tt, ctx[m], color="tab:blue", lw=0.5, alpha=0.8)
            ax2.set_ylabel("cortex dF/F (%)", color="tab:blue")
            cm = (cues >= lo) & (cues <= hi)
            for c in (cues[cm] - t0) / 60:
                ax.axvline(c, color="tab:green", lw=0.4, alpha=0.5)
            ax.set_xlim(tt[0], tt[-1])
            ax.set_title(f"{lab} ({ttl}): frames > +1.5 mm/s {frac1:.1%}, > +3 {frac3:.1%}, < -0.3 {fracneg:.1%}; green = cues", fontsize=9)
            if i == 0 and row == 0:
                ax.legend(loc="upper right", fontsize=7)
        ax = axs[2 * i][1]
        ax.hist(np.sign(spd) * np.log10(1 + np.abs(spd) / 0.05), bins=160, color="0.4")
        for v, c in ((1.5, "tab:orange"), (3, "tab:red"), (-0.3, "tab:purple"), (0, "0.5")):
            ax.axvline(np.sign(v) * np.log10(1 + abs(v) / 0.05), color=c, ls="--")
        ax.set_yscale("log")
        ticks = np.array([-3, -1, -0.3, 0, 0.3, 1, 3, 10, 25])
        ax.set_xticks(np.sign(ticks) * np.log10(1 + np.abs(ticks) / 0.05))
        ax.set_xticklabels([f"{x:g}" for x in ticks], fontsize=6)
        ax.set_xlabel("signed speed (mm/s, symlog)")
        ax.set_title("speed distribution (all frames)", fontsize=8)
        ax = axs[2 * i + 1][1]
        bins = np.array([-30, -1, -0.3, -0.1, 0.1, 0.5, 1, 1.5, 3, 5, 8, 12, 30])
        idx = np.digitize(spd, bins) - 1
        mu = [ctx[idx == k].mean() if (idx == k).sum() > 50 else np.nan for k in range(len(bins) - 1)]
        ax.plot(range(len(mu)), np.array(mu) - np.nanmean(ctx), "o-")
        ax.set_xticks(range(len(mu)))
        ax.set_xticklabels([f"{a:g}-{b:g}" for a, b in zip(bins[:-1], bins[1:])], rotation=60, fontsize=6)
        ax.set_ylabel("cortex dF/F vs mean (%)", fontsize=7)
        ax.axvline(3.5, color="0.6", lw=0.5)
        ax.set_title("raw cortex level by SIGNED speed bin", fontsize=8)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    p = out / "qc_running_threshold_signed.png"
    fig.savefig(p, dpi=100)
    print(p)


if __name__ == "__main__":
    main(sys.argv[1:] or ["PS92_0811", "PS95_0812", "PS94_0921"])
