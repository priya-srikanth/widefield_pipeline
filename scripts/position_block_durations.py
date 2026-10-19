"""Position-block durations per session (Priya 2026-10-19: "in PS92 especially position blocks last longer, because the
animal was so licky that it reset the ENL a lot").

    python scripts/position_block_durations.py [--jobs 4]

A block = a run of consecutive trials at the same spout position; duration = first trial's strobe (spout arrived) to
the last trial's stop. Why it matters: drift handling assumes position structure lives FASTER than the drift cutoff
(preprocessing order-10 polynomial ~43 min; the encoding model's DRIFT_MIN_PERIOD_S; the two-band candidate's 300 s
slow band). A block of 5-10 min puts genuine position structure inside those slow bands.

Outputs -> <LP_analysis>/<run>/imaging_encoding/qc/: position_blocks.csv (one row per block), position_block_summary.csv
(per session), position_block_durations.png.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def session_blocks(label: str) -> list[dict]:
    """Module-level worker: one row per position block of one session."""
    from wfield_local import config
    from wfield_local import epochs
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    animal, date = config.animal_of(label), "2026" + label.split("_")[-1]
    b = TW.trial_bounds(animal, date, PathResolver())
    b = b[np.isfinite(b.cue_s.to_numpy(float))].reset_index(drop=True)
    start = np.where(np.isfinite(b.strobe_s), b.strobe_s, b.cue_s - 3.0).astype(float)
    stop = np.where(np.isfinite(b.stop_s), b.stop_s, b.cue_s + 3.5).astype(float)
    pos = b.pos_name.astype(str).to_numpy()
    rows, i = [], 0
    ep, day = epochs.epoch_of(label), epochs.days_since_stroke(label)
    while i < len(pos):
        j = i
        while j + 1 < len(pos) and pos[j + 1] == pos[i]:
            j += 1
        rows.append({"label": label, "animal": animal, "epoch": ep, "day": day, "block": len(rows), "pos": pos[i],
                     "n_trials": j - i + 1, "minutes": (stop[j] - start[i]) / 60.0,
                     "s_per_trial": (stop[j] - start[i]) / (j - i + 1)})
        i = j + 1
    return rows


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    out = oo.root(PathResolver()) / a.run / "imaging_encoding" / "qc"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    got, fail = ak.fan_sessions(ce.sessions_available(), session_blocks, jobs=a.jobs)
    for it, err in fail:
        print(f"  FAILED {it}: {err}")
    B = pd.DataFrame([r for _lab, rs in got for r in rs])
    B = B[B.n_trials >= 2]                                           # single-trial "blocks" = log glitches / edges
    B.to_csv(out / "position_blocks.csv", index=False, float_format="%.4g")
    S = B.groupby(["label", "animal", "epoch", "day"]).minutes.agg(
        median="median", p90=lambda x: np.percentile(x, 90), max="max",
        frac_over_5min=lambda x: np.mean(x > 5), frac_over_10min=lambda x: np.mean(x > 10)).reset_index()
    S.to_csv(out / "position_block_summary.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print("per animal x epoch: median block (min) / 90th pct / fraction of blocks > 5 min / > 10 min")
    g = B.groupby(["animal", "epoch"]).minutes
    T = pd.DataFrame({"median": g.median(), "p90": g.quantile(0.9), "max": g.max(),
                      ">5min": g.apply(lambda x: np.mean(x > 5)), ">10min": g.apply(lambda x: np.mean(x > 10)),
                      "n_blocks": g.size()})
    print(T.round(2).to_string())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    acol = {"PS92": "#1f77b4", "PS93": "#d62728", "PS94": "#ff7f0e", "PS95": "#2ca02c"}
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for an, gg in S.groupby("animal"):
        axs[0].plot(gg.day, gg["median"], "o", color=acol.get(an, "k"), ms=5, label=f"{an} median")
        axs[0].plot(gg.day, gg.p90, "^", color=acol.get(an, "k"), ms=4, alpha=0.6)
    for y, lab_ in ((5, "two-band slow band (5 min)"), (10, "encoding drift cutoff (10 min)")):
        axs[0].axhline(y, color="0.4", ls="--", lw=0.8)
        axs[0].text(axs[0].get_xlim()[0], y, " " + lab_, va="bottom", fontsize=9, color="0.3")
    axs[0].axvline(0, color="k", lw=0.6)
    axs[0].set_xlabel("day from stroke")
    axs[0].set_ylabel("position-block duration (min)")
    axs[0].set_title("per session: median (o) and 90th percentile (^) block duration")
    axs[0].legend(frameon=False, fontsize=9)
    for an, gg in B.groupby("animal"):
        axs[1].hist(gg.minutes, bins=np.arange(0, 25, 0.5), histtype="step", lw=1.5, color=acol.get(an, "k"),
                    density=True, label=an)
    axs[1].axvline(5, color="0.4", ls="--", lw=0.8)
    axs[1].axvline(10, color="0.4", ls="--", lw=0.8)
    axs[1].set_xlabel("block duration (min)")
    axs[1].set_ylabel("density")
    axs[1].set_title("all blocks, per animal")
    axs[1].legend(frameon=False)
    fig.suptitle("Position-block durations (first strobe .. last stop of a same-position run)", fontsize=12)
    fig.tight_layout()
    oc.save_png_svg(fig, out / "position_block_durations.png", dpi=130)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
