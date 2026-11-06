"""T19: do the quiet orofacial events meet the Steinmetz et al. 2017 interictal criteria? (cohort1, 2026-11-01)

Steinmetz et al. 2017 (eNeuro, GCaMP6 transgenic lines): interictal-like events are LARGE (> ~10 % dF/F), POSITIVE,
brief (~150-200 ms), recur at 0.1-0.5 Hz with a refractory gap, largest anterolaterally and bilaterally symmetric.
Our events: v2 raw events (quiet_spikes_v2 sessions/*.npz), engaged quiet, k = 4 x the animal's August-pre robust SD.
Checks per session: (a) inter-event-interval (IEI) distribution of orofacial UP events within continuous quiet runs --
a refractory gap would show as a dearth of IEIs < 1 s relative to a Poisson process at the same rate (ratio observed /
expected for IEI < 0.5 s and 0.5-1 s; the detector's 0.3-s merge window removes IEIs < 0.3 s by construction);
(b) coefficient of variation of IEIs (periodic << 1, Poisson ~1); (c) polarity balance (UP / DOWN count ratio; interictal
events are positive-only); (d) height (median, share > 10 %); sizes and symmetry are in T11 / T2.

    python scripts/cohort2_qc/quiet_steinmetz_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_spikes_v2 import MERGE_S, V2  # noqa: E402
from wfield_local import config  # noqa: E402

OUT = V2 / "steinmetz_check"
K = 4.0
EP = ["pre", "acute", "subacute", "chronic"]


def main():
    from quiet_area_spikes import columns
    from quiet_fluct import animal_sd
    from wfield_local import epochs
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    cols = columns()
    gi = cols.index("orofacial")
    sd = animal_sd()
    rows = []
    for f in sorted((V2 / "sessions").glob("*.npz")):
        lab, ep = f.stem, epochs.epoch_of(f.stem)
        if ep is None or config.early_pre(lab) or lab[:4] not in sd:
            continue
        z = np.load(f)
        fs = float(z["fs"])
        th = K * sd[lab[:4]][gi]
        up = z["pos"]
        up = up[(up[:, 0] == gi) & (up[:, 3] == 1) & (up[:, 2] > th)]
        dn = z["neg"]
        dn = dn[(dn[:, 0] == gi) & (dn[:, 3] == 1) & (dn[:, 2] > th)]
        mins = float(z["n_quiet_eng"]) / fs / 60
        t = np.sort(up[:, 1]) / fs
        iei = np.diff(t)
        iei = iei[iei < 10.0]                                # within-run intervals (quiet runs are short; cap 10 s)
        rate = len(up) / (mins * 60) if mins else np.nan
        r = dict(label=lab, animal=lab[:4], epoch=ep, n_up=len(up), n_down=len(dn), rate_up_hz=rate,
                 up_down_ratio=len(up) / max(len(dn), 1), med_h=float(np.median(up[:, 2])) if len(up) else np.nan,
                 share_gt10=float(np.mean(up[:, 2] > 10)) if len(up) else np.nan, n_iei=len(iei))
        if len(iei) >= 20 and np.isfinite(rate) and rate > 0:
            # Poisson expectation for IEI in [a, b) conditional on IEI >= MERGE_S (merge window) and < 10 s
            def pexp(a, b, rate=rate):
                F = lambda x: 1 - np.exp(-rate * x)  # noqa: E731
                return (F(b) - F(a)) / (F(10.0) - F(MERGE_S))
            for a, b, nm in ((MERGE_S, 0.5, "iei_0.3-0.5"), (0.5, 1.0, "iei_0.5-1"), (1.0, 3.0, "iei_1-3")):
                obs = np.mean((iei >= a) & (iei < b))
                r[f"{nm}_obs_over_poisson"] = float(obs / pexp(a, b)) if pexp(a, b) > 0 else np.nan
            r["iei_cv"] = float(iei.std() / iei.mean())
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "steinmetz_check_sessions.csv", index=False)
    for m in ("rate_up_hz", "up_down_ratio", "med_h", "share_gt10", "iei_0.3-0.5_obs_over_poisson",
              "iei_0.5-1_obs_over_poisson", "iei_1-3_obs_over_poisson", "iei_cv"):
        print(f"\n{m} (animal x epoch, session median):\n",
              S.pivot_table(index="animal", columns="epoch", values=m, aggfunc="median").reindex(columns=EP)
              .round(3).to_string())
    figure(S)
    (OUT / "README.txt").write_text(__doc__)


def figure(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    ms = [("up_down_ratio", "UP / DOWN event count\n(interictal: positive only -> >> 1)"),
          ("iei_0.3-0.5_obs_over_poisson", "IEI 0.3-0.5 s, observed / Poisson\n(refractory gap -> << 1)"),
          ("iei_0.5-1_obs_over_poisson", "IEI 0.5-1 s, observed / Poisson"),
          ("iei_cv", "IEI coefficient of variation\n(periodic << 1, Poisson ~ 1)")]
    fig, ax = plt.subplots(1, len(ms), figsize=(4 * len(ms), 4))
    animals = sorted(S.animal.unique())
    for a, (m, lab) in zip(ax, ms):
        for k, ep in enumerate(EP):
            G = S[S.epoch == ep]
            for an, g in G.groupby("animal"):
                a.plot(np.full(len(g), k + (animals.index(an) - 1.5) * 0.08), g[m], ANIMAL_MARKER[an], mfc="none",
                       mec=EPOCH_LINE[ep], ms=4)
            a.plot([k - 0.25, k + 0.25], [G[m].median()] * 2, color=EPOCH_LINE[ep], lw=2.5)
        a.axhline(1, color="0.5", lw=0.8, ls="--")
        a.set_xticks(range(4))
        a.set_xticklabels(EP, fontsize=8)
        a.set_title(lab, fontsize=8)
    fig.suptitle("T19: Steinmetz et al. 2017 interictal criteria applied to quiet orofacial events (engaged quiet, "
                 "> 4 x pre SD); markers = sessions (shape = animal), bar = median", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "quiet_steinmetz_check.png", dpi=110)
    print("wrote", OUT / "quiet_steinmetz_check.png")


if __name__ == "__main__":
    main()
