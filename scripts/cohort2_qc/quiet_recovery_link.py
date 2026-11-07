"""Do quiet-period measures track behavioral recovery? (cohort1, 2026-11-01)

Priya 11-01: "does synchronization change / fluctuation event rate / 1-4 Hz (or other) power correlate with recovery?"
Within animal, over post-stroke sessions, each quiet measure (change vs the animal's August pre) against far-contra hit
rate / pre (recovery_timing.series, the T14 behavior measure), raw and with log(day) removed from both (the shared
recovery time course); permutation p (session values shuffled within animal, 5000 draws) on the mean of the four animals'
r. Measures: face-regressed orofacial event rate (log ratio to pre), synchronization index (log10 PSD 1-4 / 8-15 Hz) for
orofacial and its ipsi - contra difference, 1-4 Hz and 8-15 Hz power (log10, orofacial ipsi / contra), 0.5-1 Hz power.

    python scripts/cohort2_qc/quiet_recovery_link.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_fluct import FL, TRACE_COLS, load  # noqa: E402

OUT = FL.parent / "recovery_link"
FRT = FL.parent / "fr_regressed_own" / "quiet_events_fr_sessions.csv"


def band(fr, P, lo, hi):
    m = (fr >= lo) & (fr < hi)
    return np.log10(np.nanmean(P[:, m], 1))


def measures():
    D = load()
    rows = []
    for k, d in D.items():
        fr, P = d["freqs"], d["psd"]
        b14, b815, b051 = band(fr, P, 1, 4), band(fr, P, 8, 15), band(fr, P, 0.5, 1)
        j = {c: TRACE_COLS.index(c) for c in TRACE_COLS}
        rows.append(dict(label=k, animal=k[:4], epoch=d["epoch"],
                         sync_oro=b14[j["orofacial"]] - b815[j["orofacial"]],
                         sync_oro_ipsi_minus_contra=(b14[j["orofacial_left"]] - b815[j["orofacial_left"]])
                         - (b14[j["orofacial_right"]] - b815[j["orofacial_right"]]),
                         sync_cortex=b14[j["cortex"]] - b815[j["cortex"]],
                         p14_oro_ipsi=b14[j["orofacial_left"]], p14_oro_contra=b14[j["orofacial_right"]],
                         p815_oro_ipsi=b815[j["orofacial_left"]], p815_oro_contra=b815[j["orofacial_right"]],
                         p051_oro=b051[j["orofacial"]]))
    S = pd.DataFrame(rows)
    E = pd.read_csv(FRT)[["label", "orofacial|res_both"]].rename(columns={"orofacial|res_both": "rate"})
    S = S.merge(E, on="label", how="left")
    S["log_rate"] = np.log10(S.rate + 0.2)
    meas = [c for c in S.columns if c not in ("label", "animal", "epoch", "rate")]
    pre = S[S.epoch == "pre"].groupby("animal")[meas].mean()
    for m in meas:
        S[m] = S[m] - S.animal.map(pre[m])
    return S, meas


def resid(y, d):
    X = np.column_stack([np.ones(len(y)), d])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def main():
    from recovery_timing import series
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    S, meas = measures()
    B = series("lick")[["label", "day", "beh"]]
    G = S.merge(B, on="label").dropna(subset=["beh"])
    G = G[(G.epoch != "pre") & (G.day > 0)]
    rng = np.random.default_rng(0)
    out = []
    for m in meas:
        for kind in ("raw", "detrended"):
            rs, pairs = {}, []
            for an, g in G.dropna(subset=[m]).groupby("animal"):
                if len(g) < 5:
                    continue
                b, y = g.beh.to_numpy(float), g[m].to_numpy(float)
                if kind == "detrended":
                    dd = np.log(g.day.to_numpy(float))
                    b, y = resid(b, dd), resid(y, dd)
                rs[an] = float(np.corrcoef(b, y)[0, 1])
                pairs.append((b, y))
            obs = float(np.mean(list(rs.values())))
            null = np.array([np.mean([np.corrcoef(b, rng.permutation(y))[0, 1] for b, y in pairs]) for _ in range(5000)])
            out.append(dict(measure=m, kind=kind, mean_r=obs, p_perm=float(np.mean(np.abs(null) >= abs(obs))),
                            n_same_sign=int(sum(np.sign(v) == np.sign(obs) for v in rs.values())), **rs))
    T = pd.DataFrame(out)
    T.to_csv(OUT / "recovery_link.csv", index=False)
    G.to_csv(OUT / "recovery_link_sessions.csv", index=False)
    print("within-animal r vs far-contra hit rate / pre (post sessions); negative = measure falls as behavior recovers\n",
          T.round(3).to_string())
    figure(G, T, meas)
    (OUT / "README.txt").write_text(__doc__)


def figure(G, T, meas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    show = ["log_rate", "sync_oro", "sync_oro_ipsi_minus_contra", "p14_oro_ipsi", "p815_oro_ipsi", "p051_oro"]
    names = {"log_rate": "event rate (log10, face-regressed)", "sync_oro": "synchronization, orofacial",
             "sync_oro_ipsi_minus_contra": "synchronization, orofacial ipsi - contra",
             "p14_oro_ipsi": "1-4 Hz power, orofacial ipsi (log10)", "p815_oro_ipsi": "8-15 Hz power, orofacial ipsi",
             "p051_oro": "0.5-1 Hz power, orofacial"}
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for a, m in zip(ax.ravel(), show):
        for an, g in G.groupby("animal"):
            a.plot(g.beh, g[m], ANIMAL_MARKER[an], mfc="none", mec="k", ms=6, label=an)
        t = T.set_index(["measure", "kind"])
        a.axhline(0, color="0.6", lw=0.6)
        a.set_xlabel("far-contra hit rate / pre")
        a.set_ylabel("change vs pre")
        a.set_title(f"{names[m]}\nwithin-animal r raw {t.loc[(m, 'raw'), 'mean_r']:+.2f} "
                    f"(p {t.loc[(m, 'raw'), 'p_perm']:.3f}); log-day removed {t.loc[(m, 'detrended'), 'mean_r']:+.2f} "
                    f"(p {t.loc[(m, 'detrended'), 'p_perm']:.3f})", fontsize=8)
    ax[0, 2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.suptitle("Do quiet-period measures track behavioral recovery? Post-stroke sessions, change vs each animal's pre "
                 "against far-contra hit rate / pre (markers = animals)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.5, h_pad=2.5)
    fig.savefig(OUT / "quiet_recovery_link.png", dpi=110)
    print("wrote", OUT / "quiet_recovery_link.png")


if __name__ == "__main__":
    main()
