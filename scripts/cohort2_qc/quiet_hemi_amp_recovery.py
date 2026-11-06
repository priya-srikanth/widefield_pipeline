"""Quiet-event tests that need no imaging re-load: T8 ipsi vs contra, T11 amplitude / epileptiform, T14-T15 recovery link
(cohort1, 2026-11-01). Test numbers = docs/status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md section 4.

T8  IPSI vs CONTRA: face-regressed event rate (k = 4, up + down) per hemisphere (quiet_spikes_v2_fr session table);
    change vs the animal's August pre in log10(rate + 0.2), ipsilesional (left) vs contralesional (right); mixed model
    with a hemisphere x epoch interaction. A loop (ipsilesional thalamus) origin predicts ipsi > contra.
T11 AMPLITUDE / EPILEPTIFORM (Steinmetz et al. 2017: GCaMP6 interictal events are > ~10 % dF/F and cortex-wide): v2
    raw events (sessions/*.npz, heights in dF/F %), engaged quiet, k = 4: height distribution, events > 10 %, and the
    share of orofacial events with a limb event within +/-0.1 s (synchrony proxy; limb cortex is not orofacial).
T14 RECOVERY LINK: within animal over post-stroke sessions, face-regressed orofacial rate / pre vs far-contra hit rate /
    pre (recovery_timing.series), raw and with log(day) removed from both; permutation p (sessions shuffled within
    animal, 5000 draws).
T15 DEFICIT ORDERING (n = 4, descriptive): acute rate ratio vs half-recovery day and lesion size (FINDINGS 4c).

    python scripts/cohort2_qc/quiet_hemi_amp_recovery.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_spikes_v2 import V2  # noqa: E402
from wfield_local import config  # noqa: E402

OUT = V2 / "hemi_amp_recovery"
FRT = V2 / "fr_regressed_own" / "quiet_events_fr_sessions.csv"
EP = ["pre", "acute", "subacute", "chronic"]
K = 4.0
# FINDINGS 4c: half-recovery day (far-contra hit rate) and lesion % caudoputamen
HALF_DAY = {"PS92": 4.7, "PS93": 3.9, "PS94": 7.0, "PS95": 7.9}
LESION = {"PS92": 46.7, "PS93": 45.0, "PS94": 67.7, "PS95": 81.9}


def t8():
    from wfield_local import mixed_models as mm
    S = pd.read_csv(FRT)
    S = S[~S.label.map(config.early_pre)]
    L = []
    for g in ("orofacial", "limb"):
        for h, side in (("left", "ipsi"), ("right", "contra")):
            c = f"{g}_{h}|res_both"
            L.append(S[["label", "animal", "epoch"]].assign(group=g, side=side, y=np.log10(S[c] + 0.2)))
    L = pd.concat(L)
    pre = L[L.epoch == "pre"].groupby(["animal", "group", "side"]).y.mean()
    L["d"] = L.y - [pre.get((a, g, s), np.nan) for a, g, s in zip(L.animal, L.group, L.side)]
    tab = L[L.epoch != "pre"].pivot_table(index=["group", "animal"], columns=["epoch", "side"], values="d",
                                          aggfunc="mean").reindex(columns=pd.MultiIndex.from_product(
                                              [EP[1:], ["ipsi", "contra"]]))
    print("T8 change vs pre, log10(rate + 0.2), face-regressed, per animal (ipsi | contra):\n", tab.round(2).to_string())
    W = L.pivot_table(index=["label", "animal", "epoch", "group"], columns="side", values="y").reset_index()
    W["y"] = W.ipsi - W.contra                                                   # session ipsi - contra
    W = mm.add_time_covariates(W[["label", "group", "y"]], august_pre=True).dropna(subset=["y"])
    T = mm.lmm_table(W.assign(measure="ipsi_minus_contra"), ["measure", "group"], models=("epoch",),
                     print_animals=True, fdr_by=("measure",))
    T.to_csv(OUT / "t8_lmm_ipsi_minus_contra.csv", index=False)
    print(T[T.term != "Intercept"][["group", "term", "coef", "se", "p", "q"]].round(4).to_string())
    L.to_csv(OUT / "t8_sessions.csv", index=False)
    return L, T


def t11(write=True):
    """write=False: read the stored v2 event files only (no detection) and return the tables without writing them."""
    from quiet_area_spikes import columns
    from quiet_fluct import animal_sd
    from wfield_local import epochs
    cols = columns()
    sd = animal_sd()
    gi, li = cols.index("orofacial"), cols.index("limb")
    rows, H = [], []
    for f in sorted((V2 / "sessions").glob("*.npz")):
        lab, ep = f.stem, epochs.epoch_of(f.stem)
        if ep is None or config.early_pre(lab) or lab[:4] not in sd:
            continue
        z = np.load(f)
        fs = float(z["fs"])
        mins = float(z["n_quiet_eng"]) / fs / 60
        r = dict(label=lab, animal=lab[:4], epoch=ep)
        for kind in ("pos", "neg"):
            a = z[kind]
            o = a[(a[:, 0] == gi) & (a[:, 3] == 1) & (a[:, 2] > K * sd[lab[:4]][gi])]
            lm = a[(a[:, 0] == li) & (a[:, 3] == 1) & (a[:, 2] > K * sd[lab[:4]][li])]
            r[f"n_{kind}"] = len(o)
            r[f"rate_gt10_{kind}"] = float((o[:, 2] > 10).sum() / mins) if mins else np.nan
            r[f"max_h_{kind}"] = float(o[:, 2].max()) if len(o) else np.nan
            r[f"med_h_{kind}"] = float(np.median(o[:, 2])) if len(o) else np.nan
            if len(o) and len(lm):
                d = np.abs(o[:, 1][:, None] - lm[:, 1][None, :]).min(1)
                r[f"limb_coinc_{kind}"] = float(np.mean(d <= 0.1 * fs))
            else:
                r[f"limb_coinc_{kind}"] = np.nan
            H += [(lab[:4], ep, kind, h) for h in o[:, 2]]
        rows.append(r)
    S = pd.DataFrame(rows)
    Hd = pd.DataFrame(H, columns=["animal", "epoch", "kind", "h"])
    if not write:
        return S, Hd
    S.to_csv(OUT / "t11_sessions.csv", index=False)
    for m in ("med_h_pos", "max_h_pos", "rate_gt10_pos", "rate_gt10_neg", "limb_coinc_pos", "limb_coinc_neg"):
        print(f"\nT11 {m}, animal x epoch median:\n",
              S.pivot_table(index="animal", columns="epoch", values=m, aggfunc="median").reindex(columns=EP)
              .round(3).to_string())
    print("\nT11 share of orofacial events > 10 % dF/F (all events pooled):\n",
          Hd.assign(big=Hd.h > 10).groupby(["animal", "epoch"]).big.mean().unstack().reindex(columns=EP)
          .round(4).to_string())
    return S, Hd


def t14():
    from recovery_timing import series
    S = pd.read_csv(FRT)
    S = S[~S.label.map(config.early_pre)]
    S["rate"] = S["orofacial|res_both"]
    pre = S[S.epoch == "pre"].groupby("animal").rate.mean()
    S["rate_rel"] = S.rate / S.animal.map(pre)
    B = series("lick")[["label", "beh"]]
    G = S.merge(B, on="label").dropna(subset=["beh", "rate_rel"])
    G = G[(G.epoch != "pre") & (G.day > 0)].copy()
    rng = np.random.default_rng(0)

    def resid(y, d):
        X = np.column_stack([np.ones(len(y)), d])
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    out = []
    for kind in ("raw", "detrended"):
        rs, perm = {}, []
        for an, g in G.groupby("animal"):
            if len(g) < 5:
                continue
            b, y = g.beh.to_numpy(float), np.log(g.rate_rel.to_numpy(float))
            if kind == "detrended":
                d = np.log(g.day.to_numpy(float))
                b, y = resid(b, d), resid(y, d)
            rs[an] = float(np.corrcoef(b, y)[0, 1])
            perm.append((b, y))
        obs = np.mean(list(rs.values()))
        null = np.array([np.mean([np.corrcoef(b, rng.permutation(y))[0, 1] for b, y in perm]) for _ in range(5000)])
        p = float(np.mean(np.abs(null) >= abs(obs)))
        out.append(dict(kind=kind, mean_r=obs, p_perm=p, **rs))
    T = pd.DataFrame(out)
    print("\nT14 within-animal r (far-contra hit rate / pre vs log orofacial rate / pre), post sessions:\n",
          T.round(3).to_string())
    T.to_csv(OUT / "t14_recovery_link.csv", index=False)
    G.to_csv(OUT / "t14_sessions.csv", index=False)
    acute = S[S.epoch == "acute"].groupby("animal").rate_rel.mean()
    D = pd.DataFrame(dict(acute_rate_x_pre=acute, half_recovery_day=pd.Series(HALF_DAY),
                          lesion_pct=pd.Series(LESION)))
    from scipy.stats import spearmanr
    print("\nT15 per animal (n = 4, descriptive):\n", D.round(2).to_string())
    for c in ("half_recovery_day", "lesion_pct"):
        rho = spearmanr(D.acute_rate_x_pre, D[c]).statistic
        print(f"  Spearman acute rate vs {c}: rho {rho:+.2f} (n = 4: only |rho| = 1 reaches p .083 two-sided)")
    D.to_csv(OUT / "t15_animals.csv")
    return G, T, D


def figure(L, T8, S11, H11, G14, T14, D15, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import cat_points as cp
    from wfield_local import sig_marks
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    animals = sorted(L.animal.unique())
    plotted = {}
    # T8 (Priya 11-03 rules): bar = mean of the animal means; the SESSION changes behind it as faded open markers (shape
    # = animal, colour = epoch); per-animal means small filled; hemisphere = lateral position (ipsi left, contra right)
    for j, g in enumerate(("orofacial", "limb")):
        a = ax[0, j]
        for k, ep in enumerate(EP[1:]):
            for side, dx in (("ipsi", -0.17), ("contra", 0.17)):
                G = L[(L.group == g) & (L.side == side) & (L.epoch == ep)]
                m = G.groupby("animal").d.mean()
                plotted[(g, ep, side)] = (m.to_dict(), float(m.mean()))
                a.bar(k + dx, m.mean(), width=0.3, color=EPOCH_LINE[ep], alpha=0.5 if side == "ipsi" else 0.22,
                      zorder=1)
                cp.session_dots(a, k + dx, G.d, G.animal, EPOCH_LINE[ep], jitter=0.07, seed=10 * k + (side == "ipsi"))
                for ia, (an, v) in enumerate(m.items()):
                    a.plot(k + dx - 0.075 + 0.05 * ia, v, cp.animal_marker(an), ls="none", color=EPOCH_LINE[ep],
                           mec="w", mew=0.5, ms=5.5, zorder=4)
            t = T8[(T8.group == g) & T8.term.str.contains(ep)]
            if len(t) and float(t.q.iloc[0]) < 0.05:
                sig_marks.star_top(a, k, "**" if float(t.q.iloc[0]) < 0.01 else "*")
        a.axhline(0, color="0.5", lw=0.8)
        a.set_xticks([k + dx for k in range(3) for dx in (-0.17, 0.17)], ["ipsi", "contra"] * 3, fontsize=7)
        for k, ep in enumerate(EP[1:]):
            a.text(k, -0.07, ep, transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        a.set_ylabel("change vs the animal's pre, log10(rate + 0.2)")
        a.set_title(f"T8 {g}: change vs pre, log10(rate + 0.2)\nbar = mean of animals (dark = ipsilesional, light = "
                    "contralesional); small filled = animals,\nfaded open = sessions; stars = ipsi - contra vs pre "
                    "(LMM, FDR)", fontsize=8)
    ax[0, 0].legend(handles=cp.animal_handles(animals), fontsize=7, loc="lower left", frameon=False)
    a = ax[0, 2]
    for ep in EP:
        h = H11[(H11.kind == "pos") & (H11.epoch == ep)].h
        if len(h):
            a.hist(h, bins=np.linspace(0, 12, 49), histtype="step", density=True, color=EPOCH_LINE[ep], lw=1.6,
                   label=f"{ep} (n {len(h)}, > 10 %: {int((h > 10).sum())})")
    a.axvline(10, color="k", ls="--", lw=0.8)
    a.set_yscale("log")
    a.set_xlabel("orofacial UP event height, dF/F %")
    a.set_title("T11 height distribution (engaged quiet, k = 4); dashed = 10 %,\nthe size of GCaMP6 interictal events "
                "(Steinmetz 2017)", fontsize=8)
    a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    a = ax[1, 0]
    for k, ep in enumerate(EP):
        G = S11[S11.epoch == ep]
        for an, gg in G.groupby("animal"):
            a.plot(np.full(len(gg), k + (animals.index(an) - 1.5) * 0.08), gg.limb_coinc_pos, ANIMAL_MARKER[an],
                   mfc="none", mec=EPOCH_LINE[ep], ms=4)
    a.set_xticks(range(4))
    a.set_xticklabels(EP)
    a.set_ylabel("share of orofacial UP events with a limb event\nwithin 0.1 s")
    a.set_title("T11 synchrony proxy\n(cortex-wide events would be shared)", fontsize=8)
    a = ax[1, 1]
    for an, g in G14.groupby("animal"):
        a.plot(g.beh, g.rate_rel, ANIMAL_MARKER[an], mfc="none", mec="k", ms=6, label=an)
    a.set_yscale("log")
    a.set_xlabel("far-contra hit rate / pre")
    a.set_ylabel("orofacial event rate / pre (face regressed)")
    t = T14.set_index("kind")
    a.set_title(f"T14 post sessions: within-animal r raw {t.loc['raw', 'mean_r']:+.2f} (p {t.loc['raw', 'p_perm']:.3f}),"
                f"\nlog-day removed {t.loc['detrended', 'mean_r']:+.2f} (p {t.loc['detrended', 'p_perm']:.3f})",
                fontsize=8)
    a.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    a = ax[1, 2]
    for an, r in D15.iterrows():
        a.plot(r.half_recovery_day, r.acute_rate_x_pre, ANIMAL_MARKER[an], mfc="none", mec="k", ms=9)
        a.annotate(f"{an}\nlesion {r.lesion_pct:.0f} %", (r.half_recovery_day, r.acute_rate_x_pre), fontsize=7,
                   xytext=(4, 4), textcoords="offset points")
    a.set_xlabel("half-recovery day (far-contra hit rate)")
    a.set_ylabel("acute orofacial event rate / pre")
    a.set_title("T15 per animal (n = 4, descriptive)", fontsize=8)
    a.set_xlim(a.get_xlim()[0] - 0.3, a.get_xlim()[1] + 0.8)
    a.set_ylim(a.get_ylim()[0], a.get_ylim()[1] + 0.4)
    sig_marks.reserve_headroom(fig)
    fig.suptitle("Quiet-period events: T8 ipsilesional vs contralesional, T11 amplitude / epileptiform check, T14-T15 link "
                 "to recovery (cohort1; face-regressed or raw v2 events > 4 x pre SD, engaged quiet)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3.0, h_pad=3.0)
    out = OUT / "quiet_hemi_amp_recovery.png" if out is None else out
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)
    return plotted


def plot_only():
    """Re-render from the STORED tables (t8 / t14 / t15 CSVs; T11 heights read from the stored v2 event files, the same
    selection as t11, nothing written) -> quiet_hemi_amp_recovery_nolines.png; check plotted numbers vs the tables."""
    L = pd.read_csv(OUT / "t8_sessions.csv")
    T8 = pd.read_csv(OUT / "t8_lmm_ipsi_minus_contra.csv")
    S11, H11 = t11(write=False)
    S11s = pd.read_csv(OUT / "t11_sessions.csv")
    pd.testing.assert_frame_equal(S11.reset_index(drop=True), S11s, check_dtype=False)
    G14 = pd.read_csv(OUT / "t14_sessions.csv")
    T14 = pd.read_csv(OUT / "t14_recovery_link.csv")
    D15 = pd.read_csv(OUT / "t15_animals.csv", index_col=0)
    P = figure(L, T8, S11, H11, G14, T14, D15, out=OUT / "quiet_hemi_amp_recovery_nolines.png")
    for (g, ep, side), (am, mu) in P.items():
        W = L[(L.group == g) & (L.side == side) & (L.epoch == ep)].pivot_table(index="animal", values="d",
                                                                                 aggfunc="mean").d
        assert np.allclose([am[a] for a in W.index], W.to_numpy()) and np.isclose(mu, W.mean()), (g, ep, side)
    print(f"T8: {len(P)} bars + animal means == t8_sessions.csv; T11 sessions recomputed from stored events == "
          "t11_sessions.csv")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="re-render from the stored tables -> *_nolines.png")
    if ap.parse_args().plot_only:
        plot_only()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    L, T8 = t8()
    S11, H11 = t11()
    G14, T14, D15 = t14()
    figure(L, T8, S11, H11, G14, T14, D15)
    (OUT / "README.txt").write_text(__doc__)


if __name__ == "__main__":
    main()
