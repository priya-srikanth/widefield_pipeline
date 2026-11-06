"""T20b (EXPLORATORY, PROVISIONAL): T20-type synchronisation laterality with a fast band free of the 7-11 Hz artefact.
(cohort1 PS92-95, 2026-11-02; quiet-events handoff NEXT item 3.)

T20's index = log10(PSD 1-4 Hz / PSD 8-15 Hz) of the quiet engaged trace; its acute orofacial ipsi > contra rise was
WITHDRAWN by T22 because the 8-15 Hz denominator is dominated by a NON-CALCIUM 7-11 Hz peak (equally in 415 nm and raw
470; cardiac / brain motion) that is lower on the lesioned side. Here the denominator avoids that peak:
    I_8_15  = log10(P1-4 / P8-15)   T20 as published (reference, recomputed on the same frames)
    I_4_8   = log10(P1-4 / P4-8)    the requested band (its top 1 Hz still touches the 7-11 Hz peak)
    I_4_7   = log10(P1-4 / P4-7)    sensitivity: entirely below the peak
    I_4_8fl = log10((P1-4 - fl) / (P4-8 - fl)), fl = mean PSD 12-15 Hz (white-floor estimate; T22 variant D)
    P415_4_8 = log10 415-nm PSD 4-8 Hz (non-calcium control: does the NEW band carry a lesion-side 415 asymmetry?)
Spectra: quiet_fluct sessions (Welch 4-s, dF/F %, corrected 470) and quiet_hemo_check sessions (raw 415 on the same quiet
frames); both follow QS_LICK_SOURCE -> run with QS_LICK_SOURCE=daq+lp (LP-based quiet frames, v2_quiet_lp/).
Contrast = ipsilesional (image left, `_left`) minus contralesional, orofacial and limb; change vs the animal's AUGUST pre.
'Fraction surviving' = acute LMM coefficient of the orofacial contrast under each band / that under 8-15 Hz (and per
animal: animal-mean acute change / the same under 8-15). Orofacial-minus-limb laterality tests specificity (T22: limb
showed the same shift). Per-animal tables first (CLAUDE.md rule 8); `mixed_models.lmm_table` epoch model vs August
pre, FDR within measure. Cohort1 only.

    set QS_LICK_SOURCE=daq+lp
    python scripts/cohort2_qc/quiet_sync_4to8.py
Output: <V2>/sync_4to8/  (sessions.csv, contrasts_sessions.csv, lmm.csv, fraction_surviving.csv, quiet_sync_4to8.png)
"""
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_fluct import FL, TRACE_COLS, load  # noqa: E402
from quiet_spikes_v2 import LICK_SOURCE  # noqa: E402

V2Q = FL.parent
OUT = V2Q / "sync_4to8"
H415 = V2Q / "hemo415" / "sessions"
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
EP = ["pre", "acute", "subacute", "chronic"]
B_LO, B_HI, B_48, B_47, B_FL = (1.0, 4.0), (8.0, 15.0), (4.0, 8.0), (4.0, 7.0), (12.0, 15.0)
MEASURES = ("I_8_15", "I_4_8", "I_4_7", "I_4_8fl", "P415_4_8")


def bandmean(fr, p, b):
    return float(np.nanmean(p[(fr >= b[0]) & (fr < b[1])]))


def session_table():
    from wfield_local import epochs
    D = load()
    rows = []
    for k, d in D.items():
        if k[:4] not in ANIMALS:
            continue
        fr, P = d["freqs"], d["psd"]
        h = None
        if (H415 / f"{k}.npz").exists():
            zh = np.load(H415 / f"{k}.npz", allow_pickle=True)
            h = {str(r): i for i, r in enumerate(zh["rois"])}, zh["psd_raw415"], zh["freqs"]
        for j, c in enumerate(TRACE_COLS):
            lo, hi, b48, b47, fl = (bandmean(fr, P[j], b) for b in (B_LO, B_HI, B_48, B_47, B_FL))
            r = dict(label=k, animal=k[:4], epoch=epochs.epoch_of(k), column=c,
                     I_8_15=np.log10(lo / hi), I_4_8=np.log10(lo / b48), I_4_7=np.log10(lo / b47),
                     I_4_8fl=np.log10((lo - fl) / (b48 - fl)) if (lo > fl and b48 > fl) else np.nan,
                     P415_4_8=np.nan)
            if h is not None and c in h[0]:
                r["P415_4_8"] = np.log10(bandmean(h[2], h[1][h[0][c]], B_48))
            rows.append(r)
    return pd.DataFrame(rows)


def contrasts(S):
    V = S.pivot_table(index=["label", "animal", "epoch"], columns="column", values=list(MEASURES))
    K = pd.DataFrame(index=V.index)
    for m in MEASURES:
        for g in ("orofacial", "limb"):
            K[f"{m}|{g}_ipsi_minus_contra"] = V[(m, f"{g}_left")] - V[(m, f"{g}_right")]
        K[f"{m}|orofacial_minus_limb_laterality"] = K[f"{m}|orofacial_ipsi_minus_contra"] - K[f"{m}|limb_ipsi_minus_contra"]
        K[f"{m}|orofacial"] = V[(m, "orofacial")]
        K[f"{m}|cortex"] = V[(m, "cortex")]
    K = K.reset_index()
    cols = [c for c in K.columns if "|" in c]
    pre = K[K.epoch == "pre"].groupby("animal")[cols].mean()
    for c in cols:
        K[f"d_{c}"] = K[c].to_numpy() - pre[c].reindex(K.animal).to_numpy()
    return K, cols


def lmm(K, cols):
    from wfield_local import mixed_models as mm
    out = []
    for m in MEASURES:
        cc = [c for c in cols if c.startswith(m + "|")]
        L = pd.concat([K[["label"]].assign(measure=m, column=c.split("|")[1], y=K[c].to_numpy()) for c in cc])
        L = mm.add_time_covariates(L, august_pre=True)
        out.append(mm.lmm_table(L.dropna(subset=["y"]), ["measure", "column"], models=("epoch",), print_animals=False,
                                fdr_by=("measure",)))
    return pd.concat(out)


def per_animal(K, col):
    return K[K.epoch != "pre"].pivot_table(index="animal", columns="epoch", values=col, aggfunc="mean") \
        .reindex(columns=EP[1:])


def figure(K, T, F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from wfield_local import sig_marks
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    panels = [("I_8_15", "A  1-4 / 8-15 Hz\n(T20 as published)"), ("I_4_8", "B  1-4 / 4-8 Hz\n(requested band)"),
              ("I_4_7", "C  1-4 / 4-7 Hz\n(entirely below 7 Hz)"), ("I_4_8fl", "D  1-4 / 4-8 Hz,\nfloor-subtracted"),
              ("P415_4_8", "E  415 nm power 4-8 Hz\n(non-calcium control)")]
    rows = [("orofacial_ipsi_minus_contra", "orofacial ipsi - contra"), ("limb_ipsi_minus_contra", "limb ipsi - contra"),
            ("orofacial_minus_limb_laterality", "orofacial minus limb\nlaterality")]
    fig, ax = plt.subplots(len(rows), len(panels), figsize=(17, 11.5), sharey="row")
    for i, (col, rlab) in enumerate(rows):
        for j, (m, title) in enumerate(panels):
            a = ax[i, j]
            c = f"d_{m}|{col}"
            G = K[K.epoch != "pre"]
            for k, ep in enumerate(EP[1:]):
                g = G[G.epoch == ep]
                for an, x in g.groupby("animal"):
                    a.plot(np.full(len(x), k + (ANIMALS.index(an) - 1.5) * 0.09), x[c], ANIMAL_MARKER[an], mfc="none",
                           mec=EPOCH_LINE[ep], ms=4, alpha=0.75, ls="none")
                am = g.groupby("animal")[c].mean()
                a.bar(k, am.mean(), width=0.55, color=EPOCH_LINE[ep], alpha=0.3, zorder=0)
                t = T[(T.measure == m) & (T.column == col) & T.term.str.endswith(f"[T.{ep}]")]
                if len(t) and np.isfinite(t.q.iloc[0]) and float(t.q.iloc[0]) < 0.05:
                    q = float(t.q.iloc[0])
                    sig_marks.star_top(a, k, "***" if q < 0.001 else "**" if q < 0.01 else "*")
            a.axhline(0, color="0.5", lw=0.8)
            a.set_xticks(range(3))
            a.set_xticklabels(EP[1:], fontsize=8)
            if i == 0:
                a.set_title(title, fontsize=9)
            if j == 0:
                a.set_ylabel(f"{rlab}\nchange vs pre (log10)", fontsize=9)
    # fraction surviving, written under row 1 titles would crowd: put it in the suptitle block instead
    fo = F[(F.contrast == "orofacial_ipsi_minus_contra") & (F.animal == "LMM")].set_index("measure").frac_acute
    handles = [Line2D([], [], marker=mk, ls="", mfc="none", mec="0.3", label=an) for an, mk in ANIMAL_MARKER.items()
               if an in ANIMALS]
    handles += [Line2D([], [], marker="s", ls="", color=EPOCH_LINE[e], alpha=0.5, label=f"{e} (bar = mean of animals)")
                for e in EP[1:]]
    fig.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=7, frameon=False)
    sig_marks.reserve_headroom(fig)
    src = "LP-based quiet frames (DAQ contact + LP video licks)" if LICK_SOURCE == "daq+lp" else "DAQ-contact quiet frames"
    fig.suptitle("T20b (EXPLORATORY, PROVISIONAL n = 4): quiet-period synchronisation laterality with fast bands that avoid "
                 "the non-calcium 7-11 Hz peak (T22).\nSessions (shape = animal), colour = epoch; stars = LMM vs August pre, "
                 f"FDR within measure; {src}.\nAcute orofacial ipsi - contra kept vs 8-15 Hz (LMM coefficient ratio): "
                 f"4-8 Hz {fo.get('I_4_8', np.nan):.0%}, 4-7 Hz {fo.get('I_4_7', np.nan):.0%}, "
                 f"4-8 floor-subtracted {fo.get('I_4_8fl', np.nan):.0%}", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=2.0, w_pad=1.5)
    save_png_svg(fig, OUT / "quiet_sync_4to8.png", dpi=110)
    plt.close(fig)
    print("wrote", OUT / "quiet_sync_4to8.png")


def main():
    pd.set_option("display.width", 220)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"lick source {LICK_SOURCE}; spectra {FL}; 415 {H415}")
    S = session_table()
    S.to_csv(OUT / "sessions.csv", index=False)
    print("sessions:", S.label.nunique(), "| with 415:", S.dropna(subset=["P415_4_8"]).label.nunique(),
          "| per animal x epoch:\n", S.drop_duplicates("label").pivot_table(index="animal", columns="epoch",
                                                                           values="label", aggfunc="count")
          .reindex(columns=EP).to_string())
    K, cols = contrasts(S)
    K.to_csv(OUT / "contrasts_sessions.csv", index=False)
    print("\n==== per-animal change vs pre (rule 8), animal x epoch mean")
    for col in ("orofacial_ipsi_minus_contra", "limb_ipsi_minus_contra", "orofacial_minus_limb_laterality",
                "orofacial"):
        for m in MEASURES:
            print(f"\n-- d {m} | {col}\n" + per_animal(K, f"d_{m}|{col}").round(3).to_string())
    T = lmm(K, cols)
    T.to_csv(OUT / "lmm.csv", index=False)
    print("\n==== LMM (epoch vs August pre), FDR within measure")
    print(T[T.term != "pre_level"][["measure", "column", "term", "coef", "se", "p", "q", "n"]].round(4).to_string())
    fr = []
    for col in ("orofacial_ipsi_minus_contra", "limb_ipsi_minus_contra", "orofacial_minus_limb_laterality"):
        ref = T[(T.measure == "I_8_15") & (T.column == col) & (T.term == "C(epoch)[T.acute]")].coef
        refa = per_animal(K, f"d_I_8_15|{col}").acute
        for m in MEASURES[1:4]:
            t = T[(T.measure == m) & (T.column == col) & (T.term == "C(epoch)[T.acute]")]
            fr.append(dict(contrast=col, measure=m, animal="LMM", ref_8_15=float(ref.iloc[0]), coef=float(t.coef.iloc[0]),
                           q=float(t.q.iloc[0]), frac_acute=float(t.coef.iloc[0]) / float(ref.iloc[0])))
            am = per_animal(K, f"d_{m}|{col}").acute
            for an in ANIMALS:
                fr.append(dict(contrast=col, measure=m, animal=an, ref_8_15=float(refa.get(an, np.nan)),
                               coef=float(am.get(an, np.nan)), q=np.nan,
                               frac_acute=float(am.get(an, np.nan)) / float(refa.get(an, np.nan))))
    F = pd.DataFrame(fr)
    F.to_csv(OUT / "fraction_surviving.csv", index=False)
    print("\n==== fraction of the acute 8-15 Hz laterality kept (per animal = animal-mean acute change ratio)")
    print(F.round(3).to_string(index=False))
    figure(K, T, F)
    (OUT / "README.txt").write_text(__doc__)


if __name__ == "__main__":
    main()
