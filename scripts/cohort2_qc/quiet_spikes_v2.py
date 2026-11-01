"""Quiet-period cortical events v2: height detector, Priya's quiet rule, threshold chosen by sensitivity / specificity
(cohort1, 2026-11-01).

WHY v2. The example traces (quiet_spikes_examples.py) showed the v1 counts (quiet_area_spikes.py) were not measuring what
Priya saw: (1) restdock05 rest is ~1.5 s per trial and the big events sit OUTSIDE it (spout out, no licks: acute missed
trials); (2) v1's PROMINENCE is taken against the lowest flanking trough, so a peak after a dip (a swapped-frame dip,
a post-event undershoot) passes with a small rise. Priya 11-01: "rest for this particular analysis can include when the
spout is present or moving, as long as the animal isn't licking or running"; "1.5s before and 1.5s after with no lick or
running would be fine; no need to also have 2s in addition"; "explore different MAD thresholds to find what the best
threshold might be for sensitivity/specificity".

TRACES: quiet_area_spikes.session_traces (encoder inputs, Allen areas, dF/F %); 2-s running median removed AFTER any
injection. QUIET frame = no DAQ contact lick and no running bout within 1.5 s before or after; spout / cue ignored; no
swapped pair within 0.5 s. ENGAGED = before the quit (as v1); rates are reported for engaged quiet.
EVENT = a local peak (half-width 1.5 frames .. 0.5 s) inside quiet whose HEIGHT above the median of the preceding
1.0 .. 0.2 s exceeds k x the animal's AUGUST-pre robust SD (1.4826 x MAD of the detrended trace over engaged quiet frames,
median over pre sessions); peaks within 0.3 s merged (largest kept). Heights of every candidate are stored, so any k is
applied in the summary.
POLARITY: the same detector on the sign-flipped trace (-x) gives NEGATIVE deflections. Priya 11-01: "the events we see
have both positive and negative deflections" -- so -x is NOT a clean false-positive estimate; rates are reported for
positive, negative and either polarity ("both" = pos + neg).
THRESHOLD CHOICE (Priya 11-01: "what if we see if we can maximize ROC AUC for pre vs acute"): per k, the session-level
ROC AUC of the event rate for pre vs acute sessions; k chosen LEAVE-ONE-ANIMAL-OUT (best k on three animals, AUC scored
on the held-out fourth), so the reported held-out AUC is not inflated by the choice. The LMM at the chosen k still
re-uses the data that chose it -- its ACUTE contrast is optimistic; subacute / chronic were not optimised.
SENSITIVITY: synthetic events (difference of exponentials, rise 40 ms, decay 300 ms; half-width ~0.25 s, about the
GCaMP6s events in the scan) of 1, 2, 3, 4, 6 % added to the RAW trace at 40 random engaged-quiet frames >= 3 s apart, then
detrended and detected as above; hit = an event peak 0 .. 0.25 s after the injection onset.
STATISTICS at the chosen k: per-animal epoch table first (rule 8); `mixed_models.lmm_table` epoch + time models (August
pre; log10(rate + 0.2)); pooled figure = sessions as open markers (shape = animal, colour = epoch), pooled mean + nested
bootstrap CI in the epoch colour, stars (LMM epoch vs pre, FDR) in the top band.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_spikes_v2.py [--jobs 6] [--summarise] [--k 5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import DETREND_S, EP, EVENTS, MAX_HW_S, MIN_HW_FR, OUT, columns, mad, session_traces  # noqa: E402

V2 = OUT / "v2_quiet"
BUF_S, BASE_S, MERGE_S, BAD_S = 1.5, (1.0, 0.2), 0.3, 0.5
AMPS, N_INJ, INJ_GAP_S, HIT_S = (1.0, 2.0, 3.0, 4.0, 6.0), 40, 3.0, 0.25
INJ_COLS = ["orofacial_left", "orofacial_right", "limb_left", "limb_right"]
KS = np.round(np.arange(2.0, 10.01, 0.5), 2)
SHOW = INJ_COLS
KNOTE = ["median of the leave-one-animal-out choices"]          # set in summarise(); titles say how k was set
AREA_SHOW = ["SSp-m_left", "SSp-m_right", "tjM1_left", "tjM1_right", "ALM_left", "ALM_right", "SSp-n_left",
             "SSp-n_right", "SSp-ul_left", "SSp-ul_right", "SSp-ll_left", "SSp-ll_right"]


def kernel(fs):
    t = np.arange(int(2 * fs)) / fs
    k = np.exp(-t / 0.30) - np.exp(-t / 0.04)
    return k / k.max()


def detrend(Y, fs):
    from scipy.ndimage import median_filter
    w = int(round(DETREND_S * fs)) | 1
    return Y - median_filter(Y, (1, w) if Y.ndim == 2 else w, mode="nearest")


def detect(y, ok, fs):
    """Peaks of detrended y in frames `ok` -> (frames, heights above the preceding 1.0 .. 0.2 s), merged within 0.3 s."""
    from scipy.signal import find_peaks
    y = np.nan_to_num(y)
    pk, _ = find_peaks(y)
    b0, b1 = int(round(BASE_S[0] * fs)), int(round(BASE_S[1] * fs))
    pk = pk[(pk >= b0) & ok[pk]]
    if not len(pk):
        return pk, np.zeros(0)
    base = np.array([np.median(y[p - b0:p - b1]) for p in pk])
    h = y[pk] - base
    # half-width at HALF THE HEIGHT above this baseline (find_peaks' width is taken from the deepest flanking trough, so
    # a big event on a slow fluctuation read as > 0.5 s and was dropped: injected 6% events found less often than 3%)
    hw = np.zeros(len(pk))
    lim = int(MAX_HW_S * fs) + 2
    for j, p in enumerate(pk):
        above = y[max(0, p - lim):p + lim + 1] > base[j] + h[j] / 2
        c = p - max(0, p - lim)
        lo = c - np.argmin(above[c::-1]) if not above[c::-1].all() else 0
        hi = c + np.argmin(above[c:]) if not above[c:].all() else len(above)
        hw[j] = hi - lo
    sel = (h > 0) & (hw >= MIN_HW_FR) & (hw <= MAX_HW_S * fs)
    pk, h = pk[sel], h[sel]
    if not len(pk):
        return pk, np.zeros(0)
    o = np.argsort(-h)
    keep, m = [], int(round(MERGE_S * fs))
    taken = np.zeros(len(y), bool)
    for i in o:
        p = pk[i]
        if not taken[max(0, p - m):p + m + 1].any():
            keep.append(i)
            taken[p] = True
    keep = np.sort(np.array(keep, int))
    return pk[keep], h[keep]


def busy_frames(label, ft, fs):
    """(busy = within BUF_S of a DAQ contact lick or a running bout, badw = within BAD_S of a swapped pair) on frames ft.
    Shared with quiet_fluct.py so every quiet analysis uses the identical frames."""
    from wfield_local import config
    T = len(ft)
    animal, mmdd = label.split("_")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    fsd = float(z["fs"])
    busy = np.zeros(T, bool)
    for x in z["lick_onsets"] / fsd:
        busy[np.searchsorted(ft, x - BUF_S):np.searchsorted(ft, x + BUF_S)] = True
    for r0, r1 in np.c_[z["running_starts"], z["running_stops"]] / fsd:
        busy[np.searchsorted(ft, r0 - BUF_S):np.searchsorted(ft, r1 + BUF_S)] = True
    s = next(x for x in config.load_sessions() if x["label"] == label)
    sw = Path(s["mc"]) / "wfield_local_results" / "swapped_pairs.npz"
    bad = np.zeros(T, bool)
    if sw.exists():
        b = np.load(sw)["bad"][:T].astype(bool)
        bad[:len(b)] = b
    badw = np.convolve(bad.astype(float), np.ones(2 * int(BAD_S * fs) + 1), "same") > 0
    return busy, badw


def extract(label):
    f = V2 / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    Xr, ft, fs, rest, engaged, quit_s = session_traces(label, detrend=False)
    T = Xr.shape[1]
    busy, badw = busy_frames(label, ft, fs)
    finite = np.isfinite(Xr).all(0)
    quiet = ~busy & ~badw & finite
    qe = quiet & engaged
    cols = columns()
    Xd = detrend(Xr, fs)
    sd_q = mad(Xd[:, qe]) if qe.sum() > 100 else np.full(len(cols), np.nan)
    pos, neg = [], []
    for i, c in enumerate(cols):
        if c == "cortex" or not np.isfinite(Xd[i]).all():
            continue
        for sign, store in ((1, pos), (-1, neg)):
            pk, h = detect(sign * Xd[i], quiet, fs)
            store += [(i, int(p), float(v), int(engaged[p]), int(rest[p])) for p, v in zip(pk, h)]
    # injections (same onsets for every amplitude and column)
    rng = np.random.default_rng(abs(hash(label)) % (2 ** 32))
    cand = np.flatnonzero(qe)
    cand = cand[(cand > 2 * fs) & (cand < T - 3 * fs)]
    on = []
    for c0 in rng.permutation(cand):
        if all(abs(c0 - o) > INJ_GAP_S * fs for o in on):
            on.append(int(c0))
        if len(on) >= N_INJ:
            break
    on = np.array(sorted(on), int)
    k = kernel(fs)
    inj = []
    for c in INJ_COLS:
        i = cols.index(c)
        for A in AMPS:
            y = Xr[i].copy()
            for o in on:
                y[o:o + len(k)] += A * k[:max(0, min(len(k), T - o))]
            yd = detrend(y, fs)
            pk, h = detect(yd, quiet, fs)
            for o in on:
                j = np.flatnonzero((pk >= o) & (pk <= o + HIT_S * fs))
                inj.append((i, A, int(o), float(h[j].max()) if len(j) else np.nan))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, pos=np.array(pos, float).reshape(-1, 5), neg=np.array(neg, float).reshape(-1, 5),
                        inj=np.array(inj, float).reshape(-1, 4), sd_q=sd_q, fs=fs, cols=np.array(cols),
                        n_quiet=int(quiet.sum()), n_quiet_eng=int(qe.sum()), n_rest_eng=int((rest & engaged).sum()),
                        quit_s=quit_s, total_min=T / fs / 60)
    return label


def load_all():
    from wfield_local import epochs
    D = {}
    for f in sorted((V2 / "sessions").glob("*.npz")):
        ep = epochs.epoch_of(f.stem)
        if ep is None or (ep == "pre" and f.stem[-4:] < "0806"):
            continue
        D[f.stem] = dict(np.load(f), epoch=ep, day=epochs.days_since_stroke(f.stem))
    sd = {}
    for an in sorted({k[:4] for k in D}):
        sd[an] = np.nanmedian([d["sd_q"] for k, d in D.items() if k.startswith(an) and d["epoch"] == "pre"], 0)
    return D, sd


def threshold_table(D, sd):
    """Long table: session x column x k -> n_pos, n_neg (engaged quiet), quiet minutes; + injection hits."""
    import pandas as pd
    cols = columns()
    rows, irows = [], []
    for k_, d in D.items():
        an, mins = k_[:4], d["n_quiet_eng"] / float(d["fs"]) / 60
        for c in SHOW + ["orofacial", "limb"]:
            i = cols.index(c)
            for kind in ("pos", "neg"):
                a = d[kind]
                h = a[(a[:, 0] == i) & (a[:, 3] == 1), 2]
                for K in KS:
                    rows.append(dict(label=k_, animal=an, epoch=d["epoch"], column=c, kind=kind, k=K,
                                     n=int((h > K * sd[an][i]).sum()), quiet_min=mins))
        inj = d["inj"]
        for c in INJ_COLS:
            i = cols.index(c)
            for A in AMPS:
                h = inj[(inj[:, 0] == i) & (inj[:, 1] == A), 3]
                for K in KS:
                    irows.append(dict(label=k_, animal=an, epoch=d["epoch"], column=c, amp=A,
                                      amp_sd=A / sd[an][i], k=K, sens=float(np.mean(np.nan_to_num(h, nan=-1) > K * sd[an][i]))))
    return pd.DataFrame(rows), pd.DataFrame(irows)


def auc_tables(R):
    """Session rates per (column, kind, k) -> AUC pre vs acute; leave-one-animal-out choice of k per (column, kind)."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    W = R.pivot_table(index=["label", "animal", "epoch", "column", "k", "quiet_min"], columns="kind",
                      values="n").reset_index()
    W["both"] = W["pos"] + W["neg"]
    W = W[W.epoch.isin(["pre", "acute"])]
    rows, loao = [], []
    for (c, K), G in W.groupby(["column", "k"]):
        y = (G.epoch == "acute").to_numpy()
        for kind in ("pos", "neg", "both"):
            rows.append(dict(column=c, k=K, kind=kind, auc=roc_auc_score(y, G[kind] / G.quiet_min)))
    A = pd.DataFrame(rows)
    for (c, kind), _ in A.groupby(["column", "kind"]):
        for held in sorted(W.animal.unique()):
            tr = W[(W.column == c) & (W.animal != held)]
            te = W[(W.column == c) & (W.animal == held)]
            auc_tr = {K: roc_auc_score(g.epoch == "acute", g[kind] / g.quiet_min) for K, g in tr.groupby("k")}
            kb = max(auc_tr, key=lambda K: (auc_tr[K], -K))
            g = te[te.k == kb]
            ok = g.epoch.nunique() == 2
            loao.append(dict(column=c, kind=kind, held_out=held, k_chosen=kb, auc_train=auc_tr[kb],
                             auc_held=roc_auc_score(g.epoch == "acute", g[kind] / g.quiet_min) if ok else np.nan))
    return A, pd.DataFrame(loao)


def threshold_figure(R, I, A, ks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE
    fig, ax = plt.subplots(3, len(SHOW), figsize=(4.0 * len(SHOW), 9.5), sharex=True)
    for j, c in enumerate(SHOW):
        G = R[R.column == c]
        for ep in EP:
            E = G[G.epoch == ep]
            if E.empty:
                continue
            tot = E.groupby(["kind", "k"]).n.sum().unstack("kind")
            mins = E.drop_duplicates("label").quiet_min.sum()
            ax[0, j].semilogy(tot.index, tot["pos"] / mins, color=EPOCH_LINE[ep], lw=2, label=f"{ep} positive")
            ax[0, j].semilogy(tot.index, tot["neg"] / mins, color=EPOCH_LINE[ep], lw=1, ls="--")
        for kind, ls in (("pos", "-"), ("neg", "--"), ("both", ":")):
            B = A[(A.column == c) & (A.kind == kind)]
            ax[1, j].plot(B.k, B.auc, color="k", ls=ls, lw=1.5, label=kind if j == 0 else None)
        ax[1, j].axhline(0.5, color="0.7", lw=0.6)
        H = I[I.column == c]
        for A_, ls in zip(AMPS, (":", "-.", "--", "-", "-")):
            for ep, lw in (("pre", 2.0), ("acute", 1.0)):
                S = H[(H.amp == A_) & (H.epoch == ep)].groupby("k").sens.mean()
                ax[2, j].plot(S.index, S, color=EPOCH_LINE[ep], ls=ls, lw=lw, label=f"{A_:g}% {ep}" if j == 0 else None)
        for r in range(3):
            ax[r, j].axvline(ks[c.split("_")[0]], color="0.4", lw=0.8)
        ax[0, j].set_title(c.replace("_left", " ipsilesional").replace("_right", " contralesional"), fontsize=9)
        ax[2, j].set_xlabel("threshold k (x pre-stroke robust SD = 1.4826 x MAD)")
    ax[0, 0].set_ylabel("events / min engaged quiet\n(solid positive, dashed negative)")
    ax[1, 0].set_ylabel("ROC AUC, pre vs acute sessions")
    ax[2, 0].set_ylabel("sensitivity to injected events")
    ax[0, 0].legend(fontsize=6)
    ax[1, 0].legend(fontsize=7, title="polarity", title_fontsize=7)
    ax[2, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("Choosing the event threshold: event rate by polarity, session-level ROC AUC (pre vs acute), and "
                 "recovery of injected GCaMP6s-shaped events (engaged quiet frames, August sessions). Grey line = the k "
                 f"used ({ks['orofacial']:g}; {KNOTE[0]}).", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(V2 / "threshold_choice.png", dpi=110)


def rates(D, sd, ks):
    import pandas as pd
    cols = columns()
    rows = []
    for k_, d in D.items():
        an, mins = k_[:4], d["n_quiet_eng"] / float(d["fs"]) / 60
        if mins < 0.25:
            continue
        r = dict(label=k_, animal=an, epoch=d["epoch"], day=d["day"], quiet_min=mins,
                 quiet_frac=d["n_quiet_eng"] / max(1, d["total_min"] * 60 * float(d["fs"])))
        for i, c in enumerate(cols[:-1]):
            oro = c.startswith("orofacial") or c.split("_")[0] in ("ALM", "tjM1", "SSp-m", "SSp-n")
            K = ks["orofacial" if oro else "limb"]
            n = {}
            for kind in ("pos", "neg"):
                a = d[kind]
                n[kind] = int(((a[:, 0] == i) & (a[:, 3] == 1) & (a[:, 2] > K * sd[an][i])).sum())
            r[f"{c}|rate"], r[f"{c}|rate_neg"] = n["pos"] / mins, n["neg"] / mins
            r[f"{c}|rate_both"] = (n["pos"] + n["neg"]) / mins
        rows.append(r)
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def stats(S):
    import pandas as pd
    from wfield_local import mixed_models as mm
    L = []
    for c in SHOW + ["orofacial", "limb"] + AREA_SHOW:
        for m in ("rate", "rate_neg", "rate_both"):
            L.append(S[["label"]].assign(column=c, measure=m, y=np.log10(S[f"{c}|{m}"] + 0.2).to_numpy()))
    L.append(S[["label"]].assign(column="session", measure="quiet_min", y=S.quiet_min.to_numpy()))
    L = mm.add_time_covariates(pd.concat(L), august_pre=True).dropna(subset=["y"])
    T = mm.lmm_table(L, ["measure", "column"], models=("epoch", "time"), print_animals=True, fdr_by=("measure",))
    T.to_csv(V2 / "lmm_v2.csv", index=False)
    return T


def pooled_figure(S, T, ks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.analysis_kit import boot_ci
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    panels = [(c, "rate") for c in SHOW] + [(c, "rate_both") for c in SHOW]
    fig, ax = plt.subplots(2, len(SHOW), figsize=(3.6 * len(SHOW), 6.4), sharey=True)
    rng = np.random.default_rng(0)
    for n, (c, m) in enumerate(panels):
        a = ax.flat[n]
        col = f"{c}|{m}"
        for x, ep in enumerate(EP):
            B = S[S.epoch == ep]
            for an, A in B.groupby("animal"):
                a.scatter(x + rng.uniform(-0.18, 0.18, len(A)), A[col], marker=ANIMAL_MARKER[an], facecolors="none",
                          edgecolors=EPOCH_LINE[ep], s=18, lw=0.8, alpha=0.8)
            ci = boot_ci({an: list(A[col]) for an, A in B.groupby("animal")}, rng)
            if ci is not None:
                a.errorbar(x + 0.32, ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="o",
                           color=EPOCH_LINE[ep], mfc=EPOCH_LINE[ep], ms=6, capsize=3, lw=1.6, zorder=5)
            t = T[(T.model == "epoch") & (T.measure == m) & (T.column == c) & (T.term == f"C(epoch)[T.{ep}]")]
            if x > 0 and len(t) and t.q.iloc[0] < 0.05:
                q = t.q.iloc[0]
                sig_marks.star_top(a, x, "***" if q < 0.001 else "**" if q < 0.01 else "*", color=EPOCH_LINE[ep])
        a.set_xticks(range(len(EP)), EP, fontsize=8)
        a.set_title(("POSITIVE events\n" if m == "rate" else "EITHER polarity\n")
                    + c.replace("_left", " ipsilesional").replace("_right", " contralesional"), fontsize=9)
    ax[0, 0].set_ylabel("events / min of engaged quiet")
    ax[1, 0].set_ylabel("events / min (positive + negative)")
    handles = [plt.Line2D([], [], marker=mk, ls="", mfc="none", mec="k", label=an) for an, mk in ANIMAL_MARKER.items()]
    ax[0, 0].legend(handles=handles, fontsize=7, loc="upper left")
    sig_marks.reserve_headroom(fig)
    fig.suptitle(f"Quiet-period cortical events (no lick / running within 1.5 s; spout may be out), height > k x the "
                 f"animal's pre-stroke robust SD (k orofacial {ks['orofacial']:g}, limb {ks['limb']:g}; {KNOTE[0]}); engaged part of the session, August pre. Open markers = sessions "
                 "(shape = animal, colour = epoch); filled = pooled mean, nested animal -> session bootstrap 95% CI; "
                 "stars = LMM epoch vs pre, FDR.", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(V2 / "quiet_events_pooled_epochs.png", dpi=110)


def summarise(k_user=None):
    import pandas as pd
    pd.set_option("display.width", 240)
    D, sd = load_all()
    R, I = threshold_table(D, sd)
    R.to_csv(V2 / "threshold_counts.csv", index=False)
    I.to_csv(V2 / "threshold_sensitivity.csv", index=False)
    sens = I[I.epoch == "pre"].groupby(["k", "amp"]).sens.mean().unstack("amp")
    print("sensitivity to injected events, PRE (mean over the 4 columns), by k x amplitude %:\n",
          sens.round(2).to_string())
    oi, li = columns().index("orofacial_right"), columns().index("limb_right")
    print("\nmedian pre robust SD per animal (orofacial_right, limb_right):",
          {an: (round(float(v[oi]), 3), round(float(v[li]), 3)) for an, v in sd.items()})
    A, LO = auc_tables(R)
    A.to_csv(V2 / "threshold_auc.csv", index=False)
    LO.to_csv(V2 / "threshold_auc_loao.csv", index=False)
    print("\nROC AUC pre vs acute by k (bilateral groups):\n",
          A[A.column.isin(["orofacial", "limb"])].pivot_table(index="k", columns=["column", "kind"], values="auc")
          .round(2).to_string())
    print("\nleave-one-animal-out: k chosen on 3 animals, AUC on the held-out one:\n",
          LO[LO.column.isin(["orofacial", "limb"])].round(3).to_string(index=False))
    ks = {}
    for g in ("orofacial", "limb"):
        ch = LO[(LO.column == g) & (LO.kind == "pos")].k_chosen
        ks[g] = float(k_user if k_user is not None else ch.median())
    KNOTE[0] = ("common k for both groups, Priya 11-01" if k_user is not None
                else "median of the leave-one-animal-out choices")
    print(f"\nchosen k (median of the leave-one-animal-out choices, positive events): {ks}"
          + (" [user override]" if k_user is not None else ""))
    threshold_figure(R, I, A, ks)
    S = rates(D, sd, ks)
    S.to_csv(V2 / "quiet_events_sessions.csv", index=False)
    print("\nquiet minutes (engaged) per session, median by animal x epoch:\n",
          S.groupby(["animal", "epoch"]).quiet_min.median().unstack()[EP].round(1).to_string())
    T = stats(S)
    show = T[(T.model == "epoch") & (T.term != "pre_level") & T.column.isin(SHOW + ["orofacial", "limb", "session"])]
    print("\nLMM epoch model vs August pre (rates log10(rate + 0.2)):\n",
          show[["measure", "column", "term", "coef", "se", "p", "q", "re", "n"]].round(4).to_string(index=False))
    pooled_figure(S, T, ks)
    (V2 / "README.txt").write_text(__doc__)
    print("wrote", V2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not a.summarise:
        from wfield_local.analysis_kit import curated_labels
        from wfield_local.parallel import fan_out
        labels = a.only or [k for k in sorted(curated_labels()) if k[-4:] >= "0806"]
        res, fail = fan_out(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
        if len(fail) > 3:
            raise SystemExit("too many failures")
    summarise(a.k)


if __name__ == "__main__":
    main()
