"""Movement-corrected event-triggered maps: face movement regressed out of the PIXEL movie (cohort1, 2026-11-01).

Priya 11-01 (after T1): "let's replace the page map with the movement corrected one (can we regress out mvmt from all?)
... the acute events seem to decrease faster." Replaces Fig 15.7 (raw movie, raw events).

WHAT IS REGRESSED: the same out-of-fold ridge as quiet_spikes_v2_fr (own-session FaceRhythm factors + 20 point PCs +
speeds, 2-s high-passed, lags +/-0.3 s, fitted on ENGAGED QUIET frames, 10 contiguous folds) applied to every temporal
component of the high-passed movie (linear 2-s moving-average high-pass, quiet_fluct (2)), so the residual movie =
movie minus everything the face predicts, pixel by pixel (the SVD basis is linear, so regressing components = regressing
pixels). Area traces (2-s-median detrended) get the same regression.
EVENTS: the face-regressed events of quiet_spikes_v2_fr (res_pos / res_neg), engaged quiet, height > 4 x the animal's
August-pre robust SD of the residual trace. Maps at -0.3, 0, +0.2, +0.5, +1.0 s; traces -1 .. +2 s.
DECAY: per session, from the event-triggered mean residual trace of the triggering group: time from the peak to half its
height (interpolated), and the value at +0.5 s / peak.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_eta_fr.py [--jobs 5] [--summarise]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QS_FR_BASIS", "own")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import DETREND_S, columns, session_traces  # noqa: E402
from quiet_spikes_fr import N_FOLD, face_regressors, lagged, ridge_oof  # noqa: E402
from quiet_spikes_v2 import V2, busy_frames, detrend  # noqa: E402

OUTF = V2 / "eta_fr"
FRV = V2 / "fr_regressed_own" / "sessions"
K = 4.0
MAP_LAGS_S = (-0.3, 0.0, 0.2, 0.5, 1.0)
TR_LAG_S = (-1.0, 2.0)
TRACE_COLS = ["orofacial_left", "orofacial_right", "limb_left", "limb_right", "orofacial", "limb", "cortex"]
GROUPS = ("orofacial", "limb")
EP = ["pre", "acute", "subacute", "chronic"]


def pre_res_sd(animal):
    from wfield_local import epochs
    v = [np.load(f)["sd_res"] for f in sorted(FRV.glob(f"{animal}_*.npz"))
         if epochs.epoch_of(f.stem) == "pre" and f.stem[-4:] >= "0806"]
    return np.nanmedian(v, 0)


def extract(label):
    f = OUTF / "sessions" / f"{label}.npz"
    if f.exists() or not (FRV / f"{label}.npz").exists():
        return label
    from scipy.ndimage import uniform_filter1d
    from wfield_local import combined_encoding as ce
    Xr, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)
    X = detrend(Xr, fs)
    V, ft2, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    V = np.asarray(V, np.float64)
    Vh = V - uniform_filter1d(V, int(round(2.0 * fs)), axis=0, mode="nearest")
    busy, badw = busy_frames(label, ft, fs)
    R, _ = face_regressors(label, ft, int(round(DETREND_S * fs)) | 1)
    XL = lagged(R, fs)
    okX = np.isfinite(XL).all(1)
    qe = ~busy & ~badw & np.isfinite(X).all(0) & okX & engaged
    T = X.shape[1]
    fold = np.arange(T) * N_FOLD // T
    pv, _ = ridge_oof(XL, Vh, qe, fold)
    Vr = Vh - np.nan_to_num(pv)
    cols = columns()
    ci = [cols.index(c) for c in TRACE_COLS]
    px, _ = ridge_oof(XL, X[ci].T, qe, fold)
    Xres = X[ci] - np.nan_to_num(px.T)
    z = np.load(FRV / f"{label}.npz")
    sd = pre_res_sd(label[:4])
    lags = np.arange(int(round(TR_LAG_S[0] * fs)), int(round(TR_LAG_S[1] * fs)) + 1)
    mlag = [int(np.argmin(np.abs(lags / fs - s))) for s in MAP_LAGS_S]
    out = {}
    for g in GROUPS:
        gi = cols.index(g)
        for kind in ("pos", "neg"):
            a = z[f"res_{kind}"]
            fr_ev = a[(a[:, 0] == gi) & (a[:, 3] == 1) & (a[:, 2] > K * sd[gi]), 1].astype(int)
            fr_ev = fr_ev[(fr_ev + lags[0] >= 0) & (fr_ev + lags[-1] < T)]
            key = f"{g}_{kind}"
            out[f"n_{key}"] = len(fr_ev)
            if len(fr_ev):
                idx = fr_ev[:, None] + lags[None, :]
                out[f"etatr_{key}"] = Xres[:, idx].mean(1).astype(np.float32)
                mv = Vr[idx[:, mlag]].mean(0)
                out[f"etamap_{key}"] = (100 * (Ub @ mv.T).T).astype(np.float32)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, flat=flat, shape=np.array(shape), parea=parea, lags_s=lags / fs,
                        map_lags_s=np.array(MAP_LAGS_S), fs=fs, **out)
    return label


# ------------------------------------------------------------------------------------------------------- summary
def load():
    from wfield_local import epochs
    D = {}
    for f in sorted((OUTF / "sessions").glob("*.npz")):
        ep = epochs.epoch_of(f.stem)
        if ep is None or f.stem[-4:] < "0806":
            continue
        D[f.stem] = dict(np.load(f), epoch=ep)
    return D


def decay(tr, lags):
    """(peak -> half decay time s, value at +0.5 s / peak) of one mean trace (positive-going)."""
    j0 = int(np.argmin(np.abs(lags)))
    w = slice(j0 - 3, j0 + 4)
    jp = w.start + int(np.argmax(tr[w]))
    pk = tr[jp]
    if not np.isfinite(pk) or pk <= 0:
        return np.nan, np.nan
    half = pk / 2
    after = np.flatnonzero(tr[jp:] < half)
    if not len(after):
        return np.nan, np.nan
    k = jp + after[0]
    t = lags[k - 1] + (tr[k - 1] - half) / (tr[k - 1] - tr[k]) * (lags[k] - lags[k - 1])
    j5 = int(np.argmin(np.abs(lags - 0.5)))
    return float(t - lags[jp]), float(tr[j5] / pk)


def decay_table(D):
    import pandas as pd
    rows = []
    for k, d in D.items():
        lags = d["lags_s"]
        for g in GROUPS:
            for kind, sign in (("pos", 1), ("neg", -1)):
                key = f"{g}_{kind}"
                if f"etatr_{key}" not in d or int(d[f"n_{key}"]) < 5:
                    continue
                tr = sign * d[f"etatr_{key}"][TRACE_COLS.index(g)]
                t_half, r05 = decay(tr, lags)
                rows.append(dict(label=k, animal=k[:4], epoch=d["epoch"], group=g, kind=kind, n=int(d[f"n_{key}"]),
                                 decay_half_s=t_half, rel_at_0p5=r05))
    return pd.DataFrame(rows)


def fig_maps(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from quiet_fluct import N_BOOT, area_edges, draw_outline, to_img
    from wfield_local import beta_maps as bm
    k0 = next(iter(D))
    mlags = D[k0]["map_lags_s"]
    flat, shape, parea = D[k0]["flat"], tuple(D[k0]["shape"]), D[k0]["parea"]
    edges = area_edges(parea, flat, shape)
    animals = sorted({k[:4] for k in D})
    onpix = np.zeros(int(np.prod(shape)), bool)
    onpix[flat] = True
    smask = bm.stat_mask() & onpix.reshape(shape)
    fig, ax = plt.subplots(6, len(mlags), figsize=(2.6 * len(mlags), 15.5))
    r, sig_rows = 0, []
    for g in GROUPS:
        key = f"{g}_pos"
        S = {ep: {an: [d[f"etamap_{key}"] for k, d in sorted(D.items()) if k.startswith(an) and d["epoch"] == ep
                       and f"etamap_{key}" in d and d[f"n_{key}"] >= 3] for an in animals} for ep in ("pre", "acute")}
        both = [an for an in animals if S["pre"][an] and S["acute"][an]]
        M = {ep: np.mean([np.nanmean(S[ep][an], 0) for an in animals if S[ep][an]], 0) for ep in ("pre", "acute")}
        M["delta"] = np.mean([np.nanmean(S["acute"][an], 0) - np.nanmean(S["pre"][an], 0) for an in both], 0)
        v = np.nanpercentile(np.abs(np.r_[M["pre"].ravel(), M["acute"].ravel()]), 99)
        vd = np.nanpercentile(np.abs(M["delta"]), 99)
        for ep in ("pre", "acute", "delta"):
            for j in range(len(mlags)):
                a = ax[r, j]
                lim = vd if ep == "delta" else v
                im = a.imshow(to_img(M[ep][j], flat, shape), cmap="RdBu_r", vmin=-lim, vmax=lim)
                if ep == "delta":
                    pre_j = {an: [to_img(m[j], flat, shape) for m in S["pre"][an]] for an in both}
                    ac_j = {an: [to_img(m[j], flat, shape) for m in S["acute"][an]] for an in both}
                    full, lab = bm.significance_contour(pre_j, ac_j, method="nested", n_boot=N_BOOT, mask=smask,
                                                        normalise=False)
                    if full is not None:
                        draw_outline(a, full)
                    sig_rows.append(dict(trigger=key, lag_s=float(mlags[j]), n_animals=len(both), test=lab))
                a.imshow(np.where(edges, 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1.5, alpha=0.5)
                a.set_xticks([])
                a.set_yticks([])
                if r == 0:
                    a.set_title(f"{mlags[j]:+.1f} s", fontsize=9)
            n_an = len(both) if ep == "delta" else sum(bool(S[ep][an]) for an in animals)
            ax[r, 0].set_ylabel(f"{g} UP\n{'DELTA acute - pre' if ep == 'delta' else ep} ({n_an} an.)", fontsize=8,
                                fontweight="bold" if ep == "delta" else "normal")
            fig.colorbar(im, ax=ax[r, :].tolist(), fraction=0.02, label="delta dF/F %" if ep == "delta" else "dF/F %")
            r += 1
    fig.suptitle("MOVEMENT-CORRECTED event-triggered maps: face movement (FaceRhythm) regressed out of every pixel and "
                 "of the event-detection trace; positive events > 4 x pre SD of the residual, engaged quiet; mean of "
                 "animal means; ipsilesional = image left.\nDELTA = acute - pre within animal (own colour scale); black "
                 "outline = significant (nested animals -> sessions bootstrap, max-statistic family-wise p < 0.05, "
                 "8x8-px bins, eroded mask)", fontsize=9)
    fig.subplots_adjust(top=0.93)
    fig.savefig(OUTF / "quiet_eta_maps_fr.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    import pandas as pd
    pd.DataFrame(sig_rows).to_csv(OUTF / "quiet_eta_maps_fr_significance.csv", index=False)
    print(pd.DataFrame(sig_rows).to_string())


def fig_decay(D, Tb, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.analysis_kit import boot_ci
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    animals = sorted({k[:4] for k in D})
    k0 = next(iter(D))
    lags = D[k0]["lags_s"]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    for r, g in enumerate(GROUPS):
        a = ax[r, 0]
        for ep in EP:
            per = []
            for an in animals:
                Z = [d[f"etatr_{g}_pos"][TRACE_COLS.index(g)] for k, d in D.items() if k.startswith(an)
                     and d["epoch"] == ep and f"etatr_{g}_pos" in d and d[f"n_{g}_pos"] >= 5]
                if Z:
                    m = np.nanmean(Z, 0)
                    per.append(m / np.nanmax(m[np.abs(lags) < 0.15]))
            if per:
                a.plot(lags, np.mean(per, 0), color=EPOCH_LINE[ep], lw=2, label=f"{ep} ({len(per)} an.)")
        a.axhline(0.5, color="0.6", lw=0.6, ls="--")
        a.axvline(0, color="0.6", lw=0.6)
        a.set_xlim(-0.6, 1.5)
        a.set_title(f"{g} UP events, residual trace, normalised to peak (mean of animals)", fontsize=8)
        a.legend(fontsize=7)
        for c, m, lab in ((1, "decay_half_s", "peak -> half decay (s)"), (2, "rel_at_0p5", "value at +0.5 s / peak")):
            a = ax[r, c]
            G = Tb[(Tb.group == g) & (Tb.kind == "pos")]
            for k, ep in enumerate(EP):
                gg = G[G.epoch == ep].dropna(subset=[m])
                for an, x in gg.groupby("animal"):
                    a.plot(np.full(len(x), k + (animals.index(an) - 1.5) * 0.08), x[m], ANIMAL_MARKER[an],
                           mfc="none", mec=EPOCH_LINE[ep], ms=4, alpha=0.7)
                if gg.animal.nunique() >= 2:
                    ci = boot_ci({an: list(x[m]) for an, x in gg.groupby("animal")}, np.random.default_rng(0))
                    a.errorbar(k + 0.32, ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="s",
                               color=EPOCH_LINE[ep], capsize=3)
                t = T[(T.measure == m) & (T.group == g) & (T.term.str.contains(ep))] if k else T.iloc[0:0]
                if len(t) and float(t.q.iloc[0]) < 0.05:
                    sig_marks.star_top(a, k, "**" if float(t.q.iloc[0]) < 0.01 else "*")
            a.set_xticks(range(4))
            a.set_xticklabels(EP, fontsize=8)
            a.set_title(f"{g} UP: {lab} (sessions; square = bootstrap mean)", fontsize=8)
    sig_marks.reserve_headroom(fig)
    fig.suptitle("Do acute events decay faster? Movement-corrected (face-regressed) event-triggered residual traces; "
                 "stars = LMM epoch vs August pre, FDR", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTF / "quiet_eta_decay_fr.png", dpi=110)
    plt.close(fig)


def summarise():
    import pandas as pd
    from wfield_local import mixed_models as mm
    pd.set_option("display.width", 220)
    D = load()
    print(len(D), "sessions")
    Tb = decay_table(D)
    Tb.to_csv(OUTF / "decay_sessions.csv", index=False)
    for m in ("decay_half_s", "rel_at_0p5"):
        print(f"\n{m}, UP events, animal x epoch (session median):\n",
              Tb[Tb.kind == "pos"].pivot_table(index=["group", "animal"], columns="epoch", values=m, aggfunc="median")
              .reindex(columns=EP).round(3).to_string())
    L = pd.concat([Tb[Tb.kind == "pos"][["label", "group"]].assign(measure=m, y=Tb[Tb.kind == "pos"][m].to_numpy())
                   for m in ("decay_half_s", "rel_at_0p5")])
    L = mm.add_time_covariates(L, august_pre=True).dropna(subset=["y"])
    T = mm.lmm_table(L, ["measure", "group"], models=("epoch",), print_animals=False, fdr_by=("measure",))
    T.to_csv(OUTF / "lmm_decay.csv", index=False)
    print(T[T.term != "Intercept"][["measure", "group", "term", "coef", "se", "p", "q"]].round(4).to_string())
    fig_maps(D)
    fig_decay(D, Tb, T)
    (OUTF / "README.txt").write_text(__doc__)
    print("wrote", OUTF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not a.summarise:
        from wfield_local.parallel import fan_out
        labels = a.only or sorted(f.stem for f in FRV.glob("*.npz") if f.stem[-4:] >= "0806")
        res, fail = fan_out(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
        if a.only:
            return
    summarise()


if __name__ == "__main__":
    main()
