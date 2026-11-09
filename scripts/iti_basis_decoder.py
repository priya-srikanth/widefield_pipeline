"""The FROZEN lick-direction decoder of page 3.8 / 3.9 in two other feature bases (Priya 10-26: "would locaNMF basis be
useful? ... the encoder pixel run already built a shared spatial basis per animal from pre-stroke sessions only -- yes
let's do this too"). Allen-area averages (the primary basis, shared with the encoder) are coarse; if the ITI "remap"
were a small spatial shift inside areas (e.g. registration drift), a finer basis could recover the frozen decoder.

Sources (per animal, all fixed BEFORE the stroke; DECISIONS 2026-08-12 asks for decoders in both bases):
  locanmf   the animal's joint LocaNMF basis (`wfield_local.joint_locanmf`, footprints fit 08-15 on 11 PRE sessions
            06-06..08-14, ~90 components); EVERY session is PROJECTED onto the frozen footprints from its current
            adopted SVTcorr (`Basis.project`), the in-fit pre sessions too, so all sessions share one preprocessing
  svdbasis  the animal's pre-stroke shared spatial basis of the encoder pixel run (`frozen_encoder_<tag>/fit/
            spatial_<animal>.pkl`, 200 variance-weighted components pooled over its pre sessions)
  areas_split  the saved Allen-area features of `scripts/iti_licks.py` WITH the ALM / tjM1 split (PRIMARY since 10-27 late,
               Priya "use split bases, for consistency"; iti_licks.area_features)
  areas        the same without the split (plain Allen areas: whole MOs / MOp, ALM / tjM1 out) -- robustness row
Same licks (`<E>/iti_licks/sessions/*.npz`: ITI and in-trial post-cue licks, 3-D lateral direction), same decoder:
ridge per animal trained on its AUGUST pre-stroke licks of the set, pre sessions leave-one-session-out, features =
activity averaged over lags 0..0.2 s from the lick peak (and per lag for the time course).
Outputs <E>/iti_licks/basis_<source>/sessions/<label>.npz, <E>/iti_licks/basis_decoder.csv,
figures/lick_basis_decoder.png.
Figure data (figdata.py convention): <E>/iti_licks/figdata/lick_basis_decoder.pkl (per source x lick set: the epoch
time-course mean +- SEM, the per-lag significance strip (`il.sig_compute`) and the title numbers); `--plot-only`
redraws the figure from that file alone (no decoding, no bootstrap).
Usage: python scripts/iti_basis_decoder.py [--sources locanmf svdbasis] [--jobs 2] [--summarise] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402
from figdata import load_fd, save_fd  # noqa: E402

from wfield_local import mask_keys  # noqa: E402

FIG = "lick_basis_decoder"

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
LAGS = np.round(np.arange(-0.5, 0.5001, 0.05), 3)
WIN = (LAGS >= 0.0) & (LAGS <= 0.2)
AREAS = {"areas_split": "split", "areas": "plain"}     # Allen-area rows: source -> basis (both always; split = primary)
SPATIAL = il.te_basis_run() / "fit"       # the TE encoder basis of THIS run mode (coordinator 11-03; never _spont, 11-07)


def features(label: str, source: str):
    """(T x K) drift-removed feature traces and frame times for one session in ``source``."""
    import pickle

    from wfield_local import combined_encoding as ce
    from wfield_local import config
    V, ft, Ub, _flat, _parea, _names, _shape, pstat = ce.load_imaging(label)
    an = config.animal_of(label)
    if source == "svdbasis":
        sp = pickle.loads(il.te_basis_spatial(an, SPATIAL).read_bytes())
        R = V.astype(np.float64) @ np.asarray(sp["P"].T @ Ub[pstat].astype(np.float64)).T
    elif source == "locanmf":
        from wfield_local import joint_locanmf as jl
        s = next(x for x in config.load_sessions() if x["label"] == label)
        C = jl.load(an, warn_stale=False).project(s)                       # (ncomp, T), adopted SVTcorr
        if C.shape[1] != len(ft):
            raise ValueError(f"{label}: LocaNMF projection has {C.shape[1]} frames, imaging {len(ft)}")
        R = np.nan_to_num(np.asarray(C, np.float64).T)
    else:
        raise ValueError(source)
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    return R, ft


def extract(item) -> dict:
    label, source = item
    out = il.OUT / f"basis_{source}" / "sessions" / f"{label}.npz"
    if out.exists():
        if source == "svdbasis":                  # stale-read guard (11-09): the svdbasis rows use the stat mask
            mask_keys.check_npz(out)
        return {"label": label, "cached": True}
    d = il._load(label)
    R, ft = features(label, source)
    lo, hi = ft[0] - LAGS[0], ft[-1] - LAGS[-1]
    res = {"epoch": d["epoch"], "animal": d["animal"]}
    for st, kt, ky in (("iti", "iti_t", "iti_side"), ("trial", "tr_t", "tr_side")):
        t, y = d[kt].astype(float), d[ky].astype(float)
        ok = (t > lo) & (t < hi) & np.isfinite(y)
        res[f"{st}_X"] = il._sample(R, ft, t[ok], LAGS).astype(np.float32)
        res[f"{st}_y"] = y[ok]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res, **mask_keys.stamp())
    return {"label": label, "n_feat": R.shape[1]}


def decode(source: str) -> pd.DataFrame:
    """Frozen decoder r per session (lags 0..0.2 averaged) and per lag; same recipe as page 3.9."""
    from wfield_local import frozen_readouts
    sd = il.OUT / f"basis_{wd.basis_dir(source)}" / "sessions" if source not in AREAS else None
    labels = sorted(p.stem for p in (sd or il.XOUT / il.SESS_NAME).glob("*.npz"))
    D = {}
    for x in labels:
        if source in AREAS:                    # Allen areas, split or plain (il.area_features, per lag)
            z = il._load(x)
            li = np.isin(np.round(il.LAGS_D, 3), LAGS)
            D[x] = {"epoch": str(z["epoch"]), "animal": str(z["animal"]),
                    "iti": (il.area_features(z["iti_X"].astype(float)[:, li], z["cols"], x, basis=AREAS[source]),
                            z["iti_side"].astype(float)),
                    "trial": (il.area_features(z["tr_X"].astype(float)[:, li], z["cols"], x, basis=AREAS[source]),
                              z["tr_side"].astype(float))}
        else:
            z = np.load(sd / f"{x}.npz")
            if wd.basis_dir(source) == "svdbasis":
                mask_keys.check_obj(z, sd / f"{x}.npz")   # stale-read guard (11-09)
            k = wd.locanmf_keep(str(z["animal"])) if source == "locanmfM" else slice(None)   # masked LocaNMF
            D[x] = {"epoch": str(z["epoch"]), "animal": str(z["animal"]),
                    "iti": (z["iti_X"].astype(float)[..., k], z["iti_y"]),
                    "trial": (z["trial_X"].astype(float)[..., k], z["trial_y"])}
    rows = []
    for st in ("iti", "trial"):
        for a in AN:
            S = {x: d for x, d in D.items() if d["animal"] == a and len(d[st][1]) >= 20}
            pre = [x for x, d in S.items() if d["epoch"] == "pre"]
            if len(pre) < 3:
                continue
            for li in [None, *range(len(LAGS))]:                 # None = the 0..0.2 s window (the summary number)
                X = {x: (S[x][st][0][:, WIN].mean(1) if li is None else S[x][st][0][:, li]) for x in S}
                # STORED BY DEFAULT (11-02 pass 2, `frozen_readouts`, opt-out WIDEFIELD_ITI_FROZEN=0)
                fits = frozen_readouts.pre_loso_ridge(
                    a, f"iti_basis_decoder-{source}-{st}-{'w0-0.2' if li is None else f'lag{float(LAGS[li]):g}'}", pre,
                    X, {q: S[q][st][1] for q in pre}, np.logspace(-1, 5, 13))
                for x in S:
                    sc, m = fits[x if x in pre else None]
                    r = float(np.corrcoef(m.predict(sc.transform(X[x])), S[x][st][1])[0, 1])
                    rows.append({"source": source, "set": st, "label": x, "animal": a, "epoch": S[x]["epoch"],
                                 "lag": "0-0.2" if li is None else float(LAGS[li]), "r": r})
    return pd.DataFrame(rows)


def summarise(sources) -> None:
    from wfield_local import analysis_kit as ak
    R = pd.concat([decode(s) for s in [*AREAS, *sources]], ignore_index=True)
    R.to_csv(il.OUT / "basis_decoder.csv", index=False)
    W = R[R.lag == "0-0.2"].copy()
    pre = W[W.epoch == "pre"].groupby(["source", "set", "animal"]).r.mean()
    W["r_norm"] = W.r / [pre.get((s_, t_, a_), np.nan) for s_, t_, a_ in zip(W.source, W.set, W.animal)]
    pd.set_option("display.width", 200)
    rows = []
    for (src, st), t in W.groupby(["source", "set"]):
        print(f"\n== {src} | {st}: per animal mean r (rule 8)")
        print(t.groupby(["animal", "epoch"]).r.mean().unstack().reindex(columns=EP).round(3).to_string())
        for val in ("r", "r_norm"):
            ref = {a: t[(t.animal == a) & (t.epoch == "pre")][val].dropna().to_numpy() for a in AN}
            for ep in EP[1:]:
                post = {a: t[(t.animal == a) & (t.epoch == ep)][val].dropna().to_numpy() for a in AN}
                com = [a for a in AN if len(ref[a]) and len(post[a])]
                iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
                rows.append({"source": src, "set": st, "value": val, "epoch": ep, "delta": iv.point, "lo": iv.lo,
                             "hi": iv.hi, "pre_mean": float(np.mean([v.mean() for v in ref.values() if len(v)]))})
    T = pd.DataFrame(rows)
    T.to_csv(il.OUT / "basis_decoder_tests.csv", index=False)
    print("\n" + T[T.value == "r"].round(3).to_string(index=False))
    srcs = [*AREAS, *sources]
    TC = R[R.lag != "0-0.2"].astype({"lag": float})
    sig_recs, cells = [], {}
    for src in srcs:
        for st in ("iti", "trial"):
            curves = {}
            for ep in EP:
                g = TC[(TC.source == src) & (TC.set == st) & (TC.epoch == ep)].groupby(["lag", "label"]).r.mean().unstack()
                if g.empty:
                    continue
                curves[ep] = (g.index.to_numpy(float), g.mean(1).to_numpy(), (g.std(1, ddof=1) /
                                                                              np.sqrt(g.shape[1])).to_numpy(), g.shape[1])
            S = il.sig_compute(TC[(TC.source == src) & (TC.set == st)], tag={"source": src, "set": st})
            sig_recs += S["recs"]
            tt = T[(T.source == src) & (T.set == st) & (T.value == "r_norm")].set_index("epoch")
            txt = "  ".join(f"{ep[:3]} {tt.loc[ep, 'delta'] * 100:+.0f}%{'*' if tt.loc[ep, 'lo'] > 0 or tt.loc[ep, 'hi'] < 0 else ''}"
                            for ep in EP[1:] if ep in tt.index)
            cells[(src, st)] = dict(curves=curves, sig=S, txt=txt)
    pd.DataFrame(sig_recs).to_csv(il.OUT / "sig_bins_3_14.csv", index=False)
    D = dict(srcs=srcs, cells=cells)
    save_fd(FIG, D)
    plot(D)


def plot(D: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    srcs = D["srcs"]
    lab = {"areas_split": "Allen areas + ALM / tjM1 split (primary)", "areas": "Allen areas, plain (no split)",
           "locanmf": "joint LocaNMF (pre-stroke footprints, unmasked)",
           "locanmfM": "joint LocaNMF, glue / MOB comps dropped", "svdbasis": "pre-stroke SVD basis (200 comps)"}
    fig, axs = plt.subplots(2, len(srcs), figsize=(4.6 * len(srcs), 7.5), sharey=True, squeeze=False)
    for j, src in enumerate(srcs):
        for i, st in enumerate(("iti", "trial")):
            ax = axs[i, j]
            c = D["cells"][(src, st)]
            for ep, (x, mu, se, n) in c["curves"].items():
                ax.plot(x, mu, color=colors[ep], lw=1.8, label=f"{ep} ({n})")
                ax.fill_between(x, mu - se, mu + se, color=colors[ep], alpha=0.2, lw=0)
            il.sig_draw(ax, c["sig"], colors)
            txt = c["txt"]
            ax.set_title(f"{lab[src]}\n{'ITI (self-initiated)' if st == 'iti' else 'in-trial post-cue'} licks\n"
                         f"0-0.2 s vs pre: {txt}", fontsize=8.5)
            ax.axvline(0, color="0.7", lw=0.6)
            ax.axhline(0, color="0.85", lw=0.6)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            if j == 0:
                ax.set_ylabel("FROZEN decoder: r (decoded vs actual\nlateral tongue-tip position)", fontsize=8.5)
            if i == 1:
                ax.set_xlabel("lag from lick peak (s)", fontsize=8.5)
    axs[0, 0].legend(fontsize=7, frameon=False)
    il._headroom([axs[0, 0]])                              # y is shared: raise once
    fig.suptitle(f"NEURAL: the FROZEN pre-stroke decoder (fit once on pre licks, never refit) of lateral tongue-tip position in {len(srcs)} SPATIAL bases (same licks, same decoder; "
                 "per animal, trained on August pre licks, pre leave-one-session-out). Title numbers = change from pre "
                 "normalized to the animal's pre (lags 0-0.2 s); * = nested-bootstrap CI excludes 0.\n" + il.SIG_NOTE,
                 fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_png_svg(fig, il.OUT / "figures" / f"{FIG}.png", dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["locanmf", "svdbasis"], choices=["locanmf", "svdbasis"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from its figure data alone")
    a = ap.parse_args(argv)
    if a.plot_only:
        plot(load_fd(FIG))
        return 0
    if not a.summarise:
        labels = a.labels or il.labels()
        items = [(x, s) for s in a.sources for x in labels
                 if not (il.OUT / f"basis_{s}" / "sessions" / f"{x}.npz").exists()]
        print(f"{len(items)} session x source items", flush=True)
        _got, fail = ak.fan_sessions(items, extract, jobs=a.jobs, label="basis")
        for it, err in fail:
            print(f"  !! {it}: {err}", flush=True)
        if fail:   # a half-failed extraction must not be marked done (11-07: all svdbasis items failed, exit 0)
            print(f"  {len(fail)} of {len(items)} items FAILED; not summarising", flush=True)
            return 1
        if a.labels:
            return 0
    # the figure / tests use the MASKED LocaNMF (Priya 10-27 late "swap masked locanmf"); extraction is the same files
    summarise(["locanmfM" if s == "locanmf" else s for s in a.sources])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
