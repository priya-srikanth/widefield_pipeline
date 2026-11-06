"""SIGNIFICANCE of the Musall-style pixel maps with FaceRhythm (Priya 10-25: "can we test significance on these maps?").

Per-session maps = held-out R2 (fraction, 0-1) per 2 x 2-px block (`fr_musall.session_pixels`: the per-session encoder's stored
block SSE maps, main no-FR run + FR run). Test = the repo's map test for epoch changes
(`beta_maps.hierarchical_bootstrap_significance`: animals -> sessions nested bootstrap of epoch - pre, bootstrap-SE
interval; family-wise threshold over bins = the max-statistic, `beta_maps.CORRECTION = "maxstat"`: the 95th percentile over
draws of max over all tested bins of |boot - obs| / SE, so per panel, not corrected across panels), on the RAW R2 scale (R2 is already scale-free, so no per-animal pre normalisation).
Bins: 16 and 32 FULL-resolution px (= 8 and 16 block px on this half-resolution grid), stat mask = the eroded brain
mask (`beta_maps.stat_mask`) block-downsampled. Calibration as for the residual maps: random half-splits of each
animal's pre sessions (pseudo pre vs pseudo post) -> share of null splits with >= 1 significant bin.

--drift-control (SUPPLEMENTAL, Priya 11-06 after FR_PIXEL_CALIBRATION_20261105 s3-4): the pre sessions are not
exchangeable (late pre differs from early pre in 10 / 14 maps, 4/4 animals same sign), so a pre-mean baseline cannot
separate a post-stroke change from a pre trend that continued. Same maps, same test, same bins, same mask: LATE pre
(each animal's second half of its pre sessions in date order) minus EARLY pre (first n // 2), i.e. the chronological
member of the null's half-splits. No refit and no epoch re-test: the epoch tests are READ from this folder's
sig_<q>_<w>_<epoch>_<F>px.npz (run without the flag first) to count epoch-significant bins whose sign the pre drift
shares. Output: sig_bins_driftcontrol.csv, session_means_driftcontrol.csv, sig_<q>_<w>_driftpre_<F>px.npz and
figures/fr_pixel_significance_<w>_mode<M>_driftcontrol.png (with --zscore: in pixel_significance_z/).

Usage: python scripts/fr_pixel_significance.py [--mode A] [--windows lick cue] [--n-null 50] [--drift-control]
Output: <E>/fr_encoding_summary/pixel_significance/ (sig_bins.csv, null.csv, sig_<q>_<w>_<F>px.npz, figures/)
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E, session_pixels, table  # noqa: E402

QS = ["full_fr", "unique_fr", "alone_uninstructed", "uniquefr_uninstructed", "uniquefr_task", "uniquefr_position",
      "unique_position"]
QLAB = {"full_fr": "full R2 (with FR)", "unique_fr": "FaceRhythm unique", "alone_uninstructed": "uninstructed alone",
        "uniquefr_uninstructed": "uninstructed unique | FR", "uniquefr_task": "task unique | FR",
        "uniquefr_position": "position unique | FR", "unique_position": "position unique, no FR"}
EPS = ["acute", "subacute", "chronic"]
FACTORS = {16: 8, 32: 16}            # full-resolution px -> block px on the 270 x 320 grid


def _load(args):
    lab, mode, windows = args
    return lab, session_pixels(lab, mode, windows)


def block_mask():
    from wfield_local import beta_maps as bm
    m = bm.stat_mask().astype(float)
    return m.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2).mean((1, 3)) >= 0.5


def test(pre_by, post_by, f, mask, n_boot=2000):
    from wfield_local import beta_maps as bm
    got = bm.hierarchical_bootstrap_significance(pre_by, post_by, factor=f, n_boot=n_boot, mask=mask)
    if got is None:
        return None, 0
    return got[0], int(got[4])


def correction_label(correction: str | None = None) -> str:
    """Caption wording for the threshold `test()` actually applies (`beta_maps.CORRECTION`), so a caption cannot say
    Bonferroni while the code uses the max-statistic (FR_PIXEL_CALIBRATION_20261105 s1: every caption said Bonferroni)."""
    if correction is None:
        from wfield_local import beta_maps as bm
        correction = bm.CORRECTION
        resels = bool(getattr(bm, "CORRECT_OVER_RESELS", False))
    else:
        resels = False
    if correction == "maxstat":
        return "max-statistic over bins, family-wise per panel (not corrected across panels)"
    return "Bonferroni over " + ("resolution elements" if resels else "bins") + ", per panel"


def _date_key(label: str):
    """Sort key = days since stroke (date order inside an animal); label when no date is known."""
    try:
        from wfield_local import epochs
        d = epochs.days_since_stroke(label)
    except Exception:                                    # noqa: BLE001 - a synthetic / unknown label sorts by name
        d = None
    return (0, d, label) if d is not None else (1, 0, label)


def early_late(items: dict) -> tuple[dict, dict]:
    """{animal: [(label, map), ...]} pre sessions -> ({animal: early maps}, {animal: late maps}): in DATE order, early =
    the first n // 2, late = the rest (PS93's 7 -> 3 v 4), the split of FR_PIXEL_CALIBRATION_20261105 s3. An animal with
    < 2 pre sessions is dropped (nothing to split)."""
    early, late = {}, {}
    for an, v in items.items():
        v = sorted(v, key=lambda t: _date_key(t[0]))
        h = len(v) // 2
        if h < 1:
            continue
        early[an] = [m for _, m in v[:h]]
        late[an] = [m for _, m in v[h:]]
    return early, late


def bin_mean(mm: np.ndarray, shape: tuple, f: int) -> np.ndarray:
    """Block-px map -> bin grid `shape` (nanmean over f x f), the binning the up / down counts of `main` use."""
    import warnings
    hh, ww = shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # all-NaN (off-mask) bins stay NaN
        return np.nanmean(mm[:hh * f, :ww * f].reshape(hh, f, ww, f), (1, 3))


def drift_overlap(ep_sig, ep_mean, dr_sig, dr_mean) -> dict:
    """Epoch-significant bins vs the late - early pre drift on the same bin grid (all boolean / float arrays, same shape).
      epoch_sig                      bins significant for epoch - pre
      epoch_sig_drift_same_sign      ... whose drift POINT ESTIMATE has the same sign (a pre trend that, continued, mimics it)
      epoch_sig_drift_sig_same_sign  ... whose drift is itself significant with the same sign
    """
    if ep_sig is None or dr_sig is None or ep_sig.shape != dr_sig.shape:
        return dict(epoch_sig=np.nan, epoch_sig_drift_same_sign=np.nan, epoch_sig_drift_sig_same_sign=np.nan)
    same = np.sign(ep_mean) == np.sign(dr_mean)
    same &= np.sign(ep_mean) != 0
    return dict(epoch_sig=int(ep_sig.sum()), epoch_sig_drift_same_sign=int((ep_sig & same).sum()),
                epoch_sig_drift_sig_same_sign=int((ep_sig & dr_sig & same).sum()))


def animals_agree(sig, per_animal_binned, pooled_binned) -> int:
    """Significant bins where EVERY animal's own difference has the pooled sign (rule 8: the per-animal check)."""
    if sig is None or not len(per_animal_binned):
        return 0
    s = np.sign(pooled_binned)
    agree = np.all([np.sign(p) == s for p in per_animal_binned], 0)
    return int((sig & agree & (s != 0)).sum())


def drift_control(meta, maps, mask, out, a, zsd) -> int:
    """--drift-control: late - early pre per map with `test()` (no refit, epoch tests read from `out`)."""
    import textwrap

    import matplotlib.pyplot as plt

    from combined_encoding_figures import _outline
    from wfield_local.cat_points import animal_handles, animal_marker
    from wfield_local.epoch_figures import EPOCH_GREY
    from wfield_local.sig_marks import reserve_headroom
    from wfield_local.writeguard import assert_writable
    assert_writable(out / "figures")

    def items(q, w, z):
        d = {}
        for _, m in meta[meta.epoch == "pre"].iterrows():
            x = np.where(mask, maps[m.label][(q, w)], np.nan)
            d.setdefault(m.animal, []).append((m.label, x if z is None else x / z[m.animal]))
        return d

    rows, pics, trace = [], {}, []
    for q in QS:
        for w in a.windows:
            z = zsd(q, w)
            early, late = early_late(items(q, w, z))
            com = sorted(set(early) & set(late))
            eb, lb = {x: early[x] for x in com}, {x: late[x] for x in com}
            per = [np.nanmean(lb[x], 0) - np.nanmean(eb[x], 0) for x in com]
            mm = np.nanmean(per, 0)
            for F, f in FACTORS.items():
                sig, n = test(eb, lb, f, mask)
                sm = bin_mean(mm, sig.shape, f) if sig is not None else None
                k = 0 if sig is None else int(sig.sum())
                up = 0 if sig is None else int((sig & (sm > 0)).sum())
                dn = 0 if sig is None else int((sig & (sm < 0)).sum())
                agree = 0 if sig is None else animals_agree(sig, [bin_mean(p, sig.shape, f) for p in per], sm)
                np.savez_compressed(out / f"sig_{q}_{w}_driftpre_{F}px.npz", mean=mm.astype(np.float32),
                                    sig=np.zeros((1, 1), bool) if sig is None else sig)
                pics[(q, w, "drift", F)] = (mm, sig)
                base = dict(q=q, window=w, px=F, drift_sig=k, drift_up=up, drift_down=dn,
                            drift_sig_all_animals_same_sign=agree, n_bins=n, n_animals=len(com),
                            sessions_early_late="/".join(f"{len(eb[x])}v{len(lb[x])}" for x in com))
                for ep in EPS:
                    fp = out / f"sig_{q}_{w}_{ep}_{F}px.npz"
                    ov = drift_overlap(None, None, None, None)
                    if fp.exists() and sig is not None:
                        g = np.load(fp)
                        es, em = g["sig"].astype(bool), g["mean"].astype(float)
                        if es.shape == sig.shape:
                            ov = drift_overlap(es, bin_mean(em, es.shape, f), sig, sm)
                        pics[(q, w, ep, F)] = (em, es if es.any() else None)
                    rows.append({**base, "epoch": ep, "epoch_test_found": fp.exists(), **ov})
            # per-session in-mask mean (descriptive; the trend the split tests), every epoch
            for _, m in meta.iterrows():
                x = maps[m.label][(q, w)]
                if z is not None:
                    if m.animal not in z:
                        continue
                    x = x / z[m.animal]
                d = _date_key(m.label)
                trace.append(dict(q=q, window=w, label=m.label, animal=m.animal, epoch=m.epoch,
                                  days=d[1] if d[0] == 0 else np.nan, in_mask_mean=float(np.nanmean(x[mask]))))
            print(q, w, "drift", [(r["px"], r["drift_up"], r["drift_down"]) for r in rows[-2 * len(EPS)::len(EPS)]],
                  flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(out / "sig_bins_driftcontrol.csv", index=False)
    T = pd.DataFrame(trace)
    T.to_csv(out / "session_means_driftcontrol.csv", index=False)

    ol = _outline(meta)
    unit = "PER-ANIMAL z (each animal / its pre-session SD map)" if a.zscore else "held-out R2 (fraction, 0-1)"
    for w in a.windows:
        fig, axs = plt.subplots(3, len(QS), figsize=(2.35 * len(QS), 8.6), squeeze=False,
                                gridspec_kw={"height_ratios": [1, 1, 0.8]})
        for j, q in enumerate(QS):
            dm = pics[(q, w, "drift", 16)][0]
            cm = pics.get((q, w, "chronic", 16), (None, None))[0]
            lim = np.nanpercentile(np.abs(dm), 99)
            if cm is not None:
                lim = max(lim, np.nanpercentile(np.abs(cm), 99))          # one scale per column: drift vs chronic
            im = None
            for i, key in enumerate(("drift", "chronic")):
                ax = axs[i, j]
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                if j == 0:
                    ax.set_ylabel("late pre minus early pre" if key == "drift" else "chronic minus pre", fontsize=8.5)
                if (q, w, key, 16) not in pics:
                    ax.text(0.5, 0.5, "chronic test not found\n(run without --drift-control)", ha="center", va="center",
                            fontsize=7, transform=ax.transAxes)
                    continue
                mm = pics[(q, w, key, 16)][0]
                im = ax.imshow(mm, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                if ol is not None and ol.shape == mm.shape:
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                txt = []
                for F, col, ls in ((16, "k", "solid"), (32, "#7a0177", "dashed")):
                    sig = pics.get((q, w, key, F), (None, None))[1]
                    f = FACTORS[F]
                    if sig is not None and sig.any():
                        big = np.zeros(mm.shape)
                        hh, ww = min(sig.shape[0] * f, mm.shape[0]), min(sig.shape[1] * f, mm.shape[1])
                        big[:hh, :ww] = np.kron(sig, np.ones((f, f)))[:hh, :ww]
                        ax.contour(big, levels=[0.5], colors=col, linewidths=1.0, linestyles=ls)
                    r = S[(S.q == q) & (S.window == w) & (S.px == F) & (S.epoch == "chronic")].iloc[0]
                    if key == "drift":
                        txt.append(f"{F}px {r.drift_up}+/{r.drift_down}- of {r.n_bins}")
                    else:
                        txt.append(f"{F}px {r.epoch_sig:.0f} sig, {r.epoch_sig_drift_same_sign:.0f} drift-sign, "
                                   f"{r.epoch_sig_drift_sig_same_sign:.0f} drift-sig")
                ax.set_title(("" if i else QLAB[q] + "\n") + "\n".join(txt), fontsize=6.5)
            if im is not None:                                            # one shared scale for the column's two maps
                cb = fig.colorbar(im, ax=axs[1, j], orientation="horizontal", fraction=0.05, pad=0.03)
                cb.ax.tick_params(labelsize=6)
            ax = axs[2, j]
            t = T[(T.q == q) & (T.window == w)]
            for (ep, an), g in t.groupby(["epoch", "animal"]):
                ax.plot(g.days, g.in_mask_mean, ls="none", marker=animal_marker(an), mfc="none",
                        mec=EPOCH_GREY.get(ep, "k"), mew=0.8, ms=3.6, alpha=0.7)
            ax.axvline(0, color="0.6", lw=0.6, ls=":")
            ax.tick_params(labelsize=6)
            ax.set_xlabel("days from stroke", fontsize=7)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if j == 0:
                ax.set_ylabel("in-mask mean,\nper session", fontsize=8)
        from matplotlib.lines import Line2D
        eph = [Line2D([], [], ls="none", marker="o", mfc="none", mec=EPOCH_GREY.get(ep, "k"), mew=0.9, ms=4.6, label=ep)
               for ep in ["pre", *EPS] if ep in set(T.epoch)]
        axs[2, -1].legend(handles=eph + animal_handles(sorted(T.animal.unique())), fontsize=6, frameon=False,
                          loc="center left", bbox_to_anchor=(1.0, 0.5))
        reserve_headroom(fig)
        ttl = (f"SUPPLEMENTAL drift control, NEURAL: FaceRhythm-encoder pixel maps, {w} window, {unit}, mean of animals. "
               "Row 1: LATE pre minus EARLY pre (each animal's pre sessions in date order, second half minus first half). "
               "Row 2: chronic minus pre (the main figure's test, read back; same colour scale as row 1). Same test for "
               f"both: nested bootstrap animals -> sessions, {correction_label()}; black = 16 px, purple dashed = 32 px "
               "bins. Row 2 titles: significant bins; of them, bins whose pre drift has the same sign (point estimate); "
               "and bins whose same-sign drift is itself significant. Row 3: per-session in-mask mean (open markers: "
               "shape = animal, colour = epoch). Image left = ipsilesional.")
        fig.suptitle("\n".join(textwrap.wrap(ttl, int(26 * len(QS)))), fontsize=8)
        fig.subplots_adjust(left=0.04, right=0.93, top=0.84, bottom=0.06, wspace=0.12, hspace=0.42)
        save_png_svg(fig, out / "figures" / f"fr_pixel_significance_{w}_mode{a.mode}_driftcontrol.png", dpi=115)
        plt.close(fig)
    print("->", out / "sig_bins_driftcontrol.csv")
    return 0


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="A")
    ap.add_argument("--windows", nargs="+", default=["lick", "cue"])
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=8, help="worker processes (shared box: keep <= 3)")
    ap.add_argument("--zscore", action="store_true",
                    help="PER-ANIMAL z: every map of an animal divided by its pre-session SD map (floored at its median "
                         "pixel SD) before the test (Priya 10-25); output pixel_significance_z/")
    ap.add_argument("--drift-control", action="store_true",
                    help="SUPPLEMENTAL: late pre minus early pre per map, same test; reads the epoch tests already in the "
                         "output folder (no refit, no epoch re-test, no null) -> *_driftcontrol figure + CSV")
    a = ap.parse_args(argv)
    out = E / "fr_encoding_summary" / ("pixel_significance_z" if a.zscore else "pixel_significance")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    meta = table(a.mode, a.windows)[["label", "animal", "epoch"]].drop_duplicates().sort_values("label")
    got = Parallel(n_jobs=a.jobs)(delayed(_load)((lab, a.mode, a.windows)) for lab in meta.label)
    maps = dict(got)
    mask = block_mask()
    print(f"{len(maps)} sessions loaded; stat mask {mask.sum()} block px", flush=True)

    def by(ep, q, w, z=None):
        d = {}
        for _, m in meta[meta.epoch == ep].iterrows():
            d.setdefault(m.animal, []).append(np.where(mask, maps[m.label][(q, w)], np.nan))
        if z is not None:                                  # per-animal z units (same SD for pre and post)
            d = {an: [x / z[an] for x in v] for an, v in d.items() if an in z}
        return d

    def zsd(q, w):
        """{animal: pre-session SD map, floored at its median} (None unless --zscore)."""
        if not a.zscore:
            return None
        out_ = {}
        for an, v in by("pre", q, w).items():
            sd = np.nanstd(np.stack(v), 0, ddof=1)
            out_[an] = np.maximum(sd, np.nanmedian(sd))
        return out_

    if a.drift_control:
        return drift_control(meta, maps, mask, out, a, zsd)

    def job(q, w):
        res, pics = [], {}
        z = zsd(q, w)
        pre = by("pre", q, w, z)
        for ep in EPS:
            post = by(ep, q, w, z)
            com = sorted(set(pre) & set(post))
            pb, qb = {x: pre[x] for x in com}, {x: post[x] for x in com}
            mm = np.nanmean([np.nanmean(qb[x], 0) - np.nanmean(pb[x], 0) for x in com], 0)
            for F, f in FACTORS.items():
                sig, n = test(pb, qb, f, mask)
                k = 0 if sig is None else int(sig.sum())
                up = dn = 0
                if k:
                    hh, ww = sig.shape
                    sm = np.nanmean(mm[:hh * f, :ww * f].reshape(hh, f, ww, f), (1, 3))
                    up, dn = int((sig & (sm > 0)).sum()), int((sig & (sm < 0)).sum())
                res.append(dict(q=q, window=w, epoch=ep, px=F, sig=k, up=up, down=dn, n_bins=n, n_animals=len(com),
                                sessions="/".join(str(len(qb[x])) for x in com)))
                pics[(ep, F)] = (mm, sig)
                np.savez_compressed(out / f"sig_{q}_{w}_{ep}_{F}px.npz", mean=mm.astype(np.float32),
                                    sig=np.zeros((1, 1), bool) if sig is None else sig)
        print(q, w, [(r["epoch"], r["px"], r["sig"], r["up"], r["down"]) for r in res], flush=True)
        return res, pics

    R = [job(q, w) for q in QS for w in a.windows]
    rows = [r for x, _ in R for r in x]
    pd.DataFrame(rows).to_csv(out / "sig_bins.csv", index=False)
    pics = {(q, w): p for (q, w), (_, p) in zip([(q, w) for q in QS for w in a.windows], R)}

    # null: random half-splits of pre
    def null(q, w, i):
        rng = np.random.default_rng(1000 + i)
        pre = by("pre", q, w, zsd(q, w))
        pb, qb = {}, {}
        for an, p in pre.items():
            idx = rng.permutation(len(p))
            h = len(p) // 2
            pb[an] = [p[j] for j in idx[:h]]
            qb[an] = [p[j] for j in idx[h:]]
        rows_ = []
        for F, f in FACTORS.items():
            sig = test(pb, qb, f, mask)[0]
            rows_.append((q, w, i, F, 0 if sig is None else int(sig.sum())))
        return rows_
    N = [r for q in QS for w in a.windows for i in range(a.n_null) for r in null(q, w, i)]
    N = pd.DataFrame(N, columns=["q", "window", "split", "px", "false_bins"])
    N.to_csv(out / "null.csv", index=False)
    print(N.groupby(["px", "window"]).false_bins.agg(mean="mean", any_rate=lambda x: (x > 0).mean()).round(3), flush=True)

    # figure per window: rows = epochs, columns = quantities; 16 px contours black, 32 px dashed purple
    from combined_encoding_figures import _outline
    ol = _outline(meta)
    S = pd.DataFrame(rows)
    for w in a.windows:
        fig, axs = plt.subplots(3, len(QS), figsize=(2.35 * len(QS), 8.4), squeeze=False)
        for j, q in enumerate(QS):
            lim = max(np.nanpercentile(np.abs(pics[(q, w)][(ep, 16)][0]), 99) for ep in EPS)
            for i, ep in enumerate(EPS):
                ax = axs[i, j]
                mm = pics[(q, w)][(ep, 16)][0]
                im = ax.imshow(mm, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                if ol is not None and ol.shape == mm.shape:
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                txt = []
                for F, col, ls in ((16, "k", "solid"), (32, "#7a0177", "dashed")):
                    sig = pics[(q, w)][(ep, F)][1]
                    f = FACTORS[F]
                    if sig is not None and sig.any():
                        big = np.zeros(mm.shape)
                        big[:sig.shape[0] * f, :sig.shape[1] * f] = np.kron(sig, np.ones((f, f)))
                        ax.contour(big, levels=[0.5], colors=col, linewidths=1.0, linestyles=ls)
                    r = S[(S.q == q) & (S.window == w) & (S.epoch == ep) & (S.px == F)].iloc[0]
                    txt.append(f"{F}px {r.up}+/{r.down}- of {r.n_bins}")
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                ax.set_title(("" if i else QLAB[q] + "\n") + "\n".join(txt), fontsize=7)
                if j == 0:
                    ax.set_ylabel(f"{ep} minus pre", fontsize=9)
                if i == 2:
                    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
                    cb.ax.tick_params(labelsize=6)
        nr = N[N.window == w].groupby("px").false_bins.apply(lambda x: (x > 0).mean())
        fig.suptitle(f"NEURAL: significance of the FaceRhythm-encoder pixel maps, {w} window: epoch minus pre, " + ("PER-ANIMAL z (each animal / its pre-session SD map)" if a.zscore else "held-out R2 (fraction, 0-1)") + ", "
                     "mean of animals. Contours: black = significant 16 px bins, purple dashed = 32 px bins\n(nested bootstrap "
                     f"animals -> sessions, {correction_label()}). Null (random pre half-splits): >= 1 false bin in "
                     f"{nr.get(16, np.nan):.0%} (16 px) / {nr.get(32, np.nan):.0%} (32 px) of splits. Image left = ipsilesional.",
                     fontsize=8.5)
        fig.subplots_adjust(left=0.03, right=0.995, top=0.86, bottom=0.04, wspace=0.06, hspace=0.3)
        save_png_svg(fig, out / "figures" / f"fr_pixel_significance_{w}_mode{a.mode}.png", dpi=115)
        plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
