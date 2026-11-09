"""What do the parts of the change LOOK like on cortex? (Priya 11-08: "for the largest fractional contributors, let's
next try to see what they 'look like' (what IS the shared shift), on cortex, over time if needed").

Cohort SVD basis only (WF_SS_BASIS=svd: one basis for all animals, so a coordinate vector IS a pixel map; U_cohort
orthonormal on the glue-free mask). WF_SS_XFORM="" (raw) or gsr (cortex-wide mode removed). First-lick-aligned
(WF_SS_ALIGN=lick, 17 bins of 0.1 s, -1.0..0.7 s).

Per session, Delta = session template - the animal's pre template (pre sessions: leave-one-out vs the other pre sessions =
the null), on the spouts the session has. The parts are fitted SEQUENTIALLY in the order of `state_space_change_parts`
so they sum exactly to Delta and each is a picture:
  gain            per-feature gains of the pre response (B = T_pre - mu; global + per-feature, least squares): what the
                  gain change ADDS on cortex = fitted - B... drawn in the response window, mean over spouts
  per-area offset R1 = Delta - gain; its mean over spouts and bins: one static map
  per-spout offset R2 = R1 - offset; its mean over bins, per spout: six static maps (the spout-specific level)
  shared shift    R3 = R2 - per-spout offset; its mean over spouts: a bin x feature MOVIE (bin-mean 0 by construction),
                  drawn in six windows + time courses in split-basis areas (ALM / tjM1 / SSp-m / SSp-tr / VISa / RSPd)
  rest            R4 = R3 - shift: spout-specific, time-varying (rotation + other + off manifold of the split),
                  drawn per spout in the response window (0.1..0.5 s)
(Sequential least squares, gains first, is the visual counterpart of the split's nested projections; the energies of
`state_space_change_parts` stay the quantitative statement.)
Units: each animal's pre-template RMS per pixel (sqrt(|B|^2 / (spouts x bins x pixels)); U_c orthonormal), mean of 4
animals. Contours: `beta_maps.significance_contour` (nested bootstrap animals -> sessions, 2,000 draws, 8-px blocks,
max-stat per panel, rim suppression), post-epoch session maps vs the pre (leave-one-out) session maps, normalise=False.
Outputs: <state_space_recovery>/part_maps{tag}/maps.npz, page_figs/ss_part_maps_{shift,static,rest}{tag}.png.
EXPLORATORY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_chronic_maps as C  # noqa: E402
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402
import state_space_trajectories as TR  # noqa: E402

from wfield_local.figure_layout import save_png_svg  # noqa: E402
from wfield_local.orofacial_cohort import EPOCH_LINE  # noqa: E402

EPS = ["acute", "subacute", "chronic"]
OUT = S.ROOT / f"part_maps{S.tag()}"
FIGS = S.ROOT / "page_figs"
WINDOWS = [(-1.0, -0.7), (-0.7, -0.4), (-0.4, -0.1), (-0.1, 0.1), (0.1, 0.4), (0.4, 0.7)]
RESP = (0.1, 0.5)
ROIS = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSp-tr", "SSp-ll", "VISa", "RSPd", "MOp_rest", "MOs_rest"]


def pieces(D, B):
    """Sequential split of Delta (s, nb, K) given the pre B (s, nb, K): dict of arrays."""
    s, nb, K = D.shape
    X = np.hstack([B.reshape(-1, 1)] + [np.where(np.arange(K) == f, B, 0).reshape(-1, 1) for f in range(K)])
    coef, *_ = np.linalg.lstsq(X, D.ravel(), rcond=None)
    gain = (X @ coef).reshape(D.shape)
    R1 = D - gain
    off_a = R1.mean(axis=(0, 1))
    R2 = R1 - off_a
    off_s = R2.mean(axis=1)                                               # s x K
    R3 = R2 - off_s[:, None, :]
    shift = R3.mean(axis=0)                                               # nb x K, bin-mean 0
    rest = R3 - shift[None]
    return {"gain": gain, "off_a": off_a, "off_s": off_s, "shift": shift, "rest": rest}


def session_parts(sess):
    """[(label, epoch, unit, ok, pieces)] for one animal (pre sessions leave-one-out)."""
    pre = [s for s in sess if s[1] == "pre"]
    Tpre = R.template(pre)
    okp = np.isfinite(Tpre).all(axis=(1, 2))
    B0 = Tpre[okp] - S.flat(Tpre[okp]).mean(0)
    unit = float(np.sqrt((B0 ** 2).sum() / B0.size * B0.shape[-1] / NPIX))   # pixel RMS of the pre template
    out = []
    for lab, ep, day, Y, pos in sess:
        if ep not in ("pre", *EPS):
            continue
        ref = [s for s in pre if s[0] != lab]
        Tr = R.template(ref)
        Ts = R.template([(lab, ep, day, Y, pos)])
        ok = np.isfinite(Ts).all(axis=(1, 2)) & np.isfinite(Tr).all(axis=(1, 2))
        if ok.sum() < 3:
            continue
        mu = S.flat(Tr[ok]).mean(0)
        out.append((lab, ep, unit, ok, pieces(Ts[ok] - Tr[ok], Tr[ok] - mu)))
    return out


NPIX = 1


def main():
    global NPIX
    if S.BASIS != "svd":
        sys.exit("state_space_part_maps: needs the cohort SVD basis (WF_SS_BASIS=svd)")
    from wfield_local import area_basis, beta_maps
    from wfield_local.atlas_overlay import region_edges
    Uc = np.load(C.BASIS_DIR / "U_cohort.npy").astype(np.float64)
    NPIX = Uc.shape[0]
    atlas = np.load(S.SVD.parents[1] / "atlas.npy")
    mask = np.load(C.BASIS_DIR / "mask.npy").reshape(atlas.shape)
    edges = region_edges(atlas)                       # the FULL CCF outline (bulbs + masked glue included), as the encoder maps
    brain = beta_maps.allen_mask()
    flat = np.flatnonzero(mask.ravel())
    split = area_basis.basis_masks(area_basis.roi_masks(atlas.ravel()[flat], beta_maps.stat_mask().ravel()[flat],
                                                        dict(S._atlas_names()), flat, mask.shape))["split"]
    for k in ("ALM_left", "ALM_right", "tjM1_left", "tjM1_right"):
        f = np.zeros(mask.size, bool)
        f[flat[split[k]]] = True
        C._ROI_OUTLINES.append(f.reshape(mask.shape))
    t, _ = TR.times()
    win_idx = [np.flatnonzero((t >= a) & (t < b)) for a, b in WINDOWS]
    resp = np.flatnonzero((t >= RESP[0]) & (t < RESP[1]))
    replot = "--replot" in sys.argv                                    # maps only, from the saved maps.npz
    if replot:
        z = np.load(OUT / "maps.npz")
        res = {tuple(k.split("__")): (z[k], z[f"{k}__sig"] if z[f"{k}__sig"].any() else None)
               for k in z.files if not k.endswith("__sig")}
    else:
        data, _ = S.load_sessions()
        # per session panels (K vectors), in units of the animal's pre pixel RMS
        panels = {}                                                         # (panel, epoch) -> {animal: [K vectors]}

        def put(key, ep, an, v):
            panels.setdefault((key, ep), {}).setdefault(an, []).append(v)
        for an, sess in sorted(data.items()):
            for _lab, ep, unit, ok, pc in session_parts(sess):
                sp = [p for i, p in enumerate(S.POS) if ok[i]]
                put("gain", ep, an, pc["gain"][:, resp].mean(axis=(0, 1)) / unit)
                put("off_a", ep, an, pc["off_a"] / unit)
                for j, p in enumerate(sp):
                    put(f"off_s|{p}", ep, an, pc["off_s"][j] / unit)
                    put(f"rest|{p}", ep, an, pc["rest"][j][resp].mean(0) / unit)
                for w, ix in zip(WINDOWS, win_idx):
                    put(f"shift|{w}", ep, an, pc["shift"][ix].mean(0) / unit)
                put("shift_tc", ep, an, pc["shift"] / unit)                      # nb x K
            print(f"  {an} done", flush=True)

        def frame(v):
            img = np.full(mask.shape, np.nan, np.float32)
            img[mask] = v
            return img

        store, res = {}, {}
        for (key, ep), by in panels.items():
            if ep == "pre" or key == "shift_tc":
                continue
            pre = panels.get((key, "pre"), {})
            ans = sorted(a for a in by if pre.get(a))
            M = np.mean([np.mean(by[a], 0) - np.mean(pre[a], 0) for a in ans], 0) @ Uc.T
            cm, lab = beta_maps.significance_contour({a: [frame(v @ Uc.T) for v in pre[a]] for a in ans},
                                                     {a: [frame(v @ Uc.T) for v in by[a]] for a in ans},
                                                     n_boot=2000, normalise=False)
            res[(key, ep)] = (M, cm)
            store[f"{key}__{ep}"] = M.astype(np.float32)
            store[f"{key}__{ep}__sig"] = np.zeros(mask.shape, bool) if cm is None else cm
            print(f"  {key} {ep}: {lab or 'none'}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "maps.npz", **store)
    xf = {"": "", "gsr": "; cortex-wide mode removed"}.get(S.XFORM, f"; {S.XFORM}")
    sub = (f"first-lick-aligned, cohort SVD basis{xf}; change vs own pre, units of each animal's pre-template pixel RMS, "
           f"mean of 4 animals. Contour: nested bootstrap animals -> sessions (2,000), 8-px blocks, max-stat per panel. "
           f"Image left = ipsilesional; dashed = ALM / tjM1; white = not analyzed (glue, bulbs).")

    def grid(rows, cols, keyf, title, name, coltitle):
        fig, axs = plt.subplots(len(rows), len(cols), figsize=(2.3 * len(cols), 2.5 * len(rows) + 0.4), squeeze=False)
        vals = [res[keyf(r, c)][0] for r in rows for c in cols if keyf(r, c) in res]
        vmax = np.nanpercentile(np.abs(np.concatenate(vals)), 99) if vals else 1
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                ax = axs[i, j]
                k = keyf(r, c)
                if k not in res:
                    ax.axis("off")
                    continue
                M, cm = res[k]
                C.draw(ax, M, mask, edges, vmax, cm, brain)
                if i == 0:
                    ax.set_title(coltitle(c), fontsize=10)
                if j == 0:
                    ax.set_ylabel(r, fontsize=11)
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax))
        fig.colorbar(sm, ax=axs, fraction=0.015, pad=0.01, label="change vs own pre (pre pixel RMS)")
        fig.suptitle(C._wrap(f"{title} ({sub})"), y=1.02, fontsize=9.5)
        save_png_svg(fig, FIGS / f"{name}{S.tag()}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    grid(EPS, WINDOWS, lambda r, c: (f"shift|{c}", r),
         "SHARED SHIFT: the spout-independent, time-varying part of the change, by time from the first lick", "ss_part_maps_shift",
         lambda c: f"{c[0]:+.1f}..{c[1]:+.1f} s")
    grid(EPS, ["gain", "off_a"], lambda r, c: (c, r),
         "GAIN (what the fitted gain change adds, 0.1..0.5 s, mean over spouts) and PER-AREA OFFSET (static level, all "
         "spouts)", "ss_part_maps_static", lambda c: {"gain": "gain adds (0.1..0.5 s)", "off_a": "per-area offset"}[c])
    grid(EPS, S.POS, lambda r, c: (f"off_s|{c}", r), "PER-SPOUT OFFSET: each spout's static level beyond the per-area "
         "offset", "ss_part_maps_spout_offset", lambda c: c.replace("_", " "))
    grid(EPS, S.POS, lambda r, c: (f"rest|{c}", r), "REST: spout-specific, time-varying change beyond gain, offsets and "
         "shared shift (0.1..0.5 s)", "ss_part_maps_rest", lambda c: c.replace("_", " "))
    if replot:                                                          # the time courses need the panels
        return
    # shared-shift time courses in split-basis areas
    names = [n for n in split if n.rsplit("_", 1)[0] in ROIS]
    W = np.stack([Uc[split[n]].mean(0) for n in names], axis=1)        # K x areas
    fig, axs = plt.subplots(2, 5, figsize=(17, 6.2), sharex=True, sharey=True)
    for ax, roi in zip(axs.ravel(), ROIS):
        for side, ls in (("left", "--"), ("right", "-")):
            n = f"{roi}_{side}"
            if n not in names:
                continue
            j = names.index(n)
            for ep in EPS:
                by, pre = panels.get(("shift_tc", ep), {}), panels.get(("shift_tc", "pre"), {})
                tc = [np.mean(by[a], 0) @ W[:, j] - np.mean(pre[a], 0) @ W[:, j] for a in sorted(by) if pre.get(a)]
                if tc:
                    ax.plot(t, np.mean(tc, 0), ls, color=EPOCH_LINE[ep], lw=1.6, label=f"{ep} {'ipsi' if side == 'left' else 'contra'}")
        ax.axhline(0, color="0.6", lw=0.7)
        ax.axvline(0, color="0.5", lw=0.7, ls=":")
        ax.set_title(roi)
    axs[0, 0].legend(frameon=False, fontsize=7)
    for ax in axs[1]:
        ax.set_xlabel("time from first lick (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("shared shift (pre pixel RMS)")
    fig.suptitle(C._wrap("SHARED SHIFT over time in split-basis areas (solid = contralesional, dashed = ipsilesional; "
                         f"mean of 4 animals; {sub})"), y=1.03, fontsize=9.5)
    fig.tight_layout()
    save_png_svg(fig, FIGS / f"ss_part_maps_shift_tc{S.tag()}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
