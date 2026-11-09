"""WHERE and WHEN is the shared shift carried? (Priya 11-09: "did you do the 'where is the shift carried' analysis?" ->
"spatial and temporal change").

The shared shift = the spout-independent, time-varying part of each session's change vs its animal's pre template, from
the sequential fit of `state_space_part_maps.pieces` (gain -> per-area offset -> per-spout offset -> shared shift; pre
sessions leave-one-out = the null). Cohort SVD basis (one pixel basis for all animals), first-lick-aligned (17 bins of
0.1 s, -1.0..0.7 s), WF_SS_XFORM="" (raw) or gsr (each animal's pre cortex-wide mode removed first). Units: each animal's
pre-template pixel RMS (as the part maps).

Per session, from the shift's pixel movie (bins x brain pixels):
  WHERE x WHEN  area-mean shift per split-basis area (ROI v8; 10 areas x 2 hemispheres) x bin, signed
  WHERE         each area's share of the shift's energy (sum of squares over its pixels / over all pixels), divided by
                the area's share of pixels = ENRICHMENT (1 = carried in proportion to size), in four windows: all bins,
                before the lick (-1.0..-0.1 s), around it (-0.1..+0.4 s), late (+0.4..+0.7 s)
  WHEN          the shift's energy per bin (mean over brain pixels of shift^2, pre pixel RMS^2), and its share per bin
Change = epoch sessions vs the animal's pre (leave-one-out) sessions, animal-weighted nested bootstrap
(`analysis_kit.boot_delta_columns`, 2,000 draws). Heat-map cells: max-statistic family-wise over all cells of an epoch
panel (centered draws / their SD; 95th percentile of the max |t|). Area / bin dots: * = 95 % CI excludes 0, uncorrected,
k/4 = animals changing in the cohort direction. Pre sessions are themselves a leave-one-out change, so noise enters both
arms; a pre-null enrichment is not exactly 1 (noise is not spread in proportion to size).

REFERENCE (WF_SS_SHIFTBASE; Priya 11-09 "couldn't we use the rest or pre-cue lick-free ENL as a per session
normalization?"). Without one (default ""), the shift has ZERO MEAN OVER THE WINDOW in every pixel by construction (the
static level goes to the per-area offset), so "below pre before the lick" partly mirrors "above pre after it". With
  rest  every trial minus its session's RESTnl level (`state_space_rest_baseline.py`, as Fig 9.1's maps)
  enl   every trial minus ITS OWN lick-free (>= 1 s) pre-cue ENL window (`state_space_enl_baseline.py`, as Fig 9.5;
        trials without one dropped; the window can reach past the cue on short-latency trials)
subtracted BEFORE the templates (under the loader's WF_SS_XFORM, as `state_space_chronic_maps`), the readout is the
SHARED CHANGE = per-area offset + shift = the spout-mean of the change beyond gain, which keeps its level: each bin is
a change against the reference, not against the window mean. Sessions without the reference are dropped.
Outputs: <state_space_recovery>/shift_carried{tag}/{area_bin,where,when}.csv,
page_figs/ss_shift_{where_when,where,when}{tag}.png. EXPLORATORY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_chronic_maps as C  # noqa: E402
import state_space_part_maps as PM  # noqa: E402
import state_space_sim as S  # noqa: E402
import state_space_trajectories as TR  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402
from wfield_local.orofacial_cohort import EPOCH_LINE  # noqa: E402

EPS = ["acute", "subacute", "chronic"]
SHIFTBASE = __import__("os").environ.get("WF_SS_SHIFTBASE", "")
if SHIFTBASE not in ("", "rest", "enl"):
    raise ValueError(f"WF_SS_SHIFTBASE={SHIFTBASE!r}")
BSUF = f"_b{SHIFTBASE}" if SHIFTBASE else ""
OUT = S.ROOT / f"shift_carried{S.tag()}{BSUF}"
FIGS = S.ROOT / "page_figs"
ROIS = PM.ROIS
WINS = {"all bins": (-1.0, 0.7), "before the lick (-1.0..-0.1 s)": (-1.0, -0.1),
        "around the lick (-0.1..+0.4 s)": (-0.1, 0.4), "late (+0.4..+0.7 s)": (0.4, 0.7)}
N_BOOT = 2000


def area_names(split):
    """contralesional (image right) block, then ipsilesional, each in ROI order."""
    return [f"{r}_{side}" for side in ("right", "left") for r in ROIS if f"{r}_{side}" in split]


def label(n):
    r, side = n.rsplit("_", 1)
    return f"{r} {'contra' if side == 'right' else 'ipsi'}"


def rebase(an, sess):
    """Sessions with every trial minus the WF_SS_SHIFTBASE reference (module docstring); sessions without it dropped."""
    if not SHIFTBASE:
        return sess
    out = []
    for lab, ep, day, Y, pos, *rest in sess:
        if SHIFTBASE == "rest":
            c = C._rest(lab, an) if (C.REST / f"{lab}.npz").exists() else None
            if c is None:
                continue
            out.append((lab, ep, day, Y - c, pos, *rest))
        else:
            f = C.ENL / f"{lab}.npz"
            E = np.load(f)["b"].astype(float) if f.exists() else None
            if E is None or len(E) != len(pos):
                continue
            keep = np.isfinite(E).all(1)
            E = C._xf_vectors(E[keep], pos[keep], an)
            out.append((lab, ep, day, Y[keep] - E[:, None, :], pos[keep], *rest))
    return out


def session_rows(Uc, split, names, t):
    """Per session: dict of the three readouts. Pixel movie = shift (nb x K) @ Uc.T."""
    data, _ = S.load_sessions()
    W = np.stack([Uc[split[n]].mean(0) for n in names], axis=1)               # K x areas: area-mean readout
    npix = Uc.shape[0]
    frac = np.array([len(split[n]) / npix for n in names])
    win_ix = {w: np.flatnonzero((t >= a - 1e-9) & (t < b - 1e-9)) for w, (a, b) in WINS.items()}
    rows = []
    for an, sess in sorted(data.items()):
        for lab, ep, unit, _ok, pc in PM.session_parts(rebase(an, sess)):
            sh = (pc["shift"] + pc["off_a"] if SHIFTBASE else pc["shift"]) / unit                                             # nb x K
            mov = sh @ Uc.T                                                      # nb x pixels
            e_bin = (mov ** 2).mean(1)                                          # nb
            r = {"animal": an, "label": lab, "epoch": ep, "area_bin": sh @ W, "e_bin": e_bin,
                 "share_bin": e_bin / e_bin.sum()}
            for w, ix in win_ix.items():
                tot = (mov[ix] ** 2).sum()
                e_area = np.array([(mov[ix][:, split[n]] ** 2).sum() for n in names])
                r[f"enrich|{w}"] = e_area / tot / frac
            rows.append(r)
        print(f"  {an} done", flush=True)
    return rows


def boot(rows, key, ep, rng):
    post = {}
    pre = {}
    for r in rows:
        v = np.ravel(r[key])
        if r["epoch"] == ep:
            post.setdefault(r["animal"], []).append(v)
        elif r["epoch"] == "pre":
            pre.setdefault(r["animal"], []).append(v)
    res = kit.boot_delta_columns(post, pre, rng, n_boot=N_BOOT)
    if res is None:
        return None
    pt, lo, hi, draws, n = res
    an = sorted(set(post) & set(pre))
    dm = np.array([np.mean(post[a], 0) - np.mean(pre[a], 0) for a in an])
    up = (np.sign(dm) == np.sign(pt)).sum(0)
    return {"pt": pt, "lo": lo, "hi": hi, "draws": draws, "n": len(an), "up": up}


def maxstat(b):
    """Family-wise cells: |pt / sd| above the 95th percentile of max_c |(draw - pt) / sd| over cells."""
    sd = b["draws"].std(0)
    sd[sd == 0] = np.inf
    tmax = np.abs((b["draws"] - b["pt"]) / sd).max(1)
    return np.abs(b["pt"] / sd) > np.percentile(tmax, 95)


def main():
    if S.BASIS != "svd":
        sys.exit("state_space_shift_carried: needs the cohort SVD basis (WF_SS_BASIS=svd)")
    from wfield_local import area_basis, beta_maps
    Uc = np.load(C.BASIS_DIR / "U_cohort.npy").astype(np.float64)
    PM.NPIX = Uc.shape[0]
    atlas = np.load(S.SVD.parents[1] / "atlas.npy")
    mask = np.load(C.BASIS_DIR / "mask.npy").reshape(atlas.shape)
    flat = np.flatnonzero(mask.ravel())
    split = area_basis.basis_masks(area_basis.roi_masks(atlas.ravel()[flat], beta_maps.stat_mask().ravel()[flat],
                                                        dict(S._atlas_names()), flat, mask.shape))["split"]
    names = area_names(split)
    t, _ = TR.times()
    rows = session_rows(Uc, split, names, t)
    rng = np.random.default_rng(0)
    nb, na = len(t), len(names)
    xf = {"": "", "gsr": "; cortex-wide mode removed"}.get(S.XFORM, f"; {S.XFORM}")
    what = ({"": "shared shift = spout-independent, time-varying part of each session's change vs own pre (sequential "
                 "fit as the part maps; zero window mean per pixel)",
             "rest": "SHARED CHANGE vs the session's RESTnl level = per-area offset + shift (spout-mean change beyond "
                     "gain, level kept)",
             "enl": "SHARED CHANGE vs each trial's lick-free pre-cue ENL window = per-area offset + shift (spout-mean "
                    "change beyond gain, level kept)"}[SHIFTBASE])
    sub = (f"first-lick-aligned, cohort SVD basis{xf}; {what}; change vs own pre (pre sessions "
           f"leave-one-out), mean of 4 animals, units of each animal's pre-template pixel RMS")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- WHERE x WHEN heat map
    B = {ep: boot(rows, "area_bin", ep, rng) for ep in EPS}
    vmax = np.nanpercentile(np.abs(np.concatenate([B[e]["pt"] for e in EPS if B[e]])), 99)
    fig, axs = plt.subplots(1, 3, figsize=(15, 6.4), sharey=True)
    recs = []
    for ax, ep in zip(axs, EPS):
        b = B[ep]
        M = b["pt"].reshape(nb, na).T
        sig = maxstat(b).reshape(nb, na).T
        dt = t[1] - t[0]
        ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
                  extent=[t[0] - dt / 2, t[-1] + dt / 2, na - 0.5, -0.5])
        yy, xx = np.nonzero(sig)
        ax.scatter(t[xx], yy, s=9, c="k", marker="o", lw=0)
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.axhline(len([n for n in names if n.endswith("_right")]) - 0.5, color="k", lw=0.8)
        ax.set_title(f"{ep} (dots: family-wise over cells)")
        ax.set_xlabel("time from first lick (s)")
        for i, n in enumerate(names):
            for j in range(nb):
                recs.append({"epoch": ep, "area": label(n), "t": round(float(t[j]), 2), "change": M[i, j],
                             "lo": b["lo"].reshape(nb, na)[j, i], "hi": b["hi"].reshape(nb, na)[j, i],
                             "k_up": int(b["up"].reshape(nb, na)[j, i]), "n": b["n"], "fwer_sig": bool(sig[i, j])})
    axs[0].set_yticks(range(na), [label(n) for n in names], fontsize=8)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, ax=axs, fraction=0.02, pad=0.01, label="area-mean shared shift, change vs own pre (pre pixel RMS)")
    fig.suptitle(C._wrap(f"WHERE x WHEN the shared shift is carried: split-basis area x time ({sub}; dot = cell whose "
                         f"nested-bootstrap change (animals -> sessions, 2,000) survives the max statistic over all "
                         f"cells of the panel)"), y=1.04, fontsize=9.5)
    save_png_svg(fig, FIGS / f"ss_shift_where_when{S.tag()}{BSUF}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(recs).to_csv(OUT / "area_bin.csv", index=False)

    # ---- WHERE: enrichment per area, four windows
    fig, axs = plt.subplots(len(WINS), 1, figsize=(15, 3.1 * len(WINS)), sharex=True)
    recs = []
    for ax, w in zip(axs, WINS):
        for k, ep in enumerate(EPS):
            b = boot(rows, f"enrich|{w}", ep, rng)
            x = np.arange(na) + (k - 1) * 0.25
            ax.errorbar(x, b["pt"], yerr=[b["pt"] - b["lo"], b["hi"] - b["pt"]], fmt="o", color=EPOCH_LINE[ep], ms=4,
                        lw=1.3, capsize=0, label=ep)
            for i in range(na):
                if b["lo"][i] > 0 or b["hi"][i] < 0:
                    sig_marks.star_top(ax, x[i], "*", color=EPOCH_LINE[ep], fontsize=9)
                recs.append({"window": w, "epoch": ep, "area": label(names[i]), "change": b["pt"][i], "lo": b["lo"][i],
                             "hi": b["hi"][i], "k_up": int(b["up"][i]), "n": b["n"]})
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(len([n for n in names if n.endswith("_right")]) - 0.5, color="0.4", lw=0.8)
        ax.set_ylabel("enrichment change\nvs own pre")
        ax.set_title(w, fontsize=10)
    axs[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    axs[-1].set_xticks(range(na), [label(n) for n in names], rotation=60, ha="right", fontsize=8)
    fig.suptitle(C._wrap(f"WHERE the shared shift is carried: each area's share of the shift's energy / its share of "
                         f"pixels (1 = in proportion to size), change vs own pre ({sub}; * = 95 % CI excludes 0, "
                         f"uncorrected)"), y=1.01, fontsize=9.5)
    sig_marks.reserve_headroom(fig)
    save_png_svg(fig, FIGS / f"ss_shift_where{S.tag()}{BSUF}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(recs).to_csv(OUT / "where.csv", index=False)

    # ---- WHEN: energy per bin and share per bin
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    recs = []
    for ax, key, yl in ((axs[0], "e_bin", "shift energy per bin, change vs own pre\n(pre pixel RMS$^2$)"),
                        (axs[1], "share_bin", "share of the shift's energy per bin,\nchange vs own pre")):
        for ep in EPS:
            b = boot(rows, key, ep, rng)
            ax.plot(t, b["pt"], color=EPOCH_LINE[ep], lw=1.8, label=ep)
            ax.fill_between(t, b["lo"], b["hi"], color=EPOCH_LINE[ep], alpha=0.18, lw=0)
            sig = (b["lo"] > 0) | (b["hi"] < 0)
            ax.scatter(t[sig], np.full(sig.sum(), ax.get_ylim()[1]), marker="|", color=EPOCH_LINE[ep], s=30)
            for j in range(nb):
                recs.append({"readout": key, "epoch": ep, "t": round(float(t[j]), 2), "change": b["pt"][j],
                             "lo": b["lo"][j], "hi": b["hi"][j], "k_up": int(b["up"][j]), "n": b["n"]})
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(0, color="0.4", lw=0.8, ls=":")
        ax.set_xlabel("time from first lick (s)")
        ax.set_ylabel(yl)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle(C._wrap(f"WHEN the shared shift is carried: its energy per 0.1-s bin (mean over brain pixels) and that "
                         f"bin's share of the total ({sub}; band = 95 % CI; ticks on top = CI excludes 0, uncorrected)"),
                 y=1.05, fontsize=9.5)
    fig.tight_layout()
    save_png_svg(fig, FIGS / f"ss_shift_when{S.tag()}{BSUF}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(recs).to_csv(OUT / "when.csv", index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
