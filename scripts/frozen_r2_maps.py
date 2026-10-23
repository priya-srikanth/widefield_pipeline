"""PIXEL R^2 / UNIQUE R^2 MAPS of the frozen encoder from a SPATIAL-target run (`frozen_encoder --target basis | grid`;
Priya 10-23: "do we have the pixel/SVD level unique r2 maps?").

Per session x variant the run saved the all-frames residual second moment E and the data second moment S in TARGET
space (`resid_mom`, `data_mom`; drift-removed, session-centred data). A pixel's SSE = diag(W E W') and SST =
diag(W S W') with W the animal's shared basis (grid: each block's own diagonal, painted), so R^2 is that of the data
RECONSTRUCTED in the basis (200 components, >= 99.9 % of pre variance). Per animal x epoch SSE and SST are SUMMED over
sessions before the ratio (frames weigh equally); then the mean over animals in pixel space (rule 8: animals first).
Pre sessions are scored with their leave-one-session-out betas (held out), post sessions with the frozen all-pre betas.

  full R^2          1 - SSE[joint] / SST
  unique R^2 of g   (SSE[no_g] - SSE[joint]) / SST          (Musall's task-independent / unique dR^2)
  non-task unique   (SSE[task_only] - SSE[joint]) / SST

Figures: <E>/frozen_encoder_<tag>/figures/r2_maps_pre.png (full + unique per group, held-out pre) and
r2_maps_<epoch>_minus_pre.png (frozen unique R^2, epoch minus pre). Maps saved to figures/r2_maps.npz.
ipsi (lesioned LEFT hemisphere) = image LEFT; displayed at half resolution.

Usage: python scripts/frozen_r2_maps.py --tag gate-attempted_v6basis
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

EPOCHS = ("pre", "acute", "subacute", "chronic")
#: (label, variant whose SSE minus joint's is the unique part)
UNIQUE = (("lick", "no_lick"), ("spout", "no_spout"), ("target", "no_target"), ("exec_locked", "no_exec_locked"),
          ("exec_within", "no_exec_within"), ("contact", "no_contact"), ("other", "no_other"),
          ("timing", "no_timing"), ("task", "no_task"), ("movement", "no_movement"), ("non-task", "task_only"))


def pixel_sse(sp: dict, M: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """diag(W M W') over the stat pixels (basis) or diag(M) per block (grid), cortex-mean target dropped."""
    M = np.asarray(M, np.float64)[np.ix_(keep, keep)]
    if sp["kind"] == "basis":
        Wb = np.asarray(sp["P"], np.float64)
        return ((Wb @ M) * Wb).sum(1)
    return np.diag(M).copy()


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import combined_encoding as ce
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from frozen_area_maps import area_image, outline
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v6basis")
    a = ap.parse_args(argv)
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / f"frozen_encoder_{a.tag}"
    variants = ["joint"] + sorted({v for _l, v in UNIQUE})
    missing = [v for v in variants if not (out / v / "sessions").is_dir()]
    if missing:
        raise SystemExit(f"{out}: no sessions for variants {missing} (re-run the spatial target with the drop variants)")
    sess = sorted(js for js in (out / "joint" / "sessions").glob("*.json") if js.stem[-4:] >= "0801")
    fd = out / "figures"
    assert_writable(fd)
    fd.mkdir(parents=True, exist_ok=True)
    _V, _ft, _Ub, flat, _parea, _names, shape, pstat = ce.load_imaging(sess[0].stem)
    H, W = (int(x) for x in shape)
    pix = flat[pstat]
    img, _nm = area_image(sess[0].stem)                     # also checks ipsi = image left
    ol = outline(img)[::2, ::2]
    spatial: dict = {}
    acc: dict = {}                                          # (animal, epoch) -> {"sst": , variant: sse}
    nsess: dict = {}
    for js in sess:
        m = json.loads(js.read_text(encoding="utf-8"))
        an, ep = m["animal"], m["epoch"]
        if an not in spatial:
            spatial[an] = pickle.loads((out / "fit" / f"spatial_{an}.pkl").read_bytes())
        sp = spatial[an]
        keep = np.array([t != "cortex" for t in m["targets"]])
        z0 = np.load(js.with_suffix(".npz"))
        if "resid_mom" not in z0.files:
            raise SystemExit(f"{js}: no resid_mom (run predates the pixel-R2 statistics; re-run the spatial target)")
        cell = acc.setdefault((an, ep), {})
        cell["sst"] = cell.get("sst", 0.0) + pixel_sse(sp, z0["data_mom"], keep)
        for v in variants:
            z = z0 if v == "joint" else np.load(out / v / "sessions" / f"{js.stem}.npz")
            cell[v] = cell.get(v, 0.0) + pixel_sse(sp, z["resid_mom"], keep)
        nsess[(an, ep)] = nsess.get((an, ep), 0) + 1

    def paint(an, v):                                      # (targets or pixels,) -> (H, W)/2 image
        sp = spatial[an]
        g = np.full(H * W, np.nan)
        if sp["kind"] == "basis":
            g[pix] = v
        else:
            P = sp["P"].tocsc()
            blk = np.asarray(P.argmax(1)).ravel()
            has = np.asarray(P.sum(1)).ravel() > 0
            g[pix[has]] = v[blk[has]]
        return g.reshape(H, W)[::2, ::2]
    maps: dict = {}
    for (an, ep), c in acc.items():
        sst = np.where(c["sst"] > 0, c["sst"], np.nan)
        mm = {"full": 1.0 - c["joint"] / sst}
        for lab, v in UNIQUE:
            mm[lab] = (c[v] - c["joint"]) / sst
        maps[(an, ep)] = {k: paint(an, x) for k, x in mm.items()}
    animals = sorted({an for an, _ in maps})
    cols = ["full"] + [lab for lab, _v in UNIQUE]
    np.savez_compressed(fd / "r2_maps.npz", **{f"{an}|{ep}|{k}": x for (an, ep), mm in maps.items()
                                              for k, x in mm.items()})

    def grid_fig(rows: dict, title: str, path: Path, diff: bool):
        names = list(rows)
        fig, axs = plt.subplots(len(names), len(cols), figsize=(1.6 * len(cols), 1.6 * len(names) + 0.9), squeeze=False)
        lims = {}                                          # one scale PER COLUMN (Musall): small groups stay visible
        for c in cols:
            v = np.concatenate([rows[r][c].ravel() for r in names])
            lims[c] = float(np.nanpercentile(np.abs(v), 99) if diff else max(np.nanpercentile(v, 99), 1e-6))
        for i, r in enumerate(names):
            for j, c in enumerate(cols):
                ax = axs[i][j]
                lim = lims[c]
                if diff:
                    ax.imshow(rows[r][c], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                else:
                    ax.imshow(rows[r][c], cmap="viridis" if c == "full" else "magma", vmin=0, vmax=lim,
                              interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"{'full R2' if c == 'full' else 'unique ' + c}\n{'+/-' if diff else '0..'}"
                                 f"{lims[c]:.3g}", fontsize=7)
                if j == 0:
                    ax.set_ylabel(r, fontsize=8)
        sub = "colour scale per column (99th percentile, in each title)"
        fig.suptitle(f"{title}\n{sub}. ipsi (lesion) = image LEFT. PROVISIONAL", fontsize=8)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    kind = next(iter(spatial.values()))["kind"]
    rows = {f"{an} (n={nsess[(an, 'pre')]})": maps[(an, "pre")] for an in animals if (an, "pre") in maps}
    rows["mean"] = {c: np.nanmean([maps[(an, "pre")][c] for an in animals if (an, "pre") in maps], 0) for c in cols}
    grid_fig(rows, f"Frozen encoder ({kind} targets): pixel R2, PRE-STROKE held-out sessions (leave-one-session-out), "
                   "all frames; unique = joint - drop-group", fd / "r2_maps_pre.png", diff=False)
    for ep in EPOCHS[1:]:
        rows = {}
        for an in animals:
            if (an, "pre") in maps and (an, ep) in maps:
                rows[f"{an} (n={nsess[(an, ep)]})"] = {c: maps[(an, ep)][c] - maps[(an, "pre")][c] for c in cols}
        if not rows:
            continue
        rows["mean"] = {c: np.nanmean([rows[r][c] for r in rows], 0) for c in cols}
        grid_fig(rows, f"Frozen encoder ({kind} targets): pixel R2, {ep} (frozen pre-stroke betas) MINUS pre "
                       "(held out), all frames", fd / f"r2_maps_{ep}_minus_pre.png", diff=True)
    # PER-POSITION unique R2 (frozen drop-one-spout): spout q's own target-position response folded into the others'
    # mean; dR2_q = (SSE[base|drop:q] - SSE[base]) / SST on q's OWN trial frames, summed over sessions per animal x
    # epoch (Priya 10-23 "and what about per-position unique r2")
    dd = out / "dropspout" / "sessions"
    pj = sorted(dd.glob("*_pix.json")) if dd.is_dir() else []
    pj = [p for p in pj if p.stem.split("_pix")[0][-4:] >= "0801"]
    if pj:
        pacc: dict = {}
        for p in pj:
            m = json.loads(p.read_text(encoding="utf-8"))
            an, ep = m["animal"], m["epoch"]
            sp = spatial.get(an) or pickle.loads((out / "fit" / f"spatial_{an}.pkl").read_bytes())
            spatial[an] = sp
            keep = np.array([t != "cortex" for t in m["targets"]])
            z = np.load(p.with_suffix(".npz"))
            for k in z.files:
                parts = k.split("|")
                if len(parts) != 4 or parts[2] != parts[1]:           # own-trial subsets only
                    continue
                base, q, _sub, tag = parts
                c = pacc.setdefault((base, an, ep, q), {})
                c[tag] = c.get(tag, 0.0) + pixel_sse(sp, z[k], keep)
                if tag == "base":
                    c["sst"] = c.get("sst", 0.0) + pixel_sse(sp, z[f"data|{q}"], keep)
        qs = sorted({k[3] for k in pacc})
        for base in sorted({k[0] for k in pacc}):
            pm = {}
            for (b_, an, ep, q), c in pacc.items():
                if b_ == base and {"base", "drop", "sst"} <= set(c):
                    pm[(an, ep, q)] = paint(an, (c["drop"] - c["base"]) / np.where(c["sst"] > 0, c["sst"], np.nan))
            for ep in EPOCHS:
                rows = {}
                for an in animals:
                    if not all((an, ep, q) in pm for q in qs) or (ep != "pre" and not all((an, "pre", q) in pm for q in qs)):
                        continue
                    rows[an] = {q: pm[(an, ep, q)] - (pm[(an, "pre", q)] if ep != "pre" else 0.0) for q in qs}
                if not rows:
                    continue
                rows["mean"] = {q: np.nanmean([rows[r][q] for r in rows], 0) for q in qs}
                vals = np.concatenate([rows[r][q].ravel() for r in rows for q in qs])
                diff = ep != "pre"
                lim = np.nanpercentile(np.abs(vals), 99) if diff else np.nanpercentile(vals, 99)
                fig, axs = plt.subplots(len(rows), len(qs), figsize=(1.7 * len(qs), 1.7 * len(rows) + 0.9),
                                        squeeze=False)
                for i, r in enumerate(rows):
                    for j, q in enumerate(qs):
                        ax = axs[i][j]
                        ax.imshow(rows[r][q], cmap="RdBu_r" if diff else "magma", vmin=-lim if diff else 0, vmax=lim,
                                  interpolation="nearest")
                        ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if i == 0:
                            ax.set_title(q, fontsize=8)
                        if j == 0:
                            ax.set_ylabel(r, fontsize=8)
                what = "held-out pre" if not diff else f"{ep} (frozen) minus pre"
                fig.suptitle(f"Frozen encoder ({kind} targets, base {base}): PER-POSITION unique R2 of the target "
                             f"position response, on that spout's own trials, {what}; scale "
                             f"{'+/-' if diff else '0..'}{lim:.3f}. ipsi (lesion) = image LEFT. PROVISIONAL", fontsize=8)
                fig.savefig(fd / f"r2_maps_position_{base}_{ep}{'_minus_pre' if diff else ''}.png", dpi=120,
                            bbox_inches="tight")
                plt.close(fig)
    print(f"-> {fd} (r2_maps_*.png, r2_maps.npz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
