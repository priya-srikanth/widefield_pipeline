"""Desktop stage `report`: rank sweep, stability, leave-one-out and the factor gallery."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import EPS
from .cp import match_sets
from .files import _sidecar_module, load_master, load_plan


def direction_colours(Wk: np.ndarray) -> np.ndarray:
    """(P, 3) RGB per point from a factor's x / y spatial weights ``Wk`` (2, P): hue by the movement direction (Rich's
    convention, as the factor clips): blue = lateral (x), red = vertical (y)."""
    from matplotlib.colors import hsv_to_rgb
    ang = np.arctan2(np.abs(Wk[1]), np.abs(Wk[0])) / (np.pi / 2)              # 0 lateral .. 1 vertical
    return hsv_to_rgb(np.stack([(2 / 3) * (1 - ang), np.ones_like(ang), np.full_like(ang, 0.95)], -1))


def position_style(pos: str) -> dict:
    """Line style per spout position: left blue / right red / centre grey; close solid, far dashed."""
    c = "tab:blue" if pos.endswith("_L") else "tab:red" if pos.endswith("_R") else "0.4"
    return {"color": c, "ls": "-" if pos.startswith("close") else "--"}


def cmd_report(sd: Path, face: Path | None = None, absolute: bool = False, match: Path | None = None) -> None:
    """Factor-set QC into ``<set>/report/``: the rank sweep (rank*.json), the leave-one-day-out tables and, once frozen,
    the factor gallery and the epoch figure. Methods for every figure: ``report/README.md`` (`face_rhythm_figures`).
    Session readouts (epochs, EVR gap, morphing, movement maps) are `scripts/face_rhythm_readouts.py`. ``absolute``:
    time courses as absolute expression (weights x FR normaliser D) -> *_abs.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from wfield_local import face_rhythm_figures as frf
    rd = sd / "report"
    rd.mkdir(exist_ok=True)
    pl = load_plan(sd)
    role = pl.get("role", "master")
    S = [json.loads(f.read_text()) for f in sorted(sd.glob("rank*.json"))]
    L = pd.DataFrame([json.loads(f.read_text()) for f in sorted((sd / "loso").glob("rank*.json"))]) \
        if (sd / "loso").exists() else pd.DataFrame()
    if S:
        rk = [x["rank"] for x in S]
        loao = [x for x in S if x.get("loao")]
        with frf.style():
            fig, ax = frf.subplots(1, 4, figsize=(19, 4.6))
            ax[0].plot(rk, [x["evr_uncentred"] for x in S], "o-")
            ax[0].set(xlabel="rank", ylabel="EVR (uncentred)", title="fit")
            ax[1].plot(rk, [x["stability_median"] or np.nan for x in S], "o-", label="whole fit (median)")
            pf = [(x["rank"], v) for x in S for v in (x.get("stability_per_factor_min") or [])]
            if pf:                                       # one dot per factor: its WORST match over the other starts
                ax[1].plot(*np.array(pf).T, "k.", ms=4, alpha=0.5, label="per factor (min)")
                frf.legend_outside(ax[1])
                pd.DataFrame([{"rank": x["rank"], "factor": k, "stability_median": m, "stability_min": n}
                              for x in S if x.get("stability_per_factor_min")
                              for k, (m, n) in enumerate(zip(x["stability_per_factor_median"],
                                                             x["stability_per_factor_min"]))]
                             ).to_csv(rd / "stability_per_factor.csv", index=False, float_format="%.4f")
            ax[1].set(xlabel="rank", ylabel="similarity to best start", title="stability across starts",
                      ylim=(0, 1.02))
            for a in sorted(loao[0]["loao"]) if loao else []:
                ax[2].plot([x["rank"] for x in loao],
                           [x["loao"][a]["held_out_err"] / x["loao"][a]["in_fit_err"] for x in loao], "o-",
                           label=f"{a} left out")
            ax[2].axhline(1, color="k", lw=0.6)
            ax[2].set(xlabel="rank", ylabel="held-out / in-fit error", title="leave one animal out")
            if loao:
                frf.legend_outside(ax[2])
            if len(L):
                first = sorted(L.animal.unique())[0]
                for an, g in L.groupby("animal"):
                    for _d, gd in g.groupby("day"):
                        gd = gd.sort_values("rank")
                        ax[3].plot(gd["rank"], gd["evr_held_out"], "o-" if gd["rank"].nunique() == 1 else "-",
                                   lw=0.6, ms=3, alpha=0.6, color="tab:blue" if an == first else "tab:orange")
                md = L.groupby("rank")[["evr_held_out", "evr_in_fit"]].median()
                ax[3].plot(md.index, md.evr_in_fit, "k_", ms=18, mew=2, label="in fit (median)")
                ax[3].plot(md.index, md.evr_held_out, "kx", ms=10, mew=2, label="held out (median)")
                ax[3].set(xlabel="rank", ylabel="EVR on the day's session rows", title="leave one pre day out")
                frf.legend_outside(ax[3])
                L.sort_values(["rank", "animal", "day"]).to_csv(rd / "loso.csv", index=False, float_format="%.4f")
                (L.assign(gap=L.evr_in_fit - L.evr_held_out).groupby("rank")
                 .agg(n_days=("day", "size"), held_out_median=("evr_held_out", "median"),
                      held_out_min=("evr_held_out", "min"), in_fit_median=("evr_in_fit", "median"),
                      gap_median=("gap", "median"), gap_max=("gap", "max"),
                      similarity_median=("similarity_to_full", "median"))
                 .to_csv(rd / "loso_by_rank.csv", float_format="%.4f"))
            fig.suptitle(f"{sd.name} [{role}]: rank sweep")
            frf.save(fig, rd / "rank_sweep.png", f"""
Fit of the non-negative CP (HALS) on the set's pre-stroke fit rows (`face_rhythm_factors.py fit`), one point per rank.
* **fit**: uncentred EVR = 1 - |S - S^|^2 / |S|^2 of the best of the random starts (the paper's definition).
* **stability**: median similarity of the other starts to the best one (Williams et al. 2018: matched components, product
  of spatial x spectral cosines). Values well below 1 = the rank splits noise differently on each start. Dots (fits
  run since 10-22): each factor of the best start, its WORST matched similarity over the other starts -- which factors
  are reproducible at this rank (table `stability_per_factor.csv`, factor = the best start's index).
* **leave one animal out** (ranks fitted with it): refit without one animal, project its rows with spatial + spectral
  fixed; held-out / in-fit relative error. With two animals this is "fit one, project the other" (weak).
* **leave one pre day out** (`loso`): refit without one pre-stroke DAY's fit rows; that day's SESSION rows (the rows
  `fitsession` scores every session on) projected on the held-out fit (lines / x) and on the full fit (bar). The
  held-out pre EVR is the noise floor post-stroke frozen EVRs are read against; tables `loso.csv`, `loso_by_rank.csv`.
Plan: {json.dumps({k: pl['params'][k] for k in ('trials_per_animal', 'between_frac', 'inits', 'n_iter') if k in pl['params']})};
animals {', '.join(sorted(pl['fit']))}. Rank sweep fits may skip the animal-out refits (`fit --no-loao`).""")
        pd.DataFrame([{k: v for k, v in x.items() if k not in ("loao", "stability", "rel_errs", "stability_per_factor_median",
                                                                   "stability_per_factor_min")} for x in S]
                     ).to_csv(rd / "rank_sweep.csv", index=False, float_format="%.4f")
    if not (sd / "master.npz").exists():
        print(f"report: rank sweep only (not frozen yet) -> {rd}")
        return
    _, B, _, M = load_master(sd)
    Wsp, freqs, home, R = M["spatial"], M["freqs"], M["home"], int(M["rank"])
    mag = np.hypot(Wsp[0], Wsp[1])                                       # (P, R) weight per point
    vshare = ""
    if "view_P" in M:                                   # multi-view: maps on the reference face; share per view
        fmv = _sidecar_module("face_rhythm_multiview")
        sl = fmv.view_slices([int(x) for x in M["view_P"]])
        sh = np.array([(mag[q] ** 2).sum(0) for q in sl]) / np.maximum((mag ** 2).sum(0), EPS)    # (views, R)
        vshare = {k: ", ".join(f"{c} {100 * sh[i, k]:.0f}%" for i, c in enumerate(M["view_cams"])) for k in range(R)}
        Wsp, mag, home = Wsp[:, sl[0]], mag[sl[0]], home[sl[0]]
    lab = json.loads((sd / "labels.json").read_text()) if (sd / "labels.json").exists() else {}
    names = {str(k): lab.get(str(k), "") for k in range(R)}
    if match and (Path(match) / "master.npz").exists():                  # best match in another (named) set: a HINT
        Mo = np.load(Path(match) / "master.npz")
        lab_o = json.loads((Path(match) / "labels.json").read_text()) if (Path(match) / "labels.json").exists() else {}
        order, cos = match_sets([Wsp.reshape(-1, R), B], [Mo["spatial"].reshape(-1, Mo["spatial"].shape[-1]),
                                                            Mo["spectral"]])
        for k in range(R):
            hint = f"~ {Path(match).name}: no match (new)" if order[k] < 0 else \
                f"~ {Path(match).name}:{lab_o.get(str(order[k]), f'#{order[k]}')} (cos {cos[0, k]:.2f} / {cos[1, k]:.2f})"
            names[str(k)] = (names[str(k)] + " " if names[str(k)] else "") + hint
    tc, ep_tc, t_ms = {}, {}, None                     # pre by position; by (animal, epoch): one mean per session
    n_pre = set()
    for f in sorted((sd / "proj").glob("*.npz")) if (sd / "proj").exists() else []:
        z = np.load(f)
        t_ms = z["t_ms"]
        tr = z["trials"] * (z["trials_D"][..., None] if absolute else 1.0)
        ep_tc.setdefault((str(z["animal"]), str(z["epoch"])), []).append(np.nanmean(tr, 0))
        if str(z["epoch"]) != "pre":
            continue
        n_pre.add(f.stem.split("_cam")[0])
        for pos in np.unique(z["position"]):
            tc.setdefault(str(pos), []).append(np.nanmean(tr[z["position"] == pos], 0))
    unit = "absolute" if absolute else "share"
    ylab = "a.u. (weight x D)" if absolute else "a.u. (weight, share)"
    sfx = "_abs" if absolute else ""
    units_md = ("Y UNITS: arbitrary. The weight is the factor's amplitude in a bin (its spatial and spectral columns have "
                "unit norm) on the spectrogram -- VQT magnitude of point displacement x f^0.5 -- divided by the animal's "
                "ONE scale (RMS of its pre-stroke fit rows); " + (
                    "x D (the bin's FR normaliser, recovered from the saved spectrogram) undoes FR's per-frame "
                    "normalisation, so this is how MUCH the factor is expressed. " if absolute else
                    "without D it is the factor's SHARE of the frame's movement (FR normalises every frame). ") +
                "Comparable within an animal across sessions; between factors compare shape, not height.")
    img = plt.imread(face) if face and Path(face).exists() else None
    with frf.style():
        fig, axs = frf.subplots(R, 4, figsize=(20, 3.6 * R), gridspec_kw={"width_ratios": [1, 1, 1.1, 1.5]})
        for k, (aw, a0, a1, a2) in enumerate(np.atleast_2d(axs)):
            w = mag[:, k] / max(mag[:, k].max(), EPS)
            o = np.argsort(w)                                            # heaviest points drawn last
            for ax in (aw, a0):
                if img is not None:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.invert_yaxis()
                ax.set_aspect("equal")
                ax.axis("off")
            aw.scatter(home[o, 0], home[o, 1], c=w[o], s=5, cmap="magma", vmin=0, vmax=1, linewidths=0)
            aw.set_title(f"factor {k}: WHERE"                       # the name goes on the wide timecourse panel:
                         + (f"\n({str(M['view_cams'][0])}; views {vshare[k]})" if vshare else ""))  # it overflowed here
            rgba = np.column_stack([direction_colours(Wsp[..., k]), np.clip(w, 0, 1) ** 0.7])
            a0.scatter(home[o, 0], home[o, 1], c=rgba[o], s=2 + 24 * w[o], linewidths=0)
            a0.set_title("HOW (blue lateral, red vertical)")
            a1.plot(freqs, B[:, k])
            a1.set_xscale("log")
            a1.set(xlabel="Hz", ylabel="spectral weight", title=f"spectrum, peak {freqs[np.argmax(B[:, k])]:.1f} Hz")
            for pos, xs in sorted(tc.items()):
                a2.plot(t_ms / 1000, np.nanmean(xs, 0)[:, k], lw=1.2, label=pos, **position_style(pos))
            a2.axvline(0, color="k", lw=0.5)
            a2.set(xlabel="s from cue", ylabel=ylab,
                   title=(f"factor {k} {names[str(k)]}\n" if names[str(k)] else "") + "pre-stroke trials by position")
            if k == 0 and tc:
                frf.legend_outside(a2)
        fig.suptitle(f"{M['set']} [{str(M['role']) if 'role' in M else 'master'}] rank {R}: factor gallery ({unit})")
        frf.save(fig, rd / f"factor_gallery{sfx}.png", f"""
One row per factor of the frozen set (`master.npz`, spec {M['spec_hash']}).
* **WHERE**: the factor's spatial weight per mesh point, |(x, y) weights|, scaled to its max (magma), on the master face
  (`<dlc>/face_rhythm/cam4/master_face.png`; image right = ipsilesional on cam4).
* **HOW**: the direction the factor moves each point -- hue from the angle of its x / y weights (blue = lateral, red =
  vertical; Rich's convention, as the factor clips); opacity and size = the WHERE weight.
* **spectrum**: the factor's spectral weights over the 36 VQT frequencies (1-60 Hz, unit norm).
* **pre-stroke trials by position**: mean over engaged trials of the projected temporal weights, cue-aligned (25 Hz),
  per spout position (left blue / right red / centre grey; close solid, far dashed), mean over {len(n_pre)} pre-stroke
  sessions ({', '.join(sorted(n_pre))}). Pre-stroke sessions are projected on the fit that LEFT THEIR DAY OUT where it
  exists (`basis` = loso in the proj files), post-stroke ones on the full pre fit.
{units_md}""")

    eps = [e for e in ("pre", "acute", "subacute", "chronic") if any(k[1] == e for k in ep_tc)]
    if len(eps) > 1:                                    # cue-aligned by epoch, per animal (all engaged trials)
        ans = sorted({a for a, _ in ep_tc})
        col = {"pre": "k", "acute": "tab:red", "subacute": "tab:orange", "chronic": "tab:blue"}
        with frf.style():
            fig, axs = frf.subplots(R, len(ans), figsize=(5.8 * len(ans) + 1.5, 2.7 * R), squeeze=False, sharex=True)
            for k in range(R):
                for j, an in enumerate(ans):
                    ax = axs[k, j]
                    pre = ep_tc.get((an, "pre"))                # each animal x factor: pre-stroke mean curve max = 1
                    ref = float(np.nanmax(np.nanmean(pre, 0)[:, k])) if pre else np.nan
                    for e in eps:
                        xs = ep_tc.get((an, e))
                        if xs:
                            ax.plot(t_ms / 1000, np.nanmean(xs, 0)[:, k] / ref, color=col[e], lw=1.3,
                                    label=f"{e} (n={len(xs)})")
                    ax.axvline(0, color="k", lw=0.5)
                    ax.axhline(1, color="0.7", lw=0.6, ls=":")
                    ax.set_title(f"{an}, factor {k}{' ' + names[str(k)] if names[str(k)] else ''}")
                    if j == 0:
                        ax.set_ylabel("rel. to pre max")
                    if k == R - 1:
                        ax.set_xlabel("s from cue")
                    if k == 0 and j == len(ans) - 1:
                        frf.legend_outside(ax)
            fig.suptitle(f"{M['set']}: cue-aligned expression by epoch ({unit}, pre max = 1)")
            frf.save(fig, rd / f"factor_epochs{sfx}.png", f"""
Per animal (columns) and factor (rows): the cue-aligned mean projected expression of each epoch (pre / acute /
subacute / chronic, `wfield_local.epochs.epoch_of`; n = sessions). Per session the mean over ALL its engaged trials
(every spout position), then the mean over the epoch's sessions. NORMALISED per animal x factor: divided by the maximum
of that animal's pre-stroke mean curve, so pre peaks at 1 (dotted line) and the other epochs read as fractions of it.
Pre-stroke sessions on their left-out-day basis, post-stroke on the full pre fit. {units_md}""")
    print(f"report -> {rd}")
