"""FaceRhythm session readouts: tables + figures for a frozen factor set (computation in
`wfield_local/face_rhythm_readouts.py`; pipeline: docs/FACE_RHYTHM_PIPELINE.md).

    python scripts/face_rhythm_readouts.py [--set master_pre_v1] [--bundle fr_cam4_20261021] [--band all] [--out DIR]

Needs ``<set>/master.npz`` + ``proj/`` (and ``own/`` for the EVR gap / morphing). Writes to
``<dlc>/face_rhythm/cam4/readouts/<set>/`` (default; CSVs in its ``data/`` subfolder, `figure_layout.sidecar_for`):
  components.csv           per frozen component: peak Hz, face-region weights, lick-rate correlation, response /
                           session ratio, far-L vs far-R selectivity, SUGGESTED name; the suggestions also go to
                           <set>/labels_suggested.json -- copy to <set>/labels.json once checked against the gallery
                           (labels.json names the loader's columns; it is Priya's call, never written here)
  session_table.csv        one row per session: amplitudes per component x window, reconstruction error, EVR own /
                           frozen / gap, matched cosines, contra / ipsi movement power + asymmetry; animal / epoch / day
  position_amplitudes.csv  response-window amplitude per session x position x component
  epoch_means.csv / epoch_deltas.csv   per-animal epoch means; paired animal-weighted change vs pre (boot_delta)
  amplitudes_by_epoch.png  per component: response / between amplitude per animal across epochs
  recovery_by_day.png      reconstruction error, EVR gap, spatial / spectral morphing vs days since stroke
  movement_maps_<measure>_<window>.png   log2(epoch / pre) per point on the master face, per animal + animal mean
                           (speed = absolute movement; share = normalised spectral distribution)
  asymmetry_by_epoch.png   (contra - ipsi) / (contra + ipsi) per window, speed and share (cam4 ipsi = image right)
  task_vs_nontask.png      lick components (response) vs the others (between trials) / pre, per animal x epoch
  task_vs_nontask_pooled.png  the same, log2 vs each animal's own pre, pooled: mean +- SEM of animals, boot_delta vs
                           pre (* = CI excludes 0), plus task - non-task (`--pooled-only` redraws from session_table.csv)
  severity.png             per animal: FaceRhythm deficit vs histological lesion burden (descriptive, n = 4)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wfield_local.face_rhythm import common as frc

EPOCHS = frc.EPOCHS


def _epoch_lines(ax, S, col, title):
    from wfield_local import config
    for a, g in S.groupby("animal"):
        m = g.groupby("epoch")[col].mean().reindex(EPOCHS)
        ax.plot(range(4), m.to_numpy(), "o-", ms=3, lw=1, label=a, color=config.animal_color().get(a))
    ax.set_xticks(range(4), [e[:5] for e in EPOCHS])
    ax.set_title(title)


def task_nontask_pooled(S, names, out: Path, set_name: str, n_boot: int | None = None):
    """Cross-animal version of task_vs_nontask.png (Priya 10-25): per session, TASK (lick components, response window)
    and NON-TASK (other components, between trials) as log2 vs the animal's own pre sessions (`stats.log2_vs_pre`, the
    within-animal normalisation; pre averages 0), and their difference task - non-task (log2 task / non-task ratio vs
    pre: > 0 = task held up better, < 0 = non-task preserved relative to task). Per epoch: per-animal means (PRINTED
    first, rule 8), mean +- SEM over animals, `stats.epoch_change` (`boot_delta`, paired animals -> sessions) vs pre.
    -> task_vs_nontask_pooled.png + data/task_vs_nontask_pooled.csv; returns the stats table."""
    import pandas as pd

    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local import figure_layout
    from wfield_local.face_rhythm import stats as frs
    lick, other = frr.lick_split(names)
    T = S[["label", "animal", "epoch"]].copy()
    T["task"] = S[[f"abs_{n}_response" for n in lick]].sum(1, min_count=len(lick))
    T["nontask"] = S[[f"abs_{n}_between" for n in other]].sum(1, min_count=len(other))
    T = T.dropna(subset=["task", "nontask"]).sort_values(["animal", "label"]).reset_index(drop=True)
    for c in ("task", "nontask"):
        T[f"{c}_l2"] = frs.log2_vs_pre(T, c, ["animal"])
    T["diff_l2"] = T.task_l2 - T.nontask_l2
    Q = {"task_l2": "TASK: lick factors, response", "nontask_l2": "NON-TASK: other factors, between",
         "diff_l2": "TASK - NON-TASK (< 0: non-task kept better)"}
    per = T.groupby(["animal", "epoch"])[list(Q)].mean().unstack("epoch")
    nses = T.groupby(["animal", "epoch"]).size().unstack("epoch").reindex(columns=list(EPOCHS))
    print("PER-ANIMAL log2 vs own pre (session means; rule 8 -- read before the pooled test)")
    for q in Q:
        print(f"  {q}\n" + per[q].reindex(columns=list(EPOCHS)).round(3).to_string())
    print("  sessions per animal x epoch\n" + nses.to_string())
    rng = np.random.default_rng(0)
    rows = []
    for q in Q:                                   # fixed order -> the same rng draws every run (rule 9)
        p = per[q]["pre"]
        rows.append({"quantity": q, "epoch": "pre", **p.to_dict(), "mean": float(p.mean()), "sem": float(p.sem())})
        rows += [{"quantity": q, **r} for r in frs.epoch_change(T, q, rng, n_boot)]
    R = pd.DataFrame(rows)
    R["mark"] = ["*" if x is True else "" for x in R.excludes_0]
    R.to_csv(figure_layout.sidecar_for(out, "task_vs_nontask_pooled", ".csv"), index=False, float_format="%.5g")
    print("POOLED (mean of animals; boot_delta vs pre, * = 95 % CI excludes 0)")
    print(R[["quantity", "epoch", "mean", "sem", "delta", "ci_lo", "ci_hi", "mark", "n_below", "n_animals"]]
          .round(3).to_string(index=False))

    col = frc.animal_colors()
    animals = sorted(T.animal.unique())
    off = {an: (i - (len(animals) - 1) / 2) * 0.09 for i, an in enumerate(animals)}
    with frf.style():
        fig, axs = frf.subplots(1, 3, figsize=(13, 4.6), sharey=True)
        for ax, (q, title) in zip(axs, Q.items()):
            for an in animals:
                y = per[q].loc[an].reindex(EPOCHS).to_numpy()
                x = np.arange(4) + off[an]
                ax.plot(x, y, "-", lw=0.6, color=col.get(an, "0.5"), alpha=0.6)
                ax.plot(x, y, "o", ms=5, color=col.get(an, "0.5"), mec="none", label=an)
            r = R[R.quantity == q].set_index("epoch").reindex(EPOCHS)
            ax.errorbar(np.arange(4), r["mean"], yerr=r["sem"], fmt="s-", color="k", ms=6, lw=1.6, capsize=3,
                        zorder=5, label="mean +- SEM")
            ax.axhline(0, color="k", lw=0.5, ls=":")
            labs = ["pre"]
            for ep in EPOCHS[1:]:
                rr = r.loc[ep]
                labs.append(f"{ep}\n{rr['delta']:+.2f}{rr['mark']}\n{int(rr['n_below'])}/{int(rr['n_animals'])} below")
            ax.set_xticks(range(4), labs, fontsize=8)
            ax.set_title(title, fontsize=10)
        axs[0].set_ylabel("log2 vs own pre (animal mean)")
        frf.legend_outside(axs[-1], fontsize=8)
        fig.suptitle(f"{set_name}: task vs non-task, pooled across animals")
        frf.save(fig, out / "task_vs_nontask_pooled.png", f"""
Cross-animal version of task_vs_nontask.png, normalised WITHIN each animal (Priya 10-25). Per session: TASK = summed
absolute expression of the lick components ({', '.join(lick)}) in the response window; NON-TASK = summed absolute
expression of the other components ({', '.join(other)}) between trials; each as log2 minus the mean log2 of the same
animal's pre sessions (`face_rhythm.stats.log2_vs_pre`; pre averages 0). Right: TASK - NON-TASK (log2 of the task /
non-task ratio vs pre; < 0 = non-task preserved relative to task). Dots = per-animal epoch means of the sessions (repo
animal colours); black = mean +- SEM over animals. Tick labels: `stats.epoch_change` (`analysis_kit.boot_delta`,
paired animals -> sessions, animals weighted equally) change vs pre, * = 95 % interval excludes 0; k/n = animals below
pre. Per-animal table first (rule 8): data/task_vs_nontask_pooled.csv.""")
    return R


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import config
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local import figure_layout
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--band", default="all", choices=tuple(frr.BANDS), help="frequency band of the SHARE measure")
    ap.add_argument("--measure", default="speed", choices=("speed", "share"),
                    help="movement maps: absolute speed of the points, or the normalised spectral share")
    ap.add_argument("--midline-x", type=float, default=None, help="image x of the face midline (default: "
                                                                   "frr.MASTER_MIDLINE_X, the master's LP nose)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fdir", type=Path, default=None, help="factor-set folder (default: <bundle>/factors/<set>)")
    ap.add_argument("--pooled-only", action="store_true", help="only task_vs_nontask_pooled.png, from the cached "
                                                                 "data/session_table.csv (no recomputation)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    M = np.load(fdir / "master.npz")
    names, home, freqs = fro.labels(fdir), M["home"], M["freqs"]
    if a.pooled_only:
        import pandas as pd
        task_nontask_pooled(pd.read_csv(figure_layout.sidecar_for(out, "session_table", ".csv")), names, out, fdir.name)
        return 0

    C = frr.describe_components(fdir)
    C.to_csv(figure_layout.sidecar_for(out, "components", ".csv"), index=False, float_format="%.4g")
    (fdir / "labels_suggested.json").write_text(json.dumps({str(k): n for k, n in zip(C.comp, C.suggested)}, indent=1))
    print(C[["comp", "suggested", "peak_hz", "top_region", "w_rim", "lick_rate_corr", "farL_vs_farR"]]
          .round(3).to_string(index=False))
    S = frr.session_table(fdir, names, home, freqs)
    S.to_csv(figure_layout.sidecar_for(out, "session_table", ".csv"), index=False, float_format="%.5g")
    _, Pos = frr.amplitudes(fdir, names)
    Pos.to_csv(figure_layout.sidecar_for(out, "position_amplitudes", ".csv"), index=False, float_format="%.5g")
    metrics = [c for c in S.columns if c.startswith(("amp_", "abs_", "recon_err", "evr_", "cos_", "asym_"))]
    A, D = frr.epoch_stats(S, metrics)
    A.to_csv(figure_layout.sidecar_for(out, "epoch_means", ".csv"), index=False, float_format="%.5g")
    D.to_csv(figure_layout.sidecar_for(out, "epoch_deltas", ".csv"), index=False, float_format="%.5g")
    key = [c for c in metrics if c.startswith(("recon_err_trials", "evr_gap", "asym_speed_response"))]
    print("PER-ANIMAL epoch means (rule 8: read these before the pooled deltas)")
    print(A[["animal", "epoch", "n_sessions", *key]].round(3).to_string(index=False))

    common = (f"Set `{fdir.name}` ({len(S)} sessions; animals {', '.join(sorted(S.animal.unique()))}); epochs "
              f"`wfield_local.epochs.epoch_of` ({', '.join(EPOCHS)}); points = per-animal epoch means of the session "
              f"values (rule 8: read per animal before the pooled deltas in data/epoch_deltas.csv). Computation: "
              f"`wfield_local/face_rhythm_readouts.py`; pipeline: docs/FACE_RHYTHM_PIPELINE.md.")
    # amplitudes by epoch: absolute expression (left three) and share of the frame (right three)
    with frf.style():
        fig, axs = frf.subplots(len(names), 6, figsize=(24, 2.8 * len(names)), squeeze=False)
        for k, nm in enumerate(names):
            for j, (pre_, w) in enumerate([(p_, w_) for p_ in ("abs", "amp") for w_ in ("precue", "response",
                                                                                        "between")]):
                col = f"{pre_}_{nm}_{w}"
                if col in S:
                    _epoch_lines(axs[k, j], S, col, f"{nm} {w} ({'abs' if pre_ == 'abs' else 'share'})")
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: component amplitude by epoch")
        frf.save(fig, out / "amplitudes_by_epoch.png", f"""
{common}
Per component (rows) and window (columns): PRE-CUE ({frr.WINDOWS['precue'][0]:g}..{frr.WINDOWS['precue'][1]:g} s), RESPONSE
({frr.WINDOWS['response'][0]:g}..+{frr.WINDOWS['response'][1]:g} s; engaged trials, `frr.WINDOWS`) and
BETWEEN trials (outside cue -3..+6 s, large movement excluded, MEDIAN of the 5 Hz bins: grooming tops several
factors). Left three: ABSOLUTE expression (weight x FR normaliser D; a.u. on the animal's scale); right three: the
SHARE of the frame (weight only). Pre-stroke sessions are projected on their left-out-day basis where it exists.""")

    # recovery by day
    cols = [c for c in ("recon_err_trials", "recon_err_session", "evr_gap", "cos_spatial_mean", "cos_spectral_mean")
            if c in S]
    with frf.style():
        fig, axs = frf.subplots(1, len(cols), figsize=(4.4 * len(cols) + 1.5, 4.0), squeeze=False)
        for ax, c in zip(axs[0], cols):
            for an, g in S.groupby("animal"):
                ax.plot(g.day, g[c], "o-", ms=4, lw=1, label=an, color=config.animal_color().get(an))
            ax.axvline(0.5, color="k", lw=0.5)
            ax.set(xlabel="days since stroke", title=c)
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: reconstruction and morphing vs day")
        frf.save(fig, out / "recovery_by_day.png", f"""
{common}
Per session vs days since the stroke (vertical line = stroke): median relative reconstruction error of the frozen set
on the trial windows and on the whole session; EVR GAP = the session's own rank-R fit EVR minus the frozen set's EVR on
the same rows (`fitsession`; how much the session's structure departs from the pre-stroke basis -- read against the
leave-one-pre-day-out noise floor in the factor set's report); matched spatial / spectral cosines between the
session's own factors and the frozen ones (morphing; 1 = unchanged).""")

    # EVR: own components vs the frozen pre-stroke components (the manuscript's Fig 4 variance-explained panel)
    if "evr_own" in S:
        LF = frr.loso_floor(fdir)
        animals_ = sorted(S.animal.unique())
        with frf.style():
            fig, axs = frf.subplots(3, len(animals_), figsize=(4.2 * len(animals_), 11), squeeze=False, sharey="row")
            for j, an in enumerate(animals_):
                g = S[S.animal == an].sort_values("day")
                ax = axs[0, j]
                lf = LF[LF.animal == an] if len(LF) else LF
                if len(lf):
                    ax.axhspan(lf.evr_held_out.min(), lf.evr_held_out.max(), color="0.85", lw=0,
                               label="pre noise floor (held-out day)")
                ax.plot(g.day, g.evr_own, "o-", ms=4, lw=1, color="k", label="own components")
                ax.plot(g.day, g.evr_frozen, "o-", ms=4, lw=1, color="tab:red", mfc="white",
                        label="frozen pre-stroke components")
                ax.axvline(0.5, color="k", lw=0.5)
                ax.set(xlabel="days since stroke", ylabel="EVR (uncentred)" if j == 0 else None)
                ax.set_title(an, color=config.animal_color().get(an, "k"), fontweight="bold")
                ax = axs[1, j]
                m = g.groupby("epoch")[["evr_own", "evr_frozen"]].mean().reindex(EPOCHS)
                x = np.arange(len(EPOCHS))
                ax.bar(x - 0.18, m.evr_own, 0.36, color="k", label="own")
                ax.bar(x + 0.18, m.evr_frozen, 0.36, color="tab:red", label="frozen pre")
                for e_i, ep in enumerate(EPOCHS):
                    ge = g[g.epoch == ep]
                    ax.plot(np.full(len(ge), e_i - 0.18), ge.evr_own, ".", color="0.5", ms=3)
                    ax.plot(np.full(len(ge), e_i + 0.18), ge.evr_frozen, ".", color="0.5", ms=3)
                ax.set_xticks(x, [e[:5] for e in EPOCHS])
                lo = np.nanmin(g[["evr_own", "evr_frozen"]].to_numpy())
                ax.set_ylim(max(0.0, lo - 0.05), 1.0)
                ax.set(ylabel="EVR, epoch mean" if j == 0 else None)
                # RATIO frozen / own (Priya 10-24): divides out what lowers BOTH fits (less movement against a fixed
                # tracking-noise floor, less stereotyped movement) -> what is left is the pre-stroke vocabulary's fit
                ax = axs[2, j]
                rt = g.evr_frozen / g.evr_own                  # normalised: the animal's pre-stroke mean = 1 (Priya)
                pr = rt[g.epoch == "pre"]
                if len(pr):
                    rt, pr = rt / pr.mean(), pr / pr.mean()
                    sem = float(pr.std(ddof=1) / np.sqrt(len(pr))) if len(pr) > 1 else 0.0
                    ax.axhspan(1 - sem, 1 + sem, color="0.85", lw=0, label="pre +- 1 SEM")
                ax.axhline(1, color="k", lw=0.5)
                ax.plot(g.day, rt, "o-", ms=4, lw=1, color=config.animal_color().get(an, "tab:purple"),
                        label="frozen / own")
                ax.axvline(0.5, color="k", lw=0.5)
                ax.set(xlabel="days since stroke", ylabel="EVR frozen / own, / pre mean" if j == 0 else None)
            frf.legend_outside(axs[0, -1])
            fig.suptitle(f"{fdir.name}: variance explained by each session's own components vs the frozen pre-stroke set")
            frf.save(fig, out / "evr_own_vs_frozen.png", f"""
{common}
The 2pRAM-cohort manuscript's variance-explained panel on one frozen basis across animals. Per session: uncentred EVR
(1 - ||X - X^||^2 / ||X||^2, the TCA definition) of the session's OWN rank-R non-negative CP fit (black; `fitsession`,
best of its random starts, folder `{frr.own_sub(fdir)}`) and of the FROZEN pre-stroke master's spatial + spectral factors with
only the temporal weights refit (red), on the same rows. Top: per session over days since stroke; grey band = the
leave-one-pre-day-out held-out EVR range of that animal's pre days (`loso/`): where a pre-stroke day scored on a fit
without it lands. Middle: epoch means (bars) and sessions (dots). Bottom: the RATIO frozen / own per session, divided by the
animal's mean pre-stroke ratio (pre = 1; grey = +-1 SEM of its pre sessions): own EVR itself dips acutely (less movement against a fixed tracking-noise floor, less
stereotyped movement), the ratio divides that out. Own >= frozen by construction; the GAP (own - frozen) / the ratio is
what the pre-stroke repertoire fails to describe. Pre sessions' frozen EVR is on their held-out basis.""")

    # movement maps + asymmetry
    _, maps = frr.movement(fdir, home, freqs, a.midline_x, a.band, a.measure)
    animals = sorted({k[0] for k in maps if isinstance(k, tuple)})
    what = "speed" if a.measure == "speed" else f"spectral share, {a.band}"
    for w in ("between", "precue", "response"):
        with frf.style():
            fig, axs = frf.subplots(len(animals) + 1, 3, figsize=(11, 3.4 * (len(animals) + 1)), squeeze=False)
            sc = None
            for j, ep in enumerate(EPOCHS[1:]):
                mm = []
                for i, an in enumerate(animals):
                    ax = axs[i, j]
                    ax.axis("off")
                    v = maps.get((an, ep, w))
                    if v is None:
                        continue
                    mm.append(v)
                    sc = ax.scatter(home[:, 0], home[:, 1], c=v, s=4, cmap="RdBu_r", vmin=-1, vmax=1)
                    ax.axvline(maps["midline_x"], color="k", lw=0.4)
                    ax.set_title(f"{an} {ep}")
                    ax.invert_yaxis()
                    ax.set_aspect("equal")
                ax = axs[-1, j]
                ax.axis("off")
                if mm:
                    sc = ax.scatter(home[:, 0], home[:, 1], c=np.mean(mm, 0), s=4, cmap="RdBu_r", vmin=-1, vmax=1)
                    ax.set_title(f"animal mean {ep} (n={len(mm)})")
                    ax.invert_yaxis()
                    ax.set_aspect("equal")
            if sc is not None:
                fig.colorbar(sc, ax=axs, shrink=0.35, label="log2(epoch / pre)")
            fig.suptitle(f"{fdir.name}: {w} movement ({what}) vs pre")
            frf.save(fig, out / f"movement_maps_{a.measure}_{w}.png", f"""
{common}
Per mesh point on the master face (image right = IPSILESIONAL on cam4; vertical line = the face midline, x =
{maps['midline_x']:.0f}, the master's LP nose): log2(epoch mean / pre-stroke mean) of the point's movement in the
{w.upper()} window. Measure `{a.measure}`: speed = RMS of |d displacement / dt| of the 25 Hz tracked points (ABSOLUTE
movement); share = the point's share of the normalised spectrogram ({a.band} band). Rows: animals, then the mean of the
animal maps. Red = more movement than pre, blue = less; colour clipped at +-1 (2x).""")
    with frf.style():
        fig, axs = frf.subplots(2, 3, figsize=(14, 7.5))
        for i, m in enumerate(("speed", "share")):
            for ax, w in zip(axs[i], ("between", "precue", "response")):
                _epoch_lines(ax, S, f"asym_{m}_{w}", f"{m}, {w}")
                ax.axhline(0, color="k", lw=0.5)
                ax.set_ylabel("(contra - ipsi) / sum" if w == "between" else None)
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: contra vs ipsi face movement by epoch")
        frf.save(fig, out / "asymmetry_by_epoch.png", f"""
{common}
(contra - ipsi) / (contra + ipsi) of the movement summed over the mesh points on each side of the face midline (x =
{maps['midline_x']:.0f}; cam4 ipsilesional = image RIGHT, all lesions left), per window (between / pre-cue / response),
for absolute SPEED (top) and the normalised spectral SHARE (bottom). 0 = symmetric; > 0 = more contralesional movement.""")
    # non-task preservation + severity (the 2pRAM-cohort manuscript's Fig 4 framing; Priya 10-24)
    TN = frr.task_nontask(S, names)
    TN.to_csv(figure_layout.sidecar_for(out, "task_nontask", ".csv"), index=False, float_format="%.5g")
    SV = frr.severity_table(S, names)
    SV.to_csv(figure_layout.sidecar_for(out, "severity", ".csv"), index=False, float_format="%.5g")
    print("TASK (lick, response) vs NON-TASK (other, between) expression vs pre, per animal x epoch")
    print(TN.pivot(index="animal", columns="epoch", values=["task_vs_pre", "nontask_vs_pre"])
          .reindex(columns=list(EPOCHS), level=1).round(2).to_string())
    print("SEVERITY (descriptive, n = 4)\n" + SV.round(3).to_string(index=False))
    lick, other = frr.lick_split(names)
    with frf.style():
        fig, axs = frf.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, c in zip(axs, ("task_vs_pre", "nontask_vs_pre")):
            for an, g in TN.groupby("animal"):
                m = g.set_index("epoch")[c].reindex(EPOCHS)
                ax.plot(range(4), m.to_numpy(), "o-", ms=4, lw=1.2, label=an)
            ax.axhline(1, color="k", lw=0.5)
            ax.set_xticks(range(4), [e[:5] for e in EPOCHS])
            ax.set_title("TASK: lick components, response window" if c.startswith("task") else
                         "NON-TASK: other components, between trials")
        axs[0].set_ylabel("expression / pre")
        frf.legend_outside(axs[-1])
        fig.suptitle(f"{fdir.name}: task vs non-task orofacial movement over recovery")
        frf.save(fig, out / "task_vs_nontask.png", f"""
{common}
The 2pRAM-cohort manuscript's "relative preservation of non-task orofacial movements", on one frozen basis across
animals. TASK = summed ABSOLUTE expression (weight x D) of the lick components ({', '.join(lick)}) in the response
window; NON-TASK = summed absolute expression of the other components ({', '.join(other)}) BETWEEN trials (median of
the 5 Hz bins outside cue -3..+6 s, large movement excluded). Per animal: epoch mean of the session values / the
animal's pre-stroke mean. Lick INITIATION components ("prelick") count as non-lick here, as in fr_groups.""")
        mets = [c for c in SV.columns if c.endswith(("_acute", "_chronic", "_minus_pre")) and SV[c].notna().any()]
        fig, axs = frf.subplots(1, len(mets), figsize=(3.3 * len(mets), 3.6), squeeze=False)
        for ax, c in zip(axs[0], mets):
            ax.errorbar(SV.burden_ipsi_contra_pct, SV[c], xerr=SV.burden_sem, fmt="o", color="k", ms=5, lw=0.8)
            for _, r in SV.iterrows():
                ax.annotate(r.animal, (r.burden_ipsi_contra_pct, r[c]), fontsize=8, xytext=(3, 3),
                            textcoords="offset points")
            ax.set(xlabel="ipsi / contra CP area (%)\n(lower = larger lesion)", title=c.replace("_", " "))
        fig.suptitle(f"{fdir.name}: FaceRhythm deficit vs lesion burden (descriptive, n = {len(SV)})")
        frf.save(fig, out / "severity.png", f"""
{common}
One point per animal: x = lesion burden from histology (`wfield_local/histology_area.py`: caudoputamen ipsi / contra %
area, mean +- SEM over sections; LOWER = more tissue lost), y = the animal's FaceRhythm deficit: task / non-task
expression vs pre (acute, chronic; `task_vs_nontask.png`), and chronic minus pre reconstruction error, EVR gap and
matched spatial / spectral cosine (morphing) where the per-session fits exist. DESCRIPTIVE ONLY: four animals, no test
and no fitted line (CLAUDE.md rule 8); the 2pRAM-cohort manuscript's mild-vs-severe framing.""")
    task_nontask_pooled(S, names, out, fdir.name)
    print(f"{len(S)} sessions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
