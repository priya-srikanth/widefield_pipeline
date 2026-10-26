"""FaceRhythm factor MATCHING across independent fits ("morphing"; Priya 10-24, after the VLS-stroke methods paper):
each session's OWN rank-R fit (`face_rhythm_factors.py fitsession` -> ``<set>/own/``) Hungarian-matched to the frozen
master, spatial loading and spectral (frequency) similarity reported separately, against two references:

  * PRE-STROKE FLOOR: the same factor's cosine between two PRE sessions' own fits of one animal (both aligned to the
    master first, `face_rhythm_matching.pairwise`) -- how much a factor moves between pre-stroke days anyway;
  * CHANCE: matched cosine against the session's factors with points / frequencies permuted (95th percentile).

    python scripts/face_rhythm_match.py [--set master_pre_v1] [--on product|spatial|spectral|sum]
                                        [--spatial vector|magnitude] [--chance 200] [--out DIR]

-> ``<dlc>/face_rhythm/cam4/readouts/<set>/matching[_<on>_<spatial>]/``:
  match_sessions.csv     one row per (session, master factor): matched own factor, cos_spatial, cos_spectral (+ the
                         same as ANGLES in degrees, angle_* = arccos: the manuscript's "factor angle"), score,
                         margin (best - runner-up), split_flag (unconstrained best partner != assigned), chance p95
  match_pre_pairs.csv    pre-vs-pre pairs within each animal, per factor (the floor)
  match_epochs.csv       per animal x epoch x factor: mean cosines + the animal's pre floor (rule 8: per animal first)
  match_by_day.png       per factor: spatial and spectral cosine to the master over days since stroke, per animal,
                         grey band = pre floor (5th-95th pct of pre pairs, all animals), dashed = chance
  match_by_day_pooled.png  the same angles z-scored vs each animal's own pre sessions, pooled over animals by epoch
                         (boot_delta, * / ** Holm over factors; + data/match_by_day_pooled.csv); `--plot-only` redraws
                         it alone from data/match_sessions.csv
  match_example.png      the full score matrix (master x own) for one pre and one acute session per animal
Computation: `wfield_local/face_rhythm_matching.py` (choices and caveats in its docstring).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402

EPOCHS = frc.EPOCHS


def pooled_by_day_figure(S: pd.DataFrame, names: list[str], out: Path, set_name: str,
                         rule: str = "Hungarian-matched") -> pd.DataFrame:
    """Cross-animal, WITHIN-ANIMAL-NORMALISED version of `match_by_day.png` (Priya 10-25). Per animal x factor, each
    session's matched-factor ANGLE to the master (spatial and spectral separately) is z-scored against that animal's own
    PRE sessions (August only): (angle - mean pre) / SD pre, so pre averages exactly 0. Per post epoch: the mean of the
    animals, the `boot_delta` change vs pre (via `face_rhythm.stats.epoch_change`, seeded rng 0, factors in order) and the
    animals above pre; Holm over the 12 factors of one measure x epoch. Per-animal table printed FIRST (rule 8).
    Writes ``match_by_day_pooled.png`` + ``data/match_by_day_pooled.csv``; returns the stats table."""
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import figure_layout
    from wfield_local.face_rhythm import stats as frs
    S = S.copy()
    S = S[(S.epoch != "pre") | S.date.astype(str).str.startswith("202608")]          # pre = August only
    meas = {"spatial": "angle_spatial_deg", "spectral": "angle_spectral_deg"}
    for m, c in meas.items():
        g = S[c].where(S.epoch == "pre").groupby([S.animal, S.factor])
        S[f"z_{m}"] = (S[c] - g.transform("mean")) / g.transform("std")
    n_pre = S[S.epoch == "pre"].groupby("animal").key.nunique()
    print("pre sessions per animal (August):", n_pre.to_dict())
    for m in meas:
        A = S.groupby(["factor", "name", "animal", "epoch"])[f"z_{m}"].mean().reset_index()
        A["epoch"] = pd.Categorical(A.epoch, EPOCHS, ordered=True)
        print(f"PER-ANIMAL epoch mean of the {m} angle, z vs own pre (pre = 0 by construction)")
        print(A[A.epoch != "pre"].pivot_table(index=["factor", "name"], columns=["epoch", "animal"],
                                               values=f"z_{m}", observed=True).round(2).to_string())
    rng = np.random.default_rng(0)
    rows = []
    for m in meas:
        for k, nm in enumerate(names):
            d = S[S.factor == k].sort_values(["animal", "day"])
            rows += [{"measure": m, "factor": k, "name": nm, **r} for r in frs.epoch_change(d, f"z_{m}", rng)]
    T = frs.add_holm(pd.DataFrame(rows), ["measure", "epoch"])
    T.to_csv(figure_layout.sidecar_for(out, "match_by_day_pooled", ".csv"), index=False, float_format="%.4g")
    print("POOLED change vs pre (z units; * = 95 % CI excludes 0, ** = Holm over 12 factors)")
    print(T[["measure", "name", "epoch", "delta", "ci_lo", "ci_hi", "n_above", "n_animals", "p_holm", "mark"]]
          .round(3).to_string(index=False))

    ecol = frc.epoch_colors()
    animals = sorted(S.animal.unique())
    x = np.arange(len(names))
    off = dict(zip(frc.POST_EPOCHS, (-0.24, 0.0, 0.24)))
    with frf.style():
        fig, axs = frf.subplots(2, 1, figsize=(13, 8.2), sharex=True, squeeze=False)
        for ax, m in zip(axs[:, 0], meas):
            ax.axhline(0, color=ecol["pre"], lw=1.0, label="pre (own mean = 0)")
            ax.axhspan(-1, 1, color="0.9", lw=0, zorder=0, label="pre +/- 1 SD")
            top = []
            for ep in frc.POST_EPOCHS:
                r = T[(T.measure == m) & (T.epoch == ep)].set_index("factor").reindex(range(len(names)))
                xx = x + off[ep]
                for an in animals:
                    if an in r:
                        ax.plot(xx, r[an], "o", ms=2.5, color=ecol[ep], alpha=0.35, lw=0)
                ax.errorbar(xx, r["delta"], yerr=[r["delta"] - r["ci_lo"], r["ci_hi"] - r["delta"]], fmt="o", ms=6,
                            color=ecol[ep], ecolor=ecol[ep], elinewidth=1.2, capsize=0, label=ep)
                hi = np.fmax(r["ci_hi"], np.nanmax(r[[an for an in animals if an in r]], axis=1))
                top.append(np.nanmax(hi))
                for xi, (h, row) in zip(xx, zip(hi, r.itertuples())):
                    if np.isfinite(h):
                        lab = f"{row.mark}\n{row.n_above}/{row.n_animals}" if row.mark else f"{row.n_above}/{row.n_animals}"
                        ax.annotate(lab, (xi, h), xytext=(0, 3), textcoords="offset points", ha="center",
                                    va="bottom", fontsize=6.5, color=ecol[ep], linespacing=0.9)
            lo = ax.get_ylim()[0]
            ax.set_ylim(lo, max(top) * 1.18 + 0.5)
            ax.set(ylabel=f"{m} angle to master\n(z vs own pre)", xlim=(-0.6, len(names) - 0.4))
            ax.set_title(f"{'Spatial loading' if m == 'spatial' else 'Spectrum'}: angle to the frozen master, "
                         f"change vs own pre", fontsize=10)
        axs[-1, 0].set_xticks(x, [n.removeprefix("fr_") for n in names], rotation=35, ha="right")
        frf.legend_outside(axs[0, 0])
        fig.suptitle(f"{set_name}: own fit vs master, pooled over {len(animals)} animals")
        frf.save(fig, out / "match_by_day_pooled.png", f"""
Cross-animal version of match_by_day.png. Per animal x factor: each session's angle (arccos of the matched cosine, own
fit {rule} to the frozen master; larger = further from the master) as a z-score against that animal's own PRE
sessions (August only; mean and SD over its {', '.join(f'{a} {n}' for a, n in n_pre.items())} pre sessions), so pre is
0 by construction. Big markers: the mean over animals of each animal's epoch mean = the `boot_delta` change vs pre
(paired animals -> sessions bootstrap, 95 % interval, `face_rhythm.stats.epoch_change`); small dots: the animals. Number
above each = animals above pre; * = interval excludes 0, ** = also Holm-significant over the {len(names)} factors of that
measure x epoch. Caveats: the master was fitted on pre sessions, so pre angles are in-sample and a post increase partly
measures out-of-sample fit; an animal with few or very similar pre sessions has a small SD and large z. Descriptive.""")
    return T


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_matching as fm
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--on", default="product", choices=fm.ON, help="what the assignment maximises")
    ap.add_argument("--assign", default="hungarian", choices=("hungarian", "best"),
                    help="one-to-one Hungarian, or each master factor's BEST column (may repeat; DECISIONS 10-25: the "
                         "shape readouts use --on spatial --assign best)")
    ap.add_argument("--spatial", default="vector", choices=("vector", "magnitude"),
                    help="spatial loadings as (x, y) vectors (direction-sensitive) or |(x, y)| per point (where only)")
    ap.add_argument("--chance", type=int, default=200, help="shuffles per session for the chance line (0 = skip)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true",
                    help="only redraw match_by_day_pooled.png from data/match_sessions.csv (no matching)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    sfx = ("" if (a.on, a.spatial) == ("product", "vector") else f"_{a.on}_{a.spatial}") +         ("" if a.assign == "hungarian" else f"_{a.assign}")
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / f"matching{sfx}"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        S = pd.read_csv(figure_layout.sidecar_for(out, "match_sessions", ".csv", mkdir=False), dtype={"date": str})
        pooled_by_day_figure(S, list(S.drop_duplicates("factor").sort_values("factor").name), out, fdir.name,
                             "Hungarian-matched" if a.assign == "hungarian" else f"best-{a.on}-matched (may repeat)")
        print("->", out)
        return 0
    M = np.load(fdir / "master.npz")
    A_ref, B_ref = M["spatial"].reshape(-1, M["spectral"].shape[1]), M["spectral"]
    names = fro.labels(fdir)
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    asym_master = fm.loading_asymmetry(A_ref, home, mid)
    own = frr.own_sub(fdir)                              # own8 (8 starts) once complete, else own (3)
    print("own fits from", own)
    T = frc.sessions(fdir, own)
    if T.empty:
        sys.exit(f"no per-session fits in {fdir / 'own'} (run fitsession)")
    T["day"] = [frc.day(lb) for lb in T.label]

    rows, fits, examples = [], {}, {}
    for t in T.itertuples():
        with np.load(t.file) as z:
            A, B = z["A_own"], z["B_own"]
        fits[t.key] = (A, B)
        m = fm.match(A_ref, B_ref, A, B, on=a.on, spatial=a.spatial)
        if a.assign != "hungarian":                  # re-point every per-factor value at the chosen column
            cs_, cf_ = fm.cosines(A_ref, B_ref, A, B, a.spatial)
            col = fm.columns(m, a.assign)
            ix = np.arange(len(col))
            m = {**m, "order": col, "cos_spatial": cs_[ix, col], "cos_spectral": cf_[ix, col],
                 "score": m["score_matrix"][ix, col]}
        ch = fm.chance(A_ref, B_ref, A, B, a.on, a.spatial, n=a.chance, assign=a.assign) if a.chance else {}
        asym_own = fm.loading_asymmetry(A, home, mid)
        if t.epoch in ("pre", "acute"):
            examples.setdefault((t.animal, t.epoch), (t.key, m["score_matrix"]))
        for k in range(len(names)):
            rows.append({"key": t.key, "animal": t.animal, "date": t.date, "epoch": t.epoch, "day": t.day,
                         "factor": k, "name": names[k], "matched": int(m["order"][k]),
                         "cos_spatial": m["cos_spatial"][k], "cos_spectral": m["cos_spectral"][k],
                         "score": m["score"][k], "margin": m["margin"][k],
                         "split_flag": bool(m["free"][k] != m["order"][k]),
                         "loading_asym": float(asym_own[m["order"][k]]) if m["order"][k] >= 0 else np.nan,
                         "loading_asym_master": float(asym_master[k]),
                         "chance_spatial_p95": ch.get("cos_spatial_p95", [np.nan] * len(names))[k],
                         "chance_spectral_p95": ch.get("cos_spectral_p95", [np.nan] * len(names))[k]})
    S = pd.DataFrame(rows)
    for c in ("spatial", "spectral"):                    # the manuscript's "factor ANGLE": arccos of the matched cosine
        S[f"angle_{c}_deg"] = np.degrees(np.arccos(np.clip(S[f"cos_{c}"], -1, 1)))
    S.to_csv(figure_layout.sidecar_for(out, "match_sessions", ".csv"), index=False, float_format="%.4g")
    pooled_by_day_figure(S, names, out, fdir.name, "Hungarian-matched" if a.assign == "hungarian" else f"best-{a.on}-matched (may repeat)")

    pre = T[T.epoch == "pre"]
    P = pd.concat([fm.pairwise({k: fits[k] for k in g.key}, A_ref, B_ref, a.on, a.spatial, a.assign).assign(animal=an)
                   for an, g in pre.groupby("animal") if len(g) > 1], ignore_index=True)
    P["name"] = [names[k] for k in P.factor]
    for c in ("spatial", "spectral"):
        P[f"angle_{c}_deg"] = np.degrees(np.arccos(np.clip(P[f"cos_{c}"], -1, 1)))
    P.to_csv(figure_layout.sidecar_for(out, "match_pre_pairs", ".csv"), index=False, float_format="%.4g")

    floor = P.groupby(["animal", "factor"])[["cos_spatial", "cos_spectral"]].median() \
        .rename(columns=lambda c: f"pre_floor_{c}")
    E = S.groupby(["animal", "epoch", "factor", "name"], sort=False)[["cos_spatial", "cos_spectral", "split_flag"]] \
        .mean().reset_index().join(floor, on=["animal", "factor"])
    E["epoch"] = pd.Categorical(E.epoch, EPOCHS, ordered=True)
    E = E.sort_values(["factor", "animal", "epoch"])
    E.to_csv(figure_layout.sidecar_for(out, "match_epochs", ".csv"), index=False, float_format="%.4g")
    print("PER-ANIMAL epoch means of the matched cosine to the master (pre floor = median over pre pairs)")
    print(E.pivot_table(index=["factor", "name"], columns=["animal", "epoch"], values="cos_spatial", observed=True)
          .round(2).to_string())

    animals = sorted(S.animal.unique())
    from wfield_local import config
    col = config.animal_color()                          # the repo's one animal palette (configs/animals.yaml)
    with frf.style():
        fig, axs = frf.subplots(len(names), 2, figsize=(13, 2.4 * len(names)), squeeze=False, sharex=True)
        for k, nm in enumerate(names):
            for j, c in enumerate(("cos_spatial", "cos_spectral")):
                ax = axs[k, j]
                q = P[P.factor == k][c]
                if len(q):
                    ax.axhspan(*np.nanpercentile(q, [5, 95]), color="0.85", lw=0, label="pre floor (5-95 %)")
                ch = S[S.factor == k][c.replace("cos", "chance") + "_p95"].median()
                if np.isfinite(ch):
                    ax.axhline(ch, color="0.4", ls="--", lw=0.8, label="chance (p95)")
                for an in animals:
                    g = S[(S.factor == k) & (S.animal == an)].sort_values("day")
                    ax.plot(g.day, g[c], "o-", ms=3, lw=1, color=col[an], label=an)
                ax.axvline(0, color="k", lw=0.5)
                ax.set(ylim=(min(0.0, ax.get_ylim()[0]), 1.02), ylabel="cosine",
                       title=f"{k} {nm}: {'spatial loading' if j == 0 else 'spectrum'}")
        for ax in axs[-1]:
            ax.set_xlabel("days since stroke")
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: own-fit factors matched to the frozen master "
                     f"({'Hungarian' if a.assign == 'hungarian' else 'best match'} on {a.on}, "
                     f"spatial = {a.spatial})")
        frf.save(fig, out / "match_by_day.png", f"""
Each session's OWN rank-{len(names)} non-negative CP fit (`fitsession`, best of its random starts) is matched to the frozen
master's factors by {'the Hungarian assignment maximising' if a.assign == 'hungarian' else "each master factor's best column (may repeat) by"} the {a.on} of the spatial and spectral cosines (spatial loadings
as {'(x, y) displacement vectors over all points' if a.spatial == 'vector' else '|(x, y)| per point'}). Plotted: the
matched pair's spatial (left) and spectral (right) cosine, per session, over days since stroke, one line per animal.
Grey band: 5th-95th percentile of the same factor's cosine between two PRE-stroke sessions of one animal (both own fits
aligned to the master first) -- the day-to-day variation a post-stroke value is read against. Dashed: chance, the 95th
percentile of the matched cosine with the session's points / frequencies permuted (non-negative factors: chance > 0).
Pre sessions' own fits are matched to the full master here (in-sample for the master's fit rows). Split / merge: see
match_sessions.csv `split_flag` (unconstrained best partner differs from the assigned one) and `margin`.
Computation `wfield_local/face_rhythm_matching.py`; driver `scripts/face_rhythm_match.py`.""")

        ex = sorted(examples.items())
        if ex:
            fig, axs = frf.subplots(1, len(ex), figsize=(3.6 * len(ex), 3.8), squeeze=False)
            for ax, ((an, ep), (key, Sm)) in zip(axs[0], ex):
                im = ax.imshow(Sm, vmin=0, vmax=1, cmap="viridis")
                ax.set(title=f"{an} {ep}\n{key}", xlabel="own factor", ylabel="master factor",
                       yticks=range(len(names)), yticklabels=[f"{k} {n}" for k, n in enumerate(names)])
            fig.colorbar(im, ax=axs[0, -1], label=a.on)
            frf.save(fig, out / "match_example.png", f"""
Every master x own-factor {a.on} similarity for one PRE and one ACUTE session per animal (the matrix the Hungarian
assignment works on). A clean one-to-one match is one bright cell per row and column; a row with two bright cells is a
master factor SPLIT in that session's fit, a column with two is a MERGE.""")

        # asymmetry two ways (Priya 10-24): tracked-point movement (readouts) and own-fit spatial loading (here)
        st = Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "data" / "session_table.csv"
        pts = pd.read_csv(st) if st.exists() else None
        show = [k for k, n in enumerate(names) if "lick" in n and "prelick" not in n] + \
               [k for k, n in enumerate(names) if "jaw" in n]
        ncol = 2 + len(show)
        fig, axs = frf.subplots(1, ncol, figsize=(3.2 * ncol, 3.8), squeeze=False, sharey=False)
        x = np.arange(len(EPOCHS))
        for j, w in enumerate(("response", "between")):
            ax = axs[0, j]
            c = f"asym_speed_{w}"
            if pts is not None and c in pts:
                for an, g in pts.groupby("animal"):
                    mm = g.groupby("epoch")[c].mean().reindex(EPOCHS)
                    ax.plot(x, mm.to_numpy(), "o-", ms=4, lw=1.2, color=col.get(an), label=an)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x, [e[:5] for e in EPOCHS])
            ax.set(title=f"POINT MOVEMENT\n{w} window (speed)", ylabel="(contra - ipsi) / sum" if j == 0 else None)
        for jj, k in enumerate(show):
            ax = axs[0, 2 + jj]
            for an in animals:
                g = S[(S.factor == k) & (S.animal == an)]
                mm = g.groupby("epoch")["loading_asym"].mean().reindex(EPOCHS)
                ax.plot(x, mm.to_numpy(), "o-", ms=4, lw=1.2, color=col[an], label=an)
            ax.axhline(asym_master[k], color="0.4", ls="--", lw=0.8, label="frozen master")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x, [e[:5] for e in EPOCHS])
            ax.set(title=f"OWN-FIT LOADING\n{k} {names[k].removeprefix('fr_')}")
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: face asymmetry two ways (> 0 = more CONTRALESIONAL; cam4 ipsi = image right)")
        frf.save(fig, out / "asymmetry_two_ways.png", f"""
Two measures of contra- vs ipsilesional face asymmetry, (contra - ipsi) / (contra + ipsi), split at the face midline
(x = {mid:.0f}, the master's LP nose; cam4 ipsilesional = IMAGE RIGHT, all lesions left), per animal x epoch (mean of
sessions). LEFT TWO: tracked-point MOVEMENT -- RMS speed of the 25 Hz mesh points summed over each half of the face, in
the response window (cue 0..+4 s, engaged trials, all six spouts pooled) and between trials (`face_rhythm_readouts.movement`; independent of
the factor model). RIGHT: SPATIAL LOADING of each session's OWN-fit factor matched to the master factor named (Hungarian,
{a.on}), |(x, y)| summed over each half (`face_rhythm_matching.loading_asymmetry`); dashed = the frozen master's value.
On the frozen basis loadings are fixed, so only own fits can show a loading shift: below the master line = the factor's
movement moved off the contralesional face. Lick and jaw factors shown. Own fits: folder `{own}`.""")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
