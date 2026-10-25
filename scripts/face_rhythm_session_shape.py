"""Lick-factor SHAPE per SESSION, with statistics (Priya 10-25: "so we need the per session version for stats").

The pooled per-epoch fits (`face_rhythm_epoch_shape.py`, own_epoch / own_epoch_cohort) give ONE fit per epoch, so their
only cross-animal test is a jackknife over 4 animals -- nothing survives correction (FINDINGS 5.10). Here every session
has its own rank-12 fit (`own/`, `fr_factors.py fitsession`), so each shape readout is one value PER SESSION and the
change vs pre is tested like every other result on the FR page: per-animal table first (CLAUDE.md rule 8), then the
paired animals -> sessions bootstrap (`analysis_kit.boot_delta`), Holm over the family.

Per session and lick factor (lick_right, lick_left): the own column standing for it = its BEST SPATIAL match to the
master (`face_rhythm_matching.columns(..., "best")`, the DECISIONS 10-25 rule; < MIN_COS = not recovered, dropped):
  side_spec   mean r(loading, own-side spout templates) - mean r(other side), sign so that pre > 0 for both factors
              (the one-number version of FR page Fig 7). Templates = the animal's pre-stroke spout-specific movement
              (`proj` speed_response minus its mean over spouts); a PRE session is scored against a template built
              WITHOUT it (leave-one-session-out), so its own noise cannot inflate its r
  r_<spout>   the six r values themselves (Fig 7 cells)
  asym        (contra - ipsi) / sum of |loading| (`face_rhythm_matching.loading_asymmetry`)
  cos_master  spatial cosine of the matched column to the master factor
  loading     (P,) |loading| fraction per mesh point -> per-point change map (Fig 7b), `boot_delta_columns`, BH over
              points

    python scripts/face_rhythm_session_shape.py [--set master_pre_v1] [--plot-only]
-> <dlc>/face_rhythm/cam4/readouts/<set>/session_shape/ (session_shape.csv, session_shape_stats.csv, figures).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402

MIN_COS = 0.5
FACTORS = {"lick_right": ("close_R", "far_R"), "lick_left": ("close_L", "far_L")}   # factor -> its own-side spouts
MEASURES = ("side_spec", "asym", "cos_master")


def _speed(f) -> np.ndarray:
    """(6, P) per-spout response-window speed of one session (NaN for a spout the session lacks)."""
    with np.load(f, allow_pickle=True) as z:
        have = list(z["power_positions"].astype(str))
        out = np.full((6, z["speed_response"].shape[1]), np.nan)
        for i, p in enumerate(frc.POSITIONS):
            if p in have:
                out[i] = z["speed_response"][have.index(p)]
    return out


def compute(fdir: Path) -> tuple[pd.DataFrame, dict]:
    from wfield_local import face_rhythm_matching as fm
    from wfield_local import face_rhythm_readouts as frr
    M = np.load(fdir / "master.npz")
    A_ref, B_ref = M["spatial"].reshape(-1, M["spatial"].shape[-1]), M["spectral"]
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    names = frc.factor_names(fdir)
    own, proj = frc.sessions(fdir, "own"), frc.sessions(fdir, "proj").set_index("key")
    pre_speed = {an: {r.key: _speed(proj.loc[r.key, "file"]) for r in own[(own.animal == an) & (own.epoch == "pre")]
                      .itertuples() if r.key in proj.index} for an in sorted(own.animal.unique())}
    rows, loads, specs = [], {}, {}
    for r in own.itertuples():                                         # sorted by file name (rule 9)
        sp = np.nanmean([v for k, v in pre_speed[r.animal].items() if k != r.key], 0)   # leave this session out
        spec = sp - sp.mean(0, keepdims=True)
        with np.load(r.file) as z:
            A, B = z["A_own"].astype(float), z["B_own"].astype(float)
        m = fm.match(A_ref, B_ref, A, B, on="spatial")
        order = fm.columns(m, "best")
        cs = fm.cosines(A_ref, B_ref, A, B)[0]
        mag, asym = fm.spatial_view(A, "magnitude"), fm.loading_asymmetry(A, home, mid)
        for nm, own_side in FACTORS.items():
            k = names.index(nm)
            c = int(order[k])
            row = {"key": r.key, "label": r.label, "animal": r.animal, "date": r.date, "epoch": r.epoch, "factor": nm,
                   "column": c, "cos_master": float(cs[k, c])}
            if cs[k, c] >= MIN_COS:
                v = mag[:, c] / mag[:, c].sum()
                ok = np.isfinite(v) & np.isfinite(spec).all(0)
                rr = np.array([np.corrcoef(v[ok], spec[i][ok])[0, 1] for i in range(6)])
                own_r = np.mean([rr[frc.POSITIONS.index(p)] for p in own_side])
                oth_r = np.mean([rr[frc.POSITIONS.index(p)] for p in frc.POSITIONS     # the other side's two spouts
                                 if p[-1] in "LR" and p[-1] != own_side[0][-1]])
                row.update({"side_spec": own_r - oth_r, "asym": float(asym[c]),
                            **{f"r_{p}": float(rr[i]) for i, p in enumerate(frc.POSITIONS)}})
                loads[(r.key, nm)] = v
                b = B[:, c]
                specs[(r.key, nm)] = b / b.sum()                     # frequency loading, fraction of total (shape)
            rows.append(row)
    return pd.DataFrame(rows), {"loading": loads, "spectral": specs, "home": home, "freqs": M["freqs"]}


def stats(T: pd.DataFrame, L: dict, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Change vs pre per factor x measure x epoch: paired animals -> sessions bootstrap, Holm over each factor's family
    (measures x 3 epochs, r_<spout> x 3 epochs separately); per-point loading map: `boot_delta_columns`, BH over points."""
    from wfield_local import analysis_kit as ak
    rows = []
    for nm in FACTORS:
        d = T[(T.factor == nm) & T.side_spec.notna()]
        for fam, cols in (("summary", list(MEASURES)), ("spouts", [f"r_{p}" for p in frc.POSITIONS])):
            fam_rows = []
            for col in cols:
                pre = {an: g[col].to_numpy() for an, g in d[d.epoch == "pre"].groupby("animal")}
                for ep in frc.POST_EPOCHS:
                    post = {an: g[col].to_numpy() for an, g in d[d.epoch == ep].groupby("animal")}
                    iv, o = ak.boot_delta(post, pre, np.random.default_rng(seed), draws=True)
                    per = {an: float(np.mean(post[an]) - np.mean(pre[an])) for an in sorted(set(post) & set(pre))}
                    same = sum(np.sign(v) == np.sign(iv.point) for v in per.values()) if iv else 0
                    fam_rows.append({"factor": nm, "family": fam, "measure": col, "epoch": ep,
                                     "change": iv.point if iv else np.nan, "lo": iv.lo if iv else np.nan,
                                     "hi": iv.hi if iv else np.nan, "p": ak.boot_p(o), "same_sign": f"{same}/{len(per)}",
                                     **{f"d_{an}": v for an, v in per.items()}})
            ph = ak.holm([x["p"] for x in fam_rows])
            for x, q in zip(fam_rows, ph):
                x["p_holm"] = q
            rows += fam_rows
    return pd.DataFrame(rows), column_change(T, L, seed)


def column_change(T: pd.DataFrame, L: dict, seed: int = 0) -> pd.DataFrame:
    """Change vs pre of a per-session VECTOR readout (the per-point loading map, or the frequency loading), column by
    column: paired animals -> sessions bootstrap (`boot_delta_columns`), BH over columns. ``L`` {(session key, factor):
    vector}. Rows per factor x epoch x scale ("raw", "z"); obs / lo / hi / p / sig (FDR .05) / sig10 (FDR .10)."""
    from wfield_local import analysis_kit as ak
    mrows = []
    for nm in FACTORS:
        d = T[(T.factor == nm) & T.side_spec.notna()]
        pre0 = {an: np.array([L[(k, nm)] for k in g.key]) for an, g in d[d.epoch == "pre"].groupby("animal")}
        # "z" (Priya 10-25, as Fig 8d): each animal's sessions divided per column by the SD of its OWN pre sessions, so
        # the result is the mean over animals of change / own pre-session noise; the SD is floored at a quarter of the
        # animal's median SD (columns the factor never loads have ~0 SD and would otherwise dominate)
        sd = {an: np.maximum(v.std(0, ddof=1), 0.25 * np.median(v.std(0, ddof=1))) for an, v in pre0.items()}
        for scale in ("raw", "z"):
            div = {an: (sd[an] if scale == "z" else 1.0) for an in pre0}
            pre = {an: v / div[an] for an, v in pre0.items()}
            for ep in frc.POST_EPOCHS:
                post = {an: np.array([L[(k, nm)] for k in g.key]) / div[an]
                        for an, g in d[d.epoch == ep].groupby("animal") if an in div}
                res = ak.boot_delta_columns(post, pre, np.random.default_rng(seed))
                if res is None:
                    continue
                obs, lo, hi, o, n = res
                p = np.minimum(1.0, np.maximum(2 * np.minimum((o <= 0).mean(0), (o >= 0).mean(0)), 1.0 / len(o)))
                mrows.append({"factor": nm, "epoch": ep, "scale": scale, "obs": obs, "lo": lo, "hi": hi, "p": p,
                              "sig": ak.bh(p, 0.05), "sig10": ak.bh(p, 0.10)})
    return pd.DataFrame(mrows)


def spectral_figure(Sp: pd.DataFrame, T: pd.DataFrame, L: dict, freqs: np.ndarray, out: Path, set_name: str) -> None:
    """Spectral twin of the z loading map (Priya 10-25: "7h is our headline. can we do the spectral version?"). Left:
    the matched lick factor's frequency loading (fraction of total), mean of animals, pre and each epoch. Right: change
    vs pre per frequency in units of each animal's own pre-session SD, mean of animals with the paired-bootstrap 95 %
    band; filled markers = FDR .05 over the 36 frequencies. One y scale for both factors."""
    from wfield_local import face_rhythm_figures as frf
    ecol = {"pre": "k", **frc.epoch_colors()}
    z = Sp[Sp.scale == "z"]
    ylim = float(np.nanmax(np.abs(np.concatenate([np.r_[r.lo, r.hi] for r in z.itertuples()])))) * 1.05 if len(z) else 1
    with frf.style():
        fig, axs = frf.subplots(2, 2, figsize=(11, 7), squeeze=False)
        for r_i, nm in enumerate(FACTORS):
            d = T[(T.factor == nm) & T.side_spec.notna()]
            ax = axs[r_i, 0]
            for ep in ("pre", *frc.POST_EPOCHS):
                per = [np.mean([L[(k, nm)] for k in g.key], 0) for _, g in d[d.epoch == ep].groupby("animal")]
                if per:
                    ax.plot(freqs, np.mean(per, 0), color=ecol[ep], lw=2.0 if ep == "pre" else 1.3,
                            ls="--" if ep == "pre" else "-", label=ep)
            ax.set_xscale("log")
            ax.set_ylabel("frequency loading (fraction)")
            ax.set_title(f"{nm}: frequency loading, mean of animals", fontsize=9)
            ax.legend(fontsize=7)
            ax = axs[r_i, 1]
            for ep in frc.POST_EPOCHS:
                row = z[(z.factor == nm) & (z.epoch == ep)]
                if not len(row):
                    continue
                row = row.iloc[0]
                ax.fill_between(freqs, row.lo, row.hi, color=ecol[ep], alpha=0.15, lw=0)
                ax.plot(freqs, row.obs, color=ecol[ep], lw=1.3, label=f"{ep} ({int(row.sig.sum())} sig.)")
                ax.scatter(freqs[row.sig], row.obs[row.sig], color=ecol[ep], s=22, edgecolors="k", linewidths=0.4,
                           zorder=3)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xscale("log")
            ax.set_ylim(-ylim, ylim)
            ax.set_ylabel("change / own pre-session SD")
            ax.set_title(f"{nm}: change vs pre per frequency (filled = FDR .05)", fontsize=9)
            ax.legend(fontsize=7)
        for ax in axs[-1]:
            ax.set_xlabel("Hz")
        fig.suptitle(f"{set_name}: lick-factor FREQUENCY loading per session (own fit, best spatial match)")
        frf.save(fig, out / "session_shape_spectral_z.png", """
Each session's matched lick factor's frequency loading (B column, fraction of its total). Left: mean over animals of
each animal's session mean, pre (dashed) and each post epoch. Right: change vs pre per frequency, each animal's
sessions divided by the SD of its own pre sessions at that frequency (floored at 1/4 of its median SD), mean of
animals; band = paired animals -> sessions bootstrap 95 % interval; filled markers = Benjamini-Hochberg FDR .05 over
the 36 frequencies. Factors are matched SPATIALLY (best spatial match), so this asks whether the same face movement
changed its rhythm.""")


def map_figure(Mp: pd.DataFrame, scale: str, home: np.ndarray, out: Path, set_name: str) -> None:
    """Per-point loading change maps (rows = lick factors, columns = post epochs). Priya 10-25: direction must be
    readable at significant points and panels comparable -> diverging RdBu_r (red = more loading after the stroke), ONE
    colour scale for all six panels (98th percentile of |change| over all of them), FDR-significant points drawn full
    size and opaque with a thin black edge, the rest small and faded. ``scale`` "raw" (loading fraction) or "z" (each
    animal's change / its own pre-session SD)."""
    from wfield_local import face_rhythm_figures as frf
    rows = {(r.factor, r.epoch): r for r in Mp[Mp.scale == scale].itertuples()}
    lim = float(np.nanpercentile(np.abs(np.concatenate([r.obs for r in rows.values()])), 98)) if rows else 1.0
    with frf.style():
        fig, axs = frf.subplots(2, 3, figsize=(12.5, 7.2), squeeze=False)
        sc = None
        for r_i, nm in enumerate(FACTORS):
            for c_i, ep in enumerate(frc.POST_EPOCHS):
                ax = axs[r_i, c_i]
                if (nm, ep) not in rows:
                    ax.axis("off")
                    continue
                row = rows[(nm, ep)]
                ns = ~row.sig
                ax.scatter(home[ns, 0], home[ns, 1], c=row.obs[ns], s=3, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.35,
                           lw=0)
                sc = ax.scatter(home[row.sig, 0], home[row.sig, 1], c=row.obs[row.sig], s=11, cmap="RdBu_r", vmin=-lim,
                                vmax=lim, edgecolors="k", linewidths=0.25)
                ax.axvline(342.0, color="k", lw=0.4)
                ax.invert_yaxis()
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_title(f"{nm}: {ep}\n{int(row.sig.sum())} pts FDR .05 (+{int((row.sig & (row.obs > 0)).sum())} / "
                             f"-{int((row.sig & (row.obs < 0)).sum())}); {int(row.sig10.sum())} at .10", fontsize=8)
        if sc is not None:
            fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.5, extend="both",
                         label="change / own pre-session SD" if scale == "z" else "change in loading fraction")
        fig.suptitle(f"{set_name}: per-point loading change, sessions (paired bootstrap)"
                     + ("\nunits: each animal's change / its own pre-session SD" if scale == "z" else "")
                     + "; image left = contralesional")
        frf.save(fig, out / f"session_shape_loading_map{'_z' if scale == 'z' else ''}.png", """
Loading fraction per mesh point of each session's matched lick factor; change vs pre = mean over animals of (post
sessions - pre sessions), paired animals -> sessions bootstrap per point (`boot_delta_columns`); ringed = Benjamini-
Hochberg discovery at FDR .05 over the 1,282 points (large, opaque, black edge; the rest small and faded). Red = more
loading after the stroke, blue = less; one colour scale for all panels. _z: every session of an
animal divided per point by the SD of that animal's pre sessions (floored at 1/4 of its median SD) before the same
test -- each animal's change in units of its own pre-session noise, as Fig 8d.""")


def figures(T: pd.DataFrame, S: pd.DataFrame, out: Path, set_name: str) -> None:
    from wfield_local import face_rhythm_figures as frf
    acol, ecol = frc.animal_colors(), frc.epoch_colors()
    eps = ["pre", *frc.POST_EPOCHS]
    with frf.style():
        fig, axs = frf.subplots(2, 2, figsize=(11, 7.5), squeeze=False)
        for r_i, nm in enumerate(FACTORS):
            d = T[(T.factor == nm) & T.side_spec.notna()]
            for c_i, col in enumerate(("side_spec", "asym")):
                ax = axs[r_i, c_i]
                for an, g in d.groupby("animal"):
                    for e_i, ep in enumerate(eps):
                        v = g[g.epoch == ep][col].to_numpy()
                        ax.scatter(np.full(len(v), e_i) + (sorted(d.animal.unique()).index(an) - 1.5) * 0.08, v, s=9,
                                   color=acol.get(an, "0.5"), alpha=0.6, lw=0)
                    m = [g[g.epoch == ep][col].mean() for ep in eps]
                    ax.plot(range(len(eps)), m, "-", color=acol.get(an, "0.5"), lw=0.8, alpha=0.8,
                            label=an if r_i == 0 and c_i == 1 else None)
                pre_mean = np.mean([g[g.epoch == "pre"][col].mean() for _, g in d.groupby("animal")])
                for e_i, ep in enumerate(frc.POST_EPOCHS, start=1):
                    s = S[(S.factor == nm) & (S.measure == col) & (S.epoch == ep)].iloc[0]
                    ax.errorbar(e_i + 0.3, pre_mean + s.change, yerr=[[s.change - s.lo], [s.hi - s.change]], fmt="s",
                                color="k", ms=4, capsize=2)
                    star = "**" if s.p_holm < 0.05 else "*" if s.hi < 0 or s.lo > 0 else ""
                    ax.text(e_i + 0.3, ax.get_ylim()[1], f"{star}\n{s.same_sign}", ha="center", va="top", fontsize=7)
                    ax.axvspan(e_i - 0.45, e_i + 0.45, color=ecol[ep], alpha=0.06, lw=0)
                ax.axhline(pre_mean, color="0.5", lw=0.6, ls=":")
                ax.set_xticks(range(len(eps)), eps)
                ax.set_title(f"{nm}: {'side-specificity (own-side r - other-side r)' if col == 'side_spec' else 'loading asymmetry (contra - ipsi) / sum'}",
                             fontsize=9)
        frf.legend_outside(axs[0, 1])
        fig.suptitle(f"{set_name}: lick-factor shape PER SESSION (own fit per session, best spatial match)")
        frf.save(fig, out / "session_shape_summary.png", """
Each dot = one session's own rank-12 fit (lick factor = its best spatial match to the master; cos < .5 dropped).
Lines = animal means. Black squares = pooled change vs pre placed on the pre mean, 95 % interval from the paired animals
-> sessions bootstrap; * interval excludes 0, ** Holm-adjusted p < .05 over the factor's family (3 measures x 3 epochs);
k/4 = animals whose own mean change has the pooled sign. Side-specificity: mean r of the loading with the animal's own-
side pre-stroke spout templates minus the other side's (pre sessions scored against a template built without them).""")


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    fdir = frc.default_set_dir(set_name=a.set)
    out = frc.readouts_dir(a.set) / "session_shape"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    csv, pk = figure_layout.sidecar_for(out, "session_shape", ".csv"), figure_layout.sidecar_for(out, "session_shape", ".pkl")
    if a.plot_only:
        T, L = pd.read_csv(csv), pd.read_pickle(pk)
    else:
        T, L = compute(fdir)
        T.to_csv(csv, index=False, float_format="%.4g")
        pd.to_pickle(L, pk)
    pd.set_option("display.width", 220)
    print("sessions per animal x epoch with the factor recovered (cos >= .5):")
    print(T[T.side_spec.notna()].groupby(["factor", "animal", "epoch"]).size().unstack().to_string())
    print("\nPER ANIMAL mean (rule 8)")
    print(T.groupby(["factor", "animal", "epoch"])[list(MEASURES)].mean().round(3).unstack().to_string())
    S, Mp = stats(T, L["loading"])
    S.to_csv(figure_layout.sidecar_for(out, "session_shape_stats", ".csv"), index=False, float_format="%.4g")
    print("\nCHANGE vs PRE (paired animals -> sessions bootstrap; Holm within factor x family)")
    print(S.drop(columns=[c for c in S.columns if c.startswith("d_")]).round(3).to_string(index=False))
    for r in Mp.itertuples():
        print(f"loading map [{r.scale}] {r.factor} {r.epoch}: {int(r.sig.sum())} points FDR .05, {int(r.sig10.sum())} at .10, "
              f"p < .05 uncorrected {int((r.p < .05).sum())} (chance {0.05 * len(r.p):.0f})")
    figures(T, S, out, fdir.name)
    for scale in ("raw", "z"):
        map_figure(Mp, scale, L["home"], out, fdir.name)
    Sp = column_change(T, L["spectral"])
    for r in Sp[Sp.scale == "z"].itertuples():
        f = L["freqs"][r.sig]
        print(f"spectral [z] {r.factor} {r.epoch}: {int(r.sig.sum())} / {len(r.p)} frequencies FDR .05"
              + (f" ({', '.join(f'{x:.1f}' for x in f)} Hz; change {', '.join(f'{v:+.1f}' for v in r.obs[r.sig])})"
                 if len(f) else ""))
    spectral_figure(Sp, T, L["spectral"], L["freqs"], out, fdir.name)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
