"""Score the PERMUTATION NULL for the all-animals-pooled shape readouts (Priya 10-25: "yes set up the null distribution"
-- the jackknife over 4 animals cannot test the cohort fits; FR handoff 10-25 night / late night).

Input: `fr_factors.py fitcohortnull` (O2) -> ``<set>/own_epoch_cohort_null/<epoch>_p<NNN>_<pre|post>.npz``: per post
epoch and permutation, within every animal its pre and that epoch's sessions pooled and their labels shuffled (counts
kept), then the cohort fit redone on the pseudo-pre and pseudo-post groups by the identical procedure. OBSERVED = the
same readouts on the real ``own_epoch_cohort/COHORT_pre`` vs ``COHORT_<epoch>`` fits.

Per epoch and master factor k (best spatial match, MIN_COS .5 -- `face_rhythm_epoch_shape.py`), change = post fit - pre
fit:
  angle_spatial / angle_spectral   angle between the two fits' factor k (deg)
  asym                             loading asymmetry (contra - ipsi) / sum, post - pre
  side_spec (lick_right / left)    mean r(loading, own-side spout templates) - mean r(other side), post - pre. Templates
                                   (pre-stroke per-spout speed minus its mean over spouts, animals weighted equally)
                                   are rebuilt per permutation FROM ITS PSEUDO-PRE SESSIONS: the real pre fit shares its
                                   sessions with the real templates, so a fixed template would make the null
                                   anti-conservative
  loading map / spectrum           per mesh point (frequency) change of the loading (spectrum) fraction; each point
                                   studentised by its null SD, family-wise p by the MAX |t| over points per permutation
p = (1 + #null >= observed) / (1 + n_perm), two-sided |change| except angles (one-sided, larger). Scalars: Holm per
measure x epoch over TWO families -- the pre-specified lick pair (lick_right, lick_left; the hypothesis) and the other
ten factors (exploratory). One family of 12 could never pass: with 100 permutations the smallest p is 1 / 101, x 12 = .12. The null is WITHIN-animal (fixed effects: "these animals' sessions differ") and includes
drift between pre and post days (shuffling breaks time order).

    python scripts/face_rhythm_cohort_null.py [--set master_pre_v1] [--plot-only]
-> ``<readouts>/<set>/epoch_shape_cohort/``: cohort_null_scalars.png, cohort_null_maps.png, cohort_null_spectral.png,
data/cohort_null.csv, data/cohort_null_points.pkl.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_rhythm_epoch_shape import ASSIGN, MATCH_ON, MIN_COS, deg  # noqa: E402
from face_rhythm_session_shape import FACTORS, _speed  # noqa: E402

from wfield_local.face_rhythm import common as frc  # noqa: E402

SCALARS = ("angle_spatial", "angle_spectral", "asym", "side_spec")
ONE_SIDED = ("angle_spatial", "angle_spectral")


def readout(f: Path, A_ref, B_ref, home, mid) -> dict:
    """One fit's per-factor readouts: matched unit spatial / spectral columns, loading and spectrum fractions,
    asymmetry; NaN / None where the factor is not recovered (cos < MIN_COS)."""
    from wfield_local import face_rhythm_matching as fm
    with np.load(f, allow_pickle=True) as z:
        A, B, keys = z["A_own"].astype(float), z["B_own"].astype(float), [str(k) for k in z["keys"]]
    order = fm.columns(fm.match(A_ref, B_ref, A, B, on=MATCH_ON), ASSIGN)
    cs = fm.cosines(A_ref, B_ref, A, B)[0]
    vec, mag = fm.spatial_view(A, "vector"), fm.spatial_view(A, "magnitude")
    asym = fm.loading_asymmetry(A, home, mid)
    R = A_ref.shape[1]
    out = {"keys": keys, "ok": np.zeros(R, bool), "vec": [None] * R, "spec": [None] * R, "load": [None] * R,
           "asym": np.full(R, np.nan)}
    for k in range(R):
        c = int(order[k])
        if c < 0 or cs[k, c] < MIN_COS:
            continue
        out["ok"][k] = True
        out["vec"][k] = vec[:, c] / np.linalg.norm(vec[:, c])
        out["spec"][k] = B[:, c] / B[:, c].sum()
        out["load"][k] = mag[:, c] / mag[:, c].sum()
        out["asym"][k] = asym[c]
    return out


def templates(keys: list[str], speed: dict) -> np.ndarray:
    """(6, P) spout-specific pre-stroke movement from these sessions: per animal the session mean, then the mean over
    animals (equal weight), minus the mean over spouts."""
    by = {}
    for k in keys:
        if k in speed:
            by.setdefault(k.split("_")[0], []).append(speed[k])
    sp = np.nanmean([np.nanmean(v, 0) for _, v in sorted(by.items())], 0)
    return sp - sp.mean(0, keepdims=True)


def side_spec(v: np.ndarray, spec: np.ndarray, own_side) -> float:
    ok = np.isfinite(v) & np.isfinite(spec).all(0)
    rr = np.array([np.corrcoef(v[ok], spec[i][ok])[0, 1] for i in range(6)])
    own = np.mean([rr[frc.POSITIONS.index(p)] for p in own_side])
    oth = np.mean([rr[frc.POSITIONS.index(p)] for p in frc.POSITIONS if p[-1] in "LR" and p[-1] != own_side[0][-1]])
    return float(own - oth)


def change(pre: dict, post: dict, tmpl: np.ndarray, names: list[str]) -> dict:
    """post - pre per factor: scalars {measure: (R,)} and per-point / per-frequency vectors {k: (P,)}."""
    R = len(names)
    s = {m: np.full(R, np.nan) for m in SCALARS}
    load, spec = {}, {}
    for k in range(R):
        if not (pre["ok"][k] and post["ok"][k]):
            continue
        s["angle_spatial"][k] = deg(float(pre["vec"][k] @ post["vec"][k]))
        b0, b1 = pre["spec"][k] / np.linalg.norm(pre["spec"][k]), post["spec"][k] / np.linalg.norm(post["spec"][k])
        s["angle_spectral"][k] = deg(float(b0 @ b1))
        s["asym"][k] = post["asym"][k] - pre["asym"][k]
        if names[k] in FACTORS:
            s["side_spec"][k] = (side_spec(post["load"][k], tmpl, FACTORS[names[k]])
                                 - side_spec(pre["load"][k], tmpl, FACTORS[names[k]]))
        load[k] = post["load"][k] - pre["load"][k]
        spec[k] = post["spec"][k] - pre["spec"][k]
    return {"s": s, "load": load, "spec": spec}


def maxstat(obs: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per point: t = obs / null SD; family-wise p from the max |t| over points in each permutation; also the
    uncorrected per-point p. ``null`` (n_perm, P)."""
    sd = np.nanstd(null, 0)
    sd = np.where(sd > 0, sd, np.nan)
    t, tn = obs / sd, null / sd
    mx = np.nanmax(np.abs(tn), 1)
    p_fwe = (1 + (mx[:, None] >= np.abs(t)[None, :]).sum(0)) / (1 + len(mx))
    p_unc = (1 + (np.abs(tn) >= np.abs(t)[None, :]).sum(0)) / (1 + len(tn))
    return t, p_fwe, p_unc


def compute(fdir: Path, names: list[str]):
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_readouts as frr
    M = np.load(fdir / "master.npz")
    A_ref, B_ref = M["spatial"].reshape(-1, M["spatial"].shape[-1]), M["spectral"]
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    proj = frc.sessions(fdir, "proj")
    pre_keys = proj[proj.epoch == "pre"]
    speed = {r.key: _speed(r.file) for r in pre_keys.itertuples()}           # every session that can be a pseudo-pre
    speed.update({r.key: _speed(r.file) for r in proj[proj.epoch.isin(frc.POST_EPOCHS)].itertuples()})
    rd = lambda f: readout(f, A_ref, B_ref, home, mid)  # noqa: E731
    obs_pre = rd(fdir / "own_epoch_cohort" / "COHORT_pre.npz")
    rows, pts = [], {}
    for ep in frc.POST_EPOCHS:
        obs = change(obs_pre, rd(fdir / "own_epoch_cohort" / f"COHORT_{ep}.npz"), templates(obs_pre["keys"], speed),
                     names)
        nulls = []
        for fpre in sorted((fdir / "own_epoch_cohort_null").glob(f"{ep}_p*_pre.npz")):
            fpost = fpre.with_name(fpre.name.replace("_pre.npz", "_post.npz"))
            if not fpost.exists():
                continue
            pre = rd(fpre)
            nulls.append(change(pre, rd(fpost), templates(pre["keys"], speed), names))
        n = len(nulls)
        print(f"{ep}: {n} permutations")
        if n < 10:
            continue
        for m in SCALARS:
            o = obs["s"][m]
            N = np.array([x["s"][m] for x in nulls])                           # (n, R)
            if m in ONE_SIDED:
                ge = (N >= o[None, :]).sum(0)
            else:
                ge = (np.abs(N) >= np.abs(o)[None, :]).sum(0)
            nn = np.isfinite(N).sum(0)
            p = np.where(np.isfinite(o) & (nn >= 10), (1 + ge) / (1 + nn), np.nan)
            prim = np.array([nm in FACTORS for nm in names])
            ph = np.full(len(p), np.nan)
            ph[prim], ph[~prim] = ak.holm(p[prim]), ak.holm(p[~prim])
            for k in range(len(names)):
                rows.append({"epoch": ep, "factor": names[k], "measure": m, "obs": o[k], "null_mean": np.nanmean(N[:, k]),
                             "null_lo": np.nanpercentile(N[:, k], 2.5) if nn[k] else np.nan,
                             "null_hi": np.nanpercentile(N[:, k], 97.5 if m not in ONE_SIDED else 95) if nn[k] else np.nan,
                             "n_perm": int(nn[k]), "p": p[k], "p_holm": ph[k]})
        for kind in ("load", "spec"):
            for k in range(len(names)):
                if k not in obs[kind]:
                    continue
                N = np.array([x[kind][k] for x in nulls if k in x[kind]])
                if len(N) < 10:
                    continue
                t, p_fwe, p_unc = maxstat(obs[kind][k], N)
                pts[(kind, names[k], ep)] = {"obs": obs[kind][k], "t": t, "p_fwe": p_fwe, "p_unc": p_unc,
                                             "sig": p_fwe < 0.05, "n_perm": len(N)}
    return pd.DataFrame(rows), {"points": pts, "home": home, "freqs": M["freqs"]}


def scalar_figure(T: pd.DataFrame, out: Path, set_name: str, names: list[str]) -> None:
    from wfield_local import face_rhythm_figures as frf
    ecol = frc.epoch_colors()
    lab = {"angle_spatial": "spatial angle, post vs pre (deg)", "angle_spectral": "spectral angle, post vs pre (deg)",
           "asym": "loading asymmetry change", "side_spec": "side-specificity change"}
    eps = [e for e in frc.POST_EPOCHS if e in set(T.epoch)]
    with frf.style():
        fig, axs = frf.subplots(len(SCALARS), 1, figsize=(12.5, 2.6 * len(SCALARS)), squeeze=False)
        for r_i, m in enumerate(SCALARS):
            ax = axs[r_i, 0]
            fs = names if m != "side_spec" else list(FACTORS)
            for e_i, ep in enumerate(eps):
                d = T[(T.measure == m) & (T.epoch == ep)].set_index("factor").reindex(fs)
                x = np.arange(len(fs)) + (e_i - (len(eps) - 1) / 2) * 0.25
                ax.vlines(x, d.null_lo, d.null_hi, color=ecol[ep], lw=5, alpha=0.25)
                sig = (d.p_holm < 0.05).to_numpy()
                ax.scatter(x[~sig], d.obs[~sig], s=14, facecolor="white", edgecolor=ecol[ep], lw=1, zorder=3)
                ax.scatter(x[sig], d.obs[sig], s=22, color=ecol[ep], edgecolor="k", lw=0.5, zorder=4,
                           label=f"{ep} (n = {int(d.n_perm.max())} perms)")
            if m not in ONE_SIDED:
                ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(np.arange(len(fs)), fs, rotation=25, ha="right", fontsize=7)
            ax.set_ylabel(lab[m], fontsize=8)
        axs[0, 0].plot([], [], color="0.5", lw=5, alpha=0.3, label="null 95 % (angles: 0-95th pct)")
        frf.legend_outside(axs[0, 0])
        fig.suptitle(f"{set_name}: all-animals-pooled fit, post - pre, against the label-shuffle null")
        frf.save(fig, out / "cohort_null_scalars.png", """
Dots = the real all-animals-pooled fit change (COHORT_<epoch> vs COHORT_pre); bars = the permutation null (within each
animal, pre and the epoch's sessions shuffled, both fits redone by the identical procedure; 2.5-97.5th percentile, or
0-95th for the one-sided angles). Filled + black edge = Holm p < .05 (families: the lick pair; the other ten factors). Side-specificity templates
are rebuilt per permutation from its pseudo-pre sessions. Within-animal (fixed-effects) test; includes drift.""")


def map_figure(P: dict, out: Path, set_name: str) -> None:
    from wfield_local import face_rhythm_figures as frf
    home, pts = P["home"], P["points"]
    eps = list(frc.POST_EPOCHS)
    cells = [(nm, ep) for nm in FACTORS for ep in eps if ("load", nm, ep) in pts]
    if not cells:
        return
    lim = float(np.nanpercentile(np.abs(np.concatenate([pts[("load", nm, ep)]["t"] for nm, ep in cells])), 98))
    with frf.style():
        fig, axs = frf.subplots(2, 3, figsize=(12.5, 7.2), squeeze=False)
        sc = None
        for r_i, nm in enumerate(FACTORS):
            for c_i, ep in enumerate(eps):
                ax = axs[r_i, c_i]
                if ("load", nm, ep) not in pts:
                    ax.axis("off")
                    continue
                d = pts[("load", nm, ep)]
                s = d["sig"]
                ax.scatter(home[~s, 0], home[~s, 1], c=d["t"][~s], s=3, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.35,
                           lw=0)
                sc = ax.scatter(home[s, 0], home[s, 1], c=d["t"][s], s=11, cmap="RdBu_r", vmin=-lim, vmax=lim,
                                edgecolors="k", linewidths=0.25)
                ax.axvline(342.0, color="k", lw=0.4)
                ax.invert_yaxis()
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_title(f"{nm}: {ep}\n{int(s.sum())} pts FWE .05 (+{int((s & (d['obs'] > 0)).sum())} / "
                             f"-{int((s & (d['obs'] < 0)).sum())}); {d['n_perm']} perms", fontsize=8)
        if sc is not None:
            fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.5, extend="both", label="t (change / null SD)")
        fig.suptitle(f"{set_name}: all-animals-pooled fit, per-point loading change vs the label-shuffle null "
                     "(max-statistic); image left = contralesional")
        frf.save(fig, out / "cohort_null_maps.png", """
Loading fraction per mesh point of the all-animals-pooled fit's lick factor, COHORT_<epoch> - COHORT_pre. Colour = t: each
point's change divided by its SD over the permutation null (within-animal label shuffle, both fits redone), and its family-wise p is
the share of permutations whose MAX |t| over all points reaches it: ringed = FWE p < .05 (large, opaque, black edge).
Red = more loading after the stroke; one colour scale (t) for all panels, so low-loading points that change
reliably are visible.""")


def spectral_figure(P: dict, out: Path, set_name: str) -> None:
    from wfield_local import face_rhythm_figures as frf
    ecol = frc.epoch_colors()
    pts, fr = P["points"], np.asarray(P["freqs"], float)
    with frf.style():
        fig, axs = frf.subplots(1, 2, figsize=(12.5, 3.8), squeeze=False, sharey=True)
        for c_i, nm in enumerate(FACTORS):
            ax = axs[0, c_i]
            for ep in frc.POST_EPOCHS:
                if ("spec", nm, ep) not in pts:
                    continue
                d = pts[("spec", nm, ep)]
                ax.plot(fr, d["t"], color=ecol[ep], lw=1.3, label=ep)
                ax.scatter(fr[d["sig"]], d["t"][d["sig"]], s=22, color=ecol[ep], edgecolor="k", lw=0.5, zorder=4)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("frequency (Hz)")
            ax.set_title(f"{nm}: spectral loading change / null SD", fontsize=9)
        axs[0, 0].set_ylabel("t (change / null SD)")
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{set_name}: all-animals-pooled fit, spectral change vs the label-shuffle null")
        frf.save(fig, out / "cohort_null_spectral.png", """
Spectral loading (fraction of total) of the all-animals-pooled fit's lick factor, COHORT_<epoch> - COHORT_pre, divided
per frequency by its SD over the permutation null; filled dots = family-wise p < .05 by the max |t| over the 36
frequencies.""")


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    fdir = a.fdir or frc.default_set_dir(set_name=a.set)
    out = frc.readouts_dir(a.set) / "epoch_shape_cohort"
    writeguard.assert_writable(out)
    names = frc.factor_names(fdir)
    csv, pkl = (figure_layout.sidecar_for(out, "cohort_null", e) for e in (".csv", ".pkl"))
    if a.plot_only:
        T, P = pd.read_csv(csv), pd.read_pickle(pkl)
    else:
        T, P = compute(fdir, names)
        T.to_csv(csv, index=False, float_format="%.4g")
        pd.to_pickle(P, pkl)
    pd.set_option("display.width", 220)
    if len(T):
        for m in SCALARS:
            d = T[T.measure == m].assign(cell=lambda x: [f"{o:+.3g} p{p:.3f}{'*' if h < .05 else ''}"
                                                          for o, p, h in zip(x.obs, x.p, x.p_holm)])
            print(f"\n{m}: observed (p, * = Holm < .05)\n"
                  + d.pivot_table(index="factor", columns="epoch", values="cell", aggfunc="first").reindex(names).to_string())
    for (kind, nm, ep), d in sorted(P["points"].items()):
        if nm in FACTORS:
            print(f"{kind} {nm} {ep}: {int(d['sig'].sum())} / {len(d['sig'])} FWE .05, "
                  f"{int((d['p_unc'] < .05).sum())} uncorrected (n = {d['n_perm']})")
    if len(T):
        scalar_figure(T, out, fdir.name, names)
        map_figure(P, out, fdir.name)
        spectral_figure(P, out, fdir.name)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
