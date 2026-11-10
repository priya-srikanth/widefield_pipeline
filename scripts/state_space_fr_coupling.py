"""Does the movement -> cortex relation itself change after the stroke? (handoff 11-09 ~12:15 open item b; Priya 11-09
"go ahead with all")

The movement-removed maps (Figs 9.1-9.3 b-d, `state_space_chronic_maps.py` WF_SS_FR=Bs / Ks / KBs) subtract a FROZEN
pre-stroke fit: they assume each animal's cortex responds to a given trial-to-trial movement difference as it did
before the stroke. This script measures that relation per session and asks whether it changes.

Trials, covariates and responses as WF_SS_FR=KBs: first-lick trials with both FaceRhythm (master_pre_v1, 0.0-0.5 s
after the first lick; lick sum + the 6 non-lick factors) and 3-D kinematics + contact (n_licks, n_contacts, rt_s,
reach_mm, speed, side_ml, out_ap, down_dv), 15 covariates; response = the lick-aligned 0.1-0.5 s window, cohort SVD
basis (100 components, orthonormal: component sums of squares = pixel sums of squares). WITHIN-session x spout
variation only (covariates and responses centered on each session x spout mean; spouts with >= 3 trials), so neither
spout identity nor any session-level change enters the coupling. Covariates scaled by the animal's pre within-SD.

Per session s, coupling B_s (15 x 100, least squares) and against the animal's PRE coupling B_ref (all pre sessions;
for a pre session, the other pre sessions -- the natural null):
  gain      <B_s, B_ref> / |B_ref|^2: strength ALONG the pre coupling (1 = unchanged; noise in B_s averages out, B_ref
            is fitted on other sessions, so unbiased)
  off       off-axis coupling energy, unbiased by split halves (odd / even trials): <B_s1 - a_1 b, B_s2 - a_2 b> with
            a_i = <B_si, b>, b = B_ref / |B_ref|; relative to the pre coupling energy |B_ref|^2 (a NEW direction of
            coupling; pre sessions give its pre-to-pre heterogeneity)
  R2_ref    variance of the session's within-centered responses explained by the PRE coupling (cross-validated by
            construction: B_ref never saw the session)
also per covariate group (kinematics 8, face 7) from the same joint fit. Change vs own pre (`boot_delta`, animals ->
sessions, 2,000; * = CI excludes 0); per-animal values printed first (rule 8). Maps: per animal x epoch the pooled
coupling of four covariates (pixels, units of the animal's full pre response RMS per pre SD), mean of 4 animals.

    WF_SS_ALIGN=lick WF_SS_BASIS=svd WF_SS_MAPBASE=rest python scripts/state_space_fr_coupling.py
Outputs: <state_space_recovery>/fr_coupling{tag}/{sessions,changes}.csv, maps.npz; page_figs/ss_fr_coupling{tag}.png.
EXPLORATORY.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["WF_SS_FR"] = "KBs"                      # the KBs covariate set (state_space_chronic_maps reads it at import)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_chronic_maps as C  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402
from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE  # noqa: E402

OUT = S.ROOT / f"fr_coupling{S.tag()}"
EPS = ["acute", "subacute", "chronic"]
COVS = C.KIN_COLS + ["FR lick sum"] + [f"FR f{k}" for k in (0, 1, 3, 6, 9, 10)]
GROUPS = {"all": slice(0, 15), "kinematics": slice(0, 8), "face": slice(8, 15)}
SHOW = ["n_licks", "n_contacts", "reach_mm", "side_ml", "FR lick sum"]


def session_trials(lab, Y, pos):
    """(X (n, 15), R (n, K), spout) of the session's covariate-matched trials, within session x spout centered."""
    _, resp_b = S.BAL_BINS[S.ALIGN]
    Xf = C.fr_covariates(lab, len(pos))
    if Xf is None:
        return None
    Xk = C.kin_covariates(lab, len(pos))
    X = np.column_stack([Xk, C.fr_design(Xf)])
    ok = np.isfinite(X).all(1)
    q = [lab, None, None, Y, None, X, ok, pos]
    Xw, Rw = C._within(q, resp_b)
    return (Xw, Rw) if len(Xw) >= 30 else None


def fit(X, R):
    return np.linalg.lstsq(X, R, rcond=None)[0]                                     # (15, K)


def main():
    if S.BASIS != "svd" or S.ALIGN != "lick":
        sys.exit("state_space_fr_coupling: run with WF_SS_ALIGN=lick WF_SS_BASIS=svd (the Fig 9 frame)")
    OUT.mkdir(parents=True, exist_ok=True)
    Uc = np.load(C.BASIS_DIR / "U_cohort.npy").astype(np.float64)
    unit = C.full_unit(C.get_resp("full"), Uc)                                      # the KBs maps' unit
    data, _ = S.load_sessions()
    rows, emaps = [], {}
    for an, sess in sorted(data.items()):
        T = {}
        for lab, ep, _day, Y, pos in sess:
            if ep in ("pre", *EPS):
                t = session_trials(lab, Y, pos)
                if t is not None:
                    T[lab] = (ep, *t)
        pre = [lab for lab, v in T.items() if v[0] == "pre"]
        sd = np.concatenate([T[lab][1] for lab in pre]).std(0)
        sd[sd == 0] = 1.0
        T = {lab: (ep, X / sd, R) for lab, (ep, X, R) in T.items()}
        for ep in ["pre", *EPS]:                                                    # pooled epoch coupling for the maps
            labs = [lab for lab, v in T.items() if v[0] == ep]
            if labs:
                B = fit(np.concatenate([T[lab][1] for lab in labs]), np.concatenate([T[lab][2] for lab in labs]))
                emaps[(an, ep)] = (B / unit[an]) @ Uc.T
        for lab, (ep, X, R) in T.items():
            ref = [q for q in pre if q != lab]
            Bref = fit(np.concatenate([T[q][1] for q in ref]), np.concatenate([T[q][2] for q in ref]))
            Bs = fit(X, R)
            h = [fit(X[i::2], R[i::2]) for i in (0, 1)]
            r = {"animal": an, "label": lab, "epoch": ep, "n_trials": len(X)}
            for g, sl in GROUPS.items():
                br, bs, b1, b2 = Bref[sl].ravel(), Bs[sl].ravel(), h[0][sl].ravel(), h[1][sl].ravel()
                e = br @ br
                u = br / np.sqrt(e)
                a1, a2 = b1 @ u, b2 @ u
                r[f"gain {g}"] = float(bs @ br / e)
                r[f"off {g}"] = float((b1 - a1 * u) @ (b2 - a2 * u) / e)
            res = R - X @ Bref
            r["R2_ref"] = float(1 - (res ** 2).sum() / (R ** 2).sum())
            rows.append(r)
        print(f"  {an}: {len(T)} sessions ({len(pre)} pre)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "sessions.csv", index=False)
    rng = np.random.default_rng(0)
    recs = []
    reads = [f"gain {g}" for g in GROUPS] + [f"off {g}" for g in GROUPS] + ["R2_ref"]
    for col in reads:
        for ep in EPS:
            pre = {a: g[g.epoch == "pre"][col].tolist() for a, g in D.groupby("animal")}
            post = {a: g[g.epoch == ep][col].tolist() for a, g in D.groupby("animal")}
            post = {a: v for a, v in post.items() if v and pre.get(a)}
            per = {a: round(float(np.mean(post[a]) - np.mean(pre[a])), 3) for a in post}
            iv = kit.boot_delta(post, pre, rng)
            pm = float(np.mean([np.mean(v) for v in pre.values()]))
            recs.append({"readout": col, "epoch": ep, "pre": pm, "change": iv.point, "lo": iv.lo, "hi": iv.hi,
                         "k_up": sum(v > 0 for v in per.values()), "n": len(per),
                         "star": bool(iv.lo > 0 or iv.hi < 0), "per_animal": per})
            print(f"  {col} | {ep}: pre {pm:+.3f} {per} -> {iv.point:+.3f} [{iv.lo:+.3f}, {iv.hi:+.3f}]", flush=True)
    Cg = pd.DataFrame(recs)
    Cg.to_csv(OUT / "changes.csv", index=False)
    ans = sorted({a for a, _ in emaps})
    M = {ep: np.mean([emaps[(a, ep)] for a in ans if (a, ep) in emaps], 0) for ep in ["pre", *EPS]}
    np.savez_compressed(OUT / "maps.npz", covs=np.array(COVS), **{ep: m.astype(np.float32) for ep, m in M.items()})

    # figure: coupling maps (rows = covariates, cols = epochs) + the per-session readouts
    mask = np.load(C.BASIS_DIR / "mask.npy").reshape(np.load(S.SVD.parents[1] / "atlas.npy").shape)
    from wfield_local import beta_maps
    from wfield_local.atlas_overlay import region_edges
    edges = region_edges(np.load(S.SVD.parents[1] / "atlas.npy"))
    brain = beta_maps.allen_mask()
    fig = plt.figure(figsize=(15, 3.0 * len(SHOW) + 3.6))
    gs = fig.add_gridspec(len(SHOW) + 1, 5, height_ratios=[1] * len(SHOW) + [1.15])
    for i, cv in enumerate(SHOW):
        j = COVS.index(cv)
        # one scale per row over ALL epochs (Priya 11-09: the pre-set scale saturated acute n_licks / chronic reach);
        # each panel prints its own 99th-percentile |coupling| so a faint epoch stays readable
        vmax = max(float(np.nanpercentile(np.abs(M[ep][j]), 99)) for ep in ["pre", *EPS]) or 1.0
        for k, ep in enumerate(["pre", *EPS]):
            ax = fig.add_subplot(gs[i, k])
            C.draw(ax, M[ep][j], mask, edges, vmax, brain=brain)
            ax.text(0.02, 0.02, f"p99 |b| {np.nanpercentile(np.abs(M[ep][j]), 99):.3f}", transform=ax.transAxes,
                    fontsize=7.5, color="0.25")
            if i == 0:
                ax.set_title(ep, fontsize=10)
            if k == 0:
                ax.set_ylabel(cv, fontsize=9)
        cax = fig.add_subplot(gs[i, 4])
        cax.set_axis_off()
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax))
        fig.colorbar(sm, ax=cax, fraction=0.5, label="per pre SD\n(pre response RMS)")
    for k, g in enumerate(["gain", "off", "R2_ref"]):
        ax = fig.add_subplot(gs[-1, k])
        cols = [f"{g} {x}" for x in GROUPS] if g != "R2_ref" else ["R2_ref"]
        for c, col in enumerate(cols):
            for e, ep in enumerate(["pre", *EPS]):
                x = c + (e - 1.5) * 0.18
                for a, gg in D[D.epoch == ep].groupby("animal"):
                    ax.scatter(np.full(len(gg), x), gg[col], s=12, marker=ANIMAL_MARKER.get(a, "o"),
                               facecolors="none", edgecolors=EPOCH_LINE.get(ep, "0.4"), lw=0.8)
                if ep != "pre":
                    rr = Cg[(Cg.readout == col) & (Cg.epoch == ep)].iloc[0]
                    if rr.star:
                        sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], fontsize=9)
        ax.set_xticks(range(len(cols)), [col.split(" ", 1)[-1] if g != "R2_ref" else "all" for col in cols])
        ax.axhline(1 if g == "gain" else 0, color="0.6", lw=0.8)
        ax.set_title({"gain": "gain along the pre coupling\n(1 = unchanged)",
                      "off": "off-axis coupling energy / pre energy\n(split-half, unbiased)",
                      "R2_ref": "within-spout variance explained\nby the PRE coupling"}[g], fontsize=9)
        if g == "off":
            ax.set_yscale("symlog", linthresh=1.0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(C._wrap("Does the movement -> cortex relation change? Within-session x spout trial-to-trial coupling "
                         "of the KBs covariates (kinematics + contact + face), lick-aligned 0.1-0.5 s, cohort SVD basis; "
                         "maps = pooled epoch coupling, mean of 4 animals; points = sessions (shape = animal, color = "
                         "epoch); * = boot_delta 95 % CI of the change vs own pre excludes 0", 140), fontsize=10)
    fig.tight_layout()
    fp = C.FIGS / f"ss_fr_coupling{S.tag()}.png"
    sig_marks.reserve_headroom(fig)                  # stars in the top band, no top spine (Priya 10-30 / 11-03)
    save_png_svg(fig, fp)
    print(f"wrote {fp}", flush=True)


if __name__ == "__main__":
    main()
