"""State-space recovery on single trials, session vs session (Priya 11-07; plan docs/STATE_SPACE_RECOVERY_PLAN.md,
P1-P3b; readouts and detection limits from scripts/state_space_sim.py).

Data: per-trial area extracts (`first_cue`, first-lick trials = contact + incomplete, -0.5..2.0 s from the cue in
0.1 s bins, 62 Allen areas incl. ALM / tjM1, `cortex` dropped). Inputs final (34, 11-07). Left = ipsilesional.

Every session is compared with EACH pre-stroke session of its animal (itself excluded), at matched trial counts per
spout, in the pre-stroke top-k space of the pre sessions left after removing both. The values of pre sessions are the
null; every readout is also given as z against its animal's pre null.

Readouts (session vs session, median over the pre comparisons; sim detection limits in the plan):
  r, amp        pattern correlation and amplitude ratio
  rot_gain      ladder L4 - L3: rotation beyond per-area gains (>= ~20-30° in the main plane is detectable)
  out_var       variance outside the pre space (NOT specific: region gains raise it too)
  hemi_gain     mean log gain ipsilesional (left) - contralesional (right) areas
  shift_snr     squared shared-trajectory difference / its trial-noise expectation (condition-independent shift)
  spread_norm   within-spout trial spread in the pre space / amp^2 (amplitude-free spread)
Versus the multi-session pre template (less noisy gains): out_resid_t (outside-space variance per-area gains cannot
explain), hemi_gain_t, gain_t_<area>.
Trial clouds in ONE fixed per-animal pre space (P3b): alpha = where the session mean sits on the line from the pre
template (0) to the chronic template (1); chronic_like = share of trials nearer the chronic than the pre template;
aligned = trial spread along that line / mean spread per dimension (>1 = spread aimed at the eventual solution).
Path (P2): dist_pre = distance of the session's spout means to the pre sessions, step = distance to the previous
session; both in units of the pre-to-pre median.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import state_space_sim as S  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402  (PNG + svg/ twin, Priya 11-06)

OUT = S.ROOT / f"trials{S.tag()}"   # WF_SS_ALIGN suffix
EPOCHS = ["pre", "acute", "subacute", "chronic"]
MIN_TRIALS = 5
N_RESAMPLE = 3
READ = ["r", "amp", "rot_gain", "out_var", "hemi_gain", "shift_snr", "spread_norm", "spread_raw",
        "out_resid_t", "hemi_gain_t", "bal_oro", "bal_oro_far", "bal_all", "bal_all_far",
        "alpha", "chronic_like", "aligned", "dist_pre", "step"]


OROFACIAL = ("SSp-m", "SSp-n", "SSs")
BASE_BINS, RESP_BINS = S.BAL_BINS[S.ALIGN]          # cue: -0.5..0 s and 0..1 s; lick / splice: see state_space_sim


def balance(M, cols, sel):
    """log2 (contralesional / ipsilesional) response size of one spout's mean (bins x areas); NaN if either <= 0."""
    side = S.hemi(S.evoked(M), cols, sel)
    if not (side["contra"] > 0 and side["ipsi"] > 0):
        return np.nan
    return float(np.log2(side["contra"] / side["ipsi"]))


def matched(Ya, pa, Yb, pb, rng):
    """Per-spout matched subsample; spouts with < MIN_TRIALS in either session dropped."""
    ia, ib = [], []
    for p in S.POS:
        a, b = np.flatnonzero(pa == p), np.flatnonzero(pb == p)
        n = min(len(a), len(b))
        if n >= MIN_TRIALS:
            ia.append(rng.permutation(a)[:n])
            ib.append(rng.permutation(b)[:n])
    if len(ia) < 4:
        return None
    ia, ib = np.concatenate(ia), np.concatenate(ib)
    return Ya[ia], pa[ia], Yb[ib], pb[ib]


def noise_var(Y, pos):
    """Per-spout trial variance around the spout mean (bins x areas), pooled."""
    r = [Y[pos == p] - Y[pos == p].mean(0) for p in S.POS if (pos == p).sum() > 1]
    return np.concatenate(r).var(0, ddof=1)


def pair_readouts(Ya, pa, Yb, pb, P, left):
    base = S.readouts(Ya, pa, Yb, pb, P)
    A, B = S.spout_means(Ya, pa), S.spout_means(Yb, pb)
    ok = np.isfinite(A).all(axis=(1, 2)) & np.isfinite(B).all(axis=(1, 2))
    g, _ = S.fit_gain(S.flat(A[ok]), S.flat(B[ok]))
    lg = np.log(np.clip(g, 1e-3, None))
    As, Bs = A[ok].mean(0), B[ok].mean(0)
    na = len(pa) / ok.sum()
    exp = ((noise_var(Ya, pa) + noise_var(Yb, pb)) / (na * ok.sum())).sum()
    return {"r": base["r"], "amp": base["amp"], "rot_gain": base["rot_gain"], "out_var": base["out_var"],
            "hemi_gain": float(lg[left].mean() - lg[~left].mean()),
            "shift_snr": float(((As - Bs) ** 2).sum() / exp),
            "spread_norm": base["spread_norm"], "spread_raw": base["spread"]}


def template(sessions):
    return np.nanmean(np.stack([S.spout_means(y, p) for _, _, _, y, p in sessions]), axis=0)


def fixed_space(T):
    P = S.topk(T)
    mu = S.flat(T[np.isfinite(T).all(axis=(1, 2))]).mean(0)
    return P, mu


def frames(sess):
    """THE pre -> chronic frame of one animal (the 2-D maps, per-trial positions, contact split): the fixed pre space
    (P, mu) and per spout the pre template point zp, the unit x direction u (pre -> chronic), the unit acute-departure
    direction v (acute - pre with its x part removed), the pre -> chronic length L, the spout's pre template (bins x
    areas) and the share of acute - pre orthogonal to x ("orth"). Spouts missing any of the three epoch templates are
    left out. Returns (P, mu, {spout: dict})."""
    pre = [s for s in sess if s[1] == "pre"]
    T = template(pre)
    P, mu = fixed_space(T)
    z = {ep: (template([s for s in sess if s[1] == ep]) - mu) @ P for ep in ("pre", "acute", "chronic")}
    fr = {}
    for si, p in enumerate(S.POS):
        zp, zc, za = (z[e][si].ravel() for e in ("pre", "chronic", "acute"))
        if not (np.isfinite(zp).all() and np.isfinite(zc).all() and np.isfinite(za).all()):
            continue
        d = zc - zp
        L = np.linalg.norm(d)
        u = d / L
        a = za - zp
        ap = a - (a @ u) * u
        fr[p] = {"zp": zp, "u": u, "v": ap / np.linalg.norm(ap), "L": L, "Tpre": T[si],
                 "orth": float(np.linalg.norm(ap) / np.linalg.norm(a))}
    return P, mu, fr


def xy(q, f):
    """Position of rows q (n, bins * k; already minus f["zp"]) in frame f: x along pre (0) -> chronic (1), y along the
    acute departure, both in units of the pre -> chronic length."""
    return q @ f["u"] / f["L"], q @ f["v"] / f["L"]


def cloud(Y, pos, Tpre, Tchr, P, mu):
    """Trial-cloud readouts in the fixed pre space (P3b)."""
    zp, zc = (Tpre - mu) @ P, (Tchr - mu) @ P                          # spout x bins x k
    z = (Y - mu) @ P                                                   # trials x bins x k
    al, cl, dev = [], [], []
    for s, p in enumerate(S.POS):
        m = pos == p
        if m.sum() < MIN_TRIALS or not (np.isfinite(zp[s]).all() and np.isfinite(zc[s]).all()):
            continue
        d = (zc[s] - zp[s]).ravel()
        if not np.linalg.norm(d):
            continue
        zt = z[m].reshape(m.sum(), -1)
        a = (zt - zp[s].ravel()) @ d / (d @ d)
        al.append(a)
        cl.append(((zt - zc[s].ravel()) ** 2).sum(1) < ((zt - zp[s].ravel()) ** 2).sum(1))
        u = d / np.linalg.norm(d)
        c = zt - zt.mean(0)
        dev.append(((c @ u) ** 2).mean() / ((c ** 2).sum(1).mean() / c.shape[1]))
    if not al:
        return {"alpha": np.nan, "chronic_like": np.nan, "aligned": np.nan}
    return {"alpha": float(np.mean([a.mean() for a in al])), "chronic_like": float(np.mean(np.concatenate(cl))),
            "aligned": float(np.mean(dev))}


def vec(Y, pos, P, mu):
    """Spout means in the fixed space + the trial-noise variance of each mean (sum over bins x k / n)."""
    z = (Y - mu) @ P
    m = np.full((6,) + z.shape[1:], np.nan)
    nv = np.full(6, np.nan)
    for s, p in enumerate(S.POS):
        t = z[pos == p]
        if len(t) >= MIN_TRIALS:
            m[s] = t.mean(0)
            nv[s] = t.reshape(len(t), -1).var(0, ddof=1).sum() / len(t)
    return m, nv


def vdist(a, b):
    """Distance between two sessions' spout means with the trial-noise expectation REMOVED (unbiased squared
    distance per spout, averaged over shared spouts), so a session with few trials does not look farther away."""
    (ma, na), (mb, nb) = a, b
    ok = np.isfinite(ma).all(axis=(1, 2)) & np.isfinite(mb).all(axis=(1, 2))
    if not ok.any():
        return np.nan
    d2 = (((ma[ok] - mb[ok]) ** 2).reshape(ok.sum(), -1).sum(1) - na[ok] - nb[ok]).mean()
    return float(np.sqrt(max(d2, 0.0)))


LANDING = S.LANDING


def relabel(lab, Y, pos, mode, L):
    """Condition labels (Priya 11-07, like-for-like executed landings; `state_space_landing.py`):
    target = the target spout (default); on_target = target spout, only trials whose 3-D first lick landed in the
    target's own pre-stroke bin; landing = the landing bin, whatever the target (the executed movement).
    contact / incomplete (Priya 11-07: contact + incomplete primary, contact-only robustness, incomplete-only
    descriptive): target spout, only first licks whose 3-D lick touched / did not touch the spout (landing csv
    `contact`, the 3-D first lick matched to the trial)."""
    if mode == "target":
        return Y, pos
    g = L.get(lab)
    if mode in ("contact", "incomplete"):
        c = np.full(len(pos), np.nan)
        if g is not None:
            c[g.row.to_numpy()] = g.contact.to_numpy(float)
        keep = c == (1.0 if mode == "contact" else 0.0)
        return Y[keep], pos[keep]
    land = np.full(len(pos), None, object)
    if g is not None:
        land[g.row.to_numpy()] = g.land.to_numpy()
    keep = np.array([x is not None and x == x for x in land])
    if mode == "on_target":
        keep &= land == pos
        return Y[keep], pos[keep]
    return Y[keep], land[keep].astype(str)


def run(seed=0, mode="target"):
    rng = np.random.default_rng(seed)
    L = {}
    if mode != "target":
        D = pd.read_csv(LANDING)
        D = D[D.contact.notna()] if mode in ("contact", "incomplete") else D[D.land.notna()]
        L = {lab: g for lab, g in D.groupby("label")}
    data, cols_all = S.load_sessions(lambda lab, Y, pos: relabel(lab, Y, pos, mode, L))
    rows, pairs = [], []
    for an, sess in sorted(data.items()):
        cols = S.cols_of(cols_all, an)                                 # per animal in the LocaNMF basis
        left = np.array([c.endswith("_left") for c in cols])
        pre = [s for s in sess if s[1] == "pre"]
        chr_ = [s for s in sess if s[1] == "chronic"]
        P_all, mu_all = fixed_space(template(pre))
        pre_vecs = {s[0]: vec(s[3], s[4], P_all, mu_all) for s in pre}
        pp = [vdist(pre_vecs[a], pre_vecs[b]) for a in pre_vecs for b in pre_vecs if a < b]
        pre_lab = [s[0] for s in pre]
        cons = [vdist(pre_vecs[a], pre_vecs[b]) for a, b in zip(pre_lab[:-1], pre_lab[1:])]
        prev = None
        vecs = {}
        for lab, ep, day, Y, pos in sess:
            vals = []
            for b in pre:
                if b[0] == lab:
                    continue
                rest = [s for s in pre if s[0] not in (lab, b[0])]
                P, _ = fixed_space(template(rest))
                for _ in range(N_RESAMPLE):
                    m = matched(Y, pos, b[3], b[4], rng)
                    if m is not None:
                        vals.append(pair_readouts(*m, P, left))
            row = {"label": lab, "animal": an, "epoch": ep, "day": day, "n_trials": len(pos)}
            row.update(pd.DataFrame(vals).median().to_dict() if vals else {})
            # versus the multi-session template (pre: leave this session out)
            others = [s for s in pre if s[0] != lab]
            T = template(others)
            Pt, _ = fixed_space(T)
            A = S.spout_means(Y, pos)
            ok = np.isfinite(A).all(axis=(1, 2)) & np.isfinite(T).all(axis=(1, 2))
            for s_ in range(6):
                if (pos == S.POS[s_]).sum() < MIN_TRIALS:
                    ok[s_] = False
            Af, Tf = S.flat(A[ok]), S.flat(T[ok])
            g, b0 = S.fit_gain(Af, Tf)
            R = Af - (Tf * g + b0)
            lg = np.log(np.clip(g, 1e-3, None))
            row["out_resid_t"] = float((R - R @ Pt @ Pt.T).var() / (Af - Af.mean(0)).var())
            row["hemi_gain_t"] = float(lg[left].mean() - lg[~left].mean())
            row.update({f"gain_t_{c}": float(v) for c, v in zip(cols, g)})
            # hemispheric BALANCE as in FINDINGS 0b: response SIZE (0-1 s minus -0.5-0 s), log2 contra / ipsi,
            # within spout, for the orofacial areas and for all areas (the slope above is NOT a size measure)
            for nm, sel in (("oro", OROFACIAL), ("all", None)):
                for ring, sp in (("", range(6)), ("_far", range(3, 6))):
                    v = [balance(A[s_], cols, sel) for s_ in sp if ok[s_]]
                    row[f"bal_{nm}{ring}"] = float(np.nanmean(v)) if v else np.nan
            # trial cloud in the fixed space; chronic template leaves this session out
            Tc = template([s for s in chr_ if s[0] != lab]) if len(chr_) > 1 or ep != "chronic" else None
            Tp = T if ep == "pre" else template(pre)
            row.update(cloud(Y, pos, Tp, Tc, P_all, mu_all) if Tc is not None else {})
            v = vec(Y, pos, P_all, mu_all)
            row["dist_pre"] = float(np.nanmedian([vdist(v, pre_vecs[b]) for b in pre_vecs if b != lab])
                                    / np.median(pp))
            row["step"] = (vdist(v, prev) / np.median(cons)) if prev is not None else np.nan
            prev = v
            rows.append(row)
            vecs[lab] = (ep, day, v)
        labs = list(vecs)
        for i, a in enumerate(labs):
            for b in labs[i + 1:]:
                pairs.append({"animal": an, "a": a, "b": b, "ep_a": vecs[a][0], "ep_b": vecs[b][0],
                              "gap": abs(vecs[b][1] - vecs[a][1]), "dist": vdist(vecs[a][2], vecs[b][2]) / np.median(pp)})
    df = pd.DataFrame(rows)
    for _an, g in df.groupby("animal"):                                # z against each animal's pre null
        nul = g[g.epoch == "pre"]
        for m in READ:
            if m in g:
                df.loc[g.index, f"z_{m}"] = (g[m] - nul[m].mean()) / nul[m].std(ddof=1)
    return df, pd.DataFrame(pairs)


def drift(pairs):
    """No-stroke drift floor: distance vs day gap WITHIN pre and WITHIN chronic (both free of the acute transition)."""
    out = []
    for (an, ep), g in pairs[pairs.ep_a == pairs.ep_b].groupby(["animal", "ep_a"]):
        if ep in ("pre", "chronic") and len(g) > 3:
            sl, ic = np.polyfit(g.gap, g.dist, 1)
            out.append({"animal": an, "within": ep, "slope_per_day": sl, "intercept": ic, "max_gap": g.gap.max(), "n": len(g)})
    d = pd.DataFrame(out)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print("\n=== drift: noise-corrected distance (pre-to-pre median units) vs day gap ===")
        print(d.to_string(index=False))
    return d


def summarize(df):
    read = [m for m in READ if m in df]                                # incomplete-only: no session-pair readouts
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print("\n=== per animal x epoch, raw (rule 8: read this before any interval) ===")
        print(df.groupby(["animal", "epoch"])[read].mean().reindex(EPOCHS, level="epoch"))
    rng = np.random.default_rng(0)
    out = []
    for m in read:
        pre = {a: g[g.epoch == "pre"][m].dropna().tolist() for a, g in df.groupby("animal")}
        for ep in EPOCHS[1:]:
            post = {a: g[g.epoch == ep][m].dropna().tolist() for a, g in df.groupby("animal")}
            post = {a: v for a, v in post.items() if v}
            iv = kit.boot_delta(post, pre, rng)
            sign = sum(np.mean(post[a]) > np.mean(pre[a]) for a in post if pre.get(a))
            out.append({"readout": m, "epoch": ep, "delta": getattr(iv, "point", np.nan), "lo": getattr(iv, "lo", np.nan),
                        "hi": getattr(iv, "hi", np.nan), "n_animals": len(post), "n_up": sign})
    res = pd.DataFrame(out)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print("\n=== change vs pre: animal-weighted boot_delta (95 %), n_up = animals above their pre mean ===")
        print(res.to_string(index=False))
    return res


def figure(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    panels = [("z_r", "pattern r with pre"), ("z_amp", "amplitude"), ("z_rot_gain", "rotation beyond gain"),
              ("z_out_resid_t", "outside pre space, beyond gain"), ("z_hemi_gain_t", "gain ipsi - contra (log)"),
              ("z_shift_snr", "shared shift (noise units)"), ("z_spread_norm", "trial spread (amp-free)"),
              ("alpha", "position pre (0) -> chronic (1)"), ("chronic_like", "share of chronic-like trials"),
              ("z_aligned", "spread along pre->chronic"), ("z_dist_pre", "distance from pre sessions"),
              ("z_step", "day-to-day step")]
    fig, axs = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    for ax, (col, title) in zip(axs.ravel(), panels):
        if col not in df:
            ax.set_visible(False)
            continue
        if col.startswith("z_"):
            ax.axhspan(-1.96, 1.96, color="0.9", zorder=0)
        for an, g in df.groupby("animal"):
            for ep, h in g.groupby("epoch"):
                ax.scatter(h.day, h[col], marker=ANIMAL_MARKER.get(an, "o"), facecolors="none",
                           edgecolors=EPOCH_LINE.get(ep, "0.5"), s=26, linewidths=1)
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        ax.set_title(title + (" (z vs pre)" if col.startswith("z_") else ""), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("days from stroke")
    hs = [plt.Line2D([], [], marker=ANIMAL_MARKER[a], ls="", mfc="none", mec="k", label=a) for a in sorted(ANIMAL_MARKER)]
    hs += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=EPOCH_LINE[e], label=e) for e in EPOCHS]
    fig.legend(handles=hs, loc="upper center", ncol=8, frameon=False)
    fig.suptitle("EXPLORATORY: single-trial state space vs pre, session vs session (gray = pre 95 % range)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png_svg(fig, out / "state_space_trials_over_days.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--condition", default="target", choices=["target", "on_target", "landing", "contact", "incomplete"])
    a = ap.parse_args()
    out = Path(a.out) if a.condition == "target" else Path(a.out).with_name(f"trials_{a.condition}{S.tag()}")
    out.mkdir(parents=True, exist_ok=True)
    df, pairs = run(mode=a.condition)
    df.to_csv(out / "state_space_trials_sessions.csv", index=False)
    pairs.to_csv(out / "state_space_trials_pairs.csv", index=False)
    drift(pairs).to_csv(out / "state_space_trials_drift.csv", index=False)
    summarize(df).to_csv(out / "state_space_trials_change_vs_pre.csv", index=False)
    figure(df, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
