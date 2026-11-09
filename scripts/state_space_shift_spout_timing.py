"""Is the PHASIC post-lick orofacial increase spout-specific in timing or size? (Priya 11-09 "sure go ahead" on the open
question of page sec 10: "whether the phasic part is spout-specific in timing (it was defined spout-independent)").

The eight-part split DEFINES the shared shift as spout-independent, so it cannot answer this; here the same region readout
as Figs 10.21-10.24 is taken per spout. Per trial, cohort SVD basis, first-lick-aligned (-1.0..+0.7 s, 0.1-s bins), minus
the session's RESTnl level, units of the animal's pre-template pixel RMS, region means (orofacial = tjM1 + SSp-m + SSp-n;
posterior-medial = RSPd + SSp-tr + SSp-ll; contra / ipsi). Session x spout mean trace; change vs the animal's own pre
trace FOR THAT SPOUT (so the spout's own pre-stroke timing is removed).
Readouts per spout: early (0..0.3 s) and late (0.3..0.7 s) change (nested bootstrap animals -> sessions, `boot_delta`,
2,000), traces animal-weighted.
TEST of spout specificity, per SESSION (post-lick bins 0..0.7 s; D = spouts x bins change vs the pre spout traces; pre
sessions against a leave-session-out pre trace = the drift floor): each spout's trials are split at random into two
halves (20 splits) and only cross-half products are kept, so trial noise does not enter:
  shared energy          mean(dA * dB), d = the spout-mean change trace
  spout-specific energy  mean(rA * rB), r = D - d (what differs between spouts, size or timing)
  specific TIMING energy the same for each spout's late-minus-early contrast of r (a spout-specific shift in timing)
(units: pre pixel RMS squared; template noise is shared by both halves and biases pre sessions up slightly, which is
conservative for an increase). Session values -> change vs own pre, `boot_delta`; per-animal values printed first (rule 8).
CONTACT split (from B27's trial table, `state_space_shift_behavior.py`, trials with a 3-D match): the trial's change vs
its spout's pre mean in each window, session means over trials whose 700-ms bout touched the spout (>= 1 contact) and
over trials with none, change vs own pre (all pre trials; no-contact trials are rare before the stroke).
(A spout-label shuffle inside post sessions was tried first and dropped: it keeps the pre spout structure in the reference,
so its null is not a null.)
Outputs: <state_space_recovery>/shift_spout_timing{tag}/{traces,changes,specificity}.csv,
page_figs/ss_shift_spout_timing{tag}.png. EXPLORATORY.
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
import state_space_recovery_trials as R  # noqa: E402
import state_space_shift_behavior as SB  # noqa: E402
import state_space_sim as S  # noqa: E402
import state_space_trajectories as TR  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402
from wfield_local.orofacial_cohort import EPOCH_LINE, pos_disp  # noqa: E402

EPS = SB.EPS
POS = S.POS
OUT = S.ROOT / f"shift_spout_timing{S.tag()}"
FIGS = S.ROOT / "page_figs"
WINS = {"early (0..0.3 s)": (0.0, 0.3), "late (0.3..0.7 s)": (0.3, 0.7)}
NSPLIT = 20
MIN_TR = 4
ENER = ["shared", "spout-specific", "specific timing"]
SIDE_COLOR = {"ipsi": "#1f77b4", "mid": "0.35", "contra": "#d62728"}


def pos_style(p):
    ring, side = p.split("_")
    return dict(color=SIDE_COLOR[side], ls="-" if ring == "close" else "--")


def spout_mean(v, pos):
    """trials x bins -> spouts x bins (NaN where a spout has no trial)."""
    return np.stack([v[pos == p].mean(0) if (pos == p).any() else np.full(v.shape[1], np.nan) for p in POS])


def cv_energies(V, pos, P, tix, early, late, rng):
    """One session's (shared, spout-specific, specific timing) cross-validated energies; P = pre spouts x bins."""
    keep = [i for i, p in enumerate(POS) if (pos == p).sum() >= MIN_TR and np.isfinite(P[i]).all()]
    if len(keep) < 3:
        return (np.nan,) * 3
    out = []
    for _ in range(NSPLIT):
        A, B = [], []
        for i in keep:
            ix = rng.permutation(np.flatnonzero(pos == POS[i]))
            h = len(ix) // 2
            A.append(V[ix[:h]].mean(0) - P[i])
            B.append(V[ix[h:2 * h]].mean(0) - P[i])
        A, B = np.array(A), np.array(B)
        dA, dB = A.mean(0), B.mean(0)
        rA, rB = A - dA, B - dB
        cA = rA[:, late].mean(1) - rA[:, early].mean(1)
        cB = rB[:, late].mean(1) - rB[:, early].mean(1)
        out.append(((dA[tix] * dB[tix]).mean(), (rA[:, tix] * rB[:, tix]).mean(), (cA * cB).mean()))
    return tuple(np.mean(out, 0))


def main():
    if S.BASIS != "svd" or S.ALIGN != "lick":
        sys.exit("state_space_shift_spout_timing: needs WF_SS_BASIS=svd WF_SS_ALIGN=lick")
    from wfield_local import area_basis, beta_maps
    Uc = np.load(C.BASIS_DIR / "U_cohort.npy").astype(np.float64)
    atlas = np.load(S.SVD.parents[1] / "atlas.npy")
    mask = np.load(C.BASIS_DIR / "mask.npy").reshape(atlas.shape)
    flat = np.flatnonzero(mask.ravel())
    split = area_basis.basis_masks(area_basis.roi_masks(atlas.ravel()[flat], beta_maps.stat_mask().ravel()[flat],
                                                        dict(S._atlas_names()), flat, mask.shape))["split"]
    W = {}
    for reg, areas in SB.REGIONS.items():
        for side, hemi in SB.SIDES.items():
            px = np.logical_or.reduce([np.asarray(split[f"{a}_{hemi}"], bool) for a in areas if f"{a}_{hemi}" in split])
            W[f"{reg} {side}"] = Uc[px].mean(0)
    t, _ = TR.times()
    tix = np.flatnonzero(t > 0)                                                   # post-lick bins (0..0.7 s)
    wix = {w: np.flatnonzero((t >= a - 1e-9) & (t < b - 1e-9)) for w, (a, b) in WINS.items()}
    data, _ = S.load_sessions()
    # per session: (animal, label, epoch, {region: trials x bins}, pos)
    sess_tr = []
    for an, sess in sorted(data.items()):
        Tpre = R.template([s for s in sess if s[1] == "pre"])
        ok = np.isfinite(Tpre).all(axis=(1, 2))
        B0 = Tpre[ok] - S.flat(Tpre[ok]).mean(0)
        unit = float(np.sqrt((B0 ** 2).sum() / B0.size * B0.shape[-1] / Uc.shape[0]))
        for lab, ep, day, Y, pos in sess:
            if ep not in ("pre", *EPS) or not (C.REST / f"{lab}.npz").exists():
                continue
            c = C._rest(lab, an)
            if c is None:
                continue
            Z = (Y - c) / unit
            sess_tr.append((an, lab, ep, day, {r: Z @ w for r, w in W.items()}, np.asarray(pos)))
        print(f"  {an} done", flush=True)
    regs = list(W)

    # session x spout traces, change vs the animal's pre spout trace
    rows = []
    for an, lab, ep, day, V, pos in sess_tr:
        for r in regs:
            M = spout_mean(V[r], pos)
            for i, p in enumerate(POS):
                rows.append({"animal": an, "label": lab, "epoch": ep, "day": day, "region": r, "spout": p,
                             **{f"b{k}": M[i, k] for k in range(len(t))}})
    Tr = pd.DataFrame(rows)
    bcols = [f"b{k}" for k in range(len(t))]
    pre = Tr[Tr.epoch == "pre"].groupby(["animal", "region", "spout"])[bcols].mean()
    base = pre.reindex(pd.MultiIndex.from_frame(Tr[["animal", "region", "spout"]])).to_numpy()
    Dl = Tr.copy()
    Dl[bcols] = Tr[bcols].to_numpy() - base
    OUT.mkdir(parents=True, exist_ok=True)
    Dl.to_csv(OUT / "traces.csv", index=False)

    # ---- per-spout early / late change, nested bootstrap
    rng = np.random.default_rng(0)
    recs = []
    for r in regs:
        for p in POS:
            g = Dl[(Dl.region == r) & (Dl.spout == p)]
            for wn, ix in wix.items():
                g = g.assign(v=g[[bcols[k] for k in ix]].mean(1))
                for ep in EPS:
                    post = {a: h.v.dropna().tolist() for a, h in g[g.epoch == ep].groupby("animal")}
                    post = {a: v for a, v in post.items() if v and (g[g.epoch == "pre"].animal == a).any()}
                    pre_v = {a: g[(g.epoch == "pre") & (g.animal == a)].v.dropna().tolist() for a in post}
                    iv = kit.boot_delta(post, pre_v, rng)
                    up = sum(np.mean(post[a]) > np.mean(pre_v[a]) for a in post)
                    recs.append({"region": r, "spout": p, "window": wn, "epoch": ep, "change": iv.point, "lo": iv.lo,
                                 "hi": iv.hi, "k_up": up, "n": len(post)})
    CH = pd.DataFrame(recs)
    CH.to_csv(OUT / "changes.csv", index=False)

    # ---- contact split (B27 trial table): is the post-lick increase tied to touching the spout?
    TB = pd.read_csv(SB.OUT / "trials.csv.gz")
    TB = TB[TB.n_licks.notna()]
    TB["contact"] = np.where(TB.n_contacts > 0, "contact", "no contact")
    crec = []
    for r in regs:
        for wn in WINS:
            col = f"obs | {r} | {wn}"
            pre_v = {a: h.groupby("label")[col].mean().dropna().tolist() for a, h in TB[TB.epoch == "pre"].groupby("animal")}
            for ct in ("contact", "no contact"):
                for ep in EPS:
                    h = TB[(TB.epoch == ep) & (TB.contact == ct)]
                    post = {a: g.groupby("label")[col].mean().dropna().tolist() for a, g in h.groupby("animal")}
                    post = {a: v for a, v in post.items() if v and pre_v.get(a)}
                    per = {a: round(float(np.mean(v) - np.mean(pre_v[a])), 2) for a, v in post.items()}
                    iv = kit.boot_delta(post, pre_v, rng)
                    crec.append({"region": r, "window": wn, "trials": ct, "epoch": ep, "change": iv.point, "lo": iv.lo,
                                 "hi": iv.hi, "k_up": sum(v > 0 for v in per.values()), "n": len(per),
                                 "n_trials": len(h), "frac_trials": len(h) / max(1, (TB.epoch == ep).sum()),
                                 "per_animal": per})
                    print(f"  {r} {wn} {ct} {ep}: {per} -> {iv.point:+.2f} [{iv.lo:+.2f}, {iv.hi:+.2f}]", flush=True)
    CT = pd.DataFrame(crec)
    CT.to_csv(OUT / "contact.csv", index=False)

    # ---- specificity: per-session cross-validated shared / spout-specific / specific-timing energy
    pre_sess = {}
    for an, lab, ep, _day, V, pos in sess_tr:
        if ep == "pre":
            pre_sess.setdefault(an, []).append((lab, {r: spout_mean(V[r], pos) for r in regs}))
    srng = np.random.default_rng(1)
    erows = []
    for an, lab, ep, day, V, pos in sess_tr:
        for r in regs:
            P = np.nanmean([m[r] for lb, m in pre_sess[an] if lb != lab], 0)
            e = cv_energies(V[r], pos, P, tix, wix["early (0..0.3 s)"], wix["late (0.3..0.7 s)"], srng)
            erows.append({"animal": an, "label": lab, "epoch": ep, "day": day, "region": r, **dict(zip(ENER, e))})
    E = pd.DataFrame(erows)
    E.to_csv(OUT / "energies.csv", index=False)
    spec = []
    for r in regs:
        g = E[E.region == r]
        for en in ENER:
            for ep in EPS:
                pre_v = {a: h[en].dropna().tolist() for a, h in g[g.epoch == "pre"].groupby("animal")}
                post = {a: h[en].dropna().tolist() for a, h in g[g.epoch == ep].groupby("animal")}
                post = {a: v for a, v in post.items() if v and pre_v.get(a)}
                per = {a: round(float(np.mean(post[a]) - np.mean(pre_v[a])), 3) for a in post}
                iv = kit.boot_delta(post, pre_v, rng)
                spec.append({"region": r, "energy": en, "epoch": ep,
                             "pre_mean": float(np.mean([np.mean(v) for v in pre_v.values()])), "change": iv.point,
                             "lo": iv.lo, "hi": iv.hi, "k_up": sum(v > 0 for v in per.values()), "n": len(per),
                             "per_animal": per})
                print(f"  {r} {en} {ep}: per animal {per} -> {iv.point:+.3f} [{iv.lo:+.3f}, {iv.hi:+.3f}]", flush=True)
    SP = pd.DataFrame(spec)
    SP.to_csv(OUT / "specificity.csv", index=False)

    # ---- figure: rows = orofacial contra / ipsi; cols = 3 epochs of traces + early/late bars + specificity energies
    show = [r for r in regs if r.startswith("orofacial")]
    fig, axs = plt.subplots(len(show), 7, figsize=(29, 4.4 * len(show)), squeeze=False,
                            gridspec_kw={"width_ratios": [1, 1, 1, 1.25, 1.25, 1.1, 1.1]})
    for i, r in enumerate(show):
        for j, ep in enumerate(EPS):
            ax = axs[i, j]
            g = Dl[(Dl.region == r) & (Dl.epoch == ep)]
            for p in POS:
                m = g[g.spout == p].groupby("animal")[bcols].mean().mean(0).to_numpy()   # animal-weighted
                ax.plot(t, m, lw=1.5, label=pos_disp(p), **pos_style(p))
            sh = SP[(SP.region == r) & (SP.epoch == ep) & (SP.energy == "spout-specific")].iloc[0]
            ax.axhline(0, color="0.6", lw=0.8)
            ax.axvline(0, color="0.6", lw=0.8, ls=":")
            ax.set_title(f"{r}, {ep}\nspout-specific energy vs pre {sh.change:+.2f} [{sh.lo:+.2f}, {sh.hi:+.2f}]",
                         fontsize=9.5)
            ax.set_xlabel("time from first lick (s)")
            if j == 0:
                ax.set_ylabel(f"{r}\nchange vs own pre, same spout\n(minus RESTnl; pre pixel RMS)")
        axs[i, 0].legend(frameon=False, fontsize=7, loc="upper left")
        for k, wn in enumerate(WINS):
            ax = axs[i, 3 + k]
            for q, p in enumerate(POS):
                for e, ep in enumerate(EPS):
                    c = CH[(CH.region == r) & (CH.spout == p) & (CH.window == wn) & (CH.epoch == ep)].iloc[0]
                    x = q + (e - 1) * 0.24
                    ax.errorbar(x, c.change, yerr=[[c.change - c.lo], [c.hi - c.change]], fmt="o",
                                color=EPOCH_LINE[ep], ms=4, lw=1.2, capsize=0)
                    if c.lo > 0 or c.hi < 0:
                        sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], fontsize=7)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xticks(range(len(POS)), [pos_disp(p).replace(" ", "\n") for p in POS], fontsize=7.5)
            ax.set_title(f"{r}, {wn}", fontsize=9.5)
            ax.set_ylabel("change vs own pre")
        ax = axs[i, 5]
        for k, en in enumerate(ENER):
            for e, ep in enumerate(EPS):
                c = SP[(SP.region == r) & (SP.energy == en) & (SP.epoch == ep)].iloc[0]
                x = k + (e - 1) * 0.24
                ax.errorbar(x, c.change, yerr=[[c.change - c.lo], [c.hi - c.change]], fmt="o", color=EPOCH_LINE[ep],
                            ms=5, lw=1.4, capsize=0)
                if c.lo > 0 or c.hi < 0:
                    sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], fontsize=8)
                ax.annotate(f"{c.k_up}/{c.n}", (x, 0), xytext=(0, -12), textcoords="offset points", ha="center",
                            fontsize=6.5, color=EPOCH_LINE[ep], annotation_clip=False)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(len(ENER)), ["shared", "spout-\nspecific", "specific\ntiming"], fontsize=8)
        ax.set_title("cross-validated energy, post-lick\n(change vs own pre)", fontsize=9.5)
        ax.set_ylabel("change in energy\n(pre pixel RMS squared)")
        ax = axs[i, 6]
        lab = []
        for k, (wn, ct) in enumerate([(w, c) for w in WINS for c in ("contact", "no contact")]):
            for e, ep in enumerate(EPS):
                c = CT[(CT.region == r) & (CT.window == wn) & (CT.trials == ct) & (CT.epoch == ep)].iloc[0]
                x = k + (e - 1) * 0.24
                ax.errorbar(x, c.change, yerr=[[c.change - c.lo], [c.hi - c.change]], fmt="o" if ct == "contact" else "s",
                            color=EPOCH_LINE[ep], mfc=EPOCH_LINE[ep] if ct == "contact" else "white", ms=5, lw=1.4,
                            capsize=0)
                if c.lo > 0 or c.hi < 0:
                    sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], fontsize=8)
            lab.append(f"{wn.split(' ')[0]}\n{ct}")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(len(lab)), lab, fontsize=8)
        ax.set_title("all spouts: trials that touched the spout\n(filled) vs trials that did not (open)", fontsize=9.5)
        ax.set_ylabel("change vs own pre")
    fig.suptitle(C._wrap(
        "Is the post-lick orofacial increase spout-specific? Region means minus the session's RESTnl level (cohort SVD, "
        "first-lick-aligned, units of the animal's pre-template pixel RMS), per spout, as change vs the animal's own pre "
        "trace for the SAME spout. Left: animal-weighted change traces (solid = close, dashed = far; blue ipsi, gray mid, "
        "red contra); title = change vs own pre of the spout-specific energy. Middle: per-spout early / late change. Right: per-session "
        "split-half cross-validated energy over 0..0.7 s of the shared (spout-mean) change, its spout-specific part, and "
        "the spout-specific part of the late-minus-early contrast (pre sessions vs a leave-session-out pre trace), change "
        "vs own pre; k/4 = animals up. Far right: all spouts, trials whose 700-ms bout touched the spout vs trials with no contact (3-D "
        "licks; B27 trial table). Nested bootstrap animals -> sessions (2,000); * = 95 % CI excludes 0."), y=1.05, fontsize=9.5)
    sig_marks.reserve_headroom(fig)
    fig.tight_layout()
    save_png_svg(fig, FIGS / f"ss_shift_spout_timing{S.tag()}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(SP.drop(columns="per_animal").round(3).to_string())
    print(CH[CH.region.str.startswith("orofacial")].round(2).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
