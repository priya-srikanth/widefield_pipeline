"""Does the shared change track HOW THE ANIMAL LICKS? (Priya 11-09 "go ahead" on: "does the sustained orofacial
increase, including its presence before the lick, track how the animal licks (bout length, contact, jaw / tongue
kinematics), or is it there regardless of behavior?")

Trial level, cohort SVD basis, first-lick-aligned (WF_SS_ALIGN=lick), every trial minus its session's RESTnl level (the
reference of B26 / Figs 10.17-10.20). Neural readouts per trial = mean activity over a REGION's pixels (pixel-weighted
union of split-basis areas, ROI v8) in a WINDOW, units of the animal's pre-template pixel RMS (as the part maps):
  regions   orofacial = tjM1 + SSp-m + SSp-n; posterior-medial = RSPd + SSp-tr + SSp-ll; contra and ipsi separately
  windows   before the lick -1.0..-0.1 s, early 0..+0.3 s, late +0.3..+0.7 s
Behavior per trial, from the 3-D lick table (pose3d kinematics licks.csv.gz, joined through the landing table's
trial_id): licks in the bout = 3-D licks with a peak within 700 ms after the first lick's (incl. it), contacts among
them, median inter-lick interval, first-lick latency (rt), mean reach (mm) and peak speed of those licks.

TEST (frozen pre-stroke behavior -> neural map, the two-part-map idea of running list B14 at trial level): per animal
and readout, OLS on PRE trials: value ~ spout (fixed effects) + n_licks + n_contacts + rt + reach + speed (complete
trials only). Every trial's change vs pre = value - its spout's pre mean; the BEHAVIOR-PREDICTED change = prediction -
the spout's pre mean of the prediction (what the changed licking alone would produce under the pre-stroke relation);
RESIDUAL = change - predicted. Pre sessions are scored with a fit that leaves the session out (an honest null). Session
means -> change vs own pre, animal-weighted nested bootstrap (`boot_delta`, 2,000); * = 95 % CI excludes 0, k/4.
Caveats: a relation estimated within pre-stroke sessions may not hold after the stroke (it is a model of the pre
coupling, not a causal control); trials need a 3-D match (sessions without 3-D drop out); behavior the 3-D table does
not see (jaw-only movement, body) is not in the model.
Outputs: <state_space_recovery>/shift_behavior{tag}/{trials.csv.gz,sessions.csv,coefs.csv},
page_figs/ss_shift_behavior{tag}.png. EXPLORATORY.
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
import state_space_sim as S  # noqa: E402
import state_space_trajectories as TR  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402
from wfield_local.orofacial_cohort import EPOCH_LINE  # noqa: E402

EPS = ["acute", "subacute", "chronic"]
OUT = S.ROOT / f"shift_behavior{S.tag()}"
FIGS = S.ROOT / "page_figs"
REGIONS = {"orofacial": ["tjM1", "SSp-m", "SSp-n"], "posterior-medial": ["RSPd", "SSp-tr", "SSp-ll"]}
SIDES = {"contra": "right", "ipsi": "left"}
WINDOWS = {"before the lick": (-1.0, -0.1), "early (0..0.3 s)": (0.0, 0.3), "late (0.3..0.7 s)": (0.3, 0.7)}
BOUT_MS = 700
BEH = ["n_licks", "n_contacts", "rt_s", "reach_mm", "speed"]
BEH_LABEL = {"n_licks": "licks in the bout (700 ms)", "n_contacts": "contacts in the bout", "ili_ms":
             "inter-lick interval (ms)", "rt_s": "first-lick latency (s)", "reach_mm": "reach (mm)", "speed":
             "peak tongue speed"}


def behavior():
    """{(session label, trial_id): features} from the 3-D lick table."""
    K = pd.read_csv(S.LICKS3D, usecols=["session", "trial_id", "lick_idx", "contact", "t_peak_cam4", "reach_mm",
                                        "speed_peak"])
    K = K.dropna(subset=["t_peak_cam4"]).sort_values(["session", "trial_id", "lick_idx"])
    t0 = K[K.lick_idx == 0].set_index(["session", "trial_id"]).t_peak_cam4
    K["t0"] = K.set_index(["session", "trial_id"]).index.map(t0).to_numpy()
    B = K[(K.t_peak_cam4 >= K.t0) & (K.t_peak_cam4 <= K.t0 + BOUT_MS)]
    g = B.groupby(["session", "trial_id"])
    F = pd.DataFrame({"n_licks": g.size(), "n_contacts": g.contact.sum().astype(float),
                      "ili_ms": g.t_peak_cam4.apply(lambda x: np.median(np.diff(x)) if len(x) > 1 else np.nan),
                      "reach_mm": g.reach_mm.mean(), "speed": g.speed_peak.mean()}).reset_index()
    F["label"] = F.session.str.replace("_2026", "_", regex=False)
    return F.drop(columns="session")


def main():
    if S.BASIS != "svd":
        sys.exit("state_space_shift_behavior: needs the cohort SVD basis (WF_SS_BASIS=svd)")
    from wfield_local import area_basis, beta_maps
    Uc = np.load(C.BASIS_DIR / "U_cohort.npy").astype(np.float64)
    atlas = np.load(S.SVD.parents[1] / "atlas.npy")
    mask = np.load(C.BASIS_DIR / "mask.npy").reshape(atlas.shape)
    flat = np.flatnonzero(mask.ravel())
    split = area_basis.basis_masks(area_basis.roi_masks(atlas.ravel()[flat], beta_maps.stat_mask().ravel()[flat],
                                                        dict(S._atlas_names()), flat, mask.shape))["split"]
    W = {}
    for reg, areas in REGIONS.items():
        for side, hemi in SIDES.items():
            px = np.logical_or.reduce([np.asarray(split[f"{a}_{hemi}"], bool) for a in areas if f"{a}_{hemi}" in split])
            W[f"{reg} {side}"] = Uc[px].mean(0)                                   # K: pixel-weighted region mean
    t, _ = TR.times()
    wix = {w: np.flatnonzero((t >= a - 1e-9) & (t < b - 1e-9)) for w, (a, b) in WINDOWS.items()}
    L = pd.read_csv(R.LANDING)[["label", "row", "trial_id", "rt_ms"]]
    Lm = {lab: g for lab, g in L.groupby("label")}

    def hook(lab, Y, pos):
        tid, rt = np.full(len(pos), np.nan), np.full(len(pos), np.nan)
        g = Lm.get(lab)
        if g is not None:
            tid[g.row.to_numpy()] = g.trial_id.to_numpy(float)
            rt[g.row.to_numpy()] = g.rt_ms.to_numpy(float) / 1000
        return Y, pos, tid, rt
    data, _ = S.load_sessions(hook)
    F = behavior()
    rows = []
    for an, sess in sorted(data.items()):
        pre = [(lab, ep, day, Y, pos) for lab, ep, day, Y, pos, _t, _r in sess if ep == "pre"]
        Tpre = R.template(pre)
        ok = np.isfinite(Tpre).all(axis=(1, 2))
        B0 = Tpre[ok] - S.flat(Tpre[ok]).mean(0)
        unit = float(np.sqrt((B0 ** 2).sum() / B0.size * B0.shape[-1] / Uc.shape[0]))
        for lab, ep, day, Y, pos, tid, rt in sess:
            if ep not in ("pre", *EPS) or not (C.REST / f"{lab}.npz").exists():
                continue
            c = C._rest(lab, an)
            if c is None:
                continue
            Z = (Y - c) / unit                                                    # trials x bins x K
            d = pd.DataFrame({"animal": an, "label": lab, "epoch": ep, "day": day, "spout": pos, "trial_id": tid,
                              "rt_s": rt})
            for reg, w in W.items():
                v = Z @ w                                                         # trials x bins
                for wn, ix in wix.items():
                    d[f"{reg} | {wn}"] = v[:, ix].mean(1)
            rows.append(d)
        print(f"  {an} done", flush=True)
    T = pd.concat(rows, ignore_index=True).merge(F, on=["label", "trial_id"], how="left")
    reads = [f"{r} | {w}" for r in W for w in WINDOWS]
    print("trials:", len(T), "| with 3-D behavior:", int(T[BEH].notna().all(1).sum()))
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- frozen pre-stroke behavior -> neural fit, per animal x readout
    coefs = []
    for an, g in T.groupby("animal"):
        okb = g[BEH].notna().all(1)
        for col in reads:
            gg = g[okb & g[col].notna()]
            X = pd.get_dummies(gg.spout).astype(float)
            Xb = (gg[BEH] - gg[gg.epoch == "pre"][BEH].mean()) / gg[gg.epoch == "pre"][BEH].std()
            X = pd.concat([X, Xb], axis=1).to_numpy()
            y = gg[col].to_numpy()
            pre = (gg.epoch == "pre").to_numpy()
            labs = gg.label.to_numpy()
            pred = np.full(len(gg), np.nan)
            beta, *_ = np.linalg.lstsq(X[pre], y[pre], rcond=None)
            pred[~pre] = X[~pre] @ beta
            for lab in np.unique(labs[pre]):                                      # pre sessions: leave the session out
                m = pre & (labs != lab)
                b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
                pred[labs == lab] = X[labs == lab] @ b
            coefs.append({"animal": an, "readout": col, **{f"b_{k}": v for k, v in zip(BEH, beta[-len(BEH):])}})
            # change vs the spout's pre mean, for the observation and for the prediction
            sp = gg.spout.to_numpy()
            mo = {s: y[pre & (sp == s)].mean() for s in np.unique(sp[pre])}
            mp = {s: pred[pre & (sp == s)].mean() for s in np.unique(sp[pre])}
            base_o = np.array([mo.get(s, np.nan) for s in sp])
            base_p = np.array([mp.get(s, np.nan) for s in sp])
            T.loc[gg.index, f"obs | {col}"] = y - base_o
            T.loc[gg.index, f"pred | {col}"] = pred - base_p
            T.loc[gg.index, f"res | {col}"] = (y - base_o) - (pred - base_p)
    pd.DataFrame(coefs).to_csv(OUT / "coefs.csv", index=False)
    T.to_csv(OUT / "trials.csv.gz", index=False)
    cols = [f"{k} | {c}" for k in ("obs", "pred", "res") for c in reads] + BEH + ["ili_ms"]
    Sx = T.groupby(["animal", "label", "epoch", "day"])[cols].mean().reset_index()
    Sx.to_csv(OUT / "sessions.csv", index=False)

    rng = np.random.default_rng(0)

    def change(col, ep):
        pre = {a: g[g.epoch == "pre"][col].dropna().tolist() for a, g in Sx.groupby("animal")}
        post = {a: g[g.epoch == ep][col].dropna().tolist() for a, g in Sx.groupby("animal")}
        post = {a: v for a, v in post.items() if v and pre.get(a)}
        iv = kit.boot_delta(post, pre, rng) if post else None
        up = sum(np.mean(post[a]) > np.mean(pre[a]) for a in post)
        return (iv.point, iv.lo, iv.hi, up, len(post)) if iv is not None else (np.nan,) * 3 + (0, 0)

    # ---- figure: rows = region x side, cols = windows; per epoch: observed / behavior-predicted / residual
    kinds = [("obs", "observed change", "o"), ("pred", "predicted from licking (pre-stroke fit)", "s"),
             ("res", "residual beyond licking", "D")]
    fig, axs = plt.subplots(len(W) + 1, len(WINDOWS), figsize=(14, 3.0 * (len(W) + 1)), squeeze=False)
    recs = []
    for i, reg in enumerate(W):
        for j, wn in enumerate(WINDOWS):
            ax = axs[i, j]
            col = f"{reg} | {wn}"
            for k, ep in enumerate(EPS):
                for m, (kk, _lab, mk) in enumerate(kinds):
                    pt, lo, hi, up, n = change(f"{kk} | {col}", ep)
                    x = k + (m - 1) * 0.25
                    ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt=mk, color=EPOCH_LINE[ep], ms=5, lw=1.4,
                                capsize=0, mfc="white" if kk == "pred" else EPOCH_LINE[ep])
                    if lo > 0 or hi < 0:
                        sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], fontsize=9)
                    recs.append({"readout": col, "kind": kk, "epoch": ep, "change": pt, "lo": lo, "hi": hi,
                                 "k_up": up, "n": n})
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xticks(range(3), EPS)
            ax.set_title(f"{reg}, {wn}", fontsize=10)
            if j == 0:
                ax.set_ylabel("change vs own pre\n(pre pixel RMS)")
    hs = [plt.Line2D([], [], marker=mk, ls="", color="0.3", mfc="white" if kk == "pred" else "0.3", label=lab)
          for kk, lab, mk in kinds]
    axs[0, 0].legend(handles=hs, frameon=False, fontsize=7.5, loc="upper left")
    for j, b in enumerate(["n_licks", "n_contacts", "rt_s"]):
        ax = axs[-1, j]
        for k, ep in enumerate(EPS):
            pt, lo, hi, up, n = change(b, ep)
            ax.errorbar(k, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", color=EPOCH_LINE[ep], ms=5, lw=1.4, capsize=0)
            if lo > 0 or hi < 0:
                sig_marks.star_top(ax, k, "*", color=EPOCH_LINE[ep], fontsize=9)
            ax.annotate(f"{up}/{n}", (k, 0), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7,
                        color=EPOCH_LINE[ep], annotation_clip=False)
            recs.append({"readout": b, "kind": "behavior", "epoch": ep, "change": pt, "lo": lo, "hi": hi, "k_up": up,
                         "n": n})
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(3), EPS)
        ax.set_title(f"behavior: {BEH_LABEL[b]}", fontsize=10)
        ax.set_ylabel("change vs own pre")
    pd.DataFrame(recs).to_csv(OUT / "changes.csv", index=False)
    fig.suptitle(C._wrap(
        "Does the shared increase track how the animal licks? Region-mean activity (minus the session's RESTnl level, "
        "units of the animal's pre-template pixel RMS; cohort SVD basis, first-lick-aligned, working trials with a 3-D "
        "match) as change vs own pre: OBSERVED (filled), PREDICTED from the trial's licking by each animal's pre-stroke "
        "fit (open; spout + licks and contacts in the 700-ms bout + latency + reach + speed), RESIDUAL beyond it "
        "(diamond). Pre sessions scored leave-session-out. Nested bootstrap animals -> sessions (2,000); * = 95 % CI "
        "excludes 0. Bottom: the behavior itself."), y=1.03, fontsize=9.5)
    sig_marks.reserve_headroom(fig)
    fig.tight_layout()
    save_png_svg(fig, FIGS / f"ss_shift_behavior{S.tag()}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(pd.DataFrame(recs).round(3).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
