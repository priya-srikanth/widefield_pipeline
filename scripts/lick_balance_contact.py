"""Is the PRIMARY hemispheric balance sensory reafference? Split cued first licks by spout CONTACT vs MISS (10-29).

Hypothesis: cued licks touch the spout, ITI licks do not, and SSp-n / SSs are somatosensory, so the balance might be
partly touch reafference. Test: the primary balance (lick_recovery_primary: angle-matched gain ratio, ratio_010_050 =
0.1..0.5 s after the lick minus -1..-0.5 s, summed over ALM, tjM1, SSp-m, SSp-n, SSs; log2(contra obs / pred) -
log2(ipsi obs / pred); pred = the animal's PRE straight line on lick angle, pre sessions leave-one-session-out) computed
separately for first post-cue licks that CONTACTED the spout and those that MISSED.

CONTACT DEFINITION: licks_all.csv `contact` (tongue_kinematics, configs `contact: match: span`): a DAQ lick-sensor
(electrical spout contact) onset inside the cam4 lick's own rise start .. fall end +- 8 ms. The DAQ sensor is the
contact measure itself; the pose only says which lick it belongs to. Cross-check: the 3-D tip-to-spout miss distance
(kinematics_all miss_mm) is ~1.0 mm on contact vs ~2.1 mm on miss licks.
CAVEAT built in: a missed first lick is usually followed ~0.15 s later by a lick that DOES touch the spout, inside the
0.1..0.5 s window; `t_next_contact` = time from the first lick to the next contact lick, and class "miss_strict" = misses
with no contact lick within 0.5 s.

References (both stated): line_all = pre line fit on ALL pre first licks (the primary); line_contact = fit on pre
CONTACT first licks only; gauss_* = the like-for-like reference (lick_balance_subsets._residuals_gauss: Gaussian-weighted
mean of same-animal pre licks at the same lateral landing place, sigma 0.1 mm, NaN if summed weight < 2).
Control: contact licks DOWNSAMPLED per session to the session's miss (and strict-miss) count (N_DRAW draws), so the
contact-vs-miss recovery link is compared at equal lick numbers.
Outputs <OUT_W>/balcontact_licks.csv.gz, balcontact_sessions.csv, balcontact_animal_epoch.csv, balcontact_animal_r.csv,
balcontact_models.csv, balcontact_cv.csv, balcontact_downsample.csv; figure figures/lick_balance_contact.png.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_balance_subsets as lbs  # noqa: E402
import lick_recovery_markers as lrm  # noqa: E402

from wfield_local import mask_keys  # noqa: E402

POOL = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]
WIN = (0.1, 0.5)
MIN_CELL = 20                      # pre licks needed for a line (lick_balance_regions.MIN_CELL)
MIN_LICKS = 5                      # licks needed for a session marker
STRICT_S = 0.5
N_DRAW = 50
REFS = ["line_all", "line_contact", "gauss_all", "gauss_contact"]
CLASSES = ["all", "contact", "miss", "miss_strict"]
AN = lrm.AN


def _session(label: str) -> pd.DataFrame:
    """Per first post-cue lick: contact flag, time to next contact lick, angle covariates, pooled contra / ipsi values
    per area (ratio_010_050)."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    f = il.OUT_W / "firstlick_full" / f"{label}.npz"
    if not f.exists():
        return pd.DataFrame()
    z = np.load(f, allow_pickle=True)
    mask_keys.check_obj(z, f)                     # stale-read guard (11-09)
    X = z["X"].astype(float)
    cx = list(z["cols"])
    t = z["t"].astype(float)
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv",
                    usecols=["context", "daq_peak_s", "contact", "tip_to_spout_px"])
    A = A.dropna(subset=["daq_peak_s"]).drop_duplicates("daq_peak_s").set_index("daq_peak_s")
    if not np.isin(t, A.index.to_numpy()).all():
        raise RuntimeError(f"{label}: {int((~np.isin(t, A.index.to_numpy())).sum())} first licks not in licks_all")
    con = A.contact.reindex(t).astype(bool).to_numpy()
    ct = np.sort(A.index.to_numpy()[A.contact.astype(bool).to_numpy()])
    j = np.searchsorted(ct, t, side="right")                           # next contact lick strictly after
    t_next = np.where(j < len(ct), ct[np.minimum(j, len(ct) - 1)] - t, np.nan)
    bm, wm = lbs._wm(*il.BASE_RESP), lbs._wm(*WIN)
    R = {"label": label, "animal": an, "epoch": str(z["epoch"]), "t": t, "pos": z["pos"], "side_k": z["side_ml"],
         "reach_mm": z["reach_mm"], "contact": con, "t_next_contact": t_next,
         "tip_to_spout_px": A.tip_to_spout_px.reindex(t).to_numpy(float)}
    for a in POOL:
        for s, h in (("c", "right"), ("i", "left")):
            tr = X[:, :, cx.index(f"{a}_{h}")]
            R[f"{a}|{s}"] = tr[:, wm].mean(1) - tr[:, bm].mean(1)
    return pd.DataFrame(R)


def _predict(D: pd.DataFrame, vs: list[str], method: str, fit: np.ndarray) -> np.ndarray:
    """Pre-reference prediction for every lick; fit on pre licks with `fit` True, same animal; a pre session's own
    licks are left out of its fit (leave-one-pre-session-out)."""
    P = np.full((len(D), len(vs)), np.nan)
    for _an, g in D.groupby("animal"):
        gi = g.index.to_numpy()
        pre = g[(g.epoch == "pre").to_numpy() & fit[gi]]
        for lab in g.label.unique():
            tr = pre[pre.label != lab]
            rows = gi[(g.label == lab).to_numpy()]
            if method == "line":
                if len(tr) < MIN_CELL or tr.angle.std() == 0:
                    continue
                for k, v in enumerate(vs):
                    b1, b0 = np.polyfit(tr.angle, tr[v], 1)
                    P[rows, k] = b0 + b1 * D.angle.to_numpy()[rows]
            else:
                trd = tr.dropna(subset=["side_k"])
                if not len(trd):
                    continue
                x = D.side_k.to_numpy(float)[rows]
                W = np.exp(-0.5 * ((x[:, None] - trd.side_k.to_numpy(float)[None]) / lbs.GAUSS_SIGMA) ** 2)
                den = W.sum(1)
                ref = (W @ trd[vs].to_numpy(float)) / np.maximum(den, 1e-12)[:, None]
                ref[den < lbs.GAUSS_MINW] = np.nan
                P[rows] = ref
    return P


def licks() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    labs = sorted(p.stem for p in (il.OUT_W / "firstlick_full").glob("*.npz"))
    res, fail = ak.fan_sessions(labs, _session, jobs=6)
    if fail:
        raise RuntimeError(f"failed: {fail[:5]}")
    D = pd.concat([v for _k, v in res if len(v)], ignore_index=True)
    D["angle"] = np.degrees(np.arcsin(np.clip(D.side_k / D.reach_mm, -1, 1)))
    D = D.dropna(subset=["angle"]).reset_index(drop=True)
    vs = [f"{a}|{s}" for a in POOL for s in ("c", "i")]
    con = D.contact.to_numpy(bool)
    for ref in REFS:
        meth, fs = ref.split("_")
        P = _predict(D, vs, meth, np.ones(len(D), bool) if fs == "all" else con)
        ok = np.isfinite(P).all(1)
        for s in ("c", "i"):
            ks = [k for k, v in enumerate(vs) if v.endswith(f"|{s}")]
            D[f"{ref}|obs_{s}"] = np.where(ok, D[[vs[k] for k in ks]].sum(1), np.nan)
            D[f"{ref}|pred_{s}"] = np.where(ok, P[:, ks].sum(1), np.nan)
    D["miss_strict"] = ~con & ~(D.t_next_contact <= STRICT_S)
    return D


def _marker(g: pd.DataFrame, ref: str) -> float:
    o = g[[f"{ref}|obs_c", f"{ref}|pred_c", f"{ref}|obs_i", f"{ref}|pred_i"]].dropna()
    if len(o) < MIN_LICKS:
        return np.nan
    oc_, pc_, oi_, pi_ = o.sum().to_numpy()
    if min(oc_, pc_, oi_, pi_) <= 0:
        return np.nan
    return float(np.log2(oc_ / pc_) - np.log2(oi_ / pi_))


def _cls(g: pd.DataFrame, c: str) -> pd.DataFrame:
    return g if c == "all" else g[g.contact] if c == "contact" else g[~g.contact] if c == "miss" else g[g.miss_strict]


def sessions(D: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lab, an, ep), g in D.groupby(["label", "animal", "epoch"]):
        rec = {"label": lab, "animal": an, "epoch": ep, "n_all": len(g), "n_contact": int(g.contact.sum()),
               "n_miss": int((~g.contact).sum()), "n_miss_strict": int(g.miss_strict.sum()),
               "miss_t_next_contact_med": float(g[~g.contact].t_next_contact.median())}
        for ref in REFS:
            for c in CLASSES:
                rec[f"{ref}|{c}"] = _marker(_cls(g, c), ref)
        rows.append(rec)
    S = pd.DataFrame(rows)
    R = pd.read_csv(il.OUT_W / "recmark_sessions.csv", usecols=["label", "day", "aim"]).drop_duplicates("label")
    return S.merge(R, on="label", how="left")


def downsample(D: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """Contact licks drawn per post session to that session's miss count; per-animal r with aim, per draw."""
    rng = np.random.default_rng(0)
    post = S[(S.epoch != "pre") & (S.day > 0)].dropna(subset=["aim"])
    out = []
    for ref, ncol in (("line_all", "n_miss"), ("line_contact", "n_miss"), ("line_all", "n_miss_strict"),
                      ("line_contact", "n_miss_strict")):
        for d in range(N_DRAW):
            vals = []
            for _, r in post.iterrows():
                g = D[(D.label == r.label) & D.contact]
                n = int(r[ncol])
                vals.append(_marker(g.iloc[rng.choice(len(g), n, replace=False)], ref) if 0 < n <= len(g) else np.nan)
            t = post.assign(m=vals).dropna(subset=["m"])
            for a, h in t.groupby("animal"):
                out.append({"ref": ref, "to": ncol, "draw": d, "animal": a, "n": len(h), "r": np.corrcoef(h.m, h.aim)[0, 1]})
    return pd.DataFrame(out)


def main() -> int:
    from wfield_local import writeguard
    writeguard.assert_writable(il.OUT_W)
    D = licks()
    D.to_csv(il.OUT_W / "balcontact_licks.csv.gz", index=False)
    S = sessions(D)
    S.to_csv(il.OUT_W / "balcontact_sessions.csv", index=False)
    pd.set_option("display.width", 250)
    # sanity: 'all' x line_all must reproduce the primary (recprim_sessions_full, cued first lick)
    try:
        Q = pd.read_csv(il.OUT_W / "recprim_sessions_full.csv")
        Q = Q[Q.set == lrm.C1SET][["label", "marker"]].merge(S[["label", "line_all|all"]], on="label")
        print(f"== check vs recprim_sessions_full: n {len(Q)}, r {np.corrcoef(Q.marker, Q['line_all|all'])[0, 1]:.4f}, "
              f"max |diff| {np.nanmax(np.abs(Q.marker - Q['line_all|all'])):.4g}", flush=True)
    except Exception as ex:                                                # noqa: BLE001
        print(f"!! check vs primary failed: {ex}")
    # counts (per animal x epoch)
    C = D.groupby(["animal", "epoch"]).agg(n=("contact", "size"), n_contact=("contact", "sum"),
                                           n_miss_strict=("miss_strict", "sum"), sessions=("label", "nunique"))
    C["n_miss"] = C.n - C.n_contact
    C["frac_miss"] = C.n_miss / C.n
    C["miss_t_next_contact_med_s"] = D[~D.contact].groupby(["animal", "epoch"]).t_next_contact.median()
    C["miss_mm_proxy_tip_px_contact"] = D[D.contact].groupby(["animal", "epoch"]).tip_to_spout_px.median()
    C["tip_px_miss"] = D[~D.contact].groupby(["animal", "epoch"]).tip_to_spout_px.median()
    print("== first post-cue licks: contact vs miss counts\n" + C.round(3).to_string())
    # per animal x epoch balance (session mean, rule 8)
    E = []
    for ref in REFS:
        for c in CLASSES:
            g = S.groupby(["animal", "epoch"])[f"{ref}|{c}"].agg(["mean", "sem", "count"]).reset_index()
            E.append(g.assign(ref=ref, cls=c))
    E = pd.concat(E, ignore_index=True)
    E = E.merge(C.reset_index()[["animal", "epoch", "n", "n_contact", "n_miss", "n_miss_strict"]], on=["animal", "epoch"])
    E.to_csv(il.OUT_W / "balcontact_animal_epoch.csv", index=False)
    for ref in REFS:
        print(f"== balance per animal x epoch, {ref} (session mean)\n"
              + E[E.ref == ref].pivot_table(index=["animal", "epoch"], columns="cls", values="mean", sort=False)
              [CLASSES].round(3).to_string())
    # recovery link: per-animal r with the far-contra aim, then models
    post = S[(S.epoch != "pre") & (S.day > 0)].dropna(subset=["aim"])
    AR = []
    for ref in REFS:
        for c in CLASSES:
            v = f"{ref}|{c}"
            for a, h in post.dropna(subset=[v]).groupby("animal"):
                AR.append({"ref": ref, "cls": c, "animal": a, "n": len(h), "r": np.corrcoef(h[v], h.aim)[0, 1]})
    AR = pd.DataFrame(AR)
    DS = downsample(D, S)
    DS.to_csv(il.OUT_W / "balcontact_downsample.csv", index=False)
    dsm = DS.groupby(["ref", "to", "animal"]).agg(n=("n", "first"), r=("r", "mean"),
                                                  r_lo=("r", lambda x: np.quantile(x, 0.05)),
                                                  r_hi=("r", lambda x: np.quantile(x, 0.95))).reset_index()
    dsm["cls"] = np.where(dsm.to == "n_miss", "contact_down", "contact_down_strict")
    AR = pd.concat([AR, dsm.drop(columns="to")], ignore_index=True)
    # noise floor: SD of the marker over PRE sessions (left-out reference) per class, and median licks per session
    NF = S[S.epoch == "pre"].melt(id_vars=["animal"], value_vars=[f"{r}|{c}" for r in REFS for c in CLASSES])
    print("== pre-session SD of the marker (noise floor) per animal\n"
          + NF.pivot_table(index="variable", columns="animal", values="value", aggfunc="std", sort=False).round(3)
          .to_string())
    print("== median first licks per POST session\n" + S[S.epoch != "pre"][["n_contact", "n_miss", "n_miss_strict"]]
          .groupby(S.animal).median().to_string())
    AR.to_csv(il.OUT_W / "balcontact_animal_r.csv", index=False)
    print("== per-animal r (post sessions) with far-contra first-lick aim\n"
          + AR.pivot_table(index=["ref", "cls"], columns="animal", values="r", sort=False).round(2).to_string())
    M = S[["label", "animal", "epoch", "day", "aim"] + [f"{r}|{c}" for r in REFS for c in CLASSES]].assign(set=lrm.C1SET)
    M.columns = [c.replace("|", "__") for c in M.columns]
    F = lrm.models(M)
    F["M"].to_csv(il.OUT_W / "balcontact_models.csv", index=False)
    F["CV"].to_csv(il.OUT_W / "balcontact_cv.csv", index=False)
    print("== mixed model aim ~ within + between + log(day), (1|animal)\n" + F["M"].round(4).to_string(index=False))
    print("== cross-validated\n" + F["CV"][["pred", "n", "loao", "loao_days_same_sessions", "p_shuffle", "loblk",
                                            "loblk_days_same_sessions"]].round(3).to_string(index=False))
    plot(D, S, E, AR, F["M"])
    return 0


def plot(D, S, E, AR, M) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    eps = ["pre", "acute", "subacute", "chronic"]
    ccol = {"all": "0.5", "contact": "#4C72B0", "miss": "#C44E52", "miss_strict": "#DD8452", "contact_down": "#8FAADC"}
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))
    for j, ref in enumerate(("line_all", "line_contact")):
        ax = axs[0, j]
        for k, c in enumerate(("contact", "miss", "miss_strict")):
            e = E[(E.ref == ref) & (E.cls == c)]
            for ia, a in enumerate(AN):
                h = e[e.animal == a].set_index("epoch").reindex(eps)
                x = np.arange(len(eps)) * 5 + ia + (k - 1) * 0.25
                ax.errorbar(x, h["mean"], h["sem"], fmt=ANIMAL_MARKER[a], color=ccol[c], ms=5, lw=0.8,
                            label=c if ia == 0 else None)
        ax.set_xticks(np.arange(len(eps)) * 5 + 1.5, eps)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("balance gain ratio (log2, contra - ipsi vs pre line)", fontsize=8)
        ax.set_title(f"per animal x epoch (markers = PS92/93/94/95), pre line fit on "
                     f"{'ALL' if ref.endswith('all') else 'CONTACT'} pre first licks", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
    ax = axs[0, 2]
    m = D[~D.contact].t_next_contact.clip(upper=1.5)
    ax.hist(m.dropna(), bins=np.arange(0, 1.55, 0.025), color=ccol["miss"])
    ax.axvspan(*WIN, color="0.85", zorder=0)
    ax.set_xlabel("missed first lick: time to the next spout-contact lick (s; gray = balance window)", fontsize=8)
    ax.set_title(f"misses: median {m.median():.2f} s to next contact; {np.mean(m <= STRICT_S):.0%} within {STRICT_S} s",
                 fontsize=9)
    ax = axs[1, 0]
    cl = ["all", "contact", "contact_down", "miss", "contact_down_strict", "miss_strict"]
    ccol["contact_down_strict"] = "#C9D6EE"
    for jr, ref in enumerate(("line_all", "line_contact")):
        for k, c in enumerate(cl):
            r = AR[(AR.ref == ref) & (AR.cls == c)].set_index("animal").reindex(AN)
            x = np.arange(len(AN)) + (k - 2.5) * 0.13 + jr * 5
            ax.bar(x, r.r, width=0.12, color=ccol[c], label=c if jr == 0 else None)
    ax.set_xticks(list(range(4)) + list(range(5, 9)), AN + AN, fontsize=7.5)
    ax.text(1.5, 1.02, "pre line: all pre", ha="center", fontsize=8)
    ax.text(6.5, 1.02, "pre line: contact pre", ha="center", fontsize=8)
    ax.set_ylim(-1, 1.1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("within-animal r with far-contra aim (post sessions)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower left")
    ax.set_title("recovery link per animal (contact_down[_strict] = contact licks drawn to the miss [strict-miss] "
                 "count, mean r of draws)", fontsize=8.5)
    post = S[(S.epoch != "pre") & (S.day > 0)]
    for j, c in enumerate(("contact", "miss")):
        ax = axs[1, 1 + j]
        v = f"line_all|{c}"
        ax.plot(S[S.epoch == "pre"][v], S[S.epoch == "pre"].aim, "o", color="0.82", ms=5)
        for (ep, a), h in post.groupby(["epoch", "animal"]):
            ax.plot(h[v], h.aim, ANIMAL_MARKER[a], color=EPOCH_LINE[ep], ms=6, mec="k", mew=0.3)
        mm = M[M.pred == f"line_all__{c}"]
        t = (f"slope {mm.slope.iloc[0]:+.1f}, p {mm.p.iloc[0]:.2g}" if len(mm) and "p" in mm else "")
        r = AR[(AR.ref == "line_all") & (AR.cls == c)].set_index("animal").r.reindex(AN)
        ax.set_title(f"{c.upper()} first licks (pre line: all pre): mixed model {t}\nper-animal r "
                     + " / ".join(f"{x:+.2f}" for x in r), fontsize=9)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("balance gain ratio (log2)", fontsize=8)
        ax.set_ylabel("far-contra first-lick aim (deg vs pre)", fontsize=8)
    for a_ in axs.flat:
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.suptitle("Is the PRIMARY balance touch reafference? Cued first post-cue licks split by DAQ spout CONTACT vs MISS "
                 "(balance: ALM+tjM1+SSp-m+SSp-n+SSs, 0.1-0.5 s, angle-matched gain ratio). PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png_svg(fig, il.OUT_W / "figures" / "lick_balance_contact.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
