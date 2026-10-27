"""Figures for three checks that were table-only (Priya 10-27: "figure?"):
  rmse   the where-decoder's error split into |bias| and scatter (SD) per session (`iti_where_rmse_split.py`)
  f0     baseline fluorescence F0 (470, 415 nm) and the post-lick response in dF/F vs dF units (`iti_f0_check.py`)
  3axis  the where-decoder on lateral / forward / down axes: r / pre and offset beyond the pre pull (`iti_where_3axis.py`)
Every panel: thin lines = animals (session means per epoch), black = mean of animals, diamonds = mean of animals' pre +
change vs pre with the paired nested bootstrap 95% CI (animals > sessions, `analysis_kit.boot_delta`; * = CI excludes
0), and the count of animals changing in the point's direction. Reads the session CSVs; writes
<E>/iti_licks/figures/iti_check_{rmse,f0,3axis}.png. Usage: python scripts/iti_check_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]


def epoch_panel(ax, df: pd.DataFrame, val: str, ref: float | None = None, title: str = "", ylab: str = "") -> None:
    import matplotlib.pyplot as plt
    cols = plt.get_cmap("tab10")
    am = df.groupby(["animal", "epoch"])[val].mean()
    sess = df.groupby(["label", "epoch"])[val].mean().reset_index()      # one grey dot per SESSION
    jit = np.random.default_rng(1)
    for x, e in enumerate(EPS):
        v = sess[sess.epoch == e][val].to_numpy(float)
        ax.scatter(x - 0.18 + jit.uniform(-0.06, 0.06, len(v)), v, s=7, color="0.55", alpha=0.6, zorder=2, lw=0)
    for k, a in enumerate(AN):
        ax.plot(range(4), [am.get((a, e), np.nan) for e in EPS], "o-", color=cols(k), ms=4, lw=1, alpha=0.7, label=a)
    mu = [am.xs(e, level="epoch").mean() if e in am.index.get_level_values("epoch") else np.nan for e in EPS]
    ax.plot(range(4), mu, "-", color="k", lw=2.2, zorder=5, label="mean of animals")
    prd = {a: h[val].dropna().to_numpy() for a, h in df[df.epoch == "pre"].groupby("animal")}
    pv = sess[sess.epoch == "pre"][val].dropna().to_numpy(float)       # pre-stroke spread: SEM over pre SESSIONS
    if len(pv) > 1:
        sem = pv.std(ddof=1) / np.sqrt(len(pv))
        ax.errorbar(0.13, mu[0], yerr=sem, fmt="D", color="k", ms=5, capsize=3, zorder=6)
        ax.text(0.13, mu[0] + sem, f"pre +/-SEM\n({len(pv)} sessions)", ha="center", va="bottom", fontsize=6.5)
    for x, e in enumerate(EPS[1:], start=1):
        post = {a: h[val].dropna().to_numpy() for a, h in df[df.epoch == e].groupby("animal")}
        post = {a: v for a, v in post.items() if len(v) and a in prd}
        if not post:
            continue
        ci = ak.boot_delta(post, prd, np.random.default_rng(0))
        d = (am.xs(e, level="epoch") - am.xs("pre", level="epoch")).dropna()
        n_dir = int((d * np.sign(ci.point) > 0).sum())
        ax.errorbar(x + 0.13, mu[0] + ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="D", color="k",
                    ms=5, capsize=3, zorder=6)
        sig = ci.lo > 0 or ci.hi < 0
        ax.text(x + 0.13, mu[0] + ci.hi, f"{ci.point:+.2f}{' *' if sig else ''}\n{n_dir}/{len(d)}", ha="center",
                va="bottom", fontsize=7)
    if ref is not None:
        ax.axhline(ref, color="0.7", lw=0.7, ls=":")
    ax.set_xticks(range(4), EPS, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylab, fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def _fit(fig) -> None:
    """tight_layout leaving room for the suptitle: ~0.2 in per title line + 0.3 in margin (no title / panel overlap)."""
    t = fig._suptitle.get_text() if fig._suptitle is not None else ""
    n = t.count(chr(10)) + 1 if t else 0
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.17 * n + 0.12) / fig.get_size_inches()[1]))


def _save(fig, name: str, top: float = 0.0) -> None:
    import matplotlib.pyplot as plt
    _fit(fig)
    fig.savefig(il.OUT / "figures" / f"iti_check_{name}.png", dpi=100)
    plt.close(fig)


def rmse() -> None:
    import matplotlib.pyplot as plt
    S = pd.read_csv(il.OUT / "where_rmse_split_sessions.csv")
    rows = [(st, ax_) for st in ("iti", "trial") for ax_ in ("lat", "sag")]
    fig, axs = plt.subplots(len(rows), 3, figsize=(13, 3.3 * len(rows)), squeeze=False)
    for i, (st, ax_) in enumerate(rows):
        g = S[(S.set == st) & (S.axis == ax_)]
        for j, (val, lab) in enumerate((("rmse", "RMSE"), ("bias", "|bias| = |mean error|"),
                                        ("sd", "scatter = SD of error"))):
            epoch_panel(axs[i, j], g, val, title=f"{'ITI' if st == 'iti' else 'cued'} "
                        f"{'LATERAL' if ax_ == 'lat' else 'SAGITTAL'}: {lab}", ylab="z-scored decoder, mm")
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("NEURAL: why does the frozen where-decoder's error grow after the stroke? Per session, error = decoded - "
                 "actual; RMSE^2 = bias^2 + SD^2.\nDecoder on z-scored Allen-area features (lags 0..0.2 s; KEY-row "
                 "decoder). A growing |bias| with flat SD = predictions SHIFTED, not more scattered. PROVISIONAL",
                 fontsize=9.5)
    _save(fig, "rmse", 0.94)


def f0() -> None:
    import matplotlib.pyplot as plt
    S = pd.read_csv(il.OUT / "f0_check.csv")
    panels = (("f0_470_rel", "(1) F0 470 nm, raw fluorescence (/ animal's pre mean)"),
              ("f0_415_rel", "(2) F0 415 nm (/ pre)"),
              ("resp_auc_rel", "(3) post-lick response in dF/F (/ pre)\n= what the page's response figures show"),
              ("resp_dF_rel", "(4) post-lick response in dF, raw units (/ pre)\n= (3) x (1), session by session"))
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), squeeze=False)
    for j, (val, lab) in enumerate(panels):
        epoch_panel(axs[0, j], S, val, ref=1.0, title=lab, ylab="/ pre" if j == 0 else "")
    lo = min(axs[0, 2].get_ylim()[0], axs[0, 3].get_ylim()[0])
    hi = max(axs[0, 2].get_ylim()[1], axs[0, 3].get_ylim()[1])
    for j in (2, 3):
        axs[0, j].set_ylim(lo, hi)                        # (3) and (4) on ONE scale: the gap between them is F0
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("NEURAL / MEASUREMENT: is the larger post-lick response an F0 artefact (dF/F inflated by a fall in "
                 "baseline fluorescence)? F0 = mean of frames_average.npy over the central 60% of the image;\nresponse "
                 "= the ITI-lick response area (orofacial ROIs, the ITI-lick PETH figures). If F0 fell, dF/F would rise with no change in dF: the dF "
                 "panel answers it. PROVISIONAL", fontsize=9.5)
    _save(fig, "f0", 0.86)


def three_axis() -> None:
    import matplotlib.pyplot as plt
    S = pd.read_csv(il.OUT / "where_3axis_sessions.csv")
    axes = (("lat", "LATERAL (+ = contralesional)"), ("fwd", "FORWARD (+ = further forward)"),
            ("down", "DOWN (+ = further down)"))
    fig, axs = plt.subplots(4, 3, figsize=(14, 13.5), squeeze=False)
    for i, (st, val, lab) in enumerate((("iti", "r_norm", "r / animal's pre"), ("iti", "beyond_d", "offset beyond pre "
                                        "pull, mm (change vs pre)"), ("trial", "r_norm", "r / animal's pre"),
                                        ("trial", "beyond_d", "offset beyond pre pull, mm (change vs pre)"))):
        for j, (ax_, an_) in enumerate(axes):
            g = S[(S.set == st) & (S.axis == ax_)]
            epoch_panel(axs[i, j], g, val, ref=1.0 if val == "r_norm" else 0.0,
                        title=f"{'ITI' if st == 'iti' else 'cued'} licks, {an_}", ylab=lab if j == 0 else "")
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("NEURAL: the frozen where-decoder on all THREE axes (z-scored Allen-area features, lags 0..0.2 s; one "
                 "ridge per axis, trained on August pre licks, pre leave-one-session-out).\nSagittal extension ~ |down| "
                 "(r 0.96); forward is nearly independent of it (r -0.05). Offset = decoded - (a + b x actual), a, b fit "
                 "on pre licks. PROVISIONAL", fontsize=9.5)
    _save(fig, "3axis", 0.95)


def _days() -> pd.DataFrame:
    from recovery_timing import series
    return series("lick")[["label", "day"]].drop_duplicates("label")


def _aim() -> pd.DataFrame:
    """Per session: far-contra cued lick angle / animal's pre mean (the 3.37 behaviour)."""
    return pd.read_csv(il.OUT / "recovery_link_sessions_withpre_angle_z.csv")[["label", "beh"]].drop_duplicates("label")


def course(rows: list, title: str, name: str, ref_of: dict) -> None:
    """One column per animal, one row per (label, df, value): the measure per SESSION over days, pre shaded."""
    import matplotlib.pyplot as plt
    D = _days()
    fig, axs = plt.subplots(len(rows), 4, figsize=(17, 2.6 * len(rows)), squeeze=False, sharex="col")
    for i, row in enumerate(rows):
        lab, df, val = row[:3]
        val2 = row[3] if len(row) > 3 else None             # optional second measure overlaid in purple
        g0 = df.merge(D, on="label", how="left") if "day" not in df.columns else df
        for j, a in enumerate(AN):
            ax = axs[i, j]
            g = g0[g0.animal == a].groupby(["label", "day", "epoch"])[val].mean().reset_index().sort_values("day")
            ax.axvspan(g.day.min() - 1 if len(g) else -12, 0, color="0.93", zorder=0)
            ax.axvline(0, color="k", lw=0.7, ls="--")
            ax.plot(g.day, g[val], "o-", color="k", ms=3.5, lw=1, label=row[4] if len(row) > 4 else None)
            if val2 is not None:
                g2 = g0[g0.animal == a].groupby(["label", "day"])[val2].mean().reset_index().sort_values("day")
                ax.plot(g2.day, g2[val2], "s--", color="#8e44ad", ms=3, lw=1, label=row[5] if len(row) > 5 else val2)
                if j == 0:
                    ax.legend(fontsize=7, frameon=False)
            if ref_of.get(val) is not None:
                ax.axhline(ref_of[val], color="0.6", lw=0.6, ls=":")
            if len(g):
                pm = g[g.epoch == "pre"][val].mean()
                ax.axhline(pm, color="#2980b9", lw=0.8, ls="-", alpha=0.6)
            ax.set_title(f"{a}: {lab}", fontsize=8.5)
            if i == len(rows) - 1:
                ax.set_xlabel("day after stroke", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle(title + "\nOne dot per SESSION; grey band = pre-stroke sessions; blue line = the animal's pre mean.",
                 fontsize=9.5)
    _fit(fig)
    fig.savefig(il.OUT / "figures" / f"{name}_sessions.png", dpi=95)
    plt.close(fig)


def landing_errors(p: pd.DataFrame) -> pd.DataFrame:
    """Per CUED lick: lateral LANDING shift = the lick's lateral landing (extension peak, `act_lat` of the decoder table,
    + = contralesional; the same licks and coordinate the decoder is scored against) minus the animal's PRE mean landing
    at the same spout. 0 = lands where it landed before the stroke at that spout; - = ipsilesional of it.
    (Not the closest-approach `lat_err_mm` of the lick table: picking the frame nearest the spout pulls that toward the
    spout and hides most of the deviation.)"""
    t = p[p.set == "trial"].dropna(subset=["act_lat", "pos"]).copy()
    pre = t[t.epoch == "pre"].groupby(["animal", "pos"]).act_lat.mean()
    t["err"] = t.act_lat - [pre.get((a, q), np.nan) for a, q in zip(t.animal, t.pos)]
    return t.dropna(subset=["err"])[["animal", "epoch", "label", "err"]]


def error_hist(sets: tuple = ("iti", "trial"), name: str = "iti_check_error_per_lick", landing: bool = False,
               angle: bool = False) -> None:
    """Per LICK: signed decoding error (decoded - actual, lateral) distributions per epoch; POOLED column first (animals
    weighted equally), then per animal. A SHIFTED distribution = bias; a WIDER one = scatter. landing=True adds a
    BEHAVIOUR row below: the cued licks' lateral LANDING error vs the target spout (`landing_errors`).
    Stats (session = unit): per session, mean and SD of the signed error. Per animal: change vs pre with a within-animal
    session bootstrap 95% CI (2000 draws). Pooled: paired nested bootstrap (animals > sessions, `boot_delta`) + the
    number of animals changing in that direction. * = CI excludes 0."""
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    p = pd.read_csv(il.OUT / "zscore_check" / "where_preds_z.csv")
    if angle:   # ANGLE version: lick angle atan2(lateral, sagittal) in degrees (decoded angle from the two decoded axes)
        p = p.dropna(subset=["act_sag", "dec_sag"]).copy()
        p["act_lat"] = np.degrees(np.arctan2(p.act_lat, p.act_sag))
        p["dec_lat"] = np.degrees(np.arctan2(p.dec_lat, p.dec_sag))
    unit, rng_ = ("deg", 40) if angle else ("mm", 3)
    p["err"] = p.dec_lat - p.act_lat
    p = p.dropna(subset=["err"])
    # CONTROL for the shift: the error a PULL-ONLY decoder would make (decoded = a + b * actual, a, b fit on the animal's
    # pre licks per lick set). A changed lick distribution moves this mean too; observed shift minus this = beyond pull
    parts_ = []
    for (_st, _a), g in p.groupby(["set", "animal"]):
        pre_ = g[g.epoch == "pre"]
        b_, c_ = np.polyfit(pre_.act_lat, pre_.dec_lat, 1)
        parts_.append(g.assign(pull_err=c_ + (b_ - 1) * g.act_lat))
    p = pd.concat(parts_)
    xdec = f"decoded - actual {'ANGLE' if angle else 'LATERAL'}, {unit} (+ = read more contralesional)"
    rows = [(f"{'ITI' if st == 'iti' else 'cued'}: NEURAL decoder error", p[p.set == st], xdec, -rng_, rng_)
            for st in sets]
    if landing:
        rows.append(("cued: BEHAVIOUR landing shift", landing_errors(p),
                     f"landing - own pre landing at spout ({unit}, + = contra)", -1.33 * rng_, 1.33 * rng_))
    rng = np.random.default_rng(0)

    def within(h, ep, v):
        a_, b_ = h[h.epoch == ep][v].to_numpy(), h[h.epoch == "pre"][v].to_numpy()
        if len(a_) == 0 or len(b_) == 0:
            return None
        bs = [rng.choice(a_, len(a_)).mean() - rng.choice(b_, len(b_)).mean() for _ in range(2000)]
        return a_.mean() - b_.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    def fmt(t):
        return "" if t is None else f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]{'*' if t[1] > 0 or t[2] < 0 else ''}"
    cols = ["POOLED", *AN]
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(4.1 * len(cols), 1.2 + 3.2 * len(rows)), squeeze=False)
    for i, (rlab, P, xlab, x0, x1) in enumerate(rows):
        bins = np.linspace(x0, x1, 61)
        Ss = P.groupby(["animal", "epoch", "label"]).err.agg(["mean", "std", "size"]).reset_index()
        Ss = Ss[Ss["size"] >= 20]
        for j, a in enumerate(cols):
            ax = axs[i, j]
            lines = []
            for ep in EPS:
                if a == "POOLED":
                    parts = [P[(P.animal == b) & (P.epoch == ep)].err.to_numpy() for b in AN]
                    parts = [x for x in parts if len(x) >= 20]
                    if not parts:
                        continue
                    e = np.concatenate(parts)
                    w = np.concatenate([np.full(len(x), 1 / len(x)) for x in parts])
                else:
                    e = P[(P.animal == a) & (P.epoch == ep)].err.to_numpy()
                    w = None
                    if len(e) < 20:
                        continue
                mu = np.average(e, weights=w)
                ax.hist(np.clip(e, x0, x1), bins=bins, weights=w, density=True, histtype="step", lw=1.6,
                        color=EPOCH_LINE[ep], label=ep)
                ax.axvline(mu, ymax=0.64, color=EPOCH_LINE[ep], lw=0.9, ls="--")   # stops below the stats text
                if "pull_err" in P.columns:
                    # PULL-ONLY expected mean (triangle on the x axis): where this epoch's mean error would sit if only
                    # the lick landings had moved = pre mean + (mean pull error now - mean pull error pre), same weights
                    def _wm(q, a=a):
                        if a == "POOLED":
                            xs = [q[q.animal == b].pull_err.to_numpy() for b in AN]
                            xs = [x for x in xs if len(x) >= 20]
                            return np.mean([x.mean() for x in xs]) if xs else np.nan
                        return q[q.animal == a].pull_err.mean()
                    if ep == "pre":
                        mu_pre, pe_pre = mu, _wm(P[P.epoch == "pre"])
                    else:
                        x_exp = mu_pre + _wm(P[P.epoch == ep]) - pe_pre
                        ax.plot([x_exp], [0], marker="^", ms=8, color=EPOCH_LINE[ep], mec="k", mew=0.5, clip_on=False,
                                zorder=7)
                if ep == "pre":
                    continue
                if a == "POOLED":
                    txt = []
                    for v, nm in (("mean", "shift"), ("std", "width")):
                        prd = {b: h[v].to_numpy() for b, h in Ss[Ss.epoch == "pre"].groupby("animal")}
                        post = {b: h[v].to_numpy() for b, h in Ss[Ss.epoch == ep].groupby("animal") if b in prd}
                        ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                        am = (Ss[Ss.epoch == ep].groupby("animal")[v].mean()
                              - Ss[Ss.epoch == "pre"].groupby("animal")[v].mean()).dropna()
                        n_dir = int((am * np.sign(ci.point) > 0).sum())
                        sig = "*" if ci.lo > 0 or ci.hi < 0 else ""
                        txt.append(f"{nm} {ci.point:+.2f} [{ci.lo:+.2f},{ci.hi:+.2f}]{sig} {n_dir}/{len(am)}")
                    lines.append((ep, "; ".join(txt)))
                else:
                    h = Ss[Ss.animal == a]
                    lines.append((ep, f"shift {fmt(within(h, ep, 'mean'))}; width {fmt(within(h, ep, 'std'))}"))
            if "pull_err" in P.columns:                  # expected shift from the pull alone, per epoch (session means)
                q = P if a == "POOLED" else P[P.animal == a]
                sm = q.groupby(["animal", "epoch", "label"]).pull_err.mean().groupby(["animal", "epoch"]).mean()
                am_ = sm.unstack("epoch")
                if "pre" in am_.columns:
                    exp = [f"{e[:3]} {(am_[e] - am_['pre']).mean():+.2f}" for e in EPS[1:] if e in am_.columns]
                    lines.append(("pre", "CONTROL, pull-only expected shift: " + ", ".join(exp)))
            for k, (ep, t) in enumerate(lines):
                sig_ = (not t.startswith("CONTROL")) and "]*" in t.split(";")[0]      # shift significant -> * + bold
                ax.text(0.01, 0.99 - 0.06 * k, t if t.startswith("CONTROL") else f"{ep}{'*' if sig_ else ''}: {t}",
                        transform=ax.transAxes, fontsize=6.3, va="top", color=EPOCH_LINE[ep],
                        fontweight="bold" if sig_ else "normal")
            ax.axvline(0, ymax=0.64, color="0.5", lw=0.5)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
            if j == 0:
                ax.legend(fontsize=7, frameon=False, loc="center left")
            ax.set_title(f"{'POOLED (animals equal)' if a == 'POOLED' else a}: {rlab}", fontsize=8.5,
                         fontweight="bold" if j == 0 else None)
            ax.set_xlabel(xlab, fontsize=7.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    beh = (f" Bottom row = BEHAVIOUR: {'lick angle' if angle else 'lateral landing'} (3-D, extension peak) minus the animal's pre landing at the same spout."
           if landing else "")
    fig.suptitle(f"Per-LICK {'ANGLE' if angle else 'lateral'} error distributions ({unit}). NEURAL rows: frozen where-decoder (z-scored Allen areas), decoded - "
                 "actual." + beh + "\nA shifted distribution = BIAS, a wider one = SCATTER; dashed = epoch mean. Text = "
                 "change vs pre in the per-SESSION mean (shift) and SD (width) in axis units, 95% CI (epoch* = shift significant):\nper animal = session "
                 "bootstrap; POOLED = paired nested bootstrap (animals > sessions) + animals changing that way; * = CI "
                 "excludes 0.\nCONTROL (black text; triangles on the x axis = the expected mean): the shift a PULL-ONLY decoder (a + b x actual, fit on pre) "
                 "would show from the changed lick distribution alone.\nNB the decoder error includes its pull toward the pre mean, so a changed lick "
                 "distribution moves the mean error too; the offset BEYOND the pull (the beyond-pull offset figures) removes that. "
                 "PROVISIONAL", fontsize=9.5)
    _fit(fig)
    fig.savefig(il.OUT / "figures" / f"{name}.png", dpi=95)
    plt.close(fig)


def f0_severity() -> pd.DataFrame:
    """Does the post-lick response increase track the far-contra aim deficit? Within animal across POST sessions
    (log(day) removed from both, as 3.37), and per animal acute rise vs histology lesion burden (descriptive, n = 4)."""
    S = pd.read_csv(il.OUT / "f0_check.csv").merge(_days(), on="label").merge(_aim(), on="label")
    S = S[(S.epoch != "pre") & (S.day > 0)]
    rows = []
    for a, g in S.groupby("animal"):
        g = g.dropna(subset=["resp_auc_rel", "beh"])
        if len(g) < 5:
            continue
        d = np.log(g.day.to_numpy(float))
        res = {}
        for v in ("resp_auc_rel", "beh"):
            y = g[v].to_numpy(float)
            X = np.column_stack([np.ones(len(y)), d])
            res[v] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        rows.append(dict(animal=a, n=len(g), r_raw=float(np.corrcoef(g.resp_auc_rel, g.beh)[0, 1]),
                         r_detrended=float(np.corrcoef(res["resp_auc_rel"], res["beh"])[0, 1])))
    return pd.DataFrame(rows)


def main() -> int:
    rmse()
    f0()
    three_axis()
    error_hist()
    error_hist(("iti",), "iti_check_error_per_lick_iti")
    error_hist(("trial",), "cued_check_error_per_lick", landing=True)
    error_hist(("iti",), "iti_check_error_per_lick_iti_angle", angle=True)
    error_hist(("trial",), "cued_check_error_per_lick_angle", landing=True, angle=True)
    R0 = pd.read_csv(il.OUT / "where_rmse_split_sessions.csv")
    p = pd.read_csv(il.OUT / "zscore_check" / "where_preds_z.csv")
    p["err"] = p.dec_lat - p.act_lat
    E0 = (p.dropna(subset=["err"]).groupby(["set", "animal", "epoch", "label"]).err
          .agg(shift="mean", width="std", n="size").reset_index())
    E0 = E0[E0.n >= 20]
    for st, nm, sfx in (("iti", "ITI", "iti_check_rmse"), ("trial", "cued", "cued_check_rmse")):
        R = R0[R0.set == st]
        course([(f"{nm} {'lateral' if ax_ == 'lat' else 'sagittal'} {lab}", R[R.axis == ax_], v)
                for ax_ in ("lat", "sag") for v, lab in (("bias", "|bias|"), ("sd", "scatter (SD)"))],
               f"NEURAL: frozen where-decoder error per session, {nm} licks (z-scored decoder; mm).", sfx, {})
        E = E0[E0.set == st]
        course([(f"{nm} lat SHIFT (mean error, mm; + = contra)", E, "shift"),
                (f"{nm} lat WIDTH (SD of error, mm)", E, "width")],
               f"NEURAL: per-session summary of the per-lick error histograms ({nm} licks, lateral, z-scored decoder).",
               "iti_check_error" if st == "iti" else "cued_check_error", {"shift": 0})
    F = pd.read_csv(il.OUT / "f0_check.csv").merge(_aim(), on="label", how="left")
    course([("F0 470 (raw fluorescence) / pre", F, "f0_470_rel"),
            ("post-lick response / pre: dF/F vs dF (dF = dF/F x F0)", F, "resp_auc_rel", "resp_dF_rel",
             "dF/F (as on the page)", "dF, raw units"),
            ("far-contra lick angle / pre (behaviour)", F, "beh")],
           "NEURAL / MEASUREMENT: F0 and the ITI post-lick response per session, with far-contra aim for comparison.",
           "iti_check_f0", {"f0_470_rel": 1, "resp_auc_rel": 1, "resp_dF_rel": 1, "beh": 1})
    T = pd.read_csv(il.OUT / "where_3axis_sessions.csv")
    T = T[T.set == "iti"]
    course([(f"ITI {ax_}: {lab}", T[T.axis == ax_], v) for ax_ in ("lat", "fwd", "down")
            for v, lab in (("r_norm", "r / pre"), ("beyond_d", "offset beyond pull (mm)"))],
           "NEURAL: frozen where-decoder on three axes per session, ITI licks (z-scored).", "iti_check_3axis",
           {"r_norm": 1, "beyond_d": 0})
    sev = f0_severity()
    pd.set_option("display.width", 200)
    print("== does the ITI post-lick response (dF/F / pre) track far-contra aim within animal? (post sessions)")
    print(sev.round(2).to_string(index=False))
    print("wrote iti_check_{rmse,f0,3axis}[_sessions].png, iti_check_error_per_lick[_iti].png, "
          "iti_check_error_sessions.png; CUED: cued_check_error_per_lick.png (+ landing row), "
          "cued_check_{error,rmse}_sessions.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
