"""Figures for three checks that were table-only (Priya 10-27: "figure?"):
  rmse   the where-decoder's error split into |bias| and scatter (SD) per session (`iti_where_rmse_split.py`)
  f0     baseline fluorescence F0 (470, 415 nm) and the post-lick response in dF/F vs dF units (`iti_f0_check.py`)
  3axis  the where-decoder on lateral / forward / down axes: r / pre and the like-for-like (same-place) offset
         (`iti_where_3axis.py`; replaced 'offset beyond the pre pull' 10-27 late)
Every panel: thin lines = animals (session means per epoch), black = mean of animals, diamonds = mean of animals' pre +
change vs pre with the paired nested bootstrap 95% CI (animals > sessions, `analysis_kit.boot_delta`; * = CI excludes
0), and the count of animals changing in the point's direction. Reads the session CSVs; writes
<E>/iti_licks/figures/lick_check_{rmse,3axis}.png (ITI + cued) and lick_iti_check_f0.png (ITI). Usage: python scripts/iti_check_figures.py
FIGURE DATA (Priya 10-27: re-make plots without re-doing analysis): every figure's drawn numbers (dots, means, CIs,
histogram samples, texts) are computed first and saved to <OUT>/figdata/<figure stem>.pkl; the draw step reads only
that. `python scripts/iti_check_figures.py --plot-only` redraws all sixteen figures from those files (no CSV read, no
bootstrap).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import likeforlike as lfl  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]

WOUT = il.OUT_W if il.ITI_SUBSET else il.OUT     # subset runs (dock, ...) write to their own folder (10-29)

def epoch_compute(df: pd.DataFrame, val: str) -> dict:
    """Everything one epoch panel draws (session dots with their jitter, animal lines, mean of animals, pre SEM, the
    paired nested-bootstrap change per post epoch); picklable."""
    am = df.groupby(["animal", "epoch"])[val].mean()
    sess = df.groupby(["label", "epoch"])[val].mean().reset_index()      # one grey dot per SESSION
    jit = np.random.default_rng(1)
    dots = []
    for x, e in enumerate(EPS):
        v = sess[sess.epoch == e][val].to_numpy(float)
        dots.append((x - 0.18 + jit.uniform(-0.06, 0.06, len(v)), v))
    lines = [(a, [am.get((a, e), np.nan) for e in EPS]) for a in AN]
    mu = [am.xs(e, level="epoch").mean() if e in am.index.get_level_values("epoch") else np.nan for e in EPS]
    prd = {a: h[val].dropna().to_numpy() for a, h in df[df.epoch == "pre"].groupby("animal")}
    pv = sess[sess.epoch == "pre"][val].dropna().to_numpy(float)       # pre-stroke spread: SEM over pre SESSIONS
    pre = (pv.std(ddof=1) / np.sqrt(len(pv)), len(pv)) if len(pv) > 1 else None
    post_pts = []
    for x, e in enumerate(EPS[1:], start=1):
        post = {a: h[val].dropna().to_numpy() for a, h in df[df.epoch == e].groupby("animal")}
        post = {a: v for a, v in post.items() if len(v) and a in prd}
        if not post:
            continue
        ci = ak.boot_delta(post, prd, np.random.default_rng(0))
        d = (am.xs(e, level="epoch") - am.xs("pre", level="epoch")).dropna()
        n_dir = int((d * np.sign(ci.point) > 0).sum())
        sig = ci.lo > 0 or ci.hi < 0
        post_pts.append((x, ci.point, ci.lo, ci.hi, f"{ci.point:+.2f}{' *' if sig else ''}\n{n_dir}/{len(d)}"))
    return {"dots": dots, "lines": lines, "mu": mu, "pre": pre, "post": post_pts}


def epoch_draw(ax, P: dict, ref: float | None = None, title: str = "", ylab: str = "") -> None:
    import matplotlib.pyplot as plt
    cols = plt.get_cmap("tab10")
    for xs, v in P["dots"]:
        ax.scatter(xs, v, s=7, color="0.55", alpha=0.6, zorder=2, lw=0)
    for k, (a, ys) in enumerate(P["lines"]):
        ax.plot(range(4), ys, "o-", color=cols(k), ms=4, lw=1, alpha=0.7, label=a)
    mu = P["mu"]
    ax.plot(range(4), mu, "-", color="k", lw=2.2, zorder=5, label="mean of animals")
    if P["pre"] is not None:
        sem, n = P["pre"]
        ax.errorbar(0.13, mu[0], yerr=sem, fmt="D", color="k", ms=5, capsize=3, zorder=6)
        ax.text(0.13, mu[0] + sem, f"pre +/-SEM\n({n} sessions)", ha="center", va="bottom", fontsize=6.5)
    for x, pt, lo, hi, txt in P["post"]:
        ax.errorbar(x + 0.13, mu[0] + pt, yerr=[[pt - lo], [hi - pt]], fmt="D", color="k", ms=5, capsize=3, zorder=6)
        ax.text(x + 0.13, mu[0] + hi, txt, ha="center", va="bottom", fontsize=7)
    if ref is not None:
        ax.axhline(ref, color="0.7", lw=0.7, ls=":")
    ax.set_xticks(range(4), EPS, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylab, fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def epoch_panel(ax, df: pd.DataFrame, val: str, ref: float | None = None, title: str = "", ylab: str = "") -> None:
    """Compute + draw one epoch panel (kept for importers, e.g. iti_where_angle_reach)."""
    epoch_draw(ax, epoch_compute(df, val), ref=ref, title=title, ylab=ylab)


def _fit(fig) -> None:
    """tight_layout leaving room for the suptitle: ~0.2 in per title line + 0.3 in margin (no title / panel overlap)."""
    t = fig._suptitle.get_text() if fig._suptitle is not None else ""
    n = t.count(chr(10)) + 1 if t else 0
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.17 * n + 0.12) / fig.get_size_inches()[1]))


def grid_draw(F: dict) -> None:
    """Draw a grid of epoch panels (rmse / f0 / 3axis): F = name, shape, figsize, panels [(i, j, P, ref, title, ylab)],
    share (panels put on one y scale), suptitle."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(*F["shape"], figsize=F["figsize"], squeeze=False)
    for i, j, P, ref, title, ylab in F["panels"]:
        epoch_draw(axs[i, j], P, ref=ref, title=title, ylab=ylab)
    if F.get("share"):
        lo = min(axs[i, j].get_ylim()[0] for i, j in F["share"])
        hi = max(axs[i, j].get_ylim()[1] for i, j in F["share"])
        for i, j in F["share"]:
            axs[i, j].set_ylim(lo, hi)                    # (3) and (4) on ONE scale: the gap between them is F0
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(F["suptitle"], fontsize=9.5)
    _fit(fig)
    fig.savefig(WOUT / "figures" / f"{F['name']}.png", dpi=100)
    plt.close(fig)


def rmse() -> dict:
    S = pd.read_csv(il.OUT / "where_rmse_split_sessions.csv")
    rows = [(st, ax_) for st in ("iti", "trial") for ax_ in ("lat", "sag")]
    panels = []
    for i, (st, ax_) in enumerate(rows):
        g = S[(S.set == st) & (S.axis == ax_)]
        for j, (val, lab) in enumerate((("rmse", "RMSE"), ("bias", "|bias| = |mean error|"),
                                        ("sd", "scatter = SD of error"))):
            panels.append((i, j, epoch_compute(g, val), None, f"{'ITI' if st == 'iti' else 'cued'} "
                           f"{'LATERAL' if ax_ == 'lat' else 'SAGITTAL'}: {lab}", "z-scored decoder, mm"))
    return dict(kind="grid", name="lick_check_rmse", shape=(len(rows), 3), figsize=(13, 3.3 * len(rows)), panels=panels,
                suptitle="NEURAL: why does the frozen where-decoder's error grow after the stroke? Per session, error = decoded - "
                "actual; RMSE^2 = bias^2 + SD^2.\nDecoder on z-scored " + il.basis_label() + " features (lags 0..0.2 s; KEY-row "
                "decoder). A growing |bias| with flat SD = predictions SHIFTED, not more scattered. PROVISIONAL")


def f0() -> dict:
    S = pd.read_csv(il.OUT / "f0_check.csv")
    panels = (("f0_470_rel", "(1) F0 470 nm, raw fluorescence (/ animal's pre mean)"),
              ("f0_415_rel", "(2) F0 415 nm (/ pre)"),
              ("resp_auc_rel", "(3) post-lick response in dF/F (/ pre)\n= what the page's response figures show"),
              ("resp_dF_rel", "(4) post-lick response in dF, raw units (/ pre)\n= (3) x (1), session by session"))
    P = [(0, j, epoch_compute(S, val), 1.0, lab, "/ pre" if j == 0 else "") for j, (val, lab) in enumerate(panels)]
    return dict(kind="grid", name="lick_iti_check_f0", shape=(1, 4), figsize=(17, 4.6), panels=P, share=[(0, 2), (0, 3)],
                suptitle="NEURAL / MEASUREMENT: is the larger post-lick response an F0 artefact (dF/F inflated by a fall in "
                "baseline fluorescence)? F0 = mean of frames_average.npy over the central 60% of the image;\nresponse "
                "= the ITI-lick response area (orofacial ROIs, the ITI-lick PETH figures). If F0 fell, dF/F would rise with no change in dF: the dF "
                "panel answers it. PROVISIONAL")


def three_axis() -> dict:
    S = pd.read_csv(il.OUT / "where_3axis_sessions.csv")
    axes = (("lat", "LATERAL (+ = contralesional)"), ("fwd", "FORWARD (+ = further forward)"),
            ("down", "DOWN (+ = further down)"))
    P = []
    lfl_lab = "like-for-like offset (same place),\nmm (change vs pre)"
    for i, (st, val, lab) in enumerate((("iti", "r_norm", "r / animal's pre"), ("iti", "lfl_d", lfl_lab),
                                        ("trial", "r_norm", "r / animal's pre"), ("trial", "lfl_d", lfl_lab))):
        for j, (ax_, an_) in enumerate(axes):
            g = S[(S.set == st) & (S.axis == ax_)]
            P.append((i, j, epoch_compute(g, val), 1.0 if val == "r_norm" else 0.0,
                      f"{'ITI' if st == 'iti' else 'cued'} licks, {an_}", lab if j == 0 else ""))
    return dict(kind="grid", name="lick_check_3axis", shape=(4, 3), figsize=(14, 13.5), panels=P,
                suptitle=f"NEURAL: the frozen where-decoder on all THREE axes (z-scored {il.basis_label()} features, lags "
                "0..0.2 s; one ridge per axis, trained on August pre licks, pre leave-one-session-out).\nSagittal extension "
                "~ |down| (r 0.96); forward is nearly independent of it (r -0.05). Like-for-like offset = decoded - the "
                f"animal's pre licks that landed at the same place on that axis ({lfl.REF_LAB}). PROVISIONAL")


def _days() -> pd.DataFrame:
    from recovery_timing import series
    return series("lick")[["label", "day"]].drop_duplicates("label")


def _aim() -> pd.DataFrame:
    """Per session: far-contra cued lick angle change vs the animal's pre mean (deg; the recovery-link behaviour)."""
    return pd.read_csv(il.OUT / "recovery_link_sessions_withpre_angle_z.csv")[["label", "beh"]].drop_duplicates("label")


def course(rows: list, title: str, name: str, ref_of: dict) -> dict:
    """One column per animal, one row per (label, df, value): the measure per SESSION over days, pre shaded. Returns the
    figure data (per panel: day / value series, the optional overlay, the animal's pre mean)."""
    D = _days()
    panels = []
    for i, row in enumerate(rows):
        lab, df, val = row[:3]
        val2 = row[3] if len(row) > 3 else None             # optional second measure overlaid in purple
        g0 = df.merge(D, on="label", how="left") if "day" not in df.columns else df
        for j, a in enumerate(AN):
            g = g0[g0.animal == a].groupby(["label", "day", "epoch"])[val].mean().reset_index().sort_values("day")
            p = {"i": i, "j": j, "title": f"{a}: {lab}", "span0": g.day.min() - 1 if len(g) else -12,
                 "x": g.day.to_numpy(), "y": g[val].to_numpy(), "label": row[4] if len(row) > 4 else None,
                 "ref": ref_of.get(val), "pm": g[g.epoch == "pre"][val].mean() if len(g) else None, "x2": None}
            if val2 is not None:
                g2 = g0[g0.animal == a].groupby(["label", "day"])[val2].mean().reset_index().sort_values("day")
                p.update(x2=g2.day.to_numpy(), y2=g2[val2].to_numpy(), label2=row[5] if len(row) > 5 else val2)
            panels.append(p)
    return dict(kind="course", name=f"{name}_sessions", nrows=len(rows), panels=panels,
                suptitle=title + "\nOne dot per SESSION; grey band = pre-stroke sessions; blue line = the animal's pre mean.")


def course_draw(F: dict) -> None:
    import matplotlib.pyplot as plt
    n = F["nrows"]
    fig, axs = plt.subplots(n, 4, figsize=(17, 2.6 * n), squeeze=False, sharex="col")
    for p in F["panels"]:
        i, j = p["i"], p["j"]
        ax = axs[i, j]
        ax.axvspan(p["span0"], 0, color="0.93", zorder=0)
        ax.axvline(0, color="k", lw=0.7, ls="--")
        ax.plot(p["x"], p["y"], "o-", color="k", ms=3.5, lw=1, label=p["label"])
        if p["x2"] is not None:
            ax.plot(p["x2"], p["y2"], "s--", color="#8e44ad", ms=3, lw=1, label=p["label2"])
            if j == 0:
                ax.legend(fontsize=7, frameon=False)
        if p["ref"] is not None:
            ax.axhline(p["ref"], color="0.6", lw=0.6, ls=":")
        if p["pm"] is not None:
            ax.axhline(p["pm"], color="#2980b9", lw=0.8, ls="-", alpha=0.6)
        ax.set_title(p["title"], fontsize=8.5)
        if i == n - 1:
            ax.set_xlabel("day after stroke", fontsize=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle(F["suptitle"], fontsize=9.5)
    _fit(fig)
    fig.savefig(WOUT / "figures" / f"{F['name']}.png", dpi=95)
    plt.close(fig)


COORDS = {   # name: (source, label, unit, histogram half-range)
    "angle": ("lat_sag", "ANGLE atan2(lateral, sagittal)", "deg", 40),
    "lateral": ("lat_sag", "LATERAL (x)", "mm", 3),
    "lat": ("3axis", "LATERAL x", "mm", 3),
    "fwd": ("3axis", "FORWARD y", "mm", 3),
    "down": ("3axis", "DOWN z", "mm", 4),
}


def coord_table(coord: str) -> pd.DataFrame:
    """Per lick: set, animal, label, epoch, pos (canonical spout), act, dec in one coordinate (`COORDS`). Angle and lateral
    come from the KEY-row where-decoder (lateral + sagittal ridges; angle = atan2 of the two, degrees); lat / fwd / down
    from the three-axis decoder (`iti_where_3axis.py`, one ridge per axis). All z-scored Allen-area features."""
    from wfield_local import orofacial_cohort as oc
    src = COORDS[coord][0]
    if src == "lat_sag":
        p = pd.read_csv(il.pred_file("zscore_check/where_preds_z.csv"))
        if coord == "angle":
            p = p.dropna(subset=["act_sag", "dec_sag"]).copy()
            p["act"] = np.degrees(np.arctan2(p.act_lat, p.act_sag))
            p["dec"] = np.degrees(np.arctan2(p.dec_lat, p.dec_sag))
        else:
            p["act"], p["dec"] = p.act_lat, p.dec_lat
    else:
        p = pd.read_csv(il.pred_file("where_3axis_preds.csv.gz"))
        p["act"], p["dec"] = p[f"act_{coord}"], p[f"dec_{coord}"]
        p["pos"] = [q if q in oc.POS_ORDER or "_" not in str(q) else oc.pos_key(str(q), "L") for q in p.pos.astype(str)]
    return p.dropna(subset=["act", "dec"])[["set", "animal", "label", "epoch", "pos", "act", "dec"]]


def landing_errors(p: pd.DataFrame) -> pd.DataFrame:
    """Per CUED lick: LANDING shift in the table's coordinate = the lick's landing (extension peak, the coordinate the
    decoder is scored against; + = contralesional / forward / down) minus the animal's PRE mean landing at the same
    spout. 0 = lands where it landed before the stroke at that spout. (Not the closest-approach `lat_err_mm` of the lick
    table: picking the frame nearest the spout pulls that toward the spout and hides most of the deviation.)"""
    t = p[p.set == "trial"].dropna(subset=["act", "pos"]).copy()
    pre = t[t.epoch == "pre"].groupby(["animal", "pos"]).act.mean()
    t["err"] = t.act - [pre.get((a, q), np.nan) for a, q in zip(t.animal, t.pos)]
    return t.dropna(subset=["err"])[["animal", "epoch", "label", "err"]]


def error_hist(sets: tuple = ("iti", "trial"), name: str = "lick_check_error_per_lick", landing: bool = False,
               coords: tuple = ("lateral",)) -> dict:
    """Per LICK: signed decoding error (decoded - actual) distributions per epoch, one NEURAL row per (coordinate, lick
    set) and, with landing=True, one BEHAVIOUR row per coordinate (`landing_errors`); POOLED column first (animals
    weighted equally), then per animal. A SHIFTED distribution = bias; a WIDER one = scatter.
    Stats (session = unit): per session, mean and SD of the signed error. Per animal: change vs pre with a within-animal
    session bootstrap 95% CI (2000 draws). Pooled: paired nested bootstrap (animals > sessions, `boot_delta`) + the
    number of animals changing in that direction. * = CI excludes 0. CONTROL (neural rows): the error a PULL-ONLY decoder
    (decoded = a + b x actual, a, b fit on the animal's pre licks per lick set) would make; triangles = its mean.
    Returns the figure data (per panel: the clipped per-lick errors + weights per epoch, means, triangles, text lines)."""
    rows = []
    for coord in coords:
        _src, clab, unit, rng_ = COORDS[coord]
        p = coord_table(coord)
        p["err"] = p.dec - p.act
        parts_ = []
        for (_st, _a), g in p.groupby(["set", "animal"]):
            pre_ = g[g.epoch == "pre"]
            b_, c_ = np.polyfit(pre_.act, pre_.dec, 1)
            parts_.append(g.assign(pull_err=c_ + (b_ - 1) * g.act))
        p = pd.concat(parts_)
        for st in sets:
            rows.append((f"{'ITI' if st == 'iti' else 'cued'} {clab.split(' ')[0].lower()}: NEURAL", p[p.set == st],
                         f"decoded - actual {clab}, {unit}", -rng_, rng_))
        if landing:
            rows.append((f"cued {clab.split(' ')[0].lower()}: BEHAVIOUR", landing_errors(p),
                         f"landing - own pre landing at spout, {clab}, {unit}", -1.33 * rng_, 1.33 * rng_))
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
    panels = []
    for i, (rlab, P, xlab, x0, x1) in enumerate(rows):
        Ss = P.groupby(["animal", "epoch", "label"]).err.agg(["mean", "std", "size"]).reset_index()
        Ss = Ss[Ss["size"] >= 20]
        for j, a in enumerate(cols):
            lines, hists, tris = [], [], []
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
                hists.append((ep, np.clip(e, x0, x1), w, mu))
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
                        tris.append((ep, mu_pre + _wm(P[P.epoch == ep]) - pe_pre))
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
            panels.append(dict(i=i, j=j, a=a, rlab=rlab, xlab=xlab, x0=x0, x1=x1, hists=hists, tris=tris, lines=lines))
    beh = (" BEHAVIOUR rows: landing in that coordinate (3-D, extension peak) minus the animal's pre landing at the same spout."
           if landing else "")
    title = (f"Per-LICK error distributions: {', '.join(COORDS[c][1] + ' (' + COORDS[c][2] + ')' for c in coords)}. NEURAL rows: frozen where-decoder (z-scored {il.basis_label()}), decoded - "
             "actual." + beh + "\nA shifted distribution = BIAS, a wider one = SCATTER; dashed = epoch mean. Text = "
             "change vs pre in the per-SESSION mean (shift) and SD (width) in axis units, 95% CI (epoch* = shift significant):\nper animal = session "
             "bootstrap; POOLED = paired nested bootstrap (animals > sessions) + animals changing that way; * = CI "
             "excludes 0.\nCONTROL (black text; triangles on the x axis = the expected mean): the shift a PULL-ONLY decoder (a + b x actual, fit on pre) "
             "would show from the changed lick distribution alone.\nNB the decoder error includes its pull toward the pre mean, so a changed lick "
             "distribution moves the mean error too; the like-for-like (same-place) offset figures remove that. "
             "PROVISIONAL")
    return dict(kind="hist", name=name, nrows=len(rows), ncols=len(cols), panels=panels, suptitle=title)


def hist_draw(F: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D             # one LINE legend for the figure (no boxes, never over the data)

    from wfield_local.orofacial_cohort import EPOCH_LINE
    fig, axs = plt.subplots(F["nrows"], F["ncols"], figsize=(4.1 * F["ncols"], 1.2 + 3.2 * F["nrows"]), squeeze=False)
    for p in F["panels"]:
        ax = axs[p["i"], p["j"]]
        bins = np.linspace(p["x0"], p["x1"], 61)
        for ep, e, w, mu in p["hists"]:
            ax.hist(e, bins=bins, weights=w, density=True, histtype="step", lw=1.6, color=EPOCH_LINE[ep], label=ep)
            ax.axvline(mu, ymax=0.64, color=EPOCH_LINE[ep], lw=0.9, ls="--")   # stops below the stats text
        for ep, x_exp in p["tris"]:
            ax.plot([x_exp], [0], marker="^", ms=8, color=EPOCH_LINE[ep], mec="k", mew=0.5, clip_on=False, zorder=7)
        for k, (ep, t) in enumerate(p["lines"]):
            sig_ = (not t.startswith("CONTROL")) and "]*" in t.split(";")[0]      # shift significant -> * + bold
            ax.text(0.01, 0.99 - 0.06 * k, t if t.startswith("CONTROL") else f"{ep}{'*' if sig_ else ''}: {t}",
                    transform=ax.transAxes, fontsize=6.3, va="top", color=EPOCH_LINE[ep],
                    fontweight="bold" if sig_ else "normal")
        ax.axvline(0, ymax=0.64, color="0.5", lw=0.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
        ax.set_title(f"{'POOLED' if p['a'] == 'POOLED' else p['a']}: {p['rlab']}", fontsize=8,
                     fontweight="bold" if p["j"] == 0 else None)
        ax.set_xlabel(p["xlab"], fontsize=7.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle(F["suptitle"], fontsize=9.5)
    fig.legend(handles=[Line2D([], [], color=EPOCH_LINE[e], lw=2, label=e) for e in EPS], loc="upper left", ncol=4,
               fontsize=9, frameon=False)
    _fit(fig)
    fig.savefig(WOUT / "figures" / f"{F['name']}.png", dpi=95)
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


STEMS = ["lick_check_rmse", "lick_iti_check_f0", "lick_check_3axis", "lick_check_error_per_lick",
         "lick_iti_check_error_per_lick", "lick_cued_check_error_per_lick", "lick_iti_check_error_per_lick_angle",
         "lick_cued_check_error_per_lick_angle", "lick_iti_check_error_per_lick_xyz", "lick_cued_check_error_per_lick_xyz",
         "lick_iti_check_rmse_sessions", "lick_iti_check_error_sessions", "lick_cued_check_rmse_sessions",
         "lick_cued_check_error_sessions", "lick_iti_check_f0_sessions", "lick_iti_check_3axis_sessions"]


def draw(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    {"grid": grid_draw, "course": course_draw, "hist": hist_draw}[F["kind"]](F)


def compute_subset():
    """ITI subset run (il.ITI_SUBSET, e.g. docked-spout licks): only the two ITI error figures (per-lick histograms and
    the per-session shift / width), from the subset's where cache (il.pred_file -> iti_dock_where_preds.py copy)."""
    yield error_hist(("iti",), "lick_iti_check_error_per_lick")
    p = pd.read_csv(il.pred_file("zscore_check/where_preds_z.csv"))
    p["err"] = p.dec_lat - p.act_lat
    E = (p[p.set == "iti"].dropna(subset=["err"]).groupby(["set", "animal", "epoch", "label"]).err
         .agg(shift="mean", width="std", n="size").reset_index())
    E = E[E.n >= 20]
    yield course([("ITI lat SHIFT (mean error, mm; + = contra)", E, "shift"), ("ITI lat WIDTH (SD of error, mm)", E, "width")],
                 "NEURAL: per-session summary of the per-lick error histograms (ITI licks, lateral, z-scored decoder).",
                 "lick_iti_check_error", {"shift": 0})


def compute_all():
    """Yield the figure data of every figure, in the original drawing order (the error histograms' within-animal
    bootstraps each start their own rng, so the order does not move a number either way)."""
    yield rmse()
    yield f0()
    yield three_axis()
    yield error_hist()
    yield error_hist(("iti",), "lick_iti_check_error_per_lick")
    yield error_hist(("trial",), "lick_cued_check_error_per_lick", landing=True)
    yield error_hist(("iti",), "lick_iti_check_error_per_lick_angle", coords=("angle",))
    yield error_hist(("trial",), "lick_cued_check_error_per_lick_angle", landing=True, coords=("angle",))
    yield error_hist(("iti",), "lick_iti_check_error_per_lick_xyz", coords=("lat", "fwd", "down"))
    yield error_hist(("trial",), "lick_cued_check_error_per_lick_xyz", landing=True, coords=("lat", "fwd", "down"))
    R0 = pd.read_csv(il.OUT / "where_rmse_split_sessions.csv")
    p = pd.read_csv(il.pred_file("zscore_check/where_preds_z.csv"))
    p["err"] = p.dec_lat - p.act_lat
    E0 = (p.dropna(subset=["err"]).groupby(["set", "animal", "epoch", "label"]).err
          .agg(shift="mean", width="std", n="size").reset_index())
    E0 = E0[E0.n >= 20]
    for st, nm, sfx in (("iti", "ITI", "lick_iti_check_rmse"), ("trial", "cued", "lick_cued_check_rmse")):
        R = R0[R0.set == st]
        yield course([(f"{nm} {'lateral' if ax_ == 'lat' else 'sagittal'} {lab}", R[R.axis == ax_], v)
                      for ax_ in ("lat", "sag") for v, lab in (("bias", "|bias|"), ("sd", "scatter (SD)"))],
                     f"NEURAL: frozen where-decoder error per session, {nm} licks (z-scored decoder; mm).", sfx, {})
        E = E0[E0.set == st]
        yield course([(f"{nm} lat SHIFT (mean error, mm; + = contra)", E, "shift"),
                      (f"{nm} lat WIDTH (SD of error, mm)", E, "width")],
                     f"NEURAL: per-session summary of the per-lick error histograms ({nm} licks, lateral, z-scored decoder).",
                     "lick_iti_check_error" if st == "iti" else "lick_cued_check_error", {"shift": 0})
    F = pd.read_csv(il.OUT / "f0_check.csv").merge(_aim(), on="label", how="left")
    yield course([("F0 470 (raw fluorescence) / pre", F, "f0_470_rel"),
                  ("post-lick response / pre: dF/F vs dF (dF = dF/F x F0)", F, "resp_auc_rel", "resp_dF_rel",
                   "dF/F (as on the page)", "dF, raw units"),
                  ("far-contra lick angle, change vs pre (deg; behaviour)", F, "beh")],
                 "NEURAL / MEASUREMENT: F0 and the ITI post-lick response per session, with far-contra aim for comparison.",
                 "lick_iti_check_f0", {"f0_470_rel": 1, "resp_auc_rel": 1, "resp_dF_rel": 1, "beh": 0})
    T = pd.read_csv(il.OUT / "where_3axis_sessions.csv")
    T = T[T.set == "iti"]
    yield course([(f"ITI {ax_}: {lab}", T[T.axis == ax_], v) for ax_ in ("lat", "fwd", "down")
                  for v, lab in (("r_norm", "r / pre"), ("lfl_d", "like-for-like offset, same place (mm)"))],
                 "NEURAL: frozen where-decoder on three axes per session, ITI licks (z-scored).", "lick_iti_check_3axis",
                 {"r_norm": 1, "lfl_d": 0})


def main() -> int:
    il.split_only("iti_check_figures")
    if plot_only():                                   # redraw from figdata only: no CSV, no bootstrap
        for stem in STEMS:
            draw(load_fd(stem))
        print(f"redrew {len(STEMS)} figures from {il.OUT / 'figdata'}")
        return 0
    if il.ITI_SUBSET:
        (WOUT / "figures").mkdir(parents=True, exist_ok=True)
        for F in compute_subset():
            save_fd(F["name"], F)
            draw(F)
        return 0
    for F in compute_all():
        save_fd(F["name"], F)
        draw(F)
    sev = f0_severity()
    pd.set_option("display.width", 200)
    print("== does the ITI post-lick response (dF/F / pre) track far-contra aim within animal? (post sessions)")
    print(sev.round(2).to_string(index=False))
    print("wrote lick_check_{rmse,3axis}.png, lick_iti_check_f0.png, lick_iti_check_{rmse,f0,3axis}_sessions.png, "
          "lick_check_error_per_lick.png, lick_iti_check_error_per_lick[_angle,_xyz].png, "
          "lick_iti_check_error_sessions.png; CUED: lick_cued_check_error_per_lick[_angle,_xyz].png (+ landing row), "
          "lick_cued_check_{error,rmse}_sessions.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
