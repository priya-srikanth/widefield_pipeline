"""CUED versions of the ITI lick-response figures 14.1 / 14.3 / 14.4 (Priya 10-28: "yes add the cued fig 14 versions").

Same session files, ROIs, baselines, tests and drawing helpers as the ITI figures (`iti_licks.key_figures` /
`ramp_compute`), on the FIRST post-cue lick of each engaged trial (`first_peth`, onset-aligned, LAGS_P; `first_cue`,
cue-aligned, LAGS_C; both stored by `iti_licks.extract`). Sessions with >= MIN_N first licks.
  CAVEAT (why there are two alignments): the first cued lick comes ~0.2-0.5 s after the cue, so the lick-aligned
  pre-lick "build-up" window (-0.4..-0.05 s) overlaps the CUE response -- it is not a self-paced build-up. The
  cue-aligned PETH shows cue response then lick. The lick-aligned baseline (-1.5..-1.0 s from the lick) is before the
  cue for reaction times < 1.0 s (the large majority); the cue-aligned baseline is -1.0..-0.5 s before the cue (the
  spout has been in place ~2-3 s by then).
Figures (figdata + --plot-only):
  lick_cued_ramp_sessions.png      (= 14.1 / 14.3) lick-aligned per-session PETH, mean +- SEM over sessions; row 2 /
                                   the animal's PRE response AUC (one scale per animal); per-bin tests (sig bars)
  lick_cued_ramp_cue_sessions.png  the same cue-aligned
  lick_cued_ramp_summary.png       (= 14.4) per-session AUCs: lick response 0..0.5 s, pre-lick window -0.4..-0.05 s
                                   (cue-contaminated), cue response 0..0.5 s after the cue; raw and / pre response AUC
Outputs <E>/iti_licks/cued_ramp_auc_sessions.csv, cued_ramp_session_tests.csv.
Usage: python scripts/lick_cued_ramp.py [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

MIN_N = 10
BASE_C = (-1.0, -0.5)                 # cue-aligned baseline (s from cue)
ROIS = il.KEY_ROIS
SPEC = (("resp_auc", "lick response AUC 0..0.5 s\n(dF/F x1e3 * s)"),
        ("resp_auc_n", "lick response AUC\n/ animal's pre response AUC"),
        ("pre_auc", "pre-lick window AUC -0.4..-0.05 s\n(CUE-contaminated)"),
        ("cue_auc", "cue response AUC 0..0.5 s after cue"),
        ("cue_auc_n", "cue response AUC\n/ animal's pre cue AUC"))


def _auc(v: np.ndarray, lags: np.ndarray, t0: float, t1: float) -> float:
    m = (lags >= t0 - 1e-9) & (lags <= t1 + 1e-9)
    return float(np.trapezoid(v[m], lags[m]))


def _pe(D: dict, key: str, lags: np.ndarray, base: tuple) -> pd.DataFrame:
    rows = []
    for x, d in D.items():
        P = d[key]
        if len(P) < MIN_N:
            continue
        P = P.astype(float)
        P = P - P[:, (lags >= base[0]) & (lags <= base[1]), :].mean(1, keepdims=True)
        m = np.nanmean(P, 0)
        cols = list(d["cols"])
        for roi in ROIS:
            if roi in cols:
                rows.append({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "roi": roi,
                             "peth": m[:, cols.index(roi)], "n": len(P)})
    return pd.DataFrame(rows)


def _curves(PE: pd.DataFrame, lags: np.ndarray, den: dict, xlim: tuple) -> tuple[dict, list]:
    shown = (lags >= xlim[0]) & (lags <= xlim[1])
    panels, recs = {}, []
    for j, roi in enumerate(ROIS):
        for i, norm in enumerate((False, True)):
            long, curves = [], []
            for ep in il.EP4:
                t = PE[(PE.roi == roi) & (PE.epoch == ep)]
                if t.empty:
                    continue
                M = np.stack(t.peth.to_numpy())
                if norm:
                    M = M / np.array([den.get((roi, a), np.nan) for a in t.animal])[:, None]
                mu = np.nanmean(M, 0)
                se = np.nanstd(M, 0, ddof=1) / np.sqrt(np.isfinite(M[:, 0]).sum())
                curves.append((ep, mu, se, len(t)))
                long += [pd.DataFrame({"label": x, "animal": a, "epoch": ep, "lag": lags[shown], "v": m_[shown]})
                         for x, a, m_ in zip(t.label, t.animal, M)]
            S = il.sig_compute(pd.concat(long, ignore_index=True), value="v", tag={"roi": roi, "normalised": norm}) \
                if long else None
            if S is not None:
                recs += S["recs"]
            panels[(i, j)] = {"roi": roi, "norm": norm, "curves": curves, "sig": S}
    return panels, recs


def compute() -> dict:
    labs = il.labels()
    D = {x: il._load(x) for x in labs}
    D = {x: d for x, d in D.items() if "first_peth" in d and "first_cue" in d}
    PL = _pe(D, "first_peth", il.LAGS_P, il.BASE)
    PC = _pe(D, "first_cue", il.LAGS_C, BASE_C)
    rows = []
    for (_i, a), (_j, c) in zip(PL.iterrows(), PC.iterrows()):
        assert a.label == c.label and a.roi == c.roi
        rows.append({"label": a.label, "animal": a.animal, "epoch": a.epoch, "roi": a.roi, "n": a.n,
                     "resp_auc": _auc(a.peth, il.LAGS_P, 0.0, 0.5), "pre_auc": _auc(a.peth, il.LAGS_P, -0.4, -0.05),
                     "cue_auc": _auc(c.peth, il.LAGS_C, 0.0, 0.5)})
    T0 = pd.DataFrame(rows)
    dr = T0[T0.epoch == "pre"].groupby(["roi", "animal"]).resp_auc.mean()
    dc = T0[T0.epoch == "pre"].groupby(["roi", "animal"]).cue_auc.mean()
    T0["resp_auc_n"] = T0.resp_auc / np.array([dr.get((r_, a_), np.nan) for r_, a_ in zip(T0.roi, T0.animal)])
    T0["cue_auc_n"] = T0.cue_auc / np.array([dc.get((r_, a_), np.nan) for r_, a_ in zip(T0.roi, T0.animal)])
    T0.to_csv(il.OUT_W / "cued_ramp_auc_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): lick response AUC / pre, contralesional tjM1 + SSp-m")
    print(T0[T0.roi.isin(["tjM1_right", "SSp-m_right"])].pivot_table(index=["roi", "animal"], columns="epoch",
                                                                     values="resp_auc_n").reindex(columns=il.EP4)
          .round(2).to_string())
    rng = np.random.default_rng(1)
    tests, sp_ = [], {}
    for j, roi in enumerate(ROIS):
        T = T0[T0.roi == roi]
        for i, (val, _ylab) in enumerate(SPEC):
            S = il.dots_compute(T, val, rng)
            sp_[(i, j)] = S
            tests += [{"roi": roi, **r_} for r_ in S["tests"]]
    TS = pd.DataFrame(tests)
    TS.to_csv(il.OUT_W / "cued_ramp_session_tests.csv", index=False)
    print("== change vs pre (nested bootstrap)\n" + TS.round(3).to_string(index=False))
    lick, rl = _curves(PL, il.LAGS_P, dr.to_dict(), (-1.5, 0.75))
    cue, rc = _curves(PC, il.LAGS_C, dc.to_dict(), (-1.0, 2.5))
    pd.DataFrame(rl + rc).to_csv(il.OUT_W / "sig_bins_cued_ramp.csv", index=False)
    return dict(lick=lick, cue=cue, summary=sp_)


def _draw_sessions(F: dict, lags: np.ndarray, xlim: tuple, xlab: str, title: str, path: Path,
                   nlab: str = "/ animal's pre response AUC") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    fig, axs = plt.subplots(2, len(ROIS), figsize=(21, 7.8), sharex=True)
    for j, _roi in enumerate(ROIS):
        for i in (0, 1):
            ax = axs[i, j]
            p = F[(i, j)]
            for ep, mu, se, n in p["curves"]:
                ax.plot(lags, mu, color=colors[ep], lw=2, label=f"{ep} ({n})")
                ax.fill_between(lags, mu - se, mu + se, color=colors[ep], alpha=0.12, lw=0)
            if p["sig"] is not None:
                il.sig_draw(ax, p["sig"], colors)
                il._headroom([ax])
            ax.axvline(0, color="0.6", lw=0.6)
            ax.axhline(0, color="0.8", lw=0.6)
            ax.set_xlim(*xlim)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(p["roi"], fontsize=9)
            if j == 0:
                ax.set_ylabel("dF/F x1e3 (baseline-subtracted)" if not p["norm"] else nlab)
            if i == 1:
                ax.set_xlabel(xlab)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(title + "\n" + il.SIG_NOTE, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    out = il.OUT_W / "figures"
    _draw_sessions(F["lick"], il.LAGS_P, (-1.5, 0.75), "time from first cued lick onset (s)",
                   "NEURAL, NO DECODER: dF/F around the FIRST CUED lick of each engaged trial (cued version of 14.1 / 14.3), "
                   "lick-ONSET aligned, baseline -1.5..-1.0 s; per-session PETH, line = mean over sessions, band = SEM.\n"
                   "CAUTION: the cue comes ~0.2-0.5 s before this lick, so activity before 0 is mostly the CUE response, "
                   "not a self-paced build-up (see the cue-aligned figure). Top raw; bottom / the animal's PRE response AUC.",
                   out / "lick_cued_ramp_sessions.png")
    _draw_sessions(F["cue"], il.LAGS_C, (-1.0, 2.5), "time from cue (s)",
                   "NEURAL, NO DECODER: dF/F around the CUE of engaged trials with a first lick (cue-aligned; baseline "
                   "-1.0..-0.5 s before the cue); the first lick follows ~0.2-0.5 s after the cue.\nPer-session PETH, line = "
                   "mean over sessions, band = SEM. Top raw; bottom / the animal's PRE cue-response AUC.",
                   out / "lick_cued_ramp_cue_sessions.png", nlab="/ animal's pre cue-response AUC")
    fig, axs = plt.subplots(len(SPEC), len(ROIS), figsize=(21, 15))
    for j, roi in enumerate(ROIS):
        for i, (_val, ylab) in enumerate(SPEC):
            ax = axs[i, j]
            ax.axhline(0, color="0.8", lw=0.6)
            il.dots_draw(ax, F["summary"][(i, j)], colors)
            if i == 0:
                ax.set_title(roi, fontsize=9)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
    fig.suptitle("NEURAL, NO DECODER: FIRST CUED lick, per-session dF/F AUCs (cued version of 14.4; one dot per session, all "
                 "animals pooled; black = mean +- SEM over sessions). Lick response = AUC 0..0.5 s after lick onset;\npre-lick "
                 "window = AUC -0.4..-0.05 s before it (overlaps the CUE response); cue response = AUC 0..0.5 s after the cue. "
                 "* = change from pre with 95% nested-bootstrap CI (animals > sessions) excluding 0.", fontsize=9.5)
    fig.tight_layout(rect=(0.01, 0, 0.99, 0.94))
    fig.savefig(out / "lick_cued_ramp_summary.png", dpi=110)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_cued_ramp"))
        return 0
    F = compute()
    save_fd("lick_cued_ramp", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
