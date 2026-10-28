"""CUED versions of the ITI lick-response figures 14.1 / 14.3 / 14.4 (Priya 10-28: "yes add the cued fig 14 versions";
"can we show 2 s before to 3 s after?", "for cue-aligned also -2 to +3", "show a supplemental version per spout position").

The FIRST post-cue lick of each engaged trial, from the same session files, ROIs, baselines, tests and drawing helpers as the
ITI figures (`iti_licks.key_figures` / `ramp_compute`). The stored traces (`first_peth` -1.5..+0.75 s around the lick,
`first_cue` -1..+2.5 s around the cue) are too short for a -2..+3 s view, so `--extract` RE-SAMPLES them per session into
<OUT_W>/cued_wide/<label>.npz with exactly the steps of `iti_licks.extract` (ROI v8 masks, area means of U, drift basis
removed, x1e3; times = the stored first licks, rebuilt by `iti_licks._first_lick_times` and matched to first_rt) on LAGS_W
-2..+3 s, lick- AND cue-aligned, with each trial's spout. Every session is VALIDATED: the re-sampled trace at the stored
lags must equal the stored `first_peth` (float16 tolerance) -- the max |difference| is saved and printed.
Sessions with >= MIN_N first licks; per-spout cells >= MIN_SPOUT licks.
  CAVEAT (two alignments): the first cued lick comes ~0.2-0.5 s after the cue, so the lick-aligned pre-lick window
  (-0.4..-0.05 s) overlaps the CUE response -- not a self-paced build-up. Baselines: lick-aligned -1.5..-1.0 s before the
  lick (before the cue for reaction times < 1 s, the large majority); cue-aligned -1.0..-0.5 s before the cue (the spout has
  been in place ~2-3 s).
Figures (figdata + --plot-only):
  lick_cued_ramp_sessions.png         lick-aligned per-session PETH -2..+3 s, mean +- SEM over sessions; row 2 / the
                                      animal's PRE response AUC; per-bin tests (sig bars, -2..+3 s)
  lick_cued_ramp_cue_sessions.png     the same cue-aligned (row 2 / the animal's pre cue-response AUC)
  lick_cued_ramp_summary.png          per-session AUCs: lick response 0..0.5 s, pre-lick window, cue response 0..0.5 s
  lick_cued_ramp_perspout.png         SUPPLEMENT: lick-aligned, per spout (columns) x key ROI (rows), / the animal's
                                      pre response AUC (all spouts), mean +- SEM over sessions
  lick_cued_ramp_cue_perspout.png     the same cue-aligned
Outputs <E>/iti_licks/cued_ramp_auc_sessions.csv, cued_ramp_session_tests.csv, sig_bins_cued_ramp.csv.
Usage: python scripts/lick_cued_ramp.py --extract [--jobs N]; python scripts/lick_cued_ramp.py [--plot-only]
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
MIN_SPOUT = 5
LAGS_W = np.round(np.arange(-2.0, 3.0001, 0.05), 3)
BASE_L = il.BASE                      # lick-aligned baseline (s from lick), as the ITI figures
BASE_C = (-1.0, -0.5)                 # cue-aligned baseline (s from cue)
ROIS = il.KEY_ROIS
SPOUT_ROIS = ["ALM_right", "tjM1_right", "SSp-m_right", "cortex"]
WIDE = "cued_wide"
SPEC = (("resp_auc", "lick response AUC 0..0.5 s\n(dF/F x1e3 * s)"),
        ("resp_auc_n", "lick response AUC\n/ animal's pre response AUC"),
        ("pre_auc", "pre-lick window AUC -0.4..-0.05 s\n(CUE-contaminated)"),
        ("cue_auc", "cue response AUC 0..0.5 s after cue"),
        ("cue_auc_n", "cue response AUC\n/ animal's pre cue AUC"))


# ------------------------------------------------------------------------------------------------------------ EXTRACT
def wide_session(label: str) -> dict:
    """One session (module-level for spawn) -> <OUT_W>/cued_wide/<label>.npz; returns the validation numbers."""
    f = il.OUT_W / WIDE / f"{label}.npz"
    if f.exists():
        z = np.load(f)
        return {"label": label, "cached": True, "n": int(len(z["pos"])), "maxdiff": float(z["maxdiff"])}
    from wfield_local import combined_encoding as ce
    d = il._load(label)
    cue, fl = il._first_lick_times(label, d["first_rt"])
    # == iti_licks.extract's trace construction (keep in step with it; validated below against the stored first_peth)
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    if cols != list(d["cols"]):
        raise ValueError(f"{label}: ROI columns differ from the session file")
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    ok = np.isfinite(fl) & np.isfinite(cue)
    chk = il._sample(R, ft, fl[ok], il.LAGS_P)
    ref = np.asarray(d["first_peth"], float)[ok]
    maxdiff = float(np.nanmax(np.abs(chk - ref))) if ok.any() else np.nan
    scale = float(np.nanmax(np.abs(ref))) if ok.any() else np.nan

    def samp(t):
        X = il._sample(R, ft, t, LAGS_W)
        out = (t[:, None] + LAGS_W[None, :] < ft[0]) | (t[:, None] + LAGS_W[None, :] > ft[-1])
        X[out] = np.nan                                   # no clamped values at the recording edges
        return X.astype(np.float16)

    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, lick=samp(fl[ok]), cue=samp(cue[ok]), pos=np.asarray(d["first_pos"], str)[ok],
                        cols=np.array(cols), lags=LAGS_W, maxdiff=maxdiff, scale=scale, n_unmatched=int((~ok).sum()))
    return {"label": label, "cached": False, "n": int(ok.sum()), "maxdiff": maxdiff, "scale": scale,
            "n_unmatched": int((~ok).sum())}


def extract(jobs: int | None) -> None:
    from wfield_local import analysis_kit as ak
    labs = il.labels()
    res, fails = ak.fan_sessions(labs, wide_session, jobs=jobs)
    V = pd.DataFrame([v for _x, v in res])
    print(V.to_string(index=False))
    print(f"== validation: max |re-sampled - stored first_peth| = {V.maxdiff.max():.4f} (dF/F x1e3; traces span up to "
          f"{V.get('scale', pd.Series([np.nan])).max():.1f}); unmatched first licks {int(V.get('n_unmatched', pd.Series([0])).sum())}")
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")


# ------------------------------------------------------------------------------------------------------------ COMPUTE
def _auc(v: np.ndarray, lags: np.ndarray, t0: float, t1: float) -> float:
    m = (lags >= t0 - 1e-9) & (lags <= t1 + 1e-9)
    return float(np.trapezoid(v[m], lags[m]))


def _sessions() -> dict:
    D = {}
    for x in il.labels():
        f = il.OUT_W / WIDE / f"{x}.npz"
        if not f.exists():
            raise SystemExit(f"missing {f}: run --extract first")
        z = np.load(f)
        d = il._load(x)
        D[x] = {"lick": z["lick"].astype(float), "cue": z["cue"].astype(float), "pos": z["pos"],
                "cols": list(z["cols"]), "animal": str(d["animal"]), "epoch": str(d["epoch"])}
    return D


def _pe(D: dict, key: str, base: tuple, rois: list, pos: str | None = None, min_n: int = MIN_N) -> pd.DataFrame:
    rows = []
    for x, d in D.items():
        P = d[key] if pos is None else d[key][d["pos"] == pos]
        if len(P) < min_n:
            continue
        P = P - np.nanmean(P[:, (LAGS_W >= base[0]) & (LAGS_W <= base[1]), :], 1, keepdims=True)
        m = np.nanmean(P, 0)
        for roi in rois:
            if roi in d["cols"]:
                rows.append({"label": x, "animal": d["animal"], "epoch": d["epoch"], "roi": roi,
                             "peth": m[:, d["cols"].index(roi)], "n": len(P)})
    return pd.DataFrame(rows)


def _curves(PE: pd.DataFrame, den: dict, rois: list, test: bool = True) -> tuple[dict, list]:
    panels, recs = {}, []
    for j, roi in enumerate(rois):
        for i, norm in enumerate((False, True)):
            long, curves = [], []
            for ep in il.EP4:
                t = PE[(PE.roi == roi) & (PE.epoch == ep)] if len(PE) else PE
                if t.empty:
                    continue
                M = np.stack(t.peth.to_numpy())
                if norm:
                    M = M / np.array([den.get((roi, a), np.nan) for a in t.animal])[:, None]
                mu = np.nanmean(M, 0)
                se = np.nanstd(M, 0, ddof=1) / np.sqrt(np.isfinite(M[:, 0]).sum())
                curves.append((ep, mu, se, len(t)))
                long += [pd.DataFrame({"label": x, "animal": a, "epoch": ep, "lag": LAGS_W, "v": m_})
                         for x, a, m_ in zip(t.label, t.animal, M)]
            S = il.sig_compute(pd.concat(long, ignore_index=True).dropna(), value="v",
                               tag={"roi": roi, "normalised": norm}) if (long and test) else None
            if S is not None:
                recs += S["recs"]
            panels[(i, j)] = {"roi": roi, "norm": norm, "curves": curves, "sig": S}
    return panels, recs


def compute() -> dict:
    import iti_lick_order as lo
    D = _sessions()
    PL = _pe(D, "lick", BASE_L, ROIS)
    PC = _pe(D, "cue", BASE_C, ROIS)
    rows = []
    for (_i, a), (_j, c) in zip(PL.iterrows(), PC.iterrows()):
        assert a.label == c.label and a.roi == c.roi
        rows.append({"label": a.label, "animal": a.animal, "epoch": a.epoch, "roi": a.roi, "n": a.n,
                     "resp_auc": _auc(a.peth, LAGS_W, 0.0, 0.5), "pre_auc": _auc(a.peth, LAGS_W, -0.4, -0.05),
                     "cue_auc": _auc(c.peth, LAGS_W, 0.0, 0.5)})
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
    lick, rl = _curves(PL, dr.to_dict(), ROIS)
    cue, rc = _curves(PC, dc.to_dict(), ROIS)
    pd.DataFrame(rl + rc).to_csv(il.OUT_W / "sig_bins_cued_ramp.csv", index=False)
    # SUPPLEMENT per spout: normalised by the animal's pre response AUC over ALL spouts (one scale across spouts)
    spout = {}
    for key, base, den in (("lick", BASE_L, dr.to_dict()), ("cue", BASE_C, dc.to_dict())):
        for p in lo.POS:
            P_ = _pe(D, key, base, SPOUT_ROIS, pos=p, min_n=MIN_SPOUT)
            pan, _ = _curves(P_, den, SPOUT_ROIS, test=False)
            spout[(key, p)] = {j: pan[(1, j)] for j in range(len(SPOUT_ROIS))}
    return dict(lick=lick, cue=cue, summary=sp_, spout=spout, pos=list(lo.POS))


# -------------------------------------------------------------------------------------------------------------- PLOTS
def _draw_sessions(F: dict, xlab: str, title: str, path: Path, nlab: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    fig, axs = plt.subplots(2, len(ROIS), figsize=(23, 7.8), sharex=True)
    for j, _roi in enumerate(ROIS):
        for i in (0, 1):
            ax = axs[i, j]
            p = F[(i, j)]
            for ep, mu, se, n in p["curves"]:
                ax.plot(LAGS_W, mu, color=colors[ep], lw=2, label=f"{ep} ({n})")
                ax.fill_between(LAGS_W, mu - se, mu + se, color=colors[ep], alpha=0.12, lw=0)
            if p["sig"] is not None:
                il.sig_draw(ax, p["sig"], colors)
                il._headroom([ax])
            ax.axvline(0, color="0.6", lw=0.6)
            ax.axhline(0, color="0.8", lw=0.6)
            ax.set_xlim(-2, 3)
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


def _draw_spout(F: dict, key: str, xlab: str, title: str, path: Path, nlab: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    pos = F["pos"]
    fig, axs = plt.subplots(len(SPOUT_ROIS), len(pos), figsize=(22, 11), sharex=True, sharey="row")
    for c, p in enumerate(pos):
        for r, roi in enumerate(SPOUT_ROIS):
            ax = axs[r, c]
            for ep, mu, se, n in F["spout"][(key, p)][r]["curves"]:
                ax.plot(LAGS_W, mu, color=colors[ep], lw=1.8, label=f"{ep} ({n})")
                ax.fill_between(LAGS_W, mu - se, mu + se, color=colors[ep], alpha=0.12, lw=0)
            ax.axvline(0, color="0.6", lw=0.6)
            ax.axhline(0, color="0.8", lw=0.6)
            ax.set_xlim(-2, 3)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(p.replace("_", " "), fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{roi}\n{nlab}", fontsize=8)
            if r == len(SPOUT_ROIS) - 1:
                ax.set_xlabel(xlab, fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    out = il.OUT_W / "figures"
    _draw_sessions(F["lick"], "time from first cued lick onset (s)",
                   "NEURAL, NO DECODER: dF/F around the FIRST CUED lick of each engaged trial (cued version of 14.1 / 14.3), "
                   "lick-ONSET aligned, -2..+3 s, baseline -1.5..-1.0 s; per-session PETH, line = mean over sessions, band = "
                   "SEM.\nCAUTION: the cue comes ~0.2-0.5 s before this lick, so activity just before 0 is mostly the CUE "
                   "response, not a self-paced build-up. Top raw; bottom / the animal's PRE response AUC.",
                   out / "lick_cued_ramp_sessions.png", "/ animal's pre response AUC")
    _draw_sessions(F["cue"], "time from cue (s)",
                   "NEURAL, NO DECODER: dF/F around the CUE of engaged trials with a first lick (cue-aligned, -2..+3 s; "
                   "baseline -1.0..-0.5 s before the cue); the first lick follows ~0.2-0.5 s after the cue.\nPer-session "
                   "PETH, line = mean over sessions, band = SEM. Top raw; bottom / the animal's PRE cue-response AUC.",
                   out / "lick_cued_ramp_cue_sessions.png", "/ animal's pre cue-response AUC")
    _draw_spout(F, "lick", "time from first cued lick (s)",
                "SUPPLEMENT, NEURAL: first cued lick PER SPOUT (target), lick-onset aligned (-2..+3 s, baseline -1.5..-1.0 s), "
                "/ the animal's PRE response AUC over all spouts (one scale across spouts); line = mean over sessions "
                f"(>= {MIN_SPOUT} licks of that spout), band = SEM; _right = contralesional. Descriptive (no tests).",
                out / "lick_cued_ramp_perspout.png", "/ pre response AUC")
    _draw_spout(F, "cue", "time from cue (s)",
                "SUPPLEMENT, NEURAL: CUE-aligned per spout (-2..+3 s, baseline -1.0..-0.5 s), / the animal's PRE "
                "cue-response AUC over all spouts; line = mean over sessions, band = SEM. Descriptive (no tests).",
                out / "lick_cued_ramp_cue_perspout.png", "/ pre cue AUC")
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
    if "--extract" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
        extract(jobs)
        return 0
    if plot_only():
        plot(load_fd("lick_cued_ramp"))
        return 0
    F = compute()
    save_fd("lick_cued_ramp", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
