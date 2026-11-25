"""PI-deck slide 95 ("Lick response larger after stroke") with a SESSION-Z normalisation, for a hidden comparison slide
(Priya via a8, 11-04). Opt-in, standalone: nothing in lick_cued_ramp.py / iti_licks.py changes, no existing output is
written (all writes go to <iti_licks>/slide95_sessz_20261104/).

Slide 95 (PI_overview_2026-11-02.pptx, figure_provenance rows fig:iti_ramp_row1 + fig:cued_ramp_summary_row2):
  ITI row   = row 1 of lick_iti_ramp_wide_docked.png (lick_cued_ramp.py, WFIELD_ITI_FILTER=dock, out itidock_main):
              docked-spout ITI bout starts (gap >= 2 s), per-session PETH (baseline -1.5..-1.0 s), raw dF/F x1e3, mean
              +- SEM over sessions, per-bin bars = iti_licks.sig_compute (paired nested bootstrap, BH-FDR q .05).
  cued row  = row 2 of lick_cued_ramp_summary.png: first cued lick, resp_auc 0..1.0 s (PETH re-baselined -1.0..-0.5 s)
  ITI scalars (tables / LMM input only, not drawn): docked ITI resp_auc over lick_cued_ramp.ITI_RESP_W = 0..0.5 s
              (Priya 11-17; 0..1.0 s before)
              / the animal's pre mean; * = iti_licks.dots_compute (analysis_kit.boot_delta CI excludes 0).
SESSION-Z = variant (c) of docs/status/SD_RISE_2026-11-04.md (widefield_pipeline 371b6e1e; decoder_mix_checks/
sd_rise_1104/lickresp.py): each session x ROI is divided by sd_win = np.nanstd over ALL samples of that session's
lick-window matrix (the same kept licks, -2..+3 s, before any baseline subtraction), i.e. the SD of the same ROI trace
over the same frames. ITI row: PETH / sd_win (SD units). Cued row: (resp_auc / sd_win) / the animal's pre mean of that
quantity (= lickresp v_c_n). Tests identical to the main slide's, run on the z values.

The raw rows are recomputed with the SAME code and checked against the published tables / figdata before anything is
drawn (max |diff| printed). Figures: open session markers (shape = animal, colour = epoch), black = mean +- SEM over
sessions, stars / bars in the top band (sig_marks.star_top + reserve_headroom), no top spine. _right = contralesional.
Also writes LMM input tables (<name>_sessions.csv in the lmm_hidden format) for scripts/lmm_hidden/render on
branch lmm-hidden-20261103.

    python scripts/lick_slide95_sessz.py [--plot-only]

Re-render on another run (deck pass 11-17): env WFIELD_SESSZ_ITI_ROOT = that run's iti_licks folder (= iti_licks.OUT under
the run's env, asserted) and WFIELD_SESSZ_OUT = a new output folder (tagged by the run mode like WFIELD_ITI_OUT_W).
Defaults = the pre-rebuild v36 root and slide95_sessz_20261104 (unchanged). The ITI-row verification against
itidock_main and the lickresp cross-check are skipped when those published inputs do not exist for the run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["WFIELD_ITI_FILTER"] = "dock"            # the slide's ITI row; the cued rows are unaffected by it
from wfield_local import orofacial_outputs as _oo  # noqa: E402  11-22: the adopted run (was a literal v36 path)
_E = Path(os.environ.get("WFIELD_SESSZ_ITI_ROOT", str(_oo.encoding_root() / "iti_licks")))
OUTD = Path(os.environ.get("WFIELD_SESSZ_OUT", str(_E / "slide95_sessz_20261104")))
os.environ["WFIELD_ITI_OUT_W"] = str(OUTD)          # every write of the imported scripts (figdata included) lands here
sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_cued_ramp as lcr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

assert il.OUT == _E, (il.OUT, _E)
OUTD = il.OUT_W                                     # = OUTD by default; <OUTD>/<run tag> under a run mode
LAG = lcr.LAGS_W
ROIS = lcr.ROIS
FD = "lick_slide95_sessz"
NORM = ("SESSION-Z (SD_RISE 11-04 variant c): each session x ROI / its SD over all samples of the same lick windows "
        "(-2..+3 s, same licks, before baselining)")


def _sd(D: dict, PE: pd.DataFrame, key: str) -> np.ndarray:
    return np.array([float(np.nanstd(D[x][key][:, :, D[x]["cols"].index(r)])) for x, r in zip(PE.label, PE.roi)])


def _curves0(PE: pd.DataFrame, col: str) -> dict:
    """Row 0 of lick_cued_ramp._curves (raw curves + per-bin test), on PE[col]."""
    panels = {}
    for j, roi in enumerate(ROIS):
        long, curves = [], []
        for ep in il.EP4:
            t = PE[(PE.roi == roi) & (PE.epoch == ep)]
            if t.empty:
                continue
            M = np.stack(t[col].to_numpy())
            mu = np.nanmean(M, 0)
            se = np.nanstd(M, 0, ddof=1) / np.sqrt(np.isfinite(M[:, 0]).sum())
            curves.append((ep, mu, se, len(t)))
            long += [pd.DataFrame({"label": x, "animal": a, "epoch": ep, "lag": LAG, "v": m_})
                     for x, a, m_ in zip(t.label, t.animal, M)]
        S = il.sig_compute(pd.concat(long, ignore_index=True).dropna(), value="v", tag={"roi": roi, "row": 0})
        panels[j] = {"roi": roi, "curves": curves, "sig": S}
    return panels


def _dots(T: pd.DataFrame, val: str) -> dict:
    """il.dots_compute (same test, boot_delta rng 0) + the animal of every dot and n animals per epoch."""
    S = il.dots_compute(T, val, np.random.default_rng(1))
    an = {ep: T[T.epoch == ep].dropna(subset=[val]).animal.to_numpy() for ep in il.EP4}
    for e in S["eps"]:
        e["animal"] = an[e["ep"]]
    ref = set(T[(T.epoch == "pre")].dropna(subset=[val]).animal)
    for r in S["tests"]:
        r["n_animals"] = len(ref & set(an[r["epoch"]]))
        r["sig"] = bool(r["lo"] > 0 or r["hi"] < 0)
    return S


def _scalars(D: dict, PE: pd.DataFrame, key: str) -> pd.DataFrame:
    br = (LAG >= lcr.BASE_RESP[0]) & (LAG <= lcr.BASE_RESP[1])
    T = PE[["label", "animal", "epoch", "roi", "n"]].copy()
    # window per set (Priya 11-17): cued first lick over lcr.RESP_W (0..1.0 s), ITI bout starts over lcr.ITI_RESP_W (0..0.5 s)
    win = lcr.ITI_RESP_W if key == "iti" else lcr.RESP_W
    T["resp_auc"] = [lcr._auc(p - np.nanmean(p[br]), LAG, *win) for p in PE.peth]
    T["sd_win"] = _sd(D, PE, key)
    T["resp_z"] = T.resp_auc / T.sd_win
    for v in ("resp_auc", "resp_z"):
        pre = T[T.epoch == "pre"].groupby(["roi", "animal"])[v].mean()
        T[v + "_n"] = T[v] / np.array([pre.get(k, np.nan) for k in zip(T.roi, T.animal)])
    return T


def _maxdiff(a, b) -> float:
    return float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def verify(TC: pd.DataFrame, TI: pd.DataFrame, raw_iti: dict, SC: dict, SI: dict) -> dict:
    """Raw recomputation vs the published tables / the docked figdata the slide-95 crops were drawn from."""
    out = {}
    pc = pd.read_csv(il.OUT / "cued_ramp_auc_sessions.csv")
    m = TC.merge(pc, on=["label", "roi"], suffixes=("", "_pub"))
    out["cued sessions rows"] = f"{len(m)} / {len(TC)} / {len(pc)}"
    out["cued resp_auc"] = _maxdiff(m.resp_auc, m.resp_auc_pub)
    out["cued resp_auc_n"] = _maxdiff(m.resp_auc_n, m.resp_auc_n_pub)
    pt = pd.read_csv(il.OUT / "cued_ramp_session_tests.csv")
    pt = pt[pt.value == "resp_auc_n"]
    mine = pd.DataFrame([{"roi": ROIS[j], **r} for j, S in SC.items() for r in S["tests"]])
    mm = mine.merge(pt, on=["roi", "epoch"], suffixes=("", "_pub"))
    out["cued tests rows"] = f"{len(mm)} / {len(pt)}"
    out["cued tests delta/lo/hi"] = max(_maxdiff(mm[c], mm[c + "_pub"]) for c in ("delta", "lo", "hi"))
    if not (il.OUT / "itidock_main").exists():       # no published docked ITI ramp for this run (rebuild 11-17)
        out["iti checks"] = "skipped: no itidock_main for this run"
        return out
    pi = pd.read_csv(il.OUT / "itidock_main" / "iti_ramp_auc_sessions_wide.csv")
    m = TI.merge(pi, on=["label", "roi"], suffixes=("", "_pub"))
    out["iti sessions rows"] = f"{len(m)} / {len(TI)} / {len(pi)}"
    out["iti resp_auc_n"] = _maxdiff(m.resp_auc_n, m.resp_auc_n_pub)
    ti = pd.read_csv(il.OUT / "itidock_main" / "iti_ramp_session_tests_wide.csv")
    ti = ti[ti.value == "resp_auc_n"]
    mine = pd.DataFrame([{"roi": ROIS[j], **r} for j, S in SI.items() for r in S["tests"]])
    mm = mine.merge(ti, on=["roi", "epoch"], suffixes=("", "_pub"))
    out["iti tests delta/lo/hi"] = max(_maxdiff(mm[c], mm[c + "_pub"]) for c in ("delta", "lo", "hi"))
    F = load_fd("lick_cued_ramp", root=il.OUT / "itidock_main" / "figdata")
    dmu, dse, nbad, sbad = 0.0, 0.0, 0, 0
    for j in range(len(ROIS)):
        pub, me = F["sess"]["iti"][(0, j)], raw_iti[j]
        for (e1, mu1, se1, n1), (e2, mu2, se2, n2) in zip(pub["curves"], me["curves"]):
            nbad += (e1 != e2) or (n1 != n2)
            dmu, dse = max(dmu, _maxdiff(mu1, mu2)), max(dse, _maxdiff(se1, se2))
        sbad += pub["sig"]["rows"] != me["sig"]["rows"]
    out["iti curves mean"], out["iti curves sem"] = dmu, dse
    out["iti curves epoch/n mismatches"], out["iti sig-bar panels differing"] = nbad, sbad
    return out


def compute() -> dict:
    D = lcr._sessions()
    PC = lcr._pe(D, "lick", lcr.BASE_L, ROIS)
    PI = lcr._pe(D, "iti", lcr.BASE_L, ROIS)
    TC, TI = _scalars(D, PC, "lick"), _scalars(D, PI, "iti")
    PI["sd_win"] = TI.sd_win.to_numpy()
    PI["peth_z"] = [p / s for p, s in zip(PI.peth, PI.sd_win)]
    iti = {"raw": _curves0(PI, "peth"), "z": _curves0(PI, "peth_z")}
    cued = {v: {j: _dots(TC[TC.roi == r], v) for j, r in enumerate(ROIS)} for v in ("resp_auc_n", "resp_z_n")}
    itis = {v: {j: _dots(TI[TI.roi == r], v) for j, r in enumerate(ROIS)} for v in ("resp_auc_n", "resp_z_n")}
    V = verify(TC, TI, iti["raw"], cued["resp_auc_n"], itis["resp_auc_n"])
    # cross-check vs lickresp (variant c), if present
    c4 = Path("M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/sd_rise_1104/tables/C4_lick_sessions.csv")
    if c4.exists() and "WFIELD_SESSZ_ITI_ROOT" not in os.environ:      # lickresp table = pre-rebuild data
        C = pd.read_csv(c4)
        for st, T in (("cued", TC), ("iti_dock", TI)):
            m = T.merge(C[C.set == st][["label", "roi", "v_c_n", "sd_win"]], on=["label", "roi"], suffixes=("", "_c4"))
            V[f"{st} resp_z_n vs lickresp v_c_n"] = _maxdiff(m.resp_z_n, m.v_c_n)
            V[f"{st} sd_win vs lickresp"] = _maxdiff(m.sd_win, m.sd_win_c4)
    TC.to_csv(OUTD / "cued_sessions_raw_sessz.csv", index=False)
    TI.to_csv(OUTD / "iti_dock_sessions_raw_sessz.csv", index=False)
    rows = []
    for st, P in (("cued", cued), ("iti_dock", itis)):
        for v, S in P.items():
            for j, s_ in S.items():
                rows += [{"set": st, "norm": "raw" if v == "resp_auc_n" else "sessz", "roi": ROIS[j], **r}
                         for r in s_["tests"]]
    TS = pd.DataFrame(rows)
    TS.to_csv(OUTD / "tests_raw_vs_sessz.csv", index=False)
    srec = [{"norm": k, **r} for k, P in iti.items() for p in P.values() for r in p["sig"]["recs"]]
    pd.DataFrame(srec).to_csv(OUTD / "iti_sig_bins_raw_vs_sessz.csv", index=False)
    for st, T in (("cued", TC), ("iti_dock", TI)):     # lmm_hidden render input (= _pooled_resp.as_sessions format)
        for v, tag in (("resp_auc_n", "raw"), ("resp_z_n", "sessz")):
            L = T[np.isfinite(T[v])].assign(position=lambda t: t.roi, value=lambda t, v=v: t[v],
                                           raw=lambda t, v=v: t[v.replace("_n", "")])
            L[["position", "epoch", "animal", "label", "value", "raw", "n"]].to_csv(
                OUTD / f"lmm_{st}_{tag}_sessions.csv", index=False)
    return {"iti": iti, "cued": cued, "iti_scalar": itis, "verify": V, "tests": TS}


# ---------------------------------------------------------------------------------------------------------------- DRAW
def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_iti(ax, p: dict, colors, ylab: str | None, legend: bool):
    for ep, mu, se, n in p["curves"]:
        ax.plot(LAG, mu, color=colors[ep], lw=2, label=f"{ep} ({n})")
        ax.fill_between(LAG, mu - se, mu + se, color=colors[ep], alpha=0.12, lw=0)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlim(-2, 3)
    step = 0.05
    il.sig_draw(ax, p["sig"], colors, rows_top=0.985, row_h=step)
    ax._wf_mark_rows, ax._wf_mark_step = 3, step           # reserve_headroom: the bars sit in a free top band
    ax.set_title(p["roi"], fontsize=10)
    ax.set_xlabel("time from ITI bout-start lick (s)", fontsize=9)
    if ylab:
        ax.set_ylabel(ylab, fontsize=9)
    _style(ax)


def draw_dots(ax, S: dict, colors, ylab: str | None, roi: str):
    from wfield_local.orofacial_cohort import ANIMAL_MARKER as mk
    from wfield_local.sig_marks import star_top
    for e in S["eps"]:
        for a in np.unique(e["animal"]):
            s = e["animal"] == a
            ax.scatter(e["x"][s], e["v"][s], s=22, marker=mk.get(a, "o"), facecolors="none", edgecolors=colors[e["ep"]],
                       lw=0.9, alpha=0.8, zorder=3)
        ax.errorbar(e["k"] + 0.3, e["m"], yerr=e["se"], fmt="o", color="k", ms=6, capsize=5, lw=2.2, capthick=2.2,
                    zorder=5)
        ax.text(e["k"], 0.01, f"n={len(e['v'])}", ha="center", va="bottom", fontsize=6.5, color="0.4",
                transform=ax.get_xaxis_transform())
    ax.axhline(1, color="0.8", lw=0.6)
    ax.set_xticks(range(4), ["pre", "acute", "sub", "chronic"], fontsize=8)
    for k in S["stars"]:
        star_top(ax, k)
    ax.set_title(roi, fontsize=10)
    if ylab:
        ax.set_ylabel(ylab, fontsize=9)
    _style(ax)


def _legend(fig, colors, iti_curves=None, dots=True):
    """Figure legend below the panels (never on the data): ITI epoch lines with their session n, animal shapes."""
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER as mk
    h = []
    if iti_curves is not None:
        h += [Line2D([], [], color=colors[ep], lw=2, label=f"ITI {ep} ({n} sessions)") for ep, _m, _s, n in iti_curves]
    if dots:
        h += [Line2D([], [], marker=m, ls="", mfc="none", mec="0.3", label=a) for a, m in mk.items()]
        h += [Line2D([], [], marker="o", ls="", mfc="none", mec=colors[e], label=e) for e in il.EP4]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=8.5, frameon=False)


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    from wfield_local.sig_marks import reserve_headroom
    (OUTD / "figures").mkdir(parents=True, exist_ok=True)
    spec = {"raw": ("RAW dF/F x1e3 (as slide 95; re-rendered from the same code)",
                    "dF/F x1e3 (baseline-subtracted)", "resp_auc_n", "lick response AUC 0..1.0 s\n/ animal's pre"),
            "sessz": (NORM, "dF/F / session SD\n(baseline-subtracted)", "resp_z_n",
                      "(AUC 0..1.0 s / session SD)\n/ animal's pre")}
    test = ("Bars (ITI row): bins where that epoch differs from pre, paired nested bootstrap animals > sessions per bin, "
            "BH-FDR q .05 across bins, one row per epoch. * (cued row): change from pre, 95% nested-bootstrap CI "
            "(animals > sessions) excludes 0.\nOpen markers = sessions (shape = animal, colour = epoch); black = mean +- "
            "SEM over sessions. _right = contralesional.")
    for k, (what, yl_i, val, yl_c) in spec.items():
        for part in ("iti", "cued", "both"):
            nr = 2 if part == "both" else 1
            fig, axs = plt.subplots(nr, len(ROIS), figsize=(23, 4.2 * nr + 1.2), squeeze=False)
            r = 0
            if part in ("iti", "both"):
                for j in range(len(ROIS)):
                    draw_iti(axs[r, j], F["iti"]["raw" if k == "raw" else "z"][j], colors, yl_i if j == 0 else None,
                             j == 0)
                r += 1
            if part in ("cued", "both"):
                for j, roi in enumerate(ROIS):
                    draw_dots(axs[r, j], F["cued"][val][j], colors, yl_c if j == 0 else None, roi)
            reserve_headroom(fig)
            rows = {"iti": "docked-spout ITI bout-start licks (per-session PETH, baseline -1.5..-1.0 s)",
                    "cued": "FIRST CUED lick, per-session response AUC 0..1.0 s (baseline -1.0..-0.5 s) / the animal's pre",
                    "both": "top: docked-spout ITI bout-start licks (PETH, baseline -1.5..-1.0 s); bottom: first cued "
                            "lick, response AUC 0..1.0 s (baseline -1.0..-0.5 s) / the animal's pre"}[part]
            fig.suptitle(f"Lick response larger after stroke? {what}\n{rows}\n{test}", fontsize=10)
            _legend(fig, colors, F["iti"]["raw"][0]["curves"] if part != "cued" else None, dots=part != "iti")
            fig.tight_layout(rect=(0, 0.05 if nr == 2 else 0.08, 1, 0.85 if nr == 1 else 0.915))
            fig.savefig(OUTD / "figures" / f"slide95_{part}_{k}.png", dpi=110)
            plt.close(fig)


def report(F: dict) -> None:
    pd.set_option("display.width", 250)
    print("== VERIFICATION, raw recomputation vs published (max |diff|)")
    for k, v in F["verify"].items():
        print(f"  {k:45s} {v}")
    T = F["tests"].copy()
    T["txt"] = [f"{d:+.2f} [{lo:+.2f},{hi:+.2f}]{'*' if s else ''} n={n}" for d, lo, hi, s, n in
                zip(T.delta, T.lo, T.hi, T.sig, T.n_animals)]
    for st in ("cued", "iti_dock"):
        print(f"\n== {st}: change vs pre of value / animal's pre (boot_delta, * = CI excludes 0, n = animals)")
        print(T[T.set == st].pivot_table(index=["roi", "norm"], columns="epoch", values="txt", aggfunc="first")
              .reindex(columns=il.EP4[1:]).to_string())


def main() -> int:
    il.require_out_w(__file__)   # rule 11 (11-24): refuses the production iti_licks default before any compute
    if plot_only():
        F = load_fd(FD)
    else:
        OUTD.mkdir(parents=True, exist_ok=True)
        F = compute()
        save_fd(FD, F)
    report(F)
    plot(F)
    return 0


if __name__ == "__main__":
    sys.exit(main())
