"""Session-level view of the post-stroke gain (GAIN_BIO 11-04; Priya: show sessions, not just 4 animals).
Reads tables/ written by gain_bio.py / summarise_gain_bio.py; writes figs/fig4_gain_sessions.png.
A: rest gain per session, corrected 470 / raw 470 / 415, all with the same 0.1 Hz high-pass (like with like, Priya
11-06 (4); was unfiltered corrected 470 beside high-passed 415). B: gain vs days from stroke. C: still and far from quiet events vs all
rest frames. D: spatial uniformity (ipsi - contra; orofacial - posterior). E: orofacial and posterior gain apart.
Stars (top band): epoch_change_lmm vs pre from gain_stats.py (tables/S_lmm_*.csv); * p < .05, ** Holm p < .05 within
the panel's family (lmm_stars). B: day >= 7 per-animal slopes (tables/S_trend.csv).

PROVENANCE: ported 2026-11-05 from M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/gain_bio_1104/
  fig_sessions.py (deck slide 102, hidden, 'Backup: post-stroke gain is brain-wide'; GAIN_BIO_2026-11-04.md section 5).
  Computation, layout and styling unchanged (stars already in the top band via sig_marks.star_top + reserve_headroom;
  top spines already off; no spouts; no brain maps). Changes: tables / figs under <out> = gb_common.out_root()
  (--out; lp_licks.output_dir(<cohort_out>/gain_bio), run-tagged under the render profile) instead of the script's
  folder; the module body split into load() / draw(); --plot-only redraws from <out>/data/gain_fig4_figdata.pkl (the
  rebuild_figs/_common contract). Priya 11-06 (4): panel A = gb_common.GAIN_SIGS (A_SIGS), so 415 sits beside the raw 470
  with identical processing (and the corrected 470 with the same high-pass); the 415 / 470 ratio is printed in the panel.

    PYTHONPATH=<repo>;<repo>/scripts python scripts/gain_bio/fig_sessions.py [--out DIR] [--plot-only]
"""
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_common  # noqa: E402

gb_common.add_paths()
sys.path.insert(0, str(gb_common.SCRIPTS / "rebuild_figs"))
import _common  # noqa: E402

from wfield_local import cat_points as cp  # noqa: E402
from wfield_local import sig_marks as sm  # noqa: E402

NAME = "gain_fig4"
EP = ["pre", "acute", "subacute", "chronic"]
COL = {"pre": "#000000", "acute": "#d4237a", "subacute": "#c08a00", "chronic": "#00685f"}
STROKE = {"PS92": date(2026, 8, 17), "PS93": date(2026, 8, 17), "PS94": date(2026, 8, 16), "PS95": date(2026, 8, 16)}


#: panel A columns: (signal, tick label), like with like (gb_common.GAIN_SIGS: the same 0.1 Hz high-pass on all three)
A_SIGS = (("corr_hp", "corrected 470\n(calcium, HP)"), ("r470_hp", "raw 470\n(HP)"),
          ("r415_hp", "415 nm\n(non-calcium, HP)"))
assert tuple(k for k, _ in A_SIGS) == tuple(gb_common.GAIN_SIGS)


def day_of(label):
    a, md = label.split("_")
    return (date(2026, int(md[:2]), int(md[2:])) - STROKE[a]).days


def load(T: Path) -> dict:
    """Everything draw() reads: the six gain_bio / gain_stats tables."""
    return {"U": pd.read_csv(T / "C1_uniformity_sessions.csv"), "M": pd.read_csv(T / "C2_matched_sessions.csv"),
            "SS": pd.read_csv(T / "S_sessions.csv"), "TR": pd.read_csv(T / "S_trend.csv"),
            "LM": {k: pd.read_csv(T / f"S_lmm_{k}.csv") for k in ["A_gain", "D_space", "E_oro_post_parts"]},
            "H": pd.read_csv(T / "S_415_vs_470.csv")}


def draw(fd: dict, out_png: Path) -> Path:
    U, M, SS, TR, LM = fd["U"], fd["M"], fd["SS"], fd["TR"], fd["LM"]
    rest = U[(U.set == "R5")]

    def stars(axh, fam, sig, q, x0):
        off = cp.epoch_offsets(len(EP), 0.2)
        r = LM[fam][(LM[fam].sig == sig) & (LM[fam].quantity == q)].set_index("epoch")
        for ei, e in enumerate(EP[1:], 1):
            st = r.star.get(e, "")
            if isinstance(st, str) and st:
                sm.star_top(axh, x0 + off[ei], st, color=COL[e])

    fig, ax = plt.subplots(1, 5, figsize=(25, 4.8), gridspec_kw={"width_ratios": [1.7, 1.6, 1.0, 1.2, 1.2]})

    # A: corrected 470 / raw 470 / 415, rest, all high-passed alike (Priya 11-06 (4))
    a = ax[0]
    for ci, (sig, _lab) in enumerate(A_SIGS):
        off = cp.epoch_offsets(len(EP), 0.2)
        for ei, e in enumerate(EP):
            s = rest[(rest.sig == sig) & (rest.epoch == e)]
            x = ci + off[ei]
            cp.session_dots(a, x, s.gain.values, s.animal.values, COL[e], jitter=0.05, seed=ci * 10 + ei)
            am = s.groupby("animal").gain.mean()
            cp.summary_point(a, x + 0.06, am.mean(), am.std(ddof=1) / np.sqrt(len(am)), color=COL[e])
    for ci, (sig, _lab) in enumerate(A_SIGS):
        stars(a, "A_gain", sig, "lgain", ci)
    a.set_xticks(range(len(A_SIGS)))
    a.set_xticklabels([lab for _, lab in A_SIGS])
    h = fd.get("H")
    if h is not None and len(h):
        txt = ", ".join(f"{r.epoch} {r.ratio_415_470:.2f}" for r in h.itertuples() if np.isfinite(r.ratio_415_470))
        a.text(0.02, 0.02, f"415 / raw 470 log-gain change: {txt}", transform=a.transAxes, fontsize=7.5,
               color="0.3", va="bottom")
    a.axhline(1, color="0.6", lw=0.8, ls="--")
    a.set_ylabel("rest-frame SD / animal's pre\n(median over 46 areas)")
    a.set_title("A  Gain per session, rest frames", loc="left", fontsize=10)

    # B: gain vs days from stroke
    b = ax[1]
    s = rest[rest.sig == "corr"].copy()
    s["day"] = s.label.map(day_of)
    for e in EP:
        z = s[s.epoch == e]
        for an, g in z.groupby("animal"):
            b.plot(g.day, g.gain, ls="none", marker=cp.animal_marker(an), mfc="none", mec=COL[e], mew=1.0,
                   ms=6, alpha=0.8)
    b.axhline(1, color="0.6", lw=0.8, ls="--")
    b.axvline(0, color="0.3", lw=0.8)
    b.set_xlabel("days from stroke")
    b.set_ylabel("rest-frame SD / animal's pre")
    tr = TR[(TR.sig == "corr") & (TR.quantity == "lgain")].iloc[0]
    b.text(0.98, 0.04, f"slope from day 7: {tr.mean_slope_per10d:+.3f} log / 10 d, p = {tr.p:.2f} (n.s.)",
           transform=b.transAxes, ha="right", va="bottom", fontsize=8.5, color="0.3")
    b.set_title("B  Gain over time (corrected 470, each dot = a session)", loc="left", fontsize=10)

    # C: still + far from quiet events vs all rest frames
    c = ax[2]
    allr = M[(M.measure == "R5_chunks") & (M.sig == "corr")].set_index("label")
    stillf = M[(M.measure == "still_far") & (M.sig == "corr")].set_index("label")
    j = allr[["gain", "animal", "epoch"]].join(stillf[["gain"]], rsuffix="_still", how="inner")
    for e in EP:
        z = j[j.epoch == e]
        for an, g in z.groupby("animal"):
            c.plot(g.gain, g.gain_still, ls="none", marker=cp.animal_marker(an), mfc="none", mec=COL[e], mew=1.0,
                   ms=6, alpha=0.8)
    lo, hi = np.nanmin(j[["gain", "gain_still"]].values), np.nanmax(j[["gain", "gain_still"]].values)
    c.plot([lo, hi], [lo, hi], color="0.5", lw=0.8)
    c.set_xlabel("gain, all rest frames")
    c.set_ylabel("gain, still + > 2 s from quiet events")
    c.set_title("C  Not movement or quiet events", loc="left", fontsize=10)

    # D: uniformity per session
    d = ax[3]
    for ci, q in enumerate(["ipsi_contra", "oro_post"]):
        off = cp.epoch_offsets(len(EP), 0.2)
        for ei, e in enumerate(EP):
            z = rest[(rest.sig == "corr") & (rest.epoch == e)]
            x = ci + off[ei]
            cp.session_dots(d, x, z[q].values, z.animal.values, COL[e], jitter=0.05, seed=50 + ci * 10 + ei)
            am = z.groupby("animal")[q].mean()
            cp.summary_point(d, x + 0.06, am.mean(), am.std(ddof=1) / np.sqrt(len(am)), color=COL[e])
    d.axhline(0, color="0.6", lw=0.8, ls="--")
    stars(d, "D_space", "corr", "ipsi_contra", 0)
    stars(d, "D_space", "corr", "oro_post", 1)
    d.set_xticks([0, 1])
    d.set_xticklabels(["ipsi - contra", "orofacial -\nposterior"])
    d.set_ylabel("difference in log gain")
    d.set_title("D  No ipsi / contra difference", loc="left", fontsize=10)

    # E: orofacial and posterior gain apart (the chronic oro - post < 0 is posterior rising more, not orofacial falling)
    e_ = ax[4]
    z0 = SS[SS.sig == "corr"]
    for ci, q in enumerate(["oro", "post"]):
        off = cp.epoch_offsets(len(EP), 0.2)
        for ei, e in enumerate(EP):
            z = z0[z0.epoch == e]
            x = ci + off[ei]
            cp.session_dots(e_, x, np.exp(z[q].values), z.animal.values, COL[e], jitter=0.05, seed=80 + ci * 10 + ei)
            am = np.exp(z.groupby("animal")[q].mean())
            cp.summary_point(e_, x + 0.06, am.mean(), am.std(ddof=1) / np.sqrt(len(am)), color=COL[e])
        stars(e_, "E_oro_post_parts", "corr", q, ci)
    e_.axhline(1, color="0.6", lw=0.8, ls="--")
    e_.set_xticks([0, 1])
    e_.set_xticklabels(["orofacial\n(ALM, tjM1, SSp-m/n)", "posterior\n(VIS, RSP)"])
    e_.set_ylabel("rest-frame SD / animal's pre\n(geometric mean over areas)")
    e_.set_title("E  Orofacial stays up; posterior rises more", loc="left", fontsize=10)

    for x in ax:
        x.spines["top"].set_visible(False)
        x.spines["right"].set_visible(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="none", marker="o", mfc=COL[e], mec=COL[e], label=e) for e in EP]
    h += cp.animal_handles(["PS92", "PS93", "PS94", "PS95"])
    fig.legend(handles=h, loc="lower center", ncol=8, frameon=False, fontsize=9)
    sm.reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="gain_bio output root (default gb_common.out_root(), run-tagged under the render profile); "
                         "figdata goes to <out>/data/")
    ap.add_argument("--plot-only", action="store_true", help="redraw from <out>/data/gain_fig4_figdata.pkl")
    a = ap.parse_args(argv)
    gb_common.apply_args(a)
    out = gb_common.out_root()
    fd = _common.load_figdata(out, NAME) if a.plot_only else load(out / "tables")
    if not a.plot_only:
        gb_common.guard(out / "data")
        _common.save_figdata(out, NAME, fd)
    print(draw(fd, gb_common.guard(out / "figs") / "fig4_gain_sessions.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
