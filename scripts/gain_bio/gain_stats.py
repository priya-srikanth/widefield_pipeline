"""Stats for the session-level gain figure (Priya 11-04: add stats; does gain go DOWN in orofacial over time --
chronic orofacial - posterior is negative?). Reads tables/ from gain_bio.py; writes tables/S_*.csv and prints the
per-animal table FIRST (ground rule 8), then the epoch_change_lmm tests (stars: lmm_stars over each panel's family).

Quantities, per session, rest frames (R5 = restdock05), each area's SD / its animal's held-out pre:
  gain      median ratio over 46 split-basis areas
  oro, post mean log ratio over orofacial (ALM, tjM1, SSp-m, SSp-n, both hemis) / posterior (VIS*, RSP*) areas
  oro_post  oro - post (log); ipsi_contra as in gain_bio
Time trend: post-stroke sessions from day 7 on (after acute), value ~ day + (1 + day | animal) is too much for n = 4,
so each animal's OLS slope per 10 days, then a one-sample t on the 4 slopes (df 3).

PROVENANCE: ported 2026-11-05 from M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/gain_bio_1104/gain_stats.py
  (GAIN_BIO_2026-11-04.md section 5 "Stats"). Computation unchanged. Changes: tables/ = <out>/tables with <out> =
  gb_common.out_root() (--out; lp_licks.output_dir(<cohort_out>/gain_bio), run-tagged under the render profile)
  instead of the script's folder; the module body moved into main(argv) so it imports without running.
  Priya 11-06 (4), like with like: family A_gain = gb_common.GAIN_SIGS (corrected 470 HP, raw 470 HP, 415 HP; was
  unfiltered corr + 415 HP); the "415 rises about half as much" number = `ratio_415` (the 415 epoch beta over the raw
  470 epoch beta, same high-pass) -> tables/S_415_vs_470.csv; X_oro_post_robust adds raw 470 HP.

    PYTHONPATH=<repo>;<repo>/scripts python scripts/gain_bio/gain_stats.py [--out DIR]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_common  # noqa: E402

gb_common.add_paths()
from wfield_local import analysis_kit as ak  # noqa: E402

T = None                              # <out>/tables: set in main()
ORO = ("ALM", "tjM1", "SSp-m", "SSp-n")
STROKE = {"PS92": "0817", "PS93": "0817", "PS94": "0816", "PS95": "0816"}


def day_of(label):
    a, md = label.split("_")
    return (pd.Timestamp(f"2026-{md[:2]}-{md[2:]}") - pd.Timestamp(f"2026-{STROKE[a][:2]}-{STROKE[a][2:]}")).days


def per_session():
    U = pd.read_csv(T / "C1_uniformity_sessions.csv")
    A = pd.read_csv(T / "C1_area_ratio_sessions.csv")
    A = A[A.set == "R5"].copy()
    A["base"] = A.area.str.rsplit("_", n=1).str[0]
    A["lr"] = np.log(A.ratio)
    g = A.groupby(["label", "sig"])
    oro = A[A.base.isin(ORO)].groupby(["label", "sig"]).lr.mean().rename("oro")
    post = A[A.area.str.startswith(("VIS", "RSP"))].groupby(["label", "sig"]).lr.mean().rename("post")
    S = U[U.set == "R5"].set_index(["label", "sig"])[["animal", "epoch", "gain", "ipsi_contra", "oro_post"]]
    S = S.join(oro).join(post).reset_index()
    S["lgain"] = np.log(S.gain)
    S["day"] = S.label.map(day_of)
    assert np.allclose(S.oro - S.post, S.oro_post, atol=1e-6), "oro - post does not reproduce gain_bio's oro_post"
    del g
    return S


def lmm_family(S, sigs, qs):
    rows = []
    for sg in sigs:
        for q in qs:
            r = ak.epoch_change_lmm(S[S.sig == sg], value=q)
            r.insert(0, "quantity", q)
            r.insert(0, "sig", sg)
            rows.append(r)
    R = pd.concat(rows, ignore_index=True)
    R["holm"] = ak.holm(R.p.values)
    R["star"] = ak.lmm_stars(R.p.values)
    return R


def ratio_415(R: pd.DataFrame) -> pd.DataFrame:
    """Per epoch: the 415 epoch change (lgain beta, gb_common.NONCA) over the raw 470 one with the SAME processing
    (gb_common.CA_REF) -- "how large is the 415 rise relative to the calcium rise", like with like (Priya 11-06 (4)).
    R = an A_gain lmm_family table. Ratio of log changes; NaN when the 470 beta is 0 / missing."""
    b = R[R.quantity == "lgain"].pivot_table(index="epoch", columns="sig", values="beta", aggfunc="first")
    nan = np.full(len(b), np.nan)
    out = pd.DataFrame({"epoch": list(b.index),
                        "beta_415": b[gb_common.NONCA].to_numpy(float) if gb_common.NONCA in b else nan,
                        "beta_470": b[gb_common.CA_REF].to_numpy(float) if gb_common.CA_REF in b else nan})
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ratio_415_470"] = np.where(out.beta_470 != 0, out.beta_415 / out.beta_470, np.nan)
    order = {e: i for i, e in enumerate(("pre", "acute", "subacute", "chronic"))}
    return out.sort_values("epoch", key=lambda x: x.map(order)).reset_index(drop=True)


def slopes(S, sg, q, from_day=7):
    z = S[(S.sig == sg) & (S.day >= from_day)]
    out = {}
    for a, g in z.groupby("animal"):
        out[a] = np.polyfit(g.day / 10.0, g[q], 1)[0] if len(g) >= 3 else np.nan
    v = np.array([out[a] for a in sorted(out)])
    t = stats.ttest_1samp(v[np.isfinite(v)], 0.0)
    return out, float(np.nanmean(v)), float(t.pvalue)


def main(argv=None) -> int:
    global T
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="gain_bio output root (default gb_common.out_root())")
    gb_common.apply_args(ap.parse_args(argv))
    T = gb_common.guard(gb_common.out_root() / "tables")
    print("tables", T)
    S = per_session()
    S.to_csv(T / "S_sessions.csv", index=False)
    pd.set_option("display.width", 200)
    c = S[S.sig == "corr"]
    print("== PER ANIMAL x EPOCH (corrected 470, rest; animal mean over sessions; n sessions)")
    P = c.groupby(["animal", "epoch"]).agg(n=("gain", "size"), gain=("gain", "mean"), oro=("oro", "mean"),
                                           post=("post", "mean"), oro_post=("oro_post", "mean"),
                                           ipsi_contra=("ipsi_contra", "mean")).round(3)
    print(P.reindex(["pre", "acute", "subacute", "chronic"], level=1).to_string())

    fams = {
        "A_gain": lmm_family(S, list(gb_common.GAIN_SIGS), ["lgain"]),
        "D_space": lmm_family(S, ["corr"], ["ipsi_contra", "oro_post"]),
        "E_oro_post_parts": lmm_family(S, ["corr"], ["oro", "post"]),
        "X_oro_post_robust": lmm_family(S, ["corr_hp", "r470_hp", "r415_hp"], ["oro_post"]),
    }
    for k, R in fams.items():
        print(f"\n== {k}: epoch_change_lmm (beta = change vs pre; log units for lgain / oro / post); Holm within family")
        print(R[["sig", "quantity", "epoch", "beta", "lo", "hi", "p", "holm", "star", "n_animals", "converged",
                 "singular"]].round(4).to_string(index=False))
        R.to_csv(T / f"S_lmm_{k}.csv", index=False)
    H = ratio_415(fams["A_gain"])
    print(f"\n== 415 vs 470, like with like ({gb_common.NONCA} / {gb_common.CA_REF}, both raw, same 0.1 Hz high-pass): "
          "epoch change in log gain and its ratio")
    print(H.round(3).to_string(index=False))
    H.to_csv(T / "S_415_vs_470.csv", index=False)

    print("\n== TIME TREND, day >= 7: each animal's slope per 10 days; one-sample t on 4 slopes (df 3)")
    TR = []
    for sg, q in [("corr", "lgain"), ("corr", "oro"), ("corr", "post"), ("corr", "oro_post"),
                  ("corr_hp", "oro_post"), ("r415_hp", "oro_post"), ("r470_hp", "lgain"), ("r415_hp", "lgain")]:
        per, m, p = slopes(S, sg, q)
        TR.append(dict(sig=sg, quantity=q, mean_slope_per10d=m, p=p, **{a: per[a] for a in sorted(per)}))
    TR = pd.DataFrame(TR)
    print(TR.round(4).to_string(index=False))
    TR.to_csv(T / "S_trend.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
