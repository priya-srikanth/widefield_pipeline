"""Six-position cohort (PS92-95): manuscript statistics for HOW far-contralesional lick aim recovers (VLS paper, Priya 2026-11-21).

New manuscript endpoints, all read from existing pipeline tables (nothing upstream recomputed):
  within_lick_correction/sessions.csv  (scripts/within_lick_correction.py; cued, far_contra, all orders / outcomes)
      e_L        launch error (first 12 ms vs the animal's August-pre HIT angle at the spout; deg, + = contralesional)
      e_E        endpoint error (extension peak, same reference)
      corr       within-lick correction e_E - e_L
      csm_share  share of licks with >= 1 corrective submovement;  csm_false  its noise-surrogate control
  learned_compensation/t3_sessions.csv (scripts/learned_compensation.py T3 primary: DOCKED between-trial licks only)
      D_off      docked ITI landing vs the animal's pre mean after the SAME previous spout
      cued_mean  cued first-lick landing error at the same spouts;  GAP = D_off - cued_mean
  iti_licks/lick_correction/corr_beh_sessions.csv (scripts/lick_correction.py; all spouts, order any: step_b is NaN on
      first-lick rows, so 'any' is in effect later licks)
      step_b     extra correction toward the target on the lick after a MISS vs after a HIT at matched error
Test (the manuscript six-position rule): `orofacial_stats.session_change_tests` -- session change from each animal's
own pre mean, session MWU + animal-weighted paired nested bootstrap (`analysis_kit.boot_delta`); star = agree == both
(q < 0.05 AND CI excludes 0). FDR (Priya 2026-11-21, following the 2026-11-19 single-family decision): the endpoint
cells JOIN the 162-cell six-position scalar family (`c2_scalar_family_aligned.csv`) as ONE BH family; the primary q
(`q_fdr`) is that pooled family q. The noise-surrogate control (csm_false) and the cued reference (cued_mean) are
NOT family members: they are tested with a per-metric BH (`fdr_family = "control (outside family)"`). The pooled
family (162 + new cells, with the 162 cells' q re-computed) is written under a NEW name next to the 11-19 file; the
11-19 file is not touched. The pipeline's own primary results (nested bootstrap, Holm / >= 3 of 4 animals) are joined
for reference. Per-animal tables, with each animal's non-NaN pre and epoch session counts, are printed first (rule 8):
PS95 enters T3 (D_off / cued_mean / GAP) with ONE non-NaN pre session and no acute session, so T3 acute has 3 animals.
`n_animals` here counts animals with a non-NaN value in the epoch (orofacial_stats counts before dropna).
Inputs: the UNTAGGED v36 imaging_encoding tree (Oct 29-30), the same basis as the other six-position manuscript
tables; the rebuild-tagged (`_daqlk`) versions give the same stars (DECISIONS 2026-11-21). All inputs are 3D-kinematic
and must be regenerated when the 4-camera / 3-network 3D tracking replaces the two-view reconstruction.

Writes (refuses to overwrite) under --out-dir (default M:/MICROSCOPE/Priya/Widefield/manuscript_stats):
  c2_recovery_mechanics_v2_stats.csv       one row per endpoint x post epoch (pooled-family q; controls marked)
  c2_recovery_mechanics_v2_sessions.csv    per-session endpoint values (long: source, metric, label, animal, epoch, value)
  c2_recovery_mechanics_v2_per_animal.csv  per-animal epoch means, change from own pre, non-NaN session counts
and under --ms-data (default the paper's manuscript_code/data, next to the 11-19 family file):
  c2_scalar_family_aligned_with_recovery.csv   the pooled family: the 162 cells + the new endpoint cells, family q

Usage: python scripts/c2_recovery_mechanics_stats.py [--enc DIR] [--family CSV] [--out-dir DIR] [--ms-data DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from wfield_local import orofacial_stats as st  # noqa: E402
from wfield_local import writeguard  # noqa: E402

SRV = Path("M:/MICROSCOPE/Priya")
ENC = SRV / "Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding"
FAMILY = SRV / "stroke_orofacial_pipeline_outputs/vls_stroke_paper/manuscript_code/data/c2_scalar_family_aligned.csv"
OUT_DIR = SRV / "Widefield/manuscript_stats"
MS_DATA = FAMILY.parent
CONTROLS = ("csm_false", "cued_mean")       # tested, but not members of the BH family (Priya 2026-11-21)
EPOCHS = ("acute", "subacute", "chronic")
SOURCES = {
    "within_lick_correction": ("within_lick_correction/sessions.csv", ["e_L", "e_E", "corr", "csm_share", "csm_false"]),
    "learned_compensation_T3": ("learned_compensation/t3_sessions.csv", ["D_off", "cued_mean", "GAP"]),
    "lick_correction": ("iti_licks/lick_correction/corr_beh_sessions.csv", ["step_b"]),
}


def load(enc: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for src, (rel, _) in SOURCES.items():
        d = pd.read_csv(enc / rel)
        if src == "within_lick_correction":
            d = d[(d.context == "cued") & (d.group == "far_contra") & (d.order == "any") & (d.outcome == "any")]
            d = d.assign(pos="far_contra")
        elif src == "lick_correction":
            d = d[(d.group == "all") & (d.order == "any")].assign(pos="all")
        else:
            d = d.assign(pos="all")
        out[src] = d.assign(date=d.label).reset_index(drop=True)
    return out


def per_animal(D: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for src, (_, ms) in SOURCES.items():
        d = D[src]
        for m in ms:
            g = d.groupby(["animal", "epoch"])[m].mean().unstack("epoch")
            n = d.groupby(["animal", "epoch"])[m].count().unstack("epoch")   # non-NaN sessions (rule 8)
            for a, r in g.iterrows():
                for e in EPOCHS:
                    rows.append({"source": src, "metric": m, "animal": a, "epoch": e, "pre": r.get("pre"),
                                 "value": r.get(e), "change": r.get(e) - r.get("pre"),
                                 "n_pre": int(n.loc[a].get("pre", 0) or 0), "n_epoch": int(n.loc[a].get(e, 0) or 0)})
    return pd.DataFrame(rows)


def pipeline_ref(enc: Path) -> pd.DataFrame:
    """The pipeline's own primary results, for reference (its boot / Holm / n_same columns, renamed pipe_*)."""
    ref = []
    h = pd.read_csv(enc / "within_lick_correction/primary_holm.csv")
    ref.append(h.rename(columns={"delta": "pipe_delta", "lo": "pipe_lo", "hi": "pipe_hi", "p": "pipe_p",
                                 "p_holm": "pipe_p_holm"})[["metric", "epoch", "pipe_delta", "pipe_lo", "pipe_hi",
                                                            "pipe_p", "pipe_p_holm"]])
    t3 = pd.read_csv(enc / "learned_compensation/t3_boot.csv")
    t3 = t3[t3.variant == "docked, theta_E (primary)"]
    ref.append(t3.rename(columns={"delta": "pipe_delta", "lo": "pipe_lo", "hi": "pipe_hi", "p": "pipe_p",
                                  "p_holm": "pipe_p_holm", "n_same": "pipe_n_same"})[
        ["metric", "epoch", "pipe_delta", "pipe_lo", "pipe_hi", "pipe_p", "pipe_p_holm", "pipe_n_same"]])
    b = pd.read_csv(enc / "iti_licks/lick_correction/corr_beh_boot.csv")
    b = b[(b.group == "all") & (b.order == "any") & (b.metric == "step_b") & (b.epoch != "pre")]
    ref.append(b.rename(columns={"delta": "pipe_delta", "delta_lo": "pipe_lo", "delta_hi": "pipe_hi"})[
        ["metric", "epoch", "pipe_delta", "pipe_lo", "pipe_hi"]])
    return pd.concat(ref, ignore_index=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--enc", type=Path, default=ENC)
    ap.add_argument("--family", type=Path, default=FAMILY)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--ms-data", type=Path, default=MS_DATA)
    a = ap.parse_args(argv)
    base = a.out_dir / "c2_recovery_mechanics_v2"
    outs = [Path(f"{base}_{x}.csv") for x in ("stats", "sessions", "per_animal")]
    outs.append(a.ms_data / "c2_scalar_family_aligned_with_recovery.csv")
    clash = [str(p) for p in outs if p.exists()]
    if clash:
        sys.exit(f"refusing to overwrite {', '.join(clash)}")
    D = load(a.enc)
    PA = per_animal(D)
    print(PA.round(3).to_string(index=False))                       # rule 8: per animal first
    T = []
    for src, (_, ms) in SOURCES.items():
        r = st.session_change_tests(D[src], ms, epochs=EPOCHS, fdr_scope="metric")
        T.append(r.assign(source=src))
    T = pd.concat(T, ignore_index=True)
    S = pd.concat([D[src][["label", "animal", "epoch", "day", "pos"] + ms].melt(
        id_vars=["label", "animal", "epoch", "day", "pos"], var_name="metric").assign(source=src)
        for src, (_, ms) in SOURCES.items()], ignore_index=True)
    # animals with a non-NaN value in the epoch (orofacial_stats counts animals before dropna)
    na = S.dropna(subset=["value"]).groupby(["metric", "epoch"]).animal.nunique()
    T["n_animals"] = [int(na.get((m, e), 0)) for m, e in zip(T.metric, T.epoch)]
    is_ctl = T.metric.isin(CONTROLS)
    fam = pd.read_csv(a.family)
    assert len(fam) == 162, len(fam)
    cols = ["metric", "pos", "epoch", "p", "boot_lo", "boot_hi"]
    F = pd.concat([fam.assign(_new=0), T.loc[~is_ctl].assign(_new=1)], ignore_index=True)
    P = st.apply_family_fdr(F[cols + ["_new"]])                     # ONE family: 162 cells + new endpoint cells
    F["q_family_pooled"] = P.q_fdr.to_numpy()
    F["agree_pooled"] = P.agree.to_numpy()
    dq = (F.loc[F._new == 0, "q_family_pooled"] - F.loc[F._new == 0, "q_family"]).abs()
    flips = int(((F.loc[F._new == 0, "agree_pooled"] == "both") != (F.loc[F._new == 0, "star_aligned"] == 1)).sum())
    print(f"162 published cells: max |q shift| {dq.max():.4g}, star flips {flips}")
    new = F[F._new == 1]
    T.loc[~is_ctl, "q_fdr_metric"] = T.loc[~is_ctl, "q_fdr"].to_numpy()
    T.loc[~is_ctl, "q_fdr"] = new.q_family_pooled.to_numpy()
    T.loc[~is_ctl, "agree"] = new.agree_pooled.to_numpy()
    T["fdr_family"] = "six-position scalar family (162 + recovery)"
    T.loc[is_ctl, "fdr_family"] = "control (outside family; per-metric BH)"
    T["star"] = (T.agree == "both").astype(int)
    T = T.merge(pipeline_ref(a.enc), on=["metric", "epoch"], how="left", validate="many_to_one")
    for d in (a.out_dir, a.ms_data):
        writeguard.assert_writable(d)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.ms_data.mkdir(parents=True, exist_ok=True)
    T.to_csv(outs[0], index=False)
    S.to_csv(outs[1], index=False)
    PA.to_csv(outs[2], index=False)
    F.drop(columns=["_new"]).to_csv(outs[3], index=False)
    print(T[["metric", "epoch", "n_animals", "n_sessions", "boot_change", "boot_lo", "boot_hi", "q_fdr", "fdr_family",
             "star", "pipe_delta", "pipe_p_holm", "pipe_n_same"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
