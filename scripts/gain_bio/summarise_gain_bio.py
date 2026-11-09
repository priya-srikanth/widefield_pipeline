"""Summarize gain_bio.py (read only). Per animal x epoch FIRST (rule 8), then analysis_kit.boot_delta.

RATIO = session SD / the mean SD of the animal's pre sessions (a pre session: the mean of its OTHER pre sessions), per
area / pixel bin, per signal (corr, corr_hp, r470_hp, r415_hp) and frame set (ALL, R5, R5far, TASK). Animal x epoch
cells = geometric mean over sessions. Pixel maps: sqrt(variance ratio), same reference.
CHECK 1 (spatial):
  gain        median over areas of the area SD ratio (cortex-wide 'cortex' ROI also shown)
  spread      SD over areas of log SD ratio (uniform gain -> same spread as held-out pre)
  ipsi-contra mean over paired areas of log ratio (left = ipsi, image left) minus right
  oro-post    mean log ratio over orofacial (ALM, tjM1, SSp-m, SSp-n, both hemis) minus posterior (VIS*, RSP*)
  415 / 470   r415_hp ratio beside r470_hp ratio: like with like (both halves of raw SVT, same 0.1 Hz high-pass, same
              frames; Priya 11-06 (4)); corr_hp / r470_hp = the hemo-correction check (both high-passed). Until 11-06
              this line set r415_hp against corr_hp, and the 11-04 '415 about half' text against unfiltered corr.
  r_F0        across areas within a session: r(log SD ratio, log F0 ratio) (470 F0 = frames_average_atlas); also
              across pixel bins of the maps
  r_events    across the 12 quiet-event areas (ALM, tjM1, SSp-m, SSp-n, SSp-ul, SSp-ll x L/R): r(log R5 SD ratio,
              log quiet-event rate ratio), rate = v2 'rate_both' (quiet_events_sessions.csv)
CHECK 2 (movement / arousal at rest, R5 chunks):
  speed bins  face speed / the animal's pre median R5 face speed: < .25, .25-.5, .5-1, 1-2, > 2
  matched     per session, variance over each bin's chunks (about the bin mean); speed-matched variance = sum over
              bins of the animal's PRE bin weights x the bin variance (weights renormalized over the bins present);
              the same for pre sessions -> matched SD ratio
  still       chunks with face speed < 0.5 x pre median only
  still_far   still AND > 2 s from a quiet event (R5far chunks)
  session reg per animal, across sessions: log SD ratio ~ epoch dummies (+ log face-speed ratio); epoch coefficients
              with and without the covariate
  face resid  out-of-fold face-regression residual SD ratio vs raw (corr_hp, R5)

PROVENANCE: ported 2026-11-05 from M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/gain_bio_1104/
  summarise_gain_bio.py (docs/status/GAIN_BIO_2026-11-04.md sections 3-4 / 7). Computation, tables, figures, layout
  and styling unchanged. Changes: reads <out>/sessions/*.npz and <out>/sessions.csv (gain_bio.py's outputs) and
  writes <out>/tables, <out>/figs, with <out> = gb_common.out_root() (--out;
  lp_licks.output_dir(<cohort_out>/gain_bio), run-tagged under the render profile) instead of the script's folder; the quiet-event rates come from gb_common.quiet_dir() /
  quiet_events_sessions.csv (--quiet-dir; THIS run's quiet_spikes_v2 output) instead of the adopted v2_quiet; no repo
  path in code.
  Priya 11-06 (4): every 415 comparison is like with like (`ratio_table`, gb_common.NONCA / CA_REF); the pixel-map figures
  1 / 2 show raw 470 and 415 with the same high-pass (were corrected 470 unfiltered beside 415 high-passed).

    PYTHONPATH=<repo>;<repo>/scripts python scripts/gain_bio/summarise_gain_bio.py [--out DIR] [--quiet-dir DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_common  # noqa: E402

gb_common.add_paths()
TAB = FIG = None                     # <out>/tables, <out>/figs: set by set_dirs() (main)
AN = ("PS92", "PS93", "PS94", "PS95")
EPS = ("pre", "acute", "subacute", "chronic")
POST = EPS[1:]
SIGS = ("corr", "corr_hp", "r470_hp", "r415_hp")
SETS = ("ALL", "R5", "R5far", "TASK")
ORO = ("ALM", "tjM1", "SSp-m", "SSp-n")
EVA = ("ALM", "tjM1", "SSp-m", "SSp-n", "SSp-ul", "SSp-ll")
SBINS = (0.0, 0.25, 0.5, 1.0, 2.0, np.inf)


def still_text() -> str:
    """The CHECK 2 "still" rule in words (`check2`: chunks in face-speed bins 0-1, i.e. below SBINS[2])."""
    import gain_bio as gb
    return (f"R5 chunks (<= {gb.CHUNK} frames inside one R5 run) with mean face speed < {SBINS[2]:g} x the animal's pre "
            "median R5 face speed")


def map_frame_set(tag: str) -> str:
    """The frame set of a `fig_maps` figure: tag "R5" / "TASK" = that set; "F0" = the R5 SD map beside the F0 ratio
    (F0 = frames_average_atlas, every frame)."""
    import gain_bio as gb
    if tag == "F0":
        return gb.frame_set_sentence("R5", extra=["F0 panel: the session's frames_average_atlas (every frame)"])
    return gb.frame_set_sentence(tag)


def ratio_table(W: pd.DataFrame) -> pd.DataFrame:
    """Gain table (rows animal x epoch, one column per signal) -> + the like-with-like ratio columns (Priya 11-06 (4)):
    "415/470" = NONCA / CA_REF (raw 415 vs raw 470, same high-pass) and "corr/470" = CORR_CHECK (corrected vs raw 470,
    both high-passed). Nothing sets 415 against the corrected or the unfiltered signal."""
    W = W.copy()
    W["415/470"] = W[gb_common.NONCA] / W[gb_common.CA_REF]
    W["corr/470"] = W[gb_common.CORR_CHECK[0]] / W[gb_common.CORR_CHECK[1]]
    return W


def QEV() -> Path:
    return gb_common.quiet_dir() / "quiet_events_sessions.csv"


def set_dirs() -> Path:
    global TAB, FIG
    out = gb_common.out_root()
    TAB, FIG = gb_common.guard(out / "tables"), gb_common.guard(out / "figs")
    return out


def load():
    S = gb_common.session_table()[["label", "animal", "epoch"]]
    D = {}
    for lab in S.label:
        f = gb_common.out_root() / "sessions" / f"{lab}.npz"
        if f.exists():
            D[lab] = dict(np.load(f, allow_pickle=True))
    S = S[S.label.isin(D)].sort_values("label").reset_index(drop=True)
    areas = sorted({str(a) for d in D.values() for a in d["areas"]} - {"cortex"}) + ["cortex"]
    return S, D, areas


def area_vec(d, key, areas):
    m = dict(zip([str(a) for a in d["areas"]], d[key]))
    return np.array([m.get(a, np.nan) for a in areas], float)


def rel(S, vals):
    """vals: {label: array} -> {label: array / mean of the animal's (other) pre sessions}."""
    out = {}
    for an, g in S.groupby("animal"):  # noqa: B007
        pre = list(g[g.epoch == "pre"].label)
        for lab in g.label:
            ref = [x for x in pre if x != lab]
            out[lab] = vals[lab] / np.nanmean([vals[x] for x in ref], 0)
    return out


def ae_mean_log(S, R, keys=None):
    """{(animal, epoch): geometric mean over sessions} of the arrays R[label]."""
    out = {}
    for (an, ep), g in S.groupby(["animal", "epoch"]):
        out[(an, ep)] = np.exp(np.nanmean([np.log(R[x]) for x in g.label], 0))
    return out


def boot(S, col, df, log=False):
    from wfield_local import analysis_kit as ak
    rows = []
    for ep in POST:
        post = {a: g[col].dropna().tolist() for a, g in df[df.epoch == ep].groupby("animal")}
        pre = {a: g[col].dropna().tolist() for a, g in df[df.epoch == "pre"].groupby("animal")}
        iv = ak.boot_delta(post, pre, np.random.default_rng(0))
        up = sum(np.mean(post[a]) > np.mean(pre[a]) for a in post if a in pre and post[a] and pre[a])
        rows.append(dict(quantity=col, epoch=ep, delta=iv.point if iv else np.nan, lo=iv.lo if iv else np.nan,
                         hi=iv.hi if iv else np.nan, n_up=up, n_animals=iv.n_animals if iv else 0))
    return rows


def fmt_boot(B):
    B = B.copy()
    B["txt"] = [f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if (lo > 0 or hi < 0) else ''} {u}/{n}"
                for d, lo, hi, u, n in zip(B.delta, B.lo, B.hi, B.n_up, B.n_animals)]
    return B.pivot(index="quantity", columns="epoch", values="txt").reindex(columns=list(POST))


def check1(S, D, areas):
    ia = [i for i, a in enumerate(areas) if a != "cortex"]
    side = {a: a.rsplit("_", 1) for a in areas if a != "cortex"}
    pairs = sorted({b for b, h in side.values() if f"{b}_left" in areas and f"{b}_right" in areas})
    rows, area_rows = [], []
    RA = {}
    for st in SETS:
        for sg in SIGS:
            vals = {lab: area_vec(D[lab], f"{st}_{sg}_sd", areas) for lab in S.label}
            R = rel(S, vals)
            RA[(st, sg)] = R
            for lab in S.label:
                r = R[lab]
                lr = np.log(r)
                row = dict(label=lab, set=st, sig=sg, gain=float(np.nanmedian(r[ia])),
                           cortex=float(r[areas.index("cortex")]), spread=float(np.nanstd(lr[ia])),
                           ipsi_contra=float(np.nanmean([lr[areas.index(f"{b}_left")] - lr[areas.index(f"{b}_right")]
                                                          for b in pairs])),
                           oro_ipsi_contra=float(np.nanmean([lr[areas.index(f"{b}_left")] - lr[areas.index(f"{b}_right")]
                                                              for b in ORO if b in pairs])),
                           oro_post=float(np.nanmean([lr[i] for i, a in enumerate(areas) if a.rsplit("_", 1)[0] in ORO])
                                          - np.nanmean([lr[i] for i, a in enumerate(areas)
                                                        if a.startswith(("VIS", "RSP"))])))
                rows.append(row)
                for i, a in enumerate(areas):
                    area_rows.append(dict(label=lab, set=st, sig=sg, area=a, ratio=r[i]))
    C = pd.DataFrame(rows).merge(S, on="label")
    A = pd.DataFrame(area_rows).merge(S, on="label")
    C.to_csv(TAB / "C1_uniformity_sessions.csv", index=False)
    A.to_csv(TAB / "C1_area_ratio_sessions.csv", index=False)
    return C, A, RA


def maps(S, D):
    """{(set, sig): {label: sqrt(var map / pre mean var map)}} and F0 ratio maps."""
    M = {}
    for st in ("R5", "TASK", "R5far", "ALL"):
        for sg in SIGS:
            v = {lab: D[lab][f"{st}_{sg}_map"].astype(float) for lab in S.label}
            r = rel(S, v)
            M[(st, sg)] = {k: np.sqrt(x) for k, x in r.items()}
    for ch in ("470", "415"):
        M[("F0", ch)] = rel(S, {lab: D[lab][f"f0_{ch}_map"].astype(float) for lab in S.label})
    return M


def event_rates(S):
    Q = pd.read_csv(QEV())
    cols = [f"{a}_{h}" for a in EVA for h in ("left", "right")]
    vals = {}
    for lab in S.label:
        r = Q[Q.label == lab]
        vals[lab] = r[[f"{c}|rate_both" for c in cols]].to_numpy(float)[0] if len(r) else np.full(len(cols), np.nan)
    return cols, rel(S, vals)


def spatial_corr(S, D, areas, RA, M):
    ecols, ER = event_rates(S)
    F470 = rel(S, {lab: area_vec(D[lab], "f0_470_area", areas) for lab in S.label})
    F415 = rel(S, {lab: area_vec(D[lab], "f0_415_area", areas) for lab in S.label})
    ia = [i for i, a in enumerate(areas) if a != "cortex"]
    ie = [areas.index(c) if c in areas else None for c in ecols]
    rows = []
    for lab in S.label:
        row = dict(label=lab)
        for st in ("R5", "TASK"):
            for sg in ("corr", "r415_hp"):
                y = np.log(RA[(st, sg)][lab])
                for nm, F in (("f470", F470), ("f415", F415)):
                    x = np.log(F[lab])
                    ok = np.isfinite(x[ia]) & np.isfinite(y[ia])
                    row[f"r_{nm}_{st}_{sg}"] = float(np.corrcoef(x[ia][ok], y[ia][ok])[0, 1]) if ok.sum() > 5 else np.nan
                ev = np.log(ER[lab])
                yy = np.array([y[i] if i is not None else np.nan for i in ie])
                ok = np.isfinite(ev) & np.isfinite(yy)
                row[f"r_events_{st}_{sg}"] = float(np.corrcoef(ev[ok], yy[ok])[0, 1]) if ok.sum() > 5 else np.nan
                # pixel-bin map correlation with F0 470
                mv, fv = np.log(M[(st, sg)][lab]).ravel(), np.log(M[("F0", "470")][lab]).ravel()
                ok = np.isfinite(mv) & np.isfinite(fv)
                row[f"rpix_f470_{st}_{sg}"] = float(np.corrcoef(mv[ok], fv[ok])[0, 1]) if ok.sum() > 50 else np.nan
        row["ev_rate_ratio_med"] = float(np.nanmedian(ER[lab]))
        rows.append(row)
    R = pd.DataFrame(rows).merge(S, on="label")
    R.to_csv(TAB / "C1_spatial_corr_sessions.csv", index=False)
    return R


# ------------------------------------------------------------------------------------------------- check 2
def chunk_table(S, D, areas):
    """Per session x speed bin x {all, far} x signal: pooled variance per area (about the bin mean) and frames."""
    pre_face = {}
    for an, g in S.groupby("animal"):
        pre_face[an] = float(np.nanmedian([D[x]["R5_face_med"] for x in g[g.epoch == "pre"].label]))
    out = {}
    for lab in S.label:
        d = D[lab]
        n = np.diff(d["ch_se"], axis=1).ravel().astype(float)
        sp = d["ch_face"] / pre_face[lab[:4]]
        b = np.digitize(sp, SBINS) - 1
        b[~np.isfinite(sp)] = -1
        far = d["ch_far"].astype(bool)
        rec = {"n": n, "bin": b, "far": far}
        for sg in SIGS:
            rec[sg] = (area_vec_rows(d, f"ch_{sg}_S1", areas), area_vec_rows(d, f"ch_{sg}_S2", areas))
        out[lab] = rec
    return out, pre_face


def area_vec_rows(d, key, areas):
    nm = [str(a) for a in d["areas"]]
    X = np.full((len(d[key]), len(areas)), np.nan)
    for j, a in enumerate(nm):
        X[:, areas.index(a)] = d[key][:, j]
    return X


def pooled_var(rec, sg, sel):
    n = rec["n"][sel].sum()
    if n < 100:
        return np.full(rec[sg][0].shape[1], np.nan), n
    S1, S2 = rec[sg][0][sel].sum(0), rec[sg][1][sel].sum(0)
    return S2 / n - (S1 / n) ** 2, n


def check2(S, D, areas):
    CH, pre_face = chunk_table(S, D, areas)
    nb = len(SBINS) - 1
    # animal pre bin weights (frames)
    wpre = {}
    for an, g in S.groupby("animal"):
        w = np.zeros(nb)
        for lab in g[g.epoch == "pre"].label:
            rec = CH[lab]
            for k in range(nb):
                w[k] += rec["n"][rec["bin"] == k].sum()
        wpre[an] = w / w.sum()
    vals = {}          # (measure, sig) -> {label: area array of variance}
    occ = []
    for lab in S.label:
        rec, an = CH[lab], lab[:4]
        tot = rec["n"].sum()
        occ.append(dict(label=lab, **{f"frac_bin{k}": rec["n"][rec["bin"] == k].sum() / tot for k in range(nb)},
                        frac_nan=rec["n"][rec["bin"] < 0].sum() / tot, face_rel=D[lab]["R5_face_med"] / pre_face[an],
                        run_p95=D[lab]["R5_run_p95"], R5_n=D[lab]["R5_n"]))
        for sg in SIGS:
            allv, _ = pooled_var(rec, sg, np.ones(len(rec["n"]), bool))
            vals.setdefault(("R5_chunks", sg), {})[lab] = allv
            bv = [pooled_var(rec, sg, rec["bin"] == k) for k in range(nb)]
            for k in range(nb):
                vals.setdefault((f"bin{k}", sg), {})[lab] = bv[k][0]
            ok = np.array([n >= 100 for _, n in bv])
            w = wpre[an] * ok
            vals.setdefault(("matched", sg), {})[lab] = (np.nansum([w[k] * bv[k][0] for k in range(nb) if ok[k]], 0)
                                                        / w.sum()) if w.sum() > 0 else np.full(len(areas), np.nan)
            still = (rec["bin"] >= 0) & (rec["bin"] <= 1)
            vals.setdefault(("still", sg), {})[lab] = pooled_var(rec, sg, still)[0]
            vals.setdefault(("still_far", sg), {})[lab] = pooled_var(rec, sg, still & rec["far"])[0]
            vals.setdefault(("far", sg), {})[lab] = pooled_var(rec, sg, rec["far"])[0]
            vals.setdefault(("near", sg), {})[lab] = pooled_var(rec, sg, ~rec["far"])[0]
    O = pd.DataFrame(occ).merge(S, on="label")
    O.to_csv(TAB / "C2_speed_occupancy_sessions.csv", index=False)
    ia = [i for i, a in enumerate(areas) if a != "cortex"]
    ic = areas.index("cortex")
    rows = []
    for (ms, sg), v in vals.items():
        R = rel(S, {k: np.sqrt(x) for k, x in v.items()})
        for lab in S.label:
            rows.append(dict(label=lab, measure=ms, sig=sg, gain=float(np.nanmedian(R[lab][ia])),
                             cortex=float(R[lab][ic]), oro=float(np.exp(np.nanmean(
                                 [np.log(R[lab][i]) for i, a in enumerate(areas) if a.rsplit("_", 1)[0] in ORO])))))
    # face-regression residual
    for key, ms in (("fr_raw_sd", "fr_raw"), ("fr_res_sd", "fr_resid")):
        v = {lab: area_vec(D[lab], key, areas) if key in D[lab] else np.full(len(areas), np.nan) for lab in S.label}
        R = rel(S, v)
        for lab in S.label:
            rows.append(dict(label=lab, measure=ms, sig="corr_hp", gain=float(np.nanmedian(R[lab][ia])),
                             cortex=float(R[lab][ic]), oro=float(np.exp(np.nanmean(
                                 [np.log(R[lab][i]) for i, a in enumerate(areas) if a.rsplit("_", 1)[0] in ORO])))))
    for lab in S.label:
        r2 = area_vec(D[lab], "fr_r2", areas) if "fr_r2" in D[lab] else np.full(len(areas), np.nan)
        rows.append(dict(label=lab, measure="fr_r2", sig="corr_hp", gain=float(np.nanmedian(r2[ia])),
                         cortex=float(r2[ic]), oro=np.nan))
    T = pd.DataFrame(rows).merge(S, on="label")
    T.to_csv(TAB / "C2_matched_sessions.csv", index=False)
    return T, O


def session_regression(S, C, O):
    """Per animal: log R5 SD ratio (corr, median over areas) ~ epoch dummies [+ log face ratio]."""
    X = C[(C.set == "R5") & (C.sig == "corr")][["label", "gain", "cortex"]].merge(O[["label", "face_rel", "animal",
                                                                                     "epoch"]], on="label")
    rows = []
    for an, g in X.groupby("animal"):
        g = g.dropna(subset=["gain", "face_rel"])
        y = np.log(g.gain.to_numpy())
        E = np.column_stack([np.ones(len(g))] + [(g.epoch == e).to_numpy(float) for e in POST])
        keep = E[:, 1:].sum(0) > 0
        E = np.column_stack([E[:, 0], E[:, 1:][:, keep]])
        eps = [e for e, k in zip(POST, keep) if k]
        b0 = np.linalg.lstsq(E, y, rcond=None)[0]
        E1 = np.column_stack([E, np.log(g.face_rel.to_numpy())])
        b1 = np.linalg.lstsq(E1, y, rcond=None)[0]
        pre = g.epoch == "pre"
        r_pre = float(np.corrcoef(np.log(g.face_rel[pre]), y[pre])[0, 1]) if pre.sum() > 3 else np.nan
        for j, e in enumerate(eps):
            rows.append(dict(animal=an, epoch=e, gain_unadj=float(np.exp(b0[1 + j])),
                             gain_adj_face=float(np.exp(b1[1 + j])), face_slope=float(b1[-1]),
                             r_face_sd_pre=r_pre, n=len(g)))
    R = pd.DataFrame(rows)
    R.to_csv(TAB / "C2_session_regression_animals.csv", index=False)
    return R


# ------------------------------------------------------------------------------------------------- figures
def fig_maps(S, M, tag, keys, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    bm = None
    try:
        from wfield_local import beta_maps
        bm = beta_maps.brain_mask()  # noqa: F841 -- (unused in the original too)
    except Exception:
        pass
    rows = list(AN) + ["animal mean"]
    fig, axs = plt.subplots(len(rows), len(keys) * 3, figsize=(2.0 * 3 * len(keys), 1.75 * len(rows)),
                            squeeze=False)
    norm = LogNorm(0.6, 1.0 / 0.6)
    for kk, (key, klab) in enumerate(keys):
        R = M[key]
        cells = {}
        for (an, ep), g in S.groupby(["animal", "epoch"]):
            cells[(an, ep)] = np.exp(np.nanmean([np.log(R[x]) for x in g.label], 0))
        for j, ep in enumerate(POST):
            have = [cells[(a, ep)] for a in AN if (a, ep) in cells]
            cells[("animal mean", ep)] = np.exp(np.nanmean(np.log(have), 0))
            for i, an in enumerate(rows):
                ax = axs[i, kk * 3 + j]
                ax.set_xticks([]), ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if (an, ep) not in cells:
                    continue
                im = ax.imshow(cells[(an, ep)], cmap="RdBu_r", norm=norm, interpolation="nearest")
                n = int((S[(S.animal == an) & (S.epoch == ep)]).shape[0]) if an in AN else None
                if i == 0:
                    ax.set_title(f"{klab}\n{ep}", fontsize=8)
                if j == 0 and kk == 0:
                    ax.set_ylabel(an, fontsize=8)
                if n is not None:
                    ax.text(2, 8, f"n={n}", fontsize=6)
    cb = fig.colorbar(im, ax=axs, shrink=0.5, pad=0.01)
    cb.set_label("SD ratio vs animal's pre (log scale)", fontsize=8)
    cb.set_ticks([0.6, 0.8, 1.0, 1.25, 1.6])
    cb.set_ticklabels(["0.6", "0.8", "1.0", "1.25", "1.6"])
    fig.suptitle(title + "  (image left = ipsilesional)", fontsize=9)
    from wfield_local import trial_set_text as tst
    with tst.active(None, text=map_frame_set(tag)):          # THE FRAME SET of this map figure (Priya 11-09)
        fig.savefig(FIG / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    gb_common.common_args(ap)
    gb_common.apply_args(ap.parse_args(argv))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    print("out", set_dirs(), "| quiet", QEV())
    S, D, areas = load()
    print(f"sessions {len(S)}; per animal x epoch:\n{S.groupby(['animal', 'epoch']).size().unstack()[list(EPS)]}")
    C, A, RA = check1(S, D, areas)
    # ---- per animal x epoch: gain / uniformity
    q = ["gain", "cortex", "spread", "ipsi_contra", "oro_ipsi_contra", "oro_post"]
    G = C.groupby(["set", "sig", "animal", "epoch"])[q].agg(
        lambda x: float(np.exp(np.nanmean(np.log(x)))) if x.name in ("gain", "cortex") else float(np.nanmean(x)))
    G = G.reset_index()
    G["epoch"] = pd.Categorical(G.epoch, EPS, ordered=True)
    G = G.sort_values(["set", "sig", "animal", "epoch"])
    G.to_csv(TAB / "C1_uniformity_animal_epoch.csv", index=False)
    for st in ("R5", "TASK", "R5far", "ALL"):
        W = G[G.set == st].pivot_table(index=["animal", "epoch"], columns="sig", values="gain", observed=True)
        W = ratio_table(W[list(SIGS)])
        print(f"\n== CHECK 1 [{st}] PER ANIMAL x EPOCH: SD ratio vs pre (median over areas; geometric mean over sessions)")
        print(W.round(3).to_string())
    for st in ("R5", "TASK"):
        print(f"\n== CHECK 1 [{st}, corr] uniformity per animal x epoch (spread = SD over areas of log ratio; "
              f"ipsi_contra / oro_post = log differences)")
        print(G[(G.set == st) & (G.sig == "corr")].drop(columns=["set", "sig"]).round(3).to_string(index=False))
    B = []
    for st in SETS:
        for sg in SIGS:
            sub = C[(C.set == st) & (C.sig == sg)]
            for col in q:
                for r in boot(S, col, sub):
                    B.append(dict(set=st, sig=sg, **r))
    B = pd.DataFrame(B)
    B.to_csv(TAB / "C1_uniformity_pooled.csv", index=False)
    for st in ("R5", "TASK"):
        for sg in ("corr", "corr_hp", "r470_hp", "r415_hp"):
            print(f"\n== CHECK 1 POOLED [{st}, {sg}] boot_delta (post - pre; * CI excludes 0; k/n animals up)")
            print(fmt_boot(B[(B.set == st) & (B.sig == sg)]).to_string())
    # ---- area table (animal-epoch geometric means, plus animal mean)
    AE = A.groupby(["set", "sig", "area", "animal", "epoch"]).ratio.apply(
        lambda x: float(np.exp(np.nanmean(np.log(x))))).reset_index()
    AE.to_csv(TAB / "C1_area_ratio_animal_epoch.csv", index=False)
    AM = AE[AE.epoch != "pre"].groupby(["set", "sig", "area", "epoch"]).ratio.agg(
        mean=lambda x: float(np.exp(np.nanmean(np.log(x)))), n_up=lambda x: int((x > 1).sum()), n="count").reset_index()
    AM.to_csv(TAB / "C1_area_ratio_animal_mean.csv", index=False)
    for st in ("R5", "TASK"):
        pair = [gb_common.CA_REF, gb_common.NONCA]                  # like with like (Priya 11-06 (4))
        P = AM[(AM.set == st) & AM.sig.isin(pair)].copy()
        P["txt"] = [f"{m:.2f} ({u}/{n})" for m, u, n in zip(P["mean"], P.n_up, P.n)]
        P = P.pivot_table(index="area", columns=["sig", "epoch"], values="txt", aggfunc="first")
        P = P.reindex(columns=pd.MultiIndex.from_product([pair, list(POST)]))
        print(f"\n== CHECK 1 AREA TABLE [{st}]: animal-mean SD ratio (animals > 1 / n)")
        print(P.to_string())
    # ---- spatial correlations
    M = maps(S, D)
    R = spatial_corr(S, D, areas, RA, M)
    rc = [c for c in R.columns if c.startswith(("r_", "rpix_"))] + ["ev_rate_ratio_med"]
    RG = R.groupby(["animal", "epoch"])[rc].mean().reindex(pd.MultiIndex.from_product([AN, EPS]))
    RG.reset_index().to_csv(TAB / "C1_spatial_corr_animal_epoch.csv", index=False)
    show = ["r_f470_R5_corr", "rpix_f470_R5_corr", "r_f470_TASK_corr", "r_events_R5_corr", "r_events_R5_r415_hp",
            "r_f415_R5_r415_hp", "ev_rate_ratio_med"]
    print("\n== CHECK 1 SPATIAL CORRELATION per animal x epoch (pre rows = held-out pre: the null)")
    print(RG[show].round(3).to_string())
    BR = []
    for col in show:
        BR += boot(S, col, R)
    BR = pd.DataFrame(BR)
    BR.to_csv(TAB / "C1_spatial_corr_pooled.csv", index=False)
    print("\n== CHECK 1 SPATIAL CORRELATION pooled (post - held-out pre)\n" + fmt_boot(BR).to_string())
    # figures
    # like with like (Priya 11-06 (4)): raw 470 and 415, the two halves of raw SVT, the same 0.1 Hz high-pass
    pair = [(gb_common.CA_REF, "raw 470 (0.1 Hz HP)"), (gb_common.NONCA, "415 (0.1 Hz HP)")]
    fig_maps(S, M, "R5", [(("R5", k), t) for k, t in pair],
             "fig1_rest_gain_maps.png", "Fig 1. REST (restdock05, engaged): SD ratio post / pre per pixel")
    fig_maps(S, M, "TASK", [(("TASK", k), t) for k, t in pair],
             "fig2_task_gain_maps.png", "Fig 2. TASK (cue..+2 s, camera working period, engaged): SD ratio post / pre "
             "per pixel")
    fig_maps(S, M, "F0", [(("R5", "corr_hp"), "corr 470 HP (rest)"), (("F0", "470"), "F0 470 ratio")],
             "fig3_rest_hp_vs_f0_maps.png", "Fig 3. REST corr_hp SD ratio beside the F0 (470) ratio map")
    # ---- check 2
    T, O = check2(S, D, areas)
    print("\n== CHECK 2 OCCUPANCY per animal x epoch: fraction of R5 frames per face-speed bin (x pre median), "
          "face_rel = R5 median face speed / pre, run_p95 (mm/s)")
    oc = [c for c in O.columns if c.startswith("frac_")] + ["face_rel", "run_p95", "R5_n"]
    print(O.groupby(["animal", "epoch"])[oc].mean().reindex(pd.MultiIndex.from_product([AN, EPS])).round(3)
          .to_string())
    for sg in ("corr", gb_common.CA_REF, gb_common.NONCA):
        sub = T[(T.sig == sg) | (T.measure.isin(["fr_raw", "fr_resid"]) & (sg == "corr"))]
        W = sub.pivot_table(index=["animal", "epoch"], columns="measure", values="gain",
                            aggfunc=lambda x: float(np.exp(np.nanmean(np.log(x)))))
        cols = [c for c in ("R5_chunks", "matched", "still", "still_far", "far", "near", "bin0", "bin1", "bin2",
                            "bin3", "bin4", "fr_raw", "fr_resid") if c in W.columns]
        W = W[cols].reindex(pd.MultiIndex.from_product([AN, EPS]))
        print(f"\n== CHECK 2 [{sg}] PER ANIMAL x EPOCH: R5 SD ratio vs pre (median over areas), overall / speed-matched"
              f" / still (< .5 x pre median face speed) / still & far from events / per speed bin"
              + (" / face-regression raw vs residual (corr_hp)" if sg == "corr" else ""))
        print(W.round(3).to_string())
    W = T[(T.measure == "fr_r2")].groupby(["animal", "epoch"])[["gain", "cortex"]].mean()
    print("\n== CHECK 2 face-model R2 (out of fold, corr_hp, R5): median over areas / cortex")
    print(W.reindex(pd.MultiIndex.from_product([AN, EPS])).round(3).to_string())
    B2 = []
    for (ms, sg), sub in T.groupby(["measure", "sig"]):
        for col in ("gain", "cortex", "oro"):
            for r in boot(S, col, sub):
                B2.append(dict(measure=ms, sig=sg, **r))
    B2 = pd.DataFrame(B2)
    B2.to_csv(TAB / "C2_matched_pooled.csv", index=False)
    for sg in ("corr", "corr_hp", "r470_hp", "r415_hp"):
        sub = B2[(B2.sig == sg) & (B2.quantity == "gain")].copy()
        sub["quantity"] = sub.measure
        print(f"\n== CHECK 2 POOLED [{sg}] gain (median over areas) boot_delta\n" + fmt_boot(sub).to_string())
    RR = session_regression(S, C, O)
    print("\n== CHECK 2 SESSION REGRESSION per animal: R5 corr gain, epoch effect without / with log face-speed "
          "covariate")
    print(RR.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
