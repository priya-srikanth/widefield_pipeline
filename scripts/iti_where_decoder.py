"""WHERE did the lick go: the frozen decoder of page 3.8 / 3.9 for lateral AND outward position (Priya 10-26: "should
we look at angle and distance together? ... or lateral location instead of angle if better").

Targets per lick (3-D tip at the lick peak, mouth-relative, `pose3d/kinematics_all/licks.csv.gz` joined on the session's
licks_all.csv row = lick_uid via the exact DAQ peak time stored by `iti_licks.py`):
  lateral   side_ml (mm; the page-3.9 target)
  outward   out_ap (mm; separates the close from the far spout ring)
  2-D       lateral + outward together: one ridge per axis, scored as a VECTOR correlation of predicted vs actual
            positions after scaling each axis by the animal's pre-stroke SD (1 = perfect, 0 = none)
  angle / reach   polar cross-check: atan2(side_ml, out_ap) (deg) and reach_mm
Same features and decoder as page 3.9 (per animal ridge trained on AUGUST pre licks, pre leave-one-session-out, lags
0..0.2 s averaged), in each feature basis available: Allen areas (`<E>/iti_licks/<SESS_NAME>`; WITH the ALM / tjM1 split by
default since 10-27 late -- `area_basis` / iti_licks.area_features; WFIELD_AREA_BASIS=plain = the earlier set, outputs
tagged _plain), joint LocaNMF and the
pre-stroke SVD basis (`iti_basis_decoder.py` outputs, when present). Reach spread per epoch is printed too (r is
blind to a mean shift, not to a change in spread).
Outputs <E>/iti_licks/where_decoder.csv, where_decoder_tests.csv, figures/lick_where_decoder.png (Allen areas, every
target) + lick_where_decoder_bases.png (lateral, three bases); split 10-26 so the panels are readable.
Usage: python scripts/iti_where_decoder.py [--plot-only | --z]   (--plot-only redraws from the saved CSVs; --z draws
lick_where_decoder_z.png + zscore_check/where_decoder_tests_z.csv from the z-scored run of iti_zscore_check.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
TARGETS = ("lateral", "sagittal", "2-D", "angle", "reach")         # outward is also scored (where_decoder.csv)
SOURCES = ("areas", "locanmfM", "svdbasis")
# LocaNMF robustness row = MASKED joint LocaNMF (Priya 10-27 late "swap masked locanmf"): components with > half their
# |footprint| outside the glue / olfactory-bulb brain mask dropped (locanmf_keep); same basis files as "locanmf"
# "areas" = Allen areas in the AREA BASIS (il.AREA_BASIS: split by default, Priya 10-27 late "use split bases, for
# consistency"; plain with WFIELD_AREA_BASIS=plain -> every output tagged il.BASIS_TAG). "areas_split" / "areas_plain"
# name a basis explicitly (iti_lick_order SOURCE, the basis summary). The source column of where_decoder*.csv stays
# "areas", so its readers (mixed models, recovery link, z check) follow the basis of the run.
LAB = {"areas": il.basis_label(), "locanmf": "joint LocaNMF (unmasked)",
       "locanmfM": "joint LocaNMF, glue / MOB comps dropped", "svdbasis": "pre-stroke SVD basis",
       "areas_split": il.basis_label("split"), "areas_plain": il.basis_label("plain")}
AREA_SOURCES = {"areas": None, "areas_split": "split", "areas_plain": "plain"}     # source -> basis (None = il.AREA_BASIS)
ZHEAD = "Z-SCORED FEATURES (each area / its frame-wise SD in the session, pre and post alike; iti_zscore_check.py)\n"


def coords(label: str, t: np.ndarray) -> np.ndarray:
    """(n, 5) side_ml, out_ap, reach_mm, SAGITTAL extension, down_dv for licks at exact DAQ peak times t (NaN where unmatched).
    Sagittal = sqrt(out_ap^2 + down_dv^2): the tongue's extension out of the mouth excluding its lateral part (licks run
    mostly DOWN, so out_ap alone misses most of it); lateral / sagittal = tan(encoder exec_angle)."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_peak_s"])
    uid = pd.Series(np.arange(len(A)), index=A.daq_peak_s.to_numpy(float))
    uid = uid[~uid.index.duplicated()]
    K = _K()
    K = K[K.session == f"{an}_{date}"].drop_duplicates("lick_uid").set_index("lick_uid")
    u = uid.reindex(np.asarray(t, float)).to_numpy()
    out = np.full((len(t), 5), np.nan)
    ok = np.isfinite(u)
    R = K.reindex(u[ok].astype(int))
    out[ok, :3] = R[["side_ml", "out_ap", "reach_mm"]].to_numpy(float)
    out[ok, 3] = np.hypot(R.out_ap.to_numpy(float), R.down_dv.to_numpy(float))
    out[ok, 4] = R.down_dv.to_numpy(float)              # 3-axis version (Priya 10-27); consumers index 0..3
    return out


_KC = {}


def _K():
    if "K" not in _KC:
        _KC["K"] = il.lick_table()[["session", "lick_uid", "side_ml", "out_ap", "down_dv", "reach_mm"]]
    return _KC["K"]


def _frame_range(x: str, ce) -> np.ndarray:
    """(first, last) imaging frame time of a session -- all the basis branch needs from the imaging (the edge rule).
    Cached in basis_frame_range/<label>.npy so the full imaging is read once per session, ever."""
    f = il.OUT / "basis_frame_range" / f"{x}.npy"
    if f.exists():
        return np.load(f)
    ft = ce.load_imaging(x)[1]
    r = np.array([ft[0], ft[-1]], float)
    f.parent.mkdir(exist_ok=True)
    np.save(f, r)
    return r


SPLIT_PAIRS = il.SPLIT_PAIRS          # the split itself lives in iti_licks (with_rest / area_features): one definition


def split_names(cols: list[str]) -> list[str]:
    """Feature names of the split basis (= il.area_names(cols, "split"))."""
    return il.area_names(cols, "split")


def sess_labels(sess_dir: Path | None = None) -> list[str]:
    """Session labels of an ITI session folder (default: the current ROI version's, il.SESS_NAME)."""
    return sorted(p.stem for p in (Path(sess_dir) if sess_dir else il.OUT / il.SESS_NAME).glob("*.npz"))


def area_basis(sess_dir: Path | None = None, jobs: int | None = None, basis: str | None = None):
    """THE shared loader of the Allen-area decoders (Priya 10-27 late: split basis everywhere). Returns (D, names, SD):
    D      `load` of the basis -- per session: iti / trial = (X lags 0..0.2 s mean, coords), epoch, animal, *_pos
    names  the basis's feature names (il.area_names; split: ALM, tjM1 + MOs_rest / MOp_rest per hemisphere)
    SD     frame-level SD per session x feature, columns = names (il.area_sd: zscore_check/<SD_NAME> + <SPLIT_NAME>; for a
           reference folder its parent's tables, e.g. june_ref/). X / SD.loc[label] = the KEY-row z-scored features.
    basis None = il.AREA_BASIS (env WFIELD_AREA_BASIS)."""
    src = {None: "areas", "split": "areas_split", "plain": "areas_plain"}[basis]
    return (load(src, sess_dir, jobs), *area_meta(sess_dir, basis))


def area_meta(sess_dir: Path | None = None, basis: str | None = None) -> tuple[list[str], pd.DataFrame]:
    """(names, SD) of `area_basis` without loading the sessions."""
    names = il.area_names(il._load(sess_labels(sess_dir)[0], sess_dir)["cols"], basis)
    return names, il.area_sd(sess_dir, basis)[names]


MASK_OUT_MAX = 0.5
_KEEP = {}


def basis_dir(source: str) -> str:
    """Basis-file folder name of a decoder source (the masked LocaNMF reuses the LocaNMF projections)."""
    return "locanmf" if source == "locanmfM" else source


def locanmf_keep(animal: str) -> np.ndarray:
    """Boolean per joint-LocaNMF component: <= MASK_OUT_MAX of its |footprint| outside `beta_maps.brain_mask()` (the
    imaging-map mask that excludes fibre glue and the olfactory bulbs). Shared by every masked-LocaNMF decoder."""
    if animal not in _KEEP:
        from wfield_local import beta_maps
        from wfield_local import joint_locanmf as jl
        b = jl.load(animal, warn_stale=False)
        A = np.abs(np.nan_to_num(np.asarray(b.A, float))).reshape(-1, b.ncomp)
        _KEEP[animal] = A[~beta_maps.brain_mask().ravel()].sum(0) / A.sum(0) <= MASK_OUT_MAX
    return _KEEP[animal]


def _load_one(item):
    """One session -> (label, record | None). Module-level so `analysis_kit.fan_sessions` can spawn it (the basis
    branch reads the session's imaging for its frame times, which is the slow part)."""
    x, source, sess_dir = item
    sess_dir = Path(sess_dir) if sess_dir else None
    W_d = (il.LAGS_D >= 0.0) & (il.LAGS_D <= 0.2)
    z = il._load(x, sess_dir)
    rec = {"epoch": str(z["epoch"]), "animal": str(z["animal"])}
    if source in AREA_SOURCES:                          # Allen areas: lag-average, then the basis (il.area_features)
        for st, kx, kt in (("iti", "iti_X", "iti_t"), ("trial", "tr_X", "tr_t")):
            Xa = z[kx].astype(float)[:, W_d].mean(1)
            rec[st] = (il.area_features(Xa, z["cols"], x, sess_dir, AREA_SOURCES[source]), z[kt].astype(float))
            rec[f"{st}_pos"] = z["iti_pos" if st == "iti" else "tr_pos"].astype(str)
    else:
        f = il.OUT / f"basis_{basis_dir(source)}" / "sessions" / f"{x}.npz"
        if not f.exists():
            return x, None
        keepc = locanmf_keep(str(z["animal"])) if source == "locanmfM" else None
        b = np.load(f)
        from iti_basis_decoder import LAGS as LB
        from wfield_local import combined_encoding as ce
        ft = _frame_range(x, ce)                                # same edge rule as iti_basis_decoder.extract
        wb = (LB >= 0.0) & (LB <= 0.2)
        for st, kt, kd in (("iti", "iti_t", "iti_side"), ("trial", "tr_t", "tr_side")):
            t, y = z[kt].astype(float), z[kd].astype(float)
            keep = (t > ft[0] - LB[0]) & (t < ft[-1] - LB[-1]) & np.isfinite(y)
            Xb = b[f"{st}_X"].astype(float)[:, wb].mean(1)
            if Xb.shape[0] != keep.sum() or not np.allclose(b[f"{st}_y"], y[keep]):
                raise ValueError(f"{x} {source} {st}: basis licks do not line up with the ITI session file")
            rec[st] = (Xb if keepc is None else Xb[:, keepc], t[keep])
            rec[f"{st}_pos"] = z["iti_pos" if st == "iti" else "tr_pos"].astype(str)[keep]   # spout per lick
    for st in ("iti", "trial"):
        X, t = rec[st]
        rec[st] = (X, coords(x, t))
    return x, rec


def load(source: str, sess_dir: Path | None = None, jobs: int | None = None) -> dict:
    """{label: {epoch, animal, iti: (X (n, feat) lags 0..0.2 mean, Y (n, 3)), trial: (...)}} for one feature basis.
    sess_dir: another folder of `iti_licks.extract` session files (e.g. the June reference, `iti_june_ref.py`).
    Sessions load in PARALLEL (analysis_kit.fan_sessions: fan_out + sort, CLAUDE.md rule 6); returned in label order."""
    from wfield_local import analysis_kit as ak
    labels = sess_labels(sess_dir)
    items = [(x, source, str(sess_dir) if sess_dir else "") for x in labels]
    res, fails = ak.fan_sessions(items, _load_one, jobs=jobs, key=lambda it: it[0], label=f"load {source}",
                                 log=lambda m: None)
    if fails:
        raise RuntimeError(f"load({source}) failed for {[f[0][0] for f in fails]}: {fails[0][1]}")
    return {x: rec for _it, (x, rec) in res if rec is not None}


def _vec_r(P: np.ndarray, Y: np.ndarray) -> float:
    P, Y = P - P.mean(0), Y - Y.mean(0)
    return float((P * Y).sum() / np.sqrt((P ** 2).sum() * (Y ** 2).sum()))


def decode(source: str) -> pd.DataFrame:
    from wfield_local import frozen_readouts
    D = load(source)
    rows = []
    for st in ("iti", "trial"):
        for a in AN:
            S = {}
            for x, d in D.items():
                if d["animal"] != a:
                    continue
                X, C = d[st]
                ok = np.isfinite(C).all(1)
                if ok.sum() >= 20:
                    # columns: lateral, SAGITTAL, angle = atan2(lateral, sagittal) (the encoder's exec_angle; the old
                    # horizontal atan2(lateral, outward) wraps when outward ~ 0, licks running mostly down), reach,
                    # outward (kept for continuity with the 10-26 outward version)
                    ang = np.degrees(np.arctan2(C[ok, 0], C[ok, 3]))
                    S[x] = (d["epoch"], X[ok], np.column_stack([C[ok, 0], C[ok, 3], ang, C[ok, 2], C[ok, 1]]))
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            Ypre = np.vstack([S[q][2] for q in pre])
            sd = Ypre[:, :2].std(0)                                    # per-axis pre SD for the 2-D score
            # STORED BY DEFAULT (11-02 pass 2, `frozen_readouts`, opt-out WIDEFIELD_ITI_FROZEN=0)
            fits = frozen_readouts.pre_loso_ridge(a, f"iti_where_decoder-{source}-{st}", pre, {q: S[q][1] for q in pre},
                                                  {q: S[q][2] for q in pre}, np.logspace(-1, 5, 13), k=5)
            for x, (ep, X, Y) in S.items():
                sc, ms = fits[x if x in pre else None]
                Pr = np.column_stack([m.predict(sc.transform(X)) for m in ms])
                r = {"lateral": np.corrcoef(Pr[:, 0], Y[:, 0])[0, 1], "sagittal": np.corrcoef(Pr[:, 1], Y[:, 1])[0, 1],
                     "2-D": _vec_r(Pr[:, :2] / sd, Y[:, :2] / sd), "angle": np.corrcoef(Pr[:, 2], Y[:, 2])[0, 1],
                     "reach": np.corrcoef(Pr[:, 3], Y[:, 3])[0, 1], "outward": np.corrcoef(Pr[:, 4], Y[:, 4])[0, 1]}
                for tg, v in r.items():
                    rows.append({"source": source, "set": st, "label": x, "animal": a, "epoch": ep, "target": tg,
                                 "r": float(v), "n": len(Y), "sd_lateral": float(Y[:, 0].std()),
                                 "sd_sagittal": float(Y[:, 1].std()), "sd_angle": float(Y[:, 2].std()),
                                 "sd_reach": float(Y[:, 3].std()), "sd_outward": float(Y[:, 4].std())})
    return pd.DataFrame(rows)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")

    srcs =[s for s in SOURCES if s == "areas" or (il.OUT / f"basis_{basis_dir(s)}" / "sessions").exists()]
    R = pd.concat([decode(s) for s in srcs], ignore_index=True)
    R.to_csv(il.OUT / il.tagged("where_decoder.csv"), index=False)
    pre = R[R.epoch == "pre"].groupby(["source", "set", "target", "animal"]).r.mean()
    R["r_norm"] = R.r / [pre.get(k, np.nan) for k in zip(R.source, R.set, R.target, R.animal)]
    pd.set_option("display.width", 220)
    sp = R[(R.source == "areas") & (R.target == "lateral")].groupby(["set", "epoch"])[["sd_lateral", "sd_sagittal", "sd_angle", "sd_reach"]] \
        .median().unstack("epoch")
    print("== lick spread per session (median over sessions, mm)\n" + sp.round(3).to_string())
    T = tests(R)
    T.to_csv(il.OUT / il.tagged("where_decoder_tests.csv"), index=False)
    print(T[(T.value == "r_norm")].round(3).to_string(index=False))
    plot(R, T)
    return 0


def tests(R: pd.DataFrame) -> pd.DataFrame:
    """Change in r and r / pre vs pre per (source, set, target, epoch): nested bootstrap (animals > sessions)."""
    from wfield_local import analysis_kit as ak
    if "r_norm" not in R:
        pre = R[R.epoch == "pre"].groupby(["source", "set", "target", "animal"]).r.mean()
        R = R.assign(r_norm=R.r / [pre.get(k, np.nan) for k in zip(R.source, R.set, R.target, R.animal)])
    rows = []
    for (src, st, tg), t in R.groupby(["source", "set", "target"]):
        ref = {a: t[(t.animal == a) & (t.epoch == "pre")] for a in AN}
        for ep in EP[1:]:
            post = {a: t[(t.animal == a) & (t.epoch == ep)] for a in AN}
            com = [a for a in AN if len(ref[a]) and len(post[a])]
            for val in ("r", "r_norm"):
                iv = ak.boot_delta({a: post[a][val].dropna().to_numpy() for a in com},
                                   {a: ref[a][val].dropna().to_numpy() for a in com}, np.random.default_rng(0))
                per = {a: float(post[a][val].mean() - ref[a][val].mean()) for a in com}
                rows.append({"source": src, "set": st, "target": tg, "value": val, "epoch": ep, "pre_r":
                             float(np.mean([ref[a].r.mean() for a in com])), "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                             "n_down": int(sum(v < 0 for v in per.values())), "n": len(com)})
    return pd.DataFrame(rows)


def _panel(ax, g, tt, colors, rng, title):
    """One target x lick set: session dots / animal's pre, black mean +- SEM, stars in a strip above the data."""
    import matplotlib.patheffects as pe
    for e, ep in enumerate(EP):
        v = g[g.epoch == ep].r_norm.dropna().to_numpy()
        ax.scatter(e + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=colors[ep], alpha=0.6, lw=0)
        if len(v) > 1:
            ax.errorbar(e + 0.3, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt="o", color="k", ms=6, capsize=5, lw=2.2,
                        capthick=2.2)
    ax.axhline(1, color="0.7", lw=0.8, ls="--")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))
    for e, ep in enumerate(EP):
        t = tt[tt.epoch == ep]
        if len(t) and (t.lo.iloc[0] > 0 or t.hi.iloc[0] < 0):
            ax.text(e, 0.95, "*", transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=15,
                    fontweight="bold", path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])
    ax.set_xticks(range(4), ["pre", "acute", "sub", "chronic"], fontsize=9)
    ax.tick_params(labelsize=9)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title(title, fontsize=10.5)


def plot(R: pd.DataFrame, T: pd.DataFrame, tag: str = "") -> None:
    """tag "_z": within-session z-scored features (`iti_zscore_check.py`); writes lick_where_decoder_z.png only (the
    basis figures need per-component session SDs, which do not exist)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    if "r_norm" not in R:
        pre = R[R.epoch == "pre"].groupby(["source", "set", "target", "animal"]).r.mean()
        R = R.assign(r_norm=R.r / [pre.get(k, np.nan) for k in zip(R.source, R.set, R.target, R.animal)])
    TN = T[T.value == "r_norm"]
    yl = "FROZEN decoder r (decoded vs actual)\n/ animal's pre mean (pre = 1)"
    tl = {"lateral": "LATERAL position", "sagittal": "SAGITTAL extension", "2-D": "lateral + sagittal (2-D)",
          "angle": "ANGLE (lateral vs sagittal)", "reach": "REACH (3-D distance)", "outward": "OUTWARD position"}
    sl = {"iti": "ITI (self-initiated) licks", "trial": "CUED in-trial licks"}
    rng = np.random.default_rng(1)
    # A: every target, one figure per basis (Allen areas = main page figure; the other bases = supplements, Priya 10-26)
    for src in [s_ for s_ in SOURCES if s_ in set(R.source)]:
        fig, axs = plt.subplots(2, len(TARGETS), figsize=(4.2 * len(TARGETS), 9.5), squeeze=False)
        for k, st in enumerate(("iti", "trial")):
            for j, tg in enumerate(TARGETS):
                g = R[(R.source == src) & (R.set == st) & (R.target == tg)]
                _panel(axs[k, j], g, TN[(TN.source == src) & (TN.set == st) & (TN.target == tg)], colors, rng,
                       f"{sl[st]}: {tl.get(tg, tg)}\n(pre r {g[g.epoch == 'pre'].r.mean():.2f})")
            axs[k, 0].set_ylabel(yl, fontsize=10)
        fig.suptitle(ZHEAD * bool(tag) + "NEURAL: WHERE did the lick go? FROZEN pre-stroke decoder (fit once on each animal's August pre licks, "
                     f"never refit), {LAB[src]} features.\nOne decoder per target; one dot per session = r between decoded and "
                     "actual tongue-tip position, divided by the animal's pre mean;\nblack = mean +- SEM over sessions; * = "
                     "nested-bootstrap change from pre (animals > sessions) excluding 0.\nTOP ROW: self-initiated ITI licks "
                     "(spout retracted) | BOTTOM ROW: cued in-trial licks (spout in place); separate frozen decoder per row. "
                     "Sagittal = sqrt(forward^2 + down^2); angle = atan2(lateral, sagittal).", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.subplots_adjust(top=0.82 if tag else 0.84)      # tight_layout left a large gap under the 5-line suptitle
        fig.savefig((il.OUT_W if il.ITI_SUBSET else il.OUT) / "figures" / il.tagged(f"lick_where_decoder{tag}.png" if src == "areas"
                                                   else f"lick_where_decoder_{src}{tag}.png"), dpi=105)
        plt.close(fig)
    if tag:
        return
    # B: lateral position in each basis
    srcs = [s_ for s_ in SOURCES if s_ in set(R.source)]
    fig, axs = plt.subplots(2, len(srcs), figsize=(4.6 * len(srcs), 9.5), squeeze=False)
    for k, st in enumerate(("iti", "trial")):
        for j, src in enumerate(srcs):
            g = R[(R.source == src) & (R.set == st) & (R.target == "lateral")]
            _panel(axs[k, j], g, TN[(TN.source == src) & (TN.set == st) & (TN.target == "lateral")], colors, rng,
                   f"{LAB[src]}\n{sl[st]}: LATERAL position (pre r {g[g.epoch == 'pre'].r.mean():.2f})")
        axs[k, 0].set_ylabel(yl, fontsize=10)
    fig.suptitle("NEURAL: the LATERAL-position result in three feature bases (FROZEN pre-stroke decoder, never refit; same "
                 "licks).\nDots = sessions, r / animal's pre; black = mean +- SEM; * = nested-bootstrap change from pre "
                 "excluding 0. Other targets per basis: where_decoder_tests.csv.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(il.OUT / "figures" / il.tagged("lick_where_decoder_bases.png"), dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    if "--z" in sys.argv:                    # redraw from the z-scored run (iti_zscore_check.py); Allen areas only
        Z = (il.OUT_W if il.ITI_SUBSET else il.OUT) / "zscore_check"   # subset: iti_dock_where_preds.py's r table
        Rz = pd.read_csv(Z / il.tagged("where_decoder_z.csv"))
        Tz = tests(Rz)
        Tz.to_csv(Z / il.tagged("where_decoder_tests_z.csv"), index=False)
        plot(Rz, Tz, "_z")
        raise SystemExit(0)
    if "--plot-only" in sys.argv:
        plot(pd.read_csv(il.OUT / il.tagged("where_decoder.csv")),
             pd.read_csv(il.OUT / il.tagged("where_decoder_tests.csv")))
        raise SystemExit(0)
    raise SystemExit(main())
