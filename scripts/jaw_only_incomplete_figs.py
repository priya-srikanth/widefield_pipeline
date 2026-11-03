"""JAW-WITHOUT-LICK and INCOMPLETE-LICK trials: how often per session, and what the jaw / tongue do on them (Priya 11-02:
"kinematics Fig 6.1 is titled jaw without a lick but plots just jaw -- a figure showing traces of / frequency of jaw
without lick trials (and incomplete lick trials) per session across epochs").

    python -m scripts.jaw_only_incomplete_figs [--jobs 8]

TRIAL CLASSES, per ENGAGED trial (`orofacial_cohort.engagement_keep`, the behaviour gate; EVERY engaged trial is the
denominator, so the five shares sum to 1): contact / incomplete / jaw_only / no_response / unknown from THE trial
classification, `wfield_local.trial_classes` (Priya 11-03: one definition for every analysis; cue -> trial stop window,
DAQ spout contact, jaw cam4 OR cam1, close_center jaw unjudged). History: until 11-03 this script kept only
``jaw_pass_qc`` trials (the jaw-MOVED call read as a QC gate), then classified all engaged trials with its own rule
(cam4 jaw only, video-matched contact, no window); the shared definition replaced both (regression in DECISIONS
2026-11-03: 99.35 % of engaged trials unchanged).

FIG A (frequency, POOLED): share of trials per class, per session x spout, minus the animal's August pre mean at that
spout (change vs own pre); dots = sessions coloured by animal; marker = animal-weighted change with the paired nested
bootstrap 95 % CI (`analysis_kit.boot_delta`, 2,000 draws); * CI excludes 0, ** bootstrap p < 0.05 / 18 (Bonferroni over
6 spouts x 3 post epochs, per class); pre column = pre sessions vs their own mean (noise floor). Stars in the top band
(`sig_marks.star_top`, Priya 10-30 rule).
FIG B (frequency over days): per animal thin lines (change vs own pre) + the POOLED time model of the goal page
(`mixed_models`: y ~ post + post:lday, animal random effects), BH over the 6 spouts per term and class.
FIG C / D (traces): cue-aligned 3-D JAW OPENING (distance from the trial's pre-cue resting jaw, mm) and TONGUE-OUT share
(frames with a 3-D tongue point), per class x spout x epoch: 20 ms bins, mean per session (>= 2 trials of the class),
then per animal x epoch, then mean over animals (band = SEM over animals; animals weighted equally). LEVELS, not
changes: the classes are defined by what the jaw / tongue did, so their traces differ by construction; the question is
what each class LOOKS like and whether it changes with epoch.
Pre = August only (baseline rule); curated sessions; 3-D-gate failures left out (`pose3d.failed_sessions`).
Outputs session_poses/cam1_licks/trial_classes/{sessions.csv, tests.csv, time_lmm.csv, traces.csv.gz, figures/}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

V36 = "lp_r4_heatmap_20261013__clean_v36"
POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")     # = canonical order (lesions left: L = ipsi)
LAB = {"close_L": "close ipsi", "close_center": "close mid", "close_R": "close contra", "far_L": "far ipsi",
       "far_center": "far mid", "far_R": "far contra"}
CLASSES = ("contact", "incomplete", "jaw_only", "no_response", "unknown")   # = wfield_local.trial_classes.CLASSES
SHOW = {"jaw_only": "jaw moves, no lick", "incomplete": "incomplete lick (licked, no contact)",
        "contact": "contact lick"}
EPS = ("pre", "acute", "subacute", "chronic")
EP_COL = {"pre": "0.35", "acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}
AN_COL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}
BIN_MS, WIN_MS, MIN_FRAMES, MIN_TR = 20.0, (-500.0, 3000.0), 5, 2
MIN_AN = 2  # animals needed to draw a pooled trace line
PRE_FROM = "20260801"
NB = 2000


# THE trial classification lives in `wfield_local.trial_classes` (Priya 11-03: one definition for every analysis; read
# its docstring for the rules). This script keeps the names other windows import -- GATES, session_classes, cue_classes,
# in_gate -- as thin wrappers, so the imaging section 1 table and the cohort-1 rebuild need no change.
from wfield_local import trial_classes as TC  # noqa: E402

GATES = TC.GATES
CUE_MATCH_S = 0.020          # a DAQ cue and a video trial are the same trial within 20 ms of cue time (as `lp_licks`)


def session_classes(label: str, v36: Path | str | None = None, c1: Path | str | None = None,
                    window: str = TC.DEFAULT_WINDOW) -> pd.DataFrame | None:
    """Every trial of one session (NO engagement filter) -> DataFrame [trial_id, position, cue_daq, cls, has_cam1],
    from `trial_classes.session_trials` (window "cue_stop" by default, Priya 11-03; "trial" = from the strobe). None
    when the session has no video outputs (no DAQ fallback, so the two definitions are never mixed silently)."""
    T = TC.session_trials(label, window=window, v36=v36, c1=c1)
    return None if T is None else T[["trial_id", "position", "cue_daq", "cls", "has_cam1"]]


def cue_classes(table: pd.DataFrame | None, cue_s: np.ndarray, tol: float = CUE_MATCH_S) -> np.ndarray:
    """Class per DAQ cue (``cue_s``, DAQ seconds), matched to ``session_classes`` rows by nearest cue time within
    ``tol``; "unknown" where nothing matches (or ``table`` is None). Matched by TIME, never by trial id (ids are
    offset in some sessions; `lp_licks`). ONE definition (rule 9): `trial_classes.cue_classes` (moved there 11-03)."""
    return TC.cue_classes(table, cue_s, tol)


def in_gate(cls: np.ndarray, gate: str) -> np.ndarray:
    """Boolean per trial: class in GATES[gate] ("working", "camera_lick" or "attempted"); `trial_classes.in_gate`."""
    return TC.in_gate(cls, gate)


def session_job(job) -> dict:
    """fan_sessions worker (module level): per-trial classes, counts per position, and per class x position traces."""
    s, d3, v36, c1 = job
    from wfield_local import orofacial_cohort as oc
    animal, date = s.split("_")
    sd = Path(v36) / animal / date
    keep, _ = oc.engagement_keep(animal, date, sd)
    T = TC.session_trials(s, window=TC.DEFAULT_WINDOW, v36=v36, c1=c1)[["trial_id", "position", "cls"]]
    if keep is not None:
        T = T[T.trial_id.isin(keep)]
    counts = [{"session": s, "animal": animal, "date": date, "position": p, "n_trials": len(g),
               **{f"share_{c}": float((g.cls == c).mean()) for c in CLASSES},
               **{f"n_{c}": int((g.cls == c).sum()) for c in CLASSES}} for p, g in T.groupby("position")]
    fr = Path(d3) / f"{s}_frames.csv.gz"
    traces = []
    if fr.exists():
        S = pd.read_csv(fr, usecols=["trial_id", "position", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv", "jaw_ml",
                                     "jaw_ap", "jaw_dv"]).drop_duplicates(["trial_id", "t_ms"])
        S = S[S.trial_id.isin(T.trial_id)]
        J = S[["jaw_ml", "jaw_ap", "jaw_dv"]].to_numpy(float)
        t = S.t_ms.to_numpy(float)
        rest = (S.assign(_x=J[:, 0], _y=J[:, 1], _z=J[:, 2])[(t < 0) & (t >= -500)]
                .groupby("trial_id")[["_x", "_y", "_z"]].median())
        V = {"jaw_open": np.linalg.norm(J - rest.reindex(S.trial_id).to_numpy(float), axis=1),
             "tongue_out": np.isfinite(S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)).all(1).astype(float)}
        edges = np.arange(WIN_MS[0], WIN_MS[1] + BIN_MS, BIN_MS)
        b = np.digitize(t, edges) - 1
        inwin = (b >= 0) & (b < len(edges) - 1)
        c_of = S.trial_id.map(dict(zip(T.trial_id, T.cls))).to_numpy(object)
        pos = S.position.to_numpy(str)
        for p in POS:
            for c in SHOW:
                m = inwin & (pos == p) & (c_of == c)
                ntr = int(S.trial_id[m].nunique())
                if ntr < MIN_TR:
                    continue
                for k, v in V.items():
                    ok = m & np.isfinite(v)
                    n = np.bincount(b[ok], minlength=len(edges) - 1)
                    with np.errstate(all="ignore"):
                        mu = np.where(n >= MIN_FRAMES, np.bincount(b[ok], weights=v[ok], minlength=len(edges) - 1) / n,
                                      np.nan)
                    traces += [{"session": s, "animal": animal, "date": date, "position": p, "cls": c, "n_trials": ntr,
                                "measure": k, "t_ms": edges[i] + BIN_MS / 2, "v": mu[i]} for i in range(len(edges) - 1)]
    return {"counts": counts, "traces": traces}


def _pre_mean(S: pd.DataFrame, col: str) -> pd.Series:
    return S[S.epoch == "pre"].groupby(["animal", "position"])[col].mean()


def tests(S: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(0)
    rows = []
    for c in SHOW:
        col = f"share_{c}"
        for p in POS:
            g = S[S.position == p]
            pre = {a: x[col].tolist() for a, x in g[g.epoch == "pre"].groupby("animal")}
            for ep in EPS[1:]:
                post = {a: x[col].tolist() for a, x in g[g.epoch == ep].groupby("animal")}
                iv, o = ak.boot_delta(post, pre, rng, n_boot=NB, draws=True)
                if iv is None:
                    continue
                pv = ak.boot_p(o)
                rows.append(dict(cls=c, position=p, epoch=ep, change=iv.point, lo=iv.lo, hi=iv.hi, p=pv,
                                 n_animals=iv.n_animals, n_sessions=sum(map(len, post.values())),
                                 sig=iv.lo > 0 or iv.hi < 0, sig_bonf=pv < 0.05 / 18,
                                 **{a: float(np.mean(post[a])) - float(np.mean(pre[a])) for a in post if a in pre}))
    return pd.DataFrame(rows)


def _tests_and_epochs(S: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Pool worker: the bootstrap tests, then the epoch figure that needs them."""
    R = tests(S)
    fig_epochs(S, R, path)
    return R


def fig_epochs(S: pd.DataFrame, R: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import sig_marks
    rng = np.random.default_rng(1)
    rows = ("jaw_only", "incomplete")
    fig, axs = plt.subplots(len(rows), len(POS), figsize=(17, 7.6), sharey="row", squeeze=False)
    for i, c in enumerate(rows):
        col = f"share_{c}"
        base = _pre_mean(S, col)
        d = S.assign(chg=S[col] - base.reindex(pd.MultiIndex.from_arrays([S.animal, S.position])).to_numpy())
        for j, p in enumerate(POS):
            ax = axs[i, j]
            for k, ep in enumerate(EPS):
                g = d[(d.position == p) & (d.epoch == ep)]
                ax.scatter(k + rng.uniform(-0.18, 0.18, len(g)), g.chg, s=9, c=[AN_COL[a] for a in g.animal], alpha=0.55,
                           lw=0)
                if ep == "pre":
                    ax.plot(k + 0.32, 0, "o", ms=7, color=EP_COL[ep])
                    continue
                r = R[(R.cls == c) & (R.position == p) & (R.epoch == ep)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(k + 0.32, r.change, yerr=[[r.change - r.lo], [r.hi - r.change]], fmt="o", ms=7,
                            color=EP_COL[ep], capsize=3, lw=1.8)
                if r.sig:
                    sig_marks.star_top(ax, k + 0.32, "**" if r.sig_bonf else "*", color=EP_COL[ep], fontsize=13)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["pre", "acute", "subac", "chron"], fontsize=8.5)
            if i == 0:
                ax.set_title(LAB[p], fontsize=11)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        axs[i, 0].set_ylabel(f"share of trials: {SHOW[c]}\nchange vs own August pre", fontsize=9)
    hs = [plt.Line2D([], [], marker="o", ls="", color=AN_COL[a], label=a) for a in AN_COL]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Jaw-without-lick and incomplete-lick trials per session, POOLED: share of ALL engaged trials, change "
                 "vs each animal's own August pre; dots = sessions; marker = animal-weighted change, paired nested "
                 "bootstrap 95% CI;\n* CI excludes 0, ** p < 0.05 / 18 (Bonferroni, 6 spouts x 3 epochs, per row); pre "
                 "column = pre sessions vs their own mean (noise floor)", fontsize=9.5)
    sig_marks.reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fig_days(S: pd.DataFrame, path: Path) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import epochs
    from wfield_local import mixed_models as mm
    rows = ("jaw_only", "incomplete")
    out = []
    S = S.assign(day=[epochs.days_since_stroke(f"{a}_{d[4:]}") for a, d in zip(S.animal, S.date)])
    fig, axs = plt.subplots(len(rows), len(POS), figsize=(17, 7.4), sharey="row", squeeze=False)
    for i, c in enumerate(rows):
        col = f"share_{c}"
        base = _pre_mean(S, col)
        d = S.assign(chg=S[col] - base.reindex(pd.MultiIndex.from_arrays([S.animal, S.position])).to_numpy())
        L = mm.add_time_covariates(S.drop(columns=["day"]).rename(columns={"session": "sid"})
                                   .assign(label=lambda x: x.animal + "_" + x.date.str[4:]).rename(columns={col: "y"}))
        Mt = mm.lmm_table(L, ["position"], models=("time",), print_animals=False, fdr_by=()).assign(cls=c)
        out.append(Mt)
        for j, p in enumerate(POS):
            ax = axs[i, j]
            for a in AN_COL:
                g = d[(d.position == p) & (d.animal == a)].sort_values("day")
                ax.plot(g.day, g.chg, "o-", ms=2, lw=0.8, alpha=0.6, color=AN_COL[a], label=a)
            cf = Mt[Mt.position == p].set_index("term")
            if len(cf):
                dd = np.linspace(1, d.day.max(), 200)
                ax.plot(dd, cf.coef["post"] + cf.coef["post:lday"] * np.log(dd), "k-", lw=2.2, label="pooled fit")
                ax.plot([d.day.min(), -0.5], [0, 0], "k-", lw=2.2)
                st = lambda t, cf=cf: f"{cf.coef[t]:+.2f} (q {cf.q[t]:.3f}){'*' if cf.q[t] < 0.05 else ''}"  # noqa: E731
                ax.text(0.97, 0.97, f"step {st('post')}\nrecovery/log-day {st('post:lday')}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.5)
            ax.axvline(0, color="k", lw=0.7, ls="--")
            ax.axhline(0, color="0.6", lw=0.7, ls=":")
            if i == 0:
                ax.set_title(LAB[p], fontsize=11)
            if i == len(rows) - 1:
                ax.set_xlabel("day after stroke")
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        axs[i, 0].set_ylabel(f"share of trials: {SHOW[c]}\nchange vs own August pre", fontsize=9)
    hs, ls_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(hs[:5], ls_[:5], loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Jaw-without-lick and incomplete-lick trial shares over days: thin = per animal (one point per session); "
                 "BLACK = POOLED time model y ~ post + post:lday (animal random effects; the goal page's Fig 16.9 model), "
                 "step = day-1 change vs pre,\nrecovery = change per e-fold of days; q = BH over the 6 spouts per row",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return pd.concat(out, ignore_index=True)


def fig_traces(Tr: pd.DataFrame, measure: str, ylab: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # count only FINITE values: a session whose jaw is hidden on every frame of the class still has rows (all NaN), and
    # counting rows labelled a one-animal line "2a" (chronic far-mid jaw-only, e3 11-02). A bin is drawn only where
    # >= MIN_AN animals have a value in it.
    X = Tr[(Tr.measure == measure) & Tr.v.notna()]
    A = X.groupby(["animal", "epoch", "cls", "position", "t_ms"]).v.mean().reset_index()
    G = A.groupby(["epoch", "cls", "position", "t_ms"]).v.agg(["mean", "sem", "count"]).reset_index()
    G.loc[G["count"] < MIN_AN, ["mean", "sem"]] = np.nan
    nA = A.groupby(["epoch", "cls", "position"]).animal.nunique()
    nS = X.groupby(["epoch", "cls", "position"]).session.nunique()
    nT = X.drop_duplicates(["session", "cls", "position"]).groupby(["epoch", "cls", "position"]).n_trials.sum()
    fig, axs = plt.subplots(len(SHOW), len(POS), figsize=(18, 9.5), sharex=True, sharey=True, squeeze=False)
    for i, c in enumerate(SHOW):
        for j, p in enumerate(POS):
            ax = axs[i, j]
            ax.axvspan(WIN_MS[0], 0, color="0.93", zorder=0)
            lines = []
            for ep in EPS:
                g = G[(G.epoch == ep) & (G.cls == c) & (G.position == p)].sort_values("t_ms")
                k = (ep, c, p)
                if g.empty:
                    continue
                if nA.get(k, 0) < MIN_AN or g["mean"].isna().all():   # < 2 animals in any bin: counted, not drawn
                    lines.append(f"{ep[:4]} {nA.get(k, 0)}a/{nS.get(k, 0)}s/{int(nT.get(k, 0))}t (not drawn)")
                    continue
                ax.plot(g.t_ms, g["mean"], color=EP_COL[ep], lw=1.3)
                ax.fill_between(g.t_ms, g["mean"] - g["sem"].fillna(0), g["mean"] + g["sem"].fillna(0),
                                color=EP_COL[ep], alpha=0.18, lw=0)
                lines.append(f"{ep[:4]} {nA.get(k, 0)}a/{nS.get(k, 0)}s/{int(nT.get(k, 0))}t")
            ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                    color="0.3")
            if i == 0:
                ax.set_title(LAB[p], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{SHOW[c]}\n{ylab}", fontsize=9)
            if i == len(SHOW) - 1:
                ax.set_xlabel("ms from cue")
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    hs = [plt.Line2D([], [], color=EP_COL[e], lw=2, label=e) for e in EPS]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"Cue-aligned {ylab} by trial class, POOLED: 20 ms bins, mean per session (>= {MIN_TR} trials of the "
                 "class at the spout), then per animal x epoch, then mean over animals (band = SEM over animals); pre = "
                 "August;\ncorner = animals / sessions / trials per epoch (< 2 animals: not drawn); grey = before the cue. Rows: jaw moves without a "
                 "lick (cam1-corrected) | licked but never touched the spout | touched the spout", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    from wfield_local import pose3d as P3
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--reuse", action="store_true", help="read sessions.csv / traces.csv.gz from the last run "
                    "instead of re-running the session loop (stats / figure changes only; tables are saved at 4 significant "
                    "digits, so a CI can move in the 3rd decimal)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    c1 = Path(rv.root("dlc")) / "session_poses" / "cam1_licks"
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    v36 = Path(rv.root("orofacial")) / V36 / "sessions"
    out = c1 / "trial_classes"
    if a.reuse:
        S = pd.read_csv(out / "sessions.csv", dtype={"date": str})
        Tr = pd.read_csv(out / "traces.csv.gz", dtype={"date": str})
    else:
        S, Tr = _collect(a, ak, epochs, P3, c1, d3, v36, out)
    _report(S, Tr, out)
    return 0


def _collect(a, ak, epochs, P3, c1: Path, d3: Path, v36: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Session loop (fanned over cores) -> per-session class shares S and binned traces Tr, written to out."""
    bad = P3.failed_sessions(d3)
    sess = sorted({f.name.replace("_events.csv", "") for f in c1.glob("PS*_events.csv")} - set(bad) & ak.curated_ids())
    sess = [s for s in sess if (v36 / s.split("_")[0] / s.split("_")[1] / "tongue_jaw_mismatch.csv").exists()]
    res, fails = ak.fan_sessions([(s, str(d3), str(v36), str(c1)) for s in sess], session_job, jobs=a.jobs,
                                 key=lambda j: j[0])
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    S = pd.DataFrame([r for _, x in res for r in x["counts"]])
    Tr = pd.DataFrame([r for _, x in res for r in x["traces"]])
    # epoch_of costs ~35 ms a call: look up once per session, not once per trace row (~700k rows ran for hours)
    ep_of = {k: epochs.epoch_of(f"{k[0]}_{k[1][4:]}") for k in set(zip(S.animal, S.date)) | set(zip(Tr.animal, Tr.date))}
    for D in (S, Tr):
        D["epoch"] = [ep_of[k] for k in zip(D.animal, D.date)]
    S = S[S.epoch.notna() & ((S.epoch != "pre") | (S.date >= PRE_FROM))].reset_index(drop=True)
    Tr = Tr[Tr.epoch.notna() & ((Tr.epoch != "pre") | (Tr.date >= PRE_FROM))].reset_index(drop=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "sessions.csv", index=False, float_format="%.4g")
    Tr.to_csv(out / "traces.csv.gz", index=False, float_format="%.4g")
    return S, Tr


def _report(S: pd.DataFrame, Tr: pd.DataFrame, out: Path) -> None:
    """Per-animal tables, then the tests, time model and four figures (in parallel)."""
    pd.set_option("display.width", 250)
    print(f"{S.session.nunique()} sessions; per epoch:", S.drop_duplicates("session").epoch.value_counts().to_dict())
    print("\n== per animal first (rule 8): share of trials, epoch means (pre = August)")
    for c in SHOW:
        print(f"-- {SHOW[c]}")
        print(S.pivot_table(index=["animal", "epoch"], columns="position", values=f"share_{c}", sort=False)
              .reindex(columns=list(POS)).round(3).to_string())
    # the four outputs are independent (tests + epoch figure, time model + days figure, two trace figures): one process
    # each; every worker is module-level and seeds its own rng, so the result does not depend on scheduling
    from concurrent.futures import ProcessPoolExecutor
    F = out / "figures"
    with ProcessPoolExecutor(max_workers=4) as ex:
        f_tests = ex.submit(_tests_and_epochs, S, F / "trial_class_share_epochs.png")
        f_days = ex.submit(fig_days, S, F / "trial_class_share_days.png")
        f_tr = [ex.submit(fig_traces, Tr, m, y, F / f"trial_class_{n}_traces.png")
                for m, y, n in (("jaw_open", "jaw opening from rest (mm, 3-D)", "jaw"),
                                ("tongue_out", "share of frames with the tongue out", "tongue"))]
        R, M = f_tests.result(), f_days.result()
        for f in f_tr:
            f.result()
    R.to_csv(out / "tests.csv", index=False, float_format="%.4g")
    print("\n== pooled change vs own pre (animal-weighted, paired nested bootstrap)")
    print(R[["cls", "position", "epoch", "change", "lo", "hi", "p", "sig", "sig_bonf", "n_animals", "n_sessions"]]
          .round(3).to_string(index=False))
    M.to_csv(out / "time_lmm.csv", index=False, float_format="%.4g")
    print("\n== pooled time model (q = BH over 6 spouts per class x term)")
    print(M[M.term != "pre_level"][["cls", "position", "term", "coef", "q", "re"]].round(3).to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    raise SystemExit(main())
