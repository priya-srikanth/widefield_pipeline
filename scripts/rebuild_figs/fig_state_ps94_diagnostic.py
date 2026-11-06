"""Imaging results page Fig 3.5 (`state_ps94_diagnostic.png`; PI-deck slide 14, hidden): PS94 behavioural-state
diagnostic -- the frozen state decoder (the 11-02 PRODUCTION arm, per-session z-score) against a class-balanced-z
diagnostic and a within-session refit, per animal x epoch, and where PS94's post-stroke errors go.

PROVENANCE: recreated 2026-11-03 (cohort-1 rebuild, step 3) from the share-only
  M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-02/fig_state_ps94_diagnostic.py
  (11-02 22:49; the page image is its 22:58 render). NOT the later fig_state_ps94_balem.py (11-03 08:59, bal_em arm),
  which is not on the page.
  Computation (arms, LOSO, GroupKFold, seeds), layout and dpi unchanged. Changes: argparse (--out / --plot-only /
  --cache), no sys.path insert of widefield_pipeline (uses the repo on PYTHONPATH), output folder no longer
  build_inputs/, figdata pickle (<out>/data/state_ps94_diagnostic_figdata.pkl = the per-session rows, the speed
  histograms and the pooled confusions; --plot-only needs no cache, no models).

Question (Priya, 11-02): is PS94's loss on the frozen state decoder a problem with the state DEFINITIONS?
Arms (all scored by `locomotor_decoder.score`, pre column leave-one-session-out, post = model on all pre sessions):
  frozen (production)  `session_zscore` features + the STORED scaling="session_z" models, read-only via
                       `frozen_models.find` (aborts rather than fits if a model is missing).
  class-balanced z     diagnostic only: same pipeline (`fit_frozen`), features z-scored with the class-balanced mean,
                       fitted in memory, never stored.
  within-session refit GroupKFold(min(5, n_periods)) by period inside each session.

INPUTS:
  --cache  raw segment features + per-session treadmill-speed histograms cached by
           labcams/state_decoder_ps94_check/collect.py (built exactly as `by_animal_day` builds them); default = the
           original's hard-coded M:/MICROSCOPE/Priya/Widefield/labcams/state_decoder_ps94_check/cache.
  stored session_z state models, found through `frozen_models.find` on the spec built from the joint-LocaNMF basis
           (`joint_locanmf.load(an, sessions=locanmf_cue_lick_analysis.SESSIONS)`). The spec / store are those of the
           `wfield_local` on the path. FREEZE_VERSION 1 (widefield_pipeline, wf_attempt: the 11-02 models) finds the
           models the page used. FREEZE_VERSION 2 (the rebuild, 11-03: spec carries a training-data fingerprint,
           passed here exactly as `locomotor_decoder.frozen_state_models` builds it, and a NEW store) finds nothing
           until the rebuild refits the state models, and the script then stops ("refusing to fit") by design.
           So the reproduction of the CURRENT figure is run with PYTHONPATH = widefield_pipeline (the original's
           repo); the rebuild run needs the refit state models first.
REBUILD: re-run collect.py (new data -> new cache; give --cache) and the frozen state models; note the production state
  decoder changed to bal_em on 11-03 (fig_state_ps94_balem.py), so the "frozen (production)" label is 11-02's.

REPRODUCIBILITY (measured 11-03): with the default BLAS threading this reproduces the page's CSV and PNG exactly.
  With OMP/MKL/OPENBLAS_NUM_THREADS=2 the in-memory class-balanced-z fit (`fit_frozen`, lbfgs) moved 27 PS92 / PS93
  balz rows by 1-2 segments (balanced accuracy <= .005); production and refit arms were unchanged. So do not cap
  BLAS threads when checking this figure against the page.

PLOT: open markers = sessions (shape = animal; colour = arm), filled marker + bar = mean over sessions +- SEM, no lines
across epochs, no significance marks (single-animal descriptive panels).
OUT: <out>/state_ps94_diagnostic.png / .csv + state_ps94_diagnostic_confusions.csv
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import inspect
import pickle
import sys
from pathlib import Path

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

NAME = "state_ps94_diagnostic"
CACHE = Path(__import__("wfield_local.config", fromlist=["x"]).resolver().root("cohort_out")) / "state_decoder_ps94_check" / "cache"
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
EPOCHS = ("pre", "acute", "subacute", "chronic")
CL = ["quiet", "running", "licking"]          # == list(locomotor_state.THREE_WAY), asserted in collect()
ARMS = (("frozen", "Frozen pre-stroke decoder (production)", "#1f4e9c"),
        ("balz", "Frozen, class-balanced z (diagnostic)", "#d9822b"),
        ("within", "Within-session refit", "0.55"))
EDGE = np.linspace(-20, 60, 161)


def bal_z(X, y):
    mu = np.mean([X[y == c].mean(0) for c in CL if (y == c).sum()], 0)
    sd = X.std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def within_conf(X, y, g):
    from sklearn.model_selection import GroupKFold
    from wfield_local import locomotor_decoder as ld
    yi = np.array([CL.index(v) for v in y])
    k = min(5, len(np.unique(g)))
    M = np.zeros((3, 3))
    if k < 2:
        return None
    for tr, te in GroupKFold(k).split(X, yi, g):
        if len(np.unique(yi[tr])) < 3:
            continue
        m = ld._pipe().fit(X[tr], yi[tr])
        for t, p in zip(yi[te], m.predict(X[te])):
            M[t, p] += 1
    return M if M.sum() else None


def _row(an, lab, d, arm, M):
    from wfield_local.epoch_figures import epoch_of_day
    ep = "pre" if d <= 0 else epoch_of_day(an, int(d))
    rec = {c: (M[i, i] / M[i].sum() if M[i].sum() else np.nan) for i, c in enumerate(CL)}
    ok = [v for v in rec.values() if np.isfinite(v)]
    r = {"animal": an, "label": lab, "day": int(d), "epoch": ep, "arm": arm,
         "balanced_acc": float(np.mean(ok)) if len(ok) >= 2 else np.nan}
    for i, a in enumerate(CL):
        r[f"recall_{a}"] = rec[a]
        r[f"n_{a}"] = int(M[i].sum())
        for j, b in enumerate(CL):
            r[f"C_{a}->{b}"] = int(M[i, j])
    return r


def collect(cache: Path):
    from wfield_local import frozen_models as fm
    from wfield_local import joint_locanmf
    from wfield_local import locomotor_decoder as ld
    from wfield_local import locomotor_state as ls
    from wfield_local.epoch_figures import epoch_of_day
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    assert list(ls.THREE_WAY) == CL, ls.THREE_WAY
    rows, hist, model_dirs = [], [], {}
    for an in ANIMALS:
        D = pickle.load(open(cache / f"{an}.pkl", "rb"))
        raw = D["got"]
        for lab, h in D["hist"].items():
            d = raw[lab][3] if lab in raw else None
            if d is not None:
                hist.append((an, lab, "pre" if d <= 0 else epoch_of_day(an, int(d)), np.asarray(h, float),
                             D["beh"].get(lab, {}).get("run_sp_median", np.nan)))
        # PRODUCTION arm: stored session_z models, read-only
        got = {lab: (ld.session_zscore(X), np.asarray(y), g, d) for lab, (X, y, g, d) in raw.items()}
        pre = {lab: v for lab, v in got.items() if v[3] <= 0}
        basis = joint_locanmf.load(an, sessions=SESSIONS)
        kw = {}
        if "train_data" in inspect.signature(fm.make_spec).parameters:
            # FREEZE_VERSION >= 2 (rebuild): the training-data fingerprint, built exactly as
            # locomotor_decoder.frozen_state_models builds it (z-scored X, y, groups per pre session, sorted)
            kw["train_data"] = {lab: v[:3] for lab, v in sorted(pre.items())}
        spec = fm.make_spec(an, "state", align="segments", source=f"joint_{basis.basis_id}", train_labels=list(pre),
                            basis_id=basis.basis_id, n_features=int(next(iter(pre.values()))[0].shape[1]),
                            post_s=ls.SEGMENT_S, zscore=True, scaling="session_z", **kw)
        mdir, _o = fm.find(spec)
        if mdir is None:
            raise SystemExit(f"{an}: no stored session_z state model -- refusing to fit one here")
        models = fm._read(mdir)
        model_dirs[an] = str(mdir)
        # DIAGNOSTIC arm: class-balanced z, fitted in memory, never stored
        gb = {lab: (bal_z(X, np.asarray(y)), np.asarray(y), d) for lab, (X, y, g, d) in raw.items()}
        preb = {lab: v for lab, v in gb.items() if v[2] <= 0}
        fullb = ld.fit_frozen(np.vstack([v[0] for v in preb.values()]), np.concatenate([v[1] for v in preb.values()]))
        for lab, (X, y, g, d) in sorted(got.items()):
            m = models["loso"].get(lab) if d <= 0 else models["full"]
            r = ld.score(m, X, y) if m is not None else None
            if r is None:
                continue
            rows.append(_row(an, lab, d, "frozen", r["confusion"]))
            if d <= 0:
                keep = [v for l2, v in preb.items() if l2 != lab]
                mb = ld.fit_frozen(np.vstack([v[0] for v in keep]), np.concatenate([v[1] for v in keep]))
            else:
                mb = fullb
            rb = ld.score(mb, gb[lab][0], y) if mb is not None else None
            if rb is not None:
                rows.append(_row(an, lab, d, "balz", rb["confusion"]))
            Mw = within_conf(X, y, np.asarray(g))
            if Mw is not None:
                rows.append(_row(an, lab, d, "within", Mw))
        print(f"{an}: {mdir.name}", flush=True)
    return pd.DataFrame(rows), hist, model_dirs


def confusions(D: pd.DataFrame) -> pd.DataFrame:
    conf = []
    for (an, ep, arm), s in D.groupby(["animal", "epoch", "arm"]):
        M = np.array([[s[f"C_{a}->{b}"].sum() for b in CL] for a in CL], float)
        for i, a in enumerate(CL):
            conf.append({"animal": an, "epoch": ep, "arm": arm, "true": a, "n": int(M[i].sum()),
                         **{f"pred_{b}": M[i, j] / max(M[i].sum(), 1) for j, b in enumerate(CL)}})
    return pd.DataFrame(conf)


def _points(ax, sub, col, x0, color, mark_by_animal=True):
    rng = np.random.default_rng(1)
    for _i, (a, s) in enumerate(sub.groupby("animal")):
        v = s[col].dropna()
        ax.scatter(x0 + rng.uniform(-0.05, 0.05, len(v)), v, marker=MARK[a] if mark_by_animal else "o", s=22,
                   facecolors="none", edgecolors=color, linewidths=0.9, alpha=0.8, zorder=2)
    v = sub[col].dropna()
    if len(v):
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        ax.errorbar(x0 + 0.13, v.mean(), yerr=se, color=color, ls="none", marker="o", ms=6, mfc=color, mec=color,
                    elinewidth=2, capsize=3, zorder=3)


def _epoch_axis(ax, ylim=(0.0, 1.04)):
    ax.set_xticks(np.arange(len(EPOCHS)))
    ax.set_xticklabels(["Pre", "Acute", "Sub.", "Chron."], fontsize=10)
    ax.set_xlim(-0.5, len(EPOCHS) - 0.4)
    ax.set_ylim(*ylim)
    ax.axhline(1 / 3, color="0.6", ls=":", lw=1, zorder=1)


def draw(D: pd.DataFrame, hist: list, C: pd.DataFrame, png: Path) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(15, 10.2))
    gs = fig.add_gridspec(3, 12, height_ratios=(1, 1, 0.95), hspace=0.62, wspace=1.6)
    dx = {"frozen": -0.27, "balz": 0.0, "within": 0.27}
    dx = {k: v - 0.06 for k, v in dx.items()}
    # row 1: balanced accuracy per animal
    for k, an in enumerate(ANIMALS):
        ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
        for arm, _l, col in ARMS:
            for e in EPOCHS:
                sub = D[(D.animal == an) & (D.arm == arm) & (D.epoch == e)]
                _points(ax, sub, "balanced_acc", EPOCHS.index(e) + dx[arm], col)
        _epoch_axis(ax)
        ax.set_title(an + ("  (no acute sessions scored)" if an == "PS95" and
                           not len(D[(D.animal == an) & (D.epoch == "acute")]) else ""), fontsize=11,
                     fontweight="bold" if an == "PS94" else "normal")
        if k == 0:
            ax.set_ylabel("Balanced accuracy\n(3-way, chance 1/3)")
    # row 2: PS94 recall per class + running speed per session
    for k, c in enumerate(CL):
        ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
        for arm, _l, col in ARMS:
            for e in EPOCHS:
                sub = D[(D.animal == "PS94") & (D.arm == arm) & (D.epoch == e)]
                _points(ax, sub, f"recall_{c}", EPOCHS.index(e) + dx[arm], col)
        _epoch_axis(ax)
        ax.set_title(f"PS94: {c.upper() if c != 'quiet' else 'QUIET (rest)'} recall", fontsize=11)
        if k == 0:
            ax.set_ylabel("Recall (per class)")
    ax = fig.add_subplot(gs[1, 9:12])
    H = pd.DataFrame([{"animal": a, "label": lab, "epoch": e, "run_sp_median": m,
                       "frac_run": h[EDGE[1:] > 3].sum() / max(h.sum(), 1)} for a, lab, e, h, m in hist])
    for e in EPOCHS:
        sub = H[(H.animal == "PS94") & (H.epoch == e)]
        _points(ax, sub, "run_sp_median", EPOCHS.index(e) - 0.06, "0.25")
    ax.set_xticks(np.arange(len(EPOCHS)))
    ax.set_xticklabels(["Pre", "Acute", "Sub.", "Chron."], fontsize=10)
    ax.set_xlim(-0.5, len(EPOCHS) - 0.4)
    ax.set_ylim(0, None)
    ax.set_ylabel("Median speed in running bouts (mm/s)")
    ax.set_title("PS94: running speed per session", fontsize=11)
    # row 3: PS94 production confusions + speed distribution
    for k, e in enumerate(EPOCHS):
        ax = fig.add_subplot(gs[2, 2 * k:2 * k + 2])
        s = C[(C.animal == "PS94") & (C.epoch == e) & (C.arm == "frozen")].set_index("true").reindex(CL)
        M = s[[f"pred_{b}" for b in CL]].values
        ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] > 0.6 else "0.1",
                        fontweight="bold" if (i == 1 and j != 1 and M[i, j] >= 0.1) else "normal")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Q", "R", "L"])
        ax.set_yticks(range(3))
        ax.set_yticklabels(["Q", "R", "L"] if k == 0 else [])
        ax.set_title(f"PS94 {e} (production)", fontsize=10)
        ax.set_xlabel("Predicted")
        if k == 0:
            ax.set_ylabel("True")
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = fig.add_subplot(gs[2, 8:12])
    cent = 0.5 * (EDGE[1:] + EDGE[:-1])
    keep = cent > 3
    sty = {"pre": ("0.2", "-"), "post": ("#b2182b", "-")}
    for grp, eps in (("pre", ("pre",)), ("post", ("acute", "subacute", "chronic"))):
        hs = [h for a, _l, e, h, _m in hist if a == "PS94" and e in eps]
        Hn = np.mean([h[keep] / max(h[keep].sum(), 1) for h in hs], 0)
        share = np.mean([h[keep].sum() / max(h.sum(), 1) for h in hs])
        ax.plot(cent[keep], Hn / 0.5, color=sty[grp][0], lw=2,
                label=f"{'Pre' if grp == 'pre' else 'Post (all epochs)'}: {len(hs)} sessions, "
                      f"{100 * share:.0f}% of time > 3 mm/s")
    ax.set_xlim(3, 40)
    ax.set_ylim(0, None)
    ax.set_xlabel("Treadmill speed (mm/s), samples above the 3 mm/s running threshold")
    ax.set_ylabel("Density")
    ax.set_title("PS94: running-speed distribution, pre vs post", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    h = [Line2D([], [], ls="none", marker="o", ms=7, mfc=col, mec=col, label=lab) for _a, lab, col in ARMS]
    h += [Line2D([], [], ls="none", marker=MARK[a], mfc="none", mec="0.3", ms=7, label=a) for a in ANIMALS]
    h += [Line2D([], [], ls=":", color="0.6", label="chance 1/3")]
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.1)
    fig.text(0.5, 0.955, "Open markers = sessions; filled marker +- bar = mean over sessions +- SEM. Q / R / L = "
             "quiet / running / licking.", ha="center", fontsize=9.5, color="0.3")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.07)
    save_png_svg(fig, png, dpi=180, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE, help="state_decoder_ps94_check/cache folder (<animal>.pkl)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        fd = _common.load_figdata(a.out, NAME)
        D, hist, C = fd["D"], fd["hist"], fd["C"]
    else:
        _common.require_explicit(a.cache, CACHE, "--cache")
        D, hist, model_dirs = collect(a.cache)
        D.to_csv(a.out / f"{NAME}.csv", index=False, float_format="%.4f")
        C = confusions(D)
        C.to_csv(a.out / f"{NAME}_confusions.csv", index=False, float_format="%.4f")
        _common.save_figdata(a.out, NAME, {"D": D, "hist": hist, "C": C, "cache": str(a.cache),
                                           "model_dirs": model_dirs})
        pd.set_option("display.width", 220)
        print(D.groupby(["animal", "arm", "epoch"], sort=False)[["balanced_acc", "recall_quiet", "recall_running",
                                                                  "recall_licking"]].mean().round(3).to_string())
    draw(D, hist, C, a.out / f"{NAME}.png")
    print(f"-> {a.out / f'{NAME}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
