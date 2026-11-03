"""FaceRhythm: is a post-stroke change in a lick factor FEWER licks or DIFFERENT licks? (10-22 FINDINGS 5 caveat:
absolute expression scales with how much the animal licks.) Computation: `face_rhythm_validation.lick_epoch_session`.

    python scripts/face_rhythm_lick_epochs.py --set master_pre_v1 [--jobs 3] [--stats-only | --plot-only]

-> ``<dlc>/face_rhythm/cam4/validation/<set>/lick_epochs/`` (methods in its README.md, `face_rhythm_figures`):
  licks_per_trial.png    response-window licks per engaged trial by executed direction, per epoch.
  per_lick_by_epoch.png  per factor: (1) expression AT a lick by direction, relative to pre; (2) the factor's
                         response-window expression split into licks / trial x expression per lick, its PREFERRED
                         direction (largest pre per-lick expression), relative to pre; (3) the preferred direction's
                         lick-triggered curve by epoch (each animal's pre peak = 1, then the mean over animals).
  per_lick_stats.png     (Priya 10-24: statistics on the per-lick claim, FINDINGS 5b-master) right / left licks per
                         trial and each factor per lick of its preferred direction, LOG2 vs each animal's pre; per-animal
                         epoch means, `boot_delta` change vs pre, * = 95 % interval excludes 0, ** = also Holm over the
                         12 factors of an epoch (licks per trial: planned, * only). ``--stats-only`` redraws it (and
                         RECOMPUTES data/per_lick_stats.csv) from data/sessions.csv without reloading sessions.
  data/                  sessions.csv (one row per session), epochs.csv (per animal x epoch, relative to pre),
                         lick_triggered.npz (per-session lick-triggered curves, column 3 of per_lick_by_epoch).
``--plot-only`` redraws all three figures from the STORED tables (sessions.csv, epochs.csv, per_lick_stats.csv): no
statistic recomputed, no CSV written. The lick-triggered curves come from data/lick_triggered.npz (built once from the
projections if absent -- traces only, no table).
Categorical x (directions, quantities, epochs) follows Priya's 11-03 rules (`wfield_local.cat_points`): no lines between
categories, epochs side by side, summary = mean +- SEM over animals, faded OPEN markers = the SESSIONS behind it (shape =
animal, colour = epoch), significance in the top band (`sig_marks.star_top`).
Session means first, then the mean over an animal's sessions of an epoch (each session counts once); per animal, never
pooled before the per-animal table (CLAUDE.md rule 8).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402
from wfield_local.face_rhythm import stats as frs  # noqa: E402

EPOCHS = list(frc.EPOCHS)
DIRECTIONS = ("left", "centre", "right")          # = face_rhythm_validation.DIRECTIONS
DIR_LABEL = {"left": "left\n(ipsi)", "centre": "centre", "right": "right\n(contra)"}
#: log2 floor for licks / trial: a session with no licks of a direction (acute contralesional) has 0 licks / trial --
#: 0.01 keeps it as a large drop without an infinite log. Per-lick expression is NaN there (nothing to average): dropped.
LPT_FLOOR = 0.01

EPOCH_COL = frc.epoch_colors()   # FR epoch palette (orofacial / 3-D kinematics)
CAT_RULE = ("Each category (direction / quantity) holds the epochs side by side, NO lines between categories (Priya "
            "11-03): filled marker = mean +- SEM across the animals' epoch means (epoch colours); faded OPEN markers = "
            "the individual SESSIONS behind it (shape = animal, colour = epoch, small jitter).")


def _per_lick_quantities(names: list[str], pref: list[str]) -> list[tuple]:
    q = [("lpt_right", "right (contra) licks / trial", "lpt"), ("lpt_left", "left (ipsi) licks / trial", "lpt")]
    return q + [(f"pl_{pref[k]}_f{k}", f"{names[k] or 'f%d' % k} per {pref[k]} lick", "pl") for k in range(len(pref))]


def _log2_sessions(T: pd.DataFrame, col: str, kind: str) -> pd.DataFrame:
    """Per session: log2 of ``col`` minus the mean log2 of the animal's pre sessions (column ``l``)."""
    d = T[["animal", "epoch", "label", col]].dropna(subset=[col])
    d = d.assign(v=np.maximum(d[col], LPT_FLOOR) if kind == "lpt" else d[col], g=col)
    d = d[d.v > 0].copy()
    d["l"] = frs.log2_vs_pre(d, "v", ["animal", "g"])
    return d


def per_lick_stats(T: pd.DataFrame, names: list[str], pref: list[str], out: Path, set_name: str,
                   S: pd.DataFrame | None = None) -> pd.DataFrame:
    """Statistics for the per-lick claim (FINDINGS 5b-master): per quantity, log2 session value vs the animal's pre
    mean log2, `stats.epoch_change` (per-animal epoch means first, then `boot_delta`), Holm over the 12 per-lick
    factors of an epoch; one figure. ``S`` given (``--plot-only``: the stored data/per_lick_stats.csv) = draw only."""
    from wfield_local import figure_layout
    T = T.copy()
    T["label"] = [frc.label(a, d) for a, d in zip(T.animal, T.date)]
    q = _per_lick_quantities(names, pref)
    if S is None:
        rng = np.random.default_rng(0)
        rows = []
        for col, title, kind in q:
            for r in frs.epoch_change(_log2_sessions(T, col, kind), "l", rng):
                rows.append({"quantity": col, "title": title, "kind": kind, **r})
        S = pd.DataFrame(rows)
        lpt = S[S.kind == "lpt"]
        S = pd.concat([frs.add_holm(S[S.kind == "pl"], ["epoch"]),
                       lpt.assign(p_holm=np.nan, mark=np.where(lpt.excludes_0, "*", ""))], ignore_index=True)
        S.to_csv(figure_layout.sidecar_for(out, "per_lick_stats", ".csv"), index=False, float_format="%.4g")
    S = S.assign(mark=S["mark"].fillna("").astype(str))
    animals = sorted(T.animal.unique())
    pd.set_option("display.width", 220)
    print("PER LICK / LICKS PER TRIAL, log2 vs pre: per animal FIRST (rule 8), then boot_delta (+ Holm over factors)")
    print(S[["quantity", "epoch", *animals, "delta", "ci_lo", "ci_hi", "p_holm", "mark"]].round(3).to_string(index=False))
    plot_per_lick_stats(T, S, q, out, set_name)
    return S


def plot_per_lick_stats(T: pd.DataFrame, S: pd.DataFrame, q: list[tuple], out: Path, set_name: str) -> None:
    """per_lick_stats.png: x = epoch (categories, no connecting lines); faded open markers = sessions (shape = animal,
    colour = epoch), filled = mean +- SEM over animals; * / ** in the top band."""
    from wfield_local import cat_points as cp
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import sig_marks
    animals = sorted(T.animal.unique())
    ncol = 5
    nrow = int(np.ceil(len(q) / ncol))
    x = np.arange(len(EPOCHS))
    with frf.style():
        fig, axs = frf.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False, sharey=True)
        for i, (col, title, kind) in enumerate(q):
            ax = axs.flat[i]
            st = S[S.quantity == col].set_index("epoch")
            d = _log2_sessions(T, col, kind)
            for e_i, e in enumerate(EPOCHS):
                de = d[d.epoch == e]
                cp.session_dots(ax, e_i, de.l.to_numpy(), de.animal.to_numpy(), EPOCH_COL[e], jitter=0.12,
                                seed=97 * i + e_i)
                if e == "pre":
                    m, se = 0.0, None                       # pre = 0 by construction (each animal's own pre mean)
                elif e in st.index:
                    m, se = st.loc[e, "mean"], st.loc[e, "sem"]
                    if st.loc[e, "mark"]:
                        sig_marks.star_top(ax, e_i, st.loc[e, "mark"], color=EPOCH_COL[e])
                else:
                    continue
                cp.summary_point(ax, e_i, m, se, EPOCH_COL[e], ms=6)
            ax.axhline(0, color="0.6", lw=0.6, zorder=0)
            ax.set_xlim(-0.5, len(EPOCHS) - 0.5)
            ax.set_xticks(x, ["pre", *(f"{e[:5]}\n{st.loc[e, 'delta']:+.2f}" if e in st.index else e[:5]
                                       for e in EPOCHS[1:])], fontsize=7)
            red = col == "lpt_right" or title.startswith("lick_right per right")
            ax.set_title(title, fontsize=9, color="tab:red" if red else "k")
            if i % ncol == 0:
                ax.set_ylabel("log2 vs own pre")
        for ax in axs.flat[len(q):]:
            ax.set_visible(False)
        lg = axs[0, -1]
        for e in EPOCHS:
            lg.plot([], [], "o", color=EPOCH_COL[e], mec="w", ms=6, label=f"{e}: mean +- SEM")
        cp.add_animal_entries(lg, animals)
        frf.legend_outside(lg, fontsize=8)
        sig_marks.reserve_headroom(fig)
        fig.suptitle(f"{set_name}: fewer licks or different licks -- statistics (* 95 % CI excludes 0; ** also Holm "
                     "over the 12 factors)")
        frf.save(fig, out / "per_lick_stats.png", f"""
Per session (data/sessions.csv): licks per engaged trial by executed direction (LP v36; floored at {LPT_FLOOR} licks /
trial so a session without licks of a direction counts as a large drop) and each factor's expression PER LICK of its
preferred direction (largest pre per-lick expression), as log2 minus the mean log2 of the animal's pre sessions (pre =
0). Each epoch is its own category (Priya 11-03: no lines between epochs): filled marker = mean +- SEM across the
animals' epoch means (epoch colours; pre = 0 by construction); faded OPEN markers = the individual SESSIONS behind it
(shape = animal, colour = epoch, small jitter). Under each epoch: change vs pre, `analysis_kit.boot_delta` (paired
animals -> sessions, animal-weighted); top band: * = 95 % interval excludes 0; ** = also significant after Holm over the
12 per-lick factors of that epoch (bootstrap p from the same draws). Licks per trial are planned tests (* only). Red
titles: the contralesional quantities of FINDINGS 5b-master. Statistics: data/per_lick_stats.csv.""")


def _cat_panel(ax, cats: list, epochs: list[str], sess: dict, summ: dict, seed: int = 0, label: bool = False) -> None:
    """One categorical panel: x = ``cats``, epochs side by side (`cat_points.epoch_offsets`). ``sess[(c, e)]`` =
    (values, animals) of the sessions; ``summ[(c, e)]`` = per-animal epoch means (array). Summary = mean +- SEM over
    animals; sessions = faded open markers (shape = animal, colour = epoch)."""
    from wfield_local import cat_points as cp
    off = cp.epoch_offsets(len(epochs), step=0.2)
    for c_i, c in enumerate(cats):
        for e_i, e in enumerate(epochs):
            xo = c_i + off[e_i]
            v, an = sess.get((c, e), ([], []))
            cp.session_dots(ax, xo, v, an, EPOCH_COL[e], jitter=0.035, seed=seed + 13 * c_i + e_i)
            pa = np.asarray(summ.get((c, e), []), float)
            pa = pa[np.isfinite(pa)]
            if not len(pa):
                continue
            se = float(pa.std(ddof=1) / np.sqrt(len(pa))) if len(pa) > 1 else None
            cp.summary_point(ax, xo, float(pa.mean()), se, EPOCH_COL[e], ms=5.5,
                             label=f"{e}: mean +- SEM" if label and c_i == 0 else None)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    for c_i in range(1, len(cats)):
        ax.axvline(c_i - 0.5, color="0.88", lw=0.8, zorder=0)


def _session_rel(T: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Session values / the same animal's mean over its pre sessions (= what epochs.csv's rel_* average)."""
    pre = T[T.epoch == "pre"].groupby("animal")[cols].mean()
    return T[cols].div(pre.reindex(T.animal).to_numpy())


def lta_table(S: list[dict]) -> dict:
    """Per-session lick-triggered curves (n_sessions, LAGS, R) by direction, with animal / date / epoch."""
    out = {"animal": np.array([s["animal"] for s in S]), "date": np.array([str(s["date"]) for s in S]),
           "epoch": np.array([s["epoch"] for s in S])}
    out.update({f"lta_{d}": np.stack([s[f"lta_{d}"] for s in S]) for d in DIRECTIONS})
    return out


def plot_licks_per_trial(T, E, rel, ep, animals, out, set_name, common) -> None:
    from wfield_local import cat_points as cp
    from wfield_local import face_rhythm_figures as frf
    cols = [f"lpt_{d}" for d in DIRECTIONS]
    Tr = _session_rel(T, cols)
    sess_abs, sess_rel, summ_abs, summ_rel = {}, {}, {}, {}
    for d, c in zip(DIRECTIONS, cols):
        for e in ep:
            m = (T.epoch == e).to_numpy()
            sess_abs[(d, e)] = (T.loc[m, c].to_numpy(), T.animal[m].to_numpy())
            sess_rel[(d, e)] = (Tr.loc[m, c].to_numpy(), T.animal[m].to_numpy())
            summ_abs[(d, e)] = E.xs(e, level="epoch")[c].to_numpy() if e in E.index.get_level_values("epoch") else []
            summ_rel[(d, e)] = rel.xs(e, level="epoch")[c].to_numpy() if e in rel.index.get_level_values("epoch") else []
    with frf.style():
        fig, axs = frf.subplots(1, 2, figsize=(12, 4.4))
        _cat_panel(axs[0], list(DIRECTIONS), ep, sess_abs, summ_abs, seed=1, label=True)
        _cat_panel(axs[1], list(DIRECTIONS), ep, sess_rel, summ_rel, seed=2)
        for ax in axs:
            ax.set_xticks(range(len(DIRECTIONS)), [DIR_LABEL[d] for d in DIRECTIONS])
        axs[0].set(ylabel="licks / engaged trial (0-4 s)", title="licks per trial")
        axs[1].axhline(1, color="0.6", lw=0.6, zorder=0)
        axs[1].set(ylabel="relative to the animal's pre", title="relative to the animal's pre")
        cp.add_animal_entries(axs[0], animals)
        frf.legend_outside(axs[1], handles=axs[0].get_legend_handles_labels()[0], fontsize=8)
        fig.suptitle(f"{set_name}: response-window licks by direction")
        frf.save(fig, out / "licks_per_trial.png", f"""
{common}
Licks per engaged trial = the epoch's licks of a direction in the response window / its projected engaged trials.
{CAT_RULE} Right: each session / the animal's pre mean.""")


def plot_per_lick_by_epoch(T, E, rel, ep, animals, names, pref, lta, out, set_name, common) -> None:
    from wfield_local import cat_points as cp
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_validation as frv
    R = len(pref)
    rcols = [c for c in T.columns if c.startswith(("lpt_", "pl_", "resp_"))]
    Tr = _session_rel(T, rcols)
    an_t = T.animal.to_numpy()

    def cats(cols):
        sess, summ = {}, {}
        for c in cols:
            for e in ep:
                m = (T.epoch == e).to_numpy()
                sess[(c, e)] = (Tr.loc[m, c].to_numpy(), an_t[m])
                summ[(c, e)] = rel.xs(e, level="epoch")[c].to_numpy() if e in rel.index.get_level_values("epoch") \
                    else []
        return sess, summ
    with frf.style():
        fig, axs = frf.subplots(R, 3, figsize=(19, 3.3 * R), squeeze=False)
        for k in range(R):
            a0, a1, a2 = axs[k]
            nm = f" {names[k]}" if names[k] else ""
            d0 = pref[k]
            c0 = [f"pl_{d}_f{k}" for d in DIRECTIONS]
            _cat_panel(a0, c0, ep, *cats(c0), seed=100 * k, label=k == 0)
            a0.set_xticks(range(3), [DIR_LABEL[d] for d in DIRECTIONS], fontsize=8)
            c1 = [f"resp_f{k}", f"lpt_{d0}", f"pl_{d0}_f{k}"]
            _cat_panel(a1, c1, ep, *cats(c1), seed=100 * k + 50)
            a1.set_xticks(range(3), ["total\n(0-4 s)", f"{d0} licks\n/ trial", f"per {d0}\nlick"], fontsize=8)
            for e in ep:                                   # lick-triggered, each animal's pre peak = 1, mean over animals
                cs = []
                for an in animals:
                    sel_pre = (lta["animal"] == an) & (lta["epoch"] == "pre")
                    sel_ep = (lta["animal"] == an) & (lta["epoch"] == e)
                    if sel_pre.any() and sel_ep.any():
                        pre_c = lta[f"lta_{d0}"][sel_pre][:, :, k]
                        ep_c = lta[f"lta_{d0}"][sel_ep][:, :, k]
                        cs.append(np.nanmean(ep_c, 0) / np.nanmax(np.nanmean(pre_c, 0)))
                if cs:
                    a2.plot(frv.LAGS, np.nanmean(cs, 0), color=EPOCH_COL[e], lw=1.4, label=e)
            a2.axvline(0, color="k", lw=0.5)
            a2.axvspan(0, frv.PER_LICK_S, color="0.9", zorder=0)
            for ax in (a0, a1):
                ax.axhline(1, color="0.6", lw=0.6, zorder=0)
            a0.set(title=f"factor {k}{nm}: per lick, by direction", ylabel="relative to pre")
            a1.set(title=f"total = licks x per lick ({d0} licks)")
            a2.set(title=f"around {d0} licks", ylabel="pre peak = 1")
            if k == R - 1:
                a2.set_xlabel("s from lick onset")
            if k == 0:
                cp.add_animal_entries(a0, animals)
                frf.legend_outside(a0, fontsize=8)
                frf.legend_outside(a2, fontsize=8)
        fig.suptitle(f"{set_name}: fewer licks or different licks?")
        frf.save(fig, out / "per_lick_by_epoch.png", f"""
{common}
* **per lick, by direction**: expression AT a lick = mean over 0..+{frv.PER_LICK_S} s after onset (shaded in column 3)
  of the lick-triggered expression (25 Hz trial grid, `lick_triggered`), mean over the session's licks of that
  direction; relative to pre. Below 1 with licks of the same direction = the lick itself moves the factor less.
* **total = licks x per lick**: the factor's mean response-window (0-4 s) expression per trial; licks per trial of the
  factor's PREFERRED direction (largest per-lick expression in pre, mean over animals: {', '.join(f'f{k} {pref[k]}' for k in range(R))});
  per-lick expression at those licks. Roughly, total ~ licks x per lick + expression between licks: a total drop that
  follows the licks / trial category = fewer licks; one that follows per lick = different licks.
* columns 1-2: {CAT_RULE} Sessions are each session / the animal's pre mean.
* **around preferred licks**: the lick-triggered mean by epoch, each animal's pre curve peak = 1, mean over animals
  (data/lick_triggered.npz). Bouts are 6-8 Hz, so the +-1 s window also holds neighbouring licks.""")


def _common(set_name, T, E, ep, animals) -> str:
    from wfield_local import face_rhythm_validation as frv
    nl = {d: int(T[f"n_{d}"].sum()) for d in DIRECTIONS}
    ns = ", ".join(f"{an} " + "/".join(str(int(E.loc[(an, e), "n_sessions"])) if (an, e) in E.index else "0"
                                       for e in ep) for an in animals)
    return (f"Set `{set_name}`; {len(T)} projected sessions ({ns} "
            f"sessions {'/'.join(ep)}). Licks: v36 LP `licks_all.csv`, executed DIRECTION from the lick's own angle "
            f"(left > +{frv.ANGLE_DEG:.0f} deg, right < -{frv.ANGLE_DEG:.0f} deg, centre between; + = the animal's "
            f"left; all lesions LEFT, so RIGHT = contralesional); licks with onset 0..{frv.RESP_S:.0f} s after an "
            f"engaged trial's cue (n = {nl['left']} left / {nl['centre']} centre / {nl['right']} right). "
            f"Expression = projected temporal weights x FR normaliser D (ABSOLUTE, a.u. on the animal's scale; pre "
            f"sessions on their left-out-day basis). Per session first, then the mean over the animal's sessions of "
            f"the epoch; 'relative to pre' = epoch mean / the same animal's pre mean (1 = unchanged).")


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")

    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_validation as frv
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="pilot_pre_PS92_PS93", help="factor set (or <bundle>/<set>)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--stats-only", action="store_true", help="per_lick_stats from data/sessions.csv, no recompute")
    ap.add_argument("--plot-only", action="store_true",
                    help="all three figures from the stored data/*.csv (+ lick_triggered.npz); no statistics recomputed")
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = fro.factor_dir(rv, set_name=a.set)
    out = Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "validation" / fro.set_tag(a.set) / "lick_epochs"
    writeguard.assert_writable(out.parent.parent)
    lta_f = figure_layout.sidecar_for(out, "lick_triggered", ".npz", mkdir=False)
    if a.stats_only or a.plot_only:
        T = pd.read_csv(figure_layout.sidecar_for(out, "sessions", ".csv", mkdir=False), dtype={"date": str})
        R = len([c for c in T.columns if c.startswith("resp_f")])
        pre = T[T.epoch == "pre"].groupby("animal").mean(numeric_only=True)
        pref = [max(frv.DIRECTIONS, key=lambda d, k=k: np.nanmean(pre[f"pl_{d}_f{k}"])) for k in range(R)]
        if a.stats_only:
            names = [lb.removeprefix("fr_") if not lb[3:].isdigit() else "" for lb in fro.labels(fdir)]
            per_lick_stats(T, names, pref, out, a.set)
            return 0
        labels = fro.labels(fdir)
        names = [lb if not lb.startswith("fr_") or not lb[3:].isdigit() else "" for lb in labels]
        Eall = pd.read_csv(figure_layout.sidecar_for(out, "epochs", ".csv", mkdir=False), index_col=[0, 1])
        vals = [c for c in T.columns if c.startswith(("lpt_", "pl_", "resp_", "n_"))]
        E = Eall[vals + ["n_sessions"]]
        rel = Eall[[f"rel_{v}" for v in vals]].rename(columns=lambda c: c.removeprefix("rel_"))
        ep = [e for e in EPOCHS if e in set(T.epoch)]
        animals = sorted(T.animal.unique())
        if lta_f.exists():
            lta = dict(np.load(lta_f, allow_pickle=False))
        else:                                    # traces only: the per-session curves were never stored
            sess = sorted({tuple(f.stem.split("_")[:2]) for f in (fdir / "proj").glob("*.npz")})
            res, fails = ak.fan_sessions([(str(fdir), an, d) for an, d in sess], frv.lick_epoch_session,
                                         jobs=a.jobs, key=lambda j: (j[1], j[2]))
            if fails:
                sys.exit("failed sessions: " + "; ".join(f"{j[1]}_{j[2]}: {e}" for j, e in fails))
            S = [v for _, v in res]
            chk = pd.DataFrame({"animal": [s["animal"] for s in S], "date": [str(s["date"]) for s in S],
                                **{f"lpt_{d}": [s[f"licks_per_trial_{d}"] for s in S] for d in DIRECTIONS}})
            m = T.merge(chk, on=["animal", "date"], suffixes=("", "_new"))
            dev = max(float(np.nanmax(np.abs(m[f"lpt_{d}"] - m[f"lpt_{d}_new"]) / np.maximum(m[f"lpt_{d}"].abs(), 1e-9)))
                      for d in DIRECTIONS)
            print(f"lick-triggered curves rebuilt for {len(S)} sessions ({len(m)} matched to sessions.csv; max rel. "
                  f"deviation of licks / trial vs the stored table {dev:.2e})")
            lta = lta_table(S)
            np.savez(figure_layout.sidecar_for(out, "lick_triggered", ".npz"), **lta)
        common = _common(a.set, T, E, ep, animals)
        plot_licks_per_trial(T, E, rel, ep, animals, out, a.set, common)
        plot_per_lick_by_epoch(T, E, rel, ep, animals, names, pref, lta, out, a.set, common)
        S_stats = pd.read_csv(figure_layout.sidecar_for(out, "per_lick_stats", ".csv", mkdir=False))
        per_lick_stats(T, [n.removeprefix("fr_") for n in names], pref, out, a.set, S=S_stats)
        print(f"-> {out}")
        return 0
    sess = sorted({tuple(f.stem.split("_")[:2]) for f in (fdir / "proj").glob("*.npz")})
    res, fails = ak.fan_sessions([(str(fdir), an, d) for an, d in sess], frv.lick_epoch_session, jobs=a.jobs,
                                 key=lambda j: (j[1], j[2]))
    if fails:
        sys.exit("failed sessions (refusing a partial table): " + "; ".join(f"{j[1]}_{j[2]}: {e}" for j, e in fails))
    S = [v for _, v in res]
    R = len(S[0]["resp_expr"])
    labels = fro.labels(fdir)
    names = [lb if not lb.startswith("fr_") or not lb[3:].isdigit() else "" for lb in labels]
    writeguard.assert_writable(out.parent.parent)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ tables: session rows, then animal x epoch
    rows = []
    for s in S:
        r = {"animal": s["animal"], "date": s["date"], "epoch": s["epoch"], "n_trials": s["n_trials"]}
        for d in frv.DIRECTIONS:
            r[f"n_{d}"] = s[f"n_{d}"]
            r[f"lpt_{d}"] = s[f"licks_per_trial_{d}"]
            r.update({f"pl_{d}_f{k}": s[f"per_lick_{d}"][k] for k in range(R)})
        r.update({f"resp_f{k}": s["resp_expr"][k] for k in range(R)})
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(figure_layout.sidecar_for(out, "sessions", ".csv"), index=False, float_format="%.5g")
    np.savez(figure_layout.sidecar_for(out, "lick_triggered", ".npz"), **lta_table(S))
    vals = [c for c in T.columns if c.startswith(("lpt_", "pl_", "resp_", "n_"))]
    E = T.groupby(["animal", "epoch"])[vals].mean()
    E["n_sessions"] = T.groupby(["animal", "epoch"]).size()
    rel = E[vals].div(E[vals].xs("pre", level="epoch"), level="animal")
    animals = sorted(T.animal.unique())
    ep = [e for e in EPOCHS if e in set(T.epoch)]
    (rel.add_prefix("rel_").join(E).reindex(ep, level="epoch")
     .to_csv(figure_layout.sidecar_for(out, "epochs", ".csv"), float_format="%.4g"))
    # preferred direction per factor: largest per-lick expression in pre, mean over animals of the animal's pre mean
    pre = E.xs("pre", level="epoch")
    pref = [max(frv.DIRECTIONS, key=lambda d, k=k: np.nanmean(pre[f"pl_{d}_f{k}"])) for k in range(R)]
    common = _common(a.set, T, E, ep, animals)
    plot_licks_per_trial(T, E, rel, ep, animals, out, a.set, common)
    plot_per_lick_by_epoch(T, E, rel, ep, animals, names, pref, lta_table(S), out, a.set, common)

    pd.set_option("display.width", 200)
    show = [f"lpt_{d}" for d in frv.DIRECTIONS] + [f"pl_{pref[k]}_f{k}" for k in range(R)] + \
        [f"resp_f{k}" for k in range(R)]
    print(rel[show].reindex(ep, level="epoch").round(2).to_string())
    print("preferred direction:", {f"f{k}": pref[k] for k in range(R)})
    per_lick_stats(T, [n.removeprefix("fr_") for n in names], pref, out, a.set)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
