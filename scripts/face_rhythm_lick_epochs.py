"""FaceRhythm: is a post-stroke change in a lick factor FEWER licks or DIFFERENT licks? (10-22 FINDINGS 5 caveat:
absolute expression scales with how much the animal licks.) Computation: `face_rhythm_validation.lick_epoch_session`.

    python scripts/face_rhythm_lick_epochs.py --set master_pre_v1 [--jobs 3] [--stats-only]

-> ``<dlc>/face_rhythm/cam4/validation/<set>/lick_epochs/`` (methods in its README.md, `face_rhythm_figures`):
  licks_per_trial.png    response-window licks per engaged trial by executed direction, per animal x epoch.
  per_lick_by_epoch.png  per factor: (1) expression AT a lick by direction, relative to pre; (2) the factor's
                         response-window expression split into licks / trial x expression per lick, its PREFERRED
                         direction (largest pre per-lick expression), relative to pre; (3) the preferred direction's
                         lick-triggered curve by epoch (each animal's pre peak = 1, then the mean over animals).
  per_lick_stats.png     (Priya 10-24: statistics on the per-lick claim, FINDINGS 5b-master) right / left licks per
                         trial and each factor per lick of its preferred direction, LOG2 vs each animal's pre; per-animal
                         epoch means, `boot_delta` change vs pre, * = 95 % interval excludes 0, ** = also Holm over the
                         12 factors of an epoch (licks per trial: planned, * only). ``--stats-only`` redraws it (and
                         data/per_lick_stats.csv) from data/sessions.csv without recomputing.
  data/                  sessions.csv (one row per session), epochs.csv (per animal x epoch, relative to pre).
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
DIR_COL = {"left": "tab:blue", "centre": "0.4", "right": "tab:red"}
#: log2 floor for licks / trial: a session with no licks of a direction (acute contralesional) has 0 licks / trial --
#: 0.01 keeps it as a large drop without an infinite log. Per-lick expression is NaN there (nothing to average): dropped.
LPT_FLOOR = 0.01

EPOCH_COL = frc.epoch_colors()   # FR epoch palette (orofacial / 3-D kinematics)
ANIMAL_LS = ["-", "--", ":", "-."]


def per_lick_stats(T: pd.DataFrame, names: list[str], pref: list[str], out: Path, set_name: str) -> pd.DataFrame:
    """Statistics for the per-lick claim (FINDINGS 5b-master): per quantity, log2 session value vs the animal's pre
    mean log2, `stats.epoch_change` (per-animal epoch means first, then `boot_delta`), Holm over the 12 per-lick
    factors of an epoch; one figure, per-animal lines (repo animal colours) + mean +- SEM, marks under each epoch."""
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import figure_layout
    T = T.copy()
    T["label"] = [frc.label(a, d) for a, d in zip(T.animal, T.date)]
    q = [("lpt_right", "right (contra) licks / trial", "lpt"), ("lpt_left", "left (ipsi) licks / trial", "lpt")]
    q += [(f"pl_{pref[k]}_f{k}", f"{names[k] or 'f%d' % k} per {pref[k]} lick", "pl") for k in range(len(pref))]
    rng = np.random.default_rng(0)
    rows = []
    for col, title, kind in q:
        d = T[["animal", "epoch", "label", col]].dropna(subset=[col])
        d = d.assign(v=np.maximum(d[col], LPT_FLOOR) if kind == "lpt" else d[col], g=col)
        d = d[d.v > 0].copy()
        d["l"] = frs.log2_vs_pre(d, "v", ["animal", "g"])
        for r in frs.epoch_change(d, "l", rng):
            rows.append({"quantity": col, "title": title, "kind": kind, **r})
    S = pd.DataFrame(rows)
    lpt = S[S.kind == "lpt"]
    S = pd.concat([frs.add_holm(S[S.kind == "pl"], ["epoch"]),
                   lpt.assign(p_holm=np.nan, mark=np.where(lpt.excludes_0, "*", ""))], ignore_index=True)
    S.to_csv(figure_layout.sidecar_for(out, "per_lick_stats", ".csv"), index=False, float_format="%.4g")
    animals = sorted(T.animal.unique())
    pd.set_option("display.width", 220)
    print("PER LICK / LICKS PER TRIAL, log2 vs pre: per animal FIRST (rule 8), then boot_delta (+ Holm over factors)")
    print(S[["quantity", "epoch", *animals, "delta", "ci_lo", "ci_hi", "p_holm", "mark"]].round(3).to_string(index=False))
    acol = frc.animal_colors()
    ncol = 5
    nrow = int(np.ceil(len(q) / ncol))
    x = np.arange(len(EPOCHS))
    with frf.style():
        fig, axs = frf.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False, sharey=True)
        for i, (col, title, _kind) in enumerate(q):
            ax = axs.flat[i]
            st = S[S.quantity == col].set_index("epoch")
            for an in animals:
                y = [0.0] + [st.loc[e, an] if e in st.index and an in st.columns else np.nan for e in EPOCHS[1:]]
                ax.plot(x, y, ".-", color=acol.get(an, "0.6"), lw=0.9, ms=4, alpha=0.85)
            m = [0.0] + [st.loc[e, "mean"] if e in st.index else np.nan for e in EPOCHS[1:]]
            se = [0.0] + [st.loc[e, "sem"] if e in st.index else np.nan for e in EPOCHS[1:]]
            ax.errorbar(x, m, yerr=se, fmt="o", color="k", ms=5, capsize=3, zorder=5)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x, ["pre", *(frs.tick_label(e, st.loc[e] if e in st.index else None, interval=False)
                                       for e in EPOCHS[1:])], fontsize=7)
            red = col == "lpt_right" or title.startswith("lick_right per right")
            ax.set_title(title, fontsize=9, color="tab:red" if red else "k")
            if i % ncol == 0:
                ax.set_ylabel("log2 vs own pre")
        for ax in axs.flat[len(q):]:
            ax.set_visible(False)
        for an in animals:
            axs[0, -1].plot([], [], ".-", color=acol.get(an, "0.6"), label=an)
        frf.legend_outside(axs[0, -1], fontsize=8)
        fig.suptitle(f"{set_name}: fewer licks or different licks -- statistics (* 95 % CI excludes 0; ** also Holm "
                     "over the 12 factors)")
        frf.save(fig, out / "per_lick_stats.png", f"""
Per session (data/sessions.csv): licks per engaged trial by executed direction (LP v36; floored at {LPT_FLOOR} licks /
trial so a session without licks of a direction counts as a large drop) and each factor's expression PER LICK of its
preferred direction (largest pre per-lick expression), as log2 minus the mean log2 of the animal's pre sessions (pre =
0). Coloured: per-animal epoch means (repo animal colours); black: mean +- SEM across animals. Under each epoch: change
vs pre, `analysis_kit.boot_delta` (paired animals -> sessions, animal-weighted); * = 95 % interval excludes 0; ** =
also significant after Holm over the 12 per-lick factors of that epoch (bootstrap p from the same draws). Licks per
trial are planned tests (* only). Red titles: the contralesional quantities of FINDINGS 5b-master.""")
    return S


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")

    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_validation as frv
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="pilot_pre_PS92_PS93", help="factor set (or <bundle>/<set>)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--stats-only", action="store_true", help="per_lick_stats from data/sessions.csv, no recompute")
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = fro.factor_dir(rv, set_name=a.set)
    if a.stats_only:
        out = Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "validation" / fro.set_tag(a.set) / "lick_epochs"
        writeguard.assert_writable(out.parent.parent)
        T = pd.read_csv(figure_layout.sidecar_for(out, "sessions", ".csv", mkdir=False), dtype={"date": str})
        R = len([c for c in T.columns if c.startswith("resp_f")])
        names = [lb.removeprefix("fr_") if not lb[3:].isdigit() else "" for lb in fro.labels(fdir)]
        pre = T[T.epoch == "pre"].groupby("animal").mean(numeric_only=True)
        pref = [max(frv.DIRECTIONS, key=lambda d, k=k: np.nanmean(pre[f"pl_{d}_f{k}"])) for k in range(R)]
        per_lick_stats(T, names, pref, out, a.set)
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
    out = Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "validation" / fro.set_tag(a.set) / "lick_epochs"
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
    nl = {d: int(T[f"n_{d}"].sum()) for d in frv.DIRECTIONS}
    common = (f"Set `{a.set}`; {len(S)} projected sessions ({', '.join(f'{an} ' + '/'.join(str(int(E.loc[(an, e), 'n_sessions'])) if (an, e) in E.index else '0' for e in ep) for an in animals)} "
              f"sessions {'/'.join(ep)}). Licks: v36 LP `licks_all.csv`, executed DIRECTION from the lick's own angle "
              f"(left > +{frv.ANGLE_DEG:.0f} deg, right < -{frv.ANGLE_DEG:.0f} deg, centre between; + = the animal's "
              f"left; all lesions LEFT, so RIGHT = contralesional); licks with onset 0..{frv.RESP_S:.0f} s after an "
              f"engaged trial's cue (n = {nl['left']} left / {nl['centre']} centre / {nl['right']} right). "
              f"Expression = projected temporal weights x FR normaliser D (ABSOLUTE, a.u. on the animal's scale; pre "
              f"sessions on their left-out-day basis). Per session first, then the mean over the animal's sessions of "
              f"the epoch; 'relative to pre' = epoch mean / the same animal's pre mean (1 = unchanged).")

    with frf.style():
        # -------------------------------------------------------------- licks per trial
        fig, axs = frf.subplots(1, 2, figsize=(13, 4.4))
        for i, an in enumerate(animals):
            for d in frv.DIRECTIONS:
                y = [E.loc[(an, e), f"lpt_{d}"] if (an, e) in E.index else np.nan for e in ep]
                yr = [rel.loc[(an, e), f"lpt_{d}"] if (an, e) in rel.index else np.nan for e in ep]
                kw = dict(color=DIR_COL[d], ls=ANIMAL_LS[i % 4], marker="o", ms=4, label=f"{an} {d}")
                axs[0].plot(ep, y, **kw)
                axs[1].plot(ep, yr, **kw)
        axs[0].set(ylabel="licks / engaged trial (0-4 s)", title="licks per trial")
        axs[1].axhline(1, color="k", lw=0.6)
        axs[1].set(ylabel="relative to pre", title="relative to the animal's pre")
        frf.legend_outside(axs[1])
        fig.suptitle(f"{a.set}: response-window licks by direction")
        frf.save(fig, out / "licks_per_trial.png", f"""
{common}
Licks per engaged trial = the epoch's licks of a direction in the response window / its projected engaged trials.""")

        # -------------------------------------------------------------- per factor
        fig, axs = frf.subplots(R, 3, figsize=(19, 3.3 * R), squeeze=False)
        for k in range(R):
            a0, a1, a2 = axs[k]
            nm = f" {names[k]}" if names[k] else ""
            d0 = pref[k]
            for i, an in enumerate(animals):
                ls = ANIMAL_LS[i % 4]
                g = lambda col, an=an: [rel.loc[(an, e), col] if (an, e) in rel.index else np.nan for e in ep]
                for d in frv.DIRECTIONS:
                    a0.plot(ep, g(f"pl_{d}_f{k}"), color=DIR_COL[d], ls=ls, marker="o", ms=4, label=f"{an} {d}")
                a1.plot(ep, g(f"resp_f{k}"), color="k", ls=ls, marker="o", ms=4, label=f"{an} total (0-4 s)")
                a1.plot(ep, g(f"lpt_{d0}"), color="tab:purple", ls=ls, marker="s", ms=4, label=f"{an} {d0} licks / trial")
                a1.plot(ep, g(f"pl_{d0}_f{k}"), color="tab:olive", ls=ls, marker="^", ms=4, label=f"{an} per {d0} lick")
            for e in ep:                                   # lick-triggered, each animal's pre peak = 1, mean over animals
                cs = []
                for an in animals:
                    pre_c = [s[f"lta_{d0}"][:, k] for s in S if s["animal"] == an and s["epoch"] == "pre"]
                    ep_c = [s[f"lta_{d0}"][:, k] for s in S if s["animal"] == an and s["epoch"] == e]
                    if pre_c and ep_c:
                        cs.append(np.nanmean(ep_c, 0) / np.nanmax(np.nanmean(pre_c, 0)))
                if cs:
                    a2.plot(frv.LAGS, np.nanmean(cs, 0), color=EPOCH_COL[e], lw=1.4, label=e)
            a2.axvline(0, color="k", lw=0.5)
            a2.axvspan(0, frv.PER_LICK_S, color="0.9", zorder=0)
            for ax in (a0, a1):
                ax.axhline(1, color="k", lw=0.6)
            a0.set(title=f"factor {k}{nm}: per lick, by direction", ylabel="relative to pre")
            a1.set(title=f"total = licks x per lick ({d0} licks)")
            a2.set(title=f"around {d0} licks", ylabel="pre peak = 1")
            if k == R - 1:
                a2.set_xlabel("s from lick onset")
            if k == 0:
                for ax in (a0, a1, a2):
                    frf.legend_outside(ax, fontsize=8)
        fig.suptitle(f"{a.set}: fewer licks or different licks?")
        frf.save(fig, out / "per_lick_by_epoch.png", f"""
{common}
* **per lick, by direction**: expression AT a lick = mean over 0..+{frv.PER_LICK_S} s after onset (shaded in column 3)
  of the lick-triggered expression (25 Hz trial grid, `lick_triggered`), mean over the session's licks of that
  direction; relative to pre. Below 1 with licks of the same direction = the lick itself moves the factor less.
* **total = licks x per lick**: black = the factor's mean response-window (0-4 s) expression per trial; purple = licks
  per trial of the factor's PREFERRED direction (largest per-lick expression in pre, mean over animals: {', '.join(f'f{k} {pref[k]}' for k in range(R))});
  olive = per-lick expression at those licks. Roughly, total ~ licks x per lick + expression between licks: a total
  drop that follows purple = fewer licks; one that follows olive = different licks.
* **around preferred licks**: the lick-triggered mean by epoch, each animal's pre curve peak = 1, mean over animals.
  Bouts are 6-8 Hz, so the +-1 s window also holds neighbouring licks.""")

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
