"""FaceRhythm: is a post-stroke change in a lick factor FEWER licks or DIFFERENT licks? (10-22 FINDINGS 5 caveat:
absolute expression scales with how much the animal licks.) Computation: `face_rhythm_validation.lick_epoch_session`.

    python scripts/face_rhythm_lick_epochs.py --set pilot_pre_PS92_PS93 [--jobs 3]

-> ``<dlc>/face_rhythm/cam4/validation/<set>/lick_epochs/`` (methods in its README.md, `face_rhythm_figures`):
  licks_per_trial.png    response-window licks per engaged trial by executed direction, per animal x epoch.
  per_lick_by_epoch.png  per factor: (1) expression AT a lick by direction, relative to pre; (2) the factor's
                         response-window expression split into licks / trial x expression per lick, its PREFERRED
                         direction (largest pre per-lick expression), relative to pre; (3) the preferred direction's
                         lick-triggered curve by epoch (each animal's pre peak = 1, then the mean over animals).
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

EPOCHS = ["pre", "acute", "subacute", "chronic"]
DIR_COL = {"left": "tab:blue", "centre": "0.4", "right": "tab:red"}
EPOCH_COL = {"pre": "k", "acute": "tab:red", "subacute": "tab:orange", "chronic": "tab:green"}
ANIMAL_LS = ["-", "--", ":", "-."]


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
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = fro.factor_dir(rv, set_name=a.set)
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
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
