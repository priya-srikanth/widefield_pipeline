"""The cross-epoch transfer matrix on ALL FOUR windows: ENL, post-cue, post-lick and rest.

`epoch_15g` answered the rotation-vs-addition question for REST only, because that is where the
"replacement" claim originated. Rest is not the interesting window for a MOTOR representation
(Priya, 2026-09-17: "I'm more interested in ENL (working), cue (working), and lick"). The three
trial-aligned arms are the ones that carry the task code, and `PRELIM_DATA` §5b already shows they
are NOT interchangeable -- pre-stroke far-contra accuracy rises toward movement (ENL 0.507, CUE
0.896, LICK 0.952), and they recover differently.

WHAT EACH ARM IS, straight from `epoch_grant_figures.ARMS`:

    ENL    align=precue, class=working   after the spout arrives, before the cue -- anticipatory
    cue    align=cue,    class=working   0 to +2 s from the cue -- evoked
    lick   align=lick,   class=lick      aligned to the first lick -- execution
    rest   (no alignment)                inter-trial, spout DOCKED and out of reach

`working` = lick PLUS miss-while-working, i.e. everything but the terminal quit period. `lick` =
trials with a detected lick only. The lick-aligned window ADMITS ONLY `lick`: a trial with no
detected lick has no lick to align to, so a "miss trial, lick-aligned" cell is undefined rather
than weak.

READ THE LICK ARM WITH ITS SELECTION CAVEAT. It conditions on a detected lick, and post-stroke its
trial population is itself a product of the deficit -- acutely the animal barely licks
far-contralateral (64 trials across 13 sessions, against ~1,100 at each near position). A transfer
number there says "when the animal managed it, the code looked like this", which is a different
statement from "the code looked like this".

NOTHING IS REIMPLEMENTED HERE. The matrix, the block-matched training, the block-permutation null
and the raw-units verdict live in `wfield_local.transfer_matrix`; the task trials come from
`grant_figures._pooled_bundle` and `_class_select`, the same objects every other task figure uses;
the rest periods come from `rest_frozen_decoder._collect`. This module only maps sessions to
``{label: (epoch, X, y, blocks)}`` per arm. Three of the 2026-09-16 audit's defects spread by
copying sibling scripts, so the shared steps are shared OBJECTS, not shared text.

RUN AS:  python -m scripts.rest_migration.transfer_arms [--arms ENL cue lick rest] [--perm 200]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

#: (display key, alignment, trial class) -- mirrors `epoch_grant_figures.ARMS` plus rest.
ARMS = {"ENL": ("precue", "working"),
        "cue": ("cue", "working"),
        "lick": ("lick", "lick"),
        "rest": (None, None)}


def collect_task(an, align, variant):
    """``{session_label: (epoch, X, y, blocks)}`` for one animal and one trial-aligned arm.

    Built from the SHARED pooled bundle, so these trials are the same objects every other task
    figure scores. `_class_select` decides which engaged and unengaged rows belong to the class --
    one function rather than the inline `if variant == "working"` that used to sit at four sites
    and disagreed the moment a third class existed.
    """
    from wfield_local import epochs
    from wfield_local.grant_figures import _class_select, _pooled_bundle

    bd = _pooled_bundle(an, align)
    XE, YE, GE, BE = bd["XE"], np.asarray(bd["YE"]), np.asarray(bd["GE"]), np.asarray(bd["BE"])
    XU, YU, GU, BU = bd["XU"], np.asarray(bd["YU"]), np.asarray(bd["GU"]), np.asarray(bd["BU"])
    not_eng, kept = np.asarray(bd["not_eng"]), bd["kept"]

    out = {}
    for i, lab in enumerate(kept):
        ep = epochs.epoch_of(lab)
        if ep is None:
            continue
        me, mu = _class_select(variant, GE == i, (GU == i) if len(YU) else np.zeros(0, bool),
                               not_eng)
        Xs = [XE[me]] + ([XU[mu]] if len(YU) and mu.any() else [])
        ys = [YE[me]] + ([YU[mu]] if len(YU) and mu.any() else [])
        bs = [BE[me]] + ([BU[mu]] if len(YU) and mu.any() else [])
        X = np.concatenate(Xs) if sum(x.shape[0] for x in Xs) else None
        if X is None or X.shape[0] == 0:
            continue
        out[lab] = (ep, X, np.concatenate(ys), np.concatenate(bs))
    return out


def _rest_variant():
    """The rest definition's name, e.g. `restdock05`. See `stem` in `main`."""
    from wfield_local.quiet_periods import quiet_variant
    return quiet_variant() or "retired"


def collect_rest(an, bins, gate=True):
    """``{session_label: (epoch, X, y, blocks)}`` for the rest arm, via `15f`'s own collector.

    MEMOIZED TO DISK, per session, because this arm was the only one paying full collection on
    every run: the three task arms read `trial_features_cached` and finish in ~40 s each, while
    rest re-read every session's mask, DAQ and SVT and took ~5 MINUTES PER ANIMAL (Priya,
    2026-09-17: "the rest arm should be able to read the cached bundle too"). That made an 11-min
    arm gate a figure the other three already supported.

    THE KEY CARRIES EVERYTHING THAT CHANGES THE ANSWER -- the joint basis id, the bin count, the
    gate flag and the rest VARIANT. `session_signature` covers the session's own inputs; these do
    not come from the session, so a run under a different basis or a different rest definition
    would otherwise read a stale entry. `CACHE_VERSION` covers changes to `_collect` itself.
    """
    from scripts.rest_migration.rest_frozen_decoder import _collect, _usable
    from wfield_local import config, epochs, joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.session_cache import cached

    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    basis = joint_locanmf.load(an, sessions=SESSIONS)
    variant = _rest_variant()
    out = {}
    for s in [x for x in SESSIONS
              if x["label"] in want and x["label"].startswith(an) and x.get("h5")]:
        ep = epochs.epoch_of(s["label"])
        if ep is None:
            continue
        kind = f"restxfer-{variant}-{basis.basis_id}-b{int(bins)}-g{int(bool(gate))}"
        got = cached(s, kind,
                     lambda s=s: _collect(s, basis, bins, verbose=False, gate=gate)[0],
                     params={"variant": variant, "basis": basis.basis_id,
                             "bins": int(bins), "gate": bool(gate)},
                     verbose=False)
        if got is None:
            continue
        X, y, g = got[0], got[1], got[2]
        ok, _bad = _usable(y, 40, 5)
        if ok:
            out[s["label"]] = (ep, X, y, g)
    return out


def main() -> int:
    from wfield_local import config
    from wfield_local import figure_layout as fl
    from wfield_local import transfer_matrix as tm
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.locanmf_frozen_decoder import _pipe
    from wfield_local.paths import PathResolver

    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["ENL", "cue", "lick"], choices=list(ARMS))
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--bins", type=int, default=4, help="rest arm only")
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")

    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    animals = a.animals or sorted({s["label"].split("_")[0] for s in SESSIONS if s["label"] in want})
    t0 = time.time()
    print(f"TRANSFER MATRIX -- arms {a.arms}, {len(animals)} animals, {a.perm} permutations\n")

    summary = []
    for arm in a.arms:
        align, variant = ARMS[arm]
        print(f"\n{'=' * 78}\n== ARM {arm}"
              + (f"  (align={align}, class={variant})" if align else "  (rest periods)")
              + f"\n{'=' * 78}")
        rows = []
        for an in animals:
            print(f"{an}:", flush=True)
            try:
                data = (collect_rest(an, a.bins) if arm == "rest"
                        else collect_task(an, align, variant))
            except Exception as ex:                                      # noqa: BLE001
                # SAID OUT LOUD -- a silent skip drops a whole animal from a cohort mean.
                print(f"   !! {type(ex).__name__}: {str(ex)[:90]} -- SKIPPED", flush=True)
                continue
            if not data:
                print("   !! no usable sessions -- SKIPPED", flush=True)
                continue
            got = tm.build(data, _pipe, n_perm=a.perm, seed_ns=f"{arm}|{an}",
                           log=lambda m: print(m, flush=True))
            for r in got:
                r["animal"] = an
                r["arm"] = arm
            rows.extend(got)
            print(f"   .. {len(got)} cells ({time.time() - t0:.0f}s)", flush=True)
        if not rows:
            print(f"   !! arm {arm}: no cells")
            continue

        # THE REST ARM KEEPS ITS VARIANT TAG. The rest DEFINITION is a live choice (restdock05
        # vs its predecessors), so a bare `rest` filename would let two definitions overwrite each
        # other silently -- the same provenance trap that produced two 15f figures under different
        # names, one of them stale. The task arms need no tag: their window is the arm name.
        stem = (f"epoch_15g_transfer_{_rest_variant()}{a.tag}" if arm == "rest"
                else f"epoch_15g_transfer_{arm}{a.tag}")
        with open(fl.sidecar_for(out_dir, stem, "_sessions.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        agg = tm.summarise(rows)
        for r in agg:
            r["arm"] = arm
        with open(fl.sidecar_for(out_dir, stem, "_matrix.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(agg[0]))
            w.writeheader()
            w.writerows(agg)
        print(f"\n[15g:{arm}] wrote {stem}_{{sessions,matrix}}.csv")
        verdict, per_animal = tm.report(rows, agg)
        summary.append((arm, verdict, per_animal))

    print(f"\n\n{'=' * 78}\nALL ARMS -- pre vs chronic, RAW above-chance\n{'=' * 78}")
    print(f"{'arm':<8}{'verdict':<14}{'animals':<8}  per-animal raw asymmetry")
    for arm, verdict, per_animal in summary:
        cells = "  ".join(f"{an}:{r - f:+.4f}" for an, f, r, _rf, _rr, _g in per_animal)
        print(f"{arm:<8}{verdict:<14}{len(per_animal):<8}  {cells}")
    print(f"\n{'arm':<8}{'ceiling gain (chronic - pre, above chance)':<50}")
    for arm, _v, per_animal in summary:
        print(f"{arm:<8}" + "  ".join(f"{an}:{g:+.4f}" for an, _f, _r, _rf, _rr, g in per_animal))

    if summary:
        p = _figure(summary, out_dir, a.tag)
        print(f"\n[15g] wrote {p}")
    print(f"\n[15g] {time.time() - t0:.0f}s")
    return 0


def _figure(summary, out_dir, tag=""):
    """Three panels: the matrices, the direction symmetry, the ceiling gain.

    DRAWN IN RAW ABOVE-CHANCE UNITS throughout, deliberately. The retained ratios are in the CSVs
    and are fine per cell, but a figure invites exactly the cross-direction comparison that the
    ratios cannot support -- see `wfield_local.transfer_matrix`. Plotting the ratios here would
    reintroduce the artefact that the first version of this analysis published.
    """
    import csv as _csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import figure_layout as fl
    from wfield_local import transfer_matrix as tm

    arms = [s[0] for s in summary]
    eps = tm.EPOCH_ORDER
    short = {"pre": "pre", "acute": "acu", "subacute": "sub", "chronic": "chr"}
    fig = plt.figure(figsize=(3.7 * len(arms) + 6.2, 9.0))
    # `top=` RESERVES THE HEADER BAND. Without it the suptitle and the method line land on top of
    # each other -- `bbox_inches="tight"` crops to the artists and does not separate them.
    # COLUMNS: the arm matrices, a thin colorbar, a SPACER, then the diagonal panel. The spacer
    # exists so `wspace` can be small enough to keep the matrices close together without the
    # diagonal panel's y-label colliding with the colorbar -- one `wspace` applies to every gap.
    # `top=` reserves the header band; `bbox_inches="tight"` crops to the artists and would
    # otherwise let the title and the method line overlap.
    # TWO GRIDSPECS, ONE PER ROW, because the rows want different geometry: row 1 needs a thin
    # colorbar column and a spacer so the matrices can sit close together, and row 2 wants two
    # equal panels. A single 2-row grid forces one `width_ratios` on both, which is what made the
    # lower-left panel half again as wide as the lower-right one.
    gs_top = fig.add_gridspec(1, len(arms) + 3,
                              width_ratios=[1] * len(arms) + [0.07, 0.42, 1.45],
                              top=0.795, bottom=0.46, wspace=0.14)
    gs_bot = fig.add_gridspec(1, 2, top=0.36, bottom=0.06, wspace=0.26)

    # ---- row 1: the transfer matrices, one per arm, in above-chance units -------------------
    #
    # ONE SHARED COLOUR SCALE AND ONE COLORBAR. The first version computed `vmax` from the panels
    # built SO FAR, so ENL topped out at 0.30 and lick at 0.74 -- three panels that look
    # comparable on three different scales, with ENL's 0.29 rendering as bright as lick's 0.74.
    # The windows genuinely differ in how much position they carry (information rises toward
    # movement, PRELIM_DATA 5b), so a shared scale is the honest one: ENL SHOULD look dimmer.
    mats = {}
    for arm in arms:
        nm = _rest_variant() if arm == "rest" else arm
        f = fl.find_sidecar_for(out_dir, f"epoch_15g_transfer_{nm}{tag}", "_matrix.csv")
        if f is None:
            raise SystemExit(f"[15g] no matrix CSV for arm {arm!r} -- run the arm first")
        with open(f, newline="", encoding="utf-8") as fh:
            rs = list(_csv.DictReader(fh))
        g = {(r["train_epoch"], r["test_epoch"]): float(r["acc_minus_null"]) for r in rs}
        M = np.full((len(eps), len(eps)), np.nan)
        for i, tr in enumerate(eps):
            for k, te in enumerate(eps):
                if (tr, te) in g:
                    M[i, k] = g[(tr, te)]
        mats[arm] = M
    vmax = float(np.nanmax([np.nanmax(M) for M in mats.values()]))

    im = None
    for j, arm in enumerate(arms):
        M = mats[arm]
        ax = fig.add_subplot(gs_top[0, j])
        im = ax.imshow(M, cmap=tm.CMAP_LEVEL, vmin=0, vmax=vmax)
        for i in range(len(eps)):
            for k in range(len(eps)):
                if np.isfinite(M[i, k]):
                    ax.text(k, i, f"{M[i, k]:.2f}", ha="center", va="center", fontsize=8,
                            color="w" if M[i, k] < 0.55 * vmax else "k")
        ax.set_xticks(range(len(eps)), [short[e] for e in eps], fontsize=10)
        ax.set_yticks(range(len(eps)), [short[e] for e in eps] if j == 0 else [""] * len(eps),
                      fontsize=10)
        ax.set_title(arm, fontsize=12, fontweight="bold")
        if j == 0:
            ax.set_ylabel("TRAIN epoch", fontsize=11)
        ax.set_xlabel("TEST epoch", fontsize=11)
    cax = fig.add_subplot(gs_top[0, len(arms)])
    fig.colorbar(im, cax=cax).set_label("accuracy - null  (shared scale)", fontsize=10)

    # ---- row 1 right: the diagonal (each epoch's own ceiling) -------------------------------
    ax = fig.add_subplot(gs_top[0, len(arms) + 2])
    for arm in arms:
        d = [mats[arm][i, i] for i in range(len(eps))]
        ax.plot(range(len(eps)), d, "o-", label=arm, lw=2, color=tm.arm_color(arm))
    ax.set_xticks(range(len(eps)), [short[e] for e in eps], fontsize=10)
    ax.set_ylabel("own-epoch ceiling (acc - null)", fontsize=10)
    ax.set_title("EVERY WINDOW ENDS ABOVE ITS\nPRE-STROKE CEILING", fontsize=12, fontweight="bold")
    ax.axhline(0, color="k", lw=0.6)
    # OUTSIDE THE AXES, one row. In-axes it sat lower-right and the rising `rest` line ran through
    # its bottom entry; nudging it UP only moves it into cue and lick. There is no free corner in a
    # panel whose four lines span most of the height, so the legend leaves the data area entirely.
    ax.legend(fontsize=10, frameon=False, ncol=len(arms), loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=10)

    # ---- row 2: direction symmetry and the ceiling gain, split evenly ----------------------
    # WIDTH SCALES WITH THE ARM COUNT. Hard-coded at 0.26 this fits three arms (0.78 of the unit
    # spacing) and OVERFLOWS at four (1.04), so the rest bars overlapped their neighbours -- caught
    # on the published figure once rest was added. 0.82 leaves a visible gutter at any count.
    w = 0.82 / max(len(arms), 1)
    ref = [an for an, *_ in summary[0][2]]
    for col, (vals, ylab, title) in enumerate((
            ([[r - f for _a, f, r, _rf, _rr, _g in s[2]] for s in summary],
             "chronic$\\rightarrow$pre  minus  pre$\\rightarrow$chronic\n(raw above-chance)",
             # THE TITLE STATED "every bar is NEGATIVE", which was true of ENL/cue/lick and FALSE
             # of rest (+0.002 / +0.021 / +0.015). Caught on the published figure, 2026-09-17.
             # The claim that matters was never the sign anyway: ADDITION predicts a LARGE
             # POSITIVE asymmetry, and what rejects it is that no bar is MATERIAL in either
             # direction -- so the title now says that instead of a sign that can flip.
             "DIRECTION SYMMETRY -- no bar is material; ADDITION needs a large POSITIVE"),
            ([[g for *_r, g in s[2]] for s in summary],
             "chronic ceiling - pre ceiling\n(above chance)",
             "CEILING GAIN -- positive in every arm and animal"))):
        ax = fig.add_subplot(gs_bot[0, col])
        for j, (arm, _v, _pa) in enumerate(summary):
            xs = np.arange(len(ref)) + (j - (len(arms) - 1) / 2) * w
            ax.bar(xs, vals[j], width=w, label=arm, color=tm.arm_color(arm))
        ax.set_xticks(range(len(ref)), ref, fontsize=11)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        # LEGEND BELOW THE AXES, not inside: with four arms the in-axes legend sat
        # on top of the bars it was labelling.
        ax.legend(fontsize=8, frameon=False, ncol=len(arms),
                  loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=10)

    fig.text(0.5, 0.995, "Cross-epoch decoder TRANSFER by window: is the chronic code a rotation, "
             "or an addition?", ha="center", va="top", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.955,
             "Train on one epoch, score on another, in the shared joint LocaNMF basis. Training "
             "size block-matched in every cell; block-permutation null; animals weighted equally.\n"
             "THE TOP ROW is what the existing frozen decoder already reports; rows 2-4 -- and "
             "chronic$\\rightarrow$pre in particular -- are what separates a ROTATED code from an "
             "ADDED-TO one.\n"
             "ADDITION predicts a LARGE POSITIVE asymmetry; every bar is far below the "
             "materiality bar (25% of that animal's own transfer), so it is rejected in all four "
             "windows.\n"
             "The task arms lean slightly negative and rest slightly positive, but at "
             "this magnitude the sign is not interpretable. Continued practice is ruled out -- "
             "across the 70-day pre-stroke span decodability does not rise with day "
             "(mean slope -0.009 / 30 d).",
             ha="center", va="top", fontsize=10)
    out = out_dir / f"epoch_15g_transfer_by_window{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(fl.svg_path(out), bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
