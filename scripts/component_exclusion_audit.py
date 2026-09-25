"""Does the glue/bulb exclusion change the CD or the decoders? The measurements behind DECISIONS.md.

Priya, 2026-09-24: *"for the map analyses we do in this repo, we masked the area covered by glue and
some rim regions. (1) should we apply that before doing CD on the locaNMF components? and bigger blast
radius (2) should we have done this for ALL the decoder analyses??"* and, on the mechanism,
*"isn't their weighting equally to all other components (unique to cd analysis) something we wouldn't
want?"*

THIS IS A MODULE RATHER THAN A SCRATCH SCRIPT BECAUSE DECISIONS.md CITES ITS NUMBERS. The repo has
been here: `_compare_415_470_corr.py` was a root one-off deleted as dead scratch and the figures it
left behind outlived the code that made them, which is how a measured number becomes stale prose in a
speaker note that nothing regenerates.

FOUR ARMS, `--arm`:

  exposure   how much of each component's spatial mass lies under that animal's painted glue or in a
             bulb. The FACT: about a quarter of every basis, several components 100% inside the glue.
  cd         the CD's |w| share on those components, and the pole gap with them dropped, against a
             RANDOM subset of the same size -- because dropping a quarter of any basis costs
             something and the question is whether this quarter costs more.
  noise      the direction's noise floor, by refitting on labels shuffled WITHIN SESSION. This is the
             arm that answers the weighting question: under equal-variance weighting a component with
             no signal still gets a weight, and this measures how much.
  decoder    balanced accuracy with all components, with the occluded ones dropped, and with ONLY the
             occluded ones -- the three that separate "the readout survives the mask" from "there is
             nothing under the glue". They are different claims and only the first is about validity.

    python -m scripts.component_exclusion_audit --arm exposure cd
    python -m scripts.component_exclusion_audit --arm decoder --animal PS95
"""
from __future__ import annotations

import argparse

import numpy as np

from wfield_local import analysis_kit as ak
from wfield_local import cd_trajectories as cdt
from wfield_local import component_exclusion as cex
from wfield_local import config, epochs, joint_locanmf, session_cache
from wfield_local.locanmf_frozen_decoder import _args
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
EPOCH_ORDER = ("pre", "acute", "subacute", "chronic")

#: Random subsets drawn for the `cd` arm's control, and label shuffles for `noise`. Both are small on
#: purpose: they calibrate a single number each, and the pool they draw from has 87-95 members.
N_DRAW = 30


def _features(animal, align, basis, only_pre=True):
    """``(X, y, session_label)`` from the CACHED `cdfit-` window means -- one column per component.

    ONE WINDOW MEAN PER COMPONENT, not `decode.bins` sub-bins: applying a per-component mask needs an
    unambiguous component per column, and the production feature vector is (component x sub-bin). So
    the LEVEL here is not comparable to a published decode; the DIFFERENCE between column sets is,
    and that is the whole question.
    """
    post_s = float(config.defaults()["decode"].get(f"{align}_post_s", 2.0))
    args = _args(source="roi", align=align, post_s=post_s)
    win_n = int(round(args.post_s * args.fs))
    akind = cdt.arms_cache_kind(args, align)
    X, y, g = [], [], []
    for s in [x for x in ak.curated_sessions() if x["label"].startswith(animal)]:
        ep = epochs.epoch_of(s["label"])
        if only_pre and ep != "pre":
            continue
        try:
            arms = session_cache.cached(s, akind,
                                        lambda s=s: cdt.session_arms(s, args, basis, align),
                                        verbose=False)
        except Exception:                                              # noqa: BLE001
            continue
        if not arms["success"]["y"]:
            continue
        got = cdt._OnceSignal(joint_locanmf.BasisSource(basis, s))
        X.append(session_cache.cached(
            s, f"cdfit-{align}-success-{basis.basis_id[:8]}-{win_n}",
            lambda got=got, arms=arms: cdt.window_means(got(), arms["success"]["fit"], win_n),
            verbose=False))
        y.append(np.asarray(arms["success"]["y"]))
        g.append(np.array([(s["label"], ep)] * len(arms["success"]["y"]), dtype=object))
    if not X:
        return None
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def arm_exposure(animals, align):
    print(f"{'animal':7s} {'ncomp':>5s} {'>50% glue/bulb':>14s} {'>25%':>6s} {'>10%':>6s} "
          f"{'basis mass excluded':>19s}")
    for an in animals:
        b = joint_locanmf.load(an, warn_stale=False)
        f = cex.mass_fraction(b, an)
        A = np.abs(np.nan_to_num(np.asarray(b.A, dtype=np.float32)))
        tot = A.reshape(-1, A.shape[2]).sum(0)
        share = float(np.nansum(np.nan_to_num(f) * tot) / max(tot.sum(), 1e-9))
        print(f"{an:7s} {b.ncomp:5d} {int(np.nansum(f > 0.50)):14d} "
              f"{int(np.nansum(f > 0.25)):6d} {int(np.nansum(f > 0.10)):6d} {share:18.1%}")


def arm_cd(animals, align):
    print(f"{'animal':7s} {'dropped':>7s} {'|w| share':>10s} {'by count':>9s} "
          f"{'gap kept':>9s} {'RANDOM kept [2.5,97.5]':>25s} {'pctile':>7s}")
    for an in animals:
        b = joint_locanmf.load(an, warn_stale=False)
        bad = cex.occluded(b, an)
        got = _features(an, align, b)
        if got is None:
            print(f"{an:7s} no cached features")
            continue
        X, y, _g = got
        stats = cdt.component_stats(X)

        def kept(mask, X=X, y=y, stats=stats):
            a = cdt.fit_directions(X, y, DISPLAY_ORDER, stats=stats)
            c = cdt.fit_directions(X, y, DISPLAY_ORDER, stats=stats, drop=mask)
            r = [(c[q][2] - c[q][1]) / (a[q][2] - a[q][1]) for q in a if a[q][2] - a[q][1]]
            return float(np.mean(r))

        dirs = cdt.fit_directions(X, y, DISPLAY_ORDER, stats=stats)
        share = float(np.mean([np.abs(w[bad]).sum() / np.abs(w).sum() for w, _, _ in dirs.values()]))
        obs = kept(bad)
        rng = np.random.default_rng(0)
        null = []
        for _ in range(N_DRAW):
            m = np.zeros(len(bad), bool)
            m[rng.permutation(len(bad))[:int(bad.sum())]] = True
            null.append(kept(m))
        null = np.asarray(null)
        print(f"{an:7s} {int(bad.sum()):7d} {share:9.1%} {bad.mean():8.1%} {obs:8.1%} "
              f"{f'{null.mean():.1%} [{np.percentile(null, 2.5):.1%},{np.percentile(null, 97.5):.1%}]':>25s} "
              f"{float((null < obs).mean()):6.0%}")
    print("pctile = fraction of random subsets that cost MORE than the occluded one. Near 100% means "
          "the occluded quarter is the cheapest quarter to drop.")


def arm_noise(animals, align):
    """The weighting question. See `DECISIONS.md`, 2026-09-24."""
    from wfield_local.position_coding_directions import direction

    print(f"{'animal':7s} {'noise/signal':>12s} {'noise POWER':>11s} {'occ real':>8s} "
          f"{'occ SHUFFLED':>12s} {'occ lr':>7s} {'by count':>8s}")
    for an in animals:
        b = joint_locanmf.load(an, warn_stale=False)
        bad = cex.occluded(b, an)
        got = _features(an, align, b)
        if got is None:
            continue
        X, y, g = got
        lab = np.array([q[0] for q in g])
        mu, sd = cdt.component_stats(X)
        ok = np.isfinite(X).all(1)
        Z, y, lab = (X[ok] - mu) / sd, y[ok], lab[ok]

        def raw(yv, method="dom", Z=Z):
            # `Z` BOUND AS A DEFAULT: this closure is defined inside the per-animal loop, so a free
            # reference would resolve to the LAST animal's features for every animal (ruff B023).
            out = {}
            for pq in DISPLAY_ORDER:
                m = yv == pq
                if m.sum() < cdt.MIN_TRIALS or (~m).sum() < cdt.MIN_TRIALS:
                    continue
                out[int(pq)] = (Z[m].mean(0) - Z[~m].mean(0) if method == "dom"
                                else direction(Z[m], Z[~m], method="lr"))
            return out

        real, lr = raw(y), raw(y, "lr")
        rng = np.random.default_rng(0)
        slen, socc = [], []
        for _ in range(N_DRAW):
            ysh = y.copy()
            for s_ in np.unique(lab):                      # WITHIN SESSION
                m = lab == s_
                ysh[m] = rng.permutation(y[m])
            d = raw(ysh)
            slen.append(np.mean([np.linalg.norm(v) for v in d.values()]))
            socc.append(np.mean([np.abs(v[bad]).sum() / np.abs(v).sum() for v in d.values()]))
        ratio = float(np.mean(slen) / np.mean([np.linalg.norm(v) for v in real.values()]))
        f_real = float(np.mean([np.abs(v[bad]).sum() / np.abs(v).sum() for v in real.values()]))
        f_lr = float(np.mean([np.abs(v[bad]).sum() / np.abs(v).sum() for v in lr.values()]))
        print(f"{an:7s} {ratio:12.3f} {ratio ** 2:10.1%} {f_real:8.1%} "
              f"{float(np.mean(socc)):12.1%} {f_lr:7.1%} {bad.mean():8.1%}")
    print("THE SHUFFLED COLUMN IS THE NULL EXPECTATION: under equal-variance weighting a component "
          "with no signal still gets about its share by count. Read `occ real` against it, not "
          "against zero.")


def arm_decoder(animals, align):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from wfield_local import nolick_analysis as na

    print(f"{'animal':7s} {'epoch':9s} {'n':>5s} {'arm':9s} {'ncol':>5s} {'balanced':>8s} "
          f"{'null':>6s} {'above null':>10s}")
    for an in animals:
        b = joint_locanmf.load(an, warn_stale=False)
        bad = cex.occluded(b, an)
        got = _features(an, align, b, only_pre=False)
        if got is None:
            continue
        X, y, g = got
        lab = np.array([q[0] for q in g])
        ep_of = np.array([q[1] for q in g])
        for ep in EPOCH_ORDER:
            m = ep_of == ep
            if m.sum() < 50 or np.unique(lab[m]).size < 2:
                continue
            for name, cols in (("all", np.ones(X.shape[1], bool)), ("clean", ~bad),
                               ("occluded", bad)):
                if not cols.any():
                    continue
                # GROUPED BY SESSION, stricter than the production block CV and deliberately not a
                # locally re-derived block definition (the `enl_state_counts` lesson, rule 9).
                ng = min(5, int(np.unique(lab[m]).size))
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, C=0.5))
                pred = cross_val_predict(clf, X[m][:, cols], y[m], cv=GroupKFold(ng),
                                         groups=lab[m])
                bal = na.balanced_accuracy(y[m], pred, labels=DISPLAY_ORDER)
                nb = float(na.permutation_null(y[m], pred, n_perm=500,
                                              labels=DISPLAY_ORDER)["bal_null_mean"])
                print(f"{an:7s} {ep:9s} {int(m.sum()):5d} {name:9s} {int(cols.sum()):5d} "
                      f"{bal:8.3f} {nb:6.3f} {bal - nb:10.3f}", flush=True)


ARMS = {"exposure": arm_exposure, "cd": arm_cd, "noise": arm_noise, "decoder": arm_decoder}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="+", default=["exposure", "cd", "noise"], choices=tuple(ARMS))
    ap.add_argument("--animal", action="append", default=None)
    ap.add_argument("--align", default="precue", choices=("precue", "cue", "lick"))
    args = ap.parse_args(argv)
    animals = args.animal or list(ANIMALS)
    for a in args.arm:
        print(f"\n===== {a.upper()}   align={args.align}   "
              f"threshold {cex.MASS_THRESHOLD:.0%} of spatial mass =====", flush=True)
        ARMS[a](animals, args.align)
    print(f"\n(positions: {', '.join(POSITION_NAMES.get(p, str(p)) for p in DISPLAY_ORDER)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
