"""Pre-cue activity across five lick/engagement states, in the shared joint-LocaNMF basis.

Priya, 2026-08-20. REPORTED, not a filter: nothing here changes what any existing analysis uses.
``POSTSTROKE_ENGAGEMENT_FILTERING`` stays False and the all-trials post-stroke rule is untouched.

THE FIVE STATES, all in the PRE-CUE window:

  1  pre-stroke,  LICK          engaged, movement happened
  2  pre-stroke,  NO-LICK       pre-stroke this is essentially the sated/disengaged state
  4  post-stroke, LICK          engaged, movement happened
  5  post-stroke, NO-LICK, ENGAGED       the animal was working; this trial produced no lick
  6  post-stroke, NO-LICK, NOT ENGAGED   after the animal quit

THE TEST THAT MAKES IT MORE THAN FIVE NUMBERS. A discriminator is trained on the PRE-STROKE
lick-vs-no-lick contrast -- i.e. on data where engagement is not in question and there is no motor
deficit to confuse it with -- and states 5 and 6 are pushed through it. If the engagement gate is
capturing a real state, 6 should score DISENGAGED and 5 should not. If 5 scores engaged while
producing no movement, that is the "plan formed, execution failed" reading with an independent
witness. If 5 and 6 score alike, the gate is not capturing anything the cortex shows, and that
is a real and reportable outcome.

WHY THE TRAINING SET IS THE LATE WINDOW. Pre-stroke no-lick trials sit late in the session -- 79%,
85% and 78% of them in the final quarter for PS92, PS94 and PS95 -- because pre-stroke that IS what
a no-lick trial is (Priya, 2026-08-20). Taking both classes from the same late window therefore does
NOT strip the state; it isolates it, and it removes elapsed time as an explanation at the same time.
State 6 is terminal by construction, so without that control a discriminator trained on early-vs-late
would separate 5 from 6 by the clock alone and look like a state readout.

PS93 IS TREATED SEPARATELY. Its no-lick trials are not mainly satiety: 90% of its EARLY no-lick
trials are at far_L/far_center, i.e. spatial misses (Priya, 2026-08-20, confirmed by count). Those
are excluded from its disengaged class, or its axis would be a mixture of "disengaged" and "missed a
hard position".

THE TRIAL UNIVERSE IS THE IMAGING PIPELINE'S OWN. The engaged/no-lick split, the positions and the
trial ordering all come from ``_trial_features`` via ``with_indices``, never from the behaviour
trial table. The two disagree in ways that would silently mis-label trials: different free-reward
handling, and a response window of ``decode.max_rt_s`` here against the session's real 3.5 s there.
One universe, one definition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_frozen_decoder import _pipe, pool_sessions
from wfield_local.locanmf_position_decoder import _trial_features
from wfield_local.paths import PathResolver
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES

#: "reliable locations" the engagement gate is judged at (Priya, 2026-08-20).
REFERENCE = ("close_L", "close_center")
#: positions whose no-lick trials are spatial misses rather than disengagement, per animal.
SPATIAL_MISS = {"PS93": ("far_L", "far_center")}
LATE_FRACTION = 0.75      # "late window" = final quarter of a session
WINDOW = 15               # trailing window, in REFERENCE trials
MIN_RATE = 0.5            # reference response rate below this = collapsed
CHANCE = 1.0 / 6.0

STATES = ("1_pre_lick", "2_pre_nolick", "4_post_lick", "5_post_nolick_engaged",
          "6_post_nolick_notengaged")


def _disc():
    """The state discriminator. Plain regularised LR: the axis must stay interpretable."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))


def features_with_indices(basis):
    """``joint_features``, but it also keeps each session's trial INDICES.

    The indices are what let the engagement gate be computed on exactly the trials the features
    come from, in their true within-session order -- including the fact that the pre-cue alignment
    DROPS trials with no lick-free window, so the sequence has gaps that must not be closed up.
    """
    idx, vc = {}, {}

    def _feat(s, args):
        lab = s["label"]
        if lab in basis.labels:
            sig, vc[lab] = basis.signal(lab), 1.0
        else:
            sig, diag = basis.project(s, with_diagnostics=True)
            vc[lab] = float(diag["variance_captured"])
        out = _trial_features(s, args, signal=sig, feat_region=basis.regions, with_indices=True)
        idx[lab] = (out[6], out[7])
        return out[:6]

    _feat.indices = idx
    _feat.variance_captured = vc
    return _feat


def engagement_gate(order, responded, positions, reference=REFERENCE):
    """Per-trial engaged/not-engaged for ONE session, from a NON-RECOVERING reference collapse.

    ``order`` is the trial index of each entry (gaps allowed), already sorted. Returns a bool array
    aligned to it, True = NOT engaged.

    Requiring non-recovery is what separates satiety from a motor patch: PS94_0817's reference rate
    drops around trial 420 and is back near 0.95 by 480, and that session should not be called
    disengaged at all. Only the FINAL sustained collapse counts.
    """
    ref = np.isin(positions, reference)
    roll = np.full(len(order), np.nan)
    hist: list[bool] = []
    for i in range(len(order)):
        if ref[i]:
            hist.append(bool(responded[i]))
            if len(hist) > WINDOW:
                hist.pop(0)
        if hist:
            roll[i] = float(np.mean(hist))
    low = np.nan_to_num(roll, nan=1.0) < MIN_RATE
    out = np.zeros(len(order), bool)
    if low.any() and low[-1]:
        i = len(low) - 1
        while i > 0 and low[i - 1]:
            i -= 1
        out[i:] = True
    return out


def _session_tables(feat, kept):
    """Per session: ordered trial index, responded, position name, and late-window flag."""
    tables = {}
    for si, lab in enumerate(kept):
        got = feat.indices.get(lab)
        if got is None:
            continue
        idx_e, idx_n = got
        order = np.concatenate([idx_e, idx_n])
        responded = np.concatenate([np.ones(len(idx_e), bool), np.zeros(len(idx_n), bool)])
        o = np.argsort(order, kind="stable")
        order, responded = order[o], responded[o]
        tables[si] = {"order": order, "responded": responded, "label": lab}
    return tables


def _positions_for(tables, si, y_eng, g_eng, y_nl, g_nl, idx_e, idx_n):
    """Position NAME per entry of ``tables[si]['order']``."""
    pos_by_idx = {}
    for k, p in zip(idx_e, y_eng[g_eng == si]):
        pos_by_idx[int(k)] = POSITION_NAMES.get(int(p), str(p))
    for k, p in zip(idx_n, y_nl[g_nl == si]):
        pos_by_idx[int(k)] = POSITION_NAMES.get(int(p), str(p))
    return np.array([pos_by_idx.get(int(k), "?") for k in tables[si]["order"]])


def run_animal(animal, align="precue", post_s=None, verbose=True):
    """The five states plus the pre-stroke discriminator, for one animal. ``None`` if unavailable."""
    post_s = post_s or float(config.defaults()["decode"]["precue_post_s"])
    pre = [l for l in config.phase_labels("pre") if l.startswith(animal)]
    post = [l for l in config.phase_labels("post") if l.startswith(animal)]
    if not pre or not post:
        return None
    try:
        from wfield_local.locanmf_cue_lick_analysis import SESSIONS as _ALL
        basis = joint_locanmf.load(animal, sessions=_ALL)
    except FileNotFoundError as ex:
        print(f"[precue_states] {animal}: {ex}", flush=True)
        return None

    feat = features_with_indices(basis)
    pooled = pool_sessions(pre + post, source="locanmf", align=align, post_s=post_s, features=feat)
    if pooled is None:
        return None
    XE, YE, GE, _BE, XU, YU, kept, _common, GU = pooled
    YU = YU.astype(int)
    pre_i = {i for i, l in enumerate(kept) if l in set(pre)}

    tables = _session_tables(feat, kept)
    if not tables:
        return None

    # ---- engagement gate + late-window flag, per session, on the IMAGING trial universe ----------
    not_eng_nl, late_nl, sess_nl = [], [], []          # aligned to XU rows
    late_e = []                                        # aligned to XE rows
    for si in range(len(kept)):
        t = tables.get(si)
        if t is None:
            continue
        idx_e, idx_n = feat.indices[kept[si]]
        pos = _positions_for(tables, si, YE, GE, YU, GU, idx_e, idx_n)
        ne = engagement_gate(t["order"], t["responded"], pos)
        span = max(int(t["order"].max()), 1)
        late = t["order"] > LATE_FRACTION * span
        by_idx_ne = {int(k): bool(v) for k, v in zip(t["order"], ne)}
        by_idx_late = {int(k): bool(v) for k, v in zip(t["order"], late)}
        for k in idx_n:
            not_eng_nl.append(by_idx_ne.get(int(k), False))
            late_nl.append(by_idx_late.get(int(k), False))
            sess_nl.append(si)
        for k in idx_e:
            late_e.append(by_idx_late.get(int(k), False))
    not_eng_nl = np.array(not_eng_nl, bool)
    late_nl, late_e = np.array(late_nl, bool), np.array(late_e, bool)
    if len(not_eng_nl) != len(XU) or len(late_e) != len(XE):
        print(f"[precue_states] {animal}: trial bookkeeping mismatch "
              f"(nolick {len(not_eng_nl)}/{len(XU)}, eng {len(late_e)}/{len(XE)}) -- skipped",
              flush=True)
        return None

    e_pre = np.isin(GE, list(pre_i))
    u_pre = np.isin(GU, list(pre_i))
    states = {
        "1_pre_lick": (XE[e_pre], YE[e_pre], GE[e_pre]),
        "2_pre_nolick": (XU[u_pre], YU[u_pre], GU[u_pre]),
        "4_post_lick": (XE[~e_pre], YE[~e_pre], GE[~e_pre]),
        "5_post_nolick_engaged": (XU[~u_pre & ~not_eng_nl], YU[~u_pre & ~not_eng_nl],
                                  GU[~u_pre & ~not_eng_nl]),
        "6_post_nolick_notengaged": (XU[~u_pre & not_eng_nl], YU[~u_pre & not_eng_nl],
                                     GU[~u_pre & not_eng_nl]),
    }

    # ---- A. position decoding: ONE frozen pre-stroke decoder, applied to every state -------------
    clf = _pipe().fit(XE[e_pre], YE[e_pre])
    dec = {}
    for k, (X, y, g) in states.items():
        if len(y) < 10:
            dec[k] = {"n": len(y), "accuracy": None}
            continue
        if k == "1_pre_lick":       # its own training data: leave-one-session-out instead
            p = cross_val_predict(_pipe(), X, y, cv=LeaveOneGroupOut(), groups=g)
        else:
            p = clf.predict(X)
        dec[k] = {"n": len(y), "accuracy": float(np.mean(p == y))}

    # ---- C. the pre-stroke state discriminator, trained in the LATE window -----------------------
    miss = SPATIAL_MISS.get(animal, ())
    nl_names = np.array([POSITION_NAMES.get(int(p), str(p)) for p in YU])
    keep_dis = u_pre & late_nl & ~np.isin(nl_names, miss)
    keep_eng = e_pre & late_e
    disc = {"n_engaged_train": int(keep_eng.sum()), "n_disengaged_train": int(keep_dis.sum()),
            "excluded_positions": list(miss)}
    if keep_dis.sum() >= 20 and keep_eng.sum() >= 20:
        Xd = np.vstack([XE[keep_eng], XU[keep_dis]])
        yd = np.concatenate([np.zeros(int(keep_eng.sum())), np.ones(int(keep_dis.sum()))])
        gd = np.concatenate([GE[keep_eng], GU[keep_dis]])
        model = _disc().fit(Xd, yd)
        if len(np.unique(gd)) > 1:
            cvp = cross_val_predict(_disc(), Xd, yd, cv=LeaveOneGroupOut(), groups=gd,
                                    method="predict_proba")[:, 1]
            disc["train_auc_loso"] = _auc(yd, cvp)
        for k, (X, _y, _g) in states.items():
            disc[k] = (float(np.mean(model.predict_proba(X)[:, 1])) if len(X) else None)
        # D. DRIFT CONTROL: both halves are ENGAGED, so any separation here is time, not state.
        eng5 = ~u_pre & ~not_eng_nl
        if eng5.sum() >= 20:
            early = XU[eng5 & ~late_nl]
            lateX = XU[eng5 & late_nl]
            disc["drift_control"] = {
                "n_early": len(early), "n_late": len(lateX),
                "early_p_disengaged": (float(np.mean(model.predict_proba(early)[:, 1]))
                                       if len(early) else None),
                "late_p_disengaged": (float(np.mean(model.predict_proba(lateX)[:, 1]))
                                      if len(lateX) else None)}
    if verbose:
        print(f"  {animal}: " + "  ".join(
            f"{k.split('_', 1)[0]}n={dec[k]['n']}" for k in STATES), flush=True)
    return {"animal": animal, "align": align, "basis_id": basis.basis_id, "ncomp": basis.ncomp,
            "chance": CHANCE, "decode": dec, "discriminator": disc,
            "sessions": list(kept), "n_pre_sessions": len(pre_i)}


def _auc(y, p):
    """ROC AUC without importing another metric; ties handled by rank averaging."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    r = np.argsort(np.argsort(np.asarray(p, float))) + 1.0
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def figure(results, out, align="precue"):
    """Two panels: position decoding per state, and the pre-stroke discriminator's verdict."""
    animals = [a for a in sorted(results) if results[a]]
    if not animals:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2))
    w = 0.8 / max(len(animals), 1)
    x = np.arange(len(STATES))
    cols = plt.get_cmap("tab10")(np.linspace(0, 0.4, max(len(animals), 2)))
    for k, an in enumerate(animals):
        r = results[an]
        acc = [(r["decode"].get(s, {}) or {}).get("accuracy") for s in STATES]
        ns = [(r["decode"].get(s, {}) or {}).get("n", 0) for s in STATES]
        xs = x + (k - (len(animals) - 1) / 2) * w
        axes[0].bar(xs, [a if a is not None else 0 for a in acc], w, color=cols[k],
                    edgecolor="k", linewidth=0.4, label=an)
        for xi, a, n in zip(xs, acc, ns):
            if a is None:
                axes[0].text(xi, 0.01, f"n={n}", ha="center", va="bottom", fontsize=6,
                             rotation=90, color="firebrick")
        d = r["discriminator"]
        pv = [d.get(s) for s in STATES]
        axes[1].bar(xs, [p if p is not None else 0 for p in pv], w, color=cols[k],
                    edgecolor="k", linewidth=0.4, label=an)
    axes[0].axhline(CHANCE, color="k", ls=":", lw=1.2)
    axes[0].set_ylabel("position decoding accuracy (frozen pre-stroke decoder)")
    axes[0].set_title(f"A. Position information in the {align} window, by state\n"
                      "dotted = chance (1/6)", fontsize=10)
    axes[1].axhline(0.5, color="k", ls=":", lw=1.2)
    axes[1].set_ylabel("P(disengaged) under the PRE-STROKE lick/no-lick discriminator")
    axes[1].set_title("B. Where each state falls on the pre-stroke ENGAGEMENT axis\n"
                      "trained late-window only, so it is not a clock", fontsize=10)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n", 1) for s in STATES], fontsize=7.5)
        ax.legend(fontsize=8)
    fig.suptitle("Pre-cue activity across five lick/engagement states, shared joint-LocaNMF basis. "
                 "REPORTED ONLY — no existing analysis filters on this.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = Path(out) / f"precue_states_{align}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--align", default="precue", choices=("precue", "cue"))
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    out.mkdir(parents=True, exist_ok=True)
    animals = config.normalize_animals(args.animals) or [a for a in config.animals()]
    results = {}
    for an in animals:
        try:
            results[an] = run_animal(an, align=args.align)
        except Exception as ex:                                           # noqa: BLE001
            print(f"  !! {an}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
            results[an] = None
    p = figure(results, out, align=args.align)
    (out / f"precue_states_{args.align}.json").write_text(
        json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"wrote {p} and precue_states_{args.align}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
