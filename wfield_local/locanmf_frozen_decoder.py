"""Persist per-session position decoders and apply a FROZEN decoder across days — the
confirmatory arm of the stroke study (train pre-stroke, apply post-stroke).

Two transfer paths for the cross-day problem (LocaNMF components are session-specific):
  - ROI (default, registration-free): Allen-region features are atlas-anchored, so a decoder
    trained on day A applies directly to day B (features aligned by region label).
  - Fixed-A / refit-C (scaffold, for the LocaNMF basis): keep pre-stroke footprints A_ref and
    project a new session onto them, C_new = pinv(A_ref) @ (U_new @ SVT_new), on the shared
    Allen-aligned grid. Valid because the stroke is subcortical (cortical map intact).

    python -m wfield_local.locanmf_frozen_decoder --save            # save all baseline decoders
    python -m wfield_local.locanmf_frozen_decoder --transfer --output "<dir>"   # cross-day ROI demo
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

from wfield_local import config
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES, DISPLAY_ORDER

FS = 31.23
BASELINE_DAYS = ("0601", "0602", "0603", "0604")


def _sess(label):
    return next(s for s in SESSIONS if s["label"] == label)


def _args(source="locanmf", align="lick", post_s=2.0, baseline="none"):
    return SimpleNamespace(source=source, align=align, baseline=baseline,
                           pre_s=1.0, post_s=post_s, fs=FS, max_rt=2.0)


def _pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))


def _decoder_dir(s):
    d = Path(s["mc"]) / "decoder_affine8v1"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session_decoder(label, source="locanmf", align="lick", post_s=2.0):
    """Fit on ALL engaged trials and persist the pipeline + metadata to MICROSCOPE."""
    s = _sess(label)
    X, y, g, Xnl, ynl, feat_reg = _trial_features(s, _args(source, align, post_s))
    pipe = _pipe().fit(X, y)
    # honest block-CV accuracy for the record
    ng = min(5, int(np.unique(g).size))
    cv_acc = accuracy_score(y, cross_val_predict(_pipe(), X, y, cv=GroupKFold(ng), groups=g)) if ng >= 2 else float("nan")
    dd = _decoder_dir(s)
    stem = f"{label}_decoder_{source}_{align}_2s"
    joblib.dump({"pipeline": pipe, "feat_region": feat_reg, "classes": pipe.named_steps["logisticregression"].classes_,
                 "source": source, "align": align, "post_s": post_s, "baseline": "none"}, dd / f"{stem}.joblib")
    meta = {"label": label, "source": source, "align": align, "post_s": post_s, "cv_block_accuracy": cv_acc,
            "n_engaged": int(X.shape[0]), "n_features": int(X.shape[1]), "n_nolick": int(Xnl.shape[0]),
            "positions": [POSITION_NAMES[c] for c in DISPLAY_ORDER], "chance": 1 / 6,
            "feature_region_labels": [int(r) for r in feat_reg]}
    (dd / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return dd / f"{stem}.joblib", cv_acc


def _aligned(Xtr, rtr, Xte, rte):
    """Subset two ROI feature matrices to their common region labels, in a shared order."""
    common = [r for r in rtr if r in set(rte)]
    itr = [list(rtr).index(r) for r in common]; ite = [list(rte).index(r) for r in common]
    return Xtr[:, itr], Xte[:, ite]


def cross_day_transfer(train_label, test_label, source="roi", align="lick", post_s=2.0):
    """Train a frozen decoder on train_label, apply to test_label (ROI: registration-free)."""
    Xtr, ytr, _, _, _, rtr = _trial_features(_sess(train_label), _args(source, align, post_s))
    Xte, yte, _, Xnl, ynl, rte = _trial_features(_sess(test_label), _args(source, align, post_s))
    if source == "roi":
        Xtr, Xte = _aligned(Xtr, rtr, Xte, rte)
    elif Xtr.shape[1] != Xte.shape[1]:
        raise ValueError("LocaNMF bases differ across sessions; use source='roi' or the fixed-A path.")
    pipe = _pipe().fit(Xtr, ytr)
    return accuracy_score(yte, pipe.predict(Xte))


def _align_many(mats, regs):
    """Subset N ROI feature matrices to the region labels common to ALL of them, in one shared order.

    :func:`_aligned` handles a pair; pooling needs the N-way intersection so that column j is the
    same Allen area in every session.
    """
    common = [r for r in regs[0] if all(r in set(rr) for rr in regs[1:])]
    return [m[:, [list(rr).index(r) for r in common]] for m, rr in zip(mats, regs)], common


def _entropy_norm(proba):
    """Mean Shannon entropy of the predicted distributions, normalized so 1.0 == uniform.

    Accuracy alone cannot separate "no information" from "confidently wrong": the softmax always
    sums to 1, so it never abstains. Entropy can. 1.0 = perfectly smeared over all classes.
    """
    p = np.clip(proba, 1e-12, 1.0)
    return float((-(p * np.log(p)).sum(1)).mean() / np.log(proba.shape[1]))


def pool_sessions(labels, source="roi", align="cue", post_s=2.0, zscore=True):
    """Pool several sessions into ONE feature matrix that is comparable across days.

    Shared by the frozen decoder and the frozen encoder. Returns
    ``(XE, YE, GE, BE, XU, YU, kept_labels, common_regions)`` or ``None`` if <2 sessions load,
    where ``GE`` is the SESSION index (the CV group for leave-one-day-out) and ``BE[i]`` is
    session *i*'s within-session position-block ids (the CV group for the same-day ceiling).

    Two things make pooling valid, and both matter:
      * ROI features are ATLAS-ANCHORED, so column j is the same cortical area in every session.
        LocaNMF components are not (session-specific count AND identity), which is why the
        cross-day work uses ``source="roi"``.
      * each session is z-scored using its OWN engaged trials, so session-level differences in F0
        / SNR cannot drive the result; the same transform is applied to that session's no-lick
        trials so both stay on one scale.
    """
    per, regs, kept = [], [], []
    for lab in labels:
        s = next((x for x in SESSIONS if x["label"] == lab), None)
        if s is None:
            continue
        try:
            X, y, g, Xnl, ynl, reg = _trial_features(s, _args(source, align, post_s))
        except Exception as e:                                  # noqa: BLE001 - report and continue
            print(f"  !! {lab}: {type(e).__name__} {str(e)[:60]}", flush=True)
            continue
        per.append((X, y, g, Xnl, ynl)); regs.append(reg); kept.append(lab)
    if len(kept) < 2:
        return None
    if source == "roi":
        mats, common = _align_many([p[0] for p in per], regs)
        nl, _ = _align_many([p[3] if len(p[3]) else np.zeros((0, len(r))) for p, r in zip(per, regs)], regs)
    else:
        mats, nl, common = [p[0] for p in per], [p[3] for p in per], regs[0]

    XE, YE, GE, BE, XU, YU = [], [], [], [], [], []
    for i, ((_X, y, g, _Xnl, ynl), Xa, Na) in enumerate(zip(per, mats, nl)):
        mu, sd = (Xa.mean(0), Xa.std(0)) if zscore else (0.0, 1.0)
        sd = np.where(np.asarray(sd) > 0, sd, 1.0) if zscore else 1.0
        XE.append((Xa - mu) / sd); YE.append(y); GE.append(np.full(len(y), i)); BE.append(g)
        if len(Na):
            XU.append((Na - mu) / sd); YU.append(ynl)
    XE = np.vstack(XE); YE = np.concatenate(YE); GE = np.concatenate(GE)
    XU = np.vstack(XU) if XU else np.zeros((0, XE.shape[1]))
    YU = np.concatenate(YU) if YU else np.array([])
    return XE, YE, GE, BE, XU, YU, kept, common


def _ev(X, pred):
    """Fraction of variance explained, pooled over features (1 - SS_res/SS_tot)."""
    ss_tot = ((X - X.mean(0)) ** 2).sum()
    return float(1 - ((X - pred) ** 2).sum() / max(ss_tot, 1e-12))


def _ceiling(X, y):
    """Encoder noise ceiling: between-position SS / total SS (the max a position-only model can reach)."""
    gm = X.mean(0); betw = np.zeros(X.shape[1]); wit = np.zeros(X.shape[1])
    for p in np.unique(y):
        m = y == p
        mu = X[m].mean(0)
        betw += m.sum() * (mu - gm) ** 2
        wit += ((X[m] - mu) ** 2).sum(0)
    return float(betw.sum() / max(betw.sum() + wit.sum(), 1e-12)), betw, wit


def pooled_frozen_encoder(labels, source="roi", align="cue", post_s=2.0, alpha=1.0, verbose=True):
    """FROZEN cross-day ENCODER: position -> expected cortical activity, applied to an unseen day.

    The mirror of :func:`pooled_frozen_loso`. Ridge from a one-hot position design to ROI activity,
    fit on an animal's OTHER curated days and evaluated on the held-out day, so it answers "does the
    position->activity mapping itself transfer across days" -- the forward-model half of the
    post-stroke confirmatory arm (a frozen encoder's RESIDUAL on post-stroke trials is the
    representational-change readout).

    Reported per held-out day:
      * ``ev``       held-out explained variance of the frozen model,
      * ``ceiling``  that day's own noise ceiling (between-position SS / total SS) -- the most any
        position-only model could achieve there, so a low EV on a low-ceiling day is not a failure,
      * ``feve``     ev / ceiling, the ceiling-normalised score that is comparable across days,
      * ``within``   the same-day encoder (block CV) for reference.
    """
    pooled = pool_sessions(labels, source=source, align=align, post_s=post_s)
    if pooled is None:
        return None
    XE, YE, GE, BE, _XU, _YU, kept, common = pooled
    pos = np.array(DISPLAY_ORDER)
    P = np.stack([(YE == p).astype(float) for p in pos], 1)

    per_sess, ceil, feve, within = {}, {}, {}, {}
    for i, lab in enumerate(kept):
        te = GE == i
        tr = ~te
        pred = Ridge(alpha=alpha).fit(P[tr], XE[tr]).predict(P[te])
        per_sess[lab] = _ev(XE[te], pred)
        c, _, _ = _ceiling(XE[te], YE[te])
        ceil[lab] = c
        feve[lab] = per_sess[lab] / c if c > 1e-6 else float("nan")
        gb = BE[i]
        ng = min(5, int(np.unique(gb).size))
        if ng >= 2:
            pw = np.zeros_like(XE[te])
            Xi, Yi, Pi = XE[te], YE[te], P[te]
            for a, b in GroupKFold(ng).split(Xi, Yi, gb):
                pw[b] = Ridge(alpha=alpha).fit(Pi[a], Xi[a]).predict(Pi[b])
            within[lab] = _ev(Xi, pw)
    res = {"labels": kept, "source": source, "align": align, "n_trials": int(len(YE)),
           "n_features": int(XE.shape[1]), "n_sessions": len(kept), "n_regions": len(common),
           "per_session_ev": per_sess, "ceiling": ceil, "feve": feve, "within_session_ev": within,
           "mean_ev": float(np.mean(list(per_sess.values()))),
           "mean_feve": float(np.nanmean(list(feve.values()))),
           "mean_within_ev": float(np.mean(list(within.values()))) if within else float("nan")}
    res["transfer_cost_ev"] = res["mean_ev"] - res["mean_within_ev"]
    if verbose:
        print(f"\n  pooled: {len(YE)} trials, {XE.shape[1]} ROI features, {len(kept)} sessions")
        print(f"  {'session':12s} {'EV(frozen)':>11s} {'ceiling':>8s} {'FEVE':>7s} {'EV(within)':>11s}")
        for lab in kept:
            print(f"    {lab:12s} {per_sess[lab]:>11.4f} {ceil[lab]:>8.4f} {feve[lab]:>7.3f} "
                  f"{within.get(lab, float('nan')):>11.4f}")
        print(f"  mean EV frozen {res['mean_ev']:+.4f}   within-day {res['mean_within_ev']:+.4f}   "
              f"transfer cost {res['transfer_cost_ev']:+.4f}   mean FEVE {res['mean_feve']:.3f}")
    return res


def pooled_frozen_loso(labels, source="roi", align="cue", post_s=2.0, zscore=True, verbose=True):
    """Leave-one-SESSION-out frozen decoder over an animal's pooled sessions (ROI features).

    LocaNMF components are session-specific (different count AND identity per session), so they
    cannot be pooled. Allen-ROI features are atlas-anchored -- column j is the same area in every
    session -- which makes a single across-day model well posed. This is the pre-stroke dress
    rehearsal for train-pre / apply-post.

    Per session the features are z-scored using that session's own ENGAGED trials (removing
    session-level offsets in F0/SNR that would otherwise dominate pooling); the same transform is
    applied to that session's no-lick trials so both stay on one scale. Groups = SESSION, so the
    held-out fold is an entire unseen day.

    Reported alongside:
      * ``within_session``  -- per-session block-CV (GroupKFold by position block), the honest
        same-day ceiling. LOSO minus this is the true cost of freezing across days.
      * ``ood`` -- the out-of-distribution control (see :func:`ood_control`).
    """
    pooled = pool_sessions(labels, source=source, align=align, post_s=post_s, zscore=zscore)
    if pooled is None:
        return None
    XE, YE, GE, BE, XU, YU, kept, common = pooled

    pred = cross_val_predict(_pipe(), XE, YE, cv=LeaveOneGroupOut(), groups=GE)
    loso = float(accuracy_score(YE, pred))
    per_sess = {lab: float(accuracy_score(YE[GE == i], pred[GE == i])) for i, lab in enumerate(kept)}
    within = {}
    for i, lab in enumerate(kept):
        m = GE == i
        gb = BE[i]
        ng = min(5, int(np.unique(gb).size))
        if ng >= 2:
            within[lab] = float(accuracy_score(YE[m], cross_val_predict(
                _pipe(), XE[m], YE[m], cv=GroupKFold(ng), groups=gb)))
    res = {"labels": kept, "source": source, "align": align, "n_engaged": int(len(YE)),
           "n_features": int(XE.shape[1]), "n_sessions": len(kept), "n_regions": len(common),
           "loso_accuracy": loso, "per_session": per_sess, "within_session": within,
           "mean_within": float(np.mean(list(within.values()))) if within else float("nan"),
           "chance": 1 / 6}
    res["transfer_cost"] = res["loso_accuracy"] - res["mean_within"]
    # per-held-out-DAY confusion, so the deck can show the frozen decoder date by date exactly as it
    # shows the per-session (within-day) decoders. Rows = true position, in DISPLAY_ORDER.
    res["confusion"] = {lab: confusion_matrix(YE[GE == i], pred[GE == i],
                                              labels=DISPLAY_ORDER).tolist()
                        for i, lab in enumerate(kept)}
    res["ood"] = ood_control(XE, YE, GE, XU, YU)
    if verbose:
        print(f"\n  pooled: {len(YE)} engaged trials, {len(YU)} no-lick, {XE.shape[1]} ROI features, "
              f"{len(kept)} sessions")
        print(f"  LOSO accuracy {loso:.3f}   mean within-session {res['mean_within']:.3f}   "
              f"transfer cost {res['transfer_cost']:+.3f}   (chance {1/6:.3f})")
        for lab in kept:
            w = within.get(lab, float("nan"))
            print(f"    {lab:12s} LOSO {per_sess[lab]:.3f}   within {w:.3f}   {per_sess[lab]-w:+.3f}")
        o = res["ood"]
        print(f"  OOD control: engaged H={o['engaged_H']:.3f}  no-lick H={o['nolick_H']:.3f}  "
              f"shuffled H={o['shuffle_H']:.3f}  (1.0 = uniform)")
        print(f"               no-lick acc={o['nolick_acc']:.3f} CI{np.round(o['nolick_ci'],3).tolist()} "
              f"-> {'ABOVE' if o['nolick_above_chance'] else 'NOT above'} chance")
    return res


def ood_control(XE, YE, GE, XU, YU):
    """Out-of-distribution control for a frozen decoder. Confidence is NOT evidence.

    A softmax decoder never abstains: applied to input carrying no position information it still
    emits a normalized distribution, and empirically it does so CONFIDENTLY (measured on quiet /
    running periods, mean max-probability up to 0.997 with predictions collapsed onto a single
    position). So a frozen decoder's post-stroke confidence cannot be read as preserved coding
    unless it is compared against these references:

      * ``shuffle_H`` -- entropy with labels permuted: the empirical NO-INFORMATION floor. A real
        signal must sit well below it.
      * ``nolick_*``  -- the no-lick (unengaged) trials, which pre-stroke carry no decodable
        position (pooled across curated sessions: accuracy is at chance) yet still draw confident
        predictions. The nearest available analogue of a failed post-stroke attempt.
    """
    clf = _pipe().fit(XE, YE)
    classes = clf.named_steps["logisticregression"].classes_
    pe = cross_val_predict(_pipe(), XE, YE, cv=LeaveOneGroupOut(), groups=GE, method="predict_proba")
    ysh = np.random.RandomState(0).permutation(YE)
    psh = cross_val_predict(_pipe(), XE, ysh, cv=LeaveOneGroupOut(), groups=GE, method="predict_proba")
    out = {"engaged_H": _entropy_norm(pe), "engaged_maxp": float(pe.max(1).mean()),
           "shuffle_H": _entropy_norm(psh), "shuffle_maxp": float(psh.max(1).mean()),
           "shuffle_acc": float(accuracy_score(ysh, np.unique(ysh)[psh.argmax(1)]))}
    if len(YU) >= 30:
        pu = clf.predict_proba(XU)
        acc = float(accuracy_score(YU, classes[pu.argmax(1)]))
        se = float(np.sqrt(acc * (1 - acc) / len(YU)))
        out.update(n_nolick=int(len(YU)), nolick_acc=acc, nolick_H=_entropy_norm(pu),
                   nolick_maxp=float(pu.max(1).mean()),
                   nolick_ci=[acc - 1.96 * se, acc + 1.96 * se],
                   nolick_above_chance=bool(acc - 1.96 * se > 1 / 6),
                   nolick_pred_dist=(np.bincount(pu.argmax(1), minlength=len(classes)) / len(pu)).tolist())
    else:
        out.update(n_nolick=int(len(YU)), nolick_acc=float("nan"), nolick_H=float("nan"),
                   nolick_maxp=float("nan"), nolick_ci=[float("nan")] * 2, nolick_above_chance=False)
    return out


def write_session_confusions(results, out):
    """One PNG per held-out DAY: the frozen decoder's confusion + per-position recall on that day.

    Deliberately mirrors the per-session within-day decoder figures the deck already shows
    (``locanmf_position_session_<label>_...``) so the two read side by side; the only difference is
    that NO trial from this day was used to fit -- the model was trained on the animal's OTHER days.
    Filename: ``locanmf_frozen_session_<label>_roi_<align>.png``.
    """
    posn = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    written = []
    for an, r in results.items():
        for lab, cm in r.get("confusion", {}).items():
            cm = np.asarray(cm, dtype=float)
            n = cm.sum()
            row = cm.sum(1, keepdims=True)
            cmn = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)   # row-normalized
            recall = np.diag(cmn)
            acc = r["per_session"][lab]
            within = r["within_session"].get(lab, np.nan)
            # Squarer than wide: these are shown 2-per-row in the deck (like the per-session
            # decoders), so a short/wide figure would waste the cell height and render small.
            fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
            im = ax[0].imshow(cmn, vmin=0, vmax=1, cmap="magma")
            for i in range(len(posn)):
                for j in range(len(posn)):
                    ax[0].text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                               color="w" if cmn[i, j] < 0.55 else "k")
            ax[0].set_xticks(range(len(posn))); ax[0].set_xticklabels(posn, rotation=45, ha="right", fontsize=7)
            ax[0].set_yticks(range(len(posn))); ax[0].set_yticklabels(posn, fontsize=7)
            ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true")
            ax[0].set_title("row-normalized confusion", fontsize=9)
            fig.colorbar(im, ax=ax[0], shrink=0.8)
            ax[1].bar(range(len(posn)), recall, color="tab:orange")
            ax[1].axhline(1 / 6, color="grey", ls=":", lw=1)
            ax[1].text(len(posn) - 0.5, 1 / 6, " chance", va="bottom", ha="right", fontsize=7, color="grey")
            ax[1].set_xticks(range(len(posn))); ax[1].set_xticklabels(posn, rotation=45, ha="right", fontsize=7)
            ax[1].set_ylim(0, 1); ax[1].set_ylabel("recall")
            ax[1].set_title("per-position recall", fontsize=9)
            fig.suptitle(f"{lab} — FROZEN ROI decoder (trained on this animal's OTHER days)\n"
                         f"held-out-day acc {acc:.3f}   |   same-day ceiling {within:.3f}   |   "
                         f"n={int(n)} trials   |   chance 0.167", fontsize=10, y=0.99)
            fig.tight_layout(rect=(0, 0, 1, 0.87))
            p = Path(out) / f"locanmf_frozen_session_{lab}_roi_{r['align']}.png"
            fig.savefig(p, dpi=130); plt.close(fig)
            written.append(p)
    return written


def _encoder_fig(results, out):
    """Frozen ENCODER: per-day EV vs same-day ceiling, ceiling-normalised FEVE, and transfer cost."""
    animals = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    off = 0; ticks, tlabs = [], []
    for an in animals:
        r = results[an]
        start = off
        for lab in r["labels"]:
            w = r["within_session_ev"].get(lab, np.nan)
            ax[0].plot([off - 0.16, off + 0.16], [w, r["per_session_ev"][lab]], "-", color="0.75", lw=1)
            ax[0].scatter(off - 0.16, w, s=34, color="tab:blue", zorder=3)
            ax[0].scatter(off + 0.16, r["per_session_ev"][lab], s=34, color="tab:orange", zorder=3)
            ax[0].scatter(off, r["ceiling"][lab], s=26, marker="_", color="k", zorder=4)
            ticks.append(off); tlabs.append(f"{lab[-4:-2]}/{lab[-2:]}"); off += 1
        ax[0].annotate(an, xy=((start + off - 1) / 2, 1.02), xycoords=("data", "axes fraction"),
                       ha="center", fontsize=10, fontweight="bold")
        if an != animals[-1]:
            ax[0].axvline(off + 0.3, color="0.85", lw=1)
        off += 1.2
    ax[0].axhline(0, color="k", lw=1)
    ax[0].scatter([], [], s=34, color="tab:blue", label="within-day encoder (block CV)")
    ax[0].scatter([], [], s=34, color="tab:orange", label="FROZEN (unseen day)")
    ax[0].scatter([], [], s=26, marker="_", color="k", label="that day's noise ceiling")
    ax[0].set_xticks(ticks); ax[0].set_xticklabels(tlabs, fontsize=7, rotation=90)
    ax[0].set_ylabel("explained variance"); ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_xlim(-1, off - 0.8)
    ax[0].set_title("Frozen ROI encoder: held-out DAY vs same-day encoder", fontsize=10.5, pad=22)
    x = np.arange(len(animals)); w = 0.38
    ax[1].bar(x - w / 2, [results[a]["mean_ev"] for a in animals], w, label="frozen (unseen day)",
              color="tab:orange")
    ax[1].bar(x + w / 2, [results[a]["mean_within_ev"] for a in animals], w, label="within-day",
              color="tab:blue")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(animals); ax[1].set_ylabel("mean explained variance")
    ax[1].legend(fontsize=8); ax[1].set_title("Encoder EV: does position->activity transfer?", fontsize=10.5)
    ax[2].bar(x, [results[a]["mean_feve"] for a in animals], color="tab:green")
    ax[2].axhline(1.0, color="k", ls="--", lw=1)
    ax[2].text(0.02, 1.01, "= all explainable variance captured", fontsize=8,
               transform=ax[2].get_yaxis_transform())
    ax[2].set_xticks(x); ax[2].set_xticklabels(animals)
    ax[2].set_ylabel("FEVE  (EV / that day's ceiling)")
    ax[2].set_title("Ceiling-normalised: comparable across days & animals", fontsize=10.5)
    for xi, a in zip(x, animals):
        ax[2].text(xi, results[a]["mean_feve"], f"{results[a]['mean_feve']:.2f}",
                   ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = out / "locanmf_frozen_encoder_loso_roi.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def _loso_fig(results, out):
    """Three panels: per-session LOSO vs within-day ceiling, the transfer cost, and the OOD control."""
    animals = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    # (1) per-session LOSO vs within
    off = 0; ticks, tlabs = [], []
    for an in animals:
        r = results[an]
        start = off
        for lab in r["labels"]:
            w = r["within_session"].get(lab, np.nan)
            ax[0].plot([off - 0.16, off + 0.16], [w, r["per_session"][lab]], "-", color="0.75", lw=1, zorder=1)
            ax[0].scatter(off - 0.16, w, s=34, color="tab:blue", zorder=3)
            ax[0].scatter(off + 0.16, r["per_session"][lab], s=34, color="tab:orange", zorder=3)
            # tick = MMDD only; the animal is annotated once per group (full labels collide badly)
            ticks.append(off); tlabs.append(f"{lab[-4:-2]}/{lab[-2:]}"); off += 1
        ax[0].annotate(an, xy=((start + off - 1) / 2, 1.02), xycoords=("data", "axes fraction"),
                       ha="center", fontsize=10, fontweight="bold")
        if an != animals[-1]:
            ax[0].axvline(off + 0.3, color="0.85", lw=1)
        off += 1.2
    ax[0].axhline(1 / 6, color="grey", ls=":", lw=1)
    ax[0].scatter([], [], s=34, color="tab:blue", label="within-session (block CV)")
    ax[0].scatter([], [], s=34, color="tab:orange", label="LOSO (frozen, unseen day)")
    ax[0].set_xticks(ticks); ax[0].set_xticklabels(tlabs, fontsize=7, rotation=90)
    ax[0].set_ylabel("accuracy"); ax[0].legend(fontsize=8, loc="lower right"); ax[0].set_ylim(0, 1.0)
    ax[0].set_xlim(-1, off - 0.8)
    ax[0].set_title("Frozen ROI decoder: held-out DAY vs same-day ceiling", fontsize=10.5, pad=22)
    # (2) transfer cost
    x = np.arange(len(animals))
    tc = [results[a]["transfer_cost"] for a in animals]
    ax[1].bar(x, tc, color=["tab:green" if t >= 0 else "tab:red" for t in tc])
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(animals)
    ax[1].set_ylabel("LOSO - within-session")
    ax[1].set_title("Transfer cost of freezing across days\n(>0 = frozen model BEATS the same-day model)",
                    fontsize=10.5)
    for xi, t in zip(x, tc):
        ax[1].text(xi, t, f"{t:+.3f}", ha="center", va="bottom" if t >= 0 else "top", fontsize=9)
    # (3) OOD control
    keys = ["engaged", "no-lick", "shuffled"]
    w = 0.8 / max(len(animals), 1)
    for ai, an in enumerate(animals):
        o = results[an]["ood"]
        vals = [o["engaged_H"], o["nolick_H"], o["shuffle_H"]]
        ax[2].bar(np.arange(3) + ai * w - 0.4 + w / 2, vals, w, label=an)
    ax[2].axhline(1.0, color="k", ls="--", lw=1)
    ax[2].text(0.02, 1.005, "uniform = no information", fontsize=8,
               transform=ax[2].get_yaxis_transform())
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(keys)
    ax[2].set_ylabel("normalized entropy  H / log(n_classes)"); ax[2].set_ylim(0, 1.1)
    ax[2].legend(fontsize=8)
    ax[2].set_title("OOD control: the decoder never abstains\n"
                    "no-lick decodes at CHANCE yet stays confident", fontsize=10.5)
    fig.tight_layout()
    p = out / "locanmf_frozen_decoder_loso_roi.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def project_C_fixed_A(a_ref_label, new_label):
    """SCAFFOLD (post-stroke LocaNMF path): project a new session onto fixed pre-stroke footprints.
    C_new = pinv(A_ref) @ (U_new @ SVT_new) on the shared Allen-aligned grid. Returns C_new so a
    frozen LocaNMF decoder (trained on A_ref's components) can be applied to the new session.
    Requires both sessions on the same affine8v1 pixel grid; not exercised until post-stroke data."""
    ref = _sess(a_ref_label); new = _sess(new_label)
    A = np.load(f"{ref['mc']}/locanmf_affine8v1_final/{a_ref_label}_locanmf_A.npy")  # (H, W, ncomp)
    ad = glob.glob(f"{new['mc']}/wfield_local_results/allen_aligned_affine8v1")[0]
    U = np.load(f"{ad}/U_atlas.npy"); SVT = np.load(f"{new['mc']}/wfield_local_results/SVTcorr.npy")
    A2 = np.nan_to_num(A.reshape(-1, A.shape[2]))            # (npix, ncomp); A is NaN outside the brain
    Uf = np.nan_to_num(U.reshape(-1, U.shape[2]))            # (npix, K)
    # REASSOCIATE: pinv(A) @ (U @ SVT) == (pinv(A) @ U) @ SVT. The bracketed form never materializes
    # the (npix x T) movie -- 345600 x ~200k float32 is ~276 GB and would OOM instantly. The
    # reassociated form contracts to (ncomp x K) first, which is ~150 x 100.
    return (np.linalg.pinv(A2) @ Uf) @ SVT                   # (ncomp, T) = refit C on the frozen basis


def _transfer_matrix_fig(out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    posn = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    for ax, an in zip(axes[0], ("PS92", "PS94", "PS95")):
        days = [s["label"] for s in SESSIONS if s["label"].startswith(an) and s["label"][-4:] in BASELINE_DAYS]
        n = len(days); M = np.full((n, n), np.nan)
        for i, tr in enumerate(days):
            for j, te in enumerate(days):
                try:
                    if i == j:  # within-day: honest block-CV
                        X, y, g, _, _, r = _trial_features(_sess(tr), _args("roi", "lick", 2.0))
                        ng = min(5, int(np.unique(g).size))
                        M[i, j] = accuracy_score(y, cross_val_predict(_pipe(), X, y, cv=GroupKFold(ng), groups=g))
                    else:
                        M[i, j] = cross_day_transfer(tr, te, source="roi")
                except Exception as e:
                    print(f"  {an} {tr}->{te} failed: {str(e)[:50]}")
        im = ax.imshow(M, vmin=1 / 6, vmax=0.9, cmap="viridis")
        for i in range(n):
            for j in range(n):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            color="w" if M[i, j] < 0.6 else "k", fontsize=10)
        dl = [d[-4:-2] + "/" + d[-2:] for d in days]
        ax.set_xticks(range(n)); ax.set_xticklabels(dl, fontsize=8); ax.set_yticks(range(n)); ax.set_yticklabels(dl, fontsize=8)
        ax.set_xlabel("test day"); ax.set_ylabel("train day"); ax.set_title(an, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Frozen ROI decoder cross-day transfer (diagonal = within-day block-CV; off-diagonal = train->test). "
                 "The off-diagonal drop is the baseline cost of freezing a decoder across days.", fontsize=10.5)
    fig.tight_layout(); p = out / "locanmf_frozen_decoder_crossday_roi.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true", help="fit + persist decoders for all baseline sessions to MICROSCOPE")
    ap.add_argument("--transfer", action="store_true", help="cross-day ROI transfer demo + figure")
    ap.add_argument("--loso", action="store_true",
                    help="pooled leave-one-SESSION-out frozen ROI decoder + OOD control, per animal "
                         "(over the curated set) -> summary JSON + figure")
    ap.add_argument("--animals", nargs="+", default=None, help="animals for --loso (default: all in animals.yaml)")
    ap.add_argument("--align", default="cue", choices=("cue", "lick", "precue"), help="alignment for --loso")
    ap.add_argument("--output", type=Path, default=Path("."))
    args = ap.parse_args()
    labels = [s["label"] for s in SESSIONS if s["label"][-4:] in BASELINE_DAYS]
    if args.save:
        for lab in labels:
            for source in ("locanmf", "roi"):
                try:
                    p, acc = save_session_decoder(lab, source=source)
                    print(f"saved {p.name}  (block-CV acc={acc:.2f})", flush=True)
                except Exception as e:
                    print(f"FAILED {lab} {source}: {str(e)[:60]}", flush=True)
    if args.transfer:
        args.output.mkdir(parents=True, exist_ok=True)
        print("wrote", _transfer_matrix_fig(args.output), flush=True)
    if args.loso:
        args.output.mkdir(parents=True, exist_ok=True)
        dates = set(config.curated_dates())
        animals = args.animals or list(config.animals())
        results = {}
        for an in animals:
            labs = [s["label"] for s in SESSIONS
                    if s["label"].startswith(an) and s["label"][-4:] in dates]
            if len(labs) < 2:
                print(f"[loso] {an}: <2 curated sessions -> skipped", flush=True)
                continue
            print(f"\n=== {an}: {len(labs)} curated sessions ===", flush=True)
            r = pooled_frozen_loso(labs, source="roi", align=args.align)
            if r:
                results[an] = r
        if results:
            (args.output / f"locanmf_frozen_decoder_loso_roi_{args.align}.json").write_text(
                json.dumps(results, indent=2, default=float))
            n = len(write_session_confusions(results, args.output))
            print(f"\nwrote {n} per-held-out-day confusion figure(s)", flush=True)
            print("wrote", _loso_fig(results, args.output), flush=True)
        # frozen ENCODER over the same pooled sessions (position -> activity, applied to an unseen day)
        enc = {}
        for an in animals:
            labs = [s["label"] for s in SESSIONS
                    if s["label"].startswith(an) and s["label"][-4:] in dates]
            if len(labs) < 2:
                continue
            print(f"\n=== {an}: frozen ENCODER ===", flush=True)
            r = pooled_frozen_encoder(labs, source="roi", align=args.align)
            if r:
                enc[an] = r
        if enc:
            (args.output / f"locanmf_frozen_encoder_loso_roi_{args.align}.json").write_text(
                json.dumps(enc, indent=2, default=float))
            print("wrote", _encoder_fig(enc, args.output), flush=True)
    if not (args.save or args.transfer or args.loso):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
