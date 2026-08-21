"""Per-position pre-stroke coding directions, and where the post-stroke failure classes fall on them.

Priya, 2026-08-20. REPORTED, not a filter: nothing here changes what any existing analysis uses.

VOCABULARY. "pre-cue"/"post-cue" collided with "pre-stroke"/"post-stroke", so the WINDOW is named
**ENL / cue / lick** and the PHASE stays **pre-stroke / post-stroke**. The internal align tokens are
unchanged (``precue``/``cue``/``lick``) because they are baked into 292 figure filenames, the
``section_g.json`` keys and the per-session caches; renaming those would strand every existing
figure under a name nothing reads, which is the exact failure this deck spent 2026-08-20 clearing.

THE DIRECTION. For each spout position P, a direction is fitted on PRE-STROKE trials WITH A
SUCCESSFUL LICK in the response window: P against the other positions. It is therefore "the pattern
that precedes/accompanies a successful lick to P" -- defined entirely where behaviour is intact.

WHY ONE DIRECTION PER POSITION RATHER THAN ONE ENGAGEMENT AXIS. The two post-stroke phenomena differ
enormously in position composition -- MISS-WHILE-WORKING is 34-44% far_R, STOPPED is near-uniform, total
variation 0.31-0.65 between them -- and ENL activity CARRIES position. A single position-blind axis
therefore compares the spout, not the state, which is what produced a spurious PS95 "effect" on the
first pass. Fitting per position and comparing only WITHIN a position removes that by construction.

THE TWO POST-STROKE FAILURE MODES (Priya, 2026-08-20):
  MISS WHILE WORKING  the animal is still working the task and fails to lick at THIS position.
                      Position-specific, and graded by SEVERITY: far_R > far_center > far_L >
                      close_R > close_center > close_L -- contraversive within each ring, far
                      worse than close throughout.
  STOPPED             the animal has quit for the day and licks nowhere. Verified position-GENERAL:
                      inside that window the response rate is ~0 at every position, close included.

WHAT EACH WINDOW CAN ANSWER, AND WHAT IT CANNOT:
  ENL   all five classes. Nothing has happened yet, and the window is already lick-free by
        construction (``decode.precue_lickfree``), so it is the clean one.
  cue   all five classes. Note that a lick trial contains its lick from ~140 ms (median first-lick
        latency is 0.137-0.255 s pre-stroke, minimum 0.109 s), so there is NO movement-free cue
        window to retreat to. The per-position construction is what keeps this interpretable:
        movement is common to every training class, so it cannot define the direction.
  lick  ONLY the classes that have licks. A no-lick trial has no lick to align to. This matters
        operationally: at ``align="lick"`` ``_trial_features`` still RETURNS no-lick trials, but
        referenced to the cue instead -- so they arrive populated, plausible, and on a different
        alignment from everything they would be compared with. They are excluded explicitly here
        rather than assumed absent (the same trap as the post-lick confusion bug of 2026-08-20).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_frozen_decoder import pool_sessions
from wfield_local.paths import PathResolver
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
from wfield_local.precue_engagement_states import (
    _auc,
    _disc,
    _positions_for,
    _session_tables,
    engagement_gate,
    features_with_indices,
)

#: internal align token -> the name used in every figure, label and printed line.
ALIGNS = (("precue", "ENL"), ("cue", "cue"), ("lick", "lick"))

#: most impaired first (Priya, 2026-08-20).
SEVERITY = {"far_R": 1, "far_center": 2, "far_L": 3,
            "close_R": 4, "close_center": 5, "close_L": 6}
BY_SEVERITY = [p for p, _ in sorted(SEVERITY.items(), key=lambda kv: kv[1])]

#: classes each window can carry. `lick` drops both no-lick classes: no lick, no alignment point.
CLASSES_FULL = ("prestroke_lick", "prestroke_nolick", "poststroke_lick", "poststroke_miss_working", "poststroke_stopped")
CLASSES_LICK = ("prestroke_lick", "poststroke_lick")
MIN_TRIALS = 12


def _gate_all(feat, kept, XE, YE, GE, XU, YU, GU):
    """`not_eng` per no-lick trial, or None if the bookkeeping does not line up."""
    tables = _session_tables(feat, kept)
    if not tables:
        return None
    not_eng = []
    for si in range(len(kept)):
        t = tables.get(si)
        if t is None:
            continue
        ie, inl = feat.indices[kept[si]]
        pos = _positions_for(tables, si, YE, GE, YU, GU, ie, inl)
        ne = engagement_gate(t["order"], t["responded"], pos)
        bne = {int(k): bool(v) for k, v in zip(t["order"], ne)}
        not_eng += [bne.get(int(k), False) for k in inl]
    not_eng = np.array(not_eng, bool)
    return not_eng if len(not_eng) == len(XU) else None


def run_animal(animal, align="precue", verbose=True):
    """Per-position directions for one animal at one window. ``None`` if unavailable."""
    disp = dict(ALIGNS)[align]
    post_s = float(config.defaults()["decode"].get(f"{align}_post_s", 2.0))
    pre = [l for l in config.phase_labels("pre") if l.startswith(animal)]
    post = [l for l in config.phase_labels("post") if l.startswith(animal)]
    if not pre or not post:
        return None
    try:
        from wfield_local.locanmf_cue_lick_analysis import SESSIONS as _ALL
        basis = joint_locanmf.load(animal, sessions=_ALL)
    except FileNotFoundError as ex:
        print(f"[coding_dirs] {animal}: {ex}", flush=True)
        return None

    feat = features_with_indices(basis)
    pooled = pool_sessions(pre + post, source="locanmf", align=align, post_s=post_s, features=feat)
    if pooled is None:
        return None
    XE, YE, GE, _BE, XU, YU, kept, _c, GU = pooled
    YU = YU.astype(int)
    pre_i = {i for i, l in enumerate(kept) if l in set(pre)}
    e_pre, u_pre = np.isin(GE, list(pre_i)), np.isin(GU, list(pre_i))
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
    un = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YU])

    use_nolick = align != "lick"
    not_eng = None
    if use_nolick:
        not_eng = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
        if not_eng is None:
            print(f"[coding_dirs] {animal} {disp}: trial bookkeeping mismatch -- skipped", flush=True)
            return None

    out = {"animal": animal, "align": align, "window": disp, "basis_id": basis.basis_id,
           "ncomp": int(basis.ncomp), "positions": {}}
    for P in BY_SEVERITY:
        # TRAIN: pre-stroke, SUCCESSFUL LICK only -- XE is by definition the engaged (licked) arm.
        trX, trY, trG = XE[e_pre], (en[e_pre] == P).astype(int), GE[e_pre]
        rec = {"severity_rank": SEVERITY[P], "n_train_pos": int(trY.sum()),
               "n_train_neg": int((1 - trY).sum()), "classes": {}}
        if trY.sum() < 20 or (1 - trY).sum() < 20 or len(np.unique(trG)) < 2:
            out["positions"][P] = rec
            continue
        model = _disc().fit(trX, trY)
        cv = cross_val_predict(_disc(), trX, trY, cv=LeaveOneGroupOut(), groups=trG,
                               method="predict_proba")[:, 1]
        rec["auc_loso"] = _auc(trY.astype(bool), cv)

        def score(X, _m=model):     # bind the loop's model explicitly, not by closure
            return (float(np.mean(_m.predict_proba(X)[:, 1])) if len(X) >= MIN_TRIALS else None)

        # pre-stroke lick at P: HELD OUT, or it would sit at its own training optimum
        sel = e_pre & (en == P)
        rec["classes"]["prestroke_lick"] = {
            "n": int(sel.sum()),
            "score": (float(np.mean(cv[(en[e_pre] == P)])) if sel.sum() >= MIN_TRIALS else None)}
        m = ~e_pre & (en == P)
        rec["classes"]["poststroke_lick"] = {"n": int(m.sum()), "score": score(XE[m])}
        if use_nolick:
            m = u_pre & (un == P)
            rec["classes"]["prestroke_nolick"] = {"n": int(m.sum()), "score": score(XU[m])}
            m = ~u_pre & ~not_eng & (un == P)
            rec["classes"]["poststroke_miss_working"] = {"n": int(m.sum()), "score": score(XU[m])}
            m = ~u_pre & not_eng & (un == P)
            rec["classes"]["poststroke_stopped"] = {"n": int(m.sum()), "score": score(XU[m])}
        out["positions"][P] = rec

    if verbose:
        got = [P for P in BY_SEVERITY if out["positions"][P].get("auc_loso")]
        print(f"  {animal} [{disp}]: directions for {len(got)}/6 positions", flush=True)
    return out


def figure(results, out, align="precue"):
    """One panel per animal: position (most impaired first) x class, on that position's direction."""
    disp = dict(ALIGNS)[align]
    animals = [a for a in sorted(results) if results[a]]
    if not animals:
        return None
    classes = CLASSES_LICK if align == "lick" else CLASSES_FULL
    style = {"prestroke_lick": ("tab:blue", "o", "pre-stroke LICK (held out)"),
             "prestroke_nolick": ("tab:grey", "s", "pre-stroke NO-LICK (sated / not working)"),
             "poststroke_lick": ("tab:green", "^", "post-stroke LICK"),
             "poststroke_miss_working": ("tab:red", "D", "post-stroke MISS while still working"),
             "poststroke_stopped": ("tab:purple", "v", "post-stroke STOPPED (quit for the day)")}
    fig, axes = plt.subplots(1, len(animals), figsize=(4.6 * len(animals) + 1.2, 4.6),
                             squeeze=False, sharey=True)
    x = np.arange(len(BY_SEVERITY))
    for k, an in enumerate(animals):
        ax = axes[0][k]
        pos = results[an]["positions"]
        for c in classes:
            col, mk, lab = style[c]
            ys = [((pos.get(P, {}).get("classes", {}).get(c) or {}).get("score")) for P in BY_SEVERITY]
            ax.plot(x, [np.nan if y is None else y for y in ys], "-", marker=mk, color=col,
                    label=lab if k == 0 else None, lw=1.6, ms=6)
        ax.set_xticks(x)
        ax.set_xticklabels(BY_SEVERITY, rotation=45, ha="right", fontsize=8)
        ax.set_title(an, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(f"P(this position) on its own {disp} direction")
    fig.legend(loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(
        f"{disp} window: each spout position's PRE-STROKE successful-lick coding direction, and where "
        f"each class falls on it.\nPositions ordered MOST IMPAIRED first. Compared only WITHIN a "
        f"position, so class position-composition cannot contribute.", fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.90))
    p = Path(out) / f"coding_direction_{disp}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--windows", nargs="+", default=["ENL", "cue", "lick"],
                    choices=("ENL", "cue", "lick"))
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    out.mkdir(parents=True, exist_ok=True)
    animals = config.normalize_animals(args.animals) or [a for a in config.animals()]
    want = set(args.windows)
    everything = {}
    for align, disp in ALIGNS:
        if disp not in want:
            continue
        print(f"=== {disp} window (align={align}) ===", flush=True)
        res = {}
        for an in animals:
            try:
                res[an] = run_animal(an, align=align)
            except Exception as ex:                                       # noqa: BLE001
                print(f"  !! {an} [{disp}]: {type(ex).__name__} {str(ex)[:90]}", flush=True)
                res[an] = None
        everything[disp] = res
        p = figure(res, out, align=align)
        print(f"  wrote {p}", flush=True)
    (out / "coding_direction.json").write_text(
        json.dumps(everything, indent=1, default=float), encoding="utf-8")
    print("wrote coding_direction.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
