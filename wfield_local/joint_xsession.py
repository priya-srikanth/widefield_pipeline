"""Cross-session decoding and encoding in the SHARED joint-LocaNMF basis.

WHY THIS EXISTS. The cross-day arm of the study (train pre-stroke, apply post-stroke) needs features
that mean the same thing on every day. Until now only Allen-ROI features qualified: a session's own
LocaNMF components are session-specific in both count and identity, so ``pooled_frozen_loso`` refuses
them outright. That left the cross-day result resting on ONE parcellation, and on the coarser of the
two -- 66 anatomical ROIs rather than the ~100-140 functionally-defined components that decode better
within a session (LocaNMF beats ROI within-session in 4/4 animals, 0.824 vs 0.763).

The joint basis (``wfield_local.joint_locanmf``) removes that restriction. Its footprints A are fitted
ONCE over an animal's curated sessions and then held fixed, so component j is the same footprint on
every day, and a day that was not in the fit is PROJECTED onto those same footprints rather than
refitted. That makes LocaNMF components poolable across sessions for the first time, and this module
is the cross-session decode/encode built on them.

WHY THIS FITS ITS OWN FROZEN DECODER RATHER THAN REUSING THE ROI ONE (Priya asked, 2026-08-26:
*"why would we want joint_xsession to compute a new frozen decoder? I thought we'd want to use the
same frozen pre-stroke models"*). It is not a duplicate and the two models cannot be shared, because
they do not live in the same space: the ROI frozen decoder is fitted on **264** features (66 Allen
areas x 4 sub-bins) and this one on **380** joint components. A weight vector over 264 ROI features
is not applicable to a 380-dimensional component vector -- there is no mapping to reuse. "The same
frozen model in both bases" is not a thing that can exist.

What IS shared, and what makes them comparable, is everything else: the same pre-stroke training
sessions, the same window and alignment, the same `_pipe()`, and since 2026-08-26 the same
pre-stroke-only training restriction inside `pooled_frozen_loso`. So the two are the SAME analysis
run through two parcellations, which is the point -- DECISIONS 2026-08-12 requires decoders and
encoders in BOTH bases precisely so that a result appearing in only one can be recognised as a basis
artefact rather than a finding. Agreement between them is evidence; a single shared model would
destroy the check by construction.

THE FROZEN MODEL IS NOW AN OBJECT, not a recipe (2026-08-27). This paragraph used to record the gap:
several modules each fitted their own pre-stroke decoder and agreed only because they shared `_pipe()`
and the same trial conventions, with nothing enforcing it. `pooled_frozen_loso` and
`pooled_frozen_encoder` now load a stored artefact from `wfield_local.frozen_models`, keyed on the
PRE-STROKE training set and its input signatures, so adding a post-stroke session cannot change the
model and adding a pre-stroke one mints a new id rather than moving the old one.

Note that a ROI frozen decoder and a joint one still cannot be the same artefact -- as above, 264
features against 380, no mapping between them -- and the spec records `source` and `basis_id` so the
two can never be served for each other. The remaining unpackaged fitters (`grant_figures`,
`nolick_decoder`, `ood_control`, `poststroke_compare`) are the next step, not this one.

WHAT THIS IS NOT. Not the rejected frozen fixed-A path. That nominated a single session as the
reference, and the choice mattered enormously -- no reference won for every animal and the within-animal
swing reached 0.36, i.e. tuning a free parameter on the outcome. The joint basis is reference-FREE: it
is fitted over all the curated sessions jointly, so there is no knob to tune. See DECISIONS.md
"Cross-session RDM basis: what we tested and what we rejected".

HOW TO READ THE OUTPUT. Identical in form to the ROI cross-day figures, deliberately, so the two read
side by side and the ONLY difference between them is the feature basis:
  * held-out-day accuracy vs that day's own within-day ceiling -> the cost of freezing across days,
  * per-held-out-day confusion matrices,
  * the encoder's held-out EV against that day's noise ceiling, and FEVE.

Plus one diagnostic the ROI version does not need: ``variance_captured``, the fraction of a projected
session's energy that survives projection onto the joint subspace. A session in the fit is 1.0 by
construction; a NEW session (8/12 here, and every post-stroke session later) is not, and a low value
means its components under-describe that day rather than that its representation changed. It is
reported per session and plotted, because a frozen basis applied to new data without that number is an
unfalsifiable claim.

    python -m wfield_local.joint_xsession --output <dir> [--align cue precue] [--animals PS92 ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_frozen_decoder import (
    _encoder_fig,
    _loso_fig,
    pooled_frozen_encoder,
    pooled_frozen_loso,
    write_animal_confusion_grid,
    write_session_confusions,
)
from wfield_local.locanmf_position_decoder import _trial_features, trial_features_cached

BASIS = "joint"

# Trial features are cached per (session, basis, align, window) rather than the SIGNALS they come
# from: projecting a session costs a U/SVT load and the (ncomp, T) result is ~100 MB, while the trial
# matrix is a few hundred rows. Decode and encode both pool the same sessions, so without this each
# session would be projected four times per animal.
_FEAT_CACHE: dict = {}


def joint_features(basis):
    """A ``features`` callable for :func:`locanmf_frozen_decoder.pool_sessions`.

    Uses ``signal_or_project`` so a session in the fit keeps its fitted time courses and a new one is
    projected onto the SAME fixed footprints -- never refitted, which would silently change the basis.
    """
    vc = {}

    def _feat(s, args):
        # TWO TIERS, deliberately. The dict below is the IN-PROCESS tier and stays: within one run it
        # avoids re-reading and re-unpickling an entry that is already in memory. `trial_features_cached`
        # is the ON-DISK tier underneath it, and is the one that matters -- the nightly is 17 separate
        # processes, so nothing in this dict survives a step boundary, and per-session features do not
        # depend on which other sessions exist, so they need not be rebuilt between nights either.
        key = (s["label"], basis.basis_id, args.align, args.post_s)
        if key not in _FEAT_CACHE:
            lab = s["label"]
            src = joint_locanmf.BasisSource(basis, s)
            _FEAT_CACHE[key] = trial_features_cached(
                s, args, signal_fn=src.signal, signal_key=src.key)
            vc[lab] = src.variance_captured()
        return _FEAT_CACHE[key]

    _feat.variance_captured = vc
    _feat.basis_id = basis.basis_id      # see precue_engagement_states: the feature space is identity
    return _feat


def run_animal(animal, labels, align="cue", post_s=2.0, verbose=True):
    """Joint-basis LOSO decoder + encoder for one animal. ``None`` if it has no basis / <2 sessions."""
    try:
        from wfield_local.locanmf_cue_lick_analysis import SESSIONS as _ALL
        basis = joint_locanmf.load(animal, sessions=_ALL)   # warns if it predates the inputs
    except FileNotFoundError as ex:
        print(f"[joint_xsession] {animal}: {ex}", flush=True)
        return None
    feat = joint_features(basis)
    dec = pooled_frozen_loso(labels, source="locanmf", align=align, post_s=post_s,
                             verbose=verbose, features=feat)
    enc = pooled_frozen_encoder(labels, source="locanmf", align=align, post_s=post_s,
                                verbose=verbose, features=feat)
    if dec is None and enc is None:
        return None
    meta = {"basis_id": basis.basis_id, "ncomp": basis.ncomp,
            "basis_labels": basis.labels,
            "variance_captured": dict(feat.variance_captured),
            "projected": [l for l in (dec or enc)["labels"] if l not in basis.labels]}
    for r in (dec, enc):
        if r is not None:
            r.update(meta)
    return {"decoder": dec, "encoder": enc, **meta}


def fig_basis_health(results, out, align="cue"):
    """The diagnostic a frozen basis owes you: how much of each session it actually spans.

    Sessions IN the fit sit at 1.0 by construction and are drawn hollow; a PROJECTED session (a new
    pre-stroke day now, every post-stroke day later) is drawn filled, and its height is the fraction of
    that session's own energy the frozen footprints can carry. A projected day that decodes poorly AND
    sits low here is under-described by the basis; one that decodes poorly while sitting high is a real
    change in the representation. Without this panel the two are indistinguishable.
    """
    animals = [a for a in results if results[a]]
    if not animals:
        return None
    fig, ax = plt.subplots(figsize=(11, 4.2))
    off, ticks, tlabs = 0, [], []
    for an in animals:
        r = results[an]
        vc = r.get("variance_captured", {})
        infit = set(r.get("basis_labels", []))
        start = off
        for lab in sorted(vc):
            inside = lab in infit
            ax.bar(off, 100 * vc[lab], 0.75,
                   color="none" if inside else "#3b7dd8",
                   edgecolor="#3b7dd8", lw=1.4, hatch="" if inside else None)
            ticks.append(off); tlabs.append(f"{lab[-4:-2]}/{lab[-2:]}"); off += 1
        ax.annotate(f"{an}  ({r.get('ncomp', '?')} comps)",
                    xy=((start + off - 1) / 2, 1.02), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=9.5, fontweight="bold")
        if an != animals[-1]:
            ax.axvline(off + 0.3, color="0.85", lw=1)
        off += 1.2
    ax.bar([], [], color="none", edgecolor="#3b7dd8", lw=1.4, label="in the fit (1.0 by construction)")
    ax.bar([], [], color="#3b7dd8", edgecolor="#3b7dd8", label="PROJECTED onto the frozen footprints")
    ax.set_xticks(ticks); ax.set_xticklabels(tlabs, rotation=90, fontsize=7)
    ax.set_ylabel("% of session energy captured\nby the joint basis")
    ax.set_ylim(0, 105); ax.legend(fontsize=8, loc="lower right")
    # THE ALIGNMENT IS PART OF THE MEASUREMENT, not just the filename: the fraction of a session's
    # energy the footprints span is computed on the ALIGNED window, so the cue and pre-cue figures
    # are different numbers. `align` reached only `joint_basis_health_{align}.png`, so the two were
    # captioned identically and the deck (which shows the pre-cue one alone) said nothing either.
    # Audited 2026-08-24.
    _win = {"precue": "pre-cue", "cue": "post-cue", "lick": "post-lick"}.get(align, align)
    ax.set_title(f"Joint-basis health ({_win} window): how much of each "
                 "session the frozen footprints span\n"
                 "(a low PROJECTED bar means the components under-describe that day — read its "
                 "decode accuracy accordingly)", fontsize=10.5, pad=26)
    fig.tight_layout()
    p = Path(out) / f"joint_basis_health_{align}.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--align", nargs="+", default=["cue", "precue"], choices=("cue", "precue", "lick"))
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--from", dest="from_dates", default=None,
                    help="date spec for the pooled sessions (default: the curated set)")
    args = ap.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    dates = (set(config.expand_dates(args.from_dates, width=4)) if args.from_dates
             else set(config.curated_dates()))
    # default from SESSIONS, not config.animals(): SESSIONS already honours WIDEFIELD_ONLY_ANIMALS,
    # so `nightly_figs --only PS93` scopes this step through the environment like every other one.
    animals = config.normalize_animals(args.animals) or sorted({s["label"][:4] for s in SESSIONS})

    rc = 0
    for align in args.align:
        dec, enc, health = {}, {}, {}
        for an in animals:
            # POOLED = pre + post, never the raw date list: PS92/PS93 0817 is a post-lesion attempt
            # that `session_phase` calls 'excluded', and a date list cannot express a per-animal phase.
            labs = [x for x in config.pooled_labels(an) if x[-4:] in dates]
            if len(labs) < 2:
                print(f"[joint_xsession] {an}: <2 curated sessions -> skipped", flush=True)
                continue
            print(f"\n=== {an} [{align}]: {len(labs)} curated sessions ===", flush=True)
            r = run_animal(an, labs, align=align)
            if not r:
                rc = 1
                continue
            health[an] = r
            if r["decoder"]:
                dec[an] = r["decoder"]
            if r["encoder"]:
                enc[an] = r["encoder"]
        if dec:
            (args.output / f"joint_xsession_decoder_{align}.json").write_text(
                json.dumps(dec, indent=2, default=float))
            n = len(write_session_confusions(dec, args.output, basis=BASIS))
            write_animal_confusion_grid(dec, args.output, basis=BASIS, align=align)
            print("wrote", _loso_fig(dec, args.output, align, basis=BASIS).name,
                  f"and {n} per-day confusion figure(s)", flush=True)
        if enc:
            (args.output / f"joint_xsession_encoder_{align}.json").write_text(
                json.dumps(enc, indent=2, default=float))
            print("wrote", _encoder_fig(enc, args.output, align, basis=BASIS).name, flush=True)
        if health:
            p = fig_basis_health(health, args.output, align)
            if p:
                print("wrote", p.name, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
