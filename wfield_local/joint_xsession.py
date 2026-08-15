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
import numpy as np

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_frozen_decoder import (
    _encoder_fig,
    _loso_fig,
    pooled_frozen_encoder,
    pooled_frozen_loso,
    write_session_confusions,
)
from wfield_local.locanmf_position_decoder import _trial_features

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
        key = (s["label"], basis.basis_id, args.align, args.post_s)
        if key not in _FEAT_CACHE:
            lab = s["label"]
            if lab in basis.labels:
                sig, vc[lab] = basis.signal(lab), 1.0
            else:
                sig, diag = basis.project(s, with_diagnostics=True)
                vc[lab] = float(diag["variance_captured"])
            _FEAT_CACHE[key] = _trial_features(s, args, signal=sig, feat_region=basis.regions)
        return _FEAT_CACHE[key]

    _feat.variance_captured = vc
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
    ax.set_title("Joint-basis health: how much of each session the frozen footprints span\n"
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
            labs = [s["label"] for s in SESSIONS
                    if s["label"].startswith(an) and s["label"][-4:] in dates]
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
