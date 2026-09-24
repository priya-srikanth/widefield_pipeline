"""Is the ENL position code a SENSORY code, or the tail of a spout-arrival lick?

Priya, 2026-09-24: *"I'd like to look at ENL decoding for success and miss-while-working trials pre-
and post-stroke, divided by trials that had a pre-ENL lick within 1s of the decoding window and
those that did NOT"* -- and, on the framing: *"the pre-ENL-lick vs no-pre-ENL-lick can be on the hit
and working trials. The success vs miss while working ENL decoding is a SEPARATE question."*

So this module asks ONE question, WITHIN each arm: does a lick in the second before the decoding
window change what the decoder reads out of that window? It deliberately does NOT re-report the
`success`-vs-`miss_working` ladder -- that is `enl_decode` readout 4, a different question, and
mixing the two would let a cross-arm difference be read as a contamination effect.

WHY THIS IS NOT A MINOR CONTROL. Measured over 104 curated sessions
(`scripts/enl_lick_rates.py`, 2026-09-24):

    a lick INSIDE the window      0.1 - 8.3%   -- caught by `decode.precue_lickfree`, which slides
                                                 the window to a clean gap or drops the trial
    a lick in the second BEFORE  13.7 - 90.9%  -- caught by NOTHING, and the MAJORITY of trials in
                                                 three of four animals

The window is ``[cue - 2s, cue]`` and the spout arrives ~3 s before the cue, so the second before
the window IS the first second after spout arrival. This is spout-arrival licking.

**AND IT IS DIRECTED AT THE SPOUT, SO IT CARRIES POSITION.** That is what makes it a confound rather
than noise: the contaminating signal is correlated with the very label being decoded. A haemodynamic
response peaks 1-2 s after the lick -- inside the window -- so a window that is lick-free by
construction can still contain a position-informative MOTOR signal. If the ENL code were mostly
that, "the pre-cue code is not only a held plan" would be true for the wrong reason.

THE DESIGN, and why the decoder trains on CLEAN trials only:

  train    `success` trials with NO lead lick, blocks held out across every arm
  score    success_clean (held out) · success_lead · miss_working_clean · miss_working_lead

Training on clean trials forces the decoder onto information that is present without a preceding
lick. Then, per arm:

    clean ~ lead    the lick is not what the decoder is reading. The control passes.
    lead >> clean   the lick ADDS position information -- the confound is real and sized.
    clean at null   the code is present only when a lick preceded it. Worst case, and the one this
                    module exists to be able to state.

`transfer_lead_to_clean` runs the reverse -- train on LEAD trials, score CLEAN ones. If a
lick-trained decoder reads lick-free trials above null, the two share a representation rather than
the lick-trained one having learnt a movement artefact.

POST-STROKE IS DESCRIPTIVE ONLY. `enl_states.witness_verdict` has not been run, so no post-stroke
cell here supports a claim about whether a plan was formed. As a CONTAMINATION control the epoch
matters less -- the question is about licking, not engagement -- but the stamp stays.

CLI::

    python -m wfield_local.enl_lick_control --epoch pre
    python -m wfield_local.enl_lick_control --all-epochs --out <json>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import enl_decode as ed
from wfield_local import nolick_analysis as na

#: The arms the split runs on. `stopped` is excluded deliberately: Priya framed this control as
#: being about hit and working trials, and `stopped` is already the scarcest arm in every animal
#: (6/40/326/495 pre-stroke) -- splitting it in two would leave nothing to test either half with.
ARMS = ("success", "miss_working")

#: Group names, in the order they are reported. `clean` first because it is the reference.
GROUPS = ("clean", "lead")

#: Below this many trials a group is reported as UNDERPOWERED rather than as a result. Lower than
#: `enl_decode.MIN_TRIALS` (500) because these arms are SCORED by a decoder trained elsewhere, not
#: fitted -- the same reason `enl_decode.underpowered` exempts the transfer readout. It still has to
#: be enough to estimate six per-position recalls, which is what 120 is for: ~20 a position.
MIN_SCORED = 120


def split_arms(pooled):
    """``{f"{arm}_{group}": arm-dict}`` -- each arm cut by its own `lead_lick` flag.

    The flag rides with the trials from `nolick_decoder.session_features`, which computed it against
    THAT TRIAL'S OWN window start. It is not recomputed here, and must not be: a slid window makes
    "1 s before the cue" and "1 s before the window" different intervals on exactly the trials that
    licked, and a mask rebuilt outside the function that defines the trial set is the failure
    `locanmf_position_decoder` records three times over.
    """
    out = {}
    for arm in ARMS:
        a = pooled.get(arm)
        if a is None or not len(a["y"]):
            continue
        lead = np.asarray(a.get("lead_lick", np.zeros(len(a["y"]), bool)), bool)
        for grp, m in (("clean", ~lead), ("lead", lead)):
            out[f"{arm}_{grp}"] = {"X": a["X"][m], "y": a["y"][m], "g": np.asarray(a["g"])[m]}
    return out


def analyse(pooled, n_perm=na.N_PERM, n_boot=2000):
    """The contamination control for one animal x epoch. Never raises on a thin group."""
    groups = split_arms(pooled)
    res = {"n": {k: int(len(v["y"])) for k, v in groups.items()},
           "lead_fraction": {}}
    for arm in ARMS:
        c, ll = groups.get(f"{arm}_clean"), groups.get(f"{arm}_lead")
        tot = (len(c["y"]) if c else 0) + (len(ll["y"]) if ll else 0)
        res["lead_fraction"][arm] = (float(len(ll["y"]) / tot) if tot and ll else
                                     (0.0 if tot else float("nan")))
    res["power"] = {k: {"underpowered": len(v["y"]) < MIN_SCORED,
                        "reason": f"{len(v['y'])} scored trials < {MIN_SCORED}"}
                    for k, v in groups.items()}

    train = groups.get("success_clean")
    if train is None or not len(train["y"]):
        res["skipped"] = "no lick-free success trials to train on"
        return res

    order = [f"{a}_{g}" for a in ARMS for g in GROUPS if f"{a}_{g}" in groups]
    labels = ed.common_positions([groups[k] for k in order])
    tf = ed.shared_profile([groups[k]["y"] for k in order], labels=labels)
    preds, cover = ed.shared_decoder(train, {k: groups[k] for k in order})
    if preds is None:
        res["skipped"] = "clean success arm is not cross-validatable"
        return res
    scored = {k: ed.scored_arm(groups[k], preds[k], labels=labels) for k in order}
    res["arms"] = {k: ed._score_shared(scored[k], tf, n_perm, labels) for k in order}
    res["coverage"] = cover
    res["positions"] = {"used": [na.POSITION_NAMES[c] for c in labels],
                        "dropped": [na.POSITION_NAMES[c] for c in na.DISPLAY_ORDER
                                    if c not in labels]}

    # WITHIN-ARM pairs only. The cross-arm comparison (`success` vs `miss_working`) is a separate
    # question and `enl_decode` readout 4 answers it; reporting it here would let a cross-arm
    # difference be misread as a contamination effect.
    pairs = [(f"{a}_lead", f"{a}_clean") for a in ARMS
             if f"{a}_lead" in scored and f"{a}_clean" in scored]
    nulls = {k: res["arms"][k].get("bal_null_mean", float("nan")) for k in order}
    res["bootstrap"] = ed.bootstrap_arms(scored, nulls, labels, n_boot=n_boot, pairs=pairs)

    # The reverse direction: does a LICK-TRAINED decoder read LICK-FREE trials?
    lead_train = groups.get("success_lead")
    if lead_train is not None and len(lead_train["y"]) and scored.get("success_clean") is not None:
        p = ed.transfer(lead_train, groups["success_clean"])
        res["transfer_lead_to_clean"] = (
            na.evaluate_arm(groups["success_clean"]["y"], p, target_frac=tf, n_perm=n_perm,
                            labels=list(labels))
            if p is not None else {"skipped": "no usable lead arm"})
    return res


def analyse_animal(animal, epoch="pre", *, source=ed.POOLED_SOURCE, post_s=2.0, n_perm=na.N_PERM,
                   n_boot=2000, verbose=True):
    """One animal x epoch. Reuses `enl_decode`'s session loop so the trial sets are identical."""
    labs = [s["label"] for s in ed.sessions_for(animal, epoch)]
    if not labs:
        return {"animal": animal, "epoch": epoch, "skipped": "no curated sessions"}
    from wfield_local import config
    basis, per, errs = None, [], []
    if source == "joint":
        from wfield_local import joint_locanmf
        try:
            basis = joint_locanmf.load(animal)
        except Exception as exc:
            errs.append(f"no joint basis for {animal} ({type(exc).__name__})")
            source = ed.FALLBACK_SOURCE
    for lab in labs:
        sess = next((x for x in config.load_sessions() if x["label"] == lab), None)
        if sess is None:
            errs.append(f"{lab}: no session record")
            continue
        try:
            per.append(ed.arms_for_session(sess, align=ed.ALIGN, source=source, post_s=post_s,
                                           basis=basis))
        except Exception as exc:
            errs.append(f"{lab}: {type(exc).__name__}: {exc}"[:140])
    if not per:
        return {"animal": animal, "epoch": epoch, "skipped": "no usable sessions", "errors": errs}

    out = analyse(ed.pool_arms(per), n_perm=n_perm, n_boot=n_boot)
    out.update({"animal": animal, "epoch": epoch, "align": ed.ALIGN, "source": source,
                "basis_id": getattr(basis, "basis_id", None), "ncomp": getattr(basis, "ncomp", None),
                "n_sessions": len(per), "errors": errs})
    if verbose:
        report(out)
    return out


def report(r, fh=None):
    """Human-readable. The VERDICT line states what the control shows, or why it cannot."""
    pr = (lambda *a: print(*a, file=fh, flush=True)) if fh else (lambda *a: print(*a, flush=True))
    bas = "" if not r.get("basis_id") else f" {str(r['basis_id'])[:12]} {r.get('ncomp')}c"
    pr("")
    pr(f"=== {r.get('animal')} {r.get('epoch')} [lead-lick control{bas}] "
       f"{r.get('n_sessions', 0)} session(s) ===")
    if r.get("skipped"):
        pr(f"  SKIPPED: {r['skipped']}")
        return
    for e in r.get("errors", [])[:5]:
        pr(f"  ! {e}")
    for arm in ARMS:
        f = r["lead_fraction"].get(arm, float("nan"))
        pr(f"  {arm:13s} lead-lick on {f:.1%} of trials")
    drop = r.get("positions", {}).get("dropped") or []
    if drop:
        pr(f"  POSITIONS RESTRICTED to {6 - len(drop)}/6 shared by every group; "
           f"dropped {', '.join(drop)}")
    for k, d in (r.get("arms") or {}).items():
        if "skipped" in d:
            pr(f"  {k:22s} skipped: {d['skipped']}")
            continue
        ci = r.get("bootstrap", {}).get("arms", {}).get(k, {}).get("ci")
        cis = "" if not ci else f"  CI [{ci[0]:.3f}, {ci[1]:.3f}]"
        und = " UNDERPOWERED" if r["power"][k]["underpowered"] else ""
        pr(f"  {k:22s} n={d['n']:5d}  bal={d['balanced_accuracy']:.3f}  "
           f"null={d['bal_null_mean']:.3f}  p={d['bal_p']:.4f}  "
           f"{'ABOVE NULL' if d['above_null_balanced'] else 'at null'}{cis}{und}")
    for k, d in (r.get("bootstrap", {}).get("differences") or {}).items():
        verdict = "DIFFER -- lead licks change the readout" if d["p"] < 0.05 else \
                  "not distinguishable"
        pr(f"  diff {k:36s} {d['value']:+.3f}  CI [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  "
           f"p={d['p']:.4f}  {verdict}")
    t = r.get("transfer_lead_to_clean")
    if t and "skipped" not in t:
        pr(f"  transfer lead->clean   bal={t['balanced_accuracy']:.3f} "
           f"null={t['bal_null_mean']:.3f} p={t['bal_p']:.4f} "
           f"{'ABOVE NULL' if t['above_null_balanced'] else 'at null'}")
    pr("  " + verdict_line(r))


def verdict_line(r):
    """One sentence saying what the control established, or that it could not."""
    arms = r.get("arms") or {}
    diffs = r.get("bootstrap", {}).get("differences") or {}
    bits = []
    for arm in ARMS:
        c, ll = arms.get(f"{arm}_clean"), arms.get(f"{arm}_lead")
        if not c or "skipped" in c:
            continue
        if r["power"][f"{arm}_clean"]["underpowered"]:
            bits.append(f"{arm}: clean arm underpowered -- could not test")
            continue
        if not c["above_null_balanced"]:
            bits.append(f"{arm}: CLEAN TRIALS AT NULL -- code present only with a preceding lick")
            continue
        d = diffs.get(f"{arm}_lead_minus_{arm}_clean")
        if d is None or not ll or "skipped" in ll:
            bits.append(f"{arm}: decodable without a preceding lick (lead arm not scored)")
        elif d["p"] >= 0.05:
            bits.append(f"{arm}: decodable without a preceding lick, and lead licks do not "
                        f"change it")
        else:
            sign = "ADDS" if d["value"] > 0 else "REDUCES"
            bits.append(f"{arm}: decodable without a preceding lick, but a lead lick {sign} "
                        f"{abs(d['value']):.3f}")
    return "VERDICT: " + ("; ".join(bits) if bits else "nothing scorable")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch", default="pre")
    ap.add_argument("--all-epochs", action="store_true",
                    help="pre, acute, subacute and chronic (Priya, 2026-09-24)")
    ap.add_argument("--animal", action="append", default=None)
    ap.add_argument("--n-perm", type=int, default=na.N_PERM)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--source", default=ed.POOLED_SOURCE)
    ap.add_argument("--post-s", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    animals = args.animal or ["PS92", "PS93", "PS94", "PS95"]
    eps = ("pre", "acute", "subacute", "chronic") if args.all_epochs else (args.epoch,)
    results = [analyse_animal(a, e, source=args.source, post_s=args.post_s,
                              n_perm=args.n_perm, n_boot=args.n_boot)
               for e in eps for a in animals]
    if args.out:
        import json
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
        print(f"\n[enl_lick_control] -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
