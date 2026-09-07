"""Evaluate the epoch definitions against behaviour, every run. Derives chronic; checks acute.

Priya, 2026-09-07: *"I'd like the pipeline to run the epoch definitions and just determine if we
have met 'chronic' criteria, order sessions into epochs appropriately, and analyze."*

TWO DIFFERENT CONTRACTS, and conflating them is the easiest mistake to make here:

  * `resolve()` DERIVES `chronic_from` and installs it -- sessions really do move between epochs on
    the strength of it, and the figures are built on the result.
  * the acute half REPORTS ONLY. `verify_against_behaviour` re-derives the acute boundary and says
    whether it agrees with the stored `EPOCH_SPEC`; it never reassigns. The acute rule currently
    reproduces its stored boundaries exactly on all four animals, so deriving it would change
    nothing today while adding a way for a published acute boundary to move.

Until 2026-09-07 neither verifier was called by anything. `verify_against_behaviour` was mentioned
only in deck prose and `derive_chronic_boundaries` not at all, so the definitions were checkable in
principle and unchecked in practice.

WHAT MAKES DERIVING SAFE IS THE RECORD, not the rule. `resolve` writes the boundaries it produced to
`epoch_boundaries.json` beside the deck and diffs them against the previous run, so a figure set can
always be asked which epochs it was built from and a run that moved one has to say so. See the
DERIVED BOUNDARIES section of `epochs`.

IT READS THE BEHAVIOUR STAGE'S OWN CSV, and does not re-derive anything from the DAQ. Two reasons,
and the second is the important one:

  * SPEED. Re-parsing every session's trials and lick microstructure takes ~11 minutes; reading the
    cohort table takes milliseconds, and the nightly is already 18-20 h.
  * CONSISTENCY. `cohort_session_metrics.csv` is what the behaviour figures are drawn from. An audit
    that recomputed its own numbers could disagree with the figures it is auditing, and then nobody
    would know which was right. Verified 2026-09-07: `hit__far_R` reproduces an independent
    re-derivation from `load_trials` + `reference_engagement` on all ten of PS94's post-stroke
    sessions, including the ones where gating moves the value by 0.18.

ENGAGED COLUMNS, NOT ALL-TRIAL ONES. `hit__<pos>` is the engagement-gated hit rate and
`hitall__<pos>` the ungated one; `CHRONIC_RULE` and the 2026-09-07 disengagement decision both
specify engaged trials, and the two differ by up to 0.18 post-stroke. Picking the wrong column here
would silently audit a different quantity than the rule defines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wfield_local import epochs
from wfield_local.paths import PathResolver

#: Written by `spout_behavior --cohort`, which the CAMERA/behaviour nightly runs in stage 1 -- ahead
#: of the figs stage this audit is wired into, so it is current by the time it is read.
COHORT_CSV = "cohort/cohort_session_metrics.csv"


def _cohort_path(rv=None) -> Path:
    rv = rv or PathResolver()
    return Path(rv.root("behavior_out")) / COHORT_CSV


def load_far_position_tables(rv=None, position=None):
    """``(hit_by_session, licks_by_session)`` keyed by session LABEL, engaged-gated.

    Both are in the ``{label: {position: value}}`` shape the two verifiers take. Returns
    ``(None, None)`` when the cohort table is missing, so a box that has not run the behaviour stage
    reports "not checked" rather than "no disagreement" -- those must not look alike.

    DUPLICATE (animal, date) ROWS keep the one with the most engaged trials. Two sessions were
    recorded on PS94's 0604, and a label is a date: the alternative is an arbitrary pick that could
    change between runs. No CURRENTLY pooled label is affected (0604 is outside the curated set),
    so this is a guard rather than a live correction.
    """
    import pandas as pd

    position = position or epochs.RULE_POSITION
    fp = _cohort_path(rv)
    if not fp.exists():
        return None, None
    df = pd.read_csv(fp)
    hit_col, lick_col = f"hit__{position}", f"lpt__{position}"
    if hit_col not in df.columns or lick_col not in df.columns:
        return None, None
    if "n_engaged" in df.columns:
        df = df.sort_values("n_engaged").drop_duplicates(["animal", "date"], keep="last")
    hit, lick = {}, {}
    for _i, r in df.iterrows():
        lab = f"{r['animal']}_{str(int(r['date']))[4:]}"
        for tbl, col in ((hit, hit_col), (lick, lick_col)):
            v = r.get(col)
            # A MISSING VALUE MUST NOT BECOME A ZERO. `float(nan)` is a perfectly good float, and a
            # session silently entering the series as 0.0 licks would read as total loss of vigour
            # and could manufacture a plateau failure out of a blank cell.
            if v is not None and pd.notna(v):
                tbl[lab] = {position: float(v)}
    return hit, lick


def audit(rv=None, position=None):
    """Run BOTH verifiers and return ``{"available": bool, "acute": {...}, "chronic": {...}}``.

    ``available`` is False when the cohort table could not be read. Callers must not treat that as
    agreement.
    """
    position = position or epochs.RULE_POSITION
    hit, lick = load_far_position_tables(rv, position)
    if hit is None:
        return {"available": False, "reason": f"no {_cohort_path(rv)}", "position": position}
    return {"available": True, "position": position,
            "csv": str(_cohort_path(rv)), "n_sessions": len(hit),
            "acute": epochs.verify_against_behaviour(hit, position=position),
            "chronic": epochs.derive_chronic_boundaries(hit, lick, position=position)}


def derived_spec(rep) -> dict:
    """``{animal: {"chronic_from": int|None}}`` from an audit report -- what the rule decided.

    Only `chronic_from`. Acute and subacute stay stored (see the `epochs` module docstring), so this
    is deliberately a PARTIAL specification that `epochs.spec_for` merges over the stored one.
    """
    return {animal: {"chronic_from": r.get("derived_day")}
            for animal, r in sorted((rep.get("chronic") or {}).items())}


def changes(new_spec, old_spec) -> list[str]:
    """One line per boundary that MOVED between two runs. Empty when stable.

    This is what replaces the stored spec's guarantee. A stored boundary could not move without
    someone editing it; a derived one can, so the pipeline has to say when it did -- otherwise a
    reader comparing two decks has no way to learn that the epochs under them are not the same.

    ``old_spec`` of None means NO PREVIOUS FILE, and returns empty rather than reporting every
    animal as moved. There is nothing to have moved from: the first run -- and any run after the
    file is deleted or the share is remounted -- would otherwise announce that the epochs changed
    and that the panels are not comparable, which is false and would train the reader to skip the
    one message that matters when it is true.
    """
    if old_spec is None:
        return []
    out = []
    for animal in sorted(set(new_spec) | set(old_spec)):
        a = old_spec.get(animal, {}).get("chronic_from", "absent")
        b = new_spec.get(animal, {}).get("chronic_from", "absent")
        if a != b:
            out.append(f"{animal} chronic_from: {a} -> {b}")
    return out


def resolve(rv=None, position=None, write=True):
    """Derive the boundaries, install them for THIS process, and publish them for the subprocesses.

    Returns ``{"report", "spec", "changes", "path", "available"}``.

    THE FILE IS THE MECHANISM, not a log. `nightly_figs` runs `grant_figures` and
    `epoch_grant_figures` as subprocesses, which re-import `epochs` fresh and pick the boundaries up
    from disk; installing them only in the parent would build half the deck on derived boundaries
    and half on stored ones, and it would render cleanly.

    ON UNAVAILABLE BEHAVIOUR IT WRITES NOTHING AND INSTALLS NOTHING. The stored spec stays in force,
    which is the safe direction: a missing cohort table must not silently move every animal to
    "no chronic epoch" and quietly restage the whole deck.
    """
    rep = audit(rv=rv, position=position)
    if not rep.get("available"):
        return {"available": False, "report": rep, "spec": {}, "changes": [], "path": None}
    spec = derived_spec(rep)
    old = epochs.load_boundaries()
    moved = changes(spec, old)
    path = None
    if write:
        path = epochs.save_boundaries(spec)
    epochs.set_resolved(spec, source=str(path) if path else "derived (not written)")
    return {"available": True, "report": rep, "spec": spec, "changes": moved, "path": path,
            "first_run": old is None}


def disagreements(rep) -> list[str]:
    """One line per animal whose behaviour disagrees with the stored spec. Empty when all agree.

    This is the return value the nightly acts on, and it is deliberately a list of STRINGS rather
    than a boolean: "PS93 chronic: behaviour says day 11, spec says none" is the whole point, and
    a bare False would send someone back to the log to find out which animal and which boundary.
    """
    if not rep.get("available"):
        return []
    out = []
    for animal, r in sorted((rep.get("acute") or {}).items()):
        if r.get("agree") is False:
            bad = ", ".join(f"{d['label']}(d{d['day']}) {d['stored']}->{d['derived']}"
                            for d in r.get("disagreements", [])[:4])
            out.append(f"{animal} acute: {len(r.get('disagreements', []))} session(s) disagree: {bad}")
    for animal, r in sorted((rep.get("chronic") or {}).items()):
        if r.get("agree") is False:
            out.append(f"{animal} chronic: behaviour says {r.get('derived_day')}, "
                       f"spec says {r.get('stored_day')}")
    return out


def report_lines(rep) -> list[str]:
    """The full verdict, agreement included -- what gets logged every night.

    Agreement is printed, not just disagreement. A check that is silent when it passes is
    indistinguishable from a check that did not run, and this one went uncalled for ten days.
    """
    if not rep.get("available"):
        return [f"epoch audit: NOT CHECKED -- {rep.get('reason')}"]
    lines = [f"epoch audit ({rep['position']}, engaged trials, {rep['n_sessions']} sessions)"]
    lines.append(f"  {'animal':7s} {'acute':>10s}   {'chronic (derived / stored)':>28s}")
    for animal in sorted(rep["chronic"]):
        a = (rep["acute"] or {}).get(animal, {})
        c = rep["chronic"][animal]
        a_txt = {True: "agree", False: "DISAGREE", None: "no data"}.get(a.get("agree"), "?")
        mark = "" if c.get("agree") else "   <-- DISAGREE"
        lines.append(f"  {animal:7s} {a_txt:>10s}   "
                     f"{c.get('derived_day')!s:>12s} / {c.get('stored_day')!s:<12s}{mark}")
        for name in ("hit", "licks"):
            s = c.get(name) or {}
            if s.get("day") is not None:
                lines.append(f"            {name:6s} plateau day {s['day']:>3}  "
                             f"at {100 * (s.get('level') or 0):.0f}% of baseline")
            else:
                lines.append(f"            {name:6s} no plateau"
                             + (f" ({s['note']})" if s.get("note") else ""))
    bad = disagreements(rep)
    if bad:
        lines.append("  BEHAVIOUR HAS MOVED AWAY FROM THE STORED SPEC:")
        lines += [f"    {b}" for b in bad]
        lines.append("    ACUTE disagreements are reported only -- edit `epochs.EPOCH_SPEC` if "
                     "real. CHRONIC is derived and has already been applied to this run's "
                     "figures; `epoch_boundaries.json` records what was used.")
    else:
        lines.append("  stored spec agrees with behaviour on every animal and both boundaries")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--position", default=None, help=f"default {epochs.RULE_POSITION}")
    a = ap.parse_args(argv)
    rep = audit(position=a.position)
    for line in report_lines(rep):
        print(line)
    # EXIT 0 EVEN ON DISAGREEMENT. A disagreement is information, not a broken step: the nightly
    # wraps this in `cli()`, whose nonzero exits land in FAILURES, and a run with failed steps
    # REFUSES TO PUBLISH THE DECK. Blocking an entire night's deck because an animal's behaviour
    # moved -- which is the expected and interesting outcome -- would be exactly backwards.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
