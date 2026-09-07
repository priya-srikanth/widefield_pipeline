"""Evaluate the epoch definitions against behaviour, every run. Derives chronic; checks acute.

Priya, 2026-09-07: *"I'd like the pipeline to run the epoch definitions and just determine if we
have met 'chronic' criteria, order sessions into epochs appropriately, and analyze."*

ALL THREE BOUNDARIES ARE DERIVED as of 2026-09-07. Priya: *"we have clear derivation definitions
for acute and subacute right? like, we don't have to hard-code them?"* -- correct. The acute rule
reproduces all four declared ranges exactly, and subacute has no rule of its own: it is the
complement, with `subacute_from` falling out as the first session day after the acute prefix ends.

ACUTE IS THE SAFEST OF THE THREE TO DERIVE, which is the opposite of what was assumed at first. It
depends only on early post-stroke sessions and the pre-stroke baseline, neither of which changes as
the cohort grows -- recording day 21 cannot alter day 1-5 accuracy. Chronic is the volatile one: it
re-evaluates at the END of the series, exactly where every new session lands.

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

from wfield_local import config, epochs
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
            "acute_derived": epochs.derive_acute_boundaries(hit, position=position),
            "chronic": epochs.derive_chronic_boundaries(hit, lick, position=position)}


def _same(a, b) -> bool:
    """Equality that does not care whether a range arrived as a list or a tuple.

    `acute` is declared in YAML as a list, normalised to a tuple by `config.epoch_spec`, and
    round-trips through JSON as a list again. Comparing them raw reports `(1, 5) -> [1, 5]` as a
    change every single night -- noise that teaches the reader to skip the block that matters.
    """
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) == list(b)
    return a == b


def derived_spec(rep) -> dict:
    """``{animal: {"acute", "subacute_from", "chronic_from"}}`` -- what the rules decided.

    ALL THREE boundaries are derived as of 2026-09-07. Priya: *"we have clear derivation
    definitions for acute and subacute right? like, we don't have to hard-code them?"* -- correct,
    and the acute rule reproduces all four declared ranges exactly.

    A boundary the rules could not determine is OMITTED rather than written as None, so
    `epochs.spec_for` falls back to the declared value for that key. Writing None would assert
    "there is no acute epoch", which is a different and much stronger claim than "behaviour could
    not tell me" -- and for acute it would unassign every early post-stroke session.

    `chronic_from` is the exception: None there IS the assertion "tested, has not stabilised", which
    is why it is always written.
    """
    acute = rep.get("acute_derived") or {}
    out = {}
    for animal, r in sorted((rep.get("chronic") or {}).items()):
        spec = {"chronic_from": r.get("derived_day")}
        a = acute.get(animal) or {}
        if a.get("acute") is not None:
            spec["acute"] = list(a["acute"])            # JSON has no tuple; spec_for restores it
        if a.get("subacute_from") is not None:
            spec["subacute_from"] = a["subacute_from"]
        out[animal] = spec
    return out


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
        for key in ("acute", "subacute_from", "chronic_from"):
            a = old_spec.get(animal, {}).get(key, "absent")
            b = new_spec.get(animal, {}).get(key, "absent")
            if not _same(a, b):
                out.append(f"{animal} {key}: {a} -> {b}")
    return out


def stale_fallback(spec) -> list[str]:
    """Animals where the DERIVED boundary differs from the declared fallback in `animals.yaml`.

    A DIFFERENT COMPARISON FROM `changes`, and both are needed. `changes` asks "did this move since
    last night" -- transient news. This asks "is the committed fallback still right", a standing
    condition that persists until a human acts.

    IT IS NOT URGENT, and saying so matters as much as reporting it. A crashed or behaviour-less run
    falls back to the LAST DERIVED `epoch_boundaries.json`, not to this, so a stale fallback does
    not silently revert tonight's epochs. It bites in exactly two places: a fresh clone or a box
    with no artifact yet, and `WIDEFIELD_EPOCHS_PINNED=1`. Both are about REPRODUCING a figure set
    rather than producing one, which is why promotion is a deliberate human step and not a nightly
    write.
    """
    out = []
    for animal, derived in sorted(spec.items()):
        have_all = config.epoch_spec(animal) or {}
        for key in ("acute", "subacute_from", "chronic_from"):
            if key not in derived:
                continue
            want, have = derived[key], have_all.get(key)
            if not _same(want, have):
                out.append(f"{animal}: animals.yaml says {key} {have}, behaviour says {want}")
    return out


def promotion_yaml(spec) -> list[str]:
    """The exact `animals.yaml` lines to paste, for the animals whose fallback is stale.

    Printed rather than written. The pipeline must not edit `animals.yaml` -- it is
    version-controlled, both machines push `main`, and most of its value is hand-written comments a
    YAML dump would delete. But a human promotion that requires re-deriving the number by hand is a
    human promotion that does not happen, so the diff is produced ready to paste.
    """
    # YAML SPELLING THROUGHOUT, comments included. The whole block is meant to be pasted into
    # animals.yaml, and a "was None" beside a "chronic_from: null" invites writing Python's spelling
    # into a YAML file, where it is the STRING "None" and parses truthy rather than as an absent
    # boundary. Lists render as YAML flow sequences for the same reason.
    def _y(v):
        if v is None:
            return "null"
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(str(x) for x in v) + "]"
        return str(v)

    lines = []
    for animal, derived in sorted(spec.items()):
        have_all = config.epoch_spec(animal) or {}
        rows = []
        for key in ("acute", "subacute_from", "chronic_from"):
            if key not in derived:
                continue
            want, have = derived[key], have_all.get(key)
            if _same(want, have):
                continue
            note = ("has not stabilised" if (key == "chronic_from" and want is None)
                    else f"derived from behaviour, was {_y(have)}")
            rows.append(f"      {key}: {_y(want)}    # {note}")
        if rows:
            lines += [f"  {animal}:", "    epochs:", *rows]
    return lines


def resolve(rv=None, position=None, write=True):
    """Derive the boundaries, install them for THIS process, and publish them for the subprocesses.

    Returns ``{"report", "spec", "changes", "path", "available"}``.

    THE FILE IS THE MECHANISM, not a log. `nightly_figs` runs `grant_figures` and
    `epoch_grant_figures` as subprocesses, which re-import `epochs` fresh and pick the boundaries up
    from disk; installing them only in the parent would build half the deck on derived boundaries
    and half on stored ones, and it would render cleanly.

    ON UNAVAILABLE BEHAVIOUR IT WRITES NOTHING AND INSTALLS NOTHING, which leaves the LAST DERIVED
    `epoch_boundaries.json` in force (`epochs` loads it lazily). That is the safe direction: a
    missing cohort table must not silently move every animal to "no chronic epoch" and restage the
    whole deck, and it must not quietly revert to a hand-declared fallback that may be much older
    than the last good derivation.
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
            "first_run": old is None, "stale_fallback": stale_fallback(spec),
            "promote_yaml": promotion_yaml(spec)}


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
    lines = [f"epoch audit ({rep['position']}, engaged trials, {rep['n_sessions']} sessions)",
             "  all three boundaries DERIVED from behaviour; animals.yaml is the seed/pinned ref"]
    lines.append(f"  {'animal':7s} {'acute':>10s} {'subacute':>9s} {'chronic':>8s}   "
                 f"{'(derived, vs animals.yaml)':<28s}")
    ad = rep.get("acute_derived") or {}
    for animal in sorted(rep["chronic"]):
        c = rep["chronic"][animal]
        a = ad.get(animal) or {}
        decl = config.epoch_spec(animal) or {}
        acute = a.get("acute")
        same = (list(acute or []) == list(decl.get("acute") or [])
                and a.get("subacute_from") == decl.get("subacute_from")
                and c.get("derived_day") == decl.get("chronic_from"))
        lines.append(f"  {animal:7s} {acute!s:>10s} {a.get('subacute_from')!s:>9s} "
                     f"{c.get('derived_day')!s:>8s}   "
                     + ("matches animals.yaml" if same else "<-- DIFFERS from animals.yaml"))
        for name in ("hit", "licks"):
            s = c.get(name) or {}
            if s.get("day") is not None:
                lines.append(f"            {name:6s} plateau day {s['day']:>3}  "
                             f"at {100 * (s.get('level') or 0):.0f}% of baseline")
            else:
                lines.append(f"            {name:6s} no plateau"
                             + (f" ({s['note']})" if s.get("note") else ""))
    relapsed = [(a, r["relapse"]) for a, r in sorted((rep.get("acute_derived") or {}).items())
                if r.get("relapse")]
    if relapsed:
        # NOT folded into the acute range. `(lo, hi)` cannot express "acute, recovered, acute
        # again", so absorbing a later dip would silently relabel the recovered sessions between.
        lines.append("  RELAPSE -- post-stroke session(s) back below the acute threshold AFTER "
                     "recovering. Not folded into the acute range; reported so it is a finding:")
        for animal, days in relapsed:
            lines.append(f"    {animal}: day(s) {', '.join(str(d) for d in days)}")
    bad = disagreements(rep)
    if bad:
        lines.append("  BEHAVIOUR HAS MOVED AWAY FROM THE STORED SPEC:")
        lines += [f"    {b}" for b in bad]
        lines.append("    ALL THREE boundaries are DERIVED and have already been applied to this "
                     "run's figures; `epoch_boundaries.json` records what was used. animals.yaml "
                     "is the seed and the pinned reference -- promote when you want a fresh clone "
                     "or a pinned rebuild to reproduce these epochs.")
    else:
        lines.append("  animals.yaml agrees with behaviour on every animal and all three "
                     "boundaries")
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
