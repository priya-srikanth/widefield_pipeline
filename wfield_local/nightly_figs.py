"""Date-parametrized nightly analysis-figure generation for the spout-position summary deck.

  python -m wfield_local.nightly_figs [<DATE>...] [--from <DATE spec>] [--only <ANIMAL>...] [--output <dir>]
    e.g. python -m wfield_local.nightly_figs 0807
         python -m wfield_local.nightly_figs 0806 0807            # per-day figs for several dates
         python -m wfield_local.nightly_figs 0806-0808            # a range
         python -m wfield_local.nightly_figs all                  # every registered date
         python -m wfield_local.nightly_figs 0807 --from 0605-0608 --only PS93

Generates (into OUT): per-day decode (lick/cue 2 s, pre-cue 1 s), rolling cue + first-lick temporal
dynamics + laterality, top components, encoder (+ FEVE raw & normalized), and the cross-mouse /
within-animal consistency / RSA (incl. crossnobis) comparison. The per-day figures are built for each
DATE (default: the latest registered session); the cross-SESSION comparisons span the CURATED set by
default (override with --from), tag `<first>-<last>`.

The date grammar (MMDD or YYYYMMDD; list, range `0806-0808`, or `all`; comma/space tolerant) and
`--only` (animals, or `all`) are shared verbatim with the preprocessing CLI
(:mod:`wfield_local.preprocess`); see :func:`wfield_local.config.expand_dates`.

Canonical orchestrator for the nightly pipeline (see runbooks/analysis_computer_nightly.md). Runs the individual
locanmf_* modules as subprocesses so each gets a clean process; the repo root is derived from this file's
location, so it is portable across checkouts.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local import config

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
DEFAULT_OUT = "C:/Users/sabatini/source/cue_lick"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cli(*a):
    log("CLI " + " ".join(a))
    if subprocess.call([PY, "-u", "-m", *a], cwd=str(REPO)):
        log("  !! nonzero exit: " + a[0])


def _per_day_figs(date, out, from_dates, only):
    """Per-day decode (lick/cue/pre-cue), in-process dynamics/laterality/components, and the encoder."""
    dp = config.defaults()["decode"]   # aligns + per-align windows (configs/defaults.yaml decode.*_post_s)
    for al in dp["aligns"]:
        cli("wfield_local.locanmf_position_decoder", "--date", date, "--align", al,
            "--post-s", str(dp[f"{al}_post_s"]), "--per-session", "--output", out)

    from wfield_local.locanmf_decoder_weights import (_avail, fig_rolling_cue, fig_temporal_dynamics,
                                                      fig_rolling_laterality, fig_top_components)
    labs = _avail(date)
    if only:   # _avail reads the import-time SESSIONS, so filter the in-process figs explicitly
        labs = [l for l in labs if l[:4] in set(only)]
    log(f"{date} sessions: {labs}")
    for desc, fn in (("rolling_cue", lambda: fig_rolling_cue(labs, Path(out), date)),
                     ("temporal_dynamics", lambda: fig_temporal_dynamics(labs, Path(out))),
                     ("rolling_laterality", lambda: fig_rolling_laterality(labs, Path(out), date))):
        try:
            log("wrote " + fn().name)
        except Exception as ex:
            log(f"  !! {desc}: {type(ex).__name__} {str(ex)[:70]}")
    for lab in labs:
        try:
            log("wrote " + fig_top_components(lab, Path(out)).name)
        except Exception as ex:
            log(f"  !! top_components {lab}: {str(ex)[:60]}")

    cli("wfield_local.locanmf_position_encoder", "--date", date, "--pool-dates", from_dates, "--output", out)


def _perday_figs_incomplete(out, d) -> bool:
    """True if date ``d`` has registered sessions but is MISSING the per-day cue-decode figure for any of
    them in ``out`` (Section A's anchor, ``locanmf_position_session_<label>_locanmf_cue_base-none_cv-block.png``).
    The deck spans the whole curated set, so a date whose figs failed on its own night stays blank forever;
    nightly_figs backfills any such date so the deck self-heals."""
    labs = [s["label"] for s in config.load_sessions(dates=[d])]
    return bool(labs) and not all(
        (Path(out) / f"locanmf_position_session_{lab}_locanmf_cue_base-none_cv-block.png").exists()
        for lab in labs)


def _publish_figs(out, rv) -> int:
    """Copy the analysis component PNGs to MICROSCOPE (``cue_analysis_out``) so the individual figures
    persist on the server next to the deck -- durable + accessible, not just embedded in the .pptx and
    living only on this box (matches how the preprocessing figures persist under labcams/<date>/).
    Incremental: copies a PNG only when the destination is missing, a different size, or newer. Never
    deletes (mirror cleanup stays a manual step, per the MICROSCOPE no-auto-delete rule)."""
    import shutil
    from wfield_local import writeguard
    dst = Path(rv.root("cue_analysis_out"))
    writeguard.assert_writable(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(out).glob("*.png")):
        d = dst / p.name
        if (not d.exists()) or p.stat().st_size != d.stat().st_size or p.stat().st_mtime > d.stat().st_mtime + 2:
            shutil.copy2(p, d)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dates", nargs="*", metavar="DATE",
                    help="per-day figure date(s): MMDD or YYYYMMDD; a range (0806-0808), 'all', or a "
                         "comma/space list (default: the latest registered session)")
    ap.add_argument("--from", dest="from_dates", default=None,
                    help="cross-session comparison dates — same grammar (default: the curated set)")
    ap.add_argument("--output", default=DEFAULT_OUT, help="figure output dir (also where the deck is built)")
    ap.add_argument("--only", nargs="+", metavar="ANIMAL",
                    help="restrict analysis to these animals (e.g. PS93), or 'all'; scopes the decode/"
                         "encode/cross-mouse/RSA subprocesses via WIDEFIELD_ONLY_ANIMALS + the in-process figs")
    args = ap.parse_args()

    out = args.output
    registered = sorted({s["label"][-4:] for s in SESSIONS})
    only = config.normalize_animals(args.only)   # None => all animals
    if only:
        os.environ["WIDEFIELD_ONLY_ANIMALS"] = ",".join(only)  # inherited by the subprocesses below
        log(f"animal subset: {only}")

    try:
        per_day = config.expand_dates(args.dates, width=4, available=registered) if args.dates else [registered[-1]]
    except ValueError as e:
        ap.error(str(e))

    # Cross-session comparisons use the CURATED "good" set by default: 6/6-6/8 + 8/6 onward (auto-includes
    # future dates), excluding noisy early June (6/1-6/5) and the wonky 8/5. Policy in configs/animals.yaml.
    exclude = set(config.date_policy().get("cross_session_exclude", []))
    curated = [d for d in registered if d not in exclude]
    try:
        from_list = config.expand_dates(args.from_dates, width=4, available=registered) if args.from_dates else curated
    except ValueError as e:
        ap.error(str(e))
    from_dates = ",".join(from_list)
    tag = f"{from_list[0]}-{from_list[-1]}"

    # Backfill: also (re)generate per-day figures for any CURATED date whose figs are MISSING on disk, so
    # the deck (which spans the whole curated set) never keeps blank date columns from a past night whose
    # figs failed (e.g. a transient missing-input during incremental LocaNMF processing). Self-healing.
    backfill = [d for d in from_list if d not in per_day and _perday_figs_incomplete(out, d)]
    if backfill:
        log(f"backfilling per-day figs for curated dates missing them on disk: {backfill}")
    per_day = sorted(set(per_day) | set(backfill))

    log(f"per-day dates={per_day} cross-session dates={from_dates} tag={tag} out={out}")
    for date in per_day:
        _per_day_figs(date, out, from_dates, only)

    # cross-session comparisons: once, spanning the whole --from set (not per per-day date)
    cli("wfield_local.locanmf_cross_mouse", "--output", out, "--dates", from_dates, "--tag", tag)
    cli("wfield_local.locanmf_rsa", "--output", out, "--dates", from_dates, "--tag", tag)

    # per-animal rolling decoder across the curated sessions (Section A of the analysis deck) — NOT emitted
    # by the per-day pass, so wire it here over the whole --from set (this is the figure that went stale).
    try:
        from wfield_local.locanmf_decoder_weights import _avail, fig_rolling_cue_by_animal
        by_animal = {}
        for d in from_list:
            for lab in _avail(d):
                by_animal.setdefault(lab[:4], []).append(lab)
        for a, labs in sorted(by_animal.items()):
            log("wrote " + fig_rolling_cue_by_animal(a, sorted(labs, key=lambda x: x[-4:]), Path(out)).name)
    except Exception as ex:
        log(f"  !! rolling_by_animal: {type(ex).__name__} {str(ex)[:70]}")

    # build the refined ANALYSIS deck (animal -> type -> date, curated) at the labcams top level
    try:
        from wfield_local.locanmf_analysis_deck import build_analysis_deck
        deck_out = Path(config.resolver().root("labcams")) / "spout_position_analysis_summary.pptx"
        d = build_analysis_deck(Path(out), deck_out, dates=from_list, tag=tag)
        log(f"== analysis deck: {d['out']} ({d['slides']} slides, {d['figures_present']} figs, "
            f"{d['figures_missing']} missing) ==")
    except Exception as ex:
        log(f"  !! analysis deck: {type(ex).__name__} {str(ex)[:80]}")

    # Publish the component PNGs to MICROSCOPE so the individual analysis figures persist on the server
    # next to the deck (durable + accessible), not only embedded in the .pptx / on this box.
    try:
        n = _publish_figs(out, config.resolver())
        log(f"published {n} analysis PNG(s) to MICROSCOPE cue_analysis")
    except Exception as ex:
        log(f"  !! publish figs: {type(ex).__name__} {str(ex)[:80]}")

    log(f"== nightly figures complete: per-day {per_day}, cross-session tag {tag} ==")


if __name__ == "__main__":
    main()
