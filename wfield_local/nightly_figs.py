"""Date-parametrized nightly analysis-figure generation for the spout-position summary deck.

  python -m wfield_local.nightly_figs [<DATE>...] [--from <DATE spec>] [--only <ANIMAL>...] [--output <dir>]
    e.g. python -m wfield_local.nightly_figs 0807
         python -m wfield_local.nightly_figs 0806 0807            # per-day figs for several dates
         python -m wfield_local.nightly_figs 0806-0808            # a range
         python -m wfield_local.nightly_figs all                  # every registered date
         python -m wfield_local.nightly_figs 0807 --from 0605-0608 --only PS93

Generates (into OUT): per-day decode (lick/cue/pre-cue 2 s, from configs/defaults.yaml decode.*_post_s), rolling cue + first-lick temporal
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
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local import config, writeguard
from wfield_local.console import use_utf8_stdout

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
def _default_out() -> str:
    """Local working figure dir for THIS machine.

    Was a hardcoded "C:/Users/sabatini/source/cue_lick" -- the analysis box's path. On the helper box
    (imaging mounts, analysis work) every figure step then failed with FileNotFoundError and the deck
    built with 0 figures and 287 missing, while the run still exited 0. Resolve it per machine, and
    fail loudly if the machine has no such root rather than writing somewhere that does not exist.
    """
    env = os.environ.get("WIDEFIELD_FIGURES_WORKING")
    if env:
        return env
    try:
        return config.resolver().root("figures_working")
    except Exception as exc:                      # noqa: BLE001
        raise SystemExit(
            f"no 'figures_working' root for this machine ({exc}). Set WIDEFIELD_FIGURES_WORKING or add "
            f"a mount for this machine in configs/paths.yaml.") from exc


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


FAILURES: list[str] = []
#: epoch seconds this run began -- the reference the deck uses to tell a figure it
#: refreshed from one left over from an earlier run.
RUN_START: float = time.time()


def cli(*a):
    log("CLI " + " ".join(a))
    if subprocess.call([PY, "-u", "-m", *a], cwd=str(REPO)):
        log("  !! nonzero exit: " + a[0])
        FAILURES.append(a[0])


def _write_run_record(deck_out, date, tag):
    """Leave this run's failed-step list on disk, beside the deck.

    WHY. On 2026-08-21 the deck refused to publish and the reason existed ONLY in the terminal
    scrollback of the window that launched the run. Reconstructing it afterwards took a full rebuild
    to a scratch path just to learn the missing count was 0, which meant the FAILED-STEP gate had
    fired -- and even then the step could not be named. A gate that refuses to publish has to say
    what it refused over, durably, or the refusal is a mystery instead of a diagnosis.

    Soft-fails: a run must never die because it could not write its own record.
    """
    import json as _json

    rec = {"date": date, "tag": tag,
           "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(RUN_START)),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "failed_steps": sorted(set(FAILURES)),
           "deck_published": not FAILURES}
    try:
        p = Path(deck_out).with_suffix(".run.json")
        writeguard.assert_writable(p)
        p.write_text(_json.dumps(rec, indent=1), encoding="utf-8")
        log(f"   run record: {p}")
    except Exception as ex:                                       # noqa: BLE001
        log(f"  !! could not write run record: {type(ex).__name__} {str(ex)[:80]}")


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


#: Where each analysis JSON lands under ``analysis_json/``, matched on the filename PREFIX in order.
#: Derived from the name alone so the routing is deterministic and a new artifact cannot silently
#: change where an old one lives; anything unmatched goes to ``other/`` rather than being dropped.
JSON_GROUPS = (
    ("nolick_reference", "references"),
    ("coding_direction", "coding_directions"),
    ("section_g", "section_g"),
    ("poststroke_", "section_g"),
    ("spatial_reorganisation", "spatial"),
    ("evoked_amplitude", "evoked"),
    ("joint_", "joint"),
    ("locanmf_", "locanmf"),
)

#: Files that are FROZEN by construction -- written once and deliberately never regenerated, because
#: recomputing them over a grown session set would produce a DIFFERENT reference and a reference that
#: moves after the comparison data arrive is not a reference. The publisher will create these but
#: will never overwrite them; a mismatch is reported, never resolved silently.
JSON_FROZEN = ("nolick_reference_prestroke",)


def json_group(name: str) -> str:
    """Subdirectory for one artifact, from its filename prefix."""
    for pref, grp in JSON_GROUPS:
        if name.startswith(pref):
            return grp
    return "other"


def _publish_json(out, rv, log=print) -> dict:
    """Copy the analysis JSON artifacts to MICROSCOPE beside the published PNGs.

    WHY THIS EXISTS (Priya, 2026-08-25). ``_publish_figs`` globs ``*.png``, so every JSON the analysis
    writes lived only on the box that produced it. Two of them are load-bearing:
    ``coding_direction.json`` is the source behind section G9 and the miss-while-working vs stopped
    result recorded in DECISIONS, and ``nolick_reference_prestroke.json`` is a reference the pipeline
    deliberately FREEZES. A frozen reference on one machine's local disk is the worst of both worlds --
    it cannot be restored if that disk dies (regenerating it today gives a different reference, which
    is precisely what the freeze prevents), and the second analysis box holds its own copy frozen at
    whatever date IT first ran, so the two can disagree with nothing to notice it.

    FROZEN FILES ARE NEVER OVERWRITTEN. For those, a destination that already exists is compared by
    CONTENT: identical is a silent skip, DIFFERENT is reported and left alone. Overwriting would let
    whichever box published second silently redefine the reference the other one used. Everything else
    follows the PNG rule (missing / different size / newer). Never deletes, per the MICROSCOPE
    no-auto-delete rule.
    """
    import hashlib
    import shutil

    from wfield_local import writeguard
    root = Path(rv.root("cue_analysis_out")) / "analysis_json"
    writeguard.assert_writable(root)
    res = {"copied": 0, "skipped": 0, "frozen_conflicts": []}
    for p in sorted(Path(out).glob("*.json")):
        d = root / json_group(p.name) / p.name
        frozen = any(p.name.startswith(f) for f in JSON_FROZEN)
        if d.exists() and frozen:
            same = (hashlib.sha256(p.read_bytes()).hexdigest()
                    == hashlib.sha256(d.read_bytes()).hexdigest())
            if not same:
                res["frozen_conflicts"].append(p.name)
                log(f"  !! FROZEN artifact differs from the published copy, NOT overwritten: "
                    f"{p.name} -- the two analysis boxes have diverged; compare before trusting "
                    f"any result that reads it")
            else:
                res["skipped"] += 1
            continue
        if d.exists() and not (p.stat().st_size != d.stat().st_size
                               or p.stat().st_mtime > d.stat().st_mtime + 2):
            res["skipped"] += 1
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d)
        res["copied"] += 1
    return res


def main():
    # A subprocess inherits this, so every `cli(...)` step below is covered too. See
    # wfield_local/console.py: a cp1252 stdout turns a printed U+2212 into a lost figure.
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dates", nargs="*", metavar="DATE",
                    help="per-day figure date(s): MMDD or YYYYMMDD; a range (0806-0808), 'all', or a "
                         "comma/space list (default: the latest registered session)")
    ap.add_argument("--from", dest="from_dates", default=None,
                    help="cross-session comparison dates — same grammar (default: the curated set)")
    ap.add_argument("--output", default=None, help="figure output dir (also where the deck is built)")
    ap.add_argument("--skip-nolick", action="store_true",
                    help="skip the no-detected-lick reference (engaged vs late vs undetected, "
                         "pre-cue vs post-cue). Adds ~1-2 h; see wfield_local.nolick_analysis.")
    ap.add_argument("--skip-poststroke", action="store_true",
                    help="skip the post-stroke stage (section G + the map-level analyses)")
    ap.add_argument("--skip-frozen", action="store_true",
                    help="skip the frozen cross-day decoder/encoder step (Allen-ROI, leave-one-session-out). "
                         "It adds ~30-40 min; skipping leaves those deck slides blank.")
    ap.add_argument("--only", nargs="+", metavar="ANIMAL",
                    help="restrict analysis to these animals (e.g. PS93), or 'all'; scopes the decode/"
                         "encode/cross-mouse/RSA subprocesses via WIDEFIELD_ONLY_ANIMALS + the in-process figs")
    args = ap.parse_args()

    out = args.output or _default_out()
    Path(out).mkdir(parents=True, exist_ok=True)
    log(f"figure working dir: {out}")
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

    # Pre-cue MOTOR control: decode/encode on trials with no licks in the window. Runs over the whole
    # --from set (exposure varies BY SESSION, not just by animal -- PS93 drops to 76% lick-free on
    # 8/9 while sitting at 98% in June), and in BOTH bases because the regional story should not
    # depend on the basis. Deck section B2.
    for src_name in ("roi", "locanmf"):
        cli("wfield_local.precue_lickfree", "--output", out, "--from", from_dates,
            "--source", src_name)

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

    # FROZEN cross-day decoder + encoder (Allen-ROI, leave-one-session-out), over the same curated set.
    # MUST be written into `out` (the deck's source dir), not straight to MICROSCOPE: the deck is built
    # from `out`, so a figure that exists only on the server is invisible to it and its slides come out
    # blank. That is how 40 slides silently emptied on 2026-08-12.
    if not args.skip_frozen:
        try:
            from wfield_local.locanmf_frozen_decoder import (
                _encoder_fig,
                _loso_fig,
                pooled_frozen_encoder,
                pooled_frozen_loso,
                write_session_confusions,
            )
            # ALL THREE alignments. post-cue is the readout during/after the movement; PRE-CUE is the
            # pre-cue POSITION INFORMATION (the 2 s window ENDING at the cue), which is the
            # one the stroke arm actually leans on -- so whether IT survives being frozen across
            # days matters more than whether the post-cue one does. LICK-ALIGNED added 2026-08-26
            # (Priya: "we should have a frozen lick-aligned decoder and encoder"): it existed only as
            # a by-hand run whose output nothing regenerated and nothing read, so it sat two days
            # stale on an 11-session pre-stroke-only pool while cue and precue moved on. An arm worth
            # having is an arm the nightly maintains; otherwise it silently becomes an orphan.
            for al in ("cue", "precue", "lick"):
                dec, enc = {}, {}
                for a in sorted({s["label"][:4] for s in SESSIONS if s["label"][-4:] in set(from_list)}):
                    if only and a not in set(only):
                        continue
                    labs = [s["label"] for s in SESSIONS
                            if s["label"].startswith(a) and s["label"][-4:] in set(from_list)]
                    if len(labs) < 2:
                        continue
                    r = pooled_frozen_loso(labs, source="roi", align=al, verbose=False)
                    if r:
                        dec[a] = r
                    e = pooled_frozen_encoder(labs, source="roi", align=al, verbose=False)
                    if e:
                        enc[a] = e
                if dec:
                    n = len(write_session_confusions(dec, Path(out)))
                    _loso_fig(dec, Path(out), al)
                    # PERSIST the numbers, not just the pictures. Only the standalone
                    # `locanmf_frozen_decoder --loso` CLI used to write these, so the nightly left
                    # whatever JSON an old manual run had produced sitting on disk -- on 2026-08-13
                    # that was an 8-session file next to 9-session figures, and reading it as current
                    # gave a per-session comparison that silently omitted 8/12. joint_xsession always
                    # wrote its JSON; ROI now does too, so the two bases are equally auditable.
                    (Path(out) / f"locanmf_frozen_decoder_loso_roi_{al}.json").write_text(
                        json.dumps(dec, indent=2, default=float))
                    log(f"frozen decoder [{al}]: {len(dec)} animal(s), {n} confusion figure(s)  "
                        + "  ".join(f"{k}={v['loso_accuracy']:.3f}" for k, v in sorted(dec.items())))
                if enc:
                    _encoder_fig(enc, Path(out), al)
                    (Path(out) / f"locanmf_frozen_encoder_loso_roi_{al}.json").write_text(
                        json.dumps(enc, indent=2, default=float))
                    log(f"frozen encoder [{al}]: {len(enc)} animal(s)  "
                        + "  ".join(f"{k}={v['mean_ev']:+.3f}" for k, v in sorted(enc.items())))
        except Exception as ex:
            log(f"  !! frozen decoder/encoder: {type(ex).__name__} {str(ex)[:80]}")

    # CROSS-SESSION decode/encode in the shared JOINT-LocaNMF basis -- the second, independent basis
    # for the same cross-day question the frozen ROI step above answers. Runs only if a joint basis
    # has been built for the animal (wfield_local.joint_locanmf); a missing basis is reported, never
    # silently refitted, because a refit over a grown session set is a DIFFERENT reference frame.
    if not args.skip_frozen:
        # Same three alignments as the ROI arm above, so the two bases always answer the same
        # question and can be compared. Until 2026-08-26 this passed only cue+precue while a hand-run
        # produced the lick figures, which is how the joint lick arm ended up on a different session
        # pool from its own siblings.
        cli("wfield_local.joint_xsession", "--output", out, "--from", from_dates,
            "--align", "cue", "precue", "lick")

    # NO-DETECTED-LICK arm. The pre-stroke reference for reading post-stroke failed trials: does the
    # PRE-cue position code survive on trials the animal did not lick, when the POST-cue one does
    # not? Rebuilt each night so post-stroke sessions join it, but the PRE-stroke reference itself is
    # written once to `nolick_reference_prestroke.json` and never overwritten -- a reference that
    # moves after the comparison data arrive is not a reference.
    if not args.skip_nolick:
        try:
            from wfield_local import nolick_decoder, plot_nolick_reference
            ref_path = Path(out) / "nolick_reference.json"
            ref = nolick_decoder.build_reference(dates=from_list, out=ref_path)
            plot_nolick_reference.figures(ref, out)
            frozen = Path(out) / "nolick_reference_prestroke.json"
            if not frozen.exists():
                # FREEZE WHAT IT SAYS ON THE TIN, not whatever happened to be in the live file.
                # This used to copy `nolick_reference.json` unconditionally. That file is built from
                # `from_list`, which is ALL phases -- today 19 dates including 0817-0824 -- so the
                # only reason the artifact frozen on 2026-08-19 is genuinely pre-stroke is that the
                # date list happened to be pre-only that night. Re-freezing today, or freezing for
                # the first time on the other analysis box, would mint a "pre-stroke reference"
                # containing post-stroke sessions and nothing would object: `exists()` cannot see
                # what is inside the file. (Raised by the second window 2026-08-26; the existing
                # artifact was checked and IS clean -- 11 pre-stroke dates, 44 pre-stroke labels,
                # zero post -- but that was luck, not a guarantee.)
                _post = sorted(set(from_list) & set(config.poststroke_dates()))
                if _post:
                    log(f"  !! NOT freezing the pre-stroke no-lick reference: the live reference "
                        f"covers post-stroke dates {_post}. Freezing it would preserve the "
                        f"contamination rather than prevent it. Rebuild with pre-stroke dates and "
                        f"freeze that, deliberately.")
                else:
                    frozen.write_text(ref_path.read_text())
                    log(f"froze the PRE-STROKE no-lick reference ({len(from_list)} pre-stroke "
                        f"dates) -> {frozen.name}")
            for an, v in (ref.get("consensus") or {}).items():
                log(f"  no-lick {an}: {v if isinstance(v, str) else 'BASES DISAGREE -- see deck section D2'}")
        except Exception as ex:
            log(f"  !! no-lick reference: {type(ex).__name__} {str(ex)[:80]}")

    # ---------------- POST-STROKE (deck section G) ----------------
    # Runs only once a post-stroke session exists, so a pre-stroke-only cohort is unaffected. Wired
    # in here rather than left to be typed by hand: until 2026-08-19 every one of these was invoked
    # from a scratchpad script, which is how twelve deck figures came to be a day stale on a basis
    # that had been corrected, with nothing on the slide to say so. If it is part of the deck it is
    # part of the nightly.
    if not args.skip_poststroke and config.phase_labels("post"):
        log("== POST-STROKE stage (section G)")
        cli("wfield_local.poststroke_section_g", "--output", out)
        cli("wfield_local.section_g_figures", "--src", out, "--output", out)
        # the map-level analyses behind sections G8d-G8f
        cli("wfield_local.fixed_scale_maps", "--output", out)
        cli("wfield_local.evoked_amplitude", "--output", out, "--align", "cue", "lick")
        for _arm in ([], ["--lick-only"]):
            cli("wfield_local.spatial_reorganisation", "--output", out,
                "--align", "cue", "precue", *_arm)
        cli("wfield_local.vessel_contrast", "--output", out)
        cli("wfield_local.hemispheric_dynamics", "--output", out)
        cli("wfield_local.hemispheric_intensity", "--output", out)
        # Per-position coding directions (deck section G9). Wired in BEFORE the deck slide
        # was added, deliberately: a slide reading a figure no nightly step regenerates is
        # frozen at the day it was made, which is what section G spent 2026-08-20 undoing.
        # ~30-40 min: pool_sessions rebuilds trial features per (animal, window).
        cli("wfield_local.position_coding_directions", "--output", out)
        # reads coding_direction.json, so it must follow the line above. Seconds, not
        # minutes: nothing is recomputed, the values are already stored per session.
        cli("wfield_local.miss_vs_stopped", "--output", out)

    # GRANT FIGURES -- deck section H places 19 of these patterns, so by the rule this file already
    # follows for the post-stroke stage ("if it is part of the deck it is part of the nightly") they
    # cannot be left to a hand-run command. Until 2026-08-27 they were: `nightly_figs` never invoked
    # `grant_figures`, so every deck built section H from whatever happened to be on disk, and a
    # night nobody re-rendered by hand shipped a silently day-stale H. Exactly the failure the
    # post-stroke stage was wired in to end.
    #
    # BEFORE THE DECK BUILD, because the deck reads the PNGs this writes. `--compact` keeps the
    # digit-free variants current for the grant document; they are deliberately NOT placed as
    # slides (see the section-H filter in locanmf_analysis_deck).
    #
    # Its own root: these land under `labcams/grant_figures`, not `out` -- a deliverable rather than
    # an analysis intermediate -- so no --output is passed.
    cli("wfield_local.grant_figures", "--compact")

    # build the refined ANALYSIS deck (animal -> type -> date, curated) at the labcams top level
    # Bound OUTSIDE the try: the run record below needs it even when the deck step dies early,
    # and that is exactly the run whose failure list is worth having on disk.
    deck_out = Path(config.resolver().root("labcams")) / "spout_position_analysis_summary.pptx"
    try:
        from wfield_local.locanmf_analysis_deck import (
            DeckFromFailedRun,
            DeckIncomplete,
            build_analysis_deck,
        )
        # FAILURES and RUN_START are what let the deck judge its own inputs: a failed step means
        # stale panels (its outputs are last run's), and RUN_START separates figures this run
        # refreshed from ones it did not touch. Neither is visible from the figure tree alone.
        d = build_analysis_deck(Path(out), deck_out, dates=from_list, tag=tag,
                                failed_steps=sorted(set(FAILURES)), run_start=RUN_START)
        log(f"== analysis deck: {d['out']} ({d['slides']} slides, {d['figures_present']} figs, "
            f"{d['figures_missing']} missing) ==")
        _stale = d.get("stale_detail") or []
        if _stale:
            # NOT a failure: most of the figure tree is one-off analyses that do not regenerate.
            # Reported so an ORPHANED reference -- a slide reading a filename no step writes any
            # more -- is visible, which no other check can see because the file is present.
            log(f"   {len(_stale)} placed figure(s) NOT refreshed by this run "
                f"(manifest: {d.get('manifest')}):")
            for r in sorted(_stale, key=lambda r: -r["age_days"])[:15]:
                log(f"       {r['age_days']:6.2f}d  {r['figure']}")
            if len(_stale) > 15:
                log(f"       ... and {len(_stale) - 15} more (see the manifest)")
    except DeckFromFailedRun as ex:
        FAILURES.append("analysis deck (run had failed steps -- NOT published)")
        log(f"  !! analysis deck NOT PUBLISHED: {len(ex.failed_steps)} step(s) failed in this run, "
            f"so some panels would be left over from an earlier run; existing deck left untouched")
        for f_ in sorted(set(ex.failed_steps)):
            log(f"       failed step: {f_}")
        log("     fix them and rerun, or rebuild with allow_failed_steps=True to publish anyway")
    except DeckIncomplete as ex:
        # NOT published: the previous deck is still in place. Name every gap -- the generic handler
        # below truncates to 80 chars, which is exactly the detail needed to fix the upstream step.
        FAILURES.append("analysis deck (incomplete -- NOT published)")
        log(f"  !! analysis deck NOT PUBLISHED: {len(ex.missing_figures)} figure(s) missing, "
            f"existing deck left untouched")
        for m in ex.missing_figures:
            log(f"       missing: {m}")
        log("     fix the step that owns them, or rerun the deck with "
            f"--allow-missing {len(ex.missing_figures)} to publish anyway")
    except Exception as ex:
        FAILURES.append("analysis deck")
        log(f"  !! analysis deck: {type(ex).__name__} {str(ex)[:80]}")

    # Publish the component PNGs to MICROSCOPE so the individual analysis figures persist on the server
    # next to the deck (durable + accessible), not only embedded in the .pptx / on this box.
    try:
        n = _publish_figs(out, config.resolver())
        log(f"published {n} analysis PNG(s) to MICROSCOPE cue_analysis")
    except Exception as ex:
        log(f"  !! publish figs: {type(ex).__name__} {str(ex)[:80]}")
    # The JSON half: the numbers BEHIND the figures, including the frozen pre-stroke reference.
    # Separate try/except so a JSON failure cannot cost the PNG publish, or vice versa.
    try:
        j = _publish_json(out, config.resolver(), log=log)
        log(f"published {j['copied']} analysis JSON(s) to MICROSCOPE analysis_json "
            f"({j['skipped']} already current)")
        if j["frozen_conflicts"]:
            FAILURES.append(f"frozen artifact divergence: {', '.join(j['frozen_conflicts'])}")
    except Exception as ex:
        log(f"  !! publish json: {type(ex).__name__} {str(ex)[:80]}")

    log(f"== nightly figures complete: per-day {per_day}, cross-session tag {tag} ==")
    _write_run_record(deck_out, date, tag)
    # A run whose figure steps all failed used to exit 0 and leave a deck with 0 figures and 287
    # missing -- indistinguishable from success to any caller or cron job. Report the truth.
    if FAILURES:
        log(f"== {len(FAILURES)} STEP(S) FAILED: {sorted(set(FAILURES))} ==")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
