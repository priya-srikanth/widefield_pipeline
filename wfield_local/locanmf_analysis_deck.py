"""Refined ANALYSIS deck — spout-position decode/encode/RSA, grouped ANIMAL -> analysis-type -> date.

A focused successor to the date-first summary deck and the 6/5-8 xsession deck: it includes ONLY the
curated (filtered) sessions (``configs/animals.yaml date_policy.cross_session``) and only the analyses
Priya wants, and it is written to the ``labcams`` TOP LEVEL (not two dirs deep) as
``spout_position_analysis_summary.pptx``.

Sections (per animal, then type, then date):
  A  WITHIN-DAY decoding — post-cue 2 s (no-lick generalization) + pre-cue 2 s confusion + recall per
     date, then the rolling decoder (pre-cue ENL -> post-cue) across sessions.
  B  WITHIN-DAY encoder — expected SSp/MO activity by position, encoder predicted maps, explained
     variance per position (raw + ceiling-relative) across sessions, and r2 per Allen region.
  C  Pre-cue code without licking — the motor-confound control on the study's key readout.
  D  CROSS-SESSION (frozen) decoders and encoders, in TWO INDEPENDENT BASES — Allen-ROI and the shared
     joint-LocaNMF basis. Each day is predicted by a model trained only on that animal's OTHER days.
     Two bases because a cross-day claim that holds in only one parcellation is a claim about the
     parcellation; one that holds in both is a claim about the cortex.
  E  Cross-session summary — decoder recall + encoder accuracy across sessions; within-animal consistency.
  F  RSA — within- vs across-animal geometry, per-animal RDM, crossnobis RDM.

RESTRUCTURED 2026-08-13: the frozen cross-day slides used to be buried inside each animal's Section A,
which mixed two different questions (does the code exist today? does it transfer across days?) under one
heading and put ~6 cross-day slides between an animal's within-day decode and its encoder. They are now
Section D, in both bases side by side.

Dropped vs the old decks: laterality decoder, top-10 component maps, hemisphere-resolved RDMs, and the
per-session (rather than per-animal) encoder variance panels. Also NOT here: the frozen fixed-A /
refit-C basis, REJECTED because its score depends on which session is nominated as the reference and no
reference wins for every animal (within-animal swing up to 0.36) — the joint basis in Section D is the
reference-free version of that same idea. See DECISIONS.md. Missing PNGs are skipped, so the deck builds
from whatever figures are present.

    python -m wfield_local.locanmf_analysis_deck                 # src = figures_working, out = labcams
    python -m wfield_local.locanmf_analysis_deck --src <dir> --out <path.pptx>

WHERE THE DECK'S CODE LIVES (split 2026-09-21; it was 5,484 lines in this file alone):

    locanmf_analysis_deck   THIS FILE -- orchestration. What exists on disk, what order it goes
                            in, the completeness and failed-step gates, provenance and captions.
    deck_text               the PROSE: trial-population lines, per-slide reading notes,
                            methodology blurbs. Pure data, ~1,155 lines.
    deck_registry           the FIGURE REGISTRIES: EPOCH_FIGURES, GRANT_FIGURES, ALIGNS, BASES,
                            NOLICK_BASES and the shared legends. Pure data, ~2,120 lines.
    deck_layout             the DRAWING SURFACE: `SlideCanvas`, which owns the presentation and
                            the placed/missing bookkeeping the completeness gate reads.

Both are re-exported here, so `from wfield_local.locanmf_analysis_deck import EPOCH_FIGURES` and
every existing caller still work. THE SPLIT WAS VERIFIED, not assumed: 87 module constants hashed
before and after and compared exactly, and a 531-slide build fingerprinted shape-for-shape and
note-for-note against the pre-split build. What it DID break, and what had to be fixed with it,
was five source-scraping tests and two audit scripts that read this file by path and would
otherwise have gone on passing over the fragment left behind -- see `tests/conftest.py`
`DECK_MODULES`.
"""
from __future__ import annotations

import argparse
import re
import shutil
import time
from datetime import date
from pathlib import Path

from pptx.util import Inches, Pt

from wfield_local import config, deck_layout, deck_values
from wfield_local import epochs as _epochs

# The figure REGISTRIES live in `deck_registry` (see its docstring): what the deck knows
# about, as data. This module decides which of it EXISTS on disk, and where it goes.
from wfield_local.deck_registry import (   # noqa: F401  (re-exported: the coverage
                                            #  script and tests import EPOCH_FIGURES from here)
    ALIGNS, BASES, EPOCH_FIGURES, GRANT_FIGURES, NOLICK_BASES, _CCF_LEGEND, _CI_LEGEND, _REF_LEGEND, _ROT_LEGEND,
)
# The deck's PROSE lives in `deck_text` (see its docstring). Imported by name rather than
# `import *` so what the builder uses stays greppable and ruff can still see an unused one.
from wfield_local.deck_text import (   # noqa: F401  (re-exported: the audit
                                        #  scripts and tests read these off this module)
    M_CODING_DIR, M_COMMON, M_DECODE, M_ENCODE, M_EVOKED, M_FIXEDSCALE, M_FROZEN, M_FROZEN_ENC, M_GATE, M_HEMI,
    M_HEMIDYN, M_JOINT, M_LICKFREE, M_MISS_STOPPED, M_NOLICK, M_POSTSTROKE, M_PRECUE_CAVEAT, M_RECODING, M_RSA,
    M_SPATIAL, M_VESSEL, S_DEC_CUE, S_DEC_LICK, S_DEC_PRECUE, S_DEC_ROLL, S_DRIFT, S_ENC_FEVE, S_ENC_MATRIX,
    S_ENC_POS, S_FROZEN_ALL, S_FROZEN_ENC, S_FROZEN_SESS, S_G0, S_G1, S_G1B, S_G2, S_G2B, S_G2C, S_G3, S_G4,
    S_G4B, S_G5, S_G6, S_G7, S_G7B, S_G7C, S_G7D, S_G8, S_G8B, S_G8C, S_G8D, S_G8E, S_G8F, S_G9, S_G9B, S_G9C,
    S_G9E, S_GEXCL, S_JOINT, S_LICKFREE, S_NOLICK_A, S_NOLICK_B, S_NOLICK_C, S_RSA_A, S_RSA_B, S_RSA_C,
    S_XCONSIST, S_XMOUSE, TRIALS_BEHAVIOUR, TRIALS_LICK, TRIALS_NOLICK, TRIALS_WORKING, _M_LICK_UNIT, _NL2,
)
from wfield_local.paths import PathResolver

#: Re-exported from `deck_layout`, which owns them now: several tests and the
#: provenance footer below still reach for them on this module.
NAVY, GREY, SLATE = deck_layout.NAVY, deck_layout.GREY, deck_layout.SLATE





def _mmdd_label(mmdd: str) -> str:
    return f"{int(mmdd[:2])}/{int(mmdd[2:])}"


class DeckIncomplete(RuntimeError):
    """Raised by :func:`_refuse_incomplete_overwrite` when a rebuild would publish a deck that is
    missing figures. Carries ``missing_figures`` so the caller can name them instead of truncating."""

    def __init__(self, message, missing_figures):
        super().__init__(message)
        self.missing_figures = list(missing_figures)


def _refuse_incomplete_overwrite(out_path, missing_figures, allow_missing=0):
    """Refuse to replace an EXISTING deck with a rebuild that could not find every figure.

    The deck is rebuilt in place onto MICROSCOPE, so a run whose upstream steps failed quietly
    replaces a good deck with a worse one. That happened on 2026-08-19: await_locanmf fitted LocaNMF
    to the superseded SVTcorr and wrote it to a directory no consumer reads, so the position decoder
    and spatial_reorganisation raised FileNotFoundError, the whole 8/19 LocaNMF column went missing,
    and the build published itself anyway at 20 missing -- 96 MB of deck became 52 MB. Nothing
    raised, because a deck with holes in it is a perfectly valid deck.

    The missing count is only meaningful as a check because the build already excludes figures that
    should not exist (see the G1 loop over post-stroke animals), so a non-zero count means something
    upstream genuinely failed. ``allow_missing=N`` tolerates N of them for the deliberate case; a
    deck that does not exist yet is always allowed, since a tree still filling up is a real case.
    """
    if not missing_figures or len(missing_figures) <= allow_missing:
        return
    if not Path(out_path).exists():
        return
    shown = "\n  ".join(str(m) for m in missing_figures[:20])
    more = f"\n  ... and {len(missing_figures) - 20} more" if len(missing_figures) > 20 else ""
    raise DeckIncomplete(
        f"refusing to overwrite {out_path} with a deck missing {len(missing_figures)} figure(s) "
        f"({Path(out_path).stat().st_size / 1e6:.0f} MB already there). Fix the upstream step that "
        f"did not produce them, or pass allow_missing={len(missing_figures)} to publish anyway. "
        f"Missing:\n  {shown}{more}",
        missing_figures)


class DeckFromFailedRun(RuntimeError):
    """Raised when a rebuild would publish a deck assembled from a run that had failing steps.
    Carries ``failed_steps`` so the caller can name them."""

    def __init__(self, message, failed_steps):
        super().__init__(message)
        self.failed_steps = list(failed_steps)


def _refuse_failed_steps(out_path, failed_steps, allow_failed_steps=False):
    """Refuse to replace an EXISTING deck when a step in the run that fed it FAILED.

    The missing-figure gate cannot see this. A step that dies PART WAY leaves its earlier outputs
    rewritten and its later ones at yesterday's values -- every file present, nothing missing, and
    a deck that silently mixes two days. That is what happened on 2026-08-20: spatial_reorganisation
    rewrote its all-trials arm, then the lick-only arm raised KeyError, and the deck published at
    0 missing with the lick-only panels a day old.

    The run already KNOWS which steps failed, so this needs no freshness heuristic and has no false
    positives: a failed step means its outputs are untrustworthy by definition. Enforced rather than
    warned because the whole point is that the resulting deck looks healthy.
    """
    if not failed_steps or allow_failed_steps:
        return
    if not Path(out_path).exists():
        return
    raise DeckFromFailedRun(
        f"refusing to overwrite {out_path}: {len(failed_steps)} step(s) FAILED in the run that "
        f"produced these figures, so some panels may be left over from an earlier run. Fix them, "
        f"or pass allow_failed_steps=True to publish anyway. Failed: {sorted(set(failed_steps))}",
        failed_steps)


def _write_manifest(out_path, placed_figures, run_start=None):
    """Record every figure the deck PLACED, with its mtime, beside the deck.

    Staleness is REPORTED, not enforced, and the measurement is why. Of 1311 PNGs in the figure tree
    on 2026-08-20 only 402 were touched by that night's run; the rest are one-off and cross-sectional
    analyses going back to June that legitimately do not regenerate. A blanket "must be recent" rule
    would fire on two thirds of the tree every night, and an alarm that always fires is one nobody
    reads.

    Scoped to what the deck actually PLACES it becomes informative: the manifest is the evidence for
    which figures a run refreshed and which it did not, diffable night to night. It is how an
    ORPHANED reference surfaces -- a slide pointing at a filename no step writes any more, invisible
    to every other check because the file is present, just never updated (poststroke_grid.png sat at
    its 08-19 content until 2026-08-20).
    """
    import json as _json

    # one row per FIGURE, not per placement: a figure checked once and drawn again lands in
    # placed_figures twice, and a manifest that lists it twice reads as two different files.
    seen = {}
    for n, m in sorted(placed_figures):
        seen.setdefault(n, m)
    rows = [{"figure": n,
             "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m)),
             "age_days": round((time.time() - m) / 86400, 2),
             "refreshed_this_run": (run_start is not None and m >= run_start)}
            for n, m in sorted(seen.items())]
    man = Path(out_path).with_suffix(".manifest.json")
    try:
        man.write_text(_json.dumps(
            {"deck": str(out_path), "n_placed": len(rows),
             "run_start": (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_start))
                           if run_start else None),
             "figures": rows}, indent=1), encoding="utf-8")
    except OSError as ex:
        print(f"[analysis_deck] could not write manifest: {ex}", flush=True)
        return None, []
    stale = [r for r in rows if run_start is not None and not r["refreshed_this_run"]]
    return man, stale


#: How an alignment token in a figure filename reads in prose.
_ALIGN_PROSE = {
    "precue": "PRE-CUE (ENL): the window ENDS at the cue",
    "cue": "POST-CUE: the window STARTS at the cue",
    "lick": "POST-LICK: the window STARTS at the first detected lick",
}


def window_provenance(fig_names) -> str:
    """One line stating the window and binning behind the figures on a slide.

    WHY EVERY SLIDE CARRIES THIS. On 2026-08-21 decode.max_rt_s moved from 2.0 s to 3.5 s and
    eleven modules kept their own hardcoded 2.0, so for a day this deck showed figures cut at two
    different definitions of "engaged" with nothing on either saying which. A reader cannot tell
    those apart by looking, and neither could I. The parameters are read from configs/defaults.yaml
    at build time and the alignment from the figure's own filename, so this line cannot drift from
    the thing it describes the way a hand-written note does.

    Returns "" when no figure on the slide encodes an alignment -- a schematic or a text slide gets
    no claim rather than a guessed one.
    """
    from wfield_local import config

    d = config.defaults()["decode"]
    names = " ".join(str(n) for n in fig_names)
    align = next((a for a in ("precue", "cue", "lick")
                  if ("_" + a + "_") in names or ("_" + a + ".") in names), None)
    if align is None:
        return ""
    post = float(d.get(align + "_post_s", 2.0))
    nb = int((d.get("bins") or {}).get(align, 1) or 1)
    binning = (f"{nb} sub-bins of {post / nb:.2f} s (the decoder sees a time course, not one mean)"
               if nb > 1 else "1 bin (the window mean)")
    bits = [f"- window: {_ALIGN_PROSE[align]}, {post:.1f} s (+1.0 s before it)",
            f"- binning: {binning}",
            (f"- engaged: first lick within {float(d['max_rt_s']):.1f} s of the cue "
            f"(decode.max_rt_s)")]
    if "base-none" in names:
        bits.append("- baseline: none")
    if "cv-block" in names:
        bits.append("- CV: block (GroupKFold over ~6-trial position blocks)")
    return "HOW IT WAS BUILT" + chr(10) + chr(10).join(bits)


_BASIS_PROSE = {
    "roi": "Allen-ROI basis",
    "joint": "shared joint-LocaNMF basis (the same components on every day, so days are comparable)",
    "locanmf": "that session's OWN LocaNMF basis (within-day only; components are not comparable "
               "across days)",
}
# LONGEST FIRST. "poststroke_all" is a prefix of "poststroke_all_working", and a shortest-first scan
# would label the outcome-blind-minus-quit-period arm as the outcome-blind one -- a wrong caption
# reads exactly like a right one.
_ARM_PROSE = (
    ("poststroke_all_working", ("post-stroke arm: EVERY trial except the terminal quit period "
                                "(outcome-blind, so no lick/miss selection)")),
    ("poststroke_miss_working", "post-stroke arm: MISS trials on which the animal was still working"),
    ("poststroke_stopped", "post-stroke arm: the terminal quit period ONLY"),
    ("poststroke_lick", "post-stroke arm: trials with a detected lick"),
    ("poststroke_all", "post-stroke arm: ALL trials, outcome-blind"),
    ("lickonly", "LICK-ONLY arm: both sides restricted to trials with a lick"),
)
_METHOD_PROSE = {
    "dom": "difference-of-means coding axis",
    "lr": "logistic (LDA-like) coding axis",
}
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _session_years() -> dict:
    """MMDD -> calendar year, read off the YYYYMMDD prefix of each session's own path.

    The captions state a real day count from the lesion, which needs a year. Hardcoding 2026 would
    be right today and silently wrong the first time this cohort runs over New Year, so the year is
    recovered from `sessions.yaml` -- the file that already knows it.
    """
    out = {}
    try:
        for s in config.load_sessions():
            m = re.search(r"(20\d{2})(\d{2})(\d{2})", str(s.get("mc") or ""))
            if m:
                out[m.group(2) + m.group(3)] = int(m.group(1))
    except Exception as exc:                         # noqa: BLE001 - a caption must never break a build
        print(f"[analysis_deck] caption years unavailable ({exc}); captions will name the phase "
              f"instead of a day count")
    return out


def _date_prose(mmdd: str, animal: str | None, years: dict) -> str:
    """'22 Aug 2026 (day 5 after the lesion)'. Falls back to the bare date when anything is unknown."""
    mm, dd = mmdd[:2], mmdd[2:]
    try:
        label = f"{int(dd)} {_MONTHS[int(mm) - 1]}"
    except (ValueError, IndexError):
        return mmdd
    yr = years.get(mmdd)
    if yr:
        label += f" {yr}"
    if not animal:
        return label
    sd = config.stroke_date(animal)
    if not sd:
        return f"{label} (no lesion in this animal)"
    # THE LESION DAY IS USUALLY NOT AN IMAGING DAY, so `sd` is normally absent from `years`
    # (PS94/PS95 were lesioned on 0816 and nothing was recorded that day). While every registered
    # session falls in ONE calendar year, that year is the lesion's year too; the moment the cohort
    # straddles New Year it stops being inferable and the caption says which SIDE of the lesion the
    # session is on rather than invent a day count.
    sd_yr = years.get(sd)
    if sd_yr is None and len(set(years.values())) == 1:
        sd_yr = next(iter(set(years.values())))
    if not yr or sd_yr is None:
        return f"{label} ({'pre-stroke' if mmdd <= sd else 'post-stroke'})"
    d0 = date(sd_yr, int(sd[:2]), int(sd[2:]))
    d1 = date(yr, int(mm), int(dd))
    n = (d1 - d0).days
    if n == 0:
        return f"{label} (the last pre-stroke session; the lesion was induced after it)"
    rel = f"day {n} after the lesion" if n > 0 else f"{abs(n)} days before the lesion"
    return f"{label} ({rel})"


def figure_caption(fig_names, years=None) -> str:
    """A caption naming what THIS slide's figure actually is: animal, day, window, basis, arm.

    WHY THIS IS GENERATED AND NOT WRITTEN. Every figure slide already carried speaker notes, so the
    gap this fills is not absence -- it is SPECIFICITY. The notes are written per FAMILY and the deck
    places a family once per animal x date x window, so on 2026-08-25 an audit found the same "THIS
    SLIDE" paragraph on 88 slides, another on 80 and another on 72. A caption repeated 88 times tells
    a reader which family they are in and nothing about the panel in front of them; the one fact they
    need -- is this PS93 on day 4 in the joint basis, or PS95 pre-stroke in the ROI basis -- was
    legible only from the filename, which the deck does not show.

    Derived from the figure's own name, like `window_provenance`, so it cannot drift from what it
    describes. Returns "" when a name encodes nothing (a schematic, a cohort-wide summary): a slide
    gets no caption rather than a guessed one.
    """
    years = _session_years() if years is None else years
    lines = []
    for raw in fig_names:
        stem = Path(str(raw)).stem
        toks = stem.split("_")
        bits = []
        an = next((t for t in toks if re.fullmatch(r"PS\d{2}", t)), None)
        if an:
            bits.append(an)
        rng = next((t for t in toks if re.fullmatch(r"\d{4}-\d{4}", t)), None)
        dates_ = [t for t in toks if re.fullmatch(r"(0[1-9]|1[0-2])[0-3]\d", t)]
        if rng:
            a, b = rng.split("-")
            bits.append(f"sessions {_date_prose(a, None, years)} to {_date_prose(b, None, years)}")
        elif dates_:
            bits.append("; ".join(_date_prose(d, an, years) for d in dates_))
        align = next((a for a in ("precue", "cue", "lick") if a in toks), None)
        if align:
            bits.append(_ALIGN_PROSE[align].split(":")[0] + " window")
        basis = next((b for b in ("joint", "roi", "locanmf") if b in toks), None)
        # "locanmf" leads almost every filename as a FAMILY prefix; it only names a basis when it
        # sits beside an alignment token, as in ..._locanmf_precue_base-none_cv-block.
        if basis == "locanmf" and not (align and f"locanmf_{align}" in stem):
            basis = None
        if basis:
            bits.append(_BASIS_PROSE[basis])
        arm = next((p for k, p in _ARM_PROSE if k in stem), None)
        if arm:
            bits.append(arm)
        meth = next((_METHOD_PROSE[m] for m in ("dom", "lr") if m in toks), None)
        if meth:
            bits.append(meth + (", orthogonalised to the engagement axis" if "orth" in toks else ""))
        pg = next((t for t in toks if re.fullmatch(r"p\d+", t)), None)
        if pg:
            bits.append(f"page {pg[1:]} of this family")
        # THE FILENAME GOES IN EITHER WAY. 29 figure slides come from families whose names encode
        # none of the tokens above (the G1b coverage grids, the pooled encoder panels, the G8 raw
        # fluorescence series, the grant set) and returning "" for them left exactly the slides
        # whose titles are least self-explanatory with no caption at all. A filename invents
        # nothing -- it is the identity of the thing on the slide, and it is what a reader needs to
        # regenerate or interrogate it.
        lines.append(Path(str(raw)).name + ((chr(10) + "    " + " · ".join(bits)) if bits else ""))
    return ("FIGURE" + chr(10) + (chr(10)).join(lines)) if lines else ""


def keep_previous(out_path) -> Path | None:
    """Copy the deck that is about to be overwritten into ``deck_history/``, stamped with ITS mtime.

    Every rebuild writes the same filename, so until 2026-08-22 each one destroyed the last. That is
    fine while rebuilds only add -- and not fine the moment one is worse: a hand-run rebuild that day
    published 249 slides over 265, and the only way back was to rebuild again, not to recover. A deck
    is cheap to copy and expensive to reproduce (hours of figures), so the previous one is kept.

    Returns the archived path, or None if there was nothing to keep. Never raises: failing to make a
    backup must not stop the deck from being written.
    """
    from wfield_local import writeguard

    try:
        out_path = Path(out_path)
        if not out_path.exists():
            return None
        hist = out_path.parent / "deck_history"
        stamp = time.strftime("%Y%m%d_%H%M", time.localtime(out_path.stat().st_mtime))
        dst = hist / f"{out_path.stem}__{stamp}{out_path.suffix}"
        if dst.exists():
            return dst
        writeguard.assert_writable(dst)
        hist.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, dst)
        print(f"[deck] kept the previous version as {dst.name}", flush=True)
        return dst
    except Exception as ex:                                       # noqa: BLE001
        print(f"[deck] could not keep the previous version ({type(ex).__name__}: {str(ex)[:60]}); "
              f"writing anyway", flush=True)
        return None




def build_analysis_deck(src: Path, out_path: Path, dates=None, animals=None, tag=None, allow_missing=0,
                        failed_steps=(), allow_failed_steps=False, run_start=None,
                        grant_dir=None) -> dict:
    """Build the refined analysis deck at ``out_path`` from figures in ``src``. Returns a summary dict.

    ``grant_dir`` is the ONE input that does not live under ``src`` -- the grant summary set is a
    deliverable under ``labcams``, not an analysis intermediate under ``figures_working``. It is an
    explicit parameter rather than a bare resolver call so a caller can point it somewhere else, and
    so a test can keep the build hermetic instead of silently reaching onto the MICROSCOPE share.
    """
    src = Path(src)
    grant_dir = Path(grant_dir) if grant_dir is not None else (
        Path(PathResolver().root("labcams")) / "grant_figures")
    # phase="all" IS LOAD-BEARING. curated_dates() defaults to phase="pre" (stroke-aware since
    # 2026-08-17), so the bare call returns 0606-0814 and SILENTLY DROPS EVERY POST-STROKE DATE.
    # The comment that used to sit here said a hand-run deck covers the same dates as the nightly.
    # That was true when written and stopped being true the day the phase default landed: on
    # 2026-08-22 a hand-run rebuild published 249 slides over the nightly's 265, and the 16 it lost
    # were the post-stroke sections -- the part of the study this deck exists to show. The nightly
    # never hit it because it computes its own list (registered minus excluded) and passes it in.
    dates = dates or config.curated_dates(phase="all")
    animals = animals or [a for a in config.animals()]
    tag = tag or f"{dates[0]}-{dates[-1]}"
    date_labels = [(d, _mmdd_label(d)) for d in dates]

    # THE DRAWING SURFACE lives in `deck_layout` (see its docstring): the presentation, the
    # placed/missing counters, the figure manifest rows and the methods-dedup table, which were
    # eleven closures sharing eleven locals here until 2026-09-21.
    #
    # SIDECAR RESOLVER for the canvas's `note`. Searched in order: the grant summary sets, the
    # epoch set, then the working figure dir, which is the same precedence the placement patterns
    # below use -- which is why the ROOTS are chosen here and not in the canvas. A family whose CSV
    # is absent yields a visible marker in the note and a line in the build log -- never a stale
    # number and never a failed build.
    canvas = deck_layout.SlideCanvas(deck_values.Resolver([grant_dir, grant_dir / "epoch", src]))

    # BOUND AS LOCAL NAMES so the ~250 section call sites below read as the narrative they are --
    # `title(s, ...)`, `big(s, fig)` -- rather than repeating the same receiver on every line. It
    # also meant the 2026-09-21 split moved the definitions without editing the body, so the
    # 531-slide fingerprint that proved the deck byte-for-byte unchanged tested the MOVE.
    prs = canvas.prs
    slide, title, note, big, grid, bullets, divider = (
        canvas.slide, canvas.title, canvas.note, canvas.big, canvas.grid, canvas.bullets,
        canvas.divider)
    _exists = canvas.exists
    placed = canvas.placed
    missing_figures = canvas.missing_figures
    placed_figures = canvas.placed_figures
    slide_order = canvas.slide_order
    figs_by_slide = canvas.figs_by_slide
    values = canvas.values

    # A SESSION THAT DOES NOT EXIST IS NOT A MISSING FIGURE.
    # The per-session slides iterate animals x dates, which assumes every animal ran every night.
    # It does not: 8/22 is PS92 and PS93 only, and the deck counted eight PS94/PS95 figures as
    # missing and refused to publish over sessions that were never recorded (2026-08-23). The
    # completeness gate is worth keeping -- it is what catches a step that genuinely failed -- so
    # the fix is to stop it expecting the impossible rather than to loosen it.
    _registered = {(config.animal_of(x["label"]), x["label"].split("_")[-1])
                   for x in config.load_sessions()}

    def have(animal, mmdd) -> bool:
        """Was this animal actually recorded on this date?"""
        return (animal, mmdd) in _registered

    def sess(label, align):
        return src / f"locanmf_position_session_{label}_locanmf_{align}_base-none_cv-block.png"

    # ---------------- title ----------------
    s = slide()
    tf = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(3.0)).text_frame
    tf.word_wrap = True
    r = tf.paragraphs[0].add_run()
    r.text = "Spout-position decoding & encoding from cortex"
    r.font.size = Pt(38)
    r.font.bold = True
    r.font.color.rgb = NAVY
    for t in [(f"Curated pre-stroke sessions ({', '.join(_mmdd_label(d) for d in dates)}) — "
              f"{', '.join(animals)} (PS93 = right orofacial deficit)"),
              "Individual LocaNMF components, block-aware CV, no per-trial baseline, chance = 0.17.",
              ("A–C within-day, grouped animal → analysis type → date.  D cross-session (frozen), "
              "grouped basis → alignment → animal.  E–F cohort summaries.")]:
        rr = tf.add_paragraph().add_run()
        rr.text = t
        rr.font.size = Pt(15)
        rr.font.color.rgb = GREY
    note(s, "Refined analysis deck: spout-position decode/encode/RSA, grouped animal -> analysis type -> "
            "date, curated pre-stroke sessions only. Each slide's speaker notes give how that figure is "
            "made. " + M_COMMON)

    # ---------------- METHODOLOGY: right after the title, before any result ----------------
    # This slide used to warn that the pre-cue numbers were inflated ~2x. They are not any more: the
    # deck is built on the corrected variant. A stale warning is worse than none -- it tells a reader
    # to discount numbers that are now right.
    s = slide()
    title(s, "Drift removal — the pre-cue numbers in this deck are CORRECTED",
          "Built on meegkit_hpfit, not the pipeline default. Read this before comparing with anything "
          "produced before 14 Aug 2026.")
    tf = s.shapes.add_textbox(Inches(0.6), Inches(1.7), Inches(12.1), Inches(5.3)).text_frame
    tf.word_wrap = True
    for i, line in enumerate([
        ("WHAT WAS WRONG: wfield.hemodynamic_correction high-passes both channels at 0.1 Hz with scipy "
        "filtfilt — zero-phase, therefore ACAUSAL — and that high-passed 470 channel becomes SVTcorr. "
        "Its impulse response is symmetric in time (−0.496 before an impulse, −0.496 after), so a "
        "position-specific POST-cue response cast a sign-flipped shadow BACKWARDS into the pre-cue "
        "window. A linear decoder does not care about sign, so the shadow read as pre-cue information."),
        ("THE FIX (adopted 2026-08-14): keep the 0.1 Hz high-pass for the hemodynamic COEFFICIENT fit — "
        "that is what it is for — and replace it for the OUTPUT with de Cheveigné robust polynomial "
        "detrending (order 10, 600 s) on a mask that excludes whole trials."),
        "MEASURED over ALL 36 CURATED SESSIONS:",
        "                        pre-cue        post-cue (control)",
        "        zerophase (old)       0.486          0.684",
        "        meegkit_hpfit (now)   0.352          0.759      post-cue IMPROVED",
        ("The variant that most IMPROVES the readout we trust also most REDUCES the one we suspected — "
        "the strongest form this comparison could take."),
        ("WHAT SURVIVES: pre-cue position information is REAL and significant in 35/36 sessions, at "
        "~72% of the previously reported size. PS92 0.225, PS93 0.349, PS94 0.500, PS95 0.334 "
        "(chance 0.167; empirical null 0.137–0.147 by block-label permutation). PS94 was essentially "
        "untouched; PS92 was the one substantially inflated and is now well above chance, not at it."),
        ("SIGN TEST: the pre-cue pattern used to be ANTI-correlated with the post-cue pattern (negative "
        "in 30 of 36 sessions; on the worst days the pre-cue MAP was literally the negative of the "
        "post-cue map, r = −0.93). After correction that signature is gone — negative in 2 of 36."),
        ("NOT A LOCAL BUG: churchlandlab/WidefieldImager SvdHemoCorrect.m does the same in-place "
        "filtfilt; Musall et al. 2019 state it in their methods. The artifact class is published — "
        "van Driel, Olivers & Fahrenfort 2021, J Neurosci Methods — including the negative sign, with "
        "trial-masked robust detrending as the recommended fix, which is what was adopted."),
        ("REPRODUCE: python -m wfield_local.filter_acausality_test <LABEL,...>   •   see "
        "docs/PREPROCESSING_DECISION.md and DECISIONS.md"),
    ]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = line
        r.font.size = Pt(12.5)
        r.font.color.rgb = NAVY if line.startswith(("WHAT", "THE FIX", "MEASURED", "SIGN", "NOT")) else GREY
    note(s, M_PRECUE_CAVEAT, specific=S_DRIFT)

    # ---------------- A. per-animal WITHIN-DAY decoding ----------------
    divider("A. Per-animal WITHIN-DAY decoding across sessions",
            "Post-cue 2 s (predicts no-lick trials too = no lick generalization) and pre-cue 2 s "
            "(pre-cue position information) confusion + recall; then the rolling decoder across sessions. "
            "Cross-day (frozen) decoding is Section D.")
    for a in animals:
        s = slide()
        title(s, f"{a} — post-cue 2 s decoder (engaged, no-lick generalization)",
              "Per session: confusion matrix + per-position recall (engaged vs held-out no-lick trials).",
              trials=TRIALS_LICK)
        note(s, M_DECODE, specific=S_DEC_CUE)
        grid(s, [sess(f"{a}_{d}", "cue") for d, _ in date_labels if have(a, d)],
             cols=3)
        s = slide()
        title(s, f"{a} — pre-cue 2 s decoder (pre-cue position information)",
              "Position decodable in the pre-cue ENL window, before movement. NB the accuracies shown "
              "are corrected (meegkit_hpfit); see slide 2 for the drift-removal decision.",
              trials=TRIALS_LICK)
        note(s, M_DECODE + M_PRECUE_CAVEAT, specific=S_DEC_PRECUE)
        grid(s, [sess(f"{a}_{d}", "precue") for d, _ in date_labels if have(a, d)],
             cols=3)
        # LICK-ALIGNED, added 2026-08-26. `nightly_figs` has been running --align lick per day all
        # along, so these figures were written every night and shown nowhere -- computed-and-hidden,
        # the mirror of the frozen lick arm that was shown-nowhere-because-never-computed.
        s = slide()
        title(s, f"{a} — post-LICK 2 s decoder (aligned to the movement)",
              "Per session, registered on the FIRST LICK rather than the cue. Read against the "
              "post-cue slide above: cue-aligned mixes trials with different reaction times, so a "
              "movement-locked signal is smeared by RT jitter there and sharp here.",
              trials=TRIALS_LICK)
        note(s, M_DECODE, specific=S_DEC_LICK)
        grid(s, [sess(f"{a}_{d}", "lick") for d, _ in date_labels if have(a, d)],
             cols=3)
        s = slide()
        title(s, f"{a} — rolling decoder across sessions (pre-cue ENL → post-cue)",
              "Sliding 0.5 s window, block-CV, one line per session. Above-chance in the ENL = position information present before the cue. "
              "(Per-animal accuracy across sessions is in the cross-session summary, Section E.)",
              trials=TRIALS_LICK)
        note(s, M_DECODE, specific=S_DEC_ROLL)
        big(s, src / f"locanmf_decoder_rolling_by_animal_{a}.png", top=1.5, width=11.2)

    # ---------------- B. per-animal encoder ----------------
    divider("B. Per-animal WITHIN-DAY encoder — expected activity, predicted maps & explained variance",
            "Position → expected cortical activity (SSp / MO), footprint-reconstructed predicted maps, and "
            "encoding explained variance per position (raw + relative to the noise ceiling) across sessions.")
    for a in animals:
        # CUT 2026-08-19 (Priya): the expected-SSp/MO time-courses and the footprint-reconstructed
        # predicted maps. They were a gut check that the encoder is not degenerate, and in three
        # months never changed a conclusion -- the noise ceiling on the next slide does that job
        # quantitatively. Recover from git history if a reviewer ever asks to see them.
        s = slide()
        title(s, f"{a} — encoder explained variance per position across sessions (raw & vs ceiling)",
              "One graph per animal; sessions distinguished by colour/marker. Left: raw held-out R²; "
              "right: relative to the per-position noise ceiling.",
              trials=TRIALS_LICK)
        note(s, M_ENCODE, specific=S_ENC_POS)
        grid(s, [src / f"locanmf_encoder_ev_by_position_animal_{a}.png",
                 src / f"locanmf_encoder_ev_ceiling_by_position_animal_{a}.png"], cols=2, top=1.5)
        # CUT 2026-08-19 (Priya): per-SESSION encoder r2 by region. The region axis is rebuilt per
        # session, so a cell in one panel is not the same region-set as the cell beside it and the
        # panels cannot be read against each other.
    # CUT 2026-09-13 (Priya, "it's the pooled FEVE i wanted to get rid of, slide 25"): the POOLED
    # FEVE-by-region slide, `locanmf_encoder_feve_by_region_pooled.png`.
    #
    # THE PER-ANIMAL ENCODER EV SLIDES ABOVE STAY -- they are the four this was easy to confuse with,
    # and the confusion was real: the edit list named "FEVE pooled" against slide 23, which is PS94's
    # encoder EV, while the pooled FEVE is slide 25. Deleting on the number alone would have removed
    # ONE animal's EV panel and left the other three, which is not a coherent deck state and would
    # have looked deliberate.
    #
    # WHY IT GOES, now that the 2026-08-19 note above no longer holds it up: that note kept the
    # pooled slide as "the comparable form" of the per-session r2 it was cutting. The comparison it
    # served has since moved to the encoder-CEILING family, which reports the same fraction-of-
    # explainable-variance question per POSITION and per EPOCH with bootstrap intervals, rather than
    # per Allen region pooled over a phase. The figure is still written nightly; only the slide goes.
    # CUT 2026-08-19 (Priya): the per-SESSION FEVE heatmap. Same objection as the per-session
    # per-region r2 above -- the region axis is not fixed across sessions, so "stability" cannot be
    # read off it.

    if (src / "locanmf_encoder_ev_matrix.png").exists():
        s = slide()
        title(s, "Encoder — encoded variance per POSITION x SESSION, all animals on one scale",
              "The summary the per-animal bar charts could not give (Priya, 2026-08-19): a position "
              "that degrades across days is a COLUMN that changes colour, and the lesion is a rule "
              "rather than something the reader has to hold in mind. Ridge from a one-hot position "
              "design, scored per position with the same block GroupKFold the decoders use. ONE "
              "colour scale across animals, so the panels are comparable.",
              trials=TRIALS_LICK)
        note(s, M_ENCODE, specific=S_ENC_MATRIX)
        big(s, src / "locanmf_encoder_ev_matrix.png", top=1.6, width=12.6)

    # ---------------- C. pre-cue without licking ----------------
    divider("C. Pre-cue code without licking — the motor-confound control",
            "Decode AND encode on a SEARCHED 2 s window: 2 consecutive lick-free seconds between the "
            "spout-position strobe and the cue, taken as late as possible.")
    for src_name in ("roi", "locanmf"):
        for a in animals:
            p = src / f"precue_lickfree_{a}_{src_name}.png"
            if not p.exists():
                continue
            s = slide()
            title(s, f"{a} — pre-cue position code with NO licking in the window ({src_name})"
                     "",
                  "Exposure, decode (lick-free vs all vs with-licks), lick-free confusion matrix, and "
                  "per-region encoding EV. The lick control itself is VALID — it just sits on top of "
                  "corrected pre-cue values (meegkit_hpfit); see slide 2.",
                  trials=TRIALS_NOLICK)
            note(s, M_LICKFREE + M_PRECUE_CAVEAT, specific=S_LICKFREE)
            big(s, p, top=1.5, width=12.9)

    # ---------------- D. cross-session (frozen) decoders & encoders, BOTH bases ----------------
    # Was interleaved into each animal's Section A, which mixed "does the code exist today" with "does
    # it transfer across days". Now its own section, and now in TWO bases: Allen-ROI (conservative,
    # atlas-anchored) and the shared joint-LocaNMF basis (finer, more sensitive). A cross-day claim
    # that survives both is about the cortex; one that appears in only one is about the parcellation.
    divider("D. CROSS-SESSION (frozen) decoders & encoders — two independent bases",
            "Every day predicted by a model trained ONLY on that animal's other days "
            "(leave-one-session-out). Allen-ROI (66 anatomical areas) and the shared joint-LocaNMF "
            "basis (~95–137 functional components, footprints frozen and shared across days).")
    # PAGINATED 4-per-slide (2x2). All curated dates on one slide at cols=2 gives 4+ rows, so each
    # panel gets ~1/4 of the slide height and the 6x6 confusion cells become unreadable. 2x2 doubles
    # the height per panel; the cost is one extra slide per animal.
    pages = [date_labels[i:i + 4] for i in range(0, len(date_labels), 4)]
    for bkey, bname, m_dec, m_enc, bdesc in BASES:
        if not any((src / f"locanmf_frozen_decoder_loso_{bkey}_{al}.png").exists()
                   for al, _, _ in ALIGNS):
            continue                      # basis not computed (e.g. no joint basis built yet)
        divider(f"D — {bname} basis", bdesc)
        if bkey == "joint":
            # NO `slide()` HERE. There used to be one, left behind when this block stopped showing a
            # single basis-health figure and started looping over the three alignments below -- each
            # of which opens its own slide. The orphan published a completely BLANK slide in every
            # deck (narrative slide 57 at the 2026-09-12 build) and nothing flagged it, because an
            # empty slide is not a missing figure: `_exists` counts figures, and this one asked for
            # none. Priya, 2026-09-13: delete it.
            # THE PRE-CUE FILE, and the slide has to say so: `joint_basis_health_{align}.png` is
            # written per alignment and the span is computed on the ALIGNED window, so the cue
            # figure is a different measurement. Only one is shown, and until 2026-08-24 neither
            # the figure nor this title named it.
            # ALL THREE ALIGNMENTS, not the pre-cue one alone. The span is computed on the
            # ALIGNED window, so these are three different measurements, and the deck used to show
            # one and say in its own subtitle that the others "are a different measurement and are
            # not shown here" -- an accurate note about an incomplete slide. Each of the three
            # decode arms in section D is read against its OWN basis-health figure now, which is the
            # only way "low-and-low means the basis under-describes that day" can be checked for the
            # arm actually being read.
            for _bal, _balname in (("precue", "PRE-CUE"), ("cue", "POST-CUE"), ("lick", "POST-LICK")):
                _bh = src / f"joint_basis_health_{_bal}.png"
                if not _bh.exists():
                    continue
                s = slide()
                title(s, f"Joint-basis health ({_balname} window) \u2014 how much of each session "
                         f"the frozen footprints span",
                      "Sessions IN the fit are 1.0 by construction (hollow); a PROJECTED day "
                      "(filled) is not. Read a projected day's decode accuracy against its bar: "
                      "low-and-low means the basis under-describes that day, not that its "
                      "representation changed. THE SPAN IS MEASURED ON THE ALIGNED WINDOW, so the "
                      "three alignments are three different numbers for the same session and each "
                      "belongs with its own decode arm \u2014 which is why all three are here "
                      "rather than the pre-cue one standing for all of them.",
                      trials=TRIALS_LICK)
                note(s, M_JOINT, specific=S_JOINT)
                big(s, _bh, top=1.7, width=12.2)
        for al, al_name, al_desc in ALIGNS:
            # the pre-cue arm inherits the zero-phase-filter inflation; the post-cue arm does not
            cav = M_PRECUE_CAVEAT if al == "precue" else ""
            warn = ""
            # ONE SLIDE PER ANIMAL (Priya, 2026-08-28: "can we make those figures smaller to fit
            # figures for one animal on the slides"). Four-to-a-slide meant five slides per animal
            # per alignment per basis -- 155 in section D -- and a reader comparing day 1 with day 12
            # had to page between them. The whole point of a frozen decoder is the TRAJECTORY across
            # days, and the layout hid exactly that.
            #
            # `write_animal_confusion_grid` draws every held-out day as one small matrix with
            # `POS_SHORT` tick labels on the edge panels only, post-stroke dates in red. It drops the
            # recall bars (they have their own summary figure below) and the in-cell numbers, which
            # at ~1.5in are too small to read and compete with the colour that is not.
            #
            # The per-day figures are still written and still reachable; they are simply no longer
            # the thing 155 slides are spent on.
            for a in animals:
                _g = src / f"locanmf_frozen_grid_{a}_{bkey}_{al}.png"
                if _exists(_g):
                    s = slide()
                    title(s, f"{a} — FROZEN cross-day decoder, {al_name}, every held-out day "
                             f"({bname}){warn}",
                          f"One matrix per date, rows = TRUE position. The number beside each date "
                          f"is that day's held-out accuracy; POST-STROKE dates are red. Trained on "
                          f"this animal's PRE-STROKE days only. {al_desc}.",
                          trials=TRIALS_LICK)
                    note(s, m_dec + cav, specific=S_FROZEN_SESS)
                    big(s, _g, top=1.7, width=12.9)
                    continue
                for page in pages:                      # fallback: the per-day figures, 4 to a slide
                    span = f"{page[0][1]}–{page[-1][1]}" if len(page) > 1 else page[0][1]
                    suffix = f"  ({span})" if len(pages) > 1 else ""
                    s = slide()
                    title(s, f"{a} — FROZEN cross-day decoder, {al_name}, held-out day "
                             f"({bname}){suffix}{warn}",
                          f"Per date: confusion + per-position recall from a decoder trained on this "
                          f"animal's OTHER days only. {al_desc}.",
                          trials=TRIALS_LICK)
                    note(s, m_dec + cav, specific=S_FROZEN_SESS)
                    grid(s, [src / f"locanmf_frozen_session_{a}_{d}_{bkey}_{al}.png"
                             for d, _ in page if have(a, d)],
                         cols=2, top=1.35)
            s = slide()
            title(s, f"FROZEN decoder ({al_name}, {bname}): transfer cost & OOD control — all "
                     f"animals{warn}",
                  "Held-out day vs same-day ceiling per session; the cost of freezing across days; and "
                  "the OOD control — a softmax decoder never abstains, so confidence alone is not "
                  "evidence.",
                  trials=TRIALS_LICK)
            note(s, m_dec + cav, specific=S_FROZEN_ALL)
            big(s, src / f"locanmf_frozen_decoder_loso_{bkey}_{al}.png", top=1.9, width=12.7)
            s = slide()
            title(s, f"FROZEN cross-day ENCODER ({al_name}, {bname}): position → activity — all "
                     f"animals{warn}",
                  "Held-out-day EV against that day's own noise ceiling, and the ceiling-normalised "
                  "FEVE. The forward model for post-stroke residuals — note its transfer cost is "
                  "NEGATIVE where the decoder's is positive.",
                  trials=TRIALS_LICK)
            note(s, m_enc + cav, specific=S_FROZEN_ENC)
            big(s, src / f"locanmf_frozen_encoder_loso_{bkey}_{al}.png", top=1.9, width=12.7)

    # ---------------- D2. no-detected-lick reference ----------------
    # The pre-stroke reference for reading POST-stroke failed trials. Placed immediately after the
    # frozen decoder because it uses the same frozen model and answers the question that motivates
    # freezing one at all.
    # Rendered in BOTH poolable bases (Allen-ROI and joint-LocaNMF), like Section D, plus an
    # agreement panel -- a result that appears in only one parcellation is a result about the
    # parcellation, and quoting whichever basis was run is exactly the failure two bases prevent.
    # BOTH BASES, ONE CUT. The response-window variants were dropped from the deck on 2026-08-28
    # (Priya: "remove D2 respwin once the early/late lands").
    #
    # The reason they were here still stands as a QUESTION -- moving the cut to the response window
    # reclassifies the late-but-successful trials as engaged, and a deck showing one cut would be
    # showing a choice rather than a result. What changed is that G9e now answers that question
    # DIRECTLY: the trials that move between the two cuts ARE its "late" class, so they get their
    # own panel instead of having to be inferred by differencing two four-slide arms.
    #
    # NOT deleted, only unplaced. `nolick_decoder` still writes both cuts (`--cut respwin`) and
    # `nolick_reference_{roi,joint}_respwin.png` is still on disk, so nothing has to be recomputed to
    # put them back. Read G9e with one caveat the differencing did not have: the respwin arm uses
    # each SESSION's own response window from gui_config.json, while G9e's boundary is a fixed 2.0 s,
    # so the two populations are close but not identical.
    _nl = [(nice, src / f"nolick_reference_{b}.png") for b, nice in NOLICK_BASES
           if (src / f"nolick_reference_{b}.png").exists()]
    if _nl:
        divider("D2 - Trials with NO DETECTED LICK",
                "The pre-stroke reference for post-stroke failures. A failed trial can mean the plan "
                "was never formed or that it was formed and the movement failed; those are different "
                "injuries and identical in the behaviour log.")
        for nice, fig_ref in _nl:
            s_ = slide()
            title(s_, f"No-detected-lick ({nice}): does the position code survive without a movement?",
                  "Balanced accuracy (macro-recall) per arm, pre-cue beside post-cue. The BLACK RULE "
                  "on each bar is that arm's OWN permutation null, not a shared 1/6 - the nulls "
                  "differ per arm and a single chance line would misrepresent every bar but the "
                  "engaged one.")
            note(s_, M_NOLICK, specific=S_NOLICK_A)
            big(s_, fig_ref, top=1.9, width=12.7)
            s_ = slide()
            title(s_, f"No-detected-lick ({nice}): PRE-cue surviving while POST-cue collapses = "
                      f"plan formed, movement failed",
                  "The discriminating quantity. Post-cue decoding is largely driven by the lick "
                  "itself, so it should collapse without one; pre-cue reflects a maintained code "
                  "that need not.")
            note(s_, M_NOLICK, specific=S_NOLICK_B)
            big(s_, fig_ref.with_name(fig_ref.name.replace("reference", "survival")),
                top=1.9, width=10.5)
            # THE PER-SESSION SLIDE IS CUT (Priya, 2026-09-13), and the reason it was added is
            # recorded here because it has not stopped being true. It went in on 2026-08-28 against
            # the computed-and-hidden pattern -- `nolick_per_session_*.png` was written nightly and
            # embedded nowhere -- and the argument was that the pooled bar says what an animal does
            # on AVERAGE while the post-stroke comparison is made one session at a time. What has
            # changed is that section I now carries the epoch-stratified per-session form of the
            # same quantity, with bootstrap intervals this slide never had. THE FIGURE IS STILL
            # WRITTEN EVERY NIGHT and can be put back by restoring this block; nothing is recomputed.
        if (src / "nolick_basis_agreement.png").exists():
            s_ = slide()
            title(s_, "No-detected-lick: do the two bases agree?",
                  "Same trials, same statistics, two independent feature sets. Disagreement is "
                  "reported in red rather than resolved by preference.")
            note(s_, M_NOLICK, specific=S_NOLICK_C)
            big(s_, src / "nolick_basis_agreement.png", top=1.9, width=11.5)

    # ---------------- E. cross-session summary ----------------
    divider("E. Cross-session summary — decoder recall & encoder accuracy across sessions")
    s = slide()
    # SCOPE COMES FROM THE PHASE, NOT FROM `dates`. This interpolated the full cross-session range
    # (e.g. 0606-0825) while the figure itself is pre-stroke only, so the title asserted a span the
    # panels do not cover.
    _pre_d = config.prestroke_dates()
    title(s, f"Cross-mouse decoding & encoding — PRE-STROKE baseline "
             f"({_mmdd_label(_pre_d[0])}–{_mmdd_label(_pre_d[-1])})",
          "Per-mouse overall + per-position decoding and encoding EV, mean ± SEM across that animal's "
          "PRE-STROKE sessions (points = sessions). Baseline question: do the mice differ from each "
          "other? The post-stroke comparison lives in section G.",
          trials=TRIALS_LICK)
    note(s, M_DECODE + " " + M_ENCODE, specific=S_XMOUSE)
    big(s, src / f"locanmf_cross_mouse_comparison_{tag}.png", top=1.5, width=12.7)
    s = slide()
    title(s, "Within-animal consistency of per-position decode / encode — PRE-STROKE noise floor",
          "Per-position profile per session + mean ± SD. This IS the session-to-session floor a "
          "post-stroke change must exceed, so it is built from pre-stroke sessions only.")
    note(s, M_DECODE + " " + M_ENCODE, specific=S_XCONSIST)
    big(s, src / f"locanmf_within_animal_consistency_{tag}.png", top=1.5, width=12.9)

    # ---------------- F. RSA ----------------
    divider("F. RSA — representational geometry of spout position",
            "Within- vs across-animal second-order RSA, per-animal RDM, and the noise-unbiased crossnobis RDM.")
    s = slide()
    title(s, "RSA — within- vs across-animal representational geometry",
          "6×6 position RDM per session; 2nd-order RSA (basis-free). Within-animal > across = stable individual geometry.",
          trials=TRIALS_LICK)
    note(s, M_RSA, specific=S_RSA_A)
    big(s, src / f"locanmf_rsa_sessions_{tag}.png", top=1.6, width=13.0)
    s = slide()
    title(s, "RSA — mean representational dissimilarity matrix per animal",
          "How the 6 positions relate (dark = similar patterns, bright = distinct).",
          trials=TRIALS_LICK)
    note(s, M_RSA, specific=S_RSA_B)
    big(s, src / f"locanmf_rsa_rdms_{tag}.png", top=1.9, width=12.7)
    s = slide()
    title(s, "RSA — crossnobis (noise-unbiased) RDM",
          "Crossnobis removes the positive noise bias → the honest cross-day / pre-post geometry metric.",
          trials=TRIALS_LICK)
    note(s, M_RSA, specific=S_RSA_C)
    big(s, src / f"locanmf_rsa_crossnobis_{tag}.png", top=1.65, width=13.0)

    # HEMISPHERE-RESOLVED RSA. Written by `locanmf_rsa` on the same tag as the three slides above and
    # never placed on any of them, so the one RSA arm that is explicitly ABOUT the lesion's side has
    # been rendered every night and read by nobody.
    #
    # WHY IT BELONGS HERE RATHER THAN IN SECTION G. It is a pre-stroke geometry measurement, like the
    # rest of F -- it establishes what left-vs-right asymmetry looks like BEFORE any lesion, which is
    # the reference a post-stroke asymmetry has to be read against. Put in G it would look like a
    # result about the stroke.
    #
    # EVERY lesion in this cohort is LEFT-sided (`configs/animals.yaml stroke_laterality`), and
    # PS93's orofacial deficit is on the RIGHT -- crossed, as expected. So the LEFT hemisphere is
    # the IPSILESIONAL one, and it is also the hemisphere that REPRESENTS the impaired right side.
    # Read the left panel for the deficit.
    #
    # THIS COMMENT AND THE SLIDE BOTH SAID "CONTRALESIONAL" UNTIL 2026-09-17, which is the exact
    # inversion they warn about. The ADVICE was right -- read the left panel -- so the error was
    # invisible from the figure; only the stated reason was wrong, which is how a wrong reason
    # survives to be quoted somewhere the advice does not come with it.
    #
    # AND THE LABELS THEMSELVES WERE VERIFIED, not assumed, because an image-space flip would have
    # made the correct word the wrong one. Pre-stroke, family `restw`, six independent area pairs:
    # a RIGHT-side spout drives every `_left` region harder and every `_right` region less
    # (laterality index SSp-n +0.34/-0.35, MOp +0.23/-0.14, SSp-un +0.18/-0.17, SSp-bfd, SSp-ul
    # all the same sign pattern). Somatosensation is crossed, so `_left` IS the animal's left.
    for _hk, _htitle, _hsub in (
        ("rdms", "per-animal RDMs, split by hemisphere",
         "Top row = LEFT-hemisphere components, bottom = RIGHT, same 6 positions and the same "
         "colour scale in both. A position geometry that is genuinely bilateral looks the same in "
         "the two rows; a hemisphere that carries the code differently is visible as a row that "
         "does not match. This is the PRE-stroke reference for that comparison, so the asymmetry "
         "seen here is the baseline a post-stroke asymmetry has to beat."),
        ("summary", "animal x animal RDM similarity, per hemisphere",
         "Is the geometry shared ACROSS animals within a hemisphere? Left and right are scored "
         "separately, so a value that is high in one panel and low in the other says the shared "
         "structure is hemisphere-specific. EVERY LESION IN THIS COHORT IS LEFT-SIDED and PS93's "
         "orofacial deficit is on the RIGHT, so the LEFT hemisphere is the IPSILESIONAL one and "
         "also the one representing the impaired side \u2014 read the left panel for it, not the "
         "right. Atlas '_left' is the animal's left: pre-stroke, right-side spouts drive every "
         "_left area harder and every _right area less, across six area pairs."),
    ):
        _hp = src / f"locanmf_rsa_hemisphere_{_hk}_{tag}.png"
        if not _hp.exists():
            _hp = src / f"locanmf_rsa_hemisphere_{_hk}.png"     # untagged, older runs
        if not _hp.exists():
            continue
        s = slide()
        title(s, f"RSA \u2014 {_htitle}", _hsub)
        note(s, M_RSA, specific=S_RSA_B)
        big(s, _hp, top=1.75, width=12.7)

    # ---------------- G. POST-STROKE ----------------
    # ORDER IS THE ARGUMENT, and it was chosen after getting it wrong once. Behaviour comes FIRST
    # because on 8/17 both animals stopped attempting the far positions, so any decoding number that
    # precedes that fact is uninterpretable -- the first pass reported a PS94 "neural deficit" whose
    # larger part was trial composition. Everything after G1 is position-matched.
    #
    # POOLS COME FROM config.phase_labels("post"), NEVER FROM A DATE. PS92/PS93 8/17 exists and is
    # projectable but belongs to neither phase (8/16 lesion, no deficit, redone after that session);
    # selecting by date would pool them silently. tests/test_stroke_phase.py pins this, and
    # tests/test_deck_section_g.py pins that this section obeys it.
    _post_labels = list(config.phase_labels("post"))
    _excluded = [f"{a}_0817" for a in animals if config.session_phase(a, "0817") == "excluded"]

    #: Place the SMALL-LESION (failed-laser) slides? Priya, 2026-09-13: no -- cut G2d (6 slides),
    #: G7, G7b and G7d from the narrative deck. ONE SWITCH FOR ALL FOUR, because they are one
    #: argument and a deck carrying two of them is worse than one carrying none or all.
    #:
    #: WHAT THE DECK LOSES, recorded because the case for showing them is still good and is argued
    #: at length at each site below: PS92/PS93 on 8/17 are neither pre- nor post-stroke -- lasered
    #: 8/16 with no deficit, re-lesioned after that session -- which makes them the strongest
    #: control this design has, a within-animal comparison one day before the effective lesion.
    #: After this cut the control survives in the deck only as two grey squares inside G2c's grid
    #: and as G7c in the per-session appendix: the deck ASSERTS the control rather than showing it.
    #: Every figure is still written nightly by `section_g_figures`; flip this to True to restore.
    _SHOW_SMALL_LESION = False
    if _post_labels and (src / "section_g_matched_all.png").exists():
        divider("G. POST-STROKE \u2014 the frozen pre-stroke model applied after the lesion",
                f"Lesion {config.stroke_cutoff()}; post-stroke pool = {', '.join(_post_labels)}. "
                f"Behaviour first: what the animal still attempts bounds what any decoding number "
                f"can mean.")

        # --- G0. the design, before any number
        s = slide()
        title(s, "G0. What is compared to what \u2014 and what cannot be compared",
              "Read this before the numbers. Four of these constraints changed a conclusion already.")
        note(s, M_POSTSTROKE, specific=S_G0)
        bullets(s, [
            ("PRE-STROKE reference is FROZEN: 11 curated dates ending 8/14, 44 sessions, every one "
            "resolving to phase=='pre'."),
            ("PRE keeps the ENGAGED cut (decode.max_rt_s); POST uses ALL trials \u2014 the missing "
            "licks ARE the phenotype, so filtering them out would delete the effect being measured. "
            "Declared by name in nolick_analysis.SANCTIONED_MISMATCHES."),
            "EVERY post-stroke slide is shown on BOTH ARMS. ALL trials scores all six positions, so chance is 1/6 for every session and the panels are comparable across sessions and animals. LICK-ONLY uses that session's own preserved positions, so its chance level MOVES with the behaviour (PS95: 4 positions on 8/17, 6 on 8/18) and its accuracies must NOT be laid side by side. Neither arm is comparable to the 6-way numbers in sections A\u2013F, which are engaged-only throughout.",
            "The DIFFERENCE between the arms is the point: it separates a code that degraded from a code that is fine whenever the animal manages to lick.",
            ("There is NO post-stroke 'disengaged' label. Engagement filtering post-stroke is RETIRED: "
            "a local dip in response rate cannot be distinguished from a run of motor failures, and "
            "in a severe stroke no spared reference position exists to anchor one."),
            ("'No lick detected' is NOT 'no tongue protrusion' \u2014 the spout needs contact. PS93 "
            "already shows this pre-stroke at far_L. Every no-lick conclusion is provisional on DLC."),
            f"POST-STROKE POOL: {', '.join(_post_labels)}. Every slide is per SESSION, never pooled across days — PS94's two nights differ more from each other than pre differs from post, so averaging them would destroy the effect.",
            "The day-1 plan/execution dissociation now REPLICATES in all four animals (G2c), so it is no longer a description of two animals. What remains n=1 is each animal's TRAJECTORY: one session per animal per day.",
        ])

        # --- G1. behaviour, from the nightly pipeline's own longitudinal figures
        # Iterate the POST-STROKE animals, not every animal: PS92/PS93 have no post-stroke figure by
        # design, and counting them through _exists reported two "missing figures" for files that
        # should not exist -- which makes the build's own missing-figure count useless as a check.
        _post_animals = sorted({l.split("_")[0] for l in _post_labels})
        for a in _post_animals:
            beh = src / f"poststroke_G1a_behaviour_{a}.png"
            if not _exists(beh):
                continue
            s = slide()
            title(s, f"G1. {a} \u2014 behaviour across sessions, lesion marked",
                  "Every per-position behavioural metric over that animal's sessions. DASHED RED = "
                  "the lesion; grey shading = a session excluded from both phases. This is the "
                  "figure the nightly behaviour pipeline already produces, not a bespoke plot.")
            note(s, M_POSTSTROKE, specific=S_G1)
            big(s, beh, top=1.5, width=12.9)
        # ONE counts slide: trial counts are trial counts, so this figure does not depend on the
        # arm. Rendering it per arm produced two byte-identical files under two titles (Priya,
        # 2026-08-19).
        # EVERY CHUNK, not just the first. fig_behaviour splits at COUNTS_PER_FIG sessions, so
        # this is a glob and not a filename: with 14 post-stroke sessions the single figure was
        # 12.50 x 1.10 in on the slide and unreadable (Priya, 2026-08-22). Sorted so _2 follows the
        # unsuffixed original.
        _counts = sorted(src.glob("section_g_counts*.png"),
                         key=lambda q: (len(q.stem), q.stem))
        for _i, _cf in enumerate(_counts):
            s = slide()
            _part = f" ({_i + 1} of {len(_counts)})" if len(_counts) > 1 else ""
            title(s, f"G1b. Which positions still have trials at all{_part}",
                  "Per-position engaged and no-lick counts, ONE PANEL PER POST-STROKE SESSION "
                  "against the pre-stroke per-session mean. A position with ZERO engaged trials "
                  "cannot have a lick-only decoding number at all; PS94 has two, and reading "
                  "that as a neural deficit is how the first pass went wrong.")
            note(s, M_POSTSTROKE, specific=S_G1B)
            big(s, _cf, top=1.6, width=12.5)

        # --- G2. position-matched decoding -- CUT 2026-09-13 (Priya).
        #
        # SUPERSEDED, NOT WITHDRAWN. G2 was the per-session form of the frozen-decoder result; H5c
        # and section I now carry it with bootstrap intervals and an epoch break G2 could not express, and
        # G2c below states the conclusion G2 was read for ("the PLAN survives, EXECUTION does not")
        # over all four animals in one panel. `section_g_matched_{all,lickonly}.png` is still
        # written every night, so restoring this block costs nothing but the indentation.
        for _arm, _armn in ():
            _mf = src / f"section_g_matched_{_arm}.png"
            if not _mf.exists():
                continue
            s = slide()
            title(s, f"G2. The FROZEN pre-stroke decoder after the lesion ({_armn} arm)",
                  "One panel per POST-STROKE SESSION. BAND = that animal's pre-stroke "
                  "leave-one-session-out range for the same measure. "
                  # THIS SENTENCE USED TO STOP AT "BY " -- an unfinished clause, shipped on the
                  # slide, promising a reason it never gave (found 2026-08-24 while building the
                  # grant figures, which hit the same absence in the data and had to work out why).
                  + ("All six positions, chance 1/6 on every panel. POST-LICK IS ABSENT HERE BY "
                     "CONSTRUCTION: this arm includes trials with NO detected lick, and a "
                     "lick-aligned window cannot be defined for a trial that has no lick. At the "
                     "impaired positions that is most of the trials, which is exactly where the "
                     "question is. The lick window appears in G9, where the no-lick classes are "
                     "placed at an INFERRED would-be-lick time and labelled as inference."
                     if _arm == "all" else
                     "Each session on ITS OWN preserved positions, so the chance line differs "
                     "between panels and the accuracies are NOT comparable across them."))
            note(s, M_POSTSTROKE, specific=S_G2)
            big(s, _mf, top=1.7, width=12.3)
        # G2b now comes from the unified runner: per_position_table derives it from the confusion
        # DIAGONALS in section_g.json, which are the per-position recall table by construction. The
        # superseded version was built from per_position_pre_vs_post_0817.json -- day 1 only, so
        # only the two animals whose lesion took on 8/17, and frozen at its 8/18 content because no
        # step rewrote it (Priya, 2026-08-20).
        if (src / "section_g_G2b_per_position.png").exists():
            s = slide()
            title(s, "G2b. Per-position recall in all four conditions \u2014 all four animals, every "
                     "post-stroke day",
                  "post-cue, post-lick, pre-cue WITH lick, pre-cue NO lick \u2014 the pre-stroke bar "
                  "then ONE BAR PER POST-STROKE DAY at every position. 'With/without lick' is the "
                  "RESPONSE lick, i.e. engaged vs no-lick trials; the no-lick condition pairs "
                  "PRE-stroke no-lick against POST-stroke no-lick, so it differs in phase alone. "
                  "'n/a' means the position was never attempted, which is not zero recall; RED "
                  "HATCHED means fewer than 10 trials.")
            note(s, M_POSTSTROKE, specific=S_G2B)
            big(s, src / "section_g_G2b_per_position.png", top=1.75, width=12.3)

        # --- G2c. recoding vs loss: the test that reframes G2
        # Reads the WITHCONTROL grid (post + excluded rendered TOGETHER). section_g_grid_all.png
        # holds only the 10 post sessions and section_g_smalllesion_grid_* only the 2 excluded ones,
        # so neither carries the grey-square before/after pairing this slide argues from. Until
        # 2026-08-20 this read poststroke_grid.png, a scratchpad-era file that no step rewrote after
        # the section-G consolidation -- so the headline four-animal slide silently kept its
        # 8/19 10:17 content and never showed 8/19 itself (Priya, 2026-08-20).
        _rf = src / "section_g_grid_withcontrol_all.png"
        if _rf.exists():
            s = slide()
            title(s, "G2c. After an effective lesion: the PLAN survives, EXECUTION does not "
                     "\u2014 in all four animals, every post-stroke day",
                  "Pre-cue and post-cue are two windows on the SAME trials, so every session-level "
                  "confound acts on both equally and cannot produce a difference between them. GREY "
                  "SQUARES = PS92/PS93 on 8/17 after the laser that did NOT take: nothing outside the "
                  "band, then the dissociation appears one day later after the effective lesion \u2014 "
                  "a within-animal before/after control. PURPLE = outside the band but ABOVE it.")
            note(s, M_RECODING, specific=S_G2C)
            big(s, _rf, top=1.85, width=11.4)

        # THE LICK-ONLY ARM OF THE SAME GRID. Written every night since the section-G consolidation
        # and never placed, while the all-trials arm above carried the headline alone.
        #
        # It is not decoration: the standing objection to G2c is that the post-stroke ALL-trials arm
        # includes trials with no lick, so a reader can say the dissociation is about MISSING
        # MOVEMENTS rather than about coding. This arm scores only trials the animal licked on, which
        # removes that confound -- at the cost of the abandoned positions, which have no lick trials
        # to score. The two arms are therefore complementary and neither is complete: all-trials
        # keeps every position and admits the confound, lick-only removes the confound and loses
        # positions. Showing one without the other lets a reader assume the choice was neutral.
        _rf2 = src / "section_g_grid_withcontrol_lickonly.png"
        if _rf2.exists():
            s = slide()
            title(s, "G2c (LICK-ONLY arm). The same grid, scored only on trials the animal "
                     "actually licked on",
                  "THE CONTROL FOR THE OBVIOUS OBJECTION to the slide before this one: that the "
                  "all-trials dissociation is about missing MOVEMENTS rather than about coding. "
                  "Here every scored trial has a lick, so it cannot be. THE COST is positions: an "
                  "abandoned position has no lick trials, so it drops out entirely and the chance "
                  "level moves with it (4-way rather than 6-way for PS94 and PS95 on day 1). Read "
                  "the two arms TOGETHER \u2014 all-trials keeps every position and admits the "
                  "confound, lick-only removes the confound and loses positions. Neither is the "
                  "complete figure, and choosing one silently would be choosing a result.")
            note(s, M_RECODING, specific=S_G2C)
            big(s, _rf2, top=1.85, width=11.4)

        # --- G2d. THE SMALL-LESION FAMILY, in full.
        #
        # `section_g_figures` renders the whole readout family TWICE -- once for the post-stroke
        # sessions and once for the 'excluded' ones under `section_g_smalllesion_*` -- and nine of
        # those figures had never reached a slide. Only the two grey squares inside G2c's grid
        # represented them, which is the summary of this family rather than the family.
        #
        # WHAT THEY ARE, and the label has to be exact: PS92 and PS93 on 8/17, after a laser that
        # did NOT take. They are neither pre-stroke nor post-stroke -- `session_phase` returns
        # 'excluded' -- and `config.pooled_labels` keeps them out of every pooled result by
        # construction. They are shown here and NOWHERE ELSE, because a within-animal control one
        # day before the effective lesion is the strongest control this design has, and because a
        # figure that exists and is never shown is a figure nobody has checked.
        # NO CONFUSION ENTRIES HERE, and their absence is the fix rather than an omission. This
        # tuple used to carry ("confusion_precue", ...) and ("confusion_cue", ...), which resolve to
        # `section_g_smalllesion_confusion_<align>_<arm>.png` -- POOLED filenames that
        # `plot_poststroke.fig_confusion_alltrials` stopped writing on 2026-08-19, when it went one
        # figure per session. The four files it kept placing were the last ones written under the
        # old name, so G2d showed 2026-08-19 content for three weeks while every other figure on
        # the slide beside it was current, and the deck's own staleness manifest was the only place
        # that said so. The per-session versions ARE shown, on their own slides, at G7c.
        _SMALL = (("grid", "the four-condition grid"),
                  ("similarity", "pattern similarity to the pre-stroke reference"),
                  ("matched", "position-matched frozen decoding"))
        # CUT 2026-09-13 (Priya), together with G7/G7b/G7d below: six slides of small-lesion detail
        # collapse to the two grey squares in G2c's grid plus the per-session G7c in the appendix.
        # WORTH KNOWING WHAT THAT COSTS, since the argument for showing them is still on record
        # above: after this cut the deck asserts the failed-laser control rather than displaying it,
        # and the six figures remain on disk, written nightly, shown nowhere. Restore by putting
        # `_SMALL` back in the comprehension.
        _small_found = [(k, nice, arm, armn, q)
                        for k, nice in (_SMALL if _SHOW_SMALL_LESION else ())
                        for arm, armn in (("all", "ALL trials"), ("lickonly", "LICK-ONLY"))
                        if (q := src / f"section_g_smalllesion_{k}_{arm}.png").exists()]
        if _small_found:
            divider("G2d - SMALL-LESION COMPARISON (the laser did not take)",
                    "PS92 and PS93 on 8/17. Neither pre-stroke nor post-stroke, never pooled with "
                    "either, and the within-animal control for everything in G2: the same animals, "
                    "the same rig, one day before the lesion that did take.")
            for _k, _nice, _arm, _armn, _q in _small_found:
                s = slide()
                title(s, f"G2d. Small lesion \u2014 {_nice} ({_armn})",
                      "THE SAME ANALYSIS AS THE POST-STROKE SLIDES ABOVE, on the sessions where the "
                      "laser did not take. What it should show is NOTHING: no dissociation, values "
                      "inside the pre-stroke band. That is what makes the effective-lesion result "
                      "one day later a lesion effect rather than a day-to-day effect, a handling "
                      "effect, or an anaesthesia effect. THESE SESSIONS ARE NEVER POOLED with either "
                      "phase \u2014 `session_phase` returns 'excluded' and `config.pooled_labels` "
                      "drops them by construction, so nothing on any other slide contains them.")
                note(s, M_RECODING, specific=S_G2C)
                big(s, _q, top=1.85, width=11.4)

        # --- G3. crossed confusion: WHERE the errors go
        # G3. Crossed confusion, per POST-STROKE SESSION and on BOTH arms.
        #
        # The engaged-only 6x6 that used to sit here was built from a day-1-only JSON on a pooled
        # position basis, and it left the abandoned positions BLANK -- which are the rows worth
        # reading. It is superseded on every axis by the figures below (both normalisations,
        # precision annotated, no-lick rows filled) and is not shown beside them, because two
        # confusion figures that disagree invite the reader to pick.
        for _al, _nice in (("precue", "PRE-cue"), ("cue", "POST-cue")):
            for _arm, _armn in (("all", "ALL trials"), ("lickonly", "LICK-ONLY")):
                # FOUR SESSIONS PER SLIDE (Priya, 2026-08-28: "There are too many slides and I think
                # we should consolidate"). One slide per session gave 2 alignments x 2 arms x ~28
                # animal-days = over a hundred slides in G3 alone, and a reader comparing days had
                # to page between them. A 2x2 grid puts a whole animal's post-stroke progression --
                # or four consecutive days of it -- in one visual field, which is the comparison the
                # figure exists for.
                #
                # GROUPED BY ANIMAL FIRST, then by date, so a slide never straddles two animals: the
                # earlier bug here was a title naming only the DATE, which made PS92_0818 and
                # PS93_0818 two slides with identical headings (Priya, 2026-08-20: "what is the
                # difference between slide 136 and 138"). Grouping by animal makes that
                # unrepresentable rather than merely fixed.
                _files = sorted(src.glob(f"section_g_confusion_{_al}_{_arm}_*.png"))
                _by_an: dict[str, list] = {}
                for _f in _files:
                    _by_an.setdefault(_f.stem.split("_")[-2], []).append(_f)
                for _an, _fs in sorted(_by_an.items()):
                    for _i in range(0, len(_fs), 4):
                        _chunk = _fs[_i:_i + 4]
                        _days = ", ".join(p.stem.split("_")[-1] for p in _chunk)
                        s = slide()
                        title(s, f"G3. {_nice} crossed confusion — {_an} ({_armn} arm): {_days}",
                              "Rows = TRUE position. PANEL 2 OF EACH FIGURE IS THE MATCHED CONTROL: "
                              "pre-stroke NO-LICK trials, scored by a decoder trained on the OTHER "
                              "pre-stroke sessions' engaged trials, so it differs from the post "
                              "panel in PHASE alone rather than in phase and the absence of a "
                              "movement together (Priya, 2026-08-19). On the ALL arm the abandoned "
                              "positions are filled by no-lick trials, the only evidence that "
                              "exists there. '(pred)' under each column is how often the decoder "
                              "picks that position at all, which IS the recall expected under a "
                              "label permutation; '(prec)' is precision. Read the OFF-diagonal.")
                        note(s, M_POSTSTROKE, specific=S_G3)
                        grid(s, _chunk, cols=2, top=1.95)

        # --- G4. identity, with its control read first
        if (src / "poststroke_G4_identity.png").exists():
            s = slide()
            title(s, "G4. Do post-stroke NO-LICK trials look like pre-stroke LICKING trials?",
                  "Discriminator trained on pre-stroke engaged-vs-no-lick, POSITION-BALANCED so it "
                  "cannot simply answer 'far'. READ THE CONTROL FIRST: post-stroke ENGAGED trials "
                  "must sit above post-stroke no-lick, or the boundary is tracking 'post-stroke' "
                  "rather than licking and the answer means nothing.")
            note(s, M_POSTSTROKE, specific=S_G4)
            big(s, src / "poststroke_G4_identity.png", top=1.7, width=11.5)

        # --- G4b. does the post-stroke session fit the PRE-stroke ENGAGED distribution?
        # Replaces G4's control for Priya's hypothesis. G4 asks whether post-stroke engaged and
        # no-lick trials still SEPARATE and treats failure to separate as a broken boundary -- but
        # execution-failure predicts they should NOT separate, so that control can disqualify the very
        # result it exists to license. This one places the post-stroke value against reference
        # distributions built from PRE-stroke sessions, where the answer is known.
        for _al, _nice in (("precue", "PRE-cue"), ("cue", "POST-cue")):
            # section_g prefix: the runner stored only the PRE-CUE record until 2026-08-20,
            # so the cue-aligned slide had no producer and sat on a scratchpad figure.
            _ff = src / f"section_g_fits_engaged_{_al}.png"
            if not _ff.exists():
                continue
            s = slide()
            title(s, f"G4b. Do post-stroke NO-LICK trials fit the pre-stroke ENGAGED distribution? "
                     f"({_nice})",
                  "Each dot is a SESSION, held out from the discriminator that scored it, so the "
                  "spread of the dots IS the confidence interval \u2014 and it is the right one, "
                  "because sessions differ from one another far more than trials within a session. "
                  "Makes NO assumption that the two post-stroke classes should differ.")
            note(s, M_POSTSTROKE, specific=S_G4B)
            big(s, _ff, top=1.85, width=11.0)

        # --- G5. same code weaker, or a different code?
        for _arm, _armn in (("all", "ALL trials"), ("lickonly", "LICK-ONLY")):
            _sf5 = src / f"section_g_similarity_{_arm}.png"
            if not _sf5.exists():
                continue
            s = slide()
            title(s, f"G5. Same code at reduced strength, or a different code? ({_armn} arm)",
                  "Per-position correlation between the pre- and post-stroke mean activity "
                  "patterns, one series per post-stroke session. Decoding accuracy alone cannot "
                  "separate a weakened code from a reorganised one; this can. G8f asks the same "
                  "question of the whole 6x6 geometry, and adds the midline test.")
            note(s, M_POSTSTROKE, specific=S_G5)
            big(s, _sf5, top=1.7, width=11.8)

        # --- G6. was a plan formed on the no-lick trials?
        if (src / "poststroke_G6_nolick_readout.png").exists():
            s = slide()
            title(s, "G6. Was a plan formed on the trials with no lick? Impaired vs preserved "
                     "positions",
                  "REPLACES the working-vs-disengaged split, retired 2026-08-18 because 'disengaged' "
                  "has no valid post-stroke construction. This splits on the TRUE spout position, "
                  "which is measured rather than inferred. Above the black null at IMPAIRED "
                  "positions = position represented, movement did not happen.")
            note(s, M_POSTSTROKE, specific=S_G6)
            big(s, src / "poststroke_G6_nolick_readout.png", top=1.75, width=11.0)

        # --- G6b. the miss/stopped contrast, per position and per session.
        if (src / "poststroke_miss_vs_stopped.png").exists():
            s = slide()
            title(s, "G6b. Is the plan there when the animal is TRYING? Miss-while-working vs "
                     "stopped",
                  "Same position, same session, the two post-stroke failure modes side by "
                  "side. 1.0 = that position's own pre-stroke pole. Miss above zero with "
                  "stopped AT zero is plan-intact / execution-failed. far_L is the control: "
                  "the effect is absent there.")
            note(s, M_MISS_STOPPED)
            big(s, src / "poststroke_miss_vs_stopped.png", top=1.7, width=12.4)

        # --- G7. SMALL-LESION COMPARISON: the excluded sessions.
        # NOT a negative control -- PS92/PS93 were lesioned too, just mildly (Priya,
        # 2026-08-18). They control for the DAY and give a severity contrast; they cannot
        # show that a lesion is necessary for an effect.
        # PS92/PS93 8/17 belongs to neither phase, which is exactly what makes it the control. These
        # slides are built from an EXPLICIT label list (poststroke_compare._pooled(post_labels=...)),
        # never from phase_labels("post"), and their JSON carries excluded_from_pooled_summaries.
        # CUT 2026-09-13 (Priya), with G2d above and G7d below. The three small-lesion slides in the
        # narrative deck go; G7c (per session) stays in the appendix and the two grey squares in
        # G2c's grid stay in the narrative. SAME CAVEAT AS G2d: the failed-laser control is then
        # asserted in prose rather than shown, and the argument for showing it -- recorded three
        # comment blocks up, and unchanged -- is that a within-animal control one day before the
        # effective lesion is the strongest control this design has. Flip the `False` to restore.
        if _SHOW_SMALL_LESION and _excluded and (src / "section_g_smalllesion_matched_all.png").exists():
            s = slide()
            title(s, "G7. SMALL-LESION COMPARISON \u2014 the two animals without an overt deficit",
                  f"{', '.join(_excluded)}: lesioned 8/16, no behavioural deficit, re-lesioned AFTER "
                  f"this session. Same day, same anaesthesia, same handling, same frozen decoder. If "
                  f"these two also dropped, the G2\u2013G6 effects would be the DAY, not the lesion.")
            note(s, M_POSTSTROKE, specific=S_G7)
            big(s, src / "section_g_smalllesion_matched_all.png", top=1.75, width=12.3)
            if (src / "section_g_smalllesion_counts.png").exists():
                s = slide()
                title(s, "G7b. SMALL-LESION behaviour \u2014 all six positions still attempted",
                      "Against G1b, where PS94 has ZERO engaged trials at far_center and far_R. The "
                      "behavioural collapse is specific to the animals whose lesion took, which is "
                      "what makes the decoding comparison interpretable at all.")
                note(s, M_POSTSTROKE, specific=S_G7B)
                big(s, src / "section_g_smalllesion_counts.png", top=1.6, width=12.5)

        # G7c: the same all-trials matrix as G3b, for the control animals. At 8 spaces, NOT 12 --
        # it was nested inside the G9 loop and rendered twice (slides 140-141 duplicated 137-138),
        # and it rebound that loop's own `_f`. Loop variable renamed so it cannot shadow again.
        # ONE SLIDE PER SESSION, as G3 does: the runner emits per-session confusion figures, and
        # the single stacked file this used to read is a scratchpad-era orphan nothing rewrites.
        # The LICK alignment exists on the lick-only arm alone (a no-lick trial has no lick to
        # align to), so the all-trials glob simply finds nothing for it.
        for _al, _nice in (("precue", "PRE-cue"), ("cue", "POST-cue"), ("lick", "POST-lick")):
            for _arm, _armn in (("all", "ALL trials"), ("lickonly", "LICK-ONLY")):
                for _cf in sorted(src.glob(f"section_g_smalllesion_confusion_{_al}_{_arm}_*.png")):
                    _lab = "_".join(_cf.stem.split("_")[-2:])
                    s = slide()
                    title(s, f"G7c. SMALL-LESION COMPARISON \u2014 {_nice} confusion, {_lab} "
                             f"({_armn} arm)",
                          "The same matrix as G3, for the two animals whose strokes were small "
                          "enough to leave no overt deficit. Near-diagonal, with prediction rates "
                          "of 0.09-0.21 (uniform is 0.167) \u2014 NO systematic pull toward any "
                          "position. That is what makes PS94's far_R over-prediction (0.35 of all "
                          "its post-stroke trials) a lesion effect rather than a property of the "
                          "frozen decoder or of 8/17.")
                    note(s, M_POSTSTROKE, specific=S_G7C)
                    big(s, _cf, top=1.85, width=9.6)

        # G7d: the same fits-engaged test on the SMALL-LESION animals. ONE slide, both alignments
        # side by side (Priya, 2026-08-20).
        #
        # THIS USED TO READ A PAIR OF HAND-MADE 2026-08-18 FILES, frozen on the grounds that "this
        # comparison is permanently PS92/PS93 on 8/17, so its content cannot change". The SESSIONS
        # cannot change; the ANALYSIS did. `fits_engaged` is computed on ENGAGED trials and the
        # engagement gate was replaced twice after those files were drawn -- position-blind, then
        # reference-restricted and backdated (880e6bd) -- so this slide was showing the only figures
        # in section G still built on a retired gate, and saying so nowhere. `section_g_figures`
        # now renders them with everything else; the legacy name is kept as a fallback only so a
        # deck built before that step has run does not lose the slide.
        _g7d = [src / f"section_g_smalllesion_fits_engaged_{_al}.png"
                for _al in ("precue", "cue")]
        if not any(q.exists() for q in _g7d):
            _g7d = [src / f"poststroke_G7d_smalllesion_fits_engaged_{_al}.png"
                    for _al in ("precue", "cue")]
        # CUT 2026-09-13 (Priya) -- the last of the four small-lesion slides (G2d, G7, G7b, G7d).
        _g7d = [q for q in _g7d if q.exists()]
        if _SHOW_SMALL_LESION and _g7d:
            s = slide()
            title(s, "G7d. FAILED-LASER CONTROL — do the no-lick trials fit the ENGAGED "
                     "distribution?  LEFT: PRE-cue.  RIGHT: POST-cue.",
                  "The same test as G4b on PS92/PS93 8/17 — after the 8/16 laser that did NOT take "
                  "and before the effective 8/17 stroke, so these animals were un-lesioned here. "
                  "That is what it controls for: whether a no-lick trial fails the fits-engaged test "
                  "just by being a no-lick trial. NOT the small-lesion arm, which is these same "
                  "animals from 8/18 onward. PS92 has too few no-lick trials to test (it responded "
                  "on essentially every trial), so in practice this is PS93 alone. The two "
                  "SESSIONS are all there will ever be, but the figure is REGENERATED with the "
                  "rest of section G rather than frozen: it is scored on ENGAGED trials, so it "
                  "moves when the engagement gate does.")
            note(s, M_POSTSTROKE, specific=S_G7D)
            grid(s, _g7d, cols=2, top=1.9)

        # --- G9. PER-POSITION CODING DIRECTIONS (Priya, 2026-08-20/21)
        #
        # Each spout position gets a direction fitted on PRE-STROKE trials WITH A SUCCESSFUL LICK --
        # that position against the others -- and every class is projected onto it WITHIN that
        # position, so the classes' very different position composition cannot contribute.
        #
        # WHICH VARIANT IS SHOWN, AND WHY IT IS NOT THE PLAIN ONE. The plain difference-of-means
        # direction is badly contaminated by the lick/no-lick axis: cos(w, engagement) reaches 0.82,
        # 0.91, 0.71 and 0.52 in PS92/93/94/95, and it lands on a DIFFERENT position in each animal
        # (far_center, far_center, far_L, far_R), so it cannot be inspected around. PS93's
        # far_center direction is 91% engagement axis wearing a position label. ENL and cue
        # therefore show the ORTHOGONALISED direction, with that axis projected out; after removal
        # the pre-stroke no-lick trials collapse to one consistent value on every axis (0.16-0.17
        # for PS94) instead of scattering from -2.03 to +1.38, and the pre-stroke lick diagonal
        # IMPROVES rather than degrading.
        #
        # THE LICK WINDOW USED TO KEEP THE PLAIN DIRECTION, on the grounds that "its only classes
        # are lick trials on both sides, so the engagement axis cannot contaminate the comparison --
        # and no no-lick trials exist there to build one from". Both halves stopped being true on
        # 2026-08-21, when the would-be-lick reference gave a no-lick trial a window at the cue plus
        # its position's median RT: all five classes are now in this window, so an engagement axis
        # both exists and matters. Here it IS a licking axis -- movement present against absent --
        # so orthogonalising asks what position structure survives once movement PRESENCE is
        # removed. Right for the no-lick classes, deliberately conservative for the lick ones, since
        # licks to different spouts differ in kinematics and no projection can separate position
        # from position-specific movement in this window.
        #
        # AUDITED 2026-08-24 (`scripts/coding_direction_audit.py`), against the LOGISTIC directions
        # that were kept from the start as the independent check -- they reach a near-uncontaminated
        # direction WITHOUT any projection, by accounting for covariance. Median |dom - lr| ->
        # |dom_orth - lr|:
        #
        #            ENL                 cue                 lick
        #   PS92     0.841 -> 0.128      0.160 -> 0.192      0.202 -> 0.279     <-- AWAY, twice
        #   PS93     0.377 -> 0.127      0.279 -> 0.206      0.219 -> 0.135
        #   PS94     0.155 -> 0.122      0.254 -> 0.140      0.181 -> 0.112
        #   PS95     0.429 -> 0.188      0.569 -> 0.244      0.499 -> 0.191
        #            4/4 toward lr       3/4                 3/4
        #
        # ENL IS SETTLED: every animal moves toward the reference, and projecting an ALREADY-CLEAN
        # lr direction costs only 0.014-0.072 there -- the engagement axis carries almost no
        # position structure in a window with no movement in it.
        #
        # IN CUE AND LICK IT COSTS MORE (lr_orth vs lr 0.024-0.116), because there the axis is a
        # LICKING axis and removing it takes position-linked movement with it. For PS92 that
        # tips the balance: orthogonalising moves it AWAY from lr in both windows, correlation
        # +0.77 -> +0.69 and +0.78 -> +0.62. Its plain directions were already the cleanest of the
        # four in those windows (0.160/0.202 against 0.841 in ENL), so it had little contamination
        # to remove and real structure to lose. "Deliberately conservative" was the right instinct
        # for the lick classes; for PS92 specifically it is not conservative but wrong-signed, and
        # THIS SLIDE SHOWS THE WORSE OF THE TWO ESTIMATES FOR PS92 cue AND PS92 lick. Read those two
        # panels against the plain-direction ones, or against lr directly.
        #
        # RESOLVED (Priya, 2026-08-24): cue and lick show BOTH variants, ENL shows only the
        # orthogonalised one. Not switching PER ANIMAL -- six panels built by different rules are
        # incommensurable with each other -- but showing both everywhere the choice is CONTESTED
        # costs nothing (both figure sets are already rendered by the default
        # `--methods dom dom_orth`) and lets the reader see the disagreement instead of taking my
        # word for its size. ENL stays single because there the audit is 4/4 and the plain
        # direction is badly contaminated (|dom - lr| 0.841 in PS92) -- showing it would invite the
        # misreading the orthogonalisation exists to prevent.
        _G9_METHODS = {"ENL": ("dom_orth",), "cue": ("dom_orth", "dom"), "lick": ("dom_orth", "dom")}
        #: G9 kinds narrow enough to place TWO ANIMALS per slide. Measured widths:
        #: pooled 11.5in, normunit 12.0in -> ~6.2in each at 2-up, labels still ~6pt.
        #: The dense kinds are 16-23in and are deliberately absent.
        _G9_PACK_2UP = {"pooled", "normunit"}
        #: how to read a pair of slides that disagree, by window
        _G9_PAIR_NOTE = {
            "cue": ("  BOTH VARIANTS ARE SHOWN for this window. Audited 2026-08-24 against the "
                    "logistic directions: orthogonalising moves the estimate TOWARD that reference "
                    "in PS93/PS94/PS95 (0.279->0.206, 0.254->0.140, 0.569->0.244) and AWAY in PS92 "
                    "(0.160->0.192). Prefer ORTH except in PS92, where the plain direction is the "
                    "better estimate here."),
            "lick": ("  BOTH VARIANTS ARE SHOWN for this window. Audited 2026-08-24: orthogonalising "
                     "moves the estimate TOWARD the logistic reference in PS93/PS94/PS95 "
                     "(0.219->0.135, 0.181->0.112, 0.499->0.191) and AWAY in PS92 (0.202->0.279). "
                     "Prefer ORTH except in PS92. After the cue the engagement axis is a LICKING "
                     "axis, so removing it also removes position-linked movement -- which is why "
                     "this window is the contested one and ENL is not."),
        }
        for _w in ("ENL", "cue", "lick"):
            for _kind, _tag, _blurb in (
                ("direction", "time course",
                 ("One panel per spout position, MOST IMPAIRED first, every class over sessions with "
                 "the stroke marked. LINEAR projection, pole-normalised: 0 = pre-stroke "
                 "NOT-this-position, 1 = pre-stroke LICK here. Error bars are SEM over trials; a "
                 "HOLLOW marker means fewer than 10 trials, shown rather than dropped.")),
                ("pooled", "pooled over sessions",
                 ("The same classes collapsed across every session of a phase, so each position is "
                 "one point per class. Read it BESIDE the time course: pooling hides whether a "
                 "class was steady or swinging, and a post-stroke class that moved a lot looks "
                 "identical here to one that never did.")),
                ("within", "over the COURSE of a session",
                 ("Trials binned by where they fall within their OWN session, pooled across the "
                 "sessions of a phase, so a state that drifts as the animal tires shows here and "
                 "cannot show in a session-level split. A cell is drawn only if its own SEM is "
                 "under 0.25 -- a quarter of the pole separation -- because a 4-trial point at "
                 "+-2 dominates the eye and invents a shape. WARNING: do NOT read 1.0 as a flat "
                 "baseline. Pre-stroke LICK itself declines across the session at the CLOSE "
                 "positions (PS94 close_center 1.39 -> 1.02 -> 0.80 -> 0.67, close_L 1.33 -> 0.71, "
                 "SEM 0.03-0.09 on 183-290 trials per bin) while staying flat at the far ones, so "
                 "a within-session comparison has to be read against the pre-stroke profile AT THE "
                 "SAME POSITION, not against the poles. THAT DECLINE IS DISENGAGEMENT: it appears "
                 "in PS94 and PS95, which lose 0.25 and 0.35 of their pre-stroke response rate by "
                 "the last quartile, and NOT in PS92/PS93, which lose 0.09 and 0.06. RT stays flat "
                 "throughout (0.13 s in every quartile at close positions), so these are trials "
                 "SKIPPED, not slowed -- the sated tail. But the behavioural drop is UNIFORM across "
                 "positions (PS95 -0.31 to -0.39 at all six) while the neural decline is not, so it "
                 "cannot be read position by position: a close-position one-vs-rest axis is largely "
                 "a close-vs-far contrast, and a uniform state shift along that dimension loads on "
                 "it asymmetrically.")),
                ("cross", "cross-position matrix",
                 ("Rows = TRUE spout position, columns = which position's direction it was scored "
                 "on. Panel 1 is the PRE-STROKE baseline, because neighbouring positions are "
                 "intrinsically similar before any stroke; the rest are DIFFERENCES from it, so a "
                 "row going red OFF the diagonal is a remapping rather than a large number.")),
                ("engagement", "BEHAVIOUR: response rate over the session",
                 ("THE FIGURE THE WITHIN-SESSION PANEL MUST BE READ AGAINST. Response rate per "
                 "position, binned by where a trial falls within its OWN session, pooled over the "
                 "sessions of a phase. Reward is auto-held after a miss run, so a terminal collapse "
                 "here is DISENGAGEMENT rather than spatial inaccuracy. Pre-stroke, PS94 and PS95 "
                 "lose 0.25 and 0.35 of their responding by the last quartile while PS92 and PS93 "
                 "lose 0.09 and 0.06 -- and the two that disengage are exactly the two whose neural "
                 "projection drifts. The drop is UNIFORM across positions (PS95 -0.31 to -0.39 at "
                 "all six), so it cannot explain a decline that appears at only some of them; see "
                 "the cos-vs-drift slide for what does. NOT method-dependent: one per animal.")),
                ("normunit", "direction or magnitude?",
                 ("DOES THE POST-STROKE VALUE MEAN THE PATTERN CHANGED, OR JUST GOT BIGGER? The "
                 "projection x\u00b7w rises either because the trial points more along the "
                 "direction (position structure) or because it sits further from its session's "
                 "engaged centroid (everything else) -- and correlating the two CANNOT separate "
                 "them, since a trial moving further out ALONG the direction raises both. LEFT: the "
                 "same post-stroke LICK value scored raw and on UNIT-NORMALISED trials, cos(x,w), "
                 "which is blind to magnitude. Bars that agree = directional. RIGHT: each position "
                 "against its post/pre norm ratio; pure gain would put it on the dashed line. "
                 "Measured 2026-08-22: every cell moves by at most 0.15 except PS92 far_center "
                 "(2.12 -> 1.75), so post-stroke values ABOVE 1.0 are real.")),
                ("pairwise", "pairwise axes — ONE PANEL PER POSITION",
                 # SAY WHAT A PANEL IS. This blurb described the CONTRAST and never the LAYOUT,
                 # while the `direction` blurb next to it opens "One panel per spout position" --
                 # so the one figure that is already split per position read as though it was not
                 # (Priya, 2026-08-24: "why isn't each position on its own graph"). The x-tick
                 # labels being position names makes the misreading the natural one.
                 ("ONE PANEL PER POSITION, and the panel is the trials' TRUE position: the "
                  "top-left panel is far_R trials only. The X-AXIS INSIDE A PANEL is the PARTNER "
                  "position the axis contrasts against, with that pair's pre-stroke separation in "
                  "brackets — so you read 'far_R trials, how far toward far_center / far_L / "
                  "close_R / …'. "
                  "THERE IS NO SELF COLUMN, and that is deliberate: an axis is a contrast between "
                  "TWO positions, so no far_R-vs-far_R axis exists. The panel's own position is "
                  "the SCALE instead — 1 = pre-stroke lick at it, 0 = pre-stroke lick at the "
                  "partner — so the flat line at 1.0 IS far_R, and every class is read as its "
                  "distance from it. For a single 'how far_R-like is far_R' number, use the "
                  "one-vs-rest panels (the time-course slides, and the diagonal of the "
                  "cross-position matrix); dropping a one-vs-rest value into this panel would put "
                  "two different axis constructions on one line. "
                  "Each contrast is A vs B ALONE. Sharper than one-vs-rest for remapping: 'not P' "
                 "mixes five positions and, for the MIDDLE positions, is majority-far -- PS94's "
                 "close_center axis orders close_L 1.23 > close_R 0.83 > close_center 0.71, i.e. "
                 "the position it is named for is only third on its own axis. READ THE WITHIN-RING "
                 "CELLS (close-vs-close, far-vs-far) FIRST: they carry half the close-vs-far "
                 "loading (|cos| 0.33 vs 0.70) and no coherent within-session drift, while every "
                 "one of the 18 cross-ring cells in the two disengaging animals drifts the same "
                 "way (mean +0.19, the far position becoming more far-like over the session).")),
            ):
                # THE BEHAVIOUR PANEL IS NOT METHOD-DEPENDENT and its file carries no method in the
                # name (`coding_engagement_<window>_<animal>.png`). The loop used to build every
                # name with the method in it, so that file never matched and TWELVE BEHAVIOUR
                # SLIDES WERE SILENTLY ABSENT from the deck -- while the within-session note called
                # this "THE FIGURE THE WITHIN-SESSION PANEL MUST BE READ AGAINST". Found 2026-08-24
                # while wiring the two variants; `_f.exists()` skips are invisible by design, which
                # is what let it sit.
                _methods = ("",) if _kind == "engagement" else _G9_METHODS[_w]
                for _m in _methods:
                    _found = []
                    for _an in sorted({s_["label"][:4] for s_ in config.load_sessions()}):
                        _f = src / (f"coding_{_kind}_{_w}_{_an}.png" if not _m else
                                    f"coding_{_kind}_{_w}_{_m}_{_an}.png")
                        if _f.exists():
                            _found.append((_an, _f))
                    # TWO ANIMALS PER SLIDE where the figure is narrow enough to survive it.
                    # MEASURED, not assumed: a figure placed 2-up sits ~6.2in wide, so a label
                    # reaches the reader at fontsize x 6.2/figure_width. `pooled` (11.5in) and
                    # `normunit` (12.0in) keep 9pt labels at ~5in-equivalent and stay readable; the
                    # dense kinds in this same loop are 16-23in and would land at 3-4pt, which is
                    # why they are NOT in this set. See DECISIONS 2026-08-28.
                    if _kind in _G9_PACK_2UP and len(_found) > 1:
                        # THE CAVEATS TRAVEL WITH THE SLIDE. The one-per-animal path below appends
                        # the lick-window inference note and the plain-vs-orth pair note to the
                        # blurb; a packed slide showing the same figures must carry the same
                        # qualifications, or consolidating the deck would quietly strip the reasons
                        # its numbers are conditional.
                        _lick2 = ("" if _w != "lick" else
                                  "  The no-lick classes sit at an INFERRED time here: a no-lick "
                                  "trial has no lick to align to, so its window starts at the cue "
                                  "plus that session's own median RT at that position. A position "
                                  "with NO engaged trial that session is DROPPED rather than given "
                                  "the session median — read those classes as inference.")
                        _pair2 = _G9_PAIR_NOTE.get(_w, "") if len(_methods) > 1 else ""
                        _mlabel2 = ("" if not _m else
                                    f", {'ORTHOGONALISED' if _m.endswith('_orth') else 'PLAIN'} "
                                    f"direction")
                        for _i in range(0, len(_found), 2):
                            _two = _found[_i:_i + 2]
                            s = slide()
                            title(s, f"G9. {' & '.join(a for a, _ in _two)} — {_w} window, "
                                     f"{_tag}{_mlabel2}", _blurb + _lick2 + _pair2)
                            note(s, M_CODING_DIR, specific=S_G9)
                            grid(s, [q for _, q in _two], cols=2, top=1.95)
                        continue
                    for _an, _f in _found:
                        s = slide()
                        _lickonly = ("" if _w != "lick" else
                                     "  The no-lick classes sit at an INFERRED time here: a no-lick "
                                     "trial has no lick to align to, so its window starts at the cue "
                                     "plus that session's own median RT at that position. A position "
                                     "with NO engaged trial that session is DROPPED rather than given "
                                     "the session median \u2014 read those classes as inference.")
                        _pair = _G9_PAIR_NOTE.get(_w, "") if len(_methods) > 1 else ""
                        _mlabel = ("" if not _m else
                                   f", {'ORTHOGONALISED' if _m.endswith('_orth') else 'PLAIN'} "
                                   f"direction")
                        title(s, f"G9. {_an} \u2014 {_w} window, {_tag}{_mlabel}",
                              _blurb + _lickonly + _pair)
                        note(s, M_CODING_DIR, specific=S_G9)
                        big(s, _f, top=1.95, width=12.7)

        # --- G9c. THE PER-SESSION, PER-CLASS versions of the two matrices.
        #
        # These were RENDERED EVERY NIGHT AND NEVER PLACED. `position_coding_directions` writes
        # `coding_{crosssess,pairsess}_{window}_{method}_{class}_{animal}.png` -- 144 files across
        # 3 windows x 2 methods x 3 classes x 4 animals -- and no slide referenced them, so they
        # existed only on disk (found 2026-08-24, while answering "how does the best match change
        # over recovery sessions, split by miss-while-working vs stopped?" -- which is exactly what
        # they show and nothing in the deck did).
        #
        # ONE METHOD ONLY here. Three classes x six sessions is already dense; adding the plain
        # direction would double it again for a comparison the pooled G9 panels above already
        # carry.
        _SESS_CLS = (("poststroke_lick", "LICK trials"),
                     ("poststroke_miss_working", "MISS while still working"),
                     ("poststroke_stopped", "STOPPED (quit for the day)"))
        for _w in ("ENL", "cue", "lick"):
            _m = _G9_METHODS[_w][0]
            for _kind, _tag, _blurb in (
                ("crosssess", "cross-position matrix, PER SESSION",
                 ("The pooled cross-position matrix split by post-stroke SESSION. Rows = TRUE spout "
                  "position, columns = which position's direction it was scored on, 1.0 = that "
                  "column's own pre-stroke lick signature. THIS IS THE RECOVERY VIEW: pooling every "
                  "post-stroke day hides whether a row moved toward another position and stayed "
                  "there, moved and came back, or never moved at all.")),
                ("pairsess", "pairwise axes, PER SESSION",
                 ("The pairwise A-vs-B axes split by post-stroke SESSION, same anchoring as the "
                  "pooled version: 1 = pre-stroke lick at the panel's position, 0 = pre-stroke lick "
                  "at the partner.")),
            ):
                for _cls, _clsn in _SESS_CLS:
                    for _an in sorted({s_["label"][:4] for s_ in config.load_sessions()}):
                        _f = src / f"coding_{_kind}_{_w}_{_m}_{_cls}_{_an}.png"
                        if not _f.exists():
                            continue
                        s = slide()
                        title(s, f"G9c. {_an} — {_w} window, {_tag} — {_clsn}",
                              _blurb + "  READ THE CLASS: miss-while-working is position-specific, "
                              "STOPPED is the animal having quit and is position-GENERAL, so a row "
                              "that moves in STOPPED at every position is a state change and not a "
                              "remapping.")
                        note(s, M_CODING_DIR, specific=S_G9)
                        big(s, _f, top=1.95, width=12.7)

        # --- G9b. COHORT diagnostics. Neither can be drawn per animal: the first needs every
        # position of every animal on one axes to be a relationship at all, and the second VANISHES
        # when animals are pooled, which is itself the finding.
        for _w in ("ENL", "cue", "lick"):
            # The COHORT diagnostics stay on the orthogonalised variant alone. They are arguments
            # about the geometry of the axes (how much an axis IS the close-vs-far dimension, and
            # whether that predicts drift), and both were measured on the orthogonalised
            # directions; drawing the plain ones beside them would put two different measurements
            # under one claim. The per-animal G9 panels are where both variants belong.
            _m = _G9_METHODS[_w][0]
            for _kind, _tag, _blurb in (
                ("cosslope",
                 "why the decline is close-specific when the disengagement is not",
                 ("READ IT AS: how much is this axis really the close-vs-far dimension (x), and how "
                 "much do its own pre-stroke LICK trials drift over the session (y). ONE POINT PER "
                 "POSITION PER ANIMAL; circles are close positions, triangles far. THE ARGUMENT: "
                 "the behavioural disengagement is uniform across positions, so it cannot by itself "
                 "produce a decline at only some of them. What it CAN do is move activity along one "
                 "dimension -- close-vs-far -- and a one-vs-rest axis for a close position is "
                 "largely that dimension, because 'not close_center' is majority-far. So the more "
                 "an axis points along close-vs-far, the more drift it must show even if nothing "
                 "about that position's coding changed. A sloped cloud here says the drift is a "
                 "property of the AXIS, not of the spout. Measured: r=-0.567, with axes pointing "
                 "CLOSE averaging -0.347 and those pointing FAR +0.030. It is NOT the engagement "
                 "axis -- these directions are already orthogonalised against lick-vs-no-lick, and "
                 "cos(close-vs-far, engagement) is only -0.43 to +0.34.")),
                ("pairsplit", "which pairwise cells are safe to read",
                 ("A pairwise axis contrasts two spouts DIRECTLY, so it need not carry the "
                 "close-vs-far dimension at all -- if both spouts sit at the same distance. LEFT: "
                 "it does not (|cos| 0.33 within-ring against 0.70 cross-ring). RIGHT: the drift, "
                 "split by animal, because pooling destroys the effect -- over all 60 pairs "
                 "r(cos, drift) is only -0.143, which read alone says the pairwise axes drift as "
                 "much as anything else. Split, the two DISENGAGING animals put all 18 of their "
                 "cross-ring pairs in the same direction (p ~ 4e-6, mean +0.19) while their "
                 "within-ring pairs are a coin flip (6/12, mean -0.01), and the two steady animals "
                 "are 9/18 and 5/12 -- what 'nothing to detect' looks like. A is the FAR position "
                 "in every cross-ring pair, so POSITIVE means far trials become MORE far-like as "
                 "the session runs, the same drift the one-vs-rest axes show from the other end. "
                 "USE THE WITHIN-RING CELLS for remapping questions.")),
            ):
                _f = src / f"coding_{_kind}_{_w}_{_m}.png"
                if not _f.exists():
                    continue
                s = slide()
                title(s, f"G9b. {_w} window \u2014 {_tag}",
                      "Diagnostic, not a result: it explains how to read the G9 panels.")
                note(s, M_CODING_DIR, specific=S_G9B)
                big(s, _f, top=1.95, width=12.7)

        # --- G9d. ONE-OFF control (wfield_local.rt_drift, not a nightly step).
        # RELABELLED G9c -> G9d on 2026-08-28: two unrelated analyses both called themselves G9c
        # (the per-session matrices above, and this), so a spoken reference to "G9c" picked out two
        # different slides. Section labels are navigation; a duplicate one is a broken link.
        _rt = src / "coding_rtdrift.png"
        if _rt.exists():
            s = slide()
            title(s, "G9d. First-lick latency across the course of a session",
                  "Two controls in one figure, and the answer splits by RING. (1) SLOWED vs "
                  "SKIPPED: a late collapse in response rate could be an animal getting slower or "
                  "an animal stopping. FLAT latency with a falling response rate is the sated tail "
                  "\u2014 the licks that still happen are as fast as ever, there are just fewer. "
                  "(2) THE WOULD-BE-LICK OFFSET: a no-lick trial's window uses ONE median RT for "
                  "the whole session, so an animal that slowed through it would have its late "
                  "trials placed progressively too early \u2014 the exact shape of a within-session "
                  "decline, which must be excluded before any such decline is believed.\n\n"
                  "MEASURED. CLOSE positions are flat in every animal (drift <=0.03 s). FAR "
                  "positions are flat in PS94 (<=0.05) and PS95 (<=0.03) but NOT in PS92 "
                  "(+0.13/+0.15/+0.23) or PS93 (far_center +0.27, far_L +0.50, i.e. 0.53 s to "
                  "1.03 s across the session). So the two processes are separable and are not the "
                  "same thing: DISENGAGEMENT is uniform across positions and shows as SKIPPING, "
                  "while FATIGUE is position-specific and shows as SLOWING at the animal's hard "
                  "positions \u2014 PS93's far_L and far_center are exactly where its right "
                  "orofacial deficit lives, and the two animals that disengage most barely slow at "
                  "all.\n\nSO: the control holds where it was used, since the within-session "
                  "neural decline sits at CLOSE positions in PS94/PS95 and latency there is flat "
                  "everywhere. But the offset IS wrong late in a session for PS93 far_L \u2014 a "
                  "session median near 0.6 s against a last-quartile 1.03 s misplaces those windows "
                  "by ~0.4 s, a fifth of the window. Read PS93's far no-lick cells with that in "
                  "mind. Rebuild with: python -m wfield_local.rt_drift")
            note(s, M_CODING_DIR, specific=S_G9C)
            big(s, _rt, top=1.95, width=12.7)

        # --- G9e. EARLY vs LATE rewarded trials (Priya, 2026-08-28).
        #
        # PLACED HERE because it is the payoff of the latency control immediately above: G9d
        # establishes that first-lick latency is a real, position-specific quantity that moves, and
        # this splits the decode on it. Reading them the other way round makes the split look
        # arbitrary.
        #
        # ONE ANIMAL PER SLIDE, not 2-up. The figure is 10.4in and carries 6x6 tick labels at 7pt;
        # placed at 11.0in those reach the reader at ~7.4pt, while 2-up at 6.2in would land them at
        # 4.2pt. The same measurement that put `pooled` and `normunit` INTO the 2-up set keeps this
        # one out of it (DECISIONS 2026-08-28).
        # THE KIND AS A LITERAL, like every other block here. `test_analysis_deck` reads the deck's
        # `("<kind>"` literals to check that every kind the module declares reaches a slide, and it
        # has to: the sibling blocks all build their filenames from a loop variable, so the
        # filename itself never appears in this source and cannot be searched for. Interpolating
        # "rtsplit" straight into the f-string would place the figure while reading, to that guard,
        # as an unplaced kind.
        _kind, _tag = ("rtsplit", "EARLY vs LATE rewarded trials")
        for _w in ("ENL", "cue", "lick"):
            for _an in sorted({s_["label"][:4] for s_ in config.load_sessions()}):
                _f = src / f"coding_{_kind}_{_w}_{_an}.png"
                if not _f.exists():
                    continue
                s = slide()
                _lickwin = ("" if _w != "lick" else
                            "  IN THE LICK WINDOW the split is between trials whose window STARTS "
                            "early and late; the window still opens at the animal's own first lick, "
                            "so this is not the cue-referenced timing the other two windows show.")
                title(s, f"G9e. {_an} \u2014 {_w} window, {_tag}",
                      "The same frozen pre-stroke decoder, the same post-stroke LICK trials, "
                      "regrouped by reaction time at 2.0 s. Left panel is the pre-stroke "
                      "leave-one-session-out reference; the two post panels ADD BACK to the "
                      "poststroke_lick matrix shown everywhere else. Rightmost panel is the actual "
                      "comparison \u2014 per-position recall, early against late \u2014 because "
                      "judging that by matching colours across two heatmaps is the one thing the "
                      "eye is worst at.  PRESERVED on late trials = plan intact, execution slow; "
                      "DEGRADED = a different result. BUT READ n FIRST: measured 2026-08-28 the late "
                      "arm is only 3.4% of post-stroke rewarded trials (PS92 5.6%, PS93 7.5%, PS94 "
                      "1.0% = 26 trials, PS95 0.8% = 27), so PS94's and PS95's late panels are "
                      "marked TOO FEW TO READ in red on the figure. The emptiness of that arm is "
                      "itself the finding: the impaired animals do not lick slowly, they lick fast "
                      "or not at all." + _lickwin)
                note(s, M_CODING_DIR, specific=S_G9E)
                big(s, _f, top=1.95, width=11.0)

        # --- G8. hemispheric raw fluorescence: the 470 question cannot be asked without the 415 one
        _hemi = [(g, src / f"hemispheric_intensity_{g}.png")
                 for g in ("all", "SSp") if (src / f"hemispheric_intensity_{g}.png").exists()]
        for _g, _f in _hemi:
            s = slide()
            title(s, f"G8. LEFT/RIGHT raw fluorescence across days \u2014 {_g}",
                  "Is the lesioned hemisphere brighter? 415 nm is the ISOSBESTIC channel, so a "
                  "left-sided 470 rise is not an activity change unless 470/415 moves too \u2014 "
                  "the two questions are one measurement. The PERFUSION DIRECTION of a 415 shift "
                  "is UNRESOLVED: measured on this data the raw 415 RISES with activation "
                  "(+0.5 to +2.0%), contradicting a simple absorption account, so do NOT read a "
                  "415 change as hypo- or hyper-perfusion.")
            note(s, M_HEMI, specific=S_G8)
            big(s, _f, top=1.9, width=12.9)

        # --- G8b. per-hemisphere dynamics + cross-hemisphere concordance
        for _src_ in ("roi", "joint"):
            _df = src / f"hemispheric_dynamics_{_src_}.png"
            if not _df.exists():
                continue
            s = slide()
            title(s, f"G8b. Per-hemisphere DYNAMICS and cross-hemisphere COUPLING ({_src_})",
                  "Temporal SD is what a mean image cannot show, and homotopic correlation is what "
                  "survives the optical asymmetries that make amplitudes fragile. Third row is the "
                  "specificity check: a homotopic drop only means interhemispheric decoupling if "
                  "WITHIN-hemisphere coupling holds. Grey = the small-lesion sessions (not no-lesion).")
            note(s, M_HEMIDYN, specific=S_G8B)
            big(s, _df, top=1.85, width=12.9)

        # G8c: surface vessel contrast -- Priya's observation that vessels look fainter post-stroke
        _vf = src / "vessel_contrast.png"
        if _vf.exists():
            s = slide()
            title(s, "G8c. Surface vessel contrast — do the vessels get fainter?",
                  "Vessels image dark because haemoglobin absorbs, so their contrast is an optical "
                  "readout of blood in the light path. Gain-invariant by construction. Read the L/R "
                  "ROW: focus drift and a clouding window reduce contrast BILATERALLY. CAVEAT: these "
                  "are PIAL vessels over DORSAL cortex and the lesion is ventrolateral striatum, so a "
                  "null here is weak evidence about perfusion at the lesion.")
            note(s, M_VESSEL, specific=S_G8C)
            big(s, _vf, top=1.85, width=12.6)

        # G8d: the OBSERVATION itself -- pre/post maps on one colour scale. This has to come
        # BEFORE the decomposition, because the decomposition is only interesting once the reader has
        # seen the thing being decomposed. Priya read the amplitude difference off colourbar numbers;
        # no figure in the deck showed it, and the per-session renormalisation actively hid it.
        for _al, _nice in (("cue", "POST-cue"), ("lick", "POST-lick")):
            # EVERY PAGE, not just the first (Priya, 2026-08-28: "only the first few post-stroke
            # dates are shown"). `fixed_scale_maps` paginates at MAX_POST_PER_FIG and part 1 keeps
            # the historical filename, so this glob matched page 1 alone and the later post-stroke
            # days -- exactly the ones the recovery story is about -- were written every night and
            # shown nowhere. The colour limit is shared across ALL parts and the PRE row is repeated
            # on each, so the pages are directly comparable to one another.
            for _an in sorted({p.name.split("_")[3] for p in src.glob(f"fixed_scale_maps_*_{_al}.png")}):
                _pages = [src / f"fixed_scale_maps_{_an}_{_al}.png"]
                _pages += sorted(src.glob(f"fixed_scale_maps_{_an}_{_al}__p*.png"),
                                 key=lambda q: int(q.stem.rsplit("__p", 1)[1]))
                _pages = [q for q in _pages if q.exists()]
                for _pi, _fsf in enumerate(_pages, 1):
                    s = slide()
                    _of = f" (part {_pi} of {len(_pages)})" if len(_pages) > 1 else ""
                    title(s, f"G8d. Pre- vs post-stroke maps on ONE COMMON COLOUR SCALE "
                             f"— {_an}, {_nice}{_of}",
                          "Every panel shares one symmetric vmin/vmax, so a 2-3x amplitude "
                          "difference shows as a 2-3x difference in saturation. The standard maps "
                          "renormalise per session and cannot show this. Baseline F is unchanged "
                          "(ratios 0.99-1.02), so the change is in the numerator."
                          + (" The colour limit is shared across ALL parts and the PRE row is "
                             "repeated on each, so the pages compare directly." if _of else ""))
                    note(s, M_FIXEDSCALE, specific=S_G8D)
                    big(s, _fsf, top=1.9, width=11.6)

        # G8e: per-area evoked amplitude -- the measure aimed at Priya's map observation, and the
        # only one in the hemispheric line that is not a null.
        for _al, _nice in (("cue", "POST-cue"), ("lick", "POST-lick")):
            _ef = src / f"evoked_amplitude_{_al}.png"
            if not _ef.exists():
                continue
            s = slide()
            title(s, f"G8e. Per-AREA evoked amplitude ({_nice}) — lateralisation collapses in PS94 "
                     f"and ONLY in PS94",
                  "ROW 1 is what the map colourbars show and is the only row carrying the baseline "
                  "confound; it SUMS across 66 areas, so it conflates amplitude with spatial "
                  "extent — see G8d. ROWS 2–3 are scale-free. DIRECTION IS THE RESULT, not "
                  "whether a value leaves the band: among positions that were lateralised "
                  "pre-stroke, PS94 moves 4 TOWARD ZERO and reverses a 5th, identically on both "
                  "days and both alignments, while PS93 and PS95 move AWAY from zero and PS92 "
                  "does not move at all.")
            note(s, M_EVOKED, specific=S_G8E)
            big(s, _ef, top=1.9, width=11.6)

        # G8f: the two tests aimed at the mechanism behind the map observation -- do the position
        # patterns converge, and did any of them cross the midline. Comes after G8e because it is
        # the follow-up to the lateralisation collapse, not an independent question.
        for _al, _nice in (("cue", "POST-cue"), ("precue", "PRE-cue")):
            for _armf, _armn in (("", "ALL trials"), ("_lickonly", "LICK-ONLY")):
                # ONE SLIDE PER PART. spatial_reorganisation draws one column per post-stroke
                # session and chunks at MAX_COLS_PER_FIG, because a single figure of 18 columns
                # placed at 11.6 in is 1.3 in tall and unreadable. Part 1 keeps the historical
                # filename; the rest carry __pN.
                _sfs = [src / f"spatial_reorganisation_{_al}{_armf}.png"]
                _sfs += sorted(src.glob(f"spatial_reorganisation_{_al}{_armf}__p*.png"),
                               key=lambda q: int(q.stem.rsplit("__p", 1)[1]))
                _sfs = [q for q in _sfs if q.exists()]
                for _pi, _sf in enumerate(_sfs, 1):
                    _part = f" \u2014 part {_pi}/{len(_sfs)}" if len(_sfs) > 1 else ""
                    s = slide()
                    title(s, f"G8f. Pattern CONVERGENCE and the MIDLINE test ({_nice}, {_armn})"
                             f"{_part}",
                          "Crossnobis is noise-unbiased, so sessions of different trial count "
                          "and response extent can be compared; the pre-stroke band is rebuilt "
                          "on each session's OWN positions, because mean distance averages "
                          "over PAIRS. Bars: correlation with the animal's own pre-stroke "
                          "pattern (blue) vs the "
                          "HEMISPHERE-SWAPPED one (orange). Orange above blue would mean the "
                          "pattern relocated across the midline \u2014 it never happens.")
                    note(s, M_SPATIAL, specific=S_G8F)
                    big(s, _sf, top=1.9, width=11.6)

        # --- G9. what is NOT here, and why
        s = slide()
        title(s, "G9. Excluded sessions and deferred analyses",
              "A section that does not say what it left out reads as though it covered everything.")
        note(s, M_POSTSTROKE, specific=S_GEXCL)
        bullets(s, [
            (f"EXCLUDED from every POOLED slide: {', '.join(_excluded)}. Their 8/16 attempt produced "
             "no deficit; the effective lesion (3.75 / 5.5 mW) followed the 8/17 session, so 8/17 is "
             "neither a clean baseline nor post-stroke and belongs to neither phase. They "
             "are NOT unanalysed \u2014 they are the SMALL-LESION COMPARISON, shown per session at G7c and as the two grey squares in G2c's grid, and more "
             "importantly, the WITHIN-ANIMAL BEFORE/AFTER CONTROL in G2c: these same animals' 8/18 "
             "sessions ARE post-stroke and carry the dissociation, while their 8/17 sessions show "
             "nothing outside the band at any alignment. Same animal, same rig, one day apart. That "
             "control exists only because these sessions were kept analysable instead of discarded. "
             "They also remain registered, are projected onto the joint bases, and appear "
             "per-session in sections A\u2013D.")
            if _excluded else
            "No sessions are currently in the 'excluded' phase.",
            ("RETIRED, not merely omitted: the working-vs-disengaged identity split. Its comparison "
            "class was never validated, so its result (PS94 \u22120.060) is uninterpretable rather "
            "than negative. G6 asks the same question without an engagement label."),
            ("DEFERRED to the second post-stroke session: joint-LocaNMF replication of G2b\u2013G6, "
            "and the independently-trained-decoder similarity analysis. Both need n > 1."),
            ("BLOCKED on DLC/facial tracking: splitting 'no lick detected' into attempted-and-missed "
            "vs never-attempted. Until then every no-lick claim above carries that ambiguity."),
            ("PS92/PS93 HAVE re-entered as post-stroke: their effective lesion followed the 8/17 "
            "session, so 8/18 is their post-stroke day 1 and appears in every pooled slide. 0817 "
            "stays in their exclude list, which is what makes the before/after control above "
            "possible."),
        ])

    # ---------------- H. GRANT FIGURES ----------------
    # The summary set built by `wfield_local.grant_figures` into <labcams>/grant_figures. Included
    # here so the deck and the grant tell the same story from the same numbers -- a figure that
    # exists only in a document is one nothing regenerates, which is how prose goes stale
    # (Priya, 2026-08-25: "add these figures to the analysis deck code too").
    #
    #
    # A DIFFERENT ROOT: these are NOT in `src` (figures_working) but under `labcams`, because they
    # are a deliverable rather than an analysis intermediate.
    _grant = grant_dir
    if _grant.exists():
        divider("H. GRANT FIGURES — the summary set",
                "Built by `python -m wfield_local.grant_figures` into <labcams>/grant_figures. "
                "Deliberately caveat-light for a non-specialist reader; the caveats are in the "
                "speaker notes here and in DECISIONS.md.")
        # SLIDES FOLLOW THE LABELS, not the order the entries happen to be written in. Entries get
        # appended next to the family they relate to, which had already put H7d before H7b and
        # H8e/H9 before H8b -- labels that promise an order the deck did not follow. Sorting on the
        # label here means a future insertion cannot reintroduce that, wherever it is written.
        #: Families placed out of label order, as (label, sort-key). Priya, 2026-09-13: "move H7d
        #: after H6d". H7d IS the control for H6d -- the same difference-from-pre-stroke, computed
        #: within session so no lesion comparison enters it -- and its blurb already says "the pair
        #: to put side by side". Sorted by label they land twelve slides apart with H7 and H7b in
        #: between, which is the one arrangement that stops a reader making the comparison the
        #: figure exists for. THE LABEL DOES NOT CHANGE: it is still H7d on the slide and in every
        #: document that cites it; only where the deck puts it moves.
        _HPLACE = {"H7d": (6, "e")}     # immediately after H6d (6, "d")

        def _hkey(entry):
            m = re.match(r"H(\d+)([a-z]*)\.", entry[1])
            if not m:
                return (99, "")
            return _HPLACE.get(f"H{m.group(1)}{m.group(2)}", (int(m.group(1)), m.group(2)))

        # EXCLUDE THE COMPACT VARIANTS. `grant_figures --compact` writes `<stem>_compact.png` beside
        # each dense grid, and every pattern here ends in `_*.png`, so they matched: the 2026-08-27
        # deck placed 13 of them as if they were separate figures -- the same numbers twice, once
        # with in-cell digits and once without. Four were worse than duplicates. They predate the
        # lick/working variant split (`d064ae4`), so no step writes those filenames any more and no
        # render can refresh them; they would have sat at their 2026-08-26 21:44 content in every
        # future deck. The compact PNGs are a deliverable for pasting into the grant document, not
        # deck slides, and nothing else in the repo reads them.
        for _pat, _title, _blurb in sorted(GRANT_FIGURES, key=_hkey):
            for _gf in sorted(_grant.glob(_pat)) if "*" in _pat else [_grant / _pat]:
                if not _gf.exists() or _gf.stem.endswith("_compact"):
                    continue
                s = slide()
                # The suffix is whatever the glob's `*` matched (window, and for the pattern
                # figures the trial class) -- NOT a blind split of the stem, which would repeat
                # the family name already in the title.
                _suffix = ""
                if "*" in _pat:
                    _suffix = " — " + _gf.stem[len(_pat.split("*")[0]):].replace("_", " ")
                title(s, f"{_title}{_suffix}", _blurb)
                note(s, "Grant summary figure. Source: wfield_local.grant_figures. The coverage "
                        "footer on each figure states which post-stroke sessions ITS OWN SOURCE "
                        "contains, and flags any that are registered but absent -- read it before "
                        "comparing animals.")
                big(s, _gf, top=1.95, width=12.6)

    # ----------------------------------------------------------- SECTION I: pooled EPOCH figures
    #
    # `wfield_local.epoch_grant_figures` into <labcams>/grant_figures/epoch. Cross-animal, four
    # panels -- pre / acute / subacute / chronic -- instead of a time axis, sized to be read at a
    # quarter page (Priya, 2026-08-28; chronic added 2026-09-07).
    #
    # THE CHRONIC PANEL IS PS92 ONLY, and that is the finding rather than a gap in the data: it is
    # the one animal whose far_R hit rate AND licks/trial both satisfy `epochs.CHRONIC_RULE`. The
    # per-epoch counts in every subtitle name the contributing animals, so a reader sees n=1 stated
    # rather than inferred. Adding chronic also REMOVED PS92's days 11/15/18 from the subacute
    # panel, so subacute bars moved on 2026-09-07 -- they are not comparable to a deck built before
    # that date.
    #
    # THE SPEAKER NOTES ARE WRITTEN AS GRANT FIGURE LEGENDS, not as deck commentary: a reader
    # should be able to lift one into a proposal and have it stand alone -- what is plotted, what
    # the unit is, what the error bars and the marks mean, and what the panel shows. Where a legend
    # would mislead without a caveat the caveat is IN the legend, because a caveat that lives only
    # in the deck does not travel with the figure.
    #
    # EVERY BAR FAMILY IS FOLLOWED BY ITS INTERVAL PANEL. The bars carry the marks and the
    # magnitudes; the companion carries the effect SIZE and what the multiple-comparison correction
    # costs. A mark says only "not zero", which is the half of the result that travels worst.
    _epoch = _grant / "epoch"
    if _epoch.exists():
        # THE BOUNDARIES ARE NAMED ON THE DIVIDER because `chronic_from` is DERIVED each run
        # (2026-09-07). A deck that shows epoch-stratified panels without saying which epochs it
        # used cannot be compared with last week's, and the boundaries can now differ between the
        # two without anyone having edited anything.
        _bnd = ", ".join(
            f"{a} chronic from day {_epochs.spec_for(a).get('chronic_from')}"
            if _epochs.spec_for(a).get("chronic_from") is not None else f"{a} not chronic"
            for a in sorted(_epochs.EPOCH_SPEC))
        divider("I. POOLED EPOCH FIGURES — pre / acute / subacute / chronic",
                "Built by `python -m wfield_local.epoch_grant_figures` into "
                "<labcams>/grant_figures/epoch. Pooled across all four animals and stratified by "
                "recovery epoch instead of a time axis. Speaker notes are written as grant figure "
                "legends."
                f"\n\nBOUNDARIES USED IN THIS DECK ({_epochs.resolved_source()}): {_bnd}. "
                f"Chronic is derived from behaviour each run -- {_epochs.CHRONIC_RULE}.")
        # NUMBERED BY POSITION. The keys used to be written into the data, so inserting a
        # family in narrative order meant renumbering every entry after it -- and the cost of not
        # doing that was 33 of 75 figures simply not referenced anywhere, which the deck's own
        # completeness check cannot report: it reports figures it EXPECTS and is silent about ones
        # it was never told about.
        for _n, (_pat, _ttl, _legend) in enumerate(EPOCH_FIGURES, 1):
            for _ef in (sorted(_epoch.glob(_pat)) if "*" in _pat else [_epoch / _pat]):
                if not _ef.exists():
                    continue
                _sl = slide()
                _suffix = ""
                if "*" in _pat:
                    _suffix = " — " + _ef.stem[len(_pat.split("*")[0]):].replace("_", " ")
                title(_sl, f"I{_n}. {_ttl}{_suffix}", "")
                note(_sl, (_legend or _CI_LEGEND) + "\n\nSource: "
                     "wfield_local.epoch_grant_figures. Epoch boundaries are Priya's, stored in "
                     "wfield_local/epochs.py and counted from each animal's own lesion date; "
                     "`epochs.verify_against_behaviour` re-derives them from the far-contralateral "
                     "accuracy rule and REPORTS agreement rather than reassigning, so a published "
                     "boundary cannot move when a session is registered.")
                # `big` fits BOTH dimensions, so for anything taller than the space available
                # the placed scale is avail_h / fig_h and the WIDTH never binds. At top=1.95 that
                # left 5.40in, and the 5.76in matrix families were placed at 0.94x -- SMALLER than
                # native, rendering their 9pt titles at 8.4pt. The 1.95 was reserving room for a
                # subtitle this section does not pass (`title(_sl, ..., "")`); the title box ends
                # at 1.11in, so the deck's own 1.40 default clears it and buys back 0.55in.
                big(_sl, _ef, top=1.40, width=12.6)

    out_path = Path(out_path)
    _refuse_incomplete_overwrite(out_path, missing_figures, allow_missing)
    _refuse_failed_steps(out_path, failed_steps, allow_failed_steps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # WINDOW PROVENANCE ON EVERY SLIDE, appended last so it sits under whatever note the
    # section already wrote. Derived from each figure's own filename plus defaults.yaml,
    # so it cannot drift from what it describes.
    #
    # THE CAPTION IS PREPENDED, the provenance appended, so the notes read: what this figure IS,
    # then what is specific about it, then the shared methods, then how it was built. Both are
    # derived from the figure's own filename in the same pass -- see `figure_caption`.
    _prov = 0
    _caps = 0
    _years = _session_years()
    # NOTES FIRST, now that every figure is recorded, so a `SELF` token finds the figure on
    # its own slide however early that slide's `note` was called. It must also come BEFORE
    # the caption pass below, which PREPENDS to the note text -- flushing after it silently
    # overwrote every per-figure caption.
    canvas.flush_notes()
    for _sl in slide_order:
        _figs = figs_by_slide.get(id(_sl._element), [])
        _cap = figure_caption(_figs, years=_years)
        _line = window_provenance(_figs)
        if not _cap and not _line:
            continue
        _tf = _sl.notes_slide.notes_text_frame
        if _cap:
            _tf.text = (_cap + chr(10) + chr(10) + _tf.text) if _tf.text else _cap
            _caps += 1
        if _line:
            _tf.text = (_tf.text + chr(10) + chr(10) + _line) if _tf.text else _line
            _prov += 1
    print(f"[analysis_deck] per-figure captions written to {_caps} slide(s)", flush=True)
    print(f"[analysis_deck] window/binning provenance written to {_prov} slide(s)",
          flush=True)
    keep_previous(out_path)
    prs.save(str(out_path))
    manifest, stale = _write_manifest(out_path, placed_figures, run_start)
    # UNRESOLVED SIDECAR TOKENS ARE REPORTED, NOT RAISED. Each one already left a visible marker on
    # its slide; this makes them findable without opening the deck, which is what a nightly needs.
    unresolved = values.report()
    if unresolved:
        print(f"  [notes] {len(unresolved)} sidecar reference(s) did not resolve:", flush=True)
        for line in unresolved[:20]:
            print(f"    {line}", flush=True)
    return {"out": str(out_path), "slides": len(prs.slides),
            "figures_present": placed["present"], "figures_missing": placed["missing"],
            "missing_figures": missing_figures, "tag": tag,
            "manifest": (str(manifest) if manifest else None),
            "stale_figures": [r["figure"] for r in stale],
            "stale_detail": stale,
            "unresolved_values": unresolved}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None, help="figure dir (default: figures_working root)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output pptx (default: <labcams>/spout_position_analysis_summary.pptx)")
    ap.add_argument("--machine", default=None)
    ap.add_argument("--allow-missing", type=int, default=0, metavar="N",
                    help="publish even though N figures are missing (default 0: refuse)")
    ap.add_argument("--run-start", type=float, default=None, metavar="EPOCH",
                    help="epoch seconds this run began; placed figures older than it are "
                         "reported as not-refreshed in the manifest")
    args = ap.parse_args(argv)
    rv = PathResolver(machine=args.machine)
    src = args.src or Path(rv.root("figures_working"))
    out = args.out or (Path(rv.root("labcams")) / "spout_position_analysis_summary.pptx")
    summary = build_analysis_deck(src, out, allow_missing=args.allow_missing,
                                  run_start=args.run_start)
    print(f"[analysis_deck] wrote {summary['out']}  ({summary['slides']} slides, "
          f"{summary['figures_present']} figures placed, {summary['figures_missing']} missing)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
