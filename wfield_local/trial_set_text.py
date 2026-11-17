"""THE TRIAL SET OF A FIGURE, IN ONE SENTENCE, FROM THE ACTIVE SELECTION (Priya 11-09).

Priya 11-09: *"EVERY figure must state EXACTLY which trials were included (and which excluded)."* Titles said
"post-cue, lick + miss-while-working", which is the DAQ quit-tail wording: under the camera gate the WORKING trial set
is camera contact + incomplete trials inside the working period (first -> last camera lick; stop rule A2 removes the
stopped runs; the quit tail after the last lick is excluded), and jaw-only, no response, unknown, stopped /
stopped_licking and after-end trials are NOT working trials (they still count for the rest-baseline working PERIOD).

ONE SOURCE OF THE WORDS. `sentence(variant)` reads the run's switches -- the trial gate (WIDEFIELD_TRIAL_GATE), the
working selection (WIDEFIELD_WORKING_TRIALS), the stop rule (WIDEFIELD_TRIAL_STOPPED + WIDEFIELD_STOP_LICKS), the end
anchor (WIDEFIELD_TRIAL_END), the lick source (WIDEFIELD_LICK_SOURCE) and the DAQ-rule gate source
(WIDEFIELD_GATE_SOURCE) -- through the modules that define them (`trial_classes`, `working_trials`, `lp_licks`,
`gate_source`), so a caption cannot describe a selection the run did not make. `short_label(variant)` is the
title-sized form; `arm_caption(align, variant)` the epoch families' arm caption (replaces the stale ARMS strings).

HOW IT REACHES EVERY FIGURE. A renderer declares the trial set it is drawing (`activate` / `active`; the epoch arm
loop, restnl_maps, lp_final_render), and `stamp(fig)` writes the sentence onto the figure at save time -- from the
shared save path (`figure_layout.save_png_svg`), so every writer that goes through it is covered, plus the two
stamp-aware wrappers (`grant_kit._save`, `epoch_figures._save_png_svg`) that fit it into the run-stamp band. With no
active trial set nothing is drawn (a PNG is byte-identical to a plain savefig, `tests/test_save_png_svg.py`). Map
bundles record the sentence (`epoch_figures.save_map_bundle`), so a figdata replot carries it too.

THIS MODULE IS IN NO CACHE KEY (`map_collect.code_digest` targets, `result_cache.code_version` targets, the LP lick-
times sidecars): captions can change without cold-starting a map cache.
"""
from __future__ import annotations

import contextlib
import contextvars
import textwrap

#: gid of the text artist the stamp writes (or prepends to): the idempotency mark, so the shared save path and the
#: wrapper that called it never stamp a figure twice.
GID = "trial_set"
#: the run-stamp colour of `lp_licks.stamp` (the stamp prepends to that text when it is present)
RUN_STAMP_COLOR = "#8a2be2"
VARIANTS = ("working", "lick", "miss_working", "stopped", "jaw_only", "working_period")
#: the PERIOD SUPPLEMENT (Priya 11-09 option c): every judged trial in the working period
WORKING_PERIOD = "working_period"
#: the camera class a jaw-only panel selects (`trial_classes.CLASSES`; trial_attempts.csv cls == "jaw_only")
JAW_ONLY_CLASS = "jaw_only"

ALIGN_TEXT = {"precue": "ENL (pre-cue)", "cue": "post-cue", "lick": "post-lick"}
_CLASS_WORD = {"contact": "contact", "incomplete": "incomplete", "jaw_only": "jaw-only", "no_response": "no response",
               "unknown": "unknown"}
_END_TEXT = {"lick": "last camera lick", "contact": "last camera contact", "close": "close-spout cutoff",
             "none": "session end (no end cut)"}
_STOP_TEXT = {"both": "A2", "run": "A0", "long": "A3", "any": "every no-response trial stopped"}
_STALE = ("lick + miss-while-working", "LICK + miss-while-working (quit period removed)")

_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("trial_set_active", default=None)


# ----------------------------------------------------------------------------------------------- the words
def _classes(names) -> str:
    return " + ".join(_CLASS_WORD.get(c, c) for c in ("contact", "incomplete", "jaw_only") if c in set(names))


def lick_text() -> str:
    from wfield_local import lp_licks
    return "LP licks" if lp_licks.is_lp() else "DAQ licks"


def _max_rt_s() -> str:
    try:
        from wfield_local import trial_classes as tc
        return f"{tc.max_rt_ms() / 1000:g} s"
    except Exception:                                                  # noqa: BLE001 - a caption never kills a figure
        return "max_rt"


def _camera() -> bool:
    from wfield_local import trial_classes as tc
    return tc.camera_gate_on()


def stop_text() -> str:
    """The camera stop rule in words, e.g. "A2 + lick excise"."""
    from wfield_local import trial_classes as tc
    rule = tc.stopped_rule()
    txt = _STOP_TEXT.get(rule, rule)
    if rule in tc.GUARDED_RULES:
        lk = tc.stop_lick_rule()
        txt += {"excise": " + lick excise", "strict": " + strict lick break", "none": ""}[lk]
        if tc.stop_tail_start() == "lick_event":
            txt += ", tail from the last lick event"
    return txt


def working_classes() -> frozenset:
    """The camera classes of the run's WORKING set (`trial_classes.sel_classes`)."""
    from wfield_local import trial_classes as tc
    return tc.sel_classes("working")


def _period() -> str:
    from wfield_local import trial_classes as tc
    return f"within the working period (first-{_END_TEXT.get(tc.end_anchor(), tc.end_anchor())})"


def _removed() -> str:
    return f"stopped runs ({stop_text()}) and quit tail removed"


def _excluded(kept) -> str:
    out = [_CLASS_WORD[c] for c in ("contact", "incomplete", "jaw_only") if c not in set(kept)]
    return "excluded: " + ", ".join(out + ["no response", "unknown"])


def _daq_gate() -> str:
    from wfield_local import gate_source
    return "DAQ quit-tail rule" + ("; quit = no DAQ contact AND no LP attempt" if gate_source.is_lp() else "")


def sentence(variant: str = "working") -> str:
    """THE trial-set sentence of ``variant`` under the run's ACTIVE selection, e.g. (production camera run)::

        trials: camera contact + incomplete (LP licks), within the working period (first-last camera lick),
        stopped runs (A2 + lick excise) and quit tail removed; excluded: jaw-only, no response, unknown
    """
    if variant not in VARIANTS:
        raise ValueError(f"trial_set_text: unknown trial class {variant!r} (expected one of {VARIANTS})")
    lk = lick_text()
    if variant == "lick":
        # camera_rows / camera_select "lick": the engaged rows UNCHANGED -- a lick within max_rt, no camera class cut
        from wfield_local import lp_licks
        what = ("a video lick, contact or incomplete (cam4 + cam1)" if lp_licks.is_lp() else "a DAQ spout contact")
        return (f"trials: lick trials ({lk}: {what} within {_max_rt_s()} of the cue; no camera working-period cut); "
                "excluded: no-lick trials")
    if not _camera():
        if variant == "working":
            return f"trials: lick + no-lick trials outside the terminal quit period ({lk}; {_daq_gate()})"
        if variant == "miss_working":
            return f"trials: no-lick trials before the terminal quit period ({lk}; {_daq_gate()}); excluded: lick trials"
        if variant == "stopped":
            return f"trials: the terminal quit period only ({lk}; {_daq_gate()}); excluded: every earlier trial"
        raise ValueError(f"trial_set_text: the {variant} class set needs the camera trial gate "
                         "(WIDEFIELD_TRIAL_GATE=camera)")
    if variant == "working":
        kept = working_classes()
        return f"trials: camera {_classes(kept)} ({lk}), {_period()}, {_removed()}; {_excluded(kept)}"
    if variant == "miss_working":
        from wfield_local import trial_classes as tc
        kept = tc.sel_classes("miss_working")
        return (f"trials: no-lick trials ({lk}: none within {_max_rt_s()}) of camera class {_classes(kept)}, "
                f"{_period()}, {_removed()}; excluded: lick trials, "
                + ", ".join([_CLASS_WORD[c] for c in ("contact", "incomplete", "jaw_only") if c not in kept]
                            + ["no response", "unknown"]))
    if variant == WORKING_PERIOD:
        from wfield_local import trial_classes as tc
        return (f"trials: all trials in the working period (first-{_END_TEXT.get(tc.end_anchor(), tc.end_anchor())}): "
                f"camera contact + incomplete + jaw-only + no response ({lk}); excluded: stopped runs ({stop_text()}), "
                "stopped-licking, quit tail, unknown (no video / unmatched cue)")
    if variant == "jaw_only":
        return (f"trials: camera jaw-only (jaw moved, no camera lick; never judged at close center / close mid), "
                f"{_period()}, {_removed()}; excluded: contact, incomplete, no response, unknown")
    from wfield_local import trial_classes as tc
    return (f"trials: camera stopped (no-lick trials in {stop_text()} stop runs or the quit tail after the "
            f"{_END_TEXT.get(tc.end_anchor(), tc.end_anchor())}; {lk}); excluded: the working period, stopped-licking, "
            "unknown")


def safe_sentence(variant: str | None) -> str | None:
    """`sentence`, or None when ``variant`` is None or the run cannot name it (a replot must never die on a caption)."""
    if not variant:
        return None
    try:
        return sentence(variant)
    except Exception:                                                  # noqa: BLE001
        return None


def short_label(variant: str = "working") -> str:
    """Title-sized class name under the active selection (no "lick + miss-while-working" under the camera gate)."""
    if variant == "lick":
        return "lick trials only"
    if not _camera():
        return {"working": "lick + miss-while-working (DAQ quit-tail rule)",
                "miss_working": "miss-while-working (DAQ quit-tail rule)",
                "stopped": "stopped (terminal quit period)"}.get(variant, variant)
    if variant == "working":
        return f"camera working trials ({_classes(working_classes())})"
    if variant == "miss_working":
        from wfield_local import trial_classes as tc
        return f"no-lick camera {_classes(tc.sel_classes('miss_working'))} trials"
    if variant == "jaw_only":
        return "JAW-ONLY camera trials"
    if variant == WORKING_PERIOD:
        return "ALL working-period trials (contact + incomplete + jaw-only + no response)"
    if variant == "stopped":
        return f"camera stopped trials ({stop_text()})"
    raise ValueError(f"trial_set_text: unknown trial class {variant!r}")


def gate_text() -> str:
    """Which rule selects the trials, in words: "trial gate: camera classes" or the DAQ quit-tail rule (+ its source)."""
    return "trial gate: camera classes" if _camera() else f"trial gate: {_daq_gate()}"


def arm_caption(align: str, variant: str, *, enl_window=None) -> str:
    """The epoch families' arm caption, e.g. "post-cue, camera working trials (contact + incomplete)"; ``enl_window``
    (seconds) adds the ENL lick-free window on the precue arm."""
    head = ALIGN_TEXT.get(align, align)
    if align == "precue" and enl_window is not None:
        head = f"{head} {float(enl_window):g} s lick-free"
    tail = short_label(variant)
    if variant == "lick" and align != "lick":
        tail = tail.upper()
    return f"{head}, {tail}"


def refresh_title(title: str, variant: str) -> str:
    """``title`` with the stale DAQ wording ("lick + miss-while-working") replaced by `short_label(variant)` -- for a
    figdata replot of a figure drawn before 11-09. Unchanged when nothing stale is in it."""
    out = str(title or "")
    for s in _STALE:
        if s in out:
            out = out.replace(s, short_label(variant))
    return out


def variant_from_name(stem: str):
    """``(align, variant)`` parsed from a figure stem ending ``_<align>_<variant>`` (the epoch map families' naming),
    else ``(None, None)``."""
    s = str(stem).replace("_bundle", "").replace("_replot", "")
    for v in sorted(VARIANTS, key=len, reverse=True):
        if s.endswith("_" + v):
            head = s[: -len(v) - 1]
            al = next((a for a in ALIGN_TEXT if head.endswith("_" + a)), None)
            return al, v
    return None, None


# ----------------------------------------------------------------------------------------------- event / frame sets
# NOT EVERY FIGURE IS DRAWN ON A TRIAL CLASS (Priya 11-09: "EVERY figure must state EXACTLY which trials were included
# / excluded"). The lick families draw LICK EVENTS (cued first licks, ITI licks with their filters, bout starts), the
# rest families draw REST FRAMES, the interhemispheric family both. Their sentence is built here from the defining
# parameters the driver passes (its own constants, never a copy), in the same "<what>: ...; excluded: ..." shape, and
# declared like a trial class (`declare(text=...)`).

#: the ITI lick filters by name (WFIELD_ITI_FILTER), their definitions in words. REFIT filters (`iti_lick_order`:
#: decoder refit on the filtered set) and SUBSET filters (`iti_licks.ITI_SUBSET_FILTERS`: frozen all-ITI decoder,
#: rows dropped downstream; flags from `iti_dock_audit.flags`).
ITI_FILTER_TEXT = {
    "gap2": "no lick onset in the preceding 2 s",
    "far1": ">= 1 s after the response-window end and >= 1 s before the next spout arrival",
    "gap2far1": "no lick onset in the preceding 2 s, and >= 1 s after the response-window end and >= 1 s before the "
                "next spout arrival",
    "dock": "onset while the spout is DOCKED (dock .. next trial_start)",
    "dock01": "onset while the spout is docked, >= 0.1 s after the dock and >= 0.1 s before the next trial_start",
    "gap2dock": "onset while the spout is docked and no lick onset in the preceding 2 s",
    "far1s": ">= 1 s after the response-window end and >= 1 s before the next spout arrival (subset of the frozen "
             "all-ITI predictions, no refit)",
    "notdock": "onset while the spout MOVES (before / during retraction, outbound to the next target)",
    "notdock_n": "onset while the spout moves, randomly drawn to the number of docked licks per session x gap stratum "
                 "(seed 0)",
}
ITI_REFIT_FILTERS = ("gap2", "far1", "gap2far1")
ITI_SUBSET_FILTERS = ("dock", "dock01", "gap2dock", "far1s", "notdock", "notdock_n")
#: what an ITI lick is, in the 3-D whole-session lick table (`iti_licks.lick_table`, context "iti")
ITI_CONTEXT = ("3-D whole-session lick table, context iti: after a trial's response window and before the next spout "
               "arrival; no cue")
ITI_LICK_WHAT = f"ITI licks ({ITI_CONTEXT})"
CUED_LICK_WHAT = "post-cue licks (3-D lick table, context post_cue: inside a trial's response window)"


def iti_filter() -> str:
    """The ITI lick filter this run applies (WFIELD_ITI_FILTER), "" for none / an unknown name (as the drivers read it)."""
    import os
    f = os.environ.get("WFIELD_ITI_FILTER", "")
    return f if f in ITI_FILTER_TEXT else ""


def iti_filter_text(name: str | None = None) -> str:
    """``"ITI filter <name>: <definition>"`` for ``name`` (default: the run's), "" when no filter applies."""
    name = iti_filter() if name is None else name
    if not name:
        return ""
    return f"ITI filter {name}: {ITI_FILTER_TEXT.get(name, name)}"


def events(what: str, *, keep=(), excluded=None, unit: str = "events") -> str:
    """THE event-set sentence: ``"<unit>: <what>; <keep 1>; <keep 2>; excluded: <excluded>"`` (empty parts dropped).
    ``keep`` = the inclusion rules beyond ``what`` (a filter, a window), each a phrase; ``excluded`` a phrase or list."""
    parts = [f"{unit}: {what}"] + [str(k) for k in (keep or ()) if k]
    if excluded:
        ex = excluded if isinstance(excluded, str) else ", ".join(str(e) for e in excluded if e)
        if ex:
            parts.append(f"excluded: {ex}")
    return "; ".join(parts)


def frames(what: str, *, keep=(), excluded=None) -> str:
    """A frame-set sentence (rest baselines, quiet periods): `events` with the unit "frames"."""
    return events(what, keep=keep, excluded=excluded, unit="frames")


def iti_licks(*, filt: str | None = None, extra=(), excluded=("post-cue (trial) licks",)) -> str:
    """Every ITI lick of the 3-D table (+ the run's ITI filter, or ``filt``; ``extra`` = the driver's own rules)."""
    return events(ITI_LICK_WHAT, keep=[iti_filter_text(filt), *extra], excluded=excluded)


def iti_bout_starts(gap_s: float, cue_excl=(-2.0, 0.75), *, filt: str | None = None, extra=()) -> str:
    """ITI BOUT-START licks (`iti_licks.extract` (A)): onset with no lick onset in the preceding ``gap_s`` and no cue /
    spout strobe within ``cue_excl`` of the onset."""
    a, b = (float(x) for x in cue_excl)
    return events(f"ITI bout-start licks ({ITI_CONTEXT})",
                  keep=[f"no lick onset in the preceding {float(gap_s):g} s",
                        f"no cue / spout strobe within [{a:+g}, {b:+g}] s of the onset", iti_filter_text(filt), *extra],
                  excluded=("later licks of a bout", "post-cue (trial) licks"))


def cued_licks(*, first_only: bool = False, extra=(), excluded=None) -> str:
    """Post-cue (trial) licks of the 3-D table; ``first_only`` = the first post-cue lick of each trial (lick_idx 0)."""
    what = CUED_LICK_WHAT if not first_only else CUED_LICK_WHAT.replace("post-cue licks", "FIRST post-cue lick of each "
                                                                        "trial (lick_idx 0)")
    return events(what, keep=extra, excluded=excluded if excluded is not None
                  else (["later licks of the trial"] if first_only else []) + ["ITI licks"])


def union(*sentences: str, unit: str = "events") -> str:
    """Two or more sets drawn together, e.g. ITI and cued licks: ``"events: A; ...  AND  B; ..."``."""
    body = [s.split(": ", 1)[1] if s.startswith(unit + ": ") else s for s in sentences if s]
    return f"{unit}: " + "  AND  ".join(body)


# ----------------------------------------------------------------------------------------------- the active set
#: A driver-level declaration (`declare`): the trial set of EVERY figure this process draws, unless a renderer
#: activated a narrower one (`activate` / `active`, a context variable) -- and visible from worker THREADS too, which
#: do not inherit context variables.
_PROCESS = None
#: `activate(None)` = "no trial set here", which also hides a process declaration (a replot of a bundle that named none)
_OFF = ("off",)


def activate(variant: str | None, align: str | None = None, *, text: str | None = None):
    """Declare the trial set the figures drawn from now on show; returns a token for `deactivate`. ``text`` = a
    literal sentence (a replot of a bundle that recorded one, or an event / frame set). ``variant`` None clears it."""
    if variant is not None and text is None and variant not in VARIANTS:
        raise ValueError(f"trial_set_text: unknown trial class {variant!r}")
    install_savefig_hook()
    return _ACTIVE.set(_OFF if (variant is None and text is None) else (variant, align, text))


def declare(variant: str | None = None, align: str | None = None, *, text: str | None = None) -> None:
    """THE DRIVER-LEVEL DECLARATION: every figure this process saves from now on carries this set (a narrower
    `activate` / `active` wins while it is open). ``declare()`` with nothing clears it. Covers figures saved by a plain
    ``fig.savefig`` / ``plt.savefig`` as well (`install_savefig_hook`)."""
    global _PROCESS
    if variant is not None and text is None and variant not in VARIANTS:
        raise ValueError(f"trial_set_text: unknown trial class {variant!r}")
    install_savefig_hook()
    _PROCESS = None if (variant is None and text is None) else (variant, align, text)


def deactivate(token=None) -> None:
    if token is not None:
        _ACTIVE.reset(token)
    else:
        _ACTIVE.set(None)


@contextlib.contextmanager
def active(variant: str | None, align: str | None = None, *, text: str | None = None):
    tok = activate(variant, align, text=text)
    try:
        yield
    finally:
        _ACTIVE.reset(tok)


def current():
    """``(variant, align, text)`` of the active trial set (the open `activate`, else the process `declare`), or None."""
    cur = _ACTIVE.get()
    if cur is _OFF:
        return None
    return cur if cur is not None else _PROCESS


def current_sentence() -> str | None:
    cur = current()
    if not cur:
        return None
    variant, _align, text = cur
    return text if text else sentence(variant)


# ----------------------------------------------------------------------------------------------- every savefig
_HOOKED = False


def install_savefig_hook() -> None:
    """Make ``matplotlib.figure.Figure.savefig`` stamp the active trial set (idempotent; installed by the first
    `activate` / `declare`). Drivers that write with a plain ``fig.savefig`` / ``plt.savefig`` are then covered by their
    one declaration; the stamp itself is idempotent (gid mark), so the shared save paths that stamp first are unchanged,
    and with no active set the hook does nothing (a PNG is byte-identical to a plain savefig)."""
    global _HOOKED
    if _HOOKED:
        return
    try:
        from matplotlib.figure import Figure
    except Exception:                                                  # noqa: BLE001 - no matplotlib, no figures
        return
    orig = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        if current() is not None:
            try:
                stamp(self, below=kwargs.get("bbox_inches") == "tight")
            except Exception:                                          # noqa: BLE001 - never cost the figure
                pass
        return orig(self, fname, *args, **kwargs)

    savefig.__wrapped__ = orig
    savefig.__doc__ = orig.__doc__
    Figure.savefig = savefig
    _HOOKED = True


# ----------------------------------------------------------------------------------------------- on the figure
def _wrap(txt: str, fig) -> str:
    try:
        w = float(fig.get_size_inches()[0])
    except Exception:                                                  # noqa: BLE001
        w = 8.0
    width = max(40, int(w * 72 / 3.3))                                  # ~3.3 pt per character at 6 pt (epoch_figures)
    # never split "jaw-only" / "first-last" at the hyphen: a wrapped caption must read back as the sentence
    return "\n".join(textwrap.wrap(txt, width, break_on_hyphens=False)) if len(txt) > width else txt


def stamp(fig, text: str | None = None, *, below: bool = False):
    """Write the active trial-set sentence (or ``text``) on ``fig``; returns the text artist, or None when there is
    nothing to say or the figure already carries it. Prepended to the run stamp (`lp_licks.stamp`, bottom right) when
    that is present, else its own bottom-right line in the same style; ``below`` puts a new line just under the canvas
    (for a ``bbox_inches="tight"`` save, which grows to include it instead of drawing over the axes)."""
    try:
        txt = text or current_sentence()
    except Exception as ex:                                            # noqa: BLE001 - a caption never kills a figure
        import warnings
        warnings.warn(f"[trial set] no caption ({type(ex).__name__}: {ex})", RuntimeWarning, stacklevel=2)
        return None
    if not txt:
        return None
    fig = getattr(fig, "figure", fig) if not hasattr(fig, "texts") else fig
    texts = list(getattr(fig, "texts", []) or [])
    if any(t.get_gid() == GID for t in texts):
        return None
    txt = _wrap(txt, fig)
    for t in texts:
        if t.get_color() == RUN_STAMP_COLOR:
            t.set_text(txt + "\n" + t.get_text())
            t.set_gid(GID)
            return t
    if below:
        return fig.text(0.995, -0.005, txt, ha="right", va="top", fontsize=6, color=RUN_STAMP_COLOR, gid=GID)
    return fig.text(0.995, 0.002, txt, ha="right", va="bottom", fontsize=6, color=RUN_STAMP_COLOR, gid=GID)
