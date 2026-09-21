"""THE GRANT RENDERER'S SHARED MACHINERY — plotting, saving, seeding, caching, labelling.

Split out of `grant_figures` on 2026-09-21. `grant_figures.py` was 6,599 lines and 133 functions;
these twenty are the ones that are not about any particular figure. Measured by call graph rather
than chosen by eye: they call NOTHING outside this module, and forty-five functions across every
figure family call INTO them.

  layout      `_suptitle`, `_fit_header`, `_fit_bottom`, `_twinned`, `_overlaps`, `_footer`
  output      `_save` -- PNG plus the vector copy, and the overlap report
  identity    `_windows`, `_variants`, `_day`, `_sessions`, `_sg_labels`, `_cd_labels`, `pos_style`
  determinism `_seed`, `_digest`, `_boot_cached`, `_feed`
  context     `coverage_note`, `_fig_root`

THE TWO RESTRICTION GLOBALS LIVE HERE NOW, AND THAT IS THE ONE THING THIS MOVE HAD TO GET RIGHT.
`_ONLY_WINDOW` and `_ONLY_VARIANT` are read by `_windows`/`_variants` and were assigned with
`global` from `main` and `_render_unit`. A module global assigned in one module and read in
another is two different variables: the readers would have seen a value nobody ever set, every
worker would have rendered EVERY alignment instead of the one unit it was handed, and nothing
would have raised -- the output would just have been wrong and slow. They are set through
`set_only()` now, which is one place and can be asserted; `tests/test_grant_kit.py` pins it.

WHY THEY ARE STILL MODULE STATE rather than parameters, which the original comment already
answered: every figure function loops over WINDOWS internally, and threading two arguments through
twenty-six of them would be a far larger and riskier change than a filter they all read through
one accessor.

DETERMINISM IS NOT INCIDENTAL HERE. `_seed` and `_digest` are why `_boot_cached` is sound at all --
a cached draw has to be the draw the uncached path would have produced, which was false while
seeds were `hash()`-salted and false again while one RNG stream was shared across an animal's
days. Both were fixed before the cache was added. See `_boot_cached`.
"""
from __future__ import annotations

import csv
import json
import os
import textwrap
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from wfield_local import config
from wfield_local.figure_layout import svg_path as _svg_path
from wfield_local.paths import PathResolver

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
POS = ["far_R", "far_center", "far_L", "close_R", "close_center", "close_L"]
#: THE COHORT PALETTE, from `spout_behavior.position_style`: hue = SIDE, lightness = RING.
#:
#: This dict used to be written out here as colour = RING (far reds, close blues) with a comment
#: claiming it was "the convention the behaviour figures already use". It was the exact opposite of
#: what those figures use, and the two sets of figures sit in the same deck. Derived now, so the
#: claim cannot go stale again (Priya, 2026-09-10).
#:
#: LAZY, because `spout_behavior` pulls the behaviour stack and this module is imported by the
#: parallel render workers; the same reason `position_coding_directions._pos_color` defers it.
@lru_cache(maxsize=1)
def pos_style() -> dict:
    from wfield_local.spout_behavior import position_style

    return {q: position_style(q) for q in POS}
WINDOWS = (("ENL", "precue", "ENL (pre-cue)"), ("cue", "cue", "post-cue"),
           ("lick", "lick", "post-lick"))


#: Restrict the render to ONE alignment / trial class. Set from `--window` / `--variant`, and the
#: reason they exist: `(figure, alignment, variant)` is the unit the parallel driver hands to a
#: worker, and a worker must be able to render exactly its own unit and nothing else.
#:
#: Module globals rather than threaded parameters because every figure function already loops over
#: WINDOWS internally, and adding two arguments to twenty-six of them would be a far larger and
#: riskier change than a filter they all read through one accessor.
_ONLY_WINDOW: str | None = None
_ONLY_VARIANT: str | None = None


#: Bumped when the BOOTSTRAP arithmetic changes. Separate from `session_cache.CACHE_VERSION`,
#: which covers the per-session features underneath: the two invalidate for different reasons, and
#: sharing one number would throw away hours of still-valid bootstraps every time a feature moved.
BOOT_CACHE_VERSION = 1


def _feed(h, a):
    """One array -- or a nested list/dict of them -- into a digest, shape and dtype included.

    Shape and dtype matter: two differently shaped arrays can share a byte string, and an int64
    block vector reinterpreted as float64 would collide with one that means something else.
    """
    if a is None:
        h.update(b"\x00none")
        return
    if isinstance(a, dict):
        for k in sorted(a, key=str):
            h.update(str(k).encode("utf-8"))
            _feed(h, a[k])
        return
    if isinstance(a, (list, tuple)):
        h.update(f"[{len(a)}]".encode("utf-8"))
        for x in a:
            _feed(h, x)
        return
    if isinstance(a, (str, int, float, bool)):
        h.update(repr(a).encode("utf-8"))
        return
    arr = np.ascontiguousarray(a)
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())


def _digest(*parts) -> str:
    """A content digest of what a bootstrap actually consumes.

    THE KEY IS THE DATA, NOT THE SESSION NAME. A name-keyed cache goes stale silently the moment a
    session is re-preprocessed under the same label -- the contamination class this repo keeps
    finding, most recently a frozen decoder that carried a stale basis for eight days behind a name
    asserting it did not. Hashing the bytes cannot do that: identical inputs give identical
    outputs, and changed inputs simply miss.
    """
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        _feed(h, p)
    return h.hexdigest()


def _boot_cached(tag, parts, compute):
    """Memoise one bootstrap result to disk under a digest of its inputs.

    Priya, 2026-08-28: store bootstrap results so a nightly run recomputes only what changed. That
    is what makes this worth having -- the 2026-08-28 render spent 94.7% of 5.79 hours in six
    bootstrap families, and on a typical night exactly one session is new. The other seventy-three
    have identical inputs and therefore identical draws.

    IT IS ONLY SOUND BECAUSE THE SEEDS ARE STABLE AND PER-DAY. A cached draw has to be the draw the
    uncached path would have produced. With `hash()`-salted seeds it never was, and with one RNG
    stream shared across an animal's days a cached day would silently depend on which other days
    were present in the run that produced it. Both were fixed first, deliberately, and neither is
    optional for this.

    Stored beside the session cache and honouring the same disable switch, so one environment
    variable turns off every memoisation at once when a result is under suspicion.
    """
    import pickle

    from wfield_local import session_cache as _sc

    if _sc._disabled():
        return compute()
    fp = _sc.CACHE_DIR / "bootstrap" / f"{tag}__v{BOOT_CACHE_VERSION}__{_digest(*parts)}.pkl"
    if fp.exists():
        try:
            with open(fp, "rb") as fh:
                return pickle.load(fh)
        except Exception:                                              # noqa: BLE001
            pass                    # truncated by a killed run -> recompute and republish
    res = compute()
    # A FALSY RESULT IS NEVER PERSISTED. Every producer here returns None or {} by way of a broad
    # `except Exception`, so an empty result is the signature of a FAILURE rather than an answer --
    # and memoising it makes one bad run permanent. Caught in the act: a NameError inside `_rdm_one`
    # was swallowed, None was pickled under twelve keys, and the CORRECTED code then read those back
    # and produced nothing, with no error either time. Recomputing an empty result costs one run;
    # caching it costs every run until someone thinks to look in the cache directory.
    if not res:
        return res
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(res, fh)
        os.replace(tmp, fp)
    except OSError:
        # A worker racing us to the same entry, or a full disk. The result is computed and correct;
        # failing a render over a cache write would be the wrong trade.
        pass
    return res


def _seed(*parts) -> int:
    """A STABLE integer seed from a tuple of labels.

    NOT `hash()`, which this module used at fourteen sites. Python SALTS string hashing per
    process unless PYTHONHASHSEED is set, and it is not set here -- three consecutive interpreters
    returned 1125027485, 2138950357 and 223190567 for `hash(("PS92", "cue", "lick"))`. So every
    bootstrap in this module drew a different resample on every run: the point estimates never
    moved, but every confidence interval did, and two renders of the same data could not be
    compared to each other. That is a reproducibility defect that predates any parallelism.

    It becomes unignorable with a worker pool, because each worker is its own process with its own
    salt -- but the fix is owed to the serial render just as much.

    blake2b of the joined labels is stable across processes, machines and Python versions.
    """
    import hashlib
    key = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def _windows():
    """The alignments this process should render. Iterate this, never WINDOWS directly."""
    return tuple(w for w in WINDOWS if _ONLY_WINDOW in (None, w[1]))


def _variants(align):
    """Trial classes for one alignment.

    THE LICK-ALIGNED WINDOW ADMITS ONLY ``lick``: a trial with no detected lick has no lick to
    align to, so a "miss trial, lick-aligned" panel is not a weak result but an undefined one.

    This rule was written out SEVENTEEN times as an inline conditional before it was a function.
    That is exactly the shape of duplication `_pooled_bundle` was extracted for -- seventeen copies
    agree today and one of them grows a third class tomorrow.
    """
    # `stopped` IS NOT A CLASS HERE ANY MORE (Priya, 2026-09-12: "I think we can get rid of the
    # post-stroke stopped vs pre-stroke lick analyses and figures"). It was added as a third class
    # so every generic family would render a stopped version, but `_collect_7` hardcodes the
    # PRE-STROKE side to `lick` for every class -- harmless for `lick` and `working`, which are
    # nearly the same trials before a lesion, and the whole comparison for `stopped`, which by
    # construction contains no licking trials at all. Those arms therefore scored post-stroke quit
    # periods against pre-stroke ENGAGED cortex and a drop was what two different behavioural
    # states would produce on their own.
    #
    # THE QUESTION IS STILL ASKED, PROPERLY, BY FAMILIES 12 AND 12b, which is why this deletes arms
    # and not the class. `_class_select`, `_session_trials(..., "stopped")`, `_collect_stopped` and
    # `_collect_stopped_pooled` all stay: 12b scores post-stroke stopped against pre-stroke STOPPED
    # (state-matched, leave-one-session-out) and 12 does the per-position version against a common
    # engaged template. Both render on the `working` pass, so neither is affected by this.
    vs = ("lick",) if align == "lick" else ("lick", "working")
    return tuple(v for v in vs if _ONLY_VARIANT in (None, v))


def coverage_note(source_labels=None):
    """Footer for one figure. `source_labels` = the post-stroke sessions THAT FIGURE actually used.

    READS THE SOURCE, NOT THE CONFIG, and the difference is not academic. The first version counted
    `config.phase_labels("post")` on every figure. That is correct for the figures that recompute
    from the pooled data, and FALSE for the ones built from `section_g.json` /
    `coding_direction.json`, which are written by a nightly and can lag the config by days. On
    2026-08-25 those JSONs predated both 8/24 sessions while the config had them, so the footer
    printed a reassuring "6, 6, 6, 6" on exactly the three figures that were stale — a check giving
    false comfort about the thing it was built to catch (Priya: "fix it please").

    Pass the labels the figure really used and this compares them to the config, reporting the lag.
    Called with nothing, it falls back to the config and says so.
    """
    cfg = {a: {x for x in config.phase_labels("post") if x.startswith(a)} for a in ANIMALS}
    if source_labels is None:
        used = cfg
        src = "from the session config"
    else:
        keep = set(source_labels)
        used = {a: {x for x in keep if x.startswith(a)} for a in ANIMALS}
        src = "as actually present in the data this figure was built from"
    counts = {a: len(used[a]) for a in ANIMALS}
    note = (f"Post-stroke sessions used ({src}): "
            + ",  ".join(f"{a} {counts[a]}" for a in ANIMALS))
    if len(set(counts.values())) > 1:
        note += ("   —  UNEQUAL: an animal with fewer sessions contributes less to every pooled "
                 "panel.")
    missing = sorted({x for a in ANIMALS for x in cfg[a] - used[a]})
    if missing:
        note += ("   —  STALE: registered but ABSENT here: " + ", ".join(missing)
                 + ". Re-run the analysis that writes this figure's source.")
    return note


def _sg_labels():
    """Post-stroke sessions actually present in `section_g.json` (figures 3b and 5)."""
    f = _fig_root() / "section_g.json"
    if not f.exists():
        return []
    return [k for k in json.loads(f.read_text(encoding="utf-8"))
            if config.session_phase(k.split("_")[0], k.split("_")[-1]) == "post"]


def _cd_labels():
    """Post-stroke sessions actually present in `coding_direction.json` (figure 3a).

    Read from the per-session store of whichever window/animal the file has, since every window
    carries the same session set.
    """
    f = _fig_root() / "coding_direction.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text(encoding="utf-8"))
    out = set()
    for res in (d.get("ENL") or {}).values():
        if not res:
            continue
        for meth in res.get("methods", {}).values():
            for cls in (meth.get("cross_by_session") or {}).values():
                out |= set(cls)
    return sorted(out)


def _suptitle(fig, text, fontsize=9.5, width=150):
    """Wrap a long header, then reserve vertical room PROPORTIONAL TO ITS LINE COUNT.

    Matplotlib anchors a suptitle near y=0.98 and grows it DOWNWARD, so a header gains lines at the
    expense of the top row of panels. Every figure here reserved a hand-tuned constant instead
    (``rect=(0, 0, 1, 0.88)`` and friends), which held only while the text did -- and the moment the
    bootstrap description was added, several headers overlapped their own figures (Priya,
    2026-08-26).

    The reservation is computed from the figure's ACTUAL height in inches, because these range from
    4.5 to 15.5: a five-line header costs a third of a 4.5-inch figure and a tenth of a 15-inch one,
    so a single fraction cannot serve both.

    Call this AFTER any ``tight_layout``: the adjustment is applied with ``subplots_adjust``, which
    is a one-shot override of what tight_layout computed and would otherwise be recomputed away.
    """
    text = "\n".join(textwrap.fill(ln, width=width) if len(ln) > width else ln
                     for ln in str(text).split("\n"))
    n_lines = text.count("\n") + 1
    line_frac = (fontsize * 1.5) / (fig.get_figheight() * 72.0)     # points -> figure fraction
    top = min(0.97, max(0.45, 1.0 - (n_lines * line_frac + 0.015)))
    # COMPRESS EVERY AXES, not just the subplot grid. `subplots_adjust` moves only axes that belong
    # to the gridspec, so a colour bar placed by `fig.colorbar(ax=...)` -- which fixes its position
    # from the panel geometry AT THE MOMENT IT IS CREATED -- stayed put while the panels moved under
    # it, and ended up drawn over them. That is the fault reported in figures 4 and 5, and it would
    # recur in any figure that made a colour bar before its header.
    #
    # Scaling y into [0, top] preserves the relative layout of everything, including colour bars and
    # any manually added axes, and needs no knowledge of which is which.
    def _apply(frac):
        for _ax in fig.axes:
            _pos = _ax.get_position()
            _ax.set_position([_pos.x0, _pos.y0 * frac, _pos.width, _pos.height * frac])

    _apply(top)
    # NOTE the bare matplotlib call: the sweep that routed every `fig.suptitle` in this module
    # through `_suptitle` rewrote this line too, making the helper call itself. Caught by reading
    # the diff rather than by a test, which would have hit a RecursionError at render time.
    Figure.suptitle(fig, text, fontsize=fontsize, y=0.997, va="top")

    _fit_header(fig)
    return top


def _fit_header(fig):
    """Shrink the axes until no PANEL TITLE reaches into the header. Idempotent.

    A panel title sits ABOVE its axes, so the reservation `_suptitle` computes from the header's own
    line count does not account for it -- figure 3a's four panel titles landed under the header even
    though the axes cleared it.

    CALLED AGAIN FROM `_save`, and that is the point. Six figures call `tight_layout` AFTER their
    header, which recomputes every position and discards the reservation; reordering all six would
    fix them and would not stop the seventh. Re-fitting at save time is ordering-independent, and
    idempotent because each pass shrinks by the measured shortfall and stops once there is none.
    """
    if getattr(fig, "_suptitle", None) is None:
        return
    try:
        for _ in range(3):
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            inv = fig.transFigure.inverted()
            sup_y0 = inv.transform_bbox(fig._suptitle.get_window_extent(rend)).y0
            tops = [inv.transform_bbox(a.title.get_window_extent(rend)).y1
                    for a in fig.axes if a.get_visible() and str(a.title.get_text()).strip()]
            if not tops or max(tops) <= sup_y0 - 1e-4:
                return
            frac = max(0.5, 1.0 - (max(tops) - sup_y0) - 0.004)
            for ax in fig.axes:
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0 * frac, pos.width, pos.height * frac])
    except Exception as exc:                                         # noqa: BLE001
        # A layout refinement must never fail a render -- but it should say so, or a figure that
        # silently skipped it looks identical to one that did not need it.
        print(f"  [layout] header refinement skipped ({type(exc).__name__})", flush=True)


def _fit_bottom(fig, pad=0.008):
    """Raise the axes until a bottom-anchored figure legend clears the x labels.

    `tight_layout(rect=(0, bottom, ...))` cannot do this: `_fit_header` rescales every axes AFTER
    it, so the reserved band is recomputed away -- figure 8g's axes bottom came out at 0.097 whether
    the rect asked for 0.10 or 0.18, and the legend sat squarely on the bottom row's "days from
    lesion". A figure legend is not an axes, so it does not move with them either.

    Measure instead: if the legend intrudes on the lowest x label, shift every axes up by exactly
    the shortfall and take it out of their height.
    """
    if not fig.legends:
        return
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        lg = max((inv.transform_bbox(g.get_window_extent(rend)) for g in fig.legends),
                 key=lambda b: b.y1)
        xls = [inv.transform_bbox(a.xaxis.label.get_window_extent(rend))
               for a in fig.axes if a.get_visible() and str(a.xaxis.label.get_text()).strip()]
        if not xls:
            return
        lo = min(b.y0 for b in xls)
        shift = (lg.y1 + pad) - lo
        if shift <= 0:
            return
        for ax in fig.axes:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 + shift, pos.width, max(0.02, pos.height - shift)])
    except Exception as exc:                                         # noqa: BLE001
        print(f"  [layout] bottom fit skipped ({type(exc).__name__})", flush=True)


def _twinned(axa, axb):
    """True when two axes are a twinx/twiny pair, which overlay each other BY CONSTRUCTION.

    Figure 10 puts a second y-scale (mean rank) on its trend panel, and the axes check reported all
    four of them as faults. A twin having its parent's rectangle is the entire point of it, and a
    checker that cries wolf trains the reader to ignore it -- which is worse than not checking.

    BOTH CONDITIONS ARE REQUIRED. Sharing an axis is not enough: `plt.subplots(sharex=True)` puts
    every panel of figure 8g in one shared group, so testing siblings alone would silently disable
    the overlap check for a whole figure. A twin also has the SAME rectangle, and panels of a shared
    grid do not.
    """
    pa, pb = axa.get_position(), axb.get_position()
    if max(abs(pa.x0 - pb.x0), abs(pa.y0 - pb.y0),
           abs(pa.x1 - pb.x1), abs(pa.y1 - pb.y1)) > 1e-6:
        return False
    for getter in ("get_shared_x_axes", "get_shared_y_axes"):
        try:
            if axb in getattr(axa, getter)().get_siblings(axa):
                return True
        except Exception:                                    # noqa: BLE001,S112  matplotlib version
            continue                                         # differences in the sharing API only
    return False


def _overlaps(fig):
    """Intersecting pairs among a figure's CHROME: suptitle, figure texts, legends, colour-bar
    labels, axis labels and panel titles. Tick labels are excluded -- they sit close to their own
    axis by design.

    Returns [(name_a, name_b, area)], largest first.
    """
    import itertools as _it
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    items = []

    def add(artist, name):
        if artist is None:
            return
        if hasattr(artist, "get_text") and not str(artist.get_text()).strip():
            return
        try:
            bb = inv.transform_bbox(artist.get_window_extent(rend))
        except Exception:                                            # noqa: BLE001
            return
        if bb.width > 0 and bb.height > 0:
            items.append((name, bb))

    owner = {}

    def own(name, idx):
        owner[name] = idx

    sup = getattr(fig, "_suptitle", None)
    add(sup, "suptitle")
    for t in fig.texts:
        if t is not sup:
            add(t, f"text:{str(t.get_text())[:18]}")
    for lg in fig.legends:
        add(lg, "legend")
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        for artist, nm in ((ax.xaxis.label, f"ax{i}.xlabel"),
                           (ax.yaxis.label, f"ax{i}.ylabel"),
                           (ax.title, f"ax{i}.title")):
            add(artist, nm)
            own(nm, i)
    # AXES RECTANGLES TOO, not only text. A colour bar drawn ON TOP of a panel carries no label that
    # would collide, so a chrome-only check reports nothing while the figure is plainly wrong -- and
    # a colour bar over a panel is the single most common fault in this module's history. Axes with
    # no content are skipped: the delta grids reserve an empty spacer column on purpose.
    drawn = [(i, ax) for i, ax in enumerate(fig.axes)
             if ax.get_visible() and (ax.images or ax.lines or ax.patches or ax.collections)]
    bad = []
    for (ia, axa), (ib, axb) in _it.combinations(drawn, 2):
        pa, pb = axa.get_position(), axb.get_position()
        ox = min(pa.x1, pb.x1) - max(pa.x0, pb.x0)
        oy = min(pa.y1, pb.y1) - max(pa.y0, pb.y0)
        if ox > 1e-3 and oy > 1e-3 and not _twinned(axa, axb):
            bad.append((f"AXES ax{ia}", f"AXES ax{ib}", ox * oy))

    for a, b in _it.combinations(range(len(items)), 2):
        (na, ba), (nb, bb) = items[a], items[b]
        ox = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
        oy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
        if ox > 1e-4 and oy > 1e-4:
            bad.append((na, nb, ox * oy))

    # TICK LABELS AGAINST EACH OTHER, on the same axis. They are excluded from the checks above
    # because they legitimately sit close to their own axis and to its neighbours' -- but "cL cC cR"
    # running together into "cLcCcR" is a real fault and was visible in figure 5c while this
    # function reported the figure clean.
    for i, ax in drawn:
        for getter, which in ((ax.get_xticklabels, "x"), (ax.get_yticklabels, "y")):
            labs = [t for t in getter() if str(t.get_text()).strip()]
            boxes = []
            for t in labs:
                try:
                    boxes.append((t.get_text(), inv.transform_bbox(t.get_window_extent(rend))))
                except Exception as exc:                             # noqa: BLE001
                    print(f"  [layout] tick extent unavailable ({type(exc).__name__})", flush=True)
            boxes.sort(key=lambda r: (r[1].x0, r[1].y0))
            for (t1, b1), (t2, b2) in _it.pairwise(boxes):
                ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                if oy <= 1e-4:
                    continue
                # A MINIMUM GAP, not merely "not intersecting". "cL cC cR" with a hairline between
                # them reads as "cLcCcR" -- which is what figure 5c did while a pure overlap test
                # passed it -- and the whole point of these labels is to survive a reduction to a
                # quarter page, where a hairline gap closes completely.
                need = 0.18 * max(1e-9, (b1.width + b2.width) / 2)
                if ox > -need:
                    bad.append((f"ax{i}.{which}tick {t1!r}", f"crowds {t2!r}", ox + need))

    # TEXT OVER SOMEONE ELSE'S PANEL. Checking text-vs-text and axes-vs-axes leaves the commonest
    # crowding fault invisible: a two-line panel title printed across the BOTTOM ROW OF CELLS of the
    # panel above it. That is what figure 5 did while this function reported it clean. A title over
    # its OWN axes is normal and excluded.
    for name, box in items:
        src = owner.get(name)
        for i, ax in drawn:
            if src == i:
                continue
            pos = ax.get_position()
            ox = min(box.x1, pos.x1) - max(box.x0, pos.x0)
            oy = min(box.y1, pos.y1) - max(box.y0, pos.y0)
            if ox > 1e-3 and oy > 1e-3:
                bad.append((name, f"over AXES ax{i}", ox * oy))
    return sorted(bad, key=lambda r: -r[2])


def _save(fig, path, **kw):
    """Save, and REPORT any chrome overlap on the way out.

    Layout faults in this module have been found by eye, one at a time, each fix pushing the problem
    onto a neighbour -- the reference colour bar alone collided with three different things. A render
    that names its own overlaps turns that into a list, and the list is checkable after every run
    instead of after every complaint.

    It warns rather than raises: a figure with a cosmetic collision is still worth having, and
    failing the render would lose the other twenty.
    """
    # RE-FIT THE HEADER FIRST. Six figures call tight_layout after their header, which discards the
    # reservation `_suptitle` made; this is the last point before the file is written, so it is the
    # one place the fix cannot be undone by call order.
    _fit_header(fig)
    _fit_bottom(fig)
    try:
        bad = _overlaps(fig)
    except Exception as ex:                                          # noqa: BLE001
        bad = []
        print(f"  [layout] {Path(path).name}: check failed ({type(ex).__name__})", flush=True)
    for na, nb, _area in bad[:6]:
        print(f"  [layout] {Path(path).name}: {na} x {nb}", flush=True)
    # UNBOUND CALL, for the same reason `_suptitle` uses `Figure.suptitle`: the sweep that routed
    # every `fig.savefig(` in this module through `_save` rewrote this line too, and a helper that
    # calls itself surfaces as a RecursionError on the first figure of a two-hour render. Second
    # time this exact trap has been sprung by a mechanical sweep over a file containing its own
    # definition -- hence the explicit form rather than a comment asking the next person to be
    # careful.
    Figure.savefig(fig, path, **kw)
    # Also emit a vector SVG beside the PNG (Priya, 2026-08-29). The deck places the PNGs, so those
    # stay; the SVGs are for editing / print. UNBOUND call for the same self-reference reason as above,
    # and it WARNS rather than raises so an un-vectorisable figure never loses the PNG or its neighbours.
    _p = Path(path)
    if _p.suffix.lower() == ".png":
        try:
            Figure.savefig(fig, _svg_path(_p),
                           **{k: v for k, v in kw.items() if k != "dpi"})   # dpi is a no-op for vector
        except Exception as ex:                                            # noqa: BLE001
            print(f"  [svg] {_p.name}: failed ({type(ex).__name__})", flush=True)
    return path


def _footer(fig, source_labels=None):
    """Stamp session coverage on a figure. Pass the sessions the figure's SOURCE actually contains.

    Figures that recompute from the pooled data can pass None (their source IS the config). Figures
    built from a nightly-written JSON must pass that JSON's session list, or the footer reassures
    about data it never saw -- which is exactly how it failed on 2026-08-25.
    """
    # THE ABBREVIATION KEY RIDES WITH THE FIGURE. Axis labels are two characters so they survive a
    # reduction to a quarter page, which is only legible if the expansion travels with them -- a key
    # that lives in the speaker notes is not present when the panel is lifted into a grant.
    fig.text(0.5, 0.004,
             coverage_note(source_labels)
             + "      positions: c = close, f = far;  L / C / R = left / centre / right",
             ha="center", va="bottom", fontsize=7, color="0.30")


def _fig_root():
    """Where the analysis figures/JSONs live, from the config -- NOT a literal.

    `E:/cue_lick` is the analysis box's path and would be wrong on the imaging box;
    tests/test_no_hardcoded_machine_paths.py fails the build for exactly this, and did.
    """
    return Path(PathResolver().root("figures_working"))


def _day(animal, mmdd):
    """Days from that animal's OWN lesion date. Negative = pre-stroke.

    `config.stroke_date` returns MMDD ('0817'), not YYYYMMDD -- taking it for the longer form
    silently yields an empty slice and an int() crash. Month*31 is a within-year ordering, not a
    calendar difference; it is monotone and that is all the x-axis needs, but do not read a gap of
    "31" as a month.
    """
    def ord_(s):
        return int(s[:2]) * 31 + int(s[2:])
    return ord_(mmdd) - ord_(str(config.stroke_date(animal)))


def _sessions(animal, phases=("pre", "post")):
    """(mmdd, day) for this animal's registered sessions, EXCLUDED ones dropped."""
    out = []
    for s in config.load_sessions():
        lab = s["label"]
        if not lab.startswith(animal):
            continue
        mmdd = lab.split("_")[1]
        ph = config.session_phase(animal, mmdd)
        if ph in phases:
            out.append((mmdd, _day(animal, mmdd)))
    return sorted(set(out), key=lambda t: t[1])


def set_only(window=None, variant=None):
    """Restrict this PROCESS to one alignment / trial class.

    THE ONLY WAY THESE ARE ASSIGNED. `main` and `_render_unit` used `global` from inside
    `grant_figures`, which stopped working the moment the readers moved here -- a global assigned
    in one module and read in another is two variables, and the failure is silent: every unit
    renders every alignment and the output is wrong rather than absent.
    """
    global _ONLY_WINDOW, _ONLY_VARIANT
    _ONLY_WINDOW, _ONLY_VARIANT = window, variant


def only() -> tuple:
    """(window, variant) currently in force. For tests and for the driver's own logging."""
    return _ONLY_WINDOW, _ONLY_VARIANT


# ============================================================================================
# SHARED ACROSS FIGURE GROUPS (moved up 2026-09-21, ahead of the family split)
#
# Each of these is reached from MORE THAN ONE of the six figure families. A helper used by one
# family travels with it; one used across families has to live above all of them, or importing
# any family would drag in a sibling and the split would be a cycle wearing a different name.
# Which is which was computed from the call graph, not chosen by reading the names.
#
# `_BUNDLE_CACHE` travels with `_pooled_bundle`, its only user. A module-level cache separated
# from the function that fills it is two caches, and the one nobody writes to always misses.
# ============================================================================================
# ------------------------------------------------------------------ 1. behaviour
def _position_metrics(animal, mmdd):
    """{position: (hit_rate, ci_lo, ci_hi, n_engaged)} from the behaviour per-session CSV."""
    root = Path(PathResolver().root("behavior_out")) / "sessions" / animal / f"2026{mmdd}"
    if not root.exists():
        return {}
    files = sorted(root.glob("*position_metrics.csv"))
    if not files:
        return {}
    out = {}
    with files[-1].open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[r["pos_name"]] = (float(r["hit_rate"]), float(r["ci_lo"]), float(r["ci_hi"]),
                                      int(r["trials_engaged"]))
            except (ValueError, KeyError):
                continue
    return out
#: The confusion matrices are stored in DISPLAY_ORDER -- the spatial layout of the spouts
#: (left-to-right, close row then far row), not the raw position codes. Labelling them in code order
#: would transpose the picture into nonsense while still looking like a plausible matrix.
CONF_LABELS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]
#: Two-character position labels for dense axes. A 6-column matrix panel is ~2.05in wide, so a cell
#: is ~0.34in; "close_center" is twelve characters and cannot be enlarged without collision, while
#: "cC" can. The full names stay in every header, legend and speaker note, and the key is one line:
#: c = close, f = far; L / C / R = left / centre / right.
POS_SHORT = {"close_L": "cL", "close_center": "cC", "close_R": "cR",
             "far_L": "fL", "far_center": "fC", "far_R": "fR"}
def _colw(full=1.80):
    """Inches per matrix column.

    MEASURED, not chosen. Six rotated two-character tick labels set the floor: at 1.45 the
    panels come out 0.911in and the labels crowd -- by a hairline, but a hairline closes
    completely when the figure is reproduced small. The value has been raised twice as
    post-stroke days accumulated, because the per-column width shrinks as columns are added:
    the additive margin constant in each grid's figsize is diluted by matplotlib's
    FRACTIONAL default margins, so more sessions means less width each.

    THE COMPACT VARIANT IS GONE (2026-08-28) and this is why. It was built on the assumption
    that the in-cell numbers forced the panels wide; measuring said otherwise -- the TICK
    LABELS set the floor and are present in both variants, so compact reached 13.2in against
    full's 13.6in, a 3% saving for a second full render pass. Priya: "just get rid of
    compact grant figures."
    """
    return full
def _txt(ax, *args, **kw):
    """`ax.text`, kept as a seam.

    It existed so a `--compact` render could drop every in-cell number from one place
    rather than from eight call sites. That variant is gone (2026-08-28: measured at 3%
    narrower for a second full render pass), but the indirection stays: eight call sites
    routed through one function is how the next global change to in-cell text stays a
    one-line change instead of a sweep that misses one.
    """
    return ax.text(*args, **kw)
def _out(out_dir, stem):
    """Output path for a figure stem."""
    return Path(out_dir) / f"{stem}.png"
def _short(labels):
    """Position labels shortened for an axis. Anything unrecognised passes through unchanged."""
    return [POS_SHORT.get(str(q), str(q)) for q in labels]
#: One bundle per (animal, alignment), reused across every figure that needs it.
#:
#: MEASURED 2026-08-26: `_collect_7` appears 14 times in this module and loops over 4 animals, so a
#: full render built this bundle 56 times -- while only 4 animals x 3 alignments = 12 are distinct.
#: Each build loads the joint basis, PROJECTS every one of ~18 sessions onto it, and re-derives the
#: engagement gate. That is the dominant cost of figures 6, 6b, 6d, 7, 7b, 7d, 8, 8b, 8d, 8e, and it
#: is the same work every time: nothing between two calls can change it within one process.
#:
#: In-process rather than on disk, deliberately. The bundle holds the pooled feature matrices for
#: every session, so persisting it would write hundreds of MB per (animal, align) and invite exactly
#: the staleness question this session has spent all day on. A render is one process, so an
#: in-process memo captures the entire duplication with none of that.
_BUNDLE_CACHE: dict = {}
def _pooled_bundle(an, align):
    """The shared load behind figures 6, 6b, 7 and 8: joint basis, pooled sessions, engagement gate.

    Extracted because it was character-identical in `fig_pattern_similarity` and
    `fig_pattern_similarity_per_session`, and a third and fourth copy is how two figures that claim
    to describe the same trials quietly stop doing so. Memoized per (animal, alignment) for the same
    reason it was extracted: two figures that claim to describe the same trials should not be able to
    disagree, and now they cannot even in principle -- they hold the same object.
    """
    key = (an, align)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]
    from wfield_local import joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
    from wfield_local.locanmf_frozen_decoder import pool_sessions
    from wfield_local.position_coding_directions import _gate_all
    from wfield_local.precue_engagement_states import features_with_indices

    pre = [x for x in config.phase_labels("pre") if x.startswith(an)]
    post = [x for x in config.phase_labels("post") if x.startswith(an)]
    basis = joint_locanmf.load(an, sessions=SESSIONS)
    feat = features_with_indices(basis, nolick_ref="cue")
    XE, YE, GE, BE, XU, YU, kept, _c, GU = pool_sessions(
        pre + post, source="locanmf", align=align, post_s=2.0, features=feat)
    g = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
    not_eng = g[0] if g else np.zeros(len(YU), bool)
    pre_i = {i for i, lab in enumerate(kept) if lab in set(pre)}
    GU = np.asarray(GU)
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
    un = (np.array([POSITION_NAMES.get(int(v), str(v)) for v in YU])
          if len(YU) else np.zeros(0, str))
    # BLOCK IDS, for the block bootstrap. `pool_sessions` returns BE as a LIST of per-session
    # vectors in the same order it stacks XE, so concatenating aligns them row for row. A block is
    # a run of trials at ONE position, ended by a position change or the scheduler's block_size_max
    # (locanmf_position_decoder, audited against the firmware's own block_number to 2.8%).
    BE_all = np.concatenate([np.asarray(b) for b in BE]) if len(BE) else np.zeros(0, int)
    # Make ids unique ACROSS sessions -- they restart per session and a bootstrap that pooled two
    # sessions' block 3 would resample a unit that does not exist.
    BE_all = np.asarray(GE, dtype=np.int64) * 1_000_000 + BE_all.astype(np.int64)
    # THE NO-LICK ARM HAS NO BLOCK IDS: pool_sessions does not return them for XU. They are
    # reconstructed by the same rule minus the size cap -- a new block wherever the position
    # changes in that session's trial order. Coarser than the real blocks, never finer, so it
    # cannot make the intervals too narrow.
    BU_all = _runs_to_blocks(np.asarray(GU), un) if len(YU) else np.zeros(0, np.int64)
    bundle = {"XE": XE, "en": en, "GE": np.asarray(GE), "XU": XU, "un": un, "GU": GU,
              # The NUMERIC labels as well as the names. `pool_sessions` returns them and
              # this used to discard them, which is the only reason two callers could not
              # adopt the bundle: they fit and score on the integer codes, not the names.
              "YE": YE, "YU": YU,
              "BE": BE_all, "BU": BU_all,
              "not_eng": not_eng, "kept": kept, "pre_i": pre_i,
              "e_pre": np.isin(np.asarray(GE), list(pre_i))}
    _BUNDLE_CACHE[key] = bundle
    return bundle
def _runs_to_blocks(sess, pos):
    """Block ids from runs of the same (session, position), for trials that carry none of their own.

    Coarser than the scheduler's real blocks -- it misses the size cap that splits a long run in two
    -- and never finer. A too-coarse block resamples larger correlated chunks, which WIDENS a
    bootstrap interval; a too-fine one would narrow it. Erring wide is the safe direction.
    """
    sess, pos = np.asarray(sess), np.asarray(pos)
    if not len(sess):
        return np.zeros(0, np.int64)
    changed = np.ones(len(sess), bool)
    changed[1:] = (sess[1:] != sess[:-1]) | (pos[1:] != pos[:-1])
    # negative ids so they can never collide with the real BE ids, which are non-negative
    return -(np.cumsum(changed).astype(np.int64) + 1)
def _class_select(variant, sess_e, sess_u, not_eng):
    """Which ENGAGED and which UNENGAGED rows belong to a trial class, as two boolean masks.

    THE THREE COPIES OF THIS RULE DISAGREED THE MOMENT A THIRD CLASS EXISTED. `lick`, `working` and
    `stopped` were each written inline as ``if v == "working" and len(un)`` at three collector sites
    plus `_session_trials`, and every one of them included the ENGAGED rows unconditionally -- which
    is right for the first two classes and catastrophically wrong for `stopped`, where it would
    silently fold every licking trial of the session into a set defined as "the animal had quit".
    One function, so that cannot happen at one site and not the others.

        lick     engaged only
        working  engaged + unengaged OUTSIDE the terminal quit period
        stopped  the terminal quit period ALONE, and NO engaged rows

    `stopped` and `working` partition the unengaged trials and `stopped` takes none of the engaged
    ones, so the three classes are not nested: a trial is in `working` or in `stopped`, never both.
    """
    import numpy as _np
    e_none = _np.zeros(len(sess_e), bool)
    u_none = _np.zeros(len(sess_u), bool)
    if variant == "stopped":
        return e_none, (sess_u & not_eng) if len(sess_u) else u_none
    if variant == "working":
        return sess_e, (sess_u & ~not_eng) if len(sess_u) else u_none
    return sess_e, u_none
#: How a trial class reads in a figure caption. One place, because fifteen copies of a two-branch
#: conditional cannot survive a third branch being added -- every one of them captioned `stopped`
#: as "LICK + miss-while-working", i.e. as its own complement.
#:
#: THE CLASS GATES THE POST-STROKE SIDE ONLY. `_collect_7` hardcodes the pre-stroke reference to
#: `lick` for every class, so the note has to say so or the caption describes half a correlation.
#: For `lick` and `working` that is a distinction without a difference -- pre-stroke those two sets
#: are nearly identical (a typical session is 305 lick against 312 working, because the pre-stroke
#: animal is not missing). For `stopped` it is the whole comparison: `_class_select` gives `stopped`
#: NO engaged rows, so the arm scores post-stroke quit-period patterns against pre-stroke ENGAGED
#: ones and a drop is what two different behavioural states would produce on their own.
#:
#: A symmetric reference is not available AT THIS RESOLUTION. Only 3 of 44 pre-stroke sessions clear
#: min_trials=10 at all six positions (PS94_0806, PS95_0806, PS95_0812) against 20 of 48 post; PS92
#: has 6 stopped trials in its entire pre-stroke set and PS93 has one session at 1/6. Pre-stroke
#: animals rarely quit, which is exactly why the class is interesting after stroke.
#:
#: THE STATE-MATCHED CONTRAST EXISTS -- IT IS JUST NOT HERE. Pooling the positions away drops the
#: floor to 20 trials per SESSION and PS93, PS94 and PS95 all clear it, which is what
#: `_collect_stopped_pooled` and family 12b are for: post-stroke stopped against pre-stroke STOPPED,
#: state-matched on both sides, leave-one-session-out on the pre bar. Family 12 is the per-position
#: version with the pre-stroke stopped column as an explicit "quitting alone" control. So do not
#: read these generic class arms as the stopped-trial ANALYSIS -- they are the engaged-trial
#: families with the class switch flipped, and 12/12b are the purpose-built ones.
def _class_note(variant):
    return {"lick": "LICK trials only",
            "working": "LICK + miss-while-working (quit period removed)",
            "stopped": ("THE TERMINAL QUIT PERIOD ONLY -- no licking trials; PRE-STROKE REFERENCE "
                        "IS LICK TRIALS -- for the state-matched contrast see family 12b")}.get(
                variant, str(variant))
def _session_trials(bd, i, q, variant, field="X"):
    """Trials (or their BLOCK IDS) for session ``i`` at position ``q`` under a trial class.

    ``lick`` is the engaged (licking) set; ``working`` adds miss-while-working, i.e. everything but
    the terminal quit period; ``stopped`` is that quit period ALONE. Returns an empty array rather
    than None so callers can stack freely.

    ``field`` selects what comes back -- "X" the patterns, "blk" the block id of each of those same
    rows. THE MASK IS COMPUTED ONCE HERE for both, so the two cannot drift apart; a bootstrap whose
    block vector did not line up with its data would silently resample the wrong trials.
    """
    # ``stopped`` IS THE COMPLEMENT OF EVERY OTHER CLASS: the terminal quit period ONLY, and no
    # licking trials at all. Every other class here is `~not_eng` and this is `not_eng`, so a trial
    # belongs to `stopped` or to `working` and never to both. It exists because "the animal stopped"
    # is a behavioural state the imaging can be asked about (Priya, 2026-09-11) and the gate has so
    # far only ever been used to THROW those trials away.
    me, mu = _class_select(variant, (bd["GE"] == i) & (bd["en"] == q),
                           (bd["GU"] == i) & (bd["un"] == q) if len(bd["un"])
                           else np.zeros(0, bool), bd["not_eng"])
    if field == "blk":
        parts = [bd["BE"][me]] + ([bd["BU"][mu]] if mu.any() else [])
        keep = [z for z in parts if len(z)]
        return np.concatenate(keep) if keep else np.zeros(0, np.int64)
    parts = [bd["XE"][me]] + ([bd["XU"][mu]] if mu.any() else [])
    keep = [z for z in parts if len(z)]
    return np.vstack(keep) if keep else np.zeros((0, bd["XE"].shape[1]))
#: Figures 7, 7b, 8 and 8b all need the SAME per-animal trial collection, and loading it means
#: reading every session's LocaNMF fit. Six entries covers every (window, class) this module builds,
#: so a full render loads each one once instead of four times. The features are LocaNMF components,
#: not pixels, so the whole cache is ~100 MB.
@lru_cache(maxsize=12)
def _collect_7(align, variant, min_trials, field="X"):
    """(per-animal) pre-stroke reference/other halves + per-day post trials, kept AS TRIALS.

    CACHED, so callers must treat the result as read-only -- mutating it would corrupt every later
    figure in the same process.

    PRE-STROKE SESSIONS ARE KEPT SEPARATE, not pooled, and that is the whole point of this
    collector. The first render of figure 7 compared the split-half reliability of the POOLED
    pre-stroke set (six sessions) against one post-stroke session at a time and showed 0.72-0.92
    against 0.14-0.67. Split-half reliability rises with trial count, so most of that gap was six
    times the trials -- and it is exactly the comparison the figure invites and exactly the question
    ("is the lost code just a noisier one?") it exists to answer. Keeping the sessions apart lets
    every caller build a LEAVE-ONE-SESSION-OUT reference that is both trial-count-matched and
    disjoint from the session being scored.
    """
    out, all_days = {}, set()
    for an in ANIMALS:
        try:
            bd = _pooled_bundle(an, align)
        except Exception as ex:                                          # noqa: BLE001
            print(f"  !! 7 {an} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
            continue
        pre_by_sess, by_day = {}, {}
        for i, lab in enumerate(bd["kept"]):
            mmdd = lab.split("_")[-1]
            if i in bd["pre_i"]:
                # PRE-STROKE trials are the LICKING set in every class: the pre-stroke animal is
                # not missing, so "working" would add nothing and would silently make the reference
                # a different kind of trial from itself.
                pat = {q: _session_trials(bd, i, q, "lick", field) for q in CONF_LABELS
                       if len(_session_trials(bd, i, q, "lick")) >= min_trials}
                if pat:
                    pre_by_sess[mmdd] = pat
                continue
            day = _day(an, mmdd)
            # THE GATE IS ALWAYS ON THE TRIAL COUNT, never on the length of `field`, so the "blk"
            # collection contains exactly the same (session, position) cells as the "X" one.
            pat = {q: _session_trials(bd, i, q, variant, field) for q in CONF_LABELS
                   if len(_session_trials(bd, i, q, variant)) >= min_trials}
            if pat:
                by_day[day] = pat
                all_days.add(day)
        out[an] = (pre_by_sess, by_day)
    return out, sorted(all_days)
def _pre_reference(pre_by_sess, exclude=None):
    """Pooled pre-stroke trials per position, optionally leaving one session out.

    Leaving the scored session out is what keeps a pre-stroke column from being circular: a session
    correlated against a pool it is itself part of is scored partly against itself.
    """
    ref = {}
    for s, pat in pre_by_sess.items():
        if s == exclude:
            continue
        for q, Z in pat.items():
            ref.setdefault(q, []).append(Z)
    return {q: np.vstack(v) for q, v in ref.items()}
#: Resamples for the delta intervals. Fewer than the pattern figures' 400 because each draw here
#: rebuilds a full 6x6 from scratch for every animal-day; 200 is ample for a 95% percentile interval.
N_BOOT_DELTA = 200
def _block_index(blk):
    """{block id -> row indices} for one (session, position) trial set."""
    out = {}
    for j, b in enumerate(np.asarray(blk)):
        out.setdefault(int(b), []).append(j)
    return {k: np.asarray(v) for k, v in out.items()}
def _block_boot(pat_x, pat_blk, rng, min_trials=4):
    """One block-bootstrap draw of a whole session: {position -> resampled trials}.

    The session's blocks are pooled ACROSS positions and drawn once with replacement, so every
    position moves together in a single draw exactly as they do in a real session. A position whose
    draw leaves it under ``min_trials`` is dropped from that replicate rather than estimated from
    two trials -- it then shows as a wider interval, which is the honest consequence.
    """
    idx = {q: _block_index(pat_blk[q]) for q in pat_x if q in pat_blk}
    blocks = [(q, b) for q, m in idx.items() for b in m]
    if not blocks:
        return {}
    pick = rng.integers(0, len(blocks), size=len(blocks))
    take = {}
    for k in pick:
        q, b = blocks[k]
        take.setdefault(q, []).append(idx[q][b])
    out = {}
    for q, parts in take.items():
        rows = np.concatenate(parts)
        if len(rows) >= min_trials:
            out[q] = pat_x[q][rows]
    return out
def _delta_diag_ci(mats_for, x_store, blk_store, an, days, seed_parts, n_boot=N_BOOT_DELTA):
    """95% interval on (day diagonal - PRE diagonal), block-bootstrapped, ONE DAY AT A TIME.

    ``mats_for(animal, rng)`` returns the matrix builder; it is called per day so the builder and
    the draws share that day's generator. The PRE reference is resampled in the SAME draw as the
    day, so the two are correlated exactly as they are in the data and the difference is taken draw
    by draw -- differencing two independently published intervals would overstate the spread.

    EACH DAY IS SEEDED AND CACHED SEPARATELY. Previously every day of an animal came off one shared
    stream, which had two costs: a day's interval depended on how many days preceded it in that
    run, and no day could be stored and replayed. Priya, 2026-08-28 -- store the bootstraps so a
    nightly run recomputes only the sessions that changed. This family is 49% of a full render
    (6d, 7d, 8d and 9 all come through here), so on a night with one new session it is most of the
    saving.
    """
    pre_x, day_x = x_store
    pre_b, day_b = blk_store
    # Taken once: a re-preprocessed PRE-stroke session must invalidate every day of this animal.
    pre_key = _digest(pre_x, pre_b)
    out = {}
    for d in days:
        if d not in day_x:
            continue
        rng = np.random.default_rng(_seed(*seed_parts, d))
        mats_fn = mats_for(an, rng)
        rec = _boot_cached(
            f"delta_{seed_parts[-1]}",
            (pre_key, day_x[d], day_b[d], tuple(str(p) for p in seed_parts), n_boot),
            lambda mats_fn=mats_fn, rng=rng, d=d: _delta_diag_one(
                mats_fn, pre_x, pre_b, day_x[d], day_b[d], rng, n_boot))
        if rec and rec.get("mean"):
            out[d] = rec
    return out
def _delta_diag_one(mats_fn, pre_x, pre_b, dx, db, rng, n_boot):
    """One day's delta record: ``{"mean": (lo, hi, med), "pos": {position: (lo, hi, med)}}``.

    Extracted from `_delta_diag_ci` so a single day is the unit that gets cached. The arithmetic is
    unchanged, including the leave-one-session-out baseline the synthetic test was built to catch.
    """
    deltas = []
    for _ in range(n_boot):
        # Resample every pre-stroke session ONCE per draw, then reuse those same resampled
        # sessions for both the baseline and the day's reference, so the two share their noise
        # and the difference below is taken draw by draw.
        pre_r = {s: _block_boot(pre_x[s], pre_b[s], rng) for s in pre_x}
        pre_r = {s: v for s, v in pre_r.items() if v}
        day_r = _block_boot(dx, db, rng)
        if not pre_r or not day_r:
            continue

        def _pool(exclude=None, pre_r=pre_r):
            acc = {}
            for s, Z in pre_r.items():
                if s == exclude:
                    continue
                for q, z in Z.items():
                    acc.setdefault(q, []).append(z)
            return {q: np.vstack(v) for q, v in acc.items()}

        # THE BASELINE IS LEAVE-ONE-SESSION-OUT, not the reference against itself. Scoring the
        # resampled reference on itself gives a diagonal of exactly 1.0 -- a mean correlated
        # with its own mean -- so every delta came out at about -1 regardless of the data. The
        # synthetic test caught it; on real data it would have looked like a catastrophic and
        # perfectly uniform loss at every position in every animal.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # PER-POSITION VECTORS, averaged across held-out sessions as VECTORS. Averaging the
            # scalar mean-diagonal per session first and differencing that would give the same
            # overall number but no per-position breakdown -- and the per-position trajectory is
            # what the deficit is actually about.
            bases = []
            for s, held in pre_r.items():
                rest = _pool(exclude=s)
                if held and rest:
                    bases.append(np.diag(mats_fn(held, rest)).copy())
            full = _pool()
            if not bases or not full:
                continue
            base_vec = np.nanmean(np.stack(bases), axis=0)
            cur_vec = np.diag(mats_fn(day_r, full)).copy()
            deltas.append(cur_vec - base_vec)
    if len(deltas) < n_boot // 4:
        return {}
    D = np.stack(deltas)                                   # draws x positions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(D, axis=1)                          # per draw, mean over positions
        m = m[np.isfinite(m)]
        rec = {}
        if len(m):
            rec["mean"] = (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)),
                           float(np.median(m)))
        pos = {}
        for k, q in enumerate(CONF_LABELS):
            col = D[:, k]
            col = col[np.isfinite(col)]
            if len(col) >= n_boot // 4:
                pos[q] = (float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5)),
                          float(np.median(col)))
        rec["pos"] = pos
    return rec
def _corr_matrix(src_means, ref_means, labels=None):
    """M[i, j] = corr(src pattern at label i, reference pattern at label j)."""
    labels = labels or CONF_LABELS
    M = np.full((len(labels), len(labels)), np.nan)
    for i, p in enumerate(labels):
        for j, q in enumerate(labels):
            a, b = src_means.get(p), ref_means.get(q)
            if a is None or b is None or not np.std(a) or not np.std(b):
                continue
            M[i, j] = float(np.corrcoef(a, b)[0, 1])
    return M
def _means(pat):
    return {q: Z.mean(0) for q, Z in pat.items()}
def _nanmean_stack(Ms):
    if not Ms:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # positions absent in every entry
        return np.nanmean(np.stack(Ms), axis=0)
@lru_cache(maxsize=6)
def _matrices_pattern(align, variant, min_trials=10):
    """{animal: {"PRE": M, day: M, ...}} of mean-pattern correlation matrices (figures 6b / 6d)."""
    store, days = _collect_7(align, variant, min_trials)
    out = {}
    for an, (pre_by_sess, by_day) in store.items():
        ref_m = _means(_pre_reference(pre_by_sess))
        loo = [_corr_matrix(_means(pat), _means(_pre_reference(pre_by_sess, exclude=s)))
               for s, pat in pre_by_sess.items()]
        d = {}
        base = _nanmean_stack(loo)
        if base is not None:
            d["PRE"] = base
        for day, pat in by_day.items():
            d[day] = _corr_matrix(_means(pat), ref_m)
        if d:
            out[an] = d
    return out, days
def _delta_grid(mats, days, out_dir, fname, *, title, abs_label, delta_label,
                vmin, vmax, cmap, dmax, summary, ylab, figh=9.5, cis=None,
                higher_is_better=True):
    """Column 0 = the pre-stroke reference in its own units; every later column = that column MINUS
    the reference, on a diverging scale centred at zero.

    TWO COLOURBARS, DELIBERATELY. One shared scale would either compress the deltas into the middle
    of an absolute ramp or draw the reference on a diverging map centred somewhere meaningless. The
    two columns groups are different quantities and are scaled as such.
    """
    # A DEDICATED SPACER COLUMN FOR THE REFERENCE COLOUR BAR. Attaching it to `ax=axes[:, 0]` puts
    # it against the RIGHT EDGE of column 1's bounding box -- i.e. in the narrow gap between the
    # reference panel and the first delta panel, where it overlapped the day-1 matrices (Priya,
    # 2026-08-25). Reserving a real column and drawing into an explicit `cax` inside it is
    # deterministic; shrinking `fraction` would only have made the overlap thinner.
    ncol = 1 + len(days)
    # HSPACE 0.60 AT HEIGHT 9.5, the combination verified clean across 6, 9 and 12
    # post-stroke days. This grid set no hspace at all, so it ran at matplotlib's default
    # 0.2 while every sibling matrix grid had already moved to 0.60 -- the per-day titles
    # sat on the row above.
    #
    # THE WIDTH IS DERIVED FROM THE RATIOS, not from `ncol`, and that is the fix rather than the
    # tidiness. The figure was sized `_colw() * ncol` while the grid divided that space among
    # `ncol + SPACER` units -- the colour-bar spacer is a real column and was taking its share
    # from panels the width had not paid for. Every panel therefore came out narrower than
    # `_colw()` promises, by a fraction that GROWS with the day count.
    #
    # It went unnoticed while the slack absorbed it. Registering 0827 took the delta grids to
    # EIGHT post-stroke days, and driving the real function with fabricated data (two-line 9.5pt
    # titles, which is what `cis` produces -- an earlier probe passed cis=None, got one-line
    # 7.5pt titles, and reported clean while the shipped figure was not) gives 8 title-vs-title
    # overlaps at 8 days and 0 once the spacer is paid for. `hspace` cannot help: the collision
    # is horizontal.
    #
    # ABSOLUTE INCH MARGINS (2026-08-28), which is the durable fix the note above promised and
    # the same root cause as `_colw`'s two raises.
    #
    # `subplots_adjust` and `gridspec_kw` take FRACTIONS. The old figsize was
    # `_colw() * sum(ratios) + 2.0`, and matplotlib's default left=0.125/right=0.9 then took
    # 22.5% of whatever that came to -- so at eight post-stroke days each panel arrived 1.56in
    # wide against the 1.80in `_colw()` promises, and the shortfall GREW with every session
    # registered. That is why the width constant had to be raised twice and why the crowding came
    # back on schedule at the eleventh day: an additive margin diluted by a fractional one is not
    # a margin, it is a slowly closing gap.
    #
    # Deriving the width from the margins instead of the margins from the width gives every panel
    # a column exactly `_colw()` inches wide at any day count.
    #
    # MEASURED, and it corrects what `_colw` believes about itself: the DRAWN axes is 1.216in, not
    # 1.80, at every day count both before and after this change. `imshow` fixes a square aspect,
    # and four rows inside figh=9.5 leave about 1.2in of height each -- so the panels are HEIGHT
    # limited, and the column width has never been what sets their size. What the extra width buys
    # is space BETWEEN panels, and that is precisely what the titles were colliding for. Driven at
    # 6, 8, 11 and 14 post-stroke days: 16 title-vs-title overlaps at 14 days before, 0 after, and
    # 0 at every count in between.
    #
    # So `_colw()`'s docstring is half right. Six rotated tick labels do set a floor, but raising
    # it twice worked by widening the gaps, not by widening the panels. Anyone wanting genuinely
    # larger panels has to raise `figh`.
    #
    # HORIZONTAL ONLY, deliberately. The collision this fixes is horizontal, and the vertical
    # layout (hspace 0.60 at figh 9.5) was measured clean across 6, 9 and 12 days -- rewriting it
    # here would put a verified layout back at risk for nothing.
    LEFT_IN, RIGHT_IN, GAP_IN = 1.15, 0.95, 0.14      # row labels; colour bars; between panels
    ratios = [1, 0.62] + [1] * len(days)
    panel_in = _colw()
    fig_w = (LEFT_IN + RIGHT_IN + panel_in * sum(ratios) + GAP_IN * (len(ratios) - 1))
    fig, grid = plt.subplots(len(ANIMALS), ncol + 1,
                             figsize=(fig_w, figh),
                             squeeze=False,
                             gridspec_kw={"hspace": 0.60, "width_ratios": ratios,
                                          "left": LEFT_IN / fig_w,
                                          "right": 1.0 - RIGHT_IN / fig_w,
                                          # wspace is a fraction of the MEAN panel width
                                          "wspace": GAP_IN / panel_in})
    spacer = grid[:, 1]
    for ax in spacer:
        ax.axis("off")
    axes = np.delete(np.asarray(grid, dtype=object), 1, axis=1)
    im_abs = im_del = None
    for ri, an in enumerate(ANIMALS):
        d = mats.get(an) or {}
        base = d.get("PRE")
        for ci in range(ncol):
            ax = axes[ri][ci]
            M = base if ci == 0 else d.get(days[ci - 1])
            if M is None or base is None or not np.isfinite(M).any():
                ax.axis("off")
                continue
            # THE WHOLE ROW, NOT ONLY ITS DIAGONAL (Priya, 2026-08-26). A diagonal of 0.2 cannot
            # separate "the code is gone" from "the code moved to far_L" -- 0.2 against everything,
            # and 0.2 against itself with 0.7 elsewhere, are the same number. `self n/6` counts the
            # positions whose BEST match is still themselves: it uses all six entries and is
            # invariant to any monotone change across a row, so the uniform row shifts that dominate
            # the distance panels -- amplitude rather than resemblance -- cannot move it.
            _bm, _rk = _best_match(M, higher_is_better=higher_is_better)
            _self = int(sum(i == j for i, j in enumerate(_bm) if j >= 0))
            _nrow = int(sum(1 for j in _bm if j >= 0))
            if ci == 0:
                im_abs = ax.imshow(np.ma.masked_invalid(M), vmin=vmin, vmax=vmax, cmap=cmap)
                # "PRE", not "PRE (reference)": the header already says column 1 is the
                # reference, and the longer title reached right far enough to collide with the
                # colour bar's rotated label.
                head = "PRE"
                stat = summary(M)
            else:
                D = M - base
                im_del = ax.imshow(np.ma.masked_invalid(D), vmin=-dmax, vmax=dmax, cmap="PuOr_r")
                head = f"day {days[ci - 1]}"
                stat = summary(M) - summary(base)
            ax.set_xticks(range(len(CONF_LABELS)))
            ax.set_yticks(range(len(CONF_LABELS)))
            ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                               rotation=90, fontsize=9)
            ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
            # THE INTERVAL GOES WHERE THE NUMBER IS. The change in mean diagonal is the claim each
            # panel makes, so a bare point estimate there is the one place an interval is most
            # needed. Blank when the bootstrap could not resolve that cell -- never an interval
            # silently omitted from a panel that has one everywhere else.
            band = (cis or {}).get(an, {}).get(days[ci - 1]) if ci else None
            _sm = f"  self {_self}/{_nrow}" if _nrow else ""
            if ci == 0:
                lab = f"{head}  {stat:.2f}{_sm}"
            elif band:
                lab = f"{head}  {stat:+.2f}{_sm}\n[{band[0]:+.2f}, {band[1]:+.2f}]"
            else:
                lab = f"{head}  {stat:+.2f}{_sm}"
            ax.set_title(lab, fontsize=9.5 if band else 7.5,
                         fontweight="bold" if ci == 0 else "normal")
            if ci == 0:
                ax.set_ylabel(f"{an}\n{ylab}", fontsize=11, fontweight="bold")
    if im_abs is None or im_del is None:
        plt.close(fig)
        return None
    # WRAP LONG TITLE LINES. `bbox_inches="tight"` sizes the saved canvas around EVERYTHING it
    # contains, so one over-long suptitle line stretches the whole image and squashes the panels
    # into a fraction of it -- which is what a 420-character line did to 6d the moment the bootstrap
    # interval was described in the header. Explicit newlines the caller wrote are preserved; only
    # over-long lines are broken, so this cannot silently reflow a deliberate layout.
    title = "\n".join(textwrap.fill(ln, width=150) if len(ln) > 150 else ln
                      for ln in title.split("\n"))
    fig.colorbar(im_del, ax=axes[:, 1:].ravel().tolist(), fraction=0.012, pad=0.02,
                 label=delta_label)
    # Explicit cax INSIDE the reserved spacer column, computed after the delta bar has taken its
    # own space (it shrinks only the day columns, never the spacer).
    #
    # TICKS AND LABEL ON THE **LEFT** OF THE BAR. Matplotlib puts both on the right by default, so
    # even with the bar itself safely inside the spacer the rotated label was drawn past the
    # spacer's right edge and over the day-1 matrices (Priya, 2026-08-25 -- the second report of
    # this, the first having been the bar itself). Everything now extends LEFT, toward the gap
    # beside the reference panel, which carries no labels of its own.
    top, bot = spacer[0].get_position(), spacer[-1].get_position()
    # The bar sits in the RIGHT part of the spacer and its ticks and label extend LEFT into the
    # rest of it. The spacer therefore has to hold three things, not one: label, tick labels, bar.
    # At 0.42 it did not, and the rotated label reached back over the reference column's panel
    # TITLES -- the same colour bar intruding on a third neighbour, caught this time by measurement
    # rather than by eye.
    cax = fig.add_axes([top.x0 + 0.70 * top.width, bot.y0,
                        0.16 * top.width, top.y1 - bot.y0])
    cb = fig.colorbar(im_abs, cax=cax)
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    cax.tick_params(labelsize=7)
    cb.set_label(abs_label, fontsize=11)
    _suptitle(fig, title, fontsize=9.5)
    _footer(fig)
    p = _out(out_dir, fname.removesuffix(".png"))
    _save(fig, p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p
def _diag(M):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(np.diag(M)))
def _delta_cis(align, variant, min_trials, mats_for, tag, n_boot=N_BOOT_DELTA, full=False):
    """{animal: {day: (lo, hi)}} on the change in mean diagonal, block-bootstrapped.

    ``mats_for(animal, rng)`` returns the matrix builder for one animal. It is a hook rather than a
    fixed function because the crossnobis figure needs a whitener held FIXED across resamples --
    re-estimating it per draw would make the cross-validated product depend on the resample and stop
    being unbiased, the same reason the estimator wants a whitener independent of the data it
    whitens.

    ONE DRIVER FOR EVERY DELTA FIGURE, so they cannot drift apart in what they resample.
    """
    x_store, days = _collect_7(align, variant, min_trials)
    b_store, _ = _collect_7(align, variant, min_trials, "blk")
    out = {}
    for an in ANIMALS:
        if an not in x_store or an not in b_store:
            continue
        (pre_x, day_x), (pre_b, day_b) = x_store[an], b_store[an]
        try:
            rich = _delta_diag_ci(mats_for, (pre_x, day_x), (pre_b, day_b), an, days,
                                  (an, align, variant, tag), n_boot=n_boot)
            # `_delta_grid` prints only the mean's (lo, hi); figure 9 plots the whole record.
            out[an] = rich if full else {d: r["mean"][:2] for d, r in rich.items()}
        except Exception as ex:                                          # noqa: BLE001
            print(f"  !! {tag} CI {an} {align}/{variant}: {type(ex).__name__} {str(ex)[:80]}",
                  flush=True)
    return out
def _mats_pattern(_an, _rng):
    return lambda pat, ref: _corr_matrix(_means(pat), _means(ref))
def _pct3(v):
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.median(v)))
def _anchor(iv, theta, lo=None, hi=None):
    """Move a percentile interval so it contains the estimate the figure actually plots.

    A block bootstrap of a CORRELATION or an R^2 is biased LOW, and not subtly. Resampling blocks
    with replacement leaves only ~63% of a session's distinct trials in a draw, so every resampled
    mean is noisier than the observed one, and two noisier means agree less. It showed the first
    time this was run on real data: several encoder point estimates sat at or ABOVE the upper limit
    of their own percentile interval (PS92 PRE +0.57 against [+0.32, +0.56]). That is the same
    "interval that does not contain its own estimate" failure figure 8d announced itself with.

    THE SPREAD IS STILL RIGHT; only the location is wrong. Shifting by (estimate - bootstrap median)
    keeps the width and the asymmetry and guarantees the band contains the point drawn on top of it.
    A pivotal interval (2*theta - hi, 2*theta - lo) corrects the same bias but REFLECTS the
    asymmetry, and where the bias exceeds half the width it returns a band lying entirely to one
    side of the estimate -- true to the arithmetic and unreadable on a figure.

    FOR THE DELTAS THE BIAS LARGELY CANCELS: the day and the ceiling are resampled in the SAME draw
    and both are pulled down together, so the shift there comes out small. That is a check on this
    correction rather than a use of it.
    """
    if not iv or theta is None or not np.isfinite(theta):
        return iv
    d = float(theta) - iv[2]
    lo_, hi_ = iv[0] + d, iv[1] + d
    # CLIPPED TO THE PARAMETER SPACE. Shifting a band that already sits near a bound pushes it past
    # one: a row correlation of 0.98 acquires an upper limit of 1.08, which no correlation can take.
    # The bound is a fact about the quantity, not a cosmetic trim.
    if lo is not None:
        lo_ = max(lo_, lo)
    if hi is not None:
        hi_ = min(hi_, hi)
    return (lo_, hi_, float(theta))
def _excludes_zero(iv):
    """True when a (lo, hi, med) interval lies wholly on one side of zero."""
    return bool(iv) and (iv[0] > 0 or iv[1] < 0)
def _best_match(M, higher_is_better=True):
    """Per row: (index of the best-matching column, rank of the diagonal, 1..n).

    USES THE WHOLE ROW, which is the point. The diagonal alone cannot separate "the code is gone"
    from "the code moved to a specific other position" -- 0.2 against every position and 0.2 against
    its own with 0.7 against far_L are the same diagonal and different results.

    ARGMAX AND RANK ARE INVARIANT to any monotone transform applied ACROSS a row, so the uniform
    row shifts that dominate the distance figures -- which are amplitude, not resemblance -- cannot
    move them. That is precisely where the diagonal is weakest.
    """
    n = M.shape[0]
    best = np.full(n, -1)
    rank = np.full(n, np.nan)
    for i in range(n):
        row = M[i].astype(float)
        ok = np.isfinite(row)
        if ok.sum() < 2 or not np.isfinite(row[i]):
            continue
        v = row.copy()
        if not higher_is_better:
            v = -v
        # TIES GO TO THE DIAGONAL. A row that is flat -- the code is gone, with no particular
        # substitute -- has every entry equal, and a bare argmax then returns whichever position
        # happens to come first in DISPLAY_ORDER, reporting a substitution that does not exist. It
        # would also disagree with `rank`, which correctly calls the diagonal tied-best. Preferring
        # the diagonal on a tie is the conservative direction: it never invents a move.
        vmax = np.nanmax(np.where(ok, v, -np.inf))
        best[i] = i if v[i] >= vmax else int(np.nanargmax(np.where(ok, v, -np.inf)))
        # rank of the diagonal among the usable entries, 1 = best match is itself
        rank[i] = 1 + int((v[ok] > v[i]).sum())
    return best, rank
def _impaired(an, thresh=0.5, min_n=10):
    """Positions that DROPPED below `thresh` on any post-stroke session, from behaviour alone.

    THE WORST SESSION, not the pooled rate. Pooling across every post-stroke day averages a
    recovery away: PS95's far_R goes 0.00 on day 1 to 0.87 by day 2, which pools to 0.48-0.55 and
    reported that animal as having NO impaired position at all -- in the animal whose day-1 far_R
    collapse is the cleanest in the cohort. "Positions with a licking deficit" means positions that
    HAD one.
    """
    worst = {}
    for mmdd, _day_ in _sessions(an, phases=("post",)):
        for pos, (hr, _lo, _hi, n) in _position_metrics(an, mmdd).items():
            if n >= min_n:
                worst[pos] = min(worst.get(pos, 1.0), hr)
    return {p for p, v in worst.items() if v < thresh}
