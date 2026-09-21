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

import json
import os
import textwrap
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
