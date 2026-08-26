"""GRANT FIGURES — a small, self-contained set for a progress report and a new application.

    python -m wfield_local.grant_figures [--output <dir>] [--only 1 1b 2 2b 3a 3b 4 5 5b 5c 6 6b]

Priya, 2026-08-24. Deliberately NOT deck figures: the deck exists to be interrogated and carries every
caveat on the slide, which is right there and wrong here. These are meant to be read in ten seconds by
someone who has not been in the weeds, so each one makes ONE point, and the caveats live in this
docstring and in DECISIONS.md rather than on the axes.

WHAT IS SHOWN

  1  BEHAVIOUR at all six spout positions, per animal, against days-from-lesion. Engaged hit rate --
     the "stopped" (terminal quit) trials are excluded, which is what `hit_rate` in the per-position
     metrics CSV already means -- with Wilson CIs. This is the deficit the rest of the work is about.

  2  PRE-STROKE CROSS-SESSION DECODING in the shared joint-LocaNMF basis, ENL / cue / lick. Each bar
     is leave-one-session-out accuracy pooled over the curated pre-stroke sessions, the error bar is
     the 95% CI across HELD-OUT SESSIONS (not across trials -- the session is the unit that
     generalisation is claimed over), and every held-out session is plotted as a point. Chance is
     1/6.

  3a POST-STROKE CODING RETAINED, per animal, per window, over days. y = the projection of that
     position's own trials onto its own pre-stroke coding direction, pole-normalised so 1.0 = the
     pre-stroke lick signature and 0 = the other positions. Error bars are SEM over trials. This is
     the "how much of the normal code is left" view.

  3b FROZEN vs WITHIN-SESSION DECODING, per animal, per window, over days. The frozen pre-stroke
     decoder asks whether the OLD code still reads out; a decoder trained on the post-stroke session
     itself asks whether position information is present AT ALL. Frozen falling while within-session
     holds up is reorganisation rather than loss -- the two lines and the gap between them are the
     point. Error bars are binomial 95% CIs on each session's own trial count.

EXCLUSIONS, and they are not cosmetic
  * PS92_0817 and PS93_0817 are dropped everywhere. They follow the 8/16 laser that did NOT take, so
    they are neither baseline nor post-stroke, and `config.session_phase` already labels them
    "excluded" -- this module asks the config rather than hardcoding dates.
  * PS94/PS95 were lesioned 2026-08-16 and PS92/PS93 2026-08-17, so DAY-FROM-LESION is per animal
    (`config.stroke_date`). Plotting against calendar date would misalign the cohort by a day.

WHAT THESE FIGURES DO NOT SAY, kept here so it is not lost when they are pasted into a document:
  * "Miss" is defined by SPOUT CONTACT, so an off-target lick counts as a miss in panel 1 and sits in
    the miss class in panel 3. The DAQ cannot distinguish "did not try" from "tried and missed".
  * Panel 3a's y can exceed 1.0. The projection rises either because a trial points more along the
    direction or because it sits further from the session centroid; above-1 values mean "at least
    intact", not "better than pre-stroke".
  * Panel 2 is PRE-STROKE only and is a capability claim (we can decode), not a lesion result.
"""
from __future__ import annotations

import argparse
import csv
import json
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
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
POS = ["far_R", "far_center", "far_L", "close_R", "close_center", "close_L"]
#: colour = ring, marker/linestyle = side -- the convention the behaviour figures already use, so a
#: reader who has seen those does not have to relearn it here.
POS_STYLE = {
    "far_R":        ("#b2182b", "o", "-"),
    "far_center":   ("#d6604d", "s", "-"),
    "far_L":        ("#f4a582", "^", "-"),
    "close_R":      ("#2166ac", "o", "--"),
    "close_center": ("#4393c3", "s", "--"),
    "close_L":      ("#92c5de", "^", "--"),
}
WINDOWS = (("ENL", "precue", "ENL (pre-cue)"), ("cue", "cue", "post-cue"),
           ("lick", "lick", "post-lick"))


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
        if ox > 1e-3 and oy > 1e-3:
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


#: Sessions earlier than this many days before the lesion are collapsed into ONE baseline point.
#: The June block sits at day -70 and the whole post-stroke story inside +/-8, so plotting the true
#: axis spends 85% of the width on empty space and squeezes the result into a sliver.
BASELINE_BEFORE = -20
BASELINE_X = -16


def _wilson(hits, n, z=1.96):
    """Wilson 95% interval -- the same construction the per-session CSV uses, for pooled counts."""
    if not n:
        return (float("nan"),) * 3
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fig_behaviour(out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.6), sharey=True, squeeze=False)
    for k, an in enumerate(ANIMALS):
        ax = axes[0][k]
        for pos in POS:
            col, mk, ls = POS_STYLE[pos]
            pre_x, pre_y, pre_e = [], [], [[], []]
            post_x, post_y, post_e = [], [], [[], []]
            base_h = base_n = 0
            for mmdd, day in _sessions(an):
                m = _position_metrics(an, mmdd).get(pos)
                if not m or m[3] < 5:
                    continue
                if day < BASELINE_BEFORE:
                    base_h += round(m[0] * m[3]); base_n += m[3]
                    continue
                # CLAMP AT ZERO. A position at ceiling has hit_rate == ci_hi == 1.0 and the
                # subtraction lands on -1e-16, which matplotlib rejects outright rather than
                # rounding -- 51 cells across the cohort are at ceiling.
                tgt = (pre_x, pre_y, pre_e) if day < 0 else (post_x, post_y, post_e)
                tgt[0].append(day); tgt[1].append(m[0])
                tgt[2][0].append(max(0.0, m[0] - m[1])); tgt[2][1].append(max(0.0, m[2] - m[0]))
            if base_n:
                p, lo, hi = _wilson(base_h, base_n)
                ax.errorbar([BASELINE_X], [p], yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]],
                            color=col, marker=mk, ms=5.5, capsize=2, elinewidth=0.7, lw=0)
            # PRE AND POST ARE DRAWN AS SEPARATE SEGMENTS. Joining day -2 to day +1 draws a line
            # through the lesion and reads as a continuous decline that was measured; nothing was
            # recorded in between.
            for xs, ys, es in ((pre_x, pre_y, pre_e), (post_x, post_y, post_e)):
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ls=ls, ms=4.5, lw=1.4,
                                capsize=2, elinewidth=0.7,
                                label=pos if (xs is pre_x and k == 0) else None)
        ax.axvline(0, color="k", lw=1.6, ls=":")
        ax.text(0, 1.035, "lesion", ha="center", fontsize=8, color="k",
                transform=ax.get_xaxis_transform())
        ax.axvline((BASELINE_X + BASELINE_BEFORE) / 2 + 2, color="0.6", lw=1.0, ls=(0, (2, 3)))
        ax.text(BASELINE_X, -0.075, "June\nbaseline", ha="center", va="top", fontsize=8,
                color="0.35", transform=ax.get_xaxis_transform())
        ax.set_title(an, fontsize=12, fontweight="bold")
        ax.set_xlabel("days from lesion")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(BASELINE_X - 2.5, None)          # else the collapsed baseline sits on the spine
        if k == 0:
            ax.set_ylabel("hit rate (engaged trials)")
        ax.grid(alpha=0.25, lw=0.5)
    # Below the panels, for the same reason as 1b: lower-left is where the impaired traces live.
    _h, _lab = axes[0][0].get_legend_handles_labels()
    if _h:
        fig.legend(_h, _lab, loc="lower center", ncol=len(POS), fontsize=11.5, frameon=False)
    _suptitle(fig, "Licking accuracy at each spout position, relative to the lesion. "
                 "Engaged trials only (the terminal quit period is excluded); bars are Wilson 95% "
                 "CIs. The two sessions after the laser that did not take are omitted.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 1.0))          # reserve the band the legend sits in
    p = Path(out_dir) / "grant_1_behaviour_by_position.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p


def fig_behaviour_collapsed(out_dir, jitter=0.11):
    """1b: EVERY pre-stroke day as one mean +/- SEM per position, then the post-stroke days.

    Priya, 2026-08-24. The dated version (panel 1) shows that baseline is flat and stays flat, which
    is worth showing once; after that the pre-stroke days are 11 points of the same number and the
    eye has to do the averaging. Here the whole baseline is ONE point per position and the deficit
    is read directly against it.

    SEM IS ACROSS SESSIONS, not across trials. The claim being made is "this position's hit rate on
    a typical day", so the day is the unit -- a trial-level interval would be several times tighter
    and would understate how much a position varies from session to session.

    Positions are offset horizontally because at baseline five of the six sit on top of each other
    near 1.0 and are otherwise unreadable.
    """
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.7), sharey=True, squeeze=False)
    for k, an in enumerate(ANIMALS):
        ax = axes[0][k]
        imp = _impaired(an)
        post_days = sorted({d for _m, d in _sessions(an, phases=("post",))})
        for pi, pos in enumerate(POS):
            col, mk, _ls = POS_STYLE[pos]
            off = (pi - (len(POS) - 1) / 2) * jitter
            pre_vals = [m[0] for mmdd, d in _sessions(an, phases=("pre",))
                        if (m := _position_metrics(an, mmdd).get(pos)) and m[3] >= 5]
            if pre_vals:
                ax.errorbar([-1 + off], [np.mean(pre_vals)],
                            yerr=[np.std(pre_vals, ddof=1) / np.sqrt(len(pre_vals))]
                            if len(pre_vals) > 1 else None,
                            color=col, marker=mk, ms=7, capsize=3, lw=0, elinewidth=1.2,
                            markeredgecolor="k", markeredgewidth=0.5,
                            # NO PER-POSITION ASTERISK IN THE LEGEND: the legend is drawn from
                            # the first animal only, so a mark meaning "impaired" there would be
                            # read as applying to all four. The per-panel title carries it.
                            label=pos if k == 0 else None)
            xs, ys = [], []
            for mmdd, d in _sessions(an, phases=("post",)):
                m = _position_metrics(an, mmdd).get(pos)
                if not m or m[3] < 5:
                    continue
                xs.append(d + off); ys.append(m[0])
            if xs:
                ax.errorbar(xs, ys, color=col, marker=mk, ms=4.5, lw=1.3, alpha=0.95)
        # ONE dotted line, at the TRUE ZERO of the axis (Priya, 2026-08-25). There were two -- a
        # grey one at -0.5 marking the break between the collapsed baseline point and the post days,
        # and a black one at an arbitrary +0.35 for the lesion -- which read as a pair of unexplained
        # rules with a gap between them. The x axis is already "days from lesion", so x = 0 IS the
        # lesion and needs no second marker; the "ALL pre" tick says where the collapsed point sits.
        # NO "lesion" TEXT: it collided with the per-panel title, which carries the impaired
        # positions and is the more useful text.
        ax.axvline(0, color="k", lw=1.4, ls=":")
        ax.set_xticks([-1] + post_days)
        ax.set_xticklabels(["ALL\npre"] + [str(d) for d in post_days], fontsize=11)
        ax.set_title(f"{an}\nimpaired: {', '.join(sorted(imp)) or 'none'}", fontsize=10,
                     fontweight="bold")
        ax.set_xlabel("days from lesion")
        ax.set_ylim(-0.02, 1.05)
        if k == 0:
            ax.set_ylabel("hit rate (engaged trials)")
        ax.grid(alpha=0.25, lw=0.5)
    # LEGEND BELOW ALL PANELS, not inside one (Priya, 2026-08-25). At lower-left of PS92 it sat
    # exactly on that animal's far_R trace, which runs 0.0-0.5 over the early post-stroke days --
    # i.e. it covered the deficit the figure exists to show. A figure-level legend cannot collide
    # with data whatever the values turn out to be, which an in-panel "best" location cannot promise.
    _h, _lab = axes[0][0].get_legend_handles_labels()
    if _h:
        fig.legend(_h, _lab, loc="lower center", ncol=len(POS), fontsize=11.5, frameon=False)
    _suptitle(fig, "Licking accuracy per spout position: the WHOLE pre-stroke baseline as one point "
                 "(mean +/- SEM across sessions), then each day after the lesion.\n"
                 "Engaged trials only; positions offset horizontally so all six are visible. "
                 "Each panel title lists the positions that dropped below 50% on any post-stroke day.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 1.0))          # reserve the band the legend sits in
    p = Path(out_dir) / "grant_1b_behaviour_pre_collapsed.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 2. pre-stroke decoding
def fig_prestroke_decoding(out_dir):
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    width = 0.26
    x = np.arange(len(ANIMALS))
    any_drawn = False
    for wi, (_disp, align, wname) in enumerate(WINDOWS):
        f = _fig_root() / f"joint_xsession_decoder_{align}.json"
        if not f.exists():
            print(f"  [grant] MISSING {f.name} -- run `joint_xsession --align {align}`", flush=True)
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        xs = x + (wi - (len(WINDOWS) - 1) / 2) * width
        for k, an in enumerate(ANIMALS):
            r = d.get(an)
            if not r:
                continue
            pre = {lab for lab in config.phase_labels("pre") if lab.startswith(an)}
            vals = [v for lab, v in (r.get("per_session") or {}).items() if lab in pre]
            if not vals:
                continue
            m = float(np.mean(vals))
            # CI ACROSS HELD-OUT SESSIONS. The claim is that the code generalises to a session the
            # decoder never saw, so the session is the unit -- a trial-level CI would be far tighter
            # and would be answering a different question.
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ax.bar(xs[k], m, width * 0.92, color=f"C{wi}", edgecolor="k", linewidth=0.5,
                   label=wname if k == 0 else None, zorder=2)
            ax.errorbar(xs[k], m, yerr=1.96 * sem, color="k", capsize=3, lw=1.1, zorder=3)
            ax.plot(np.full(len(vals), xs[k]) + np.linspace(-0.05, 0.05, len(vals)), vals,
                    "o", ms=2.6, color="k", alpha=0.45, zorder=4)
            any_drawn = True
    if not any_drawn:
        plt.close(fig)
        return None
    ax.axhline(1 / 6, color="k", ls=":", lw=1.2)
    ax.text(len(ANIMALS) - 0.4, 1 / 6 + 0.012, "chance (1/6)", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(ANIMALS, fontsize=11)
    ax.set_ylabel("cross-session decoding accuracy")
    ax.set_ylim(0, 1.0)
    # LEGEND BELOW THE AXES. Bars run from 0 to ~0.95 in every animal, so there is no interior
    # corner it can sit in without covering data -- in-axes it landed on PS92's post-cue bar.
    ax.legend(fontsize=11.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_title("Spout position decodes across sessions from a shared LocaNMF basis (pre-stroke)\n"
                 "Leave-one-session-out: the decoder never saw the session it is scored on.\n"
                 "Bar = mean over held-out sessions, error bar = 95% CI across sessions, "
                 "dots = individual sessions.", fontsize=9.5)
    fig.tight_layout()
    p = Path(out_dir) / "grant_2_prestroke_crossday_decoding.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p


def fig_prestroke_decoding_cohort(out_dir):
    """2b: the cohort version of panel 2 -- one bar per window, all four animals.

    THE ANIMAL IS THE UNIT. Bar = mean of the four per-animal leave-one-session-out accuracies,
    error bar = SEM across ANIMALS (n=4), and each animal is drawn as a labelled point. Pooling all
    ~44 held-out sessions into one bar instead would give a far tighter interval that describes how
    much a SESSION varies, not how much an ANIMAL does -- and a cohort claim is a claim about
    animals. With n=4 the SEM is wide, which is honest: it is what four mice support.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    marks = ("o", "s", "^", "D")
    for wi, (_disp, align, wname) in enumerate(WINDOWS):
        f = _fig_root() / f"joint_xsession_decoder_{align}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        per_animal = []
        for an in ANIMALS:
            r = d.get(an)
            if not r:
                continue
            pre = {lab for lab in config.phase_labels("pre") if lab.startswith(an)}
            vals = [v for lab, v in (r.get("per_session") or {}).items() if lab in pre]
            if vals:
                per_animal.append((an, float(np.mean(vals))))
        if not per_animal:
            continue
        ys = [v for _a, v in per_animal]
        m = float(np.mean(ys))
        sem = float(np.std(ys, ddof=1) / np.sqrt(len(ys))) if len(ys) > 1 else 0.0
        ax.bar(wi, m, 0.62, color=f"C{wi}", edgecolor="k", linewidth=0.6, zorder=2, alpha=0.9)
        ax.errorbar(wi, m, yerr=sem, color="k", capsize=4, lw=1.3, zorder=3)
        for j, (an, v) in enumerate(per_animal):
            ax.plot(wi + (j - (len(per_animal) - 1) / 2) * 0.12, v, marks[j % len(marks)],
                    ms=6, color="k", mfc="white", zorder=4,
                    label=an if wi == 0 else None)
        ax.text(wi, m + sem + 0.03, f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1 / 6, color="k", ls=":", lw=1.2)
    ax.text(len(WINDOWS) - 0.55, 1 / 6 + 0.015, "chance (1/6)", fontsize=8, ha="right")
    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels([w[2] for w in WINDOWS], fontsize=11)
    ax.set_ylabel("cross-session decoding accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_title("Spout position decodes across sessions, all four animals (pre-stroke)\n"
                 "Leave-one-session-out in a shared LocaNMF basis. Bar = mean across animals, "
                 "error bar = SEM across animals (n=4),\npoints = individual animals.",
                 fontsize=9.5)
    fig.tight_layout()
    p = Path(out_dir) / "grant_2b_prestroke_crossday_cohort.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p


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


#: Set by `--compact`: drop the number printed in each matrix cell and narrow the panels, for
#: reproduction at a quarter of a letter page. The colour still carries the value; the digits are
#: what cannot survive the reduction, and they are also what forces the panel to stay wide.
COMPACT = False


def _colw(full=1.55, compact=1.50):
    """Inches per matrix column. Narrower without in-cell numbers to print.

    MEASURED, not chosen. At 1.45 the panels come out 0.911in and six rotated two-character tick
    labels crowd -- by a hairline, 0.0001 of the figure's width, but a hairline closes completely
    when the figure is reproduced small. 1.50 is the first value that clears; 1.55 leaves margin.

    THE COMPACT VARIANT IS BARELY NARROWER, AND THAT IS THE FINDING. It was built on the assumption
    that the in-cell numbers were what forced the panel wide; measuring says otherwise -- the six
    TICK LABELS set the floor, and they are present in both variants. 1.30 crowded, 1.50 clears, so
    compact saves 0.4in of 13.6 rather than the third it was expected to. What it still buys is a
    panel with no digits to compete with the colour at reproduction size; what it does not buy is
    much width. The earlier claim that compact reaches a quarter page while the full version does
    not was resting on the wider gap, and is withdrawn: at 13.2in against 13.6in the two are within
    3% of each other.
    """
    return compact if COMPACT else full


def _txt(ax, *args, **kw):
    """`ax.text` that draws nothing in a compact render.

    Every in-cell number in this module goes through here, so the compact variant cannot be
    half-applied -- which is what would happen if each of the eight call sites carried its own
    `if not COMPACT` and one were added later without it.
    """
    if COMPACT:
        return None
    return ax.text(*args, **kw)


def _out(out_dir, stem):
    """Output path, tagged when compact so the two variants never overwrite each other."""
    return Path(out_dir) / f"{stem}{'_compact' if COMPACT else ''}.png"


def _short(labels):
    """Position labels shortened for an axis. Anything unrecognised passes through unchanged."""
    return [POS_SHORT.get(str(q), str(q)) for q in labels]


def fig_confusion_prestroke(out_dir):
    """4: mean PRE-STROKE leave-one-session-out confusion, 2x2 animals, one file per window.

    Counts are SUMMED over the held-out pre-stroke sessions and then row-normalised, so each row is
    P(predicted | true) over the whole baseline -- not the mean of per-session rates, which would
    weight a 200-trial session the same as a 500-trial one.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        f = _fig_root() / f"joint_xsession_decoder_{align}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.0), squeeze=False)
        drew = False
        for k, an in enumerate(ANIMALS):
            ax = axes[k // 2][k % 2]
            r = d.get(an)
            pre = {lab for lab in config.phase_labels("pre") if lab.startswith(an)}
            mats = [np.array(m, float) for lab, m in ((r or {}).get("confusion") or {}).items()
                    if lab in pre]
            if not mats:
                ax.axis("off")
                continue
            C = np.sum(mats, axis=0)
            row = C.sum(1, keepdims=True)
            M = np.divide(C, row, out=np.zeros_like(C), where=row > 0)
            acc = float(np.trace(C) / C.sum()) if C.sum() else float("nan")
            im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
            for i in range(len(CONF_LABELS)):
                for j in range(len(CONF_LABELS)):
                    if M[i, j] >= 0.01:
                        _txt(ax, j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                                color="white" if M[i, j] < 0.6 else "black")
            ax.set_xticks(range(len(CONF_LABELS)))
            ax.set_xticklabels(_short(CONF_LABELS), rotation=90, ha="center", fontsize=10)
            ax.set_yticks(range(len(CONF_LABELS)))
            ax.set_yticklabels(_short(CONF_LABELS), fontsize=10)
            ax.set_title(f"{an} — {acc:.2f} correct ({len(mats)} held-out sessions)",
                         fontsize=10, fontweight="bold")
            # X-LABEL ON THE BOTTOM ROW ONLY -- on the top row it lands on the row below's title.
            if k // 2 == 1:
                ax.set_xlabel("predicted")
            if k % 2 == 0:
                ax.set_ylabel("true")
            drew = True
        if not drew:
            plt.close(fig)
            continue
        fig.colorbar(im, ax=axes, fraction=0.035, pad=0.09, label="P(predicted | true)")
        _suptitle(fig, f"Pre-stroke cross-session decoding — {wname} window\n"
                     "Frozen leave-one-session-out in the shared LocaNMF basis: every trial scored "
                     "by a decoder that never saw its session.\n"
                     "Counts summed over held-out sessions, then row-normalised. Chance = 0.17.",
                     fontsize=10)
        p = Path(out_dir) / f"grant_4_confusion_prestroke_{align}.png"
        _save(fig, p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    return made[0] if len(made) == 1 else (made or None)


def fig_confusion_pre_post(out_dir):
    """5: pre-stroke lick / pre-stroke NO-LICK control / post-stroke, one row per animal.

    THE MIDDLE PANEL IS WHY THIS IS THREE PANELS AND NOT TWO. Post-stroke the impaired positions are
    almost entirely no-lick trials, so an honest pre-vs-post pair on the ALL-trials arm compares
    pre-stroke LICK rows against post-stroke NON-LICK rows and confounds the lesion with the absence
    of a movement. `pre_nolick` is the matched control that already exists in `section_g`:
    PRE-stroke no-lick trials scored by a decoder trained on the OTHER pre-stroke sessions' engaged
    trials, so it differs from the post panel in PHASE ALONE. Read left-to-right: what the code
    looks like normally, what it looks like without a lick but without a lesion, and what it looks
    like after the lesion.

    Counts are reconstructed from the stored row-normalised matrices (matrix * n_per_true_position),
    summed across post-stroke sessions, and re-normalised. The `pre` and `pre_nolick` panels are the
    POOLED pre-stroke reference and are byte-identical in every session record -- summing them across
    sessions would multiply the same matrix by the session count and change nothing except to imply
    a sample size that does not exist.
    """
    sg = _fig_root() / "section_g.json"
    if not sg.exists():
        return None
    G = json.loads(sg.read_text(encoding="utf-8"))
    # TWO LINES, because width is the scarce dimension. A one-line title plus the accuracy is wider
    # than a third of a 9in figure at this font, and neighbouring titles collided. Height is free:
    # the header reservation now measures panel titles rather than assuming their size.
    PANELS = (("pre", "PRE-stroke\nLICK trials"),
              ("pre_nolick", "PRE-stroke, NO-LICK\n(matched control)"),
              ("post", "POST-stroke\nALL trials"))
    made = []
    for gkey, wname in (("pre-cue", "ENL (pre-cue)"), ("post-cue", "post-cue")):
        fig, axes = plt.subplots(len(ANIMALS), 3, figsize=(9.0, 12.4), squeeze=False,
                                 gridspec_kw={"hspace": 0.42})
        drew = False
        for ri, an in enumerate(ANIMALS):
            sessions = sorted(k for k in G if k.startswith(an)
                              and config.session_phase(an, k.split("_")[-1]) == "post")
            if not sessions:
                for ci in range(3):
                    axes[ri][ci].axis("off")
                continue
            blocks = [((G[s].get("arms") or {}).get("all") or {}).get("confusion", {}).get(gkey)
                      for s in sessions]
            blocks = [b for b in blocks if b]
            if not blocks:
                for ci in range(3):
                    axes[ri][ci].axis("off")
                continue
            for ci, (key, ptitle) in enumerate(PANELS):
                ax = axes[ri][ci]
                use = blocks if key == "post" else blocks[:1]
                C = None
                for b in use:
                    d = b.get(key)
                    if not d:
                        continue
                    M = np.array(d["matrix"], float)
                    n = np.array(d["n_per_true_position"], float)
                    C = (np.nan_to_num(M) * n[:, None]) if C is None \
                        else C + np.nan_to_num(M) * n[:, None]
                if C is None:
                    ax.axis("off")
                    continue
                row = C.sum(1, keepdims=True)
                P = np.divide(C, row, out=np.full_like(C, np.nan), where=row > 0)
                acc = float(np.nansum(np.diag(C)) / C.sum()) if C.sum() else float("nan")
                im = ax.imshow(np.ma.masked_invalid(P), vmin=0, vmax=1, cmap="magma")
                for i in range(len(CONF_LABELS)):
                    if row[i, 0] == 0:
                        ax.text(2.5, i, "no trials", ha="center", va="center", fontsize=8,
                                color="firebrick", fontweight="bold")
                        continue
                    for j in range(len(CONF_LABELS)):
                        if P[i, j] >= 0.02:
                            _txt(ax, j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                                    color="white" if P[i, j] < 0.6 else "black")
                ax.set_xticks(range(len(CONF_LABELS)))
                ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                   rotation=90, ha="center", fontsize=9.5)
                ax.set_yticks(range(len(CONF_LABELS)))
                ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9.5)
                # THE ANIMAL MOVES TO THE Y LABEL, where every other figure in this module puts it.
                # In the title it competed for the panel's WIDTH, which is the scarce dimension --
                # the first column's title then reached its neighbour's.
                if ci == 0:
                    ax.set_ylabel(an, fontsize=11, fontweight="bold")
                ax.set_title(f"{ptitle}  ({acc:.2f})", fontsize=10,
                             fontweight="bold" if ci == 0 else "normal")
                drew = True
        if not drew:
            plt.close(fig)
            continue
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.07, label="P(predicted | true)")
        _suptitle(fig, f"The frozen pre-stroke decoder before and after the lesion — {wname} window\n"
                     "Rows = TRUE spout position, columns = predicted. MIDDLE PANEL IS THE MATCHED "
                     "CONTROL: pre-stroke NO-LICK trials scored by a decoder trained on the other "
                     "pre-stroke sessions,\nso it differs from the post panel in PHASE alone rather "
                     "than in phase and the absence of a movement together. "
                     "Post-stroke sessions pooled. Chance = 0.17.", fontsize=9.5)
        _footer(fig, _sg_labels())
        p = Path(out_dir) / f"grant_5_confusion_pre_post_{gkey.replace('-', '')}.png"
        _save(fig, p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    return made[0] if len(made) == 1 else (made or None)


class _Stored(Exception):
    """Raised to skip the LocaNMF recompute when the stored per-class confusions covered it."""


def fig_confusion_pre_post_working(out_dir):
    """5b: figure 5 with the post-stroke TERMINAL QUIT PERIOD removed.

    Priya, 2026-08-25: the same comparison on engaged post-stroke trials, without the "stopped"
    ones. Recomputed rather than filtered, because `section_g.json` stores confusions already summed
    over trials and a summed matrix cannot be un-summed.

    WHICH TRIALS. Post-stroke = lick trials PLUS miss-while-working, i.e. everything except the
    terminal non-recovering collapse. That is the `poststroke_all_working` population the coding
    directions already use. Dropping the no-lick trials entirely would be a different figure -- it
    would empty the impaired rows, which is the whole reason the all-trials arm exists.

    THIS IS NOT THE GATE `POSTSTROKE_ENGAGEMENT_FILTERING = False` FORBIDS, and the distinction
    matters. That flag rejects the ROLLING reference-rate gate, on the ground that a local dip is
    indistinguishable from a run of motor failures, so splitting trials on it would label the effect
    as the confound. `engagement_gate` requires a NON-RECOVERING FINAL collapse -- PS94_0817's rate
    dips at trial ~420 and is back near 0.95 by 480, and that session is correctly NOT called
    disengaged. Removing a terminal quit period is a much weaker claim than adjudicating individual
    trials, and it is the same construct the miss-vs-stopped split rests on throughout.

    STILL NOT VALIDATED, and the figure says so: nothing in the spout data proves the terminal run is
    satiety rather than a late motor collapse. It is reported BESIDE figure 5, never instead of it.
    """
    from wfield_local.locanmf_frozen_decoder import _pipe, pool_sessions
    from wfield_local.position_coding_directions import _gate_all

    PANELS = (("pre", "PRE-stroke\nLICK trials"),
              ("post_all", "POST-stroke\nALL trials"),
              ("post_working", "POST-stroke\nquit period REMOVED"))

    def _from_json(an, disp):
        """The three panels as SUMS of the stored per-class confusions, or None to recompute.

        Added 2026-08-26. This figure used to redo the whole LocaNMF pooling -- >10 min of network
        reads -- because `section_g.json` stores a confusion already summed over trials and "a summed
        matrix cannot be un-summed". `coding_direction.json` now stores one matrix PER CLASS, so the
        populations this figure needs are additions:

            post_working = poststroke_lick + poststroke_miss_working
            post_all     = post_working    + poststroke_stopped

        Returning None (missing file, missing block, missing class) falls through to the recompute
        path, so this is a shortcut and never a new source of truth.
        """
        f = _fig_root() / "coding_direction.json"
        if not f.exists():
            return None
        try:
            rec = ((json.loads(f.read_text(encoding="utf-8")).get(disp) or {}).get(an) or {})
            c = rec.get("confusions")
            if not c or c.get("prestroke_lick") is None:
                return None
            lick, work, stop = (c.get("poststroke_lick"), c.get("poststroke_miss_working"),
                                c.get("poststroke_stopped"))
            if lick is None or work is None:
                return None
            L, W = np.array(lick, float), np.array(work, float)
            S = np.zeros_like(L) if stop is None else np.array(stop, float)
            return {"pre": np.array(c["prestroke_lick"], float),
                    "post_working": L + W, "post_all": L + W + S}
        except Exception:                                             # noqa: BLE001
            return None
    made = []
    for _disp, align, wname in (("ENL", "precue", "ENL (pre-cue)"), ("cue", "cue", "post-cue")):
        fig, axes = plt.subplots(len(ANIMALS), 3, figsize=(9.0, 12.4), squeeze=False,
                                 gridspec_kw={"hspace": 0.42})
        drew = False
        stored = []                # animals served from coding_direction.json rather than recomputed
        for ri, an in enumerate(ANIMALS):
            pre = [x for x in config.phase_labels("pre") if x.startswith(an)]
            post = [x for x in config.phase_labels("post") if x.startswith(an)]
            # STORED CLASSES FIRST; the LocaNMF recompute is the fallback, not the default.
            mats = _from_json(an, _disp)
            if mats is not None:
                stored.append(an)
            try:
                if mats is not None:
                    raise _Stored          # skip the pooling; panels are drawn below either way
                mats = {}
                from wfield_local import joint_locanmf
                from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
                from wfield_local.precue_engagement_states import features_with_indices
                basis = joint_locanmf.load(an, sessions=SESSIONS)
                feat = features_with_indices(basis, nolick_ref="cue")
                XE, YE, GE, _B, XU, YU, kept, _c, GU = pool_sessions(
                    pre + post, source="locanmf", align=align, post_s=2.0, features=feat)
                g = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
                not_eng = g[0] if g else np.zeros(len(YU), bool)
                pre_i = {i for i, lab in enumerate(kept) if lab in set(pre)}
                e_pre = np.isin(GE, list(pre_i))
                GU = np.asarray(GU)
                u_pre = np.isin(GU, list(pre_i)) if len(GU) else np.zeros(0, bool)
                clf = _pipe().fit(XE[e_pre], YE[e_pre])
                name = np.vectorize(lambda v: POSITION_NAMES.get(int(v), str(v)))

                def conf(X, y, clf=clf, name=name):
                    """Counts matrix in CONF_LABELS order. `clf`/`name` bound as defaults because
                    they are loop variables and a late-binding closure would silently score every
                    animal with the LAST animal's decoder."""
                    if not len(y):
                        return None
                    pred = name(clf.predict(X))
                    true = name(y)
                    M = np.zeros((len(CONF_LABELS), len(CONF_LABELS)), float)
                    for t, q in zip(true, pred):
                        if t in CONF_LABELS and q in CONF_LABELS:
                            M[CONF_LABELS.index(t), CONF_LABELS.index(q)] += 1
                    return M

                # THE PRE PANEL MUST BE LEAVE-ONE-SESSION-OUT. Scoring the training trials with the
                # decoder fitted on them gave 0.89-0.99 here against 0.45-0.66 for the same animals
                # in figure 5, and a reader comparing post 0.48 to an in-sample 0.97 would read a
                # collapse that is mostly overfitting. The POST panels need no such care: those
                # trials are held out by construction.
                Cpre = None
                for i in sorted(pre_i):
                    tr = e_pre & (GE != i)
                    te = e_pre & (GE == i)
                    if te.sum() < 5 or len(np.unique(YE[tr])) < 2:
                        continue
                    c1 = conf(XE[te], YE[te], clf=_pipe().fit(XE[tr], YE[tr]))
                    Cpre = c1 if Cpre is None else Cpre + c1
                mats["pre"] = Cpre
                pe, pu = ~e_pre, (~u_pre if len(u_pre) else np.zeros(0, bool))
                Xa = np.vstack([XE[pe]] + ([XU[pu]] if len(u_pre) and pu.any() else []))
                ya = np.concatenate([YE[pe]] + ([YU[pu]] if len(u_pre) and pu.any() else []))
                mats["post_all"] = conf(Xa, ya)
                pw = pu & ~not_eng if len(u_pre) else np.zeros(0, bool)
                Xw = np.vstack([XE[pe]] + ([XU[pw]] if len(u_pre) and pw.any() else []))
                yw = np.concatenate([YE[pe]] + ([YU[pw]] if len(u_pre) and pw.any() else []))
                mats["post_working"] = conf(Xw, yw)
            except _Stored:
                pass                       # the stored per-class matrices are already in `mats`
            except Exception as ex:                                       # noqa: BLE001
                print(f"  !! 5b {an} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
                mats = mats or {}
            for ci, (key, ptitle) in enumerate(PANELS):
                ax = axes[ri][ci]
                C = mats.get(key)
                if C is None:
                    ax.axis("off")
                    continue
                row = C.sum(1, keepdims=True)
                P = np.divide(C, row, out=np.full_like(C, np.nan), where=row > 0)
                acc = float(np.trace(C) / C.sum()) if C.sum() else float("nan")
                im = ax.imshow(np.ma.masked_invalid(P), vmin=0, vmax=1, cmap="magma")
                for i in range(len(CONF_LABELS)):
                    if row[i, 0] == 0:
                        ax.text(2.5, i, "no trials", ha="center", va="center", fontsize=8,
                                color="firebrick", fontweight="bold")
                        continue
                    for j in range(len(CONF_LABELS)):
                        if P[i, j] >= 0.02:
                            _txt(ax, j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                                    color="white" if P[i, j] < 0.6 else "black")
                ax.set_xticks(range(len(CONF_LABELS)))
                ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                   rotation=90, ha="center", fontsize=9.5)
                ax.set_yticks(range(len(CONF_LABELS)))
                ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9.5)
                # Same treatment as figure 5: the animal on the Y LABEL, not in the title where it
                # competes for the panel's width. This figure's titles are the longest in the module
                # ("quit period REMOVED") and the trial count made them longer still, so n moves onto
                # its own line rather than extending the widest one.
                if ci == 0:
                    ax.set_ylabel(an, fontsize=11, fontweight="bold")
                ax.set_title(f"{ptitle}\n({acc:.2f}, n={int(C.sum())})", fontsize=9.5,
                             fontweight="bold" if ci == 0 else "normal")
                drew = True
        if not drew:
            plt.close(fig)
            continue
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.07, label="P(predicted | true)")
        _suptitle(fig, f"Frozen pre-stroke decoder, with and without the terminal quit period — "
                     f"{wname} window\n"
                     "Rows = TRUE spout position, columns = predicted. RIGHT panel drops the "
                     "post-stroke trials after a NON-RECOVERING collapse in responding at the "
                     "positions the animal can still reach;\nlick and miss-while-working trials are "
                     "kept, so the impaired rows still have trials. Post-stroke sessions pooled. "
                     "Chance = 0.17.\nThe quit period is not independently validated as satiety "
                     "rather than a late motor collapse — read this beside figure 5, not instead "
                     "of it.", fontsize=9.5)
        # SAY WHERE THE NUMBERS CAME FROM, and whether that source lags. A figure served from
        # coding_direction.json is only as current as the last position_coding_directions run, and
        # this module already has the scar: on 2026-08-25 the JSONs predated both 8/24 sessions while
        # the config had them, so the footer printed a reassuring "6, 6, 6, 6" on exactly the three
        # stale figures. Reading a stored artifact silently would re-create that.
        if stored:
            lag = sorted(set(config.phase_labels("post")) - set(_cd_labels()))
            print(f"  [5b] {align}: {len(stored)}/{len(ANIMALS)} animal(s) from stored per-class "
                  f"confusions ({', '.join(stored)})"
                  + (f" -- coding_direction.json LAGS the config by {len(lag)} session(s): "
                     f"{', '.join(lag)}; re-run position_coding_directions" if lag else
                     " -- coding_direction.json is current"), flush=True)
        _footer(fig)
        p = Path(out_dir) / f"grant_5b_confusion_working_{align}.png"
        _save(fig, p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    return made[0] if len(made) == 1 else (made or None)


def fig_confusion_per_session(out_dir):
    """5c: figure 5b unpooled -- one column per post-stroke SESSION, animals aligned by day.

    Priya, 2026-08-25. 5b pools every post-stroke day into one matrix, which is the same objection
    that produced the per-block coding-direction view: pooling averages a recovery and a collapse
    into "no change". PS94's far_R row and PS95's whole matrix move a lot across days and the pooled
    panel cannot show it.

    COLUMNS ARE DAYS FROM LESION, not session index, so a column means the same thing in every row
    even though the animals were lesioned on different dates and PS93 has one fewer post-stroke
    session than the others. A missing session is a blank cell rather than a shift.

    Post-stroke trials are LICK + MISS-WHILE-WORKING with the terminal quit period removed, as in
    5b, and the pre-stroke column is leave-one-session-out for the reason recorded there.
    """
    made = []
    for _disp, align, wname in (("ENL", "precue", "ENL (pre-cue)"), ("cue", "cue", "post-cue")):
        per_animal, days = _collect_5c(align)
        if not days:
            continue
        p = _draw_5c(per_animal, days, out_dir, align, wname)
        if p:
            made.append(p)
    return made[0] if len(made) == 1 else (made or None)


@lru_cache(maxsize=2)
def _collect_5c(align):
    """{animal: (pre-stroke LOSO confusion counts, {day: counts})} plus the sorted day list.

    Shared by figure 5c and its delta twin 5d: fitting the frozen decoder means reading every
    session's LocaNMF fit, and doing it twice to draw the same numbers two ways is pure waste.
    """
    from wfield_local import joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
    from wfield_local.locanmf_frozen_decoder import _pipe, pool_sessions
    from wfield_local.position_coding_directions import _gate_all
    from wfield_local.precue_engagement_states import features_with_indices

    if True:
        per_animal, all_days = {}, set()
        for an in ANIMALS:
            pre = [x for x in config.phase_labels("pre") if x.startswith(an)]
            post = [x for x in config.phase_labels("post") if x.startswith(an)]
            try:
                basis = joint_locanmf.load(an, sessions=SESSIONS)
                feat = features_with_indices(basis, nolick_ref="cue")
                XE, YE, GE, _B, XU, YU, kept, _c, GU = pool_sessions(
                    pre + post, source="locanmf", align=align, post_s=2.0, features=feat)
                g = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
                not_eng = g[0] if g else np.zeros(len(YU), bool)
                pre_i = {i for i, lab in enumerate(kept) if lab in set(pre)}
                e_pre = np.isin(GE, list(pre_i))
                GU = np.asarray(GU)
                name = np.vectorize(lambda v: POSITION_NAMES.get(int(v), str(v)))

                def conf(X, y, clf, name=name):
                    if not len(y):
                        return None
                    M = np.zeros((len(CONF_LABELS), len(CONF_LABELS)), float)
                    for t, q in zip(name(y), name(clf.predict(X))):
                        if t in CONF_LABELS and q in CONF_LABELS:
                            M[CONF_LABELS.index(t), CONF_LABELS.index(q)] += 1
                    return M

                clf = _pipe().fit(XE[e_pre], YE[e_pre])
                Cpre = None
                for i in sorted(pre_i):
                    tr, te = e_pre & (GE != i), e_pre & (GE == i)
                    if te.sum() < 5 or len(np.unique(YE[tr])) < 2:
                        continue
                    c1 = conf(XE[te], YE[te], _pipe().fit(XE[tr], YE[tr]))
                    Cpre = c1 if Cpre is None else Cpre + c1
                by_day = {}
                for i, lab in enumerate(kept):
                    if i in pre_i:
                        continue
                    day = _day(an, lab.split("_")[-1])
                    me = (GE == i)
                    mu = (GU == i) & ~not_eng if len(GU) else np.zeros(0, bool)
                    Xs = np.vstack([XE[me]] + ([XU[mu]] if mu.any() else []))
                    ys = np.concatenate([YE[me]] + ([YU[mu]] if mu.any() else []))
                    C = conf(Xs, ys, clf)
                    if C is not None and C.sum():
                        by_day[day] = C
                        all_days.add(day)
                per_animal[an] = (Cpre, by_day)
            except Exception as ex:                                       # noqa: BLE001
                print(f"  !! 5c {an} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
        return per_animal, sorted(all_days)


def _draw_5c(per_animal, days, out_dir, align, wname):
    """The absolute rendering of 5c. Split from the collector so 5d reuses the same numbers."""
    if True:
        ncol = 1 + len(days)
        fig, axes = plt.subplots(len(ANIMALS), ncol, figsize=(_colw() * ncol + 1.2, 8.5),
                                     gridspec_kw={"hspace": 0.45},
                                 squeeze=False)
        im = None
        for ri, an in enumerate(ANIMALS):
            got = per_animal.get(an)
            for ci in range(ncol):
                ax = axes[ri][ci]
                C = None if not got else (got[0] if ci == 0 else got[1].get(days[ci - 1]))
                if C is None or not C.sum():
                    ax.axis("off")
                    continue
                row = C.sum(1, keepdims=True)
                P = np.divide(C, row, out=np.full_like(C, np.nan), where=row > 0)
                acc = float(np.trace(C) / C.sum())
                im = ax.imshow(np.ma.masked_invalid(P), vmin=0, vmax=1, cmap="magma")
                ax.set_xticks(range(len(CONF_LABELS)))
                ax.set_yticks(range(len(CONF_LABELS)))
                ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                   rotation=90, fontsize=9)
                ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
                head = "PRE" if ci == 0 else f"day {days[ci - 1]}"
                ax.set_title(f"{head}\n{acc:.2f}", fontsize=10,
                             fontweight="bold" if ci == 0 else "normal")
                if ci == 0:
                    ax.set_ylabel(f"{an}\ntrue position", fontsize=11, fontweight="bold")
        if im is None:
            plt.close(fig)
            return None
        fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label="P(predicted | true)")
        _suptitle(fig, f"Frozen pre-stroke decoder, session by session — {wname} window\n"
                     "Post-stroke trials are LICK + MISS-WHILE-WORKING (terminal quit period "
                     "removed). Columns are DAYS FROM LESION so they mean the same thing in every "
                     "row;\na blank cell is a session that animal does not have. Rows = TRUE spout "
                     "position, columns within a panel = predicted. Chance = 0.17.", fontsize=9.5)
        _footer(fig)
        p = _out(out_dir, f"grant_5c_confusion_per_session_{align}")
        _save(fig, p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return p


def fig_pattern_similarity_per_session(out_dir, min_trials=10):
    """6b: figure 6 unpooled -- one column per post-stroke DAY, animals aligned by day.

    Same construction as figure 6: every panel is scored against ONE half of the pre-stroke trials,
    so the first column (the other pre-stroke half) is the no-lesion expectation and its diagonal is
    the split-half ceiling. Read every later column against that first one.

    A SINGLE SESSION'S MEAN PATTERN IS NOISIER than the pooled one, and at an impaired position it
    can rest on a few dozen trials, so `min_trials` gates each cell and a position below it is blank
    rather than drawn. The pooled figure 6 is the one to quote; this is the one that shows whether a
    pooled cell is a steady state or an average of a collapse and a recovery.
    """
    from wfield_local import joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
    from wfield_local.locanmf_frozen_decoder import pool_sessions
    from wfield_local.position_coding_directions import _gate_all
    from wfield_local.precue_engagement_states import features_with_indices

    # NO rng: the pre-stroke split is no longer a random draw over trials but a leave-one-SESSION-out
    # over the days themselves, so nothing here is stochastic.
    made = []
    for _disp, align, wname in WINDOWS:
        variants = ("lick",) if align == "lick" else ("lick", "working")
        store = {v: {} for v in variants}
        all_days = set()
        for an in ANIMALS:
            pre = [x for x in config.phase_labels("pre") if x.startswith(an)]
            post = [x for x in config.phase_labels("post") if x.startswith(an)]
            try:
                basis = joint_locanmf.load(an, sessions=SESSIONS)
                feat = features_with_indices(basis, nolick_ref="cue")
                XE, YE, GE, _B, XU, YU, kept, _c, GU = pool_sessions(
                    pre + post, source="locanmf", align=align, post_s=2.0, features=feat)
                g = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
                not_eng = g[0] if g else np.zeros(len(YU), bool)
                pre_i = {i for i, lab in enumerate(kept) if lab in set(pre)}
                e_pre = np.isin(GE, list(pre_i))
                GU = np.asarray(GU)
                en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
                un = (np.array([POSITION_NAMES.get(int(v), str(v)) for v in YU])
                      if len(YU) else np.zeros(0, str))
                # THE REFERENCE IS ALL PRE-STROKE TRIALS; the no-lesion column is LEAVE-ONE-SESSION-
                # OUT (corrected 2026-08-25). This used to take a random half of the pooled trials
                # as the reference and the other half as the ceiling, so both came from the SAME
                # DAYS and the ceiling carried no between-session drift at all -- while every post
                # column compares a DIFFERENT day against those days. It was an unreachable ceiling.
                # Now each pre-stroke session in turn is scored against the pool of the OTHERS and
                # the results averaged: one session against a pool, exactly like every post column.
                ref, loo = {}, []
                pre_ids = sorted(pre_i)
                for q in CONF_LABELS:
                    idx = np.flatnonzero(e_pre & (en == q))
                    if len(idx) < 2 * min_trials:
                        continue
                    ref[q] = _mean_pattern(XE[idx])
                for i in pre_ids:
                    held, rest = {}, {}
                    for q in CONF_LABELS:
                        h = XE[(GE == i) & (en == q)]
                        r = XE[e_pre & (GE != i) & (en == q)]
                        if len(h) >= min_trials and len(r) >= min_trials:
                            held[q], rest[q] = _mean_pattern(h), _mean_pattern(r)
                    if held:
                        loo.append((held, rest))
                for v in variants:
                    by_day = {}
                    for i, lab in enumerate(kept):
                        if i in pre_i:
                            continue
                        day = _day(an, lab.split("_")[-1])
                        pat = {}
                        for q in CONF_LABELS:
                            parts = [XE[(GE == i) & (en == q)]]
                            if v == "working" and len(un):
                                m = (GU == i) & (un == q) & ~not_eng
                                if m.any():
                                    parts.append(XU[m])
                            Xp = [z for z in parts if len(z)]
                            if Xp:
                                Z = np.vstack(Xp)
                                if len(Z) >= min_trials:
                                    pat[q] = _mean_pattern(Z)
                        if pat:
                            by_day[day] = pat
                            all_days.add(day)
                    store[v][an] = (ref, loo, by_day)
            except Exception as ex:                                       # noqa: BLE001
                print(f"  !! 6b {an} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
        if not all_days:
            continue
        days = sorted(all_days)
        for v in variants:
            ncol = 1 + len(days)
            fig, axes = plt.subplots(len(ANIMALS), ncol, figsize=(_colw() * ncol + 1.2, 8.7),
                                     gridspec_kw={"hspace": 0.45},
                                     squeeze=False)
            im = None
            for ri, an in enumerate(ANIMALS):
                got = store[v].get(an)
                for ci in range(ncol):
                    ax = axes[ri][ci]
                    if not got:
                        ax.axis("off")
                        continue
                    ref, loo, by_day = got
                    # COLUMN 0 averages the per-held-out-session matrices; every later column is one
                    # post-stroke session against the pooled pre-stroke reference. Both are "one
                    # session against a pool of other days", which is the comparison being made.
                    pairs = loo if ci == 0 else ([(by_day[days[ci - 1]], ref)]
                                                 if days[ci - 1] in by_day else [])
                    if not pairs:
                        ax.axis("off")
                        continue
                    Ms = []
                    for src, rf in pairs:
                        m1 = np.full((len(CONF_LABELS), len(CONF_LABELS)), np.nan)
                        for i, pp in enumerate(CONF_LABELS):
                            for j, q in enumerate(CONF_LABELS):
                                if src.get(pp) is None or rf.get(q) is None:
                                    continue
                                m1[i, j] = float(np.corrcoef(src[pp], rf[q])[0, 1])
                        Ms.append(m1)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN cells
                        M = np.nanmean(np.stack(Ms), axis=0)
                    im = ax.imshow(np.ma.masked_invalid(M), vmin=-1, vmax=1, cmap="RdBu_r")
                    ax.set_xticks(range(len(CONF_LABELS)))
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                       rotation=90, fontsize=9)
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
                    diag = np.nanmean(np.diag(M))
                    # TWO LINES, and the head shortened to "PRE". The column-0 title was several times wider
                    # than a day column's ("PRE, leave-1-out  diag 0.77" against "day 1  0.59"), so it
                    # overran into column 1 and reached left into its own row's ylabel -- the ax0/8/16/24
                    # stride in the layout report, one fault per row across three whole families. What
                    # "PRE" means is in the header of every one of these figures.
                    head = "PRE" if ci == 0 else f"day {days[ci - 1]}"
                    ax.set_title(f"{head}\ndiag {diag:.2f}", fontsize=10,
                                 fontweight="bold" if ci == 0 else "normal")
                    if ci == 0:
                        ax.set_ylabel(f"{an}\nthis position", fontsize=11, fontweight="bold")
            if im is None:
                plt.close(fig)
                continue
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label="pattern correlation r")
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            _suptitle(fig, f"Mean-pattern similarity session by session — {wname} window\n"
                         f"Post-stroke class: {cls}.  Rows within a panel = the pattern being "
                         f"described; columns within a panel = the PRE-STROKE reference.\n"
                         "FIRST COLUMN is the no-lesion expectation and its diagonal is the "
                         "CEILING: each pre-stroke session in turn scored against the pool of the "
                         "OTHERS, averaged -- one session against other days, exactly like every "
                         "post column.\nIt is NOT 1.0 and must not be read against 1.0: two "
                         "pre-stroke days differ by ordinary drift. 'diag' above each panel is the "
                         "mean of that panel's diagonal.\nColumns are DAYS FROM LESION; a blank is "
                         "a session that animal does not have.", fontsize=9.5)
            _footer(fig)
            p = _out(out_dir, f"grant_6b_pattern_per_session_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made[0] if len(made) == 1 else (made or None)


#: Resamples for the pattern-similarity intervals and null. 400 is enough for a 95% percentile
#: interval and keeps a full render (4 animals x 3 windows x 2 variants) inside a few minutes.
N_BOOT = 400


def _strat_mean(by_session, rng):
    """One stratified bootstrap mean: resample TRIALS WITHIN each session, sessions kept fixed.

    SESSIONS ARE NOT RESAMPLED, and that is the whole point (Priya, 2026-08-25: "session-level is
    problematic with things dynamically changing over sessions"). A session bootstrap assumes days
    are exchangeable draws from one distribution; they are not -- PS94's frozen accuracy runs
    0.39 -> 0.76 across six days and PS95 sits at 0.81-0.84 then falls to 0.60. Resampling days
    would fold that trajectory into "sampling noise" and produce an interval for a post-stroke
    state that does not exist.

    Conditioning on the sessions instead makes the interval mean: "how well determined is this
    estimate GIVEN THESE DAYS". It does NOT license generalisation to other days -- that question
    needs the trajectory, which is what the per-session figure shows rather than summarises.
    """
    tot = None
    n = 0
    for X in by_session:
        if not len(X):
            continue
        idx = rng.integers(0, len(X), len(X))       # trials within this session, count preserved
        s = X[idx].sum(0)
        tot = s if tot is None else tot + s
        n += len(X)
    return None if not n else tot / n


def _pattern_stats(post_by_session, ref_by_session, labels, rng, n_boot=N_BOOT):
    """(r, lo, hi, null_hi) per (row, col), all from the same stratified resampling scheme.

    `null_hi` is the 97.5th percentile of |r| under a POSITION-LABEL PERMUTATION: the post-stroke
    trials keep their session, their count and the global post-stroke pattern, and only which
    position they belong to is shuffled. That is the right null for an off-diagonal claim, because
    it asks "is there position-specific structure here" rather than "is r different from zero" --
    and positions are intrinsically similar, so a zero-null would call almost every cell
    significant.
    """
    K = len(labels)
    obs = np.full((K, K), np.nan)
    ref_mean = {q: _strat_mean(ref_by_session.get(q, []), rng) for q in labels}
    post_mean = {p: _strat_mean(post_by_session.get(p, []), rng) for p in labels}
    for i, p in enumerate(labels):
        for j, q in enumerate(labels):
            if post_mean.get(p) is None or ref_mean.get(q) is None:
                continue
            obs[i, j] = float(np.corrcoef(post_mean[p], ref_mean[q])[0, 1])
    boots = np.full((n_boot, K, K), np.nan)
    nulls = np.full((n_boot, K, K), np.nan)
    # PER-SESSION STACKS, built ONCE. The first version rebuilt Python lists of single trial rows on
    # every resample -- O(sessions x trials) list work per iteration, minutes per animal. Stacking
    # each session's trials with a label vector makes a permutation one `rng.permutation` and six
    # boolean means in numpy.
    stacks = []
    for si in range(max((len(v) for v in post_by_session.values()), default=0)):
        Xs, ys = [], []
        for p in labels:
            v = post_by_session.get(p, [])
            if si < len(v) and len(v[si]):
                Xs.append(v[si])
                ys += [p] * len(v[si])
        if Xs:
            stacks.append((np.vstack(Xs), np.array(ys)))
    for b in range(n_boot):
        rm = {q: _strat_mean(ref_by_session.get(q, []), rng) for q in labels}
        pm = {p: _strat_mean(post_by_session.get(p, []), rng) for p in labels}
        for i, p in enumerate(labels):
            for j, q in enumerate(labels):
                if pm.get(p) is None or rm.get(q) is None:
                    continue
                boots[b, i, j] = float(np.corrcoef(pm[p], rm[q])[0, 1])
        tot, cnt = {}, {}
        for Xs, ys in stacks:
            yp = ys[rng.permutation(len(ys))]        # labels shuffled WITHIN this session
            for p in labels:
                m = yp == p
                if m.any():
                    tot[p] = Xs[m].sum(0) if p not in tot else tot[p] + Xs[m].sum(0)
                    cnt[p] = cnt.get(p, 0) + int(m.sum())
        pmn = {p: tot[p] / cnt[p] for p in tot if cnt.get(p)}
        for i, p in enumerate(labels):
            for j, q in enumerate(labels):
                if pmn.get(p) is None or rm.get(q) is None:
                    continue
                nulls[b, i, j] = float(np.corrcoef(pmn[p], rm[q])[0, 1])
    with np.errstate(invalid="ignore"):
        lo = np.nanpercentile(boots, 2.5, axis=0)
        hi = np.nanpercentile(boots, 97.5, axis=0)
        null_hi = np.nanpercentile(np.abs(nulls), 97.5, axis=0)
    # `boots` is returned so a CALLER can difference two matrices draw-by-draw. Differencing the
    # published CIs instead would be wrong: the post and baseline panels share the same resampled
    # reference, so their errors are correlated and an unpaired difference overstates the interval.
    return obs, lo, hi, null_hi, boots


def _mean_pattern(X):
    return X.mean(0) if len(X) else None


def fig_pattern_similarity(out_dir, min_trials=10):
    """6: WITHIN- and ACROSS-position pattern similarity, model-free.

    Priya, 2026-08-25. The complement to the coding directions: correlate the post-stroke MEAN
    ACTIVITY PATTERN at each position against the pre-stroke mean pattern at EVERY position. No
    discriminant, no contrast -- which is exactly why it survives where the coding directions do
    not. A pairwise axis needs trials on BOTH sides, so it fails at the positions the lesion broke;
    a mean pattern for far_R is perfectly well defined from 400 miss trials with no partner at all.

    DIAGONAL = within-position ("is this still the same code"). OFF-DIAGONAL = across-position
    ("what does it look like instead"). `pattern_similarity` in `poststroke_compare` already computes
    the diagonal; the off-diagonal is what is new here.

    THE BASELINE PANEL IS NOT OPTIONAL. Positions are intrinsically similar before any lesion, so a
    raw r of 0.8 between post far_R and pre far_L means nothing on its own. Both panels are scored
    against the SAME reference -- one half of the pre-stroke trials -- so:
        LEFT  corr(other pre-stroke half at P, reference at Q): the no-lesion expectation, and its
              DIAGONAL is the split-half reliability, i.e. the ceiling this measure can reach.
        RIGHT corr(post-stroke at P, reference at Q).
    Comparing the right panel to the left is the only way to read it; comparing it to 1.0 is not.

    CLASS VARIANTS. `lick` uses post-stroke trials with a lick. `working` adds miss-while-working
    (everything but the terminal quit period) and exists for ENL and cue ONLY -- in the lick window
    a no-lick trial is placed at the CUE, so pooling the two classes there would average patterns
    from two different times and call the result a position effect.

    WHAT THIS SHARES WITH NOTHING ELSE, and its weakness: mean-pattern correlation is sensitive to
    global gain and offset, so a uniform post-stroke amplitude change moves every cell together.
    The coding directions are immune to that by construction (unit vectors). The two measures agree
    or they do not, and agreement is the claim worth making.
    """
    from wfield_local import joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
    from wfield_local.locanmf_frozen_decoder import pool_sessions
    from wfield_local.position_coding_directions import _gate_all
    from wfield_local.precue_engagement_states import features_with_indices

    rng = np.random.default_rng(0)
    made = []
    for _disp, align, wname in WINDOWS:
        variants = ("lick",) if align == "lick" else ("lick", "working")
        store = {v: {} for v in variants}
        for an in ANIMALS:
            pre = [x for x in config.phase_labels("pre") if x.startswith(an)]
            post = [x for x in config.phase_labels("post") if x.startswith(an)]
            try:
                basis = joint_locanmf.load(an, sessions=SESSIONS)
                feat = features_with_indices(basis, nolick_ref="cue")
                XE, YE, GE, _B, XU, YU, kept, _c, GU = pool_sessions(
                    pre + post, source="locanmf", align=align, post_s=2.0, features=feat)
                g = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
                not_eng = g[0] if g else np.zeros(len(YU), bool)
                pre_i = {i for i, lab in enumerate(kept) if lab in set(pre)}
                GU = np.asarray(GU)
                # No e_pre mask: the pre-stroke split is now over SESSION IDS, not over a pooled
                # trial mask, so trials are selected per session throughout.
                en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
                un = (np.array([POSITION_NAMES.get(int(v), str(v)) for v in YU])
                      if len(YU) else np.zeros(0, str))
                # SPLIT THE PRE-STROKE **SESSIONS**, NOT THE TRIALS (corrected 2026-08-25).
                #
                # Until now this drew a random half of the pooled pre-stroke TRIALS as the
                # reference and the other half as the no-lesion expectation. Both halves then came
                # from the SAME DAYS, so the baseline panel contained within-session trial noise
                # and NO between-session drift whatever -- while the post panel compares different
                # days against those days. The baseline was therefore an upper bound no
                # across-session comparison can reach, and "post minus baseline is negative at
                # every position" was measured against it.
                #
                # Splitting by session makes both sides "one set of DAYS against another set of
                # DAYS", which is what the post panel is. Trials stay GROUPED BY SESSION because
                # the stratified bootstrap resamples within sessions and a mean has already thrown
                # that structure away.
                ref, other = {}, {}
                pre_ids = sorted(pre_i)
                sh = rng.permutation(len(pre_ids))
                g_ref = {pre_ids[k] for k in sh[:max(1, len(pre_ids) // 2)]}
                g_oth = {pre_ids[k] for k in sh[max(1, len(pre_ids) // 2):]}
                for p in CONF_LABELS:
                    a = [XE[(GE == i) & (en == p)] for i in pre_ids if i in g_ref]
                    b = [XE[(GE == i) & (en == p)] for i in pre_ids if i in g_oth]
                    a = [z for z in a if len(z)]
                    b = [z for z in b if len(z)]
                    if sum(len(z) for z in a) < min_trials or sum(len(z) for z in b) < min_trials:
                        continue
                    ref[p], other[p] = a, b
                post_ids = [i for i in range(len(kept)) if i not in pre_i]
                for v in variants:
                    postm = {}
                    for p in CONF_LABELS:
                        by_sess, tot = [], 0
                        for i in post_ids:
                            parts = [XE[(GE == i) & (en == p)]]
                            if v == "working" and len(un):
                                m = (GU == i) & (un == p) & ~not_eng
                                if m.any():
                                    parts.append(XU[m])
                            keep = [q for q in parts if len(q)]
                            if keep:
                                Z = np.vstack(keep)
                                by_sess.append(Z)
                                tot += len(Z)
                        if tot >= min_trials:
                            postm[p] = by_sess
                    store[v][an] = (ref, other, postm)
            except Exception as ex:                                       # noqa: BLE001
                print(f"  !! 6 {an} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
        for v in variants:
            fig, axes = plt.subplots(len(ANIMALS), 3, figsize=(10.4, 12.6), squeeze=False)
            drew = False
            for ri, an in enumerate(ANIMALS):
                got = store[v].get(an)
                if not got:
                    for ci in range(3):
                        axes[ri][ci].axis("off")
                    continue
                ref, other, postm = got
                # SAME SEED FOR BOTH PANELS so the reference is resampled identically and the
                # post-minus-baseline difference can be taken draw by draw.
                seed = abs(hash((an, align, v))) % (2 ** 31)
                base_obs, _bl, _bh, base_null, base_bt = _pattern_stats(
                    other, ref, CONF_LABELS, np.random.default_rng(seed))
                post_obs, _pl, _ph, post_null, post_bt = _pattern_stats(
                    postm, ref, CONF_LABELS, np.random.default_rng(seed))
                for ci, (M, NH, ptitle) in enumerate(
                        ((base_obs, base_null, "PRE-stroke, other half\n(no-lesion expectation)"),
                         (post_obs, post_null, "POST-stroke"))):
                    ax = axes[ri][ci]
                    im = ax.imshow(np.ma.masked_invalid(M), vmin=-1, vmax=1, cmap="RdBu_r")
                    for i in range(len(CONF_LABELS)):
                        for j in range(len(CONF_LABELS)):
                            if not np.isfinite(M[i, j]):
                                continue
                            _txt(ax, j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                                    fontsize=7.5, color="k")
                            # RING = beats the position-shuffled null. Not "r != 0": the null keeps
                            # the global post-stroke pattern and shuffles only WHICH position a
                            # trial belongs to, so a ring means position-specific structure.
                            if np.isfinite(NH[i, j]) and abs(M[i, j]) > NH[i, j]:
                                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                                           edgecolor="lime", lw=1.6))
                    ax.set_xticks(range(len(CONF_LABELS)))
                    ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                       rotation=90, ha="center", fontsize=9.5)
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9.5)
                    ax.set_title(f"{an if ci == 0 else ''}  {ptitle}", fontsize=11,
                                 fontweight="bold" if ci == 0 else "normal")
                    if ci == 0:
                        ax.set_ylabel("this position's pattern", fontsize=9.5)
                    if ri == len(ANIMALS) - 1:
                        ax.set_xlabel("vs PRE-STROKE reference at", fontsize=9.5)
                    drew = True
                # THIRD PANEL: the claim, with an interval. Own-position r post MINUS baseline,
                # differenced draw by draw so the shared reference cancels.
                ax = axes[ri][2]
                d = np.array([post_bt[:, k, k] - base_bt[:, k, k]
                              for k in range(len(CONF_LABELS))])
                y = np.arange(len(CONF_LABELS))
                with np.errstate(invalid="ignore"):
                    med = np.nanmedian(d, axis=1)
                    lo = np.nanpercentile(d, 2.5, axis=1)
                    hi = np.nanpercentile(d, 97.5, axis=1)
                ok = np.isfinite(med)
                ax.errorbar(med[ok], y[ok], xerr=[med[ok] - lo[ok], hi[ok] - med[ok]],
                            fmt="o", ms=5, color="#b2182b", capsize=3, lw=1.2)
                ax.axvline(0, color="k", lw=1.0)
                ax.set_yticks(y)
                ax.set_yticklabels([])
                ax.set_ylim(len(CONF_LABELS) - 0.5, -0.5)
                ax.set_title("post − baseline (own position)", fontsize=11)
                ax.grid(alpha=0.25, lw=0.5)
                if ri == len(ANIMALS) - 1:
                    ax.set_xlabel("Δr, 95% stratified bootstrap", fontsize=9.5)
            if not drew:
                plt.close(fig)
                continue
            fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03, label="pattern correlation r")
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            _suptitle(fig, f"Mean-pattern similarity, within and across positions — {wname} window\n"
                         f"Post-stroke class: {cls}.  Rows = the pattern being described, columns = "
                         f"the PRE-STROKE reference it is correlated with.\n"
                         "DIAGONAL = is it still the same code. OFF-DIAGONAL = what it looks like "
                         "instead. READ THE RIGHT PANEL AGAINST THE LEFT, not against 1.0:\n"
                         "positions are intrinsically similar before any lesion, and the left "
                         "diagonal is the split-half ceiling this measure can reach.", fontsize=9)
            _footer(fig)
            p = Path(out_dir) / f"grant_6_pattern_{align}_{v}.png"
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made[0] if len(made) == 1 else (made or None)


# ------------------------------------------------------------------ 3a. coding retained
# ---------------------------------------------------------------------------------------------
# 7 / 7b: IS THE "LOST CODE" JUST A NOISIER ONE?  (Priya, 2026-08-25)
# ---------------------------------------------------------------------------------------------

#: Reliability below this makes the disattenuation ratio unstable -- dividing by sqrt(0.1) inflates
#: both the estimate and its error without bound. Same threshold and same reasoning as the coding
#: directions' MIN_REL, so the two analyses declare a cell uninterpretable on the same criterion.
MIN_REL = 0.5

#: Split-half draws per cell. The split is random, so one draw is itself a noisy estimate of the
#: reliability; averaging over draws costs nothing (a correlation of two means) and removes it.
SPLIT_REPS = 40


def _pooled_bundle(an, align):
    """The shared load behind figures 6, 6b, 7 and 8: joint basis, pooled sessions, engagement gate.

    Extracted because it was character-identical in `fig_pattern_similarity` and
    `fig_pattern_similarity_per_session`, and a third and fourth copy is how two figures that claim
    to describe the same trials quietly stop doing so.
    """
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
    return {"XE": XE, "en": en, "GE": np.asarray(GE), "XU": XU, "un": un, "GU": GU,
            "BE": BE_all, "BU": BU_all,
            "not_eng": not_eng, "kept": kept, "pre_i": pre_i,
            "e_pre": np.isin(np.asarray(GE), list(pre_i))}


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


def _session_trials(bd, i, q, variant, field="X"):
    """Trials (or their BLOCK IDS) for session ``i`` at position ``q`` under a trial class.

    ``lick`` is the engaged (licking) set; ``working`` adds miss-while-working, i.e. everything but
    the terminal quit period. Returns an empty array rather than None so callers can stack freely.

    ``field`` selects what comes back -- "X" the patterns, "blk" the block id of each of those same
    rows. THE MASK IS COMPUTED ONCE HERE for both, so the two cannot drift apart; a bootstrap whose
    block vector did not line up with its data would silently resample the wrong trials.
    """
    me = (bd["GE"] == i) & (bd["en"] == q)
    mu = None
    if variant == "working" and len(bd["un"]):
        m = (bd["GU"] == i) & (bd["un"] == q) & ~bd["not_eng"]
        mu = m if m.any() else None
    if field == "blk":
        parts = [bd["BE"][me]] + ([bd["BU"][mu]] if mu is not None else [])
        keep = [z for z in parts if len(z)]
        return np.concatenate(keep) if keep else np.zeros(0, np.int64)
    parts = [bd["XE"][me]] + ([bd["XU"][mu]] if mu is not None else [])
    keep = [z for z in parts if len(z)]
    return np.vstack(keep) if keep else np.zeros((0, bd["XE"].shape[1]))


def _split_half(Z, rng, reps=SPLIT_REPS):
    """Mean correlation between the means of two disjoint halves of ``Z``. NaN under 4 trials.

    This is the reliability of the MEAN PATTERN this many trials can support -- the ceiling any
    correlation involving that mean can reach, and the quantity figure 6 does not show.
    """
    n = len(Z)
    if n < 4:
        return np.nan
    h = n // 2
    rs = []
    for _ in range(reps):
        idx = rng.permutation(n)
        a, b = Z[idx[:h]].mean(0), Z[idx[h:2 * h]].mean(0)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(rs)) if rs else np.nan


def _reliability(Z, rng, reps=SPLIT_REPS):
    """Reliability of the mean of ALL of ``Z``, not of half of it.

    `_split_half` correlates two means built from n/2 trials each, so it estimates the reliability
    of an n/2-trial mean -- but the quantity figure 6 correlates is the mean of all n. Spearman-Brown
    projects the split-half value up to the full length: rel(n) = 2r / (1 + r).

    THIS IS NOT COSMETIC. Skipping it makes every reliability too LOW, and since 7b divides by
    sqrt(rel_post * rel_pre), too low a denominator makes every disattenuated correlation too HIGH --
    i.e. it would systematically manufacture the "the code moved" verdict the panel exists to test.
    """
    r = _split_half(Z, rng, reps)
    if not np.isfinite(r):
        return np.nan
    if r <= -1:
        return np.nan
    return float(2.0 * r / (1.0 + r))


def _split_half_matrix(src, rng, labels=None):
    """6x6 WITHIN-set split-half matrix, SYMMETRIC by construction.

    The diagonal is each position's own reliability -- corr(half A at P, half B at P), which must
    use the two halves because a mean correlated with itself is 1 by definition. The off-diagonal is
    how similar two positions look to each other MEASURED IN THE SAME SESSION, the within-session
    counterpart of figure 6's off-diagonal.

    WHY IT IS AVERAGED OVER BOTH PAIRINGS (Priya, 2026-08-25: "why aren't the fig 7 matrices
    symmetrical about the diagonal?"). The first version took M[P,Q] = corr(A_P, B_Q) and left it,
    so cell (far_R, close_L) and cell (close_L, far_R) were corr(A_far_R, B_close_L) and
    corr(A_close_L, B_far_R) -- two estimates of ONE quantity, differing only in which random half
    of each position's trials landed on which side. The asymmetry was therefore pure estimation
    noise being drawn as if it were structure, in a figure whose entire job is to say how much
    estimation noise there is. Averaging the two pairings is symmetric, has the same expectation, is
    still cross-validated (no half is ever correlated with itself) and halves the variance.

    The remaining asymmetry is zero by construction; if the two pairings disagreed a lot that fact
    is worth knowing, so `_split_half_asymmetry` reports it separately rather than smuggling it into
    the picture.
    """
    labels = labels or CONF_LABELS
    n = len(labels)
    M = np.full((n, n), np.nan)
    halves = {}
    for q in labels:
        Z = src.get(q)
        if Z is None or len(Z) < 4:
            continue
        idx = rng.permutation(len(Z))
        h = len(Z) // 2
        halves[q] = (Z[idx[:h]].mean(0), Z[idx[h:2 * h]].mean(0))

    def _r(u, v):
        return float(np.corrcoef(u, v)[0, 1]) if (np.std(u) and np.std(v)) else np.nan

    for i, p in enumerate(labels):
        if p not in halves:
            continue
        for j, q in enumerate(labels):
            if q not in halves or j < i:
                continue
            if i == j:
                M[i, j] = _r(halves[p][0], halves[p][1])
                continue
            both = [_r(halves[p][0], halves[q][1]), _r(halves[q][0], halves[p][1])]
            both = [x for x in both if np.isfinite(x)]
            M[i, j] = M[j, i] = float(np.mean(both)) if both else np.nan
    return M


def _split_half_asymmetry(src, rng, labels=None):
    """Mean |corr(A_P, B_Q) - corr(A_Q, B_P)| over off-diagonal pairs -- a pure noise read.

    The two orderings estimate the same thing, so any difference is sampling noise in the split.
    Useful as a diagnostic on how much to trust a panel; deliberately NOT drawn into the matrix.
    """
    labels = labels or CONF_LABELS
    halves = {}
    for q in labels:
        Z = src.get(q)
        if Z is None or len(Z) < 4:
            continue
        idx = rng.permutation(len(Z))
        h = len(Z) // 2
        halves[q] = (Z[idx[:h]].mean(0), Z[idx[h:2 * h]].mean(0))
    d = []
    ks = [q for q in labels if q in halves]
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            p, q = ks[a], ks[b]
            r1 = float(np.corrcoef(halves[p][0], halves[q][1])[0, 1])
            r2 = float(np.corrcoef(halves[q][0], halves[p][1])[0, 1])
            if np.isfinite(r1) and np.isfinite(r2):
                d.append(abs(r1 - r2))
    return float(np.mean(d)) if d else np.nan


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


def fig_splithalf_matrix(out_dir, min_trials=10):
    """7: the WITHIN-session split-half matrix for every post-stroke session.

    Figure 6 asks whether the post-stroke pattern still matches the pre-stroke one. It cannot
    distinguish a code that MOVED from a code that merely became NOISIER, because a correlation
    between two means is bounded above by the reliability of each mean, and a session with more
    variable responses has a lower ceiling at every position at once. That is a live alternative
    here: the headline result is that own-position similarity drops at EVERY position with far_R
    largest, which is precisely the signature of a global change in how repeatable the responses are.

    This figure supplies the missing ceiling. Both halves come from the SAME session, so the
    diagonal is that session's own reliability and nothing about the lesion, the pre-stroke
    reference or the alignment enters it. Off-diagonal is how similar the positions look to each
    other WITHIN one session.

    Read it beside figure 6 panel by panel: where 6's diagonal falls and this diagonal does not, the
    code moved; where both fall together, the code is noisier and 6 cannot see the difference. 7b
    does that division explicitly.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        for v in (("lick",) if align == "lick" else ("lick", "working")):
            store, days = _collect_7(align, v, min_trials)
            if not days:
                continue
            ncol = 1 + len(days)
            fig, axes = plt.subplots(len(ANIMALS), ncol, figsize=(_colw() * ncol + 1.2, 8.7),
                                     gridspec_kw={"hspace": 0.45},
                                     squeeze=False)
            im = None
            for ri, an in enumerate(ANIMALS):
                got = store.get(an)
                for ci in range(ncol):
                    ax = axes[ri][ci]
                    src = None
                    if got:
                        pre_by_sess, by_day = got
                        src = pre_by_sess if ci == 0 else by_day.get(days[ci - 1])
                    if not src:
                        ax.axis("off")
                        continue
                    rng = np.random.default_rng(abs(hash((an, align, v, ci))) % (2 ** 31))
                    if ci == 0:
                        # COLUMN 0 IS ONE PRE-STROKE SESSION AT A TIME, AVERAGED -- not the pooled
                        # set. Pooling six sessions gives the reliability of a six-session mean and
                        # compares it against one-session post-stroke means, so most of the gap
                        # would be trial count rather than the lesion. Averaging per-session
                        # matrices matches the units of every other column in the row.
                        Ms = [_split_half_matrix(p, rng) for p in pre_by_sess.values()]
                        M = np.nanmean(np.stack(Ms), axis=0) if Ms else np.full((6, 6), np.nan)
                        n_med = int(np.median([len(z) for p in pre_by_sess.values()
                                               for z in p.values()] or [0]))
                    else:
                        M = _split_half_matrix(src, rng)
                        n_med = int(np.median([len(z) for z in src.values()] or [0]))
                    im = ax.imshow(np.ma.masked_invalid(M), vmin=-1, vmax=1, cmap="RdBu_r")
                    ax.set_xticks(range(len(CONF_LABELS)))
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                       rotation=90, fontsize=9)
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
                    head = "PRE" if ci == 0 else f"day {days[ci - 1]}"
                    # 'sh', NOT 'rel': this is the split-half correlation itself, the reliability of
                    # a HALF-length mean. 7b applies the Spearman-Brown step that turns it into the
                    # reliability of the full mean it actually divides by.
                    # n IS PRINTED because sh depends on it, and a reader comparing two panels needs
                    # to know whether they rest on comparable amounts of data.
                    ax.set_title(f"{head}\nsh {np.nanmean(np.diag(M)):.2f}  n{n_med}", fontsize=9.5,
                                 fontweight="bold" if ci == 0 else "normal")
                    if ci == 0:
                        ax.set_ylabel(f"{an}\nhalf A at", fontsize=11, fontweight="bold")
            if im is None:
                plt.close(fig)
                continue
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label="split-half correlation r")
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            _suptitle(fig, 
                f"WITHIN-session split-half pattern similarity — {wname} window\n"
                f"Post-stroke class: {cls}.  BOTH HALVES COME FROM THE SAME SESSION: no lesion "
                f"comparison, no pre-stroke reference, no alignment inference enters this.\n"
                f"SYMMETRIC BY CONSTRUCTION: an off-diagonal cell averages both half-pairings, "
                "since (P,Q) and (Q,P) estimate one quantity and differ only by split noise.\n"
                f"DIAGONAL ('sh' above each panel) = that session's own reliability, i.e. the "
                f"CEILING figure 6's correlations can reach. Off-diagonal = how similar the "
                f"positions look to each other within one session.\n"
                f"If a post-stroke diagonal is low HERE, figure 6's matching low value is a noisier "
                "code, not a moved one. Columns are days from lesion.\n"
                f"FIRST COLUMN IS ONE PRE-STROKE SESSION AT A TIME, AVERAGED -- not the pooled set: "
                f"split-half reliability rises with trial count, and pooling six sessions would "
                f"compare a six-session mean with one-session post-stroke means. 'n' is the median "
                f"trials per position in that panel.", fontsize=9.0)
            _footer(fig)
            p = _out(out_dir, f"grant_7_splithalf_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made


def fig_reliability_verdict(out_dir, min_trials=10):
    """7b: figure 6's own-position similarity, RAW and DISATTENUATED by both sides' reliability.

    THE ARITHMETIC. A correlation between two independently estimated means is attenuated by each
    side's reliability: E[r_obs] ~ r_true * sqrt(rel_post * rel_ref). Dividing it out estimates the
    correlation the two patterns would have if both were measured without noise -- so a drop that
    survives disattenuation is a code that MOVED, and a drop that disappears was a code measured
    less repeatably. This is the same correction the coding-direction analysis has used since
    2026-08-20, applied to the pattern measure, which has never had it.

    WHERE IT CANNOT BE TRUSTED. Dividing by sqrt(rel) with rel near zero inflates without bound, so
    a cell whose reliability is below MIN_REL on either side is drawn hollow and its number
    suppressed. That is not a formality: an impaired position post-stroke is exactly where trials
    are fewest and reliability lowest, which is exactly where the ratio is least stable -- the
    correction is weakest precisely where the question is sharpest, and the figure has to say so
    rather than print a confident number.

    A disattenuated value may exceed 1. That is expected of a ratio estimator at low reliability and
    is left visible rather than clipped: clipping would hide the instability the hollow marker is
    there to declare.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        for v in (("lick",) if align == "lick" else ("lick", "working")):
            store, days = _collect_7(align, v, min_trials)
            if not days:
                continue
            fig, axes = plt.subplots(len(ANIMALS), 3, figsize=(10.8, 9.4), squeeze=False)
            drew = False
            for ri, an in enumerate(ANIMALS):
                got = store.get(an)
                if not got:
                    for ci in range(3):
                        axes[ri][ci].axis("off")
                    continue
                pre_by_sess, by_day = got
                rng = np.random.default_rng(abs(hash((an, align, v))) % (2 ** 31))
                # COLUMN 0 IS THE NO-LESION EXPECTATION, built LEAVE-ONE-SESSION-OUT: each
                # pre-stroke session in turn is scored against the pool of the OTHERS, then the six
                # results are averaged. Disjoint, so it is not a session correlated against itself;
                # and one session against a pool, exactly like every post-stroke column, so the
                # reliability shown for it is a ONE-SESSION reliability and can be compared with
                # them.
                #
                # IT DOES NOT COME OUT AT 1, AND THAT IS THE POINT. PS94 post-cue disattenuates to
                # 0.75-0.86 here, not 1.0, because disattenuating by a WITHIN-session reliability
                # removes within-session trial noise and nothing else: a pre-stroke session's mean
                # pattern also differs from the other sessions' by day-to-day drift, which no
                # within-session split half can see. So this column, not 1.0, is the ceiling the
                # post-stroke columns are to be read against -- the same reason figure 6 has a
                # baseline panel and the same reason the axis work uses a matched null instead of
                # comparing cosines with unity.
                cols = ["PRE"] + [f"d{d}" for d in days]
                shape = (len(CONF_LABELS), len(cols))
                rel = np.full(shape, np.nan)
                raw = np.full(shape, np.nan)
                dis = np.full(shape, np.nan)

                def _score(scored, ref_trials, rng=rng):
                    """(reliability, raw r, disattenuated r) per position for one scored session."""
                    o_rel = np.full(len(CONF_LABELS), np.nan)
                    o_raw = np.full(len(CONF_LABELS), np.nan)
                    o_dis = np.full(len(CONF_LABELS), np.nan)
                    for i, q in enumerate(CONF_LABELS):
                        Z, R = scored.get(q), ref_trials.get(q)
                        if Z is None or R is None:
                            continue
                        o_rel[i] = _reliability(Z, rng)
                        rr = _reliability(R, rng)
                        m, rm = Z.mean(0), R.mean(0)
                        if np.std(m) and np.std(rm):
                            o_raw[i] = float(np.corrcoef(m, rm)[0, 1])
                        if (np.isfinite(rr) and np.isfinite(o_rel[i])
                                and rr >= MIN_REL and o_rel[i] >= MIN_REL):
                            o_dis[i] = o_raw[i] / np.sqrt(rr * o_rel[i])
                    return o_rel, o_raw, o_dis

                held = [_score(pat, _pre_reference(pre_by_sess, exclude=s))
                        for s, pat in pre_by_sess.items()]
                if held:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN position rows
                        rel[:, 0] = np.nanmean([h[0] for h in held], axis=0)
                        raw[:, 0] = np.nanmean([h[1] for h in held], axis=0)
                        dis[:, 0] = np.nanmean([h[2] for h in held], axis=0)
                full_ref = _pre_reference(pre_by_sess)
                for ci in range(1, len(cols)):
                    a, b, c = _score(by_day.get(days[ci - 1]) or {}, full_ref)
                    rel[:, ci], raw[:, ci], dis[:, ci] = a, b, c
                for ci, (M, ttl, vmin, vmax, cmap) in enumerate((
                        (rel, ("RELIABILITY of that session's own mean\n"
                               "(split half, Spearman-Brown to full length)"),
                         0.0, 1.0, "viridis"),
                        (raw, "RAW similarity to pre-stroke\n(this is figure 6's diagonal)",
                         -1.0, 1.0, "RdBu_r"),
                        (dis, "DISATTENUATED\n(raw / sqrt(rel_post x rel_pre))",
                         -1.0, 1.0, "RdBu_r"))):
                    ax = axes[ri][ci]
                    ax.imshow(np.ma.masked_invalid(M), vmin=vmin, vmax=vmax, cmap=cmap,
                              aspect="auto")
                    for i in range(len(CONF_LABELS)):
                        for j in range(len(cols)):
                            if not np.isfinite(M[i, j]):
                                # A CELL SUPPRESSED BY MIN_REL IS NOT A CELL WITH NO DATA. Mark the
                                # first case so it cannot be read as the second.
                                if ci == 2 and np.isfinite(raw[i, j]):
                                    _txt(ax, j, i, "·", ha="center", va="center", fontsize=11,
                                            color="0.35")
                                continue
                            _txt(ax, j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                                    color="w" if (ci == 0 and M[i, j] < 0.5) else "k")
                    ax.set_xticks(range(len(cols)))
                    ax.set_xticklabels(cols if ri == len(ANIMALS) - 1 else [], fontsize=9.5)
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9.5)
                    if ci == 0:
                        ax.set_ylabel(an, fontsize=11.5, fontweight="bold")
                    if ri == 0:
                        ax.set_title(ttl, fontsize=11)
                    drew = True
                # THE DENOMINATOR, spelled out. The ratio divides by the reference's reliability as
                # well as the session's, and a reader cannot otherwise tell whether a low
                # disattenuated value came from a weak numerator or a strong denominator.
                rel_ref = {q: _reliability(Z, rng) for q, Z in full_ref.items()}
                axes[ri][0].set_xlabel(
                    "pooled pre-stroke reference reliability: "
                    + "  ".join(f"{q.replace('close_', 'c').replace('far_', 'f')}"
                                f" {rel_ref.get(q, float('nan')):.2f}" for q in CONF_LABELS),
                    fontsize=5.6)
            if not drew:
                plt.close(fig)
                continue
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            _suptitle(fig, 
                f"Is the lost code a MOVED code or a NOISIER one? — {wname} window\n"
                f"Post-stroke class: {cls}.  Rows = spout position, columns = days from lesion.\n"
                f"LEFT: how repeatable that session's own pattern is (split half, within session). "
                f"MIDDLE: figure 6's own-position correlation to the pre-stroke reference.\n"
                f"RIGHT: the middle divided by sqrt(rel_post x rel_pre) -- what the correlation "
                f"would be if both means were measured without noise.\n"
                f"A drop that SURVIVES the right panel is a code that moved; a drop that "
                f"DISAPPEARS was a code measured less repeatably. A grey dot = reliability below "
                f"{MIN_REL} on one side, where the ratio is not stable enough to print.\n"
                f"PRE COLUMN IS LEAVE-ONE-SESSION-OUT: each pre-stroke session scored against the "
                f"pool of the others, averaged -- one session against a pool, exactly like every "
                f"post-stroke column, so the columns are comparable.\n"
                f"IT IS THE CEILING, AND IT IS NOT 1.0: disattenuating by a within-session "
                f"reliability removes trial noise and NOT day-to-day drift, which a pre-stroke "
                f"session also carries. Read the post columns against PRE, never against 1.",
                fontsize=9.5)
            fig.tight_layout(rect=(0, 0, 1, 1.0))   # top reserved by _suptitle
            _footer(fig)
            p = Path(out_dir) / f"grant_7b_reliability_{align}_{v}.png"
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made


# ---------------------------------------------------------------------------------------------
# 8 / 8b: THE CROSSNOBIS VERSION -- and what it can and cannot say per position
# ---------------------------------------------------------------------------------------------
#
# Priya, 2026-08-25: "would this still give us any per-position information though? or just
# overall representational geometry similarity?"  BOTH, but they are different questions and the
# two figures below separate them deliberately.
#
#   8   CROSS-SET distances d(post at P, pre at Q). The direct translation of figure 6: per
#       position, fully, with the diagonal reading "did this position's pattern move". Noise-
#       UNBIASED, which figure 6's correlation is not. Still sensitive to a global amplitude
#       change, exactly as figure 6 is -- an unbiased distance between two patterns is not a
#       gain-invariant one.
#
#   8b  SECOND-ORDER: the within-set 6x6 RDM for each session correlated against the pre-stroke
#       RDM. This is RSA proper, and it IS gain-invariant, because scaling every distance leaves
#       a correlation between RDMs unchanged. Per-position information survives as each
#       position's ROW -- its five distances to the other positions -- so "is far_R still
#       arranged the way it was relative to everything else" is answerable, while "did far_R's
#       pattern move" is not. That is the trade: 8 keeps the position and loses gain-invariance,
#       8b keeps gain-invariance and can only speak about a position's RELATIONS.
#
# Together they bracket the headline pattern result. If the graded drop at every position in
# figure 6 were a global amplitude change, 8 would show it and 8b would NOT.


def _whitener(res):
    """Inverse residual covariance, Ledoit-Wolf shrunk, with a diagonal fallback.

    Estimated from trials MINUS their own cell mean, so it is independent of the differences it
    later whitens -- which is what keeps the cross-validated product unbiased.
    """
    if not res:
        return None
    R = np.vstack(res)
    if len(R) < 3:
        return None
    try:
        from sklearn.covariance import LedoitWolf
        return np.linalg.pinv(LedoitWolf().fit(R).covariance_)
    except Exception:                                                # noqa: BLE001
        return np.diag(1.0 / np.maximum(R.var(axis=0), 1e-12))


def _halves(src, rng, labels, min_n=4):
    """Per-position (half A mean, half B mean) plus the residuals both halves leave behind."""
    m0, m1, res = {}, {}, []
    for q in labels:
        Z = src.get(q)
        if Z is None or len(Z) < min_n:
            continue
        idx = rng.permutation(len(Z))
        h = len(Z) // 2
        A, B = Z[idx[:h]], Z[idx[h:2 * h]]
        m0[q], m1[q] = A.mean(0), B.mean(0)
        res.append(A - m0[q])
        res.append(B - m1[q])
    return m0, m1, res


def _crossnobis_within(src, rng, labels):
    """Noise-unbiased 6x6 RDM within ONE set of trials. Diagonal is 0 by construction, left NaN."""
    m0, m1, res = _halves(src, rng, labels)
    P = _whitener(res)
    if P is None or len(m0) < 2:
        return np.full((len(labels), len(labels)), np.nan)
    D = np.full((len(labels), len(labels)), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i >= j or a not in m0 or b not in m0:
                continue
            D[i, j] = D[j, i] = float((m0[a] - m0[b]) @ P @ (m1[a] - m1[b]))
    return D


def _crossnobis_cross(post, ref, rng, labels):
    """Unbiased d(post at P, reference at Q) for every ordered pair -- figure 6's matrix as distance.

    The two factors of the product use DISJOINT halves on both sides (post half A against reference
    half 1, post half B against reference half 2), so trial noise contributes zero in expectation
    and the diagonal is an unbiased estimate of how far the post-stroke pattern has actually moved.
    A raw squared distance would instead grow with noise alone, which is the bias this removes and
    the reason a plain distance version of figure 6 would have been unreadable across sessions whose
    amplitude differs 2-3x.
    """
    pm0, pm1, pres = _halves(post, rng, labels)
    rm0, rm1, rres = _halves(ref, rng, labels)
    P = _whitener(pres + rres)
    if P is None:
        return np.full((len(labels), len(labels)), np.nan)
    D = np.full((len(labels), len(labels)), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if a not in pm0 or b not in rm0:
                continue
            D[i, j] = float((pm0[a] - rm0[b]) @ P @ (pm1[a] - rm1[b]))
    return D


def _triu_vals(D):
    iu = np.triu_indices(D.shape[0], 1)
    return D[iu]


def fig_crossnobis_cross(out_dir, min_trials=10):
    """8: figure 6 rebuilt on cross-validated (crossnobis) distances instead of correlations.

    ROWS = the post-stroke position, COLUMNS = the pre-stroke reference position, matching figure 6.
    The DIAGONAL is what changed: near zero means the pattern did not move, and unlike a correlation
    it is not bounded by how repeatable either mean was -- the cross-validated product is unbiased by
    trial noise, so a session with more variable responses does not automatically read as a bigger
    distance. That is the one thing figure 6 cannot do and the reason this exists.

    UNITS. Distances are divided by the mean pre-stroke within-set pairwise distance for that animal
    and window, so 1.0 = "as far apart as two different pre-stroke positions were". Raw crossnobis
    units depend on the whitener and the dimensionality and are not comparable across animals.

    WHAT IT STILL CANNOT DO: it is a distance between two patterns, so a uniform post-stroke gain
    change moves every cell -- the same exposure figure 6 has. 8b is the gain-invariant companion.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        for v in (("lick",) if align == "lick" else ("lick", "working")):
            store, days = _collect_7(align, v, min_trials)
            if not days:
                continue
            ncol = 1 + len(days)
            fig, axes = plt.subplots(len(ANIMALS), ncol, figsize=(_colw() * ncol + 1.2, 8.7),
                                     gridspec_kw={"hspace": 0.45},
                                     squeeze=False)
            im = None
            for ri, an in enumerate(ANIMALS):
                got = store.get(an)
                if not got:
                    for ci in range(ncol):
                        axes[ri][ci].axis("off")
                    continue
                pre_by_sess, by_day = got
                rng = np.random.default_rng(abs(hash((an, align, v))) % (2 ** 31))
                full_ref = _pre_reference(pre_by_sess)
                scale = np.nanmean(_triu_vals(_crossnobis_within(full_ref, rng, CONF_LABELS)))
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
                for ci in range(ncol):
                    ax = axes[ri][ci]
                    # COLUMN 0 IS THE NO-LESION EXPECTATION, LEAVE-ONE-SESSION-OUT: each
                    # pre-stroke session scored against the pool of the others, averaged. One
                    # session against a pool, exactly like every post-stroke column, so the columns
                    # are comparable; disjoint, so nothing is scored against itself. Its diagonal is
                    # what "did not move" looks like measured this way, and it is not 0.
                    if ci == 0:
                        Ds = [_crossnobis_cross(pat, _pre_reference(pre_by_sess, exclude=s),
                                                rng, CONF_LABELS)
                              for s, pat in pre_by_sess.items()]
                        if not Ds:
                            ax.axis("off")
                            continue
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            D = np.nanmean(np.stack(Ds), axis=0) / scale
                    else:
                        src = by_day.get(days[ci - 1])
                        if not src:
                            ax.axis("off")
                            continue
                        D = _crossnobis_cross(src, full_ref, rng, CONF_LABELS) / scale
                    # SALIENCE FOLLOWS THE RESULT. `magma_r` at vmax=2.0 made LARGE distances
                    # darkest -- so the eye landed on off-diagonal noise while the finding (a LOW
                    # diagonal = the pattern did not move) faded to pale yellow. It also clipped:
                    # PS93 day 7 holds 3.3-3.6 and every such cell rendered identically black.
                    # `magma` at vmax=2.5 puts the bright end on SMALL distances and leaves the top
                    # of the range distinguishable.
                    im = ax.imshow(np.ma.masked_invalid(D), vmin=0, vmax=2.5, cmap="magma")
                    for i in range(len(CONF_LABELS)):
                        for j in range(len(CONF_LABELS)):
                            if np.isfinite(D[i, j]):
                                _txt(ax, j, i, f"{D[i, j]:.1f}", ha="center", va="center",
                                        fontsize=6,
                                        color="k" if D[i, j] > 1.3 else "w")
                    ax.set_xticks(range(len(CONF_LABELS)))
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                       rotation=90, fontsize=9)
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
                    head = "PRE" if ci == 0 else f"day {days[ci - 1]}"
                    ax.set_title(f"{head}\ndiag {np.nanmean(np.diag(D)):.2f}", fontsize=10,
                                 fontweight="bold" if ci == 0 else "normal")
                    if ci == 0:
                        ax.set_ylabel(f"{an}\nthis position", fontsize=11, fontweight="bold")
            if im is None:
                plt.close(fig)
                continue
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02,
                         label="crossnobis distance -- BRIGHT = unchanged "
                               "(1.0 = mean pre-stroke between-position distance)")
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            _suptitle(fig, 
                f"Cross-validated (crossnobis) distance to the pre-stroke pattern — {wname} window\n"
                f"Post-stroke class: {cls}.  Rows = the post-stroke position, columns = the "
                f"PRE-STROKE reference position. Figure 6's layout, as DISTANCE.\n"
                f"DIAGONAL = did this position's pattern move. LOW is unchanged. Unlike figure 6 "
                f"this is NOISE-UNBIASED: a noisier session does not read as a larger distance.\n"
                f"First column is the no-lesion expectation. Units: mean pre-stroke between-position "
                "distance for that animal.\n"
                "UNBIASED IS NOT GAIN-INVARIANT, and the difference bites: a uniform amplitude "
                "change moves every cell here while leaving 8b untouched. Where this figure and 8b "
                "disagree, 8b is the one to believe about GEOMETRY.\n"
                f"Concretely (post-cue, working): PS92 and PS93 read their WORST diagonal on day 7 "
                f"(1.01, 1.31) while 8b puts day 7 among their BEST (0.66, 0.80) -- that spike is "
                f"amplitude, not a code that moved further away.", fontsize=8.8)
            _footer(fig)
            p = _out(out_dir, f"grant_8_crossnobis_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made


def fig_crossnobis_geometry(out_dir, min_trials=10):
    """8b: RSA proper -- each session's own 6x6 crossnobis RDM against the pre-stroke RDM.

    THIS IS THE ONE THAT ANSWERS THE GAIN QUESTION. Correlating two RDMs is invariant to scaling
    every distance by a constant, so a uniform post-stroke amplitude change cannot move it. Figure
    6's headline -- every position drops, far_R most -- is precisely the signature a global change
    would leave, and this is the measure that cannot be fooled by one.

    PER-POSITION INFORMATION SURVIVES, in a weaker form. A whole-RDM correlation is one number per
    session. Each position's ROW of the RDM -- its five distances to the other positions -- gives a
    per-position number, so the question "is far_R still arranged relative to everything else the
    way it was" is answerable. The question "did far_R's pattern move" is NOT, because second-order
    RSA has thrown away the patterns themselves. That is the trade against figure 8, which keeps the
    positions and gives up gain-invariance.

    Rows = position, columns = day; the strip above each animal is the whole-RDM correlation.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        for v in (("lick",) if align == "lick" else ("lick", "working")):
            store, days = _collect_7(align, v, min_trials)
            if not days:
                continue
            fig, axes = plt.subplots(len(ANIMALS), 2, figsize=(4.6 + 0.62 * len(days), 10.4),
                                     squeeze=False,
                                     gridspec_kw={"width_ratios": [len(days) + 1, 3.4]})
            drew = False
            for ri, an in enumerate(ANIMALS):
                got = store.get(an)
                if not got:
                    for ci in range(2):
                        axes[ri][ci].axis("off")
                    continue
                pre_by_sess, by_day = got
                rng = np.random.default_rng(abs(hash((an, align, v, "8b"))) % (2 ** 31))
                full_ref = _pre_reference(pre_by_sess)
                Dpre = _crossnobis_within(full_ref, rng, CONF_LABELS)
                rows = np.full((len(CONF_LABELS), 1 + len(days)), np.nan)
                whole = np.full(1 + len(days), np.nan)
                def _score_rdm(D, Dref):
                    """(whole-RDM r, per-position row r) for one session against one reference."""
                    a, b = _triu_vals(D), _triu_vals(Dref)
                    ok = np.isfinite(a) & np.isfinite(b)
                    w = (float(np.corrcoef(a[ok], b[ok])[0, 1])
                         if ok.sum() >= 4 and np.std(a[ok]) and np.std(b[ok]) else np.nan)
                    rr = np.full(len(CONF_LABELS), np.nan)
                    for i in range(len(CONF_LABELS)):
                        ra, rb = np.delete(D[i], i), np.delete(Dref[i], i)
                        m = np.isfinite(ra) & np.isfinite(rb)
                        # A ROW IS FIVE NUMBERS. Below four usable ones a correlation is not an
                        # estimate of anything, so the cell stays blank rather than printing an
                        # r built from three points.
                        if m.sum() >= 4 and np.std(ra[m]) and np.std(rb[m]):
                            rr[i] = float(np.corrcoef(ra[m], rb[m])[0, 1])
                    return w, rr

                for ci in range(1 + len(days)):
                    # COLUMN 0 IS GENUINELY LEAVE-ONE-SESSION-OUT (corrected 2026-08-25). It used to
                    # correlate the MEAN of the per-session RDMs against the RDM of the POOLED set --
                    # which CONTAINS every one of those sessions. That is circular, and it showed:
                    # the ceiling read 0.90-1.00 while the post columns it was meant to calibrate ran
                    # 0.52-0.86. Each pre-stroke session is now scored against an RDM built from the
                    # OTHER sessions only, and the resulting correlations are averaged -- one session
                    # against other days, exactly like every post column.
                    if ci == 0:
                        got = [_score_rdm(_crossnobis_within(pat, rng, CONF_LABELS),
                                          _crossnobis_within(
                                              _pre_reference(pre_by_sess, exclude=s),
                                              rng, CONF_LABELS))
                               for s, pat in pre_by_sess.items()]
                        if not got:
                            continue
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            whole[ci] = float(np.nanmean([g[0] for g in got]))
                            rows[:, ci] = np.nanmean([g[1] for g in got], axis=0)
                        continue
                    src = by_day.get(days[ci - 1])
                    if not src:
                        continue
                    whole[ci], rows[:, ci] = _score_rdm(
                        _crossnobis_within(src, rng, CONF_LABELS), Dpre)
                ax = axes[ri][0]
                ax.imshow(np.ma.masked_invalid(rows), vmin=-1, vmax=1, cmap="RdBu_r",
                          aspect="auto")
                for i in range(len(CONF_LABELS)):
                    for j in range(1 + len(days)):
                        if np.isfinite(rows[i, j]):
                            _txt(ax, j, i, f"{rows[i, j]:.2f}", ha="center", va="center",
                                    fontsize=7.5)
                ax.set_xticks(range(1 + len(days)))
                ax.set_xticklabels((["PRE"] + [f"d{d}" for d in days])
                                   if ri == len(ANIMALS) - 1 else [], fontsize=9.5)
                ax.set_yticks(range(len(CONF_LABELS)))
                ax.set_yticklabels(_short(CONF_LABELS), fontsize=9.5)
                ax.set_ylabel(an, fontsize=11.5, fontweight="bold")
                if ri == 0:
                    ax.set_title("per-position: is this position's ROW of the RDM preserved?",
                                 fontsize=11)
                ax2 = axes[ri][1]
                xs = np.arange(1 + len(days))
                ax2.plot(xs, whole, "o-", color="#2166ac", ms=4.5, lw=1.4)
                ax2.axhline(0, color="k", lw=0.8)
                ax2.set_ylim(-0.3, 1.05)
                ax2.set_xticks(xs)
                ax2.set_xticklabels((["PRE"] + [f"d{d}" for d in days])
                                    if ri == len(ANIMALS) - 1 else [], fontsize=9.5)
                ax2.grid(alpha=0.25, lw=0.5)
                if ri == 0:
                    ax2.set_title("whole-RDM correlation\n(gain-invariant)", fontsize=11)
                drew = True
            if not drew:
                plt.close(fig)
                continue
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            _suptitle(fig, 
                f"Second-order RSA on crossnobis RDMs — {wname} window\n"
                f"Post-stroke class: {cls}.  Each session's OWN 6x6 crossnobis RDM correlated "
                f"against the pre-stroke RDM.\n"
                f"INVARIANT TO A GLOBAL AMPLITUDE CHANGE, which figures 6 and 8 are not: scaling "
                f"every distance leaves a correlation between RDMs unchanged.\n"
                f"PRE column = each pre-stroke session against an RDM built from the OTHERS only "
                f"(leave-one-session-out), which is the ceiling. Per-position numbers are that "
                "position's five distances to the others --\n'is it still arranged the same way', "
                f"NOT 'did its pattern move'. Read the post columns against PRE, never against 1.",
                fontsize=9.5)
            fig.tight_layout(rect=(0, 0, 1, 1.0))   # top reserved by _suptitle
            _footer(fig)
            p = Path(out_dir) / f"grant_8b_crossnobis_geometry_{align}_{v}.png"
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made


# ---------------------------------------------------------------------------------------------
# DELTA VIEWS: every post-stroke panel as a DIFFERENCE from the pre-stroke reference
# ---------------------------------------------------------------------------------------------
#
# Priya, 2026-08-25: "make additional versions of fig 5, 6, 7 as differences from prestroke --
# show pre-stroke for reference, but then subsequent columns expressed as deltas."
#
# WHY THIS IS WORTH A SEPARATE FIGURE RATHER THAN A READER'S SUBTRACTION. The absolute panels ask
# the eye to hold a six-by-six reference in memory and compare it with a panel three columns away,
# and the pre-stroke reference is NOT uniform -- close positions are intrinsically more confusable
# than far ones, and every animal's baseline has its own texture. A cell that reads "0.4, low" may
# be 0.4 against a baseline of 0.45 (nothing happened) or against 0.9 (half the code gone). The
# delta answers that directly and the absolute panel cannot.
#
# WHAT IS SUBTRACTED, in every case, is the SAME pre-stroke reference the absolute figure draws in
# its first column -- leave-one-session-out, so it is one session against other days exactly like
# the post columns, and NOT a random half of the pooled trials (see `_collect_7`). Getting this
# wrong would put a floor under every delta.
#
# NOT r-SQUARED. Priya's phrasing was "deltas of r2 / accuracy"; for the correlation figures the
# quantity differenced is r ITSELF, because the sign carries the result -- a far_R pattern moving
# ONTO far_L shows as a positive off-diagonal, and squaring would erase exactly that. Accuracy
# figures difference the row-normalised probability, which is already on [0, 1].


# ---------------------------------------------------------------------------------------------
# BLOCK BOOTSTRAP for the delta figures (Priya, 2026-08-25)
# ---------------------------------------------------------------------------------------------
#
# WHAT IS RESAMPLED, AND WHAT IS NOT.
#
#   BLOCKS, not trials. Trials adjacent in time share arousal, satiety and drift, so an i.i.d.
#   trial bootstrap treats correlated samples as independent and returns intervals that are too
#   narrow. The scheduler's own ~6-trial position blocks are the natural unit and the pipeline
#   already uses them for GroupKFold. A block belongs to ONE position by construction (a new block
#   starts when the position changes), so resampling a session's blocks also resamples each
#   position's trial count -- which is right: how many trials a position got is itself uncertain.
#
#   SESSIONS ARE HELD FIXED. Days are not exchangeable while an animal is recovering: PS94's
#   figure-8 diagonal runs 0.99 -> 0.46 across one week, so there is no single post-stroke value for
#   an interval to be about. Resampling days would fold that trajectory into "sampling noise" and
#   quote an interval for a state that does not exist. The interval therefore means "how well
#   determined GIVEN THESE DAYS" and licenses no generalisation to other days -- the trajectory in
#   the per-session panels is what speaks to that. Same decision, same reasoning, as the pattern
#   bootstrap of 2026-08-25 (DECISIONS.md).
#
# WHY BOOTSTRAP AND NOT A PERMUTATION, for the asymmetry question specifically: permuting position
# labels equalises the condition means, so the true distances collapse toward zero -- but the
# sampling variance of a crossnobis distance scales with the true difference vector, so a real and
# perfectly SYMMETRIC separation still yields a larger |D - D.T| than permuted data does. The
# permuted null therefore sits too low and would call ordinary noise "asymmetry". A bootstrap
# interval on D[P,Q] - D[Q,P] has no such problem.

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


def _delta_diag_ci(mats_fn, x_store, blk_store, an, days, rng, n_boot=N_BOOT_DELTA):
    """95% interval on (day diagonal - PRE diagonal), block-bootstrapped.

    ``mats_fn(pat_by_pos, ref_by_pos) -> 6x6`` builds one matrix from resampled trial sets. The PRE
    reference is resampled in the SAME draw as the day, so the two are correlated exactly as they
    are in the data and the difference is taken draw by draw -- differencing two independently
    published intervals would overstate the spread.
    """
    pre_x, day_x = x_store
    pre_b, day_b = blk_store
    out = {}
    for d in days:
        if d not in day_x:
            continue
        deltas = []
        for _ in range(n_boot):
            # Resample every pre-stroke session ONCE per draw, then reuse those same resampled
            # sessions for both the baseline and the day's reference, so the two share their noise
            # and the difference below is taken draw by draw.
            pre_r = {s: _block_boot(pre_x[s], pre_b[s], rng) for s in pre_x}
            pre_r = {s: v for s, v in pre_r.items() if v}
            day_r = _block_boot(day_x[d], day_b[d], rng)
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
            continue
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
        if rec.get("mean"):
            out[d] = rec
    return out


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


@lru_cache(maxsize=6)
def _matrices_splithalf(align, variant, min_trials=10):
    """{animal: {"PRE": M, day: M, ...}} of WITHIN-session split-half matrices (figures 7 / 7d)."""
    store, days = _collect_7(align, variant, min_trials)
    out = {}
    for an, (pre_by_sess, by_day) in store.items():
        rng = np.random.default_rng(abs(hash((an, align, variant))) % (2 ** 31))
        d = {}
        base = _nanmean_stack([_split_half_matrix(pat, rng) for pat in pre_by_sess.values()])
        if base is not None:
            d["PRE"] = base
        for day, pat in by_day.items():
            d[day] = _split_half_matrix(pat, rng)
        if d:
            out[an] = d
    return out, days


def _delta_grid(mats, days, out_dir, fname, *, title, abs_label, delta_label,
                vmin, vmax, cmap, dmax, summary, ylab, figh=9.2, cis=None):
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
    fig, grid = plt.subplots(len(ANIMALS), ncol + 1, figsize=(_colw() * ncol + 2.0, figh),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [1, 0.62] + [1] * len(days)})
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
            if ci == 0:
                lab = f"{head}  {stat:.2f}"
            elif band:
                lab = f"{head}  {stat:+.2f}\n[{band[0]:+.2f}, {band[1]:+.2f}]"
            else:
                lab = f"{head}  {stat:+.2f}"
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
        rng = np.random.default_rng(abs(hash((an, align, variant, tag))) % (2 ** 31))
        try:
            rich = _delta_diag_ci(mats_for(an, rng), (pre_x, day_x), (pre_b, day_b), an, days,
                                  rng, n_boot=n_boot)
            # `_delta_grid` prints only the mean's (lo, hi); figure 9 plots the whole record.
            out[an] = rich if full else {d: r["mean"][:2] for d, r in rich.items()}
        except Exception as ex:                                          # noqa: BLE001
            print(f"  !! {tag} CI {an} {align}/{variant}: {type(ex).__name__} {str(ex)[:80]}",
                  flush=True)
    return out


def _mats_pattern(_an, _rng):
    return lambda pat, ref: _corr_matrix(_means(pat), _means(ref))


def _mats_splithalf(_an, rng):
    # WITHIN one set, so the reference argument is unused by design -- figure 7's whole point is
    # that no pre-stroke reference enters a single panel.
    return lambda pat, _ref: _split_half_matrix(pat, rng)


def _mats_crossnobis(_an, rng, sign=-1):
    """Crossnobis with the whitener fixed at the first call and reused for every later resample.

    ``sign=-1`` (the default) negates the distances so "larger diagonal = more preserved" holds as
    it does for the correlation figures -- see the note at the return. ``sign=+1`` gives the raw
    distances, which is what the ASYMMETRY needs: D[P,Q] - D[Q,P] must be in distance units to be
    read as "post P is further from pre Q than post Q is from pre P".
    """
    held = {}

    def build(pat, ref):
        pm0, pm1, pres = _halves(pat, rng, CONF_LABELS)
        rm0, rm1, rres = _halves(ref, rng, CONF_LABELS)
        if "P" not in held:
            held["P"] = _whitener(pres + rres)
            # SAME UNITS AS THE FIGURE. `_matrices_crossnobis` divides every distance by the mean
            # pre-stroke between-position distance, so 1.0 reads as "as far apart as two different
            # pre-stroke positions". The interval was computed in RAW crossnobis units and printed
            # beside a normalised point estimate -- figure 8d showed "day 1 +0.28 [+6.63, +69.41]",
            # an interval not containing its own estimate, which is how the mismatch announced
            # itself. Fixed from the first reference seen, so both numbers share one scale.
            s = np.nanmean(_triu_vals(_crossnobis_within(ref, rng, CONF_LABELS)))
            held["scale"] = float(s) if np.isfinite(s) and s > 0 else 1.0
        P = held["P"]
        D = np.full((len(CONF_LABELS), len(CONF_LABELS)), np.nan)
        if P is None:
            return D
        for i, a in enumerate(CONF_LABELS):
            for j, b in enumerate(CONF_LABELS):
                if a in pm0 and b in rm0:
                    D[i, j] = float((pm0[a] - rm0[b]) @ P @ (pm1[a] - rm1[b])) / held["scale"]
        # NEGATED BY DEFAULT so "larger diagonal = more preserved" holds here as it does for the
        # correlation figures. `_delta_diag_ci` differences mean diagonals, and for a DISTANCE a
        # SMALLER diagonal means less change -- without the flip the interval would carry the
        # opposite sign to the number printed beside it, which is worse than no interval at all.
        return sign * D

    return build


def fig_pattern_delta(out_dir, min_trials=10):
    """6d: figure 6b as DIFFERENCES from the pre-stroke reference.

    Every post-stroke panel minus the leave-one-session-out pre-stroke matrix. Zero means "this day
    looks exactly like one pre-stroke day looks against the others" -- which is the honest null, not
    a correlation of 1.

    READ THE OFF-DIAGONAL AS WELL AS THE DIAGONAL. A negative diagonal cell says the position lost
    its own code; a POSITIVE off-diagonal cell at (far_R, far_L) says far_R trials came to look more
    like pre-stroke far_L than they used to -- a substitution, which the absolute panel shows only if
    the reader remembers what that cell looked like before. The delta is the whole reason the
    substitution result is legible at a glance.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        for v in (("lick",) if align == "lick" else ("lick", "working")):
            mats, days = _matrices_pattern(align, v, min_trials)
            if not days or not mats:
                continue
            cis = _delta_cis(align, v, min_trials, _mats_pattern, "6d")
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            p = _delta_grid(
                mats, days, out_dir, f"grant_6d_pattern_delta_{align}_{v}.png",
                title=(f"Mean-pattern similarity, CHANGE FROM PRE-STROKE — {wname} window\n"
                       f"Post-stroke class: {cls}.  Column 1 is the pre-stroke reference "
                       f"(leave-one-session-out); every later column is THAT DAY MINUS IT.\n"
                       f"Rows = the pattern being described, columns within a panel = the "
                       f"pre-stroke position it is correlated with. ZERO = indistinguishable from "
                       f"an ordinary pre-stroke day.\nNegative on the DIAGONAL = the position lost "
                       f"its own code. Positive OFF-DIAGONAL = it came to look like a different "
                       "position.\n"
                       "Above each panel: the change in mean diagonal and its 95% BLOCK-BOOTSTRAP "
                       "interval -- the scheduler's ~6-trial position blocks resampled within each "
                       "session,\nsessions NOT resampled (days are not exchangeable while an animal "
                       "recovers), and the baseline resampled in the SAME draw so the difference is "
                       "taken draw by draw."),
                abs_label="pre-stroke r", delta_label="change in r vs pre-stroke",
                vmin=-1, vmax=1, cmap="RdBu_r", dmax=1.0, summary=_diag,
                ylab="this position", cis=cis)
            if p:
                made.append(p)
    return made


def fig_splithalf_delta(out_dir, min_trials=10):
    """7d: figure 7 as DIFFERENCES from the pre-stroke reference.

    The within-session split-half matrix minus the per-pre-session average. Zero means this session
    reproduces its own patterns exactly as repeatably as a pre-stroke session did.

    THIS IS THE CONTROL FIGURE IN ITS MOST DIRECT FORM. If the pattern deltas in 6d were really a
    reliability story, the diagonal here would fall by a comparable amount at the same positions on
    the same days. Where 6d falls and this does not, the code moved.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        for v in (("lick",) if align == "lick" else ("lick", "working")):
            mats, days = _matrices_splithalf(align, v, min_trials)
            if not days or not mats:
                continue
            cis = _delta_cis(align, v, min_trials, _mats_splithalf, "7d")
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            p = _delta_grid(
                mats, days, out_dir, f"grant_7d_splithalf_delta_{align}_{v}.png",
                title=(f"WITHIN-session split-half similarity, CHANGE FROM PRE-STROKE — {wname} "
                       f"window\nPost-stroke class: {cls}.  Column 1 is the average over pre-stroke "
                       f"sessions; every later column is THAT DAY MINUS IT.\n"
                       f"BOTH HALVES COME FROM THE SAME SESSION, so nothing about the lesion or the "
                       f"pre-stroke reference enters a single panel -- only how repeatable that "
                       f"day's own patterns are.\nA diagonal that falls HERE as much as it falls in "
                       f"6d means a noisier code; a 6d fall without one here means a MOVED code. "
                       f"The number above each panel is the change in mean diagonal."),
                abs_label="pre-stroke split-half r", delta_label="change in split-half r",
                vmin=-1, vmax=1, cmap="RdBu_r", dmax=1.0, summary=_diag,
                ylab="half A at", cis=cis)
            if p:
                made.append(p)
    return made


@lru_cache(maxsize=12)
def _matrices_crossnobis(align, variant, min_trials=10, row_centre=False):
    """{animal: {"PRE": D, day: D}} of cross-set crossnobis distances, in pre-stroke units.

    ``row_centre`` subtracts each ROW's own mean, which is the difference between asking "did this
    position move" and "which position did it move TOWARD".

    WHY THAT MATTERS (Priya, 2026-08-26, on whole rows shifting together). Writing the
    cross-validated distance out, in the whitened metric:

        d(post P, pre Q) = |mu_postP|^2 - 2 mu_postP . mu_preQ + |mu_preQ|^2

    the first term depends ONLY ON P. So a change in the overall magnitude of position P's
    post-stroke response moves its distance to EVERY pre-stroke position by the same amount, and the
    panel shows a uniform orange or purple row. That is amplitude, not "this position came to
    resemble all six". Row-centring removes the term that carries it and leaves the CONTRAST within
    the row, which is where a substitution lives. Same gain sensitivity that makes 8b the arbiter
    for anything about geometry.
    """
    store, days = _collect_7(align, variant, min_trials)
    out = {}
    for an, (pre_by_sess, by_day) in store.items():
        rng = np.random.default_rng(abs(hash((an, align, variant, "8"))) % (2 ** 31))
        full_ref = _pre_reference(pre_by_sess)
        scale = np.nanmean(_triu_vals(_crossnobis_within(full_ref, rng, CONF_LABELS)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        d = {}
        base = _nanmean_stack([_crossnobis_cross(pat, _pre_reference(pre_by_sess, exclude=s),
                                                 rng, CONF_LABELS)
                               for s, pat in pre_by_sess.items()])
        if base is not None:
            d["PRE"] = base / scale
        for day, pat in by_day.items():
            d[day] = _crossnobis_cross(pat, full_ref, rng, CONF_LABELS) / scale
        if row_centre:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)      # a row can be entirely NaN
                d = {k: M - np.nanmean(M, axis=1, keepdims=True) for k, M in d.items()}
        if d:
            out[an] = d
    return out, days


def fig_crossnobis_delta(out_dir, min_trials=10):
    """8d: figure 8 as DIFFERENCES from the pre-stroke reference.

    The reference distances are NOT uniform -- close positions sit nearer each other than far ones
    do, and each animal's baseline has its own texture -- so an absolute cell of 1.2 means different
    things in different places. Subtracting leaves only what the lesion did.

    POSITIVE = further from the pre-stroke pattern than a held-out pre-stroke session is; NEGATIVE =
    closer. The diagonal is the headline and the off-diagonal carries the substitution: a post-stroke
    far_R row going NEGATIVE under the close_L column means far_R trials moved TOWARD pre-stroke
    close_L.

    Read beside 8b before concluding anything about geometry: these are distances, so a uniform
    amplitude change shifts the whole panel while leaving 8b untouched.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        for v in (("lick",) if align == "lick" else ("lick", "working")):
            mats, days = _matrices_crossnobis(align, v, min_trials)
            if not days or not mats:
                continue
            # SIGN: _mats_crossnobis returns NEGATED distances so bigger = more preserved, matching
            # the correlation figures; flip the interval back into distance units for display.
            cis = {a2: {d: (-hi, -lo) for d, (lo, hi) in v2.items()}
                   for a2, v2 in _delta_cis(align, v, min_trials, _mats_crossnobis, "8d").items()}
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            p = _delta_grid(
                mats, days, out_dir, f"grant_8d_crossnobis_delta_{align}_{v}.png",
                title=(f"Crossnobis distance to the pre-stroke pattern, CHANGE FROM PRE-STROKE — "
                       f"{wname} window\nPost-stroke class: {cls}.  Column 1 is the pre-stroke "
                       f"reference (leave-one-session-out); every later column is THAT DAY MINUS "
                       f"IT.\nPOSITIVE = further from the pre-stroke pattern than a held-out "
                       f"pre-stroke session is. NEGATIVE = closer. ZERO = an ordinary pre-stroke "
                       f"day.\nDiagonal = did the pattern move. A NEGATIVE off-diagonal cell is a "
                       f"substitution: that row's trials moved TOWARD the column's pre-stroke "
                       f"pattern. Distances are NOT gain-invariant -- read with 8b."),
                abs_label="pre-stroke distance",
                delta_label="change in distance (1.0 = mean pre-stroke between-position)",
                vmin=0, vmax=2.5, cmap="magma", dmax=1.5, summary=_diag,
                ylab="this position", cis=cis)
            if p:
                made.append(p)
    return made


def fig_confusion_delta(out_dir):
    """5d: figure 5c as DIFFERENCES from the pre-stroke reference.

    The frozen decoder's per-session confusion minus the leave-one-session-out pre-stroke confusion,
    cell by cell, in row-normalised probability. Zero means the decoder makes the same errors in the
    same proportions it made before the lesion.

    WHY THIS IS THE MOST INFORMATIVE OF THE THREE. The pre-stroke confusion is far from uniform --
    close positions are confusable with each other and far ones are not -- so an absolute
    post-stroke cell of 0.3 means different things in different places. Subtracting removes the
    baseline structure and leaves only what the lesion did: a negative diagonal cell is recall lost
    at that position, and the positive cell in the same ROW says where those trials went instead.
    """
    made = []
    for _disp, align, wname in (("ENL", "precue", "ENL (pre-cue)"), ("cue", "cue", "post-cue")):
        per_animal, days = _collect_5c(align)
        if not days or not per_animal:
            continue

        def _norm(C):
            row = C.sum(1, keepdims=True)
            return np.divide(C, row, out=np.full_like(C, np.nan), where=row > 0)

        mats = {}
        for an, (Cpre, by_day) in per_animal.items():
            if Cpre is None or not Cpre.sum():
                continue
            d = {"PRE": _norm(Cpre)}
            for day, C in by_day.items():
                if C is not None and C.sum():
                    d[day] = _norm(C)
            mats[an] = d
        p = _delta_grid(
            mats, days, out_dir, f"grant_5d_confusion_delta_{align}.png",
            title=(f"Frozen pre-stroke decoder, CHANGE FROM PRE-STROKE — {wname} window\n"
                   f"Post-stroke trials are LICK + MISS-WHILE-WORKING (terminal quit period "
                   f"removed). Column 1 is the pre-stroke confusion (leave-one-session-out);\n"
                   f"every later column is THAT DAY MINUS IT. Rows = TRUE spout position, columns "
                   f"within a panel = predicted. ZERO = the same errors in the same proportions.\n"
                   f"A negative DIAGONAL cell is recall lost at that position; the positive cell in "
                   f"the SAME ROW says where those trials went instead. The number above each panel "
                   f"is the change in overall accuracy."),
            abs_label="pre-stroke P(pred | true)",
            delta_label="change in P(predicted | true)",
            vmin=0, vmax=1, cmap="magma", dmax=0.6, summary=_diag,
            ylab="true position", figh=9.0)
        if p:
            made.append(p)
    return made


def fig_delta_trajectory(out_dir, min_trials=10):
    """9: THE BOOTSTRAP RESULTS AS A FIGURE -- change from pre-stroke over days, with intervals.

    Priya, 2026-08-26: "is there a figure that shows the bootstrap results?" There was not. The
    intervals existed only as text above the 6x6 matrices of 6d/7d/8d, which is the wrong shape for
    the question they answer: whether a position is recovering, holding or getting worse is a
    TRAJECTORY, and a trajectory read out of twenty-eight small matrices by eye is not read at all.

    LEFT PANEL, per animal: the mean own-position change with its 95% block-bootstrap interval, one
    point per day. This is the number printed in 6d's panel titles, plotted.
    RIGHT PANEL: the same split BY POSITION, which is where the deficit lives -- far_R is the
    position the lesion takes and the others are the control it has to be read against.

    ZERO IS THE NULL AND IT IS A REAL ONE: it means "this day differs from the pre-stroke reference
    no more than one pre-stroke day differs from the others", because the baseline is
    leave-one-session-out rather than a correlation of 1. A point whose interval excludes zero has
    changed by more than ordinary day-to-day drift.

    SESSIONS ARE NOT RESAMPLED, so an interval says how well determined a day is GIVEN THESE DAYS
    and licenses no claim about days not recorded. The trajectory is what speaks to that, which is
    the whole reason this figure is per-day rather than pooled.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        for v in (("lick",) if align == "lick" else ("lick", "working")):
            _, days = _collect_7(align, v, min_trials)
            if not days:
                continue
            cis = _delta_cis(align, v, min_trials, _mats_pattern, "9", full=True)
            if not any(cis.values()):
                continue
            fig, axes = plt.subplots(len(ANIMALS), 2, figsize=(10.0, 2.0 * len(ANIMALS) + 1.3),
                                     squeeze=False, sharex=True)
            drew = False
            for ri, an in enumerate(ANIMALS):
                rec = cis.get(an) or {}
                ax = axes[ri][0]
                xs = [d for d in days if d in rec]
                if xs:
                    med = [rec[d]["mean"][2] for d in xs]
                    lo = [rec[d]["mean"][2] - rec[d]["mean"][0] for d in xs]
                    hi = [rec[d]["mean"][1] - rec[d]["mean"][2] for d in xs]
                    ax.errorbar(xs, med, yerr=[lo, hi], fmt="o-", color="#b2182b", ms=5,
                                capsize=3, lw=1.5)
                    drew = True
                ax.axhline(0, color="k", lw=1.2)
                ax.set_ylabel(f"{an}\nchange in r", fontsize=11.5, fontweight="bold")
                ax.grid(alpha=0.25, lw=0.5)
                if ri == 0:
                    ax.set_title("mean own-position change (95% block bootstrap)", fontsize=9.5)

                ax2 = axes[ri][1]
                for q in CONF_LABELS:
                    col, mk, _ls = POS_STYLE[q]
                    qx = [d for d in days if d in rec and q in rec[d]["pos"]]
                    if not qx:
                        continue
                    qm = [rec[d]["pos"][q][2] for d in qx]
                    ql = [rec[d]["pos"][q][2] - rec[d]["pos"][q][0] for d in qx]
                    qh = [rec[d]["pos"][q][1] - rec[d]["pos"][q][2] for d in qx]
                    ax2.errorbar(qx, qm, yerr=[ql, qh], fmt=mk + "-", color=col, ms=4,
                                 capsize=2, lw=1.1, alpha=0.9,
                                 label=q if ri == 0 else None)
                ax2.axhline(0, color="k", lw=1.2)
                ax2.grid(alpha=0.25, lw=0.5)
                if ri == 0:
                    ax2.set_title("the same, per position", fontsize=9.5)
                if ri == len(ANIMALS) - 1:
                    ax.set_xlabel("days from lesion")
                    ax2.set_xlabel("days from lesion")
            if not drew:
                plt.close(fig)
                continue
            h, lab = axes[0][1].get_legend_handles_labels()
            if h:
                # ABOVE THE FOOTER, not on it. `_footer` writes at y=0.004, and a legend at
                # "lower center" lands in the same place -- the two overprinted each other in the
                # first render of this figure. This is the only figure in the module carrying both.
                fig.legend(h, lab, loc="lower center", ncol=len(POS), fontsize=11, frameon=False,
                           bbox_to_anchor=(0.5, 0.035))
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            _suptitle(fig, 
                f"Change from pre-stroke over days, with block-bootstrap intervals — {wname} "
                f"window\n"
                f"Post-stroke class: {cls}.  ZERO = this day differs from the pre-stroke reference "
                f"no more than one pre-stroke day differs from the others\n"
                f"(the baseline is leave-one-session-out, NOT a correlation of 1). An interval "
                f"excluding zero is a change beyond ordinary day-to-day drift.\n"
                f"Blocks resampled within session; sessions NOT resampled, so an interval is "
                f"conditional on these days and the trajectory is what speaks to the rest.",
                fontsize=9.5)
            # Bottom band holds BOTH the legend (y=0.035) and the footer (y=0.004), so it needs
            # more than the usual 0.05.
            fig.tight_layout(rect=(0, 0.10, 1, 1.0))   # top reserved by _suptitle
            _footer(fig)
            p = Path(out_dir) / f"grant_9_delta_trajectory_{align}_{v}.png"
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made


def _asymmetry_ci(align, variant, min_trials, n_boot=N_BOOT_DELTA):
    """{animal: {"PRE"|day: (observed 6x6 asymmetry, bool 6x6 "interval excludes zero")}}.

    THE QUANTITY. A[P,Q] = d(post at P, pre at Q) - d(post at Q, pre at P). Rows and columns index
    genuinely different sets, so there is no reason for D to be symmetric and the gap between the
    two orderings is the substitution signal: a large positive A[far_R, close_L] says post-stroke
    far_R sits further from pre-stroke close_L than post-stroke close_L sits from pre-stroke far_R.

    WHY THIS REPLACES THE EARLIER COMPARISON. The first pass compared each post column's mean
    asymmetry against a SINGLE held-out pre-stroke session (n = 1, no spread), after establishing
    that the leave-one-out AVERAGE was not a fair baseline -- averaging six matrices shrinks the
    noise whose absolute value is being measured, so it sits systematically low. One draw with no
    spread is not a baseline either: "PS93 post 0.31-0.47 vs pre 0.523" could be an unlucky 0606.
    A bootstrap interval per PAIR needs no pre-stroke baseline at all -- if the interval on
    A[P,Q] excludes zero, that pair is asymmetric, full stop -- and trial counts are matched by
    construction, which disposes of the other objection to the earlier comparison.

    NOT A PERMUTATION. Shuffling position labels equalises the condition means, so the true
    distances collapse toward zero -- but the sampling variance of a crossnobis distance scales with
    the true difference vector, so a real and perfectly SYMMETRIC separation still yields a larger
    |A| than permuted data does. The permuted null sits too low and would call noise asymmetry.

    The PRE column is leave-one-session-out and is computed ONCE, not per day: it does not depend on
    which post-stroke day is being scored.
    """
    x_store, days = _collect_7(align, variant, min_trials)
    b_store, _ = _collect_7(align, variant, min_trials, "blk")
    out = {}
    for an in ANIMALS:
        if an not in x_store or an not in b_store:
            continue
        (pre_x, day_x), (pre_b, day_b) = x_store[an], b_store[an]
        rng = np.random.default_rng(abs(hash((an, align, variant, "asym"))) % (2 ** 31))
        build = _mats_crossnobis(an, rng, sign=+1)

        def _pool(pre_r, exclude=None):
            acc = {}
            for s, Z in pre_r.items():
                if s == exclude:
                    continue
                for q, z in Z.items():
                    acc.setdefault(q, []).append(z)
            return {q: np.vstack(v) for q, v in acc.items()}

        def _summarise(draws):
            if len(draws) < n_boot // 4:
                return None
            S = np.stack(draws)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                obs = np.nanmedian(S, axis=0)
                lo = np.nanpercentile(S, 2.5, axis=0)
                hi = np.nanpercentile(S, 97.5, axis=0)
            sig = np.isfinite(lo) & np.isfinite(hi) & ((lo > 0) | (hi < 0))
            np.fill_diagonal(sig, False)                 # A[P,P] is 0 by construction
            return obs, sig

        rec, pre_draws, day_draws = {}, [], {d: [] for d in days}
        for _ in range(n_boot):
            pre_r = {s: _block_boot(pre_x[s], pre_b[s], rng) for s in pre_x}
            pre_r = {s: v for s, v in pre_r.items() if v}
            if not pre_r:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                As = []
                for s, held in pre_r.items():
                    rest = _pool(pre_r, exclude=s)
                    if held and rest:
                        D = build(held, rest)
                        As.append(D - D.T)
                if As:
                    pre_draws.append(np.nanmean(np.stack(As), axis=0))
                full = _pool(pre_r)
                for d in days:
                    if d not in day_x:
                        continue
                    day_r = _block_boot(day_x[d], day_b[d], rng)
                    if not day_r or not full:
                        continue
                    D = build(day_r, full)
                    day_draws[d].append(D - D.T)
        got = _summarise(pre_draws)
        if got:
            rec["PRE"] = got
        for d in days:
            got = _summarise(day_draws[d])
            if got:
                rec[d] = got
        if rec:
            out[an] = rec
    return out


def fig_asymmetry(out_dir, min_trials=10):
    """8e: is the post-stroke distance matrix ASYMMETRIC, and where -- with intervals.

    Priya, 2026-08-25: "the first column is prestroke LOSO right? so we don't necessarily expect it
    to be symmetric across the diagonal?" Right, and the two column kinds differ. Column 1 scores one
    pre-stroke session against OTHER pre-stroke sessions, so both sides estimate the same underlying
    patterns and A[P,Q] has expectation zero -- whatever is there is noise. Every later column scores
    a genuinely different distribution against the pre-stroke reference, and there the asymmetry is
    the SUBSTITUTION: which way a position moved, not merely that it moved.

    A GREEN RING marks a pair whose 95% block-bootstrap interval excludes zero. Read the PRE column
    first: rings there are the false-positive rate this construction actually achieves, and they
    should be few.

    THIS FIGURE MUST NOT BE SYMMETRISED, unlike figure 7. There the two cells estimated ONE quantity
    and differed only by which random half went where, so averaging them was strictly better. Here
    they estimate different quantities and the difference is the result.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        for v in (("lick",) if align == "lick" else ("lick", "working")):
            _, days = _collect_7(align, v, min_trials)
            if not days:
                continue
            cis = _asymmetry_ci(align, v, min_trials)
            if not cis:
                continue
            cols = ["PRE"] + list(days)
            fig, axes = plt.subplots(len(ANIMALS), len(cols),
                                     figsize=(_colw() * len(cols) + 1.4, 7.4), squeeze=False)
            im = None
            for ri, an in enumerate(ANIMALS):
                rec = cis.get(an) or {}
                for ci, key in enumerate(cols):
                    ax = axes[ri][ci]
                    got = rec.get(key)
                    if got is None:
                        ax.axis("off")
                        continue
                    A, sig = got
                    lim = float(np.nanpercentile(np.abs(A), 95)) or 1.0
                    im = ax.imshow(np.ma.masked_invalid(A), vmin=-lim, vmax=lim, cmap="PuOr_r")
                    for i in range(len(CONF_LABELS)):
                        for j in range(len(CONF_LABELS)):
                            if sig[i, j]:
                                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                                           edgecolor="lime", lw=1.4))
                    ax.set_xticks(range(len(CONF_LABELS)))
                    ax.set_yticks(range(len(CONF_LABELS)))
                    ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                       rotation=90, fontsize=9)
                    ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
                    n_sig = int(sig.sum() // 2)          # antisymmetric: each pair rings twice
                    head = "PRE" if key == "PRE" else f"day {key}"
                    ax.set_title(f"{head}\n{n_sig}/15", fontsize=10,
                                 fontweight="bold" if ci == 0 else "normal")
                    if ci == 0:
                        ax.set_ylabel(f"{an}\nthis position", fontsize=11, fontweight="bold")
            if im is None:
                plt.close(fig)
                continue
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02,
                         label="d(post P, pre Q) - d(post Q, pre P)")
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            _suptitle(fig, 
                f"Is the distance matrix ASYMMETRIC, and where? -- {wname} window\n"
                f"Post-stroke class: {cls}.  A[P,Q] = d(post at P, pre at Q) - d(post at Q, pre at "
                f"P). Rows and columns index different sets, so symmetry is not expected.\n"
                f"GREEN RING = 95% block-bootstrap interval excludes zero. The count above each "
                f"panel is rung pairs out of 15.\n"
                f"READ THE PRE COLUMN FIRST: both sides there estimate the same patterns, so its "
                f"asymmetry has expectation zero and its rings are this construction's own "
                f"false-positive rate.", fontsize=9.5)
            _footer(fig)
            p = _out(out_dir, f"grant_8e_asymmetry_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made


@lru_cache(maxsize=6)
def _rdm_rows(align, variant, min_trials=10):
    """{animal: {"PRE"|day: (per-position row r, whole-RDM r, n_positions)}} for figures 8b and 8g.

    Extracted so the per-position TRAJECTORY figure and the heatmap cannot disagree: they are the
    same numbers drawn two ways.
    """
    x_store, days = _collect_7(align, variant, min_trials)
    out = {}
    for an in ANIMALS:
        if an not in x_store:
            continue
        pre_by_sess, by_day = x_store[an]
        rng = np.random.default_rng(abs(hash((an, align, variant, "8g"))) % (2 ** 31))
        full_ref = _pre_reference(pre_by_sess)
        Dpre = _crossnobis_within(full_ref, rng, CONF_LABELS)

        def score(D, Dref):
            a, b = _triu_vals(D), _triu_vals(Dref)
            ok = np.isfinite(a) & np.isfinite(b)
            whole = (float(np.corrcoef(a[ok], b[ok])[0, 1])
                     if ok.sum() >= 4 and np.std(a[ok]) and np.std(b[ok]) else np.nan)
            rows = np.full(len(CONF_LABELS), np.nan)
            for i in range(len(CONF_LABELS)):
                ra, rb = np.delete(D[i], i), np.delete(Dref[i], i)
                m = np.isfinite(ra) & np.isfinite(rb)
                if m.sum() >= 4 and np.std(ra[m]) and np.std(rb[m]):
                    rows[i] = float(np.corrcoef(ra[m], rb[m])[0, 1])
            n_pos = int(np.isfinite(np.diag(Dref)).sum() or 0)
            return rows, whole, n_pos

        rec = {}
        loo = [score(_crossnobis_within(pat, rng, CONF_LABELS),
                     _crossnobis_within(_pre_reference(pre_by_sess, exclude=s), rng, CONF_LABELS))
               for s, pat in pre_by_sess.items()]
        if loo:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rec["PRE"] = (np.nanmean([r for r, _w, _n in loo], axis=0),
                              float(np.nanmean([w for _r, w, _n in loo])), 6)
        for d, pat in by_day.items():
            D = _crossnobis_within(pat, rng, CONF_LABELS)
            r, w, _n = score(D, Dpre)
            # POSITIONS PRESENT IN THIS SESSION, which is what governs whether a row is computable
            rec[d] = (r, w, len(pat))
        if rec:
            out[an] = rec
    return out, days


def fig_geometry_by_position(out_dir, min_trials=10):
    """8g: figure 8b split BY SPOUT POSITION -- one panel per position, animals as lines.

    Priya, 2026-08-26. 8b's left panel is already per position, but as a heatmap of animal-major
    rows: comparing one position ACROSS animals and days means reading four separate blocks. Here
    each position gets a panel and each animal a line, which is the comparison the deficit is about
    -- far_R against the positions that were spared, in every animal at once.

    THE DASHED LINE IS THAT ANIMAL'S PRE CEILING for that position, leave-one-session-out. Read a
    trace against its own dashed line, not against 1: a held-out pre-stroke session does not
    reproduce the others perfectly either.

    A GAP IS NOT A ZERO. A row correlation needs at least four of the five partner positions, so a
    session missing two positions has every row uncomputable -- including the positions the animal
    licked normally. That is why whole days vanish for PS94 in the LICK class, and it is a property
    of the estimator, not of the animal. The `working` class keeps miss-while-working trials and
    fills most of them; the panel titles carry how many sessions actually contributed.
    """
    made = []
    for _disp, align, wname in WINDOWS:
        for v in (("lick",) if align == "lick" else ("lick", "working")):
            rows, days = _rdm_rows(align, v, min_trials)
            if not days or not rows:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), squeeze=False, sharex=True,
                                     sharey=True, gridspec_kw={"hspace": 0.32})
            drew = False
            for k, q in enumerate(CONF_LABELS):
                ax = axes[k // 3][k % 3]
                n_have = 0
                for an in ANIMALS:
                    rec = rows.get(an) or {}
                    xs = [d for d in days if d in rec and np.isfinite(rec[d][0][k])]
                    ys = [rec[d][0][k] for d in xs]
                    col = (config.animals().get(an) or {}).get("color", "0.4")
                    if xs:
                        ax.plot(xs, ys, "o-", color=col, ms=4, lw=1.4,
                                label=an if k == 0 else None)
                        n_have += 1
                        drew = True
                    if "PRE" in rec and np.isfinite(rec["PRE"][0][k]):
                        ax.axhline(rec["PRE"][0][k], color=col, ls=(0, (2, 3)), lw=1.0, alpha=0.8)
                ax.axhline(0, color="k", lw=1.0)
                ax.set_ylim(-1.05, 1.05)
                ax.grid(alpha=0.25, lw=0.5)
                ax.set_title(f"{q}   ({n_have}/4 animals)", fontsize=11, fontweight="bold")
                if k % 3 == 0:
                    ax.set_ylabel("row of the RDM preserved (r)", fontsize=10)
                if k // 3 == 1:
                    ax.set_xlabel("days from lesion", fontsize=11)
            if not drew:
                plt.close(fig)
                continue
            h, lab = axes[0][0].get_legend_handles_labels()
            if h:
                fig.legend(h, lab, loc="lower center", ncol=len(ANIMALS), fontsize=10,
                           frameon=False, bbox_to_anchor=(0.5, 0.035))
            cls = ("LICK trials only" if v == "lick" else
                   "LICK + miss-while-working (quit period removed)")
            fig.tight_layout(rect=(0, 0.10, 1, 1.0))
            _suptitle(fig,
                      f"Is each position's RDM row preserved? -- by POSITION -- {wname} window\n"
                      f"Post-stroke class: {cls}.  Figure 8b split so one position can be compared "
                      f"across animals and days in a single panel.\n"
                      f"DASHED = that animal's own pre-stroke ceiling for that position "
                      f"(leave-one-session-out). Read a trace against its own dashed line, never "
                      f"against 1.\n"
                      f"A GAP IS NOT A ZERO: a row needs 4 of its 5 partner positions, so a session "
                      f"missing two positions has EVERY row uncomputable -- including positions the "
                      f"animal licked normally.")
            _footer(fig)
            p = _out(out_dir, f"grant_8g_geometry_by_position_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made


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


def fig_coding_retained(out_dir, meth="dom_orth"):
    """Two lines per panel, not six.

    THE SIX-LINE VERSION DID NOT WORK and the reason is structural, not cosmetic. It read the
    `poststroke_lick` class, so at an IMPAIRED position -- where the animal barely licks -- there is
    no cell to plot, and the positions the figure exists to describe were the ones missing from it.
    Twelve panels of six overlapping traces also had no legible message.

    So: the impaired positions are shown from MISS-WHILE-WORKING trials (the only trials they have)
    and the preserved positions from LICK trials, averaged within each group, SEM across positions.
    The two lines therefore come from different trial classes ON PURPOSE, which is stated on the
    figure -- an impaired position has no lick trials to average, and pretending otherwise is what
    produced the empty panels.
    """
    src = _fig_root() / "coding_direction.json"
    if not src.exists():
        return None
    data = json.loads(src.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(len(WINDOWS), len(ANIMALS), figsize=(10.4, 6.6),
                             sharey="row", squeeze=False)
    for ri, (disp, _align, wname) in enumerate(WINDOWS):
        for ci, an in enumerate(ANIMALS):
            ax = axes[ri][ci]
            ax.axhline(1.0, color="tab:green", ls=":", lw=1.4)
            ax.axhline(0.0, color="k", lw=0.8)
            res = (data.get(disp) or {}).get(an)
            imp = _impaired(an)
            if res and meth in res.get("methods", {}):
                by_cls = res["methods"][meth].get("cross_by_session", {})
                for group, cls, col, mk, lbl in (
                        (imp, "poststroke_miss_working", "#b2182b", "o",
                         "IMPAIRED positions (miss trials)"),
                        (set(POS) - imp, "poststroke_lick", "#2166ac", "s",
                         "positions still licked (lick trials)")):
                    cs = by_cls.get(cls, {})
                    xs, ys, es = [], [], []
                    for lab in sorted(cs):
                        vals = [(cs[lab].get(p) or {}).get(p) or {} for p in group]
                        vals = [c["mean"] for c in vals
                                if c.get("mean") is not None and (c.get("n") or 0) >= 10]
                        if not vals:
                            continue
                        xs.append(_day(an, lab.split("_")[-1]))
                        ys.append(float(np.mean(vals)))
                        es.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                                  if len(vals) > 1 else 0.0)
                    if xs:
                        order = np.argsort(xs)
                        ax.errorbar(np.array(xs)[order], np.array(ys)[order],
                                    yerr=np.array(es)[order], color=col, marker=mk, ms=5, lw=1.8,
                                    capsize=3, elinewidth=0.9,
                                    label=lbl if (ri == 0 and ci == 0) else None)
                        ax.set_xticks(sorted({int(v) for v in xs}))
            if ri == 0:
                ax.set_title(f"{an}\nimpaired: {', '.join(sorted(imp)) or 'none'}", fontsize=10,
                             fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{wname}\ncoding retained")
            if ri == len(WINDOWS) - 1:
                ax.set_xlabel("days from lesion")
            ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(fontsize=11, loc="best")
    _suptitle(fig, "How much of each position's PRE-STROKE code survives, over days after the lesion.\n"
                 "1.0 (green) = that position's own pre-stroke signature; 0 = indistinguishable from "
                 "the other positions. Mean over positions in each group, error bars = SEM across "
                 "positions.\nThe two groups use DIFFERENT trial classes because an impaired "
                 "position has almost no lick trials to average.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1.0))   # top reserved by _suptitle
    _footer(fig, _cd_labels())
    p = Path(out_dir) / "grant_3a_coding_retained.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 3b. frozen vs within
def _binom_ci(p, n):
    """Half-width of a normal-approx binomial 95% CI, 0 when n is 0."""
    return 1.96 * float(np.sqrt(max(p * (1 - p), 0) / n)) if n else 0.0


#: (window, section_g condition key, display name, which ARM it comes from).
#:
#: POST-LICK COMES FROM THE LICK-ONLY ARM, and that is not a workaround -- it is the only arm in
#: which the condition is defined. The ALL-trials arm includes trials with no detected lick and a
#: lick-aligned window cannot be built for a trial with no lick, so `poststroke_section_g` skips it
#: there (`if align == "lick" and arm_all: continue`) and computes it for lick-only, where every
#: trial has a lick by construction. It has been computed all along, in all 24 session records, with
#: permutation nulls and the pre-stroke band -- I asserted otherwise on 2026-08-24 after checking
#: `arms["all"]` alone and generalising from one arm to the analysis.
#:
#: THE COST, which is why the row is drawn with its own chance line: the lick-only arm scores each
#: session on ITS OWN preserved positions, so the class count and therefore chance differ between
#: sessions -- 4-way at 0.25 on one day and 6-way at 0.167 on another. Accuracies in that row are
#: NOT comparable across sessions or with the two rows above it.
FROZEN_WINDOWS = (("ENL", "pre-cue", "ENL (pre-cue)", "all"),
                  ("cue", "post-cue", "post-cue", "all"),
                  ("lick", "post-lick", "post-lick  [lick-only arm]", "lickonly"))


def fig_frozen_vs_within(out_dir):
    sg = _fig_root() / "section_g.json"
    if not sg.exists():
        return None
    G = json.loads(sg.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(len(FROZEN_WINDOWS), len(ANIMALS), figsize=(10.4, 7.2),
                             sharey="row", squeeze=False)
    for ri, (_disp, gkey, wname, armkey) in enumerate(FROZEN_WINDOWS):
        for ci, an in enumerate(ANIMALS):
            ax = axes[ri][ci]
            fx, fy, fe, wx, wy, we, band = [], [], [], [], [], [], None
            chance_x, chance_y = [], []
            for sess in sorted(k for k in G if k.startswith(an)):
                mmdd = sess.split("_")[-1]
                if config.session_phase(an, mmdd) != "post":
                    continue
                day = _day(an, mmdd)
                arm = (G[sess].get("arms") or {}).get(armkey) or {}
                if arm.get("chance"):
                    chance_x.append(day); chance_y.append(arm["chance"])
                cell = arm.get(gkey) or {}
                n = cell.get("n") or 0
                if cell.get("accuracy") is not None:
                    fx.append(day); fy.append(cell["accuracy"])
                    fe.append(_binom_ci(cell["accuracy"], n))
                # WITHIN-SESSION LIVES IN section_g TOO, under "<cond> within-session" -- but as a
                # POOLED block carrying a per_session LIST across every session, not a scalar for
                # this one. Reading it as a scalar silently produced nothing; reading it from
                # poststroke_grid.json (the other obvious source) gives only day 1 and 2, which is
                # one or two points per animal and no trajectory at all.
                wblk = arm.get(f"{gkey} within-session") or {}
                if wblk and band is None:
                    band = wblk.get("within_pre_band")
                # EACH SESSION'S BLOCK CARRIES ONLY ITS OWN POST ROW -- the rest of `per_session` is
                # that animal's pre-stroke sessions, which is what the band is built from. Taking
                # the list from the first session and stopping (the obvious read) yields exactly one
                # green point per animal, which is what the first version of this figure showed.
                for row in wblk.get("per_session", []):
                    if row.get("post") and row.get("label") == sess \
                            and row.get("within_accuracy") is not None:
                        wx.append(day); wy.append(row["within_accuracy"])
                        we.append(_binom_ci(row["within_accuracy"], row.get("n") or n))
            if wx:
                order = np.argsort(wx)
                wx = list(np.array(wx)[order]); wy = list(np.array(wy)[order])
                we = list(np.array(we)[order])
            if band:
                ax.axhspan(band["min"], band["max"], color="tab:blue", alpha=0.15, zorder=1,
                           label="pre-stroke range" if (ri == 0 and ci == 0) else None)
            if fx:
                ax.errorbar(fx, fy, yerr=fe, color="tab:red", marker="o", ms=5, lw=1.6, capsize=3,
                            label="FROZEN pre-stroke decoder" if (ri == 0 and ci == 0) else None)
            if wx:
                ax.errorbar(wx, wy, yerr=we, color="tab:green", marker="s", ms=5, lw=1.6, capsize=3,
                            ls="--",
                            label="trained on that session" if (ri == 0 and ci == 0) else None)
            # CHANCE IS PER SESSION IN THE LICK-ONLY ARM. One flat 1/6 line would be wrong on
            # every 4-position session, and drawing it anyway is how a 4-way 0.5 gets read as
            # twice chance when it is exactly twice a DIFFERENT chance.
            if armkey == "lickonly" and chance_x:
                o = np.argsort(chance_x)
                ax.step(np.array(chance_x)[o], np.array(chance_y)[o], where="mid", color="k",
                        ls=":", lw=1.1)
            else:
                ax.axhline(1 / 6, color="k", ls=":", lw=1.0)
            ax.set_ylim(0, 1.02)
            if fx or wx:
                ax.set_xticks(sorted({int(v) for v in list(fx) + list(wx)}))
            if ri == 0:
                ax.set_title(an, fontsize=12, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{wname}\naccuracy")
            if ri == len(FROZEN_WINDOWS) - 1:
                ax.set_xlabel("days from lesion")
            ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(fontsize=9.5, loc="lower left")
    _suptitle(fig, "Does the OLD code still read out, and is position information still there?\n"
                 "RED = frozen pre-stroke decoder.  GREEN = decoder trained on that session.  "
                 "Band = pre-stroke range.  Bars = binomial 95% CIs.\n"
                 "Top two rows: all trials, 6 positions, chance 1/6. Bottom row: LICK-ONLY arm "
                 "(a trial with no lick has no lick-aligned window), so chance is per session "
                 "(dotted step) and those panels are NOT comparable across sessions.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1.0))   # top reserved by _suptitle
    _footer(fig, _sg_labels())
    p = Path(out_dir) / "grant_3b_frozen_vs_within.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--compact", action="store_true",
                    help="also emit *_compact.png of the dense per-day grids: no in-cell numbers "
                         "and narrower panels, for reproduction at a quarter of a letter page")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=("1", "1b", "2", "2b", "3a", "3b", "4", "5", "5b", "5c", "5d", "6",
                             "6b", "6d", "7", "7b", "7d", "8", "8b", "8d", "8e", "8g", "9"))
    args = ap.parse_args(argv)
    out = args.output or (Path(PathResolver().root("labcams")) / "grant_figures")
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ("1", "1b", "2", "2b", "3a", "3b", "4", "5", "5b", "5c", "5d", "6",
                             "6b", "6d", "7", "7b", "7d", "8", "8b", "8d", "8e", "8g", "9"))
    jobs = (("1", fig_behaviour), ("1b", fig_behaviour_collapsed),
            ("2", fig_prestroke_decoding), ("2b", fig_prestroke_decoding_cohort),
            ("3a", fig_coding_retained), ("3b", fig_frozen_vs_within),
            ("4", fig_confusion_prestroke), ("5", fig_confusion_pre_post),
            ("5b", fig_confusion_pre_post_working),
            ("5c", fig_confusion_per_session), ("5d", fig_confusion_delta),
            ("6", fig_pattern_similarity),
            ("6b", fig_pattern_similarity_per_session), ("6d", fig_pattern_delta),
            ("7", fig_splithalf_matrix), ("7b", fig_reliability_verdict),
            ("7d", fig_splithalf_delta),
            ("8", fig_crossnobis_cross), ("8b", fig_crossnobis_geometry),
            ("8d", fig_crossnobis_delta), ("8e", fig_asymmetry), ("8g", fig_geometry_by_position),
            ("9", fig_delta_trajectory))
    def _run(tag=""):
        for key, fn in jobs:
            if key not in want:
                continue
            try:
                p = fn(out)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! {tag}{key}: {type(ex).__name__} {str(ex)[:120]}", flush=True)
                continue
            print(f"  {tag}{'wrote ' + str(p) if p else f'{key}: no data'}", flush=True)

    _run()
    if args.compact:
        # SECOND PASS IN THE SAME PROCESS, deliberately. The compact variant differs only in how the
        # figures are DRAWN, and the expensive part is loading every session's LocaNMF fit, which
        # `_collect_7` and friends have already cached by now. A separate invocation would pay that
        # cost twice -- roughly an hour of the two -- to redraw the same numbers.
        #
        # COMPACT is a module flag rather than a parameter threaded through eighteen figure
        # functions; it is read by _colw/_txt/_out and nothing else branches on it.
        globals()["COMPACT"] = True
        print("\n  [compact] second pass: in-cell numbers dropped, panels narrowed, "
              "writing *_compact.png", flush=True)
        try:
            _run("[compact] ")
        finally:
            globals()["COMPACT"] = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
