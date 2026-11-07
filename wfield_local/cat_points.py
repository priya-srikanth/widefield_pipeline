"""CATEGORICAL-x summary plots: one definition of how an epoch summary is drawn at a spout / epoch category.

Priya 2026-11-03 ("can cohort metrics minus pre contra (and similar figures) (1) have no lines connecting spout
positions -- review all figures across pages -- and (2) have faded dots for session data with each? can increase
lateral offset between epochs if needed"). The rules, for every figure whose x axis is a CATEGORY (spout position,
epoch):

1. NO LINE joins categories -- each spout / epoch is its own category (the earlier rule already banned lines across
   epochs). An epoch summary is a marker + interval only (`summary_point`). Time-continuous x (ms from cue, lick phase,
   days) keeps its traces; this module is not for those.
2. Every summary marker carries the SESSION values behind it as faded dots (`session_dots`): OPEN markers, shape =
   animal (`ANIMAL_MARKER`, Priya 10-27), color = epoch, alpha ~0.4, a small seeded jitter.
3. Epochs within one category sit at clearly separated lateral offsets (`epoch_offsets`, default 0.26 apart for 3
   epochs) so their dots do not overlap.

Significance marks stay in the top band (`sig_marks.star_top` + `reserve_headroom`, or `orofacial_cohort._tidy`);
pass the same offsets to the star x so a star sits over its epoch.
"""
from __future__ import annotations

import functools

import numpy as np

#: one marker per animal (colour = epoch, shape = animal; Priya 10-27). The SOURCE is now each animal's `marker:` in
#: its cohort's animals file (configs/animals.yaml, configs/cohort2/animals.yaml; added 2026-11-04). This dict is the
#: cohort-1 copy kept for importers and as the fallback; it equals `orofacial_cohort.ANIMAL_MARKER`, and
#: tests/test_cat_points.py pins it against the yaml so no published cohort-1 figure changes shape.
ANIMAL_MARKER = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
#: Fallback for an animal with no `marker:` in any cohort file. A name hash, so it can COLLIDE: before the yaml key
#: existed PS118 and PS121 both got ">" (sum of ord 317 and 311, both 5 mod 6; code review 2026-11-04). Give every
#: new animal an explicit marker rather than relying on this.
_SPARE = ("v", "P", "X", "h", "<", ">")
SESSION_ALPHA = 0.45
SESSION_MS = 3.6
JITTER = 0.045


@functools.lru_cache(maxsize=None)
def configured_markers() -> dict:
    """{animal: marker} from the `marker:` key of EVERY cohort's animals file, not only the active cohort's, so a
    cohort-2 animal keeps its shape in a figure rendered while cohort1 is active (and vice versa)."""
    from wfield_local import config
    out: dict = {}
    for spec in (config._load_file("cohorts.yaml").get("cohorts") or {}).values():
        for animal, meta in (config._load_file(spec["animals_file"]).get("animals") or {}).items():
            if (meta or {}).get("marker"):
                out[str(animal)] = str(meta["marker"])
    return out


def animal_marker(animal: str) -> str:
    m = configured_markers().get(str(animal))
    if m:
        return m
    if animal in ANIMAL_MARKER:
        return ANIMAL_MARKER[animal]
    return _SPARE[sum(map(ord, str(animal))) % len(_SPARE)]          # stable across runs (no hash seed)


def epoch_offsets(n: int, step: float = 0.26) -> np.ndarray:
    """Lateral offsets for n epochs drawn side by side inside one category (centered on 0)."""
    return (np.arange(n) - (n - 1) / 2.0) * step


def _jitter(n: int, width: float, seed: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return np.random.default_rng(seed).uniform(-width, width, n)


def session_dots(ax, x: float, values, animals, color, *, jitter: float = JITTER, alpha: float = SESSION_ALPHA,
                 ms: float = SESSION_MS, seed: int = 0, zorder: float = 1.5) -> None:
    """Faded OPEN markers for the session values at one category x (shape = animal, color = the epoch's)."""
    v = np.asarray(values, float)
    an = np.asarray(list(animals), dtype=object)
    ok = np.isfinite(v)
    v, an = v[ok], an[ok]
    if not len(v):
        return
    xs = x + _jitter(len(v), jitter, seed)
    for a in sorted(set(an)):
        k = an == a
        ax.plot(xs[k], v[k], ls="none", marker=animal_marker(a), ms=ms, mfc="none", mec=color, mew=0.8, alpha=alpha,
                zorder=zorder)


def summary_point(ax, x, mu, se=None, color="k", *, ms: float = 5.5, label=None, capsize: float = 2.5,
                  marker: str = "o", zorder: float = 4):
    """Epoch summary at a category: filled marker + interval, NO connecting line (rule 1). x / mu / se may be arrays."""
    x, mu = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(mu, float))
    yerr = None if se is None else np.atleast_1d(np.asarray(se, float))
    if yerr is not None:
        yerr = np.where(np.isfinite(yerr), yerr, 0.0)
    return ax.errorbar(x, mu, yerr=yerr, fmt=marker, ls="none", color=color, ms=ms, mec="w", mew=0.6,
                       capsize=capsize, elinewidth=1.1, zorder=zorder, label=label)


def add_animal_entries(ax, animals, color="0.35") -> None:
    """Put the session-dot shapes in ``ax``'s legend as empty artists, so a layout pass that rebuilds legends from
    `ax.get_legend_handles_labels()` (e.g. `orofacial_cohort._tidy`) keeps them."""
    for a in sorted(animals):
        ax.plot([], [], ls="none", marker=animal_marker(a), mfc="none", mec=color, mew=0.9, ms=SESSION_MS + 1,
                label=f"{a} sessions")


def animal_handles(animals, color="0.35"):
    """Legend handles for the session-dot shapes (open, gray)."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], ls="none", marker=animal_marker(a), mfc="none", mec=color, mew=0.9, ms=SESSION_MS + 1,
                   label=f"{a} sessions") for a in sorted(animals)]


# ---- EPOCH-COLOUR CONVENTION for session points (PI decision A13, Priya 11-07) ------------------------------------------
# Every session point: OPEN marker, SHAPE = animal (`animal_marker`), COLOUR = EPOCH (`epoch_color`, the canonical
# `orofacial_cohort.EPOCH_LINE` -- never a private copy). NO line joins sessions across epochs. The per-epoch summary is a
# FILLED dot in the epoch colour with +- SEM across ANIMALS (`epoch_mean_sem`; Priya 11-07: "can the mean +- SEM be a
# dot and bars in the epoch color, as in other figures"). `day_dots` applies all of it to a days-from-stroke axis.

#: ONE palette for MODEL / series identity on the model-comparison figures (frozen_fr_compare.epoch_lines / bars and
#: their callers; Priya 11-07: a model keeps its coloured line across epochs, but its colour must not reuse a colour that
#: means something else). Paul Tol's muted set (sand swapped for a violet: sand read as subacute's goldenrod) minus the members near an EPOCH colour (EPOCH_LINE: black / #d01c8b /
#: #b8860b / #01665e), an ANIMAL colour (animals.yaml: tab:blue / red / orange / green, cohort2 tab:purple / brown / pink
#: / olive) or a trial-CLASS colour (#2c3e50 / #c0392b, Dark2): every entry is >= 25 CIELAB dE from all of them
#: (tests/test_cat_points_epoch_convention.py). Index order = the order a caller assigns them; past 7, cycle + hatch.
MODEL_COLORS = ("#332288", "#88CCEE", "#882255", "#B017CF", "#CC6677", "#AAAAAA", "#FFAABB")


def model_color(k: int) -> str:
    """The k-th model colour (cycles)."""
    return MODEL_COLORS[int(k) % len(MODEL_COLORS)]


#: order the epoch summaries / legend entries are drawn in
EPOCH_ORDER = ("pre", "acute", "subacute", "chronic")
#: colour of a session in NO epoch (the acute -> subacute gap, an animal without an epoch spec): neutral, never a
#: neighbouring epoch's colour
NO_EPOCH = "0.6"
DAY_MS = 4.2          # a day-axis session point is the data itself, so it is less faded than a `session_dots` dot
DAY_ALPHA = 0.8
DAY_MEW = 0.9


def epoch_color(epoch) -> str:
    """The canonical colour of ``epoch`` (`orofacial_cohort.EPOCH_LINE`); `NO_EPOCH` for None / an unknown label."""
    from wfield_local.orofacial_cohort import EPOCH_LINE
    return EPOCH_LINE.get(str(epoch), NO_EPOCH) if epoch is not None else NO_EPOCH


def epochs_of_days(animals, days) -> list:
    """Epoch of each (animal, day-from-stroke) pair via `epoch_figures.epoch_of_day` (the ONE day -> epoch rule);
    None where the day is not finite or falls in no epoch."""
    from wfield_local.epoch_figures import epoch_of_day
    out = []
    for a, d in zip(animals, days):
        try:
            d = float(d)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(epoch_of_day(str(a), int(np.floor(d))) if np.isfinite(d) else None)
    return out


def epoch_mean_sem(ax, x, per_animal_means, epoch, *, ms: float = 6.5, capsize: float = 3.0, zorder: float = 5,
                   label=None, xerr=None):
    """The per-epoch summary: FILLED dot in the epoch colour at (x, mean of ``per_animal_means``) with +- SEM across
    animals (ddof 1; no bar for a single animal). Returns ``(mean, sem, n_animals)``, or None when nothing is finite.
    ``xerr`` (optional) adds a horizontal bar for a 2-D summary."""
    v = np.asarray([u for u in np.atleast_1d(np.asarray(per_animal_means, float)) if np.isfinite(u)], float)
    if not len(v):
        return None
    mu = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
    if xerr is None:
        summary_point(ax, x, mu, se, epoch_color(epoch), ms=ms, capsize=capsize, zorder=zorder, label=label)
    else:
        ax.errorbar([x], [mu], yerr=None if se is None else [se], xerr=[xerr], fmt="o", ls="none",
                    color=epoch_color(epoch), ms=ms, mec="w", mew=0.6, capsize=capsize, elinewidth=1.1,
                    zorder=zorder, label=label)
    return mu, (np.nan if se is None else se), len(v)


def day_dots(ax, days, values, animals, epochs=None, *, ms: float = DAY_MS, alpha: float = DAY_ALPHA,
             mew: float = DAY_MEW, zorder: float = 3, summary: bool = True, summary_ms: float = 6.5,
             summary_x=None) -> dict:
    """Session values on a DAYS axis under the epoch-colour convention: open marker per session (shape = animal,
    colour = epoch), NO connecting line, and (``summary``) per epoch a filled dot in the epoch colour +- SEM across
    animals, at x = the mean over animals of each animal's mean day in that epoch (``summary_x`` = {epoch: x}
    overrides). ``epochs`` defaults to `epochs_of_days` (the per-animal boundaries). Each animal is weighted equally in
    the summary: its sessions in the epoch are averaged first.

    Returns ``{epoch: (x, mean, sem, n_animals)}`` for the summaries drawn."""
    d = np.asarray(days, float)
    v = np.asarray(values, float)
    an = np.asarray([str(a) for a in animals], dtype=object)
    ep = np.asarray(epochs_of_days(an, d) if epochs is None else
                    [None if (e is None or (isinstance(e, float) and not np.isfinite(e))) else str(e) for e in epochs],
                    dtype=object)
    ok = np.isfinite(d) & np.isfinite(v)
    d, v, an, ep = d[ok], v[ok], an[ok], ep[ok]
    for a in sorted(set(an)):
        for e in sorted(set(ep[an == a]), key=lambda s: (s is None, str(s))):
            k = (an == a) & np.array([x == e for x in ep], bool)
            ax.plot(d[k], v[k], ls="none", marker=animal_marker(a), ms=ms, mfc="none", mec=epoch_color(e), mew=mew,
                    alpha=alpha, zorder=zorder)
    out = {}
    if not summary:
        return out
    present = [e for e in EPOCH_ORDER if e in set(ep)]
    present += sorted({e for e in ep if e is not None and e not in EPOCH_ORDER})
    for e in present:
        k = np.array([x == e for x in ep], bool)
        am = {a: (float(v[k & (an == a)].mean()), float(d[k & (an == a)].mean())) for a in sorted(set(an[k]))}
        x = (summary_x or {}).get(e, float(np.mean([t[1] for t in am.values()])))
        r = epoch_mean_sem(ax, x, [t[0] for t in am.values()], e, ms=summary_ms, zorder=zorder + 2)
        if r is not None:
            out[e] = (x, *r)
    return out


def epoch_handles(epochs=EPOCH_ORDER, label: str = "{e}: mean +- SEM over animals"):
    """Legend handles for the epoch summaries (filled dot in the epoch colour)."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], ls="none", marker="o", color=epoch_color(e), mec="w", ms=6.5, label=label.format(e=e))
            for e in epochs]


def epoch_colour_handles(epochs=EPOCH_ORDER, label: str = "{e}"):
    """Legend handles saying "point colour = epoch" (small open circle per epoch)."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], ls="none", marker="o", mfc="none", mec=epoch_color(e), mew=1.1, ms=SESSION_MS + 1,
                   label=label.format(e=e)) for e in epochs]


def legend_handles(animals, epochs=EPOCH_ORDER, *, sessions_label: str = "{a} sessions",
                   epoch_label: str = "{e}: mean +- SEM over animals"):
    """Animal shapes (open, grey) + epoch colours: the legend every convention figure carries."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="none", marker=animal_marker(a), mfc="none", mec="0.35", mew=0.9, ms=SESSION_MS + 1,
                label=sessions_label.format(a=a)) for a in sorted(animals)]
    return h + epoch_handles(epochs, epoch_label)


def fit_ylim(ax, values, pad: float = 0.04) -> None:
    """Grow ``ax``'s y range (never shrink it) so every finite value in ``values`` is inside, with ``pad`` of the range
    as margin: a figure with a fixed ylim must not clip a plotted point silently."""
    v = np.asarray([u for u in np.ravel(np.asarray(values, dtype=float)) if np.isfinite(u)], float)
    if not len(v):
        return
    lo, hi = ax.get_ylim()
    vlo, vhi = float(v.min()), float(v.max())
    if vlo >= lo and vhi <= hi:
        return
    span = max(max(hi, vhi) - min(lo, vlo), 1e-12)
    ax.set_ylim(min(lo, vlo - pad * span), max(hi, vhi + pad * span))


def no_cross_epoch_lines(ax) -> list:
    """Line2D artists in ``ax`` that join >= 2 finite points with a visible line (a test helper: a convention figure's
    session layer must have none; reference lines -- axhline / axvline / fits -- are the caller's to exclude)."""
    bad = []
    for ln in ax.get_lines():
        if ln.get_linestyle() in ("None", "none", "", " ") or ln.get_transform() != ax.transData:
            continue                                # no line, or an axhline / axvline (axes-fraction coordinates)
        x, y = (np.asarray(t, float) for t in ln.get_data())
        if np.isfinite(x).sum() >= 2 and np.isfinite(y).sum() >= 2 and len(set(np.round(x[np.isfinite(x)], 9))) >= 2:
            bad.append(ln)
    return bad
