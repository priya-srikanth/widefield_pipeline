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
