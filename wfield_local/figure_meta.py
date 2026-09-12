"""The ANNOTATION layer of a figure, recorded beside it as ``<name>_meta.csv``.

Priya, 2026-09-12: "do the sidecars carry all information we would need to regenerate the figures
(without redoing analysis)?" -- they did not, and this closes the difference.

WHAT THE VALUE SIDECARS ALREADY COVER. `epoch_figures.write_values` and its four siblings record
every number a figure PLOTS: bar heights with both interval pairs, per-session dots, matrix cells
with the delta recomputed as the figure draws it, per-day traces. From those alone a reader can
redraw the data.

WHAT THEY DO NOT, AND WHY IT MATTERS. A figure also STATES things, and some of those statements are
data rather than styling:

    counts / coverage / pre_sessions   how many animals and sessions stand behind each epoch -- the
                                       imbalance the per-panel counts exist to expose (the acute
                                       panel is six PS94 sessions against one PS95 session).
    subtitle                           carries the stats line: N animals, n sessions, block counts,
                                       bootstrap draws. Recorded VERBATIM rather than re-plumbed,
                                       because `stats_line` composes it from four inputs and a
                                       reconstruction could disagree with what was printed.
    chance / reference                 the line a bar is read against. A bar chart without its
                                       chance level is not interpretable, and 1/6 and 1/3 both
                                       appear in this deck.
    n_comparisons                      the Bonferroni divisor. `write_values` stores the MARK but
                                       not what it was corrected against, so a reader could see
                                       `**` with no way to check the correction.
    vmin / vmax / unit                 a matrix panel's scale. The same array under two colour
                                       limits makes opposite visual claims.

So this file is not a convenience: without it a regenerated figure would carry the right bars and be
unable to say what they rest on.

DELIBERATELY NOT HERE: the cortical maps' pixels. `write_map_summary` digests them on purpose --
see its docstring -- and the arrays remain the collector's cached output.

ONE FILE, KEY-VALUE, so it survives a field being added. Columns are ``kind, key, value``: scalars
get an empty key, a mapping gets one row per entry, and a nested mapping (``coverage`` is
``{epoch: {animal: n}}``) flattens to ``epoch|animal``. That shape diffs cleanly across renders,
which is the point -- "did the acute panel gain a session?" should be a diff, not an act of memory.
"""
from __future__ import annotations

import csv
import pathlib


def _flatten(kind, value):
    """``(kind, key, value)`` rows for one field, whatever shape it arrived in."""
    if value is None:
        return []
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            if isinstance(v, dict):
                out += [(kind, f"{k}|{k2}", v2) for k2, v2 in v.items()]
            else:
                out.append((kind, str(k), v))
        return out
    if isinstance(value, (list, tuple)):
        # A 2-tuple that is plainly an axis limit reads better as lo/hi than as 0/1.
        if len(value) == 2 and kind in ("ylim", "xlim", "clim"):
            return [(kind, "lo", value[0]), (kind, "hi", value[1])]
        return [(kind, str(i), v) for i, v in enumerate(value)]
    return [(kind, "", value)]


def write_meta(q, **fields):
    """Write ``<name>_meta.csv`` beside the figure at ``q``. Returns the path, or None if empty.

    Every field is optional; ``None`` is dropped rather than written as a blank, because "not
    applicable to this family" and "applicable and empty" are different facts and only the second
    should occupy a row.

    Never raises. A sidecar must not cost a figure that has already rendered -- the rule
    `_save_png_svg` follows for SVG and every value writer follows here.
    """
    rows = []
    for kind, value in fields.items():
        rows += _flatten(kind, value)
    if not rows:
        return None
    q = pathlib.Path(q)
    out = q.with_name(q.stem + "_meta.csv")
    try:
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["kind", "key", "value"])
            for kind, key, value in rows:
                w.writerow([kind, key, value])
    except Exception as ex:                                            # noqa: BLE001
        print(f"  [meta] {out.name}: failed ({type(ex).__name__})", flush=True)
        return None
    return out
