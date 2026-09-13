"""Let a deck note QUOTE a figure's sidecar instead of hard-coding the number.

Priya, 2026-09-12: "yes have notes reference the sidecar. we should be able to do that and implement
before the deck build."

THE PROBLEM, MEASURED. 171 decimal statistics are hand-copied into 33 deck notes. Nothing links them
to the figures they describe, so a re-render moves the figure and leaves the caption asserting the
old value in the same confident prose. That is not hypothetical and it is not rare: changing the
state decoder's licking anchor on 2026-09-12 invalidated the class balance, the refit ceiling and
the session-time control quoted across six `epoch_13*` notes, and the only thing that caught it was
being asked whether accuracy had been checked. `deck_audit` verifies that a note EXISTS and is
structurally blind to whether it is true.

A token resolves at DECK-BUILD time from the CSV the render wrote beside the figure:

    "far-contralateral keeps {{epoch_5rmogapdelta_frozen_vs_refit_cue_lick: epoch=acute,
     position=gap -> point:+.3f}} of recoverable component"

Read it as: open that figure's sidecar, take the row where ``epoch`` is ``acute`` and ``position``
is ``gap``, print its ``point`` column with that format. Any sidecar works, because the selector is
just column filters -- bar values, contrast intervals, matrix cells, per-cell bootstraps, the
metadata file. See `epoch_figures.write_values` and its siblings for the column names.

``SELF`` AS THE STEM RESOLVES TO THE FIGURE ON THAT SLIDE, and it is the form most notes need. A
note is placed by a GLOB -- one caption serves all five trial-class arms -- so a token naming
``..._cue_lick`` explicitly would print the post-cue lick numbers onto the pre-cue slide too. With
``SELF`` each slide quotes its own figure:

    "the frozen arm loses {{SELF: epoch=acute, position=frozen -> point:+.3f}} acutely"

Name a stem explicitly only when the note deliberately cites ANOTHER figure -- comparing this arm
against a named one, for instance -- and then say in the prose which figure it is.

A FAILURE IS LOUD AND LOCAL, NEVER SILENT AND NEVER FATAL. A missing sidecar or an unmatched
selector substitutes a visible ``[[? ...]]`` marker and is counted; the deck still builds, because a
family that has not been re-rendered yet must not block publishing every other slide. The marker is
the point: a caption that cannot find its number should say so on the slide rather than quietly keep
the last one that worked.

THE TOKEN IS RESOLVED BEFORE THE METHODS-BLOCK DEDUP HASHES IT, so two notes whose prose is
identical but whose numbers differ stay two blocks rather than the second being replaced by "same as
slide N". Hashing first would point a reader at another slide's numbers, and a wrong
cross-reference reads exactly like a right one.
"""
from __future__ import annotations

import csv
import pathlib
import re

#: ``{{<figure stem>: <col>=<value>, ... -> <field>[:<format spec>]}}``
TOKEN = re.compile(
    r"\{\{\s*(?P<stem>[A-Za-z0-9_.\-]+)\s*:\s*(?P<where>[^}]*?)\s*->\s*"
    r"(?P<field>[A-Za-z0-9_]+)\s*(?::(?P<spec>[^}]+?))?\s*\}\}")


def _parse_where(text):
    """``"epoch=acute, position=gap"`` -> ``{"epoch": "acute", "position": "gap"}``."""
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out


class Resolver:
    """Resolves sidecar tokens in note text, remembering every failure for the build log.

    ``dirs`` are searched in order for ``<stem>.csv``. Rows are read once per file per build.
    """

    def __init__(self, dirs):
        self.dirs = [pathlib.Path(d) for d in dirs]
        self.misses = []
        self._cache = {}

    def rows(self, stem):
        """Rows of ``<stem>.csv``, or None if no directory has it."""
        if stem not in self._cache:
            self._cache[stem] = None
            for d in self.dirs:
                q = d / f"{stem}.csv"
                if q.is_file():
                    try:
                        with open(q, newline="", encoding="utf-8") as fh:
                            self._cache[stem] = list(csv.DictReader(fh))
                    except Exception as ex:                            # noqa: BLE001
                        self._cache[stem] = None
                        self.misses.append(f"{stem}.csv unreadable ({type(ex).__name__})")
                    break
        return self._cache[stem]

    def lookup(self, stem, where, field, spec=None):
        """One formatted value, or a visible marker. Never raises."""
        rows = self.rows(stem)
        if rows is None:
            self.misses.append(f"{stem}.csv not found")
            return f"[[? {stem} sidecar missing]]"
        hit = [r for r in rows
               if all(str(r.get(k, "")).strip() == v for k, v in where.items())]
        if not hit:
            self.misses.append(f"{stem}: no row for {where}")
            return f"[[? {stem} has no row {where}]]"
        if field not in hit[0]:
            self.misses.append(f"{stem}: no column {field!r}")
            return f"[[? {stem} has no column {field}]]"
        vals = {r[field] for r in hit}
        if len(vals) > 1:
            # AMBIGUOUS IS A FAILURE, NOT A MEAN. Silently averaging two rows would invent a number
            # that appears in no sidecar and can never be traced back.
            self.misses.append(f"{stem}: {where} matched {len(hit)} rows with different {field}")
            return f"[[? {stem} {where} is ambiguous ({len(hit)} rows)]]"
        raw = hit[0][field]
        if raw == "":
            return "n/a"
        if spec:
            try:
                return format(float(raw), spec)
            except (TypeError, ValueError):
                return raw
        return raw

    def resolve(self, text, self_stem=None):
        """Substitute every token in ``text``. Returns the text unchanged if it has none.

        ``self_stem`` is the figure on the slide being annotated; it backs the ``SELF`` stem.
        """
        if not text or "{{" not in text:
            return text

        def one(m):
            stem = m["stem"]
            if stem == "SELF" or stem.startswith("SELF_"):
                if not self_stem:
                    self.misses.append("SELF used on a slide with no figure")
                    return "[[? SELF has no figure on this slide]]"
                # A FIGURE HAS SEVERAL SIDECARS, not one. `SELF` is the value file; `SELF_stats`,
                # `SELF_sessions`, `SELF_meta` and `SELF_cells` reach the others, which is how a map
                # note quotes the STATISTICS its claims are made from rather than the per-panel
                # digest that happens to share the figure's name.
                stem = self_stem + stem[len("SELF"):]
            return self.lookup(stem, _parse_where(m["where"]), m["field"], m["spec"])

        return TOKEN.sub(one, text)

    def report(self):
        """One line per distinct failure, for the build log. Empty when everything resolved."""
        seen, out = set(), []
        for m in self.misses:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out


def stem_of(filename):
    """``"epoch_9_delta_cue_working.png"`` -> ``"epoch_9_delta_cue_working"``."""
    return pathlib.Path(filename).stem if filename else None
