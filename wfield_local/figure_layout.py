"""WHERE A FIGURE'S COMPANION FILES GO. One definition, so the layout cannot drift per writer.

Adopted 2026-09-21. Before it, every renderer built its own sidecar path with `with_suffix(".csv")`
or `with_name(stem + "_bundle.json")`, and the result was `grant_figures/epoch` holding 1,747 files
in one flat directory: 402 PNG, 352 SVG, 894 CSV, 42 JSON. **54% of the figure directory was not
figures**, and `ls` could not answer the one question anybody asks of it -- what figures exist.

    <dir>/<name>.png          the figure. The deck places these, the coverage report scans these.
    <dir>/svg/<name>.svg      the vector copy, a deliverable for pasting into the grant document.
    <dir>/data/<name>.csv     the numbers the figure plots, plus every other sidecar:
    <dir>/data/<name>_sessions.csv, _meta.csv, _stats.csv, _bundle.json, _bundle.npz

WHY THE PNG STAYS AT THE TOP AND EVERYTHING ELSE MOVES DOWN. The registry in `deck_registry` names
figures by bare filename and the deck globs the directory for them; keeping PNGs where they were
means 111 registry entries, the deck's placement loop and the completeness gate are all untouched
by this change. The things that moved are the things nothing addresses by path.

WHY SIDECARS ARE NOT OPTIONAL, since a reader may wonder why there are 894 of them. From
`write_values`: *a figure whose values exist nowhere else drifts from every text that quotes it,
and nothing can notice.* `PRELIM_DATA_VLS_STROKE.md` carried chronic frozen-vs-refit numbers from
an older session set for two days, because the only machine-readable route to them was re-running
a twenty-minute script and the only other route was someone re-reading a bar chart. The sidecars
are the fix; this module is only about where they live.

ALWAYS GO THROUGH `sidecar()` AND `svg_path()`. A writer that builds its own path puts a file back
in the flat directory, and nothing will report it -- the coverage tool counts PNGs, and a stray CSV
is invisible to every check in this repo.

SAVE A PAGE / DECK FIGURE WITH `save_png_svg(fig, png, **savefig_kw)` (2026-11-06): the PNG exactly as
`fig.savefig` would write it, and the SVG twin in `svg/`. `tests/test_save_png_svg.py` lists the
writers that must use it.
"""
from __future__ import annotations

import os
import pathlib
import warnings

#: Subdirectory names, in one place so a future move is one edit.
SVG_DIR = "svg"
DATA_DIR = "data"


def svg_path(png, *, mkdir: bool = True) -> pathlib.Path:
    """Where the vector copy of the figure at ``png`` belongs."""
    png = pathlib.Path(png)
    d = png.parent / SVG_DIR
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return d / (png.stem + ".svg")


#: savefig keywords that only mean something for a raster file. Dropped for the SVG twin: `dpi` is a
#: no-op for vector output (same rule `grant_kit._save` used since 2026-08-29), and PNG `metadata` keys /
#: `pil_kwargs` / an explicit `format="png"` would make the SVG call raise or write a PNG under .svg.
_RASTER_ONLY_KW = ("dpi", "format", "metadata", "pil_kwargs")


def save_png_svg(fig, fname, *args, **kwargs):
    """``fig.savefig(fname, *args, **kwargs)`` EXACTLY as the caller wrote it, plus a vector twin at
    ``svg_path(fname)`` when ``fname`` is a ``.png`` path (Priya, 2026-11-06: every page / deck figure
    is also saved as SVG, going forward).

    THE PNG IS UNCHANGED BY CONSTRUCTION. It is written first, by the same call with the same
    arguments the writer used before it was routed here, so dpi / bbox / facecolor / transparent
    all mean what they meant; `tests/test_save_png_svg.py` pins the bytes against a plain savefig.

    THE SVG CAN NEVER COST THE PNG. Any failure (an artist that cannot vectorize, a full disk, an
    unwritable `svg/` dir) warns and returns; a partial SVG is removed so a broken twin never sits
    beside a good figure. Text stays text (``svg.fonttype = "none"``), the setting the histology /
    orofacial writers already used, so labels are editable in Illustrator.

    Anything that is not a ``.png`` path (a ``.pdf``, an ``.svg``, a ``BytesIO``) is saved and nothing
    else happens. ``fig`` only needs a ``savefig`` method, so a seaborn grid works too; for
    ``plt.savefig(...)`` pass ``plt.gcf()``. Returns what ``savefig`` returned.
    """
    out = fig.savefig(fname, *args, **kwargs)
    if not isinstance(fname, (str, os.PathLike)):
        return out
    png = pathlib.Path(fname)
    if png.suffix.lower() != ".png" or str(kwargs.get("format") or "png").lower() != "png":
        return out
    svg = None
    try:
        import matplotlib as mpl
        svg = svg_path(png)
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig.savefig(svg, **{k: v for k, v in kwargs.items() if k not in _RASTER_ONLY_KW})
    except Exception as ex:                                            # noqa: BLE001
        if svg is not None:
            try:
                svg.unlink(missing_ok=True)
            except OSError:
                pass
        warnings.warn(f"[svg] {png.name}: SVG twin not written ({type(ex).__name__}: {ex}); PNG kept",
                      RuntimeWarning, stacklevel=2)
    return out


def sidecar(png, suffix: str, *, mkdir: bool = True) -> pathlib.Path:
    """Where ``<figure-stem><suffix>`` belongs, e.g. ``sidecar(q, ".csv")``.

    ``suffix`` carries its own separator: ``".csv"``, ``"_sessions.csv"``, ``"_bundle.json"``.
    """
    png = pathlib.Path(png)
    return sidecar_for(png.parent, png.stem, suffix, mkdir=mkdir)


def sidecar_for(out_dir, stem: str, suffix: str, *, mkdir: bool = True) -> pathlib.Path:
    """Same rule as `sidecar`, for a writer that has a DIRECTORY AND A STEM but no figure path.

    Most renderers hold the PNG they just saved. The `scripts/rest_migration` statistics writers
    do not -- they build ``out_dir / f"{stem}_sessions.csv"`` directly, and several of them write
    the CSV in a different function from the one that draws the figure. Before this existed they
    each reimplemented the join, and on 2026-09-22 all five were still writing FLAT while
    `find_sidecar` had already moved to ``data/``: the fresh numbers landed where nothing reads
    them and every reader kept resolving a two-day-old copy. Neither side errored.
    """
    d = pathlib.Path(out_dir) / DATA_DIR
    if mkdir:
        from wfield_local import lp_licks
        lp_licks.refuse_lp_folder(d, "figure sidecar")    # N5b: an LP_licks folder needs WIDEFIELD_LICK_SOURCE=lp
        d.mkdir(parents=True, exist_ok=True)
    return d / (stem + suffix)


def find_sidecar(png, suffix: str) -> pathlib.Path | None:
    """Read side: the new location, falling back to the old flat one.

    THE FALLBACK IS DELIBERATE AND IS NOT DEAD CODE. MICROSCOPE keeps originals, so sidecars
    written before 2026-09-21 -- and any directory somebody has not migrated, including a
    colleague's copy -- still sit beside their figure. A reader that only knew the new path would
    report "no data" for them, which reads exactly like a figure that never had any.
    """
    png = pathlib.Path(png)
    return find_sidecar_for(png.parent, png.stem, suffix)


def find_sidecar_for(out_dir, stem: str, suffix: str) -> pathlib.Path | None:
    """Read side of `sidecar_for`: the new location, falling back to the old flat one."""
    out_dir = pathlib.Path(out_dir)
    new = out_dir / DATA_DIR / (stem + suffix)
    if new.exists():
        return new
    old = out_dir / (stem + suffix)
    return old if old.exists() else None


def data_dirs(*roots) -> list[pathlib.Path]:
    """Each root plus its ``data/`` subdirectory, for consumers that search by stem.

    `deck_values.Resolver` is the caller that matters: it looks for ``<stem>.csv`` across a list of
    directories, and after the split the CSVs are one level down. Both are returned, old first, so
    an unmigrated tree keeps resolving.
    """
    out = []
    for r in roots:
        r = pathlib.Path(r)
        out += [r, r / DATA_DIR]
    return out
