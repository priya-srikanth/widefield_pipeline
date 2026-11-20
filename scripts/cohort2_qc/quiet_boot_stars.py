"""Bootstrap-primary significance marks for the quiet-event follow-on figures (Priya 11-18: "yes add" bootstrap stars to
the section-23 re-renders; the 11-07 rule: * = bootstrap, dagger = mixed model backup).

`mark(ax, x, S, col, ep, q)` replaces a script's LMM star at one (panel, epoch) position: * when the nested animals ->
sessions bootstrap 95 % CI of the CHANGE vs each animal's own pre mean (`analysis_kit.epoch_change`, the same object
as quiet_restyle_figs.change_cells) excludes 0; dagger (row below) when the script's mixed-model q < .05. Both in the
top band (sig_marks.star_top). The figure layout is untouched. Every mark is logged in RECORDS, written by `save` next to
the figure as <name>_boot_stars.csv (point, CI, animals in the direction, per-animal changes, LMM q).

S = per-session table with columns animal, epoch, <col>; `panel` must name the table + column uniquely (cache key). off: analyze log10(v + off), for measures whose model is on
the log scale (rates, power). The draw is seeded per (col, epoch), so a figure re-render reproduces its marks.
This module only places marks; every CALLER reserves the band with `sig_marks.reserve_headroom(fig)` once the figure is
drawn (all eight quiet_* callers do; tests/test_sig_marks_top_band checks for the name).
"""
from __future__ import annotations

import zlib

import numpy as np

EP = ("pre", "acute", "subacute", "chronic")
POST = ("acute", "subacute", "chronic")
DAGGER = "†"
RECORDS: list[dict] = []
_CACHE: dict = {}


def change(S, col, *, off=None, transform=None, key=None):
    """{epoch: EpochChange} for the change of `col` vs each animal's own pre mean. Cached on `key` (the caller's panel
    name; NOT id(S), which Python reuses for a garbage-collected slice); key=None = no cache."""
    from wfield_local import analysis_kit as kit
    if key is not None and key in _CACHE:
        return _CACHE[key]
    D = S[["animal", "epoch", col]].copy()
    v = D[col].astype(float)
    if transform is not None:
        v = transform(v)
    if off is not None:
        v = np.log10(v + off)
    D["v"] = v.replace([np.inf, -np.inf], np.nan)
    D = D.dropna(subset=["v"])
    D = D[D.animal.isin(D[D.epoch == "pre"].animal.unique())]
    rng = np.random.default_rng(zlib.crc32(f"{col}|{off}".encode()))
    out = kit.epoch_change(D, "v", rng, [e for e in POST if (D.epoch == e).any()])
    if key is not None:
        _CACHE[key] = out
    return out


def mark(ax, x, S, col, ep, q=np.nan, *, off=None, transform=None, color="k", row=0, panel="", fontsize=None):
    """Draw * (bootstrap) and dagger (LMM q < .05) above x for epoch `ep`; returns True when starred."""
    from wfield_local import sig_marks
    if ep not in POST:
        return False
    ch = change(S, col, off=off, transform=transform, key=(panel or col, col, off))
    c = ch[ep] if ep in ch else None
    star = bool(c is not None and np.isfinite(c.lo) and (c.lo > 0 or c.hi < 0))
    kw = {} if fontsize is None else {"fontsize": fontsize}
    if star:
        sig_marks.star_top(ax, x, "*", color=color, row=row, **kw)
    q = float(q) if q is not None and np.isfinite(float(q)) else np.nan
    if np.isfinite(q) and q < 0.05:
        sig_marks.star_top(ax, x, DAGGER, color=color, row=row + 1, fontsize=10)
    if c is not None:
        same = (c.n_up if c.point > 0 else c.n_animals - c.n_up) if np.isfinite(c.point) else 0
        RECORDS.append(dict(panel=panel or col, epoch=ep, scale="log10 fold" if off is not None else "difference",
                            point=c.point, lo=c.lo, hi=c.hi, k_same=same, n_animals=c.n_animals,
                            per_animal="; ".join(f"{a} {d:+.3g}" for a, d in sorted(c.per_animal.items())),
                            boot_star="*" if star else "", lmm_q=q,
                            lmm_star="***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""))
    return star


def save(path):
    """Write RECORDS to <path> (csv) and clear them; call once per figure after drawing."""
    import pandas as pd
    if RECORDS:
        pd.DataFrame(RECORDS).to_csv(path, index=False)
    RECORDS.clear()
    _CACHE.clear()


FOOT = ("* = nested animals -> sessions bootstrap 95% CI of the change vs each animal's own pre excludes 0 (primary); "
        "† = mixed model q < .05 (backup)")


def mark_level(ax, x, S, col, ep, q=np.nan, *, color="k", row=0, panel=""):
    """LEVEL test at one epoch (e.g. 'the pre slope differs from 0'): * when the nested bootstrap 95 % CI of the epoch's
    mean (`analysis_kit.boot_ci`, animals -> sessions) excludes 0; dagger = the model's q < .05. Logged like `mark`."""
    from wfield_local import sig_marks
    from wfield_local.analysis_kit import boot_ci
    B = S[S.epoch == ep]
    d = {an: list(A[col].dropna()) for an, A in sorted(B.groupby("animal")) if A[col].notna().any()}
    ci = boot_ci(d, np.random.default_rng(zlib.crc32(f"{panel}|{col}|{ep}|level".encode()))) if d else None
    star = bool(ci is not None and np.isfinite(ci.lo) and (ci.lo > 0 or ci.hi < 0))
    if star:
        sig_marks.star_top(ax, x, "*", color=color, row=row)
    q = float(q) if q is not None and np.isfinite(float(q)) else np.nan
    if np.isfinite(q) and q < 0.05:
        sig_marks.star_top(ax, x, DAGGER, color=color, row=row + 1, fontsize=10)
    if ci is not None:
        means = {an: float(np.mean(v)) for an, v in d.items() if v}
        RECORDS.append(dict(panel=panel or col, epoch=ep, scale="level (vs 0)", point=ci.point, lo=ci.lo, hi=ci.hi,
                            k_same=sum((m > 0) == (ci.point > 0) for m in means.values()), n_animals=len(means),
                            per_animal="; ".join(f"{a} {m:+.3g}" for a, m in sorted(means.items())),
                            boot_star="*" if star else "", lmm_q=q,
                            lmm_star="***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""))
    return star
