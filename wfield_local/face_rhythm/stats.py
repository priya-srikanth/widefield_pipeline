"""Change vs each animal's own pre-stroke sessions, the way every FaceRhythm readout reports it.

Two steps, always in this order (CLAUDE.md rules 8 and 9):
  1. per animal: the mean of its sessions in each epoch, relative to its own pre-stroke sessions -- PRINTED FIRST, so a
     cell carried by one animal is visible before any interval is;
  2. the change vs pre by `analysis_kit.boot_delta` (paired animals -> sessions bootstrap, animals weighted equally,
     95 % interval). There is no second bootstrap here: this module only arranges the inputs.

Marks (Priya 10-24 "mark with asterisks where appropriate"): ``*`` = the 95 % interval excludes 0 (one planned test);
``**`` = also significant after a Holm correction over the FAMILY the figure tests at once (e.g. all 12 factors of an
epoch x trial group in the survey), from the bootstrap p of the same draws (`analysis_kit.boot_p`, crude at n = 4:
used to rank and correct, not quoted as an exact probability).

Scales: LOG2 for amplitudes (halving and doubling equally large, Priya 10-24; pre averages exactly 0), ratio / pre or
difference - pre for quantities that are already relative (EVR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wfield_local.face_rhythm.common import POST_EPOCHS


def log2_vs_pre(S: pd.DataFrame, value: str, by: list[str], floor: float = 1e-12) -> pd.Series:
    """log2(value) minus the mean log2 value of the same animal's PRE sessions within each ``by`` group (``by``
    includes "animal"); pre sessions average exactly 0. ``floor`` guards log2(0)."""
    lg = np.log2(np.maximum(S[value], floor))
    return lg - lg.where(S.epoch == "pre").groupby([S[b] for b in by]).transform("mean")


def by_animal(d: pd.DataFrame, value: str) -> dict[str, list]:
    """{animal: [session values]} in the frame's row order -- the input `boot_delta` takes."""
    return {an: list(g[value]) for an, g in d.groupby("animal")}


def epoch_change(d: pd.DataFrame, value: str, rng, n_boot: int | None = None, epochs=POST_EPOCHS,
                 baseline: float = 0.0) -> list[dict]:
    """One row per post epoch for ONE quantity (``d`` = its sessions, all epochs, column ``value`` already relative
    to pre): per-animal epoch means (one column per animal), their mean and SEM across animals, the `boot_delta`
    change vs pre (delta, ci_lo, ci_hi, n_animals, excludes_0) and how many animals sit below / above ``baseline``.
    ``rng`` is advanced in epoch order, so a caller looping over quantities reproduces the same draws every run."""
    from wfield_local import analysis_kit as ak
    pre = by_animal(d[d.epoch == "pre"], value)
    rows = []
    for ep in epochs:
        post = by_animal(d[d.epoch == ep], value)
        per = {an: float(np.mean(v)) for an, v in sorted(post.items())}
        iv, draws = ak.boot_delta(post, pre, rng, n_boot or ak.N_BOOT, draws=True)
        vals = list(per.values())
        rows.append({"epoch": ep, **per,
                     "mean": float(np.mean(vals)) if vals else np.nan,
                     "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                     "delta": iv.point if iv else np.nan, "ci_lo": iv.lo if iv else np.nan,
                     "ci_hi": iv.hi if iv else np.nan, "n_animals": iv.n_animals if iv else len(per),
                     "excludes_0": bool(iv and (iv.lo > 0 or iv.hi < 0)),
                     "p_boot": ak.boot_p(draws) if iv else np.nan,
                     "n_below": sum(v < baseline for v in vals), "n_above": sum(v > baseline for v in vals)})
    return rows


def add_holm(T: pd.DataFrame, family: list[str], p: str = "p_boot") -> pd.DataFrame:
    """Holm-adjust ``p`` within each family (rows sharing the ``family`` columns, e.g. ["epoch", "group"] for the 12
    factors of one epoch x trial group) -> columns ``p_holm`` and ``mark`` ("", "*", "**")."""
    from wfield_local import analysis_kit as ak
    T = T.copy()
    T["p_holm"] = np.nan
    for _, idx in T.groupby(family).groups.items():
        T.loc[idx, "p_holm"] = ak.holm(T.loc[idx, p].to_numpy(float))
    T["mark"] = [("**" if ph < 0.05 else "*") if ex else "" for ex, ph in zip(T.excludes_0, T.p_holm.fillna(1.0))]
    return T


def mark(row) -> str:
    """"*" / "**" of a stats row: its ``mark`` column when Holm-adjusted (`add_holm`), else "*" if the interval
    excludes 0."""
    if row is None:
        return ""
    m = row.get("mark") if hasattr(row, "get") else None
    if isinstance(m, str):
        return m
    return "*" if row["excludes_0"] else ""


def tick_label(epoch: str, row: dict | pd.Series | None, fmt: str = "+.2f", interval: bool = True) -> str:
    """Epoch tick label carrying its statistics: ``acute\\n-1.35*\\n[-1.56, -1.13]`` (* = interval excludes 0)."""
    if row is None or not np.isfinite(row["delta"]):
        return epoch[:5]
    s = f"{epoch[:5]}\n{row['delta']:{fmt}}{mark(row)}"
    return s + (f"\n[{row['ci_lo']:{fmt}}, {row['ci_hi']:{fmt}}]" if interval else "")
