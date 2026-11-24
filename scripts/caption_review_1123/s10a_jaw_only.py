"""S10A caption review (Priya 11-23, item 5): what changes if the "first movement" of the first-lick delay is jaw
opening ONLY, instead of the earliest of jaw opening / cam1 (below-camera) opening / FaceRhythm lick-vigor threshold.

Reads the four-camera rev-up per-trial table (`scripts/revup_behaviour.py` output, cam3-on run) and recomputes, with
the SAME rules as `revup_behaviour._revup` / `_cell` / `summarise` (canonical = camera working-period trials, licked
trials only, per animal x spout x epoch medians):
  move_all  = earliest of jaw_ms / first_cam1_ms / fr_onset_ms that is <= the first-lick latency (production)
  move_jaw  = jaw_ms if <= latency, else latency when the jaw was judged (the production fallback: no movement before
              the lick -> wait = latency, rev-up 0)
Writes cells tables in the production schema (wait_med_ms, revup_med_ms) so the S10A renderer can redraw them, plus
a source-share table (which signal sets the first movement) and a summary (mean over animals per position x epoch).

    python s10a_jaw_only.py <revup_trials.csv.gz> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
EP = ["pre", "acute", "subacute", "chronic"]


def move(T: pd.DataFrame, cols: list[str], judged: np.ndarray) -> np.ndarray:
    lat = T.lat_ms.to_numpy(float)
    cand = T[cols].to_numpy(float)
    cand = np.where(cand <= lat[:, None] + 1e-6, cand, np.nan)
    mv = np.nanmin(np.where(np.isfinite(cand), cand, np.inf), 1)
    return np.where(np.isfinite(mv), mv, np.where(np.isfinite(lat) & judged, lat, np.nan))


def main(src: str, out: str) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(src)
    T["canonical"] = T.canonical.astype(str).str.lower().eq("true")
    for c in ("jaw_judged", "has_cam1", "has_fr"):
        T[c] = T[c].astype(str).str.lower().eq("true")
    C = T[T.canonical & np.isfinite(T.lat_ms)].copy()
    meas_all = (C.jaw_judged | C.has_cam1 | C.has_fr).to_numpy()
    C["move_all"] = move(C, ["jaw_ms", "first_cam1_ms", "fr_onset_ms"], meas_all)
    chk = np.nanmax(np.abs(C.move_all - C.move_ms))
    print(f"reproduces production move_ms: max |diff| = {chk:.3g} ms")
    C["move_jaw"] = move(C, ["jaw_ms"], C.jaw_judged.to_numpy())
    # which signal sets the production first movement (ties -> first in this order)
    lat = C.lat_ms.to_numpy(float)
    src_cols = {"jaw": "jaw_ms", "cam1": "first_cam1_ms", "fr": "fr_onset_ms"}
    win = np.full(len(C), "none (= lick)", object)
    for k, c in reversed(list(src_cols.items())):
        v = C[c].to_numpy(float)
        win = np.where(np.isfinite(v) & (v <= lat + 1e-6) & (np.abs(v - C.move_all) < 1e-6), k, win)
    C["move_src"] = win
    C["jaw_before_lick"] = np.isfinite(C.jaw_ms) & (C.jaw_ms <= C.lat_ms + 1e-6)
    rows = []
    for (a, p, e), g in C.groupby(["animal", "pos", "epoch"]):
        r = {"animal": a, "pos": p, "epoch": e, "n_licked": len(g),
             "lat_med_ms": np.median(g.lat_ms),
             "wait_med_ms_all": np.nanmedian(g.move_all), "revup_med_ms_all": np.nanmedian(g.lat_ms - g.move_all),
             "wait_med_ms_jaw": np.nanmedian(g.move_jaw), "revup_med_ms_jaw": np.nanmedian(g.lat_ms - g.move_jaw),
             "jaw_judged": g.jaw_judged.mean(), "jaw_before_lick": g.jaw_before_lick.mean()}
        for k in ("jaw", "cam1", "fr", "none (= lick)"):
            r[f"src_{k.split()[0]}"] = float((g.move_src == k).mean())
        rows.append(r)
    cells = pd.DataFrame(rows)
    cells.to_csv(out / "s10a_cells_jaw_vs_all.csv", index=False, float_format="%.4g")
    for tag in ("all", "jaw"):
        cells.rename(columns={f"wait_med_ms_{tag}": "wait_med_ms", f"revup_med_ms_{tag}": "revup_med_ms"})[
            ["animal", "pos", "epoch", "n_licked", "lat_med_ms", "wait_med_ms", "revup_med_ms"]].to_csv(
            out / f"s10a_revup_cells_{tag}.csv", index=False, float_format="%.4g")
    num = [c for c in cells.columns if c not in ("animal", "pos", "epoch")]
    S = cells.groupby(["pos", "epoch"])[num].mean().reset_index()
    S["pos"] = pd.Categorical(S.pos, POS, ordered=True)
    S["epoch"] = pd.Categorical(S.epoch, EP, ordered=True)
    S = S.sort_values(["pos", "epoch"])
    S.to_csv(out / "s10a_summary_mean_of_animals.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(S[["pos", "epoch", "lat_med_ms", "wait_med_ms_all", "revup_med_ms_all", "wait_med_ms_jaw",
             "revup_med_ms_jaw", "jaw_before_lick", "src_jaw", "src_cam1", "src_fr", "src_none"]].round(2).to_string())
    # overall shares by epoch (trial-pooled within animal, mean of animals)
    sh = C.groupby(["animal", "epoch"]).move_src.value_counts(normalize=True).unstack(fill_value=0)
    print(sh.groupby("epoch").mean().round(3))
    print("jaw onset before the first lick, mean of animals by epoch:")
    print(C.groupby(["animal", "epoch"]).jaw_before_lick.mean().groupby("epoch").mean().round(3))


if __name__ == "__main__":
    main(*sys.argv[1:3])
