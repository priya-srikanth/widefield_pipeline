"""Re-export `recov/balance_aim_sessions.csv` and `recov/balance_aim_models.csv` (Figure 5F). No saved script wrote
the originals (staged 9 Nov 14:34). They were a selection of `scripts/lick_recovery_primary.py`'s outputs:
`recprim_sessions.csv` restricted to panel `cued first lick -> aim` (that file is the CSV twin of figdata object `T`),
and `recprim_models.csv` (object `M`), whose 9 rows are the three panels that script had on 9 Nov. This does the same
selection and refits the models with the pipeline's own `lick_recovery_primary.fit_panels`.

FAILED FITS ARE KEPT AS FAILURES. On 9 Nov (`ef7404f0`) `fit_panels` tried one default fit per model and recorded an
exception as `"<model> FAILED <error>"` with empty estimates; that is the staged ITI row
`concurrent, per-animal spline FAILED Singular matrix`. Since 17 Nov `fit_panels` retries with other optimizers and
notes it in `fit_note`. Any row with a `fit_note` is written back in the 9 Nov form (model + " FAILED " + the default
chain's error, estimates empty), so a retry never silently repairs a recorded failure. Tested 17 Nov: a stubbed
retried row comes out character-identical to the staged `ITI first after quiet -> iti_ang,"concurrent, per-animal
spline FAILED Singular matrix",76,,,,`, and a fit that fails in every optimizer gives the same form.

VERIFIED 2026-11-17 against the archived staged copies
(`MICROSCOPE/Priya/Widefield/grant_figures_archive/k08_2026/staged_inputs/recov/`):

  sessions, from the CURRENT server `recprim_sessions.csv` (11 Nov 05:38, `<run>/imaging_encoding/iti_licks/<daqlk>/`):
    107 / 107 rows, same order; label, animal, epoch, day, y and n_licks bit-identical in 107 / 107;
    marker in 0 / 107 (max |d| 0.092, median 0.010, r 0.998; largest in acute sessions).
  models, cued panel, refit by `fit_panels` FROM THE STAGED SESSIONS: 3 / 3 rows reproduce to < 1e-12 relative
    (n 76 / 76 / 72; log(day) 13.597 [6.178, 21.015] p 3.28e-4; spline 17.548 p 5.09e-5; predictive 0.506 p 0.907),
    per-animal r 0.69 / 0.89 / 0.93 / 0.75.
  models, six ITI rows: NOT reproduced. Their sessions were never staged, and the 9 Nov `recprim_sessions.csv` /
    figdata pkl they were fitted on no longer exist (rewritten 11 Nov 05:38; the archive holds no pkl). On today's
    server table they come out different (log(day) iti_ang 5.63 vs 5.73 staged), and the spline that failed on
    9 Nov now converges with the default fit.

  WHY THE MARKER CANNOT BE RECOVERED: `marker` is computed from `balsub_sessions.csv`, which was rewritten 11 Nov 05:31,
  after the staging. Every `balsub_sessions*.csv` on the share (rebuild, pre-rebuild, `_superseded_glue_20261109`) was
  run through `marker()`; none reproduces the staged column. The 9 Nov marker survives only in the staged file.
  Figure 5F does not plot it: `render_fig4_rows.py row_FGH` reads only label / animal / epoch / y from the sessions
  file (x = SSp-m lateral ratio from `data/lick_lateral_ratio_auc_docked_sessions.csv`) and never reads the models
  file. The plotted columns are bit-identical, so the panel itself is reproducible from the current server file.

    PYTHONPATH=<repo>:<repo>/scripts python export_balance_aim.py [recprim_sessions.csv] [-o DIR] [--verify STAGED_RECOV_DIR]

Default input: the routed `iti_licks.OUT_W / recprim_sessions.csv`, where `lick_recovery_primary.compute` writes it.
Run from the repo root (it imports `scripts/` modules, as `export_paths3d_front_sig.py` does).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import lick_recovery_primary as L  # noqa: E402

PANEL = "cued first lick -> aim"
#: the panels lick_recovery_primary had on 9 Nov, in its compute order (= the 9 rows of the staged models file)
PANELS_9NOV = ["cued first lick -> aim", "ITI first after quiet -> iti_ang", "ITI first after quiet -> aim"]
SESSION_COLS = ["label", "animal", "epoch", "day", "y", "marker", "n_licks"]
MODEL_COLS = ["panel", "model", "n", "slope", "lo", "hi", "p"]
EST = ["slope", "lo", "hi", "p"]


def models(T: pd.DataFrame) -> pd.DataFrame:
    """fit_panels on a session table with columns panel, set + SESSION_COLS; retried fits reverted to 9 Nov FAILED rows."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M, _, ANr, _ = L.fit_panels(T)
    for pn, g in ANr.groupby("panel", sort=False):
        print(f"per-animal r, {pn}:", {r.animal: round(r.r, 2) for r in g.itertuples()})
    M = M.copy()
    for c in ("fit_note", *EST):
        if c not in M:
            M[c] = np.nan if c in EST else ""
    retried = M.fit_note.fillna("").astype(str).str.len() > 0
    for i in M.index[retried]:
        err = M.at[i, "fit_note"].split("raised: ", 1)[-1].rstrip(")")
        print(f"  reverting retried fit to its 9 Nov FAILED form: {M.at[i, 'panel']} | {M.at[i, 'model']} | {err}")
        M.loc[i, "model"] = f"{M.at[i, 'model']} FAILED {err}"
        M.loc[i, EST] = np.nan
    return M[MODEL_COLS]


def with_panel(S: pd.DataFrame, panel: str) -> pd.DataFrame:
    return S.assign(panel=panel, set=panel.split(" -> ")[0])


def verify(S: pd.DataFrame, staged: Path) -> None:
    St = pd.read_csv(staged / "balance_aim_sessions.csv")
    print(f"sessions: staged {len(St)} rows, re-export {len(S)}; same order: "
          f"{len(St) == len(S) and bool((St.label.values == S.label.values).all())}")
    m = St.merge(S, on="label", suffixes=("_s", "_r"))
    for c in SESSION_COLS[1:]:
        a, b = m[f"{c}_s"], m[f"{c}_r"]
        if a.dtype.kind in "fi":
            ok = (a == b) | (a.isna() & b.isna())
            print(f"  {c:8s} {int(ok.sum())} / {len(m)} bit-identical, max |d| {np.nanmax(np.abs(a - b)):.3g}")
        else:
            print(f"  {c:8s} {int((a == b).sum())} / {len(m)} identical")
    # models: the cued panel refit from the STAGED sessions; the ITI rows have no staged sessions to refit from
    Ms = pd.read_csv(staged / "balance_aim_models.csv")
    j = Ms.merge(models(with_panel(St, PANEL)), on=["panel", "model"], suffixes=("_s", "_r"))
    rel = {c: float(np.max(np.abs(j[f"{c}_s"] - j[f"{c}_r"]) / np.abs(j[f"{c}_s"]))) for c in EST}
    print(f"models, cued rows refit from the staged sessions: {len(j)} / {int((Ms.panel == PANEL).sum())} matched, "
          f"n equal {bool((j.n_s == j.n_r).all())}, max relative diff {rel}")
    print(f"models, ITI rows: {int((Ms.panel != PANEL).sum())} staged, not verifiable (no staged ITI sessions)")


def main(argv=None) -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("recprim", nargs="?", help="recprim_sessions.csv; default = iti_licks.OUT_W / recprim_sessions.csv")
    a.add_argument("-o", "--out", default=".", help="directory for balance_aim_{sessions,models}.csv")
    a.add_argument("--verify", metavar="DIR", help="staged recov/ directory to compare against")
    o = a.parse_args(argv)

    if o.recprim:
        src = Path(o.recprim)
    else:
        import iti_licks as il
        src = il.OUT_W / "recprim_sessions.csv"
    T = pd.read_csv(src)
    S = T[T.panel == PANEL][SESSION_COLS].reset_index(drop=True)
    print(f"{src}: {len(S)} sessions, {int((S.epoch != 'pre').sum())} post")
    missing = [p for p in PANELS_9NOV if p not in set(T.panel)]
    if missing:
        print(f"  panels absent from the input, so their model rows are not written: {missing}")

    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "balance_aim_sessions.csv", index=False)
    T9 = T[T.panel.isin(PANELS_9NOV)]
    T9 = T9.assign(_o=T9.panel.map(PANELS_9NOV.index)).sort_values("_o", kind="stable").drop(columns="_o")
    models(T9).to_csv(out / "balance_aim_models.csv", index=False)
    print(f"wrote {out / 'balance_aim_sessions.csv'} and {out / 'balance_aim_models.csv'}")
    if o.verify:
        verify(S, Path(o.verify))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
