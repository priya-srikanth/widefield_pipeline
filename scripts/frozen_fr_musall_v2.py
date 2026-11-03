"""Musall-style UNIQUE R2 in the FROZEN frame -- v2 with more AREAS (Priya 10-30: add SSp-n, ALM, tjM1 to Figs 0.2 / 0.8).
Same quantities, frames, statistic and bootstrap as `scripts/frozen_fr_musall.py` (its musall_{sessions,change}.csv are
NOT touched); this writes musall_{sessions,change}_v2.csv beside them.

NO REFIT -- a re-SUMMARY of saved per-session outputs:
  * Allen areas (cortex, SSp-m, SSp-n, MOp, MOs, SSs, VISp, RSPd): the AREA run's per-session `r2_musall[window]["frame"]`
    arrays (session json), which already hold every Allen area on the stat mask; v1 only kept cortex / SSp-m / MOp.
    VISp has no LEFT (ipsilesional) area on the stat mask -> VISp_right only.
  * ALM / tjM1 (ROI v8 on the stat mask): the frame's FROZEN_PIX_MUSALL basis twin (`<run>_basis_pixmusall`), ROI R2 =
    1 - w'Ew / w'Sw (`encoder_split_rows.musall_rows`), exactly as `frozen_claims.py` builds the ALM / tjM1 rows. That
    twin saved moments only for joint / no_target / no_spout / no_movement / no_exec_* (/ no_fr_all in FR-A), so ALM /
    tjM1 have q = target, spout, movement (and fr in FR-A); task and fr_dir are area-run only.
  * BASIS CHECK: SSp-m / SSp-n / MOp are ALSO computed by the basis route and compared with the area route (printed and
    saved as musall_v2_basis_check.csv), so the ALM / tjM1 rows can be read against the same areas on the same route.
unique R2 of group g = R2[joint] - R2[no_g] per session x window x area (frame R2 over the window's frames).
Areas: *_right = contralesional (lesion left), *_c-i = contra - ipsi.
Usage: python scripts/frozen_fr_musall_v2.py
Output: <E>/frozen_fr_compare/musall_sessions_v2.csv, musall_change_v2.csv, musall_v2_basis_check.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402
from frozen_fr_compare import boot, session_r2  # noqa: E402

RUNS = {"noFR": "frozen_encoder_gate-attempted_v7g3_t3d", "FR-A": "frozen_encoder_gate-attempted_v7g3_t3d_frA"}
Q = [("target", "no_target"), ("spout", "no_spout"), ("task", "no_task"), ("movement", "no_movement"),
     ("fr", "no_fr_all"), ("fr_dir", "no_fr_dir")]
BILAT = ["SSp-m", "SSp-n", "MOp", "MOs", "SSs", "RSPd"]
AREA_RUN = ["cortex", "VISp_right"] + [f"{b}_{h}" for b in BILAT for h in ("right", "left")]
SPLIT = ("ALM_right", "ALM_left", "tjM1_right", "tjM1_left")
CHECK = ("SSp-m_right", "SSp-m_left", "SSp-n_right", "SSp-n_left", "MOp_right", "MOp_left")
CI = BILAT + ["ALM", "tjM1"]
KEY = ["label", "animal", "epoch", "window", "area"]


def basis_r2(folder: str, variant: str, areas) -> pd.DataFrame:
    import encoder_split_rows as esr
    pix = E / f"{folder}_basis_pixmusall"
    if not (pix / variant / "sessions").is_dir():
        return pd.DataFrame(columns=KEY + ["r2"])
    return esr.musall_rows(pix, variant, "frame", tuple(areas), fit_run=E / f"{folder}_basis")


def main() -> int:
    out = E / "frozen_fr_compare"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    rows, chk = [], []
    for fr, folder in RUNS.items():
        J = session_r2(folder, "joint", areas=AREA_RUN).set_index(KEY).r2
        Jb = basis_r2(folder, "joint", SPLIT + CHECK).set_index(KEY).r2
        for q, v in Q:
            if (E / folder / v / "sessions").is_dir():
                u = (J - session_r2(folder, v, areas=AREA_RUN).set_index(KEY).r2).rename("u")
                rows.append(u.reset_index().assign(frame=fr, q=q))
            b = basis_r2(folder, v, SPLIT + CHECK)
            if len(b):
                ub = (Jb - b.set_index(KEY).r2).rename("u").reset_index()
                rows.append(ub[ub.area.isin(SPLIT)].assign(frame=fr, q=q))
                c = ub[ub.area.isin(CHECK)].merge(u.reset_index(), on=KEY, suffixes=("_basis", "_area"))
                chk.append(c.assign(frame=fr, q=q))
    U = pd.concat(rows, ignore_index=True)
    w_ = U.pivot_table(index=["frame", "q", "label", "animal", "epoch", "window"], columns="area", values="u")
    for b in CI:
        if f"{b}_right" in w_ and f"{b}_left" in w_:
            w_[f"{b}_c-i"] = w_[f"{b}_right"] - w_[f"{b}_left"]
    U = w_.reset_index().melt(id_vars=["frame", "q", "label", "animal", "epoch", "window"], var_name="area",
                              value_name="u").dropna(subset=["u"])
    U = U.sort_values(["frame", "q", "window", "area", "label"]).reset_index(drop=True)
    U.to_csv(out / "musall_sessions_v2.csv", index=False)

    C = pd.concat(chk, ignore_index=True)
    C.to_csv(out / "musall_v2_basis_check.csv", index=False)
    cs = C.groupby(["frame", "q", "area"]).apply(lambda t: pd.Series({
        "r": np.corrcoef(t.u_basis, t.u_area)[0, 1], "max_abs_diff": (t.u_basis - t.u_area).abs().max(),
        "median_abs_diff": (t.u_basis - t.u_area).abs().median(), "area_sd": t.u_area.std()}))
    print("== basis route vs area route (unique R2 per session x window): the ALM / tjM1 route checked on its parents\n"
          + cs.round(4).to_string())

    key = ["SSp-n_right", "SSp-n_c-i", "ALM_right", "ALM_c-i", "tjM1_right", "tjM1_c-i"]
    per = U[U.area.isin(key) & U.q.isin(["target", "spout"])].groupby(
        ["q", "frame", "window", "area", "animal", "epoch"]).u.mean().unstack("epoch")
    print("\n== per animal (rule 8): unique R2, new areas (contra and contra - ipsi)\n"
          + per.reindex(columns=["pre", "acute", "subacute", "chronic"]).round(4).to_string())
    B = pd.concat([boot(U[(U.frame == fr) & (U.q == q)], "u").assign(frame=fr, q=q) for fr in RUNS for q, _v in Q
                   if ((U.frame == fr) & (U.q == q)).any()], ignore_index=True)
    B.to_csv(out / "musall_change_v2.csv", index=False)
    print("\n== epoch - pre (nested bootstrap), new areas, target / spout\n" + B[B.area.isin(key) & B.q.isin(
        ["target", "spout"]) & (B.epoch != "pre_split")][["q", "frame", "window", "area", "epoch", "ref", "point", "lo",
                                                          "hi", "n_up", "n"]].round(4).to_string(index=False))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
