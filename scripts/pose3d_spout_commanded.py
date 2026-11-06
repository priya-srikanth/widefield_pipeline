"""Per session x spout position, the COMMANDED 3-D spout tip (mouth-zero mm) in the format of `prep_3d_inputs`'
``pose3d_spout_positions_<tag>.csv`` -- the encoder's TARGET3D table (`combined_encoding._target_3d`).

    python -m scripts.pose3d_spout_commanded <labels.txt> <out.csv>          # labels: PS92_0806 ... (MMDD, 2026)

WHY (2026-11-04, e0 + 34; Priya approved the spout fix). The measured table took the median of `pose3d.session_3d`'s
per-frame spout, which until bddf02e7 triangulated cam1's inferior and cam4's superior rod edge as ONE point: a blend
1.02 mm (range 0.91-1.07, all 126 production sessions) behind the commanded tip in ap and 0.14 mm low. With the fix the
measured spout lands on the commanded tip to the PnP residual (PS93 0814: 0.00 / 0.01 / -0.01 mm), so the commanded
positions ARE the corrected measurement, consistent with lick aim and kinematics (which use the commanded targets), and
need no 2.5 h production re-run. Source: each session's own gui_config (`apparatus_mouth._rig` ->
`spout_world.session_rig`), `spout_world.commanded_world`, world axes (ml + = mouse right, ap + = out, dv + = up). Stage
mm, so no June camera scaling applies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def build(labels: list[str]) -> pd.DataFrame:
    from scripts.apparatus_mouth import _rig
    from wfield_local import spout_world as SW
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    rows = []
    for lab in labels:
        animal, mmdd = lab.split("_")
        date = mmdd if len(mmdd) == 8 else f"2026{mmdd}"
        geo = _rig(rv, animal, date)["geometry"]
        for p in SW.POSITIONS:
            ml, ap, dv = SW.commanded_world(geo, p)
            rows.append({"session": f"{animal}_{date}", "position": p, "spout_ml": ml, "spout_ap": ap, "spout_dv": dv})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    labels = [s.strip() for s in Path(argv[0]).read_text().split() if s.strip()]
    out = Path(argv[1])
    if out.exists():
        raise SystemExit(f"{out} exists -- write a new tag, never overwrite")
    D = build(labels)
    D.to_csv(out, index=False, float_format="%.4f")
    print(f"{D.session.nunique()} sessions x {D.position.nunique()} positions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
