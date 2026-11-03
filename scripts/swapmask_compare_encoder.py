"""Cohort-1 swap-mask re-run, family (b): frozen pre-stroke encoder headline (final frame, no FaceRhythm,
`frozen_encoder_gate-attempted_v7g3_t3d`), adopted vs swap-mask variant.

Headline numbers (goal page section 0, Figs 0.1 / 0.2 grey = no-FR frame):
  1. frozen R2 (frame measure, held-out, joint model) per area x window: epoch - pre, per animal first, then the nested
     bootstrap (`frozen_fr_compare.boot`, = the page's lines);
  2. Musall UNIQUE R2 of target and spout = R2(joint) - R2(no_target / no_spout), epoch - pre; spout contra - ipsi.
Both runs are read from their per-session JSONs with the same code, so any difference is the hemo variant (the
variant run used the adopted run's commit 9f458196). Read-only on the adopted folder.

    python scripts/swapmask_compare_encoder.py --new <root>/encoding [--measure frame]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_fr_compare as ffc  # noqa: E402

FOLDER = "frozen_encoder_gate-attempted_v7g3_t3d"
AREAS = ["cortex", "SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left"]


def unique(root, measure):
    ffc.E = root
    J = ffc.session_r2(FOLDER, "joint", measure)
    out = []
    for g in ("target", "spout"):
        D = ffc.session_r2(FOLDER, f"no_{g}", measure)
        m = J.merge(D, on=["label", "animal", "epoch", "window", "area"], suffixes=("", "_drop"))
        m["r2"] = m.r2 - m.r2_drop
        m["group"] = f"unique:{g}"
        out.append(m.drop(columns="r2_drop"))
    J["group"] = "joint R2"
    return pd.concat([J, *out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--measure", default="frame")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.new / "compare_adopted"
    out.mkdir(parents=True, exist_ok=True)
    adopted_root = ffc.E
    T = {"adopted": unique(adopted_root, a.measure), "swapmask": unique(a.new, a.measure)}
    for k, v in T.items():
        print(f"{k}: {v.label.nunique()} sessions")
    rows, brows = [], []
    for ver, D in T.items():
        D = D[D.area.isin(AREAS)]
        pa = D.groupby(["group", "window", "area", "animal", "epoch"]).r2.mean().unstack("epoch")[ffc.EP4]
        for e in ffc.EPS:
            pa[f"{e}-pre"] = pa[e] - pa["pre"]
        pa["version"] = ver
        rows.append(pa.reset_index())
        for g, t in D.groupby("group"):
            B = ffc.boot(t, "r2")
            B["group"], B["version"] = g, ver
            brows.append(B)
        # spout unique contra - ipsi (SSp-m right - left), per session
        s = D[(D.group == "unique:spout") & D.area.isin(["SSp-m_right", "SSp-m_left"])]
        s = s.pivot_table(index=["label", "animal", "epoch", "window"], columns="area", values="r2").reset_index()
        s["r2"] = s["SSp-m_right"] - s["SSp-m_left"]
        s["area"] = "SSp-m contra-ipsi"
        B = ffc.boot(s, "r2")
        B["group"], B["version"] = "unique:spout", ver
        brows.append(B)
    PA = pd.concat(rows)
    PA.to_csv(out / "encoder_per_animal.csv", index=False)
    BB = pd.concat(brows)
    BB.to_csv(out / "encoder_pooled_boot.csv", index=False)
    pd.set_option("display.width", 250)
    for g in ("joint R2", "unique:target", "unique:spout"):
        for w in ffc.WINDOWS:
            t = PA[(PA.group == g) & (PA.window == w) & PA.area.isin(["cortex", "SSp-m_right", "SSp-m_left"])]
            t = t.set_index(["area", "animal", "version"])[["pre", "acute-pre", "subacute-pre", "chronic-pre"]]
            print(f"\n=== PER ANIMAL {g}, {w} window ({a.measure}) ===\n{t.sort_index().round(4).to_string()}")
    show = BB[~BB.epoch.astype(str).str.endswith("_split")]
    show = show[show.area.isin(["cortex", "SSp-m_right", "SSp-m_left", "SSp-m contra-ipsi"])]
    print("\n=== POOLED (mean of animals' epoch - pre, nested 95% CI) ===")
    print(show[["group", "window", "area", "epoch", "version", "point", "lo", "hi", "n_up", "n"]]
          .sort_values(["group", "window", "area", "epoch", "version"]).round(4).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
