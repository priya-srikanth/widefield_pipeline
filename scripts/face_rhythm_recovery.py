"""Which factors DISAPPEAR after the stroke? (Priya 11-07: "it's actually interesting to note the factors that
'disappear' post stroke" -- the poor-fit cells the v2 matching rule leaves out of the shape readouts, read on their own.)

A reference unit (master factor, or merged twin unit under v2, `face_rhythm_epoch_shape.MATCH_RULES`) is RECOVERED by a
fit when the column picked for it is like it in place AND rhythm (`face_rhythm_epoch_shape.pick`). Per post epoch:
  observed   lost = 1 when COHORT_<epoch> does not recover the unit while COHORT_pre does (0 otherwise), and the drop in
             the picked column's spectral cosine, pre - post
  null       the same two numbers on every permutation's pseudo-pre / pseudo-post fits (`own_epoch_cohort_null/`, the
             fits the cohort permutation null scores): sessions shuffled between pre and that epoch within animal
  p          (1 + #null >= observed) / (1 + n); Holm over units per epoch. Lost is binary, so its p is the share of
             permutations that lose the unit post while keeping it pre (+1); the spectral drop is the graded version.
Per-animal fits (`own_epoch/`) are listed alongside: in how many of 4 animals the epoch fit loses the unit.

    python scripts/face_rhythm_recovery.py [--set master_pre_v1] [--match-rule v2]
-> ``<readouts>/<set>/epoch_shape_cohort_<rule>/``: recovery.png, data/recovery.csv.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import face_rhythm_epoch_shape as es  # noqa: E402

from wfield_local.face_rhythm import common as frc  # noqa: E402


def quality(f: Path, A_ref, B_ref) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(recovered, spatial cos, spectral cos) of each reference unit's picked column in one fit."""
    with np.load(f, allow_pickle=True) as z:
        A, B = z["A_own"].astype(float), z["B_own"].astype(float)
    m, cols, ok = es.pick(A_ref, B_ref, A, B)
    k = np.arange(len(cols))
    return ok, m["cs_all"][k, cols], m["cf_all"][k, cols]


def compute(fdir: Path) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    _M, A_ref, B_ref, names = es.reference(fdir)
    q = {g: quality(fdir / "own_epoch_cohort" / f"COHORT_{g}.npz", A_ref, B_ref) for g in ("pre", *frc.POST_EPOCHS)}
    nulls = {}
    for f in sorted((fdir / "own_epoch_cohort_null").glob("*_pre.npz")):
        fpost = f.with_name(f.name.replace("_pre.npz", "_post.npz"))
        if fpost.exists():
            ep = re.match(r"(\w+?)_p\d+_pre", f.stem).group(1)
            nulls.setdefault(ep, []).append((quality(f, A_ref, B_ref), quality(fpost, A_ref, B_ref)))
    own = {}
    for f in sorted((fdir / "own_epoch").glob("*.npz")):
        an, g = f.stem.split("_", 1)
        own[(an, g)] = quality(f, A_ref, B_ref)[0]
    animals = sorted({an for an, _ in own})
    rows = []
    for ep in frc.POST_EPOCHS:
        okp, _, cfp = q["pre"]
        oke, cse, cfe = q[ep]
        lost = (okp & ~oke).astype(float)
        drop = cfp - cfe
        N = nulls.get(ep, [])
        Nl = np.array([(a[0] & ~b[0]).astype(float) for a, b in N])          # (n, R)
        Nd = np.array([a[2] - b[2] for a, b in N])
        n = len(N)
        p_lost = np.where(lost > 0, (1 + (Nl >= 1).sum(0)) / (1 + n), np.nan) if n else np.full(len(names), np.nan)
        p_drop = (1 + (Nd >= drop[None, :]).sum(0)) / (1 + n) if n else np.full(len(names), np.nan)
        ph_drop = ak.holm(p_drop) if n else p_drop
        for k, nm in enumerate(names):
            own_lost = [an for an in animals if (an, ep) in own and (an, "pre") in own
                        and own[(an, "pre")][k] and not own[(an, ep)][k]]
            rows.append(dict(epoch=ep, unit=nm, recovered_pre=bool(okp[k]), recovered=bool(oke[k]), lost=bool(lost[k]),
                             cos_spatial=cse[k], cos_spectral=cfe[k], cos_spectral_pre=cfp[k], spectral_drop=drop[k],
                             n_perm=n, null_lost_share=float(Nl[:, k].mean()) if n else np.nan, p_lost=p_lost[k],
                             p_drop=p_drop[k], p_drop_holm=ph_drop[k],
                             animals_lost=f"{len(own_lost)}/{len(animals)}", animals_lost_ids=",".join(own_lost)))
    T = pd.DataFrame(rows)
    T["p_lost_holm"] = np.nan
    for _ep, d in T.groupby("epoch"):
        idx = d.index[d.lost]
        if len(idx):
            T.loc[idx, "p_lost_holm"] = ak.holm(T.loc[idx, "p_lost"].to_numpy())
    return T, names


def figure(T: pd.DataFrame, names: list[str], out: Path, set_name: str) -> None:
    import matplotlib.pyplot as plt

    from wfield_local import face_rhythm_figures as frf
    eps = list(frc.POST_EPOCHS)
    C = T.pivot(index="unit", columns="epoch", values="cos_spectral").reindex(index=names, columns=eps)
    with frf.style():
        fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(names) + 1.6))
        im = ax.imshow(C.to_numpy(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i, nm in enumerate(names):
            for j, ep in enumerate(eps):
                r = T[(T.unit == nm) & (T.epoch == ep)].iloc[0]
                txt = f"{r.cos_spectral:.2f}"
                if r.lost:
                    txt = f"LOST {txt}" + ("*" if r.p_lost_holm < .05 else "")
                txt += f"\n{r.animals_lost} animals"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                        color="w" if r.cos_spectral < .55 else "k", fontweight="bold" if r.lost else "normal")
        ax.set_xticks(range(len(eps)), eps)
        ax.set_yticks(range(len(names)), names)
        ax.spines[["top", "right"]].set_visible(False)
        plt.colorbar(im, ax=ax, fraction=.05, label="spectral cos of the matched factor (COHORT fit)")
        ax.set_title(f"{set_name}: factors lost after the stroke ({T.n_perm.max()} / {T.n_perm.min()} permutations)",
                     fontsize=9)
        frf.save(fig, out / "recovery.png", f"""
Which reference units (master factors; twin units merged under the {es.RULE} matching rule) the pooled all-animals
COHORT fit of each post epoch does NOT recover while the pre fit does. Colour and number = spectral cosine between the
unit and the factor picked for it (place match with a frequency tie-break); LOST = picked factor below spatial cos .8 or
spectral cos .5 in that epoch's fit but not in the pre fit; * = Holm < .05 over the lost units of that epoch, against the
permutation null (sessions shuffled between pre and that epoch within animal; p = share of permutations that lose the
unit post while keeping it pre). Second line: in how many of the 4 per-animal fits the epoch loses the unit.""")


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--match-rule", choices=sorted(es.MATCH_RULES), default="v2")
    a = ap.parse_args(argv)
    es.RULE = a.match_rule
    fdir = a.fdir or frc.default_set_dir(set_name=a.set)
    out = frc.readouts_dir(a.set) / ("epoch_shape_cohort" + es.rule_suffix())
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    T, names = compute(fdir)
    T.to_csv(figure_layout.sidecar_for(out, "recovery", ".csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(T[["epoch", "unit", "recovered_pre", "recovered", "lost", "cos_spatial", "cos_spectral", "spectral_drop",
             "n_perm", "null_lost_share", "p_lost", "p_lost_holm", "p_drop", "p_drop_holm", "animals_lost",
             "animals_lost_ids"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    figure(T, names, out, fdir.name)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
