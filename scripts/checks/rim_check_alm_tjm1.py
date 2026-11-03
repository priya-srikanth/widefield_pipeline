"""Rim check (Priya 10-28 "did the expanded rim make a difference?"): ALM / tjM1 / rest rows of the FaceRhythm direction
figures (22.3 / 22.4, fr_direction) on the encoder STAT mask (published) vs the full v8 ROI incl. the thin lateral rim +
ALM front line. Read-only; writes only to the scratchpad."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\SabatiniLab\Github\wf_attempt\scripts")
import encoder_split_rows as esr  # noqa: E402
import fr_direction as fd  # noqa: E402
import iti_licks as il  # noqa: E402

OUT = Path(sys.argv[1])
WIN = ["cue", "lick"]


def full_masks() -> dict:
    flat, parea, names, shape, pstat = esr.grid()
    roi = il.roi_masks(parea, pstat, names, flat, shape)
    code = {v: k for k, v in names.items()}
    out = {}
    for src, dst in il.SPLIT_PAIRS:
        for h in il.HEMIS:
            par = parea == code[f"{src}_{h}"]                     # parent on ALL brain pixels (no stat mask)
            sub = roi[f"{dst}_{h}"] & par
            out[f"{src}_{h}"], out[f"{dst}_{h}"], out[f"{src}_rest_{h}"] = par, sub, par & ~sub
    return out


def main():
    from wfield_local import config, epochs
    labs = [f.stem.replace("_regions", "") for f in sorted(fd.FRD["A"].glob("*_regions.csv"))]
    labs = [x for x in labs if x[-4:] >= "0801" and all((fd.FRD[m] / f"{x}_regions.csv").exists() for m in fd.MODES)]
    ids = np.load(fd.FRD["A"] / f"{labs[0]}.npz")["flat"]
    Wstat = esr.block_weights(ids, fd.SPLIT)
    m_stat = esr.masks()
    m_full = full_masks()
    print("pixels stat vs full (rim share):")
    for a in fd.SPLIT:
        print(f"  {a:16s} {int(m_stat[a].sum()):6d} {int(m_full[a].sum()):6d}  +{m_full[a].sum() / m_stat[a].sum() - 1:.0%}")
    esr.masks = lambda: m_full                                   # block_weights reads masks()
    Wfull = esr.block_weights(ids, fd.SPLIT)
    rows = []
    for lab in labs:
        base = dict(label=lab, animal=config.animal_of(lab), epoch=epochs.epoch_of(lab))
        for tag, Wb in (("stat", Wstat), ("full", Wfull)):
            d = fd.split_r2(lab, WIN, Wb)
            P = d.pivot_table(index=["area", "window"], columns=["mode", "model"], values="r2")
            for (a, w), x in P.iterrows():
                r = dict(base, mask=tag, area=a, window=w)
                for m in fd.MODES:
                    r[f"gain_{m}"] = x[(m, "fr:full")] - x[(m, "full")]
                    r[f"pos_{m}"] = x[(m, "fr:full")] - x[(m, "fr:drop:position")]
                for q in ("gain", "pos"):
                    r[f"{q}_side"] = r[f"{q}_S"] - r[f"{q}_B"]
                    r[f"{q}_both"] = r[f"{q}_A"] - r[f"{q}_B"]
                rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rim_check_sessions.csv", index=False)
    vals = ["gain_side", "pos_side", "gain_both", "pos_both", "gain_A", "pos_A"]
    # per animal x epoch means, then epoch change vs pre (mean of animals), stat vs full
    A = R.groupby(["mask", "area", "window", "animal", "epoch"])[vals].mean()
    out = []
    for (mk, a, w), g in A.groupby(level=[0, 1, 2]):
        g = g.droplevel([0, 1, 2])
        for v in vals:
            pre = g[v].xs("pre", level="epoch")
            for ep in fd.EPS:
                if ep in g.index.get_level_values("epoch"):
                    dlt = (g[v].xs(ep, level="epoch") - pre).dropna()
                    out.append(dict(mask=mk, area=a, window=w, value=v, epoch=ep, change=dlt.mean(),
                                    n_pos=int((dlt > 0).sum()), n=len(dlt)))
    C = pd.DataFrame(out).pivot_table(index=["area", "window", "value", "epoch"], columns="mask",
                                      values=["change", "n_pos"])
    C.columns = [f"{a}_{b}" for a, b in C.columns]
    C["diff"] = C.change_full - C.change_stat
    C.to_csv(OUT / "rim_check_changes.csv")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    print("\nsession-level r(stat, full) per area x value:")
    S = R.pivot_table(index=["label", "area", "window"], columns="mask", values=vals)
    for v in vals:
        print(f"  {v}: r = {np.corrcoef(S[(v, 'stat')], S[(v, 'full')])[0, 1]:.4f}, max |diff| "
              f"{(S[(v, 'stat')] - S[(v, 'full')]).abs().max():.5f}, median |value| {S[(v, 'stat')].abs().median():.5f}")
    big = C[C.columns[C.columns.str.startswith(("change", "n_pos", "diff"))]]
    print("\nepoch change vs pre (mean of animals), ALM / tjM1 rows only:")
    print(big[big.index.get_level_values("area").str.match("ALM|tjM1")].round(5).to_string())


if __name__ == "__main__":
    main()
