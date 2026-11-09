"""June no-stroke drift reference for the state-space MAPS (handoff 11-09 ~12:15 open item c; Priya 11-09 "go ahead with
all"). Where on cortex does the June -> August-pre drift sit, against the August-pre -> post-stroke change?

Like-for-like trials (as `state_space_june_drift.py`): June rows are first DAQ spout CONTACTS by construction, so every
August and later session is restricted to CONTACT first licks (landing table `contact`; Figs 9.1-9.3 use contact +
incomplete). Lick-aligned 0.1-0.5 s minus the session's RESTnl level (June sessions have RESTnl: `rest_baseline_svd`),
cohort SVD basis, units = the animal's full August-pre response RMS (contact trials), versions full / nocw / spout as
Figs 9.1-9.3 (nocw: the August pre PC1 projected out of every session, June included). Columns: the June -> August-pre
DRIFT (pre - June, forward in time, ~64 days) and the August-pre -> acute / chronic change (~37-43 days to chronic).
Contours: `beta_maps.significance_contour` (nested bootstrap animals -> sessions, as the Fig 9 maps; drift: June
sessions vs August pre sessions). Per animal, over `stat_mask` pixels, mean over spouts (full, nocw) or far contra
(spout): RMS of drift and of chronic change, their cosine, and the chronic change's projection on the drift direction.
CAVEATS (bound any reading): June -> August includes learning, a room move and another rig (FINDINGS 0c / lick-code
note), so it is an UPPER bound on no-stroke drift; 3 June sessions per animal (6/6-6/8, curated); no FaceRhythm in June
(no movement-removed version).

    WF_SS_ALIGN=lick WF_SS_BASIS=svd WF_SS_MAPBASE=rest python scripts/state_space_june_maps.py
Outputs: <state_space_recovery>/june_maps{tag}/{maps.npz,animals.csv}; page_figs/ss_june_maps_<version>{tag}.png.
EXPLORATORY.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_chronic_maps as C  # noqa: E402
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import beta_maps  # noqa: E402
from wfield_local.atlas_overlay import region_edges  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402

OUT = S.ROOT / f"june_maps{S.tag()}"
COLS = [("drift", "june", "pre", "June -> Aug pre\n(no stroke, ~64 d)"), ("acute", "pre", "acute", "Aug pre -> acute"),
        ("chronic", "pre", "chronic", "Aug pre -> chronic\n(~37-43 d)")]


def responses(version):
    """{animal: {epoch incl. june: (n, 6, K)}} on contact first licks, cached (RESP_CACHE, `_june_contact`)."""
    S.XFORM = C.VERSIONS_ALL[version]
    rc = C.RESP_CACHE / f"{C.SUF}{C.BSUF}_june_contact_{version}.pkl"
    if rc.exists():
        return pickle.loads(rc.read_bytes())
    D = pd.read_csv(R.LANDING)
    L = {lab: g for lab, g in D[D.contact.notna()].groupby("label")}
    hook = lambda lab, Y, pos: (Y, pos) if S._epoch(lab, "") == "june" else R.relabel(lab, Y, pos, "contact", L)  # noqa: E731
    data, _ = S.load_sessions(hook, june=True)
    out = {}
    for an, sess in sorted(data.items()):
        d = {}
        for ep in ["june", "pre", *C.EPS]:
            rows = [M for lab, e, _day, Y, pos in sess if e == ep
                    for M in [C.session_spouts(lab, an, Y, pos)] if np.isfinite(M).any()]
            if rows:
                d[ep] = np.stack(rows)
        out[an] = d
    rc.write_bytes(pickle.dumps(out))
    return out


def main():
    if S.BASIS != "svd" or S.ALIGN != "lick" or C.BASE != "rest" or C.FR:
        sys.exit("state_space_june_maps: WF_SS_ALIGN=lick WF_SS_BASIS=svd WF_SS_MAPBASE=rest, no WF_SS_FR")
    OUT.mkdir(parents=True, exist_ok=True)
    Uc = np.load(C.BASIS_DIR / "U_cohort.npy").astype(np.float64)
    atlas = np.load(S.SVD.parents[1] / "atlas.npy")
    mask = np.load(C.BASIS_DIR / "mask.npy").reshape(atlas.shape)
    edges = region_edges(atlas)
    brain = beta_maps.allen_mask()
    statpx = beta_maps.stat_mask()[mask]
    resp = {v: responses(v) for v in C.VERSIONS_ALL}
    unit = C.full_unit(resp["full"], Uc)
    S.XFORM = ""

    def frame(v):
        img = np.full(mask.shape, np.nan, np.float32)
        img[mask] = v
        return img
    store, recs = {}, []
    for version, rs in resp.items():
        ci_main = 0 if version != "spout" else 1 + S.POS.index("far_contra")
        cmaps, conts, per = {}, {}, {}
        for key, a_ep, b_ep, _t in COLS:
            ans = [a for a in sorted(rs) if a_ep in rs[a] and b_ep in rs[a]]
            V = np.stack([C.with_mean(np.nanmean(rs[a][b_ep], 0) - np.nanmean(rs[a][a_ep], 0)) / unit[a] for a in ans])
            px = np.nan_to_num(V) @ Uc.T                                         # n_an x 7 x npix
            per[key] = dict(zip(ans, px))
            cmaps[key] = px.mean(0)
            sess = {a: {e: (C.with_mean(rs[a][e]) / unit[a]) @ Uc.T for e in (a_ep, b_ep)} for a in ans}
            conts[key] = []
            for ci in range(7):
                A = {a: [frame(m) for m in sess[a][a_ep][:, ci] if np.isfinite(m).all()] for a in ans}
                B = {a: [frame(m) for m in sess[a][b_ep][:, ci] if np.isfinite(m).all()] for a in ans}
                cm, _lab = beta_maps.significance_contour(A, B, n_boot=C.N_BOOT, normalise=False)
                conts[key].append(cm)
            store[f"{version}__{key}__mean"] = cmaps[key].astype(np.float32)
            store[f"{version}__{key}__sig"] = np.stack([np.zeros(mask.shape, bool) if c is None else c
                                                        for c in conts[key]])
        for a in sorted(per["drift"]):
            if a not in per["chronic"]:
                continue
            dj, dc = per["drift"][a][ci_main][statpx], per["chronic"][a][ci_main][statpx]
            rms = lambda v: float(np.sqrt(np.mean(v ** 2)))  # noqa: E731
            recs.append({"version": version, "column": "mean" if ci_main == 0 else "far_contra", "animal": a,
                         "rms_drift": rms(dj), "rms_chronic": rms(dc), "ratio_chronic_drift": rms(dc) / rms(dj),
                         "cos": float(dj @ dc / np.linalg.norm(dj) / np.linalg.norm(dc)),
                         "chronic_along_drift": float(dc @ dj / (dj @ dj))})
        # figure: rows = mean + six spouts, cols = drift / acute / chronic; one scale per version family
        vmax = {"full": 0.9, "nocw": 0.6, "spout": 0.43}[version]
        rws = list(range(7)) if version != "spout" else list(range(1, 7))
        fig, axs = plt.subplots(len(rws), 3, figsize=(9.6, 2.5 * len(rws)), squeeze=False)
        for i, ci in enumerate(rws):
            for j, (key, _a, _b, title) in enumerate(COLS):
                ax = axs[i, j]
                C.draw(ax, cmaps[key][ci], mask, edges, vmax, sig=conts[key][ci], brain=brain)
                if i == 0:
                    ax.set_title(title, fontsize=9.5)
                if j == 0:
                    ax.set_ylabel((["mean", *S.POS])[ci].replace("_", " "), fontsize=9)
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax))
        fig.colorbar(sm, ax=axs, fraction=0.025, label="change (own Aug-pre response RMS)")
        fig.suptitle(C._wrap(f"June no-stroke drift vs post-stroke change, {C.LABEL[version]}: contact first licks "
                             "only (June = first DAQ contacts), lick-aligned 0.1-0.5 s minus RESTnl, cohort SVD basis, "
                             "mean of 4 animals; outline = nested bootstrap (animals -> sessions) significant. June -> "
                             "August includes learning + a room / rig move: an upper bound on drift", 110), fontsize=9.5)
        fp = C.FIGS / f"ss_june_maps_{version}{S.tag()}.png"
        save_png_svg(fig, fp)
        plt.close(fig)
        print(f"wrote {fp}", flush=True)
    np.savez_compressed(OUT / "maps.npz", mask=mask, cols=np.array(["mean", *S.POS]), **store)
    T = pd.DataFrame(recs)
    T.to_csv(OUT / "animals.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(T.to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
