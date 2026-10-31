"""Per-spout cue- and first-lick-aligned maps, cohort2, every session that has its DAQ .h5 on the share (2026-10-31).

Priya 10-31: "how do spout position cue / lick aligned maps look across animals and sessions". Built on the
re-run sessions (pre-SVD repair + clip mask; adopted variant hemo_meegkit_hpfit), native pixels, dark-relative
brain mask minus clipped pixels, DAQ events on the frame clock (`_frames`).

  cue map   = mean over [cue, cue + 1 s]          minus [cue - 1 s, cue]
  lick map  = mean over [first lick, + 0.5 s]     minus the same PRE-CUE baseline [cue - 1 s, cue]
              (first DAQ lick within the 3.5-s response window; a pre-lick baseline would contain the
               cue response and anticipatory licking)
Trials dropped where the window leaves the recording, and in PS122 10-29 after 110 min (epileptiform discharges,
QC doc §0o). Columns in canonical order close L / C / R, far L / C / R, then the all-spout mean and the
SPOUT-SPECIFIC part (each spout minus the all-spout mean, shown as the RMS over spouts). One colour scale per
SESSION (shared by its cue and lick rows): GCaMP6s and 8s amplitudes differ ~3x, so a cohort scale would blank
the RiboL1 rows. Sessions without a DAQ .h5 on the share are listed and skipped.

    conda activate locanmf
    python scripts/cohort2_qc/spout_maps.py 20261029 [20261030]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402

LABCAMS = Path("M:/MICROSCOPE/Priya/Widefield/labcams")
DAQ = Path("M:/MICROSCOPE/Priya/Widefield/DAQ_recorder_output")
OUT = LABCAMS / "cohort2" / "_tmp_qc_figures_20261031"
FS = 31.23
VAR = "hemo_meegkit_hpfit"
CODE = {0: "close C", 1: "close L", 2: "close R", 3: "far C", 4: "far L", 5: "far R"}
ORDER = [1, 0, 2, 4, 3, 5]
GENO = {"PS118": "Thy1-6s", "PS120": "Thy1-6s, no signal", "PS121": "RiboL1-8s", "PS122": "RiboL1-8s"}
EXCLUDE_AFTER_MIN = {("PS122", "20261029"): 110.0}
RESP_S = 3.5


def one(item):
    sdir, h5 = item
    sdir = Path(sdir)
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    animal, date = sdir.name.split("_")[:2]
    mc = sdir / "motion_corrected"
    res = mc / "wfield_local_results"
    s = dict(label=f"{animal}_{date[4:]}", mc=str(mc), h5=str(h5), regime="B", fmdir=None)
    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, lf, _ = _frames(s, cue, lk)
    cf, lf = np.asarray(cf), np.sort(np.asarray(lf)[np.asarray(lf) >= 0])
    st = np.asarray(cue["strobe_samples"])
    pos = np.asarray(cue["strobe_codes"])[np.clip(np.searchsorted(st, cue["cue_samples"]) - 1, 0, None)]
    C = np.load(res / VAR / "SVTcorr.npy", mmap_mode="r")
    n = C.shape[1]
    k = np.searchsorted(lf, cf)
    fl = np.where(k < len(lf), lf[np.minimum(k, len(lf) - 1)], -1)
    fl = np.where((fl >= cf) & (fl - cf <= int(RESP_S * FS)), fl, -1)
    ok = (cf > 40) & (cf < n - 40)
    cut = EXCLUDE_AFTER_MIN.get((animal, date))
    if cut is not None:
        ok &= cf < int(cut * 60 * FS)
    C = np.asarray(C, np.float64)
    base = C[:, cf[ok, None] + np.arange(-31, 0)].mean(2)                       # K x trials
    cue_r = C[:, cf[ok, None] + np.arange(0, 31)].mean(2) - base
    hl = fl[ok] > 0
    lick_r = np.full_like(cue_r, np.nan)
    if hl.any():
        lick_r[:, hl] = C[:, fl[ok][hl, None] + np.arange(0, 16)].mean(2) - base[:, hl]
    p = pos[ok]
    fa = np.load(res / "frames_average.npy")
    br = brain_mask(fa, 1, res=res)
    U = np.nan_to_num(np.load(res / "U.npy"))
    Ub = U.reshape(-1, U.shape[-1])[br.reshape(-1)].astype(np.float64)
    maps, ntr = {}, {}
    for kind, R in (("cue", cue_r), ("lick", lick_r)):
        mm = []
        for q in ORDER:
            sel = (p == q) & np.isfinite(R[0])
            ntr[(kind, q)] = int(sel.sum())
            mm.append(Ub @ R[:, sel].mean(1) if sel.any() else np.full(Ub.shape[0], np.nan))
        maps[kind] = 100 * np.array(mm)                                          # 6 x pix, dF/F %
    return dict(label=s["label"], animal=animal, date=date, br=br, maps=maps, ntr=ntr,
                n_trials=int(ok.sum()), n_hit=int(hl.sum()), shape=br.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="+")
    a = ap.parse_args()
    from wfield_local.parallel import fan_out
    from wfield_local.writeguard import assert_writable
    items, missing = [], []
    for d in a.dates:
        for sd in sorted((LABCAMS / d).glob("PS*")):
            h5 = sorted((DAQ / d).glob(f"{sd.name.split('_')[0]}_{d}_*.h5"))
            (items.append((str(sd), str(h5[0]))) if h5 else missing.append(sd.name))
    if missing:
        print("NO DAQ .h5 on the share, skipped:", missing, flush=True)
    res, fail = fan_out(items, one, jobs=4, label="session")
    if fail:
        print("FAILED", fail)
    R = sorted((r for _, r in res), key=lambda r: (r["date"], r["animal"]))
    assert_writable(OUT)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = len(ORDER) + 2
    fig, ax = plt.subplots(2 * len(R), ncol, figsize=(2.3 * ncol, 2.5 * 2 * len(R)))
    summary = []
    for i, r in enumerate(R):
        H, W = r["shape"]
        lim = np.nanpercentile(np.abs(np.concatenate([r["maps"]["cue"], r["maps"]["lick"]])), 99)
        for j, kind in enumerate(("cue", "lick")):
            M = r["maps"][kind]
            mu = np.nanmean(M, 0)
            spec = np.sqrt(np.nanmean((M - mu) ** 2, 0))
            panels = list(M) + [mu, spec]
            for c, m in enumerate(panels):
                z = np.full(H * W, np.nan)
                z[r["br"].reshape(-1)] = m
                axx = ax[2 * i + j, c]
                if c < ncol - 1:
                    axx.imshow(z.reshape(H, W), cmap="RdBu_r", vmin=-lim, vmax=lim)
                else:
                    axx.imshow(z.reshape(H, W), cmap="magma", vmin=0, vmax=lim / 2)
                ttl = (f"{CODE[ORDER[c]]} n={r['ntr'][(kind, ORDER[c])]}" if c < len(ORDER)
                       else ("all spouts" if c == len(ORDER) else "spout-specific RMS"))
                if c == 0:
                    ttl = f"{r['animal']} {r['date'][4:6]}-{r['date'][6:]} {kind} ({GENO[r['animal']]})\n±{lim:.2f}%  " + ttl
                axx.set_title(ttl, fontsize=7)
                axx.axis("off")
            share = float(np.nansum((M - mu) ** 2) / np.nansum(M ** 2))
            summary.append(dict(session=r["label"], kind=kind, n_trials=r["n_trials"], n_with_lick=r["n_hit"],
                                peak_all_spouts_pct=float(np.nanpercentile(np.abs(mu), 99)),
                                spout_specific_share=share))
    fig.suptitle("cohort2 per-spout maps (re-run data, hemo_meegkit_hpfit): cue 0-1 s and first lick 0-0.5 s, both "
                 "minus pre-cue 1 s; one scale per session; last column = spout-specific RMS (0 to half the scale)",
                 fontsize=10)
    fig.tight_layout()
    tag = "_".join(d[4:] for d in a.dates)
    fig.savefig(OUT / f"cohort2_spout_maps_{tag}.png", dpi=80)
    (OUT / f"cohort2_spout_maps_{tag}.json").write_text(json.dumps(summary, indent=1))
    for s_ in summary:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s_.items()}))
    print("wrote", OUT / f"cohort2_spout_maps_{tag}.png")


if __name__ == "__main__":
    main()
