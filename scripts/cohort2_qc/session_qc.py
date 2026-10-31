"""Per-session QC of the cohort2 sessions processed with the pre-SVD repair + clip mask (2026-10-31).

Priya 10-31: "new processed sessions from 10/29 and 10/30 on the server, lets check them out". 10-29 is the
re-run (repair now default); 10-30 is the first day at higher LED power for PS121 / PS122 (ALM exposure, sinus
allowed to saturate). Per session, from files on M: only (no DAQ needed):

  * levels: raw mean image (frames_average_raw) per channel -- dark, brain p50 / p99 counts, % of 16 bit;
  * clipping: swap_repair.json clip stats + clip_mask.npy (pixels clipped in > 1% of a channel's frames);
  * repair: bad frames per slot (swap_repair.json);
  * hemo coefficient: median rcoeff in the brain, saved product (rcoeffs.npy) and hemo_meegkit_hpfit/;
  * photobleach: cortex-mean dF/F (U . SVT, per channel) mean of the last 5 min minus the first 5 min;
  * residual spikes: pairs whose cortex-mean dF/F jumps > 3% from its 11-pair running median, per channel
    (the RiboL1 swaps were +-50-110%; after the repair this should be ~0).

Figure: one row per session -- 470 and 415 raw means on ONE fixed 0-65535 scale (so power is comparable across
days), clip mask, rcoeff map, bleach traces (1-min means).

    conda activate locanmf
    python scripts/cohort2_qc/session_qc.py 20261029 20261030
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402

LABCAMS = Path("M:/MICROSCOPE/Priya/Widefield/labcams")
OUT = LABCAMS / "cohort2" / "_tmp_qc_figures_20261031"
FS = 31.23
GENO = {"PS118": "Thy1-6s", "PS120": "Thy1-6s (no signal)", "PS121": "RiboL1-8s", "PS122": "RiboL1-8s"}


def _rc(path, shape):
    r = np.load(path).astype(np.float64)
    return r.reshape(shape) if r.size == shape[0] * shape[1] else r.reshape(shape + (-1,))[..., 0]


def one(sess: str) -> dict:
    sdir = Path(sess)
    res = sdir / "motion_corrected" / "wfield_local_results"
    summ = json.loads((res / "local_wfield_summary.json").read_text())
    func = int(summ["functional_channel"])
    iso = 1 - func
    fa = np.load(res / "frames_average_raw.npy").astype(np.float64)
    H, W = fa.shape[1:]
    br = brain_mask(fa, func)               # WITHOUT the clip cut: the clipped share of the brain is reported
    clip = np.load(res / "clip_mask.npy").astype(bool) if (res / "clip_mask.npy").exists() else np.zeros((H, W), bool)
    sw = json.loads((res / "swap_repair.json").read_text())
    out = dict(session=sdir.name, animal=sdir.name.split("_")[0], date=sdir.name.split("_")[1], func=func,
               shape=[H, W], pairs=sw["pairs"], bad_415=sw["bad_415_slot"], bad_470=sw["bad_470_slot"],
               clip_px=sw.get("clip_pixels_masked", 0), clip_pct_brain=float(100 * (clip & br).sum() / br.sum()),
               clip_max_frac_470=sw.get("clip_max_frame_frac_470_slot"),
               clip_max_frac_415=sw.get("clip_max_frame_frac_415_slot"))
    for c, nm in ((func, "470"), (iso, "415")):
        im = fa[c]
        dark = min(im[:5, :5].mean(), im[:5, -5:].mean(), im[-5:, :5].mean(), im[-5:, -5:].mean())
        v = im[br & ~clip]
        out[f"dark_{nm}"] = float(dark)
        out[f"p50_{nm}"] = float(np.median(v))
        out[f"p99_{nm}"] = float(np.percentile(v, 99))
        out[f"max_{nm}"] = float(im[br].max())
    rc = _rc(res / "rcoeffs.npy", (H, W))
    out["rcoeff_saved"] = float(np.nanmedian(rc[br & ~clip]))
    hp = res / "hemo_meegkit_hpfit" / "rcoeffs.npy"
    out["rcoeff_meegkit_hpfit"] = float(np.nanmedian(_rc(hp, (H, W))[br & ~clip])) if hp.exists() else None
    U = np.load(res / "U.npy", mmap_mode="r")
    um = np.nan_to_num(np.asarray(U).reshape(-1, U.shape[-1])[(br & ~clip).reshape(-1)]).mean(0)
    del U
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    tr = {}
    for c, nm in ((func, "470"), (iso, "415")):
        x = 100 * (um @ np.asarray(SVT[:, c::2], np.float64))       # cortex-mean dF/F %
        n5 = int(5 * 60 * FS)
        out[f"bleach_{nm}_pct"] = float(x[-n5:].mean() - x[:n5].mean())
        from scipy.ndimage import median_filter
        out[f"spikes_{nm}"] = int((np.abs(x - median_filter(x, 11, mode="nearest")) > 3.0).sum())
        m = int(60 * FS)
        tr[nm] = x[: len(x) // m * m].reshape(-1, m).mean(1)
    out["dur_min"] = float(len(x) / FS / 60)
    imgs = dict(f470=fa[func].astype(np.float32), f415=fa[iso].astype(np.float32), clip=clip, brain=br,
                rc=rc.astype(np.float32), tr=tr)
    return dict(stats=out, imgs=imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    from wfield_local.parallel import fan_out
    from wfield_local.writeguard import assert_writable
    sessions = sorted(str(p) for d in a.dates for p in (LABCAMS / d).glob("PS*")
                      if (p / "motion_corrected" / "wfield_local_results" / "swap_repair.json").exists())
    res, fail = fan_out(sessions, one, jobs=a.jobs, label="session")
    if fail:
        raise SystemExit(f"failed: {fail}")
    R = sorted((r for _, r in res), key=lambda r: (r["stats"]["animal"], r["stats"]["date"]))
    assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    stats = [r["stats"] for r in R]
    (OUT / "session_qc.json").write_text(json.dumps(stats, indent=1))
    cols = ["session", "dur_min", "p50_470", "p99_470", "max_470", "p50_415", "p99_415", "clip_px", "clip_pct_brain",
            "bad_415", "bad_470", "rcoeff_saved", "rcoeff_meegkit_hpfit", "bleach_470_pct", "bleach_415_pct",
            "spikes_470", "spikes_415"]
    print("\t".join(cols))
    for s in stats:
        print("\t".join(f"{s[c]:.2f}" if isinstance(s[c], float) else str(s[c]) for c in cols))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(R)
    fig, ax = plt.subplots(n, 5, figsize=(19, 3.3 * n), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.6]))
    for i, r in enumerate(R):
        s, im = r["stats"], r["imgs"]
        lab = f"{s['animal']} {s['date'][4:6]}-{s['date'][6:]} ({GENO.get(s['animal'], '')})"
        h0 = ax[i, 0].imshow(im["f470"], cmap="gray", vmin=0, vmax=65535)
        ax[i, 0].set_title(f"{lab}\n470 raw mean (0-65535)  brain p99 {s['p99_470']:.0f}", fontsize=8)
        ax[i, 1].imshow(im["f415"], cmap="gray", vmin=0, vmax=65535)
        ax[i, 1].set_title(f"415 raw mean (0-65535)  brain p99 {s['p99_415']:.0f}", fontsize=8)
        ov = np.dstack([im["f470"] / 65535] * 3)
        ov[im["clip"]] = [1, 0, 0]
        ax[i, 2].imshow(ov)
        ax[i, 2].contour(im["brain"], [0.5], colors="c", linewidths=0.6)
        ax[i, 2].set_title(f"clipped (red): {s['clip_px']} px, {s['clip_pct_brain']:.2f}% of brain", fontsize=8)
        rc = np.where(im["brain"] & ~im["clip"], im["rc"], np.nan)
        ax[i, 3].imshow(rc, cmap="viridis", vmin=0, vmax=3)
        ax[i, 3].set_title(f"hemo coefficient (saved), median {s['rcoeff_saved']:.2f}", fontsize=8)
        for a_ in ax[i, :4]:
            a_.axis("off")
        for nm, col in (("470", "tab:green"), ("415", "tab:purple")):
            t = np.arange(len(im["tr"][nm]))
            ax[i, 4].plot(t, im["tr"][nm], color=col, label=f"{nm}: {s[f'bleach_{nm}_pct']:+.1f}% last-first 5 min")
        ax[i, 4].set_xlabel("min")
        ax[i, 4].set_ylabel("cortex dF/F %")
        ax[i, 4].legend(fontsize=7)
        ax[i, 4].set_title(f"residual >3% single-pair spikes: 470 {s['spikes_470']}, 415 {s['spikes_415']}", fontsize=8)
    fig.colorbar(h0, ax=ax[:, 0], fraction=0.02, label="counts")
    fig.suptitle("cohort2 sessions after pre-SVD repair + clip mask (repair default since 10-30)", fontsize=12)
    fig.savefig(OUT / "cohort2_session_qc.png", dpi=80, bbox_inches="tight")
    print("wrote", OUT / "cohort2_session_qc.png")


if __name__ == "__main__":
    main()
