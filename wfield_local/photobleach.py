"""Photobleaching across several sessions: per-session figures + one summary.

For each (label, DAT, DAQ): label every camera frame 415/470 from the DAQ LED
TTLs (DAQ = ground truth, so channel identity is correct regardless of parity),
restrict to a brain ROI, and track the time-binned MEDIAN intensity per channel.
Saves a per-session PNG and finally a combined summary (normalized trends +
%-drift bars). Pairs whose DAQ doesn't match the DAT (frame-count mismatch or
missing LED channels) are skipped with a printed warning.

Run with an env that has h5py+scipy+matplotlib but WITHOUT importing wfield
(avoids the wfield/h5py DLL clash). `analyze()`/`summary()` are the reusable engine,
driven per session by `wfield_local.preprocess` (which discovers the date's sessions from
config); import stays wfield-free because `wfield_local/__init__.py` is empty.
"""
import glob, os, re, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

NSAMP, NB = 3000, 40
MIN_DUR_MIN = 15.0  # skip short baseline blips (unreliable trend)

COL = {"415": "violet", "470": "royalblue"}

AGG_JSON = "photobleach_results.json"      # the date-level aggregate (all sessions)


def record_path(out_dir, label):
    """Per-session record: a SELF-CONTAINED json next to the session's PNG.

    Why this exists: ``summary()`` needs each session's ``_norm`` traces, which used to live only
    in the analysing process's memory (they were stripped from the aggregate json). A machine that
    had not analysed a session therefore could not include it, so whichever box ran ``run()`` LAST
    overwrote the shared summary with only ITS OWN animals. With two boxes splitting a date the
    aggregate could never be complete. Persisting the full record makes the aggregate a pure
    function of what is on disk: order-independent, and rebuildable without re-reading the raw .dat.
    """
    return os.path.join(out_dir, f"photobleach_{label}.json")


def load_records(out_dir):
    """Every session record discoverable in ``out_dir``, keyed by label.

    Sources, later winning: (1) the legacy aggregate ``photobleach_results.json`` (pre-dates
    per-session records, so it carries no ``_norm`` -- such sessions still contribute their drift
    bars, just not their trend lines), then (2) per-session ``photobleach_<label>.json``.
    """
    recs = {}
    agg = os.path.join(out_dir, AGG_JSON)
    if os.path.exists(agg):
        try:
            with open(agg, encoding="utf-8") as fh:
                for r in json.load(fh):
                    if isinstance(r, dict) and r.get("label"):
                        recs[r["label"]] = r
        except (OSError, ValueError) as e:
            print(f"[photobleach] could not read {AGG_JSON} ({e}); ignoring it")
    for p in sorted(glob.glob(os.path.join(out_dir, "photobleach_*.json"))):
        if os.path.basename(p) == AGG_JSON:
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                r = json.load(fh)
            if isinstance(r, dict) and r.get("label"):
                recs[r["label"]] = r
        except (OSError, ValueError) as e:
            print(f"[photobleach] could not read {os.path.basename(p)} ({e}); skipping")
    return recs


def analyze(label, dat, daq, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if not (os.path.exists(dat) and os.path.exists(daq)):
        print(f"[{label}] SKIP missing file"); return None
    m = re.search(r"_(\d+)_(\d+)_(\d+)_uint16", os.path.basename(dat))
    H, W = int(m.group(2)), int(m.group(3))
    import h5py
    with h5py.File(daq, "r") as f:
        fs = float(f.attrs["sample_rate_hz"])
        an = [s.decode() for s in f["analog/channel_names"][:]]
        di = [s.decode() for s in f["digital/channel_names"][:]]
        if not {"led415_ttl", "led470_ttl"} <= set(an) or "pco_exposure" not in di:
            print(f"[{label}] SKIP DAQ missing LED/pco channels"); return None
        packed = f["digital/packed_samples"][:, 0]
        sc = f["analog/int16_scale_volts_per_count"][:]; of = f["analog/int16_offset_volts"][:]
        def ac(n): i = an.index(n); return f["analog/samples_int16"][:, i].astype(np.float32) * sc[i] + of[i]
        led415, led470 = ac("led415_ttl"), ac("led470_ttl")

    def db(b): return (packed >> b) & 1
    def rises(x, t=0.5): b = (np.asarray(x) > t).astype(np.int8); return np.flatnonzero(np.diff(b) == 1) + 1
    pco = rises(db(di.index("pco_exposure")))
    i2 = np.clip(pco + int(0.002 * fs), 0, len(packed) - 1)
    code = np.where((led415[i2] > 1.5) & ~(led470[i2] > 1.5), 415,
            np.where((led470[i2] > 1.5) & ~(led415[i2] > 1.5), 470, 0))

    nphys = os.path.getsize(dat) // (H * W * 2)
    if len(pco) == 0 or abs(len(pco) - nphys) / max(nphys, 1) > 0.2:
        print(f"[{label}] SKIP frame/pco mismatch (dat={nphys} pco={len(pco)})"); return None
    mm = np.memmap(dat, mode="r", dtype=np.uint16, shape=(nphys, H, W))
    n = min(nphys, len(pco))
    if (((pco / fs)[min(n - 1, len(pco) - 1)] - (pco / fs)[0]) / 60.0) < MIN_DUR_MIN:
        print(f"[{label}] SKIP too short (<{MIN_DUR_MIN:.0f} min)"); return None
    samp = np.linspace(0, n - 1, min(NSAMP, n)).astype(int)
    avg = np.zeros((H, W), np.float64)
    for k in samp[::8]: avg += mm[k]
    avg /= len(samp[::8])
    P = binary_erosion(avg > (0.45 * avg.max()), iterations=6).ravel()
    if P.sum() < 200:
        print(f"[{label}] SKIP tiny ROI ({int(P.sum())} px)"); return None

    roi = np.array([mm[k].reshape(-1)[P].mean() for k in samp], np.float64)
    lab = code[np.clip(samp, 0, len(code) - 1)]
    t = (pco / fs)[np.clip(samp, 0, len(pco) - 1)]
    dur_min = (t.max() - t.min()) / 60.0

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    edges = np.linspace(t.min(), t.max(), NB + 1); ctr = 0.5 * (edges[:-1] + edges[1:])
    res = {"label": label, "dat": dat, "daq": daq, "n_frames": int(nphys),
           "dur_min": float(dur_min), "roi_px": int(P.sum()), "channels": {}}
    norm = {}
    for c, name in [(415, "415"), (470, "470")]:
        msk = lab == c
        if msk.sum() < 30: continue
        tt, vv = t[msk], roi[msk]
        bmed = np.array([np.median(vv[(tt >= edges[i]) & (tt < edges[i + 1])])
                         if np.any((tt >= edges[i]) & (tt < edges[i + 1])) else np.nan for i in range(NB)])
        good = np.isfinite(bmed)
        p = np.polyfit(ctr[good], bmed[good], 1)
        start, end = np.polyval(p, ctr[good][0]), np.polyval(p, ctr[good][-1])
        pct = 100 * (end - start) / start
        res["channels"][name] = dict(median=float(np.median(vv)), pct=float(pct),
                                      per_min=float(p[0] * 60), start=float(start), end=float(end))
        ax[0].plot(tt, vv, ".", ms=2, color=COL[name], alpha=0.15)
        ax[0].plot(ctr[good], bmed[good], "-o", ms=4, lw=2, color=COL[name],
                   label=f"{name}  ({pct:+.1f}%)")
        nrm = bmed[good] / bmed[good][0]
        norm[name] = (ctr[good] - ctr[good][0], nrm)
        ax[1].plot(ctr[good] - ctr[good][0], nrm, "-o", ms=4, lw=2, color=COL[name], label=name)
        print(f"[{label}] {name}: median={np.median(vv):.0f} drift={pct:+.1f}% ({p[0]*60:+.1f}/min)")
    ax[0].set_xlabel("session time (s)"); ax[0].set_ylabel("brain-ROI mean"); ax[0].legend()
    ax[0].set_title(f"{label}  ROI intensity (binned median + fit)")
    ax[1].axhline(1.0, color="k", lw=0.6)
    allnrm = np.concatenate([y for _, y in norm.values()]) if norm else np.array([1.0])
    pad = max(0.01, 0.05 * (allnrm.max() - allnrm.min()))
    ax[1].set_ylim(allnrm.min() - pad, max(allnrm.max() + pad, 1.0 + pad))
    ax[1].set_xlabel("time since start (s)"); ax[1].set_ylabel("normalized (median / first bin)")
    ax[1].legend(); ax[1].set_title(f"{label}  normalized trend  ({dur_min:.1f} min)")
    plt.tight_layout()
    fp = os.path.join(out_dir, f"photobleach_{label}.png"); plt.savefig(fp, dpi=120); plt.close(fig)
    print(f"[{label}] saved {fp}")
    res["_norm"] = {k: (x.tolist(), y.tolist()) for k, (x, y) in norm.items()}
    # Persist the FULL record (incl. _norm) so any later run -- on any machine -- can rebuild a
    # complete summary from disk without re-reading the raw .dat. See record_path().
    with open(record_path(out_dir, label), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


def summary(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = [r for r in results if r]
    if not results:
        print("no sessions analyzed"); return
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(results)))
    # shared y-range across both channel panels so they stay comparable AND fit data
    ally = [v for r in results for nm in ("470", "415")
            for v in r.get("_norm", {}).get(nm, ([], []))[1]]
    ally = np.array(ally) if ally else np.array([1.0])
    pad = max(0.01, 0.05 * (ally.max() - ally.min()))
    ylo, yhi = ally.min() - pad, max(ally.max() + pad, 1.0 + pad)
    for ci, name in enumerate(["470", "415"]):
        for r, col in zip(results, cmap):
            if name in r.get("_norm", {}):
                x, y = r["_norm"][name]
                ax[ci].plot(np.array(x) / 60.0, y, "-", lw=1.8, color=col, label=r["label"])
        ax[ci].axhline(1.0, color="k", lw=0.6); ax[ci].set_ylim(ylo, yhi)
        ax[ci].set_xlabel("time since start (min)"); ax[ci].set_ylabel("normalized intensity")
        ax[ci].set_title(f"{name} nm  normalized trend (all sessions)")
        # legend only when something was plotted: legacy records recovered from the aggregate carry
        # no _norm (drift bars only), so these panels can legitimately be empty.
        if ax[ci].get_legend_handles_labels()[0]:
            ax[ci].legend(fontsize=8)
    labels = [r["label"] for r in results]
    x = np.arange(len(labels)); w = 0.38
    p415 = [r["channels"].get("415", {}).get("pct", np.nan) for r in results]
    p470 = [r["channels"].get("470", {}).get("pct", np.nan) for r in results]
    ax[2].bar(x - w / 2, p415, w, color="violet", label="415")
    ax[2].bar(x + w / 2, p470, w, color="royalblue", label="470")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, rotation=45, ha="right")
    ax[2].set_ylabel("% change over session (linear fit)")
    ax[2].set_title("Per-channel drift by session"); ax[2].legend()
    plt.suptitle("Photobleaching summary across sessions  (415 isosbestic vs 470 functional)", fontsize=13)
    plt.tight_layout()
    fp = os.path.join(out_dir, "photobleach_SUMMARY.png"); plt.savefig(fp, dpi=130); plt.close(fig)
    print("saved", fp)
    print("\n=== drift table (% over session) ===")
    print(f"{'session':12s} {'min':>5s} {'415%':>8s} {'470%':>8s}")
    for r in results:
        c = r["channels"]
        print(f"{r['label']:12s} {r['dur_min']:5.1f} "
              f"{c.get('415',{}).get('pct',float('nan')):8.1f} {c.get('470',{}).get('pct',float('nan')):8.1f}")


def run(sessions, out_dir, merge=True):
    """`sessions` = iterable of (label, dat, daq); analyze each, then rebuild the date's summary.

    MERGE BY DEFAULT. The summary and the aggregate json are rebuilt from the UNION of every
    session record on disk and the ones just analysed -- not from this run's list alone. That makes
    the shared outputs order-independent when two machines split a date: previously whichever box
    called ``run()`` last replaced ``photobleach_SUMMARY.png`` / ``photobleach_results.json`` with
    only its own animals, silently dropping the other box's (observed 2026-08-11: the 8/11 summary
    ended up containing a single animal). Pass ``merge=False`` for the old replace-everything
    behaviour, e.g. to rebuild a date from scratch after deliberately clearing it.
    """
    out = [analyze(lbl, dat, daq, out_dir) for (lbl, dat, daq) in sessions]
    fresh = {r["label"]: r for r in out if r}
    if merge:
        recs = load_records(out_dir)      # whatever any other run/machine already produced
        pre_existing = [k for k in recs if k not in fresh]
        recs.update(fresh)                # this run is authoritative for the labels it just did
        if pre_existing:
            print(f"[photobleach] merging {len(pre_existing)} session(s) from disk: "
                  f"{sorted(pre_existing)}")
    else:
        recs = fresh
    merged = [recs[k] for k in sorted(recs)]
    summary(merged, out_dir)
    with open(os.path.join(out_dir, AGG_JSON), "w", encoding="utf-8") as fh:
        json.dump([{k: v for k, v in r.items() if k != "_norm"} for r in merged], fh, indent=2)
    print(f"done -> {out_dir}  ({len(merged)} session(s) in the summary)")
    return out


def rebuild_summary(out_dir):
    """Rebuild the summary + aggregate json from the per-session records already on disk.

    Repairs a date whose shared outputs were clobbered by a partial ``run()``, with no raw .dat
    access and no recomputation. Returns the number of sessions covered.
    """
    recs = load_records(out_dir)
    if not recs:
        print(f"[photobleach] no session records under {out_dir}")
        return 0
    merged = [recs[k] for k in sorted(recs)]
    summary(merged, out_dir)
    with open(os.path.join(out_dir, AGG_JSON), "w", encoding="utf-8") as fh:
        json.dump([{k: v for k, v in r.items() if k != "_norm"} for r in merged], fh, indent=2)
    print(f"[photobleach] rebuilt summary from disk: {len(merged)} session(s) {sorted(recs)}")
    return len(merged)


if __name__ == "__main__":
    raise SystemExit("Photobleach is driven per-date by `python -m wfield_local.preprocess "
                     "<YYYYMMDD>` (session discovery lives there); call run()/analyze() directly "
                     "for ad-hoc use.")
