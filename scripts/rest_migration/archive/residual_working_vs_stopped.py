"""How much SLOW structure survives the drift correction, WORKING period vs STOPPED period.

The two example figures (`plot_session_residual`) show the same thing twice: during the task the
corrected signal is flat, and after the animal stops working it develops large slow swings the
order-10 polynomial only partly absorbs. This turns that into a number over every stopped session, so
the claim does not rest on two pictures.

MEASURE. Row 0 of the corrected `SVTcorr` -- the dominant global mode, and a one-row mmap read, so no
138 MB `U_atlas` per session. Bin to `--bin-s` (default 30 s), take the median per bin, and report the
SD ACROSS BIN MEDIANS: that is slow-band amplitude with the fast task-locked content averaged out. The
sign ambiguity of an SVD component does not matter to an SD.

Computed separately over the ENGAGED period and the STOPPED tail, for the analysed variant and for
the stock zerophase product, so the comparison is against something.

WHY IT MATTERS. The rest definition requires frames to lie INSIDE the engaged period, so the stopped
tail is already excluded from every rest baseline. If the slow residual is confined there, the drift
correction is adequate everywhere the analyses actually look -- and that is a claim worth checking
rather than assuming.

    python -m scripts.rest_migration.archive.residual_working_vs_stopped
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import config
from wfield_local.hemo_variants import FS


def _last_engaged_s(label):
    import glob as _g

    import pandas as pd

    an, mmdd = label.split("_")
    root = Path(config.resolver().resolve("behavior_out", "")) / "sessions" / an / f"2026{mmdd}"
    hits = _g.glob(str(root / "*_trials.csv"))
    if not hits:
        return None
    d = pd.read_csv(hits[0])
    if "engaged" not in d.columns or "cue_s" not in d.columns or not len(d):
        return None
    eng = d[d["engaged"].astype(bool)]
    return float(eng["cue_s"].max()) if len(eng) else None


def _slow_sd(x, bin_n):
    """SD across bin MEDIANS -- slow-band amplitude with fast content averaged out."""
    n = (x.size // bin_n) * bin_n
    if n < 4 * bin_n:
        return np.nan
    return float(np.std(np.median(x[:n].reshape(-1, bin_n), axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-mmdd", default="0813")
    ap.add_argument("--min-stop-min", type=float, default=5.0)
    ap.add_argument("--bin-s", type=float, default=30.0)
    a = ap.parse_args()
    bin_n = int(round(a.bin_s * FS))

    rows = []
    for s in config.load_sessions():
        if s["label"].split("_")[1] < a.from_mmdd:
            continue
        p_new = Path(config.svtcorr_path(s["mc"]))
        p_old = Path(s["mc"]) / "wfield_local_results" / "SVTcorr.npy"
        if not p_new.exists():
            continue
        last_s = _last_engaged_s(s["label"])
        if last_s is None:
            continue
        try:
            new = np.asarray(np.load(p_new, mmap_mode="r")[0], np.float64)
            old = (np.asarray(np.load(p_old, mmap_mode="r")[0], np.float64)
                   if p_old.exists() else None)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:50]}", flush=True)
            continue
        n = new.size
        k = int(np.clip(last_s * FS, 1, n - 1))
        if (n - k) / FS / 60.0 < a.min_stop_min:
            continue
        w_new, t_new = _slow_sd(new[:k], bin_n), _slow_sd(new[k:], bin_n)
        w_old = _slow_sd(old[:k], bin_n) if old is not None else np.nan
        t_old = _slow_sd(old[k:], bin_n) if old is not None else np.nan
        if not np.isfinite(w_new) or not np.isfinite(t_new) or w_new <= 0:
            continue
        rows.append((s["label"], w_new, t_new, t_new / w_new, w_old, t_old,
                     (t_old / w_old) if (np.isfinite(w_old) and w_old > 0) else np.nan))
        print(f"  {s['label']:12s} analysed: working {w_new:.5f}  stopped {t_new:.5f}  "
              f"ratio {t_new/w_new:5.2f}x", flush=True)

    if not rows:
        print("no qualifying sessions")
        return
    r_new = np.array([r[3] for r in rows])
    r_old = np.array([r[6] for r in rows], float)
    print(f"\n{len(rows)} STOPPED sessions, slow-band SD in {a.bin_s:.0f} s bins "
          f"(stopped / working):")
    print(f"  ANALYSED (meegkit_hpfit)  median {np.median(r_new):.2f}x   "
          f"mean {np.mean(r_new):.2f}x   range {r_new.min():.2f}-{r_new.max():.2f}x")
    print(f"  above 1x: {int((r_new > 1).sum())}/{len(r_new)} sessions")
    ok = np.isfinite(r_old)
    if ok.any():
        print(f"  STOCK zerophase           median {np.median(r_old[ok]):.2f}x   "
              f"mean {np.mean(r_old[ok]):.2f}x   ({int(ok.sum())} sessions)")
        print(f"  above 1x: {int((r_old[ok] > 1).sum())}/{int(ok.sum())} sessions")
    print("\nLARGEST ratios:")
    for lab, w, t, rr, *_ in sorted(rows, key=lambda r: -r[3])[:8]:
        print(f"  {lab:12s} {rr:5.2f}x   working {w:.5f} -> stopped {t:.5f}")


if __name__ == "__main__":
    main()
