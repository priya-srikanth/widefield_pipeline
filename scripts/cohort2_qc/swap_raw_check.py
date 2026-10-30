"""IMAGING BOX: look at the RAW frames and the camlog around the other-LED ("swapped") frames. READ-ONLY on raw data.

Context (docs/COHORT2_QC_2026-10-29.md §0c): in the SVD products some frames carry the other LED's light: whole
frames, or a BAND OF ROWS (rolling-shutter signature). The DAQ says those exposures were clean (LED pulse =
exposure TTL, 16.0 ms period, nothing irregular at 0.2 ms resolution). This script answers what the SVD cannot:
  1. Is a swapped frame an exact DUPLICATE of another frame nearby (camera/labcams buffer repeat), or a real new
     exposure? (np.array_equal against every frame in +-8.)
  2. Its brightness and ROW-BAND profile vs same-LED neighbours (+-2), from the raw counts: which rows got the
     other LED's light (top, bottom, centre-out -- the PCO edge reads out from the centre).
  3. The camlog around it: camera frame_id step (should be 1), timestamp step (should be ~16 ms), the #LED line
     sequence, and the labcams write queue ("# Queue: N" lines). A frame_id skip / repeat / time hiccup / queue
     backlog at exactly these frames points at acquisition software or disk load.
  4. Session-wide camlog stats: every frame_id step != 1, every dt outlier, every Queue > 0, and how many of them
     coincide with a swapped frame (vs chance).

DO NOT RUN WHILE ACQUIRING: it reads the .dat from E:, and disk contention during acquisition is itself a suspect.

    conda activate wfield
    python scripts/cohort2_qc/swap_raw_check.py E:/labcams_data/20261029/PS121_20261029_160848 [more sessions]
        [--mask <...>/swapped_pairs.npz] [--max 60] [--out E:/labcams_data/20261029/_swap_raw_check]

`swapped_pairs.npz` comes from scripts/cohort2_qc/swapmask_1029.py (analysis box). It is read from the session's
local wfield_local_results/ if present, else from the MICROSCOPE copy (N: on this box), else pass --mask.
Writes a CSV per session, a JSON summary and a row-band PNG into --out (never next to the raw data).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

SHARE = Path("N:/MICROSCOPE/Priya/Widefield/labcams")
NB = 16              # row bands
WIN = 8              # frames either side for the duplicate test


def find_mask(sess: Path, given: Path | None) -> Path:
    if given:
        return given
    loc = sess / "motion_corrected" / "wfield_local_results" / "swapped_pairs.npz"
    if loc.is_file():
        return loc
    sh = SHARE / sess.parent.name / sess.name / "motion_corrected" / "wfield_local_results" / "swapped_pairs.npz"
    if sh.is_file():
        return sh
    raise SystemExit(f"{sess.name}: no swapped_pairs.npz locally or at {sh}; pass --mask")


def parse_camlog(p: Path):
    """frame lines in WRITE order -> (frame_id, t_s); #LED lines -> (id, n, t_ms); queue values with the index of
    the frame line they follow."""
    from datetime import datetime
    fid, ts, led, queue = [], [], [], []
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#LED:"):
                a = line[5:].strip().split(",")
                led.append((int(a[0]), int(a[1]), float(a[2])))
            elif line.startswith("#"):
                m = re.search(r"Queue:\s*(\d+)", line)
                if m:
                    queue.append((len(fid), int(m.group(1))))
            elif line[:1].isdigit():
                a, b = line.strip().split(",", 1)
                fid.append(int(a))
                ts.append(datetime.fromisoformat(b.strip()).timestamp())
    return np.array(fid), np.array(ts), np.array(led), np.array(queue).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--mask", type=Path, default=None)
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    from wfield_local.writeguard import assert_writable
    summary = {}
    for sess in a.sessions:
        out = a.out or sess.parent / "_swap_raw_check"
        assert_writable(out)
        out.mkdir(parents=True, exist_ok=True)
        mc = sess / "motion_corrected"
        dat = next((sess / "raw_widefield_data").glob("*_uint16.dat"))
        m = re.search(r"_(\d+)_(\d+)_(\d+)_uint16", dat.name)
        H, W = int(m[2]), int(m[3])
        nphys = dat.stat().st_size // (H * W * 2)
        raw = np.memmap(dat, dtype=np.uint16, mode="r", shape=(nphys, H, W))        # READ-ONLY
        fm = np.load(next(mc.glob("*cleanpairs_frame_map.npz")))
        lab = fm["labels_per_original_frame"]
        mk = np.load(find_mask(sess, a.mask))
        fa = np.load(mc / "wfield_local_results" / "frames_average.npy")
        brain = fa[1] > 0.45 * fa[1].max()
        hits = []
        for ch, nm in ((0, "415"), (1, "470")):
            core = mk["bright_" + nm] | mk["pattern_" + nm]
            for p in np.flatnonzero(core[:len(fm[f"original_frame_index_ch{ch}"])]):
                hits.append((int(fm[f"original_frame_index_ch{ch}"][p]), nm, int(p)))
        hits.sort()
        camlog = next((sess / "raw_widefield_data").glob("*.camlog"), None)
        cl = parse_camlog(camlog) if camlog else None
        rows = []
        for of, nm, p in hits[:a.max]:
            lo, hi = max(of - WIN, 0), min(of + WIN + 1, nphys)
            X = np.asarray(raw[of])
            dup = [k - of for k in range(lo, hi) if k != of and np.array_equal(np.asarray(raw[k]), X)]
            nb = [k for k in (of - 2, of + 2) if 0 <= k < nphys]
            Y = np.mean([np.asarray(raw[k], np.float64) for k in nb], 0)
            other = [k for k in (of - 1, of + 1) if 0 <= k < nphys]
            Z = np.mean([np.asarray(raw[k], np.float64) for k in other], 0)
            r_own = X[brain].mean() / Y[brain].mean()
            r_oth = Z[brain].mean() / Y[brain].mean()
            band = [float(np.nanmean(np.where(brain[i:i + H // NB], X[i:i + H // NB] / Y[i:i + H // NB], np.nan)))
                    for i in range(0, H - H // NB + 1, H // NB)]
            rec = dict(orig_frame=of, slot=nm, pair=p, daq_label=int(lab[of]), ratio_to_same_led=r_own,
                       other_led_ratio=r_oth, frac_other=(r_own - 1) / (r_oth - 1) if r_oth != 1 else np.nan,
                       duplicate_of=dup, bands=band)
            if cl is not None:
                fid, ts, led, queue = cl
                if of + 1 < len(fid) and of >= 1:
                    rec.update(frame_id_step_in=int(fid[of] - fid[of - 1]), frame_id_step_out=int(fid[of + 1] - fid[of]),
                               dt_in_ms=1e3 * (ts[of] - ts[of - 1]), dt_out_ms=1e3 * (ts[of + 1] - ts[of]))
                near = queue[np.abs(queue[:, 0] - of) <= 50] if len(queue) else queue
                rec["queue_near_max"] = int(near[:, 1].max()) if len(near) else None
            rows.append(rec)
            print(f"{sess.name} {nm} frame {of}: x{r_own:.2f} vs same LED (other LED x{r_oth:.2f}, frac "
                  f"{rec['frac_other']:.2f}) dup {dup} bands " + " ".join(f"{b:.2f}" for b in band)
                  + (f" | id steps {rec.get('frame_id_step_in')}/{rec.get('frame_id_step_out')} dt "
                     f"{rec.get('dt_in_ms', np.nan):.1f}/{rec.get('dt_out_ms', np.nan):.1f} ms queue {rec.get('queue_near_max')}"
                     if cl is not None else ""), flush=True)
        s = dict(n_hits=len(hits), n_checked=len(rows), n_duplicates=sum(bool(r["duplicate_of"]) for r in rows),
                 median_frac_other=float(np.nanmedian([r["frac_other"] for r in rows])) if rows else None)
        if cl is not None:
            fid, ts, led, queue = cl
            n = min(len(fid), nphys)
            step = np.diff(fid[:n])
            dt = 1e3 * np.diff(ts[:n])
            odd = np.flatnonzero((step != 1) | (np.abs(dt - np.median(dt)) > 4.0)) + 1
            qpos = queue[queue[:, 1] > 0, 0] if len(queue) else np.array([], int)
            hf = np.array([h[0] for h in hits])

            def near(ev, hf=hf):
                return float(np.mean(np.min(np.abs(hf[:, None] - ev[None, :]), 1) <= 2)) if len(ev) and len(hf) else None
            s.update(camlog_frames=int(len(fid)), dat_frames=int(nphys), id_steps_not_1=int((step != 1).sum()),
                     dt_outliers=int((np.abs(dt - np.median(dt)) > 4.0).sum()), dt_median_ms=float(np.median(dt)),
                     queue_lines=int(len(queue)), queue_gt0=int(len(qpos)), queue_max=int(queue[:, 1].max()) if len(queue) else 0,
                     hits_within2_of_camlog_oddity=near(odd), hits_within2_of_queue_gt0=near(qpos),
                     chance_within2_of_oddity=float(min(1.0, 5 * len(odd) / max(n, 1))))
        summary[sess.name] = s
        print(sess.name, json.dumps(s), flush=True)
        import csv
        with open(out / f"{sess.name}_swap_raw_check.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        if rows:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            for r in rows[:24]:
                ax.plot(r["bands"], np.arange(len(r["bands"])), lw=0.8)
            ax.invert_yaxis()
            ax.set_xlabel("raw brightness / same-LED neighbours")
            ax.set_ylabel(f"row band (top -> bottom, {NB})")
            ax.set_title(f"{sess.name}: swapped frames, row profile")
            fig.tight_layout()
            fig.savefig(out / f"{sess.name}_swap_rowbands.png", dpi=110)
            plt.close(fig)
    (Path(a.out or a.sessions[0].parent / "_swap_raw_check") / "swap_raw_check_summary.json").write_text(
        json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
