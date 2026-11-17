"""Cross-day RAW fluorescence intensity trend per animal (folds in the retired root
``_crossday_intensity.py``, now config-driven).

For every processed session under the ``labcams`` root
(``<labcams>/<YYYYMMDD>/<session>/motion_corrected/wfield_local_results/frames_average.npy``)
it takes the brain-ROI median of the raw-count motion-corrected mean, per channel
(ch0 = 415 isosbestic, ch1 = 470 functional), and plots the per-animal trend across days.
No raw re-read. Writes ``crossday_raw_intensity.png`` to the output dir, which
``preprocess_deck`` globs for its cross-day intensity slide.

CAVEAT (also printed on the figure): LED power is manually titrated day to day, so a trend may
reflect the LED setting, not photobleaching.

COHORT-AWARE (Priya 2026-11-05: "cross-day ROI intensity for cohort 2 should be just cohort 2 animals, for both
decks"). Only the ACTIVE cohort's animals are plotted (`config.animals()`), and a cohort with an `output_subdir`
writes to ``<xday_qc>/<subdir>/`` (cohort2: ``xday/cohort2/``), so cohort1's ``xday/crossday_raw_intensity.png``
stays where it was and no deck can pick up the other cohort's figure (`out_dir_for`). A session whose folder carries
the ``_490_`` tag (PS121 / PS122 on 2026-11-04) is plotted in the functional panel with a star: same slot, other LED.

CLI (defaults resolved from the PathResolver — ``labcams`` root, ``xday_qc`` output)::

    python -m wfield_local.crossday_intensity
    python -m wfield_local.crossday_intensity --labcams <root> --output <dir>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import binary_erosion  # noqa: E402

EXCLUDE_ANIMALS = ("PS104",)      # separate cranial-window mouse; not analyzed with the cohort
LED490: set = set()               # (animal, date) imaged with the 490 nm functional LED, filled by analyze()


def _brain_medians(frames_average_path: Path) -> tuple[float, float]:
    """(med415, med470) over a 470-defined eroded brain ROI (percentile fallback if it collapses)."""
    favg = np.load(frames_average_path)                 # (2, H, W): ch0 = 415, ch1 = 470
    ref = favg[1]
    brain = binary_erosion(ref > 0.45 * ref.max(), iterations=6)
    if brain.sum() < 200:
        brain = ref > np.percentile(ref, 40)
    return float(np.median(favg[0][brain])), float(np.median(favg[1][brain]))


def out_dir_for(xday_root) -> str:
    """The active cohort's cross-day output dir: ``xday_root`` for cohort1, ``xday_root/<output_subdir>`` else."""
    from wfield_local.paths import cohort_spec
    sub = (cohort_spec().get("output_subdir") or "").strip("/")
    return f"{str(xday_root).rstrip('/')}/{sub}" if sub else str(xday_root)


def cohort_animals() -> list[str]:
    """Animals of the active cohort (the deck and the figure must agree on who is in it)."""
    from wfield_local import config
    return list(config.animals())


def analyze(labcams_root, exclude_animals=EXCLUDE_ANIMALS, animals=None) -> list[tuple]:
    """Rows ``[(animal, date, med415, med470)]`` over every session under ``labcams_root``.

    ``animals``: keep only these (default: every animal found, as before)."""
    root = Path(labcams_root)
    if not root.exists():
        return []
    rows = []
    for date in sorted(p.name for p in root.iterdir() if p.is_dir()):
        if not re.fullmatch(r"\d{8}", date):
            continue
        for sess_dir in sorted((root / date).iterdir()):
            m = re.match(r"(PS\d+)", sess_dir.name)
            fa = sess_dir / "motion_corrected" / "wfield_local_results" / "frames_average.npy"
            if not (m and fa.exists()) or m.group(1) in exclude_animals:
                continue
            if animals is not None and m.group(1) not in animals:
                continue
            med415, med470 = _brain_medians(fa)
            rows.append((m.group(1), date, med415, med470))
            if re.match(r"PS\d+_490_", sess_dir.name):
                LED490.add((m.group(1), date))
    return rows


def plot(rows, out_dir) -> Path:
    """Two-panel (415, 470) per-animal cross-day median plot -> crossday_raw_intensity.png."""
    animals = sorted({r[0] for r in rows})
    alldates = sorted({r[1] for r in rows})               # chronological x-axis (fixed order)
    xpos = {d: i for i, d in enumerate(alldates)}
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for ci, (nm, idx) in enumerate([("415", 2), ("470", 3)]):
        for a in animals:
            rr = sorted((r for r in rows if r[0] == a), key=lambda r: r[1])
            col = f"C{animals.index(a) % 10}"          # fixed per animal: the star lines must not shift the cycle
            ax[ci].plot([xpos[r[1]] for r in rr], [r[idx] for r in rr], "-o", color=col, label=a)
            star = [r for r in rr if (r[0], r[1]) in LED490]
            if idx == 3 and star:
                ax[ci].plot([xpos[r[1]] for r in star], [r[idx] for r in star], "*", ms=16, mfc="none",
                            mec=col, mew=1.5, label=f"{a} 490 nm")
        ax[ci].set_xticks(range(len(alldates)))
        ax[ci].set_xticklabels([f"{d[4:6]}/{d[6:]}" for d in alldates])
        ax[ci].set_title(f"{nm} nm raw brain-ROI median across days" if nm == "415" or not LED490 else
                         "functional raw brain-ROI median across days (470 nm; star = 490 nm)")
        ax[ci].set_xlabel("date (MM/DD)")
        ax[ci].set_ylabel("raw counts")
        ax[ci].legend()
        ax[ci].tick_params(axis="x", rotation=45)
    fig.suptitle("Cross-day RAW fluorescence intensity (frames_average brain-ROI median)\n"
                 "CAVEAT: LED power is manually titrated day-to-day -> a trend may reflect LED "
                 "setting, not bleaching", fontsize=12)
    plt.tight_layout()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "crossday_raw_intensity.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def run(labcams_root, out_dir, animals=None) -> Path | None:
    """Compute + plot the cross-day intensity trend; returns the PNG path (None if no sessions).

    ``animals`` defaults to the ACTIVE cohort's; ``out_dir`` is taken as the xday root and scoped by `out_dir_for`."""
    animals = cohort_animals() if animals is None else animals
    out_dir = out_dir_for(out_dir)
    rows = analyze(labcams_root, animals=animals)
    if not rows:
        print(f"[crossday_intensity] no frames_average.npy found under {labcams_root}", flush=True)
        return None
    fp = plot(rows, out_dir)
    print(f"[crossday_intensity] wrote {fp}  ({len(rows)} sessions)", flush=True)
    return fp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labcams", default=None, help="labcams root (default: PathResolver labcams)")
    ap.add_argument("--output", default=None, help="output dir (default: PathResolver xday_qc)")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    from wfield_local.paths import PathResolver
    rv = PathResolver(machine=args.machine)
    run(args.labcams or rv.root("labcams"), args.output or rv.root("xday_qc"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
