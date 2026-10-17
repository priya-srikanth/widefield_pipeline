"""QC sheet for the tongue-detector amplitude floor (`orofacial_kinematics.tongue.detector.min_peak_y_abs`, 20 px, an
OLD-RIG value marked retune). Would lowering it to 10 px add real (incomplete) licks, or noise?

    python -m scripts.lick_floor_qc [--floor 2] [--per-sheet 24]

For a few pre-stroke and acute sessions it re-runs the SAME cleaning + v7 detector as `orofacial_outputs` with only
`min_peak_y_abs` lowered to ~0 (Priya 10-16: "why even the 10 px floor"), plots the DISTRIBUTION of peak
protrusion (a dip between noise and licks is where a floor belongs), and draws licks from each low band (2-10, 10-20 px
-- the ones the production floor drops) each as five video frames around its peak (-8, -4, 0, +4, +8 ms) cropped around the mouth,
with the raw LP tongue point (o, filled when likelihood >= 0.4) and the mouth (+). A CONTROL sheet shows licks of
20-30 px, which the production floor already counts. Sheets go to
`<LP_analysis>/<run>/qc/lick_floor/` (Priya reviews them before any parameter changes; a change is a new run).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RUN = "lp_r4_heatmap_20261013__clean_v35"
SESSIONS = {"pre": [("PS92", "20260809"), ("PS94", "20260809"), ("PS93", "20260814")],
            "acute": [("PS92", "20260818"), ("PS94", "20260817"), ("PS93", "20260821")]}
OFFSETS = (-2, -1, 0, 1, 2)                 # frames around the peak (4 ms each)
CROP = (-110, 110, -40, 200)                # x0, x1, y0, y1 around the mouth (image px; the tongue goes DOWN)
LK_THR = 0.4


def redetect(rv, animal: str, date: str, floor: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """(licks with the lowered floor, production licks, window index, raw pose) for one processed session."""
    import scripts.pose_kinematics_demo as D
    from wfield_local import orofacial_outputs as oo
    from wfield_local import spout_frame as SF
    from wfield_local import trial_windows as TW
    sd = oo.session_dir(rv, RUN, animal, date)
    w = np.load(sd / "windows.npz")
    ix = pd.DataFrame({k[3:]: w[k] for k in w.files if k.startswith("ix_")})
    parts = [str(x) for x in w["parts"]]
    flat = oo._flat(len(ix), w["pose"], parts)
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in ix.groupby("trial_k")]
    frame = SF.from_medians(SF.position_medians(flat[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    b = TW.trial_bounds(animal, date, rv)
    stops = dict(zip(b.trial_id, b.stop_s - b.cue_s))
    contacts = D.daq_contacts_ms(animal, date, b, rv)
    T, _, _, _ = D.run(flat, ix, frame, stops, contacts, params_override={"detector": {"min_peak_y_abs": floor}})
    return T.per_lick, pd.read_csv(sd / "licks.csv"), ix, w["pose"]


def peak_rows(L: pd.DataFrame, ix: pd.DataFrame) -> np.ndarray:
    first = ix.reset_index().groupby("trial_id").agg(row0=("index", "min"), t0=("t_ms", "min"))
    return (first.row0.reindex(L.trial_id).to_numpy() +
            np.round((L.t_ms.to_numpy() - first.t0.reindex(L.trial_id).to_numpy()) / 4.0)).astype(int)


def draw_sheet(items: list[dict], path: Path, title: str) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(items)
    if not n:
        return
    fig, axs = plt.subplots(n, len(OFFSETS), figsize=(1.55 * len(OFFSETS), 1.75 * n), squeeze=False)
    caps = {}
    for i, it in enumerate(items):
        cap = caps.get(it["video"]) or cv2.VideoCapture(str(it["video"]))
        caps[it["video"]] = cap
        ox, oy = it["origin"]
        x0, x1, y0, y1 = (int(ox + CROP[0]), int(ox + CROP[1]), int(oy + CROP[2]), int(oy + CROP[3]))
        for j, d in enumerate(OFFSETS):
            ax = axs[i][j]
            ax.axis("off")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(it["frame"] + d))
            ok, img = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ax.imshow(g[max(y0, 0):y1, max(x0, 0):x1], cmap="gray", vmin=0, vmax=255)
            tx, ty, lk = it["tongue"][j]
            ax.plot(ox - max(x0, 0), oy - max(y0, 0), "+", color="cyan", ms=6, mew=1)
            if np.isfinite(tx):
                ax.plot(tx - max(x0, 0), ty - max(y0, 0), "o", ms=4, mew=1, color="red",
                        mfc="red" if lk >= LK_THR else "none")
            if j == 0:
                ax.text(0, -4, it["label"], fontsize=5.5, va="bottom")
            if i == 0:
                ax.set_title(f"{d * 4:+d} ms", fontsize=7)
    for cap in caps.values():
        cap.release()
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    from wfield_local.orofacial_cohort import save_png_svg
    save_png_svg(fig, path, dpi=110)
    plt.close(fig)


def _histogram(A: pd.DataFrame, floor: float, path: Path) -> None:
    """Peak-protrusion distribution with the floor lowered, per epoch x animal, 0-80 px in 2-px bins; production floor
    (20 px) dashed. Shaded = fraction of licks in the bin with a DAQ spout contact (right axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    animals = sorted(A.animal.unique())
    fig, axs = plt.subplots(2, len(animals), figsize=(3.6 * len(animals), 5.2), squeeze=False)
    edges = np.arange(0, 82, 2)
    for i, ep in enumerate(("pre", "acute")):
        for j, an in enumerate(animals):
            ax = axs[i][j]
            g = A[(A.epoch == ep) & (A.animal == an)]
            ax.hist(g.protrusion_max_px, bins=edges, color="0.35", zorder=3)
            ax.axvline(20, color="tab:red", ls="--", lw=1, zorder=0)
            ax.axvline(floor, color="0.6", ls=":", lw=1, zorder=0)
            b = np.digitize(g.protrusion_max_px, edges) - 1
            cf = pd.Series(g.contact.astype(float).to_numpy()).groupby(b).mean().reindex(range(len(edges) - 1))
            ax2 = ax.twinx()
            ax2.plot(edges[:-1] + 1, cf.to_numpy(), color="tab:blue", lw=1, zorder=2)
            ax2.set_ylim(0, 1)
            ax2.tick_params(labelsize=6, colors="tab:blue")
            ax.set_title(f"{an} {ep} (n={len(g)})", fontsize=8)
            ax.tick_params(labelsize=6.5)
            if i == 1:
                ax.set_xlabel("peak protrusion (px)", fontsize=7.5)
    fig.suptitle(f"lick peak protrusion with the floor at {floor:.0f} px (production: 20 px, red dashed); "
                 "blue = fraction with DAQ spout contact; 0-80 px shown", fontsize=8.5)
    fig.tight_layout()
    from wfield_local.orofacial_cohort import save_png_svg
    save_png_svg(fig, path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor", type=float, default=2.0)
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rv = PathResolver()
    out = oo.root(rv) / RUN / "qc" / "lick_floor"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    summary, amps = [], []
    bands = [(a.floor, 10.0), (10.0, 20.0)]
    for epoch, sess in SESSIONS.items():
        band_items = {bd: [] for bd in bands}
        ctrl_items = []
        for animal, date in sess:
            Lnew, Lprod, ix, pose = redetect(rv, animal, date, a.floor)
            man = pd.read_json(oo.session_dir(rv, RUN, animal, date) / "manifest.json", typ="series")
            video = Path(rv.root("behavior_cameras")) / date / animal / man["videos"][0]["video"]
            c = np.load(oo.session_dir(rv, RUN, animal, date) / "clean.npz")
            origin = tuple(c["tongue_origin"])
            parts = [str(x) for x in np.load(oo.session_dir(rv, RUN, animal, date) / "windows.npz")["parts"]]
            kt = parts.index("tongue")
            ctrl = Lprod[(Lprod.protrusion_max_px >= 20.0) & (Lprod.protrusion_max_px < 30.0)]
            amps.append(Lnew.assign(epoch=epoch, animal=animal)[["epoch", "animal", "protrusion_max_px", "contact"]])
            row = {"epoch": epoch, "animal": animal, "date": date, "n_licks_prod": len(Lprod),
                   "n_licks_lowfloor": len(Lnew), "n_ctrl_20_30": len(ctrl),
                   "ctrl_with_daq_contact": float(ctrl.contact.mean()) if len(ctrl) else np.nan}
            pairs = [(ctrl, ctrl_items)]
            for lo, hi in bands:
                bnd = Lnew[(Lnew.protrusion_max_px >= lo) & (Lnew.protrusion_max_px < hi)]
                row[f"n_{lo:.0f}_{hi:.0f}"] = len(bnd)
                row[f"contact_{lo:.0f}_{hi:.0f}"] = float(bnd.contact.mean()) if len(bnd) else np.nan
                pairs.append((bnd, band_items[(lo, hi)]))
            summary.append(row)
            for L, dest in pairs:
                if not len(L):
                    continue
                pick = L.iloc[rng.choice(len(L), size=min(len(L), max(1, a.per_sheet // len(sess))), replace=False)]
                rows = peak_rows(pick, ix)
                for (_, r), row in zip(pick.iterrows(), rows):
                    if not (0 <= row < len(ix)) or "video" in ix and str(ix.video.iloc[row]) != video.name:
                        continue
                    f = int(ix.src_frame.iloc[row])
                    tong = [tuple(pose[min(max(row + d, 0), len(pose) - 1), kt]) for d in OFFSETS]
                    dest.append({"video": video, "frame": f, "origin": origin, "tongue": tong,
                                 "label": f"{animal} {date[4:]} t{int(r.trial_id)} {r.position} {r.t_ms:.0f}ms "
                                          f"{r.protrusion_max_px:.0f}px contact={bool(r.contact)}"})
        for (lo, hi), items in band_items.items():
            draw_sheet(items, out / f"new_licks_{lo:.0f}_{hi:.0f}px_{epoch}.png",
                       f"{epoch}: licks ADDED by lowering the floor 20 -> {a.floor:.0f} px, peak protrusion {lo:.0f}-{hi:.0f} "
                       "px\no = LP tongue (filled = likelihood >= 0.4), + = mouth; frames around the detected peak")
        draw_sheet(ctrl_items, out / f"control_licks_20_30px_{epoch}.png",
                   f"{epoch}: CONTROL -- licks of 20-30 px the production floor already counts"
                   "\no = LP tongue (filled = likelihood >= 0.4), + = mouth; frames around the detected peak")
    _histogram(pd.concat(amps, ignore_index=True), a.floor, out / "peak_protrusion_histogram.png")
    S = pd.DataFrame(summary)
    S.to_csv(out / "summary.csv", index=False)
    print(S.to_string(index=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
