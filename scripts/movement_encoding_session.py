"""Movement-regressor encoding model on REAL sessions: LocaNMF dF/F ~ cue per position + DAQ contacts + DLC tongue
onsets + continuous tongue / jaw + per-lick direction modulators, fit on the imaging frames covered by the session's
pose windows (`scripts.session_poses` clips: 60 trials).

    python -m scripts.movement_encoding_session PS93:20260814 PS93:20260821 PS93:20260908

Priya, 2026-10-02: "ok yes a per-lick modulator then. can you test this out on the DLC session data we already have".
A first end-to-end test of `movement_encoding` / `movement_inputs` -- the pose covers only the subset windows
(strobe -0.5 s -> stop +3 s of 60 trials), so the model is fit on those frames only; whole sessions come with the next
DLC / LP iteration (O2).

Regressors (groups -> partitions task / movement / direction):
  task         cue_<position> kernels (6, the target)
  lick_events  DAQ spout contact (configs `lick_detection`, Priya's choice for DLC analyses), DLC tongue onset
               (every kept lick incl. incomplete; DAQ time = cue + on_ms)
  tongue / jaw continuous protrusion (tongue-in = lip level), protrusion speed, sideways position; jaw y / speed
  direction    tongue onset x DEVIATION from the session's contact-lick path (lick_reference) and x raw executed angle,
               licks reaching >= min_reach_px only
Outputs (session_poses/<a>_<d>/): movement_encoding_vp.csv (per component and partition) and a figure; a printed
per-area summary (Allen areas; region label > 0 = LEFT hemisphere = ipsilesional for these L-lesioned animals).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import movement_encoding as ME
from wfield_local import movement_inputs as MI

AREA_NAMES = {3: "MOp", 4: "MOs", 5: "SSp-n", 6: "SSp-m"}


def area_name(lab: int) -> str:
    return f"{AREA_NAMES.get(abs(int(lab)), str(abs(int(lab))))}_{'L' if lab > 0 else 'R'}"


def imaging(label: str):
    """(dF/F (T, ncomp), region labels, frame DAQ seconds) for one imaging session (regime B)."""
    import glob
    import json

    from wfield_local import config
    from wfield_local.locanmf_crossanimal_dff import _footprint_scale
    from wfield_local.locanmf_lick_aligned import _corrected_frame_samples
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events
    s = next(x for x in config.load_sessions() if x["label"] == label)
    if s["regime"] != "B":
        raise SystemExit(f"{label}: regime {s['regime']} -- this test handles regime B (corrected frame map) only")
    d = config.locanmf_dir(s["mc"])
    C = np.load(f"{d}/{label}_locanmf_C.npy").astype(np.float64)
    A = np.load(f"{d}/{label}_locanmf_A.npy", mmap_mode="r")
    reg = np.load(f"{d}/{label}_locanmf_regions.npy")
    dff = (_footprint_scale(A, C.shape[0])[:, None] * C).T
    cue = _load_cue_events(s["h5"])
    fmdir = s["fmdir"] or s["mc"]
    fm = glob.glob(f"{fmdir}/*cleanpairs_frame_map.npz")[0]
    off = json.loads(Path(glob.glob(f"{fmdir}/*cleanpairs_summary.json")[0]).read_text())["chosen_exposure_offset"]
    csmp = _corrected_frame_samples(fm, cue["pco_samples"], int(off))
    ft = np.asarray(csmp, float) / float(cue["sample_rate_hz"])
    n = min(len(ft), dff.shape[0])
    return dff[:n], reg, ft[:n]


def pose_inputs(animal: str, date: str, rv, frame_times_s, min_reach_px: float, spec: str | None = None):
    """MovementInputs + a mask of imaging frames inside the pose windows (``spec``: see `session_pieces`)."""
    P = session_pieces(animal, date, rv, frame_times_s, min_reach_px, spec=spec)
    L = P["licks"]
    inputs = MI.build_inputs(frame_times_s, cues=P["trials"][["cue_s", "pos_name"]],
                             events={"contact": P["contact_s"], "tongue_onset": L.on_s.to_numpy()},
                             modulated={"tongue_onset_x_deviation": (L.on_s.to_numpy(), L.dev_dir.to_numpy()),
                                        "tongue_onset_x_angle": (L.on_s.to_numpy(), L.angle_dir.to_numpy())},
                             video_signals=P["sig"], video_t_s=P["vt"], trial_starts_s=P["trial_starts_s"])
    return inputs, P["mask"], P["info"]


def session_pieces(animal: str, date: str, rv, frame_times_s, min_reach_px: float, spec: str | None = None) -> dict:
    """Everything a model variant needs, on the DAQ clock: trials (cue_s, pos_name), DAQ contacts, a per-lick table
    (on_s, position, angle, deviation, reach; *_dir = NaN for licks under min_reach_px), continuous pose signals with
    their DAQ times, trial starts, and the mask of imaging frames inside the pose windows. ``spec`` = a tagged
    session_poses folder (``animal:date:tag``, e.g. ``:full`` from O2, ``:r4``); default the 60-trial clip."""
    import scripts.pose_kinematics_demo as D
    from scripts.session_poses import session_dir
    from wfield_local import lick_reference as LR
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    from wfield_local import tongue_detect as td
    from wfield_local import trial_windows as TW
    animal, date, d = session_dir(rv, spec or f"{animal}:{date}")
    idx = pd.read_csv(d / "windows_index.csv")
    raw = oc.read_pose(d / "windows_DLC.csv").iloc[:len(idx)]
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    frame = SF.from_medians(SF.position_medians(raw[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    b = TW.trial_bounds(animal, date, rv)
    contacts = D.daq_contacts_ms(animal, date, b, rv)
    T, J, M, clean = D.run(d / "windows_DLC.csv", idx, frame, dict(zip(b.trial_id, b.stop_s - b.cue_s)),
                           contacts_ms=contacts)
    tpl = dict(np.load(Path(rv.root("alignment_templates")) / "cam4" / animal / f"{date}.npz", allow_pickle=True))
    cue_of = dict(zip(b.trial_id, b.cue_s))
    # --- per-lick events (DAQ s) and direction modulators
    pl, ph = T.per_lick, T.lick_phase
    pl, _ = LR.add_deviation(pl, ph, LR.build_reference([pl], [ph]), "session")
    on_s = pl.trial_id.map(cue_of).to_numpy(float) + pl.on_ms.to_numpy(float) / 1000.0
    reach_ok = pl.protrusion_max_px.to_numpy(float) >= min_reach_px
    dev = np.where(reach_ok, pl.dev_session_deg.to_numpy(float), np.nan)
    ang = np.where(reach_ok, LR.peak_angle(pl).to_numpy(float), np.nan)
    # --- DAQ contacts in the windowed trials (config lick_detection)
    sel = set(idx.trial_id)
    contact_s = np.concatenate([cue_of[t] + np.asarray(c, float) / 1000.0 for t, c in contacts.items() if t in sel])
    # --- continuous signals per clip row -> DAQ time via the camera template
    tg, pos_map, _ = clean["tongue"]
    jw = clean["jaw"][0]
    _, tin = td.interp_masks(tg.fill_method)
    _, jbase = td.interp_masks(jw.fill_method)
    sig = MI.pose_signals(tg.x_final[pos_map] + tg.X0, tg.y_final[pos_map] + tg.Y0, tin[pos_map],
                          jw.y_final[pos_map], origin=frame.origin, ap_axis=frame.ap_axis, fps=tg.fps,
                          jaw_unknown=~np.isfinite(jw.y_final[pos_map]))
    vt = MI.cam_frames_to_daq_s(tpl, idx.src_frame.to_numpy())
    trials = b[b.trial_id.isin(sel)]
    # imaging frames inside a pose window (per trial: first .. last clip row time)
    ft = np.asarray(frame_times_s, float)
    mask = np.zeros(len(ft), bool)
    for _, g in idx.groupby("trial_k"):
        t0, t1 = MI.cam_frames_to_daq_s(tpl, [g.src_frame.min(), g.src_frame.max()])
        mask |= (ft >= t0) & (ft <= t1)
    info = {"n_licks": len(pl), "n_dir_licks": int(np.isfinite(dev).sum()), "n_contacts": len(contact_s),
            "n_trials": len(trials)}
    licks = pd.DataFrame({"on_s": on_s, "trial_id": pl.trial_id.to_numpy(), "position": pl.position.to_numpy(),
                          "cue_s": pl.trial_id.map(cue_of).to_numpy(float),
                          "contact": pl.contact.to_numpy(), "reach_ok": reach_ok,
                          "angle": LR.peak_angle(pl).to_numpy(float), "dev": pl.dev_session_deg.to_numpy(float),
                          "angle_dir": ang, "dev_dir": dev})
    return {"trials": trials, "contact_s": contact_s, "licks": licks, "sig": sig, "vt": vt, "mask": mask,
            "trial_starts_s": trials.cue_s.to_numpy(float) - 0.5, "info": info}


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="animal:date (a scripts.session_poses folder)")
    ap.add_argument("--areas", nargs="+", type=int, default=[4, 3, 6, 5])
    a = ap.parse_args(argv)
    rv = PathResolver()
    p = MI.params()
    parts = (config.defaults().get("movement_encoding", {}) or {}).get("partitions") or {
        "task": ["task"], "movement": ["lick_events", "tongue", "jaw"], "direction": ["direction"]}
    summary = []
    for spec in a.sessions:
        animal, date = spec.split(":")[:2]
        label = f"{animal}_{date[4:]}"
        Y, reg, ft = imaging(label)
        label += "".join(f"_{t}" for t in spec.split(":")[2:])   # the pose source, e.g. PS93_0814_full
        inputs, mask, info = pose_inputs(animal, date, rv, ft, float(p["min_reach_px"]), spec=spec)
        d_full = ME.build_design(inputs)
        d = ME.select_rows(d_full, mask)
        Ym = Y[mask]
        folds = ME.trial_block_folds(ft[mask], inputs.trial_starts_s, int(p["n_folds"]))
        keep = [k for k, lab in enumerate(reg) if abs(int(lab)) in a.areas]
        Yk = Ym[:, keep]
        names = [f"{area_name(reg[k])}#{k}" for k in keep]
        model = ME.fit(d, Yk, folds=folds, grid=tuple(p["alpha_grid"]))
        parts_here = {k: v for k, v in parts.items() if np.isin(d.group, v).any()}
        vp = ME.variance_partition(d, Yk, folds, model.alphas, out_names=names, partitions=parts_here)
        vp["area"] = vp.output.str.split("#").str[0]
        from scripts.session_poses import session_dir
        out = session_dir(rv, spec)[2]
        vp.to_csv(out / "movement_encoding_vp.csv", index=False)
        s = vp.groupby(["area", "group"]).agg(r2_full=("r2_full", "median"), unique=("unique", "median"),
                                              alone=("r2_alone", "median")).reset_index()
        s.insert(0, "session", label)
        summary.append(s)
        print(f"\n== {label}: {info['n_trials']} trials, {mask.sum()} imaging frames ({mask.sum() / len(ft):.0%}), "
              f"{info['n_licks']} DLC licks ({info['n_dir_licks']} with direction), {info['n_contacts']} contacts; "
              f"alphas {model.alphas}")
        print(s.pivot(index="area", columns="group", values="unique").round(4).to_string())
        print("   median CV R^2 (full):", s.groupby("area").r2_full.first().round(3).to_dict())
        # kernels for one area (first MOs component)
        ks = ME.kernels(model)
        fig, axs = plt.subplots(1, 4, figsize=(16, 3.4))
        k0 = next((i for i, nm in enumerate(names) if nm.startswith("MOs")), 0)
        for ax, rname in zip(axs, ["tongue_onset", "contact", "tongue_onset_x_deviation", "tongue_onset_x_angle"]):
            if rname in ks:
                lg, w = ks[rname]
                ax.plot(lg, w[:, k0], "-", lw=2)
                ax.axvline(0, color="k", lw=0.5)
                ax.axhline(0, color="0.6", lw=0.5)
            ax.set_title(f"{rname}\n({names[k0]})", fontsize=8)
            ax.set_xlabel("lag from event (s)")
        fig.suptitle(f"{label}: kernels (standardised units), one MOs component", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "movement_encoding_kernels.png", dpi=110)
        plt.close(fig)
    S = pd.concat(summary, ignore_index=True)
    out = Path(rv.root("microscope")) / "DeepLabCut" / "Widefield" / "session_poses" / "movement_encoding_summary.csv"
    S.to_csv(out, index=False)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
