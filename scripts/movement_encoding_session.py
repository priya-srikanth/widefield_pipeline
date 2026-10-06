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


def lick_events(P: dict, licks: str = "onset_contact") -> dict:
    """The lick event regressors. ``onset_contact`` (pre-registered): every DLC tongue onset + every DAQ contact
    (two kernels ~1 frame apart that trade weight). ``split`` (Priya, 2026-10-06): DISJOINT kernels for licks that
    touched the spout (DAQ contact within `tongue_kinematics` contact.match_ms of the lick's peak) and licks that
    did not -- no lick is in both, so they cannot trade weight, and no-contact licks (more frequent post-stroke)
    get their own kernel rather than being a remainder -- plus the DAQ contacts no DLC lick accounts for."""
    L = P["licks"]
    if licks == "onset_contact":
        return {"contact": P["contact_s"], "tongue_onset": L.on_s.to_numpy()}
    if licks == "split":
        c = L.contact.astype(bool).to_numpy()
        return {"lick_contact": L.on_s.to_numpy()[c], "lick_nocontact": L.on_s.to_numpy()[~c],
                "contact_unmatched": P["contact_unmatched_s"]}
    raise ValueError(f"licks must be onset_contact or split, not {licks!r}")


def video_signals(animal: str, date: str, rv, frame_times_s, cams, k: int = 30) -> dict:
    """{name: (t_s, values)} motion-energy components per camera (`scripts.video_motion_session` output, top ``k``)
    on the imaging frames, BLANKED (NaN = contributes nothing) outside position strobe .. trial end: the spout is
    repositioned between trial end and the next strobe, and spout motion is the TARGET (Priya 2026-10-06: "masking
    spout repositioning sounds good. using position-strobe to trial-end will eliminate this problem"); a stationary
    spout makes no motion energy, so close-spout pixels stay in on far-spout trials."""
    from scripts.session_poses import session_dir
    from wfield_local import trial_windows as TW
    ft = np.asarray(frame_times_s, float)
    b = TW.trial_bounds(animal, date, rv)
    start = np.where(np.isfinite(b.strobe_s), b.strobe_s, b.cue_s - 0.5).astype(float)
    ok = np.isfinite(start) & np.isfinite(b.stop_s.to_numpy(float))
    keep = np.zeros(len(ft), bool)
    for a0, a1 in zip(start[ok], b.stop_s.to_numpy(float)[ok]):
        keep |= (ft >= a0) & (ft <= a1)
    out = {}
    d = session_dir(rv, f"{animal}:{date}:full")[2]
    for cam in cams:
        z = np.load(d / f"video_motion_{cam}.npz")
        if not np.allclose(z["frame_times_s"][:5], ft[:5]):
            raise ValueError(f"{cam}: motion energy was binned on a different imaging clock")
        tc = np.array(z["timecourses"][:, :k], float)
        tc[~keep] = np.nan
        for j in range(tc.shape[1]):
            out[f"video_{cam}_pc{j}"] = (ft, tc[:, j])
    return out


def running_signal(animal: str, date: str, rv) -> dict:
    """{"running": (t_s, treadmill speed mm/s)} -- `behavior_events.session_speed` (configs segmentation.treadmill),
    the same speed the running bouts use."""
    from wfield_local.behavior_events import session_speed
    h5 = sorted((Path(rv.root("daq_recorder_output")) / date).glob(f"{animal}_{date}_*.h5"))[0]
    return {"running": session_speed(h5)}


def pose_inputs(animal: str, date: str, rv, frame_times_s, min_reach_px: float, spec: str | None = None,
                licks: str = "onset_contact", basis_spacing_s: float | None = None, extra_state: dict | None = None):
    """MovementInputs + a mask of imaging frames inside the pose windows (``spec``: see `session_pieces`; ``licks``:
    see `lick_events`; ``basis_spacing_s``: raised-cosine event kernels, None = per-frame FIR)."""
    P = session_pieces(animal, date, rv, frame_times_s, min_reach_px, spec=spec)
    L = P["licks"]
    inputs = MI.build_inputs(frame_times_s, cues=P["trials"][["cue_s", "pos_name"]],
                             events=lick_events(P, licks),
                             overrides={"kernel_basis_spacing_s": basis_spacing_s,
                                        "groups": {n: ("state" if n == "running" else "video") for n in (extra_state or {})}},
                             state_signals=extra_state,
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
    from scripts.session_poses import assert_no_double_events, read_index, session_dir
    from wfield_local import lick_reference as LR
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    from wfield_local import tongue_detect as td
    from wfield_local import contact_classes as CC
    from wfield_local import trial_windows as TW
    animal, date, d = session_dir(rv, spec or f"{animal}:{date}")
    idx = read_index(d)
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
    assert_no_double_events(on_s, pl.trial_id.to_numpy(), what="DLC lick onsets")
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
    # Each DAQ contact classified (`contact_classes`, Priya 2026-10-06): inside a kept lick's span = lick; else
    # no_tongue / tongue_other; long touches without the tongue = grooming (widened periods)
    cue_l = pl.trial_id.map(cue_of).to_numpy(float)
    tongue_out_f = np.asarray(sig["tongue_protrusion"], float) > float(sig["lip_px"]) + float(CC.params()["tongue_out_px"])
    try:
        dur = CC.session_durations(animal, date, contact_s, rv)
    except (IndexError, OSError, KeyError):
        dur = np.full(len(contact_s), np.nan)
    pk_l = cue_l + pl.t_ms.to_numpy(float) / 1000.0         # lick window as `tongue_kinematics` contact "span"
    lo_l = np.fmin(cue_l + pl.rise_start_ms.to_numpy(float) / 1000.0, pk_l - 0.060)
    hi_l = np.fmax(cue_l + pl.fall_end_ms.to_numpy(float) / 1000.0, pk_l + 0.120)
    ctab = CC.classify(contact_s, dur, lo_l, hi_l, vt, tongue_out_f)
    groom = CC.grooming_periods(ctab)
    contact_unmatched_s = ctab.t_s[ctab["class"] != "lick"].to_numpy()
    info = {"n_licks": len(pl), "n_dir_licks": int(np.isfinite(dev).sum()), "n_contacts": len(contact_s),
            "n_trials": len(trials), "n_licks_contact": int(pl.contact.astype(bool).sum()),
            "n_contacts_unmatched": len(contact_unmatched_s),
            "contact_classes": ctab["class"].value_counts().to_dict(), "n_grooming_periods": len(groom)}
    licks = pd.DataFrame({"on_s": on_s, "trial_id": pl.trial_id.to_numpy(), "position": pl.position.to_numpy(),
                          "cue_s": pl.trial_id.map(cue_of).to_numpy(float),
                          "peak_s": pl.trial_id.map(cue_of).to_numpy(float) + pl.t_ms.to_numpy(float) / 1000.0,
                          "contact": pl.contact.to_numpy(), "reach_ok": reach_ok,
                          "angle": LR.peak_angle(pl).to_numpy(float), "dev": pl.dev_session_deg.to_numpy(float),
                          "angle_dir": ang, "dev_dir": dev})
    licks["grooming"] = CC.in_spans(licks.on_s.to_numpy(), np.array([g[0] for g in groom]),
                                    np.array([g[1] for g in groom])) if groom else False
    return {"trials": trials, "contact_s": contact_s, "contact_unmatched_s": contact_unmatched_s, "licks": licks,
            "contacts": ctab, "grooming_periods": groom,
            "sig": sig, "vt": vt, "mask": mask,
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
    ap.add_argument("--kernels", choices=["fir", "smooth"], default="fir",
                    help="fir = one weight per frame lag (pre-registered); smooth = raised-cosine bumps")
    ap.add_argument("--basis-spacing", type=float, default=0.15, help="smooth: bump spacing (s)")
    ap.add_argument("--video", nargs="*", default=[], metavar="CAM",
                    help="add motion-energy components of these cameras (blanked outside strobe .. trial end)")
    ap.add_argument("--video-k", type=int, default=30, help="components per camera")
    ap.add_argument("--running", action="store_true", help="add DAQ treadmill speed (group state)")
    ap.add_argument("--licks", choices=["onset_contact", "split"], default="onset_contact",
                    help="onset_contact = tongue onset + DAQ contact (pre-registered); split = contact / no-contact")
    a = ap.parse_args(argv)
    rv = PathResolver()
    p = MI.params()
    spacing = a.basis_spacing if a.kernels == "smooth" else None
    tag = (("" if a.kernels == "fir" else "_smooth") + ("" if a.licks == "onset_contact" else "_splitlicks")
           + (f"_video{''.join(c[-1] for c in a.video)}" if a.video else "") + ("_run" if a.running else ""))
    parts = (config.defaults().get("movement_encoding", {}) or {}).get("partitions") or {
        "task": ["task"], "movement": ["lick_events", "tongue", "jaw"], "direction": ["direction"]}
    if a.video or a.running:
        mov = list(parts.get("movement", ["lick_events", "tongue", "jaw"]))
        parts = {**parts, "movement_dlc": mov, **({"video": ["video"]} if a.video else {}),
                 **({"running": ["state"]} if a.running else {}),
                 "movement_all": mov + (["video"] if a.video else []) + (["state"] if a.running else [])}
    summary = []
    for spec in a.sessions:
        animal, date = spec.split(":")[:2]
        label = f"{animal}_{date[4:]}"
        Y, reg, ft = imaging(label)
        label += "".join(f"_{t}" for t in spec.split(":")[2:])   # the pose source, e.g. PS93_0814_full
        extra = {**(video_signals(animal, date, rv, ft, a.video, a.video_k) if a.video else {}),
                 **(running_signal(animal, date, rv) if a.running else {})}
        inputs, mask, info = pose_inputs(animal, date, rv, ft, float(p["min_reach_px"]), spec=spec, licks=a.licks,
                                         basis_spacing_s=spacing, extra_state=extra or None)
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
        vp.to_csv(out / f"movement_encoding_vp{tag}.csv", index=False)
        s = vp.groupby(["area", "group"]).agg(r2_full=("r2_full", "median"), unique=("unique", "median"),
                                              alone=("r2_alone", "median")).reset_index()
        s.insert(0, "session", label + tag)
        summary.append(s)
        print(f"\n== {label}: {info['n_trials']} trials, {mask.sum()} imaging frames ({mask.sum() / len(ft):.0%}), "
              f"{info['n_licks']} DLC licks ({info['n_licks_contact']} with contact, {info['n_dir_licks']} with "
              f"direction), {info['n_contacts']} contacts ({info['n_contacts_unmatched']} unmatched); "
              f"kernels {a.kernels}, licks {a.licks}; "
              f"alphas {model.alphas}")
        print(s.pivot(index="area", columns="group", values="unique").round(4).to_string())
        print("   median CV R^2 (full):", s.groupby("area").r2_full.first().round(3).to_dict())
        # kernels: per area (mean over its components), in dF/F per event (modulators: per unit of the modulator)
        ks = ME.kernels(model, per_event=True)
        rnames = [r for r in ks if d.group[d.regressor == r][0] in ("lick_events", "direction")]
        areas = sorted({n.split("#")[0] for n in names})
        fig, axs = plt.subplots(1, len(rnames), figsize=(3.6 * len(rnames), 3.4), squeeze=False)
        for ax, rname in zip(axs[0], rnames):
            lg, w = ks[rname]
            for ar in areas:
                m = np.array([n.split("#")[0] == ar for n in names])
                ax.plot(lg, w[:, m].mean(1), lw=1.6, label=ar)
            ax.axvline(0, color="k", lw=0.5)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.set_title(rname, fontsize=8)
            ax.set_xlabel("lag from event (s)")
        axs[0, 0].set_ylabel("dF/F per event")
        axs[0, 0].legend(fontsize=6)
        fig.suptitle(f"{label}: event kernels, area means ({a.kernels}"
                     f"{f', {spacing:g} s bumps' if spacing else ', one weight per 32 ms frame'}; licks {a.licks})",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"movement_encoding_kernels{tag}.png", dpi=110)
        plt.close(fig)
    S = pd.concat(summary, ignore_index=True)
    out = Path(rv.root("microscope")) / "DeepLabCut" / "Widefield" / "session_poses" / f"movement_encoding_summary{tag}.csv"
    S.to_csv(out, index=False)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
