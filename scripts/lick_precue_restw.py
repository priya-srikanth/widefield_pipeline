"""PRE-CUE LEVEL per area against a rest reference -- the extra row for the cued area maps (Priya 10-28, approved).

DEFAULT SINCE 11-09 (Priya approved 11-09): THE LICK-FREE ENL WINDOW, RESTnl PRIMARY + SESSION-MEAN TWIN.
  window  per trial the shared lick-free ENL window: [cue - W, cue) if it holds no lick, else the last W s of the latest
          lick-free gap in [strobe + 0.3 s, cue), and no earlier than -2.0 s (the trace grid) -- `enl_window.sliding_lags`
          (the families' wrapper of `lp_licks.enl_window_starts_s` = the ENL decoders' `precue_window_start`; the 0.3 s
          spout-arrival settle is its guard). W = `lp_licks.enl_post_s()` (1.0 s under WIDEFIELD_LICK_SOURCE=lp, the
          production source; 2.0 s under DAQ). Licks = `lp_licks.gate_licks_s` (the edited LP train under LP, cues
          without an LP verdict dropped; DAQ contact onsets otherwise). Trials with NO lick-free window are DROPPED.
          level = mean of the trace over that window (`enl_window.window_mean` on the 0.05 s grid, bounds inclusive);
          the secondary value (precue05) = the LAST 0.5 s of the same window (= -0.5..0 s on an unslid trial).
  trials  the camera WORKING trials (WIDEFIELD_TRIAL_GATE=camera: class contact + incomplete of
          `trial_classes.cue_classes_for` over the session's full DAQ cue train, `in_sel("working")` -- inside the
          working period, stopped runs and quit tail removed; jaw-only / no response / unknown out), as the camera
          `working` rows of the ENL decoders (`trial_classes.camera_select`). Without the camera gate: the behavior
          table's engaged column.
  ref     --rest-variant restnl (DEFAULT): minus the session's RESTnl level (`scripts/restnl.py`, rule wr: restdock05
          docked, not-running rest minus every DAQ / cam4 lick [0.5 s, 2.0 s], camera working period, per-position flat
          median, equal-weight mean over positions with >= 50 frames, >= 4 of 6). --rest-variant session: NO rest
          subtraction (the dF/F is already referenced to the session mean) -- the session-mean twin. --rest-variant
          restw: minus the old RESTW level (read from the precue_restw cache; robustness only).
  cache   <OUT_W>/precue_enl_<lick source><W>s/<label>.npz (one per session serves all three references); outputs
          precue_enl_*[_<spout>]<sfx>.csv, figures/lick_cued_precue_enl_maps[_farcontra]<sfx>.png, sfx = _restnl |
          _sessmean | _restw. Old caches / outputs are never read or written by this path.
THE FIXED-WINDOW VERSION BELOW (10-28 .. 11-09) stays reachable with --precue-window fixed (old caches, old names,
old default reference restw).

WHY A REST REFERENCE. The spout is in place ~2-3 s before the cue, so every in-trial window (incl. the -1.0..-0.5 s cue
baseline of the cued maps) already contains the pre-cue state; an in-trial baseline cannot measure it. The pre-cue level
is therefore referenced to the session's REST baseline instead.

DEFINITION (per session x area):
  trace   = exactly the area traces of `lick_cued_ramp.wide_session` / `iti_licks.extract` (ROI v8 masks, area means of
            U_atlas on the adopted hemo-corrected SVT, slow drift basis removed, x1e3 -- NO per-trial baseline), with the
            ALM / tjM1 / MOs_rest / MOp_rest split of the cued maps (`iti_licks.with_rest`, applied to the whole-session
            time series BEFORE the baseline, since the median is not linear).
  restw   = `rest_by_position.restw_from_frames` on that whole-session area time series: per position the MEDIAN over
            its rest frames, positions averaged with equal weight (all 6 required, >= 200 frames each). Rest frames =
            `rest_frames_by_position(session, T, docked=False, engaged_only=True)` -- the same call as the encoder's
            `_restw_baseline` and `position_reference_maps.session_restw_svt` (both docked=False): the rest MASK is
            already the configured `restdock05` variant (docked spout, lick buffer [0.5, 1.0] s), so docked=False adds
            no second docked intersection; engagement-gated (quit-period rest excluded, docs/REST_ENGAGEMENT_AUDIT.md).
            A session that cannot form restw gets the unweighted rest median, FLAGGED (`restw_ok` False) and left OUT of
            every statistic and map.
  precue  = mean over ENGAGED trials (combined_encoding behavior table, engaged & finite cue) of the trial's mean trace
            over -1.0..0 s before the cue (0.05 s grid, as LAGS_W), minus restw. Secondary window -0.5..0 s (precue05).
            Also on the first-lick trials only (the cued maps' trial set; `precue_fl`) for the correlation check.
  per target spout: the same restricted to engaged trials at that target (>= 5 trials; all spouts >= 10).
Stats: change from pre per area / epoch with `analysis_kit.boot_delta` (nested animals > sessions), significance = CI
excludes 0, no multiple-comparison correction -- as `iti_area_maps.changes` (the cued maps). AUGUST-ONLY pre baseline
(session list = `iti_licks.labels()`, the August+ curated frozen-run set). Image LEFT = ipsilesional.
Outputs (under iti_licks.OUT_W): precue_restw/<label>.npz (per-session cache), precue_restw_sessions[_<spout>].csv,
precue_restw_changes[_<spout>].csv, precue_restw_session_notes.csv, figures/lick_cued_precue_restw_maps[_farcontra].png
(+ _levels.png), figdata/<stem>.pkl.
RESTnl VARIANT (Priya 10-29, `--rest-variant restnl`; default restw = everything above, unchanged): the baseline is the
LICK-FREE rest level of `scripts/restnl.py` (restw_strict_check strict_w50; sessions without it left out), the trial set
is the CURRENT engaged gate (the behavior table now; the precue_restw cache holds an older gate in 86 / 107 sessions).
Cache <OUT_W>/precue_restnl/<label>.npz (field `restw` = the RESTnl level there, `restw_old` = RESTW for comparison);
every output file name gets the suffix _restnl.
Usage: python scripts/lick_precue_restw.py [--jobs N] [--plot-only] [--rest-variant restnl|session|restw]
       python scripts/lick_precue_restw.py --precue-window fixed [--rest-variant restnl]     (the old version)
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import restnl  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

from wfield_local import mask_keys  # noqa: E402

CACHE = "precue_restw"
VERSION = 2
LAGS = np.round(np.arange(-2.0, 3.0001, 0.05), 3)          # = lick_cued_ramp.LAGS_W (validation against cued_wide)
WIN = (-1.0, 0.0)
WIN2 = (-0.5, 0.0)
VALS = (("precue", "PRE-CUE LEVEL -1.0..0 s above restw"), ("precue05", "pre-cue level -0.5..0 s above restw"))
KEY = ["ALM", "tjM1", "SSp-m", "SSp-n"]
#: trial sets of the maps: target spout (None = all spouts) -> engaged trials a session needs (`session_tables`)
SPOUT_MIN = {None: 10, "far_contra": 5}


def session_worker(label: str) -> dict:
    """One session (module-level for spawn) -> <OUT_W>/precue_restw/<label>.npz; returns the notes."""
    return _session(label, "restw")


def session_worker_restnl(label: str) -> dict:
    """RESTnl variant (module-level for spawn) -> <OUT_W>/precue_restnl/<label>.npz (current engaged gate)."""
    return _session(label, "restnl")


def _cache(var: str) -> str:
    return CACHE if var == "restw" else restnl.cache("precue_restnl")   # strict rule = precue_restnl, wr = _wr


def _session(label: str, var: str) -> dict:
    f = il.OUT_W / _cache(var) / f"{label}.npz"
    if f.exists():
        z = np.load(f)
        if int(z["version"]) == VERSION:
            mask_keys.check_obj(z, f)             # stale-read guard: built on the current analysis masks (11-09)
            return {"label": label, "cached": True}
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.rest_by_position import rest_frames_by_position, restw_from_frames
    d = il._load(label)
    # == lick_cued_ramp.wide_session's trace construction (validated below against the stored cued_wide cue traces)
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    if cols != list(d["cols"]):
        raise ValueError(f"{label}: ROI columns differ from the session file")
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    Rs, cs = il.with_rest(R, cols, label, basis="split")    # (T, cols + 4 rest areas)
    extra = {}
    if var == "restnl":                                        # the lick-free level (restw_strict_check, strict_w50)
        base, ok = restnl.level(label, cs)
        old = np.load(il.OUT_W / CACHE / f"{label}.npz")
        mask_keys.check_obj(old, il.OUT_W / CACHE / f"{label}.npz")
        if [str(c) for c in old["cols"]] != cs:
            raise ValueError(f"{label}: columns differ from the precue_restw cache")
        used, n_rest = old["used"], int(old["n_rest"])
        extra = {"restw_old": np.asarray(old["restw"], float), "restw_old_ok": bool(old["restw_ok"]),
                 "n_eng_old": len(old["pos"])}
    else:
        # restw on the whole-session area series
        s = next(x for x in config.load_sessions() if x["label"] == label)
        per, info = rest_frames_by_position(s, Rs.shape[0], docked=False, engaged_only=True)
        if info.get("error"):
            raise RuntimeError(f"{label}: no rest mask ({info['error']})")
        base, used = restw_from_frames(Rs.T, per, label=label)
        ok = base is not None
        n_rest = int(sum(len(v) for v in per.values()))
        if not ok:                                             # loud fallback: unweighted rest median, flagged
            idx = np.concatenate([np.asarray(v) for v in per.values()]) if per else np.array([], int)
            idx = idx[idx < Rs.shape[0]]
            base = np.median(Rs[idx], 0)[:, None] if idx.size else np.full((Rs.shape[1], 1), np.nan)
        base = base[:, 0]
    # engaged trials
    B = il._behaviour(label)
    T = B.trials
    Eg = T[T.engaged.astype(bool) & np.isfinite(T.cue_s.to_numpy(float))]
    cue = Eg.cue_s.to_numpy(float)

    def win_means(t):
        X = il._sample(Rs, ft, t, LAGS).astype(np.float64)
        out = (t[:, None] + LAGS[None, :] < ft[0]) | (t[:, None] + LAGS[None, :] > ft[-1])
        X[out] = np.nan
        w1 = (LAGS >= WIN[0]) & (LAGS <= WIN[1])
        w2 = (LAGS >= WIN2[0]) & (LAGS <= WIN2[1])
        return np.nanmean(X[:, w1], 1), np.nanmean(X[:, w2], 1), X

    p10, p05, _ = win_means(cue)
    # the cued maps' trial set (first-lick trials) + validation against cued_wide
    cc, _fl = il._first_lick_times(label, d["first_rt"])
    cc = cc[np.isfinite(cc)]
    f10, f05, Xf = win_means(cc)
    maxdiff = np.nan
    wf = il.OUT / "cued_wide" / f"{label}.npz"
    if wf.exists():
        z = np.load(wf)
        ref = z["cue"].astype(float)
        if ref.shape[0] == Xf.shape[0]:
            maxdiff = float(np.nanmax(np.abs(Xf[..., :len(cols)] - ref)))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, version=VERSION, cols=np.array(cs), restw=base, restw_ok=ok, used=np.array(used, int),
                        n_rest=n_rest, p10=p10, p05=p05, pos=np.asarray(Eg.pos.astype(str).to_numpy(), str), f10=f10, f05=f05,
                        maxdiff=maxdiff, animal=str(d["animal"]), epoch=str(d["epoch"]), **extra, **mask_keys.stamp())
    return {"label": label, "cached": False, "restw_ok": ok, "n_pos": len(used), "maxdiff": maxdiff}


def session_tables(labels: list, var: str = "restw") -> tuple[dict, pd.DataFrame]:
    """{spout or None: per-session x area table}, notes."""
    out, notes = {None: [], "far_contra": []}, []
    for x in labels:
        z = np.load(il.OUT_W / _cache(var) / f"{x}.npz")
        mask_keys.check_obj(z, il.OUT_W / _cache(var) / f"{x}.npz")
        cs = [str(c) for c in z["cols"]]
        ok = bool(z["restw_ok"])
        pos = z["pos"].astype(str)
        notes.append({"label": x, "animal": str(z["animal"]), "epoch": str(z["epoch"]), "restw_ok": ok,
                      "positions_used": " ".join(map(str, z["used"])), "n_rest_frames": int(z["n_rest"]),
                      "n_engaged": len(pos), "n_firstlick": len(z["f10"]), "maxdiff_vs_cued_wide": float(z["maxdiff"])})
        for sp, mn in SPOUT_MIN.items():
            m = np.ones(len(pos), bool) if sp is None else pos == sp
            if m.sum() < mn:
                continue
            a10 = np.nanmean(z["p10"][m], 0) - z["restw"]
            a05 = np.nanmean(z["p05"][m], 0) - z["restw"]
            fl = np.nanmean(z["f10"], 0) - z["restw"] if sp is None and len(z["f10"]) else np.full(len(cs), np.nan)
            for j, c in enumerate(cs):
                if c == "cortex" or c.rsplit("_", 1)[0] in ("MOs", "MOp"):
                    continue
                out[sp].append({"label": x, "animal": str(z["animal"]), "epoch": str(z["epoch"]), "area": c,
                                "precue": a10[j], "precue05": a05[j], "precue_fl": fl[j], "restw": z["restw"][j],
                                "restw_ok": ok, "n": int(m.sum())})
    return {k: pd.DataFrame(v) for k, v in out.items()}, pd.DataFrame(notes)


def trial_set_sentence(spout: str | None = None, var: str | None = None, window: str | None = None) -> str:
    """THE trial + frame set of a figure (Priya 11-09) under the active (or given) pre-cue ``window`` and reference
    ``var``: `enl_trial_set_sentence` for the lick-free ENL window (default), `fixed_trial_set_sentence` for the old
    fixed window."""
    window = window or window_mode()
    if window == "enl":
        return enl_trial_set_sentence(spout, var or reference("enl"))
    return fixed_trial_set_sentence(spout, var or reference("fixed"))


def fixed_trial_set_sentence(spout: str | None = None, var: str = "restw") -> str:
    """THE trial + frame set of a FIXED-window figure (--precue-window fixed): the engaged cued trials (all spouts, or
    one target spout) whose pre-cue window is read, and the REST frames of the session baseline it is referenced to
    (RESTW, or RESTnl = `restnl.DEF` under --rest-variant restnl)."""
    import lick_refit_decoders as rd

    from wfield_local import rest_by_position as rbp
    from wfield_local import trial_set_text as tst
    gate = rd.engaged_text() + (" (the gate stored in the precue_restw cache)" if var == "restw"
                                else " (the CURRENT gate)")
    what = "engaged cued trials with a finite cue" + ("" if spout is None else f", {spout.replace('_', '-')} target only")
    trials = tst.events(what, unit="trials", keep=[
        gate, f"sessions with >= {SPOUT_MIN[spout]} such trials",
        f"pre-cue window {WIN[0]:g}..{WIN[1]:g} s (secondary {WIN2[0]:g}..{WIN2[1]:g} s), no lick-free requirement"],
        excluded="non-engaged trials" + ("" if spout is None else ", other target spouts"))
    if var == "restnl":
        rest = tst.frames(restnl.DEF, excluded="sessions without a valid RESTnl level")
    else:
        rest = tst.frames(f"RESTW rest frames ({rd.rest_mask_text()}), terminal quit period excluded (engagement-gated)",
                          keep=["per spout position the median over its rest frames, the 6 positions averaged with "
                                f"equal weight (all 6 required, >= {rbp.MIN_FRAMES_PER_POSITION} frames each)"],
                          excluded="sessions without a valid RESTW (restw_ok False)")
    return f"{trials}  AND  {rest}"


def per_animal(T: pd.DataFrame, val: str) -> pd.DataFrame:
    """Rule 8: area x animal, session-mean per epoch and change vs pre."""
    P = T.pivot_table(index=["area", "animal"], columns="epoch", values=val, aggfunc="mean").reindex(columns=il.EP4)
    for ep in il.EP4[1:]:
        P[f"d_{ep}"] = P[ep] - P["pre"]
    return P


def run(spout: str | None, T: pd.DataFrame, sfx: str = "") -> None:
    import iti_area_maps as iam
    from frozen_area_maps import outline
    tag = "" if spout is None else "_" + spout.replace("_", "")
    T = T[T.restw_ok]
    T.to_csv(il.OUT_W / f"precue_restw_sessions{tag}{sfx}.csv", index=False)
    pa = per_animal(T, "precue")
    pa.to_csv(il.OUT_W / f"precue_restw_per_animal{tag}{sfx}.csv")
    keyrows = pa[pa.index.get_level_values("area").str.rsplit("_", n=1).str[0].isin(KEY)]
    print(f"== per animal (rule 8), PRE-CUE LEVEL -1..0 s above {'RESTnl' if sfx else 'restw'}{tag}\n"
          + keyrows.round(2).to_string())
    C = pd.concat([iam.changes(T, v) for v, _t in VALS])
    C.to_csv(il.OUT_W / f"precue_restw_changes{tag}{sfx}.csv", index=False)
    print(C[C.sig].sort_values(["value", "epoch", "delta"]).round(3).to_string(index=False))
    labels = sorted(T.label.unique())
    img, nm = iam.split_image(labels[0])
    rows = []
    for val, ttl in VALS:
        c = C[C.value == val]
        lim = np.nanpercentile(np.abs(c.delta), 98) or 1.0
        pans = []
        for ep in il.EP4[1:]:
            ce_ = c[c.epoch == ep].set_index("area")
            pans.append((ep, ce_.delta.to_dict(), list(ce_.index[ce_.sig]), int(ce_.sig.sum())))
        rows.append((ttl, lim, pans))
    what = "CUED trials, all engaged trials" + ("" if spout is None else f", {spout.replace('_', '-')} target only")
    stem = f"lick_cued_precue_restw_maps{tag}{sfx}"
    save_fd(stem, {"img": img, "nm": nm, "ol": outline(img), "rows": rows, "what": what,
                   "levels": iam._levels(T, VALS), "sfx": sfx})
    draw(stem)


def draw(stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import iti_area_maps as iam
    import matplotlib.pyplot as plt
    from frozen_area_maps import outline, paint
    F = load_fd(stem)
    sfx = F.get("sfx", "")
    from wfield_local import trial_set_text as tst
    tst.activate(None, text=fixed_trial_set_sentence("far_contra" if "_farcontra" in stem else None,
                                                     "restnl" if sfx else "restw"))   # this figure pair's set (11-09)
    rl = restnl.relabel if sfx else (lambda t: t)
    base = stem[:-len(sfx)] if sfx else stem
    nm = F["nm"]
    img = iam.trim_rest(F["img"], nm)
    ol = outline(img)
    note = ("pre-cue level = mean dF/F (x1e3) over the window before the cue (spout already in place), engaged trials, "
            "minus the session's RESTW baseline (position-weighted flat rest median, restdock05, engagement-gated)")
    (il.OUT_W / "figures").mkdir(parents=True, exist_ok=True)
    nr = len(F["rows"])
    fig, axs = plt.subplots(nr, 3, figsize=(14, 4.5 * nr), squeeze=False)
    for i, (ttl, lim, pans) in enumerate(F["rows"]):
        for j, (ep, delta, sig_areas, n_sig) in enumerate(pans):
            ax = axs[i, j]
            im = ax.imshow(paint(img, nm, delta), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            iam._sig_halo(ax, img, nm, sig_areas, list(delta))
            ax.set_title(rl(f"{ttl}\n{ep} - pre ({n_sig} areas significant)"), fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axs[i, :], fraction=0.02, pad=0.01).set_label(rl("change in dF/F above restw (x1e3)"))
    fig.suptitle(rl(f"NEURAL, NO DECODER: {F['what']} -- change from pre per area in the PRE-CUE LEVEL; {note}; ALM / tjM1 "
                 "separate from the rest of MOs / MOp (ROI v8); black outline on a white halo = nested-bootstrap CI "
                 "excludes 0. Image left = ipsilesional." + (" Trials = CURRENT engaged gate." if sfx else "")),
                 fontsize=10, wrap=True)
    save_png_svg(fig, il.OUT_W / "figures" / f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    nr = len(F["levels"])
    fig, axs = plt.subplots(nr, 4, figsize=(18, 4.5 * nr), squeeze=False)
    for i, (ttl, lim, pans) in enumerate(F["levels"]):
        for j, (ep, lev) in enumerate(pans):
            ax = axs[i, j]
            im = ax.imshow(paint(img, nm, lev), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            ax.set_title(rl(f"{ttl}\n{ep}"), fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axs[i, :], fraction=0.015, pad=0.01).set_label(rl("dF/F above restw (x1e3)"))
    fig.suptitle(rl(f"NEURAL, NO DECODER: {F['what']} -- per-epoch PRE-CUE LEVEL per area (mean over animals of each "
                 f"animal's session mean; not the change from pre); {note}; one color scale per row. Image left = "
                 "ipsilesional. Descriptive." + (" Trials = CURRENT engaged gate." if sfx else "")), fontsize=10, wrap=True)
    save_png_svg(fig, il.OUT_W / "figures" / f"{base}_levels{sfx}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def sanity(T: pd.DataFrame, notes: pd.DataFrame, sfx: str = "") -> None:
    print(f"== restw fallback sessions: {int((~notes.restw_ok).sum())}/{len(notes)} "
          f"{list(notes.label[~notes.restw_ok])}")
    print(f"== validation vs cued_wide cue traces: max |diff| = {notes.maxdiff_vs_cued_wide.max():.4f} "
          f"(NaN sessions {int(notes.maxdiff_vs_cued_wide.isna().sum())})")
    f = il.OUT / "area_auc_sessions_split_cued.csv"
    if f.exists() and not sfx:
        A = pd.read_csv(f)[["label", "area", "cue"]]
        J = T[T.restw_ok].merge(A, on=["label", "area"])
        r1 = np.corrcoef(J.precue, J.cue)[0, 1]
        r2 = np.corrcoef(J.precue_fl.astype(float), J.cue)[0, 1]
        rr = J.groupby("area").apply(lambda g: np.corrcoef(g.precue, g.cue)[0, 1], include_groups=False)
        print(f"== corr(pre-cue level, in-trial cue response AUC) over session x area: r = {r1:.3f} (first-lick trials "
              f"{r2:.3f}); within-area median r = {rr.median():.3f} [{rr.min():.3f}, {rr.max():.3f}], n = {len(J)}")
        print(f"== corr(pre-cue level all engaged vs first-lick trials): "
              f"{np.corrcoef(J.precue, J.precue_fl.astype(float))[0, 1]:.3f}")


def main_fixed() -> int:
    """The FIXED-window version (--precue-window fixed): unchanged since 11-09 apart from this entry point."""
    from wfield_local import analysis_kit as ak
    var = restnl.variant()
    sfx = restnl.SFX if var == "restnl" else ""
    from wfield_local import trial_set_text as tst
    tst.declare(text=fixed_trial_set_sentence(None, var))       # draw() narrows it per figure pair
    if plot_only():
        for st in ("lick_cued_precue_restw_maps", "lick_cued_precue_restw_maps_farcontra"):
            draw(st + sfx)
        return 0
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
    labs = il.labels()
    worker = session_worker_restnl if var == "restnl" else session_worker
    res, fails = ak.fan_sessions(labs, worker, jobs=min(jobs, 4 if var == "restnl" else 8))
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    Ts, notes = session_tables(labs, var)
    notes.to_csv(il.OUT_W / f"precue_restw_session_notes{sfx}.csv", index=False)
    sanity(Ts[None], notes, sfx)
    run(None, Ts[None], sfx)
    run("far_contra", Ts["far_contra"], sfx)
    return 0


# ==================================================================================================================
# THE LICK-FREE ENL WINDOW (default since 11-09, Priya approved): see the module docstring, first block.
# ==================================================================================================================
WINDOWS = ("enl", "fixed")
#: references of the ENL path: RESTnl (primary), the session-mean twin (no rest subtraction), the old RESTW (robustness)
REFS = ("restnl", "session", "restw")
ENL_CACHE = "precue_enl"
ENL_VERSION = 1
ENL_PREFIX = "precue_enl"
ENL_STEM = "lick_cued_precue_enl_maps"
#: the secondary value: the LAST this-many seconds of each trial's ENL window (= -0.5..0 s on an unslid trial)
LAST_S = 0.5


def _arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def window_mode() -> str:
    """`--precue-window enl|fixed` (default enl, the lick-free ENL window; fixed = the 10-28 version)."""
    w = _arg("--precue-window", "enl")
    if w not in WINDOWS:
        raise SystemExit(f"--precue-window must be one of {WINDOWS}, not {w!r}")
    return w


def reference(window: str | None = None) -> str:
    """The reference of this run: under the ENL window `--rest-variant restnl|session|restw` (default restnl); under
    the fixed window the old `restnl.variant()` (default restw, unchanged)."""
    if (window or window_mode()) == "fixed":
        return restnl.variant()
    r = _arg("--rest-variant", "restnl")
    if r not in REFS:
        raise SystemExit(f"--rest-variant must be one of {REFS} (ENL window), not {r!r}")
    return r


def ref_sfx(ref: str) -> str:
    """Output-name suffix of an ENL reference (RESTnl keeps `restnl.SFX`, i.e. _restnl / _restnl_strict per rule)."""
    return {"restnl": restnl.SFX, "session": "_sessmean", "restw": "_restw"}[ref]


def enl_win_s() -> float:
    """THE ENL window length of this run (`lp_licks.enl_post_s`: 1.0 s under LP licks, 2.0 s under DAQ)."""
    from wfield_local import lp_licks
    return float(lp_licks.enl_post_s())


def enl_cache() -> str:
    """Per-session cache dir of the ENL path: lick source + window in the name, so an LP and a DAQ run (or another
    window) can never read each other's entries."""
    from wfield_local import lp_licks
    return f"{ENL_CACHE}_{lp_licks.lick_source()}{enl_win_s():g}s"


def rest_root() -> Path:
    """The iti_licks folder of this run WITHOUT the lick-source part: where the RESTnl levels (restw_check/sessions)
    and the old precue_restw cache live. Both are lick-source free (the RESTnl mask removes every DAQ and cam4 lick
    whatever the run's source), so an LP run reads them from the DAQ-tagged folder instead of an empty LP_licks one."""
    from fr_musall import E_BASE
    from wfield_local import run_mode
    t = run_mode.tag()
    return E_BASE / "iti_licks" / t if t else E_BASE / "iti_licks"


def enl_select(cue_s, strobe_s, lick_s, has_verdict, working, win_s: float, span_lo: float = float(LAGS[0])):
    """``(keep, lo, hi)`` per trial: the trial set + lick-free ENL window of the pre-cue level. ``lo`` / ``hi`` = the
    window in s from the cue (NaN = dropped). THE SEARCH IS THE SHARED ONE: `enl_window.sliding_lags` (the decoders'
    `precue_window_start` through `lp_licks.enl_window_starts_s`, + the 0.3 s spout-arrival settle and the ``span_lo``
    trace-grid limit); nothing is re-implemented here. A trial is kept when it has an LP verdict (``has_verdict``, from
    `lp_licks.gate_licks_s`; all True under DAQ), is a WORKING trial (``working``) and has a lick-free window."""
    from wfield_local import enl_window
    lo, hi = enl_window.sliding_lags(np.asarray(cue_s, float), np.asarray(strobe_s, float),
                                     np.asarray(lick_s, float), float(win_s), end_lag=0.0, span_lo=float(span_lo))
    keep = np.asarray(has_verdict, bool) & np.asarray(working, bool) & np.isfinite(lo)
    return keep, np.where(keep, lo, np.nan), np.where(keep, hi, np.nan)


def gate_licks(label: str, cue_s) -> tuple[np.ndarray, np.ndarray]:
    """``(lick_s, has_verdict)``: `lp_licks.gate_licks_s` on these cues -- the edited LP train under LP (cues without an
    LP verdict marked False), the DAQ contact onsets (`dlc_frames.lick_onsets`) otherwise (as
    `within_spout_deviation.enl_window_starts`)."""
    from wfield_local import config, dlc_frames, lp_licks
    from wfield_local.paths import PathResolver
    daq = np.zeros(0)
    if not lp_licks.is_lp():                                   # the DAQ train is only read under the DAQ source
        rv = PathResolver()
        animal, mmdd = label.split("_")
        date = "2026" + mmdd
        sid = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob("*_trials.csv"))[-1].name[:-11]
        daq = dlc_frames.lick_onsets(animal, date, sid, rv)
    return lp_licks.gate_licks_s(label, np.asarray(cue_s, float), daq, float(config.defaults()["decode"]["max_rt_s"]))


def working_mask(label: str, T: pd.DataFrame) -> np.ndarray:
    """Per row of the behavior trial table ``T``: a camera WORKING trial (`trial_classes.cue_classes_for` over the
    session's FULL DAQ cue train, matched by cue time, `in_sel("working")`: production contact + incomplete inside the
    working period, stopped runs / quit tail / jaw-only / no response / unknown out) -- the `working` rows of
    `trial_classes.camera_select`. Without the camera gate: the table's engaged column (the DAQ gate)."""
    from wfield_local import trial_classes as tc
    if not tc.camera_gate_on():
        return T.engaged.to_numpy(bool)
    full = tc.daq_cue_seconds(label)
    cls = tc.cue_classes(pd.DataFrame({"cue_daq": full, "cls": tc.cue_classes_for(label, full)}),
                         T.cue_s.to_numpy(float))
    return tc.in_sel(cls, "working")


def _enl_meta() -> dict:
    from wfield_local import enl_window, lp_licks
    from wfield_local import trial_classes as tc
    return {"lick_source": lp_licks.lick_source(), "win_s": enl_win_s(), "settle_s": float(enl_window.SETTLE_S),
            "span_lo": float(LAGS[0]), "last_s": LAST_S,
            "trials": ("camera:" + "+".join(sorted(tc.sel_classes("working")))) if tc.camera_gate_on() else "daq_engaged"}


def area_series(label: str):
    """``(Rs, cols, ft)``: the whole-session area series of `session_worker` (ROI v8 area means of U_atlas on the
    adopted SVT, slow drift removed, x1e3, ALM / tjM1 / MOs_rest / MOp_rest split) -- the same lines, for the ENL path."""
    from wfield_local import combined_encoding as ce
    d = il._load(label)
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    if cols != list(d["cols"]):
        raise ValueError(f"{label}: ROI columns differ from the session file")
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    Rs, cs = il.with_rest(R, cols, label, basis="split")
    return Rs, cs, ft, d


def sample_cues(Rs, ft, cue) -> np.ndarray:
    """(trials x LAGS x cols) of the area series around each cue, NaN outside the recording (as `win_means`)."""
    X = il._sample(Rs, ft, cue, LAGS).astype(np.float64)
    X[(cue[:, None] + LAGS[None, :] < ft[0]) | (cue[:, None] + LAGS[None, :] > ft[-1])] = np.nan
    return X


def _rest_levels(label: str, cs: list) -> dict:
    """RESTnl (both rules) and the old RESTW per column, each with its ok flag (NaN / False when missing)."""
    out = {}
    src = rest_root() / "restw_check" / "sessions"
    for r in restnl.RULES:
        try:
            lev, ok = restnl.level(label, cs, r, src=src)
        except FileNotFoundError:
            lev, ok = np.full(len(cs), np.nan), False
        out[f"restnl_{r}"], out[f"restnl_{r}_ok"] = lev, ok
    f = rest_root() / CACHE / f"{label}.npz"
    out["restw"], out["restw_ok"] = np.full(len(cs), np.nan), False
    if f.exists():
        z = np.load(f)
        mask_keys.check_obj(z, f)                 # the old precue_restw cache is mask-dependent too (11-09)
        if [str(c) for c in z["cols"]] == list(cs):
            out["restw"], out["restw_ok"] = np.asarray(z["restw"], float), bool(z["restw_ok"])
    return out


def session_worker_enl(label: str) -> dict:
    """One session (module-level for spawn) -> <OUT_W>/<enl_cache()>/<label>.npz; returns the trial counts."""
    import json
    f = il.OUT_W / enl_cache() / f"{label}.npz"
    meta = _enl_meta()
    if f.exists():
        z = np.load(f)
        if int(z["version"]) == ENL_VERSION and json.loads(str(z["meta"])) == meta:
            mask_keys.check_obj(z, f)             # stale-read guard: built on the current analysis masks (11-09)
            return {"label": label, "cached": True}
    from wfield_local import enl_window
    Rs, cs, ft, d = area_series(label)
    B = il._behaviour(label)
    T = B.trials[np.isfinite(B.trials.cue_s.to_numpy(float))].reset_index(drop=True)
    cue, strobe = T.cue_s.to_numpy(float), T.strobe_s.to_numpy(float)
    licks, has = gate_licks(label, cue)
    work = working_mask(label, T)
    keep, lo, hi = enl_select(cue, strobe, licks, has, work, meta["win_s"])
    k = np.flatnonzero(keep)
    X = sample_cues(Rs, ft, cue[k])
    p10 = enl_window.window_mean(X, LAGS, lo[k], hi[k])
    p05 = enl_window.window_mean(X, LAGS, hi[k] - LAST_S, hi[k])
    counts = {"n_trials": len(T), "n_verdict": int(has.sum()), "n_working": int((has & work).sum()),
              "n_kept": len(k), "n_slid": int((np.abs(lo[k] + meta["win_s"]) > 1e-3).sum()),
              "n_engaged_daq": int(T.engaged.astype(bool).sum())}
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, version=ENL_VERSION, meta=json.dumps(meta), cols=np.array(cs), cue=cue[k], lo=lo[k],
                        hi=hi[k], stop=T.stop_s.to_numpy(float)[k], pos=np.asarray(T.pos.astype(str).to_numpy()[k], str),
                        p10=p10, p05=p05, animal=str(d["animal"]), epoch=str(d["epoch"]), **counts,
                        **_rest_levels(label, cs), **mask_keys.stamp())
    return {"label": label, "cached": False, **counts}


def ref_base(z, ref: str) -> tuple[np.ndarray, bool]:
    """(reference level per column, ok) of one ENL cache entry: RESTnl under the active rule, zeros for the session-
    mean twin (the dF/F is already session-mean referenced), or the old RESTW."""
    if ref == "session":
        return np.zeros(len(z["cols"])), True
    if ref == "restw":
        return np.asarray(z["restw"], float), bool(z["restw_ok"])
    k = f"restnl_{restnl.RULE}"
    return np.asarray(z[k], float), bool(z[k + "_ok"])


def session_tables_enl(labels: list, ref: str) -> tuple[dict, pd.DataFrame]:
    """{spout or None: per-session x area table (column ``ref_ok``)}, notes -- the ENL counterpart of `session_tables`."""
    out, notes = {None: [], "far_contra": []}, []
    for x in labels:
        z = np.load(il.OUT_W / enl_cache() / f"{x}.npz")
        mask_keys.check_obj(z, il.OUT_W / enl_cache() / f"{x}.npz")   # stale-read guard (11-09)
        cs = [str(c) for c in z["cols"]]
        base, ok = ref_base(z, ref)
        pos = z["pos"].astype(str)
        notes.append({"label": x, "animal": str(z["animal"]), "epoch": str(z["epoch"]), "ref_ok": ok,
                      **{c: int(z[c]) for c in ("n_trials", "n_engaged_daq", "n_verdict", "n_working", "n_kept",
                                                "n_slid")},
                      "n_far_contra": int((pos == "far_contra").sum())})
        for sp, mn in SPOUT_MIN.items():
            m = np.ones(len(pos), bool) if sp is None else pos == sp
            if m.sum() < mn:
                continue
            a10 = np.nanmean(z["p10"][m], 0) - base
            a05 = np.nanmean(z["p05"][m], 0) - base
            for j, c in enumerate(cs):
                if c == "cortex" or c.rsplit("_", 1)[0] in ("MOs", "MOp"):
                    continue
                out[sp].append({"label": x, "animal": str(z["animal"]), "epoch": str(z["epoch"]), "area": c,
                                "precue": a10[j], "precue05": a05[j], "ref": base[j], "ref_ok": ok, "n": int(m.sum())})
    return {k: pd.DataFrame(v) for k, v in out.items()}, pd.DataFrame(notes)


def ref_text(ref: str) -> str:
    """The reference in words (captions)."""
    if ref == "session":
        return ("NO rest subtraction: the dF/F is referenced to the session mean (the session-mean twin of the RESTnl "
                "figure)")
    if ref == "restw":
        return ("minus the session's RESTW baseline (position-weighted flat rest median, restdock05, engagement-gated; "
                "robustness only)")
    return f"minus the session's RESTnl level ({restnl.DEF})"


def ref_unit(ref: str) -> str:
    return {"restnl": "dF/F above RESTnl (x1e3)", "session": "dF/F, session-mean referenced (x1e3)",
            "restw": "dF/F above RESTW (x1e3)"}[ref]


def enl_window_text() -> str:
    """The ENL window rule in words (captions)."""
    from wfield_local import enl_window
    from wfield_local import trial_set_text as tst
    w = enl_win_s()
    return (f"{w:g} s lick-free ENL window ({tst.lick_text()}): [cue - {w:g} s, cue) if it holds no lick, else the last "
            f"{w:g} s of the latest lick-free gap starting >= strobe + {enl_window.SETTLE_S:g} s and >= {-LAGS[0]:g} s "
            "before the cue (enl_window.sliding_lags -> lp_licks.enl_window_starts_s); trials without one DROPPED")


def enl_trial_set_sentence(spout: str | None = None, ref: str = "restnl") -> str:
    """THE trial + frame set of an ENL-window figure: the camera working trials (`trial_set_text.sentence("working")`)
    with a lick-free ENL window, and the reference it is read against."""
    from wfield_local import lp_licks
    from wfield_local import trial_classes as tc
    from wfield_local import trial_set_text as tst
    if tc.camera_gate_on():
        head, _, exc = tst.sentence("working").partition("; excluded: ")
        what = head.split(": ", 1)[1]
    else:
        import lick_refit_decoders as rd
        what, exc = f"engaged cued trials ({rd.engaged_text()})", "non-engaged trials"
    keep = ([f"{spout.replace('_', '-')} target only"] if spout else []) + [
        enl_window_text(), "cues without an LP lick verdict dropped" if lp_licks.is_lp() else "",
        f"level = mean over the window (secondary: its last {LAST_S:g} s)",
        f"sessions with >= {SPOUT_MIN[spout]} such trials"]
    excl = [exc, "trials without a lick-free ENL window"] + (["other target spouts"] if spout else [])
    trials = tst.events(what, unit="trials", keep=keep, excluded=excl)
    if ref == "restnl":
        rest = tst.frames(restnl.DEF + " (rest frames inside the camera working period)",
                          excluded="sessions without a valid RESTnl level")
    elif ref == "restw":
        rest = tst.frames("RESTW rest frames (restdock05, docked, lick buffer [0.5, 1.0] s), engagement-gated",
                          keep=["per spout position the median over its rest frames, the 6 positions averaged with "
                                "equal weight"], excluded="sessions without a valid RESTW (restw_ok False)")
    else:
        rest = "baseline: none (dF/F referenced to the session mean; no rest subtraction)"
    return f"{trials}  AND  {rest}"


def enl_vals(ref: str) -> tuple:
    nm = {"restnl": "RESTnl", "session": "session mean", "restw": "RESTW"}[ref]
    w = enl_win_s()
    return (("precue", f"PRE-CUE LEVEL, {w:g} s lick-free ENL window, vs {nm}"),
            ("precue05", f"pre-cue level, last {LAST_S:g} s of the ENL window, vs {nm}"))


def run_enl(spout: str | None, T: pd.DataFrame, ref: str) -> None:
    """Tables + figdata + figures of one trial set (all spouts / far-contra target) on the ENL window."""
    import iti_area_maps as iam
    from frozen_area_maps import outline
    sfx = ref_sfx(ref)
    tag = "" if spout is None else "_" + spout.replace("_", "")
    T = T[T.ref_ok]
    T.to_csv(il.OUT_W / f"{ENL_PREFIX}_sessions{tag}{sfx}.csv", index=False)
    pa = per_animal(T, "precue")
    pa.to_csv(il.OUT_W / f"{ENL_PREFIX}_per_animal{tag}{sfx}.csv")
    keyrows = pa[pa.index.get_level_values("area").str.rsplit("_", n=1).str[0].isin(KEY)]
    print(f"== per animal (rule 8), PRE-CUE LEVEL (ENL {enl_win_s():g} s) vs {ref}{tag}\n" + keyrows.round(2).to_string())
    vals = enl_vals(ref)
    C = pd.concat([iam.changes(T, v) for v, _t in vals])
    C.to_csv(il.OUT_W / f"{ENL_PREFIX}_changes{tag}{sfx}.csv", index=False)
    print(C[C.sig].sort_values(["value", "epoch", "delta"]).round(3).to_string(index=False))
    img, nm = iam.split_image(sorted(T.label.unique())[0])
    rows = []
    for val, ttl in vals:
        c = C[C.value == val]
        lim = np.nanpercentile(np.abs(c.delta), 98) or 1.0
        pans = []
        for ep in il.EP4[1:]:
            ce_ = c[c.epoch == ep].set_index("area")
            pans.append((ep, ce_.delta.to_dict(), list(ce_.index[ce_.sig]), int(ce_.sig.sum())))
        rows.append((ttl, lim, pans))
    what = "camera WORKING trials, all spouts" if spout is None else \
        f"camera WORKING trials, {spout.replace('_', '-')} target only"
    stem = f"{ENL_STEM}{tag}{sfx}"
    save_fd(stem, {"img": img, "nm": nm, "ol": outline(img), "rows": rows, "what": what, "enl": True, "ref": ref,
                   "spout": spout, "levels": iam._levels(T, vals), "sfx": sfx,
                   "n_sess": T.groupby("epoch").label.nunique().to_dict()})
    draw_enl(stem)


def draw_enl(stem: str) -> None:
    """The change maps (rows = level, last-0.5 s level; columns = acute / subacute / chronic minus pre) and the per-epoch
    level maps of one ENL figdata."""
    import matplotlib
    matplotlib.use("Agg")
    import iti_area_maps as iam
    import matplotlib.pyplot as plt
    from frozen_area_maps import outline, paint
    from wfield_local import trial_set_text as tst
    F = load_fd(stem)
    ref, sfx = F["ref"], F["sfx"]
    tst.activate(None, text=enl_trial_set_sentence(F.get("spout"), ref))      # this figure pair's set (11-09)
    base = stem[:-len(sfx)] if sfx else stem
    nm = F["nm"]
    img = iam.trim_rest(F["img"], nm)
    ol = outline(img)
    ns = ", ".join(f"{e} {n}" for e, n in F.get("n_sess", {}).items())
    note = (f"pre-cue level = per trial the mean dF/F (x1e3) over its {enl_window_text()}; per session the mean over "
            f"its trials, {ref_text(ref)}")
    stats = ("change = each animal's epoch sessions vs its OWN pre sessions (analysis_kit.boot_delta, nested "
             "bootstrap animals > sessions); black outline on a white halo = 95% CI excludes 0; no multiple-comparison "
             "correction; August-only pre")
    (il.OUT_W / "figures").mkdir(parents=True, exist_ok=True)
    nr = len(F["rows"])
    fig, axs = plt.subplots(nr, 3, figsize=(14, 4.5 * nr), squeeze=False)
    for i, (ttl, lim, pans) in enumerate(F["rows"]):
        for j, (ep, delta, sig_areas, n_sig) in enumerate(pans):
            ax = axs[i, j]
            im = ax.imshow(paint(img, nm, delta), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            iam._sig_halo(ax, img, nm, sig_areas, list(delta))
            ax.set_title(f"{ttl}\n{ep} - pre ({n_sig} areas significant)", fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axs[i, :], fraction=0.02, pad=0.01).set_label(f"change in {ref_unit(ref)}")
    fig.suptitle(f"NEURAL, NO DECODER: {F['what']} -- change from pre per area in the PRE-CUE LEVEL; {note}; {stats}; "
                 f"ALM / tjM1 separate from the rest of MOs / MOp (ROI v8). Sessions: {ns}. Image left = ipsilesional.",
                 fontsize=10, wrap=True)
    save_png_svg(fig, il.OUT_W / "figures" / f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    nr = len(F["levels"])
    fig, axs = plt.subplots(nr, 4, figsize=(18, 4.5 * nr), squeeze=False)
    for i, (ttl, lim, pans) in enumerate(F["levels"]):
        for j, (ep, lev) in enumerate(pans):
            ax = axs[i, j]
            im = ax.imshow(paint(img, nm, lev), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            ax.set_title(f"{ttl}\n{ep}", fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axs[i, :], fraction=0.015, pad=0.01).set_label(ref_unit(ref))
    fig.suptitle(f"NEURAL, NO DECODER: {F['what']} -- per-epoch PRE-CUE LEVEL per area (mean over animals of each "
                 f"animal's session mean; not the change from pre); {note}; one color scale per row. Image left = "
                 "ipsilesional. Descriptive.", fontsize=10, wrap=True)
    save_png_svg(fig, il.OUT_W / "figures" / f"{base}_levels{sfx}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if window_mode() == "fixed":
        return main_fixed()
    from wfield_local import analysis_kit as ak
    from wfield_local import trial_set_text as tst
    ref = reference("enl")
    sfx = ref_sfx(ref)
    tst.declare(text=enl_trial_set_sentence(None, ref))         # draw_enl() narrows it per figure pair
    if plot_only():
        for st in (ENL_STEM, ENL_STEM + "_farcontra"):
            draw_enl(st + sfx)
        return 0
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    labs = il.labels()
    _res, fails = ak.fan_sessions(labs, session_worker_enl, jobs=min(jobs, 6))
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    Ts, notes = session_tables_enl(labs, ref)
    notes.to_csv(il.OUT_W / f"{ENL_PREFIX}_session_notes{sfx}.csv", index=False)
    print(f"== ENL {enl_win_s():g} s ({_enl_meta()['lick_source']} licks), reference {ref}: sessions without a valid "
          f"reference {int((~notes.ref_ok).sum())}/{len(notes)} {list(notes.label[~notes.ref_ok])}; trials kept "
          f"{int(notes.n_kept.sum())} of {int(notes.n_working.sum())} working ({int(notes.n_slid.sum())} slid)")
    run_enl(None, Ts[None], ref)
    run_enl("far_contra", Ts["far_contra"], ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
